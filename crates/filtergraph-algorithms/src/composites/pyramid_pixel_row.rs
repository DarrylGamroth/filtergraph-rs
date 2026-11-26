//! Row-block pyramid pupil-pixel normalization.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;
use core::fmt;
use core::ops::Range;

use num_traits::{Float, NumCast};

use crate::{
    AlgorithmProcessResult, PYRAMID_PUPIL_COUNT, PyramidPixelError, PyramidPixelImagePlan,
    PyramidPixelImagePreparationError, PyramidPixelWorkspace,
};

use super::row_block::{RowBlockDropReason, RowBlockFrame, RowBlockInfo, RowBlockResult};

/// Invalid row-block pyramid composition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPixelRowPlanError {
    /// A row block must be smaller than the detector height.
    BlockRows,
    /// The fixed row quantum must divide the detector height.
    BlockRowsDoNotDivideImage,
    /// Row blocks require compact row-major detector-image storage.
    ImageLayout,
    /// Extent arithmetic overflowed.
    ExtentOverflow,
}

impl fmt::Display for PyramidPixelRowPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlockRows => formatter
                .write_str("row-block height must be positive and smaller than image height"),
            Self::BlockRowsDoNotDivideImage => {
                formatter.write_str("row-block height must divide image height")
            }
            Self::ImageLayout => formatter
                .write_str("row-block pyramid processing requires compact row-major image storage"),
            Self::ExtentOverflow => formatter.write_str("row-block extent overflow"),
        }
    }
}

impl core::error::Error for PyramidPixelRowPlanError {}

/// Invalid preparation or sparse replacement for row-block pyramid processing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPixelRowPreparationError {
    /// Image-backed pyramid preparation failed.
    Image(PyramidPixelImagePreparationError),
    /// The prepared image operation cannot consume the configured row quantum.
    Composition(PyramidPixelRowPlanError),
}

impl fmt::Display for PyramidPixelRowPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(error) => error.fmt(formatter),
            Self::Composition(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PyramidPixelRowPreparationError {}

impl From<PyramidPixelImagePreparationError> for PyramidPixelRowPreparationError {
    fn from(error: PyramidPixelImagePreparationError) -> Self {
        Self::Image(error)
    }
}

impl From<PyramidPixelRowPlanError> for PyramidPixelRowPreparationError {
    fn from(error: PyramidPixelRowPlanError) -> Self {
        Self::Composition(error)
    }
}

/// Invalid caller storage or numerical row-block pyramid result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPixelRowError {
    /// One input block has the wrong packed pixel extent.
    InputExtent {
        /// Required pixel count.
        expected: usize,
        /// Supplied pixel count.
        actual: usize,
    },
    /// The reconstruction-pixel output has the wrong extent.
    OutputExtent {
        /// Required selected-pixel count across the four pupils.
        expected: usize,
        /// Supplied selected-pixel count.
        actual: usize,
    },
    /// The pyramid operation rejected a pixel or normalization result.
    Pyramid(PyramidPixelError),
}

impl fmt::Display for PyramidPixelRowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent { expected, actual } => write!(
                formatter,
                "row-block pixel extent {actual} does not match {expected}"
            ),
            Self::OutputExtent { expected, actual } => write!(
                formatter,
                "pyramid reconstruction-pixel extent {actual} does not match {expected}"
            ),
            Self::Pyramid(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PyramidPixelRowError {}

impl From<PyramidPixelError> for PyramidPixelRowError {
    fn from(error: PyramidPixelError) -> Self {
        Self::Pyramid(error)
    }
}

/// Failure from scientific row processing or a deployment-owned pixel consumer.
///
/// This type belongs to the hidden native-operation hook. Ordinary scientific
/// callers continue to use [`PyramidPixelRowError`].
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPixelRowExecutionError<E> {
    /// The typed scientific state machine rejected the block or result.
    Algorithm(PyramidPixelRowError),
    /// A deployment-owned consumer of newly ready pixels failed synchronously.
    Executor(E),
}

/// Invalid metadata or numerical result at the portable declaration boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPixelRowProcessError {
    /// A row-block call requires transport-neutral sample identity.
    MissingSampleMetadata,
    /// The portable row offset cannot be represented by this target.
    OffsetOverflow,
    /// The mean-pupil-intensity output is not one scalar.
    MeanOutputExtent,
    /// The prepared row operation rejected the numerical call.
    Algorithm(PyramidPixelRowError),
}

impl fmt::Display for PyramidPixelRowProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSampleMetadata => {
                formatter.write_str("row-block sample metadata is required")
            }
            Self::OffsetOverflow => formatter.write_str("row-block offset does not fit usize"),
            Self::MeanOutputExtent => {
                formatter.write_str("mean-pupil-intensity output must contain one value")
            }
            Self::Algorithm(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PyramidPixelRowProcessError {}

impl From<PyramidPixelRowError> for PyramidPixelRowProcessError {
    fn from(error: PyramidPixelRowError) -> Self {
        Self::Algorithm(error)
    }
}

/// Prepared row-block pyramid pupil-pixel normalization.
///
/// Complete rows within each pupil are normalized in detector-arrival order
/// while pupil-major output coordinates and the full-frame mean retain
/// declaration order. This lets every available pupil row advance before
/// terminal detector-frame completion even when pupil origins are not ordered
/// by detector row.
#[derive(Clone, Debug)]
pub struct PyramidPixelRowPlan<T> {
    image: PyramidPixelImagePlan<T>,
    block_rows: usize,
    block_elements: usize,
}

impl<T> PyramidPixelRowPlan<T>
where
    T: Float + NumCast,
{
    /// Compose image-backed pyramid processing with one fixed row quantum.
    pub fn new(
        image: PyramidPixelImagePlan<T>,
        block_rows: usize,
    ) -> Result<Self, PyramidPixelRowPlanError> {
        let regions = image.pupil_regions();
        let (image_rows, image_columns) = regions.image_size();
        if block_rows == 0 || block_rows >= image_rows {
            return Err(PyramidPixelRowPlanError::BlockRows);
        }
        if image_rows % block_rows != 0 {
            return Err(PyramidPixelRowPlanError::BlockRowsDoNotDivideImage);
        }
        if regions.row_stride() != image_columns || regions.first_pixel() != 0 {
            return Err(PyramidPixelRowPlanError::ImageLayout);
        }
        let block_elements = block_rows
            .checked_mul(image_columns)
            .ok_or(PyramidPixelRowPlanError::ExtentOverflow)?;
        Ok(Self {
            image,
            block_rows,
            block_elements,
        })
    }

    /// Return the prepared image-backed pyramid operation.
    pub const fn image(&self) -> &PyramidPixelImagePlan<T> {
        &self.image
    }

    /// Return the fixed detector rows in one input block.
    pub const fn block_rows(&self) -> usize {
        self.block_rows
    }

    pub(crate) const fn block_elements(&self) -> usize {
        self.block_elements
    }

    /// Replace four zero-based pupil origins without changing port formats.
    pub fn with_u32_pupil_origins(
        &self,
        values: &[u32],
    ) -> Result<Self, PyramidPixelRowPreparationError> {
        Ok(Self::new(
            self.image.with_u32_pupil_origins(values)?,
            self.block_rows,
        )?)
    }

    /// Replace the shared pupil mask without changing the output extent.
    pub fn with_pupil_mask(
        &self,
        values: &[bool],
    ) -> Result<Self, PyramidPixelRowPreparationError> {
        Ok(Self::new(
            self.image.with_pupil_mask(values)?,
            self.block_rows,
        )?)
    }

    /// Allocate bounded mutable state for one execution owner.
    pub fn workspace(&self) -> PyramidPixelRowWorkspace<T> {
        let (image_rows, image_columns) = self.image.pupil_regions().image_size();
        PyramidPixelRowWorkspace {
            image: vec![T::zero(); image_rows * image_columns].into_boxed_slice(),
            reconstruction_pixels: vec![T::zero(); self.image.normalization().output_pixel_count()]
                .into_boxed_slice(),
            normalization: self.image.normalization().workspace(),
            frame: RowBlockFrame::new(),
            processed_rows: [0; PYRAMID_PUPIL_COUNT],
            processed_selected: [0; PYRAMID_PUPIL_COUNT],
            processed_pupil_count: 0,
            summed_pupil: 0,
            summed_rows: 0,
            sum: T::zero(),
        }
    }

    fn process_ready_rows<E, F>(
        &self,
        workspace: &mut PyramidPixelRowWorkspace<T>,
        completed_rows: usize,
        consume: &mut F,
    ) -> Result<(), PyramidPixelRowExecutionError<E>>
    where
        F: FnMut(&Self, &[T], &[Range<usize>]) -> Result<(), E>,
    {
        let regions = self.image.pupil_regions();
        let normalization = self.image.normalization();
        let (pupil_rows, pupil_columns) = regions.region_size();
        let image_columns = regions.image_size().1;
        let selected_per_pupil = normalization.selected_pixel_count();
        let gain = workspace.normalization.reciprocal_previous_frame_mean();
        let mut ranges: [Range<usize>; PYRAMID_PUPIL_COUNT] = core::array::from_fn(|_| 0..0);
        let mut range_count = 0;

        for pupil in 0..PYRAMID_PUPIL_COUNT {
            let (first_row, first_column) = regions.origins()[pupil];
            let available_rows = completed_rows.saturating_sub(first_row).min(pupil_rows);
            let first_pupil_row = workspace.processed_rows[pupil];
            if available_rows <= first_pupil_row {
                continue;
            }
            let first_output = pupil * selected_per_pupil + workspace.processed_selected[pupil];
            let mut output_index = first_output;
            for pupil_row in first_pupil_row..available_rows {
                let start = (first_row + pupil_row) * image_columns + first_column;
                for (pupil_column, &input) in workspace.image[start..start + pupil_columns]
                    .iter()
                    .enumerate()
                {
                    let pixel = pupil_row * pupil_columns + pupil_column;
                    if !input.is_finite() {
                        return Err(PyramidPixelRowExecutionError::Algorithm(
                            PyramidPixelError::NonFiniteInput { pupil, pixel }.into(),
                        ));
                    }
                    if normalization.mask()[pixel] {
                        let value = gain * input;
                        if !value.is_finite() {
                            return Err(PyramidPixelRowExecutionError::Algorithm(
                                PyramidPixelError::NonFiniteOutput {
                                    index: output_index,
                                }
                                .into(),
                            ));
                        }
                        workspace.reconstruction_pixels[output_index] = value;
                        output_index += 1;
                    }
                }
            }
            workspace.processed_rows[pupil] = available_rows;
            workspace.processed_selected[pupil] = output_index - pupil * selected_per_pupil;
            if available_rows == pupil_rows {
                debug_assert_eq!(workspace.processed_selected[pupil], selected_per_pupil);
                workspace.processed_pupil_count += 1;
            }
            if first_output < output_index {
                ranges[range_count] = first_output..output_index;
                range_count += 1;
            }
        }

        // Preserve the complete-frame scientific reduction exactly. A pupil
        // row that arrives early may be normalized and reconstructed
        // immediately, but it joins the mean only after every preceding
        // declared coordinate is available. Each detector pixel is added once
        // and in pupil-major, row-major order, matching PyramidPixelPlan.
        while workspace.summed_pupil < PYRAMID_PUPIL_COUNT {
            let pupil = workspace.summed_pupil;
            let (first_row, first_column) = regions.origins()[pupil];
            while workspace.summed_rows < workspace.processed_rows[pupil] {
                let pupil_row = workspace.summed_rows;
                let start = (first_row + pupil_row) * image_columns + first_column;
                for &input in &workspace.image[start..start + pupil_columns] {
                    workspace.sum = workspace.sum + input;
                    if !workspace.sum.is_finite() {
                        return Err(PyramidPixelRowExecutionError::Algorithm(
                            PyramidPixelError::NonFiniteMean.into(),
                        ));
                    }
                }
                workspace.summed_rows += 1;
            }
            if workspace.summed_rows < pupil_rows {
                break;
            }
            workspace.summed_pupil += 1;
            workspace.summed_rows = 0;
        }
        if range_count > 0
            && let Err(error) = consume(
                self,
                &workspace.reconstruction_pixels,
                &ranges[..range_count],
            )
        {
            return Err(PyramidPixelRowExecutionError::Executor(error));
        }
        Ok(())
    }

    /// Consume one complete immutable detector-row block.
    ///
    /// Caller-owned outputs change only for [`RowBlockResult::Complete`].
    pub fn process_block(
        &self,
        reconstruction_pixels: &mut [T],
        mean_pupil_intensity: &mut T,
        workspace: &mut PyramidPixelRowWorkspace<T>,
        block: &[T],
        info: RowBlockInfo,
    ) -> Result<RowBlockResult, PyramidPixelRowError> {
        match self.process_block_with_ready_pixels(
            reconstruction_pixels,
            mean_pupil_intensity,
            workspace,
            block,
            info,
            |_plan, _pixels, _ranges| Ok::<(), Infallible>(()),
        ) {
            Ok(result) => Ok(result),
            Err(PyramidPixelRowExecutionError::Algorithm(error)) => Err(error),
            Err(PyramidPixelRowExecutionError::Executor(error)) => match error {},
        }
    }

    /// Consume one row block while exposing newly ready normalized pixels.
    ///
    /// The typed state machine retains row identity, detector-image assembly,
    /// HEART-compatible normalization history, finite-result validation,
    /// conditional publication, and recovery. The supplied closure receives
    /// the private full reconstruction-pixel vector and the bounded list of
    /// newly ready pupil-major column ranges. It is called at most once for one
    /// detector block, must return synchronously, and may not retain either
    /// borrow.
    #[doc(hidden)]
    pub fn process_block_with_ready_pixels<E, F>(
        &self,
        reconstruction_pixels: &mut [T],
        mean_pupil_intensity: &mut T,
        workspace: &mut PyramidPixelRowWorkspace<T>,
        block: &[T],
        info: RowBlockInfo,
        mut consume: F,
    ) -> Result<RowBlockResult, PyramidPixelRowExecutionError<E>>
    where
        F: FnMut(&Self, &[T], &[Range<usize>]) -> Result<(), E>,
    {
        let output_extent = self.image.normalization().output_pixel_count();
        if reconstruction_pixels.len() != output_extent {
            return Err(PyramidPixelRowExecutionError::Algorithm(
                PyramidPixelRowError::OutputExtent {
                    expected: output_extent,
                    actual: reconstruction_pixels.len(),
                },
            ));
        }
        if block.len() != self.block_elements {
            return Err(PyramidPixelRowExecutionError::Algorithm(
                PyramidPixelRowError::InputExtent {
                    expected: self.block_elements,
                    actual: block.len(),
                },
            ));
        }

        let image_rows = self.image.pupil_regions().image_size().0;
        let accepted = match workspace.frame.accept(info, self.block_rows, image_rows) {
            Ok(accepted) => accepted,
            Err(reason) => {
                workspace.clear_partial();
                return Ok(RowBlockResult::Dropped(reason));
            }
        };
        if info.first_row == 0 {
            workspace.clear_partial();
        }
        let image_columns = self.image.pupil_regions().image_size().1;
        let start = info.first_row * image_columns;
        workspace.image[start..start + self.block_elements].copy_from_slice(block);
        match self.process_ready_rows(workspace, accepted.end_row, &mut consume) {
            Ok(()) => {}
            Err(error) => {
                workspace.abandon();
                return Err(error);
            }
        }
        if !info.terminal {
            return Ok(RowBlockResult::Pending);
        }
        if workspace.processed_pupil_count != PYRAMID_PUPIL_COUNT
            || workspace.summed_pupil != PYRAMID_PUPIL_COUNT
        {
            workspace.abandon();
            return Ok(RowBlockResult::Dropped(RowBlockDropReason::RowRange));
        }
        let (mean, gain) = match self
            .image
            .normalization()
            .frame_mean_and_gain(workspace.sum)
        {
            Ok(values) => values,
            Err(error) => {
                workspace.abandon();
                return Err(PyramidPixelRowExecutionError::Algorithm(error.into()));
            }
        };
        reconstruction_pixels.copy_from_slice(&workspace.reconstruction_pixels);
        *mean_pupil_intensity = mean;
        workspace.normalization.commit_frame_gain(gain);
        let result = RowBlockResult::Complete {
            sequence: info.sequence,
            discontinuity: workspace.frame.discontinuity(),
        };
        workspace.complete();
        Ok(result)
    }
}

/// Bounded mutable state for one row-block pyramid execution owner.
#[derive(Debug)]
pub struct PyramidPixelRowWorkspace<T> {
    image: Box<[T]>,
    reconstruction_pixels: Box<[T]>,
    normalization: PyramidPixelWorkspace<T>,
    frame: RowBlockFrame,
    processed_rows: [usize; PYRAMID_PUPIL_COUNT],
    processed_selected: [usize; PYRAMID_PUPIL_COUNT],
    processed_pupil_count: usize,
    summed_pupil: usize,
    summed_rows: usize,
    sum: T,
}

impl<T> PyramidPixelRowWorkspace<T>
where
    T: Float,
{
    pub(crate) fn frame_discontinuity(&self) -> bool {
        self.frame.discontinuity()
    }

    /// Return the number of complete pupils processed so far.
    pub const fn processed_pupils(&self) -> usize {
        self.processed_pupil_count
    }

    /// Return the active detector-frame sequence, if any.
    pub const fn sequence(&self) -> Option<u64> {
        self.frame.sequence()
    }

    /// Return the normalization gain committed by the previous complete frame.
    pub fn reciprocal_previous_frame_mean(&self) -> T {
        self.normalization.reciprocal_previous_frame_mean()
    }

    /// Reset frame assembly and HEART-compatible normalization history.
    pub fn reset(&mut self) {
        self.frame = RowBlockFrame::new();
        self.normalization = PyramidPixelWorkspace::new();
        self.clear_partial();
    }

    /// Abandon a partial frame while preserving committed normalization history.
    pub fn plan_adopted(&mut self) {
        self.abandon();
    }

    fn clear_partial(&mut self) {
        self.processed_rows.fill(0);
        self.processed_selected.fill(0);
        self.processed_pupil_count = 0;
        self.summed_pupil = 0;
        self.summed_rows = 0;
        self.sum = T::zero();
    }

    fn abandon(&mut self) {
        self.frame.abandon();
        self.clear_partial();
    }

    fn complete(&mut self) {
        self.frame.complete();
        self.clear_partial();
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 row-block pyramid pupil-pixel declaration.
    pub PyramidPixelRowF32 {
        label = "pyramid-pixel-row-f32";
        configuration PyramidPixelRowF32Configuration {
            image_size: (usize, usize) = GraphRebuild;
            row_block_rows: usize = GraphRebuild;
            pupil_size: (usize, usize) = GraphRebuild;
            pupil_origins: Vec<[usize; 2]> = GraphRebuild;
            pupil_mask: Option<Vec<bool>> = Construction;
        }
        plan = PyramidPixelRowPlan<f32>;
        workspace = PyramidPixelRowWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            image_rows = plan.image().pupil_regions().image_size().0;
            image_columns = plan.image().pupil_regions().image_size().1;
            row_block_rows = plan.block_rows();
            pupil_rows = plan.image().pupil_regions().region_size().0;
            pupil_columns = plan.image().pupil_regions().region_size().1;
            selected_pixel_count = plan.image().normalization().selected_pixel_count();
            block_rate_multiplier = u32::try_from(image_rows / row_block_rows)
                .map_err(|_| crate::DeclarationPreparationError::Format(crate::FormatError::Rate))?;
        }
        inputs {
            row_block: f32 [row_block_rows, image_columns] = "row-block",
                schema = crate::schemas::CALIBRATED_PIXEL_ROW_BLOCK_V1,
                sample = row_sample,
                rate = [block_rate_multiplier, 1];
        }
        outputs {
            reconstruction_pixels: f32 [PYRAMID_PUPIL_COUNT, selected_pixel_count]
                = "reconstruction-pixels",
                schema = crate::schemas::PYRAMID_RECONSTRUCTION_PIXELS_V1,
                metadata_from = "row-block",
                publication = conditional;
            mean_pupil_intensity: f32 [1] = "mean-pupil-intensity",
                schema = crate::schemas::PYRAMID_MEAN_PUPIL_INTENSITY_V1,
                metadata_from = "row-block",
                publication = conditional;
        }
        parameters {
            pupil_origins: u32 [PYRAMID_PUPIL_COUNT, 2] = "pupil-origins",
                schema = crate::schemas::REGION_ORIGINS_V1,
                replace |plan, values| {
                    plan.with_u32_pupil_origins(values)
                };
            pupil_mask: bool [pupil_rows, pupil_columns] = "pupil-mask",
                schema = crate::schemas::PYRAMID_PUPIL_MASK_V1,
                replace |plan, values| {
                    plan.with_pupil_mask(values)
                };
        }
        parameter_error = PyramidPixelRowPreparationError;
        prepare |configuration| {
            let origins = configuration
                .pupil_origins
                .iter()
                .map(|origin| (origin[0], origin[1]))
                .collect::<Vec<_>>();
            let image = PyramidPixelImagePlan::from_detector_pupils(
                configuration.image_size,
                configuration.pupil_size,
                &origins,
                super::pyramid_measurement_block::configured_pyramid_mask(configuration.pupil_mask, configuration.pupil_size),
            )?;
            let plan = PyramidPixelRowPlan::new(image, configuration.row_block_rows)?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = PyramidPixelRowPreparationError;
        process |plan, workspace| {
            let Some(sample) = row_sample else {
                workspace.plan_adopted();
                return Err(PyramidPixelRowProcessError::MissingSampleMetadata);
            };
            let first_row = usize::try_from(sample.offset())
                .map_err(|_| PyramidPixelRowProcessError::OffsetOverflow)?;
            let [mean] = mean_pupil_intensity else {
                return Err(PyramidPixelRowProcessError::MeanOutputExtent);
            };
            let result = plan.process_block(
                reconstruction_pixels,
                mean,
                workspace,
                row_block,
                RowBlockInfo {
                    sequence: sample.sequence(),
                    first_row,
                    terminal: sample.is_terminal(),
                    discontinuity: sample.is_discontinuous(),
                    corrupted: sample.is_corrupted(),
                },
            )?;
            Ok(match result {
                RowBlockResult::Pending | RowBlockResult::Dropped(_) => {
                    AlgorithmProcessResult::deferred()
                }
                RowBlockResult::Complete {
                    sequence: _,
                    discontinuity,
                } => AlgorithmProcessResult::complete_with_metadata(
                    sample.with_offset(0)
                        .with_terminal(true)
                        .with_discontinuity(discontinuity)
                        .with_corrupted(false),
                ),
            })
        }
        process_error = PyramidPixelRowProcessError;
        reset |_plan, workspace| {
            workspace.reset();
        }
        plan_adopted |_plan, workspace| {
            workspace.plan_adopted();
        }
    }
}
