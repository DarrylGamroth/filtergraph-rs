//! Row-block pyramid normalization and incremental reconstruction.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::ops::Range;

use crate::{
    AlgorithmProcessResult, MatrixElement, PYRAMID_PUPIL_COUNT, PreparedGemv, PreparedGemvError,
    PyramidPixelImagePlan, PyramidPixelImagePreparationError, PyramidPixelRowError,
    PyramidPixelRowExecutionError, PyramidPixelRowPlan, PyramidPixelRowPlanError,
    PyramidPixelRowPreparationError, PyramidPixelRowWorkspace, multiply_add,
};

use super::row_block::{RowBlockInfo, RowBlockResult};

/// Invalid scientific preparation or sparse replacement for progressive PWFS reconstruction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidRowReconstructionPreparationError {
    /// Row-block pupil normalization preparation failed.
    Pixels(PyramidPixelRowPreparationError),
    /// Dense reconstructor preparation failed.
    Reconstructor(PreparedGemvError),
    /// The prepared normalization and reconstruction operations cannot be composed.
    Composition(PyramidRowReconstructionPlanError),
}

impl fmt::Display for PyramidRowReconstructionPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pixels(error) => error.fmt(formatter),
            Self::Reconstructor(error) => error.fmt(formatter),
            Self::Composition(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PyramidRowReconstructionPreparationError {}

impl From<PyramidPixelRowPreparationError> for PyramidRowReconstructionPreparationError {
    fn from(error: PyramidPixelRowPreparationError) -> Self {
        Self::Pixels(error)
    }
}

impl From<PyramidPixelImagePreparationError> for PyramidRowReconstructionPreparationError {
    fn from(error: PyramidPixelImagePreparationError) -> Self {
        Self::Pixels(error.into())
    }
}

impl From<PyramidPixelRowPlanError> for PyramidRowReconstructionPreparationError {
    fn from(error: PyramidPixelRowPlanError) -> Self {
        Self::Pixels(error.into())
    }
}

impl From<PreparedGemvError> for PyramidRowReconstructionPreparationError {
    fn from(error: PreparedGemvError) -> Self {
        Self::Reconstructor(error)
    }
}

impl From<PyramidRowReconstructionPlanError> for PyramidRowReconstructionPreparationError {
    fn from(error: PyramidRowReconstructionPlanError) -> Self {
        Self::Composition(error)
    }
}

/// Invalid prepared progressive PWFS composition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidRowReconstructionPlanError {
    /// The reconstructor column count does not equal the selected four-pupil pixel count.
    ReconstructorColumns {
        /// Required normalized-pixel count.
        expected: usize,
        /// Prepared reconstructor column count.
        actual: usize,
    },
    /// Extent arithmetic overflowed.
    ExtentOverflow,
}

impl fmt::Display for PyramidRowReconstructionPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReconstructorColumns { expected, actual } => write!(
                formatter,
                "reconstructor column count {actual} does not match reconstruction-pixel count {expected}"
            ),
            Self::ExtentOverflow => {
                formatter.write_str("progressive reconstructor extent overflow")
            }
        }
    }
}

impl core::error::Error for PyramidRowReconstructionPlanError {}

/// Invalid caller storage or numerical progressive PWFS result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidRowReconstructionError {
    /// Row-block pupil normalization failed.
    Pixels(PyramidPixelRowError),
    /// Caller-owned reconstructed output has the wrong extent.
    ReconstructedOutputExtent {
        /// Required reconstructed coordinate count.
        expected: usize,
        /// Supplied reconstructed coordinate count.
        actual: usize,
    },
    /// Incremental reconstruction produced a non-finite value.
    NonFiniteReconstruction,
    /// A reconstruction partition did not cover the prepared pixel extent.
    PartitionPixelExtent {
        /// Required normalized-pixel count.
        expected: usize,
        /// Supplied normalized-pixel count.
        actual: usize,
    },
    /// A reconstruction partition selected an invalid pixel-column range.
    PartitionColumnRange,
    /// A reconstruction partition selected an invalid output-coordinate range.
    PartitionOutputRange,
}

impl fmt::Display for PyramidRowReconstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pixels(error) => error.fmt(formatter),
            Self::ReconstructedOutputExtent { expected, actual } => write!(
                formatter,
                "reconstructed output extent {actual} does not match {expected}"
            ),
            Self::NonFiniteReconstruction => {
                formatter.write_str("incremental reconstruction produced a non-finite value")
            }
            Self::PartitionPixelExtent { expected, actual } => write!(
                formatter,
                "partition pixel extent {actual} does not match {expected}"
            ),
            Self::PartitionColumnRange => {
                formatter.write_str("invalid reconstruction partition column range")
            }
            Self::PartitionOutputRange => {
                formatter.write_str("invalid reconstruction partition output range")
            }
        }
    }
}

impl core::error::Error for PyramidRowReconstructionError {}

impl From<PyramidPixelRowError> for PyramidRowReconstructionError {
    fn from(error: PyramidPixelRowError) -> Self {
        Self::Pixels(error)
    }
}

/// Failure from scientific row processing or its deployment-owned executor.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidRowReconstructionExecutionError<E> {
    /// The typed scientific state machine rejected the block or result.
    Algorithm(PyramidRowReconstructionError),
    /// A deployment-owned reconstruction executor failed synchronously.
    Executor(E),
}

/// Invalid metadata or numerical result at the portable declaration boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidRowReconstructionProcessError {
    /// A row-block call requires transport-neutral sample identity.
    MissingSampleMetadata,
    /// The portable row offset cannot be represented by this target.
    OffsetOverflow,
    /// The mean-pupil-intensity output is not one scalar.
    MeanOutputExtent,
    /// The prepared row composition rejected the numerical call.
    Algorithm(PyramidRowReconstructionError),
}

impl fmt::Display for PyramidRowReconstructionProcessError {
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

impl core::error::Error for PyramidRowReconstructionProcessError {}

impl From<PyramidRowReconstructionError> for PyramidRowReconstructionProcessError {
    fn from(error: PyramidRowReconstructionError) -> Self {
        Self::Algorithm(error)
    }
}

/// Caller-owned outputs for one progressive pyramid frame.
#[derive(Debug)]
pub struct PyramidRowReconstructionOutput<'a, T> {
    reconstruction_pixels: &'a mut [T],
    mean_pupil_intensity: &'a mut T,
    reconstructed: &'a mut [T],
}

impl<'a, T> PyramidRowReconstructionOutput<'a, T> {
    /// Group normalized pixels, current-frame mean, and reconstructed coordinates.
    pub fn new(
        reconstruction_pixels: &'a mut [T],
        mean_pupil_intensity: &'a mut T,
        reconstructed: &'a mut [T],
    ) -> Self {
        Self {
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
        }
    }
}

/// Prepared row-block pyramid normalization and incremental reconstruction.
///
/// Complete pupil rows advance in detector-arrival order. Their pupil-major
/// normalized pixels are accumulated into a private reconstructed vector
/// immediately, while all caller-owned outputs remain unchanged until the
/// terminal detector block.
#[derive(Clone, Debug)]
pub struct PyramidRowReconstructionPlan<T> {
    pixels: PyramidPixelRowPlan<T>,
    reconstructor: PreparedGemv<T>,
    reconstructor_by_pixel: Arc<[T]>,
}

impl<T> PyramidRowReconstructionPlan<T>
where
    T: MatrixElement,
{
    /// Compose row-block pupil normalization and dense reconstruction.
    pub fn new(
        pixels: PyramidPixelRowPlan<T>,
        reconstructor: PreparedGemv<T>,
    ) -> Result<Self, PyramidRowReconstructionPlanError> {
        let pixel_count = pixels.image().normalization().output_pixel_count();
        if reconstructor.columns() != pixel_count {
            return Err(PyramidRowReconstructionPlanError::ReconstructorColumns {
                expected: pixel_count,
                actual: reconstructor.columns(),
            });
        }
        let rows = reconstructor.rows();
        let columns = reconstructor.columns();
        let extent = rows
            .checked_mul(columns)
            .ok_or(PyramidRowReconstructionPlanError::ExtentOverflow)?;
        let mut reconstructor_by_pixel = vec![T::zero(); extent];
        for row in 0..rows {
            for column in 0..columns {
                reconstructor_by_pixel[column * rows + row] =
                    reconstructor.matrix()[row * columns + column];
            }
        }
        Ok(Self {
            pixels,
            reconstructor,
            reconstructor_by_pixel: Arc::from(reconstructor_by_pixel),
        })
    }

    /// Return the prepared row-block pupil normalization.
    pub const fn pixels(&self) -> &PyramidPixelRowPlan<T> {
        &self.pixels
    }

    /// Return the prepared dense reconstructor.
    pub const fn reconstructor(&self) -> &PreparedGemv<T> {
        &self.reconstructor
    }

    /// Replace four zero-based pupil origins without changing port formats.
    pub fn with_u32_pupil_origins(
        &self,
        values: &[u32],
    ) -> Result<Self, PyramidRowReconstructionPreparationError> {
        Ok(Self::new(
            self.pixels.with_u32_pupil_origins(values)?,
            self.reconstructor.clone(),
        )?)
    }

    /// Replace the shared pupil mask without changing the normalized-pixel extent.
    pub fn with_pupil_mask(
        &self,
        values: &[bool],
    ) -> Result<Self, PyramidRowReconstructionPreparationError> {
        Ok(Self::new(
            self.pixels.with_pupil_mask(values)?,
            self.reconstructor.clone(),
        )?)
    }

    /// Replace the complete row-major reconstruction matrix.
    pub fn with_reconstructor(
        &self,
        values: &[T],
    ) -> Result<Self, PyramidRowReconstructionPreparationError> {
        let reconstructor = PreparedGemv::new(
            self.reconstructor.rows(),
            self.reconstructor.columns(),
            Arc::from(values),
        )?;
        Ok(Self::new(self.pixels.clone(), reconstructor)?)
    }

    /// Allocate bounded mutable state for one execution owner.
    pub fn workspace(&self) -> PyramidRowReconstructionWorkspace<T> {
        PyramidRowReconstructionWorkspace {
            pixels: self.pixels.workspace(),
            reconstructed: vec![T::zero(); self.reconstructor.rows()].into_boxed_slice(),
        }
    }

    /// Accumulate one ready normalized-pixel range into one disjoint output shard.
    #[doc(hidden)]
    pub fn accumulate_reconstruction_partition(
        &self,
        pixels: &[T],
        output: &mut [T],
        columns: Range<usize>,
        first_output: usize,
    ) -> Result<(), PyramidRowReconstructionError> {
        let expected_pixels = self.reconstructor.columns();
        if pixels.len() != expected_pixels {
            return Err(PyramidRowReconstructionError::PartitionPixelExtent {
                expected: expected_pixels,
                actual: pixels.len(),
            });
        }
        if columns.start > columns.end || columns.end > expected_pixels {
            return Err(PyramidRowReconstructionError::PartitionColumnRange);
        }
        let last_output = first_output
            .checked_add(output.len())
            .filter(|&last| last <= self.reconstructor.rows())
            .ok_or(PyramidRowReconstructionError::PartitionOutputRange)?;
        let rows = self.reconstructor.rows();
        for column in columns {
            let coefficient_start = column
                .checked_mul(rows)
                .and_then(|start| start.checked_add(first_output))
                .ok_or(PyramidRowReconstructionError::PartitionOutputRange)?;
            let coefficient_end = coefficient_start
                .checked_add(last_output - first_output)
                .ok_or(PyramidRowReconstructionError::PartitionOutputRange)?;
            let coefficients = self
                .reconstructor_by_pixel
                .get(coefficient_start..coefficient_end)
                .ok_or(PyramidRowReconstructionError::PartitionOutputRange)?;
            multiply_add(pixels[column], coefficients, output);
        }
        Ok(())
    }

    /// Consume one complete immutable detector-row block.
    ///
    /// Caller-owned outputs change only for [`RowBlockResult::Complete`].
    pub fn process_block(
        &self,
        output: PyramidRowReconstructionOutput<'_, T>,
        workspace: &mut PyramidRowReconstructionWorkspace<T>,
        block: &[T],
        info: RowBlockInfo,
    ) -> Result<RowBlockResult, PyramidRowReconstructionError> {
        match self.process_block_with_reconstruction(
            output,
            workspace,
            block,
            info,
            |plan, pixels, reconstructed, ranges| {
                for columns in ranges {
                    plan.accumulate_reconstruction_partition(
                        pixels,
                        reconstructed,
                        columns.clone(),
                        0,
                    )?;
                }
                Ok(())
            },
        ) {
            Ok(result) => Ok(result),
            Err(
                PyramidRowReconstructionExecutionError::Algorithm(error)
                | PyramidRowReconstructionExecutionError::Executor(error),
            ) => Err(error),
        }
    }

    /// Consume one row block while delegating only reconstruction accumulation.
    #[doc(hidden)]
    pub fn process_block_with_reconstruction<E, F>(
        &self,
        output: PyramidRowReconstructionOutput<'_, T>,
        workspace: &mut PyramidRowReconstructionWorkspace<T>,
        block: &[T],
        info: RowBlockInfo,
        mut reconstruct: F,
    ) -> Result<RowBlockResult, PyramidRowReconstructionExecutionError<E>>
    where
        F: FnMut(&Self, &[T], &mut [T], &[Range<usize>]) -> Result<(), E>,
    {
        if output.reconstructed.len() != self.reconstructor.rows() {
            return Err(PyramidRowReconstructionExecutionError::Algorithm(
                PyramidRowReconstructionError::ReconstructedOutputExtent {
                    expected: self.reconstructor.rows(),
                    actual: output.reconstructed.len(),
                },
            ));
        }
        let expected_pixels = self.reconstructor.columns();
        if output.reconstruction_pixels.len() != expected_pixels {
            return Err(PyramidRowReconstructionExecutionError::Algorithm(
                PyramidPixelRowError::OutputExtent {
                    expected: expected_pixels,
                    actual: output.reconstruction_pixels.len(),
                }
                .into(),
            ));
        }
        let expected_input = self.pixels.block_elements();
        if block.len() != expected_input {
            return Err(PyramidRowReconstructionExecutionError::Algorithm(
                PyramidPixelRowError::InputExtent {
                    expected: expected_input,
                    actual: block.len(),
                }
                .into(),
            ));
        }
        if info.first_row == 0 {
            workspace.reconstructed.fill(T::zero());
        }

        let PyramidRowReconstructionOutput {
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
        } = output;
        let PyramidRowReconstructionWorkspace {
            pixels: pixel_workspace,
            reconstructed: private_reconstructed,
        } = workspace;
        let result = self.pixels.process_block_with_ready_pixels(
            reconstruction_pixels,
            mean_pupil_intensity,
            pixel_workspace,
            block,
            info,
            |_pixels_plan, pixels, ranges| {
                reconstruct(self, pixels, private_reconstructed, ranges)
                    .map_err(PyramidRowReconstructionExecutionError::Executor)?;
                if private_reconstructed.iter().any(|value| !value.is_finite()) {
                    return Err(PyramidRowReconstructionExecutionError::Algorithm(
                        PyramidRowReconstructionError::NonFiniteReconstruction,
                    ));
                }
                Ok(())
            },
        );
        let result = match result {
            Ok(result) => result,
            Err(PyramidPixelRowExecutionError::Algorithm(error)) => {
                private_reconstructed.fill(T::zero());
                return Err(PyramidRowReconstructionExecutionError::Algorithm(
                    error.into(),
                ));
            }
            Err(PyramidPixelRowExecutionError::Executor(error)) => {
                private_reconstructed.fill(T::zero());
                return Err(error);
            }
        };
        if matches!(result, RowBlockResult::Dropped(_)) {
            private_reconstructed.fill(T::zero());
        }
        if matches!(result, RowBlockResult::Complete { .. }) {
            reconstructed.copy_from_slice(private_reconstructed);
        }
        Ok(result)
    }
}

/// Bounded mutable state for one progressive pyramid execution owner.
#[derive(Debug)]
pub struct PyramidRowReconstructionWorkspace<T> {
    pixels: PyramidPixelRowWorkspace<T>,
    reconstructed: Box<[T]>,
}

impl<T> PyramidRowReconstructionWorkspace<T>
where
    T: MatrixElement,
{
    /// Return the number of pupil images normalized in the active frame.
    pub const fn processed_pupils(&self) -> usize {
        self.pixels.processed_pupils()
    }

    /// Return the active detector-frame sequence, if any.
    pub const fn sequence(&self) -> Option<u64> {
        self.pixels.sequence()
    }

    /// Return the normalization gain committed by the previous complete frame.
    pub fn reciprocal_previous_frame_mean(&self) -> T {
        self.pixels.reciprocal_previous_frame_mean()
    }

    /// Reset frame assembly and HEART-compatible normalization history.
    pub fn reset(&mut self) {
        self.pixels.reset();
        self.reconstructed.fill(T::zero());
    }

    /// Abandon a partial frame while preserving committed normalization history.
    pub fn plan_adopted(&mut self) {
        self.pixels.plan_adopted();
        self.reconstructed.fill(T::zero());
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 row-block pyramid normalization and reconstruction declaration.
    pub PyramidRowReconstructorF32 {
        label = "pwfs-row-reconstructor-f32";
        configuration PyramidRowReconstructorF32Configuration {
            image_size: (usize, usize) = GraphRebuild;
            row_block_rows: usize = GraphRebuild;
            pupil_size: (usize, usize) = GraphRebuild;
            pupil_origins: Vec<[usize; 2]> = GraphRebuild;
            pupil_mask: Option<Vec<bool>> = Construction;
            reconstructed_count: usize = GraphRebuild;
            initial_reconstructor: Option<Vec<f32>> = Construction;
            reconstructed_schema: String = GraphRebuild;
        }
        plan = PyramidRowReconstructionPlan<f32>;
        operation = PyramidRowReconstructionF32;
        workspace = PyramidRowReconstructionWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            image_rows = plan.pixels().image().pupil_regions().image_size().0;
            image_columns = plan.pixels().image().pupil_regions().image_size().1;
            row_block_rows = plan.pixels().block_rows();
            pupil_rows = plan.pixels().image().pupil_regions().region_size().0;
            pupil_columns = plan.pixels().image().pupil_regions().region_size().1;
            selected_pixel_count = plan
                .pixels()
                .image()
                .normalization()
                .selected_pixel_count();
            reconstruction_pixel_count = plan.reconstructor().columns();
            reconstructed_count = plan.reconstructor().rows();
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
            reconstructed: f32 [reconstructed_count] = "reconstructed",
                schema = configuration.reconstructed_schema.as_str(),
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
            reconstructor: f32 [reconstructed_count, reconstruction_pixel_count]
                = "reconstructor",
                schema = crate::schemas::PWFS_RECONSTRUCTOR_V1,
                replace |plan, values| {
                    plan.with_reconstructor(values)
                };
        }
        parameter_error = PyramidRowReconstructionPreparationError;
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
            let pixels = PyramidPixelRowPlan::new(image, configuration.row_block_rows)?;
            let reconstruction_pixel_count = pixels.image().normalization().output_pixel_count();
            let reconstructor_elements = configuration
                .reconstructed_count
                .checked_mul(reconstruction_pixel_count)
                .ok_or(PyramidRowReconstructionPlanError::ExtentOverflow)?;
            let reconstructor = configuration
                .initial_reconstructor
                .unwrap_or_else(|| vec![0.0; reconstructor_elements]);
            let plan = PyramidRowReconstructionPlan::new(
                pixels,
                PreparedGemv::new(
                    configuration.reconstructed_count,
                    reconstruction_pixel_count,
                    Arc::from(reconstructor),
                )?,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = PyramidRowReconstructionPreparationError;
        process |plan, workspace| {
            let Some(sample) = row_sample else {
                workspace.plan_adopted();
                return Err(PyramidRowReconstructionProcessError::MissingSampleMetadata);
            };
            let first_row = usize::try_from(sample.offset())
                .map_err(|_| PyramidRowReconstructionProcessError::OffsetOverflow)?;
            let [mean] = mean_pupil_intensity else {
                return Err(PyramidRowReconstructionProcessError::MeanOutputExtent);
            };
            let result = plan.process_block(
                PyramidRowReconstructionOutput::new(
                    reconstruction_pixels,
                    mean,
                    reconstructed,
                ),
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
        process_error = PyramidRowReconstructionProcessError;
        reset |_plan, workspace| {
            workspace.reset();
        }
        plan_adopted |_plan, workspace| {
            workspace.plan_adopted();
        }
    }
}
