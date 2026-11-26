//! Row-block Shack–Hartmann sensing with composable measurement-block output.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use super::row_block::{RowBlockDropReason, RowBlockFrame, RowBlockInfo, RowBlockResult};
use super::shack_hartmann_image::initial_subaperture_origins;
use crate::{
    AlgorithmProcessResult, MatrixElement, PreparedGemv, PreparedGemvError, ProcessError,
    ShackHartmannImagePlan, ShackHartmannImagePreparationError, ShackHartmannOutput,
    ShackHartmannOutputError,
};

/// Invalid preparation of row-block sensing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannRowReconstructionPreparationError {
    /// Image-backed sensing preparation failed.
    Sensing(ShackHartmannImagePreparationError),
    /// Dense reconstructor calibration is invalid.
    Reconstructor(PreparedGemvError),
    /// The block height must be positive, smaller than, and divide the image height.
    BlockRows,
    /// The image must have compact row-major storage starting at its first pixel.
    ImageLayout,
    /// The packed extent or portable column-index capacity overflowed.
    ExtentOverflow,
}

impl fmt::Display for ShackHartmannRowReconstructionPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sensing(error) => error.fmt(formatter),
            Self::Reconstructor(error) => error.fmt(formatter),
            Self::BlockRows => formatter.write_str(
                "row-block height must be positive, smaller than, and divide image height",
            ),
            Self::ImageLayout => {
                formatter.write_str("measurement blocks require compact row-major image storage")
            }
            Self::ExtentOverflow => formatter.write_str("measurement-block extent overflow"),
        }
    }
}
impl core::error::Error for ShackHartmannRowReconstructionPreparationError {}
impl From<PreparedGemvError> for ShackHartmannRowReconstructionPreparationError {
    fn from(error: PreparedGemvError) -> Self {
        Self::Reconstructor(error)
    }
}
impl From<ShackHartmannImagePreparationError> for ShackHartmannRowReconstructionPreparationError {
    fn from(error: ShackHartmannImagePreparationError) -> Self {
        Self::Sensing(error)
    }
}

/// Invalid caller storage or failure while measuring completed subapertures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannRowReconstructionError {
    /// A caller-owned port has the wrong extent.
    Extent {
        /// The affected port.
        port: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// Shack–Hartmann output scratch extents were inconsistent.
    ShackHartmannOutput(ShackHartmannOutputError),
    /// Image-backed sensing rejected a completed subaperture range.
    ShackHartmann(ProcessError),
    /// Scientific-column accumulation produced a nonfinite output coordinate.
    NonFiniteReconstruction,
}
impl fmt::Display for ShackHartmannRowReconstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Extent {
                port,
                expected,
                actual,
            } => write!(
                formatter,
                "{port} extent {actual} does not match {expected}"
            ),
            Self::ShackHartmannOutput(error) => error.fmt(formatter),
            Self::ShackHartmann(error) => error.fmt(formatter),
            Self::NonFiniteReconstruction => {
                formatter.write_str("nonfinite Shack-Hartmann row reconstruction")
            }
        }
    }
}
impl core::error::Error for ShackHartmannRowReconstructionError {}

/// Prepared image-backed sensing and scientific-order completion schedule.
#[derive(Clone, Debug)]
pub struct ShackHartmannRowReconstructionPlan<T> {
    sensing: ShackHartmannImagePlan<T>,
    reconstructor: PreparedGemv<T>,
    completion_schedule: Box<[usize]>,
    block_rows: usize,
    block_elements: usize,
    capacity: usize,
}
impl<T: MatrixElement> ShackHartmannRowReconstructionPlan<T> {
    /// Prepare sensing of fixed-height compact detector-row blocks.
    pub fn new(
        sensing: ShackHartmannImagePlan<T>,
        reconstructor: PreparedGemv<T>,
        block_rows: usize,
    ) -> Result<Self, ShackHartmannRowReconstructionPreparationError> {
        let regions = sensing.regions();
        let (image_rows, image_columns) = regions.image_size();
        if block_rows == 0 || block_rows >= image_rows || image_rows % block_rows != 0 {
            return Err(ShackHartmannRowReconstructionPreparationError::BlockRows);
        }
        if regions.row_stride() != image_columns || regions.first_pixel() != 0 {
            return Err(ShackHartmannRowReconstructionPreparationError::ImageLayout);
        }
        image_rows
            .checked_mul(image_columns)
            .filter(|&n| n <= isize::MAX as usize / core::mem::size_of::<T>())
            .ok_or(ShackHartmannRowReconstructionPreparationError::ExtentOverflow)?;
        let capacity = regions
            .len()
            .checked_mul(2)
            .filter(|&capacity| u32::try_from(capacity).is_ok())
            .ok_or(ShackHartmannRowReconstructionPreparationError::ExtentOverflow)?;
        if reconstructor.columns() != capacity {
            return Err(ShackHartmannRowReconstructionPreparationError::ExtentOverflow);
        }
        let block_elements = block_rows
            .checked_mul(image_columns)
            .ok_or(ShackHartmannRowReconstructionPreparationError::ExtentOverflow)?;
        let region_rows = regions.region_size().0;
        let mut completion_schedule = (0..regions.len()).collect::<Vec<_>>();
        completion_schedule
            .sort_by_key(|&region| (regions.origins()[region].0 + region_rows, region));
        Ok(Self {
            sensing,
            reconstructor,
            completion_schedule: completion_schedule.into_boxed_slice(),
            block_rows,
            block_elements,
            capacity,
        })
    }

    /// Return the prepared image-backed sensing operation.
    pub const fn sensing(&self) -> &ShackHartmannImagePlan<T> {
        &self.sensing
    }
    /// Borrow the canonical row-major dense reconstructor.
    pub const fn reconstructor(&self) -> &PreparedGemv<T> {
        &self.reconstructor
    }
    /// Return the fixed number of detector rows per input block.
    pub const fn block_rows(&self) -> usize {
        self.block_rows
    }
    /// Return the output measurement and column-index capacity.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
    /// Return subapertures sorted by final required detector row.
    pub fn completion_schedule(&self) -> &[usize] {
        &self.completion_schedule
    }

    /// Replace zero-based subaperture origins without changing port formats.
    pub fn with_u32_subaperture_origins(
        &self,
        values: &[u32],
    ) -> Result<Self, ShackHartmannRowReconstructionPreparationError> {
        Self::new(
            self.sensing.with_u32_subaperture_origins(values)?,
            self.reconstructor.clone(),
            self.block_rows,
        )
    }
    /// Replace pair-interleaved detector coordinates.
    pub fn with_pair_interleaved_coordinates(
        &self,
        values: &[T],
    ) -> Result<Self, ShackHartmannRowReconstructionPreparationError> {
        Self::new(
            self.sensing.with_pair_interleaved_coordinates(values)?,
            self.reconstructor.clone(),
            self.block_rows,
        )
    }
    /// Replace pair-interleaved reference slopes.
    pub fn with_reference_slopes(
        &self,
        values: &[T],
    ) -> Result<Self, ShackHartmannRowReconstructionPreparationError> {
        Self::new(
            self.sensing.with_reference_slopes(values)?,
            self.reconstructor.clone(),
            self.block_rows,
        )
    }
    /// Replace pair-interleaved pixel and flux thresholds.
    pub fn with_pair_interleaved_thresholds(
        &self,
        values: &[T],
    ) -> Result<Self, ShackHartmannRowReconstructionPreparationError> {
        Self::new(
            self.sensing.with_pair_interleaved_thresholds(values)?,
            self.reconstructor.clone(),
            self.block_rows,
        )
    }
    /// Replace the active subaperture mask.
    pub fn with_active(
        &self,
        values: &[bool],
    ) -> Result<Self, ShackHartmannRowReconstructionPreparationError> {
        Self::new(
            self.sensing.with_active(values)?,
            self.reconstructor.clone(),
            self.block_rows,
        )
    }

    /// Replace the equal-shape row-major dense calibration.
    pub fn with_reconstructor(
        &self,
        values: &[T],
    ) -> Result<Self, ShackHartmannRowReconstructionPreparationError> {
        Self::new(
            self.sensing.clone(),
            PreparedGemv::new(
                self.reconstructor.rows(),
                self.reconstructor.columns(),
                alloc::sync::Arc::from(values),
            )?,
            self.block_rows,
        )
    }

    /// Allocate bounded mutable storage for one execution owner.
    pub fn workspace(&self) -> ShackHartmannRowReconstructionWorkspace<T> {
        let (rows, columns) = self.sensing.regions().image_size();
        let subapertures = self.sensing.measurement().len();
        ShackHartmannRowReconstructionWorkspace {
            candidate_reconstructed: vec![T::zero(); self.reconstructor.rows()].into_boxed_slice(),
            image: vec![T::zero(); rows * columns].into_boxed_slice(),
            slopes: vec![T::zero(); self.capacity].into_boxed_slice(),
            flux: vec![T::zero(); subapertures].into_boxed_slice(),
            validity: vec![false; subapertures].into_boxed_slice(),
            ready: vec![false; subapertures].into_boxed_slice(),
            frame: RowBlockFrame::new(),
            schedule_cursor: 0,
            processed_subapertures: 0,
        }
    }

    /// Publish the newly ready scientific slope prefix for one accepted row block.
    ///
    /// Each prefix entry has a matching zero-based reconstructor column. Only
    /// `count[0]` entries are valid. A valid block with no completed subapertures
    /// still publishes a count of zero so downstream identity tracking advances.
    /// Rejected or failed blocks leave all caller-owned outputs unchanged.
    pub fn process_block(
        &self,
        reconstructed: &mut [T],
        workspace: &mut ShackHartmannRowReconstructionWorkspace<T>,
        block: &[T],
        info: RowBlockInfo,
    ) -> Result<RowBlockResult, ShackHartmannRowReconstructionError> {
        let (image_rows, image_columns) = self.sensing.regions().image_size();
        let subapertures = self.sensing.measurement().len();
        for (port, actual, expected) in [
            (
                "workspace-image",
                workspace.image.len(),
                image_rows * image_columns,
            ),
            ("workspace-slopes", workspace.slopes.len(), self.capacity),
            ("workspace-flux", workspace.flux.len(), subapertures),
            ("workspace-validity", workspace.validity.len(), subapertures),
            ("workspace-ready", workspace.ready.len(), subapertures),
            (
                "workspace-reconstructed",
                workspace.candidate_reconstructed.len(),
                self.reconstructor.rows(),
            ),
            (
                "reconstructed",
                reconstructed.len(),
                self.reconstructor.rows(),
            ),
            ("row-block", block.len(), self.block_elements),
        ] {
            if actual != expected {
                return Err(ShackHartmannRowReconstructionError::Extent {
                    port,
                    expected,
                    actual,
                });
            }
        }
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
        let start_pixel = info.first_row * self.sensing.regions().image_size().1;
        workspace.image[start_pixel..start_pixel + self.block_elements].copy_from_slice(block);
        let region_rows = self.sensing.regions().region_size().0;
        while let Some(&subaperture) = self.completion_schedule.get(workspace.schedule_cursor) {
            if self.sensing.regions().origins()[subaperture].0 + region_rows > accepted.end_row {
                break;
            }
            workspace.ready[subaperture] = true;
            workspace.schedule_cursor += 1;
        }
        let start = workspace.processed_subapertures;
        let mut end = start;
        while workspace.ready.get(end).copied() == Some(true) {
            end += 1;
        }
        if info.terminal && end != self.sensing.measurement().len() {
            workspace.abandon();
            return Ok(RowBlockResult::Dropped(RowBlockDropReason::RowRange));
        }
        if start != end {
            let output = ShackHartmannOutput::new(
                &mut workspace.slopes,
                &mut workspace.flux,
                &mut workspace.validity,
            )
            .map_err(ShackHartmannRowReconstructionError::ShackHartmannOutput)?;
            if let Err(error) =
                self.sensing
                    .process_image_range(output, &workspace.image, start..end)
            {
                workspace.abandon();
                return Err(ShackHartmannRowReconstructionError::ShackHartmann(error));
            }
        }
        for column in start * 2..end * 2 {
            let slope = workspace.slopes[column];
            for row in 0..self.reconstructor.rows() {
                workspace.candidate_reconstructed[row] = workspace.candidate_reconstructed[row]
                    + self.reconstructor.matrix()[row * self.capacity + column] * slope;
            }
        }
        if workspace
            .candidate_reconstructed
            .iter()
            .any(|v| !v.is_finite())
        {
            workspace.abandon();
            return Err(ShackHartmannRowReconstructionError::NonFiniteReconstruction);
        }
        workspace.processed_subapertures = end;
        if !info.terminal {
            return Ok(RowBlockResult::Pending);
        }
        reconstructed.copy_from_slice(&workspace.candidate_reconstructed);
        let outcome = RowBlockResult::Complete {
            sequence: info.sequence,
            discontinuity: workspace.frame.discontinuity(),
        };
        workspace.frame.complete();
        workspace.clear_partial();
        Ok(outcome)
    }
}

/// Bounded mutable progressive sensing state for one execution owner.
#[derive(Debug)]
pub struct ShackHartmannRowReconstructionWorkspace<T> {
    candidate_reconstructed: Box<[T]>,
    image: Box<[T]>,
    slopes: Box<[T]>,
    flux: Box<[T]>,
    validity: Box<[bool]>,
    ready: Box<[bool]>,
    frame: RowBlockFrame,
    schedule_cursor: usize,
    processed_subapertures: usize,
}
impl<T: MatrixElement> ShackHartmannRowReconstructionWorkspace<T> {
    /// Return the number of scientific-order subapertures accumulated so far.
    pub const fn processed_subapertures(&self) -> usize {
        self.processed_subapertures
    }
    /// Return the active detector-frame sequence, if any.
    pub const fn sequence(&self) -> Option<u64> {
        self.frame.sequence()
    }
    /// Clear numerical planes and begin a clean detector-frame history.
    pub fn reset(&mut self)
    where
        T: MatrixElement,
    {
        self.image.fill(T::zero());
        self.slopes.fill(T::zero());
        self.flux.fill(T::zero());
        self.validity.fill(false);
        self.frame.reset();
        self.clear_partial();
    }
    fn abandon(&mut self) {
        self.frame.abandon();
        self.clear_partial();
    }
    fn clear_partial(&mut self) {
        self.candidate_reconstructed.fill(T::zero());
        self.ready.fill(false);
        self.schedule_cursor = 0;
        self.processed_subapertures = 0;
    }
}

/// Invalid sample metadata or sensing call at the declaration boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannRowReconstructionProcessError {
    /// Input row-block identity is missing.
    MissingSampleMetadata,
    /// The row offset cannot be represented on this target.
    OffsetOverflow,
    /// The sensing call failed.
    Algorithm(ShackHartmannRowReconstructionError),
}
impl fmt::Display for ShackHartmannRowReconstructionProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSampleMetadata => {
                formatter.write_str("row-block sample metadata is required")
            }
            Self::OffsetOverflow => formatter.write_str("row-block offset does not fit usize"),
            Self::Algorithm(error) => error.fmt(formatter),
        }
    }
}
impl core::error::Error for ShackHartmannRowReconstructionProcessError {}
impl From<ShackHartmannRowReconstructionError> for ShackHartmannRowReconstructionProcessError {
    fn from(error: ShackHartmannRowReconstructionError) -> Self {
        Self::Algorithm(error)
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 progressive Shack–Hartmann measurement-block declaration.
    pub ShackHartmannRowReconstructorF32 {
        label = "shwfs-row-reconstructor-f32";
        configuration ShackHartmannRowReconstructorF32Configuration {
            image_size: (usize, usize) = GraphRebuild;
            row_block_rows: usize = GraphRebuild;
            subaperture_size: (usize, usize) = GraphRebuild;
            subaperture_count: usize = GraphRebuild;
            reconstructed_count: usize = GraphRebuild;
            initial_subaperture_origins: Option<Vec<[usize; 2]>> = Construction;
            coordinate_scale: f32 = Construction;
            pixel_threshold: f32 = Construction;
            flux_threshold: f32 = Construction;
            reference_slopes: Option<Vec<f32>> = Construction;
            active: Option<Vec<bool>> = Construction;
            initial_reconstructor: Option<Vec<f32>> = Construction;
            reconstructed_schema: String = GraphRebuild;
        }
        plan = ShackHartmannRowReconstructionPlan<f32>;
        workspace = ShackHartmannRowReconstructionWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            image_rows = plan.sensing().regions().image_size().0;
            image_columns = plan.sensing().regions().image_size().1;
            row_block_rows = plan.block_rows();
            subaperture_count = plan.sensing().measurement().len();
            pixel_count = plan.sensing().measurement().pixel_count();
            slope_count = plan.capacity();
            reconstructed_count=plan.reconstructor().rows();
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
            reconstructed:f32 [reconstructed_count]="reconstructed",schema=configuration.reconstructed_schema.as_str(),metadata_from="row-block",publication=conditional;
        }
        parameters {
            subaperture_origins: u32 [subaperture_count, 2] = "subaperture-origins",
                schema = crate::schemas::REGION_ORIGINS_V1,
                replace |plan, values| { plan.with_u32_subaperture_origins(values) };
            coordinates: f32 [pixel_count, 2] = "coordinates",
                schema = crate::schemas::SHACK_HARTMANN_COORDINATES_V1,
                replace |plan, values| { plan.with_pair_interleaved_coordinates(values) };
            reference_slopes: f32 [subaperture_count, 2] = "reference-slopes",
                schema = crate::schemas::SHACK_HARTMANN_REFERENCE_SLOPES_V1,
                replace |plan, values| { plan.with_reference_slopes(values) };
            thresholds: f32 [subaperture_count, 2] = "thresholds",
                schema = crate::schemas::SHACK_HARTMANN_THRESHOLDS_V1,
                replace |plan, values| { plan.with_pair_interleaved_thresholds(values) };
            active: bool [subaperture_count] = "active",
                schema = crate::schemas::SHACK_HARTMANN_ACTIVE_V1,
                replace |plan, values| { plan.with_active(values) };
            reconstructor:f32 [reconstructed_count,slope_count]="reconstructor",schema=crate::schemas::SHWFS_RECONSTRUCTOR_V1,replace |plan,values| {plan.with_reconstructor(values)};
        }
        parameter_error = ShackHartmannRowReconstructionPreparationError;
        prepare |configuration| {
            let origins = initial_subaperture_origins(
                configuration.image_size,
                configuration.subaperture_size,
                configuration.subaperture_count,
                configuration.initial_subaperture_origins,
            )?;
            let sensing = ShackHartmannImagePlan::from_detector_regions(
                configuration.image_size,
                configuration.subaperture_size,
                &origins, configuration.coordinate_scale, configuration.pixel_threshold,
                configuration.flux_threshold, configuration.reference_slopes, configuration.active,
            )?;
            let slope_count=configuration.subaperture_count.checked_mul(2).ok_or(ShackHartmannRowReconstructionPreparationError::ExtentOverflow)?;
            let elements=configuration.reconstructed_count.checked_mul(slope_count)
                .filter(|&n|n<=u32::MAX as usize/4).ok_or(ShackHartmannRowReconstructionPreparationError::ExtentOverflow)?;
            let matrix=configuration.initial_reconstructor.unwrap_or_else(||vec![0.0;elements]);
            let reconstructor=PreparedGemv::new(configuration.reconstructed_count,slope_count,alloc::sync::Arc::from(matrix))?;
            let plan = ShackHartmannRowReconstructionPlan::new(sensing, reconstructor, configuration.row_block_rows)?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = ShackHartmannRowReconstructionPreparationError;
        process |plan, workspace| {
            let Some(sample)=row_sample else {
                workspace.abandon();
                return Err(ShackHartmannRowReconstructionProcessError::MissingSampleMetadata);
            };
            let first_row = usize::try_from(sample.offset())
                .map_err(|_| ShackHartmannRowReconstructionProcessError::OffsetOverflow)?;
            let result = plan.process_block(reconstructed, workspace, row_block,
                RowBlockInfo {
                    sequence: sample.sequence(), first_row, terminal: sample.is_terminal(),
                    discontinuity: sample.is_discontinuous(), corrupted: sample.is_corrupted(),
                },
            )?;
            Ok(match result {
                RowBlockResult::Dropped(_) | RowBlockResult::Pending => AlgorithmProcessResult::deferred(),
                RowBlockResult::Complete { discontinuity,.. } => AlgorithmProcessResult::complete_with_metadata(sample.with_offset(0).with_terminal(true).with_discontinuity(discontinuity)),
            })
        }

        process_error = ShackHartmannRowReconstructionProcessError;
        reset |_plan, workspace| { workspace.reset(); }
        plan_adopted |_plan, workspace| { workspace.abandon(); }
    }
}
