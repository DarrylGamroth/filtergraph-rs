//! Row-block Shack–Hartmann sensing with composable measurement-block output.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use super::row_block::{RowBlockDropReason, RowBlockFrame, RowBlockInfo};
use super::shack_hartmann_image::initial_subaperture_origins;
use crate::{
    AlgorithmProcessResult, ShackHartmannImagePlan, ShackHartmannOutputError,
    ShackHartmannWeightedCentroidError, ShackHartmannWeightedCentroidPlan,
    ShackHartmannWeightedCentroidWorkspace,
};

/// Invalid preparation of row-block sensing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShackHartmannWeightedMeasurementBlockPreparationError {
    /// Image-backed sensing preparation failed.
    Sensing(ShackHartmannWeightedCentroidError),
    /// The block height must be positive, smaller than, and divide the image height.
    BlockRows,
    /// The image must have compact row-major storage starting at its first pixel.
    ImageLayout,
    /// The packed extent or portable column-index capacity overflowed.
    ExtentOverflow,
}

impl fmt::Display for ShackHartmannWeightedMeasurementBlockPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sensing(error) => error.fmt(formatter),
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
impl core::error::Error for ShackHartmannWeightedMeasurementBlockPreparationError {}
impl From<ShackHartmannWeightedCentroidError>
    for ShackHartmannWeightedMeasurementBlockPreparationError
{
    fn from(error: ShackHartmannWeightedCentroidError) -> Self {
        Self::Sensing(error)
    }
}

/// Invalid caller storage or failure while measuring completed subapertures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShackHartmannWeightedMeasurementBlockError {
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
    ShackHartmann(ShackHartmannWeightedCentroidError),
}
impl fmt::Display for ShackHartmannWeightedMeasurementBlockError {
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
        }
    }
}
impl core::error::Error for ShackHartmannWeightedMeasurementBlockError {}

/// Outcome of consuming one row block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShackHartmannWeightedMeasurementBlockResult {
    /// A measurement block was published, including when its count is zero.
    Published {
        /// Whether this block follows detected or reported frame loss.
        discontinuity: bool,
    },
    /// The partial detector frame was abandoned without publishing output.
    Dropped(RowBlockDropReason),
}

/// Prepared image-backed sensing and scientific-order completion schedule.
#[derive(Clone, Debug)]
pub struct ShackHartmannWeightedMeasurementBlockPlan {
    sensing: ShackHartmannWeightedCentroidPlan,
    completion_schedule: Box<[usize]>,
    block_rows: usize,
    block_elements: usize,
    capacity: usize,
}
impl ShackHartmannWeightedMeasurementBlockPlan {
    /// Prepare sensing of fixed-height compact detector-row blocks.
    pub fn new(
        sensing: ShackHartmannWeightedCentroidPlan,
        block_rows: usize,
    ) -> Result<Self, ShackHartmannWeightedMeasurementBlockPreparationError> {
        let regions = sensing.sensing().regions();
        let (image_rows, image_columns) = regions.image_size();
        if block_rows == 0 || block_rows >= image_rows || image_rows % block_rows != 0 {
            return Err(ShackHartmannWeightedMeasurementBlockPreparationError::BlockRows);
        }
        if regions.row_stride() != image_columns || regions.first_pixel() != 0 {
            return Err(ShackHartmannWeightedMeasurementBlockPreparationError::ImageLayout);
        }
        let capacity = regions
            .len()
            .checked_mul(2)
            .filter(|&capacity| capacity <= u32::MAX as usize / core::mem::size_of::<f32>())
            .ok_or(ShackHartmannWeightedMeasurementBlockPreparationError::ExtentOverflow)?;
        image_rows
            .checked_mul(image_columns)
            .filter(|&extent| extent <= isize::MAX as usize / core::mem::size_of::<f32>())
            .ok_or(ShackHartmannWeightedMeasurementBlockPreparationError::ExtentOverflow)?;
        let block_elements = block_rows
            .checked_mul(image_columns)
            .ok_or(ShackHartmannWeightedMeasurementBlockPreparationError::ExtentOverflow)?;
        let region_rows = regions.region_size().0;
        let mut completion_schedule = (0..regions.len()).collect::<Vec<_>>();
        completion_schedule
            .sort_by_key(|&region| (regions.origins()[region].0 + region_rows, region));
        Ok(Self {
            sensing,
            completion_schedule: completion_schedule.into_boxed_slice(),
            block_rows,
            block_elements,
            capacity,
        })
    }

    /// Return the prepared image-backed sensing operation.
    pub const fn sensing(&self) -> &ShackHartmannWeightedCentroidPlan {
        &self.sensing
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
    ) -> Result<Self, ShackHartmannWeightedMeasurementBlockPreparationError> {
        Self::new(
            self.sensing.with_u32_subaperture_origins(values)?,
            self.block_rows,
        )
    }
    /// Replace pair-interleaved detector coordinates.
    pub fn with_pair_interleaved_coordinates(
        &self,
        values: &[f32],
    ) -> Result<Self, ShackHartmannWeightedMeasurementBlockPreparationError> {
        Self::new(
            self.sensing.with_pair_interleaved_coordinates(values)?,
            self.block_rows,
        )
    }
    /// Replace pair-interleaved reference slopes.
    pub fn with_reference_slopes(
        &self,
        values: &[f32],
    ) -> Result<Self, ShackHartmannWeightedMeasurementBlockPreparationError> {
        Self::new(self.sensing.with_reference_slopes(values)?, self.block_rows)
    }
    /// Replace pair-interleaved pixel and flux thresholds.
    pub fn with_pair_interleaved_thresholds(
        &self,
        values: &[f32],
    ) -> Result<Self, ShackHartmannWeightedMeasurementBlockPreparationError> {
        Self::new(
            self.sensing.with_pair_interleaved_thresholds(values)?,
            self.block_rows,
        )
    }
    /// Replace the active subaperture mask.
    pub fn with_active(
        &self,
        values: &[bool],
    ) -> Result<Self, ShackHartmannWeightedMeasurementBlockPreparationError> {
        Self::new(self.sensing.with_active(values)?, self.block_rows)
    }

    /// Replace the nonnegative per-subaperture weight bank.
    pub fn with_weights(
        &self,
        values: &[f32],
    ) -> Result<Self, ShackHartmannWeightedMeasurementBlockPreparationError> {
        Self::new(self.sensing.with_weights(values)?, self.block_rows)
    }

    /// Allocate bounded mutable storage for one execution owner.
    pub fn workspace(&self) -> ShackHartmannWeightedMeasurementBlockWorkspace {
        let (rows, columns) = self.sensing.sensing().regions().image_size();
        let subapertures = self.sensing.sensing().measurement().len();
        ShackHartmannWeightedMeasurementBlockWorkspace {
            weighted: self.sensing.workspace(),
            image: vec![0.0; rows * columns].into_boxed_slice(),
            slopes: vec![0.0; self.capacity].into_boxed_slice(),
            flux: vec![0.0; subapertures].into_boxed_slice(),
            validity: vec![false; subapertures].into_boxed_slice(),
            ready: vec![false; subapertures].into_boxed_slice(),
            frame: RowBlockFrame::new(),
            schedule_cursor: 0,
            published_subapertures: 0,
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
        measurements: &mut [f32],
        columns: &mut [u32],
        count: &mut [u32],
        workspace: &mut ShackHartmannWeightedMeasurementBlockWorkspace,
        block: &[f32],
        info: RowBlockInfo,
    ) -> Result<
        ShackHartmannWeightedMeasurementBlockResult,
        ShackHartmannWeightedMeasurementBlockError,
    > {
        let (image_rows, image_columns) = self.sensing.sensing().regions().image_size();
        let subapertures = self.sensing.sensing().measurement().len();
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
                "workspace-weighted-pixels",
                workspace.weighted.pixel_count(),
                self.sensing.sensing().measurement().pixel_count(),
            ),
            ("measurements", measurements.len(), self.capacity),
            ("columns", columns.len(), self.capacity),
            ("count", count.len(), 1),
            ("row-block", block.len(), self.block_elements),
        ] {
            if actual != expected {
                return Err(ShackHartmannWeightedMeasurementBlockError::Extent {
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
                return Ok(ShackHartmannWeightedMeasurementBlockResult::Dropped(reason));
            }
        };
        if info.first_row == 0 {
            workspace.clear_partial();
        }
        let start_pixel = info.first_row * self.sensing.sensing().regions().image_size().1;
        workspace.image[start_pixel..start_pixel + self.block_elements].copy_from_slice(block);
        let region_rows = self.sensing.sensing().regions().region_size().0;
        while let Some(&subaperture) = self.completion_schedule.get(workspace.schedule_cursor) {
            if self.sensing.sensing().regions().origins()[subaperture].0 + region_rows
                > accepted.end_row
            {
                break;
            }
            workspace.ready[subaperture] = true;
            workspace.schedule_cursor += 1;
        }
        let start = workspace.published_subapertures;
        let mut end = start;
        while workspace.ready.get(end).copied() == Some(true) {
            end += 1;
        }
        if info.terminal && end != self.sensing.sensing().measurement().len() {
            workspace.abandon();
            return Ok(ShackHartmannWeightedMeasurementBlockResult::Dropped(
                RowBlockDropReason::RowRange,
            ));
        }
        if start != end
            && let Err(error) = self.sensing.process_range(
                (
                    &mut workspace.slopes,
                    &mut workspace.flux,
                    &mut workspace.validity,
                ),
                &mut workspace.weighted,
                &workspace.image,
                start..end,
            )
        {
            workspace.abandon();
            return Err(ShackHartmannWeightedMeasurementBlockError::ShackHartmann(
                error,
            ));
        }
        let first_column = start * 2;
        let last_column = end * 2;
        let block_count = last_column - first_column;
        measurements[..block_count].copy_from_slice(&workspace.slopes[first_column..last_column]);
        for (destination, column) in columns[..block_count]
            .iter_mut()
            .zip(first_column..last_column)
        {
            // Preparation limits the complete slope capacity to u32::MAX.
            *destination = column as u32;
        }
        count[0] = block_count as u32;
        workspace.published_subapertures = end;
        let discontinuity =
            info.discontinuity || (info.first_row == 0 && workspace.frame.discontinuity());
        if info.terminal {
            workspace.frame.complete();
            workspace.clear_partial();
        }
        Ok(ShackHartmannWeightedMeasurementBlockResult::Published { discontinuity })
    }
}

/// Bounded mutable progressive sensing state for one execution owner.
#[derive(Debug)]
pub struct ShackHartmannWeightedMeasurementBlockWorkspace {
    weighted: ShackHartmannWeightedCentroidWorkspace,
    image: Box<[f32]>,
    slopes: Box<[f32]>,
    flux: Box<[f32]>,
    validity: Box<[bool]>,
    ready: Box<[bool]>,
    frame: RowBlockFrame,
    schedule_cursor: usize,
    published_subapertures: usize,
}
impl ShackHartmannWeightedMeasurementBlockWorkspace {
    /// Return the number of scientific-order subapertures published so far.
    pub const fn published_subapertures(&self) -> usize {
        self.published_subapertures
    }
    /// Return the active detector-frame sequence, if any.
    pub const fn sequence(&self) -> Option<u64> {
        self.frame.sequence()
    }
    /// Clear numerical planes and begin a clean detector-frame history.
    pub fn reset(&mut self) {
        self.image.fill(0.0);
        self.slopes.fill(0.0);
        self.flux.fill(0.0);
        self.validity.fill(false);
        self.frame.reset();
        self.clear_partial();
    }
    fn abandon(&mut self) {
        self.frame.abandon();
        self.clear_partial();
    }
    fn clear_partial(&mut self) {
        self.ready.fill(false);
        self.schedule_cursor = 0;
        self.published_subapertures = 0;
    }
}

/// Invalid sample metadata or sensing call at the declaration boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShackHartmannWeightedMeasurementBlockProcessError {
    /// Input row-block identity is missing.
    MissingSampleMetadata,
    /// The row offset cannot be represented on this target.
    OffsetOverflow,
    /// The sensing call failed.
    Algorithm(ShackHartmannWeightedMeasurementBlockError),
}
impl fmt::Display for ShackHartmannWeightedMeasurementBlockProcessError {
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
impl core::error::Error for ShackHartmannWeightedMeasurementBlockProcessError {}
impl From<ShackHartmannWeightedMeasurementBlockError>
    for ShackHartmannWeightedMeasurementBlockProcessError
{
    fn from(error: ShackHartmannWeightedMeasurementBlockError) -> Self {
        Self::Algorithm(error)
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 progressive Shack–Hartmann measurement-block declaration.
    pub ShackHartmannWeightedMeasurementBlockF32 {
        label = "shack-hartmann-weighted-measurement-block-f32";
        configuration ShackHartmannWeightedMeasurementBlockF32Configuration {
            image_size: (usize, usize) = GraphRebuild;
            row_block_rows: usize = GraphRebuild;
            subaperture_size: (usize, usize) = GraphRebuild;
            subaperture_count: usize = GraphRebuild;
            initial_subaperture_origins: Option<Vec<[usize; 2]>> = Construction;
            coordinate_scale: f32 = Construction;
            pixel_threshold: f32 = Construction;
            flux_threshold: f32 = Construction;
            reference_slopes: Option<Vec<f32>> = Construction;
            active: Option<Vec<bool>> = Construction;
            initial_weights: Vec<Vec<f32>> = Construction;
            measurements_schema: String = GraphRebuild;
            column_indices_schema: String = GraphRebuild;
            measurement_count_schema: String = GraphRebuild;
        }
        plan = ShackHartmannWeightedMeasurementBlockPlan;
        workspace = ShackHartmannWeightedMeasurementBlockWorkspace;
        properties = none;
        dimensions |plan, _workspace| {
            image_rows = plan.sensing().sensing().regions().image_size().0;
            image_columns = plan.sensing().sensing().regions().image_size().1;
            row_block_rows = plan.block_rows();
            subaperture_count = plan.sensing().sensing().measurement().len();
            pixel_count = plan.sensing().sensing().measurement().pixel_count();
            slope_count = plan.capacity();
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
            measurements: f32 [slope_count] = "measurements",
                schema = configuration.measurements_schema.as_str(),
                metadata_from = "row-block", publication = conditional,
                rate = [block_rate_multiplier, 1];
            columns: u32 [slope_count] = "columns",
                schema = configuration.column_indices_schema.as_str(),
                metadata_from = "row-block", publication = conditional,
                rate = [block_rate_multiplier, 1];
            count: u32 [1] = "count",
                schema = configuration.measurement_count_schema.as_str(),
                metadata_from = "row-block", publication = conditional,
                rate = [block_rate_multiplier, 1];
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
            weights: f32 [subaperture_count, pixel_count] = "weights",
                schema = crate::schemas::SHACK_HARTMANN_WEIGHTS_V1,
                replace |plan, values| { plan.with_weights(values) };
        }
        parameter_error = ShackHartmannWeightedMeasurementBlockPreparationError;
        prepare |configuration| {
            let origins = initial_subaperture_origins(
                configuration.image_size,
                configuration.subaperture_size,
                configuration.subaperture_count,
                configuration.initial_subaperture_origins,
            ).map_err(|error| ShackHartmannWeightedMeasurementBlockPreparationError::Sensing(error.into()))?;
            let sensing = ShackHartmannImagePlan::from_detector_regions(
                configuration.image_size,
                configuration.subaperture_size,
                &origins, configuration.coordinate_scale, configuration.pixel_threshold,
                configuration.flux_threshold, configuration.reference_slopes, configuration.active,
            ).map_err(|error| ShackHartmannWeightedMeasurementBlockPreparationError::Sensing(error.into()))?;
            if configuration.initial_weights.len() != configuration.subaperture_count ||
                configuration.initial_weights.iter().any(|row| row.len() != sensing.measurement().pixel_count()) {
                return Err(ShackHartmannWeightedMeasurementBlockPreparationError::Sensing(ShackHartmannWeightedCentroidError::WeightExtent));
            }
            let weights = configuration.initial_weights.into_iter().flatten().collect::<Vec<_>>();
            let sensing = ShackHartmannWeightedCentroidPlan::new(sensing, weights)?;
            let plan = ShackHartmannWeightedMeasurementBlockPlan::new(sensing, configuration.row_block_rows)?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = ShackHartmannWeightedMeasurementBlockPreparationError;
        process |plan, workspace| {
            let Some(sample)=row_sample else {
                workspace.abandon();
                return Err(ShackHartmannWeightedMeasurementBlockProcessError::MissingSampleMetadata);
            };
            let first_row = usize::try_from(sample.offset())
                .map_err(|_| ShackHartmannWeightedMeasurementBlockProcessError::OffsetOverflow)?;
            let result = plan.process_block(measurements, columns, count, workspace, row_block,
                RowBlockInfo {
                    sequence: sample.sequence(), first_row, terminal: sample.is_terminal(),
                    discontinuity: sample.is_discontinuous(), corrupted: sample.is_corrupted(),
                },
            )?;
            Ok(match result {
                ShackHartmannWeightedMeasurementBlockResult::Dropped(_) => AlgorithmProcessResult::deferred(),
                ShackHartmannWeightedMeasurementBlockResult::Published { discontinuity } => {
                    AlgorithmProcessResult::complete_with_metadata(
                        sample.with_discontinuity(discontinuity),
                    )
                }
            })
        }
        process_error = ShackHartmannWeightedMeasurementBlockProcessError;
        reset |_plan, workspace| { workspace.reset(); }
        plan_adopted |_plan, workspace| { workspace.abandon(); }
    }
}
