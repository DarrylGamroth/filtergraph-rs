//! Row-block Shack–Hartmann sensing with composable measurement-block output.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use super::detector_readout::DetectorReadoutFrame;
use super::shack_hartmann_image::initial_subaperture_origins;
use crate::{
    AlgorithmProcessResult, MatrixElement, ProcessError, ShackHartmannImagePlan,
    ShackHartmannImagePreparationError, ShackHartmannOutput, ShackHartmannOutputError,
};
use crate::{
    DetectorReadoutBlockInfo, DetectorReadoutDropReason, DetectorReadoutPlan,
    DetectorReadoutProfileError,
};

/// Invalid preparation of row-block sensing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannReadoutMeasurementBlockPreparationError {
    /// Image-backed sensing preparation failed.
    Sensing(ShackHartmannImagePreparationError),
    /// The block height must be positive, smaller than, and divide the image height.
    Readout(DetectorReadoutProfileError),
    /// The image must have compact row-major storage starting at its first pixel.
    ImageLayout,
    /// The packed extent or portable column-index capacity overflowed.
    ExtentOverflow,
}

impl fmt::Display for ShackHartmannReadoutMeasurementBlockPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sensing(error) => error.fmt(formatter),
            Self::Readout(error) => error.fmt(formatter),
            Self::ImageLayout => {
                formatter.write_str("measurement blocks require compact row-major image storage")
            }
            Self::ExtentOverflow => formatter.write_str("measurement-block extent overflow"),
        }
    }
}
impl core::error::Error for ShackHartmannReadoutMeasurementBlockPreparationError {}
impl From<DetectorReadoutProfileError> for ShackHartmannReadoutMeasurementBlockPreparationError {
    fn from(error: DetectorReadoutProfileError) -> Self {
        Self::Readout(error)
    }
}
impl From<ShackHartmannImagePreparationError>
    for ShackHartmannReadoutMeasurementBlockPreparationError
{
    fn from(error: ShackHartmannImagePreparationError) -> Self {
        Self::Sensing(error)
    }
}

/// Invalid caller storage or failure while measuring completed subapertures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannReadoutMeasurementBlockError {
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
}
impl fmt::Display for ShackHartmannReadoutMeasurementBlockError {
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
impl core::error::Error for ShackHartmannReadoutMeasurementBlockError {}

/// Outcome of consuming one row block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannReadoutMeasurementBlockResult {
    /// A measurement block was published, including when its count is zero.
    Published {
        /// Whether this block follows detected or reported frame loss.
        discontinuity: bool,
    },
    /// The partial detector frame was abandoned without publishing output.
    Dropped(DetectorReadoutDropReason),
}

/// Prepared image-backed sensing and scientific-order completion schedule.
#[derive(Clone, Debug)]
pub struct ShackHartmannReadoutMeasurementBlockPlan<T> {
    sensing: ShackHartmannImagePlan<T>,
    subaperture_completion_ordinals: Box<[usize]>,
    completion_groups: Box<[Box<[usize]>]>,
    readout: DetectorReadoutPlan,
    block_elements: usize,
    capacity: usize,
}
impl<T: MatrixElement> ShackHartmannReadoutMeasurementBlockPlan<T> {
    /// Prepare sensing of fixed-height compact detector-row blocks.
    pub fn new(
        sensing: ShackHartmannImagePlan<T>,
        readout: DetectorReadoutPlan,
    ) -> Result<Self, ShackHartmannReadoutMeasurementBlockPreparationError> {
        let regions = sensing.regions();
        if regions.image_size() != readout.detector_size()
            || regions.row_stride() != regions.image_size().1
            || regions.first_pixel() != 0
        {
            return Err(ShackHartmannReadoutMeasurementBlockPreparationError::ImageLayout);
        }
        readout
            .detector_elements()
            .checked_mul(core::mem::size_of::<T>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(ShackHartmannReadoutMeasurementBlockPreparationError::ExtentOverflow)?;
        let capacity = regions
            .len()
            .checked_mul(2)
            .filter(|&n| n <= u32::MAX as usize / 4)
            .ok_or(ShackHartmannReadoutMeasurementBlockPreparationError::ExtentOverflow)?;
        let mut ordinals = Vec::with_capacity(regions.len());
        let mut groups = vec![Vec::new(); readout.blocks()];
        let (rows, columns) = regions.region_size();
        for (index, &(first_row, first_column)) in regions.origins().iter().enumerate() {
            let mut ordinal = 0;
            for row in first_row..first_row + rows {
                for column in first_column..first_column + columns {
                    ordinal = ordinal.max(
                        readout
                            .arrival_ordinal(row, column)
                            .expect("matching complete detector mapping"),
                    );
                }
            }
            ordinals.push(ordinal);
            groups[ordinal].push(index);
        }
        Ok(Self {
            sensing,
            subaperture_completion_ordinals: ordinals.into_boxed_slice(),
            completion_groups: groups.into_iter().map(Vec::into_boxed_slice).collect(),
            block_elements: readout.block_elements(),
            readout,
            capacity,
        })
    }

    /// Return the prepared image-backed sensing operation.
    pub const fn sensing(&self) -> &ShackHartmannImagePlan<T> {
        &self.sensing
    }
    /// Borrow the detector readout mapping.
    pub const fn readout(&self) -> &DetectorReadoutPlan {
        &self.readout
    }
    /// Return sparse output capacity.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
    /// Return each subaperture's zero-based completion ordinal.
    pub fn subaperture_completion_ordinals(&self) -> &[usize] {
        &self.subaperture_completion_ordinals
    }
    /// Borrow scientific-order completion groups for acquisition ordinals.
    pub fn completion_groups(&self) -> &[Box<[usize]>] {
        &self.completion_groups
    }

    /// Replace zero-based subaperture origins without changing port formats.
    pub fn with_u32_subaperture_origins(
        &self,
        values: &[u32],
    ) -> Result<Self, ShackHartmannReadoutMeasurementBlockPreparationError> {
        Self::new(
            self.sensing.with_u32_subaperture_origins(values)?,
            self.readout.clone(),
        )
    }
    /// Replace pair-interleaved detector coordinates.
    pub fn with_pair_interleaved_coordinates(
        &self,
        values: &[T],
    ) -> Result<Self, ShackHartmannReadoutMeasurementBlockPreparationError> {
        Self::new(
            self.sensing.with_pair_interleaved_coordinates(values)?,
            self.readout.clone(),
        )
    }
    /// Replace pair-interleaved reference slopes.
    pub fn with_reference_slopes(
        &self,
        values: &[T],
    ) -> Result<Self, ShackHartmannReadoutMeasurementBlockPreparationError> {
        Self::new(
            self.sensing.with_reference_slopes(values)?,
            self.readout.clone(),
        )
    }
    /// Replace pair-interleaved pixel and flux thresholds.
    pub fn with_pair_interleaved_thresholds(
        &self,
        values: &[T],
    ) -> Result<Self, ShackHartmannReadoutMeasurementBlockPreparationError> {
        Self::new(
            self.sensing.with_pair_interleaved_thresholds(values)?,
            self.readout.clone(),
        )
    }
    /// Replace the active subaperture mask.
    pub fn with_active(
        &self,
        values: &[bool],
    ) -> Result<Self, ShackHartmannReadoutMeasurementBlockPreparationError> {
        Self::new(self.sensing.with_active(values)?, self.readout.clone())
    }

    /// Allocate bounded mutable storage for one execution owner.
    pub fn workspace(&self) -> ShackHartmannReadoutMeasurementBlockWorkspace<T> {
        let (rows, columns) = self.sensing.regions().image_size();
        let subapertures = self.sensing.measurement().len();
        ShackHartmannReadoutMeasurementBlockWorkspace {
            image: vec![T::zero(); rows * columns].into_boxed_slice(),
            slopes: vec![T::zero(); self.capacity].into_boxed_slice(),
            flux: vec![T::zero(); subapertures].into_boxed_slice(),
            validity: vec![false; subapertures].into_boxed_slice(),
            frame: DetectorReadoutFrame::new(),
            measured_subapertures: 0,
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
        measurements: &mut [T],
        columns: &mut [u32],
        count: &mut [u32],
        workspace: &mut ShackHartmannReadoutMeasurementBlockWorkspace<T>,
        block: &[T],
        info: DetectorReadoutBlockInfo,
    ) -> Result<ShackHartmannReadoutMeasurementBlockResult, ShackHartmannReadoutMeasurementBlockError>
    {
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
            ("measurements", measurements.len(), self.capacity),
            ("columns", columns.len(), self.capacity),
            ("count", count.len(), 1),
            ("row-block", block.len(), self.block_elements),
        ] {
            if actual != expected {
                return Err(ShackHartmannReadoutMeasurementBlockError::Extent {
                    port,
                    expected,
                    actual,
                });
            }
        }
        if let Err(reason) = workspace.frame.accept(info, self.readout.blocks()) {
            workspace.measured_subapertures = 0;
            return Ok(ShackHartmannReadoutMeasurementBlockResult::Dropped(reason));
        }
        if info.ordinal == 0 {
            workspace.measured_subapertures = 0;
        }
        self.readout
            .scatter_block(&mut workspace.image, block, info.ordinal)
            .expect("validated block and detector extents");
        let mut block_count = 0;
        for &subaperture in &self.completion_groups[info.ordinal] {
            let output = ShackHartmannOutput::new(
                &mut workspace.slopes,
                &mut workspace.flux,
                &mut workspace.validity,
            )
            .map_err(ShackHartmannReadoutMeasurementBlockError::ShackHartmannOutput)?;
            if let Err(error) = self.sensing.process_image_range(
                output,
                &workspace.image,
                subaperture..subaperture + 1,
            ) {
                workspace.abandon();
                return Err(ShackHartmannReadoutMeasurementBlockError::ShackHartmann(
                    error,
                ));
            }
            measurements[block_count] = workspace.slopes[2 * subaperture];
            columns[block_count] = (2 * subaperture) as u32;
            measurements[block_count + 1] = workspace.slopes[2 * subaperture + 1];
            columns[block_count + 1] = (2 * subaperture + 1) as u32;
            block_count += 2;
            workspace.measured_subapertures += 1;
        }
        count[0] = block_count as u32;
        let discontinuity =
            info.discontinuity || (info.ordinal == 0 && workspace.frame.discontinuity());
        if info.terminal {
            debug_assert_eq!(
                workspace.measured_subapertures,
                self.sensing.measurement().len()
            );
            workspace.frame.complete();
            workspace.measured_subapertures = 0;
        }
        Ok(ShackHartmannReadoutMeasurementBlockResult::Published { discontinuity })
    }
}

/// Bounded mutable progressive sensing state for one execution owner.
#[derive(Debug)]
pub struct ShackHartmannReadoutMeasurementBlockWorkspace<T> {
    image: Box<[T]>,
    slopes: Box<[T]>,
    flux: Box<[T]>,
    validity: Box<[bool]>,
    frame: DetectorReadoutFrame,
    measured_subapertures: usize,
}
impl<T: MatrixElement> ShackHartmannReadoutMeasurementBlockWorkspace<T> {
    /// Return the active detector-frame identity.
    pub fn sequence(&self) -> Option<u64> {
        self.frame.sequence()
    }
    /// Return the number of completed scientific subapertures.
    pub const fn measured_subapertures(&self) -> usize {
        self.measured_subapertures
    }
    /// Clear numerical planes and abandon the detector frame, as in JFG readout sensing.
    pub fn reset(&mut self) {
        self.image.fill(T::zero());
        self.slopes.fill(T::zero());
        self.flux.fill(T::zero());
        self.validity.fill(false);
        self.abandon();
    }
    fn abandon(&mut self) {
        self.frame.abandon();
        self.measured_subapertures = 0;
    }
}

/// Invalid sample metadata or sensing call at the declaration boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannReadoutMeasurementBlockProcessError {
    /// Input row-block identity is missing.
    MissingSampleMetadata,
    /// The row offset cannot be represented on this target.
    OffsetOverflow,
    /// The sensing call failed.
    Algorithm(ShackHartmannReadoutMeasurementBlockError),
}
impl fmt::Display for ShackHartmannReadoutMeasurementBlockProcessError {
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
impl core::error::Error for ShackHartmannReadoutMeasurementBlockProcessError {}
impl From<ShackHartmannReadoutMeasurementBlockError>
    for ShackHartmannReadoutMeasurementBlockProcessError
{
    fn from(error: ShackHartmannReadoutMeasurementBlockError) -> Self {
        Self::Algorithm(error)
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 progressive Shack–Hartmann measurement-block declaration.
    pub ShackHartmannReadoutMeasurementBlockF32 {
        label = "shwfs-readout-measurement-block-f32";
        configuration ShackHartmannReadoutMeasurementBlockF32Configuration {
            detector_size: (usize, usize) = GraphRebuild;
            region_lines: usize = GraphRebuild;
            block_lines: usize = GraphRebuild;
            region_samples: usize = GraphRebuild;
            detector_origins: Vec<[usize; 2]> = GraphRebuild;
            detector_line_steps: Vec<[i32; 2]> = GraphRebuild;
            detector_sample_steps: Vec<[i32; 2]> = GraphRebuild;
            subaperture_size: (usize, usize) = GraphRebuild;
            subaperture_count: usize = GraphRebuild;
            initial_subaperture_origins: Option<Vec<[usize; 2]>> = Construction;
            coordinate_scale: f32 = Construction;
            pixel_threshold: f32 = Construction;
            flux_threshold: f32 = Construction;
            reference_slopes: Option<Vec<f32>> = Construction;
            active: Option<Vec<bool>> = Construction;
            measurements_schema: String = GraphRebuild;
            column_indices_schema: String = GraphRebuild;
            measurement_count_schema: String = GraphRebuild;
        }
        plan = ShackHartmannReadoutMeasurementBlockPlan<f32>;
        workspace = ShackHartmannReadoutMeasurementBlockWorkspace<f32>;
        properties = none;
        dimensions |plan,_workspace| {
            readout_region_count=plan.readout().regions().len();
            block_lines=plan.readout().block_lines();region_samples=plan.readout().region_samples();
            subaperture_count=plan.sensing().measurement().len();pixel_count=plan.sensing().measurement().pixel_count();slope_count=plan.capacity();
            block_rate_multiplier=u32::try_from(plan.readout().blocks()).map_err(|_|crate::DeclarationPreparationError::Format(crate::FormatError::Rate))?;
        }
        inputs {
            readout_block:f32 [readout_region_count,block_lines,region_samples]="readout-block",schema=crate::schemas::CALIBRATED_PIXEL_READOUT_BLOCK_V1,sample=readout_sample,rate=[block_rate_multiplier,1];
        }
        outputs {
            measurements:f32 [slope_count]="measurements",schema=configuration.measurements_schema.as_str(),metadata_from="readout-block",publication=conditional,rate=[block_rate_multiplier,1];
            columns:u32 [slope_count]="columns",schema=configuration.column_indices_schema.as_str(),metadata_from="readout-block",publication=conditional,rate=[block_rate_multiplier,1];
            count:u32 [1]="count",schema=configuration.measurement_count_schema.as_str(),metadata_from="readout-block",publication=conditional,rate=[block_rate_multiplier,1];
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
        }
        parameter_error = ShackHartmannReadoutMeasurementBlockPreparationError;
        prepare |configuration| {
            let readout=DetectorReadoutPlan::from_i32_coordinate_steps(
                configuration.detector_size,configuration.region_lines,configuration.block_lines,configuration.region_samples,
                &configuration.detector_origins,&configuration.detector_line_steps,&configuration.detector_sample_steps,
            )?;
            let origins = initial_subaperture_origins(
                configuration.detector_size,
                configuration.subaperture_size,
                configuration.subaperture_count,
                configuration.initial_subaperture_origins,
            )?;
            let sensing = ShackHartmannImagePlan::from_detector_regions(
                configuration.detector_size,
                configuration.subaperture_size,
                &origins, configuration.coordinate_scale, configuration.pixel_threshold,
                configuration.flux_threshold, configuration.reference_slopes, configuration.active,
            )?;
            let plan = ShackHartmannReadoutMeasurementBlockPlan::new(sensing, readout)?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = ShackHartmannReadoutMeasurementBlockPreparationError;
        process |plan, workspace| {
            let Some(sample)=readout_sample else {
                workspace.abandon();
                return Err(ShackHartmannReadoutMeasurementBlockProcessError::MissingSampleMetadata);
            };
            let ordinal = usize::try_from(sample.offset())
                .map_err(|_| ShackHartmannReadoutMeasurementBlockProcessError::OffsetOverflow)?;
            let result = plan.process_block(measurements, columns, count, workspace, readout_block,
                DetectorReadoutBlockInfo {
                    sequence: sample.sequence(), ordinal, terminal: sample.is_terminal(),
                    discontinuity: sample.is_discontinuous(), corrupted: sample.is_corrupted(),
                },
            )?;
            Ok(match result {
                ShackHartmannReadoutMeasurementBlockResult::Dropped(_) => AlgorithmProcessResult::deferred(),
                ShackHartmannReadoutMeasurementBlockResult::Published { discontinuity } => {
                    AlgorithmProcessResult::complete_with_metadata(
                        sample.with_discontinuity(discontinuity),
                    )
                }
            })
        }
        process_error = ShackHartmannReadoutMeasurementBlockProcessError;
        reset |_plan, workspace| { workspace.reset(); }
        plan_adopted |_plan, workspace| { workspace.abandon(); }
    }
}
