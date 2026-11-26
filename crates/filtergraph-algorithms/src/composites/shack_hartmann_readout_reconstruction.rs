//! Progressive Shack-Hartmann reconstruction over detector readout regions.

use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
use core::fmt;

use crate::{
    DetectorReadoutBlockInfo, DetectorReadoutPlan, DetectorReadoutProfileError,
    DetectorReadoutResult, MatrixElement, PreparedGemv, PreparedGemvError, PreparedProgressiveGemv,
    PreparedProgressiveGemvError, ProcessError, ProgressiveGemvError, ShackHartmannImagePlan,
    ShackHartmannImagePreparationError, ShackHartmannOutput, ShackHartmannOutputError,
};

use super::detector_readout::DetectorReadoutFrame;
use super::shack_hartmann_image::initial_subaperture_origins;

/// Invalid portable progressive Shack-Hartmann preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannReadoutReconstructionPreparationError {
    /// The detector readout profile is invalid.
    Readout(DetectorReadoutProfileError),
    /// Image-backed Shack-Hartmann preparation failed.
    Sensing(ShackHartmannImagePreparationError),
    /// Dense reconstructor preparation failed.
    Reconstructor(PreparedGemvError),
    /// The prepared operations are incompatible.
    Composition(ShackHartmannReadoutReconstructionPlanError),
}

impl fmt::Display for ShackHartmannReadoutReconstructionPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Readout(error) => error.fmt(formatter),
            Self::Sensing(error) => error.fmt(formatter),
            Self::Reconstructor(error) => error.fmt(formatter),
            Self::Composition(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for ShackHartmannReadoutReconstructionPreparationError {}

impl From<DetectorReadoutProfileError> for ShackHartmannReadoutReconstructionPreparationError {
    fn from(error: DetectorReadoutProfileError) -> Self {
        Self::Readout(error)
    }
}

impl From<ShackHartmannImagePreparationError>
    for ShackHartmannReadoutReconstructionPreparationError
{
    fn from(error: ShackHartmannImagePreparationError) -> Self {
        Self::Sensing(error)
    }
}

impl From<PreparedGemvError> for ShackHartmannReadoutReconstructionPreparationError {
    fn from(error: PreparedGemvError) -> Self {
        Self::Reconstructor(error)
    }
}

impl From<ShackHartmannReadoutReconstructionPlanError>
    for ShackHartmannReadoutReconstructionPreparationError
{
    fn from(error: ShackHartmannReadoutReconstructionPlanError) -> Self {
        Self::Composition(error)
    }
}

/// Invalid progressive detector-readout Shack-Hartmann preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannReadoutReconstructionPlanError {
    /// Detector readout and Shack-Hartmann image extents differ.
    ImageExtent {
        /// Prepared detector readout image size.
        readout: (usize, usize),
        /// Prepared Shack-Hartmann image size.
        sensing: (usize, usize),
    },
    /// The Shack-Hartmann image view is not a complete row-major detector image.
    ImageLayout,
    /// Pair-interleaved wavefront-slope extent arithmetic overflowed.
    SlopeExtentOverflow,
    /// Reconstructor storage extent arithmetic overflowed.
    ReconstructorExtentOverflow,
    /// Reconstructor columns do not equal the canonical slope extent.
    ReconstructorColumns {
        /// Required pair-interleaved slope extent.
        expected: usize,
        /// Supplied reconstructor column count.
        actual: usize,
    },
    /// Prepared scientific readiness matrix packing is invalid.
    Reconstructor(PreparedProgressiveGemvError),
}

impl fmt::Display for ShackHartmannReadoutReconstructionPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageExtent { readout, sensing } => write!(
                formatter,
                "detector readout image {readout:?} does not match Shack-Hartmann image {sensing:?}"
            ),
            Self::ImageLayout => formatter
                .write_str("Shack-Hartmann detector image must use complete row-major layout"),
            Self::SlopeExtentOverflow => {
                formatter.write_str("pair-interleaved wavefront-slope extent overflow")
            }
            Self::ReconstructorExtentOverflow => {
                formatter.write_str("Shack-Hartmann reconstructor extent overflow")
            }
            Self::ReconstructorColumns { expected, actual } => write!(
                formatter,
                "reconstructor column count {actual} does not match slope extent {expected}"
            ),
            Self::Reconstructor(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for ShackHartmannReadoutReconstructionPlanError {}

impl From<PreparedProgressiveGemvError> for ShackHartmannReadoutReconstructionPlanError {
    fn from(error: PreparedProgressiveGemvError) -> Self {
        Self::Reconstructor(error)
    }
}

/// Invalid progressive detector-readout Shack-Hartmann execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannReadoutReconstructionError {
    /// The input block did not carry transport-neutral sample identity.
    MissingSampleMetadata,
    /// The portable acquisition ordinal cannot be represented on this target.
    OrdinalOverflow,
    /// Caller-owned reconstructed output has the wrong extent.
    OutputExtent {
        /// Required reconstructed-vector extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// One detector readout block has the wrong extent.
    InputExtent {
        /// Required block extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// Private Shack-Hartmann output arrays are incompatible.
    ShackHartmannOutput(ShackHartmannOutputError),
    /// Shack-Hartmann measurement rejected the prepared detector workspace.
    ShackHartmann(ProcessError),
    /// Progressive reconstruction accumulation failed.
    Reconstruction(ProgressiveGemvError),
}

impl fmt::Display for ShackHartmannReadoutReconstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSampleMetadata => {
                formatter.write_str("detector readout block sample metadata is required")
            }
            Self::OrdinalOverflow => {
                formatter.write_str("detector readout acquisition ordinal does not fit usize")
            }
            Self::OutputExtent { expected, actual } => write!(
                formatter,
                "reconstructed output extent {actual} does not match {expected}"
            ),
            Self::InputExtent { expected, actual } => write!(
                formatter,
                "detector readout block extent {actual} does not match {expected}"
            ),
            Self::ShackHartmannOutput(error) => error.fmt(formatter),
            Self::ShackHartmann(error) => error.fmt(formatter),
            Self::Reconstruction(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for ShackHartmannReadoutReconstructionError {}

/// Prepared multi-region Shack-Hartmann measurement and reconstruction.
///
/// Detector acquisition order determines when work executes. Subaperture
/// declaration order continues to define pair-interleaved wavefront-slope and
/// canonical reconstructor-column identity.
#[derive(Clone, Debug)]
pub struct ShackHartmannReadoutReconstructionPlan<T> {
    sensing: ShackHartmannImagePlan<T>,
    readout: DetectorReadoutPlan,
    reconstructor: PreparedProgressiveGemv<T>,
    subaperture_completion_ordinals: Box<[usize]>,
}

impl<T> ShackHartmannReadoutReconstructionPlan<T>
where
    T: MatrixElement,
{
    /// Prepares scientific readiness and reconstructor panels.
    pub fn new(
        sensing: ShackHartmannImagePlan<T>,
        canonical_reconstructor: PreparedGemv<T>,
        readout: DetectorReadoutPlan,
    ) -> Result<Self, ShackHartmannReadoutReconstructionPlanError> {
        let regions = sensing.regions();
        if regions.image_size() != readout.detector_size() {
            return Err(ShackHartmannReadoutReconstructionPlanError::ImageExtent {
                readout: readout.detector_size(),
                sensing: regions.image_size(),
            });
        }
        if regions.row_stride() != regions.image_size().1 || regions.first_pixel() != 0 {
            return Err(ShackHartmannReadoutReconstructionPlanError::ImageLayout);
        }
        let slope_count = regions
            .len()
            .checked_mul(2)
            .ok_or(ShackHartmannReadoutReconstructionPlanError::SlopeExtentOverflow)?;
        if canonical_reconstructor.columns() != slope_count {
            return Err(
                ShackHartmannReadoutReconstructionPlanError::ReconstructorColumns {
                    expected: slope_count,
                    actual: canonical_reconstructor.columns(),
                },
            );
        }

        let (region_rows, region_columns) = regions.region_size();
        let mut completion_ordinals = Vec::with_capacity(regions.len());
        for &(first_row, first_column) in regions.origins() {
            let mut completion = 0;
            for row in first_row..first_row + region_rows {
                for column in first_column..first_column + region_columns {
                    completion = completion.max(
                        readout
                            .arrival_ordinal(row, column)
                            .expect("matching complete detector mapping"),
                    );
                }
            }
            completion_ordinals.push(completion);
        }
        let mut slope_completion_ordinals = Vec::with_capacity(slope_count);
        for &ordinal in &completion_ordinals {
            slope_completion_ordinals.extend_from_slice(&[ordinal, ordinal]);
        }
        let reconstructor = PreparedProgressiveGemv::new(
            canonical_reconstructor,
            &slope_completion_ordinals,
            readout.blocks(),
        )?;
        Ok(Self {
            sensing,
            readout,
            reconstructor,
            subaperture_completion_ordinals: completion_ordinals.into_boxed_slice(),
        })
    }

    /// Returns the canonical Shack-Hartmann scientific operation.
    pub const fn sensing(&self) -> &ShackHartmannImagePlan<T> {
        &self.sensing
    }

    /// Returns the prepared detector readout mapping.
    pub const fn readout(&self) -> &DetectorReadoutPlan {
        &self.readout
    }

    /// Returns the canonical matrix and its prepared acquisition groups.
    pub const fn reconstructor(&self) -> &PreparedProgressiveGemv<T> {
        &self.reconstructor
    }

    /// Returns each canonical subaperture's first complete acquisition ordinal.
    pub fn subaperture_completion_ordinals(&self) -> &[usize] {
        &self.subaperture_completion_ordinals
    }

    /// Allocates bounded mutable storage for one execution owner.
    pub fn workspace(&self) -> ShackHartmannReadoutReconstructionWorkspace<T> {
        let detector_elements = self.readout.detector_elements();
        let subapertures = self.sensing.measurement().len();
        ShackHartmannReadoutReconstructionWorkspace {
            detector: vec![T::zero(); detector_elements].into_boxed_slice(),
            slopes: vec![T::zero(); subapertures * 2].into_boxed_slice(),
            flux: vec![T::zero(); subapertures].into_boxed_slice(),
            validity: vec![false; subapertures].into_boxed_slice(),
            reconstructed: vec![T::zero(); self.reconstructor.canonical().rows()]
                .into_boxed_slice(),
            frame: DetectorReadoutFrame::new(),
            measured_subapertures: 0,
        }
    }

    /// Consumes one immutable detector readout block and conditionally publishes.
    pub fn process_block(
        &self,
        output: &mut [T],
        workspace: &mut ShackHartmannReadoutReconstructionWorkspace<T>,
        block: &[T],
        info: DetectorReadoutBlockInfo,
    ) -> Result<DetectorReadoutResult, ShackHartmannReadoutReconstructionError> {
        if output.len() != self.reconstructor.canonical().rows() {
            return Err(ShackHartmannReadoutReconstructionError::OutputExtent {
                expected: self.reconstructor.canonical().rows(),
                actual: output.len(),
            });
        }
        if block.len() != self.readout.block_elements() {
            return Err(ShackHartmannReadoutReconstructionError::InputExtent {
                expected: self.readout.block_elements(),
                actual: block.len(),
            });
        }
        if let Err(reason) = workspace.frame.accept(info, self.readout.blocks()) {
            workspace.clear_partial();
            return Ok(DetectorReadoutResult::Dropped(reason));
        }
        if info.ordinal == 0 {
            workspace.reconstructed.fill(T::zero());
            workspace.measured_subapertures = 0;
        }
        self.readout
            .scatter_block(&mut workspace.detector, block, info.ordinal)
            .expect("validated extents and admitted ordinal");

        let slope_indices = self
            .reconstructor
            .input_indices(info.ordinal)
            .expect("admitted ordinal");
        let mut pairs = slope_indices.chunks_exact(2);
        for pair in &mut pairs {
            let subaperture = pair[0] / 2;
            debug_assert_eq!(pair, [subaperture * 2, subaperture * 2 + 1]);
            let measurement_output = ShackHartmannOutput::new(
                &mut workspace.slopes,
                &mut workspace.flux,
                &mut workspace.validity,
            )
            .map_err(ShackHartmannReadoutReconstructionError::ShackHartmannOutput)?;
            if let Err(error) = self.sensing.process_image_range(
                measurement_output,
                &workspace.detector,
                subaperture..subaperture + 1,
            ) {
                workspace.abandon();
                return Err(ShackHartmannReadoutReconstructionError::ShackHartmann(
                    error,
                ));
            }
            workspace.measured_subapertures += 1;
        }
        debug_assert!(pairs.remainder().is_empty());
        if let Err(error) = self.reconstructor.accumulate_group(
            &mut workspace.reconstructed,
            &workspace.slopes,
            info.ordinal,
        ) {
            workspace.abandon();
            return Err(ShackHartmannReadoutReconstructionError::Reconstruction(
                error,
            ));
        }

        if !info.terminal {
            return Ok(DetectorReadoutResult::Pending);
        }
        debug_assert_eq!(
            workspace.measured_subapertures,
            self.sensing.measurement().len()
        );
        output.copy_from_slice(&workspace.reconstructed);
        let result = DetectorReadoutResult::Complete {
            sequence: info.sequence,
            discontinuity: workspace.frame.discontinuity(),
        };
        workspace.complete();
        Ok(result)
    }
}

/// Bounded mutable state for one progressive Shack-Hartmann execution owner.
#[derive(Debug)]
pub struct ShackHartmannReadoutReconstructionWorkspace<T> {
    detector: Box<[T]>,
    slopes: Box<[T]>,
    flux: Box<[T]>,
    validity: Box<[bool]>,
    reconstructed: Box<[T]>,
    frame: DetectorReadoutFrame,
    measured_subapertures: usize,
}

impl<T> ShackHartmannReadoutReconstructionWorkspace<T> {
    /// Returns the active detector-frame sequence, if any.
    pub const fn sequence(&self) -> Option<u64> {
        self.frame.sequence()
    }

    /// Returns how many subapertures have been measured in the current frame.
    pub const fn measured_subapertures(&self) -> usize {
        self.measured_subapertures
    }

    /// Returns the private progressive reconstructed-output accumulator.
    pub fn accumulated(&self) -> &[T] {
        &self.reconstructed
    }

    /// Abandons any partial detector frame.
    pub fn reset(&mut self) {
        self.abandon();
    }

    fn abandon(&mut self) {
        self.frame.abandon();
        self.clear_partial();
    }

    fn clear_partial(&mut self) {
        self.measured_subapertures = 0;
    }

    fn complete(&mut self) {
        self.frame.complete();
        self.clear_partial();
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 multi-region Shack-Hartmann reconstruction declaration.
    pub ShackHartmannReadoutReconstructorF32 {
        label = "shwfs-readout-reconstructor-f32";
        configuration ShackHartmannReadoutReconstructorF32Configuration {
            detector_size: (usize, usize) = GraphRebuild;
            region_lines: usize = GraphRebuild;
            block_lines: usize = GraphRebuild;
            region_samples: usize = GraphRebuild;
            detector_origins: Vec<[usize; 2]> = GraphRebuild;
            detector_line_steps: Vec<[i32; 2]> = GraphRebuild;
            detector_sample_steps: Vec<[i32; 2]> = GraphRebuild;
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
            reconstructed_schema: alloc::string::String = GraphRebuild;
        }
        plan = ShackHartmannReadoutReconstructionPlan<f32>;
        workspace = ShackHartmannReadoutReconstructionWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            readout_region_count = plan.readout().regions().len();
            block_lines = plan.readout().block_lines();
            region_samples = plan.readout().region_samples();
            reconstructed_count = plan.reconstructor().canonical().rows();
            block_rate_multiplier = u32::try_from(plan.readout().blocks())
                .map_err(|_| crate::DeclarationPreparationError::Format(crate::FormatError::Rate))?;
        }
        inputs {
            readout_block: f32 [readout_region_count, block_lines, region_samples]
                = "readout-block",
                schema = crate::schemas::CALIBRATED_PIXEL_READOUT_BLOCK_V1,
                sample = readout_sample,
                rate = [block_rate_multiplier, 1];
        }
        outputs {
            reconstructed: f32 [reconstructed_count] = "reconstructed",
                schema = configuration.reconstructed_schema.as_str(),
                metadata_from = "readout-block",
                publication = conditional;
        }
        prepare |configuration| {
            let readout = DetectorReadoutPlan::from_i32_coordinate_steps(
                configuration.detector_size,
                configuration.region_lines,
                configuration.block_lines,
                configuration.region_samples,
                &configuration.detector_origins,
                &configuration.detector_line_steps,
                &configuration.detector_sample_steps,
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
                &origins,
                configuration.coordinate_scale,
                configuration.pixel_threshold,
                configuration.flux_threshold,
                configuration.reference_slopes,
                configuration.active,
            )?;
            let slope_count = configuration.subaperture_count.checked_mul(2)
                .ok_or(ShackHartmannReadoutReconstructionPlanError::SlopeExtentOverflow)?;
            let elements = configuration.reconstructed_count.checked_mul(slope_count)
                .ok_or(ShackHartmannReadoutReconstructionPlanError::ReconstructorExtentOverflow)?;
            let values = configuration.initial_reconstructor
                .unwrap_or_else(|| vec![0.0; elements]);
            let plan = ShackHartmannReadoutReconstructionPlan::new(
                sensing,
                PreparedGemv::new(
                    configuration.reconstructed_count,
                    slope_count,
                    Arc::from(values),
                )?,
                readout,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = ShackHartmannReadoutReconstructionPreparationError;
        process |plan, workspace| {
            let Some(sample) = readout_sample else {
                workspace.abandon();
                return Err(ShackHartmannReadoutReconstructionError::MissingSampleMetadata);
            };
            let ordinal = usize::try_from(sample.offset())
                .map_err(|_| ShackHartmannReadoutReconstructionError::OrdinalOverflow)?;
            let result = plan.process_block(
                reconstructed,
                workspace,
                readout_block,
                DetectorReadoutBlockInfo {
                    sequence: sample.sequence(),
                    ordinal,
                    terminal: sample.is_terminal(),
                    discontinuity: sample.is_discontinuous(),
                    corrupted: sample.is_corrupted(),
                },
            )?;
            Ok(match result {
                DetectorReadoutResult::Pending | DetectorReadoutResult::Dropped(_) => {
                    crate::AlgorithmProcessResult::deferred()
                }
                DetectorReadoutResult::Complete { sequence: _, discontinuity } => {
                    crate::AlgorithmProcessResult::complete_with_metadata(
                        sample.with_offset(0)

                            .with_terminal(true)
                            .with_discontinuity(discontinuity).with_corrupted(false),
                    )
                }
            })
        }
        process_error = ShackHartmannReadoutReconstructionError;
        reset |_plan, workspace| { workspace.reset(); }
        plan_adopted |_plan, workspace| { workspace.reset(); }
    }
}
