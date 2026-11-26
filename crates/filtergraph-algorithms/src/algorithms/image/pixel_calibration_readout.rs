//! Pixel calibration over acquisition-ordered detector readout blocks.

use core::{fmt, marker::PhantomData};

use num_traits::{AsPrimitive, Float};

use crate::{
    DetectorReadoutBlockInfo, DetectorReadoutDataError, DetectorReadoutPlan,
    DetectorReadoutProfileError, PixelCalibrationError, PixelCalibrationPlan, ProcessError,
};

/// Invalid portable U16-to-F32 detector-readout calibration preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelCalibrationReadoutU16F32PreparationError {
    /// The portable detector-readout profile is invalid.
    Profile(DetectorReadoutProfileError),
    /// Detector calibration and prepared readout mapping are incompatible.
    Calibration(DetectorReadoutCalibrationError),
}

impl fmt::Display for PixelCalibrationReadoutU16F32PreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile(error) => error.fmt(formatter),
            Self::Calibration(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PixelCalibrationReadoutU16F32PreparationError {}

impl From<DetectorReadoutProfileError> for PixelCalibrationReadoutU16F32PreparationError {
    fn from(error: DetectorReadoutProfileError) -> Self {
        Self::Profile(error)
    }
}

impl From<DetectorReadoutCalibrationError> for PixelCalibrationReadoutU16F32PreparationError {
    fn from(error: DetectorReadoutCalibrationError) -> Self {
        Self::Calibration(error)
    }
}

/// Invalid prepared detector-readout pixel calibration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectorReadoutCalibrationError {
    /// A complete detector calibration plane is incompatible with the mapping.
    Readout(DetectorReadoutDataError),
    /// Acquisition-ordered flat or background calibration is invalid.
    Calibration(PixelCalibrationError),
}

impl fmt::Display for DetectorReadoutCalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Readout(error) => error.fmt(formatter),
            Self::Calibration(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for DetectorReadoutCalibrationError {}

impl From<DetectorReadoutDataError> for DetectorReadoutCalibrationError {
    fn from(error: DetectorReadoutDataError) -> Self {
        Self::Readout(error)
    }
}

impl From<PixelCalibrationError> for DetectorReadoutCalibrationError {
    fn from(error: PixelCalibrationError) -> Self {
        Self::Calibration(error)
    }
}

/// Invalid detector-readout pixel-calibration input or output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectorReadoutCalibrationProcessError {
    /// The input block did not carry transport-neutral sample identity.
    MissingSampleMetadata,
    /// The portable acquisition ordinal cannot be represented on this target.
    OrdinalOverflow,
    /// The acquisition ordinal is outside the prepared detector frame.
    OrdinalRange {
        /// Supplied acquisition ordinal.
        ordinal: usize,
        /// Prepared block count.
        blocks: usize,
    },
    /// The terminal marker does not agree with the final acquisition ordinal.
    Terminal,
    /// The pixel-calibration operation rejected an array extent or value.
    Process(ProcessError),
}

/// Bounded publication state for detector-readout calibration adoption.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DetectorReadoutCalibrationWorkspace {
    pending_discontinuity: bool,
}

impl DetectorReadoutCalibrationWorkspace {
    /// Creates calibration publication state without a pending discontinuity.
    pub const fn new() -> Self {
        Self {
            pending_discontinuity: false,
        }
    }

    /// Marks the next calibrated block as following a plan discontinuity.
    pub fn mark_discontinuity(&mut self) {
        self.pending_discontinuity = true;
    }

    fn take_discontinuity(&mut self) -> bool {
        core::mem::take(&mut self.pending_discontinuity)
    }
}

impl fmt::Display for DetectorReadoutCalibrationProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSampleMetadata => {
                formatter.write_str("detector readout block sample metadata is required")
            }
            Self::OrdinalOverflow => {
                formatter.write_str("detector readout acquisition ordinal does not fit usize")
            }
            Self::OrdinalRange { ordinal, blocks } => write!(
                formatter,
                "acquisition ordinal {ordinal} is outside {blocks} prepared blocks"
            ),
            Self::Terminal => formatter.write_str(
                "detector readout terminal marker does not agree with the prepared frame",
            ),
            Self::Process(error) => error.fmt(formatter),
        }
    }
}

crate::filtergraph_algorithm! {
    /// Portable U16-to-F32 pixel calibration in detector acquisition order.
    pub PixelCalibrationReadoutU16F32 {
        label = "pixel-calibration-readout-u16-f32";
        configuration PixelCalibrationReadoutU16F32Configuration {
            detector_size: (usize, usize) = GraphRebuild;
            region_lines: usize = GraphRebuild;
            block_lines: usize = GraphRebuild;
            region_samples: usize = GraphRebuild;
            detector_origins: alloc::vec::Vec<[usize; 2]> = GraphRebuild;
            detector_line_steps: alloc::vec::Vec<[i32; 2]> = GraphRebuild;
            detector_sample_steps: alloc::vec::Vec<[i32; 2]> = GraphRebuild;
            initial_flat: Option<alloc::vec::Vec<f32>> = Construction;
            initial_background: Option<alloc::vec::Vec<f32>> = Construction;
        }
        plan = DetectorReadoutCalibrationPlan<f32, u16>;
        workspace = DetectorReadoutCalibrationWorkspace;
        properties = none;
        dimensions |plan, _workspace| {
            detector_rows = plan.readout().detector_size().0;
            detector_columns = plan.readout().detector_size().1;
            readout_region_count = plan.readout().regions().len();
            block_lines = plan.readout().block_lines();
            region_samples = plan.readout().region_samples();
            block_rate_multiplier = u32::try_from(plan.readout().blocks())
                .map_err(|_| crate::DeclarationPreparationError::Format(crate::FormatError::Rate))?;
        }
        inputs {
            raw: u16 [readout_region_count, block_lines, region_samples] = "raw",
                schema = crate::schemas::RAW_PIXEL_READOUT_BLOCK_V1,
                sample = raw_sample,
                rate = [block_rate_multiplier, 1];
        }
        outputs {
            calibrated: f32 [readout_region_count, block_lines, region_samples] = "calibrated",
                schema = crate::schemas::CALIBRATED_PIXEL_READOUT_BLOCK_V1,
                metadata_from = "raw",
                rate = [block_rate_multiplier, 1];
        }
        parameters {
            flat: f32 [detector_rows, detector_columns] = "flat",
                schema = crate::schemas::FLAT_CALIBRATION_V1,
                replace |plan, values| {
                    plan.with_flat(values)
                };
            background: f32 [detector_rows, detector_columns] = "background",
                schema = crate::schemas::BACKGROUND_CALIBRATION_V1,
                replace |plan, values| {
                    plan.with_background(values)
                };
        }
        parameter_error = DetectorReadoutCalibrationError;
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
            let detector_elements = readout.detector_elements();
            let flat = configuration
                .initial_flat
                .unwrap_or_else(|| alloc::vec![1.0; detector_elements]);
            let background = configuration
                .initial_background
                .unwrap_or_else(|| alloc::vec![0.0; detector_elements]);
            Ok((
                DetectorReadoutCalibrationPlan::new(readout, &flat, &background)?,
                DetectorReadoutCalibrationWorkspace::new(),
            ))
        }
        prepare_error = PixelCalibrationReadoutU16F32PreparationError;
        process |plan, workspace| {
            let sample = raw_sample
                .ok_or(DetectorReadoutCalibrationProcessError::MissingSampleMetadata)?;
            let ordinal = usize::try_from(sample.offset())
                .map_err(|_| DetectorReadoutCalibrationProcessError::OrdinalOverflow)?;
            plan.process_block(
                calibrated,
                raw,
                DetectorReadoutBlockInfo {
                    sequence: sample.sequence(),
                    ordinal,
                    terminal: sample.is_terminal(),
                    discontinuity: sample.is_discontinuous(),
                    corrupted: sample.is_corrupted(),
                },
            )?;
            let pending_discontinuity = workspace.take_discontinuity();
            let discontinuity = sample.is_discontinuous() || pending_discontinuity;
            Ok(crate::AlgorithmProcessResult::complete_with_metadata(
                sample.with_discontinuity(discontinuity),
            ))
        }
        process_error = DetectorReadoutCalibrationProcessError;
        reset |_plan, workspace| { workspace.mark_discontinuity(); }
        plan_adopted |_plan, workspace| { workspace.mark_discontinuity(); }
    }
}

impl core::error::Error for DetectorReadoutCalibrationProcessError {}

impl From<ProcessError> for DetectorReadoutCalibrationProcessError {
    fn from(error: ProcessError) -> Self {
        Self::Process(error)
    }
}

/// Prepared pixel calibration in detector acquisition order.
///
/// Callers provide complete detector-plane flat and background arrays in
/// detector-coordinate row-major order. Preparation packs both arrays into
/// readout order. Repeated block processing is therefore one contiguous
/// calibration window and does not evaluate detector coordinates.
#[derive(Clone, Debug)]
pub struct DetectorReadoutCalibrationPlan<T, R> {
    readout: DetectorReadoutPlan,
    calibration: PixelCalibrationPlan<T, R>,
    raw: PhantomData<fn(R)>,
}

impl<T, R> DetectorReadoutCalibrationPlan<T, R>
where
    T: Float,
{
    /// Prepares acquisition-ordered calibration from complete detector planes.
    pub fn new(
        readout: DetectorReadoutPlan,
        flat: &[T],
        background: &[T],
    ) -> Result<Self, DetectorReadoutCalibrationError> {
        let flat = readout.pack_detector(flat)?;
        let background = readout.pack_detector(background)?;
        Ok(Self {
            readout,
            calibration: PixelCalibrationPlan::new(flat, background)?,
            raw: PhantomData,
        })
    }

    /// Returns the prepared detector readout mapping.
    pub const fn readout(&self) -> &DetectorReadoutPlan {
        &self.readout
    }

    /// Returns flat calibration in complete acquisition order.
    pub fn packed_flat(&self) -> &[T] {
        self.calibration.flat()
    }

    /// Returns background calibration in complete acquisition order.
    pub fn packed_background(&self) -> &[T] {
        self.calibration.background()
    }

    /// Replaces the complete detector-coordinate flat calibration.
    pub fn with_flat(&self, flat: &[T]) -> Result<Self, DetectorReadoutCalibrationError> {
        let flat = self.readout.pack_detector(flat)?;
        Ok(Self {
            readout: self.readout.clone(),
            calibration: self.calibration.with_flat(&flat)?,
            raw: PhantomData,
        })
    }

    /// Replaces the complete detector-coordinate background calibration.
    pub fn with_background(
        &self,
        background: &[T],
    ) -> Result<Self, DetectorReadoutCalibrationError> {
        let background = self.readout.pack_detector(background)?;
        Ok(Self {
            readout: self.readout.clone(),
            calibration: self.calibration.with_background(&background)?,
            raw: PhantomData,
        })
    }

    /// Calibrates one immutable `[region, line, sample]` block.
    pub fn process_block(
        &self,
        output: &mut [T],
        raw: &[R],
        info: DetectorReadoutBlockInfo,
    ) -> Result<(), DetectorReadoutCalibrationProcessError>
    where
        T: 'static,
        R: Copy + AsPrimitive<T> + 'static,
    {
        if info.ordinal >= self.readout.blocks() {
            return Err(DetectorReadoutCalibrationProcessError::OrdinalRange {
                ordinal: info.ordinal,
                blocks: self.readout.blocks(),
            });
        }
        if info.terminal != (info.ordinal + 1 == self.readout.blocks()) {
            return Err(DetectorReadoutCalibrationProcessError::Terminal);
        }
        let start = info.ordinal * self.readout.block_elements();
        let end = start + self.readout.block_elements();
        self.calibration
            .process_window(output, raw, start..end)
            .map_err(Into::into)
    }
}
