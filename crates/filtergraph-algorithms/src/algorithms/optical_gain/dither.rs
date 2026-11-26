//! Prepared modal sinusoids for optical-gain calibration.

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::fmt;

use num_traits::{Float, FloatConst, NumCast};

use crate::AlgorithmPlan;

/// Minimum samples in one modal dither period.
pub const MIN_OPTICAL_GAIN_DITHER_FRAMES_PER_PERIOD: usize = 4;

/// Timing and commanded amplitude of one modal optical-gain dither.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpticalGainDitherMode<T> {
    amplitude: T,
    frames_per_period: usize,
    periods_per_estimate: usize,
}

impl<T> OpticalGainDitherMode<T> {
    /// Describes one modal sinusoid.
    ///
    /// Validation belongs to [`OpticalGainDitherSchedule::new`] so errors can
    /// identify the mode within the complete synchronized schedule.
    pub const fn new(amplitude: T, frames_per_period: usize, periods_per_estimate: usize) -> Self {
        Self {
            amplitude,
            frames_per_period,
            periods_per_estimate,
        }
    }

    /// Returns the positive commanded modal amplitude.
    pub fn amplitude(&self) -> T
    where
        T: Copy,
    {
        self.amplitude
    }

    /// Returns the number of frames in one sinusoid period.
    pub const fn frames_per_period(&self) -> usize {
        self.frames_per_period
    }

    /// Returns the number of complete periods in one estimate window.
    pub const fn periods_per_estimate(&self) -> usize {
        self.periods_per_estimate
    }
}

/// Invalid modal optical-gain dither schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainDitherScheduleError {
    /// At least one dither mode is required.
    Empty,

    /// A mode has a non-finite or nonpositive commanded amplitude.
    InvalidAmplitude {
        /// Zero-based mode index.
        mode: usize,
    },

    /// A mode has fewer than four frames in one period.
    PeriodTooShort {
        /// Zero-based mode index.
        mode: usize,
        /// Supplied frames per period.
        frames: usize,
    },

    /// A mode selects no complete periods in one estimate.
    ZeroPeriodsPerEstimate {
        /// Zero-based mode index.
        mode: usize,
    },

    /// A mode's estimate-window extent cannot be represented by `usize`.
    EstimateWindowOverflow {
        /// Zero-based mode index.
        mode: usize,
    },

    /// The modes do not share one synchronous estimate-window extent.
    EstimateWindowMismatch {
        /// Zero-based mode index.
        mode: usize,
        /// Estimate-window extent established by the first mode.
        expected: usize,
        /// Estimate-window extent supplied by this mode.
        actual: usize,
    },

    /// The combined prepared quadrature-table extent overflows `usize`.
    QuadratureTableOverflow,

    /// A mode's phase or period cannot be represented by the scalar type.
    ScalarNotRepresentable {
        /// Zero-based mode index.
        mode: usize,
    },

    /// Preparing a modal quadrature produced a non-finite value.
    NonFiniteQuadrature {
        /// Zero-based mode index.
        mode: usize,
    },
}

impl fmt::Display for OpticalGainDitherScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("optical-gain dither schedule must be nonempty"),
            Self::InvalidAmplitude { mode } => write!(
                formatter,
                "optical-gain dither mode {mode} amplitude must be finite and positive"
            ),
            Self::PeriodTooShort { mode, frames } => write!(
                formatter,
                "optical-gain dither mode {mode} has {frames} frames per period, expected at least {MIN_OPTICAL_GAIN_DITHER_FRAMES_PER_PERIOD}"
            ),
            Self::ZeroPeriodsPerEstimate { mode } => write!(
                formatter,
                "optical-gain dither mode {mode} must complete at least one period per estimate"
            ),
            Self::EstimateWindowOverflow { mode } => write!(
                formatter,
                "optical-gain dither mode {mode} estimate-window extent overflows usize"
            ),
            Self::EstimateWindowMismatch {
                mode,
                expected,
                actual,
            } => write!(
                formatter,
                "optical-gain dither mode {mode} estimate window {actual} does not match {expected}"
            ),
            Self::QuadratureTableOverflow => {
                formatter.write_str("optical-gain quadrature-table extent overflows usize")
            }
            Self::ScalarNotRepresentable { mode } => write!(
                formatter,
                "optical-gain dither mode {mode} timing is not representable by the scalar type"
            ),
            Self::NonFiniteQuadrature { mode } => write!(
                formatter,
                "optical-gain dither mode {mode} produced a non-finite prepared quadrature"
            ),
        }
    }
}

impl core::error::Error for OpticalGainDitherScheduleError {}

/// Immutable synchronized modal dither timing and prepared quadratures.
///
/// The generator and estimator may share this artifact. Preparation verifies
/// that every mode spans the same estimation window and evaluates all sine
/// and cosine references outside warmed execution.
#[derive(Debug, PartialEq)]
pub struct OpticalGainDitherSchedule<T> {
    modes: Box<[OpticalGainDitherMode<T>]>,
    offsets: Box<[usize]>,
    cosines: Box<[T]>,
    sines: Box<[T]>,
    frames_per_estimate: usize,
}

impl<T> OpticalGainDitherSchedule<T>
where
    T: Float + FloatConst + NumCast,
{
    /// Validates timing and prepares the modal quadrature tables.
    pub fn new(
        modes: impl Into<Box<[OpticalGainDitherMode<T>]>>,
    ) -> Result<Self, OpticalGainDitherScheduleError> {
        let modes = modes.into();
        if modes.is_empty() {
            return Err(OpticalGainDitherScheduleError::Empty);
        }

        let mut frames_per_estimate = None;
        let mut table_len = 0_usize;
        for (mode_index, mode) in modes.iter().enumerate() {
            if !mode.amplitude.is_finite() || mode.amplitude <= T::zero() {
                return Err(OpticalGainDitherScheduleError::InvalidAmplitude { mode: mode_index });
            }
            if mode.frames_per_period < MIN_OPTICAL_GAIN_DITHER_FRAMES_PER_PERIOD {
                return Err(OpticalGainDitherScheduleError::PeriodTooShort {
                    mode: mode_index,
                    frames: mode.frames_per_period,
                });
            }
            if mode.periods_per_estimate == 0 {
                return Err(OpticalGainDitherScheduleError::ZeroPeriodsPerEstimate {
                    mode: mode_index,
                });
            }
            let mode_window = mode
                .frames_per_period
                .checked_mul(mode.periods_per_estimate)
                .ok_or(OpticalGainDitherScheduleError::EstimateWindowOverflow {
                    mode: mode_index,
                })?;
            match frames_per_estimate {
                Some(expected) if mode_window != expected => {
                    return Err(OpticalGainDitherScheduleError::EstimateWindowMismatch {
                        mode: mode_index,
                        expected,
                        actual: mode_window,
                    });
                }
                None => frames_per_estimate = Some(mode_window),
                _ => {}
            }
            table_len = table_len
                .checked_add(mode.frames_per_period)
                .ok_or(OpticalGainDitherScheduleError::QuadratureTableOverflow)?;
        }

        let mut offsets = Vec::with_capacity(modes.len());
        let mut cosines = Vec::with_capacity(table_len);
        let mut sines = Vec::with_capacity(table_len);
        let two_pi = T::PI() + T::PI();
        for (mode_index, mode) in modes.iter().enumerate() {
            offsets.push(cosines.len());
            let period = <T as NumCast>::from(mode.frames_per_period).ok_or(
                OpticalGainDitherScheduleError::ScalarNotRepresentable { mode: mode_index },
            )?;
            for phase_index in 0..mode.frames_per_period {
                let phase = <T as NumCast>::from(phase_index).ok_or(
                    OpticalGainDitherScheduleError::ScalarNotRepresentable { mode: mode_index },
                )?;
                let angle = two_pi * phase / period;
                let cosine = angle.cos();
                let sine = angle.sin();
                if !cosine.is_finite() || !sine.is_finite() {
                    return Err(OpticalGainDitherScheduleError::NonFiniteQuadrature {
                        mode: mode_index,
                    });
                }
                cosines.push(cosine);
                sines.push(sine);
            }
        }

        Ok(Self {
            modes,
            offsets: offsets.into_boxed_slice(),
            cosines: cosines.into_boxed_slice(),
            sines: sines.into_boxed_slice(),
            frames_per_estimate: frames_per_estimate.expect("nonempty modes establish a window"),
        })
    }
}

impl<T> OpticalGainDitherSchedule<T> {
    /// Returns the configured mode descriptors.
    pub fn modes(&self) -> &[OpticalGainDitherMode<T>] {
        &self.modes
    }

    /// Returns the number of synchronized modal dithers.
    pub fn len(&self) -> usize {
        self.modes.len()
    }

    /// Returns whether the schedule contains no modes.
    pub fn is_empty(&self) -> bool {
        self.modes.is_empty()
    }

    /// Returns the common number of frames in one estimate window.
    pub const fn frames_per_estimate(&self) -> usize {
        self.frames_per_estimate
    }

    /// Returns the prepared cosine quadrature for one mode period.
    pub fn cosines(&self, mode: usize) -> Option<&[T]> {
        let descriptor = self.modes.get(mode)?;
        let start = *self.offsets.get(mode)?;
        Some(&self.cosines[start..start + descriptor.frames_per_period])
    }

    /// Returns the prepared sine quadrature for one mode period.
    pub fn sines(&self, mode: usize) -> Option<&[T]> {
        let descriptor = self.modes.get(mode)?;
        let start = *self.offsets.get(mode)?;
        Some(&self.sines[start..start + descriptor.frames_per_period])
    }

    pub(crate) fn cosine(&self, mode: usize, phase: usize) -> T
    where
        T: Copy,
    {
        self.cosines[self.offsets[mode] + phase]
    }

    pub(crate) fn sine(&self, mode: usize, phase: usize) -> T
    where
        T: Copy,
    {
        self.sines[self.offsets[mode] + phase]
    }
}

/// Invalid modal dither output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainDitherError {
    /// The caller-owned output has the wrong modal extent.
    OutputExtent {
        /// Required modal extent.
        expected: usize,
        /// Supplied modal extent.
        actual: usize,
    },
    /// The workspace was prepared for an incompatible modal extent.
    WorkspaceExtent,
    /// A committed phase is outside the corresponding prepared period.
    WorkspaceState,
    /// The zero-based estimate-window frame cannot be represented by `u64`.
    WindowFrameOutOfRange,
    /// A zero-based modal phase cannot be represented by `u64`.
    PhaseIndexOutOfRange {
        /// Zero-based dither-mode index.
        mode: usize,
    },
}

impl fmt::Display for OpticalGainDitherError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputExtent { expected, actual } => write!(
                formatter,
                "optical-gain dither output extent {actual} does not match {expected}"
            ),
            Self::WorkspaceExtent => {
                formatter.write_str("optical-gain dither workspace extent is invalid")
            }
            Self::WorkspaceState => {
                formatter.write_str("optical-gain dither workspace phase is invalid")
            }
            Self::WindowFrameOutOfRange => {
                formatter.write_str("optical-gain dither window frame is not representable by u64")
            }
            Self::PhaseIndexOutOfRange { mode } => write!(
                formatter,
                "optical-gain dither phase for mode {mode} is not representable by u64"
            ),
        }
    }
}

impl core::error::Error for OpticalGainDitherError {}

/// Committed modal phases for optical-gain dither generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpticalGainDitherWorkspace {
    phases: Box<[usize]>,
    frame_count: usize,
}

impl OpticalGainDitherWorkspace {
    /// Returns the zero-based phase index committed for the next call.
    pub fn phases(&self) -> &[usize] {
        &self.phases
    }

    /// Returns the zero-based frame within the current estimate window.
    pub const fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Restarts every modal phase and the common estimate window at zero.
    pub fn reset(&mut self) {
        self.phases.fill(0);
        self.frame_count = 0;
    }
}

/// Prepared generator for commanded modal optical-gain sinusoids.
///
/// This plan emits one complete modal vector per [`AlgorithmPlan::process`]
/// call. It is unrelated to PDM quantization dither.
#[derive(Clone, Debug)]
pub struct OpticalGainDitherPlan<T> {
    schedule: Arc<OpticalGainDitherSchedule<T>>,
}

impl<T> OpticalGainDitherPlan<T> {
    /// Uses one validated schedule shared with an estimator when desired.
    pub fn new(schedule: impl Into<Arc<OpticalGainDitherSchedule<T>>>) -> Self {
        Self {
            schedule: schedule.into(),
        }
    }

    /// Returns the modal extent.
    pub fn len(&self) -> usize {
        self.schedule.len()
    }

    /// Returns whether the modal extent is zero.
    pub fn is_empty(&self) -> bool {
        self.schedule.is_empty()
    }

    /// Returns shared ownership of the immutable dither schedule.
    pub fn shared_schedule(&self) -> Arc<OpticalGainDitherSchedule<T>> {
        Arc::clone(&self.schedule)
    }

    /// Creates phase-zero execution history.
    pub fn workspace(&self) -> OpticalGainDitherWorkspace {
        OpticalGainDitherWorkspace {
            phases: alloc::vec![0; self.len()].into_boxed_slice(),
            frame_count: 0,
        }
    }

    /// Returns the common number of frames in one estimate window.
    pub fn frames_per_estimate(&self) -> usize {
        self.schedule.frames_per_estimate()
    }
}

impl<T> AlgorithmPlan for OpticalGainDitherPlan<T>
where
    T: Float + 'static,
{
    type Input<'a> = ();
    type Output<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type Workspace = OpticalGainDitherWorkspace;
    type Error = OpticalGainDitherError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        (): Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if output.len() != self.len() {
            return Err(OpticalGainDitherError::OutputExtent {
                expected: self.len(),
                actual: output.len(),
            });
        }
        if workspace.phases.len() != self.len() {
            return Err(OpticalGainDitherError::WorkspaceExtent);
        }
        if workspace.frame_count >= self.schedule.frames_per_estimate()
            || self
                .schedule
                .modes
                .iter()
                .zip(workspace.phases.iter())
                .any(|(mode, phase)| *phase >= mode.frames_per_period)
        {
            return Err(OpticalGainDitherError::WorkspaceState);
        }

        for (mode_index, destination) in output.iter_mut().enumerate() {
            let mode = &self.schedule.modes[mode_index];
            *destination = mode.amplitude
                * self
                    .schedule
                    .cosine(mode_index, workspace.phases[mode_index]);
        }
        for (mode, phase) in self.schedule.modes.iter().zip(workspace.phases.iter_mut()) {
            *phase += 1;
            if *phase == mode.frames_per_period {
                *phase = 0;
            }
        }
        workspace.frame_count += 1;
        if workspace.frame_count == self.schedule.frames_per_estimate() {
            workspace.frame_count = 0;
        }
        Ok(())
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 source of synchronized modal optical-gain dithers.
    pub OpticalGainDitherF32 {
        label = "optical-gain-dither-f32";
        configuration OpticalGainDitherF32Configuration {
            modes: Vec<OpticalGainDitherMode<f32>> = GraphRebuild;
        }
        plan = OpticalGainDitherPlan<f32>;
        workspace = OpticalGainDitherWorkspace;
        properties = none;
        dimensions |plan, _workspace| {
            mode_count = plan.len();
        }
        inputs {}
        outputs {
            commanded_modes: f32 [mode_count] = "commanded-modes",
                schema = crate::schemas::OPTICAL_GAIN_DITHER_COMMAND_V1;
            window_frame: u64 [1] = "window-frame",
                schema = crate::schemas::OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1;
            phase_indices: u64 [mode_count] = "phase-indices",
                schema = crate::schemas::OPTICAL_GAIN_DITHER_PHASE_INDICES_V1;
            completes_window: bool [1] = "completes-window",
                schema = crate::schemas::OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1;
        }
        prepare |configuration| {
            let schedule = OpticalGainDitherSchedule::new(configuration.modes)?;
            let plan = OpticalGainDitherPlan::new(schedule);
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = OpticalGainDitherScheduleError;
        process |plan, workspace| {
            let current_frame = workspace.frame_count();
            let published_frame = u64::try_from(current_frame)
                .map_err(|_| OpticalGainDitherError::WindowFrameOutOfRange)?;
            let completes = current_frame + 1 == plan.frames_per_estimate();
            for (mode, (destination, phase)) in phase_indices
                .iter_mut()
                .zip(workspace.phases().iter())
                .enumerate()
            {
                *destination = u64::try_from(*phase)
                    .map_err(|_| OpticalGainDitherError::PhaseIndexOutOfRange { mode })?;
            }
            plan.process(commanded_modes, workspace, ())?;
            window_frame[0] = published_frame;
            completes_window[0] = completes;
            Ok(())
        }
        process_error = OpticalGainDitherError;
        reset |_plan, workspace| {
            workspace.reset();
        }
    }
}
