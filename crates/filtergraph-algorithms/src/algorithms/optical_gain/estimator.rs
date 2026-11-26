//! Synchronous modal optical-gain estimation.

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::fmt;

use num_traits::{Float, NumCast};

use crate::{
    AlgorithmPlan, OpticalGainDitherMode, OpticalGainDitherSchedule, OpticalGainDitherScheduleError,
};

/// Invalid prepared optical-gain estimator configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainEstimatorPlanError {
    /// The calibration-response vector has the wrong modal extent.
    CalibrationResponseExtent {
        /// Required modal extent.
        expected: usize,
        /// Supplied modal extent.
        actual: usize,
    },

    /// A calibration response is non-finite or nonpositive.
    InvalidCalibrationResponse {
        /// Zero-based mode index.
        mode: usize,
    },

    /// The estimate-window extent cannot be represented by the scalar type.
    EstimateWindowNotRepresentable,
}

impl fmt::Display for OpticalGainEstimatorPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CalibrationResponseExtent { expected, actual } => write!(
                formatter,
                "optical-gain calibration-response extent {actual} does not match {expected}"
            ),
            Self::InvalidCalibrationResponse { mode } => write!(
                formatter,
                "optical-gain calibration response for mode {mode} must be finite and positive"
            ),
            Self::EstimateWindowNotRepresentable => formatter.write_str(
                "optical-gain estimate-window extent is not representable by the scalar type",
            ),
        }
    }
}

impl core::error::Error for OpticalGainEstimatorPlanError {}

/// Modal input stream named by an optical-gain execution error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainEstimatorInputKind {
    /// Realized modal dither command.
    RealizedCommand,
    /// Raw, uncorrected reconstructed modal sensor response.
    ReconstructedResponse,
}

/// Caller-owned estimate field named by an extent error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainEstimateField {
    /// Demodulated realized-command amplitude.
    CommandAmplitude,
    /// Demodulated reconstructed-response amplitude.
    ResponseAmplitude,
    /// Physical optical gain.
    OpticalGain,
    /// Reconstructed-response phase lag relative to realized command.
    PhaseLag,
    /// Per-mode estimate validity.
    Validity,
}

/// Invalid optical-gain estimator input, output, or numerical state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainEstimatorError {
    /// The workspace was prepared for an incompatible modal extent.
    WorkspaceExtent,

    /// The committed frame count is outside the prepared estimate window.
    WorkspaceState,

    /// A supplied window frame cannot be represented by `usize`.
    WindowFrameOutOfRange,

    /// The supplied dither window frame is not the next estimator frame.
    WindowFrameMismatch {
        /// Required zero-based frame.
        expected: usize,
        /// Supplied zero-based frame.
        actual: usize,
    },

    /// The supplied phase-index vector has the wrong modal extent.
    PhaseIndexExtent {
        /// Required modal extent.
        expected: usize,
        /// Supplied modal extent.
        actual: usize,
    },

    /// A prepared phase cannot be represented by `u64`.
    PhaseIndexOutOfRange {
        /// Zero-based dither-mode index.
        mode: usize,
    },

    /// One supplied modal phase disagrees with the prepared schedule.
    PhaseIndexMismatch {
        /// Zero-based dither-mode index.
        mode: usize,
        /// Required zero-based phase.
        expected: u64,
        /// Supplied zero-based phase.
        actual: u64,
    },

    /// The supplied completion flag disagrees with the prepared schedule.
    WindowCompletionMismatch {
        /// Required completion state.
        expected: bool,
        /// Supplied completion state.
        actual: bool,
    },

    /// One input stream has the wrong modal extent.
    InputExtent {
        /// Input stream with the wrong extent.
        input: OpticalGainEstimatorInputKind,
        /// Required modal extent.
        expected: usize,
        /// Supplied modal extent.
        actual: usize,
    },

    /// One output field has the wrong modal extent.
    OutputExtent {
        /// Output field with the wrong extent.
        field: OpticalGainEstimateField,
        /// Required modal extent.
        expected: usize,
        /// Supplied modal extent.
        actual: usize,
    },

    /// An input sample is non-finite.
    NonFiniteInput {
        /// Input stream containing the sample.
        input: OpticalGainEstimatorInputKind,
        /// Zero-based mode index.
        mode: usize,
    },

    /// Updating synchronous quadratures produced a non-finite accumulator.
    NonFiniteAccumulation {
        /// Zero-based mode index.
        mode: usize,
    },

    /// Window completion produced a non-finite amplitude or phase.
    NonFiniteEstimate {
        /// Zero-based mode index.
        mode: usize,
    },
}

impl fmt::Display for OpticalGainEstimatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceExtent => {
                formatter.write_str("optical-gain estimator workspace extent is invalid")
            }
            Self::WorkspaceState => {
                formatter.write_str("optical-gain estimator workspace frame count is invalid")
            }
            Self::WindowFrameOutOfRange => formatter
                .write_str("optical-gain estimator window frame is not representable by usize"),
            Self::WindowFrameMismatch { expected, actual } => write!(
                formatter,
                "optical-gain estimator window frame {actual} does not match {expected}"
            ),
            Self::PhaseIndexExtent { expected, actual } => write!(
                formatter,
                "optical-gain phase-index extent {actual} does not match {expected}"
            ),
            Self::PhaseIndexOutOfRange { mode } => write!(
                formatter,
                "optical-gain prepared phase for mode {mode} is not representable by u64"
            ),
            Self::PhaseIndexMismatch {
                mode,
                expected,
                actual,
            } => write!(
                formatter,
                "optical-gain phase {actual} for mode {mode} does not match {expected}"
            ),
            Self::WindowCompletionMismatch { expected, actual } => write!(
                formatter,
                "optical-gain estimator completion flag {actual} does not match {expected}"
            ),
            Self::InputExtent {
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "optical-gain {input:?} input extent {actual} does not match {expected}"
            ),
            Self::OutputExtent {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "optical-gain {field:?} output extent {actual} does not match {expected}"
            ),
            Self::NonFiniteInput { input, mode } => write!(
                formatter,
                "optical-gain {input:?} input for mode {mode} is not finite"
            ),
            Self::NonFiniteAccumulation { mode } => write!(
                formatter,
                "optical-gain quadrature accumulation for mode {mode} is not finite"
            ),
            Self::NonFiniteEstimate { mode } => write!(
                formatter,
                "optical-gain completed estimate for mode {mode} is not finite"
            ),
        }
    }
}

impl core::error::Error for OpticalGainEstimatorError {}

/// Same-frame modal samples used by synchronous optical-gain estimation.
#[derive(Clone, Copy, Debug)]
pub struct OpticalGainEstimatorInput<'a, T> {
    realized_command_modes: &'a [T],
    reconstructed_response_modes: &'a [T],
    window_state: Option<(usize, &'a [u64], bool)>,
}

impl<'a, T> OpticalGainEstimatorInput<'a, T> {
    /// Groups realized command and raw reconstructed response for one frame.
    pub const fn new(
        realized_command_modes: &'a [T],
        reconstructed_response_modes: &'a [T],
    ) -> Self {
        Self {
            realized_command_modes,
            reconstructed_response_modes,
            window_state: None,
        }
    }

    /// Attach the generator's zero-based window frame and completion flag.
    ///
    /// Graph declarations use this state to reject a missing, duplicated, or
    /// differently configured dither frame before mutating accumulation.
    pub const fn with_window_state(
        mut self,
        window_frame: usize,
        phase_indices: &'a [u64],
        completes_window: bool,
    ) -> Self {
        self.window_state = Some((window_frame, phase_indices, completes_window));
        self
    }

    /// Returns the realized modal dither command.
    pub const fn realized_command_modes(&self) -> &[T] {
        self.realized_command_modes
    }

    /// Returns the raw, uncorrected reconstructed modal response.
    pub const fn reconstructed_response_modes(&self) -> &[T] {
        self.reconstructed_response_modes
    }

    /// Returns the optional generator window state supplied by the caller.
    pub const fn window_state(&self) -> Option<(usize, &[u64], bool)> {
        self.window_state
    }
}

/// Caller-owned result of one synchronous optical-gain estimator call.
///
/// `complete` is false for intermediate frames. In that case the modal fields
/// are left unchanged. A completed window updates every modal field together.
#[derive(Debug)]
pub struct OpticalGainEstimateOutput<'a, T> {
    command_amplitudes: &'a mut [T],
    response_amplitudes: &'a mut [T],
    optical_gains: &'a mut [T],
    phase_lags: &'a mut [T],
    valid: &'a mut [bool],
    complete: &'a mut bool,
}

impl<'a, T> OpticalGainEstimateOutput<'a, T> {
    /// Groups caller-owned storage for one estimate outcome.
    pub fn new(
        command_amplitudes: &'a mut [T],
        response_amplitudes: &'a mut [T],
        optical_gains: &'a mut [T],
        phase_lags: &'a mut [T],
        valid: &'a mut [bool],
        complete: &'a mut bool,
    ) -> Self {
        Self {
            command_amplitudes,
            response_amplitudes,
            optical_gains,
            phase_lags,
            valid,
            complete,
        }
    }

    /// Returns the demodulated realized-command amplitudes.
    pub fn command_amplitudes(&self) -> &[T] {
        self.command_amplitudes
    }

    /// Returns the demodulated reconstructed-response amplitudes.
    pub fn response_amplitudes(&self) -> &[T] {
        self.response_amplitudes
    }

    /// Returns physical modal optical gains, not reciprocal corrections.
    pub fn optical_gains(&self) -> &[T] {
        self.optical_gains
    }

    /// Returns response-minus-command modal phases wrapped to `[-pi, pi]`.
    pub fn phase_lags(&self) -> &[T] {
        self.phase_lags
    }

    /// Returns per-mode estimate validity.
    pub fn valid(&self) -> &[bool] {
        self.valid
    }

    /// Returns whether this call completed and published an estimate window.
    pub fn is_complete(&self) -> bool {
        *self.complete
    }
}

/// Committed synchronous quadrature history for one estimator owner.
#[derive(Clone, Debug, PartialEq)]
pub struct OpticalGainEstimatorWorkspace<T> {
    frame_count: usize,
    command_cosines: Box<[T]>,
    command_sines: Box<[T]>,
    response_cosines: Box<[T]>,
    response_sines: Box<[T]>,
    next_command_cosines: Box<[T]>,
    next_command_sines: Box<[T]>,
    next_response_cosines: Box<[T]>,
    next_response_sines: Box<[T]>,
    command_amplitudes: Box<[T]>,
    response_amplitudes: Box<[T]>,
    optical_gains: Box<[T]>,
    phase_lags: Box<[T]>,
    valid: Box<[bool]>,
}

impl<T> OpticalGainEstimatorWorkspace<T> {
    /// Returns the number of successful samples committed in this window.
    pub const fn frame_count(&self) -> usize {
        self.frame_count
    }
}

impl<T> OpticalGainEstimatorWorkspace<T>
where
    T: Float,
{
    /// Discard partial accumulation and prior published estimates.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.command_cosines.fill(T::zero());
        self.command_sines.fill(T::zero());
        self.response_cosines.fill(T::zero());
        self.response_sines.fill(T::zero());
        self.next_command_cosines.fill(T::zero());
        self.next_command_sines.fill(T::zero());
        self.next_response_cosines.fill(T::zero());
        self.next_response_sines.fill(T::zero());
        self.command_amplitudes.fill(T::zero());
        self.response_amplitudes.fill(T::zero());
        self.optical_gains.fill(T::zero());
        self.phase_lags.fill(T::zero());
        self.valid.fill(false);
    }
}

/// Prepared synchronous estimator of physical modal optical gain.
///
/// The estimator demodulates same-frame realized command and raw reconstructed
/// response. Reciprocal correction is intentionally a separate operation.
#[derive(Clone, Debug)]
pub struct OpticalGainEstimatorPlan<T> {
    schedule: Arc<OpticalGainDitherSchedule<T>>,
    calibration_responses: Arc<[T]>,
    two_over_window: T,
}

impl<T> OpticalGainEstimatorPlan<T>
where
    T: Float + NumCast,
{
    /// Prepares calibration normalization for a shared dither schedule.
    pub fn new(
        schedule: impl Into<Arc<OpticalGainDitherSchedule<T>>>,
        calibration_responses: impl Into<Arc<[T]>>,
    ) -> Result<Self, OpticalGainEstimatorPlanError> {
        let schedule = schedule.into();
        let calibration_responses = calibration_responses.into();
        if calibration_responses.len() != schedule.len() {
            return Err(OpticalGainEstimatorPlanError::CalibrationResponseExtent {
                expected: schedule.len(),
                actual: calibration_responses.len(),
            });
        }
        for (mode, response) in calibration_responses.iter().enumerate() {
            if !response.is_finite() || *response <= T::zero() {
                return Err(OpticalGainEstimatorPlanError::InvalidCalibrationResponse { mode });
            }
        }
        let window = <T as NumCast>::from(schedule.frames_per_estimate())
            .ok_or(OpticalGainEstimatorPlanError::EstimateWindowNotRepresentable)?;
        let two_over_window = (T::one() + T::one()) / window;
        if !two_over_window.is_finite() || two_over_window <= T::zero() {
            return Err(OpticalGainEstimatorPlanError::EstimateWindowNotRepresentable);
        }
        Ok(Self {
            schedule,
            calibration_responses,
            two_over_window,
        })
    }

    /// Returns the modal extent.
    pub fn len(&self) -> usize {
        self.schedule.len()
    }

    /// Returns whether the modal extent is zero.
    pub fn is_empty(&self) -> bool {
        self.schedule.is_empty()
    }

    /// Returns the prepared calibration responses.
    pub fn calibration_responses(&self) -> &[T] {
        &self.calibration_responses
    }

    /// Returns the prepared synchronous-demodulation scale `2 / K`.
    pub const fn two_over_window(&self) -> T {
        self.two_over_window
    }

    /// Returns shared ownership of the immutable dither schedule.
    pub fn shared_schedule(&self) -> Arc<OpticalGainDitherSchedule<T>> {
        Arc::clone(&self.schedule)
    }

    /// Creates empty phase-zero synchronous accumulation history.
    pub fn workspace(&self) -> OpticalGainEstimatorWorkspace<T> {
        fn zeros<T: Float>(len: usize) -> Box<[T]> {
            alloc::vec![T::zero(); len].into_boxed_slice()
        }
        let len = self.len();
        OpticalGainEstimatorWorkspace {
            frame_count: 0,
            command_cosines: zeros(len),
            command_sines: zeros(len),
            response_cosines: zeros(len),
            response_sines: zeros(len),
            next_command_cosines: zeros(len),
            next_command_sines: zeros(len),
            next_response_cosines: zeros(len),
            next_response_sines: zeros(len),
            command_amplitudes: zeros(len),
            response_amplitudes: zeros(len),
            optical_gains: zeros(len),
            phase_lags: zeros(len),
            valid: alloc::vec![false; len].into_boxed_slice(),
        }
    }

    fn validate(
        &self,
        output: &OpticalGainEstimateOutput<'_, T>,
        input: &OpticalGainEstimatorInput<'_, T>,
    ) -> Result<(), OpticalGainEstimatorError> {
        self.validate_input_extent(
            OpticalGainEstimatorInputKind::RealizedCommand,
            input.realized_command_modes.len(),
        )?;
        self.validate_input_extent(
            OpticalGainEstimatorInputKind::ReconstructedResponse,
            input.reconstructed_response_modes.len(),
        )?;
        self.validate_output_extent(
            OpticalGainEstimateField::CommandAmplitude,
            output.command_amplitudes.len(),
        )?;
        self.validate_output_extent(
            OpticalGainEstimateField::ResponseAmplitude,
            output.response_amplitudes.len(),
        )?;
        self.validate_output_extent(
            OpticalGainEstimateField::OpticalGain,
            output.optical_gains.len(),
        )?;
        self.validate_output_extent(OpticalGainEstimateField::PhaseLag, output.phase_lags.len())?;
        self.validate_output_extent(OpticalGainEstimateField::Validity, output.valid.len())?;
        for (mode, value) in input.realized_command_modes.iter().enumerate() {
            if !value.is_finite() {
                return Err(OpticalGainEstimatorError::NonFiniteInput {
                    input: OpticalGainEstimatorInputKind::RealizedCommand,
                    mode,
                });
            }
        }
        for (mode, value) in input.reconstructed_response_modes.iter().enumerate() {
            if !value.is_finite() {
                return Err(OpticalGainEstimatorError::NonFiniteInput {
                    input: OpticalGainEstimatorInputKind::ReconstructedResponse,
                    mode,
                });
            }
        }
        Ok(())
    }

    fn validate_input_extent(
        &self,
        input: OpticalGainEstimatorInputKind,
        actual: usize,
    ) -> Result<(), OpticalGainEstimatorError> {
        if actual != self.len() {
            return Err(OpticalGainEstimatorError::InputExtent {
                input,
                expected: self.len(),
                actual,
            });
        }
        Ok(())
    }

    fn validate_output_extent(
        &self,
        field: OpticalGainEstimateField,
        actual: usize,
    ) -> Result<(), OpticalGainEstimatorError> {
        if actual != self.len() {
            return Err(OpticalGainEstimatorError::OutputExtent {
                field,
                expected: self.len(),
                actual,
            });
        }
        Ok(())
    }

    fn validate_workspace(
        &self,
        workspace: &OpticalGainEstimatorWorkspace<T>,
    ) -> Result<(), OpticalGainEstimatorError> {
        let len = self.len();
        if workspace.command_cosines.len() != len
            || workspace.command_sines.len() != len
            || workspace.response_cosines.len() != len
            || workspace.response_sines.len() != len
            || workspace.next_command_cosines.len() != len
            || workspace.next_command_sines.len() != len
            || workspace.next_response_cosines.len() != len
            || workspace.next_response_sines.len() != len
            || workspace.command_amplitudes.len() != len
            || workspace.response_amplitudes.len() != len
            || workspace.optical_gains.len() != len
            || workspace.phase_lags.len() != len
            || workspace.valid.len() != len
        {
            return Err(OpticalGainEstimatorError::WorkspaceExtent);
        }
        if workspace.frame_count >= self.schedule.frames_per_estimate() {
            return Err(OpticalGainEstimatorError::WorkspaceState);
        }
        Ok(())
    }

    fn validate_window_state(
        &self,
        workspace: &OpticalGainEstimatorWorkspace<T>,
        input: &OpticalGainEstimatorInput<'_, T>,
    ) -> Result<(), OpticalGainEstimatorError> {
        let Some((actual_frame, phase_indices, actual_complete)) = input.window_state else {
            return Ok(());
        };
        if actual_frame != workspace.frame_count {
            return Err(OpticalGainEstimatorError::WindowFrameMismatch {
                expected: workspace.frame_count,
                actual: actual_frame,
            });
        }
        if phase_indices.len() != self.len() {
            return Err(OpticalGainEstimatorError::PhaseIndexExtent {
                expected: self.len(),
                actual: phase_indices.len(),
            });
        }
        for (mode, actual_phase) in phase_indices.iter().copied().enumerate() {
            let period = self.schedule.modes()[mode].frames_per_period();
            let expected_phase = u64::try_from(workspace.frame_count % period)
                .map_err(|_| OpticalGainEstimatorError::PhaseIndexOutOfRange { mode })?;
            if actual_phase != expected_phase {
                return Err(OpticalGainEstimatorError::PhaseIndexMismatch {
                    mode,
                    expected: expected_phase,
                    actual: actual_phase,
                });
            }
        }
        let expected_complete = workspace.frame_count + 1 == self.schedule.frames_per_estimate();
        if actual_complete != expected_complete {
            return Err(OpticalGainEstimatorError::WindowCompletionMismatch {
                expected: expected_complete,
                actual: actual_complete,
            });
        }
        Ok(())
    }
}

impl<T> AlgorithmPlan for OpticalGainEstimatorPlan<T>
where
    T: Float + 'static,
{
    type Input<'a>
        = OpticalGainEstimatorInput<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = OpticalGainEstimateOutput<'a, T>
    where
        Self: 'a;
    type Workspace = OpticalGainEstimatorWorkspace<T>;
    type Error = OpticalGainEstimatorError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        self.validate(&output, &input)?;
        self.validate_workspace(workspace)?;
        self.validate_window_state(workspace, &input)?;

        for mode in 0..self.len() {
            let period = self.schedule.modes()[mode].frames_per_period();
            let phase = workspace.frame_count % period;
            let cosine = self.schedule.cosine(mode, phase);
            let sine = self.schedule.sine(mode, phase);
            workspace.next_command_cosines[mode] =
                workspace.command_cosines[mode] + input.realized_command_modes[mode] * cosine;
            workspace.next_command_sines[mode] =
                workspace.command_sines[mode] + input.realized_command_modes[mode] * sine;
            workspace.next_response_cosines[mode] = workspace.response_cosines[mode]
                + input.reconstructed_response_modes[mode] * cosine;
            workspace.next_response_sines[mode] =
                workspace.response_sines[mode] + input.reconstructed_response_modes[mode] * sine;
            if !workspace.next_command_cosines[mode].is_finite()
                || !workspace.next_command_sines[mode].is_finite()
                || !workspace.next_response_cosines[mode].is_finite()
                || !workspace.next_response_sines[mode].is_finite()
            {
                return Err(OpticalGainEstimatorError::NonFiniteAccumulation { mode });
            }
        }

        let completes_window = workspace.frame_count + 1 == self.schedule.frames_per_estimate();
        if completes_window {
            let pi = (-T::one()).acos();
            let two_pi = pi + pi;
            for mode in 0..self.len() {
                let command_cosine = workspace.next_command_cosines[mode] * self.two_over_window;
                let command_sine = workspace.next_command_sines[mode] * self.two_over_window;
                let response_cosine = workspace.next_response_cosines[mode] * self.two_over_window;
                let response_sine = workspace.next_response_sines[mode] * self.two_over_window;
                let command_amplitude = command_cosine.hypot(command_sine);
                let response_amplitude = response_cosine.hypot(response_sine);
                if !command_amplitude.is_finite() || !response_amplitude.is_finite() {
                    return Err(OpticalGainEstimatorError::NonFiniteEstimate { mode });
                }
                let optical_gain =
                    response_amplitude / (self.calibration_responses[mode] * command_amplitude);
                let valid = command_amplitude > T::zero()
                    && optical_gain.is_finite()
                    && optical_gain > T::zero();
                let phase_lag = if valid {
                    let mut phase =
                        response_sine.atan2(response_cosine) - command_sine.atan2(command_cosine);
                    if phase > pi {
                        phase = phase - two_pi;
                    } else if phase < -pi {
                        phase = phase + two_pi;
                    }
                    phase
                } else {
                    T::zero()
                };
                if !phase_lag.is_finite() {
                    return Err(OpticalGainEstimatorError::NonFiniteEstimate { mode });
                }
                workspace.command_amplitudes[mode] = command_amplitude;
                workspace.response_amplitudes[mode] = response_amplitude;
                workspace.optical_gains[mode] = if valid { optical_gain } else { T::zero() };
                workspace.phase_lags[mode] = phase_lag;
                workspace.valid[mode] = valid;
            }

            output
                .command_amplitudes
                .copy_from_slice(&workspace.command_amplitudes);
            output
                .response_amplitudes
                .copy_from_slice(&workspace.response_amplitudes);
            output
                .optical_gains
                .copy_from_slice(&workspace.optical_gains);
            output.phase_lags.copy_from_slice(&workspace.phase_lags);
            output.valid.copy_from_slice(&workspace.valid);
            *output.complete = true;

            workspace.command_cosines.fill(T::zero());
            workspace.command_sines.fill(T::zero());
            workspace.response_cosines.fill(T::zero());
            workspace.response_sines.fill(T::zero());
            workspace.frame_count = 0;
        } else {
            workspace
                .command_cosines
                .copy_from_slice(&workspace.next_command_cosines);
            workspace
                .command_sines
                .copy_from_slice(&workspace.next_command_sines);
            workspace
                .response_cosines
                .copy_from_slice(&workspace.next_response_cosines);
            workspace
                .response_sines
                .copy_from_slice(&workspace.next_response_sines);
            workspace.frame_count += 1;
            *output.complete = false;
        }
        Ok(())
    }
}

/// Invalid Float32 estimator declaration preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainEstimatorF32PreparationError {
    /// The modal dither schedule is invalid.
    Schedule(OpticalGainDitherScheduleError),
    /// Calibration normalization cannot be prepared.
    Plan(OpticalGainEstimatorPlanError),
}

impl fmt::Display for OpticalGainEstimatorF32PreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schedule(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for OpticalGainEstimatorF32PreparationError {}

impl From<OpticalGainDitherScheduleError> for OpticalGainEstimatorF32PreparationError {
    fn from(error: OpticalGainDitherScheduleError) -> Self {
        Self::Schedule(error)
    }
}

impl From<OpticalGainEstimatorPlanError> for OpticalGainEstimatorF32PreparationError {
    fn from(error: OpticalGainEstimatorPlanError) -> Self {
        Self::Plan(error)
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 synchronous modal optical-gain estimator.
    pub OpticalGainEstimatorF32 {
        label = "optical-gain-estimator-f32";
        configuration OpticalGainEstimatorF32Configuration {
            modes: Vec<OpticalGainDitherMode<f32>> = GraphRebuild;
            initial_calibration_responses: Vec<f32> = Construction;
        }
        plan = OpticalGainEstimatorPlan<f32>;
        workspace = OpticalGainEstimatorWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            mode_count = plan.len();
        }
        inputs {
            realized_command_modes: f32 [mode_count] = "realized-command-modes",
                schema = crate::schemas::OPTICAL_GAIN_DITHER_COMMAND_V1;
            reconstructed_response_modes: f32 [mode_count] = "reconstructed-response-modes",
                schema = crate::schemas::OPTICAL_GAIN_DITHER_RESPONSE_V1;
            window_frame: u64 [1] = "window-frame",
                schema = crate::schemas::OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1;
            phase_indices: u64 [mode_count] = "phase-indices",
                schema = crate::schemas::OPTICAL_GAIN_DITHER_PHASE_INDICES_V1;
            completes_window: bool [1] = "completes-window",
                schema = crate::schemas::OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1;
        }
        outputs {
            command_amplitudes: f32 [mode_count] = "command-amplitudes",
                schema = crate::schemas::OPTICAL_GAIN_COMMAND_AMPLITUDE_V1,
                metadata_from = "reconstructed-response-modes";
            response_amplitudes: f32 [mode_count] = "response-amplitudes",
                schema = crate::schemas::OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1,
                metadata_from = "reconstructed-response-modes";
            optical_gains: f32 [mode_count] = "optical-gains",
                schema = crate::schemas::OPTICAL_GAIN_DITHER_ESTIMATE_V1,
                metadata_from = "reconstructed-response-modes";
            phase_lags: f32 [mode_count] = "phase-lags",
                schema = crate::schemas::OPTICAL_GAIN_PHASE_LAG_V1,
                metadata_from = "reconstructed-response-modes";
            valid: bool [mode_count] = "valid",
                schema = crate::schemas::OPTICAL_GAIN_ESTIMATE_VALIDITY_V1,
                metadata_from = "reconstructed-response-modes";
            estimate_complete: bool [1] = "estimate-complete",
                schema = crate::schemas::OPTICAL_GAIN_ESTIMATE_COMPLETE_V1,
                metadata_from = "reconstructed-response-modes";
        }
        parameters {
            calibration_responses: f32 [mode_count] = "calibration-responses",
                schema = crate::schemas::OPTICAL_GAIN_CALIBRATION_RESPONSE_V1,
                replace |plan, values| {
                    OpticalGainEstimatorPlan::new(plan.shared_schedule(), values)
                };
        }
        parameter_error = OpticalGainEstimatorPlanError;
        prepare |configuration| {
            let schedule = OpticalGainDitherSchedule::new(configuration.modes)?;
            let plan = OpticalGainEstimatorPlan::new(
                schedule,
                configuration.initial_calibration_responses,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = OpticalGainEstimatorF32PreparationError;
        process |plan, workspace| {
            let frame = usize::try_from(window_frame[0])
                .map_err(|_| OpticalGainEstimatorError::WindowFrameOutOfRange)?;
            let mut complete = false;
            plan.process(
                OpticalGainEstimateOutput::new(
                    command_amplitudes,
                    response_amplitudes,
                    optical_gains,
                    phase_lags,
                    valid,
                    &mut complete,
                ),
                workspace,
                OpticalGainEstimatorInput::new(
                    realized_command_modes,
                    reconstructed_response_modes,
                )
                .with_window_state(frame, phase_indices, completes_window[0]),
            )?;
            estimate_complete[0] = complete;
            Ok(())
        }
        process_error = OpticalGainEstimatorError;
        reset |_plan, workspace| {
            workspace.reset();
        }
    }
}
