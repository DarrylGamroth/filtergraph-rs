//! Bounded updates of accepted physical optical-gain estimates.

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::fmt;

use num_traits::{Float, FloatConst};

use crate::AlgorithmPlan;

/// Acceptance criteria for one measured dither mode.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpticalGainUpdateCriteria<T> {
    minimum_command_amplitude: T,
    minimum_response_amplitude: T,
    minimum_optical_gain: T,
    maximum_optical_gain: T,
    maximum_absolute_phase_lag: T,
}

impl<T> OpticalGainUpdateCriteria<T> {
    /// Describes the acceptance bounds for one dither mode.
    ///
    /// Validation belongs to [`OpticalGainUpdatePlan::new`] so an error can
    /// identify the criterion within the complete prepared operation.
    pub const fn new(
        minimum_command_amplitude: T,
        minimum_response_amplitude: T,
        minimum_optical_gain: T,
        maximum_optical_gain: T,
        maximum_absolute_phase_lag: T,
    ) -> Self {
        Self {
            minimum_command_amplitude,
            minimum_response_amplitude,
            minimum_optical_gain,
            maximum_optical_gain,
            maximum_absolute_phase_lag,
        }
    }

    /// Returns the inclusive minimum realized-command amplitude.
    pub fn minimum_command_amplitude(&self) -> T
    where
        T: Copy,
    {
        self.minimum_command_amplitude
    }

    /// Returns the inclusive minimum reconstructed-response amplitude.
    pub fn minimum_response_amplitude(&self) -> T
    where
        T: Copy,
    {
        self.minimum_response_amplitude
    }

    /// Returns the inclusive minimum accepted physical optical gain.
    pub fn minimum_optical_gain(&self) -> T
    where
        T: Copy,
    {
        self.minimum_optical_gain
    }

    /// Returns the inclusive maximum accepted physical optical gain.
    pub fn maximum_optical_gain(&self) -> T
    where
        T: Copy,
    {
        self.maximum_optical_gain
    }

    /// Returns the inclusive maximum absolute response phase lag in radians.
    pub fn maximum_absolute_phase_lag(&self) -> T
    where
        T: Copy,
    {
        self.maximum_absolute_phase_lag
    }
}

/// Update assignment for one reconstructed mode.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum OpticalGainAssignment<T> {
    /// Preserve this reconstructed mode's committed physical optical gain.
    Hold,

    /// Filter an accepted estimate from one dither mode.
    DitherMode {
        /// Zero-based source dither-mode index.
        dither_mode: usize,
        /// Fraction of the estimate error adopted on one completed window.
        update_fraction: T,
    },
}

/// Invalid prepared optical-gain update policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainUpdatePlanError {
    /// At least one dither-mode criterion is required.
    EmptyCriteria,

    /// At least one reconstructed-mode assignment is required.
    EmptyAssignments,

    /// A minimum amplitude is non-finite or negative.
    InvalidAmplitudeThreshold {
        /// Zero-based dither-mode index.
        dither_mode: usize,
    },

    /// Physical optical-gain bounds are not finite, positive, and ordered.
    InvalidOpticalGainBounds {
        /// Zero-based dither-mode index.
        dither_mode: usize,
    },

    /// A maximum absolute phase lag is outside `[0, pi]`.
    InvalidMaximumAbsolutePhaseLag {
        /// Zero-based dither-mode index.
        dither_mode: usize,
    },

    /// An assignment names a dither mode outside the prepared criteria.
    AssignmentDitherModeOutOfBounds {
        /// Zero-based reconstructed-mode index.
        reconstructed_mode: usize,
        /// Supplied zero-based dither-mode index.
        dither_mode: usize,
    },

    /// An assigned update fraction is outside `(0, 1]`.
    InvalidUpdateFraction {
        /// Zero-based reconstructed-mode index.
        reconstructed_mode: usize,
    },
}

impl fmt::Display for OpticalGainUpdatePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCriteria => {
                formatter.write_str("optical-gain update criteria must be nonempty")
            }
            Self::EmptyAssignments => {
                formatter.write_str("optical-gain reconstructed-mode assignments must be nonempty")
            }
            Self::InvalidAmplitudeThreshold { dither_mode } => write!(
                formatter,
                "optical-gain amplitude threshold for dither mode {dither_mode} must be finite and nonnegative"
            ),
            Self::InvalidOpticalGainBounds { dither_mode } => write!(
                formatter,
                "physical optical-gain bounds for dither mode {dither_mode} must be finite, positive, and ordered"
            ),
            Self::InvalidMaximumAbsolutePhaseLag { dither_mode } => write!(
                formatter,
                "maximum absolute phase lag for dither mode {dither_mode} must be in [0, pi]"
            ),
            Self::AssignmentDitherModeOutOfBounds {
                reconstructed_mode,
                dither_mode,
            } => write!(
                formatter,
                "reconstructed mode {reconstructed_mode} names unavailable dither mode {dither_mode}"
            ),
            Self::InvalidUpdateFraction { reconstructed_mode } => write!(
                formatter,
                "optical-gain update fraction for reconstructed mode {reconstructed_mode} must be in (0, 1]"
            ),
        }
    }
}

impl core::error::Error for OpticalGainUpdatePlanError {}

/// Invalid initial physical optical-gain history.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainUpdateWorkspaceError {
    /// The initial physical optical-gain vector has the wrong extent.
    InitialOpticalGainExtent {
        /// Required reconstructed-mode extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },

    /// An initial physical optical gain is non-finite or nonpositive.
    InvalidInitialOpticalGain {
        /// Zero-based reconstructed-mode index.
        reconstructed_mode: usize,
    },

    /// An assigned initial gain lies outside its source dither-mode bounds.
    InitialOpticalGainOutsideBounds {
        /// Zero-based reconstructed-mode index.
        reconstructed_mode: usize,
        /// Assigned zero-based dither-mode index.
        dither_mode: usize,
    },
}

impl fmt::Display for OpticalGainUpdateWorkspaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitialOpticalGainExtent { expected, actual } => write!(
                formatter,
                "initial physical optical-gain extent {actual} does not match {expected}"
            ),
            Self::InvalidInitialOpticalGain { reconstructed_mode } => write!(
                formatter,
                "initial physical optical gain for reconstructed mode {reconstructed_mode} must be finite and positive"
            ),
            Self::InitialOpticalGainOutsideBounds {
                reconstructed_mode,
                dither_mode,
            } => write!(
                formatter,
                "initial physical optical gain for reconstructed mode {reconstructed_mode} is outside dither mode {dither_mode} bounds"
            ),
        }
    }
}

impl core::error::Error for OpticalGainUpdateWorkspaceError {}

/// Invalid optical-gain update input, output, workspace, or numerical result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainUpdateError {
    /// At least one estimator input field has the wrong dither-mode extent.
    InputExtent,

    /// At least one caller-owned output field has the wrong extent.
    OutputExtent,

    /// The workspace was created for incompatible prepared extents.
    WorkspaceExtent,

    /// A published complete-window estimator field is non-finite.
    NonFiniteInput {
        /// Zero-based dither-mode index.
        dither_mode: usize,
    },

    /// Filtering an accepted estimate produced a non-finite gain.
    NonFiniteOutput {
        /// Zero-based reconstructed-mode index.
        reconstructed_mode: usize,
    },
}

impl fmt::Display for OpticalGainUpdateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => formatter.write_str("optical-gain update input extent is invalid"),
            Self::OutputExtent => {
                formatter.write_str("optical-gain update output extent is invalid")
            }
            Self::WorkspaceExtent => {
                formatter.write_str("optical-gain update workspace extent is invalid")
            }
            Self::NonFiniteInput { dither_mode } => write!(
                formatter,
                "completed optical-gain estimate for dither mode {dither_mode} must be finite"
            ),
            Self::NonFiniteOutput { reconstructed_mode } => write!(
                formatter,
                "updated physical optical gain for reconstructed mode {reconstructed_mode} is not finite"
            ),
        }
    }
}

impl core::error::Error for OpticalGainUpdateError {}

/// Estimator fields consumed by one optical-gain update call.
#[derive(Clone, Copy, Debug)]
pub struct OpticalGainUpdateInput<'a, T> {
    command_amplitudes: &'a [T],
    response_amplitudes: &'a [T],
    optical_gains: &'a [T],
    phase_lags: &'a [T],
    valid: &'a [bool],
    complete: bool,
}

impl<'a, T> OpticalGainUpdateInput<'a, T> {
    /// Groups the synchronized output fields of an optical-gain estimator.
    pub const fn new(
        command_amplitudes: &'a [T],
        response_amplitudes: &'a [T],
        optical_gains: &'a [T],
        phase_lags: &'a [T],
        valid: &'a [bool],
        complete: bool,
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

    /// Returns whether the estimator published a complete window.
    pub const fn is_complete(&self) -> bool {
        self.complete
    }
}

/// Caller-owned result of one bounded optical-gain update.
#[derive(Debug)]
pub struct OpticalGainUpdateOutput<'a, T> {
    optical_gains: &'a mut [T],
    accepted_dither_modes: &'a mut [bool],
    updated_reconstructed_modes: &'a mut [bool],
}

impl<'a, T> OpticalGainUpdateOutput<'a, T> {
    /// Groups resulting gains and acceptance/update status storage.
    pub fn new(
        optical_gains: &'a mut [T],
        accepted_dither_modes: &'a mut [bool],
        updated_reconstructed_modes: &'a mut [bool],
    ) -> Self {
        Self {
            optical_gains,
            accepted_dither_modes,
            updated_reconstructed_modes,
        }
    }

    /// Returns resulting reconstructed-mode physical optical gains.
    pub fn optical_gains(&self) -> &[T] {
        self.optical_gains
    }

    /// Returns whether each completed dither-mode estimate was accepted.
    pub fn accepted_dither_modes(&self) -> &[bool] {
        self.accepted_dither_modes
    }

    /// Returns whether each reconstructed mode adopted an accepted estimate.
    pub fn updated_reconstructed_modes(&self) -> &[bool] {
        self.updated_reconstructed_modes
    }
}

/// Committed reconstructed-mode physical optical gains and fixed scratch.
#[derive(Clone, Debug, PartialEq)]
pub struct OpticalGainUpdateWorkspace<T> {
    optical_gains: Box<[T]>,
    next_optical_gains: Box<[T]>,
    accepted_dither_modes: Box<[bool]>,
    updated_reconstructed_modes: Box<[bool]>,
}

impl<T> OpticalGainUpdateWorkspace<T> {
    /// Returns the committed reconstructed-mode physical optical gains.
    pub fn optical_gains(&self) -> &[T] {
        &self.optical_gains
    }

    fn restore_prepared(&mut self, initial_optical_gains: &[T])
    where
        T: Copy,
    {
        for (destination, source) in self
            .optical_gains
            .iter_mut()
            .zip(initial_optical_gains.iter())
        {
            *destination = *source;
        }
        for (destination, source) in self
            .next_optical_gains
            .iter_mut()
            .zip(initial_optical_gains.iter())
        {
            *destination = *source;
        }
        self.accepted_dither_modes.fill(false);
        self.updated_reconstructed_modes.fill(false);
    }
}

/// Prepared acceptance, assignment, and filtering of physical optical gain.
///
/// This stateful operation consumes completed raw-response estimates. It does
/// not estimate optical gain, invert it into a correction, or activate a new
/// correction plan.
#[derive(Clone, Debug, PartialEq)]
pub struct OpticalGainUpdatePlan<T> {
    criteria: Arc<[OpticalGainUpdateCriteria<T>]>,
    assignments: Arc<[OpticalGainAssignment<T>]>,
}

impl<T> OpticalGainUpdatePlan<T>
where
    T: Float + FloatConst,
{
    /// Validates per-dither acceptance and per-reconstructed-mode assignment.
    pub fn new(
        criteria: impl Into<Arc<[OpticalGainUpdateCriteria<T>]>>,
        assignments: impl Into<Arc<[OpticalGainAssignment<T>]>>,
    ) -> Result<Self, OpticalGainUpdatePlanError> {
        let criteria = criteria.into();
        let assignments = assignments.into();
        if criteria.is_empty() {
            return Err(OpticalGainUpdatePlanError::EmptyCriteria);
        }
        if assignments.is_empty() {
            return Err(OpticalGainUpdatePlanError::EmptyAssignments);
        }
        for (dither_mode, criterion) in criteria.iter().enumerate() {
            if !criterion.minimum_command_amplitude.is_finite()
                || criterion.minimum_command_amplitude < T::zero()
                || !criterion.minimum_response_amplitude.is_finite()
                || criterion.minimum_response_amplitude < T::zero()
            {
                return Err(OpticalGainUpdatePlanError::InvalidAmplitudeThreshold { dither_mode });
            }
            if !criterion.minimum_optical_gain.is_finite()
                || criterion.minimum_optical_gain <= T::zero()
                || !(T::one() / criterion.minimum_optical_gain).is_finite()
                || !criterion.maximum_optical_gain.is_finite()
                || criterion.maximum_optical_gain < criterion.minimum_optical_gain
            {
                return Err(OpticalGainUpdatePlanError::InvalidOpticalGainBounds { dither_mode });
            }
            if !criterion.maximum_absolute_phase_lag.is_finite()
                || criterion.maximum_absolute_phase_lag < T::zero()
                || criterion.maximum_absolute_phase_lag > T::PI()
            {
                return Err(OpticalGainUpdatePlanError::InvalidMaximumAbsolutePhaseLag {
                    dither_mode,
                });
            }
        }
        for (reconstructed_mode, assignment) in assignments.iter().enumerate() {
            if let OpticalGainAssignment::DitherMode {
                dither_mode,
                update_fraction,
            } = assignment
            {
                if *dither_mode >= criteria.len() {
                    return Err(
                        OpticalGainUpdatePlanError::AssignmentDitherModeOutOfBounds {
                            reconstructed_mode,
                            dither_mode: *dither_mode,
                        },
                    );
                }
                if !update_fraction.is_finite()
                    || *update_fraction <= T::zero()
                    || *update_fraction > T::one()
                {
                    return Err(OpticalGainUpdatePlanError::InvalidUpdateFraction {
                        reconstructed_mode,
                    });
                }
            }
        }
        Ok(Self {
            criteria,
            assignments,
        })
    }

    /// Returns the number of measured dither modes.
    pub fn dither_mode_count(&self) -> usize {
        self.criteria.len()
    }

    /// Returns the number of reconstructed modes whose gains are retained.
    pub fn reconstructed_mode_count(&self) -> usize {
        self.assignments.len()
    }

    /// Returns the prepared per-dither-mode acceptance criteria.
    pub fn criteria(&self) -> &[OpticalGainUpdateCriteria<T>] {
        &self.criteria
    }

    /// Returns the prepared per-reconstructed-mode update assignments.
    pub fn assignments(&self) -> &[OpticalGainAssignment<T>] {
        &self.assignments
    }

    /// Creates committed history from initial physical optical gains.
    pub fn workspace(
        &self,
        initial_optical_gains: &[T],
    ) -> Result<OpticalGainUpdateWorkspace<T>, OpticalGainUpdateWorkspaceError> {
        if initial_optical_gains.len() != self.reconstructed_mode_count() {
            return Err(OpticalGainUpdateWorkspaceError::InitialOpticalGainExtent {
                expected: self.reconstructed_mode_count(),
                actual: initial_optical_gains.len(),
            });
        }
        for (reconstructed_mode, optical_gain) in initial_optical_gains.iter().enumerate() {
            if !optical_gain.is_finite() || *optical_gain <= T::zero() {
                return Err(OpticalGainUpdateWorkspaceError::InvalidInitialOpticalGain {
                    reconstructed_mode,
                });
            }
            if let OpticalGainAssignment::DitherMode { dither_mode, .. } =
                self.assignments[reconstructed_mode]
            {
                let criterion = self.criteria[dither_mode];
                if *optical_gain < criterion.minimum_optical_gain
                    || *optical_gain > criterion.maximum_optical_gain
                {
                    return Err(
                        OpticalGainUpdateWorkspaceError::InitialOpticalGainOutsideBounds {
                            reconstructed_mode,
                            dither_mode,
                        },
                    );
                }
            }
        }
        Ok(OpticalGainUpdateWorkspace {
            optical_gains: initial_optical_gains.into(),
            next_optical_gains: initial_optical_gains.into(),
            accepted_dither_modes: alloc::vec![false; self.dither_mode_count()].into_boxed_slice(),
            updated_reconstructed_modes: alloc::vec![false; self.reconstructed_mode_count()]
                .into_boxed_slice(),
        })
    }

    fn input_extents_are_valid(&self, input: &OpticalGainUpdateInput<'_, T>) -> bool {
        input.command_amplitudes.len() == self.dither_mode_count()
            && input.response_amplitudes.len() == self.dither_mode_count()
            && input.optical_gains.len() == self.dither_mode_count()
            && input.phase_lags.len() == self.dither_mode_count()
            && input.valid.len() == self.dither_mode_count()
    }

    fn output_extents_are_valid(&self, output: &OpticalGainUpdateOutput<'_, T>) -> bool {
        output.optical_gains.len() == self.reconstructed_mode_count()
            && output.accepted_dither_modes.len() == self.dither_mode_count()
            && output.updated_reconstructed_modes.len() == self.reconstructed_mode_count()
    }

    fn workspace_extents_are_valid(&self, workspace: &OpticalGainUpdateWorkspace<T>) -> bool {
        workspace.optical_gains.len() == self.reconstructed_mode_count()
            && workspace.next_optical_gains.len() == self.reconstructed_mode_count()
            && workspace.accepted_dither_modes.len() == self.dither_mode_count()
            && workspace.updated_reconstructed_modes.len() == self.reconstructed_mode_count()
    }
}

impl<T> AlgorithmPlan for OpticalGainUpdatePlan<T>
where
    T: Float + FloatConst + 'static,
{
    type Input<'a>
        = OpticalGainUpdateInput<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = OpticalGainUpdateOutput<'a, T>
    where
        Self: 'a;
    type Workspace = OpticalGainUpdateWorkspace<T>;
    type Error = OpticalGainUpdateError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if !self.input_extents_are_valid(&input) {
            return Err(OpticalGainUpdateError::InputExtent);
        }
        if !self.output_extents_are_valid(&output) {
            return Err(OpticalGainUpdateError::OutputExtent);
        }
        if !self.workspace_extents_are_valid(workspace) {
            return Err(OpticalGainUpdateError::WorkspaceExtent);
        }

        if !input.complete {
            output
                .optical_gains
                .copy_from_slice(&workspace.optical_gains);
            output.accepted_dither_modes.fill(false);
            output.updated_reconstructed_modes.fill(false);
            return Ok(());
        }

        for dither_mode in 0..self.dither_mode_count() {
            if !input.command_amplitudes[dither_mode].is_finite()
                || !input.response_amplitudes[dither_mode].is_finite()
                || !input.optical_gains[dither_mode].is_finite()
                || !input.phase_lags[dither_mode].is_finite()
            {
                return Err(OpticalGainUpdateError::NonFiniteInput { dither_mode });
            }
        }

        for dither_mode in 0..self.dither_mode_count() {
            let criterion = self.criteria[dither_mode];
            workspace.accepted_dither_modes[dither_mode] = input.valid[dither_mode]
                && input.command_amplitudes[dither_mode] >= criterion.minimum_command_amplitude
                && input.response_amplitudes[dither_mode] >= criterion.minimum_response_amplitude
                && input.optical_gains[dither_mode] >= criterion.minimum_optical_gain
                && input.optical_gains[dither_mode] <= criterion.maximum_optical_gain
                && input.phase_lags[dither_mode].abs() <= criterion.maximum_absolute_phase_lag;
        }

        workspace
            .next_optical_gains
            .copy_from_slice(&workspace.optical_gains);
        workspace.updated_reconstructed_modes.fill(false);
        for reconstructed_mode in 0..self.reconstructed_mode_count() {
            let OpticalGainAssignment::DitherMode {
                dither_mode,
                update_fraction,
            } = self.assignments[reconstructed_mode]
            else {
                continue;
            };
            if !workspace.accepted_dither_modes[dither_mode] {
                continue;
            }
            let current = workspace.optical_gains[reconstructed_mode];
            let estimate = input.optical_gains[dither_mode];
            let next = if update_fraction == T::one() {
                estimate
            } else if estimate >= current {
                (current + update_fraction * (estimate - current)).min(estimate)
            } else {
                (current - update_fraction * (current - estimate)).max(estimate)
            };
            if !next.is_finite() || next <= T::zero() {
                return Err(OpticalGainUpdateError::NonFiniteOutput { reconstructed_mode });
            }
            workspace.next_optical_gains[reconstructed_mode] = next;
            workspace.updated_reconstructed_modes[reconstructed_mode] = true;
        }

        output
            .optical_gains
            .copy_from_slice(&workspace.next_optical_gains);
        output
            .accepted_dither_modes
            .copy_from_slice(&workspace.accepted_dither_modes);
        output
            .updated_reconstructed_modes
            .copy_from_slice(&workspace.updated_reconstructed_modes);
        workspace
            .optical_gains
            .copy_from_slice(&workspace.next_optical_gains);
        Ok(())
    }
}

/// Invalid Float32 optical-gain update declaration preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainUpdateF32PreparationError {
    /// The acceptance or assignment policy is invalid.
    Plan(OpticalGainUpdatePlanError),
    /// Initial physical optical gains are invalid.
    Workspace(OpticalGainUpdateWorkspaceError),
}

impl fmt::Display for OpticalGainUpdateF32PreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => error.fmt(formatter),
            Self::Workspace(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for OpticalGainUpdateF32PreparationError {}

impl From<OpticalGainUpdatePlanError> for OpticalGainUpdateF32PreparationError {
    fn from(error: OpticalGainUpdatePlanError) -> Self {
        Self::Plan(error)
    }
}

impl From<OpticalGainUpdateWorkspaceError> for OpticalGainUpdateF32PreparationError {
    fn from(error: OpticalGainUpdateWorkspaceError) -> Self {
        Self::Workspace(error)
    }
}

/// Prepared history and reset value for the Float32 update declaration.
pub struct OpticalGainUpdateF32Workspace {
    algorithm: OpticalGainUpdateWorkspace<f32>,
    initial_optical_gains: Box<[f32]>,
}

crate::filtergraph_algorithm! {
    /// Portable Float32 acceptance and filtering of optical-gain estimates.
    pub OpticalGainUpdateF32 {
        label = "optical-gain-update-f32";
        configuration OpticalGainUpdateF32Configuration {
            criteria: Vec<OpticalGainUpdateCriteria<f32>> = GraphRebuild;
            assignments: Vec<OpticalGainAssignment<f32>> = GraphRebuild;
            initial_optical_gains: Vec<f32> = Construction;
        }
        plan = OpticalGainUpdatePlan<f32>;
        workspace = OpticalGainUpdateF32Workspace;
        properties = none;
        dimensions |plan, _workspace| {
            dither_mode_count = plan.dither_mode_count();
            reconstructed_mode_count = plan.reconstructed_mode_count();
        }
        inputs {
            command_amplitudes: f32 [dither_mode_count] = "command-amplitudes",
                schema = crate::schemas::OPTICAL_GAIN_COMMAND_AMPLITUDE_V1;
            response_amplitudes: f32 [dither_mode_count] = "response-amplitudes",
                schema = crate::schemas::OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1;
            optical_gain_estimates: f32 [dither_mode_count] = "optical-gain-estimates",
                schema = crate::schemas::OPTICAL_GAIN_DITHER_ESTIMATE_V1;
            phase_lags: f32 [dither_mode_count] = "phase-lags",
                schema = crate::schemas::OPTICAL_GAIN_PHASE_LAG_V1;
            valid: bool [dither_mode_count] = "valid",
                schema = crate::schemas::OPTICAL_GAIN_ESTIMATE_VALIDITY_V1;
            estimate_complete: bool [1] = "estimate-complete",
                schema = crate::schemas::OPTICAL_GAIN_ESTIMATE_COMPLETE_V1;
        }
        outputs {
            optical_gains: f32 [reconstructed_mode_count] = "optical-gains",
                schema = crate::schemas::PHYSICAL_OPTICAL_GAIN_V1,
                metadata_from = "estimate-complete";
            accepted_dither_modes: bool [dither_mode_count] = "accepted-dither-modes",
                schema = crate::schemas::OPTICAL_GAIN_ACCEPTED_DITHER_MODES_V1,
                metadata_from = "estimate-complete";
            updated_reconstructed_modes: bool [reconstructed_mode_count]
                = "updated-reconstructed-modes",
                schema = crate::schemas::OPTICAL_GAIN_UPDATED_RECONSTRUCTED_MODES_V1,
                metadata_from = "estimate-complete";
        }
        prepare |configuration| {
            let plan = OpticalGainUpdatePlan::new(
                configuration.criteria,
                configuration.assignments,
            )?;
            let initial_optical_gains = configuration.initial_optical_gains.into_boxed_slice();
            let algorithm = plan.workspace(&initial_optical_gains)?;
            Ok((
                plan,
                OpticalGainUpdateF32Workspace {
                    algorithm,
                    initial_optical_gains,
                },
            ))
        }
        prepare_error = OpticalGainUpdateF32PreparationError;
        process |plan, workspace| {
            plan.process(
                OpticalGainUpdateOutput::new(
                    optical_gains,
                    accepted_dither_modes,
                    updated_reconstructed_modes,
                ),
                &mut workspace.algorithm,
                OpticalGainUpdateInput::new(
                    command_amplitudes,
                    response_amplitudes,
                    optical_gain_estimates,
                    phase_lags,
                    valid,
                    estimate_complete[0],
                ),
            )
        }
        process_error = OpticalGainUpdateError;
        reset |_plan, workspace| {
            workspace
                .algorithm
                .restore_prepared(&workspace.initial_optical_gains);
        }
    }
}
