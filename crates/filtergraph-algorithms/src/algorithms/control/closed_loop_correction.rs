//! Leaky integration with delayed DM anti-windup and hidden-mode removal.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use core::{fmt, mem};

use num_traits::Float;

use crate::algorithms::linear_algebra::multiply_row_major;
use crate::{AlgorithmPlan, MatrixElement, PropertyConstraint, PropertyValue};

#[cfg(all(feature = "blas", target_arch = "x86_64"))]
mod hidden_projection;

const GAIN_PROPERTY: u32 = 0;
const POLE_PROPERTY: u32 = 1;
const ANTI_WINDUP_GAIN_PROPERTY: u32 = 2;
const HIDDEN_MODE_GAIN_PROPERTY: u32 = 3;
const CLOSED_PROPERTY: u32 = 4;

/// Invalid prepared closed-loop correction configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClosedLoopCorrectionPlanError {
    /// The correction state must contain at least one coordinate.
    EmptyState,
    /// Controller state Float32 byte extent exceeds UInt32.
    StateExtent,
    /// Gain or pole does not satisfy the leaky-integrator contract.
    InvalidIntegrator,
    /// The anti-windup gain must be finite.
    InvalidAntiWindupGain,
    /// The hidden-mode removal gain must be finite.
    InvalidHiddenModeGain,
    /// One hidden-mode projection has the wrong extent.
    HiddenModeProjectionExtent,
    /// A hidden-mode projection contains a non-finite coefficient.
    NonFiniteHiddenModeProjection,
}

impl fmt::Display for ClosedLoopCorrectionPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyState => formatter.write_str("correction state must be nonempty"),
            Self::StateExtent => {
                formatter.write_str("controller state byte extent must fit UInt32")
            }
            Self::InvalidIntegrator => formatter.write_str("integrator gain or pole is invalid"),
            Self::InvalidAntiWindupGain => formatter.write_str("anti-windup gain must be finite"),
            Self::InvalidHiddenModeGain => {
                formatter.write_str("hidden-mode removal gain must be finite")
            }
            Self::HiddenModeProjectionExtent => {
                formatter.write_str("hidden-mode projection extent is invalid")
            }
            Self::NonFiniteHiddenModeProjection => {
                formatter.write_str("hidden-mode projection must be finite")
            }
        }
    }
}

impl core::error::Error for ClosedLoopCorrectionPlanError {}

/// Invalid closed-loop correction input, output, or numerical result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClosedLoopCorrectionError {
    /// The integrator input has the wrong extent.
    InputExtent,
    /// The output has the wrong extent.
    OutputExtent,
    /// The workspace was prepared for incompatible dimensions.
    WorkspaceExtent,
    /// Feedback is required after the first output.
    MissingFeedback,
    /// Feedback was supplied before any output existed.
    UnexpectedFeedback,
    /// The feedback vector has the wrong extent.
    FeedbackExtent,
    /// An input value is not finite.
    NonFiniteInput,
    /// A feedback value is not finite.
    NonFiniteFeedback,
    /// The calculation produced a non-finite value.
    NonFiniteOutput,
}

impl fmt::Display for ClosedLoopCorrectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => formatter.write_str("integrator input extent is invalid"),
            Self::OutputExtent => formatter.write_str("correction output extent is invalid"),
            Self::WorkspaceExtent => {
                formatter.write_str("closed-loop correction workspace extent is invalid")
            }
            Self::MissingFeedback => formatter.write_str("previous DM feedback is missing"),
            Self::UnexpectedFeedback => {
                formatter.write_str("DM feedback preceded the first correction output")
            }
            Self::FeedbackExtent => formatter.write_str("DM feedback extent is invalid"),
            Self::NonFiniteInput => formatter.write_str("integrator input must be finite"),
            Self::NonFiniteFeedback => formatter.write_str("DM feedback must be finite"),
            Self::NonFiniteOutput => {
                formatter.write_str("closed-loop correction produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for ClosedLoopCorrectionError {}

/// Inputs for one closed-loop correction update.
#[derive(Clone, Copy, Debug)]
pub struct ClosedLoopCorrectionInput<'a, T> {
    /// Current leaky-integrator input `x[k]`.
    pub x: &'a [T],
    /// Constraint feedback for the preceding output.
    ///
    /// This is `None` for the first update and required thereafter.
    pub constraint_feedback: Option<&'a [T]>,
}

/// Prepared leaky integration, anti-windup, and hidden-mode removal.
///
/// Given the preceding integrated output `u`, same-coordinate constraint
/// feedback `f`, and hidden-mode projections `H` and `J`, the state used by
/// the next integration is:
///
/// ```text
/// realized = u - f
/// state = u - k_aw * f - k_h * J * H * realized
/// output = p * state + g * x
/// ```
#[derive(Clone, Debug)]
pub struct ClosedLoopCorrectionPlan<T> {
    pole: T,
    gain: T,
    anti_windup_gain: T,
    hidden_mode_gain: T,
    closed: bool,
    state_length: usize,
    hidden_mode_count: usize,
    shape_to_hidden: Arc<[T]>,
    hidden_to_shape: Arc<[T]>,
    #[cfg(all(feature = "blas", target_arch = "x86_64"))]
    hidden_diagonal: Option<hidden_projection::RepeatedDiagonal<T>>,
}

// Prepared acceleration metadata can distinguish signed-zero coefficients
// that compare equal in the public projection matrices.
impl<T: PartialEq> PartialEq for ClosedLoopCorrectionPlan<T> {
    fn eq(&self, other: &Self) -> bool {
        self.pole == other.pole
            && self.gain == other.gain
            && self.anti_windup_gain == other.anti_windup_gain
            && self.hidden_mode_gain == other.hidden_mode_gain
            && self.closed == other.closed
            && self.state_length == other.state_length
            && self.hidden_mode_count == other.hidden_mode_count
            && self.shape_to_hidden == other.shape_to_hidden
            && self.hidden_to_shape == other.hidden_to_shape
    }
}

impl<T> ClosedLoopCorrectionPlan<T>
where
    T: MatrixElement,
{
    /// Prepares closed-loop correction without hidden-mode removal.
    pub fn new(
        gain: T,
        pole: T,
        anti_windup_gain: T,
        state_length: usize,
    ) -> Result<Self, ClosedLoopCorrectionPlanError> {
        Self::with_hidden_modes(
            gain,
            pole,
            anti_windup_gain,
            state_length,
            0,
            T::zero(),
            Arc::from([]),
            Arc::from([]),
        )
    }

    /// Prepares closed-loop correction with explicit hidden-mode projections.
    #[allow(clippy::too_many_arguments)]
    pub fn with_hidden_modes(
        gain: T,
        pole: T,
        anti_windup_gain: T,
        state_length: usize,
        hidden_mode_count: usize,
        hidden_mode_gain: T,
        shape_to_hidden: Arc<[T]>,
        hidden_to_shape: Arc<[T]>,
    ) -> Result<Self, ClosedLoopCorrectionPlanError> {
        if state_length == 0 {
            return Err(ClosedLoopCorrectionPlanError::EmptyState);
        }
        if !gain.is_finite() || !pole.is_finite() || pole < T::zero() || pole > T::one() {
            return Err(ClosedLoopCorrectionPlanError::InvalidIntegrator);
        }
        if !anti_windup_gain.is_finite() {
            return Err(ClosedLoopCorrectionPlanError::InvalidAntiWindupGain);
        }
        if !hidden_mode_gain.is_finite() {
            return Err(ClosedLoopCorrectionPlanError::InvalidHiddenModeGain);
        }
        let projection_length = state_length
            .checked_mul(hidden_mode_count)
            .ok_or(ClosedLoopCorrectionPlanError::HiddenModeProjectionExtent)?;
        if shape_to_hidden.len() != projection_length || hidden_to_shape.len() != projection_length
        {
            return Err(ClosedLoopCorrectionPlanError::HiddenModeProjectionExtent);
        }
        if shape_to_hidden.iter().any(|value| !value.is_finite())
            || hidden_to_shape.iter().any(|value| !value.is_finite())
        {
            return Err(ClosedLoopCorrectionPlanError::NonFiniteHiddenModeProjection);
        }

        #[cfg(all(feature = "blas", target_arch = "x86_64"))]
        let hidden_diagonal = hidden_projection::RepeatedDiagonal::prepare(
            state_length,
            hidden_mode_count,
            &hidden_to_shape,
        );
        Ok(Self {
            #[cfg(all(feature = "blas", target_arch = "x86_64"))]
            hidden_diagonal,
            pole,
            gain,
            anti_windup_gain,
            hidden_mode_gain,
            closed: true,
            state_length,
            hidden_mode_count,
            shape_to_hidden,
            hidden_to_shape,
        })
    }

    /// Allocates the fixed workspace for this prepared plan.
    pub fn workspace(&self) -> ClosedLoopCorrectionWorkspace<T> {
        ClosedLoopCorrectionWorkspace {
            state: vec![T::zero(); self.state_length].into_boxed_slice(),
            last_output: vec![T::zero(); self.state_length].into_boxed_slice(),
            next_state: vec![T::zero(); self.state_length].into_boxed_slice(),
            next_output: vec![T::zero(); self.state_length].into_boxed_slice(),
            realized: vec![T::zero(); self.state_length].into_boxed_slice(),
            hidden_modes: vec![T::zero(); self.hidden_mode_count].into_boxed_slice(),
            hidden_in_state: vec![T::zero(); self.state_length].into_boxed_slice(),
            has_output: false,
        }
    }

    /// Returns the discrete-time integrator pole `p`.
    pub const fn pole(&self) -> T {
        self.pole
    }

    /// Returns the leaky-integrator gain `g`.
    pub const fn gain(&self) -> T {
        self.gain
    }

    /// Returns the selected anti-windup gain `k_aw`.
    pub const fn anti_windup_gain(&self) -> T {
        self.anti_windup_gain
    }

    /// Returns the selected hidden-mode removal gain `k_h`.
    pub const fn hidden_mode_gain(&self) -> T {
        self.hidden_mode_gain
    }

    /// Returns the controller-state coordinate extent.
    pub const fn state_length(&self) -> usize {
        self.state_length
    }

    /// Returns the number of explicitly prepared hidden modes.
    pub const fn hidden_mode_count(&self) -> usize {
        self.hidden_mode_count
    }

    /// Returns the row-major state-to-hidden-mode projection.
    pub fn shape_to_hidden(&self) -> &[T] {
        &self.shape_to_hidden
    }

    /// Returns the independently prepared row-major hidden-mode-to-state projection.
    pub fn hidden_to_shape(&self) -> &[T] {
        &self.hidden_to_shape
    }

    /// Prepares an otherwise identical plan with a new state-to-hidden-mode
    /// projection.
    pub fn with_shape_to_hidden(
        &self,
        shape_to_hidden: Arc<[T]>,
    ) -> Result<Self, ClosedLoopCorrectionPlanError> {
        Self::with_hidden_modes(
            self.gain,
            self.pole,
            self.anti_windup_gain,
            self.state_length,
            self.hidden_mode_count,
            self.hidden_mode_gain,
            shape_to_hidden,
            Arc::clone(&self.hidden_to_shape),
        )
        .map(|mut replacement| {
            replacement.closed = self.closed;
            replacement
        })
    }

    /// Prepares an otherwise identical plan with a new hidden-mode-to-state
    /// projection.
    pub fn with_hidden_to_shape(
        &self,
        hidden_to_shape: Arc<[T]>,
    ) -> Result<Self, ClosedLoopCorrectionPlanError> {
        Self::with_hidden_modes(
            self.gain,
            self.pole,
            self.anti_windup_gain,
            self.state_length,
            self.hidden_mode_count,
            self.hidden_mode_gain,
            Arc::clone(&self.shape_to_hidden),
            hidden_to_shape,
        )
        .map(|mut replacement| {
            replacement.closed = self.closed;
            replacement
        })
    }

    fn apply_feedback(&self, workspace: &mut ClosedLoopCorrectionWorkspace<T>, feedback: &[T]) {
        for (((realized, next_state), last_output), feedback) in workspace
            .realized
            .iter_mut()
            .zip(workspace.next_state.iter_mut())
            .zip(workspace.last_output.iter().copied())
            .zip(feedback.iter().copied())
        {
            *realized = last_output - feedback;
            *next_state = (-self.anti_windup_gain).mul_add(feedback, last_output);
        }

        if self.hidden_mode_count != 0 {
            multiply_row_major(
                &self.shape_to_hidden,
                self.hidden_mode_count,
                self.state_length,
                &workspace.realized,
                &mut workspace.hidden_modes,
            );
            self.project_hidden_modes(&workspace.hidden_modes, &mut workspace.hidden_in_state);
            for (next_state, hidden) in workspace
                .next_state
                .iter_mut()
                .zip(workspace.hidden_in_state.iter().copied())
            {
                *next_state = (-self.hidden_mode_gain).mul_add(hidden, *next_state);
            }
        }
    }

    fn project_hidden_modes(&self, hidden: &[T], shape: &mut [T]) {
        #[cfg(all(feature = "blas", target_arch = "x86_64"))]
        if let Some(diagonal) = &self.hidden_diagonal
            && diagonal.project(hidden, shape)
        {
            return;
        }
        multiply_row_major(
            &self.hidden_to_shape,
            self.state_length,
            self.hidden_mode_count,
            hidden,
            shape,
        );
    }

    fn workspace_is_compatible(&self, workspace: &ClosedLoopCorrectionWorkspace<T>) -> bool {
        workspace.state.len() == self.state_length
            && workspace.last_output.len() == self.state_length
            && workspace.next_state.len() == self.state_length
            && workspace.next_output.len() == self.state_length
            && workspace.realized.len() == self.state_length
            && workspace.hidden_modes.len() == self.hidden_mode_count
            && workspace.hidden_in_state.len() == self.state_length
    }
}

impl<T> AlgorithmPlan for ClosedLoopCorrectionPlan<T>
where
    T: MatrixElement + 'static,
{
    type Input<'a>
        = ClosedLoopCorrectionInput<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type Workspace = ClosedLoopCorrectionWorkspace<T>;
    type Error = ClosedLoopCorrectionError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if input.x.len() != self.state_length {
            return Err(ClosedLoopCorrectionError::InputExtent);
        }
        if output.len() != self.state_length {
            return Err(ClosedLoopCorrectionError::OutputExtent);
        }
        if !self.workspace_is_compatible(workspace) {
            return Err(ClosedLoopCorrectionError::WorkspaceExtent);
        }
        if !input
            .x
            .iter()
            .fold(true, |finite, value| finite & value.is_finite())
        {
            return Err(ClosedLoopCorrectionError::NonFiniteInput);
        }

        match (workspace.has_output, input.constraint_feedback) {
            (false, None) => workspace.next_state.copy_from_slice(&workspace.state),
            (false, Some(_)) => return Err(ClosedLoopCorrectionError::UnexpectedFeedback),
            (true, None) => return Err(ClosedLoopCorrectionError::MissingFeedback),
            (true, Some(feedback)) => {
                if feedback.len() != self.state_length {
                    return Err(ClosedLoopCorrectionError::FeedbackExtent);
                }
                if !feedback
                    .iter()
                    .fold(true, |finite, value| finite & value.is_finite())
                {
                    return Err(ClosedLoopCorrectionError::NonFiniteFeedback);
                }
                if self.closed {
                    self.apply_feedback(workspace, feedback);
                } else {
                    workspace.next_state.copy_from_slice(&workspace.state);
                }
            }
        }

        let mut finite_output = true;
        if self.closed {
            // All slices have the checked state extent. Keep independent lanes
            // contiguous and validate disposable scratch before publication.
            for ((next_output, state), x) in workspace
                .next_output
                .iter_mut()
                .zip(workspace.next_state.iter().copied())
                .zip(input.x.iter().copied())
            {
                let value = self.gain.mul_add(x, self.pole * state);
                finite_output &= state.is_finite() & value.is_finite();
                *next_output = value;
            }
        } else {
            workspace
                .next_output
                .copy_from_slice(&workspace.last_output);
            finite_output = workspace
                .next_state
                .iter()
                .chain(workspace.next_output.iter())
                .fold(true, |finite, value| finite & value.is_finite());
        }
        if !finite_output {
            return Err(ClosedLoopCorrectionError::NonFiniteOutput);
        }

        output.copy_from_slice(&workspace.next_output);
        mem::swap(&mut workspace.state, &mut workspace.next_state);
        mem::swap(&mut workspace.last_output, &mut workspace.next_output);
        workspace.has_output = true;
        Ok(())
    }
}

fn replace_closed_loop_correction_properties<T>(
    plan: &ClosedLoopCorrectionPlan<T>,
    gain: T,
    pole: T,
    anti_windup_gain: T,
    hidden_mode_gain: T,
    closed: bool,
) -> Result<ClosedLoopCorrectionPlan<T>, ClosedLoopCorrectionPlanError>
where
    T: MatrixElement,
{
    ClosedLoopCorrectionPlan::with_hidden_modes(
        gain,
        pole,
        anti_windup_gain,
        plan.state_length,
        plan.hidden_mode_count,
        hidden_mode_gain,
        Arc::clone(&plan.shape_to_hidden),
        Arc::clone(&plan.hidden_to_shape),
    )
    .map(|mut replacement| {
        replacement.closed = closed;
        replacement
    })
}

macro_rules! impl_closed_loop_correction_properties {
    ($scalar:ty, $value:ident) => {
        crate::algorithm_properties! {
            impl ClosedLoopCorrectionPlan<$scalar> {
                plan_error = ClosedLoopCorrectionPlanError;
                replace = replace_closed_loop_correction_properties;
                properties {
                    GAIN_PROPERTY => gain: $value {
                        name = "gain";
                        description = "Leaky-integrator input gain g";
                        unit = "1";
                        update = Runtime;
                        default = 1.0;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(<$scalar>::MIN),
                            maximum: PropertyValue::$value(<$scalar>::MAX),
                        };
                    }
                    POLE_PROPERTY => pole: $value {
                        name = "pole";
                        description = "Discrete-time integrator pole p";
                        unit = "1";
                        update = Runtime;
                        default = 0.0;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(0.0),
                            maximum: PropertyValue::$value(1.0),
                        };
                    }
                    ANTI_WINDUP_GAIN_PROPERTY => anti_windup_gain: $value {
                        name = "anti-windup-gain";
                        description = "Constraint-feedback anti-windup gain k_aw";
                        unit = "1";
                        update = Runtime;
                        default = 0.0;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(<$scalar>::MIN),
                            maximum: PropertyValue::$value(<$scalar>::MAX),
                        };
                    }
                    HIDDEN_MODE_GAIN_PROPERTY => hidden_mode_gain: $value {
                        name = "hidden-mode-gain";
                        description = "Hidden-mode removal gain k_h";
                        unit = "1";
                        update = Runtime;
                        default = 0.0;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(<$scalar>::MIN),
                            maximum: PropertyValue::$value(<$scalar>::MAX),
                        };
                    }
                    CLOSED_PROPERTY => closed: Bool {
                        name = "closed";
                        description = "Close feedback integration; false holds committed correction and state";
                        unit = "1";
                        update = Runtime;
                        default = true;
                        constraint = PropertyConstraint::None;
                    }
                }
            }
        }
    };
}

impl_closed_loop_correction_properties!(f32, Float);
impl_closed_loop_correction_properties!(f64, Double);

/// Carried state and fixed scratch for closed-loop correction.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedLoopCorrectionWorkspace<T> {
    state: Box<[T]>,
    last_output: Box<[T]>,
    next_state: Box<[T]>,
    next_output: Box<[T]>,
    realized: Box<[T]>,
    hidden_modes: Box<[T]>,
    hidden_in_state: Box<[T]>,
    has_output: bool,
}

impl<T> ClosedLoopCorrectionWorkspace<T>
where
    T: Float,
{
    /// Returns the most recent feedback-corrected integrator state.
    pub fn state(&self) -> &[T] {
        &self.state
    }

    /// Returns the most recent integrated output awaiting or using feedback.
    pub fn last_output(&self) -> Option<&[T]> {
        self.has_output.then_some(&self.last_output)
    }

    /// Publish held correction/state without consuming residuals or delayed feedback.
    /// A held first frame establishes a preceding output for the next valid frame.
    pub fn hold(
        &mut self,
        correction: &mut [T],
        controller_state: &mut [T],
    ) -> Result<(), ClosedLoopCorrectionError> {
        if correction.len() != self.last_output.len() || controller_state.len() != self.state.len()
        {
            return Err(ClosedLoopCorrectionError::OutputExtent);
        }
        correction.copy_from_slice(&self.last_output);
        controller_state.copy_from_slice(&self.state);
        self.has_output = true;
        Ok(())
    }

    /// Clear correction history at a serialized protective-recovery boundary.
    pub fn recover(&mut self) {
        self.reset();
    }

    /// Clears integrator state, prior output, and fixed scratch.
    pub fn reset(&mut self) {
        self.state.fill(T::zero());
        self.last_output.fill(T::zero());
        self.next_state.fill(T::zero());
        self.next_output.fill(T::zero());
        self.realized.fill(T::zero());
        self.hidden_modes.fill(T::zero());
        self.hidden_in_state.fill(T::zero());
        self.has_output = false;
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 closed-loop correction declaration.
    pub ClosedLoopCorrectionF32 {
        label = "closed-loop-correction-f32";
        configuration ClosedLoopCorrectionF32Configuration {
            state_length: usize = GraphRebuild;
            hidden_mode_count: usize = GraphRebuild;
        }
        plan = ClosedLoopCorrectionPlan<f32>;
        properties = runtime;
        dimensions |plan, _workspace| {
            state_length = plan.state_length();
            hidden_mode_count = plan.hidden_mode_count();
        }
        inputs {
            residual_error: f32 [state_length] = "residual-error",
                schema = crate::schemas::CONTROLLER_RESIDUAL_ERROR_V1;
            constraint_feedback: f32 [state_length] = "constraint-feedback",
                schema = crate::schemas::CONTROLLER_CONSTRAINT_FEEDBACK_V1;
        }
        outputs {
            correction: f32 [state_length] = "correction",
                schema = crate::schemas::CONTROLLER_COMMAND_V1,
                metadata_from = "residual-error";
            controller_state: f32 [state_length] = "controller-state",
                schema = crate::schemas::CONTROLLER_STATE_V1,
                metadata_from = "residual-error";
        }
        parameters {
            shape_to_hidden: f32 [hidden_mode_count, state_length] = "shape-to-hidden",
                schema = crate::schemas::SHAPE_TO_HIDDEN_PROJECTION_V1,
                replace |plan, values| {
                    plan.with_shape_to_hidden(Arc::from(values))
                };
            hidden_to_shape: f32 [state_length, hidden_mode_count] = "hidden-to-shape",
                schema = crate::schemas::HIDDEN_TO_SHAPE_PROJECTION_V1,
                replace |plan, values| {
                    plan.with_hidden_to_shape(Arc::from(values))
                };
        }
        parameter_error = ClosedLoopCorrectionPlanError;
        prepare |configuration| {
            let projection_elements = configuration
                .state_length
                .checked_mul(configuration.hidden_mode_count)
                .ok_or(ClosedLoopCorrectionPlanError::HiddenModeProjectionExtent)?;
            let plan = ClosedLoopCorrectionPlan::with_hidden_modes(
                1.0,
                0.0,
                0.0,
                configuration.state_length,
                configuration.hidden_mode_count,
                0.0,
                Arc::from(vec![0.0; projection_elements]),
                Arc::from(vec![0.0; projection_elements]),
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = ClosedLoopCorrectionPlanError;
        process |plan, workspace| {
            let constraint_feedback = workspace
                .last_output()
                .is_some()
                .then_some(constraint_feedback);
            plan.process(
                correction,
                workspace,
                ClosedLoopCorrectionInput {
                    x: residual_error,
                    constraint_feedback,
                },
            )?;
            controller_state.copy_from_slice(workspace.state());
            Ok(())
        }
        process_error = ClosedLoopCorrectionError;
        reset |_plan, workspace| {
            workspace.reset();
        }
    }
}

crate::filtergraph_algorithm! {
    /// Feedback-aware leaky integration without hidden-mode projection ports.
    pub ClosedLoopIntegratorF32 {
        label = "closed-loop-integrator-f32";
        configuration ClosedLoopIntegratorF32Configuration {
            state_length: usize = GraphRebuild;
        }
        plan = ClosedLoopCorrectionPlan<f32>;
        properties = runtime;
        dimensions |plan, _workspace| {
            state_length = plan.state_length();
        }
        inputs {
            residual_error: f32 [state_length] = "residual-error",
                schema = crate::schemas::CONTROLLER_RESIDUAL_ERROR_V1, optional_sample_metadata = ["chopped_image_kind", "trigger_state", "acquisition_identity", "exposure_duration_nanoseconds"];
            constraint_feedback: f32 [state_length] = "constraint-feedback",
                schema = crate::schemas::CONTROLLER_CONSTRAINT_FEEDBACK_V1;
        }
        outputs {
            correction: f32 [state_length] = "correction",
                schema = crate::schemas::CONTROLLER_COMMAND_V1,
                metadata_from = "residual-error";
            controller_state: f32 [state_length] = "controller-state",
                schema = crate::schemas::CONTROLLER_STATE_V1,
                metadata_from = "residual-error";
        }
        prepare |configuration| {
            if configuration.state_length == 0 { return Err(ClosedLoopCorrectionPlanError::EmptyState); }
            if configuration.state_length > u32::MAX as usize / 4 { return Err(ClosedLoopCorrectionPlanError::StateExtent); }
            let plan = ClosedLoopCorrectionPlan::new(1.0,0.0,0.0,configuration.state_length)?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = ClosedLoopCorrectionPlanError;
        process |plan, workspace| {
            let constraint_feedback = workspace
                .last_output()
                .is_some()
                .then_some(constraint_feedback);
            plan.process(
                correction,
                workspace,
                ClosedLoopCorrectionInput {
                    x: residual_error,
                    constraint_feedback,
                },
            )?;
            controller_state.copy_from_slice(workspace.state());
            Ok(())
        }
        process_error = ClosedLoopCorrectionError;
        reset |_plan, workspace| {
            workspace.reset();
        }
    }
}

crate::filtergraph_algorithm! {
    /// Closed-loop integration held whenever any measurement validity element is false.
    pub ValidityGatedClosedLoopIntegratorF32 {
        label = "validity-gated-closed-loop-integrator-f32";
        configuration ValidityGatedClosedLoopIntegratorF32Configuration {
            state_length: usize = GraphRebuild;
            validity_length: usize = GraphRebuild;
            validity_schema: alloc::string::String = GraphRebuild;
        }
        plan = ClosedLoopCorrectionPlan<f32>;
        workspace = (ClosedLoopCorrectionWorkspace<f32>, usize);
        properties = runtime;
        dimensions |plan, workspace| {
            state_length = plan.state_length();
            validity_length = workspace.1;
        }
        inputs {
            residual_error: f32 [state_length] = "residual-error", schema = crate::schemas::CONTROLLER_RESIDUAL_ERROR_V1, optional_sample_metadata = ["chopped_image_kind", "trigger_state", "acquisition_identity", "exposure_duration_nanoseconds"];
            constraint_feedback: f32 [state_length] = "constraint-feedback", schema = crate::schemas::CONTROLLER_CONSTRAINT_FEEDBACK_V1;
            measurement_validity: bool [validity_length] = "measurement-validity", schema = configuration.validity_schema.as_str();
        }
        outputs {
            correction: f32 [state_length] = "correction", schema = crate::schemas::CONTROLLER_COMMAND_V1, metadata_from = "residual-error";
            controller_state: f32 [state_length] = "controller-state", schema = crate::schemas::CONTROLLER_STATE_V1, metadata_from = "residual-error";
        }
        prepare |configuration| {
            if configuration.state_length == 0 { return Err(ClosedLoopCorrectionPlanError::EmptyState); }
            if configuration.state_length > u32::MAX as usize / 4 || configuration.validity_length == 0 || configuration.validity_length > u32::MAX as usize || configuration.validity_schema.is_empty() { return Err(ClosedLoopCorrectionPlanError::StateExtent); }
            let plan = ClosedLoopCorrectionPlan::new(1.0,0.0,0.0,configuration.state_length)?;
            let workspace = (plan.workspace(), configuration.validity_length);
            Ok((plan,workspace))
        }
        prepare_error = ClosedLoopCorrectionPlanError;
        process |plan,workspace| {
            let workspace = &mut workspace.0;
            if measurement_validity.iter().all(|v|*v) {
                let feedback = workspace.last_output().is_some().then_some(constraint_feedback);
                plan.process(correction,workspace,ClosedLoopCorrectionInput{x:residual_error,constraint_feedback:feedback})?;
                controller_state.copy_from_slice(workspace.state());
            } else {
                workspace.hold(correction,controller_state)?;
            }
            Ok(())
        }
        process_error = ClosedLoopCorrectionError;
        reset |_plan,workspace| { workspace.0.reset(); }
    }
}
