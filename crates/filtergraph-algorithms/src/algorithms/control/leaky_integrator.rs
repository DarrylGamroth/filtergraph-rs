//! Prepared elementwise leaky integration with abort-safe mathematical history.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use core::fmt;
use core::mem;
use core::ops::Range;

use num_traits::Float;

use crate::{
    AlgorithmPlan, InputProgress, OutputProgress, ProcessError, ProgressResult, PropertyConstraint,
    PropertyValue,
};

const GAIN_PROPERTY: u32 = 0;
const POLE_PROPERTY: u32 = 1;
const CLOSED_PROPERTY: u32 = 2;

/// Invalid prepared leaky-integrator coefficients.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeakyIntegratorPlanError {
    /// Gain is non-finite.
    InvalidGain,

    /// Pole is non-finite or outside the closed interval `[0, 1]`.
    InvalidPole,
}

impl fmt::Display for LeakyIntegratorPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGain => formatter.write_str("gain must be finite"),
            Self::InvalidPole => formatter.write_str("pole must be finite and in [0, 1]"),
        }
    }
}

impl core::error::Error for LeakyIntegratorPlanError {}

/// Invalid leaky-integrator workspace preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeakyIntegratorWorkspaceError {
    /// Mathematical state contains no coordinates.
    EmptyState,
}

impl fmt::Display for LeakyIntegratorWorkspaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyState => formatter.write_str("state length must be positive"),
        }
    }
}

impl core::error::Error for LeakyIntegratorWorkspaceError {}

/// Numerical rejection or direct-call misuse of leaky integration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeakyIntegratorError {
    /// A direct-call sequencing, extent, or prepared-invariant error.
    Process(ProcessError),
    /// The supplied sample contains a nonfinite input coordinate.
    NonFiniteInput,
    /// The recurrence produces a nonfinite next-state coordinate.
    NonFiniteOutput,
}

impl fmt::Display for LeakyIntegratorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Process(error) => error.fmt(formatter),
            Self::NonFiniteInput => formatter.write_str("leaky-integrator input must be finite"),
            Self::NonFiniteOutput => formatter.write_str("leaky-integrator result must be finite"),
        }
    }
}

impl core::error::Error for LeakyIntegratorError {}

impl From<ProcessError> for LeakyIntegratorError {
    fn from(error: ProcessError) -> Self {
        Self::Process(error)
    }
}

/// An extent-independent scalar leaky-integrator plan.
///
/// For each coordinate, the pole product is rounded before the fused gain
/// product and addition. With `closed = false`, successful calls hold state:
///
/// ```text
/// y[k] = p * y[k - 1] + g * x[k]
/// g = gain
/// ```
///
/// Complete processing rejects nonfinite input or results before changing
/// caller output or committed history. Each progressive range has the same
/// rejection guarantee for its own output; earlier successful prefixes remain
/// immutable, and history is adopted only by a successful finish.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeakyIntegratorPlan<T> {
    pole: T,
    gain: T,
    closed: bool,
}

impl<T> LeakyIntegratorPlan<T>
where
    T: Float,
{
    /// Prepares a finite signed gain and a pole in `[0, 1]`.
    pub fn new(gain: T, pole: T) -> Result<Self, LeakyIntegratorPlanError> {
        if !gain.is_finite() {
            return Err(LeakyIntegratorPlanError::InvalidGain);
        }
        if !pole.is_finite() || pole < T::zero() || pole > T::one() {
            return Err(LeakyIntegratorPlanError::InvalidPole);
        }

        Ok(Self {
            pole,
            gain,
            closed: true,
        })
    }

    /// Returns the discrete-time pole `p`.
    pub const fn pole(&self) -> T {
        self.pole
    }

    /// Returns the integrator gain `g`.
    pub const fn gain(&self) -> T {
        self.gain
    }
}

impl<T> Default for LeakyIntegratorPlan<T>
where
    T: Float,
{
    /// Prepare the property-contract defaults `gain = 1` and `pole = 0`.
    fn default() -> Self {
        Self {
            pole: T::zero(),
            gain: T::one(),
            closed: true,
        }
    }
}

/// Previous and next mathematical state for one leaky integrator.
///
/// `state` is `y[k-1]`; private `next_state` receives `y[k]`. Complete
/// processing and progressive finish exchange the two buffers only after every
/// coordinate is ready. Progressive abort preserves `state`.
/// Reset restores the scalar construction value in both buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct LeakyIntegratorWorkspace<T> {
    initial_state: T,
    state: Box<[T]>,
    next_state: Box<[T]>,
    processed: usize,
    started: bool,
}

impl<T> LeakyIntegratorWorkspace<T>
where
    T: Float,
{
    /// Prepares two zero-filled state buffers for a positive extent.
    pub fn new(state_length: usize) -> Result<Self, LeakyIntegratorWorkspaceError> {
        if state_length == 0 {
            return Err(LeakyIntegratorWorkspaceError::EmptyState);
        }
        Ok(Self {
            initial_state: T::zero(),
            state: vec![T::zero(); state_length].into_boxed_slice(),
            next_state: vec![T::zero(); state_length].into_boxed_slice(),
            processed: 0,
            started: false,
        })
    }

    /// Prepares two state buffers filled with one initial value.
    pub fn filled(
        state_length: usize,
        initial_state: T,
    ) -> Result<Self, LeakyIntegratorWorkspaceError> {
        if state_length == 0 {
            return Err(LeakyIntegratorWorkspaceError::EmptyState);
        }
        Ok(Self {
            initial_state,
            state: vec![initial_state; state_length].into_boxed_slice(),
            next_state: vec![initial_state; state_length].into_boxed_slice(),
            processed: 0,
            started: false,
        })
    }

    /// Returns the committed mathematical state `y[k-1]`.
    pub fn state(&self) -> &[T] {
        &self.state
    }

    /// Returns the prepared state extent.
    pub const fn len(&self) -> usize {
        self.state.len()
    }

    /// Returns `false`; a prepared workspace always has positive extent.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Restores current and next mathematical state to the construction value.
    ///
    /// The owning block performs reset only at a complete input boundary. It
    /// also terminates any private progressive call sequence.
    pub fn reset(&mut self) {
        self.state.fill(self.initial_state);
        self.next_state.fill(self.initial_state);
        self.processed = 0;
        self.started = false;
    }

    fn validate(&self, output_len: usize, input_len: usize) -> Result<(), ProcessError> {
        if input_len != self.len() {
            return Err(ProcessError::InputExtent {
                expected: self.len(),
                actual: input_len,
            });
        }
        if output_len != self.len() {
            return Err(ProcessError::OutputExtent {
                expected: self.len(),
                actual: output_len,
            });
        }
        Ok(())
    }

    pub(crate) fn commit(&mut self) {
        self.commit_next_state();
    }

    pub(crate) fn discard(&mut self) {
        self.next_state.copy_from_slice(&self.state);
        self.processed = 0;
        self.started = false;
    }

    fn commit_next_state(&mut self) {
        mem::swap(&mut self.state, &mut self.next_state);
        self.processed = 0;
        self.started = false;
    }
}

impl<T> LeakyIntegratorPlan<T>
where
    T: Float,
{
    fn integrate(
        &self,
        output: &mut [T],
        workspace: &mut LeakyIntegratorWorkspace<T>,
        input: &[T],
        workrange: Range<usize>,
    ) -> Result<(), LeakyIntegratorError> {
        if input[workrange.clone()]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(LeakyIntegratorError::NonFiniteInput);
        }
        if self.closed {
            for index in workrange.clone() {
                let state_term = self.pole * workspace.state[index];
                workspace.next_state[index] = self.gain.mul_add(input[index], state_term);
            }
        } else {
            workspace.next_state[workrange.clone()]
                .copy_from_slice(&workspace.state[workrange.clone()]);
        }
        if workspace.next_state[workrange.clone()]
            .iter()
            .any(|value| !value.is_finite())
        {
            workspace.next_state[workrange.clone()]
                .copy_from_slice(&workspace.state[workrange.clone()]);
            return Err(LeakyIntegratorError::NonFiniteOutput);
        }
        output[workrange.clone()].copy_from_slice(&workspace.next_state[workrange]);
        Ok(())
    }
}

impl<T: Float> LeakyIntegratorPlan<T> {
    /// Stage one complete sample for an outer scientific transaction.
    pub(crate) fn stage_process(
        &self,
        output: &mut [T],
        workspace: &mut LeakyIntegratorWorkspace<T>,
        input: &[T],
    ) -> Result<(), LeakyIntegratorError> {
        if workspace.started {
            return Err(ProcessError::AlreadyStarted.into());
        }
        workspace.validate(output.len(), input.len())?;
        let extent = workspace.len();
        self.integrate(output, workspace, input, 0..extent)
    }
}

impl<T> AlgorithmPlan for LeakyIntegratorPlan<T>
where
    T: Float + 'static,
{
    type Input<'a> = &'a [T];
    type Output<'a> = &'a mut [T];
    type Workspace = LeakyIntegratorWorkspace<T>;
    type Error = LeakyIntegratorError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        self.stage_process(output, workspace, input)?;
        workspace.commit();
        Ok(())
    }
}

impl<T> InputProgress for LeakyIntegratorPlan<T>
where
    T: Float + 'static,
{
    fn start(&self, workspace: &mut Self::Workspace) -> Result<(), Self::Error> {
        if workspace.started {
            return Err(ProcessError::AlreadyStarted.into());
        }
        workspace.processed = 0;
        workspace.started = true;
        Ok(())
    }

    fn process_range<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
        workrange: Range<usize>,
    ) -> Result<ProgressResult, Self::Error> {
        if !workspace.started {
            return Err(ProcessError::NotStarted.into());
        }
        workspace.validate(output.len(), input.len())?;
        if workrange.is_empty() {
            return Err(ProcessError::EmptyRange.into());
        }
        if workrange.start != workspace.processed {
            return Err((ProcessError::RangeNotContiguous {
                start: workrange.start,
                expected: workspace.processed,
            })
            .into());
        }
        if workrange.end > workspace.len() {
            return Err((ProcessError::RangeOutOfBounds {
                end: workrange.end,
                extent: workspace.len(),
            })
            .into());
        }

        let immutable_count = workrange.end;
        self.integrate(output, workspace, input, workrange)?;
        workspace.processed = immutable_count;
        Ok(ProgressResult::new(immutable_count))
    }

    fn finish<'output>(
        &self,
        _output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
    ) -> Result<(), Self::Error> {
        if !workspace.started {
            return Err(ProcessError::NotStarted.into());
        }
        if workspace.processed != workspace.len() {
            return Err((ProcessError::Incomplete {
                processed: workspace.processed,
                extent: workspace.len(),
            })
            .into());
        }
        workspace.commit_next_state();
        Ok(())
    }

    fn abort(&self, workspace: &mut Self::Workspace) -> Result<(), Self::Error> {
        if !workspace.started {
            return Err(ProcessError::NotStarted.into());
        }
        workspace.processed = 0;
        workspace.started = false;
        Ok(())
    }
}

impl<T> OutputProgress for LeakyIntegratorPlan<T> where T: Float + 'static {}

fn replace_leaky_integrator_properties<T>(
    _plan: &LeakyIntegratorPlan<T>,
    gain: T,
    pole: T,
    closed: bool,
) -> Result<LeakyIntegratorPlan<T>, LeakyIntegratorPlanError>
where
    T: Float,
{
    let mut replacement = LeakyIntegratorPlan::new(gain, pole)?;
    replacement.closed = closed;
    Ok(replacement)
}

macro_rules! impl_leaky_integrator_properties {
    ($scalar:ty, $value:ident) => {
        crate::algorithm_properties! {
            impl LeakyIntegratorPlan<$scalar> {
                plan_error = LeakyIntegratorPlanError;
                replace = replace_leaky_integrator_properties;
                properties {
                    GAIN_PROPERTY => gain: $value {
                        name = "gain";
                        description = "Integrator input gain g";
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
                    CLOSED_PROPERTY => closed: Bool {
                        name = "closed";
                        description = "Close integration; false holds committed state";
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

impl_leaky_integrator_properties!(f32, Float);
impl_leaky_integrator_properties!(f64, Double);

crate::filtergraph_algorithm! {
    /// Portable Float32 leaky-integrator graph declaration.
    pub LeakyIntegratorF32 {
        label = "leaky-integrator-f32";
        configuration LeakyIntegratorF32Configuration {
            extent: usize = GraphRebuild;
            initial_state: f32 = Construction;
            input_schema: String = GraphRebuild;
            output_schema: String = GraphRebuild;
        }
        plan = LeakyIntegratorPlan<f32>;
        properties = runtime;
        dimensions |_plan, workspace| {
            extent = workspace.len();
        }
        inputs {
            input: f32 [extent] = "input", schema = configuration.input_schema.as_str(), optional_sample_metadata = ["chopped_image_kind", "trigger_state", "acquisition_identity", "exposure_duration_nanoseconds"];
        }
        outputs {
            output: f32 [extent] = "output", schema = configuration.output_schema.as_str(),
                metadata_from = "input";
        }
        prepare |configuration| {
            Ok((
                LeakyIntegratorPlan::default(),
                LeakyIntegratorWorkspace::filled(
                    configuration.extent,
                    configuration.initial_state,
                )?,
            ))
        }
        prepare_error = LeakyIntegratorWorkspaceError;
        process |plan, workspace| {
            plan.process(output, workspace, input)
        }
        process_error = LeakyIntegratorError;
        reset |_plan, workspace| {
            workspace.reset();
        }
    }
}

#[cfg(test)]
mod transaction_tests {
    use super::*;
    #[test]
    fn staging_discard_and_progressive_lifecycle_preserve_local_adoption() {
        let plan = LeakyIntegratorPlan::new(0.5_f32, 0.75).unwrap();
        let mut workspace = LeakyIntegratorWorkspace::new(2).unwrap();
        let mut output = [71.0; 2];
        plan.stage_process(&mut output, &mut workspace, &[2.0, -4.0])
            .unwrap();
        assert_eq!(output, [1.0, -2.0]);
        assert_eq!(workspace.state(), [0.0, 0.0]);
        workspace.discard();
        plan.process(&mut output, &mut workspace, &[4.0, 8.0])
            .unwrap();
        assert_eq!(workspace.state(), [2.0, 4.0]);
        plan.stage_process(&mut output, &mut workspace, &[2.0, -2.0])
            .unwrap();
        assert_eq!(output, [2.5, 2.0]);
        assert_eq!(workspace.state(), [2.0, 4.0]);
        workspace.discard();
        plan.start(&mut workspace).unwrap();
        let _ = plan
            .process_range(&mut output, &mut workspace, &[0.0; 2], 0..1)
            .unwrap();
        assert_eq!(
            plan.stage_process(&mut output, &mut workspace, &[0.0; 2]),
            Err(LeakyIntegratorError::Process(ProcessError::AlreadyStarted))
        );
        assert!(workspace.started);
        let _ = plan
            .process_range(&mut output, &mut workspace, &[0.0; 2], 1..2)
            .unwrap();
        plan.finish(&mut output, &mut workspace).unwrap();
        assert_eq!(workspace.state(), [1.5, 3.0]);
    }
}
