//! Shape-preserving physical-DM actuator-value power projection.

use core::fmt;

use num_traits::Float;

use crate::{AlgorithmPlan, NoWorkspace};

/// Invalid prepared PDM command-power configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmCommandPowerLimitPlanError {
    /// At least one physical actuator is required.
    Empty,
    /// The maximum command power is non-finite or negative.
    InvalidMaximum,
}

impl fmt::Display for PdmCommandPowerLimitPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("PDM command-power extent must be nonempty"),
            Self::InvalidMaximum => {
                formatter.write_str("maximum PDM command power must be finite and nonnegative")
            }
        }
    }
}

impl core::error::Error for PdmCommandPowerLimitPlanError {}

/// Invalid PDM command-power input or output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmCommandPowerLimitError {
    /// The input vector has the wrong PDM extent.
    InputExtent,
    /// The output vector has the wrong PDM extent.
    OutputExtent,
    /// The input vector contains a non-finite value.
    NonFiniteInput,
}

impl fmt::Display for PdmCommandPowerLimitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => formatter.write_str("PDM command-power input extent is invalid"),
            Self::OutputExtent => formatter.write_str("PDM command-power output extent is invalid"),
            Self::NonFiniteInput => formatter.write_str("PDM command-power input must be finite"),
        }
    }
}

impl core::error::Error for PdmCommandPowerLimitError {}

/// Prepared projection onto a total PDM command-power ball.
///
/// Command power is `sum(command[i] * command[i])`. A command outside the
/// prepared bound is multiplied by one common gain, preserving its direction
/// in physical-actuator coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PdmCommandPowerLimitPlan<T> {
    pdm_size: usize,
    max_command_power: T,
    max_norm: T,
}

impl<T> PdmCommandPowerLimitPlan<T>
where
    T: Float,
{
    /// Prepares a nonnegative sum-of-squares command-power bound.
    pub fn new(
        pdm_size: usize,
        max_command_power: T,
    ) -> Result<Self, PdmCommandPowerLimitPlanError> {
        if pdm_size == 0 {
            return Err(PdmCommandPowerLimitPlanError::Empty);
        }
        if !max_command_power.is_finite() || max_command_power < T::zero() {
            return Err(PdmCommandPowerLimitPlanError::InvalidMaximum);
        }
        Ok(Self {
            pdm_size,
            max_command_power,
            max_norm: max_command_power.sqrt(),
        })
    }

    /// Returns the physical-actuator extent.
    pub fn len(&self) -> usize {
        self.pdm_size
    }

    /// Returns whether the physical-actuator extent is zero.
    pub fn is_empty(&self) -> bool {
        self.pdm_size == 0
    }

    /// Returns the prepared sum-of-squares command-power bound.
    pub fn max_command_power(&self) -> T {
        self.max_command_power
    }
}

impl<T> AlgorithmPlan for PdmCommandPowerLimitPlan<T>
where
    T: Float + 'static,
{
    type Input<'a>
        = &'a [T]
    where
        Self: 'a;
    type Output<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type Workspace = NoWorkspace;
    type Error = PdmCommandPowerLimitError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if input.len() != self.pdm_size {
            return Err(PdmCommandPowerLimitError::InputExtent);
        }
        if output.len() != self.pdm_size {
            return Err(PdmCommandPowerLimitError::OutputExtent);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(PdmCommandPowerLimitError::NonFiniteInput);
        }
        let (scale, sum_squares) = scaled_l2_components(input);
        let gain = if scale == T::zero() {
            T::one()
        } else if self.max_norm == T::zero() {
            T::zero()
        } else {
            let normalized_norm = sum_squares.sqrt();
            if scale <= self.max_norm / normalized_norm {
                T::one()
            } else {
                (self.max_norm / scale) / normalized_norm
            }
        };
        for (destination, source) in output.iter_mut().zip(input) {
            *destination = *source * gain;
        }
        Ok(())
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 PDM command-power projection declaration.
    pub PdmCommandPowerLimitF32 {
        label = "pdm-command-power-limit-f32";
        configuration PdmCommandPowerLimitF32Configuration {
            actuator_count: usize = GraphRebuild;
            max_command_power: f32 = Construction;
        }
        plan = PdmCommandPowerLimitPlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            actuator_count = plan.len();
        }
        inputs {
            requested: f32 [actuator_count] = "requested",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1;
        }
        outputs {
            power_limited: f32 [actuator_count] = "power-limited",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1,
                metadata_from = "requested";
        }
        prepare |configuration| {
            Ok((
                PdmCommandPowerLimitPlan::new(
                    configuration.actuator_count,
                    configuration.max_command_power,
                )?,
                NoWorkspace,
            ))
        }
        prepare_error = PdmCommandPowerLimitPlanError;
        process |plan, workspace| {
            plan.process(power_limited, workspace, requested)
        }
        process_error = PdmCommandPowerLimitError;
    }
}

fn scaled_l2_components<T>(values: &[T]) -> (T, T)
where
    T: Float,
{
    let mut scale = T::zero();
    let mut sum_squares = T::one();
    for value in values {
        let magnitude = value.abs();
        if magnitude == T::zero() {
            continue;
        }
        if scale < magnitude {
            let ratio = scale / magnitude;
            sum_squares = T::one() + sum_squares * ratio * ratio;
            scale = magnitude;
        } else {
            let ratio = magnitude / scale;
            sum_squares = sum_squares + ratio * ratio;
        }
    }
    (scale, sum_squares)
}
