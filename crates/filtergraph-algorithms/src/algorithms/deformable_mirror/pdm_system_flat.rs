//! Additive physical-deformable-mirror system-flat compensation.

use alloc::sync::Arc;
use alloc::vec;
use core::fmt;

use num_traits::Float;

use crate::{AlgorithmPlan, NoWorkspace};

/// Invalid prepared PDM system-flat configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmSystemFlatPlanError {
    /// At least one physical actuator is required.
    Empty,
    /// The supplied system flat does not match the physical-actuator extent.
    Extent,
    /// A supplied system-flat value is non-finite.
    NonFiniteConfiguration,
}

impl fmt::Display for PdmSystemFlatPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("PDM system-flat extent must be nonempty"),
            Self::Extent => formatter.write_str("PDM system-flat extent is invalid"),
            Self::NonFiniteConfiguration => formatter.write_str("PDM system flat must be finite"),
        }
    }
}

impl core::error::Error for PdmSystemFlatPlanError {}

/// Invalid PDM system-flat input, output, or result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmSystemFlatError {
    /// The requested command has the wrong physical-actuator extent.
    InputExtent,
    /// The output command has the wrong physical-actuator extent.
    OutputExtent,
    /// The requested command contains a non-finite value.
    NonFiniteInput,
    /// Adding the system flat produced a non-finite command.
    NonFiniteOutput,
}

impl fmt::Display for PdmSystemFlatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => formatter.write_str("requested PDM command extent is invalid"),
            Self::OutputExtent => formatter.write_str("system-flat PDM command extent is invalid"),
            Self::NonFiniteInput => {
                formatter.write_str("requested PDM command contains a non-finite value")
            }
            Self::NonFiniteOutput => {
                formatter.write_str("PDM system-flat addition produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for PdmSystemFlatError {}

/// Prepared additive physical-deformable-mirror system flat.
#[derive(Clone, Debug, PartialEq)]
pub struct PdmSystemFlatPlan<T> {
    system_flat: Arc<[T]>,
}

impl<T> PdmSystemFlatPlan<T>
where
    T: Float,
{
    /// Validates and retains one system flat in physical-actuator order.
    pub fn new(system_flat: Arc<[T]>) -> Result<Self, PdmSystemFlatPlanError> {
        if system_flat.is_empty() {
            return Err(PdmSystemFlatPlanError::Empty);
        }
        if system_flat.iter().any(|value| !value.is_finite()) {
            return Err(PdmSystemFlatPlanError::NonFiniteConfiguration);
        }
        Ok(Self { system_flat })
    }

    /// Returns the physical-actuator extent.
    pub fn len(&self) -> usize {
        self.system_flat.len()
    }

    /// Returns whether the physical-actuator extent is zero.
    pub fn is_empty(&self) -> bool {
        self.system_flat.is_empty()
    }

    /// Returns the prepared system flat in physical-actuator order.
    pub fn system_flat(&self) -> &[T] {
        &self.system_flat
    }

    /// Replaces the system flat while preserving the physical-actuator extent.
    pub fn with_system_flat(&self, system_flat: Arc<[T]>) -> Result<Self, PdmSystemFlatPlanError> {
        if system_flat.len() != self.len() {
            return Err(PdmSystemFlatPlanError::Extent);
        }
        Self::new(system_flat)
    }
}

impl<T> AlgorithmPlan for PdmSystemFlatPlan<T>
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
    type Error = PdmSystemFlatError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _: &mut Self::Workspace,
        requested: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if requested.len() != self.len() {
            return Err(PdmSystemFlatError::InputExtent);
        }
        if output.len() != self.len() {
            return Err(PdmSystemFlatError::OutputExtent);
        }
        for (&requested_value, &flat_value) in requested.iter().zip(self.system_flat.iter()) {
            if !requested_value.is_finite() {
                return Err(PdmSystemFlatError::NonFiniteInput);
            }
            if !(requested_value + flat_value).is_finite() {
                return Err(PdmSystemFlatError::NonFiniteOutput);
            }
        }
        for (destination, (&requested_value, &flat_value)) in output
            .iter_mut()
            .zip(requested.iter().zip(self.system_flat.iter()))
        {
            *destination = requested_value + flat_value;
        }
        Ok(())
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 additive PDM system-flat declaration.
    pub PdmSystemFlatF32 {
        label = "pdm-system-flat-f32";
        configuration PdmSystemFlatF32Configuration {
            actuator_count: usize = Construction;
        }
        plan = PdmSystemFlatPlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            actuator_count = plan.len();
        }
        inputs {
            requested: f32 [actuator_count] = "requested",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1;
        }
        outputs {
            requested_with_system_flat: f32 [actuator_count] = "requested-with-system-flat",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1,
                metadata_from = "requested";
        }
        parameters {
            system_flat: f32 [actuator_count] = "system-flat",
                schema = crate::schemas::PDM_SYSTEM_FLAT_V1,
                replace |plan, values| {
                    plan.with_system_flat(Arc::from(values))
                };
        }
        parameter_error = PdmSystemFlatPlanError;
        prepare |configuration| {
            Ok((
                PdmSystemFlatPlan::new(Arc::from(vec![0.0; configuration.actuator_count]))?,
                NoWorkspace,
            ))
        }
        prepare_error = PdmSystemFlatPlanError;
        process |plan, workspace| {
            plan.process(requested_with_system_flat, workspace, requested)
        }
        process_error = PdmSystemFlatError;
    }
}
