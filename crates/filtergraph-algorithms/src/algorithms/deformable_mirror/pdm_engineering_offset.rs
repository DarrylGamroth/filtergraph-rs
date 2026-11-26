//! Additive physical-deformable-mirror engineering-offset compensation.

use alloc::sync::Arc;
use alloc::vec;
use core::fmt;

use num_traits::Float;

use crate::{AlgorithmPlan, NoWorkspace};

/// Invalid prepared PDM engineering-offset configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmEngineeringOffsetPlanError {
    /// At least one physical actuator is required.
    Empty,
    /// The supplied engineering offset does not match the physical-actuator extent.
    Extent,
    /// A supplied engineering-offset value is non-finite.
    NonFiniteConfiguration,
}

impl fmt::Display for PdmEngineeringOffsetPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("PDM engineering-offset extent must be nonempty"),
            Self::Extent => formatter.write_str("PDM engineering-offset extent is invalid"),
            Self::NonFiniteConfiguration => {
                formatter.write_str("PDM engineering offset must be finite")
            }
        }
    }
}

impl core::error::Error for PdmEngineeringOffsetPlanError {}

/// Invalid PDM engineering-offset input, output, or result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmEngineeringOffsetError {
    /// The requested command has the wrong physical-actuator extent.
    InputExtent,
    /// The output command has the wrong physical-actuator extent.
    OutputExtent,
    /// The current-offset output has the wrong physical-actuator extent.
    CurrentOffsetExtent,
    /// The requested command contains a non-finite value.
    NonFiniteInput,
    /// Adding the engineering offset produced a non-finite command.
    NonFiniteOutput,
}

impl fmt::Display for PdmEngineeringOffsetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => formatter.write_str("requested PDM command extent is invalid"),
            Self::OutputExtent => {
                formatter.write_str("engineering-offset PDM command extent is invalid")
            }
            Self::CurrentOffsetExtent => {
                formatter.write_str("current PDM offset extent is invalid")
            }
            Self::NonFiniteInput => {
                formatter.write_str("requested PDM command contains a non-finite value")
            }
            Self::NonFiniteOutput => {
                formatter.write_str("PDM engineering-offset addition produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for PdmEngineeringOffsetError {}

/// Mutable destinations for a successful engineering-offset frame.
pub struct PdmEngineeringOffsetOutput<'a, T> {
    /// Requested command plus the adopted engineering offset.
    pub requested_with_offset: &'a mut [T],
    /// Adopted parameter values in physical-actuator order.
    pub current_offset: &'a mut [T],
}

pub(super) fn offset_sum<T: Float>(requested: T, offset: T) -> T {
    if offset.is_zero() {
        requested
    } else {
        requested + offset
    }
}

/// Prepared additive physical-deformable-mirror engineering offset.
#[derive(Clone, Debug, PartialEq)]
pub struct PdmEngineeringOffsetPlan<T> {
    engineering_offset: Arc<[T]>,
}

impl<T> PdmEngineeringOffsetPlan<T>
where
    T: Float,
{
    /// Validates and retains one engineering offset in physical-actuator order.
    pub fn new(engineering_offset: Arc<[T]>) -> Result<Self, PdmEngineeringOffsetPlanError> {
        if engineering_offset.is_empty() {
            return Err(PdmEngineeringOffsetPlanError::Empty);
        }
        if engineering_offset.iter().any(|value| !value.is_finite()) {
            return Err(PdmEngineeringOffsetPlanError::NonFiniteConfiguration);
        }
        Ok(Self { engineering_offset })
    }

    /// Returns the physical-actuator extent.
    pub fn len(&self) -> usize {
        self.engineering_offset.len()
    }

    /// Returns whether the physical-actuator extent is zero.
    pub fn is_empty(&self) -> bool {
        self.engineering_offset.is_empty()
    }

    /// Returns the prepared engineering offset in physical-actuator order.
    pub fn engineering_offset(&self) -> &[T] {
        &self.engineering_offset
    }

    /// Replaces the engineering offset while preserving the physical-actuator extent.
    pub fn with_engineering_offset(
        &self,
        engineering_offset: Arc<[T]>,
    ) -> Result<Self, PdmEngineeringOffsetPlanError> {
        if engineering_offset.len() != self.len() {
            return Err(PdmEngineeringOffsetPlanError::Extent);
        }
        Self::new(engineering_offset)
    }
}

impl<T> AlgorithmPlan for PdmEngineeringOffsetPlan<T>
where
    T: Float + 'static,
{
    type Input<'a>
        = &'a [T]
    where
        Self: 'a;
    type Output<'a>
        = PdmEngineeringOffsetOutput<'a, T>
    where
        Self: 'a;
    type Workspace = NoWorkspace;
    type Error = PdmEngineeringOffsetError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _: &mut Self::Workspace,
        requested: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if requested.len() != self.len() {
            return Err(PdmEngineeringOffsetError::InputExtent);
        }
        if output.requested_with_offset.len() != self.len() {
            return Err(PdmEngineeringOffsetError::OutputExtent);
        }
        if output.current_offset.len() != self.len() {
            return Err(PdmEngineeringOffsetError::CurrentOffsetExtent);
        }
        for (&requested_value, &offset_value) in
            requested.iter().zip(self.engineering_offset.iter())
        {
            if !requested_value.is_finite() {
                return Err(PdmEngineeringOffsetError::NonFiniteInput);
            }
            if !offset_sum(requested_value, offset_value).is_finite() {
                return Err(PdmEngineeringOffsetError::NonFiniteOutput);
            }
        }
        for (destination, (&requested_value, &offset_value)) in output
            .requested_with_offset
            .iter_mut()
            .zip(requested.iter().zip(self.engineering_offset.iter()))
        {
            *destination = offset_sum(requested_value, offset_value);
        }
        output
            .current_offset
            .copy_from_slice(&self.engineering_offset);
        Ok(())
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 PDM engineering-offset addition in µm wavefront.
    /// Connect requested-with-offset to the existing PDM command limiter.
    pub PdmEngineeringOffsetF32 {
        label = "pdm-engineering-offset-f32";
        configuration PdmEngineeringOffsetF32Configuration {
            actuator_count: usize = Construction;
        }
        plan = PdmEngineeringOffsetPlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            actuator_count = plan.len();
        }
        inputs {
            requested: f32 [actuator_count] = "requested",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1;
        }
        outputs {
            requested_with_offset: f32 [actuator_count] = "requested-with-offset",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1,
                metadata_from = "requested";
            current_offset: f32 [actuator_count] = "current-offset",
                schema = crate::schemas::PDM_ENGINEERING_OFFSET_V1,
                metadata_from = "requested";
        }
        parameters {
            engineering_offset: f32 [actuator_count] = "engineering-offset",
                schema = crate::schemas::PDM_ENGINEERING_OFFSET_V1,
                replace |plan, values| {
                    plan.with_engineering_offset(Arc::from(values))
                };
        }
        parameter_error = PdmEngineeringOffsetPlanError;
        prepare |configuration| {
            Ok((
                PdmEngineeringOffsetPlan::new(Arc::from(vec![0.0; configuration.actuator_count]))?,
                NoWorkspace,
            ))
        }
        prepare_error = PdmEngineeringOffsetPlanError;
        process |plan, workspace| {
            plan.process(PdmEngineeringOffsetOutput { requested_with_offset, current_offset }, workspace, requested)
        }
        process_error = PdmEngineeringOffsetError;
    }
}
