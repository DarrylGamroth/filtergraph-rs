//! Projection of common-VDM coordinate constraint feedback into controller coordinates.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::algorithms::linear_algebra::multiply_row_major;
use crate::{AlgorithmPlan, MatrixElement, NoWorkspace};

use super::controller_projection::{
    ControllerProjection, ControllerProjectionError, ControllerProjectionKind,
    VdmToControllerProjection, validate_vdm_to_controller_projections,
};

/// Invalid VDM-to-controller feedback input, output, or result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VdmFeedbackToControllerError {
    /// The common-VDM feedback has the wrong extent.
    InputExtent,
    /// The concatenated controller feedback has the wrong extent.
    OutputExtent,
    /// The common-VDM feedback contains a non-finite value.
    NonFiniteInput,
    /// A projected controller feedback vector is non-finite.
    NonFiniteOutput,
}

impl fmt::Display for VdmFeedbackToControllerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => {
                formatter.write_str("VDM-to-controller feedback input extent is invalid")
            }
            Self::OutputExtent => {
                formatter.write_str("VDM-to-controller feedback output extent is invalid")
            }
            Self::NonFiniteInput => formatter.write_str("VDM constraint feedback must be finite"),
            Self::NonFiniteOutput => {
                formatter.write_str("VDM-to-controller feedback produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for VdmFeedbackToControllerError {}

/// Prepared reverse projections from one common VDM into controller coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct VdmFeedbackToControllerPlan<T> {
    vdm_size: usize,
    controller_command_extent: usize,
    controllers: Arc<[VdmToControllerProjection<T>]>,
}

impl<T> VdmFeedbackToControllerPlan<T>
where
    T: MatrixElement,
{
    /// Prepare the reverse direction from explicit projection pairs.
    pub fn new(
        vdm_size: usize,
        controllers: Arc<[ControllerProjection<T>]>,
    ) -> Result<Self, ControllerProjectionError> {
        let controllers = controllers
            .iter()
            .map(ControllerProjection::vdm_to_controller_projection)
            .collect::<Vec<_>>()
            .into();
        Self::from_vdm_to_controller_projections(vdm_size, controllers)
    }

    /// Prepare a fixed topology from only VDM-to-controller projections.
    pub fn from_vdm_to_controller_projections(
        vdm_size: usize,
        controllers: Arc<[VdmToControllerProjection<T>]>,
    ) -> Result<Self, ControllerProjectionError> {
        let controller_command_extent =
            validate_vdm_to_controller_projections(vdm_size, &controllers)?;
        Ok(Self {
            vdm_size,
            controller_command_extent,
            controllers,
        })
    }

    /// Returns the concatenated controller feedback extent.
    pub fn controller_command_extent(&self) -> usize {
        self.controller_command_extent
    }

    /// Returns the common VDM extent.
    pub const fn vdm_size(&self) -> usize {
        self.vdm_size
    }

    /// Return the ordered VDM-to-controller projections retained by this plan.
    pub fn vdm_to_controller_projections(&self) -> &[VdmToControllerProjection<T>] {
        &self.controllers
    }
}

impl<T> AlgorithmPlan for VdmFeedbackToControllerPlan<T>
where
    T: MatrixElement + 'static,
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
    type Error = VdmFeedbackToControllerError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _workspace: &mut Self::Workspace,
        vdm_constraint_feedback: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if vdm_constraint_feedback.len() != self.vdm_size {
            return Err(VdmFeedbackToControllerError::InputExtent);
        }
        if output.len() != self.controller_command_extent {
            return Err(VdmFeedbackToControllerError::OutputExtent);
        }
        if vdm_constraint_feedback
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(VdmFeedbackToControllerError::NonFiniteInput);
        }
        let mut offset = 0;
        for controller in self.controllers.iter() {
            let end = offset + controller.command_size;
            let result = &mut output[offset..end];
            match controller.kind.for_current_float_mode() {
                ControllerProjectionKind::Identity => {
                    result.copy_from_slice(vdm_constraint_feedback);
                }
                #[cfg(all(feature = "blas", target_arch = "x86_64"))]
                ControllerProjectionKind::Contiguous(start) => {
                    let (result, input) = if self.vdm_size > controller.command_size {
                        (
                            result,
                            &vdm_constraint_feedback[start..start + controller.command_size],
                        )
                    } else {
                        result.fill(T::zero());
                        (
                            &mut result[start..start + self.vdm_size],
                            vdm_constraint_feedback,
                        )
                    };
                    for (output, &input) in result.iter_mut().zip(input) {
                        // GEMV starts each output at positive zero.
                        *output = T::zero() + input;
                    }
                }
                ControllerProjectionKind::RowMajor => multiply_row_major(
                    &controller.vdm_to_controller,
                    controller.command_size,
                    self.vdm_size,
                    vdm_constraint_feedback,
                    result,
                ),
            }
            offset = end;
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(VdmFeedbackToControllerError::NonFiniteOutput);
        }
        Ok(())
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 active-VDM feedback to controller-coordinate declaration.
    pub VdmFeedbackToControllerF32 {
        label = "vdm-feedback-to-controller-f32";
        configuration VdmFeedbackToControllerF32Configuration {
            controller_command_count: usize = GraphRebuild;
            active_vdm_count: usize = GraphRebuild;
        }
        plan = VdmFeedbackToControllerPlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            controller_command_extent = plan.controller_command_extent();
            vdm_size = plan.vdm_size();
        }
        inputs {
            vdm_constraint_feedback: f32 [vdm_size] = "vdm-constraint-feedback",
                schema = crate::schemas::VDM_CONSTRAINT_FEEDBACK_V1;
        }
        outputs {
            controller_constraint_feedback: f32 [controller_command_extent] = "controller-constraint-feedback",
                schema = crate::schemas::CONTROLLER_CONSTRAINT_FEEDBACK_V1,
                metadata_from = "vdm-constraint-feedback";
        }
        parameters {
            vdm_to_controller: f32 [controller_command_extent, vdm_size] = "vdm-to-controller",
                schema = crate::schemas::VDM_TO_CONTROLLER_PROJECTION_V1,
                replace |plan, values| {
                    let projection = VdmToControllerProjection::new(
                        plan.controller_command_extent(),
                        Arc::from(values),
                    )?;
                    VdmFeedbackToControllerPlan::from_vdm_to_controller_projections(
                        plan.vdm_size(),
                        Arc::from([projection]),
                    )
                };
        }
        parameter_error = ControllerProjectionError;
        prepare |configuration| {
            if configuration.controller_command_count == 0 || configuration.active_vdm_count == 0 {
                return Err(ControllerProjectionError::EmptyDimension);
            }
            let projection_extent = super::declared_f32_matrix_extent(
                configuration.controller_command_count,
                configuration.active_vdm_count,
            )
                .ok_or(ControllerProjectionError::ExtentOverflow)?;
            let projection = VdmToControllerProjection::new(
                configuration.controller_command_count,
                Arc::from(vec![0.0; projection_extent]),
            )?;
            let plan = VdmFeedbackToControllerPlan::from_vdm_to_controller_projections(
                configuration.active_vdm_count,
                Arc::from([projection]),
            )?;
            Ok((plan, NoWorkspace))
        }
        prepare_error = ControllerProjectionError;
        process |plan, workspace| {
            plan.process(
                controller_constraint_feedback,
                workspace,
                vdm_constraint_feedback,
            )
        }
        process_error = VdmFeedbackToControllerError;
    }
}
