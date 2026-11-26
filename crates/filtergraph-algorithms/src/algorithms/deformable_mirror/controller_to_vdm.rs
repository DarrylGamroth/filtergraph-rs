//! Projection and sum of controller values in one common VDM coordinate space.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::algorithms::linear_algebra::multiply_row_major;
use crate::{AlgorithmPlan, MatrixElement};

use super::controller_projection::{
    ControllerProjection, ControllerProjectionError, ControllerProjectionKind,
    ControllerToVdmProjection, validate_controller_to_vdm_projections,
};

/// Invalid controller-to-VDM input, output, or result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControllerToVdmError {
    /// An input vector has the wrong extent.
    InputExtent,
    /// An output vector has the wrong extent.
    OutputExtent,
    /// The workspace was prepared for an incompatible VDM extent.
    WorkspaceExtent,
    /// An input vector contains a non-finite value.
    NonFiniteInput,
    /// The projected command sum is non-finite.
    NonFiniteOutput,
}

impl fmt::Display for ControllerToVdmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => formatter.write_str("controller-to-VDM input extent is invalid"),
            Self::OutputExtent => formatter.write_str("controller-to-VDM output extent is invalid"),
            Self::WorkspaceExtent => {
                formatter.write_str("controller-to-VDM workspace extent is invalid")
            }
            Self::NonFiniteInput => formatter.write_str("controller commands must be finite"),
            Self::NonFiniteOutput => {
                formatter.write_str("controller-to-VDM projection produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for ControllerToVdmError {}

/// Prepared controller projections into one common VDM command space.
#[derive(Clone, Debug, PartialEq)]
pub struct ControllerToVdmPlan<T> {
    vdm_size: usize,
    controller_command_extent: usize,
    controllers: Arc<[ControllerToVdmProjection<T>]>,
}

impl<T> ControllerToVdmPlan<T>
where
    T: MatrixElement,
{
    /// Prepare the forward direction from explicit projection pairs.
    pub fn new(
        vdm_size: usize,
        controllers: Arc<[ControllerProjection<T>]>,
    ) -> Result<Self, ControllerProjectionError> {
        let controllers = controllers
            .iter()
            .map(ControllerProjection::controller_to_vdm_projection)
            .collect::<Vec<_>>()
            .into();
        Self::from_controller_to_vdm_projections(vdm_size, controllers)
    }

    /// Prepare a fixed topology from only controller-to-VDM projections.
    pub fn from_controller_to_vdm_projections(
        vdm_size: usize,
        controllers: Arc<[ControllerToVdmProjection<T>]>,
    ) -> Result<Self, ControllerProjectionError> {
        let controller_command_extent =
            validate_controller_to_vdm_projections(vdm_size, &controllers)?;
        Ok(Self {
            vdm_size,
            controller_command_extent,
            controllers,
        })
    }

    /// Returns the fixed number of controller inputs.
    pub fn controller_count(&self) -> usize {
        self.controllers.len()
    }

    /// Returns the common VDM extent.
    pub const fn vdm_size(&self) -> usize {
        self.vdm_size
    }

    /// Return the ordered controller-to-VDM projections retained by this plan.
    pub fn controller_to_vdm_projections(&self) -> &[ControllerToVdmProjection<T>] {
        &self.controllers
    }

    /// Returns one controller command extent.
    pub fn controller_command_size(&self, index: usize) -> Option<usize> {
        self.controllers
            .get(index)
            .map(ControllerToVdmProjection::command_size)
    }

    /// Returns the concatenated controller command extent.
    pub fn controller_command_extent(&self) -> usize {
        self.controller_command_extent
    }

    /// Allocates fixed scratch for the projected sum.
    pub fn workspace(&self) -> ControllerToVdmWorkspace<T> {
        ControllerToVdmWorkspace {
            projected_command: vec![T::zero(); self.vdm_size].into_boxed_slice(),
        }
    }
}

impl<T> AlgorithmPlan for ControllerToVdmPlan<T>
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
    type Workspace = ControllerToVdmWorkspace<T>;
    type Error = ControllerToVdmError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        controller_commands: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if controller_commands.len() != self.controller_command_extent {
            return Err(ControllerToVdmError::InputExtent);
        }
        if output.len() != self.vdm_size {
            return Err(ControllerToVdmError::OutputExtent);
        }
        if workspace.projected_command.len() != self.vdm_size {
            return Err(ControllerToVdmError::WorkspaceExtent);
        }
        if controller_commands.iter().any(|value| !value.is_finite()) {
            return Err(ControllerToVdmError::NonFiniteInput);
        }
        output.fill(T::zero());
        let mut offset = 0;
        for controller in self.controllers.iter() {
            let end = offset + controller.command_size;
            let command = &controller_commands[offset..end];
            let (sum, contribution) = match controller.kind.for_current_float_mode() {
                ControllerProjectionKind::Identity => (&mut *output, command),
                #[cfg(all(feature = "blas", target_arch = "x86_64"))]
                ControllerProjectionKind::Contiguous(start) => {
                    if self.vdm_size > controller.command_size {
                        (&mut output[start..start + controller.command_size], command)
                    } else {
                        (&mut *output, &command[start..start + self.vdm_size])
                    }
                }
                ControllerProjectionKind::RowMajor => {
                    multiply_row_major(
                        &controller.controller_to_vdm,
                        self.vdm_size,
                        controller.command_size,
                        command,
                        &mut workspace.projected_command,
                    );
                    (&mut *output, &*workspace.projected_command)
                }
            };
            for (sum, contribution) in sum.iter_mut().zip(contribution) {
                *sum = *sum + *contribution;
            }
            offset = end;
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(ControllerToVdmError::NonFiniteOutput);
        }
        Ok(())
    }
}

/// Fixed scratch for one controller-to-VDM operation owner.
#[derive(Clone, Debug, PartialEq)]
pub struct ControllerToVdmWorkspace<T> {
    projected_command: Box<[T]>,
}

crate::filtergraph_algorithm! {
    /// Portable Float32 controller-command to active-VDM declaration.
    pub ControllerToVdmF32 {
        label = "controller-to-vdm-f32";
        configuration ControllerToVdmF32Configuration {
            controller_command_count: usize = GraphRebuild;
            active_vdm_count: usize = GraphRebuild;
        }
        plan = ControllerToVdmPlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            controller_command_extent = plan.controller_command_extent();
            vdm_size = plan.vdm_size();
        }
        inputs {
            controller_command: f32 [controller_command_extent] = "controller-command",
                schema = crate::schemas::CONTROLLER_COMMAND_V1;
        }
        outputs {
            vdm_command: f32 [vdm_size] = "vdm-command",
                schema = crate::schemas::VDM_COMMAND_V1,
                metadata_from = "controller-command";
        }
        parameters {
            controller_to_vdm: f32 [vdm_size, controller_command_extent] = "controller-to-vdm",
                schema = crate::schemas::CONTROLLER_TO_VDM_PROJECTION_V1,
                replace |plan, values| {
                    let projection = ControllerToVdmProjection::new(
                        plan.controller_command_extent(),
                        Arc::from(values),
                    )?;
                    ControllerToVdmPlan::from_controller_to_vdm_projections(
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
                configuration.active_vdm_count,
                configuration.controller_command_count,
            )
                .ok_or(ControllerProjectionError::ExtentOverflow)?;
            let projection = ControllerToVdmProjection::new(
                configuration.controller_command_count,
                Arc::from(vec![0.0; projection_extent]),
            )?;
            let plan = ControllerToVdmPlan::from_controller_to_vdm_projections(
                configuration.active_vdm_count,
                Arc::from([projection]),
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = ControllerProjectionError;
        process |plan, workspace| {
            plan.process(vdm_command, workspace, controller_command)
        }
        process_error = ControllerToVdmError;
    }
}
