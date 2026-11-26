//! Deformable-mirror actuator-coordinate transformations, constraints, and feedback.
//!
//! These operations project controller values into a common VDM, decompose
//! that VDM into one or more PDM actuator orders, condition each physical
//! actuator vector, and map resulting constraint feedback back through PDM,
//! VDM, and controller coordinates. Controller-state evolution, value
//! composition, transport, device timing, and device I/O remain outside this
//! module.
//!
//! Canonical VDM and PDM actuator-coordinate values are µm of wavefront. A
//! physical-device adapter owns conversion to mirror-surface displacement,
//! voltage, DAC codes, or other device-native units.

fn declared_f32_matrix_extent(rows: usize, columns: usize) -> Option<usize> {
    u32::try_from(rows).ok()?;
    u32::try_from(columns).ok()?;
    let elements = rows.checked_mul(columns)?;
    let bytes = elements.checked_mul(core::mem::size_of::<f32>())?;
    (bytes <= u32::MAX as usize).then_some(elements)
}

mod controller_projection;
mod controller_to_vdm;
mod inter_actuator_stroke_limit;
mod pdm_command;
mod pdm_command_power_limit;
mod pdm_engineering_offset;
mod pdm_feedback_to_vdm;
mod pdm_projection;
mod pdm_stroke_optimization;
mod pdm_system_flat;
mod vdm_actuator_partition;
mod vdm_feedback_to_controller;
mod vdm_to_pdm;

pub use controller_projection::{
    ControllerProjection, ControllerProjectionError, ControllerToVdmProjection,
    VdmToControllerProjection,
};
pub use controller_to_vdm::{
    ControllerToVdmError, ControllerToVdmF32, ControllerToVdmF32Configuration, ControllerToVdmPlan,
    ControllerToVdmWorkspace,
};
pub use inter_actuator_stroke_limit::{
    InterActuatorStrokeLimit, InterActuatorStrokeLimitError, InterActuatorStrokeLimitF32,
    InterActuatorStrokeLimitF32Configuration, InterActuatorStrokeLimitPlan,
    InterActuatorStrokeLimitPlanError,
};
pub use pdm_command::{
    PdmCommandError, PdmCommandF32, PdmCommandF32Configuration, PdmCommandF32PreparationError,
    PdmCommandF32Workspace, PdmCommandInput, PdmCommandOutput, PdmCommandPlan, PdmCommandPlanError,
    PdmCommandWorkspace, PdmCommandWorkspaceError, VectorConditioningF32,
    VectorConditioningF32Configuration, VectorConditioningPreparationError,
};
pub use pdm_command_power_limit::{
    PdmCommandPowerLimitError, PdmCommandPowerLimitF32, PdmCommandPowerLimitF32Configuration,
    PdmCommandPowerLimitPlan, PdmCommandPowerLimitPlanError,
};
pub use pdm_engineering_offset::{
    PdmEngineeringOffsetError, PdmEngineeringOffsetF32, PdmEngineeringOffsetF32Configuration,
    PdmEngineeringOffsetOutput, PdmEngineeringOffsetPlan, PdmEngineeringOffsetPlanError,
};
pub use pdm_feedback_to_vdm::{
    PdmFeedbackToVdmError, PdmFeedbackToVdmF32, PdmFeedbackToVdmF32Configuration,
    PdmFeedbackToVdmPlan, PdmFeedbackToVdmWorkspace, VdmTruncationArtifacts, VdmTruncationStorage,
};
pub use pdm_projection::{
    PdmProjection, PdmProjectionArtifacts, PdmProjectionError, PdmProjectionStorage,
    PdmToVdmProjection, VdmToPdmProjection,
};
pub use pdm_stroke_optimization::{
    PdmStrokeOptimizationError, PdmStrokeOptimizationF32, PdmStrokeOptimizationF32Configuration,
    PdmStrokeOptimizationPlan, PdmStrokeOptimizationPlanError, PdmStrokeOptimizationWorkspace,
};
pub use pdm_system_flat::{
    PdmSystemFlatError, PdmSystemFlatF32, PdmSystemFlatF32Configuration, PdmSystemFlatPlan,
    PdmSystemFlatPlanError,
};
pub use vdm_actuator_partition::{
    VdmActuatorPartition, VdmActuatorPartitionError, VdmActuatorRole,
};
pub use vdm_feedback_to_controller::{
    VdmFeedbackToControllerError, VdmFeedbackToControllerF32,
    VdmFeedbackToControllerF32Configuration, VdmFeedbackToControllerPlan,
};
pub use vdm_to_pdm::{
    VdmExtrapolationArtifacts, VdmExtrapolationStorage, VdmToPdmError, VdmToPdmF32,
    VdmToPdmF32Configuration, VdmToPdmPlan, VdmToPdmWorkspace,
};

mod vdm_shape_offset;
pub use vdm_shape_offset::*;
