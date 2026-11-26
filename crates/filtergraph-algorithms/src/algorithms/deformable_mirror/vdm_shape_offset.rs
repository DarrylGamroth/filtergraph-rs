//! Adopt a physical shape offset in independent VDM coordinates.
use super::{
    declared_f32_matrix_extent, pdm_engineering_offset::offset_sum,
    vdm_to_pdm::full_correction_roles,
};
use crate::algorithms::linear_algebra::multiply_row_major;
use crate::{AlgorithmPlan, VectorLinearCombinationError};
use alloc::{sync::Arc, vec, vec::Vec};
use core::fmt;

/// Invalid shape-calibration dimensions, direct selector, or projected values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VdmShapeOffsetPlanError {
    /// Positive matrix byte extents must fit UInt32, with active count at most full count.
    Dimensions,
    /// Supplied matrix or offset extent differs from the prepared geometry.
    Extent,
    /// Matrices, physical offsets, and projected active offsets must be finite.
    NonFinite,
    /// Each independent coordinate needs a direct unit row in the completion.
    DirectSelector,
}
impl fmt::Display for VdmShapeOffsetPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VDM shape offset: {self:?}")
    }
}
impl core::error::Error for VdmShapeOffsetPlanError {}

/// Immutable calibrated physical-to-independent offset projection.
#[derive(Clone, Debug)]
pub struct VdmShapeOffsetPlan {
    active: usize,
    full: usize,
    physical: usize,
    completion: Arc<[f32]>,
    inverse: Arc<[f32]>,
    projection: Arc<[f32]>,
    controlled_actuators: Arc<[bool]>,
    shape_offset: Vec<f32>,
    active_offset: Vec<f32>,
}
/// Disposable staged independent-command output.
#[derive(Clone, Debug)]
pub struct VdmShapeOffsetWorkspace {
    candidate: Vec<f32>,
}
impl VdmShapeOffsetWorkspace {
    /// Clear scratch, retaining adopted calibration and shape.
    pub fn reset(&mut self) {
        self.candidate.fill(0.0);
    }
}
impl VdmShapeOffsetPlan {
    /// Prepare row-major completion and inverse matrices and a physical shape.
    pub fn new(
        active: usize,
        full: usize,
        physical: usize,
        completion: &[f32],
        inverse: &[f32],
        offset: &[f32],
    ) -> Result<Self, VdmShapeOffsetPlanError> {
        if active == 0 || full == 0 || physical == 0 || active > full {
            return Err(VdmShapeOffsetPlanError::Dimensions);
        }
        let nc =
            declared_f32_matrix_extent(full, active).ok_or(VdmShapeOffsetPlanError::Dimensions)?;
        let ni = declared_f32_matrix_extent(full, physical)
            .ok_or(VdmShapeOffsetPlanError::Dimensions)?;
        if completion.len() != nc || inverse.len() != ni || offset.len() != physical {
            return Err(VdmShapeOffsetPlanError::Extent);
        }
        if completion
            .iter()
            .chain(inverse)
            .chain(offset)
            .any(|v| !v.is_finite())
        {
            return Err(VdmShapeOffsetPlanError::NonFinite);
        }
        let roles = full_correction_roles(active, completion);
        let mut selected = vec![None; active];
        for (row, &role) in roles.iter().enumerate() {
            if role == 1 {
                let column = completion[row * active..(row + 1) * active]
                    .iter()
                    .position(|&v| v == 1.0)
                    .ok_or(VdmShapeOffsetPlanError::DirectSelector)?;
                selected[column] = Some(row);
            }
        }
        let mut projection = Vec::with_capacity(active * physical);
        for row in selected {
            let row = row.ok_or(VdmShapeOffsetPlanError::DirectSelector)?;
            projection.extend_from_slice(&inverse[row * physical..(row + 1) * physical]);
        }
        let controlled_actuators: Arc<[bool]> = (0..physical)
            .map(|column| (0..active).any(|row| projection[row * physical + column] != 0.0))
            .collect();
        let mut active_offset = vec![0.0; active];
        multiply_row_major(&projection, active, physical, offset, &mut active_offset);
        if active_offset.iter().any(|v| !v.is_finite()) {
            return Err(VdmShapeOffsetPlanError::NonFinite);
        }
        Ok(Self {
            active,
            full,
            physical,
            completion: completion.into(),
            inverse: inverse.into(),
            projection: projection.into(),
            controlled_actuators,
            shape_offset: offset.to_vec(),
            active_offset,
        })
    }
    /// Prepare the neutral declaration builder before calibration adoption.
    pub fn neutral(
        active: usize,
        full: usize,
        physical: usize,
    ) -> Result<Self, VdmShapeOffsetPlanError> {
        let nc =
            declared_f32_matrix_extent(full, active).ok_or(VdmShapeOffsetPlanError::Dimensions)?;
        let ni = declared_f32_matrix_extent(full, physical)
            .ok_or(VdmShapeOffsetPlanError::Dimensions)?;
        if active == 0 || full == 0 || physical == 0 || active > full {
            return Err(VdmShapeOffsetPlanError::Dimensions);
        }
        let mut completion = vec![0.0; nc];
        for i in 0..active {
            completion[i * active + i] = 1.0;
        }
        Self::new(
            active,
            full,
            physical,
            &completion,
            &vec![0.0; ni],
            &vec![0.0; physical],
        )
    }
    /// Copy a finite physical shape and project it using the retained calibration.
    pub fn with_shape_offset(&self, offset: &[f32]) -> Result<Self, VdmShapeOffsetPlanError> {
        if offset.len() != self.physical {
            return Err(VdmShapeOffsetPlanError::Extent);
        }
        if offset.iter().any(|v| !v.is_finite()) {
            return Err(VdmShapeOffsetPlanError::NonFinite);
        }
        let mut active_offset = vec![0.0; self.active];
        multiply_row_major(
            &self.projection,
            self.active,
            self.physical,
            offset,
            &mut active_offset,
        );
        if active_offset.iter().any(|v| !v.is_finite()) {
            return Err(VdmShapeOffsetPlanError::NonFinite);
        }
        Ok(Self {
            active: self.active,
            full: self.full,
            physical: self.physical,
            completion: Arc::clone(&self.completion),
            inverse: Arc::clone(&self.inverse),
            projection: Arc::clone(&self.projection),
            controlled_actuators: Arc::clone(&self.controlled_actuators),
            shape_offset: offset.to_vec(),
            active_offset,
        })
    }
    /// Replace the completion and rebuild its direct selector.
    pub fn with_completion(&self, completion: &[f32]) -> Result<Self, VdmShapeOffsetPlanError> {
        Self::new(
            self.active,
            self.full,
            self.physical,
            completion,
            &self.inverse,
            &self.shape_offset,
        )
    }
    /// Replace the calibrated physical-to-full inverse.
    pub fn with_inverse(&self, inverse: &[f32]) -> Result<Self, VdmShapeOffsetPlanError> {
        Self::new(
            self.active,
            self.full,
            self.physical,
            &self.completion,
            inverse,
            &self.shape_offset,
        )
    }
    /// Adopted physical shape, in requested PDM-command units and order.
    pub fn shape_offset(&self) -> &[f32] {
        &self.shape_offset
    }
    /// Independent-coordinate offset selected from the calibrated inverse.
    pub fn active_offset(&self) -> &[f32] {
        &self.active_offset
    }
    /// Physical actuators that contribute to the selected inverse rows.
    pub fn controlled_actuators(&self) -> &[bool] {
        &self.controlled_actuators
    }
    /// Allocate fixed output staging during preparation.
    pub fn workspace(&self) -> VdmShapeOffsetWorkspace {
        VdmShapeOffsetWorkspace {
            candidate: vec![0.0; self.active],
        }
    }
}
impl AlgorithmPlan for VdmShapeOffsetPlan {
    type Input<'a> = &'a [f32];
    type Output<'a> = (&'a mut [f32], &'a mut [f32]);
    type Workspace = VdmShapeOffsetWorkspace;
    type Error = VectorLinearCombinationError;
    fn process<'input, 'output>(
        &self,
        (output, current): Self::Output<'output>,
        workspace: &mut Self::Workspace,
        requested: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if requested.len() != self.active
            || output.len() != self.active
            || current.len() != self.physical
            || workspace.candidate.len() != self.active
        {
            return Err(VectorLinearCombinationError::DimensionMismatch);
        }
        // Input and sum validation is intentionally interleaved in coordinate order.
        for ((candidate, &request), &offset) in workspace
            .candidate
            .iter_mut()
            .zip(requested)
            .zip(&self.active_offset)
        {
            if !request.is_finite() {
                return Err(VectorLinearCombinationError::NonFiniteInput);
            }
            let value = offset_sum(request, offset);
            if !value.is_finite() {
                return Err(VectorLinearCombinationError::NonFiniteOutput);
            }
            *candidate = value;
        }
        output.copy_from_slice(&workspace.candidate);
        current.copy_from_slice(&self.shape_offset);
        Ok(())
    }
}
crate::filtergraph_algorithm! {
    /// Add an adopted physical shape in independent VDM coordinates before completion.
    pub VdmShapeOffsetF32{
        label="vdm-shape-offset-f32";
        configuration VdmShapeOffsetF32Configuration{
            active_vdm_count:usize=GraphRebuild;
            full_vdm_count:usize=GraphRebuild;
            physical_actuator_count:usize=GraphRebuild;
        }
        plan=VdmShapeOffsetPlan;properties=none;
        dimensions|plan,_workspace|{active=plan.active;full=plan.full;physical=plan.physical;}
        inputs{requested:f32[active]="requested",schema=crate::schemas::VDM_COMMAND_V1;}
        outputs{
            output:f32[active]="output",schema=crate::schemas::VDM_COMMAND_V1,metadata_from="requested";
            current_offset:f32[physical]="current-offset",schema=crate::schemas::REQUESTED_PDM_COMMAND_V1,metadata_from="requested";
        }
        parameters{
            shape_offset:f32[physical]="shape-offset",schema=crate::schemas::REQUESTED_PDM_COMMAND_V1,replace|plan,values|{plan.with_shape_offset(values)};
            active_to_full:f32[full,active]="active-to-full",schema=crate::schemas::ACTIVE_TO_FULL_VDM_PROJECTION_V1,replace|plan,values|{plan.with_completion(values)};
            pdm_to_vdm:f32[full,physical]="pdm-to-vdm",schema=crate::schemas::PDM_TO_VDM_PROJECTION_V1,replace|plan,values|{plan.with_inverse(values)};
        }
        parameter_error=VdmShapeOffsetPlanError;
        prepare|configuration|{let plan=VdmShapeOffsetPlan::neutral(configuration.active_vdm_count,configuration.full_vdm_count,configuration.physical_actuator_count)?;let workspace=plan.workspace();Ok((plan,workspace))}
        prepare_error=VdmShapeOffsetPlanError;
        process|plan,workspace|{plan.process((output,current_offset),workspace,requested)}
        process_error=VectorLinearCombinationError;
        reset|_plan,workspace|{workspace.reset();}
    }
}
