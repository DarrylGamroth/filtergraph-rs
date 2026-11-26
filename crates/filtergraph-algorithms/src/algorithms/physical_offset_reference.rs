//! Calibrated physical offset and its matched residual-reference correction.
use crate::{
    AlgorithmPlan, PreparedGemv, VectorLinearCombinationError, VectorLinearCombinationPlan,
    VectorLinearCombinationWorkspace,
};
use alloc::{string::String, sync::Arc, vec, vec::Vec};
use core::fmt;

/// Invalid physical-offset calibration, sign, dimensions or correction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalOffsetReferencePlanError {
    /// Positive Float32 matrix byte extents must fit UInt32.
    Dimensions,
    /// Calibration or offset extent differs from the prepared dimensions.
    Extent,
    /// Calibration, offset, modal coefficients and correction must be finite.
    NonFinite,
    /// The calibrated reference sign must equal +1 or -1.
    ReferenceSign,
}
impl fmt::Display for PhysicalOffsetReferencePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "physical offset reference: {self:?}")
    }
}
impl core::error::Error for PhysicalOffsetReferencePlanError {}

/// One immutable physical offset and correction `reference_sign C(Q offset)`.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalOffsetReferencePlan {
    combination: VectorLinearCombinationPlan,
    physical_to_modes: PreparedGemv<f32>,
    modes_to_reference: PreparedGemv<f32>,
    offset: Vec<f32>,
    correction: Vec<f32>,
    reference_sign: f32,
}
impl PhysicalOffsetReferencePlan {
    /// Prepare neutral factors and offset, retaining the explicit calibrated sign.
    pub fn new(
        residual_count: usize,
        physical_count: usize,
        mode_count: usize,
        reference_sign: f32,
    ) -> Result<Self, PhysicalOffsetReferencePlanError> {
        let q = matrix_extent(mode_count, physical_count)?;
        let c = matrix_extent(residual_count, mode_count)?;
        if reference_sign != 1.0 && reference_sign != -1.0 {
            return Err(PhysicalOffsetReferencePlanError::ReferenceSign);
        }
        let combination = VectorLinearCombinationPlan::new(residual_count, 1.0, 1.0)
            .map_err(|_| PhysicalOffsetReferencePlanError::Dimensions)?;
        Self::calibrated(
            combination,
            PreparedGemv::new(mode_count, physical_count, Arc::from(vec![0.0; q]))
                .map_err(|_| PhysicalOffsetReferencePlanError::Dimensions)?,
            PreparedGemv::new(residual_count, mode_count, Arc::from(vec![0.0; c]))
                .map_err(|_| PhysicalOffsetReferencePlanError::Dimensions)?,
            vec![0.0; physical_count],
            reference_sign,
        )
    }
    fn calibrated(
        combination: VectorLinearCombinationPlan,
        physical_to_modes: PreparedGemv<f32>,
        modes_to_reference: PreparedGemv<f32>,
        offset: Vec<f32>,
        reference_sign: f32,
    ) -> Result<Self, PhysicalOffsetReferencePlanError> {
        if offset.len() != physical_to_modes.columns() {
            return Err(PhysicalOffsetReferencePlanError::Extent);
        }
        if offset.iter().any(|v| !v.is_finite()) {
            return Err(PhysicalOffsetReferencePlanError::NonFinite);
        }
        let mut modes = vec![0.0; physical_to_modes.rows()];
        physical_to_modes
            .multiply(&mut modes, &offset)
            .map_err(|_| PhysicalOffsetReferencePlanError::NonFinite)?;
        let mut correction = vec![0.0; modes_to_reference.rows()];
        modes_to_reference
            .multiply(&mut correction, &modes)
            .map_err(|_| PhysicalOffsetReferencePlanError::NonFinite)?;
        for value in &mut correction {
            *value *= reference_sign;
        }
        if correction.iter().any(|v| !v.is_finite()) {
            return Err(PhysicalOffsetReferencePlanError::NonFinite);
        }
        Ok(Self {
            combination,
            physical_to_modes,
            modes_to_reference,
            offset,
            correction,
            reference_sign,
        })
    }
    /// Adopt a copied finite physical offset and its matched correction.
    pub fn with_offset(&self, offset: &[f32]) -> Result<Self, PhysicalOffsetReferencePlanError> {
        Self::calibrated(
            self.combination.clone(),
            self.physical_to_modes.clone(),
            self.modes_to_reference.clone(),
            offset.to_vec(),
            self.reference_sign,
        )
    }
    /// Adopt a copied row-major physical-to-modal factor Q.
    pub fn with_physical_to_modes(
        &self,
        values: &[f32],
    ) -> Result<Self, PhysicalOffsetReferencePlanError> {
        let factor = replace_factor(&self.physical_to_modes, values)?;
        Self::calibrated(
            self.combination.clone(),
            factor,
            self.modes_to_reference.clone(),
            self.offset.clone(),
            self.reference_sign,
        )
    }
    /// Adopt a copied row-major modal-to-reference factor C.
    pub fn with_modes_to_reference(
        &self,
        values: &[f32],
    ) -> Result<Self, PhysicalOffsetReferencePlanError> {
        let factor = replace_factor(&self.modes_to_reference, values)?;
        Self::calibrated(
            self.combination.clone(),
            self.physical_to_modes.clone(),
            factor,
            self.offset.clone(),
            self.reference_sign,
        )
    }
    /// Calibrated row-major factor Q.
    pub fn physical_to_modes(&self) -> &PreparedGemv<f32> {
        &self.physical_to_modes
    }
    /// Calibrated row-major factor C.
    pub fn modes_to_reference(&self) -> &PreparedGemv<f32> {
        &self.modes_to_reference
    }
    /// Exact adopted physical offset.
    pub fn offset(&self) -> &[f32] {
        &self.offset
    }
    /// Reference correction from the same adopted snapshot.
    pub fn correction(&self) -> &[f32] {
        &self.correction
    }
    /// Calibrated sign, always +1 or -1.
    pub fn reference_sign(&self) -> f32 {
        self.reference_sign
    }
    /// Allocate disposable residual-output scratch.
    pub fn workspace(&self) -> VectorLinearCombinationWorkspace {
        self.combination.workspace()
    }
}
fn matrix_extent(rows: usize, columns: usize) -> Result<usize, PhysicalOffsetReferencePlanError> {
    rows.checked_mul(columns)
        .filter(|&n| {
            rows > 0 && columns > 0 && n <= u32::MAX as usize / core::mem::size_of::<f32>()
        })
        .ok_or(PhysicalOffsetReferencePlanError::Dimensions)
}
fn replace_factor(
    factor: &PreparedGemv<f32>,
    values: &[f32],
) -> Result<PreparedGemv<f32>, PhysicalOffsetReferencePlanError> {
    if values.len() != factor.matrix().len() {
        return Err(PhysicalOffsetReferencePlanError::Extent);
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PhysicalOffsetReferencePlanError::NonFinite);
    }
    PreparedGemv::new(factor.rows(), factor.columns(), Arc::from(values))
        .map_err(|_| PhysicalOffsetReferencePlanError::Extent)
}
impl AlgorithmPlan for PhysicalOffsetReferencePlan {
    type Input<'a> = &'a [f32];
    type Output<'a> = (&'a mut [f32], &'a mut [f32]);
    type Workspace = VectorLinearCombinationWorkspace;
    type Error = VectorLinearCombinationError;
    fn process<'input, 'output>(
        &self,
        (compensated, physical): Self::Output<'output>,
        workspace: &mut Self::Workspace,
        residual: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if physical.len() != self.offset.len() {
            return Err(VectorLinearCombinationError::DimensionMismatch);
        }
        self.combination
            .process(compensated, workspace, (residual, &self.correction))?;
        physical.copy_from_slice(&self.offset);
        Ok(())
    }
}
crate::filtergraph_algorithm! {
    /// Compensate a residual while publishing the exact matched physical offset.
    pub PhysicalOffsetReferenceF32 {
        label="physical-offset-reference-f32";
        configuration PhysicalOffsetReferenceF32Configuration {
            residual_count:usize=GraphRebuild;
            physical_count:usize=GraphRebuild;
            mode_count:usize=GraphRebuild;
            reference_sign:f32=Construction;
            residual_schema:String=GraphRebuild;
            physical_schema:String=GraphRebuild;
            physical_to_modes_schema:String=GraphRebuild;
            modes_to_reference_schema:String=GraphRebuild;
        }
        plan=PhysicalOffsetReferencePlan;
        workspace=VectorLinearCombinationWorkspace;
        properties=none;
        dimensions|plan,_workspace| { residual_count=plan.correction.len(); physical_count=plan.offset.len(); mode_count=plan.physical_to_modes.rows(); }
        inputs { reconstructed_residual:f32[residual_count]="reconstructed-residual",schema=configuration.residual_schema.as_str(); }
        outputs {
            compensated_residual:f32[residual_count]="compensated-residual",schema=configuration.residual_schema.as_str(),metadata_from="reconstructed-residual";
            physical_offset:f32[physical_count]="physical-offset",schema=configuration.physical_schema.as_str(),metadata_from="reconstructed-residual";
        }
        parameters {
            offset:f32[physical_count]="offset",schema=configuration.physical_schema.as_str(),replace|plan,values| { plan.with_offset(values) };
            physical_to_modes:f32[mode_count,physical_count]="physical-to-modes",schema=configuration.physical_to_modes_schema.as_str(),replace|plan,values| { plan.with_physical_to_modes(values) };
            modes_to_reference:f32[residual_count,mode_count]="modes-to-reference",schema=configuration.modes_to_reference_schema.as_str(),replace|plan,values| { plan.with_modes_to_reference(values) };
        }
        parameter_error=PhysicalOffsetReferencePlanError;
        prepare|configuration| {
            let plan=PhysicalOffsetReferencePlan::new(configuration.residual_count,configuration.physical_count,configuration.mode_count,configuration.reference_sign)?;
            let workspace=plan.workspace();
            Ok((plan,workspace))
        }
        prepare_error=PhysicalOffsetReferencePlanError;
        process|plan,workspace| { plan.process((compensated_residual,physical_offset),workspace,reconstructed_residual) }
        process_error=VectorLinearCombinationError;
        reset|_plan,workspace| { workspace.reset(); }
    }
}
