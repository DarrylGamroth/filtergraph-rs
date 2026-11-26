//! Same-schema vector arithmetic, following JFG's Float32 staging contract.
use crate::AlgorithmPlan;
use alloc::{string::String, vec, vec::Vec};
use core::fmt;

/// Invalid configuration, frame dimensions, or rejected numerical arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorLinearCombinationError {
    /// The positive Float32 byte extent must fit UInt32.
    Extent,
    /// A coefficient or adopted offset is nonfinite.
    NonFiniteConfiguration,
    /// An input, output or workspace has the wrong extent.
    DimensionMismatch,
    /// An input is nonfinite, including a zero-scaled input.
    NonFiniteInput,
    /// A scaled sum is nonfinite.
    NonFiniteOutput,
}
impl fmt::Display for VectorLinearCombinationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Extent => "vector byte extent must be positive and fit UInt32",
            Self::NonFiniteConfiguration => "vector coefficients and offsets must be finite",
            Self::DimensionMismatch => "vector extents must match the prepared plan",
            Self::NonFiniteInput => "vector input must be finite",
            Self::NonFiniteOutput => "vector arithmetic produced a nonfinite result",
        })
    }
}
impl core::error::Error for VectorLinearCombinationError {}

/// Fixed disposable output scratch, without mathematical history.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorLinearCombinationWorkspace {
    candidate_output: Vec<f32>,
}
impl VectorLinearCombinationWorkspace {
    /// Clear disposable scratch; adopted coefficients and offsets are retained.
    pub fn reset(&mut self) {
        self.candidate_output.fill(0.0);
    }
}

/// `output[i] = left_scale * left[i] + right_scale * right[i]`.
/// The dimensionless scales do not convert units or coordinate order.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorLinearCombinationPlan {
    extent: usize,
    left_scale: f32,
    right_scale: f32,
}
impl VectorLinearCombinationPlan {
    /// Prepare finite scales and a positive Float32 extent fitting UInt32 bytes.
    pub fn new(
        extent: usize,
        left_scale: f32,
        right_scale: f32,
    ) -> Result<Self, VectorLinearCombinationError> {
        if extent == 0 || extent > u32::MAX as usize / core::mem::size_of::<f32>() {
            return Err(VectorLinearCombinationError::Extent);
        }
        if !left_scale.is_finite() || !right_scale.is_finite() {
            return Err(VectorLinearCombinationError::NonFiniteConfiguration);
        }
        Ok(Self {
            extent,
            left_scale,
            right_scale,
        })
    }
    /// Number of coordinates sharing one scientific schema.
    pub fn extent(&self) -> usize {
        self.extent
    }
    /// Allocate fixed scratch during preparation.
    pub fn workspace(&self) -> VectorLinearCombinationWorkspace {
        VectorLinearCombinationWorkspace {
            candidate_output: vec![0.0; self.extent],
        }
    }
}
impl AlgorithmPlan for VectorLinearCombinationPlan {
    type Input<'a> = (&'a [f32], &'a [f32]);
    type Output<'a> = &'a mut [f32];
    type Workspace = VectorLinearCombinationWorkspace;
    type Error = VectorLinearCombinationError;
    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        (left, right): Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if [
            output.len(),
            left.len(),
            right.len(),
            workspace.candidate_output.len(),
        ]
        .iter()
        .any(|&n| n != self.extent)
        {
            return Err(VectorLinearCombinationError::DimensionMismatch);
        }
        if left.iter().chain(right).any(|v| !v.is_finite()) {
            return Err(VectorLinearCombinationError::NonFiniteInput);
        }
        for (candidate, (&l, &r)) in workspace
            .candidate_output
            .iter_mut()
            .zip(left.iter().zip(right))
        {
            *candidate = num_traits::Float::mul_add(self.left_scale, l, self.right_scale * r);
        }
        if workspace.candidate_output.iter().any(|v| !v.is_finite()) {
            return Err(VectorLinearCombinationError::NonFiniteOutput);
        }
        output.copy_from_slice(&workspace.candidate_output);
        Ok(())
    }
}

/// Replaceable same-schema offset, separate from disposable scratch.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorOffsetPlan {
    combination: VectorLinearCombinationPlan,
    offset: Vec<f32>,
}
impl VectorOffsetPlan {
    /// Prepare an initially zero offset.
    pub fn new(extent: usize) -> Result<Self, VectorLinearCombinationError> {
        Ok(Self {
            combination: VectorLinearCombinationPlan::new(extent, 1.0, 1.0)?,
            offset: vec![0.0; extent],
        })
    }
    /// Prepare an immutable replacement, rejecting malformed or nonfinite data.
    pub fn with_offset(&self, values: &[f32]) -> Result<Self, VectorLinearCombinationError> {
        if values.len() != self.offset.len() {
            return Err(VectorLinearCombinationError::DimensionMismatch);
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(VectorLinearCombinationError::NonFiniteConfiguration);
        }
        Ok(Self {
            combination: self.combination.clone(),
            offset: values.to_vec(),
        })
    }
    /// Adopted offset in the requested vector's coordinate order and units.
    pub fn offset(&self) -> &[f32] {
        &self.offset
    }
    /// Allocate disposable scratch during preparation.
    pub fn workspace(&self) -> VectorLinearCombinationWorkspace {
        self.combination.workspace()
    }
}
impl AlgorithmPlan for VectorOffsetPlan {
    type Input<'a> = &'a [f32];
    type Output<'a> = (&'a mut [f32], &'a mut [f32]);
    type Workspace = VectorLinearCombinationWorkspace;
    type Error = VectorLinearCombinationError;
    fn process<'input, 'output>(
        &self,
        (output, current_offset): Self::Output<'output>,
        workspace: &mut Self::Workspace,
        requested: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if current_offset.len() != self.offset.len() {
            return Err(VectorLinearCombinationError::DimensionMismatch);
        }
        self.combination
            .process(output, workspace, (requested, &self.offset))?;
        current_offset.copy_from_slice(&self.offset);
        Ok(())
    }
}

crate::filtergraph_algorithm! {
    /// Finite Float32 linear combination preserving the configured schema.
    pub VectorLinearCombinationF32 {
        label = "vector-linear-combination-f32";
        configuration VectorLinearCombinationF32Configuration {
            extent: usize = GraphRebuild;
            left_scale: f32 = Construction;
            right_scale: f32 = Construction;
            schema: String = GraphRebuild;
        }
        plan = VectorLinearCombinationPlan;
        properties = none;
        dimensions |plan, _workspace| { extent = plan.extent(); }
        inputs {
            left: f32 [extent] = "left", schema = configuration.schema.as_str();
            right: f32 [extent] = "right", schema = configuration.schema.as_str();
        }
        outputs { output: f32 [extent] = "output", schema = configuration.schema.as_str(), metadata_from = "left"; }
        prepare |configuration| {
            let plan = VectorLinearCombinationPlan::new(configuration.extent, configuration.left_scale, configuration.right_scale)?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = VectorLinearCombinationError;
        process |plan, workspace| { plan.process(output, workspace, (left, right)) }
        process_error = VectorLinearCombinationError;
        reset |_plan, workspace| { workspace.reset(); }
    }
}
crate::filtergraph_algorithm! {
    /// Same-schema offset with successful-frame current-offset telemetry.
    pub VectorOffsetF32 {
        label = "vector-offset-f32";
        configuration VectorOffsetF32Configuration {
            extent: usize = GraphRebuild;
            schema: String = GraphRebuild;
        }
        plan = VectorOffsetPlan;
        properties = none;
        dimensions |plan, _workspace| { extent = plan.offset().len(); }
        inputs { requested: f32 [extent] = "requested", schema = configuration.schema.as_str(), optional_sample_metadata = ["chopped_image_kind", "trigger_state", "acquisition_identity", "exposure_duration_nanoseconds"]; }
        outputs {
            output: f32 [extent] = "output", schema = configuration.schema.as_str(), metadata_from = "requested";
            current_offset: f32 [extent] = "current-offset", schema = configuration.schema.as_str(), metadata_from = "requested";
        }
        parameters { offset: f32 [extent] = "offset", schema = configuration.schema.as_str(), replace |plan, values| { plan.with_offset(values) }; }
        parameter_error = VectorLinearCombinationError;
        prepare |configuration| {
            let plan = VectorOffsetPlan::new(configuration.extent)?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = VectorLinearCombinationError;
        process |plan, workspace| { plan.process((output, current_offset), workspace, requested) }
        process_error = VectorLinearCombinationError;
        reset |_plan, workspace| { workspace.reset(); }
    }
}
