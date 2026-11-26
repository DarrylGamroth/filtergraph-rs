//! Bounded SCC PCG solve for (H′WH + λL′L)x = H′Wy, following JFG.
//! Fused multiply-adds, normalized RHS, scaled norms, and failure staging retain
//! the reference's representability and dynamic-weight semantics.
use crate::AlgorithmPlan;
use alloc::{string::String, vec, vec::Vec};
use core::fmt;
use num_traits::Float;
/// Invalid matrix dimensions/calibration or bounded solver policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccWeightedRegularizedReconstructionPlanError {
    /// Positive matrix/port byte extents must fit UInt32.
    Extent,
    /// A matrix extent or coefficient is invalid.
    Matrix,
    /// Regularization must be finite and nonnegative.
    Regularization,
    /// The bounded iteration count must be positive.
    Iterations,
    /// Relative tolerance must be finite and positive.
    RelativeTolerance,
    /// Preconditioner minimum diagonal must be finite and positive.
    MinimumDiagonal,
}
impl fmt::Display for SccWeightedRegularizedReconstructionPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid SCC regularized reconstruction preparation: {self:?}"
        )
    }
}
impl core::error::Error for SccWeightedRegularizedReconstructionPlanError {}
/// Dynamic sample rejection or bounded numerical solve disposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccWeightedRegularizedReconstructionError {
    /// Caller/workspace vectors have incompatible extents.
    DimensionMismatch,
    /// A weight is nonfinite or negative.
    InvalidWeight,
    /// A nonzero-weight measurement is nonfinite.
    InvalidMeasurement,
    /// Preconditioning, inner products, norms, or rescaling broke down; this
    /// disposition does not by itself prove mathematical singularity.
    SingularSystem,
    /// The bounded solve did not meet its relative convergence criterion.
    NotConverged,
}
impl fmt::Display for SccWeightedRegularizedReconstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SCC regularized reconstruction disposition: {self:?}")
    }
}
impl core::error::Error for SccWeightedRegularizedReconstructionError {}
/// Immutable row-major interaction/regularization matrices and PCG policy.
#[derive(Clone, Debug)]
pub struct SccWeightedRegularizedReconstructionPlan {
    measurement_count: usize,
    controller_count: usize,
    regularization_count: usize,
    interaction_matrix: Vec<f32>,
    regularization_operator: Vec<f32>,
    regularization: f32,
    iterations: usize,
    relative_tolerance: f32,
    minimum_diagonal: f32,
}
impl SccWeightedRegularizedReconstructionPlan {
    /// Validate explicit H/L row-major matrices and the source bounded policy.
    #[expect(
        clippy::too_many_arguments,
        reason = "Retains the scientific calibration arguments; the declaration supplies structured configuration"
    )]
    pub fn new(
        measurement_count: usize,
        controller_count: usize,
        regularization_count: usize,
        interaction_matrix: Vec<f32>,
        regularization_operator: Vec<f32>,
        regularization: f32,
        iterations: usize,
        relative_tolerance: f32,
        minimum_diagonal: f32,
    ) -> Result<Self, SccWeightedRegularizedReconstructionPlanError> {
        use SccWeightedRegularizedReconstructionPlanError as E;
        let (h, l) = matrix_extents(measurement_count, controller_count, regularization_count)?;
        if interaction_matrix.len() != h
            || regularization_operator.len() != l
            || interaction_matrix
                .iter()
                .chain(&regularization_operator)
                .any(|x| !x.is_finite())
        {
            return Err(E::Matrix);
        }
        if !regularization.is_finite() || regularization < 0.0 {
            return Err(E::Regularization);
        }
        if iterations == 0 {
            return Err(E::Iterations);
        }
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
            return Err(E::RelativeTolerance);
        }
        if !minimum_diagonal.is_finite() || minimum_diagonal <= 0.0 {
            return Err(E::MinimumDiagonal);
        }
        Ok(Self {
            measurement_count,
            controller_count,
            regularization_count,
            interaction_matrix,
            regularization_operator,
            regularization,
            iterations,
            relative_tolerance,
            minimum_diagonal,
        })
    }
    /// Number of dynamic measurements and weights.
    pub const fn measurement_count(&self) -> usize {
        self.measurement_count
    }
    /// Number of controller residual-error coordinates.
    pub const fn controller_count(&self) -> usize {
        self.controller_count
    }
    /// Number of regularization rows.
    pub const fn regularization_count(&self) -> usize {
        self.regularization_count
    }
    /// Prepare finite fixed-shape interaction replacement.
    pub fn with_interaction_matrix(
        &self,
        values: &[f32],
    ) -> Result<Self, SccWeightedRegularizedReconstructionPlanError> {
        Self::new(
            self.measurement_count,
            self.controller_count,
            self.regularization_count,
            values.to_vec(),
            self.regularization_operator.clone(),
            self.regularization,
            self.iterations,
            self.relative_tolerance,
            self.minimum_diagonal,
        )
    }
    /// Prepare finite fixed-shape regularization-operator replacement.
    pub fn with_regularization_operator(
        &self,
        values: &[f32],
    ) -> Result<Self, SccWeightedRegularizedReconstructionPlanError> {
        Self::new(
            self.measurement_count,
            self.controller_count,
            self.regularization_count,
            self.interaction_matrix.clone(),
            values.to_vec(),
            self.regularization,
            self.iterations,
            self.relative_tolerance,
            self.minimum_diagonal,
        )
    }
    /// Allocate all PCG and operator vectors once.
    pub fn workspace(&self) -> SccWeightedRegularizedReconstructionWorkspace {
        let n = self.controller_count;
        SccWeightedRegularizedReconstructionWorkspace {
            candidate: vec![0.0; n],
            right_hand_side: vec![0.0; n],
            residual: vec![0.0; n],
            preconditioned_residual: vec![0.0; n],
            direction: vec![0.0; n],
            operator_direction: vec![0.0; n],
            diagonal: vec![0.0; n],
            measurement_product: vec![0.0; self.measurement_count],
            regularization_product: vec![0.0; self.regularization_count],
        }
    }
    fn assemble(
        &self,
        w: &mut SccWeightedRegularizedReconstructionWorkspace,
        measurements: &[f32],
        weights: &[f32],
    ) -> Result<(), SccWeightedRegularizedReconstructionError> {
        for controller in 0..self.controller_count {
            let mut rhs = 0.0_f32;
            let mut diagonal = 0.0_f32;
            for measurement in 0..self.measurement_count {
                let weight = weights[measurement];
                if weight == 0.0 {
                    continue;
                }
                let coefficient =
                    self.interaction_matrix[measurement * self.controller_count + controller];
                rhs = coefficient.mul_add(weight * measurements[measurement], rhs);
                diagonal = (weight * coefficient).mul_add(coefficient, diagonal);
            }
            for penalty in 0..self.regularization_count {
                let coefficient =
                    self.regularization_operator[penalty * self.controller_count + controller];
                diagonal = (self.regularization * coefficient).mul_add(coefficient, diagonal);
            }
            if !diagonal.is_finite() || diagonal < self.minimum_diagonal {
                return Err(SccWeightedRegularizedReconstructionError::SingularSystem);
            }
            w.right_hand_side[controller] = rhs;
            w.diagonal[controller] = diagonal;
        }
        Ok(())
    }
    fn operator(&self, w: &mut SccWeightedRegularizedReconstructionWorkspace, weights: &[f32]) {
        for (measurement, &weight) in weights.iter().enumerate() {
            let mut value = 0.0_f32;
            for controller in 0..self.controller_count {
                value = self.interaction_matrix[measurement * self.controller_count + controller]
                    .mul_add(w.direction[controller], value);
            }
            w.measurement_product[measurement] = weight * value;
        }
        for penalty in 0..self.regularization_count {
            let mut value = 0.0_f32;
            for controller in 0..self.controller_count {
                value = self.regularization_operator[penalty * self.controller_count + controller]
                    .mul_add(w.direction[controller], value);
            }
            w.regularization_product[penalty] = value;
        }
        for controller in 0..self.controller_count {
            let mut value = 0.0_f32;
            for measurement in 0..self.measurement_count {
                value = self.interaction_matrix[measurement * self.controller_count + controller]
                    .mul_add(w.measurement_product[measurement], value);
            }
            let mut regularized = 0.0_f32;
            for penalty in 0..self.regularization_count {
                regularized = self.regularization_operator
                    [penalty * self.controller_count + controller]
                    .mul_add(w.regularization_product[penalty], regularized);
            }
            w.operator_direction[controller] = self.regularization.mul_add(regularized, value);
        }
    }
    fn solve(
        &self,
        w: &mut SccWeightedRegularizedReconstructionWorkspace,
        measurements: &[f32],
        weights: &[f32],
    ) -> Result<(), SccWeightedRegularizedReconstructionError> {
        use SccWeightedRegularizedReconstructionError as E;
        for (&measurement, &weight) in measurements.iter().zip(weights) {
            if !weight.is_finite() || weight < 0.0 {
                return Err(E::InvalidWeight);
            }
            if weight != 0.0 && !measurement.is_finite() {
                return Err(E::InvalidMeasurement);
            }
        }
        self.assemble(w, measurements, weights)?;
        w.candidate.fill(0.0);
        let mut scale = 0.0_f32;
        for &value in &w.right_hand_side {
            if !value.is_finite() {
                return Err(E::SingularSystem);
            }
            scale = scale.max(value.abs());
        }
        if scale == 0.0 {
            return Ok(());
        }
        for i in 0..self.controller_count {
            w.residual[i] = w.right_hand_side[i] / scale;
            w.preconditioned_residual[i] = w.residual[i] / w.diagonal[i];
            w.direction[i] = w.preconditioned_residual[i];
        }
        let mut rho = dot(&w.residual, &w.preconditioned_residual);
        if !rho.is_finite() || rho <= 0.0 {
            return Err(E::SingularSystem);
        }
        let tolerance = self.relative_tolerance * norm(&w.residual);
        for _ in 0..self.iterations {
            self.operator(w, weights);
            let denominator = dot(&w.direction, &w.operator_direction);
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(E::SingularSystem);
            }
            let alpha = rho / denominator;
            for i in 0..self.controller_count {
                w.candidate[i] = alpha.mul_add(w.direction[i], w.candidate[i]);
                w.residual[i] = (-alpha).mul_add(w.operator_direction[i], w.residual[i]);
            }
            let residual_norm = norm(&w.residual);
            if !residual_norm.is_finite() {
                return Err(E::SingularSystem);
            }
            if residual_norm <= tolerance {
                for value in &mut w.candidate {
                    let scaled = *value * scale;
                    if !scaled.is_finite() {
                        return Err(E::SingularSystem);
                    }
                    *value = scaled;
                }
                return Ok(());
            }
            for i in 0..self.controller_count {
                w.preconditioned_residual[i] = w.residual[i] / w.diagonal[i];
            }
            let next_rho = dot(&w.residual, &w.preconditioned_residual);
            if !next_rho.is_finite() || next_rho <= 0.0 {
                return Err(E::SingularSystem);
            }
            let beta = next_rho / rho;
            for i in 0..self.controller_count {
                w.direction[i] = beta.mul_add(w.direction[i], w.preconditioned_residual[i]);
            }
            rho = next_rho;
        }
        Err(E::NotConverged)
    }
}
fn matrix_extents(
    m: usize,
    n: usize,
    l: usize,
) -> Result<(usize, usize), SccWeightedRegularizedReconstructionPlanError> {
    let maximum = u32::MAX as usize / 4;
    if m == 0 || n == 0 || l == 0 {
        return Err(SccWeightedRegularizedReconstructionPlanError::Extent);
    }
    let h = m
        .checked_mul(n)
        .filter(|&v| v <= maximum)
        .ok_or(SccWeightedRegularizedReconstructionPlanError::Extent)?;
    let regularized = l
        .checked_mul(n)
        .filter(|&v| v <= maximum)
        .ok_or(SccWeightedRegularizedReconstructionPlanError::Extent)?;
    Ok((h, regularized))
}
fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .fold(0.0_f32, |value, (&l, &r)| l.mul_add(r, value))
}
fn norm(values: &[f32]) -> f32 {
    let mut scale = 0.0_f32;
    let mut squares = 1.0_f32;
    for &value in values {
        let magnitude = value.abs();
        if !magnitude.is_finite() {
            return magnitude;
        }
        if magnitude == 0.0 {
            continue;
        }
        if scale < magnitude {
            let ratio = scale / magnitude;
            squares = 1.0 + squares * ratio * ratio;
            scale = magnitude;
        } else {
            let ratio = magnitude / scale;
            squares += ratio * ratio;
        }
    }
    scale * Float::sqrt(squares)
}
/// Fixed disposable PCG and matrix-free operator storage.
#[derive(Clone, Debug, PartialEq)]
pub struct SccWeightedRegularizedReconstructionWorkspace {
    candidate: Vec<f32>,
    right_hand_side: Vec<f32>,
    residual: Vec<f32>,
    preconditioned_residual: Vec<f32>,
    direction: Vec<f32>,
    operator_direction: Vec<f32>,
    diagonal: Vec<f32>,
    measurement_product: Vec<f32>,
    regularization_product: Vec<f32>,
}
impl SccWeightedRegularizedReconstructionWorkspace {
    /// Clear scratch without changing adopted matrices or policy.
    pub fn reset(&mut self) {
        for values in [
            &mut self.candidate,
            &mut self.right_hand_side,
            &mut self.residual,
            &mut self.preconditioned_residual,
            &mut self.direction,
            &mut self.operator_direction,
            &mut self.diagonal,
            &mut self.measurement_product,
            &mut self.regularization_product,
        ] {
            values.fill(0.0);
        }
    }
}
impl AlgorithmPlan for SccWeightedRegularizedReconstructionPlan {
    type Input<'a> = (&'a [f32], &'a [f32]);
    type Output<'a> = &'a mut [f32];
    type Workspace = SccWeightedRegularizedReconstructionWorkspace;
    type Error = SccWeightedRegularizedReconstructionError;
    fn process<'i, 'o>(
        &self,
        output: Self::Output<'o>,
        w: &mut Self::Workspace,
        (measurements, weights): Self::Input<'i>,
    ) -> Result<(), Self::Error> {
        if output.len() != self.controller_count
            || measurements.len() != self.measurement_count
            || weights.len() != self.measurement_count
            || w.measurement_product.len() != self.measurement_count
            || w.regularization_product.len() != self.regularization_count
            || [
                w.candidate.len(),
                w.right_hand_side.len(),
                w.residual.len(),
                w.preconditioned_residual.len(),
                w.direction.len(),
                w.operator_direction.len(),
                w.diagonal.len(),
            ]
            .iter()
            .any(|&v| v != self.controller_count)
        {
            return Err(SccWeightedRegularizedReconstructionError::DimensionMismatch);
        }
        self.solve(w, measurements, weights)?;
        output.copy_from_slice(&w.candidate);
        Ok(())
    }
}
crate::filtergraph_algorithm! {
    /// Dynamic weighted regularized SCC controller reconstruction with bounded PCG.
    pub SccWeightedRegularizedReconstructorF32 {
        label="scc-weighted-regularized-reconstructor-f32";
        configuration SccWeightedRegularizedReconstructorF32Configuration {
            measurement_count:usize=GraphRebuild;controller_coordinate_count:usize=GraphRebuild;regularization_row_count:usize=GraphRebuild;measurement_schema:String=GraphRebuild;
            initial_interaction_matrix:Option<Vec<f32>>=Construction;initial_regularization_operator:Option<Vec<f32>>=Construction;regularization:f32=Construction;iterations:usize=Construction;relative_tolerance:f32=Construction;minimum_diagonal:f32=Construction;
        }
        plan=SccWeightedRegularizedReconstructionPlan;workspace=SccWeightedRegularizedReconstructionWorkspace;properties=none;
        dimensions |plan,_workspace|{measurement_count=plan.measurement_count();controller_count=plan.controller_count();regularization_count=plan.regularization_count();}
        inputs {measurements:f32 [measurement_count]="measurements",schema=configuration.measurement_schema.as_str();weights:f32 [measurement_count]="weights",schema=crate::schemas::SCC_MEASUREMENT_WEIGHTS_V1;}
        outputs {controller_residual_error:f32 [controller_count]="controller-residual-error",schema=crate::schemas::CONTROLLER_RESIDUAL_ERROR_V1,metadata_from="measurements";}
        parameters {interaction_matrix:f32 [measurement_count,controller_count]="interaction-matrix",schema=crate::schemas::SCC_INTERACTION_MATRIX_V1,replace|plan,values|{plan.with_interaction_matrix(values)};regularization_operator:f32 [regularization_count,controller_count]="regularization-operator",schema=crate::schemas::SCC_REGULARIZATION_OPERATOR_V1,replace|plan,values|{plan.with_regularization_operator(values)};}
        parameter_error=SccWeightedRegularizedReconstructionPlanError;
        prepare|configuration|{let (h,l)=matrix_extents(configuration.measurement_count,configuration.controller_coordinate_count,configuration.regularization_row_count)?;let plan=SccWeightedRegularizedReconstructionPlan::new(configuration.measurement_count,configuration.controller_coordinate_count,configuration.regularization_row_count,configuration.initial_interaction_matrix.unwrap_or_else(||vec![0.0;h]),configuration.initial_regularization_operator.unwrap_or_else(||vec![0.0;l]),configuration.regularization,configuration.iterations,configuration.relative_tolerance,configuration.minimum_diagonal)?;let workspace=plan.workspace();Ok((plan,workspace))}
        prepare_error=SccWeightedRegularizedReconstructionPlanError;
        process|plan,workspace|{plan.process(controller_residual_error,workspace,(measurements,weights))}
        process_error=SccWeightedRegularizedReconstructionError;
        reset|_plan,workspace|{workspace.reset();}
    }
}
