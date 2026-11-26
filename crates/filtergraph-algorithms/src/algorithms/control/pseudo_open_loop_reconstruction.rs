//! Pseudo-open-loop reconstruction from residual measurements and one
//! exposure-associated corrector shape.

use alloc::{boxed::Box, string::String, sync::Arc, vec, vec::Vec};
use core::fmt;

use crate::algorithms::linear_algebra::multiply_row_major;
use crate::{AlgorithmPlan, MatrixElement};

/// Prepared matrix named by a pseudo-open-loop preparation error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PseudoOpenLoopMatrix {
    /// Corrector shape to measurement response, conventionally `D`.
    Interaction,
    /// Pseudo-open-loop measurements to open-loop estimate, conventionally `R`.
    Reconstructor,
    /// Corrector shape to residual-error coordinates, conventionally `P`.
    CorrectorToError,
}

/// Invalid prepared pseudo-open-loop reconstruction configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PseudoOpenLoopReconstructionPlanError {
    /// Measurement, corrector-shape, and residual-error extents must be nonzero.
    EmptyDimension,
    /// One prepared matrix extent overflowed addressable storage.
    ExtentOverflow {
        /// Matrix whose required extent overflowed.
        matrix: PseudoOpenLoopMatrix,
    },
    /// One matrix has the wrong row-major extent.
    MatrixExtent {
        /// Matrix with the wrong extent.
        matrix: PseudoOpenLoopMatrix,
        /// Required coefficient count.
        expected: usize,
        /// Supplied coefficient count.
        actual: usize,
    },
    /// One prepared matrix contains a non-finite coefficient.
    NonFiniteMatrix {
        /// Matrix containing the coefficient.
        matrix: PseudoOpenLoopMatrix,
    },
}

impl fmt::Display for PseudoOpenLoopReconstructionPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension => formatter.write_str(
                "pseudo-open-loop measurement, corrector-shape, and residual-error dimensions must be nonzero",
            ),
            Self::ExtentOverflow { matrix } => {
                write!(formatter, "pseudo-open-loop {matrix:?} matrix extent overflow")
            }
            Self::MatrixExtent {
                matrix,
                expected,
                actual,
            } => write!(
                formatter,
                "pseudo-open-loop {matrix:?} matrix extent {actual} does not match {expected}"
            ),
            Self::NonFiniteMatrix { matrix } => write!(
                formatter,
                "pseudo-open-loop {matrix:?} matrix must be finite"
            ),
        }
    }
}

impl core::error::Error for PseudoOpenLoopReconstructionPlanError {}

/// Input vector named by a pseudo-open-loop execution error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PseudoOpenLoopInputKind {
    /// Residual sensor measurements.
    ResidualMeasurements,
    /// Complete corrector shape associated with the same sensor acquisition.
    CorrectorShape,
}

/// Output vector named by a pseudo-open-loop execution error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PseudoOpenLoopOutputKind {
    /// Residual measurements plus the predicted corrector response.
    PseudoOpenLoopMeasurements,
    /// Reconstructed estimate after the corrector contribution is removed.
    ResidualError,
}

/// Numerical stage named by a pseudo-open-loop execution error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PseudoOpenLoopStage {
    /// Interaction-matrix projection `D*u`.
    PredictedMeasurements,
    /// Measurement-space sum `s + D*u`.
    PseudoOpenLoopMeasurements,
    /// Reconstruction `R*(s + D*u)`.
    OpenLoopEstimate,
    /// Corrector projection `P*u`.
    CorrectorInErrorCoordinates,
    /// Residualization `R*(s + D*u) - P*u`.
    ResidualError,
}

/// Invalid pseudo-open-loop input, output, workspace, or numerical result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PseudoOpenLoopReconstructionError {
    /// One caller-owned input has the wrong extent.
    InputExtent {
        /// Input with the wrong extent.
        input: PseudoOpenLoopInputKind,
        /// Required extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// One caller-owned output has the wrong extent.
    OutputExtent {
        /// Output with the wrong extent.
        output: PseudoOpenLoopOutputKind,
        /// Required extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// The supplied workspace was prepared for different extents.
    WorkspaceExtent,
    /// One input value is not finite.
    NonFiniteInput {
        /// Input containing the value.
        input: PseudoOpenLoopInputKind,
        /// Zero-based value index.
        index: usize,
    },
    /// One numerical stage produced a non-finite value.
    NonFiniteCalculation {
        /// Stage that produced the value.
        stage: PseudoOpenLoopStage,
    },
}

impl fmt::Display for PseudoOpenLoopReconstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent {
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "pseudo-open-loop {input:?} input extent {actual} does not match {expected}"
            ),
            Self::OutputExtent {
                output,
                expected,
                actual,
            } => write!(
                formatter,
                "pseudo-open-loop {output:?} output extent {actual} does not match {expected}"
            ),
            Self::WorkspaceExtent => {
                formatter.write_str("pseudo-open-loop workspace extent is invalid")
            }
            Self::NonFiniteInput { input, index } => write!(
                formatter,
                "pseudo-open-loop {input:?} input at index {index} must be finite"
            ),
            Self::NonFiniteCalculation { stage } => write!(
                formatter,
                "pseudo-open-loop {stage:?} calculation produced a non-finite value"
            ),
        }
    }
}

impl core::error::Error for PseudoOpenLoopReconstructionError {}

/// Same-acquisition inputs for pseudo-open-loop reconstruction.
#[derive(Clone, Copy, Debug)]
pub struct PseudoOpenLoopReconstructionInput<'a, T> {
    residual_measurements: &'a [T],
    corrector_shape: &'a [T],
}

impl<'a, T> PseudoOpenLoopReconstructionInput<'a, T> {
    /// Groups residual measurements and their exposure-associated corrector shape.
    pub const fn new(residual_measurements: &'a [T], corrector_shape: &'a [T]) -> Self {
        Self {
            residual_measurements,
            corrector_shape,
        }
    }

    /// Returns the residual sensor measurements.
    pub const fn residual_measurements(&self) -> &[T] {
        self.residual_measurements
    }

    /// Returns the complete corrector shape associated with the acquisition.
    pub const fn corrector_shape(&self) -> &[T] {
        self.corrector_shape
    }
}

/// Synchronized caller-owned pseudo-open-loop reconstruction outputs.
#[derive(Debug)]
pub struct PseudoOpenLoopReconstructionOutput<'a, T> {
    pseudo_open_loop_measurements: &'a mut [T],
    residual_error: &'a mut [T],
}

impl<'a, T> PseudoOpenLoopReconstructionOutput<'a, T> {
    /// Groups the measurement-space intermediate and controller residual error.
    pub fn new(pseudo_open_loop_measurements: &'a mut [T], residual_error: &'a mut [T]) -> Self {
        Self {
            pseudo_open_loop_measurements,
            residual_error,
        }
    }

    /// Returns residual measurements plus the predicted corrector response.
    pub fn pseudo_open_loop_measurements(&self) -> &[T] {
        self.pseudo_open_loop_measurements
    }

    /// Returns the reconstructed residual error suitable for loop integration.
    pub fn residual_error(&self) -> &[T] {
        self.residual_error
    }
}

/// Prepared pseudo-open-loop reconstruction and residualization.
///
/// For residual measurements `s`, exposure-associated corrector shape `u`,
/// interaction matrix `D`, reconstructor `R`, and independently prepared
/// corrector-to-error projection `P`, processing computes:
///
/// ```text
/// pseudo_open_loop_measurements = s + D*u
/// residual_error = R*pseudo_open_loop_measurements - P*u
/// ```
///
/// This plan does not select a latest corrector shape, infer temporal
/// association, or treat constraint feedback as the shape.
#[derive(Clone, Debug, PartialEq)]
pub struct PseudoOpenLoopReconstructionPlan<T> {
    measurement_count: usize,
    corrector_shape_count: usize,
    residual_error_count: usize,
    interaction_matrix: Arc<[T]>,
    reconstructor: Arc<[T]>,
    corrector_to_error: Arc<[T]>,
}

impl<T> PseudoOpenLoopReconstructionPlan<T>
where
    T: MatrixElement,
{
    /// Validates and retains all three independent row-major matrices.
    pub fn new(
        measurement_count: usize,
        corrector_shape_count: usize,
        residual_error_count: usize,
        interaction_matrix: Arc<[T]>,
        reconstructor: Arc<[T]>,
        corrector_to_error: Arc<[T]>,
    ) -> Result<Self, PseudoOpenLoopReconstructionPlanError> {
        if measurement_count == 0 || corrector_shape_count == 0 || residual_error_count == 0 {
            return Err(PseudoOpenLoopReconstructionPlanError::EmptyDimension);
        }
        validate_matrix(
            PseudoOpenLoopMatrix::Interaction,
            measurement_count,
            corrector_shape_count,
            &interaction_matrix,
        )?;
        validate_matrix(
            PseudoOpenLoopMatrix::Reconstructor,
            residual_error_count,
            measurement_count,
            &reconstructor,
        )?;
        validate_matrix(
            PseudoOpenLoopMatrix::CorrectorToError,
            residual_error_count,
            corrector_shape_count,
            &corrector_to_error,
        )?;
        Ok(Self {
            measurement_count,
            corrector_shape_count,
            residual_error_count,
            interaction_matrix,
            reconstructor,
            corrector_to_error,
        })
    }

    /// Returns the residual and pseudo-open-loop measurement extent.
    pub const fn measurement_count(&self) -> usize {
        self.measurement_count
    }

    /// Returns the exposure-associated corrector-shape extent.
    pub const fn corrector_shape_count(&self) -> usize {
        self.corrector_shape_count
    }

    /// Returns the reconstructed residual-error extent.
    pub const fn residual_error_count(&self) -> usize {
        self.residual_error_count
    }

    /// Returns the row-major corrector-to-measurement interaction matrix `D`.
    pub fn interaction_matrix(&self) -> &[T] {
        &self.interaction_matrix
    }

    /// Returns shared ownership of the corrector-to-measurement interaction matrix.
    pub fn shared_interaction_matrix(&self) -> Arc<[T]> {
        Arc::clone(&self.interaction_matrix)
    }

    /// Returns the row-major pseudo-open-loop reconstructor `R`.
    pub fn reconstructor(&self) -> &[T] {
        &self.reconstructor
    }

    /// Returns shared ownership of the pseudo-open-loop reconstructor.
    pub fn shared_reconstructor(&self) -> Arc<[T]> {
        Arc::clone(&self.reconstructor)
    }

    /// Returns the independently prepared row-major corrector-to-error projection `P`.
    pub fn corrector_to_error(&self) -> &[T] {
        &self.corrector_to_error
    }

    /// Returns shared ownership of the corrector-to-error projection.
    pub fn shared_corrector_to_error(&self) -> Arc<[T]> {
        Arc::clone(&self.corrector_to_error)
    }

    /// Replaces the interaction matrix while preserving the other calibrated maps.
    pub fn with_interaction_matrix(
        &self,
        interaction_matrix: Arc<[T]>,
    ) -> Result<Self, PseudoOpenLoopReconstructionPlanError> {
        Self::new(
            self.measurement_count,
            self.corrector_shape_count,
            self.residual_error_count,
            interaction_matrix,
            self.shared_reconstructor(),
            self.shared_corrector_to_error(),
        )
    }

    /// Replaces the reconstructor while preserving the other calibrated maps.
    pub fn with_reconstructor(
        &self,
        reconstructor: Arc<[T]>,
    ) -> Result<Self, PseudoOpenLoopReconstructionPlanError> {
        Self::new(
            self.measurement_count,
            self.corrector_shape_count,
            self.residual_error_count,
            self.shared_interaction_matrix(),
            reconstructor,
            self.shared_corrector_to_error(),
        )
    }

    /// Replaces the corrector-to-error projection while preserving the other maps.
    pub fn with_corrector_to_error(
        &self,
        corrector_to_error: Arc<[T]>,
    ) -> Result<Self, PseudoOpenLoopReconstructionPlanError> {
        Self::new(
            self.measurement_count,
            self.corrector_shape_count,
            self.residual_error_count,
            self.shared_interaction_matrix(),
            self.shared_reconstructor(),
            corrector_to_error,
        )
    }

    /// Allocates fixed scratch for one execution owner.
    pub fn workspace(&self) -> PseudoOpenLoopReconstructionWorkspace<T> {
        PseudoOpenLoopReconstructionWorkspace {
            predicted_measurements: vec![T::zero(); self.measurement_count].into_boxed_slice(),
            staged_pseudo_open_loop_measurements: vec![T::zero(); self.measurement_count]
                .into_boxed_slice(),
            open_loop_estimate: vec![T::zero(); self.residual_error_count].into_boxed_slice(),
            corrector_in_error_coordinates: vec![T::zero(); self.residual_error_count]
                .into_boxed_slice(),
            staged_residual_error: vec![T::zero(); self.residual_error_count].into_boxed_slice(),
        }
    }

    fn validate_workspace(
        &self,
        workspace: &PseudoOpenLoopReconstructionWorkspace<T>,
    ) -> Result<(), PseudoOpenLoopReconstructionError> {
        if workspace.predicted_measurements.len() != self.measurement_count
            || workspace.staged_pseudo_open_loop_measurements.len() != self.measurement_count
            || workspace.open_loop_estimate.len() != self.residual_error_count
            || workspace.corrector_in_error_coordinates.len() != self.residual_error_count
            || workspace.staged_residual_error.len() != self.residual_error_count
        {
            return Err(PseudoOpenLoopReconstructionError::WorkspaceExtent);
        }
        Ok(())
    }
}

impl<T> AlgorithmPlan for PseudoOpenLoopReconstructionPlan<T>
where
    T: MatrixElement + 'static,
{
    type Input<'a>
        = PseudoOpenLoopReconstructionInput<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = PseudoOpenLoopReconstructionOutput<'a, T>
    where
        Self: 'a;
    type Workspace = PseudoOpenLoopReconstructionWorkspace<T>;
    type Error = PseudoOpenLoopReconstructionError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        validate_extent(
            PseudoOpenLoopInputKind::ResidualMeasurements,
            self.measurement_count,
            input.residual_measurements.len(),
        )?;
        validate_extent(
            PseudoOpenLoopInputKind::CorrectorShape,
            self.corrector_shape_count,
            input.corrector_shape.len(),
        )?;
        validate_output_extent(
            PseudoOpenLoopOutputKind::PseudoOpenLoopMeasurements,
            self.measurement_count,
            output.pseudo_open_loop_measurements.len(),
        )?;
        validate_output_extent(
            PseudoOpenLoopOutputKind::ResidualError,
            self.residual_error_count,
            output.residual_error.len(),
        )?;
        self.validate_workspace(workspace)?;
        validate_finite(
            PseudoOpenLoopInputKind::ResidualMeasurements,
            input.residual_measurements,
        )?;
        validate_finite(
            PseudoOpenLoopInputKind::CorrectorShape,
            input.corrector_shape,
        )?;

        multiply_row_major(
            &self.interaction_matrix,
            self.measurement_count,
            self.corrector_shape_count,
            input.corrector_shape,
            &mut workspace.predicted_measurements,
        );
        validate_stage(
            PseudoOpenLoopStage::PredictedMeasurements,
            &workspace.predicted_measurements,
        )?;

        for ((staged, residual), predicted) in workspace
            .staged_pseudo_open_loop_measurements
            .iter_mut()
            .zip(input.residual_measurements)
            .zip(workspace.predicted_measurements.iter())
        {
            *staged = *residual + *predicted;
        }
        validate_stage(
            PseudoOpenLoopStage::PseudoOpenLoopMeasurements,
            &workspace.staged_pseudo_open_loop_measurements,
        )?;

        multiply_row_major(
            &self.reconstructor,
            self.residual_error_count,
            self.measurement_count,
            &workspace.staged_pseudo_open_loop_measurements,
            &mut workspace.open_loop_estimate,
        );
        validate_stage(
            PseudoOpenLoopStage::OpenLoopEstimate,
            &workspace.open_loop_estimate,
        )?;

        multiply_row_major(
            &self.corrector_to_error,
            self.residual_error_count,
            self.corrector_shape_count,
            input.corrector_shape,
            &mut workspace.corrector_in_error_coordinates,
        );
        validate_stage(
            PseudoOpenLoopStage::CorrectorInErrorCoordinates,
            &workspace.corrector_in_error_coordinates,
        )?;

        for ((staged, estimate), correction) in workspace
            .staged_residual_error
            .iter_mut()
            .zip(workspace.open_loop_estimate.iter())
            .zip(workspace.corrector_in_error_coordinates.iter())
        {
            *staged = *estimate - *correction;
        }
        validate_stage(
            PseudoOpenLoopStage::ResidualError,
            &workspace.staged_residual_error,
        )?;

        output
            .pseudo_open_loop_measurements
            .copy_from_slice(&workspace.staged_pseudo_open_loop_measurements);
        output
            .residual_error
            .copy_from_slice(&workspace.staged_residual_error);
        Ok(())
    }
}

/// Fixed scratch for one pseudo-open-loop reconstruction owner.
#[derive(Clone, Debug, PartialEq)]
pub struct PseudoOpenLoopReconstructionWorkspace<T> {
    predicted_measurements: Box<[T]>,
    staged_pseudo_open_loop_measurements: Box<[T]>,
    open_loop_estimate: Box<[T]>,
    corrector_in_error_coordinates: Box<[T]>,
    staged_residual_error: Box<[T]>,
}

fn validate_matrix<T>(
    matrix: PseudoOpenLoopMatrix,
    rows: usize,
    columns: usize,
    coefficients: &[T],
) -> Result<(), PseudoOpenLoopReconstructionPlanError>
where
    T: MatrixElement,
{
    let expected = rows
        .checked_mul(columns)
        .ok_or(PseudoOpenLoopReconstructionPlanError::ExtentOverflow { matrix })?;
    if coefficients.len() != expected {
        return Err(PseudoOpenLoopReconstructionPlanError::MatrixExtent {
            matrix,
            expected,
            actual: coefficients.len(),
        });
    }
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(PseudoOpenLoopReconstructionPlanError::NonFiniteMatrix { matrix });
    }
    Ok(())
}

fn validate_extent(
    input: PseudoOpenLoopInputKind,
    expected: usize,
    actual: usize,
) -> Result<(), PseudoOpenLoopReconstructionError> {
    if actual != expected {
        return Err(PseudoOpenLoopReconstructionError::InputExtent {
            input,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_output_extent(
    output: PseudoOpenLoopOutputKind,
    expected: usize,
    actual: usize,
) -> Result<(), PseudoOpenLoopReconstructionError> {
    if actual != expected {
        return Err(PseudoOpenLoopReconstructionError::OutputExtent {
            output,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_finite<T>(
    input: PseudoOpenLoopInputKind,
    values: &[T],
) -> Result<(), PseudoOpenLoopReconstructionError>
where
    T: MatrixElement,
{
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(PseudoOpenLoopReconstructionError::NonFiniteInput { input, index });
    }
    Ok(())
}

fn validate_stage<T>(
    stage: PseudoOpenLoopStage,
    values: &[T],
) -> Result<(), PseudoOpenLoopReconstructionError>
where
    T: MatrixElement,
{
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PseudoOpenLoopReconstructionError::NonFiniteCalculation { stage });
    }
    Ok(())
}

crate::filtergraph_algorithm! {
    /// Portable Float32 pseudo-open-loop reconstruction declaration.
    pub PseudoOpenLoopReconstructionF32 {
        label = "pseudo-open-loop-reconstruction-f32";
        configuration PseudoOpenLoopReconstructionF32Configuration {
            measurement_count: usize = GraphRebuild;
            corrector_shape_count: usize = GraphRebuild;
            residual_error_count: usize = GraphRebuild;
            initial_interaction_matrix: Option<Vec<f32>> = Construction;
            initial_reconstructor: Option<Vec<f32>> = Construction;
            initial_corrector_to_error: Option<Vec<f32>> = Construction;
            residual_measurements_schema: String = GraphRebuild;
            corrector_shape_schema: String = GraphRebuild;
            pseudo_open_loop_measurements_schema: String = GraphRebuild;
            residual_error_schema: String = GraphRebuild;
        }
        plan = PseudoOpenLoopReconstructionPlan<f32>;
        workspace = PseudoOpenLoopReconstructionWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            measurement_count = plan.measurement_count();
            corrector_shape_count = plan.corrector_shape_count();
            residual_error_count = plan.residual_error_count();
        }
        inputs {
            residual_measurements: f32 [measurement_count] = "residual-measurements",
                schema = configuration.residual_measurements_schema.as_str();
            corrector_shape: f32 [corrector_shape_count] = "corrector-shape",
                schema = configuration.corrector_shape_schema.as_str();
        }
        outputs {
            pseudo_open_loop_measurements: f32 [measurement_count]
                = "pseudo-open-loop-measurements",
                schema = configuration.pseudo_open_loop_measurements_schema.as_str(),
                metadata_from = "residual-measurements";
            residual_error: f32 [residual_error_count] = "residual-error",
                schema = configuration.residual_error_schema.as_str(),
                metadata_from = "residual-measurements";
        }
        parameters {
            interaction_matrix: f32 [measurement_count, corrector_shape_count]
                = "interaction-matrix",
                schema = crate::schemas::PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1,
                replace |plan, values| {
                    plan.with_interaction_matrix(Arc::from(values))
                };
            reconstructor: f32 [residual_error_count, measurement_count]
                = "reconstructor",
                schema = crate::schemas::PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1,
                replace |plan, values| {
                    plan.with_reconstructor(Arc::from(values))
                };
            corrector_to_error: f32 [residual_error_count, corrector_shape_count]
                = "corrector-to-error",
                schema = crate::schemas::PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1,
                replace |plan, values| {
                    plan.with_corrector_to_error(Arc::from(values))
                };
        }
        parameter_error = PseudoOpenLoopReconstructionPlanError;
        prepare |configuration| {
            let interaction_elements = configuration
                .measurement_count
                .checked_mul(configuration.corrector_shape_count)
                .ok_or(PseudoOpenLoopReconstructionPlanError::ExtentOverflow {
                    matrix: PseudoOpenLoopMatrix::Interaction,
                })?;
            let reconstructor_elements = configuration
                .residual_error_count
                .checked_mul(configuration.measurement_count)
                .ok_or(PseudoOpenLoopReconstructionPlanError::ExtentOverflow {
                    matrix: PseudoOpenLoopMatrix::Reconstructor,
                })?;
            let corrector_to_error_elements = configuration
                .residual_error_count
                .checked_mul(configuration.corrector_shape_count)
                .ok_or(PseudoOpenLoopReconstructionPlanError::ExtentOverflow {
                    matrix: PseudoOpenLoopMatrix::CorrectorToError,
                })?;
            let plan = PseudoOpenLoopReconstructionPlan::new(
                configuration.measurement_count,
                configuration.corrector_shape_count,
                configuration.residual_error_count,
                Arc::from(configuration.initial_interaction_matrix
                    .unwrap_or_else(|| vec![0.0; interaction_elements])),
                Arc::from(configuration.initial_reconstructor
                    .unwrap_or_else(|| vec![0.0; reconstructor_elements])),
                Arc::from(configuration.initial_corrector_to_error
                    .unwrap_or_else(|| vec![0.0; corrector_to_error_elements])),
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = PseudoOpenLoopReconstructionPlanError;
        process |plan, workspace| {
            plan.process(
                PseudoOpenLoopReconstructionOutput::new(
                    pseudo_open_loop_measurements,
                    residual_error,
                ),
                workspace,
                PseudoOpenLoopReconstructionInput::new(
                    residual_measurements,
                    corrector_shape,
                ),
            )
        }
        process_error = PseudoOpenLoopReconstructionError;
    }
}
