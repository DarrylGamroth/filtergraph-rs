//! Direct interaction-matrix decoding for paired structured probes.

use alloc::{boxed::Box, string::String, vec, vec::Vec};
use core::fmt;

use num_traits::NumCast;

use super::structured_calibration::{
    checked_hadamard_order, hadamard_sign_is_positive, validate_probe_amplitudes,
};
use crate::{AlgorithmPlan, CalibrationProbePlanError, MatrixElement};

/// Prepared extent named by a structured interaction-matrix error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuredInteractionMatrixExtent {
    /// Complete paired response cycle.
    Responses,
    /// Row-major measurement-by-corrector interaction matrix.
    InteractionMatrix,
}

/// Invalid prepared structured interaction-matrix operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuredInteractionMatrixPlanError {
    /// The associated structured calibration probe is invalid.
    InvalidProbe(CalibrationProbePlanError),
    /// At least one measurement is required.
    EmptyMeasurement,
    /// A required extent overflowed addressable storage.
    ExtentOverflow {
        /// Extent whose calculation overflowed.
        extent: StructuredInteractionMatrixExtent,
    },
    /// The Hadamard order cannot be represented by the selected floating type.
    UnrepresentableHadamardOrder,
    /// A prepared reciprocal amplitude scale is zero or non-finite.
    InvalidReciprocalScale {
        /// Zero-based corrector-coordinate index.
        corrector_coordinate: usize,
    },
}

impl fmt::Display for StructuredInteractionMatrixPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProbe(error) => write!(formatter, "invalid structured probe: {error}"),
            Self::EmptyMeasurement => {
                formatter.write_str("structured interaction matrix requires measurements")
            }
            Self::ExtentOverflow { extent } => {
                write!(
                    formatter,
                    "structured interaction-matrix {extent:?} extent overflow"
                )
            }
            Self::UnrepresentableHadamardOrder => formatter
                .write_str("structured interaction-matrix Hadamard order is not representable"),
            Self::InvalidReciprocalScale {
                corrector_coordinate,
            } => write!(
                formatter,
                "structured interaction-matrix reciprocal scale for corrector coordinate {corrector_coordinate} must be finite and positive"
            ),
        }
    }
}

impl core::error::Error for StructuredInteractionMatrixPlanError {}

impl From<CalibrationProbePlanError> for StructuredInteractionMatrixPlanError {
    fn from(error: CalibrationProbePlanError) -> Self {
        Self::InvalidProbe(error)
    }
}

/// Invalid structured interaction-matrix input or calculation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuredInteractionMatrixError {
    /// The complete caller-associated response cycle has the wrong extent.
    ResponseExtent {
        /// Required response-scalar count.
        expected: usize,
        /// Supplied response-scalar count.
        actual: usize,
    },
    /// The interaction-matrix output has the wrong extent.
    OutputExtent {
        /// Required matrix-coefficient count.
        expected: usize,
        /// Supplied matrix-coefficient count.
        actual: usize,
    },
    /// One response scalar is not finite.
    NonFiniteResponse {
        /// Zero-based row-major response index.
        index: usize,
    },
    /// The supplied workspace was prepared for an incompatible matrix extent.
    WorkspaceExtent,
    /// A response difference, accumulation, or scaled coefficient is not finite.
    NonFiniteCalculation,
}

impl fmt::Display for StructuredInteractionMatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResponseExtent { expected, actual } => write!(
                formatter,
                "structured response extent {actual} does not match {expected}"
            ),
            Self::OutputExtent { expected, actual } => write!(
                formatter,
                "structured interaction-matrix output extent {actual} does not match {expected}"
            ),
            Self::NonFiniteResponse { index } => write!(
                formatter,
                "structured response at row-major index {index} must be finite"
            ),
            Self::WorkspaceExtent => {
                formatter.write_str("structured interaction-matrix workspace extent is invalid")
            }
            Self::NonFiniteCalculation => formatter
                .write_str("structured interaction-matrix calculation produced a non-finite value"),
        }
    }
}

impl core::error::Error for StructuredInteractionMatrixError {}

/// Fixed failure-atomic result storage for one structured decoder owner.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuredInteractionMatrixWorkspace<T> {
    staged_matrix: Box<[T]>,
}

/// Prepared direct decoder for a complete zonal push-pull response cycle.
#[derive(Clone, Debug, PartialEq)]
pub struct ZonalInteractionMatrixPlan<T> {
    measurement_count: usize,
    response_count: usize,
    matrix_count: usize,
    amplitudes: Box<[T]>,
    reciprocal_scales: Box<[T]>,
}

impl<T> ZonalInteractionMatrixPlan<T>
where
    T: MatrixElement,
{
    /// Validates dimensions and prepares one reciprocal pair scale per coordinate.
    pub fn new(
        measurement_count: usize,
        amplitudes: impl Into<Box<[T]>>,
    ) -> Result<Self, StructuredInteractionMatrixPlanError> {
        let amplitudes = amplitudes.into();
        validate_preparation(measurement_count, &amplitudes)?;
        let (response_count, matrix_count) =
            checked_counts(measurement_count, amplitudes.len(), amplitudes.len())?;
        let reciprocal_scales = reciprocal_scales(&amplitudes, T::one() + T::one())?;
        Ok(Self {
            measurement_count,
            response_count,
            matrix_count,
            amplitudes,
            reciprocal_scales,
        })
    }

    /// Returns the measurement-vector extent.
    pub const fn measurement_count(&self) -> usize {
        self.measurement_count
    }

    /// Returns the corrector-coordinate extent.
    pub fn corrector_coordinate_count(&self) -> usize {
        self.amplitudes.len()
    }

    /// Returns the per-coordinate positive probe amplitudes.
    pub fn amplitudes(&self) -> &[T] {
        &self.amplitudes
    }

    /// Returns the prepared `1 / (2 * amplitude[i])` scales.
    pub fn reciprocal_scales(&self) -> &[T] {
        &self.reciprocal_scales
    }

    /// Returns the scalar extent of one complete paired response cycle.
    pub const fn response_count(&self) -> usize {
        self.response_count
    }

    /// Allocates fixed failure-atomic result storage for one owner.
    pub fn workspace(&self) -> StructuredInteractionMatrixWorkspace<T> {
        StructuredInteractionMatrixWorkspace {
            staged_matrix: vec![T::zero(); self.matrix_count].into_boxed_slice(),
        }
    }
}

impl<T> AlgorithmPlan for ZonalInteractionMatrixPlan<T>
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
    type Workspace = StructuredInteractionMatrixWorkspace<T>;
    type Error = StructuredInteractionMatrixError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        responses: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        validate_execution(
            self.response_count,
            self.matrix_count,
            responses,
            output.len(),
            workspace,
        )?;

        let corrector_count = self.corrector_coordinate_count();
        for corrector_coordinate in 0..corrector_count {
            let positive_row = 2 * corrector_coordinate * self.measurement_count;
            let negative_row = positive_row + self.measurement_count;
            for measurement in 0..self.measurement_count {
                let difference =
                    responses[positive_row + measurement] - responses[negative_row + measurement];
                let coefficient = difference * self.reciprocal_scales[corrector_coordinate];
                if !difference.is_finite() || !coefficient.is_finite() {
                    return Err(StructuredInteractionMatrixError::NonFiniteCalculation);
                }
                workspace.staged_matrix[measurement * corrector_count + corrector_coordinate] =
                    coefficient;
            }
        }

        output.copy_from_slice(&workspace.staged_matrix);
        Ok(())
    }
}

/// Prepared orthogonal decoder for a complete balanced Hadamard response cycle.
#[derive(Clone, Debug, PartialEq)]
pub struct HadamardInteractionMatrixPlan<T> {
    measurement_count: usize,
    response_count: usize,
    matrix_count: usize,
    order: usize,
    amplitudes: Box<[T]>,
    reciprocal_scales: Box<[T]>,
}

impl<T> HadamardInteractionMatrixPlan<T>
where
    T: MatrixElement,
{
    /// Validates dimensions and prepares the inverse orthogonal-transform scales.
    pub fn new(
        measurement_count: usize,
        amplitudes: impl Into<Box<[T]>>,
    ) -> Result<Self, StructuredInteractionMatrixPlanError> {
        let amplitudes = amplitudes.into();
        validate_preparation(measurement_count, &amplitudes)?;
        let order = checked_hadamard_order(amplitudes.len())?;
        let (response_count, matrix_count) =
            checked_counts(measurement_count, order, amplitudes.len())?;
        let order_scalar = exact_scalar::<T>(order)?;
        let normalization = (T::one() + T::one()) * order_scalar;
        let reciprocal_scales = reciprocal_scales(&amplitudes, normalization)?;
        Ok(Self {
            measurement_count,
            response_count,
            matrix_count,
            order,
            amplitudes,
            reciprocal_scales,
        })
    }

    /// Returns the measurement-vector extent.
    pub const fn measurement_count(&self) -> usize {
        self.measurement_count
    }

    /// Returns the corrector-coordinate extent.
    pub fn corrector_coordinate_count(&self) -> usize {
        self.amplitudes.len()
    }

    /// Returns the per-coordinate positive probe amplitudes.
    pub fn amplitudes(&self) -> &[T] {
        &self.amplitudes
    }

    /// Returns the prepared `1 / (2 * order * amplitude[i])` scales.
    pub fn reciprocal_scales(&self) -> &[T] {
        &self.reciprocal_scales
    }

    /// Returns the prepared Sylvester-Hadamard order.
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Returns the scalar extent of one complete paired response cycle.
    pub const fn response_count(&self) -> usize {
        self.response_count
    }

    /// Allocates fixed failure-atomic result storage for one owner.
    pub fn workspace(&self) -> StructuredInteractionMatrixWorkspace<T> {
        StructuredInteractionMatrixWorkspace {
            staged_matrix: vec![T::zero(); self.matrix_count].into_boxed_slice(),
        }
    }
}

impl<T> AlgorithmPlan for HadamardInteractionMatrixPlan<T>
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
    type Workspace = StructuredInteractionMatrixWorkspace<T>;
    type Error = StructuredInteractionMatrixError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        responses: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        validate_execution(
            self.response_count,
            self.matrix_count,
            responses,
            output.len(),
            workspace,
        )?;

        workspace.staged_matrix.fill(T::zero());
        let corrector_count = self.corrector_coordinate_count();
        for row in 0..self.order {
            let positive_row = 2 * row * self.measurement_count;
            let negative_row = positive_row + self.measurement_count;
            for measurement in 0..self.measurement_count {
                let difference =
                    responses[positive_row + measurement] - responses[negative_row + measurement];
                if !difference.is_finite() {
                    return Err(StructuredInteractionMatrixError::NonFiniteCalculation);
                }
                for corrector_coordinate in 0..corrector_count {
                    let base_positive = hadamard_sign_is_positive(row, corrector_coordinate);
                    let signed_difference = if base_positive {
                        difference
                    } else {
                        -difference
                    };
                    let index = measurement * corrector_count + corrector_coordinate;
                    let accumulation = workspace.staged_matrix[index] + signed_difference;
                    if !accumulation.is_finite() {
                        return Err(StructuredInteractionMatrixError::NonFiniteCalculation);
                    }
                    workspace.staged_matrix[index] = accumulation;
                }
            }
        }

        for (index, coefficient) in workspace.staged_matrix.iter_mut().enumerate() {
            *coefficient = *coefficient * self.reciprocal_scales[index % corrector_count];
            if !coefficient.is_finite() {
                return Err(StructuredInteractionMatrixError::NonFiniteCalculation);
            }
        }

        output.copy_from_slice(&workspace.staged_matrix);
        Ok(())
    }
}

fn validate_preparation<T>(
    measurement_count: usize,
    amplitudes: &[T],
) -> Result<(), StructuredInteractionMatrixPlanError>
where
    T: MatrixElement,
{
    if measurement_count == 0 {
        return Err(StructuredInteractionMatrixPlanError::EmptyMeasurement);
    }
    validate_probe_amplitudes(amplitudes)?;
    Ok(())
}

fn checked_counts(
    measurement_count: usize,
    pattern_count: usize,
    corrector_count: usize,
) -> Result<(usize, usize), StructuredInteractionMatrixPlanError> {
    let response_count = pattern_count
        .checked_mul(2)
        .and_then(|pair_count| pair_count.checked_mul(measurement_count))
        .ok_or(StructuredInteractionMatrixPlanError::ExtentOverflow {
            extent: StructuredInteractionMatrixExtent::Responses,
        })?;
    let matrix_count = measurement_count.checked_mul(corrector_count).ok_or(
        StructuredInteractionMatrixPlanError::ExtentOverflow {
            extent: StructuredInteractionMatrixExtent::InteractionMatrix,
        },
    )?;
    Ok((response_count, matrix_count))
}

fn exact_scalar<T>(value: usize) -> Result<T, StructuredInteractionMatrixPlanError>
where
    T: MatrixElement,
{
    let scalar = NumCast::from(value)
        .ok_or(StructuredInteractionMatrixPlanError::UnrepresentableHadamardOrder)?;
    let round_trip: usize = NumCast::from(scalar)
        .ok_or(StructuredInteractionMatrixPlanError::UnrepresentableHadamardOrder)?;
    if round_trip != value {
        return Err(StructuredInteractionMatrixPlanError::UnrepresentableHadamardOrder);
    }
    Ok(scalar)
}

fn reciprocal_scales<T>(
    amplitudes: &[T],
    normalization: T,
) -> Result<Box<[T]>, StructuredInteractionMatrixPlanError>
where
    T: MatrixElement,
{
    let mut scales = amplitudes.to_vec().into_boxed_slice();
    for (corrector_coordinate, (scale, amplitude)) in scales.iter_mut().zip(amplitudes).enumerate()
    {
        *scale = (T::one() / normalization) / *amplitude;
        if !scale.is_finite() || *scale <= T::zero() {
            return Err(
                StructuredInteractionMatrixPlanError::InvalidReciprocalScale {
                    corrector_coordinate,
                },
            );
        }
    }
    Ok(scales)
}

fn validate_execution<T>(
    response_count: usize,
    matrix_count: usize,
    responses: &[T],
    output_count: usize,
    workspace: &StructuredInteractionMatrixWorkspace<T>,
) -> Result<(), StructuredInteractionMatrixError>
where
    T: MatrixElement,
{
    if responses.len() != response_count {
        return Err(StructuredInteractionMatrixError::ResponseExtent {
            expected: response_count,
            actual: responses.len(),
        });
    }
    if output_count != matrix_count {
        return Err(StructuredInteractionMatrixError::OutputExtent {
            expected: matrix_count,
            actual: output_count,
        });
    }
    if workspace.staged_matrix.len() != matrix_count {
        return Err(StructuredInteractionMatrixError::WorkspaceExtent);
    }
    if let Some(index) = responses.iter().position(|response| !response.is_finite()) {
        return Err(StructuredInteractionMatrixError::NonFiniteResponse { index });
    }
    Ok(())
}

crate::filtergraph_algorithm! {
    /// Portable Float32 decoder for one complete zonal push/pull response cycle.
    pub ZonalInteractionMatrixF32 {
        label = "zonal-interaction-matrix-f32";
        configuration ZonalInteractionMatrixF32Configuration {
            measurement_count: usize = GraphRebuild;
            amplitudes: Vec<f32> = GraphRebuild;
            response_schema: String = GraphRebuild;
            interaction_matrix_schema: String = GraphRebuild;
        }
        plan = ZonalInteractionMatrixPlan<f32>;
        workspace = StructuredInteractionMatrixWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            measurement_count = plan.measurement_count();
            corrector_coordinate_count = plan.corrector_coordinate_count();
            response_step_count = plan.response_count() / plan.measurement_count();
        }
        inputs {
            responses: f32 [response_step_count, measurement_count] = "responses",
                schema = configuration.response_schema.as_str();
        }
        outputs {
            interaction_matrix: f32 [measurement_count, corrector_coordinate_count]
                = "interaction-matrix",
                schema = configuration.interaction_matrix_schema.as_str(),
                metadata_from = "responses";
        }
        parameters {
            amplitudes: f32 [corrector_coordinate_count] = "amplitudes",
                schema = crate::schemas::CALIBRATION_PROBE_AMPLITUDES_V1,
                replace |plan, values| {
                    ZonalInteractionMatrixPlan::new(plan.measurement_count(), values)
                };
        }
        parameter_error = StructuredInteractionMatrixPlanError;
        prepare |configuration| {
            let plan = ZonalInteractionMatrixPlan::new(
                configuration.measurement_count,
                configuration.amplitudes,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = StructuredInteractionMatrixPlanError;
        process |plan, workspace| {
            plan.process(interaction_matrix, workspace, responses)
        }
        process_error = StructuredInteractionMatrixError;
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 decoder for one complete Hadamard push/pull response cycle.
    pub HadamardInteractionMatrixF32 {
        label = "hadamard-interaction-matrix-f32";
        configuration HadamardInteractionMatrixF32Configuration {
            measurement_count: usize = GraphRebuild;
            amplitudes: Vec<f32> = GraphRebuild;
            response_schema: String = GraphRebuild;
            interaction_matrix_schema: String = GraphRebuild;
        }
        plan = HadamardInteractionMatrixPlan<f32>;
        workspace = StructuredInteractionMatrixWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            measurement_count = plan.measurement_count();
            corrector_coordinate_count = plan.corrector_coordinate_count();
            response_step_count = plan.response_count() / plan.measurement_count();
        }
        inputs {
            responses: f32 [response_step_count, measurement_count] = "responses",
                schema = configuration.response_schema.as_str();
        }
        outputs {
            interaction_matrix: f32 [measurement_count, corrector_coordinate_count]
                = "interaction-matrix",
                schema = configuration.interaction_matrix_schema.as_str(),
                metadata_from = "responses";
        }
        parameters {
            amplitudes: f32 [corrector_coordinate_count] = "amplitudes",
                schema = crate::schemas::CALIBRATION_PROBE_AMPLITUDES_V1,
                replace |plan, values| {
                    HadamardInteractionMatrixPlan::new(plan.measurement_count(), values)
                };
        }
        parameter_error = StructuredInteractionMatrixPlanError;
        prepare |configuration| {
            let plan = HadamardInteractionMatrixPlan::new(
                configuration.measurement_count,
                configuration.amplitudes,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = StructuredInteractionMatrixPlanError;
        process |plan, workspace| {
            plan.process(interaction_matrix, workspace, responses)
        }
        process_error = StructuredInteractionMatrixError;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        StructuredInteractionMatrixExtent, StructuredInteractionMatrixPlanError, checked_counts,
        exact_scalar,
    };

    #[test]
    fn response_and_matrix_extent_overflow_are_distinct() {
        assert_eq!(
            checked_counts(usize::MAX, 1, 1),
            Err(StructuredInteractionMatrixPlanError::ExtentOverflow {
                extent: StructuredInteractionMatrixExtent::Responses,
            })
        );
        assert_eq!(
            checked_counts(usize::MAX / 2, 1, 3),
            Err(StructuredInteractionMatrixPlanError::ExtentOverflow {
                extent: StructuredInteractionMatrixExtent::InteractionMatrix,
            })
        );
    }

    #[test]
    fn exact_hadamard_order_conversion_is_checked() {
        assert_eq!(exact_scalar::<f32>(1 << 24), Ok((1 << 24) as f32));
        assert_eq!(
            exact_scalar::<f32>((1 << 24) + 1),
            Err(StructuredInteractionMatrixPlanError::UnrepresentableHadamardOrder)
        );
    }
}
