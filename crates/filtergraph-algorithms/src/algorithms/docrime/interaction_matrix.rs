//! Conditioned open- and closed-loop DO-CRIME interaction-matrix solves.

use alloc::{boxed::Box, string::String, vec};
use core::fmt;

use num_traits::NumCast;

use crate::{AlgorithmPlan, DocrimeCovariances, MatrixElement, PropertyConstraint, PropertyValue};

const DOCRIME_OPEN_LOOP_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_PROPERTY: u32 = 0;
const DOCRIME_OPEN_LOOP_MINIMUM_FACTORIZATION_PIVOT_RATIO_PROPERTY: u32 = 1;
const DOCRIME_CLOSED_LOOP_GAIN_PROPERTY: u32 = 0;
const DOCRIME_CLOSED_LOOP_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_PROPERTY: u32 = 1;
const DOCRIME_CLOSED_LOOP_MINIMUM_FACTORIZATION_PIVOT_RATIO_PROPERTY: u32 = 2;

/// Invalid prepared DO-CRIME conditioning criteria.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeConditioningCriteriaError {
    /// The minimum corrector variance must be finite and positive.
    InvalidMinimumCorrectorVariance,
    /// The maximum absolute corrector correlation must be finite and in `[0, 1)`.
    InvalidMaximumAbsoluteCorrectorCorrelation,
    /// The minimum factorization pivot ratio must be finite and in `(0, 1]`.
    InvalidMinimumFactorizationPivotRatio,
}

impl fmt::Display for DocrimeConditioningCriteriaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMinimumCorrectorVariance => formatter
                .write_str("minimum DO-CRIME corrector variance must be finite and positive"),
            Self::InvalidMaximumAbsoluteCorrectorCorrelation => formatter.write_str(
                "maximum absolute DO-CRIME corrector correlation must be finite and in [0, 1)",
            ),
            Self::InvalidMinimumFactorizationPivotRatio => formatter.write_str(
                "minimum DO-CRIME factorization pivot ratio must be finite and in (0, 1]",
            ),
        }
    }
}

impl core::error::Error for DocrimeConditioningCriteriaError {}

/// Prepared covariance-conditioning acceptance criteria for DO-CRIME.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DocrimeConditioningCriteria<T> {
    minimum_corrector_variance: T,
    maximum_absolute_corrector_correlation: T,
    minimum_factorization_pivot_ratio: T,
}

impl<T> DocrimeConditioningCriteria<T>
where
    T: MatrixElement,
{
    /// Validates all variance, correlation, and factorization thresholds.
    pub fn new(
        minimum_corrector_variance: T,
        maximum_absolute_corrector_correlation: T,
        minimum_factorization_pivot_ratio: T,
    ) -> Result<Self, DocrimeConditioningCriteriaError> {
        if !minimum_corrector_variance.is_finite() || minimum_corrector_variance <= T::zero() {
            return Err(DocrimeConditioningCriteriaError::InvalidMinimumCorrectorVariance);
        }
        if !maximum_absolute_corrector_correlation.is_finite()
            || maximum_absolute_corrector_correlation < T::zero()
            || maximum_absolute_corrector_correlation >= T::one()
        {
            return Err(
                DocrimeConditioningCriteriaError::InvalidMaximumAbsoluteCorrectorCorrelation,
            );
        }
        if !minimum_factorization_pivot_ratio.is_finite()
            || minimum_factorization_pivot_ratio <= T::zero()
            || minimum_factorization_pivot_ratio > T::one()
        {
            return Err(DocrimeConditioningCriteriaError::InvalidMinimumFactorizationPivotRatio);
        }
        Ok(Self {
            minimum_corrector_variance,
            maximum_absolute_corrector_correlation,
            minimum_factorization_pivot_ratio,
        })
    }

    /// Returns the smallest accepted corrector variance.
    pub const fn minimum_corrector_variance(self) -> T {
        self.minimum_corrector_variance
    }

    /// Returns the largest accepted absolute off-diagonal correlation.
    pub const fn maximum_absolute_corrector_correlation(self) -> T {
        self.maximum_absolute_corrector_correlation
    }

    /// Returns the smallest accepted factorization pivot ratio.
    pub const fn minimum_factorization_pivot_ratio(self) -> T {
        self.minimum_factorization_pivot_ratio
    }
}

/// Matrix named by a DO-CRIME interaction-matrix plan error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeInteractionMatrixKind {
    /// Measurement--corrector cross-covariance or interaction matrix.
    MeasurementByCorrector,
    /// Square corrector covariance or factor matrix.
    CorrectorByCorrector,
}

/// Invalid prepared DO-CRIME interaction-matrix solve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeInteractionMatrixPlanError {
    /// Measurement and corrector-coordinate extents must both be nonzero.
    EmptyDimension,
    /// One prepared matrix extent overflowed addressable storage.
    ExtentOverflow {
        /// Matrix whose required extent overflowed.
        matrix: DocrimeInteractionMatrixKind,
    },
    /// A closed-loop plan requires a finite positive scalar loop gain.
    InvalidLoopGain,
}

impl fmt::Display for DocrimeInteractionMatrixPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension => formatter.write_str(
                "DO-CRIME measurement and corrector-coordinate dimensions must be nonzero",
            ),
            Self::ExtentOverflow { matrix } => {
                write!(formatter, "DO-CRIME {matrix:?} matrix extent overflow")
            }
            Self::InvalidLoopGain => {
                formatter.write_str("DO-CRIME closed-loop gain must be finite and positive")
            }
        }
    }
}

impl core::error::Error for DocrimeInteractionMatrixPlanError {}

/// Invalid preparation or runtime-property replacement for a DO-CRIME solve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeInteractionMatrixPreparationError {
    /// One covariance-conditioning criterion is invalid.
    Criteria(DocrimeConditioningCriteriaError),
    /// The declared dimensions or closed-loop gain are invalid.
    Plan(DocrimeInteractionMatrixPlanError),
}

impl fmt::Display for DocrimeInteractionMatrixPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Criteria(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for DocrimeInteractionMatrixPreparationError {}

impl From<DocrimeConditioningCriteriaError> for DocrimeInteractionMatrixPreparationError {
    fn from(error: DocrimeConditioningCriteriaError) -> Self {
        Self::Criteria(error)
    }
}

impl From<DocrimeInteractionMatrixPlanError> for DocrimeInteractionMatrixPreparationError {
    fn from(error: DocrimeInteractionMatrixPlanError) -> Self {
        Self::Plan(error)
    }
}

/// Expected validity of one DO-CRIME interaction-matrix result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeInteractionMatrixStatus {
    /// The interaction matrix and diagnostics are valid.
    Valid,
    /// At least two covariance samples are required.
    InsufficientSamples,
    /// One corrector coordinate has insufficient sample variance.
    InsufficientCorrectorVariance {
        /// Zero-based corrector-coordinate index.
        corrector_coordinate: usize,
    },
    /// One corrector-coordinate pair is too strongly correlated.
    ExcessiveCorrectorCorrelation {
        /// Zero-based first corrector-coordinate index.
        first: usize,
        /// Zero-based second corrector-coordinate index.
        second: usize,
    },
    /// The corrector covariance is not positive definite.
    NonPositiveDefinite {
        /// Zero-based factorization coordinate.
        corrector_coordinate: usize,
    },
    /// One positive factorization pivot is too small relative to maximum variance.
    InsufficientFactorizationPivot {
        /// Zero-based factorization coordinate.
        corrector_coordinate: usize,
    },
}

impl DocrimeInteractionMatrixStatus {
    /// Returns whether the interaction matrix is valid for artifact review.
    pub const fn is_valid(self) -> bool {
        matches!(self, Self::Valid)
    }
}

/// Conditioning diagnostics published with a DO-CRIME interaction matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DocrimeInteractionMatrixDiagnostics<T> {
    status: DocrimeInteractionMatrixStatus,
    sample_count: usize,
    minimum_corrector_variance: T,
    maximum_corrector_variance: T,
    maximum_absolute_corrector_correlation: T,
    minimum_factorization_pivot_ratio: T,
}

impl<T> Default for DocrimeInteractionMatrixDiagnostics<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            status: DocrimeInteractionMatrixStatus::InsufficientSamples,
            sample_count: 0,
            minimum_corrector_variance: T::default(),
            maximum_corrector_variance: T::default(),
            maximum_absolute_corrector_correlation: T::default(),
            minimum_factorization_pivot_ratio: T::default(),
        }
    }
}

impl<T> DocrimeInteractionMatrixDiagnostics<T> {
    /// Returns the validity status for the matrix published with these diagnostics.
    pub const fn status(&self) -> DocrimeInteractionMatrixStatus {
        self.status
    }

    /// Returns the covariance sample count.
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the minimum corrector sample variance, or zero when unavailable.
    pub const fn minimum_corrector_variance(&self) -> T
    where
        T: Copy,
    {
        self.minimum_corrector_variance
    }

    /// Returns the maximum corrector sample variance, or zero when unavailable.
    pub const fn maximum_corrector_variance(&self) -> T
    where
        T: Copy,
    {
        self.maximum_corrector_variance
    }

    /// Returns the maximum absolute off-diagonal corrector correlation.
    pub const fn maximum_absolute_corrector_correlation(&self) -> T
    where
        T: Copy,
    {
        self.maximum_absolute_corrector_correlation
    }

    /// Returns the minimum factorization pivot ratio, or zero when unavailable.
    pub const fn minimum_factorization_pivot_ratio(&self) -> T
    where
        T: Copy,
    {
        self.minimum_factorization_pivot_ratio
    }
}

/// Caller-owned coherent DO-CRIME interaction matrix and diagnostics.
#[derive(Debug)]
pub struct DocrimeInteractionMatrixOutput<'a, T> {
    interaction_matrix: &'a mut [T],
    diagnostics: &'a mut DocrimeInteractionMatrixDiagnostics<T>,
}

impl<'a, T> DocrimeInteractionMatrixOutput<'a, T> {
    /// Groups row-major interaction-matrix storage and its diagnostics.
    pub fn new(
        interaction_matrix: &'a mut [T],
        diagnostics: &'a mut DocrimeInteractionMatrixDiagnostics<T>,
    ) -> Self {
        Self {
            interaction_matrix,
            diagnostics,
        }
    }

    /// Returns the row-major measurement-by-corrector interaction matrix.
    pub fn interaction_matrix(&self) -> &[T] {
        self.interaction_matrix
    }

    /// Returns the diagnostics published with the interaction matrix.
    pub fn diagnostics(&self) -> &DocrimeInteractionMatrixDiagnostics<T> {
        self.diagnostics
    }
}

/// Input covariance matrix named by a DO-CRIME execution error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeInteractionMatrixInputKind {
    /// Measurement--corrector cross-covariance sum.
    MeasurementCorrectorCrossCovarianceSum,
    /// Corrector covariance sum.
    CorrectorCovarianceSum,
}

/// Numerical stage named by a DO-CRIME execution error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeInteractionMatrixStage {
    /// Sample-covariance normalization.
    CovarianceNormalization,
    /// Off-diagonal correlation calculation.
    CorrectorCorrelation,
    /// Cholesky factorization.
    Factorization,
    /// Triangular interaction-matrix solve.
    Solve,
}

/// Invalid DO-CRIME interaction-matrix input, output, workspace, or calculation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeInteractionMatrixError {
    /// One covariance sum has the wrong row-major extent.
    InputExtent {
        /// Input with the wrong extent.
        input: DocrimeInteractionMatrixInputKind,
        /// Required coefficient count.
        expected: usize,
        /// Supplied coefficient count.
        actual: usize,
    },
    /// The caller-owned interaction-matrix output has the wrong extent.
    OutputExtent {
        /// Required coefficient count.
        expected: usize,
        /// Supplied coefficient count.
        actual: usize,
    },
    /// The supplied workspace was prepared for different extents.
    WorkspaceExtent,
    /// One covariance-sum coefficient is not finite.
    NonFiniteInput {
        /// Input containing the coefficient.
        input: DocrimeInteractionMatrixInputKind,
        /// Zero-based coefficient index.
        index: usize,
    },
    /// The corrector covariance sum is not exactly symmetric.
    NonSymmetricCorrectorCovariance {
        /// Zero-based matrix row.
        row: usize,
        /// Zero-based matrix column.
        column: usize,
    },
    /// The sample covariance divisor is not representable by the scalar type.
    SampleCountNotRepresentable,
    /// The U64 sample-count input does not fit the native plan's count type.
    SampleCountOutOfRange,
    /// One numerical stage produced a non-finite value.
    NonFiniteCalculation {
        /// Stage that produced the value.
        stage: DocrimeInteractionMatrixStage,
    },
}

impl fmt::Display for DocrimeInteractionMatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent {
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "DO-CRIME {input:?} extent {actual} does not match {expected}"
            ),
            Self::OutputExtent { expected, actual } => write!(
                formatter,
                "DO-CRIME interaction-matrix output extent {actual} does not match {expected}"
            ),
            Self::WorkspaceExtent => {
                formatter.write_str("DO-CRIME interaction-matrix workspace extent is invalid")
            }
            Self::NonFiniteInput { input, index } => write!(
                formatter,
                "DO-CRIME {input:?} coefficient at index {index} must be finite"
            ),
            Self::NonSymmetricCorrectorCovariance { row, column } => write!(
                formatter,
                "DO-CRIME corrector covariance differs at ({row}, {column}) and ({column}, {row})"
            ),
            Self::SampleCountNotRepresentable => formatter.write_str(
                "DO-CRIME sample covariance divisor is not representable by the scalar type",
            ),
            Self::SampleCountOutOfRange => {
                formatter.write_str("DO-CRIME sample count is outside the native count range")
            }
            Self::NonFiniteCalculation { stage } => write!(
                formatter,
                "DO-CRIME {stage:?} calculation produced a non-finite value"
            ),
        }
    }
}

impl core::error::Error for DocrimeInteractionMatrixError {}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DocrimeInteractionMatrixPreparation<T> {
    measurement_count: usize,
    corrector_coordinate_count: usize,
    interaction_matrix_count: usize,
    corrector_covariance_count: usize,
    criteria: DocrimeConditioningCriteria<T>,
}

impl<T> DocrimeInteractionMatrixPreparation<T>
where
    T: MatrixElement,
{
    fn new(
        measurement_count: usize,
        corrector_coordinate_count: usize,
        criteria: DocrimeConditioningCriteria<T>,
    ) -> Result<Self, DocrimeInteractionMatrixPlanError> {
        if measurement_count == 0 || corrector_coordinate_count == 0 {
            return Err(DocrimeInteractionMatrixPlanError::EmptyDimension);
        }
        let interaction_matrix_count = measurement_count
            .checked_mul(corrector_coordinate_count)
            .ok_or(DocrimeInteractionMatrixPlanError::ExtentOverflow {
                matrix: DocrimeInteractionMatrixKind::MeasurementByCorrector,
            })?;
        let corrector_covariance_count = corrector_coordinate_count
            .checked_mul(corrector_coordinate_count)
            .ok_or(DocrimeInteractionMatrixPlanError::ExtentOverflow {
                matrix: DocrimeInteractionMatrixKind::CorrectorByCorrector,
            })?;
        Ok(Self {
            measurement_count,
            corrector_coordinate_count,
            interaction_matrix_count,
            corrector_covariance_count,
            criteria,
        })
    }

    fn workspace(&self) -> DocrimeInteractionMatrixWorkspace<T> {
        DocrimeInteractionMatrixWorkspace {
            corrector_covariance: vec![T::zero(); self.corrector_covariance_count]
                .into_boxed_slice(),
            factor: vec![T::zero(); self.corrector_covariance_count].into_boxed_slice(),
            staged_interaction_matrix: vec![T::zero(); self.interaction_matrix_count]
                .into_boxed_slice(),
            solve_scratch: vec![T::zero(); self.corrector_coordinate_count].into_boxed_slice(),
        }
    }

    fn validate_workspace(
        &self,
        workspace: &DocrimeInteractionMatrixWorkspace<T>,
    ) -> Result<(), DocrimeInteractionMatrixError> {
        if workspace.corrector_covariance.len() != self.corrector_covariance_count
            || workspace.factor.len() != self.corrector_covariance_count
            || workspace.staged_interaction_matrix.len() != self.interaction_matrix_count
            || workspace.solve_scratch.len() != self.corrector_coordinate_count
        {
            return Err(DocrimeInteractionMatrixError::WorkspaceExtent);
        }
        Ok(())
    }

    fn validate_execution(
        &self,
        output: &DocrimeInteractionMatrixOutput<'_, T>,
        workspace: &DocrimeInteractionMatrixWorkspace<T>,
        input: DocrimeCovariances<'_, T>,
    ) -> Result<(), DocrimeInteractionMatrixError> {
        validate_input_extent(
            DocrimeInteractionMatrixInputKind::MeasurementCorrectorCrossCovarianceSum,
            self.interaction_matrix_count,
            input.measurement_corrector_cross_covariance_sum().len(),
        )?;
        validate_input_extent(
            DocrimeInteractionMatrixInputKind::CorrectorCovarianceSum,
            self.corrector_covariance_count,
            input.corrector_covariance_sum().len(),
        )?;
        if output.interaction_matrix.len() != self.interaction_matrix_count {
            return Err(DocrimeInteractionMatrixError::OutputExtent {
                expected: self.interaction_matrix_count,
                actual: output.interaction_matrix.len(),
            });
        }
        self.validate_workspace(workspace)?;
        validate_finite(
            DocrimeInteractionMatrixInputKind::MeasurementCorrectorCrossCovarianceSum,
            input.measurement_corrector_cross_covariance_sum(),
        )?;
        validate_finite(
            DocrimeInteractionMatrixInputKind::CorrectorCovarianceSum,
            input.corrector_covariance_sum(),
        )?;
        validate_symmetric(
            input.corrector_covariance_sum(),
            self.corrector_coordinate_count,
        )
    }

    fn normalize_corrector_covariance(
        &self,
        workspace: &mut DocrimeInteractionMatrixWorkspace<T>,
        input: DocrimeCovariances<'_, T>,
    ) -> Result<T, DocrimeInteractionMatrixError>
    where
        T: NumCast,
    {
        let divisor_count = input.sample_count() - 1;
        let divisor = <T as NumCast>::from(divisor_count)
            .filter(|value| value.to_usize() == Some(divisor_count))
            .ok_or(DocrimeInteractionMatrixError::SampleCountNotRepresentable)?;
        let normalization = T::one() / divisor;
        for (normalized, sum) in workspace
            .corrector_covariance
            .iter_mut()
            .zip(input.corrector_covariance_sum())
        {
            *normalized = *sum * normalization;
        }
        ensure_finite(
            &workspace.corrector_covariance,
            DocrimeInteractionMatrixStage::CovarianceNormalization,
        )?;
        Ok(normalization)
    }

    fn evaluate_corrector_variances(
        &self,
        corrector_covariance: &[T],
        diagnostics: &mut DocrimeInteractionMatrixDiagnostics<T>,
    ) -> Option<DocrimeInteractionMatrixStatus> {
        let mut minimum_variance = corrector_covariance[0];
        let mut maximum_variance = corrector_covariance[0];
        let mut insufficient_variance = None;
        for corrector_coordinate in 0..self.corrector_coordinate_count {
            let variance = corrector_covariance
                [corrector_coordinate * self.corrector_coordinate_count + corrector_coordinate];
            minimum_variance = minimum_variance.min(variance);
            maximum_variance = maximum_variance.max(variance);
            if insufficient_variance.is_none()
                && variance < self.criteria.minimum_corrector_variance
            {
                insufficient_variance = Some(corrector_coordinate);
            }
        }
        diagnostics.minimum_corrector_variance = minimum_variance;
        diagnostics.maximum_corrector_variance = maximum_variance;
        insufficient_variance.map(|corrector_coordinate| {
            DocrimeInteractionMatrixStatus::InsufficientCorrectorVariance {
                corrector_coordinate,
            }
        })
    }

    fn evaluate_corrector_correlations(
        &self,
        corrector_covariance: &[T],
        diagnostics: &mut DocrimeInteractionMatrixDiagnostics<T>,
    ) -> Result<Option<DocrimeInteractionMatrixStatus>, DocrimeInteractionMatrixError> {
        let mut maximum_correlation = T::zero();
        let mut maximum_correlation_coordinates = (0, 0);
        for row in 0..self.corrector_coordinate_count {
            let row_standard_deviation =
                corrector_covariance[row * self.corrector_coordinate_count + row].sqrt();
            for column in (row + 1)..self.corrector_coordinate_count {
                let column_standard_deviation =
                    corrector_covariance[column * self.corrector_coordinate_count + column].sqrt();
                let covariance =
                    corrector_covariance[row * self.corrector_coordinate_count + column];
                let correlation =
                    (covariance.abs() / row_standard_deviation) / column_standard_deviation;
                if !correlation.is_finite() {
                    return Err(DocrimeInteractionMatrixError::NonFiniteCalculation {
                        stage: DocrimeInteractionMatrixStage::CorrectorCorrelation,
                    });
                }
                if correlation > maximum_correlation {
                    maximum_correlation = correlation;
                    maximum_correlation_coordinates = (row, column);
                }
            }
        }
        diagnostics.maximum_absolute_corrector_correlation = maximum_correlation;
        Ok(
            (maximum_correlation > self.criteria.maximum_absolute_corrector_correlation).then_some(
                DocrimeInteractionMatrixStatus::ExcessiveCorrectorCorrelation {
                    first: maximum_correlation_coordinates.0,
                    second: maximum_correlation_coordinates.1,
                },
            ),
        )
    }

    fn factor_corrector_covariance(
        &self,
        workspace: &mut DocrimeInteractionMatrixWorkspace<T>,
        diagnostics: &mut DocrimeInteractionMatrixDiagnostics<T>,
    ) -> Result<Option<DocrimeInteractionMatrixStatus>, DocrimeInteractionMatrixError> {
        workspace.factor.fill(T::zero());
        let mut minimum_pivot_ratio = T::one();
        for row in 0..self.corrector_coordinate_count {
            for column in 0..=row {
                let mut factor_sum = T::zero();
                for index in 0..column {
                    factor_sum = factor_sum
                        + workspace.factor[row * self.corrector_coordinate_count + index]
                            * workspace.factor[column * self.corrector_coordinate_count + index];
                }
                let factor_coefficient = workspace.corrector_covariance
                    [row * self.corrector_coordinate_count + column]
                    - factor_sum;
                if !factor_coefficient.is_finite() {
                    return Err(DocrimeInteractionMatrixError::NonFiniteCalculation {
                        stage: DocrimeInteractionMatrixStage::Factorization,
                    });
                }
                let factor_index = row * self.corrector_coordinate_count + column;
                if row == column {
                    if factor_coefficient <= T::zero() {
                        diagnostics.minimum_factorization_pivot_ratio = T::zero();
                        return Ok(Some(DocrimeInteractionMatrixStatus::NonPositiveDefinite {
                            corrector_coordinate: row,
                        }));
                    }
                    let pivot_ratio = factor_coefficient / diagnostics.maximum_corrector_variance;
                    if !pivot_ratio.is_finite() {
                        return Err(DocrimeInteractionMatrixError::NonFiniteCalculation {
                            stage: DocrimeInteractionMatrixStage::Factorization,
                        });
                    }
                    minimum_pivot_ratio = minimum_pivot_ratio.min(pivot_ratio);
                    diagnostics.minimum_factorization_pivot_ratio = minimum_pivot_ratio;
                    if pivot_ratio < self.criteria.minimum_factorization_pivot_ratio {
                        return Ok(Some(
                            DocrimeInteractionMatrixStatus::InsufficientFactorizationPivot {
                                corrector_coordinate: row,
                            },
                        ));
                    }
                    workspace.factor[factor_index] = factor_coefficient.sqrt();
                } else {
                    workspace.factor[factor_index] = factor_coefficient
                        / workspace.factor[column * self.corrector_coordinate_count + column];
                }
                if !workspace.factor[factor_index].is_finite() {
                    return Err(DocrimeInteractionMatrixError::NonFiniteCalculation {
                        stage: DocrimeInteractionMatrixStage::Factorization,
                    });
                }
            }
        }
        Ok(None)
    }

    fn solve_interaction_matrix(
        &self,
        response_scale: T,
        normalization: T,
        workspace: &mut DocrimeInteractionMatrixWorkspace<T>,
        input: DocrimeCovariances<'_, T>,
    ) -> Result<(), DocrimeInteractionMatrixError> {
        for measurement in 0..self.measurement_count {
            let row_offset = measurement * self.corrector_coordinate_count;
            for corrector in 0..self.corrector_coordinate_count {
                workspace.solve_scratch[corrector] = input
                    .measurement_corrector_cross_covariance_sum()[row_offset + corrector]
                    * normalization
                    * response_scale;
            }
            ensure_finite(
                &workspace.solve_scratch,
                DocrimeInteractionMatrixStage::Solve,
            )?;

            for row in 0..self.corrector_coordinate_count {
                let mut value = workspace.solve_scratch[row];
                for column in 0..row {
                    value = value
                        - workspace.factor[row * self.corrector_coordinate_count + column]
                            * workspace.solve_scratch[column];
                }
                workspace.solve_scratch[row] =
                    value / workspace.factor[row * self.corrector_coordinate_count + row];
            }
            for row in (0..self.corrector_coordinate_count).rev() {
                let mut value = workspace.solve_scratch[row];
                for column in (row + 1)..self.corrector_coordinate_count {
                    value = value
                        - workspace.factor[column * self.corrector_coordinate_count + row]
                            * workspace.staged_interaction_matrix[row_offset + column];
                }
                workspace.staged_interaction_matrix[row_offset + row] =
                    value / workspace.factor[row * self.corrector_coordinate_count + row];
            }
        }
        ensure_finite(
            &workspace.staged_interaction_matrix,
            DocrimeInteractionMatrixStage::Solve,
        )
    }

    fn process(
        &self,
        response_scale: T,
        output: DocrimeInteractionMatrixOutput<'_, T>,
        workspace: &mut DocrimeInteractionMatrixWorkspace<T>,
        input: DocrimeCovariances<'_, T>,
    ) -> Result<(), DocrimeInteractionMatrixError> {
        self.validate_execution(&output, workspace, input)?;

        workspace.staged_interaction_matrix.fill(T::zero());
        let mut diagnostics = DocrimeInteractionMatrixDiagnostics {
            status: DocrimeInteractionMatrixStatus::InsufficientSamples,
            sample_count: input.sample_count(),
            minimum_corrector_variance: T::zero(),
            maximum_corrector_variance: T::zero(),
            maximum_absolute_corrector_correlation: T::zero(),
            minimum_factorization_pivot_ratio: T::zero(),
        };
        if input.sample_count() < 2 {
            publish(output, workspace, diagnostics);
            return Ok(());
        }

        let normalization = self.normalize_corrector_covariance(workspace, input)?;
        if let Some(status) =
            self.evaluate_corrector_variances(&workspace.corrector_covariance, &mut diagnostics)
        {
            diagnostics.status = status;
            publish(output, workspace, diagnostics);
            return Ok(());
        }

        if let Some(status) =
            self.evaluate_corrector_correlations(&workspace.corrector_covariance, &mut diagnostics)?
        {
            diagnostics.status = status;
            publish(output, workspace, diagnostics);
            return Ok(());
        }

        if let Some(status) = self.factor_corrector_covariance(workspace, &mut diagnostics)? {
            diagnostics.status = status;
            publish(output, workspace, diagnostics);
            return Ok(());
        }

        self.solve_interaction_matrix(response_scale, normalization, workspace, input)?;

        diagnostics.status = DocrimeInteractionMatrixStatus::Valid;
        publish(output, workspace, diagnostics);
        Ok(())
    }
}

/// Prepared open-loop DO-CRIME interaction-matrix solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DocrimeOpenLoopInteractionMatrixPlan<T> {
    preparation: DocrimeInteractionMatrixPreparation<T>,
}

impl<T> DocrimeOpenLoopInteractionMatrixPlan<T>
where
    T: MatrixElement,
{
    /// Validates dimensions and retains prepared conditioning criteria.
    pub fn new(
        measurement_count: usize,
        corrector_coordinate_count: usize,
        criteria: DocrimeConditioningCriteria<T>,
    ) -> Result<Self, DocrimeInteractionMatrixPlanError> {
        Ok(Self {
            preparation: DocrimeInteractionMatrixPreparation::new(
                measurement_count,
                corrector_coordinate_count,
                criteria,
            )?,
        })
    }

    /// Returns the measurement-vector and interaction-matrix row count.
    pub const fn measurement_count(&self) -> usize {
        self.preparation.measurement_count
    }

    /// Returns the corrector-coordinate and interaction-matrix column count.
    pub const fn corrector_coordinate_count(&self) -> usize {
        self.preparation.corrector_coordinate_count
    }

    /// Returns the prepared conditioning criteria.
    pub const fn criteria(&self) -> DocrimeConditioningCriteria<T> {
        self.preparation.criteria
    }

    /// Allocates fixed factorization and solve scratch for one owner.
    pub fn workspace(&self) -> DocrimeInteractionMatrixWorkspace<T> {
        self.preparation.workspace()
    }
}

fn replace_docrime_open_loop_properties<T>(
    plan: &DocrimeOpenLoopInteractionMatrixPlan<T>,
    maximum_absolute_corrector_correlation: T,
    minimum_factorization_pivot_ratio: T,
) -> Result<DocrimeOpenLoopInteractionMatrixPlan<T>, DocrimeConditioningCriteriaError>
where
    T: MatrixElement,
{
    let criteria = DocrimeConditioningCriteria::new(
        plan.criteria().minimum_corrector_variance(),
        maximum_absolute_corrector_correlation,
        minimum_factorization_pivot_ratio,
    )?;
    Ok(DocrimeOpenLoopInteractionMatrixPlan {
        preparation: DocrimeInteractionMatrixPreparation {
            criteria,
            ..plan.preparation
        },
    })
}

macro_rules! impl_docrime_open_loop_properties {
    ($scalar:ty, $value:ident, $correlation_maximum:expr, $pivot_minimum:expr) => {
        crate::algorithm_properties! {
            impl DocrimeOpenLoopInteractionMatrixPlan<$scalar> {
                plan_error = DocrimeConditioningCriteriaError;
                replace = replace_docrime_open_loop_properties;
                properties {
                    DOCRIME_OPEN_LOOP_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_PROPERTY =>
                        maximum_absolute_corrector_correlation
                        from preparation.criteria.maximum_absolute_corrector_correlation: $value {
                        name = "maximum-absolute-corrector-correlation";
                        description = "Largest accepted absolute corrector correlation";
                        unit = "1";
                        update = Runtime;
                        default = 0.9;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(0.0),
                            maximum: PropertyValue::$value($correlation_maximum),
                        };
                    }
                    DOCRIME_OPEN_LOOP_MINIMUM_FACTORIZATION_PIVOT_RATIO_PROPERTY =>
                        minimum_factorization_pivot_ratio
                        from preparation.criteria.minimum_factorization_pivot_ratio: $value {
                        name = "minimum-factorization-pivot-ratio";
                        description = "Smallest accepted factorization pivot ratio";
                        unit = "1";
                        update = Runtime;
                        default = 0.01;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value($pivot_minimum),
                            maximum: PropertyValue::$value(1.0),
                        };
                    }
                }
            }
        }
    };
}

impl_docrime_open_loop_properties!(
    f32,
    Float,
    f32::from_bits(1.0_f32.to_bits() - 1),
    f32::from_bits(1)
);
impl_docrime_open_loop_properties!(
    f64,
    Double,
    f64::from_bits(1.0_f64.to_bits() - 1),
    f64::from_bits(1)
);

impl<T> AlgorithmPlan for DocrimeOpenLoopInteractionMatrixPlan<T>
where
    T: MatrixElement + NumCast + 'static,
{
    type Input<'a>
        = DocrimeCovariances<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = DocrimeInteractionMatrixOutput<'a, T>
    where
        Self: 'a;
    type Workspace = DocrimeInteractionMatrixWorkspace<T>;
    type Error = DocrimeInteractionMatrixError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        self.preparation.process(T::one(), output, workspace, input)
    }
}

/// Prepared closed-loop DO-CRIME interaction-matrix solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DocrimeClosedLoopInteractionMatrixPlan<T> {
    preparation: DocrimeInteractionMatrixPreparation<T>,
    loop_gain: T,
}

impl<T> DocrimeClosedLoopInteractionMatrixPlan<T>
where
    T: MatrixElement,
{
    /// Validates dimensions and loop gain and retains conditioning criteria.
    pub fn new(
        measurement_count: usize,
        corrector_coordinate_count: usize,
        loop_gain: T,
        criteria: DocrimeConditioningCriteria<T>,
    ) -> Result<Self, DocrimeInteractionMatrixPlanError> {
        if !loop_gain.is_finite() || loop_gain <= T::zero() {
            return Err(DocrimeInteractionMatrixPlanError::InvalidLoopGain);
        }
        Ok(Self {
            preparation: DocrimeInteractionMatrixPreparation::new(
                measurement_count,
                corrector_coordinate_count,
                criteria,
            )?,
            loop_gain,
        })
    }

    /// Returns the measurement-vector and interaction-matrix row count.
    pub const fn measurement_count(&self) -> usize {
        self.preparation.measurement_count
    }

    /// Returns the corrector-coordinate and interaction-matrix column count.
    pub const fn corrector_coordinate_count(&self) -> usize {
        self.preparation.corrector_coordinate_count
    }

    /// Returns the prepared scalar loop gain.
    pub const fn loop_gain(&self) -> T {
        self.loop_gain
    }

    /// Returns the prepared conditioning criteria.
    pub const fn criteria(&self) -> DocrimeConditioningCriteria<T> {
        self.preparation.criteria
    }

    /// Allocates fixed factorization and solve scratch for one owner.
    pub fn workspace(&self) -> DocrimeInteractionMatrixWorkspace<T> {
        self.preparation.workspace()
    }
}

fn replace_docrime_closed_loop_properties<T>(
    plan: &DocrimeClosedLoopInteractionMatrixPlan<T>,
    loop_gain: T,
    maximum_absolute_corrector_correlation: T,
    minimum_factorization_pivot_ratio: T,
) -> Result<DocrimeClosedLoopInteractionMatrixPlan<T>, DocrimeInteractionMatrixPreparationError>
where
    T: MatrixElement,
{
    let criteria = DocrimeConditioningCriteria::new(
        plan.criteria().minimum_corrector_variance(),
        maximum_absolute_corrector_correlation,
        minimum_factorization_pivot_ratio,
    )?;
    Ok(DocrimeClosedLoopInteractionMatrixPlan::new(
        plan.measurement_count(),
        plan.corrector_coordinate_count(),
        loop_gain,
        criteria,
    )?)
}

macro_rules! impl_docrime_closed_loop_properties {
    ($scalar:ty, $value:ident, $correlation_maximum:expr, $positive_minimum:expr) => {
        crate::algorithm_properties! {
            impl DocrimeClosedLoopInteractionMatrixPlan<$scalar> {
                plan_error = DocrimeInteractionMatrixPreparationError;
                replace = replace_docrime_closed_loop_properties;
                properties {
                    DOCRIME_CLOSED_LOOP_GAIN_PROPERTY => loop_gain: $value {
                        name = "loop-gain";
                        description = "Closed-loop response scale";
                        unit = "1";
                        update = Runtime;
                        default = 1.0;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value($positive_minimum),
                            maximum: PropertyValue::$value(<$scalar>::MAX),
                        };
                    }
                    DOCRIME_CLOSED_LOOP_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_PROPERTY =>
                        maximum_absolute_corrector_correlation
                        from preparation.criteria.maximum_absolute_corrector_correlation: $value {
                        name = "maximum-absolute-corrector-correlation";
                        description = "Largest accepted absolute corrector correlation";
                        unit = "1";
                        update = Runtime;
                        default = 0.9;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(0.0),
                            maximum: PropertyValue::$value($correlation_maximum),
                        };
                    }
                    DOCRIME_CLOSED_LOOP_MINIMUM_FACTORIZATION_PIVOT_RATIO_PROPERTY =>
                        minimum_factorization_pivot_ratio
                        from preparation.criteria.minimum_factorization_pivot_ratio: $value {
                        name = "minimum-factorization-pivot-ratio";
                        description = "Smallest accepted factorization pivot ratio";
                        unit = "1";
                        update = Runtime;
                        default = 0.01;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value($positive_minimum),
                            maximum: PropertyValue::$value(1.0),
                        };
                    }
                }
            }
        }
    };
}

impl_docrime_closed_loop_properties!(
    f32,
    Float,
    f32::from_bits(1.0_f32.to_bits() - 1),
    f32::from_bits(1)
);
impl_docrime_closed_loop_properties!(
    f64,
    Double,
    f64::from_bits(1.0_f64.to_bits() - 1),
    f64::from_bits(1)
);

impl<T> AlgorithmPlan for DocrimeClosedLoopInteractionMatrixPlan<T>
where
    T: MatrixElement + NumCast + 'static,
{
    type Input<'a>
        = DocrimeCovariances<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = DocrimeInteractionMatrixOutput<'a, T>
    where
        Self: 'a;
    type Workspace = DocrimeInteractionMatrixWorkspace<T>;
    type Error = DocrimeInteractionMatrixError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        self.preparation
            .process(self.loop_gain, output, workspace, input)
    }
}

/// Fixed covariance, factorization, and solve scratch for DO-CRIME.
#[derive(Clone, Debug, PartialEq)]
pub struct DocrimeInteractionMatrixWorkspace<T> {
    corrector_covariance: Box<[T]>,
    factor: Box<[T]>,
    staged_interaction_matrix: Box<[T]>,
    solve_scratch: Box<[T]>,
}

fn validate_input_extent(
    input: DocrimeInteractionMatrixInputKind,
    expected: usize,
    actual: usize,
) -> Result<(), DocrimeInteractionMatrixError> {
    if actual != expected {
        return Err(DocrimeInteractionMatrixError::InputExtent {
            input,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_finite<T>(
    input: DocrimeInteractionMatrixInputKind,
    values: &[T],
) -> Result<(), DocrimeInteractionMatrixError>
where
    T: MatrixElement,
{
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(DocrimeInteractionMatrixError::NonFiniteInput { input, index });
    }
    Ok(())
}

fn validate_symmetric<T>(
    covariance: &[T],
    corrector_coordinate_count: usize,
) -> Result<(), DocrimeInteractionMatrixError>
where
    T: MatrixElement,
{
    for row in 0..corrector_coordinate_count {
        for column in (row + 1)..corrector_coordinate_count {
            if covariance[row * corrector_coordinate_count + column]
                != covariance[column * corrector_coordinate_count + row]
            {
                return Err(
                    DocrimeInteractionMatrixError::NonSymmetricCorrectorCovariance { row, column },
                );
            }
        }
    }
    Ok(())
}

fn ensure_finite<T>(
    values: &[T],
    stage: DocrimeInteractionMatrixStage,
) -> Result<(), DocrimeInteractionMatrixError>
where
    T: MatrixElement,
{
    if values.iter().any(|value| !value.is_finite()) {
        return Err(DocrimeInteractionMatrixError::NonFiniteCalculation { stage });
    }
    Ok(())
}

fn publish<T>(
    output: DocrimeInteractionMatrixOutput<'_, T>,
    workspace: &DocrimeInteractionMatrixWorkspace<T>,
    diagnostics: DocrimeInteractionMatrixDiagnostics<T>,
) where
    T: Copy,
{
    output
        .interaction_matrix
        .copy_from_slice(&workspace.staged_interaction_matrix);
    *output.diagnostics = diagnostics;
}

fn encode_docrime_status(status: DocrimeInteractionMatrixStatus) -> (u8, [u64; 2]) {
    match status {
        DocrimeInteractionMatrixStatus::Valid => (0, [0, 0]),
        DocrimeInteractionMatrixStatus::InsufficientSamples => (1, [0, 0]),
        DocrimeInteractionMatrixStatus::InsufficientCorrectorVariance {
            corrector_coordinate,
        } => (2, [corrector_coordinate as u64; 2]),
        DocrimeInteractionMatrixStatus::ExcessiveCorrectorCorrelation { first, second } => {
            (3, [first as u64, second as u64])
        }
        DocrimeInteractionMatrixStatus::NonPositiveDefinite {
            corrector_coordinate,
        } => (4, [corrector_coordinate as u64; 2]),
        DocrimeInteractionMatrixStatus::InsufficientFactorizationPivot {
            corrector_coordinate,
        } => (5, [corrector_coordinate as u64; 2]),
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 open-loop DO-CRIME interaction-matrix solve declaration.
    pub DocrimeOpenLoopInteractionMatrixF32 {
        label = "docrime-open-loop-interaction-matrix-f32";
        configuration DocrimeOpenLoopInteractionMatrixF32Configuration {
            measurement_count: usize = GraphRebuild;
            corrector_coordinate_count: usize = GraphRebuild;
            minimum_corrector_variance: f32 = Construction;
            cross_covariance_schema: String = GraphRebuild;
            corrector_covariance_schema: String = GraphRebuild;
            interaction_matrix_schema: String = GraphRebuild;
        }
        plan = DocrimeOpenLoopInteractionMatrixPlan<f32>;
        workspace = DocrimeInteractionMatrixWorkspace<f32>;
        properties = runtime;
        dimensions |plan, _workspace| {
            measurement_count = plan.measurement_count();
            corrector_coordinate_count = plan.corrector_coordinate_count();
        }
        inputs {
            measurement_corrector_cross_covariance_sum: f32
                [measurement_count, corrector_coordinate_count]
                = "measurement-corrector-cross-covariance-sum",
                schema = configuration.cross_covariance_schema.as_str();
            corrector_covariance_sum: f32
                [corrector_coordinate_count, corrector_coordinate_count]
                = "corrector-covariance-sum",
                schema = configuration.corrector_covariance_schema.as_str();
            sample_count: u64 [1] = "sample-count",
                schema = crate::schemas::DOCRIME_COVARIANCE_SAMPLE_COUNT_V1;
        }
        outputs {
            interaction_matrix: f32 [measurement_count, corrector_coordinate_count]
                = "interaction-matrix",
                schema = configuration.interaction_matrix_schema.as_str(),
                metadata_from = "measurement-corrector-cross-covariance-sum";
            status: u8 [1] = "status",
                schema = crate::schemas::DOCRIME_INTERACTION_MATRIX_STATUS_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
            status_coordinates: u64 [2] = "status-coordinates",
                schema = crate::schemas::DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
            diagnostic_sample_count: u64 [1] = "diagnostic-sample-count",
                schema = crate::schemas::DOCRIME_COVARIANCE_SAMPLE_COUNT_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
            corrector_variance_range: f32 [2] = "corrector-variance-range",
                schema = crate::schemas::DOCRIME_CORRECTOR_VARIANCE_RANGE_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
            maximum_absolute_corrector_correlation: f32 [1]
                = "maximum-absolute-corrector-correlation",
                schema = crate::schemas::DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
            minimum_factorization_pivot_ratio: f32 [1]
                = "minimum-factorization-pivot-ratio",
                schema = crate::schemas::DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
        }
        prepare |configuration| {
            let criteria = DocrimeConditioningCriteria::new(
                configuration.minimum_corrector_variance,
                0.9,
                0.01,
            )?;
            let plan = DocrimeOpenLoopInteractionMatrixPlan::new(
                configuration.measurement_count,
                configuration.corrector_coordinate_count,
                criteria,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = DocrimeInteractionMatrixPreparationError;
        process |plan, workspace| {
            let input_sample_count = usize::try_from(sample_count[0])
                .map_err(|_| DocrimeInteractionMatrixError::SampleCountOutOfRange)?;
            let mut diagnostics = DocrimeInteractionMatrixDiagnostics::default();
            plan.process(
                DocrimeInteractionMatrixOutput::new(interaction_matrix, &mut diagnostics),
                workspace,
                DocrimeCovariances::new(
                    input_sample_count,
                    measurement_corrector_cross_covariance_sum,
                    corrector_covariance_sum,
                ),
            )?;
            let (status_value, coordinates) = encode_docrime_status(diagnostics.status());
            status[0] = status_value;
            status_coordinates.copy_from_slice(&coordinates);
            diagnostic_sample_count[0] = diagnostics.sample_count() as u64;
            corrector_variance_range[0] = diagnostics.minimum_corrector_variance();
            corrector_variance_range[1] = diagnostics.maximum_corrector_variance();
            maximum_absolute_corrector_correlation[0] =
                diagnostics.maximum_absolute_corrector_correlation();
            minimum_factorization_pivot_ratio[0] =
                diagnostics.minimum_factorization_pivot_ratio();
            Ok(())
        }
        process_error = DocrimeInteractionMatrixError;
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 closed-loop DO-CRIME interaction-matrix solve declaration.
    pub DocrimeClosedLoopInteractionMatrixF32 {
        label = "docrime-closed-loop-interaction-matrix-f32";
        configuration DocrimeClosedLoopInteractionMatrixF32Configuration {
            measurement_count: usize = GraphRebuild;
            corrector_coordinate_count: usize = GraphRebuild;
            minimum_corrector_variance: f32 = Construction;
            cross_covariance_schema: String = GraphRebuild;
            corrector_covariance_schema: String = GraphRebuild;
            interaction_matrix_schema: String = GraphRebuild;
        }
        plan = DocrimeClosedLoopInteractionMatrixPlan<f32>;
        workspace = DocrimeInteractionMatrixWorkspace<f32>;
        properties = runtime;
        dimensions |plan, _workspace| {
            measurement_count = plan.measurement_count();
            corrector_coordinate_count = plan.corrector_coordinate_count();
        }
        inputs {
            measurement_corrector_cross_covariance_sum: f32
                [measurement_count, corrector_coordinate_count]
                = "measurement-corrector-cross-covariance-sum",
                schema = configuration.cross_covariance_schema.as_str();
            corrector_covariance_sum: f32
                [corrector_coordinate_count, corrector_coordinate_count]
                = "corrector-covariance-sum",
                schema = configuration.corrector_covariance_schema.as_str();
            sample_count: u64 [1] = "sample-count",
                schema = crate::schemas::DOCRIME_COVARIANCE_SAMPLE_COUNT_V1;
        }
        outputs {
            interaction_matrix: f32 [measurement_count, corrector_coordinate_count]
                = "interaction-matrix",
                schema = configuration.interaction_matrix_schema.as_str(),
                metadata_from = "measurement-corrector-cross-covariance-sum";
            status: u8 [1] = "status",
                schema = crate::schemas::DOCRIME_INTERACTION_MATRIX_STATUS_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
            status_coordinates: u64 [2] = "status-coordinates",
                schema = crate::schemas::DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
            diagnostic_sample_count: u64 [1] = "diagnostic-sample-count",
                schema = crate::schemas::DOCRIME_COVARIANCE_SAMPLE_COUNT_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
            corrector_variance_range: f32 [2] = "corrector-variance-range",
                schema = crate::schemas::DOCRIME_CORRECTOR_VARIANCE_RANGE_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
            maximum_absolute_corrector_correlation: f32 [1]
                = "maximum-absolute-corrector-correlation",
                schema = crate::schemas::DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
            minimum_factorization_pivot_ratio: f32 [1]
                = "minimum-factorization-pivot-ratio",
                schema = crate::schemas::DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1,
                metadata_from = "measurement-corrector-cross-covariance-sum";
        }
        prepare |configuration| {
            let criteria = DocrimeConditioningCriteria::new(
                configuration.minimum_corrector_variance,
                0.9,
                0.01,
            )?;
            let plan = DocrimeClosedLoopInteractionMatrixPlan::new(
                configuration.measurement_count,
                configuration.corrector_coordinate_count,
                1.0,
                criteria,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = DocrimeInteractionMatrixPreparationError;
        process |plan, workspace| {
            let input_sample_count = usize::try_from(sample_count[0])
                .map_err(|_| DocrimeInteractionMatrixError::SampleCountOutOfRange)?;
            let mut diagnostics = DocrimeInteractionMatrixDiagnostics::default();
            plan.process(
                DocrimeInteractionMatrixOutput::new(interaction_matrix, &mut diagnostics),
                workspace,
                DocrimeCovariances::new(
                    input_sample_count,
                    measurement_corrector_cross_covariance_sum,
                    corrector_covariance_sum,
                ),
            )?;
            let (status_value, coordinates) = encode_docrime_status(diagnostics.status());
            status[0] = status_value;
            status_coordinates.copy_from_slice(&coordinates);
            diagnostic_sample_count[0] = diagnostics.sample_count() as u64;
            corrector_variance_range[0] = diagnostics.minimum_corrector_variance();
            corrector_variance_range[1] = diagnostics.maximum_corrector_variance();
            maximum_absolute_corrector_correlation[0] =
                diagnostics.maximum_absolute_corrector_correlation();
            minimum_factorization_pivot_ratio[0] =
                diagnostics.minimum_factorization_pivot_ratio();
            Ok(())
        }
        process_error = DocrimeInteractionMatrixError;
    }
}
