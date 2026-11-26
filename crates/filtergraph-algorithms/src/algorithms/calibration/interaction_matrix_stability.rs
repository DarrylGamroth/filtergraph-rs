//! Independent-window interaction-matrix stability diagnostics.

use alloc::string::String;
use core::{fmt, marker::PhantomData};

use crate::{AlgorithmPlan, MatrixElement, NoWorkspace, PropertyConstraint, PropertyValue};

const MAXIMUM_RELATIVE_COLUMN_DIFFERENCE_PROPERTY: u32 = 0;
const MAXIMUM_RELATIVE_FROBENIUS_DIFFERENCE_PROPERTY: u32 = 1;

/// Invalid prepared interaction-matrix stability criteria.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InteractionMatrixStabilityCriteriaError {
    /// The minimum column norm must be finite and positive.
    InvalidMinimumColumnNorm,
    /// The maximum relative column difference must be finite and in `[0, 2]`.
    InvalidMaximumRelativeColumnDifference,
    /// The maximum relative Frobenius difference must be finite and in `[0, 2]`.
    InvalidMaximumRelativeFrobeniusDifference,
}

impl fmt::Display for InteractionMatrixStabilityCriteriaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMinimumColumnNorm => {
                formatter.write_str("minimum interaction-matrix column norm must be positive")
            }
            Self::InvalidMaximumRelativeColumnDifference => formatter.write_str(
                "maximum relative interaction-matrix column difference must be in [0, 2]",
            ),
            Self::InvalidMaximumRelativeFrobeniusDifference => formatter.write_str(
                "maximum relative interaction-matrix Frobenius difference must be in [0, 2]",
            ),
        }
    }
}

impl core::error::Error for InteractionMatrixStabilityCriteriaError {}

/// Prepared acceptance criteria for independent-window matrix stability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InteractionMatrixStabilityCriteria<T> {
    minimum_column_norm: T,
    maximum_relative_column_difference: T,
    maximum_relative_frobenius_difference: T,
}

impl<T> InteractionMatrixStabilityCriteria<T>
where
    T: MatrixElement,
{
    /// Validates the minimum energy and symmetric relative-difference limits.
    pub fn new(
        minimum_column_norm: T,
        maximum_relative_column_difference: T,
        maximum_relative_frobenius_difference: T,
    ) -> Result<Self, InteractionMatrixStabilityCriteriaError> {
        let two = T::one() + T::one();
        if !minimum_column_norm.is_finite() || minimum_column_norm <= T::zero() {
            return Err(InteractionMatrixStabilityCriteriaError::InvalidMinimumColumnNorm);
        }
        if !maximum_relative_column_difference.is_finite()
            || maximum_relative_column_difference < T::zero()
            || maximum_relative_column_difference > two
        {
            return Err(
                InteractionMatrixStabilityCriteriaError::InvalidMaximumRelativeColumnDifference,
            );
        }
        if !maximum_relative_frobenius_difference.is_finite()
            || maximum_relative_frobenius_difference < T::zero()
            || maximum_relative_frobenius_difference > two
        {
            return Err(
                InteractionMatrixStabilityCriteriaError::InvalidMaximumRelativeFrobeniusDifference,
            );
        }
        Ok(Self {
            minimum_column_norm,
            maximum_relative_column_difference,
            maximum_relative_frobenius_difference,
        })
    }

    /// Returns the smallest accepted Euclidean norm for each matrix column.
    pub const fn minimum_column_norm(self) -> T {
        self.minimum_column_norm
    }

    /// Returns the largest accepted symmetric relative column difference.
    pub const fn maximum_relative_column_difference(self) -> T {
        self.maximum_relative_column_difference
    }

    /// Returns the largest accepted symmetric relative Frobenius difference.
    pub const fn maximum_relative_frobenius_difference(self) -> T {
        self.maximum_relative_frobenius_difference
    }
}

/// Input matrix named by an interaction-matrix stability error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InteractionMatrixStabilityInputKind {
    /// Matrix identified from the first independent acquisition window.
    FirstWindow,
    /// Matrix identified from the second independent acquisition window.
    SecondWindow,
}

/// Invalid prepared interaction-matrix stability operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InteractionMatrixStabilityPlanError {
    /// Measurement and corrector-coordinate extents must both be nonzero.
    EmptyDimension,
    /// The measurement-by-corrector matrix extent overflowed addressable storage.
    ExtentOverflow,
}

impl fmt::Display for InteractionMatrixStabilityPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension => {
                formatter.write_str("interaction-matrix stability dimensions must both be nonzero")
            }
            Self::ExtentOverflow => {
                formatter.write_str("interaction-matrix stability extent overflow")
            }
        }
    }
}

impl core::error::Error for InteractionMatrixStabilityPlanError {}

/// Invalid preparation of the portable Float32 stability declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InteractionMatrixStabilityF32PreparationError {
    /// One scalar acceptance criterion is invalid.
    Criteria(InteractionMatrixStabilityCriteriaError),
    /// The declared matrix dimensions are invalid.
    Plan(InteractionMatrixStabilityPlanError),
}

impl fmt::Display for InteractionMatrixStabilityF32PreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Criteria(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for InteractionMatrixStabilityF32PreparationError {}

impl From<InteractionMatrixStabilityCriteriaError>
    for InteractionMatrixStabilityF32PreparationError
{
    fn from(error: InteractionMatrixStabilityCriteriaError) -> Self {
        Self::Criteria(error)
    }
}

impl From<InteractionMatrixStabilityPlanError> for InteractionMatrixStabilityF32PreparationError {
    fn from(error: InteractionMatrixStabilityPlanError) -> Self {
        Self::Plan(error)
    }
}

/// Invalid interaction-matrix stability input or calculation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InteractionMatrixStabilityError {
    /// One input matrix has the wrong row-major extent.
    InputExtent {
        /// Matrix with the wrong extent.
        input: InteractionMatrixStabilityInputKind,
        /// Required coefficient count.
        expected: usize,
        /// Supplied coefficient count.
        actual: usize,
    },
    /// One matrix coefficient is not finite.
    NonFiniteInput {
        /// Matrix containing the coefficient.
        input: InteractionMatrixStabilityInputKind,
        /// Zero-based row-major coefficient index.
        index: usize,
    },
    /// A norm or relative-difference calculation was not finite.
    NonFiniteCalculation,
}

impl fmt::Display for InteractionMatrixStabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent {
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "interaction-matrix {input:?} extent {actual} does not match {expected}"
            ),
            Self::NonFiniteInput { input, index } => write!(
                formatter,
                "interaction-matrix {input:?} coefficient at index {index} must be finite"
            ),
            Self::NonFiniteCalculation => formatter
                .write_str("interaction-matrix stability calculation produced a non-finite value"),
        }
    }
}

impl core::error::Error for InteractionMatrixStabilityError {}

/// Expected validity of one independent-window stability result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InteractionMatrixStabilityStatus {
    /// Both matrices satisfy the prepared energy and difference criteria.
    Valid,
    /// At least one matrix has insufficient energy in the named column.
    InsufficientColumnNorm {
        /// Lowest corrector coordinate failing the norm criterion.
        corrector_coordinate: usize,
    },
    /// The named column has excessive symmetric relative difference.
    ExcessiveRelativeColumnDifference {
        /// Lowest coordinate attaining the maximum relative difference.
        corrector_coordinate: usize,
    },
    /// The complete matrices have excessive symmetric relative difference.
    ExcessiveRelativeFrobeniusDifference,
}

impl InteractionMatrixStabilityStatus {
    /// Returns whether both matrices satisfy every prepared criterion.
    pub const fn is_valid(self) -> bool {
        matches!(self, Self::Valid)
    }
}

/// Diagnostics comparing two independently identified interaction matrices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InteractionMatrixStabilityDiagnostics<T> {
    status: InteractionMatrixStabilityStatus,
    minimum_first_window_column_norm: T,
    minimum_second_window_column_norm: T,
    maximum_relative_column_difference: T,
    maximum_relative_column_difference_coordinate: usize,
    relative_frobenius_difference: T,
}

impl<T> Default for InteractionMatrixStabilityDiagnostics<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            status: InteractionMatrixStabilityStatus::InsufficientColumnNorm {
                corrector_coordinate: 0,
            },
            minimum_first_window_column_norm: T::default(),
            minimum_second_window_column_norm: T::default(),
            maximum_relative_column_difference: T::default(),
            maximum_relative_column_difference_coordinate: 0,
            relative_frobenius_difference: T::default(),
        }
    }
}

impl<T> InteractionMatrixStabilityDiagnostics<T> {
    /// Returns the deterministic stability status.
    pub const fn status(&self) -> InteractionMatrixStabilityStatus {
        self.status
    }

    /// Returns the smallest column norm in the first-window matrix.
    pub const fn minimum_first_window_column_norm(&self) -> T
    where
        T: Copy,
    {
        self.minimum_first_window_column_norm
    }

    /// Returns the smallest column norm in the second-window matrix.
    pub const fn minimum_second_window_column_norm(&self) -> T
    where
        T: Copy,
    {
        self.minimum_second_window_column_norm
    }

    /// Returns the maximum symmetric relative column difference.
    pub const fn maximum_relative_column_difference(&self) -> T
    where
        T: Copy,
    {
        self.maximum_relative_column_difference
    }

    /// Returns the lowest coordinate attaining the maximum column difference.
    pub const fn maximum_relative_column_difference_coordinate(&self) -> usize {
        self.maximum_relative_column_difference_coordinate
    }

    /// Returns the symmetric relative Frobenius difference.
    pub const fn relative_frobenius_difference(&self) -> T
    where
        T: Copy,
    {
        self.relative_frobenius_difference
    }
}

/// Pair of row-major interaction matrices from independent windows.
#[derive(Clone, Copy, Debug)]
pub struct InteractionMatrixStabilityInput<'a, T> {
    first_window: &'a [T],
    second_window: &'a [T],
}

impl<'a, T> InteractionMatrixStabilityInput<'a, T> {
    /// Groups the two matrices to compare.
    pub const fn new(first_window: &'a [T], second_window: &'a [T]) -> Self {
        Self {
            first_window,
            second_window,
        }
    }

    /// Returns the first-window row-major matrix.
    pub const fn first_window(&self) -> &[T] {
        self.first_window
    }

    /// Returns the second-window row-major matrix.
    pub const fn second_window(&self) -> &[T] {
        self.second_window
    }
}

/// Prepared independent-window interaction-matrix stability comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InteractionMatrixStabilityPlan<T> {
    measurement_count: usize,
    corrector_coordinate_count: usize,
    matrix_count: usize,
    criteria: InteractionMatrixStabilityCriteria<T>,
    scalar: PhantomData<T>,
}

impl<T> InteractionMatrixStabilityPlan<T>
where
    T: MatrixElement,
{
    /// Validates dimensions and retains the prepared stability criteria.
    pub fn new(
        measurement_count: usize,
        corrector_coordinate_count: usize,
        criteria: InteractionMatrixStabilityCriteria<T>,
    ) -> Result<Self, InteractionMatrixStabilityPlanError> {
        if measurement_count == 0 || corrector_coordinate_count == 0 {
            return Err(InteractionMatrixStabilityPlanError::EmptyDimension);
        }
        let matrix_count = measurement_count
            .checked_mul(corrector_coordinate_count)
            .ok_or(InteractionMatrixStabilityPlanError::ExtentOverflow)?;
        Ok(Self {
            measurement_count,
            corrector_coordinate_count,
            matrix_count,
            criteria,
            scalar: PhantomData,
        })
    }

    /// Returns the measurement-vector extent.
    pub const fn measurement_count(&self) -> usize {
        self.measurement_count
    }

    /// Returns the corrector-coordinate extent.
    pub const fn corrector_coordinate_count(&self) -> usize {
        self.corrector_coordinate_count
    }

    /// Returns the prepared acceptance criteria.
    pub const fn criteria(&self) -> InteractionMatrixStabilityCriteria<T> {
        self.criteria
    }
}

fn replace_interaction_matrix_stability_properties<T>(
    plan: &InteractionMatrixStabilityPlan<T>,
    maximum_relative_column_difference: T,
    maximum_relative_frobenius_difference: T,
) -> Result<InteractionMatrixStabilityPlan<T>, InteractionMatrixStabilityCriteriaError>
where
    T: MatrixElement,
{
    let criteria = InteractionMatrixStabilityCriteria::new(
        plan.criteria.minimum_column_norm,
        maximum_relative_column_difference,
        maximum_relative_frobenius_difference,
    )?;
    Ok(InteractionMatrixStabilityPlan { criteria, ..*plan })
}

macro_rules! impl_interaction_matrix_stability_properties {
    ($scalar:ty, $value:ident) => {
        crate::algorithm_properties! {
            impl InteractionMatrixStabilityPlan<$scalar> {
                plan_error = InteractionMatrixStabilityCriteriaError;
                replace = replace_interaction_matrix_stability_properties;
                properties {
                    MAXIMUM_RELATIVE_COLUMN_DIFFERENCE_PROPERTY =>
                        maximum_relative_column_difference
                        from criteria.maximum_relative_column_difference: $value {
                        name = "maximum-relative-column-difference";
                        description = "Largest accepted symmetric relative column difference";
                        unit = "1";
                        update = Runtime;
                        default = 0.5;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(0.0),
                            maximum: PropertyValue::$value(2.0),
                        };
                    }
                    MAXIMUM_RELATIVE_FROBENIUS_DIFFERENCE_PROPERTY =>
                        maximum_relative_frobenius_difference
                        from criteria.maximum_relative_frobenius_difference: $value {
                        name = "maximum-relative-frobenius-difference";
                        description = "Largest accepted symmetric relative Frobenius difference";
                        unit = "1";
                        update = Runtime;
                        default = 0.5;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(0.0),
                            maximum: PropertyValue::$value(2.0),
                        };
                    }
                }
            }
        }
    };
}

impl_interaction_matrix_stability_properties!(f32, Float);
impl_interaction_matrix_stability_properties!(f64, Double);

impl<T> AlgorithmPlan for InteractionMatrixStabilityPlan<T>
where
    T: MatrixElement + 'static,
{
    type Input<'a>
        = InteractionMatrixStabilityInput<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = &'a mut InteractionMatrixStabilityDiagnostics<T>
    where
        Self: 'a;
    type Workspace = NoWorkspace;
    type Error = InteractionMatrixStabilityError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        validate_extent(
            InteractionMatrixStabilityInputKind::FirstWindow,
            self.matrix_count,
            input.first_window.len(),
        )?;
        validate_extent(
            InteractionMatrixStabilityInputKind::SecondWindow,
            self.matrix_count,
            input.second_window.len(),
        )?;
        validate_finite(
            InteractionMatrixStabilityInputKind::FirstWindow,
            input.first_window,
        )?;
        validate_finite(
            InteractionMatrixStabilityInputKind::SecondWindow,
            input.second_window,
        )?;

        let mut minimum_first_norm = T::infinity();
        let mut minimum_second_norm = T::infinity();
        let mut maximum_relative_column_difference = T::zero();
        let mut maximum_relative_column_difference_coordinate = 0;
        let mut first_insufficient_coordinate = None;
        let mut total_first_squares = T::zero();
        let mut total_second_squares = T::zero();
        let mut total_difference_squares = T::zero();
        let two = T::one() + T::one();

        for corrector_coordinate in 0..self.corrector_coordinate_count {
            let mut first_squares = T::zero();
            let mut second_squares = T::zero();
            let mut difference_squares = T::zero();
            for measurement in 0..self.measurement_count {
                let index = measurement * self.corrector_coordinate_count + corrector_coordinate;
                let first = input.first_window[index];
                let second = input.second_window[index];
                let difference = first - second;
                if !difference.is_finite() {
                    return Err(InteractionMatrixStabilityError::NonFiniteCalculation);
                }
                first_squares = first_squares + first * first;
                second_squares = second_squares + second * second;
                difference_squares = difference_squares + difference * difference;
            }
            if !first_squares.is_finite()
                || !second_squares.is_finite()
                || !difference_squares.is_finite()
            {
                return Err(InteractionMatrixStabilityError::NonFiniteCalculation);
            }
            total_first_squares = total_first_squares + first_squares;
            total_second_squares = total_second_squares + second_squares;
            total_difference_squares = total_difference_squares + difference_squares;

            let first_norm = first_squares.sqrt();
            let second_norm = second_squares.sqrt();
            let difference_norm = difference_squares.sqrt();
            minimum_first_norm = minimum_first_norm.min(first_norm);
            minimum_second_norm = minimum_second_norm.min(second_norm);
            if first_norm < self.criteria.minimum_column_norm
                || second_norm < self.criteria.minimum_column_norm
            {
                first_insufficient_coordinate.get_or_insert(corrector_coordinate);
                continue;
            }

            let relative_difference = two * difference_norm / (first_norm + second_norm);
            if !relative_difference.is_finite() {
                return Err(InteractionMatrixStabilityError::NonFiniteCalculation);
            }
            if relative_difference > maximum_relative_column_difference {
                maximum_relative_column_difference = relative_difference;
                maximum_relative_column_difference_coordinate = corrector_coordinate;
            }
        }

        if !total_first_squares.is_finite()
            || !total_second_squares.is_finite()
            || !total_difference_squares.is_finite()
        {
            return Err(InteractionMatrixStabilityError::NonFiniteCalculation);
        }
        let total_first_norm = total_first_squares.sqrt();
        let total_second_norm = total_second_squares.sqrt();
        let relative_frobenius_difference = if total_first_norm + total_second_norm > T::zero() {
            two * total_difference_squares.sqrt() / (total_first_norm + total_second_norm)
        } else {
            T::zero()
        };
        if !relative_frobenius_difference.is_finite() {
            return Err(InteractionMatrixStabilityError::NonFiniteCalculation);
        }

        let status = if let Some(corrector_coordinate) = first_insufficient_coordinate {
            InteractionMatrixStabilityStatus::InsufficientColumnNorm {
                corrector_coordinate,
            }
        } else if maximum_relative_column_difference
            > self.criteria.maximum_relative_column_difference
        {
            InteractionMatrixStabilityStatus::ExcessiveRelativeColumnDifference {
                corrector_coordinate: maximum_relative_column_difference_coordinate,
            }
        } else if relative_frobenius_difference
            > self.criteria.maximum_relative_frobenius_difference
        {
            InteractionMatrixStabilityStatus::ExcessiveRelativeFrobeniusDifference
        } else {
            InteractionMatrixStabilityStatus::Valid
        };

        *output = InteractionMatrixStabilityDiagnostics {
            status,
            minimum_first_window_column_norm: minimum_first_norm,
            minimum_second_window_column_norm: minimum_second_norm,
            maximum_relative_column_difference,
            maximum_relative_column_difference_coordinate,
            relative_frobenius_difference,
        };
        Ok(())
    }
}

fn validate_extent(
    input: InteractionMatrixStabilityInputKind,
    expected: usize,
    actual: usize,
) -> Result<(), InteractionMatrixStabilityError> {
    if actual != expected {
        return Err(InteractionMatrixStabilityError::InputExtent {
            input,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_finite<T>(
    input: InteractionMatrixStabilityInputKind,
    values: &[T],
) -> Result<(), InteractionMatrixStabilityError>
where
    T: MatrixElement,
{
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(InteractionMatrixStabilityError::NonFiniteInput { input, index });
    }
    Ok(())
}

crate::filtergraph_algorithm! {
    /// Portable Float32 independent-window interaction-matrix stability declaration.
    pub InteractionMatrixStabilityF32 {
        label = "interaction-matrix-stability-f32";
        configuration InteractionMatrixStabilityF32Configuration {
            measurement_count: usize = GraphRebuild;
            corrector_coordinate_count: usize = GraphRebuild;
            minimum_column_norm: f32 = Construction;
            interaction_matrix_schema: String = GraphRebuild;
        }
        plan = InteractionMatrixStabilityPlan<f32>;
        workspace = NoWorkspace;
        properties = runtime;
        dimensions |plan, _workspace| {
            measurement_count = plan.measurement_count();
            corrector_coordinate_count = plan.corrector_coordinate_count();
        }
        inputs {
            first_window: f32 [measurement_count, corrector_coordinate_count]
                = "first-window",
                schema = configuration.interaction_matrix_schema.as_str();
            second_window: f32 [measurement_count, corrector_coordinate_count]
                = "second-window",
                schema = configuration.interaction_matrix_schema.as_str();
        }
        outputs {
            status: u8 [1] = "status",
                schema = crate::schemas::INTERACTION_MATRIX_STABILITY_STATUS_V1,
                metadata_from = "second-window";
            status_coordinate: u64 [1] = "status-coordinate",
                schema = crate::schemas::CORRECTOR_COORDINATE_V1,
                metadata_from = "second-window";
            minimum_column_norms: f32 [2] = "minimum-column-norms",
                schema = crate::schemas::INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1,
                metadata_from = "second-window";
            maximum_relative_column_difference: f32 [1]
                = "maximum-relative-column-difference",
                schema = crate::schemas::INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1,
                metadata_from = "second-window";
            maximum_relative_column_difference_coordinate: u64 [1]
                = "maximum-relative-column-difference-coordinate",
                schema = crate::schemas::CORRECTOR_COORDINATE_V1,
                metadata_from = "second-window";
            relative_frobenius_difference: f32 [1]
                = "relative-frobenius-difference",
                schema = crate::schemas::INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1,
                metadata_from = "second-window";
        }
        prepare |configuration| {
            let criteria = InteractionMatrixStabilityCriteria::new(
                configuration.minimum_column_norm,
                0.5,
                0.5,
            )?;
            Ok((
                InteractionMatrixStabilityPlan::new(
                    configuration.measurement_count,
                    configuration.corrector_coordinate_count,
                    criteria,
                )?,
                NoWorkspace,
            ))
        }
        prepare_error = InteractionMatrixStabilityF32PreparationError;
        process |plan, workspace| {
            let mut diagnostics = InteractionMatrixStabilityDiagnostics::default();
            plan.process(
                &mut diagnostics,
                workspace,
                InteractionMatrixStabilityInput::new(first_window, second_window),
            )?;
            let (status_value, status_coordinate_value) = match diagnostics.status() {
                InteractionMatrixStabilityStatus::Valid => (0, 0),
                InteractionMatrixStabilityStatus::InsufficientColumnNorm {
                    corrector_coordinate,
                } => (1, corrector_coordinate as u64),
                InteractionMatrixStabilityStatus::ExcessiveRelativeColumnDifference {
                    corrector_coordinate,
                } => (2, corrector_coordinate as u64),
                InteractionMatrixStabilityStatus::ExcessiveRelativeFrobeniusDifference => (3, 0),
            };
            status[0] = status_value;
            status_coordinate[0] = status_coordinate_value;
            minimum_column_norms[0] = diagnostics.minimum_first_window_column_norm();
            minimum_column_norms[1] = diagnostics.minimum_second_window_column_norm();
            maximum_relative_column_difference[0] =
                diagnostics.maximum_relative_column_difference();
            maximum_relative_column_difference_coordinate[0] =
                diagnostics.maximum_relative_column_difference_coordinate() as u64;
            relative_frobenius_difference[0] = diagnostics.relative_frobenius_difference();
            Ok(())
        }
        process_error = InteractionMatrixStabilityError;
    }
}
