//! Centered measurement--corrector covariance accumulation for DO-CRIME.

use alloc::{boxed::Box, string::String, vec};
use core::{fmt, marker::PhantomData, mem};

use num_traits::{NumCast, ToPrimitive};

use crate::{AlgorithmPlan, MatrixElement};

/// Covariance matrix named by a DO-CRIME preparation error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeCovarianceMatrix {
    /// Measurement--corrector cross-covariance matrix.
    MeasurementCorrectorCross,
    /// Corrector covariance matrix.
    Corrector,
}

/// Invalid prepared DO-CRIME covariance accumulation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeCovariancePlanError {
    /// Measurement and corrector-sample extents must both be nonzero.
    EmptyDimension,
    /// One covariance matrix extent overflowed addressable storage.
    ExtentOverflow {
        /// Matrix whose required extent overflowed.
        matrix: DocrimeCovarianceMatrix,
    },
    /// A selected finite covariance window must contain at least one sample.
    ZeroSampleLimit,
}

impl fmt::Display for DocrimeCovariancePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension => formatter
                .write_str("DO-CRIME measurement and corrector-sample dimensions must be nonzero"),
            Self::ExtentOverflow { matrix } => {
                write!(formatter, "DO-CRIME {matrix:?} covariance extent overflow")
            }
            Self::ZeroSampleLimit => {
                formatter.write_str("DO-CRIME covariance sample limit must be nonzero")
            }
        }
    }
}

impl core::error::Error for DocrimeCovariancePlanError {}

/// Input vector named by a DO-CRIME covariance error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeCovarianceInputKind {
    /// Sensor measurement vector.
    Measurements,
    /// Exposure-associated corrector sample.
    CorrectorSample,
}

/// Invalid DO-CRIME covariance input, workspace, count, or calculation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeCovarianceError {
    /// One input vector has the wrong extent.
    InputExtent {
        /// Input with the wrong extent.
        input: DocrimeCovarianceInputKind,
        /// Required extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// The supplied workspace was prepared for different extents.
    WorkspaceExtent,
    /// One input coefficient is not finite.
    NonFiniteInput {
        /// Input containing the coefficient.
        input: DocrimeCovarianceInputKind,
        /// Zero-based coefficient index.
        index: usize,
    },
    /// The committed sample count cannot be incremented.
    SampleCountOverflow,
    /// The next sample count is not exactly representable by the scalar type.
    SampleCountNotRepresentable,
    /// A computed mean or covariance coefficient is not finite.
    NonFiniteCalculation,
    /// The selected finite covariance window already contains its maximum samples.
    SampleLimitReached {
        /// Maximum number of committed samples in this plan.
        sample_limit: usize,
    },
}

impl fmt::Display for DocrimeCovarianceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent {
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "DO-CRIME {input:?} input extent {actual} does not match {expected}"
            ),
            Self::WorkspaceExtent => {
                formatter.write_str("DO-CRIME covariance workspace extent is invalid")
            }
            Self::NonFiniteInput { input, index } => write!(
                formatter,
                "DO-CRIME {input:?} input at index {index} must be finite"
            ),
            Self::SampleCountOverflow => {
                formatter.write_str("DO-CRIME covariance sample count overflow")
            }
            Self::SampleCountNotRepresentable => formatter.write_str(
                "DO-CRIME covariance sample count is not exactly representable by the scalar type",
            ),
            Self::NonFiniteCalculation => {
                formatter.write_str("DO-CRIME covariance calculation produced a non-finite value")
            }
            Self::SampleLimitReached { sample_limit } => write!(
                formatter,
                "DO-CRIME covariance sample limit {sample_limit} has been reached"
            ),
        }
    }
}

impl core::error::Error for DocrimeCovarianceError {}

/// One exposure-associated measurement and corrector sample.
#[derive(Clone, Copy, Debug)]
pub struct DocrimeCovarianceInput<'a, T> {
    measurements: &'a [T],
    corrector_sample: &'a [T],
}

impl<'a, T> DocrimeCovarianceInput<'a, T> {
    /// Groups one measurement with its exposure-associated corrector sample.
    pub const fn new(measurements: &'a [T], corrector_sample: &'a [T]) -> Self {
        Self {
            measurements,
            corrector_sample,
        }
    }

    /// Returns the sensor measurement vector.
    pub const fn measurements(&self) -> &[T] {
        self.measurements
    }

    /// Returns the exposure-associated corrector sample.
    pub const fn corrector_sample(&self) -> &[T] {
        self.corrector_sample
    }
}

/// Frozen centered covariance sums consumed by a DO-CRIME solve.
#[derive(Clone, Copy, Debug)]
pub struct DocrimeCovariances<'a, T> {
    sample_count: usize,
    measurement_corrector_cross_covariance_sum: &'a [T],
    corrector_covariance_sum: &'a [T],
}

impl<'a, T> DocrimeCovariances<'a, T> {
    /// Groups a sample count and row-major centered covariance sums.
    pub const fn new(
        sample_count: usize,
        measurement_corrector_cross_covariance_sum: &'a [T],
        corrector_covariance_sum: &'a [T],
    ) -> Self {
        Self {
            sample_count,
            measurement_corrector_cross_covariance_sum,
            corrector_covariance_sum,
        }
    }

    /// Returns the number of samples accumulated into both sums.
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the row-major measurement--corrector cross-covariance sum.
    pub const fn measurement_corrector_cross_covariance_sum(&self) -> &[T] {
        self.measurement_corrector_cross_covariance_sum
    }

    /// Returns the symmetric row-major corrector covariance sum.
    pub const fn corrector_covariance_sum(&self) -> &[T] {
        self.corrector_covariance_sum
    }
}

/// Prepared centered covariance accumulation for DO-CRIME.
///
/// For open-loop identification the corrector sample is the realized random
/// excitation. For closed-loop identification it is the complete corrector
/// command increment. The caller establishes exposure association and any
/// matched temporal filtering before this operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DocrimeCovariancePlan<T> {
    measurement_count: usize,
    corrector_sample_count: usize,
    cross_covariance_count: usize,
    corrector_covariance_count: usize,
    sample_limit: Option<usize>,
    scalar: PhantomData<T>,
}

impl<T> DocrimeCovariancePlan<T> {
    /// Validates the measurement and corrector-sample matrix extents.
    pub fn new(
        measurement_count: usize,
        corrector_sample_count: usize,
    ) -> Result<Self, DocrimeCovariancePlanError> {
        Self::prepare(measurement_count, corrector_sample_count, None)
    }

    /// Prepares covariance accumulation with an exact nonzero sample limit.
    pub fn with_sample_limit(
        measurement_count: usize,
        corrector_sample_count: usize,
        sample_limit: usize,
    ) -> Result<Self, DocrimeCovariancePlanError> {
        if sample_limit == 0 {
            return Err(DocrimeCovariancePlanError::ZeroSampleLimit);
        }
        Self::prepare(
            measurement_count,
            corrector_sample_count,
            Some(sample_limit),
        )
    }

    fn prepare(
        measurement_count: usize,
        corrector_sample_count: usize,
        sample_limit: Option<usize>,
    ) -> Result<Self, DocrimeCovariancePlanError> {
        if measurement_count == 0 || corrector_sample_count == 0 {
            return Err(DocrimeCovariancePlanError::EmptyDimension);
        }
        let cross_covariance_count = measurement_count
            .checked_mul(corrector_sample_count)
            .ok_or(DocrimeCovariancePlanError::ExtentOverflow {
                matrix: DocrimeCovarianceMatrix::MeasurementCorrectorCross,
            })?;
        let corrector_covariance_count = corrector_sample_count
            .checked_mul(corrector_sample_count)
            .ok_or(DocrimeCovariancePlanError::ExtentOverflow {
                matrix: DocrimeCovarianceMatrix::Corrector,
            })?;
        Ok(Self {
            measurement_count,
            corrector_sample_count,
            cross_covariance_count,
            corrector_covariance_count,
            sample_limit,
            scalar: PhantomData,
        })
    }

    /// Returns the measurement-vector extent.
    pub const fn measurement_count(&self) -> usize {
        self.measurement_count
    }

    /// Returns the corrector-sample extent.
    pub const fn corrector_sample_count(&self) -> usize {
        self.corrector_sample_count
    }

    /// Returns the selected maximum sample count, if the plan is finite-windowed.
    pub const fn sample_limit(&self) -> Option<usize> {
        self.sample_limit
    }

    /// Allocates zero-sample history and next-state storage for one owner.
    pub fn workspace(&self) -> DocrimeCovarianceWorkspace<T>
    where
        T: MatrixElement,
    {
        DocrimeCovarianceWorkspace {
            sample_count: 0,
            measurement_mean: vec![T::zero(); self.measurement_count].into_boxed_slice(),
            next_measurement_mean: vec![T::zero(); self.measurement_count].into_boxed_slice(),
            corrector_sample_mean: vec![T::zero(); self.corrector_sample_count].into_boxed_slice(),
            next_corrector_sample_mean: vec![T::zero(); self.corrector_sample_count]
                .into_boxed_slice(),
            measurement_corrector_cross_covariance_sum: vec![
                T::zero();
                self.cross_covariance_count
            ]
            .into_boxed_slice(),
            next_measurement_corrector_cross_covariance_sum: vec![
                T::zero();
                self.cross_covariance_count
            ]
            .into_boxed_slice(),
            corrector_covariance_sum: vec![T::zero(); self.corrector_covariance_count]
                .into_boxed_slice(),
            next_corrector_covariance_sum: vec![T::zero(); self.corrector_covariance_count]
                .into_boxed_slice(),
            measurement_delta: vec![T::zero(); self.measurement_count].into_boxed_slice(),
            corrector_sample_delta: vec![T::zero(); self.corrector_sample_count].into_boxed_slice(),
        }
    }

    fn validate_workspace(
        &self,
        workspace: &DocrimeCovarianceWorkspace<T>,
    ) -> Result<(), DocrimeCovarianceError> {
        if workspace.measurement_mean.len() != self.measurement_count
            || workspace.next_measurement_mean.len() != self.measurement_count
            || workspace.corrector_sample_mean.len() != self.corrector_sample_count
            || workspace.next_corrector_sample_mean.len() != self.corrector_sample_count
            || workspace.measurement_corrector_cross_covariance_sum.len()
                != self.cross_covariance_count
            || workspace
                .next_measurement_corrector_cross_covariance_sum
                .len()
                != self.cross_covariance_count
            || workspace.corrector_covariance_sum.len() != self.corrector_covariance_count
            || workspace.next_corrector_covariance_sum.len() != self.corrector_covariance_count
            || workspace.measurement_delta.len() != self.measurement_count
            || workspace.corrector_sample_delta.len() != self.corrector_sample_count
        {
            return Err(DocrimeCovarianceError::WorkspaceExtent);
        }
        Ok(())
    }
}

impl<T> AlgorithmPlan for DocrimeCovariancePlan<T>
where
    T: MatrixElement + NumCast + ToPrimitive + 'static,
{
    type Input<'a>
        = DocrimeCovarianceInput<'a, T>
    where
        Self: 'a;
    type Output<'a> = ();
    type Workspace = DocrimeCovarianceWorkspace<T>;
    type Error = DocrimeCovarianceError;

    fn process<'input, 'output>(
        &self,
        (): Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        validate_input_extent(
            DocrimeCovarianceInputKind::Measurements,
            self.measurement_count,
            input.measurements.len(),
        )?;
        validate_input_extent(
            DocrimeCovarianceInputKind::CorrectorSample,
            self.corrector_sample_count,
            input.corrector_sample.len(),
        )?;
        self.validate_workspace(workspace)?;
        if let Some(sample_limit) = self.sample_limit
            && workspace.sample_count >= sample_limit
        {
            return Err(DocrimeCovarianceError::SampleLimitReached { sample_limit });
        }
        validate_finite(DocrimeCovarianceInputKind::Measurements, input.measurements)?;
        validate_finite(
            DocrimeCovarianceInputKind::CorrectorSample,
            input.corrector_sample,
        )?;

        let next_sample_count = workspace
            .sample_count
            .checked_add(1)
            .ok_or(DocrimeCovarianceError::SampleCountOverflow)?;
        let count = <T as NumCast>::from(next_sample_count)
            .filter(|value| value.to_usize() == Some(next_sample_count))
            .ok_or(DocrimeCovarianceError::SampleCountNotRepresentable)?;
        let weight = (count - T::one()) / count;

        let mut finite_measurement = true;
        for (((delta, next_mean), mean), sample) in workspace
            .measurement_delta
            .iter_mut()
            .zip(workspace.next_measurement_mean.iter_mut())
            .zip(workspace.measurement_mean.iter())
            .zip(input.measurements)
        {
            *delta = *sample - *mean;
            *next_mean = *mean + *delta / count;
            finite_measurement &= delta.is_finite() & next_mean.is_finite();
        }
        if !finite_measurement {
            return Err(DocrimeCovarianceError::NonFiniteCalculation);
        }
        let mut finite_corrector = true;
        for (((delta, next_mean), mean), sample) in workspace
            .corrector_sample_delta
            .iter_mut()
            .zip(workspace.next_corrector_sample_mean.iter_mut())
            .zip(workspace.corrector_sample_mean.iter())
            .zip(input.corrector_sample)
        {
            *delta = *sample - *mean;
            *next_mean = *mean + *delta / count;
            finite_corrector &= delta.is_finite() & next_mean.is_finite();
        }
        if !finite_corrector {
            return Err(DocrimeCovarianceError::NonFiniteCalculation);
        }

        let mut finite_cross = true;
        for ((next_row, row), delta) in workspace
            .next_measurement_corrector_cross_covariance_sum
            .chunks_exact_mut(self.corrector_sample_count)
            .zip(
                workspace
                    .measurement_corrector_cross_covariance_sum
                    .chunks_exact(self.corrector_sample_count),
            )
            .zip(workspace.measurement_delta.iter())
        {
            let weighted_delta = weight * *delta;
            let mut finite_row = true;
            for ((next, previous), corrector_delta) in next_row
                .iter_mut()
                .zip(row)
                .zip(workspace.corrector_sample_delta.iter())
            {
                let candidate = weighted_delta.mul_add(*corrector_delta, *previous);
                finite_row &= candidate.is_finite();
                *next = candidate;
            }
            finite_cross &= finite_row;
        }
        if !finite_cross {
            return Err(DocrimeCovarianceError::NonFiniteCalculation);
        }

        let width = self.corrector_sample_count;
        // Calculate contiguous upper rows before mirroring. A lower destination
        // never overwrites an upper source, and each copied coefficient retains
        // the original weight * delta[row] multiplication and FMA rounding.
        for row_start in (0..width).step_by(16) {
            let row_end = row_start.saturating_add(16).min(width);
            for row in row_start..row_end {
                let start = row * width + row;
                let end = (row + 1) * width;
                let weighted_delta = weight * workspace.corrector_sample_delta[row];
                let mut finite_row = true;
                for ((next, previous), delta) in workspace.next_corrector_covariance_sum[start..end]
                    .iter_mut()
                    .zip(&workspace.corrector_covariance_sum[start..end])
                    .zip(&workspace.corrector_sample_delta[row..])
                {
                    let candidate = weighted_delta.mul_add(*delta, *previous);
                    finite_row &= candidate.is_finite();
                    *next = candidate;
                }
                if !finite_row {
                    return Err(DocrimeCovarianceError::NonFiniteCalculation);
                }
            }
            for column_start in (row_start..width).step_by(16) {
                let column_end = column_start.saturating_add(16).min(width);
                for column in column_start..column_end {
                    for row in row_start..row_end.min(column) {
                        workspace.next_corrector_covariance_sum[column * width + row] =
                            workspace.next_corrector_covariance_sum[row * width + column];
                    }
                }
            }
        }

        mem::swap(
            &mut workspace.measurement_mean,
            &mut workspace.next_measurement_mean,
        );
        mem::swap(
            &mut workspace.corrector_sample_mean,
            &mut workspace.next_corrector_sample_mean,
        );
        mem::swap(
            &mut workspace.measurement_corrector_cross_covariance_sum,
            &mut workspace.next_measurement_corrector_cross_covariance_sum,
        );
        mem::swap(
            &mut workspace.corrector_covariance_sum,
            &mut workspace.next_corrector_covariance_sum,
        );
        workspace.sample_count = next_sample_count;
        Ok(())
    }
}

/// Committed centered DO-CRIME covariance history and next-state storage.
#[derive(Clone, Debug, PartialEq)]
pub struct DocrimeCovarianceWorkspace<T> {
    sample_count: usize,
    measurement_mean: Box<[T]>,
    next_measurement_mean: Box<[T]>,
    corrector_sample_mean: Box<[T]>,
    next_corrector_sample_mean: Box<[T]>,
    measurement_corrector_cross_covariance_sum: Box<[T]>,
    next_measurement_corrector_cross_covariance_sum: Box<[T]>,
    corrector_covariance_sum: Box<[T]>,
    next_corrector_covariance_sum: Box<[T]>,
    measurement_delta: Box<[T]>,
    corrector_sample_delta: Box<[T]>,
}

impl<T> DocrimeCovarianceWorkspace<T> {
    /// Returns the committed sample count.
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the committed measurement mean.
    pub fn measurement_mean(&self) -> &[T] {
        &self.measurement_mean
    }

    /// Returns the committed corrector-sample mean.
    pub fn corrector_sample_mean(&self) -> &[T] {
        &self.corrector_sample_mean
    }

    /// Returns the row-major centered measurement--corrector covariance sum.
    pub fn measurement_corrector_cross_covariance_sum(&self) -> &[T] {
        &self.measurement_corrector_cross_covariance_sum
    }

    /// Returns the symmetric row-major centered corrector covariance sum.
    pub fn corrector_covariance_sum(&self) -> &[T] {
        &self.corrector_covariance_sum
    }

    /// Borrows the committed covariance sums as one frozen solve input.
    pub fn covariances(&self) -> DocrimeCovariances<'_, T> {
        DocrimeCovariances::new(
            self.sample_count,
            &self.measurement_corrector_cross_covariance_sum,
            &self.corrector_covariance_sum,
        )
    }

    /// Clears all committed and staged history without replacing prepared storage.
    pub fn reset(&mut self)
    where
        T: MatrixElement,
    {
        self.sample_count = 0;
        self.measurement_mean.fill(T::zero());
        self.next_measurement_mean.fill(T::zero());
        self.corrector_sample_mean.fill(T::zero());
        self.next_corrector_sample_mean.fill(T::zero());
        self.measurement_corrector_cross_covariance_sum
            .fill(T::zero());
        self.next_measurement_corrector_cross_covariance_sum
            .fill(T::zero());
        self.corrector_covariance_sum.fill(T::zero());
        self.next_corrector_covariance_sum.fill(T::zero());
        self.measurement_delta.fill(T::zero());
        self.corrector_sample_delta.fill(T::zero());
    }
}

fn validate_input_extent(
    input: DocrimeCovarianceInputKind,
    expected: usize,
    actual: usize,
) -> Result<(), DocrimeCovarianceError> {
    if actual != expected {
        return Err(DocrimeCovarianceError::InputExtent {
            input,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_finite<T>(
    input: DocrimeCovarianceInputKind,
    values: &[T],
) -> Result<(), DocrimeCovarianceError>
where
    T: MatrixElement,
{
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(DocrimeCovarianceError::NonFiniteInput { input, index });
    }
    Ok(())
}

crate::filtergraph_algorithm! {
    /// Portable Float32 centered DO-CRIME covariance accumulator declaration.
    pub DocrimeCovarianceF32 {
        label = "docrime-covariance-f32";
        configuration DocrimeCovarianceF32Configuration {
            measurement_count: usize = GraphRebuild;
            corrector_coordinate_count: usize = GraphRebuild;
            sample_limit: Option<usize> = Construction;
            measurements_schema: String = GraphRebuild;
            corrector_sample_schema: String = GraphRebuild;
            cross_covariance_schema: String = GraphRebuild;
            corrector_covariance_schema: String = GraphRebuild;
        }
        plan = DocrimeCovariancePlan<f32>;
        workspace = DocrimeCovarianceWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            measurement_count = plan.measurement_count();
            corrector_sample_count = plan.corrector_sample_count();
        }
        inputs {
            measurements: f32 [measurement_count] = "measurements",
                schema = configuration.measurements_schema.as_str();
            corrector_sample: f32 [corrector_sample_count] = "corrector-sample",
                schema = configuration.corrector_sample_schema.as_str();
        }
        outputs {
            measurement_corrector_cross_covariance_sum: f32
                [measurement_count, corrector_sample_count]
                = "measurement-corrector-cross-covariance-sum",
                schema = configuration.cross_covariance_schema.as_str(),
                metadata_from = "measurements";
            corrector_covariance_sum: f32 [corrector_sample_count, corrector_sample_count]
                = "corrector-covariance-sum",
                schema = configuration.corrector_covariance_schema.as_str(),
                metadata_from = "measurements";
            sample_count: u64 [1] = "sample-count",
                schema = crate::schemas::DOCRIME_COVARIANCE_SAMPLE_COUNT_V1,
                metadata_from = "measurements";
            complete: bool [1] = "complete",
                schema = crate::schemas::DOCRIME_COVARIANCE_COMPLETE_V1,
                metadata_from = "measurements";
        }
        prepare |configuration| {
            let plan = match configuration.sample_limit {
                Some(sample_limit) => DocrimeCovariancePlan::with_sample_limit(
                    configuration.measurement_count,
                    configuration.corrector_coordinate_count,
                    sample_limit,
                )?,
                None => DocrimeCovariancePlan::new(
                    configuration.measurement_count,
                    configuration.corrector_coordinate_count,
                )?,
            };
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = DocrimeCovariancePlanError;
        process |plan, workspace| {
            plan.process(
                (),
                workspace,
                DocrimeCovarianceInput::new(measurements, corrector_sample),
            )?;
            measurement_corrector_cross_covariance_sum.copy_from_slice(
                workspace.measurement_corrector_cross_covariance_sum(),
            );
            corrector_covariance_sum.copy_from_slice(workspace.corrector_covariance_sum());
            sample_count[0] = workspace.sample_count() as u64;
            complete[0] = plan
                .sample_limit()
                .is_some_and(|limit| workspace.sample_count() == limit);
            Ok(())
        }
        process_error = DocrimeCovarianceError;
        reset |_plan, workspace| {
            workspace.reset();
        }
    }
}
