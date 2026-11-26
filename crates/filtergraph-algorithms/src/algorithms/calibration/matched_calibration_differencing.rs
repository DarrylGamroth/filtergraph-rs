//! Matched first-difference filtering for empirical calibration samples.

use alloc::{boxed::Box, string::String, vec};
use core::{fmt, marker::PhantomData};

use crate::{AlgorithmPlan, MatrixElement};

/// Invalid prepared matched calibration differencing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchedCalibrationDifferencingPlanError {
    /// Measurement and corrector-sample extents must both be nonzero.
    EmptyDimension,
}

impl fmt::Display for MatchedCalibrationDifferencingPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension => formatter.write_str(
                "matched calibration measurement and corrector-sample dimensions must be nonzero",
            ),
        }
    }
}

impl core::error::Error for MatchedCalibrationDifferencingPlanError {}

/// Input vector named by a matched calibration differencing error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchedCalibrationDifferencingInputKind {
    /// Sensor measurement vector.
    Measurements,
    /// Exposure-associated corrector sample.
    CorrectorSample,
}

/// Output vector named by a matched calibration differencing error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchedCalibrationDifferencingOutputKind {
    /// First-differenced sensor measurements.
    FilteredMeasurements,
    /// First-differenced corrector sample.
    FilteredCorrectorSample,
}

/// Invalid matched calibration differencing input, output, or workspace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchedCalibrationDifferencingError {
    /// One input vector has the wrong extent.
    InputExtent {
        /// Input with the wrong extent.
        input: MatchedCalibrationDifferencingInputKind,
        /// Required extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// One output vector has the wrong extent.
    OutputExtent {
        /// Output with the wrong extent.
        output: MatchedCalibrationDifferencingOutputKind,
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
        input: MatchedCalibrationDifferencingInputKind,
        /// Zero-based coefficient index.
        index: usize,
    },
    /// One staged first difference is not finite.
    NonFiniteCalculation,
}

impl fmt::Display for MatchedCalibrationDifferencingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent {
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "matched calibration {input:?} input extent {actual} does not match {expected}"
            ),
            Self::OutputExtent {
                output,
                expected,
                actual,
            } => write!(
                formatter,
                "matched calibration {output:?} output extent {actual} does not match {expected}"
            ),
            Self::WorkspaceExtent => {
                formatter.write_str("matched calibration differencing workspace extent is invalid")
            }
            Self::NonFiniteInput { input, index } => write!(
                formatter,
                "matched calibration {input:?} input at index {index} must be finite"
            ),
            Self::NonFiniteCalculation => formatter
                .write_str("matched calibration first-difference calculation is not finite"),
        }
    }
}

impl core::error::Error for MatchedCalibrationDifferencingError {}

/// Publication state of one matched calibration differencing result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MatchedCalibrationDifferencingStatus {
    /// The input pair primed history; the published vectors are zero and unusable.
    #[default]
    Priming,
    /// Both published vectors are valid first differences.
    Filtered,
}

/// One exposure-associated measurement and corrector sample.
#[derive(Clone, Copy, Debug)]
pub struct MatchedCalibrationDifferencingInput<'a, T> {
    measurements: &'a [T],
    corrector_sample: &'a [T],
}

impl<'a, T> MatchedCalibrationDifferencingInput<'a, T> {
    /// Groups one measurement with its already exposure-associated corrector sample.
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

/// Caller-owned coherent matched first differences and their publication state.
#[derive(Debug)]
pub struct MatchedCalibrationDifferencingOutput<'a, T> {
    filtered_measurements: &'a mut [T],
    filtered_corrector_sample: &'a mut [T],
    status: &'a mut MatchedCalibrationDifferencingStatus,
}

impl<'a, T> MatchedCalibrationDifferencingOutput<'a, T> {
    /// Groups both filtered vectors and their coherent status destination.
    pub fn new(
        filtered_measurements: &'a mut [T],
        filtered_corrector_sample: &'a mut [T],
        status: &'a mut MatchedCalibrationDifferencingStatus,
    ) -> Self {
        Self {
            filtered_measurements,
            filtered_corrector_sample,
            status,
        }
    }

    /// Returns the first-differenced measurements.
    pub fn filtered_measurements(&self) -> &[T] {
        self.filtered_measurements
    }

    /// Returns the first-differenced corrector sample.
    pub fn filtered_corrector_sample(&self) -> &[T] {
        self.filtered_corrector_sample
    }

    /// Returns whether the vectors are priming zeros or valid differences.
    pub const fn status(&self) -> MatchedCalibrationDifferencingStatus {
        *self.status
    }
}

/// Prepared matched `[1, -1]` filter for measurements and corrector samples.
///
/// The caller associates the input pair with one sensor exposure before this
/// operation. This plan does not infer command delay or choose among samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatchedCalibrationDifferencingPlan<T> {
    measurement_count: usize,
    corrector_sample_count: usize,
    scalar: PhantomData<T>,
}

impl<T> MatchedCalibrationDifferencingPlan<T> {
    /// Validates the measurement and corrector-sample extents.
    pub fn new(
        measurement_count: usize,
        corrector_sample_count: usize,
    ) -> Result<Self, MatchedCalibrationDifferencingPlanError> {
        if measurement_count == 0 || corrector_sample_count == 0 {
            return Err(MatchedCalibrationDifferencingPlanError::EmptyDimension);
        }
        Ok(Self {
            measurement_count,
            corrector_sample_count,
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

    /// Allocates previous-pair history and fixed staged storage for one owner.
    pub fn workspace(&self) -> MatchedCalibrationDifferencingWorkspace<T>
    where
        T: MatrixElement,
    {
        MatchedCalibrationDifferencingWorkspace {
            primed: false,
            previous_measurements: vec![T::zero(); self.measurement_count].into_boxed_slice(),
            previous_corrector_sample: vec![T::zero(); self.corrector_sample_count]
                .into_boxed_slice(),
            staged_measurements: vec![T::zero(); self.measurement_count].into_boxed_slice(),
            staged_corrector_sample: vec![T::zero(); self.corrector_sample_count]
                .into_boxed_slice(),
        }
    }

    fn validate_workspace(
        &self,
        workspace: &MatchedCalibrationDifferencingWorkspace<T>,
    ) -> Result<(), MatchedCalibrationDifferencingError> {
        if workspace.previous_measurements.len() != self.measurement_count
            || workspace.previous_corrector_sample.len() != self.corrector_sample_count
            || workspace.staged_measurements.len() != self.measurement_count
            || workspace.staged_corrector_sample.len() != self.corrector_sample_count
        {
            return Err(MatchedCalibrationDifferencingError::WorkspaceExtent);
        }
        Ok(())
    }
}

impl<T> AlgorithmPlan for MatchedCalibrationDifferencingPlan<T>
where
    T: MatrixElement + 'static,
{
    type Input<'a>
        = MatchedCalibrationDifferencingInput<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = MatchedCalibrationDifferencingOutput<'a, T>
    where
        Self: 'a;
    type Workspace = MatchedCalibrationDifferencingWorkspace<T>;
    type Error = MatchedCalibrationDifferencingError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        validate_input_extent(
            MatchedCalibrationDifferencingInputKind::Measurements,
            self.measurement_count,
            input.measurements.len(),
        )?;
        validate_input_extent(
            MatchedCalibrationDifferencingInputKind::CorrectorSample,
            self.corrector_sample_count,
            input.corrector_sample.len(),
        )?;
        validate_output_extent(
            MatchedCalibrationDifferencingOutputKind::FilteredMeasurements,
            self.measurement_count,
            output.filtered_measurements.len(),
        )?;
        validate_output_extent(
            MatchedCalibrationDifferencingOutputKind::FilteredCorrectorSample,
            self.corrector_sample_count,
            output.filtered_corrector_sample.len(),
        )?;
        self.validate_workspace(workspace)?;
        validate_finite(
            MatchedCalibrationDifferencingInputKind::Measurements,
            input.measurements,
        )?;
        validate_finite(
            MatchedCalibrationDifferencingInputKind::CorrectorSample,
            input.corrector_sample,
        )?;

        let status = if workspace.primed {
            for ((staged, current), previous) in workspace
                .staged_measurements
                .iter_mut()
                .zip(input.measurements)
                .zip(&workspace.previous_measurements)
            {
                *staged = *current - *previous;
                if !staged.is_finite() {
                    return Err(MatchedCalibrationDifferencingError::NonFiniteCalculation);
                }
            }
            for ((staged, current), previous) in workspace
                .staged_corrector_sample
                .iter_mut()
                .zip(input.corrector_sample)
                .zip(&workspace.previous_corrector_sample)
            {
                *staged = *current - *previous;
                if !staged.is_finite() {
                    return Err(MatchedCalibrationDifferencingError::NonFiniteCalculation);
                }
            }
            MatchedCalibrationDifferencingStatus::Filtered
        } else {
            workspace.staged_measurements.fill(T::zero());
            workspace.staged_corrector_sample.fill(T::zero());
            MatchedCalibrationDifferencingStatus::Priming
        };

        output
            .filtered_measurements
            .copy_from_slice(&workspace.staged_measurements);
        output
            .filtered_corrector_sample
            .copy_from_slice(&workspace.staged_corrector_sample);
        *output.status = status;
        workspace
            .previous_measurements
            .copy_from_slice(input.measurements);
        workspace
            .previous_corrector_sample
            .copy_from_slice(input.corrector_sample);
        workspace.primed = true;
        Ok(())
    }
}

/// Committed previous input pair and scratch for matched calibration differencing.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchedCalibrationDifferencingWorkspace<T> {
    primed: bool,
    previous_measurements: Box<[T]>,
    previous_corrector_sample: Box<[T]>,
    staged_measurements: Box<[T]>,
    staged_corrector_sample: Box<[T]>,
}

impl<T> MatchedCalibrationDifferencingWorkspace<T> {
    /// Returns whether a previous input pair is committed.
    pub const fn is_primed(&self) -> bool {
        self.primed
    }

    /// Returns the committed previous measurement vector.
    pub fn previous_measurements(&self) -> &[T] {
        &self.previous_measurements
    }

    /// Returns the committed previous corrector sample.
    pub fn previous_corrector_sample(&self) -> &[T] {
        &self.previous_corrector_sample
    }
}

impl<T> MatchedCalibrationDifferencingWorkspace<T>
where
    T: MatrixElement,
{
    /// Clears committed history and all fixed scratch for a new sequence.
    pub fn reset(&mut self) {
        self.primed = false;
        self.previous_measurements.fill(T::zero());
        self.previous_corrector_sample.fill(T::zero());
        self.staged_measurements.fill(T::zero());
        self.staged_corrector_sample.fill(T::zero());
    }
}

fn validate_input_extent(
    input: MatchedCalibrationDifferencingInputKind,
    expected: usize,
    actual: usize,
) -> Result<(), MatchedCalibrationDifferencingError> {
    if actual != expected {
        return Err(MatchedCalibrationDifferencingError::InputExtent {
            input,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_output_extent(
    output: MatchedCalibrationDifferencingOutputKind,
    expected: usize,
    actual: usize,
) -> Result<(), MatchedCalibrationDifferencingError> {
    if actual != expected {
        return Err(MatchedCalibrationDifferencingError::OutputExtent {
            output,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_finite<T>(
    input: MatchedCalibrationDifferencingInputKind,
    values: &[T],
) -> Result<(), MatchedCalibrationDifferencingError>
where
    T: MatrixElement,
{
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(MatchedCalibrationDifferencingError::NonFiniteInput { input, index });
    }
    Ok(())
}

crate::filtergraph_algorithm! {
    /// Portable Float32 matched calibration first-difference declaration.
    pub MatchedCalibrationDifferencingF32 {
        label = "matched-calibration-differencing-f32";
        configuration MatchedCalibrationDifferencingF32Configuration {
            measurement_count: usize = GraphRebuild;
            corrector_coordinate_count: usize = GraphRebuild;
            measurements_schema: String = GraphRebuild;
            corrector_sample_schema: String = GraphRebuild;
            filtered_measurements_schema: String = GraphRebuild;
            filtered_corrector_sample_schema: String = GraphRebuild;
        }
        plan = MatchedCalibrationDifferencingPlan<f32>;
        workspace = MatchedCalibrationDifferencingWorkspace<f32>;
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
            filtered_measurements: f32 [measurement_count] = "filtered-measurements",
                schema = configuration.filtered_measurements_schema.as_str(),
                metadata_from = "measurements";
            filtered_corrector_sample: f32 [corrector_sample_count]
                = "filtered-corrector-sample",
                schema = configuration.filtered_corrector_sample_schema.as_str(),
                metadata_from = "measurements";
            filtered: bool [1] = "filtered",
                schema = crate::schemas::MATCHED_CALIBRATION_FILTERED_V1,
                metadata_from = "measurements";
        }
        prepare |configuration| {
            let plan = MatchedCalibrationDifferencingPlan::new(
                configuration.measurement_count,
                configuration.corrector_coordinate_count,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = MatchedCalibrationDifferencingPlanError;
        process |plan, workspace| {
            let mut status = MatchedCalibrationDifferencingStatus::Priming;
            plan.process(
                MatchedCalibrationDifferencingOutput::new(
                    filtered_measurements,
                    filtered_corrector_sample,
                    &mut status,
                ),
                workspace,
                MatchedCalibrationDifferencingInput::new(measurements, corrector_sample),
            )?;
            filtered[0] = status == MatchedCalibrationDifferencingStatus::Filtered;
            Ok(())
        }
        process_error = MatchedCalibrationDifferencingError;
        reset |_plan, workspace| {
            workspace.reset();
        }
    }
}
