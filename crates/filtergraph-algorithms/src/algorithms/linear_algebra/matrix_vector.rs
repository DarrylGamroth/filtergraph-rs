//! Prepared row-major GEMV for Rust callers.

use alloc::sync::Arc;
use core::fmt;

use crate::MatrixElement;

use super::gemv::multiply_row_major;

/// Invalid prepared GEMV operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedGemvError {
    /// Matrix rows and columns must both be nonzero.
    EmptyDimension,
    /// The matrix extent overflowed addressable storage.
    ExtentOverflow,
    /// The matrix does not contain `rows * columns` coefficients.
    MatrixExtent,
    /// A matrix coefficient is not finite.
    NonFiniteMatrix,
}

impl fmt::Display for PreparedGemvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension => formatter.write_str("matrix dimensions must be nonzero"),
            Self::ExtentOverflow => formatter.write_str("matrix extent overflow"),
            Self::MatrixExtent => formatter.write_str("matrix extent is invalid"),
            Self::NonFiniteMatrix => formatter.write_str("matrix coefficients must be finite"),
        }
    }
}

impl core::error::Error for PreparedGemvError {}

/// Invalid GEMV input, output, or result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemvError {
    /// The input vector length does not match the matrix columns.
    InputExtent,
    /// The output vector length does not match the matrix rows.
    OutputExtent,
    /// An input value is not finite.
    NonFiniteInput,
    /// The multiplication produced a non-finite result.
    NonFiniteOutput,
}

impl fmt::Display for GemvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => formatter.write_str("matrix-vector input extent is invalid"),
            Self::OutputExtent => formatter.write_str("matrix-vector output extent is invalid"),
            Self::NonFiniteInput => formatter.write_str("matrix-vector input must be finite"),
            Self::NonFiniteOutput => {
                formatter.write_str("matrix-vector multiplication produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for GemvError {}

/// Prepared direct operation `y = A*x` for a row-major matrix `A`.
///
/// Rust has no standard matrix-vector multiplication operator or `mul!`
/// equivalent. This value retains the validated matrix and selects the
/// precision-specific execution boundary without turning the primitive into an
/// [`crate::AlgorithmPlan`]. Scientific reconstruction and projection plans
/// call this primitive while retaining their own operation contracts.
#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "experimental-simd-gemv"), derive(PartialEq))]
pub struct PreparedGemv<T> {
    #[cfg(feature = "blas")]
    pupil_rows: Option<PupilRows>,
    #[cfg(feature = "experimental-simd-gemv")]
    pupil_simd_input_limit: Option<f32>,
    rows: usize,
    columns: usize,
    matrix: Arc<[T]>,
}

// Equality follows the released plan, irrespective of an experimental
// kernel being available when the pupil declaration was prepared.
#[cfg(feature = "experimental-simd-gemv")]
impl<T: PartialEq> PartialEq for PreparedGemv<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.columns == other.columns
            && self.pupil_rows == other.pupil_rows
            && self.matrix == other.matrix
    }
}

impl<T> PreparedGemv<T>
where
    T: MatrixElement,
{
    /// Validates and retains a prepared row-major matrix.
    pub fn new(rows: usize, columns: usize, matrix: Arc<[T]>) -> Result<Self, PreparedGemvError> {
        if rows == 0 || columns == 0 {
            return Err(PreparedGemvError::EmptyDimension);
        }
        let elements = rows
            .checked_mul(columns)
            .ok_or(PreparedGemvError::ExtentOverflow)?;
        if matrix.len() != elements {
            return Err(PreparedGemvError::MatrixExtent);
        }
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(PreparedGemvError::NonFiniteMatrix);
        }
        Ok(Self {
            #[cfg(feature = "blas")]
            pupil_rows: None,
            #[cfg(feature = "experimental-simd-gemv")]
            pupil_simd_input_limit: None,
            rows,
            columns,
            matrix,
        })
    }

    /// Returns the matrix row count and output-vector length.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the matrix column count and input-vector length.
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the prepared row-major matrix coefficients.
    pub fn matrix(&self) -> &[T] {
        &self.matrix
    }

    /// Multiplies the prepared matrix by `input` into caller-owned `output`.
    pub fn multiply(&self, output: &mut [T], input: &[T]) -> Result<(), GemvError> {
        self.multiply_with(output, input, |output, input| {
            multiply_row_major(&self.matrix, self.rows, self.columns, input, output);
        })
    }

    pub(super) fn multiply_with(
        &self,
        output: &mut [T],
        input: &[T],
        multiply: impl FnOnce(&mut [T], &[T]),
    ) -> Result<(), GemvError> {
        if input.len() != self.columns {
            return Err(GemvError::InputExtent);
        }
        if output.len() != self.rows {
            return Err(GemvError::OutputExtent);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(GemvError::NonFiniteInput);
        }
        multiply(output, input);
        if output.iter().any(|value| !value.is_finite()) {
            return Err(GemvError::NonFiniteOutput);
        }
        Ok(())
    }
}

#[cfg(feature = "blas")]
#[derive(Clone, Debug, PartialEq)]
struct PupilRows {
    rows: core::ops::Range<usize>,
    input_limit: f32,
}

#[cfg(feature = "blas")]
fn max_abs_row_sum(matrix: &[f32], columns: usize) -> f64 {
    matrix
        .chunks_exact(columns)
        .map(|row| row.iter().map(|value| f64::from(value.abs())).sum::<f64>())
        .fold(0.0, f64::max)
}

#[cfg(feature = "experimental-simd-gemv")]
fn pupil_simd_input_limit(matrix: &[f32], columns: usize) -> f32 {
    let row_bound = max_abs_row_sum(matrix, columns);
    if row_bound == 0.0 {
        return f32::MAX;
    }
    // For either reduction, at most 2n rounded Float32 operations bound
    // the absolute forward error by gamma(2n) times the absolute dot
    // product bound, plus an allowance for gradual underflow. Budget half
    // of the 2e-6 absolute pupil parity tolerance for the difference
    // between the two reductions. The remainder covers underflow and
    // rounding of this bound. The same bound keeps all partial sums far
    // inside the finite Float32 range regardless of cancellation.
    let rounding = columns as f64 * f64::from(f32::EPSILON);
    let gamma = rounding / (1.0 - rounding);
    (1.0e-6 / (2.0 * gamma * row_bound)).min(f64::from(f32::MAX)) as f32
}

impl PreparedGemv<f32> {
    pub(crate) fn new_pupil(
        rows: usize,
        columns: usize,
        matrix: Arc<[f32]>,
    ) -> Result<Self, PreparedGemvError> {
        let plan = Self::new(rows, columns, matrix)?;
        #[cfg(feature = "blas")]
        let plan = {
            let mut plan = plan;
            plan.pupil_rows = plan.active_pupil_rows();
            // In particular, widening a subnormal coefficient must preserve it.
            #[cfg(feature = "experimental-simd-gemv")]
            if filtergraph_blas::pupil_sgemv_kernel(rows, columns).is_some()
                && filtergraph_blas::sgemv_default_float_mode()
            {
                plan.pupil_simd_input_limit = Some(pupil_simd_input_limit(&plan.matrix, columns));
            }
            plan
        };
        Ok(plan)
    }

    #[cfg(feature = "blas")]
    fn active_pupil_rows(&self) -> Option<PupilRows> {
        if !(128..=512).contains(&self.rows)
            || self.matrix.len() < 1 << 18
            || !filtergraph_blas::supports_sgemv_dimensions(self.rows, self.columns)
        {
            return None;
        }

        // Leave signed zero coefficients in the evaluated rows.
        let mut active = self
            .matrix
            .chunks_exact(self.columns)
            .enumerate()
            .filter(|(_, row)| row.iter().any(|value| value.to_bits() != 0));
        let (first, _) = active.next()?;
        let last = active.next_back().map_or(first, |(row, _)| row);
        // Keep groups of eight whole so the dispatched kernel need not move
        // active rows from a full group into its scalar tail.
        let rows = (first / 8 * 8)..(last + 1).next_multiple_of(8).min(self.rows);
        if rows.len() > self.rows / 2 {
            return None;
        }

        let row_bound = max_abs_row_sum(
            &self.matrix[rows.start * self.columns..rows.end * self.columns],
            self.columns,
        )
        .max(1.0);
        let rounding = self.columns as f64 * f64::from(f32::EPSILON);
        let input_limit = (f64::from(f32::MAX) / 8.0 / row_bound * (1.0 - rounding)) as f32;
        Some(PupilRows { rows, input_limit })
    }

    pub(crate) fn multiply_pupil(
        &self,
        output: &mut [f32],
        input: &[f32],
    ) -> Result<(), GemvError> {
        self.multiply_with(output, input, |output, input| {
            #[cfg(feature = "experimental-simd-gemv")]
            if let Some(input_limit) = self.pupil_simd_input_limit
                && let Some(kernel) = filtergraph_blas::pupil_sgemv_kernel(self.rows, self.columns)
                && filtergraph_blas::sgemv_default_float_mode()
                && input.iter().all(|value| value.abs() <= input_limit)
            {
                kernel(&self.matrix, input, output);
                return;
            }
            #[cfg(feature = "blas")]
            if let Some(active) = &self.pupil_rows
                && filtergraph_blas::sgemv_default_float_mode()
                && input.iter().all(|value| value.abs() <= active.input_limit)
            {
                let rows = active.rows.clone();
                output[..rows.start].fill(0.0);
                output[rows.end..].fill(0.0);
                let coefficients = &self.matrix[rows.start * self.columns..rows.end * self.columns];
                multiply_row_major(
                    coefficients,
                    rows.len(),
                    self.columns,
                    input,
                    &mut output[rows],
                );
                return;
            }
            multiply_row_major(&self.matrix, self.rows, self.columns, input, output);
        })
    }
}

#[cfg(all(test, feature = "blas"))]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn pupil_span_reuses_dense_coefficients_and_preserves_rejection() {
        let (rows, columns) = (128, 2048);
        let mut matrix = vec![0.0; rows * columns];
        matrix[3 * columns..32 * columns].fill(2.0);
        let plan = PreparedGemv::new_pupil(rows, columns, Arc::from(matrix.clone())).unwrap();
        assert_eq!(plan.pupil_rows.as_ref().unwrap().rows, 0..32);
        let mut output = vec![f32::NAN; rows];
        let mut input = vec![1.0; columns];
        plan.multiply_pupil(&mut output, &input).unwrap();
        assert!(output[3..32].iter().all(|&value| value == 4096.0));
        assert!(
            output[..3]
                .iter()
                .chain(&output[32..])
                .all(|value| value.to_bits() == 0)
        );
        input[0] = f32::NAN;
        assert_eq!(
            plan.multiply_pupil(&mut output, &input),
            Err(GemvError::NonFiniteInput)
        );
        input[0] = f32::MAX;
        assert_eq!(
            plan.multiply_pupil(&mut output, &input),
            Err(GemvError::NonFiniteOutput)
        );

        matrix[(rows - 1) * columns] = -0.0;
        let replacement = PreparedGemv::new_pupil(rows, columns, Arc::from(matrix)).unwrap();
        assert!(replacement.pupil_rows.is_none());
    }
}
