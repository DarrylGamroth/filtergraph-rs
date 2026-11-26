//! Shared matrix-vector arithmetic and backend boundary.

use num_traits::Float;

mod private {
    use num_traits::Float;

    pub trait Gemv: Float {
        fn axpy(alpha: Self, input: &[Self], output: &mut [Self]);
        fn gemv(matrix: &[Self], rows: usize, columns: usize, input: &[Self], output: &mut [Self]);
    }

    fn scalar_axpy<T: Float>(alpha: T, input: &[T], output: &mut [T]) {
        // Match JFG's fused update, preserving column order within each output.
        // Outputs are independent lanes over contiguous prepared coefficients.
        for (&input, output) in input.iter().zip(output) {
            *output = input.mul_add(alpha, *output);
        }
    }

    fn scalar_gemv<T: Float>(matrix: &[T], columns: usize, input: &[T], output: &mut [T]) {
        for (row, value) in matrix.chunks_exact(columns).zip(output) {
            *value = row
                .iter()
                .zip(input)
                .fold(T::zero(), |sum, (&a, &x)| sum + a * x);
        }
    }

    impl Gemv for f32 {
        fn axpy(alpha: Self, input: &[Self], output: &mut [Self]) {
            scalar_axpy(alpha, input, output);
        }
        fn gemv(matrix: &[Self], rows: usize, columns: usize, input: &[Self], output: &mut [Self]) {
            #[cfg(feature = "blas")]
            if filtergraph_blas::supports_sgemv_dimensions(rows, columns) {
                filtergraph_blas::sgemv(rows, columns, matrix, input, output);
                return;
            }
            // Retain the no_std/no-backend profile and generic extents beyond LP64.
            let _ = rows;
            scalar_gemv(matrix, columns, input, output);
        }
    }

    impl Gemv for f64 {
        fn axpy(alpha: Self, input: &[Self], output: &mut [Self]) {
            scalar_axpy(alpha, input, output);
        }
        fn gemv(matrix: &[Self], rows: usize, columns: usize, input: &[Self], output: &mut [Self]) {
            #[cfg(feature = "blas")]
            if filtergraph_blas::supports_dimensions(rows, columns) {
                filtergraph_blas::dgemv(rows, columns, matrix, input, output);
                return;
            }
            let _ = rows;
            scalar_gemv(matrix, columns, input, output);
        }
    }
}

/// Floating-point element supported by prepared matrix operations.
///
/// This sealed trait is implemented for `f32` and `f64` only.
pub trait MatrixElement: Float + private::Gemv {}

impl MatrixElement for f32 {}
impl MatrixElement for f64 {}

pub(crate) fn multiply_row_major<T>(
    matrix: &[T],
    rows: usize,
    columns: usize,
    input: &[T],
    output: &mut [T],
) where
    T: MatrixElement,
{
    assert_eq!(matrix.len(), rows * columns);
    assert_eq!(input.len(), columns);
    assert_eq!(output.len(), rows);
    <T as private::Gemv>::gemv(matrix, rows, columns, input, output);
}

pub(crate) fn multiply_add<T>(alpha: T, input: &[T], output: &mut [T])
where
    T: MatrixElement,
{
    assert_eq!(input.len(), output.len());
    <T as private::Gemv>::axpy(alpha, input, output);
}
