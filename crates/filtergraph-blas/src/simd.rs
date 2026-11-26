//! Experimental row-major Float32 GEMV for a fixed pupil-matrix shape.

use core::arch::x86_64 as arch;

#[cfg(not(target_feature = "avx512f"))]
use arch::{
    __m256 as Vector, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    _mm256_add_ps as add, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps as fmadd,
    _mm256_loadu_ps as load, _mm256_setzero_ps as zero,
};
#[cfg(target_feature = "avx512f")]
use arch::{
    __m512 as Vector, _mm512_add_ps as add, _mm512_fmadd_ps as fmadd, _mm512_loadu_ps as load,
    _mm512_reduce_add_ps as sum, _mm512_setzero_ps as zero,
};

pub(super) const PUPIL_ROWS: usize = 1_509;
pub(super) const PUPIL_COLUMNS: usize = 11_104;

const LANES: usize = size_of::<Vector>() / size_of::<f32>();
const ROW_BLOCK: usize = 4;

pub(super) fn multiply(matrix: &[f32], input: &[f32], output: &mut [f32]) {
    assert_eq!(matrix.len(), PUPIL_ROWS * PUPIL_COLUMNS);
    assert_eq!(input.len(), PUPIL_COLUMNS);
    assert_eq!(output.len(), PUPIL_ROWS);
    // SAFETY: the extents match and the checked borrows are disjoint.
    unsafe {
        sgemv::<PUPIL_ROWS, PUPIL_COLUMNS>(matrix.as_ptr(), input.as_ptr(), output.as_mut_ptr());
    }
}

#[cfg(not(target_feature = "avx512f"))]
#[inline]
fn sum(vector: Vector) -> f32 {
    // SAFETY: this module compiles only for the required target features.
    unsafe {
        let halves = _mm_add_ps(
            _mm256_castps256_ps128(vector),
            _mm256_extractf128_ps::<1>(vector),
        );
        let pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
        _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps::<0x55>(pairs, pairs)))
    }
}

/// Matrix and input must contain ROWS * COLUMNS and COLUMNS readable elements,
/// respectively. Output must permit writing ROWS elements without aliasing
/// either read region. The pointers need only ordinary Float32 alignment.
#[inline(never)]
unsafe fn sgemv<const ROWS: usize, const COLUMNS: usize>(
    matrix: *const f32,
    input: *const f32,
    output: *mut f32,
) {
    let blocked_rows = ROWS / ROW_BLOCK * ROW_BLOCK;
    // SAFETY: each call covers a disjoint consecutive group inside the checked
    // matrix and output extents and shares the same full input vector.
    unsafe {
        for row in (0..blocked_rows).step_by(ROW_BLOCK) {
            dot_rows::<ROW_BLOCK, COLUMNS>(matrix.add(row * COLUMNS), input, output.add(row));
        }
        for row in blocked_rows..ROWS {
            dot_rows::<1, COLUMNS>(matrix.add(row * COLUMNS), input, output.add(row));
        }
    }
}

#[inline(always)]
unsafe fn dot_rows<const ROWS: usize, const COLUMNS: usize>(
    matrix: *const f32,
    input: *const f32,
    output: *mut f32,
) {
    let block_end = COLUMNS / (2 * LANES) * (2 * LANES);
    let vector_end = COLUMNS / LANES * LANES;
    // SAFETY: the caller supplies ROWS complete matrix rows, COLUMNS readable
    // input elements and ROWS exclusive output elements. Vector loads are
    // unaligned and stop at the last full vector. The scalar tail stays in the
    // row. The module's compile-time gate guarantees the ISA requirements.
    unsafe {
        let matrix_rows: [_; ROWS] = core::array::from_fn(|row| matrix.add(row * COLUMNS));
        let mut even = [zero(); ROWS];
        let mut odd = [zero(); ROWS];
        for column in (0..block_end).step_by(2 * LANES) {
            let x0 = load(input.add(column));
            let x1 = load(input.add(column + LANES));
            for row in 0..ROWS {
                let coefficients = matrix_rows[row].add(column);
                even[row] = fmadd(load(coefficients), x0, even[row]);
                odd[row] = fmadd(load(coefficients.add(LANES)), x1, odd[row]);
            }
        }
        if block_end != vector_end {
            let x = load(input.add(block_end));
            for row in 0..ROWS {
                even[row] = fmadd(load(matrix_rows[row].add(block_end)), x, even[row]);
            }
        }
        for row in 0..ROWS {
            let mut dot = sum(add(even[row], odd[row]));
            for column in vector_end..COLUMNS {
                dot = (*matrix_rows[row].add(column)).mul_add(*input.add(column), dot);
            }
            output.add(row).write(dot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn operands(rows: usize, columns: usize) -> (Vec<f32>, Vec<f32>) {
        let input = (0..columns).map(|j| (j % 17) as f32 - 8.0).collect();
        let matrix = (0..rows)
            .flat_map(|i| (0..columns).map(move |j| ((i * 13 + j * 7) % 37) as f32 - 18.0))
            .collect();
        (matrix, input)
    }

    fn check(matrix: &[f32], input: &[f32], output: &[f32]) {
        for (row, (coefficients, &actual)) in
            matrix.chunks_exact(input.len()).zip(output).enumerate()
        {
            let expected = coefficients
                .iter()
                .zip(input)
                .map(|(a, x)| a * x)
                .sum::<f32>();
            // All products and partial sums are small integers represented
            // exactly, independently of accumulation order.
            assert_eq!(actual, expected, "row {row}");
        }
    }

    fn check_tails<const ROWS: usize, const COLUMNS: usize>() {
        let (matrix, input) = operands(ROWS, COLUMNS);
        let mut output = [f32::NAN; ROWS];
        // SAFETY: all three allocations are disjoint and have the exact extents.
        unsafe {
            sgemv::<ROWS, COLUMNS>(matrix.as_ptr(), input.as_ptr(), output.as_mut_ptr());
        }
        check(&matrix, &input, &output);
    }

    #[test]
    fn row_blocks_and_column_tails() {
        check_tails::<1, { LANES - 1 }>();
        check_tails::<3, LANES>();
        check_tails::<4, { 2 * LANES }>();
        check_tails::<5, { 3 * LANES + 1 }>();
        check_tails::<9, { 2 * LANES + 3 }>();
    }

    #[test]
    fn checked_pupil_sgemv() {
        let (matrix, input) = operands(PUPIL_ROWS, PUPIL_COLUMNS);
        let mut output = [f32::NAN; PUPIL_ROWS];
        let kernel = crate::pupil_sgemv_kernel(PUPIL_ROWS, PUPIL_COLUMNS).unwrap();
        kernel(&matrix, &input, &mut output);
        check(&matrix, &input, &output);
    }
}
