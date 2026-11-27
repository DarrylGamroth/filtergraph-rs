//! Experimental row-major Float32 GEMV for a fixed pupil-matrix shape.

#![forbid(unsafe_code)]

pub(super) const PUPIL_ROWS: usize = 1_509;
pub(super) const PUPIL_COLUMNS: usize = 11_104;

// The reduction has two groups of eight partial sums per row. LLVM may
// pack the groups together when the compilation target has wider vectors.
const HALF_BLOCK: usize = 8;
const ROW_BLOCK: usize = 4;

pub(super) fn multiply(matrix: &[f32], input: &[f32], output: &mut [f32]) {
    checked_sgemv::<PUPIL_ROWS, PUPIL_COLUMNS>(matrix, input, output);
}

#[inline]
fn checked_sgemv<const ROWS: usize, const COLUMNS: usize>(
    matrix: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    assert_eq!(matrix.len(), ROWS * COLUMNS);
    assert_eq!(input.len(), COLUMNS);
    assert_eq!(output.len(), ROWS);
    sgemv::<COLUMNS>(matrix, input.try_into().unwrap(), output);
}

#[inline(never)]
fn sgemv<const COLUMNS: usize>(matrix: &[f32], input: &[f32; COLUMNS], output: &mut [f32]) {
    let (row_blocks, remaining_rows) = matrix.as_chunks::<COLUMNS>().0.as_chunks::<ROW_BLOCK>();
    let (output_blocks, remaining_output) = output.as_chunks_mut::<ROW_BLOCK>();
    for (rows, output) in row_blocks.iter().zip(output_blocks) {
        dot_rows(rows, input, output);
    }
    for (row, output) in remaining_rows.iter().zip(remaining_output) {
        dot_rows(
            core::array::from_ref(row),
            input,
            core::array::from_mut(output),
        );
    }
}

#[inline(always)]
fn dot_rows<const ROWS: usize, const COLUMNS: usize>(
    matrix_rows: &[[f32; COLUMNS]; ROWS],
    input: &[f32; COLUMNS],
    output: &mut [f32; ROWS],
) {
    let row_blocks = matrix_rows
        .each_ref()
        .map(|row| row.as_chunks::<{ 2 * HALF_BLOCK }>().0);
    let (input_blocks, remaining_input) = input.as_chunks::<{ 2 * HALF_BLOCK }>();
    let mut even = [[0.0f32; HALF_BLOCK]; ROWS];
    let mut odd = [[0.0f32; HALF_BLOCK]; ROWS];
    for (column, x) in input_blocks.iter().enumerate() {
        for row in 0..ROWS {
            let coefficients = &row_blocks[row][column];
            for lane in 0..HALF_BLOCK {
                even[row][lane] = coefficients[lane].mul_add(x[lane], even[row][lane]);
                odd[row][lane] =
                    coefficients[HALF_BLOCK + lane].mul_add(x[HALF_BLOCK + lane], odd[row][lane]);
            }
        }
    }
    let block_end = input_blocks.len() * 2 * HALF_BLOCK;
    let (last_half_block, tail) = remaining_input.as_chunks::<HALF_BLOCK>();
    for (row, output) in output.iter_mut().enumerate() {
        if let Some(x) = last_half_block.first() {
            let coefficients = &matrix_rows[row][block_end..block_end + HALF_BLOCK];
            for lane in 0..HALF_BLOCK {
                even[row][lane] = coefficients[lane].mul_add(x[lane], even[row][lane]);
            }
        }
        // Preserve the reduction tree across compilation targets.
        let a: [f32; HALF_BLOCK] = core::array::from_fn(|i| even[row][i] + odd[row][i]);
        let mut dot = ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
        for (&coefficient, &x) in matrix_rows[row][COLUMNS - tail.len()..].iter().zip(tail) {
            dot = coefficient.mul_add(x, dot);
        }
        *output = dot;
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
        checked_sgemv::<ROWS, COLUMNS>(&matrix, &input, &mut output);
        check(&matrix, &input, &output);
    }

    #[test]
    fn reduction_preserves_small_term_during_cancellation() {
        let mut matrix = [0.0; 33];
        matrix[0] = 16_777_216.0;
        matrix[2] = 1.0;
        matrix[4] = -16_777_216.0;
        let mut output = [f32::NAN];
        checked_sgemv::<1, 33>(&matrix, &[1.0; 33], &mut output);
        assert_eq!(output, [1.0]);
    }

    #[test]
    fn row_blocks_and_column_tails() {
        check_tails::<1, { HALF_BLOCK - 1 }>();
        check_tails::<3, HALF_BLOCK>();
        check_tails::<4, { 2 * HALF_BLOCK }>();
        check_tails::<5, { 3 * HALF_BLOCK + 1 }>();
        check_tails::<9, { 2 * HALF_BLOCK + 3 }>();
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
