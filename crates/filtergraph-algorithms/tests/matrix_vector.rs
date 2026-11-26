//! Prepared GEMV operation tests.

use std::sync::Arc;

use filtergraph_algorithms::{GemvError, PreparedGemv, PreparedGemvError};

#[test]
fn row_major_f32_reconstruction_is_direct() {
    let gemv = PreparedGemv::new(2, 3, Arc::from([1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
    let mut output = [0.0; 2];
    gemv.multiply(&mut output, &[7.0, 11.0, 13.0]).unwrap();
    assert_eq!(output, [68.0, 161.0]);
}

#[test]
fn row_major_f32_reconstruction_matches_a_scalar_reference() {
    const ROWS: usize = 221;
    const COLUMNS: usize = 376;
    let matrix: Arc<[f32]> = (0..ROWS * COLUMNS)
        .map(|index| ((index * 29 % 257) as f32 - 128.0) / 64.0)
        .collect::<Vec<_>>()
        .into();
    let input: Vec<f32> = (0..COLUMNS)
        .map(|index| ((index * 17 % 113) as f32 - 56.0) / 32.0)
        .collect();
    let expected: Vec<(f32, f32)> = matrix
        .chunks_exact(COLUMNS)
        .map(|row| {
            let reference = row
                .iter()
                .zip(&input)
                .fold(0.0_f32, |sum, (coefficient, value)| {
                    sum + coefficient * value
                });
            let product_magnitude = row
                .iter()
                .zip(&input)
                .map(|(coefficient, value)| (coefficient * value).abs())
                .sum();
            (reference, product_magnitude)
        })
        .collect();
    let gemv = PreparedGemv::new(ROWS, COLUMNS, matrix).unwrap();
    let mut output = vec![0.0; ROWS];

    gemv.multiply(&mut output, &input).unwrap();

    let rounding = COLUMNS as f32 * f32::EPSILON;
    let gamma = rounding / (1.0 - rounding);
    for (row, (actual, (reference, product_magnitude))) in output.iter().zip(&expected).enumerate()
    {
        // Both reduction orders must remain within their combined standard
        // forward-error scale, including rows with substantial cancellation.
        let tolerance = 2.0 * gamma * product_magnitude + f32::MIN_POSITIVE;
        assert!(
            (actual - reference).abs() <= tolerance,
            "row {row}: expected {reference}, received {actual}, tolerance {tolerance}"
        );
    }
}

#[test]
fn f64_and_extent_validation_use_the_same_contract() {
    let gemv = PreparedGemv::new(2, 2, Arc::from([0.5_f64, -2.0, 3.0, 0.25])).unwrap();
    let mut output = [0.0; 2];
    gemv.multiply(&mut output, &[4.0, 8.0]).unwrap();
    assert_eq!(output, [-14.0, 14.0]);
    assert_eq!(
        gemv.multiply(&mut output, &[1.0]),
        Err(GemvError::InputExtent)
    );
    assert_eq!(
        gemv.multiply(&mut output[..1], &[1.0, 2.0]),
        Err(GemvError::OutputExtent)
    );
    assert_eq!(
        gemv.multiply(&mut output, &[f64::NAN, 2.0]),
        Err(GemvError::NonFiniteInput)
    );
}

#[test]
fn preparation_rejects_invalid_dimensions_and_coefficients() {
    assert_eq!(
        PreparedGemv::<f32>::new(0, 1, Arc::from([])),
        Err(PreparedGemvError::EmptyDimension)
    );
    assert_eq!(
        PreparedGemv::<f32>::new(usize::MAX, 2, Arc::from([])),
        Err(PreparedGemvError::ExtentOverflow)
    );
    assert_eq!(
        PreparedGemv::new(2, 2, Arc::from([1.0_f32; 3])),
        Err(PreparedGemvError::MatrixExtent)
    );
    assert_eq!(
        PreparedGemv::new(1, 1, Arc::from([f32::NAN])),
        Err(PreparedGemvError::NonFiniteMatrix)
    );
}

#[test]
fn nonfinite_result_is_reported() {
    let gemv = PreparedGemv::new(1, 1, Arc::from([f32::MAX])).unwrap();
    let mut output = [0.0];
    assert_eq!(
        gemv.multiply(&mut output, &[2.0]),
        Err(GemvError::NonFiniteOutput)
    );
}

#[test]
fn matrix_beyond_native_scratch_limit_retains_scalar_execution() {
    // The native interface caps rows+columns at 8064 for its private stack.
    let columns = 8064;
    let gemv = PreparedGemv::new(1, columns, Arc::from(vec![1.0_f64; columns])).unwrap();
    let mut output = [0.0];
    gemv.multiply(&mut output, &vec![1.0; columns]).unwrap();
    assert_eq!(output, [columns as f64]);
}
