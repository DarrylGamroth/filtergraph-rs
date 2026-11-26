//! Portable pyramid reconstructor declaration tests.

use std::sync::Arc;

use filtergraph_algorithms::schemas::{CONTROLLER_RESIDUAL_ERROR_V1, PWFS_RECONSTRUCTOR_V1};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmOperation, AlgorithmOperationPlan, PreparedGemv,
    PyramidPupilReconstructorF32, PyramidPupilReconstructorF32Configuration,
};

#[test]
fn pupil_reconstructor_is_the_typed_dense_operation_with_pwfs_schemas() {
    const SELECTED_PIXELS: usize = 2;
    const INPUTS: usize = 4 * SELECTED_PIXELS;
    const RECONSTRUCTED: usize = 3;

    let matrix = (0..RECONSTRUCTED * INPUTS)
        .map(|index| (index as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let input = (0..INPUTS)
        .map(|index| (index as f32 + 1.0) / 9.0)
        .collect::<Vec<_>>();
    let direct = PreparedGemv::new(RECONSTRUCTED, INPUTS, Arc::from(matrix.clone())).unwrap();
    let mut expected = [0.0_f32; RECONSTRUCTED];
    direct.multiply(&mut expected, &input).unwrap();

    let mut declared =
        PyramidPupilReconstructorF32::prepare(PyramidPupilReconstructorF32Configuration {
            reconstructed_count: RECONSTRUCTED,
            selected_pixel_count: SELECTED_PIXELS,
            initial_reconstructor: Some(matrix),
            reconstructed_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
        })
        .unwrap();
    assert_eq!(
        PyramidPupilReconstructorF32::OPERATION,
        AlgorithmOperation::DenseF32
    );
    assert!(matches!(
        PyramidPupilReconstructorF32::operation_plan(declared.plan()),
        AlgorithmOperationPlan::DenseF32(_)
    ));
    assert_eq!(declared.formats()[0].shape(), &[4, SELECTED_PIXELS as u32]);
    assert_eq!(
        declared.formats()[0].schema(),
        "pipewireao/pyramid-reconstruction-pixels-f32/1"
    );
    assert_eq!(declared.formats()[1].schema(), CONTROLLER_RESIDUAL_ERROR_V1);
    assert_eq!(declared.formats()[2].schema(), PWFS_RECONSTRUCTOR_V1);

    let mut actual = [0.0_f32; RECONSTRUCTED];
    PyramidPupilReconstructorF32::process(&mut declared, &input, &mut actual).unwrap();
    assert_eq!(actual, expected);

    let identity_rows = [
        1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 2
    ];
    declared
        .replace_parameter(2, AlgorithmInput::from(&identity_rows[..]))
        .unwrap()
        .unwrap();
    PyramidPupilReconstructorF32::process(&mut declared, &input, &mut actual).unwrap();
    assert_eq!(actual, input[..RECONSTRUCTED]);
}

#[cfg(feature = "experimental-simd-gemv")]
#[test]
fn specialized_pupil_dispatch_preserves_both_openblas_overflow_outcomes() {
    const ROWS: usize = 1509;
    const COLUMNS: usize = 11104;

    let mut matrix = vec![0.0; ROWS * COLUMNS];
    matrix[0] = f32::MAX;
    matrix[8] = -f32::MAX;
    matrix[16] = f32::MAX;
    let mut declared =
        PyramidPupilReconstructorF32::prepare(PyramidPupilReconstructorF32Configuration {
            reconstructed_count: ROWS,
            selected_pixel_count: COLUMNS / 4,
            initial_reconstructor: Some(matrix.clone()),
            reconstructed_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
        })
        .unwrap();
    let mut input = vec![1.0; COLUMNS];
    let mut actual = [0.0; ROWS];
    let mut expected = [0.0; ROWS];

    for large_input in [false, true] {
        if large_input {
            // Move the large magnitudes from the coefficients to the input.
            // This replacement gives the opposite OpenBLAS overflow outcome.
            matrix[0] = 1.0;
            matrix[8] = 1.0;
            matrix[16] = -1.0;
            declared
                .replace_parameter(2, AlgorithmInput::from(&matrix[..]))
                .unwrap()
                .unwrap();
            input[0] = f32::MAX;
            input[8] = f32::MAX;
            input[16] = f32::MAX;
        }

        filtergraph_blas::sgemv(ROWS, COLUMNS, &matrix, &input, &mut expected);
        let result = PyramidPupilReconstructorF32::process(&mut declared, &input, &mut actual);
        let finite = expected.iter().all(|value| value.is_finite());
        assert_eq!(result.is_ok(), finite);
        if finite {
            assert_eq!(actual, expected);
        }

        // The archived Zen 3 OpenBLAS kernel and prototype AVX2 kernel make
        // opposite acceptance decisions for these two reduction orders.
        if filtergraph_blas::core_name().eq_ignore_ascii_case("zen")
            && cfg!(not(target_feature = "avx512f"))
            && let Some(kernel) = filtergraph_blas::pupil_sgemv_kernel(ROWS, COLUMNS)
        {
            let mut unguarded = [0.0; ROWS];
            kernel(&matrix, &input, &mut unguarded);
            assert_ne!(unguarded[0].is_finite(), finite);
            assert_eq!(finite, !large_input);
        }
    }

    input[0] = 1.0;
    input[8] = 1.0;
    input[16] = 1.0;
    PyramidPupilReconstructorF32::process(&mut declared, &input, &mut actual).unwrap();
    assert_eq!(actual[0], 1.0);

    let before = actual;
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // Even a coordinate with all-zero coefficients must be checked.
        input[COLUMNS - 1] = nonfinite;
        assert!(PyramidPupilReconstructorF32::process(&mut declared, &input, &mut actual).is_err());
        assert_eq!(actual, before);
    }
}

#[cfg(feature = "experimental-simd-gemv")]
#[test]
fn specialized_pupil_dispatch_bounds_cancellation_error() {
    const ROWS: usize = 1509;
    const COLUMNS: usize = 11104;
    const LAST_ROW: usize = (ROWS - 1) * COLUMNS;

    let Some(kernel) = filtergraph_blas::pupil_sgemv_kernel(ROWS, COLUMNS) else {
        return;
    };
    assert!(filtergraph_blas::pupil_sgemv_kernel(ROWS + 1, COLUMNS).is_none());
    assert!(filtergraph_blas::pupil_sgemv_kernel(ROWS, COLUMNS + 4).is_none());
    let mut matrix = vec![0.0; ROWS * COLUMNS];
    matrix[0] = f32::from_bits(1);
    matrix[LAST_ROW] = 16_777_216.0;
    matrix[LAST_ROW + 8] = 1.0;
    matrix[LAST_ROW + 16] = -16_777_216.0;
    matrix[LAST_ROW + COLUMNS - 1] = f32::from_bits(1);
    let input = vec![1.0; COLUMNS];
    let mut declared =
        PyramidPupilReconstructorF32::prepare(PyramidPupilReconstructorF32Configuration {
            reconstructed_count: ROWS,
            selected_pixel_count: COLUMNS / 4,
            initial_reconstructor: Some(matrix.clone()),
            reconstructed_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
        })
        .unwrap();
    let mut actual = [0.0; ROWS];
    let mut expected = actual;
    let mut simd = actual;

    assert!(filtergraph_blas::sgemv_default_float_mode());
    // This same-shaped direct plan has no pupil kernel selected.
    let direct = PreparedGemv::new(ROWS, COLUMNS, Arc::from(matrix.as_slice())).unwrap();
    assert_eq!(declared.plan(), &direct);
    direct.multiply(&mut expected, &input).unwrap();
    kernel(&matrix, &input, &mut simd);
    assert_eq!(simd[ROWS - 1], 1.0);
    if filtergraph_blas::core_name().eq_ignore_ascii_case("zen") {
        assert!((simd[ROWS - 1] - expected[ROWS - 1]).abs() > 2.0e-6);
    }
    PyramidPupilReconstructorF32::process(&mut declared, &input, &mut actual).unwrap();
    assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));

    // The same cancellation scaled by 2^-40 stays within the absolute
    // error budget while still distinguishing both reduction orders on Zen.
    let scale = 2.0_f32.powi(-40);
    for offset in [0, 8, 16] {
        matrix[LAST_ROW + offset] *= scale;
    }
    declared
        .replace_parameter(2, AlgorithmInput::from(&matrix[..]))
        .unwrap()
        .unwrap();
    declared.plan().multiply(&mut expected, &input).unwrap();
    kernel(&matrix, &input, &mut simd);
    PyramidPupilReconstructorF32::process(&mut declared, &input, &mut actual).unwrap();
    assert_eq!(actual.map(f32::to_bits), simd.map(f32::to_bits));
    assert_eq!(actual[0].to_bits(), 1);
    assert_eq!(actual[ROWS - 1], scale);
    if filtergraph_blas::core_name().eq_ignore_ascii_case("zen") {
        assert_ne!(actual[ROWS - 1], expected[ROWS - 1]);
    }
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(x, y)| (x - y).abs() <= 2.0e-6)
    );
    assert!(
        PyramidPupilReconstructorF32::process(&mut declared, &input[..COLUMNS - 1], &mut actual)
            .is_err()
    );
    assert!(
        PyramidPupilReconstructorF32::process(&mut declared, &input, &mut actual[..ROWS - 1])
            .is_err()
    );
    assert_eq!(actual.map(f32::to_bits), simd.map(f32::to_bits));
}

#[test]
fn empty_matrix_and_invalid_sparse_coefficients_preserve_prepared_pupil_shapes() {
    use filtergraph_algorithms::{DeclarationParameterError, PreparedGemvError};
    let mut prepared =
        PyramidPupilReconstructorF32::prepare(PyramidPupilReconstructorF32Configuration {
            reconstructed_count: 1,
            selected_pixel_count: 2,
            initial_reconstructor: None,
            reconstructed_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
        })
        .unwrap();

    let input = [2.0_f32; 8];
    let mut output = [17.0];
    PyramidPupilReconstructorF32::process(&mut prepared, &input, &mut output).unwrap();
    assert_eq!(output, [0.0]);
    assert_eq!(prepared.formats()[2].shape(), &[1, 8]);

    let error = prepared
        .replace_parameter(2, AlgorithmInput::from(&[f32::NAN; 8][..]))
        .unwrap()
        .unwrap_err();
    assert_eq!(
        error,
        DeclarationParameterError::Algorithm(PreparedGemvError::NonFiniteMatrix)
    );
    assert!(error.to_string().contains("finite"));
    PyramidPupilReconstructorF32::process(&mut prepared, &input, &mut output).unwrap();
    assert_eq!(output, [0.0]);
}
