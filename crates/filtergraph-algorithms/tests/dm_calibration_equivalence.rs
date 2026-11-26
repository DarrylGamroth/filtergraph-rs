//! Independent coordinate, representability, and accepted-history obligations.
use filtergraph_algorithms::*;
#[test]
fn asymmetric_controller_projection_adopts_only_finite_complete_replacements() {
    let mut p = ControllerToVdmF32::prepare(ControllerToVdmF32Configuration {
        controller_command_count: 2,
        active_vdm_count: 3,
    })
    .unwrap();
    let matrix: [f32; 6] = [1., 0.25, -0.5, 1.5, 0.75, -1.];
    p.replace_parameter(2, AlgorithmInput::from(&matrix[..]))
        .unwrap()
        .unwrap();
    let mut out = [71.; 3];
    ControllerToVdmF32::process(&mut p, &[0.375, -1.25], &mut out).unwrap();
    assert_eq!(out, [0.0625, -2.0625, 1.53125]);
    assert!(
        p.replace_parameter(2, AlgorithmInput::from(&[f32::NAN; 6][..]))
            .unwrap()
            .is_err()
    );
    ControllerToVdmF32::reset(&mut p);
    ControllerToVdmF32::process(&mut p, &[0.375, -1.25], &mut out).unwrap();
    assert_eq!(out, [0.0625, -2.0625, 1.53125]);
    assert!(ControllerToVdmF32::process(&mut p, &[f32::NAN, 1.], &mut out).is_err());
    assert_eq!(out, [0.0625, -2.0625, 1.53125]);
}
#[test]
fn power_projection_handles_unrepresentable_squared_norm_and_zero_budget() {
    let mut p = PdmCommandPowerLimitF32::prepare(PdmCommandPowerLimitF32Configuration {
        actuator_count: 4,
        max_command_power: 2.,
    })
    .unwrap();
    let mut out = [71.; 4];
    PdmCommandPowerLimitF32::process(&mut p, &[f32::MAX, -f32::MAX, 0., 0.], &mut out).unwrap();
    assert!((out[0] - 1.).abs() < 2. * f32::EPSILON);
    assert!((out[1] + 1.).abs() < 2. * f32::EPSILON);
    assert_eq!(out[2..], [0., 0.]);
    let mut zero = PdmCommandPowerLimitF32::prepare(PdmCommandPowerLimitF32Configuration {
        actuator_count: 4,
        max_command_power: 0.,
    })
    .unwrap();
    PdmCommandPowerLimitF32::process(&mut zero, &[1., -2., 3., -4.], &mut out).unwrap();
    assert_eq!(out, [0.; 4]);
}
#[test]
fn calibrated_null_mode_removal_uses_preferred_command_and_preserves_failure_output() {
    let mut p = PdmStrokeOptimizationF32::prepare(PdmStrokeOptimizationF32Configuration {
        null_mode_count: 1,
        preferred_command: vec![0.5, 0., 0., 0.],
        command_to_null_modes: vec![0.25; 4],
        null_modes_to_command: vec![1.; 4],
        actuator_count: 4,
    })
    .unwrap();
    let mut out = [71.; 4];
    PdmStrokeOptimizationF32::process(&mut p, &[1., 2., 3., 4.], &mut out).unwrap();
    assert_eq!(out, [-1.375, -0.375, 0.625, 1.625]);
    assert!(PdmStrokeOptimizationF32::process(&mut p, &[f32::NAN, 2., 3., 4.], &mut out).is_err());
    assert_eq!(out, [-1.375, -0.375, 0.625, 1.625]);
}
#[test]
fn matched_rejected_sample_does_not_advance_the_reference_exposure() {
    let mut p = MatchedCalibrationDifferencingF32::prepare(
        MatchedCalibrationDifferencingF32Configuration {
            measurement_count: 2,
            measurements_schema: "org.example.m/1".into(),
            corrector_sample_schema: "org.example.c/1".into(),
            filtered_measurements_schema: "org.example.fm/1".into(),
            filtered_corrector_sample_schema: "org.example.fc/1".into(),
            corrector_coordinate_count: 1,
        },
    )
    .unwrap();
    let (mut a, mut b, mut filtered) = ([71.; 2], [71.], [true]);
    MatchedCalibrationDifferencingF32::process(
        &mut p,
        &[1., -2.],
        &[0.5],
        &mut a,
        &mut b,
        &mut filtered,
    )
    .unwrap();
    assert!(!filtered[0]);
    assert!(
        MatchedCalibrationDifferencingF32::process(
            &mut p,
            &[9., 9.],
            &[f32::NAN],
            &mut a,
            &mut b,
            &mut filtered
        )
        .is_err()
    );
    MatchedCalibrationDifferencingF32::process(
        &mut p,
        &[1.5, -1.],
        &[-0.25],
        &mut a,
        &mut b,
        &mut filtered,
    )
    .unwrap();
    assert_eq!(a, [0.5, 1.]);
    assert_eq!(b, [-0.75]);
    assert!(filtered[0]);
}
#[test]
fn stability_declaration_reports_zero_based_coordinate_for_second_column_sign_change() {
    let mut p =
        InteractionMatrixStabilityF32::prepare(InteractionMatrixStabilityF32Configuration {
            measurement_count: 2,
            corrector_coordinate_count: 2,
            minimum_column_norm: 0.01,
            interaction_matrix_schema: "org.example.matrix/1".into(),
        })
        .unwrap();
    let (mut status, mut coordinate, mut norms, mut column, mut max_coordinate, mut frobenius) =
        ([99], [99], [71.; 2], [71.], [99], [71.]);
    InteractionMatrixStabilityF32::process(
        &mut p,
        &[1., 0., 0., 1.],
        &[1., 0., 0., -1.],
        &mut status,
        &mut coordinate,
        &mut norms,
        &mut column,
        &mut max_coordinate,
        &mut frobenius,
    )
    .unwrap();
    assert_eq!(status, [2]);
    assert_eq!(coordinate, [1]);
    assert_eq!(max_coordinate, [1]);
    assert_eq!(norms, [1., 1.]);
    assert_eq!(column, [2.]);
}
#[test]
fn null_mode_coefficient_overflow_rejects_before_caller_publication() {
    let mut p = PdmStrokeOptimizationF32::prepare(PdmStrokeOptimizationF32Configuration {
        null_mode_count: 2,
        preferred_command: vec![0.125, -0.25, 0.375, -0.5],
        command_to_null_modes: vec![0.5, -0.25, 0.125, 0.75, -0.375, 0.5, 0.25, -0.125],
        null_modes_to_command: vec![1., 0.25, -0.5, 0.75, 0.375, -0.25, 0.125, 1.],
        actuator_count: 4,
    })
    .unwrap();
    let mut out = [71.; 4];
    assert!(matches!(
        PdmStrokeOptimizationF32::process(&mut p, &[f32::MAX; 4], &mut out),
        Err(DeclarationProcessError::Algorithm(
            PdmStrokeOptimizationError::NonFiniteOutput
        ))
    ));
    assert_eq!(out, [71.; 4]);
    PdmStrokeOptimizationF32::process(&mut p, &[0.5, -1.25, 2., -0.75], &mut out).unwrap();
    assert!(out.iter().all(|x| x.is_finite()));
}
#[test]
fn generic_f64_null_mode_overflow_preserves_output() {
    let p = PdmStrokeOptimizationPlan::new(
        4,
        2,
        std::sync::Arc::from([0.125_f64, -0.25, 0.375, -0.5]),
        std::sync::Arc::from([0.5, -0.25, 0.125, 0.75, -0.375, 0.5, 0.25, -0.125]),
        std::sync::Arc::from([1., 0.25, -0.5, 0.75, 0.375, -0.25, 0.125, 1.]),
    )
    .unwrap();
    let mut w = p.workspace();
    let mut out = [71.; 4];
    assert_eq!(
        p.process(&mut out, &mut w, &[f64::MAX; 4]),
        Err(PdmStrokeOptimizationError::NonFiniteOutput)
    );
    assert_eq!(out, [71.; 4]);
    p.process(&mut out, &mut w, &[0.5, -1.25, 2., -0.75])
        .unwrap();
    assert!(out.iter().all(|x| x.is_finite()));
}
