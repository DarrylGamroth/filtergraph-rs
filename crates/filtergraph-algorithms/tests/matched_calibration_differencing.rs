//! Matched calibration first-difference tests.

use filtergraph_algorithms::schemas::MATCHED_CALIBRATION_FILTERED_V1;
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmPlan, DocrimeCovarianceInput, DocrimeCovariancePlan,
    ElementType, MatchedCalibrationDifferencingError, MatchedCalibrationDifferencingF32,
    MatchedCalibrationDifferencingF32Configuration, MatchedCalibrationDifferencingInput,
    MatchedCalibrationDifferencingInputKind, MatchedCalibrationDifferencingOutput,
    MatchedCalibrationDifferencingOutputKind, MatchedCalibrationDifferencingPlan,
    MatchedCalibrationDifferencingPlanError, MatchedCalibrationDifferencingStatus,
};

#[test]
fn first_pair_primes_zero_output_and_later_pairs_are_matched_differences() {
    let plan = MatchedCalibrationDifferencingPlan::new(2, 2).unwrap();
    let mut workspace = plan.workspace();
    let mut measurements = [9.0_f64; 2];
    let mut corrector_sample = [8.0_f64; 2];
    let mut status = MatchedCalibrationDifferencingStatus::Filtered;

    plan.process(
        MatchedCalibrationDifferencingOutput::new(
            &mut measurements,
            &mut corrector_sample,
            &mut status,
        ),
        &mut workspace,
        MatchedCalibrationDifferencingInput::new(&[10.0, 20.0], &[1.0, -2.0]),
    )
    .unwrap();
    assert_eq!(measurements, [0.0, 0.0]);
    assert_eq!(corrector_sample, [0.0, 0.0]);
    assert_eq!(status, MatchedCalibrationDifferencingStatus::Priming);
    assert!(workspace.is_primed());

    plan.process(
        MatchedCalibrationDifferencingOutput::new(
            &mut measurements,
            &mut corrector_sample,
            &mut status,
        ),
        &mut workspace,
        MatchedCalibrationDifferencingInput::new(&[13.0, 14.0], &[0.5, 4.0]),
    )
    .unwrap();
    assert_eq!(measurements, [3.0, -6.0]);
    assert_eq!(corrector_sample, [-0.5, 6.0]);
    assert_eq!(status, MatchedCalibrationDifferencingStatus::Filtered);
    assert_eq!(workspace.previous_measurements(), [13.0, 14.0]);
    assert_eq!(workspace.previous_corrector_sample(), [0.5, 4.0]);

    plan.process(
        MatchedCalibrationDifferencingOutput::new(
            &mut measurements,
            &mut corrector_sample,
            &mut status,
        ),
        &mut workspace,
        MatchedCalibrationDifferencingInput::new(&[13.0, 14.0], &[0.5, 4.0]),
    )
    .unwrap();
    assert_eq!(measurements, [0.0, 0.0]);
    assert_eq!(corrector_sample, [0.0, 0.0]);
}

#[test]
fn f32_and_f64_apply_the_same_matched_filter() {
    let plan32 = MatchedCalibrationDifferencingPlan::new(1, 1).unwrap();
    let plan64 = MatchedCalibrationDifferencingPlan::new(1, 1).unwrap();
    let mut workspace32 = plan32.workspace();
    let mut workspace64 = plan64.workspace();
    let mut measurement32 = [0.0_f32];
    let mut measurement64 = [0.0_f64];
    let mut corrector32 = [0.0_f32];
    let mut corrector64 = [0.0_f64];
    let mut status32 = MatchedCalibrationDifferencingStatus::Priming;
    let mut status64 = MatchedCalibrationDifferencingStatus::Priming;

    for (measurement, corrector) in [(1.0, -2.0), (1.25, -1.5), (0.5, 3.0)] {
        plan32
            .process(
                MatchedCalibrationDifferencingOutput::new(
                    &mut measurement32,
                    &mut corrector32,
                    &mut status32,
                ),
                &mut workspace32,
                MatchedCalibrationDifferencingInput::new(
                    &[measurement as f32],
                    &[corrector as f32],
                ),
            )
            .unwrap();
        plan64
            .process(
                MatchedCalibrationDifferencingOutput::new(
                    &mut measurement64,
                    &mut corrector64,
                    &mut status64,
                ),
                &mut workspace64,
                MatchedCalibrationDifferencingInput::new(&[measurement], &[corrector]),
            )
            .unwrap();
        assert_eq!(measurement32[0] as f64, measurement64[0]);
        assert_eq!(corrector32[0] as f64, corrector64[0]);
        assert_eq!(status32, status64);
    }
}

#[test]
fn rejected_input_preserves_history_and_all_outputs() {
    let plan = MatchedCalibrationDifferencingPlan::new(1, 1).unwrap();
    let other_plan = MatchedCalibrationDifferencingPlan::new(2, 1).unwrap();
    let mut workspace = plan.workspace();
    let mut output_measurement = [0.0_f32];
    let mut output_corrector = [0.0_f32];
    let mut status = MatchedCalibrationDifferencingStatus::Priming;
    plan.process(
        MatchedCalibrationDifferencingOutput::new(
            &mut output_measurement,
            &mut output_corrector,
            &mut status,
        ),
        &mut workspace,
        MatchedCalibrationDifferencingInput::new(&[f32::MAX], &[f32::MAX]),
    )
    .unwrap();

    for (input, expected) in [
        (
            MatchedCalibrationDifferencingInput::new(&[], &[0.0]),
            MatchedCalibrationDifferencingError::InputExtent {
                input: MatchedCalibrationDifferencingInputKind::Measurements,
                expected: 1,
                actual: 0,
            },
        ),
        (
            MatchedCalibrationDifferencingInput::new(&[0.0], &[]),
            MatchedCalibrationDifferencingError::InputExtent {
                input: MatchedCalibrationDifferencingInputKind::CorrectorSample,
                expected: 1,
                actual: 0,
            },
        ),
        (
            MatchedCalibrationDifferencingInput::new(&[f32::NAN], &[0.0]),
            MatchedCalibrationDifferencingError::NonFiniteInput {
                input: MatchedCalibrationDifferencingInputKind::Measurements,
                index: 0,
            },
        ),
        (
            MatchedCalibrationDifferencingInput::new(&[0.0], &[f32::INFINITY]),
            MatchedCalibrationDifferencingError::NonFiniteInput {
                input: MatchedCalibrationDifferencingInputKind::CorrectorSample,
                index: 0,
            },
        ),
        (
            MatchedCalibrationDifferencingInput::new(&[-f32::MAX], &[-f32::MAX]),
            MatchedCalibrationDifferencingError::NonFiniteCalculation,
        ),
    ] {
        let previous_measurements = workspace.previous_measurements().to_vec();
        let previous_corrector = workspace.previous_corrector_sample().to_vec();
        output_measurement = [7.0];
        output_corrector = [8.0];
        status = MatchedCalibrationDifferencingStatus::Filtered;
        assert_eq!(
            plan.process(
                MatchedCalibrationDifferencingOutput::new(
                    &mut output_measurement,
                    &mut output_corrector,
                    &mut status,
                ),
                &mut workspace,
                input,
            ),
            Err(expected)
        );
        assert_eq!(workspace.previous_measurements(), previous_measurements);
        assert_eq!(workspace.previous_corrector_sample(), previous_corrector);
        assert!(workspace.is_primed());
        assert_eq!(output_measurement, [7.0]);
        assert_eq!(output_corrector, [8.0]);
        assert_eq!(status, MatchedCalibrationDifferencingStatus::Filtered);
    }

    let mut wrong_workspace = other_plan.workspace();
    assert_eq!(
        plan.process(
            MatchedCalibrationDifferencingOutput::new(
                &mut output_measurement,
                &mut output_corrector,
                &mut status,
            ),
            &mut wrong_workspace,
            MatchedCalibrationDifferencingInput::new(&[0.0], &[0.0]),
        ),
        Err(MatchedCalibrationDifferencingError::WorkspaceExtent)
    );
}

#[test]
fn rejected_output_extent_does_not_prime_history() {
    let plan = MatchedCalibrationDifferencingPlan::new(1, 1).unwrap();
    let mut workspace = plan.workspace();
    let mut empty = [];
    let mut corrector = [4.0_f32];
    let mut status = MatchedCalibrationDifferencingStatus::Filtered;

    assert_eq!(
        plan.process(
            MatchedCalibrationDifferencingOutput::new(&mut empty, &mut corrector, &mut status,),
            &mut workspace,
            MatchedCalibrationDifferencingInput::new(&[1.0], &[2.0]),
        ),
        Err(MatchedCalibrationDifferencingError::OutputExtent {
            output: MatchedCalibrationDifferencingOutputKind::FilteredMeasurements,
            expected: 1,
            actual: 0,
        })
    );
    assert!(!workspace.is_primed());
    assert_eq!(corrector, [4.0]);
    assert_eq!(status, MatchedCalibrationDifferencingStatus::Filtered);

    let mut measurement = [5.0_f32];
    assert_eq!(
        plan.process(
            MatchedCalibrationDifferencingOutput::new(&mut measurement, &mut empty, &mut status,),
            &mut workspace,
            MatchedCalibrationDifferencingInput::new(&[1.0], &[2.0]),
        ),
        Err(MatchedCalibrationDifferencingError::OutputExtent {
            output: MatchedCalibrationDifferencingOutputKind::FilteredCorrectorSample,
            expected: 1,
            actual: 0,
        })
    );
    assert!(!workspace.is_primed());
    assert_eq!(measurement, [5.0]);
    assert_eq!(status, MatchedCalibrationDifferencingStatus::Filtered);
}

#[test]
fn preparation_rejects_empty_dimensions() {
    assert_eq!(
        MatchedCalibrationDifferencingPlan::<f32>::new(0, 1),
        Err(MatchedCalibrationDifferencingPlanError::EmptyDimension)
    );
    assert_eq!(
        MatchedCalibrationDifferencingPlan::<f32>::new(1, 0),
        Err(MatchedCalibrationDifferencingPlanError::EmptyDimension)
    );
}

#[test]
fn filtered_pairs_feed_covariance_without_priming_output() {
    let filter = MatchedCalibrationDifferencingPlan::new(1, 1).unwrap();
    let covariance = DocrimeCovariancePlan::with_sample_limit(1, 1, 2).unwrap();
    let mut filter_workspace = filter.workspace();
    let mut covariance_workspace = covariance.workspace();
    let mut measurement = [0.0_f32];
    let mut corrector = [0.0_f32];
    let mut status = MatchedCalibrationDifferencingStatus::Priming;

    for (raw_measurement, raw_corrector) in [(10.0, 1.0), (14.0, 3.0), (11.0, 0.0)] {
        filter
            .process(
                MatchedCalibrationDifferencingOutput::new(
                    &mut measurement,
                    &mut corrector,
                    &mut status,
                ),
                &mut filter_workspace,
                MatchedCalibrationDifferencingInput::new(&[raw_measurement], &[raw_corrector]),
            )
            .unwrap();
        if status == MatchedCalibrationDifferencingStatus::Filtered {
            covariance
                .process(
                    (),
                    &mut covariance_workspace,
                    DocrimeCovarianceInput::new(&measurement, &corrector),
                )
                .unwrap();
        }
    }

    assert_eq!(covariance_workspace.sample_count(), 2);
    assert_eq!(covariance_workspace.measurement_mean(), [0.5]);
    assert_eq!(covariance_workspace.corrector_sample_mean(), [-0.5]);
    assert_eq!(
        covariance_workspace.measurement_corrector_cross_covariance_sum(),
        [17.5]
    );
    assert_eq!(covariance_workspace.corrector_covariance_sum(), [12.5]);
}

#[test]
fn declaration_preserves_both_generic_domains_and_resets_pair_history() {
    let measurements_schema = "test.filtergraph/test-measurements/1";
    let corrector_schema = "test.filtergraph/test-corrector-sample/1";
    let filtered_measurements_schema = "test.filtergraph/test-filtered-measurements/1";
    let filtered_corrector_schema = "test.filtergraph/test-filtered-corrector-sample/1";
    let mut declared = MatchedCalibrationDifferencingF32::prepare(
        MatchedCalibrationDifferencingF32Configuration {
            measurement_count: 2,
            measurements_schema: measurements_schema.into(),
            corrector_sample_schema: corrector_schema.into(),
            filtered_measurements_schema: filtered_measurements_schema.into(),
            filtered_corrector_sample_schema: filtered_corrector_schema.into(),
            corrector_coordinate_count: 1,
        },
    )
    .unwrap();

    assert_eq!(
        MatchedCalibrationDifferencingF32::NAME,
        "matched-calibration-differencing-f32"
    );
    let formats = declared.formats();
    assert_eq!(formats.len(), 5);
    assert_eq!(formats[0].schema(), measurements_schema);
    assert_eq!(formats[1].schema(), corrector_schema);
    assert_eq!(formats[2].schema(), filtered_measurements_schema);
    assert_eq!(formats[3].schema(), filtered_corrector_schema);
    assert_eq!(formats[4].element_type(), ElementType::Bool8);
    assert_eq!(formats[4].schema(), MATCHED_CALIBRATION_FILTERED_V1);

    let mut filtered_measurements = [9.0_f32; 2];
    let mut filtered_corrector = [8.0_f32];
    let mut filtered = [true];
    MatchedCalibrationDifferencingF32::process(
        &mut declared,
        &[10.0, 20.0],
        &[1.0],
        &mut filtered_measurements,
        &mut filtered_corrector,
        &mut filtered,
    )
    .unwrap();
    assert_eq!(filtered_measurements, [0.0, 0.0]);
    assert_eq!(filtered_corrector, [0.0]);
    assert_eq!(filtered, [false]);

    MatchedCalibrationDifferencingF32::process(
        &mut declared,
        &[13.0, 14.0],
        &[0.5],
        &mut filtered_measurements,
        &mut filtered_corrector,
        &mut filtered,
    )
    .unwrap();
    assert_eq!(filtered_measurements, [3.0, -6.0]);
    assert_eq!(filtered_corrector, [-0.5]);
    assert_eq!(filtered, [true]);

    MatchedCalibrationDifferencingF32::reset(&mut declared);
    MatchedCalibrationDifferencingF32::process(
        &mut declared,
        &[13.0, 14.0],
        &[0.5],
        &mut filtered_measurements,
        &mut filtered_corrector,
        &mut filtered,
    )
    .unwrap();
    assert_eq!(filtered_measurements, [0.0, 0.0]);
    assert_eq!(filtered_corrector, [0.0]);
    assert_eq!(filtered, [false]);
}
