//! DO-CRIME covariance-accumulation tests.

use filtergraph_algorithms::schemas::{
    DOCRIME_COVARIANCE_COMPLETE_V1, DOCRIME_COVARIANCE_SAMPLE_COUNT_V1,
};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmPlan, DocrimeCovarianceError, DocrimeCovarianceF32,
    DocrimeCovarianceF32Configuration, DocrimeCovarianceInput, DocrimeCovarianceInputKind,
    DocrimeCovarianceMatrix, DocrimeCovariancePlan, DocrimeCovariancePlanError,
    DocrimeCovarianceWorkspace,
};

fn assert_committed_history_eq<T>(
    actual: &DocrimeCovarianceWorkspace<T>,
    expected: &DocrimeCovarianceWorkspace<T>,
) where
    T: PartialEq + core::fmt::Debug,
{
    assert_eq!(actual.sample_count(), expected.sample_count());
    assert_eq!(actual.measurement_mean(), expected.measurement_mean());
    assert_eq!(
        actual.corrector_sample_mean(),
        expected.corrector_sample_mean()
    );
    assert_eq!(
        actual.measurement_corrector_cross_covariance_sum(),
        expected.measurement_corrector_cross_covariance_sum()
    );
    assert_eq!(
        actual.corrector_covariance_sum(),
        expected.corrector_covariance_sum()
    );
}

#[test]
fn centered_online_update_matches_hand_calculated_covariance_sums() {
    let plan = DocrimeCovariancePlan::new(2, 2).unwrap();
    let mut workspace = plan.workspace();
    for (measurements, corrector_sample) in [
        ([10.0, 4.5], [1.0, 2.0]),
        ([16.0, -0.5], [3.0, 0.0]),
        ([16.0, 12.5], [5.0, 4.0]),
    ] {
        plan.process(
            (),
            &mut workspace,
            DocrimeCovarianceInput::new(&measurements, &corrector_sample),
        )
        .unwrap();
    }

    assert_eq!(workspace.sample_count(), 3);
    assert_eq!(workspace.measurement_mean(), [14.0, 5.5]);
    assert_eq!(workspace.corrector_sample_mean(), [3.0, 2.0]);
    assert_eq!(
        workspace.measurement_corrector_cross_covariance_sum(),
        [12.0, 0.0, 16.0, 26.0]
    );
    assert_eq!(workspace.corrector_covariance_sum(), [8.0, 4.0, 4.0, 8.0]);

    let covariances = workspace.covariances();
    assert_eq!(covariances.sample_count(), 3);
    assert_eq!(
        covariances.measurement_corrector_cross_covariance_sum(),
        [12.0, 0.0, 16.0, 26.0]
    );
    assert_eq!(covariances.corrector_covariance_sum(), [8.0, 4.0, 4.0, 8.0]);
}

#[test]
fn declaration_publishes_committed_sums_count_completion_and_in_place_reset() {
    let measurement_schema = "test.filtergraph/test-measurement/1";
    let corrector_schema = "test.filtergraph/test-corrector-sample/1";
    let cross_schema = "test.filtergraph/test-measurement-corrector-covariance/1";
    let corrector_covariance_schema = "test.filtergraph/test-corrector-covariance/1";
    let mut declared = DocrimeCovarianceF32::prepare(DocrimeCovarianceF32Configuration {
        measurement_count: 2,
        sample_limit: Some(2),
        measurements_schema: measurement_schema.into(),
        corrector_sample_schema: corrector_schema.into(),
        cross_covariance_schema: cross_schema.into(),
        corrector_covariance_schema: corrector_covariance_schema.into(),
        corrector_coordinate_count: 2,
    })
    .unwrap();

    assert_eq!(DocrimeCovarianceF32::NAME, "docrime-covariance-f32");
    let formats = declared.formats();
    assert_eq!(formats.len(), 6);
    assert_eq!(formats[0].shape(), &[2]);
    assert_eq!(formats[0].schema(), measurement_schema);
    assert_eq!(formats[1].shape(), &[2]);
    assert_eq!(formats[1].schema(), corrector_schema);
    assert_eq!(formats[2].shape(), &[2, 2]);
    assert_eq!(formats[2].schema(), cross_schema);
    assert_eq!(formats[3].shape(), &[2, 2]);
    assert_eq!(formats[3].schema(), corrector_covariance_schema);
    assert_eq!(formats[4].schema(), DOCRIME_COVARIANCE_SAMPLE_COUNT_V1);
    assert_eq!(formats[5].schema(), DOCRIME_COVARIANCE_COMPLETE_V1);

    let first_measurement = [10.0_f32, 4.5];
    let first_corrector = [1.0_f32, 2.0];
    let mut cross_covariance = [f32::NAN; 4];
    let mut corrector_covariance = [f32::NAN; 4];
    let mut sample_count = [u64::MAX];
    let mut complete = [true];
    DocrimeCovarianceF32::process(
        &mut declared,
        &first_measurement,
        &first_corrector,
        &mut cross_covariance,
        &mut corrector_covariance,
        &mut sample_count,
        &mut complete,
    )
    .unwrap();
    assert_eq!(cross_covariance, [0.0; 4]);
    assert_eq!(corrector_covariance, [0.0; 4]);
    assert_eq!(sample_count, [1]);
    assert_eq!(complete, [false]);

    DocrimeCovarianceF32::process(
        &mut declared,
        &[16.0, -0.5],
        &[3.0, 0.0],
        &mut cross_covariance,
        &mut corrector_covariance,
        &mut sample_count,
        &mut complete,
    )
    .unwrap();
    assert_eq!(cross_covariance, [6.0, -6.0, -5.0, 5.0]);
    assert_eq!(corrector_covariance, [2.0, -2.0, -2.0, 2.0]);
    assert_eq!(sample_count, [2]);
    assert_eq!(complete, [true]);

    DocrimeCovarianceF32::reset(&mut declared);
    DocrimeCovarianceF32::process(
        &mut declared,
        &first_measurement,
        &first_corrector,
        &mut cross_covariance,
        &mut corrector_covariance,
        &mut sample_count,
        &mut complete,
    )
    .unwrap();
    assert_eq!(cross_covariance, [0.0; 4]);
    assert_eq!(corrector_covariance, [0.0; 4]);
    assert_eq!(sample_count, [1]);
    assert_eq!(complete, [false]);
}

#[test]
fn constant_measurement_offsets_do_not_change_centered_covariances() {
    let plan = DocrimeCovariancePlan::new(1, 2).unwrap();
    let mut first = plan.workspace();
    let mut shifted = plan.workspace();
    for (measurement, corrector_sample) in
        [(2.0, [1.0, -1.0]), (8.0, [2.0, 1.0]), (7.0, [4.0, -2.0])]
    {
        plan.process(
            (),
            &mut first,
            DocrimeCovarianceInput::new(&[measurement], &corrector_sample),
        )
        .unwrap();
        plan.process(
            (),
            &mut shifted,
            DocrimeCovarianceInput::new(&[measurement + 100.0], &corrector_sample),
        )
        .unwrap();
    }

    assert_eq!(
        first.measurement_corrector_cross_covariance_sum(),
        shifted.measurement_corrector_cross_covariance_sum()
    );
    assert_eq!(
        first.corrector_covariance_sum(),
        shifted.corrector_covariance_sum()
    );
}

#[test]
fn preparation_rejects_empty_and_overflowing_dimensions() {
    assert_eq!(
        DocrimeCovariancePlan::<f32>::new(0, 1),
        Err(DocrimeCovariancePlanError::EmptyDimension)
    );
    assert_eq!(
        DocrimeCovariancePlan::<f32>::new(1, 0),
        Err(DocrimeCovariancePlanError::EmptyDimension)
    );
    assert_eq!(
        DocrimeCovariancePlan::<f32>::new(usize::MAX, 2),
        Err(DocrimeCovariancePlanError::ExtentOverflow {
            matrix: DocrimeCovarianceMatrix::MeasurementCorrectorCross,
        })
    );
    assert_eq!(
        DocrimeCovariancePlan::<f32>::new(1, usize::MAX),
        Err(DocrimeCovariancePlanError::ExtentOverflow {
            matrix: DocrimeCovarianceMatrix::Corrector,
        })
    );
    assert_eq!(
        DocrimeCovariancePlan::<f32>::with_sample_limit(1, 1, 0),
        Err(DocrimeCovariancePlanError::ZeroSampleLimit)
    );
}

#[test]
fn finite_window_stops_at_the_exact_selected_sample_count() {
    let plan = DocrimeCovariancePlan::with_sample_limit(1, 1, 2).unwrap();
    let mut workspace = plan.workspace();
    assert_eq!(plan.sample_limit(), Some(2));

    for (measurement, corrector) in [(1.0_f32, -1.0), (3.0, 1.0)] {
        plan.process(
            (),
            &mut workspace,
            DocrimeCovarianceInput::new(&[measurement], &[corrector]),
        )
        .unwrap();
    }
    let completed = workspace.clone();
    assert_eq!(workspace.sample_count(), 2);
    assert_eq!(
        plan.process(
            (),
            &mut workspace,
            DocrimeCovarianceInput::new(&[f32::NAN], &[f32::INFINITY]),
        ),
        Err(DocrimeCovarianceError::SampleLimitReached { sample_limit: 2 })
    );
    assert_committed_history_eq(&workspace, &completed);

    let plan64 = DocrimeCovariancePlan::with_sample_limit(1, 1, 1).unwrap();
    let mut workspace64 = plan64.workspace();
    plan64
        .process(
            (),
            &mut workspace64,
            DocrimeCovarianceInput::new(&[1.0_f64], &[-1.0_f64]),
        )
        .unwrap();
    assert_eq!(workspace64.sample_count(), 1);
    assert_eq!(
        plan64.process(
            (),
            &mut workspace64,
            DocrimeCovarianceInput::new(&[2.0_f64], &[2.0_f64]),
        ),
        Err(DocrimeCovarianceError::SampleLimitReached { sample_limit: 1 })
    );
}

#[test]
fn rejected_samples_preserve_every_committed_history_array() {
    let plan = DocrimeCovariancePlan::new(1, 1).unwrap();
    let other_plan = DocrimeCovariancePlan::new(2, 1).unwrap();
    let mut workspace = plan.workspace();
    plan.process(
        (),
        &mut workspace,
        DocrimeCovarianceInput::new(&[f32::MAX], &[f32::MAX]),
    )
    .unwrap();
    let committed = workspace.clone();

    for (input, expected) in [
        (
            DocrimeCovarianceInput::new(&[], &[0.0]),
            DocrimeCovarianceError::InputExtent {
                input: DocrimeCovarianceInputKind::Measurements,
                expected: 1,
                actual: 0,
            },
        ),
        (
            DocrimeCovarianceInput::new(&[0.0], &[]),
            DocrimeCovarianceError::InputExtent {
                input: DocrimeCovarianceInputKind::CorrectorSample,
                expected: 1,
                actual: 0,
            },
        ),
        (
            DocrimeCovarianceInput::new(&[f32::NAN], &[0.0]),
            DocrimeCovarianceError::NonFiniteInput {
                input: DocrimeCovarianceInputKind::Measurements,
                index: 0,
            },
        ),
        (
            DocrimeCovarianceInput::new(&[0.0], &[f32::INFINITY]),
            DocrimeCovarianceError::NonFiniteInput {
                input: DocrimeCovarianceInputKind::CorrectorSample,
                index: 0,
            },
        ),
        (
            DocrimeCovarianceInput::new(&[-f32::MAX], &[-f32::MAX]),
            DocrimeCovarianceError::NonFiniteCalculation,
        ),
    ] {
        assert_eq!(plan.process((), &mut workspace, input), Err(expected));
        assert_committed_history_eq(&workspace, &committed);
    }

    let mut wrong_workspace = other_plan.workspace();
    assert_eq!(
        plan.process(
            (),
            &mut wrong_workspace,
            DocrimeCovarianceInput::new(&[0.0], &[0.0]),
        ),
        Err(DocrimeCovarianceError::WorkspaceExtent)
    );
    assert_committed_history_eq(&wrong_workspace, &other_plan.workspace());
}

macro_rules! covariance_matches_scalar_update {
    ($test:ident, $scalar:ty) => {
        #[test]
        fn $test() {
            // Include both sides of the mirror-tile boundary, an incomplete
            // tile, and rectangular cross-covariance matrices.
            for (measurement_count, corrector_count) in [
                (1, 1),
                (7, 15),
                (19, 16),
                (9, 17),
                (5, 31),
                (33, 33),
                (37, 65),
                (277, 277),
            ] {
                let plan = DocrimeCovariancePlan::<$scalar>::with_sample_limit(
                    measurement_count,
                    corrector_count,
                    8,
                )
                .unwrap();
                let mut workspace = plan.workspace();
                let mut measurement_mean = vec![0.0 as $scalar; measurement_count];
                let mut corrector_mean = vec![0.0 as $scalar; corrector_count];
                let mut cross = vec![0.0 as $scalar; measurement_count * corrector_count];
                let mut covariance = vec![0.0 as $scalar; corrector_count * corrector_count];
                for frame in 0..8 {
                    let measurements: Vec<$scalar> = (0..measurement_count)
                        .map(|i| ((i * 19 + frame * 7) % 113) as $scalar / 29.0 - 1.75)
                        .collect();
                    let correctors: Vec<$scalar> = (0..corrector_count)
                        .map(|i| ((i * 13 + frame * 23) % 97) as $scalar / 31.0 - 1.125)
                        .collect();
                    let count = (frame + 1) as $scalar;
                    let weight = (count - 1.0) / count;
                    let measurement_delta: Vec<$scalar> = measurements
                        .iter()
                        .zip(&measurement_mean)
                        .map(|(sample, mean)| *sample - *mean)
                        .collect();
                    let corrector_delta: Vec<$scalar> = correctors
                        .iter()
                        .zip(&corrector_mean)
                        .map(|(sample, mean)| *sample - *mean)
                        .collect();
                    for (mean, delta) in measurement_mean.iter_mut().zip(&measurement_delta) {
                        *mean = *mean + *delta / count;
                    }
                    for (mean, delta) in corrector_mean.iter_mut().zip(&corrector_delta) {
                        *mean = *mean + *delta / count;
                    }
                    // Preserve the original scalar traversal and mathematical
                    // orientation independently of the production loop form.
                    for row in 0..measurement_count {
                        for column in 0..corrector_count {
                            let index = row * corrector_count + column;
                            cross[index] = (weight * measurement_delta[row])
                                .mul_add(corrector_delta[column], cross[index]);
                        }
                    }
                    for row in 0..corrector_count {
                        for column in row..corrector_count {
                            let index = row * corrector_count + column;
                            let candidate = (weight * corrector_delta[row])
                                .mul_add(corrector_delta[column], covariance[index]);
                            covariance[index] = candidate;
                            covariance[column * corrector_count + row] = candidate;
                        }
                    }
                    plan.process(
                        (),
                        &mut workspace,
                        DocrimeCovarianceInput::new(&measurements, &correctors),
                    )
                    .unwrap();
                    assert_eq!(workspace.sample_count(), frame + 1);
                    for (actual, expected) in [
                        (workspace.measurement_mean(), measurement_mean.as_slice()),
                        (workspace.corrector_sample_mean(), corrector_mean.as_slice()),
                        (
                            workspace.measurement_corrector_cross_covariance_sum(),
                            cross.as_slice(),
                        ),
                        (workspace.corrector_covariance_sum(), covariance.as_slice()),
                    ] {
                        assert!(
                            actual
                                .iter()
                                .zip(expected)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "{measurement_count}×{corrector_count}, frame {frame}"
                        );
                    }
                }
            }
        }
    };
}

covariance_matches_scalar_update!(tiled_covariance_matches_scalar_update_f32, f32);
covariance_matches_scalar_update!(tiled_covariance_matches_scalar_update_f64, f64);

#[test]
fn rejection_after_partial_tiles_preserves_history_and_allows_recovery() {
    let plan = DocrimeCovariancePlan::<f32>::new(19, 33).unwrap();
    let mut workspace = plan.workspace();
    let measurements = [1.0; 19];
    let correctors = [1.0; 33];
    plan.process(
        (),
        &mut workspace,
        DocrimeCovarianceInput::new(&measurements, &correctors),
    )
    .unwrap();
    let committed = workspace.clone();

    // The last diagonal overflows after earlier upper rows and mirror tiles
    // have been staged. A rejected candidate must never become history.
    let mut overflowing = correctors;
    overflowing[32] = f32::MAX;
    assert_eq!(
        plan.process(
            (),
            &mut workspace,
            DocrimeCovarianceInput::new(&measurements, &overflowing),
        ),
        Err(DocrimeCovarianceError::NonFiniteCalculation)
    );
    assert_committed_history_eq(&workspace, &committed);

    let mut expected = committed;
    let next_measurements = [2.0; 19];
    let next_correctors = [-2.0; 33];
    for state in [&mut workspace, &mut expected] {
        plan.process(
            (),
            state,
            DocrimeCovarianceInput::new(&next_measurements, &next_correctors),
        )
        .unwrap();
    }
    assert_committed_history_eq(&workspace, &expected);
    workspace.reset();
    assert_eq!(workspace, plan.workspace());
}

#[test]
fn rejected_declaration_leaves_outputs_unpublished_and_keeps_input_error_order() {
    let mut declared = DocrimeCovarianceF32::prepare(DocrimeCovarianceF32Configuration {
        measurement_count: 17,
        corrector_coordinate_count: 33,
        sample_limit: None,
        measurements_schema: "test/measurements/1".into(),
        corrector_sample_schema: "test/corrector/1".into(),
        cross_covariance_schema: "test/cross/1".into(),
        corrector_covariance_schema: "test/covariance/1".into(),
    })
    .unwrap();
    let mut cross = [91.0; 17 * 33];
    let mut covariance = [92.0; 33 * 33];
    let mut count = [93];
    let mut complete = [true];
    let mut measurements = [0.0; 17];
    let mut correctors = [0.0; 33];
    DocrimeCovarianceF32::process(
        &mut declared,
        &measurements,
        &correctors,
        &mut cross,
        &mut covariance,
        &mut count,
        &mut complete,
    )
    .unwrap();
    cross.fill(91.0);
    covariance.fill(92.0);
    count[0] = 93;
    complete[0] = true;
    measurements[16] = f32::NAN;
    correctors[0] = f32::INFINITY;
    assert_eq!(
        DocrimeCovarianceF32::process(
            &mut declared,
            &measurements,
            &correctors,
            &mut cross,
            &mut covariance,
            &mut count,
            &mut complete,
        ),
        Err(filtergraph_algorithms::DeclarationProcessError::Algorithm(
            DocrimeCovarianceError::NonFiniteInput {
                input: DocrimeCovarianceInputKind::Measurements,
                index: 16,
            },
        ))
    );
    measurements.fill(0.0);
    correctors.fill(0.0);
    correctors[32] = f32::MAX;
    assert_eq!(
        DocrimeCovarianceF32::process(
            &mut declared,
            &measurements,
            &correctors,
            &mut cross,
            &mut covariance,
            &mut count,
            &mut complete,
        ),
        Err(filtergraph_algorithms::DeclarationProcessError::Algorithm(
            DocrimeCovarianceError::NonFiniteCalculation,
        ))
    );
    assert_eq!(cross, [91.0; 17 * 33]);
    assert_eq!(covariance, [92.0; 33 * 33]);
    assert_eq!(count, [93]);
    assert_eq!(complete, [true]);
}
