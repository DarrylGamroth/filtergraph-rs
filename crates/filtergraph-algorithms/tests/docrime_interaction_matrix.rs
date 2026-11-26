//! Conditioned DO-CRIME interaction-matrix tests.

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmPlan, AlgorithmProperties,
    DocrimeClosedLoopInteractionMatrixF32, DocrimeClosedLoopInteractionMatrixF32Configuration,
    DocrimeClosedLoopInteractionMatrixPlan, DocrimeConditioningCriteria,
    DocrimeConditioningCriteriaError, DocrimeCovariances, DocrimeInteractionMatrixDiagnostics,
    DocrimeInteractionMatrixError, DocrimeInteractionMatrixInputKind, DocrimeInteractionMatrixKind,
    DocrimeInteractionMatrixOutput, DocrimeInteractionMatrixPlanError,
    DocrimeInteractionMatrixStage, DocrimeInteractionMatrixStatus,
    DocrimeOpenLoopInteractionMatrixF32, DocrimeOpenLoopInteractionMatrixF32Configuration,
    DocrimeOpenLoopInteractionMatrixPlan, PropertyAssignment, PropertyValue,
};
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
struct SharedFixture {
    schema: String,
    sample_count: usize,
    measurement_count: usize,
    corrector_coordinate_count: usize,
    measurement_corrector_cross_covariance_sum: Vec<f64>,
    corrector_covariance_sum: Vec<f64>,
    expected_open_loop_interaction_matrix: Vec<f64>,
    closed_loop_gain: f64,
    expected_closed_loop_interaction_matrix: Vec<f64>,
    absolute_tolerance_f64: f64,
    absolute_tolerance_f32: f32,
    conditioning_criteria: SharedCriteria,
    expected_diagnostics: SharedDiagnostics,
}

#[derive(Debug, Deserialize)]
struct SharedCriteria {
    minimum_corrector_variance: f64,
    maximum_absolute_corrector_correlation: f64,
    minimum_factorization_pivot_ratio: f64,
}

#[derive(Debug, Deserialize)]
struct SharedDiagnostics {
    minimum_corrector_variance: f64,
    maximum_corrector_variance: f64,
    maximum_absolute_corrector_correlation: f64,
    minimum_factorization_pivot_ratio: f64,
}

fn criteria64() -> DocrimeConditioningCriteria<f64> {
    DocrimeConditioningCriteria::new(0.1, 0.9, 0.01).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_close_with_tolerance(actual, expected, 1.0e-12);
}

fn assert_close_with_tolerance(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} != {expected}"
        );
    }
}

fn shared_fixture() -> SharedFixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/algorithms/docrime_interaction_matrix.toml"
    );
    let source = fs::read_to_string(path).expect("read shared DO-CRIME fixture");
    toml::from_str(&source).expect("parse shared DO-CRIME fixture")
}

#[test]
fn julia_and_rust_share_the_docrime_equation_fixture() {
    let fixture = shared_fixture();
    assert_eq!(
        fixture.schema,
        "org.calculon.algorithm-fixture/docrime-interaction-matrix/1"
    );
    assert_eq!(
        fixture.measurement_corrector_cross_covariance_sum.len(),
        fixture.measurement_count * fixture.corrector_coordinate_count
    );
    assert_eq!(
        fixture.corrector_covariance_sum.len(),
        fixture.corrector_coordinate_count * fixture.corrector_coordinate_count
    );
    let criteria = DocrimeConditioningCriteria::new(
        fixture.conditioning_criteria.minimum_corrector_variance,
        fixture
            .conditioning_criteria
            .maximum_absolute_corrector_correlation,
        fixture
            .conditioning_criteria
            .minimum_factorization_pivot_ratio,
    )
    .unwrap();
    let covariances = DocrimeCovariances::new(
        fixture.sample_count,
        &fixture.measurement_corrector_cross_covariance_sum,
        &fixture.corrector_covariance_sum,
    );

    let open = DocrimeOpenLoopInteractionMatrixPlan::new(
        fixture.measurement_count,
        fixture.corrector_coordinate_count,
        criteria,
    )
    .unwrap();
    let mut open_workspace = open.workspace();
    let mut open_matrix = vec![f64::NAN; fixture.expected_open_loop_interaction_matrix.len()];
    let mut open_diagnostics = DocrimeInteractionMatrixDiagnostics::default();
    open.process(
        DocrimeInteractionMatrixOutput::new(&mut open_matrix, &mut open_diagnostics),
        &mut open_workspace,
        covariances,
    )
    .unwrap();
    assert_close_with_tolerance(
        &open_matrix,
        &fixture.expected_open_loop_interaction_matrix,
        fixture.absolute_tolerance_f64,
    );
    assert_eq!(
        open_diagnostics.status(),
        DocrimeInteractionMatrixStatus::Valid
    );
    assert!(
        (open_diagnostics.minimum_corrector_variance()
            - fixture.expected_diagnostics.minimum_corrector_variance)
            .abs()
            < fixture.absolute_tolerance_f64
    );
    assert!(
        (open_diagnostics.maximum_corrector_variance()
            - fixture.expected_diagnostics.maximum_corrector_variance)
            .abs()
            < fixture.absolute_tolerance_f64
    );
    assert!(
        (open_diagnostics.maximum_absolute_corrector_correlation()
            - fixture
                .expected_diagnostics
                .maximum_absolute_corrector_correlation)
            .abs()
            < fixture.absolute_tolerance_f64
    );
    assert!(
        (open_diagnostics.minimum_factorization_pivot_ratio()
            - fixture
                .expected_diagnostics
                .minimum_factorization_pivot_ratio)
            .abs()
            < fixture.absolute_tolerance_f64
    );

    let closed = DocrimeClosedLoopInteractionMatrixPlan::new(
        fixture.measurement_count,
        fixture.corrector_coordinate_count,
        fixture.closed_loop_gain,
        criteria,
    )
    .unwrap();
    let mut closed_workspace = closed.workspace();
    let mut closed_matrix = vec![f64::NAN; fixture.expected_closed_loop_interaction_matrix.len()];
    let mut closed_diagnostics = DocrimeInteractionMatrixDiagnostics::default();
    closed
        .process(
            DocrimeInteractionMatrixOutput::new(&mut closed_matrix, &mut closed_diagnostics),
            &mut closed_workspace,
            covariances,
        )
        .unwrap();
    assert_close_with_tolerance(
        &closed_matrix,
        &fixture.expected_closed_loop_interaction_matrix,
        fixture.absolute_tolerance_f64,
    );
    assert_eq!(
        closed_diagnostics.status(),
        DocrimeInteractionMatrixStatus::Valid
    );
}

#[test]
fn open_loop_solve_recovers_a_correlated_interaction_matrix_and_diagnostics() {
    let plan = DocrimeOpenLoopInteractionMatrixPlan::new(2, 2, criteria64()).unwrap();
    let mut workspace = plan.workspace();
    let mut interaction_matrix = [f64::NAN; 4];
    let mut diagnostics = DocrimeInteractionMatrixDiagnostics::default();

    plan.process(
        DocrimeInteractionMatrixOutput::new(&mut interaction_matrix, &mut diagnostics),
        &mut workspace,
        DocrimeCovariances::new(3, &[6.0, 0.0, 8.0, 13.0], &[4.0, 2.0, 2.0, 4.0]),
    )
    .unwrap();

    assert_close(&interaction_matrix, &[2.0, -1.0, 0.5, 3.0]);
    assert_eq!(diagnostics.status(), DocrimeInteractionMatrixStatus::Valid);
    assert!(diagnostics.status().is_valid());
    assert_eq!(diagnostics.sample_count(), 3);
    assert_eq!(diagnostics.minimum_corrector_variance(), 2.0);
    assert_eq!(diagnostics.maximum_corrector_variance(), 2.0);
    assert!((diagnostics.maximum_absolute_corrector_correlation() - 0.5).abs() < 1.0e-12);
    assert!((diagnostics.minimum_factorization_pivot_ratio() - 0.75).abs() < 1.0e-12);
}

#[test]
fn open_loop_declaration_publishes_complete_diagnostics_and_runtime_criteria() {
    let cross_schema = "test.filtergraph/test-cross-covariance/1";
    let corrector_schema = "test.filtergraph/test-corrector-covariance/1";
    let matrix_schema = "test.filtergraph/test-interaction-matrix/1";
    let mut declared = DocrimeOpenLoopInteractionMatrixF32::prepare(
        DocrimeOpenLoopInteractionMatrixF32Configuration {
            measurement_count: 2,
            corrector_coordinate_count: 2,
            minimum_corrector_variance: 0.1,
            cross_covariance_schema: cross_schema.into(),
            corrector_covariance_schema: corrector_schema.into(),
            interaction_matrix_schema: matrix_schema.into(),
        },
    )
    .unwrap();

    assert_eq!(
        DocrimeOpenLoopInteractionMatrixF32::NAME,
        "docrime-open-loop-interaction-matrix-f32"
    );
    let formats = declared.formats();
    assert_eq!(formats.len(), 10);
    assert_eq!(formats[0].shape(), &[2, 2]);
    assert_eq!(formats[0].schema(), cross_schema);
    assert_eq!(formats[1].shape(), &[2, 2]);
    assert_eq!(formats[1].schema(), corrector_schema);
    assert_eq!(formats[2].shape(), &[1]);
    assert_eq!(formats[3].shape(), &[2, 2]);
    assert_eq!(formats[3].schema(), matrix_schema);

    let maximum_correlation = declared.plan().property_info(0).unwrap();
    let minimum_pivot = declared.plan().property_info(1).unwrap();
    assert_eq!(maximum_correlation.unit, "1");
    assert_eq!(minimum_pivot.unit, "1");
    assert_eq!(declared.plan().property_info(2), None);

    let cross_covariance = [6.0_f32, 0.0, 8.0, 13.0];
    let corrector_covariance = [4.0_f32, 2.0, 2.0, 4.0];
    let sample_count = [3_u64];
    let mut interaction_matrix = [f32::NAN; 4];
    let mut status = [u8::MAX];
    let mut status_coordinates = [u64::MAX; 2];
    let mut diagnostic_sample_count = [u64::MAX];
    let mut variance_range = [f32::NAN; 2];
    let mut correlation = [f32::NAN];
    let mut pivot_ratio = [f32::NAN];
    DocrimeOpenLoopInteractionMatrixF32::process(
        &mut declared,
        &cross_covariance,
        &corrector_covariance,
        &sample_count,
        &mut interaction_matrix,
        &mut status,
        &mut status_coordinates,
        &mut diagnostic_sample_count,
        &mut variance_range,
        &mut correlation,
        &mut pivot_ratio,
    )
    .unwrap();
    for (actual, expected) in interaction_matrix.iter().zip([2.0, -1.0, 0.5, 3.0]) {
        assert!((actual - expected).abs() < 1.0e-6);
    }
    assert_eq!(status, [0]);
    assert_eq!(status_coordinates, [0, 0]);
    assert_eq!(diagnostic_sample_count, [3]);
    assert_eq!(variance_range, [2.0, 2.0]);
    assert!((correlation[0] - 0.5).abs() < 1.0e-6);
    assert!((pivot_ratio[0] - 0.75).abs() < 1.0e-6);

    declared
        .replace_properties(&[PropertyAssignment {
            id: maximum_correlation.id,
            value: PropertyValue::Float(0.4),
        }])
        .unwrap();
    DocrimeOpenLoopInteractionMatrixF32::process(
        &mut declared,
        &cross_covariance,
        &corrector_covariance,
        &sample_count,
        &mut interaction_matrix,
        &mut status,
        &mut status_coordinates,
        &mut diagnostic_sample_count,
        &mut variance_range,
        &mut correlation,
        &mut pivot_ratio,
    )
    .unwrap();
    assert_eq!(interaction_matrix, [0.0; 4]);
    assert_eq!(status, [3]);
    assert_eq!(status_coordinates, [0, 1]);
}

#[test]
fn closed_loop_solve_applies_the_prepared_loop_gain() {
    let tolerance = shared_fixture().absolute_tolerance_f32;
    let criteria = DocrimeConditioningCriteria::new(0.1_f32, 0.9, 0.01).unwrap();
    let plan = DocrimeClosedLoopInteractionMatrixPlan::new(2, 2, 0.4, criteria).unwrap();
    let mut workspace = plan.workspace();
    let mut interaction_matrix = [f32::NAN; 4];
    let mut diagnostics = DocrimeInteractionMatrixDiagnostics::default();

    plan.process(
        DocrimeInteractionMatrixOutput::new(&mut interaction_matrix, &mut diagnostics),
        &mut workspace,
        DocrimeCovariances::new(3, &[6.0_f32, 0.0, 8.0, 13.0], &[4.0_f32, 2.0, 2.0, 4.0]),
    )
    .unwrap();

    for (actual, expected) in interaction_matrix.iter().zip([0.8, -0.4, 0.2, 1.2]) {
        assert!((actual - expected).abs() < tolerance);
    }
    assert_eq!(diagnostics.status(), DocrimeInteractionMatrixStatus::Valid);
    assert_eq!(plan.loop_gain(), 0.4);
}

#[test]
fn closed_loop_declaration_applies_loop_gain_and_conditioning_properties_atomically() {
    let mut declared = DocrimeClosedLoopInteractionMatrixF32::prepare(
        DocrimeClosedLoopInteractionMatrixF32Configuration {
            measurement_count: 2,
            corrector_coordinate_count: 2,
            minimum_corrector_variance: 0.1,
            cross_covariance_schema: "test.filtergraph/test-cross-covariance/1".into(),
            corrector_covariance_schema: "test.filtergraph/test-corrector-covariance/1".into(),
            interaction_matrix_schema: "test.filtergraph/test-interaction-matrix/1".into(),
        },
    )
    .unwrap();

    assert_eq!(
        DocrimeClosedLoopInteractionMatrixF32::NAME,
        "docrime-closed-loop-interaction-matrix-f32"
    );
    assert_eq!(declared.formats().len(), 10);
    let loop_gain = declared.plan().property_info(0).unwrap();
    let maximum_correlation = declared.plan().property_info(1).unwrap();
    let minimum_pivot = declared.plan().property_info(2).unwrap();
    assert_eq!(declared.plan().property_info(3), None);
    declared
        .replace_properties(&[
            PropertyAssignment {
                id: loop_gain.id,
                value: PropertyValue::Float(0.4),
            },
            PropertyAssignment {
                id: maximum_correlation.id,
                value: PropertyValue::Float(0.8),
            },
            PropertyAssignment {
                id: minimum_pivot.id,
                value: PropertyValue::Float(0.02),
            },
        ])
        .unwrap();

    let cross_covariance = [6.0_f32, 0.0, 8.0, 13.0];
    let corrector_covariance = [4.0_f32, 2.0, 2.0, 4.0];
    let sample_count = [3_u64];
    let mut interaction_matrix = [f32::NAN; 4];
    let mut status = [u8::MAX];
    let mut status_coordinates = [u64::MAX; 2];
    let mut diagnostic_sample_count = [u64::MAX];
    let mut variance_range = [f32::NAN; 2];
    let mut correlation = [f32::NAN];
    let mut pivot_ratio = [f32::NAN];
    DocrimeClosedLoopInteractionMatrixF32::process(
        &mut declared,
        &cross_covariance,
        &corrector_covariance,
        &sample_count,
        &mut interaction_matrix,
        &mut status,
        &mut status_coordinates,
        &mut diagnostic_sample_count,
        &mut variance_range,
        &mut correlation,
        &mut pivot_ratio,
    )
    .unwrap();
    for (actual, expected) in interaction_matrix.iter().zip([0.8, -0.4, 0.2, 1.2]) {
        assert!((actual - expected).abs() < 2.0e-6);
    }
    assert_eq!(status, [0]);
    assert_eq!(status_coordinates, [0, 0]);
    assert_eq!(diagnostic_sample_count, [3]);
}

fn open_loop_status(
    corrector_coordinate_count: usize,
    sample_count: usize,
    corrector_covariance_sum: &[f64],
    criteria: DocrimeConditioningCriteria<f64>,
) -> (
    DocrimeInteractionMatrixStatus,
    Vec<f64>,
    DocrimeInteractionMatrixDiagnostics<f64>,
) {
    let plan =
        DocrimeOpenLoopInteractionMatrixPlan::new(1, corrector_coordinate_count, criteria).unwrap();
    let mut workspace = plan.workspace();
    let cross_covariance_sum = vec![0.0; corrector_coordinate_count];
    let mut interaction_matrix = vec![7.0; corrector_coordinate_count];
    let mut diagnostics = DocrimeInteractionMatrixDiagnostics::default();
    plan.process(
        DocrimeInteractionMatrixOutput::new(&mut interaction_matrix, &mut diagnostics),
        &mut workspace,
        DocrimeCovariances::new(
            sample_count,
            &cross_covariance_sum,
            corrector_covariance_sum,
        ),
    )
    .unwrap();
    (diagnostics.status(), interaction_matrix, diagnostics)
}

#[test]
fn conditioning_failures_are_expected_zero_matrix_statuses() {
    let cases = [
        (
            2,
            1,
            vec![1.0, 0.0, 0.0, 1.0],
            criteria64(),
            DocrimeInteractionMatrixStatus::InsufficientSamples,
        ),
        (
            2,
            2,
            vec![0.01, 0.0, 0.0, 1.0],
            criteria64(),
            DocrimeInteractionMatrixStatus::InsufficientCorrectorVariance {
                corrector_coordinate: 0,
            },
        ),
        (
            2,
            2,
            vec![1.0, 0.95, 0.95, 1.0],
            criteria64(),
            DocrimeInteractionMatrixStatus::ExcessiveCorrectorCorrelation {
                first: 0,
                second: 1,
            },
        ),
        (
            3,
            2,
            vec![1.0, -0.75, -0.75, -0.75, 1.0, -0.75, -0.75, -0.75, 1.0],
            criteria64(),
            DocrimeInteractionMatrixStatus::NonPositiveDefinite {
                corrector_coordinate: 2,
            },
        ),
        (
            2,
            2,
            vec![1.0, 0.99, 0.99, 1.0],
            DocrimeConditioningCriteria::new(0.1, 0.999, 0.05).unwrap(),
            DocrimeInteractionMatrixStatus::InsufficientFactorizationPivot {
                corrector_coordinate: 1,
            },
        ),
    ];

    for (extent, sample_count, covariance, criteria, expected_status) in cases {
        let (status, interaction_matrix, diagnostics) =
            open_loop_status(extent, sample_count, &covariance, criteria);
        assert_eq!(status, expected_status);
        assert!(!status.is_valid());
        assert_eq!(interaction_matrix, vec![0.0; extent]);
        assert_eq!(diagnostics.sample_count(), sample_count);
    }
}

#[test]
fn preparation_rejects_every_invalid_criterion_dimension_and_loop_gain() {
    for (minimum_variance, maximum_correlation, minimum_pivot, expected) in [
        (
            0.0,
            0.9,
            0.1,
            DocrimeConditioningCriteriaError::InvalidMinimumCorrectorVariance,
        ),
        (
            f32::NAN,
            0.9,
            0.1,
            DocrimeConditioningCriteriaError::InvalidMinimumCorrectorVariance,
        ),
        (
            0.1,
            -0.1,
            0.1,
            DocrimeConditioningCriteriaError::InvalidMaximumAbsoluteCorrectorCorrelation,
        ),
        (
            0.1,
            1.0,
            0.1,
            DocrimeConditioningCriteriaError::InvalidMaximumAbsoluteCorrectorCorrelation,
        ),
        (
            0.1,
            0.9,
            0.0,
            DocrimeConditioningCriteriaError::InvalidMinimumFactorizationPivotRatio,
        ),
        (
            0.1,
            0.9,
            1.1,
            DocrimeConditioningCriteriaError::InvalidMinimumFactorizationPivotRatio,
        ),
    ] {
        assert_eq!(
            DocrimeConditioningCriteria::new(minimum_variance, maximum_correlation, minimum_pivot,),
            Err(expected)
        );
    }

    let criteria = DocrimeConditioningCriteria::new(0.1_f32, 0.9, 0.01).unwrap();
    assert_eq!(
        DocrimeOpenLoopInteractionMatrixPlan::new(0, 1, criteria),
        Err(DocrimeInteractionMatrixPlanError::EmptyDimension)
    );
    assert_eq!(
        DocrimeOpenLoopInteractionMatrixPlan::new(1, 0, criteria),
        Err(DocrimeInteractionMatrixPlanError::EmptyDimension)
    );
    assert_eq!(
        DocrimeOpenLoopInteractionMatrixPlan::new(usize::MAX, 2, criteria),
        Err(DocrimeInteractionMatrixPlanError::ExtentOverflow {
            matrix: DocrimeInteractionMatrixKind::MeasurementByCorrector,
        })
    );
    assert_eq!(
        DocrimeOpenLoopInteractionMatrixPlan::new(1, usize::MAX, criteria),
        Err(DocrimeInteractionMatrixPlanError::ExtentOverflow {
            matrix: DocrimeInteractionMatrixKind::CorrectorByCorrector,
        })
    );
    for loop_gain in [0.0, -0.1, f32::NAN, f32::INFINITY] {
        assert_eq!(
            DocrimeClosedLoopInteractionMatrixPlan::new(1, 1, loop_gain, criteria),
            Err(DocrimeInteractionMatrixPlanError::InvalidLoopGain)
        );
    }
}

#[test]
fn malformed_inputs_and_calculations_preserve_both_outputs() {
    let plan = DocrimeOpenLoopInteractionMatrixPlan::new(1, 2, criteria64()).unwrap();
    let other_plan = DocrimeOpenLoopInteractionMatrixPlan::new(2, 2, criteria64()).unwrap();
    let mut workspace = plan.workspace();
    let mut other_workspace = other_plan.workspace();
    let mut interaction_matrix = [7.0, 8.0];
    let mut diagnostics = DocrimeInteractionMatrixDiagnostics::default();
    let unchanged_diagnostics = diagnostics;

    let short_cross = [1.0];
    let cross = [1.0, 0.0];
    let short_corrector = [1.0, 0.0, 0.0];
    let corrector = [1.0, 0.0, 0.0, 1.0];
    let nonfinite_cross = [f64::NAN, 0.0];
    let nonfinite_corrector = [1.0, 0.0, 0.0, f64::INFINITY];
    let asymmetric_corrector = [1.0, 0.5, 0.25, 1.0];
    let cases = [
        (
            DocrimeCovariances::new(2, &short_cross, &corrector),
            DocrimeInteractionMatrixError::InputExtent {
                input: DocrimeInteractionMatrixInputKind::MeasurementCorrectorCrossCovarianceSum,
                expected: 2,
                actual: 1,
            },
        ),
        (
            DocrimeCovariances::new(2, &cross, &short_corrector),
            DocrimeInteractionMatrixError::InputExtent {
                input: DocrimeInteractionMatrixInputKind::CorrectorCovarianceSum,
                expected: 4,
                actual: 3,
            },
        ),
        (
            DocrimeCovariances::new(2, &nonfinite_cross, &corrector),
            DocrimeInteractionMatrixError::NonFiniteInput {
                input: DocrimeInteractionMatrixInputKind::MeasurementCorrectorCrossCovarianceSum,
                index: 0,
            },
        ),
        (
            DocrimeCovariances::new(2, &cross, &nonfinite_corrector),
            DocrimeInteractionMatrixError::NonFiniteInput {
                input: DocrimeInteractionMatrixInputKind::CorrectorCovarianceSum,
                index: 3,
            },
        ),
        (
            DocrimeCovariances::new(2, &cross, &asymmetric_corrector),
            DocrimeInteractionMatrixError::NonSymmetricCorrectorCovariance { row: 0, column: 1 },
        ),
    ];

    for (covariances, expected) in cases {
        assert_eq!(
            plan.process(
                DocrimeInteractionMatrixOutput::new(&mut interaction_matrix, &mut diagnostics,),
                &mut workspace,
                covariances,
            ),
            Err(expected)
        );
        assert_eq!(interaction_matrix, [7.0, 8.0]);
        assert_eq!(diagnostics, unchanged_diagnostics);
    }

    assert_eq!(
        plan.process(
            DocrimeInteractionMatrixOutput::new(&mut interaction_matrix, &mut diagnostics),
            &mut other_workspace,
            DocrimeCovariances::new(2, &cross, &corrector),
        ),
        Err(DocrimeInteractionMatrixError::WorkspaceExtent)
    );
    assert_eq!(interaction_matrix, [7.0, 8.0]);
    assert_eq!(diagnostics, unchanged_diagnostics);

    let mut short_output = [9.0];
    assert_eq!(
        plan.process(
            DocrimeInteractionMatrixOutput::new(&mut short_output, &mut diagnostics),
            &mut workspace,
            DocrimeCovariances::new(2, &cross, &corrector),
        ),
        Err(DocrimeInteractionMatrixError::OutputExtent {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(short_output, [9.0]);
    assert_eq!(diagnostics, unchanged_diagnostics);
}

#[test]
fn nonfinite_closed_loop_solve_preserves_both_outputs() {
    let criteria = DocrimeConditioningCriteria::new(0.1_f32, 0.9, 0.01).unwrap();
    let plan = DocrimeClosedLoopInteractionMatrixPlan::new(1, 1, f32::MAX, criteria).unwrap();
    let mut workspace = plan.workspace();
    let mut interaction_matrix = [7.0];
    let mut diagnostics = DocrimeInteractionMatrixDiagnostics::default();
    let unchanged_diagnostics = diagnostics;

    assert_eq!(
        plan.process(
            DocrimeInteractionMatrixOutput::new(&mut interaction_matrix, &mut diagnostics),
            &mut workspace,
            DocrimeCovariances::new(2, &[f32::MAX], &[1.0]),
        ),
        Err(DocrimeInteractionMatrixError::NonFiniteCalculation {
            stage: DocrimeInteractionMatrixStage::Solve,
        })
    );
    assert_eq!(interaction_matrix, [7.0]);
    assert_eq!(diagnostics, unchanged_diagnostics);
}
