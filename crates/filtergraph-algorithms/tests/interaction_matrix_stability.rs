//! Independent-window interaction-matrix stability tests.

use filtergraph_algorithms::schemas::{
    CORRECTOR_COORDINATE_V1, INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1,
    INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1,
    INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1, INTERACTION_MATRIX_STABILITY_STATUS_V1,
};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmOutput, AlgorithmPlan, AlgorithmProperties,
    InteractionMatrixStabilityCriteria, InteractionMatrixStabilityCriteriaError,
    InteractionMatrixStabilityDiagnostics, InteractionMatrixStabilityError,
    InteractionMatrixStabilityF32, InteractionMatrixStabilityF32Configuration,
    InteractionMatrixStabilityF32PreparationError, InteractionMatrixStabilityInput,
    InteractionMatrixStabilityInputKind, InteractionMatrixStabilityPlan,
    InteractionMatrixStabilityPlanError, InteractionMatrixStabilityStatus, NoWorkspace,
    PropertyAssignment, PropertyValue,
};
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
struct SharedFixture {
    schema: String,
    measurement_count: usize,
    corrector_coordinate_count: usize,
    first_window: Vec<f64>,
    second_window: Vec<f64>,
    minimum_column_norm: f64,
    maximum_relative_column_difference: f64,
    maximum_relative_frobenius_difference: f64,
    expected_status: String,
    expected_status_corrector_coordinate_zero_based: usize,
    expected_minimum_first_window_column_norm: f64,
    expected_minimum_second_window_column_norm: f64,
    expected_maximum_relative_column_difference: f64,
    expected_maximum_relative_column_difference_coordinate_zero_based: usize,
    expected_relative_frobenius_difference: f64,
    absolute_tolerance_f64: f64,
}

fn shared_fixture() -> SharedFixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/algorithms/interaction_matrix_stability.toml"
    );
    let source = fs::read_to_string(path).expect("read shared matrix-stability fixture");
    toml::from_str(&source).expect("parse shared matrix-stability fixture")
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

#[test]
fn julia_and_rust_share_the_stability_equation_fixture() {
    let fixture = shared_fixture();
    assert_eq!(
        fixture.schema,
        "org.calculon.algorithm-fixture/interaction-matrix-stability/1"
    );
    assert_eq!(
        fixture.expected_status,
        "excessive_relative_column_difference"
    );
    let criteria = InteractionMatrixStabilityCriteria::new(
        fixture.minimum_column_norm,
        fixture.maximum_relative_column_difference,
        fixture.maximum_relative_frobenius_difference,
    )
    .unwrap();
    let plan = InteractionMatrixStabilityPlan::new(
        fixture.measurement_count,
        fixture.corrector_coordinate_count,
        criteria,
    )
    .unwrap();
    let mut diagnostics = InteractionMatrixStabilityDiagnostics::default();

    plan.process(
        &mut diagnostics,
        &mut NoWorkspace,
        InteractionMatrixStabilityInput::new(&fixture.first_window, &fixture.second_window),
    )
    .unwrap();

    assert_eq!(
        diagnostics.status(),
        InteractionMatrixStabilityStatus::ExcessiveRelativeColumnDifference {
            corrector_coordinate: fixture.expected_status_corrector_coordinate_zero_based,
        }
    );
    assert_close(
        diagnostics.minimum_first_window_column_norm(),
        fixture.expected_minimum_first_window_column_norm,
        fixture.absolute_tolerance_f64,
    );
    assert_close(
        diagnostics.minimum_second_window_column_norm(),
        fixture.expected_minimum_second_window_column_norm,
        fixture.absolute_tolerance_f64,
    );
    assert_close(
        diagnostics.maximum_relative_column_difference(),
        fixture.expected_maximum_relative_column_difference,
        fixture.absolute_tolerance_f64,
    );
    assert_eq!(
        diagnostics.maximum_relative_column_difference_coordinate(),
        fixture.expected_maximum_relative_column_difference_coordinate_zero_based
    );
    assert_close(
        diagnostics.relative_frobenius_difference(),
        fixture.expected_relative_frobenius_difference,
        fixture.absolute_tolerance_f64,
    );
}

fn criteria<T>() -> InteractionMatrixStabilityCriteria<T>
where
    T: filtergraph_algorithms::MatrixElement,
{
    InteractionMatrixStabilityCriteria::new(
        T::from(0.1).unwrap(),
        T::from(0.5).unwrap(),
        T::from(0.5).unwrap(),
    )
    .unwrap()
}

#[test]
fn identical_matrices_are_valid_with_zero_difference() {
    let plan = InteractionMatrixStabilityPlan::new(2, 2, criteria()).unwrap();
    let matrix = [3.0_f64, 0.0, 4.0, 2.0];
    let mut diagnostics = InteractionMatrixStabilityDiagnostics::default();

    plan.process(
        &mut diagnostics,
        &mut NoWorkspace,
        InteractionMatrixStabilityInput::new(&matrix, &matrix),
    )
    .unwrap();

    assert_eq!(
        diagnostics.status(),
        InteractionMatrixStabilityStatus::Valid
    );
    assert_eq!(diagnostics.minimum_first_window_column_norm(), 2.0);
    assert_eq!(diagnostics.minimum_second_window_column_norm(), 2.0);
    assert_eq!(diagnostics.maximum_relative_column_difference(), 0.0);
    assert_eq!(
        diagnostics.maximum_relative_column_difference_coordinate(),
        0
    );
    assert_eq!(diagnostics.relative_frobenius_difference(), 0.0);
}

#[test]
fn symmetric_difference_is_independent_of_window_order() {
    let plan = InteractionMatrixStabilityPlan::new(
        2,
        2,
        InteractionMatrixStabilityCriteria::new(0.1_f64, 2.0, 2.0).unwrap(),
    )
    .unwrap();
    let first = [1.0, 0.0, 0.0, 1.0];
    let second = [2.0, 0.0, 0.0, 2.0];
    let mut forward = InteractionMatrixStabilityDiagnostics::default();
    let mut reverse = InteractionMatrixStabilityDiagnostics::default();

    plan.process(
        &mut forward,
        &mut NoWorkspace,
        InteractionMatrixStabilityInput::new(&first, &second),
    )
    .unwrap();
    plan.process(
        &mut reverse,
        &mut NoWorkspace,
        InteractionMatrixStabilityInput::new(&second, &first),
    )
    .unwrap();

    assert_eq!(forward.status(), reverse.status());
    assert_eq!(
        forward.minimum_first_window_column_norm(),
        reverse.minimum_second_window_column_norm()
    );
    assert_eq!(
        forward.minimum_second_window_column_norm(),
        reverse.minimum_first_window_column_norm()
    );
    assert_eq!(
        forward.maximum_relative_column_difference(),
        reverse.maximum_relative_column_difference()
    );
    assert_eq!(
        forward.relative_frobenius_difference(),
        reverse.relative_frobenius_difference()
    );
    assert!((forward.maximum_relative_column_difference() - 2.0 / 3.0).abs() < 1.0e-14);
    assert!((forward.relative_frobenius_difference() - 2.0 / 3.0).abs() < 1.0e-14);
    assert_eq!(
        forward.maximum_relative_column_difference_coordinate(),
        0,
        "equal column differences retain the lowest coordinate"
    );
}

#[test]
fn status_order_is_norm_then_column_then_frobenius() {
    let plan = InteractionMatrixStabilityPlan::new(2, 2, criteria()).unwrap();
    let mut diagnostics = InteractionMatrixStabilityDiagnostics::default();

    plan.process(
        &mut diagnostics,
        &mut NoWorkspace,
        InteractionMatrixStabilityInput::new(&[0.0_f64, 0.0, 0.0, 1.0], &[2.0_f64, 0.0, 0.0, 1.0]),
    )
    .unwrap();
    assert_eq!(
        diagnostics.status(),
        InteractionMatrixStabilityStatus::InsufficientColumnNorm {
            corrector_coordinate: 0,
        }
    );
    assert_eq!(diagnostics.maximum_relative_column_difference(), 0.0);

    plan.process(
        &mut diagnostics,
        &mut NoWorkspace,
        InteractionMatrixStabilityInput::new(&[1.0_f64, 0.0, 0.0, 1.0], &[2.0_f64, 0.0, 0.0, 2.0]),
    )
    .unwrap();
    assert_eq!(
        diagnostics.status(),
        InteractionMatrixStabilityStatus::ExcessiveRelativeColumnDifference {
            corrector_coordinate: 0,
        }
    );

    let frobenius_plan = InteractionMatrixStabilityPlan::new(
        2,
        2,
        InteractionMatrixStabilityCriteria::new(0.1_f64, 1.0, 0.5).unwrap(),
    )
    .unwrap();
    frobenius_plan
        .process(
            &mut diagnostics,
            &mut NoWorkspace,
            InteractionMatrixStabilityInput::new(
                &[1.0_f64, 0.0, 0.0, 1.0],
                &[2.0_f64, 0.0, 0.0, 2.0],
            ),
        )
        .unwrap();
    assert_eq!(
        diagnostics.status(),
        InteractionMatrixStabilityStatus::ExcessiveRelativeFrobeniusDifference
    );
}

#[test]
fn f32_and_f64_produce_equivalent_diagnostics() {
    let plan32 = InteractionMatrixStabilityPlan::new(
        2,
        2,
        InteractionMatrixStabilityCriteria::new(0.1_f32, 2.0, 2.0).unwrap(),
    )
    .unwrap();
    let plan64 = InteractionMatrixStabilityPlan::new(
        2,
        2,
        InteractionMatrixStabilityCriteria::new(0.1_f64, 2.0, 2.0).unwrap(),
    )
    .unwrap();
    let mut diagnostics32 = InteractionMatrixStabilityDiagnostics::default();
    let mut diagnostics64 = InteractionMatrixStabilityDiagnostics::default();
    plan32
        .process(
            &mut diagnostics32,
            &mut NoWorkspace,
            InteractionMatrixStabilityInput::new(
                &[1.0_f32, 0.0, 0.0, 1.0],
                &[2.0_f32, 0.0, 0.0, 2.0],
            ),
        )
        .unwrap();
    plan64
        .process(
            &mut diagnostics64,
            &mut NoWorkspace,
            InteractionMatrixStabilityInput::new(
                &[1.0_f64, 0.0, 0.0, 1.0],
                &[2.0_f64, 0.0, 0.0, 2.0],
            ),
        )
        .unwrap();

    assert_eq!(diagnostics32.status(), diagnostics64.status());
    assert!(
        (diagnostics32.maximum_relative_column_difference() as f64
            - diagnostics64.maximum_relative_column_difference())
        .abs()
            < 1.0e-6
    );
    assert!(
        (diagnostics32.relative_frobenius_difference() as f64
            - diagnostics64.relative_frobenius_difference())
        .abs()
            < 1.0e-6
    );
}

#[test]
fn preparation_rejects_invalid_dimensions_and_criteria() {
    let valid = criteria::<f32>();
    assert_eq!(
        InteractionMatrixStabilityPlan::new(0, 1, valid),
        Err(InteractionMatrixStabilityPlanError::EmptyDimension)
    );
    assert_eq!(
        InteractionMatrixStabilityPlan::new(1, 0, valid),
        Err(InteractionMatrixStabilityPlanError::EmptyDimension)
    );
    assert_eq!(
        InteractionMatrixStabilityPlan::new(usize::MAX, 2, valid),
        Err(InteractionMatrixStabilityPlanError::ExtentOverflow)
    );

    for minimum in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            InteractionMatrixStabilityCriteria::new(minimum, 1.0, 1.0),
            Err(InteractionMatrixStabilityCriteriaError::InvalidMinimumColumnNorm)
        );
    }
    for maximum in [-1.0_f32, 2.1, f32::NAN, f32::INFINITY] {
        assert_eq!(
            InteractionMatrixStabilityCriteria::new(0.1, maximum, 1.0),
            Err(InteractionMatrixStabilityCriteriaError::InvalidMaximumRelativeColumnDifference)
        );
        assert_eq!(
            InteractionMatrixStabilityCriteria::new(0.1, 1.0, maximum),
            Err(InteractionMatrixStabilityCriteriaError::InvalidMaximumRelativeFrobeniusDifference)
        );
    }
}

#[test]
fn rejected_input_or_calculation_preserves_diagnostics() {
    let plan = InteractionMatrixStabilityPlan::new(1, 2, criteria()).unwrap();
    let original = InteractionMatrixStabilityDiagnostics::<f32>::default();
    let mut diagnostics = original;

    for (input, expected, diagnostic) in [
        (
            InteractionMatrixStabilityInput::new(&[1.0], &[1.0, 2.0]),
            InteractionMatrixStabilityError::InputExtent {
                input: InteractionMatrixStabilityInputKind::FirstWindow,
                expected: 2,
                actual: 1,
            },
            "FirstWindow extent 1 does not match 2",
        ),
        (
            InteractionMatrixStabilityInput::new(&[1.0, 2.0], &[1.0]),
            InteractionMatrixStabilityError::InputExtent {
                input: InteractionMatrixStabilityInputKind::SecondWindow,
                expected: 2,
                actual: 1,
            },
            "SecondWindow extent 1 does not match 2",
        ),
        (
            InteractionMatrixStabilityInput::new(&[f32::NAN, 1.0], &[1.0, 2.0]),
            InteractionMatrixStabilityError::NonFiniteInput {
                input: InteractionMatrixStabilityInputKind::FirstWindow,
                index: 0,
            },
            "FirstWindow coefficient at index 0",
        ),
        (
            InteractionMatrixStabilityInput::new(&[1.0, 2.0], &[1.0, f32::INFINITY]),
            InteractionMatrixStabilityError::NonFiniteInput {
                input: InteractionMatrixStabilityInputKind::SecondWindow,
                index: 1,
            },
            "SecondWindow coefficient at index 1",
        ),
        (
            InteractionMatrixStabilityInput::new(&[f32::MAX, 1.0], &[-f32::MAX, 2.0]),
            InteractionMatrixStabilityError::NonFiniteCalculation,
            "calculation produced a non-finite value",
        ),
    ] {
        let error = plan
            .process(&mut diagnostics, &mut NoWorkspace, input)
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(diagnostic), "{error}");
        assert_eq!(diagnostics, original);
    }
}

#[test]
fn declaration_exposes_typed_diagnostics_and_dimensionless_runtime_thresholds() {
    let matrix_schema = "test.filtergraph/test-interaction-matrix/1";
    let mut declared =
        InteractionMatrixStabilityF32::prepare(InteractionMatrixStabilityF32Configuration {
            measurement_count: 2,
            corrector_coordinate_count: 2,
            minimum_column_norm: 0.1,
            interaction_matrix_schema: matrix_schema.into(),
        })
        .unwrap();

    assert_eq!(
        InteractionMatrixStabilityF32::NAME,
        "interaction-matrix-stability-f32"
    );
    let formats = declared.formats();
    assert_eq!(formats.len(), 8);
    assert_eq!(formats[0].shape(), &[2, 2]);
    assert_eq!(formats[0].schema(), matrix_schema);
    assert_eq!(formats[1].schema(), matrix_schema);
    assert_eq!(formats[2].schema(), INTERACTION_MATRIX_STABILITY_STATUS_V1);
    assert_eq!(formats[3].schema(), CORRECTOR_COORDINATE_V1);
    assert_eq!(
        formats[4].schema(),
        INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1
    );
    assert_eq!(
        formats[5].schema(),
        INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1
    );
    assert_eq!(formats[6].schema(), CORRECTOR_COORDINATE_V1);
    assert_eq!(
        formats[7].schema(),
        INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1
    );

    let column_threshold = declared.plan().property_info(0).unwrap();
    let frobenius_threshold = declared.plan().property_info(1).unwrap();
    assert_eq!(column_threshold.unit, "1");
    assert_eq!(frobenius_threshold.unit, "1");
    assert_eq!(declared.plan().property_info(2), None);
    declared
        .replace_properties(&[
            PropertyAssignment {
                id: column_threshold.id,
                value: PropertyValue::Float(1.0),
            },
            PropertyAssignment {
                id: frobenius_threshold.id,
                value: PropertyValue::Float(0.5),
            },
        ])
        .unwrap();

    let first_window = [1.0_f32, 0.0, 0.0, 1.0];
    let second_window = [2.0_f32, 0.0, 0.0, 2.0];
    let inputs = [
        AlgorithmInput::from(&first_window[..]),
        AlgorithmInput::from(&second_window[..]),
    ];
    let mut status = [u8::MAX];
    let mut status_coordinate = [u64::MAX];
    let mut minimum_column_norms = [0.0_f32; 2];
    let mut maximum_relative_column_difference = [0.0_f32];
    let mut maximum_relative_column_difference_coordinate = [u64::MAX];
    let mut relative_frobenius_difference = [0.0_f32];
    declared
        .process(
            &inputs,
            &mut [
                AlgorithmOutput::from(&mut status[..]),
                AlgorithmOutput::from(&mut status_coordinate[..]),
                AlgorithmOutput::from(&mut minimum_column_norms[..]),
                AlgorithmOutput::from(&mut maximum_relative_column_difference[..]),
                AlgorithmOutput::from(&mut maximum_relative_column_difference_coordinate[..]),
                AlgorithmOutput::from(&mut relative_frobenius_difference[..]),
            ],
        )
        .unwrap();

    assert_eq!(status, [3]);
    assert_eq!(status_coordinate, [0]);
    assert_eq!(minimum_column_norms, [1.0, 2.0]);
    assert!((maximum_relative_column_difference[0] - 2.0 / 3.0).abs() < 1.0e-6);
    assert_eq!(maximum_relative_column_difference_coordinate, [0]);
    assert!((relative_frobenius_difference[0] - 2.0 / 3.0).abs() < 1.0e-6);

    assert!(matches!(
        InteractionMatrixStabilityF32::prepare(InteractionMatrixStabilityF32Configuration {
            measurement_count: 2,
            corrector_coordinate_count: 2,
            minimum_column_norm: 0.0,
            interaction_matrix_schema: matrix_schema.into(),
        }),
        Err(
            filtergraph_algorithms::DeclarationPreparationError::Algorithm(
                InteractionMatrixStabilityF32PreparationError::Criteria(
                    InteractionMatrixStabilityCriteriaError::InvalidMinimumColumnNorm
                )
            )
        )
    ));
}
