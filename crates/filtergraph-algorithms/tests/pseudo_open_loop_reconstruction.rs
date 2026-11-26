//! Pseudo-open-loop reconstruction tests.

use std::sync::Arc;

use filtergraph_algorithms::schemas::{
    CONTROLLER_RESIDUAL_ERROR_V1, PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1,
    PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1, PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1,
    REQUESTED_PDM_COMMAND_V1, SHACK_HARTMANN_SLOPES_V1,
};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmOutput, AlgorithmPlan,
    DeclarationParameterError, PseudoOpenLoopInputKind, PseudoOpenLoopMatrix,
    PseudoOpenLoopOutputKind, PseudoOpenLoopReconstructionError, PseudoOpenLoopReconstructionF32,
    PseudoOpenLoopReconstructionF32Configuration, PseudoOpenLoopReconstructionInput,
    PseudoOpenLoopReconstructionOutput, PseudoOpenLoopReconstructionPlan,
    PseudoOpenLoopReconstructionPlanError, PseudoOpenLoopStage,
};

fn scalar_plan(
    interaction: f32,
    reconstructor: f32,
    corrector_to_error: f32,
) -> PseudoOpenLoopReconstructionPlan<f32> {
    PseudoOpenLoopReconstructionPlan::new(
        1,
        1,
        1,
        Arc::from([interaction]),
        Arc::from([reconstructor]),
        Arc::from([corrector_to_error]),
    )
    .unwrap()
}

#[test]
fn nonidentity_matrices_follow_the_complete_residualized_equation() {
    let plan = PseudoOpenLoopReconstructionPlan::new(
        2,
        2,
        2,
        Arc::from([1.0_f64, 2.0, -1.0, 0.5]),
        Arc::from([2.0_f64, -1.0, 0.5, 3.0]),
        Arc::from([0.25_f64, -1.0, 2.0, 0.0]),
    )
    .unwrap();
    assert_eq!(plan.measurement_count(), 2);
    assert_eq!(plan.corrector_shape_count(), 2);
    assert_eq!(plan.residual_error_count(), 2);
    let mut workspace = plan.workspace();
    let mut pseudo_open_loop_measurements = [f64::NAN; 2];
    let mut residual_error = [f64::NAN; 2];

    plan.process(
        PseudoOpenLoopReconstructionOutput::new(
            &mut pseudo_open_loop_measurements,
            &mut residual_error,
        ),
        &mut workspace,
        PseudoOpenLoopReconstructionInput::new(&[4.0, 2.5], &[0.5, -2.0]),
    )
    .unwrap();

    assert_eq!(pseudo_open_loop_measurements, [0.5, 1.0]);
    assert_eq!(residual_error, [-2.125, 2.25]);
}

#[test]
fn matched_interaction_and_error_projections_cancel_the_corrector_shape() {
    let plan = PseudoOpenLoopReconstructionPlan::new(
        2,
        1,
        1,
        Arc::from([2.0_f32, 1.0]),
        Arc::from([0.5_f32, 1.0]),
        Arc::from([2.0_f32]),
    )
    .unwrap();
    let mut workspace = plan.workspace();
    let mut pseudo_open_loop_measurements = [0.0; 2];
    let mut residual_error = [0.0];

    plan.process(
        PseudoOpenLoopReconstructionOutput::new(
            &mut pseudo_open_loop_measurements,
            &mut residual_error,
        ),
        &mut workspace,
        PseudoOpenLoopReconstructionInput::new(&[4.0, 3.0], &[1.0]),
    )
    .unwrap();

    assert_eq!(pseudo_open_loop_measurements, [6.0, 4.0]);
    assert_eq!(residual_error, [5.0]);
}

#[test]
fn preparation_rejects_each_invalid_extent_and_matrix() {
    let one = || Arc::from([1.0_f32]);
    assert_eq!(
        PseudoOpenLoopReconstructionPlan::new(0, 1, 1, one(), one(), one()),
        Err(PseudoOpenLoopReconstructionPlanError::EmptyDimension)
    );
    assert_eq!(
        PseudoOpenLoopReconstructionPlan::new(1, 0, 1, one(), one(), one()),
        Err(PseudoOpenLoopReconstructionPlanError::EmptyDimension)
    );
    assert_eq!(
        PseudoOpenLoopReconstructionPlan::new(1, 1, 0, one(), one(), one()),
        Err(PseudoOpenLoopReconstructionPlanError::EmptyDimension)
    );
    assert_eq!(
        PseudoOpenLoopReconstructionPlan::<f32>::new(
            usize::MAX,
            2,
            1,
            Arc::from([]),
            Arc::from([]),
            Arc::from([]),
        ),
        Err(PseudoOpenLoopReconstructionPlanError::ExtentOverflow {
            matrix: PseudoOpenLoopMatrix::Interaction,
        })
    );

    for (matrix, interaction, reconstructor, corrector_to_error) in [
        (
            PseudoOpenLoopMatrix::Interaction,
            Arc::from([]),
            one(),
            one(),
        ),
        (
            PseudoOpenLoopMatrix::Reconstructor,
            one(),
            Arc::from([]),
            one(),
        ),
        (
            PseudoOpenLoopMatrix::CorrectorToError,
            one(),
            one(),
            Arc::from([]),
        ),
    ] {
        assert_eq!(
            PseudoOpenLoopReconstructionPlan::new(
                1,
                1,
                1,
                interaction,
                reconstructor,
                corrector_to_error,
            ),
            Err(PseudoOpenLoopReconstructionPlanError::MatrixExtent {
                matrix,
                expected: 1,
                actual: 0,
            })
        );
    }

    for (matrix, interaction, reconstructor, corrector_to_error) in [
        (
            PseudoOpenLoopMatrix::Interaction,
            Arc::from([f32::NAN]),
            one(),
            one(),
        ),
        (
            PseudoOpenLoopMatrix::Reconstructor,
            one(),
            Arc::from([f32::NAN]),
            one(),
        ),
        (
            PseudoOpenLoopMatrix::CorrectorToError,
            one(),
            one(),
            Arc::from([f32::NAN]),
        ),
    ] {
        assert_eq!(
            PseudoOpenLoopReconstructionPlan::new(
                1,
                1,
                1,
                interaction,
                reconstructor,
                corrector_to_error,
            ),
            Err(PseudoOpenLoopReconstructionPlanError::NonFiniteMatrix { matrix })
        );
    }
}

#[test]
fn execution_rejects_each_view_extent_before_publication() {
    let plan = scalar_plan(1.0, 1.0, 1.0);
    let other_plan = PseudoOpenLoopReconstructionPlan::new(
        2,
        1,
        1,
        Arc::from([1.0_f32, 1.0]),
        Arc::from([1.0_f32, 1.0]),
        Arc::from([1.0_f32]),
    )
    .unwrap();
    let mut workspace = plan.workspace();
    let mut other_workspace = other_plan.workspace();
    let mut pseudo_open_loop_measurements = [7.0_f32];
    let mut residual_error = [8.0_f32];

    let mut process = |residual_measurements: &[f32],
                       corrector_shape: &[f32],
                       pseudo_open_loop_measurements: &mut [f32],
                       residual_error: &mut [f32]| {
        plan.process(
            PseudoOpenLoopReconstructionOutput::new(pseudo_open_loop_measurements, residual_error),
            &mut workspace,
            PseudoOpenLoopReconstructionInput::new(residual_measurements, corrector_shape),
        )
    };

    assert_eq!(
        process(
            &[],
            &[1.0],
            &mut pseudo_open_loop_measurements,
            &mut residual_error,
        ),
        Err(PseudoOpenLoopReconstructionError::InputExtent {
            input: PseudoOpenLoopInputKind::ResidualMeasurements,
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(
        process(
            &[1.0],
            &[],
            &mut pseudo_open_loop_measurements,
            &mut residual_error,
        ),
        Err(PseudoOpenLoopReconstructionError::InputExtent {
            input: PseudoOpenLoopInputKind::CorrectorShape,
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(
        process(&[1.0], &[1.0], &mut [], &mut residual_error),
        Err(PseudoOpenLoopReconstructionError::OutputExtent {
            output: PseudoOpenLoopOutputKind::PseudoOpenLoopMeasurements,
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(
        process(&[1.0], &[1.0], &mut pseudo_open_loop_measurements, &mut [],),
        Err(PseudoOpenLoopReconstructionError::OutputExtent {
            output: PseudoOpenLoopOutputKind::ResidualError,
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(pseudo_open_loop_measurements, [7.0]);
    assert_eq!(residual_error, [8.0]);

    assert_eq!(
        plan.process(
            PseudoOpenLoopReconstructionOutput::new(
                &mut pseudo_open_loop_measurements,
                &mut residual_error,
            ),
            &mut other_workspace,
            PseudoOpenLoopReconstructionInput::new(&[1.0], &[1.0]),
        ),
        Err(PseudoOpenLoopReconstructionError::WorkspaceExtent)
    );
    assert_eq!(pseudo_open_loop_measurements, [7.0]);
    assert_eq!(residual_error, [8.0]);
}

#[test]
fn nonfinite_inputs_and_every_nonfinite_stage_preserve_both_outputs() {
    let cases = [
        (
            scalar_plan(f32::MAX, 0.0, 0.0),
            0.0,
            2.0,
            PseudoOpenLoopStage::PredictedMeasurements,
        ),
        (
            scalar_plan(1.0, 0.0, 0.0),
            f32::MAX,
            f32::MAX,
            PseudoOpenLoopStage::PseudoOpenLoopMeasurements,
        ),
        (
            scalar_plan(0.0, 2.0, 0.0),
            f32::MAX,
            1.0,
            PseudoOpenLoopStage::OpenLoopEstimate,
        ),
        (
            scalar_plan(0.0, 0.0, f32::MAX),
            0.0,
            2.0,
            PseudoOpenLoopStage::CorrectorInErrorCoordinates,
        ),
        (
            scalar_plan(0.0, 1.0, -1.0),
            f32::MAX,
            f32::MAX,
            PseudoOpenLoopStage::ResidualError,
        ),
    ];

    for (plan, residual_measurement, corrector_shape, stage) in cases {
        let mut workspace = plan.workspace();
        let mut pseudo_open_loop_measurements = [7.0];
        let mut residual_error = [8.0];
        assert_eq!(
            plan.process(
                PseudoOpenLoopReconstructionOutput::new(
                    &mut pseudo_open_loop_measurements,
                    &mut residual_error,
                ),
                &mut workspace,
                PseudoOpenLoopReconstructionInput::new(
                    &[residual_measurement],
                    &[corrector_shape],
                ),
            ),
            Err(PseudoOpenLoopReconstructionError::NonFiniteCalculation { stage })
        );
        assert_eq!(pseudo_open_loop_measurements, [7.0]);
        assert_eq!(residual_error, [8.0]);
    }

    let plan = scalar_plan(1.0, 1.0, 1.0);
    for (input, residual_measurements, corrector_shape) in [
        (
            PseudoOpenLoopInputKind::ResidualMeasurements,
            [f32::NAN],
            [0.0],
        ),
        (PseudoOpenLoopInputKind::CorrectorShape, [0.0], [f32::NAN]),
    ] {
        let mut workspace = plan.workspace();
        let mut pseudo_open_loop_measurements = [7.0];
        let mut residual_error = [8.0];
        assert_eq!(
            plan.process(
                PseudoOpenLoopReconstructionOutput::new(
                    &mut pseudo_open_loop_measurements,
                    &mut residual_error,
                ),
                &mut workspace,
                PseudoOpenLoopReconstructionInput::new(&residual_measurements, &corrector_shape,),
            ),
            Err(PseudoOpenLoopReconstructionError::NonFiniteInput { input, index: 0 })
        );
        assert_eq!(pseudo_open_loop_measurements, [7.0]);
        assert_eq!(residual_error, [8.0]);
    }
}

#[test]
fn declaration_preserves_generic_schemas_and_replaces_each_calibrated_map() {
    let mut declared =
        PseudoOpenLoopReconstructionF32::prepare(PseudoOpenLoopReconstructionF32Configuration {
            measurement_count: 2,
            corrector_shape_count: 1,
            residual_error_count: 1,
            initial_interaction_matrix: Some(vec![2.0, 1.0]),
            initial_reconstructor: Some(vec![0.5, 1.0]),
            initial_corrector_to_error: Some(vec![2.0]),
            residual_measurements_schema: SHACK_HARTMANN_SLOPES_V1.into(),
            corrector_shape_schema: REQUESTED_PDM_COMMAND_V1.into(),
            pseudo_open_loop_measurements_schema: SHACK_HARTMANN_SLOPES_V1.into(),
            residual_error_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
        })
        .unwrap();

    assert_eq!(
        PseudoOpenLoopReconstructionF32::NAME,
        "pseudo-open-loop-reconstruction-f32"
    );
    let formats = declared.formats();
    assert_eq!(formats[0].shape(), &[2]);
    assert_eq!(formats[0].schema(), SHACK_HARTMANN_SLOPES_V1);
    assert_eq!(formats[1].shape(), &[1]);
    assert_eq!(formats[1].schema(), REQUESTED_PDM_COMMAND_V1);
    assert_eq!(formats[2].schema(), SHACK_HARTMANN_SLOPES_V1);
    assert_eq!(formats[3].schema(), CONTROLLER_RESIDUAL_ERROR_V1);
    assert_eq!(formats[4].shape(), &[2, 1]);
    assert_eq!(formats[4].schema(), PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1);
    assert_eq!(formats[5].shape(), &[1, 2]);
    assert_eq!(formats[5].schema(), PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1);
    assert_eq!(formats[6].shape(), &[1, 1]);
    assert_eq!(formats[6].schema(), PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1);

    let residual_measurements = [4.0_f32, 3.0];
    let corrector_shape = [1.0_f32];
    let inputs = [
        AlgorithmInput::from(&residual_measurements[..]),
        AlgorithmInput::from(&corrector_shape[..]),
    ];
    let mut pseudo_open_loop_measurements = [0.0_f32; 2];
    let mut residual_error = [0.0_f32];
    declared
        .process(
            &inputs,
            &mut [
                AlgorithmOutput::from(&mut pseudo_open_loop_measurements[..]),
                AlgorithmOutput::from(&mut residual_error[..]),
            ],
        )
        .unwrap();
    assert_eq!(pseudo_open_loop_measurements, [6.0, 4.0]);
    assert_eq!(residual_error, [5.0]);

    assert_eq!(
        declared.replace_parameter(4, AlgorithmInput::from(&[0.0_f32, 0.0][..])),
        Some(Ok(()))
    );
    declared
        .process(
            &inputs,
            &mut [
                AlgorithmOutput::from(&mut pseudo_open_loop_measurements[..]),
                AlgorithmOutput::from(&mut residual_error[..]),
            ],
        )
        .unwrap();
    assert_eq!(pseudo_open_loop_measurements, residual_measurements);
    assert_eq!(residual_error, [3.0]);

    assert_eq!(
        declared.replace_parameter(5, AlgorithmInput::from(&[f32::NAN, 0.0][..])),
        Some(Err(DeclarationParameterError::Algorithm(
            PseudoOpenLoopReconstructionPlanError::NonFiniteMatrix {
                matrix: PseudoOpenLoopMatrix::Reconstructor,
            }
        )))
    );
    declared
        .process(
            &inputs,
            &mut [
                AlgorithmOutput::from(&mut pseudo_open_loop_measurements[..]),
                AlgorithmOutput::from(&mut residual_error[..]),
            ],
        )
        .unwrap();
    assert_eq!(residual_error, [3.0]);
}
