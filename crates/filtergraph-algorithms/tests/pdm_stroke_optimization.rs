//! Calibrated PDM stroke-optimization tests.

use std::sync::Arc;

use filtergraph_algorithms::schemas::{
    PDM_COMMAND_TO_NULL_MODES_V1, PDM_NULL_MODES_TO_COMMAND_V1, PREFERRED_PDM_COMMAND_V1,
    REQUESTED_PDM_COMMAND_V1,
};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, DeclarationParameterError,
    PdmStrokeOptimizationError, PdmStrokeOptimizationF32, PdmStrokeOptimizationF32Configuration,
    PdmStrokeOptimizationPlan, PdmStrokeOptimizationPlanError,
};

#[test]
fn uniform_piston_fixture_reduces_stroke_without_changing_shape() {
    let plan = PdmStrokeOptimizationPlan::new(
        3,
        1,
        Arc::from([0.0_f32; 3]),
        Arc::from([1.0_f32 / 3.0; 3]),
        Arc::from([1.0_f32; 3]),
    )
    .unwrap();
    let mut workspace = plan.workspace();
    let mut output = [0.0; 3];

    plan.process(&mut output, &mut workspace, &[4.0, 1.0, -2.0])
        .unwrap();

    assert_eq!(output, [3.0, 0.0, -3.0]);
    assert_eq!(output[0] - output[1], 3.0);
    assert_eq!(output[1] - output[2], 3.0);
    assert!(output.iter().map(|value| value * value).sum::<f32>() < 21.0);
}

#[test]
fn preferred_command_is_a_fixed_point() {
    let preferred = [0.5_f64, -0.25];
    let plan = PdmStrokeOptimizationPlan::new(
        2,
        1,
        Arc::from(preferred),
        Arc::from([0.5_f64, 0.5]),
        Arc::from([1.0_f64, 1.0]),
    )
    .unwrap();
    let mut workspace = plan.workspace();
    let mut output = [0.0; 2];

    plan.process(&mut output, &mut workspace, &preferred)
        .unwrap();

    assert_eq!(output, preferred);
}

#[test]
fn independently_prepared_maps_are_used_without_transpose_inference() {
    let plan = PdmStrokeOptimizationPlan::new(
        2,
        1,
        Arc::from([0.0_f64; 2]),
        Arc::from([0.25_f64, 0.75]),
        Arc::from([2.0_f64, 2.0 / 3.0]),
    )
    .unwrap();
    let mut workspace = plan.workspace();
    let mut output = [0.0; 2];

    plan.process(&mut output, &mut workspace, &[4.0, 0.0])
        .unwrap();

    assert_eq!(output, [2.0, -2.0 / 3.0]);
}

#[test]
fn preparation_rejects_empty_invalid_and_nonfinite_artifacts() {
    assert_eq!(
        PdmStrokeOptimizationPlan::<f32>::new(0, 1, Arc::from([]), Arc::from([]), Arc::from([]),),
        Err(PdmStrokeOptimizationPlanError::EmptyPdm)
    );
    assert_eq!(
        PdmStrokeOptimizationPlan::new(1, 0, Arc::from([0.0_f32]), Arc::from([]), Arc::from([])),
        Err(PdmStrokeOptimizationPlanError::EmptyNullModes)
    );
    assert_eq!(
        PdmStrokeOptimizationPlan::new(
            2,
            1,
            Arc::from([0.0_f32; 2]),
            Arc::from([1.0_f32]),
            Arc::from([1.0_f32; 2])
        ),
        Err(PdmStrokeOptimizationPlanError::Extent)
    );
    assert_eq!(
        PdmStrokeOptimizationPlan::new(
            1,
            1,
            Arc::from([f32::NAN]),
            Arc::from([1.0_f32]),
            Arc::from([1.0_f32])
        ),
        Err(PdmStrokeOptimizationPlanError::NonFiniteConfiguration)
    );
}

#[test]
fn processing_rejects_invalid_extents_and_nonfinite_values() {
    let plan = PdmStrokeOptimizationPlan::new(
        1,
        1,
        Arc::from([-f32::MAX]),
        Arc::from([1.0_f32]),
        Arc::from([1.0_f32]),
    )
    .unwrap();
    let mut workspace = plan.workspace();
    let mut output = [0.0];

    assert_eq!(
        plan.process(&mut output, &mut workspace, &[]),
        Err(PdmStrokeOptimizationError::InputExtent)
    );
    assert_eq!(
        plan.process(&mut [], &mut workspace, &[0.0]),
        Err(PdmStrokeOptimizationError::OutputExtent)
    );
    assert_eq!(
        plan.process(&mut output, &mut workspace, &[f32::NAN]),
        Err(PdmStrokeOptimizationError::NonFiniteInput)
    );
    assert_eq!(
        plan.process(&mut output, &mut workspace, &[f32::MAX]),
        Err(PdmStrokeOptimizationError::NonFiniteOutput)
    );
}

#[test]
fn declaration_exposes_each_calibrated_artifact_without_copying_unchanged_maps() {
    let mut declared = PdmStrokeOptimizationF32::prepare(PdmStrokeOptimizationF32Configuration {
        null_mode_count: 1,
        preferred_command: vec![0.0; 3],
        command_to_null_modes: vec![1.0 / 3.0; 3],
        null_modes_to_command: vec![1.0; 3],
        actuator_count: 3,
    })
    .unwrap();
    assert_eq!(
        PdmStrokeOptimizationF32::NAME,
        "pdm-stroke-optimization-f32"
    );
    assert_eq!(declared.formats()[0].schema(), REQUESTED_PDM_COMMAND_V1);
    assert_eq!(declared.formats()[1].schema(), REQUESTED_PDM_COMMAND_V1);
    assert_eq!(declared.formats()[2].shape(), &[3]);
    assert_eq!(declared.formats()[2].schema(), PREFERRED_PDM_COMMAND_V1);
    assert_eq!(declared.formats()[3].shape(), &[1, 3]);
    assert_eq!(declared.formats()[3].schema(), PDM_COMMAND_TO_NULL_MODES_V1);
    assert_eq!(declared.formats()[4].shape(), &[3, 1]);
    assert_eq!(declared.formats()[4].schema(), PDM_NULL_MODES_TO_COMMAND_V1);

    let requested = [4.0_f32, 1.0, -2.0];
    let mut optimized = [0.0_f32; 3];
    PdmStrokeOptimizationF32::process(&mut declared, &requested, &mut optimized).unwrap();
    assert_eq!(optimized, [3.0, 0.0, -3.0]);

    let preferred = [1.0_f32; 3];
    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&preferred[..])),
        Some(Ok(()))
    );
    PdmStrokeOptimizationF32::process(&mut declared, &preferred, &mut optimized).unwrap();
    assert_eq!(optimized, preferred);

    let invalid_map = [f32::NAN, 0.0, 0.0];
    assert_eq!(
        declared.replace_parameter(3, AlgorithmInput::from(&invalid_map[..])),
        Some(Err(DeclarationParameterError::Algorithm(
            PdmStrokeOptimizationPlanError::NonFiniteConfiguration
        )))
    );
    PdmStrokeOptimizationF32::process(&mut declared, &preferred, &mut optimized).unwrap();
    assert_eq!(optimized, preferred);
}
