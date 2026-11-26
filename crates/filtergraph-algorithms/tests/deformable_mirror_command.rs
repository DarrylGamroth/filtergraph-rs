//! Deformable-mirror projection, limiting, slew, and feedback tests.

use filtergraph_algorithms::{
    AlgorithmPlan, DeformableMirrorCommandError, DeformableMirrorCommandInput,
    DeformableMirrorCommandOutput, DeformableMirrorCommandPlan, DeformableMirrorCommandPlanError,
    InterActuatorStrokeLimit, InterActuatorStrokeLimitPlan, PdmCommandPlan,
    PdmCommandPowerLimitPlan, PdmFeedbackToVdmPlan, PdmProjection, PdmStrokeOptimizationPlan,
    VdmToPdmPlan,
};
use std::sync::Arc;

fn scao_plan(slew: Option<Arc<[f32]>>) -> DeformableMirrorCommandPlan<f32> {
    DeformableMirrorCommandPlan::new(
        2,
        3,
        3,
        Arc::from([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        Arc::from([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        Arc::from([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        Arc::from([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        Arc::from([-0.25; 3]),
        Arc::from([0.25; 3]),
        slew,
    )
    .unwrap()
}

fn component_plans(
    active_vdm_size: usize,
    full_vdm_size: usize,
    pdm_size: usize,
) -> (
    VdmToPdmPlan<f32>,
    PdmFeedbackToVdmPlan<f32>,
    PdmCommandPlan<f32>,
) {
    let projection = PdmProjection::new(
        pdm_size,
        Arc::from(vec![1.0; full_vdm_size * pdm_size]),
        Arc::from(vec![1.0; full_vdm_size * pdm_size]),
    )
    .unwrap();
    let projections: Arc<[PdmProjection<f32>]> = Arc::from([projection]);
    let vdm_to_pdm = VdmToPdmPlan::new(
        active_vdm_size,
        full_vdm_size,
        Arc::from(vec![1.0; active_vdm_size * full_vdm_size]),
        Arc::clone(&projections),
    )
    .unwrap();
    let pdm_feedback_to_vdm = PdmFeedbackToVdmPlan::new(
        active_vdm_size,
        full_vdm_size,
        Arc::from(vec![1.0; active_vdm_size * full_vdm_size]),
        projections,
    )
    .unwrap();
    let pdm = PdmCommandPlan::new(
        Arc::from(vec![-1.0; pdm_size]),
        Arc::from(vec![1.0; pdm_size]),
        None,
    )
    .unwrap();
    (vdm_to_pdm, pdm_feedback_to_vdm, pdm)
}

#[test]
fn compatible_prepared_plans_form_the_composite() {
    let (vdm_to_pdm, pdm_feedback_to_vdm, pdm) = component_plans(1, 1, 1);
    let plan =
        DeformableMirrorCommandPlan::from_plans(vdm_to_pdm, pdm_feedback_to_vdm, pdm).unwrap();

    assert!(plan.workspace(&[0.0]).is_ok());
}

#[test]
fn prepared_plan_composition_rejects_incompatible_coordinate_spaces() {
    let (vdm_to_pdm, _, pdm) = component_plans(1, 1, 1);
    let (_, pdm_feedback_to_vdm, _) = component_plans(2, 1, 1);
    assert_eq!(
        DeformableMirrorCommandPlan::from_plans(vdm_to_pdm, pdm_feedback_to_vdm, pdm),
        Err(DeformableMirrorCommandPlanError::MatrixExtent),
    );

    let (vdm_to_pdm, pdm_feedback_to_vdm, _) = component_plans(1, 1, 1);
    let (_, _, pdm) = component_plans(1, 1, 2);
    assert_eq!(
        DeformableMirrorCommandPlan::from_plans(vdm_to_pdm, pdm_feedback_to_vdm, pdm),
        Err(DeformableMirrorCommandPlanError::MatrixExtent),
    );

    let projections: Arc<[PdmProjection<f32>]> = Arc::from([
        PdmProjection::new(1, Arc::from([1.0]), Arc::from([1.0])).unwrap(),
        PdmProjection::new(1, Arc::from([1.0]), Arc::from([1.0])).unwrap(),
    ]);
    let vdm_to_pdm = VdmToPdmPlan::new(1, 1, Arc::from([1.0]), Arc::clone(&projections)).unwrap();
    let pdm_feedback_to_vdm =
        PdmFeedbackToVdmPlan::new(1, 1, Arc::from([1.0]), projections).unwrap();
    let (_, _, pdm) = component_plans(1, 1, 2);
    assert_eq!(
        DeformableMirrorCommandPlan::from_plans(vdm_to_pdm, pdm_feedback_to_vdm, pdm),
        Err(DeformableMirrorCommandPlanError::MatrixExtent),
    );
}

#[test]
fn extrapolation_range_limits_and_reverse_feedback_are_ordered() {
    let plan = scao_plan(None);
    let mut workspace = plan.workspace(&[0.0; 3]).unwrap();
    let mut requested = [0.0; 3];
    let mut demanded = [0.0; 3];
    let mut feedback = [0.0; 2];

    plan.process(
        DeformableMirrorCommandOutput {
            requested: &mut requested,
            demanded: &mut demanded,
            vdm_constraint_feedback: &mut feedback,
        },
        &mut workspace,
        DeformableMirrorCommandInput {
            vdm_command: &[0.5, -0.5],
            best_flat: &[0.1, 0.0, 0.0],
            pdm_offset: &[0.0; 3],
        },
    )
    .unwrap();

    assert!((requested[0] - 0.6).abs() <= f32::EPSILON);
    assert_eq!(requested[1..], [-0.5, 0.0]);
    assert_eq!(demanded, [0.25, -0.25, 0.0]);
    assert!((feedback[0] - 0.35).abs() <= f32::EPSILON);
    assert_eq!(feedback[1], -0.25);
    assert_eq!(workspace.current(), demanded);
}

#[test]
fn repeated_dm_ticks_slew_toward_one_latched_target() {
    let plan: DeformableMirrorCommandPlan<f32> = DeformableMirrorCommandPlan::new(
        1,
        1,
        1,
        Arc::from([1.0]),
        Arc::from([1.0]),
        Arc::from([1.0]),
        Arc::from([1.0]),
        Arc::from([-1.0]),
        Arc::from([1.0]),
        Some(Arc::from([0.2])),
    )
    .unwrap();
    let mut workspace = plan.workspace(&[0.0]).unwrap();
    let mut requested = [0.0];
    let mut demanded = [0.0];
    let mut feedback = [0.0];

    for expected in [0.2_f32, 0.4, 0.6, 0.8, 1.0] {
        plan.process(
            DeformableMirrorCommandOutput {
                requested: &mut requested,
                demanded: &mut demanded,
                vdm_constraint_feedback: &mut feedback,
            },
            &mut workspace,
            DeformableMirrorCommandInput {
                vdm_command: &[1.0],
                best_flat: &[0.0],
                pdm_offset: &[0.0],
            },
        )
        .unwrap();
        assert!((demanded[0] - expected).abs() <= f32::EPSILON);
        assert!((feedback[0] - (1.0 - expected)).abs() <= f32::EPSILON);
    }
}

#[test]
fn reverse_projection_is_not_inferred_from_the_forward_projection() {
    let plan: DeformableMirrorCommandPlan<f32> = DeformableMirrorCommandPlan::new(
        1,
        1,
        2,
        Arc::from([1.0]),
        Arc::from([1.0, 1.0]),
        Arc::from([0.25, 0.75]),
        Arc::from([1.0]),
        Arc::from([-0.25; 2]),
        Arc::from([0.25; 2]),
        None,
    )
    .unwrap();
    let mut workspace = plan.workspace(&[0.0; 2]).unwrap();
    let mut requested = [0.0; 2];
    let mut demanded = [0.0; 2];
    let mut feedback = [0.0];
    plan.process(
        DeformableMirrorCommandOutput {
            requested: &mut requested,
            demanded: &mut demanded,
            vdm_constraint_feedback: &mut feedback,
        },
        &mut workspace,
        DeformableMirrorCommandInput {
            vdm_command: &[1.0],
            best_flat: &[0.0; 2],
            pdm_offset: &[0.0; 2],
        },
    )
    .unwrap();

    assert_eq!(requested, [1.0, 1.0]);
    assert_eq!(demanded, [0.25, 0.25]);
    assert!((feedback[0] - 0.75).abs() <= f32::EPSILON);
}

#[test]
fn float64_uses_the_same_calculation() {
    let plan = DeformableMirrorCommandPlan::new(
        1,
        1,
        1,
        Arc::from([1.0_f64]),
        Arc::from([1.0_f64]),
        Arc::from([1.0_f64]),
        Arc::from([1.0_f64]),
        Arc::from([-1.0_f64]),
        Arc::from([1.0_f64]),
        None,
    )
    .unwrap();
    let mut workspace = plan.workspace(&[0.0]).unwrap();
    let mut requested = [0.0];
    let mut demanded = [0.0];
    let mut feedback = [0.0];
    plan.process(
        DeformableMirrorCommandOutput {
            requested: &mut requested,
            demanded: &mut demanded,
            vdm_constraint_feedback: &mut feedback,
        },
        &mut workspace,
        DeformableMirrorCommandInput {
            vdm_command: &[0.5],
            best_flat: &[0.0],
            pdm_offset: &[0.0],
        },
    )
    .unwrap();
    assert_eq!(demanded, [0.5]);
}

#[test]
fn reverse_feedback_includes_the_slew_limit() {
    let plan = DeformableMirrorCommandPlan::<f32>::new(
        1,
        1,
        1,
        Arc::from([1.0]),
        Arc::from([1.0]),
        Arc::from([1.0]),
        Arc::from([1.0]),
        Arc::from([-1.0]),
        Arc::from([1.0]),
        Some(Arc::from([0.2])),
    )
    .unwrap();
    let mut workspace = plan.workspace(&[0.0]).unwrap();
    let mut requested = [0.0];
    let mut demanded = [0.0];
    let mut feedback = [0.0];

    plan.process(
        DeformableMirrorCommandOutput {
            requested: &mut requested,
            demanded: &mut demanded,
            vdm_constraint_feedback: &mut feedback,
        },
        &mut workspace,
        DeformableMirrorCommandInput {
            vdm_command: &[0.75],
            best_flat: &[0.0],
            pdm_offset: &[0.0],
        },
    )
    .unwrap();

    assert_eq!(requested, [0.75]);
    assert!((workspace.current()[0] - 0.2).abs() <= f32::EPSILON);
    assert_eq!(demanded, [0.2_f32]);
    assert!((feedback[0] - (requested[0] - demanded[0])).abs() <= f32::EPSILON);
}

#[test]
fn reverse_feedback_includes_selected_power_and_inter_actuator_limits() {
    let plan = DeformableMirrorCommandPlan::new(
        2,
        2,
        2,
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([-10.0_f32; 2]),
        Arc::from([10.0_f32; 2]),
        None,
    )
    .unwrap()
    .with_inter_actuator_stroke_limit(
        InterActuatorStrokeLimitPlan::new(2, Arc::from([InterActuatorStrokeLimit::new(0, 1, 1.0)]))
            .unwrap(),
    )
    .unwrap()
    .with_command_power_limit(PdmCommandPowerLimitPlan::new(2, 0.125).unwrap())
    .unwrap();
    let mut workspace = plan.workspace(&[0.0; 2]).unwrap();
    let mut requested = [0.0; 2];
    let mut demanded = [0.0; 2];
    let mut feedback = [0.0; 2];

    plan.process(
        DeformableMirrorCommandOutput {
            requested: &mut requested,
            demanded: &mut demanded,
            vdm_constraint_feedback: &mut feedback,
        },
        &mut workspace,
        DeformableMirrorCommandInput {
            vdm_command: &[2.0, -2.0],
            best_flat: &[0.0; 2],
            pdm_offset: &[0.0; 2],
        },
    )
    .unwrap();

    assert_eq!(requested, [2.0, -2.0]);
    assert!((demanded[0] - 0.25).abs() <= 4.0 * f32::EPSILON);
    assert!((demanded[1] + 0.25).abs() <= 4.0 * f32::EPSILON);
    assert!((feedback[0] - 1.75).abs() <= 4.0 * f32::EPSILON);
    assert!((feedback[1] + 1.75).abs() <= 4.0 * f32::EPSILON);
}

#[test]
fn calibrated_pdm_null_mode_is_optimized_after_complete_composition() {
    let plan = DeformableMirrorCommandPlan::new(
        1,
        1,
        2,
        Arc::from([1.0_f32]),
        Arc::from([1.0_f32, -1.0]),
        Arc::from([0.5_f32, -0.5]),
        Arc::from([1.0_f32]),
        Arc::from([-3.0_f32; 2]),
        Arc::from([3.0_f32; 2]),
        None,
    )
    .unwrap()
    .with_stroke_optimization(
        PdmStrokeOptimizationPlan::new(
            2,
            1,
            Arc::from([0.0_f32; 2]),
            Arc::from([0.5_f32, 0.5]),
            Arc::from([1.0_f32, 1.0]),
        )
        .unwrap(),
    )
    .unwrap();
    let mut workspace = plan.workspace(&[0.0; 2]).unwrap();
    let mut requested = [0.0; 2];
    let mut demanded = [0.0; 2];
    let mut feedback = [0.0];

    plan.process(
        DeformableMirrorCommandOutput {
            requested: &mut requested,
            demanded: &mut demanded,
            vdm_constraint_feedback: &mut feedback,
        },
        &mut workspace,
        DeformableMirrorCommandInput {
            vdm_command: &[1.0],
            best_flat: &[1.0; 2],
            pdm_offset: &[0.0; 2],
        },
    )
    .unwrap();

    assert_eq!(requested, [2.0, 0.0]);
    assert_eq!(demanded, [1.0, -1.0]);
    assert_eq!(feedback, [0.0]);
}

#[test]
fn reverse_projection_failure_does_not_advance_pdm_state() {
    let plan = DeformableMirrorCommandPlan::new(
        1,
        1,
        2,
        Arc::from([1.0_f32]),
        Arc::from([1.0_f32, 1.0]),
        Arc::from([f32::MAX, f32::MAX]),
        Arc::from([1.0_f32]),
        Arc::from([-1.0_f32; 2]),
        Arc::from([1.0_f32; 2]),
        Some(Arc::from([0.2_f32; 2])),
    )
    .unwrap();
    let mut workspace = plan.workspace(&[0.0; 2]).unwrap();
    let mut requested = [0.0; 2];
    let mut demanded = [0.0; 2];
    let mut feedback = [0.0];

    let result = plan.process(
        DeformableMirrorCommandOutput {
            requested: &mut requested,
            demanded: &mut demanded,
            vdm_constraint_feedback: &mut feedback,
        },
        &mut workspace,
        DeformableMirrorCommandInput {
            vdm_command: &[1.0],
            best_flat: &[0.0; 2],
            pdm_offset: &[0.0; 2],
        },
    );

    assert_eq!(result, Err(DeformableMirrorCommandError::NonFiniteOutput));
    assert_eq!(workspace.current(), [0.0; 2]);
}
