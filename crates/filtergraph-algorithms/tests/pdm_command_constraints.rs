//! PDM command constraint-composition matrix tests.

use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmPlan, InterActuatorStrokeLimit, InterActuatorStrokeLimitPlan, PdmCommandInput,
    PdmCommandOutput, PdmCommandPlan, PdmCommandPlanError, PdmCommandPowerLimitPlan,
    PdmCommandWorkspaceError, PdmStrokeOptimizationPlan,
};

fn piston_optimization(pdm_size: usize) -> PdmStrokeOptimizationPlan<f32> {
    PdmStrokeOptimizationPlan::new(
        pdm_size,
        1,
        Arc::from(vec![0.0; pdm_size]),
        Arc::from(vec![1.0 / pdm_size as f32; pdm_size]),
        Arc::from(vec![1.0; pdm_size]),
    )
    .unwrap()
}

fn constrained_plan() -> PdmCommandPlan<f32> {
    PdmCommandPlan::new(Arc::from([-10.0; 3]), Arc::from([10.0; 3]), None)
        .unwrap()
        .with_inter_actuator_stroke_limit(
            InterActuatorStrokeLimitPlan::new(
                3,
                Arc::from([
                    InterActuatorStrokeLimit::new(0, 1, 0.5),
                    InterActuatorStrokeLimit::new(1, 2, 0.5),
                ]),
            )
            .unwrap(),
        )
        .unwrap()
        .with_command_power_limit(PdmCommandPowerLimitPlan::new(3, 1.0).unwrap())
        .unwrap()
}

#[test]
fn selected_limits_condition_state_output_and_feedback() {
    let plan = constrained_plan();
    let mut workspace = plan.workspace(&[0.0; 3]).unwrap();
    let mut demanded = [0.0; 3];
    let mut feedback = [0.0; 3];

    plan.process(
        PdmCommandOutput {
            demanded: &mut demanded,
            constraint_feedback: &mut feedback,
        },
        &mut workspace,
        PdmCommandInput {
            requested: &[2.0, -2.0, 1.0],
        },
    )
    .unwrap();

    assert_eq!(demanded, [0.25, -0.25, 0.125]);
    assert_eq!(workspace.current(), demanded);
    for index in 0..3 {
        assert!((feedback[index] - ([2.0, -2.0, 1.0][index] - demanded[index])).abs() <= 1.0e-6);
    }
}

#[test]
fn stroke_optimization_precedes_range_and_slew() {
    let plan = PdmCommandPlan::new(
        Arc::from([-0.5_f32; 2]),
        Arc::from([0.5_f32; 2]),
        Some(Arc::from([0.1_f32; 2])),
    )
    .unwrap()
    .with_stroke_optimization(piston_optimization(2))
    .unwrap();
    let mut workspace = plan.workspace(&[0.0; 2]).unwrap();
    let mut demanded = [0.0; 2];
    let mut feedback = [0.0; 2];

    plan.process(
        PdmCommandOutput {
            demanded: &mut demanded,
            constraint_feedback: &mut feedback,
        },
        &mut workspace,
        PdmCommandInput {
            requested: &[1.2, 0.8],
        },
    )
    .unwrap();

    assert!((demanded[0] - 0.1).abs() <= f32::EPSILON);
    assert!((demanded[1] + 0.1).abs() <= f32::EPSILON);
    assert_eq!(workspace.current(), demanded);
    assert!((feedback[0] - 1.1).abs() <= f32::EPSILON);
    assert!((feedback[1] - 0.9).abs() <= f32::EPSILON);
}

#[test]
fn stroke_optimization_composes_with_hard_constraints() {
    let plan = PdmCommandPlan::new(Arc::from([-3.0_f32; 2]), Arc::from([3.0_f32; 2]), None)
        .unwrap()
        .with_stroke_optimization(piston_optimization(2))
        .unwrap()
        .with_inter_actuator_stroke_limit(
            InterActuatorStrokeLimitPlan::new(
                2,
                Arc::from([InterActuatorStrokeLimit::new(0, 1, 0.5)]),
            )
            .unwrap(),
        )
        .unwrap()
        .with_command_power_limit(PdmCommandPowerLimitPlan::new(2, 0.125).unwrap())
        .unwrap();
    let mut workspace = plan.workspace(&[0.0; 2]).unwrap();
    let mut demanded = [0.0; 2];
    let mut feedback = [0.0; 2];

    plan.process(
        PdmCommandOutput {
            demanded: &mut demanded,
            constraint_feedback: &mut feedback,
        },
        &mut workspace,
        PdmCommandInput {
            requested: &[2.0, 0.0],
        },
    )
    .unwrap();

    assert_eq!(demanded, [0.25, -0.25]);
    assert_eq!(feedback, [1.75, 0.25]);
}

#[test]
fn command_power_limit_handles_an_unviolated_inter_actuator_graph() {
    let plan = constrained_plan();
    let mut workspace = plan.workspace(&[0.0; 3]).unwrap();
    let mut demanded = [0.0; 3];
    let mut feedback = [0.0; 3];
    plan.process(
        PdmCommandOutput {
            demanded: &mut demanded,
            constraint_feedback: &mut feedback,
        },
        &mut workspace,
        PdmCommandInput {
            requested: &[2.0, 2.0, 2.0],
        },
    )
    .unwrap();

    let expected = 1.0 / 3.0_f32.sqrt();
    assert!(
        demanded
            .iter()
            .all(|value| (*value - expected).abs() <= 4.0 * f32::EPSILON)
    );
    assert!(demanded.iter().map(|value| value * value).sum::<f32>() <= 1.0 + 8.0 * f32::EPSILON);
}

#[test]
fn selected_limits_hold_across_repeated_commands() {
    let plan = PdmCommandPlan::new(Arc::from([-1.0_f32; 2]), Arc::from([1.0_f32; 2]), None)
        .unwrap()
        .with_stroke_optimization(piston_optimization(2))
        .unwrap()
        .with_inter_actuator_stroke_limit(
            InterActuatorStrokeLimitPlan::new(
                2,
                Arc::from([InterActuatorStrokeLimit::new(0, 1, 0.1)]),
            )
            .unwrap(),
        )
        .unwrap()
        .with_command_power_limit(PdmCommandPowerLimitPlan::new(2, 0.02).unwrap())
        .unwrap();
    let mut workspace = plan.workspace(&[0.0; 2]).unwrap();
    let mut demanded = [0.0; 2];
    let mut feedback = [0.0; 2];

    for _ in 0..32 {
        plan.process(
            PdmCommandOutput {
                demanded: &mut demanded,
                constraint_feedback: &mut feedback,
            },
            &mut workspace,
            PdmCommandInput {
                requested: &[0.9, -0.7],
            },
        )
        .unwrap();
        assert!((demanded[0] - demanded[1]).abs() <= 0.1 + 8.0 * f32::EPSILON);
        assert!(
            demanded.iter().map(|value| value * value).sum::<f32>() <= 0.02 + 8.0 * f32::EPSILON
        );
    }
}

#[test]
fn preparation_requires_matching_extents_and_ranges_containing_zero() {
    let plan = PdmCommandPlan::new(Arc::from([-1.0_f32]), Arc::from([1.0]), None).unwrap();
    assert_eq!(
        plan.clone()
            .with_command_power_limit(PdmCommandPowerLimitPlan::new(2, 1.0).unwrap()),
        Err(PdmCommandPlanError::CommandPowerLimitExtent)
    );
    assert_eq!(
        plan.with_inter_actuator_stroke_limit(
            InterActuatorStrokeLimitPlan::new(
                2,
                Arc::from([InterActuatorStrokeLimit::new(0, 1, 1.0)])
            )
            .unwrap()
        ),
        Err(PdmCommandPlanError::InterActuatorStrokeLimitExtent)
    );
    assert_eq!(
        PdmCommandPlan::new(Arc::from([-1.0_f32]), Arc::from([1.0]), None)
            .unwrap()
            .with_stroke_optimization(piston_optimization(2)),
        Err(PdmCommandPlanError::StrokeOptimizationExtent)
    );

    let positive_range = PdmCommandPlan::new(Arc::from([0.1_f32]), Arc::from([1.0]), None).unwrap();
    assert_eq!(
        positive_range.with_command_power_limit(PdmCommandPowerLimitPlan::new(1, 1.0).unwrap()),
        Err(PdmCommandPlanError::ConstraintOriginOutsideRange)
    );
}

#[test]
fn initial_state_must_satisfy_every_selected_limit() {
    let power = PdmCommandPlan::new(Arc::from([-2.0_f32; 2]), Arc::from([2.0; 2]), None)
        .unwrap()
        .with_command_power_limit(PdmCommandPowerLimitPlan::new(2, 1.0).unwrap())
        .unwrap();
    assert_eq!(
        power.workspace(&[1.0, 1.0]),
        Err(PdmCommandWorkspaceError::InitialCurrentExceedsCommandPower)
    );

    let inter_actuator = PdmCommandPlan::new(Arc::from([-2.0_f32; 2]), Arc::from([2.0; 2]), None)
        .unwrap()
        .with_inter_actuator_stroke_limit(
            InterActuatorStrokeLimitPlan::new(
                2,
                Arc::from([InterActuatorStrokeLimit::new(0, 1, 0.5)]),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        inter_actuator.workspace(&[1.0, 0.0]),
        Err(PdmCommandWorkspaceError::InitialCurrentExceedsInterActuatorStroke)
    );
}

#[test]
fn invalid_input_preserves_all_conditioning_state() {
    let plan = constrained_plan();
    let mut workspace = plan.workspace(&[0.0; 3]).unwrap();
    let before = workspace.clone();
    let mut demanded = [0.0; 3];
    let mut feedback = [0.0; 3];

    assert!(
        plan.process(
            PdmCommandOutput {
                demanded: &mut demanded,
                constraint_feedback: &mut feedback,
            },
            &mut workspace,
            PdmCommandInput {
                requested: &[f32::NAN, 0.0, 0.0],
            },
        )
        .is_err()
    );
    assert_eq!(workspace, before);
}

#[test]
fn failed_stroke_optimization_preserves_command_state() {
    let optimization = PdmStrokeOptimizationPlan::new(
        1,
        1,
        Arc::from([-f32::MAX]),
        Arc::from([1.0_f32]),
        Arc::from([1.0_f32]),
    )
    .unwrap();
    let plan = PdmCommandPlan::new(Arc::from([-f32::MAX]), Arc::from([f32::MAX]), None)
        .unwrap()
        .with_stroke_optimization(optimization)
        .unwrap();
    let mut workspace = plan.workspace(&[0.0]).unwrap();
    let before = workspace.clone();
    let mut demanded = [0.0];
    let mut feedback = [0.0];

    assert!(
        plan.process(
            PdmCommandOutput {
                demanded: &mut demanded,
                constraint_feedback: &mut feedback,
            },
            &mut workspace,
            PdmCommandInput {
                requested: &[f32::MAX],
            },
        )
        .is_err()
    );
    assert_eq!(workspace.current(), before.current());
}
