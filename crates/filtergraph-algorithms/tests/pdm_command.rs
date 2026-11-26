//! Physical-DM command conditioning and feedback tests.

use std::sync::Arc;

use filtergraph_algorithms::{AlgorithmPlan, PdmCommandInput, PdmCommandOutput, PdmCommandPlan};

#[test]
fn requested_command_is_conditioned_and_reported_as_constraint_feedback() {
    let plan =
        PdmCommandPlan::new(Arc::from([-0.25_f32; 2]), Arc::from([0.25_f32; 2]), None).unwrap();
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
            requested: &[0.45, -0.3],
        },
    )
    .unwrap();

    assert_eq!(demanded, [0.25, -0.25]);
    assert!((feedback[0] - 0.2).abs() <= f32::EPSILON);
    assert!((feedback[1] + 0.05).abs() <= f32::EPSILON);
}

#[test]
fn each_pdm_owns_independent_device_rate_state() {
    let plan = PdmCommandPlan::new(
        Arc::from([-1.0_f32]),
        Arc::from([1.0_f32]),
        Some(Arc::from([0.2_f32])),
    )
    .unwrap();
    let mut workspace = plan.workspace(&[0.0]).unwrap();
    let mut demanded = [0.0];
    let mut feedback = [0.0];

    for expected in [0.2_f32, 0.4, 0.6] {
        plan.process(
            PdmCommandOutput {
                demanded: &mut demanded,
                constraint_feedback: &mut feedback,
            },
            &mut workspace,
            PdmCommandInput { requested: &[1.0] },
        )
        .unwrap();
        assert!((workspace.current()[0] - expected).abs() <= f32::EPSILON);
        assert!((feedback[0] - (1.0 - demanded[0])).abs() <= f32::EPSILON);
    }
}

#[test]
fn a_request_equal_to_a_bound_keeps_its_zero_sign() {
    macro_rules! check {
        ($float:ty) => {{
            let plan =
                PdmCommandPlan::<$float>::new(Arc::from([0.0, -1.0]), Arc::from([1.0, -0.0]), None)
                    .unwrap();
            let mut workspace = plan.workspace(&[0.0; 2]).unwrap();
            let request = [-0.0, 0.0];
            let mut demanded = [1.0; 2];
            let mut feedback = [1.0; 2];
            plan.process(
                PdmCommandOutput {
                    demanded: &mut demanded,
                    constraint_feedback: &mut feedback,
                },
                &mut workspace,
                PdmCommandInput {
                    requested: &request,
                },
            )
            .unwrap();
            assert_eq!(
                demanded.map(<$float>::to_bits),
                request.map(<$float>::to_bits)
            );
            assert_eq!(feedback.map(<$float>::to_bits), [0; 2]);
        }};
    }
    check!(f32);
    check!(f64);
}

#[test]
fn physical_actuator_preparation_reports_initial_state_and_selected_limit_failures() {
    use filtergraph_algorithms::{
        InterActuatorStrokeLimit, InterActuatorStrokeLimitPlan, PdmCommandPlanError as Prep,
        PdmCommandPowerLimitPlan, PdmCommandWorkspaceError as State,
    };

    let plan = PdmCommandPlan::new(Arc::from([-1.0_f32; 2]), Arc::from([1.0; 2]), None).unwrap();
    for (candidate, expected, detail) in [
        (
            plan.clone()
                .with_command_power_limit(PdmCommandPowerLimitPlan::new(1, 1.0).unwrap()),
            Prep::CommandPowerLimitExtent,
            "command-power limit extent",
        ),
        (
            plan.clone().with_inter_actuator_stroke_limit(
                InterActuatorStrokeLimitPlan::new(
                    3,
                    Arc::from([InterActuatorStrokeLimit::new(0, 1, 0.25)]),
                )
                .unwrap(),
            ),
            Prep::InterActuatorStrokeLimitExtent,
            "inter-actuator stroke-limit extent",
        ),
    ] {
        let error = candidate.unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
    }
    let power = plan
        .clone()
        .with_command_power_limit(PdmCommandPowerLimitPlan::new(2, 0.25).unwrap())
        .unwrap();
    let stroke = plan
        .clone()
        .with_inter_actuator_stroke_limit(
            InterActuatorStrokeLimitPlan::new(
                2,
                Arc::from([InterActuatorStrokeLimit::new(0, 1, 0.25)]),
            )
            .unwrap(),
        )
        .unwrap();
    for (plan, initial, expected, detail) in [
        (
            &plan,
            &[0.0][..],
            State::InitialCurrentExtent,
            "command extent",
        ),
        (
            &plan,
            &[f32::NAN, 0.0][..],
            State::NonFiniteInitialCurrent,
            "command must be finite",
        ),
        (
            &plan,
            &[0.0, 2.0][..],
            State::InitialCurrentOutsideRange,
            "violates actuator limits",
        ),
        (
            &power,
            &[1.0, 1.0][..],
            State::InitialCurrentExceedsCommandPower,
            "exceeds the command-power limit",
        ),
        (
            &stroke,
            &[0.5, 0.0][..],
            State::InitialCurrentExceedsInterActuatorStroke,
            "exceeds an inter-actuator stroke limit",
        ),
    ] {
        let error = plan.workspace(initial).unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
        assert_eq!(plan.workspace(&[0.0, 0.0]).unwrap().current(), [0.0; 2]);
    }
}

#[test]
fn rejected_actuator_tick_preserves_slew_state_and_both_outputs() {
    use filtergraph_algorithms::PdmCommandError::*;
    let plan = PdmCommandPlan::new(
        Arc::from([-1.0_f32; 2]),
        Arc::from([1.0; 2]),
        Some(Arc::from([0.2; 2])),
    )
    .unwrap();
    let other = PdmCommandPlan::new(Arc::from([-1.0_f32]), Arc::from([1.0]), None).unwrap();
    let mut workspace = plan.workspace(&[0.1, -0.1]).unwrap();
    let mut wrong_workspace = other.workspace(&[0.0]).unwrap();
    let mut demanded = [19.0; 2];
    let mut feedback = [19.0; 2];
    for (input, nout, wrong, expected, detail) in [
        (&[1.0][..], 2, false, InputExtent, "input extent"),
        (&[1.0, -1.0][..], 1, false, OutputExtent, "output extent"),
        (
            &[1.0, -1.0][..],
            2,
            true,
            WorkspaceExtent,
            "workspace extent",
        ),
        (
            &[1.0, f32::NAN][..],
            2,
            false,
            NonFiniteInput,
            "input must be finite",
        ),
    ] {
        let storage = if wrong {
            &mut wrong_workspace
        } else {
            &mut workspace
        };
        let error = plan
            .process(
                PdmCommandOutput {
                    demanded: &mut demanded[..nout],
                    constraint_feedback: &mut feedback,
                },
                storage,
                PdmCommandInput { requested: input },
            )
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
        assert_eq!((demanded, feedback), ([19.0; 2], [19.0; 2]));
        assert_eq!(workspace.current(), [0.1, -0.1]);
    }
    plan.process(
        PdmCommandOutput {
            demanded: &mut demanded,
            constraint_feedback: &mut feedback,
        },
        &mut workspace,
        PdmCommandInput {
            requested: &[1.0, -1.0],
        },
    )
    .unwrap();
    assert_eq!(demanded, [0.3, -0.3]);
    assert_eq!(feedback, [0.7, -0.7]);
}
