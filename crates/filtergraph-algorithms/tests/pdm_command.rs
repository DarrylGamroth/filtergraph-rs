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
