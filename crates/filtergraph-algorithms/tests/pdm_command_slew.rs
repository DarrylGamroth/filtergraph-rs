//! PDM command slew and failure-atomicity tests.

use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmPlan, MatrixElement, PdmCommandError, PdmCommandInput, PdmCommandOutput,
    PdmCommandPlan,
};

fn process<T>(
    plan: &PdmCommandPlan<T>,
    workspace: &mut filtergraph_algorithms::PdmCommandWorkspace<T>,
    requested: &[T],
    demanded: &mut [T],
    feedback: &mut [T],
) where
    T: MatrixElement + num_traits::NumCast + 'static,
{
    plan.process(
        PdmCommandOutput {
            demanded,
            constraint_feedback: feedback,
        },
        workspace,
        PdmCommandInput { requested },
    )
    .unwrap();
}

#[test]
fn slew_matches_the_reference_ramp() {
    let plan = PdmCommandPlan::new(
        Arc::from([-1.0_f32, -1.0]),
        Arc::from([1.0_f32, 1.0]),
        Some(Arc::from([0.2_f32, 0.1])),
    )
    .unwrap();
    let mut workspace = plan.workspace(&[0.0, 0.0]).unwrap();
    let mut demanded = [0.0; 2];
    let mut feedback = [0.0; 2];

    process(
        &plan,
        &mut workspace,
        &[0.5, -0.5],
        &mut demanded,
        &mut feedback,
    );
    assert_eq!(demanded, [0.2, -0.1]);
    assert_eq!(workspace.current(), [0.2, -0.1]);

    process(
        &plan,
        &mut workspace,
        &[0.5, -0.5],
        &mut demanded,
        &mut feedback,
    );
    assert_eq!(demanded, [0.4, -0.2]);
}

#[test]
fn invalid_input_does_not_advance_state() {
    let plan = PdmCommandPlan::new(
        Arc::from([-1.0_f32]),
        Arc::from([1.0_f32]),
        Some(Arc::from([0.1_f32])),
    )
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
                requested: &[f32::NAN],
            },
        )
        .is_err()
    );
    assert_eq!(workspace, before);
}

#[test]
fn slew_state_tracks_the_range_limited_safe_target() {
    let plan = PdmCommandPlan::new(
        Arc::from([-0.25_f32]),
        Arc::from([0.25_f32]),
        Some(Arc::from([1.0_f32])),
    )
    .unwrap();
    let mut workspace = plan.workspace(&[0.0]).unwrap();
    let mut demanded = [0.0];
    let mut feedback = [0.0];

    process(&plan, &mut workspace, &[10.0], &mut demanded, &mut feedback);
    assert_eq!(workspace.current(), [0.25]);
    assert_eq!(demanded, [0.25]);
    assert_eq!(feedback, [9.75]);
}

#[test]
fn omitted_slew_reaches_the_target_across_the_complete_finite_range() {
    let plan = PdmCommandPlan::new(Arc::from([-f32::MAX]), Arc::from([f32::MAX]), None).unwrap();
    let mut workspace = plan.workspace(&[-f32::MAX]).unwrap();
    let mut demanded = [0.0];
    let mut feedback = [1.0];

    process(
        &plan,
        &mut workspace,
        &[f32::MAX],
        &mut demanded,
        &mut feedback,
    );

    assert_eq!(demanded, [f32::MAX]);
    assert_eq!(feedback, [0.0]);
    assert_eq!(workspace.current(), [f32::MAX]);
}

#[test]
fn nonfinite_constraint_feedback_does_not_commit_slew_state() {
    let plan = PdmCommandPlan::new(
        Arc::from([-f32::MAX]),
        Arc::from([f32::MAX]),
        Some(Arc::from([0.0])),
    )
    .unwrap();
    let mut workspace = plan.workspace(&[-f32::MAX]).unwrap();
    let mut demanded = [0.0];
    let mut feedback = [1.0];

    assert_eq!(
        plan.process(
            PdmCommandOutput {
                demanded: &mut demanded,
                constraint_feedback: &mut feedback,
            },
            &mut workspace,
            PdmCommandInput {
                requested: &[f32::MAX],
            },
        ),
        Err(PdmCommandError::NonFiniteOutput)
    );
    assert_eq!(workspace.current(), [-f32::MAX]);
    assert_eq!(feedback, [1.0]);
}
