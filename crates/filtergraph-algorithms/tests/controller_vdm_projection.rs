//! Controller/VDM forward-command and reverse-feedback projection fixtures.

use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmPlan, ControllerProjection, ControllerToVdmPlan, NoWorkspace,
    VdmFeedbackToControllerPlan,
};

#[test]
fn low_and_high_order_commands_sum_in_complementary_vdm_subspaces() {
    let low = ControllerProjection::new(
        2,
        Arc::from([1.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0]),
        Arc::from([1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    .unwrap();
    let high = ControllerProjection::new(
        1,
        Arc::from([0.0_f32, 0.0, 1.0]),
        Arc::from([0.0_f32, 0.0, 1.0]),
    )
    .unwrap();
    let plan = ControllerToVdmPlan::new(3, Arc::from([low, high])).unwrap();
    let mut workspace = plan.workspace();
    let mut command = [0.0; 3];

    plan.process(&mut command, &mut workspace, &[2.0, -3.0, 4.0])
        .unwrap();

    assert_eq!(command, [2.0, -3.0, 4.0]);
}

#[test]
fn downstream_constraint_feedback_is_projected_to_each_controller() {
    let low = ControllerProjection::new(
        2,
        Arc::from([1.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0]),
        Arc::from([1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    .unwrap();
    let high = ControllerProjection::new(
        1,
        Arc::from([0.0_f32, 0.0, 1.0]),
        Arc::from([0.0_f32, 0.0, 1.0]),
    )
    .unwrap();
    let plan = VdmFeedbackToControllerPlan::new(3, Arc::from([low, high])).unwrap();
    let mut workspace = NoWorkspace;
    let mut feedback = [0.0; 3];

    plan.process(&mut feedback, &mut workspace, &[0.25, -0.5, 0.75])
        .unwrap();

    assert_eq!(feedback, [0.25, -0.5, 0.75]);
}

#[test]
fn forward_and_reverse_projections_are_not_inferred_from_each_other() {
    let controller =
        ControllerProjection::new(1, Arc::from([2.0_f64, 3.0]), Arc::from([5.0_f64, 7.0])).unwrap();
    let projections = Arc::from([controller]);
    let command_plan = ControllerToVdmPlan::new(2, Arc::clone(&projections)).unwrap();
    let feedback_plan = VdmFeedbackToControllerPlan::new(2, projections).unwrap();
    let mut command_workspace = command_plan.workspace();
    let mut feedback_workspace = NoWorkspace;
    let mut command = [0.0; 2];
    let mut feedback = [0.0; 1];

    command_plan
        .process(&mut command, &mut command_workspace, &[11.0])
        .unwrap();
    feedback_plan
        .process(&mut feedback, &mut feedback_workspace, &[13.0, 17.0])
        .unwrap();

    assert_eq!(command, [22.0, 33.0]);
    assert_eq!(feedback, [184.0]);
}

#[test]
fn projection_admission_rejects_incompatible_controller_domains_and_lengths() {
    use filtergraph_algorithms::ControllerProjectionError::*;
    for (command_count, forward, reverse, expected, diagnostic) in [
        (
            0,
            &[1.0_f32][..],
            &[1.0][..],
            EmptyDimension,
            "dimensions must be nonempty",
        ),
        (
            2,
            &[1.0][..],
            &[1.0][..],
            ProjectionExtent,
            "projection extent is invalid",
        ),
        (
            1,
            &[1.0, f32::NAN][..],
            &[1.0; 2][..],
            NonFiniteConfiguration,
            "projections must be finite",
        ),
    ] {
        let error =
            ControllerProjection::new(command_count, Arc::from(forward), Arc::from(reverse))
                .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(diagnostic), "{error}");
    }
    let none: Arc<[ControllerProjection<f32>]> = Arc::from([]);
    let error = ControllerToVdmPlan::new(2, none).unwrap_err();
    assert_eq!(error, EmptyControllerSet);
    assert!(error.to_string().contains("at least one controller"));
    let valid =
        ControllerProjection::new(2, Arc::from([1.0_f32; 2]), Arc::from([1.0_f32; 2])).unwrap();
    let error = ControllerToVdmPlan::new(usize::MAX, Arc::from([valid])).unwrap_err();
    assert_eq!(error, ExtentOverflow);
    assert!(error.to_string().contains("extent overflow"));
}
