//! Thompson four-image SCC calibration-response tests.

use filtergraph_algorithms::schemas::{CALIBRATED_PIXELS_V1, SCC_PUSH_PULL_RESPONSE_V1};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmPlan, SccPushPullImageKind, SccPushPullResponseError,
    SccPushPullResponseF32, SccPushPullResponseF32Configuration, SccPushPullResponseInput,
    SccPushPullResponsePlan, SccPushPullResponsePlanError,
};

#[test]
fn computes_the_thompson_four_image_double_difference() {
    let plan = SccPushPullResponsePlan::new(3).unwrap();
    let mut workspace = plan.workspace();
    let input = SccPushPullResponseInput::new(
        &[10.0_f64, 5.0, -1.0],
        &[4.0, 7.0, -2.0],
        &[3.0, -1.0, 8.0],
        &[2.0, 2.0, 3.0],
    );
    let mut response = [0.0; 3];

    assert_eq!(plan.pixel_count(), 3);
    assert_eq!(workspace.pixel_count(), 3);
    assert_eq!(input.push_fringed(), [10.0, 5.0, -1.0]);
    assert_eq!(input.push_unfringed(), [4.0, 7.0, -2.0]);
    assert_eq!(input.pull_fringed(), [3.0, -1.0, 8.0]);
    assert_eq!(input.pull_unfringed(), [2.0, 2.0, 3.0]);

    plan.process(&mut response, &mut workspace, input).unwrap();

    assert_eq!(response, [5.0, 1.0, -4.0]);
}

#[test]
fn float_representations_compute_the_same_response() {
    let plan32 = SccPushPullResponsePlan::new(2).unwrap();
    let plan64 = SccPushPullResponsePlan::new(2).unwrap();
    let mut workspace32 = plan32.workspace();
    let mut workspace64 = plan64.workspace();
    let mut response32 = [0.0_f32; 2];
    let mut response64 = [0.0_f64; 2];

    plan32
        .process(
            &mut response32,
            &mut workspace32,
            SccPushPullResponseInput::new(&[4.0, 2.0], &[1.0, 3.0], &[0.5, -1.0], &[1.5, 2.0]),
        )
        .unwrap();
    plan64
        .process(
            &mut response64,
            &mut workspace64,
            SccPushPullResponseInput::new(&[4.0, 2.0], &[1.0, 3.0], &[0.5, -1.0], &[1.5, 2.0]),
        )
        .unwrap();

    assert_eq!(response32.map(f64::from), response64);
}

#[test]
fn rejects_empty_preparation() {
    assert_eq!(
        SccPushPullResponsePlan::<f32>::new(0),
        Err(SccPushPullResponsePlanError::EmptyResponse)
    );
}

#[test]
fn rejects_each_incompatible_image_extent_before_publication() {
    let plan = SccPushPullResponsePlan::new(1).unwrap();
    let mut workspace = plan.workspace();
    let mut response = [9.0_f32];
    let cases = [
        (
            SccPushPullResponseInput::new(&[], &[0.0], &[0.0], &[0.0]),
            SccPushPullImageKind::PushFringed,
        ),
        (
            SccPushPullResponseInput::new(&[0.0], &[], &[0.0], &[0.0]),
            SccPushPullImageKind::PushUnfringed,
        ),
        (
            SccPushPullResponseInput::new(&[0.0], &[0.0], &[], &[0.0]),
            SccPushPullImageKind::PullFringed,
        ),
        (
            SccPushPullResponseInput::new(&[0.0], &[0.0], &[0.0], &[]),
            SccPushPullImageKind::PullUnfringed,
        ),
    ];

    for (input, image) in cases {
        assert_eq!(
            plan.process(&mut response, &mut workspace, input),
            Err(SccPushPullResponseError::ImageExtent {
                image,
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(response, [9.0]);
    }
}

#[test]
fn rejects_incompatible_output_and_workspace_before_publication() {
    let plan = SccPushPullResponsePlan::new(1).unwrap();
    let other_plan = SccPushPullResponsePlan::new(2).unwrap();
    let mut workspace = plan.workspace();
    let mut other_workspace = other_plan.workspace();
    let input = SccPushPullResponseInput::new(&[1.0], &[0.0], &[-1.0], &[0.0]);
    let mut empty = [];

    assert_eq!(
        plan.process(&mut empty, &mut workspace, input),
        Err(SccPushPullResponseError::ResponseExtent {
            expected: 1,
            actual: 0,
        })
    );

    let mut response = [9.0];
    assert_eq!(
        plan.process(&mut response, &mut other_workspace, input),
        Err(SccPushPullResponseError::WorkspaceExtent)
    );
    assert_eq!(response, [9.0]);
}

#[test]
fn rejects_non_finite_images_and_calculations_without_publishing() {
    let plan = SccPushPullResponsePlan::new(2).unwrap();
    let mut workspace = plan.workspace();
    let mut response = [8.0_f32, 9.0];

    assert_eq!(
        plan.process(
            &mut response,
            &mut workspace,
            SccPushPullResponseInput::new(&[1.0, 2.0], &[0.0, 0.0], &[0.0, f32::NAN], &[0.0, 0.0]),
        ),
        Err(SccPushPullResponseError::NonFiniteImage {
            image: SccPushPullImageKind::PullFringed,
            pixel: 1,
        })
    );
    assert_eq!(response, [8.0, 9.0]);

    assert_eq!(
        plan.process(
            &mut response,
            &mut workspace,
            SccPushPullResponseInput::new(
                &[1.0, f32::MAX],
                &[0.0, -f32::MAX],
                &[0.0, 0.0],
                &[0.0, 0.0],
            ),
        ),
        Err(SccPushPullResponseError::NonFiniteCalculation { pixel: 1 })
    );
    assert_eq!(response, [8.0, 9.0]);

    plan.process(
        &mut response,
        &mut workspace,
        SccPushPullResponseInput::new(&[2.0, 3.0], &[1.0, 1.0], &[0.0, 1.0], &[0.0, 0.0]),
    )
    .unwrap();
    assert_eq!(response, [1.0, 1.0]);
}

#[test]
fn declaration_maps_four_named_images_to_one_atomic_response() {
    let mut declared =
        SccPushPullResponseF32::prepare(SccPushPullResponseF32Configuration { pixel_count: 3 })
            .unwrap();
    assert_eq!(SccPushPullResponseF32::NAME, "scc-push-pull-response-f32");
    assert_eq!(SccPushPullResponseF32::PORTS.len(), 5);
    assert_eq!(SccPushPullResponseF32::PORTS[0].name(), "push-fringed");
    assert_eq!(SccPushPullResponseF32::PORTS[1].name(), "push-unfringed");
    assert_eq!(SccPushPullResponseF32::PORTS[2].name(), "pull-fringed");
    assert_eq!(SccPushPullResponseF32::PORTS[3].name(), "pull-unfringed");
    assert_eq!(SccPushPullResponseF32::PORTS[4].name(), "response");
    assert_eq!(
        SccPushPullResponseF32::PORTS[4].metadata_source(),
        Some("push-fringed")
    );
    for format in &declared.formats()[..4] {
        assert_eq!(format.shape(), &[3]);
        assert_eq!(format.schema(), CALIBRATED_PIXELS_V1);
    }
    assert_eq!(declared.formats()[4].shape(), &[3]);
    assert_eq!(declared.formats()[4].schema(), SCC_PUSH_PULL_RESPONSE_V1);

    let push_fringed = [10.0_f32, 5.0, -1.0];
    let push_unfringed = [4.0_f32, 7.0, -2.0];
    let pull_fringed = [3.0_f32, -1.0, 8.0];
    let pull_unfringed = [2.0_f32, 2.0, 3.0];
    let mut response = [0.0_f32; 3];
    SccPushPullResponseF32::process(
        &mut declared,
        &push_fringed,
        &push_unfringed,
        &pull_fringed,
        &pull_unfringed,
        &mut response,
    )
    .unwrap();
    assert_eq!(response, [5.0, 1.0, -4.0]);
}
