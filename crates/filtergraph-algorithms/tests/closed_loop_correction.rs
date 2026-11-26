//! Closed-loop correction recurrence, feedback, and hidden-mode tests.

use filtergraph_algorithms::{
    AlgorithmPlan, AlgorithmProperties, ClosedLoopCorrectionError, ClosedLoopCorrectionInput,
    ClosedLoopCorrectionPlan, ClosedLoopCorrectionPlanError, PropertyAssignment, PropertyValue,
};
use std::sync::Arc;

#[test]
fn preparation_validates_state_coefficients_and_hidden_projections() {
    assert_eq!(
        ClosedLoopCorrectionPlan::new(0.5_f32, 0.75, 0.75, 0),
        Err(ClosedLoopCorrectionPlanError::EmptyState)
    );
    assert_eq!(
        ClosedLoopCorrectionPlan::new(0.5_f32, 0.75, f32::NAN, 2),
        Err(ClosedLoopCorrectionPlanError::InvalidAntiWindupGain)
    );
    assert_eq!(
        ClosedLoopCorrectionPlan::with_hidden_modes(
            0.5_f32,
            0.75,
            0.75,
            2,
            1,
            1.0,
            Arc::from([1.0]),
            Arc::from([1.0, 1.0]),
        ),
        Err(ClosedLoopCorrectionPlanError::HiddenModeProjectionExtent)
    );
}

#[test]
fn signed_loop_gain_applies_the_error_convention_in_clwc() {
    let plan = ClosedLoopCorrectionPlan::new(-0.3_f32, 0.99, 1.0, 2).unwrap();
    let mut workspace = plan.workspace();
    let mut output = [0.0; 2];

    plan.process(
        &mut output,
        &mut workspace,
        ClosedLoopCorrectionInput {
            x: &[2.0, -4.0],
            constraint_feedback: None,
        },
    )
    .unwrap();

    assert_eq!(output, [-0.6, 1.2]);
}

#[test]
fn heart_pole_scales_previous_constraint_feedback() {
    let plan = ClosedLoopCorrectionPlan::new(0.5_f32, 0.75, 0.75, 2).unwrap();
    let mut workspace = plan.workspace();
    let mut output = [0.0; 2];

    plan.process(
        &mut output,
        &mut workspace,
        ClosedLoopCorrectionInput {
            x: &[1.0, -1.0],
            constraint_feedback: None,
        },
    )
    .unwrap();
    assert_eq!(output, [0.5, -0.5]);
    assert_eq!(workspace.state(), [0.0, 0.0]);

    plan.process(
        &mut output,
        &mut workspace,
        ClosedLoopCorrectionInput {
            x: &[0.0, 0.0],
            constraint_feedback: Some(&[0.2, -0.4]),
        },
    )
    .unwrap();
    assert!((workspace.state()[0] - 0.35).abs() <= f32::EPSILON);
    assert!((workspace.state()[1] + 0.2).abs() <= f32::EPSILON);
    assert!((output[0] - 0.2625).abs() <= f32::EPSILON);
    assert!((output[1] + 0.15).abs() <= f32::EPSILON);
}

#[test]
fn controller_can_subtract_complete_constraint_feedback() {
    let plan = ClosedLoopCorrectionPlan::new(0.5_f32, 0.75, 1.0, 1).unwrap();
    let mut workspace = plan.workspace();
    let mut output = [0.0];

    plan.process(
        &mut output,
        &mut workspace,
        ClosedLoopCorrectionInput {
            x: &[1.0],
            constraint_feedback: None,
        },
    )
    .unwrap();
    plan.process(
        &mut output,
        &mut workspace,
        ClosedLoopCorrectionInput {
            x: &[0.0],
            constraint_feedback: Some(&[0.2]),
        },
    )
    .unwrap();

    assert!((workspace.state()[0] - 0.3).abs() <= f32::EPSILON);
    assert!((output[0] - 0.225).abs() <= f32::EPSILON);
}

#[test]
fn hidden_modes_are_removed_from_the_realized_shape() {
    let plan = ClosedLoopCorrectionPlan::with_hidden_modes(
        1.0_f32,
        1.0,
        0.0,
        2,
        1,
        1.0,
        Arc::from([1.0, 1.0]),
        Arc::from([0.5, 0.5]),
    )
    .unwrap();
    let mut workspace = plan.workspace();
    let mut output = [0.0; 2];

    plan.process(
        &mut output,
        &mut workspace,
        ClosedLoopCorrectionInput {
            x: &[1.0, 3.0],
            constraint_feedback: None,
        },
    )
    .unwrap();
    plan.process(
        &mut output,
        &mut workspace,
        ClosedLoopCorrectionInput {
            x: &[0.0, 0.0],
            constraint_feedback: Some(&[0.0, 0.0]),
        },
    )
    .unwrap();

    assert_eq!(workspace.state(), [-1.0, 1.0]);
    assert_eq!(output, [-1.0, 1.0]);
}

#[test]
fn feedback_errors_preserve_state_and_last_output() {
    let plan = ClosedLoopCorrectionPlan::new(0.5_f32, 1.0, 1.0, 1).unwrap();
    let mut workspace = plan.workspace();
    let mut output = [0.0];
    plan.process(
        &mut output,
        &mut workspace,
        ClosedLoopCorrectionInput {
            x: &[2.0],
            constraint_feedback: None,
        },
    )
    .unwrap();
    let state = workspace.state().to_vec();
    let last_output = workspace.last_output().unwrap().to_vec();

    assert_eq!(
        plan.process(
            &mut output,
            &mut workspace,
            ClosedLoopCorrectionInput {
                x: &[2.0],
                constraint_feedback: None,
            },
        ),
        Err(ClosedLoopCorrectionError::MissingFeedback)
    );
    assert_eq!(workspace.state(), state);
    assert_eq!(workspace.last_output().unwrap(), last_output);
}

#[test]
fn plan_supports_float64() {
    let plan = ClosedLoopCorrectionPlan::new(0.5_f64, 0.75, 0.75, 2).unwrap();
    let mut workspace = plan.workspace();
    let mut output = [0.0_f64; 2];
    plan.process(
        &mut output,
        &mut workspace,
        ClosedLoopCorrectionInput {
            x: &[2.0, -2.0],
            constraint_feedback: None,
        },
    )
    .unwrap();
    assert_eq!(output, [1.0, -1.0]);
}

#[test]
fn runtime_properties_preserve_dimensions_and_hidden_mode_projections() {
    let plan = ClosedLoopCorrectionPlan::with_hidden_modes(
        0.5_f32,
        0.75,
        0.8,
        2,
        1,
        0.25,
        Arc::from([1.0, 2.0]),
        Arc::from([3.0, 4.0]),
    )
    .unwrap();
    let shape_to_hidden = plan.shape_to_hidden().as_ptr();
    let hidden_to_shape = plan.hidden_to_shape().as_ptr();

    let replacement = plan
        .prepare_properties(&[
            PropertyAssignment {
                id: 0,
                value: PropertyValue::Float(-0.4),
            },
            PropertyAssignment {
                id: 1,
                value: PropertyValue::Float(0.9),
            },
            PropertyAssignment {
                id: 2,
                value: PropertyValue::Float(1.0),
            },
            PropertyAssignment {
                id: 3,
                value: PropertyValue::Float(0.5),
            },
        ])
        .unwrap();

    assert_eq!(replacement.gain(), -0.4);
    assert_eq!(replacement.pole(), 0.9);
    assert_eq!(replacement.anti_windup_gain(), 1.0);
    assert_eq!(replacement.hidden_mode_gain(), 0.5);
    assert_eq!(replacement.state_length(), 2);
    assert_eq!(replacement.hidden_mode_count(), 1);
    assert_eq!(replacement.shape_to_hidden().as_ptr(), shape_to_hidden);
    assert_eq!(replacement.hidden_to_shape().as_ptr(), hidden_to_shape);
}

#[test]
fn feedback_admission_reports_bad_phase_extent_and_values_without_advancing_state() {
    use ClosedLoopCorrectionError::*;
    let plan = ClosedLoopCorrectionPlan::new(0.5_f32, 1.0, 1.0, 2).unwrap();
    let mut workspace = plan.workspace();
    let mut output = [17.0; 2];
    let command = [2.0; 2];
    let error = plan
        .process(
            &mut output,
            &mut workspace,
            ClosedLoopCorrectionInput {
                x: &command,
                constraint_feedback: Some(&[0.0; 2]),
            },
        )
        .unwrap_err();
    assert_eq!(error, UnexpectedFeedback);
    assert!(error.to_string().contains("preceded the first correction"));
    assert_eq!(output, [17.0; 2]);
    plan.process(
        &mut output,
        &mut workspace,
        ClosedLoopCorrectionInput {
            x: &command,
            constraint_feedback: None,
        },
    )
    .unwrap();
    let committed = workspace.state().to_vec();
    let published = output;
    for (input, feedback, nout, expected, detail) in [
        (
            &command[..1],
            Some(&[0.0; 2][..]),
            2,
            InputExtent,
            "input extent",
        ),
        (
            &command[..],
            Some(&[0.0; 2][..]),
            1,
            OutputExtent,
            "output extent",
        ),
        (
            &command[..],
            None,
            2,
            MissingFeedback,
            "feedback is missing",
        ),
        (
            &command[..],
            Some(&[0.0][..]),
            2,
            FeedbackExtent,
            "feedback extent",
        ),
        (
            &command[..],
            Some(&[0.0, f32::NAN][..]),
            2,
            NonFiniteFeedback,
            "feedback must be finite",
        ),
        (
            &[f32::INFINITY, 0.0][..],
            Some(&[0.0; 2][..]),
            2,
            NonFiniteInput,
            "input must be finite",
        ),
    ] {
        let error = plan
            .process(
                &mut output[..nout],
                &mut workspace,
                ClosedLoopCorrectionInput {
                    x: input,
                    constraint_feedback: feedback,
                },
            )
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
        assert_eq!(workspace.state(), committed);
        assert_eq!(workspace.last_output(), Some(&published[..]));
        assert_eq!(output, published);
    }
}
