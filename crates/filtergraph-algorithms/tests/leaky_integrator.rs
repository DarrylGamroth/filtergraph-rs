//! Leaky-integrator recurrence, state, progressive, and abort tests.

use filtergraph_algorithms::{
    AlgorithmPlan, InputProgress, LeakyIntegratorError, LeakyIntegratorPlan,
    LeakyIntegratorPlanError, LeakyIntegratorWorkspace, LeakyIntegratorWorkspaceError,
    ProcessError, ProgressResult,
};

fn workspace_with_state(initial_state: &[f32]) -> LeakyIntegratorWorkspace<f32> {
    let mut workspace = LeakyIntegratorWorkspace::new(initial_state.len()).unwrap();
    let plan = LeakyIntegratorPlan::new(1.0_f32, 0.0).unwrap();
    let mut output = vec![0.0_f32; initial_state.len()];
    plan.process(&mut output, &mut workspace, initial_state)
        .unwrap();
    assert_eq!(workspace.state(), initial_state);
    workspace
}

#[test]
fn preparation_validates_coefficients_and_workspace_extent() {
    let signed = LeakyIntegratorPlan::new(-0.5_f32, 0.25).unwrap();
    assert_eq!(signed.gain(), -0.5);
    assert_eq!(
        LeakyIntegratorPlan::new(f32::INFINITY, 0.25),
        Err(LeakyIntegratorPlanError::InvalidGain)
    );
    assert_eq!(
        LeakyIntegratorPlan::new(0.5_f32, -0.25),
        Err(LeakyIntegratorPlanError::InvalidPole)
    );
    assert_eq!(
        LeakyIntegratorPlan::new(0.5_f32, 1.25),
        Err(LeakyIntegratorPlanError::InvalidPole)
    );
    assert_eq!(
        LeakyIntegratorWorkspace::<f32>::new(0),
        Err(LeakyIntegratorWorkspaceError::EmptyState)
    );

    let zero_pole = LeakyIntegratorPlan::new(0.0_f32, 0.0).unwrap();
    assert_eq!(zero_pole.gain(), 0.0);
    assert_eq!(zero_pole.pole(), 0.0);
    let unity_pole = LeakyIntegratorPlan::new(0.5_f32, 1.0).unwrap();
    assert_eq!(unity_pole.gain(), 0.5);
    assert_eq!(unity_pole.pole(), 1.0);
}

#[test]
fn complete_processing_matches_the_frozen_float32_recurrence() {
    let plan = LeakyIntegratorPlan::new(0.5_f32, 0.75).unwrap();
    let mut workspace = workspace_with_state(&[4.0, -4.0, 2.0, -2.0]);
    let mut output = [0.0_f32; 4];

    plan.process(&mut output, &mut workspace, &[2.0, 2.0, -2.0, -2.0])
        .unwrap();
    assert_eq!(output, [4.0, -2.0, 0.5, -2.5]);
    assert_eq!(workspace.state(), output);

    plan.process(&mut output, &mut workspace, &[0.0, 4.0, 2.0, -4.0])
        .unwrap();
    assert_eq!(output, [3.0, 0.5, 1.375, -3.875]);
    assert_eq!(workspace.state(), output);
}

#[test]
fn plan_is_extent_independent_and_supports_float64() {
    let plan = LeakyIntegratorPlan::new(0.5_f64, 0.75).unwrap();
    let mut short = LeakyIntegratorWorkspace::new(4).unwrap();
    let long = LeakyIntegratorWorkspace::<f64>::new(240).unwrap();
    let mut output = [0.0_f64; 4];

    plan.process(&mut output, &mut short, &[2.0; 4]).unwrap();

    assert_eq!(output, [1.0; 4]);
    assert_eq!(short.len(), 4);
    assert_eq!(long.len(), 240);
}

#[test]
fn extent_errors_precede_state_or_output_mutation() {
    let plan = LeakyIntegratorPlan::new(0.5_f32, 0.75).unwrap();
    let initial_state = [1.0_f32, 2.0, 3.0, 4.0];
    let mut workspace = workspace_with_state(&initial_state);
    let mut output = [-1.0_f32; 4];

    assert_eq!(
        plan.process(&mut output, &mut workspace, &[2.0; 3]),
        Err(LeakyIntegratorError::Process(ProcessError::InputExtent {
            expected: 4,
            actual: 3,
        }))
    );
    assert_eq!(workspace.state(), initial_state);
    assert_eq!(output, [-1.0; 4]);
}

#[test]
fn progressive_finish_commits_and_abort_preserves_history() {
    let plan = LeakyIntegratorPlan::new(0.5_f32, 0.75).unwrap();
    let initial_state = [1.0_f32, 2.0, 3.0, 4.0];
    let expected_state = [1.75_f32, 2.5, 3.25, 4.0];
    let input = [2.0_f32; 4];
    let mut workspace = workspace_with_state(&initial_state);
    let mut output = [-1.0_f32; 4];

    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &input, 0..2),
        Ok(ProgressResult::new(2))
    );
    assert_eq!(output, [1.75, 2.5, -1.0, -1.0]);
    assert_eq!(workspace.state(), initial_state);
    plan.abort(&mut workspace).unwrap();
    assert_eq!(workspace.state(), initial_state);

    plan.start(&mut workspace).unwrap();
    let _ = plan
        .process_range(&mut output, &mut workspace, &input, 0..2)
        .unwrap();
    let _ = plan
        .process_range(&mut output, &mut workspace, &input, 2..4)
        .unwrap();
    assert_eq!(workspace.state(), initial_state);
    plan.finish(&mut output, &mut workspace).unwrap();
    assert_eq!(output, expected_state);
    assert_eq!(workspace.state(), expected_state);
}

#[test]
fn progressive_sequence_errors_are_explicit() {
    let plan = LeakyIntegratorPlan::new(0.5_f32, 0.75).unwrap();
    let mut workspace = LeakyIntegratorWorkspace::new(4).unwrap();
    let input = [2.0_f32; 4];
    let mut output = [-1.0_f32; 4];

    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &input, 0..1),
        Err(LeakyIntegratorError::Process(ProcessError::NotStarted))
    );
    assert_eq!(
        plan.finish(&mut output, &mut workspace),
        Err(LeakyIntegratorError::Process(ProcessError::NotStarted))
    );
    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.start(&mut workspace),
        Err(LeakyIntegratorError::Process(ProcessError::AlreadyStarted))
    );
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &input, 1..2),
        Err(LeakyIntegratorError::Process(
            ProcessError::RangeNotContiguous {
                start: 1,
                expected: 0,
            }
        ))
    );
    let _ = plan
        .process_range(&mut output, &mut workspace, &input, 0..2)
        .unwrap();
    assert_eq!(
        plan.finish(&mut output, &mut workspace),
        Err(LeakyIntegratorError::Process(ProcessError::Incomplete {
            processed: 2,
            extent: 4,
        }))
    );
    plan.abort(&mut workspace).unwrap();
}

#[test]
fn reset_clears_state_and_next_state() {
    let plan = LeakyIntegratorPlan::new(0.5_f32, 0.75).unwrap();
    let mut workspace = LeakyIntegratorWorkspace::new(2).unwrap();
    let mut output = [0.0; 2];

    plan.process(&mut output, &mut workspace, &[2.0, 4.0])
        .unwrap();
    assert_eq!(workspace.state(), &[1.0, 2.0]);
    plan.start(&mut workspace).unwrap();
    let _ = plan
        .process_range(&mut output, &mut workspace, &[8.0, 8.0], 0..1)
        .unwrap();

    workspace.reset();
    assert_eq!(workspace.state(), &[0.0, 0.0]);
    plan.process(&mut output, &mut workspace, &[2.0, 4.0])
        .unwrap();
    assert_eq!(output, [1.0, 2.0]);
}

#[test]
fn aborting_an_update_preserves_the_previous_state() {
    let plan = LeakyIntegratorPlan::new(0.5_f32, 1.0).unwrap();
    let mut workspace = LeakyIntegratorWorkspace::new(1).unwrap();
    let mut state = [0.0];
    plan.process(&mut state, &mut workspace, &[1.0]).unwrap();

    plan.start(&mut workspace).unwrap();
    let _ = plan
        .process_range(&mut state, &mut workspace, &[1.0], 0..1)
        .unwrap();
    plan.abort(&mut workspace).unwrap();
    assert_eq!(workspace.state(), [0.5]);
}

#[test]
fn rejected_numeric_sample_preserves_committed_recurrence_for_the_next_tick() {
    let plan = LeakyIntegratorPlan::new(2.0_f32, 0.5).unwrap();
    let mut workspace = workspace_with_state(&[4.0, 8.0]);
    let mut output = [19.0; 2];
    for (input, expected, detail) in [
        (
            [1.0, f32::NAN],
            LeakyIntegratorError::NonFiniteInput,
            "input must be finite",
        ),
        (
            [1.0, f32::MAX],
            LeakyIntegratorError::NonFiniteOutput,
            "result must be finite",
        ),
    ] {
        let error = plan
            .process(&mut output, &mut workspace, &input)
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
        assert_eq!(workspace.state(), [4.0, 8.0]);
        assert_eq!(output, [19.0; 2]);
    }
    plan.process(&mut output, &mut workspace, &[1.0, 1.0])
        .unwrap();
    assert_eq!(workspace.state(), [4.0, 6.0]);
    assert_eq!(output, [4.0, 6.0]);
}
