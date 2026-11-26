//! Sequential, held-command, and all-stage transaction obligations from JFG.
use filtergraph_algorithms::{
    CorrectorAllocationF32 as D, CorrectorAllocationF32Configuration as C,
};

fn configuration() -> C {
    C {
        controller_command_count: 1,
        stage_command_counts: vec![1, 1],
        update_periods: vec![1, 1],
        command_projection: vec![1.0, 1.0],
        realization_projection: vec![1.0, 1.0],
        section_count: 1,
        initial_coefficients: None,
        command_limits: (vec![-1.0, -2.0], vec![1.0, 2.0]),
        slew_limits: None,
        command_schema: "org.example.corrector.commands/1".into(),
    }
}

#[test]
fn stages_subtract_conditioned_commands_before_allocating_next_residual() {
    let mut d = D::prepare(configuration()).unwrap();
    let mut commands = [71.0; 2];
    let mut feedback = [71.0];
    D::process(&mut d, &[3.0], &mut commands, &mut feedback).unwrap();
    assert_eq!(commands, [1.0, 2.0]);
    assert_eq!(feedback, [0.0]);
    D::process(&mut d, &[-4.0], &mut commands, &mut feedback).unwrap();
    assert_eq!(commands, [-1.0, -2.0]);
    assert_eq!(feedback, [-1.0]);
    assert_eq!(
        d.formats()[0].schema(),
        "pipewireao/controller-command-f32/1"
    );
    assert_eq!(d.formats()[1].shape(), &[2]);
    assert_eq!(
        d.formats()[2].schema(),
        "pipewireao/controller-constraint-feedback-f32/1"
    );
}

#[test]
fn filter_runs_at_base_rate_while_slower_commands_hold_and_reset() {
    let mut c = configuration();
    c.stage_command_counts = vec![1];
    c.update_periods = vec![2];
    c.command_projection = vec![1.0];
    c.realization_projection = vec![1.0];
    c.command_limits = (vec![-100.0], vec![100.0]);
    c.initial_coefficients = Some(vec![0.5, 0.25, 0.0, 0.0, 0.0]);
    let mut d = D::prepare(c).unwrap();
    let mut commands = [71.0];
    let mut feedback = [71.0];
    for (input, command, residual) in [
        (2.0, 1.0, 1.0),
        (4.0, 1.0, 3.0),
        (0.0, 1.0, -1.0),
        (8.0, 1.0, 7.0),
        (0.0, 2.0, -2.0),
    ] {
        D::process(&mut d, &[input], &mut commands, &mut feedback).unwrap();
        assert_eq!(commands, [command]);
        assert_eq!(feedback, [residual]);
    }
    d.reset();
    D::process(&mut d, &[0.0], &mut commands, &mut feedback).unwrap();
    assert_eq!(commands, [0.0]);
    assert_eq!(feedback, [0.0]);
}

#[test]
fn later_stage_rejection_does_not_adopt_earlier_filter_or_counters() {
    let mut c = configuration();
    c.command_projection = vec![1.0, f32::MAX];
    c.realization_projection = vec![0.0, 0.0];
    c.update_periods = vec![2, 1];
    c.initial_coefficients = Some(vec![0.5, 0.25, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let mut actual = D::prepare(c.clone()).unwrap();
    let mut reference = D::prepare(c).unwrap();
    let mut commands = [71.0; 2];
    let mut feedback = [-17.0];
    let mut expected = [0.0; 2];
    let mut expected_feedback = [0.0];
    D::process(&mut actual, &[0.25], &mut commands, &mut feedback).unwrap();
    D::process(
        &mut reference,
        &[0.25],
        &mut expected,
        &mut expected_feedback,
    )
    .unwrap();
    let old_commands = commands;
    let old_feedback = feedback;
    for input in [2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(D::process(&mut actual, &[input], &mut commands, &mut feedback).is_err());
        assert_eq!(commands, old_commands);
        assert_eq!(feedback, old_feedback);
    }
    for input in [0.5, 0.125, -0.25, 0.0] {
        D::process(&mut actual, &[input], &mut commands, &mut feedback).unwrap();
        D::process(
            &mut reference,
            &[input],
            &mut expected,
            &mut expected_feedback,
        )
        .unwrap();
        assert_eq!(commands, expected);
        assert_eq!(feedback, expected_feedback);
    }
}

#[test]
fn residual_overflow_withholds_both_outputs_and_restores_nested_candidates() {
    let mut c = configuration();
    c.command_projection = vec![1.0, f32::MAX];
    c.realization_projection = vec![0.0, f32::MAX];
    c.command_limits = (vec![-f32::MAX; 2], vec![f32::MAX; 2]);
    c.initial_coefficients = Some(vec![0.5, 0.25, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let mut d = D::prepare(c).unwrap();
    let mut commands = [71.0; 2];
    let mut feedback = [-17.0];
    assert!(D::process(&mut d, &[0.25], &mut commands, &mut feedback).is_err());
    assert_eq!(commands, [71.0; 2]);
    assert_eq!(feedback, [-17.0]);
    D::process(&mut d, &[0.0], &mut commands, &mut feedback).unwrap();
    assert_eq!(commands, [0.0; 2]);
    assert_eq!(feedback, [0.0]);
}

#[test]
fn conditioner_rejection_preserves_all_prior_stage_history() {
    let mut c = configuration();
    c.realization_projection = vec![0.0, 0.0];
    c.update_periods = vec![2, 1];
    c.command_limits = (vec![-f32::MAX; 2], vec![f32::MAX; 2]);
    c.slew_limits = Some(vec![f32::MAX * 0.5; 2]);
    c.initial_coefficients = Some(vec![0.5, 0.25, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let mut actual = D::prepare(c.clone()).unwrap();
    let mut control = D::prepare(c).unwrap();
    let mut commands = [71.0; 2];
    let mut feedback = [-17.0];
    let mut expected = [71.0; 2];
    let mut expected_feedback = [-17.0];
    for _ in 0..2 {
        D::process(&mut actual, &[-f32::MAX], &mut commands, &mut feedback).unwrap();
        D::process(
            &mut control,
            &[-f32::MAX],
            &mut expected,
            &mut expected_feedback,
        )
        .unwrap();
    }
    let old = commands;
    let old_feedback = feedback;
    assert!(D::process(&mut actual, &[f32::MAX], &mut commands, &mut feedback).is_err());
    assert_eq!(commands, old);
    assert_eq!(feedback, old_feedback);
    for value in [0.0, 0.0, 0.25] {
        D::process(&mut actual, &[value], &mut commands, &mut feedback).unwrap();
        D::process(
            &mut control,
            &[value],
            &mut expected,
            &mut expected_feedback,
        )
        .unwrap();
        assert_eq!(commands, expected);
        assert_eq!(feedback, expected_feedback);
    }
}

#[test]
fn invalid_configuration_is_rejected_before_large_allocation() {
    let invalid: [fn(&mut C); 12] = [
        |c| c.controller_command_count = 0,
        |c| c.stage_command_counts.clear(),
        |c| c.stage_command_counts[0] = 0,
        |c| c.update_periods[0] = 0,
        |c| c.update_periods.clear(),
        |c| c.update_periods[0] = 2,
        |c| c.command_limits.0[0] = 0.25,
        |c| c.command_projection[0] = f32::NAN,
        |c| c.realization_projection[0] = f32::INFINITY,
        |c| c.controller_command_count = usize::MAX,
        |c| c.section_count = usize::MAX,
        |c| c.stage_command_counts = vec![usize::MAX, 1],
    ];
    for change in invalid {
        let mut c = configuration();
        change(&mut c);
        assert!(D::prepare(c).is_err());
    }
    let mut c = configuration();
    c.initial_coefficients = Some(vec![1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(D::prepare(c).is_err());
    let mut c = configuration();
    c.command_limits.1.pop();
    assert!(D::prepare(c).is_err());
}

#[test]
fn ordered_projection_fma_cancels_overflow_before_publication() {
    let mut c = configuration();
    c.controller_command_count = 2;
    c.stage_command_counts = vec![1];
    c.update_periods = vec![1];
    c.command_projection = vec![-f32::MAX * 0.5, f32::MAX * 0.5];
    c.realization_projection = vec![0.0, 0.0];
    c.command_limits = (vec![-f32::MAX], vec![f32::MAX]);
    let mut d = D::prepare(c).unwrap();
    let mut commands = [71.0];
    let mut feedback = [-17.0; 2];
    let expected = (f32::MAX * 0.5).mul_add(2.25, -f32::MAX);
    assert!(expected.is_finite());
    D::process(&mut d, &[2.0, 2.25], &mut commands, &mut feedback).unwrap();
    assert_eq!(commands, [expected]);
    assert_eq!(feedback, [2.0, 2.25]);
}

#[test]
fn ordered_residual_fma_cancels_overflow_before_publication() {
    let mut c = configuration();
    c.stage_command_counts = vec![1];
    c.update_periods = vec![1];
    c.command_projection = vec![2.0 / f32::MAX];
    c.realization_projection = vec![f32::MAX * 0.75];
    c.command_limits = (vec![-f32::MAX], vec![f32::MAX]);
    let mut d = D::prepare(c.clone()).unwrap();
    let mut commands = [71.0];
    let mut feedback = [-17.0];
    let command = c.command_projection[0].mul_add(f32::MAX, 0.0);
    let expected = (-c.realization_projection[0]).mul_add(command, f32::MAX);
    assert!(expected.is_finite());
    D::process(&mut d, &[f32::MAX], &mut commands, &mut feedback).unwrap();
    assert_eq!(commands, [command]);
    assert_eq!(feedback, [expected]);
}
