//! Modal optical-gain dither schedule and generator tests.

use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmPlan, OpticalGainDitherError, OpticalGainDitherF32,
    OpticalGainDitherF32Configuration, OpticalGainDitherMode, OpticalGainDitherPlan,
    OpticalGainDitherSchedule, OpticalGainDitherScheduleError,
};

fn assert_near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 32.0 * f64::EPSILON * expected.abs().max(1.0),
        "{actual} is not near {expected}"
    );
}

#[test]
fn schedule_rejects_invalid_modes_and_unsynchronized_windows() {
    assert_eq!(
        OpticalGainDitherSchedule::<f32>::new([]).unwrap_err(),
        OpticalGainDitherScheduleError::Empty
    );
    assert_eq!(
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(0.0_f32, 4, 1)]).unwrap_err(),
        OpticalGainDitherScheduleError::InvalidAmplitude { mode: 0 }
    );
    assert_eq!(
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f32, 3, 1)]).unwrap_err(),
        OpticalGainDitherScheduleError::PeriodTooShort { mode: 0, frames: 3 }
    );
    assert_eq!(
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f32, 4, 0)]).unwrap_err(),
        OpticalGainDitherScheduleError::ZeroPeriodsPerEstimate { mode: 0 }
    );
    assert_eq!(
        OpticalGainDitherSchedule::new([
            OpticalGainDitherMode::new(1.0_f32, 4, 2),
            OpticalGainDitherMode::new(1.0_f32, 5, 2),
        ])
        .unwrap_err(),
        OpticalGainDitherScheduleError::EstimateWindowMismatch {
            mode: 1,
            expected: 8,
            actual: 10,
        }
    );
    assert_eq!(
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f32, usize::MAX, 2,)])
            .unwrap_err(),
        OpticalGainDitherScheduleError::EstimateWindowOverflow { mode: 0 }
    );
}

#[test]
fn generator_starts_at_phase_zero_and_repeats_complete_periods() {
    let schedule = Arc::new(
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(2.0_f64, 4, 2)]).unwrap(),
    );
    let plan = OpticalGainDitherPlan::new(schedule);
    let mut workspace = plan.workspace();
    let mut output = [f64::NAN];

    for expected in [2.0, 0.0, -2.0, 0.0, 2.0, 0.0, -2.0, 0.0] {
        plan.process(&mut output, &mut workspace, ()).unwrap();
        assert_near(output[0], expected);
    }
    assert_eq!(workspace.phases(), &[0]);
    assert_eq!(workspace.frame_count(), 0);
}

#[test]
fn generator_supports_multiple_frequencies_in_one_window() {
    let schedule = Arc::new(
        OpticalGainDitherSchedule::new([
            OpticalGainDitherMode::new(1.0_f64, 4, 3),
            OpticalGainDitherMode::new(0.5_f64, 6, 2),
        ])
        .unwrap(),
    );
    assert_eq!(schedule.frames_per_estimate(), 12);
    assert_eq!(schedule.modes()[0].frames_per_period(), 4);
    assert_eq!(schedule.modes()[1].periods_per_estimate(), 2);
    let plan = OpticalGainDitherPlan::new(Arc::clone(&schedule));
    assert!(Arc::ptr_eq(&schedule, &plan.shared_schedule()));
    let mut workspace = plan.workspace();
    let mut output = [0.0; 2];

    for frame in 0..12 {
        plan.process(&mut output, &mut workspace, ()).unwrap();
        assert_near(
            output[0],
            (core::f64::consts::TAU * frame as f64 / 4.0).cos(),
        );
        assert_near(
            output[1],
            0.5 * (core::f64::consts::TAU * frame as f64 / 6.0).cos(),
        );
    }
    assert_eq!(workspace.phases(), &[0, 0]);
    assert_eq!(workspace.frame_count(), 0);
}

#[test]
fn output_rejection_preserves_committed_phase() {
    let schedule = OpticalGainDitherSchedule::new([
        OpticalGainDitherMode::new(1.0_f32, 4, 1),
        OpticalGainDitherMode::new(1.0_f32, 4, 1),
    ])
    .unwrap();
    let plan = OpticalGainDitherPlan::new(schedule);
    let mut workspace = plan.workspace();
    let mut wrong = [0.0_f32; 1];

    assert_eq!(
        plan.process(&mut wrong, &mut workspace, ()).unwrap_err(),
        OpticalGainDitherError::OutputExtent {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(workspace.phases(), &[0, 0]);
    assert_eq!(workspace.frame_count(), 0);

    let mut output = [0.0_f32; 2];
    plan.process(&mut output, &mut workspace, ()).unwrap();
    assert_eq!(output, [1.0, 1.0]);
    assert_eq!(workspace.phases(), &[1, 1]);
    assert_eq!(workspace.frame_count(), 1);
}

#[test]
fn f32_generator_uses_the_same_modal_equation() {
    let schedule =
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(0.25_f32, 4, 1)]).unwrap();
    let plan = OpticalGainDitherPlan::new(schedule);
    let mut workspace = plan.workspace();
    let mut output = [0.0_f32];

    plan.process(&mut output, &mut workspace, ()).unwrap();
    assert_eq!(output, [0.25]);
    plan.process(&mut output, &mut workspace, ()).unwrap();
    assert!(output[0].abs() <= 2.0 * f32::EPSILON);
}

#[test]
fn declaration_publishes_window_state_and_reset_restarts_phase_zero() {
    let mut declared = OpticalGainDitherF32::prepare(OpticalGainDitherF32Configuration {
        modes: vec![OpticalGainDitherMode::new(0.5, 4, 1)],
    })
    .unwrap();
    assert_eq!(OpticalGainDitherF32::NAME, "optical-gain-dither-f32");
    assert_eq!(declared.formats().len(), 4);

    let mut command = [f32::NAN];
    let mut window_frame = [u64::MAX];
    let mut phase_indices = [u64::MAX];
    let mut complete = [true];
    for (expected_frame, expected_command) in [0.5_f32, 0.0, -0.5, 0.0].into_iter().enumerate() {
        OpticalGainDitherF32::process(
            &mut declared,
            &mut command,
            &mut window_frame,
            &mut phase_indices,
            &mut complete,
        )
        .unwrap();
        assert!((command[0] - expected_command).abs() <= 2.0 * f32::EPSILON);
        assert_eq!(window_frame, [expected_frame as u64]);
        assert_eq!(phase_indices, [expected_frame as u64]);
        assert_eq!(complete, [expected_frame == 3]);
    }

    OpticalGainDitherF32::reset(&mut declared);
    OpticalGainDitherF32::process(
        &mut declared,
        &mut command,
        &mut window_frame,
        &mut phase_indices,
        &mut complete,
    )
    .unwrap();
    assert_eq!(command, [0.5]);
    assert_eq!(window_frame, [0]);
    assert_eq!(phase_indices, [0]);
    assert_eq!(complete, [false]);
}
