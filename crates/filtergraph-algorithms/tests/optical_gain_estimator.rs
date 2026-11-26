//! Synchronous modal optical-gain estimator tests.

use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, OpticalGainDitherMode,
    OpticalGainDitherPlan, OpticalGainDitherSchedule, OpticalGainEstimateField,
    OpticalGainEstimateOutput, OpticalGainEstimatorError, OpticalGainEstimatorF32,
    OpticalGainEstimatorF32Configuration, OpticalGainEstimatorInput, OpticalGainEstimatorInputKind,
    OpticalGainEstimatorPlan, OpticalGainEstimatorPlanError,
};

fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} is not within {tolerance} of {expected}"
    );
}

struct OutputStorage<T> {
    command_amplitudes: Vec<T>,
    response_amplitudes: Vec<T>,
    optical_gains: Vec<T>,
    phase_lags: Vec<T>,
    valid: Vec<bool>,
    complete: bool,
}

impl<T: Clone> OutputStorage<T> {
    fn filled(len: usize, value: T) -> Self {
        Self {
            command_amplitudes: vec![value.clone(); len],
            response_amplitudes: vec![value.clone(); len],
            optical_gains: vec![value.clone(); len],
            phase_lags: vec![value; len],
            valid: vec![false; len],
            complete: true,
        }
    }

    fn output(&mut self) -> OpticalGainEstimateOutput<'_, T> {
        OpticalGainEstimateOutput::new(
            &mut self.command_amplitudes,
            &mut self.response_amplitudes,
            &mut self.optical_gains,
            &mut self.phase_lags,
            &mut self.valid,
            &mut self.complete,
        )
    }
}

#[test]
fn preparation_requires_one_positive_calibration_response_per_mode() {
    let schedule = Arc::new(
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f32, 4, 1)]).unwrap(),
    );
    assert_eq!(
        OpticalGainEstimatorPlan::new(Arc::clone(&schedule), []).unwrap_err(),
        OpticalGainEstimatorPlanError::CalibrationResponseExtent {
            expected: 1,
            actual: 0,
        }
    );
    assert_eq!(
        OpticalGainEstimatorPlan::new(Arc::clone(&schedule), [0.0]).unwrap_err(),
        OpticalGainEstimatorPlanError::InvalidCalibrationResponse { mode: 0 }
    );
    assert_eq!(
        OpticalGainEstimatorPlan::new(schedule, [f32::NAN]).unwrap_err(),
        OpticalGainEstimatorPlanError::InvalidCalibrationResponse { mode: 0 }
    );
}

#[test]
fn estimator_recovers_physical_gain_and_response_phase_lag() {
    let schedule = Arc::new(
        OpticalGainDitherSchedule::new([
            OpticalGainDitherMode::new(0.4_f64, 8, 3),
            OpticalGainDitherMode::new(0.7_f64, 6, 4),
        ])
        .unwrap(),
    );
    let generator = OpticalGainDitherPlan::new(Arc::clone(&schedule));
    let estimator = OpticalGainEstimatorPlan::new(Arc::clone(&schedule), [1.25, 0.8]).unwrap();
    assert!(Arc::ptr_eq(&schedule, &estimator.shared_schedule()));
    assert_eq!(estimator.calibration_responses(), &[1.25, 0.8]);
    let mut generator_workspace = generator.workspace();
    let mut estimator_workspace = estimator.workspace();
    let mut command = [0.0_f64; 2];
    let gains = [0.55_f64, 0.9];
    let lags = [0.3_f64, -0.45];
    let amplitudes = [0.4_f64, 0.7];
    let calibrations = [1.25_f64, 0.8];
    let periods = [8.0_f64, 6.0];
    let mut storage = OutputStorage::filled(2, f64::NAN);

    for frame in 0..schedule.frames_per_estimate() {
        generator
            .process(&mut command, &mut generator_workspace, ())
            .unwrap();
        let response: [f64; 2] = core::array::from_fn(|mode| {
            let theta = core::f64::consts::TAU * frame as f64 / periods[mode];
            gains[mode] * calibrations[mode] * amplitudes[mode] * (theta - lags[mode]).cos()
        });
        estimator
            .process(
                storage.output(),
                &mut estimator_workspace,
                OpticalGainEstimatorInput::new(&command, &response),
            )
            .unwrap();
        assert_eq!(
            storage.complete,
            frame + 1 == schedule.frames_per_estimate()
        );
        if !storage.complete {
            assert!(storage.optical_gains.iter().all(|value| value.is_nan()));
        }
    }

    for mode in 0..2 {
        assert_near(storage.command_amplitudes[mode], amplitudes[mode], 2.0e-15);
        assert_near(
            storage.response_amplitudes[mode],
            gains[mode] * calibrations[mode] * amplitudes[mode],
            2.0e-15,
        );
        assert_near(storage.optical_gains[mode], gains[mode], 2.0e-15);
        assert_near(storage.phase_lags[mode], lags[mode], 2.0e-15);
        assert!(storage.valid[mode]);
    }
    assert_eq!(estimator_workspace.frame_count(), 0);

    // A completed estimate starts a fresh phase-zero window.
    generator
        .process(&mut command, &mut generator_workspace, ())
        .unwrap();
    let response = [
        command[0] * gains[0] * calibrations[0],
        command[1] * gains[1] * calibrations[1],
    ];
    estimator
        .process(
            storage.output(),
            &mut estimator_workspace,
            OpticalGainEstimatorInput::new(&command, &response),
        )
        .unwrap();
    assert!(!storage.complete);
    assert_eq!(estimator_workspace.frame_count(), 1);
}

#[test]
fn zero_command_or_response_is_an_invalid_expected_estimate() {
    let schedule = OpticalGainDitherSchedule::new([
        OpticalGainDitherMode::new(1.0_f32, 4, 1),
        OpticalGainDitherMode::new(1.0_f32, 4, 1),
    ])
    .unwrap();
    let estimator = OpticalGainEstimatorPlan::new(schedule, [1.0, 1.0]).unwrap();
    let mut workspace = estimator.workspace();
    let mut storage = OutputStorage::filled(2, f32::NAN);

    for command in [[0.0, 1.0], [0.0, 0.0], [0.0, -1.0], [0.0, 0.0]] {
        estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(&command, &[1.0, 0.0]),
            )
            .unwrap();
    }
    assert!(storage.complete);
    assert_eq!(storage.valid, [false, false]);
    assert_eq!(storage.optical_gains, [0.0, 0.0]);
    assert_eq!(storage.phase_lags, [0.0, 0.0]);
}

#[test]
fn rejected_sample_preserves_window_history() {
    let schedule =
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f64, 4, 1)]).unwrap();
    let estimator = OpticalGainEstimatorPlan::new(schedule, [1.0]).unwrap();
    let mut workspace = estimator.workspace();
    let mut storage = OutputStorage::filled(1, f64::NAN);

    estimator
        .process(
            storage.output(),
            &mut workspace,
            OpticalGainEstimatorInput::new(&[1.0], &[0.5]),
        )
        .unwrap();
    assert_eq!(workspace.frame_count(), 1);
    assert_eq!(
        estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(&[0.0], &[0.0]).with_window_state(1, &[0], false),
            )
            .unwrap_err(),
        OpticalGainEstimatorError::PhaseIndexMismatch {
            mode: 0,
            expected: 1,
            actual: 0,
        }
    );
    assert_eq!(workspace.frame_count(), 1);
    assert_eq!(
        estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(&[f64::NAN], &[0.0]),
            )
            .unwrap_err(),
        OpticalGainEstimatorError::NonFiniteInput {
            input: OpticalGainEstimatorInputKind::RealizedCommand,
            mode: 0,
        }
    );
    assert_eq!(workspace.frame_count(), 1);

    for (command, response) in [(0.0, 0.0), (-1.0, -0.5), (0.0, 0.0)] {
        estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(&[command], &[response]),
            )
            .unwrap();
    }
    assert!(storage.complete);
    assert_near(storage.optical_gains[0], 0.5, f64::EPSILON);
}

#[test]
fn nonfinite_accumulation_does_not_consume_the_sample() {
    let schedule =
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f64, 4, 1)]).unwrap();
    let estimator = OpticalGainEstimatorPlan::new(schedule, [1.0]).unwrap();
    let mut workspace = estimator.workspace();
    let mut storage = OutputStorage::filled(1, f64::NAN);

    for command in [f64::MAX / 2.0, f64::MAX / 2.0] {
        estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(&[command], &[0.0]),
            )
            .unwrap();
    }
    assert_eq!(workspace.frame_count(), 2);
    assert_eq!(
        estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(&[-f64::MAX], &[0.0]),
            )
            .unwrap_err(),
        OpticalGainEstimatorError::NonFiniteAccumulation { mode: 0 }
    );
    assert_eq!(workspace.frame_count(), 2);

    for command in [0.0, 0.0] {
        estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(&[command], &[0.0]),
            )
            .unwrap();
    }
    assert!(storage.complete);
    assert_eq!(workspace.frame_count(), 0);
}

#[test]
fn extent_rejection_precedes_window_mutation() {
    let schedule =
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f32, 4, 1)]).unwrap();
    let estimator = OpticalGainEstimatorPlan::new(schedule, [1.0]).unwrap();
    let mut workspace = estimator.workspace();
    let mut storage = OutputStorage::filled(1, 0.0_f32);

    assert_eq!(
        estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(&[], &[]),
            )
            .unwrap_err(),
        OpticalGainEstimatorError::InputExtent {
            input: OpticalGainEstimatorInputKind::RealizedCommand,
            expected: 1,
            actual: 0,
        }
    );
    assert_eq!(workspace.frame_count(), 0);

    let mut empty = [];
    let mut response = [0.0_f32; 1];
    let mut gain = [0.0_f32; 1];
    let mut phase = [0.0_f32; 1];
    let mut valid = [false; 1];
    let mut complete = false;
    assert_eq!(
        estimator
            .process(
                OpticalGainEstimateOutput::new(
                    &mut empty,
                    &mut response,
                    &mut gain,
                    &mut phase,
                    &mut valid,
                    &mut complete,
                ),
                &mut workspace,
                OpticalGainEstimatorInput::new(&[1.0], &[1.0]),
            )
            .unwrap_err(),
        OpticalGainEstimatorError::OutputExtent {
            field: OpticalGainEstimateField::CommandAmplitude,
            expected: 1,
            actual: 0,
        }
    );
    assert_eq!(workspace.frame_count(), 0);
}

#[test]
fn f32_estimation_applies_calibration_response() {
    let schedule =
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f32, 4, 2)]).unwrap();
    let estimator = OpticalGainEstimatorPlan::new(schedule, [2.0]).unwrap();
    let mut workspace = estimator.workspace();
    let mut storage = OutputStorage::filled(1, 0.0_f32);

    for command in [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0] {
        estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(&[command], &[command]),
            )
            .unwrap();
    }
    assert!(storage.complete);
    assert!((storage.optical_gains[0] - 0.5).abs() <= 4.0 * f32::EPSILON);
    assert!(storage.valid[0]);
}

#[test]
fn synchronized_input_rejects_a_skipped_generator_frame_without_mutation() {
    let schedule =
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f32, 4, 1)]).unwrap();
    let estimator = OpticalGainEstimatorPlan::new(schedule, [1.0]).unwrap();
    let mut workspace = estimator.workspace();
    let mut storage = OutputStorage::filled(1, f32::NAN);

    estimator
        .process(
            storage.output(),
            &mut workspace,
            OpticalGainEstimatorInput::new(&[1.0], &[0.5]).with_window_state(0, &[0], false),
        )
        .unwrap();
    assert_eq!(workspace.frame_count(), 1);
    assert_eq!(
        estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(&[-1.0], &[-0.5]).with_window_state(2, &[2], false),
            )
            .unwrap_err(),
        OpticalGainEstimatorError::WindowFrameMismatch {
            expected: 1,
            actual: 2,
        }
    );
    assert_eq!(workspace.frame_count(), 1);
}

#[test]
fn declaration_estimates_one_window_and_replaces_calibration_response() {
    let mut declared = OpticalGainEstimatorF32::prepare(OpticalGainEstimatorF32Configuration {
        modes: vec![OpticalGainDitherMode::new(0.5, 4, 1)],
        initial_calibration_responses: vec![1.0],
    })
    .unwrap();
    assert_eq!(OpticalGainEstimatorF32::NAME, "optical-gain-estimator-f32");
    assert_eq!(declared.formats().len(), 12);

    let commands = [0.5_f32, 0.0, -0.5, 0.0];
    let responses = [0.25_f32, 0.0, -0.25, 0.0];
    let mut command_amplitude = [f32::NAN];
    let mut response_amplitude = [f32::NAN];
    let mut optical_gain = [f32::NAN];
    let mut phase_lag = [f32::NAN];
    let mut valid = [false];
    let mut estimate_complete = [true];
    for frame in 0..4 {
        OpticalGainEstimatorF32::process(
            &mut declared,
            &[commands[frame]],
            &[responses[frame]],
            &[frame as u64],
            &[frame as u64],
            &[frame == 3],
            &mut command_amplitude,
            &mut response_amplitude,
            &mut optical_gain,
            &mut phase_lag,
            &mut valid,
            &mut estimate_complete,
        )
        .unwrap();
        assert_eq!(estimate_complete, [frame == 3]);
    }
    assert!((command_amplitude[0] - 0.5).abs() <= 2.0 * f32::EPSILON);
    assert!((response_amplitude[0] - 0.25).abs() <= 2.0 * f32::EPSILON);
    assert!((optical_gain[0] - 0.5).abs() <= 2.0 * f32::EPSILON);
    assert!(phase_lag[0].abs() <= 2.0 * f32::EPSILON);
    assert_eq!(valid, [true]);

    assert_eq!(
        declared.replace_parameter(11, AlgorithmInput::from(&[0.5_f32][..])),
        Some(Ok(()))
    );
    OpticalGainEstimatorF32::reset(&mut declared);
    for frame in 0..4 {
        OpticalGainEstimatorF32::process(
            &mut declared,
            &[commands[frame]],
            &[responses[frame]],
            &[frame as u64],
            &[frame as u64],
            &[frame == 3],
            &mut command_amplitude,
            &mut response_amplitude,
            &mut optical_gain,
            &mut phase_lag,
            &mut valid,
            &mut estimate_complete,
        )
        .unwrap();
    }
    assert!((optical_gain[0] - 1.0).abs() <= 4.0 * f32::EPSILON);
}

#[test]
fn estimator_rejects_inconsistent_window_metadata_and_accepts_a_corrected_sample() {
    use OpticalGainEstimatorError::*;

    let schedule =
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f32, 4, 1)]).unwrap();
    let estimator = OpticalGainEstimatorPlan::new(schedule, [1.0]).unwrap();
    let mut workspace = estimator.workspace();
    let mut storage = OutputStorage::filled(1, 71.0);
    let input = OpticalGainEstimatorInput::new(&[1.0], &[0.5]);
    assert_eq!(input.realized_command_modes(), &[1.0]);
    assert_eq!(input.reconstructed_response_modes(), &[0.5]);
    for (wrong, expected, detail) in [
        (
            input.with_window_state(1, &[1], false),
            WindowFrameMismatch {
                expected: 0,
                actual: 1,
            },
            "window frame 1 does not match 0",
        ),
        (
            input.with_window_state(0, &[], false),
            PhaseIndexExtent {
                expected: 1,
                actual: 0,
            },
            "phase-index extent 0 does not match 1",
        ),
        (
            input.with_window_state(0, &[2], false),
            PhaseIndexMismatch {
                mode: 0,
                expected: 0,
                actual: 2,
            },
            "phase 2 for mode 0 does not match 0",
        ),
        (
            input.with_window_state(0, &[0], true),
            WindowCompletionMismatch {
                expected: false,
                actual: true,
            },
            "completion flag true does not match false",
        ),
    ] {
        let error = estimator
            .process(storage.output(), &mut workspace, wrong)
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
        assert_eq!(workspace.frame_count(), 0);
        assert!(storage.complete);
        assert_eq!(storage.optical_gains, [71.0]);
    }
    estimator
        .process(
            storage.output(),
            &mut workspace,
            input.with_window_state(0, &[0], false),
        )
        .unwrap();
    assert_eq!(workspace.frame_count(), 1);
    assert!(!storage.complete);

    let other_schedule = OpticalGainDitherSchedule::new([
        OpticalGainDitherMode::new(1.0_f32, 4, 2),
        OpticalGainDitherMode::new(1.0_f32, 8, 1),
    ])
    .unwrap();
    let other_estimator = OpticalGainEstimatorPlan::new(other_schedule, [1.0; 2]).unwrap();
    let error = estimator
        .process(storage.output(), &mut other_estimator.workspace(), input)
        .unwrap_err();
    assert_eq!(error, WorkspaceExtent);
    assert!(error.to_string().contains("workspace extent"));
}

#[test]
fn estimator_distinguishes_the_malformed_input_and_output_fields() {
    use OpticalGainEstimateField::*;
    use OpticalGainEstimatorError::*;
    use OpticalGainEstimatorInputKind::*;

    let schedule =
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f32, 4, 1)]).unwrap();
    let estimator = OpticalGainEstimatorPlan::new(schedule, [1.0]).unwrap();
    let mut workspace = estimator.workspace();

    let mut storage = OutputStorage::filled(1, 71.0);
    for (command, response, expected, detail) in [
        (
            &[1.0][..],
            &[][..],
            InputExtent {
                input: ReconstructedResponse,
                expected: 1,
                actual: 0,
            },
            "ReconstructedResponse input extent 0 does not match 1",
        ),
        (
            &[1.0][..],
            &[f32::INFINITY][..],
            NonFiniteInput {
                input: ReconstructedResponse,
                mode: 0,
            },
            "ReconstructedResponse input for mode 0 is not finite",
        ),
    ] {
        let error = estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(command, response),
            )
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
        assert_eq!(workspace.frame_count(), 0);
        assert!(storage.complete);
    }

    for field in [
        CommandAmplitude,
        ResponseAmplitude,
        OpticalGain,
        PhaseLag,
        Validity,
    ] {
        let mut storage = OutputStorage::filled(1, 71.0);
        match field {
            CommandAmplitude => storage.command_amplitudes.clear(),
            ResponseAmplitude => storage.response_amplitudes.clear(),
            OpticalGain => storage.optical_gains.clear(),
            PhaseLag => storage.phase_lags.clear(),
            Validity => storage.valid.clear(),
        }
        let error = estimator
            .process(
                storage.output(),
                &mut workspace,
                OpticalGainEstimatorInput::new(&[1.0], &[0.5]),
            )
            .unwrap_err();
        assert_eq!(
            error,
            OutputExtent {
                field,
                expected: 1,
                actual: 0
            }
        );
        assert!(
            error
                .to_string()
                .contains(&format!("{field:?} output extent 0 does not match 1"))
        );
        assert_eq!(workspace.frame_count(), 0);
        assert!(storage.complete);
    }
}
