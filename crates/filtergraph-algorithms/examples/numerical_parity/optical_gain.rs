//! Live optical-gain CPU lifecycle evidence using the public declarations.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, DeclarationProcessError, OpticalGainAssignment, OpticalGainCorrectionError,
    OpticalGainCorrectionF32, OpticalGainCorrectionF32Configuration, OpticalGainDitherF32,
    OpticalGainDitherF32Configuration, OpticalGainDitherMode, OpticalGainEstimatorError,
    OpticalGainEstimatorF32, OpticalGainEstimatorF32Configuration, OpticalGainEstimatorInputKind,
    OpticalGainUpdateCriteria, OpticalGainUpdateError, OpticalGainUpdateF32,
    OpticalGainUpdateF32Configuration, PreparedAlgorithm,
};
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize)]
struct Fixture {
    correction: Correction,
    dither: Dither,
    estimator: Estimator,
    update: Update,
}
#[derive(Deserialize)]
struct Correction {
    gains: Vec<f32>,
    replacement_gains: Vec<f32>,
    input: Vec<f32>,
}
#[derive(Deserialize)]
struct Dither {
    modes: Vec<OpticalGainDitherMode<f32>>,
    frames: usize,
    reset_before_frame: usize,
}
#[derive(Deserialize)]
struct Estimator {
    modes: Vec<OpticalGainDitherMode<f32>>,
    calibration_responses: Vec<f32>,
    replacement_responses: Vec<f32>,
    replace_before_frame: usize,
    commands: Vec<Vec<f32>>,
    responses: Vec<Vec<f32>>,
    window_frames: Vec<u64>,
    phase_indices: Vec<Vec<u64>>,
    completes_window: Vec<bool>,
}
#[derive(Deserialize)]
struct Update {
    criteria: Vec<OpticalGainUpdateCriteria<f32>>,
    assignments: Vec<OpticalGainAssignment<f32>>,
    initial_gains: Vec<f32>,
    cases: Vec<UpdateCase>,
}
#[derive(Clone, Deserialize)]
struct UpdateCase {
    name: String,
    command: Vec<f32>,
    response: Vec<f32>,
    gains: Vec<f32>,
    phases: Vec<f32>,
    valid: Vec<bool>,
    complete: bool,
}
fn flag(v: bool) -> f64 {
    if v { 1.0 } else { 0.0 }
}
fn emit(results: &mut Vec<Record>, label: &str, case: &str, values: &[f32], status: Vec<f64>) {
    let mut status = status;
    status.extend(values.iter().map(|v| {
        if v.is_nan() {
            1.0
        } else if *v == f32::INFINITY {
            2.0
        } else if *v == f32::NEG_INFINITY {
            3.0
        } else {
            0.0
        }
    }));
    record(
        results,
        label,
        format!("{case}-values"),
        values
            .iter()
            .map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 })
            .collect(),
    );
    record(results, label, format!("{case}-status"), status);
}
fn correction(f: Correction, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let mut prepared = OpticalGainCorrectionF32::prepare(OpticalGainCorrectionF32Configuration {
        optical_gains: f.gains,
    })?;
    let n = f.input.len();
    macro_rules! step {
        ($case:expr,$input:expr) => {{
            let mut output = vec![71.0; n];
            let outcome = OpticalGainCorrectionF32::process(&mut prepared, $input, &mut output);
            let reason = match outcome {
                Ok(_) => 0.0,
                Err(DeclarationProcessError::Algorithm(
                    OpticalGainCorrectionError::NonFiniteInput { .. },
                )) => 1.0,
                Err(DeclarationProcessError::Algorithm(
                    OpticalGainCorrectionError::NonFiniteOutput { .. },
                )) => 2.0,
                Err(e) => return Err(e.into()),
            };
            emit(
                results,
                "optical-correction-f32",
                $case,
                &output,
                vec![reason, flag(reason == 0.0 || output == vec![71.0; n])],
            );
        }};
    }
    step!("initial", &f.input);
    let mut invalid = f.input.clone();
    invalid[n - 1] = f32::NAN;
    step!("nan", &invalid);
    step!("after-nan", &f.input);
    step!("overflow", &vec![f32::MAX; n]);
    step!("after-overflow", &f.input);
    prepared
        .replace_parameter(2, AlgorithmInput::from(f.replacement_gains.as_slice()))
        .ok_or("missing gains")??;
    step!("replaced", &f.input);
    let previous = prepared.plan().optical_gains().to_vec();
    let mut invalid = f.replacement_gains;
    invalid[n - 1] = 0.0;
    let rejected = prepared
        .replace_parameter(2, AlgorithmInput::from(invalid.as_slice()))
        .ok_or("missing gains")?
        .is_err();
    record(
        results,
        "optical-correction-f32",
        "invalid-gains-status".into(),
        vec![
            flag(!rejected),
            flag(prepared.plan().optical_gains() == previous),
        ],
    );
    step!("after-invalid-gains", &f.input);
    Ok(())
}
fn dither(f: Dither, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let n = f.modes.len();
    let mut prepared =
        OpticalGainDitherF32::prepare(OpticalGainDitherF32Configuration { modes: f.modes })?;
    for frame in 0..f.frames {
        if frame == f.reset_before_frame {
            prepared.reset();
        }
        let mut command = vec![71.0; n];
        let mut published = [99_u64];
        let mut phases = vec![99_u64; n];
        let mut completes = [false];
        OpticalGainDitherF32::process(
            &mut prepared,
            &mut command,
            &mut published,
            &mut phases,
            &mut completes,
        )?;
        let mut status = vec![published[0] as f64];
        status.extend(phases.iter().map(|&v| v as f64));
        status.push(flag(completes[0]));
        status.push(prepared.workspace().frame_count() as f64);
        status.extend(prepared.workspace().phases().iter().map(|&v| v as f64));
        emit(
            results,
            "optical-dither-f32",
            &format!("frame-{frame}"),
            &command,
            status,
        );
    }
    Ok(())
}
fn estimator_step(
    prepared: &mut PreparedAlgorithm<OpticalGainEstimatorF32>,
    case: &str,
    signals: (&[f32], &[f32]),
    frame: u64,
    phases: &[u64],
    completes: bool,
    results: &mut Vec<Record>,
) -> Result<(), Box<dyn Error>> {
    let (command, response) = signals;
    let n = command.len();
    let mut ca = vec![71.0; n];
    let mut ra = ca.clone();
    let mut gains = ca.clone();
    let mut lags = ca.clone();
    let mut valid = vec![true; n];
    let mut complete = [true];
    let before = prepared.workspace().frame_count();
    let outcome = OpticalGainEstimatorF32::process(
        prepared,
        command,
        response,
        &[frame],
        phases,
        &[completes],
        &mut ca,
        &mut ra,
        &mut gains,
        &mut lags,
        &mut valid,
        &mut complete,
    );
    let reason = match outcome {
        Ok(_) => 0.0,
        Err(DeclarationProcessError::Algorithm(
            OpticalGainEstimatorError::WindowFrameMismatch { .. },
        )) => 1.0,
        Err(DeclarationProcessError::Algorithm(
            OpticalGainEstimatorError::PhaseIndexMismatch { .. },
        )) => 2.0,
        Err(DeclarationProcessError::Algorithm(
            OpticalGainEstimatorError::WindowCompletionMismatch { .. },
        )) => 3.0,
        Err(DeclarationProcessError::Algorithm(OpticalGainEstimatorError::NonFiniteInput {
            input: OpticalGainEstimatorInputKind::RealizedCommand,
            ..
        })) => 4.0,
        Err(DeclarationProcessError::Algorithm(OpticalGainEstimatorError::NonFiniteInput {
            input: OpticalGainEstimatorInputKind::ReconstructedResponse,
            ..
        })) => 5.0,
        Err(DeclarationProcessError::Algorithm(
            OpticalGainEstimatorError::NonFiniteAccumulation { .. },
        )) => 6.0,
        Err(DeclarationProcessError::Algorithm(OpticalGainEstimatorError::NonFiniteEstimate {
            ..
        })) => 7.0,
        Err(e) => return Err(e.into()),
    };
    let mut values = ca;
    values.extend(ra);
    values.extend(gains);
    values.extend(lags);
    let preserved = values.iter().all(|&v| v == 71.0) && valid.iter().all(|&v| v) && complete[0];
    let mut status = vec![
        reason,
        flag(reason == 0.0 || preserved),
        flag(reason == 0.0 || prepared.workspace().frame_count() == before),
        prepared.workspace().frame_count() as f64,
        flag(complete[0]),
    ];
    status.extend(valid.into_iter().map(flag));
    emit(results, "optical-estimator-f32", case, &values, status);
    Ok(())
}
fn estimator(f: Estimator, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let n = f.modes.len();
    let mut prepared = OpticalGainEstimatorF32::prepare(OpticalGainEstimatorF32Configuration {
        modes: f.modes,
        initial_calibration_responses: f.calibration_responses,
    })?;
    for i in 0..f.commands.len() {
        if i == 1 {
            estimator_step(
                &mut prepared,
                "wrong-frame",
                (&f.commands[i], &f.responses[i]),
                2,
                &f.phase_indices[i],
                f.completes_window[i],
                results,
            )?;
            let mut wrong = f.phase_indices[i].clone();
            wrong[0] = 0;
            estimator_step(
                &mut prepared,
                "wrong-phase",
                (&f.commands[i], &f.responses[i]),
                f.window_frames[i],
                &wrong,
                f.completes_window[i],
                results,
            )?;
            estimator_step(
                &mut prepared,
                "wrong-completion",
                (&f.commands[i], &f.responses[i]),
                f.window_frames[i],
                &f.phase_indices[i],
                true,
                results,
            )?;
            let mut command = f.commands[i].clone();
            command[n - 1] = f32::NAN;
            estimator_step(
                &mut prepared,
                "nan-command",
                (&command, &f.responses[i]),
                f.window_frames[i],
                &f.phase_indices[i],
                f.completes_window[i],
                results,
            )?;
            let mut response = f.responses[i].clone();
            response[n - 1] = f32::INFINITY;
            estimator_step(
                &mut prepared,
                "inf-response",
                (&f.commands[i], &response),
                f.window_frames[i],
                &f.phase_indices[i],
                f.completes_window[i],
                results,
            )?;
        }
        if i == f.replace_before_frame {
            let before = prepared.workspace().frame_count();
            prepared
                .replace_parameter(11, AlgorithmInput::from(f.replacement_responses.as_slice()))
                .ok_or("missing calibrations")??;
            record(
                results,
                "optical-estimator-f32",
                "replacement-status".into(),
                vec![0.0, flag(prepared.workspace().frame_count() == before)],
            );
            let previous = prepared.plan().calibration_responses().to_vec();
            let mut invalid = f.replacement_responses.clone();
            invalid[n - 1] = 0.0;
            let rejected = prepared
                .replace_parameter(11, AlgorithmInput::from(invalid.as_slice()))
                .ok_or("missing calibrations")?
                .is_err();
            record(
                results,
                "optical-estimator-f32",
                "invalid-calibration-status".into(),
                vec![
                    flag(!rejected),
                    flag(prepared.plan().calibration_responses() == previous),
                    flag(prepared.workspace().frame_count() == before),
                ],
            );
        }
        estimator_step(
            &mut prepared,
            &format!("frame-{i}"),
            (&f.commands[i], &f.responses[i]),
            f.window_frames[i],
            &f.phase_indices[i],
            f.completes_window[i],
            results,
        )?;
    }
    // Establish nonzero partial quadratures so reset must discard mathematical history.
    for i in 0..2 {
        estimator_step(
            &mut prepared,
            &format!("partial-before-reset-{i}"),
            (&f.commands[i], &f.responses[i]),
            f.window_frames[i],
            &f.phase_indices[i],
            f.completes_window[i],
            results,
        )?;
    }
    prepared.reset();
    record(
        results,
        "optical-estimator-f32",
        "reset-status".into(),
        vec![prepared.workspace().frame_count() as f64],
    );
    let replay_count = f
        .completes_window
        .iter()
        .position(|&v| v)
        .ok_or("missing complete estimator window")?
        + 1;
    for i in 0..replay_count {
        estimator_step(
            &mut prepared,
            &format!("replay-{i}"),
            (&f.commands[i], &f.responses[i]),
            f.window_frames[i],
            &f.phase_indices[i],
            f.completes_window[i],
            results,
        )?;
    }
    Ok(())
}
fn update_step(
    prepared: &mut PreparedAlgorithm<OpticalGainUpdateF32>,
    case: &UpdateCase,
    results: &mut Vec<Record>,
) -> Result<(), Box<dyn Error>> {
    let n = prepared.plan().reconstructed_mode_count();
    let d = case.command.len();
    let mut gains = vec![71.0; n];
    let mut accepted = vec![true; d];
    let mut updated = vec![true; n];
    let outcome = OpticalGainUpdateF32::process(
        prepared,
        &case.command,
        &case.response,
        &case.gains,
        &case.phases,
        &case.valid,
        &[case.complete],
        &mut gains,
        &mut accepted,
        &mut updated,
    );
    let reason = match outcome {
        Ok(_) => 0.0,
        Err(DeclarationProcessError::Algorithm(OpticalGainUpdateError::NonFiniteInput {
            ..
        })) => 1.0,
        Err(DeclarationProcessError::Algorithm(OpticalGainUpdateError::NonFiniteOutput {
            ..
        })) => 2.0,
        Err(e) => return Err(e.into()),
    };
    let preserved = gains.iter().all(|&v| v == 71.0)
        && accepted.iter().all(|&v| v)
        && updated.iter().all(|&v| v);
    let mut status = vec![reason, flag(reason == 0.0 || preserved)];
    status.extend(accepted.into_iter().map(flag));
    status.extend(updated.into_iter().map(flag));
    emit(results, "optical-update-f32", &case.name, &gains, status);
    Ok(())
}
fn update(f: Update, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let mut prepared = OpticalGainUpdateF32::prepare(OpticalGainUpdateF32Configuration {
        criteria: f.criteria,
        assignments: f.assignments,
        initial_optical_gains: f.initial_gains,
    })?;
    let mut incomplete = f.cases[0].clone();
    incomplete.name = "incomplete-unpublished".into();
    incomplete.complete = false;
    incomplete.command.fill(f32::NAN);
    incomplete.response.fill(f32::NAN);
    incomplete.gains.fill(f32::NAN);
    incomplete.phases.fill(f32::NAN);
    update_step(&mut prepared, &incomplete, results)?;
    for case in &f.cases {
        if case.name == "recovery" {
            let mut invalid = case.clone();
            invalid.name = "nan-estimate".into();
            invalid.gains[0] = f32::NAN;
            update_step(&mut prepared, &invalid, results)?;
        }
        update_step(&mut prepared, case, results)?;
    }
    prepared.reset();
    incomplete.name = "after-reset".into();
    update_step(&mut prepared, &incomplete, results)?;
    let mut replay = f.cases[0].clone();
    replay.name = "after-reset-accepted".into();
    update_step(&mut prepared, &replay, results)?;
    Ok(())
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    correction(f.correction, results)?;
    dither(f.dither, results)?;
    estimator(f.estimator, results)?;
    update(f.update, results)?;
    Ok(())
}
