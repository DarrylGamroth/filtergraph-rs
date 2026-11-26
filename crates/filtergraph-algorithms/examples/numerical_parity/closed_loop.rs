//! Closed-loop correction and zero-hidden-mode integrator declaration lifecycle.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, ClosedLoopCorrectionError as E, ClosedLoopCorrectionF32 as C,
    ClosedLoopCorrectionF32Configuration, ClosedLoopIntegratorF32 as I,
    ClosedLoopIntegratorF32Configuration, DeclarationProcessError, PropertyAssignment,
    PropertyValue,
};
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    gain: f32,
    pole: f32,
    anti_windup_gain: f32,
    hidden_mode_gain: f32,
    shape_to_hidden: Vec<f32>,
    hidden_to_shape: Vec<f32>,
    frames: Vec<Vec<f32>>,
    feedback: Vec<Vec<f32>>,
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let n = f.frames[0].len();
    macro_rules! execute {
        ($d:ident,$prepared:expr,$label:literal,$hidden:expr) => {{
            let mut p = $prepared;
            if $hidden {
                p.replace_parameter(4, AlgorithmInput::from(f.shape_to_hidden.as_slice()))
                    .ok_or("missing projection")??;
                p.replace_parameter(5, AlgorithmInput::from(f.hidden_to_shape.as_slice()))
                    .ok_or("missing projection")??;
            }
            let properties = [
                PropertyAssignment {
                    id: 0,
                    value: PropertyValue::Float(f.gain),
                },
                PropertyAssignment {
                    id: 1,
                    value: PropertyValue::Float(f.pole),
                },
                PropertyAssignment {
                    id: 2,
                    value: PropertyValue::Float(f.anti_windup_gain),
                },
                PropertyAssignment {
                    id: 3,
                    value: PropertyValue::Float(f.hidden_mode_gain),
                },
            ];
            p.replace_properties(&properties)?;
            let mut out = vec![71.0; n];
            let mut state = vec![-17.0; n];
            for (i, frame) in f.frames.iter().enumerate() {
                let feedback = if i == 0 {
                    vec![f32::NAN; n]
                } else {
                    f.feedback[i].clone()
                };
                $d::process(&mut p, frame, &feedback, &mut out, &mut state)?;
                record(
                    results,
                    $label,
                    format!("frame-{i}-output"),
                    out.iter().map(|&v| v as f64).collect(),
                );
                record(
                    results,
                    $label,
                    format!("frame-{i}-state"),
                    state.iter().map(|&v| v as f64).collect(),
                );
            }
            let rejected = p
                .replace_properties(&[PropertyAssignment {
                    id: 1,
                    value: PropertyValue::Float(f32::NAN),
                }])
                .is_err();
            record(
                results,
                $label,
                "invalid-property-status".into(),
                vec![rejected as u8 as f64],
            );
            for (case, bad_input, bad_feedback) in
                [("nan-input", true, false), ("nan-feedback", false, true)]
            {
                let mut input = f.frames[1].clone();
                let mut feedback = f.feedback[1].clone();
                if bad_input {
                    input[n - 1] = f32::NAN;
                }
                if bad_feedback {
                    feedback[n - 1] = f32::NAN;
                }
                let old_out = out.clone();
                let old_state = state.clone();
                let outcome = $d::process(&mut p, &input, &feedback, &mut out, &mut state);
                let reason = match outcome {
                    Err(DeclarationProcessError::Algorithm(E::NonFiniteInput)) => 1,
                    Err(DeclarationProcessError::Algorithm(E::NonFiniteFeedback)) => 2,
                    Err(DeclarationProcessError::Algorithm(E::NonFiniteOutput)) => 3,
                    Ok(()) => 0,
                    Err(e) => return Err(e.into()),
                };
                record(
                    results,
                    $label,
                    format!("{case}-status"),
                    vec![
                        reason as f64,
                        (out == old_out && state == old_state) as u8 as f64,
                    ],
                );
                $d::process(&mut p, &f.frames[1], &f.feedback[1], &mut out, &mut state)?;
                record(
                    results,
                    $label,
                    format!("after-{case}-output"),
                    out.iter().map(|&v| v as f64).collect(),
                );
                record(
                    results,
                    $label,
                    format!("after-{case}-state"),
                    state.iter().map(|&v| v as f64).collect(),
                );
            }
            p.replace_properties(&[PropertyAssignment {
                id: 0,
                value: PropertyValue::Float(f32::MAX),
            }])?;
            let old_out = out.clone();
            let old_state = state.clone();
            let rejected = matches!(
                $d::process(&mut p, &vec![2.0; n], &vec![0.0; n], &mut out, &mut state),
                Err(DeclarationProcessError::Algorithm(E::NonFiniteOutput))
            );
            record(
                results,
                $label,
                "overflow-status".into(),
                vec![
                    rejected as u8 as f64,
                    (out == old_out && state == old_state) as u8 as f64,
                ],
            );
            p.replace_properties(&properties)?;
            $d::process(&mut p, &f.frames[1], &f.feedback[1], &mut out, &mut state)?;
            record(
                results,
                $label,
                "after-overflow-output".into(),
                out.iter().map(|&v| v as f64).collect(),
            );
            record(
                results,
                $label,
                "after-overflow-state".into(),
                state.iter().map(|&v| v as f64).collect(),
            );
            p.reset();
            $d::process(
                &mut p,
                &f.frames[0],
                &vec![f32::NAN; n],
                &mut out,
                &mut state,
            )?;
            record(
                results,
                $label,
                "reset-output".into(),
                out.iter().map(|&v| v as f64).collect(),
            );
            record(
                results,
                $label,
                "reset-state".into(),
                state.iter().map(|&v| v as f64).collect(),
            );
            for phase in ["held", "held-nan-feedback", "resumed", "held-reset"] {
                p.replace_properties(&[PropertyAssignment {
                    id: 4,
                    value: PropertyValue::Bool(phase == "resumed"),
                }])?;
                if phase == "held-reset" {
                    p.reset();
                }
                let feedback = vec![
                    if phase == "held-nan-feedback" {
                        f32::NAN
                    } else {
                        0.0
                    };
                    n
                ];
                let old_out = out.clone();
                let old_state = state.clone();
                let outcome = $d::process(&mut p, &f.frames[1], &feedback, &mut out, &mut state);
                let reason = match outcome {
                    Ok(()) => 0,
                    Err(DeclarationProcessError::Algorithm(E::NonFiniteFeedback)) => 2,
                    Err(e) => return Err(e.into()),
                };
                record(
                    results,
                    $label,
                    format!("{phase}-status"),
                    vec![
                        reason as f64,
                        (reason == 0 || (out == old_out && state == old_state)) as u8 as f64,
                    ],
                );
                record(
                    results,
                    $label,
                    format!("{phase}-output"),
                    out.iter().map(|&v| v as f64).collect(),
                );
                record(
                    results,
                    $label,
                    format!("{phase}-state"),
                    state.iter().map(|&v| v as f64).collect(),
                );
            }
        }};
    }
    execute!(
        I,
        I::prepare(ClosedLoopIntegratorF32Configuration { state_length: n })?,
        "closed-loop-integrator-f32",
        false
    );
    execute!(
        C,
        C::prepare(ClosedLoopCorrectionF32Configuration {
            state_length: n,
            hidden_mode_count: 1
        })?,
        "closed-loop-correction-f32",
        true
    );
    Ok(())
}
