//! Stroke and slew limiting through the PDM and same-unit vector declarations.
use super::{Record, record};
use filtergraph_algorithms::{
    DeclarationProcessError, PdmCommandError, PdmCommandF32, PdmCommandF32Configuration,
    ScalarOrVector, VectorConditioningF32, VectorConditioningF32Configuration,
};
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    lower: Vec<f32>,
    upper: Vec<f32>,
    slew: Vec<f32>,
    initial: Vec<f32>,
    frames: Vec<Vec<f32>>,
    overflow_limit: f32,
}
type Limits = (ScalarOrVector<f32>, ScalarOrVector<f32>);

#[derive(Default)]
struct LegacySettings {
    origin: Option<f32>,
    step: Option<f32>,
    seed: Option<u64>,
}

pub(super) fn run(
    source: &str,
    check_legacy: bool,
    results: &mut Vec<Record>,
) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    trajectory(
        &f,
        "pdm-command-f32",
        |actuator_count, command_limits, slew_limit, initial_current, legacy| {
            Ok(PdmCommandF32::prepare(PdmCommandF32Configuration {
                demanded_schema: filtergraph_algorithms::schemas::DEMANDED_PDM_COMMAND_V1.into(),
                actuator_count,
                command_limits,
                slew_limit,
                initial_current,
                quantization_origin: legacy.origin,
                quantization_step: legacy.step,
                dither_seed: legacy.seed,
            })?)
        },
        PdmCommandF32::process,
        |prepared| prepared.reset(),
        check_legacy,
        results,
    )?;
    trajectory(
        &f,
        "vector-conditioning-f32",
        |count, command_limits, slew_limit, initial_current, legacy| {
            Ok(VectorConditioningF32::prepare(
                VectorConditioningF32Configuration {
                    count,
                    requested_schema: "org.example.parity.vector/1".into(),
                    demanded_schema: "org.example.parity.vector/1".into(),
                    feedback_schema: "org.example.parity.vector/1".into(),
                    command_limits,
                    slew_limit,
                    initial_current,
                    quantization_origin: legacy.origin,
                    quantization_step: legacy.step,
                    dither_seed: legacy.seed,
                },
            )?)
        },
        VectorConditioningF32::process,
        |prepared| prepared.reset(),
        check_legacy,
        results,
    )
}

fn trajectory<P>(
    f: &Fixture,
    label: &str,
    prepare: impl Fn(
        usize,
        Limits,
        Option<ScalarOrVector<f32>>,
        Option<Vec<f32>>,
        LegacySettings,
    ) -> Result<P, Box<dyn Error>>,
    process: impl Fn(
        &mut P,
        &[f32],
        &mut [f32],
        &mut [f32],
    ) -> Result<(), DeclarationProcessError<PdmCommandError>>,
    reset: impl Fn(&mut P),
    check_legacy: bool,
    results: &mut Vec<Record>,
) -> Result<(), Box<dyn Error>> {
    let n = f.initial.len();
    let prepare_scenario = |slew_limit, legacy| {
        prepare(
            n,
            (
                ScalarOrVector::Vector(f.lower.clone()),
                ScalarOrVector::Vector(f.upper.clone()),
            ),
            slew_limit,
            Some(f.initial.clone()),
            legacy,
        )
    };
    for (mode, slew_limit) in [
        ("bounds", None),
        ("slew", Some(ScalarOrVector::Vector(f.slew.clone()))),
    ] {
        let mut d = prepare_scenario(slew_limit, LegacySettings::default())?;
        let mut out = vec![71.0; n];
        let mut feedback = vec![71.0; n];
        for phase in ["first", "after-reject", "third", "fourth", "reset"] {
            let i = match phase {
                "first" | "reset" => 0,
                "after-reject" => 1,
                "third" => 2,
                _ => 3,
            };
            if phase == "reset" {
                reset(&mut d);
            }
            if phase == "after-reject" {
                for (name, value) in [
                    ("nan", f32::NAN),
                    ("positive-inf", f32::INFINITY),
                    ("negative-inf", f32::NEG_INFINITY),
                ] {
                    let mut invalid = f.frames[i].clone();
                    invalid[n - 1] = value;
                    let old_out = out.clone();
                    let old_feedback = feedback.clone();
                    let rejected = matches!(
                        process(&mut d, &invalid, &mut out, &mut feedback),
                        Err(DeclarationProcessError::Algorithm(
                            PdmCommandError::NonFiniteInput
                        ))
                    );
                    record(
                        results,
                        label,
                        format!("{mode}-{name}-status"),
                        vec![
                            rejected as u8 as f64,
                            (out == old_out && feedback == old_feedback) as u8 as f64,
                        ],
                    );
                }
            }
            process(&mut d, &f.frames[i], &mut out, &mut feedback)?;
            for (output, values) in [("demanded", &out), ("feedback", &feedback)] {
                record(
                    results,
                    label,
                    format!("{mode}-{phase}-{output}"),
                    values.iter().map(|&v| v as f64).collect(),
                );
            }
        }
    }
    // Recovery would advance the first actuator twice if the rejected frame committed.
    let mut d = prepare(
        2,
        (
            ScalarOrVector::Scalar(-f.overflow_limit),
            ScalarOrVector::Scalar(f.overflow_limit),
        ),
        Some(ScalarOrVector::Scalar(1.0)),
        Some(vec![0.0, -f.overflow_limit]),
        LegacySettings::default(),
    )?;
    let mut output = [71.0; 2];
    let mut feedback = [-17.0; 2];
    let rejected = matches!(
        process(&mut d, &[2.0, f.overflow_limit], &mut output, &mut feedback),
        Err(DeclarationProcessError::Algorithm(
            PdmCommandError::NonFiniteOutput
        ))
    );
    record(
        results,
        label,
        "feedback-overflow-status".into(),
        vec![
            rejected as u8 as f64,
            (output == [71.0; 2] && feedback == [-17.0; 2]) as u8 as f64,
        ],
    );
    process(
        &mut d,
        &[2.0, -f.overflow_limit],
        &mut output,
        &mut feedback,
    )?;
    record(
        results,
        label,
        "after-feedback-overflow-output".into(),
        output.iter().map(|&v| v as f64).collect(),
    );
    record(
        results,
        label,
        "after-feedback-overflow-feedback".into(),
        feedback.iter().map(|&v| v as f64).collect(),
    );
    if check_legacy {
        let slew_limit = Some(ScalarOrVector::Vector(f.slew.clone()));
        for (name, origin, step) in [
            ("origin", Some(0.0), None),
            ("step", None, Some(0.25)),
            ("grid", Some(0.0), Some(0.25)),
        ] {
            let rejected = prepare_scenario(
                slew_limit.clone(),
                LegacySettings {
                    origin,
                    step,
                    seed: None,
                },
            )
            .is_err_and(|error| error.to_string().contains("device driver"));
            if !rejected {
                return Err(
                    format!("{label} did not reject legacy {name} with driver guidance").into(),
                );
            }
            record(results, label, format!("legacy-{name}-status"), vec![1.0]);
        }
        let mut unseeded = prepare_scenario(slew_limit.clone(), LegacySettings::default())?;
        let mut seeded = prepare_scenario(
            slew_limit,
            LegacySettings {
                seed: Some(u64::MAX),
                ..LegacySettings::default()
            },
        )?;
        let (mut demanded, mut feedback) = (vec![0.0; n], vec![0.0; n]);
        let (mut seeded_demanded, mut seeded_feedback) = (demanded.clone(), feedback.clone());
        for frame in &f.frames {
            process(&mut unseeded, frame, &mut demanded, &mut feedback)?;
            process(
                &mut seeded,
                frame,
                &mut seeded_demanded,
                &mut seeded_feedback,
            )?;
            if demanded != seeded_demanded || feedback != seeded_feedback {
                return Err(
                    format!("{label} changed its command or feedback with a legacy seed").into(),
                );
            }
        }
        record(results, label, "legacy-seed-status".into(), vec![1.0]);
    }
    Ok(())
}
