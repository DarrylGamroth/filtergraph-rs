//! Live corrector-allocation evidence through declared construction and processing.
use super::{Record, record};
use filtergraph_algorithms::{
    CorrectorAllocationError, CorrectorAllocationF32 as D,
    CorrectorAllocationF32Configuration as C, DeclarationProcessError,
};
use serde::Deserialize;
use std::error::Error;

#[derive(Clone, Deserialize)]
struct Configuration {
    controller_command_count: usize,
    stage_command_counts: Vec<usize>,
    update_periods: Vec<usize>,
    command_projection: Vec<f32>,
    realization_projection: Vec<f32>,
    section_count: usize,
    initial_coefficients: Vec<f32>,
    lower_limits: Vec<f32>,
    upper_limits: Vec<f32>,
    slew_limits: Option<Vec<f32>>,
}
impl Configuration {
    fn configuration(&self) -> C {
        C {
            controller_command_count: self.controller_command_count,
            stage_command_counts: self.stage_command_counts.clone(),
            update_periods: self.update_periods.clone(),
            command_projection: self.command_projection.clone(),
            realization_projection: self.realization_projection.clone(),
            section_count: self.section_count,
            initial_coefficients: Some(self.initial_coefficients.clone()),
            command_limits: (self.lower_limits.clone(), self.upper_limits.clone()),
            slew_limits: self.slew_limits.clone(),
            command_schema: "org.example.corrector.commands/1".into(),
        }
    }
}
#[derive(Deserialize)]
struct Allocation {
    #[serde(flatten)]
    configuration: Configuration,
    frames: Vec<Vec<f32>>,
    reject_before_frame: usize,
    reset_before_frame: usize,
}
#[derive(Deserialize)]
struct Failure {
    #[serde(flatten)]
    configuration: Configuration,
    overflow_realization_projection: Vec<f32>,
    projection_input: Vec<f32>,
    projection_recovery: Vec<Vec<f32>>,
    residual_input: Vec<f32>,
    residual_recovery: Vec<Vec<f32>>,
}
#[derive(Deserialize)]
struct Fma {
    #[serde(flatten)]
    configuration: Configuration,
    name: String,
    frames: Vec<Vec<f32>>,
}
#[derive(Deserialize)]
struct Fixture {
    allocation: Allocation,
    failure: Failure,
    fma: Vec<Fma>,
}

pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    const LABEL: &str = "corrector-allocation-f32";
    let f: Fixture = toml::from_str(source)?;
    let mut prepared = D::prepare(f.allocation.configuration.configuration())?;
    let n: usize = f.allocation.configuration.stage_command_counts.iter().sum();
    let mut commands = vec![71.0; n];
    let mut feedback = vec![-17.0; f.allocation.configuration.controller_command_count];
    macro_rules! rejection {
        ($phase:expr,$input:expr) => {{
            let old_commands = commands.clone();
            let old_feedback = feedback.clone();
            let result = D::process(&mut prepared, $input, &mut commands, &mut feedback);
            let reason = match result {
                Ok(()) => 0.0,
                Err(DeclarationProcessError::Algorithm(
                    CorrectorAllocationError::NonFiniteInput,
                )) => 1.0,
                Err(DeclarationProcessError::Algorithm(
                    CorrectorAllocationError::NonFiniteOutput,
                )) => 2.0,
                Err(e) => return Err(e.into()),
            };
            record(
                results,
                LABEL,
                format!("{}-status", $phase),
                vec![
                    reason,
                    if commands == old_commands && feedback == old_feedback {
                        1.0
                    } else {
                        0.0
                    },
                ],
            );
        }};
    }
    macro_rules! accepted {
        ($phase:expr,$input:expr) => {{
            D::process(&mut prepared, $input, &mut commands, &mut feedback)?;
            record(
                results,
                LABEL,
                format!("{}-commands", $phase),
                commands.iter().map(|&v| v as f64).collect(),
            );
            record(
                results,
                LABEL,
                format!("{}-feedback", $phase),
                feedback.iter().map(|&v| v as f64).collect(),
            );
        }};
    }
    for (frame, input) in f.allocation.frames.iter().enumerate() {
        if frame == f.allocation.reset_before_frame {
            prepared.reset();
        }
        if frame == f.allocation.reject_before_frame {
            for (name, value) in [
                ("nan", f32::NAN),
                ("positive-inf", f32::INFINITY),
                ("negative-inf", f32::NEG_INFINITY),
                ("projection-overflow", f32::MAX),
            ] {
                let mut invalid = input.clone();
                if name == "projection-overflow" {
                    invalid.fill(value);
                } else {
                    *invalid.last_mut().ok_or("empty corrector demand")? = value;
                }
                rejection!(name, &invalid);
            }
        }
        accepted!(format!("frame-{frame}"), input);
    }
    for c in &f.fma {
        prepared = D::prepare(c.configuration.configuration())?;
        commands = vec![71.0; c.configuration.stage_command_counts.iter().sum()];
        feedback = vec![-17.0; c.configuration.controller_command_count];
        for (frame, input) in c.frames.iter().enumerate() {
            accepted!(format!("fma-{}-frame-{frame}", c.name), input);
        }
        prepared.reset();
        accepted!(format!("fma-{}-after-reset", c.name), &c.frames[0]);
    }
    for kind in ["late-projection", "late-residual"] {
        let mut c = f.failure.configuration.configuration();
        if kind == "late-residual" {
            c.realization_projection = f.failure.overflow_realization_projection.clone();
        }
        prepared = D::prepare(c)?;
        commands.fill(71.0);
        commands.resize(
            f.failure.configuration.stage_command_counts.iter().sum(),
            71.0,
        );
        feedback.fill(-17.0);
        feedback.resize(f.failure.configuration.controller_command_count, -17.0);
        let (input, recovery) = if kind == "late-projection" {
            (&f.failure.projection_input, &f.failure.projection_recovery)
        } else {
            (&f.failure.residual_input, &f.failure.residual_recovery)
        };
        rejection!(kind, input);
        for (frame, values) in recovery.iter().enumerate() {
            accepted!(format!("{kind}-recovery-{frame}"), values);
        }
    }
    type PreparationChange = fn(&mut C);
    let changes: [(&str, PreparationChange); 10] = [
        ("empty-controller", |c| c.controller_command_count = 0),
        ("empty-stage", |c| c.stage_command_counts[0] = 0),
        ("period-zero", |c| c.update_periods[0] = 0),
        ("missing-coefficients", |c| c.initial_coefficients = None),
        ("projection-nonfinite", |c| {
            c.command_projection[0] = f32::NAN
        }),
        ("exclude-zero", |c| c.command_limits.0[0] = 0.25),
        ("unstable-filter", |c| {
            c.initial_coefficients.as_mut().unwrap()[3] = 2.0
        }),
        ("port-cap", |c| {
            c.controller_command_count = u32::MAX as usize
        }),
        ("projection-cap", |c| {
            c.controller_command_count = 1 << 20;
            c.stage_command_counts = vec![1 << 20];
            c.update_periods = vec![1];
        }),
        ("coefficient-cap", |c| c.section_count = usize::MAX),
    ];
    for (name, change) in changes {
        let mut c = f.allocation.configuration.configuration();
        change(&mut c);
        record(
            results,
            LABEL,
            format!("configuration-{name}-status"),
            vec![if D::prepare(c).is_err() { 0.0 } else { 1.0 }],
        );
    }
    Ok(())
}
