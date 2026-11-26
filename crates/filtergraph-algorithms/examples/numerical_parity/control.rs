//! Integrator lifecycle evidence; all nonfinite experiments emit finite status flags.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmPlan, AlgorithmProperties, LeakyIntegratorError, LeakyIntegratorPlan,
    LeakyIntegratorWorkspace, NoWorkspace, PdmCommandInput, PdmCommandOutput, PdmCommandPlan,
    PdmEngineeringOffsetOutput, PdmEngineeringOffsetPlan, PropertyAssignment, PropertyValue,
};
use serde::Deserialize;
use std::{error::Error, sync::Arc};

#[derive(Deserialize)]
pub(super) struct Fixture {
    extent: usize,
    initial_state: f64,
    gain: f64,
    pole: f64,
    input: Vec<f64>,
    recovery_input: Vec<f64>,
    replacement_gain: f64,
    replacement_pole: f64,
    invalid_pole: f64,
    output_sentinel: f64,
}

pub(super) fn run(f: &Fixture, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    macro_rules! lifecycle {
        ($t:ty, $property:ident, $label:literal) => {{
            let mut plan = LeakyIntegratorPlan::new(f.gain as $t, f.pole as $t)?;
            let mut workspace = LeakyIntegratorWorkspace::filled(f.extent, f.initial_state as $t)?;
            let mut output = vec![f.output_sentinel as $t; f.extent];
            let input: Vec<$t> = f.input.iter().map(|&v| v as $t).collect();
            let recovery: Vec<$t> = f.recovery_input.iter().map(|&v| v as $t).collect();
            record(
                results,
                $label,
                "initial-state".into(),
                workspace.state().iter().map(|&v| v as f64).collect(),
            );
            macro_rules! process {
                ($phase:expr, $plan:expr, $input:expr) => {{
                    output.fill(f.output_sentinel as $t);
                    let previous_output = output.clone();
                    let previous_state = workspace.state().to_vec();
                    let outcome = $plan.process(&mut output, &mut workspace, $input);
                    let accepted = outcome.is_ok();
                    let rejection = match outcome {
                        Ok(()) => 0.0,
                        Err(LeakyIntegratorError::NonFiniteInput) => 1.0,
                        Err(LeakyIntegratorError::NonFiniteOutput) => 2.0,
                        Err(error) => return Err(error.into()),
                    };
                    let flags = [
                        accepted,
                        accepted || output == previous_output,
                        accepted || workspace.state() == previous_state,
                        output.iter().all(|v| v.is_finite()),
                        workspace.state().iter().all(|v| v.is_finite()),
                    ];
                    record(
                        results,
                        $label,
                        format!("{}-status", $phase),
                        flags
                            .into_iter()
                            .map(|v| if v { 1.0 } else { 0.0 })
                            .chain([rejection])
                            .collect(),
                    );
                    // Nonfinite values cannot compare equal by accident: status flags above fail.
                    record(
                        results,
                        $label,
                        format!("{}-output", $phase),
                        output
                            .iter()
                            .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
                            .collect(),
                    );
                    record(
                        results,
                        $label,
                        format!("{}-state", $phase),
                        workspace
                            .state()
                            .iter()
                            .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
                            .collect(),
                    );
                }};
            }
            process!("first", plan, &input);
            for (name, value) in [
                ("nan", <$t>::NAN),
                ("positive-inf", <$t>::INFINITY),
                ("negative-inf", <$t>::NEG_INFINITY),
            ] {
                let mut invalid = input.clone();
                *invalid.last_mut().ok_or("empty lifecycle input")? = value;
                process!(name, plan, &invalid);
                process!(format!("after-{name}"), plan, &recovery);
            }
            let overflow_plan = LeakyIntegratorPlan::new(<$t>::MAX, 1.0 as $t)?;
            let overflow_input = vec![2.0 as $t; f.extent];
            process!("overflow", overflow_plan, &overflow_input);
            process!("after-overflow", plan, &recovery);
            let previous_state = workspace.state().to_vec();
            plan = plan.prepare_properties(&[
                PropertyAssignment {
                    id: 0,
                    value: PropertyValue::$property(f.replacement_gain as $t),
                },
                PropertyAssignment {
                    id: 1,
                    value: PropertyValue::$property(f.replacement_pole as $t),
                },
            ])?;
            record(
                results,
                $label,
                "replacement-status".into(),
                vec![
                    1.0,
                    if workspace.state() == previous_state {
                        1.0
                    } else {
                        0.0
                    },
                    plan.gain() as f64,
                    plan.pole() as f64,
                ],
            );
            process!("after-replacement", plan, &input);
            for (name, gain, pole) in [
                ("invalid-pole", f.gain as $t, f.invalid_pole as $t),
                ("invalid-gain", <$t>::NAN, f.pole as $t),
            ] {
                let previous_state = workspace.state().to_vec();
                let previous_plan = plan;
                let rejected = plan
                    .prepare_properties(&[
                        PropertyAssignment {
                            id: 0,
                            value: PropertyValue::$property(gain),
                        },
                        PropertyAssignment {
                            id: 1,
                            value: PropertyValue::$property(pole),
                        },
                    ])
                    .is_err();
                record(
                    results,
                    $label,
                    format!("{name}-status"),
                    vec![
                        if rejected { 0.0 } else { 1.0 },
                        if plan == previous_plan { 1.0 } else { 0.0 },
                        if workspace.state() == previous_state {
                            1.0
                        } else {
                            0.0
                        },
                    ],
                );
                process!(format!("after-{name}"), plan, &recovery);
            }
            workspace.reset();
            record(
                results,
                $label,
                "reset-state".into(),
                workspace.state().iter().map(|&v| v as f64).collect(),
            );
            process!("after-reset", plan, &input);
            let held = plan.prepare_properties(&[PropertyAssignment {
                id: 2,
                value: PropertyValue::Bool(false),
            }])?;
            process!("held", held, &input);
            process!("held-nan", held, &vec![<$t>::NAN; f.extent]);
            let resumed = held.prepare_properties(&[PropertyAssignment {
                id: 2,
                value: PropertyValue::Bool(true),
            }])?;
            process!("resumed", resumed, &input);
            workspace.reset();
            process!("held-reset", held, &input);
            workspace = LeakyIntegratorWorkspace::filled(f.extent, -<$t>::MAX)?;
            process!("fused-cancellation", overflow_plan, &overflow_input);
        }};
    }
    lifecycle!(f32, Float, "leaky-lifecycle-f32");
    lifecycle!(f64, Double, "leaky-lifecycle-f64");
    Ok(())
}

#[derive(Deserialize)]
pub(super) struct PdmFixture {
    engineering_offset: Vec<f32>,
    replacement_offset: Vec<f32>,
    lower_limits: Vec<f32>,
    upper_limits: Vec<f32>,
    initial_current: Vec<f32>,
    frames: Vec<Vec<f32>>,
    replace_before_frame: usize,
}

pub(super) fn pdm(f: PdmFixture, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    const LABEL: &str = "pdm-offset-command-f32";
    let mut offset = PdmEngineeringOffsetPlan::new(Arc::from(f.engineering_offset))?;
    let command = PdmCommandPlan::new(Arc::from(f.lower_limits), Arc::from(f.upper_limits), None)?;
    let mut workspace = command.workspace(&f.initial_current)?;
    let mut composed = vec![0.0; f.initial_current.len()];
    let mut current_offset = composed.clone();
    let mut demanded = composed.clone();
    let mut feedback = composed.clone();
    for (frame, input) in f.frames.iter().enumerate() {
        if frame == f.replace_before_frame {
            let previous = offset.clone();
            let wrong_extent = offset
                .with_engineering_offset(Arc::from(vec![0.0_f32]))
                .is_err();
            let mut invalid = f.replacement_offset.clone();
            *invalid.last_mut().ok_or("empty PDM offset")? = f32::NAN;
            let nonfinite = offset.with_engineering_offset(Arc::from(invalid)).is_err();
            record(
                results,
                LABEL,
                "invalid-replacement-status".into(),
                vec![
                    if wrong_extent { 0.0 } else { 1.0 },
                    if nonfinite { 0.0 } else { 1.0 },
                    if offset == previous { 1.0 } else { 0.0 },
                ],
            );
            offset = offset.with_engineering_offset(Arc::from(f.replacement_offset.clone()))?;
        }
        offset.process(
            PdmEngineeringOffsetOutput {
                requested_with_offset: &mut composed,
                current_offset: &mut current_offset,
            },
            &mut NoWorkspace,
            input,
        )?;
        command.process(
            PdmCommandOutput {
                demanded: &mut demanded,
                constraint_feedback: &mut feedback,
            },
            &mut workspace,
            PdmCommandInput {
                requested: &composed,
            },
        )?;
        for (name, values) in [
            ("composed", composed.as_slice()),
            ("offset", current_offset.as_slice()),
            ("demanded", demanded.as_slice()),
            ("feedback", feedback.as_slice()),
            ("state", workspace.current()),
        ] {
            record(
                results,
                LABEL,
                format!("frame-{frame}-{name}"),
                values.iter().copied().map(f64::from).collect(),
            );
        }
    }
    Ok(())
}
