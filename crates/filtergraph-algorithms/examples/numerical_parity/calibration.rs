//! Shared-input calibration cycles and structured decoding, with logical row-major records.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmPlan, CalibrationProbeOutput, CalibrationProbePolarity, CalibrationProbeStep,
    HadamardInteractionMatrixPlan, HadamardPushPullPlan, ModalPushPullPlan,
    StructuredInteractionMatrixError, ZonalInteractionMatrixPlan, ZonalPushPullPlan,
};
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize)]
struct Fixture {
    probe: Probe,
    decode: Decode,
}
#[derive(Deserialize)]
struct Probe {
    mode_count: usize,
    coordinate_count: usize,
    positive_commands: Vec<f64>,
    amplitudes: Vec<f64>,
    cycles: usize,
    output_sentinel: f64,
}
#[derive(Deserialize)]
struct Decode {
    measurement_count: usize,
    amplitudes: Vec<f64>,
    output_sentinel: f64,
    zonal_responses: Vec<f64>,
    hadamard_responses: Vec<f64>,
}

pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    macro_rules! precision {
        ($t:ty, $suffix:literal) => {{
            let p = &f.probe;
            let amps: Vec<$t> = p.amplitudes.iter().map(|&x| x as $t).collect();
            macro_rules! probe {
                ($plan:expr, $kind:literal) => {{
                    let plan = $plan;
                    let label = concat!($kind, "-push-pull-", $suffix);
                    let mut workspace = plan.workspace();
                    let mut excitation = vec![p.output_sentinel as $t; p.coordinate_count];
                    let mut step = CalibrationProbeStep::default();
                    macro_rules! emit {
                        ($phase:expr) => {{
                            plan.process(
                                CalibrationProbeOutput::new(&mut excitation, &mut step),
                                &mut workspace,
                                (),
                            )?;
                            record(
                                results,
                                label,
                                format!("{}-excitation", $phase),
                                excitation.iter().map(|&x| x as f64).collect(),
                            );
                            record(
                                results,
                                label,
                                format!("{}-status", $phase),
                                vec![
                                    step.pattern_index() as f64,
                                    (step.polarity() == CalibrationProbePolarity::Negative) as u8
                                        as f64,
                                    step.completes_cycle() as u8 as f64,
                                ],
                            );
                        }};
                    }
                    for frame in 0..p.cycles * plan.cycle_len() {
                        emit!(format!("frame-{frame}"));
                    }
                    let previous = workspace;
                    let mut bad = vec![p.output_sentinel as $t; p.coordinate_count - 1];
                    let old_step = step;
                    let rejected = plan
                        .process(
                            CalibrationProbeOutput::new(&mut bad, &mut step),
                            &mut workspace,
                            (),
                        )
                        .is_err();
                    record(
                        results,
                        label,
                        "invalid-extent-status".into(),
                        vec![
                            (!rejected) as u8 as f64,
                            (workspace == previous) as u8 as f64,
                            (step == old_step) as u8 as f64,
                            bad.iter().all(|&x| x == p.output_sentinel as $t) as u8 as f64,
                        ],
                    );
                    emit!("after-invalid");
                    workspace.reset();
                    emit!("after-reset");
                }};
            }
            probe!(
                ModalPushPullPlan::new(
                    p.mode_count,
                    p.coordinate_count,
                    p.positive_commands
                        .iter()
                        .map(|&x| x as $t)
                        .collect::<Vec<_>>()
                )?,
                "modal"
            );
            probe!(ZonalPushPullPlan::new(amps.clone())?, "zonal");
            probe!(HadamardPushPullPlan::new(amps)?, "hadamard");
            let d = &f.decode;
            let amps: Vec<$t> = d.amplitudes.iter().map(|&x| x as $t).collect();
            macro_rules! decode {
                ($plan:expr, $responses:expr, $kind:literal) => {{
                    let plan = $plan;
                    let label = concat!($kind, "-interaction-matrix-", $suffix);
                    let mut workspace = plan.workspace();
                    let responses: Vec<$t> = $responses.iter().map(|&x| x as $t).collect();
                    let mut output =
                        vec![d.output_sentinel as $t; d.measurement_count * amps.len()];
                    plan.process(&mut output, &mut workspace, &responses)?;
                    record(
                        results,
                        label,
                        "complete-output".into(),
                        output.iter().map(|&x| x as f64).collect(),
                    );
                    for (name, value) in [
                        ("nan", <$t>::NAN),
                        ("positive-inf", <$t>::INFINITY),
                        ("negative-inf", <$t>::NEG_INFINITY),
                    ] {
                        let mut invalid = responses.clone();
                        *invalid.last_mut().unwrap() = value;
                        output.fill(d.output_sentinel as $t);
                        let outcome = plan.process(&mut output, &mut workspace, &invalid);
                        let rejected = matches!(
                            outcome,
                            Err(StructuredInteractionMatrixError::NonFiniteResponse { .. })
                        );
                        record(
                            results,
                            label,
                            format!("{name}-status"),
                            vec![
                                (!rejected) as u8 as f64,
                                output.iter().all(|&x| x == d.output_sentinel as $t) as u8 as f64,
                            ],
                        );
                        plan.process(&mut output, &mut workspace, &responses)?;
                        record(
                            results,
                            label,
                            format!("after-{name}-output"),
                            output.iter().map(|&x| x as f64).collect(),
                        );
                    }
                }};
            }
            decode!(
                ZonalInteractionMatrixPlan::new(d.measurement_count, amps.clone())?,
                d.zonal_responses,
                "zonal"
            );
            decode!(
                HadamardInteractionMatrixPlan::new(d.measurement_count, amps.clone())?,
                d.hadamard_responses,
                "hadamard"
            );
        }};
    }
    precision!(f32, "f32");
    precision!(f64, "f64");
    Ok(())
}
