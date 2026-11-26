//! Current JFG scientific additions: atomic calibration, statistics, and validity gating.
use super::{Record, record};
use filtergraph_algorithms::*;
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    sentinel: f32,
    physical: Physical,
    statistics: Statistics,
    gated: Gated,
}
#[derive(Deserialize)]
struct Physical {
    residual: Vec<f32>,
    offset: Vec<f32>,
    physical_to_modes: Vec<f32>,
    modes_to_reference: Vec<f32>,
    reference_sign: f32,
}
#[derive(Deserialize)]
struct Statistics {
    size: Vec<usize>,
    window_samples: usize,
    time_constant_s: f64,
    sample_period_s: f64,
    frames: Vec<Vec<f32>>,
}
#[derive(Deserialize)]
struct Gated {
    gain: f32,
    pole: f32,
    anti_windup_gain: f32,
    frames: Vec<Vec<f32>>,
    feedback: Vec<Vec<f32>>,
    validity: Vec<Vec<bool>>,
}
fn values(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let c = &f.physical;
    let n = c.residual.len();
    let k = c.offset.len();
    let m = c.physical_to_modes.len() / k;
    let mut p = PhysicalOffsetReferenceF32::prepare(PhysicalOffsetReferenceF32Configuration {
        residual_count: n,
        physical_count: k,
        mode_count: m,
        reference_sign: c.reference_sign,
        residual_schema: "test.residual/1".into(),
        physical_schema: "test.physical/1".into(),
        physical_to_modes_schema: "test.physical-to-modes/1".into(),
        modes_to_reference_schema: "test.modes-to-reference/1".into(),
    })?;
    let (mut y, mut offset) = (vec![f.sentinel; n], vec![f.sentinel; k]);
    for phase in [
        "initial",
        "calibrated",
        "offset-replaced",
        "factor-replaced",
        "reset",
        "after-invalid",
    ] {
        if phase == "calibrated" {
            p.replace_parameter(4, AlgorithmInput::from(c.physical_to_modes.as_slice()))
                .unwrap()?;
            p.replace_parameter(5, AlgorithmInput::from(c.modes_to_reference.as_slice()))
                .unwrap()?;
            p.replace_parameter(3, AlgorithmInput::from(c.offset.as_slice()))
                .unwrap()?;
        }
        if phase == "offset-replaced" {
            let v: Vec<_> = c.offset.iter().map(|v| -v).collect();
            p.replace_parameter(3, AlgorithmInput::from(v.as_slice()))
                .unwrap()?;
        }
        if phase == "factor-replaced" {
            let v: Vec<_> = c.modes_to_reference.iter().map(|v| 2.0 * v).collect();
            p.replace_parameter(5, AlgorithmInput::from(v.as_slice()))
                .unwrap()?;
        }
        if phase == "reset" {
            p.reset();
        }
        if phase == "after-invalid" {
            let invalid = vec![f32::NAN; k];
            let rejected = p
                .replace_parameter(3, AlgorithmInput::from(invalid.as_slice()))
                .unwrap()
                .is_err();
            record(
                results,
                "physical-offset-reference-f32",
                "invalid-replacement-status".into(),
                vec![rejected as u8 as f64],
            );
        }
        PhysicalOffsetReferenceF32::process(&mut p, &c.residual, &mut y, &mut offset)?;
        record(
            results,
            "physical-offset-reference-f32",
            format!("{phase}-output"),
            values(&y),
        );
        record(
            results,
            "physical-offset-reference-f32",
            format!("{phase}-offset"),
            values(&offset),
        );
    }
    y.fill(f.sentinel);
    offset.fill(f.sentinel);
    let invalid = vec![f32::INFINITY; n];
    let rejected =
        PhysicalOffsetReferenceF32::process(&mut p, &invalid, &mut y, &mut offset).is_err();
    record(
        results,
        "physical-offset-reference-f32",
        "nonfinite-status".into(),
        vec![
            rejected as u8 as f64,
            y.iter().chain(&offset).all(|&x| x == f.sentinel) as u8 as f64,
        ],
    );
    let mut pixel = PixelValidityF32::prepare(PixelValidityF32Configuration {
        extent: 7,
        input_schema: "test.pixel/1".into(),
    })?;
    let mut validity = vec![false; 7];
    PixelValidityF32::process(
        &mut pixel,
        &[
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
        &mut validity,
    )?;
    record(
        results,
        "pixel-validity-f32",
        "finite-and-nonfinite".into(),
        validity.iter().map(|&x| x as u8 as f64).collect(),
    );
    let c = &f.statistics;
    let n = c.frames[0].len();
    for mode in ["finite", "sliding", "exponential"] {
        let label = format!("array-statistics-{mode}-f32");
        let mut p = ArrayStatisticsF32::prepare(ArrayStatisticsF32Configuration {
            size: c.size.clone(),
            mode: mode.into(),
            window_samples: c.window_samples,
            time_constant_s: c.time_constant_s,
            sample_period_s: c.sample_period_s,
            max_state_bytes: 268435456,
            input_schema: "test.sample/1".into(),
            mean_schema: "test.mean/1".into(),
            variance_schema: "test.variance/1".into(),
            stddev_schema: "test.stddev/1".into(),
        })?;
        let (mut mean, mut var, mut stddev) = (
            vec![f.sentinel; n],
            vec![f.sentinel; n],
            vec![f.sentinel; n],
        );
        let (mut count, mut complete, mut valid) = ([0], [false], [false]);
        macro_rules! step {
            ($sample:expr) => {
                ArrayStatisticsF32::process(
                    &mut p,
                    $sample,
                    &mut mean,
                    &mut var,
                    &mut stddev,
                    &mut count,
                    &mut complete,
                    &mut valid,
                )
            };
        }
        macro_rules! output {
            ($case:expr) => {{
                let mut v = values(&mean);
                v.extend(values(&var));
                v.extend(values(&stddev));
                v.extend([
                    count[0] as f64,
                    complete[0] as u8 as f64,
                    valid[0] as u8 as f64,
                ]);
                record(results, &label, $case, v);
            }};
        }
        for (i, frame) in c.frames.iter().enumerate() {
            step!(frame)?;
            output!(format!("frame-{i}"));
        }
        let invalid = vec![f32::NAN; n];
        let previous = (
            mean.clone(),
            var.clone(),
            stddev.clone(),
            count,
            complete,
            valid,
        );
        let rejected = step!(&invalid).is_err();
        record(
            results,
            &label,
            "nonfinite-status".into(),
            vec![
                rejected as u8 as f64,
                (previous
                    == (
                        mean.clone(),
                        var.clone(),
                        stddev.clone(),
                        count,
                        complete,
                        valid,
                    )) as u8 as f64,
            ],
        );
        p.reset();
        step!(&c.frames[0])?;
        output!("reset".into());
        mean.fill(f.sentinel);
        var.fill(f.sentinel);
        stddev.fill(f.sentinel);
        let previous = (count, complete, valid);
        let overflow = vec![f32::MAX; n];
        let rejected = step!(&overflow).is_err();
        record(
            results,
            &label,
            "overflow-status".into(),
            vec![
                rejected as u8 as f64,
                (mean
                    .iter()
                    .chain(&var)
                    .chain(&stddev)
                    .all(|&x| x == f.sentinel)
                    && previous == (count, complete, valid)) as u8 as f64,
            ],
        );
        step!(&c.frames[1])?;
        output!("after-overflow".into());
    }
    let c = &f.gated;
    let n = c.frames[0].len();
    let mut p = ValidityGatedClosedLoopIntegratorF32::prepare(
        ValidityGatedClosedLoopIntegratorF32Configuration {
            state_length: n,
            validity_length: c.validity[0].len(),
            validity_schema: "test.validity/1".into(),
        },
    )?;
    p.replace_properties(&[
        PropertyAssignment {
            id: 0,
            value: PropertyValue::Float(c.gain),
        },
        PropertyAssignment {
            id: 1,
            value: PropertyValue::Float(c.pole),
        },
        PropertyAssignment {
            id: 2,
            value: PropertyValue::Float(c.anti_windup_gain),
        },
    ])?;
    let (mut y, mut state) = (vec![f.sentinel; n], vec![f.sentinel; n]);
    for i in 0..c.frames.len() {
        let invalid = vec![f32::NAN; n];
        let held = c.validity[i].iter().any(|v| !*v);
        let x = if held { &invalid } else { &c.frames[i] };
        let feedback = if held { &invalid } else { &c.feedback[i] };
        ValidityGatedClosedLoopIntegratorF32::process(
            &mut p,
            x,
            feedback,
            &c.validity[i],
            &mut y,
            &mut state,
        )?;
        record(
            results,
            "validity-gated-closed-loop-integrator-f32",
            format!("frame-{i}-output"),
            values(&y),
        );
        record(
            results,
            "validity-gated-closed-loop-integrator-f32",
            format!("frame-{i}-state"),
            values(&state),
        );
    }
    let old = (y.clone(), state.clone());
    let rejected = ValidityGatedClosedLoopIntegratorF32::process(
        &mut p,
        &vec![f32::NAN; n],
        &c.feedback[1],
        &vec![true; c.validity[0].len()],
        &mut y,
        &mut state,
    )
    .is_err();
    record(
        results,
        "validity-gated-closed-loop-integrator-f32",
        "nonfinite-status".into(),
        vec![
            rejected as u8 as f64,
            (old == (y.clone(), state.clone())) as u8 as f64,
        ],
    );
    p.reset();
    ValidityGatedClosedLoopIntegratorF32::process(
        &mut p,
        &c.frames[1],
        &vec![f32::NAN; n],
        &vec![true; c.validity[0].len()],
        &mut y,
        &mut state,
    )?;
    record(
        results,
        "validity-gated-closed-loop-integrator-f32",
        "reset-output".into(),
        values(&y),
    );
    record(
        results,
        "validity-gated-closed-loop-integrator-f32",
        "reset-state".into(),
        values(&state),
    );
    Ok(())
}
