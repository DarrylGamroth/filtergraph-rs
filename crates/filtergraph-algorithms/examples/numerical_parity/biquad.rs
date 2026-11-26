//! Shared-source CPU cascade comparisons in both arithmetic precisions.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, AlgorithmPlan, BiquadCascadeError, BiquadCascadeF32,
    BiquadCascadeF32Configuration, BiquadCascadePlan, BiquadCascadeWorkspace,
    DeclarationProcessError,
};
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    cascades: Vec<Cascade>,
    regions: Regions,
    replacement: Replacement,
    fma: Vec<Fma>,
}
#[derive(Deserialize)]
struct Cascade {
    name: String,
    extent: usize,
    section_count: usize,
    set_count: usize,
    coefficients: Vec<f64>,
    frames: Vec<Vec<f64>>,
    reset_before_frame: usize,
}
#[derive(Deserialize)]
struct Fma {
    name: String,
    numerator_factors: [f64; 3],
    denominator: [f64; 2],
    frames: Vec<Vec<f64>>,
    reject_input: Vec<f64>,
    recovery_input: Vec<f64>,
}
#[derive(Deserialize)]
struct Regions {
    extent: usize,
    ranges: Vec<[usize; 2]>,
    coefficients: Vec<Vec<[f64; 5]>>,
    frames: Vec<Vec<f64>>,
}
#[derive(Deserialize)]
struct Replacement {
    extent: usize,
    section_count: usize,
    set_count: usize,
    initial_coefficients: Vec<f32>,
    coefficients: Vec<f32>,
    frames: Vec<Vec<f32>>,
    replace_before_frame: usize,
}
fn flag(v: bool) -> f64 {
    if v { 1.0 } else { 0.0 }
}
fn emit(
    results: &mut Vec<Record>,
    label: &str,
    case: &str,
    output: &[f64],
    state: &[f64],
    status: Vec<f64>,
) {
    record(
        results,
        label,
        format!("{case}-output"),
        output
            .iter()
            .map(|&v| if v.is_finite() { v } else { 0.0 })
            .collect(),
    );
    record(
        results,
        label,
        format!("{case}-state"),
        state
            .iter()
            .map(|&v| if v.is_finite() { v } else { 0.0 })
            .collect(),
    );
    let mut status = status;
    status.push(flag(output.iter().all(|v| v.is_finite())));
    status.push(flag(state.iter().all(|v| v.is_finite())));
    record(results, label, format!("{case}-status"), status);
}
fn reason(result: Result<(), BiquadCascadeError>) -> Result<f64, Box<dyn Error>> {
    match result {
        Ok(()) => Ok(0.0),
        Err(BiquadCascadeError::NonFiniteInput) => Ok(1.0),
        Err(BiquadCascadeError::NonFiniteOutput) => Ok(2.0),
        Err(e) => Err(e.into()),
    }
}
macro_rules! step {
    ($t:ty,$plan:expr,$workspace:expr,$input:expr,$label:expr,$case:expr,$results:expr) => {{
        let workspace: &mut BiquadCascadeWorkspace<$t> = $workspace;
        let before = workspace.state().to_vec();
        let mut output = vec![71.0 as $t; $plan.extent()];
        let reason = reason($plan.process(&mut output, workspace, $input))?;
        emit(
            $results,
            $label,
            $case,
            &output.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            &workspace
                .state()
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
            vec![
                reason,
                flag(reason == 0.0 || output.iter().all(|&v| v == 71.0 as $t)),
                flag(reason == 0.0 || workspace.state() == before),
            ],
        );
    }};
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    macro_rules! precision {
        ($t:ty,$label:literal) => {{
            for case in &f.cascades {
                let plan = BiquadCascadePlan::new(
                    case.coefficients
                        .iter()
                        .map(|&v| v as $t)
                        .collect::<Vec<_>>(),
                    case.extent,
                    case.section_count,
                    case.set_count,
                )?;
                let mut workspace = plan.workspace();
                for (i, frame) in case.frames.iter().enumerate() {
                    if i == case.reset_before_frame {
                        workspace.reset();
                        record(
                            results,
                            $label,
                            format!("{}-reset-state", case.name),
                            workspace.state().iter().map(|&v| v as f64).collect(),
                        );
                    }
                    let input = frame.iter().map(|&v| v as $t).collect::<Vec<_>>();
                    if i == 1 {
                        for (name, value) in [
                            ("nan", <$t>::NAN),
                            ("positive-inf", <$t>::INFINITY),
                            ("negative-inf", <$t>::NEG_INFINITY),
                        ] {
                            let mut invalid = input.clone();
                            invalid[case.extent - 1] = value;
                            step!(
                                $t,
                                plan,
                                &mut workspace,
                                &invalid,
                                $label,
                                &format!("{}-{name}", case.name),
                                results
                            );
                        }
                    }
                    step!(
                        $t,
                        plan,
                        &mut workspace,
                        &input,
                        $label,
                        &format!("{}-frame-{i}", case.name),
                        results
                    );
                }
            }
            for c in &f.fma {
                let case = format!("fma-{}", c.name);
                let coefficients = c
                    .numerator_factors
                    .iter()
                    .map(|&v| (v as $t) * <$t>::MAX)
                    .chain(c.denominator.iter().map(|&v| v as $t))
                    .collect::<Vec<_>>();
                let plan = BiquadCascadePlan::new(coefficients, 1, 1, 1)?;
                let mut workspace = plan.workspace();
                for (i, frame) in c.frames.iter().enumerate() {
                    let input = frame.iter().map(|&v| v as $t).collect::<Vec<_>>();
                    step!(
                        $t,
                        plan,
                        &mut workspace,
                        &input,
                        $label,
                        &format!("{case}-frame-{i}"),
                        results
                    );
                }
                let invalid = c.reject_input.iter().map(|&v| v as $t).collect::<Vec<_>>();
                step!(
                    $t,
                    plan,
                    &mut workspace,
                    &invalid,
                    $label,
                    &format!("{case}-overflow"),
                    results
                );
                let input = c
                    .recovery_input
                    .iter()
                    .map(|&v| v as $t)
                    .collect::<Vec<_>>();
                step!(
                    $t,
                    plan,
                    &mut workspace,
                    &input,
                    $label,
                    &format!("{case}-after-overflow"),
                    results
                );
                let first = c.frames[0].iter().map(|&v| v as $t).collect::<Vec<_>>();
                for phase in ["recover", "reset"] {
                    if phase == "recover" {
                        workspace.recover();
                    } else {
                        workspace.reset();
                    }
                    record(
                        results,
                        $label,
                        format!("{case}-{phase}-state"),
                        workspace.state().iter().map(|&v| v as f64).collect(),
                    );
                    step!(
                        $t,
                        plan,
                        &mut workspace,
                        &first,
                        $label,
                        &format!("{case}-after-{phase}"),
                        results
                    );
                }
            }
            let regions = f
                .regions
                .ranges
                .iter()
                .zip(&f.regions.coefficients)
                .map(|(r, c)| {
                    (
                        r[0]..r[1],
                        c.iter()
                            .map(|tuple| tuple.map(|v| v as $t))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            let plan = BiquadCascadePlan::from_regions(&regions, f.regions.extent)?;
            let mut workspace = plan.workspace();
            record(
                results,
                $label,
                "region-coefficients".into(),
                plan.coefficients().iter().map(|&v| v as f64).collect(),
            );
            for (i, frame) in f.regions.frames.iter().enumerate() {
                let input = frame.iter().map(|&v| v as $t).collect::<Vec<_>>();
                step!(
                    $t,
                    plan,
                    &mut workspace,
                    &input,
                    $label,
                    &format!("region-frame-{i}"),
                    results
                );
            }
            for (name, coefficients) in [
                (
                    "output-overflow",
                    vec![<$t>::MAX, 0.0 as $t, 0.0 as $t, 0.0 as $t, 0.0 as $t],
                ),
                (
                    "state-overflow",
                    vec![0.0 as $t, <$t>::MAX, 0.0 as $t, 0.0 as $t, 0.0 as $t],
                ),
            ] {
                let plan = BiquadCascadePlan::new(coefficients, 3, 1, 1)?;
                let mut workspace = plan.workspace();
                step!(
                    $t,
                    plan,
                    &mut workspace,
                    &[0.0 as $t, 0.0 as $t, 2.0 as $t],
                    $label,
                    name,
                    results
                );
                step!(
                    $t,
                    plan,
                    &mut workspace,
                    &[0.0 as $t; 3],
                    $label,
                    &format!("after-{name}"),
                    results
                );
            }
        }};
    }
    precision!(f32, "biquad-f32");
    precision!(f64, "biquad-f64");
    let c = f.replacement;
    let mut prepared = BiquadCascadeF32::prepare(BiquadCascadeF32Configuration {
        extent: c.extent,
        section_count: c.section_count,
        per_mode: c.set_count != 1,
        initial_coefficients: Some(c.initial_coefficients),
        input_schema: "org.example.parity.biquad-input/1".into(),
        output_schema: "org.example.parity.biquad-output/1".into(),
    })?;
    let mut row_major = vec![0.0; c.coefficients.len()];
    for set in 0..c.set_count {
        for section in 0..c.section_count {
            for coefficient in 0..5 {
                row_major[(coefficient * c.section_count + section) * c.set_count + set] =
                    c.coefficients[coefficient + 5 * (section + c.section_count * set)];
            }
        }
    }
    for (i, input) in c.frames.iter().enumerate() {
        if i == c.replace_before_frame {
            for name in ["nonfinite", "unstable"] {
                let before = prepared.workspace().state().to_vec();
                let previous = prepared.plan().coefficients().to_vec();
                let mut invalid = row_major.clone();
                if name == "nonfinite" {
                    invalid[0] = f32::NAN;
                } else {
                    invalid[4 * c.section_count * c.set_count] = 1.0;
                }
                let rejected = prepared
                    .replace_parameter(2, AlgorithmInput::from(invalid.as_slice()))
                    .ok_or("missing coefficients")?
                    .is_err();
                record(
                    results,
                    "biquad-replacement-f32",
                    format!("{name}-status"),
                    vec![
                        flag(!rejected),
                        flag(prepared.plan().coefficients() == previous),
                        flag(prepared.workspace().state() == before),
                    ],
                );
            }
            prepared
                .replace_parameter(2, AlgorithmInput::from(row_major.as_slice()))
                .ok_or("missing coefficients")??;
            record(
                results,
                "biquad-replacement-f32",
                "adopted-state".into(),
                prepared
                    .workspace()
                    .state()
                    .iter()
                    .copied()
                    .map(f64::from)
                    .collect(),
            );
        }
        let mut output = vec![71.0; c.extent];
        let outcome = BiquadCascadeF32::process(&mut prepared, input, &mut output);
        let result = match outcome {
            Ok(_) => Ok(()),
            Err(DeclarationProcessError::Algorithm(e)) => Err(e),
            Err(e) => return Err(e.into()),
        };
        let reason = reason(result)?;
        emit(
            results,
            "biquad-replacement-f32",
            &format!("frame-{i}"),
            &output.iter().copied().map(f64::from).collect::<Vec<_>>(),
            &prepared
                .workspace()
                .state()
                .iter()
                .copied()
                .map(f64::from)
                .collect::<Vec<_>>(),
            vec![reason, 1.0, 1.0],
        );
    }
    Ok(())
}
