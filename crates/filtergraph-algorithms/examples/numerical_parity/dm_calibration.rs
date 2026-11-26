//! Shared logical coordinates for real DM/calibration declarations.
use super::{Record, record};
use filtergraph_algorithms::*;
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    dm: Dm,
    matched: Matched,
    response: Response,
    stability: Stability,
}
#[derive(Deserialize)]
struct Dm {
    controller_command: Vec<f32>,
    vdm_command: Vec<f32>,
    pdm_command: Vec<f32>,
    controller_to_vdm: Vec<f32>,
    active_to_full: Vec<f32>,
    vdm_to_pdm: Vec<f32>,
    full_to_active: Vec<f32>,
    pdm_to_vdm: Vec<f32>,
    vdm_to_controller: Vec<f32>,
    system_flat: Vec<f32>,
    max_command_power: f32,
    edges: Vec<[usize; 2]>,
    max_differences: Vec<f32>,
    preferred_command: Vec<f32>,
    command_to_null_modes: Vec<f32>,
    null_modes_to_command: Vec<f32>,
}
#[derive(Deserialize)]
struct Matched {
    measurements: Vec<Vec<f32>>,
    corrector_sample: Vec<Vec<f32>>,
}
#[derive(Deserialize)]
struct Response {
    push_fringed: Vec<f32>,
    push_unfringed: Vec<f32>,
    pull_fringed: Vec<f32>,
    pull_unfringed: Vec<f32>,
}
#[derive(Deserialize)]
struct Stability {
    measurement_count: usize,
    corrector_coordinate_count: usize,
    first_window: Vec<f32>,
    second_window: Vec<f32>,
    minimum_column_norm: f32,
}
fn values(r: &mut Vec<Record>, label: &str, case: String, out: &[f32]) {
    record(
        r,
        label,
        case,
        out.iter()
            .map(|&x| if x.is_finite() { x as f64 } else { 0. })
            .collect(),
    );
}
fn publication(r: &mut Vec<Record>, label: &str, case: &str, out: &[f32], code: u8, bounds: bool) {
    // A rejected dense BLAS call has no valid numerical payload. Preserve its
    // rejection, mutation, nonfinite witness and bounds contract across backends.
    if matches!(
        label,
        "controller-to-vdm-f32"
            | "vdm-to-pdm-f32"
            | "pdm-feedback-to-vdm-f32"
            | "vdm-feedback-to-controller-f32"
    ) {
        if code != 2 {
            values(r, label, format!("{case}-output"), out);
        }
        record(
            r,
            label,
            format!("{case}-status"),
            vec![
                code as f64,
                out.iter().all(|&x| x == 71.) as u8 as f64,
                out.iter().any(|x| !x.is_finite()) as u8 as f64,
                bounds as u8 as f64,
            ],
        );
        return;
    }
    assert!(bounds, "caller output guard was modified");
    values(r, label, format!("{case}-output"), out);
    let mut status = vec![code as f64, out.iter().all(|&x| x == 71.) as u8 as f64];
    status.extend(out.iter().map(|x| {
        if x.is_finite() {
            0.
        } else if x.is_nan() {
            1.
        } else if x.is_sign_positive() {
            2.
        } else {
            3.
        }
    }));
    record(r, label, format!("{case}-status"), status);
}
macro_rules! one {
    ($d:ident,$e:ident,$p:ident,$r:ident,$label:expr,$case:expr,$input:expr,$n:expr) => {{
        let mut storage = vec![71.; $n + 2];
        storage[0] = 89.;
        storage[$n + 1] = 97.;
        let code = match $d::process(&mut $p, $input, &mut storage[1..$n + 1]) {
            Ok(_) => 0,
            Err(DeclarationProcessError::Algorithm($e::NonFiniteInput)) => 1,
            Err(DeclarationProcessError::Algorithm($e::NonFiniteOutput)) => 2,
            Err(e) => return Err(e.into()),
        };
        publication(
            $r,
            $label,
            &$case,
            &storage[1..$n + 1],
            code,
            storage[0] == 89. && storage[$n + 1] == 97.,
        );
    }};
}
macro_rules! single {
    ($d:ident,$e:ident,$p:ident,$r:ident,$label:expr,$case:expr,$input:expr,$n:expr) => {{
        let mut out = vec![71.; $n];
        let code = match $d::process(&mut $p, $input, &mut out) {
            Ok(_) => 0,
            Err(DeclarationProcessError::Algorithm($e::NonFiniteInput)) => 1,
            Err(e) => return Err(e.into()),
        };
        publication($r, $label, &$case, &out, code, true);
    }};
}
macro_rules! vdm {
    ($d:ident,$e:ident,$p:ident,$r:ident,$label:expr,$case:expr,$input:expr,$n:expr) => {{
        let mut storage = vec![71.; $n + 2];
        storage[0] = 89.;
        storage[$n + 1] = 97.;
        let mut roles = vec![99; $n];
        let code = match $d::process(&mut $p, $input, &mut storage[1..$n + 1], &mut roles) {
            Ok(_) => 0,
            Err(DeclarationProcessError::Algorithm($e::NonFiniteInput)) => 1,
            Err(DeclarationProcessError::Algorithm($e::NonFiniteOutput)) => 2,
            Err(e) => return Err(e.into()),
        };
        let case = $case;
        publication(
            $r,
            $label,
            &case,
            &storage[1..$n + 1],
            code,
            storage[0] == 89. && storage[$n + 1] == 97.,
        );
        record(
            $r,
            $label,
            format!("{case}-roles-status"),
            roles.iter().map(|&x| x as f64).collect(),
        );
    }};
}
macro_rules! variants {
    ($mac:ident,$d:ident,$e:ident,$p:ident,$r:ident,$label:expr,$input:expr,$n:expr) => {{
        $mac!($d, $e, $p, $r, $label, "initial", $input, $n);
        for (name, value) in [
            ("nan", f32::NAN),
            ("positive-inf", f32::INFINITY),
            ("negative-inf", f32::NEG_INFINITY),
        ] {
            let mut input = $input.to_vec();
            input[1] = value;
            $mac!($d, $e, $p, $r, $label, name, &input, $n);
        }
        $mac!($d, $e, $p, $r, $label, "recovery", $input, $n);
        $mac!(
            $d,
            $e,
            $p,
            $r,
            $label,
            "large",
            &vec![f32::MAX; $input.len()],
            $n
        );
        $mac!($d, $e, $p, $r, $label, "after-large", $input, $n);
    }};
}
macro_rules! replacements {
    ($mac:ident,$d:ident,$e:ident,$p:ident,$r:ident,$label:expr,$input:expr,$n:expr,$parameters:expr) => {{
        for &(port, name, values) in $parameters {
            let replacement: Vec<_> = values.iter().map(|v| v * 0.75).collect();
            $p.replace_parameter(port, AlgorithmInput::from(replacement.as_slice()))
                .unwrap()?;
            $mac!(
                $d,
                $e,
                $p,
                $r,
                $label,
                format!("replace-{name}"),
                $input,
                $n
            );
            for (kind, bad) in [
                ("nan", vec![f32::NAN; replacement.len()]),
                ("extent", vec![0.; replacement.len() - 1]),
            ] {
                let rejected = $p
                    .replace_parameter(port, AlgorithmInput::from(bad.as_slice()))
                    .unwrap()
                    .is_err();
                record(
                    $r,
                    $label,
                    format!("invalid-{name}-{kind}-status"),
                    vec![rejected as u8 as f64],
                );
            }
            $mac!(
                $d,
                $e,
                $p,
                $r,
                $label,
                format!("after-invalid-{name}"),
                $input,
                $n
            );
        }
    }};
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let c = &f.dm;

    {
        let label = "controller-to-vdm-f32";
        let mut p = ControllerToVdmF32::prepare(ControllerToVdmF32Configuration {
            controller_command_count: 2,
            active_vdm_count: 3,
        })?;
        let parameters: &[(usize, &str, &[f32])] =
            &[(2, "controller_to_vdm", c.controller_to_vdm.as_slice())];
        one!(
            ControllerToVdmF32,
            ControllerToVdmError,
            p,
            results,
            label,
            "default",
            &c.controller_command,
            3
        );
        for &(port, _, values) in parameters {
            p.replace_parameter(port, AlgorithmInput::from(values))
                .unwrap()?;
        }
        variants!(
            one,
            ControllerToVdmF32,
            ControllerToVdmError,
            p,
            results,
            label,
            &c.controller_command,
            3
        );
        replacements!(
            one,
            ControllerToVdmF32,
            ControllerToVdmError,
            p,
            results,
            label,
            &c.controller_command,
            3,
            parameters
        );
        ControllerToVdmF32::reset(&mut p);
        one!(
            ControllerToVdmF32,
            ControllerToVdmError,
            p,
            results,
            label,
            "reset",
            &c.controller_command,
            3
        );
    }

    {
        let label = "vdm-to-pdm-f32";
        let mut p = VdmToPdmF32::prepare(VdmToPdmF32Configuration {
            active_vdm_count: 3,
            full_vdm_count: 4,
            physical_actuator_count: 4,
        })?;
        let parameters: &[(usize, &str, &[f32])] = &[
            (3, "active_to_full", c.active_to_full.as_slice()),
            (4, "vdm_to_pdm", c.vdm_to_pdm.as_slice()),
        ];
        vdm!(
            VdmToPdmF32,
            VdmToPdmError,
            p,
            results,
            label,
            "default",
            &c.vdm_command,
            4
        );
        for &(port, _, values) in parameters {
            p.replace_parameter(port, AlgorithmInput::from(values))
                .unwrap()?;
        }
        variants!(
            vdm,
            VdmToPdmF32,
            VdmToPdmError,
            p,
            results,
            label,
            &c.vdm_command,
            4
        );
        replacements!(
            vdm,
            VdmToPdmF32,
            VdmToPdmError,
            p,
            results,
            label,
            &c.vdm_command,
            4,
            parameters
        );
        VdmToPdmF32::reset(&mut p);
        vdm!(
            VdmToPdmF32,
            VdmToPdmError,
            p,
            results,
            label,
            "reset",
            &c.vdm_command,
            4
        );
    }

    {
        let label = "pdm-system-flat-f32";
        let mut p = PdmSystemFlatF32::prepare(PdmSystemFlatF32Configuration { actuator_count: 4 })?;
        let parameters: &[(usize, &str, &[f32])] = &[(2, "system_flat", c.system_flat.as_slice())];
        one!(
            PdmSystemFlatF32,
            PdmSystemFlatError,
            p,
            results,
            label,
            "default",
            &c.pdm_command,
            4
        );
        for &(port, _, values) in parameters {
            p.replace_parameter(port, AlgorithmInput::from(values))
                .unwrap()?;
        }
        variants!(
            one,
            PdmSystemFlatF32,
            PdmSystemFlatError,
            p,
            results,
            label,
            &c.pdm_command,
            4
        );
        p.replace_parameter(2, AlgorithmInput::from(&[f32::MAX; 4][..]))
            .unwrap()?;
        one!(
            PdmSystemFlatF32,
            PdmSystemFlatError,
            p,
            results,
            label,
            "addition-overflow",
            &[f32::MAX; 4],
            4
        );
        one!(
            PdmSystemFlatF32,
            PdmSystemFlatError,
            p,
            results,
            label,
            "addition-recovery",
            &[-f32::MAX; 4],
            4
        );
        p.replace_parameter(2, AlgorithmInput::from(c.system_flat.as_slice()))
            .unwrap()?;
        replacements!(
            one,
            PdmSystemFlatF32,
            PdmSystemFlatError,
            p,
            results,
            label,
            &c.pdm_command,
            4,
            parameters
        );
        PdmSystemFlatF32::reset(&mut p);
        one!(
            PdmSystemFlatF32,
            PdmSystemFlatError,
            p,
            results,
            label,
            "reset",
            &c.pdm_command,
            4
        );
    }

    {
        let label = "pdm-command-power-limit-f32";
        let mut p = PdmCommandPowerLimitF32::prepare(PdmCommandPowerLimitF32Configuration {
            actuator_count: 4,
            max_command_power: c.max_command_power,
        })?;
        let parameters: &[(usize, &str, &[f32])] = &[];
        single!(
            PdmCommandPowerLimitF32,
            PdmCommandPowerLimitError,
            p,
            results,
            label,
            "default",
            &c.pdm_command,
            4
        );
        for &(port, _, values) in parameters {
            p.replace_parameter(port, AlgorithmInput::from(values))
                .unwrap()?;
        }
        variants!(
            single,
            PdmCommandPowerLimitF32,
            PdmCommandPowerLimitError,
            p,
            results,
            label,
            &c.pdm_command,
            4
        );
        replacements!(
            single,
            PdmCommandPowerLimitF32,
            PdmCommandPowerLimitError,
            p,
            results,
            label,
            &c.pdm_command,
            4,
            parameters
        );
        PdmCommandPowerLimitF32::reset(&mut p);
        single!(
            PdmCommandPowerLimitF32,
            PdmCommandPowerLimitError,
            p,
            results,
            label,
            "reset",
            &c.pdm_command,
            4
        );
    }

    {
        let label = "inter-actuator-stroke-limit-f32";
        let mut p =
            InterActuatorStrokeLimitF32::prepare(InterActuatorStrokeLimitF32Configuration {
                actuator_count: 4,
                edges: c.edges.clone(),
                max_differences: c.max_differences.clone(),
            })?;
        let parameters: &[(usize, &str, &[f32])] =
            &[(2, "max_differences", c.max_differences.as_slice())];
        single!(
            InterActuatorStrokeLimitF32,
            InterActuatorStrokeLimitError,
            p,
            results,
            label,
            "default",
            &c.pdm_command,
            4
        );
        for &(port, _, values) in parameters {
            p.replace_parameter(port, AlgorithmInput::from(values))
                .unwrap()?;
        }
        variants!(
            single,
            InterActuatorStrokeLimitF32,
            InterActuatorStrokeLimitError,
            p,
            results,
            label,
            &c.pdm_command,
            4
        );
        single!(
            InterActuatorStrokeLimitF32,
            InterActuatorStrokeLimitError,
            p,
            results,
            label,
            "opposite-large",
            &[f32::MAX, -f32::MAX, f32::MAX, -f32::MAX],
            4
        );
        replacements!(
            single,
            InterActuatorStrokeLimitF32,
            InterActuatorStrokeLimitError,
            p,
            results,
            label,
            &c.pdm_command,
            4,
            parameters
        );
        p.replace_parameter(2, AlgorithmInput::from(&[0_f32; 3][..]))
            .unwrap()?;
        single!(
            InterActuatorStrokeLimitF32,
            InterActuatorStrokeLimitError,
            p,
            results,
            label,
            "zero-limits",
            &c.pdm_command,
            4
        );
        InterActuatorStrokeLimitF32::reset(&mut p);
        single!(
            InterActuatorStrokeLimitF32,
            InterActuatorStrokeLimitError,
            p,
            results,
            label,
            "reset",
            &c.pdm_command,
            4
        );
    }

    {
        let label = "pdm-stroke-optimization-f32";
        let mut p = PdmStrokeOptimizationF32::prepare(PdmStrokeOptimizationF32Configuration {
            null_mode_count: 2,
            preferred_command: c.preferred_command.clone(),
            command_to_null_modes: c.command_to_null_modes.clone(),
            null_modes_to_command: c.null_modes_to_command.clone(),
            actuator_count: 4,
        })?;
        let parameters: &[(usize, &str, &[f32])] = &[
            (2, "preferred_command", c.preferred_command.as_slice()),
            (
                3,
                "command_to_null_modes",
                c.command_to_null_modes.as_slice(),
            ),
            (
                4,
                "null_modes_to_command",
                c.null_modes_to_command.as_slice(),
            ),
        ];
        one!(
            PdmStrokeOptimizationF32,
            PdmStrokeOptimizationError,
            p,
            results,
            label,
            "default",
            &c.pdm_command,
            4
        );
        for &(port, _, values) in parameters {
            p.replace_parameter(port, AlgorithmInput::from(values))
                .unwrap()?;
        }
        variants!(
            one,
            PdmStrokeOptimizationF32,
            PdmStrokeOptimizationError,
            p,
            results,
            label,
            &c.pdm_command,
            4
        );
        replacements!(
            one,
            PdmStrokeOptimizationF32,
            PdmStrokeOptimizationError,
            p,
            results,
            label,
            &c.pdm_command,
            4,
            parameters
        );
        PdmStrokeOptimizationF32::reset(&mut p);
        one!(
            PdmStrokeOptimizationF32,
            PdmStrokeOptimizationError,
            p,
            results,
            label,
            "reset",
            &c.pdm_command,
            4
        );
    }

    {
        let label = "pdm-feedback-to-vdm-f32";
        let mut p = PdmFeedbackToVdmF32::prepare(PdmFeedbackToVdmF32Configuration {
            active_vdm_count: 3,
            full_vdm_count: 4,
            physical_actuator_count: 4,
        })?;
        let parameters: &[(usize, &str, &[f32])] = &[
            (2, "full_to_active", c.full_to_active.as_slice()),
            (3, "pdm_to_vdm", c.pdm_to_vdm.as_slice()),
        ];
        one!(
            PdmFeedbackToVdmF32,
            PdmFeedbackToVdmError,
            p,
            results,
            label,
            "default",
            &c.pdm_command,
            3
        );
        for &(port, _, values) in parameters {
            p.replace_parameter(port, AlgorithmInput::from(values))
                .unwrap()?;
        }
        variants!(
            one,
            PdmFeedbackToVdmF32,
            PdmFeedbackToVdmError,
            p,
            results,
            label,
            &c.pdm_command,
            3
        );
        replacements!(
            one,
            PdmFeedbackToVdmF32,
            PdmFeedbackToVdmError,
            p,
            results,
            label,
            &c.pdm_command,
            3,
            parameters
        );
        PdmFeedbackToVdmF32::reset(&mut p);
        one!(
            PdmFeedbackToVdmF32,
            PdmFeedbackToVdmError,
            p,
            results,
            label,
            "reset",
            &c.pdm_command,
            3
        );
    }

    {
        let label = "vdm-feedback-to-controller-f32";
        let mut p = VdmFeedbackToControllerF32::prepare(VdmFeedbackToControllerF32Configuration {
            controller_command_count: 2,
            active_vdm_count: 3,
        })?;
        let parameters: &[(usize, &str, &[f32])] =
            &[(2, "vdm_to_controller", c.vdm_to_controller.as_slice())];
        one!(
            VdmFeedbackToControllerF32,
            VdmFeedbackToControllerError,
            p,
            results,
            label,
            "default",
            &c.vdm_command,
            2
        );
        for &(port, _, values) in parameters {
            p.replace_parameter(port, AlgorithmInput::from(values))
                .unwrap()?;
        }
        variants!(
            one,
            VdmFeedbackToControllerF32,
            VdmFeedbackToControllerError,
            p,
            results,
            label,
            &c.vdm_command,
            2
        );
        replacements!(
            one,
            VdmFeedbackToControllerF32,
            VdmFeedbackToControllerError,
            p,
            results,
            label,
            &c.vdm_command,
            2,
            parameters
        );
        VdmFeedbackToControllerF32::reset(&mut p);
        one!(
            VdmFeedbackToControllerF32,
            VdmFeedbackToControllerError,
            p,
            results,
            label,
            "reset",
            &c.vdm_command,
            2
        );
    }
    calibration(&f, results)?;
    Ok(())
}
fn calibration(f: &Fixture, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let c = &f.matched;
    let label = "matched-calibration-differencing-f32";
    let m = c.measurements[0].len();
    let n = c.corrector_sample[0].len();
    let mut p = MatchedCalibrationDifferencingF32::prepare(
        MatchedCalibrationDifferencingF32Configuration {
            measurement_count: m,
            measurements_schema: "org.example.measurements/1".into(),
            corrector_sample_schema: "org.example.corrector/1".into(),
            filtered_measurements_schema: "org.example.filtered-measurements/1".into(),
            filtered_corrector_sample_schema: "org.example.filtered-corrector/1".into(),
            corrector_coordinate_count: n,
        },
    )?;
    macro_rules! matched {
        ($name:expr,$m:expr,$c:expr) => {{
            let mut a = vec![71.; m];
            let mut b = vec![71.; n];
            let mut filtered = [true];
            let code = match MatchedCalibrationDifferencingF32::process(
                &mut p,
                $m,
                $c,
                &mut a,
                &mut b,
                &mut filtered,
            ) {
                Ok(_) => 0,
                Err(DeclarationProcessError::Algorithm(
                    MatchedCalibrationDifferencingError::NonFiniteInput {
                        input: MatchedCalibrationDifferencingInputKind::Measurements,
                        ..
                    },
                )) => 1,
                Err(DeclarationProcessError::Algorithm(
                    MatchedCalibrationDifferencingError::NonFiniteInput {
                        input: MatchedCalibrationDifferencingInputKind::CorrectorSample,
                        ..
                    },
                )) => 2,
                Err(DeclarationProcessError::Algorithm(
                    MatchedCalibrationDifferencingError::NonFiniteCalculation,
                )) => 3,
                Err(e) => return Err(e.into()),
            };
            let name = $name;
            values(results, label, format!("{name}-measurements"), &a);
            values(results, label, format!("{name}-corrector"), &b);
            record(
                results,
                label,
                format!("{name}-status"),
                vec![
                    code as f64,
                    filtered[0] as u8 as f64,
                    p.workspace().is_primed() as u8 as f64,
                    (a.iter().chain(&b).all(|&x| x == 71.)) as u8 as f64,
                ],
            );
        }};
    }
    matched!("prime", &c.measurements[0], &c.corrector_sample[0]);
    for (name, value) in [
        ("nan", f32::NAN),
        ("positive-inf", f32::INFINITY),
        ("negative-inf", f32::NEG_INFINITY),
    ] {
        let mut a = c.measurements[1].clone();
        a[1] = value;
        matched!(format!("measurement-{name}"), &a, &c.corrector_sample[1]);
        let mut b = c.corrector_sample[1].clone();
        b[0] = value;
        matched!(format!("corrector-{name}"), &c.measurements[1], &b);
    }
    for i in 1..c.measurements.len() {
        matched!(
            format!("frame-{i}"),
            &c.measurements[i],
            &c.corrector_sample[i]
        );
    }
    MatchedCalibrationDifferencingF32::reset(&mut p);
    matched!("reset", &c.measurements[0], &c.corrector_sample[0]);
    MatchedCalibrationDifferencingF32::reset(&mut p);
    matched!("overflow-prime", &vec![f32::MAX; m], &c.corrector_sample[0]);
    matched!("overflow", &vec![-f32::MAX; m], &c.corrector_sample[1]);
    matched!(
        "overflow-recovery",
        &vec![f32::MAX; m],
        &c.corrector_sample[1]
    );
    let c = &f.response;
    let label = "scc-push-pull-response-f32";
    let n = c.push_fringed.len();
    let mut p =
        SccPushPullResponseF32::prepare(SccPushPullResponseF32Configuration { pixel_count: n })?;
    macro_rules! response {
        ($name:expr,$a:expr,$b:expr,$c:expr,$d:expr) => {{
            let mut out = vec![71.; n];
            let code = match SccPushPullResponseF32::process(&mut p, $a, $b, $c, $d, &mut out) {
                Ok(_) => 0,
                Err(DeclarationProcessError::Algorithm(
                    SccPushPullResponseError::NonFiniteImage { image, .. },
                )) => match image {
                    SccPushPullImageKind::PushFringed => 1,
                    SccPushPullImageKind::PushUnfringed => 2,
                    SccPushPullImageKind::PullFringed => 3,
                    SccPushPullImageKind::PullUnfringed => 4,
                },
                Err(DeclarationProcessError::Algorithm(
                    SccPushPullResponseError::NonFiniteCalculation { .. },
                )) => 5,
                Err(e) => return Err(e.into()),
            };
            publication(results, label, &$name, &out, code, true);
        }};
    }
    response!(
        "initial",
        &c.push_fringed,
        &c.push_unfringed,
        &c.pull_fringed,
        &c.pull_unfringed
    );
    for i in 0..4 {
        let mut frames = [
            c.push_fringed.clone(),
            c.push_unfringed.clone(),
            c.pull_fringed.clone(),
            c.pull_unfringed.clone(),
        ];
        frames[i][1] = f32::NAN;
        response!(
            format!("nonfinite-{i}"),
            &frames[0],
            &frames[1],
            &frames[2],
            &frames[3]
        );
    }
    response!(
        "overflow",
        &vec![f32::MAX; n],
        &vec![-f32::MAX; n],
        &c.pull_fringed,
        &c.pull_unfringed
    );
    response!(
        "recovery",
        &c.push_fringed,
        &c.push_unfringed,
        &c.pull_fringed,
        &c.pull_unfringed
    );
    SccPushPullResponseF32::reset(&mut p);
    response!(
        "reset",
        &c.push_fringed,
        &c.push_unfringed,
        &c.pull_fringed,
        &c.pull_unfringed
    );
    let c = &f.stability;
    let label = "interaction-matrix-stability-f32";
    let mut p =
        InteractionMatrixStabilityF32::prepare(InteractionMatrixStabilityF32Configuration {
            measurement_count: c.measurement_count,
            corrector_coordinate_count: c.corrector_coordinate_count,
            minimum_column_norm: c.minimum_column_norm,
            interaction_matrix_schema: "org.example.interaction/1".into(),
        })?;
    macro_rules! stability {
        ($name:expr,$a:expr,$b:expr) => {{
            let mut status = [99];
            let mut coord = [99];
            let mut norms = [71.; 2];
            let mut column = [71.];
            let mut maxcoord = [99];
            let mut frob = [71.];
            let code = match InteractionMatrixStabilityF32::process(
                &mut p,
                $a,
                $b,
                &mut status,
                &mut coord,
                &mut norms,
                &mut column,
                &mut maxcoord,
                &mut frob,
            ) {
                Ok(_) => 0,
                Err(DeclarationProcessError::Algorithm(
                    InteractionMatrixStabilityError::NonFiniteInput {
                        input: InteractionMatrixStabilityInputKind::FirstWindow,
                        ..
                    },
                )) => 1,
                Err(DeclarationProcessError::Algorithm(
                    InteractionMatrixStabilityError::NonFiniteInput {
                        input: InteractionMatrixStabilityInputKind::SecondWindow,
                        ..
                    },
                )) => 2,
                Err(DeclarationProcessError::Algorithm(
                    InteractionMatrixStabilityError::NonFiniteCalculation,
                )) => 3,
                Err(e) => return Err(e.into()),
            };
            let name = $name;
            values(results, label, format!("{name}-norms"), &norms);
            values(results, label, format!("{name}-relative-column"), &column);
            values(results, label, format!("{name}-relative-frobenius"), &frob);
            record(
                results,
                label,
                format!("{name}-status"),
                vec![
                    code as f64,
                    status[0] as f64,
                    coord[0] as f64,
                    maxcoord[0] as f64,
                ],
            );
        }};
    }
    stability!("initial", &c.first_window, &c.second_window);
    stability!("identical", &c.first_window, &c.first_window);
    stability!(
        "zero-column",
        &vec![0.; c.first_window.len()],
        &c.second_window
    );
    stability!(
        "opposite",
        &c.first_window,
        &c.first_window.iter().map(|v| -v).collect::<Vec<_>>()
    );
    for (name, value) in [("nan", f32::NAN), ("infinite", f32::INFINITY)] {
        let mut a = c.first_window.clone();
        a[1] = value;
        stability!(format!("first-{name}"), &a, &c.second_window);
        let mut b = c.second_window.clone();
        b[0] = value;
        stability!(format!("second-{name}"), &c.first_window, &b);
    }
    p.replace_properties(&[
        PropertyAssignment {
            id: 0,
            value: PropertyValue::Float(0.),
        },
        PropertyAssignment {
            id: 1,
            value: PropertyValue::Float(2.),
        },
    ])?;
    stability!("column-policy", &c.first_window, &c.second_window);
    p.replace_properties(&[
        PropertyAssignment {
            id: 0,
            value: PropertyValue::Float(2.),
        },
        PropertyAssignment {
            id: 1,
            value: PropertyValue::Float(0.),
        },
    ])?;
    stability!("frobenius-policy", &c.first_window, &c.second_window);
    record(
        results,
        label,
        "invalid-property-status".into(),
        vec![
            p.replace_properties(&[PropertyAssignment {
                id: 0,
                value: PropertyValue::Float(2.25),
            }])
            .is_err() as u8 as f64,
        ],
    );
    stability!("after-invalid", &c.first_window, &c.second_window);
    InteractionMatrixStabilityF32::reset(&mut p);
    stability!("reset", &c.first_window, &c.second_window);
    Ok(())
}
