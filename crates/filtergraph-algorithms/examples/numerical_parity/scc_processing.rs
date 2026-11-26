//! Real declaration parity for SCC spatial measurement, weighted PCG, and exposure correction.
use super::{Record, record};
use filtergraph_algorithms::*;
use serde::Deserialize;
use std::{error::Error, num::NonZeroU64};
#[derive(Deserialize)]
struct Fixture {
    spatial: Spatial,
    pcg: Pcg,
    correction: Correction,
}
#[derive(Deserialize)]
struct Spatial {
    image_size: (usize, usize),
    roi: Vec<bool>,
    reference: Vec<f32>,
    phase: Vec<f32>,
    fir: Vec<f32>,
    difference: Vec<f32>,
    minimum_support: f32,
    minimum_valid_fraction: f32,
}
#[derive(Deserialize)]
struct Pcg {
    measurements: Vec<f32>,
    weights: Vec<f32>,
    interaction: Vec<f32>,
    regularizer: Vec<f32>,
    regularization: f32,
    iterations: usize,
    tolerance: f32,
    minimum_diagonal: f32,
}
#[derive(Deserialize)]
struct Correction {
    frame_size: (usize, usize),
    reference: Vec<f32>,
    mask: Vec<bool>,
    reconstructor: Vec<f32>,
    fringed_response: Vec<f32>,
    unfringed_response: Vec<f32>,
    bias: Vec<f32>,
    gain: f32,
    pole: f32,
    events: Vec<Event>,
}
#[derive(Deserialize)]
struct Event {
    name: String,
    sequence: u64,
    kind: i8,
    frame: Vec<f32>,
    #[serde(default)]
    missing_metadata: bool,
    #[serde(default)]
    missing_identity: bool,
    #[serde(default)]
    discontinuity: bool,
    #[serde(default)]
    corrupted: bool,
    #[serde(default)]
    nonfinite: bool,
    #[serde(default)]
    reset: bool,
    gain: Option<f32>,
    pole: Option<f32>,
    duration: Option<u64>,
    domain: Option<u8>,
    generation: Option<u64>,
}
fn values(r: &mut Vec<Record>, label: &str, name: impl Into<String>, v: &[f32]) {
    record(r, label, name.into(), v.iter().map(|&x| x as f64).collect());
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let c = &f.spatial;
    let n = c.reference.len();
    let q = c.roi.iter().filter(|&&x| x).count() * 2;
    let label = "scc-spatial-carrier-demodulation-f32";
    let mut p =
        SccSpatialCarrierDemodulationF32::prepare(SccSpatialCarrierDemodulationF32Configuration {
            image_size: c.image_size,
            roi: c.roi.clone(),
            fir_tap_count: c.fir.len(),
            initial_reference_difference: Some(c.reference.clone()),
            initial_static_validity: None,
            initial_carrier_phase: Some(c.phase.clone()),
            initial_fir: Some(c.fir.clone()),
            minimum_support: c.minimum_support,
            minimum_valid_fraction: c.minimum_valid_fraction,
        })?;
    macro_rules! spatial {
        ($name:expr,$d:expr,$v:expr,$s:expr) => {{
            let mut a = vec![71.; q];
            let mut b = vec![73.; q];
            let mut v = vec![true; q];
            let mut fraction = [-1.];
            let code = match SccSpatialCarrierDemodulationF32::process(
                &mut p,
                $d,
                $v,
                &[$s],
                &mut a,
                &mut b,
                &mut v,
                &mut fraction,
            ) {
                Ok(_) => 0,
                Err(DeclarationProcessError::Algorithm(
                    SccSpatialCarrierDemodulationError::IncompletePair,
                )) => 1,
                Err(DeclarationProcessError::Algorithm(
                    SccSpatialCarrierDemodulationError::InvalidPairStatus,
                )) => 2,
                Err(DeclarationProcessError::Algorithm(
                    SccSpatialCarrierDemodulationError::InsufficientSupport,
                )) => 3,
                Err(e) => return Err(e.into()),
            };
            let name = $name;
            values(results, label, format!("{name}-measurements"), &a);
            values(results, label, format!("{name}-weights"), &b);
            values(results, label, format!("{name}-fraction"), &fraction);
            record(
                results,
                label,
                format!("{name}-validity-status"),
                v.iter().map(|&x| x as u8 as f64).collect(),
            );
            record(
                results,
                label,
                format!("{name}-status"),
                vec![
                    code as f64,
                    (a.iter().all(|&x| x == 71.)
                        && b.iter().all(|&x| x == 73.)
                        && v.iter().all(|&x| x)
                        && fraction == [-1.]) as u8 as f64,
                ],
            );
        }};
    }
    spatial!("initial", &c.difference, &vec![true; n], 1);
    spatial!("priming", &c.difference, &vec![true; n], 0);
    spatial!("undefined", &c.difference, &vec![true; n], 255);
    spatial!("no-support", &c.difference, &vec![false; n], 1);
    let mut d = c.difference.clone();
    d[2] = f32::NAN;
    spatial!("nonfinite", &d, &vec![true; n], 1);
    spatial!("recovery", &c.difference, &vec![true; n], 1);
    for (port, name, v) in [
        (
            7,
            "reference",
            c.reference.iter().map(|x| x * 0.5).collect::<Vec<_>>(),
        ),
        (9, "phase", c.phase.iter().map(|x| -x).collect()),
        (10, "fir", vec![2., 1., 2.]),
    ] {
        p.replace_parameter(port, AlgorithmInput::from(v.as_slice()))
            .unwrap()?;
        spatial!(format!("replace-{name}"), &c.difference, &vec![true; n], 1);
    }
    let stat = vec![true, false, true, false, true, true];
    p.replace_parameter(8, AlgorithmInput::from(stat.as_slice()))
        .unwrap()?;
    spatial!("replace-static", &c.difference, &vec![true; n], 1);
    let mut roi = c.roi.clone();
    roi.swap(0, 1);
    p.replace_parameter(11, AlgorithmInput::from(roi.as_slice()))
        .unwrap()?;
    spatial!("replace-roi", &c.difference, &vec![true; n], 1);
    for (port, name, v) in [
        (7, "reference", vec![f32::NAN; n]),
        (9, "phase", vec![f32::INFINITY; n]),
        (10, "fir", vec![0.; 3]),
    ] {
        let rejected = p
            .replace_parameter(port, AlgorithmInput::from(v.as_slice()))
            .unwrap()
            .is_err();
        record(
            results,
            label,
            format!("invalid-{name}-status"),
            vec![rejected as u8 as f64],
        );
        spatial!(
            format!("after-invalid-{name}"),
            &c.difference,
            &vec![true; n],
            1
        );
    }
    SccSpatialCarrierDemodulationF32::reset(&mut p);
    spatial!("reset", &c.difference, &vec![true; n], 1);
    let c = &f.pcg;
    let label = "scc-weighted-regularized-reconstructor-f32";
    let cfg = |h, l, iterations, tolerance| SccWeightedRegularizedReconstructorF32Configuration {
        measurement_count: 4,
        controller_coordinate_count: 2,
        regularization_row_count: 3,
        measurement_schema: "org.example.scc/1".into(),
        initial_interaction_matrix: Some(h),
        initial_regularization_operator: Some(l),
        regularization: c.regularization,
        iterations,
        relative_tolerance: tolerance,
        minimum_diagonal: c.minimum_diagonal,
    };
    let mut p = SccWeightedRegularizedReconstructorF32::prepare(cfg(
        c.interaction.clone(),
        c.regularizer.clone(),
        c.iterations,
        c.tolerance,
    ))?;
    macro_rules! pcg {
        ($name:expr,$m:expr,$w:expr) => {{
            let mut out = [71.; 2];
            let code =
                match SccWeightedRegularizedReconstructorF32::process(&mut p, $m, $w, &mut out) {
                    Ok(_) => 0,
                    Err(DeclarationProcessError::Algorithm(
                        SccWeightedRegularizedReconstructionError::InvalidWeight,
                    )) => 1,
                    Err(DeclarationProcessError::Algorithm(
                        SccWeightedRegularizedReconstructionError::InvalidMeasurement,
                    )) => 2,
                    Err(DeclarationProcessError::Algorithm(
                        SccWeightedRegularizedReconstructionError::SingularSystem,
                    )) => 3,
                    Err(DeclarationProcessError::Algorithm(
                        SccWeightedRegularizedReconstructionError::NotConverged,
                    )) => 4,
                    Err(e) => return Err(e.into()),
                };
            let name = $name;
            values(results, label, format!("{name}-command"), &out);
            record(
                results,
                label,
                format!("{name}-status"),
                vec![code as f64, (out == [71.; 2]) as u8 as f64],
            );
        }};
    }
    pcg!("initial", &c.measurements, &c.weights);
    for (name, scale) in [("small", 1e-30), ("large", 1e30)] {
        let m: Vec<_> = c.measurements.iter().map(|v| v * scale).collect();
        pcg!(name, &m, &c.weights);
    }
    pcg!("zero-rhs", &[0.; 4], &c.weights);
    let mut m = c.measurements.clone();
    m[2] = f32::NAN;
    pcg!("ignored-nan", &m, &c.weights);
    for (name, v) in [
        ("negative", -1.),
        ("nan", f32::NAN),
        ("infinite", f32::INFINITY),
    ] {
        let mut w = c.weights.clone();
        w[1] = v;
        pcg!(format!("weight-{name}"), &c.measurements, &w);
    }
    m[0] = f32::NAN;
    let mut w = c.weights.clone();
    w[1] = -1.;
    pcg!("failure-order", &m, &w);
    pcg!("measurement-nan", &m, &c.weights);
    pcg!("recovery", &c.measurements, &c.weights);
    for (port, name, v) in [
        (
            3,
            "interaction",
            c.interaction.iter().map(|x| x * 0.75).collect::<Vec<_>>(),
        ),
        (
            4,
            "regularizer",
            c.regularizer.iter().map(|x| x * 1.5).collect(),
        ),
    ] {
        p.replace_parameter(port, AlgorithmInput::from(v.as_slice()))
            .unwrap()?;
        pcg!(format!("replace-{name}"), &c.measurements, &c.weights);
        let bad = vec![f32::NAN; v.len()];
        record(
            results,
            label,
            format!("invalid-{name}-status"),
            vec![
                p.replace_parameter(port, AlgorithmInput::from(bad.as_slice()))
                    .unwrap()
                    .is_err() as u8 as f64,
            ],
        );
        pcg!(format!("after-invalid-{name}"), &c.measurements, &c.weights);
    }
    SccWeightedRegularizedReconstructorF32::reset(&mut p);
    pcg!("reset", &c.measurements, &c.weights);
    p = SccWeightedRegularizedReconstructorF32::prepare(cfg(
        vec![0.; 8],
        vec![0.; 6],
        c.iterations,
        c.tolerance,
    ))?;
    pcg!("singular", &c.measurements, &c.weights);
    p = SccWeightedRegularizedReconstructorF32::prepare(cfg(
        c.interaction.clone(),
        c.regularizer.clone(),
        1,
        1e-12,
    ))?;
    pcg!("bounded", &c.measurements, &c.weights);
    let c = &f.correction;
    let label = "scc-correction-f32";
    for sign in ["command_error", "correction"] {
        let mut p = SccCorrectionF32::prepare(SccCorrectionF32Configuration {
            frame_size: c.frame_size,
            controller_coordinate_count: c.bias.len(),
            image_schema: "org.example.frame/1".into(),
            exposure_duration_nanoseconds: 1000,
            coordinate_order: vec![1, 2],
            command_unit: "controller-unit".into(),
            reconstructor_sign: sign.into(),
            reference_difference: c.reference.clone(),
            correction_mask: c.mask.clone(),
            reconstructor: c.reconstructor.clone(),
            fringed_response: c.fringed_response.clone(),
            unfringed_response: c.unfringed_response.clone(),
            bias_command: c.bias.clone(),
            command_limits: (ScalarOrVector::Scalar(-10.), ScalarOrVector::Scalar(10.)),
            gain: c.gain,
            pole: c.pole,
        })?;
        let rejected = p
            .replace_properties(&[PropertyAssignment {
                id: 1,
                value: PropertyValue::Float(1.25),
            }])
            .is_err();
        record(
            results,
            label,
            format!("{sign}-invalid-pole-status"),
            vec![rejected as u8 as f64],
        );
        for e in &c.events {
            for (id, value) in [(0, e.gain), (1, e.pole)] {
                if let Some(value) = value {
                    p.replace_properties(&[PropertyAssignment {
                        id,
                        value: PropertyValue::Float(value),
                    }])?;
                }
            }
            if e.reset {
                SccCorrectionF32::reset(&mut p);
            }
            let mut domain = [0; 16];
            domain[0] = e.domain.unwrap_or(1);
            let id = SampleAcquisitionIdentity::new(domain, e.generation.unwrap_or(1), e.sequence)?;
            let kind = match e.kind {
                0 => Some(ChoppedImageKind::Fringed),
                1 => Some(ChoppedImageKind::Unfringed),
                2 => Some(ChoppedImageKind::Mixed),
                _ => None,
            };
            let sample = SampleMetadata::new(e.sequence, 0)
                .with_acquisition_identity(if e.missing_identity { None } else { Some(id) })
                .with_chopped_image_kind(kind)
                .with_corrupted(e.corrupted)
                .with_discontinuity(e.discontinuity)
                .with_exposure_duration_nanoseconds(e.duration.and_then(NonZeroU64::new));
            let mut frame = e.frame.clone();
            if e.nonfinite {
                frame[0] = f32::NAN;
            }
            let mut out = vec![71.; c.bias.len()];
            let code = match SccCorrectionF32::process_with_metadata(
                &mut p,
                &[if e.missing_metadata {
                    None
                } else {
                    Some(sample)
                }],
                &frame,
                &mut out,
            ) {
                Ok(_) => 0,
                Err(NativeProcessError::Process(DeclarationProcessError::Algorithm(e))) => {
                    match e {
                        SccCorrectionError::MetadataMissing => 1,
                        SccCorrectionError::ExposureMismatch => 2,
                        SccCorrectionError::SequenceInvalid => 3,
                        SccCorrectionError::NonFiniteInput => 4,
                        SccCorrectionError::NonFiniteCorrection => 5,
                        SccCorrectionError::CommandOutOfBounds => 6,
                        e => return Err(e.into()),
                    }
                }
                Err(e) => return Err(e.into()),
            };
            let name = format!("{sign}-{}", e.name);
            values(results, label, format!("{name}-output"), &out);
            let w = p.workspace();
            values(results, label, format!("{name}-command"), w.command());
            values(
                results,
                label,
                format!("{name}-exposure"),
                w.exposure_command(),
            );
            values(
                results,
                label,
                format!("{name}-integrator"),
                w.integrator_state(),
            );
            let id = w.acquisition_identity();
            record(
                results,
                label,
                format!("{name}-status"),
                vec![
                    code as f64,
                    out.iter().all(|&x| x == 71.) as u8 as f64,
                    w.is_primed() as u8 as f64,
                    id.map_or(0, |i| i.domain()[0]) as f64,
                    id.map_or(0, |i| i.generation()) as f64,
                    id.map_or(0, |i| i.sequence()) as f64,
                ],
            );
        }
    }
    chopped(results)?;
    Ok(())
}
fn chopped(results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let label = "chopped-image-difference-f32";
    for polarity in [true, false] {
        let mut p = ChoppedImageDifferenceF32::prepare(ChoppedImageDifferenceF32Configuration {
            input_schema: filtergraph_algorithms::schemas::CALIBRATED_PIXELS_V1.into(),
            require_acquisition_identity: false,
            pixel_count: 2,
            trigger_high_is_fringed: polarity,
        })?;
        for (i, (name, kind, trigger, missing)) in [
            ("missing", None, SampleTriggerState::Absent, true),
            ("trigger-missing", None, SampleTriggerState::Absent, false),
            ("high", None, SampleTriggerState::High, false),
            ("low", None, SampleTriggerState::Low, false),
            (
                "optical-override",
                Some(ChoppedImageKind::Fringed),
                SampleTriggerState::Low,
                false,
            ),
            (
                "mixed",
                Some(ChoppedImageKind::Mixed),
                SampleTriggerState::High,
                false,
            ),
            ("trigger-mixed", None, SampleTriggerState::Mixed, false),
            (
                "recovery",
                Some(ChoppedImageKind::Unfringed),
                SampleTriggerState::High,
                false,
            ),
            (
                "reset",
                Some(ChoppedImageKind::Fringed),
                SampleTriggerState::Absent,
                false,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            if name == "reset" {
                ChoppedImageDifferenceF32::reset(&mut p);
            }
            let sample = SampleMetadata::new(i as u64 + 1, 0)
                .with_chopped_image_kind(kind)
                .with_trigger_state(Some(trigger));
            let mut out = [71.; 2];
            let mut valid = [true; 2];
            let mut status = [99];
            let image = [10. + i as f32, 20. - i as f32];
            let code = match ChoppedImageDifferenceF32::process_with_metadata(
                &mut p,
                &[if missing { None } else { Some(sample) }, None],
                &image,
                &[true, true],
                &mut out,
                &mut valid,
                &mut status,
            ) {
                Ok(_) => 0,
                Err(NativeProcessError::Process(DeclarationProcessError::Algorithm(
                    ChoppedImageDifferenceError::MetadataMissing,
                ))) => 1,
                Err(NativeProcessError::Process(DeclarationProcessError::Algorithm(
                    ChoppedImageDifferenceError::TriggerMissing,
                ))) => 2,
                Err(e) => return Err(e.into()),
            };
            let name = format!("{polarity}-{name}");
            values(results, label, format!("{name}-difference"), &out);
            record(
                results,
                label,
                format!("{name}-status"),
                vec![
                    code as f64,
                    status[0] as f64,
                    valid[0] as u8 as f64,
                    valid[1] as u8 as f64,
                    p.workspace().is_primed() as u8 as f64,
                ],
            );
        }
    }
    Ok(())
}
