//! Exact new JFG public scientific contracts, including acquisition lifecycle.
use super::{Record, record};
use filtergraph_algorithms::*;
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    physical: Physical,
    validity: Validity,
    chopped: Chopped,
}
#[derive(Deserialize)]
struct Physical {
    residual_count: usize,
    physical_count: usize,
    mode_count: usize,
    physical_to_modes: Vec<f32>,
    modes_to_reference: Vec<f32>,
    offset: Vec<f32>,
    replacement_offset: Vec<f32>,
    replacement_physical_to_modes: Vec<f32>,
    replacement_modes_to_reference: Vec<f32>,
    residual: Vec<f32>,
    sentinel: f32,
}
#[derive(Deserialize)]
struct Validity {
    input: Vec<f32>,
}
#[derive(Deserialize)]
struct Chopped {
    pixel_count: usize,
    events: Vec<Event>,
}
#[derive(Deserialize)]
struct Event {
    name: String,
    sequence: u64,
    kind: u8,
    #[serde(default = "one")]
    domain: u8,
    #[serde(default = "one64")]
    generation: u64,
    #[serde(default)]
    trigger: u8,
    #[serde(default)]
    missing_metadata: bool,
    #[serde(default)]
    missing_identity: bool,
    #[serde(default)]
    corrupted: bool,
    #[serde(default)]
    discontinuity: bool,
    #[serde(default)]
    nonfinite: bool,
    #[serde(default)]
    invalid_pixel: bool,
    #[serde(default)]
    reset: bool,
}
fn one() -> u8 {
    1
}
fn one64() -> u64 {
    1
}
fn values(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let c = f.physical;
    let label = "physical-offset-reference-f32";
    for sign in [-1.0, 1.0] {
        let prefix = if sign < 0.0 { "negative" } else { "positive" };
        let mut p = PhysicalOffsetReferenceF32::prepare(PhysicalOffsetReferenceF32Configuration {
            residual_count: c.residual_count,
            physical_count: c.physical_count,
            mode_count: c.mode_count,
            reference_sign: sign,
            residual_schema: "org.example.residual/1".into(),
            physical_schema: "org.example.physical/1".into(),
            physical_to_modes_schema: "org.example.q/1".into(),
            modes_to_reference_schema: "org.example.c/1".into(),
        })?;
        for name in [
            "neutral",
            "q",
            "c",
            "offset",
            "replace-offset",
            "replace-q",
            "replace-c",
            "reset",
            "after-invalid",
        ] {
            let parameter = match name {
                "q" => Some((4, &c.physical_to_modes)),
                "c" => Some((5, &c.modes_to_reference)),
                "offset" => Some((3, &c.offset)),
                "replace-offset" => Some((3, &c.replacement_offset)),
                "replace-q" => Some((4, &c.replacement_physical_to_modes)),
                "replace-c" => Some((5, &c.replacement_modes_to_reference)),
                _ => None,
            };
            if let Some((port, v)) = parameter {
                p.replace_parameter(port, AlgorithmInput::from(v.as_slice()))
                    .ok_or("parameter absent")??;
            }
            if name == "reset" {
                PhysicalOffsetReferenceF32::reset(&mut p);
            }
            if name == "after-invalid" {
                for (port, key, len) in [
                    (3, "offset", c.physical_count),
                    (4, "q", c.mode_count * c.physical_count),
                    (5, "c", c.residual_count * c.mode_count),
                ] {
                    let rejected = p
                        .replace_parameter(
                            port,
                            AlgorithmInput::from(vec![f32::NAN; len].as_slice()),
                        )
                        .ok_or("parameter absent")?
                        .is_err();
                    record(
                        results,
                        label,
                        format!("{prefix}-invalid-{key}-status"),
                        vec![rejected as u8 as f64],
                    );
                }
            }
            let mut out = vec![c.sentinel; c.residual_count];
            let mut physical = vec![c.sentinel; c.physical_count];
            PhysicalOffsetReferenceF32::process(&mut p, &c.residual, &mut out, &mut physical)?;
            record(
                results,
                label,
                format!("{prefix}-{name}-residual"),
                values(&out),
            );
            record(
                results,
                label,
                format!("{prefix}-{name}-offset"),
                values(&physical),
            );
        }
        for (name, v) in [("nonfinite", f32::NAN), ("overflow", f32::MAX)] {
            if name == "overflow" {
                p.replace_parameter(
                    3,
                    AlgorithmInput::from(vec![1.0_f32; c.physical_count].as_slice()),
                )
                .ok_or("parameter absent")??;
                p.replace_parameter(
                    4,
                    AlgorithmInput::from(vec![1.0_f32; c.mode_count * c.physical_count].as_slice()),
                )
                .ok_or("parameter absent")??;
                p.replace_parameter(
                    5,
                    AlgorithmInput::from(
                        vec![
                            f32::MAX / (c.physical_count * c.mode_count) as f32 * 0.25;
                            c.residual_count * c.mode_count
                        ]
                        .as_slice(),
                    ),
                )
                .ok_or("parameter absent")??;
            }
            let mut residual = c.residual.clone();
            residual[0] = v * sign;
            let mut out = vec![c.sentinel; c.residual_count];
            let mut physical = vec![c.sentinel; c.physical_count];
            let code = match PhysicalOffsetReferenceF32::process(
                &mut p,
                &residual,
                &mut out,
                &mut physical,
            ) {
                Ok(()) => 0,
                Err(DeclarationProcessError::Algorithm(
                    VectorLinearCombinationError::NonFiniteInput,
                )) => 1,
                Err(DeclarationProcessError::Algorithm(
                    VectorLinearCombinationError::NonFiniteOutput,
                )) => 2,
                Err(e) => return Err(e.into()),
            };
            record(
                results,
                label,
                format!("{prefix}-{name}-status"),
                vec![
                    code as f64,
                    out.iter().all(|&v| v == c.sentinel) as u8 as f64,
                    physical.iter().all(|&v| v == c.sentinel) as u8 as f64,
                ],
            );
        }
        for (key, port, len) in [
            ("modes-overflow", 4, c.mode_count * c.physical_count),
            ("correction-overflow", 5, c.residual_count * c.mode_count),
        ] {
            let rejected = p
                .replace_parameter(port, AlgorithmInput::from(vec![f32::MAX; len].as_slice()))
                .ok_or("parameter absent")?
                .is_err();
            record(
                results,
                label,
                format!("{prefix}-{key}-status"),
                vec![rejected as u8 as f64],
            );
        }
    }
    let label = "pixel-validity-f32";
    let mut p = PixelValidityF32::prepare(PixelValidityF32Configuration {
        extent: f.validity.input.len(),
        input_schema: "org.example.adu/1".into(),
    })?;
    for name in ["finite", "nan", "positive-inf", "negative-inf", "reset"] {
        let mut input = f.validity.input.clone();
        match name {
            "nan" => input[1] = f32::NAN,
            "positive-inf" => input[2] = f32::INFINITY,
            "negative-inf" => input[3] = f32::NEG_INFINITY,
            "reset" => PixelValidityF32::reset(&mut p),
            _ => {}
        }
        let mut out = vec![true; input.len()];
        PixelValidityF32::process(&mut p, &input, &mut out)?;
        record(
            results,
            label,
            format!("{name}-validity-status"),
            out.iter().map(|&v| v as u8 as f64).collect(),
        );
    }
    let label = "chopped-image-difference-f32";
    for polarity in [true, false] {
        let n = f.chopped.pixel_count;
        let mut p = ChoppedImageDifferenceF32::prepare(ChoppedImageDifferenceF32Configuration {
            pixel_count: n,
            trigger_high_is_fringed: polarity,
            input_schema: "org.example.raw/1".into(),
            require_acquisition_identity: true,
        })?;
        for (i, e) in f.chopped.events.iter().enumerate() {
            if e.reset {
                ChoppedImageDifferenceF32::reset(&mut p);
            }
            let mut domain = [0; 16];
            domain[0] = e.domain;
            let kind = match e.kind {
                0 => Some(ChoppedImageKind::Fringed),
                1 => Some(ChoppedImageKind::Unfringed),
                2 => Some(ChoppedImageKind::Mixed),
                _ => None,
            };
            let trigger = match e.trigger {
                1 => SampleTriggerState::Low,
                2 => SampleTriggerState::High,
                3 => SampleTriggerState::Mixed,
                _ => SampleTriggerState::Absent,
            };
            let sample = if e.missing_metadata {
                None
            } else {
                Some(
                    SampleMetadata::new(e.sequence, 0)
                        .with_acquisition_identity(if e.missing_identity {
                            None
                        } else {
                            Some(SampleAcquisitionIdentity::new(
                                domain,
                                e.generation,
                                e.sequence,
                            )?)
                        })
                        .with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(20))
                        .with_chopped_image_kind(kind)
                        .with_trigger_state(Some(trigger))
                        .with_corrupted(e.corrupted)
                        .with_discontinuity(e.discontinuity),
                )
            };
            let mut image: Vec<f32> = (0..n)
                .map(|j| 10.0 + 2.0 * i as f32 + 3.0 * j as f32)
                .collect();
            if e.nonfinite {
                image[0] = f32::NAN;
            }
            let mut valid = vec![true; n];
            if e.invalid_pixel {
                valid[n - 1] = false;
            }
            let mut out = vec![71.0; n];
            let mut output_validity = vec![true; n];
            let mut status = [99];
            let code = match ChoppedImageDifferenceF32::process_with_metadata(
                &mut p,
                &[sample, None],
                &image,
                &valid,
                &mut out,
                &mut output_validity,
                &mut status,
            ) {
                Ok(_) => 0,
                Err(NativeProcessError::Process(DeclarationProcessError::Algorithm(
                    ChoppedImageDifferenceError::MetadataMissing,
                ))) => 1,
                Err(NativeProcessError::Process(DeclarationProcessError::Algorithm(
                    ChoppedImageDifferenceError::TriggerMissing,
                ))) => 2,
                Err(NativeProcessError::Process(DeclarationProcessError::Algorithm(
                    ChoppedImageDifferenceError::SequenceInvalid,
                ))) => 3,
                Err(e) => return Err(e.into()),
            };
            let name = format!("strict-{polarity}-{}", e.name);
            record(results, label, format!("{name}-difference"), values(&out));
            let mut state = vec![code as f64, status[0] as f64];
            state.extend(output_validity.iter().map(|&v| v as u8 as f64));
            state.push(p.workspace().is_primed() as u8 as f64);
            record(results, label, format!("{name}-status"), state);
        }
    }
    Ok(())
}
