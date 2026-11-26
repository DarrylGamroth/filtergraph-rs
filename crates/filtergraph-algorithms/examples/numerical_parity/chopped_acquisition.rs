//! Opt-in physical acquisition admission through the actual checked declaration.
use super::{Record, record};
use filtergraph_algorithms::*;
use serde::Deserialize;
use std::{error::Error, num::NonZeroU64};
#[derive(Deserialize)]
struct Fixture {
    sentinel: f32,
    events: Vec<Event>,
}
#[derive(Deserialize)]
struct Event {
    name: String,
    sequence: u64,
    publication_sequence: u64,
    kind: i8,
    frame: Vec<f32>,
    domain: Option<u8>,
    generation: Option<u64>,
    duration: Option<u64>,
    #[serde(default)]
    missing_metadata: bool,
    #[serde(default)]
    missing_identity: bool,
    #[serde(default)]
    missing_duration: bool,
    #[serde(default)]
    discontinuity: bool,
    #[serde(default)]
    corrupted: bool,
    #[serde(default)]
    reset: bool,
}
pub(super) fn run(source: &str, records: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let fixture: Fixture = toml::from_str(source)?;
    for polarity in [true, false] {
        let mut p = ChoppedImageDifferenceF32::prepare(ChoppedImageDifferenceF32Configuration {
            pixel_count: 2,
            trigger_high_is_fringed: polarity,
            input_schema: "test.chopped-image/1".into(),
            require_acquisition_identity: true,
        })?;
        for event in &fixture.events {
            if event.reset {
                p.reset();
            }
            let mut domain = [0; 16];
            domain[0] = event.domain.unwrap_or(1);
            let identity = if event.missing_identity {
                None
            } else {
                Some(SampleAcquisitionIdentity::new(
                    domain,
                    event.generation.unwrap_or(3),
                    event.sequence,
                )?)
            };
            let kind = match event.kind {
                0 => Some(ChoppedImageKind::Fringed),
                1 => Some(ChoppedImageKind::Unfringed),
                2 => Some(ChoppedImageKind::Mixed),
                3 => Some(ChoppedImageKind::Absent),
                _ => None,
            };
            let metadata = if event.missing_metadata {
                None
            } else {
                Some(
                    SampleMetadata::new(event.publication_sequence, 0)
                        .with_acquisition_identity(identity)
                        .with_chopped_image_kind(kind)
                        .with_exposure_duration_nanoseconds(if event.missing_duration {
                            None
                        } else {
                            NonZeroU64::new(event.duration.unwrap_or(20))
                        })
                        .with_trigger_state(Some(SampleTriggerState::High))
                        .with_discontinuity(event.discontinuity)
                        .with_corrupted(event.corrupted),
                )
            };
            let (mut difference, mut valid, mut status) =
                ([fixture.sentinel; 2], [true; 2], [99_u8]);
            let code = match ChoppedImageDifferenceF32::process_with_metadata(
                &mut p,
                &[metadata, None],
                &event.frame,
                &[true; 2],
                &mut difference,
                &mut valid,
                &mut status,
            ) {
                Ok(_) => 0,
                Err(NativeProcessError::Process(DeclarationProcessError::Algorithm(error))) => {
                    match error {
                        ChoppedImageDifferenceError::MetadataMissing => 1,
                        ChoppedImageDifferenceError::TriggerMissing => 2,
                        ChoppedImageDifferenceError::SequenceInvalid => 3,
                        ChoppedImageDifferenceError::ExposureMismatch => 4,
                        other => return Err(other.into()),
                    }
                }
                Err(other) => return Err(other.into()),
            };
            let name = format!("acquisition-{polarity}-{}", event.name);
            let values = difference
                .into_iter()
                .map(f64::from)
                .chain(valid.into_iter().map(|v| v as u8 as f64))
                .chain([status[0] as f64])
                .collect();
            record(
                records,
                "chopped-image-difference-f32",
                format!("{name}-outputs"),
                values,
            );
            record(
                records,
                "chopped-image-difference-f32",
                format!("{name}-status"),
                vec![
                    code as f64,
                    (difference == [fixture.sentinel; 2] && valid == [true; 2] && status == [99])
                        as u8 as f64,
                    p.workspace().is_primed() as u8 as f64,
                ],
            );
            // Portable test codes: none=0, fringed=1, unfringed=2,
            // mixed=3, absent=4. Native enum discriminants differ by language.
            let kind = p.workspace().retained_kind().map_or(0, |kind| match kind {
                ChoppedImageKind::Fringed => 1,
                ChoppedImageKind::Unfringed => 2,
                ChoppedImageKind::Mixed => 3,
                ChoppedImageKind::Absent => 4,
            });
            let image = p.workspace().retained_image().unwrap_or(&[0.0; 2]);
            let validity = p.workspace().retained_validity().unwrap_or(&[false; 2]);
            record(
                records,
                "chopped-image-difference-f32",
                format!("{name}-retained"),
                vec![
                    kind as f64,
                    image[0] as f64,
                    image[1] as f64,
                    validity[0] as u8 as f64,
                    validity[1] as u8 as f64,
                ],
            );
        }
    }
    Ok(())
}
