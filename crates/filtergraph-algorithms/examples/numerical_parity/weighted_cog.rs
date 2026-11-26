//! Weighted-CoG declarations with nonfinite flux encoded as discrete categories.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, PropertyAssignment, PropertyValue, WeightedCenterOfGravityF32 as D,
    WeightedCenterOfGravityF32Configuration as C,
};
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    coordinates: Vec<[f32; 2]>,
    weights: Vec<f32>,
    pixels: Vec<f32>,
    negative_pixels: Vec<f32>,
    reference_centroid: [f32; 2],
    pixel_threshold: f32,
    flux_threshold_equal: f32,
    flux_threshold_above: f32,
    replacement_coordinates: Vec<[f32; 2]>,
    replacement_weights: Vec<f32>,
    zero_weight_coordinate: usize,
    nonzero_weight_coordinate: usize,
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let label = "weighted-center-of-gravity-f32";
    let n = f.pixels.len();
    let mut p = D::prepare(C {
        pixel_count: n,
        initial_coordinates: f.coordinates.clone(),
        initial_weights: f.weights.clone(),
    })?;
    let mut offset = [71.0; 2];
    let mut flux = [73.0];
    let mut valid = [false];
    macro_rules! emit {
        ($phase:expr,$pixels:expr) => {{
            D::process(&mut p, $pixels, &mut offset, &mut flux, &mut valid)?;
            let category = if flux[0].is_nan() {
                1
            } else if flux[0] == f32::INFINITY {
                2
            } else if flux[0] == f32::NEG_INFINITY {
                3
            } else {
                0
            };
            record(
                results,
                label,
                format!("{}-status", $phase),
                vec![valid[0] as u8 as f64, category as f64],
            );
            record(
                results,
                label,
                format!("{}-offset", $phase),
                offset.iter().map(|&x| x as f64).collect(),
            );
            record(
                results,
                label,
                format!("{}-flux", $phase),
                vec![if flux[0].is_finite() {
                    flux[0] as f64
                } else {
                    0.0
                }],
            );
        }};
    }
    emit!("initial", &f.pixels);
    p.replace_properties(&[
        PropertyAssignment {
            id: 0,
            value: PropertyValue::Float(f.reference_centroid[0]),
        },
        PropertyAssignment {
            id: 1,
            value: PropertyValue::Float(f.reference_centroid[1]),
        },
    ])?;
    emit!("reference", &f.pixels);
    p.replace_properties(&[
        PropertyAssignment {
            id: 2,
            value: PropertyValue::Bool(true),
        },
        PropertyAssignment {
            id: 3,
            value: PropertyValue::Float(f.pixel_threshold),
        },
    ])?;
    emit!("pixel-threshold", &f.pixels);
    p.replace_properties(&[PropertyAssignment {
        id: 4,
        value: PropertyValue::Float(f.flux_threshold_equal),
    }])?;
    emit!("flux-equal", &f.pixels);
    p.replace_properties(&[PropertyAssignment {
        id: 4,
        value: PropertyValue::Float(f.flux_threshold_above),
    }])?;
    emit!("flux-above", &f.pixels);
    p.replace_properties(&[
        PropertyAssignment {
            id: 2,
            value: PropertyValue::Bool(false),
        },
        PropertyAssignment {
            id: 4,
            value: PropertyValue::Float(0.0),
        },
    ])?;
    emit!("threshold-disabled", &f.pixels);
    emit!("negative-pixels", &f.negative_pixels);
    let zeros = vec![0.0_f32; n];
    p.replace_parameter(5, AlgorithmInput::from(zeros.as_slice()))
        .ok_or("weights missing")??;
    emit!("zero-weights", &f.pixels);
    for (name, value) in [
        ("nan", f32::NAN),
        ("positive-inf", f32::INFINITY),
        ("negative-inf", f32::NEG_INFINITY),
    ] {
        let mut pixels = f.pixels.clone();
        pixels[f.zero_weight_coordinate] = value;
        emit!(format!("zero-weight-{name}"), &pixels);
    }
    p.replace_parameter(5, AlgorithmInput::from(f.weights.as_slice()))
        .ok_or("weights missing")??;
    emit!("restored-weights", &f.pixels);
    for (name, value) in [
        ("nan", f32::NAN),
        ("positive-inf", f32::INFINITY),
        ("negative-inf", f32::NEG_INFINITY),
        ("weighted-overflow", f32::MAX),
        ("moment-overflow", f32::MAX / 4.0),
    ] {
        let mut pixels = f.pixels.clone();
        pixels[f.nonzero_weight_coordinate] = value;
        emit!(name, &pixels);
        emit!(format!("after-{name}"), &f.pixels);
    }
    for (name, id, value) in [
        ("reference-nan", 0, f32::NAN),
        ("pixel-threshold-inf", 3, f32::INFINITY),
        ("flux-negative", 4, -0.25),
    ] {
        let rejected = p
            .replace_properties(&[PropertyAssignment {
                id,
                value: PropertyValue::Float(value),
            }])
            .is_err();
        record(
            results,
            label,
            format!("invalid-{name}-status"),
            vec![rejected as u8 as f64],
        );
        emit!(format!("after-invalid-{name}"), &f.pixels);
    }
    let coordinates: Vec<f32> = f
        .replacement_coordinates
        .iter()
        .flatten()
        .copied()
        .collect();
    p.replace_parameter(4, AlgorithmInput::from(coordinates.as_slice()))
        .ok_or("coordinates missing")??;
    emit!("replaced-coordinates", &f.pixels);
    let mut invalid = coordinates.clone();
    invalid[1] = f32::NAN;
    let rejected = p
        .replace_parameter(4, AlgorithmInput::from(invalid.as_slice()))
        .ok_or("coordinates missing")?
        .is_err();
    record(
        results,
        label,
        "invalid-coordinate-nan-status".into(),
        vec![rejected as u8 as f64],
    );
    emit!("after-invalid-coordinate-nan", &f.pixels);
    let rejected = p
        .replace_parameter(
            4,
            AlgorithmInput::from(&coordinates[..coordinates.len() - 1]),
        )
        .ok_or("coordinates missing")?
        .is_err();
    record(
        results,
        label,
        "invalid-coordinate-extent-status".into(),
        vec![rejected as u8 as f64],
    );
    emit!("after-invalid-coordinate-extent", &f.pixels);
    p.replace_parameter(5, AlgorithmInput::from(f.replacement_weights.as_slice()))
        .ok_or("weights missing")??;
    emit!("replaced-weights", &f.pixels);
    for (name, value) in [
        ("negative", -0.25),
        ("nan", f32::NAN),
        ("positive-inf", f32::INFINITY),
        ("negative-inf", f32::NEG_INFINITY),
    ] {
        let mut weights = f.replacement_weights.clone();
        weights[f.nonzero_weight_coordinate] = value;
        let rejected = p
            .replace_parameter(5, AlgorithmInput::from(weights.as_slice()))
            .ok_or("weights missing")?
            .is_err();
        record(
            results,
            label,
            format!("invalid-weight-{name}-status"),
            vec![rejected as u8 as f64],
        );
        emit!(format!("after-invalid-weight-{name}"), &f.pixels);
    }
    let rejected = p
        .replace_parameter(5, AlgorithmInput::from(&f.replacement_weights[..n - 1]))
        .ok_or("weights missing")?
        .is_err();
    record(
        results,
        label,
        "invalid-weight-extent-status".into(),
        vec![rejected as u8 as f64],
    );
    emit!("after-invalid-weight-extent", &f.pixels);
    p.reset();
    emit!("after-reset", &f.pixels);
    offset.fill(71.0);
    flux.fill(73.0);
    valid.fill(true);
    let rejected = D::process(
        &mut p,
        &f.pixels[..n - 1],
        &mut offset,
        &mut flux,
        &mut valid,
    )
    .is_err();
    record(
        results,
        label,
        "input-extent-status".into(),
        vec![
            rejected as u8 as f64,
            (offset == [71.0; 2] && flux == [73.0] && valid == [true]) as u8 as f64,
        ],
    );
    emit!("after-input-extent", &f.pixels);
    for name in ["offset", "flux", "validity"] {
        offset.fill(71.0);
        flux.fill(73.0);
        valid.fill(true);
        let rejected = match name {
            "offset" => {
                D::process(&mut p, &f.pixels, &mut offset[..1], &mut flux, &mut valid).is_err()
            }
            "flux" => {
                D::process(&mut p, &f.pixels, &mut offset, &mut flux[..0], &mut valid).is_err()
            }
            _ => D::process(&mut p, &f.pixels, &mut offset, &mut flux, &mut valid[..0]).is_err(),
        };
        record(
            results,
            label,
            format!("{name}-extent-status"),
            vec![
                rejected as u8 as f64,
                (offset == [71.0; 2] && flux == [73.0] && valid == [true]) as u8 as f64,
            ],
        );
    }
    emit!("after-output-extents", &f.pixels);
    Ok(())
}
