//! Summary diagnostics and complete-image weighted sensing against real JFG APIs.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, ShackHartmannSummaryF32, ShackHartmannSummaryF32Configuration,
    ShackHartmannWeightedCentroidF32 as W,
    ShackHartmannWeightedCentroidF32Configuration as Configuration,
};
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    summary: Summary,
    weighted: Weighted,
}
#[derive(Deserialize)]
struct Summary {
    subaperture_count: usize,
    cases: Vec<SummaryCase>,
}
#[derive(Deserialize)]
struct SummaryCase {
    name: String,
    slopes: Vec<[f32; 2]>,
    flux: Vec<f32>,
    validity: Vec<bool>,
}
#[derive(Deserialize)]
struct Weighted {
    image_rows: usize,
    image_columns: usize,
    subaperture_rows: usize,
    subaperture_columns: usize,
    origins: Vec<[usize; 2]>,
    coordinate_scale: f32,
    pixel_threshold: f32,
    flux_threshold: f32,
    reference_slopes: Vec<f32>,
    active: Vec<bool>,
    weights: Vec<Vec<f32>>,
    image: Vec<f32>,
    replacement_weights: Vec<Vec<f32>>,
    replacement_coordinates: Vec<[f32; 2]>,
    replacement_reference_slopes: Vec<[f32; 2]>,
    replacement_thresholds: Vec<[f32; 2]>,
    replacement_active: Vec<bool>,
    replacement_origins: Vec<[u32; 2]>,
}
fn flag(v: bool) -> f64 {
    if v { 1.0 } else { 0.0 }
}
fn emit(results: &mut Vec<Record>, label: &str, case: &str, values: &[f32], validity: &[bool]) {
    let mut classes: Vec<f64> = values
        .iter()
        .map(|v| {
            if v.is_nan() {
                1.0
            } else if *v == f32::INFINITY {
                2.0
            } else if *v == f32::NEG_INFINITY {
                3.0
            } else {
                0.0
            }
        })
        .collect();
    classes.extend(validity.iter().copied().map(flag));
    record(
        results,
        label,
        format!("{case}-values"),
        values
            .iter()
            .map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 })
            .collect(),
    );
    record(results, label, format!("{case}-status"), classes);
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let mut summary = ShackHartmannSummaryF32::prepare(ShackHartmannSummaryF32Configuration {
        subaperture_count: f.summary.subaperture_count,
    })?;
    for case in &f.summary.cases {
        let slopes = case.slopes.iter().flatten().copied().collect::<Vec<_>>();
        let mut output = [71.0; 8];
        ShackHartmannSummaryF32::process(
            &mut summary,
            &slopes,
            &case.flux,
            &case.validity,
            &mut output,
        )?;
        emit(results, "shack-summary-f32", &case.name, &output, &[]);
    }
    for (name, count) in [("empty", 0), ("limit", (1 << 24) + 1), ("maximum", 1 << 24)] {
        let accepted = ShackHartmannSummaryF32::prepare(ShackHartmannSummaryF32Configuration {
            subaperture_count: count,
        })
        .is_ok();
        record(
            results,
            "shack-summary-f32",
            format!("configuration-{name}-status"),
            vec![flag(accepted)],
        );
    }
    let case = &f.summary.cases[0];
    let slopes = case.slopes.iter().flatten().copied().collect::<Vec<_>>();
    for name in [
        "slopes-extent",
        "flux-extent",
        "validity-extent",
        "output-extent",
    ] {
        let mut output = [71.0; 8];
        let rejected = match name {
            "slopes-extent" => ShackHartmannSummaryF32::process(
                &mut summary,
                &slopes[..slopes.len() - 1],
                &case.flux,
                &case.validity,
                &mut output,
            )
            .is_err(),
            "flux-extent" => ShackHartmannSummaryF32::process(
                &mut summary,
                &slopes,
                &case.flux[..case.flux.len() - 1],
                &case.validity,
                &mut output,
            )
            .is_err(),
            "validity-extent" => ShackHartmannSummaryF32::process(
                &mut summary,
                &slopes,
                &case.flux,
                &case.validity[..case.validity.len() - 1],
                &mut output,
            )
            .is_err(),
            _ => ShackHartmannSummaryF32::process(
                &mut summary,
                &slopes,
                &case.flux,
                &case.validity,
                &mut output[..7],
            )
            .is_err(),
        };
        record(
            results,
            "shack-summary-f32",
            format!("{name}-status"),
            vec![flag(!rejected), flag(output == [71.0; 8])],
        );
    }
    let c = f.weighted;
    let n = c.origins.len();
    let mut p = W::prepare(Configuration {
        image_schema: "pipewireao/calibrated-pixels-f32/1".into(),
        subaperture_count: n,
        initial_subaperture_origins: Some(c.origins.clone()),
        coordinate_scale: c.coordinate_scale,
        pixel_threshold: c.pixel_threshold,
        flux_threshold: c.flux_threshold,
        reference_slopes: Some(c.reference_slopes),
        active: Some(c.active),
        initial_weights: c.weights,
        image_size: (c.image_rows, c.image_columns),
        subaperture_size: (c.subaperture_rows, c.subaperture_columns),
        simd_width: 0,
    })?;
    macro_rules! measure {
        ($case:expr,$image:expr) => {{
            let mut slopes = vec![71.0; 2 * n];
            let mut flux = vec![71.0; n];
            let mut valid = vec![false; n];
            W::process(&mut p, $image, &mut slopes, &mut flux, &mut valid)?;
            slopes.extend(flux);
            emit(results, "shack-weighted-f32", $case, &slopes, &valid);
        }};
    }
    measure!("initial", &c.image);
    measure!("low-flux", &vec![0.0; c.image.len()]);
    let zero_weight = c.origins[0][0] * c.image_columns + c.origins[0][1];
    for (name, value) in [
        ("zero-weight-nan", f32::NAN),
        ("zero-weight-inf", f32::INFINITY),
        ("weighted-overflow", f32::MAX),
    ] {
        let mut image = c.image.clone();
        let index = if name == "weighted-overflow" {
            zero_weight + 2
        } else {
            zero_weight
        };
        image[index] = value;
        measure!(name, &image);
    }
    let replacements = [
        (
            "weights",
            9,
            c.replacement_weights
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        ),
        (
            "coordinates",
            5,
            c.replacement_coordinates
                .iter()
                .flatten()
                .copied()
                .collect(),
        ),
        (
            "references",
            6,
            c.replacement_reference_slopes
                .iter()
                .flatten()
                .copied()
                .collect(),
        ),
        (
            "thresholds",
            7,
            c.replacement_thresholds.iter().flatten().copied().collect(),
        ),
    ];
    for (name, port, values) in replacements {
        p.replace_parameter(port, AlgorithmInput::from(values.as_slice()))
            .ok_or("missing parameter")??;
        measure!(&format!("replaced-{name}"), &c.image);
        let mut invalid = values.clone();
        invalid[0] = f32::NAN;
        let rejected = p
            .replace_parameter(port, AlgorithmInput::from(invalid.as_slice()))
            .ok_or("missing parameter")?
            .is_err();
        record(
            results,
            "shack-weighted-f32",
            format!("invalid-{name}-status"),
            vec![flag(!rejected)],
        );
        measure!(&format!("after-invalid-{name}"), &c.image);
    }
    p.replace_parameter(8, AlgorithmInput::from(c.replacement_active.as_slice()))
        .ok_or("missing active")??;
    measure!("replaced-active", &c.image);
    let invalid_active = &c.replacement_active[..c.replacement_active.len() - 1];
    let rejected = p
        .replace_parameter(8, AlgorithmInput::from(invalid_active))
        .ok_or("missing active")?
        .is_err();
    record(
        results,
        "shack-weighted-f32",
        "invalid-active-status".into(),
        vec![flag(!rejected)],
    );
    measure!("after-invalid-active", &c.image);
    let origins = c
        .replacement_origins
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    p.replace_parameter(4, AlgorithmInput::from(origins.as_slice()))
        .ok_or("missing origins")??;
    measure!("replaced-origins", &c.image);
    let mut invalid = origins;
    invalid[0] = c.image_rows as u32;
    let rejected = p
        .replace_parameter(4, AlgorithmInput::from(invalid.as_slice()))
        .ok_or("missing origins")?
        .is_err();
    record(
        results,
        "shack-weighted-f32",
        "invalid-origins-status".into(),
        vec![flag(!rejected)],
    );
    measure!("after-invalid-origins", &c.image);
    p.reset();
    measure!("after-reset", &c.image);
    Ok(())
}
