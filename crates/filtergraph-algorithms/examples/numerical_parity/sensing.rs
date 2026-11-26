//! CPU sensing parity inputs; IEEE categories are compared separately from finite values.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, AlgorithmPlan, AlgorithmProperties, CenterOfGravityOutput, CenterOfGravityPlan,
    PixelSelectionF32, PixelSelectionF32Configuration, PropertyAssignment, PropertyValue,
    RegionExtractionF32, RegionExtractionF32Configuration, ShackHartmannImagePlan,
    ShackHartmannOutput,
};
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize)]
struct Fixture {
    image: Image,
    cog: Cog,
    shack: Shack,
    selection: Selection,
    regions: Regions,
}
#[derive(Deserialize)]
struct Image {
    rows: usize,
    columns: usize,
    pixels: Vec<f32>,
}
#[derive(Deserialize)]
struct Cog {
    coordinates: Vec<[f32; 2]>,
    replacement_coordinates: Vec<[f32; 2]>,
    reference: [f32; 2],
    pixel_threshold: f32,
    flux_threshold: f32,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    pixels: Vec<f32>,
}
#[derive(Deserialize)]
struct Shack {
    region_rows: usize,
    region_columns: usize,
    origins: Vec<[usize; 2]>,
    replacement_origins: Vec<[u32; 2]>,
    coordinate_scale: f32,
    pixel_threshold: f32,
    flux_threshold: f32,
    reference_slopes: Vec<f32>,
    active: Vec<bool>,
    replacement_active: Vec<bool>,
}
#[derive(Deserialize)]
struct Selection {
    mask: Vec<bool>,
    replacement_mask: Vec<bool>,
}
#[derive(Deserialize)]
struct Regions {
    region_rows: usize,
    region_columns: usize,
    origins: Vec<[usize; 2]>,
    replacement_origins: Vec<[u32; 2]>,
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
    classes.extend(validity.iter().map(|&v| if v { 1.0 } else { 0.0 }));
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
fn cog_case(
    plan: &CenterOfGravityPlan<f32>,
    pixels: &[f32],
    name: &str,
    results: &mut Vec<Record>,
) -> Result<(), Box<dyn Error>> {
    let mut offset = [71.0; 2];
    let mut flux = 71.0;
    let mut valid = true;
    plan.process(
        CenterOfGravityOutput::new(&mut offset, &mut flux, &mut valid),
        &mut plan.workspace(),
        pixels,
    )?;
    emit(
        results,
        "sensing-cog-f32",
        name,
        &[offset[0], offset[1], flux],
        &[valid],
    );
    Ok(())
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let mut cog = CenterOfGravityPlan::new(
        f.cog.coordinates.iter().map(|p| p[0]).collect::<Vec<_>>(),
        f.cog.coordinates.iter().map(|p| p[1]).collect::<Vec<_>>(),
        f.cog.reference,
        Some(f.cog.pixel_threshold),
        f.cog.flux_threshold,
    )?;
    for case in &f.cog.cases {
        cog_case(&cog, &case.pixels, &case.name, results)?;
    }
    for (name, value) in [
        ("nan", f32::NAN),
        ("positive-inf", f32::INFINITY),
        ("negative-inf", f32::NEG_INFINITY),
        ("overflow", f32::MAX),
    ] {
        let mut pixels = vec![value; f.cog.coordinates.len()];
        if name != "overflow" {
            pixels = f.cog.cases[0].pixels.clone();
            *pixels.last_mut().ok_or("empty CoG")? = value;
        }
        cog_case(&cog, &pixels, name, results)?;
    }
    cog = cog.with_pair_interleaved_coordinates(
        &f.cog
            .replacement_coordinates
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
    )?;
    cog_case(
        &cog,
        &f.cog.cases[0].pixels,
        "coordinates-replaced",
        results,
    )?;
    cog = cog.prepare_properties(&[PropertyAssignment {
        id: 2,
        value: PropertyValue::Bool(false),
    }])?;
    cog_case(
        &cog,
        &f.cog.cases.last().ok_or("empty CoG cases")?.pixels,
        "threshold-disabled",
        results,
    )?;
    let old = cog.clone();
    let mut bad = f
        .cog
        .replacement_coordinates
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    bad[0] = f32::NAN;
    let rejected = cog.with_pair_interleaved_coordinates(&bad).is_err();
    record(
        results,
        "sensing-cog-f32",
        "invalid-coordinates-status".into(),
        vec![
            if rejected { 0.0 } else { 1.0 },
            if cog.x_coordinates() == old.x_coordinates()
                && cog.y_coordinates() == old.y_coordinates()
            {
                1.0
            } else {
                0.0
            },
        ],
    );
    cog_case(
        &cog,
        &f.cog.cases[0].pixels,
        "after-invalid-coordinates",
        results,
    )?;

    let origins: Vec<_> = f.shack.origins.iter().map(|p| (p[0], p[1])).collect();
    let mut shack = ShackHartmannImagePlan::from_detector_regions(
        (f.image.rows, f.image.columns),
        (f.shack.region_rows, f.shack.region_columns),
        &origins,
        f.shack.coordinate_scale,
        f.shack.pixel_threshold,
        f.shack.flux_threshold,
        Some(f.shack.reference_slopes),
        Some(f.shack.active),
    )?;
    let mut workspace = shack.workspace();
    let count = origins.len();
    macro_rules! shack_case {
        ($name:expr, $image:expr) => {{
            let mut slopes = vec![71.0; 2 * count];
            let mut flux = vec![71.0; count];
            let mut valid = vec![true; count];
            shack.process(
                ShackHartmannOutput::new(&mut slopes, &mut flux, &mut valid)?,
                &mut workspace,
                $image,
            )?;
            slopes.extend(flux);
            emit(results, "sensing-shack-f32", $name, &slopes, &valid);
        }};
    }
    shack_case!("bright", &f.image.pixels);
    shack_case!("low-flux", &vec![0.0; f.image.pixels.len()]);
    for (name, value) in [
        ("nan", f32::NAN),
        ("positive-inf", f32::INFINITY),
        ("overflow", f32::MAX),
    ] {
        let mut image = f.image.pixels.clone();
        image[0] = value;
        image[1] = value;
        shack_case!(name, &image);
    }
    shack = shack.with_active(&f.shack.replacement_active)?;
    shack_case!("active-replaced", &f.image.pixels);
    shack = shack.with_u32_subaperture_origins(
        &f.shack
            .replacement_origins
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
    )?;
    shack_case!("origins-replaced", &f.image.pixels);
    workspace = shack.workspace();
    shack_case!("reset", &f.image.pixels);

    let mut selected = PixelSelectionF32::prepare(PixelSelectionF32Configuration {
        mask: f.selection.mask.clone(),
        input_schema: "pipewireao/calibrated-pixels-f32/1".into(),
        output_schema: "org.example.parity.selected/1".into(),
        image_size: (f.image.rows, f.image.columns),
    })?;
    let mut output = vec![71.0; f.selection.mask.iter().filter(|&&v| v).count()];
    PixelSelectionF32::process(&mut selected, &f.image.pixels, &mut output)?;
    emit(results, "sensing-selection-f32", "initial", &output, &[]);
    selected
        .replace_parameter(
            2,
            AlgorithmInput::from(f.selection.replacement_mask.as_slice()),
        )
        .ok_or("missing mask")??;
    PixelSelectionF32::process(&mut selected, &f.image.pixels, &mut output)?;
    emit(results, "sensing-selection-f32", "replaced", &output, &[]);
    let invalid = vec![false; f.selection.mask.len()];
    let rejected = selected
        .replace_parameter(2, AlgorithmInput::from(invalid.as_slice()))
        .ok_or("missing mask")?
        .is_err();
    record(
        results,
        "sensing-selection-f32",
        "invalid-mask-status".into(),
        vec![if rejected { 0.0 } else { 1.0 }],
    );
    PixelSelectionF32::process(&mut selected, &f.image.pixels, &mut output)?;
    emit(
        results,
        "sensing-selection-f32",
        "after-invalid-mask",
        &output,
        &[],
    );
    let mut unusual = f.image.pixels.clone();
    let indices: Vec<_> = f
        .selection
        .replacement_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| v.then_some(i))
        .collect();
    unusual[indices[0]] = f32::NAN;
    unusual[indices[1]] = f32::INFINITY;
    unusual[indices[2]] = f32::NEG_INFINITY;
    PixelSelectionF32::process(&mut selected, &unusual, &mut output)?;
    emit(results, "sensing-selection-f32", "ieee-copy", &output, &[]);

    let mut regions = RegionExtractionF32::prepare(RegionExtractionF32Configuration {
        origins: f.regions.origins.clone(),
        image_size: (f.image.rows, f.image.columns),
        region_size: (f.regions.region_rows, f.regions.region_columns),
    })?;
    let mut output =
        vec![71.0; f.regions.origins.len() * f.regions.region_rows * f.regions.region_columns];
    RegionExtractionF32::process(&mut regions, &f.image.pixels, &mut output)?;
    emit(results, "sensing-regions-f32", "initial", &output, &[]);
    let replacement: Vec<_> = f
        .regions
        .replacement_origins
        .iter()
        .flatten()
        .copied()
        .collect();
    regions
        .replace_parameter(2, AlgorithmInput::from(replacement.as_slice()))
        .ok_or("missing origins")??;
    RegionExtractionF32::process(&mut regions, &f.image.pixels, &mut output)?;
    emit(results, "sensing-regions-f32", "replaced", &output, &[]);
    let mut invalid = replacement.clone();
    invalid[0] = f.image.rows as u32;
    let rejected = regions
        .replace_parameter(2, AlgorithmInput::from(invalid.as_slice()))
        .ok_or("missing origins")?
        .is_err();
    record(
        results,
        "sensing-regions-f32",
        "invalid-origins-status".into(),
        vec![if rejected { 0.0 } else { 1.0 }],
    );
    RegionExtractionF32::process(&mut regions, &f.image.pixels, &mut output)?;
    emit(
        results,
        "sensing-regions-f32",
        "after-invalid-origins",
        &output,
        &[],
    );
    let mut unusual = f.image.pixels.clone();
    let start = f.regions.replacement_origins[0][0] as usize * f.image.columns
        + f.regions.replacement_origins[0][1] as usize;
    unusual[start] = f32::NAN;
    unusual[start + 1] = f32::INFINITY;
    unusual[start + 2] = f32::NEG_INFINITY;
    RegionExtractionF32::process(&mut regions, &unusual, &mut output)?;
    emit(results, "sensing-regions-f32", "ieee-copy", &output, &[]);
    Ok(())
}
