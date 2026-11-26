//! Real signed-pixel declaration execution in native logical row-major storage.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, DeclarationProcessError, PixelCalibrationF32 as P,
    PixelCalibrationF32Configuration as Configuration, PixelCalibrationF32Error as E,
};
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    image_size: (usize, usize),
    input_schema: String,
    flat: Vec<f32>,
    background: Vec<f32>,
    replacement_flat: Vec<f32>,
    replacement_background: Vec<f32>,
    raw: Vec<f32>,
    sentinel: f32,
    row_first_coordinate: usize,
    column_first_coordinate: usize,
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let label = "pixel-calibration-f32";
    let n = f.raw.len();
    let configuration = |flat, background| Configuration {
        image_size: f.image_size,
        input_schema: f.input_schema.clone(),
        initial_flat: flat,
        initial_background: background,
    };
    let mut output = vec![f.sentinel; n];
    let mut default = P::prepare(configuration(None, None))?;
    P::process(&mut default, &f.raw, &mut output)?;
    record(
        results,
        label,
        "default-output".into(),
        output.iter().map(|&x| x as f64).collect(),
    );
    let mut p = P::prepare(configuration(
        Some(f.flat.clone()),
        Some(f.background.clone()),
    ))?;
    P::process(&mut p, &f.raw, &mut output)?;
    record(
        results,
        label,
        "initial-output".into(),
        output.iter().map(|&x| x as f64).collect(),
    );
    for name in [
        "nan",
        "positive-inf",
        "negative-inf",
        "overflow",
        "mixed-output-first",
        "mixed-input-first",
    ] {
        let mut raw = f.raw.clone();
        match name {
            "nan" => raw[n - 1] = f32::NAN,
            "positive-inf" => raw[n - 1] = f32::INFINITY,
            "negative-inf" => raw[n - 1] = f32::NEG_INFINITY,
            "overflow" => raw[n - 1] = f32::MAX,
            "mixed-output-first" => {
                raw[f.column_first_coordinate] = f32::MAX;
                raw[f.row_first_coordinate] = f32::NAN;
            }
            _ => {
                raw[f.column_first_coordinate] = f32::NAN;
                raw[f.row_first_coordinate] = f32::MAX;
            }
        }
        output.fill(f.sentinel);
        let code = match P::process(&mut p, &raw, &mut output) {
            Ok(()) => 0,
            Err(DeclarationProcessError::Algorithm(E::NonFiniteInput)) => 1,
            Err(DeclarationProcessError::Algorithm(E::NonFiniteOutput)) => 2,
            Err(error) => return Err(error.into()),
        };
        record(
            results,
            label,
            format!("{name}-status"),
            vec![
                code as f64,
                output.iter().all(|&x| x == f.sentinel) as u8 as f64,
            ],
        );
        P::process(&mut p, &f.raw, &mut output)?;
        record(
            results,
            label,
            format!("after-{name}-output"),
            output.iter().map(|&x| x as f64).collect(),
        );
    }
    for name in ["input-extent", "output-extent"] {
        output.fill(f.sentinel);
        let rejected = if name == "input-extent" {
            P::process(&mut p, &f.raw[..n - 1], &mut output).is_err()
        } else {
            P::process(&mut p, &f.raw, &mut output[..n - 1]).is_err()
        };
        record(
            results,
            label,
            format!("{name}-status"),
            vec![
                rejected as u8 as f64,
                output.iter().all(|&x| x == f.sentinel) as u8 as f64,
            ],
        );
    }
    for (name, port, replacement) in [
        ("flat", 2, &f.replacement_flat),
        ("background", 3, &f.replacement_background),
    ] {
        p.replace_parameter(port, AlgorithmInput::from(replacement.as_slice()))
            .ok_or("parameter missing")??;
        P::process(&mut p, &f.raw, &mut output)?;
        record(
            results,
            label,
            format!("replaced-{name}-output"),
            output.iter().map(|&x| x as f64).collect(),
        );
        for (invalid, value) in [
            ("nan", f32::NAN),
            ("positive-inf", f32::INFINITY),
            ("negative-inf", f32::NEG_INFINITY),
        ] {
            let mut values = replacement.clone();
            values[n - 1] = value;
            let rejected = p
                .replace_parameter(port, AlgorithmInput::from(values.as_slice()))
                .ok_or("parameter missing")?
                .is_err();
            record(
                results,
                label,
                format!("invalid-{name}-{invalid}-status"),
                vec![rejected as u8 as f64],
            );
            P::process(&mut p, &f.raw, &mut output)?;
            record(
                results,
                label,
                format!("after-invalid-{name}-{invalid}-output"),
                output.iter().map(|&x| x as f64).collect(),
            );
        }
        let rejected = p
            .replace_parameter(port, AlgorithmInput::from(&replacement[..n - 1]))
            .ok_or("parameter missing")?
            .is_err();
        record(
            results,
            label,
            format!("invalid-{name}-extent-status"),
            vec![rejected as u8 as f64],
        );
        P::process(&mut p, &f.raw, &mut output)?;
        record(
            results,
            label,
            format!("after-invalid-{name}-extent-output"),
            output.iter().map(|&x| x as f64).collect(),
        );
    }
    p.reset();
    P::process(&mut p, &f.raw, &mut output)?;
    record(
        results,
        label,
        "after-reset-output".into(),
        output.iter().map(|&x| x as f64).collect(),
    );
    for name in [
        "empty-rows",
        "empty-columns",
        "byte-extent",
        "empty-schema",
        "blank-schema",
        "flat-nan",
        "background-inf",
        "flat-extent",
        "background-extent",
    ] {
        let mut c = configuration(Some(f.flat.clone()), Some(f.background.clone()));
        match name {
            "empty-rows" => c.image_size.0 = 0,
            "empty-columns" => c.image_size.1 = 0,
            "byte-extent" => c.image_size = (u32::MAX as usize / 4 + 1, 1),
            "empty-schema" => c.input_schema.clear(),
            "blank-schema" => c.input_schema = "  ".into(),
            "flat-nan" => c.initial_flat.as_mut().unwrap()[n - 1] = f32::NAN,
            "background-inf" => c.initial_background.as_mut().unwrap()[n - 1] = f32::INFINITY,
            "flat-extent" => {
                c.initial_flat.as_mut().unwrap().pop();
            }
            _ => {
                c.initial_background.as_mut().unwrap().pop();
            }
        }
        record(
            results,
            label,
            format!("configuration-{name}-status"),
            vec![P::prepare(c).is_err() as u8 as f64],
        );
    }
    Ok(())
}
