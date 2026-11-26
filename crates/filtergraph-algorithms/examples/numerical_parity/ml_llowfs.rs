//! Live fixed-model inference and declared parameter lifecycle.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, DeclarationProcessError, ML_LLOWFS_MODEL, ML_LLOWFS_PARAMETER_COUNT,
    MlLlowfsError as E, MlLlowfsF32 as D, MlLlowfsF32Configuration,
};
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    parameter_period: usize,
    parameter_shift: i32,
    parameter_scale: f32,
    sentinel: f32,
    image_seeds: Vec<usize>,
}
fn reason(outcome: Result<(), DeclarationProcessError<E>>) -> Result<u8, Box<dyn Error>> {
    Ok(match outcome {
        Ok(()) => 0,
        Err(DeclarationProcessError::Algorithm(E::ParametersMissing)) => 1,
        Err(DeclarationProcessError::Algorithm(E::NonFiniteInput)) => 2,
        Err(DeclarationProcessError::Algorithm(E::InvalidFlux)) => 3,
        Err(DeclarationProcessError::Algorithm(E::InvalidStandardDeviation)) => 4,
        Err(DeclarationProcessError::Algorithm(E::NonFiniteOutput)) => 5,
        Err(e) => return Err(e.into()),
    })
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let mut prepared = D::prepare(MlLlowfsF32Configuration {
        model: ML_LLOWFS_MODEL.into(),
    })?;
    let parameters: Vec<_> = (0..ML_LLOWFS_PARAMETER_COUNT)
        .map(|i| ((i % f.parameter_period) as i32 - f.parameter_shift) as f32 * f.parameter_scale)
        .collect();
    let image = |seed| {
        (0..400)
            .map(|i| (1 + (i * 7 + i / 20 * 3 + seed) % 97) as f32)
            .collect::<Vec<_>>()
    };
    let mut output = vec![f.sentinel; 14];
    let r = reason(D::process(&mut prepared, &image(0), &mut output))?;
    record(
        results,
        "ml-llowfs-f32",
        "missing-status".into(),
        vec![
            r as f64,
            output.iter().all(|v| *v == f.sentinel) as u8 as f64,
        ],
    );
    prepared
        .replace_parameter(2, AlgorithmInput::from(parameters.as_slice()))
        .ok_or("parameter missing")??;
    for &seed in &f.image_seeds {
        D::process(&mut prepared, &image(seed), &mut output)?;
        record(
            results,
            "ml-llowfs-f32",
            format!("image-{seed}"),
            output.iter().map(|v| *v as f64).collect(),
        );
    }
    for (name, values) in [
        ("nan", {
            let mut v = image(0);
            v[399] = f32::NAN;
            v
        }),
        ("infinity", {
            let mut v = image(0);
            v[399] = f32::INFINITY;
            v
        }),
        ("zero-flux", vec![0.0; 400]),
        ("negative-flux", vec![-1.0; 400]),
        ("overflow-flux", vec![f32::MAX; 400]),
        ("constant", vec![1.0; 400]),
    ] {
        output.fill(f.sentinel);
        let r = reason(D::process(&mut prepared, &values, &mut output))?;
        record(
            results,
            "ml-llowfs-f32",
            format!("{name}-status"),
            vec![
                r as f64,
                output.iter().all(|v| *v == f.sentinel) as u8 as f64,
            ],
        );
        D::process(&mut prepared, &image(0), &mut output)?;
        record(
            results,
            "ml-llowfs-f32",
            format!("after-{name}"),
            output.iter().map(|v| *v as f64).collect(),
        );
    }
    let mut bad = parameters.clone();
    bad[42] = f32::NAN;
    let rejected = prepared
        .replace_parameter(2, AlgorithmInput::from(bad.as_slice()))
        .ok_or("parameter missing")?
        .is_err();
    record(
        results,
        "ml-llowfs-f32",
        "invalid-replacement-status".into(),
        vec![rejected as u8 as f64],
    );
    prepared.reset();
    D::process(&mut prepared, &image(0), &mut output)?;
    record(
        results,
        "ml-llowfs-f32",
        "after-reset".into(),
        output.iter().map(|v| *v as f64).collect(),
    );
    let zeros = vec![0.0f32; ML_LLOWFS_PARAMETER_COUNT];
    prepared
        .replace_parameter(2, AlgorithmInput::from(zeros.as_slice()))
        .ok_or("parameter missing")??;
    let mut cancellation = vec![1.0; 400];
    cancellation[0] = 16777216.0;
    cancellation[399] = -16777216.0;
    output.fill(f.sentinel);
    let r = reason(D::process(&mut prepared, &cancellation, &mut output))?;
    record(
        results,
        "ml-llowfs-f32",
        "cancellation-status".into(),
        vec![r as f64],
    );
    record(
        results,
        "ml-llowfs-f32",
        "cancellation-output".into(),
        output.iter().map(|v| *v as f64).collect(),
    );
    let bad = vec![f32::MAX; ML_LLOWFS_PARAMETER_COUNT];
    prepared
        .replace_parameter(2, AlgorithmInput::from(bad.as_slice()))
        .ok_or("parameter missing")??;
    output.fill(f.sentinel);
    let r = reason(D::process(&mut prepared, &image(0), &mut output))?;
    record(
        results,
        "ml-llowfs-f32",
        "overflow-model-status".into(),
        vec![
            r as f64,
            output.iter().all(|v| *v == f.sentinel) as u8 as f64,
        ],
    );
    Ok(())
}
