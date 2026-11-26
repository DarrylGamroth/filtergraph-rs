//! Full declaration execution for the same-schema vector operations.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, DeclarationProcessError, VectorLinearCombinationError as E,
    VectorLinearCombinationF32 as C, VectorLinearCombinationF32Configuration, VectorOffsetF32 as O,
    VectorOffsetF32Configuration,
};
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    left_scale: f32,
    right_scale: f32,
    left: Vec<f32>,
    right: Vec<f32>,
    offset: Vec<f32>,
    sentinel: f32,
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let n = f.left.len();
    let mut c = C::prepare(VectorLinearCombinationF32Configuration {
        extent: n,
        left_scale: f.left_scale,
        right_scale: f.right_scale,
        schema: "org.example.parity.vector/1".into(),
    })?;
    let mut output = vec![f.sentinel; n];
    C::process(&mut c, &f.left, &f.right, &mut output)?;
    record(
        results,
        "vector-combination-f32",
        "finite".into(),
        output.iter().map(|&v| v as f64).collect(),
    );
    for (case, value) in [
        ("nan", f32::NAN),
        ("positive-inf", f32::INFINITY),
        ("negative-inf", f32::NEG_INFINITY),
        ("overflow", f32::MAX),
    ] {
        let mut left = f.left.clone();
        left[n - 1] = value;
        let mut right = f.right.clone();
        if case == "overflow" {
            right[n - 1] = -value;
        }
        output.fill(f.sentinel);
        let outcome = C::process(&mut c, &left, &right, &mut output);
        let reason = match outcome {
            Err(DeclarationProcessError::Algorithm(E::NonFiniteInput)) => 1,
            Err(DeclarationProcessError::Algorithm(E::NonFiniteOutput)) => 2,
            Ok(()) => 0,
            Err(e) => return Err(e.into()),
        };
        record(
            results,
            "vector-combination-f32",
            format!("{case}-status"),
            vec![
                reason as f64,
                output.iter().all(|&v| v == f.sentinel) as u8 as f64,
            ],
        );
        C::process(&mut c, &f.left, &f.right, &mut output)?;
        record(
            results,
            "vector-combination-f32",
            format!("after-{case}"),
            output.iter().map(|&v| v as f64).collect(),
        );
    }
    let mut o = O::prepare(VectorOffsetF32Configuration {
        extent: n,
        schema: "org.example.parity.vector/1".into(),
    })?;
    let mut current = vec![f.sentinel; n];
    for phase in ["initial", "replaced", "reset", "after-invalid"] {
        if phase == "replaced" {
            o.replace_parameter(3, AlgorithmInput::from(f.offset.as_slice()))
                .ok_or("missing offset")??;
        }
        if phase == "reset" {
            o.reset();
        }
        if phase == "after-invalid" {
            let mut invalid = f.offset.clone();
            invalid[n - 1] = f32::NAN;
            let rejected = o
                .replace_parameter(3, AlgorithmInput::from(invalid.as_slice()))
                .ok_or("missing offset")?
                .is_err();
            record(
                results,
                "vector-offset-f32",
                "invalid-replacement-status".into(),
                vec![rejected as u8 as f64],
            );
        }
        O::process(&mut o, &f.left, &mut output, &mut current)?;
        record(
            results,
            "vector-offset-f32",
            format!("{phase}-output"),
            output.iter().map(|&v| v as f64).collect(),
        );
        record(
            results,
            "vector-offset-f32",
            format!("{phase}-offset"),
            current.iter().map(|&v| v as f64).collect(),
        );
    }
    output.fill(f.sentinel);
    current.fill(f.sentinel);
    let mut invalid = f.left.clone();
    invalid[n - 1] = f32::INFINITY;
    let rejected = matches!(
        O::process(&mut o, &invalid, &mut output, &mut current),
        Err(DeclarationProcessError::Algorithm(E::NonFiniteInput))
    );
    record(
        results,
        "vector-offset-f32",
        "nonfinite-status".into(),
        vec![
            rejected as u8 as f64,
            output.iter().chain(&current).all(|&v| v == f.sentinel) as u8 as f64,
        ],
    );
    Ok(())
}
