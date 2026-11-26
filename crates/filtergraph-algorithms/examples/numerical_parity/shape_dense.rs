//! Declared direct-selector offsets and explicitly ordered dense reconstruction.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, DeclarationProcessError, DenseVectorReconstructorF32 as D,
    DenseVectorReconstructorF32Configuration, GemvError, VdmShapeOffsetF32 as S,
    VdmShapeOffsetF32Configuration, VectorLinearCombinationError as E,
};
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    active_count: usize,
    full_count: usize,
    physical_count: usize,
    completion: Vec<f32>,
    replacement_completion: Vec<f32>,
    inverse: Vec<f32>,
    offset: Vec<f32>,
    requested: Vec<f32>,
    sentinel: f32,
    dense_input: Vec<f32>,
    dense_matrix: Vec<f32>,
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let mut s = S::prepare(VdmShapeOffsetF32Configuration {
        active_vdm_count: f.active_count,
        full_vdm_count: f.full_count,
        physical_actuator_count: f.physical_count,
    })?;
    let mut out = vec![f.sentinel; f.active_count];
    let mut current = vec![f.sentinel; f.physical_count];
    for phase in [
        "neutral",
        "inverse",
        "completion",
        "shape",
        "changed-selector",
        "reset",
        "after-invalid",
    ] {
        match phase {
            "inverse" => {
                s.replace_parameter(5, AlgorithmInput::from(f.inverse.as_slice()))
                    .ok_or("inverse absent")??;
            }
            "completion" => {
                s.replace_parameter(4, AlgorithmInput::from(f.completion.as_slice()))
                    .ok_or("completion absent")??;
            }
            "shape" => {
                s.replace_parameter(3, AlgorithmInput::from(f.offset.as_slice()))
                    .ok_or("shape absent")??;
            }
            "changed-selector" => {
                s.replace_parameter(4, AlgorithmInput::from(f.replacement_completion.as_slice()))
                    .ok_or("completion absent")??;
            }
            "reset" => s.reset(),
            "after-invalid" => {
                for (name, port, values) in [
                    ("selector", 4, vec![0.0; f.completion.len()]),
                    ("shape", 3, vec![f32::NAN; f.offset.len()]),
                    ("inverse", 5, vec![f32::INFINITY; f.inverse.len()]),
                ] {
                    let rejected = s
                        .replace_parameter(port, AlgorithmInput::from(values.as_slice()))
                        .ok_or("parameter absent")?
                        .is_err();
                    record(
                        results,
                        "vdm-shape-offset-f32",
                        format!("invalid-{name}-status"),
                        vec![rejected as u8 as f64],
                    );
                }
            }
            _ => {}
        }
        S::process(&mut s, &f.requested, &mut out, &mut current)?;
        record(
            results,
            "vdm-shape-offset-f32",
            format!("{phase}-output"),
            out.iter().map(|v| *v as f64).collect(),
        );
        record(
            results,
            "vdm-shape-offset-f32",
            format!("{phase}-current"),
            current.iter().map(|v| *v as f64).collect(),
        );
        if phase == "neutral" {
            record(
                results,
                "vdm-shape-offset-f32",
                "neutral-negative-zero-status".into(),
                vec![(out[0].to_bits() == (-0.0f32).to_bits()) as u8 as f64],
            );
        }
    }
    for (name, values) in [
        ("nan", vec![f32::NAN; f.active_count]),
        ("infinity", vec![f32::INFINITY; f.active_count]),
    ] {
        out.fill(f.sentinel);
        current.fill(f.sentinel);
        let r = match S::process(&mut s, &values, &mut out, &mut current) {
            Err(DeclarationProcessError::Algorithm(E::NonFiniteInput)) => 1,
            Ok(()) => 0,
            Err(e) => return Err(e.into()),
        };
        record(
            results,
            "vdm-shape-offset-f32",
            format!("{name}-status"),
            vec![
                r as f64,
                out.iter().chain(&current).all(|v| *v == f.sentinel) as u8 as f64,
            ],
        );
    }
    let mut d = D::prepare(DenseVectorReconstructorF32Configuration {
        input_count: f.dense_input.len(),
        output_count: 2,
        initial_reconstructor: None,
        input_schema: "org.example.parity.ordered/1".into(),
        output_schema: "org.example.parity.result/1".into(),
        reconstructor_schema: "org.example.parity.matrix/1".into(),
    })?;
    let mut out = vec![f.sentinel; 2];
    for phase in ["neutral", "adopted", "reset", "after-invalid"] {
        if phase == "adopted" {
            d.replace_parameter(2, AlgorithmInput::from(f.dense_matrix.as_slice()))
                .ok_or("matrix absent")??;
        }
        if phase == "reset" {
            d.reset();
        }
        if phase == "after-invalid" {
            let bad = vec![f32::NAN; f.dense_matrix.len()];
            let rejected = d
                .replace_parameter(2, AlgorithmInput::from(bad.as_slice()))
                .ok_or("matrix absent")?
                .is_err();
            record(
                results,
                "dense-vector-reconstructor-f32",
                "invalid-replacement-status".into(),
                vec![rejected as u8 as f64],
            );
        }
        D::process(&mut d, &f.dense_input, &mut out)?;
        record(
            results,
            "dense-vector-reconstructor-f32",
            phase.into(),
            out.iter().map(|v| *v as f64).collect(),
        );
    }
    for (name, value) in [
        ("nan", f32::NAN),
        ("infinity", f32::INFINITY),
        ("overflow", f32::MAX),
    ] {
        let mut input = f.dense_input.clone();
        let n = input.len();
        input[n - 1] = value;
        out.fill(f.sentinel);
        let r = match D::process(&mut d, &input, &mut out) {
            Ok(()) => 0,
            Err(DeclarationProcessError::Algorithm(GemvError::NonFiniteInput)) => 1,
            Err(DeclarationProcessError::Algorithm(GemvError::NonFiniteOutput)) => 2,
            Err(e) => return Err(e.into()),
        };
        record(
            results,
            "dense-vector-reconstructor-f32",
            format!("{name}-status"),
            vec![r as f64, out.iter().all(|v| *v == f.sentinel) as u8 as f64],
        );
        record(
            results,
            "dense-vector-reconstructor-f32",
            format!("{name}-classes-status"),
            out.iter()
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
                .collect(),
        );
        D::process(&mut d, &f.dense_input, &mut out)?;
        record(
            results,
            "dense-vector-reconstructor-f32",
            format!("after-{name}"),
            out.iter().map(|v| *v as f64).collect(),
        );
    }
    Ok(())
}
