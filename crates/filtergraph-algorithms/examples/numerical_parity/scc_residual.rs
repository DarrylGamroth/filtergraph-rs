//! Live SCC selected-residual and controller reconstruction public declarations.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, DeclarationProcessError, GemvError, SccPixelResidualError as E,
    SccPixelResidualF32 as D, SccPixelResidualF32Configuration as C, SccReconstructorF32 as R,
    SccReconstructorF32Configuration as RC,
};
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    residual: Residual,
    reconstruction: Reconstruction,
}
#[derive(Deserialize)]
struct Residual {
    reference: Vec<f32>,
    mask: Vec<bool>,
    minimum_valid_fraction: f32,
    difference: Vec<f32>,
    replacement_reference: Vec<f32>,
    replacement_mask: Vec<bool>,
    selected_coordinate: usize,
    unselected_coordinate: usize,
    sentinel: f32,
    fraction_sentinel: f32,
    rounding_mask: Vec<bool>,
    rounding_fraction: f32,
    rounding_validity: Vec<bool>,
}
#[derive(Deserialize)]
struct Reconstruction {
    controller_count: usize,
    selected_count: usize,
    matrix: Vec<f32>,
    replacement_matrix: Vec<f32>,
    input: Vec<f32>,
    sentinel: f32,
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    let c = &f.residual;
    let n = c.reference.len();
    let label = "scc-pixel-residual-f32";
    let configuration = |reference, mask, fraction| C {
        pixel_count: n,
        mask,
        initial_reference_image: reference,
        minimum_valid_fraction: fraction,
    };
    let mut p = D::prepare(configuration(
        Some(c.reference.clone()),
        c.mask.clone(),
        c.minimum_valid_fraction,
    ))?;
    macro_rules! residual {
        ($prepared:expr,$name:expr,$difference:expr,$validity:expr,$status:expr) => {{
            let prepared = $prepared;
            let count = prepared.plan().selected_count();
            let mut residuals = vec![c.sentinel; count];
            let mut validity = vec![true; count];
            let mut fraction = [c.fraction_sentinel];
            let code = match D::process(
                prepared,
                $difference,
                $validity,
                &[$status],
                &mut residuals,
                &mut validity,
                &mut fraction,
            ) {
                Ok(_) => 0,
                Err(DeclarationProcessError::Algorithm(E::IncompletePair)) => 1,
                Err(DeclarationProcessError::Algorithm(E::InvalidPairStatus)) => 2,
                Err(DeclarationProcessError::Algorithm(E::InsufficientValidPixels)) => 3,
                Err(error) => return Err(error.into()),
            };
            record(
                results,
                label,
                format!("{}-residuals", $name),
                residuals.iter().map(|&x| x as f64).collect(),
            );
            record(
                results,
                label,
                format!("{}-validity-status", $name),
                validity.iter().map(|&x| x as u8 as f64).collect(),
            );
            record(
                results,
                label,
                format!("{}-fraction", $name),
                vec![fraction[0] as f64],
            );
            record(
                results,
                label,
                format!("{}-status", $name),
                vec![
                    code as f64,
                    prepared.plan().minimum_valid_count() as f64,
                    (residuals.iter().all(|&x| x == c.sentinel)
                        && validity.iter().all(|&x| x)
                        && fraction == [c.fraction_sentinel]) as u8 as f64,
                ],
            );
        }};
    }
    let all_valid = vec![true; n];
    let mut default = D::prepare(configuration(
        None,
        c.mask.clone(),
        c.minimum_valid_fraction,
    ))?;
    residual!(
        &mut default,
        "default-reference",
        &c.difference,
        &all_valid,
        1
    );
    residual!(&mut p, "initial", &c.difference, &all_valid, 1);
    for (name, status) in [("priming", 0), ("undefined", 2), ("undefined-high", 255)] {
        residual!(&mut p, name, &c.difference, &all_valid, status);
    }
    let selected = p.plan().selected_indices().to_vec();
    let mut validity = all_valid.clone();
    validity[selected[0]] = false;
    residual!(&mut p, "fraction-equal", &c.difference, &validity, 1);
    validity[selected[1]] = false;
    residual!(&mut p, "fraction-below", &c.difference, &validity, 1);
    for (name, value) in [
        ("nan", f32::NAN),
        ("positive-inf", f32::INFINITY),
        ("negative-inf", f32::NEG_INFINITY),
    ] {
        let mut difference = c.difference.clone();
        difference[c.selected_coordinate] = value;
        residual!(&mut p, name, &difference, &all_valid, 1);
        residual!(
            &mut p,
            format!("after-{name}"),
            &c.difference,
            &all_valid,
            1
        );
    }
    let mut difference = c.difference.clone();
    difference[c.unselected_coordinate] = f32::NAN;
    residual!(&mut p, "unselected-nan", &difference, &all_valid, 1);
    let mut reference = c.reference.clone();
    reference[c.selected_coordinate] = -f32::MAX;
    p.replace_parameter(6, AlgorithmInput::from(reference.as_slice()))
        .ok_or("reference missing")??;
    let mut difference = c.difference.clone();
    difference[c.selected_coordinate] = f32::MAX;
    residual!(&mut p, "subtraction-overflow", &difference, &all_valid, 1);
    p.replace_parameter(6, AlgorithmInput::from(c.replacement_reference.as_slice()))
        .ok_or("reference missing")??;
    residual!(&mut p, "replaced-reference", &c.difference, &all_valid, 1);
    for (name, values) in [
        ("nan", vec![f32::NAN; n]),
        ("positive-inf", vec![f32::INFINITY; n]),
        ("extent", vec![0.0; n - 1]),
    ] {
        let rejected = p
            .replace_parameter(6, AlgorithmInput::from(values.as_slice()))
            .ok_or("reference missing")?
            .is_err();
        record(
            results,
            label,
            format!("invalid-reference-{name}-status"),
            vec![rejected as u8 as f64],
        );
        residual!(
            &mut p,
            format!("after-invalid-reference-{name}"),
            &c.difference,
            &all_valid,
            1
        );
    }
    for (name, mask) in [
        ("count", vec![true; n]),
        ("empty", vec![false; n]),
        ("extent", vec![true; n - 1]),
    ] {
        let rejected = p
            .replace_parameter(7, AlgorithmInput::from(mask.as_slice()))
            .ok_or("mask missing")?
            .is_err();
        record(
            results,
            label,
            format!("invalid-mask-{name}-status"),
            vec![rejected as u8 as f64],
        );
        residual!(
            &mut p,
            format!("after-invalid-mask-{name}"),
            &c.difference,
            &all_valid,
            1
        );
    }
    p.replace_parameter(7, AlgorithmInput::from(c.replacement_mask.as_slice()))
        .ok_or("mask missing")??;
    residual!(&mut p, "replaced-mask", &c.difference, &all_valid, 1);
    p.reset();
    residual!(&mut p, "after-reset", &c.difference, &all_valid, 1);
    let mut rounding = D::prepare(configuration(
        Some(c.reference.clone()),
        c.rounding_mask.clone(),
        c.rounding_fraction,
    ))?;
    residual!(
        &mut rounding,
        "fraction-rounding",
        &c.difference,
        &c.rounding_validity,
        1
    );
    let c = &f.reconstruction;
    let label = "scc-reconstructor-f32";
    let m = c.controller_count;
    let n = c.selected_count;
    let configuration = |matrix| RC {
        controller_coordinate_count: m,
        selected_pixel_count: n,
        initial_reconstructor: matrix,
    };
    let mut p = R::prepare(configuration(Some(c.matrix.clone())))?;
    macro_rules! reconstruct {
        ($prepared:expr,$name:expr,$input:expr) => {{
            let mut output = vec![c.sentinel; m];
            let code = match R::process($prepared, $input, &mut output) {
                Ok(_) => 0,
                Err(DeclarationProcessError::Algorithm(GemvError::NonFiniteInput)) => 1,
                Err(DeclarationProcessError::Algorithm(GemvError::NonFiniteOutput)) => 2,
                Err(error) => return Err(error.into()),
            };
            let mut status = vec![code as f64];
            status.extend(output.iter().map(|&x| {
                if x.is_nan() {
                    1.0
                } else if x == f32::INFINITY {
                    2.0
                } else if x == f32::NEG_INFINITY {
                    3.0
                } else {
                    0.0
                }
            }));
            status.push(output.iter().all(|&x| x == c.sentinel) as u8 as f64);
            record(results, label, format!("{}-status", $name), status);
            record(
                results,
                label,
                format!("{}-output", $name),
                output
                    .iter()
                    .map(|&x| if x.is_finite() { x as f64 } else { 0.0 })
                    .collect(),
            );
        }};
    }
    let mut default = R::prepare(configuration(None))?;
    reconstruct!(&mut default, "default", &c.input);
    reconstruct!(&mut p, "initial", &c.input);
    for (name, value) in [
        ("nan", f32::NAN),
        ("positive-inf", f32::INFINITY),
        ("negative-inf", f32::NEG_INFINITY),
    ] {
        let mut input = c.input.clone();
        input[n - 1] = value;
        reconstruct!(&mut p, name, &input);
        reconstruct!(&mut p, format!("after-{name}"), &c.input);
    }
    p.replace_parameter(2, AlgorithmInput::from(c.replacement_matrix.as_slice()))
        .ok_or("reconstructor missing")??;
    reconstruct!(&mut p, "replaced", &c.input);
    for (name, values) in [
        ("nan", vec![f32::NAN; m * n]),
        ("positive-inf", vec![f32::INFINITY; m * n]),
        ("extent", vec![0.0; m * n - 1]),
    ] {
        let rejected = p
            .replace_parameter(2, AlgorithmInput::from(values.as_slice()))
            .ok_or("reconstructor missing")?
            .is_err();
        record(
            results,
            label,
            format!("invalid-matrix-{name}-status"),
            vec![rejected as u8 as f64],
        );
        reconstruct!(&mut p, format!("after-invalid-matrix-{name}"), &c.input);
    }
    let mut overflow = c.replacement_matrix.clone();
    overflow[n - 1] = f32::MAX;
    overflow[(m - 1) * n + n - 1] = -f32::MAX;
    p.replace_parameter(2, AlgorithmInput::from(overflow.as_slice()))
        .ok_or("reconstructor missing")??;
    let mut input = c.input.clone();
    input[n - 1] = 2.0;
    reconstruct!(&mut p, "output-overflow", &input);
    p.replace_parameter(2, AlgorithmInput::from(c.replacement_matrix.as_slice()))
        .ok_or("reconstructor missing")??;
    reconstruct!(&mut p, "after-overflow", &c.input);
    p.reset();
    reconstruct!(&mut p, "after-reset", &c.input);
    Ok(())
}
