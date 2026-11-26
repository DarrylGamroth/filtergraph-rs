//! Differential records for detector frontends and composed reconstruction.
use super::{Record, record};
use filtergraph_algorithms::*;
use serde::Deserialize;
#[derive(Deserialize)]
struct Fixture {
    incremental: Incremental,
    detector: Detector,
    complete: Complete,
    pol: Pol,
}
#[derive(Deserialize)]
struct Incremental {
    measurements: usize,
    capacity: usize,
    blocks: usize,
    rows: usize,
    matrix: Vec<f32>,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Step {
    name: String,
    sequence: u64,
    offset: u32,
    #[serde(default)]
    values: Vec<f32>,
    #[serde(default)]
    columns: Vec<u32>,
    #[serde(default)]
    count: u32,
    #[serde(default)]
    action: String,
    #[serde(default)]
    missing: bool,
    #[serde(default)]
    replacement: Option<Vec<f32>>,
    #[serde(default)]
    terminal: bool,
    #[serde(default)]
    discontinuity: bool,
    #[serde(default)]
    corrupted: bool,
}
fn emit(
    results: &mut Vec<Record>,
    label: &str,
    name: &str,
    outcome: Option<AlgorithmProcessResult>,
    values: &[f32],
) {
    let mut status = vec![
        f64::from(outcome.is_none()),
        f64::from(outcome.is_some_and(|r| r.is_complete())),
    ];
    if let Some(m) = outcome.and_then(|r| r.metadata()) {
        status.extend([
            m.sequence() as f64,
            m.offset() as f64,
            f64::from(m.is_terminal()),
            f64::from(m.is_discontinuous()),
            f64::from(m.is_corrupted()),
            m.timestamp_nanoseconds().unwrap_or(-1) as f64,
        ]);
    } else {
        status.extend([-1.0; 6]);
    }
    status.extend(values.iter().map(|v| {
        if v.is_nan() {
            1.0
        } else if *v == f32::INFINITY {
            2.0
        } else if *v == f32::NEG_INFINITY {
            3.0
        } else {
            0.0
        }
    }));
    record(results, label, format!("{name}-status"), status);
    record(
        results,
        label,
        format!("{name}-values"),
        values
            .iter()
            .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
            .collect(),
    );
}
fn sample(s: &Step) -> SampleMetadata {
    SampleMetadata::new(s.sequence, s.offset)
        .with_terminal(s.terminal)
        .with_discontinuity(s.discontinuity)
        .with_corrupted(s.corrupted)
        .with_timestamp_nanoseconds(Some(1234))
}
pub(super) fn run(
    source: &str,
    results: &mut Vec<Record>,
) -> Result<(), Box<dyn std::error::Error>> {
    let f: Fixture = toml::from_str(source)?;
    let c = &f.incremental;
    let mut p =
        IncrementalDenseReconstructorF32::prepare(IncrementalDenseReconstructorF32Configuration {
            measurement_count: c.measurements,
            measurement_block_capacity: c.capacity,
            blocks_per_sample: c.blocks,
            reconstructed_count: c.rows,
            initial_reconstructor: Some(c.matrix.clone()),
            measurements_schema: "test.frontend/measurements/1".into(),
            column_indices_schema: "test.frontend/columns/1".into(),
            measurement_count_schema: "test.frontend/count/1".into(),
            reconstructed_schema: "test.frontend/reconstructed/1".into(),
            reconstructor_schema: "test.frontend/matrix/1".into(),
        })?;
    for s in &c.steps {
        if let Some(matrix) = &s.replacement {
            let rejected = p
                .replace_parameter(4, AlgorithmInput::from(matrix.as_slice()))
                .ok_or("missing incremental reconstructor")?
                .is_err();
            record(
                results,
                "frontend-incremental-f32",
                format!("{}-replacement-status", s.name),
                vec![f64::from(rejected)],
            );
        }
        if s.action == "reset" {
            p.reset();
        }
        let mut output = vec![-99.0; c.rows];
        let outcome = IncrementalDenseReconstructorF32::process_with_metadata(
            &mut p,
            &[(!s.missing).then_some(sample(s)), None, None],
            &s.values,
            &s.columns,
            &[s.count],
            &mut output,
        )
        .ok();
        emit(
            results,
            "frontend-incremental-f32",
            &s.name,
            outcome,
            &output,
        );
    }
    detector(&f.detector, results)?;
    complete(&f.complete, results)?;
    pol(&f.pol, results)?;
    Ok(())
}

#[derive(Deserialize)]
struct Detector {
    invalid_pixel_index: usize,
    rows: usize,
    columns: usize,
    block_rows: usize,
    region_lines: usize,
    block_lines: usize,
    region_samples: usize,
    detector_origins: Vec<[usize; 2]>,
    line_steps: Vec<[i32; 2]>,
    sample_steps: Vec<[i32; 2]>,
    raw: Vec<u16>,
    image: Vec<f32>,
    flat: Vec<f32>,
    background: Vec<f32>,
    replacement_flat: Vec<f32>,
    replacement_background: Vec<f32>,
    subaperture_origins: Vec<[usize; 2]>,
    reference_slopes: Vec<f32>,
    active: Vec<bool>,
    reconstructor_rows: usize,
    matrix: Vec<f32>,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Complete {
    rows: usize,
    subapertures: usize,
    matrix: Vec<f32>,
    slopes: Vec<f32>,
    replacement: Vec<f32>,
}
#[derive(Deserialize)]
struct Pol {
    measurements: usize,
    correctors: usize,
    rows: usize,
    interaction: Vec<f32>,
    reconstructor: Vec<f32>,
    corrector_to_error: Vec<f32>,
    residual: Vec<f32>,
    shape: Vec<f32>,
}
fn detector(c: &Detector, results: &mut Vec<Record>) -> Result<(), Box<dyn std::error::Error>> {
    let mapping = DetectorReadoutPlan::from_i32_coordinate_steps(
        (c.rows, c.columns),
        c.region_lines,
        c.block_lines,
        c.region_samples,
        &c.detector_origins,
        &c.line_steps,
        &c.sample_steps,
    )?;
    macro_rules! calibrate {
        ($declaration:ty,$configuration:expr,$label:literal,$readout:literal) => {{
            let mut p = <$declaration>::prepare($configuration)?;
            for s in &c.steps {
                if s.action == "reset" {
                    p.reset();
                }
                if matches!(
                    s.action.as_str(),
                    "flat" | "background" | "invalid-flat" | "overflow-flat"
                ) {
                    let mut replacement = if s.action == "background" {
                        c.replacement_background.clone()
                    } else {
                        c.replacement_flat.clone()
                    };
                    if s.action == "invalid-flat" {
                        replacement[0] = f32::NAN;
                    }
                    if s.action == "overflow-flat" {
                        replacement.fill(f32::MAX);
                    }
                    let rejected = p
                        .replace_parameter(
                            if s.action == "background" { 3 } else { 2 },
                            AlgorithmInput::from(replacement.as_slice()),
                        )
                        .ok_or("missing calibration port")?
                        .is_err();
                    record(
                        results,
                        $label,
                        format!("{}-replacement-status", s.name),
                        vec![f64::from(rejected)],
                    );
                }
                let ordinal = s.offset as usize / c.block_rows;
                let raw = if $readout {
                    mapping
                        .block_detector_indices(ordinal)?
                        .iter()
                        .map(|&i| c.raw[i])
                        .collect::<Vec<_>>()
                } else {
                    let first = s.offset as usize * c.columns;
                    c.raw[first..first + c.block_rows * c.columns].to_vec()
                };
                let metadata =
                    sample(s).with_offset(if $readout { ordinal as u32 } else { s.offset });
                let mut output = vec![-99.0; raw.len()];
                let outcome = <$declaration>::process_with_metadata(
                    &mut p,
                    &[(!s.missing).then_some(metadata)],
                    &raw,
                    &mut output,
                )
                .ok();
                emit(results, $label, &s.name, outcome, &output);
            }
        }};
    }
    calibrate!(
        PixelCalibrationRowU16F32,
        PixelCalibrationRowU16F32Configuration {
            row_block_rows: c.block_rows,
            initial_flat: Some(c.flat.clone()),
            initial_background: Some(c.background.clone()),
            image_size: (c.rows, c.columns),
        },
        "frontend-pixel-row-f32",
        false
    );
    calibrate!(
        PixelCalibrationReadoutU16F32,
        PixelCalibrationReadoutU16F32Configuration {
            region_lines: c.region_lines,
            block_lines: c.block_lines,
            region_samples: c.region_samples,
            detector_origins: c.detector_origins.clone(),
            detector_line_steps: c.line_steps.clone(),
            detector_sample_steps: c.sample_steps.clone(),
            initial_flat: Some(c.flat.clone()),
            initial_background: Some(c.background.clone()),
            detector_size: (c.rows, c.columns),
        },
        "frontend-pixel-readout-f32",
        true
    );
    macro_rules! reconstruct {
        ($declaration:ty,$configuration:expr,$label:literal,$rows:expr) => {{
            let mut p = <$declaration>::prepare($configuration)?;
            for s in &c.steps {
                if s.action == "reset" {
                    p.reset();
                }
                let ordinal = s.offset as usize / c.block_rows;
                let mut image = c.image.clone();
                if s.action == "nan" {
                    image[c.invalid_pixel_index] = f32::NAN;
                }
                let block = mapping
                    .block_detector_indices(ordinal)?
                    .iter()
                    .map(|&i| image[i])
                    .collect::<Vec<_>>();
                let mut output = vec![-99.0; $rows];
                let outcome = <$declaration>::process_with_metadata(
                    &mut p,
                    &[(!s.missing).then_some(sample(s).with_offset(ordinal as u32))],
                    &block,
                    &mut output,
                )
                .ok();
                emit(results, $label, &s.name, outcome, &output);
            }
        }};
    }
    reconstruct!(
        DetectorFrameAssemblerF32,
        DetectorFrameAssemblerF32Configuration {
            region_lines: c.region_lines,
            block_lines: c.block_lines,
            region_samples: c.region_samples,
            detector_origins: c.detector_origins.clone(),
            detector_line_steps: c.line_steps.clone(),
            detector_sample_steps: c.sample_steps.clone(),
            detector_size: (c.rows, c.columns),
        },
        "frontend-frame-assembler-f32",
        c.rows * c.columns
    );
    reconstruct!(
        ShackHartmannReadoutReconstructorF32,
        ShackHartmannReadoutReconstructorF32Configuration {
            region_lines: c.region_lines,
            block_lines: c.block_lines,
            region_samples: c.region_samples,
            detector_origins: c.detector_origins.clone(),
            detector_line_steps: c.line_steps.clone(),
            detector_sample_steps: c.sample_steps.clone(),
            subaperture_count: c.subaperture_origins.len(),
            initial_subaperture_origins: Some(c.subaperture_origins.clone()),
            coordinate_scale: 1.25,
            pixel_threshold: 1.0,
            flux_threshold: 0.5,
            reference_slopes: Some(c.reference_slopes.clone()),
            active: Some(c.active.clone()),
            reconstructed_count: c.reconstructor_rows,
            initial_reconstructor: Some(c.matrix.clone()),
            reconstructed_schema: "test.frontend/reconstructed/1".into(),
            subaperture_size: (2, 2),
            detector_size: (c.rows, c.columns),
        },
        "frontend-shack-readout-f32",
        c.reconstructor_rows
    );
    Ok(())
}
fn complete(c: &Complete, results: &mut Vec<Record>) -> Result<(), Box<dyn std::error::Error>> {
    let mut p =
        ShackHartmannReconstructorF32::prepare(ShackHartmannReconstructorF32Configuration {
            subaperture_count: c.subapertures,
            initial_reconstructor: Some(c.matrix.clone()),
            reconstructed_schema: "test.frontend/reconstructed/1".into(),
            reconstructed_count: c.rows,
        })?;
    for name in [
        "initial",
        "nan",
        "after-nan",
        "overflow",
        "after-overflow",
        "replacement",
        "invalid-replacement",
        "after-reset",
    ] {
        let mut slopes = c.slopes.clone();
        if name == "nan" {
            *slopes.last_mut().ok_or("empty slopes")? = f32::NAN;
        }
        if name == "overflow" {
            slopes.fill(f32::MAX);
        }
        if name == "replacement" || name == "invalid-replacement" {
            let mut matrix = c.replacement.clone();
            if name == "invalid-replacement" {
                matrix[0] = f32::NAN;
            }
            let rejected = p
                .replace_parameter(2, AlgorithmInput::from(matrix.as_slice()))
                .ok_or("missing reconstructor port")?
                .is_err();
            record(
                results,
                "frontend-shack-complete-f32",
                format!("{name}-replacement-status"),
                vec![f64::from(rejected)],
            );
        }
        if name == "after-reset" {
            p.reset();
        }
        let mut output = vec![-99.0; c.rows];
        let outcome = ShackHartmannReconstructorF32::process_with_metadata(
            &mut p,
            &[None],
            &slopes,
            &mut output,
        )
        .ok();
        emit(
            results,
            "frontend-shack-complete-f32",
            name,
            outcome,
            &output,
        );
    }
    Ok(())
}
fn pol(c: &Pol, results: &mut Vec<Record>) -> Result<(), Box<dyn std::error::Error>> {
    let mut p =
        PseudoOpenLoopReconstructionF32::prepare(PseudoOpenLoopReconstructionF32Configuration {
            measurement_count: c.measurements,
            corrector_shape_count: c.correctors,
            residual_error_count: c.rows,
            initial_interaction_matrix: Some(c.interaction.clone()),
            initial_reconstructor: Some(c.reconstructor.clone()),
            initial_corrector_to_error: Some(c.corrector_to_error.clone()),
            residual_measurements_schema: "test.frontend/residual/1".into(),
            corrector_shape_schema: "test.frontend/shape/1".into(),
            pseudo_open_loop_measurements_schema: "test.frontend/pol/1".into(),
            residual_error_schema: "test.frontend/error/1".into(),
        })?;
    for name in [
        "initial",
        "nonfinite-residual",
        "after-nonfinite-residual",
        "nonfinite-shape",
        "after-nonfinite-shape",
        "overflow",
        "after-overflow",
        "interaction",
        "reconstructor",
        "corrector-to-error",
        "invalid-replacement",
        "after-reset",
    ] {
        let mut residual = c.residual.clone();
        let mut shape = c.shape.clone();
        if name == "nonfinite-residual" {
            residual[c.measurements - 1] = f32::NAN;
        }
        if name == "nonfinite-shape" {
            shape[c.correctors - 1] = f32::INFINITY;
        }
        if name == "overflow" {
            shape.fill(f32::MAX);
        }
        let replacement = match name {
            "interaction" => Some((4, c.interaction.clone())),
            "reconstructor" => Some((5, c.reconstructor.clone())),
            "corrector-to-error" => Some((6, c.corrector_to_error.clone())),
            "invalid-replacement" => {
                let mut v = c.interaction.clone();
                v[0] = f32::NAN;
                Some((4, v))
            }
            _ => None,
        };
        if let Some((port, mut values)) = replacement {
            if name != "invalid-replacement" {
                for v in &mut values {
                    *v *= -0.5;
                }
            }
            let rejected = p
                .replace_parameter(port, AlgorithmInput::from(values.as_slice()))
                .ok_or("missing POL parameter")?
                .is_err();
            record(
                results,
                "frontend-pol-f32",
                format!("{name}-replacement-status"),
                vec![f64::from(rejected)],
            );
        }
        if name == "after-reset" {
            p.reset();
        }
        let mut measurements = vec![-99.0; c.measurements];
        let mut error = vec![-99.0; c.rows];
        let outcome = PseudoOpenLoopReconstructionF32::process_with_metadata(
            &mut p,
            &[None, None],
            &residual,
            &shape,
            &mut measurements,
            &mut error,
        )
        .ok();
        measurements.extend(error);
        emit(results, "frontend-pol-f32", name, outcome, &measurements);
    }
    Ok(())
}
