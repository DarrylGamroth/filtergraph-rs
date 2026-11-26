//! Pinned-source pyramid lifecycle inputs and logical-layout adapters only.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmOutput, DeclarationProcessError as DE,
    DetectorReadoutPlan, GemvError, NativeProcessError as NE, PreparedAlgorithm,
    PyramidMeasurementBlockError as ME, PyramidMeasurementBlockF32 as Rows,
    PyramidMeasurementBlockF32Configuration as RC, PyramidPixelError as PE,
    PyramidPixelRowProcessError as RE, PyramidPupilReconstructorF32 as Dense,
    PyramidPupilReconstructorF32Configuration as DC, PyramidReadoutMeasurementBlockF32 as Readout,
    PyramidReadoutMeasurementBlockF32Configuration as IC, PyramidReadoutReconstructionError as IE,
    SampleMetadata,
};
use serde::Deserialize;
use std::error::Error;
type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[derive(Deserialize)]
struct Fixture {
    geometry: Geometry,
    dense: DenseFixture,
    rows: Vec<Step>,
    readout: Vec<Step>,
    legacy: Legacy,
}
#[derive(Deserialize)]
struct Geometry {
    image_size: (usize, usize),
    pupil_size: (usize, usize),
    pupil_origins: Vec<[usize; 2]>,
    pupil_mask: Vec<bool>,
    region_lines: usize,
    block_lines: usize,
    region_samples: usize,
    detector_origins: Vec<[usize; 2]>,
    detector_line_steps: Vec<[i32; 2]>,
    detector_sample_steps: Vec<[i32; 2]>,
}
#[derive(Deserialize)]
struct DenseFixture {
    reconstructed_count: usize,
    selected_pixel_count: usize,
    matrix: Vec<f32>,
    input: Vec<f32>,
    replacement: Vec<f32>,
}
#[derive(Deserialize)]
struct Step {
    name: String,
    sequence: u64,
    offset: u32,
    kind: String,
    terminal: bool,
    #[serde(default)]
    discontinuity: bool,
    #[serde(default)]
    corrupted: bool,
    #[serde(default)]
    action: String,
    #[serde(default)]
    missing_metadata: bool,
    #[serde(default)]
    parameter: String,
    #[serde(default)]
    values: Vec<toml::Value>,
}
fn pixel_reason(e: &PE) -> Option<u8> {
    match e {
        PE::NonFiniteInput { .. } => Some(3),
        PE::NonFiniteMean => Some(4),
        PE::NonFiniteNormalizationGain => Some(5),
        PE::NonFiniteOutput { .. } => Some(6),
        _ => None,
    }
}
fn row_reason(e: &DE<ME>) -> Option<u8> {
    match e {
        DE::Algorithm(ME::Source(RE::MissingSampleMetadata)) => Some(1),
        DE::Algorithm(ME::Source(RE::OffsetOverflow)) => Some(2),
        DE::Algorithm(ME::Source(RE::Algorithm(
            filtergraph_algorithms::PyramidPixelRowError::Pyramid(e),
        ))) => pixel_reason(e),
        _ => None,
    }
}
fn readout_reason(e: &DE<IE>) -> Option<u8> {
    match e {
        DE::Algorithm(IE::MissingSampleMetadata) => Some(1),
        DE::Algorithm(IE::OrdinalOverflow) => Some(2),
        DE::Algorithm(IE::Pixels(e)) => pixel_reason(e),
        DE::Algorithm(IE::Reconstruction(_)) => Some(7),
        _ => None,
    }
}
fn stream<D>(
    mut p: PreparedAlgorithm<D>,
    c: &Geometry,
    steps: &[Step],
    label: &str,
    readout: Option<&DetectorReadoutPlan>,
    reason: fn(&D::ProcessError) -> Option<u8>,
    results: &mut Vec<Record>,
) -> Result<()>
where
    D: AlgorithmDeclaration,
    D::ProcessError: Error + 'static,
    D::ParameterError: Error + 'static,
{
    let n = 4 * c.pupil_mask.iter().filter(|&&v| v).count();
    for step in steps {
        if step.action == "reset" {
            p.reset();
        }
        if !step.parameter.is_empty() {
            let rejected = match step.parameter.as_str() {
                "mask" => {
                    let v = step
                        .values
                        .iter()
                        .map(|v| v.as_bool().ok_or("mask flag"))
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    p.replace_parameter(5, AlgorithmInput::from(v.as_slice()))
                        .ok_or("missing mask")?
                        .is_err()
                }
                "origins" => {
                    let v = step
                        .values
                        .iter()
                        .map(|v| {
                            v.as_integer()
                                .and_then(|v| u32::try_from(v).ok())
                                .ok_or("origin integer")
                        })
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    p.replace_parameter(4, AlgorithmInput::from(v.as_slice()))
                        .ok_or("missing origins")?
                        .is_err()
                }
                _ => return Err("unknown parameter".into()),
            };
            record(
                results,
                label,
                format!("{}-replacement-status", step.name),
                vec![f64::from(rejected)],
            );
        }
        let pixels = c.image_size.0 * c.image_size.1;
        let image: Vec<f32> = (0..pixels)
            .map(|i| match step.kind.as_str() {
                "two" => 2.0,
                "ordinary" => (i + 1) as f32 / 8.0,
                "zero" => 0.0,
                "negative" => -1.0,
                "nan" => f32::NAN,
                "maximum" => f32::MAX,
                _ => f32::NAN,
            })
            .collect();
        let block = if let Some(readout) = readout {
            readout
                .block_detector_indices(step.offset as usize)?
                .iter()
                .map(|&i| image[i])
                .collect::<Vec<_>>()
        } else {
            let first = step.offset as usize * c.image_size.1;
            image[first..first + c.image_size.1].to_vec()
        };
        let mut values = vec![71.0_f32; n];
        let mut columns = vec![999u32; n];
        let mut count = [999u32];
        let sample = SampleMetadata::new(step.sequence, step.offset)
            .with_terminal(step.terminal)
            .with_discontinuity(step.discontinuity)
            .with_corrupted(step.corrupted)
            .with_timestamp_nanoseconds(Some(1234));
        let outcome = {
            let mut outputs = [
                AlgorithmOutput::from(values.as_mut_slice()),
                AlgorithmOutput::from(columns.as_mut_slice()),
                AlgorithmOutput::from(count.as_mut_slice()),
            ];
            p.process_with_metadata(
                &[AlgorithmInput::from(block.as_slice())],
                &[if step.missing_metadata {
                    None
                } else {
                    Some(sample)
                }],
                &mut outputs,
            )
        };
        let (code, result) = match outcome {
            Ok(r) => (0, Some(r)),
            Err(NE::Process(e)) => (
                reason(&e).ok_or_else(|| format!("{label}/{} unexpected {e}", step.name))?,
                None,
            ),
            Err(e) => return Err(e.into()),
        };
        let mut status = vec![
            code as f64,
            f64::from(result.is_some_and(|r| r.is_complete())),
            count[0] as f64,
        ];
        if let Some(m) = result.and_then(|r| r.metadata()) {
            status.extend([
                m.sequence() as f64,
                m.offset() as f64,
                f64::from(m.is_terminal()),
                f64::from(m.is_discontinuous()),
                m.timestamp_nanoseconds().unwrap_or(-1) as f64,
            ]);
        } else {
            status.extend([-1.0; 5]);
        }
        status.extend(columns.iter().map(|&v| v as f64));
        record(results, label, format!("{}-status", step.name), status);
        record(
            results,
            label,
            format!("{}-values", step.name),
            values.into_iter().map(f64::from).collect(),
        );
    }
    Ok(())
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<()> {
    let f: Fixture = toml::from_str(source)?;
    let c = &f.geometry;
    let rc = RC {
        image_size: c.image_size,
        row_block_rows: 1,
        pupil_size: c.pupil_size,
        pupil_origins: c.pupil_origins.clone(),
        pupil_mask: Some(c.pupil_mask.clone()),
        measurements_schema: "test.pyramid/measurements/1".into(),
        column_indices_schema: "test.pyramid/columns/1".into(),
        measurement_count_schema: "test.pyramid/count/1".into(),
    };
    stream(
        Rows::prepare(rc)?,
        c,
        &f.rows,
        "pyramid-measurement-block-f32",
        None,
        row_reason,
        results,
    )?;
    let ic = IC {
        detector_size: c.image_size,
        region_lines: c.region_lines,
        block_lines: c.block_lines,
        region_samples: c.region_samples,
        detector_origins: c.detector_origins.clone(),
        detector_line_steps: c.detector_line_steps.clone(),
        detector_sample_steps: c.detector_sample_steps.clone(),
        pupil_size: c.pupil_size,
        pupil_origins: c.pupil_origins.clone(),
        pupil_mask: Some(c.pupil_mask.clone()),
        measurements_schema: "test.pyramid/measurements/1".into(),
        column_indices_schema: "test.pyramid/columns/1".into(),
        measurement_count_schema: "test.pyramid/count/1".into(),
    };
    let readout = DetectorReadoutPlan::from_i32_coordinate_steps(
        c.image_size,
        c.region_lines,
        c.block_lines,
        c.region_samples,
        &c.detector_origins,
        &c.detector_line_steps,
        &c.detector_sample_steps,
    )?;
    stream(
        Readout::prepare(ic)?,
        c,
        &f.readout,
        "pyramid-readout-measurement-block-f32",
        Some(&readout),
        readout_reason,
        results,
    )?;
    legacy(&f, results)?;
    let c = f.dense;
    let label = "pwfs-pupil-reconstructor-f32";
    let mut p = Dense::prepare(DC {
        reconstructed_count: c.reconstructed_count,
        selected_pixel_count: c.selected_pixel_count,
        initial_reconstructor: Some(c.matrix),
        reconstructed_schema: "test.pyramid/reconstructed/1".into(),
    })?;
    for name in [
        "ordinary",
        "nan",
        "overflow",
        "recovery",
        "replaced",
        "invalid",
        "after-invalid",
        "reset",
    ] {
        if name == "replaced" {
            p.replace_parameter(2, AlgorithmInput::from(c.replacement.as_slice()))
                .ok_or("reconstructor")??;
        }
        if name == "invalid" {
            let invalid = vec![f32::NAN; c.replacement.len()];
            let rejected = p
                .replace_parameter(2, AlgorithmInput::from(invalid.as_slice()))
                .ok_or("reconstructor")?
                .is_err();
            record(
                results,
                label,
                "invalid-replacement-status".into(),
                vec![f64::from(rejected)],
            );
        }
        if name == "reset" {
            p.reset();
        }
        let mut input = c.input.clone();
        if name == "nan" {
            input[0] = f32::NAN;
        }
        if name == "overflow" {
            input.fill(f32::MAX);
        }
        let mut values = vec![71.0_f32; c.reconstructed_count];
        let outcome = Dense::process(&mut p, &input, &mut values);
        let code = match outcome {
            Ok(_) => 0,
            Err(DE::Algorithm(GemvError::NonFiniteInput)) => 1,
            Err(DE::Algorithm(GemvError::NonFiniteOutput)) => 2,
            Err(e) => return Err(e.into()),
        };
        let mut status = vec![code as f64];
        status.extend(values.iter().map(|&v| {
            if v.is_nan() {
                1.0
            } else if v == f32::INFINITY {
                2.0
            } else if v == f32::NEG_INFINITY {
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
                .into_iter()
                .map(|v| if v.is_finite() { v as f64 } else { 0.0 })
                .collect(),
        );
    }
    Ok(())
}

#[derive(Deserialize)]
struct Legacy {
    reconstructed_count: usize,
    matrix: Vec<f32>,
    replacement: Vec<f32>,
    image_cases: Vec<String>,
    differential_input: Vec<f32>,
    differential_matrix: Vec<f32>,
    differential_replacement: Vec<f32>,
}
use filtergraph_algorithms::{
    PortDirection, PyramidPixelImageError as IAE, PyramidPixelImageProcessError as IPE,
    PyramidPixelRowF32 as PixelRows, PyramidPixelRowF32Configuration as PRC,
    PyramidPupilImageF32 as Image, PyramidPupilImageF32Configuration as PIC,
    PyramidReadoutReconstructorF32 as ReadoutRecon,
    PyramidReadoutReconstructorF32Configuration as IRC, PyramidRowReconstructionError as RAE,
    PyramidRowReconstructionProcessError as RPE, PyramidRowReconstructorF32 as RowRecon,
    PyramidRowReconstructorF32Configuration as RRC,
};
fn old_row_reason(e: &DE<RE>) -> Option<u8> {
    match e {
        DE::Algorithm(RE::MissingSampleMetadata) => Some(1),
        DE::Algorithm(RE::OffsetOverflow) => Some(2),
        DE::Algorithm(RE::Algorithm(filtergraph_algorithms::PyramidPixelRowError::Pyramid(e))) => {
            pixel_reason(e)
        }
        _ => None,
    }
}
fn image_reason(e: &DE<IPE>) -> Option<u8> {
    match e {
        DE::Algorithm(IPE::Algorithm(IAE::Pyramid(e))) => pixel_reason(e).map(|v| v - 2),
        _ => None,
    }
}
fn row_reconstruction_reason(e: &DE<RPE>) -> Option<u8> {
    match e {
        DE::Algorithm(RPE::MissingSampleMetadata) => Some(1),
        DE::Algorithm(RPE::OffsetOverflow) => Some(2),
        DE::Algorithm(RPE::Algorithm(RAE::Pixels(
            filtergraph_algorithms::PyramidPixelRowError::Pyramid(e),
        ))) => pixel_reason(e),
        DE::Algorithm(RPE::Algorithm(RAE::NonFiniteReconstruction)) => Some(7),
        _ => None,
    }
}
fn observe<D>(
    p: &mut PreparedAlgorithm<D>,
    label: &str,
    case: &str,
    input: &[f32],
    sample: Option<SampleMetadata>,
    reason: fn(&D::ProcessError) -> Option<u8>,
    results: &mut Vec<Record>,
) -> Result<()>
where
    D: AlgorithmDeclaration,
    D::ProcessError: Error + 'static,
{
    let descriptors: Vec<_> = D::PORTS
        .iter()
        .zip(p.formats())
        .filter(|(port, _)| port.direction() == PortDirection::Output)
        .map(|(port, format)| (port.name(), format.elements()))
        .collect();
    let mut buffers: Vec<_> = descriptors
        .iter()
        .map(|(_, n)| vec![71.0_f32; *n])
        .collect();
    let mut outputs: Vec<_> = buffers
        .iter_mut()
        .map(|v| AlgorithmOutput::from(v.as_mut_slice()))
        .collect();
    let outcome = p.process_with_metadata(&[AlgorithmInput::from(input)], &[sample], &mut outputs);
    drop(outputs);
    let (code, result) = match outcome {
        Ok(r) => (0, Some(r)),
        Err(NE::Process(e)) => (
            reason(&e).ok_or_else(|| format!("{label}/{case}: unexpected {e}"))?,
            None,
        ),
        Err(e) => return Err(e.into()),
    };
    let mut status = vec![
        code as f64,
        f64::from(result.is_some_and(|r| r.is_complete())),
    ];
    if let Some(m) = result.and_then(|r| r.metadata()) {
        status.extend([
            m.sequence() as f64,
            m.offset() as f64,
            f64::from(m.is_terminal()),
            f64::from(m.is_discontinuous()),
            m.timestamp_nanoseconds().unwrap_or(-1) as f64,
        ]);
    } else {
        status.extend([-1.0; 5]);
    }
    for ((name, _), values) in descriptors.iter().zip(buffers) {
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
        record(
            results,
            label,
            format!("{case}-{name}"),
            values
                .into_iter()
                .map(|v| if v.is_finite() { v as f64 } else { 0.0 })
                .collect(),
        );
    }
    record(results, label, format!("{case}-status"), status);
    Ok(())
}
fn legacy_stream<D>(
    mut p: PreparedAlgorithm<D>,
    f: &Fixture,
    label: &str,
    readout: Option<&DetectorReadoutPlan>,
    base: usize,
    reason: fn(&D::ProcessError) -> Option<u8>,
    results: &mut Vec<Record>,
) -> Result<()>
where
    D: AlgorithmDeclaration,
    D::ProcessError: Error + 'static,
    D::ParameterError: Error + 'static,
{
    let c = &f.geometry;
    for step in if readout.is_some() {
        &f.readout
    } else {
        &f.rows
    } {
        if step.action == "reset" {
            p.reset();
        }
        if !step.parameter.is_empty() {
            let rejected = if step.parameter == "mask" {
                let v = step
                    .values
                    .iter()
                    .map(|v| v.as_bool().ok_or("mask flag"))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                p.replace_parameter(base + 1, AlgorithmInput::from(v.as_slice()))
                    .ok_or("mask port")?
                    .is_err()
            } else {
                let v = step
                    .values
                    .iter()
                    .map(|v| {
                        v.as_integer()
                            .and_then(|v| u32::try_from(v).ok())
                            .ok_or("origin")
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                p.replace_parameter(base, AlgorithmInput::from(v.as_slice()))
                    .ok_or("origin port")?
                    .is_err()
            };
            record(
                results,
                label,
                format!("{}-replacement-status", step.name),
                vec![f64::from(rejected)],
            );
        }
        let image: Vec<f32> = (0..c.image_size.0 * c.image_size.1)
            .map(|i| match step.kind.as_str() {
                "two" => 2.0,
                "ordinary" => (i + 1) as f32 / 8.0,
                "zero" => 0.0,
                "negative" => -1.0,
                "nan" => f32::NAN,
                "maximum" => f32::MAX,
                _ => f32::NAN,
            })
            .collect();
        let input = if let Some(readout) = readout {
            readout
                .block_detector_indices(step.offset as usize)?
                .iter()
                .map(|&i| image[i])
                .collect::<Vec<_>>()
        } else {
            let start = step.offset as usize * c.image_size.1;
            image[start..start + c.image_size.1].to_vec()
        };
        let sample = SampleMetadata::new(step.sequence, step.offset)
            .with_terminal(step.terminal)
            .with_discontinuity(step.discontinuity)
            .with_corrupted(step.corrupted)
            .with_timestamp_nanoseconds(Some(1234));
        observe(
            &mut p,
            label,
            &step.name,
            &input,
            if step.missing_metadata {
                None
            } else {
                Some(sample)
            },
            reason,
            results,
        )?;
    }

    if label == "pwfs-row-reconstructor-f32" {
        for (case, sequence, row, action) in [
            ("matrix-before", 40, 0, 0),
            ("matrix-adopted-partial", 40, 1, 1),
        ] {
            if action == 1 {
                p.replace_parameter(6, AlgorithmInput::from(f.legacy.replacement.as_slice()))
                    .ok_or("matrix")??;
                record(
                    results,
                    label,
                    "matrix-replacement-status".into(),
                    vec![0.0],
                );
            }
            observe(
                &mut p,
                label,
                case,
                &vec![2.0_f32; c.image_size.1],
                Some(SampleMetadata::new(sequence, row).with_timestamp_nanoseconds(Some(1234))),
                reason,
                results,
            )?;
        }
        for row in 0..c.image_size.0 {
            observe(
                &mut p,
                label,
                &format!("matrix-recovery-{row}"),
                &vec![2.0_f32; c.image_size.1],
                Some(
                    SampleMetadata::new(41, row as u32)
                        .with_terminal(row + 1 == c.image_size.0)
                        .with_timestamp_nanoseconds(Some(1234)),
                ),
                reason,
                results,
            )?;
        }
        for row in 0..c.image_size.0 {
            if row == 1 {
                let invalid = vec![f32::NAN; f.legacy.replacement.len()];
                let rejected = p
                    .replace_parameter(6, AlgorithmInput::from(invalid.as_slice()))
                    .ok_or("matrix")?
                    .is_err();
                record(
                    results,
                    label,
                    "invalid-matrix-status".into(),
                    vec![f64::from(rejected)],
                );
            }
            observe(
                &mut p,
                label,
                &format!("after-invalid-matrix-{row}"),
                &vec![2.0_f32; c.image_size.1],
                Some(
                    SampleMetadata::new(42, row as u32)
                        .with_terminal(row + 1 == c.image_size.0)
                        .with_timestamp_nanoseconds(Some(1234)),
                ),
                reason,
                results,
            )?;
        }
    }
    Ok(())
}
fn legacy(f: &Fixture, results: &mut Vec<Record>) -> Result<()> {
    let c = &f.geometry;
    let l = &f.legacy;
    let p = PixelRows::prepare(PRC {
        row_block_rows: 1,
        pupil_origins: c.pupil_origins.clone(),
        pupil_mask: Some(c.pupil_mask.clone()),
        image_size: (c.image_size.0, c.image_size.1),
        pupil_size: (c.pupil_size.0, c.pupil_size.1),
    })?;
    legacy_stream(
        p,
        f,
        "pyramid-pixel-row-f32",
        None,
        3,
        old_row_reason,
        results,
    )?;
    let p = RowRecon::prepare(RRC {
        row_block_rows: 1,
        pupil_origins: c.pupil_origins.clone(),
        pupil_mask: Some(c.pupil_mask.clone()),
        reconstructed_count: l.reconstructed_count,
        initial_reconstructor: Some(l.matrix.clone()),
        reconstructed_schema: "test.pyramid/reconstructed/1".into(),
        image_size: (c.image_size.0, c.image_size.1),
        pupil_size: (c.pupil_size.0, c.pupil_size.1),
    })?;
    legacy_stream(
        p,
        f,
        "pwfs-row-reconstructor-f32",
        None,
        4,
        row_reconstruction_reason,
        results,
    )?;
    let readout = DetectorReadoutPlan::from_i32_coordinate_steps(
        c.image_size,
        c.region_lines,
        c.block_lines,
        c.region_samples,
        &c.detector_origins,
        &c.detector_line_steps,
        &c.detector_sample_steps,
    )?;
    let p = ReadoutRecon::prepare(IRC {
        region_lines: c.region_lines,
        block_lines: c.block_lines,
        region_samples: c.region_samples,
        detector_origins: c.detector_origins.clone(),
        detector_line_steps: c.detector_line_steps.clone(),
        detector_sample_steps: c.detector_sample_steps.clone(),
        pupil_origins: c.pupil_origins.clone(),
        pupil_mask: Some(c.pupil_mask.clone()),
        reconstructed_count: l.reconstructed_count,
        initial_reconstructor: Some(l.matrix.clone()),
        reconstructed_schema: "test.pyramid/reconstructed/1".into(),
        pupil_size: (c.pupil_size.0, c.pupil_size.1),
        detector_size: (c.image_size.0, c.image_size.1),
    })?;
    legacy_stream(
        p,
        f,
        "pwfs-readout-reconstructor-f32",
        Some(&readout),
        0,
        readout_reason,
        results,
    )?;
    let mut p = Image::prepare(PIC {
        pupil_origins: c.pupil_origins.clone(),
        pupil_mask: Some(c.pupil_mask.clone()),
        image_size: (c.image_size.0, c.image_size.1),
        pupil_size: (c.pupil_size.0, c.pupil_size.1),
    })?;
    for case in &l.image_cases {
        if case == "reset" {
            p.reset();
        }
        if case == "mask" || case == "invalid-mask" {
            let mask = if case == "mask" {
                vec![false, true, true, true]
            } else {
                vec![false; 4]
            };
            let rejected = p
                .replace_parameter(4, AlgorithmInput::from(mask.as_slice()))
                .ok_or("mask")?
                .is_err();
            record(
                results,
                "pyramid-pupil-image-f32",
                format!("{case}-replacement-status"),
                vec![f64::from(rejected)],
            );
        }
        if case == "origins" || case == "invalid-origins" {
            let origins = if case == "origins" {
                vec![2u32, 2, 4, 0, 2, 0, 4, 2]
            } else {
                vec![6u32, 2, 4, 0, 2, 0, 4, 2]
            };
            let rejected = p
                .replace_parameter(3, AlgorithmInput::from(origins.as_slice()))
                .ok_or("origins")?
                .is_err();
            record(
                results,
                "pyramid-pupil-image-f32",
                format!("{case}-replacement-status"),
                vec![f64::from(rejected)],
            );
        }
        let image: Vec<f32> = (0..c.image_size.0 * c.image_size.1)
            .map(|i| match case.as_str() {
                "startup" => 2.0,
                "zero" => 0.0,
                "negative" => -1.0,
                "nan" => f32::NAN,
                "overflow" => f32::MAX,
                _ => (i + 1) as f32 / 8.0,
            })
            .collect();
        observe(
            &mut p,
            "pyramid-pupil-image-f32",
            case,
            &image,
            None,
            image_reason,
            results,
        )?;
    }
    let mut differential = filtergraph_algorithms::PyramidReconstructorF32::prepare(
        filtergraph_algorithms::PyramidReconstructorF32Configuration {
            reconstructed_count: l.reconstructed_count,
            selected_coordinate_count: l.differential_input.len() / 2,
            initial_reconstructor: Some(l.differential_matrix.clone()),
            reconstructed_schema: "test.pyramid/reconstructed/1".into(),
        },
    )?;
    fn differential_reason(e: &DE<GemvError>) -> Option<u8> {
        match e {
            DE::Algorithm(GemvError::NonFiniteInput) => Some(1),
            DE::Algorithm(GemvError::NonFiniteOutput) => Some(2),
            _ => None,
        }
    }
    for name in [
        "ordinary",
        "nan",
        "overflow",
        "recovery",
        "replaced",
        "invalid",
        "after-invalid",
        "reset",
    ] {
        if name == "replaced" {
            differential
                .replace_parameter(
                    2,
                    AlgorithmInput::from(l.differential_replacement.as_slice()),
                )
                .ok_or("I4Q matrix")??;
        }
        if name == "invalid" {
            let invalid = vec![f32::NAN; l.differential_replacement.len()];
            let rejected = differential
                .replace_parameter(2, AlgorithmInput::from(invalid.as_slice()))
                .ok_or("I4Q matrix")?
                .is_err();
            record(
                results,
                "pwfs-reconstructor-f32",
                "invalid-replacement-status".into(),
                vec![f64::from(rejected)],
            );
        }
        if name == "reset" {
            differential.reset();
        }
        let mut input = l.differential_input.clone();
        if name == "nan" {
            input[0] = f32::NAN;
        }
        if name == "overflow" {
            input.fill(f32::MAX);
        }
        observe(
            &mut differential,
            "pwfs-reconstructor-f32",
            name,
            &input,
            None,
            differential_reason,
            results,
        )?;
    }

    Ok(())
}
