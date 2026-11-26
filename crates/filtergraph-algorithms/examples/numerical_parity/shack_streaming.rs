//! Shared physical-row protocol observations for JFG streaming Shack–Hartmann.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmInput, SampleMetadata, ShackHartmannMeasurementBlockF32,
    ShackHartmannMeasurementBlockF32Configuration, ShackHartmannWeightedMeasurementBlockF32,
    ShackHartmannWeightedMeasurementBlockF32Configuration,
};
use serde::Deserialize;
#[derive(Deserialize)]
struct Fixture {
    region_lines: usize,
    block_lines: usize,
    region_samples: usize,
    detector_origins: Vec<[usize; 2]>,
    detector_line_steps: Vec<[i32; 2]>,
    detector_sample_steps: Vec<[i32; 2]>,
    readout_subaperture_origins: Vec<[usize; 2]>,
    reconstructor_rows: usize,
    reconstructor: Vec<f32>,
    image_rows: usize,
    image_columns: usize,
    block_rows: usize,
    region_rows: usize,
    region_columns: usize,
    origins: Vec<[usize; 2]>,
    weights: Vec<Vec<f32>>,
    active: Vec<bool>,
    reference_slopes: Vec<f32>,
    image: Vec<f32>,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Step {
    name: String,
    #[serde(default)]
    missing: bool,
    sequence: u64,
    row: u32,
    #[serde(default)]
    action: String,
    #[serde(default)]
    terminal: bool,
    #[serde(default)]
    discontinuity: bool,
    #[serde(default)]
    corrupted: bool,
    #[serde(default)]
    parameter: String,
    #[serde(default)]
    values: Vec<f32>,
    #[serde(default)]
    pixel_index: Option<usize>,
    #[serde(default)]
    pixel_value: Option<f32>,
}
pub(super) fn run(
    source: &str,
    results: &mut Vec<Record>,
) -> Result<(), Box<dyn std::error::Error>> {
    let c: Fixture = toml::from_str(source)?;
    macro_rules! family {
        ($declaration:ty,$configuration:expr,$label:literal,$weighted:literal) => {{
            let mut p = <$declaration>::prepare($configuration)?;
            let n = c.origins.len() * 2;
            for step in &c.steps {
                let mut image = c.image.clone();
                if let Some(index) = step.pixel_index {
                    image[index] = step.pixel_value.ok_or("missing pixel value")?;
                }
                if step.action == "reset" {
                    p.reset();
                }
                if !step.parameter.is_empty()
                    && step.parameter != "reconstructor"
                    && ($weighted || step.parameter != "weights")
                {
                    let port = match step.parameter.as_str() {
                        "origins" => 4,
                        "coordinates" => 5,
                        "references" => 6,
                        "thresholds" => 7,
                        "active" => 8,
                        "weights" => 9,
                        _ => return Err("unknown calibration parameter".into()),
                    };
                    let replacement = step.values.clone();
                    let flags = replacement.iter().map(|&v| v != 0.0).collect::<Vec<_>>();
                    let origins = replacement.iter().map(|&v| v as u32).collect::<Vec<_>>();
                    let input = if port == 8 {
                        AlgorithmInput::from(flags.as_slice())
                    } else if port == 4 {
                        AlgorithmInput::from(origins.as_slice())
                    } else {
                        AlgorithmInput::from(replacement.as_slice())
                    };
                    let rejected = p
                        .replace_parameter(port, input)
                        .ok_or("missing calibration parameter")?
                        .is_err();
                    record(
                        results,
                        $label,
                        format!("{}-replacement-status", step.name),
                        vec![f64::from(rejected)],
                    );
                }
                let mut values = vec![-99.0; n];
                let mut columns = vec![999; n];
                let mut count = [999];
                let sample = SampleMetadata::new(step.sequence, step.row)
                    .with_terminal(step.terminal)
                    .with_discontinuity(step.discontinuity)
                    .with_corrupted(step.corrupted)
                    .with_timestamp_nanoseconds(Some(1234));
                let first = step.row as usize * c.image_columns;
                let outcome = <$declaration>::process_with_metadata(
                    &mut p,
                    &[(!step.missing).then_some(sample)],
                    &image[first..first + c.block_rows * c.image_columns],
                    &mut values,
                    &mut columns,
                    &mut count,
                );
                let rejected = outcome.is_err();
                let mut status = vec![
                    f64::from(outcome.as_ref().is_ok_and(|r| r.is_complete())),
                    count[0] as f64,
                ];
                if let Some(m) = outcome.ok().and_then(|r| r.metadata()) {
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
                status.push(f64::from(rejected));
                record(results, $label, format!("{}-status", step.name), status);
                record(
                    results,
                    $label,
                    format!("{}-values", step.name),
                    values.iter().map(|&v| v as f64).collect(),
                );
            }
        }};
    }
    family!(
        ShackHartmannMeasurementBlockF32,
        ShackHartmannMeasurementBlockF32Configuration {
            row_block_rows: c.block_rows,
            subaperture_count: c.origins.len(),
            initial_subaperture_origins: Some(c.origins.clone()),
            coordinate_scale: 1.25,
            pixel_threshold: 1.0,
            flux_threshold: 0.5,
            reference_slopes: Some(c.reference_slopes.clone()),
            active: Some(c.active.clone()),
            measurements_schema: "test.filtergraph/measurement-block/1".into(),
            column_indices_schema: "test.filtergraph/measurement-column-indices/1".into(),
            measurement_count_schema: "test.filtergraph/measurement-count/1".into(),
            image_size: (c.image_rows, c.image_columns),
            subaperture_size: (c.region_rows, c.region_columns),
        },
        "shack-stream-unweighted-f32",
        false
    );
    family!(
        ShackHartmannWeightedMeasurementBlockF32,
        ShackHartmannWeightedMeasurementBlockF32Configuration {
            initial_weights: c.weights.clone(),
            row_block_rows: c.block_rows,
            subaperture_count: c.origins.len(),
            initial_subaperture_origins: Some(c.origins.clone()),
            coordinate_scale: 1.25,
            pixel_threshold: 1.0,
            flux_threshold: 0.5,
            reference_slopes: Some(c.reference_slopes.clone()),
            active: Some(c.active.clone()),
            measurements_schema: "test.filtergraph/measurement-block/1".into(),
            column_indices_schema: "test.filtergraph/measurement-column-indices/1".into(),
            measurement_count_schema: "test.filtergraph/measurement-count/1".into(),
            image_size: (c.image_rows, c.image_columns),
            subaperture_size: (c.region_rows, c.region_columns),
        },
        "shack-stream-weighted-f32",
        true
    );
    readout(&c, results)?;
    row_reconstruction(&c, results)?;
    Ok(())
}

fn readout(c: &Fixture, results: &mut Vec<Record>) -> Result<(), Box<dyn std::error::Error>> {
    use filtergraph_algorithms::{
        ShackHartmannReadoutMeasurementBlockF32 as Block,
        ShackHartmannReadoutMeasurementBlockF32Configuration as Configuration,
    };
    let mut p = Block::prepare(Configuration {
        detector_size: (c.image_rows, c.image_columns),
        region_lines: c.region_lines,
        block_lines: c.block_lines,
        region_samples: c.region_samples,
        detector_origins: c.detector_origins.clone(),
        detector_line_steps: c.detector_line_steps.clone(),
        detector_sample_steps: c.detector_sample_steps.clone(),
        subaperture_count: c.origins.len(),
        initial_subaperture_origins: Some(c.readout_subaperture_origins.clone()),
        coordinate_scale: 1.25,
        pixel_threshold: 1.0,
        flux_threshold: 0.5,
        reference_slopes: Some(c.reference_slopes.clone()),
        active: Some(c.active.clone()),
        measurements_schema: "test.filtergraph/measurement-block/1".into(),
        column_indices_schema: "test.filtergraph/measurement-column-indices/1".into(),
        measurement_count_schema: "test.filtergraph/measurement-count/1".into(),
        subaperture_size: (c.region_rows, c.region_columns),
    })?;
    let n = 2 * c.origins.len();
    for step in &c.steps {
        let mut image = c.image.clone();
        if let Some(index) = step.pixel_index {
            image[index] = step.pixel_value.ok_or("missing pixel value")?;
        }
        if step.action == "reset" {
            p.reset();
        }
        if !step.parameter.is_empty()
            && step.parameter != "weights"
            && step.parameter != "reconstructor"
        {
            let port = match step.parameter.as_str() {
                "origins" => 4,
                "coordinates" => 5,
                "references" => 6,
                "thresholds" => 7,
                "active" => 8,
                _ => return Err("unknown calibration parameter".into()),
            };
            let flags = step.values.iter().map(|&v| v != 0.0).collect::<Vec<_>>();
            let origins = step.values.iter().map(|&v| v as u32).collect::<Vec<_>>();
            let input = if port == 8 {
                AlgorithmInput::from(flags.as_slice())
            } else if port == 4 {
                AlgorithmInput::from(origins.as_slice())
            } else {
                AlgorithmInput::from(step.values.as_slice())
            };
            let rejected = p
                .replace_parameter(port, input)
                .ok_or("missing parameter")?
                .is_err();
            record(
                results,
                "shack-stream-readout-f32",
                format!("{}-replacement-status", step.name),
                vec![f64::from(rejected)],
            );
        }
        let ordinal = step.row as usize / c.block_rows;
        let block = p
            .plan()
            .readout()
            .block_detector_indices(ordinal)?
            .iter()
            .map(|&index| image[index])
            .collect::<Vec<_>>();
        let sample = SampleMetadata::new(step.sequence, ordinal as u32)
            .with_terminal(step.terminal)
            .with_discontinuity(step.discontinuity)
            .with_corrupted(step.corrupted)
            .with_timestamp_nanoseconds(Some(1234));
        let mut values = vec![-99.0; n];
        let mut columns = vec![999; n];
        let mut count = [999];
        let outcome = Block::process_with_metadata(
            &mut p,
            &[(!step.missing).then_some(sample)],
            &block,
            &mut values,
            &mut columns,
            &mut count,
        );
        let rejected = outcome.is_err();
        let mut status = vec![
            f64::from(outcome.as_ref().is_ok_and(|r| r.is_complete())),
            count[0] as f64,
        ];
        if let Some(m) = outcome.ok().and_then(|r| r.metadata()) {
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
        status.push(f64::from(rejected));
        record(
            results,
            "shack-stream-readout-f32",
            format!("{}-status", step.name),
            status,
        );
        record(
            results,
            "shack-stream-readout-f32",
            format!("{}-values", step.name),
            values.iter().map(|&v| v as f64).collect(),
        );
    }
    Ok(())
}
fn row_reconstruction(
    c: &Fixture,
    results: &mut Vec<Record>,
) -> Result<(), Box<dyn std::error::Error>> {
    use filtergraph_algorithms::{
        ShackHartmannRowReconstructorF32 as Block,
        ShackHartmannRowReconstructorF32Configuration as Configuration,
    };
    let mut p = Block::prepare(Configuration {
        row_block_rows: c.block_rows,
        subaperture_count: c.origins.len(),
        initial_subaperture_origins: Some(c.origins.clone()),
        coordinate_scale: 1.25,
        pixel_threshold: 1.0,
        flux_threshold: 0.5,
        reference_slopes: Some(c.reference_slopes.clone()),
        active: Some(c.active.clone()),
        reconstructed_count: c.reconstructor_rows,
        initial_reconstructor: Some(c.reconstructor.clone()),
        reconstructed_schema: "test.filtergraph/reconstructed/1".into(),
        image_size: (c.image_rows, c.image_columns),
        subaperture_size: (c.region_rows, c.region_columns),
    })?;
    for step in &c.steps {
        let mut image = c.image.clone();
        if let Some(index) = step.pixel_index {
            image[index] = step.pixel_value.ok_or("missing pixel value")?;
        }
        if step.action == "reset" {
            p.reset();
        }
        if !step.parameter.is_empty() && step.parameter != "weights" {
            let port = match step.parameter.as_str() {
                "origins" => 2,
                "coordinates" => 3,
                "references" => 4,
                "thresholds" => 5,
                "active" => 6,
                "reconstructor" => 7,
                _ => return Err("unknown parameter".into()),
            };
            let flags = step.values.iter().map(|&v| v != 0.0).collect::<Vec<_>>();
            let origins = step.values.iter().map(|&v| v as u32).collect::<Vec<_>>();
            let input = if port == 6 {
                AlgorithmInput::from(flags.as_slice())
            } else if port == 2 {
                AlgorithmInput::from(origins.as_slice())
            } else {
                AlgorithmInput::from(step.values.as_slice())
            };
            let rejected = p
                .replace_parameter(port, input)
                .ok_or("missing parameter")?
                .is_err();
            record(
                results,
                "shack-stream-row-reconstruction-f32",
                format!("{}-replacement-status", step.name),
                vec![f64::from(rejected)],
            );
        }
        let sample = SampleMetadata::new(step.sequence, step.row)
            .with_terminal(step.terminal)
            .with_discontinuity(step.discontinuity)
            .with_corrupted(step.corrupted)
            .with_timestamp_nanoseconds(Some(1234));
        let first = step.row as usize * c.image_columns;
        let mut values = vec![-99.0; c.reconstructor_rows];
        let outcome = Block::process_with_metadata(
            &mut p,
            &[(!step.missing).then_some(sample)],
            &image[first..first + c.block_rows * c.image_columns],
            &mut values,
        );
        let mut status = vec![f64::from(outcome.is_err())];
        if let Ok(outcome) = outcome {
            status.push(f64::from(outcome.is_complete()));
            if let Some(m) = outcome.metadata() {
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
        } else {
            status.push(0.0);
            status.extend([-1.0; 5]);
        }
        record(
            results,
            "shack-stream-row-reconstruction-f32",
            format!("{}-status", step.name),
            status,
        );
        record(
            results,
            "shack-stream-row-reconstruction-f32",
            format!("{}-values", step.name),
            values.iter().map(|&v| v as f64).collect(),
        );
    }
    Ok(())
}
