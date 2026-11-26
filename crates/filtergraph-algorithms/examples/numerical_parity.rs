//! Execute shared scientific parity inputs with the maintained Rust plans.
use std::{error::Error, sync::Arc};

use filtergraph_algorithms::{
    AlgorithmPlan, LeakyIntegratorPlan, LeakyIntegratorWorkspace, PixelCalibrationPlan,
    PreparedGemv,
};
use serde::{Deserialize, Serialize};

#[path = "numerical_parity/biquad.rs"]
mod biquad;
#[path = "numerical_parity/calibration.rs"]
mod calibration;
#[path = "numerical_parity/closed_loop.rs"]
mod closed_loop;
#[path = "numerical_parity/conditioning.rs"]
mod conditioning;
#[path = "numerical_parity/control.rs"]
mod control;
#[path = "numerical_parity/corrector_allocation.rs"]
mod corrector_allocation;
#[path = "numerical_parity/dm_calibration.rs"]
mod dm_calibration;
#[path = "numerical_parity/docrime.rs"]
mod docrime;
#[path = "numerical_parity/frame_delay.rs"]
mod frame_delay;
#[path = "numerical_parity/frontends.rs"]
mod frontends;
#[path = "numerical_parity/ml_llowfs.rs"]
mod ml_llowfs;
#[path = "numerical_parity/optical_gain.rs"]
mod optical_gain;
#[path = "numerical_parity/pixel_f32.rs"]
mod pixel_f32;
#[path = "numerical_parity/pupil_flux.rs"]
mod pupil_flux;
#[path = "numerical_parity/pupil_signals.rs"]
mod pupil_signals;
#[path = "numerical_parity/pyramid_composites.rs"]
mod pyramid_composites;
#[path = "numerical_parity/scc_processing.rs"]
mod scc_processing;
#[path = "numerical_parity/scc_residual.rs"]
mod scc_residual;
#[path = "numerical_parity/scientific_contracts.rs"]
mod scientific_contracts;
#[path = "numerical_parity/sensing.rs"]
mod sensing;
#[path = "numerical_parity/shack_streaming.rs"]
mod shack_streaming;
#[path = "numerical_parity/shape_dense.rs"]
mod shape_dense;
#[path = "numerical_parity/vector.rs"]
mod vector;
#[path = "numerical_parity/weighted_cog.rs"]
mod weighted_cog;
#[path = "numerical_parity/weighted_shack.rs"]
mod weighted_shack;

#[path = "numerical_parity/catalog_additions.rs"]
mod catalog_additions;
#[path = "numerical_parity/chopped_acquisition.rs"]
mod chopped_acquisition;

#[derive(Deserialize)]
struct Fixture {
    leaky: Leaky,
    control: control::Fixture,
    pdm: control::PdmFixture,
    gemv: Vec<Gemv>,
    pixel: Pixel,
}
#[derive(Deserialize)]
struct Leaky {
    gain: f64,
    pole: f64,
    reset_before_frame: usize,
    frames: Vec<Vec<f64>>,
}
#[derive(Deserialize)]
struct Gemv {
    name: String,
    rows: usize,
    columns: usize,
    matrix: Vec<f32>,
    input: Vec<f32>,
}
#[derive(Deserialize)]
struct Pixel {
    rows: usize,
    columns: usize,
    flat: Vec<f32>,
    background: Vec<f32>,
    frames: Vec<Vec<u16>>,
}
#[derive(Serialize)]
struct Record {
    algorithm: String,
    case: String,
    values: Vec<f64>,
}
#[derive(Serialize)]
struct Results {
    results: Vec<Record>,
}
fn record(results: &mut Vec<Record>, algorithm: &str, case: String, values: Vec<f64>) {
    results.push(Record {
        algorithm: algorithm.into(),
        case,
        values,
    });
}
fn run(f: Fixture) -> Result<Results, Box<dyn Error>> {
    let mut results = Vec::new();
    control::run(&f.control, &mut results)?;
    control::pdm(f.pdm, &mut results)?;
    // Both precisions exercise the public state access and reset contract.
    macro_rules! leaky {
        ($t:ty, $label:literal) => {{
            let plan = LeakyIntegratorPlan::new(f.leaky.gain as $t, f.leaky.pole as $t)?;
            let mut workspace = LeakyIntegratorWorkspace::new(f.leaky.frames[0].len())?;
            let mut output = vec![0.0 as $t; workspace.len()];
            for (frame, values) in f.leaky.frames.iter().enumerate() {
                if frame == f.leaky.reset_before_frame {
                    workspace.reset();
                    record(
                        &mut results,
                        $label,
                        "reset-state".into(),
                        workspace.state().iter().map(|&v| v as f64).collect(),
                    );
                }
                let input: Vec<$t> = values.iter().map(|&v| v as $t).collect();
                plan.process(&mut output, &mut workspace, &input)?;
                record(
                    &mut results,
                    $label,
                    format!("frame-{frame}-output"),
                    output.iter().map(|&v| v as f64).collect(),
                );
                record(
                    &mut results,
                    $label,
                    format!("frame-{frame}-state"),
                    workspace.state().iter().map(|&v| v as f64).collect(),
                );
            }
        }};
    }
    leaky!(f32, "leaky-f32");
    leaky!(f64, "leaky-f64");
    for case in f.gemv {
        let plan = PreparedGemv::new(case.rows, case.columns, Arc::from(case.matrix))?;
        let mut output = vec![0.0_f32; case.rows];
        plan.multiply(&mut output, &case.input)?;
        record(
            &mut results,
            "gemv-f32",
            case.name,
            output.into_iter().map(f64::from).collect(),
        );
    }
    let p = f.pixel;
    assert_eq!(p.flat.len(), p.rows * p.columns);
    let plan = PixelCalibrationPlan::<f32, u16>::new(p.flat, p.background)?;
    let mut workspace = plan.workspace();
    let mut output = vec![0.0_f32; p.rows * p.columns];
    for (frame, raw) in p.frames.iter().enumerate() {
        plan.process(&mut output, &mut workspace, raw)?;
        record(
            &mut results,
            "pixel-u16-f32",
            format!("frame-{frame}"),
            output.iter().copied().map(f64::from).collect(),
        );
    }
    Ok(Results { results })
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: numerical_parity <fixture.toml> [--conditioning-only]")?;
    let conditioning_only = match args.next().as_deref() {
        None => false,
        Some("--conditioning-only") => true,
        Some(option) => return Err(format!("unknown option: {option}").into()),
    };
    let path = std::path::Path::new(&path);
    if conditioning_only {
        let mut results = Vec::new();
        conditioning::run(&std::fs::read_to_string(path)?, true, &mut results)?;
        print!("{}", toml::to_string(&Results { results })?);
        return Ok(());
    }
    let fixture = toml::from_str(&std::fs::read_to_string(path)?)?;
    let mut results = run(fixture)?;
    catalog_additions::run(
        &std::fs::read_to_string(path.with_file_name("catalog_additions.toml"))?,
        &mut results.results,
    )?;
    chopped_acquisition::run(
        &std::fs::read_to_string(path.with_file_name("chopped_acquisition.toml"))?,
        &mut results.results,
    )?;
    pupil_flux::run(
        &std::fs::read_to_string(path.with_file_name("pupil_flux.toml"))?,
        &mut results.results,
    )?;

    sensing::run(
        &std::fs::read_to_string(path.with_file_name("sensing.toml"))?,
        &mut results.results,
    )?;
    calibration::run(
        &std::fs::read_to_string(path.with_file_name("calibration.toml"))?,
        &mut results.results,
    )?;
    vector::run(
        &std::fs::read_to_string(path.with_file_name("vector.toml"))?,
        &mut results.results,
    )?;
    optical_gain::run(
        &std::fs::read_to_string(path.with_file_name("optical_gain.toml"))?,
        &mut results.results,
    )?;
    docrime::run(
        &std::fs::read_to_string(path.with_file_name("docrime.toml"))?,
        &mut results.results,
    )?;
    frame_delay::run(
        &std::fs::read_to_string(path.with_file_name("frame_delay.toml"))?,
        &mut results.results,
    )?;
    conditioning::run(
        &std::fs::read_to_string(path.with_file_name("conditioning.toml"))?,
        false,
        &mut results.results,
    )?;
    pixel_f32::run(
        &std::fs::read_to_string(path.with_file_name("pixel_f32.toml"))?,
        &mut results.results,
    )?;
    biquad::run(
        &std::fs::read_to_string(path.with_file_name("biquad.toml"))?,
        &mut results.results,
    )?;
    closed_loop::run(
        &std::fs::read_to_string(path.with_file_name("closed_loop.toml"))?,
        &mut results.results,
    )?;
    weighted_cog::run(
        &std::fs::read_to_string(path.with_file_name("weighted_cog.toml"))?,
        &mut results.results,
    )?;
    corrector_allocation::run(
        &std::fs::read_to_string(path.with_file_name("corrector_allocation.toml"))?,
        &mut results.results,
    )?;
    weighted_shack::run(
        &std::fs::read_to_string(path.with_file_name("weighted_shack.toml"))?,
        &mut results.results,
    )?;
    scc_residual::run(
        &std::fs::read_to_string(path.with_file_name("scc_residual.toml"))?,
        &mut results.results,
    )?;
    shack_streaming::run(
        &std::fs::read_to_string(path.with_file_name("shack_streaming.toml"))?,
        &mut results.results,
    )?;
    ml_llowfs::run(
        &std::fs::read_to_string(path.with_file_name("ml_llowfs.toml"))?,
        &mut results.results,
    )?;
    shape_dense::run(
        &std::fs::read_to_string(path.with_file_name("shape_dense.toml"))?,
        &mut results.results,
    )?;
    pupil_signals::run(
        &std::fs::read_to_string(path.with_file_name("pupil_signals.toml"))?,
        &mut results.results,
    )?;
    scientific_contracts::run(
        &std::fs::read_to_string(path.with_file_name("scientific_contracts.toml"))?,
        &mut results.results,
    )?;
    scc_processing::run(
        &std::fs::read_to_string(path.with_file_name("scc_processing.toml"))?,
        &mut results.results,
    )?;
    frontends::run(
        &std::fs::read_to_string(path.with_file_name("frontends.toml"))?,
        &mut results.results,
    )?;
    dm_calibration::run(
        &std::fs::read_to_string(path.with_file_name("dm_calibration.toml"))?,
        &mut results.results,
    )?;
    pyramid_composites::run(
        &std::fs::read_to_string(path.with_file_name("pyramid_composites.toml"))?,
        &mut results.results,
    )?;
    print!("{}", toml::to_string(&results)?);
    Ok(())
}
