//! Multi-region detector-readout service-time benchmark.

use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

use filtergraph_algorithms::{
    AlgorithmPlan, DetectorReadoutBlockInfo, DetectorReadoutCalibrationPlan, DetectorReadoutPlan,
    DetectorReadoutRegion, DetectorReadoutResult, PixelCalibrationPlan, PreparedGemv,
    PyramidPixelImagePlan, PyramidPixelOutput, PyramidReadoutReconstructionOutput,
    PyramidReadoutReconstructionPlan, ShackHartmannImagePlan, ShackHartmannOutput,
    ShackHartmannReadoutReconstructionPlan,
};

const SHWFS_ROWS: usize = 352;
const SHWFS_COLUMNS: usize = 352;
const SHWFS_SUBAPERTURE_ROWS: usize = 22;
const SHWFS_SUBAPERTURE_COLUMNS: usize = 22;
const SHWFS_SUBAPERTURES: usize = 256;
const SHWFS_RECONSTRUCTED: usize = 277;

const PWFS_ROWS: usize = 64;
const PWFS_COLUMNS: usize = 64;
const PWFS_PUPIL_ROWS: usize = 30;
const PWFS_PUPIL_COLUMNS: usize = 30;
const PWFS_RECONSTRUCTED: usize = 253;
const PWFS_ORIGINS: [(usize, usize); 4] = [(33, 34), (0, 0), (34, 1), (0, 34)];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Sensor {
    ShackHartmann,
    Pyramid,
}

impl Sensor {
    fn parse(value: &str) -> Self {
        match value {
            "shwfs" => Self::ShackHartmann,
            "pwfs" => Self::Pyramid,
            _ => panic!("--sensor must be shwfs or pwfs"),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::ShackHartmann => "shwfs",
            Self::Pyramid => "pwfs",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Readout {
    Single,
    Opposed,
    Eight,
}

impl Readout {
    fn parse(value: &str) -> Self {
        match value {
            "single" => Self::Single,
            "opposed" => Self::Opposed,
            "eight" => Self::Eight,
            _ => panic!("--readout must be single, opposed, or eight"),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Opposed => "opposed",
            Self::Eight => "eight",
        }
    }

    fn plan(self, detector: (usize, usize), blocks: usize) -> DetectorReadoutPlan {
        let (rows, columns) = detector;
        assert!(blocks > 0);
        let (region_lines, region_samples, regions) = match self {
            Self::Single => (
                rows,
                columns,
                vec![DetectorReadoutRegion::new((0, 0), (1, 0), (0, 1))],
            ),
            Self::Opposed => {
                assert!(rows.is_multiple_of(2));
                (
                    rows / 2,
                    columns,
                    vec![
                        DetectorReadoutRegion::new((0, 0), (1, 0), (0, 1)),
                        DetectorReadoutRegion::new((rows - 1, 0), (-1, 0), (0, 1)),
                    ],
                )
            }
            Self::Eight => {
                assert!(columns.is_multiple_of(8));
                let region_columns = columns / 8;
                let mut regions = Vec::with_capacity(8);
                for region in 0..8 {
                    let reverse_lines = region % 2 != 0;
                    let reverse_samples = region % 4 >= 2;
                    let first_column = region * region_columns;
                    regions.push(DetectorReadoutRegion::new(
                        (
                            if reverse_lines { rows - 1 } else { 0 },
                            if reverse_samples {
                                first_column + region_columns - 1
                            } else {
                                first_column
                            },
                        ),
                        (if reverse_lines { -1 } else { 1 }, 0),
                        (0, if reverse_samples { -1 } else { 1 }),
                    ));
                }
                (rows, region_columns, regions)
            }
        };
        assert!(region_lines.is_multiple_of(blocks));
        DetectorReadoutPlan::new(
            detector,
            region_lines,
            region_lines / blocks,
            region_samples,
            regions,
        )
        .unwrap()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Both,
    Complete,
    Readout,
}

impl Phase {
    fn parse(value: &str) -> Self {
        match value {
            "both" => Self::Both,
            "complete" => Self::Complete,
            "readout" => Self::Readout,
            _ => panic!("--phase must be both, complete, or readout"),
        }
    }

    const fn includes_complete(self) -> bool {
        matches!(self, Self::Both | Self::Complete)
    }

    const fn includes_readout(self) -> bool {
        matches!(self, Self::Both | Self::Readout)
    }
}

#[derive(Debug)]
struct Options {
    sensor: Sensor,
    readout: Readout,
    blocks: usize,
    samples: usize,
    warmup: usize,
    progressive_first: bool,
    profile: bool,
    phase: Phase,
}

impl Options {
    fn parse() -> Self {
        let mut options = Self {
            sensor: Sensor::ShackHartmann,
            readout: Readout::Opposed,
            blocks: 16,
            samples: 1000,
            warmup: 1000,
            progressive_first: false,
            profile: false,
            phase: Phase::Both,
        };
        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--sensor" => options.sensor = Sensor::parse(&arguments.next().unwrap()),
                "--readout" => options.readout = Readout::parse(&arguments.next().unwrap()),
                "--blocks" => options.blocks = positive(&arguments.next().unwrap()),
                "--samples" => options.samples = positive(&arguments.next().unwrap()),
                "--warmup" => options.warmup = positive(&arguments.next().unwrap()),
                "--phase" => options.phase = Phase::parse(&arguments.next().unwrap()),
                "--progressive-first" => options.progressive_first = true,
                "--profile" => options.profile = true,
                // Cargo supplies this flag to custom benchmark harnesses.
                "--bench" => {}
                "--help" => {
                    println!(
                        "detector_readout [--sensor shwfs|pwfs] \
                         [--readout single|opposed|eight] [--blocks N] \
                         [--samples N] [--warmup N] [--progressive-first] \
                         [--profile] [--phase both|complete|readout]"
                    );
                    std::process::exit(0);
                }
                _ => panic!("unknown argument: {argument}"),
            }
        }
        if options.phase != Phase::Both && !options.profile {
            panic!("--phase complete or readout requires --profile");
        }
        options
    }
}

fn positive(value: &str) -> usize {
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap()
}

enum Workload {
    ShackHartmann(Box<ShackHartmannWorkload>),
    Pyramid(Box<PyramidWorkload>),
}

impl Workload {
    fn new(options: &Options) -> Self {
        match options.sensor {
            Sensor::ShackHartmann => Self::ShackHartmann(Box::new(ShackHartmannWorkload::new(
                options.readout,
                options.blocks,
            ))),
            Sensor::Pyramid => Self::Pyramid(Box::new(PyramidWorkload::new(
                options.readout,
                options.blocks,
            ))),
        }
    }

    fn verify(&mut self) {
        match self {
            Self::ShackHartmann(workload) => workload.verify(),
            Self::Pyramid(workload) => workload.verify(),
        }
    }

    fn blocks(&self) -> usize {
        match self {
            Self::ShackHartmann(workload) => workload.readout.blocks(),
            Self::Pyramid(workload) => workload.readout.blocks(),
        }
    }

    fn complete(&mut self, sequence: u64) {
        match self {
            Self::ShackHartmann(workload) => workload.complete(sequence),
            Self::Pyramid(workload) => workload.complete(sequence),
        }
    }

    fn readout_block(&mut self, sequence: u64, ordinal: usize) {
        match self {
            Self::ShackHartmann(workload) => workload.readout_block(sequence, ordinal),
            Self::Pyramid(workload) => workload.readout_block(sequence, ordinal),
        }
    }
}

struct ShackHartmannWorkload {
    raw: Box<[u16]>,
    packed: Box<[u16]>,
    readout: DetectorReadoutPlan,
    full_calibration: PixelCalibrationPlan<f32, u16>,
    full_calibration_workspace: filtergraph_algorithms::ProgressWorkspace,
    full_pixels: Box<[f32]>,
    sensing: ShackHartmannImagePlan<f32>,
    sensing_workspace: filtergraph_algorithms::ProgressWorkspace,
    slopes: Box<[f32]>,
    flux: Box<[f32]>,
    validity: Box<[bool]>,
    reconstructor: PreparedGemv<f32>,
    full_output: Box<[f32]>,
    calibration: DetectorReadoutCalibrationPlan<f32, u16>,
    calibrated_block: Box<[f32]>,
    progressive: ShackHartmannReadoutReconstructionPlan<f32>,
    progressive_workspace: filtergraph_algorithms::ShackHartmannReadoutReconstructionWorkspace<f32>,
    progressive_output: Box<[f32]>,
}

impl ShackHartmannWorkload {
    fn new(readout_kind: Readout, blocks: usize) -> Self {
        let readout = readout_kind.plan((SHWFS_ROWS, SHWFS_COLUMNS), blocks);
        let raw = (0..SHWFS_ROWS * SHWFS_COLUMNS)
            .map(|index| {
                let row = index / SHWFS_COLUMNS % SHWFS_SUBAPERTURE_ROWS;
                let column = index % SHWFS_COLUMNS % SHWFS_SUBAPERTURE_COLUMNS;
                (100 + 2 * row + column) as u16
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let packed = readout.pack_detector(&raw).unwrap();
        let flat = vec![1.0_f32; raw.len()];
        let background = vec![0.0_f32; raw.len()];
        let origins = (0..SHWFS_SUBAPERTURES)
            .map(|index| {
                (
                    index / 16 * SHWFS_SUBAPERTURE_ROWS,
                    index % 16 * SHWFS_SUBAPERTURE_COLUMNS,
                )
            })
            .collect::<Vec<_>>();
        let sensing = ShackHartmannImagePlan::from_detector_regions(
            (SHWFS_ROWS, SHWFS_COLUMNS),
            (SHWFS_SUBAPERTURE_ROWS, SHWFS_SUBAPERTURE_COLUMNS),
            &origins,
            1.0,
            0.0,
            1.0,
            None,
            None,
        )
        .unwrap();
        let slope_count = SHWFS_SUBAPERTURES * 2;
        let matrix = matrix(SHWFS_RECONSTRUCTED, slope_count);
        let reconstructor =
            PreparedGemv::new(SHWFS_RECONSTRUCTED, slope_count, Arc::clone(&matrix)).unwrap();
        let progressive = ShackHartmannReadoutReconstructionPlan::new(
            sensing.clone(),
            PreparedGemv::new(SHWFS_RECONSTRUCTED, slope_count, matrix).unwrap(),
            readout.clone(),
        )
        .unwrap();
        Self {
            raw,
            packed,
            full_calibration: PixelCalibrationPlan::new(flat.clone(), background.clone()).unwrap(),
            full_calibration_workspace: filtergraph_algorithms::ProgressWorkspace::new(),
            full_pixels: vec![0.0; SHWFS_ROWS * SHWFS_COLUMNS].into_boxed_slice(),
            sensing_workspace: sensing.workspace(),
            sensing,
            slopes: vec![0.0; slope_count].into_boxed_slice(),
            flux: vec![0.0; SHWFS_SUBAPERTURES].into_boxed_slice(),
            validity: vec![false; SHWFS_SUBAPERTURES].into_boxed_slice(),
            reconstructor,
            full_output: vec![0.0; SHWFS_RECONSTRUCTED].into_boxed_slice(),
            calibration: DetectorReadoutCalibrationPlan::new(readout.clone(), &flat, &background)
                .unwrap(),
            calibrated_block: vec![0.0; readout.block_elements()].into_boxed_slice(),
            progressive_workspace: progressive.workspace(),
            progressive,
            progressive_output: vec![0.0; SHWFS_RECONSTRUCTED].into_boxed_slice(),
            readout,
        }
    }

    fn complete(&mut self, _sequence: u64) {
        self.full_calibration
            .process(
                &mut self.full_pixels,
                &mut self.full_calibration_workspace,
                &self.raw,
            )
            .unwrap();
        self.sensing
            .process(
                ShackHartmannOutput::new(&mut self.slopes, &mut self.flux, &mut self.validity)
                    .unwrap(),
                &mut self.sensing_workspace,
                &self.full_pixels,
            )
            .unwrap();
        self.reconstructor
            .multiply(&mut self.full_output, &self.slopes)
            .unwrap();
        black_box(&self.full_output);
    }

    fn readout_block(&mut self, sequence: u64, ordinal: usize) {
        let start = ordinal * self.readout.block_elements();
        let info = block_info(sequence, ordinal, self.readout.blocks());
        self.calibration
            .process_block(
                &mut self.calibrated_block,
                &self.packed[start..start + self.readout.block_elements()],
                info,
            )
            .unwrap();
        let result = self
            .progressive
            .process_block(
                &mut self.progressive_output,
                &mut self.progressive_workspace,
                &self.calibrated_block,
                info,
            )
            .unwrap();
        assert_publication(result, sequence, ordinal, self.readout.blocks());
        black_box(&self.progressive_output);
    }

    fn verify(&mut self) {
        self.complete(1);
        for ordinal in 0..self.readout.blocks() {
            self.readout_block(1, ordinal);
        }
        assert_close(&self.progressive_output, &self.full_output);
    }
}

struct PyramidWorkload {
    raw: Box<[u16]>,
    packed: Box<[u16]>,
    readout: DetectorReadoutPlan,
    full_calibration: PixelCalibrationPlan<f32, u16>,
    full_calibration_workspace: filtergraph_algorithms::ProgressWorkspace,
    full_pixels: Box<[f32]>,
    sensing: PyramidPixelImagePlan<f32>,
    sensing_workspace: filtergraph_algorithms::PyramidPixelWorkspace<f32>,
    reconstruction_pixels: Box<[f32]>,
    mean: f32,
    reconstructor: PreparedGemv<f32>,
    full_output: Box<[f32]>,
    calibration: DetectorReadoutCalibrationPlan<f32, u16>,
    calibrated_block: Box<[f32]>,
    progressive: PyramidReadoutReconstructionPlan<f32>,
    progressive_workspace: filtergraph_algorithms::PyramidReadoutReconstructionWorkspace<f32>,
    progressive_pixels: Box<[f32]>,
    progressive_mean: f32,
    progressive_output: Box<[f32]>,
}

impl PyramidWorkload {
    fn new(readout_kind: Readout, blocks: usize) -> Self {
        let readout = readout_kind.plan((PWFS_ROWS, PWFS_COLUMNS), blocks);
        let raw = (0..PWFS_ROWS * PWFS_COLUMNS)
            .map(|index| (100 + 3 * (index / PWFS_COLUMNS) + 2 * (index % PWFS_COLUMNS)) as u16)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let packed = readout.pack_detector(&raw).unwrap();
        let flat = vec![1.0_f32; raw.len()];
        let background = vec![0.0_f32; raw.len()];
        let sensing = PyramidPixelImagePlan::from_detector_pupils(
            (PWFS_ROWS, PWFS_COLUMNS),
            (PWFS_PUPIL_ROWS, PWFS_PUPIL_COLUMNS),
            &PWFS_ORIGINS,
            vec![true; PWFS_PUPIL_ROWS * PWFS_PUPIL_COLUMNS],
        )
        .unwrap();
        let selected = sensing.normalization().output_pixel_count();
        let matrix = matrix(PWFS_RECONSTRUCTED, selected);
        let reconstructor =
            PreparedGemv::new(PWFS_RECONSTRUCTED, selected, Arc::clone(&matrix)).unwrap();
        let progressive = PyramidReadoutReconstructionPlan::new(
            sensing.clone(),
            PreparedGemv::new(PWFS_RECONSTRUCTED, selected, matrix).unwrap(),
            readout.clone(),
        )
        .unwrap();
        Self {
            raw,
            packed,
            full_calibration: PixelCalibrationPlan::new(flat.clone(), background.clone()).unwrap(),
            full_calibration_workspace: filtergraph_algorithms::ProgressWorkspace::new(),
            full_pixels: vec![0.0; PWFS_ROWS * PWFS_COLUMNS].into_boxed_slice(),
            sensing_workspace: sensing.workspace(),
            sensing,
            reconstruction_pixels: vec![0.0; selected].into_boxed_slice(),
            mean: 0.0,
            reconstructor,
            full_output: vec![0.0; PWFS_RECONSTRUCTED].into_boxed_slice(),
            calibration: DetectorReadoutCalibrationPlan::new(readout.clone(), &flat, &background)
                .unwrap(),
            calibrated_block: vec![0.0; readout.block_elements()].into_boxed_slice(),
            progressive_workspace: progressive.workspace(),
            progressive,
            progressive_pixels: vec![0.0; selected].into_boxed_slice(),
            progressive_mean: 0.0,
            progressive_output: vec![0.0; PWFS_RECONSTRUCTED].into_boxed_slice(),
            readout,
        }
    }

    fn complete(&mut self, _sequence: u64) {
        self.full_calibration
            .process(
                &mut self.full_pixels,
                &mut self.full_calibration_workspace,
                &self.raw,
            )
            .unwrap();
        self.sensing
            .process(
                PyramidPixelOutput::new(&mut self.reconstruction_pixels, &mut self.mean),
                &mut self.sensing_workspace,
                &self.full_pixels,
            )
            .unwrap();
        self.reconstructor
            .multiply(&mut self.full_output, &self.reconstruction_pixels)
            .unwrap();
        black_box(&self.full_output);
    }

    fn readout_block(&mut self, sequence: u64, ordinal: usize) {
        let start = ordinal * self.readout.block_elements();
        let info = block_info(sequence, ordinal, self.readout.blocks());
        self.calibration
            .process_block(
                &mut self.calibrated_block,
                &self.packed[start..start + self.readout.block_elements()],
                info,
            )
            .unwrap();
        let result = self
            .progressive
            .process_block(
                PyramidReadoutReconstructionOutput::new(
                    &mut self.progressive_pixels,
                    &mut self.progressive_mean,
                    &mut self.progressive_output,
                ),
                &mut self.progressive_workspace,
                &self.calibrated_block,
                info,
            )
            .unwrap();
        assert_publication(result, sequence, ordinal, self.readout.blocks());
        black_box(&self.progressive_output);
    }

    fn verify(&mut self) {
        for sequence in 1..=2 {
            self.complete(sequence);
            for ordinal in 0..self.readout.blocks() {
                self.readout_block(sequence, ordinal);
            }
        }
        assert_eq!(self.progressive_pixels, self.reconstruction_pixels);
        assert_eq!(self.progressive_mean, self.mean);
        assert_close(&self.progressive_output, &self.full_output);
    }
}

fn matrix(rows: usize, columns: usize) -> Arc<[f32]> {
    Arc::from(
        (0..rows * columns)
            .map(|index| {
                let coefficient = ((index / columns * 17 + index % columns * 13) % 31) as i32 - 15;
                coefficient as f32 * 0.000_01
            })
            .collect::<Vec<_>>(),
    )
}

fn block_info(sequence: u64, ordinal: usize, blocks: usize) -> DetectorReadoutBlockInfo {
    DetectorReadoutBlockInfo {
        sequence,
        ordinal,
        terminal: ordinal + 1 == blocks,
        discontinuity: false,
        corrupted: false,
    }
}

fn assert_publication(result: DetectorReadoutResult, sequence: u64, ordinal: usize, blocks: usize) {
    if ordinal + 1 == blocks {
        assert_eq!(
            result,
            DetectorReadoutResult::Complete {
                sequence,
                discontinuity: false,
            }
        );
    } else {
        assert_eq!(result, DetectorReadoutResult::Pending);
    }
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 2.0e-6 * (1.0 + expected.abs()),
            "{actual} does not match {expected}"
        );
    }
}

fn clock_overhead_ns() -> u64 {
    let mut values = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let start = Instant::now();
        values.push(start.elapsed().as_nanos() as u64);
    }
    values.sort_unstable();
    values[values.len() / 2]
}

fn report(name: &str, values: &mut [u64]) -> u64 {
    values.sort_unstable();
    let sum = values.iter().map(|&value| value as u128).sum::<u128>();
    let p50 = values[values.len() / 2];
    print!(
        "{name}: n={} mean={:.1} ns p50={} ns",
        values.len(),
        sum as f64 / values.len() as f64,
        p50
    );
    if values.len() >= 10 {
        print!(" p90={} ns", nearest_rank(values, 90, 100));
    }
    if values.len() >= 100 {
        print!(" p99={} ns", nearest_rank(values, 99, 100));
    }
    if values.len() >= 1000 {
        print!(" p99.9={} ns", nearest_rank(values, 999, 1000));
    }
    println!(" max={} ns", values[values.len() - 1]);
    p50
}

fn nearest_rank(values: &[u64], numerator: usize, denominator: usize) -> u64 {
    values[(values.len() * numerator).div_ceil(denominator) - 1]
}

fn process_complete_frames(workload: &mut Workload, sequence: &mut u64, count: usize) {
    for _ in 0..count {
        *sequence += 1;
        workload.complete(*sequence);
    }
}

fn process_readout_frames(workload: &mut Workload, sequence: &mut u64, count: usize) {
    let blocks = workload.blocks();
    for _ in 0..count {
        *sequence += 1;
        for ordinal in 0..blocks {
            workload.readout_block(*sequence, ordinal);
        }
    }
}

fn profile(options: &Options, workload: &mut Workload) {
    let mut sequence = 10;
    if options.phase.includes_complete() {
        process_complete_frames(workload, &mut sequence, options.warmup);
        let start = Instant::now();
        process_complete_frames(workload, &mut sequence, options.samples);
        println!(
            "complete-frame profile phase: {} frames in {} ns",
            options.samples,
            start.elapsed().as_nanos()
        );
    }
    if options.phase.includes_readout() {
        process_readout_frames(workload, &mut sequence, options.warmup);
        let start = Instant::now();
        process_readout_frames(workload, &mut sequence, options.samples);
        println!(
            "detector-readout profile phase: {} frames in {} ns",
            options.samples,
            start.elapsed().as_nanos()
        );
    }
}

fn measure(options: &Options, workload: &mut Workload) {
    let blocks = workload.blocks();
    let mut complete = vec![0_u64; options.samples];
    let mut readout_total = vec![0_u64; options.samples];
    let mut readout_terminal = vec![0_u64; options.samples];
    let mut readout_maximum = vec![0_u64; options.samples];
    let mut readout_blocks = vec![0_u64; options.samples * blocks];
    let mut sequence = 10;

    for phase in 0..2 {
        let readout_phase = (phase == 0) == options.progressive_first;
        if readout_phase {
            process_readout_frames(workload, &mut sequence, options.warmup);
            for sample in 0..options.samples {
                sequence += 1;
                for ordinal in 0..blocks {
                    let start = Instant::now();
                    workload.readout_block(sequence, ordinal);
                    let elapsed = start.elapsed().as_nanos() as u64;
                    readout_blocks[sample * blocks + ordinal] = elapsed;
                    readout_total[sample] += elapsed;
                    readout_maximum[sample] = readout_maximum[sample].max(elapsed);
                    if ordinal + 1 == blocks {
                        readout_terminal[sample] = elapsed;
                    }
                }
            }
        } else {
            process_complete_frames(workload, &mut sequence, options.warmup);
            for elapsed in &mut complete {
                sequence += 1;
                let start = Instant::now();
                workload.complete(sequence);
                *elapsed = start.elapsed().as_nanos() as u64;
            }
        }
    }

    write_csv(
        &complete,
        &readout_total,
        &readout_terminal,
        &readout_maximum,
        &readout_blocks,
        blocks,
    );
    let complete_p50 = report("complete-frame service", &mut complete);
    let total_p50 = report("detector-readout total service", &mut readout_total);
    let terminal_p50 = report(
        "detector-readout terminal-block service",
        &mut readout_terminal,
    );
    report(
        "detector-readout maximum-block service",
        &mut readout_maximum,
    );
    println!(
        "median total-service ratio: {:.4} readout/complete ({:+.2}%)",
        total_p50 as f64 / complete_p50 as f64,
        100.0 * (total_p50 as f64 / complete_p50 as f64 - 1.0)
    );
    println!(
        "median post-readout service ratio: {:.4} terminal/complete ({:+.2}%)",
        terminal_p50 as f64 / complete_p50 as f64,
        100.0 * (terminal_p50 as f64 / complete_p50 as f64 - 1.0)
    );
}

fn write_csv(
    complete: &[u64],
    readout_total: &[u64],
    readout_terminal: &[u64],
    readout_maximum: &[u64],
    readout_blocks: &[u64],
    blocks: usize,
) {
    let Ok(path) = env::var("FILTERGRAPH_DETECTOR_READOUT_BENCHMARK_CSV") else {
        return;
    };
    let mut output = BufWriter::new(File::create(path).unwrap());
    write!(
        output,
        "iteration,complete_frame_ns,readout_total_ns,readout_terminal_ns,readout_max_ns"
    )
    .unwrap();
    for ordinal in 0..blocks {
        write!(output, ",readout_block_{ordinal}_ns").unwrap();
    }
    writeln!(output).unwrap();
    for sample in 0..complete.len() {
        write!(
            output,
            "{sample},{},{},{},{}",
            complete[sample],
            readout_total[sample],
            readout_terminal[sample],
            readout_maximum[sample]
        )
        .unwrap();
        for ordinal in 0..blocks {
            write!(output, ",{}", readout_blocks[sample * blocks + ordinal]).unwrap();
        }
        writeln!(output).unwrap();
    }
    output.flush().unwrap();
}

fn main() {
    let options = Options::parse();
    let mut workload = Workload::new(&options);
    assert_eq!(workload.blocks(), options.blocks);
    workload.verify();
    println!(
        "numerical-equivalence: {} complete-frame and detector-readout outputs agree",
        options.sensor.label()
    );
    println!(
        "workload: sensor={} readout={} blocks={} warmup={} samples={} profile={} phase={:?}",
        options.sensor.label(),
        options.readout.label(),
        options.blocks,
        options.warmup,
        options.samples,
        options.profile,
        options.phase
    );
    println!(
        "boundary: warmed portable Rust calibration, sensing, and reconstruction; detector arrival, PipeWire scheduling, publication, and preparation excluded"
    );
    if options.profile {
        println!("clock-read overhead probe: skipped in profile mode");
        profile(&options, &mut workload);
    } else {
        println!("clock-read p50 overhead: {} ns", clock_overhead_ns());
        measure(&options, &mut workload);
    }
}
