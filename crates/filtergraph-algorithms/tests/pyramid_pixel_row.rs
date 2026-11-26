//! Row-block pyramid normalization and declaration tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::convert::Infallible;

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, AlgorithmProcessResult, PortPublication,
    PyramidPixelImagePlan, PyramidPixelOutput, PyramidPixelRowF32, PyramidPixelRowF32Configuration,
    PyramidPixelRowPlan, RowBlockInfo, RowBlockResult, SampleMetadata,
};

const IMAGE_ROWS: usize = 8;
const IMAGE_COLUMNS: usize = 8;
const PUPIL_ROWS: usize = 2;
const PUPIL_COLUMNS: usize = 2;
const BLOCK_ROWS: usize = 2;
// Later pupil-major coordinates can become available first.
const ORIGINS: [(usize, usize); 4] = [(6, 0), (0, 0), (6, 4), (0, 4)];
const MASK: [bool; 4] = [true, false, true, true];

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static DEALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if COUNTING.get() {
            DEALLOCATIONS.set(DEALLOCATIONS.get() + 1);
        }
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocation_activity(callback: impl FnOnce()) -> (usize, usize) {
    ALLOCATIONS.set(0);
    DEALLOCATIONS.set(0);
    COUNTING.set(true);
    callback();
    COUNTING.set(false);
    (ALLOCATIONS.get(), DEALLOCATIONS.get())
}

fn image() -> Vec<f32> {
    (1..=IMAGE_ROWS * IMAGE_COLUMNS)
        .map(|value| value as f32)
        .collect()
}

fn image_plan(mask: Vec<bool>) -> PyramidPixelImagePlan<f32> {
    PyramidPixelImagePlan::from_detector_pupils(
        (IMAGE_ROWS, IMAGE_COLUMNS),
        (PUPIL_ROWS, PUPIL_COLUMNS),
        &ORIGINS,
        mask,
    )
    .unwrap()
}

fn block_info(sequence: u64, first_row: usize) -> RowBlockInfo {
    RowBlockInfo {
        sequence,
        first_row,
        terminal: first_row + BLOCK_ROWS == IMAGE_ROWS,
        discontinuity: false,
        corrupted: false,
    }
}

#[test]
fn row_plan_matches_full_frame_and_advances_complete_pupils() {
    let full = image_plan(MASK.to_vec());
    let row = PyramidPixelRowPlan::new(full.clone(), BLOCK_ROWS).unwrap();
    let pixels = image();
    let mut full_workspace = full.workspace();
    let mut row_workspace = row.workspace();
    let mut full_pixels = vec![0.0; 12];
    let mut row_pixels = vec![-1.0; 12];
    let mut full_mean = 0.0;
    let mut row_mean = -1.0;

    for sequence in 1..=2 {
        full.process(
            PyramidPixelOutput::new(&mut full_pixels, &mut full_mean),
            &mut full_workspace,
            &pixels,
        )
        .unwrap();
        for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
            let result = row
                .process_block(
                    &mut row_pixels,
                    &mut row_mean,
                    &mut row_workspace,
                    &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
                    block_info(sequence, first_row),
                )
                .unwrap();
            if first_row == 0 {
                assert_eq!(row_workspace.processed_pupils(), 2);
            }
            if first_row + BLOCK_ROWS < IMAGE_ROWS {
                assert_eq!(result, RowBlockResult::Pending);
                if sequence == 1 {
                    assert_eq!(row_pixels, vec![-1.0; 12]);
                    assert_eq!(row_mean, -1.0);
                }
            } else {
                assert_eq!(
                    result,
                    RowBlockResult::Complete {
                        sequence,
                        discontinuity: false,
                    }
                );
            }
        }
        assert_eq!(row_pixels, full_pixels);
        assert_eq!(row_mean, full_mean);
        assert_eq!(
            row_workspace.reciprocal_previous_frame_mean(),
            full_workspace.reciprocal_previous_frame_mean()
        );
    }
}

#[test]
fn complete_pupils_are_exposed_without_waiting_for_a_declaration_prefix() {
    let full = image_plan(MASK.to_vec());
    let row = PyramidPixelRowPlan::new(full, BLOCK_ROWS).unwrap();
    let pixels = image();
    let mut workspace = row.workspace();
    let mut output = vec![-1.0; 12];
    let mut mean = -1.0;
    let mut ready = Vec::new();

    for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
        row.process_block_with_ready_pixels(
            &mut output,
            &mut mean,
            &mut workspace,
            &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
            block_info(1, first_row),
            |_plan, _normalized, ranges| {
                ready.extend(ranges.iter().cloned().map(|columns| (first_row, columns)));
                Ok::<(), Infallible>(())
            },
        )
        .unwrap();
    }

    assert_eq!(ready, [(0, 3..6), (0, 9..12), (6, 0..3), (6, 6..9)]);
}

#[test]
fn complete_rows_are_exposed_before_their_pupil_is_complete() {
    let origins = [(4, 0), (0, 0), (4, 4), (0, 4)];
    let full = PyramidPixelImagePlan::from_detector_pupils(
        (IMAGE_ROWS, IMAGE_COLUMNS),
        (4, 2),
        &origins,
        vec![true; 8],
    )
    .unwrap();
    let row = PyramidPixelRowPlan::new(full.clone(), BLOCK_ROWS).unwrap();
    let pixels = image();
    let mut workspace = row.workspace();
    let mut output = vec![-1.0; 32];
    let mut mean = -1.0;
    let mut ready = Vec::new();

    for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
        row.process_block_with_ready_pixels(
            &mut output,
            &mut mean,
            &mut workspace,
            &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
            block_info(1, first_row),
            |_plan, _normalized, ranges| {
                ready.extend(ranges.iter().cloned().map(|columns| (first_row, columns)));
                Ok::<(), Infallible>(())
            },
        )
        .unwrap();
    }

    assert_eq!(
        ready,
        [
            (0, 8..12),
            (0, 24..28),
            (2, 12..16),
            (2, 28..32),
            (4, 0..4),
            (4, 16..20),
            (6, 4..8),
            (6, 20..24),
        ]
    );

    let mut expected_workspace = full.workspace();
    let mut expected_output = vec![0.0; 32];
    let mut expected_mean = 0.0;
    full.process(
        PyramidPixelOutput::new(&mut expected_output, &mut expected_mean),
        &mut expected_workspace,
        &pixels,
    )
    .unwrap();
    assert_eq!(output, expected_output);
    assert_eq!(mean.to_bits(), expected_mean.to_bits());
}

#[test]
fn warmed_row_processing_allocates_and_destroys_nothing() {
    let row = PyramidPixelRowPlan::new(image_plan(MASK.to_vec()), BLOCK_ROWS).unwrap();
    let pixels = image();
    let mut workspace = row.workspace();
    let mut output = vec![0.0; 12];
    let mut mean = 0.0;

    for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
        row.process_block(
            &mut output,
            &mut mean,
            &mut workspace,
            &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
            block_info(1, first_row),
        )
        .unwrap();
    }
    assert_eq!(
        allocation_activity(|| {
            for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
                row.process_block(
                    &mut output,
                    &mut mean,
                    &mut workspace,
                    &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
                    block_info(2, first_row),
                )
                .unwrap();
            }
        }),
        (0, 0)
    );
}

#[test]
fn declaration_preserves_gain_when_a_plan_change_abandons_a_frame() {
    let mut declared = PyramidPixelRowF32::prepare(PyramidPixelRowF32Configuration {
        row_block_rows: BLOCK_ROWS,
        pupil_origins: ORIGINS.map(|(row, column)| [row, column]).to_vec(),
        pupil_mask: Some(MASK.to_vec()),
        image_size: (IMAGE_ROWS, IMAGE_COLUMNS),
        pupil_size: (PUPIL_ROWS, PUPIL_COLUMNS),
    })
    .unwrap();
    assert_eq!(declared.formats()[0].rate_ratio(), (4, 1));
    assert_eq!(declared.formats()[1].rate_ratio(), (1, 1));
    assert_eq!(declared.formats()[2].rate_ratio(), (1, 1));
    assert_eq!(
        <PyramidPixelRowF32 as AlgorithmDeclaration>::PORTS[1].publication(),
        PortPublication::Conditional
    );
    assert_eq!(
        <PyramidPixelRowF32 as AlgorithmDeclaration>::PORTS[2].publication(),
        PortPublication::Conditional
    );

    let pixels = image();
    let mut output = vec![-1.0; 12];
    let mut mean = [-1.0];
    for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
        let sample = SampleMetadata::new(10, first_row as u32)
            .with_terminal(first_row + BLOCK_ROWS == IMAGE_ROWS);
        let result = PyramidPixelRowF32::process_with_metadata(
            &mut declared,
            &[Some(sample)],
            &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
            &mut output,
            &mut mean,
        )
        .unwrap();
        if first_row + BLOCK_ROWS == IMAGE_ROWS {
            assert!(matches!(result, AlgorithmProcessResult::Complete { .. }));
        } else {
            assert_eq!(result, AlgorithmProcessResult::Deferred);
        }
    }
    assert!(mean[0] > 0.0);
    assert_eq!(output, vec![0.0; 12]);

    let sample = SampleMetadata::new(11, 0);
    assert_eq!(
        PyramidPixelRowF32::process_with_metadata(
            &mut declared,
            &[Some(sample)],
            &pixels[..BLOCK_ROWS * IMAGE_COLUMNS],
            &mut output,
            &mut mean,
        )
        .unwrap(),
        AlgorithmProcessResult::Deferred
    );
    let replacement_mask = [false, true, true, true];
    assert_eq!(
        declared.replace_parameter(4, AlgorithmInput::from(&replacement_mask[..])),
        Some(Ok(()))
    );
    for first_row in (BLOCK_ROWS..IMAGE_ROWS).step_by(BLOCK_ROWS) {
        let sample = SampleMetadata::new(11, first_row as u32)
            .with_terminal(first_row + BLOCK_ROWS == IMAGE_ROWS);
        assert_eq!(
            PyramidPixelRowF32::process_with_metadata(
                &mut declared,
                &[Some(sample)],
                &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
                &mut output,
                &mut mean,
            )
            .unwrap(),
            AlgorithmProcessResult::Deferred
        );
    }

    let previous_output = output.clone();
    for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
        let sample = SampleMetadata::new(12, first_row as u32)
            .with_terminal(first_row + BLOCK_ROWS == IMAGE_ROWS);
        let result = PyramidPixelRowF32::process_with_metadata(
            &mut declared,
            &[Some(sample)],
            &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
            &mut output,
            &mut mean,
        )
        .unwrap();
        if first_row + BLOCK_ROWS == IMAGE_ROWS {
            assert_eq!(
                result,
                AlgorithmProcessResult::complete_with_metadata(
                    SampleMetadata::new(12, 0)
                        .with_terminal(true)
                        .with_discontinuity(true)
                )
            );
        } else {
            assert_eq!(result, AlgorithmProcessResult::Deferred);
            assert_eq!(output, previous_output);
        }
    }
    assert!(output.iter().any(|value| *value != 0.0));
}

#[test]
fn row_quantum_and_nonfinite_normalization_report_the_failing_contract() {
    use filtergraph_algorithms::{
        PyramidPixelError, PyramidPixelRowError, PyramidPixelRowPlanError,
    };
    let image = image_plan(MASK.to_vec());
    for (rows, expected, detail) in [
        (
            0,
            PyramidPixelRowPlanError::BlockRows,
            "height must be positive",
        ),
        (
            8,
            PyramidPixelRowPlanError::BlockRows,
            "smaller than image height",
        ),
        (
            3,
            PyramidPixelRowPlanError::BlockRowsDoNotDivideImage,
            "must divide image height",
        ),
    ] {
        let error = PyramidPixelRowPlan::new(image.clone(), rows).unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
    }

    let plan = PyramidPixelRowPlan::new(image, BLOCK_ROWS).unwrap();
    let mut workspace = plan.workspace();
    let mut pixels = [19.0; 12];
    let mut mean = 19.0;
    let tiny = [f32::MIN_POSITIVE; BLOCK_ROWS * IMAGE_COLUMNS];
    for row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
        plan.process_block(
            &mut pixels,
            &mut mean,
            &mut workspace,
            &tiny,
            block_info(100, row),
        )
        .unwrap();
    }
    let previous = workspace.reciprocal_previous_frame_mean();
    let published = (pixels, mean);
    let mut invalid = [2.0; BLOCK_ROWS * IMAGE_COLUMNS];
    invalid[0] = f32::MAX;
    let error = plan
        .process_block(
            &mut pixels,
            &mut mean,
            &mut workspace,
            &invalid,
            block_info(101, 0),
        )
        .unwrap_err();
    assert_eq!(
        error,
        PyramidPixelRowError::Pyramid(PyramidPixelError::NonFiniteOutput { index: 3 })
    );
    assert!(error.to_string().contains("3"));
    assert_eq!(workspace.sequence(), None);
    assert_eq!(workspace.reciprocal_previous_frame_mean(), previous);
    assert_eq!((pixels, mean), published);
}
