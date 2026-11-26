//! Progressive pyramid normalization and reconstruction tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmOperation, AlgorithmPlan, AlgorithmProcessResult, PreparedGemv,
    PyramidPixelImagePlan, PyramidPixelOutput, PyramidRowReconstructionExecutionError,
    PyramidRowReconstructionOutput, PyramidRowReconstructionPlan, PyramidRowReconstructorF32,
    PyramidRowReconstructorF32Configuration, RowBlockInfo, RowBlockResult, SampleMetadata,
};

const IMAGE_ROWS: usize = 8;
const IMAGE_COLUMNS: usize = 8;
const PUPIL_ROWS: usize = 2;
const PUPIL_COLUMNS: usize = 2;
const BLOCK_ROWS: usize = 2;
const RECONSTRUCTED: usize = 5;
// Pupil-major order intentionally differs from detector arrival order.
const ORIGINS: [(usize, usize); 4] = [(6, 0), (0, 0), (6, 4), (0, 4)];
const MASK: [bool; 4] = [true, false, true, true];
const SELECTED_PIXELS: usize = 12;

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

fn image(sequence: usize) -> Vec<f32> {
    (1..=IMAGE_ROWS * IMAGE_COLUMNS)
        .map(|value| value as f32 + sequence as f32 * 0.25)
        .collect()
}

fn matrix() -> Vec<f32> {
    (0..RECONSTRUCTED * SELECTED_PIXELS)
        .map(|index| ((index % 13) as f32 - 6.0) * 0.003_125)
        .collect()
}

fn plans() -> (
    PyramidPixelImagePlan<f32>,
    PyramidRowReconstructionPlan<f32>,
) {
    let image = PyramidPixelImagePlan::from_detector_pupils(
        (IMAGE_ROWS, IMAGE_COLUMNS),
        (PUPIL_ROWS, PUPIL_COLUMNS),
        &ORIGINS,
        MASK.to_vec(),
    )
    .unwrap();
    let row = filtergraph_algorithms::PyramidPixelRowPlan::new(image.clone(), BLOCK_ROWS).unwrap();
    let reconstruction =
        PreparedGemv::new(RECONSTRUCTED, SELECTED_PIXELS, Arc::from(matrix())).unwrap();
    (
        image,
        PyramidRowReconstructionPlan::new(row, reconstruction).unwrap(),
    )
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
fn progressive_plan_matches_full_frame_science_and_publishes_only_at_terminal() {
    let (full, progressive) = plans();
    let mut full_workspace = full.workspace();
    let mut progressive_workspace = progressive.workspace();
    let mut full_pixels = vec![0.0; SELECTED_PIXELS];
    let mut full_mean = 0.0;
    let mut full_reconstructed = vec![0.0; RECONSTRUCTED];
    let mut progressive_pixels = vec![-1.0; SELECTED_PIXELS];
    let mut progressive_mean = -1.0;
    let mut progressive_reconstructed = vec![-1.0; RECONSTRUCTED];

    for sequence in 1..=2 {
        let pixels = image(sequence);
        full.process(
            PyramidPixelOutput::new(&mut full_pixels, &mut full_mean),
            &mut full_workspace,
            &pixels,
        )
        .unwrap();
        progressive
            .reconstructor()
            .multiply(&mut full_reconstructed, &full_pixels)
            .unwrap();

        let previous_pixels = progressive_pixels.clone();
        let previous_mean = progressive_mean;
        let previous_reconstructed = progressive_reconstructed.clone();
        for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
            let result = progressive
                .process_block(
                    PyramidRowReconstructionOutput::new(
                        &mut progressive_pixels,
                        &mut progressive_mean,
                        &mut progressive_reconstructed,
                    ),
                    &mut progressive_workspace,
                    &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
                    block_info(sequence as u64, first_row),
                )
                .unwrap();
            if first_row + BLOCK_ROWS < IMAGE_ROWS {
                assert_eq!(result, RowBlockResult::Pending);
                assert_eq!(progressive_pixels, previous_pixels);
                assert_eq!(progressive_mean, previous_mean);
                assert_eq!(progressive_reconstructed, previous_reconstructed);
            } else {
                assert_eq!(
                    result,
                    RowBlockResult::Complete {
                        sequence: sequence as u64,
                        discontinuity: false,
                    }
                );
            }
        }

        assert_eq!(progressive_pixels, full_pixels);
        assert_eq!(progressive_mean, full_mean);
        for (&actual, &expected) in progressive_reconstructed.iter().zip(&full_reconstructed) {
            assert!((actual - expected).abs() <= 2.0e-6 * expected.abs().max(1.0));
        }
        assert_eq!(
            progressive_workspace.reciprocal_previous_frame_mean(),
            full_workspace.reciprocal_previous_frame_mean()
        );
    }
}

#[test]
fn disjoint_output_shards_are_bitwise_equal_to_serial_progressive_reconstruction() {
    let (_, plan) = plans();
    for lanes in 2..=4 {
        let mut serial_workspace = plan.workspace();
        let mut sharded_workspace = plan.workspace();
        let mut serial_pixels = vec![0.0; SELECTED_PIXELS];
        let mut sharded_pixels = vec![0.0; SELECTED_PIXELS];
        let mut serial_mean = 0.0;
        let mut sharded_mean = 0.0;
        let mut serial_reconstructed = vec![0.0; RECONSTRUCTED];
        let mut sharded_reconstructed = vec![0.0; RECONSTRUCTED];
        let mut early_dispatches = 0;

        for sequence in 1..=2 {
            let pixels = image(sequence);
            for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
                plan.process_block(
                    PyramidRowReconstructionOutput::new(
                        &mut serial_pixels,
                        &mut serial_mean,
                        &mut serial_reconstructed,
                    ),
                    &mut serial_workspace,
                    &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
                    block_info(sequence as u64, first_row),
                )
                .unwrap();

                plan.process_block_with_reconstruction(
                    PyramidRowReconstructionOutput::new(
                        &mut sharded_pixels,
                        &mut sharded_mean,
                        &mut sharded_reconstructed,
                    ),
                    &mut sharded_workspace,
                    &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
                    block_info(sequence as u64, first_row),
                    |plan, normalized, reconstructed, ranges| {
                        if first_row + BLOCK_ROWS < IMAGE_ROWS {
                            early_dispatches += 1;
                        }
                        for lane in 0..lanes {
                            let first = reconstructed.len() * lane / lanes;
                            let last = reconstructed.len() * (lane + 1) / lanes;
                            for columns in ranges {
                                plan.accumulate_reconstruction_partition(
                                    normalized,
                                    &mut reconstructed[first..last],
                                    columns.clone(),
                                    first,
                                )?;
                            }
                        }
                        Ok::<(), filtergraph_algorithms::PyramidRowReconstructionError>(())
                    },
                )
                .unwrap();
            }
        }

        assert!(early_dispatches > 0);
        assert_eq!(sharded_pixels, serial_pixels);
        assert_eq!(sharded_mean.to_bits(), serial_mean.to_bits());
        assert_eq!(
            sharded_reconstructed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            serial_reconstructed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn executor_failure_abandons_the_frame_and_marks_recovery_discontinuous() {
    let (_, plan) = plans();
    let mut workspace = plan.workspace();
    let mut normalized = vec![-1.0; SELECTED_PIXELS];
    let mut mean = -1.0;
    let mut reconstructed = vec![-1.0; RECONSTRUCTED];
    let pixels = image(1);

    let error = plan
        .process_block_with_reconstruction(
            PyramidRowReconstructionOutput::new(&mut normalized, &mut mean, &mut reconstructed),
            &mut workspace,
            &pixels[..BLOCK_ROWS * IMAGE_COLUMNS],
            block_info(1, 0),
            |_plan, _pixels, _output, _columns| Err("worker failed"),
        )
        .unwrap_err();
    assert_eq!(
        error,
        PyramidRowReconstructionExecutionError::Executor("worker failed")
    );
    assert_eq!(normalized, vec![-1.0; SELECTED_PIXELS]);
    assert_eq!(mean, -1.0);
    assert_eq!(reconstructed, vec![-1.0; RECONSTRUCTED]);

    for first_row in (BLOCK_ROWS..IMAGE_ROWS).step_by(BLOCK_ROWS) {
        assert!(matches!(
            plan.process_block(
                PyramidRowReconstructionOutput::new(
                    &mut normalized,
                    &mut mean,
                    &mut reconstructed,
                ),
                &mut workspace,
                &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
                block_info(1, first_row),
            )
            .unwrap(),
            RowBlockResult::Dropped(_)
        ));
    }

    for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
        let result = plan
            .process_block(
                PyramidRowReconstructionOutput::new(&mut normalized, &mut mean, &mut reconstructed),
                &mut workspace,
                &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
                block_info(2, first_row),
            )
            .unwrap();
        if first_row + BLOCK_ROWS == IMAGE_ROWS {
            assert_eq!(
                result,
                RowBlockResult::Complete {
                    sequence: 2,
                    discontinuity: true,
                }
            );
        }
    }
}

#[test]
fn rejected_caller_storage_does_not_partially_reset_an_active_frame() {
    let (_, plan) = plans();
    let pixels = image(1);
    let mut expected_workspace = plan.workspace();
    let mut actual_workspace = plan.workspace();
    let mut expected_pixels = vec![0.0; SELECTED_PIXELS];
    let mut actual_pixels = vec![0.0; SELECTED_PIXELS];
    let mut expected_mean = 0.0;
    let mut actual_mean = 0.0;
    let mut expected_reconstructed = vec![0.0; RECONSTRUCTED];
    let mut actual_reconstructed = vec![0.0; RECONSTRUCTED];

    for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
        plan.process_block(
            PyramidRowReconstructionOutput::new(
                &mut expected_pixels,
                &mut expected_mean,
                &mut expected_reconstructed,
            ),
            &mut expected_workspace,
            &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
            block_info(1, first_row),
        )
        .unwrap();

        if first_row == 0 {
            plan.process_block(
                PyramidRowReconstructionOutput::new(
                    &mut actual_pixels,
                    &mut actual_mean,
                    &mut actual_reconstructed,
                ),
                &mut actual_workspace,
                &pixels[..BLOCK_ROWS * IMAGE_COLUMNS],
                block_info(1, 0),
            )
            .unwrap();
            assert!(
                plan.process_block(
                    PyramidRowReconstructionOutput::new(
                        &mut actual_pixels,
                        &mut actual_mean,
                        &mut actual_reconstructed,
                    ),
                    &mut actual_workspace,
                    &pixels[..BLOCK_ROWS * IMAGE_COLUMNS - 1],
                    block_info(1, 0),
                )
                .is_err()
            );
        } else {
            plan.process_block(
                PyramidRowReconstructionOutput::new(
                    &mut actual_pixels,
                    &mut actual_mean,
                    &mut actual_reconstructed,
                ),
                &mut actual_workspace,
                &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
                block_info(1, first_row),
            )
            .unwrap();
        }
    }

    assert_eq!(actual_pixels, expected_pixels);
    assert_eq!(actual_mean.to_bits(), expected_mean.to_bits());
    assert_eq!(
        actual_reconstructed
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_reconstructed
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn warmed_progressive_processing_allocates_and_destroys_nothing() {
    let (_, plan) = plans();
    let pixels = image(1);
    let mut workspace = plan.workspace();
    let mut normalized = vec![0.0; SELECTED_PIXELS];
    let mut mean = 0.0;
    let mut reconstructed = vec![0.0; RECONSTRUCTED];

    for sequence in 1..=2 {
        let activity = allocation_activity(|| {
            for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
                plan.process_block(
                    PyramidRowReconstructionOutput::new(
                        &mut normalized,
                        &mut mean,
                        &mut reconstructed,
                    ),
                    &mut workspace,
                    &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
                    block_info(sequence, first_row),
                )
                .unwrap();
            }
        });
        assert_eq!(activity, (0, 0));
    }
}

#[test]
fn declaration_keeps_worker_selection_out_of_scientific_configuration() {
    assert_eq!(
        <PyramidRowReconstructorF32 as AlgorithmDeclaration>::OPERATION,
        AlgorithmOperation::PyramidRowReconstructionF32
    );
    assert!(
        <PyramidRowReconstructorF32 as AlgorithmDeclaration>::CONFIGURATION
            .iter()
            .all(|field| !field.name().contains("worker"))
    );

    let mut declared =
        PyramidRowReconstructorF32::prepare(PyramidRowReconstructorF32Configuration {
            row_block_rows: BLOCK_ROWS,
            pupil_origins: ORIGINS.map(|(row, column)| [row, column]).to_vec(),
            pupil_mask: Some(MASK.to_vec()),
            reconstructed_count: RECONSTRUCTED,
            initial_reconstructor: Some(matrix()),
            reconstructed_schema: "test.filtergraph.pwfs-output/1".to_owned(),
            image_size: (IMAGE_ROWS, IMAGE_COLUMNS),
            pupil_size: (PUPIL_ROWS, PUPIL_COLUMNS),
        })
        .unwrap();
    let pixels = image(1);
    let mut normalized = vec![0.0; SELECTED_PIXELS];
    let mut mean = [0.0];
    let mut reconstructed = vec![0.0; RECONSTRUCTED];
    for first_row in (0..IMAGE_ROWS).step_by(BLOCK_ROWS) {
        let result = PyramidRowReconstructorF32::process_with_metadata(
            &mut declared,
            &[Some(
                SampleMetadata::new(7, first_row as u32)
                    .with_terminal(first_row + BLOCK_ROWS == IMAGE_ROWS),
            )],
            &pixels[first_row * IMAGE_COLUMNS..(first_row + BLOCK_ROWS) * IMAGE_COLUMNS],
            &mut normalized,
            &mut mean,
            &mut reconstructed,
        )
        .unwrap();
        if first_row + BLOCK_ROWS == IMAGE_ROWS {
            assert!(matches!(result, AlgorithmProcessResult::Complete { .. }));
        } else {
            assert_eq!(result, AlgorithmProcessResult::Deferred);
        }
    }
}
