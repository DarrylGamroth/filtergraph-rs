//! Composable row-block sensing, scientific ordering, and sample identity.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, AlgorithmProcessResult,
    DeclarationProcessError, IncrementalDenseReconstructionPlan, NativeProcessError,
    PortPublication, PreparedGemv, RowBlockDropReason, RowBlockInfo, RowBlockResult,
    SampleMetadata, ShackHartmannImagePlan, ShackHartmannMeasurementBlockF32,
    ShackHartmannMeasurementBlockF32Configuration, ShackHartmannMeasurementBlockPlan,
    ShackHartmannMeasurementBlockPreparationError, ShackHartmannMeasurementBlockProcessError,
    ShackHartmannMeasurementBlockResult, ShackHartmannOutput,
};

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

const ROWS: usize = 6;
const COLUMNS: usize = 6;
const ORIGINS: [(usize, usize); 4] = [(0, 0), (0, 3), (3, 0), (3, 3)];

fn configuration(block_rows: usize) -> ShackHartmannMeasurementBlockF32Configuration {
    ShackHartmannMeasurementBlockF32Configuration {
        row_block_rows: block_rows,
        subaperture_count: 4,
        initial_subaperture_origins: Some(
            ORIGINS.iter().map(|&(row, column)| [row, column]).collect(),
        ),
        coordinate_scale: 1.0,
        pixel_threshold: 0.0,
        flux_threshold: 0.0,
        reference_slopes: None,
        active: None,
        measurements_schema: "test.filtergraph/measurement-block/1".into(),
        column_indices_schema: "test.filtergraph/measurement-column-indices/1".into(),
        measurement_count_schema: "test.filtergraph/measurement-count/1".into(),
        image_size: (ROWS, COLUMNS),
        subaperture_size: (2, 2),
    }
}

fn image() -> Vec<f32> {
    (0..ROWS)
        .flat_map(|row| {
            (0..COLUMNS).map(move |column| 1.0 + 0.2 * row as f32 + 0.1 * column as f32)
        })
        .collect()
}
fn sensing(origins: &[(usize, usize)]) -> ShackHartmannImagePlan<f32> {
    ShackHartmannImagePlan::from_detector_regions(
        (ROWS, COLUMNS),
        (2, 2),
        origins,
        1.0,
        0.0,
        0.0,
        None,
        None,
    )
    .unwrap()
}
fn info(sequence: u64, first_row: usize, block_rows: usize) -> RowBlockInfo {
    RowBlockInfo {
        sequence,
        first_row,
        terminal: first_row + block_rows == ROWS,
        discontinuity: false,
        corrupted: false,
    }
}
fn full_frame_slopes(plan: &ShackHartmannImagePlan<f32>, image: &[f32]) -> [f32; 8] {
    let mut slopes = [0.0; 8];
    let mut flux = [0.0; 4];
    let mut validity = [false; 4];
    plan.process(
        ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
        &mut plan.workspace(),
        image,
    )
    .unwrap();
    slopes
}

#[test]
fn publishes_new_slope_prefixes_and_empty_blocks_with_identical_sensing() {
    let sensing = sensing(&ORIGINS);
    let pixels = image();
    let expected = full_frame_slopes(&sensing, &pixels);
    let plan = ShackHartmannMeasurementBlockPlan::new(sensing, 1).unwrap();
    let mut workspace = plan.workspace();
    let mut measurements = [-99.0; 8];
    let mut columns = [99; 8];
    let mut count = [99];
    let mut collected = [0.0; 8];
    let mut next_column = 0;
    for row in 0..ROWS {
        assert_eq!(
            plan.process_block(
                &mut measurements,
                &mut columns,
                &mut count,
                &mut workspace,
                &pixels[row * COLUMNS..(row + 1) * COLUMNS],
                info(41, row, 1)
            )
            .unwrap(),
            ShackHartmannMeasurementBlockResult::Published {
                discontinuity: false
            }
        );
        assert_eq!(count[0], if row == 1 || row == 4 { 4 } else { 0 });
        for entry in 0..count[0] as usize {
            assert_eq!(columns[entry] as usize, next_column);
            collected[next_column] = measurements[entry];
            next_column += 1;
        }
    }
    assert_eq!(next_column, 8);
    assert_eq!(collected, expected);
    assert_eq!(workspace.sequence(), None);
}

#[test]
fn detector_completion_never_reorders_scientific_coordinates() {
    let origins = [(3, 0), (0, 0), (3, 3), (0, 3)];
    let sensing = sensing(&origins);
    let pixels = image();
    let expected = full_frame_slopes(&sensing, &pixels);
    let plan = ShackHartmannMeasurementBlockPlan::new(sensing, 1).unwrap();
    assert_eq!(plan.completion_schedule(), &[1, 3, 0, 2]);
    let mut workspace = plan.workspace();
    let mut measurements = [0.0; 8];
    let mut columns = [0; 8];
    let mut count = [0];
    for row in 0..ROWS {
        plan.process_block(
            &mut measurements,
            &mut columns,
            &mut count,
            &mut workspace,
            &pixels[row * COLUMNS..(row + 1) * COLUMNS],
            info(41, row, 1),
        )
        .unwrap();
        assert_eq!(count[0], if row == 4 { 8 } else { 0 });
        if row == 4 {
            assert_eq!(measurements, expected);
            assert_eq!(columns, [0, 1, 2, 3, 4, 5, 6, 7]);
        }
    }
}

#[test]
fn loss_does_not_modify_outputs_and_recovery_marks_next_start() {
    let plan = ShackHartmannMeasurementBlockPlan::new(sensing(&ORIGINS), 1).unwrap();
    let pixels = image();
    let reasons = [
        RowBlockDropReason::MissingStart,
        RowBlockDropReason::RowGap,
        RowBlockDropReason::Sequence,
        RowBlockDropReason::MidFrameDiscontinuity,
        RowBlockDropReason::Corrupted,
        RowBlockDropReason::Terminal,
    ];
    for reason in reasons {
        let mut workspace = plan.workspace();
        let mut measurements = [-7.0; 8];
        let mut columns = [77; 8];
        let mut count = [77];
        if reason != RowBlockDropReason::MissingStart {
            plan.process_block(
                &mut measurements,
                &mut columns,
                &mut count,
                &mut workspace,
                &pixels[..COLUMNS],
                info(1, 0, 1),
            )
            .unwrap();
        }
        measurements.fill(-7.0);
        columns.fill(77);
        count[0] = 77;
        let mut bad = info(1, 1, 1);
        match reason {
            RowBlockDropReason::RowGap => bad.first_row = 2,
            RowBlockDropReason::Sequence => bad.sequence = 2,
            RowBlockDropReason::MidFrameDiscontinuity => bad.discontinuity = true,
            RowBlockDropReason::Corrupted => bad.corrupted = true,
            RowBlockDropReason::Terminal => bad.terminal = true,
            _ => (),
        }
        assert_eq!(
            plan.process_block(
                &mut measurements,
                &mut columns,
                &mut count,
                &mut workspace,
                &pixels[COLUMNS..2 * COLUMNS],
                bad
            )
            .unwrap(),
            ShackHartmannMeasurementBlockResult::Dropped(reason)
        );
        assert_eq!(measurements, [-7.0; 8]);
        assert_eq!(columns, [77; 8]);
        assert_eq!(count, [77]);
        assert_eq!(workspace.sequence(), None);
        assert_eq!(
            plan.process_block(
                &mut measurements,
                &mut columns,
                &mut count,
                &mut workspace,
                &pixels[..COLUMNS],
                info(3, 0, 1)
            )
            .unwrap(),
            ShackHartmannMeasurementBlockResult::Published {
                discontinuity: true
            }
        );
    }
}

#[test]
fn replacement_and_reset_do_not_carry_partial_frame_measurements() {
    let plan = ShackHartmannMeasurementBlockPlan::new(sensing(&ORIGINS), 1).unwrap();
    let replaced = plan
        .with_u32_subaperture_origins(&[0, 0, 0, 2, 2, 0, 2, 2])
        .unwrap()
        .with_pair_interleaved_coordinates(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        .unwrap()
        .with_pair_interleaved_thresholds(&[1.0, 0.0, 1.0, 3.0, 2.0, 0.0, 1.0, 0.0])
        .unwrap()
        .with_reference_slopes(&[0.3; 8])
        .unwrap()
        .with_active(&[true, false, true, false])
        .unwrap();
    let pixels = image();
    let expected = full_frame_slopes(replaced.sensing(), &pixels);
    let mut workspace = plan.workspace();
    let mut measurements = [0.0; 8];
    let mut columns = [0; 8];
    let mut count = [0];
    plan.process_block(
        &mut measurements,
        &mut columns,
        &mut count,
        &mut workspace,
        &pixels[..COLUMNS],
        info(1, 0, 1),
    )
    .unwrap();
    workspace.reset();
    let mut collected = [0.0; 8];
    for row in 0..ROWS {
        let result = replaced
            .process_block(
                &mut measurements,
                &mut columns,
                &mut count,
                &mut workspace,
                &pixels[row * COLUMNS..(row + 1) * COLUMNS],
                info(2, row, 1),
            )
            .unwrap();
        assert_eq!(
            result,
            ShackHartmannMeasurementBlockResult::Published {
                discontinuity: false
            }
        );
        for entry in 0..count[0] as usize {
            collected[columns[entry] as usize] = measurements[entry];
        }
    }
    assert_eq!(collected, expected);
}

#[test]
fn invalid_extents_leave_output_unchanged() {
    let plan = ShackHartmannMeasurementBlockPlan::new(sensing(&ORIGINS), 1).unwrap();
    let mut workspace = plan.workspace();
    let mut measurements = [-7.0; 8];
    let mut columns = [77; 8];
    let mut count = [77];
    use filtergraph_algorithms::ShackHartmannMeasurementBlockError;
    for (values, indices, counts, samples, port, expected, actual) in [
        (7, 8, 1, COLUMNS, "measurements", 8, 7),
        (8, 7, 1, COLUMNS, "columns", 8, 7),
        (8, 8, 0, COLUMNS, "count", 1, 0),
        (8, 8, 1, COLUMNS - 1, "row-block", COLUMNS, COLUMNS - 1),
    ] {
        let error = plan
            .process_block(
                &mut measurements[..values],
                &mut columns[..indices],
                &mut count[..counts],
                &mut workspace,
                &[1.0; COLUMNS][..samples],
                info(1, 0, 1),
            )
            .unwrap_err();
        assert_eq!(
            error,
            ShackHartmannMeasurementBlockError::Extent {
                port,
                expected,
                actual
            }
        );
        assert_eq!(
            error.to_string(),
            format!("{port} extent {actual} does not match {expected}")
        );
        assert_eq!(measurements, [-7.0; 8]);
        assert_eq!(columns, [77; 8]);
        assert_eq!(count, [77]);
        assert_eq!(workspace.sequence(), None);
    }
    for block_rows in [0, 4, ROWS] {
        let error =
            ShackHartmannMeasurementBlockPlan::new(sensing(&ORIGINS), block_rows).unwrap_err();
        assert_eq!(
            error,
            ShackHartmannMeasurementBlockPreparationError::BlockRows
        );
        assert!(error.to_string().contains("divide image height"));
    }
}

#[test]
fn declaration_has_three_row_rate_ports_and_physical_row_metadata() {
    let mut declared = ShackHartmannMeasurementBlockF32::prepare(configuration(2)).unwrap();
    assert_eq!(declared.formats().len(), 9);
    assert!(
        declared.formats()[..4]
            .iter()
            .all(|format| format.rate_ratio() == (3, 1))
    );
    for port in &<ShackHartmannMeasurementBlockF32 as AlgorithmDeclaration>::PORTS[1..4] {
        assert_eq!(port.publication(), PortPublication::Conditional);
    }
    let pixels = image();
    let mut measurements = [0.0; 8];
    let mut columns = [0; 8];
    let mut count = [0];
    for block in 0..3 {
        let first_row = block * 2;
        let sample = SampleMetadata::new(77, first_row as u32)
            .with_timestamp_nanoseconds(Some(123456 + block as i64))
            .with_terminal(block == 2);
        assert_eq!(
            ShackHartmannMeasurementBlockF32::process_with_metadata(
                &mut declared,
                &[Some(sample)],
                &pixels[first_row * COLUMNS..(first_row + 2) * COLUMNS],
                &mut measurements,
                &mut columns,
                &mut count
            )
            .unwrap(),
            AlgorithmProcessResult::complete_with_metadata(sample)
        );
    }
}

#[test]
fn a_new_first_row_supersedes_partial_sensing_and_marks_its_start() {
    let plan = ShackHartmannMeasurementBlockPlan::new(sensing(&ORIGINS), 1).unwrap();
    let mut workspace = plan.workspace();
    let pixels = image();
    let mut measurements = [0.0; 8];
    let mut columns = [0; 8];
    let mut count = [0];
    for row in 0..2 {
        plan.process_block(
            &mut measurements,
            &mut columns,
            &mut count,
            &mut workspace,
            &pixels[row * COLUMNS..(row + 1) * COLUMNS],
            info(1, row, 1),
        )
        .unwrap();
    }
    assert_eq!(workspace.published_subapertures(), 2);
    assert_eq!(
        plan.process_block(
            &mut measurements,
            &mut columns,
            &mut count,
            &mut workspace,
            &pixels[..COLUMNS],
            info(2, 0, 1)
        )
        .unwrap(),
        ShackHartmannMeasurementBlockResult::Published {
            discontinuity: true
        }
    );
    assert_eq!(count, [0]);
    assert_eq!(workspace.published_subapertures(), 0);
    let expected = full_frame_slopes(plan.sensing(), &pixels);
    let mut collected = [0.0; 8];
    for row in 1..ROWS {
        plan.process_block(
            &mut measurements,
            &mut columns,
            &mut count,
            &mut workspace,
            &pixels[row * COLUMNS..(row + 1) * COLUMNS],
            info(2, row, 1),
        )
        .unwrap();
        for entry in 0..count[0] as usize {
            collected[columns[entry] as usize] = measurements[entry];
        }
    }
    assert_eq!(collected, expected);
}

#[test]
fn nonfinite_pixels_and_thresholds_preserve_full_frame_validity_behavior() {
    let mut pixels = image();
    pixels[0] = f32::NAN;
    pixels[3 * COLUMNS + 3] = f32::INFINITY;
    let sensing = sensing(&ORIGINS)
        .with_pair_interleaved_thresholds(&[0.0, 0.0, 99.0, 0.0, 0.0, 99.0, 0.0, 0.0])
        .unwrap();
    let expected = full_frame_slopes(&sensing, &pixels);
    assert_eq!(expected, [0.0; 8]);
    let plan = ShackHartmannMeasurementBlockPlan::new(sensing, 1).unwrap();
    let mut workspace = plan.workspace();
    let mut measurements = [99.0; 8];
    let mut columns = [99; 8];
    let mut count = [99];
    let mut collected = [99.0; 8];
    for row in 0..ROWS {
        plan.process_block(
            &mut measurements,
            &mut columns,
            &mut count,
            &mut workspace,
            &pixels[row * COLUMNS..(row + 1) * COLUMNS],
            info(1, row, 1),
        )
        .unwrap();
        for entry in 0..count[0] as usize {
            collected[columns[entry] as usize] = measurements[entry];
        }
    }
    assert_eq!(collected, expected);
}

#[test]
fn each_sensing_parameter_replacement_matches_the_complete_image_plan() {
    let original = ShackHartmannMeasurementBlockPlan::new(sensing(&ORIGINS), 1).unwrap();
    let origins = [3, 0, 0, 0, 3, 3, 0, 3];
    let coordinates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let plans = [
        original.with_u32_subaperture_origins(&origins).unwrap(),
        original
            .with_pair_interleaved_coordinates(&coordinates)
            .unwrap(),
        original.with_reference_slopes(&[0.3; 8]).unwrap(),
        original
            .with_pair_interleaved_thresholds(&[0.0, 99.0, 0.0, 0.0, 99.0, 0.0, 0.0, 0.0])
            .unwrap(),
        original.with_active(&[false, true, true, true]).unwrap(),
    ];
    let pixels = image();
    for plan in plans {
        let expected = full_frame_slopes(plan.sensing(), &pixels);
        let mut workspace = plan.workspace();
        let mut measurements = [0.0; 8];
        let mut columns = [0; 8];
        let mut count = [0];
        let mut collected = [0.0; 8];
        for row in 0..ROWS {
            plan.process_block(
                &mut measurements,
                &mut columns,
                &mut count,
                &mut workspace,
                &pixels[row * COLUMNS..(row + 1) * COLUMNS],
                info(1, row, 1),
            )
            .unwrap();
            for entry in 0..count[0] as usize {
                collected[columns[entry] as usize] = measurements[entry];
            }
        }
        assert_eq!(collected, expected);
    }
    assert!(original.with_u32_subaperture_origins(&[99; 8]).is_err());
    assert!(
        original
            .with_pair_interleaved_coordinates(&[f32::NAN; 8])
            .is_err()
    );
    assert!(original.with_reference_slopes(&[f32::NAN; 8]).is_err());
    assert!(
        original
            .with_pair_interleaved_thresholds(&[-1.0; 8])
            .is_err()
    );
    assert!(original.with_active(&[true]).is_err());
}

#[test]
fn composed_row_processing_preserves_full_frame_and_ordered_axpy_results_without_allocation() {
    for origins in [ORIGINS, [(3, 0), (0, 0), (3, 3), (0, 3)]] {
        let sensing = sensing(&origins);
        let pixels = image();
        let expected_slopes = full_frame_slopes(&sensing, &pixels);
        let matrix: Vec<f32> = (0..24).map(|index| (index as f32 + 1.0) * 0.01).collect();
        let gemv = PreparedGemv::new(3, 8, Arc::from(matrix.clone())).unwrap();
        let mut full_frame = [0.0; 3];
        gemv.multiply(&mut full_frame, &expected_slopes).unwrap();
        // The prior fused algorithm accumulated columns in this same AXPY order.
        let mut ordered = [0.0; 3];
        for column in 0..8 {
            let coefficients = [matrix[column], matrix[8 + column], matrix[16 + column]];
            for (coefficient, destination) in coefficients.iter().zip(&mut ordered) {
                *destination = expected_slopes[column] * coefficient + *destination;
            }
        }
        let sh = ShackHartmannMeasurementBlockPlan::new(sensing, 1).unwrap();
        let reconstruction = IncrementalDenseReconstructionPlan::new(gemv, 8, ROWS).unwrap();
        let mut sh_workspace = sh.workspace();
        let mut reconstruction_workspace = reconstruction.workspace();
        let mut measurements = [0.0; 8];
        let mut columns = [0; 8];
        let mut count = [0];
        let mut output = [-99.0; 3];
        for sequence in 1..=2 {
            let activity = allocation_activity(|| {
                for row in 0..ROWS {
                    let sensing_result = sh
                        .process_block(
                            &mut measurements,
                            &mut columns,
                            &mut count,
                            &mut sh_workspace,
                            &pixels[row * COLUMNS..(row + 1) * COLUMNS],
                            info(sequence, row, 1),
                        )
                        .unwrap();
                    let ShackHartmannMeasurementBlockResult::Published { discontinuity } =
                        sensing_result
                    else {
                        panic!("valid sensing block was dropped");
                    };
                    let result = reconstruction
                        .process_block(
                            &mut output,
                            &mut reconstruction_workspace,
                            &measurements,
                            &columns,
                            count[0],
                            RowBlockInfo {
                                discontinuity,
                                ..info(sequence, row, 1)
                            },
                        )
                        .unwrap();
                    if row + 1 == ROWS {
                        assert_eq!(
                            result,
                            RowBlockResult::Complete {
                                sequence,
                                discontinuity: false
                            }
                        );
                    } else {
                        assert_eq!(result, RowBlockResult::Pending);
                        assert_eq!(output, if sequence == 1 { [-99.0; 3] } else { ordered });
                    }
                }
            });
            assert_eq!(activity, (0, 0));
            assert_eq!(output, ordered);
            for (actual, expected) in output.iter().zip(full_frame) {
                assert!((*actual - expected).abs() <= 2.0e-6);
            }
        }
    }
}

#[test]
fn native_metadata_rejection_and_plan_adoption_preserve_outputs_and_recover() {
    let mut declared = ShackHartmannMeasurementBlockF32::prepare(configuration(2)).unwrap();
    let pixels = image();
    let mut measurements = [-99.0; 8];
    let mut columns = [99; 8];
    let mut count = [99];
    assert_eq!(
        ShackHartmannMeasurementBlockF32::process_with_metadata(
            &mut declared,
            &[None],
            &pixels[..2 * COLUMNS],
            &mut measurements,
            &mut columns,
            &mut count
        )
        .unwrap_err(),
        NativeProcessError::Process(DeclarationProcessError::Algorithm(
            ShackHartmannMeasurementBlockProcessError::MissingSampleMetadata
        ))
    );
    assert_eq!(measurements, [-99.0; 8]);
    assert_eq!(columns, [99; 8]);
    assert_eq!(count, [99]);
    let first = SampleMetadata::new(1, 0);
    ShackHartmannMeasurementBlockF32::process_with_metadata(
        &mut declared,
        &[Some(first)],
        &pixels[..2 * COLUMNS],
        &mut measurements,
        &mut columns,
        &mut count,
    )
    .unwrap();
    let old_measurements = measurements;
    let old_columns = columns;
    let old_count = count;
    assert!(
        declared
            .replace_parameter(6, AlgorithmInput::from(&[f32::NAN; 8][..]))
            .unwrap()
            .is_err()
    );
    let second = SampleMetadata::new(1, 2);
    assert_eq!(
        ShackHartmannMeasurementBlockF32::process_with_metadata(
            &mut declared,
            &[Some(second)],
            &pixels[2 * COLUMNS..4 * COLUMNS],
            &mut measurements,
            &mut columns,
            &mut count
        )
        .unwrap(),
        AlgorithmProcessResult::complete_with_metadata(second)
    );
    declared
        .replace_parameter(6, AlgorithmInput::from(&[0.3_f32; 8][..]))
        .unwrap()
        .unwrap();
    assert_eq!(
        ShackHartmannMeasurementBlockF32::process_with_metadata(
            &mut declared,
            &[Some(SampleMetadata::new(1, 4).with_terminal(true))],
            &pixels[4 * COLUMNS..6 * COLUMNS],
            &mut measurements,
            &mut columns,
            &mut count
        )
        .unwrap(),
        AlgorithmProcessResult::Deferred
    );
    assert_eq!(measurements, old_measurements);
    assert_eq!(columns, old_columns);
    assert_eq!(count, [0]);
    assert_eq!(old_count, [4]);
    let sample = SampleMetadata::new(2, 0).with_timestamp_nanoseconds(Some(3456));
    assert_eq!(
        ShackHartmannMeasurementBlockF32::process_with_metadata(
            &mut declared,
            &[Some(sample)],
            &pixels[..2 * COLUMNS],
            &mut measurements,
            &mut columns,
            &mut count
        )
        .unwrap(),
        AlgorithmProcessResult::complete_with_metadata(sample.with_discontinuity(true))
    );
    let expected = full_frame_slopes(declared.plan().sensing(), &pixels);
    assert_eq!(&measurements[..4], &expected[..4]);
    let before = (measurements, columns, count);
    assert_eq!(
        ShackHartmannMeasurementBlockF32::process_with_metadata(
            &mut declared,
            &[Some(SampleMetadata::new(2, 2).with_corrupted(true))],
            &pixels[2 * COLUMNS..4 * COLUMNS],
            &mut measurements,
            &mut columns,
            &mut count
        )
        .unwrap(),
        AlgorithmProcessResult::Deferred
    );
    assert_eq!((measurements, columns, count), before);
}
