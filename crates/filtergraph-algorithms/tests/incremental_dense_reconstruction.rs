//! Sensor-independent indexed measurement reconstruction and recovery.

use filtergraph_algorithms::{
    AlgorithmProcessResult, IncrementalDenseReconstructionError as Error,
    IncrementalDenseReconstructionExecutionError, IncrementalDenseReconstructionPlan,
    IncrementalDenseReconstructorF32, IncrementalDenseReconstructorF32Configuration, PreparedGemv,
    RowBlockDropReason, RowBlockInfo, RowBlockResult, SampleMetadata,
};
use std::sync::Arc;

fn plan(blocks: usize) -> IncrementalDenseReconstructionPlan<f32> {
    IncrementalDenseReconstructionPlan::new(
        PreparedGemv::new(
            2,
            4,
            Arc::from([1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]),
        )
        .unwrap(),
        4,
        blocks,
    )
    .unwrap()
}
fn info(sequence: u64, index: usize, terminal: bool) -> RowBlockInfo {
    RowBlockInfo {
        sequence,
        first_row: index,
        terminal,
        discontinuity: false,
        corrupted: false,
    }
}

#[test]
fn arbitrary_column_batches_zero_count_and_terminal_only_publication() {
    let plan = plan(3);
    let mut workspace = plan.workspace();
    let mut output = [-7.0, -9.0];
    assert_eq!(
        plan.process_block(
            &mut output,
            &mut workspace,
            &[3.0, 1.0, f32::NAN, f32::NAN],
            &[2, 0, u32::MAX, u32::MAX],
            2,
            info(10, 0, false)
        ),
        Ok(RowBlockResult::Pending)
    );
    assert_eq!(output, [-7.0, -9.0]);
    assert_eq!(workspace.seen_count(), 2);
    assert_eq!(
        plan.process_block(
            &mut output,
            &mut workspace,
            &[f32::NAN; 4],
            &[u32::MAX; 4],
            0,
            info(10, 1, false)
        ),
        Ok(RowBlockResult::Pending)
    );
    assert_eq!(
        plan.process_block(
            &mut output,
            &mut workspace,
            &[4.0, 2.0, 0.0, 0.0],
            &[3, 1, 0, 0],
            2,
            info(10, 2, true)
        ),
        Ok(RowBlockResult::Complete {
            sequence: 10,
            discontinuity: false
        })
    );
    assert_eq!(output, [30.0, 300.0]);
    assert_eq!(workspace.sequence(), None);
}

#[test]
fn faults_abandon_partial_sample_preserve_output_and_mark_recovery() {
    let plan = plan(2);
    let cases = [
        (info(7, 1, true), RowBlockDropReason::MissingStart),
        (info(8, 1, true), RowBlockDropReason::Sequence),
        (
            RowBlockInfo {
                discontinuity: true,
                ..info(7, 1, true)
            },
            RowBlockDropReason::MidFrameDiscontinuity,
        ),
        (
            RowBlockInfo {
                corrupted: true,
                ..info(7, 1, true)
            },
            RowBlockDropReason::Corrupted,
        ),
    ];
    for (index, (fault, expected)) in cases.into_iter().enumerate() {
        let mut workspace = plan.workspace();
        let mut output = [17.0, 19.0];
        if index != 0 {
            plan.process_block(
                &mut output,
                &mut workspace,
                &[1.0; 4],
                &[0, 1, 0, 0],
                2,
                info(7, 0, false),
            )
            .unwrap();
        }
        assert_eq!(
            plan.process_block(
                &mut output,
                &mut workspace,
                &[1.0; 4],
                &[2, 3, 0, 0],
                2,
                fault
            ),
            Ok(RowBlockResult::Dropped(expected))
        );
        assert_eq!(output, [17.0, 19.0]);
        plan.process_block(
            &mut output,
            &mut workspace,
            &[1.0; 4],
            &[0, 1, 0, 0],
            2,
            info(9, 0, false),
        )
        .unwrap();
        assert_eq!(
            plan.process_block(
                &mut output,
                &mut workspace,
                &[1.0; 4],
                &[2, 3, 0, 0],
                2,
                info(9, 1, true)
            ),
            Ok(RowBlockResult::Complete {
                sequence: 9,
                discontinuity: true
            })
        );
        assert_eq!(output, [10.0, 100.0]);
    }
}

#[test]
fn malformed_measurement_batches_do_not_publish() {
    let plan = plan(1);
    let cases = [
        ([1.0; 4], [0, 1, 2, 3], 5, Error::CountOutOfRange),
        ([1.0; 4], [0, 1, 2, 4], 4, Error::ColumnOutOfRange),
        ([1.0; 4], [0, 1, 1, 3], 4, Error::DuplicateColumn),
        (
            [1.0, 1.0, f32::NAN, 1.0],
            [0, 1, 2, 3],
            4,
            Error::NonFiniteInput,
        ),
        ([1.0; 4], [0, 1, 2, 3], 3, Error::IncompleteSample),
    ];
    for (values, columns, count, error) in cases {
        let mut workspace = plan.workspace();
        let mut output = [7.0; 2];
        assert_eq!(
            plan.process_block(
                &mut output,
                &mut workspace,
                &values,
                &columns,
                count,
                info(1, 0, true)
            ),
            Err(error)
        );
        assert_eq!(output, [7.0; 2]);
        assert_eq!(workspace.sequence(), None);
        assert_eq!(
            plan.process_block(
                &mut output,
                &mut workspace,
                &[1.0; 4],
                &[0, 1, 2, 3],
                4,
                info(2, 0, true)
            ),
            Ok(RowBlockResult::Complete {
                sequence: 2,
                discontinuity: true
            })
        );
    }
}

#[test]
fn output_partitioning_preserves_serial_arithmetic_exactly() {
    let matrix = PreparedGemv::new(
        17,
        4,
        Arc::from(
            (0..68)
                .map(|x| (x as f32 - 32.0) / 17.0)
                .collect::<Vec<_>>(),
        ),
    )
    .unwrap();
    let plan = IncrementalDenseReconstructionPlan::new(matrix, 4, 2).unwrap();
    let mut serial = plan.workspace();
    let mut partitioned = plan.workspace();
    let mut output_serial = [0.0; 17];
    let mut output_partitioned = [0.0; 17];
    for (block, columns) in [[3, 1, 0, 0], [2, 0, 0, 0]].into_iter().enumerate() {
        let values = [0.123, -4.56, 0.0, 0.0];
        plan.process_block(
            &mut output_serial,
            &mut serial,
            &values,
            &columns,
            2,
            info(1, block, block == 1),
        )
        .unwrap();
        plan.process_block_with_reconstruction(
            &mut output_partitioned,
            &mut partitioned,
            &values,
            &columns,
            2,
            info(1, block, block == 1),
            |plan, values, columns, output| {
                let (first, last) = output.split_at_mut(7);
                plan.accumulate_reconstruction_partition(
                    values,
                    columns,
                    first,
                    0..values.len(),
                    0,
                )?;
                plan.accumulate_reconstruction_partition(values, columns, last, 0..values.len(), 7)
            },
        )
        .unwrap();
    }
    assert_eq!(output_serial, output_partitioned);
}

#[test]
fn executor_failure_abandons_candidate_without_publishing() {
    let plan = plan(1);
    let mut workspace = plan.workspace();
    let mut output = [123.0; 2];
    let result = plan.process_block_with_reconstruction(
        &mut output,
        &mut workspace,
        &[1.0; 4],
        &[0, 1, 2, 3],
        4,
        info(1, 0, true),
        |_, _, _, candidate| {
            candidate[0] = 999.0;
            Err::<(), _>(5)
        },
    );
    assert_eq!(
        result,
        Err(IncrementalDenseReconstructionExecutionError::Executor(5))
    );
    assert_eq!(output, [123.0; 2]);
    assert_eq!(workspace.sequence(), None);
}

#[test]
fn nonfinite_accumulation_preserves_previous_output() {
    let plan = plan(1);
    let mut workspace = plan.workspace();
    let mut output = [17.0; 2];
    assert_eq!(
        plan.process_block(
            &mut output,
            &mut workspace,
            &[f32::MAX; 4],
            &[0, 1, 2, 3],
            4,
            info(1, 0, true)
        ),
        Err(Error::NonFiniteOutput)
    );
    assert_eq!(output, [17.0; 2]);
    assert_eq!(workspace.sequence(), None);
}

#[test]
fn replacement_rebuilds_coefficients_but_reuses_workspace_extents() {
    let plan = plan(2);
    let mut workspace = plan.workspace();
    let mut output = [9.0; 2];
    plan.process_block(
        &mut output,
        &mut workspace,
        &[1.0; 4],
        &[0, 1, 0, 0],
        2,
        info(1, 0, false),
    )
    .unwrap();
    let replacement = plan.with_reconstructor(&[2.0; 8]).unwrap();
    workspace.reset();
    replacement
        .process_block(
            &mut output,
            &mut workspace,
            &[1.0; 4],
            &[0, 1, 0, 0],
            2,
            info(2, 0, false),
        )
        .unwrap();
    assert_eq!(
        replacement.process_block(
            &mut output,
            &mut workspace,
            &[1.0; 4],
            &[2, 3, 0, 0],
            2,
            info(2, 1, true)
        ),
        Ok(RowBlockResult::Complete {
            sequence: 2,
            discontinuity: false
        })
    );
    assert_eq!(output, [8.0; 2]);
    assert!(plan.with_reconstructor(&[2.0; 7]).is_err());
}

#[test]
fn declaration_has_three_block_rate_ports_and_conditional_frame_output() {
    let mut prepared =
        IncrementalDenseReconstructorF32::prepare(IncrementalDenseReconstructorF32Configuration {
            measurement_count: 4,
            measurement_block_capacity: 4,
            blocks_per_sample: 2,
            reconstructed_count: 2,
            initial_reconstructor: Some(vec![1.0; 8]),
            measurements_schema: "ao.measurements/1".into(),
            column_indices_schema: "ao.measurement-columns/1".into(),
            measurement_count_schema: "ao.measurement-count/1".into(),
            reconstructed_schema: "ao.controller-residual-error/1".into(),
            reconstructor_schema: "ao.reconstructor/1".into(),
        })
        .unwrap();
    for format in &prepared.formats()[..3] {
        assert_eq!(format.rate_ratio(), (2, 1));
    }
    assert_eq!(prepared.formats()[3].rate_ratio(), (1, 1));
    let mut output = [8.0; 2];
    let result = IncrementalDenseReconstructorF32::process_with_metadata(
        &mut prepared,
        &[Some(SampleMetadata::new(8, 0)), None, None],
        &[1.0; 4],
        &[0, 1, 0, 0],
        &[2],
        &mut output,
    )
    .unwrap();
    assert_eq!(result, AlgorithmProcessResult::deferred());
    assert_eq!(output, [8.0; 2]);
    let result = IncrementalDenseReconstructorF32::process_with_metadata(
        &mut prepared,
        &[
            Some(SampleMetadata::new(8, 1).with_terminal(true)),
            None,
            None,
        ],
        &[1.0; 4],
        &[2, 3, 0, 0],
        &[2],
        &mut output,
    )
    .unwrap();
    assert_eq!(result.metadata().unwrap().sequence(), 8);
    assert_eq!(output, [4.0; 2]);
}

#[test]
fn batched_updates_preserve_fma_order_for_a_partition() {
    let rows = 19;
    let cols = 11;
    let matrix: Arc<[f32]> = (0..rows * cols)
        .map(|i| ((i * 89 % 103) as f32 - 51.0) / 33.0)
        .collect::<Vec<_>>()
        .into();
    let reconstructor = PreparedGemv::new(rows, cols, Arc::clone(&matrix)).unwrap();
    let plan = IncrementalDenseReconstructionPlan::new(reconstructor, cols, 2).unwrap();
    let selected = [8, 2, 5, 1, 9, 0, 3, 6, 10, 7, 4];
    let measurements = [
        0.0, -0.33, 0.07, -1.2, 1.4, -2.4, 1.32, 4.5, -0.11, 0.12, -0.2,
    ];
    let mut actual = [0.125_f32; 13];
    let mut expected = actual;
    for column in 1..10 {
        for (row, out) in expected.iter_mut().enumerate() {
            *out = matrix[(row + 3) * cols + selected[column] as usize]
                .mul_add(measurements[column], *out);
        }
    }
    plan.accumulate_reconstruction_partition(&measurements, &selected, &mut actual, 1..10, 3)
        .unwrap();
    assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
}
