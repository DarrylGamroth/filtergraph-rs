//! Actual JFG acceptance/rejection boundaries for incremental reconstruction.
use filtergraph_algorithms::*;
use std::sync::Arc;

#[test]
fn finite_fma_cancellation_is_accepted() {
    let p = IncrementalDenseReconstructionPlan::new(
        PreparedGemv::new(1, 2, Arc::from([-0.5_f32, 1.5])).unwrap(),
        2,
        1,
    )
    .unwrap();
    let mut w = p.workspace();
    let mut out = [71.0];
    let info = RowBlockInfo {
        sequence: 1,
        first_row: 0,
        terminal: true,
        discontinuity: false,
        corrupted: false,
    };
    p.process_block(&mut out, &mut w, &[f32::MAX; 2], &[0, 1], 2, info)
        .expect("current JFG accepts finite FMA cancellation");
    assert_eq!(out, [f32::MAX]);
    p.process_block(
        &mut out,
        &mut w,
        &[1.0, 2.0],
        &[0, 1],
        2,
        RowBlockInfo {
            sequence: 2,
            ..info
        },
    )
    .unwrap();
    assert_eq!(out, [2.5]);
}

fn configuration(rows: usize) -> IncrementalDenseReconstructorF32Configuration {
    IncrementalDenseReconstructorF32Configuration {
        measurement_count: 1,
        measurement_block_capacity: 1,
        blocks_per_sample: 1,
        reconstructed_count: rows,
        initial_reconstructor: None,
        measurements_schema: "test.measurements/1".into(),
        column_indices_schema: "test.columns/1".into(),
        measurement_count_schema: "test.count/1".into(),
        reconstructed_schema: "test.output/1".into(),
        reconstructor_schema: "test.matrix/1".into(),
    }
}
#[test]
fn impossible_matrix_layout_is_rejected_before_default_allocation() {
    // This cannot reach an allocator: Vec rejects the isize byte extent first.
    let outcome = std::panic::catch_unwind(|| {
        IncrementalDenseReconstructorF32::prepare(configuration(isize::MAX as usize))
    });
    assert!(
        matches!(outcome, Ok(Err(_))),
        "preparation must return an error rather than panic"
    );
}

#[test]
fn fused_progressive_lanes_and_tail_preserve_failure_atomicity() {
    for rows in [1, 7, 8, 9, 17, 257] {
        let matrix: Vec<f64> = (0..rows).flat_map(|_| [-0.5, 1.5]).collect();
        let plan = IncrementalDenseReconstructionPlan::new(
            PreparedGemv::new(rows, 2, Arc::from(matrix)).unwrap(),
            2,
            1,
        )
        .unwrap();
        let mut workspace = plan.workspace();
        let mut output = vec![71.0; rows];
        let info = RowBlockInfo {
            sequence: 1,
            first_row: 0,
            terminal: true,
            discontinuity: false,
            corrupted: false,
        };
        plan.process_block(
            &mut output,
            &mut workspace,
            &[f64::MAX; 2],
            &[0, 1],
            2,
            info,
        )
        .unwrap();
        assert_eq!(output, vec![f64::MAX; rows]);
        output.fill(71.0);
        assert!(
            plan.process_block(
                &mut output,
                &mut workspace,
                &[0.0, f64::MAX],
                &[0, 1],
                2,
                RowBlockInfo {
                    sequence: 2,
                    ..info
                }
            )
            .is_err()
        );
        assert_eq!(output, vec![71.0; rows]);
        plan.process_block(
            &mut output,
            &mut workspace,
            &[1.0, 2.0],
            &[0, 1],
            2,
            RowBlockInfo {
                sequence: 3,
                ..info
            },
        )
        .unwrap();
        assert_eq!(output, vec![2.5; rows]);
    }
}
