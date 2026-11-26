//! Detector frontend and reconstruction protocol regressions measured against JFG.
use filtergraph_algorithms::*;
use std::sync::Arc;

#[test]
fn incremental_offsets_are_strictly_increasing_coordinates_not_fixed_ordinals() {
    let plan = IncrementalDenseReconstructionPlan::new(
        PreparedGemv::new(2, 2, Arc::from([2.0_f32, 3.0, 4.0, 5.0])).unwrap(),
        1,
        3,
    )
    .unwrap();
    let mut workspace = plan.workspace();
    let mut output = [-99.0; 2];
    let info = |first_row, terminal| RowBlockInfo {
        sequence: 7,
        first_row,
        terminal,
        discontinuity: false,
        corrupted: false,
    };
    assert_eq!(
        plan.process_block(&mut output, &mut workspace, &[1.0], &[0], 1, info(0, false))
            .unwrap(),
        RowBlockResult::Pending
    );
    assert_eq!(output, [-99.0; 2]);
    assert_eq!(
        plan.process_block(&mut output, &mut workspace, &[2.0], &[1], 1, info(7, true))
            .unwrap(),
        RowBlockResult::Complete {
            sequence: 7,
            discontinuity: false
        }
    );
    assert_eq!(output, [8.0, 14.0]);
}

#[test]
fn incremental_reset_is_clean_and_terminal_may_complete_at_start() {
    let plan = IncrementalDenseReconstructionPlan::new(
        PreparedGemv::new(1, 2, Arc::from([2.0_f32, 3.0])).unwrap(),
        2,
        3,
    )
    .unwrap();
    let mut workspace = plan.workspace();
    let mut output = [-99.0];
    let info = |sequence, first_row, terminal| RowBlockInfo {
        sequence,
        first_row,
        terminal,
        discontinuity: false,
        corrupted: false,
    };
    plan.process_block(
        &mut output,
        &mut workspace,
        &[1.0, 0.0],
        &[0, 99],
        1,
        info(8, 0, false),
    )
    .unwrap();
    workspace.reset();
    assert_eq!(
        plan.process_block(
            &mut output,
            &mut workspace,
            &[1.0, 2.0],
            &[0, 1],
            2,
            info(9, 0, true)
        )
        .unwrap(),
        RowBlockResult::Complete {
            sequence: 9,
            discontinuity: false
        }
    );
    assert_eq!(output, [8.0]);
}

#[test]
fn missing_readout_metadata_abandons_partial_assembly_and_reconstruction() {
    macro_rules! check {
        ($declaration:ty,$configuration:expr,$extent:expr) => {{
            let mut prepared = <$declaration>::prepare($configuration).unwrap();
            let mut output = vec![-99.0; $extent];
            let block = [2.0; 8];
            let call = |p: &mut _, o: &mut [f32], sequence, ordinal| {
                <$declaration>::process_with_metadata(
                    p,
                    &[Some(
                        SampleMetadata::new(sequence, ordinal).with_terminal(ordinal == 2),
                    )],
                    &block,
                    o,
                )
                .unwrap()
            };
            assert!(!call(&mut prepared, &mut output, 1, 0).is_complete());
            assert!(
                <$declaration>::process_with_metadata(&mut prepared, &[None], &block, &mut output)
                    .is_err()
            );
            assert!(!call(&mut prepared, &mut output, 1, 1).is_complete());
            assert!(!call(&mut prepared, &mut output, 1, 2).is_complete());
            assert!(output.iter().all(|&v| v == -99.0));
            for ordinal in 0..3 {
                let result = call(&mut prepared, &mut output, 2, ordinal);
                assert_eq!(result.is_complete(), ordinal == 2);
                if ordinal == 2 {
                    assert!(result.metadata().unwrap().is_discontinuous());
                }
            }
        }};
    }
    check!(
        DetectorFrameAssemblerF32,
        DetectorFrameAssemblerF32Configuration {
            region_lines: 3,
            block_lines: 1,
            region_samples: 4,
            detector_origins: vec![[0, 0], [5, 3]],
            detector_line_steps: vec![[1, 0], [-1, 0]],
            detector_sample_steps: vec![[0, 1], [0, -1]],
            detector_size: (6, 4),
        },
        24
    );
    check!(
        ShackHartmannReadoutReconstructorF32,
        ShackHartmannReadoutReconstructorF32Configuration {
            region_lines: 3,
            block_lines: 1,
            region_samples: 4,
            detector_origins: vec![[0, 0], [5, 3]],
            detector_line_steps: vec![[1, 0], [-1, 0]],
            detector_sample_steps: vec![[0, 1], [0, -1]],
            subaperture_count: 3,
            reconstructed_count: 2,
            initial_subaperture_origins: Some(vec![[3, 0], [0, 2], [1, 0]]),
            coordinate_scale: 1.25,
            pixel_threshold: 1.0,
            flux_threshold: 0.5,
            reference_slopes: None,
            active: None,
            initial_reconstructor: Some(vec![1.0; 12]),
            reconstructed_schema: "test.frontend/reconstructed/1".into(),
            subaperture_size: (2, 2),
            detector_size: (6, 4),
        },
        2
    );
}

#[test]
fn incremental_missing_metadata_discards_partial_column_coverage() {
    let mut p =
        IncrementalDenseReconstructorF32::prepare(IncrementalDenseReconstructorF32Configuration {
            measurement_count: 2,
            measurement_block_capacity: 1,
            blocks_per_sample: 3,
            reconstructed_count: 1,
            initial_reconstructor: Some(vec![2.0, 3.0]),
            measurements_schema: "test.frontend/measurements/1".into(),
            column_indices_schema: "test.frontend/columns/1".into(),
            measurement_count_schema: "test.frontend/count/1".into(),
            reconstructed_schema: "test.frontend/reconstructed/1".into(),
            reconstructor_schema: "test.frontend/matrix/1".into(),
        })
        .unwrap();
    let mut output = [-99.0];
    let call = |p: &mut _, o: &mut [f32], sequence, offset, column, terminal| {
        IncrementalDenseReconstructorF32::process_with_metadata(
            p,
            &[
                Some(SampleMetadata::new(sequence, offset).with_terminal(terminal)),
                None,
                None,
            ],
            &[1.0],
            &[column],
            &[1],
            o,
        )
        .unwrap()
    };
    assert!(!call(&mut p, &mut output, 1, 0, 0, false).is_complete());
    assert!(
        IncrementalDenseReconstructorF32::process_with_metadata(
            &mut p,
            &[None, None, None],
            &[1.0],
            &[1],
            &[1],
            &mut output
        )
        .is_err()
    );
    assert!(!call(&mut p, &mut output, 1, 1, 1, true).is_complete());
    assert_eq!(output, [-99.0]);
    assert!(!call(&mut p, &mut output, 2, 0, 0, false).is_complete());
    let result = call(&mut p, &mut output, 2, 7, 1, true);
    assert_eq!(output, [5.0]);
    assert!(result.metadata().unwrap().is_discontinuous());
}
