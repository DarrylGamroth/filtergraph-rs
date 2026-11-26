//! Multi-region detector mapping, calibration, assembly, and recovery tests.

use filtergraph_algorithms::{
    DetectorFrameAssemblerPlan, DetectorReadoutBlockInfo, DetectorReadoutCalibrationPlan,
    DetectorReadoutDropReason, DetectorReadoutError, DetectorReadoutPlan, DetectorReadoutRegion,
    DetectorReadoutResult,
};

fn info(sequence: u64, ordinal: usize, terminal: bool) -> DetectorReadoutBlockInfo {
    DetectorReadoutBlockInfo {
        sequence,
        ordinal,
        terminal,
        discontinuity: false,
        corrupted: false,
    }
}

fn row_readout() -> DetectorReadoutPlan {
    DetectorReadoutPlan::new(
        (4, 4),
        4,
        2,
        4,
        [DetectorReadoutRegion::new((0, 0), (1, 0), (0, 1))],
    )
    .unwrap()
}

fn opposed_readout() -> DetectorReadoutPlan {
    DetectorReadoutPlan::new(
        (4, 4),
        2,
        1,
        4,
        [
            DetectorReadoutRegion::new((0, 0), (1, 0), (0, 1)),
            DetectorReadoutRegion::new((3, 0), (-1, 0), (0, 1)),
        ],
    )
    .unwrap()
}

fn eight_region_readout() -> DetectorReadoutPlan {
    let mut regions = Vec::new();
    for tile_row in 0..2 {
        for tile_column in 0..4 {
            let reverse_lines = (tile_row + tile_column) % 2 != 0;
            let reverse_samples = tile_column % 2 != 0;
            let first_row = tile_row * 4;
            let first_column = tile_column * 2;
            let origin_row = if reverse_lines {
                first_row + 3
            } else {
                first_row
            };
            let origin_column = if reverse_samples {
                first_column + 1
            } else {
                first_column
            };
            regions.push(DetectorReadoutRegion::new(
                (origin_row, origin_column),
                (if reverse_lines { -1 } else { 1 }, 0),
                (0, if reverse_samples { -1 } else { 1 }),
            ));
        }
    }
    DetectorReadoutPlan::new((8, 8), 4, 2, 2, regions).unwrap()
}

#[test]
fn row_mapping_is_the_single_region_compatibility_case() {
    let plan = row_readout();
    let detector = (0_u16..16).collect::<Vec<_>>();

    assert_eq!(plan.detector_size(), (4, 4));
    assert_eq!(plan.regions().len(), 1);
    assert_eq!(plan.blocks(), 2);
    assert_eq!(plan.block_elements(), 8);
    assert_eq!(&*plan.pack_detector(&detector).unwrap(), detector);
    assert_eq!(plan.arrival_ordinal(0, 3), Some(0));
    assert_eq!(plan.arrival_ordinal(3, 0), Some(1));
}

#[test]
fn opposed_regions_arrive_from_both_detector_edges() {
    let plan = opposed_readout();
    let detector = (0_u16..16).collect::<Vec<_>>();
    let packed = plan.pack_detector(&detector).unwrap();

    assert_eq!(&packed[..8], &[0, 1, 2, 3, 12, 13, 14, 15]);
    assert_eq!(&packed[8..], &[4, 5, 6, 7, 8, 9, 10, 11]);
    assert_eq!(plan.arrival_ordinal(0, 0), Some(0));
    assert_eq!(plan.arrival_ordinal(3, 3), Some(0));
    assert_eq!(plan.arrival_ordinal(1, 0), Some(1));
    assert_eq!(plan.arrival_ordinal(2, 3), Some(1));
}

#[test]
fn transposed_mapping_uses_coordinate_steps_not_memory_stride() {
    let plan = DetectorReadoutPlan::new(
        (2, 3),
        3,
        1,
        2,
        [DetectorReadoutRegion::new((0, 0), (0, 1), (1, 0))],
    )
    .unwrap();
    let detector = [0_u16, 1, 2, 3, 4, 5];

    assert_eq!(
        &*plan.pack_detector(&detector).unwrap(),
        &[0, 3, 1, 4, 2, 5]
    );
}

#[test]
fn eight_regions_form_one_complete_detector_bijection() {
    let plan = eight_region_readout();
    let detector = (0_u16..64).collect::<Vec<_>>();
    let packed = plan.pack_detector(&detector).unwrap();
    let mut assembled = [u16::MAX; 64];

    for ordinal in 0..plan.blocks() {
        let start = ordinal * plan.block_elements();
        plan.scatter_block(
            &mut assembled,
            &packed[start..start + plan.block_elements()],
            ordinal,
        )
        .unwrap();
    }

    assert_eq!(assembled, detector.as_slice());
    assert_eq!(plan.regions().len(), 8);
    assert_eq!(plan.blocks(), 2);
}

#[test]
fn preparation_rejects_invalid_geometry() {
    assert_eq!(
        DetectorReadoutPlan::new(
            (2, 2),
            1,
            1,
            2,
            [
                DetectorReadoutRegion::new((0, 0), (1, 0), (0, 1)),
                DetectorReadoutRegion::new((0, 0), (1, 0), (0, 1)),
            ],
        )
        .unwrap_err(),
        DetectorReadoutError::DuplicateCoordinate { row: 0, column: 0 }
    );
    assert_eq!(
        DetectorReadoutPlan::new(
            (2, 2),
            2,
            1,
            2,
            [DetectorReadoutRegion::new((0, 0), (-1, 0), (0, 1))],
        )
        .unwrap_err(),
        DetectorReadoutError::CoordinateRange { region: 0 }
    );
    assert_eq!(
        DetectorReadoutPlan::new(
            (2, 2),
            2,
            2,
            2,
            [DetectorReadoutRegion::new((0, 0), (0, 0), (0, 1))],
        )
        .unwrap_err(),
        DetectorReadoutError::ZeroLineStep { region: 0 }
    );
    assert_eq!(
        DetectorReadoutPlan::new(
            (2, 2),
            2,
            2,
            2,
            [DetectorReadoutRegion::new((0, 0), (1, 0), (0, 0))],
        )
        .unwrap_err(),
        DetectorReadoutError::ZeroSampleStep { region: 0 }
    );
    assert_eq!(
        DetectorReadoutPlan::new(
            (2, 2),
            2,
            3,
            2,
            [DetectorReadoutRegion::new((0, 0), (1, 0), (0, 1))],
        )
        .unwrap_err(),
        DetectorReadoutError::BlockLines
    );
}

#[test]
fn frame_assembly_publishes_only_a_valid_terminal_image() {
    let readout = opposed_readout();
    let detector = (0_u16..16).collect::<Vec<_>>();
    let packed = readout.pack_detector(&detector).unwrap();
    let plan = DetectorFrameAssemblerPlan::new(readout.clone());
    let mut workspace = plan.workspace::<u16>();
    let mut output = [u16::MAX; 16];

    assert_eq!(
        plan.process_block(&mut output, &mut workspace, &packed[..8], info(7, 0, false),),
        Ok(DetectorReadoutResult::Pending)
    );
    assert_eq!(output, [u16::MAX; 16]);
    assert_eq!(workspace.sequence(), Some(7));
    assert_eq!(
        plan.process_block(&mut output, &mut workspace, &packed[8..], info(7, 1, true),),
        Ok(DetectorReadoutResult::Complete {
            sequence: 7,
            discontinuity: false,
        })
    );
    assert_eq!(output, detector.as_slice());
    assert_eq!(workspace.sequence(), None);
}

#[test]
fn frame_admission_drops_faults_and_marks_recovery() {
    let readout = opposed_readout();
    let detector = (0_u16..16).collect::<Vec<_>>();
    let packed = readout.pack_detector(&detector).unwrap();
    let plan = DetectorFrameAssemblerPlan::new(readout);
    let mut workspace = plan.workspace::<u16>();
    let mut output = [99_u16; 16];

    assert_eq!(
        plan.process_block(&mut output, &mut workspace, &packed[8..], info(1, 1, true),),
        Ok(DetectorReadoutResult::Dropped(
            DetectorReadoutDropReason::MissingStart
        ))
    );
    assert_eq!(output, [99; 16]);

    assert_eq!(
        plan.process_block(&mut output, &mut workspace, &packed[..8], info(2, 0, false),),
        Ok(DetectorReadoutResult::Pending)
    );
    assert_eq!(
        plan.process_block(&mut output, &mut workspace, &packed[8..], info(3, 1, true),),
        Ok(DetectorReadoutResult::Dropped(
            DetectorReadoutDropReason::Sequence
        ))
    );
    assert_eq!(output, [99; 16]);

    plan.process_block(&mut output, &mut workspace, &packed[..8], info(4, 0, false))
        .unwrap();
    assert_eq!(
        plan.process_block(&mut output, &mut workspace, &packed[8..], info(4, 1, true),),
        Ok(DetectorReadoutResult::Complete {
            sequence: 4,
            discontinuity: true,
        })
    );
    assert_eq!(output, detector.as_slice());
}

#[test]
fn calibration_is_packed_off_path_and_assembles_to_the_reference_image() {
    let readout = opposed_readout();
    let raw_detector = vec![3_u16; 16];
    let flat = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
    let background = vec![1.0_f32; 16];
    let calibration =
        DetectorReadoutCalibrationPlan::new(readout.clone(), &flat, &background).unwrap();
    let packed_raw = readout.pack_detector(&raw_detector).unwrap();
    let assembler = DetectorFrameAssemblerPlan::new(readout.clone());
    let mut workspace = assembler.workspace::<f32>();
    let mut detector_output = [-1.0_f32; 16];
    let mut calibrated = vec![0.0_f32; readout.block_elements()];

    for ordinal in 0..readout.blocks() {
        let start = ordinal * readout.block_elements();
        let terminal = ordinal + 1 == readout.blocks();
        let block_info = info(11, ordinal, terminal);
        calibration
            .process_block(
                &mut calibrated,
                &packed_raw[start..start + readout.block_elements()],
                block_info,
            )
            .unwrap();
        let result = assembler
            .process_block(
                &mut detector_output,
                &mut workspace,
                &calibrated,
                block_info,
            )
            .unwrap();
        assert_eq!(
            result,
            if terminal {
                DetectorReadoutResult::Complete {
                    sequence: 11,
                    discontinuity: false,
                }
            } else {
                DetectorReadoutResult::Pending
            }
        );
    }

    let expected = flat.iter().map(|value| value * 2.0).collect::<Vec<_>>();
    assert_eq!(detector_output, expected.as_slice());
    assert_ne!(calibration.packed_flat(), flat.as_slice());
}

#[test]
fn invalid_readout_storage_is_failure_atomic_and_reports_the_wrong_extent() {
    use filtergraph_algorithms::{DetectorFrameAssemblerError, DetectorReadoutDataError};
    let readout = opposed_readout();
    let mut output = [99_u16; 16];
    let block = [1; 8];

    let error = readout.pack_detector(&output[..15]).unwrap_err();
    assert_eq!(
        error,
        DetectorReadoutDataError::DetectorExtent {
            expected: 16,
            actual: 15
        }
    );
    assert!(error.to_string().contains("15") && error.to_string().contains("16"));
    for (detector_len, samples, ordinal, expected) in [
        (
            15,
            &block[..],
            0,
            DetectorReadoutDataError::DetectorExtent {
                expected: 16,
                actual: 15,
            },
        ),
        (
            16,
            &block[..7],
            0,
            DetectorReadoutDataError::BlockExtent {
                expected: 8,
                actual: 7,
            },
        ),
        (
            16,
            &block[..],
            2,
            DetectorReadoutDataError::OrdinalRange {
                ordinal: 2,
                blocks: 2,
            },
        ),
    ] {
        let error = readout
            .scatter_block(&mut output[..detector_len], samples, ordinal)
            .unwrap_err();
        assert_eq!(error, expected);
        assert_eq!(output, [99; 16]);
    }

    let plan = DetectorFrameAssemblerPlan::new(readout);
    let mut workspace = plan.workspace::<u16>();
    assert_eq!(
        plan.process_block(&mut output, &mut workspace, &block, info(7, 0, false)),
        Ok(DetectorReadoutResult::Pending)
    );
    for (output_len, samples, expected, diagnostic) in [
        (
            15,
            &block[..],
            DetectorFrameAssemblerError::OutputExtent {
                expected: 16,
                actual: 15,
            },
            "output extent 15 does not match 16",
        ),
        (
            16,
            &block[..7],
            DetectorFrameAssemblerError::InputExtent {
                expected: 8,
                actual: 7,
            },
            "block extent 7 does not match 8",
        ),
    ] {
        let error = plan
            .process_block(
                &mut output[..output_len],
                &mut workspace,
                samples,
                info(7, 1, true),
            )
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(diagnostic));
        assert_eq!(workspace.sequence(), Some(7));
        assert_eq!(output, [99; 16]);
    }
    assert_eq!(
        plan.process_block(&mut output, &mut workspace, &[2; 8], info(7, 1, true)),
        Ok(DetectorReadoutResult::Complete {
            sequence: 7,
            discontinuity: false
        })
    );
    assert_eq!(output, [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]);
}

#[test]
fn invalid_block_metadata_abandons_partial_frames_and_recovery_is_marked_once() {
    use DetectorReadoutDropReason::*;
    use DetectorReadoutResult::*;

    let plan = DetectorFrameAssemblerPlan::new(eight_region_readout());
    let mut workspace = plan.workspace::<u16>();
    let mut output = [99_u16; 64];
    let block = [1; 32];
    for (bad, reason) in [
        (
            DetectorReadoutBlockInfo {
                corrupted: true,
                ..info(10, 1, true)
            },
            Corrupted,
        ),
        (
            DetectorReadoutBlockInfo {
                discontinuity: true,
                ..info(10, 1, true)
            },
            MidFrameDiscontinuity,
        ),
        (info(10, 2, true), OrdinalRange),
        (info(10, 1, false), Terminal),
    ] {
        assert_eq!(
            plan.process_block(&mut output, &mut workspace, &block, info(10, 0, false)),
            Ok(Pending)
        );
        assert_eq!(
            plan.process_block(&mut output, &mut workspace, &block, bad),
            Ok(Dropped(reason))
        );
        assert_eq!(workspace.sequence(), None);
        assert_eq!(output, [99; 64]);
    }
    for (sequence, discontinuity) in [(11, true), (12, false)] {
        assert_eq!(
            plan.process_block(
                &mut output,
                &mut workspace,
                &block,
                info(sequence, 0, false)
            ),
            Ok(Pending)
        );
        assert_eq!(
            plan.process_block(&mut output, &mut workspace, &block, info(sequence, 1, true)),
            Ok(Complete {
                sequence,
                discontinuity
            })
        );
    }
    assert_eq!(output, [1; 64]);
}

#[test]
fn geometry_diagnostics_distinguish_missing_pixels_from_invalid_coordinate_steps() {
    use DetectorReadoutError::*;
    let normal = DetectorReadoutRegion::new((0, 0), (1, 0), (0, 1));
    for (size, lines, block, samples, region, expected, detail) in [
        ((0, 2), 1, 1, 2, normal, EmptyExtent, "positive"),
        (
            (usize::MAX, 2),
            2,
            1,
            2,
            normal,
            ExtentOverflow,
            "extent overflow",
        ),
        (
            (2, 2),
            2,
            1,
            1,
            normal,
            ImageExtent {
                readout: 2,
                detector: 4,
            },
            "pixel extent 2 does not match detector extent 4",
        ),
        (
            (2, 2),
            2,
            2,
            2,
            DetectorReadoutRegion::new((0, 0), (1, 0), (0, -1)),
            CoordinateRange { region: 0 },
            "region 0 exceeds",
        ),
        (
            (2, 2),
            2,
            1,
            2,
            DetectorReadoutRegion::new((0, 0), (0, 1), (0, 1)),
            DuplicateCoordinate { row: 0, column: 1 },
            "(0, 1) is mapped more than once",
        ),
    ] {
        let error = DetectorReadoutPlan::new(size, lines, block, samples, [region]).unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
    }
}
