//! Progressive multi-region pyramid reconstruction tests.

use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmPlan, DetectorReadoutBlockInfo, DetectorReadoutPlan, DetectorReadoutRegion,
    DetectorReadoutResult, PreparedGemv, PyramidPixelImagePlan, PyramidPixelOutput,
    PyramidReadoutReconstructionOutput, PyramidReadoutReconstructionPlan,
};

const RECONSTRUCTED: usize = 3;

fn info(sequence: u64, ordinal: usize, blocks: usize) -> DetectorReadoutBlockInfo {
    DetectorReadoutBlockInfo {
        sequence,
        ordinal,
        terminal: ordinal + 1 == blocks,
        discontinuity: false,
        corrupted: false,
    }
}

fn opposed_readout() -> DetectorReadoutPlan {
    DetectorReadoutPlan::new(
        (8, 4),
        4,
        1,
        4,
        [
            DetectorReadoutRegion::new((0, 0), (1, 0), (0, 1)),
            DetectorReadoutRegion::new((7, 0), (-1, 0), (0, 1)),
        ],
    )
    .unwrap()
}

fn opposed_image() -> PyramidPixelImagePlan<f32> {
    // Pupil-major order intentionally differs from acquisition readiness.
    PyramidPixelImagePlan::from_detector_pupils(
        (8, 4),
        (1, 2),
        &[(7, 0), (0, 0), (4, 2), (2, 2)],
        vec![true, false],
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
            regions.push(DetectorReadoutRegion::new(
                (
                    if reverse_lines {
                        first_row + 3
                    } else {
                        first_row
                    },
                    if reverse_samples {
                        first_column + 1
                    } else {
                        first_column
                    },
                ),
                (if reverse_lines { -1 } else { 1 }, 0),
                (0, if reverse_samples { -1 } else { 1 }),
            ));
        }
    }
    DetectorReadoutPlan::new((8, 8), 4, 2, 2, regions).unwrap()
}

fn eight_region_image() -> PyramidPixelImagePlan<f32> {
    PyramidPixelImagePlan::from_detector_pupils(
        (8, 8),
        (2, 2),
        &[(6, 6), (0, 0), (4, 2), (2, 4)],
        vec![true, false, true, true],
    )
    .unwrap()
}

fn matrix(columns: usize) -> PreparedGemv<f32> {
    let values = (0..RECONSTRUCTED * columns)
        .map(|index| ((index % 11) as f32 - 5.0) * 0.031_25)
        .collect::<Vec<_>>();
    PreparedGemv::new(RECONSTRUCTED, columns, Arc::from(values)).unwrap()
}

fn detector(rows: usize, columns: usize, sequence: u64) -> Vec<f32> {
    (0..rows * columns)
        .map(|index| index as f32 + 1.0 + sequence as f32 * 0.25)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 2.0e-6 * expected.abs().max(1.0),
            "{actual} does not match {expected}"
        );
    }
}

fn assert_complete_image_equivalence(
    image: PyramidPixelImagePlan<f32>,
    readout: DetectorReadoutPlan,
) {
    let selected = image.normalization().output_pixel_count();
    let plan =
        PyramidReadoutReconstructionPlan::new(image.clone(), matrix(selected), readout.clone())
            .unwrap();
    let mut full_workspace = image.workspace();
    let mut progressive_workspace = plan.workspace();
    let mut full_pixels = vec![0.0; selected];
    let mut full_mean = 0.0;
    let mut full_reconstructed = vec![0.0; RECONSTRUCTED];
    let mut progressive_pixels = vec![-1.0; selected];
    let mut progressive_mean = -1.0;
    let mut progressive_reconstructed = vec![-1.0; RECONSTRUCTED];

    for sequence in 1..=2 {
        let detector = detector(
            readout.detector_size().0,
            readout.detector_size().1,
            sequence,
        );
        image
            .process(
                PyramidPixelOutput::new(&mut full_pixels, &mut full_mean),
                &mut full_workspace,
                &detector,
            )
            .unwrap();
        plan.reconstructor()
            .canonical()
            .multiply(&mut full_reconstructed, &full_pixels)
            .unwrap();

        let packed = readout.pack_detector(&detector).unwrap();
        let old_pixels = progressive_pixels.clone();
        let old_mean = progressive_mean;
        let old_reconstructed = progressive_reconstructed.clone();
        for ordinal in 0..readout.blocks() {
            let start = ordinal * readout.block_elements();
            let result = plan
                .process_block(
                    PyramidReadoutReconstructionOutput::new(
                        &mut progressive_pixels,
                        &mut progressive_mean,
                        &mut progressive_reconstructed,
                    ),
                    &mut progressive_workspace,
                    &packed[start..start + readout.block_elements()],
                    info(sequence, ordinal, readout.blocks()),
                )
                .unwrap();
            if ordinal + 1 == readout.blocks() {
                assert_eq!(
                    result,
                    DetectorReadoutResult::Complete {
                        sequence,
                        discontinuity: false,
                    }
                );
            } else {
                assert_eq!(result, DetectorReadoutResult::Pending);
                assert_eq!(progressive_pixels, old_pixels);
                assert_eq!(progressive_mean, old_mean);
                assert_eq!(progressive_reconstructed, old_reconstructed);
            }
        }

        assert_eq!(progressive_pixels, full_pixels);
        assert_eq!(progressive_mean, full_mean);
        assert_close(&progressive_reconstructed, &full_reconstructed);
        assert_eq!(
            progressive_workspace.reciprocal_previous_frame_mean(),
            full_workspace.reciprocal_previous_frame_mean()
        );
    }
}

#[test]
fn opposed_regions_use_scientific_pixel_readiness_and_terminal_publication() {
    let readout = opposed_readout();
    let plan =
        PyramidReadoutReconstructionPlan::new(opposed_image(), matrix(4), readout.clone()).unwrap();

    assert_eq!(plan.pixel_completion_ordinals(), &[0, 0, 3, 2]);
    assert_eq!(plan.reconstructor().input_indices(0), Some(&[0, 1][..]));
    assert_eq!(plan.reconstructor().input_indices(1), Some(&[][..]));
    assert_eq!(plan.reconstructor().input_indices(2), Some(&[3][..]));
    assert_eq!(plan.reconstructor().input_indices(3), Some(&[2][..]));
    assert_complete_image_equivalence(opposed_image(), readout);
}

#[test]
fn eight_region_readout_matches_complete_image_science() {
    assert_complete_image_equivalence(eight_region_image(), eight_region_readout());
}

#[test]
fn failed_frame_preserves_committed_gain_and_marks_recovery() {
    let image = opposed_image();
    let readout = opposed_readout();
    let plan = PyramidReadoutReconstructionPlan::new(
        image.clone(),
        matrix(image.normalization().output_pixel_count()),
        readout.clone(),
    )
    .unwrap();
    let mut workspace = plan.workspace();
    let mut pixels = vec![0.0; 4];
    let mut mean = 0.0;
    let mut reconstructed = vec![0.0; RECONSTRUCTED];

    let first = detector(8, 4, 1);
    let packed = readout.pack_detector(&first).unwrap();
    for ordinal in 0..readout.blocks() {
        let start = ordinal * readout.block_elements();
        plan.process_block(
            PyramidReadoutReconstructionOutput::new(&mut pixels, &mut mean, &mut reconstructed),
            &mut workspace,
            &packed[start..start + readout.block_elements()],
            info(1, ordinal, readout.blocks()),
        )
        .unwrap();
    }
    let committed_gain = workspace.reciprocal_previous_frame_mean();
    let committed_pixels = pixels.clone();
    let committed_mean = mean;
    let committed_reconstructed = reconstructed.clone();

    plan.process_block(
        PyramidReadoutReconstructionOutput::new(&mut pixels, &mut mean, &mut reconstructed),
        &mut workspace,
        &packed[..readout.block_elements()],
        info(2, 0, readout.blocks()),
    )
    .unwrap();
    let mut corrupt = info(2, 1, readout.blocks());
    corrupt.corrupted = true;
    assert_eq!(
        plan.process_block(
            PyramidReadoutReconstructionOutput::new(&mut pixels, &mut mean, &mut reconstructed),
            &mut workspace,
            &packed[readout.block_elements()..2 * readout.block_elements()],
            corrupt,
        )
        .unwrap(),
        DetectorReadoutResult::Dropped(
            filtergraph_algorithms::DetectorReadoutDropReason::Corrupted
        )
    );
    assert_eq!(workspace.reciprocal_previous_frame_mean(), committed_gain);
    assert_eq!(pixels, committed_pixels);
    assert_eq!(mean, committed_mean);
    assert_eq!(reconstructed, committed_reconstructed);

    let recovery = detector(8, 4, 3);
    let packed = readout.pack_detector(&recovery).unwrap();
    let mut result = DetectorReadoutResult::Pending;
    for ordinal in 0..readout.blocks() {
        let start = ordinal * readout.block_elements();
        result = plan
            .process_block(
                PyramidReadoutReconstructionOutput::new(&mut pixels, &mut mean, &mut reconstructed),
                &mut workspace,
                &packed[start..start + readout.block_elements()],
                info(3, ordinal, readout.blocks()),
            )
            .unwrap();
    }
    assert_eq!(
        result,
        DetectorReadoutResult::Complete {
            sequence: 3,
            discontinuity: true,
        }
    );
}

#[test]
fn masked_detector_sample_still_participates_in_frame_mean_before_publication() {
    use filtergraph_algorithms::{PyramidPixelError, PyramidReadoutReconstructionError};

    let image = opposed_image();
    let readout = opposed_readout();
    let plan = PyramidReadoutReconstructionPlan::new(image, matrix(4), readout.clone()).unwrap();
    let mut workspace = plan.workspace();
    let mut pixels = [71.0; 4];
    let mut mean = 71.0;
    let mut reconstructed = [71.0; RECONSTRUCTED];

    // The second pixel of the first pupil is masked from reconstruction.
    // It is still part of the four-pupil normalization mean.
    let mut invalid = [2.0_f32; 32];
    invalid[7 * 4 + 1] = f32::NAN;
    for (sequence, detector) in [(51, invalid), (52, [4.0; 32])] {
        let packed = readout.pack_detector(&detector).unwrap();
        for (ordinal, block) in packed.chunks_exact(readout.block_elements()).enumerate() {
            let result = plan.process_block(
                PyramidReadoutReconstructionOutput::new(&mut pixels, &mut mean, &mut reconstructed),
                &mut workspace,
                block,
                info(sequence, ordinal, readout.blocks()),
            );
            match (sequence, ordinal) {
                (51, 3) => {
                    let error = result.unwrap_err();
                    assert_eq!(
                        error,
                        PyramidReadoutReconstructionError::Pixels(
                            PyramidPixelError::NonFiniteInput { pupil: 0, pixel: 1 }
                        )
                    );
                    assert!(error.to_string().contains("pupil 0"));
                    assert_eq!(workspace.sequence(), None);
                    assert_eq!(workspace.reciprocal_previous_frame_mean(), 0.0);
                    assert_eq!(
                        (pixels, mean, reconstructed),
                        ([71.0; 4], 71.0, [71.0; RECONSTRUCTED])
                    );
                }
                (52, 3) => {
                    assert_eq!(
                        result,
                        Ok(DetectorReadoutResult::Complete {
                            sequence: 52,
                            discontinuity: true
                        })
                    );
                    assert_eq!((pixels, mean), ([0.0; 4], 4.0));
                    assert_eq!(workspace.reciprocal_previous_frame_mean(), 0.25);
                }
                _ => {
                    assert_eq!(result, Ok(DetectorReadoutResult::Pending));
                    assert_eq!(
                        (pixels, mean, reconstructed),
                        ([71.0; 4], 71.0, [71.0; RECONSTRUCTED])
                    );
                }
            }
        }
    }
}

#[test]
fn pyramid_readout_checks_format_compatibility_before_assembly() {
    use filtergraph_algorithms::{
        PyramidReadoutReconstructionError as Execution,
        PyramidReadoutReconstructionPlanError as Preparation,
    };
    let readout = opposed_readout();
    let error =
        PyramidReadoutReconstructionPlan::new(eight_region_image(), matrix(12), readout.clone())
            .unwrap_err();
    assert!(matches!(error, Preparation::ImageExtent { .. }), "{error}");
    let plan =
        PyramidReadoutReconstructionPlan::new(opposed_image(), matrix(4), readout.clone()).unwrap();
    let mut workspace = plan.workspace();
    let mut pixels = [71.0; 4];
    let mut mean = 71.0;
    let mut reconstructed = [71.0; RECONSTRUCTED];
    for (npixels, nreconstructed, ninput, expected) in [
        (
            3,
            RECONSTRUCTED,
            8,
            Execution::PixelOutputExtent {
                expected: 4,
                actual: 3,
            },
        ),
        (
            4,
            2,
            8,
            Execution::ReconstructedOutputExtent {
                expected: RECONSTRUCTED,
                actual: 2,
            },
        ),
        (
            4,
            RECONSTRUCTED,
            7,
            Execution::InputExtent {
                expected: 8,
                actual: 7,
            },
        ),
    ] {
        let error = plan
            .process_block(
                PyramidReadoutReconstructionOutput::new(
                    &mut pixels[..npixels],
                    &mut mean,
                    &mut reconstructed[..nreconstructed],
                ),
                &mut workspace,
                &[2.0; 8][..ninput],
                info(53, 0, readout.blocks()),
            )
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains("extent"));
        assert_eq!(workspace.sequence(), None);
        assert_eq!(
            (pixels, mean, reconstructed),
            ([71.0; 4], 71.0, [71.0; RECONSTRUCTED])
        );
    }
}

#[test]
fn normalization_overflow_and_reconstruction_overflow_leave_the_last_frame_committed() {
    use filtergraph_algorithms::{PyramidPixelError, PyramidReadoutReconstructionError as Error};

    let readout = opposed_readout();
    let plan = PyramidReadoutReconstructionPlan::new(
        opposed_image(),
        PreparedGemv::new(3, 4, Arc::from([f32::MAX; 12])).unwrap(),
        readout.clone(),
    )
    .unwrap();
    let safe_plan =
        PyramidReadoutReconstructionPlan::new(opposed_image(), matrix(4), readout.clone()).unwrap();
    let mut workspace = plan.workspace();
    let mut pixels = [71.0; 4];
    let mut mean = 71.0;
    let mut modes = [71.0; 3];
    let mut first = [4.0_f32; 32];
    first[0] = f32::MAX;
    first[1] = f32::MAX;
    for (sequence, image, reconstruct, expected) in [
        (
            71,
            first,
            &safe_plan,
            Some(Error::Pixels(PyramidPixelError::NonFiniteMean)),
        ),
        (72, [0.5; 32], &safe_plan, None),
        (73, [8.0; 32], &plan, None),
    ] {
        let packed = readout.pack_detector(&image).unwrap();
        for (ordinal, block) in packed.chunks_exact(readout.block_elements()).enumerate() {
            let result = reconstruct.process_block(
                PyramidReadoutReconstructionOutput::new(&mut pixels, &mut mean, &mut modes),
                &mut workspace,
                block,
                info(sequence, ordinal, readout.blocks()),
            );
            if sequence == 73 {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::Reconstruction(_)), "{error}");
                assert_eq!(mean, 0.5);
                assert_eq!(pixels, [0.0; 4]);
                assert_eq!(modes, [0.0; 3]);
                assert_eq!(workspace.reciprocal_previous_frame_mean(), 2.0);
                assert_eq!(workspace.sequence(), None);
                break;
            } else if ordinal + 1 < readout.blocks() {
                assert_eq!(result, Ok(DetectorReadoutResult::Pending));
                assert_eq!(mean, 71.0);
            } else if let Some(expected) = expected {
                let error = result.unwrap_err();
                assert_eq!(error, expected);
                assert!(error.to_string().contains("mean"), "{error}");
                assert_eq!((mean, pixels, modes), (71.0, [71.0; 4], [71.0; 3]));
            } else {
                assert_eq!(
                    result,
                    Ok(DetectorReadoutResult::Complete {
                        sequence,
                        discontinuity: true
                    })
                );
                assert_eq!((mean, pixels, modes), (0.5, [0.0; 4], [0.0; 3]));
            }
        }
    }
}
