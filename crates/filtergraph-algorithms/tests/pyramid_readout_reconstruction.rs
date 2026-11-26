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
