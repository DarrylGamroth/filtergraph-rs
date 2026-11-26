//! Progressive multi-region Shack-Hartmann reconstruction tests.

use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmPlan, DetectorReadoutBlockInfo, DetectorReadoutDropReason, DetectorReadoutPlan,
    DetectorReadoutRegion, DetectorReadoutResult, PreparedGemv, ShackHartmannImagePlan,
    ShackHartmannOutput, ShackHartmannReadoutReconstructionPlan,
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

fn canonical_result(
    sensing: &ShackHartmannImagePlan<f32>,
    reconstructor: &PreparedGemv<f32>,
    image: &[f32],
) -> Vec<f32> {
    let subapertures = sensing.measurement().len();
    let mut slopes = vec![0.0_f32; subapertures * 2];
    let mut flux = vec![0.0_f32; subapertures];
    let mut validity = vec![false; subapertures];
    sensing
        .process(
            ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
            &mut sensing.workspace(),
            image,
        )
        .unwrap();
    assert!(validity.iter().all(|&valid| valid));
    let mut output = vec![0.0_f32; reconstructor.rows()];
    reconstructor.multiply(&mut output, &slopes).unwrap();
    output
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2.0e-5,
            "coordinate {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn opposed_regions_drive_scientific_readiness_and_packed_columns() {
    let origins = [
        (1, 0),
        (0, 0),
        (2, 0),
        (3, 0),
        (1, 2),
        (0, 2),
        (2, 2),
        (3, 2),
    ];
    let sensing = ShackHartmannImagePlan::from_detector_regions(
        (4, 4),
        (1, 2),
        &origins,
        1.0,
        0.0,
        0.0,
        None,
        None,
    )
    .unwrap();
    let slopes = origins.len() * 2;
    let matrix = (0..3 * slopes)
        .map(|index| ((index / slopes + 1) * (index % slopes + 1)) as f32 / 32.0)
        .collect::<Vec<_>>();
    let canonical = PreparedGemv::new(3, slopes, Arc::from(matrix)).unwrap();
    let readout = opposed_readout();
    let plan = ShackHartmannReadoutReconstructionPlan::new(
        sensing.clone(),
        canonical.clone(),
        readout.clone(),
    )
    .unwrap();

    assert_eq!(
        plan.subaperture_completion_ordinals(),
        [1, 0, 1, 0, 1, 0, 1, 0]
    );
    assert_eq!(
        plan.reconstructor().input_indices(0),
        Some([2, 3, 6, 7, 10, 11, 14, 15].as_slice())
    );
    assert_eq!(
        plan.reconstructor().canonical().matrix(),
        canonical.matrix()
    );

    let image = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
    let expected = canonical_result(&sensing, &canonical, &image);
    let packed = readout.pack_detector(&image).unwrap();
    let mut output = [-99.0_f32; 3];
    let mut workspace = plan.workspace();

    assert_eq!(
        plan.process_block(
            &mut output,
            &mut workspace,
            &packed[..readout.block_elements()],
            info(9, 0, false),
        ),
        Ok(DetectorReadoutResult::Pending)
    );
    assert_eq!(output, [-99.0; 3]);
    assert_eq!(workspace.measured_subapertures(), 4);
    assert!(workspace.accumulated().iter().any(|&value| value != 0.0));

    assert_eq!(
        plan.process_block(
            &mut output,
            &mut workspace,
            &packed[readout.block_elements()..],
            info(9, 1, true),
        ),
        Ok(DetectorReadoutResult::Complete {
            sequence: 9,
            discontinuity: false,
        })
    );
    assert_close(&output, &expected);
    assert_eq!(workspace.sequence(), None);
}

#[test]
fn eight_region_reconstruction_matches_complete_image_processing() {
    let readout = eight_region_readout();
    let mut origins = Vec::new();
    for row in 0..8 {
        for column in (0..8).step_by(2) {
            origins.push((row, column));
        }
    }
    origins.rotate_left(11);
    let sensing = ShackHartmannImagePlan::from_detector_regions(
        (8, 8),
        (1, 2),
        &origins,
        0.5,
        0.0,
        0.0,
        None,
        None,
    )
    .unwrap();
    let slopes = origins.len() * 2;
    let matrix = (0..2 * slopes)
        .map(|index| ((index % 13) as f32 - 6.0) / 16.0)
        .collect::<Vec<_>>();
    let canonical = PreparedGemv::new(2, slopes, Arc::from(matrix)).unwrap();
    let plan = ShackHartmannReadoutReconstructionPlan::new(
        sensing.clone(),
        canonical.clone(),
        readout.clone(),
    )
    .unwrap();
    let image = (1..=64).map(|value| value as f32).collect::<Vec<_>>();
    let expected = canonical_result(&sensing, &canonical, &image);
    let packed = readout.pack_detector(&image).unwrap();
    let mut output = [0.0_f32; 2];
    let mut workspace = plan.workspace();

    for ordinal in 0..readout.blocks() {
        let start = ordinal * readout.block_elements();
        plan.process_block(
            &mut output,
            &mut workspace,
            &packed[start..start + readout.block_elements()],
            info(4, ordinal, ordinal + 1 == readout.blocks()),
        )
        .unwrap();
    }

    assert_close(&output, &expected);
}

#[test]
fn acquisition_failure_preserves_output_and_marks_recovery() {
    let origins = [(0, 0), (1, 0), (2, 0), (3, 0)];
    let sensing = ShackHartmannImagePlan::from_detector_regions(
        (4, 4),
        (1, 2),
        &origins,
        1.0,
        0.0,
        0.0,
        None,
        None,
    )
    .unwrap();
    let canonical = PreparedGemv::new(1, 8, Arc::from([1.0_f32; 8])).unwrap();
    let readout = opposed_readout();
    let plan =
        ShackHartmannReadoutReconstructionPlan::new(sensing, canonical, readout.clone()).unwrap();
    let image = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
    let packed = readout.pack_detector(&image).unwrap();
    let mut output = [77.0_f32];
    let mut workspace = plan.workspace();

    plan.process_block(
        &mut output,
        &mut workspace,
        &packed[..readout.block_elements()],
        info(1, 0, false),
    )
    .unwrap();
    assert_eq!(
        plan.process_block(
            &mut output,
            &mut workspace,
            &packed[readout.block_elements()..],
            info(2, 1, true),
        ),
        Ok(DetectorReadoutResult::Dropped(
            DetectorReadoutDropReason::Sequence
        ))
    );
    assert_eq!(output, [77.0]);

    plan.process_block(
        &mut output,
        &mut workspace,
        &packed[..readout.block_elements()],
        info(3, 0, false),
    )
    .unwrap();
    assert!(matches!(
        plan.process_block(
            &mut output,
            &mut workspace,
            &packed[readout.block_elements()..],
            info(3, 1, true),
        ),
        Ok(DetectorReadoutResult::Complete {
            sequence: 3,
            discontinuity: true,
        })
    ));
    assert_ne!(output, [77.0]);
}

#[test]
fn wrong_stream_dimensions_do_not_consume_a_block_and_overflow_drops_the_frame() {
    use filtergraph_algorithms::{
        ShackHartmannReadoutReconstructionError as Execution,
        ShackHartmannReadoutReconstructionPlanError as Preparation,
    };
    let sensing = ShackHartmannImagePlan::from_detector_regions(
        (4, 4),
        (1, 2),
        &[(0, 0), (1, 0)],
        1.0_f32,
        0.0,
        0.0,
        Some(vec![-4.0; 4]),
        None,
    )
    .unwrap();
    let readout = opposed_readout();
    let wrong_matrix = PreparedGemv::new(1, 2, Arc::from([1.0; 2])).unwrap();
    let error =
        ShackHartmannReadoutReconstructionPlan::new(sensing.clone(), wrong_matrix, readout.clone())
            .unwrap_err();
    assert_eq!(
        error,
        Preparation::ReconstructorColumns {
            expected: 4,
            actual: 2
        }
    );
    assert!(error.to_string().contains("column count 2"));

    let overflow_matrix = PreparedGemv::new(1, 4, Arc::from([f32::MAX; 4])).unwrap();
    let overflow_plan = ShackHartmannReadoutReconstructionPlan::new(
        sensing.clone(),
        overflow_matrix,
        readout.clone(),
    )
    .unwrap();
    let valid_matrix = PreparedGemv::new(1, 4, Arc::from([1.0; 4])).unwrap();
    let valid_plan =
        ShackHartmannReadoutReconstructionPlan::new(sensing, valid_matrix, readout).unwrap();
    let mut workspace = overflow_plan.workspace();
    let mut output = [71.0];
    let input = [2.0; 8];
    for (out, samples, expected) in [
        (
            0,
            &input[..],
            Execution::OutputExtent {
                expected: 1,
                actual: 0,
            },
        ),
        (
            1,
            &input[..7],
            Execution::InputExtent {
                expected: 8,
                actual: 7,
            },
        ),
    ] {
        let error = overflow_plan
            .process_block(
                &mut output[..out],
                &mut workspace,
                samples,
                info(61, 0, false),
            )
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains("extent"));
        assert_eq!(workspace.sequence(), None);
        assert_eq!(output, [71.0]);
    }

    let error = overflow_plan
        .process_block(&mut output, &mut workspace, &input, info(61, 0, false))
        .unwrap_err();
    assert!(matches!(error, Execution::Reconstruction(_)), "{error}");
    assert_eq!(workspace.sequence(), None);
    assert_eq!(output, [71.0]);
    assert_eq!(
        valid_plan.process_block(&mut output, &mut workspace, &input, info(62, 0, false)),
        Ok(DetectorReadoutResult::Pending)
    );
    assert_eq!(
        valid_plan.process_block(&mut output, &mut workspace, &input, info(62, 1, true)),
        Ok(DetectorReadoutResult::Complete {
            sequence: 62,
            discontinuity: true
        })
    );
    assert_eq!(output, [16.0]);
}
