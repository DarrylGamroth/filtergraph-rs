//! Rectangular region-extraction parity and layout tests.

use filtergraph_algorithms::schemas::{CALIBRATED_PIXELS_V1, REGION_ORIGINS_V1, REGION_PIXELS_V1};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, DeclarationParameterError, InputProgress,
    ProcessError, ProgressResult, RegionExtractionError, RegionExtractionF32,
    RegionExtractionF32Configuration, RegionExtractionPlan, RegionPixels, RegionPixelsError,
    RegionPixelsMut,
};

fn compact_plan() -> RegionExtractionPlan<f32> {
    RegionExtractionPlan::new((2, 4), &[(0, 0), (0, 2)], (2, 2)).unwrap()
}

fn padded_plan() -> RegionExtractionPlan<f32> {
    RegionExtractionPlan::with_layout((2, 4), &[(0, 0), (0, 2)], (2, 2), 6, 2).unwrap()
}

#[test]
fn preparation_fixes_zero_based_geometry_and_completion_indices() {
    let compact = compact_plan();
    assert_eq!(compact.image_size(), (2, 4));
    assert_eq!(compact.region_size(), (2, 2));
    assert_eq!(compact.row_stride(), 4);
    assert_eq!(compact.first_pixel(), 0);
    assert_eq!(compact.origins(), [(0, 0), (0, 2)]);
    assert_eq!(compact.completion_indices(), [5, 7]);

    let padded = padded_plan();
    assert_eq!(padded.row_stride(), 6);
    assert_eq!(padded.first_pixel(), 2);
    assert_eq!(padded.completion_indices(), [9, 11]);
}

#[test]
fn preparation_rejects_incompatible_detector_geometry() {
    assert_eq!(
        RegionExtractionPlan::<f32>::new((2, 4), &[], (1, 1)).unwrap_err(),
        RegionExtractionError::EmptyOrigins
    );
    assert_eq!(
        RegionExtractionPlan::<f32>::with_layout((2, 4), &[(0, 0)], (1, 1), 3, 0).unwrap_err(),
        RegionExtractionError::RowStrideTooSmall
    );
    assert_eq!(
        RegionExtractionPlan::<f32>::new((2, 4), &[(1, 3)], (2, 1)).unwrap_err(),
        RegionExtractionError::RegionOutOfBounds { region: 0 }
    );
}

#[test]
fn complete_extraction_supports_compact_and_padded_layouts() {
    let calibrated = [6.0_f32, 2.0, 4.0, 3.0, 6.0, 5.0, 4.0, 6.0];
    let mut compact_output = [0.0_f32; 8];
    let compact = compact_plan();
    compact
        .process(
            RegionPixelsMut::new(&mut compact_output, 4, 2).unwrap(),
            &mut compact.workspace(),
            &calibrated,
        )
        .unwrap();
    assert_eq!(compact_output, [6.0, 2.0, 6.0, 5.0, 4.0, 3.0, 4.0, 6.0]);

    let mut padded_calibrated = [f32::NAN; 14];
    padded_calibrated[2..6].copy_from_slice(&[6.0, 2.0, 4.0, 3.0]);
    padded_calibrated[8..12].copy_from_slice(&[6.0, 5.0, 4.0, 6.0]);
    let mut padded_output = [-1.0_f32; 16];
    let padded = padded_plan();
    padded
        .process(
            RegionPixelsMut::with_stride(&mut padded_output, 4, 2, 8).unwrap(),
            &mut padded.workspace(),
            &padded_calibrated,
        )
        .unwrap();
    assert_eq!(&padded_output[0..4], &[6.0, 2.0, 6.0, 5.0]);
    assert_eq!(&padded_output[4..8], &[-1.0; 4]);
    assert_eq!(&padded_output[8..12], &[4.0, 3.0, 4.0, 6.0]);
    assert_eq!(&padded_output[12..16], &[-1.0; 4]);
}

#[test]
fn extraction_preserves_prepared_region_order() {
    let plan = RegionExtractionPlan::<f32>::new((2, 4), &[(0, 2), (0, 0)], (2, 2)).unwrap();
    let calibrated = [6.0_f32, 2.0, 4.0, 3.0, 6.0, 5.0, 4.0, 6.0];
    let mut output = [0.0_f32; 8];

    assert_eq!(plan.origins(), [(0, 2), (0, 0)]);
    assert_eq!(plan.completion_indices(), [7, 5]);
    plan.process(
        RegionPixelsMut::new(&mut output, 4, 2).unwrap(),
        &mut plan.workspace(),
        &calibrated,
    )
    .unwrap();

    assert_eq!(output, [4.0, 3.0, 4.0, 6.0, 6.0, 2.0, 6.0, 5.0]);
}

#[test]
fn selected_floating_representation_is_preserved() {
    let plan = RegionExtractionPlan::<f64>::new((2, 4), &[(0, 0), (0, 2)], (2, 2)).unwrap();
    let calibrated = [6.0_f64, 2.0, 4.0, 3.0, 6.0, 5.0, 4.0, 6.0];
    let mut output = [0.0_f64; 8];

    plan.process(
        RegionPixelsMut::new(&mut output, 4, 2).unwrap(),
        &mut plan.workspace(),
        &calibrated,
    )
    .unwrap();

    assert_eq!(output, [6.0, 2.0, 6.0, 5.0, 4.0, 3.0, 4.0, 6.0]);
}

#[test]
fn progressive_extraction_reports_immutable_region_prefixes() {
    let plan = padded_plan();
    let mut calibrated = [f32::NAN; 14];
    calibrated[2..6].copy_from_slice(&[6.0, 2.0, 4.0, 3.0]);
    calibrated[8..12].copy_from_slice(&[6.0, 5.0, 4.0, 6.0]);
    let mut output = [-1.0_f32; 8];
    let mut workspace = plan.workspace();

    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.process_range(
            RegionPixelsMut::new(&mut output, 4, 2).unwrap(),
            &mut workspace,
            &calibrated,
            0..1,
        ),
        Ok(ProgressResult::new(1))
    );
    assert_eq!(&output[0..4], &[6.0, 2.0, 6.0, 5.0]);
    assert_eq!(&output[4..8], &[-1.0; 4]);

    assert_eq!(
        plan.process_range(
            RegionPixelsMut::new(&mut output, 4, 2).unwrap(),
            &mut workspace,
            &calibrated,
            1..2,
        ),
        Ok(ProgressResult::new(2))
    );
    plan.finish(
        RegionPixelsMut::new(&mut output, 4, 2).unwrap(),
        &mut workspace,
    )
    .unwrap();
    assert_eq!(output, [6.0, 2.0, 6.0, 5.0, 4.0, 3.0, 4.0, 6.0]);
}

#[test]
fn slab_and_process_boundaries_are_validated_before_mutation() {
    assert_eq!(
        RegionPixelsMut::with_stride(&mut [0.0_f32; 8], 4, 2, 3).unwrap_err(),
        RegionPixelsError::StrideTooSmall {
            pixel_count: 4,
            stride: 3,
        }
    );
    let plan = compact_plan();
    let mut output = [-1.0_f32; 8];
    let mut workspace = plan.workspace();
    assert_eq!(
        plan.process(
            RegionPixelsMut::new(&mut output, 4, 2).unwrap(),
            &mut workspace,
            &[0.0_f32; 7],
        ),
        Err(ProcessError::InputExtent {
            expected: 8,
            actual: 7,
        })
    );
    assert_eq!(output, [-1.0; 8]);

    assert_eq!(
        plan.process_range(
            RegionPixelsMut::new(&mut output, 4, 2).unwrap(),
            &mut workspace,
            &[0.0_f32; 8],
            0..1,
        ),
        Err(ProcessError::NotStarted)
    );
    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.process_range(
            RegionPixelsMut::new(&mut output, 4, 2).unwrap(),
            &mut workspace,
            &[0.0_f32; 8],
            1..2,
        ),
        Err(ProcessError::RangeNotContiguous {
            start: 1,
            expected: 0,
        })
    );
    assert_eq!(output, [-1.0; 8]);
    assert_eq!(
        plan.process_range(
            RegionPixelsMut::new(&mut output, 4, 2).unwrap(),
            &mut workspace,
            &[0.0_f32; 8],
            0..1,
        ),
        Ok(ProgressResult::new(1))
    );
    assert_eq!(
        plan.finish(
            RegionPixelsMut::new(&mut output, 4, 2).unwrap(),
            &mut workspace,
        ),
        Err(ProcessError::Incomplete {
            processed: 1,
            extent: 2,
        })
    );
    plan.abort(&mut workspace).unwrap();
}

#[test]
fn declaration_matches_packed_extraction_and_adopts_origins() {
    let mut declared = RegionExtractionF32::prepare(RegionExtractionF32Configuration {
        origins: vec![[0, 0], [2, 3]],
        image_size: (4, 5),
        region_size: (2, 2),
    })
    .unwrap();
    assert_eq!(RegionExtractionF32::NAME, "region-extraction-f32");
    assert_eq!(declared.formats()[0].shape(), &[4, 5]);
    assert_eq!(declared.formats()[0].schema(), CALIBRATED_PIXELS_V1);
    assert_eq!(declared.formats()[1].shape(), &[2, 2, 2]);
    assert_eq!(declared.formats()[1].schema(), REGION_PIXELS_V1);
    assert_eq!(declared.formats()[2].shape(), &[2, 2]);
    assert_eq!(declared.formats()[2].schema(), REGION_ORIGINS_V1);

    let image = core::array::from_fn::<_, 20, _>(|index| index as f32);
    let mut regions = [-1.0_f32; 8];
    RegionExtractionF32::process(&mut declared, &image, &mut regions).unwrap();
    assert_eq!(regions, [0.0, 1.0, 5.0, 6.0, 13.0, 14.0, 18.0, 19.0]);

    let origins = [0_u32, 1, 2, 2];
    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&origins[..])),
        Some(Ok(()))
    );
    RegionExtractionF32::process(&mut declared, &image, &mut regions).unwrap();
    assert_eq!(regions, [1.0, 2.0, 6.0, 7.0, 12.0, 13.0, 17.0, 18.0]);

    let invalid = [0_u32, 1, 3, 4];
    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&invalid[..])),
        Some(Err(DeclarationParameterError::Algorithm(
            RegionExtractionError::RegionOutOfBounds { region: 1 }
        )))
    );
    RegionExtractionF32::process(&mut declared, &image, &mut regions).unwrap();
    assert_eq!(regions, [1.0, 2.0, 6.0, 7.0, 12.0, 13.0, 17.0, 18.0]);
}

#[test]
fn malformed_detector_geometry_reports_logical_extent_or_physical_layout() {
    use RegionExtractionError::*;
    for (image, regions, size, stride, offset, expected, detail) in [
        (
            (0, 3),
            &[(0, 0)][..],
            (1, 1),
            3,
            0,
            EmptyImage,
            "image dimensions must be positive",
        ),
        (
            (3, 3),
            &[(0, 0)][..],
            (0, 1),
            3,
            0,
            EmptyRegion,
            "region dimensions must be positive",
        ),
        (
            (3, 3),
            &[][..],
            (1, 1),
            3,
            0,
            EmptyOrigins,
            "at least one region",
        ),
        (
            (3, 3),
            &[(0, 0)][..],
            (1, 1),
            2,
            0,
            RowStrideTooSmall,
            "row stride must contain every logical image column",
        ),
        (
            (3, 3),
            &[(0, 0), (2, 2)][..],
            (2, 2),
            8,
            0,
            RegionOutOfBounds { region: 1 },
            "region 1 lies outside",
        ),
        (
            (3, 3),
            &[(0, 0)][..],
            (1, 1),
            8,
            usize::MAX,
            LayoutOverflow,
            "detector layout overflows",
        ),
    ] {
        let error = RegionExtractionPlan::<f32>::with_layout(image, regions, size, stride, offset)
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
    }
}

#[test]
fn packed_region_view_checks_extents_before_borrowing_storage() {
    use RegionPixelsError::*;
    for (pixels, regions, stride, expected, detail) in [
        (0, 1, 1, EmptyRegion, "region must contain pixels"),
        (2, 0, 2, EmptySlab, "at least one region"),
        (
            2,
            2,
            1,
            StrideTooSmall {
                pixel_count: 2,
                stride: 1,
            },
            "stride 1 is smaller than pixel count 2",
        ),
        (2, 2, usize::MAX, LayoutOverflow, "region layout overflows"),
        (
            2,
            2,
            5,
            StorageTooSmall {
                required: 7,
                actual: 4,
            },
            "storage extent 4 is smaller than required extent 7",
        ),
    ] {
        let error =
            RegionPixels::<f32>::with_stride(&[0.0; 4], pixels, regions, stride).unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
    }
}

#[test]
fn rejected_origin_updates_leave_the_original_padded_plan_intact() {
    let plan = padded_plan();
    let error = plan.with_u32_origins(&[0, 0, 0]).unwrap_err();
    assert_eq!(
        error,
        RegionExtractionError::OriginCoordinateExtent {
            expected: 4,
            actual: 3
        }
    );
    assert!(
        error
            .to_string()
            .contains("coordinate extent 3 does not match 4")
    );
    let error = plan.with_u32_origins(&[0, 0, 1, 3]).unwrap_err();
    assert_eq!(
        error,
        RegionExtractionError::RegionOutOfBounds { region: 1 }
    );
    assert!(error.to_string().contains("region 1"));
    let updated = plan.with_u32_origins(&[0, 2, 0, 0]).unwrap();
    assert_eq!(updated.origins(), [(0, 2), (0, 0)]);
    assert_eq!(updated.row_stride(), 4);
    assert_eq!(plan.row_stride(), 6);
    assert_eq!(updated.first_pixel(), 0);
    assert_eq!(plan.first_pixel(), 2);
}
