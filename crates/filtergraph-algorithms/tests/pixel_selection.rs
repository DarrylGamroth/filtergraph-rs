//! Arbitrary pixel-selection ordering, layout, and interface tests.

use filtergraph_algorithms::schemas::{CALIBRATED_PIXELS_V1, PIXEL_SELECTION_MASK_V1};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, DeclarationParameterError, InputProgress,
    OutputProgress, PixelSelectionF32, PixelSelectionF32Configuration,
    PixelSelectionParameterError, PixelSelectionPlan, PixelSelectionPlanError, ProcessError,
    ProgressResult,
};

fn compact_plan<T>() -> PixelSelectionPlan<T> {
    PixelSelectionPlan::new(
        (3, 4),
        [
            false, true, true, false, true, false, false, true, false, true, false, true,
        ],
    )
    .unwrap()
}

fn padded_plan<T>() -> PixelSelectionPlan<T> {
    PixelSelectionPlan::with_layout(
        (3, 4),
        [
            false, true, true, false, true, false, false, true, false, true, false, true,
        ],
        6,
        2,
    )
    .unwrap()
}

fn require_output_progress<P: OutputProgress>(_plan: &P) {}

#[test]
fn preparation_derives_canonical_physical_indices() {
    let compact = compact_plan::<f32>();
    assert_eq!(compact.image_size(), (3, 4));
    assert_eq!(compact.row_stride(), 4);
    assert_eq!(compact.first_pixel(), 0);
    assert_eq!(compact.image_extent(), 12);
    assert_eq!(compact.selected_indices(), [1, 2, 4, 7, 9, 11]);
    assert_eq!(compact.len(), 6);
    assert!(!compact.is_empty());

    let padded = padded_plan::<f32>();
    assert_eq!(padded.image_size(), (3, 4));
    assert_eq!(padded.row_stride(), 6);
    assert_eq!(padded.first_pixel(), 2);
    assert_eq!(padded.image_extent(), 18);
    assert_eq!(padded.selected_indices(), [3, 4, 8, 11, 15, 17]);
    assert_eq!(padded.mask(), compact.mask());
}

#[test]
fn preparation_rejects_empty_malformed_and_unrepresentable_layouts() {
    assert_eq!(
        PixelSelectionPlan::<f32>::new((0, 4), []).unwrap_err(),
        PixelSelectionPlanError::EmptyImage
    );
    assert_eq!(
        PixelSelectionPlan::<f32>::new((2, 2), [true; 3]).unwrap_err(),
        PixelSelectionPlanError::MaskExtent {
            expected: 4,
            actual: 3,
        }
    );
    assert_eq!(
        PixelSelectionPlan::<f32>::new((2, 2), [false; 4]).unwrap_err(),
        PixelSelectionPlanError::NoSelectedPixels
    );
    assert_eq!(
        PixelSelectionPlan::<f32>::with_layout((2, 4), [true; 8], 3, 0).unwrap_err(),
        PixelSelectionPlanError::RowStrideTooSmall {
            columns: 4,
            stride: 3,
        }
    );
    assert_eq!(
        PixelSelectionPlan::<f32>::with_layout((usize::MAX, 2), [], 2, 0).unwrap_err(),
        PixelSelectionPlanError::LayoutOverflow
    );
    assert_eq!(
        PixelSelectionPlan::<f32>::with_layout((1, 1), [true], 1, usize::MAX).unwrap_err(),
        PixelSelectionPlanError::LayoutOverflow
    );
}

#[test]
fn complete_selection_supports_sparse_and_padded_images() {
    let compact = compact_plan::<u16>();
    let compact_image = core::array::from_fn::<_, 12, _>(|index| index as u16 + 100);
    let mut compact_output = [0_u16; 6];
    compact
        .process(
            &mut compact_output,
            &mut compact.workspace(),
            &compact_image,
        )
        .unwrap();
    assert_eq!(compact_output, [101, 102, 104, 107, 109, 111]);

    let padded = padded_plan::<u16>();
    let padded_image = core::array::from_fn::<_, 20, _>(|index| index as u16 + 200);
    let mut padded_output = [0_u16; 6];
    padded
        .process(&mut padded_output, &mut padded.workspace(), &padded_image)
        .unwrap();
    assert_eq!(padded_output, [203, 204, 208, 211, 215, 217]);
}

#[test]
fn half_plane_mask_preserves_logical_row_major_order() {
    let plan = PixelSelectionPlan::<u8>::new(
        (4, 6),
        [
            false, false, false, true, true, true, false, false, false, true, true, true, false,
            false, false, true, true, true, false, false, false, true, true, true,
        ],
    )
    .unwrap();
    let image = core::array::from_fn::<_, 24, _>(|index| index as u8);
    let mut output = [0_u8; 12];

    plan.process(&mut output, &mut plan.workspace(), &image)
        .unwrap();

    assert_eq!(output, [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23]);
}

#[test]
fn selection_copies_values_without_applying_validity_policy() {
    let plan = PixelSelectionPlan::<f32>::new((1, 3), [true, false, true]).unwrap();
    let mut output = [0.0_f32; 2];

    plan.process(
        &mut output,
        &mut plan.workspace(),
        &[f32::NAN, 1.0, f32::INFINITY],
    )
    .unwrap();

    assert!(output[0].is_nan());
    assert_eq!(output[1], f32::INFINITY);
}

#[test]
fn contiguous_spans_and_padded_runs_preserve_float_bits_and_progress_prefixes() {
    const BITS: [u32; 6] = [
        0x7fa0_1234,
        0xffc0_4321,
        0x7f80_0000,
        0xff80_0000,
        0x8000_0000,
        1,
    ];
    const SENTINEL: f32 = -15.0;
    let compact = PixelSelectionPlan::<f32>::with_layout((3, 48), [true; 144], 48, 2).unwrap();
    let mask = (0..144)
        .map(|i| (4..44).contains(&(i % 48)))
        .collect::<Vec<_>>();
    let padded = PixelSelectionPlan::<f32>::with_layout((3, 48), mask, 52, 2).unwrap();
    for (plan, indices) in [
        (compact, (2..146).collect::<Vec<_>>()),
        (padded, (6..46).chain(58..98).chain(110..150).collect()),
    ] {
        let mut image: Vec<_> = (0..160).map(|i| i as f32).collect();
        for (&index, bits) in indices.iter().zip(BITS) {
            image[index] = f32::from_bits(bits);
        }
        let expected: Vec<_> = indices.iter().map(|&i| image[i].to_bits()).collect();
        let mut workspace = plan.workspace();
        let mut output = vec![SENTINEL; plan.len()];
        assert_eq!(plan.selected_indices(), indices);
        plan.process(&mut output, &mut workspace, &image).unwrap();
        assert!(
            output
                .iter()
                .map(|p| p.to_bits())
                .eq(expected.iter().copied())
        );

        output.fill(SENTINEL);
        plan.start(&mut workspace).unwrap();
        for range in [0..5, 5..43, 43..81, 81..plan.len()] {
            let end = range.end;
            assert_eq!(
                plan.process_range(&mut output, &mut workspace, &image, range),
                Ok(ProgressResult::new(end))
            );
            assert!(
                output[..end]
                    .iter()
                    .map(|p| p.to_bits())
                    .eq(expected[..end].iter().copied())
            );
            assert!(output[end..].iter().all(|&p| p == SENTINEL));
        }
        plan.finish(&mut output, &mut workspace).unwrap();
    }
}

#[test]
fn process_boundaries_are_validated_before_output_mutation() {
    let plan = padded_plan::<f32>();
    let mut output = [-1.0_f32; 6];
    let mut workspace = plan.workspace();

    assert_eq!(
        plan.process(&mut output, &mut workspace, &[0.0; 17]),
        Err(ProcessError::InputExtent {
            expected: 18,
            actual: 17,
        })
    );
    assert_eq!(output, [-1.0; 6]);

    assert_eq!(
        plan.process(&mut output[..5], &mut workspace, &[0.0; 18]),
        Err(ProcessError::OutputExtent {
            expected: 6,
            actual: 5,
        })
    );
    assert_eq!(output, [-1.0; 6]);

    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.process(&mut output, &mut workspace, &[0.0; 18]),
        Err(ProcessError::AlreadyStarted)
    );
    assert_eq!(output, [-1.0; 6]);
    plan.abort(&mut workspace).unwrap();
}

#[test]
fn progressive_selection_exposes_immutable_selected_pixel_prefixes() {
    let plan = padded_plan::<u16>();
    require_output_progress(&plan);
    let image = core::array::from_fn::<_, 18, _>(|index| index as u16 + 300);
    let mut output = [0_u16; 6];
    let mut workspace = plan.workspace();

    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &image, 0..2),
        Ok(ProgressResult::new(2))
    );
    assert_eq!(output, [303, 304, 0, 0, 0, 0]);
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &image, 3..4),
        Err(ProcessError::RangeNotContiguous {
            start: 3,
            expected: 2,
        })
    );
    assert_eq!(output, [303, 304, 0, 0, 0, 0]);
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &image, 2..6),
        Ok(ProgressResult::new(6))
    );
    plan.finish(&mut output, &mut workspace).unwrap();

    assert_eq!(output, [303, 304, 308, 311, 315, 317]);
    assert!(!workspace.is_started());
}

#[test]
fn declaration_binds_generic_pixel_meanings_and_a_shape_preserving_mask() {
    const SELECTED_SCHEMA: &str = "org.example.selected-calibrated-pixels/1";
    let configuration = PixelSelectionF32Configuration {
        mask: vec![true, false, true, false, true, false],
        input_schema: CALIBRATED_PIXELS_V1.to_owned(),
        output_schema: SELECTED_SCHEMA.to_owned(),
        image_size: (2, 3),
    };
    let mut declared = PixelSelectionF32::prepare(configuration).unwrap();

    assert_eq!(PixelSelectionF32::NAME, "pixel-selection-f32");
    assert_eq!(declared.formats()[0].shape(), &[2, 3]);
    assert_eq!(declared.formats()[0].schema(), CALIBRATED_PIXELS_V1);
    assert_eq!(declared.formats()[1].shape(), &[3]);
    assert_eq!(declared.formats()[1].schema(), SELECTED_SCHEMA);
    assert_eq!(declared.formats()[2].shape(), &[2, 3]);
    assert_eq!(declared.formats()[2].schema(), PIXEL_SELECTION_MASK_V1);

    let image = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut selected = [-1.0_f32; 3];
    PixelSelectionF32::process(&mut declared, &image, &mut selected).unwrap();
    assert_eq!(selected, [0.0, 2.0, 4.0]);

    let replacement = [false, true, false, true, false, true];
    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&replacement[..])),
        Some(Ok(()))
    );
    PixelSelectionF32::process(&mut declared, &image, &mut selected).unwrap();
    assert_eq!(selected, [1.0, 3.0, 5.0]);

    let shape_changing = [true, false, false, false, false, false];
    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&shape_changing[..])),
        Some(Err(DeclarationParameterError::Algorithm(
            PixelSelectionParameterError::SelectedPixelCount {
                expected: 3,
                actual: 1,
            }
        )))
    );
    PixelSelectionF32::process(&mut declared, &image, &mut selected).unwrap();
    assert_eq!(selected, [1.0, 3.0, 5.0]);
}

#[test]
fn declaration_replaces_the_packing_when_mask_geometry_changes() {
    let indexed = core::array::from_fn::<_, 128, _>(|i| i % 2 == 0);
    let runs = core::array::from_fn::<_, 128, _>(|i| (8..40).contains(&(i % 64)));
    let mut declared = PixelSelectionF32::prepare(PixelSelectionF32Configuration {
        image_size: (2, 64),
        mask: indexed.to_vec(),
        input_schema: CALIBRATED_PIXELS_V1.to_owned(),
        output_schema: "org.example.selected-calibrated-pixels/1".to_owned(),
    })
    .unwrap();
    let image = core::array::from_fn::<_, 128, _>(|i| i as f32);
    let mut selected = [-1.0_f32; 64];
    for mask in [runs, indexed] {
        assert_eq!(
            declared.replace_parameter(2, AlgorithmInput::from(&mask[..])),
            Some(Ok(()))
        );
        PixelSelectionF32::reset(&mut declared);
        PixelSelectionF32::process(&mut declared, &image, &mut selected).unwrap();
        assert!(
            selected.iter().copied().eq(image
                .iter()
                .zip(mask)
                .filter_map(|(&pixel, include)| include.then_some(pixel)))
        );
    }
}
