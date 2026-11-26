//! HEART-compatible pyramid pupil-pixel normalization tests.

use filtergraph_algorithms::{
    AlgorithmPlan, PYRAMID_PUPIL_COUNT, PyramidPixelError, PyramidPixelImagePlan,
    PyramidPixelImagePlanError, PyramidPixelOutput, PyramidPixelPlan, PyramidPixelPlanError,
    RegionExtractionPlan, RegionPixels, RegionPixelsMut,
};

fn assert_near(actual: f32, expected: f32) {
    assert!((actual - expected).abs() <= 2.0 * f32::EPSILON * expected.abs().max(1.0));
}

#[test]
fn preparation_requires_a_nonempty_selected_mask() {
    assert_eq!(
        PyramidPixelPlan::<f32>::new([]).unwrap_err(),
        PyramidPixelPlanError::EmptyMask
    );
    assert_eq!(
        PyramidPixelPlan::<f32>::new([false, false]).unwrap_err(),
        PyramidPixelPlanError::NoSelectedPixels
    );

    let plan = PyramidPixelPlan::<f32>::new([true, false, true]).unwrap();
    assert_eq!(plan.mask(), &[true, false, true]);
    assert_eq!(plan.pupil_pixel_count(), 3);
    assert_eq!(plan.selected_pixel_count(), 2);
    assert_eq!(plan.output_pixel_count(), 2 * PYRAMID_PUPIL_COUNT);
    assert_eq!(plan.shared_mask().as_ref(), plan.mask());
}

#[test]
fn first_frame_measures_mean_and_second_frame_uses_its_reciprocal() {
    let plan = PyramidPixelPlan::<f32>::new([true, false, true]).unwrap();
    let mut workspace = plan.workspace();
    let first = [
        1.0, 2.0, 3.0, // pupil 0
        4.0, 5.0, 6.0, // pupil 1
        7.0, 8.0, 9.0, // pupil 2
        10.0, 11.0, 12.0, // pupil 3
    ];
    let mut pixels = [f32::NAN; 8];
    let mut mean = f32::NAN;

    plan.process(
        PyramidPixelOutput::new(&mut pixels, &mut mean),
        &mut workspace,
        RegionPixels::new(&first, 3, 4).unwrap(),
    )
    .unwrap();

    assert_eq!(pixels, [0.0; 8]);
    assert_near(mean, 6.5);
    assert_near(workspace.reciprocal_previous_frame_mean(), 2.0 / 13.0);

    let second = [
        2.0, 20.0, 4.0, // pupil 0
        6.0, 40.0, 8.0, // pupil 1
        10.0, 60.0, 12.0, // pupil 2
        14.0, 80.0, 16.0, // pupil 3
    ];
    let output = PyramidPixelOutput::new(&mut pixels, &mut mean);
    plan.process(
        output,
        &mut workspace,
        RegionPixels::new(&second, 3, 4).unwrap(),
    )
    .unwrap();

    for (actual, selected) in pixels
        .iter()
        .zip([2.0_f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    {
        assert_near(*actual, selected * (2.0 / 13.0));
    }
    // The masked values 20, 40, 60, and 80 remain part of the full-pupil
    // mean even though they are absent from the reconstruction vector.
    assert_near(mean, 272.0 / 12.0);
    assert_near(workspace.reciprocal_previous_frame_mean(), 12.0 / 272.0);
}

#[test]
fn padded_pupil_storage_preserves_pupil_major_mask_order() {
    let plan = PyramidPixelPlan::<f64>::new([false, true, true]).unwrap();
    let mut workspace = plan.workspace();
    let first = [1.0_f64; 15];
    let mut pixels = [0.0; 8];
    let mut mean = 0.0;
    plan.process(
        PyramidPixelOutput::new(&mut pixels, &mut mean),
        &mut workspace,
        RegionPixels::with_stride(&first, 3, 4, 4).unwrap(),
    )
    .unwrap();

    let second = [
        1.0, 2.0, 3.0, 99.0, // pupil 0 and padding
        4.0, 5.0, 6.0, 99.0, // pupil 1 and padding
        7.0, 8.0, 9.0, 99.0, // pupil 2 and padding
        10.0, 11.0, 12.0, // pupil 3
    ];
    plan.process(
        PyramidPixelOutput::new(&mut pixels, &mut mean),
        &mut workspace,
        RegionPixels::with_stride(&second, 3, 4, 4).unwrap(),
    )
    .unwrap();

    assert_eq!(pixels, [2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0]);
    assert_eq!(mean, 6.5);
}

#[test]
fn nonpositive_mean_commits_zero_gain_for_the_following_frame() {
    let plan = PyramidPixelPlan::<f32>::new([true]).unwrap();
    let mut workspace = plan.workspace();
    let mut pixels = [0.0; 4];
    let mut mean = 0.0;
    plan.process(
        PyramidPixelOutput::new(&mut pixels, &mut mean),
        &mut workspace,
        RegionPixels::new(&[2.0; 4], 1, 4).unwrap(),
    )
    .unwrap();
    plan.process(
        PyramidPixelOutput::new(&mut pixels, &mut mean),
        &mut workspace,
        RegionPixels::new(&[-1.0; 4], 1, 4).unwrap(),
    )
    .unwrap();

    assert_eq!(pixels, [-0.5; 4]);
    assert_eq!(mean, -1.0);
    assert_eq!(workspace.reciprocal_previous_frame_mean(), 0.0);

    plan.process(
        PyramidPixelOutput::new(&mut pixels, &mut mean),
        &mut workspace,
        RegionPixels::new(&[3.0; 4], 1, 4).unwrap(),
    )
    .unwrap();
    assert_eq!(pixels, [0.0; 4]);
}

#[test]
fn rejected_frames_preserve_committed_normalization_history() {
    let plan = PyramidPixelPlan::<f32>::new([true, false]).unwrap();
    let mut workspace = plan.workspace();
    let mut pixels = [0.0; 4];
    let mut mean = 0.0;
    plan.process(
        PyramidPixelOutput::new(&mut pixels, &mut mean),
        &mut workspace,
        RegionPixels::new(&[2.0; 8], 2, 4).unwrap(),
    )
    .unwrap();
    let committed = workspace.reciprocal_previous_frame_mean();

    let error = plan
        .process(
            PyramidPixelOutput::new(&mut pixels, &mut mean),
            &mut workspace,
            RegionPixels::new(&[1.0; 4], 1, 4).unwrap(),
        )
        .unwrap_err();
    assert_eq!(
        error,
        PyramidPixelError::InputPixelExtent {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(workspace.reciprocal_previous_frame_mean(), committed);

    let nonfinite = [1.0, 1.0, 1.0, f32::NAN, 1.0, 1.0, 1.0, 1.0];
    let error = plan
        .process(
            PyramidPixelOutput::new(&mut pixels, &mut mean),
            &mut workspace,
            RegionPixels::new(&nonfinite, 2, 4).unwrap(),
        )
        .unwrap_err();
    assert_eq!(
        error,
        PyramidPixelError::NonFiniteInput { pupil: 1, pixel: 1 }
    );
    assert_eq!(workspace.reciprocal_previous_frame_mean(), committed);
}

#[test]
fn nonfinite_normalized_output_does_not_commit_the_current_mean() {
    let plan = PyramidPixelPlan::<f32>::new([true, false]).unwrap();
    let mut workspace = plan.workspace();
    let mut pixels = [0.0; 4];
    let mut mean = 0.0;
    plan.process(
        PyramidPixelOutput::new(&mut pixels, &mut mean),
        &mut workspace,
        RegionPixels::new(&[f32::MIN_POSITIVE; 8], 2, 4).unwrap(),
    )
    .unwrap();
    let committed = workspace.reciprocal_previous_frame_mean();

    let values = [
        f32::MAX,
        -f32::MAX,
        f32::MAX,
        -f32::MAX,
        f32::MAX,
        -f32::MAX,
        f32::MAX,
        -f32::MAX,
    ];
    let error = plan
        .process(
            PyramidPixelOutput::new(&mut pixels, &mut mean),
            &mut workspace,
            RegionPixels::new(&values, 2, 4).unwrap(),
        )
        .unwrap_err();
    assert_eq!(error, PyramidPixelError::NonFiniteOutput { index: 0 });
    assert_eq!(workspace.reciprocal_previous_frame_mean(), committed);
}

#[test]
fn image_backed_pupils_match_materialized_normalization_across_frames() {
    let extraction =
        RegionExtractionPlan::with_layout((4, 8), &[(0, 1), (0, 5), (2, 1), (2, 5)], (2, 2), 10, 1)
            .unwrap();
    let mut image = [f32::NAN; 39];
    for row in 0..4 {
        for column in 0..8 {
            image[1 + row * 10 + column] = 1.0 + (row * 8 + column) as f32;
        }
    }
    let mut materialized = [0.0_f32; 16];
    extraction
        .process(
            RegionPixelsMut::new(&mut materialized, 4, PYRAMID_PUPIL_COUNT).unwrap(),
            &mut extraction.workspace(),
            &image,
        )
        .unwrap();

    let plan = PyramidPixelPlan::new([true, false, true, true]).unwrap();
    let mut materialized_workspace = plan.workspace();
    let mut viewed_workspace = plan.workspace();
    let composite = PyramidPixelImagePlan::new(extraction.clone(), plan.clone()).unwrap();
    let mut composite_workspace = composite.workspace();
    let mut materialized_pixels = [f32::NAN; 12];
    let mut viewed_pixels = [f32::NAN; 12];
    let mut composite_pixels = [f32::NAN; 12];
    let mut materialized_mean = f32::NAN;
    let mut viewed_mean = f32::NAN;
    let mut composite_mean = f32::NAN;

    for _ in 0..2 {
        plan.process(
            PyramidPixelOutput::new(&mut materialized_pixels, &mut materialized_mean),
            &mut materialized_workspace,
            RegionPixels::new(&materialized, 4, PYRAMID_PUPIL_COUNT).unwrap(),
        )
        .unwrap();
        plan.process_image_regions(
            PyramidPixelOutput::new(&mut viewed_pixels, &mut viewed_mean),
            &mut viewed_workspace,
            extraction.image_regions(&image).unwrap(),
        )
        .unwrap();
        composite
            .process(
                PyramidPixelOutput::new(&mut composite_pixels, &mut composite_mean),
                &mut composite_workspace,
                &image,
            )
            .unwrap();

        assert_eq!(viewed_pixels, materialized_pixels);
        assert_eq!(viewed_mean, materialized_mean);
        assert_eq!(viewed_workspace, materialized_workspace);
        assert_eq!(composite_pixels, materialized_pixels);
        assert_eq!(composite_mean, materialized_mean);
        assert_eq!(composite_workspace, materialized_workspace);
    }
}

#[test]
fn image_backed_pupils_reject_incompatible_geometry_before_mutation() {
    let extraction =
        RegionExtractionPlan::new((2, 4), &[(0, 0), (0, 1), (0, 2), (0, 3)], (2, 1)).unwrap();
    let plan = PyramidPixelPlan::new([true]).unwrap();
    assert_eq!(
        PyramidPixelImagePlan::new(extraction.clone(), plan.clone()).unwrap_err(),
        PyramidPixelImagePlanError::PupilPixelExtent {
            region_pixels: 2,
            mask_pixels: 1,
        }
    );
    let mut pixels = [-1.0_f32; 4];
    let mut mean = -1.0;

    assert_eq!(
        plan.process_image_regions(
            PyramidPixelOutput::new(&mut pixels, &mut mean),
            &mut plan.workspace(),
            extraction.image_regions(&[1.0_f32; 8]).unwrap(),
        ),
        Err(PyramidPixelError::InputPixelExtent {
            expected: 1,
            actual: 2,
        })
    );
    assert_eq!(pixels, [-1.0; 4]);
    assert_eq!(mean, -1.0);
}

#[test]
fn pupil_and_output_shape_errors_leave_normalization_uncommitted() {
    let plan = PyramidPixelPlan::<f32>::new([true, false]).unwrap();
    let mut workspace = plan.workspace();
    let mut pixels = [57.0; 4];
    let mut mean = 57.0;
    for (pupils, extent, npixels, values, expected, detail) in [
        (
            3,
            2,
            4,
            &[2.0; 6][..],
            PyramidPixelError::InputPupilExtent { actual: 3 },
            "3",
        ),
        (
            4,
            2,
            3,
            &[2.0; 8][..],
            PyramidPixelError::OutputExtent {
                expected: 4,
                actual: 3,
            },
            "extent 3",
        ),
        (
            4,
            2,
            4,
            &[f32::MAX; 8][..],
            PyramidPixelError::NonFiniteMean,
            "mean",
        ),
    ] {
        let error = plan
            .process(
                PyramidPixelOutput::new(&mut pixels[..npixels], &mut mean),
                &mut workspace,
                RegionPixels::new(values, extent, pupils).unwrap(),
            )
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
        assert_eq!((pixels, mean), ([57.0; 4], 57.0));
        assert_eq!(workspace.reciprocal_previous_frame_mean(), 0.0);
    }
    plan.process(
        PyramidPixelOutput::new(&mut pixels, &mut mean),
        &mut workspace,
        RegionPixels::new(&[4.0; 8], 2, 4).unwrap(),
    )
    .unwrap();
    assert_eq!((pixels, mean), ([0.0; 4], 4.0));
}
