//! Shack-Hartmann centroid, validity, layout, and progressive tests.

use filtergraph_algorithms::{
    AlgorithmPlan, InputProgress, ProcessError, ProgressResult, RegionExtractionPlan, RegionPixels,
    RegionPixelsMut, ShackHartmannError, ShackHartmannImagePlan, ShackHartmannImagePlanError,
    ShackHartmannOutput, ShackHartmannOutputError, ShackHartmannPlan,
};

fn one_subaperture_plan(
    flux_threshold: f32,
    active: bool,
    reference: [f32; 2],
) -> ShackHartmannPlan<f32> {
    ShackHartmannPlan::new(
        vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
        vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        reference.to_vec(),
        vec![0.0],
        vec![flux_threshold],
        vec![active],
    )
    .unwrap()
}

#[test]
fn preparation_fixes_shape_and_finite_values() {
    assert_eq!(
        ShackHartmannPlan::<f32>::new(
            vec![],
            vec![],
            vec![0.0; 2],
            vec![0.0],
            vec![0.0],
            vec![true],
        )
        .unwrap_err(),
        ShackHartmannError::EmptyCoordinates
    );
    assert_eq!(
        ShackHartmannPlan::new(
            vec![0.0_f32],
            vec![0.0, 1.0],
            vec![0.0; 2],
            vec![0.0],
            vec![0.0],
            vec![true],
        )
        .unwrap_err(),
        ShackHartmannError::CoordinateExtent { x: 1, y: 2 }
    );
    assert_eq!(
        ShackHartmannPlan::new(
            vec![0.0_f32],
            vec![0.0],
            vec![0.0],
            vec![0.0],
            vec![0.0],
            vec![true],
        )
        .unwrap_err(),
        ShackHartmannError::ReferenceSlopeExtent {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        ShackHartmannPlan::new(
            vec![0.0_f32],
            vec![0.0],
            vec![0.0; 2],
            vec![-1.0],
            vec![0.0],
            vec![true],
        )
        .unwrap_err(),
        ShackHartmannError::InvalidPixelThreshold { subaperture: 0 }
    );
    assert_eq!(
        ShackHartmannPlan::new(
            vec![0.0_f32],
            vec![0.0],
            vec![0.0; 2],
            vec![0.0],
            vec![-1.0],
            vec![true],
        )
        .unwrap_err(),
        ShackHartmannError::InvalidFluxThreshold { subaperture: 0 }
    );
}

#[test]
fn output_view_requires_pair_interleaved_slopes() {
    let mut slopes = [0.0_f32; 3];
    let mut flux = [0.0_f32; 1];
    let mut validity = [false; 1];
    assert_eq!(
        ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap_err(),
        ShackHartmannOutputError::SlopeExtent {
            expected: 2,
            actual: 3,
        }
    );
}

#[test]
fn centroid_uses_detector_coordinates_and_reference() {
    let plan = one_subaperture_plan(4.0, true, [0.25, -0.5]);
    let mut slab = [f32::NAN; 16];
    slab[0..9].fill(0.0);
    slab[8] = 4.0;
    let input = RegionPixels::with_stride(&slab, 9, 1, 16).unwrap();
    let mut slopes = [0.0_f32; 2];
    let mut flux = [0.0_f32; 1];
    let mut validity = [false; 1];

    plan.process(
        ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
        &mut plan.workspace(),
        input,
    )
    .unwrap();

    assert_eq!(flux, [4.0]);
    assert_eq!(slopes, [0.75, 1.5]);
    assert_eq!(validity, [true]);
    assert!(slab[9..16].iter().all(|value| value.is_nan()));
}

#[test]
fn selected_floating_representation_is_preserved() {
    let plan = ShackHartmannPlan::new(
        vec![-1.0_f64, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
        vec![-1.0_f64, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.25_f64, -0.5],
        vec![0.0_f64],
        vec![4.0_f64],
        vec![true],
    )
    .unwrap();
    let mut pixels = [0.0_f64; 9];
    pixels[8] = 4.0;
    let mut slopes = [0.0_f64; 2];
    let mut flux = [0.0_f64; 1];
    let mut validity = [false; 1];

    plan.process(
        ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
        &mut plan.workspace(),
        RegionPixels::new(&pixels, 9, 1).unwrap(),
    )
    .unwrap();

    assert_eq!(slopes, [0.75, 1.5]);
    assert_eq!(flux, [4.0]);
    assert_eq!(validity, [true]);
}

#[test]
fn image_backed_non_power_of_two_regions_match_materialized_pixels() {
    let extraction =
        RegionExtractionPlan::with_layout((3, 7), &[(0, 1), (1, 3)], (2, 3), 10, 2).unwrap();
    let mut image = [f32::NAN; 29];
    image[2..9].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    image[12..19].copy_from_slice(&[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]);
    image[22..29].copy_from_slice(&[15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0]);

    let plan = ShackHartmannPlan::new(
        vec![-1.0_f32, 0.0, 1.0, -1.0, 0.0, 1.0],
        vec![-0.5_f32, -0.5, -0.5, 0.5, 0.5, 0.5],
        vec![0.25_f32, -0.5, -0.75, 0.25],
        vec![0.0_f32; 2],
        vec![1.0_f32; 2],
        vec![true; 2],
    )
    .unwrap();

    let mut materialized = [0.0_f32; 12];
    extraction
        .process(
            RegionPixelsMut::new(&mut materialized, 6, 2).unwrap(),
            &mut extraction.workspace(),
            &image,
        )
        .unwrap();
    assert_eq!(
        materialized,
        [
            2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 11.0, 12.0, 13.0, 18.0, 19.0, 20.0
        ]
    );

    let mut materialized_slopes = [0.0_f32; 4];
    let mut materialized_flux = [0.0_f32; 2];
    let mut materialized_validity = [false; 2];
    plan.process(
        ShackHartmannOutput::new(
            &mut materialized_slopes,
            &mut materialized_flux,
            &mut materialized_validity,
        )
        .unwrap(),
        &mut plan.workspace(),
        RegionPixels::new(&materialized, 6, 2).unwrap(),
    )
    .unwrap();

    let image_regions = extraction.image_regions(&image).unwrap();
    assert_eq!(image_regions.region_size(), (2, 3));
    assert_eq!(image_regions.pixel_count(), 6);
    let mut viewed_slopes = [0.0_f32; 4];
    let mut viewed_flux = [0.0_f32; 2];
    let mut viewed_validity = [false; 2];
    plan.process_image_regions(
        ShackHartmannOutput::new(&mut viewed_slopes, &mut viewed_flux, &mut viewed_validity)
            .unwrap(),
        &mut plan.workspace(),
        image_regions,
    )
    .unwrap();

    let composite = ShackHartmannImagePlan::new(extraction.clone(), plan.clone()).unwrap();
    let mut composite_slopes = [0.0_f32; 4];
    let mut composite_flux = [0.0_f32; 2];
    let mut composite_validity = [false; 2];
    composite
        .process(
            ShackHartmannOutput::new(
                &mut composite_slopes,
                &mut composite_flux,
                &mut composite_validity,
            )
            .unwrap(),
            &mut composite.workspace(),
            &image,
        )
        .unwrap();

    assert_eq!(viewed_slopes, materialized_slopes);
    assert_eq!(viewed_flux, materialized_flux);
    assert_eq!(viewed_validity, materialized_validity);
    assert_eq!(composite_slopes, materialized_slopes);
    assert_eq!(composite_flux, materialized_flux);
    assert_eq!(composite_validity, materialized_validity);
}

#[test]
fn adjacent_image_regions_and_partition_match_ordered_centroids() {
    let origins = [
        (2, 1),
        (0, 1),
        (0, 3),
        (0, 5),
        (0, 7),
        (0, 9),
        (0, 11),
        (0, 13),
        (0, 15),
    ];
    let subnormal = f32::from_bits(1);
    let samples = [
        [1.0, 3.0, 0.0, 2.0],
        [1.0, 1.0, -1.0, -1.0],
        [f32::MAX, 0.0, -1.0, -1.0],
        [f32::NAN, 0.0, 1.0, 0.0],
        [f32::NEG_INFINITY, -1.0, 1.0, 0.0],
        [subnormal, -0.0, 0.0, -1.0],
        [f32::MAX, f32::MAX, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
    ];
    let mut image = [f32::NAN; 82];
    for ((row, column), pixels) in origins.into_iter().zip(samples) {
        let start = 2 + row * 21 + column;
        image[start..start + 2].copy_from_slice(&pixels[..2]);
        image[start + 21..start + 23].copy_from_slice(&pixels[2..]);
    }
    let measurement = ShackHartmannPlan::new(
        vec![2.0, -3.0, 0.0, 0.5],
        vec![-1.0, 0.0, 1.0, 2.0],
        vec![0.25; 18],
        vec![0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0, 0.0],
        vec![1.0, 2.0, 0.0, 1.0, 1.0, subnormal, 0.0, 0.0, 5.0],
        vec![true, false, true, true, true, true, true, true, true],
    )
    .unwrap();
    let extraction = RegionExtractionPlan::with_layout((4, 17), &origins, (2, 2), 21, 2).unwrap();
    let plan = ShackHartmannImagePlan::new(extraction, measurement).unwrap();
    let mut expected_slopes = [0.0; 18];
    let mut expected_flux = [0.0; 9];
    let mut expected_validity = [false; 9];
    plan.measurement()
        .process(
            ShackHartmannOutput::new(
                &mut expected_slopes,
                &mut expected_flux,
                &mut expected_validity,
            )
            .unwrap(),
            &mut plan.workspace(),
            RegionPixels::new(samples.as_flattened(), 4, 9).unwrap(),
        )
        .unwrap();

    let mut slopes = [71.0; 18];
    let mut flux = [71.0; 9];
    let mut validity = [false; 9];
    plan.process(
        ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
        &mut plan.workspace(),
        &image,
    )
    .unwrap();
    assert_eq!(slopes.map(f32::to_bits), expected_slopes.map(f32::to_bits));
    assert_eq!(validity, expected_validity);
    for (value, expected) in flux.into_iter().zip(expected_flux) {
        assert!(value.to_bits() == expected.to_bits() || (value.is_nan() && expected.is_nan()));
    }

    let mut slopes = [71.0; 14];
    let mut flux = [71.0; 7];
    let mut validity = [false; 7];
    plan.process_image_partition(&mut slopes, &mut flux, &mut validity, &image, 2)
        .unwrap();
    assert_eq!(
        slopes.map(f32::to_bits),
        expected_slopes.map(f32::to_bits)[4..]
    );
    assert_eq!(validity, expected_validity[2..]);
    for (value, expected) in flux.into_iter().zip(&expected_flux[2..]) {
        assert!(value.to_bits() == expected.to_bits() || (value.is_nan() && expected.is_nan()));
    }
}

#[test]
fn image_backed_regions_validate_storage_and_shape_before_output_mutation() {
    let extraction = RegionExtractionPlan::with_layout((2, 3), &[(0, 0)], (2, 3), 4, 1).unwrap();
    let image = [0.0_f32; 8];
    assert_eq!(
        extraction.image_regions(&image[..7]).unwrap_err(),
        ProcessError::InputExtent {
            expected: 8,
            actual: 7,
        }
    );

    let incompatible_plan = ShackHartmannPlan::new(
        vec![0.0_f32; 5],
        vec![0.0_f32; 5],
        vec![0.0_f32; 2],
        vec![0.0_f32],
        vec![0.0_f32],
        vec![true],
    )
    .unwrap();
    assert_eq!(
        ShackHartmannImagePlan::new(extraction.clone(), incompatible_plan.clone()).unwrap_err(),
        ShackHartmannImagePlanError::SubaperturePixelExtent {
            region_pixels: 6,
            measurement_pixels: 5,
        }
    );
    let mut slopes = [-1.0_f32; 2];
    let mut flux = [-1.0_f32; 1];
    let mut validity = [false; 1];
    assert_eq!(
        incompatible_plan.process_image_regions(
            ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
            &mut incompatible_plan.workspace(),
            extraction.image_regions(&image).unwrap(),
        ),
        Err(ProcessError::InputExtent {
            expected: 5,
            actual: 6,
        })
    );
    assert_eq!(slopes, [-1.0; 2]);
    assert_eq!(flux, [-1.0]);
}

#[test]
fn pixel_threshold_is_inclusive_and_precedes_flux_and_moments() {
    let plan = ShackHartmannPlan::new(
        vec![-1.0_f32, 0.0, 1.0],
        vec![0.0_f32; 3],
        vec![0.0_f32; 2],
        vec![2.0_f32],
        vec![5.0_f32],
        vec![true],
    )
    .unwrap();
    let pixels = [1.0_f32, 2.0, 3.0];
    let mut slopes = [f32::NAN; 2];
    let mut flux = [f32::NAN; 1];
    let mut validity = [false; 1];

    plan.process(
        ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
        &mut plan.workspace(),
        RegionPixels::new(&pixels, 3, 1).unwrap(),
    )
    .unwrap();

    assert_eq!(flux, [5.0]);
    assert_eq!(slopes, [0.6, 0.0]);
    assert_eq!(validity, [true]);
}

#[test]
fn pixel_thresholds_are_selected_per_subaperture() {
    let plan = ShackHartmannPlan::new(
        vec![-1.0_f32, 1.0],
        vec![0.0_f32; 2],
        vec![0.0_f32; 4],
        vec![2.0_f32, 4.0],
        vec![0.0_f32; 2],
        vec![true; 2],
    )
    .unwrap();
    let pixels = [1.0_f32, 3.0, 3.0, 4.0];
    let mut slopes = [f32::NAN; 4];
    let mut flux = [f32::NAN; 2];
    let mut validity = [false; 2];

    plan.process(
        ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
        &mut plan.workspace(),
        RegionPixels::new(&pixels, 2, 2).unwrap(),
    )
    .unwrap();

    assert_eq!(flux, [3.0, 4.0]);
    assert_eq!(slopes, [1.0, 0.0, 1.0, 0.0]);
    assert_eq!(validity, [true, true]);
}

#[test]
fn measurement_validity_and_active_state_remain_distinct() {
    let mut pixels = [0.0_f32; 9];
    pixels[8] = 4.0;

    let mut slopes = [f32::NAN; 2];
    let mut flux = [f32::NAN; 1];
    let mut validity = [false; 1];
    let inactive_plan = one_subaperture_plan(4.0, false, [0.0, 0.0]);
    inactive_plan
        .process(
            ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
            &mut inactive_plan.workspace(),
            RegionPixels::new(&pixels, 9, 1).unwrap(),
        )
        .unwrap();
    assert_eq!(flux, [4.0]);
    assert_eq!(validity, [true]);
    assert_eq!(slopes, [0.0, 0.0]);

    let low_flux_plan = one_subaperture_plan(5.0, true, [0.0, 0.0]);
    low_flux_plan
        .process(
            ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
            &mut low_flux_plan.workspace(),
            RegionPixels::new(&pixels, 9, 1).unwrap(),
        )
        .unwrap();
    assert_eq!(flux, [4.0]);
    assert_eq!(validity, [false]);
    assert_eq!(slopes, [0.0, 0.0]);

    pixels.fill(0.0);
    pixels[0] = f32::NAN;
    let nonfinite_plan = one_subaperture_plan(0.0, true, [0.0, 0.0]);
    nonfinite_plan
        .process(
            ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
            &mut nonfinite_plan.workspace(),
            RegionPixels::new(&pixels, 9, 1).unwrap(),
        )
        .unwrap();
    assert!(flux[0].is_nan());
    assert_eq!(validity, [false]);
    assert_eq!(slopes, [0.0, 0.0]);
}

#[test]
fn progressive_processing_publishes_subaperture_prefixes() {
    let plan = ShackHartmannPlan::new(
        vec![-1.0_f32, 1.0],
        vec![0.0_f32, 0.0],
        vec![0.0_f32; 4],
        vec![0.0_f32; 2],
        vec![1.0_f32; 2],
        vec![true; 2],
    )
    .unwrap();
    let pixels = [0.0_f32, 4.0, 4.0, 0.0];
    let mut slopes = [f32::NAN; 4];
    let mut flux = [f32::NAN; 2];
    let mut validity = [false; 2];
    let mut workspace = plan.workspace();

    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.process_range(
            ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
            &mut workspace,
            RegionPixels::new(&pixels, 2, 2).unwrap(),
            0..1,
        ),
        Ok(ProgressResult::new(1))
    );
    assert_eq!(&slopes[0..2], &[1.0, 0.0]);
    assert!(slopes[2..4].iter().all(|value| value.is_nan()));
    assert_eq!(flux[0], 4.0);
    assert!(flux[1].is_nan());

    assert_eq!(
        plan.process_range(
            ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
            &mut workspace,
            RegionPixels::new(&pixels, 2, 2).unwrap(),
            1..2,
        ),
        Ok(ProgressResult::new(2))
    );
    assert_eq!(slopes, [1.0, 0.0, -1.0, 0.0]);
    assert_eq!(flux, [4.0, 4.0]);
    assert_eq!(validity, [true, true]);
    plan.finish(
        ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
        &mut workspace,
    )
    .unwrap();
}

#[test]
fn process_rejects_incompatible_input_before_mutation() {
    let plan = one_subaperture_plan(0.0, true, [0.0, 0.0]);
    let pixels = [0.0_f32; 8];
    let mut slopes = [-1.0_f32; 2];
    let mut flux = [-1.0_f32; 1];
    let mut validity = [false; 1];
    let mut workspace = plan.workspace();

    assert_eq!(
        plan.process(
            ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
            &mut workspace,
            RegionPixels::new(&pixels, 8, 1).unwrap(),
        ),
        Err(ProcessError::InputExtent {
            expected: 9,
            actual: 8,
        })
    );
    assert_eq!(slopes, [-1.0; 2]);
    assert_eq!(flux, [-1.0]);

    let pixels = [0.0_f32; 9];
    assert_eq!(
        plan.process_range(
            ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
            &mut workspace,
            RegionPixels::new(&pixels, 9, 1).unwrap(),
            0..1,
        ),
        Err(ProcessError::NotStarted)
    );
    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.finish(
            ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
            &mut workspace,
        ),
        Err(ProcessError::Incomplete {
            processed: 0,
            extent: 1,
        })
    );
    plan.abort(&mut workspace).unwrap();
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", target_feature = "avx2"))]
#[test]
fn below_threshold_pixels_do_not_raise_overflow() {
    // FE_OVERFLOW on Linux x86-64. Only the sticky status flag is cleared.
    const FE_OVERFLOW: i32 = 8;
    unsafe extern "C" {
        fn feclearexcept(flags: i32) -> i32;
        fn fetestexcept(flags: i32) -> i32;
    }
    let measurement = ShackHartmannPlan::new(
        vec![2.0_f32, 0.5],
        vec![2.0, 0.5],
        vec![0.0; 8],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![true; 4],
    )
    .unwrap();
    let extraction =
        RegionExtractionPlan::new((1, 8), &[(0, 0), (0, 2), (0, 4), (0, 6)], (1, 2)).unwrap();
    let plan = ShackHartmannImagePlan::new(extraction, measurement).unwrap();
    let (mut slopes, mut flux, mut validity) = ([0.0; 8], [0.0; 4], [false; 4]);
    let output = ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap();
    let mut workspace = plan.workspace();
    let image = std::hint::black_box([
        -f32::MAX,
        1.0,
        -f32::MAX,
        1.0,
        -f32::MAX,
        1.0,
        -f32::MAX,
        1.0,
    ]);
    // SAFETY: these C math functions take a valid flag mask and do not change
    // the rounding mode, exception masks, or denormal controls.
    unsafe {
        assert_eq!(feclearexcept(FE_OVERFLOW), 0);
    }
    plan.process(output, &mut workspace, &image).unwrap();
    let overflow = unsafe { fetestexcept(FE_OVERFLOW) };
    assert_eq!(flux, [1.0; 4]);
    assert_eq!(slopes, [0.5; 8]);
    assert_eq!(validity, [true; 4]);
    assert_eq!(overflow, 0);
}
