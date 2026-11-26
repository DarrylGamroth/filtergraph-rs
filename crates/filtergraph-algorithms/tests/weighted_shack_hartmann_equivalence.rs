//! Source-derived summary and weighted complete-image sensing obligations.
use filtergraph_algorithms::{
    AlgorithmInput, AlgorithmPlan, NoWorkspace, ShackHartmannSummaryF32,
    ShackHartmannSummaryF32Configuration, ShackHartmannSummaryPlan,
    ShackHartmannWeightedCentroidF32, ShackHartmannWeightedCentroidF32Configuration,
};
#[test]
fn summary_uses_float64_sums_uncentered_rms_and_exact_counts() {
    let plan = ShackHartmannSummaryPlan::new(2).unwrap();
    let mut summary = [71.0; 8];
    plan.process(
        &mut summary,
        &mut NoWorkspace,
        (&[3.0, 4.0, 3.0, 4.0], &[1.0, 1.0], &[true, true]),
    )
    .unwrap();
    assert_eq!(summary, [2.0, 2.0, 1.0, 3.0, 4.0, 5.0, 3.0, 4.0]);
    let plan = ShackHartmannSummaryPlan::new(3).unwrap();
    plan.process(
        &mut summary,
        &mut NoWorkspace,
        (
            &[16777216.0, 0.0, 1.0, -2.0, -16777216.0, 3.0],
            &[16777216.0, 1.0, -16777216.0],
            &[true; 3],
        ),
    )
    .unwrap();
    assert_eq!(summary[2], 1.0 / 3.0);
    assert_eq!(summary[6], 1.0 / 3.0);
    assert_eq!(summary[7], 1.0 / 3.0);
    assert!(ShackHartmannSummaryPlan::new(0).is_err());
    assert!(ShackHartmannSummaryPlan::new((1 << 24) + 1).is_err());
    assert!(ShackHartmannSummaryPlan::new(1 << 24).is_ok());
    assert!(
        ShackHartmannSummaryF32::prepare(ShackHartmannSummaryF32Configuration {
            subaperture_count: 1 << 24
        })
        .is_ok()
    );
}
#[test]
fn summary_ignores_invalid_rows_and_publishes_nan_diagnostics_without_rejecting() {
    let plan = ShackHartmannSummaryPlan::new(3).unwrap();
    let mut summary = [71.0; 8];
    let slopes = [3.0, 4.0, f32::NAN, f32::INFINITY, -3.0, -4.0];
    let flux = [10.0, f32::NAN, 30.0];
    plan.process(
        &mut summary,
        &mut NoWorkspace,
        (&slopes, &flux, &[true, false, true]),
    )
    .unwrap();
    assert_eq!(summary, [3.0, 2.0, 20.0, 3.0, 4.0, 5.0, 0.0, 0.0]);
    for validity in [[false; 3], [true; 3]] {
        plan.process(&mut summary, &mut NoWorkspace, (&slopes, &flux, &validity))
            .unwrap();
        assert_eq!(summary[..2], [3.0, if validity[0] { 3.0 } else { 0.0 }]);
        assert!(summary[2..].iter().all(|v| v.is_nan()));
    }
    summary.fill(71.0);
    assert!(
        plan.process(
            &mut summary,
            &mut NoWorkspace,
            (&slopes[..5], &flux, &[true; 3])
        )
        .is_err()
    );
    assert_eq!(summary, [71.0; 8]);
}
fn configuration() -> ShackHartmannWeightedCentroidF32Configuration {
    ShackHartmannWeightedCentroidF32Configuration {
        image_schema: "pipewireao/calibrated-pixels-f32/1".into(),
        subaperture_count: 2,
        initial_subaperture_origins: Some(vec![[0, 0], [1, 2]]),
        coordinate_scale: 1.0,
        pixel_threshold: 1.0,
        flux_threshold: 2.0,
        reference_slopes: Some(vec![0.25, -0.5, -0.25, 0.5]),
        active: Some(vec![true, false]),
        initial_weights: vec![vec![0.0, 2.0], vec![1.0, 3.0]],
        image_size: (2, 4),
        subaperture_size: (1, 2),
        simd_width: 0,
    }
}
fn two_pixel_measurement(
    coordinates: [f32; 2],
    image: [f32; 2],
) -> ([f32; 2], [f32; 1], [bool; 1]) {
    let mut c = configuration();
    c.image_size = (1, 2);
    c.subaperture_count = 1;
    c.initial_subaperture_origins = Some(vec![[0, 0]]);
    c.reference_slopes = Some(vec![0.0; 2]);
    c.active = Some(vec![true]);
    c.initial_weights = vec![vec![1.0; 2]];
    let mut p = ShackHartmannWeightedCentroidF32::prepare(c).unwrap();
    let xy = [
        coordinates[0],
        coordinates[0],
        coordinates[1],
        coordinates[1],
    ];
    p.replace_parameter(5, AlgorithmInput::from(xy.as_slice()))
        .unwrap()
        .unwrap();
    let mut slopes = [71.0; 2];
    let mut flux = [71.0];
    let mut valid = [false];
    ShackHartmannWeightedCentroidF32::process(&mut p, &image, &mut slopes, &mut flux, &mut valid)
        .unwrap();
    (slopes, flux, valid)
}
#[test]
fn weighted_image_moments_use_fma_to_retain_cancellation_residual() {
    let coordinates = [-(1.0 + 1.0 / 2048.0), 1.0 + 1.0 / 4096.0];
    let image = [1.0, 1.0 + 1.0 / 4096.0];
    let expected_flux = 2.0 + 1.0 / 4096.0;
    let expected_slope = (1.0 / 16777216.0) / expected_flux;
    assert_eq!(
        two_pixel_measurement(coordinates, image),
        ([expected_slope; 2], [expected_flux], [true])
    );
}
#[test]
fn weighted_image_moments_use_fma_to_avoid_intermediate_product_overflow() {
    assert_eq!(
        two_pixel_measurement([-f32::MAX, f32::MAX], [1.0, 2.0]),
        ([f32::MAX / 3.0; 2], [3.0], [true])
    );
}
#[test]
fn weighted_threshold_validity_mask_and_zero_weight_nonfinite_follow_source() {
    let mut p = ShackHartmannWeightedCentroidF32::prepare(configuration()).unwrap();
    let image = [0.5, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0];
    let mut slopes = [71.0; 4];
    let mut flux = [71.0; 2];
    let mut valid = [false; 2];
    ShackHartmannWeightedCentroidF32::process(&mut p, &image, &mut slopes, &mut flux, &mut valid)
        .unwrap();
    assert_eq!(flux, [4.0, 10.0]);
    assert_eq!(valid, [true, true]);
    assert_eq!(slopes, [0.25, 0.5, 0.0, 0.0]);
    let mut bad = image;
    bad[0] = f32::NAN;
    ShackHartmannWeightedCentroidF32::process(&mut p, &bad, &mut slopes, &mut flux, &mut valid)
        .unwrap();
    assert!(flux[0].is_nan());
    assert!(!valid[0]);
    assert_eq!(slopes[..2], [0.0, 0.0]);
    ShackHartmannWeightedCentroidF32::process(&mut p, &image, &mut slopes, &mut flux, &mut valid)
        .unwrap();
    assert_eq!(flux, [4.0, 10.0]);
    let thresholds = [1.0_f32, 4.0, 1.0, 10.0];
    p.replace_parameter(7, AlgorithmInput::from(thresholds.as_slice()))
        .unwrap()
        .unwrap();
    ShackHartmannWeightedCentroidF32::process(&mut p, &image, &mut slopes, &mut flux, &mut valid)
        .unwrap();
    // The flux threshold is inclusive; equality remains valid.
    assert_eq!(valid, [true, true]);
    let thresholds = [1.0_f32, 4.25, 1.0, 10.25];
    p.replace_parameter(7, AlgorithmInput::from(thresholds.as_slice()))
        .unwrap()
        .unwrap();
    ShackHartmannWeightedCentroidF32::process(&mut p, &image, &mut slopes, &mut flux, &mut valid)
        .unwrap();
    assert_eq!(valid, [false, false]);
    assert_eq!(slopes, [0.0; 4]);
}
#[test]
fn weighted_replacement_rejections_reset_and_malformed_output_preserve_calibration() {
    let mut p = ShackHartmannWeightedCentroidF32::prepare(configuration()).unwrap();
    for weights in [
        vec![-1.0_f32, 1.0, 1.0, 1.0],
        vec![f32::NAN, 1.0, 1.0, 1.0],
        vec![f32::INFINITY, 1.0, 1.0, 1.0],
        vec![1.0; 3],
    ] {
        assert!(
            p.replace_parameter(9, AlgorithmInput::from(weights.as_slice()))
                .unwrap()
                .is_err()
        );
    }
    let weights = [1.0_f32, 1.0, 2.0, 0.5];
    p.replace_parameter(9, AlgorithmInput::from(weights.as_slice()))
        .unwrap()
        .unwrap();
    let image = [1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0];
    let mut slopes = [71.0; 4];
    let mut flux = [71.0; 2];
    let mut valid = [false; 2];
    ShackHartmannWeightedCentroidF32::process(&mut p, &image, &mut slopes, &mut flux, &mut valid)
        .unwrap();
    let before = (slopes, flux, valid);
    p.reset();
    ShackHartmannWeightedCentroidF32::process(&mut p, &image, &mut slopes, &mut flux, &mut valid)
        .unwrap();
    assert_eq!((slopes, flux, valid), before);
    slopes.fill(71.0);
    flux.fill(71.0);
    valid.fill(false);
    assert!(
        ShackHartmannWeightedCentroidF32::process(
            &mut p,
            &image,
            &mut slopes[..3],
            &mut flux,
            &mut valid
        )
        .is_err()
    );
    assert_eq!(slopes, [71.0; 4]);
    assert_eq!(flux, [71.0; 2]);
    assert_eq!(valid, [false; 2]);
    for weights in [
        vec![],
        vec![vec![1.0]],
        vec![vec![1.0, 2.0], vec![1.0, -1.0]],
    ] {
        let mut c = configuration();
        c.initial_weights = weights;
        assert!(ShackHartmannWeightedCentroidF32::prepare(c).is_err());
    }
}
