//! Weighted calibrated-pixel semantics and declaration replacement boundaries.
use filtergraph_algorithms::{
    AlgorithmInput, AlgorithmPlan, AlgorithmProperties, NoWorkspace, PropertyAssignment,
    PropertyValue, WeightedCenterOfGravityF32 as D, WeightedCenterOfGravityF32Configuration as C,
    WeightedCenterOfGravityPlan as P,
};
fn plan() -> P {
    P::new(
        vec![-1.25, 0.375, 2.75, -0.5],
        vec![2.5, -0.625, 1.125, 4.25],
        vec![0.0, 1.5, 0.25, 2.0],
        [0.125, -0.375],
        None,
        0.0,
    )
    .unwrap()
}
fn measure(plan: &P, pixels: &[f32]) -> ([f32; 2], f32, bool) {
    let mut offset = [71.0; 2];
    let mut flux = 73.0;
    let mut valid = false;
    plan.process(
        (&mut offset, &mut flux, &mut valid),
        &mut NoWorkspace,
        pixels,
    )
    .unwrap();
    (offset, flux, valid)
}
#[test]
fn weighted_moments_use_fma_to_retain_cancellation_residual() {
    let coordinates = vec![-(1.0 + 1.0 / 2048.0), 1.0 + 1.0 / 4096.0];
    let p = P::new(
        coordinates.clone(),
        coordinates,
        vec![1.0; 2],
        [0.0; 2],
        None,
        0.0,
    )
    .unwrap();
    // The exact second product leaves 2^-24 after cancellation. Rounding the
    // product before addition erases that residual.
    let pixels = [1.0, 1.0 + 1.0 / 4096.0];
    let expected_flux = 2.0 + 1.0 / 4096.0;
    let expected_offset = (1.0 / 16777216.0) / expected_flux;
    assert_eq!(
        measure(&p, &pixels),
        ([expected_offset; 2], expected_flux, true)
    );
}
#[test]
fn weighted_moments_use_fma_to_avoid_intermediate_product_overflow() {
    let p = P::new(
        vec![-f32::MAX, f32::MAX],
        vec![-f32::MAX, f32::MAX],
        vec![1.0; 2],
        [0.0; 2],
        None,
        0.0,
    )
    .unwrap();
    // MAX * 2 - MAX is exactly MAX; a separately rounded product overflows.
    assert_eq!(measure(&p, &[1.0, 2.0]), ([f32::MAX / 3.0; 2], 3.0, true));
}
#[test]
fn weighted_pixel_product_is_rounded_before_moment_fma() {
    let a = 1.0 + 1.0 / 4096.0;
    let rounded_product = 1.0 + 1.0 / 2048.0;
    let p = P::new(
        vec![-rounded_product, 1.0],
        vec![-rounded_product, 1.0],
        vec![1.0, a],
        [0.0; 2],
        None,
        0.0,
    )
    .unwrap();
    // Rounding a*a first removes 2^-24, so both moment sums cancel to zero.
    assert_eq!(
        measure(&p, &[1.0, a]),
        ([0.0; 2], 1.0 + rounded_product, true)
    );
}
#[test]
fn weights_are_not_normalized_and_pixel_threshold_precedes_weighting() {
    let p = plan();
    let input = [9.0, 2.0, 4.0, 0.5];
    let (offset, flux, valid) = measure(&p, &input);
    assert!(valid);
    assert_eq!(flux, 5.0);
    assert!((offset[0] - 0.55).abs() < 1e-6);
    assert!((offset[1] - 1.075).abs() < 1e-6);
    let scaled = p.with_weights(&[0.0, 3.0, 0.5, 4.0]).unwrap();
    let (scaled_offset, scaled_flux, scaled_valid) = measure(&scaled, &input);
    assert_eq!(scaled_offset, offset);
    assert_eq!(scaled_flux, 10.0);
    assert!(scaled_valid);
    let thresholded = p
        .prepare_properties(&[
            PropertyAssignment {
                id: 2,
                value: PropertyValue::Bool(true),
            },
            PropertyAssignment {
                id: 3,
                value: PropertyValue::Float(2.0),
            },
            PropertyAssignment {
                id: 4,
                value: PropertyValue::Float(4.0),
            },
        ])
        .unwrap();
    let (_, flux, valid) = measure(&thresholded, &input);
    assert_eq!(flux, 4.0);
    assert!(valid); // Includes raw 2 with weight 1.5; excludes raw 0.5 with weight 2.
    let above = thresholded
        .prepare_properties(&[PropertyAssignment {
            id: 4,
            value: PropertyValue::Float(f32::from_bits(4.0_f32.to_bits() + 1)),
        }])
        .unwrap();
    assert_eq!(measure(&above, &input), ([0.0; 2], 4.0, false));
}
#[test]
fn nonfinite_pixels_invalidate_even_zero_weights_and_outputs_recover() {
    let p = plan();
    let good = [9.0, 2.0, 4.0, 0.5];
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut pixels = good;
        pixels[0] = value;
        let (offset, flux, valid) = measure(&p, &pixels);
        assert_eq!(offset, [0.0; 2]);
        assert!(flux.is_nan());
        assert!(!valid);
        assert!(measure(&p, &good).2);
    }
    let zero = p.with_weights(&[0.0; 4]).unwrap();
    assert_eq!(measure(&zero, &good), ([0.0; 2], 0.0, false));
    assert_eq!(
        measure(&p, &[1.0, -2.0, 1.0, 0.25]),
        ([0.0; 2], -2.25, false)
    );
    let mut offset = [71.0; 2];
    let mut flux = 73.0;
    let mut valid = true;
    assert!(
        p.process(
            (&mut offset, &mut flux, &mut valid),
            &mut NoWorkspace,
            &good[..3]
        )
        .is_err()
    );
    assert_eq!((offset, flux, valid), ([71.0; 2], 73.0, true));
}
#[test]
fn invalid_calibrations_and_properties_leave_original_plan_usable() {
    let p = plan();
    let input = [9.0, 2.0, 4.0, 0.5];
    let before = measure(&p, &input);
    for weight in [-0.25, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(p.with_weights(&[0.0, weight, 0.25, 2.0]).is_err());
    }
    assert!(p.with_weights(&[1.0; 3]).is_err());
    assert!(p.with_pair_interleaved_coordinates(&[1.0; 7]).is_err());
    assert!(p.with_pair_interleaved_coordinates(&[f32::NAN; 8]).is_err());
    for assignment in [
        PropertyAssignment {
            id: 0,
            value: PropertyValue::Float(f32::NAN),
        },
        PropertyAssignment {
            id: 3,
            value: PropertyValue::Float(f32::INFINITY),
        },
        PropertyAssignment {
            id: 4,
            value: PropertyValue::Float(-0.1),
        },
    ] {
        assert!(p.prepare_properties(&[assignment]).is_err());
    }
    assert_eq!(measure(&p, &input), before);
    assert!(P::new(Vec::new(), Vec::new(), Vec::new(), [0.0; 2], None, 0.0).is_err());
    assert!(P::new(vec![0.0], vec![0.0, 1.0], vec![1.0], [0.0; 2], None, 0.0).is_err());
    assert!(P::new(vec![0.0], vec![0.0], vec![1.0, 2.0], [0.0; 2], None, 0.0).is_err());
}
#[test]
fn declaration_parameters_are_copied_and_reset_retains_them() {
    let mut p = D::prepare(C {
        pixel_count: 4,
        initial_coordinates: vec![[-1.25, 2.5], [0.375, -0.625], [2.75, 1.125], [-0.5, 4.25]],
        initial_weights: vec![0.0, 1.5, 0.25, 2.0],
    })
    .unwrap();
    let input = [9.0, 2.0, 4.0, 0.5];
    let mut offset = [71.0; 2];
    let mut flux = [73.0];
    let mut valid = [false];
    let mut weights = [0.0_f32, 3.0, 0.5, 4.0];
    p.replace_parameter(5, AlgorithmInput::from(weights.as_slice()))
        .unwrap()
        .unwrap();
    weights.fill(99.0);
    let coords = [-3.0_f32, 0.5, 1.25, 2.0, -0.75, -2.5, 4.5, 1.375];
    p.replace_parameter(4, AlgorithmInput::from(coords.as_slice()))
        .unwrap()
        .unwrap();
    D::process(&mut p, &input, &mut offset, &mut flux, &mut valid).unwrap();
    let expected = (offset, flux, valid);
    assert_eq!(flux, [10.0]);
    assert!(valid[0]);
    assert!(
        p.replace_parameter(
            5,
            AlgorithmInput::from([0.0_f32, -1.0, 0.5, 4.0].as_slice())
        )
        .unwrap()
        .is_err()
    );
    p.reset();
    D::process(&mut p, &input, &mut offset, &mut flux, &mut valid).unwrap();
    assert_eq!((offset, flux, valid), expected);
}

#[test]
fn declaration_rejects_initial_shape_and_calibration_boundaries() {
    let configuration = || C {
        pixel_count: 2,
        initial_coordinates: vec![[0.5, -1.25], [2.75, 0.375]],
        initial_weights: vec![0.0, 1.5],
    };
    let mut c = configuration();
    c.pixel_count = 0;
    c.initial_coordinates.clear();
    c.initial_weights.clear();
    assert!(D::prepare(c).is_err());
    let mut c = configuration();
    c.pixel_count = 3;
    assert!(D::prepare(c).is_err());
    let mut c = configuration();
    c.initial_weights.pop();
    assert!(D::prepare(c).is_err());
    for coordinate in 0..2 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut c = configuration();
            c.initial_coordinates[1][coordinate] = value;
            assert!(D::prepare(c).is_err());
        }
    }
    for value in [-0.25, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut c = configuration();
        c.initial_weights[1] = value;
        assert!(D::prepare(c).is_err());
    }
    let mut p = D::prepare(configuration()).unwrap();
    let mut offset = [71.0; 2];
    let mut flux = [73.0];
    let mut valid = [true];
    assert!(D::process(&mut p, &[1.0, 2.0], &mut offset[..1], &mut flux, &mut valid).is_err());
    assert!(D::process(&mut p, &[1.0, 2.0], &mut offset, &mut flux[..0], &mut valid).is_err());
    assert!(D::process(&mut p, &[1.0, 2.0], &mut offset, &mut flux, &mut valid[..0]).is_err());
    assert_eq!((offset, flux, valid), ([71.0; 2], [73.0], [true]));
}
