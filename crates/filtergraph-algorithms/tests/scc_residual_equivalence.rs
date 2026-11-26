//! Source-derived SCC selected-pixel dispositions and dense controller reconstruction.
use filtergraph_algorithms::{
    AlgorithmInput, AlgorithmPlan, NoWorkspace, SccPixelResidualError as E,
    SccPixelResidualF32 as D, SccPixelResidualF32Configuration as C, SccPixelResidualPlan as P,
    SccReconstructorF32 as R, SccReconstructorF32Configuration as RC,
};
fn plan() -> P<f32> {
    P::new(
        vec![1.25, -3.5, 0.625, 2.25, -1.125, 0.375, 4.5],
        vec![true, false, true, false, true, false, true],
        0.75,
    )
    .unwrap()
}
#[test]
fn ordered_residuals_zero_invalid_pixels_and_publish_complete_pair_diagnostics_on_rejection() {
    let p = plan();
    let mut output = [71.0; 4];
    let mut valid = [true; 4];
    let mut fraction = -1.0;
    let mut difference = [4.125, -1.75, 3.625, 8.5, -2.875, 0.75, 6.25];
    let mut validity = [true; 7];
    p.process(
        (&mut output, &mut valid, &mut fraction),
        &mut NoWorkspace,
        (&difference, &validity, 1),
    )
    .unwrap();
    assert_eq!(output, [2.875, 3.0, -1.75, 1.75]);
    assert_eq!(fraction, 1.0);
    validity[4] = false;
    difference[6] = f32::NAN;
    assert_eq!(
        p.process(
            (&mut output, &mut valid, &mut fraction),
            &mut NoWorkspace,
            (&difference, &validity, 1)
        ),
        Err(E::InsufficientValidPixels)
    );
    assert_eq!(output, [2.875, 3.0, 0.0, 0.0]);
    assert_eq!(valid, [true, true, false, false]);
    assert_eq!(fraction, 0.5);
    difference[6] = 6.25;
    p.process(
        (&mut output, &mut valid, &mut fraction),
        &mut NoWorkspace,
        (&difference, &validity, 1),
    )
    .unwrap();
    assert_eq!(fraction, 0.75);
    let mut reference = p.reference_image().to_vec();
    reference[0] = -f32::MAX;
    let overflow = p.with_reference_image(&reference).unwrap();
    difference[0] = f32::MAX;
    validity.fill(true);
    overflow
        .process(
            (&mut output, &mut valid, &mut fraction),
            &mut NoWorkspace,
            (&difference, &validity, 1),
        )
        .unwrap();
    assert_eq!(output[0], 0.0);
    assert!(!valid[0]);
    assert_eq!(fraction, 0.75);
}
#[test]
fn incomplete_undefined_status_and_shape_errors_precede_writes() {
    let p = plan();
    let all_selected = P::new(vec![-f32::MAX; 7], vec![true; 7], 0.75).unwrap();
    for plan in [&p, &all_selected] {
        let mut output = vec![71.0; plan.selected_count()];
        let mut validity = vec![true; plan.selected_count()];
        let mut fraction = -1.0;
        for (status, error) in [
            (0, E::IncompletePair),
            (2, E::InvalidPairStatus),
            (255, E::InvalidPairStatus),
        ] {
            assert_eq!(
                plan.process(
                    (&mut output, &mut validity, &mut fraction),
                    &mut NoWorkspace,
                    (&[f32::MAX; 7], &[true; 7], status)
                ),
                Err(error)
            );
        }
        assert_eq!(
            plan.process(
                (&mut output, &mut validity, &mut fraction),
                &mut NoWorkspace,
                (&[f32::MAX; 6], &[true; 7], 0)
            ),
            Err(E::DimensionMismatch)
        );
        assert!(output.iter().all(|&value| value == 71.0));
        assert!(validity.iter().all(|&valid| valid));
        assert_eq!(fraction, -1.0);
    }
    let mut out = [71.0; 4];
    let mut valid = [true; 4];
    let mut prepared = D::prepare(C {
        pixel_count: 7,
        mask: p.mask().to_vec(),
        initial_reference_image: Some(p.reference_image().to_vec()),
        minimum_valid_fraction: 0.75,
    })
    .unwrap();
    assert!(
        D::process(
            &mut prepared,
            &[0.0; 7],
            &[true; 7],
            &[1],
            &mut out,
            &mut valid,
            &mut []
        )
        .is_err()
    );
    assert!(
        D::process(
            &mut prepared,
            &[0.0; 7],
            &[true; 7],
            &[],
            &mut out,
            &mut valid,
            &mut [-1.0]
        )
        .is_err()
    );
    assert_eq!(out, [71.0; 4]);
}
#[test]
fn all_selected_preserves_residual_bits_and_validity_through_the_tail() {
    let snan = f32::from_bits(0x7f80_0001);
    let subnormal = f32::from_bits(1);
    // (sample, reference, input validity, residual, output validity)
    let cases = [
        (-0.0, 0.0, true, -0.0, true),
        (0.0, -0.0, true, 0.0, true),
        (subnormal, 0.0, true, subnormal, true),
        (f32::MAX, -f32::MAX, true, 0.0, false),
        (f32::MAX, -f32::MAX, false, 0.0, false),
        (f32::NEG_INFINITY, 1.0, true, 0.0, false),
        (f32::NAN, 1.0, true, 0.0, false),
        (snan, 1.0, true, 0.0, false),
        (snan, 1.0, false, 0.0, false),
    ];
    const LEN: usize = 263;
    let sample: [_; LEN] = core::array::from_fn(|i| cases[i % cases.len()].0);
    let reference: [_; LEN] = core::array::from_fn(|i| cases[i % cases.len()].1);
    let input_validity: [_; LEN] = core::array::from_fn(|i| cases[i % cases.len()].2);
    let plan = P::new(reference.to_vec(), vec![true; LEN], 0.5).unwrap();
    let mut output = [71.0; LEN];
    let mut validity = [true; LEN];
    let mut fraction = -1.0;
    assert_eq!(
        plan.process(
            (&mut output, &mut validity, &mut fraction),
            &mut NoWorkspace,
            (&sample, &input_validity, 1)
        ),
        Err(E::InsufficientValidPixels)
    );
    let mut count = 0;
    for (i, (&output, &valid)) in output.iter().zip(&validity).enumerate() {
        let (_, _, _, expected, expected_valid) = cases[i % cases.len()];
        assert_eq!(output.to_bits(), expected.to_bits(), "pixel {i}");
        assert_eq!(valid, expected_valid, "pixel {i}");
        count += usize::from(expected_valid);
    }
    assert_eq!(fraction.to_bits(), (count as f32 / LEN as f32).to_bits());
}

// Exercise the packed classifier. Debug builds of the original scalar
// classifier can set FP flags even though they never subtract invalid inputs.
#[cfg(all(feature = "blas", target_arch = "x86_64", target_feature = "avx2"))]
#[test]
fn all_selected_rejected_operands_and_exact_tiny_results_do_not_raise_fp_flags() {
    fn read_mxcsr() -> u32 {
        let mut value = 0_u32;
        // SAFETY: SSE2 is guaranteed on x86_64, and the instruction writes one u32.
        unsafe { core::arch::asm!("stmxcsr [{0}]", in(reg) &mut value, options(nostack)) };
        value
    }

    const LEN: usize = 256;
    const EXCEPTION_FLAGS: u32 = 0x3f;
    const EXCEPTION_MASKS: u32 = 0x1f80;
    const FLUSH_TO_ZERO: u32 = 1 << 15;
    const DENORMALS_ARE_ZERO: u32 = 1 << 6;
    let snan = f32::from_bits(0x7f80_0001);
    let sample: [_; LEN] = core::array::from_fn(|i| match i % 8 {
        0 => f32::MAX,
        1 | 4 => snan,
        2 => f32::from_bits(1),
        3 | 5 => f32::from_bits(f32::MIN_POSITIVE.to_bits() + 1),
        _ => 1.0,
    });
    let reference: [_; LEN] = core::array::from_fn(|i| match i % 8 {
        0 => -f32::MAX,
        2 => 1.0,
        3 | 5 => f32::MIN_POSITIVE,
        _ => 0.0,
    });
    let input_validity: [_; LEN] = core::array::from_fn(|i| i % 8 >= 4);
    let plan = P::new(reference.to_vec(), vec![true; LEN], 0.25).unwrap();
    let mut output = [71.0; LEN];
    let mut validity = [true; LEN];
    let mut fraction = -1.0;

    let saved = read_mxcsr();
    let cleared =
        (saved | EXCEPTION_MASKS) & !(EXCEPTION_FLAGS | FLUSH_TO_ZERO | DENORMALS_ARE_ZERO);
    // SAFETY: This changes only defined MXCSR fields for the current thread.
    unsafe { core::arch::asm!("ldmxcsr [{0}]", in(reg) &cleared, options(nostack, readonly)) };
    let status = std::hint::black_box(&plan).process(
        (&mut output, &mut validity, &mut fraction),
        &mut NoWorkspace,
        (
            std::hint::black_box(&sample),
            std::hint::black_box(&input_validity),
            1,
        ),
    );
    let raised = read_mxcsr() & EXCEPTION_FLAGS;
    // SAFETY: Restore this thread's previously read MXCSR value.
    unsafe { core::arch::asm!("ldmxcsr [{0}]", in(reg) &saved, options(nostack, readonly)) };
    assert_eq!(status, Ok(()));
    assert_eq!(raised, 0);
    assert_eq!(output[5].to_bits(), 1);
    assert_eq!(fraction, 0.375);
}

#[test]
fn fraction_rounding_matches_float64_ceiling_and_invalid_calibration_preserves_plan() {
    let p = P::new(vec![0.0_f32; 5], vec![true; 5], 0.6).unwrap();
    assert_eq!(p.minimum_valid_count(), 4);
    let mut output = [71.0; 5];
    let mut valid = [true; 5];
    let mut fraction = -1.0;
    assert_eq!(
        p.process(
            (&mut output, &mut valid, &mut fraction),
            &mut NoWorkspace,
            (&[1.0; 5], &[true, true, true, false, false], 1)
        ),
        Err(E::InsufficientValidPixels)
    );
    assert_eq!(fraction, 0.6);
    for fraction in [0.0, -0.1, 1.01, f32::NAN, f32::INFINITY] {
        assert!(P::new(vec![0.0_f32; 5], vec![true; 5], fraction).is_err());
    }
    assert!(p.with_mask(&[true, true, true, false, false]).is_err());
    assert!(p.with_mask(&[true; 4]).is_err());
    assert!(p.with_reference_image(&[f32::NAN; 5]).is_err());
    assert_eq!(p.minimum_valid_count(), 4);
}
#[test]
fn declared_parameter_replacement_and_reset_keep_selected_order_and_owned_reference() {
    let p = plan();
    let mut prepared = D::prepare(C {
        pixel_count: 7,
        mask: p.mask().to_vec(),
        initial_reference_image: Some(p.reference_image().to_vec()),
        minimum_valid_fraction: 0.75,
    })
    .unwrap();
    let mut reference = [-0.25_f32, 1.5, 2.125, -0.75, 0.5, -2.25, 3.625];
    prepared
        .replace_parameter(6, AlgorithmInput::from(reference.as_slice()))
        .unwrap()
        .unwrap();
    reference.fill(99.0);
    let mask = [false, true, true, false, false, true, true];
    prepared
        .replace_parameter(7, AlgorithmInput::from(mask.as_slice()))
        .unwrap()
        .unwrap();
    assert!(
        prepared
            .replace_parameter(7, AlgorithmInput::from([true; 7].as_slice()))
            .unwrap()
            .is_err()
    );
    prepared.reset();
    let mut out = [71.0; 4];
    let mut valid = [false; 4];
    let mut fraction = [-1.0];
    D::process(
        &mut prepared,
        &[4.125, -1.75, 3.625, 8.5, -2.875, 0.75, 6.25],
        &[true; 7],
        &[1],
        &mut out,
        &mut valid,
        &mut fraction,
    )
    .unwrap();
    assert_eq!(out, [-3.25, 1.5, 3.0, 2.625]);
    assert_eq!(fraction, [1.0]);
}
#[test]
fn reconstruction_preserves_named_controller_rows_and_rejects_before_writes_for_nonfinite_input() {
    let mut p = R::prepare(RC {
        controller_coordinate_count: 2,
        selected_pixel_count: 3,
        initial_reconstructor: Some(vec![0.5, -1.25, 2.0, -0.75, 1.5, 0.25]),
    })
    .unwrap();
    let mut output = [73.0; 2];
    R::process(&mut p, &[1.25, -2.0, 0.5], &mut output).unwrap();
    assert_eq!(output, [4.125, -3.8125]);
    output.fill(73.0);
    assert!(R::process(&mut p, &[1.25, f32::NAN, 0.5], &mut output).is_err());
    assert_eq!(output, [73.0; 2]);
    let replacement = [2.0_f32, -0.5, 0.25, -1.5, 0.75, 3.0];
    p.replace_parameter(2, AlgorithmInput::from(replacement.as_slice()))
        .unwrap()
        .unwrap();
    assert!(
        p.replace_parameter(2, AlgorithmInput::from([f32::NAN; 6].as_slice()))
            .unwrap()
            .is_err()
    );
    p.reset();
    R::process(&mut p, &[1.25, -2.0, 0.5], &mut output).unwrap();
    assert_eq!(output, [3.625, -1.875]);
    assert!(
        R::prepare(RC {
            controller_coordinate_count: 0,
            selected_pixel_count: 3,
            initial_reconstructor: None
        })
        .is_err()
    );
    assert!(
        R::prepare(RC {
            controller_coordinate_count: u32::MAX as usize / 4 + 1,
            selected_pixel_count: 1,
            initial_reconstructor: None
        })
        .is_err()
    );
}

#[test]
fn invalid_reference_and_mask_updates_leave_the_selected_pixels_usable() {
    use filtergraph_algorithms::SccPixelResidualPlanError as PError;
    let p = P::new(vec![1.0_f32, 3.0, 4.0], vec![true, false, true], 1.0).unwrap();
    for (result, expected, detail) in [
        (
            P::new(vec![], vec![], 1.0),
            PError::EmptyImage,
            "reference image must be nonempty",
        ),
        (
            p.with_mask(&[true]),
            PError::MaskExtent,
            "mask and reference extents",
        ),
        (
            p.with_mask(&[false; 3]),
            PError::EmptyMask,
            "mask must select pixels",
        ),
        (
            p.with_mask(&[true; 3]),
            PError::SelectedCount,
            "selected-pixel count",
        ),
        (
            p.with_reference_image(&[f32::NAN, 3.0, 4.0]),
            PError::NonFiniteReference,
            "reference image must be finite",
        ),
        (
            P::new(vec![0.0_f32], vec![true], 0.0),
            PError::InvalidMinimumValidFraction,
            "minimum valid fraction",
        ),
    ] {
        let error = result.unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
    }
    let mut output = [71.0; 2];
    let mut valid = [false; 2];
    let mut fraction = -1.0;
    p.process(
        (&mut output, &mut valid, &mut fraction),
        &mut NoWorkspace,
        (&[3.0, 1.0, 9.0], &[true; 3], 1),
    )
    .unwrap();
    assert_eq!((output, valid, fraction), ([2.0, 5.0], [true; 2], 1.0));
}
