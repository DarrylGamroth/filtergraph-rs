//! Rolling chopped-image sign, pairing, validity, and state tests.

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmPlan, ChoppedImageDifferenceError, ChoppedImageDifferenceF32,
    ChoppedImageDifferenceF32Configuration, ChoppedImageDifferenceInput,
    ChoppedImageDifferenceOutput, ChoppedImageDifferencePlan, ChoppedImageDifferencePlanError,
    ChoppedImageDifferenceStatus, ChoppedImageKind, SampleMetadata,
};

#[test]
fn alternating_frames_produce_one_fixed_sign_rolling_difference_per_frame() {
    let plan = ChoppedImageDifferencePlan::new(2).unwrap();
    let mut workspace = plan.workspace();
    let mut difference = [9.0_f64; 2];
    let mut validity = [true; 2];
    let mut status = ChoppedImageDifferenceStatus::Complete;

    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(ChoppedImageKind::Fringed, &[10.0, 20.0], &[true; 2]),
    )
    .unwrap();
    assert_eq!(difference, [0.0, 0.0]);
    assert_eq!(validity, [false, false]);
    assert_eq!(status, ChoppedImageDifferenceStatus::Priming);

    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(ChoppedImageKind::Unfringed, &[3.0, 8.0], &[true; 2]),
    )
    .unwrap();
    assert_eq!(difference, [7.0, 12.0]);
    assert_eq!(validity, [true, true]);
    assert_eq!(status, ChoppedImageDifferenceStatus::Complete);

    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(ChoppedImageKind::Fringed, &[12.0, 18.0], &[true; 2]),
    )
    .unwrap();
    assert_eq!(difference, [9.0, 10.0]);

    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(ChoppedImageKind::Unfringed, &[4.0, 7.0], &[true; 2]),
    )
    .unwrap();
    assert_eq!(difference, [8.0, 11.0]);
    assert_eq!(workspace.retained_kind(), Some(ChoppedImageKind::Unfringed));
    assert_eq!(workspace.retained_image(), Some([4.0, 7.0].as_slice()));
}

#[test]
fn unfringed_first_uses_the_same_fringed_minus_unfringed_sign() {
    let plan = ChoppedImageDifferencePlan::new(1).unwrap();
    let mut workspace = plan.workspace();
    let mut difference = [0.0_f32];
    let mut validity = [false];
    let mut status = ChoppedImageDifferenceStatus::Priming;

    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(ChoppedImageKind::Unfringed, &[2.0], &[true]),
    )
    .unwrap();
    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(ChoppedImageKind::Fringed, &[5.0], &[true]),
    )
    .unwrap();

    assert_eq!(difference, [3.0]);
    assert_eq!(validity, [true]);
    assert_eq!(status, ChoppedImageDifferenceStatus::Complete);
}

#[test]
fn repeated_label_reprimes_one_retained_image() {
    let plan = ChoppedImageDifferencePlan::new(1).unwrap();
    let mut workspace = plan.workspace();
    let mut difference = [0.0_f32];
    let mut validity = [false];
    let mut status = ChoppedImageDifferenceStatus::Priming;

    for fringed in [1.0, 2.0] {
        plan.process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
            &mut workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Fringed, &[fringed], &[true]),
        )
        .unwrap();
        assert_eq!(difference, [0.0]);
        assert_eq!(validity, [false]);
        assert_eq!(status, ChoppedImageDifferenceStatus::Priming);
    }
    assert_eq!(workspace.retained_image(), Some([2.0].as_slice()));

    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(ChoppedImageKind::Unfringed, &[0.5], &[true]),
    )
    .unwrap();
    assert_eq!(difference, [1.5]);
    assert_eq!(status, ChoppedImageDifferenceStatus::Complete);
}

#[test]
fn invalid_and_nonfinite_pixels_are_zeroed_without_invalidating_the_pair() {
    let plan = ChoppedImageDifferencePlan::new(4).unwrap();
    let mut workspace = plan.workspace();
    let mut difference = [9.0_f32; 4];
    let mut validity = [true; 4];
    let mut status = ChoppedImageDifferenceStatus::Complete;

    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(
            ChoppedImageKind::Fringed,
            &[2.0, f32::NAN, f32::MAX, 4.0],
            &[true, true, true, false],
        ),
    )
    .unwrap();
    assert_eq!(
        workspace.retained_validity(),
        Some([true, false, true, false].as_slice())
    );

    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(
            ChoppedImageKind::Unfringed,
            &[1.0, 0.0, -f32::MAX, 1.0],
            &[true; 4],
        ),
    )
    .unwrap();

    assert_eq!(difference, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(validity, [true, false, false, false]);
    assert_eq!(status, ChoppedImageDifferenceStatus::Complete);
}

#[test]
fn rolling_pairs_preserve_float32_bits_and_normalize_retained_validity() {
    const LEN: usize = 37;
    let signaling_nan = f32::from_bits(0x7f80_0123);
    let subnormal = f32::from_bits(1);
    // (fringed, unfringed, their two validity flags, difference, pair validity)
    let cases = [
        (8.0, 2.0, true, true, 6.0, true),
        (-0.0, 0.0, true, true, -0.0, true),
        (0.0, -0.0, true, true, 0.0, true),
        (0.0, subnormal, true, true, -subnormal, true),
        (signaling_nan, 0.0, true, true, 0.0, false),
        (3.0, f32::INFINITY, true, true, 0.0, false),
        (f32::MAX, -f32::MAX, true, true, 0.0, false),
        (f32::MAX, -f32::MAX, false, true, 0.0, false),
        (f32::MAX, -f32::MAX, true, false, 0.0, false),
    ];
    let fringed: [_; LEN] = core::array::from_fn(|i| cases[i % cases.len()].0);
    let unfringed: [_; LEN] = core::array::from_fn(|i| cases[i % cases.len()].1);
    let fringed_validity: [_; LEN] = core::array::from_fn(|i| cases[i % cases.len()].2);
    let unfringed_validity: [_; LEN] = core::array::from_fn(|i| cases[i % cases.len()].3);
    let plan = ChoppedImageDifferencePlan::new(LEN).unwrap();
    let mut workspace = plan.workspace();
    let mut difference = [71.0; LEN];
    let mut validity = [true; LEN];
    let mut status = ChoppedImageDifferenceStatus::Complete;

    for (step, (kind, image, input_validity)) in [
        (ChoppedImageKind::Fringed, &fringed, &fringed_validity),
        (ChoppedImageKind::Unfringed, &unfringed, &unfringed_validity),
        (ChoppedImageKind::Fringed, &fringed, &fringed_validity),
    ]
    .into_iter()
    .enumerate()
    {
        plan.process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
            &mut workspace,
            ChoppedImageDifferenceInput::new(kind, image, input_validity),
        )
        .unwrap();
        assert_eq!(workspace.retained_kind(), Some(kind));
        assert_eq!(
            status,
            if step == 0 {
                ChoppedImageDifferenceStatus::Priming
            } else {
                ChoppedImageDifferenceStatus::Complete
            },
        );
        let retained = workspace.retained_image().unwrap();
        let retained_validity = workspace.retained_validity().unwrap();
        for i in 0..LEN {
            let (_, _, _, _, expected, expected_validity) = cases[i % cases.len()];
            let expected = if step == 0 { 0.0 } else { expected };
            assert_eq!(difference[i].to_bits(), expected.to_bits(), "pixel {i}");
            assert_eq!(validity[i], step != 0 && expected_validity, "pixel {i}");
            assert_eq!(retained[i].to_bits(), image[i].to_bits(), "pixel {i}");
            assert_eq!(
                retained_validity[i],
                input_validity[i] && image[i].is_finite()
            );
        }
    }
}

#[cfg(all(feature = "blas", target_arch = "x86_64", target_feature = "avx2"))]
#[test]
fn inactive_float32_lanes_do_not_raise_exceptions_and_scalar_fallback_matches() {
    fn read_mxcsr() -> u32 {
        let mut value = 0;
        // SAFETY: SSE2 is guaranteed on x86_64; the instruction writes a u32.
        unsafe { core::arch::asm!("stmxcsr [{0}]", in(reg) &mut value, options(nostack)) };
        value
    }

    fn write_mxcsr(value: u32) {
        // SAFETY: Each caller updates only defined fields of the current MXCSR.
        unsafe { core::arch::asm!("ldmxcsr [{0}]", in(reg) &value, options(nostack, readonly)) };
    }

    const LEN: usize = 37;
    const EXCEPTION_FLAGS: u32 = 0x3f;
    const EXCEPTION_MASKS: u32 = 0x1f80;
    const OVERFLOW_MASK: u32 = 1 << 10;
    let signaling_nan = f32::from_bits(0x7f80_0123);
    let fringed: [_; LEN] = core::array::from_fn(|i| [5.0, f32::MAX, signaling_nan][i % 3]);
    let unfringed: [_; LEN] = core::array::from_fn(|i| [1.0, -f32::MAX, 0.0][i % 3]);
    let unfringed_validity: [_; LEN] = core::array::from_fn(|i| i % 3 != 1);
    let plan = ChoppedImageDifferencePlan::new(LEN).unwrap();
    let saved = read_mxcsr();
    let masked = (saved | EXCEPTION_MASKS) & !EXCEPTION_FLAGS;
    for flags in [masked, masked & !OVERFLOW_MASK] {
        let mut workspace = plan.workspace();
        let mut difference = [71.0; LEN];
        let mut validity = [true; LEN];
        let mut status = ChoppedImageDifferenceStatus::Complete;
        plan.process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
            &mut workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Fringed, &fringed, &[true; LEN]),
        )
        .unwrap();

        write_mxcsr(flags);
        let result = std::hint::black_box(&plan).process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
            &mut workspace,
            ChoppedImageDifferenceInput::new(
                ChoppedImageKind::Unfringed,
                std::hint::black_box(&unfringed),
                std::hint::black_box(&unfringed_validity),
            ),
        );
        let raised = read_mxcsr() & EXCEPTION_FLAGS;
        write_mxcsr(saved);

        assert_eq!(result, Ok(()));
        assert_eq!(raised, 0);
        assert_eq!(status, ChoppedImageDifferenceStatus::Complete);
        for i in 0..LEN {
            assert_eq!(difference[i], if i % 3 == 0 { 4.0 } else { 0.0 });
            assert_eq!(validity[i], i % 3 == 0);
        }
    }
}

#[test]
fn extent_errors_preserve_outputs_and_retained_history() {
    let plan = ChoppedImageDifferencePlan::new(2).unwrap();
    let mut workspace = plan.workspace();
    let mut difference = [0.0_f32; 2];
    let mut validity = [false; 2];
    let mut status = ChoppedImageDifferenceStatus::Priming;
    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(ChoppedImageKind::Fringed, &[1.0, 2.0], &[true; 2]),
    )
    .unwrap();
    let committed = workspace.clone();

    difference = [7.0; 2];
    validity = [true; 2];
    status = ChoppedImageDifferenceStatus::Complete;
    assert_eq!(
        plan.process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
            &mut workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Unfringed, &[0.0], &[true; 2],),
        ),
        Err(ChoppedImageDifferenceError::ImageExtent {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(workspace, committed);
    assert_eq!(difference, [7.0; 2]);
    assert_eq!(validity, [true; 2]);
    assert_eq!(status, ChoppedImageDifferenceStatus::Complete);

    assert_eq!(
        plan.process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
            &mut workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Unfringed, &[0.0; 2], &[true],),
        ),
        Err(ChoppedImageDifferenceError::InputValidityExtent {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(workspace, committed);

    let mut short_difference = [8.0];
    assert_eq!(
        plan.process(
            ChoppedImageDifferenceOutput::new(&mut short_difference, &mut validity, &mut status,),
            &mut workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Unfringed, &[0.0; 2], &[true; 2],),
        ),
        Err(ChoppedImageDifferenceError::DifferenceExtent {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(short_difference, [8.0]);
    assert_eq!(workspace, committed);

    let mut short_validity = [false];
    assert_eq!(
        plan.process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut short_validity, &mut status,),
            &mut workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Unfringed, &[0.0; 2], &[true; 2],),
        ),
        Err(ChoppedImageDifferenceError::OutputValidityExtent {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(short_validity, [false]);
    assert_eq!(workspace, committed);
}

#[test]
fn incompatible_workspace_is_rejected_before_state_or_output_changes() {
    let plan = ChoppedImageDifferencePlan::new(2).unwrap();
    let other_plan = ChoppedImageDifferencePlan::new(1).unwrap();
    let mut workspace = other_plan.workspace();
    let committed = workspace.clone();
    let mut difference = [7.0_f32; 2];
    let mut validity = [true; 2];
    let mut status = ChoppedImageDifferenceStatus::Complete;

    assert_eq!(
        plan.process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
            &mut workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Fringed, &[1.0; 2], &[true; 2],),
        ),
        Err(ChoppedImageDifferenceError::WorkspaceExtent)
    );
    assert_eq!(workspace, committed);
    assert_eq!(difference, [7.0; 2]);
    assert_eq!(validity, [true; 2]);
    assert_eq!(status, ChoppedImageDifferenceStatus::Complete);
}

#[test]
fn reset_requires_a_new_priming_frame() {
    let plan = ChoppedImageDifferencePlan::new(1).unwrap();
    let mut workspace = plan.workspace();
    let mut difference = [0.0_f32];
    let mut validity = [false];
    let mut status = ChoppedImageDifferenceStatus::Priming;
    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(ChoppedImageKind::Fringed, &[2.0], &[true]),
    )
    .unwrap();
    assert!(workspace.is_primed());

    workspace.reset();
    assert!(!workspace.is_primed());
    assert_eq!(workspace.retained_image(), None);
    plan.process(
        ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
        &mut workspace,
        ChoppedImageDifferenceInput::new(ChoppedImageKind::Unfringed, &[1.0], &[true]),
    )
    .unwrap();
    assert_eq!(difference, [0.0]);
    assert_eq!(validity, [false]);
    assert_eq!(status, ChoppedImageDifferenceStatus::Priming);
}

#[test]
fn preparation_rejects_an_empty_image() {
    assert_eq!(
        ChoppedImageDifferencePlan::<f32>::new(0),
        Err(ChoppedImageDifferencePlanError::EmptyImage)
    );
}

#[test]
fn declaration_preserves_pair_sign_status_and_reset() {
    let mut declared = ChoppedImageDifferenceF32::prepare(ChoppedImageDifferenceF32Configuration {
        input_schema: filtergraph_algorithms::schemas::CALIBRATED_PIXELS_V1.into(),
        require_acquisition_identity: false,
        pixel_count: 2,
        trigger_high_is_fringed: true,
    })
    .unwrap();
    assert_eq!(
        ChoppedImageDifferenceF32::NAME,
        "chopped-image-difference-f32"
    );
    assert_eq!(declared.formats().len(), 5);

    let mut difference = [f32::NAN; 2];
    let mut output_validity = [true; 2];
    let mut status = [u8::MAX];
    ChoppedImageDifferenceF32::process_with_metadata(
        &mut declared,
        &[
            Some(
                SampleMetadata::new(1, 0).with_chopped_image_kind(Some(ChoppedImageKind::Fringed)),
            ),
            None,
        ],
        &[10.0, 20.0],
        &[true, true],
        &mut difference,
        &mut output_validity,
        &mut status,
    )
    .unwrap();
    assert_eq!(difference, [0.0, 0.0]);
    assert_eq!(output_validity, [false, false]);
    assert_eq!(status, [0]);

    ChoppedImageDifferenceF32::process_with_metadata(
        &mut declared,
        &[
            Some(
                SampleMetadata::new(2, 0)
                    .with_chopped_image_kind(Some(ChoppedImageKind::Unfringed)),
            ),
            None,
        ],
        &[3.0, 8.0],
        &[true, true],
        &mut difference,
        &mut output_validity,
        &mut status,
    )
    .unwrap();
    assert_eq!(difference, [7.0, 12.0]);
    assert_eq!(output_validity, [true, true]);
    assert_eq!(status, [1]);

    ChoppedImageDifferenceF32::reset(&mut declared);
    ChoppedImageDifferenceF32::process_with_metadata(
        &mut declared,
        &[
            Some(
                SampleMetadata::new(2, 0)
                    .with_chopped_image_kind(Some(ChoppedImageKind::Unfringed)),
            ),
            None,
        ],
        &[3.0, 8.0],
        &[true, true],
        &mut difference,
        &mut output_validity,
        &mut status,
    )
    .unwrap();
    assert_eq!(difference, [0.0, 0.0]);
    assert_eq!(status, [0]);
}

#[test]
fn explicit_absent_optical_fact_rejects_without_inference_from_electrical_level() {
    use filtergraph_algorithms::SampleTriggerState;
    let mut declared = ChoppedImageDifferenceF32::prepare(ChoppedImageDifferenceF32Configuration {
        pixel_count: 2,
        trigger_high_is_fringed: true,

        input_schema: filtergraph_algorithms::schemas::CALIBRATED_PIXELS_V1.into(),

        require_acquisition_identity: false,
    })
    .unwrap();
    let mut difference = [71.; 2];
    let mut validity = [true; 2];
    let mut status = [99];
    let sample = SampleMetadata::new(1, 0)
        .with_chopped_image_kind(Some(ChoppedImageKind::Absent))
        .with_trigger_state(Some(SampleTriggerState::High));
    assert!(
        ChoppedImageDifferenceF32::process_with_metadata(
            &mut declared,
            &[Some(sample), None],
            &[1.; 2],
            &[true; 2],
            &mut difference,
            &mut validity,
            &mut status
        )
        .is_err()
    );
    assert_eq!(difference, [71.; 2]);
    assert_eq!(validity, [true; 2]);
    assert_eq!(status, [99]);
    assert!(!declared.workspace().is_primed());
}
