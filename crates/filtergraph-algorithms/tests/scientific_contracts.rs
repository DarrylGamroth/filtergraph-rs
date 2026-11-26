//! New JFG scientific snapshot and acquisition contracts.
use filtergraph_algorithms::*;

fn physical_configuration(sign: f32) -> PhysicalOffsetReferenceF32Configuration {
    PhysicalOffsetReferenceF32Configuration {
        residual_count: 3,
        physical_count: 3,
        mode_count: 2,
        reference_sign: sign,
        residual_schema: "org.example.residual/1".into(),
        physical_schema: "org.example.physical/1".into(),
        physical_to_modes_schema: "org.example.q/1".into(),
        modes_to_reference_schema: "org.example.c/1".into(),
    }
}
#[test]
fn physical_offset_snapshot_has_explicit_sign_factor_order_and_atomic_publication() {
    for sign in [-1.0, 1.0] {
        let mut p = PhysicalOffsetReferenceF32::prepare(physical_configuration(sign)).unwrap();
        assert_eq!(p.formats()[3].shape(), &[3]);
        p.replace_parameter(
            4,
            AlgorithmInput::from([1.0_f32, 2., -1., 0.5, -2., 3.].as_slice()),
        )
        .unwrap()
        .unwrap();
        p.replace_parameter(
            5,
            AlgorithmInput::from([1.0_f32, -0.5, 2., 1., -1., 3.].as_slice()),
        )
        .unwrap()
        .unwrap();
        p.replace_parameter(3, AlgorithmInput::from([1.0_f32, -2., 0.5].as_slice()))
            .unwrap()
            .unwrap();
        let mut out = [71.; 3];
        let mut physical = [73.; 3];
        PhysicalOffsetReferenceF32::process(&mut p, &[2., -3., 0.25], &mut out, &mut physical)
            .unwrap();
        // Q o = [-3.5, 6], C(Q o) = [-6.5, -1, 21.5].
        assert_eq!(out, [2. - 6.5 * sign, -3. - sign, 0.25 + 21.5 * sign]);
        assert_eq!(physical, [1., -2., 0.5]);
        let plan = p.plan().clone();
        assert!(
            p.replace_parameter(3, AlgorithmInput::from([f32::NAN; 3].as_slice()))
                .unwrap()
                .is_err()
        );
        assert_eq!(p.plan(), &plan);
        PhysicalOffsetReferenceF32::reset(&mut p);
        assert_eq!(p.plan(), &plan);
        out.fill(71.);
        physical.fill(73.);
        assert!(matches!(
            PhysicalOffsetReferenceF32::process(
                &mut p,
                &[f32::NAN, 0., 0.],
                &mut out,
                &mut physical
            ),
            Err(DeclarationProcessError::Algorithm(
                VectorLinearCombinationError::NonFiniteInput
            ))
        ));
        assert_eq!(out, [71.; 3]);
        assert_eq!(physical, [73.; 3]);
        let replacement = p.plan().with_offset(&[-0.5, 1., 2.]).unwrap();
        assert_eq!(replacement.offset(), &[-0.5, 1., 2.]);
        assert_eq!(p.plan().offset(), &[1., -2., 0.5]);
    }
    assert!(PhysicalOffsetReferenceF32::prepare(physical_configuration(0.)).is_err());
    let mut c = physical_configuration(1.);
    c.mode_count = 0;
    assert!(PhysicalOffsetReferenceF32::prepare(c).is_err());
}
#[test]
fn physical_offset_rejects_cold_overflow_and_keeps_both_outputs_on_hot_overflow() {
    let mut p = PhysicalOffsetReferenceF32::prepare(physical_configuration(1.)).unwrap();
    p.replace_parameter(3, AlgorithmInput::from([1.0_f32; 3].as_slice()))
        .unwrap()
        .unwrap();
    assert!(
        p.replace_parameter(4, AlgorithmInput::from([f32::MAX; 6].as_slice()))
            .unwrap()
            .is_err()
    );
    p.replace_parameter(4, AlgorithmInput::from([1.0_f32; 6].as_slice()))
        .unwrap()
        .unwrap();
    assert!(
        p.replace_parameter(5, AlgorithmInput::from([f32::MAX; 6].as_slice()))
            .unwrap()
            .is_err()
    );
    p.replace_parameter(5, AlgorithmInput::from([f32::MAX / 24.; 6].as_slice()))
        .unwrap()
        .unwrap();
    let mut out = [71.; 3];
    let mut physical = [73.; 3];
    assert!(matches!(
        PhysicalOffsetReferenceF32::process(&mut p, &[f32::MAX, 0., 0.], &mut out, &mut physical),
        Err(DeclarationProcessError::Algorithm(
            VectorLinearCombinationError::NonFiniteOutput
        ))
    ));
    assert_eq!(out, [71.; 3]);
    assert_eq!(physical, [73.; 3]);
}
#[test]
fn pixel_validity_only_checks_finiteness_and_preserves_custom_units() {
    let mut p = PixelValidityF32::prepare(PixelValidityF32Configuration {
        extent: 6,
        input_schema: "org.example.adu/1".into(),
    })
    .unwrap();
    assert_eq!(p.formats()[0].schema(), "org.example.adu/1");
    let mut out = [true; 6];
    PixelValidityF32::process(
        &mut p,
        &[0., -0., -3., f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        &mut out,
    )
    .unwrap();
    assert_eq!(out, [true, true, true, false, false, false]);
    PixelValidityF32::reset(&mut p);
    PixelValidityF32::process(&mut p, &[-1.; 6], &mut out).unwrap();
    assert_eq!(out, [true; 6]);
}
#[test]
fn chopped_configuration_defaults_retain_legacy_schema_and_identity_policy() {
    let configuration: ChoppedImageDifferenceF32Configuration =
        toml::from_str("pixel_count=2\ntrigger_high_is_fringed=true").unwrap();
    assert_eq!(configuration.input_schema, schemas::CALIBRATED_PIXELS_V1);
    assert!(!configuration.require_acquisition_identity);
}
fn sample(seq: u64, kind: ChoppedImageKind) -> SampleMetadata {
    let mut domain = [0; 16];
    domain[0] = 1;
    SampleMetadata::new(seq, 0)
        .with_acquisition_identity(Some(
            SampleAcquisitionIdentity::new(domain, 1, seq).unwrap(),
        ))
        .with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(20))
        .with_chopped_image_kind(Some(kind))
}
#[test]
fn strict_chopped_gaps_mixed_corruption_and_sequence_rejection_preserve_acquisition_contract() {
    let mut p = ChoppedImageDifferenceF32::prepare(ChoppedImageDifferenceF32Configuration {
        pixel_count: 2,
        trigger_high_is_fringed: true,
        input_schema: "org.example.raw/1".into(),
        require_acquisition_identity: true,
    })
    .unwrap();
    let mut out = [71.; 2];
    let mut valid = [true; 2];
    let mut status = [99];
    for (metadata, expected) in [
        (sample(1, ChoppedImageKind::Fringed), 0),
        (sample(2, ChoppedImageKind::Unfringed), 1),
        (sample(4, ChoppedImageKind::Fringed), 0),
        (sample(5, ChoppedImageKind::Unfringed), 1),
        (sample(6, ChoppedImageKind::Mixed), 0),
        (sample(7, ChoppedImageKind::Fringed), 0),
        (sample(8, ChoppedImageKind::Unfringed), 1),
        (sample(9, ChoppedImageKind::Fringed).with_corrupted(true), 0),
        (sample(10, ChoppedImageKind::Unfringed), 0),
        (sample(11, ChoppedImageKind::Fringed), 1),
    ] {
        ChoppedImageDifferenceF32::process_with_metadata(
            &mut p,
            &[Some(metadata), None],
            &[metadata.sequence() as f32, 20.],
            &[true; 2],
            &mut out,
            &mut valid,
            &mut status,
        )
        .unwrap();
        assert_eq!(status, [expected]);
        assert_eq!(valid, [expected == 1; 2]);
    }
    let before = p.workspace().clone();
    out.fill(71.);
    valid.fill(true);
    status = [99];
    for seq in [11, 10] {
        assert!(matches!(
            ChoppedImageDifferenceF32::process_with_metadata(
                &mut p,
                &[Some(sample(seq, ChoppedImageKind::Unfringed)), None],
                &[1., 2.],
                &[true; 2],
                &mut out,
                &mut valid,
                &mut status
            ),
            Err(NativeProcessError::Process(
                DeclarationProcessError::Algorithm(ChoppedImageDifferenceError::SequenceInvalid)
            ))
        ));
        assert_eq!(p.workspace(), &before);
        assert_eq!(out, [71.; 2]);
        assert_eq!(valid, [true; 2]);
        assert_eq!(status, [99]);
    }
    ChoppedImageDifferenceF32::process_with_metadata(
        &mut p,
        &[Some(sample(12, ChoppedImageKind::Unfringed)), None],
        &[3., 4.],
        &[true; 2],
        &mut out,
        &mut valid,
        &mut status,
    )
    .unwrap();
    assert_eq!(out, [8., 16.]);
    assert_eq!(status, [1]);
    ChoppedImageDifferenceF32::reset(&mut p);
    ChoppedImageDifferenceF32::process_with_metadata(
        &mut p,
        &[Some(sample(1, ChoppedImageKind::Fringed)), None],
        &[3., 4.],
        &[true; 2],
        &mut out,
        &mut valid,
        &mut status,
    )
    .unwrap();
    assert_eq!(status, [0]);
}
