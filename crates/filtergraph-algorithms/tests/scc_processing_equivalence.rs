//! Spatial FIR/complex ordering, bounded weighted PCG, and transactional SCC exposure updates.
use filtergraph_algorithms::{
    AlgorithmPlan, ChoppedImageKind, PropertyAssignment, PropertyValue, SampleAcquisitionIdentity,
    SampleMetadata, ScalarOrVector, SccCorrectionF32 as Correction,
    SccCorrectionF32Configuration as CorrectionConfiguration,
    SccSpatialCarrierDemodulationError as SE, SccSpatialCarrierDemodulationPlan as Spatial,
    SccWeightedRegularizedReconstructionError as PE,
    SccWeightedRegularizedReconstructionPlan as Pcg,
};
fn spatial(taps: &[f32], minimum: f32) -> Spatial {
    Spatial::new(
        (2, 3),
        &[true; 6],
        taps.len(),
        None,
        None,
        None,
        Some(taps),
        minimum,
        1.0,
    )
    .unwrap()
}
#[test]
fn spatial_separable_fir_uses_literal_column_major_detector_geometry() {
    let p = spatial(&[1.0, 2.0, 1.0], 0.9);
    let mut w = p.workspace();
    let mut values = [71.0; 12];
    let mut weights = [73.0; 12];
    let mut valid = [false; 12];
    let mut fraction = -1.0;
    p.process(
        (&mut values, &mut weights, &mut valid, &mut fraction),
        &mut w,
        (&[1.0, 2.0, 4.0, 8.0, 16.0, 32.0], &[true; 6], 1),
    )
    .unwrap();
    assert_eq!(
        &values[..6],
        &[2.1875, 3.0625, 7.8125, 10.9375, 16.25, 22.75]
    );
    assert_eq!(&values[6..], &[0.0; 6]);
    assert_eq!(weights, [1.0; 12]);
    assert!(valid.iter().all(|&x| x));
    assert_eq!(fraction, 1.0);
}
#[test]
fn spatial_rejection_preserves_all_outputs_and_replacement_reset_retains_calibration() {
    let p = spatial(&[0.25, 0.5, 0.25], 1.0);
    let mut w = p.workspace();
    let mut values = [71.0; 12];
    let mut weights = [73.0; 12];
    let mut valid = [true; 12];
    let mut fraction = -1.0;
    for (status, expected) in [
        (0, SE::IncompletePair),
        (255, SE::InvalidPairStatus),
        (1, SE::InsufficientSupport),
    ] {
        let input_valid = [false, true, true, true, true, true];
        assert_eq!(
            p.process(
                (&mut values, &mut weights, &mut valid, &mut fraction),
                &mut w,
                (&[1.0; 6], &input_valid, status)
            ),
            Err(expected)
        );
        assert_eq!(
            (values, weights, valid, fraction),
            ([71.0; 12], [73.0; 12], [true; 12], -1.0)
        );
    }
    assert!(p.with_carrier_phase(&[f32::NAN; 6]).is_err());
    assert!(p.with_fir(&[1.0, -2.0, 1.0]).is_err());
    assert!(
        p.with_roi(&[true, false, false, false, false, false])
            .is_err()
    );
    let phase = p
        .with_carrier_phase(&[0.0, 0.5, 1.0, -0.75, 0.25, -1.5])
        .unwrap();
    phase
        .process(
            (&mut values, &mut weights, &mut valid, &mut fraction),
            &mut w,
            (&[1.0; 6], &[true; 6], 1),
        )
        .unwrap();
    let before = values;
    w.reset();
    phase
        .process(
            (&mut values, &mut weights, &mut valid, &mut fraction),
            &mut w,
            (&[1.0; 6], &[true; 6], 1),
        )
        .unwrap();
    assert_eq!(values, before);
    let rounding = Spatial::new((1, 5), &[true; 5], 1, None, None, None, None, 1.0, 0.6).unwrap();
    assert_eq!(rounding.minimum_valid_count(), 3);
}
fn pcg(iterations: usize) -> Pcg {
    Pcg::new(
        2,
        2,
        2,
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        0.25,
        iterations,
        1e-5,
        1e-8,
    )
    .unwrap()
}
#[test]
fn pcg_rhs_normalization_handles_large_small_zero_and_zero_weight_nonfinite_measurements() {
    let p = pcg(8);
    let mut w = p.workspace();
    let mut output = [71.0; 2];
    for scale in [1.0, 1e-30, 1e30] {
        p.process(
            &mut output,
            &mut w,
            (&[2.0 * scale, -3.0 * scale], &[1.0, 2.0]),
        )
        .unwrap();
        assert!((output[0] / scale - 1.6).abs() < 2e-6);
        assert!((output[1] / scale + 8.0 / 3.0).abs() < 2e-6);
    }
    p.process(
        &mut output,
        &mut w,
        (&[f32::NAN, f32::INFINITY], &[0.0, 0.0]),
    )
    .unwrap();
    assert_eq!(output, [0.0; 2]);
    w.reset();
    p.process(&mut output, &mut w, (&[0.0; 2], &[1.0; 2]))
        .unwrap();
    assert_eq!(output, [0.0; 2]);
}
#[test]
fn pcg_failure_categories_follow_measurement_order_and_preserve_output() {
    let p = pcg(8);
    let mut w = p.workspace();
    let mut output = [71.0; 2];
    for (y, weights, expected) in [
        ([f32::NAN, 1.0], [1.0, -1.0], PE::InvalidMeasurement),
        ([1.0, f32::NAN], [-1.0, 1.0], PE::InvalidWeight),
        ([1.0, 2.0], [f32::INFINITY, 1.0], PE::InvalidWeight),
    ] {
        assert_eq!(
            p.process(&mut output, &mut w, (&y, &weights)),
            Err(expected)
        );
        assert_eq!(output, [71.0; 2]);
    }
    let singular = Pcg::new(2, 2, 2, vec![0.0; 4], vec![0.0; 4], 0.0, 8, 1e-5, 1e-8).unwrap();
    assert_eq!(
        singular.process(
            &mut output,
            &mut singular.workspace(),
            (&[1.0; 2], &[1.0; 2])
        ),
        Err(PE::SingularSystem)
    );
    assert_eq!(output, [71.0; 2]);
    let bounded = Pcg::new(
        2,
        2,
        2,
        vec![1.0, 1.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        0.1,
        1,
        1e-12,
        1e-8,
    )
    .unwrap();
    assert_eq!(
        bounded.process(
            &mut output,
            &mut bounded.workspace(),
            (&[1.0, 2.0], &[1.0; 2])
        ),
        Err(PE::NotConverged)
    );
    assert_eq!(output, [71.0; 2]);
    p.process(&mut output, &mut w, (&[2.0, -3.0], &[1.0, 2.0]))
        .unwrap();
    assert!(output.iter().all(|x| x.is_finite()));
}
fn correction() -> CorrectionConfiguration {
    CorrectionConfiguration {
        frame_size: (2, 2),
        controller_coordinate_count: 2,
        image_schema: "org.example.scc.frame/1".into(),
        exposure_duration_nanoseconds: 1000,
        coordinate_order: vec![1, 2],
        command_unit: "controller-unit".into(),
        reconstructor_sign: "command_error".into(),
        reference_difference: vec![0.0; 4],
        correction_mask: vec![true, false, false, true],
        reconstructor: vec![1.0, 0.0, 0.0, 2.0],
        fringed_response: vec![0.0; 4],
        unfringed_response: vec![0.0; 4],
        bias_command: vec![0.01; 2],
        command_limits: (ScalarOrVector::Scalar(-1.0), ScalarOrVector::Scalar(1.0)),
        gain: 0.5,
        pole: 0.8,
    }
}
fn sample(sequence: u64, kind: ChoppedImageKind) -> SampleMetadata {
    let mut domain = [0; 16];
    domain[0] = 1;
    SampleMetadata::new(sequence, 0)
        .with_acquisition_identity(Some(
            SampleAcquisitionIdentity::new(domain, 1, sequence).unwrap(),
        ))
        .with_chopped_image_kind(Some(kind))
        .with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(1000))
}
#[test]
fn correction_holds_priming_repeated_gap_mixed_and_resets_to_bias() {
    let mut p = Correction::prepare(correction()).unwrap();
    let mut command = [71.0; 2];
    let fringe = [0.1, 9.0, 8.0, -0.1];
    Correction::process_with_metadata(
        &mut p,
        &[Some(sample(1, ChoppedImageKind::Fringed))],
        &fringe,
        &mut command,
    )
    .unwrap();
    assert_eq!(command, [0.01; 2]);
    Correction::process_with_metadata(
        &mut p,
        &[Some(sample(2, ChoppedImageKind::Unfringed))],
        &[0.0; 4],
        &mut command,
    )
    .unwrap();
    let updated = command;
    assert!((command[0] + 0.04).abs() < 1e-7);
    assert!((command[1] - 0.11).abs() < 1e-7);
    for (seq, kind) in [
        (3, ChoppedImageKind::Unfringed),
        (5, ChoppedImageKind::Fringed),
        (6, ChoppedImageKind::Mixed),
        (7, ChoppedImageKind::Unfringed),
    ] {
        Correction::process_with_metadata(
            &mut p,
            &[Some(sample(seq, kind))],
            &fringe,
            &mut command,
        )
        .unwrap();
        assert_eq!(command, updated);
    }
    p.reset();
    assert_eq!(p.workspace().command(), [0.01; 2]);
    assert_eq!(p.workspace().integrator_state(), [0.0; 2]);
    assert!(!p.workspace().is_primed());
}
#[test]
fn correction_rejections_preserve_nested_history_and_recover_like_no_rejected_exposure() {
    let mut p = Correction::prepare(correction()).unwrap();
    let mut control = Correction::prepare(correction()).unwrap();
    let mut command = [71.0; 2];
    let mut expected = [71.0; 2];
    let fringe = [0.1, 9.0, 8.0, -0.1];
    for prepared in [&mut p, &mut control] {
        Correction::process_with_metadata(
            prepared,
            &[Some(sample(1, ChoppedImageKind::Fringed))],
            &fringe,
            &mut expected,
        )
        .unwrap();
    }
    let before = p.workspace().command().to_vec();
    let history = p.workspace().integrator_state().to_vec();
    let identity = p.workspace().acquisition_identity();
    for (metadata, frame) in [
        (None, fringe),
        (Some(sample(1, ChoppedImageKind::Unfringed)), fringe),
        (
            Some(sample(2, ChoppedImageKind::Unfringed)),
            [f32::NAN, 0.0, 0.0, 0.0],
        ),
        (
            Some(sample(2, ChoppedImageKind::Unfringed)),
            [-100.0, 0.0, 0.0, 100.0],
        ),
    ] {
        command.fill(71.0);
        assert!(
            Correction::process_with_metadata(&mut p, &[metadata], &frame, &mut command).is_err()
        );
        assert_eq!(command, [71.0; 2]);
        assert_eq!(p.workspace().command(), before);
        assert_eq!(p.workspace().integrator_state(), history);
        assert_eq!(p.workspace().acquisition_identity(), identity);
    }
    Correction::process_with_metadata(
        &mut p,
        &[Some(sample(2, ChoppedImageKind::Unfringed))],
        &[0.0; 4],
        &mut command,
    )
    .unwrap();
    Correction::process_with_metadata(
        &mut control,
        &[Some(sample(2, ChoppedImageKind::Unfringed))],
        &[0.0; 4],
        &mut expected,
    )
    .unwrap();
    assert_eq!(command, expected);
    assert_eq!(
        p.workspace().integrator_state(),
        control.workspace().integrator_state()
    );
    let previous = p.workspace().integrator_state().to_vec();
    p.replace_properties(&[PropertyAssignment {
        id: 0,
        value: PropertyValue::Float(0.25),
    }])
    .unwrap();
    assert_eq!(p.workspace().integrator_state(), previous);
    assert!(
        p.replace_properties(&[PropertyAssignment {
            id: 1,
            value: PropertyValue::Float(1.25)
        }])
        .is_err()
    );
}
#[test]
fn spatial_pcg_and_correction_preparation_reject_invalid_boundaries_without_large_allocations() {
    assert!(Spatial::new((usize::MAX, 2), &[], 1, None, None, None, None, 0.8, 1.0).is_err());
    assert!(Spatial::new((2, 3), &[true; 6], 2, None, None, None, None, 0.8, 1.0).is_err());
    assert!(Pcg::new(usize::MAX, 2, 1, vec![], vec![], 0.1, 2, 1e-5, 1e-8).is_err());
    for change in [0, 1, 2, 3, 4, 5] {
        let mut c = correction();
        match change {
            0 => c.coordinate_order = vec![2, 1],
            1 => c.command_unit.clear(),
            2 => c.exposure_duration_nanoseconds = 0,
            3 => c.reconstructor_sign = "implicit".into(),
            4 => c.reference_difference[1] = f32::NAN,
            _ => c.bias_command[0] = 2.0,
        };
        assert!(Correction::prepare(c).is_err());
    }
}

#[test]
fn correction_admits_physical_identity_independently_and_rejects_missing_facts_atomically() {
    use filtergraph_algorithms::{
        AlgorithmDeclaration, DeclarationProcessError, NativeProcessError, SccCorrectionError as E,
    };
    assert_eq!(
        <Correction as AlgorithmDeclaration>::PORTS[0].required_sample_metadata(),
        &[
            "acquisition_identity",
            "exposure_duration_nanoseconds",
            "chopped_image_kind"
        ]
    );
    let mut p = Correction::prepare(correction()).unwrap();
    let mut control = Correction::prepare(correction()).unwrap();
    let identity = |generation, sequence| {
        SampleAcquisitionIdentity::new([1; 16], generation, sequence).unwrap()
    };
    let first =
        sample(100, ChoppedImageKind::Fringed).with_acquisition_identity(Some(identity(2, 1)));
    let mut command = [71.; 2];
    let mut expected = [71.; 2];
    Correction::process_with_metadata(&mut p, &[Some(first)], &[0.1, 0., 0., -0.1], &mut command)
        .unwrap();
    Correction::process_with_metadata(
        &mut control,
        &[Some(first)],
        &[0.1, 0., 0., -0.1],
        &mut expected,
    )
    .unwrap();
    let next =
        sample(400, ChoppedImageKind::Unfringed).with_acquisition_identity(Some(identity(2, 2)));
    for (metadata, error) in [
        (
            next.with_exposure_duration_nanoseconds(None),
            E::MetadataMissing,
        ),
        (
            next.with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(1)),
            E::ExposureMismatch,
        ),
        (next.with_acquisition_identity(None), E::MetadataMissing),
        (next.with_chopped_image_kind(None), E::MetadataMissing),
        (
            next.with_chopped_image_kind(Some(ChoppedImageKind::Absent)),
            E::MetadataMissing,
        ),
        (
            next.with_acquisition_identity(Some(identity(1, 999))),
            E::SequenceInvalid,
        ),
        (
            next.with_acquisition_identity(Some(identity(2, 1))),
            E::SequenceInvalid,
        ),
    ] {
        let before = p.workspace().command().to_vec();
        let history = p.workspace().integrator_state().to_vec();
        let exposure = p.workspace().exposure_command().to_vec();
        let previous = p.workspace().acquisition_identity();
        command.fill(71.);
        assert_eq!(
            Correction::process_with_metadata(&mut p, &[Some(metadata)], &[0.; 4], &mut command),
            Err(NativeProcessError::Process(
                DeclarationProcessError::Algorithm(error)
            ))
        );
        assert_eq!(command, [71.; 2]);
        assert_eq!(p.workspace().command(), before);
        assert_eq!(p.workspace().integrator_state(), history);
        assert_eq!(p.workspace().exposure_command(), exposure);
        assert_eq!(p.workspace().acquisition_identity(), previous);
        assert!(p.workspace().is_primed());
    }
    Correction::process_with_metadata(&mut p, &[Some(next)], &[0.; 4], &mut command).unwrap();
    Correction::process_with_metadata(&mut control, &[Some(next)], &[0.; 4], &mut expected)
        .unwrap();
    assert_eq!(command, expected);
    assert_eq!(
        p.workspace().integrator_state(),
        control.workspace().integrator_state()
    );
    // A new cohort can restart the physical sequence; another source can use its own generation.
    for id in [
        identity(3, 0),
        SampleAcquisitionIdentity::new([2; 16], 0, 0).unwrap(),
    ] {
        Correction::process_with_metadata(
            &mut p,
            &[Some(next.with_acquisition_identity(Some(id)))],
            &[0.; 4],
            &mut command,
        )
        .unwrap();
        assert_eq!(p.workspace().acquisition_identity(), Some(id));
    }
}

#[test]
fn matrix_replacement_preserves_shape_and_applies_each_new_calibration() {
    use filtergraph_algorithms::SccWeightedRegularizedReconstructionPlanError as Prep;
    let plan = Pcg::new(1, 1, 1, vec![1.0], vec![1.0], 1.0, 4, 1.0e-5, 1.0e-8).unwrap();
    let mut workspace = plan.workspace();
    for replacement in [
        plan.with_interaction_matrix(&[]),
        plan.with_regularization_operator(&[f32::NAN]),
    ] {
        let error = replacement.unwrap_err();
        assert_eq!(error, Prep::Matrix);
        assert!(error.to_string().contains("Matrix"), "{error}");
    }
    let h = plan.with_interaction_matrix(&[2.0]).unwrap();
    let l = h.with_regularization_operator(&[3.0]).unwrap();
    // In one dimension x = H w y / (H² w + λ L²).
    for (calibration, expected) in [(&plan, 6.0 / 3.0), (&h, 12.0 / 9.0), (&l, 12.0 / 17.0)] {
        let mut command = [19.0];
        calibration
            .process(&mut command, &mut workspace, (&[3.0], &[2.0]))
            .unwrap();
        assert!((command[0] - expected).abs() < 2.0e-6, "{command:?}");
    }
}

#[test]
fn spatial_calibration_identifies_bad_planes_fir_and_thresholds() {
    use filtergraph_algorithms::SccSpatialCarrierDemodulationPlanError as E;

    for (roi, phase, taps, support, fraction, expected) in [
        (
            &[true][..],
            None,
            &[1.0][..],
            1.0,
            1.0,
            E::CalibrationExtent,
        ),
        (&[false; 2][..], None, &[1.0][..], 1.0, 1.0, E::EmptyRoi),
        (
            &[true; 2][..],
            Some(&[f32::NAN, 0.0][..]),
            &[1.0][..],
            1.0,
            1.0,
            E::NonFiniteCalibration,
        ),
        (
            &[true; 2][..],
            None,
            &[1.0, -2.0, 1.0][..],
            1.0,
            1.0,
            E::Fir,
        ),
        (&[true; 2][..], None, &[f32::INFINITY][..], 1.0, 1.0, E::Fir),
        (
            &[true; 2][..],
            None,
            &[1.0][..],
            0.0,
            1.0,
            E::MinimumSupport,
        ),
        (
            &[true; 2][..],
            None,
            &[1.0][..],
            1.0,
            1.1,
            E::MinimumValidFraction,
        ),
    ] {
        let error = Spatial::new(
            (1, 2),
            roi,
            taps.len(),
            None,
            None,
            phase,
            Some(taps),
            support,
            fraction,
        )
        .unwrap_err();
        assert_eq!(error, expected);
        assert!(
            error.to_string().contains(&format!("{expected:?}")),
            "{error}"
        );
    }
    let plan = Spatial::new(
        (1, 2),
        &[true; 2],
        1,
        None,
        None,
        None,
        Some(&[2.0]),
        1.0,
        1.0,
    )
    .unwrap();
    let mut values = [71.0; 4];
    let mut weights = [71.0; 4];
    let mut valid = [false; 4];
    let mut fraction = -1.0;
    plan.process(
        (&mut values, &mut weights, &mut valid, &mut fraction),
        &mut plan.workspace(),
        (&[2.0, 3.0], &[true; 2], 1),
    )
    .unwrap();
    assert_eq!(values, [2.0, 3.0, 0.0, 0.0]);
    assert_eq!(valid, [true; 4]);
    assert_eq!(fraction, 1.0);
}
