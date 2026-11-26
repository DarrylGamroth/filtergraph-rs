//! Boundary and publication checks supplementing the live JFG model fixtures.
use filtergraph_algorithms::*;

#[test]
fn shape_selector_uses_first_direct_rows_and_retains_physical_readback() {
    let completion = [0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let inverse = [
        1.0, 0.0, 2.0, 3.0, -1.0, 0.5, -2.0, 0.25, 1.0, 9.0, 8.0, 7.0,
    ];
    let plan = VdmShapeOffsetPlan::new(2, 4, 3, &completion, &inverse, &[0.5, -2.0, 1.0]).unwrap();
    assert_eq!(plan.active_offset(), &[-0.5, 4.0]);
    assert_eq!(plan.controlled_actuators(), &[true, true, true]);
    let mut w = plan.workspace();
    let mut output = [71.0; 2];
    let mut current = [71.0; 3];
    plan.process((&mut output, &mut current), &mut w, &[1.0, 2.0])
        .unwrap();
    assert_eq!(output, [0.5, 6.0]);
    assert_eq!(current, [0.5, -2.0, 1.0]);
    w.reset();
    plan.process((&mut output, &mut current), &mut w, &[1.0, 2.0])
        .unwrap();
    assert_eq!(output, [0.5, 6.0]);
    assert!(matches!(
        plan.with_completion(&[0.0; 8]),
        Err(VdmShapeOffsetPlanError::DirectSelector)
    ));
    assert!(matches!(
        plan.with_shape_offset(&[f32::MAX; 3]),
        Err(VdmShapeOffsetPlanError::NonFinite)
    ));
}

#[test]
fn shape_rejection_is_coordinate_ordered_and_preserves_both_outputs() {
    let plan = VdmShapeOffsetPlan::new(
        2,
        2,
        2,
        &[1.0, 0.0, 0.0, 1.0],
        &[1.0, 0.0, 0.0, 1.0],
        &[f32::MAX, 0.0],
    )
    .unwrap();
    let mut w = plan.workspace();
    let mut output = [71.0; 2];
    let mut current = [71.0; 2];
    // The first sum overflows before a later nonfinite input is inspected.
    assert_eq!(
        plan.process((&mut output, &mut current), &mut w, &[f32::MAX, f32::NAN]),
        Err(VectorLinearCombinationError::NonFiniteOutput)
    );
    assert_eq!(output, [71.0; 2]);
    assert_eq!(current, [71.0; 2]);
    assert_eq!(
        plan.process((&mut output, &mut current[..1]), &mut w, &[0.0; 2]),
        Err(VectorLinearCombinationError::DimensionMismatch)
    );
    assert_eq!(output, [71.0; 2]);
    plan.process((&mut output, &mut current), &mut w, &[0.0, -0.0])
        .unwrap();
    assert_eq!(output[1].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn invalid_shape_dimensions_are_rejected_before_neutral_allocation() {
    for (active, full, physical) in [
        (0, 1, 1),
        (2, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, usize::MAX, 1),
        (1, 1, u32::MAX as usize),
    ] {
        assert!(VdmShapeOffsetPlan::neutral(active, full, physical).is_err());
    }
}

#[test]
fn ml_model_readiness_parameter_validation_and_reset_keep_output_atomic() {
    assert!(matches!(
        MlLlowfsPlan::new("unknown"),
        Err(MlLlowfsError::Model)
    ));
    let p = MlLlowfsPlan::new(ML_LLOWFS_MODEL).unwrap();
    let mut w = p.workspace();
    let mut out = [71.0; 14];
    assert_eq!(
        p.process(&mut out, &mut w, &[f32::NAN; 400]),
        Err(MlLlowfsError::ParametersMissing)
    );
    assert_eq!(out, [71.0; 14]);
    assert!(p.with_parameters(&[0.0; 1]).is_err());
    let mut parameters = vec![0.0; ML_LLOWFS_PARAMETER_COUNT];
    for (i, v) in parameters[ML_LLOWFS_PARAMETER_COUNT - 14..]
        .iter_mut()
        .enumerate()
    {
        *v = i as f32 - 3.0;
    }
    let p = p.with_parameters(&parameters).unwrap();
    let input: Vec<_> = (0..400).map(|i| 1.0 + (i % 7) as f32).collect();
    p.process(&mut out, &mut w, &input).unwrap();
    assert_eq!(out, core::array::from_fn(|i| i as f32 - 3.0));
    w.reset();
    p.process(&mut out, &mut w, &input).unwrap();
    assert_eq!(out, core::array::from_fn(|i| i as f32 - 3.0));
    out.fill(71.0);
    for (value, expected) in [
        (f32::NAN, MlLlowfsError::NonFiniteInput),
        (f32::INFINITY, MlLlowfsError::NonFiniteInput),
        (0.0, MlLlowfsError::InvalidFlux),
        (-1.0, MlLlowfsError::InvalidFlux),
        (f32::MAX, MlLlowfsError::InvalidFlux),
        (1.0, MlLlowfsError::InvalidStandardDeviation),
    ] {
        assert_eq!(p.process(&mut out, &mut w, &[value; 400]), Err(expected));
        assert_eq!(out, [71.0; 14]);
    }
    parameters[42] = f32::NAN;
    assert!(p.with_parameters(&parameters).is_err());
}

#[test]
fn dense_declaration_preserves_computed_overflow_and_recovers() {
    let mut d = DenseVectorReconstructorF32::prepare(DenseVectorReconstructorF32Configuration {
        input_count: 2,
        output_count: 1,
        initial_reconstructor: Some(vec![1.0, 2.0]),
        input_schema: "org.example.input/1".into(),
        output_schema: "org.example.output/1".into(),
        reconstructor_schema: "org.example.matrix/1".into(),
    })
    .unwrap();
    let mut out = [71.0];
    assert!(DenseVectorReconstructorF32::process(&mut d, &[1.0, f32::NAN], &mut out).is_err());
    assert_eq!(out, [71.0]);
    assert!(DenseVectorReconstructorF32::process(&mut d, &[1.0, f32::MAX], &mut out).is_err());
    assert_eq!(out, [f32::INFINITY]);
    DenseVectorReconstructorF32::process(&mut d, &[1.0, 2.0], &mut out).unwrap();
    assert_eq!(out, [5.0]);
}

#[test]
fn acquisition_metadata_preserves_identity_optical_role_and_positive_duration() {
    assert!(SampleAcquisitionIdentity::new([0; 16], 0, 0).is_err());
    let full_width = SampleAcquisitionIdentity::new([1; 16], u64::MAX, u64::MAX).unwrap();
    assert_eq!(full_width.generation(), u64::MAX);
    assert_eq!(full_width.sequence(), u64::MAX);
    let id = SampleAcquisitionIdentity::new([1; 16], 7, 19).unwrap();
    let m = SampleMetadata::new(19, 4)
        .with_acquisition_identity(Some(id))
        .with_chopped_image_kind(Some(ChoppedImageKind::Mixed))
        .with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(123));
    assert_eq!(m.acquisition_identity(), Some(id));
    assert_eq!(id.domain(), [1; 16]);
    assert_eq!(id.generation(), 7);
    assert_eq!(id.sequence(), 19);
    assert_eq!(m.chopped_image_kind(), Some(ChoppedImageKind::Mixed));
    assert_eq!(m.exposure_duration_nanoseconds(), Some(123));
    let derived = SampleMetadata::new(20, 0).with_acquisition_identity(Some(id));
    assert_eq!(derived.sequence(), 20);
    assert_eq!(derived.acquisition_identity().unwrap().sequence(), 19);
    let changed = m.with_offset(0).with_terminal(true);
    assert_eq!(changed.acquisition_identity(), Some(id));
    assert_eq!(changed.chopped_image_kind(), m.chopped_image_kind());
    assert_eq!(changed.exposure_duration_nanoseconds(), Some(123));
    assert_eq!(
        m.with_exposure_duration_nanoseconds(None)
            .exposure_duration_nanoseconds(),
        None
    );
}

#[test]
fn ml_cancellation_flux_uses_the_qualified_reference_reduction() {
    let p = MlLlowfsPlan::new(ML_LLOWFS_MODEL)
        .unwrap()
        .with_parameters(&vec![0.0; ML_LLOWFS_PARAMETER_COUNT])
        .unwrap();
    let mut w = p.workspace();
    let mut input = vec![1.0; 400];
    input[0] = 16777216.0;
    input[399] = -16777216.0;
    let mut output = [71.0; 14];
    p.process(&mut output, &mut w, &input).unwrap();
    assert_eq!(output, [0.0; 14]);
}

#[test]
fn exposure_start_and_duration_are_independent_facts() {
    for clock in [
        SampleClock::Monotonic,
        SampleClock::Realtime,
        SampleClock::Tai,
        SampleClock::Model,
    ] {
        let start = SampleExposureStart::new(-42, clock);
        let m = SampleMetadata::new(9, 2).with_exposure_start(Some(start));
        assert_eq!(m.exposure_start(), Some(start));
        assert_eq!(m.exposure_duration_nanoseconds(), None);
        assert_eq!(start.nanoseconds(), -42);
        assert_eq!(start.clock(), clock);
        let duration_only = m
            .with_exposure_start(None)
            .with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(u64::MAX));
        assert_eq!(duration_only.exposure_start(), None);
        assert_eq!(
            duration_only.exposure_duration_nanoseconds(),
            Some(u64::MAX)
        );
    }
}

#[test]
fn compact_metadata_preserves_unsigned_reference_and_independent_classifications() {
    let empty = SampleMetadata::new(19, 0);
    assert_eq!(empty.reference_timestamp(), None);
    assert_eq!(empty.trigger_state(), None);
    assert_eq!(empty.chopped_image_kind(), None);
    let reference = SampleReferenceTimestamp::new(u64::MAX - 1, Some(0), SampleClock::Tai).unwrap();
    assert_eq!(reference.nanoseconds(), u64::MAX - 1);
    assert_eq!(reference.duration_nanoseconds(), Some(0));
    assert_eq!(reference.clock(), SampleClock::Tai);
    assert_eq!(
        SampleReferenceTimestamp::new(u64::MAX, None, SampleClock::Model),
        Err(SampleReferenceTimestampError::Timestamp)
    );
    assert_eq!(
        SampleReferenceTimestamp::new(1, Some(u64::MAX), SampleClock::Model),
        Err(SampleReferenceTimestampError::Duration)
    );
    assert_eq!(
        SampleReferenceTimestamp::new(0, None, SampleClock::Monotonic)
            .unwrap()
            .duration_nanoseconds(),
        None
    );
    for optical in [
        ChoppedImageKind::Absent,
        ChoppedImageKind::Unfringed,
        ChoppedImageKind::Fringed,
        ChoppedImageKind::Mixed,
    ] {
        for electrical in [
            SampleTriggerState::Absent,
            SampleTriggerState::Low,
            SampleTriggerState::High,
            SampleTriggerState::Mixed,
        ] {
            let m = empty
                .with_reference_timestamp(Some(reference))
                .with_chopped_image_kind(Some(optical))
                .with_trigger_state(Some(electrical));
            assert_eq!(m.chopped_image_kind(), Some(optical));
            assert_eq!(m.trigger_state(), Some(electrical));
            assert_eq!(m.reference_timestamp(), Some(reference));
            assert_eq!(
                m.trigger_high(),
                match electrical {
                    SampleTriggerState::Low => Some(false),
                    SampleTriggerState::High => Some(true),
                    _ => None,
                }
            );
            assert_eq!(
                m.with_trigger_state(None).chopped_image_kind(),
                Some(optical)
            );
            assert_eq!(
                m.with_chopped_image_kind(None).trigger_state(),
                Some(electrical)
            );
        }
    }
    assert_eq!(
        [
            SampleClock::Monotonic as u32,
            SampleClock::Realtime as u32,
            SampleClock::Tai as u32,
            SampleClock::Model as u32
        ],
        [1, 2, 3, 4]
    );
}
