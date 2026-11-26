//! Discriminating contracts introduced by current JFG CPU arithmetic and hold.
use filtergraph_algorithms::{
    AlgorithmPlan, AlgorithmProperties, ClosedLoopCorrectionError, ClosedLoopCorrectionInput,
    ClosedLoopCorrectionPlan, LeakyIntegratorError, LeakyIntegratorPlan, LeakyIntegratorWorkspace,
    PropertyAssignment, PropertyValue, VectorLinearCombinationPlan,
};

#[test]
fn leaky_fused_cancellation_stays_finite() {
    let p = LeakyIntegratorPlan::new(f32::MAX, 1.0).unwrap();
    let mut w = LeakyIntegratorWorkspace::filled(1, -f32::MAX).unwrap();
    let mut out = [0.0];
    p.process(&mut out, &mut w, &[2.0]).unwrap();
    assert_eq!(out, [f32::MAX]);
    assert_eq!(w.state(), out);
}

#[test]
fn vector_fuses_left_product_after_rounding_right_product() {
    let p = VectorLinearCombinationPlan::new(1, f32::MAX, 1.0).unwrap();
    let mut w = p.workspace();
    let mut out = [0.0];
    p.process(&mut out, &mut w, (&[2.0], &[-f32::MAX])).unwrap();
    assert_eq!(out, [f32::MAX]);
}

#[test]
fn leaky_closed_property_holds_validated_state_and_resumes() {
    let p = LeakyIntegratorPlan::new(0.5_f32, 0.75).unwrap();
    let held = p
        .prepare_properties(&[PropertyAssignment {
            id: 2,
            value: PropertyValue::Bool(false),
        }])
        .unwrap();
    let mut w = LeakyIntegratorWorkspace::filled(2, 1.0).unwrap();
    let mut out = [9.0; 2];
    held.process(&mut out, &mut w, &[f32::MAX; 2]).unwrap();
    assert_eq!(out, [1.0; 2]);
    assert_eq!(
        held.process(&mut out, &mut w, &[f32::NAN; 2]),
        Err(LeakyIntegratorError::NonFiniteInput)
    );
    assert_eq!(out, [1.0; 2]);
    let resumed = held
        .prepare_properties(&[PropertyAssignment {
            id: 2,
            value: PropertyValue::Bool(true),
        }])
        .unwrap();
    resumed.process(&mut out, &mut w, &[2.0; 2]).unwrap();
    assert_eq!(out, [1.75; 2]);
    w.reset();
    held.process(&mut out, &mut w, &[0.0; 2]).unwrap();
    assert_eq!(out, [1.0; 2]);
}

#[test]
fn correction_hold_checks_feedback_publishes_initial_zero_and_resumes() {
    let p = ClosedLoopCorrectionPlan::new(0.5_f32, 0.75, 0.25, 1).unwrap();
    let held = p
        .prepare_properties(&[PropertyAssignment {
            id: 4,
            value: PropertyValue::Bool(false),
        }])
        .unwrap();
    let mut w = p.workspace();
    let mut out = [9.0];
    held.process(
        &mut out,
        &mut w,
        ClosedLoopCorrectionInput {
            x: &[f32::MAX],
            constraint_feedback: None,
        },
    )
    .unwrap();
    assert_eq!(out, [0.0]);
    assert_eq!(
        held.process(
            &mut out,
            &mut w,
            ClosedLoopCorrectionInput {
                x: &[1.0],
                constraint_feedback: None
            }
        ),
        Err(ClosedLoopCorrectionError::MissingFeedback)
    );
    assert_eq!(
        held.process(
            &mut out,
            &mut w,
            ClosedLoopCorrectionInput {
                x: &[1.0],
                constraint_feedback: Some(&[f32::NAN])
            }
        ),
        Err(ClosedLoopCorrectionError::NonFiniteFeedback)
    );
    p.process(
        &mut out,
        &mut w,
        ClosedLoopCorrectionInput {
            x: &[2.0],
            constraint_feedback: Some(&[0.0]),
        },
    )
    .unwrap();
    assert_eq!(out, [1.0]);
    held.process(
        &mut out,
        &mut w,
        ClosedLoopCorrectionInput {
            x: &[f32::MAX],
            constraint_feedback: Some(&[f32::MAX]),
        },
    )
    .unwrap();
    assert_eq!(out, [1.0]);
    p.process(
        &mut out,
        &mut w,
        ClosedLoopCorrectionInput {
            x: &[2.0],
            constraint_feedback: Some(&[0.5]),
        },
    )
    .unwrap();
    assert_eq!(out, [1.65625]);
}

#[test]
fn covariance_fused_cancellation_adopts_finite_third_sample() {
    use filtergraph_algorithms::{DocrimeCovarianceInput, DocrimeCovariancePlan};
    macro_rules! check {
        ($t:ty) => {{
            let p = DocrimeCovariancePlan::<$t>::new(1, 1).unwrap();
            let mut w = p.workspace();
            for (x, y) in [(0.0, 0.0), (<$t>::MAX / 2.0, -2.0)] {
                p.process((), &mut w, DocrimeCovarianceInput::new(&[x], &[y]))
                    .unwrap();
            }
            let x = 0.75 * <$t>::MAX;
            let dx = x - w.measurement_mean()[0];
            let dy = 3.0 - w.corrector_sample_mean()[0];
            let weighted = (2.0 as $t / 3.0) * dx;
            assert!(!(weighted * dy).is_finite());
            let expected = weighted.mul_add(dy, w.measurement_corrector_cross_covariance_sum()[0]);
            assert!(expected.is_finite());
            p.process((), &mut w, DocrimeCovarianceInput::new(&[x], &[3.0]))
                .unwrap();
            assert_eq!(w.sample_count(), 3);
            assert_eq!(w.measurement_corrector_cross_covariance_sum(), [expected]);
        }};
    }
    check!(f32);
    check!(f64);
}

#[test]
fn correction_fused_cancellation_covers_each_arithmetic_stage() {
    use std::sync::Arc;
    macro_rules! check {
        ($t:ty) => {{
            for stage in 0..3 {
                let hidden = usize::from(stage == 2);
                let forward: Arc<[$t]> = vec![1.0; hidden].into();
                let backward = Arc::clone(&forward);
                let initial = ClosedLoopCorrectionPlan::with_hidden_modes(
                    1.0 as $t,
                    0.0,
                    0.0,
                    1,
                    hidden,
                    0.0,
                    Arc::clone(&forward),
                    Arc::clone(&backward),
                )
                .unwrap();
                let p = ClosedLoopCorrectionPlan::with_hidden_modes(
                    if stage == 0 { 2.0 } else { 1.0 },
                    if stage == 0 { 1.0 } else { 0.0 },
                    if stage == 1 { 2.0 } else { 0.0 },
                    1,
                    hidden,
                    if stage == 2 { 2.0 } else { 0.0 },
                    forward,
                    backward,
                )
                .unwrap();
                let mut w = p.workspace();
                let mut out = [0.0];
                initial
                    .process(
                        &mut out,
                        &mut w,
                        ClosedLoopCorrectionInput {
                            x: &[if stage == 0 { -<$t>::MAX } else { <$t>::MAX }],
                            constraint_feedback: None,
                        },
                    )
                    .unwrap();
                p.process(
                    &mut out,
                    &mut w,
                    ClosedLoopCorrectionInput {
                        x: &[if stage == 0 { <$t>::MAX } else { 0.0 }],
                        constraint_feedback: Some(&[if stage == 1 { <$t>::MAX } else { 0.0 }]),
                    },
                )
                .unwrap();
                assert_eq!(out, [if stage == 0 { <$t>::MAX } else { 0.0 }]);
                assert_eq!(w.state(), [-<$t>::MAX]);
            }
        }};
    }
    check!(f32);
    check!(f64);
}

#[test]
fn recovery_clears_controller_and_filter_but_retains_conditioned_physical_history() {
    use filtergraph_algorithms::{
        BiquadCascadePlan, PdmCommandInput, PdmCommandOutput, PdmCommandPlan,
    };
    use std::sync::Arc;
    let controller = ClosedLoopCorrectionPlan::new(0.5_f32, 0.9, 0.1, 1).unwrap();
    let mut c = controller.workspace();
    let mut out = [0.0];
    controller
        .process(
            &mut out,
            &mut c,
            ClosedLoopCorrectionInput {
                x: &[1.0],
                constraint_feedback: None,
            },
        )
        .unwrap();
    c.recover();
    assert_eq!(c.state(), [0.0]);
    assert_eq!(c.last_output(), None);
    let filter = BiquadCascadePlan::new(vec![0.5_f32, 0.25, 0.0, 0.0, 0.0], 1, 1, 1).unwrap();
    let mut f = filter.workspace();
    filter.process(&mut out, &mut f, &[1.0]).unwrap();
    assert!(f.state().iter().any(|&v| v != 0.0));
    f.recover();
    assert!(f.state().iter().all(|&v| v == 0.0));
    let pdm = PdmCommandPlan::new(
        Arc::from([-1.0_f32]),
        Arc::from([1.0]),
        Some(Arc::from([0.1])),
    )
    .unwrap();
    let mut actual = pdm.workspace(&[0.0]).unwrap();
    let mut demanded = [0.0];
    let mut feedback = [0.0];
    pdm.process(
        PdmCommandOutput {
            demanded: &mut demanded,
            constraint_feedback: &mut feedback,
        },
        &mut actual,
        PdmCommandInput { requested: &[0.7] },
    )
    .unwrap();
    let mut expected = actual.clone();
    actual.recover();
    assert_eq!(actual.current(), expected.current());
    for requested in [0.8, -0.5, 0.0, 0.25] {
        let mut expected_demand = [0.0];
        let mut expected_feedback = [0.0];
        pdm.process(
            PdmCommandOutput {
                demanded: &mut demanded,
                constraint_feedback: &mut feedback,
            },
            &mut actual,
            PdmCommandInput {
                requested: &[requested],
            },
        )
        .unwrap();
        pdm.process(
            PdmCommandOutput {
                demanded: &mut expected_demand,
                constraint_feedback: &mut expected_feedback,
            },
            &mut expected,
            PdmCommandInput {
                requested: &[requested],
            },
        )
        .unwrap();
        assert_eq!(demanded, expected_demand);
        assert_eq!(feedback, expected_feedback);
        assert_eq!(actual.current(), expected.current());
    }
}

#[test]
fn allocation_recovery_retains_slow_stage_phase_and_physical_slew_state() {
    use filtergraph_algorithms::{
        CorrectorAllocationF32 as D, CorrectorAllocationF32Configuration, CorrectorAllocationOutput,
    };
    let declared = D::prepare(CorrectorAllocationF32Configuration {
        controller_command_count: 1,
        stage_command_counts: vec![1],
        update_periods: vec![3],
        command_projection: vec![1.0],
        realization_projection: vec![1.0],
        section_count: 1,
        initial_coefficients: Some(vec![0.5, 0.25, 0.0, 0.0, 0.0]),
        command_limits: (vec![-1.0], vec![1.0]),
        slew_limits: Some(vec![0.1]),
        command_schema: "test.recovery/1".into(),
    })
    .unwrap();
    let (plan, mut w, _) = declared.into_parts();
    let mut commands = [0.0];
    let mut feedback = [0.0];
    plan.process(
        CorrectorAllocationOutput {
            commands: &mut commands,
            constraint_feedback: &mut feedback,
        },
        &mut w,
        &[1.0],
    )
    .unwrap();
    assert_eq!(commands, [0.1]);
    w.recover();
    // The two non-due samples retain the held command. The next due sample
    // slews from that physical command, rather than the cold-start origin.
    for expected in [0.1, 0.1, 0.2] {
        plan.process(
            CorrectorAllocationOutput {
                commands: &mut commands,
                constraint_feedback: &mut feedback,
            },
            &mut w,
            &[1.0],
        )
        .unwrap();
        assert_eq!(commands, [expected]);
    }
}
