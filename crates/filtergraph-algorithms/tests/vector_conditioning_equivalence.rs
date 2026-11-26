//! Generic conditioning construction, schema, and state/publication boundaries.
use filtergraph_algorithms::{
    ScalarOrVector, VectorConditioningF32 as D, VectorConditioningF32Configuration as C,
};
fn configuration() -> C {
    C {
        count: 2,
        requested_schema: "org.example.vector/1".into(),
        demanded_schema: "org.example.vector/1".into(),
        feedback_schema: "org.example.vector/1".into(),
        slew_limit: Some(ScalarOrVector::Scalar(0.25)),
        initial_current: Some(vec![0.5, -0.5]),
        quantization_origin: None,
        quantization_step: None,
        dither_seed: None,
        command_limits: (ScalarOrVector::Scalar(-1.0), ScalarOrVector::Scalar(1.0)),
    }
}
#[test]
fn validates_extent_schema_and_legacy_quantization_before_allocating() {
    for count in [0, u32::MAX as usize / 4 + 1, usize::MAX] {
        let mut c = configuration();
        c.count = count;
        assert!(D::prepare(c).is_err());
    }
    for index in 0..3 {
        let mut c = configuration();
        match index {
            0 => c.requested_schema = " \n".into(),
            1 => c.demanded_schema = "".into(),
            _ => c.feedback_schema = "\t".into(),
        };
        assert!(D::prepare(c).is_err());
    }
    let mut c = configuration();
    c.quantization_origin = Some(0.0);
    assert!(D::prepare(c).is_err());
    let mut c = configuration();
    c.quantization_step = Some(0.25);
    assert!(D::prepare(c).is_err());
}
#[test]
fn malformed_initial_state_and_component_bounds_are_rejected() {
    let mut c = configuration();
    c.initial_current = Some(vec![f32::NAN, 0.0]);
    assert!(D::prepare(c).is_err());
    let mut c = configuration();
    c.command_limits.0 = ScalarOrVector::Vector(vec![-1.0]);
    assert!(D::prepare(c).is_err());
    let mut c = configuration();
    c.slew_limit = Some(ScalarOrVector::Scalar(-0.1));
    assert!(D::prepare(c).is_err());
}
#[test]
fn rejection_does_not_adopt_slew_state_and_reset_restores_construction_value() {
    let mut d = D::prepare(configuration()).unwrap();
    let mut out = [71.0; 2];
    let mut feedback = [71.0; 2];
    D::process(&mut d, &[1.0, -1.0], &mut out, &mut feedback).unwrap();
    assert_eq!(out, [0.75, -0.75]);
    assert_eq!(feedback, [0.25, -0.25]);
    assert!(D::process(&mut d, &[1.0, f32::INFINITY], &mut out, &mut feedback).is_err());
    assert_eq!(out, [0.75, -0.75]);
    assert_eq!(feedback, [0.25, -0.25]);
    D::process(&mut d, &[-1.0, 1.0], &mut out, &mut feedback).unwrap();
    assert_eq!(out, [0.5, -0.5]);
    assert_eq!(feedback, [-1.5, 1.5]);
    d.reset();
    D::process(&mut d, &[1.0, -1.0], &mut out, &mut feedback).unwrap();
    assert_eq!(out, [0.75, -0.75]);
}

#[test]
fn feedback_overflow_preserves_outputs_and_committed_slew_history() {
    use filtergraph_algorithms::{
        AlgorithmPlan, PdmCommandError, PdmCommandInput, PdmCommandOutput, PdmCommandPlan,
    };
    macro_rules! check {
        ($t:ty) => {{
            let p = PdmCommandPlan::new(
                vec![-<$t>::MAX; 2].into(),
                vec![<$t>::MAX; 2].into(),
                Some(vec![0.0 as $t; 2].into()),
            )
            .unwrap();
            let mut w = p.workspace(&[0.0 as $t, -<$t>::MAX]).unwrap();
            let mut out = [71.0 as $t; 2];
            let mut feedback = [-17.0 as $t; 2];
            assert_eq!(
                p.process(
                    PdmCommandOutput {
                        demanded: &mut out,
                        constraint_feedback: &mut feedback
                    },
                    &mut w,
                    PdmCommandInput {
                        requested: &[1.0 as $t, <$t>::MAX]
                    }
                ),
                Err(PdmCommandError::NonFiniteOutput)
            );
            assert_eq!(out, [71.0 as $t; 2]);
            assert_eq!(feedback, [-17.0 as $t; 2]);
            assert_eq!(w.current(), &[0.0 as $t, -<$t>::MAX]);
            p.process(
                PdmCommandOutput {
                    demanded: &mut out,
                    constraint_feedback: &mut feedback,
                },
                &mut w,
                PdmCommandInput {
                    requested: &[0.0 as $t, -<$t>::MAX],
                },
            )
            .unwrap();
            assert_eq!(out, [0.0 as $t, -<$t>::MAX]);
            assert_eq!(feedback, [0.0 as $t; 2]);
        }};
    }
    check!(f32);
    check!(f64);
}
