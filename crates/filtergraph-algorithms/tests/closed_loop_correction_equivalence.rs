//! Contiguous-lane correction, rejection, and fused-arithmetic boundaries.

use filtergraph_algorithms::{
    AlgorithmPlan, AlgorithmProperties, ClosedLoopCorrectionError, ClosedLoopCorrectionInput,
    ClosedLoopCorrectionPlan, PropertyAssignment, PropertyValue,
};

macro_rules! correction_lanes {
    ($module:ident, $scalar:ty, $property:ident) => {
        mod $module {
            use super::*;

            #[test]
            fn lanes_and_tails_match_the_scalar_recurrence() {
                for n in [1, 7, 8, 9, 31, 32, 33, 277, 4096] {
                    let plan = ClosedLoopCorrectionPlan::new(
                        0.37 as $scalar,
                        0.91 as $scalar,
                        0.41 as $scalar,
                        n,
                    )
                    .unwrap();
                    let mut workspace = plan.workspace();
                    let mut output = vec![0.0; n];
                    let mut expected_output = vec![0.0 as $scalar; n];
                    let mut expected_state = vec![0.0 as $scalar; n];
                    for frame in 0..8 {
                        let x: Vec<$scalar> = (0..n)
                            .map(|i| ((i * 7 + frame * 11) % 31) as $scalar / 17.0 - 0.9)
                            .collect();
                        let feedback: Vec<$scalar> = (0..n)
                            .map(|i| ((i * 3 + frame * 5) % 13) as $scalar / 19.0 - 0.3)
                            .collect();
                        for i in 0..n {
                            if frame != 0 {
                                expected_state[i] = (-plan.anti_windup_gain())
                                    .mul_add(feedback[i], expected_output[i]);
                            }
                            expected_output[i] =
                                plan.gain().mul_add(x[i], plan.pole() * expected_state[i]);
                        }
                        plan.process(
                            &mut output,
                            &mut workspace,
                            ClosedLoopCorrectionInput {
                                x: &x,
                                constraint_feedback: (frame != 0).then_some(feedback.as_slice()),
                            },
                        )
                        .unwrap();
                        assert_eq!(output, expected_output, "n={n}, frame={frame}");
                        assert_eq!(workspace.state(), expected_state, "n={n}, frame={frame}");
                    }
                }
            }

            #[test]
            fn invalid_lanes_preserve_publication_and_allow_the_next_update() {
                for n in [33, 277] {
                    let plan = ClosedLoopCorrectionPlan::new(0.5 as $scalar, 0.75, 0.8, n).unwrap();
                    let x = vec![0.25; n];
                    let feedback = vec![0.1; n];
                    let mut workspace = plan.workspace();
                    let mut output = vec![0.0; n];
                    plan.process(
                        &mut output,
                        &mut workspace,
                        ClosedLoopCorrectionInput {
                            x: &x,
                            constraint_feedback: None,
                        },
                    )
                    .unwrap();
                    for index in [0, 7, 8, 31, 32, n - 1] {
                        for invalid in
                            [<$scalar>::NAN, <$scalar>::INFINITY, <$scalar>::NEG_INFINITY]
                        {
                            for invalid_feedback in [false, true] {
                                let mut rejected = workspace.clone();
                                let mut rejected_output = output.clone();
                                let mut bad_x = x.clone();
                                let mut bad_feedback = feedback.clone();
                                if invalid_feedback {
                                    bad_feedback[index] = invalid;
                                } else {
                                    bad_x[index] = invalid;
                                }
                                assert_eq!(
                                    plan.process(
                                        &mut rejected_output,
                                        &mut rejected,
                                        ClosedLoopCorrectionInput {
                                            x: &bad_x,
                                            constraint_feedback: Some(&bad_feedback),
                                        },
                                    ),
                                    Err(if invalid_feedback {
                                        ClosedLoopCorrectionError::NonFiniteFeedback
                                    } else {
                                        ClosedLoopCorrectionError::NonFiniteInput
                                    }),
                                );
                                assert_eq!(rejected_output, output);
                                assert_eq!(rejected.state(), workspace.state());
                                assert_eq!(rejected.last_output(), workspace.last_output());
                                let mut expected = workspace.clone();
                                let mut expected_output = output.clone();
                                for (candidate, candidate_output) in [
                                    (&mut rejected, &mut rejected_output),
                                    (&mut expected, &mut expected_output),
                                ] {
                                    plan.process(
                                        candidate_output,
                                        candidate,
                                        ClosedLoopCorrectionInput {
                                            x: &x,
                                            constraint_feedback: Some(&feedback),
                                        },
                                    )
                                    .unwrap();
                                }
                                assert_eq!(rejected_output, expected_output);
                                assert_eq!(rejected.state(), expected.state());
                            }
                        }
                    }
                }
            }

            #[test]
            fn output_overflow_rejects_every_lane_and_retains_fused_cancellation() {
                let n = 33;
                let initial = ClosedLoopCorrectionPlan::new(1.0 as $scalar, 1.0, 2.0, n).unwrap();
                let mut workspace = initial.workspace();
                let mut output = vec![0.0; n];
                initial
                    .process(
                        &mut output,
                        &mut workspace,
                        ClosedLoopCorrectionInput {
                            x: &vec![<$scalar>::MAX; n],
                            constraint_feedback: None,
                        },
                    )
                    .unwrap();
                // The anti-windup product overflows in isolation; its FMA is finite.
                initial
                    .process(
                        &mut output,
                        &mut workspace,
                        ClosedLoopCorrectionInput {
                            x: &vec![0.0; n],
                            constraint_feedback: Some(&vec![<$scalar>::MAX; n]),
                        },
                    )
                    .unwrap();
                assert_eq!(output, vec![-<$scalar>::MAX; n]);
                let large_gain = initial
                    .prepare_properties(&[PropertyAssignment {
                        id: 0,
                        value: PropertyValue::$property(<$scalar>::MAX),
                    }])
                    .unwrap();
                // The gain product also overflows in isolation, before cancellation.
                large_gain
                    .process(
                        &mut output,
                        &mut workspace,
                        ClosedLoopCorrectionInput {
                            x: &vec![2.0; n],
                            constraint_feedback: Some(&vec![0.0; n]),
                        },
                    )
                    .unwrap();
                assert_eq!(output, vec![<$scalar>::MAX; n]);
                for index in [0, 7, 8, 31, 32] {
                    let mut rejected = workspace.clone();
                    let mut rejected_output = output.clone();
                    let mut x = vec![0.0; n];
                    x[index] = 2.0;
                    assert_eq!(
                        large_gain.process(
                            &mut rejected_output,
                            &mut rejected,
                            ClosedLoopCorrectionInput {
                                x: &x,
                                constraint_feedback: Some(&vec![0.0; n])
                            },
                        ),
                        Err(ClosedLoopCorrectionError::NonFiniteOutput)
                    );
                    assert_eq!(rejected_output, output);
                    assert_eq!(rejected.state(), workspace.state());
                    assert_eq!(rejected.last_output(), workspace.last_output());
                }
            }
        }
    };
}

correction_lanes!(float32, f32, Float);
correction_lanes!(float64, f64, Double);
