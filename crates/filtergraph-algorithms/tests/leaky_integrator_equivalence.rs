//! Regression obligations established from JFG's CPU integrator contract.

use filtergraph_algorithms::{
    AlgorithmPlan, InputProgress, LeakyIntegratorError, LeakyIntegratorPlan,
    LeakyIntegratorWorkspace, ProcessError,
};

macro_rules! equivalence_tests {
    ($nonfinite:ident, $reset:ident, $scalar:ty) => {
        #[test]
        fn $nonfinite() {
            let regular = LeakyIntegratorPlan::new(0.5 as $scalar, 0.75).unwrap();
            let overflow = LeakyIntegratorPlan::new(2.0 as $scalar, 1.0).unwrap();
            for (index, (plan, input)) in [
                (regular, [1.0, <$scalar>::NAN]),
                (regular, [1.0, <$scalar>::INFINITY]),
                (regular, [1.0, <$scalar>::NEG_INFINITY]),
                (overflow, [1.0, <$scalar>::MAX]),
            ]
            .into_iter()
            .enumerate()
            {
                let mut workspace = LeakyIntegratorWorkspace::filled(2, 1.25).unwrap();
                let mut output = [-123.0; 2];
                let expected = if index == 3 {
                    LeakyIntegratorError::NonFiniteOutput
                } else {
                    LeakyIntegratorError::NonFiniteInput
                };
                assert_eq!(
                    plan.process(&mut output, &mut workspace, &input),
                    Err(expected)
                );
                assert_eq!(output, [-123.0; 2]);
                assert_eq!(workspace.state(), [1.25; 2]);
                regular
                    .process(&mut output, &mut workspace, &[2.0, -2.0])
                    .unwrap();
                assert_eq!(output, [1.9375, -0.0625]);
                assert_eq!(workspace.state(), output);
            }
        }

        #[test]
        fn $reset() {
            let plan = LeakyIntegratorPlan::new(0.5 as $scalar, 0.75).unwrap();
            let mut workspace = LeakyIntegratorWorkspace::filled(2, -2.5).unwrap();
            let mut output = [0.0; 2];
            plan.process(&mut output, &mut workspace, &[4.0, -8.0])
                .unwrap();
            workspace.reset();
            assert_eq!(workspace.state(), [-2.5; 2]);
            plan.process(&mut output, &mut workspace, &[4.0, -8.0])
                .unwrap();
            assert_eq!(output, [0.125, -5.875]);
        }
    };
}

equivalence_tests!(
    f32_rejection_preserves_output_and_state,
    f32_reset_restores_construction_state,
    f32
);

macro_rules! progressive_tests {
    ($name:ident, $scalar:ty) => {
        #[test]
        fn $name() {
            let plan = LeakyIntegratorPlan::new(0.5 as $scalar, 0.75).unwrap();
            let mut workspace = LeakyIntegratorWorkspace::filled(4, 1.25).unwrap();
            let mut output = [-123.0; 4];
            let mut input = [2.0, -2.0, 4.0, <$scalar>::NAN];
            plan.start(&mut workspace).unwrap();
            let _ = plan
                .process_range(&mut output, &mut workspace, &input, 0..2)
                .unwrap();
            assert_eq!(output, [1.9375, -0.0625, -123.0, -123.0]);
            assert_eq!(
                plan.process_range(&mut output, &mut workspace, &input, 2..4),
                Err(LeakyIntegratorError::NonFiniteInput)
            );
            assert_eq!(output, [1.9375, -0.0625, -123.0, -123.0]);
            assert_eq!(workspace.state(), [1.25; 4]);
            assert_eq!(
                plan.finish(&mut output, &mut workspace),
                Err(LeakyIntegratorError::Process(ProcessError::Incomplete {
                    processed: 2,
                    extent: 4,
                }))
            );
            input[3] = -4.0;
            let _ = plan
                .process_range(&mut output, &mut workspace, &input, 2..4)
                .unwrap();
            plan.finish(&mut output, &mut workspace).unwrap();
            assert_eq!(output, [1.9375, -0.0625, 2.9375, -1.0625]);
            assert_eq!(workspace.state(), output);

            let previous_state = workspace.state().to_vec();
            let overflow = LeakyIntegratorPlan::new(2.0 as $scalar, 1.0).unwrap();
            overflow.start(&mut workspace).unwrap();
            input = [1.0, -1.0, 2.0, <$scalar>::MAX];
            let _ = overflow
                .process_range(&mut output, &mut workspace, &input, 0..2)
                .unwrap();
            let successful_prefix = output;
            assert_eq!(
                overflow.process_range(&mut output, &mut workspace, &input, 2..4),
                Err(LeakyIntegratorError::NonFiniteOutput)
            );
            assert_eq!(output, successful_prefix);
            assert_eq!(workspace.state(), previous_state);
            overflow.abort(&mut workspace).unwrap();
            assert_eq!(workspace.state(), previous_state);
            workspace.reset();
            assert_eq!(workspace.state(), [1.25; 4]);
            plan.process(&mut output, &mut workspace, &[2.0; 4])
                .unwrap();
            assert_eq!(output, [1.9375; 4]);
        }
    };
}

progressive_tests!(f32_rejected_range_preserves_prefix_and_history, f32);
progressive_tests!(f64_rejected_range_preserves_prefix_and_history, f64);
equivalence_tests!(
    f64_rejection_preserves_output_and_state,
    f64_reset_restores_construction_state,
    f64
);
