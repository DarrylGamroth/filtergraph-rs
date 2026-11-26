//! Pseudo-open-loop reconstruction to closed-loop correction chain evidence.

use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmPlan, ClosedLoopCorrectionInput, ClosedLoopCorrectionPlan,
    PseudoOpenLoopReconstructionInput, PseudoOpenLoopReconstructionOutput,
    PseudoOpenLoopReconstructionPlan,
};

#[test]
fn closed_loop_correction_integrates_the_residual_error() {
    let pseudo_open_loop = PseudoOpenLoopReconstructionPlan::new(
        1,
        1,
        1,
        Arc::from([1.0_f32]),
        Arc::from([1.0_f32]),
        Arc::from([1.0_f32]),
    )
    .unwrap();
    let mut pseudo_open_loop_workspace = pseudo_open_loop.workspace();
    let mut pseudo_open_loop_measurements = [0.0];
    let mut residual_error = [0.0];

    pseudo_open_loop
        .process(
            PseudoOpenLoopReconstructionOutput::new(
                &mut pseudo_open_loop_measurements,
                &mut residual_error,
            ),
            &mut pseudo_open_loop_workspace,
            PseudoOpenLoopReconstructionInput::new(&[2.0], &[3.0]),
        )
        .unwrap();

    assert_eq!(pseudo_open_loop_measurements, [5.0]);
    assert_eq!(residual_error, [2.0]);

    let correction = ClosedLoopCorrectionPlan::new(-0.5_f32, 1.0, 1.0, 1).unwrap();
    let mut correction_workspace = correction.workspace();
    let mut correction_output = [0.0];
    correction
        .process(
            &mut correction_output,
            &mut correction_workspace,
            ClosedLoopCorrectionInput {
                x: &residual_error,
                constraint_feedback: None,
            },
        )
        .unwrap();

    assert_eq!(correction_output, [-1.0]);
    assert_ne!(correction_output, [-2.5]);
}
