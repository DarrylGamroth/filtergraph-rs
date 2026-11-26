//! Rejection of workspaces prepared for incompatible algorithm plans.

use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmPlan, ClosedLoopCorrectionError, ClosedLoopCorrectionInput, ClosedLoopCorrectionPlan,
    ControllerProjection, ControllerToVdmError, ControllerToVdmPlan, DeformableMirrorCommandError,
    DeformableMirrorCommandInput, DeformableMirrorCommandOutput, DeformableMirrorCommandPlan,
    OpticalGainDitherError, OpticalGainDitherMode, OpticalGainDitherPlan,
    OpticalGainDitherSchedule, OpticalGainEstimateOutput, OpticalGainEstimatorError,
    OpticalGainEstimatorInput, OpticalGainEstimatorPlan, PdmCommandError, PdmCommandInput,
    PdmCommandOutput, PdmCommandPlan, PdmFeedbackToVdmError, PdmFeedbackToVdmPlan, PdmProjection,
    PdmStrokeOptimizationError, PdmStrokeOptimizationPlan, VdmToPdmError, VdmToPdmPlan,
};

fn controller_plan(vdm_size: usize) -> ControllerToVdmPlan<f32> {
    let projection = ControllerProjection::new(
        1,
        Arc::from(vec![1.0; vdm_size]),
        Arc::from(vec![1.0; vdm_size]),
    )
    .unwrap();
    ControllerToVdmPlan::new(vdm_size, Arc::from([projection])).unwrap()
}

fn pdm_projection(full_vdm_size: usize) -> PdmProjection<f32> {
    PdmProjection::new(
        1,
        Arc::from(vec![1.0; full_vdm_size]),
        Arc::from(vec![1.0; full_vdm_size]),
    )
    .unwrap()
}

fn vdm_to_pdm_plan(full_vdm_size: usize) -> VdmToPdmPlan<f32> {
    VdmToPdmPlan::new(
        1,
        full_vdm_size,
        Arc::from(vec![1.0; full_vdm_size]),
        Arc::from([pdm_projection(full_vdm_size)]),
    )
    .unwrap()
}

fn pdm_feedback_plan(full_vdm_size: usize) -> PdmFeedbackToVdmPlan<f32> {
    PdmFeedbackToVdmPlan::new(
        1,
        full_vdm_size,
        Arc::from(vec![1.0; full_vdm_size]),
        Arc::from([pdm_projection(full_vdm_size)]),
    )
    .unwrap()
}

fn stroke_plan(pdm_size: usize) -> PdmStrokeOptimizationPlan<f32> {
    PdmStrokeOptimizationPlan::new(
        pdm_size,
        1,
        Arc::from(vec![0.0; pdm_size]),
        Arc::from(vec![1.0; pdm_size]),
        Arc::from(vec![1.0; pdm_size]),
    )
    .unwrap()
}

fn composite_plan(full_vdm_size: usize) -> DeformableMirrorCommandPlan<f32> {
    DeformableMirrorCommandPlan::new(
        1,
        full_vdm_size,
        1,
        Arc::from(vec![1.0; full_vdm_size]),
        Arc::from(vec![1.0; full_vdm_size]),
        Arc::from(vec![1.0; full_vdm_size]),
        Arc::from(vec![1.0; full_vdm_size]),
        Arc::from([-1.0]),
        Arc::from([1.0]),
        None,
    )
    .unwrap()
}

fn schedule(mode_count: usize, frames_per_period: usize) -> OpticalGainDitherSchedule<f32> {
    OpticalGainDitherSchedule::new(
        vec![OpticalGainDitherMode::new(1.0, frames_per_period, 1); mode_count].into_boxed_slice(),
    )
    .unwrap()
}

#[test]
fn vector_algorithms_reject_incompatible_workspace_extents() {
    let correction = ClosedLoopCorrectionPlan::new(1.0_f32, 0.0, 0.0, 2).unwrap();
    let mut correction_workspace = ClosedLoopCorrectionPlan::new(1.0_f32, 0.0, 0.0, 1)
        .unwrap()
        .workspace();
    assert_eq!(
        correction.process(
            &mut [9.0; 2],
            &mut correction_workspace,
            ClosedLoopCorrectionInput {
                x: &[0.0; 2],
                constraint_feedback: None,
            },
        ),
        Err(ClosedLoopCorrectionError::WorkspaceExtent)
    );

    let controller = controller_plan(2);
    let mut controller_workspace = controller_plan(1).workspace();
    assert_eq!(
        controller.process(&mut [9.0; 2], &mut controller_workspace, &[0.0]),
        Err(ControllerToVdmError::WorkspaceExtent)
    );

    let vdm_to_pdm = vdm_to_pdm_plan(2);
    let mut vdm_to_pdm_workspace = vdm_to_pdm_plan(1).workspace();
    assert_eq!(
        vdm_to_pdm.process(&mut [9.0], &mut vdm_to_pdm_workspace, &[0.0]),
        Err(VdmToPdmError::WorkspaceExtent)
    );

    let feedback = pdm_feedback_plan(2);
    let mut feedback_workspace = pdm_feedback_plan(1).workspace();
    assert_eq!(
        feedback.process(&mut [9.0], &mut feedback_workspace, &[0.0]),
        Err(PdmFeedbackToVdmError::WorkspaceExtent)
    );

    let stroke = stroke_plan(2);
    let mut stroke_workspace = stroke_plan(1).workspace();
    assert_eq!(
        stroke.process(&mut [9.0; 2], &mut stroke_workspace, &[0.0; 2]),
        Err(PdmStrokeOptimizationError::WorkspaceExtent)
    );
}

#[test]
fn command_plans_reject_incompatible_workspaces_before_state_changes() {
    let command = PdmCommandPlan::new(Arc::from([-1.0; 2]), Arc::from([1.0; 2]), None).unwrap();
    let mut command_workspace = PdmCommandPlan::new(Arc::from([-1.0]), Arc::from([1.0]), None)
        .unwrap()
        .workspace(&[0.0])
        .unwrap();
    assert_eq!(
        command.process(
            PdmCommandOutput {
                demanded: &mut [9.0; 2],
                constraint_feedback: &mut [9.0; 2],
            },
            &mut command_workspace,
            PdmCommandInput {
                requested: &[0.0; 2],
            },
        ),
        Err(PdmCommandError::WorkspaceExtent)
    );
    assert_eq!(command_workspace.current(), &[0.0]);

    let optimized_command = PdmCommandPlan::new(Arc::from([-1.0]), Arc::from([1.0]), None)
        .unwrap()
        .with_stroke_optimization(stroke_plan(1))
        .unwrap();
    let mut unoptimized_workspace = PdmCommandPlan::new(Arc::from([-1.0]), Arc::from([1.0]), None)
        .unwrap()
        .workspace(&[0.0])
        .unwrap();
    assert_eq!(
        optimized_command.process(
            PdmCommandOutput {
                demanded: &mut [9.0],
                constraint_feedback: &mut [9.0],
            },
            &mut unoptimized_workspace,
            PdmCommandInput { requested: &[0.0] },
        ),
        Err(PdmCommandError::WorkspaceExtent)
    );

    let composite = composite_plan(2);
    let mut composite_workspace = composite_plan(1).workspace(&[0.0]).unwrap();
    assert_eq!(
        composite.process(
            DeformableMirrorCommandOutput {
                requested: &mut [9.0],
                demanded: &mut [9.0],
                vdm_constraint_feedback: &mut [9.0],
            },
            &mut composite_workspace,
            DeformableMirrorCommandInput {
                vdm_command: &[0.0],
                best_flat: &[0.0],
                pdm_offset: &[0.0],
            },
        ),
        Err(DeformableMirrorCommandError::WorkspaceExtent)
    );
    assert_eq!(composite_workspace.current(), &[0.0]);
}

#[test]
fn optical_gain_dither_rejects_incompatible_workspace_history() {
    let dither = OpticalGainDitherPlan::new(schedule(2, 4));
    let mut dither_workspace = OpticalGainDitherPlan::new(schedule(1, 4)).workspace();
    let mut dither_output = [9.0; 2];
    assert_eq!(
        dither.process(&mut dither_output, &mut dither_workspace, ()),
        Err(OpticalGainDitherError::WorkspaceExtent)
    );
    assert_eq!(dither_output, [9.0; 2]);

    let long_dither = OpticalGainDitherPlan::new(schedule(1, 8));
    let mut phase_workspace = long_dither.workspace();
    for _ in 0..5 {
        long_dither
            .process(&mut [0.0], &mut phase_workspace, ())
            .unwrap();
    }
    let short_dither = OpticalGainDitherPlan::new(schedule(1, 4));
    assert_eq!(
        short_dither.process(&mut [9.0], &mut phase_workspace, ()),
        Err(OpticalGainDitherError::WorkspaceState)
    );
}

#[test]
fn optical_gain_estimator_rejects_incompatible_workspace_history() {
    let estimator = OpticalGainEstimatorPlan::new(schedule(2, 4), [1.0; 2]).unwrap();
    let mut estimator_workspace = OpticalGainEstimatorPlan::new(schedule(1, 4), [1.0])
        .unwrap()
        .workspace();
    let mut command_amplitudes = [9.0; 2];
    let mut response_amplitudes = [9.0; 2];
    let mut gains = [9.0; 2];
    let mut lags = [9.0; 2];
    let mut valid = [true; 2];
    let mut complete = true;
    assert_eq!(
        estimator.process(
            OpticalGainEstimateOutput::new(
                &mut command_amplitudes,
                &mut response_amplitudes,
                &mut gains,
                &mut lags,
                &mut valid,
                &mut complete,
            ),
            &mut estimator_workspace,
            OpticalGainEstimatorInput::new(&[0.0; 2], &[0.0; 2]),
        ),
        Err(OpticalGainEstimatorError::WorkspaceExtent)
    );
    assert_eq!(command_amplitudes, [9.0; 2]);
    assert!(complete);

    let long_estimator = OpticalGainEstimatorPlan::new(schedule(1, 8), [1.0]).unwrap();
    let mut frame_workspace = long_estimator.workspace();
    let mut command_amplitude = [0.0];
    let mut response_amplitude = [0.0];
    let mut gain = [0.0];
    let mut lag = [0.0];
    let mut frame_valid = [false];
    let mut frame_complete = false;
    for _ in 0..5 {
        long_estimator
            .process(
                OpticalGainEstimateOutput::new(
                    &mut command_amplitude,
                    &mut response_amplitude,
                    &mut gain,
                    &mut lag,
                    &mut frame_valid,
                    &mut frame_complete,
                ),
                &mut frame_workspace,
                OpticalGainEstimatorInput::new(&[0.0], &[0.0]),
            )
            .unwrap();
    }
    let short_estimator = OpticalGainEstimatorPlan::new(schedule(1, 4), [1.0]).unwrap();
    assert_eq!(
        short_estimator.process(
            OpticalGainEstimateOutput::new(
                &mut command_amplitude,
                &mut response_amplitude,
                &mut gain,
                &mut lag,
                &mut frame_valid,
                &mut frame_complete,
            ),
            &mut frame_workspace,
            OpticalGainEstimatorInput::new(&[0.0], &[0.0]),
        ),
        Err(OpticalGainEstimatorError::WorkspaceState)
    );
}
