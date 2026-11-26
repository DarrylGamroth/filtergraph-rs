//! Portable closed-loop correction declaration and atomic-state tests.

use std::sync::Arc;

use filtergraph_algorithms::schemas::{
    CONTROLLER_COMMAND_V1, CONTROLLER_CONSTRAINT_FEEDBACK_V1, CONTROLLER_RESIDUAL_ERROR_V1,
    CONTROLLER_STATE_V1, HIDDEN_TO_SHAPE_PROJECTION_V1, SHAPE_TO_HIDDEN_PROJECTION_V1,
};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, ArrayLayout, ClosedLoopCorrectionF32,
    ClosedLoopCorrectionF32Configuration, ClosedLoopCorrectionInput, ClosedLoopCorrectionPlan,
    ClosedLoopCorrectionPlanError, ConfigurationUpdate, DeclarationParameterError, ElementType,
    PortDirection, PortRole, PropertyPolicy,
};

#[test]
fn declaration_exposes_the_scientific_roles_once() {
    assert_eq!(ClosedLoopCorrectionF32::NAME, "closed-loop-correction-f32");
    assert_eq!(ClosedLoopCorrectionF32::PROPERTIES, PropertyPolicy::Runtime);
    assert_eq!(ClosedLoopCorrectionF32::CONFIGURATION.len(), 2);
    assert_eq!(
        ClosedLoopCorrectionF32::CONFIGURATION[0].name(),
        "state_length"
    );
    assert_eq!(
        ClosedLoopCorrectionF32::CONFIGURATION[0].update(),
        ConfigurationUpdate::GraphRebuild
    );
    assert_eq!(
        ClosedLoopCorrectionF32::CONFIGURATION[1].name(),
        "hidden_mode_count"
    );
    assert_eq!(
        ClosedLoopCorrectionF32::CONFIGURATION[1].update(),
        ConfigurationUpdate::GraphRebuild
    );

    let ports = ClosedLoopCorrectionF32::PORTS;
    assert_eq!(ports.len(), 6);
    assert_eq!(
        ports.iter().map(|port| port.name()).collect::<Vec<_>>(),
        [
            "residual-error",
            "constraint-feedback",
            "correction",
            "controller-state",
            "shape-to-hidden",
            "hidden-to-shape",
        ]
    );
    assert_eq!(ports[0].direction(), PortDirection::Input);
    assert_eq!(ports[0].role(), PortRole::Data);
    assert_eq!(ports[1].direction(), PortDirection::Input);
    assert_eq!(ports[1].role(), PortRole::Data);
    assert_eq!(ports[2].direction(), PortDirection::Output);
    assert_eq!(ports[2].role(), PortRole::Data);
    assert_eq!(ports[2].metadata_source(), Some("residual-error"));
    assert_eq!(ports[3].direction(), PortDirection::Output);
    assert_eq!(ports[3].role(), PortRole::Data);
    assert_eq!(ports[3].metadata_source(), Some("residual-error"));
    assert_eq!(ports[4].direction(), PortDirection::Input);
    assert_eq!(ports[4].role(), PortRole::Parameter);
    assert_eq!(ports[5].direction(), PortDirection::Input);
    assert_eq!(ports[5].role(), PortRole::Parameter);
    assert!(
        ports
            .iter()
            .all(|port| port.element_type() == ElementType::F32)
    );
}

#[test]
fn native_declaration_matches_direct_atomic_state_transitions() {
    const STATE_LENGTH: usize = 3;
    const HIDDEN_MODE_COUNT: usize = 2;
    let shape_to_hidden = [1.0_f32, -0.5, 0.25, 0.0, 0.75, -1.0];
    let hidden_to_shape = [0.5_f32, -0.25, 0.75, 0.0, -0.5, 1.0];

    let mut declared = ClosedLoopCorrectionF32::prepare(ClosedLoopCorrectionF32Configuration {
        state_length: STATE_LENGTH,
        hidden_mode_count: HIDDEN_MODE_COUNT,
    })
    .unwrap();
    assert_eq!(declared.formats().len(), 6);
    assert_eq!(declared.formats()[0].shape(), &[STATE_LENGTH as u32]);
    assert_eq!(declared.formats()[1].shape(), &[STATE_LENGTH as u32]);
    assert_eq!(declared.formats()[2].shape(), &[STATE_LENGTH as u32]);
    assert_eq!(declared.formats()[3].shape(), &[STATE_LENGTH as u32]);
    assert_eq!(
        declared.formats()[4].shape(),
        &[HIDDEN_MODE_COUNT as u32, STATE_LENGTH as u32]
    );
    assert_eq!(
        declared.formats()[5].shape(),
        &[STATE_LENGTH as u32, HIDDEN_MODE_COUNT as u32]
    );
    assert_eq!(declared.formats()[0].schema(), CONTROLLER_RESIDUAL_ERROR_V1);
    assert_eq!(
        declared.formats()[1].schema(),
        CONTROLLER_CONSTRAINT_FEEDBACK_V1
    );
    assert_eq!(declared.formats()[2].schema(), CONTROLLER_COMMAND_V1);
    assert_eq!(declared.formats()[3].schema(), CONTROLLER_STATE_V1);
    assert_eq!(
        declared.formats()[4].schema(),
        SHAPE_TO_HIDDEN_PROJECTION_V1
    );
    assert_eq!(
        declared.formats()[5].schema(),
        HIDDEN_TO_SHAPE_PROJECTION_V1
    );
    assert!(
        declared
            .formats()
            .iter()
            .all(|format| format.layout() == ArrayLayout::RowMajor)
    );

    assert_eq!(
        declared.replace_parameter(4, AlgorithmInput::from(&shape_to_hidden[..])),
        Some(Ok(()))
    );
    assert_eq!(
        declared.replace_parameter(5, AlgorithmInput::from(&hidden_to_shape[..])),
        Some(Ok(()))
    );

    let direct = ClosedLoopCorrectionPlan::with_hidden_modes(
        1.0,
        0.0,
        0.0,
        STATE_LENGTH,
        HIDDEN_MODE_COUNT,
        0.0,
        Arc::from(shape_to_hidden),
        Arc::from(hidden_to_shape),
    )
    .unwrap();
    let mut direct_workspace = direct.workspace();
    let residuals = [[0.5_f32, -1.0, 0.25], [0.1, 0.2, -0.3], [-0.4, 0.6, 0.8]];
    let feedback = [[99.0_f32, 99.0, 99.0], [0.05, -0.1, 0.0], [0.0, 0.2, -0.25]];
    let mut expected_correction = [0.0_f32; STATE_LENGTH];
    let mut actual_correction = [0.0_f32; STATE_LENGTH];
    let mut controller_state = [0.0_f32; STATE_LENGTH];

    for frame in 0..residuals.len() {
        direct
            .process(
                &mut expected_correction,
                &mut direct_workspace,
                ClosedLoopCorrectionInput {
                    x: &residuals[frame],
                    constraint_feedback: (frame != 0).then_some(&feedback[frame][..]),
                },
            )
            .unwrap();
        ClosedLoopCorrectionF32::process(
            &mut declared,
            &residuals[frame],
            &feedback[frame],
            &mut actual_correction,
            &mut controller_state,
        )
        .unwrap();

        assert_eq!(actual_correction, expected_correction);
        assert_eq!(controller_state, direct_workspace.state());
        assert_eq!(declared.workspace().state(), direct_workspace.state());
        assert_eq!(
            declared.workspace().last_output(),
            direct_workspace.last_output()
        );
    }

    ClosedLoopCorrectionF32::reset(&mut declared);
    assert_eq!(declared.workspace().state(), [0.0; STATE_LENGTH]);
    assert_eq!(declared.workspace().last_output(), None);
}

#[test]
fn rejected_parameter_and_processing_updates_preserve_the_active_operation() {
    let mut declared = ClosedLoopCorrectionF32::prepare(ClosedLoopCorrectionF32Configuration {
        state_length: 2,
        hidden_mode_count: 1,
    })
    .unwrap();
    let original_plan = declared.plan().clone();
    assert_eq!(
        declared.replace_parameter(4, AlgorithmInput::from(&[1.0_f32][..])),
        Some(Err(DeclarationParameterError::Algorithm(
            ClosedLoopCorrectionPlanError::HiddenModeProjectionExtent
        )))
    );
    assert_eq!(declared.plan(), &original_plan);

    let mut correction = [0.0_f32; 2];
    let mut state = [0.0_f32; 2];
    ClosedLoopCorrectionF32::process(
        &mut declared,
        &[1.0, -1.0],
        &[123.0, 456.0],
        &mut correction,
        &mut state,
    )
    .unwrap();
    let committed_state = declared.workspace().state().to_vec();
    let committed_output = declared.workspace().last_output().unwrap().to_vec();
    let published_state = state;
    let published_correction = correction;

    assert!(
        ClosedLoopCorrectionF32::process(
            &mut declared,
            &[1.0, -1.0],
            &[f32::NAN, 0.0],
            &mut correction,
            &mut state,
        )
        .is_err()
    );
    assert_eq!(declared.workspace().state(), committed_state);
    assert_eq!(
        declared.workspace().last_output().unwrap(),
        committed_output
    );
    assert_eq!(correction, published_correction);
    assert_eq!(state, published_state);
}

#[test]
fn declaration_rejects_zero_and_overflowing_graph_dimensions() {
    assert!(
        ClosedLoopCorrectionF32::prepare(ClosedLoopCorrectionF32Configuration {
            state_length: 0,
            hidden_mode_count: 1,
        })
        .is_err()
    );
    assert!(
        ClosedLoopCorrectionF32::prepare(ClosedLoopCorrectionF32Configuration {
            state_length: 1,
            hidden_mode_count: 0,
        })
        .is_err()
    );
    assert!(
        ClosedLoopCorrectionF32::prepare(ClosedLoopCorrectionF32Configuration {
            state_length: usize::MAX,
            hidden_mode_count: 2,
        })
        .is_err()
    );
}
