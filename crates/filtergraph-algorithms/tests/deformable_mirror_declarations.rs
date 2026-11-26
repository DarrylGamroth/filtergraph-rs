//! Portable deformable-mirror declarations.

use std::sync::Arc;

use serde_json::{Value, json};

use filtergraph_algorithms::schemas::{
    ACTIVE_TO_FULL_VDM_PROJECTION_V1, CONTROLLER_COMMAND_V1, CONTROLLER_CONSTRAINT_FEEDBACK_V1,
    CONTROLLER_TO_VDM_PROJECTION_V1, FULL_TO_ACTIVE_VDM_PROJECTION_V1, PDM_CONSTRAINT_FEEDBACK_V1,
    PDM_TO_VDM_PROJECTION_V1, REQUESTED_PDM_COMMAND_V1, VDM_COMMAND_V1, VDM_CONSTRAINT_FEEDBACK_V1,
    VDM_TO_CONTROLLER_PROJECTION_V1, VDM_TO_PDM_PROJECTION_V1,
};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, ControllerProjection, ControllerToVdmF32,
    ControllerToVdmF32Configuration, ControllerToVdmPlan, DeclarationParameterError, PdmCommandF32,
    PdmFeedbackToVdmF32, PdmFeedbackToVdmF32Configuration, PdmFeedbackToVdmPlan, PdmProjection,
    PdmProjectionArtifacts, PdmProjectionError, PortDirection, PortRole, PropertyPolicy,
    VdmFeedbackToControllerF32, VdmFeedbackToControllerF32Configuration,
    VdmFeedbackToControllerPlan, VdmToPdmF32, VdmToPdmF32Configuration, VdmToPdmPlan,
    VectorConditioningF32,
};

const CONTROLLER_SIZE: usize = 2;
const ACTIVE_VDM_SIZE: usize = 2;
const FULL_VDM_SIZE: usize = 3;
const PDM_SIZE: usize = 2;

const CONTROLLER_TO_VDM: [f32; 4] = [1.0, 0.5, -0.25, 2.0];
const VDM_TO_CONTROLLER: [f32; 4] = [0.75, -0.5, 0.25, 1.25];
const ACTIVE_TO_FULL: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 0.5, -1.0];
const VDM_TO_PDM: [f32; 6] = [1.0, 0.0, 0.2, 0.0, 1.0, -0.3];
const PDM_TO_VDM: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 0.25, -0.5];
const FULL_TO_ACTIVE: [f32; 6] = [1.0, 0.0, 0.25, 0.0, 1.0, -0.5];

#[test]
fn command_declarations_reject_graph_dac_shaping() {
    let pdm_config = json!({
        "actuator_count": 2,
        "command_limits": [-1.0, 1.0],
        "slew_limit": null,
        "initial_current": null,
    });
    let vector_config = json!({
        "count": 2,
        "requested_schema": "org.example.vector/1",
        "demanded_schema": "org.example.vector/1",
        "feedback_schema": "org.example.vector/1",
        "command_limits": [-1.0, 1.0],
        "slew_limit": null,
        "initial_current": null,
    });

    for (origin, step) in [
        (Some(0.0), None),
        (None, Some(0.25)),
        (Some(0.0), Some(0.25)),
    ] {
        let with_grid = |mut config: Value| {
            config["quantization_origin"] = json!(origin);
            config["quantization_step"] = json!(step);
            config["dither_seed"] = json!(u64::MAX);
            config
        };
        let pdm_error =
            PdmCommandF32::prepare(serde_json::from_value(with_grid(pdm_config.clone())).unwrap())
                .err()
                .expect("PDM command must reject DAC shaping");
        let vector_error = VectorConditioningF32::prepare(
            serde_json::from_value(with_grid(vector_config.clone())).unwrap(),
        )
        .err()
        .expect("vector conditioning must reject DAC shaping");
        for message in [pdm_error.to_string(), vector_error.to_string()] {
            assert!(
                message.contains("Configure DAC quantization and dither in the device driver"),
                "{message}",
            );
        }
    }
}

#[test]
fn declarations_expose_coordinate_boundaries_and_sparse_artifacts_once() {
    let cases = [
        (
            ControllerToVdmF32::NAME,
            ControllerToVdmF32::PORTS,
            &[
                ("controller-command", PortDirection::Input, PortRole::Data),
                ("vdm-command", PortDirection::Output, PortRole::Data),
                (
                    "controller-to-vdm",
                    PortDirection::Input,
                    PortRole::Parameter,
                ),
            ][..],
        ),
        (
            VdmToPdmF32::NAME,
            VdmToPdmF32::PORTS,
            &[
                ("vdm-command", PortDirection::Input, PortRole::Data),
                (
                    "requested-pdm-command",
                    PortDirection::Output,
                    PortRole::Data,
                ),
                ("correction-role", PortDirection::Output, PortRole::Data),
                ("active-to-full", PortDirection::Input, PortRole::Parameter),
                ("vdm-to-pdm", PortDirection::Input, PortRole::Parameter),
            ][..],
        ),
        (
            PdmFeedbackToVdmF32::NAME,
            PdmFeedbackToVdmF32::PORTS,
            &[
                (
                    "pdm-constraint-feedback",
                    PortDirection::Input,
                    PortRole::Data,
                ),
                (
                    "vdm-constraint-feedback",
                    PortDirection::Output,
                    PortRole::Data,
                ),
                ("full-to-active", PortDirection::Input, PortRole::Parameter),
                ("pdm-to-vdm", PortDirection::Input, PortRole::Parameter),
            ][..],
        ),
        (
            VdmFeedbackToControllerF32::NAME,
            VdmFeedbackToControllerF32::PORTS,
            &[
                (
                    "vdm-constraint-feedback",
                    PortDirection::Input,
                    PortRole::Data,
                ),
                (
                    "controller-constraint-feedback",
                    PortDirection::Output,
                    PortRole::Data,
                ),
                (
                    "vdm-to-controller",
                    PortDirection::Input,
                    PortRole::Parameter,
                ),
            ][..],
        ),
    ];

    assert_eq!(cases[0].0, "controller-to-vdm-f32");
    assert_eq!(cases[1].0, "vdm-to-pdm-f32");
    assert_eq!(cases[2].0, "pdm-feedback-to-vdm-f32");
    assert_eq!(cases[3].0, "vdm-feedback-to-controller-f32");
    for (_, ports, expected) in cases {
        assert_eq!(ports.len(), expected.len());
        for (port, &(name, direction, role)) in ports.iter().zip(expected) {
            assert_eq!(port.name(), name);
            assert_eq!(port.direction(), direction);
            assert_eq!(port.role(), role);
        }
    }
    assert_eq!(ControllerToVdmF32::PROPERTIES, PropertyPolicy::None);
    assert_eq!(VdmToPdmF32::PROPERTIES, PropertyPolicy::None);
    assert_eq!(PdmFeedbackToVdmF32::PROPERTIES, PropertyPolicy::None);
    assert_eq!(VdmFeedbackToControllerF32::PROPERTIES, PropertyPolicy::None);
}

#[test]
fn native_declarations_match_the_direct_projection_chain() {
    let controller_projection = ControllerProjection::new(
        CONTROLLER_SIZE,
        Arc::from(CONTROLLER_TO_VDM),
        Arc::from(VDM_TO_CONTROLLER),
    )
    .unwrap();
    let pdm_projection =
        PdmProjection::new(PDM_SIZE, Arc::from(VDM_TO_PDM), Arc::from(PDM_TO_VDM)).unwrap();

    let controller_to_vdm =
        ControllerToVdmPlan::new(ACTIVE_VDM_SIZE, Arc::from([controller_projection.clone()]))
            .unwrap();
    let vdm_to_pdm = VdmToPdmPlan::new(
        ACTIVE_VDM_SIZE,
        FULL_VDM_SIZE,
        Arc::from(ACTIVE_TO_FULL),
        Arc::from([pdm_projection.clone()]),
    )
    .unwrap();
    let pdm_feedback_to_vdm = PdmFeedbackToVdmPlan::new(
        ACTIVE_VDM_SIZE,
        FULL_VDM_SIZE,
        Arc::from(FULL_TO_ACTIVE),
        Arc::from([pdm_projection]),
    )
    .unwrap();
    let vdm_feedback_to_controller =
        VdmFeedbackToControllerPlan::new(ACTIVE_VDM_SIZE, Arc::from([controller_projection]))
            .unwrap();

    let mut declared_controller_to_vdm =
        ControllerToVdmF32::prepare(ControllerToVdmF32Configuration {
            controller_command_count: CONTROLLER_SIZE,
            active_vdm_count: ACTIVE_VDM_SIZE,
        })
        .unwrap();
    let mut declared_vdm_to_pdm = VdmToPdmF32::prepare(VdmToPdmF32Configuration {
        active_vdm_count: ACTIVE_VDM_SIZE,
        full_vdm_count: FULL_VDM_SIZE,
        physical_actuator_count: PDM_SIZE,
    })
    .unwrap();
    let mut declared_pdm_feedback_to_vdm =
        PdmFeedbackToVdmF32::prepare(PdmFeedbackToVdmF32Configuration {
            active_vdm_count: ACTIVE_VDM_SIZE,
            full_vdm_count: FULL_VDM_SIZE,
            physical_actuator_count: PDM_SIZE,
        })
        .unwrap();
    let mut declared_vdm_feedback_to_controller =
        VdmFeedbackToControllerF32::prepare(VdmFeedbackToControllerF32Configuration {
            controller_command_count: CONTROLLER_SIZE,
            active_vdm_count: ACTIVE_VDM_SIZE,
        })
        .unwrap();

    assert_eq!(
        declared_controller_to_vdm
            .replace_parameter(2, AlgorithmInput::from(&CONTROLLER_TO_VDM[..]),),
        Some(Ok(()))
    );
    assert_eq!(
        declared_vdm_to_pdm.replace_parameter(3, AlgorithmInput::from(&ACTIVE_TO_FULL[..])),
        Some(Ok(()))
    );
    assert_eq!(
        declared_vdm_to_pdm.replace_parameter(4, AlgorithmInput::from(&VDM_TO_PDM[..])),
        Some(Ok(()))
    );
    assert_eq!(
        declared_pdm_feedback_to_vdm
            .replace_parameter(2, AlgorithmInput::from(&FULL_TO_ACTIVE[..])),
        Some(Ok(()))
    );
    assert_eq!(
        declared_pdm_feedback_to_vdm.replace_parameter(3, AlgorithmInput::from(&PDM_TO_VDM[..])),
        Some(Ok(()))
    );
    assert_eq!(
        declared_vdm_feedback_to_controller
            .replace_parameter(2, AlgorithmInput::from(&VDM_TO_CONTROLLER[..]),),
        Some(Ok(()))
    );
    assert_eq!(
        declared_controller_to_vdm
            .plan()
            .controller_to_vdm_projections()[0]
            .controller_to_vdm(),
        CONTROLLER_TO_VDM
    );
    assert_eq!(
        declared_vdm_feedback_to_controller
            .plan()
            .vdm_to_controller_projections()[0]
            .vdm_to_controller(),
        VDM_TO_CONTROLLER
    );
    assert_eq!(
        declared_vdm_to_pdm.plan().vdm_to_pdm_projections()[0].artifacts(),
        PdmProjectionArtifacts::RowMajor(&VDM_TO_PDM)
    );
    assert_eq!(
        declared_pdm_feedback_to_vdm.plan().pdm_to_vdm_projections()[0].artifacts(),
        PdmProjectionArtifacts::RowMajor(&PDM_TO_VDM)
    );

    let controller_command = [0.6_f32, -1.2];
    let pdm_feedback = [0.4_f32, -0.75];
    let mut expected_vdm_command = [0.0_f32; ACTIVE_VDM_SIZE];
    let mut actual_vdm_command = [0.0_f32; ACTIVE_VDM_SIZE];
    let mut expected_pdm_command = [0.0_f32; PDM_SIZE];
    let mut actual_pdm_command = [0.0_f32; PDM_SIZE];
    let mut expected_vdm_feedback = [0.0_f32; ACTIVE_VDM_SIZE];
    let mut actual_vdm_feedback = [0.0_f32; ACTIVE_VDM_SIZE];
    let mut expected_controller_feedback = [0.0_f32; CONTROLLER_SIZE];
    let mut actual_controller_feedback = [0.0_f32; CONTROLLER_SIZE];

    controller_to_vdm
        .process(
            &mut expected_vdm_command,
            &mut controller_to_vdm.workspace(),
            &controller_command,
        )
        .unwrap();
    ControllerToVdmF32::process(
        &mut declared_controller_to_vdm,
        &controller_command,
        &mut actual_vdm_command,
    )
    .unwrap();
    assert_eq!(actual_vdm_command, expected_vdm_command);

    vdm_to_pdm
        .process(
            &mut expected_pdm_command,
            &mut vdm_to_pdm.workspace(),
            &expected_vdm_command,
        )
        .unwrap();
    VdmToPdmF32::process(
        &mut declared_vdm_to_pdm,
        &actual_vdm_command,
        &mut actual_pdm_command,
        &mut [0u8; 2],
    )
    .unwrap();
    assert_eq!(actual_pdm_command, expected_pdm_command);

    pdm_feedback_to_vdm
        .process(
            &mut expected_vdm_feedback,
            &mut pdm_feedback_to_vdm.workspace(),
            &pdm_feedback,
        )
        .unwrap();
    PdmFeedbackToVdmF32::process(
        &mut declared_pdm_feedback_to_vdm,
        &pdm_feedback,
        &mut actual_vdm_feedback,
    )
    .unwrap();
    assert_eq!(actual_vdm_feedback, expected_vdm_feedback);

    vdm_feedback_to_controller
        .process(
            &mut expected_controller_feedback,
            &mut filtergraph_algorithms::NoWorkspace,
            &expected_vdm_feedback,
        )
        .unwrap();
    VdmFeedbackToControllerF32::process(
        &mut declared_vdm_feedback_to_controller,
        &actual_vdm_feedback,
        &mut actual_controller_feedback,
    )
    .unwrap();
    assert_eq!(actual_controller_feedback, expected_controller_feedback);
}

#[test]
fn concatenated_matrices_preserve_multiple_controller_and_pdm_blocks() {
    let controller_a =
        ControllerProjection::new(1, Arc::from([1.0_f32, 2.0]), Arc::from([0.25_f32, 0.5]))
            .unwrap();
    let controller_b =
        ControllerProjection::new(1, Arc::from([-1.0_f32, 0.5]), Arc::from([-0.75_f32, 1.0]))
            .unwrap();
    let direct_forward =
        ControllerToVdmPlan::new(2, Arc::from([controller_a.clone(), controller_b.clone()]))
            .unwrap();
    let direct_reverse =
        VdmFeedbackToControllerPlan::new(2, Arc::from([controller_a, controller_b])).unwrap();
    let mut declared_forward = ControllerToVdmF32::prepare(ControllerToVdmF32Configuration {
        controller_command_count: 2,
        active_vdm_count: 2,
    })
    .unwrap();
    let mut declared_reverse =
        VdmFeedbackToControllerF32::prepare(VdmFeedbackToControllerF32Configuration {
            controller_command_count: 2,
            active_vdm_count: 2,
        })
        .unwrap();
    declared_forward
        .replace_parameter(2, AlgorithmInput::from(&[1.0_f32, -1.0, 2.0, 0.5][..]))
        .unwrap()
        .unwrap();
    declared_reverse
        .replace_parameter(2, AlgorithmInput::from(&[0.25_f32, 0.5, -0.75, 1.0][..]))
        .unwrap()
        .unwrap();
    let controller_commands = [0.4_f32, -0.6];
    let vdm_feedback = [0.3_f32, -0.2];
    let mut direct_vdm = [0.0_f32; 2];
    let mut declared_vdm = [0.0_f32; 2];
    let mut direct_controller_feedback = [0.0_f32; 2];
    let mut declared_controller_feedback = [0.0_f32; 2];
    direct_forward
        .process(
            &mut direct_vdm,
            &mut direct_forward.workspace(),
            &controller_commands,
        )
        .unwrap();
    ControllerToVdmF32::process(
        &mut declared_forward,
        &controller_commands,
        &mut declared_vdm,
    )
    .unwrap();
    direct_reverse
        .process(
            &mut direct_controller_feedback,
            &mut filtergraph_algorithms::NoWorkspace,
            &vdm_feedback,
        )
        .unwrap();
    VdmFeedbackToControllerF32::process(
        &mut declared_reverse,
        &vdm_feedback,
        &mut declared_controller_feedback,
    )
    .unwrap();
    assert_eq!(declared_vdm, direct_vdm);
    assert_eq!(declared_controller_feedback, direct_controller_feedback);

    let pdm_a =
        PdmProjection::new(1, Arc::from([1.0_f32, 0.5]), Arc::from([0.1_f32, 0.2])).unwrap();
    let pdm_b =
        PdmProjection::new(1, Arc::from([-0.25_f32, 2.0]), Arc::from([-0.5_f32, 1.5])).unwrap();
    let direct_pdm_forward = VdmToPdmPlan::new(
        2,
        2,
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([pdm_a.clone(), pdm_b.clone()]),
    )
    .unwrap();
    let direct_pdm_reverse = PdmFeedbackToVdmPlan::new(
        2,
        2,
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([pdm_a, pdm_b]),
    )
    .unwrap();
    let mut declared_pdm_forward = VdmToPdmF32::prepare(VdmToPdmF32Configuration {
        active_vdm_count: 2,
        full_vdm_count: 2,
        physical_actuator_count: 2,
    })
    .unwrap();
    let mut declared_pdm_reverse = PdmFeedbackToVdmF32::prepare(PdmFeedbackToVdmF32Configuration {
        active_vdm_count: 2,
        full_vdm_count: 2,
        physical_actuator_count: 2,
    })
    .unwrap();
    let identity = [1.0_f32, 0.0, 0.0, 1.0];
    declared_pdm_forward
        .replace_parameter(3, AlgorithmInput::from(&identity[..]))
        .unwrap()
        .unwrap();
    declared_pdm_forward
        .replace_parameter(4, AlgorithmInput::from(&[1.0_f32, 0.5, -0.25, 2.0][..]))
        .unwrap()
        .unwrap();
    declared_pdm_reverse
        .replace_parameter(2, AlgorithmInput::from(&identity[..]))
        .unwrap()
        .unwrap();
    declared_pdm_reverse
        .replace_parameter(3, AlgorithmInput::from(&[0.1_f32, -0.5, 0.2, 1.5][..]))
        .unwrap()
        .unwrap();
    let vdm_command = [0.75_f32, -0.4];
    let pdm_feedback = [0.2_f32, -0.3];
    let mut direct_pdm_command = [0.0_f32; 2];
    let mut declared_pdm_command = [0.0_f32; 2];
    let mut direct_active_feedback = [0.0_f32; 2];
    let mut declared_active_feedback = [0.0_f32; 2];
    direct_pdm_forward
        .process(
            &mut direct_pdm_command,
            &mut direct_pdm_forward.workspace(),
            &vdm_command,
        )
        .unwrap();
    VdmToPdmF32::process(
        &mut declared_pdm_forward,
        &vdm_command,
        &mut declared_pdm_command,
        &mut [0u8; 2],
    )
    .unwrap();
    direct_pdm_reverse
        .process(
            &mut direct_active_feedback,
            &mut direct_pdm_reverse.workspace(),
            &pdm_feedback,
        )
        .unwrap();
    PdmFeedbackToVdmF32::process(
        &mut declared_pdm_reverse,
        &pdm_feedback,
        &mut declared_active_feedback,
    )
    .unwrap();
    assert_eq!(declared_pdm_command, direct_pdm_command);
    assert_eq!(declared_active_feedback, direct_active_feedback);
}

#[test]
fn formats_preserve_coordinate_schemas_and_matrix_axes() {
    let controller_to_vdm = ControllerToVdmF32::prepare(ControllerToVdmF32Configuration {
        controller_command_count: CONTROLLER_SIZE,
        active_vdm_count: ACTIVE_VDM_SIZE,
    })
    .unwrap();
    assert_eq!(
        controller_to_vdm.formats()[0].schema(),
        CONTROLLER_COMMAND_V1
    );
    assert_eq!(controller_to_vdm.formats()[1].schema(), VDM_COMMAND_V1);
    assert_eq!(
        controller_to_vdm.formats()[2].schema(),
        CONTROLLER_TO_VDM_PROJECTION_V1
    );
    assert_eq!(controller_to_vdm.formats()[2].shape(), &[2, 2]);

    let vdm_to_pdm = VdmToPdmF32::prepare(VdmToPdmF32Configuration {
        active_vdm_count: ACTIVE_VDM_SIZE,
        full_vdm_count: FULL_VDM_SIZE,
        physical_actuator_count: PDM_SIZE,
    })
    .unwrap();
    assert_eq!(vdm_to_pdm.formats()[0].schema(), VDM_COMMAND_V1);
    assert_eq!(vdm_to_pdm.formats()[1].schema(), REQUESTED_PDM_COMMAND_V1);
    assert_eq!(
        vdm_to_pdm.formats()[3].schema(),
        ACTIVE_TO_FULL_VDM_PROJECTION_V1
    );
    assert_eq!(vdm_to_pdm.formats()[3].shape(), &[3, 2]);
    assert_eq!(vdm_to_pdm.formats()[4].schema(), VDM_TO_PDM_PROJECTION_V1);
    assert_eq!(vdm_to_pdm.formats()[4].shape(), &[2, 3]);

    let pdm_feedback_to_vdm = PdmFeedbackToVdmF32::prepare(PdmFeedbackToVdmF32Configuration {
        active_vdm_count: ACTIVE_VDM_SIZE,
        full_vdm_count: FULL_VDM_SIZE,
        physical_actuator_count: PDM_SIZE,
    })
    .unwrap();
    assert_eq!(
        pdm_feedback_to_vdm.formats()[0].schema(),
        PDM_CONSTRAINT_FEEDBACK_V1
    );
    assert_eq!(
        pdm_feedback_to_vdm.formats()[1].schema(),
        VDM_CONSTRAINT_FEEDBACK_V1
    );
    assert_eq!(
        pdm_feedback_to_vdm.formats()[2].schema(),
        FULL_TO_ACTIVE_VDM_PROJECTION_V1
    );
    assert_eq!(pdm_feedback_to_vdm.formats()[2].shape(), &[2, 3]);
    assert_eq!(
        pdm_feedback_to_vdm.formats()[3].schema(),
        PDM_TO_VDM_PROJECTION_V1
    );
    assert_eq!(pdm_feedback_to_vdm.formats()[3].shape(), &[3, 2]);

    let vdm_feedback_to_controller =
        VdmFeedbackToControllerF32::prepare(VdmFeedbackToControllerF32Configuration {
            controller_command_count: CONTROLLER_SIZE,
            active_vdm_count: ACTIVE_VDM_SIZE,
        })
        .unwrap();
    assert_eq!(
        vdm_feedback_to_controller.formats()[0].schema(),
        VDM_CONSTRAINT_FEEDBACK_V1
    );
    assert_eq!(
        vdm_feedback_to_controller.formats()[1].schema(),
        CONTROLLER_CONSTRAINT_FEEDBACK_V1
    );
    assert_eq!(
        vdm_feedback_to_controller.formats()[2].schema(),
        VDM_TO_CONTROLLER_PROJECTION_V1
    );
}

#[test]
fn rejected_updates_preserve_every_other_prepared_projection() {
    let mut declared = VdmToPdmF32::prepare(VdmToPdmF32Configuration {
        active_vdm_count: ACTIVE_VDM_SIZE,
        full_vdm_count: FULL_VDM_SIZE,
        physical_actuator_count: PDM_SIZE,
    })
    .unwrap();
    declared
        .replace_parameter(3, AlgorithmInput::from(&ACTIVE_TO_FULL[..]))
        .unwrap()
        .unwrap();
    declared
        .replace_parameter(4, AlgorithmInput::from(&VDM_TO_PDM[..]))
        .unwrap()
        .unwrap();
    let prepared_plan = declared.plan().clone();

    assert_eq!(
        declared.replace_parameter(3, AlgorithmInput::from(&[1.0_f32][..])),
        Some(Err(DeclarationParameterError::Algorithm(
            PdmProjectionError::ProjectionExtent
        )))
    );
    assert_eq!(declared.plan(), &prepared_plan);

    let replacement_active_to_full = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    declared
        .replace_parameter(3, AlgorithmInput::from(&replacement_active_to_full[..]))
        .unwrap()
        .unwrap();
    assert_eq!(
        declared.plan().vdm_to_pdm_projections(),
        prepared_plan.vdm_to_pdm_projections()
    );
}

#[test]
fn declarations_reject_zero_and_overflowing_topologies() {
    assert!(
        ControllerToVdmF32::prepare(ControllerToVdmF32Configuration {
            controller_command_count: 0,
            active_vdm_count: 2,
        })
        .is_err()
    );
    assert!(
        VdmToPdmF32::prepare(VdmToPdmF32Configuration {
            active_vdm_count: usize::MAX,
            full_vdm_count: 2,
            physical_actuator_count: 1,
        })
        .is_err()
    );
    assert!(
        PdmFeedbackToVdmF32::prepare(PdmFeedbackToVdmF32Configuration {
            active_vdm_count: 1,
            full_vdm_count: 0,
            physical_actuator_count: 1,
        })
        .is_err()
    );
    assert!(
        VdmFeedbackToControllerF32::prepare(VdmFeedbackToControllerF32Configuration {
            controller_command_count: 1,
            active_vdm_count: usize::MAX,
        })
        .is_err()
    );
}
