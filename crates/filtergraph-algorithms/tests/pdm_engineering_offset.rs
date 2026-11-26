//! PDM engineering offset addition, adoption telemetry, and limiter composition.

use filtergraph_algorithms::schemas::{PDM_ENGINEERING_OFFSET_V1, REQUESTED_PDM_COMMAND_V1};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, DeclarationParameterError, NoWorkspace,
    PdmCommandInput, PdmCommandOutput, PdmCommandPlan, PdmEngineeringOffsetError,
    PdmEngineeringOffsetF32, PdmEngineeringOffsetF32Configuration, PdmEngineeringOffsetOutput,
    PdmEngineeringOffsetPlan, PdmEngineeringOffsetPlanError, PortDirection, PortRole,
};
use std::sync::Arc;

#[test]
fn declaration_reports_the_adopted_offset_in_physical_actuator_order() {
    assert_eq!(PdmEngineeringOffsetF32::NAME, "pdm-engineering-offset-f32");
    assert_eq!(PdmEngineeringOffsetF32::CONFIGURATION.len(), 1);
    assert_eq!(PdmEngineeringOffsetF32::PORTS.len(), 4);
    assert_eq!(PdmEngineeringOffsetF32::PORTS[0].name(), "requested");
    assert_eq!(
        PdmEngineeringOffsetF32::PORTS[1].name(),
        "requested-with-offset"
    );
    assert_eq!(PdmEngineeringOffsetF32::PORTS[2].name(), "current-offset");
    assert_eq!(
        PdmEngineeringOffsetF32::PORTS[3].name(),
        "engineering-offset"
    );
    assert_eq!(
        PdmEngineeringOffsetF32::PORTS[3].role(),
        PortRole::Parameter
    );
    assert_eq!(
        PdmEngineeringOffsetF32::PORTS[3].direction(),
        PortDirection::Input
    );
    for port in &PdmEngineeringOffsetF32::PORTS[1..3] {
        assert_eq!(port.direction(), PortDirection::Output);
        assert_eq!(port.metadata_source(), Some("requested"));
    }
    let mut declared = PdmEngineeringOffsetF32::prepare(PdmEngineeringOffsetF32Configuration {
        actuator_count: 3,
    })
    .unwrap();
    for (format, schema) in declared.formats().iter().zip([
        REQUESTED_PDM_COMMAND_V1,
        REQUESTED_PDM_COMMAND_V1,
        PDM_ENGINEERING_OFFSET_V1,
        PDM_ENGINEERING_OFFSET_V1,
    ]) {
        assert_eq!(format.shape(), &[3]);
        assert_eq!(format.schema(), schema);
    }
    let requested = [1.0_f32, 2.0, 3.0];
    let mut output = [0.0; 3];
    let mut current = [9.0; 3];
    PdmEngineeringOffsetF32::process(&mut declared, &requested, &mut output, &mut current).unwrap();
    assert_eq!(output, requested);
    assert_eq!(current, [0.0; 3]);
    for offset in [[0.5_f32, -2.0, 3.0], [-1.0_f32, 0.0, 1.0]] {
        assert_eq!(
            declared.replace_parameter(3, AlgorithmInput::from(&offset[..])),
            Some(Ok(()))
        );
        PdmEngineeringOffsetF32::process(&mut declared, &requested, &mut output, &mut current)
            .unwrap();
        assert_eq!(current, offset);
        for i in 0..3 {
            assert_eq!(output[i], requested[i] + offset[i]);
        }
    }
    for invalid in [
        [0.0, 0.0, f32::NAN],
        [0.0, 0.0, f32::INFINITY],
        [0.0, 0.0, f32::NEG_INFINITY],
    ] {
        assert_eq!(
            declared.replace_parameter(3, AlgorithmInput::from(&invalid[..])),
            Some(Err(DeclarationParameterError::Algorithm(
                PdmEngineeringOffsetPlanError::NonFiniteConfiguration
            )))
        );
        assert_eq!(declared.plan().engineering_offset(), &[-1.0, 0.0, 1.0]);
    }
    assert_eq!(
        declared.replace_parameter(3, AlgorithmInput::from(&[0.0_f32][..])),
        Some(Err(DeclarationParameterError::Algorithm(
            PdmEngineeringOffsetPlanError::Extent
        )))
    );
    assert!(
        PdmEngineeringOffsetF32::prepare(PdmEngineeringOffsetF32Configuration {
            actuator_count: 0
        })
        .is_err()
    );
}

#[test]
fn invalid_frames_leave_both_outputs_unchanged() {
    assert_eq!(
        PdmEngineeringOffsetPlan::<f32>::new(Arc::from([])),
        Err(PdmEngineeringOffsetPlanError::Empty)
    );
    assert_eq!(
        PdmEngineeringOffsetPlan::new(Arc::from([f32::NAN])),
        Err(PdmEngineeringOffsetPlanError::NonFiniteConfiguration)
    );
    let plan = PdmEngineeringOffsetPlan::new(Arc::from([0.0_f32, f32::MAX])).unwrap();
    let mut output = [71.0; 2];
    let mut current = [72.0; 2];
    for (requested, expected) in [
        (&[0.0][..], PdmEngineeringOffsetError::InputExtent),
        (
            &[0.0, f32::NAN][..],
            PdmEngineeringOffsetError::NonFiniteInput,
        ),
        (
            &[0.0, f32::INFINITY][..],
            PdmEngineeringOffsetError::NonFiniteInput,
        ),
        (
            &[0.0, f32::NEG_INFINITY][..],
            PdmEngineeringOffsetError::NonFiniteInput,
        ),
        (
            &[0.0, f32::MAX][..],
            PdmEngineeringOffsetError::NonFiniteOutput,
        ),
    ] {
        assert_eq!(
            plan.process(
                PdmEngineeringOffsetOutput {
                    requested_with_offset: &mut output,
                    current_offset: &mut current
                },
                &mut NoWorkspace,
                requested
            ),
            Err(expected)
        );
        assert_eq!(output, [71.0; 2]);
        assert_eq!(current, [72.0; 2]);
    }
    assert_eq!(
        plan.process(
            PdmEngineeringOffsetOutput {
                requested_with_offset: &mut output[..1],
                current_offset: &mut current
            },
            &mut NoWorkspace,
            &[0.0; 2]
        ),
        Err(PdmEngineeringOffsetError::OutputExtent)
    );
    assert_eq!(
        plan.process(
            PdmEngineeringOffsetOutput {
                requested_with_offset: &mut output,
                current_offset: &mut current[..1]
            },
            &mut NoWorkspace,
            &[0.0; 2]
        ),
        Err(PdmEngineeringOffsetError::CurrentOffsetExtent)
    );
    assert_eq!(output, [71.0; 2]);
    assert_eq!(current, [72.0; 2]);
}

#[test]
fn engineering_offset_saturation_feeds_back_the_composed_requested_command() {
    let requested = [0.125_f32, -0.125];
    let offset = [0.5_f32, -0.5];
    let plan = PdmEngineeringOffsetPlan::new(Arc::from(offset)).unwrap();
    let limiter = PdmCommandPlan::new(Arc::from([-0.25; 2]), Arc::from([0.25; 2]), None).unwrap();
    let mut workspace = limiter.workspace(&[0.0; 2]).unwrap();
    let mut demanded = [0.0; 2];
    let mut feedback = [0.0; 2];
    limiter
        .process(
            PdmCommandOutput {
                demanded: &mut demanded,
                constraint_feedback: &mut feedback,
            },
            &mut workspace,
            PdmCommandInput {
                requested: &requested,
            },
        )
        .unwrap();
    assert_eq!(demanded, requested);
    assert_eq!(feedback, [0.0; 2]);

    let mut composed_requested = [0.0; 2];
    let mut current_offset = [0.0; 2];
    plan.process(
        PdmEngineeringOffsetOutput {
            requested_with_offset: &mut composed_requested,
            current_offset: &mut current_offset,
        },
        &mut NoWorkspace,
        &requested,
    )
    .unwrap();
    assert_eq!(composed_requested, [0.625, -0.625]);
    assert_eq!(current_offset, offset);
    limiter
        .process(
            PdmCommandOutput {
                demanded: &mut demanded,
                constraint_feedback: &mut feedback,
            },
            &mut workspace,
            PdmCommandInput {
                requested: &composed_requested,
            },
        )
        .unwrap();
    assert_eq!(demanded, [0.25, -0.25]);
    assert_eq!(feedback, [0.375, -0.375]);
    for index in 0..2 {
        assert_eq!(feedback[index], composed_requested[index] - demanded[index]);
    }
}

#[test]
fn zero_offset_preserves_requested_bits_and_existing_limiter_behavior() {
    let requested = [-0.0_f32, 0.0, f32::from_bits(1), -0.45, 0.6];
    let mut output = [0.0; 5];
    let mut current = [0.0; 5];
    for zero in [0.0, -0.0] {
        let plan = PdmEngineeringOffsetPlan::new(Arc::from([zero; 5])).unwrap();
        plan.process(
            PdmEngineeringOffsetOutput {
                requested_with_offset: &mut output,
                current_offset: &mut current,
            },
            &mut NoWorkspace,
            &requested,
        )
        .unwrap();
        assert_eq!(output.map(f32::to_bits), requested.map(f32::to_bits));
        assert_eq!(current.map(f32::to_bits), [zero; 5].map(f32::to_bits));
    }
    let limiter = PdmCommandPlan::new(Arc::from([-0.25; 5]), Arc::from([0.25; 5]), None).unwrap();
    let mut first_workspace = limiter.workspace(&[0.0; 5]).unwrap();
    let mut second_workspace = limiter.workspace(&[0.0; 5]).unwrap();
    let mut direct = [0.0; 5];
    let mut composed = [0.0; 5];
    let mut direct_feedback = [0.0; 5];
    let mut composed_feedback = [0.0; 5];
    limiter
        .process(
            PdmCommandOutput {
                demanded: &mut direct,
                constraint_feedback: &mut direct_feedback,
            },
            &mut first_workspace,
            PdmCommandInput {
                requested: &requested,
            },
        )
        .unwrap();
    limiter
        .process(
            PdmCommandOutput {
                demanded: &mut composed,
                constraint_feedback: &mut composed_feedback,
            },
            &mut second_workspace,
            PdmCommandInput { requested: &output },
        )
        .unwrap();
    assert_eq!(direct.map(f32::to_bits), composed.map(f32::to_bits));
    assert_eq!(
        direct_feedback.map(f32::to_bits),
        composed_feedback.map(f32::to_bits)
    );
    let plan = PdmEngineeringOffsetPlan::new(Arc::from([0.0, 0.0, 0.0, 0.5, -0.5])).unwrap();
    plan.process(
        PdmEngineeringOffsetOutput {
            requested_with_offset: &mut output,
            current_offset: &mut current,
        },
        &mut NoWorkspace,
        &requested,
    )
    .unwrap();
    limiter
        .process(
            PdmCommandOutput {
                demanded: &mut composed,
                constraint_feedback: &mut composed_feedback,
            },
            &mut second_workspace,
            PdmCommandInput { requested: &output },
        )
        .unwrap();
    assert!((composed[3] - 0.05).abs() <= f32::EPSILON);
    assert!((composed[4] - 0.1).abs() <= f32::EPSILON);
}
