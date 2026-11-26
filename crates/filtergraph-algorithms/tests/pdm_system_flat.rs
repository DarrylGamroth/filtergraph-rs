//! PDM system-flat declaration tests.

use std::sync::Arc;

use filtergraph_algorithms::schemas::{PDM_SYSTEM_FLAT_V1, REQUESTED_PDM_COMMAND_V1};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, DeclarationParameterError, NoWorkspace,
    PdmSystemFlatError, PdmSystemFlatF32, PdmSystemFlatF32Configuration, PdmSystemFlatPlan,
    PdmSystemFlatPlanError, PortDirection, PortRole, PropertyPolicy,
};

#[test]
fn preparation_rejects_empty_and_nonfinite_system_flats() {
    assert_eq!(
        PdmSystemFlatPlan::<f32>::new(Arc::from([])),
        Err(PdmSystemFlatPlanError::Empty)
    );
    assert_eq!(
        PdmSystemFlatPlan::new(Arc::from([0.0_f32, f32::NAN])),
        Err(PdmSystemFlatPlanError::NonFiniteConfiguration)
    );
}

#[test]
fn system_flat_addition_preserves_requested_coordinate_order() {
    let plan = PdmSystemFlatPlan::new(Arc::from([0.5_f32, -1.25, 2.0])).unwrap();
    let mut output = [0.0; 3];

    plan.process(&mut output, &mut NoWorkspace, &[1.0, 3.25, -0.5])
        .unwrap();

    assert_eq!(output, [1.5, 2.0, 1.5]);
}

#[test]
fn processing_rejects_invalid_extents_and_nonfinite_values_before_writing() {
    let plan = PdmSystemFlatPlan::new(Arc::from([f32::MAX, 0.0])).unwrap();
    let mut output = [7.0_f32; 2];

    assert_eq!(
        plan.process(&mut output, &mut NoWorkspace, &[0.0]),
        Err(PdmSystemFlatError::InputExtent)
    );
    assert_eq!(output, [7.0; 2]);
    assert_eq!(
        plan.process(&mut output[..1], &mut NoWorkspace, &[0.0, 0.0]),
        Err(PdmSystemFlatError::OutputExtent)
    );
    assert_eq!(output, [7.0; 2]);
    assert_eq!(
        plan.process(&mut output, &mut NoWorkspace, &[f32::NAN, 0.0]),
        Err(PdmSystemFlatError::NonFiniteInput)
    );
    assert_eq!(output, [7.0; 2]);
    assert_eq!(
        plan.process(&mut output, &mut NoWorkspace, &[f32::MAX, 0.0]),
        Err(PdmSystemFlatError::NonFiniteOutput)
    );
    assert_eq!(output, [7.0; 2]);
}

#[test]
fn declaration_exposes_a_composable_sparse_system_flat() {
    assert_eq!(PdmSystemFlatF32::NAME, "pdm-system-flat-f32");
    assert_eq!(PdmSystemFlatF32::PROPERTIES, PropertyPolicy::None);
    assert_eq!(PdmSystemFlatF32::CONFIGURATION.len(), 1);
    assert_eq!(PdmSystemFlatF32::PORTS.len(), 3);
    assert_eq!(PdmSystemFlatF32::PORTS[0].name(), "requested");
    assert_eq!(PdmSystemFlatF32::PORTS[0].direction(), PortDirection::Input);
    assert_eq!(PdmSystemFlatF32::PORTS[0].role(), PortRole::Data);
    assert_eq!(
        PdmSystemFlatF32::PORTS[1].name(),
        "requested-with-system-flat"
    );
    assert_eq!(
        PdmSystemFlatF32::PORTS[1].direction(),
        PortDirection::Output
    );
    assert_eq!(PdmSystemFlatF32::PORTS[1].role(), PortRole::Data);
    assert_eq!(
        PdmSystemFlatF32::PORTS[1].metadata_source(),
        Some("requested")
    );
    assert_eq!(PdmSystemFlatF32::PORTS[2].name(), "system-flat");
    assert_eq!(PdmSystemFlatF32::PORTS[2].direction(), PortDirection::Input);
    assert_eq!(PdmSystemFlatF32::PORTS[2].role(), PortRole::Parameter);

    let mut declared = PdmSystemFlatF32::prepare(PdmSystemFlatF32Configuration {
        actuator_count: 277,
    })
    .unwrap();
    assert_eq!(declared.formats()[0].shape(), &[277]);
    assert_eq!(declared.formats()[0].schema(), REQUESTED_PDM_COMMAND_V1);
    assert_eq!(declared.formats()[1].shape(), &[277]);
    assert_eq!(declared.formats()[1].schema(), REQUESTED_PDM_COMMAND_V1);
    assert_eq!(declared.formats()[2].shape(), &[277]);
    assert_eq!(declared.formats()[2].schema(), PDM_SYSTEM_FLAT_V1);

    let requested = [0.25_f32; 277];
    let flat = [0.75_f32; 277];
    let mut output = [0.0_f32; 277];
    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&flat[..])),
        Some(Ok(()))
    );
    PdmSystemFlatF32::process(&mut declared, &requested, &mut output).unwrap();
    assert_eq!(output, [1.0; 277]);

    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&[0.0_f32][..])),
        Some(Err(DeclarationParameterError::Algorithm(
            PdmSystemFlatPlanError::Extent
        )))
    );
    assert_eq!(declared.plan().system_flat(), &flat);
}

#[test]
fn declaration_rejects_a_zero_actuator_topology() {
    assert!(
        PdmSystemFlatF32::prepare(PdmSystemFlatF32Configuration { actuator_count: 0 }).is_err()
    );
}
