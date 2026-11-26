//! PDM total command-power projection tests.

use filtergraph_algorithms::schemas::REQUESTED_PDM_COMMAND_V1;
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmPlan, DeclarationPreparationError, NoWorkspace,
    PdmCommandPowerLimitError, PdmCommandPowerLimitF32, PdmCommandPowerLimitF32Configuration,
    PdmCommandPowerLimitPlan, PdmCommandPowerLimitPlanError, PortDirection, PortRole,
    PropertyPolicy,
};

#[test]
fn preparation_rejects_empty_and_invalid_limits() {
    assert_eq!(
        PdmCommandPowerLimitPlan::new(0, 1.0_f32),
        Err(PdmCommandPowerLimitPlanError::Empty)
    );
    for maximum in [-1.0_f32, f32::NAN, f32::INFINITY] {
        assert_eq!(
            PdmCommandPowerLimitPlan::new(2, maximum),
            Err(PdmCommandPowerLimitPlanError::InvalidMaximum)
        );
    }
}

#[test]
fn commands_inside_and_on_the_power_boundary_are_preserved() {
    let plan = PdmCommandPowerLimitPlan::new(2, 25.0_f32).unwrap();
    let mut workspace = NoWorkspace;
    for input in [[1.0, 2.0], [3.0, 4.0]] {
        let mut output = [0.0; 2];
        plan.process(&mut output, &mut workspace, &input).unwrap();
        assert_eq!(output, input);
    }
}

#[test]
fn command_outside_the_power_boundary_uses_one_common_gain() {
    let plan = PdmCommandPowerLimitPlan::new(2, 4.0_f64).unwrap();
    let mut output = [0.0; 2];
    plan.process(&mut output, &mut NoWorkspace, &[3.0, 4.0])
        .unwrap();

    assert!((output[0] - 1.2).abs() <= 8.0 * f64::EPSILON);
    assert!((output[1] - 1.6).abs() <= 8.0 * f64::EPSILON);
    assert!((output[0] / output[1] - 0.75).abs() <= 8.0 * f64::EPSILON);
    assert!((output.iter().map(|value| value * value).sum::<f64>() - 4.0).abs() <= 1.0e-14);
}

#[test]
fn zero_maximum_maps_every_finite_command_to_zero() {
    let plan = PdmCommandPowerLimitPlan::new(3, 0.0_f32).unwrap();
    let mut output = [1.0; 3];
    plan.process(&mut output, &mut NoWorkspace, &[1.0, -2.0, 0.0])
        .unwrap();
    assert_eq!(output, [0.0; 3]);
}

#[test]
fn large_finite_float64_input_is_projected_without_overflow() {
    let plan = PdmCommandPowerLimitPlan::new(2, 1.0_f64).unwrap();
    let mut output = [0.0; 2];
    plan.process(&mut output, &mut NoWorkspace, &[f64::MAX, -f64::MAX])
        .unwrap();

    let expected = 1.0 / 2.0_f64.sqrt();
    assert!(output.iter().all(|value| value.is_finite()));
    assert!((output[0] - expected).abs() <= 8.0 * f64::EPSILON);
    assert!((output[1] + expected).abs() <= 8.0 * f64::EPSILON);
}

#[test]
fn execution_rejects_extent_and_nonfinite_input_before_projection() {
    let plan = PdmCommandPowerLimitPlan::new(2, 1.0_f32).unwrap();
    assert_eq!(
        plan.process(&mut [0.0; 2], &mut NoWorkspace, &[0.0]),
        Err(PdmCommandPowerLimitError::InputExtent)
    );
    assert_eq!(
        plan.process(&mut [0.0], &mut NoWorkspace, &[0.0; 2]),
        Err(PdmCommandPowerLimitError::OutputExtent)
    );
    assert_eq!(
        plan.process(&mut [0.0; 2], &mut NoWorkspace, &[f32::NAN, 0.0]),
        Err(PdmCommandPowerLimitError::NonFiniteInput)
    );
}

#[test]
fn declaration_exposes_one_composable_command_projection() {
    assert_eq!(PdmCommandPowerLimitF32::NAME, "pdm-command-power-limit-f32");
    assert_eq!(PdmCommandPowerLimitF32::PROPERTIES, PropertyPolicy::None);
    assert_eq!(PdmCommandPowerLimitF32::CONFIGURATION.len(), 2);

    let ports = PdmCommandPowerLimitF32::PORTS;
    assert_eq!(ports.len(), 2);
    assert_eq!(ports[0].name(), "requested");
    assert_eq!(ports[0].direction(), PortDirection::Input);
    assert_eq!(ports[0].role(), PortRole::Data);
    assert_eq!(ports[0].metadata_source(), None);
    assert_eq!(ports[1].name(), "power-limited");
    assert_eq!(ports[1].direction(), PortDirection::Output);
    assert_eq!(ports[1].role(), PortRole::Data);
    assert_eq!(ports[1].metadata_source(), Some("requested"));

    let prepared = PdmCommandPowerLimitF32::prepare(PdmCommandPowerLimitF32Configuration {
        actuator_count: 277,
        max_command_power: 4.0,
    })
    .unwrap();
    for format in prepared.formats() {
        assert_eq!(format.shape(), &[277]);
        assert_eq!(format.schema(), REQUESTED_PDM_COMMAND_V1);
    }
}

#[test]
fn declared_command_power_matches_the_direct_plan() {
    let configuration = PdmCommandPowerLimitF32Configuration {
        actuator_count: 3,
        max_command_power: 4.0,
    };
    let mut declared = PdmCommandPowerLimitF32::prepare(configuration).unwrap();
    let direct = PdmCommandPowerLimitPlan::new(3, 4.0_f32).unwrap();
    let input = [3.0_f32, 4.0, 0.0];
    let mut expected = [0.0_f32; 3];
    let mut actual = [0.0_f32; 3];

    direct
        .process(&mut expected, &mut NoWorkspace, &input)
        .unwrap();
    PdmCommandPowerLimitF32::process(&mut declared, &input, &mut actual).unwrap();
    assert_eq!(actual, expected);

    assert!(matches!(
        PdmCommandPowerLimitF32::prepare(PdmCommandPowerLimitF32Configuration {
            actuator_count: 0,
            max_command_power: 4.0,
        }),
        Err(DeclarationPreparationError::Algorithm(
            PdmCommandPowerLimitPlanError::Empty
        ))
    ));
}
