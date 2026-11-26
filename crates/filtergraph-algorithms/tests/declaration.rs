//! Transport-neutral algorithm declaration tests.

use filtergraph_algorithms::schemas::{CONTROLLER_COMMAND_V1, CONTROLLER_RESIDUAL_ERROR_V1};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmOutput, AlgorithmPlan, ArrayLayout,
    ConfigurationUpdate, DeclarationParameterError, DeclarationPreparationError,
    DeclarationProcessError, ElementType, FormatError, LeakyIntegratorF32,
    LeakyIntegratorF32Configuration, LeakyIntegratorPlan, LeakyIntegratorWorkspace,
    LeakyIntegratorWorkspaceError, OpticalGainCorrectionF32, OpticalGainCorrectionF32Configuration,
    PortDirection, PortError, PortRole, PropertyPolicy,
};
use serde::Deserialize;
use std::convert::Infallible;

filtergraph_algorithms::filtergraph_algorithm! {
    SourceF32Test {
        label = "source-f32-test";
        configuration SourceF32TestConfiguration {
            extent: usize = GraphRebuild;
        }
        plan = usize;
        workspace = filtergraph_algorithms::NoWorkspace;
        properties = none;
        dimensions |plan, _workspace| {
            extent = *plan;
        }
        inputs {}
        outputs {
            output: f32 [extent] = "output", schema = "test.source.f32/1";
        }
        prepare |configuration| {
            Ok((configuration.extent, filtergraph_algorithms::NoWorkspace))
        }
        prepare_error = Infallible;
        process |_plan, _workspace| {
            output.fill(2.0);
            Ok(())
        }
        process_error = Infallible;
    }
}

#[derive(Deserialize)]
struct Fixture {
    declaration: DeclarationFixture,
    configuration: Vec<ConfigurationFixture>,
    ports: Vec<PortFixture>,
    native: NativeFixture,
}

#[derive(Deserialize)]
struct DeclarationFixture {
    label: String,
    property_policy: String,
}

#[derive(Deserialize)]
struct ConfigurationFixture {
    name: String,
    r#type: String,
    update: String,
}

#[derive(Deserialize)]
struct PortFixture {
    name: String,
    direction: String,
    role: String,
    element_type: String,
    shape: Vec<String>,
    schema: String,
    metadata_from: Option<String>,
}

#[derive(Deserialize)]
struct NativeFixture {
    extent: usize,
    initial_state: f32,
    input_schema: String,
    output_schema: String,
    inputs: Vec<Vec<f32>>,
    outputs: Vec<Vec<f32>>,
}

fn fixture() -> Fixture {
    toml::from_str(include_str!(
        "../../../fixtures/algorithms/leaky_integrator_declaration.toml"
    ))
    .unwrap()
}

#[test]
fn leaky_integrator_declaration_is_complete_and_transport_neutral() {
    let fixture = fixture();
    assert_eq!(LeakyIntegratorF32::NAME, fixture.declaration.label);
    assert_eq!(LeakyIntegratorF32::PROPERTIES, PropertyPolicy::Runtime);
    assert_eq!(fixture.declaration.property_policy, "runtime");

    let configuration = LeakyIntegratorF32::CONFIGURATION;
    assert_eq!(configuration.len(), 4);
    assert_eq!(configuration[0].name(), "extent");
    assert_eq!(configuration[0].rust_type(), "usize");
    assert_eq!(configuration[0].update(), ConfigurationUpdate::GraphRebuild);
    assert_eq!(configuration[1].name(), "initial_state");
    assert_eq!(configuration[1].rust_type(), "f32");
    assert_eq!(configuration[1].update(), ConfigurationUpdate::Construction);
    assert_eq!(configuration[2].name(), "input_schema");
    assert_eq!(configuration[2].rust_type(), "String");
    assert_eq!(configuration[2].update(), ConfigurationUpdate::GraphRebuild);
    assert_eq!(configuration[3].name(), "output_schema");
    assert_eq!(configuration[3].rust_type(), "String");
    assert_eq!(configuration[3].update(), ConfigurationUpdate::GraphRebuild);
    for (actual, expected) in configuration.iter().zip(&fixture.configuration) {
        assert_eq!(actual.name(), expected.name);
        assert_eq!(
            actual.update(),
            match expected.update.as_str() {
                "construction" => ConfigurationUpdate::Construction,
                "graph-rebuild" => ConfigurationUpdate::GraphRebuild,
                update => panic!("unknown fixture update {update}"),
            }
        );
    }
    assert_eq!(fixture.configuration[0].r#type, "extent");
    assert_eq!(fixture.configuration[1].r#type, "float32");
    assert_eq!(fixture.configuration[2].r#type, "string");
    assert_eq!(fixture.configuration[3].r#type, "string");

    let ports = LeakyIntegratorF32::PORTS;
    assert_eq!(ports.len(), 2);
    assert_eq!(ports[0].name(), "input");
    assert_eq!(ports[0].direction(), PortDirection::Input);
    assert_eq!(ports[0].role(), PortRole::Data);
    assert_eq!(ports[0].element_type(), ElementType::F32);
    assert_eq!(ports[0].metadata_source(), None);
    assert_eq!(ports[1].name(), "output");
    assert_eq!(ports[1].direction(), PortDirection::Output);
    assert_eq!(ports[1].role(), PortRole::Data);
    assert_eq!(ports[1].element_type(), ElementType::F32);
    assert_eq!(ports[1].metadata_source(), Some("input"));
    for (actual, expected) in ports.iter().zip(&fixture.ports) {
        assert_eq!(actual.name(), expected.name);
        assert_eq!(expected.role, "data");
        assert_eq!(expected.element_type, "float32");
        assert_eq!(expected.shape, ["extent"]);
        assert!(
            [CONTROLLER_RESIDUAL_ERROR_V1, CONTROLLER_COMMAND_V1]
                .contains(&expected.schema.as_str())
        );
        assert_eq!(actual.metadata_source(), expected.metadata_from.as_deref());
        assert_eq!(
            actual.direction(),
            match expected.direction.as_str() {
                "input" => PortDirection::Input,
                "output" => PortDirection::Output,
                direction => panic!("unknown fixture direction {direction}"),
            }
        );
    }
}

#[test]
fn native_declaration_matches_direct_plan_over_stateful_samples() {
    let fixture = fixture();
    let configuration = LeakyIntegratorF32Configuration {
        extent: fixture.native.extent,
        initial_state: fixture.native.initial_state,
        input_schema: fixture.native.input_schema.clone(),
        output_schema: fixture.native.output_schema.clone(),
    };
    let mut declared = LeakyIntegratorF32::prepare(configuration).unwrap();

    assert_eq!(declared.formats().len(), 2);
    for (format, schema) in declared
        .formats()
        .iter()
        .zip([CONTROLLER_RESIDUAL_ERROR_V1, CONTROLLER_COMMAND_V1])
    {
        assert_eq!(format.element_type(), ElementType::F32);
        assert_eq!(format.layout(), ArrayLayout::RowMajor);
        assert_eq!(format.shape(), &[fixture.native.extent as u32]);
        assert_eq!(format.schema(), schema);
    }

    let plan = LeakyIntegratorPlan::<f32>::default();
    let mut workspace =
        LeakyIntegratorWorkspace::filled(fixture.native.extent, fixture.native.initial_state)
            .unwrap();
    let mut expected = vec![0.0_f32; fixture.native.extent];
    let mut actual = vec![0.0_f32; fixture.native.extent];

    for (input, fixture_output) in fixture.native.inputs.iter().zip(&fixture.native.outputs) {
        plan.process(&mut expected, &mut workspace, input).unwrap();
        LeakyIntegratorF32::process(&mut declared, input, &mut actual).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(&actual, fixture_output);
        assert_eq!(declared.workspace().state(), workspace.state());
    }

    LeakyIntegratorF32::reset(&mut declared);
    assert_eq!(
        declared.workspace().state(),
        vec![fixture.native.initial_state; fixture.native.extent]
    );
}

#[test]
fn native_sparse_parameter_replaces_the_complete_prepared_plan() {
    let mut declared = OpticalGainCorrectionF32::prepare(OpticalGainCorrectionF32Configuration {
        optical_gains: vec![1.0, 2.0],
    })
    .unwrap();
    let input = [2.0_f32, 8.0];
    let mut output = [0.0_f32; 2];

    OpticalGainCorrectionF32::process(&mut declared, &input, &mut output).unwrap();
    assert_eq!(output, [2.0, 4.0]);

    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&[2.0_f32, 4.0][..])),
        Some(Ok(()))
    );
    OpticalGainCorrectionF32::process(&mut declared, &input, &mut output).unwrap();
    assert_eq!(output, [1.0, 2.0]);

    assert!(matches!(
        declared.replace_parameter(2, AlgorithmInput::from(&[2_u32, 4][..])),
        Some(Err(DeclarationParameterError::Port(PortError::ElementType)))
    ));
    assert_eq!(
        declared.replace_parameter(0, AlgorithmInput::from(&[1.0_f32][..])),
        None
    );
}

#[test]
fn declaration_rejects_invalid_construction_and_generic_port_bindings() {
    assert!(matches!(
        LeakyIntegratorF32::prepare(LeakyIntegratorF32Configuration {
            extent: 0,
            initial_state: 0.0,
            input_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
            output_schema: CONTROLLER_COMMAND_V1.into(),
        }),
        Err(DeclarationPreparationError::Algorithm(
            LeakyIntegratorWorkspaceError::EmptyState
        ))
    ));

    for schema in ["", "missing-version", "schema/0", "schema/not-a-number"] {
        assert!(matches!(
            LeakyIntegratorF32::prepare(LeakyIntegratorF32Configuration {
                extent: 2,
                initial_state: 0.0,
                input_schema: schema.into(),
                output_schema: CONTROLLER_COMMAND_V1.into(),
            }),
            Err(DeclarationPreparationError::Format(FormatError::Schema))
        ));
    }

    let mut declared = LeakyIntegratorF32::prepare(LeakyIntegratorF32Configuration {
        extent: 2,
        initial_state: 0.0,
        input_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
        output_schema: CONTROLLER_COMMAND_V1.into(),
    })
    .unwrap();
    let input = [1.0_f32, 2.0];
    let wrong_input = [1_i32, 2];
    let mut output = [0.0_f32; 2];

    assert_eq!(
        declared.process(&[], &mut [AlgorithmOutput::from(&mut output[..])]),
        Err(DeclarationProcessError::Port(PortError::Missing))
    );
    assert_eq!(
        declared.process(
            &[AlgorithmInput::from(&wrong_input[..])],
            &mut [AlgorithmOutput::from(&mut output[..])],
        ),
        Err(DeclarationProcessError::Port(PortError::ElementType))
    );
    assert_eq!(
        declared.process(
            &[
                AlgorithmInput::from(&input[..]),
                AlgorithmInput::from(&input[..]),
            ],
            &mut [AlgorithmOutput::from(&mut output[..])],
        ),
        Err(DeclarationProcessError::Port(PortError::Extra))
    );
}

#[test]
fn source_declaration_does_not_invent_an_input_metadata_source() {
    let mut declared = SourceF32Test::prepare(SourceF32TestConfiguration { extent: 4 }).unwrap();
    assert_eq!(SourceF32Test::PORTS.len(), 1);
    assert_eq!(SourceF32Test::PORTS[0].direction(), PortDirection::Output);
    assert_eq!(SourceF32Test::PORTS[0].metadata_source(), None);
    let mut output = [0.0_f32; 4];
    SourceF32Test::process(&mut declared, &mut output).unwrap();
    assert_eq!(output, [2.0; 4]);
}

#[test]
fn sample_metadata_annotations_are_owned_by_data_inputs() {
    use filtergraph_algorithms::{AlgorithmPort, ChoppedImageDifferenceF32, SccCorrectionF32};
    let optical = &<ChoppedImageDifferenceF32 as AlgorithmDeclaration>::PORTS[0];
    assert_eq!(optical.required_sample_metadata(), &[] as &[&str]);
    assert_eq!(
        optical.optional_sample_metadata(),
        &[
            "chopped_image_kind",
            "trigger_state",
            "acquisition_identity",
            "exposure_duration_nanoseconds"
        ]
    );
    let scc = &<SccCorrectionF32 as AlgorithmDeclaration>::PORTS[0];
    assert_eq!(
        scc.required_sample_metadata(),
        &[
            "acquisition_identity",
            "exposure_duration_nanoseconds",
            "chopped_image_kind"
        ]
    );
    assert!(scc.optional_sample_metadata().is_empty());
    for (direction, role) in [
        (PortDirection::Output, PortRole::Data),
        (PortDirection::Input, PortRole::Parameter),
    ] {
        let port = AlgorithmPort::new("value", direction, role, ElementType::F32, None);
        assert!(
            std::panic::catch_unwind(
                || port.with_required_sample_metadata(&["acquisition_identity"])
            )
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| port.with_optional_sample_metadata(&["trigger_state"]))
                .is_err()
        );
    }
    assert!(
        <SourceF32Test as AlgorithmDeclaration>::PORTS[0]
            .required_sample_metadata()
            .is_empty()
    );
    assert!(
        <SourceF32Test as AlgorithmDeclaration>::PORTS[0]
            .optional_sample_metadata()
            .is_empty()
    );
}
