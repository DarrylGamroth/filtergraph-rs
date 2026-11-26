//! FilterGraph scalar-property contract conformance.

use core::convert::Infallible;
use core::fmt;

use filtergraph_algorithms::{
    AlgorithmPlan, AlgorithmProperties, LeakyIntegratorPlan, LeakyIntegratorPlanError,
    LeakyIntegratorWorkspace, NoWorkspace, PropertyAssignment, PropertyBinding,
    PropertyBindingError, PropertyChoice, PropertyConstraint, PropertyError, PropertyInfo,
    PropertySchemaError, PropertyType, PropertyUpdate, PropertyValue, validate_property_info,
    validate_property_schema,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    property: PropertyFixture,
    initial: StepFixture,
    update: UpdateFixture,
}

#[derive(Deserialize)]
struct PropertyFixture {
    gain_id: u32,
    gain_name: String,
    pole_id: u32,
    pole_name: String,
    unit: String,
}

#[derive(Deserialize)]
struct StepFixture {
    gain: f32,
    pole: f32,
    state: Vec<f32>,
    input: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Deserialize)]
struct UpdateFixture {
    gain: f32,
    pole: f32,
    input: Vec<f32>,
    output: Vec<f32>,
    requested_generation: u64,
    active_generation: u64,
}

const MODE_PROPERTY: u32 = 0;
const CONSTRUCTION_PROPERTY: u32 = 1;
const GRAPH_REBUILD_PROPERTY: u32 = 2;
const MODE_CHOICES: [PropertyChoice<'static>; 2] = [
    PropertyChoice {
        value: PropertyValue::Id(0),
        name: "off",
        description: "Off",
    },
    PropertyChoice {
        value: PropertyValue::Id(1),
        name: "on",
        description: "On",
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct UpdateClassPlan {
    mode: u32,
    construction: i32,
    graph_rebuild: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DuplicateIdPlan {
    first: i32,
    second: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CoupledPlan {
    lower: i32,
    upper: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CoupledPlanError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NestedValues {
    lower: i32,
    upper: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NestedPlan {
    values: NestedValues,
}

impl fmt::Display for CoupledPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("lower must not exceed upper")
    }
}

impl core::error::Error for CoupledPlanError {}

impl AlgorithmPlan for CoupledPlan {
    type Input<'a> = ();
    type Output<'a> = ();
    type Workspace = NoWorkspace;
    type Error = Infallible;

    fn process<'input, 'output>(
        &self,
        _output: Self::Output<'output>,
        _workspace: &mut Self::Workspace,
        _input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn replace_coupled_properties(
    _plan: &CoupledPlan,
    lower: i32,
    upper: i32,
) -> Result<CoupledPlan, CoupledPlanError> {
    (lower <= upper)
        .then_some(CoupledPlan { lower, upper })
        .ok_or(CoupledPlanError)
}

filtergraph_algorithms::algorithm_properties! {
    impl CoupledPlan {
        plan_error = CoupledPlanError;
        replace = replace_coupled_properties;
        properties {
            10 => lower: Int {
                name = "lower";
                description = "Lower admitted value";
                unit = "1";
                update = Runtime;
                default = 0;
                constraint = PropertyConstraint::None;
            }
            11 => upper: Int {
                name = "upper";
                description = "Upper admitted value";
                unit = "1";
                update = Runtime;
                default = 1;
                constraint = PropertyConstraint::None;
            }
        }
    }
}

impl AlgorithmPlan for NestedPlan {
    type Input<'a> = ();
    type Output<'a> = ();
    type Workspace = NoWorkspace;
    type Error = Infallible;

    fn process<'input, 'output>(
        &self,
        _output: Self::Output<'output>,
        _workspace: &mut Self::Workspace,
        _input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn replace_nested_properties(
    _plan: &NestedPlan,
    lower: i32,
    upper: i32,
) -> Result<NestedPlan, Infallible> {
    Ok(NestedPlan {
        values: NestedValues { lower, upper },
    })
}

filtergraph_algorithms::algorithm_properties! {
    impl NestedPlan {
        plan_error = Infallible;
        replace = replace_nested_properties;
        properties {
            20 => lower from values.lower: Int {
                name = "nested-lower";
                description = "Nested lower value";
                unit = "1";
                update = Runtime;
                default = 0;
                constraint = PropertyConstraint::None;
            }
            21 => upper from values.upper: Int {
                name = "nested-upper";
                description = "Nested upper value";
                unit = "1";
                update = Runtime;
                default = 1;
                constraint = PropertyConstraint::None;
            }
        }
    }
}

impl AlgorithmPlan for DuplicateIdPlan {
    type Input<'a> = ();
    type Output<'a> = ();
    type Workspace = NoWorkspace;
    type Error = Infallible;

    fn process<'input, 'output>(
        &self,
        _output: Self::Output<'output>,
        _workspace: &mut Self::Workspace,
        _input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn replace_duplicate_id_properties(
    _plan: &DuplicateIdPlan,
    first: i32,
    second: i32,
) -> Result<DuplicateIdPlan, Infallible> {
    Ok(DuplicateIdPlan { first, second })
}

filtergraph_algorithms::algorithm_properties! {
    impl DuplicateIdPlan {
        plan_error = Infallible;
        replace = replace_duplicate_id_properties;
        properties {
            5 => first: Int {
                name = "first";
                description = "First value";
                unit = "1";
                update = Runtime;
                default = 0;
                constraint = PropertyConstraint::None;
            }
            5 => second: Int {
                name = "second";
                description = "Second value";
                unit = "1";
                update = Runtime;
                default = 0;
                constraint = PropertyConstraint::None;
            }
        }
    }
}

impl AlgorithmPlan for UpdateClassPlan {
    type Input<'a> = ();
    type Output<'a> = ();
    type Workspace = NoWorkspace;
    type Error = Infallible;

    fn process<'input, 'output>(
        &self,
        _output: Self::Output<'output>,
        _workspace: &mut Self::Workspace,
        _input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn replace_update_class_properties(
    _plan: &UpdateClassPlan,
    mode: u32,
    construction: i32,
    graph_rebuild: i32,
) -> Result<UpdateClassPlan, Infallible> {
    Ok(UpdateClassPlan {
        mode,
        construction,
        graph_rebuild,
    })
}

filtergraph_algorithms::algorithm_properties! {
    impl UpdateClassPlan {
        plan_error = Infallible;
        replace = replace_update_class_properties;
        properties {
            MODE_PROPERTY => mode: Id {
                name = "mode";
                description = "Observable mode";
                unit = "1";
                update = ReadOnly;
                default = 0;
                constraint = PropertyConstraint::Choices(&MODE_CHOICES);
            }
            CONSTRUCTION_PROPERTY => construction: Int {
                name = "construction";
                description = "Construction-only value";
                unit = "1";
                update = Construction;
                default = 1;
                constraint = PropertyConstraint::None;
            }
            GRAPH_REBUILD_PROPERTY => graph_rebuild: Int {
                name = "graph-rebuild";
                description = "Graph-defining value";
                unit = "1";
                update = GraphRebuild;
                default = 1;
                constraint = PropertyConstraint::None;
            }
        }
    }
}

fn fixture() -> Fixture {
    toml::from_str(include_str!(
        "../../../fixtures/algorithms/leaky_integrator_properties.toml"
    ))
    .unwrap()
}

fn workspace_with_state(state: &[f32]) -> LeakyIntegratorWorkspace<f32> {
    let identity = LeakyIntegratorPlan::new(1.0_f32, 0.0).unwrap();
    let mut workspace = LeakyIntegratorWorkspace::new(state.len()).unwrap();
    let mut output = vec![0.0; state.len()];
    identity
        .process(&mut output, &mut workspace, state)
        .unwrap();
    workspace
}

#[test]
fn common_scalar_domain_and_choice_metadata_are_distinct() {
    let values = [
        PropertyValue::Bool(true),
        PropertyValue::Int(-1),
        PropertyValue::Long(-2),
        PropertyValue::Float(0.25),
        PropertyValue::Double(0.5),
        PropertyValue::Id(3),
        PropertyValue::String("mode"),
    ];
    assert_eq!(
        values.map(PropertyValue::property_type),
        [
            PropertyType::Bool,
            PropertyType::Int,
            PropertyType::Long,
            PropertyType::Float,
            PropertyType::Double,
            PropertyType::Id,
            PropertyType::String,
        ]
    );

    let choices = [
        PropertyChoice {
            value: PropertyValue::Id(0),
            name: "off",
            description: "Off",
        },
        PropertyChoice {
            value: PropertyValue::Id(1),
            name: "on",
            description: "On",
        },
    ];
    assert_eq!(
        PropertyConstraint::Choices(&choices),
        PropertyConstraint::Choices(&choices)
    );
}

#[test]
fn schema_admission_rejects_ambiguous_and_incompatible_descriptors() {
    let plan = DuplicateIdPlan {
        first: 0,
        second: 0,
    };
    assert_eq!(
        validate_property_schema(&plan),
        Err(PropertySchemaError::DuplicateId {
            id: 5,
            first: 0,
            second: 1,
        })
    );
    assert_eq!(
        plan.prepare_properties(&[PropertyAssignment {
            id: 5,
            value: PropertyValue::Int(1),
        }]),
        Err(PropertyError::InvalidSchema(
            PropertySchemaError::DuplicateId {
                id: 5,
                first: 0,
                second: 1,
            }
        ))
    );

    let invalid_range = PropertyInfo {
        id: 8,
        name: "enabled",
        description: "Whether processing is enabled",
        unit: "1",
        update: PropertyUpdate::Runtime,
        default_value: PropertyValue::Bool(false),
        constraint: PropertyConstraint::Range {
            minimum: PropertyValue::Bool(false),
            maximum: PropertyValue::Bool(true),
        },
    };
    assert_eq!(
        validate_property_info(invalid_range),
        Err(PropertySchemaError::InvalidRange { id: 8 })
    );
}

#[test]
fn declarative_schema_enforces_update_class_and_choice_constraints() {
    let plan = UpdateClassPlan {
        mode: 1,
        construction: 1,
        graph_rebuild: 1,
    };
    let info = plan.property_info(0).unwrap();

    assert_eq!(info.constraint, PropertyConstraint::Choices(&MODE_CHOICES));
    assert_eq!(plan.property(MODE_PROPERTY), Ok(PropertyValue::Id(1)));
    for assignment in [
        PropertyAssignment {
            id: MODE_PROPERTY,
            value: PropertyValue::Id(0),
        },
        PropertyAssignment {
            id: CONSTRUCTION_PROPERTY,
            value: PropertyValue::Int(2),
        },
        PropertyAssignment {
            id: GRAPH_REBUILD_PROPERTY,
            value: PropertyValue::Int(2),
        },
    ] {
        assert_eq!(
            plan.prepare_properties(&[assignment]),
            Err(PropertyError::<Infallible>::PropertyUpdateClass { id: assignment.id })
        );
    }
}

#[test]
fn leaky_integrator_descriptors_match_the_shared_contract() {
    let fixture = fixture();
    let plan = LeakyIntegratorPlan::new(fixture.initial.gain, fixture.initial.pole).unwrap();
    let gain = plan.property_info(0).unwrap();
    let pole = plan.property_info(1).unwrap();

    assert_eq!(gain.id, fixture.property.gain_id);
    assert_eq!(gain.name, fixture.property.gain_name);
    assert_eq!(gain.unit, fixture.property.unit);
    assert_eq!(gain.update, PropertyUpdate::Runtime);
    assert_eq!(gain.property_type(), PropertyType::Float);
    assert_eq!(pole.id, fixture.property.pole_id);
    assert_eq!(pole.name, fixture.property.pole_name);
    assert_eq!(pole.unit, fixture.property.unit);
    assert_eq!(pole.update, PropertyUpdate::Runtime);
    assert_eq!(pole.property_type(), PropertyType::Float);
    assert_eq!(plan.property_info(2).unwrap().name, "closed");
    assert_eq!(plan.property_info(3), None);
}

#[test]
fn property_preparation_is_typed_transactional_and_failure_atomic() {
    let fixture = fixture();
    let plan = LeakyIntegratorPlan::new(fixture.initial.gain, fixture.initial.pole).unwrap();
    let gain = fixture.property.gain_id;
    let pole = fixture.property.pole_id;

    let prepared = plan
        .prepare_properties(&[
            PropertyAssignment {
                id: gain,
                value: PropertyValue::Float(fixture.update.gain),
            },
            PropertyAssignment {
                id: pole,
                value: PropertyValue::Float(fixture.update.pole),
            },
        ])
        .unwrap();
    assert_eq!(prepared.gain(), fixture.update.gain);
    assert_eq!(prepared.pole(), fixture.update.pole);
    assert_eq!(plan.gain(), fixture.initial.gain);
    assert_eq!(plan.pole(), fixture.initial.pole);

    assert_eq!(
        plan.prepare_properties(&[
            PropertyAssignment {
                id: gain,
                value: PropertyValue::Float(1.0),
            },
            PropertyAssignment {
                id: gain,
                value: PropertyValue::Float(2.0),
            },
        ]),
        Err(PropertyError::DuplicateProperty { id: gain })
    );
    assert_eq!(
        plan.prepare_properties(&[PropertyAssignment {
            id: pole,
            value: PropertyValue::Double(0.5),
        }]),
        Err(PropertyError::PropertyType {
            id: pole,
            expected: PropertyType::Float,
            actual: PropertyType::Double,
        })
    );
    assert_eq!(
        plan.prepare_properties(&[PropertyAssignment {
            id: 99,
            value: PropertyValue::Float(0.5),
        }]),
        Err(PropertyError::UnknownProperty { id: 99 })
    );
    assert_eq!(
        plan.prepare_properties(&[PropertyAssignment {
            id: pole,
            value: PropertyValue::Float(1.5),
        }]),
        Err(PropertyError::<LeakyIntegratorPlanError>::PropertyConstraint { id: pole })
    );

    let coupled = CoupledPlan { lower: 0, upper: 1 };
    assert_eq!(
        coupled.prepare_properties(&[
            PropertyAssignment {
                id: 10,
                value: PropertyValue::Int(3),
            },
            PropertyAssignment {
                id: 11,
                value: PropertyValue::Int(2),
            },
        ]),
        Err(PropertyError::InvalidPlan(CoupledPlanError))
    );
    assert_eq!(coupled, CoupledPlan { lower: 0, upper: 1 });
}

#[test]
fn property_macro_can_bind_nested_scientific_values() {
    let plan = NestedPlan {
        values: NestedValues { lower: 2, upper: 8 },
    };
    assert_eq!(plan.property(20), Ok(PropertyValue::Int(2)));
    assert_eq!(plan.property(21), Ok(PropertyValue::Int(8)));

    let replaced = plan
        .prepare_properties(&[
            PropertyAssignment {
                id: 20,
                value: PropertyValue::Int(3),
            },
            PropertyAssignment {
                id: 21,
                value: PropertyValue::Int(9),
            },
        ])
        .unwrap();
    assert_eq!(replaced.values, NestedValues { lower: 3, upper: 9 });
    assert_eq!(plan.values, NestedValues { lower: 2, upper: 8 });
}

#[test]
fn portable_binding_publishes_one_update_and_adopts_at_a_process_boundary() {
    let fixture = fixture();
    let plan = LeakyIntegratorPlan::new(fixture.initial.gain, fixture.initial.pole).unwrap();
    let mut binding = PropertyBinding::new(plan);
    let assignments = [
        PropertyAssignment {
            id: fixture.property.gain_id,
            value: PropertyValue::Float(fixture.update.gain),
        },
        PropertyAssignment {
            id: fixture.property.pole_id,
            value: PropertyValue::Float(fixture.update.pole),
        },
    ];
    let mut workspace = workspace_with_state(&fixture.initial.state);
    let mut output = vec![0.0; fixture.initial.output.len()];

    binding.prepare_properties(&[]).unwrap();
    binding.commit_properties();
    assert_eq!(binding.requested_generation(), 0);
    assert_eq!(binding.active_generation(), 0);

    binding
        .active_plan()
        .process(&mut output, &mut workspace, &fixture.initial.input)
        .unwrap();
    assert_eq!(output, fixture.initial.output);

    binding.prepare_properties(&assignments).unwrap();
    assert_eq!(binding.requested_generation(), 0);
    assert_eq!(binding.active_generation(), 0);
    assert_eq!(
        binding.prepare_properties(&assignments),
        Err(PropertyBindingError::Busy)
    );
    binding.commit_properties();
    assert_eq!(
        binding.requested_generation(),
        fixture.update.requested_generation
    );
    assert_eq!(binding.active_generation(), 0);
    assert_eq!(binding.requested_plan().gain(), fixture.update.gain);
    assert_eq!(binding.active_plan().gain(), fixture.initial.gain);
    assert_eq!(
        binding.prepare_properties(&assignments),
        Err(PropertyBindingError::Busy)
    );

    assert!(binding.adopt_properties());
    assert_eq!(
        binding.active_generation(),
        fixture.update.active_generation
    );
    assert!(!binding.adopt_properties());
    binding
        .active_plan()
        .process(&mut output, &mut workspace, &fixture.update.input)
        .unwrap();
    assert_eq!(output, fixture.update.output);

    binding.prepare_properties(&assignments).unwrap();
    binding.discard_properties();
    assert_eq!(
        binding.requested_generation(),
        fixture.update.requested_generation
    );
    assert_eq!(
        binding.active_generation(),
        fixture.update.active_generation
    );
}
