//! Physical inter-actuator stroke projection tests.

use std::sync::Arc;

use filtergraph_algorithms::schemas::{INTER_ACTUATOR_STROKE_LIMIT_V1, REQUESTED_PDM_COMMAND_V1};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, DeclarationParameterError,
    InterActuatorStrokeLimit, InterActuatorStrokeLimitError, InterActuatorStrokeLimitF32,
    InterActuatorStrokeLimitF32Configuration, InterActuatorStrokeLimitPlan,
    InterActuatorStrokeLimitPlanError, NoWorkspace,
};

#[test]
fn preparation_validates_the_undirected_physical_graph() {
    assert_eq!(
        InterActuatorStrokeLimitPlan::new(
            0,
            Arc::from([InterActuatorStrokeLimit::new(0, 1, 1.0_f32)])
        ),
        Err(InterActuatorStrokeLimitPlanError::EmptyActuatorSet)
    );
    assert_eq!(
        InterActuatorStrokeLimitPlan::<f32>::new(2, Arc::from([])),
        Err(InterActuatorStrokeLimitPlanError::EmptyGraph)
    );
    assert_eq!(
        InterActuatorStrokeLimitPlan::new(
            2,
            Arc::from([InterActuatorStrokeLimit::new(0, 2, 1.0_f32)])
        ),
        Err(InterActuatorStrokeLimitPlanError::ActuatorIndex)
    );
    assert_eq!(
        InterActuatorStrokeLimitPlan::new(
            2,
            Arc::from([InterActuatorStrokeLimit::new(1, 1, 1.0_f32)])
        ),
        Err(InterActuatorStrokeLimitPlanError::SelfEdge)
    );
    assert_eq!(
        InterActuatorStrokeLimitPlan::new(
            2,
            Arc::from([
                InterActuatorStrokeLimit::new(0, 1, 1.0_f32),
                InterActuatorStrokeLimit::new(1, 0, 0.5),
            ])
        ),
        Err(InterActuatorStrokeLimitPlanError::DuplicateEdge)
    );
    for limit in [-1.0_f32, f32::NAN, f32::INFINITY] {
        assert_eq!(
            InterActuatorStrokeLimitPlan::new(
                2,
                Arc::from([InterActuatorStrokeLimit::new(0, 1, limit)])
            ),
            Err(InterActuatorStrokeLimitPlanError::InvalidLimit)
        );
    }
}

#[test]
fn an_unviolated_graph_preserves_the_command() {
    let plan = InterActuatorStrokeLimitPlan::new(
        3,
        Arc::from([
            InterActuatorStrokeLimit::new(0, 1, 0.5_f32),
            InterActuatorStrokeLimit::new(1, 2, 0.5),
        ]),
    )
    .unwrap();
    let input = [0.4, 0.1, -0.2];
    let mut output = [0.0; 3];
    plan.process(&mut output, &mut NoWorkspace, &input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn the_most_restrictive_edge_sets_one_shape_preserving_gain() {
    let plan = InterActuatorStrokeLimitPlan::new(
        3,
        Arc::from([
            InterActuatorStrokeLimit::new(0, 1, 1.0_f64),
            InterActuatorStrokeLimit::new(1, 2, 0.25),
        ]),
    )
    .unwrap();
    let mut output = [0.0; 3];
    plan.process(&mut output, &mut NoWorkspace, &[2.0, -2.0, 1.0])
        .unwrap();

    assert_eq!(output, [1.0 / 6.0, -1.0 / 6.0, 1.0 / 12.0]);
    assert!((output[0] - output[1]).abs() <= 1.0);
    assert!((output[1] - output[2]).abs() <= 0.25);
}

#[test]
fn a_violated_zero_limit_maps_the_command_to_zero() {
    let plan = InterActuatorStrokeLimitPlan::new(
        2,
        Arc::from([InterActuatorStrokeLimit::new(0, 1, 0.0_f32)]),
    )
    .unwrap();
    let mut output = [1.0; 2];
    plan.process(&mut output, &mut NoWorkspace, &[1.0, -1.0])
        .unwrap();
    assert_eq!(output, [0.0; 2]);
}

#[test]
fn large_opposite_float64_commands_are_limited_without_overflow() {
    let plan = InterActuatorStrokeLimitPlan::new(
        2,
        Arc::from([InterActuatorStrokeLimit::new(0, 1, 1.0_f64)]),
    )
    .unwrap();
    let mut output = [0.0; 2];
    plan.process(&mut output, &mut NoWorkspace, &[f64::MAX, -f64::MAX])
        .unwrap();
    assert!(output.iter().all(|value| value.is_finite()));
    assert!((output[0] - 0.5).abs() <= 8.0 * f64::EPSILON);
    assert!((output[1] + 0.5).abs() <= 8.0 * f64::EPSILON);
}

#[test]
fn execution_rejects_extent_and_nonfinite_input() {
    let plan = InterActuatorStrokeLimitPlan::new(
        2,
        Arc::from([InterActuatorStrokeLimit::new(0, 1, 1.0_f32)]),
    )
    .unwrap();
    assert_eq!(
        plan.process(&mut [0.0; 2], &mut NoWorkspace, &[0.0]),
        Err(InterActuatorStrokeLimitError::InputExtent)
    );
    assert_eq!(
        plan.process(&mut [0.0], &mut NoWorkspace, &[0.0; 2]),
        Err(InterActuatorStrokeLimitError::OutputExtent)
    );
    assert_eq!(
        plan.process(&mut [0.0; 2], &mut NoWorkspace, &[f32::NAN, 0.0]),
        Err(InterActuatorStrokeLimitError::NonFiniteInput)
    );
}

#[test]
fn separate_edge_and_limit_preparation_rejects_extent_mismatch() {
    assert_eq!(
        InterActuatorStrokeLimitPlan::<f32>::from_edges_and_limits(3, &[(0, 1), (1, 2)], &[1.0],),
        Err(InterActuatorStrokeLimitPlanError::LimitExtent {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn declaration_keeps_topology_in_configuration_and_limits_on_one_parameter() {
    let mut declared =
        InterActuatorStrokeLimitF32::prepare(InterActuatorStrokeLimitF32Configuration {
            actuator_count: 3,
            edges: vec![[0, 1], [1, 2]],
            max_differences: vec![1.0, 0.25],
        })
        .unwrap();
    assert_eq!(
        InterActuatorStrokeLimitF32::NAME,
        "inter-actuator-stroke-limit-f32"
    );
    assert_eq!(declared.formats()[0].shape(), &[3]);
    assert_eq!(declared.formats()[0].schema(), REQUESTED_PDM_COMMAND_V1);
    assert_eq!(declared.formats()[1].shape(), &[3]);
    assert_eq!(declared.formats()[1].schema(), REQUESTED_PDM_COMMAND_V1);
    assert_eq!(declared.formats()[2].shape(), &[2]);
    assert_eq!(
        declared.formats()[2].schema(),
        INTER_ACTUATOR_STROKE_LIMIT_V1
    );

    let requested = [2.0_f32, -2.0, 1.0];
    let mut limited = [0.0_f32; 3];
    InterActuatorStrokeLimitF32::process(&mut declared, &requested, &mut limited).unwrap();
    assert_eq!(limited, [1.0 / 6.0, -1.0 / 6.0, 1.0 / 12.0]);

    let relaxed = [4.0_f32, 4.0];
    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&relaxed[..])),
        Some(Ok(()))
    );
    InterActuatorStrokeLimitF32::process(&mut declared, &requested, &mut limited).unwrap();
    assert_eq!(limited, requested);

    let invalid = [4.0_f32, -1.0];
    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&invalid[..])),
        Some(Err(DeclarationParameterError::Algorithm(
            InterActuatorStrokeLimitPlanError::InvalidLimit
        )))
    );
    InterActuatorStrokeLimitF32::process(&mut declared, &requested, &mut limited).unwrap();
    assert_eq!(limited, requested);
}
