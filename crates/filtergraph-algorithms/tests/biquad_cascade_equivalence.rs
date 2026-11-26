//! Source-derived CPU cascade contracts; live Julia evidence is a separate opt-in check.
use filtergraph_algorithms::{
    AlgorithmInput, AlgorithmPlan, ArrayLayout, BiquadCascadeError, BiquadCascadeF32,
    BiquadCascadeF32Configuration, BiquadCascadePlan, BiquadCascadePlanError,
};

#[test]
fn impulse_history_and_zero_reset_preserve_df2t_order() {
    let plan = BiquadCascadePlan::new(vec![0.5_f64, 0.25, 0.125, -0.5, 0.125], 1, 1, 1).unwrap();
    let mut workspace = plan.workspace();
    let mut output = [71.0];
    for (input, expected) in [1.0, 0.0, 0.0, 0.0, 0.0]
        .into_iter()
        .zip([0.5, 0.5, 0.3125, 0.09375, 0.0078125])
    {
        plan.process(&mut output, &mut workspace, &[input]).unwrap();
        assert_eq!(output, [expected]);
    }
    assert!(workspace.state().iter().any(|&v| v != 0.0));
    workspace.reset();
    assert_eq!(workspace.state(), [0.0, 0.0]);
    plan.process(&mut output, &mut workspace, &[1.0]).unwrap();
    assert_eq!(output, [0.5]);
}

#[test]
fn strict_schur_conditions_and_shapes_are_checked_in_the_selected_precision() {
    for coefficients in [
        vec![1.0_f32, 0.0, 0.0, 1.5, 0.5],
        vec![1.0, 0.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 0.0, -1.0],
    ] {
        assert_eq!(
            BiquadCascadePlan::new(coefficients, 2, 1, 1).unwrap_err(),
            BiquadCascadePlanError::UnstableDenominator
        );
    }
    assert_eq!(
        BiquadCascadePlan::new(vec![1.0_f32, 0.0, f32::NAN, 0.0, 0.0], 2, 1, 1).unwrap_err(),
        BiquadCascadePlanError::NonFiniteCoefficient
    );
    assert_eq!(
        BiquadCascadePlan::new(vec![1.0_f32; 10], 3, 1, 2).unwrap_err(),
        BiquadCascadePlanError::CoefficientSetCount
    );
    assert!(BiquadCascadePlan::<f32>::new(Vec::new(), 0, 1, 1).is_err());
    assert!(BiquadCascadePlan::<f32>::new(Vec::new(), 1, 0, 1).is_err());
    assert_eq!(
        BiquadCascadePlan::<f32>::new(Vec::new(), usize::MAX, 2, 1).unwrap_err(),
        BiquadCascadePlanError::ExtentOverflow
    );
}

#[test]
fn last_coordinate_numerical_failure_preserves_output_and_committed_history() {
    let plan = BiquadCascadePlan::new(vec![0.5_f32, 0.25, 0.125, -0.5, 0.125], 3, 1, 1).unwrap();
    let mut workspace = plan.workspace();
    let mut output = [0.0; 3];
    plan.process(&mut output, &mut workspace, &[1.0, 2.0, 3.0])
        .unwrap();
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let previous = workspace.state().to_vec();
        let mut expected = workspace.clone();
        output.fill(71.0);
        assert_eq!(
            plan.process(&mut output, &mut workspace, &[4.0, 5.0, invalid]),
            Err(BiquadCascadeError::NonFiniteInput)
        );
        assert_eq!(output, [71.0; 3]);
        assert_eq!(workspace.state(), previous);
        let mut reference = [0.0; 3];
        plan.process(&mut reference, &mut expected, &[0.5, -1.0, 2.0])
            .unwrap();
        plan.process(&mut output, &mut workspace, &[0.5, -1.0, 2.0])
            .unwrap();
        assert_eq!(output, reference);
        assert_eq!(workspace.state(), expected.state());
    }
    for tuple in [
        [f32::MAX, 0.0, 0.0, 0.0, 0.0],
        [0.0, f32::MAX, 0.0, 0.0, 0.0],
    ] {
        let overflow = BiquadCascadePlan::new(tuple.to_vec(), 3, 1, 1).unwrap();
        let mut workspace = overflow.workspace();
        let before = workspace.state().to_vec();
        output.fill(71.0);
        assert_eq!(
            overflow.process(&mut output, &mut workspace, &[0.0, 0.0, 2.0]),
            Err(BiquadCascadeError::NonFiniteOutput)
        );
        assert_eq!(output, [71.0; 3]);
        assert_eq!(workspace.state(), before);
        overflow
            .process(&mut output, &mut workspace, &[0.0; 3])
            .unwrap();
        assert_eq!(output, [0.0; 3]);
    }
}

#[test]
fn mode_regions_cover_once_and_pad_short_cascades_with_identity() {
    let regions = vec![
        (
            2..3,
            vec![[2.0_f32, 0.0, 0.0, 0.0, 0.0], [0.75, 0.0, 0.0, 0.0, 0.0]],
        ),
        (0..2, vec![[0.5, 0.0, 0.0, 0.0, 0.0]]),
    ];
    let plan = BiquadCascadePlan::from_regions(&regions, 3).unwrap();
    assert_eq!(plan.section_count(), 2);
    assert_eq!(plan.coefficient_set_count(), 3);
    assert_eq!(&plan.coefficients()[5..10], &[1.0, 0.0, 0.0, 0.0, 0.0]);
    let mut output = [0.0; 3];
    plan.process(&mut output, &mut plan.workspace(), &[1.0, 2.0, 3.0])
        .unwrap();
    assert_eq!(output, [0.5, 1.0, 4.5]);
    assert_eq!(
        BiquadCascadePlan::from_regions(
            &[
                (0..2, vec![[1.0_f32, 0.0, 0.0, 0.0, 0.0]]),
                (1..3, vec![[1.0; 5]])
            ],
            3
        )
        .unwrap_err(),
        BiquadCascadePlanError::OverlappingRegions
    );
    assert_eq!(
        BiquadCascadePlan::from_regions(&regions[0..1], 3).unwrap_err(),
        BiquadCascadePlanError::UncoveredMode
    );
    assert_eq!(
        BiquadCascadePlan::from_regions(&[(0..0, vec![[1.0_f32; 5]])], 3).unwrap_err(),
        BiquadCascadePlanError::InvalidRegion
    );
}

#[test]
fn parameter_replacement_maps_port_layout_and_resets_only_after_valid_adoption() {
    let mut prepared = BiquadCascadeF32::prepare(BiquadCascadeF32Configuration {
        extent: 2,
        section_count: 2,
        per_mode: true,
        initial_coefficients: Some(vec![
            0.5, 0.25, 0.125, -0.5, 0.125, 1.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.125, 0.0, -0.25, 0.0,
            0.5, 0.0, 0.0, 0.0, 0.0,
        ]),
        input_schema: "org.example.parity.input/1".into(),
        output_schema: "org.example.parity.output/1".into(),
    })
    .unwrap();
    assert_eq!(prepared.formats()[2].shape(), &[5, 2, 2]);
    assert_eq!(prepared.formats()[2].layout(), ArrayLayout::RowMajor);
    let mut output = [0.0; 2];
    BiquadCascadeF32::process(&mut prepared, &[1.0, 2.0], &mut output).unwrap();
    let previous = prepared.workspace().state().to_vec();
    let previous_coefficients = prepared.plan().coefficients().to_vec();
    let invalid = vec![f32::NAN; 20];
    assert!(
        prepared
            .replace_parameter(2, AlgorithmInput::from(invalid.as_slice()))
            .unwrap()
            .is_err()
    );
    assert_eq!(prepared.workspace().state(), previous);
    assert_eq!(prepared.plan().coefficients(), previous_coefficients);
    // Logical [coefficient,section,set] row-major replacement: two sections,
    // distinct first gains for each element, then distinct second-section gains.
    let row_major = vec![
        2.0_f32, 3.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ];
    prepared
        .replace_parameter(2, AlgorithmInput::from(row_major.as_slice()))
        .unwrap()
        .unwrap();
    assert!(prepared.workspace().state().iter().all(|&v| v == 0.0));
    assert_eq!(
        prepared.plan().coefficients(),
        &[
            2.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0,
            0.0, 0.0, 0.0
        ]
    );
    BiquadCascadeF32::process(&mut prepared, &[1.0, 2.0], &mut output).unwrap();
    assert_eq!(output, [1.0, 1.5]);
}

#[test]
fn incompatible_workspace_topology_and_extents_reject_before_mutation() {
    let plan = BiquadCascadePlan::new(
        vec![1.0_f32, 0.25, 0.0, -0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        2,
        2,
        1,
    )
    .unwrap();
    // Both state buffers contain eight values, but their element/section axes differ.
    let other = BiquadCascadePlan::new(vec![1.0_f32, 0.0, 0.0, 0.0, 0.0], 4, 1, 1).unwrap();
    let mut wrong = other.workspace();
    let before = wrong.clone();
    let mut output = [71.0; 2];
    assert_eq!(
        plan.process(&mut output, &mut wrong, &[1.0, 2.0]),
        Err(BiquadCascadeError::WorkspaceExtent)
    );
    assert_eq!(output, [71.0; 2]);
    assert_eq!(wrong, before);
    let mut workspace = plan.workspace();
    let before = workspace.clone();
    assert!(plan.process(&mut output, &mut workspace, &[1.0]).is_err());
    assert_eq!(output, [71.0; 2]);
    assert_eq!(workspace, before);
    assert!(
        plan.process(&mut output[..1], &mut workspace, &[1.0, 2.0])
            .is_err()
    );
    assert_eq!(output, [71.0; 2]);
    assert_eq!(workspace, before);
    assert_eq!(
        BiquadCascadePlan::<f32>::new(vec![1.0; 4], 1, 1, 1).unwrap_err(),
        BiquadCascadePlanError::CoefficientExtent
    );
    assert_eq!(
        BiquadCascadePlan::from_regions(&[(0..1, vec![[1.0_f32, 0.0, 0.0, 0.0, 0.0]])], usize::MAX)
            .unwrap_err(),
        BiquadCascadePlanError::ExtentOverflow
    );
}

// The accepted finite result requires the current JFG FMA grouping: separate
// multiplication overflows before cancellation in both cases.
#[test]
fn fma_overflow_cancellation_preserves_state_and_failure_transaction() {
    macro_rules! precision {
        ($t:ty) => {{
            for (coefficients, frames) in [
                (vec![<$t>::MAX * 0.5, 0.0, 0.0, 0.5, 0.0], vec![1.0, 2.25]),
                (
                    vec![<$t>::MAX * 0.5, <$t>::MAX * 0.5, 0.0, 1.5, 0.75],
                    vec![2.0],
                ),
            ] {
                let plan = BiquadCascadePlan::new(coefficients.clone(), 1, 1, 1).unwrap();
                let mut workspace = plan.workspace();
                let mut output = [71.0 as $t];
                let mut state = [0.0 as $t; 2];
                for input in frames {
                    let y = coefficients[0].mul_add(input, state[0]);
                    state = [
                        (-coefficients[3]).mul_add(y, coefficients[1] * input) + state[1],
                        (-coefficients[4]).mul_add(y, coefficients[2] * input),
                    ];
                    assert!(y.is_finite() && state.iter().all(|v| v.is_finite()));
                    plan.process(&mut output, &mut workspace, &[input]).unwrap();
                    assert_eq!(output, [y]);
                    assert_eq!(workspace.state(), state);
                }
                let committed = workspace.state().to_vec();
                let mut reference = workspace.clone();
                output.fill(71.0);
                assert_eq!(
                    plan.process(&mut output, &mut workspace, &[8.0]),
                    Err(BiquadCascadeError::NonFiniteOutput)
                );
                assert_eq!(output, [71.0]);
                assert_eq!(workspace.state(), committed);
                let mut expected = [0.0];
                plan.process(&mut expected, &mut reference, &[0.0]).unwrap();
                plan.process(&mut output, &mut workspace, &[0.0]).unwrap();
                assert_eq!(output, expected);
                assert_eq!(workspace.state(), reference.state());
                workspace.recover();
                assert_eq!(workspace.state(), [0.0, 0.0]);
            }
        }};
    }
    precision!(f32);
    precision!(f64);
}

#[test]
fn section_tiles_preserve_logical_modes_and_public_state_layout() {
    macro_rules! precision {
        ($t:ty) => {{
            let extent = 4097;
            let sections = 3;
            for sets in [1, extent] {
                let mut coefficients = Vec::new();
                for mode in 0..sets {
                    for section in 0..sections {
                        coefficients.extend_from_slice(&[
                            0.5 as $t + (mode % 7 + section) as $t / 64.0,
                            0.125,
                            -0.0625,
                            -0.25,
                            0.0625,
                        ]);
                    }
                }
                let plan =
                    BiquadCascadePlan::new(coefficients.clone(), extent, sections, sets).unwrap();
                let mut workspace = plan.workspace();
                let mut output = vec![71.0 as $t; extent];
                let mut state = vec![0.0 as $t; 2 * extent * sections];
                for frame in 0..3 {
                    let input = (0..extent)
                        .map(|mode| ((mode + frame) % 29) as $t / 16.0 - 0.75)
                        .collect::<Vec<_>>();
                    let mut expected = input.clone();
                    // Independent scalar mode traversal, unlike the section tiles.
                    for (mode, value) in expected.iter_mut().enumerate() {
                        let set = if sets == 1 { 0 } else { mode };
                        for section in 0..sections {
                            let c = &coefficients[5 * (section + sections * set)..];
                            let index = 2 * (mode + extent * section);
                            let y = c[0].mul_add(*value, state[index]);
                            state[index] = (-c[3]).mul_add(y, c[1] * *value) + state[index + 1];
                            state[index + 1] = (-c[4]).mul_add(y, c[2] * *value);
                            *value = y;
                        }
                    }
                    plan.process(&mut output, &mut workspace, &input).unwrap();
                    assert_eq!(output, expected);
                    assert_eq!(workspace.state(), state);
                }
            }
        }};
    }
    precision!(f32);
    precision!(f64);
}

#[test]
fn numerator_products_round_before_feedback_fma() {
    macro_rules! precision {
        ($t:ty) => {{
            for coefficients in [
                vec![<$t>::MAX * 0.25, <$t>::MAX * 0.5, 0.0, 1.5, 0.75],
                vec![<$t>::MAX * 0.25, 0.0, <$t>::MAX * 0.5, 0.0, 0.75],
            ] {
                let plan = BiquadCascadePlan::new(coefficients, 1, 1, 1).unwrap();
                let mut workspace = plan.workspace();
                let mut output = [71.0 as $t];
                assert_eq!(
                    plan.process(&mut output, &mut workspace, &[2.25]),
                    Err(BiquadCascadeError::NonFiniteOutput)
                );
                assert_eq!(output, [71.0]);
                assert_eq!(workspace.state(), [0.0, 0.0]);
            }
        }};
    }
    precision!(f32);
    precision!(f64);
}
