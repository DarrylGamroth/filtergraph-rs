//! Obligations derived from JFG's successful-adoption frame delay contract.
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmPlan, FrameDelayError, FrameDelayF32,
    FrameDelayF32Configuration, FrameDelayPlan, FrameDelayPreparationError, FrameDelayWorkspace,
    ProcessError,
};

macro_rules! delay_tests {
    ($name:ident, $scalar:ty) => {
        #[test]
        fn $name() {
            for delay in [0, 1, 3] {
                let plan = FrameDelayPlan::<$scalar>::new(delay);
                let initial = vec![-1.25, 2.5];
                let mut workspace = FrameDelayWorkspace::new(initial.clone(), &plan).unwrap();
                let mut output = [71.0; 2];
                let mut accepted: Vec<Vec<$scalar>> = Vec::new();
                for frame in 0..8 {
                    let input = [frame as $scalar + 0.25, -(frame as $scalar) - 0.5];
                    let before = workspace.clone();
                    for invalid in [<$scalar>::NAN, <$scalar>::INFINITY, <$scalar>::NEG_INFINITY] {
                        output.fill(71.0);
                        assert_eq!(
                            plan.process(&mut output, &mut workspace, &[input[0], invalid]),
                            Err(FrameDelayError::NonFiniteInput)
                        );
                        assert_eq!(output, [71.0; 2]);
                        assert_eq!(workspace, before);
                    }
                    plan.process(&mut output, &mut workspace, &input).unwrap();
                    let expected = if delay == 0 {
                        input.to_vec()
                    } else if frame < delay {
                        initial.clone()
                    } else {
                        accepted[frame - delay].clone()
                    };
                    assert_eq!(output.as_slice(), expected);
                    accepted.push(input.to_vec());
                    assert_eq!(
                        workspace.head(),
                        if delay == 0 { 0 } else { (frame + 1) % delay }
                    );
                }
                workspace.reset();
                assert_eq!(workspace.head(), 0);
                for history in workspace.history().chunks_exact(2) {
                    assert_eq!(history, initial);
                }
                plan.process(&mut output, &mut workspace, &[4.0, -3.0])
                    .unwrap();
                assert_eq!(
                    output.as_slice(),
                    if delay == 0 {
                        &[4.0, -3.0]
                    } else {
                        initial.as_slice()
                    }
                );
            }
        }
    };
}
delay_tests!(f32_delay_rejection_wrap_and_reset, f32);
delay_tests!(f64_delay_rejection_wrap_and_reset, f64);

#[test]
fn malformed_calls_preserve_output_and_history() {
    let plan = FrameDelayPlan::<f32>::new(2);
    let mut workspace = FrameDelayWorkspace::filled(2, 1.25, &plan).unwrap();
    let before = workspace.clone();
    let mut output = [71.0; 2];
    assert_eq!(
        plan.process(&mut output, &mut workspace, &[1.0]),
        Err(FrameDelayError::Process(ProcessError::InputExtent {
            expected: 2,
            actual: 1
        }))
    );
    assert_eq!(
        plan.process(&mut output[..1], &mut workspace, &[1.0, 2.0]),
        Err(FrameDelayError::Process(ProcessError::OutputExtent {
            expected: 2,
            actual: 1
        }))
    );
    assert_eq!(
        FrameDelayPlan::new(1).process(&mut output, &mut workspace, &[1.0, 2.0]),
        Err(FrameDelayError::HistoryExtent)
    );
    assert_eq!(workspace, before);
    assert_eq!(output, [71.0; 2]);
}

fn configuration(
    extent: usize,
    delay_frames: usize,
    initial_value: f32,
) -> FrameDelayF32Configuration {
    FrameDelayF32Configuration {
        extent,
        delay_frames,
        initial_value,
        input_schema: "pipewireao/controller-command-f32/1".into(),
        output_schema: "pipewireao/controller-command-f32/1".into(),
    }
}

#[test]
fn declaration_checks_preparation_before_allocating_history() {
    for (extent, delay, value, expected) in [
        (0, 1, 0.0, FrameDelayPreparationError::EmptyState),
        (
            1,
            1,
            f32::NAN,
            FrameDelayPreparationError::NonFiniteInitialState,
        ),
        (
            1,
            1,
            f32::INFINITY,
            FrameDelayPreparationError::NonFiniteInitialState,
        ),
        (
            1,
            u32::MAX as usize / 4 + 1,
            0.0,
            FrameDelayPreparationError::HistoryExtent,
        ),
        (
            usize::MAX,
            2,
            0.0,
            FrameDelayPreparationError::HistoryExtent,
        ),
    ] {
        let failure = FrameDelayF32::prepare(configuration(extent, delay, value))
            .err()
            .expect("invalid configuration accepted");
        assert_eq!(
            failure,
            filtergraph_algorithms::DeclarationPreparationError::Algorithm(expected)
        );
    }
    let mut prepared = FrameDelayF32::prepare(configuration(2, 1, 1.25)).unwrap();
    let mut output = [71.0; 2];
    FrameDelayF32::process(&mut prepared, &[4.0, -3.0], &mut output).unwrap();
    assert_eq!(output, [1.25; 2]);
    FrameDelayF32::process(&mut prepared, &[5.0, -6.0], &mut output).unwrap();
    assert_eq!(output, [4.0, -3.0]);
    FrameDelayF32::reset(&mut prepared);
    FrameDelayF32::process(&mut prepared, &[5.0, -6.0], &mut output).unwrap();
    assert_eq!(output, [1.25; 2]);
    assert_eq!(prepared.formats()[0].shape(), &[2]);
    assert_eq!(
        prepared.formats()[1].schema(),
        "pipewireao/controller-command-f32/1"
    );
    assert_eq!(FrameDelayF32::PORTS[1].metadata_source(), Some("input"));
}

#[test]
fn impossible_native_capacities_return_errors_before_allocation_even_at_zero_delay() {
    let zero = FrameDelayPlan::<f64>::new(0);
    assert!(FrameDelayWorkspace::filled(usize::MAX, 0.0, &zero).is_err());
    let huge = FrameDelayPlan::<f64>::new(isize::MAX as usize / 8 + 1);
    assert!(FrameDelayWorkspace::new(vec![0.0], &huge).is_err());
    assert!(FrameDelayF32::prepare(configuration(usize::MAX, 0, 0.0)).is_err());
}
