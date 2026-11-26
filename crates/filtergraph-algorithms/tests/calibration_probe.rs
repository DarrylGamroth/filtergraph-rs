//! Deterministic structured calibration-probe tests.

use filtergraph_algorithms::schemas::{
    CALIBRATION_CYCLE_COMPLETE_V1, CALIBRATION_PATTERN_INDEX_V1, CALIBRATION_PROBE_AMPLITUDES_V1,
    CALIBRATION_PROBE_POLARITY_V1, MODAL_POSITIVE_COMMANDS_V1,
};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, CalibrationProbeError,
    CalibrationProbeOutput, CalibrationProbePlanError, CalibrationProbePolarity,
    CalibrationProbeStep, HadamardPushPullF32, HadamardPushPullF32Configuration,
    HadamardPushPullPlan, ModalPushPullF32, ModalPushPullF32Configuration, ModalPushPullPlan,
    ModalPushPullPlanError, ZonalPushPullF32, ZonalPushPullF32Configuration, ZonalPushPullPlan,
};

#[test]
fn modal_cycle_publishes_each_exact_command_and_its_negative() {
    let plan = ModalPushPullPlan::new(2, 3, [0.5_f64, -1.0, 0.0, -0.25, 0.5, 2.0]).unwrap();
    let mut workspace = plan.workspace();
    let mut excitation = [9.0; 3];
    let mut step = CalibrationProbeStep::default();
    let expected = [
        (
            [0.5, -1.0, 0.0],
            0,
            CalibrationProbePolarity::Positive,
            false,
        ),
        (
            [-0.5, 1.0, -0.0],
            0,
            CalibrationProbePolarity::Negative,
            false,
        ),
        (
            [-0.25, 0.5, 2.0],
            1,
            CalibrationProbePolarity::Positive,
            false,
        ),
        (
            [0.25, -0.5, -2.0],
            1,
            CalibrationProbePolarity::Negative,
            true,
        ),
    ];

    assert_eq!(plan.mode_count(), 2);
    assert_eq!(plan.corrector_coordinate_count(), 3);
    assert_eq!(plan.positive_commands(), [0.5, -1.0, 0.0, -0.25, 0.5, 2.0]);
    assert_eq!(plan.cycle_len(), expected.len());
    for (command, mode, polarity, completes_cycle) in expected {
        plan.process(
            CalibrationProbeOutput::new(&mut excitation, &mut step),
            &mut workspace,
            (),
        )
        .unwrap();
        assert_eq!(excitation, command);
        assert_eq!(step.pattern_index(), mode);
        assert_eq!(step.polarity(), polarity);
        assert_eq!(step.completes_cycle(), completes_cycle);
    }

    plan.process(
        CalibrationProbeOutput::new(&mut excitation, &mut step),
        &mut workspace,
        (),
    )
    .unwrap();
    assert_eq!(excitation, [0.5, -1.0, 0.0]);
    assert_eq!(step.pattern_index(), 0);
    assert_eq!(step.polarity(), CalibrationProbePolarity::Positive);
}

#[test]
fn modal_float_representations_follow_the_same_sequence() {
    let plan32 = ModalPushPullPlan::new(2, 2, [0.25_f32, -2.0, 1.5, 0.5]).unwrap();
    let plan64 = ModalPushPullPlan::new(2, 2, [0.25_f64, -2.0, 1.5, 0.5]).unwrap();
    let mut workspace32 = plan32.workspace();
    let mut workspace64 = plan64.workspace();
    let mut excitation32 = [0.0_f32; 2];
    let mut excitation64 = [0.0_f64; 2];
    let mut step32 = CalibrationProbeStep::default();
    let mut step64 = CalibrationProbeStep::default();

    for _ in 0..plan32.cycle_len() {
        plan32
            .process(
                CalibrationProbeOutput::new(&mut excitation32, &mut step32),
                &mut workspace32,
                (),
            )
            .unwrap();
        plan64
            .process(
                CalibrationProbeOutput::new(&mut excitation64, &mut step64),
                &mut workspace64,
                (),
            )
            .unwrap();
        assert_eq!(excitation32.map(f64::from), excitation64);
        assert_eq!(step32, step64);
    }
}

#[test]
fn modal_declaration_publishes_step_metadata_and_preserves_position_on_update() {
    let excitation_schema = "test.filtergraph/test-modal-excitation/1";
    let mut declared = ModalPushPullF32::prepare(ModalPushPullF32Configuration {
        mode_count: 2,
        corrector_coordinate_count: 2,
        initial_positive_commands: vec![1.0, 0.0, 0.0, 2.0],
        excitation_schema: excitation_schema.into(),
    })
    .unwrap();

    assert_eq!(ModalPushPullF32::NAME, "modal-push-pull-f32");
    let formats = declared.formats();
    assert_eq!(formats.len(), 5);
    assert_eq!(formats[0].shape(), &[2]);
    assert_eq!(formats[0].schema(), excitation_schema);
    assert_eq!(formats[1].schema(), CALIBRATION_PATTERN_INDEX_V1);
    assert_eq!(formats[2].schema(), CALIBRATION_PROBE_POLARITY_V1);
    assert_eq!(formats[3].schema(), CALIBRATION_CYCLE_COMPLETE_V1);
    assert_eq!(formats[4].shape(), &[2, 2]);
    assert_eq!(formats[4].schema(), MODAL_POSITIVE_COMMANDS_V1);

    let mut excitation = [0.0_f32; 2];
    let mut pattern_index = [u64::MAX];
    let mut polarity = [u8::MAX];
    let mut completes_cycle = [true];
    ModalPushPullF32::process(
        &mut declared,
        &mut excitation,
        &mut pattern_index,
        &mut polarity,
        &mut completes_cycle,
    )
    .unwrap();
    assert_eq!(excitation, [1.0, 0.0]);
    assert_eq!(pattern_index, [0]);
    assert_eq!(polarity, [0]);
    assert_eq!(completes_cycle, [false]);

    let replacement = [3.0_f32, 0.0, 0.0, 4.0];
    assert_eq!(
        declared.replace_parameter(4, AlgorithmInput::from(&replacement[..])),
        Some(Ok(()))
    );
    ModalPushPullF32::process(
        &mut declared,
        &mut excitation,
        &mut pattern_index,
        &mut polarity,
        &mut completes_cycle,
    )
    .unwrap();
    assert_eq!(excitation, [-3.0, -0.0]);
    assert_eq!(pattern_index, [0]);
    assert_eq!(polarity, [1]);
    assert_eq!(completes_cycle, [false]);

    ModalPushPullF32::reset(&mut declared);
    ModalPushPullF32::process(
        &mut declared,
        &mut excitation,
        &mut pattern_index,
        &mut polarity,
        &mut completes_cycle,
    )
    .unwrap();
    assert_eq!(excitation, [3.0, 0.0]);
    assert_eq!(pattern_index, [0]);
    assert_eq!(polarity, [0]);
    assert_eq!(completes_cycle, [false]);
}

#[test]
fn modal_preparation_rejects_invalid_command_matrices() {
    assert_eq!(
        ModalPushPullPlan::<f32>::new(0, 1, []),
        Err(ModalPushPullPlanError::EmptyDimension)
    );
    assert_eq!(
        ModalPushPullPlan::<f32>::new(1, 0, []),
        Err(ModalPushPullPlanError::EmptyDimension)
    );
    assert_eq!(
        ModalPushPullPlan::<f32>::new(2, 2, [1.0; 3]),
        Err(ModalPushPullPlanError::CommandExtent {
            expected: 4,
            actual: 3,
        })
    );
    assert_eq!(
        ModalPushPullPlan::<f32>::new(usize::MAX, 2, []),
        Err(ModalPushPullPlanError::ExtentOverflow)
    );
    assert_eq!(
        ModalPushPullPlan::new(2, 2, [1.0_f32, 0.0, 0.5, f32::NAN]),
        Err(ModalPushPullPlanError::NonFiniteCommand {
            mode: 1,
            corrector_coordinate: 1,
        })
    );
    assert_eq!(
        ModalPushPullPlan::new(2, 2, [1.0_f32, 0.0, -0.0, 0.0]),
        Err(ModalPushPullPlanError::ZeroCommand { mode: 1 })
    );
}

#[test]
fn rejected_modal_output_or_workspace_preserves_sequence_and_outputs() {
    let plan = ModalPushPullPlan::new(2, 2, [1.0_f32, 0.0, 0.0, 1.0]).unwrap();
    let other_plan = ModalPushPullPlan::new(1, 1, [1.0_f32]).unwrap();
    let mut workspace = plan.workspace();
    let mut other_workspace = other_plan.workspace();
    let initial = workspace;
    let mut short_output = [7.0];
    let mut step = CalibrationProbeStep::default();

    assert_eq!(
        plan.process(
            CalibrationProbeOutput::new(&mut short_output, &mut step),
            &mut workspace,
            (),
        ),
        Err(CalibrationProbeError::OutputExtent {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(short_output, [7.0]);
    assert_eq!(step, CalibrationProbeStep::default());
    assert_eq!(workspace, initial);

    let mut output = [8.0, 9.0];
    assert_eq!(
        plan.process(
            CalibrationProbeOutput::new(&mut output, &mut step),
            &mut other_workspace,
            (),
        ),
        Err(CalibrationProbeError::WorkspaceExtent)
    );
    assert_eq!(output, [8.0, 9.0]);
    assert_eq!(step, CalibrationProbeStep::default());
    assert_eq!(other_workspace, other_plan.workspace());
}

#[test]
fn zonal_cycle_publishes_ordered_exact_push_pull_pairs() {
    let plan = ZonalPushPullPlan::new([1.0_f64, 2.0]).unwrap();
    let mut workspace = plan.workspace();
    let mut excitation = [9.0; 2];
    let mut step = CalibrationProbeStep::default();
    let expected = [
        ([1.0, 0.0], 0, CalibrationProbePolarity::Positive, false),
        ([-1.0, 0.0], 0, CalibrationProbePolarity::Negative, false),
        ([0.0, 2.0], 1, CalibrationProbePolarity::Positive, false),
        ([0.0, -2.0], 1, CalibrationProbePolarity::Negative, true),
    ];

    assert_eq!(plan.cycle_len(), expected.len());
    for (command, pattern_index, polarity, completes_cycle) in expected {
        plan.process(
            CalibrationProbeOutput::new(&mut excitation, &mut step),
            &mut workspace,
            (),
        )
        .unwrap();
        assert_eq!(excitation, command);
        assert_eq!(step.pattern_index(), pattern_index);
        assert_eq!(step.polarity(), polarity);
        assert_eq!(step.completes_cycle(), completes_cycle);
    }

    plan.process(
        CalibrationProbeOutput::new(&mut excitation, &mut step),
        &mut workspace,
        (),
    )
    .unwrap();
    assert_eq!(excitation, [1.0, 0.0]);
    assert_eq!(step.pattern_index(), 0);
    assert_eq!(step.polarity(), CalibrationProbePolarity::Positive);
}

#[test]
fn zonal_declaration_preserves_sequence_across_amplitude_adoption_and_reset() {
    let excitation_schema = "test.filtergraph/test-zonal-excitation/1";
    let mut declared = ZonalPushPullF32::prepare(ZonalPushPullF32Configuration {
        amplitudes: vec![1.0, 2.0],
        excitation_schema: excitation_schema.into(),
    })
    .unwrap();

    assert_eq!(ZonalPushPullF32::NAME, "zonal-push-pull-f32");
    let formats = declared.formats();
    assert_eq!(formats.len(), 5);
    assert_eq!(formats[0].shape(), &[2]);
    assert_eq!(formats[0].schema(), excitation_schema);
    assert_eq!(formats[1].schema(), CALIBRATION_PATTERN_INDEX_V1);
    assert_eq!(formats[2].schema(), CALIBRATION_PROBE_POLARITY_V1);
    assert_eq!(formats[3].schema(), CALIBRATION_CYCLE_COMPLETE_V1);
    assert_eq!(formats[4].shape(), &[2]);
    assert_eq!(formats[4].schema(), CALIBRATION_PROBE_AMPLITUDES_V1);

    let mut excitation = [0.0_f32; 2];
    let mut pattern_index = [u64::MAX];
    let mut polarity = [u8::MAX];
    let mut completes_cycle = [true];
    ZonalPushPullF32::process(
        &mut declared,
        &mut excitation,
        &mut pattern_index,
        &mut polarity,
        &mut completes_cycle,
    )
    .unwrap();
    assert_eq!(excitation, [1.0, 0.0]);
    assert_eq!(pattern_index, [0]);
    assert_eq!(polarity, [0]);
    assert_eq!(completes_cycle, [false]);

    let replacement = [3.0_f32, 4.0];
    assert_eq!(
        declared.replace_parameter(4, AlgorithmInput::from(&replacement[..])),
        Some(Ok(()))
    );
    ZonalPushPullF32::process(
        &mut declared,
        &mut excitation,
        &mut pattern_index,
        &mut polarity,
        &mut completes_cycle,
    )
    .unwrap();
    assert_eq!(excitation, [-3.0, 0.0]);
    assert_eq!(pattern_index, [0]);
    assert_eq!(polarity, [1]);
    assert_eq!(completes_cycle, [false]);

    ZonalPushPullF32::reset(&mut declared);
    ZonalPushPullF32::process(
        &mut declared,
        &mut excitation,
        &mut pattern_index,
        &mut polarity,
        &mut completes_cycle,
    )
    .unwrap();
    assert_eq!(excitation, [3.0, 0.0]);
    assert_eq!(pattern_index, [0]);
    assert_eq!(polarity, [0]);
    assert_eq!(completes_cycle, [false]);
}

#[test]
fn hadamard_cycle_uses_non_dc_columns_and_exact_row_negatives() {
    let plan = HadamardPushPullPlan::new([1.0_f64, 2.0, 3.0]).unwrap();
    let mut workspace = plan.workspace();
    let mut excitation = [0.0; 3];
    let mut step = CalibrationProbeStep::default();
    let rows = [
        [1.0, 2.0, 3.0],
        [-1.0, 2.0, -3.0],
        [1.0, -2.0, -3.0],
        [-1.0, -2.0, 3.0],
    ];

    assert_eq!(plan.order(), rows.len());
    assert_eq!(plan.cycle_len(), rows.len() * 2);
    for (row, expected) in rows.into_iter().enumerate() {
        plan.process(
            CalibrationProbeOutput::new(&mut excitation, &mut step),
            &mut workspace,
            (),
        )
        .unwrap();
        assert_eq!(excitation, expected);
        assert_eq!(step.pattern_index(), row);
        assert_eq!(step.polarity(), CalibrationProbePolarity::Positive);
        assert!(!step.completes_cycle());

        plan.process(
            CalibrationProbeOutput::new(&mut excitation, &mut step),
            &mut workspace,
            (),
        )
        .unwrap();
        assert_eq!(
            excitation,
            expected.map(|coefficient| -coefficient),
            "row {row} negative"
        );
        assert_eq!(step.pattern_index(), row);
        assert_eq!(step.polarity(), CalibrationProbePolarity::Negative);
        assert_eq!(step.completes_cycle(), row + 1 == plan.order());
    }
}

#[test]
fn hadamard_declaration_preserves_sequence_across_amplitude_adoption_and_reset() {
    let excitation_schema = "test.filtergraph/test-hadamard-excitation/1";
    let mut declared = HadamardPushPullF32::prepare(HadamardPushPullF32Configuration {
        amplitudes: vec![1.0, 2.0, 3.0],
        excitation_schema: excitation_schema.into(),
    })
    .unwrap();

    assert_eq!(HadamardPushPullF32::NAME, "hadamard-push-pull-f32");
    let formats = declared.formats();
    assert_eq!(formats.len(), 5);
    assert_eq!(formats[0].shape(), &[3]);
    assert_eq!(formats[0].schema(), excitation_schema);
    assert_eq!(formats[1].schema(), CALIBRATION_PATTERN_INDEX_V1);
    assert_eq!(formats[2].schema(), CALIBRATION_PROBE_POLARITY_V1);
    assert_eq!(formats[3].schema(), CALIBRATION_CYCLE_COMPLETE_V1);
    assert_eq!(formats[4].shape(), &[3]);
    assert_eq!(formats[4].schema(), CALIBRATION_PROBE_AMPLITUDES_V1);

    let mut excitation = [0.0_f32; 3];
    let mut pattern_index = [u64::MAX];
    let mut polarity = [u8::MAX];
    let mut completes_cycle = [true];
    HadamardPushPullF32::process(
        &mut declared,
        &mut excitation,
        &mut pattern_index,
        &mut polarity,
        &mut completes_cycle,
    )
    .unwrap();
    assert_eq!(excitation, [1.0, 2.0, 3.0]);
    assert_eq!(pattern_index, [0]);
    assert_eq!(polarity, [0]);
    assert_eq!(completes_cycle, [false]);

    let replacement = [3.0_f32, 4.0, 5.0];
    assert_eq!(
        declared.replace_parameter(4, AlgorithmInput::from(&replacement[..])),
        Some(Ok(()))
    );
    HadamardPushPullF32::process(
        &mut declared,
        &mut excitation,
        &mut pattern_index,
        &mut polarity,
        &mut completes_cycle,
    )
    .unwrap();
    assert_eq!(excitation, [-3.0, -4.0, -5.0]);
    assert_eq!(pattern_index, [0]);
    assert_eq!(polarity, [1]);
    assert_eq!(completes_cycle, [false]);

    HadamardPushPullF32::reset(&mut declared);
    HadamardPushPullF32::process(
        &mut declared,
        &mut excitation,
        &mut pattern_index,
        &mut polarity,
        &mut completes_cycle,
    )
    .unwrap();
    assert_eq!(excitation, [3.0, 4.0, 5.0]);
    assert_eq!(pattern_index, [0]);
    assert_eq!(polarity, [0]);
    assert_eq!(completes_cycle, [false]);
}

#[test]
fn complete_hadamard_cycle_is_zero_mean_and_orthogonal() {
    let plan = HadamardPushPullPlan::new([1.0_f64, 0.5, 2.0, 1.5, 3.0]).unwrap();
    let mut workspace = plan.workspace();
    let mut excitation = [0.0; 5];
    let mut step = CalibrationProbeStep::default();
    let mut sum = [0.0; 5];
    let mut gram = [[0.0; 5]; 5];

    for _ in 0..plan.cycle_len() {
        plan.process(
            CalibrationProbeOutput::new(&mut excitation, &mut step),
            &mut workspace,
            (),
        )
        .unwrap();
        for first in 0..excitation.len() {
            sum[first] += excitation[first];
            for second in 0..excitation.len() {
                gram[first][second] += excitation[first] * excitation[second];
            }
        }
    }

    assert_eq!(sum, [0.0; 5]);
    for (first, row) in gram.iter().enumerate() {
        for (second, value) in row.iter().enumerate() {
            if first == second {
                assert_eq!(
                    *value,
                    plan.cycle_len() as f64 * plan.amplitudes()[first].powi(2)
                );
            } else {
                assert_eq!(*value, 0.0);
            }
        }
    }
    assert!(step.completes_cycle());
}

#[test]
fn f32_and_f64_structured_sequences_are_identical() {
    let plan32 = HadamardPushPullPlan::new([0.25_f32, 2.0]).unwrap();
    let plan64 = HadamardPushPullPlan::new([0.25_f64, 2.0]).unwrap();
    let mut workspace32 = plan32.workspace();
    let mut workspace64 = plan64.workspace();
    let mut excitation32 = [0.0; 2];
    let mut excitation64 = [0.0; 2];
    let mut step32 = CalibrationProbeStep::default();
    let mut step64 = CalibrationProbeStep::default();

    for _ in 0..plan32.cycle_len() {
        plan32
            .process(
                CalibrationProbeOutput::new(&mut excitation32, &mut step32),
                &mut workspace32,
                (),
            )
            .unwrap();
        plan64
            .process(
                CalibrationProbeOutput::new(&mut excitation64, &mut step64),
                &mut workspace64,
                (),
            )
            .unwrap();
        assert_eq!(excitation32[0] as f64, excitation64[0]);
        assert_eq!(excitation32[1] as f64, excitation64[1]);
        assert_eq!(step32, step64);
    }

    let zonal32 = ZonalPushPullPlan::new([0.25_f32, 2.0]).unwrap();
    let zonal64 = ZonalPushPullPlan::new([0.25_f64, 2.0]).unwrap();
    let mut zonal_workspace32 = zonal32.workspace();
    let mut zonal_workspace64 = zonal64.workspace();
    for _ in 0..zonal32.cycle_len() {
        zonal32
            .process(
                CalibrationProbeOutput::new(&mut excitation32, &mut step32),
                &mut zonal_workspace32,
                (),
            )
            .unwrap();
        zonal64
            .process(
                CalibrationProbeOutput::new(&mut excitation64, &mut step64),
                &mut zonal_workspace64,
                (),
            )
            .unwrap();
        assert_eq!(excitation32[0] as f64, excitation64[0]);
        assert_eq!(excitation32[1] as f64, excitation64[1]);
        assert_eq!(step32, step64);
    }
}

#[test]
fn preparation_rejects_empty_and_invalid_amplitudes() {
    assert_eq!(
        ZonalPushPullPlan::<f32>::new([]),
        Err(CalibrationProbePlanError::Empty)
    );
    assert_eq!(
        HadamardPushPullPlan::<f32>::new([]),
        Err(CalibrationProbePlanError::Empty)
    );
    for amplitude in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
        let expected = CalibrationProbePlanError::InvalidAmplitude {
            corrector_coordinate: 1,
        };
        let zonal = ZonalPushPullPlan::new([1.0, amplitude]).unwrap_err();
        let hadamard = HadamardPushPullPlan::new([1.0, amplitude]).unwrap_err();
        for error in [zonal, hadamard] {
            assert_eq!(error, expected);
            assert!(
                error
                    .to_string()
                    .contains("corrector coordinate 1 must be finite and positive")
            );
        }
    }
}

#[test]
fn rejected_output_or_workspace_preserves_zonal_sequence_and_outputs() {
    let plan = ZonalPushPullPlan::new([1.0_f32, 2.0]).unwrap();
    let other_plan = ZonalPushPullPlan::new([1.0_f32]).unwrap();
    let mut workspace = plan.workspace();
    let mut other_workspace = other_plan.workspace();
    let initial = workspace;
    let mut short_output = [7.0];
    let mut step = CalibrationProbeStep::default();

    let error = plan
        .process(
            CalibrationProbeOutput::new(&mut short_output, &mut step),
            &mut workspace,
            (),
        )
        .unwrap_err();
    assert_eq!(
        error,
        CalibrationProbeError::OutputExtent {
            expected: 2,
            actual: 1
        }
    );
    assert!(
        error
            .to_string()
            .contains("output extent 1 does not match 2")
    );
    assert_eq!(short_output, [7.0]);
    assert_eq!(step, CalibrationProbeStep::default());
    assert_eq!(workspace, initial);

    let mut output = [8.0, 9.0];
    assert_eq!(
        plan.process(
            CalibrationProbeOutput::new(&mut output, &mut step),
            &mut other_workspace,
            (),
        ),
        Err(CalibrationProbeError::WorkspaceExtent)
    );
    assert_eq!(output, [8.0, 9.0]);
    assert_eq!(step, CalibrationProbeStep::default());
    assert_eq!(other_workspace, other_plan.workspace());
}

#[test]
fn rejected_output_or_workspace_preserves_hadamard_sequence_and_outputs() {
    let plan = HadamardPushPullPlan::new([1.0_f32, 2.0]).unwrap();
    let other_plan = HadamardPushPullPlan::new([1.0_f32]).unwrap();
    let mut workspace = plan.workspace();
    let mut other_workspace = other_plan.workspace();
    let initial = workspace;
    let mut short_output = [7.0];
    let mut step = CalibrationProbeStep::default();

    assert_eq!(
        plan.process(
            CalibrationProbeOutput::new(&mut short_output, &mut step),
            &mut workspace,
            (),
        ),
        Err(CalibrationProbeError::OutputExtent {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(short_output, [7.0]);
    assert_eq!(step, CalibrationProbeStep::default());
    assert_eq!(workspace, initial);

    let mut output = [8.0, 9.0];
    assert_eq!(
        plan.process(
            CalibrationProbeOutput::new(&mut output, &mut step),
            &mut other_workspace,
            (),
        ),
        Err(CalibrationProbeError::WorkspaceExtent)
    );
    assert_eq!(output, [8.0, 9.0]);
    assert_eq!(step, CalibrationProbeStep::default());
    assert_eq!(other_workspace, other_plan.workspace());
}
