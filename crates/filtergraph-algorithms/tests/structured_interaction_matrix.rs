//! Direct structured-probe interaction-matrix fixtures.

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, CalibrationProbeOutput,
    CalibrationProbePlanError, CalibrationProbeStep, HadamardInteractionMatrixF32,
    HadamardInteractionMatrixF32Configuration, HadamardInteractionMatrixPlan, HadamardPushPullPlan,
    StructuredInteractionMatrixError, StructuredInteractionMatrixPlanError,
    ZonalInteractionMatrixF32, ZonalInteractionMatrixF32Configuration, ZonalInteractionMatrixPlan,
    ZonalPushPullPlan,
};
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
struct SharedFixture {
    schema: String,
    absolute_tolerance_f64: f64,
    zonal: SharedZonalFixture,
    hadamard: SharedHadamardFixture,
}

#[derive(Debug, Deserialize)]
struct SharedZonalFixture {
    measurement_count: usize,
    corrector_coordinate_count: usize,
    amplitudes: Vec<f64>,
    responses: Vec<f64>,
    expected_interaction_matrix: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct SharedHadamardFixture {
    measurement_count: usize,
    corrector_coordinate_count: usize,
    order: usize,
    amplitudes: Vec<f64>,
    responses: Vec<f64>,
    expected_interaction_matrix: Vec<f64>,
}

fn shared_fixture() -> SharedFixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/algorithms/structured_interaction_matrix.toml"
    );
    let source = fs::read_to_string(path).expect("read shared structured-decoder fixture");
    toml::from_str(&source).expect("parse shared structured-decoder fixture")
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn julia_and_rust_share_the_structured_decoder_equation_fixture() {
    let fixture = shared_fixture();
    assert_eq!(
        fixture.schema,
        "org.calculon.algorithm-fixture/structured-interaction-matrix/1"
    );

    let zonal = ZonalInteractionMatrixPlan::new(
        fixture.zonal.measurement_count,
        fixture.zonal.amplitudes.clone(),
    )
    .unwrap();
    assert_eq!(
        zonal.corrector_coordinate_count(),
        fixture.zonal.corrector_coordinate_count
    );
    assert_eq!(zonal.response_count(), fixture.zonal.responses.len());
    let mut zonal_workspace = zonal.workspace();
    let mut zonal_matrix = vec![f64::NAN; fixture.zonal.expected_interaction_matrix.len()];
    zonal
        .process(
            &mut zonal_matrix,
            &mut zonal_workspace,
            &fixture.zonal.responses,
        )
        .unwrap();
    assert_close(
        &zonal_matrix,
        &fixture.zonal.expected_interaction_matrix,
        fixture.absolute_tolerance_f64,
    );

    let hadamard = HadamardInteractionMatrixPlan::new(
        fixture.hadamard.measurement_count,
        fixture.hadamard.amplitudes.clone(),
    )
    .unwrap();
    assert_eq!(
        hadamard.corrector_coordinate_count(),
        fixture.hadamard.corrector_coordinate_count
    );
    assert_eq!(hadamard.order(), fixture.hadamard.order);
    assert_eq!(hadamard.response_count(), fixture.hadamard.responses.len());
    let mut hadamard_workspace = hadamard.workspace();
    let mut hadamard_matrix = vec![f64::NAN; fixture.hadamard.expected_interaction_matrix.len()];
    hadamard
        .process(
            &mut hadamard_matrix,
            &mut hadamard_workspace,
            &fixture.hadamard.responses,
        )
        .unwrap();
    assert_close(
        &hadamard_matrix,
        &fixture.hadamard.expected_interaction_matrix,
        fixture.absolute_tolerance_f64,
    );
}

fn sensor_response(
    interaction_matrix: &[f64],
    measurement_count: usize,
    corrector_count: usize,
    command: &[f64],
    offset: &[f64],
) -> Vec<f64> {
    (0..measurement_count)
        .map(|measurement| {
            offset[measurement]
                + (0..corrector_count)
                    .map(|corrector| {
                        interaction_matrix[measurement * corrector_count + corrector]
                            * command[corrector]
                    })
                    .sum::<f64>()
        })
        .collect()
}

#[test]
fn zonal_probe_cycle_recovers_a_known_interaction_matrix() {
    let amplitudes = [0.5_f64, 2.0, 1.5];
    let expected = [1.0, 2.0, -1.0, 0.5, -2.0, 4.0];
    let offset = [10.0, -7.0];
    let probe = ZonalPushPullPlan::new(amplitudes).unwrap();
    let decoder = ZonalInteractionMatrixPlan::new(2, amplitudes).unwrap();
    let mut probe_workspace = probe.workspace();
    let mut decoder_workspace = decoder.workspace();
    let mut command = [0.0; 3];
    let mut step = CalibrationProbeStep::default();
    let mut responses = Vec::with_capacity(decoder.response_count());

    for expected_step in 0..probe.cycle_len() {
        probe
            .process(
                CalibrationProbeOutput::new(&mut command, &mut step),
                &mut probe_workspace,
                (),
            )
            .unwrap();
        assert_eq!(step.pattern_index(), expected_step / 2);
        responses.extend(sensor_response(&expected, 2, 3, &command, &offset));
    }

    let mut actual = [0.0; 6];
    decoder
        .process(&mut actual, &mut decoder_workspace, &responses)
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn zonal_declaration_decodes_one_complete_cycle_and_adopts_amplitudes() {
    let response_schema = "test.filtergraph/test-zonal-response-cycle/1";
    let matrix_schema = "test.filtergraph/test-interaction-matrix/1";
    let mut declared = ZonalInteractionMatrixF32::prepare(ZonalInteractionMatrixF32Configuration {
        measurement_count: 2,
        amplitudes: vec![0.5, 2.0],
        response_schema: response_schema.into(),
        interaction_matrix_schema: matrix_schema.into(),
    })
    .unwrap();

    assert_eq!(
        ZonalInteractionMatrixF32::NAME,
        "zonal-interaction-matrix-f32"
    );
    let formats = declared.formats();
    assert_eq!(formats.len(), 3);
    assert_eq!(formats[0].shape(), &[4, 2]);
    assert_eq!(formats[0].schema(), response_schema);
    assert_eq!(formats[1].shape(), &[2, 2]);
    assert_eq!(formats[1].schema(), matrix_schema);
    assert_eq!(formats[2].shape(), &[2]);
    assert_eq!(
        formats[2].schema(),
        filtergraph_algorithms::schemas::CALIBRATION_PROBE_AMPLITUDES_V1
    );

    let responses = [3.0_f32, 5.0, 1.0, 1.0, 9.0, -1.0, 1.0, 3.0];
    let mut interaction_matrix = [0.0_f32; 4];
    ZonalInteractionMatrixF32::process(&mut declared, &responses, &mut interaction_matrix).unwrap();
    assert_eq!(interaction_matrix, [2.0, 2.0, 4.0, -1.0]);

    let replacement = [1.0_f32, 4.0];
    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&replacement[..])),
        Some(Ok(()))
    );
    ZonalInteractionMatrixF32::process(&mut declared, &responses, &mut interaction_matrix).unwrap();
    assert_eq!(interaction_matrix, [1.0, 1.0, 2.0, -0.5]);
}

#[test]
fn hadamard_probe_cycle_recovers_a_known_interaction_matrix() {
    let amplitudes = [0.25_f64, 2.0, 0.5];
    let expected = [1.0, -3.0, 2.0, 0.5, 4.0, -2.0];
    let offset = [100.0, -30.0];
    let probe = HadamardPushPullPlan::new(amplitudes).unwrap();
    let decoder = HadamardInteractionMatrixPlan::new(2, amplitudes).unwrap();
    let mut probe_workspace = probe.workspace();
    let mut decoder_workspace = decoder.workspace();
    let mut command = [0.0; 3];
    let mut step = CalibrationProbeStep::default();
    let mut responses = Vec::with_capacity(decoder.response_count());

    assert_eq!(decoder.order(), 4);
    for expected_step in 0..probe.cycle_len() {
        probe
            .process(
                CalibrationProbeOutput::new(&mut command, &mut step),
                &mut probe_workspace,
                (),
            )
            .unwrap();
        assert_eq!(step.pattern_index(), expected_step / 2);
        responses.extend(sensor_response(&expected, 2, 3, &command, &offset));
    }

    let mut actual = [0.0; 6];
    decoder
        .process(&mut actual, &mut decoder_workspace, &responses)
        .unwrap();
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

#[test]
fn hadamard_declaration_decodes_one_complete_cycle_and_adopts_amplitudes() {
    let response_schema = "test.filtergraph/test-hadamard-response-cycle/1";
    let matrix_schema = "test.filtergraph/test-interaction-matrix/1";
    let mut declared =
        HadamardInteractionMatrixF32::prepare(HadamardInteractionMatrixF32Configuration {
            measurement_count: 1,
            amplitudes: vec![0.5, 2.0],
            response_schema: response_schema.into(),
            interaction_matrix_schema: matrix_schema.into(),
        })
        .unwrap();

    assert_eq!(
        HadamardInteractionMatrixF32::NAME,
        "hadamard-interaction-matrix-f32"
    );
    let formats = declared.formats();
    assert_eq!(formats.len(), 3);
    assert_eq!(formats[0].shape(), &[8, 1]);
    assert_eq!(formats[0].schema(), response_schema);
    assert_eq!(formats[1].shape(), &[1, 2]);
    assert_eq!(formats[1].schema(), matrix_schema);
    assert_eq!(formats[2].shape(), &[2]);
    assert_eq!(
        formats[2].schema(),
        filtergraph_algorithms::schemas::CALIBRATION_PROBE_AMPLITUDES_V1
    );

    let responses = [6.0_f32, -6.0, -2.0, 2.0, 4.0, -4.0, 0.0, 0.0];
    let mut interaction_matrix = [0.0_f32; 2];
    HadamardInteractionMatrixF32::process(&mut declared, &responses, &mut interaction_matrix)
        .unwrap();
    assert_eq!(interaction_matrix, [6.0, 0.0]);

    let replacement = [1.0_f32, 4.0];
    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&replacement[..])),
        Some(Ok(()))
    );
    HadamardInteractionMatrixF32::process(&mut declared, &responses, &mut interaction_matrix)
        .unwrap();
    assert_eq!(interaction_matrix, [3.0, 0.0]);
}

#[test]
fn f32_and_f64_decoders_apply_the_same_pair_scaling() {
    let zonal32 = ZonalInteractionMatrixPlan::new(1, [0.5_f32, 2.0]).unwrap();
    let zonal64 = ZonalInteractionMatrixPlan::new(1, [0.5_f64, 2.0]).unwrap();
    let mut zonal_workspace32 = zonal32.workspace();
    let mut zonal_workspace64 = zonal64.workspace();
    let mut output32 = [0.0; 2];
    let mut output64 = [0.0; 2];
    zonal32
        .process(
            &mut output32,
            &mut zonal_workspace32,
            &[3.0, 1.0, 7.0, -1.0],
        )
        .unwrap();
    zonal64
        .process(
            &mut output64,
            &mut zonal_workspace64,
            &[3.0, 1.0, 7.0, -1.0],
        )
        .unwrap();
    assert_eq!(output32.map(f64::from), output64);

    let hadamard32 = HadamardInteractionMatrixPlan::new(1, [0.5_f32, 2.0]).unwrap();
    let hadamard64 = HadamardInteractionMatrixPlan::new(1, [0.5_f64, 2.0]).unwrap();
    let responses32 = [6.0_f32, -6.0, -2.0, 2.0, 4.0, -4.0, 0.0, 0.0];
    let responses64 = responses32.map(f64::from);
    let mut hadamard_workspace32 = hadamard32.workspace();
    let mut hadamard_workspace64 = hadamard64.workspace();
    hadamard32
        .process(&mut output32, &mut hadamard_workspace32, &responses32)
        .unwrap();
    hadamard64
        .process(&mut output64, &mut hadamard_workspace64, &responses64)
        .unwrap();
    assert_eq!(output32.map(f64::from), output64);
}

#[test]
fn preparation_rejects_empty_invalid_and_unusable_scales() {
    assert_eq!(
        ZonalInteractionMatrixPlan::<f32>::new(0, [1.0]),
        Err(StructuredInteractionMatrixPlanError::EmptyMeasurement)
    );
    assert_eq!(
        ZonalInteractionMatrixPlan::<f32>::new(1, []),
        Err(StructuredInteractionMatrixPlanError::InvalidProbe(
            CalibrationProbePlanError::Empty
        ))
    );
    assert_eq!(
        HadamardInteractionMatrixPlan::<f64>::new(1, [1.0, f64::NAN]),
        Err(StructuredInteractionMatrixPlanError::InvalidProbe(
            CalibrationProbePlanError::InvalidAmplitude {
                corrector_coordinate: 1
            }
        ))
    );
    assert_eq!(
        ZonalInteractionMatrixPlan::<f32>::new(1, [f32::from_bits(1)]),
        Err(
            StructuredInteractionMatrixPlanError::InvalidReciprocalScale {
                corrector_coordinate: 0
            }
        )
    );
}

#[test]
fn execution_errors_preserve_the_complete_output() {
    let plan = ZonalInteractionMatrixPlan::new(1, [1.0_f64]).unwrap();
    let other_plan = ZonalInteractionMatrixPlan::new(1, [1.0_f64, 1.0]).unwrap();
    let mut workspace = plan.workspace();
    let mut wrong_workspace = other_plan.workspace();
    let mut output = [17.0];

    use StructuredInteractionMatrixError::*;
    for (responses, nout, wrong_scratch, expected, detail) in [
        (
            &[1.0][..],
            1,
            false,
            ResponseExtent {
                expected: 2,
                actual: 1,
            },
            "response extent 1 does not match 2",
        ),
        (
            &[1.0, -1.0][..],
            0,
            false,
            OutputExtent {
                expected: 1,
                actual: 0,
            },
            "output extent 0 does not match 1",
        ),
        (
            &[f64::NAN, 0.0][..],
            1,
            false,
            NonFiniteResponse { index: 0 },
            "row-major index 0",
        ),
        (
            &[1.0, -1.0][..],
            1,
            true,
            WorkspaceExtent,
            "workspace extent",
        ),
        (
            &[f64::MAX, -f64::MAX][..],
            1,
            false,
            NonFiniteCalculation,
            "calculation produced a non-finite value",
        ),
    ] {
        let scratch = if wrong_scratch {
            &mut wrong_workspace
        } else {
            &mut workspace
        };
        let error = plan
            .process(&mut output[..nout], scratch, responses)
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
        assert_eq!(output, [17.0]);
    }

    plan.process(&mut output, &mut workspace, &[3.0, -1.0])
        .unwrap();
    assert_eq!(output, [2.0]);
}

#[test]
fn hadamard_calculation_failure_preserves_output_and_scratch_is_reusable() {
    let plan = HadamardInteractionMatrixPlan::new(1, [1.0_f64]).unwrap();
    let mut workspace = plan.workspace();
    let mut output = [23.0];
    assert_eq!(
        plan.process(
            &mut output,
            &mut workspace,
            &[f64::MAX, -f64::MAX, 0.0, 0.0]
        ),
        Err(StructuredInteractionMatrixError::NonFiniteCalculation)
    );
    assert_eq!(output, [23.0]);

    plan.process(&mut output, &mut workspace, &[2.0, -2.0, -2.0, 2.0])
        .unwrap();
    assert_eq!(output, [2.0]);
}
