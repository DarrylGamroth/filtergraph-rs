//! RTC-DEV-008 evidence for portable, scientist-owned declarations.
//!
//! This fixture deliberately imports only `filtergraph_algorithms`.  It runs
//! declarations with borrowed Rust arrays and assembles a small native graph;
//! neither path loads PipeWire or an FGN adapter.

use filtergraph_algorithms::schemas::{
    CONTROLLER_RESIDUAL_ERROR_V1, PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1,
    PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1, PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1,
    REQUESTED_PDM_COMMAND_V1, SHACK_HARTMANN_SLOPES_V1,
};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmOutput, AlgorithmPlan,
    DeclarationParameterError, DeclarationProcessError, LeakyIntegratorF32,
    LeakyIntegratorF32Configuration, LeakyIntegratorPlan, LeakyIntegratorWorkspace,
    OpticalGainCorrectionF32, OpticalGainCorrectionF32Configuration, PixelCalibrationError,
    PixelCalibrationU16F32, PixelCalibrationU16F32Configuration, PortError, PortRole,
    PropertyAssignment, PropertyValue, PseudoOpenLoopReconstructionF32,
    PseudoOpenLoopReconstructionF32Configuration,
};

/// A deliberately small ordinary-array graph executor.
///
/// The edge is a Rust slice.  This is intentionally a minimal non-PipeWire
/// executor: a graph author can prepare declarations and connect their typed
/// array results without an adapter registry or transport-specific source.
struct NativeCalibrationIntegratorGraph {
    calibration: filtergraph_algorithms::PreparedAlgorithm<PixelCalibrationU16F32>,
    integrator: filtergraph_algorithms::PreparedAlgorithm<LeakyIntegratorF32>,
}

impl NativeCalibrationIntegratorGraph {
    fn new() -> Self {
        let calibration = PixelCalibrationU16F32::prepare(PixelCalibrationU16F32Configuration {
            initial_flat: Some(vec![2.0, 0.5]),
            initial_background: Some(vec![1.0, 2.0]),
            image_size: (1, 2),
        })
        .unwrap();
        let mut integrator = LeakyIntegratorF32::prepare(LeakyIntegratorF32Configuration {
            extent: 2,
            initial_state: 1.0,
            input_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
            output_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
        })
        .unwrap();
        integrator
            .replace_properties(&[
                PropertyAssignment {
                    id: 0,
                    value: PropertyValue::Float(0.5),
                },
                PropertyAssignment {
                    id: 1,
                    value: PropertyValue::Float(0.5),
                },
            ])
            .unwrap();
        Self {
            calibration,
            integrator,
        }
    }

    fn process(&mut self, raw: &[u16; 2]) -> ([f32; 2], [f32; 2]) {
        let mut calibrated = [0.0_f32; 2];
        self.calibration
            .process(
                &[AlgorithmInput::from(&raw[..])],
                &mut [AlgorithmOutput::from(&mut calibrated[..])],
            )
            .unwrap();
        let mut integrated = [0.0_f32; 2];
        self.integrator
            .process(
                &[AlgorithmInput::from(&calibrated[..])],
                &mut [AlgorithmOutput::from(&mut integrated[..])],
            )
            .unwrap();
        (calibrated, integrated)
    }
}

#[test]
fn stateless_stateful_parameter_and_multi_input_declarations_are_local_and_typed() {
    // Stateless: reciprocal correction has no mutable workspace or scalar
    // properties. Its sparse gain vector remains a scientific parameter.
    let correction = OpticalGainCorrectionF32::prepare(OpticalGainCorrectionF32Configuration {
        optical_gains: vec![0.5, 2.0],
    })
    .unwrap();
    assert_eq!(
        OpticalGainCorrectionF32::NAME,
        "optical-gain-correction-f32"
    );
    assert_eq!(
        OpticalGainCorrectionF32::PORTS[2].role(),
        PortRole::Parameter
    );
    let mut correction = correction;
    let reconstructed = [1.0_f32, -4.0];
    let mut corrected = [0.0_f32; 2];
    correction
        .process(
            &[AlgorithmInput::from(&reconstructed[..])],
            &mut [AlgorithmOutput::from(&mut corrected[..])],
        )
        .unwrap();
    assert_eq!(corrected, [2.0, -2.0]);

    // Stateful: the declaration produces exactly the direct leaky-integrator
    // recurrence across two ordinary-array samples.
    let raw = [4_u16, 6];
    let direct_plan = LeakyIntegratorPlan::new(0.5_f32, 0.5).unwrap();
    let mut direct_workspace = LeakyIntegratorWorkspace::filled(2, 1.0).unwrap();
    let mut direct = [0.0_f32; 2];
    let expected_calibrated = [6.0_f32, 2.0];
    direct_plan
        .process(&mut direct, &mut direct_workspace, &expected_calibrated)
        .unwrap();
    let mut graph = NativeCalibrationIntegratorGraph::new();
    let (calibrated, integrated) = graph.process(&raw);
    assert_eq!(calibrated, expected_calibrated);
    assert_eq!(integrated, direct);
    assert_eq!(graph.process(&raw).1, [4.75, 1.75]);

    // Multi-input: the pseudo-open-loop declaration identifies independent
    // scientific inputs and outputs without adapter-specific glue.
    let mut pseudo =
        PseudoOpenLoopReconstructionF32::prepare(PseudoOpenLoopReconstructionF32Configuration {
            measurement_count: 2,
            corrector_shape_count: 1,
            residual_error_count: 1,
            initial_interaction_matrix: Some(vec![2.0, 1.0]),
            initial_reconstructor: Some(vec![0.5, 1.0]),
            initial_corrector_to_error: Some(vec![2.0]),
            residual_measurements_schema: SHACK_HARTMANN_SLOPES_V1.into(),
            corrector_shape_schema: REQUESTED_PDM_COMMAND_V1.into(),
            pseudo_open_loop_measurements_schema: SHACK_HARTMANN_SLOPES_V1.into(),
            residual_error_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
        })
        .unwrap();
    assert_eq!(PseudoOpenLoopReconstructionF32::PORTS.len(), 7);
    assert_eq!(
        PseudoOpenLoopReconstructionF32::PORTS[4].role(),
        PortRole::Parameter
    );
    assert_eq!(
        pseudo.formats()[4].schema(),
        PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1
    );
    assert_eq!(
        pseudo.formats()[5].schema(),
        PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1
    );
    assert_eq!(
        pseudo.formats()[6].schema(),
        PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1
    );
    let measurements = [4.0_f32, 3.0];
    let corrector_shape = [1.0_f32];
    let mut pseudo_measurements = [0.0_f32; 2];
    let mut residual_error = [0.0_f32; 1];
    pseudo
        .process(
            &[
                AlgorithmInput::from(&measurements[..]),
                AlgorithmInput::from(&corrector_shape[..]),
            ],
            &mut [
                AlgorithmOutput::from(&mut pseudo_measurements[..]),
                AlgorithmOutput::from(&mut residual_error[..]),
            ],
        )
        .unwrap();
    assert_eq!(pseudo_measurements, [6.0, 4.0]);
    assert_eq!(residual_error, [5.0]);
}

#[test]
fn native_declarations_report_scientific_and_port_diagnostics_without_an_adapter() {
    let mut calibration = NativeCalibrationIntegratorGraph::new().calibration;
    assert_eq!(
        calibration.replace_parameter(2, AlgorithmInput::from(&[f32::NAN, 1.0][..])),
        Some(Err(DeclarationParameterError::Algorithm(
            PixelCalibrationError::NonFiniteFlat { index: 0 }
        )))
    );

    let mut pseudo =
        PseudoOpenLoopReconstructionF32::prepare(PseudoOpenLoopReconstructionF32Configuration {
            measurement_count: 1,
            corrector_shape_count: 1,
            residual_error_count: 1,
            initial_interaction_matrix: Some(vec![1.0]),
            initial_reconstructor: Some(vec![1.0]),
            initial_corrector_to_error: Some(vec![1.0]),
            residual_measurements_schema: SHACK_HARTMANN_SLOPES_V1.into(),
            corrector_shape_schema: REQUESTED_PDM_COMMAND_V1.into(),
            pseudo_open_loop_measurements_schema: SHACK_HARTMANN_SLOPES_V1.into(),
            residual_error_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
        })
        .unwrap();
    let mut pseudo_measurements = [0.0_f32; 1];
    let mut residual_error = [0.0_f32; 1];
    assert_eq!(
        pseudo.process(
            &[AlgorithmInput::from(&[1.0_f32][..])],
            &mut [
                AlgorithmOutput::from(&mut pseudo_measurements[..]),
                AlgorithmOutput::from(&mut residual_error[..]),
            ],
        ),
        Err(DeclarationProcessError::Port(PortError::Missing))
    );
}
