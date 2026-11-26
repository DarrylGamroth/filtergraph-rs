//! Generated ndarray declaration tests for multi-region detector readout.

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmProcessResult, DetectorFrameAssemblerF32,
    DetectorFrameAssemblerF32Configuration, PixelCalibrationReadoutU16F32,
    PixelCalibrationReadoutU16F32Configuration, PortPublication, PyramidReadoutReconstructorF32,
    PyramidReadoutReconstructorF32Configuration, SampleMetadata,
    ShackHartmannReadoutReconstructorF32, ShackHartmannReadoutReconstructorF32Configuration,
};

fn origins() -> Vec<[usize; 2]> {
    vec![[0, 0], [3, 0]]
}

fn line_steps() -> Vec<[i32; 2]> {
    vec![[1, 0], [-1, 0]]
}

fn sample_steps() -> Vec<[i32; 2]> {
    vec![[0, 1], [0, 1]]
}

fn sample(sequence: u64, ordinal: u32) -> SampleMetadata {
    SampleMetadata::new(sequence, ordinal).with_terminal(ordinal == 1)
}

#[test]
fn calibration_and_assembly_declarations_preserve_rank_three_identity() {
    let mut calibration =
        PixelCalibrationReadoutU16F32::prepare(PixelCalibrationReadoutU16F32Configuration {
            region_lines: 2,
            block_lines: 1,
            region_samples: 4,
            detector_origins: origins(),
            detector_line_steps: line_steps(),
            detector_sample_steps: sample_steps(),
            initial_flat: None,
            initial_background: None,
            detector_size: (4, 4),
        })
        .unwrap();
    let mut assembler =
        DetectorFrameAssemblerF32::prepare(DetectorFrameAssemblerF32Configuration {
            region_lines: 2,
            block_lines: 1,
            region_samples: 4,
            detector_origins: origins(),
            detector_line_steps: line_steps(),
            detector_sample_steps: sample_steps(),
            detector_size: (4, 4),
        })
        .unwrap();

    assert_eq!(calibration.formats()[0].shape(), &[2, 1, 4]);
    assert_eq!(calibration.formats()[0].rate_ratio(), (2, 1));
    assert_eq!(assembler.formats()[1].shape(), &[4, 4]);
    assert_eq!(
        DetectorFrameAssemblerF32::PORTS[1].publication(),
        PortPublication::Conditional
    );

    let detector = (1_u16..=16).collect::<Vec<_>>();
    let acquisition = [
        &detector[..4],
        &detector[12..16],
        &detector[4..8],
        &detector[8..12],
    ]
    .into_iter()
    .flatten()
    .copied()
    .collect::<Vec<_>>();
    let mut calibrated = [0.0_f32; 8];
    let mut frame = [-1.0_f32; 16];
    for ordinal in 0..2 {
        let first = ordinal * 8;
        let metadata = sample(9, ordinal as u32);
        assert_eq!(
            PixelCalibrationReadoutU16F32::process_with_metadata(
                &mut calibration,
                &[Some(metadata)],
                &acquisition[first..first + 8],
                &mut calibrated,
            )
            .unwrap(),
            AlgorithmProcessResult::complete_with_metadata(metadata)
        );
        let result = DetectorFrameAssemblerF32::process_with_metadata(
            &mut assembler,
            &[Some(metadata)],
            &calibrated,
            &mut frame,
        )
        .unwrap();
        if ordinal == 0 {
            assert_eq!(result, AlgorithmProcessResult::Deferred);
            assert_eq!(frame, [-1.0; 16]);
        } else {
            assert!(matches!(result, AlgorithmProcessResult::Complete { .. }));
        }
    }
    assert_eq!(
        frame,
        detector
            .iter()
            .map(|&value| value as f32)
            .collect::<Vec<_>>()
            .as_slice()
    );

    PixelCalibrationReadoutU16F32::reset(&mut calibration);
    let metadata = sample(10, 0);
    assert_eq!(
        PixelCalibrationReadoutU16F32::process_with_metadata(
            &mut calibration,
            &[Some(metadata)],
            &acquisition[..8],
            &mut calibrated,
        )
        .unwrap(),
        AlgorithmProcessResult::complete_with_metadata(metadata.with_discontinuity(true))
    );
}

#[test]
fn progressive_wfs_declarations_publish_only_at_terminal() {
    let mut shack = ShackHartmannReadoutReconstructorF32::prepare(
        ShackHartmannReadoutReconstructorF32Configuration {
            region_lines: 2,
            block_lines: 1,
            region_samples: 4,
            detector_origins: origins(),
            detector_line_steps: line_steps(),
            detector_sample_steps: sample_steps(),
            subaperture_count: 4,
            reconstructed_count: 1,
            initial_subaperture_origins: Some(vec![[3, 0], [0, 0], [2, 0], [1, 0]]),
            coordinate_scale: 1.0,
            pixel_threshold: 0.0,
            flux_threshold: 0.0,
            reference_slopes: None,
            active: None,
            initial_reconstructor: Some(vec![1.0; 8]),
            reconstructed_schema: "test.filtergraph.shwfs-readout/1".into(),
            subaperture_size: (1, 2),
            detector_size: (4, 4),
        },
    )
    .unwrap();
    let mut pyramid =
        PyramidReadoutReconstructorF32::prepare(PyramidReadoutReconstructorF32Configuration {
            region_lines: 2,
            block_lines: 1,
            region_samples: 4,
            detector_origins: origins(),
            detector_line_steps: line_steps(),
            detector_sample_steps: sample_steps(),
            pupil_origins: vec![[3, 0], [0, 0], [2, 0], [1, 0]],
            pupil_mask: Some(vec![true]),
            reconstructed_count: 1,
            initial_reconstructor: Some(vec![1.0; 4]),
            reconstructed_schema: "test.filtergraph.pwfs-readout/1".into(),
            pupil_size: (1, 1),
            detector_size: (4, 4),
        })
        .unwrap();
    let blocks = [[1.0_f32; 8], [2.0_f32; 8]];
    let mut shack_output = [-1.0_f32];
    let mut pyramid_pixels = [-1.0_f32; 4];
    let mut pyramid_mean = [-1.0_f32];
    let mut pyramid_output = [-1.0_f32];

    for (ordinal, block) in blocks.iter().enumerate() {
        let metadata = sample(17, ordinal as u32);
        let shack_result = ShackHartmannReadoutReconstructorF32::process_with_metadata(
            &mut shack,
            &[Some(metadata)],
            block,
            &mut shack_output,
        )
        .unwrap();
        let pyramid_result = PyramidReadoutReconstructorF32::process_with_metadata(
            &mut pyramid,
            &[Some(metadata)],
            block,
            &mut pyramid_pixels,
            &mut pyramid_mean,
            &mut pyramid_output,
        )
        .unwrap();
        if ordinal == 0 {
            assert_eq!(shack_result, AlgorithmProcessResult::Deferred);
            assert_eq!(pyramid_result, AlgorithmProcessResult::Deferred);
            assert_eq!(shack_output, [-1.0]);
            assert_eq!(pyramid_output, [-1.0]);
        } else {
            assert!(matches!(
                shack_result,
                AlgorithmProcessResult::Complete { .. }
            ));
            assert!(matches!(
                pyramid_result,
                AlgorithmProcessResult::Complete { .. }
            ));
        }
    }
}

fn check_readout_pending_discontinuity(reset: bool) {
    let mut calibration =
        PixelCalibrationReadoutU16F32::prepare(PixelCalibrationReadoutU16F32Configuration {
            region_lines: 3,
            block_lines: 1,
            region_samples: 4,
            detector_origins: vec![[0, 0]],
            detector_line_steps: vec![[1, 0]],
            detector_sample_steps: vec![[0, 1]],
            initial_flat: None,
            initial_background: None,
            detector_size: (3, 4),
        })
        .unwrap();
    if reset {
        PixelCalibrationReadoutU16F32::reset(&mut calibration);
    } else {
        let flat = [1.0_f32; 12];
        assert_eq!(
            calibration.replace_parameter(2, AlgorithmInput::from(&flat[..])),
            Some(Ok(()))
        );
    }
    let raw = [2_u16; 4];
    let mut calibrated = [0.0_f32; 4];
    for ordinal in 0..3 {
        let metadata = SampleMetadata::new(0, ordinal)
            .with_discontinuity(ordinal == 0)
            .with_terminal(ordinal == 2);
        assert_eq!(
            PixelCalibrationReadoutU16F32::process_with_metadata(
                &mut calibration,
                &[Some(metadata)],
                &raw,
                &mut calibrated,
            )
            .unwrap(),
            AlgorithmProcessResult::complete_with_metadata(metadata),
            "reset={reset}, ordinal={ordinal}"
        );
        assert_eq!(calibrated, [2.0; 4]);
    }
}

#[test]
fn readout_calibration_consumes_pending_discontinuity_after_plan_adoption() {
    check_readout_pending_discontinuity(false);
}

#[test]
fn readout_calibration_consumes_pending_discontinuity_after_reset() {
    check_readout_pending_discontinuity(true);
}

#[test]
fn rejected_calibration_blocks_do_not_consume_pending_plan_discontinuity() {
    use filtergraph_algorithms::{
        DeclarationProcessError, DetectorReadoutCalibrationProcessError as E, NativeProcessError,
    };

    let mut calibration =
        PixelCalibrationReadoutU16F32::prepare(PixelCalibrationReadoutU16F32Configuration {
            region_lines: 2,
            block_lines: 1,
            region_samples: 4,
            detector_origins: origins(),
            detector_line_steps: line_steps(),
            detector_sample_steps: sample_steps(),
            initial_flat: None,
            initial_background: None,
            detector_size: (4, 4),
        })
        .unwrap();
    let backgrounds: Vec<_> = (1..=16).map(|background| background as f32).collect();
    let port = PixelCalibrationReadoutU16F32::PORTS
        .iter()
        .position(|port| port.name() == "background")
        .unwrap();
    calibration
        .replace_parameter(port, AlgorithmInput::from(backgrounds.as_slice()))
        .unwrap()
        .unwrap();
    let mut output = [83.0; 8];
    for (sample, expected, detail) in [
        (
            None,
            E::MissingSampleMetadata,
            "sample metadata is required",
        ),
        (
            Some(sample(1, 2)),
            E::OrdinalRange {
                ordinal: 2,
                blocks: 2,
            },
            "ordinal 2 is outside 2",
        ),
        (
            Some(sample(1, 0).with_terminal(true)),
            E::Terminal,
            "terminal marker",
        ),
    ] {
        let error = PixelCalibrationReadoutU16F32::process_with_metadata(
            &mut calibration,
            &[sample],
            &[20; 8],
            &mut output,
        )
        .unwrap_err();
        assert_eq!(
            error,
            NativeProcessError::Process(DeclarationProcessError::Algorithm(expected))
        );
        assert!(error.to_string().contains(detail), "{error}");
        assert_eq!(output, [83.0; 8]);
    }

    for (ordinal, expected) in [
        (0, [19.0, 18.0, 17.0, 16.0, 7.0, 6.0, 5.0, 4.0]),
        (1, [15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0]),
    ] {
        let metadata = sample(7, ordinal);
        let result = PixelCalibrationReadoutU16F32::process_with_metadata(
            &mut calibration,
            &[Some(metadata)],
            &[20; 8],
            &mut output,
        )
        .unwrap();
        assert_eq!(
            result.metadata(),
            Some(metadata.with_discontinuity(ordinal == 0))
        );
        assert_eq!(output, expected);
    }
}

#[test]
fn readout_configuration_preserves_geometry_failure_through_declaration_admission() {
    use filtergraph_algorithms::{
        DeclarationPreparationError, DetectorReadoutProfileError,
        PixelCalibrationReadoutU16F32PreparationError,
    };
    let cfg = |line_steps, size, flat| PixelCalibrationReadoutU16F32Configuration {
        region_lines: 2,
        block_lines: 1,
        region_samples: 4,
        detector_origins: origins(),
        detector_line_steps: line_steps,
        detector_sample_steps: sample_steps(),
        initial_flat: flat,
        initial_background: None,
        detector_size: size,
    };
    let error = PixelCalibrationReadoutU16F32::prepare(cfg(vec![[1, 0]], (4, 4), None))
        .err()
        .unwrap();
    assert_eq!(
        error,
        DeclarationPreparationError::Algorithm(
            PixelCalibrationReadoutU16F32PreparationError::Profile(
                DetectorReadoutProfileError::RegionExtent {
                    origins: 2,
                    line_steps: 1,
                    sample_steps: 2,
                }
            )
        )
    );
    assert!(
        error
            .to_string()
            .contains("2 origins, 1 line steps, and 2 sample steps")
    );

    let error = PixelCalibrationReadoutU16F32::prepare(cfg(line_steps(), (4, 3), None))
        .err()
        .unwrap();
    assert!(matches!(
        error,
        DeclarationPreparationError::Algorithm(
            PixelCalibrationReadoutU16F32PreparationError::Profile(
                DetectorReadoutProfileError::Mapping(_)
            )
        )
    ));
    assert!(
        error
            .to_string()
            .contains("pixel extent 16 does not match detector extent 12")
    );

    let mut flat = vec![1.0; 16];
    flat[1] = f32::NAN;
    let error = PixelCalibrationReadoutU16F32::prepare(cfg(line_steps(), (4, 4), Some(flat)))
        .err()
        .unwrap();
    assert!(matches!(
        error,
        DeclarationPreparationError::Algorithm(
            PixelCalibrationReadoutU16F32PreparationError::Calibration(_)
        )
    ));
    assert!(error.to_string().contains("flat value at index 1"));
}
