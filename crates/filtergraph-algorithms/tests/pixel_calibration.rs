//! Pixel-calibration parity and progressive-processing tests.

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, AlgorithmProcessResult, ArrayLayout,
    ConfigurationUpdate, DeclarationParameterError, DeclarationPreparationError, ElementType,
    InputProgress, OutputProgress, PixelCalibrationError, PixelCalibrationPlan,
    PixelCalibrationRowU16F32, PixelCalibrationRowU16F32Configuration,
    PixelCalibrationRowU16F32PreparationError, PixelCalibrationRowU16F32ProcessError,
    PixelCalibrationU16F32, PixelCalibrationU16F32Configuration,
    PixelCalibrationU16F32PreparationError, PortPublication, PortRole, ProcessError,
    ProgressResult, PropertyPolicy, SampleMetadata,
};

fn require_output_progress<P: OutputProgress>(_plan: &P) {}

#[test]
fn preparation_requires_finite_equal_nonempty_arrays() {
    assert_eq!(
        PixelCalibrationPlan::<f32, u16>::new(Vec::new(), Vec::new()).unwrap_err(),
        PixelCalibrationError::Empty
    );
    assert_eq!(
        PixelCalibrationPlan::<f32, u16>::new(vec![1.0, 1.0], vec![0.0]).unwrap_err(),
        PixelCalibrationError::ExtentMismatch {
            flat: 2,
            background: 1,
        }
    );
    assert_eq!(
        PixelCalibrationPlan::<f32, u16>::new(vec![1.0, f32::NAN], vec![0.0; 2]).unwrap_err(),
        PixelCalibrationError::NonFiniteFlat { index: 1 }
    );
    assert_eq!(
        PixelCalibrationPlan::<f32, u16>::new(vec![1.0; 2], vec![0.0, f32::INFINITY]).unwrap_err(),
        PixelCalibrationError::NonFiniteBackground { index: 1 }
    );
}

#[test]
fn complete_processing_follows_the_prepared_equation() {
    let plan =
        PixelCalibrationPlan::<f32, u16>::new(vec![2.0, 1.0, 0.5], vec![1.0, 2.0, 3.0]).unwrap();
    let raw = [4_u16, 2, 1];
    let mut output = [0.0_f32; 3];

    plan.process(&mut output, &mut plan.workspace(), &raw)
        .unwrap();

    assert_eq!(output, [6.0, 0.0, -1.0]);
    assert_eq!(plan.flat(), [2.0, 1.0, 0.5]);
    assert_eq!(plan.background(), [1.0, 2.0, 3.0]);
}

#[test]
fn selected_float_and_raw_representations_are_preserved() {
    let mut output = [0.0_f32; 3];

    let u8_plan =
        PixelCalibrationPlan::<f32, u8>::new(vec![2.0, 1.0, 0.5], vec![1.0, 2.0, 3.0]).unwrap();
    u8_plan
        .process(&mut output, &mut u8_plan.workspace(), &[4_u8, 2, 1])
        .unwrap();
    assert_eq!(output, [6.0, 0.0, -1.0]);

    let i16_plan =
        PixelCalibrationPlan::<f32, i16>::new(vec![2.0, 1.0, 0.5], vec![1.0, 2.0, 3.0]).unwrap();
    i16_plan
        .process(&mut output, &mut i16_plan.workspace(), &[4_i16, 2, 1])
        .unwrap();
    assert_eq!(output, [6.0, 0.0, -1.0]);

    let u32_plan =
        PixelCalibrationPlan::<f32, u32>::new(vec![2.0, 1.0, 0.5], vec![1.0, 2.0, 3.0]).unwrap();
    u32_plan
        .process(&mut output, &mut u32_plan.workspace(), &[4_u32, 2, 1])
        .unwrap();
    assert_eq!(output, [6.0, 0.0, -1.0]);

    let f64_raw_plan =
        PixelCalibrationPlan::<f32, f64>::new(vec![2.0, 1.0, 0.5], vec![1.0, 2.0, 3.0]).unwrap();
    f64_raw_plan
        .process(
            &mut output,
            &mut f64_raw_plan.workspace(),
            &[4.5_f64, 2.0, 1.0],
        )
        .unwrap();
    assert_eq!(output, [7.0, 0.0, -1.0]);

    let f64_plan =
        PixelCalibrationPlan::<f64, u16>::new(vec![2.0, 1.0, 0.5], vec![1.0, 2.0, 3.0]).unwrap();
    let mut f64_output = [0.0_f64; 3];
    f64_plan
        .process(&mut f64_output, &mut f64_plan.workspace(), &[4_u16, 2, 1])
        .unwrap();
    assert_eq!(f64_output, [6.0, 0.0, -1.0]);
}

#[test]
fn progressive_processing_reports_immutable_pixel_prefixes() {
    let plan =
        PixelCalibrationPlan::<f32, u16>::new(vec![2.0, 1.0, 0.5, 4.0], vec![1.0, 2.0, 3.0, 1.0])
            .unwrap();
    require_output_progress(&plan);
    let raw = [4_u16, 2, 1, 3];
    let mut output = [-1.0_f32; 4];
    let mut workspace = plan.workspace();

    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &raw, 0..2),
        Ok(ProgressResult::new(2))
    );
    assert_eq!(output, [6.0, 0.0, -1.0, -1.0]);
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &raw, 2..4),
        Ok(ProgressResult::new(4))
    );
    plan.finish(&mut output, &mut workspace).unwrap();
    assert_eq!(output, [6.0, 0.0, -1.0, 8.0]);
    assert_eq!(plan.abort(&mut workspace), Err(ProcessError::NotStarted));
}

#[test]
fn range_local_output_preserves_the_published_prefix() {
    let plan =
        PixelCalibrationPlan::<f32, u16>::new(vec![2.0, 1.0, 0.5, 4.0], vec![1.0, 2.0, 3.0, 1.0])
            .unwrap();
    let raw = [4_u16, 2, 1, 3];
    let mut first = [-1.0_f32; 2];
    let mut second = [-1.0_f32; 2];
    let mut workspace = plan.workspace();

    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.process_range_window(&mut first, &mut workspace, &raw, 0..2),
        Ok(ProgressResult::new(2))
    );
    assert_eq!(first, [6.0, 0.0]);
    assert_eq!(
        plan.process_range_window(&mut second, &mut workspace, &raw, 2..4),
        Ok(ProgressResult::new(4))
    );
    plan.finish(&mut [], &mut workspace).unwrap();

    assert_eq!(first, [6.0, 0.0]);
    assert_eq!(second, [-1.0, 8.0]);
}

#[test]
fn independent_windows_use_detector_calibration_indices() {
    let plan = PixelCalibrationPlan::<f32, u16>::new(
        vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.25],
        vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
    )
    .unwrap();
    let mut output = [-1.0_f32; 3];

    plan.process_window(&mut output, &[3_u16, 4, 7], 3..6)
        .unwrap();

    assert_eq!(output, [8.0, 1.0, 1.0]);
    assert_eq!(
        plan.process_window(&mut output, &[1_u16; 3], 4..6),
        Err(ProcessError::InputExtent {
            expected: 2,
            actual: 3,
        })
    );
}

#[test]
fn range_local_output_extent_is_checked_before_mutation() {
    let plan = PixelCalibrationPlan::<f32, u16>::new(vec![1.0; 3], vec![0.0; 3]).unwrap();
    let raw = [1_u16, 2, 3];
    let mut output = [-1.0_f32; 2];
    let mut workspace = plan.workspace();

    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.process_range_window(&mut output, &mut workspace, &raw, 0..1),
        Err(ProcessError::OutputExtent {
            expected: 1,
            actual: 2,
        })
    );
    assert_eq!(output, [-1.0; 2]);
    assert_eq!(workspace.processed(), 0);
    plan.abort(&mut workspace).unwrap();
}

#[test]
fn extent_and_sequence_errors_precede_mutation() {
    let plan = PixelCalibrationPlan::<f32, u16>::new(vec![1.0; 3], vec![0.0; 3]).unwrap();
    let mut output = [-1.0_f32; 3];
    let mut workspace = plan.workspace();

    assert_eq!(
        plan.process(&mut output, &mut workspace, &[0_u16; 2]),
        Err(ProcessError::InputExtent {
            expected: 3,
            actual: 2,
        })
    );
    assert_eq!(output, [-1.0; 3]);
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &[0_u16; 3], 0..1),
        Err(ProcessError::NotStarted)
    );
    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.start(&mut workspace),
        Err(ProcessError::AlreadyStarted)
    );
    assert_eq!(
        plan.process(&mut output, &mut workspace, &[0_u16; 3]),
        Err(ProcessError::AlreadyStarted)
    );
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &[0_u16; 3], 0..0),
        Err(ProcessError::EmptyRange)
    );
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &[0_u16; 3], 1..2),
        Err(ProcessError::RangeNotContiguous {
            start: 1,
            expected: 0,
        })
    );
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &[0_u16; 3], 0..4),
        Err(ProcessError::RangeOutOfBounds { end: 4, extent: 3 })
    );
    assert_eq!(output, [-1.0; 3]);
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &[0_u16; 3], 0..1),
        Ok(ProgressResult::new(1))
    );
    assert_eq!(workspace.processed(), 1);
    assert_eq!(
        plan.finish(&mut output, &mut workspace),
        Err(ProcessError::Incomplete {
            processed: 1,
            extent: 3,
        })
    );
    plan.abort(&mut workspace).unwrap();
    assert!(!workspace.is_started());
    assert_eq!(workspace.processed(), 0);
}

#[test]
fn declaration_preserves_mixed_elements_and_named_calibration_planes() {
    let mut declared = PixelCalibrationU16F32::prepare(PixelCalibrationU16F32Configuration {
        initial_flat: Some(vec![2.0, 1.0, 0.5]),
        initial_background: Some(vec![1.0, 2.0, 3.0]),
        image_size: (1, 3),
    })
    .unwrap();
    assert_eq!(PixelCalibrationU16F32::NAME, "pixel-calibration-u16-f32");
    assert_eq!(PixelCalibrationU16F32::PROPERTIES, PropertyPolicy::None);
    assert_eq!(PixelCalibrationU16F32::CONFIGURATION.len(), 3);
    assert_eq!(
        PixelCalibrationU16F32::CONFIGURATION[0].update(),
        ConfigurationUpdate::GraphRebuild
    );
    assert_eq!(PixelCalibrationU16F32::PORTS.len(), 4);
    assert_eq!(
        PixelCalibrationU16F32::PORTS[0].element_type(),
        ElementType::U16
    );
    assert_eq!(
        PixelCalibrationU16F32::PORTS[1].element_type(),
        ElementType::F32
    );
    assert_eq!(PixelCalibrationU16F32::PORTS[2].role(), PortRole::Parameter);
    assert_eq!(PixelCalibrationU16F32::PORTS[3].role(), PortRole::Parameter);
    assert_eq!(declared.formats()[0].shape(), &[1, 3]);
    assert!(
        declared
            .formats()
            .iter()
            .all(|format| format.layout() == ArrayLayout::RowMajor)
    );

    let raw = [4_u16, 2, 1];
    let mut calibrated = [0.0_f32; 3];
    PixelCalibrationU16F32::process(&mut declared, &raw, &mut calibrated).unwrap();
    assert_eq!(calibrated, [6.0, 0.0, -1.0]);

    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&[1.0_f32, 2.0, 4.0][..])),
        Some(Ok(()))
    );
    PixelCalibrationU16F32::process(&mut declared, &raw, &mut calibrated).unwrap();
    assert_eq!(calibrated, [3.0, 0.0, -8.0]);

    assert_eq!(
        declared.replace_parameter(3, AlgorithmInput::from(&[0.0_f32; 3][..])),
        Some(Ok(()))
    );
    PixelCalibrationU16F32::process(&mut declared, &raw, &mut calibrated).unwrap();
    assert_eq!(calibrated, [4.0, 4.0, 4.0]);
}

#[test]
fn declaration_rejects_overflow_and_initial_plane_mismatches() {
    assert!(matches!(
        PixelCalibrationU16F32::prepare(PixelCalibrationU16F32Configuration {
            initial_flat: None,
            initial_background: None,
            image_size: (usize::MAX, 2),
        }),
        Err(DeclarationPreparationError::Algorithm(
            PixelCalibrationU16F32PreparationError::ImageExtentOverflow
        ))
    ));
    assert!(matches!(
        PixelCalibrationU16F32::prepare(PixelCalibrationU16F32Configuration {
            initial_flat: Some(vec![1.0]),
            initial_background: Some(vec![0.0; 2]),
            image_size: (1, 2),
        }),
        Err(DeclarationPreparationError::Algorithm(
            PixelCalibrationU16F32PreparationError::Calibration(
                PixelCalibrationError::ExtentMismatch {
                    flat: 1,
                    background: 2,
                }
            )
        ))
    ));
}

#[test]
fn declaration_defaults_to_identity_calibration_planes() {
    let mut declared = PixelCalibrationU16F32::prepare(PixelCalibrationU16F32Configuration {
        initial_flat: None,
        initial_background: None,
        image_size: (1, 3),
    })
    .unwrap();
    let raw = [4_u16, 2, 1];
    let mut calibrated = [0.0_f32; 3];

    PixelCalibrationU16F32::process(&mut declared, &raw, &mut calibrated).unwrap();

    assert_eq!(calibrated, [4.0, 2.0, 1.0]);
}

#[test]
fn native_parameter_transaction_is_failure_atomic() {
    let mut declared = PixelCalibrationU16F32::prepare(PixelCalibrationU16F32Configuration {
        initial_flat: Some(vec![2.0, 1.0, 0.5]),
        initial_background: Some(vec![1.0, 2.0, 3.0]),
        image_size: (1, 3),
    })
    .unwrap();
    let raw = [4_u16, 2, 1];
    let replacement_flat = [1.0_f32, 2.0, 4.0];
    let replacement_background = [0.0_f32; 3];
    declared
        .replace_parameters(&[
            (2, AlgorithmInput::from(&replacement_flat[..])),
            (3, AlgorithmInput::from(&replacement_background[..])),
        ])
        .unwrap();
    let mut calibrated = [0.0_f32; 3];
    PixelCalibrationU16F32::process(&mut declared, &raw, &mut calibrated).unwrap();
    assert_eq!(calibrated, [4.0, 4.0, 4.0]);

    let invalid_background = [0.0_f32, f32::NAN, 0.0];
    assert_eq!(
        declared.replace_parameters(&[
            (2, AlgorithmInput::from(&[8.0_f32; 3][..])),
            (3, AlgorithmInput::from(&invalid_background[..])),
        ]),
        Err(DeclarationParameterError::Algorithm(
            PixelCalibrationError::NonFiniteBackground { index: 1 }
        ))
    );
    PixelCalibrationU16F32::process(&mut declared, &raw, &mut calibrated).unwrap();
    assert_eq!(calibrated, [4.0, 4.0, 4.0]);

    assert_eq!(
        declared.replace_parameters(&[
            (2, AlgorithmInput::from(&replacement_flat[..])),
            (2, AlgorithmInput::from(&replacement_flat[..])),
        ]),
        Err(DeclarationParameterError::DuplicatePort { port: 2 })
    );
    assert_eq!(
        declared.replace_parameters(&[]),
        Err(DeclarationParameterError::EmptyTransaction)
    );
}

#[test]
fn row_declaration_preserves_identity_rate_and_plan_change_discontinuity() {
    let mut declared = PixelCalibrationRowU16F32::prepare(PixelCalibrationRowU16F32Configuration {
        row_block_rows: 2,
        initial_flat: Some((1..=12).map(|value| value as f32).collect()),
        initial_background: Some(vec![1.0; 12]),
        image_size: (4, 3),
    })
    .unwrap();
    assert_eq!(
        PixelCalibrationRowU16F32::NAME,
        "pixel-calibration-row-u16-f32"
    );
    assert_eq!(PixelCalibrationRowU16F32::PORTS.len(), 4);
    assert_eq!(
        PixelCalibrationRowU16F32::PORTS[1].publication(),
        PortPublication::Always
    );
    assert_eq!(declared.formats()[0].shape(), &[2, 3]);
    assert_eq!(declared.formats()[0].rate_ratio(), (2, 1));
    assert_eq!(declared.formats()[1].rate_ratio(), (2, 1));
    assert_eq!(declared.formats()[2].shape(), &[4, 3]);

    let first_raw = [2_u16, 3, 4, 5, 6, 7];
    let mut calibrated = [-1.0_f32; 6];
    let first = SampleMetadata::new(41, 0).with_timestamp_nanoseconds(Some(100));
    let first_result = PixelCalibrationRowU16F32::process_with_metadata(
        &mut declared,
        &[Some(first)],
        &first_raw,
        &mut calibrated,
    )
    .unwrap();
    assert_eq!(calibrated, [1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    assert_eq!(
        first_result,
        AlgorithmProcessResult::complete_with_metadata(first)
    );

    let replacement_flat = [2.0_f32; 12];
    assert_eq!(
        declared.replace_parameter(2, AlgorithmInput::from(&replacement_flat[..])),
        Some(Ok(()))
    );
    let final_raw = [8_u16, 9, 10, 11, 12, 13];
    let final_sample = SampleMetadata::new(41, 2)
        .with_timestamp_nanoseconds(Some(200))
        .with_terminal(true);
    let final_result = PixelCalibrationRowU16F32::process_with_metadata(
        &mut declared,
        &[Some(final_sample)],
        &final_raw,
        &mut calibrated,
    )
    .unwrap();
    assert_eq!(calibrated, [14.0, 16.0, 18.0, 20.0, 22.0, 24.0]);
    assert_eq!(
        final_result,
        AlgorithmProcessResult::complete_with_metadata(final_sample.with_discontinuity(true))
    );
}

#[test]
fn row_declaration_rejects_invalid_quantum_and_metadata() {
    assert!(matches!(
        PixelCalibrationRowU16F32::prepare(PixelCalibrationRowU16F32Configuration {
            row_block_rows: 3,
            initial_flat: None,
            initial_background: None,
            image_size: (4, 3),
        }),
        Err(DeclarationPreparationError::Algorithm(
            PixelCalibrationRowU16F32PreparationError::RowBlockRows
        ))
    ));

    let mut declared = PixelCalibrationRowU16F32::prepare(PixelCalibrationRowU16F32Configuration {
        row_block_rows: 2,
        initial_flat: None,
        initial_background: None,
        image_size: (4, 3),
    })
    .unwrap();
    let raw = [1_u16; 6];
    let mut calibrated = [0.0_f32; 6];
    assert!(matches!(
        PixelCalibrationRowU16F32::process_with_metadata(
            &mut declared,
            &[None],
            &raw,
            &mut calibrated,
        ),
        Err(filtergraph_algorithms::NativeProcessError::Process(
            filtergraph_algorithms::DeclarationProcessError::Algorithm(
                PixelCalibrationRowU16F32ProcessError::MissingSampleMetadata
            )
        ))
    ));
    let misaligned = SampleMetadata::new(1, 1);
    assert!(matches!(
        PixelCalibrationRowU16F32::process_with_metadata(
            &mut declared,
            &[Some(misaligned)],
            &raw,
            &mut calibrated,
        ),
        Err(filtergraph_algorithms::NativeProcessError::Process(
            filtergraph_algorithms::DeclarationProcessError::Algorithm(
                PixelCalibrationRowU16F32ProcessError::MisalignedOffset { .. }
            )
        ))
    ));
}

fn check_row_pending_discontinuity(reset: bool) {
    let mut declared = PixelCalibrationRowU16F32::prepare(PixelCalibrationRowU16F32Configuration {
        row_block_rows: 2,
        initial_flat: None,
        initial_background: None,
        image_size: (6, 3),
    })
    .unwrap();
    if reset {
        PixelCalibrationRowU16F32::reset(&mut declared);
    } else {
        let flat = [1.0_f32; 18];
        assert_eq!(
            declared.replace_parameter(2, AlgorithmInput::from(&flat[..])),
            Some(Ok(()))
        );
    }
    let raw = [2_u16; 6];
    let mut calibrated = [0.0_f32; 6];
    for block in 0..3 {
        let metadata = SampleMetadata::new(0, block * 2)
            .with_discontinuity(block == 0)
            .with_terminal(block == 2);
        assert_eq!(
            PixelCalibrationRowU16F32::process_with_metadata(
                &mut declared,
                &[Some(metadata)],
                &raw,
                &mut calibrated,
            )
            .unwrap(),
            AlgorithmProcessResult::complete_with_metadata(metadata),
            "reset={reset}, block={block}"
        );
        assert_eq!(calibrated, [2.0; 6]);
    }
}

#[test]
fn row_calibration_consumes_pending_discontinuity_after_plan_adoption() {
    check_row_pending_discontinuity(false);
}

#[test]
fn row_calibration_consumes_pending_discontinuity_after_reset() {
    check_row_pending_discontinuity(true);
}

#[test]
fn invalid_independent_windows_report_extent_and_preserve_output() {
    let plan = PixelCalibrationPlan::<f32, u16>::new(vec![1.0; 6], vec![0.0; 6]).unwrap();
    let mut output = [19.0; 3];
    for (range, nraw, nout, expected, diagnostic) in [
        (
            core::ops::Range { start: 4, end: 3 },
            0,
            0,
            ProcessError::EmptyRange,
            "nonempty",
        ),
        (
            5..7,
            2,
            2,
            ProcessError::RangeOutOfBounds { end: 7, extent: 6 },
            "7",
        ),
        (
            3..6,
            2,
            3,
            ProcessError::InputExtent {
                expected: 3,
                actual: 2,
            },
            "input extent",
        ),
        (
            3..6,
            3,
            2,
            ProcessError::OutputExtent {
                expected: 3,
                actual: 2,
            },
            "output extent",
        ),
    ] {
        let error = plan
            .process_window(&mut output[..nout], &[2; 3][..nraw], range)
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(diagnostic), "{error}");
        assert_eq!(output, [19.0; 3]);
    }
}

#[test]
fn row_calibration_reports_invalid_geometry_and_metadata() {
    use PixelCalibrationRowU16F32PreparationError as Prep;
    use PixelCalibrationRowU16F32ProcessError as Run;
    use core::error::Error;
    use filtergraph_algorithms::{DeclarationProcessError, NativeProcessError};

    let cfg = |image_size, row_block_rows, flat| PixelCalibrationRowU16F32Configuration {
        row_block_rows,
        initial_flat: flat,
        initial_background: None,
        image_size,
    };
    for (config, expected, diagnostic) in [
        (
            cfg((usize::MAX - 1, 3), 2, None),
            Prep::ImageExtentOverflow,
            "image extent overflow",
        ),
        (cfg((4, 3), 4, None), Prep::RowBlockRows, "row-block height"),
        (
            cfg((4, 3), 2, Some(vec![1.0; 11])),
            Prep::Calibration(PixelCalibrationError::ExtentMismatch {
                flat: 11,
                background: 12,
            }),
            "flat extent 11",
        ),
    ] {
        let error = PixelCalibrationRowU16F32::prepare(config).err().unwrap();
        assert_eq!(error, DeclarationPreparationError::Algorithm(expected));
        assert!(error.to_string().contains(diagnostic), "{error}");
        assert_eq!(
            expected.source().is_some(),
            matches!(expected, Prep::Calibration(_))
        );
    }

    let mut prepared = PixelCalibrationRowU16F32::prepare(cfg((4, 3), 2, None)).unwrap();
    let mut output = [19.0; 6];
    for (sample, expected, detail) in [
        (None, Run::MissingSampleMetadata, "requires sample metadata"),
        (
            Some(SampleMetadata::new(7, 1)),
            Run::MisalignedOffset {
                first_row: 1,
                block_rows: 2,
            },
            "first row 1 is not aligned",
        ),
        (
            Some(SampleMetadata::new(7, 4).with_terminal(true)),
            Run::RowRange,
            "exceeds the detector height",
        ),
        (
            Some(SampleMetadata::new(7, 2)),
            Run::Terminal,
            "terminal marker",
        ),
    ] {
        let error = PixelCalibrationRowU16F32::process_with_metadata(
            &mut prepared,
            &[sample],
            &[2; 6],
            &mut output,
        )
        .unwrap_err();
        assert_eq!(
            error,
            NativeProcessError::Process(DeclarationProcessError::Algorithm(expected))
        );
        assert!(error.to_string().contains(detail), "{error}");
        assert_eq!(output, [19.0; 6]);
    }
}
