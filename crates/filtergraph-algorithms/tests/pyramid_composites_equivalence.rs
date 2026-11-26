//! Source-derived sparse publication, normalization history, and dense ordering.
use filtergraph_algorithms::{
    AlgorithmInput, PyramidMeasurementBlockF32 as Rows,
    PyramidMeasurementBlockF32Configuration as RC, PyramidPupilReconstructorF32 as Dense,
    PyramidPupilReconstructorF32Configuration as MC, PyramidReadoutMeasurementBlockF32 as Readout,
    PyramidReadoutMeasurementBlockF32Configuration as DC, SampleMetadata,
};
fn rows() -> RC {
    RC {
        image_size: (6, 4),
        row_block_rows: 1,
        pupil_size: (2, 2),
        pupil_origins: vec![[4, 2], [2, 0], [4, 0], [2, 2]],
        pupil_mask: Some(vec![true, false, true, true]),
        measurements_schema: "test.pyramid/measurements/1".into(),
        column_indices_schema: "test.pyramid/columns/1".into(),
        measurement_count_schema: "test.pyramid/count/1".into(),
    }
}
fn readout() -> DC {
    let c = rows();
    DC {
        detector_size: c.image_size,
        region_lines: 3,
        block_lines: 1,
        region_samples: 4,
        detector_origins: vec![[0, 0], [3, 0]],
        detector_line_steps: vec![[1, 0], [1, 0]],
        detector_sample_steps: vec![[0, 1], [0, 1]],
        pupil_size: c.pupil_size,
        pupil_origins: c.pupil_origins,
        pupil_mask: c.pupil_mask,
        measurements_schema: c.measurements_schema,
        column_indices_schema: c.column_indices_schema,
        measurement_count_schema: c.measurement_count_schema,
    }
}
#[test]
fn row_prefixes_include_empty_blocks_and_keep_physical_sample_offsets() {
    let mut p = Rows::prepare(rows()).unwrap();
    for frame in 0..2 {
        for row in 0..6 {
            let mut values = [71.0; 12];
            let mut columns = [999u32; 12];
            let mut count = [999];
            let sample = SampleMetadata::new(frame, row)
                .with_terminal(row == 5)
                .with_timestamp_nanoseconds(Some(333));
            let result = Rows::process_with_metadata(
                &mut p,
                &[Some(sample)],
                &[if frame == 0 { 2.0 } else { 4.0 }; 4],
                &mut values,
                &mut columns,
                &mut count,
            )
            .unwrap();
            assert_eq!(result.metadata(), Some(sample));
            let expected: &[u32] = match row {
                0 | 1 => &[],
                2 => &[3, 9],
                3 => &[4, 5, 10, 11],
                4 => &[0, 6],
                _ => &[1, 2, 7, 8],
            };
            let n = count[0] as usize;
            assert_eq!(n, expected.len());
            assert_eq!(&columns[..n], expected);
            assert!(
                values[..n]
                    .iter()
                    .all(|&v| v == if frame == 0 { 0.0 } else { 2.0 })
            );
            assert!(values[n..].iter().all(|&v| v == 71.0));
            assert!(columns[n..].iter().all(|&v| v == 999));
        }
    }
    assert_eq!(p.workspace().reciprocal_previous_frame_mean(), 0.25);
}
#[test]
fn readout_prefixes_follow_canonical_completion_groups() {
    let mut p = Readout::prepare(readout()).unwrap();
    for frame in 0..2 {
        for ordinal in 0..3 {
            let mut values = [71.0; 12];
            let mut columns = [999; 12];
            let mut count = [999];
            let sample = SampleMetadata::new(frame, ordinal).with_terminal(ordinal == 2);
            let result = Readout::process_with_metadata(
                &mut p,
                &[Some(sample)],
                &[if frame == 0 { 2.0 } else { 4.0 }; 8],
                &mut values,
                &mut columns,
                &mut count,
            )
            .unwrap();
            assert_eq!(result.metadata(), Some(sample));
            let expected: &[u32] = match ordinal {
                0 => &[4, 5, 10, 11],
                1 => &[0, 6],
                _ => &[1, 2, 3, 7, 8, 9],
            };
            let n = count[0] as usize;
            assert_eq!(&columns[..n], expected);
            assert!(
                values[..n]
                    .iter()
                    .all(|&v| v == if frame == 0 { 0.0 } else { 2.0 })
            );
            assert!(values[n..].iter().all(|&v| v == 71.0));
        }
    }
    assert_eq!(p.workspace().reciprocal_previous_frame_mean(), 0.25);
}
#[test]
fn row_rejection_and_parameter_adoption_abandon_only_partial_history() {
    let mut p = Rows::prepare(rows()).unwrap();
    let mut values = [71.0; 12];
    let mut columns = [999; 12];
    let mut count = [999];
    for row in 0..6 {
        Rows::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(0, row).with_terminal(row == 5))],
            &[2.0; 4],
            &mut values,
            &mut columns,
            &mut count,
        )
        .unwrap();
    }
    assert_eq!(p.workspace().reciprocal_previous_frame_mean(), 0.5);
    for row in 0..2 {
        Rows::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(1, row))],
            &[4.0; 4],
            &mut values,
            &mut columns,
            &mut count,
        )
        .unwrap();
    }
    let previous = (values, columns, count);
    assert!(
        Rows::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(1, 2))],
            &[f32::NAN, 4.0, 4.0, 4.0],
            &mut values,
            &mut columns,
            &mut count
        )
        .is_err()
    );
    assert_eq!((values, columns, count), previous);
    assert_eq!(p.workspace().reciprocal_previous_frame_mean(), 0.5);
    assert!(
        !Rows::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(1, 3))],
            &[4.0; 4],
            &mut values,
            &mut columns,
            &mut count
        )
        .unwrap()
        .is_complete()
    );
    let first = Rows::process_with_metadata(
        &mut p,
        &[Some(SampleMetadata::new(2, 0))],
        &[4.0; 4],
        &mut values,
        &mut columns,
        &mut count,
    )
    .unwrap();
    assert!(first.metadata().unwrap().is_discontinuous());
    p.replace_parameter(
        5,
        AlgorithmInput::from([true, false, true, true].as_slice()),
    )
    .unwrap()
    .unwrap();
    assert!(
        !Rows::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(2, 1))],
            &[4.0; 4],
            &mut values,
            &mut columns,
            &mut count
        )
        .unwrap()
        .is_complete()
    );
    assert_eq!(p.workspace().reciprocal_previous_frame_mean(), 0.5);
    p.reset();
    assert_eq!(p.workspace().reciprocal_previous_frame_mean(), 0.0);
}
#[test]
fn readout_late_terminal_mean_failure_never_publishes_or_commits() {
    let mut p = Readout::prepare(readout()).unwrap();
    let mut values = [71.0; 12];
    let mut columns = [999; 12];
    let mut count = [999];
    for ordinal in 0..2 {
        Readout::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(0, ordinal))],
            &[f32::MAX; 8],
            &mut values,
            &mut columns,
            &mut count,
        )
        .unwrap();
    }
    let previous = (values, columns, count);
    assert!(
        Readout::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(0, 2).with_terminal(true))],
            &[f32::MAX; 8],
            &mut values,
            &mut columns,
            &mut count
        )
        .is_err()
    );
    assert_eq!((values, columns, count), previous);
    assert_eq!(p.workspace().reciprocal_previous_frame_mean(), 0.0);
    let first = Readout::process_with_metadata(
        &mut p,
        &[Some(SampleMetadata::new(1, 0))],
        &[2.0; 8],
        &mut values,
        &mut columns,
        &mut count,
    )
    .unwrap();
    assert!(first.metadata().unwrap().is_discontinuous());
    p.reset();
    assert_eq!(p.workspace().reciprocal_previous_frame_mean(), 0.0);
}
#[test]
fn pupil_reconstructor_orders_four_complete_pupil_blocks_and_replaces_matrix() {
    let mut p = Dense::prepare(MC {
        reconstructed_count: 2,
        selected_pixel_count: 2,
        initial_reconstructor: Some(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0,
        ]),
        reconstructed_schema: "test.pyramid/reconstructed/1".into(),
    })
    .unwrap();
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = [71.0; 2];
    Dense::process(&mut p, &input, &mut output).unwrap();
    assert_eq!(output, [204.0, -20.0]);
    let mut bad = input;
    bad[7] = f32::NAN;
    assert!(Dense::process(&mut p, &bad, &mut output).is_err());
    assert_eq!(output, [204.0, -20.0]);
    p.replace_parameter(2, AlgorithmInput::from([1.0_f32; 16].as_slice()))
        .unwrap()
        .unwrap();
    Dense::process(&mut p, &input, &mut output).unwrap();
    assert_eq!(output, [36.0; 2]);
    p.reset();
    Dense::process(&mut p, &input, &mut output).unwrap();
    assert_eq!(output, [36.0; 2]);
    assert!(
        p.replace_parameter(2, AlgorithmInput::from([f32::NAN; 16].as_slice()))
            .unwrap()
            .is_err()
    );
}
#[test]
fn portable_configuration_extents_reject_before_allocation() {
    let mut c = rows();
    c.image_size = (u32::MAX as usize, 2);
    assert!(Rows::prepare(c).is_err());
    let mut c = readout();
    c.pupil_size = (usize::MAX, 2);
    assert!(Readout::prepare(c).is_err());
    assert!(
        Dense::prepare(MC {
            reconstructed_count: 1,
            selected_pixel_count: usize::MAX,
            initial_reconstructor: None,
            reconstructed_schema: "test.pyramid/reconstructed/1".into()
        })
        .is_err()
    );
}
