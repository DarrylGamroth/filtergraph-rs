//! Scientific-order streaming publication and calibration lifecycle.
#[path = "../examples/support/fixture_configuration.rs"]
mod fixture_configuration;

use filtergraph_algorithms::{
    AlgorithmInput, SampleMetadata, ShackHartmannWeightedMeasurementBlockF32 as Block,
    ShackHartmannWeightedMeasurementBlockF32Configuration as Configuration,
};

fn configuration() -> Configuration {
    Configuration {
        row_block_rows: 2,
        subaperture_count: 3,
        initial_subaperture_origins: Some(vec![[0, 0], [3, 1], [1, 3]]),
        coordinate_scale: 1.25,
        pixel_threshold: 1.0,
        flux_threshold: 0.5,
        reference_slopes: Some(vec![0.1, -0.2, 0.25, 0.5, -0.15, 0.3]),
        active: Some(vec![true, false, true]),
        initial_weights: vec![
            vec![1.0, 0.5, 2.0, 1.5],
            vec![0.25, 1.0, 2.0, 0.75],
            vec![2.0, 1.0, 0.5, 1.0],
        ],
        measurements_schema: "test.filtergraph/measurement-block/1".into(),
        column_indices_schema: "test.filtergraph/measurement-column-indices/1".into(),
        measurement_count_schema: "test.filtergraph/measurement-count/1".into(),
        image_size: (6, 5),
        subaperture_size: (2, 2),
    }
}

#[test]
fn publishes_physical_offsets_empty_prefixes_and_untouched_tails() {
    let mut p = Block::prepare(configuration()).unwrap();
    let pixels = vec![2.0; 10];
    for (row, expected) in [(0, 2), (2, 0), (4, 4)] {
        let mut values = [-99.0; 6];
        let mut columns = [999; 6];
        let mut count = [999];
        let metadata = SampleMetadata::new(7, row)
            .with_terminal(row == 4)
            .with_timestamp_nanoseconds(Some(333));
        let outcome = Block::process_with_metadata(
            &mut p,
            &[Some(metadata)],
            &pixels,
            &mut values,
            &mut columns,
            &mut count,
        )
        .unwrap();
        assert_eq!(outcome.metadata(), Some(metadata));
        assert_eq!(count[0], expected);
        let start = if row == 4 { 2 } else { 0 };
        for (i, column) in columns[..expected as usize].iter().enumerate() {
            assert_eq!(*column, start + i as u32);
        }
        assert!(columns[expected as usize..].iter().all(|&v| v == 999));
        assert!(values[expected as usize..].iter().all(|&v| v == -99.0));
        if row == 4 {
            assert_eq!(&values[..2], &[0.0, 0.0]);
        }
    }
}

#[test]
fn reset_is_clean_but_parameter_adoption_abandons_partial_frame() {
    let mut p = Block::prepare(configuration()).unwrap();
    let pixels = [2.0; 10];
    let mut values = [0.0; 6];
    let mut columns = [0; 6];
    let mut count = [0];
    let mut call = |p: &mut _, sequence, row| {
        Block::process_with_metadata(
            p,
            &[Some(SampleMetadata::new(sequence, row))],
            &pixels,
            &mut values,
            &mut columns,
            &mut count,
        )
        .unwrap()
    };
    call(&mut p, 1, 0);
    p.reset();
    assert!(!call(&mut p, 2, 0).metadata().unwrap().is_discontinuous());
    p.replace_parameter(9, AlgorithmInput::from([1.0_f32; 12].as_slice()))
        .unwrap()
        .unwrap();
    assert!(!call(&mut p, 2, 2).is_complete());
    assert!(call(&mut p, 3, 0).metadata().unwrap().is_discontinuous());
    assert!(
        p.replace_parameter(8, AlgorithmInput::from([true].as_slice()))
            .unwrap()
            .is_err()
    );
    assert!(call(&mut p, 3, 2).is_complete());
}

#[test]
fn rejected_storage_preserves_output_and_invalid_pixel_keeps_column_identity() {
    let mut p = Block::prepare(configuration()).unwrap();
    let mut values = [-99.0; 6];
    let mut columns = [999; 6];
    let mut count = [999];
    let metadata = SampleMetadata::new(4, 0);
    assert!(
        Block::process_with_metadata(
            &mut p,
            &[Some(metadata)],
            &[2.0; 9],
            &mut values,
            &mut columns,
            &mut count
        )
        .is_err()
    );
    assert_eq!(values, [-99.0; 6]);
    assert_eq!(columns, [999; 6]);
    assert_eq!(count, [999]);
    let mut pixels = [2.0; 10];
    pixels[0] = f32::NAN;
    assert!(
        Block::process_with_metadata(
            &mut p,
            &[Some(metadata)],
            &pixels,
            &mut values,
            &mut columns,
            &mut count
        )
        .unwrap()
        .is_complete()
    );
    assert_eq!(count, [2]);
    assert_eq!(&columns[..2], &[0, 1]);
    assert_eq!(&values[..2], &[0.0, 0.0]);
}

#[test]
fn readout_publishes_ready_nonprefix_scientific_columns_and_reset_marks_loss() {
    use filtergraph_algorithms::{
        ShackHartmannReadoutMeasurementBlockF32 as Readout,
        ShackHartmannReadoutMeasurementBlockF32Configuration as Configuration,
    };
    let mut p = Readout::prepare(Configuration {
        detector_size: (6, 5),
        region_lines: 3,
        block_lines: 1,
        region_samples: 5,
        detector_origins: vec![[0, 0], [5, 4]],
        detector_line_steps: vec![[1, 0], [-1, 0]],
        detector_sample_steps: vec![[0, 1], [0, -1]],
        subaperture_count: 3,
        initial_subaperture_origins: Some(vec![[3, 1], [0, 0], [1, 3]]),
        coordinate_scale: 1.0,
        pixel_threshold: 0.0,
        flux_threshold: 0.0,
        reference_slopes: None,
        active: None,
        measurements_schema: "test.filtergraph/measurement-block/1".into(),
        column_indices_schema: "test.filtergraph/measurement-column-indices/1".into(),
        measurement_count_schema: "test.filtergraph/measurement-count/1".into(),
        subaperture_size: (2, 2),
    })
    .unwrap();
    let mut values = [-99.0; 6];
    let mut columns = [999; 6];
    let mut count = [999];
    for ordinal in 0..3 {
        let sample = SampleMetadata::new(19, ordinal).with_terminal(ordinal == 2);
        let outcome = Readout::process_with_metadata(
            &mut p,
            &[Some(sample)],
            &[2.0; 10],
            &mut values,
            &mut columns,
            &mut count,
        )
        .unwrap();
        assert_eq!(outcome.metadata(), Some(sample));
        match ordinal {
            0 => assert_eq!(count, [0]),
            1 => {
                assert_eq!(count, [2]);
                assert_eq!(&columns[..2], &[2, 3]);
            }
            _ => {
                assert_eq!(count, [4]);
                assert_eq!(&columns[..4], &[0, 1, 4, 5]);
            }
        }
    }
    p.reset();
    let outcome = Readout::process_with_metadata(
        &mut p,
        &[Some(SampleMetadata::new(20, 0))],
        &[2.0; 10],
        &mut values,
        &mut columns,
        &mut count,
    )
    .unwrap();
    assert!(outcome.metadata().unwrap().is_discontinuous());
}

#[test]
fn row_reconstruction_defers_preserves_output_on_overflow_and_recovers() {
    use filtergraph_algorithms::{
        ShackHartmannRowReconstructorF32 as Row,
        ShackHartmannRowReconstructorF32Configuration as Configuration,
    };
    let mut p = Row::prepare(Configuration {
        row_block_rows: 2,
        subaperture_count: 3,
        initial_subaperture_origins: Some(vec![[0, 0], [3, 1], [1, 3]]),
        coordinate_scale: 1.0,
        pixel_threshold: 0.0,
        flux_threshold: 0.0,
        reference_slopes: Some(vec![-10.0; 6]),
        active: None,
        reconstructed_count: 2,
        initial_reconstructor: Some(vec![f32::MAX; 12]),
        reconstructed_schema: "test.filtergraph/reconstructed/1".into(),
        image_size: (6, 5),
        subaperture_size: (2, 2),
    })
    .unwrap();
    let mut output = [-99.0; 2];
    assert!(
        Row::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(10, 0))],
            &[2.0; 10],
            &mut output
        )
        .is_err()
    );
    assert_eq!(output, [-99.0; 2]);
    assert!(
        !Row::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(10, 2))],
            &[2.0; 10],
            &mut output
        )
        .unwrap()
        .is_complete()
    );
    p.replace_parameter(7, AlgorithmInput::from([1.0_f32; 12].as_slice()))
        .unwrap()
        .unwrap();
    for row in [0, 2, 4] {
        let outcome = Row::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(11, row).with_terminal(row == 4))],
            &[2.0; 10],
            &mut output,
        )
        .unwrap();
        assert_eq!(outcome.is_complete(), row == 4);
        if row != 4 {
            assert_eq!(output, [-99.0; 2]);
        } else {
            assert_eq!(output, [60.0; 2]);
            assert_eq!(
                outcome.metadata(),
                Some(
                    SampleMetadata::new(11, 0)
                        .with_terminal(true)
                        .with_discontinuity(true)
                )
            );
        }
    }
    p.reset();
    assert!(
        !Row::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(12, 0))],
            &[2.0; 10],
            &mut output
        )
        .unwrap()
        .is_complete()
    );
    assert!(
        p.replace_parameter(4, AlgorithmInput::from([f32::NAN; 6].as_slice()))
            .unwrap()
            .is_err()
    );
    assert!(
        !Row::process_with_metadata(
            &mut p,
            &[Some(SampleMetadata::new(12, 2))],
            &[2.0; 10],
            &mut output
        )
        .unwrap()
        .is_complete()
    );
    let outcome = Row::process_with_metadata(
        &mut p,
        &[Some(SampleMetadata::new(12, 4).with_terminal(true))],
        &[2.0; 10],
        &mut output,
    )
    .unwrap();
    assert!(!outcome.metadata().unwrap().is_discontinuous());
}

#[test]
fn unweighted_incompatible_workspace_is_rejected_before_frame_mutation() {
    use filtergraph_algorithms::{
        RowBlockInfo, ShackHartmannImagePlan, ShackHartmannMeasurementBlockPlan,
    };
    let sensing = |rows, columns| {
        ShackHartmannImagePlan::from_detector_regions(
            (rows, columns),
            (1, 1),
            &[(0, 0)],
            1.0_f32,
            0.0,
            0.0,
            None,
            None,
        )
        .unwrap()
    };
    let plan = ShackHartmannMeasurementBlockPlan::new(sensing(6, 5), 2).unwrap();
    let other = ShackHartmannMeasurementBlockPlan::new(sensing(2, 2), 1).unwrap();
    let mut workspace = other.workspace();
    let mut values = [-99.0; 2];
    let mut columns = [999; 2];
    let mut count = [999];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plan.process_block(
            &mut values,
            &mut columns,
            &mut count,
            &mut workspace,
            &[2.0; 10],
            RowBlockInfo {
                sequence: 7,
                first_row: 0,
                terminal: false,
                discontinuity: false,
                corrupted: false,
            },
        )
    }));
    assert!(
        result.is_ok(),
        "incompatible workspace panicked instead of returning an extent error"
    );
    assert!(result.unwrap().is_err());
    assert_eq!(workspace.sequence(), None);
    assert_eq!(values, [-99.0; 2]);
    assert_eq!(columns, [999; 2]);
    assert_eq!(count, [999]);
}

#[test]
fn weighted_nested_scratch_rejection_does_not_create_discontinuity() {
    use filtergraph_algorithms::{RowBlockInfo, ShackHartmannWeightedMeasurementBlockResult};
    let target = Block::prepare(configuration()).unwrap();
    let mut c = configuration();
    c.subaperture_size.1 = 1;
    c.initial_weights = vec![vec![1.0; 2]; 3];
    let other = Block::prepare(c).unwrap();
    let mut workspace = other.plan().workspace();
    let info = RowBlockInfo {
        sequence: 9,
        first_row: 0,
        terminal: false,
        discontinuity: false,
        corrupted: false,
    };
    let mut values = [-99.0; 6];
    let mut columns = [999; 6];
    let mut count = [999];
    assert!(
        target
            .plan()
            .process_block(
                &mut values,
                &mut columns,
                &mut count,
                &mut workspace,
                &[2.0; 10],
                info
            )
            .is_err()
    );
    assert_eq!(values, [-99.0; 6]);
    assert_eq!(columns, [999; 6]);
    assert_eq!(count, [999]);
    assert_eq!(
        other
            .plan()
            .process_block(
                &mut values,
                &mut columns,
                &mut count,
                &mut workspace,
                &[2.0; 10],
                info
            )
            .unwrap(),
        ShackHartmannWeightedMeasurementBlockResult::Published {
            discontinuity: false
        }
    );
}

#[test]
fn missing_metadata_abandons_all_four_streaming_histories() {
    use filtergraph_algorithms::*;
    // Shared geometry is projected into each concrete declaration configuration.
    let source = r#"
image_rows=6
image_columns=5
row_block_rows=2
detector_rows=6
detector_columns=5
region_lines=3
block_lines=1
region_samples=5
detector_origins=[[0,0],[5,4]]
detector_line_steps=[[1,0],[-1,0]]
detector_sample_steps=[[0,1],[0,-1]]
subaperture_rows=2
subaperture_columns=2
subaperture_count=3
initial_subaperture_origins=[[0,0],[3,1],[1,3]]
coordinate_scale=1.0
pixel_threshold=0.0
flux_threshold=0.0
initial_weights=[[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0]]
measurements_schema="test.filtergraph/measurements/1"
column_indices_schema="test.filtergraph/columns/1"
measurement_count_schema="test.filtergraph/count/1"
reconstructed_count=2
initial_reconstructor=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
reconstructed_schema="test.filtergraph/reconstructed/1"
"#;
    fn configured(
        source: &str,
        readout: bool,
        weighted: bool,
        reconstruction: bool,
    ) -> toml::Value {
        let mut value: toml::Value = toml::from_str(source).unwrap();
        value.as_table_mut().unwrap().retain(|key, _| {
            let row = ["image_rows", "image_columns", "row_block_rows"].contains(&key);
            let detector = key.starts_with("detector_")
                || ["region_lines", "block_lines", "region_samples"].contains(&key);
            let output = [
                "measurements_schema",
                "column_indices_schema",
                "measurement_count_schema",
            ]
            .contains(&key);
            let reconstructed = [
                "reconstructed_count",
                "initial_reconstructor",
                "reconstructed_schema",
            ]
            .contains(&key);
            (!row || !readout)
                && (!detector || readout)
                && (!output || !reconstruction)
                && (!reconstructed || reconstruction)
                && (key != "initial_weights" || weighted)
        });
        value
    }
    macro_rules! check {
        ($declaration:ty,$configuration:ty,$stride:expr,$($extra:expr),*)=>{{
            let mut p=<$declaration>::prepare(fixture_configuration::configuration::<$declaration>(&configured(source, $stride == 1, stringify!($declaration).contains("Weighted"), false), &configured(source, $stride == 1, stringify!($declaration).contains("Weighted"), false).as_table().unwrap().keys().map(String::as_str).collect::<Vec<_>>()).unwrap()).unwrap();
            let mut values=[-99.0;6];
            let mut call=|p:&mut _,sample| <$declaration>::process_with_metadata(p,&[sample],&[2.0;10],&mut values,$($extra),*);
            call(&mut p,Some(SampleMetadata::new(7,0))).unwrap();
            assert!(call(&mut p,None).is_err());
            // Retrying the missing offset cannot complete the abandoned sample.
            for offset in [$stride,2*$stride] {
                let outcome=call(&mut p,Some(SampleMetadata::new(7,offset).with_terminal(offset==2*$stride))).unwrap();
                assert!(!outcome.is_complete(),"{} retained abandoned history",stringify!($declaration));
                assert!(outcome.metadata().is_none());
            }
            let outcome=call(&mut p,Some(SampleMetadata::new(8,0))).unwrap();
            assert!(outcome.metadata().unwrap().is_discontinuous());
        }};
    }
    let mut columns = [999; 6];
    let mut count = [999];
    check!(
        ShackHartmannMeasurementBlockF32,
        ShackHartmannMeasurementBlockF32Configuration,
        2,
        &mut columns,
        &mut count
    );
    check!(
        ShackHartmannWeightedMeasurementBlockF32,
        ShackHartmannWeightedMeasurementBlockF32Configuration,
        2,
        &mut columns,
        &mut count
    );
    check!(
        ShackHartmannReadoutMeasurementBlockF32,
        ShackHartmannReadoutMeasurementBlockF32Configuration,
        1,
        &mut columns,
        &mut count
    );
    // The reconstruction declaration withholds output until terminal completion.
    let mut p = ShackHartmannRowReconstructorF32::prepare(
        fixture_configuration::configuration::<ShackHartmannRowReconstructorF32>(
            &configured(source, false, false, true),
            &configured(source, false, false, true)
                .as_table()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut values = [-99.0; 2];
    let mut call = |p: &mut _, sample| {
        ShackHartmannRowReconstructorF32::process_with_metadata(
            p,
            &[sample],
            &[2.0; 10],
            &mut values,
        )
    };
    call(&mut p, Some(SampleMetadata::new(7, 0))).unwrap();
    assert!(call(&mut p, None).is_err());
    for offset in [2, 4] {
        assert!(
            !call(
                &mut p,
                Some(SampleMetadata::new(7, offset).with_terminal(offset == 4))
            )
            .unwrap()
            .is_complete()
        );
    }
    for offset in [0, 2, 4] {
        let outcome = call(
            &mut p,
            Some(SampleMetadata::new(8, offset).with_terminal(offset == 4)),
        )
        .unwrap();
        assert_eq!(outcome.is_complete(), offset == 4);
        if offset == 4 {
            assert!(outcome.metadata().unwrap().is_discontinuous());
        }
    }
}

fn replace_calibration<D: filtergraph_algorithms::AlgorithmDeclaration>(
    prepared: &mut filtergraph_algorithms::PreparedAlgorithm<D>,
) where
    D::ParameterError: core::fmt::Debug,
{
    let origins = [0_u32, 0, 0, 2, 2, 2];
    let coordinates = [2.0_f32, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0];
    let reference = [1.0_f32; 6];
    let thresholds = [0.0_f32, 0.0, 3.0, 0.0, 0.0, 0.0];
    let active = [true, true, false];
    let updates = [
        ("subaperture-origins", AlgorithmInput::from(&origins[..])),
        ("coordinates", AlgorithmInput::from(&coordinates[..])),
        ("reference-slopes", AlgorithmInput::from(&reference[..])),
        ("thresholds", AlgorithmInput::from(&thresholds[..])),
        ("active", AlgorithmInput::from(&active[..])),
    ];
    let parameters = updates.map(|(name, input)| {
        let index = D::PORTS
            .iter()
            .position(|port| port.name() == name)
            .unwrap();
        (index, input)
    });
    prepared.replace_parameters(&parameters).unwrap();
}

#[test]
fn weighted_row_calibration_updates_scientific_columns_without_changing_capacity() {
    let mut prepared = Block::prepare(configuration()).unwrap();
    replace_calibration(&mut prepared);
    let mut measurements = [-99.0; 6];
    let mut columns = [999; 6];
    let mut count = [999];
    assert_eq!(prepared.plan().capacity(), 6);
    for (row, expected_columns, expected_values) in [
        (0, &[0, 1, 2, 3][..], &[1.0, 2.0, 0.0, 0.0][..]),
        (2, &[4, 5][..], &[0.0, 0.0][..]),
        (4, &[][..], &[][..]),
    ] {
        let result = Block::process_with_metadata(
            &mut prepared,
            &[Some(SampleMetadata::new(31, row).with_terminal(row == 4))],
            &[2.0; 10],
            &mut measurements,
            &mut columns,
            &mut count,
        )
        .unwrap();
        assert!(result.is_complete());
        assert_eq!(count[0] as usize, expected_columns.len());
        assert_eq!(&columns[..count[0] as usize], expected_columns);
        assert_eq!(&measurements[..count[0] as usize], expected_values);
    }
}

#[test]
fn opposed_readout_and_row_reconstruction_apply_the_same_calibration_transaction() {
    use filtergraph_algorithms::{
        ShackHartmannReadoutMeasurementBlockF32 as Readout,
        ShackHartmannReadoutMeasurementBlockF32Configuration as ReadoutConfig,
        ShackHartmannRowReconstructorF32 as Row,
        ShackHartmannRowReconstructorF32Configuration as RowConfig,
    };
    let geometry = configuration();
    let mut readout = Readout::prepare(ReadoutConfig {
        detector_size: geometry.image_size,
        region_lines: 3,
        block_lines: 1,
        region_samples: 5,
        detector_origins: vec![[0, 0], [5, 4]],
        detector_line_steps: vec![[1, 0], [-1, 0]],
        detector_sample_steps: vec![[0, 1], [0, -1]],
        subaperture_count: geometry.subaperture_count,
        initial_subaperture_origins: geometry.initial_subaperture_origins.clone(),
        coordinate_scale: geometry.coordinate_scale,
        pixel_threshold: geometry.pixel_threshold,
        flux_threshold: geometry.flux_threshold,
        reference_slopes: None,
        active: None,
        measurements_schema: geometry.measurements_schema,
        column_indices_schema: geometry.column_indices_schema,
        measurement_count_schema: geometry.measurement_count_schema,
        subaperture_size: geometry.subaperture_size,
    })
    .unwrap();
    let mut row = Row::prepare(RowConfig {
        row_block_rows: geometry.row_block_rows,
        subaperture_count: geometry.subaperture_count,
        initial_subaperture_origins: geometry.initial_subaperture_origins,
        coordinate_scale: geometry.coordinate_scale,
        pixel_threshold: geometry.pixel_threshold,
        flux_threshold: geometry.flux_threshold,
        reference_slopes: None,
        active: None,
        reconstructed_count: 1,
        initial_reconstructor: Some(vec![1.0; 6]),
        reconstructed_schema: "test.filtergraph/reconstructed/1".into(),
        image_size: geometry.image_size,
        subaperture_size: geometry.subaperture_size,
    })
    .unwrap();

    replace_calibration(&mut readout);
    replace_calibration(&mut row);
    let mut measurements = [-99.0; 6];
    let mut columns = [999; 6];
    let mut count = [999];
    let mut scientific_order = [-99.0; 6];
    let mut reconstructed = [-99.0];
    for ordinal in 0..3 {
        let terminal = ordinal == 2;
        assert!(
            Readout::process_with_metadata(
                &mut readout,
                &[Some(
                    SampleMetadata::new(41, ordinal).with_terminal(terminal)
                )],
                &[2.0; 10],
                &mut measurements,
                &mut columns,
                &mut count,
            )
            .unwrap()
            .is_complete()
        );
        for i in 0..count[0] as usize {
            scientific_order[columns[i] as usize] = measurements[i];
        }
        let outcome = Row::process_with_metadata(
            &mut row,
            &[Some(
                SampleMetadata::new(41, 2 * ordinal).with_terminal(terminal),
            )],
            &[2.0; 10],
            &mut reconstructed,
        )
        .unwrap();
        assert_eq!(outcome.is_complete(), terminal);
        if !terminal {
            assert_eq!(reconstructed, [-99.0]);
        }
    }
    assert_eq!(scientific_order, [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(reconstructed, [scientific_order.into_iter().sum::<f32>()]);
}
