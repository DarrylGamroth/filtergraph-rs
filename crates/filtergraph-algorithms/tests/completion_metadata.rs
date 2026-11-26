//! Terminal publication must retain scientific acquisition qualifications.
use filtergraph_algorithms::*;

fn rich_sample(offset: u32) -> SampleMetadata {
    let identity = SampleAcquisitionIdentity::new([9; 16], 4, 77).unwrap();
    let exposure = SampleExposureStart::new(42, SampleClock::Tai);
    SampleMetadata::new(177, offset)
        .with_acquisition_identity(Some(identity))
        .with_chopped_image_kind(Some(ChoppedImageKind::Mixed))
        .with_trigger_state(Some(SampleTriggerState::Mixed))
        .with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(123))
        .with_exposure_start(Some(exposure))
        .with_reference_timestamp(Some(
            filtergraph_algorithms::SampleReferenceTimestamp::new(
                u64::MAX - 1,
                None,
                filtergraph_algorithms::SampleClock::Realtime,
            )
            .unwrap(),
        ))
        .with_timestamp_nanoseconds(Some(1234))
        .with_terminal(offset == 2)
        .with_discontinuity(offset == 0)
}

#[test]
fn terminal_frontend_publication_preserves_all_optional_metadata() {
    macro_rules! check {
        ($declaration:ty,$configuration:expr,$extent:expr,$inputs:expr,$offset:ident,$($port:expr),+)=>{{
            let mut prepared=<$declaration>::prepare($configuration).unwrap();
            let mut output=vec![-99.0;$extent];
            for $offset in 0..3 {
                let sample=rich_sample($offset);
                let mut metadata=[None;$inputs];metadata[0]=Some(sample);
                let result=<$declaration>::process_with_metadata(&mut prepared,&metadata,$($port),+,&mut output).unwrap();
                assert_eq!(result.is_complete(),$offset==2);
                if $offset==2 {
                    let expected=sample.with_offset(0).with_terminal(true).with_corrupted(false).with_discontinuity(true);
                    assert_eq!(result.metadata(),Some(expected),"{} dropped acquisition qualifications",stringify!($declaration));
                }
            }
        }};
    }
    check!(
        DetectorFrameAssemblerF32,
        DetectorFrameAssemblerF32Configuration {
            region_lines: 3,
            block_lines: 1,
            region_samples: 4,
            detector_origins: vec![[0, 0], [5, 3]],
            detector_line_steps: vec![[1, 0], [-1, 0]],
            detector_sample_steps: vec![[0, 1], [0, -1]],
            detector_size: (6, 4),
        },
        24,
        1,
        offset,
        &[2.0_f32; 8]
    );
    check!(
        ShackHartmannReadoutReconstructorF32,
        ShackHartmannReadoutReconstructorF32Configuration {
            region_lines: 3,
            block_lines: 1,
            region_samples: 4,
            detector_origins: vec![[0, 0], [5, 3]],
            detector_line_steps: vec![[1, 0], [-1, 0]],
            detector_sample_steps: vec![[0, 1], [0, -1]],
            subaperture_count: 3,
            reconstructed_count: 2,
            initial_subaperture_origins: Some(vec![[3, 0], [0, 2], [1, 0]]),
            coordinate_scale: 1.25,
            pixel_threshold: 1.0,
            flux_threshold: 0.5,
            reference_slopes: None,
            active: None,
            initial_reconstructor: Some(vec![1.0; 12]),
            reconstructed_schema: "test.metadata/reconstructed/1".into(),
            subaperture_size: (2, 2),
            detector_size: (6, 4),
        },
        2,
        1,
        offset,
        &[2.0_f32; 8]
    );
    check!(
        IncrementalDenseReconstructorF32,
        IncrementalDenseReconstructorF32Configuration {
            measurement_count: 2,
            measurement_block_capacity: 1,
            blocks_per_sample: 3,
            reconstructed_count: 1,
            initial_reconstructor: Some(vec![2.0, 3.0]),
            measurements_schema: "test.metadata/measurements/1".into(),
            column_indices_schema: "test.metadata/columns/1".into(),
            measurement_count_schema: "test.metadata/count/1".into(),
            reconstructed_schema: "test.metadata/reconstructed/1".into(),
            reconstructor_schema: "test.metadata/matrix/1".into(),
        },
        1,
        3,
        offset,
        &[2.0_f32],
        &[if offset == 2 { 1 } else { 0 }],
        &[u32::from(offset != 1)]
    );
}
