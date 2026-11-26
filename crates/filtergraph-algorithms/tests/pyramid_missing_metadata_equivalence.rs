//! Missing required identity must abandon an active frame, as JFG discard! does.
#[path = "../examples/support/fixture_configuration.rs"]
mod fixture_configuration;

fn configuration<D: filtergraph_algorithms::AlgorithmDeclaration>(source: &str) -> D::Configuration
where
    D::Configuration: serde::de::DeserializeOwned,
{
    let value: toml::Value = toml::from_str(source).unwrap();
    let fields: Vec<_> = value
        .as_table()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    fixture_configuration::configuration::<D>(&value, &fields).unwrap()
}

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmOutput, AlgorithmProcessResult, ElementType,
    NativeProcessError, PortDirection, PreparedAlgorithm, SampleMetadata,
};
#[derive(Clone, Debug, PartialEq)]
enum Buffer {
    F32(Vec<f32>),
    U32(Vec<u32>),
}
fn call<D: AlgorithmDeclaration>(
    p: &mut PreparedAlgorithm<D>,
    input: &[f32],
    sample: Option<SampleMetadata>,
) -> Result<AlgorithmProcessResult, NativeProcessError<D::ProcessError>> {
    let mut buffers: Vec<_> = D::PORTS
        .iter()
        .zip(p.formats())
        .filter(|(port, _)| port.direction() == PortDirection::Output)
        .map(|(_, f)| match f.element_type() {
            ElementType::F32 => Buffer::F32(vec![71.0; f.elements()]),
            ElementType::U32 => Buffer::U32(vec![999; f.elements()]),
            _ => panic!("unexpected pyramid output"),
        })
        .collect();
    let before = buffers.clone();
    let mut outputs: Vec<_> = buffers
        .iter_mut()
        .map(|b| match b {
            Buffer::F32(v) => AlgorithmOutput::from(v.as_mut_slice()),
            Buffer::U32(v) => AlgorithmOutput::from(v.as_mut_slice()),
        })
        .collect();
    let result = p.process_with_metadata(&[AlgorithmInput::from(input)], &[sample], &mut outputs);
    drop(outputs);
    if result.is_err() || result.as_ref().is_ok_and(|r| !r.is_complete()) {
        assert_eq!(buffers, before);
    }
    result
}
fn check<D: AlgorithmDeclaration>(mut p: PreparedAlgorithm<D>, width: usize, blocks: u32)
where
    D::ProcessError: core::fmt::Debug,
{
    let input = vec![2.0_f32; width];
    call(&mut p, &input, Some(SampleMetadata::new(1, 0))).unwrap();
    assert!(call(&mut p, &input, None).is_err());
    // Retry the missing block: skipping ahead would conceal retained assembly.
    for offset in 1..blocks {
        let result = call(
            &mut p,
            &input,
            Some(SampleMetadata::new(1, offset).with_terminal(offset + 1 == blocks)),
        )
        .unwrap();
        assert!(
            !result.is_complete(),
            "missing identity must discard the active frame"
        );
    }
    let mut saw_discontinuity = false;
    for offset in 0..blocks {
        let result = call(
            &mut p,
            &input,
            Some(SampleMetadata::new(2, offset).with_terminal(offset + 1 == blocks)),
        )
        .unwrap();
        if let Some(m) = result.metadata() {
            saw_discontinuity |= m.is_discontinuous();
        }
        if offset + 1 == blocks {
            assert!(result.is_complete());
        }
    }
    assert!(saw_discontinuity);
    p.reset();
    let identity = filtergraph_algorithms::SampleAcquisitionIdentity::new([1; 16], 7, 9).unwrap();
    let start = filtergraph_algorithms::SampleExposureStart::new(
        42,
        filtergraph_algorithms::SampleClock::Tai,
    );
    let mut saw_rich = false;
    for offset in 0..blocks {
        let sample = SampleMetadata::new(9, offset)
            .with_acquisition_identity(Some(identity))
            .with_trigger_state(Some(filtergraph_algorithms::SampleTriggerState::High))
            .with_chopped_image_kind(Some(filtergraph_algorithms::ChoppedImageKind::Mixed))
            .with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(123))
            .with_exposure_start(Some(start))
            .with_reference_timestamp(Some(
                filtergraph_algorithms::SampleReferenceTimestamp::new(
                    u64::MAX - 1,
                    None,
                    filtergraph_algorithms::SampleClock::Realtime,
                )
                .unwrap(),
            ))
            .with_timestamp_nanoseconds(Some(12345))
            .with_terminal(offset + 1 == blocks)
            .with_discontinuity(offset == 0);
        let result = call(&mut p, &input, Some(sample)).unwrap();
        if let Some(metadata) = result.metadata() {
            let expected = if D::NAME.contains("measurement-block") {
                sample
            } else {
                sample
                    .with_offset(0)
                    .with_terminal(true)
                    .with_discontinuity(true)
                    .with_corrupted(false)
            };
            assert_eq!(
                metadata, expected,
                "all scientific sample metadata must survive publication"
            );
            saw_rich = true;
        }
    }
    assert!(saw_rich);
}
const PUPILS: &str = "pupil_origins=[[4,2],[2,0],[4,0],[2,2]]\npupil_mask=[true,false,true,true]\n";
const ROWS: &str = "image_size=[6,4]\nrow_block_rows=1\npupil_size=[2,2]\n";
const READOUT: &str = "region_lines=3\nblock_lines=1\nregion_samples=4\ndetector_origins=[[0,0],[3,0]]\ndetector_line_steps=[[1,0],[1,0]]\ndetector_sample_steps=[[0,1],[0,1]]\n";
const SCHEMAS: &str = "measurements_schema='test.pyramid/measurements/1'\ncolumn_indices_schema='test.pyramid/columns/1'\nmeasurement_count_schema='test.pyramid/count/1'\n";
#[test]
fn pixel_rows() {
    let c = format!("{ROWS}{PUPILS}");
    check(
        filtergraph_algorithms::PyramidPixelRowF32::prepare(configuration::<
            filtergraph_algorithms::PyramidPixelRowF32,
        >(&c))
        .unwrap(),
        4,
        6,
    );
}
#[test]
fn row_reconstruction() {
    let c = format!(
        "{ROWS}{PUPILS}reconstructed_count=1\nreconstructed_schema='test.pyramid/reconstructed/1'\n"
    );
    check(
        filtergraph_algorithms::PyramidRowReconstructorF32::prepare(configuration::<
            filtergraph_algorithms::PyramidRowReconstructorF32,
        >(&c))
        .unwrap(),
        4,
        6,
    );
}
#[test]
fn readout_reconstruction() {
    let c = format!(
        "detector_size=[6,4]\npupil_size=[2,2]\n{PUPILS}{READOUT}reconstructed_count=1\nreconstructed_schema='test.pyramid/reconstructed/1'\n"
    );
    check(
        filtergraph_algorithms::PyramidReadoutReconstructorF32::prepare(configuration::<
            filtergraph_algorithms::PyramidReadoutReconstructorF32,
        >(&c))
        .unwrap(),
        8,
        3,
    );
}
#[test]
fn row_measurements() {
    let c = format!("image_size=[6,4]\nrow_block_rows=1\npupil_size=[2,2]\n{PUPILS}{SCHEMAS}");
    check(
        filtergraph_algorithms::PyramidMeasurementBlockF32::prepare(configuration::<
            filtergraph_algorithms::PyramidMeasurementBlockF32,
        >(&c))
        .unwrap(),
        4,
        6,
    );
}
#[test]
fn readout_measurements() {
    let c = format!("detector_size=[6,4]\npupil_size=[2,2]\n{PUPILS}{READOUT}{SCHEMAS}");
    check(
        filtergraph_algorithms::PyramidReadoutMeasurementBlockF32::prepare(configuration::<
            filtergraph_algorithms::PyramidReadoutMeasurementBlockF32,
        >(&c))
        .unwrap(),
        8,
        3,
    );
}
