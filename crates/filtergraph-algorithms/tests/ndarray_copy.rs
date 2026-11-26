//! Exact-copy numerical, extent, and declaration regression checks.

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmPlan, NdarrayCopyError, NdarrayCopyF32,
    NdarrayCopyF32Configuration, NdarrayCopyPlan, NoWorkspace,
};

#[test]
fn copy_preserves_every_float_bit_pattern() {
    let bits = [
        0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc00001, 0xffc12345, 0x3f123456,
    ];
    let input = bits.map(f32::from_bits);
    let mut output = [0.0; 8];
    NdarrayCopyPlan::new(8)
        .unwrap()
        .process(&mut output, &mut NoWorkspace, &input)
        .unwrap();
    assert_eq!(output.map(f32::to_bits), bits);
}
#[test]
fn invalid_extents_are_checked_before_any_output_write() {
    assert_eq!(NdarrayCopyPlan::new(0), Err(NdarrayCopyError::Empty));
    let plan = NdarrayCopyPlan::new(2).unwrap();
    let mut output = [71.0; 2];
    assert_eq!(
        plan.process(&mut output, &mut NoWorkspace, &[1.0]),
        Err(NdarrayCopyError::InputExtent)
    );
    assert_eq!(output, [71.0; 2]);
    assert_eq!(
        plan.process(&mut output[..1], &mut NoWorkspace, &[1.0, 2.0]),
        Err(NdarrayCopyError::OutputExtent)
    );
    assert_eq!(output, [71.0; 2]);
}
#[test]
fn declaration_preserves_configured_schema_and_metadata() {
    let schema = "pipewireao/dm-command-opd-um-f32/1";
    let mut prepared = NdarrayCopyF32::prepare(NdarrayCopyF32Configuration {
        extent: 277,
        schema: schema.into(),
    })
    .unwrap();
    assert_eq!(NdarrayCopyF32::NAME, "ndarray-copy-f32");
    assert_eq!(NdarrayCopyF32::PORTS[1].metadata_source(), Some("input"));
    for format in prepared.formats() {
        assert_eq!(format.shape(), &[277]);
        assert_eq!(format.schema(), schema);
    }
    let input = [0.125; 277];
    let mut output = [0.0; 277];
    NdarrayCopyF32::process(&mut prepared, &input, &mut output).unwrap();
    assert_eq!(output, input);
    for schema in ["", "invalid schema/1", "missing-version"] {
        assert!(
            NdarrayCopyF32::prepare(NdarrayCopyF32Configuration {
                extent: 2,
                schema: schema.into()
            })
            .is_err()
        );
    }
}
