//! The current JFG I4Q declaration is distinct from four-pupil pixel reconstruction.
use filtergraph_algorithms::{
    AlgorithmInput, PyramidReconstructorF32 as D, PyramidReconstructorF32Configuration as C,
};
#[test]
fn i4q_shape_schema_column_order_and_replacement_match_reference() {
    let configuration:C=toml::from_str("reconstructed_count=1\nselected_coordinate_count=3\ninitial_reconstructor=[1.0,2.0,4.0,8.0,16.0,32.0]\nreconstructed_schema='test.pyramid/reconstructed/1'\n").expect("current source configuration accepts selected_coordinate_count");
    let mut p = D::prepare(configuration).unwrap();
    assert_eq!(p.formats()[0].shape(), &[3, 2]);
    assert_eq!(
        p.formats()[0].schema(),
        filtergraph_algorithms::schemas::PYRAMID_I4Q_SIGNAL_V1
    );
    let mut output = [71.0];
    let input = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0];
    D::process(&mut p, &input, &mut output).unwrap();
    assert_eq!(output, [1377.0]);
    let mut bad = input;
    bad[5] = f32::NAN;
    assert!(D::process(&mut p, &bad, &mut output).is_err());
    assert_eq!(output, [1377.0]);
    p.replace_parameter(
        2,
        AlgorithmInput::from([1.0_f32, 0.0, 0.0, 0.0, 0.0, -1.0].as_slice()),
    )
    .unwrap()
    .unwrap();
    D::process(&mut p, &input, &mut output).unwrap();
    assert_eq!(output, [-29.0]);
    assert!(
        p.replace_parameter(2, AlgorithmInput::from([f32::NAN; 6].as_slice()))
            .unwrap()
            .is_err()
    );
    p.reset();
    D::process(&mut p, &input, &mut output).unwrap();
    assert_eq!(output, [-29.0]);
}

#[test]
fn finite_cancellation_does_not_become_backend_overflow() {
    // Actual pinned JFG BLAS result is finite: (-1 + 0.5 - 0.25) * MAX.
    let configuration:C=toml::from_str("reconstructed_count=1\nselected_coordinate_count=3\ninitial_reconstructor=[-1.0,0.0,0.5,0.0,-0.25,0.0]\nreconstructed_schema='test.pyramid/reconstructed/1'\n").unwrap();
    let mut p = D::prepare(configuration).unwrap();
    let mut output = [71.0_f32];
    D::process(&mut p, &[f32::MAX; 6], &mut output)
        .expect("reference finite cancellation must remain accepted");
    assert_eq!(output, [-0.75 * f32::MAX]);
}
