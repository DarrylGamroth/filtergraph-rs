//! Signed detector calibration boundaries from the pinned JFG contract.
use filtergraph_algorithms::{
    AlgorithmInput, DeclarationProcessError, PixelCalibrationF32 as P,
    PixelCalibrationF32Configuration as Configuration, PixelCalibrationF32Error as E,
};
fn configuration() -> Configuration {
    Configuration {
        image_size: (2, 3),
        input_schema: "org.example.detector-signed/1".into(),
        initial_flat: None,
        initial_background: None,
    }
}
#[test]
fn configuration_validates_dimensions_schema_and_finite_planes() {
    for image_size in [
        (0, 3),
        (2, 0),
        (usize::MAX, 2),
        (u32::MAX as usize / 4 + 1, 1),
    ] {
        let mut c = configuration();
        c.image_size = image_size;
        assert!(P::prepare(c).is_err());
    }
    for schema in ["", "  ", "invalid-schema"] {
        let mut c = configuration();
        c.input_schema = schema.into();
        assert!(P::prepare(c).is_err());
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for plane in 0..2 {
            let mut c = configuration();
            let mut values = vec![1.0; 6];
            values[5] = invalid;
            if plane == 0 {
                c.initial_flat = Some(values)
            } else {
                c.initial_background = Some(values)
            };
            assert!(P::prepare(c).is_err());
        }
    }
    for plane in 0..2 {
        let mut c = configuration();
        if plane == 0 {
            c.initial_flat = Some(vec![1.0; 5])
        } else {
            c.initial_background = Some(vec![1.0; 5])
        };
        assert!(P::prepare(c).is_err());
    }
}
#[test]
fn signed_samples_replacements_reset_and_rejected_replacement_preserve_coefficients() {
    let mut p = P::prepare(configuration()).unwrap();
    let raw = [-0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut out = [77.0; 6];
    P::process(&mut p, &raw, &mut out).unwrap();
    assert_eq!(out, raw);
    let mut flat = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let background = [0.5_f32; 6];
    p.replace_parameter(2, AlgorithmInput::from(flat.as_slice()))
        .unwrap()
        .unwrap();
    p.replace_parameter(3, AlgorithmInput::from(background.as_slice()))
        .unwrap()
        .unwrap();
    flat.fill(99.0);
    P::process(&mut p, &raw, &mut out).unwrap();
    assert_eq!(out, [-1.0, 1.0, 4.5, 10.0, 17.5, 27.0]);
    for port in [2, 3] {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = [1.0; 6];
            values[5] = invalid;
            assert!(
                p.replace_parameter(port, AlgorithmInput::from(values.as_slice()))
                    .unwrap()
                    .is_err()
            );
        }
        assert!(
            p.replace_parameter(port, AlgorithmInput::from([1.0_f32; 5].as_slice()))
                .unwrap()
                .is_err()
        );
    }
    p.reset();
    P::process(&mut p, &raw, &mut out).unwrap();
    assert_eq!(out, [-1.0, 1.0, 4.5, 10.0, 17.5, 27.0]);
}
#[test]
fn nonfinite_and_overflow_preserve_output_and_recover() {
    let mut c = configuration();
    c.initial_flat = Some(vec![2.0; 6]);
    let mut p = P::prepare(c).unwrap();
    let raw = [-0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut out = [77.0; 6];
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let mut bad = raw;
        bad[5] = invalid;
        out.fill(77.0);
        let expected = if invalid.is_finite() {
            E::NonFiniteOutput
        } else {
            E::NonFiniteInput
        };
        assert!(
            matches!(P::process(&mut p,&bad,&mut out),Err(DeclarationProcessError::Algorithm(error)) if error==expected)
        );
        assert_eq!(out, [77.0; 6]);
        P::process(&mut p, &raw, &mut out).unwrap();
        assert_eq!(out, [-1.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    }
    out.fill(77.0);
    assert!(P::process(&mut p, &raw[..5], &mut out).is_err());
    assert_eq!(out, [77.0; 6]);
    assert!(P::process(&mut p, &raw, &mut out[..5]).is_err());
    assert_eq!(out, [77.0; 6]);
}
#[test]
fn first_rejection_uses_dense_julia_column_order_in_native_row_storage() {
    let mut c = configuration();
    c.initial_flat = Some(vec![2.0; 6]);
    let mut p = P::prepare(c).unwrap();
    let mut out = [77.0; 6];
    // Logical [1,0] is scanned before [0,1], opposite their row-major indices.
    let mut raw = [1.0; 6];
    raw[3] = f32::MAX;
    raw[1] = f32::NAN;
    assert!(matches!(
        P::process(&mut p, &raw, &mut out),
        Err(DeclarationProcessError::Algorithm(E::NonFiniteOutput))
    ));
    assert_eq!(out, [77.0; 6]);
    raw[3] = f32::NAN;
    raw[1] = f32::MAX;
    assert!(matches!(
        P::process(&mut p, &raw, &mut out),
        Err(DeclarationProcessError::Algorithm(E::NonFiniteInput))
    ));
    assert_eq!(out, [77.0; 6]);
}

#[test]
fn tile_boundaries_preserve_logical_failure_priority_and_recovery() {
    let mut c = configuration();
    c.image_size = (3, 259);
    c.initial_flat = Some(vec![2.0; 777]);
    let mut p = P::prepare(c).unwrap();
    let mut raw = vec![1.0_f32; 777];
    let mut output = vec![71.0; 777];
    // Row-major tile zero sees a NaN first; logical column zero's last row
    // takes precedence even though it resides in the third physical tile.
    raw[1] = f32::NAN;
    raw[518] = f32::MAX;
    assert!(matches!(
        P::process(&mut p, &raw, &mut output),
        Err(DeclarationProcessError::Algorithm(E::NonFiniteOutput))
    ));
    assert_eq!(output, vec![71.0; 777]);
    raw.fill(-3.0);
    P::process(&mut p, &raw, &mut output).unwrap();
    assert_eq!(output, vec![-6.0; 777]);
}
