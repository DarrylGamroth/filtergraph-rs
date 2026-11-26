//! Current JFG raw ADU schema, exact sums, order and rejection boundaries.
use filtergraph_algorithms::{
    AlgorithmPlan, NoWorkspace, PyramidPupilFluxError as E, PyramidPupilFluxPlan as P,
    PyramidPupilFluxU16F64 as D, PyramidPupilFluxU16F64Configuration as C,
};
fn config() -> C {
    C {
        image_size: (4, 6),
        pupil_size: (2, 3),
        pupil_origins: vec![[0, 0], [0, 3], [2, 0], [2, 3]],
    }
}
#[test]
fn exact_raw_sums_preserve_origin_order_and_reset() {
    let mut d = D::prepare(config()).unwrap();
    let mut out = [0.0; 4];
    let raw: Vec<u16> = (1..=24).collect();
    D::process(&mut d, &raw, &mut out).unwrap();
    assert_eq!(out, [30.0, 48.0, 102.0, 120.0]);
    D::process(&mut d, &[u16::MAX; 24], &mut out).unwrap();
    assert_eq!(out, [393210.0; 4]);
    d.reset();
    D::process(&mut d, &raw, &mut out).unwrap();
    assert_eq!(out, [30.0, 48.0, 102.0, 120.0]);
    assert_eq!(
        d.formats()[0].schema(),
        filtergraph_algorithms::schemas::RAW_DETECTOR_PIXELS_V1
    );
    assert_eq!(
        d.formats()[1].schema(),
        filtergraph_algorithms::schemas::PYRAMID_PUPIL_FLUX_ADU_F64_V1
    );
    let p = P::new((4, 6), (2, 3), vec![[0, 3], [0, 3], [2, 0], [0, 0]]).unwrap();
    p.process(&mut out, &mut NoWorkspace, &raw).unwrap();
    assert_eq!(out, [48.0, 48.0, 102.0, 30.0]);
}
#[test]
fn invalid_geometry_and_extent_reject_before_publication() {
    for c in [
        C {
            pupil_origins: vec![[0, 0]; 3],
            ..config()
        },
        C {
            pupil_origins: vec![[3, 0]; 4],
            ..config()
        },
        C {
            pupil_size: (0, 3),
            ..config()
        },
    ] {
        assert!(D::prepare(c).is_err());
    }
    assert!(matches!(
        P::new((500000, 500000), (500000, 500000), vec![[0, 0]; 4]),
        Err(E::ExactSum)
    ));
    assert!(P::new((usize::MAX, 2), (usize::MAX, 2), vec![[0, 0]; 4]).is_err());
    let p = P::new((4, 6), (2, 3), config().pupil_origins).unwrap();
    let mut out = [71.0; 4];
    assert_eq!(
        p.process(&mut out, &mut NoWorkspace, &[0; 23]),
        Err(E::PortExtent)
    );
    assert_eq!(out, [71.0; 4]);
    assert_eq!(
        p.process(&mut out[..3], &mut NoWorkspace, &[0; 24]),
        Err(E::PortExtent)
    );
    assert_eq!(out, [71.0; 4]);
}
