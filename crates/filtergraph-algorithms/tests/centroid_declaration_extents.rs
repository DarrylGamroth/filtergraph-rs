//! Invalid scalar output extents must return errors without publication or panic.
use filtergraph_algorithms::{CenterOfGravityF32 as C, CenterOfGravityF32Configuration};
#[test]
fn scalar_centroid_outputs_require_exact_declared_extents() {
    let mut p = C::prepare(CenterOfGravityF32Configuration {
        pixel_count: 2,
        initial_coordinates: Some(vec![[0.0, 0.0], [1.0, 1.0]]),
    })
    .unwrap();
    for (flux_len, valid_len) in [(0, 1), (1, 0), (2, 1), (1, 2)] {
        let mut out = [71.0; 2];
        let mut flux = vec![-17.0; flux_len];
        let mut validity = vec![true; valid_len];
        assert!(C::process(&mut p, &[2.0, 3.0], &mut out, &mut flux, &mut validity).is_err());
        assert_eq!(out, [71.0; 2]);
        assert!(flux.iter().all(|&v| v == -17.0));
        assert!(validity.iter().all(|&v| v));
    }
}
