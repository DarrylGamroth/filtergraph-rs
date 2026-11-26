//! Scientific vector boundaries and rejection/publication regressions.
use filtergraph_algorithms::{
    AlgorithmPlan, VectorLinearCombinationError as E, VectorLinearCombinationPlan as C,
    VectorOffsetPlan as O,
};
#[test]
fn validates_configuration_without_allocating_invalid_extents() {
    for n in [0, u32::MAX as usize / 4 + 1] {
        assert_eq!(C::new(n, 1.0, 1.0), Err(E::Extent));
    }
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(C::new(2, scale, 1.0), Err(E::NonFiniteConfiguration));
        assert_eq!(C::new(2, 1.0, scale), Err(E::NonFiniteConfiguration));
    }
}
#[test]
fn rejects_zero_scaled_nonfinite_and_preserves_output_on_last_coordinate_overflow() {
    let p = C::new(3, 0.0, 2.0).unwrap();
    let mut w = p.workspace();
    let mut out = [71.0; 3];
    for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            p.process(&mut out, &mut w, (&[1.0, 2.0, v], &[1.0; 3])),
            Err(E::NonFiniteInput)
        );
        assert_eq!(out, [71.0; 3]);
    }
    assert_eq!(
        p.process(&mut out, &mut w, (&[1.0; 3], &[1.0, 2.0, f32::MAX])),
        Err(E::NonFiniteOutput)
    );
    assert_eq!(out, [71.0; 3]);
    p.process(&mut out, &mut w, (&[1.0; 3], &[1.0, 2.0, 3.0]))
        .unwrap();
    assert_eq!(out, [2.0, 4.0, 6.0]);
    w.reset();
    p.process(&mut out, &mut w, (&[1.0; 3], &[1.0, 2.0, 3.0]))
        .unwrap();
    assert_eq!(out, [2.0, 4.0, 6.0]);
}
#[test]
fn all_dimension_errors_precede_publication_including_incompatible_workspace() {
    let p = C::new(3, 1.0, 1.0).unwrap();
    let mut w = p.workspace();
    let mut out = [71.0; 3];
    assert_eq!(
        p.process(&mut out, &mut w, (&[1.0; 2], &[1.0; 3])),
        Err(E::DimensionMismatch)
    );
    assert_eq!(
        p.process(&mut out, &mut w, (&[1.0; 3], &[1.0; 2])),
        Err(E::DimensionMismatch)
    );
    let mut incompatible = C::new(2, 1.0, 1.0).unwrap().workspace();
    assert_eq!(
        p.process(&mut out, &mut incompatible, (&[1.0; 3], &[1.0; 3])),
        Err(E::DimensionMismatch)
    );
    assert_eq!(out, [71.0; 3]);
}
#[test]
fn offset_reset_retains_adopted_parameter_and_rejection_preserves_both_outputs() {
    let p = O::new(3).unwrap().with_offset(&[0.5, -1.0, 2.0]).unwrap();
    let mut w = p.workspace();
    let mut out = [71.0; 3];
    let mut current = [71.0; 3];
    assert_eq!(p.with_offset(&[1.0; 2]), Err(E::DimensionMismatch));
    assert_eq!(
        p.with_offset(&[1.0, 2.0, f32::NAN]),
        Err(E::NonFiniteConfiguration)
    );
    assert_eq!(
        p.process((&mut out, &mut current), &mut w, &[1.0, 2.0, f32::INFINITY]),
        Err(E::NonFiniteInput)
    );
    assert_eq!(out, [71.0; 3]);
    assert_eq!(current, [71.0; 3]);
    assert_eq!(
        p.process((&mut out, &mut current[..2]), &mut w, &[1.0; 3]),
        Err(E::DimensionMismatch)
    );
    assert_eq!(out, [71.0; 3]);
    assert_eq!(current, [71.0; 3]);
    w.reset();
    p.process((&mut out, &mut current), &mut w, &[1.0; 3])
        .unwrap();
    assert_eq!(out, [1.5, 0.0, 3.0]);
    assert_eq!(current, [0.5, -1.0, 2.0]);
}

#[test]
fn offset_finite_overflow_preserves_both_outputs_and_recovers() {
    let p = O::new(2).unwrap().with_offset(&[1.0, f32::MAX]).unwrap();
    let mut w = p.workspace();
    let mut output = [71.0; 2];
    let mut current = [71.0; 2];
    assert_eq!(
        p.process((&mut output, &mut current), &mut w, &[1.0, f32::MAX]),
        Err(E::NonFiniteOutput)
    );
    assert_eq!(output, [71.0; 2]);
    assert_eq!(current, [71.0; 2]);
    p.process((&mut output, &mut current), &mut w, &[2.0, 0.0])
        .unwrap();
    assert_eq!(output, [3.0, f32::MAX]);
    assert_eq!(current, [1.0, f32::MAX]);
}
#[test]
fn declarations_reject_empty_scientific_schemas() {
    use filtergraph_algorithms::{
        VectorLinearCombinationF32, VectorLinearCombinationF32Configuration, VectorOffsetF32,
        VectorOffsetF32Configuration,
    };
    assert!(
        VectorOffsetF32::prepare(VectorOffsetF32Configuration {
            extent: 2,
            schema: "".into()
        })
        .is_err()
    );
    assert!(
        VectorLinearCombinationF32::prepare(VectorLinearCombinationF32Configuration {
            extent: 2,
            left_scale: 1.0,
            right_scale: 1.0,
            schema: "".into()
        })
        .is_err()
    );
}
