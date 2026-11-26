//! Prepared progressive matrix-packing tests.

use std::sync::Arc;

use filtergraph_algorithms::{
    PreparedGemv, PreparedProgressiveGemv, PreparedProgressiveGemvError, ProgressiveGemvError,
};

#[test]
fn preparation_packs_canonical_columns_by_completion_ordinal() {
    let canonical = PreparedGemv::new(
        2,
        4,
        Arc::from([1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();
    let plan = PreparedProgressiveGemv::new(canonical.clone(), &[1, 0, 1, 0], 2).unwrap();

    assert_eq!(plan.canonical().matrix(), canonical.matrix());
    assert_eq!(plan.input_indices(0), Some([1, 3].as_slice()));
    assert_eq!(plan.input_indices(1), Some([0, 2].as_slice()));
    assert_eq!(
        plan.coefficients(),
        [2.0, 20.0, 4.0, 40.0, 1.0, 10.0, 3.0, 30.0]
    );

    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut progressive = [0.0_f32; 2];
    plan.accumulate_group(&mut progressive, &input, 0).unwrap();
    assert_eq!(progressive, [20.0, 200.0]);
    plan.accumulate_group(&mut progressive, &input, 1).unwrap();

    let mut complete = [0.0_f32; 2];
    canonical.multiply(&mut complete, &input).unwrap();
    assert_eq!(progressive, complete);
}

#[test]
fn preparation_rejects_incomplete_or_invalid_grouping() {
    let canonical = PreparedGemv::new(1, 2, Arc::from([1.0_f32, 2.0])).unwrap();

    assert_eq!(
        PreparedProgressiveGemv::new(canonical.clone(), &[0, 0], 0).unwrap_err(),
        PreparedProgressiveGemvError::EmptyGroups
    );
    assert_eq!(
        PreparedProgressiveGemv::new(canonical.clone(), &[0], 1).unwrap_err(),
        PreparedProgressiveGemvError::CompletionExtent {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        PreparedProgressiveGemv::new(canonical, &[0, 1], 1).unwrap_err(),
        PreparedProgressiveGemvError::OrdinalRange {
            input: 1,
            ordinal: 1,
            groups: 1,
        }
    );
}

#[test]
fn invalid_group_input_is_rejected_before_output_mutation() {
    let canonical = PreparedGemv::new(1, 2, Arc::from([1.0_f32, 2.0])).unwrap();
    let plan = PreparedProgressiveGemv::new(canonical, &[0, 1], 2).unwrap();
    let mut output = [7.0_f32];

    assert_eq!(
        plan.accumulate_group(&mut output, &[f32::NAN, 2.0], 0),
        Err(ProgressiveGemvError::NonFiniteInput { input: 0 })
    );
    assert_eq!(output, [7.0]);
    assert_eq!(
        plan.accumulate_group(&mut output, &[1.0, 2.0], 2),
        Err(ProgressiveGemvError::OrdinalRange {
            ordinal: 2,
            groups: 2,
        })
    );
    assert_eq!(output, [7.0]);
}
