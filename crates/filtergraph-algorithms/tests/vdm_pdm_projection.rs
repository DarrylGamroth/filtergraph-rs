//! VDM/PDM forward-command and reverse-feedback projection fixtures.

use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmPlan, PdmFeedbackToVdmPlan, PdmProjection, PdmProjectionError, PdmProjectionStorage,
    VdmExtrapolationArtifacts, VdmExtrapolationStorage, VdmToPdmError, VdmToPdmPlan,
    VdmTruncationStorage,
};

fn identity_matrix(size: usize) -> Arc<[f32]> {
    Arc::from(
        (0..size * size)
            .map(|index| {
                if index / size == index % size {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>(),
    )
}

fn two_pdm_plans() -> (VdmToPdmPlan<f32>, PdmFeedbackToVdmPlan<f32>) {
    let projections = Arc::from([
        PdmProjection::new(
            2,
            Arc::from([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            Arc::from([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        )
        .unwrap(),
        PdmProjection::new(1, Arc::from([0.0, 0.0, 1.0]), Arc::from([0.5, 0.5, 0.0])).unwrap(),
    ]);
    let decomposition = VdmToPdmPlan::new(
        2,
        3,
        Arc::from([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        Arc::clone(&projections),
    )
    .unwrap();
    let feedback =
        PdmFeedbackToVdmPlan::new(2, 3, Arc::from([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), projections)
            .unwrap();
    (decomposition, feedback)
}

#[test]
fn one_vdm_projects_to_two_different_pdm_extents() {
    let (plan, feedback_plan) = two_pdm_plans();
    let mut workspace = plan.workspace();
    let mut feedback_workspace = feedback_plan.workspace();
    let mut corrections = [0.0; 3];
    let mut feedback = [0.0; 2];

    plan.process(&mut corrections, &mut workspace, &[0.4, -0.2])
        .unwrap();
    feedback_plan
        .process(&mut feedback, &mut feedback_workspace, &[0.1, -0.2, 0.4])
        .unwrap();

    assert_eq!(plan.pdm_count(), 2);
    assert_eq!(plan.pdm_size(0), Some(2));
    assert_eq!(plan.pdm_size(1), Some(1));
    assert_eq!(plan.physical_actuator_extent(), 3);
    assert_eq!(corrections, [0.4, -0.2, 0.2]);
    assert!((feedback[0] - 0.3).abs() <= f32::EPSILON);
    assert_eq!(feedback[1], 0.0);
}

#[test]
fn forward_and_reverse_decomposition_can_run_at_separate_boundaries() {
    let (plan, feedback_plan) = two_pdm_plans();
    let mut workspace = plan.workspace();
    let mut feedback_workspace = feedback_plan.workspace();
    let mut corrections = [0.0; 3];
    let mut feedback = [0.0; 2];

    plan.process(&mut corrections, &mut workspace, &[0.25, 0.5])
        .unwrap();
    feedback_plan
        .process(&mut feedback, &mut feedback_workspace, &[0.0, 0.1, -0.2])
        .unwrap();

    assert_eq!(corrections, [0.25, 0.5, 0.75]);
    assert_eq!(feedback, [-0.1, 0.0]);
}

#[test]
fn float64_uses_the_same_multi_pdm_calculation() {
    let projections = Arc::from([
        PdmProjection::new(1, Arc::from([0.25]), Arc::from([1.0])).unwrap(),
        PdmProjection::new(1, Arc::from([0.75]), Arc::from([1.0])).unwrap(),
    ]);
    let plan = VdmToPdmPlan::new(1, 1, Arc::from([1.0_f64]), Arc::clone(&projections)).unwrap();
    let feedback_plan = PdmFeedbackToVdmPlan::new(1, 1, Arc::from([1.0_f64]), projections).unwrap();
    let mut workspace = plan.workspace();
    let mut feedback_workspace = feedback_plan.workspace();
    let mut corrections = [0.0; 2];
    let mut feedback = [0.0];

    plan.process(&mut corrections, &mut workspace, &[2.0])
        .unwrap();
    feedback_plan
        .process(&mut feedback, &mut feedback_workspace, &[0.1, 0.2])
        .unwrap();

    assert_eq!(corrections, [0.5, 1.5]);
    assert!((feedback[0] - 0.3).abs() <= f64::EPSILON);
}

#[test]
fn extrapolation_uses_the_smallest_prepared_representation() {
    let identity_projection =
        Arc::from([PdmProjection::new(10, identity_matrix(10), identity_matrix(10)).unwrap()]);
    let identity = VdmToPdmPlan::new(
        10,
        10,
        identity_matrix(10),
        Arc::clone(&identity_projection),
    )
    .unwrap();
    let mut sparse_matrix = vec![0.0; 100];
    sparse_matrix[0] = 0.25;
    sparse_matrix[1] = 0.75;
    sparse_matrix[5 * 10 + 3] = -2.0;
    let sparse = VdmToPdmPlan::new(
        10,
        10,
        Arc::from(sparse_matrix),
        Arc::clone(&identity_projection),
    )
    .unwrap();
    let dense = VdmToPdmPlan::new(10, 10, Arc::from(vec![0.5; 100]), identity_projection).unwrap();

    assert_eq!(
        identity.extrapolation_storage(),
        VdmExtrapolationStorage::Identity
    );
    assert_eq!(identity.extrapolation_coefficient_count(), 0);
    assert_eq!(
        sparse.extrapolation_storage(),
        VdmExtrapolationStorage::CompressedSparseRow
    );
    assert_eq!(sparse.extrapolation_coefficient_count(), 3);
    assert_eq!(
        dense.extrapolation_storage(),
        VdmExtrapolationStorage::RowMajor
    );
    assert_eq!(dense.extrapolation_coefficient_count(), 100);

    let mut workspace = sparse.workspace();
    let mut output = [0.0; 10];
    sparse
        .process(
            &mut output,
            &mut workspace,
            &[1.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
    assert_eq!(output, [2.5, 0.0, 0.0, 0.0, 0.0, -8.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn controlled_and_extrapolated_rows_use_dense_boundary_completion() {
    let size = 10;
    let mut extrapolation = vec![0.0; size * size];
    for &controlled in &[1, 4, 7] {
        extrapolation[controlled * size + controlled] = 1.0;
    }
    for extrapolated in [0, 2, 3, 5, 8, 9] {
        let row = &mut extrapolation[extrapolated * size..(extrapolated + 1) * size];
        row[1] = 0.25;
        row[4] = 0.5;
        row[7] = 0.25;
    }
    let projections =
        Arc::from([
            PdmProjection::new(size, identity_matrix(size), identity_matrix(size)).unwrap(),
        ]);
    let plan = VdmToPdmPlan::new(size, size, Arc::from(extrapolation), projections).unwrap();
    let mut workspace = plan.workspace();
    let mut output = [f32::NAN; 10];
    let mut input = [0.0; 10];
    input[1] = 2.0;
    input[4] = 4.0;
    input[7] = 10.0;

    plan.process(&mut output, &mut workspace, &input).unwrap();

    assert_eq!(
        plan.extrapolation_storage(),
        VdmExtrapolationStorage::ControlledExtrapolation
    );
    assert_eq!(plan.extrapolation_coefficient_count(), 18);
    match plan.extrapolation_artifacts() {
        VdmExtrapolationArtifacts::ControlledExtrapolation {
            controlled_rows,
            input_columns,
            extrapolated_rows,
            coefficients,
        } => {
            assert_eq!(controlled_rows, &[1, 4, 7]);
            assert_eq!(input_columns, Some(&[1, 4, 7][..]));
            assert_eq!(extrapolated_rows, &[0, 2, 3, 5, 8, 9]);
            assert_eq!(coefficients.len(), 18);
        }
        other => panic!("unexpected extrapolation: {other:?}"),
    }
    assert_eq!(output, [5.0, 2.0, 5.0, 5.0, 4.0, 5.0, 0.0, 10.0, 5.0, 5.0]);
}

#[test]
fn reordered_controlled_inputs_preserve_dense_completion_and_multi_pdm_order() {
    const ACTIVE: usize = 10;
    const FULL: usize = 21;
    let mut matrix = vec![0.0_f32; ACTIVE * FULL];
    for position in 0..ACTIVE {
        matrix[(2 * position + 1) * ACTIVE + (ACTIVE - 1 - position)] = 1.0;
        for column in 0..ACTIVE {
            matrix[2 * position * ACTIVE + column] = (column + 1) as f32;
        }
    }
    let mut first = vec![0.0; 2 * FULL];
    first[0] = 1.0;
    first[FULL + 1] = 1.0;
    let mut second = vec![0.0; 2 * FULL];
    second[19] = 1.0;
    second[FULL + 20] = 1.0;
    let projections = Arc::from([
        PdmProjection::new(2, Arc::from(first), Arc::from(vec![0.0; 2 * FULL])).unwrap(),
        PdmProjection::new(2, Arc::from(second), Arc::from(vec![0.0; 2 * FULL])).unwrap(),
    ]);
    let plan = VdmToPdmPlan::new(ACTIVE, FULL, Arc::from(matrix.clone()), projections).unwrap();
    assert_eq!(
        plan.extrapolation_storage(),
        VdmExtrapolationStorage::ControlledExtrapolation
    );
    assert_eq!(plan.extrapolation_coefficient_count(), ACTIVE * ACTIVE);
    match plan.extrapolation_artifacts() {
        VdmExtrapolationArtifacts::ControlledExtrapolation {
            controlled_rows,
            input_columns,
            extrapolated_rows,
            coefficients,
        } => {
            assert_eq!(controlled_rows, &[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);
            assert_eq!(input_columns, Some(&[9, 8, 7, 6, 5, 4, 3, 2, 1, 0][..]));
            assert_eq!(extrapolated_rows, &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
            assert_eq!(coefficients.len(), ACTIVE * ACTIVE);
        }
        other => panic!("unexpected extrapolation: {other:?}"),
    }

    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let dense_first = (0..ACTIVE)
        .map(|column| matrix[column] * input[column])
        .sum::<f32>();
    let mut owner_a = plan.workspace();
    let mut owner_b = plan.workspace();
    let mut output_a = [f32::NAN; 4];
    let mut output_b = [f32::NAN; 4];
    plan.process(&mut output_a, &mut owner_a, &input).unwrap();
    plan.process(&mut output_b, &mut owner_b, &[2.0; ACTIVE])
        .unwrap();
    assert_eq!(output_a, [dense_first, 10.0, 1.0, 0.0]);
    assert_eq!(output_b, [110.0, 2.0, 2.0, 0.0]);
    plan.process(&mut output_b, &mut owner_b, &input).unwrap();
    assert_eq!(output_b, output_a);

    let mut overflow = matrix;
    overflow[0] = f32::MAX;
    let overflow_plan = plan.with_active_to_full(Arc::from(overflow)).unwrap();
    assert_eq!(
        overflow_plan.extrapolation_storage(),
        VdmExtrapolationStorage::ControlledExtrapolation
    );
    assert_eq!(
        overflow_plan.process(&mut output_a, &mut owner_a, &[2.0; ACTIVE]),
        Err(VdmToPdmError::NonFiniteOutput)
    );
    let mut invalid = vec![0.0; ACTIVE * FULL];
    invalid[0] = f32::NAN;
    assert_eq!(
        plan.with_active_to_full(Arc::from(invalid)),
        Err(PdmProjectionError::NonFiniteConfiguration)
    );
}

#[test]
fn plans_with_equal_dimensions_share_workspace_across_storage_shapes() {
    let size = 10;
    let mut extrapolation = vec![0.0; size * size];
    for &controlled in &[1, 4, 7] {
        extrapolation[controlled * size + controlled] = 1.0;
    }
    for extrapolated in [0, 2, 3, 5, 8, 9] {
        let row = &mut extrapolation[extrapolated * size..(extrapolated + 1) * size];
        row[1] = 0.25;
        row[4] = 0.5;
        row[7] = 0.25;
    }
    let projection =
        Arc::from([
            PdmProjection::new(size, identity_matrix(size), identity_matrix(size)).unwrap(),
        ]);
    let partitioned = VdmToPdmPlan::new(
        size,
        size,
        Arc::from(extrapolation),
        Arc::clone(&projection),
    )
    .unwrap();
    let dense = VdmToPdmPlan::new(size, size, Arc::from(vec![0.5; 100]), projection).unwrap();
    let mut incompatible_workspace = dense.workspace();

    let mut output = [f32::NAN; 10];
    assert!(
        partitioned
            .process(&mut output, &mut incompatible_workspace, &[0.0; 10])
            .is_ok()
    );
    assert_eq!(output, [0.0; 10]);
}

#[test]
fn identity_pdm_projection_and_vdm_row_selection_elide_dense_matrices() {
    let projection = PdmProjection::new(3, identity_matrix(3), identity_matrix(3)).unwrap();
    assert_eq!(
        projection.vdm_to_pdm_storage(),
        PdmProjectionStorage::Identity
    );
    assert_eq!(
        projection.pdm_to_vdm_storage(),
        PdmProjectionStorage::Identity
    );
    let feedback = PdmFeedbackToVdmPlan::new(
        2,
        3,
        Arc::from([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        Arc::from([projection]),
    )
    .unwrap();
    assert_eq!(
        feedback.truncation_storage(),
        VdmTruncationStorage::RowSelection
    );
    let mut workspace = feedback.workspace();
    let mut output = [0.0; 2];

    feedback
        .process(&mut output, &mut workspace, &[2.0, 4.0, 8.0])
        .unwrap();

    assert_eq!(output, [4.0, 8.0]);
}

#[test]
fn correction_provenance_tracks_prepared_replacements() {
    use filtergraph_algorithms::VdmToPdmProjection;
    let completion = Arc::from([1., 0., 0., 1., 0.5, 0.5, 0., 0., 1., 0.]);
    let projection = VdmToPdmProjection::new(5, identity_matrix(5)).unwrap();
    let plan = VdmToPdmPlan::from_vdm_to_pdm_projections(2, 5, completion, Arc::from([projection]))
        .unwrap();
    assert_eq!(plan.correction_roles(), &[1, 1, 2, 0, 2]);
    let mut mixed = identity_matrix(5).to_vec();
    mixed[1] = 0.5;
    let replacement = plan
        .with_vdm_to_pdm_projections(Arc::from([
            VdmToPdmProjection::new(5, Arc::from(mixed)).unwrap()
        ]))
        .unwrap();
    assert_eq!(replacement.correction_roles(), &[3, 1, 2, 0, 2]);
    let completion = Arc::from([1., 0., 0., 1., 0., 0., 0., 0., 1., 0.]);
    assert_eq!(
        plan.with_active_to_full(completion)
            .unwrap()
            .correction_roles(),
        &[1, 1, 0, 0, 2]
    );
}
