//! Public immutable artifacts required to prepare another numerical backend.

use std::{any::type_name, collections::BTreeSet, sync::Arc};

use filtergraph_algorithms::{
    AlgorithmPlan, CenterOfGravityPlan, ChoppedImageDifferencePlan, ClosedLoopCorrectionPlan,
    ControllerProjection, ControllerToVdmPlan, DeformableMirrorCommandPlan,
    DocrimeBinaryExcitationPlan, DocrimeClosedLoopInteractionMatrixPlan,
    DocrimeConditioningCriteria, DocrimeCovariancePlan, DocrimeExcitationPlan,
    DocrimeOpenLoopInteractionMatrixPlan, HadamardInteractionMatrixPlan, HadamardPushPullPlan,
    InterActuatorStrokeLimit, InterActuatorStrokeLimitPlan, InteractionMatrixStabilityCriteria,
    InteractionMatrixStabilityPlan, LeakyIntegratorPlan, MatchedCalibrationDifferencingPlan,
    ModalPushPullPlan, OpticalGainAssignment, OpticalGainCorrectionPlan, OpticalGainDitherMode,
    OpticalGainDitherPlan, OpticalGainDitherSchedule, OpticalGainEstimatorPlan,
    OpticalGainUpdateCriteria, OpticalGainUpdatePlan, PdmCommandPlan, PdmCommandPowerLimitPlan,
    PdmFeedbackToVdmPlan, PdmProjection, PdmProjectionArtifacts, PdmStrokeOptimizationPlan,
    PixelCalibrationPlan, PixelSelectionPlan, PreparedGemv, PseudoOpenLoopReconstructionPlan,
    PyramidPixelPlan, RegionExtractionPlan, SccPushPullResponsePlan, ShackHartmannPlan,
    VdmExtrapolationArtifacts, VdmFeedbackToControllerPlan, VdmToPdmPlan, VdmTruncationArtifacts,
    ZonalInteractionMatrixPlan, ZonalPushPullPlan,
};

fn record_plan<P: AlgorithmPlan>(plans: &mut BTreeSet<&'static str>) {
    assert!(plans.insert(type_name::<P>()));
}

#[test]
fn inventory_names_every_production_core_algorithm_plan() {
    let mut plans = BTreeSet::new();

    record_plan::<ModalPushPullPlan<f32>>(&mut plans);
    record_plan::<ZonalPushPullPlan<f32>>(&mut plans);
    record_plan::<HadamardPushPullPlan<f32>>(&mut plans);
    record_plan::<InteractionMatrixStabilityPlan<f32>>(&mut plans);
    record_plan::<MatchedCalibrationDifferencingPlan<f32>>(&mut plans);
    record_plan::<ZonalInteractionMatrixPlan<f32>>(&mut plans);
    record_plan::<HadamardInteractionMatrixPlan<f32>>(&mut plans);
    record_plan::<ClosedLoopCorrectionPlan<f32>>(&mut plans);
    record_plan::<LeakyIntegratorPlan<f32>>(&mut plans);
    record_plan::<PseudoOpenLoopReconstructionPlan<f32>>(&mut plans);
    record_plan::<ControllerToVdmPlan<f32>>(&mut plans);
    record_plan::<InterActuatorStrokeLimitPlan<f32>>(&mut plans);
    record_plan::<PdmCommandPlan<f32>>(&mut plans);
    record_plan::<PdmCommandPowerLimitPlan<f32>>(&mut plans);
    record_plan::<PdmFeedbackToVdmPlan<f32>>(&mut plans);
    record_plan::<PdmStrokeOptimizationPlan<f32>>(&mut plans);
    record_plan::<VdmFeedbackToControllerPlan<f32>>(&mut plans);
    record_plan::<VdmToPdmPlan<f32>>(&mut plans);
    record_plan::<DocrimeCovariancePlan<f32>>(&mut plans);
    record_plan::<DocrimeExcitationPlan<f32>>(&mut plans);
    record_plan::<DocrimeBinaryExcitationPlan<f32>>(&mut plans);
    record_plan::<DocrimeOpenLoopInteractionMatrixPlan<f32>>(&mut plans);
    record_plan::<DocrimeClosedLoopInteractionMatrixPlan<f32>>(&mut plans);
    record_plan::<PixelCalibrationPlan<f32, u16>>(&mut plans);
    record_plan::<PixelSelectionPlan<f32>>(&mut plans);
    record_plan::<CenterOfGravityPlan<f32>>(&mut plans);
    record_plan::<RegionExtractionPlan<f32>>(&mut plans);
    record_plan::<OpticalGainCorrectionPlan<f32>>(&mut plans);
    record_plan::<OpticalGainDitherPlan<f32>>(&mut plans);
    record_plan::<OpticalGainEstimatorPlan<f32>>(&mut plans);
    record_plan::<OpticalGainUpdatePlan<f32>>(&mut plans);
    record_plan::<ChoppedImageDifferencePlan<f32>>(&mut plans);
    record_plan::<SccPushPullResponsePlan<f32>>(&mut plans);
    record_plan::<PyramidPixelPlan<f32>>(&mut plans);
    record_plan::<ShackHartmannPlan<f32>>(&mut plans);
    record_plan::<DeformableMirrorCommandPlan<f32>>(&mut plans);

    assert_eq!(plans.len(), 36);
}

fn materialize_pdm_projection(
    artifacts: PdmProjectionArtifacts<'_, f32>,
    rows: usize,
    columns: usize,
) -> Vec<f32> {
    match artifacts {
        PdmProjectionArtifacts::Identity => {
            assert_eq!(rows, columns);
            let mut matrix = vec![0.0; rows * columns];
            for index in 0..rows {
                matrix[index * columns + index] = 1.0;
            }
            matrix
        }
        PdmProjectionArtifacts::RowMajor(coefficients) => coefficients.to_vec(),
    }
}

fn materialize_extrapolation(
    artifacts: VdmExtrapolationArtifacts<'_, f32>,
    full_vdm_size: usize,
    active_vdm_size: usize,
) -> Vec<f32> {
    let mut matrix = vec![0.0; full_vdm_size * active_vdm_size];
    match artifacts {
        VdmExtrapolationArtifacts::Identity => {
            assert_eq!(full_vdm_size, active_vdm_size);
            for index in 0..active_vdm_size {
                matrix[index * active_vdm_size + index] = 1.0;
            }
        }
        VdmExtrapolationArtifacts::ControlledExtrapolation {
            controlled_rows,
            input_columns,
            extrapolated_rows,
            coefficients,
        } => {
            for (position, &row) in controlled_rows.iter().enumerate() {
                let column = input_columns.map_or(position, |columns| columns[position]);
                matrix[row * active_vdm_size + column] = 1.0;
            }
            for (coefficient_row, &row) in extrapolated_rows.iter().enumerate() {
                for (position, &coefficient) in coefficients
                    [coefficient_row * controlled_rows.len()..][..controlled_rows.len()]
                    .iter()
                    .enumerate()
                {
                    let column = input_columns.map_or(position, |columns| columns[position]);
                    matrix[row * active_vdm_size + column] = coefficient;
                }
            }
        }
        VdmExtrapolationArtifacts::RowMajor(coefficients) => {
            matrix.copy_from_slice(coefficients);
        }
        VdmExtrapolationArtifacts::CompressedSparseRow {
            row_offsets,
            column_indices,
            coefficients,
        } => {
            for (row, bounds) in row_offsets.windows(2).enumerate() {
                for entry in bounds[0]..bounds[1] {
                    matrix[row * active_vdm_size + column_indices[entry]] = coefficients[entry];
                }
            }
        }
    }
    matrix
}

fn materialize_truncation(
    artifacts: VdmTruncationArtifacts<'_, f32>,
    active_vdm_size: usize,
    full_vdm_size: usize,
) -> Vec<f32> {
    let mut matrix = vec![0.0; active_vdm_size * full_vdm_size];
    match artifacts {
        VdmTruncationArtifacts::Identity => {
            assert_eq!(active_vdm_size, full_vdm_size);
            for index in 0..active_vdm_size {
                matrix[index * full_vdm_size + index] = 1.0;
            }
        }
        VdmTruncationArtifacts::RowSelection(columns) => {
            for (row, &column) in columns.iter().enumerate() {
                matrix[row * full_vdm_size + column] = 1.0;
            }
        }
        VdmTruncationArtifacts::RowMajor(coefficients) => {
            matrix.copy_from_slice(coefficients);
        }
    }
    matrix
}

#[test]
fn calibration_control_and_docrime_plans_expose_complete_preparation() {
    let modal = ModalPushPullPlan::new(1, 2, [1.0_f32, -0.5]).unwrap();
    assert_eq!(modal.mode_count(), 1);
    assert_eq!(modal.corrector_coordinate_count(), 2);
    assert_eq!(modal.positive_commands(), [1.0, -0.5]);

    let zonal_probe = ZonalPushPullPlan::new([0.25_f32, 0.5]).unwrap();
    assert_eq!(zonal_probe.amplitudes(), [0.25, 0.5]);
    let hadamard_probe = HadamardPushPullPlan::new([0.25_f32, 0.5]).unwrap();
    assert_eq!(hadamard_probe.amplitudes(), [0.25, 0.5]);
    assert_eq!(hadamard_probe.order(), 4);

    let stability_criteria = InteractionMatrixStabilityCriteria::new(0.1_f32, 0.5, 0.75).unwrap();
    let stability = InteractionMatrixStabilityPlan::new(2, 3, stability_criteria).unwrap();
    assert_eq!(stability.measurement_count(), 2);
    assert_eq!(stability.corrector_coordinate_count(), 3);
    assert_eq!(stability.criteria(), stability_criteria);

    let differencing = MatchedCalibrationDifferencingPlan::<f32>::new(2, 3).unwrap();
    assert_eq!(differencing.measurement_count(), 2);
    assert_eq!(differencing.corrector_sample_count(), 3);

    let integrator = LeakyIntegratorPlan::new(0.25_f32, 0.75).unwrap();
    assert_eq!(integrator.gain(), 0.25);
    assert_eq!(integrator.pole(), 0.75);

    let covariance = DocrimeCovariancePlan::<f32>::with_sample_limit(2, 3, 7).unwrap();
    assert_eq!(covariance.measurement_count(), 2);
    assert_eq!(covariance.corrector_sample_count(), 3);
    assert_eq!(covariance.sample_limit(), Some(7));

    let conditioning = DocrimeConditioningCriteria::new(0.1_f32, 0.75, 0.01).unwrap();
    let open_loop = DocrimeOpenLoopInteractionMatrixPlan::new(2, 3, conditioning).unwrap();
    assert_eq!(open_loop.measurement_count(), 2);
    assert_eq!(open_loop.corrector_coordinate_count(), 3);
    assert_eq!(open_loop.criteria(), conditioning);
    let closed_loop = DocrimeClosedLoopInteractionMatrixPlan::new(2, 3, 0.5, conditioning).unwrap();
    assert_eq!(closed_loop.measurement_count(), 2);
    assert_eq!(closed_loop.corrector_coordinate_count(), 3);
    assert_eq!(closed_loop.loop_gain(), 0.5);
    assert_eq!(closed_loop.criteria(), conditioning);
}

#[test]
fn image_optical_gain_and_scc_plans_expose_complete_preparation() {
    let calibration = PixelCalibrationPlan::<f32, u16>::new([2.0, 3.0], [0.5, 1.0]).unwrap();
    assert_eq!(calibration.flat(), [2.0, 3.0]);
    assert_eq!(calibration.background(), [0.5, 1.0]);

    let selection =
        PixelSelectionPlan::<f32>::with_layout((2, 2), [true, false, false, true], 3, 1).unwrap();
    assert_eq!(selection.mask(), [true, false, false, true]);
    assert_eq!(selection.selected_indices(), [1, 5]);
    assert_eq!(selection.image_size(), (2, 2));
    assert_eq!(selection.row_stride(), 3);
    assert_eq!(selection.first_pixel(), 1);
    assert_eq!(selection.image_extent(), 6);

    let correction = OpticalGainCorrectionPlan::new([0.5_f32, 2.0]).unwrap();
    assert_eq!(correction.optical_gains(), [0.5, 2.0]);
    assert_eq!(correction.reciprocal_corrections(), [2.0, 0.5]);

    let schedule = Arc::new(
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f32, 4, 2)]).unwrap(),
    );
    let dither = OpticalGainDitherPlan::new(Arc::clone(&schedule));
    assert!(Arc::ptr_eq(&dither.shared_schedule(), &schedule));

    let update_criteria = OpticalGainUpdateCriteria::new(0.1_f32, 0.2, 0.5, 2.0, 1.0);
    let update_assignment = OpticalGainAssignment::DitherMode {
        dither_mode: 0,
        update_fraction: 0.25,
    };
    let update = OpticalGainUpdatePlan::new([update_criteria], [update_assignment]).unwrap();
    assert_eq!(update.criteria(), [update_criteria]);
    assert_eq!(update.assignments(), [update_assignment]);

    let chopped = ChoppedImageDifferencePlan::<f32>::new(4).unwrap();
    assert_eq!(chopped.pixel_count(), 4);
    let scc_response = SccPushPullResponsePlan::<f32>::new(4).unwrap();
    assert_eq!(scc_response.pixel_count(), 4);

    let shack_hartmann = ShackHartmannPlan::new(
        [-1.0_f32, 0.0, 1.0],
        [0.0_f32; 3],
        [0.25_f32, -0.5],
        [0.1_f32],
        [1.0_f32],
        [true],
    )
    .unwrap();
    assert_eq!(shack_hartmann.x_coordinates(), [-1.0, 0.0, 1.0]);
    assert_eq!(shack_hartmann.y_coordinates(), [0.0, 0.0, 0.0]);
    assert_eq!(shack_hartmann.reference_slopes(), [0.25, -0.5]);
    assert_eq!(shack_hartmann.pixel_thresholds(), [0.1]);
    assert_eq!(shack_hartmann.flux_thresholds(), [1.0]);
    assert_eq!(shack_hartmann.active(), [true]);

    let center_of_gravity = CenterOfGravityPlan::new(
        [-1.0_f32, 0.0, 1.0],
        [0.0_f32; 3],
        [0.25_f32, -0.5],
        None,
        1.0,
    )
    .unwrap();
    assert_eq!(center_of_gravity.x_coordinates(), [-1.0, 0.0, 1.0]);
    assert_eq!(center_of_gravity.y_coordinates(), [0.0, 0.0, 0.0]);
    assert_eq!(center_of_gravity.reference_centroid(), [0.25, -0.5]);
    assert_eq!(center_of_gravity.pixel_threshold(), None);
    assert_eq!(center_of_gravity.flux_threshold(), 1.0);
}

#[test]
fn deformable_mirror_limit_plans_expose_complete_preparation() {
    let power = PdmCommandPowerLimitPlan::new(2, 3.0_f32).unwrap();
    assert_eq!(power.len(), 2);
    assert_eq!(power.max_command_power(), 3.0);

    let edge = InterActuatorStrokeLimit::new(0, 1, 0.5_f32);
    let stroke = InterActuatorStrokeLimitPlan::new(2, Arc::from([edge])).unwrap();
    assert_eq!(stroke.len(), 2);
    assert_eq!(stroke.limits(), [edge]);
    assert_eq!(stroke.limits()[0].first(), 0);
    assert_eq!(stroke.limits()[0].second(), 1);
    assert_eq!(stroke.limits()[0].max_difference(), 0.5);
}

#[test]
fn non_dm_plans_expose_complete_prepared_artifacts() {
    let gemv = PreparedGemv::new(1, 2, Arc::from([2.0_f32, -1.0])).unwrap();
    assert_eq!(gemv.matrix(), [2.0, -1.0]);

    let closed_loop = ClosedLoopCorrectionPlan::with_hidden_modes(
        0.5_f32,
        0.75,
        1.0,
        2,
        1,
        0.25,
        Arc::from([1.0, -1.0]),
        Arc::from([0.5, -0.5]),
    )
    .unwrap();
    assert_eq!(closed_loop.state_length(), 2);
    assert_eq!(closed_loop.hidden_mode_count(), 1);
    assert_eq!(closed_loop.shape_to_hidden(), [1.0, -1.0]);
    assert_eq!(closed_loop.hidden_to_shape(), [0.5, -0.5]);

    let pseudo_open_loop = PseudoOpenLoopReconstructionPlan::new(
        1,
        2,
        1,
        Arc::from([1.0_f32, 2.0]),
        Arc::from([3.0]),
        Arc::from([4.0, 5.0]),
    )
    .unwrap();
    assert_eq!(pseudo_open_loop.interaction_matrix(), [1.0, 2.0]);
    assert_eq!(pseudo_open_loop.reconstructor(), [3.0]);
    assert_eq!(pseudo_open_loop.corrector_to_error(), [4.0, 5.0]);

    let zonal = ZonalInteractionMatrixPlan::new(2, [0.25_f32, 0.5]).unwrap();
    assert_eq!(zonal.reciprocal_scales(), [2.0, 1.0]);
    let hadamard = HadamardInteractionMatrixPlan::new(2, [0.25_f32, 0.5]).unwrap();
    assert_eq!(hadamard.order(), 4);
    assert_eq!(hadamard.reciprocal_scales(), [0.5, 0.25]);

    let excitation = DocrimeExcitationPlan::new([1.0_f32], 17).unwrap();
    let binary_excitation = DocrimeBinaryExcitationPlan::new([2.0_f32], 23).unwrap();
    assert_eq!(excitation.seed(), 17);
    assert_eq!(binary_excitation.seed(), 23);

    let regions =
        RegionExtractionPlan::<f32>::with_layout((2, 2), &[(0, 0)], (1, 1), 4, 3).unwrap();
    assert_eq!(regions.image_extent(), 9);

    let pyramid = PyramidPixelPlan::<f32>::new([true, false]).unwrap();
    assert_eq!(pyramid.mean_pupil_intensity_scale(), 0.125);

    let schedule = Arc::new(
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(1.0_f32, 4, 2)]).unwrap(),
    );
    assert_eq!(schedule.cosines(0).unwrap().len(), 4);
    assert_eq!(schedule.sines(0).unwrap().len(), 4);
    assert_eq!(schedule.cosines(1), None);
    let estimator = OpticalGainEstimatorPlan::new(schedule, [1.0]).unwrap();
    assert_eq!(estimator.two_over_window(), 0.25);
}

#[test]
fn dm_plans_expose_coordinate_preserving_prepared_artifacts() {
    let controller =
        ControllerProjection::new(1, Arc::from([1.0_f32, 0.5]), Arc::from([0.25, 2.0])).unwrap();
    assert_eq!(controller.command_size(), 1);
    assert_eq!(controller.vdm_size(), 2);
    assert_eq!(controller.controller_to_vdm(), [1.0, 0.5]);
    assert_eq!(controller.vdm_to_controller(), [0.25, 2.0]);
    let controllers: Arc<[ControllerProjection<f32>]> = Arc::from([controller]);
    let forward = ControllerToVdmPlan::new(2, Arc::clone(&controllers)).unwrap();
    let feedback = VdmFeedbackToControllerPlan::new(2, controllers).unwrap();
    assert_eq!(forward.vdm_size(), 2);
    assert_eq!(forward.controller_to_vdm_projections().len(), 1);
    assert_eq!(feedback.vdm_size(), 2);
    assert_eq!(feedback.vdm_to_controller_projections().len(), 1);

    let vdm_to_pdm_matrix: Arc<[f32]> = Arc::from([1.0, 0.0, 0.25, 1.0]);
    let pdm_to_vdm_matrix: Arc<[f32]> = Arc::from([1.0, 0.5, 0.0, 1.0]);
    let projection = PdmProjection::new(
        2,
        Arc::clone(&vdm_to_pdm_matrix),
        Arc::clone(&pdm_to_vdm_matrix),
    )
    .unwrap();
    assert_eq!(projection.full_vdm_size(), 2);
    assert_eq!(
        materialize_pdm_projection(projection.vdm_to_pdm_artifacts(), 2, 2),
        vdm_to_pdm_matrix.as_ref()
    );
    assert_eq!(
        materialize_pdm_projection(projection.pdm_to_vdm_artifacts(), 2, 2),
        pdm_to_vdm_matrix.as_ref()
    );
    let projections: Arc<[PdmProjection<f32>]> = Arc::from([projection]);

    let active_to_full: Arc<[f32]> = Arc::from([1.0, 0.5]);
    let full_to_active: Arc<[f32]> = Arc::from([1.0, 0.0]);
    let vdm_to_pdm =
        VdmToPdmPlan::new(1, 2, Arc::clone(&active_to_full), Arc::clone(&projections)).unwrap();
    let pdm_feedback_to_vdm =
        PdmFeedbackToVdmPlan::new(1, 2, Arc::clone(&full_to_active), Arc::clone(&projections))
            .unwrap();
    assert_eq!(vdm_to_pdm.active_vdm_size(), 1);
    assert_eq!(vdm_to_pdm.full_vdm_size(), 2);
    assert_eq!(vdm_to_pdm.vdm_to_pdm_projections().len(), 1);
    assert_eq!(
        materialize_extrapolation(vdm_to_pdm.extrapolation_artifacts(), 2, 1),
        active_to_full.as_ref()
    );
    assert_eq!(pdm_feedback_to_vdm.active_vdm_size(), 1);
    assert_eq!(pdm_feedback_to_vdm.full_vdm_size(), 2);
    assert_eq!(pdm_feedback_to_vdm.pdm_to_vdm_projections().len(), 1);
    assert_eq!(
        materialize_truncation(pdm_feedback_to_vdm.truncation_artifacts(), 1, 2),
        full_to_active.as_ref()
    );

    let stroke = PdmStrokeOptimizationPlan::new(
        2,
        1,
        Arc::from([0.1_f32, -0.1]),
        Arc::from([0.5, 0.5]),
        Arc::from([1.0, 1.0]),
    )
    .unwrap();
    assert_eq!(stroke.preferred_command(), [0.1, -0.1]);
    assert_eq!(stroke.command_to_null_modes(), [0.5, 0.5]);
    assert_eq!(stroke.null_modes_to_command(), [1.0, 1.0]);

    let pdm = PdmCommandPlan::new(
        Arc::from([-1.0_f32, -2.0]),
        Arc::from([1.0_f32, 2.0]),
        Some(Arc::from([0.1_f32, 0.2])),
    )
    .unwrap()
    .with_command_power_limit(PdmCommandPowerLimitPlan::new(2, 3.0).unwrap())
    .unwrap()
    .with_inter_actuator_stroke_limit(
        InterActuatorStrokeLimitPlan::new(2, Arc::from([InterActuatorStrokeLimit::new(0, 1, 0.5)]))
            .unwrap(),
    )
    .unwrap()
    .with_stroke_optimization(stroke)
    .unwrap();
    assert_eq!(pdm.lower_limits(), [-1.0, -2.0]);
    assert_eq!(pdm.upper_limits(), [1.0, 2.0]);
    assert_eq!(pdm.slew_limits(), Some([0.1, 0.2].as_slice()));
    assert_eq!(pdm.command_power_limit().unwrap().max_command_power(), 3.0);
    assert_eq!(
        pdm.inter_actuator_stroke_limit().unwrap().limits()[0],
        InterActuatorStrokeLimit::new(0, 1, 0.5)
    );
    assert_eq!(
        pdm.stroke_optimization().unwrap().preferred_command(),
        [0.1, -0.1]
    );

    let composite =
        DeformableMirrorCommandPlan::from_plans(vdm_to_pdm, pdm_feedback_to_vdm, pdm).unwrap();
    assert_eq!(composite.active_vdm_size(), 1);
    assert_eq!(composite.pdm_size(), 2);
    assert_eq!(composite.vdm_to_pdm().full_vdm_size(), 2);
    assert_eq!(composite.pdm_feedback_to_vdm().full_vdm_size(), 2);
    assert_eq!(
        composite.pdm_command().slew_limits(),
        Some([0.1, 0.2].as_slice())
    );
}
