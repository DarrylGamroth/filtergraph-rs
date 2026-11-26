//! Scientific schema identity checks.

use filtergraph_algorithms::schemas::{
    ACTIVE_TO_FULL_VDM_PROJECTION_V1, BACKGROUND_CALIBRATION_V1, CALIBRATED_PIXEL_VALIDITY_V1,
    CALIBRATED_PIXELS_V1, CALIBRATION_CYCLE_COMPLETE_V1, CALIBRATION_PATTERN_INDEX_V1,
    CALIBRATION_PROBE_AMPLITUDES_V1, CALIBRATION_PROBE_POLARITY_V1,
    CENTER_OF_GRAVITY_COORDINATES_V1, CENTER_OF_GRAVITY_FLUX_V1, CENTER_OF_GRAVITY_OFFSET_V1,
    CENTER_OF_GRAVITY_PIXELS_V1, CENTER_OF_GRAVITY_VALIDITY_V1, CONTROLLER_COMMAND_V1,
    CONTROLLER_CONSTRAINT_FEEDBACK_V1, CONTROLLER_RESIDUAL_ERROR_V1, CONTROLLER_STATE_V1,
    CONTROLLER_TO_VDM_PROJECTION_V1, CORRECTOR_COORDINATE_V1, DEMANDED_PDM_COMMAND_V1,
    DOCRIME_CORRECTOR_VARIANCE_RANGE_V1, DOCRIME_COVARIANCE_COMPLETE_V1,
    DOCRIME_COVARIANCE_SAMPLE_COUNT_V1, DOCRIME_EXCITATION_AMPLITUDE_V1, DOCRIME_EXCITATION_V1,
    DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1, DOCRIME_INTERACTION_MATRIX_STATUS_V1,
    DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1,
    DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1, FLAT_CALIBRATION_V1,
    FULL_TO_ACTIVE_VDM_PROJECTION_V1, HIDDEN_TO_SHAPE_PROJECTION_V1,
    INTER_ACTUATOR_STROKE_LIMIT_V1, INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1,
    INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1,
    INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1, INTERACTION_MATRIX_STABILITY_STATUS_V1,
    MATCHED_CALIBRATION_FILTERED_V1, MODAL_POSITIVE_COMMANDS_V1,
    OPTICAL_GAIN_ACCEPTED_DITHER_MODES_V1, OPTICAL_GAIN_CALIBRATION_RESPONSE_V1,
    OPTICAL_GAIN_COMMAND_AMPLITUDE_V1, OPTICAL_GAIN_DITHER_COMMAND_V1,
    OPTICAL_GAIN_DITHER_ESTIMATE_V1, OPTICAL_GAIN_DITHER_PHASE_INDICES_V1,
    OPTICAL_GAIN_DITHER_RESPONSE_V1, OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1,
    OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1, OPTICAL_GAIN_ESTIMATE_COMPLETE_V1,
    OPTICAL_GAIN_ESTIMATE_VALIDITY_V1, OPTICAL_GAIN_PHASE_LAG_V1,
    OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1, OPTICAL_GAIN_UPDATED_RECONSTRUCTED_MODES_V1,
    PDM_COMMAND_TO_NULL_MODES_V1, PDM_CONSTRAINT_FEEDBACK_V1, PDM_NULL_MODES_TO_COMMAND_V1,
    PDM_TO_VDM_PROJECTION_V1, PHYSICAL_OPTICAL_GAIN_V1, PIXEL_SELECTION_MASK_V1,
    PREFERRED_PDM_COMMAND_V1, PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1,
    PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1, PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1,
    PWFS_RECONSTRUCTOR_V1, PYRAMID_MEAN_PUPIL_INTENSITY_V1, PYRAMID_PUPIL_MASK_V1,
    PYRAMID_RECONSTRUCTION_PIXELS_V1, RAW_DETECTOR_PIXELS_V1, RECONSTRUCTED_MODES_V1,
    REGION_ORIGINS_V1, REGION_PIXELS_V1, REQUESTED_PDM_COMMAND_V1,
    SCC_CHOPPED_DIFFERENCE_STATUS_V1, SCC_CHOPPED_DIFFERENCE_VALIDITY_V1,
    SCC_CHOPPED_IMAGE_DIFFERENCE_V1, SCC_PUSH_PULL_RESPONSE_V1, SHACK_HARTMANN_ACTIVE_V1,
    SHACK_HARTMANN_COORDINATES_V1, SHACK_HARTMANN_FLUX_V1, SHACK_HARTMANN_REFERENCE_SLOPES_V1,
    SHACK_HARTMANN_SLOPES_V1, SHACK_HARTMANN_THRESHOLDS_V1, SHACK_HARTMANN_VALIDITY_V1,
    SHAPE_TO_HIDDEN_PROJECTION_V1, SHWFS_RECONSTRUCTOR_V1, VDM_COMMAND_V1,
    VDM_CONSTRAINT_FEEDBACK_V1, VDM_TO_CONTROLLER_PROJECTION_V1, VDM_TO_PDM_PROJECTION_V1,
};

#[test]
fn scientific_schema_identifiers_are_exact_and_versioned() {
    assert_eq!(
        RAW_DETECTOR_PIXELS_V1,
        "pipewireao/raw-detector-pixels-u16/1"
    );
    assert_eq!(CALIBRATED_PIXELS_V1, "pipewireao/calibrated-pixels-f32/1");
    assert_eq!(
        CALIBRATED_PIXEL_VALIDITY_V1,
        "pipewireao/calibrated-pixel-validity-bool/1"
    );
    assert_eq!(FLAT_CALIBRATION_V1, "pipewireao/flat-calibration-f32/1");
    assert_eq!(
        BACKGROUND_CALIBRATION_V1,
        "pipewireao/background-calibration-f32/1"
    );
    assert_eq!(REGION_PIXELS_V1, "pipewireao/region-pixels-f32/1");
    assert_eq!(REGION_ORIGINS_V1, "pipewireao/region-origins-u32/1");
    assert_eq!(
        PIXEL_SELECTION_MASK_V1,
        "pipewireao/pixel-selection-mask-bool/1"
    );
    assert_eq!(
        CENTER_OF_GRAVITY_PIXELS_V1,
        "pipewireao/center-of-gravity-pixels-f32/1"
    );
    assert_eq!(
        CENTER_OF_GRAVITY_COORDINATES_V1,
        "pipewireao/center-of-gravity-coordinates-f32/1"
    );
    assert_eq!(
        CENTER_OF_GRAVITY_OFFSET_V1,
        "pipewireao/center-of-gravity-offset-f32/1"
    );
    assert_eq!(
        CENTER_OF_GRAVITY_FLUX_V1,
        "pipewireao/center-of-gravity-flux-f32/1"
    );
    assert_eq!(
        CENTER_OF_GRAVITY_VALIDITY_V1,
        "pipewireao/center-of-gravity-validity-bool/1"
    );
    assert_eq!(
        SCC_PUSH_PULL_RESPONSE_V1,
        "pipewireao/scc-push-pull-response-f32/1"
    );
    assert_eq!(
        SCC_CHOPPED_IMAGE_DIFFERENCE_V1,
        "pipewireao/scc-chopped-image-difference-f32/1"
    );
    assert_eq!(
        SCC_CHOPPED_DIFFERENCE_VALIDITY_V1,
        "pipewireao/scc-chopped-difference-validity-bool/1"
    );
    assert_eq!(
        SCC_CHOPPED_DIFFERENCE_STATUS_V1,
        "pipewireao/scc-chopped-difference-status-u8/1"
    );
    assert_eq!(
        PYRAMID_RECONSTRUCTION_PIXELS_V1,
        "pipewireao/pyramid-reconstruction-pixels-f32/1"
    );
    assert_eq!(
        PYRAMID_MEAN_PUPIL_INTENSITY_V1,
        "pipewireao/pyramid-mean-pupil-intensity-f32/1"
    );
    assert_eq!(
        PYRAMID_PUPIL_MASK_V1,
        "pipewireao/pyramid-pupil-mask-bool/1"
    );
    assert_eq!(PWFS_RECONSTRUCTOR_V1, "pipewireao/pwfs-reconstructor-f32/1");
    assert_eq!(
        SHACK_HARTMANN_SLOPES_V1,
        "pipewireao/shack-hartmann-slopes-f32/1"
    );
    assert_eq!(
        SHACK_HARTMANN_FLUX_V1,
        "pipewireao/shack-hartmann-flux-f32/1"
    );
    assert_eq!(
        SHACK_HARTMANN_VALIDITY_V1,
        "pipewireao/shack-hartmann-validity-bool/1"
    );
    assert_eq!(
        SHACK_HARTMANN_COORDINATES_V1,
        "pipewireao/shack-hartmann-coordinates-f32/1"
    );
    assert_eq!(
        SHACK_HARTMANN_REFERENCE_SLOPES_V1,
        "pipewireao/shack-hartmann-reference-slopes-f32/1"
    );
    assert_eq!(
        SHACK_HARTMANN_THRESHOLDS_V1,
        "pipewireao/shack-hartmann-thresholds-f32/1"
    );
    assert_eq!(
        SHACK_HARTMANN_ACTIVE_V1,
        "pipewireao/shack-hartmann-active-bool/1"
    );
    assert_eq!(
        SHWFS_RECONSTRUCTOR_V1,
        "pipewireao/shwfs-reconstructor-f32/1"
    );
    assert_eq!(
        OPTICAL_GAIN_DITHER_COMMAND_V1,
        "pipewireao/optical-gain-dither-command-f32/1"
    );
    assert_eq!(
        OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1,
        "pipewireao/optical-gain-dither-window-frame-u64/1"
    );
    assert_eq!(
        OPTICAL_GAIN_DITHER_PHASE_INDICES_V1,
        "pipewireao/optical-gain-dither-phase-indices-u64/1"
    );
    assert_eq!(
        OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1,
        "pipewireao/optical-gain-dither-window-complete-bool/1"
    );
    assert_eq!(
        OPTICAL_GAIN_DITHER_RESPONSE_V1,
        "pipewireao/optical-gain-dither-response-f32/1"
    );
    assert_eq!(
        OPTICAL_GAIN_CALIBRATION_RESPONSE_V1,
        "pipewireao/optical-gain-calibration-response-f32/1"
    );
    assert_eq!(
        OPTICAL_GAIN_COMMAND_AMPLITUDE_V1,
        "pipewireao/optical-gain-command-amplitude-f32/1"
    );
    assert_eq!(
        OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1,
        "pipewireao/optical-gain-response-amplitude-f32/1"
    );
    assert_eq!(
        OPTICAL_GAIN_DITHER_ESTIMATE_V1,
        "pipewireao/optical-gain-dither-estimate-f32/1"
    );
    assert_eq!(
        OPTICAL_GAIN_PHASE_LAG_V1,
        "pipewireao/optical-gain-phase-lag-f32/1"
    );
    assert_eq!(
        OPTICAL_GAIN_ESTIMATE_VALIDITY_V1,
        "pipewireao/optical-gain-estimate-validity-bool/1"
    );
    assert_eq!(
        OPTICAL_GAIN_ESTIMATE_COMPLETE_V1,
        "pipewireao/optical-gain-estimate-complete-bool/1"
    );
    assert_eq!(
        OPTICAL_GAIN_ACCEPTED_DITHER_MODES_V1,
        "pipewireao/optical-gain-accepted-dither-modes-bool/1"
    );
    assert_eq!(
        OPTICAL_GAIN_UPDATED_RECONSTRUCTED_MODES_V1,
        "pipewireao/optical-gain-updated-reconstructed-modes-bool/1"
    );
    assert_eq!(
        DEMANDED_PDM_COMMAND_V1,
        "pipewireao/dm-command-opd-um-f32/1"
    );
    assert_eq!(
        PDM_CONSTRAINT_FEEDBACK_V1,
        "pipewireao/pdm-constraint-feedback-f32/1"
    );
    assert_eq!(
        REQUESTED_PDM_COMMAND_V1,
        "pipewireao/requested-pdm-command-f32/1"
    );
    assert_eq!(
        INTER_ACTUATOR_STROKE_LIMIT_V1,
        "pipewireao/inter-actuator-stroke-limit-f32/1"
    );
    assert_eq!(
        PREFERRED_PDM_COMMAND_V1,
        "pipewireao/preferred-pdm-command-f32/1"
    );
    assert_eq!(
        PDM_COMMAND_TO_NULL_MODES_V1,
        "pipewireao/pdm-command-to-null-modes-f32/1"
    );
    assert_eq!(
        PDM_NULL_MODES_TO_COMMAND_V1,
        "pipewireao/pdm-null-modes-to-command-f32/1"
    );
    assert_eq!(
        CONTROLLER_RESIDUAL_ERROR_V1,
        "pipewireao/controller-residual-error-f32/1"
    );
    assert_eq!(
        PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1,
        "pipewireao/pseudo-open-loop-interaction-matrix-f32/1"
    );
    assert_eq!(
        PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1,
        "pipewireao/pseudo-open-loop-reconstructor-f32/1"
    );
    assert_eq!(
        PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1,
        "pipewireao/pseudo-open-loop-corrector-to-error-f32/1"
    );
    assert_eq!(
        INTERACTION_MATRIX_STABILITY_STATUS_V1,
        "pipewireao/interaction-matrix-stability-status-u8/1"
    );
    assert_eq!(
        CORRECTOR_COORDINATE_V1,
        "pipewireao/corrector-coordinate-u64/1"
    );
    assert_eq!(
        INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1,
        "pipewireao/interaction-matrix-stability-column-norms-f32/1"
    );
    assert_eq!(
        INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1,
        "pipewireao/interaction-matrix-stability-relative-column-difference-f32/1"
    );
    assert_eq!(
        INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1,
        "pipewireao/interaction-matrix-stability-relative-frobenius-difference-f32/1"
    );
    assert_eq!(
        MATCHED_CALIBRATION_FILTERED_V1,
        "pipewireao/matched-calibration-filtered-bool/1"
    );
    assert_eq!(
        CALIBRATION_PATTERN_INDEX_V1,
        "pipewireao/calibration-pattern-index-u64/1"
    );
    assert_eq!(
        CALIBRATION_PROBE_POLARITY_V1,
        "pipewireao/calibration-probe-polarity-u8/1"
    );
    assert_eq!(
        CALIBRATION_CYCLE_COMPLETE_V1,
        "pipewireao/calibration-cycle-complete-bool/1"
    );
    assert_eq!(
        CALIBRATION_PROBE_AMPLITUDES_V1,
        "pipewireao/calibration-probe-amplitudes-f32/1"
    );
    assert_eq!(
        MODAL_POSITIVE_COMMANDS_V1,
        "pipewireao/modal-positive-commands-f32/1"
    );
    assert_eq!(CONTROLLER_COMMAND_V1, "pipewireao/controller-command-f32/1");
    assert_eq!(CONTROLLER_STATE_V1, "pipewireao/controller-state-f32/1");
    assert_eq!(
        CONTROLLER_CONSTRAINT_FEEDBACK_V1,
        "pipewireao/controller-constraint-feedback-f32/1"
    );
    assert_eq!(VDM_COMMAND_V1, "pipewireao/vdm-command-f32/1");
    assert_eq!(
        VDM_CONSTRAINT_FEEDBACK_V1,
        "pipewireao/vdm-constraint-feedback-f32/1"
    );
    assert_eq!(
        CONTROLLER_TO_VDM_PROJECTION_V1,
        "pipewireao/controller-to-vdm-projection-f32/1"
    );
    assert_eq!(
        VDM_TO_CONTROLLER_PROJECTION_V1,
        "pipewireao/vdm-to-controller-projection-f32/1"
    );
    assert_eq!(
        ACTIVE_TO_FULL_VDM_PROJECTION_V1,
        "pipewireao/active-to-full-vdm-projection-f32/1"
    );
    assert_eq!(
        FULL_TO_ACTIVE_VDM_PROJECTION_V1,
        "pipewireao/full-to-active-vdm-projection-f32/1"
    );
    assert_eq!(
        VDM_TO_PDM_PROJECTION_V1,
        "pipewireao/vdm-to-pdm-projection-f32/1"
    );
    assert_eq!(
        PDM_TO_VDM_PROJECTION_V1,
        "pipewireao/pdm-to-vdm-projection-f32/1"
    );
    assert_eq!(
        SHAPE_TO_HIDDEN_PROJECTION_V1,
        "pipewireao/shape-to-hidden-projection-f32/1"
    );
    assert_eq!(
        HIDDEN_TO_SHAPE_PROJECTION_V1,
        "pipewireao/hidden-to-shape-projection-f32/1"
    );
    assert_eq!(
        RECONSTRUCTED_MODES_V1,
        "pipewireao/reconstructed-modes-f32/1"
    );
    assert_eq!(
        PHYSICAL_OPTICAL_GAIN_V1,
        "pipewireao/physical-optical-gain-f32/1"
    );
    assert_eq!(DOCRIME_EXCITATION_V1, "pipewireao/docrime-excitation-f32/1");
    assert_eq!(
        DOCRIME_EXCITATION_AMPLITUDE_V1,
        "pipewireao/docrime-excitation-amplitude-f32/1"
    );
    assert_eq!(
        DOCRIME_COVARIANCE_SAMPLE_COUNT_V1,
        "pipewireao/docrime-covariance-sample-count-u64/1"
    );
    assert_eq!(
        DOCRIME_COVARIANCE_COMPLETE_V1,
        "pipewireao/docrime-covariance-complete-bool/1"
    );
    assert_eq!(
        DOCRIME_INTERACTION_MATRIX_STATUS_V1,
        "pipewireao/docrime-interaction-matrix-status-u8/1"
    );
    assert_eq!(
        DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1,
        "pipewireao/docrime-interaction-matrix-status-coordinates-u64/1"
    );
    assert_eq!(
        DOCRIME_CORRECTOR_VARIANCE_RANGE_V1,
        "pipewireao/docrime-corrector-variance-range-f32/1"
    );
    assert_eq!(
        DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1,
        "pipewireao/docrime-maximum-absolute-corrector-correlation-f32/1"
    );
    assert_eq!(
        DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1,
        "pipewireao/docrime-minimum-factorization-pivot-ratio-f32/1"
    );
}
