//! Versioned scientific schema identifiers owned by FilterGraph.
//!
//! These identifiers name scientific meaning. Transport implementations own
//! the structural element type, shape, layout, and rate fields that carry a
//! schema.

/// F32 calibrated detector pixels in PDE, in detector row-major order.
pub const CALIBRATED_PIXELS_V1: &str = "pipewireao/calibrated-pixels-f32/1";

/// One complete immutable contiguous group of F32 calibrated detector rows.
pub const CALIBRATED_PIXEL_ROW_BLOCK_V1: &str = "pipewireao/calibrated-pixel-row-block-f32/1";

/// One complete immutable F32 detector readout block in
/// `[readout_region, line, sample]` acquisition order.
pub const CALIBRATED_PIXEL_READOUT_BLOCK_V1: &str =
    "pipewireao/calibrated-pixel-readout-block-f32/1";

/// U16 raw detector pixels in ADU, in detector row-major order.
pub const RAW_DETECTOR_PIXELS_V1: &str = "pipewireao/raw-detector-pixels-u16/1";

/// One complete immutable contiguous group of U16 detector rows in ADU.
pub const RAW_PIXEL_ROW_BLOCK_V1: &str = "pipewireao/raw-pixel-row-block-u16/1";

/// One complete immutable U16 detector readout block in
/// `[readout_region, line, sample]` acquisition order.
pub const RAW_PIXEL_READOUT_BLOCK_V1: &str = "pipewireao/raw-pixel-readout-block-u16/1";

/// BOOL8 upstream validity in calibrated detector-pixel order.
pub const CALIBRATED_PIXEL_VALIDITY_V1: &str = "pipewireao/calibrated-pixel-validity-bool/1";

/// F32 detector flat-calibration coefficients in PDE/ADU, in detector
/// row-major order.
pub const FLAT_CALIBRATION_V1: &str = "pipewireao/flat-calibration-f32/1";

/// F32 detector background calibration in ADU, in detector row-major order.
pub const BACKGROUND_CALIBRATION_V1: &str = "pipewireao/background-calibration-f32/1";

/// BOOL8 logical detector-pixel selection mask in detector row-major order.
pub const PIXEL_SELECTION_MASK_V1: &str = "pipewireao/pixel-selection-mask-bool/1";

/// F32 Thompson four-image SCC double-difference response in calibrated-pixel
/// order.
pub const SCC_PUSH_PULL_RESPONSE_V1: &str = "pipewireao/scc-push-pull-response-f32/1";

/// F32 rolling fringed-minus-unfringed SCC chopped-image difference.
pub const SCC_CHOPPED_IMAGE_DIFFERENCE_V1: &str = "pipewireao/scc-chopped-image-difference-f32/1";

/// BOOL8 per-pixel validity of the rolling SCC chopped-image difference.
pub const SCC_CHOPPED_DIFFERENCE_VALIDITY_V1: &str =
    "pipewireao/scc-chopped-difference-validity-bool/1";

/// U8 chopped-image pair status: priming `0` or complete `1`.
pub const SCC_CHOPPED_DIFFERENCE_STATUS_V1: &str = "pipewireao/scc-chopped-difference-status-u8/1";

/// F32 calibrated pixels grouped in region-major order as
/// `[region, row-major pixel]`.
pub const REGION_PIXELS_V1: &str = "pipewireao/region-pixels-f32/1";

/// Zero-based `(row, column)` origins for region extraction.
pub const REGION_ORIGINS_V1: &str = "pipewireao/region-origins-u32/1";

/// F32 calibrated detector pixels in one prepared center-of-gravity region,
/// in the order of the corresponding coordinate pairs.
pub const CENTER_OF_GRAVITY_PIXELS_V1: &str = "pipewireao/center-of-gravity-pixels-f32/1";

/// F32 pair-interleaved detector-pixel X/Y coordinates in selected-region
/// pixel order.
pub const CENTER_OF_GRAVITY_COORDINATES_V1: &str = "pipewireao/center-of-gravity-coordinates-f32/1";

/// F32 measured-minus-reference center-of-gravity offset in detector-pixel
/// X/Y order.
pub const CENTER_OF_GRAVITY_OFFSET_V1: &str = "pipewireao/center-of-gravity-offset-f32/1";

/// F32 selected-region center-of-gravity flux in PDE.
pub const CENTER_OF_GRAVITY_FLUX_V1: &str = "pipewireao/center-of-gravity-flux-f32/1";

/// BOOL8 numerical and flux-threshold validity for one center-of-gravity
/// measurement.
pub const CENTER_OF_GRAVITY_VALIDITY_V1: &str = "pipewireao/center-of-gravity-validity-bool/1";

/// F32 normalized pyramid reconstruction pixels in pupil-major order. Within
/// each pupil, pixels retain the row-major order selected by the shared mask.
pub const PYRAMID_RECONSTRUCTION_PIXELS_V1: &str = "pipewireao/pyramid-reconstruction-pixels-f32/1";

/// F32 current-frame mean intensity over all pixels in all four pyramid pupil
/// images, in PDE.
pub const PYRAMID_MEAN_PUPIL_INTENSITY_V1: &str = "pipewireao/pyramid-mean-pupil-intensity-f32/1";

/// BOOL8 shared row-major pyramid pupil mask. Selected entries contribute one
/// reconstruction pixel from each of the four pupil images.
pub const PYRAMID_PUPIL_MASK_V1: &str = "pipewireao/pyramid-pupil-mask-bool/1";

/// F32 row-major matrix mapping pupil-major pyramid reconstruction pixels to
/// a reconstructed controller vector. Shape is
/// `[reconstructed-coordinate, pupil-reconstruction-pixel]`.
pub const PWFS_RECONSTRUCTOR_V1: &str = "pipewireao/pwfs-reconstructor-f32/1";

/// F32 Shack-Hartmann slopes in `[subaperture, x-y]` pair-interleaved order.
pub const SHACK_HARTMANN_SLOPES_V1: &str = "pipewireao/shack-hartmann-slopes-f32/1";

/// F32 Shack-Hartmann fluxes in subaperture order.
pub const SHACK_HARTMANN_FLUX_V1: &str = "pipewireao/shack-hartmann-flux-f32/1";

/// BOOL8 Shack-Hartmann measurement validity in subaperture order.
pub const SHACK_HARTMANN_VALIDITY_V1: &str = "pipewireao/shack-hartmann-validity-bool/1";

/// Pair-interleaved detector-plane X/Y coordinates for one subaperture.
pub const SHACK_HARTMANN_COORDINATES_V1: &str = "pipewireao/shack-hartmann-coordinates-f32/1";

/// Pair-interleaved Shack-Hartmann reference slopes.
pub const SHACK_HARTMANN_REFERENCE_SLOPES_V1: &str =
    "pipewireao/shack-hartmann-reference-slopes-f32/1";

/// Pair-interleaved pixel and flux thresholds per subaperture.
pub const SHACK_HARTMANN_THRESHOLDS_V1: &str = "pipewireao/shack-hartmann-thresholds-f32/1";

/// Boolean active mask in subaperture order.
pub const SHACK_HARTMANN_ACTIVE_V1: &str = "pipewireao/shack-hartmann-active-bool/1";

/// F32 row-major matrix mapping pair-interleaved Shack-Hartmann slopes to a
/// controller vector. Shape is `[controller-coordinate, slope]`.
pub const SHWFS_RECONSTRUCTOR_V1: &str = "pipewireao/shwfs-reconstructor-f32/1";

/// F32 demanded physical deformable-mirror commands in actuator order, in µm wavefront OPD.
pub const DEMANDED_PDM_COMMAND_V1: &str = "pipewireao/dm-command-opd-um-f32/1";

/// F32 physical deformable-mirror commands in actuator order, in metres wavefront OPD.
pub const ACTUATOR_COMMAND_OPD_M_V1: &str = "pipewireao/actuator-command-opd-m-f32/1";

/// F32 native device command values in physical-actuator order with no displacement unit.
pub const NATIVE_PDM_COMMAND_V1: &str = "pipewireao/demanded-pdm-command-f32/1";

/// F32 requested-minus-demanded PDM constraint feedback in actuator order.
pub const PDM_CONSTRAINT_FEEDBACK_V1: &str = "pipewireao/pdm-constraint-feedback-f32/1";

/// F32 requested physical deformable-mirror commands in actuator order.
pub const REQUESTED_PDM_COMMAND_V1: &str = "pipewireao/requested-pdm-command-f32/1";

/// F32 physical deformable-mirror system flat in actuator order.
pub const PDM_SYSTEM_FLAT_V1: &str = "pipewireao/pdm-system-flat-f32/1";

/// F32 physical-DM engineering offset in actuator order, in µm wavefront.
pub const PDM_ENGINEERING_OFFSET_V1: &str = "pipewireao/dm-offset-opd-um-f32/1";

/// F32 maximum absolute command difference per prepared physical-actuator
/// adjacency edge.
pub const INTER_ACTUATOR_STROKE_LIMIT_V1: &str = "pipewireao/inter-actuator-stroke-limit-f32/1";

/// F32 preferred physical deformable-mirror command for calibrated stroke
/// optimization.
pub const PREFERRED_PDM_COMMAND_V1: &str = "pipewireao/preferred-pdm-command-f32/1";

/// F32 row-major map from physical-actuator commands to calibrated null-mode
/// coefficients. Shape is `[null-mode, physical-actuator]`.
pub const PDM_COMMAND_TO_NULL_MODES_V1: &str = "pipewireao/pdm-command-to-null-modes-f32/1";

/// F32 row-major map from calibrated null-mode coefficients to physical
/// actuator commands. Shape is `[physical-actuator, null-mode]`.
pub const PDM_NULL_MODES_TO_COMMAND_V1: &str = "pipewireao/pdm-null-modes-to-command-f32/1";

/// F32 reconstructed residual error in controller-coordinate order.
pub const CONTROLLER_RESIDUAL_ERROR_V1: &str = "pipewireao/controller-residual-error-f32/1";

/// F32 row-major corrector-to-measurement map used by pseudo-open-loop
/// reconstruction. Shape is `[measurement, corrector-coordinate]`.
pub const PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1: &str =
    "pipewireao/pseudo-open-loop-interaction-matrix-f32/1";

/// F32 row-major measurement-to-residual-error map used by pseudo-open-loop
/// reconstruction. Shape is `[residual-error, measurement]`.
pub const PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1: &str =
    "pipewireao/pseudo-open-loop-reconstructor-f32/1";

/// F32 row-major corrector-to-residual-error map used by pseudo-open-loop
/// reconstruction. Shape is `[residual-error, corrector-coordinate]`.
pub const PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1: &str =
    "pipewireao/pseudo-open-loop-corrector-to-error-f32/1";

/// U8 interaction-matrix stability classification: valid `0`, insufficient
/// column norm `1`, excessive relative column difference `2`, or excessive
/// relative Frobenius difference `3`.
pub const INTERACTION_MATRIX_STABILITY_STATUS_V1: &str =
    "pipewireao/interaction-matrix-stability-status-u8/1";

/// U64 zero-based corrector-coordinate index. A status coordinate is zero
/// when the corresponding classification does not identify one coordinate.
pub const CORRECTOR_COORDINATE_V1: &str = "pipewireao/corrector-coordinate-u64/1";

/// F32 minimum interaction-matrix column norms in first-window,
/// second-window order.
pub const INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1: &str =
    "pipewireao/interaction-matrix-stability-column-norms-f32/1";

/// F32 maximum symmetric relative interaction-matrix column difference.
pub const INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1: &str =
    "pipewireao/interaction-matrix-stability-relative-column-difference-f32/1";

/// F32 symmetric relative interaction-matrix Frobenius difference.
pub const INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1: &str =
    "pipewireao/interaction-matrix-stability-relative-frobenius-difference-f32/1";

/// BOOL8 validity of matched first-difference outputs. False identifies the
/// zero-valued priming output; true identifies two valid differenced vectors.
pub const MATCHED_CALIBRATION_FILTERED_V1: &str = "pipewireao/matched-calibration-filtered-bool/1";

/// U64 zero-based modal row, zonal coordinate, or Hadamard row of one
/// structured calibration command.
pub const CALIBRATION_PATTERN_INDEX_V1: &str = "pipewireao/calibration-pattern-index-u64/1";

/// U8 global calibration-command polarity: positive `0` or negative `1`.
pub const CALIBRATION_PROBE_POLARITY_V1: &str = "pipewireao/calibration-probe-polarity-u8/1";

/// BOOL8 flag identifying the final command in one prepared calibration cycle.
pub const CALIBRATION_CYCLE_COMPLETE_V1: &str = "pipewireao/calibration-cycle-complete-bool/1";

/// F32 positive probe amplitude per corrector coordinate.
pub const CALIBRATION_PROBE_AMPLITUDES_V1: &str = "pipewireao/calibration-probe-amplitudes-f32/1";

/// F32 exact positive modal commands in row-major `[mode,
/// corrector-coordinate]` order.
pub const MODAL_POSITIVE_COMMANDS_V1: &str = "pipewireao/modal-positive-commands-f32/1";

/// F32 correction command in controller-coordinate order.
pub const CONTROLLER_COMMAND_V1: &str = "pipewireao/controller-command-f32/1";

/// F32 committed closed-loop state in controller-coordinate order.
pub const CONTROLLER_STATE_V1: &str = "pipewireao/controller-state-f32/1";

/// F32 constraint feedback in the concatenated controller coordinate order.
pub const CONTROLLER_CONSTRAINT_FEEDBACK_V1: &str =
    "pipewireao/controller-constraint-feedback-f32/1";

/// F32 commands in active virtual-deformable-mirror coordinate order.
pub const VDM_COMMAND_V1: &str = "pipewireao/vdm-command-f32/1";

/// F32 constraint feedback in active virtual-deformable-mirror coordinate
/// order.
pub const VDM_CONSTRAINT_FEEDBACK_V1: &str = "pipewireao/vdm-constraint-feedback-f32/1";

/// F32 row-major projection from concatenated controller coordinates to
/// active VDM coordinates. Shape is `[active-vdm, controller-coordinate]`.
pub const CONTROLLER_TO_VDM_PROJECTION_V1: &str = "pipewireao/controller-to-vdm-projection-f32/1";

/// F32 row-major projection from active VDM coordinates to concatenated
/// controller coordinates. Shape is `[controller-coordinate, active-vdm]`.
pub const VDM_TO_CONTROLLER_PROJECTION_V1: &str = "pipewireao/vdm-to-controller-projection-f32/1";

/// F32 row-major extrapolation from active to full VDM coordinates. Shape is
/// `[full-vdm, active-vdm]`.
pub const ACTIVE_TO_FULL_VDM_PROJECTION_V1: &str = "pipewireao/active-to-full-vdm-projection-f32/1";

/// F32 row-major truncation from full to active VDM coordinates. Shape is
/// `[active-vdm, full-vdm]`.
pub const FULL_TO_ACTIVE_VDM_PROJECTION_V1: &str = "pipewireao/full-to-active-vdm-projection-f32/1";

/// F32 row-major projection from full VDM coordinates to physical actuator
/// coordinates. Shape is `[physical-actuator, full-vdm]`.
pub const VDM_TO_PDM_PROJECTION_V1: &str = "pipewireao/vdm-to-pdm-projection-f32/1";

/// F32 row-major projection from physical actuator coordinates to full VDM
/// coordinates. Shape is `[full-vdm, physical-actuator]`.
pub const PDM_TO_VDM_PROJECTION_V1: &str = "pipewireao/pdm-to-vdm-projection-f32/1";

/// F32 row-major projection from controller-state coordinates to hidden-mode
/// coordinates. Shape is `[hidden-mode, controller-state]`.
pub const SHAPE_TO_HIDDEN_PROJECTION_V1: &str = "pipewireao/shape-to-hidden-projection-f32/1";

/// F32 row-major projection from hidden-mode coordinates to controller-state
/// coordinates. Shape is `[controller-state, hidden-mode]`.
pub const HIDDEN_TO_SHAPE_PROJECTION_V1: &str = "pipewireao/hidden-to-shape-projection-f32/1";

/// F32 reconstructed modal coefficients in controller mode order.
pub const RECONSTRUCTED_MODES_V1: &str = "pipewireao/reconstructed-modes-f32/1";

/// F32 physical wavefront-sensor response ratios in controller mode order.
pub const PHYSICAL_OPTICAL_GAIN_V1: &str = "pipewireao/physical-optical-gain-f32/1";

/// F32 commanded optical-gain dither values in prepared dither-mode order.
pub const OPTICAL_GAIN_DITHER_COMMAND_V1: &str = "pipewireao/optical-gain-dither-command-f32/1";

/// U64 zero-based frame within the current optical-gain estimate window.
pub const OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1: &str =
    "pipewireao/optical-gain-dither-window-frame-u64/1";

/// U64 zero-based sinusoid phase per prepared optical-gain dither mode.
pub const OPTICAL_GAIN_DITHER_PHASE_INDICES_V1: &str =
    "pipewireao/optical-gain-dither-phase-indices-u64/1";

/// BOOL8 flag identifying the final frame in one optical-gain estimate window.
pub const OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1: &str =
    "pipewireao/optical-gain-dither-window-complete-bool/1";

/// F32 raw reconstructed response in prepared optical-gain dither-mode order.
pub const OPTICAL_GAIN_DITHER_RESPONSE_V1: &str = "pipewireao/optical-gain-dither-response-f32/1";

/// F32 calibration response per optical-gain dither mode.
pub const OPTICAL_GAIN_CALIBRATION_RESPONSE_V1: &str =
    "pipewireao/optical-gain-calibration-response-f32/1";

/// F32 demodulated realized-command amplitude per dither mode.
pub const OPTICAL_GAIN_COMMAND_AMPLITUDE_V1: &str =
    "pipewireao/optical-gain-command-amplitude-f32/1";

/// F32 demodulated reconstructed-response amplitude per dither mode.
pub const OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1: &str =
    "pipewireao/optical-gain-response-amplitude-f32/1";

/// F32 physical optical-gain estimate in prepared dither-mode order.
pub const OPTICAL_GAIN_DITHER_ESTIMATE_V1: &str = "pipewireao/optical-gain-dither-estimate-f32/1";

/// F32 response-minus-command phase lag in radians per dither mode.
pub const OPTICAL_GAIN_PHASE_LAG_V1: &str = "pipewireao/optical-gain-phase-lag-f32/1";

/// BOOL8 validity of each optical-gain dither-mode estimate.
pub const OPTICAL_GAIN_ESTIMATE_VALIDITY_V1: &str =
    "pipewireao/optical-gain-estimate-validity-bool/1";

/// BOOL8 flag identifying publication of one complete optical-gain estimate.
pub const OPTICAL_GAIN_ESTIMATE_COMPLETE_V1: &str =
    "pipewireao/optical-gain-estimate-complete-bool/1";

/// BOOL8 acceptance status in prepared optical-gain dither-mode order.
pub const OPTICAL_GAIN_ACCEPTED_DITHER_MODES_V1: &str =
    "pipewireao/optical-gain-accepted-dither-modes-bool/1";

/// BOOL8 update status in reconstructed controller-mode order.
pub const OPTICAL_GAIN_UPDATED_RECONSTRUCTED_MODES_V1: &str =
    "pipewireao/optical-gain-updated-reconstructed-modes-bool/1";

/// F32 DO-CRIME corrector excitation samples in corrector-coordinate order.
pub const DOCRIME_EXCITATION_V1: &str = "pipewireao/docrime-excitation-f32/1";

/// F32 positive DO-CRIME excitation amplitudes in corrector-coordinate order.
pub const DOCRIME_EXCITATION_AMPLITUDE_V1: &str = "pipewireao/docrime-excitation-amplitude-f32/1";

/// U64 number of samples committed into published DO-CRIME covariance sums.
pub const DOCRIME_COVARIANCE_SAMPLE_COUNT_V1: &str =
    "pipewireao/docrime-covariance-sample-count-u64/1";

/// BOOL8 flag identifying a completed finite DO-CRIME covariance window.
pub const DOCRIME_COVARIANCE_COMPLETE_V1: &str = "pipewireao/docrime-covariance-complete-bool/1";

/// U8 DO-CRIME solve status: valid `0`, insufficient samples `1`, insufficient
/// variance `2`, excessive correlation `3`, non-positive-definite covariance
/// `4`, or insufficient factorization pivot `5`.
pub const DOCRIME_INTERACTION_MATRIX_STATUS_V1: &str =
    "pipewireao/docrime-interaction-matrix-status-u8/1";

/// U64 pair of corrector coordinates associated with a DO-CRIME solve status.
/// Single-coordinate failures repeat the coordinate; excessive correlation
/// publishes the first and second coordinates; statuses without a coordinate
/// publish two zeros.
pub const DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1: &str =
    "pipewireao/docrime-interaction-matrix-status-coordinates-u64/1";

/// F32 `[minimum, maximum]` corrector sample variances.
pub const DOCRIME_CORRECTOR_VARIANCE_RANGE_V1: &str =
    "pipewireao/docrime-corrector-variance-range-f32/1";

/// F32 maximum absolute off-diagonal corrector correlation.
pub const DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1: &str =
    "pipewireao/docrime-maximum-absolute-corrector-correlation-f32/1";

/// F32 minimum Cholesky factorization pivot ratio.
pub const DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1: &str =
    "pipewireao/docrime-minimum-factorization-pivot-ratio-f32/1";

/// Physical correction provenance: U8 codes 0 none, 1 controlled, 2 extrapolated, 3 unknown.
pub const PDM_CORRECTION_ROLE_V1: &str = "pipewireao/pdm-correction-role-u8/1";

/// Normalized dimensionless biquad coefficients in (coefficient, section, set) order.
pub const BIQUAD_COEFFICIENTS_V1: &str = "pipewireao/biquad-coefficients-f32/1";

/// Dimensionless nonnegative center-of-gravity pixel weights.
pub const CENTER_OF_GRAVITY_WEIGHTS_V1: &str = "pipewireao/center-of-gravity-weights-f32/1";

/// Weighted center-of-gravity flux in calibrated-pixel units.
pub const WEIGHTED_CENTER_OF_GRAVITY_FLUX_V1: &str =
    "pipewireao/weighted-center-of-gravity-flux-f32/1";

/// Eight-field Shack–Hartmann diagnostic summary.
pub const SHACK_HARTMANN_SUMMARY_V1: &str = "pipewireao/shack-hartmann-summary-f32/1";

/// Weighted Shack–Hartmann flux in PDE.
pub const WEIGHTED_SHACK_HARTMANN_FLUX_V1: &str = "pipewireao/weighted-shack-hartmann-flux-f32/1";

/// Per-subaperture dimensionless Shack–Hartmann pixel weights.
pub const SHACK_HARTMANN_WEIGHTS_V1: &str = "pipewireao/shack-hartmann-weights-f32/1";

/// F32 reference SCC chopped-image difference in complete detector-pixel order.
pub const SCC_REFERENCE_IMAGE_V1: &str = "pipewireao/scc-reference-image-f32/1";

/// F32 reference-subtracted SCC residuals in prepared dark-hole pixel order.
pub const SCC_SELECTED_PIXEL_RESIDUALS_V1: &str = "pipewireao/scc-selected-pixel-residuals-f32/1";

/// BOOL validity in prepared SCC dark-hole pixel order.
pub const SCC_SELECTED_PIXEL_VALIDITY_V1: &str = "pipewireao/scc-selected-pixel-validity-bool/1";

/// F32 fraction of prepared SCC dark-hole pixels valid in the current chopped pair.
pub const SCC_SELECTED_PIXEL_VALID_FRACTION_V1: &str =
    "pipewireao/scc-selected-pixel-valid-fraction-f32/1";

/// F32 selected-pixel-residual to SCC-controller-error reconstructor, with rows in controller-coordinate order and columns in prepared dark-hole pixel order.
pub const SCC_RECONSTRUCTOR_V1: &str = "pipewireao/scc-reconstructor-f32/1";

/// F32 calibrated pyramid I4Q differential signal in selected-coordinate X/Y order.
pub const PYRAMID_I4Q_SIGNAL_V1: &str = "pipewireao/pyramid-i4q-signal-f32/1";

/// F32 positive current-frame divisor used to calibrate a pyramid I4Q signal.
pub const PYRAMID_NORMALIZATION_DIVISOR_V1: &str = "pipewireao/pyramid-normalization-divisor-f32/1";

/// Boolean FGA-column-major valid support for one pyramid I4Q pupil region.
pub const PYRAMID_I4Q_SUPPORT_V1: &str = "pipewireao/pyramid-i4q-support-bool/1";

/// F32 pyramid I4Q reference differential signal in selected-coordinate X/Y order.
pub const PYRAMID_REFERENCE_SIGNAL_V1: &str = "pipewireao/pyramid-reference-signal-f32/1";

/// F32 per-component optical gain for a pyramid I4Q differential signal.
pub const PYRAMID_OPTICAL_GAIN_V1: &str = "pipewireao/pyramid-optical-gain-f32/1";

/// F32 LLOWFS modal coefficients in the calibrated shared LLOWFS mode order consumed by the modes-to-controlled-VDM projection.
pub const LLOWFS_MODES_V1: &str = "pipewireao/llowfs-modes-f32/1";

/// Finite F32 SimpleChains parameters in the exact legacy-simplechains-llowfs-v1 model order. This is a sparse prepared artifact rather than a per-frame input.
pub const ML_LLOWFS_PARAMETERS_V1: &str = "pipewireao/ml-llowfs-parameters-f32/1";

/// F32 mask-aware SCC spatial-carrier measurements with all selected real components followed by all selected imaginary components.
pub const SCC_SPATIAL_CARRIER_MEASUREMENTS_V1: &str =
    "pipewireao/scc-spatial-carrier-measurements-f32/1";

/// Finite nonnegative F32 confidence weights aligned with an SCC measurement vector.
pub const SCC_MEASUREMENT_WEIGHTS_V1: &str = "pipewireao/scc-measurement-weights-f32/1";

/// BOOL validity aligned with an SCC measurement vector.
pub const SCC_MEASUREMENT_VALIDITY_V1: &str = "pipewireao/scc-measurement-validity-bool/1";

/// F32 fraction of selected SCC spatial-carrier locations with adequate normalized-convolution support.
pub const SCC_SPATIAL_CARRIER_VALID_FRACTION_V1: &str =
    "pipewireao/scc-spatial-carrier-valid-fraction-f32/1";

/// Finite F32 detector-space SCC carrier phase in radians and complete detector-pixel order.
pub const SCC_CARRIER_PHASE_V1: &str = "pipewireao/scc-carrier-phase-f32/1";

/// Finite F32 normalized-convolution FIR coefficients in tap order.
pub const SCC_SPATIAL_FIR_V1: &str = "pipewireao/scc-spatial-fir-f32/1";

/// Finite F32 SCC interaction matrix with measurement rows and controller-coordinate columns.
pub const SCC_INTERACTION_MATRIX_V1: &str = "pipewireao/scc-interaction-matrix-f32/1";

/// Finite F32 SCC regularization operator with penalty rows and controller-coordinate columns.
pub const SCC_REGULARIZATION_OPERATOR_V1: &str = "pipewireao/scc-regularization-operator-f32/1";

/// U64 calibration signature synchronized with one complete calibrated Bi-O-edge detector frame.
pub const BI_O_EDGE_FRAME_CALIBRATION_SIGNATURE_V1: &str =
    "pipewireao/bi-o-edge-frame-calibration-signature-u64/1";

/// U64 active calibration signature for complete-frame Bi-O-edge estimation.
pub const BI_O_EDGE_CALIBRATION_SIGNATURE_V1: &str =
    "pipewireao/bi-o-edge-calibration-signature-u64/1";

/// F32 calibrated Bi-O-edge differential signal in selected-coordinate X/Y order.
pub const BI_O_EDGE_SIGNAL_V1: &str = "pipewireao/bi-o-edge-signal-f32/1";

/// F32 positive current-frame divisor used to calibrate a Bi-O-edge differential signal.
pub const BI_O_EDGE_NORMALIZATION_DIVISOR_V1: &str =
    "pipewireao/bi-o-edge-normalization-divisor-f32/1";

/// Boolean FGA-column-major valid support for one Bi-O-edge pupil region.
pub const BI_O_EDGE_SUPPORT_V1: &str = "pipewireao/bi-o-edge-support-bool/1";

/// F32 Bi-O-edge reference differential signal in selected-coordinate X/Y order.
pub const BI_O_EDGE_REFERENCE_SIGNAL_V1: &str = "pipewireao/bi-o-edge-reference-signal-f32/1";

/// F32 per-component optical gain for a Bi-O-edge differential signal.
pub const BI_O_EDGE_OPTICAL_GAIN_V1: &str = "pipewireao/bi-o-edge-optical-gain-f32/1";

/// U64 calibration signature synchronized with one complete calibrated Zernike pupil frame.
pub const ZERNIKE_PUPIL_SIGNAL_FRAME_CALIBRATION_SIGNATURE_V1: &str =
    "pipewireao/zernike-pupil-signal-frame-calibration-signature-u64/1";

/// U64 active calibration signature for complete-frame Zernike pupil estimation.
pub const ZERNIKE_PUPIL_SIGNAL_CALIBRATION_SIGNATURE_V1: &str =
    "pipewireao/zernike-pupil-signal-calibration-signature-u64/1";

/// F32 calibrated Zernike normalized pupil signal in selected-coordinate order.
pub const ZERNIKE_PUPIL_SIGNAL_V1: &str = "pipewireao/zernike-pupil-signal-f32/1";

/// F32 current-frame divisor used to calibrate a Zernike pupil signal.
pub const ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_DIVISOR_V1: &str =
    "pipewireao/zernike-pupil-signal-normalization-divisor-f32/1";

/// Boolean FGA-column-major valid support for a complete Zernike pupil image.
pub const ZERNIKE_PUPIL_SIGNAL_SUPPORT_V1: &str = "pipewireao/zernike-pupil-signal-support-bool/1";

/// F32 complete-frame reference normalized pupil image for Zernike estimation.
pub const ZERNIKE_PUPIL_SIGNAL_REFERENCE_PUPIL_IMAGE_V1: &str =
    "pipewireao/zernike-pupil-signal-reference-pupil-image-f32/1";

/// F32 accepted incidence-flux calibration divisor for Zernike pupil estimation.
pub const ZERNIKE_PUPIL_SIGNAL_INCIDENCE_FLUX_V1: &str =
    "pipewireao/zernike-pupil-signal-incidence-flux-f32/1";

/// F32 complete detector image with configured positive- and negative-defocus curvature branches.
pub const CURVATURE_PAIRED_IMAGE_V1: &str = "pipewireao/curvature-paired-image-f32/1";

/// F32 complete branch-by-channel matrix for positive- and negative-defocus curvature estimation.
pub const CURVATURE_PAIRED_CHANNELS_V1: &str = "pipewireao/curvature-paired-channels-f32/1";

/// U64 calibration signature synchronized with one complete paired-curvature input.
pub const CURVATURE_PAIRED_FRAME_CALIBRATION_SIGNATURE_V1: &str =
    "pipewireao/curvature-paired-frame-calibration-signature-u64/1";

/// U64 active calibration signature for paired-curvature estimation.
pub const CURVATURE_PAIRED_CALIBRATION_SIGNATURE_V1: &str =
    "pipewireao/curvature-paired-calibration-signature-u64/1";

/// F32 reference-subtracted paired-curvature signal in full FGA column-major grid order.
pub const CURVATURE_PAIRED_SIGNAL_V1: &str = "pipewireao/curvature-paired-signal-f32/1";

/// Boolean valid support for the full FGA column-major paired-curvature signal grid.
pub const CURVATURE_PAIRED_SUPPORT_V1: &str = "pipewireao/curvature-paired-support-bool/1";

/// F32 paired-curvature reference signal in full FGA column-major grid order.
pub const CURVATURE_PAIRED_REFERENCE_SIGNAL_V1: &str =
    "pipewireao/curvature-paired-reference-signal-f32/1";

/// Two positive finite F32 paired-curvature branch scales in positive- and negative-defocus order.
pub const CURVATURE_PAIRED_BRANCH_SCALES_V1: &str =
    "pipewireao/curvature-paired-branch-scales-f32/1";

/// F32 matrix mapping packed pyramid I4Q X-then-Y signals to reconstructed coordinates.
pub const PYRAMID_I4Q_RECONSTRUCTOR_V1: &str = "pipewireao/pyramid-i4q-reconstructor-f32/1";

/// Exact raw UInt16 pyramid pupil sums in Float64 ADC counts per exposure.
pub const PYRAMID_PUPIL_FLUX_ADU_F64_V1: &str = "pipewireao/pyramid-pupil-flux-adu-f64/1";
