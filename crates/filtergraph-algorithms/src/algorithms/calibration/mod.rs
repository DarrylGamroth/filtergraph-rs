//! General calibration probing, differencing, decoding, and diagnostics.
//!
//! Probe amplitudes and generated values use the associated corrector-coordinate
//! unit. Matched differences preserve the units of their measurement and
//! corrector inputs independently. A decoded interaction-matrix coefficient has
//! measurement-unit/corrector-unit; its column norms retain that coefficient
//! unit, while relative stability diagnostics are dimensionless.

mod calibration_probe;
mod interaction_matrix_stability;
mod matched_calibration_differencing;
mod structured_calibration;
mod structured_interaction_matrix;

pub use calibration_probe::{
    CalibrationProbeError, CalibrationProbeOutput, CalibrationProbePolarity, CalibrationProbeStep,
    HadamardPushPullF32, HadamardPushPullF32Configuration, HadamardPushPullPlan,
    HadamardPushPullWorkspace, ModalPushPullF32, ModalPushPullF32Configuration, ModalPushPullPlan,
    ModalPushPullPlanError, ModalPushPullWorkspace, ZonalPushPullF32,
    ZonalPushPullF32Configuration, ZonalPushPullPlan, ZonalPushPullWorkspace,
};
pub use interaction_matrix_stability::{
    InteractionMatrixStabilityCriteria, InteractionMatrixStabilityCriteriaError,
    InteractionMatrixStabilityDiagnostics, InteractionMatrixStabilityError,
    InteractionMatrixStabilityF32, InteractionMatrixStabilityF32Configuration,
    InteractionMatrixStabilityF32PreparationError, InteractionMatrixStabilityInput,
    InteractionMatrixStabilityInputKind, InteractionMatrixStabilityPlan,
    InteractionMatrixStabilityPlanError, InteractionMatrixStabilityStatus,
};
pub use matched_calibration_differencing::{
    MatchedCalibrationDifferencingError, MatchedCalibrationDifferencingF32,
    MatchedCalibrationDifferencingF32Configuration, MatchedCalibrationDifferencingInput,
    MatchedCalibrationDifferencingInputKind, MatchedCalibrationDifferencingOutput,
    MatchedCalibrationDifferencingOutputKind, MatchedCalibrationDifferencingPlan,
    MatchedCalibrationDifferencingPlanError, MatchedCalibrationDifferencingStatus,
    MatchedCalibrationDifferencingWorkspace,
};
pub use structured_calibration::{CalibrationProbeExtent, CalibrationProbePlanError};
pub use structured_interaction_matrix::{
    HadamardInteractionMatrixF32, HadamardInteractionMatrixF32Configuration,
    HadamardInteractionMatrixPlan, StructuredInteractionMatrixError,
    StructuredInteractionMatrixExtent, StructuredInteractionMatrixPlanError,
    StructuredInteractionMatrixWorkspace, ZonalInteractionMatrixF32,
    ZonalInteractionMatrixF32Configuration, ZonalInteractionMatrixPlan,
};
