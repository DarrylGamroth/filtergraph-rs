//! Optical-gain dither, estimation, update, and correction operations.
//!
//! Dither amplitudes, realized values, and command-amplitude estimates use the
//! named command-mode unit. Reconstructed responses retain their named modal
//! unit, and the prepared calibration response uses
//! response-unit/command-unit. Physical optical gain is dimensionless, phase
//! lag is rad, and reciprocal correction preserves the reconstructed modal
//! unit.

mod correction;
mod dither;
mod estimator;
mod update;

pub use correction::{
    OpticalGainCorrectionError, OpticalGainCorrectionF32, OpticalGainCorrectionF32Configuration,
    OpticalGainCorrectionPlan, OpticalGainCorrectionPlanError,
};
pub use dither::{
    MIN_OPTICAL_GAIN_DITHER_FRAMES_PER_PERIOD, OpticalGainDitherError, OpticalGainDitherF32,
    OpticalGainDitherF32Configuration, OpticalGainDitherMode, OpticalGainDitherPlan,
    OpticalGainDitherSchedule, OpticalGainDitherScheduleError, OpticalGainDitherWorkspace,
};
pub use estimator::{
    OpticalGainEstimateField, OpticalGainEstimateOutput, OpticalGainEstimatorError,
    OpticalGainEstimatorF32, OpticalGainEstimatorF32Configuration,
    OpticalGainEstimatorF32PreparationError, OpticalGainEstimatorInput,
    OpticalGainEstimatorInputKind, OpticalGainEstimatorPlan, OpticalGainEstimatorPlanError,
    OpticalGainEstimatorWorkspace,
};
pub use update::{
    OpticalGainAssignment, OpticalGainUpdateCriteria, OpticalGainUpdateError, OpticalGainUpdateF32,
    OpticalGainUpdateF32Configuration, OpticalGainUpdateF32PreparationError,
    OpticalGainUpdateF32Workspace, OpticalGainUpdateInput, OpticalGainUpdateOutput,
    OpticalGainUpdatePlan, OpticalGainUpdatePlanError, OpticalGainUpdateWorkspace,
    OpticalGainUpdateWorkspaceError,
};
