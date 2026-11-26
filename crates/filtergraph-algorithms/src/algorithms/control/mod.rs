//! Controller-state and pseudo-open-loop operations.
//!
//! The leaky integrator requires its input, state, and output to share one unit;
//! its pole and gain are dimensionless. Canonical closed-loop DM error,
//! constraint feedback, state, and output are µm of wavefront. Pseudo-open-loop
//! matrices use the derived output-unit/input-unit ratio for each represented
//! coordinate-space mapping.

mod biquad_cascade;
mod closed_loop_correction;
mod corrector_allocation;
mod frame_delay;
pub use biquad_cascade::*;
pub use corrector_allocation::*;
pub use frame_delay::*;
mod leaky_integrator;
mod pseudo_open_loop_reconstruction;

pub use closed_loop_correction::{
    ClosedLoopCorrectionError, ClosedLoopCorrectionF32, ClosedLoopCorrectionF32Configuration,
    ClosedLoopCorrectionInput, ClosedLoopCorrectionPlan, ClosedLoopCorrectionPlanError,
    ClosedLoopCorrectionWorkspace, ClosedLoopIntegratorF32, ClosedLoopIntegratorF32Configuration,
    ValidityGatedClosedLoopIntegratorF32, ValidityGatedClosedLoopIntegratorF32Configuration,
};
pub use leaky_integrator::{
    LeakyIntegratorError, LeakyIntegratorF32, LeakyIntegratorF32Configuration, LeakyIntegratorPlan,
    LeakyIntegratorPlanError, LeakyIntegratorWorkspace, LeakyIntegratorWorkspaceError,
};
pub use pseudo_open_loop_reconstruction::{
    PseudoOpenLoopInputKind, PseudoOpenLoopMatrix, PseudoOpenLoopOutputKind,
    PseudoOpenLoopReconstructionError, PseudoOpenLoopReconstructionF32,
    PseudoOpenLoopReconstructionF32Configuration, PseudoOpenLoopReconstructionInput,
    PseudoOpenLoopReconstructionOutput, PseudoOpenLoopReconstructionPlan,
    PseudoOpenLoopReconstructionPlanError, PseudoOpenLoopReconstructionWorkspace,
    PseudoOpenLoopStage,
};
