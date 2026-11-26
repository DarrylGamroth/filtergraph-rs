//! DO-CRIME excitation, covariance, and interaction-matrix operations.
//!
//! Excitation amplitudes and samples use the associated corrector-coordinate
//! unit. Measurement--corrector cross-covariance has
//! measurement-unit·corrector-unit and corrector covariance has
//! corrector-unit². The solved interaction matrix consequently has
//! measurement-unit/corrector-unit. Variance criteria use corrector-unit²;
//! correlations, pivot ratios, and loop gain are dimensionless.

mod covariance;
mod excitation;
mod interaction_matrix;

pub use covariance::{
    DocrimeCovarianceError, DocrimeCovarianceF32, DocrimeCovarianceF32Configuration,
    DocrimeCovarianceInput, DocrimeCovarianceInputKind, DocrimeCovarianceMatrix,
    DocrimeCovariancePlan, DocrimeCovariancePlanError, DocrimeCovarianceWorkspace,
    DocrimeCovariances,
};
pub use excitation::{
    DocrimeBinaryExcitationF32, DocrimeBinaryExcitationF32Configuration,
    DocrimeBinaryExcitationPlan, DocrimeExcitationError, DocrimeExcitationF32,
    DocrimeExcitationF32Configuration, DocrimeExcitationPlan, DocrimeExcitationPlanError,
    DocrimeExcitationWorkspace,
};
pub use interaction_matrix::{
    DocrimeClosedLoopInteractionMatrixF32, DocrimeClosedLoopInteractionMatrixF32Configuration,
    DocrimeClosedLoopInteractionMatrixPlan, DocrimeConditioningCriteria,
    DocrimeConditioningCriteriaError, DocrimeInteractionMatrixDiagnostics,
    DocrimeInteractionMatrixError, DocrimeInteractionMatrixInputKind, DocrimeInteractionMatrixKind,
    DocrimeInteractionMatrixOutput, DocrimeInteractionMatrixPlanError,
    DocrimeInteractionMatrixPreparationError, DocrimeInteractionMatrixStage,
    DocrimeInteractionMatrixStatus, DocrimeInteractionMatrixWorkspace,
    DocrimeOpenLoopInteractionMatrixF32, DocrimeOpenLoopInteractionMatrixF32Configuration,
    DocrimeOpenLoopInteractionMatrixPlan,
};
