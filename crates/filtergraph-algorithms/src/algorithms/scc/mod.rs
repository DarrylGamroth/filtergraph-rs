//! Self-Coherent Camera image and calibration operations.
//!
//! Rolling and push-pull differences preserve the calibrated image unit; on
//! the canonical detector boundary that unit is PDE. Push-pull response does
//! not divide by probe amplitude. A later normalized response would therefore
//! use image-unit/probe-unit.

mod chopped_image_difference;
mod push_pull_response;

pub use chopped_image_difference::{
    ChoppedImageDifferenceError, ChoppedImageDifferenceF32, ChoppedImageDifferenceF32Configuration,
    ChoppedImageDifferenceInput, ChoppedImageDifferenceOutput, ChoppedImageDifferencePlan,
    ChoppedImageDifferencePlanError, ChoppedImageDifferenceStatus, ChoppedImageDifferenceWorkspace,
    ChoppedImageKind,
};
pub use push_pull_response::{
    SccPushPullImageKind, SccPushPullResponseError, SccPushPullResponseF32,
    SccPushPullResponseF32Configuration, SccPushPullResponseInput, SccPushPullResponsePlan,
    SccPushPullResponsePlanError, SccPushPullResponseWorkspace,
};

mod pixel_residual;
pub use pixel_residual::*;

mod spatial_carrier_demodulation;
pub use spatial_carrier_demodulation::*;

mod weighted_regularized_reconstructor;
pub use weighted_regularized_reconstructor::*;
