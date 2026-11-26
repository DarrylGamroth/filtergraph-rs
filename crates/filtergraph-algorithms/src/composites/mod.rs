//! Compatibility compositions of multiple numerical operations.
//!
//! The deformable-mirror composition adds and maps only VDM and PDM
//! actuator-coordinate values in µm of wavefront. Device-native conversion is
//! outside this compatibility layer.

mod deformable_mirror_command;
mod detector_readout;
mod pyramid_pixel_image;
mod pyramid_pixel_row;
mod pyramid_readout_reconstruction;
mod pyramid_reconstruction;
mod pyramid_row_reconstruction;
pub(crate) mod row_block;
mod shack_hartmann_image;
mod shack_hartmann_weighted;
pub use shack_hartmann_weighted::*;
mod shack_hartmann_measurement_block;
mod shack_hartmann_readout_reconstruction;
mod shack_hartmann_reconstruction;

pub use deformable_mirror_command::{
    DeformableMirrorCommandError, DeformableMirrorCommandInput, DeformableMirrorCommandOutput,
    DeformableMirrorCommandPlan, DeformableMirrorCommandPlanError,
    DeformableMirrorCommandWorkspace, DeformableMirrorCommandWorkspaceError,
};
pub use detector_readout::{
    DetectorFrameAssemblerError, DetectorFrameAssemblerF32, DetectorFrameAssemblerF32Configuration,
    DetectorFrameAssemblerPlan, DetectorFrameAssemblerProcessError,
    DetectorFrameAssemblerWorkspace, DetectorReadoutBlockInfo, DetectorReadoutDataError,
    DetectorReadoutDropReason, DetectorReadoutError, DetectorReadoutPlan,
    DetectorReadoutProfileError, DetectorReadoutRegion, DetectorReadoutResult,
};
pub use pyramid_pixel_image::{
    PyramidPixelImageError, PyramidPixelImagePlan, PyramidPixelImagePlanError,
    PyramidPixelImagePreparationError, PyramidPixelImageProcessError, PyramidPupilImageF32,
    PyramidPupilImageF32Configuration,
};
pub use pyramid_pixel_row::{
    PyramidPixelRowError, PyramidPixelRowExecutionError, PyramidPixelRowF32,
    PyramidPixelRowF32Configuration, PyramidPixelRowPlan, PyramidPixelRowPlanError,
    PyramidPixelRowPreparationError, PyramidPixelRowProcessError, PyramidPixelRowWorkspace,
};
pub use pyramid_readout_reconstruction::{
    PyramidReadoutReconstructionError, PyramidReadoutReconstructionOutput,
    PyramidReadoutReconstructionPlan, PyramidReadoutReconstructionPlanError,
    PyramidReadoutReconstructionPreparationError, PyramidReadoutReconstructionWorkspace,
    PyramidReadoutReconstructorF32, PyramidReadoutReconstructorF32Configuration,
};
pub use pyramid_reconstruction::{
    PyramidReconstructionWorkspace, PyramidReconstructorF32, PyramidReconstructorF32Configuration,
};
pub use pyramid_row_reconstruction::{
    PyramidRowReconstructionError, PyramidRowReconstructionExecutionError,
    PyramidRowReconstructionOutput, PyramidRowReconstructionPlan,
    PyramidRowReconstructionPlanError, PyramidRowReconstructionPreparationError,
    PyramidRowReconstructionProcessError, PyramidRowReconstructionWorkspace,
    PyramidRowReconstructorF32, PyramidRowReconstructorF32Configuration,
};
pub use row_block::{RowBlockDropReason, RowBlockInfo, RowBlockResult};
pub use shack_hartmann_image::{
    ShackHartmannImageF32, ShackHartmannImageF32Configuration, ShackHartmannImagePlan,
    ShackHartmannImagePlanError, ShackHartmannImagePreparationError,
    ShackHartmannImageProcessError,
};
pub use shack_hartmann_readout_reconstruction::{
    ShackHartmannReadoutReconstructionError, ShackHartmannReadoutReconstructionPlan,
    ShackHartmannReadoutReconstructionPlanError,
    ShackHartmannReadoutReconstructionPreparationError,
    ShackHartmannReadoutReconstructionWorkspace, ShackHartmannReadoutReconstructorF32,
    ShackHartmannReadoutReconstructorF32Configuration,
};
pub use shack_hartmann_reconstruction::{
    ShackHartmannReconstructorF32, ShackHartmannReconstructorF32Configuration,
};

pub use shack_hartmann_measurement_block::{
    ShackHartmannMeasurementBlockError, ShackHartmannMeasurementBlockF32,
    ShackHartmannMeasurementBlockF32Configuration, ShackHartmannMeasurementBlockPlan,
    ShackHartmannMeasurementBlockPreparationError, ShackHartmannMeasurementBlockProcessError,
    ShackHartmannMeasurementBlockResult, ShackHartmannMeasurementBlockWorkspace,
};

mod scc_reconstruction;
pub use scc_reconstruction::*;

mod shack_hartmann_weighted_measurement_block;
pub use shack_hartmann_weighted_measurement_block::*;

mod shack_hartmann_readout_measurement_block;
pub use shack_hartmann_readout_measurement_block::*;

mod shack_hartmann_row_reconstruction;
pub use shack_hartmann_row_reconstruction::*;

mod scc_correction;
pub use scc_correction::*;

mod pyramid_pupil_reconstruction;
pub use pyramid_pupil_reconstruction::*;

mod pyramid_measurement_block;
pub use pyramid_measurement_block::*;

mod pyramid_readout_measurement_block;
pub use pyramid_readout_measurement_block::*;
