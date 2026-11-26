//! General detector-image calibration, selection, and region operations.
//!
//! On the canonical detector boundary, pixel calibration maps detector ADU
//! through PDE/ADU coefficients to PDE. Selection and region extraction only
//! reorder or copy values and therefore preserve their input unit.

mod pixel_calibration;
mod pixel_calibration_f32;
pub use pixel_calibration_f32::*;
mod pixel_calibration_readout;
mod pixel_selection;
mod region_extraction;

pub use pixel_calibration::{
    PixelCalibrationError, PixelCalibrationPlan, PixelCalibrationRowU16F32,
    PixelCalibrationRowU16F32Configuration, PixelCalibrationRowU16F32PreparationError,
    PixelCalibrationRowU16F32ProcessError, PixelCalibrationRowWorkspace, PixelCalibrationU16F32,
    PixelCalibrationU16F32Configuration, PixelCalibrationU16F32PreparationError,
};
pub use pixel_calibration_readout::{
    DetectorReadoutCalibrationError, DetectorReadoutCalibrationPlan,
    DetectorReadoutCalibrationProcessError, DetectorReadoutCalibrationWorkspace,
    PixelCalibrationReadoutU16F32, PixelCalibrationReadoutU16F32Configuration,
    PixelCalibrationReadoutU16F32PreparationError,
};
pub use pixel_selection::{
    PixelSelectionF32, PixelSelectionF32Configuration, PixelSelectionParameterError,
    PixelSelectionPlan, PixelSelectionPlanError,
};
pub use region_extraction::{
    ImageRegionPixels, RegionExtractionError, RegionExtractionF32,
    RegionExtractionF32Configuration, RegionExtractionPlan, RegionPixels, RegionPixelsError,
    RegionPixelsMut,
};

mod pixel_validity;
pub use pixel_validity::*;
