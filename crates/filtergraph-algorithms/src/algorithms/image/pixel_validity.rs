//! Finite-only pixel validity without calibration, saturation or static masking.
use crate::NoWorkspace;
use alloc::string::String;
use core::fmt;

/// Invalid prepared pixel extent or mismatched frame dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelValidityError {
    /// Positive Float32 byte extent must fit UInt32.
    Extent,
    /// Input and validity output must share the prepared extent.
    DimensionMismatch,
}
impl fmt::Display for PixelValidityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pixel validity: {self:?}")
    }
}
impl core::error::Error for PixelValidityError {}
crate::filtergraph_algorithm! {
    /// Each validity coordinate equals `isfinite(input[i])`.
    pub PixelValidityF32 {
        label="pixel-validity-f32";
        configuration PixelValidityF32Configuration { extent:usize=GraphRebuild; input_schema:String=GraphRebuild; }
        plan=usize;
        workspace=NoWorkspace;
        properties=none;
        dimensions|plan,_workspace| { extent=*plan; }
        inputs { input:f32[extent]="input",schema=configuration.input_schema.as_str(), optional_sample_metadata = ["chopped_image_kind", "trigger_state", "acquisition_identity", "exposure_duration_nanoseconds"]; }
        outputs { validity:bool[extent]="validity",schema=crate::schemas::CALIBRATED_PIXEL_VALIDITY_V1,metadata_from="input"; }
        prepare|configuration| {
            let extent=configuration.extent;
            if extent==0||extent>u32::MAX as usize/core::mem::size_of::<f32>() { return Err(PixelValidityError::Extent); }
            Ok((extent,NoWorkspace))
        }
        prepare_error=PixelValidityError;
        process|plan,_workspace| {
            if input.len()!=*plan||validity.len()!=*plan { return Err(PixelValidityError::DimensionMismatch); }
            for (valid,value) in validity.iter_mut().zip(input) { *valid=value.is_finite(); }
            Ok(())
        }
        process_error=PixelValidityError;
    }
}
