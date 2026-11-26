//! Signed Float32 detector calibration with failure-atomic publication.
use super::pixel_calibration::{PixelCalibrationError, PixelCalibrationPlan};
use alloc::{string::String, vec, vec::Vec};
use core::fmt;

/// Invalid signed Float32 calibration construction or frame arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelCalibrationF32Error {
    /// Image dimensions must be positive and their Float32 byte extent fit UInt32.
    ImageExtent,
    /// The declared raw quantity must have a nonempty schema.
    InputSchema,
    /// Calibration coefficients have invalid extents or values.
    Calibration(PixelCalibrationError),
    /// Raw, calibrated, scratch, and calibration planes must share one extent.
    DimensionMismatch,
    /// A signed detector sample is nonfinite.
    NonFiniteInput,
    /// The calibrated sample is nonfinite.
    NonFiniteOutput,
}
impl fmt::Display for PixelCalibrationF32Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageExtent => f.write_str(
                "detector dimensions must be positive and Float32 byte extent fit UInt32",
            ),
            Self::InputSchema => f.write_str("detector input schema must be nonempty"),
            Self::Calibration(error) => error.fmt(f),
            Self::DimensionMismatch => {
                f.write_str("pixel calibration arrays must share one image extent")
            }
            Self::NonFiniteInput => f.write_str("signed detector sample must be finite"),
            Self::NonFiniteOutput => f.write_str("calibrated detector sample must be finite"),
        }
    }
}
impl core::error::Error for PixelCalibrationF32Error {}
impl From<PixelCalibrationError> for PixelCalibrationF32Error {
    fn from(error: PixelCalibrationError) -> Self {
        Self::Calibration(error)
    }
}

/// Disposable signed-calibration scratch, separate from immutable coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelCalibrationWorkspace {
    image_size: (usize, usize),
    calibrated: Vec<f32>,
}
impl PixelCalibrationWorkspace {
    /// Clear scratch while the declaration retains its adopted flat/background.
    pub fn reset(&mut self) {
        self.calibrated.fill(0.0);
    }
}

fn process_pixel_calibration_f32(
    calibrated: &mut [f32],
    workspace: &mut PixelCalibrationWorkspace,
    plan: &PixelCalibrationPlan<f32, f32>,
    raw: &[f32],
) -> Result<(), PixelCalibrationF32Error> {
    if [calibrated.len(), raw.len(), workspace.calibrated.len()]
        .iter()
        .any(|&len| len != plan.len())
    {
        return Err(PixelCalibrationF32Error::DimensionMismatch);
    }
    // Evaluate contiguous row-major tiles. Each lane keeps the reference's
    // separately rounded subtraction and multiplication.
    let mut finite = true;
    for (((raw, flat), background), candidate) in raw
        .chunks(256)
        .zip(plan.flat().chunks(256))
        .zip(plan.background().chunks(256))
        .zip(workspace.calibrated.chunks_mut(256))
    {
        let mut tile_finite = true;
        for (((&value, &gain), &offset), result) in
            raw.iter().zip(flat).zip(background).zip(candidate)
        {
            *result = gain * (value - offset);
            tile_finite &= value.is_finite() & result.is_finite();
        }
        finite &= tile_finite;
    }
    if !finite {
        // JFG reports the first failure in logical column-major order. A cold
        // rescan preserves that precedence without striding on valid frames.
        let (rows, columns) = workspace.image_size;
        for column in 0..columns {
            for row in 0..rows {
                let index = row * columns + column;
                if !raw[index].is_finite() {
                    return Err(PixelCalibrationF32Error::NonFiniteInput);
                }
                if !workspace.calibrated[index].is_finite() {
                    return Err(PixelCalibrationF32Error::NonFiniteOutput);
                }
            }
        }
    }
    calibrated.copy_from_slice(&workspace.calibrated);
    Ok(())
}

crate::filtergraph_algorithm! {
    /// Calibrate signed Float32 samples as flat × (raw − background), without clipping.
    pub PixelCalibrationF32 {
        label = "pixel-calibration-f32";
        configuration PixelCalibrationF32Configuration {
            image_size: (usize, usize) = GraphRebuild;
            input_schema: String = GraphRebuild;
            initial_flat: Option<Vec<f32>> = Construction;
            initial_background: Option<Vec<f32>> = Construction;
        }
        plan = PixelCalibrationPlan<f32, f32>;
        workspace = PixelCalibrationWorkspace;
        properties = none;
        dimensions |_plan, workspace| {
            rows = workspace.image_size.0;
            columns = workspace.image_size.1;
        }
        inputs { raw: f32 [rows, columns] = "raw", schema = configuration.input_schema.as_str(); }
        outputs { calibrated: f32 [rows, columns] = "calibrated", schema = crate::schemas::CALIBRATED_PIXELS_V1, metadata_from = "raw"; }
        parameters {
            flat: f32 [rows, columns] = "flat", schema = crate::schemas::FLAT_CALIBRATION_V1,
                replace |plan, values| { plan.with_flat(values) };
            background: f32 [rows, columns] = "background", schema = crate::schemas::BACKGROUND_CALIBRATION_V1,
                replace |plan, values| { plan.with_background(values) };
        }
        parameter_error = PixelCalibrationError;
        prepare |configuration| {
            let (rows, columns) = configuration.image_size;
            let count = rows.checked_mul(columns).filter(|&count| rows != 0 && columns != 0 && count <= u32::MAX as usize / core::mem::size_of::<f32>()).ok_or(PixelCalibrationF32Error::ImageExtent)?;
            if configuration.input_schema.trim().is_empty() { return Err(PixelCalibrationF32Error::InputSchema); }
            let flat = configuration.initial_flat.unwrap_or_else(|| vec![1.0;count]);
            let background = configuration.initial_background.unwrap_or_else(||vec![0.0;count]);
            if flat.len() != count || background.len() != count { return Err(PixelCalibrationF32Error::DimensionMismatch); }
            let plan = PixelCalibrationPlan::new(flat,background)?;
            let workspace = PixelCalibrationWorkspace {image_size:(rows,columns), calibrated:vec![0.0;count]};
            Ok((plan,workspace))
        }
        prepare_error = PixelCalibrationF32Error;
        process |plan,workspace| { process_pixel_calibration_f32(calibrated,workspace,plan,raw) }
        process_error = PixelCalibrationF32Error;
        reset |_plan,workspace| { workspace.reset(); }
    }
}
