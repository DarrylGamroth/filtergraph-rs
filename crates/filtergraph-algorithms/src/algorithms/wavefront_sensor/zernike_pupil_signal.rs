//! Signature-admitted current-frame Zernike pupil signals in column-major support order.
use super::pyramid_image::{area, column_major_to_row_major, finite, selected_offsets, support};
use crate::AlgorithmPlan;
use alloc::{boxed::Box, string::String, vec, vec::Vec};
use core::fmt;

/// Source Zernike normalization policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ZernikePupilSignalNormalization {
    /// Current selected-pupil mean clamped to Float32 epsilon.
    MeanValidFlux,
    /// Finite nonnegative incidence calibration; values at or below epsilon zero the signal.
    IncidenceFlux,
}
/// Invalid calibration/storage or a rejected complete Zernike frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ZernikePupilSignalError {
    /// Geometry, support, policy, or calibration is invalid.
    Configuration,
    /// Caller input/output or workspace has an incompatible extent.
    Extent,
    /// Frame and active calibration signatures differ.
    CalibrationMismatch,
    /// The complete acquired image contains a nonfinite pixel.
    NonFiniteInput,
    /// The normalization divisor is nonfinite.
    InvalidNormalization,
    /// A normalized reference-relative signal is nonfinite.
    NonFiniteOutput,
}
impl fmt::Display for ZernikePupilSignalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Configuration => "Zernike calibration or geometry is invalid",
            Self::Extent => "Zernike port or workspace extent is invalid",
            Self::CalibrationMismatch => "Zernike calibration signature mismatch",
            Self::NonFiniteInput => "Zernike acquired image contains a nonfinite pixel",
            Self::InvalidNormalization => "Zernike normalization divisor is nonfinite",
            Self::NonFiniteOutput => "Zernike pupil signal is nonfinite",
        })
    }
}
impl core::error::Error for ZernikePupilSignalError {}
/// Immutable complete detector support, reference image, normalization, and signature.
#[derive(Clone, Debug)]
pub struct ZernikePupilSignalPlan {
    image_size: (usize, usize),
    support: Box<[bool]>,
    selected_offsets: Box<[usize]>,
    reference_pupil_image: Box<[f32]>,
    normalization: ZernikePupilSignalNormalization,
    incidence_flux: f32,
    calibration_signature: u64,
}
impl ZernikePupilSignalPlan {
    /// Construction support and reference vectors retain source column-major order.
    pub fn new(c: ZernikePupilSignalF32Configuration) -> Result<Self, ZernikePupilSignalError> {
        use ZernikePupilSignalError::Configuration as E;
        let n = area(c.image_size).ok_or(E)?;
        let mask = support(c.zernike_pupil_support, c.image_size).ok_or(E)?;
        let offsets = selected_offsets(&mask, c.image_size);
        if offsets.is_empty() {
            return Err(E);
        }
        let reference = match c.initial_reference_pupil_image {
            Some(v) => column_major_to_row_major(&v, c.image_size).ok_or(E)?,
            None => vec![0.0; n],
        };
        if !finite(&reference) || !c.incidence_flux.is_finite() || c.incidence_flux < 0.0 {
            return Err(E);
        }
        let normalization = match c.normalization_policy.as_str() {
            "mean-valid-flux" => ZernikePupilSignalNormalization::MeanValidFlux,
            "incidence-flux" => ZernikePupilSignalNormalization::IncidenceFlux,
            _ => return Err(E),
        };
        Ok(Self {
            image_size: c.image_size,
            support: mask,
            selected_offsets: offsets.into_boxed_slice(),
            reference_pupil_image: reference.into_boxed_slice(),
            normalization,
            incidence_flux: c.incidence_flux,
            calibration_signature: c.initial_calibration_signature,
        })
    }
    /// Full detector rows and columns.
    pub const fn image_size(&self) -> (usize, usize) {
        self.image_size
    }
    /// Number of true support coordinates, ordered by column then row.
    pub fn selected_count(&self) -> usize {
        self.selected_offsets.len()
    }
    /// Allocate fixed selected-coordinate staging.
    pub fn workspace(&self) -> ZernikePupilSignalWorkspace {
        ZernikePupilSignalWorkspace {
            staged_signal: vec![0.0; self.selected_count()].into_boxed_slice(),
        }
    }
    /// Replace a logical row-major support with the same true count.
    pub fn with_support(&self, v: &[bool]) -> Result<Self, ZernikePupilSignalError> {
        if v.len() != self.support.len() {
            return Err(ZernikePupilSignalError::Configuration);
        }
        let offsets = selected_offsets(v, self.image_size);
        if offsets.len() != self.selected_count() {
            return Err(ZernikePupilSignalError::Configuration);
        }
        let mut p = self.clone();
        p.support = v.into();
        p.selected_offsets = offsets.into_boxed_slice();
        Ok(p)
    }
    /// Replace the finite logical row-major complete reference image.
    pub fn with_reference(&self, v: &[f32]) -> Result<Self, ZernikePupilSignalError> {
        if v.len() != self.reference_pupil_image.len() || !finite(v) {
            return Err(ZernikePupilSignalError::Configuration);
        }
        let mut p = self.clone();
        p.reference_pupil_image = v.into();
        Ok(p)
    }
    /// Replace finite nonnegative incidence flux.
    pub fn with_incidence_flux(&self, v: &[f32]) -> Result<Self, ZernikePupilSignalError> {
        if v.len() != 1 || !v[0].is_finite() || v[0] < 0.0 {
            return Err(ZernikePupilSignalError::Configuration);
        }
        let mut p = self.clone();
        p.incidence_flux = v[0];
        Ok(p)
    }
    /// Replace the active calibration signature.
    pub fn with_calibration_signature(&self, v: &[u64]) -> Result<Self, ZernikePupilSignalError> {
        if v.len() != 1 {
            return Err(ZernikePupilSignalError::Configuration);
        }
        let mut p = self.clone();
        p.calibration_signature = v[0];
        Ok(p)
    }
    fn image_index(&self, offset: usize) -> usize {
        (offset % self.image_size.0) * self.image_size.1 + offset / self.image_size.0
    }
}
/// Disposable signal staging; this operation has no temporal history.
#[derive(Clone, Debug)]
pub struct ZernikePupilSignalWorkspace {
    staged_signal: Box<[f32]>,
}
impl ZernikePupilSignalWorkspace {
    /// Reset has no mathematical state to change.
    pub fn reset(&mut self) {}
}
impl AlgorithmPlan for ZernikePupilSignalPlan {
    type Input<'a> = (&'a [f32], &'a [u64]);
    type Output<'a> = (&'a mut [f32], &'a mut [f32]);
    type Workspace = ZernikePupilSignalWorkspace;
    type Error = ZernikePupilSignalError;
    fn process<'i, 'o>(
        &self,
        (signal, divisor): Self::Output<'o>,
        w: &mut Self::Workspace,
        (image, signature): Self::Input<'i>,
    ) -> Result<(), Self::Error> {
        use ZernikePupilSignalError as E;
        if image.len() != self.reference_pupil_image.len()
            || signature.len() != 1
            || signal.len() != self.selected_count()
            || divisor.len() != 1
            || w.staged_signal.len() != signal.len()
        {
            return Err(E::Extent);
        }
        if signature[0] != self.calibration_signature {
            return Err(E::CalibrationMismatch);
        }
        if !finite(image) {
            return Err(E::NonFiniteInput);
        }
        let d = match self.normalization {
            ZernikePupilSignalNormalization::IncidenceFlux => self.incidence_flux,
            ZernikePupilSignalNormalization::MeanValidFlux => {
                let mut total = 0.0;
                for &offset in &self.selected_offsets {
                    total += image[self.image_index(offset)];
                }
                let mean = total / self.selected_count() as f32;
                if mean.is_nan() {
                    mean
                } else {
                    mean.max(f32::EPSILON)
                }
            }
        };
        if !d.is_finite() {
            return Err(E::InvalidNormalization);
        }
        if d <= f32::EPSILON {
            w.staged_signal.fill(0.0);
        } else {
            for (i, &offset) in self.selected_offsets.iter().enumerate() {
                let index = self.image_index(offset);
                let value = image[index] / d - self.reference_pupil_image[index];
                if !value.is_finite() {
                    return Err(E::NonFiniteOutput);
                }
                w.staged_signal[i] = value;
            }
        }
        signal.copy_from_slice(&w.staged_signal);
        divisor[0] = d;
        Ok(())
    }
}
crate::filtergraph_algorithm! {
    /// Current-frame normalized reference-relative Zernike pupil signal.
    pub ZernikePupilSignalF32 {
        label="zernike-pupil-signal-f32";
        configuration ZernikePupilSignalF32Configuration {
            image_size:(usize,usize)=GraphRebuild;zernike_pupil_support:Option<Vec<bool>>=Construction;
            initial_reference_pupil_image:Option<Vec<f32>>=Construction;normalization_policy:String=GraphRebuild;
            incidence_flux:f32=Construction;initial_calibration_signature:u64=Construction;
        }
        plan=ZernikePupilSignalPlan;workspace=ZernikePupilSignalWorkspace;properties=none;
        dimensions|plan,_workspace|{rows=plan.image_size().0;columns=plan.image_size().1;selected=plan.selected_count();}
        inputs{image:f32[rows,columns]="image",schema=crate::schemas::CALIBRATED_PIXELS_V1;
            frame_calibration_signature:u64[1]="frame-calibration-signature",schema=crate::schemas::ZERNIKE_PUPIL_SIGNAL_FRAME_CALIBRATION_SIGNATURE_V1;}
        outputs{signal:f32[selected]="signal",schema=crate::schemas::ZERNIKE_PUPIL_SIGNAL_V1,metadata_from="image";
            normalization_divisor:f32[1]="normalization-divisor",schema=crate::schemas::ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_DIVISOR_V1,metadata_from="image";}
        parameters{
            zernike_pupil_support:bool[rows,columns]="zernike-pupil-support",schema=crate::schemas::ZERNIKE_PUPIL_SIGNAL_SUPPORT_V1,replace|plan,values|{plan.with_support(values)};
            reference_pupil_image:f32[rows,columns]="reference-pupil-image",schema=crate::schemas::ZERNIKE_PUPIL_SIGNAL_REFERENCE_PUPIL_IMAGE_V1,replace|plan,values|{plan.with_reference(values)};
            incidence_flux:f32[1]="incidence-flux",schema=crate::schemas::ZERNIKE_PUPIL_SIGNAL_INCIDENCE_FLUX_V1,replace|plan,values|{plan.with_incidence_flux(values)};
            calibration_signature:u64[1]="calibration-signature",schema=crate::schemas::ZERNIKE_PUPIL_SIGNAL_CALIBRATION_SIGNATURE_V1,replace|plan,values|{plan.with_calibration_signature(values)};
        }
        parameter_error=ZernikePupilSignalError;
        prepare|configuration|{let plan=ZernikePupilSignalPlan::new(configuration)?;let workspace=plan.workspace();Ok((plan,workspace))}
        prepare_error=ZernikePupilSignalError;
        process|plan,workspace|{plan.process((signal,normalization_divisor),workspace,(image,frame_calibration_signature))}
        process_error=ZernikePupilSignalError;
        reset|_plan,workspace|{workspace.reset();}
    }
}
