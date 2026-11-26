//! Complete-frame Bi-O-edge differential signals with calibration signatures.
//! The pinned reference shares the four-pupil differential and reduction order
//! with PyramidImage, with signature admission and a stricter divisor threshold.
use super::pyramid_image::{
    PyramidImageError, PyramidImageF32Configuration, PyramidImagePlan, PyramidImageWorkspace,
};
use crate::AlgorithmPlan;
use alloc::{string::String, vec::Vec};
use core::fmt;

/// Invalid Bi-O-edge calibration/storage or a rejected frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BiOEdgeImageError {
    /// Geometry, support, policy, or finite calibration is invalid.
    Configuration,
    /// Caller input or output has an incompatible extent.
    PortExtent,
    /// Workspace was prepared for another selected-coordinate extent.
    WorkspaceExtent,
    /// Frame and active calibration signatures differ.
    CalibrationMismatch,
    /// A complete configured pupil contains a nonfinite pixel.
    NonFiniteInput,
    /// Normalization must be finite and greater than Float32 epsilon.
    InvalidNormalization,
    /// A calibrated differential signal is nonfinite.
    NonFiniteOutput,
}
impl fmt::Display for BiOEdgeImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Configuration => "Bi-O-edge calibration or geometry is invalid",
            Self::PortExtent => "Bi-O-edge port extent is invalid",
            Self::WorkspaceExtent => "Bi-O-edge workspace extent is invalid",
            Self::CalibrationMismatch => "Bi-O-edge calibration signature mismatch",
            Self::NonFiniteInput => "Bi-O-edge pupil contains a nonfinite pixel",
            Self::InvalidNormalization => "Bi-O-edge normalization divisor is invalid",
            Self::NonFiniteOutput => "Bi-O-edge signal is nonfinite",
        })
    }
}
impl core::error::Error for BiOEdgeImageError {}
impl From<PyramidImageError> for BiOEdgeImageError {
    fn from(e: PyramidImageError) -> Self {
        match e {
            PyramidImageError::Configuration => Self::Configuration,
            PyramidImageError::PortExtent => Self::PortExtent,
            PyramidImageError::WorkspaceExtent => Self::WorkspaceExtent,
            PyramidImageError::NonFiniteInput => Self::NonFiniteInput,
            PyramidImageError::InvalidNormalization => Self::InvalidNormalization,
            PyramidImageError::NonFiniteOutput => Self::NonFiniteOutput,
        }
    }
}
/// Immutable calibrated Bi-O-edge image operation.
#[derive(Clone, Debug)]
pub struct BiOEdgeImagePlan {
    pupils: PyramidImagePlan,
    calibration_signature: u64,
}
impl BiOEdgeImagePlan {
    /// Preserve source column-major construction calibration and validate incidence flux.
    pub fn new(c: BiOEdgeImageF32Configuration) -> Result<Self, BiOEdgeImageError> {
        if c.normalization_policy == "incidence-flux"
            && !(c.incidence_flux.is_finite() && c.incidence_flux > f32::EPSILON)
        {
            return Err(BiOEdgeImageError::Configuration);
        }
        let signature = c.initial_calibration_signature;
        let pupils = PyramidImagePlan::new(PyramidImageF32Configuration {
            image_size: c.image_size,
            pupil_size: c.pupil_size,
            pupil_origins: c.pupil_origins,
            i4q_support: c.bi_o_edge_support,
            initial_reference_signal: c.initial_reference_signal,
            initial_optical_gain: c.initial_optical_gain,
            normalization_policy: c.normalization_policy,
            incidence_flux: c.incidence_flux,
        })?;
        Ok(Self {
            pupils,
            calibration_signature: signature,
        })
    }
    /// Detector image rows and columns.
    pub fn image_size(&self) -> (usize, usize) {
        self.pupils.image_size()
    }
    /// Pupil rows and columns.
    pub fn pupil_size(&self) -> (usize, usize) {
        self.pupils.pupil_size()
    }
    /// Selected support-coordinate count.
    pub fn selected_count(&self) -> usize {
        self.pupils.selected_count()
    }
    /// Allocate fixed output and reduction scratch.
    pub fn workspace(&self) -> BiOEdgeImageWorkspace {
        BiOEdgeImageWorkspace {
            pupils: self.pupils.workspace(),
        }
    }
    /// Replace zero-based origin pairs in logical row-major port order.
    pub fn with_origins(&self, v: &[u32]) -> Result<Self, BiOEdgeImageError> {
        Ok(Self {
            pupils: self.pupils.with_origins(v)?,
            ..self.clone()
        })
    }
    /// Replace row-major support while preserving selected extent.
    pub fn with_support(&self, v: &[bool]) -> Result<Self, BiOEdgeImageError> {
        Ok(Self {
            pupils: self.pupils.with_support(v)?,
            ..self.clone()
        })
    }
    /// Replace logical selected-coordinate reference X/Y pairs.
    pub fn with_reference(&self, v: &[f32]) -> Result<Self, BiOEdgeImageError> {
        Ok(Self {
            pupils: self.pupils.with_reference(v)?,
            ..self.clone()
        })
    }
    /// Replace logical selected-coordinate gain X/Y pairs.
    pub fn with_gain(&self, v: &[f32]) -> Result<Self, BiOEdgeImageError> {
        Ok(Self {
            pupils: self.pupils.with_gain(v)?,
            ..self.clone()
        })
    }
    /// Replace the active calibration signature.
    pub fn with_calibration_signature(&self, v: &[u64]) -> Result<Self, BiOEdgeImageError> {
        if v.len() != 1 {
            return Err(BiOEdgeImageError::Configuration);
        }
        Ok(Self {
            calibration_signature: v[0],
            ..self.clone()
        })
    }
}
/// Disposable stateless output and normalization staging.
#[derive(Clone, Debug)]
pub struct BiOEdgeImageWorkspace {
    pupils: PyramidImageWorkspace,
}
impl BiOEdgeImageWorkspace {
    /// Reset has no temporal history to change.
    pub fn reset(&mut self) {}
}
impl AlgorithmPlan for BiOEdgeImagePlan {
    type Input<'a> = (&'a [f32], &'a [u64]);
    type Output<'a> = (&'a mut [f32], &'a mut [f32]);
    type Workspace = BiOEdgeImageWorkspace;
    type Error = BiOEdgeImageError;
    fn process<'i, 'o>(
        &self,
        output: Self::Output<'o>,
        workspace: &mut Self::Workspace,
        (image, signature): Self::Input<'i>,
    ) -> Result<(), Self::Error> {
        self.pupils.check(
            output.0.len(),
            output.1.len(),
            &workspace.pupils,
            image.len(),
        )?;
        if signature.len() != 1 {
            return Err(BiOEdgeImageError::PortExtent);
        }
        if signature[0] != self.calibration_signature {
            return Err(BiOEdgeImageError::CalibrationMismatch);
        }
        self.pupils
            .process_with_minimum(output, &mut workspace.pupils, image, f32::EPSILON)
            .map_err(Into::into)
    }
}
crate::filtergraph_algorithm! {
    /// Signature-admitted current-frame Bi-O-edge differential signal.
    pub BiOEdgeImageF32 {
        label="bi-o-edge-image-f32";
        configuration BiOEdgeImageF32Configuration {
            image_size:(usize,usize)=GraphRebuild;pupil_size:(usize,usize)=GraphRebuild;pupil_origins:Vec<[usize;2]>=GraphRebuild;
            bi_o_edge_support:Option<Vec<bool>>=Construction;initial_reference_signal:Option<Vec<f32>>=Construction;initial_optical_gain:Option<Vec<f32>>=Construction;
            normalization_policy:String=GraphRebuild;incidence_flux:f32=Construction;initial_calibration_signature:u64=Construction;
        }
        plan=BiOEdgeImagePlan;workspace=BiOEdgeImageWorkspace;properties=none;
        dimensions|plan,_workspace|{image_rows=plan.image_size().0;image_columns=plan.image_size().1;pupil_rows=plan.pupil_size().0;pupil_columns=plan.pupil_size().1;selected=plan.selected_count();}
        inputs{image:f32[image_rows,image_columns]="image",schema=crate::schemas::CALIBRATED_PIXELS_V1;
            frame_calibration_signature:u64[1]="frame-calibration-signature",schema=crate::schemas::BI_O_EDGE_FRAME_CALIBRATION_SIGNATURE_V1;}
        outputs{signal:f32[selected,2]="signal",schema=crate::schemas::BI_O_EDGE_SIGNAL_V1,metadata_from="image";
            normalization_divisor:f32[1]="normalization-divisor",schema=crate::schemas::BI_O_EDGE_NORMALIZATION_DIVISOR_V1,metadata_from="image";}
        parameters{
            pupil_origins:u32[4,2]="pupil-origins",schema=crate::schemas::REGION_ORIGINS_V1,replace|plan,values|{plan.with_origins(values)};
            bi_o_edge_support:bool[pupil_rows,pupil_columns]="bi-o-edge-support",schema=crate::schemas::BI_O_EDGE_SUPPORT_V1,replace|plan,values|{plan.with_support(values)};
            reference_signal:f32[selected,2]="reference-signal",schema=crate::schemas::BI_O_EDGE_REFERENCE_SIGNAL_V1,replace|plan,values|{plan.with_reference(values)};
            optical_gain:f32[selected,2]="optical-gain",schema=crate::schemas::BI_O_EDGE_OPTICAL_GAIN_V1,replace|plan,values|{plan.with_gain(values)};
            calibration_signature:u64[1]="calibration-signature",schema=crate::schemas::BI_O_EDGE_CALIBRATION_SIGNATURE_V1,replace|plan,values|{plan.with_calibration_signature(values)};
        }
        parameter_error=BiOEdgeImageError;
        prepare|configuration|{let plan=BiOEdgeImagePlan::new(configuration)?;let workspace=plan.workspace();Ok((plan,workspace))}
        prepare_error=BiOEdgeImageError;
        process|plan,workspace|{plan.process((signal,normalization_divisor),workspace,(image,frame_calibration_signature))}
        process_error=BiOEdgeImageError;
        reset|_plan,workspace|{workspace.reset();}
    }
}
