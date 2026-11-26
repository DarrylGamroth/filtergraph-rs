//! Signature-admitted paired curvature signals in column-major signal-grid order.
//! Branch order is positive then negative defocus. Tile sums retain local
//! column-then-row order, the source epsilon policy, and finite-overflow scaling.
use super::pyramid_image::{area, column_major_to_row_major, finite, support};
use crate::AlgorithmPlan;
use alloc::{boxed::Box, vec, vec::Vec};
use core::fmt;

/// Invalid paired-curvature calibration/storage or a rejected acquired sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CurvaturePairedSignalError {
    /// Geometry, channel map, support/reference shape, or positive scales are invalid.
    Configuration,
    /// Input, signature, output, or scratch has an incompatible extent.
    Extent,
    /// Frame and active calibration signatures differ.
    CalibrationMismatch,
    /// The complete acquired input contains a nonfinite sample.
    NonFiniteInput,
    /// Reduction, scaling, or normalized signal arithmetic is nonfinite.
    NonFiniteOutput,
}
impl fmt::Display for CurvaturePairedSignalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Configuration => "curvature calibration or geometry is invalid",
            Self::Extent => "curvature port or workspace extent is invalid",
            Self::CalibrationMismatch => "curvature calibration signature mismatch",
            Self::NonFiniteInput => "curvature acquired input contains a nonfinite sample",
            Self::NonFiniteOutput => "curvature signal arithmetic is nonfinite",
        })
    }
}
impl core::error::Error for CurvaturePairedSignalError {}

#[derive(Clone, Debug)]
struct CurvatureCalibration {
    signal_size: (usize, usize),
    support: Box<[bool]>,
    reference_signal: Box<[f32]>,
    branch_scales: [f32; 2],
    calibration_signature: u64,
}
impl CurvatureCalibration {
    fn new(
        signal_size: (usize, usize),
        mask: Option<Vec<bool>>,
        reference: Option<Vec<f32>>,
        scales: Option<Vec<f32>>,
        signature: u64,
    ) -> Result<Self, CurvaturePairedSignalError> {
        use CurvaturePairedSignalError::Configuration as E;
        let n = area(signal_size).ok_or(E)?;
        let mask = support(mask, signal_size).ok_or(E)?;
        let reference = match reference {
            Some(v) => column_major_to_row_major(&v, signal_size).ok_or(E)?,
            None => vec![0.0; n],
        };
        let scales: [f32; 2] = match scales {
            Some(v) => v.try_into().map_err(|_| E)?,
            None => [1.0; 2],
        };
        if !finite(&reference) || scales.iter().any(|&v| !v.is_finite() || v <= 0.0) {
            return Err(E);
        }
        Ok(Self {
            signal_size,
            support: mask,
            reference_signal: reference.into_boxed_slice(),
            branch_scales: scales,
            calibration_signature: signature,
        })
    }
    fn with_support(&self, v: &[bool]) -> Result<Self, CurvaturePairedSignalError> {
        if v.len() != self.support.len() {
            return Err(CurvaturePairedSignalError::Configuration);
        }
        let mut c = self.clone();
        c.support = v.into();
        Ok(c)
    }
    fn with_reference(&self, v: &[f32]) -> Result<Self, CurvaturePairedSignalError> {
        if v.len() != self.reference_signal.len() || !finite(v) {
            return Err(CurvaturePairedSignalError::Configuration);
        }
        let mut c = self.clone();
        c.reference_signal = v.into();
        Ok(c)
    }
    fn with_scales(&self, v: &[f32]) -> Result<Self, CurvaturePairedSignalError> {
        if v.len() != 2 || v.iter().any(|&v| !v.is_finite() || v <= 0.0) {
            return Err(CurvaturePairedSignalError::Configuration);
        }
        let mut c = self.clone();
        c.branch_scales = [v[0], v[1]];
        Ok(c)
    }
    fn with_signature(&self, v: &[u64]) -> Result<Self, CurvaturePairedSignalError> {
        if v.len() != 1 {
            return Err(CurvaturePairedSignalError::Configuration);
        }
        let mut c = self.clone();
        c.calibration_signature = v[0];
        Ok(c)
    }
    fn check(
        &self,
        signal: &[f32],
        w: &CurvaturePairedSignalWorkspace,
        input: &[f32],
        input_count: usize,
        signature: &[u64],
    ) -> Result<(), CurvaturePairedSignalError> {
        use CurvaturePairedSignalError as E;
        if signal.len() != self.support.len()
            || w.staged_signal.len() != signal.len()
            || input.len() != input_count
            || signature.len() != 1
        {
            return Err(E::Extent);
        }
        if signature[0] != self.calibration_signature {
            return Err(E::CalibrationMismatch);
        }
        if !finite(input) {
            return Err(E::NonFiniteInput);
        }
        Ok(())
    }
}
fn paired_value(plus: f32, minus: f32, reference: f32) -> f32 {
    if !(plus.is_finite() && minus.is_finite()) {
        return f32::NAN;
    }
    let total = plus + minus;
    if total <= f32::EPSILON {
        return 0.0;
    }
    let difference = plus - minus;
    if total.is_finite() && difference.is_finite() {
        return difference / (total + f32::EPSILON) - reference;
    }
    let scale = plus.abs().max(minus.abs());
    let a = plus / scale;
    let b = minus / scale;
    (a - b) / (a + b + f32::EPSILON / scale) - reference
}
/// Paired rectangular branches reduced to a complete signal grid.
#[derive(Clone, Debug)]
pub struct CurvaturePairedImagePlan {
    image_size: (usize, usize),
    branch_size: (usize, usize),
    branch_origins: [[usize; 2]; 2],
    calibration: CurvatureCalibration,
}
impl CurvaturePairedImagePlan {
    /// Support/reference construction vectors retain source column-major grid order.
    pub fn new(
        c: CurvaturePairedImageF32Configuration,
    ) -> Result<Self, CurvaturePairedSignalError> {
        use CurvaturePairedSignalError::Configuration as E;
        area(c.image_size).ok_or(E)?;
        area(c.branch_size).ok_or(E)?;
        area(c.signal_size).ok_or(E)?;
        if !c.branch_size.0.is_multiple_of(c.signal_size.0)
            || !c.branch_size.1.is_multiple_of(c.signal_size.1)
        {
            return Err(E);
        }
        let origins = [c.positive_branch_origin, c.negative_branch_origin];
        for &[r, col] in &origins {
            if r.checked_add(c.branch_size.0)
                .is_none_or(|v| v > c.image_size.0)
                || col
                    .checked_add(c.branch_size.1)
                    .is_none_or(|v| v > c.image_size.1)
            {
                return Err(E);
            }
        }
        let calibration = CurvatureCalibration::new(
            c.signal_size,
            c.curvature_support,
            c.initial_reference_signal,
            c.initial_branch_scales,
            c.initial_calibration_signature,
        )?;
        Ok(Self {
            image_size: c.image_size,
            branch_size: c.branch_size,
            branch_origins: origins,
            calibration,
        })
    }
    /// Full detector rows and columns.
    pub const fn image_size(&self) -> (usize, usize) {
        self.image_size
    }
    /// Full signal-grid rows and columns.
    pub const fn signal_size(&self) -> (usize, usize) {
        self.calibration.signal_size
    }
    /// Allocate disposable full-grid output staging.
    pub fn workspace(&self) -> CurvaturePairedSignalWorkspace {
        CurvaturePairedSignalWorkspace {
            staged_signal: vec![0.0; self.calibration.support.len()].into_boxed_slice(),
        }
    }
}
/// Paired branch rows and a fixed channel map for a full signal grid.
#[derive(Clone, Debug)]
pub struct CurvaturePairedChannelPlan {
    input_size: (usize, usize),
    branch_rows: [usize; 2],
    channel_offsets: Box<[usize]>,
    calibration: CurvatureCalibration,
}
impl CurvaturePairedChannelPlan {
    /// Channel offsets map every column-major signal-grid coordinate to an input column.
    pub fn new(
        c: CurvaturePairedChannelF32Configuration,
    ) -> Result<Self, CurvaturePairedSignalError> {
        use CurvaturePairedSignalError::Configuration as E;
        area((c.input_branch_rows, c.input_channels)).ok_or(E)?;
        let n = area(c.signal_size).ok_or(E)?;
        if c.positive_branch_row >= c.input_branch_rows
            || c.negative_branch_row >= c.input_branch_rows
            || c.channel_offsets.len() != n
            || c.channel_offsets.iter().any(|&v| v >= c.input_channels)
        {
            return Err(E);
        }
        let calibration = CurvatureCalibration::new(
            c.signal_size,
            c.curvature_support,
            c.initial_reference_signal,
            c.initial_branch_scales,
            c.initial_calibration_signature,
        )?;
        Ok(Self {
            input_size: (c.input_branch_rows, c.input_channels),
            branch_rows: [c.positive_branch_row, c.negative_branch_row],
            channel_offsets: c.channel_offsets.into_boxed_slice(),
            calibration,
        })
    }
    /// Complete branch-by-channel input dimensions.
    pub const fn input_size(&self) -> (usize, usize) {
        self.input_size
    }
    /// Full signal-grid rows and columns.
    pub const fn signal_size(&self) -> (usize, usize) {
        self.calibration.signal_size
    }
    /// Allocate disposable full-grid output staging.
    pub fn workspace(&self) -> CurvaturePairedSignalWorkspace {
        CurvaturePairedSignalWorkspace {
            staged_signal: vec![0.0; self.calibration.support.len()].into_boxed_slice(),
        }
    }
}
macro_rules! calibration_replacements {
    ($plan:ty) => {
        impl $plan {
            /// Replace row-major full-grid support; an empty support is allowed.
            pub fn with_support(&self, v: &[bool]) -> Result<Self, CurvaturePairedSignalError> {
                let mut p = self.clone();
                p.calibration = p.calibration.with_support(v)?;
                Ok(p)
            }
            /// Replace the finite row-major full-grid reference.
            pub fn with_reference(&self, v: &[f32]) -> Result<Self, CurvaturePairedSignalError> {
                let mut p = self.clone();
                p.calibration = p.calibration.with_reference(v)?;
                Ok(p)
            }
            /// Replace finite positive scales in positive/negative branch order.
            pub fn with_branch_scales(
                &self,
                v: &[f32],
            ) -> Result<Self, CurvaturePairedSignalError> {
                let mut p = self.clone();
                p.calibration = p.calibration.with_scales(v)?;
                Ok(p)
            }
            /// Replace the active calibration signature.
            pub fn with_calibration_signature(
                &self,
                v: &[u64],
            ) -> Result<Self, CurvaturePairedSignalError> {
                let mut p = self.clone();
                p.calibration = p.calibration.with_signature(v)?;
                Ok(p)
            }
        }
    };
}
calibration_replacements!(CurvaturePairedImagePlan);
calibration_replacements!(CurvaturePairedChannelPlan);
/// Disposable full-grid signal staging, shared by the two curvature geometries.
#[derive(Clone, Debug)]
pub struct CurvaturePairedSignalWorkspace {
    staged_signal: Box<[f32]>,
}
impl CurvaturePairedSignalWorkspace {
    /// Stateless reset has no mathematical history to change.
    pub fn reset(&mut self) {}
}
impl AlgorithmPlan for CurvaturePairedImagePlan {
    type Input<'a> = (&'a [f32], &'a [u64]);
    type Output<'a> = &'a mut [f32];
    type Workspace = CurvaturePairedSignalWorkspace;
    type Error = CurvaturePairedSignalError;
    fn process<'i, 'o>(
        &self,
        signal: Self::Output<'o>,
        w: &mut Self::Workspace,
        (image, signature): Self::Input<'i>,
    ) -> Result<(), Self::Error> {
        self.calibration.check(
            signal,
            w,
            image,
            self.image_size.0 * self.image_size.1,
            signature,
        )?;
        let (rows, columns) = self.signal_size();
        let tile_rows = self.branch_size.0 / rows;
        let tile_columns = self.branch_size.1 / columns;
        w.staged_signal.fill(0.0);
        for col in 0..columns {
            for row in 0..rows {
                let index = row + col * rows;
                let calibration_index = row * columns + col;
                if !self.calibration.support[calibration_index] {
                    continue;
                }
                let mut plus = 0.0;
                let mut minus = 0.0;
                for local_col in 0..tile_columns {
                    for local_row in 0..tile_rows {
                        let r = row * tile_rows + local_row;
                        let c = col * tile_columns + local_col;
                        let [pr, pc] = self.branch_origins[0];
                        let [mr, mc] = self.branch_origins[1];
                        plus += image[(pr + r) * self.image_size.1 + pc + c];
                        minus += image[(mr + r) * self.image_size.1 + mc + c];
                    }
                }
                let value = paired_value(
                    self.calibration.branch_scales[0] * plus,
                    self.calibration.branch_scales[1] * minus,
                    self.calibration.reference_signal[calibration_index],
                );
                if !value.is_finite() {
                    return Err(CurvaturePairedSignalError::NonFiniteOutput);
                }
                w.staged_signal[index] = value;
            }
        }
        signal.copy_from_slice(&w.staged_signal);
        Ok(())
    }
}
impl AlgorithmPlan for CurvaturePairedChannelPlan {
    type Input<'a> = (&'a [f32], &'a [u64]);
    type Output<'a> = &'a mut [f32];
    type Workspace = CurvaturePairedSignalWorkspace;
    type Error = CurvaturePairedSignalError;
    fn process<'i, 'o>(
        &self,
        signal: Self::Output<'o>,
        w: &mut Self::Workspace,
        (channels, signature): Self::Input<'i>,
    ) -> Result<(), Self::Error> {
        self.calibration.check(
            signal,
            w,
            channels,
            self.input_size.0 * self.input_size.1,
            signature,
        )?;
        let (rows, columns) = self.signal_size();
        w.staged_signal.fill(0.0);
        for (index, &channel) in self.channel_offsets.iter().enumerate() {
            let row = index % rows;
            let col = index / rows;
            let calibration_index = row * columns + col;
            if !self.calibration.support[calibration_index] {
                continue;
            }
            let plus = self.calibration.branch_scales[0]
                * channels[self.branch_rows[0] * self.input_size.1 + channel];
            let minus = self.calibration.branch_scales[1]
                * channels[self.branch_rows[1] * self.input_size.1 + channel];
            let value = paired_value(
                plus,
                minus,
                self.calibration.reference_signal[calibration_index],
            );
            if !value.is_finite() {
                return Err(CurvaturePairedSignalError::NonFiniteOutput);
            }
            w.staged_signal[index] = value;
        }
        signal.copy_from_slice(&w.staged_signal);
        Ok(())
    }
}
crate::filtergraph_algorithm! {
    /// Paired defocus image reduced to full-grid curvature signal.
    pub CurvaturePairedImageF32 {
        label="curvature-paired-image-f32";
        configuration CurvaturePairedImageF32Configuration {
            image_size:(usize,usize)=GraphRebuild;signal_size:(usize,usize)=GraphRebuild;branch_size:(usize,usize)=GraphRebuild;
            positive_branch_origin:[usize;2]=GraphRebuild;negative_branch_origin:[usize;2]=GraphRebuild;
            curvature_support:Option<Vec<bool>>=Construction;initial_reference_signal:Option<Vec<f32>>=Construction;
            initial_branch_scales:Option<Vec<f32>>=Construction;initial_calibration_signature:u64=Construction;
        }
        plan=CurvaturePairedImagePlan;workspace=CurvaturePairedSignalWorkspace;properties=none;
        dimensions|plan,_workspace|{input_rows=plan.image_size().0;input_columns=plan.image_size().1;rows=plan.signal_size().0;columns=plan.signal_size().1;count=rows*columns;}
        inputs{image:f32[input_rows,input_columns]="image",schema=crate::schemas::CURVATURE_PAIRED_IMAGE_V1;
            frame_calibration_signature:u64[1]="frame-calibration-signature",schema=crate::schemas::CURVATURE_PAIRED_FRAME_CALIBRATION_SIGNATURE_V1;}
        outputs{signal:f32[count]="signal",schema=crate::schemas::CURVATURE_PAIRED_SIGNAL_V1,metadata_from="image";}
        parameters{
            curvature_support:bool[rows,columns]="curvature-support",schema=crate::schemas::CURVATURE_PAIRED_SUPPORT_V1,replace|plan,values|{plan.with_support(values)};
            reference_signal:f32[rows,columns]="reference-signal",schema=crate::schemas::CURVATURE_PAIRED_REFERENCE_SIGNAL_V1,replace|plan,values|{plan.with_reference(values)};
            branch_scales:f32[2]="branch-scales",schema=crate::schemas::CURVATURE_PAIRED_BRANCH_SCALES_V1,replace|plan,values|{plan.with_branch_scales(values)};
            calibration_signature:u64[1]="calibration-signature",schema=crate::schemas::CURVATURE_PAIRED_CALIBRATION_SIGNATURE_V1,replace|plan,values|{plan.with_calibration_signature(values)};
        }
        parameter_error=CurvaturePairedSignalError;
        prepare|configuration|{let plan=CurvaturePairedImagePlan::new(configuration)?;let workspace=plan.workspace();Ok((plan,workspace))}
        prepare_error=CurvaturePairedSignalError;
        process|plan,workspace|{plan.process(signal,workspace,(image,frame_calibration_signature))}
        process_error=CurvaturePairedSignalError;
        reset|_plan,workspace|{workspace.reset();}
    }
}
crate::filtergraph_algorithm! {
    /// Paired branch-by-channel input mapped to full-grid curvature signal.
    pub CurvaturePairedChannelF32 {
        label="curvature-paired-channel-f32";
        configuration CurvaturePairedChannelF32Configuration {
            input_branch_rows:usize=GraphRebuild;input_channels:usize=GraphRebuild;signal_size:(usize,usize)=GraphRebuild;
            positive_branch_row:usize=GraphRebuild;negative_branch_row:usize=GraphRebuild;channel_offsets:Vec<usize>=GraphRebuild;
            curvature_support:Option<Vec<bool>>=Construction;initial_reference_signal:Option<Vec<f32>>=Construction;
            initial_branch_scales:Option<Vec<f32>>=Construction;initial_calibration_signature:u64=Construction;
        }
        plan=CurvaturePairedChannelPlan;workspace=CurvaturePairedSignalWorkspace;properties=none;
        dimensions|plan,_workspace|{input_rows=plan.input_size().0;input_columns=plan.input_size().1;rows=plan.signal_size().0;columns=plan.signal_size().1;count=rows*columns;}
        inputs{channels:f32[input_rows,input_columns]="channels",schema=crate::schemas::CURVATURE_PAIRED_CHANNELS_V1;
            frame_calibration_signature:u64[1]="frame-calibration-signature",schema=crate::schemas::CURVATURE_PAIRED_FRAME_CALIBRATION_SIGNATURE_V1;}
        outputs{signal:f32[count]="signal",schema=crate::schemas::CURVATURE_PAIRED_SIGNAL_V1,metadata_from="channels";}
        parameters{
            curvature_support:bool[rows,columns]="curvature-support",schema=crate::schemas::CURVATURE_PAIRED_SUPPORT_V1,replace|plan,values|{plan.with_support(values)};
            reference_signal:f32[rows,columns]="reference-signal",schema=crate::schemas::CURVATURE_PAIRED_REFERENCE_SIGNAL_V1,replace|plan,values|{plan.with_reference(values)};
            branch_scales:f32[2]="branch-scales",schema=crate::schemas::CURVATURE_PAIRED_BRANCH_SCALES_V1,replace|plan,values|{plan.with_branch_scales(values)};
            calibration_signature:u64[1]="calibration-signature",schema=crate::schemas::CURVATURE_PAIRED_CALIBRATION_SIGNATURE_V1,replace|plan,values|{plan.with_calibration_signature(values)};
        }
        parameter_error=CurvaturePairedSignalError;
        prepare|configuration|{let plan=CurvaturePairedChannelPlan::new(configuration)?;let workspace=plan.workspace();Ok((plan,workspace))}
        prepare_error=CurvaturePairedSignalError;
        process|plan,workspace|{plan.process(signal,workspace,(channels,frame_calibration_signature))}
        process_error=CurvaturePairedSignalError;
        reset|_plan,workspace|{workspace.reset();}
    }
}
