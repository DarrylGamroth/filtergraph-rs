//! Current-frame calibrated pyramid differential signals, distinct from previous-frame PWFS normalization.
//!
//! Pinned JFG orders q1 top-left, q2 bottom-left, q3 bottom-right, q4 top-right.
//! Support coordinates follow column-major order; packed signal rows contain X/Y.
use crate::AlgorithmPlan;
use alloc::{boxed::Box, string::String, vec, vec::Vec};
use core::fmt;

/// Current-frame pyramid normalization policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidImageNormalization {
    /// Mean selected four-pupil flux from the current frame.
    MeanValidFlux,
    /// Finite, strictly positive incidence calibration.
    IncidenceFlux,
}
impl PyramidImageNormalization {
    fn parse(policy: &str) -> Result<Self, PyramidImageError> {
        match policy {
            "mean-valid-flux" => Ok(Self::MeanValidFlux),
            "incidence-flux" => Ok(Self::IncidenceFlux),
            _ => Err(PyramidImageError::Configuration),
        }
    }
}
/// Invalid calibration/storage or rejected complete pyramid frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidImageError {
    /// Geometry, support, normalization policy, or finite calibration is invalid.
    Configuration,
    /// A complete input or output has an incompatible extent.
    PortExtent,
    /// Scratch was prepared for another signal extent.
    WorkspaceExtent,
    /// A configured pupil contains a nonfinite pixel, including outside support.
    NonFiniteInput,
    /// Current normalization is not finite and strictly positive.
    InvalidNormalization,
    /// A calibrated differential component is nonfinite.
    NonFiniteOutput,
}
impl fmt::Display for PyramidImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Configuration => "pyramid calibration or geometry is invalid",
            Self::PortExtent => "pyramid port extent is invalid",
            Self::WorkspaceExtent => "pyramid workspace extent is invalid",
            Self::NonFiniteInput => "pyramid pupil contains a nonfinite pixel",
            Self::InvalidNormalization => "pyramid normalization divisor is invalid",
            Self::NonFiniteOutput => "pyramid differential signal is nonfinite",
        })
    }
}
impl core::error::Error for PyramidImageError {}

pub(crate) fn area(size: (usize, usize)) -> Option<usize> {
    if size.0 == 0 || size.1 == 0 {
        return None;
    }
    size.0
        .checked_mul(size.1)
        .filter(|&n| n <= u32::MAX as usize / 4)
}
pub(crate) fn column_major_to_row_major<T: Copy>(
    values: &[T],
    size: (usize, usize),
) -> Option<Vec<T>> {
    let n = area(size)?;
    if values.len() != n {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for row in 0..size.0 {
        for col in 0..size.1 {
            out.push(values[row + col * size.0]);
        }
    }
    Some(out)
}
pub(crate) fn support(values: Option<Vec<bool>>, size: (usize, usize)) -> Option<Box<[bool]>> {
    let n = area(size)?;
    match values {
        Some(values) => Some(column_major_to_row_major(&values, size)?.into_boxed_slice()),
        None => Some(vec![true; n].into_boxed_slice()),
    }
}
pub(crate) fn selected_offsets(mask: &[bool], size: (usize, usize)) -> Vec<usize> {
    let mut selected = Vec::new();
    for col in 0..size.1 {
        for row in 0..size.0 {
            if mask[row * size.1 + col] {
                selected.push(row + col * size.0);
            }
        }
    }
    selected
}
pub(crate) fn finite(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}

/// Owned four-pupil calibration. Construction vectors retain JFG column-major meanings.
#[derive(Clone, Debug)]
pub struct PyramidImagePlan {
    image_size: (usize, usize),
    pupil_size: (usize, usize),
    origins: [[usize; 2]; 4],
    support: Box<[bool]>,
    selected_offsets: Box<[usize]>,
    reference_signal: Box<[f32]>,
    optical_gain: Box<[f32]>,
    normalization: PyramidImageNormalization,
    incidence_flux: f32,
}
impl PyramidImagePlan {
    /// Validate complete Float32 declaration calibration before frame allocation.
    pub fn new(c: PyramidImageF32Configuration) -> Result<Self, PyramidImageError> {
        use PyramidImageError::Configuration as E;
        area(c.image_size).ok_or(E)?;
        area(c.pupil_size).ok_or(E)?;
        let origins: [[usize; 2]; 4] = c.pupil_origins.try_into().map_err(|_| E)?;
        let mask = support(c.i4q_support, c.pupil_size).ok_or(E)?;
        let selected = selected_offsets(&mask, c.pupil_size);
        if selected.is_empty() || selected.len() > u32::MAX as usize / 8 {
            return Err(E);
        }
        let reference = match c.initial_reference_signal {
            Some(v) => column_major_to_row_major(&v, (selected.len(), 2)).ok_or(E)?,
            None => vec![0.0; 2 * selected.len()],
        };
        let gain = match c.initial_optical_gain {
            Some(v) => column_major_to_row_major(&v, (selected.len(), 2)).ok_or(E)?,
            None => vec![1.0; 2 * selected.len()],
        };
        let plan = Self {
            image_size: c.image_size,
            pupil_size: c.pupil_size,
            origins,
            support: mask,
            selected_offsets: selected.into_boxed_slice(),
            reference_signal: reference.into_boxed_slice(),
            optical_gain: gain.into_boxed_slice(),
            normalization: PyramidImageNormalization::parse(&c.normalization_policy)?,
            incidence_flux: c.incidence_flux,
        };
        plan.validate()?;
        Ok(plan)
    }
    fn validate(&self) -> Result<(), PyramidImageError> {
        use PyramidImageError::Configuration as E;
        if !finite(&self.reference_signal) || !finite(&self.optical_gain) {
            return Err(E);
        }
        if self.normalization == PyramidImageNormalization::IncidenceFlux
            && !(self.incidence_flux.is_finite() && self.incidence_flux > 0.0)
        {
            return Err(E);
        }
        for &[row, col] in &self.origins {
            if row
                .checked_add(self.pupil_size.0)
                .is_none_or(|v| v > self.image_size.0)
                || col
                    .checked_add(self.pupil_size.1)
                    .is_none_or(|v| v > self.image_size.1)
            {
                return Err(E);
            }
        }
        Ok(())
    }
    /// Detector image rows and columns.
    pub const fn image_size(&self) -> (usize, usize) {
        self.image_size
    }
    /// Pupil rows and columns.
    pub const fn pupil_size(&self) -> (usize, usize) {
        self.pupil_size
    }
    /// Selected coordinate count in column-major support order.
    pub fn selected_count(&self) -> usize {
        self.selected_offsets.len()
    }
    /// Prepare fixed signal staging and the reference's 256 reduction lanes.
    pub fn workspace(&self) -> PyramidImageWorkspace {
        PyramidImageWorkspace {
            staged_signal: vec![0.0; 2 * self.selected_count()].into_boxed_slice(),
            normalization_partials: Box::new([0.0; 256]),
        }
    }
    /// Replace zero-based origin pairs in logical row-major port order.
    pub fn with_origins(&self, values: &[u32]) -> Result<Self, PyramidImageError> {
        if values.len() != 8 {
            return Err(PyramidImageError::Configuration);
        }
        let mut plan = self.clone();
        for (origin, pair) in plan
            .origins
            .iter_mut()
            .zip(values.as_chunks::<2>().0.iter())
        {
            *origin = [pair[0] as usize, pair[1] as usize];
        }
        plan.validate()?;
        Ok(plan)
    }
    /// Replace a logical row-major pupil support without changing output count.
    pub fn with_support(&self, values: &[bool]) -> Result<Self, PyramidImageError> {
        if values.len() != self.support.len() {
            return Err(PyramidImageError::Configuration);
        }
        let selected = selected_offsets(values, self.pupil_size);
        if selected.len() != self.selected_count() {
            return Err(PyramidImageError::Configuration);
        }
        let mut plan = self.clone();
        plan.support = values.into();
        plan.selected_offsets = selected.into_boxed_slice();
        Ok(plan)
    }
    /// Replace logical selected-coordinate X/Y reference pairs.
    pub fn with_reference(&self, values: &[f32]) -> Result<Self, PyramidImageError> {
        if values.len() != self.reference_signal.len() || !finite(values) {
            return Err(PyramidImageError::Configuration);
        }
        let mut plan = self.clone();
        plan.reference_signal = values.into();
        Ok(plan)
    }
    /// Replace logical selected-coordinate X/Y gains.
    pub fn with_gain(&self, values: &[f32]) -> Result<Self, PyramidImageError> {
        if values.len() != self.optical_gain.len() || !finite(values) {
            return Err(PyramidImageError::Configuration);
        }
        let mut plan = self.clone();
        plan.optical_gain = values.into();
        Ok(plan)
    }
    fn pupil_values(&self, image: &[f32], offset: usize) -> [f32; 4] {
        let row = offset % self.pupil_size.0;
        let col = offset / self.pupil_size.0;
        self.origins
            .map(|[r, c]| image[(r + row) * self.image_size.1 + c + col])
    }
    pub(crate) fn check(
        &self,
        signal_len: usize,
        divisor_len: usize,
        w: &PyramidImageWorkspace,
        image_len: usize,
    ) -> Result<(), PyramidImageError> {
        if image_len != self.image_size.0 * self.image_size.1
            || signal_len != 2 * self.selected_count()
            || divisor_len != 1
        {
            return Err(PyramidImageError::PortExtent);
        }
        if w.staged_signal.len() != signal_len {
            return Err(PyramidImageError::WorkspaceExtent);
        }
        Ok(())
    }
    pub(crate) fn process_with_minimum(
        &self,
        (signal, divisor): (&mut [f32], &mut [f32]),
        w: &mut PyramidImageWorkspace,
        image: &[f32],
        minimum: f32,
    ) -> Result<(), PyramidImageError> {
        self.check(signal.len(), divisor.len(), w, image.len())?;
        let mut finite_input = true;
        for &[r, c] in &self.origins {
            for row in r..r + self.pupil_size.0 {
                let start = row * self.image_size.1 + c;
                for &value in &image[start..start + self.pupil_size.1] {
                    finite_input &= value.is_finite();
                }
            }
        }
        if !finite_input {
            return Err(PyramidImageError::NonFiniteInput);
        }
        let d = match self.normalization {
            PyramidImageNormalization::IncidenceFlux => self.incidence_flux,
            PyramidImageNormalization::MeanValidFlux => {
                w.normalization_partials.fill(0.0);
                // Preserve each reference reduction lane's selected-coordinate
                // order while traversing adjacent selected pixels together.
                for selected in self.selected_offsets.chunks(256) {
                    for (partial, &offset) in w.normalization_partials.iter_mut().zip(selected) {
                        let [a, b, c, d] = self.pupil_values(image, offset);
                        *partial += a + b + c + d;
                    }
                }
                let mut stride = 128;
                while stride > 0 {
                    for lane in 0..stride {
                        w.normalization_partials[lane] += w.normalization_partials[lane + stride];
                    }
                    stride /= 2;
                }
                w.normalization_partials[0] / self.selected_count() as f32
            }
        };
        if !(d.is_finite() && d > minimum) {
            return Err(PyramidImageError::InvalidNormalization);
        }
        let mut finite_signal = true;
        for (i, &offset) in self.selected_offsets.iter().enumerate() {
            let [a, b, c, e] = self.pupil_values(image, offset);
            let x = self.optical_gain[2 * i] * ((a - b + e - c) / d - self.reference_signal[2 * i]);
            let y = self.optical_gain[2 * i + 1]
                * ((a - e + b - c) / d - self.reference_signal[2 * i + 1]);
            finite_signal &= x.is_finite() & y.is_finite();
            w.staged_signal[2 * i] = x;
            w.staged_signal[2 * i + 1] = y;
        }
        if !finite_signal {
            return Err(PyramidImageError::NonFiniteOutput);
        }
        signal.copy_from_slice(&w.staged_signal);
        divisor[0] = d;
        Ok(())
    }
}
/// Disposable output staging and fixed normalization reduction lanes.
#[derive(Clone, Debug)]
pub struct PyramidImageWorkspace {
    staged_signal: Box<[f32]>,
    normalization_partials: Box<[f32; 256]>,
}
impl PyramidImageWorkspace {
    /// Stateless reset retains calibration and has no numerical effect.
    pub fn reset(&mut self) {}
}
impl AlgorithmPlan for PyramidImagePlan {
    type Input<'a> = &'a [f32];
    type Output<'a> = (&'a mut [f32], &'a mut [f32]);
    type Workspace = PyramidImageWorkspace;
    type Error = PyramidImageError;
    fn process<'i, 'o>(
        &self,
        output: Self::Output<'o>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'i>,
    ) -> Result<(), Self::Error> {
        self.process_with_minimum(output, workspace, input, 0.0)
    }
}
crate::filtergraph_algorithm! {
    /// Current-frame pyramid I4Q differential signal, with column-major support coordinates.
    pub PyramidImageF32 {
        label="pyramid-image-f32";
        configuration PyramidImageF32Configuration {
            image_size:(usize,usize)=GraphRebuild;pupil_size:(usize,usize)=GraphRebuild;
            pupil_origins:Vec<[usize;2]>=GraphRebuild;i4q_support:Option<Vec<bool>>=Construction;
            initial_reference_signal:Option<Vec<f32>>=Construction;initial_optical_gain:Option<Vec<f32>>=Construction;
            normalization_policy:String=GraphRebuild;incidence_flux:f32=Construction;
        }
        plan=PyramidImagePlan;workspace=PyramidImageWorkspace;properties=none;
        dimensions|plan,_workspace|{image_rows=plan.image_size().0;image_columns=plan.image_size().1;pupil_rows=plan.pupil_size().0;pupil_columns=plan.pupil_size().1;selected=plan.selected_count();}
        inputs{image:f32[image_rows,image_columns]="image",schema=crate::schemas::CALIBRATED_PIXELS_V1;}
        outputs{signal:f32[selected,2]="signal",schema=crate::schemas::PYRAMID_I4Q_SIGNAL_V1,metadata_from="image";
            normalization_divisor:f32[1]="normalization-divisor",schema=crate::schemas::PYRAMID_NORMALIZATION_DIVISOR_V1,metadata_from="image";}
        parameters{
            pupil_origins:u32[4,2]="pupil-origins",schema=crate::schemas::REGION_ORIGINS_V1,replace|plan,values|{plan.with_origins(values)};
            i4q_support:bool[pupil_rows,pupil_columns]="i4q-support",schema=crate::schemas::PYRAMID_I4Q_SUPPORT_V1,replace|plan,values|{plan.with_support(values)};
            reference_signal:f32[selected,2]="reference-signal",schema=crate::schemas::PYRAMID_REFERENCE_SIGNAL_V1,replace|plan,values|{plan.with_reference(values)};
            optical_gain:f32[selected,2]="optical-gain",schema=crate::schemas::PYRAMID_OPTICAL_GAIN_V1,replace|plan,values|{plan.with_gain(values)};
        }
        parameter_error=PyramidImageError;
        prepare|configuration|{let plan=PyramidImagePlan::new(configuration)?;let workspace=plan.workspace();Ok((plan,workspace))}
        prepare_error=PyramidImageError;
        process|plan,workspace|{plan.process((signal,normalization_divisor),workspace,image)}
        process_error=PyramidImageError;
        reset|_plan,workspace|{workspace.reset();}
    }
}
