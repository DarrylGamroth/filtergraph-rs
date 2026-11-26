//! Mask-aware SCC carrier demodulation. Detector vectors retain JFG column-major
//! geometry; measurements contain all selected real parts, then imaginary parts.
use crate::AlgorithmPlan;
use alloc::{vec, vec::Vec};
use core::fmt;
use num_traits::Float;

/// Invalid detector geometry, calibration, FIR, or support policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccSpatialCarrierDemodulationPlanError {
    /// Positive detector and measurement byte extents must fit UInt32.
    ImageExtent,
    /// A calibration plane has an incompatible detector extent.
    CalibrationExtent,
    /// A reference or phase value is nonfinite.
    NonFiniteCalibration,
    /// FIR taps must be finite, positive odd in count, and have a finite nonzero sum.
    Fir,
    /// The ROI must select a pixel.
    EmptyRoi,
    /// Minimum support must be finite and positive.
    MinimumSupport,
    /// Minimum valid fraction must be finite and in (0,1].
    MinimumValidFraction,
    /// Replacement ROI must retain the measurement extent.
    MeasurementExtent,
}
impl fmt::Display for SccSpatialCarrierDemodulationPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid SCC spatial-carrier preparation: {self:?}")
    }
}
impl core::error::Error for SccSpatialCarrierDemodulationPlanError {}
/// Complete-frame demodulation disposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccSpatialCarrierDemodulationError {
    /// Caller or workspace storage has incompatible extents.
    DimensionMismatch,
    /// The chopped pair is priming.
    IncompletePair,
    /// The pair status is undefined.
    InvalidPairStatus,
    /// Too few selected pixels have valid carrier support.
    InsufficientSupport,
}
impl fmt::Display for SccSpatialCarrierDemodulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SCC spatial-carrier disposition: {self:?}")
    }
}
impl core::error::Error for SccSpatialCarrierDemodulationError {}
/// Coherent immutable reference, static validity, carrier, FIR, and ROI.
#[derive(Clone, Debug)]
pub struct SccSpatialCarrierDemodulationPlan {
    image_size: (usize, usize),
    reference_difference: Vec<f32>,
    static_validity: Vec<bool>,
    roi: Vec<bool>,
    carrier_phase: Vec<f32>,
    carrier_conjugate: Vec<[f32; 2]>,
    fir: Vec<f32>,
    selected_indices: Vec<usize>,
    minimum_support: f32,
    minimum_valid_fraction: f32,
    minimum_valid_count: usize,
}
fn plane(
    values: Option<&[f32]>,
    count: usize,
    default: f32,
) -> Result<Vec<f32>, SccSpatialCarrierDemodulationPlanError> {
    let values = values.map_or_else(|| vec![default; count], |v| v.to_vec());
    if values.len() != count {
        return Err(SccSpatialCarrierDemodulationPlanError::CalibrationExtent);
    }
    if values.iter().any(|x| !x.is_finite()) {
        return Err(SccSpatialCarrierDemodulationPlanError::NonFiniteCalibration);
    }
    Ok(values)
}
impl SccSpatialCarrierDemodulationPlan {
    /// Prepare literal column-major detector vectors and a normalized separable FIR.
    #[expect(
        clippy::too_many_arguments,
        reason = "Retains the scientific calibration arguments; the declaration supplies structured configuration"
    )]
    pub fn new(
        image_size: (usize, usize),
        roi: &[bool],
        fir_tap_count: usize,
        reference_difference: Option<&[f32]>,
        static_validity: Option<&[bool]>,
        carrier_phase: Option<&[f32]>,
        fir: Option<&[f32]>,
        minimum_support: f32,
        minimum_valid_fraction: f32,
    ) -> Result<Self, SccSpatialCarrierDemodulationPlanError> {
        use SccSpatialCarrierDemodulationPlanError as E;
        let count = image_size
            .0
            .checked_mul(image_size.1)
            .filter(|&n| image_size.0 > 0 && image_size.1 > 0 && n <= u32::MAX as usize / 4)
            .ok_or(E::ImageExtent)?;
        if roi.len() != count {
            return Err(E::CalibrationExtent);
        }
        let selected_count = roi.iter().filter(|&&selected| selected).count();
        if selected_count == 0 {
            return Err(E::EmptyRoi);
        }
        if selected_count > u32::MAX as usize / 8 {
            return Err(E::ImageExtent);
        }
        let selected_indices: Vec<usize> = roi
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| v.then_some(i))
            .collect();
        if fir_tap_count == 0
            || fir_tap_count.is_multiple_of(2)
            || fir_tap_count > u32::MAX as usize / 4
        {
            return Err(E::Fir);
        }
        let reference_difference = plane(reference_difference, count, 0.0)?;
        let static_validity = static_validity.map_or_else(|| vec![true; count], |v| v.to_vec());
        if static_validity.len() != count {
            return Err(E::CalibrationExtent);
        }
        let carrier_phase = plane(carrier_phase, count, 0.0)?;
        let carrier_conjugate = carrier_phase
            .iter()
            .map(|&phase| [Float::cos(-phase), Float::sin(-phase)])
            .collect();
        let mut fir = fir.map_or_else(
            || {
                let mut v = vec![0.0; fir_tap_count];
                v[fir_tap_count / 2] = 1.0;
                v
            },
            |v| v.to_vec(),
        );
        if fir.len() != fir_tap_count || fir.iter().any(|x| !x.is_finite()) {
            return Err(E::Fir);
        }
        let normalization = fir.iter().copied().fold(0.0_f32, |a, b| a + b);
        if !normalization.is_finite() || normalization == 0.0 {
            return Err(E::Fir);
        }
        for coefficient in &mut fir {
            *coefficient /= normalization;
        }
        if !minimum_support.is_finite() || minimum_support <= 0.0 {
            return Err(E::MinimumSupport);
        }
        if !minimum_valid_fraction.is_finite()
            || minimum_valid_fraction <= 0.0
            || minimum_valid_fraction > 1.0
        {
            return Err(E::MinimumValidFraction);
        }
        let minimum_valid_count =
            (minimum_valid_fraction * selected_indices.len() as f32).ceil() as usize;
        Ok(Self {
            image_size,
            reference_difference,
            static_validity,
            roi: roi.to_vec(),
            carrier_phase,
            carrier_conjugate,
            fir,
            selected_indices,
            minimum_support,
            minimum_valid_fraction,
            minimum_valid_count,
        })
    }
    /// Detector-vector extent.
    pub fn pixel_count(&self) -> usize {
        self.reference_difference.len()
    }
    /// Number of complex selected pixels.
    pub fn selected_count(&self) -> usize {
        self.selected_indices.len()
    }
    /// Number of real measurement coordinates, including the imaginary block.
    pub fn measurement_count(&self) -> usize {
        2 * self.selected_count()
    }
    /// Smallest accepted number of supported complex pixels.
    pub const fn minimum_valid_count(&self) -> usize {
        self.minimum_valid_count
    }
    /// Borrow normalized separable FIR taps.
    pub fn fir(&self) -> &[f32] {
        &self.fir
    }
    /// Allocate all frame and candidate-output scratch once.
    pub fn workspace(&self) -> SccSpatialCarrierDemodulationWorkspace {
        let n = self.pixel_count();
        let m = self.measurement_count();
        SccSpatialCarrierDemodulationWorkspace {
            masked_carrier: vec![[0.0; 2]; n],
            horizontal_carrier: vec![[0.0; 2]; n],
            mask: vec![0.0; n],
            horizontal_support: vec![0.0; n],
            candidate_measurements: vec![0.0; m],
            candidate_weights: vec![0.0; m],
            candidate_validity: vec![false; m],
        }
    }
    fn replace(
        &self,
        reference: Option<&[f32]>,
        validity: Option<&[bool]>,
        phase: Option<&[f32]>,
        fir: Option<&[f32]>,
        roi: Option<&[bool]>,
    ) -> Result<Self, SccSpatialCarrierDemodulationPlanError> {
        let plan = Self::new(
            self.image_size,
            roi.unwrap_or(&self.roi),
            self.fir.len(),
            Some(reference.unwrap_or(&self.reference_difference)),
            Some(validity.unwrap_or(&self.static_validity)),
            Some(phase.unwrap_or(&self.carrier_phase)),
            Some(fir.unwrap_or(&self.fir)),
            self.minimum_support,
            self.minimum_valid_fraction,
        )?;
        if plan.selected_count() != self.selected_count() {
            return Err(SccSpatialCarrierDemodulationPlanError::MeasurementExtent);
        }
        Ok(plan)
    }
    /// Replace finite detector reference samples.
    pub fn with_reference_difference(
        &self,
        values: &[f32],
    ) -> Result<Self, SccSpatialCarrierDemodulationPlanError> {
        self.replace(Some(values), None, None, None, None)
    }
    /// Replace static detector validity.
    pub fn with_static_validity(
        &self,
        values: &[bool],
    ) -> Result<Self, SccSpatialCarrierDemodulationPlanError> {
        self.replace(None, Some(values), None, None, None)
    }
    /// Replace finite carrier phase in radians.
    pub fn with_carrier_phase(
        &self,
        values: &[f32],
    ) -> Result<Self, SccSpatialCarrierDemodulationPlanError> {
        self.replace(None, None, Some(values), None, None)
    }
    /// Replace and normalize the fixed-count FIR.
    pub fn with_fir(&self, values: &[f32]) -> Result<Self, SccSpatialCarrierDemodulationPlanError> {
        self.replace(None, None, None, Some(values), None)
    }
    /// Replace ROI without changing the complex measurement extent.
    pub fn with_roi(
        &self,
        values: &[bool],
    ) -> Result<Self, SccSpatialCarrierDemodulationPlanError> {
        self.replace(None, None, None, None, Some(values))
    }
}
/// Fixed disposable carrier, support, and complete-output staging.
#[derive(Clone, Debug, PartialEq)]
pub struct SccSpatialCarrierDemodulationWorkspace {
    masked_carrier: Vec<[f32; 2]>,
    horizontal_carrier: Vec<[f32; 2]>,
    mask: Vec<f32>,
    horizontal_support: Vec<f32>,
    candidate_measurements: Vec<f32>,
    candidate_weights: Vec<f32>,
    candidate_validity: Vec<bool>,
}
impl SccSpatialCarrierDemodulationWorkspace {
    /// Clear disposable scratch while retaining adopted calibration.
    pub fn reset(&mut self) {
        self.masked_carrier.fill([0.0; 2]);
        self.horizontal_carrier.fill([0.0; 2]);
        self.mask.fill(0.0);
        self.horizontal_support.fill(0.0);
        self.candidate_measurements.fill(0.0);
        self.candidate_weights.fill(0.0);
        self.candidate_validity.fill(false);
    }
}
impl AlgorithmPlan for SccSpatialCarrierDemodulationPlan {
    type Input<'a> = (&'a [f32], &'a [bool], u8);
    type Output<'a> = (&'a mut [f32], &'a mut [f32], &'a mut [bool], &'a mut f32);
    type Workspace = SccSpatialCarrierDemodulationWorkspace;
    type Error = SccSpatialCarrierDemodulationError;
    fn process<'i, 'o>(
        &self,
        (measurements, weights, validity, fraction): Self::Output<'o>,
        w: &mut Self::Workspace,
        (difference, difference_validity, status): Self::Input<'i>,
    ) -> Result<(), Self::Error> {
        use SccSpatialCarrierDemodulationError as E;
        let n = self.pixel_count();
        let m = self.measurement_count();
        if [
            difference.len(),
            difference_validity.len(),
            w.masked_carrier.len(),
            w.horizontal_carrier.len(),
            w.mask.len(),
            w.horizontal_support.len(),
        ]
        .iter()
        .any(|&x| x != n)
            || [
                measurements.len(),
                weights.len(),
                validity.len(),
                w.candidate_measurements.len(),
                w.candidate_weights.len(),
                w.candidate_validity.len(),
            ]
            .iter()
            .any(|&x| x != m)
        {
            return Err(E::DimensionMismatch);
        }
        match status {
            0 => return Err(E::IncompletePair),
            1 => {}
            _ => return Err(E::InvalidPairStatus),
        }
        for i in 0..n {
            let mut valid = self.static_validity[i] && difference_validity[i];
            let mut value = [0.0; 2];
            if valid {
                let residual = difference[i] - self.reference_difference[i];
                valid = residual.is_finite();
                if valid {
                    value = [
                        residual * self.carrier_conjugate[i][0],
                        residual * self.carrier_conjugate[i][1],
                    ];
                }
            }
            w.mask[i] = if valid { 1.0 } else { 0.0 };
            w.masked_carrier[i] = if valid { value } else { [0.0; 2] };
        }
        let (rows, columns) = self.image_size;
        let radius = self.fir.len() / 2;
        for target in 0..n {
            let row = target % rows;
            let column = target / rows;
            let mut carrier = [0.0_f32; 2];
            let mut support = 0.0;
            for (tap, &coefficient) in self.fir.iter().enumerate() {
                let source_column = column
                    .saturating_add(tap)
                    .saturating_sub(radius)
                    .min(columns - 1);
                let source = row + source_column * rows;
                carrier[0] += coefficient * w.masked_carrier[source][0];
                carrier[1] += coefficient * w.masked_carrier[source][1];
                support += coefficient * w.mask[source];
            }
            w.horizontal_carrier[target] = carrier;
            w.horizontal_support[target] = support;
        }
        let selected_count = self.selected_count();
        let mut valid_count = 0;
        for (selected, &pixel) in self.selected_indices.iter().enumerate() {
            let row = pixel % rows;
            let column = pixel / rows;
            let mut carrier = [0.0_f32; 2];
            let mut support = 0.0;
            for (tap, &coefficient) in self.fir.iter().enumerate() {
                let source_row = row.saturating_add(tap).saturating_sub(radius).min(rows - 1);
                let source = source_row + column * rows;
                carrier[0] += coefficient * w.horizontal_carrier[source][0];
                carrier[1] += coefficient * w.horizontal_carrier[source][1];
                support += coefficient * w.horizontal_support[source];
            }
            let mut valid = support.is_finite() && support >= self.minimum_support;
            let mut value = [0.0; 2];
            if valid {
                value = [carrier[0] / support, carrier[1] / support];
                valid = value.iter().all(|x| x.is_finite());
            }
            valid_count += usize::from(valid);
            for (component, &measurement) in value.iter().enumerate() {
                let index = selected + component * selected_count;
                w.candidate_measurements[index] = if valid { measurement } else { 0.0 };
                w.candidate_weights[index] = if valid { support.max(0.0) } else { 0.0 };
                w.candidate_validity[index] = valid;
            }
        }
        if valid_count < self.minimum_valid_count {
            return Err(E::InsufficientSupport);
        }
        measurements.copy_from_slice(&w.candidate_measurements);
        weights.copy_from_slice(&w.candidate_weights);
        validity.copy_from_slice(&w.candidate_validity);
        *fraction = valid_count as f32 / selected_count as f32;
        Ok(())
    }
}
crate::filtergraph_algorithm! {
    /// Mask-aware spatial-carrier measurement in selected real/imaginary block order.
    pub SccSpatialCarrierDemodulationF32 {
        label="scc-spatial-carrier-demodulation-f32";
        configuration SccSpatialCarrierDemodulationF32Configuration {
            image_size:(usize,usize)=GraphRebuild;roi:Vec<bool>=GraphRebuild;fir_tap_count:usize=GraphRebuild;
            initial_reference_difference:Option<Vec<f32>>=Construction;initial_static_validity:Option<Vec<bool>>=Construction;initial_carrier_phase:Option<Vec<f32>>=Construction;initial_fir:Option<Vec<f32>>=Construction;minimum_support:f32=Construction;minimum_valid_fraction:f32=Construction;
        }
        plan=SccSpatialCarrierDemodulationPlan;workspace=SccSpatialCarrierDemodulationWorkspace;properties=none;
        dimensions |plan,_workspace|{pixel_count=plan.pixel_count();measurement_count=plan.measurement_count();tap_count=plan.fir.len();}
        inputs {difference:f32 [pixel_count]="difference",schema=crate::schemas::SCC_CHOPPED_IMAGE_DIFFERENCE_V1;difference_validity:bool [pixel_count]="difference-validity",schema=crate::schemas::SCC_CHOPPED_DIFFERENCE_VALIDITY_V1;pair_status:u8 [1]="pair-status",schema=crate::schemas::SCC_CHOPPED_DIFFERENCE_STATUS_V1;}
        outputs {measurements:f32 [measurement_count]="measurements",schema=crate::schemas::SCC_SPATIAL_CARRIER_MEASUREMENTS_V1,metadata_from="difference";weights:f32 [measurement_count]="weights",schema=crate::schemas::SCC_MEASUREMENT_WEIGHTS_V1,metadata_from="difference";measurement_validity:bool [measurement_count]="measurement-validity",schema=crate::schemas::SCC_MEASUREMENT_VALIDITY_V1,metadata_from="difference";valid_fraction:f32 [1]="valid-fraction",schema=crate::schemas::SCC_SPATIAL_CARRIER_VALID_FRACTION_V1,metadata_from="difference";}
        parameters {
            reference_difference:f32 [pixel_count]="reference-difference",schema=crate::schemas::SCC_REFERENCE_IMAGE_V1,replace|plan,values|{plan.with_reference_difference(values)};
            static_validity:bool [pixel_count]="static-validity",schema=crate::schemas::CALIBRATED_PIXEL_VALIDITY_V1,replace|plan,values|{plan.with_static_validity(values)};
            carrier_phase:f32 [pixel_count]="carrier-phase",schema=crate::schemas::SCC_CARRIER_PHASE_V1,replace|plan,values|{plan.with_carrier_phase(values)};
            fir:f32 [tap_count]="fir",schema=crate::schemas::SCC_SPATIAL_FIR_V1,replace|plan,values|{plan.with_fir(values)};
            roi:bool [pixel_count]="roi",schema=crate::schemas::PIXEL_SELECTION_MASK_V1,replace|plan,values|{plan.with_roi(values)};
        }
        parameter_error=SccSpatialCarrierDemodulationPlanError;
        prepare|configuration|{let plan=SccSpatialCarrierDemodulationPlan::new(configuration.image_size,&configuration.roi,configuration.fir_tap_count,configuration.initial_reference_difference.as_deref(),configuration.initial_static_validity.as_deref(),configuration.initial_carrier_phase.as_deref(),configuration.initial_fir.as_deref(),configuration.minimum_support,configuration.minimum_valid_fraction)?;let workspace=plan.workspace();Ok((plan,workspace))}
        prepare_error=SccSpatialCarrierDemodulationPlanError;
        process|plan,workspace|{if pair_status.len()!=1||valid_fraction.len()!=1{return Err(SccSpatialCarrierDemodulationError::DimensionMismatch);}plan.process((measurements,weights,measurement_validity,&mut valid_fraction[0]),workspace,(difference,difference_validity,pair_status[0]))}
        process_error=SccSpatialCarrierDemodulationError;
        reset|_plan,workspace|{workspace.reset();}
    }
}
