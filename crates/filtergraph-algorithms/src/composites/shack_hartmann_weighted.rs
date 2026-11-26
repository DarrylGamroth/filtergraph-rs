//! Complete-image weighted Shack–Hartmann centroids.
use super::shack_hartmann_image::initial_subaperture_origins;
use crate::{
    AlgorithmPlan, NoWorkspace, ProcessError, ShackHartmannImagePlan,
    ShackHartmannImagePreparationError, WeightedCenterOfGravityPlan,
    WeightedCenterOfGravityPlanError,
};
use alloc::{boxed::Box, string::String, sync::Arc, vec, vec::Vec};
use core::{fmt, ops::Range};

/// Invalid image sensing calibration or row-major per-subaperture weights.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShackHartmannWeightedCentroidError {
    /// Ordinary Shack–Hartmann geometry or calibration preparation failed.
    Sensing(ShackHartmannImagePreparationError),
    /// Coordinates, reference, threshold, or weighted calibration is invalid.
    Centroid(WeightedCenterOfGravityPlanError),
    /// Weight bank does not contain one row per subaperture and pixel.
    WeightExtent,
    /// CPU SIMD width must be automatic or one of the supported lane counts.
    SimdWidth,
    /// Image, output, scratch, or requested range has an incompatible extent.
    Process(ProcessError),
}
impl fmt::Display for ShackHartmannWeightedCentroidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sensing(e) => e.fmt(f),
            Self::Centroid(e) => e.fmt(f),
            Self::WeightExtent => f.write_str(
                "Shack-Hartmann weight extent must be subaperture count times pixel count",
            ),
            Self::SimdWidth => f.write_str("Shack-Hartmann SIMD width must be 0, 4, 8, or 16"),
            Self::Process(e) => e.fmt(f),
        }
    }
}
impl core::error::Error for ShackHartmannWeightedCentroidError {}
impl From<ShackHartmannImagePreparationError> for ShackHartmannWeightedCentroidError {
    fn from(e: ShackHartmannImagePreparationError) -> Self {
        Self::Sensing(e)
    }
}
impl From<WeightedCenterOfGravityPlanError> for ShackHartmannWeightedCentroidError {
    fn from(e: WeightedCenterOfGravityPlanError) -> Self {
        Self::Centroid(e)
    }
}
impl From<ProcessError> for ShackHartmannWeightedCentroidError {
    fn from(e: ProcessError) -> Self {
        Self::Process(e)
    }
}

/// Weighted complete-image centroid measurement for independent subapertures.
///
/// The inclusive pixel threshold applies to original calibrated intensity.
/// Unnormalized nonnegative weights multiply selected pixels before Float32
/// flux/moment accumulation. Nonfinite pixels invalidate even at zero weight.
/// Inactive subapertures publish zero slopes while retaining measured validity
/// and weighted flux. Slopes are pair-interleaved X/Y, relative to reference.
#[derive(Clone, Debug)]
pub struct ShackHartmannWeightedCentroidPlan {
    sensing: ShackHartmannImagePlan<f32>,
    centroids: Box<[WeightedCenterOfGravityPlan]>,
}
impl ShackHartmannWeightedCentroidPlan {
    /// Validate a logical row-major `[subaperture,pixel]` weight bank.
    pub fn new(
        sensing: ShackHartmannImagePlan<f32>,
        weights: impl Into<Arc<[f32]>>,
    ) -> Result<Self, ShackHartmannWeightedCentroidError> {
        let m = sensing.measurement();
        let n = m.len();
        let pixels = m.pixel_count();
        let weights = weights.into();
        if n.checked_mul(pixels) != Some(weights.len()) {
            return Err(ShackHartmannWeightedCentroidError::WeightExtent);
        }
        let x: Arc<[f32]> = Arc::from(m.x_coordinates());
        let y: Arc<[f32]> = Arc::from(m.y_coordinates());
        let mut centroids = Vec::with_capacity(n);
        for (subaperture, weights) in weights.chunks_exact(pixels).enumerate() {
            centroids.push(WeightedCenterOfGravityPlan::new(
                Arc::clone(&x),
                Arc::clone(&y),
                Arc::from(weights),
                [
                    m.reference_slopes()[2 * subaperture],
                    m.reference_slopes()[2 * subaperture + 1],
                ],
                Some(m.pixel_thresholds()[subaperture]),
                m.flux_thresholds()[subaperture],
            )?);
        }
        Ok(Self {
            sensing,
            centroids: centroids.into_boxed_slice(),
        })
    }
    /// Borrow logical image geometry and common sensing calibration.
    pub const fn sensing(&self) -> &ShackHartmannImagePlan<f32> {
        &self.sensing
    }
    /// Borrow one row of dimensionless per-pixel weights.
    pub fn weights(&self, subaperture: usize) -> Option<&[f32]> {
        self.centroids
            .get(subaperture)
            .map(WeightedCenterOfGravityPlan::weights)
    }
    fn weight_bank(&self) -> Vec<f32> {
        self.centroids
            .iter()
            .flat_map(|plan| plan.weights().iter().copied())
            .collect()
    }
    /// Copy and validate an equal-extent replacement weight bank.
    pub fn with_weights(
        &self,
        weights: &[f32],
    ) -> Result<Self, ShackHartmannWeightedCentroidError> {
        Self::new(self.sensing.clone(), Arc::from(weights))
    }
    /// Replace zero-based origins without changing scientific output order.
    pub fn with_u32_subaperture_origins(
        &self,
        values: &[u32],
    ) -> Result<Self, ShackHartmannWeightedCentroidError> {
        Self::new(
            self.sensing.with_u32_subaperture_origins(values)?,
            self.weight_bank(),
        )
    }
    /// Replace pair-interleaved detector coordinates shared by subapertures.
    pub fn with_pair_interleaved_coordinates(
        &self,
        values: &[f32],
    ) -> Result<Self, ShackHartmannWeightedCentroidError> {
        Self::new(
            self.sensing.with_pair_interleaved_coordinates(values)?,
            self.weight_bank(),
        )
    }
    /// Replace pair-interleaved reference X/Y slopes.
    pub fn with_reference_slopes(
        &self,
        values: &[f32],
    ) -> Result<Self, ShackHartmannWeightedCentroidError> {
        Self::new(
            self.sensing.with_reference_slopes(values)?,
            self.weight_bank(),
        )
    }
    /// Replace pair-interleaved inclusive pixel and positive-flux thresholds.
    pub fn with_pair_interleaved_thresholds(
        &self,
        values: &[f32],
    ) -> Result<Self, ShackHartmannWeightedCentroidError> {
        Self::new(
            self.sensing.with_pair_interleaved_thresholds(values)?,
            self.weight_bank(),
        )
    }
    /// Replace the active mask; validity remains a measured property.
    pub fn with_active(&self, values: &[bool]) -> Result<Self, ShackHartmannWeightedCentroidError> {
        Self::new(self.sensing.with_active(values)?, self.weight_bank())
    }
    /// Allocate one fixed selected-region pixel buffer, reused across lenslets.
    pub fn workspace(&self) -> ShackHartmannWeightedCentroidWorkspace {
        ShackHartmannWeightedCentroidWorkspace {
            pixels: vec![0.0; self.sensing.measurement().pixel_count()].into_boxed_slice(),
        }
    }
    pub(crate) fn process_range(
        &self,
        (slopes, flux, validity): (&mut [f32], &mut [f32], &mut [bool]),
        workspace: &mut ShackHartmannWeightedCentroidWorkspace,
        image: &[f32],
        range: Range<usize>,
    ) -> Result<(), ShackHartmannWeightedCentroidError> {
        let n = self.centroids.len();
        let pixels = self.sensing.measurement().pixel_count();
        for (actual, expected) in [
            (slopes.len(), 2 * n),
            (flux.len(), n),
            (validity.len(), n),
            (workspace.pixels.len(), pixels),
        ] {
            if actual != expected {
                return Err(ProcessError::OutputExtent { expected, actual }.into());
            }
        }
        if range.start > range.end || range.end > n {
            return Err(ProcessError::RangeOutOfBounds {
                end: range.end,
                extent: n,
            }
            .into());
        }
        let regions = self.sensing.regions().image_regions(image)?;
        let rows = regions.region_size().0;
        let columns = regions.region_size().1;
        for subaperture in range {
            for row in 0..rows {
                workspace.pixels[row * columns..(row + 1) * columns]
                    .copy_from_slice(regions.row(subaperture, row));
            }
            let mut offset = [0.0; 2];
            let mut measured_flux = 0.0;
            let mut valid = false;
            self.centroids[subaperture].process(
                (&mut offset, &mut measured_flux, &mut valid),
                &mut NoWorkspace,
                &workspace.pixels,
            )?;
            if valid && self.sensing.measurement().active()[subaperture] {
                slopes[2 * subaperture] = offset[0];
                slopes[2 * subaperture + 1] = offset[1];
            } else {
                slopes[2 * subaperture] = 0.0;
                slopes[2 * subaperture + 1] = 0.0;
            }
            flux[subaperture] = measured_flux;
            validity[subaperture] = valid;
        }
        Ok(())
    }
}
/// Disposable selected-region scratch; this measurement has no temporal history.
#[derive(Clone, Debug, PartialEq)]
pub struct ShackHartmannWeightedCentroidWorkspace {
    pixels: Box<[f32]>,
}
impl ShackHartmannWeightedCentroidWorkspace {
    pub(crate) fn pixel_count(&self) -> usize {
        self.pixels.len()
    }
    /// Clear scratch without changing adopted sensing calibration or weights.
    pub fn reset(&mut self) {
        self.pixels.fill(0.0);
    }
}
impl AlgorithmPlan for ShackHartmannWeightedCentroidPlan {
    type Input<'a> = &'a [f32];
    type Output<'a> = (&'a mut [f32], &'a mut [f32], &'a mut [bool]);
    type Workspace = ShackHartmannWeightedCentroidWorkspace;
    type Error = ShackHartmannWeightedCentroidError;
    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        self.process_range(output, workspace, input, 0..self.centroids.len())
    }
}

crate::filtergraph_algorithm! {
    /// Complete-image weighted centroids, weighted flux, and measured validity.
    pub ShackHartmannWeightedCentroidF32{
        label = "shack-hartmann-weighted-centroid-f32";
        configuration ShackHartmannWeightedCentroidF32Configuration{
            image_size: (usize, usize) = GraphRebuild;
            image_schema: String = GraphRebuild;
            subaperture_size: (usize, usize) = GraphRebuild;
            subaperture_count: usize = GraphRebuild;
            initial_subaperture_origins: Option<Vec<[usize;2]>> = Construction;
            coordinate_scale: f32 = Construction;
            pixel_threshold: f32 = Construction;
            flux_threshold: f32 = Construction;
            reference_slopes: Option<Vec<f32>> = Construction;
            active: Option<Vec<bool>> = Construction;
            initial_weights: Vec<Vec<f32>> = Construction;
            simd_width: usize = Construction, default = 0;
        }
        plan = ShackHartmannWeightedCentroidPlan;
        workspace = ShackHartmannWeightedCentroidWorkspace;
        properties = none;
        dimensions |plan,_workspace|{
            image_rows=plan.sensing().regions().image_size().0;image_columns=plan.sensing().regions().image_size().1;
            subaperture_count=plan.sensing().measurement().len();pixel_count=plan.sensing().measurement().pixel_count();
        }
        inputs{image:f32[image_rows,image_columns]="image",schema=configuration.image_schema.as_str();}
        outputs{
            slopes:f32[subaperture_count,2]="slopes",schema=crate::schemas::SHACK_HARTMANN_SLOPES_V1,metadata_from="image";
            flux:f32[subaperture_count]="flux",schema=crate::schemas::WEIGHTED_SHACK_HARTMANN_FLUX_V1,metadata_from="image";
            validity:bool[subaperture_count]="validity",schema=crate::schemas::SHACK_HARTMANN_VALIDITY_V1,metadata_from="image";
        }
        parameters{
            subaperture_origins:u32[subaperture_count,2]="subaperture-origins",schema=crate::schemas::REGION_ORIGINS_V1,replace|plan,values|{plan.with_u32_subaperture_origins(values)};
            coordinates:f32[pixel_count,2]="coordinates",schema=crate::schemas::SHACK_HARTMANN_COORDINATES_V1,replace|plan,values|{plan.with_pair_interleaved_coordinates(values)};
            reference_slopes:f32[subaperture_count,2]="reference-slopes",schema=crate::schemas::SHACK_HARTMANN_REFERENCE_SLOPES_V1,replace|plan,values|{plan.with_reference_slopes(values)};
            thresholds:f32[subaperture_count,2]="thresholds",schema=crate::schemas::SHACK_HARTMANN_THRESHOLDS_V1,replace|plan,values|{plan.with_pair_interleaved_thresholds(values)};
            active:bool[subaperture_count]="active",schema=crate::schemas::SHACK_HARTMANN_ACTIVE_V1,replace|plan,values|{plan.with_active(values)};
            weights:f32[subaperture_count,pixel_count]="weights",schema=crate::schemas::SHACK_HARTMANN_WEIGHTS_V1,replace|plan,values|{plan.with_weights(values)};
        }
        parameter_error=ShackHartmannWeightedCentroidError;
        prepare|configuration|{
            if !matches!(configuration.simd_width, 0 | 4 | 8 | 16) { return Err(ShackHartmannWeightedCentroidError::SimdWidth); }
            let origins=initial_subaperture_origins(configuration.image_size,configuration.subaperture_size,configuration.subaperture_count,configuration.initial_subaperture_origins)?;
            let sensing=ShackHartmannImagePlan::from_detector_regions(configuration.image_size,configuration.subaperture_size,&origins,configuration.coordinate_scale,configuration.pixel_threshold,configuration.flux_threshold,configuration.reference_slopes,configuration.active)?;
            let pixels=sensing.measurement().pixel_count();
            if configuration.initial_weights.len()!=sensing.measurement().len() || configuration.initial_weights.iter().any(|weights|weights.len()!=pixels){return Err(ShackHartmannWeightedCentroidError::WeightExtent);}
            let weights=configuration.initial_weights.into_iter().flatten().collect::<Vec<f32>>();
            let plan=ShackHartmannWeightedCentroidPlan::new(sensing,weights)?;let workspace=plan.workspace();Ok((plan,workspace))
        }
        prepare_error=ShackHartmannWeightedCentroidError;
        process|plan,workspace|{plan.process((slopes,flux,validity),workspace,image)}
        process_error=ShackHartmannWeightedCentroidError;
        reset|_plan,workspace|{workspace.reset();}
    }
}
