//! Per-pixel dimensionless weighted center of gravity.
use super::center_of_gravity::{
    CenterOfGravityPlan, CenterOfGravityPlanError, finish_center_of_gravity,
};
use crate::{
    AlgorithmPlan, AlgorithmProperties, NoWorkspace, ProcessError, PropertyAssignment,
    PropertyInfo, PropertyValue,
};
use alloc::{sync::Arc, vec::Vec};
use core::fmt;
use num_traits::Float;

/// Invalid coordinates, properties, or per-pixel weighted calibration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WeightedCenterOfGravityPlanError {
    /// Center-of-gravity coordinate or property validation failed.
    CenterOfGravity(CenterOfGravityPlanError),
    /// Per-pixel weight extent differs from selected-pixel count.
    WeightExtent {
        /// Required selected-pixel count.
        expected: usize,
        /// Supplied weight count.
        actual: usize,
    },
    /// A dimensionless weight is nonfinite or negative.
    InvalidWeight {
        /// Zero-based selected-pixel coordinate.
        pixel: usize,
    },
}
impl fmt::Display for WeightedCenterOfGravityPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CenterOfGravity(error) => error.fmt(f),
            Self::WeightExtent { expected, actual } => write!(
                f,
                "weight extent {actual} does not match pixel count {expected}"
            ),
            Self::InvalidWeight { pixel } => {
                write!(f, "weight at pixel {pixel} must be finite and nonnegative")
            }
        }
    }
}
impl core::error::Error for WeightedCenterOfGravityPlanError {}
impl From<CenterOfGravityPlanError> for WeightedCenterOfGravityPlanError {
    fn from(error: CenterOfGravityPlanError) -> Self {
        Self::CenterOfGravity(error)
    }
}

/// Immutable Float32 weighted selected-region calibration.
///
/// The pixel threshold applies to the original calibrated value. Selected
/// values are multiplied by unnormalized weights with Float32 rounding before
/// accumulating flux and fused X/Y moments. Nonfinite pixels invalidate even
/// when their weight is zero.
#[derive(Clone, Debug)]
pub struct WeightedCenterOfGravityPlan {
    center_of_gravity: CenterOfGravityPlan<f32>,
    weights: Arc<[f32]>,
}
impl WeightedCenterOfGravityPlan {
    /// Validate finite coordinates/reference/thresholds and finite nonnegative weights.
    pub fn new(
        x_coordinates: impl Into<Arc<[f32]>>,
        y_coordinates: impl Into<Arc<[f32]>>,
        weights: impl Into<Arc<[f32]>>,
        reference_centroid: [f32; 2],
        pixel_threshold: Option<f32>,
        flux_threshold: f32,
    ) -> Result<Self, WeightedCenterOfGravityPlanError> {
        let center_of_gravity = CenterOfGravityPlan::new(
            x_coordinates,
            y_coordinates,
            reference_centroid,
            pixel_threshold,
            flux_threshold,
        )?;
        let weights = weights.into();
        validate_weights(&weights, center_of_gravity.len())?;
        Ok(Self {
            center_of_gravity,
            weights,
        })
    }
    /// Selected-pixel extent.
    pub fn len(&self) -> usize {
        self.center_of_gravity.len()
    }
    /// Valid plans always select at least one pixel.
    pub const fn is_empty(&self) -> bool {
        false
    }
    /// X coordinates in selected-pixel order.
    pub fn x_coordinates(&self) -> &[f32] {
        self.center_of_gravity.x_coordinates()
    }
    /// Y coordinates in selected-pixel order.
    pub fn y_coordinates(&self) -> &[f32] {
        self.center_of_gravity.y_coordinates()
    }
    /// Unnormalized dimensionless weights in selected-pixel order.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    /// Reference X/Y centroid in the coordinate unit.
    pub fn reference_centroid(&self) -> [f32; 2] {
        self.center_of_gravity.reference_centroid()
    }
    /// Inclusive threshold on the original calibrated pixels.
    pub fn pixel_threshold(&self) -> Option<f32> {
        self.center_of_gravity.pixel_threshold()
    }
    /// Minimum positive weighted flux required for validity.
    pub fn flux_threshold(&self) -> f32 {
        self.center_of_gravity.flux_threshold()
    }
    /// This operation retains no mathematical history or frame scratch.
    pub const fn workspace(&self) -> NoWorkspace {
        NoWorkspace
    }
    /// Prepare equal-extent pair-interleaved X/Y coordinate replacement.
    pub fn with_pair_interleaved_coordinates(
        &self,
        coordinates: &[f32],
    ) -> Result<Self, WeightedCenterOfGravityPlanError> {
        Ok(Self {
            center_of_gravity: self
                .center_of_gravity
                .with_pair_interleaved_coordinates(coordinates)?,
            weights: Arc::clone(&self.weights),
        })
    }
    /// Prepare a copied equal-extent nonnegative weight replacement.
    pub fn with_weights(&self, weights: &[f32]) -> Result<Self, WeightedCenterOfGravityPlanError> {
        validate_weights(weights, self.len())?;
        Ok(Self {
            center_of_gravity: self.center_of_gravity.clone(),
            weights: Arc::from(weights),
        })
    }
}
fn validate_weights(
    weights: &[f32],
    expected: usize,
) -> Result<(), WeightedCenterOfGravityPlanError> {
    if weights.len() != expected {
        return Err(WeightedCenterOfGravityPlanError::WeightExtent {
            expected,
            actual: weights.len(),
        });
    }
    if let Some(pixel) = weights
        .iter()
        .position(|&weight| !weight.is_finite() || weight < 0.0)
    {
        return Err(WeightedCenterOfGravityPlanError::InvalidWeight { pixel });
    }
    Ok(())
}
impl AlgorithmPlan for WeightedCenterOfGravityPlan {
    type Input<'a> = &'a [f32];
    type Output<'a> = (&'a mut [f32; 2], &'a mut f32, &'a mut bool);
    type Workspace = NoWorkspace;
    type Error = ProcessError;
    fn process<'input, 'output>(
        &self,
        (offset, flux, validity): Self::Output<'output>,
        _workspace: &mut Self::Workspace,
        pixels: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if pixels.len() != self.len() {
            return Err(ProcessError::InputExtent {
                expected: self.len(),
                actual: pixels.len(),
            });
        }
        let threshold = self.pixel_threshold();
        let mut x_moment = 0.0;
        let mut y_moment = 0.0;
        let mut measured_flux = 0.0;
        for (pixel, &value) in pixels.iter().enumerate() {
            if !value.is_finite() || threshold.is_none_or(|threshold| value >= threshold) {
                // Round weighting separately; only the moment product and sum
                // are fused. Nonfinite pixels participate even at zero weight.
                let weighted_pixel = self.weights[pixel] * value;
                x_moment = Float::mul_add(weighted_pixel, self.x_coordinates()[pixel], x_moment);
                y_moment = Float::mul_add(weighted_pixel, self.y_coordinates()[pixel], y_moment);
                measured_flux += weighted_pixel;
            }
        }
        let measurement = finish_center_of_gravity(
            x_moment,
            y_moment,
            measured_flux,
            self.reference_centroid(),
            self.flux_threshold(),
        );
        *offset = measurement.centroid_offset;
        *flux = measurement.flux;
        *validity = measurement.validity;
        Ok(())
    }
}
impl AlgorithmProperties for WeightedCenterOfGravityPlan {
    type PropertyError = <CenterOfGravityPlan<f32> as AlgorithmProperties>::PropertyError;
    fn property_info(&self, index: usize) -> Option<PropertyInfo<'static>> {
        self.center_of_gravity.property_info(index)
    }
    fn property(&self, id: u32) -> Result<PropertyValue<'_>, Self::PropertyError> {
        self.center_of_gravity.property(id)
    }
    fn prepare_properties(
        &self,
        assignments: &[PropertyAssignment<'_>],
    ) -> Result<Self, Self::PropertyError> {
        Ok(Self {
            center_of_gravity: self.center_of_gravity.prepare_properties(assignments)?,
            weights: Arc::clone(&self.weights),
        })
    }
}
crate::filtergraph_algorithm! {
    /// Weighted reference-relative centroid, weighted flux, and validity.
    pub WeightedCenterOfGravityF32 {
        label = "weighted-center-of-gravity-f32";
        configuration WeightedCenterOfGravityF32Configuration {
            pixel_count: usize = GraphRebuild;
            initial_coordinates: Vec<[f32;2]> = Construction;
            initial_weights: Vec<f32> = Construction;
        }
        plan = WeightedCenterOfGravityPlan;
        properties = runtime;
        dimensions |plan,_workspace| {pixel_count=plan.len();}
        inputs {pixels:f32 [pixel_count]="pixels",schema=crate::schemas::CENTER_OF_GRAVITY_PIXELS_V1;}
        outputs {
            centroid_offset:f32 [2]="centroid-offset",schema=crate::schemas::CENTER_OF_GRAVITY_OFFSET_V1,metadata_from="pixels";
            flux:f32 [1]="flux",schema=crate::schemas::WEIGHTED_CENTER_OF_GRAVITY_FLUX_V1,metadata_from="pixels";
            validity:bool [1]="validity",schema=crate::schemas::CENTER_OF_GRAVITY_VALIDITY_V1,metadata_from="pixels";
        }
        parameters {
            coordinates:f32 [pixel_count,2]="coordinates",schema=crate::schemas::CENTER_OF_GRAVITY_COORDINATES_V1,
                replace |plan,values| {plan.with_pair_interleaved_coordinates(values)};
            weights:f32 [pixel_count]="weights",schema=crate::schemas::CENTER_OF_GRAVITY_WEIGHTS_V1,
                replace |plan,values| {plan.with_weights(values)};
        }
        parameter_error = WeightedCenterOfGravityPlanError;
        prepare |configuration| {
            if configuration.initial_coordinates.len()!=configuration.pixel_count {return Err(WeightedCenterOfGravityPlanError::CenterOfGravity(CenterOfGravityPlanError::CoordinateCount{expected:configuration.pixel_count,actual:configuration.initial_coordinates.len()}));}
            let (x,y):(Vec<f32>,Vec<f32>)=configuration.initial_coordinates.into_iter().map(|[x,y]|(x,y)).unzip();
            Ok((WeightedCenterOfGravityPlan::new(x,y,configuration.initial_weights,[0.0;2],None,0.0)?,NoWorkspace))
        }
        prepare_error = WeightedCenterOfGravityPlanError;
        process |plan,workspace| {
            let actual=centroid_offset.len();
            let offset=<&mut [f32;2]>::try_from(centroid_offset).map_err(|_|ProcessError::OutputExtent{expected:2,actual})?;
            if flux.len() != 1 { return Err(ProcessError::OutputExtent { expected: 1, actual: flux.len() }); }
            if validity.len() != 1 { return Err(ProcessError::OutputExtent { expected: 1, actual: validity.len() }); }
            plan.process((offset,&mut flux[0],&mut validity[0]),workspace,pixels)
        }
        process_error=ProcessError;
    }
}
