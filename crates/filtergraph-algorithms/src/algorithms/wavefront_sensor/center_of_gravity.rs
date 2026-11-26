//! Prepared center-of-gravity measurement for one detector-pixel region.

use alloc::{sync::Arc, vec::Vec};
use core::fmt;

use num_traits::Float;

use crate::{AlgorithmPlan, NoWorkspace, ProcessError, PropertyConstraint, PropertyValue};

const REFERENCE_CENTROID_X_PROPERTY: u32 = 0;
const REFERENCE_CENTROID_Y_PROPERTY: u32 = 1;
const PIXEL_THRESHOLD_ENABLED_PROPERTY: u32 = 2;
const PIXEL_THRESHOLD_PROPERTY: u32 = 3;
const FLUX_THRESHOLD_PROPERTY: u32 = 4;

/// Invalid prepared center-of-gravity configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CenterOfGravityPlanError {
    /// The selected detector region contains no pixels.
    EmptyCoordinates,
    /// X and Y detector-coordinate arrays have different extents.
    CoordinateExtent {
        /// X-coordinate extent.
        x: usize,
        /// Y-coordinate extent.
        y: usize,
    },
    /// Pair-interleaved X/Y coordinates have the wrong scalar extent.
    PairInterleavedCoordinateExtent {
        /// Required scalar extent.
        expected: usize,
        /// Supplied scalar extent.
        actual: usize,
    },
    /// The initial coordinate count does not match the declared pixel count.
    CoordinateCount {
        /// Required coordinate-pair count.
        expected: usize,
        /// Supplied coordinate-pair count.
        actual: usize,
    },
    /// An X detector coordinate is not finite.
    NonFiniteX {
        /// Zero-based selected-pixel index.
        pixel: usize,
    },
    /// A Y detector coordinate is not finite.
    NonFiniteY {
        /// Zero-based selected-pixel index.
        pixel: usize,
    },
    /// A reference centroid coordinate is not finite.
    NonFiniteReference {
        /// Zero-based `x, y` coordinate index.
        coordinate: usize,
    },
    /// The optional inclusive pixel threshold is not finite.
    NonFinitePixelThreshold,
    /// The flux threshold is non-finite or negative.
    InvalidFluxThreshold,
}

impl fmt::Display for CenterOfGravityPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCoordinates => formatter.write_str("selected region must contain pixels"),
            Self::CoordinateExtent { x, y } => write!(
                formatter,
                "x-coordinate extent {x} does not match y-coordinate extent {y}"
            ),
            Self::PairInterleavedCoordinateExtent { expected, actual } => write!(
                formatter,
                "pair-interleaved coordinate extent {actual} does not match {expected}"
            ),
            Self::CoordinateCount { expected, actual } => write!(
                formatter,
                "coordinate-pair count {actual} does not match pixel count {expected}"
            ),
            Self::NonFiniteX { pixel } => {
                write!(
                    formatter,
                    "x coordinate at selected pixel {pixel} is not finite"
                )
            }
            Self::NonFiniteY { pixel } => {
                write!(
                    formatter,
                    "y coordinate at selected pixel {pixel} is not finite"
                )
            }
            Self::NonFiniteReference { coordinate } => write!(
                formatter,
                "reference centroid coordinate {coordinate} is not finite"
            ),
            Self::NonFinitePixelThreshold => {
                formatter.write_str("pixel threshold must be finite when supplied")
            }
            Self::InvalidFluxThreshold => {
                formatter.write_str("flux threshold must be finite and nonnegative")
            }
        }
    }
}

impl core::error::Error for CenterOfGravityPlanError {}

/// Caller-owned output for one center-of-gravity measurement.
///
/// `centroid_offset` uses `x, y` detector-coordinate order. Validity reports
/// the numerical and flux-threshold result. An invalid measurement has zero
/// centroid offset while retaining its measured flux.
#[derive(Debug)]
pub struct CenterOfGravityOutput<'a, T> {
    centroid_offset: &'a mut [T; 2],
    flux: &'a mut T,
    validity: &'a mut bool,
}

impl<'a, T> CenterOfGravityOutput<'a, T> {
    /// Groups caller-owned centroid-offset, flux, and validity outputs.
    pub fn new(centroid_offset: &'a mut [T; 2], flux: &'a mut T, validity: &'a mut bool) -> Self {
        Self {
            centroid_offset,
            flux,
            validity,
        }
    }

    /// Returns the `x, y` centroid offset.
    pub fn centroid_offset(&self) -> &[T; 2] {
        self.centroid_offset
    }

    /// Returns the selected-region flux.
    pub fn flux(&self) -> &T {
        self.flux
    }

    /// Returns measurement validity.
    pub fn validity(&self) -> bool {
        *self.validity
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CenterOfGravityMeasurement<T> {
    pub(super) centroid_offset: [T; 2],
    pub(super) flux: T,
    pub(super) validity: bool,
}

pub(super) fn measure_center_of_gravity<T>(
    pixel_count: usize,
    mut pixel: impl FnMut(usize) -> T,
    x_coordinates: &[T],
    y_coordinates: &[T],
    reference_centroid: [T; 2],
    pixel_threshold: Option<T>,
    flux_threshold: T,
) -> CenterOfGravityMeasurement<T>
where
    T: Float,
{
    let mut x_moment = T::zero();
    let mut y_moment = T::zero();
    let mut flux = T::zero();
    for index in 0..pixel_count {
        let value = pixel(index);
        let selected = pixel_threshold.is_none_or(|threshold| value >= threshold);
        // Preserve non-finite input so validity rejects the measurement instead
        // of silently turning the input into zero.
        if !value.is_finite() || selected {
            x_moment = x_moment + value * x_coordinates[index];
            y_moment = y_moment + value * y_coordinates[index];
            flux = flux + value;
        }
    }

    finish_center_of_gravity(x_moment, y_moment, flux, reference_centroid, flux_threshold)
}

// Keep the irregular detector-row loop separate from the much larger batched
// centroid routine. Both are hot on different Shack–Hartmann geometries.
#[cfg_attr(target_feature = "avx2", inline(never))]
pub(super) fn measure_center_of_gravity_rows<'pixels, T>(
    rows: impl IntoIterator<Item = &'pixels [T]>,
    x_coordinates: &[T],
    y_coordinates: &[T],
    reference_centroid: [T; 2],
    pixel_threshold: Option<T>,
    flux_threshold: T,
) -> CenterOfGravityMeasurement<T>
where
    T: Float + 'pixels,
{
    let mut x_moment = T::zero();
    let mut y_moment = T::zero();
    let mut flux = T::zero();
    let mut index = 0;
    for row in rows {
        for &value in row {
            let selected = pixel_threshold.is_none_or(|threshold| value >= threshold);
            // Preserve non-finite input so validity rejects the measurement
            // instead of silently turning the input into zero.
            if !value.is_finite() || selected {
                x_moment = x_moment + value * x_coordinates[index];
                y_moment = y_moment + value * y_coordinates[index];
                flux = flux + value;
            }
            index += 1;
        }
    }
    debug_assert_eq!(index, x_coordinates.len());
    debug_assert_eq!(index, y_coordinates.len());

    finish_center_of_gravity(x_moment, y_moment, flux, reference_centroid, flux_threshold)
}

#[inline]
pub(super) fn finish_center_of_gravity<T>(
    x_moment: T,
    y_moment: T,
    flux: T,
    reference_centroid: [T; 2],
    flux_threshold: T,
) -> CenterOfGravityMeasurement<T>
where
    T: Float,
{
    let mut validity = flux.is_finite() && flux > T::zero() && flux >= flux_threshold;
    let mut centroid_offset = [T::zero(); 2];
    if validity {
        centroid_offset[0] = x_moment / flux - reference_centroid[0];
        centroid_offset[1] = y_moment / flux - reference_centroid[1];
        validity = centroid_offset[0].is_finite() && centroid_offset[1].is_finite();
        if !validity {
            centroid_offset = [T::zero(); 2];
        }
    }
    CenterOfGravityMeasurement {
        centroid_offset,
        flux,
        validity,
    }
}

/// Prepared center-of-gravity measurement for one selected detector region.
///
/// X follows increasing detector column and Y follows increasing detector row.
/// Coordinates and `reference_centroid` share one coordinate unit; output is
/// their `x, y` centroid difference. Input pixels and flux use PDE on the
/// canonical AO boundary. When `pixel_threshold` is `Some`, pixels below the
/// inclusive threshold contribute zero to both moments and flux. When it is
/// `None`, every selected pixel contributes to the LLOWFS COG
/// convention.
#[derive(Clone, Debug)]
pub struct CenterOfGravityPlan<T> {
    x_coordinates: Arc<[T]>,
    y_coordinates: Arc<[T]>,
    reference_x: T,
    reference_y: T,
    pixel_threshold_enabled: bool,
    pixel_threshold: T,
    flux_threshold: T,
}

impl<T> CenterOfGravityPlan<T>
where
    T: Float,
{
    /// Prepares finite coordinates, reference centroid, and thresholds.
    pub fn new(
        x_coordinates: impl Into<Arc<[T]>>,
        y_coordinates: impl Into<Arc<[T]>>,
        reference_centroid: [T; 2],
        pixel_threshold: Option<T>,
        flux_threshold: T,
    ) -> Result<Self, CenterOfGravityPlanError> {
        let x_coordinates = x_coordinates.into();
        let y_coordinates = y_coordinates.into();
        if x_coordinates.is_empty() {
            return Err(CenterOfGravityPlanError::EmptyCoordinates);
        }
        if x_coordinates.len() != y_coordinates.len() {
            return Err(CenterOfGravityPlanError::CoordinateExtent {
                x: x_coordinates.len(),
                y: y_coordinates.len(),
            });
        }
        if let Some(pixel) = x_coordinates.iter().position(|value| !value.is_finite()) {
            return Err(CenterOfGravityPlanError::NonFiniteX { pixel });
        }
        if let Some(pixel) = y_coordinates.iter().position(|value| !value.is_finite()) {
            return Err(CenterOfGravityPlanError::NonFiniteY { pixel });
        }
        if let Some(coordinate) = reference_centroid
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(CenterOfGravityPlanError::NonFiniteReference { coordinate });
        }
        if pixel_threshold.is_some_and(|threshold| !threshold.is_finite()) {
            return Err(CenterOfGravityPlanError::NonFinitePixelThreshold);
        }
        if !flux_threshold.is_finite() || flux_threshold < T::zero() {
            return Err(CenterOfGravityPlanError::InvalidFluxThreshold);
        }
        let (pixel_threshold_enabled, pixel_threshold) =
            pixel_threshold.map_or((false, T::zero()), |threshold| (true, threshold));
        Ok(Self {
            x_coordinates,
            y_coordinates,
            reference_x: reference_centroid[0],
            reference_y: reference_centroid[1],
            pixel_threshold_enabled,
            pixel_threshold,
            flux_threshold,
        })
    }

    /// Returns X detector coordinates in selected-pixel order.
    pub fn x_coordinates(&self) -> &[T] {
        &self.x_coordinates
    }

    /// Returns Y detector coordinates in selected-pixel order.
    pub fn y_coordinates(&self) -> &[T] {
        &self.y_coordinates
    }

    /// Returns the `x, y` reference centroid.
    pub const fn reference_centroid(&self) -> [T; 2] {
        [self.reference_x, self.reference_y]
    }

    /// Returns the optional inclusive pixel threshold.
    pub const fn pixel_threshold(&self) -> Option<T> {
        if self.pixel_threshold_enabled {
            Some(self.pixel_threshold)
        } else {
            None
        }
    }

    /// Returns the minimum valid positive flux.
    pub const fn flux_threshold(&self) -> T {
        self.flux_threshold
    }

    /// Returns the selected-pixel extent.
    pub fn len(&self) -> usize {
        self.x_coordinates.len()
    }

    /// Returns `false`; a prepared COG region always contains pixels.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Creates the stateless workspace.
    pub const fn workspace(&self) -> NoWorkspace {
        NoWorkspace
    }

    /// Replaces X/Y detector coordinates from pair-interleaved pixel order.
    pub fn with_pair_interleaved_coordinates(
        &self,
        coordinates: &[T],
    ) -> Result<Self, CenterOfGravityPlanError> {
        let expected = self.len().saturating_mul(2);
        if coordinates.len() != expected {
            return Err(CenterOfGravityPlanError::PairInterleavedCoordinateExtent {
                expected,
                actual: coordinates.len(),
            });
        }
        let mut x_coordinates = Vec::with_capacity(self.len());
        let mut y_coordinates = Vec::with_capacity(self.len());
        for coordinate in coordinates.chunks_exact(2) {
            x_coordinates.push(coordinate[0]);
            y_coordinates.push(coordinate[1]);
        }
        Self::new(
            x_coordinates,
            y_coordinates,
            self.reference_centroid(),
            self.pixel_threshold(),
            self.flux_threshold,
        )
    }
}

fn replace_center_of_gravity_properties<T>(
    plan: &CenterOfGravityPlan<T>,
    reference_x: T,
    reference_y: T,
    pixel_threshold_enabled: bool,
    pixel_threshold: T,
    flux_threshold: T,
) -> Result<CenterOfGravityPlan<T>, CenterOfGravityPlanError>
where
    T: Float,
{
    CenterOfGravityPlan::new(
        Arc::clone(&plan.x_coordinates),
        Arc::clone(&plan.y_coordinates),
        [reference_x, reference_y],
        pixel_threshold_enabled.then_some(pixel_threshold),
        flux_threshold,
    )
}

macro_rules! impl_center_of_gravity_properties {
    ($scalar:ty, $value:ident) => {
        crate::algorithm_properties! {
            impl CenterOfGravityPlan<$scalar> {
                plan_error = CenterOfGravityPlanError;
                replace = replace_center_of_gravity_properties;
                properties {
                    REFERENCE_CENTROID_X_PROPERTY => reference_x: $value {
                        name = "reference-centroid-x";
                        description = "Reference centroid X coordinate";
                        unit = "px";
                        update = Runtime;
                        default = 0.0;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(<$scalar>::MIN),
                            maximum: PropertyValue::$value(<$scalar>::MAX),
                        };
                    }
                    REFERENCE_CENTROID_Y_PROPERTY => reference_y: $value {
                        name = "reference-centroid-y";
                        description = "Reference centroid Y coordinate";
                        unit = "px";
                        update = Runtime;
                        default = 0.0;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(<$scalar>::MIN),
                            maximum: PropertyValue::$value(<$scalar>::MAX),
                        };
                    }
                    PIXEL_THRESHOLD_ENABLED_PROPERTY => pixel_threshold_enabled: Bool {
                        name = "pixel-threshold-enabled";
                        description = "Whether the inclusive calibrated-pixel threshold is active";
                        unit = "1";
                        update = Runtime;
                        default = false;
                        constraint = PropertyConstraint::None;
                    }
                    PIXEL_THRESHOLD_PROPERTY => pixel_threshold: $value {
                        name = "pixel-threshold";
                        description = "Inclusive calibrated-pixel threshold";
                        unit = "PDE";
                        update = Runtime;
                        default = 0.0;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(<$scalar>::MIN),
                            maximum: PropertyValue::$value(<$scalar>::MAX),
                        };
                    }
                    FLUX_THRESHOLD_PROPERTY => flux_threshold: $value {
                        name = "flux-threshold";
                        description = "Minimum positive selected-region flux";
                        unit = "PDE";
                        update = Runtime;
                        default = 0.0;
                        constraint = PropertyConstraint::Range {
                            minimum: PropertyValue::$value(0.0),
                            maximum: PropertyValue::$value(<$scalar>::MAX),
                        };
                    }
                }
            }
        }
    };
}

impl_center_of_gravity_properties!(f32, Float);
impl_center_of_gravity_properties!(f64, Double);

impl<T> AlgorithmPlan for CenterOfGravityPlan<T>
where
    T: Float + 'static,
{
    type Input<'a> = &'a [T];
    type Output<'a> = CenterOfGravityOutput<'a, T>;
    type Workspace = NoWorkspace;
    type Error = ProcessError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if input.len() != self.len() {
            return Err(ProcessError::InputExtent {
                expected: self.len(),
                actual: input.len(),
            });
        }
        let measurement = measure_center_of_gravity(
            self.len(),
            |pixel| input[pixel],
            &self.x_coordinates,
            &self.y_coordinates,
            self.reference_centroid(),
            self.pixel_threshold(),
            self.flux_threshold,
        );
        *output.centroid_offset = measurement.centroid_offset;
        *output.flux = measurement.flux;
        *output.validity = measurement.validity;
        Ok(())
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 center-of-gravity measurement for one detector region.
    pub CenterOfGravityF32 {
        label = "center-of-gravity-f32";
        configuration CenterOfGravityF32Configuration {
            pixel_count: usize = GraphRebuild;
            initial_coordinates: Option<Vec<[f32; 2]>> = Construction;
        }
        plan = CenterOfGravityPlan<f32>;
        properties = runtime;
        dimensions |plan, _workspace| {
            pixel_count = plan.len();
        }
        inputs {
            pixels: f32 [pixel_count] = "pixels",
                schema = crate::schemas::CENTER_OF_GRAVITY_PIXELS_V1,
                optional_sample_metadata = ["chopped_image_kind", "trigger_state", "acquisition_identity", "exposure_duration_nanoseconds"];
        }
        outputs {
            centroid_offset: f32 [2] = "centroid-offset",
                schema = crate::schemas::CENTER_OF_GRAVITY_OFFSET_V1,
                metadata_from = "pixels";
            flux: f32 [1] = "flux",
                schema = crate::schemas::CENTER_OF_GRAVITY_FLUX_V1,
                metadata_from = "pixels";
            validity: bool [1] = "validity",
                schema = crate::schemas::CENTER_OF_GRAVITY_VALIDITY_V1,
                metadata_from = "pixels";
        }
        parameters {
            coordinates: f32 [pixel_count, 2] = "coordinates",
                schema = crate::schemas::CENTER_OF_GRAVITY_COORDINATES_V1,
                replace |plan, values| {
                    plan.with_pair_interleaved_coordinates(values)
                };
        }
        parameter_error = CenterOfGravityPlanError;
        prepare |configuration| {
            let initial_coordinates = configuration.initial_coordinates
                .unwrap_or_else(|| alloc::vec![[0.0; 2]; configuration.pixel_count]);
            if initial_coordinates.len() != configuration.pixel_count {
                return Err(CenterOfGravityPlanError::CoordinateCount {
                    expected: configuration.pixel_count,
                    actual: initial_coordinates.len(),
                });
            }
            let mut x_coordinates = Vec::with_capacity(configuration.pixel_count);
            let mut y_coordinates = Vec::with_capacity(configuration.pixel_count);
            for coordinate in initial_coordinates {
                x_coordinates.push(coordinate[0]);
                y_coordinates.push(coordinate[1]);
            }
            Ok((
                CenterOfGravityPlan::new(
                    x_coordinates,
                    y_coordinates,
                    [0.0; 2],
                    None,
                    0.0,
                )?,
                NoWorkspace,
            ))
        }
        prepare_error = CenterOfGravityPlanError;
        process |plan, workspace| {
            let actual = centroid_offset.len();
            let centroid_offset = <&mut [f32; 2]>::try_from(centroid_offset).map_err(|_| {
                ProcessError::OutputExtent {
                    expected: 2,
                    actual,
                }
            })?;
            if flux.len() != 1 { return Err(ProcessError::OutputExtent { expected: 1, actual: flux.len() }); }
            if validity.len() != 1 { return Err(ProcessError::OutputExtent { expected: 1, actual: validity.len() }); }
            plan.process(
                CenterOfGravityOutput::new(
                    centroid_offset,
                    &mut flux[0],
                    &mut validity[0],
                ),
                workspace,
                pixels,
            )
        }
        process_error = ProcessError;
    }
}
