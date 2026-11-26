//! Image-backed Shack-Hartmann measurement without a materialized region slab.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::ops::Range;

use num_traits::Float;

use crate::{
    AlgorithmPlan, ProcessError, ProgressWorkspace, RegionExtractionError, RegionExtractionPlan,
    ShackHartmannError, ShackHartmannOutput, ShackHartmannOutputError, ShackHartmannPlan,
};

/// Invalid scientific preparation or sparse calibration replacement for an
/// image-backed Shack-Hartmann measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannImagePreparationError {
    /// Detector-region geometry is invalid.
    Regions(RegionExtractionError),
    /// Shack-Hartmann coordinates, references, thresholds, or active state are invalid.
    Measurement(ShackHartmannError),
    /// Detector-region and measurement plans are incompatible.
    Composition(ShackHartmannImagePlanError),
    /// Coordinate scale must be finite and positive.
    InvalidCoordinateScale,
    /// Coordinate or pair-interleaved extent arithmetic overflowed.
    ExtentOverflow,
    /// A sparse origin array does not contain one row/column pair per subaperture.
    OriginExtent {
        /// Required flattened element extent.
        expected: usize,
        /// Supplied flattened element extent.
        actual: usize,
    },
    /// A sparse coordinate array does not contain one X/Y pair per pixel.
    CoordinateExtent {
        /// Required flattened element extent.
        expected: usize,
        /// Supplied flattened element extent.
        actual: usize,
    },
    /// A sparse threshold array does not contain one pixel/flux pair per subaperture.
    ThresholdExtent {
        /// Required flattened element extent.
        expected: usize,
        /// Supplied flattened element extent.
        actual: usize,
    },
    /// A portable `u32` origin cannot be represented by the current target.
    OriginCoordinateOverflow,
    /// An explicit initial origin table does not match the declared count.
    InitialOriginCount {
        /// Required number of row/column pairs.
        expected: usize,
        /// Supplied number of row/column pairs.
        actual: usize,
    },
    /// The requested regular initial grid does not fit on the detector.
    InitialOriginGrid,
}

impl fmt::Display for ShackHartmannImagePreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Regions(error) => error.fmt(formatter),
            Self::Measurement(error) => error.fmt(formatter),
            Self::Composition(error) => error.fmt(formatter),
            Self::InvalidCoordinateScale => {
                formatter.write_str("coordinate scale must be finite and positive")
            }
            Self::ExtentOverflow => formatter.write_str("Shack-Hartmann extent overflow"),
            Self::OriginExtent { expected, actual } => write!(
                formatter,
                "flattened subaperture-origin extent {actual} does not match {expected}"
            ),
            Self::CoordinateExtent { expected, actual } => write!(
                formatter,
                "flattened Shack-Hartmann coordinate extent {actual} does not match {expected}"
            ),
            Self::ThresholdExtent { expected, actual } => write!(
                formatter,
                "flattened Shack-Hartmann threshold extent {actual} does not match {expected}"
            ),
            Self::OriginCoordinateOverflow => {
                formatter.write_str("a subaperture origin does not fit usize")
            }
            Self::InitialOriginCount { expected, actual } => write!(
                formatter,
                "initial subaperture-origin count {actual} does not match {expected}"
            ),
            Self::InitialOriginGrid => formatter
                .write_str("the requested regular subaperture grid does not fit on the detector"),
        }
    }
}

impl core::error::Error for ShackHartmannImagePreparationError {}

impl From<RegionExtractionError> for ShackHartmannImagePreparationError {
    fn from(error: RegionExtractionError) -> Self {
        Self::Regions(error)
    }
}

impl From<ShackHartmannError> for ShackHartmannImagePreparationError {
    fn from(error: ShackHartmannError) -> Self {
        Self::Measurement(error)
    }
}

impl From<ShackHartmannImagePlanError> for ShackHartmannImagePreparationError {
    fn from(error: ShackHartmannImagePlanError) -> Self {
        Self::Composition(error)
    }
}

/// Invalid caller arrays or processing state for an image-backed
/// Shack-Hartmann measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannImageProcessError {
    /// Caller-owned slope, flux, and validity arrays are incompatible.
    Output(ShackHartmannOutputError),
    /// The prepared scientific operation rejected the call.
    Algorithm(ProcessError),
}

impl fmt::Display for ShackHartmannImageProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Output(error) => error.fmt(formatter),
            Self::Algorithm(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for ShackHartmannImageProcessError {}

/// Incompatible prepared image-region and Shack-Hartmann plans.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannImagePlanError {
    /// Region and measurement plans describe different subaperture counts.
    SubapertureExtent {
        /// Number of image regions.
        regions: usize,
        /// Number of Shack-Hartmann measurements.
        measurements: usize,
    },
    /// One image region and one Shack-Hartmann measurement use different
    /// pixel extents.
    SubaperturePixelExtent {
        /// Pixels in one image region.
        region_pixels: usize,
        /// Coordinates in one Shack-Hartmann measurement.
        measurement_pixels: usize,
    },
}

impl fmt::Display for ShackHartmannImagePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SubapertureExtent {
                regions,
                measurements,
            } => write!(
                formatter,
                "image region count {regions} does not match Shack-Hartmann measurement count {measurements}"
            ),
            Self::SubaperturePixelExtent {
                region_pixels,
                measurement_pixels,
            } => write!(
                formatter,
                "image region pixel extent {region_pixels} does not match Shack-Hartmann coordinate extent {measurement_pixels}"
            ),
        }
    }
}

impl core::error::Error for ShackHartmannImagePlanError {}

/// Prepared detector-image to Shack-Hartmann measurement composition.
///
/// The composition preserves the ordinary typed [`AlgorithmPlan`] boundary.
/// It borrows rectangular subapertures as detector-image row views, so a graph
/// adapter does not need an intermediate region tensor or a transport-specific
/// fused implementation.
#[derive(Clone, Debug)]
pub struct ShackHartmannImagePlan<T> {
    regions: RegionExtractionPlan<T>,
    measurement: ShackHartmannPlan<T>,
}

impl<T> ShackHartmannImagePlan<T>
where
    T: Float,
{
    /// Composes compatible image geometry and Shack-Hartmann measurement.
    pub fn new(
        regions: RegionExtractionPlan<T>,
        measurement: ShackHartmannPlan<T>,
    ) -> Result<Self, ShackHartmannImagePlanError> {
        if regions.len() != measurement.len() {
            return Err(ShackHartmannImagePlanError::SubapertureExtent {
                regions: regions.len(),
                measurements: measurement.len(),
            });
        }
        if regions.pixels_per_region() != measurement.pixel_count() {
            return Err(ShackHartmannImagePlanError::SubaperturePixelExtent {
                region_pixels: regions.pixels_per_region(),
                measurement_pixels: measurement.pixel_count(),
            });
        }
        Ok(Self {
            regions,
            measurement,
        })
    }

    /// Returns the prepared detector-image region geometry.
    pub const fn regions(&self) -> &RegionExtractionPlan<T> {
        &self.regions
    }

    /// Returns the prepared Shack-Hartmann measurement.
    pub const fn measurement(&self) -> &ShackHartmannPlan<T> {
        &self.measurement
    }

    /// Creates idle Shack-Hartmann execution storage.
    pub const fn workspace(&self) -> ProgressWorkspace {
        self.measurement.workspace()
    }

    /// Replaces zero-based subaperture origins while preserving detector
    /// layout and prepared measurement calibration.
    pub fn with_u32_subaperture_origins(
        &self,
        pair_interleaved_origins: &[u32],
    ) -> Result<Self, ShackHartmannImagePreparationError> {
        let expected = self
            .regions
            .len()
            .checked_mul(2)
            .ok_or(ShackHartmannImagePreparationError::ExtentOverflow)?;
        if pair_interleaved_origins.len() != expected {
            return Err(ShackHartmannImagePreparationError::OriginExtent {
                expected,
                actual: pair_interleaved_origins.len(),
            });
        }
        let mut origins = Vec::with_capacity(self.regions.len());
        for origin in pair_interleaved_origins.chunks_exact(2) {
            origins.push((
                usize::try_from(origin[0])
                    .map_err(|_| ShackHartmannImagePreparationError::OriginCoordinateOverflow)?,
                usize::try_from(origin[1])
                    .map_err(|_| ShackHartmannImagePreparationError::OriginCoordinateOverflow)?,
            ));
        }
        let regions = RegionExtractionPlan::with_layout(
            self.regions.image_size(),
            &origins,
            self.regions.region_size(),
            self.regions.row_stride(),
            self.regions.first_pixel(),
        )?;
        Self::new(regions, self.measurement.clone()).map_err(Into::into)
    }

    /// Replaces pair-interleaved X/Y coordinates for one subaperture.
    pub fn with_pair_interleaved_coordinates(
        &self,
        pair_interleaved_coordinates: &[T],
    ) -> Result<Self, ShackHartmannImagePreparationError> {
        let expected = self
            .measurement
            .pixel_count()
            .checked_mul(2)
            .ok_or(ShackHartmannImagePreparationError::ExtentOverflow)?;
        if pair_interleaved_coordinates.len() != expected {
            return Err(ShackHartmannImagePreparationError::CoordinateExtent {
                expected,
                actual: pair_interleaved_coordinates.len(),
            });
        }
        let mut x_coordinates = Vec::with_capacity(self.measurement.pixel_count());
        let mut y_coordinates = Vec::with_capacity(self.measurement.pixel_count());
        for coordinates in pair_interleaved_coordinates.chunks_exact(2) {
            x_coordinates.push(coordinates[0]);
            y_coordinates.push(coordinates[1]);
        }
        self.with_measurement(ShackHartmannPlan::new(
            x_coordinates,
            y_coordinates,
            self.measurement.reference_slopes().to_vec(),
            self.measurement.pixel_thresholds().to_vec(),
            self.measurement.flux_thresholds().to_vec(),
            self.measurement.active().to_vec(),
        )?)
    }

    /// Replaces pair-interleaved reference slopes.
    pub fn with_reference_slopes(
        &self,
        reference_slopes: &[T],
    ) -> Result<Self, ShackHartmannImagePreparationError> {
        self.with_measurement(ShackHartmannPlan::new(
            self.measurement.x_coordinates().to_vec(),
            self.measurement.y_coordinates().to_vec(),
            reference_slopes.to_vec(),
            self.measurement.pixel_thresholds().to_vec(),
            self.measurement.flux_thresholds().to_vec(),
            self.measurement.active().to_vec(),
        )?)
    }

    /// Replaces pair-interleaved pixel and flux thresholds.
    pub fn with_pair_interleaved_thresholds(
        &self,
        pair_interleaved_thresholds: &[T],
    ) -> Result<Self, ShackHartmannImagePreparationError> {
        let expected = self
            .measurement
            .len()
            .checked_mul(2)
            .ok_or(ShackHartmannImagePreparationError::ExtentOverflow)?;
        if pair_interleaved_thresholds.len() != expected {
            return Err(ShackHartmannImagePreparationError::ThresholdExtent {
                expected,
                actual: pair_interleaved_thresholds.len(),
            });
        }
        let mut pixel_thresholds = Vec::with_capacity(self.measurement.len());
        let mut flux_thresholds = Vec::with_capacity(self.measurement.len());
        for thresholds in pair_interleaved_thresholds.chunks_exact(2) {
            pixel_thresholds.push(thresholds[0]);
            flux_thresholds.push(thresholds[1]);
        }
        self.with_measurement(ShackHartmannPlan::new(
            self.measurement.x_coordinates().to_vec(),
            self.measurement.y_coordinates().to_vec(),
            self.measurement.reference_slopes().to_vec(),
            pixel_thresholds,
            flux_thresholds,
            self.measurement.active().to_vec(),
        )?)
    }

    /// Replaces the subaperture active state.
    pub fn with_active(&self, active: &[bool]) -> Result<Self, ShackHartmannImagePreparationError> {
        self.with_measurement(ShackHartmannPlan::new(
            self.measurement.x_coordinates().to_vec(),
            self.measurement.y_coordinates().to_vec(),
            self.measurement.reference_slopes().to_vec(),
            self.measurement.pixel_thresholds().to_vec(),
            self.measurement.flux_thresholds().to_vec(),
            active.to_vec(),
        )?)
    }

    pub(crate) fn process_image_range(
        &self,
        output: ShackHartmannOutput<'_, T>,
        image: &[T],
        workrange: Range<usize>,
    ) -> Result<(), ProcessError> {
        let regions = self.regions.image_regions(image)?;
        self.measurement
            .process_image_range(output, regions, workrange)
    }

    /// Processes one disjoint output partition from the complete image.
    ///
    /// This is an adapter hook for a declared native operation shape. The
    /// ordinary complete-image [`AlgorithmPlan`] implementation remains the
    /// scientist-facing serial reference.
    #[doc(hidden)]
    pub fn process_image_partition(
        &self,
        slopes: &mut [T],
        flux: &mut [T],
        validity: &mut [bool],
        image: &[T],
        first_subaperture: usize,
    ) -> Result<(), ShackHartmannImageProcessError> {
        let output = ShackHartmannOutput::new(slopes, flux, validity)
            .map_err(ShackHartmannImageProcessError::Output)?;
        let regions = self
            .regions
            .image_regions(image)
            .map_err(ShackHartmannImageProcessError::Algorithm)?;
        self.measurement
            .process_image_partition(output, regions, first_subaperture)
            .map_err(ShackHartmannImageProcessError::Algorithm)
    }

    fn with_measurement(
        &self,
        measurement: ShackHartmannPlan<T>,
    ) -> Result<Self, ShackHartmannImagePreparationError> {
        Self::new(self.regions.clone(), measurement).map_err(Into::into)
    }
}

impl ShackHartmannImagePlan<f32> {
    /// Prepares regular centered coordinates over rectangular detector
    /// subapertures with optional reference slopes and active state.
    #[allow(clippy::too_many_arguments)]
    pub fn from_detector_regions(
        image_size: (usize, usize),
        subaperture_size: (usize, usize),
        subaperture_origins: &[(usize, usize)],
        coordinate_scale: f32,
        pixel_threshold: f32,
        flux_threshold: f32,
        reference_slopes: Option<Vec<f32>>,
        active: Option<Vec<bool>>,
    ) -> Result<Self, ShackHartmannImagePreparationError> {
        if !coordinate_scale.is_finite() || coordinate_scale <= 0.0 {
            return Err(ShackHartmannImagePreparationError::InvalidCoordinateScale);
        }
        let regions = RegionExtractionPlan::new(image_size, subaperture_origins, subaperture_size)?;
        let (subaperture_rows, subaperture_columns) = subaperture_size;
        let pixel_count = subaperture_rows
            .checked_mul(subaperture_columns)
            .ok_or(ShackHartmannImagePreparationError::ExtentOverflow)?;
        let mut x_coordinates = Vec::with_capacity(pixel_count);
        let mut y_coordinates = Vec::with_capacity(pixel_count);
        let x_center = (subaperture_columns as f32 - 1.0) * 0.5;
        let y_center = (subaperture_rows as f32 - 1.0) * 0.5;
        for row in 0..subaperture_rows {
            for column in 0..subaperture_columns {
                x_coordinates.push((column as f32 - x_center) * coordinate_scale);
                y_coordinates.push((row as f32 - y_center) * coordinate_scale);
            }
        }
        let subaperture_count = regions.len();
        let slope_count = subaperture_count
            .checked_mul(2)
            .ok_or(ShackHartmannImagePreparationError::ExtentOverflow)?;
        let measurement = ShackHartmannPlan::new(
            x_coordinates,
            y_coordinates,
            reference_slopes.unwrap_or_else(|| vec![0.0; slope_count]),
            vec![pixel_threshold; subaperture_count],
            vec![flux_threshold; subaperture_count],
            active.unwrap_or_else(|| vec![true; subaperture_count]),
        )?;
        Self::new(regions, measurement).map_err(Into::into)
    }
}

impl<T> AlgorithmPlan for ShackHartmannImagePlan<T>
where
    T: Float + 'static,
{
    type Input<'a> = &'a [T];
    type Output<'a> = ShackHartmannOutput<'a, T>;
    type Workspace = ProgressWorkspace;
    type Error = ProcessError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        let regions = self.regions.image_regions(input)?;
        self.measurement
            .process_image_regions(output, workspace, regions)
    }
}

pub(crate) fn initial_subaperture_origins(
    image_size: (usize, usize),
    subaperture_size: (usize, usize),
    subaperture_count: usize,
    explicit: Option<Vec<[usize; 2]>>,
) -> Result<Vec<(usize, usize)>, ShackHartmannImagePreparationError> {
    if let Some(origins) = explicit {
        if origins.len() != subaperture_count {
            return Err(ShackHartmannImagePreparationError::InitialOriginCount {
                expected: subaperture_count,
                actual: origins.len(),
            });
        }
        return Ok(origins
            .into_iter()
            .map(|origin| (origin[0], origin[1]))
            .collect());
    }
    if subaperture_size.0 == 0 || subaperture_size.1 == 0 {
        return Err(ShackHartmannImagePreparationError::InitialOriginGrid);
    }
    let grid_rows = image_size.0 / subaperture_size.0;
    let grid_columns = image_size.1 / subaperture_size.1;
    let capacity = grid_rows
        .checked_mul(grid_columns)
        .ok_or(ShackHartmannImagePreparationError::ExtentOverflow)?;
    if subaperture_count == 0 || subaperture_count > capacity {
        return Err(ShackHartmannImagePreparationError::InitialOriginGrid);
    }
    Ok((0..subaperture_count)
        .map(|index| {
            (
                (index / grid_columns) * subaperture_size.0,
                (index % grid_columns) * subaperture_size.1,
            )
        })
        .collect())
}

crate::filtergraph_algorithm! {
    /// Portable Float32 image-backed Shack-Hartmann declaration.
    pub ShackHartmannImageF32 {
        label = "shack-hartmann-image-f32";
        configuration ShackHartmannImageF32Configuration {
            image_size: (usize, usize) = GraphRebuild;
            image_schema: String = GraphRebuild;
            subaperture_size: (usize, usize) = GraphRebuild;
            subaperture_count: usize = GraphRebuild;
            initial_subaperture_origins: Option<Vec<[usize; 2]>> = Construction;
            coordinate_scale: f32 = Construction;
            pixel_threshold: f32 = Construction;
            flux_threshold: f32 = Construction;
            reference_slopes: Option<Vec<f32>> = Construction;
            active: Option<Vec<bool>> = Construction;
        }
        plan = ShackHartmannImagePlan<f32>;
        operation = ShackHartmannImageF32;
        properties = none;
        dimensions |plan, _workspace| {
            image_rows = plan.regions().image_size().0;
            image_columns = plan.regions().image_size().1;
            subaperture_count = plan.measurement().len();
            pixel_count = plan.measurement().pixel_count();
        }
        inputs {
            image: f32 [image_rows, image_columns] = "image",
                schema = configuration.image_schema.as_str();
        }
        outputs {
            slopes: f32 [subaperture_count, 2] = "slopes",
                schema = crate::schemas::SHACK_HARTMANN_SLOPES_V1,
                metadata_from = "image";
            flux: f32 [subaperture_count] = "flux",
                schema = crate::schemas::SHACK_HARTMANN_FLUX_V1,
                metadata_from = "image";
            validity: bool [subaperture_count] = "validity",
                schema = crate::schemas::SHACK_HARTMANN_VALIDITY_V1,
                metadata_from = "image";
        }
        parameters {
            subaperture_origins: u32 [subaperture_count, 2] = "subaperture-origins",
                schema = crate::schemas::REGION_ORIGINS_V1,
                replace |plan, values| {
                    plan.with_u32_subaperture_origins(values)
                };
            coordinates: f32 [pixel_count, 2] = "coordinates",
                schema = crate::schemas::SHACK_HARTMANN_COORDINATES_V1,
                replace |plan, values| {
                    plan.with_pair_interleaved_coordinates(values)
                };
            reference_slopes: f32 [subaperture_count, 2] = "reference-slopes",
                schema = crate::schemas::SHACK_HARTMANN_REFERENCE_SLOPES_V1,
                replace |plan, values| {
                    plan.with_reference_slopes(values)
                };
            thresholds: f32 [subaperture_count, 2] = "thresholds",
                schema = crate::schemas::SHACK_HARTMANN_THRESHOLDS_V1,
                replace |plan, values| {
                    plan.with_pair_interleaved_thresholds(values)
                };
            active: bool [subaperture_count] = "active",
                schema = crate::schemas::SHACK_HARTMANN_ACTIVE_V1,
                replace |plan, values| {
                    plan.with_active(values)
                };
        }
        parameter_error = ShackHartmannImagePreparationError;
        prepare |configuration| {
            let origins = initial_subaperture_origins(
                configuration.image_size,
                configuration.subaperture_size,
                configuration.subaperture_count,
                configuration.initial_subaperture_origins,
            )?;
            let plan = ShackHartmannImagePlan::from_detector_regions(
                configuration.image_size,
                configuration.subaperture_size,
                &origins,
                configuration.coordinate_scale,
                configuration.pixel_threshold,
                configuration.flux_threshold,
                configuration.reference_slopes,
                configuration.active,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = ShackHartmannImagePreparationError;
        process |plan, workspace| {
            let output = ShackHartmannOutput::new(slopes, flux, validity)
                .map_err(ShackHartmannImageProcessError::Output)?;
            plan.process(output, workspace, image)
                .map_err(ShackHartmannImageProcessError::Algorithm)
        }
        process_error = ShackHartmannImageProcessError;
        reset |_plan, workspace| {
            *workspace = ProgressWorkspace::new();
        }
    }
}
