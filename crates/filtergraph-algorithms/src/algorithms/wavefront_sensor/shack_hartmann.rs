//! Prepared center-of-gravity processing for Shack-Hartmann wavefront sensors.

use alloc::sync::Arc;
use core::fmt;
use core::ops::Range;

use num_traits::Float;

use super::center_of_gravity::{
    CenterOfGravityMeasurement, finish_center_of_gravity, measure_center_of_gravity,
    measure_center_of_gravity_rows,
};
use crate::{
    AlgorithmPlan, ImageRegionPixels, InputProgress, OutputProgress, ProcessError, ProgressResult,
    ProgressWorkspace, RegionPixels,
};

/// Invalid prepared Shack-Hartmann configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannError {
    /// Each subaperture must contain pixels.
    EmptyCoordinates,

    /// X and Y coordinate arrays have different extents.
    CoordinateExtent {
        /// X-coordinate extent.
        x: usize,
        /// Y-coordinate extent.
        y: usize,
    },

    /// No subaperture was supplied.
    EmptySubapertures,

    /// Per-subaperture arrays have different extents.
    SubapertureExtent {
        /// Pixel-threshold extent.
        pixel_thresholds: usize,
        /// Flux-threshold extent.
        flux_thresholds: usize,
        /// Active-state extent.
        active: usize,
    },

    /// Reference slopes do not contain one pair per subaperture.
    ReferenceSlopeExtent {
        /// Required pair-interleaved slope extent.
        expected: usize,
        /// Supplied reference-slope extent.
        actual: usize,
    },

    /// A pair-interleaved slope extent cannot be represented by `usize`.
    SlopeExtentOverflow,

    /// An X gradient coordinate is not finite.
    NonFiniteX {
        /// Zero-based pixel index.
        pixel: usize,
    },

    /// A Y gradient coordinate is not finite.
    NonFiniteY {
        /// Zero-based pixel index.
        pixel: usize,
    },

    /// A prepared reference slope is not finite.
    NonFiniteReference {
        /// Zero-based pair-interleaved slope index.
        slope: usize,
    },

    /// A pixel threshold is non-finite or negative.
    InvalidPixelThreshold {
        /// Zero-based subaperture index.
        subaperture: usize,
    },

    /// A flux threshold is non-finite or negative.
    InvalidFluxThreshold {
        /// Zero-based subaperture index.
        subaperture: usize,
    },
}

impl fmt::Display for ShackHartmannError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCoordinates => formatter.write_str("each subaperture must contain pixels"),
            Self::CoordinateExtent { x, y } => {
                write!(
                    formatter,
                    "x-coordinate extent {x} does not match y extent {y}"
                )
            }
            Self::EmptySubapertures => formatter.write_str("at least one subaperture is required"),
            Self::SubapertureExtent {
                pixel_thresholds,
                flux_thresholds,
                active,
            } => write!(
                formatter,
                "pixel-threshold extent {pixel_thresholds}, flux-threshold extent \
                 {flux_thresholds}, and active extent {active} must match"
            ),
            Self::ReferenceSlopeExtent { expected, actual } => write!(
                formatter,
                "reference-slope extent {actual} does not match {expected}"
            ),
            Self::SlopeExtentOverflow => {
                formatter.write_str("pair-interleaved slope extent overflows usize")
            }
            Self::NonFiniteX { pixel } => {
                write!(formatter, "x coordinate at pixel {pixel} is not finite")
            }
            Self::NonFiniteY { pixel } => {
                write!(formatter, "y coordinate at pixel {pixel} is not finite")
            }
            Self::NonFiniteReference { slope } => {
                write!(formatter, "reference slope at index {slope} is not finite")
            }
            Self::InvalidPixelThreshold { subaperture } => write!(
                formatter,
                "pixel threshold for subaperture {subaperture} must be finite and nonnegative"
            ),
            Self::InvalidFluxThreshold { subaperture } => write!(
                formatter,
                "flux threshold for subaperture {subaperture} must be finite and nonnegative"
            ),
        }
    }
}

impl core::error::Error for ShackHartmannError {}

/// Invalid caller-owned Shack-Hartmann output arrays.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannOutputError {
    /// Output contains no subapertures.
    Empty,

    /// Flux and validity arrays have different extents.
    SubapertureExtent {
        /// Flux-array extent.
        flux: usize,
        /// Validity-array extent.
        validity: usize,
    },

    /// Slope output does not contain one pair per subaperture.
    SlopeExtent {
        /// Required pair-interleaved slope extent.
        expected: usize,
        /// Supplied slope extent.
        actual: usize,
    },

    /// Pair-interleaved slope extent cannot be represented by `usize`.
    SlopeExtentOverflow,
}

impl fmt::Display for ShackHartmannOutputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("output arrays must be nonempty"),
            Self::SubapertureExtent { flux, validity } => write!(
                formatter,
                "flux extent {flux} does not match validity extent {validity}"
            ),
            Self::SlopeExtent { expected, actual } => {
                write!(formatter, "slope extent {actual} does not match {expected}")
            }
            Self::SlopeExtentOverflow => {
                formatter.write_str("pair-interleaved slope extent overflows usize")
            }
        }
    }
}

impl core::error::Error for ShackHartmannOutputError {}

/// Grouped caller-owned Shack-Hartmann output arrays.
///
/// Slopes use pair-interleaved `x, y` order. Validity reports measurement
/// validity independently of the prepared active state.
#[derive(Debug)]
pub struct ShackHartmannOutput<'a, T> {
    slopes: &'a mut [T],
    flux: &'a mut [T],
    validity: &'a mut [bool],
}

impl<'a, T> ShackHartmannOutput<'a, T> {
    /// Validates and groups caller-owned output arrays.
    pub fn new(
        slopes: &'a mut [T],
        flux: &'a mut [T],
        validity: &'a mut [bool],
    ) -> Result<Self, ShackHartmannOutputError> {
        if flux.is_empty() {
            return Err(ShackHartmannOutputError::Empty);
        }
        if flux.len() != validity.len() {
            return Err(ShackHartmannOutputError::SubapertureExtent {
                flux: flux.len(),
                validity: validity.len(),
            });
        }
        let expected = flux
            .len()
            .checked_mul(2)
            .ok_or(ShackHartmannOutputError::SlopeExtentOverflow)?;
        if slopes.len() != expected {
            return Err(ShackHartmannOutputError::SlopeExtent {
                expected,
                actual: slopes.len(),
            });
        }
        Ok(Self {
            slopes,
            flux,
            validity,
        })
    }

    /// Returns pair-interleaved wavefront slopes.
    pub fn slopes(&self) -> &[T] {
        self.slopes
    }

    /// Returns subaperture flux.
    pub fn flux(&self) -> &[T] {
        self.flux
    }

    /// Returns measurement validity in subaperture order.
    pub fn validity(&self) -> &[bool] {
        self.validity
    }

    /// Returns the output subaperture extent.
    pub const fn len(&self) -> usize {
        self.flux.len()
    }

    /// Returns `false`; valid output contains at least one subaperture.
    pub const fn is_empty(&self) -> bool {
        false
    }
}

/// Prepared local center-of-gravity processing for a Shack-Hartmann wavefront
/// sensor.
///
/// Coordinates contain gradient coordinates from the region center in
/// row-major pixel order. Positive X follows increasing detector column and
/// positive Y follows increasing detector row. The output and pair-interleaved
/// reference slopes use the coordinate unit. On the canonical AO boundary that
/// unit is mas; a private exact replay may instead retain a matched
/// artifact-native coordinate unit. Each subaperture has an absolute pixel
/// threshold: pixels below it contribute zero to both moments and flux, while
/// pixels equal to or above it contribute their full value.
#[derive(Clone, Debug)]
pub struct ShackHartmannPlan<T> {
    x_coordinates: Arc<[T]>,
    y_coordinates: Arc<[T]>,
    reference_slopes: Arc<[T]>,
    pixel_thresholds: Arc<[T]>,
    flux_thresholds: Arc<[T]>,
    active: Arc<[bool]>,
}

impl<T> ShackHartmannPlan<T>
where
    T: Float,
{
    /// Prepares finite coordinates, references, thresholds, and active state.
    pub fn new(
        x_coordinates: impl Into<Arc<[T]>>,
        y_coordinates: impl Into<Arc<[T]>>,
        reference_slopes: impl Into<Arc<[T]>>,
        pixel_thresholds: impl Into<Arc<[T]>>,
        flux_thresholds: impl Into<Arc<[T]>>,
        active: impl Into<Arc<[bool]>>,
    ) -> Result<Self, ShackHartmannError> {
        let x_coordinates = x_coordinates.into();
        let y_coordinates = y_coordinates.into();
        let reference_slopes = reference_slopes.into();
        let pixel_thresholds = pixel_thresholds.into();
        let flux_thresholds = flux_thresholds.into();
        let active = active.into();

        if x_coordinates.is_empty() {
            return Err(ShackHartmannError::EmptyCoordinates);
        }
        if x_coordinates.len() != y_coordinates.len() {
            return Err(ShackHartmannError::CoordinateExtent {
                x: x_coordinates.len(),
                y: y_coordinates.len(),
            });
        }
        if pixel_thresholds.is_empty() {
            return Err(ShackHartmannError::EmptySubapertures);
        }
        if pixel_thresholds.len() != flux_thresholds.len() || pixel_thresholds.len() != active.len()
        {
            return Err(ShackHartmannError::SubapertureExtent {
                pixel_thresholds: pixel_thresholds.len(),
                flux_thresholds: flux_thresholds.len(),
                active: active.len(),
            });
        }
        let slope_count = pixel_thresholds
            .len()
            .checked_mul(2)
            .ok_or(ShackHartmannError::SlopeExtentOverflow)?;
        if reference_slopes.len() != slope_count {
            return Err(ShackHartmannError::ReferenceSlopeExtent {
                expected: slope_count,
                actual: reference_slopes.len(),
            });
        }
        if let Some(pixel) = x_coordinates.iter().position(|value| !value.is_finite()) {
            return Err(ShackHartmannError::NonFiniteX { pixel });
        }
        if let Some(pixel) = y_coordinates.iter().position(|value| !value.is_finite()) {
            return Err(ShackHartmannError::NonFiniteY { pixel });
        }
        if let Some(slope) = reference_slopes.iter().position(|value| !value.is_finite()) {
            return Err(ShackHartmannError::NonFiniteReference { slope });
        }
        if let Some(subaperture) = pixel_thresholds
            .iter()
            .position(|value| !value.is_finite() || *value < T::zero())
        {
            return Err(ShackHartmannError::InvalidPixelThreshold { subaperture });
        }
        if let Some(subaperture) = flux_thresholds
            .iter()
            .position(|value| !value.is_finite() || *value < T::zero())
        {
            return Err(ShackHartmannError::InvalidFluxThreshold { subaperture });
        }

        Ok(Self {
            x_coordinates,
            y_coordinates,
            reference_slopes,
            pixel_thresholds,
            flux_thresholds,
            active,
        })
    }

    /// Returns X gradient coordinates in row-major region order.
    pub fn x_coordinates(&self) -> &[T] {
        &self.x_coordinates
    }

    /// Returns Y gradient coordinates in row-major region order.
    pub fn y_coordinates(&self) -> &[T] {
        &self.y_coordinates
    }

    /// Returns pair-interleaved reference slopes.
    pub fn reference_slopes(&self) -> &[T] {
        &self.reference_slopes
    }

    /// Returns absolute pixel thresholds in subaperture order.
    pub fn pixel_thresholds(&self) -> &[T] {
        &self.pixel_thresholds
    }

    /// Returns flux thresholds in subaperture order.
    pub fn flux_thresholds(&self) -> &[T] {
        &self.flux_thresholds
    }

    /// Returns prepared active state in subaperture order.
    pub fn active(&self) -> &[bool] {
        &self.active
    }

    /// Returns pixels in one subaperture.
    pub fn pixel_count(&self) -> usize {
        self.x_coordinates.len()
    }

    /// Returns the prepared subaperture extent.
    pub fn len(&self) -> usize {
        self.pixel_thresholds.len()
    }

    /// Returns `false`; a prepared plan always contains subapertures.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Creates idle progressive sequence state.
    pub const fn workspace(&self) -> ProgressWorkspace {
        ProgressWorkspace::new()
    }

    fn validate(
        &self,
        output: &ShackHartmannOutput<'_, T>,
        input: &RegionPixels<'_, T>,
    ) -> Result<(), ProcessError> {
        if input.pixel_count() != self.pixel_count() {
            return Err(ProcessError::InputExtent {
                expected: self.pixel_count(),
                actual: input.pixel_count(),
            });
        }
        if input.region_count() != self.len() {
            return Err(ProcessError::InputExtent {
                expected: self.len(),
                actual: input.region_count(),
            });
        }
        if output.len() != self.len() {
            return Err(ProcessError::OutputExtent {
                expected: self.len(),
                actual: output.len(),
            });
        }
        Ok(())
    }

    fn process_subapertures(
        &self,
        output: &mut ShackHartmannOutput<'_, T>,
        input: RegionPixels<'_, T>,
        workrange: Range<usize>,
    ) {
        for subaperture in workrange {
            let pixel_threshold = self.pixel_thresholds[subaperture];
            let slope_index = 2 * subaperture;
            let measurement = measure_center_of_gravity(
                self.pixel_count(),
                |pixel| input.get(pixel, subaperture),
                &self.x_coordinates,
                &self.y_coordinates,
                [
                    self.reference_slopes[slope_index],
                    self.reference_slopes[slope_index + 1],
                ],
                Some(pixel_threshold),
                self.flux_thresholds[subaperture],
            );
            output.flux[subaperture] = measurement.flux;
            if measurement.validity && self.active[subaperture] {
                output.slopes[slope_index] = measurement.centroid_offset[0];
                output.slopes[slope_index + 1] = measurement.centroid_offset[1];
            } else {
                output.slopes[slope_index] = T::zero();
                output.slopes[slope_index + 1] = T::zero();
            }
            output.validity[subaperture] = measurement.validity;
        }
    }

    fn validate_image_regions(
        &self,
        output: &ShackHartmannOutput<'_, T>,
        input: &ImageRegionPixels<'_, '_, T>,
    ) -> Result<(), ProcessError> {
        if input.pixel_count() != self.pixel_count() {
            return Err(ProcessError::InputExtent {
                expected: self.pixel_count(),
                actual: input.pixel_count(),
            });
        }
        if input.region_count() != self.len() {
            return Err(ProcessError::InputExtent {
                expected: self.len(),
                actual: input.region_count(),
            });
        }
        if output.len() != self.len() {
            return Err(ProcessError::OutputExtent {
                expected: self.len(),
                actual: output.len(),
            });
        }
        Ok(())
    }

    fn process_image_subapertures(
        &self,
        output: &mut ShackHartmannOutput<'_, T>,
        input: &ImageRegionPixels<'_, '_, T>,
        mut workrange: Range<usize>,
        output_first: usize,
    ) {
        if cfg!(target_feature = "avx2") && input.has_four_adjacent_regions() {
            let mut scalar_start = workrange.start;
            while workrange.len() >= 4 {
                let first = workrange.start;
                let Some(rows) = input.four_adjacent_region_rows(first) else {
                    workrange.start += 1;
                    continue;
                };
                self.process_image_subapertures_scalar(
                    output,
                    input,
                    scalar_start..first,
                    output_first,
                );
                self.process_four_adjacent_regions(
                    output,
                    rows,
                    input.region_size().1,
                    first,
                    output_first,
                );
                workrange.start += 4;
                scalar_start = workrange.start;
            }
            workrange.start = scalar_start;
        }
        self.process_image_subapertures_scalar(output, input, workrange, output_first);
    }

    fn process_image_subapertures_scalar(
        &self,
        output: &mut ShackHartmannOutput<'_, T>,
        input: &ImageRegionPixels<'_, '_, T>,
        workrange: Range<usize>,
        output_first: usize,
    ) {
        let region_rows = input.region_size().0;
        for subaperture in workrange {
            let slope_index = 2 * subaperture;
            let measurement = measure_center_of_gravity_rows(
                (0..region_rows).map(|region_row| input.row(subaperture, region_row)),
                &self.x_coordinates,
                &self.y_coordinates,
                [
                    self.reference_slopes[slope_index],
                    self.reference_slopes[slope_index + 1],
                ],
                Some(self.pixel_thresholds[subaperture]),
                self.flux_thresholds[subaperture],
            );
            self.write_image_measurement(output, output_first, subaperture, measurement);
        }
    }

    fn process_four_adjacent_regions<'pixels>(
        &self,
        output: &mut ShackHartmannOutput<'_, T>,
        rows: impl Iterator<Item = &'pixels [T]>,
        region_columns: usize,
        subaperture: usize,
        output_first: usize,
    ) where
        T: 'pixels,
    {
        const LANES: usize = 4;
        // Each lane visits one subaperture in detector row and column order.
        let mut x = [T::zero(); LANES];
        let mut y = [T::zero(); LANES];
        let mut flux = [T::zero(); LANES];
        let thresholds = &self.pixel_thresholds[subaperture..subaperture + LANES];
        for (row, pixels) in rows.enumerate() {
            let start = row * region_columns;
            let end = start + region_columns;
            let x_coordinates = &self.x_coordinates[start..end];
            let y_coordinates = &self.y_coordinates[start..end];
            for (column, (&cx, &cy)) in x_coordinates.iter().zip(y_coordinates).enumerate() {
                let values: [T; LANES] =
                    core::array::from_fn(|lane| pixels[lane * region_columns + column]);
                let selected: [bool; LANES] = core::array::from_fn(|lane| {
                    !values[lane].is_finite() | (values[lane] >= thresholds[lane])
                });
                // Rejected finite pixels must not overflow a coordinate product.
                // The scalar pass starts all three accumulators at positive zero.
                let selected_values: [T; LANES] = core::array::from_fn(|lane| {
                    if selected[lane] {
                        values[lane]
                    } else {
                        T::zero()
                    }
                });
                for lane in 0..LANES {
                    x[lane] = x[lane] + selected_values[lane] * cx;
                    y[lane] = y[lane] + selected_values[lane] * cy;
                    flux[lane] = flux[lane] + selected_values[lane];
                }
            }
        }
        for lane in 0..LANES {
            let index = subaperture + lane;
            let measurement = finish_center_of_gravity(
                x[lane],
                y[lane],
                flux[lane],
                [
                    self.reference_slopes[2 * index],
                    self.reference_slopes[2 * index + 1],
                ],
                self.flux_thresholds[index],
            );
            self.write_image_measurement(output, output_first, index, measurement);
        }
    }

    #[inline]
    fn write_image_measurement(
        &self,
        output: &mut ShackHartmannOutput<'_, T>,
        output_first: usize,
        subaperture: usize,
        measurement: CenterOfGravityMeasurement<T>,
    ) {
        let index = subaperture - output_first;
        output.flux[index] = measurement.flux;
        let offset = if measurement.validity && self.active[subaperture] {
            measurement.centroid_offset
        } else {
            [T::zero(); 2]
        };
        output.slopes[2 * index..2 * index + 2].copy_from_slice(&offset);
        output.validity[index] = measurement.validity;
    }

    /// Measures every subaperture directly from borrowed detector-image rows.
    ///
    /// The image-backed regions must have the same pixel and subaperture
    /// extents as this plan. Processing performs no allocation and does not
    /// materialize a region-major pixel slab. Rectangular region dimensions
    /// may be arbitrary positive values and need not be powers of two.
    pub fn process_image_regions(
        &self,
        mut output: ShackHartmannOutput<'_, T>,
        workspace: &mut ProgressWorkspace,
        input: ImageRegionPixels<'_, '_, T>,
    ) -> Result<(), ProcessError> {
        workspace.ensure_idle()?;
        self.validate_image_regions(&output, &input)?;
        self.process_image_subapertures(&mut output, &input, 0..self.len(), 0);
        Ok(())
    }

    pub(crate) fn process_image_range(
        &self,
        mut output: ShackHartmannOutput<'_, T>,
        input: ImageRegionPixels<'_, '_, T>,
        workrange: Range<usize>,
    ) -> Result<(), ProcessError> {
        if workrange.is_empty() {
            return Err(ProcessError::EmptyRange);
        }
        if workrange.end > self.len() {
            return Err(ProcessError::RangeOutOfBounds {
                end: workrange.end,
                extent: self.len(),
            });
        }
        self.validate_image_regions(&output, &input)?;
        self.process_image_subapertures(&mut output, &input, workrange, 0);
        Ok(())
    }

    pub(crate) fn process_image_partition(
        &self,
        mut output: ShackHartmannOutput<'_, T>,
        input: ImageRegionPixels<'_, '_, T>,
        first_subaperture: usize,
    ) -> Result<(), ProcessError> {
        let end =
            first_subaperture
                .checked_add(output.len())
                .ok_or(ProcessError::RangeOutOfBounds {
                    end: usize::MAX,
                    extent: self.len(),
                })?;
        if end > self.len() {
            return Err(ProcessError::RangeOutOfBounds {
                end,
                extent: self.len(),
            });
        }
        if input.pixel_count() != self.pixel_count() {
            return Err(ProcessError::InputExtent {
                expected: self.pixel_count(),
                actual: input.pixel_count(),
            });
        }
        if input.region_count() != self.len() {
            return Err(ProcessError::InputExtent {
                expected: self.len(),
                actual: input.region_count(),
            });
        }
        self.process_image_subapertures(
            &mut output,
            &input,
            first_subaperture..end,
            first_subaperture,
        );
        Ok(())
    }
}

impl<T> AlgorithmPlan for ShackHartmannPlan<T>
where
    T: Float + 'static,
{
    type Input<'a> = RegionPixels<'a, T>;
    type Output<'a> = ShackHartmannOutput<'a, T>;
    type Workspace = ProgressWorkspace;
    type Error = ProcessError;

    fn process<'input, 'output>(
        &self,
        mut output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        workspace.ensure_idle()?;
        self.validate(&output, &input)?;
        self.process_subapertures(&mut output, input, 0..self.len());
        Ok(())
    }
}

impl<T> InputProgress for ShackHartmannPlan<T>
where
    T: Float + 'static,
{
    fn start(&self, workspace: &mut Self::Workspace) -> Result<(), Self::Error> {
        workspace.start()
    }

    fn process_range<'input, 'output>(
        &self,
        mut output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
        workrange: Range<usize>,
    ) -> Result<ProgressResult, Self::Error> {
        workspace.validate_range(&workrange, self.len())?;
        self.validate(&output, &input)?;

        let immutable_count = workrange.end;
        self.process_subapertures(&mut output, input, workrange);
        workspace.advance(immutable_count);
        Ok(ProgressResult::new(immutable_count))
    }

    fn finish<'output>(
        &self,
        _output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
    ) -> Result<(), Self::Error> {
        workspace.finish(self.len())
    }

    fn abort(&self, workspace: &mut Self::Workspace) -> Result<(), Self::Error> {
        workspace.abort()
    }
}

impl<T> OutputProgress for ShackHartmannPlan<T> where T: Float + 'static {}
