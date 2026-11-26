//! Prepared extraction of calibrated rectangular detector regions of interest.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::ops::Range;

use num_traits::Float;

use crate::{
    AlgorithmPlan, InputProgress, OutputProgress, ProcessError, ProgressResult, ProgressWorkspace,
};

/// Invalid region-pixel slab representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegionPixelsError {
    /// Each region must contain at least one pixel.
    EmptyRegion,

    /// The slab must contain at least one region.
    EmptySlab,

    /// Physical column stride is smaller than the valid pixel extent.
    StrideTooSmall {
        /// Valid pixels in one region.
        pixel_count: usize,
        /// Supplied physical column stride.
        stride: usize,
    },

    /// The prepared layout cannot be represented by `usize`.
    LayoutOverflow,

    /// Caller-owned storage cannot contain the prepared slab layout.
    StorageTooSmall {
        /// Minimum required storage extent.
        required: usize,
        /// Supplied storage extent.
        actual: usize,
    },
}

impl fmt::Display for RegionPixelsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRegion => formatter.write_str("each region must contain pixels"),
            Self::EmptySlab => formatter.write_str("at least one region is required"),
            Self::StrideTooSmall {
                pixel_count,
                stride,
            } => write!(
                formatter,
                "region stride {stride} is smaller than pixel count {pixel_count}"
            ),
            Self::LayoutOverflow => formatter.write_str("region layout overflows usize"),
            Self::StorageTooSmall { required, actual } => write!(
                formatter,
                "storage extent {actual} is smaller than required extent {required}"
            ),
        }
    }
}

impl core::error::Error for RegionPixelsError {}

fn slab_extent(
    storage_len: usize,
    pixel_count: usize,
    region_count: usize,
    stride: usize,
) -> Result<(), RegionPixelsError> {
    if pixel_count == 0 {
        return Err(RegionPixelsError::EmptyRegion);
    }
    if region_count == 0 {
        return Err(RegionPixelsError::EmptySlab);
    }
    if stride < pixel_count {
        return Err(RegionPixelsError::StrideTooSmall {
            pixel_count,
            stride,
        });
    }
    let required = (region_count - 1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(pixel_count))
        .ok_or(RegionPixelsError::LayoutOverflow)?;
    if storage_len < required {
        return Err(RegionPixelsError::StorageTooSmall {
            required,
            actual: storage_len,
        });
    }
    Ok(())
}

/// Borrowed region-major pixel slab.
///
/// Pixels for one region are contiguous and row-major. `stride` is the
/// physical distance between region starts and can include unused tail
/// padding.
#[derive(Clone, Copy, Debug)]
pub struct RegionPixels<'a, T> {
    data: &'a [T],
    pixel_count: usize,
    region_count: usize,
    stride: usize,
}

impl<'a, T> RegionPixels<'a, T> {
    /// Borrows a contiguous region-major slab.
    pub fn new(
        data: &'a [T],
        pixel_count: usize,
        region_count: usize,
    ) -> Result<Self, RegionPixelsError> {
        Self::with_stride(data, pixel_count, region_count, pixel_count)
    }

    /// Borrows a slab with an explicit physical region stride.
    pub fn with_stride(
        data: &'a [T],
        pixel_count: usize,
        region_count: usize,
        stride: usize,
    ) -> Result<Self, RegionPixelsError> {
        slab_extent(data.len(), pixel_count, region_count, stride)?;
        Ok(Self {
            data,
            pixel_count,
            region_count,
            stride,
        })
    }

    /// Returns the complete caller-owned backing storage.
    pub const fn data(&self) -> &'a [T] {
        self.data
    }

    /// Returns the valid pixels in one region.
    pub const fn pixel_count(&self) -> usize {
        self.pixel_count
    }

    /// Returns the number of regions.
    pub const fn region_count(&self) -> usize {
        self.region_count
    }

    /// Returns the physical distance between region starts.
    pub const fn stride(&self) -> usize {
        self.stride
    }

    pub(crate) fn get(&self, pixel: usize, region: usize) -> T
    where
        T: Copy,
    {
        self.data[region * self.stride + pixel]
    }
}

/// Mutably borrowed region-major pixel slab.
///
/// This is the caller-owned destination counterpart of [`RegionPixels`].
#[derive(Debug)]
pub struct RegionPixelsMut<'a, T> {
    data: &'a mut [T],
    pixel_count: usize,
    region_count: usize,
    stride: usize,
}

/// Borrowed rectangular region pixels backed directly by a detector image.
///
/// Unlike [`RegionPixels`], this view does not require the regions to be
/// copied into a region-major slab. Each region row remains a contiguous Rust
/// slice, while successive rows are separated by the prepared detector-row
/// stride. Region and row dimensions may be any positive values; the layout
/// does not require powers of two.
#[derive(Clone, Copy, Debug)]
pub struct ImageRegionPixels<'geometry, 'pixels, T> {
    pixels: &'pixels [T],
    origins: &'geometry [(usize, usize)],
    region_size: (usize, usize),
    row_stride: usize,
    first_pixel: usize,
    has_four_adjacent_regions: bool,
}

impl<'geometry, 'pixels, T> ImageRegionPixels<'geometry, 'pixels, T> {
    /// Returns the rectangular region size in `(rows, columns)` order.
    pub const fn region_size(&self) -> (usize, usize) {
        self.region_size
    }

    /// Returns the number of image-backed regions.
    pub const fn region_count(&self) -> usize {
        self.origins.len()
    }

    /// Returns the number of pixels in each rectangular region.
    pub const fn pixel_count(&self) -> usize {
        self.region_size.0 * self.region_size.1
    }

    pub(crate) fn row(&self, region: usize, region_row: usize) -> &'pixels [T] {
        let (first_row, first_column) = self.origins[region];
        let row_start =
            self.first_pixel + (first_row + region_row) * self.row_stride + first_column;
        &self.pixels[row_start..row_start + self.region_size.1]
    }

    pub(crate) const fn has_four_adjacent_regions(&self) -> bool {
        self.has_four_adjacent_regions
    }

    pub(crate) fn four_adjacent_region_rows(
        &self,
        first_region: usize,
    ) -> Option<impl Iterator<Item = &'pixels [T]>> {
        let (rows, columns) = self.region_size;
        let origins = &self.origins[first_region..first_region + 4];
        let (row, column) = origins[0];
        if !adjacent(origins, columns) {
            return None;
        }
        // Each individual region was checked at preparation. The last
        // region's right edge bounds the joined row.
        Some((0..rows).map(move |offset| {
            let start = self.first_pixel + (row + offset) * self.row_stride + column;
            &self.pixels[start..start + 4 * columns]
        }))
    }
}

fn adjacent(origins: &[(usize, usize)], columns: usize) -> bool {
    origins
        .windows(2)
        .all(|pair| pair[1].0 == pair[0].0 && pair[1].1.checked_sub(pair[0].1) == Some(columns))
}

impl<'a, T> RegionPixelsMut<'a, T> {
    /// Borrows a contiguous region-major destination slab.
    pub fn new(
        data: &'a mut [T],
        pixel_count: usize,
        region_count: usize,
    ) -> Result<Self, RegionPixelsError> {
        Self::with_stride(data, pixel_count, region_count, pixel_count)
    }

    /// Borrows a destination slab with an explicit physical region stride.
    pub fn with_stride(
        data: &'a mut [T],
        pixel_count: usize,
        region_count: usize,
        stride: usize,
    ) -> Result<Self, RegionPixelsError> {
        slab_extent(data.len(), pixel_count, region_count, stride)?;
        Ok(Self {
            data,
            pixel_count,
            region_count,
            stride,
        })
    }

    /// Returns the complete caller-owned backing storage.
    pub fn data(&self) -> &[T] {
        self.data
    }

    /// Returns the complete caller-owned backing storage mutably.
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    /// Returns the valid pixels in one region.
    pub const fn pixel_count(&self) -> usize {
        self.pixel_count
    }

    /// Returns the number of regions.
    pub const fn region_count(&self) -> usize {
        self.region_count
    }

    /// Returns the physical distance between region starts.
    pub const fn stride(&self) -> usize {
        self.stride
    }
}

/// Invalid prepared region-extraction geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegionExtractionError {
    /// Logical detector image has a zero dimension.
    EmptyImage,

    /// Rectangular regions have a zero dimension.
    EmptyRegion,

    /// No region origin was supplied.
    EmptyOrigins,

    /// A flat origin update does not contain one row/column pair per region.
    OriginCoordinateExtent {
        /// Required number of scalar row/column coordinates.
        expected: usize,
        /// Supplied number of scalar row/column coordinates.
        actual: usize,
    },

    /// Physical row stride cannot contain the logical image width.
    RowStrideTooSmall,

    /// A rectangular region lies outside the logical detector image.
    RegionOutOfBounds {
        /// Zero-based region index.
        region: usize,
    },

    /// Prepared detector geometry cannot be represented by `usize`.
    LayoutOverflow,
}

impl fmt::Display for RegionExtractionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage => formatter.write_str("image dimensions must be positive"),
            Self::EmptyRegion => formatter.write_str("region dimensions must be positive"),
            Self::EmptyOrigins => formatter.write_str("at least one region is required"),
            Self::OriginCoordinateExtent { expected, actual } => write!(
                formatter,
                "region-origin coordinate extent {actual} does not match {expected}"
            ),
            Self::RowStrideTooSmall => {
                formatter.write_str("row stride must contain every logical image column")
            }
            Self::RegionOutOfBounds { region } => {
                write!(formatter, "region {region} lies outside the logical image")
            }
            Self::LayoutOverflow => formatter.write_str("detector layout overflows usize"),
        }
    }
}

impl core::error::Error for RegionExtractionError {}

fn image_index(
    first_pixel: usize,
    row_stride: usize,
    row: usize,
    column: usize,
) -> Result<usize, RegionExtractionError> {
    row.checked_mul(row_stride)
        .and_then(|offset| offset.checked_add(column))
        .and_then(|offset| first_pixel.checked_add(offset))
        .ok_or(RegionExtractionError::LayoutOverflow)
}

/// Prepared extraction of rectangular regions from a calibrated detector
/// image.
///
/// `image_size`, `region_size`, and each origin use `(row, column)` order and
/// Rust's zero-based indexing. `first_pixel` is the zero-based input-storage
/// index of logical detector pixel `(0, 0)`. `completion_indices` contains the
/// greatest zero-based input-storage index needed by each rectangular region.
/// Origin order is the semantic output-region order and is preserved; the plan
/// does not reorder regions for detector traversal.
/// Pixel calibration is an independent operation and is not owned by this
/// plan.
#[derive(Clone, Debug)]
pub struct RegionExtractionPlan<T> {
    origins: Box<[(usize, usize)]>,
    completion_indices: Box<[usize]>,
    image_size: (usize, usize),
    region_size: (usize, usize),
    row_stride: usize,
    first_pixel: usize,
    image_extent: usize,
    has_four_adjacent_regions: bool,
    // Associates the geometry with its pixel representation without storing
    // or owning a pixel.
    pixel: PhantomData<fn(T)>,
}

impl<T> RegionExtractionPlan<T>
where
    T: Float,
{
    /// Prepares extraction from a compact row-major detector buffer.
    pub fn new(
        image_size: (usize, usize),
        origins: &[(usize, usize)],
        region_size: (usize, usize),
    ) -> Result<Self, RegionExtractionError> {
        Self::with_layout(image_size, origins, region_size, image_size.1, 0)
    }

    /// Prepares extraction with explicit physical detector layout.
    pub fn with_layout(
        image_size: (usize, usize),
        origins: &[(usize, usize)],
        region_size: (usize, usize),
        row_stride: usize,
        first_pixel: usize,
    ) -> Result<Self, RegionExtractionError> {
        let (image_rows, image_columns) = image_size;
        let (region_rows, region_columns) = region_size;
        if image_rows == 0 || image_columns == 0 {
            return Err(RegionExtractionError::EmptyImage);
        }
        if region_rows == 0 || region_columns == 0 {
            return Err(RegionExtractionError::EmptyRegion);
        }
        if origins.is_empty() {
            return Err(RegionExtractionError::EmptyOrigins);
        }
        if row_stride < image_columns {
            return Err(RegionExtractionError::RowStrideTooSmall);
        }

        image_rows
            .checked_mul(image_columns)
            .ok_or(RegionExtractionError::LayoutOverflow)?;
        let image_extent = image_index(first_pixel, row_stride, image_rows - 1, image_columns - 1)?
            .checked_add(1)
            .ok_or(RegionExtractionError::LayoutOverflow)?;

        let mut completion_indices = Vec::with_capacity(origins.len());
        for (region, &(first_row, first_column)) in origins.iter().enumerate() {
            let last_row = first_row
                .checked_add(region_rows - 1)
                .ok_or(RegionExtractionError::LayoutOverflow)?;
            let last_column = first_column
                .checked_add(region_columns - 1)
                .ok_or(RegionExtractionError::LayoutOverflow)?;
            if last_row >= image_rows || last_column >= image_columns {
                return Err(RegionExtractionError::RegionOutOfBounds { region });
            }
            completion_indices.push(image_index(first_pixel, row_stride, last_row, last_column)?);
        }

        Ok(Self {
            origins: origins.to_vec().into_boxed_slice(),
            completion_indices: completion_indices.into_boxed_slice(),
            image_size,
            region_size,
            row_stride,
            first_pixel,
            image_extent,
            has_four_adjacent_regions: cfg!(target_feature = "avx2")
                && origins
                    .windows(4)
                    .any(|group| adjacent(group, region_columns)),
            pixel: PhantomData,
        })
    }

    /// Returns zero-based `(row, column)` origins in region order.
    pub fn origins(&self) -> &[(usize, usize)] {
        &self.origins
    }

    /// Returns the greatest zero-based input index required by each region.
    pub fn completion_indices(&self) -> &[usize] {
        &self.completion_indices
    }

    /// Returns logical detector size in `(rows, columns)` order.
    pub const fn image_size(&self) -> (usize, usize) {
        self.image_size
    }

    /// Returns rectangular region size in `(rows, columns)` order.
    pub const fn region_size(&self) -> (usize, usize) {
        self.region_size
    }

    /// Returns physical detector-row stride in elements.
    pub const fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Returns the zero-based input index of logical detector pixel `(0, 0)`.
    pub const fn first_pixel(&self) -> usize {
        self.first_pixel
    }

    /// Returns the minimum physical input extent required by this layout.
    pub const fn image_extent(&self) -> usize {
        self.image_extent
    }

    /// Returns the number of prepared regions.
    pub const fn len(&self) -> usize {
        self.origins.len()
    }

    /// Returns `false`; valid extraction requires at least one region.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Returns valid pixels in one rectangular region.
    pub const fn pixels_per_region(&self) -> usize {
        self.region_size.0 * self.region_size.1
    }

    /// Rebuilds this packed-image extraction with one flat U32 row/column
    /// origin pair per existing region.
    pub fn with_u32_origins(&self, values: &[u32]) -> Result<Self, RegionExtractionError> {
        let expected = self
            .len()
            .checked_mul(2)
            .ok_or(RegionExtractionError::LayoutOverflow)?;
        if values.len() != expected {
            return Err(RegionExtractionError::OriginCoordinateExtent {
                expected,
                actual: values.len(),
            });
        }
        let origins = values
            .chunks_exact(2)
            .map(|pair| (pair[0] as usize, pair[1] as usize))
            .collect::<Vec<_>>();
        Self::new(self.image_size, &origins, self.region_size)
    }

    /// Borrows the prepared rectangular regions directly from an image.
    ///
    /// Construction validates the complete physical image extent once. The
    /// returned view borrows both the prepared geometry and caller-owned pixel
    /// storage and does not allocate or copy pixels.
    pub fn image_regions<'geometry, 'pixels>(
        &'geometry self,
        calibrated: &'pixels [T],
    ) -> Result<ImageRegionPixels<'geometry, 'pixels, T>, ProcessError> {
        self.validate_input(calibrated.len())?;
        Ok(ImageRegionPixels {
            pixels: calibrated,
            origins: &self.origins,
            region_size: self.region_size,
            row_stride: self.row_stride,
            first_pixel: self.first_pixel,
            has_four_adjacent_regions: self.has_four_adjacent_regions,
        })
    }

    /// Creates idle progressive sequence state.
    pub const fn workspace(&self) -> ProgressWorkspace {
        ProgressWorkspace::new()
    }

    fn validate(
        &self,
        output: &RegionPixelsMut<'_, T>,
        input_len: usize,
    ) -> Result<(), ProcessError> {
        self.validate_input(input_len)?;
        if output.pixel_count != self.pixels_per_region() {
            return Err(ProcessError::OutputExtent {
                expected: self.pixels_per_region(),
                actual: output.pixel_count,
            });
        }
        if output.region_count != self.len() {
            return Err(ProcessError::OutputExtent {
                expected: self.len(),
                actual: output.region_count,
            });
        }
        Ok(())
    }

    fn validate_input(&self, input_len: usize) -> Result<(), ProcessError> {
        if input_len < self.image_extent {
            return Err(ProcessError::InputExtent {
                expected: self.image_extent,
                actual: input_len,
            });
        }
        Ok(())
    }

    fn extract(
        &self,
        output: &mut RegionPixelsMut<'_, T>,
        calibrated: &[T],
        workrange: Range<usize>,
    ) {
        let (region_rows, region_columns) = self.region_size;

        for region in workrange {
            let (first_row, first_column) = self.origins[region];
            for region_row in 0..region_rows {
                let image_row = first_row + region_row;
                let input_start = self.first_pixel + image_row * self.row_stride + first_column;
                let output_start = region_row * region_columns;
                let destination_start = region * output.stride + output_start;
                output.data[destination_start..destination_start + region_columns]
                    .copy_from_slice(&calibrated[input_start..input_start + region_columns]);
            }
        }
    }
}

impl<T> AlgorithmPlan for RegionExtractionPlan<T>
where
    T: Float + 'static,
{
    type Input<'a> = &'a [T];
    type Output<'a> = RegionPixelsMut<'a, T>;
    type Workspace = ProgressWorkspace;
    type Error = ProcessError;

    fn process<'input, 'output>(
        &self,
        mut output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        workspace.ensure_idle()?;
        self.validate(&output, input.len())?;
        self.extract(&mut output, input, 0..self.len());
        Ok(())
    }
}

impl<T> InputProgress for RegionExtractionPlan<T>
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
        self.validate(&output, input.len())?;

        let immutable_count = workrange.end;
        self.extract(&mut output, input, workrange);
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

impl<T> OutputProgress for RegionExtractionPlan<T> where T: Float + 'static {}

crate::filtergraph_algorithm! {
    /// Portable Float32 packed-image rectangular-region extraction declaration.
    pub RegionExtractionF32 {
        label = "region-extraction-f32";
        configuration RegionExtractionF32Configuration {
            image_size: (usize, usize) = GraphRebuild;
            region_size: (usize, usize) = GraphRebuild;
            origins: Vec<[usize; 2]> = GraphRebuild;
        }
        plan = RegionExtractionPlan<f32>;
        workspace = ProgressWorkspace;
        properties = none;
        dimensions |plan, _workspace| {
            image_rows = plan.image_size().0;
            image_columns = plan.image_size().1;
            region_rows = plan.region_size().0;
            region_columns = plan.region_size().1;
            region_count = plan.len();
        }
        inputs {
            calibrated: f32 [image_rows, image_columns] = "calibrated",
                schema = crate::schemas::CALIBRATED_PIXELS_V1;
        }
        outputs {
            regions: f32 [region_count, region_rows, region_columns] = "regions",
                schema = crate::schemas::REGION_PIXELS_V1,
                metadata_from = "calibrated";
        }
        parameters {
            origins: u32 [region_count, 2] = "origins",
                schema = crate::schemas::REGION_ORIGINS_V1,
                replace |plan, values| {
                    plan.with_u32_origins(values)
                };
        }
        parameter_error = RegionExtractionError;
        prepare |configuration| {
            let origins = configuration
                .origins
                .iter()
                .map(|origin| (origin[0], origin[1]))
                .collect::<Vec<_>>();
            let plan = RegionExtractionPlan::new(
                configuration.image_size,
                &origins,
                configuration.region_size,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = RegionExtractionError;
        process |plan, workspace| {
            let expected = plan.len() * plan.pixels_per_region();
            let actual = regions.len();
            if actual != expected {
                return Err(ProcessError::OutputExtent {
                    expected,
                    actual,
                });
            }
            let output = RegionPixelsMut::new(
                regions,
                plan.pixels_per_region(),
                plan.len(),
            ).map_err(|_| ProcessError::OutputExtent {
                expected,
                actual,
            })?;
            plan.process(output, workspace, calibrated)
        }
        process_error = ProcessError;
        reset |plan, workspace| {
            *workspace = plan.workspace();
        }
    }
}
