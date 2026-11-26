//! Image-backed pyramid pupil-pixel normalization without a pupil slab.

use alloc::{boxed::Box, vec::Vec};
use core::{fmt, ops::Range};

use num_traits::{Float, NumCast};

use crate::{
    AlgorithmPlan, PYRAMID_PUPIL_COUNT, ProcessError, PyramidPixelError, PyramidPixelOutput,
    PyramidPixelPlan, PyramidPixelPlanError, PyramidPixelWorkspace, RegionExtractionError,
    RegionExtractionPlan,
};

/// Invalid preparation or sparse parameter replacement for an image-backed
/// pyramid measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPixelImagePreparationError {
    /// Detector pupil-region geometry is invalid.
    Regions(RegionExtractionError),
    /// The shared pupil mask is invalid.
    Normalization(PyramidPixelPlanError),
    /// Pupil geometry and normalization are incompatible.
    Composition(PyramidPixelImagePlanError),
    /// A sparse origin array does not contain four row/column pairs.
    OriginExtent {
        /// Required flattened element extent.
        expected: usize,
        /// Supplied flattened element extent.
        actual: usize,
    },
    /// A portable `u32` origin cannot be represented by the current target.
    OriginCoordinateOverflow,
    /// A replacement mask would change the reconstruction-pixel port shape.
    SelectedPixelExtent {
        /// Prepared reconstruction-pixel extent.
        expected: usize,
        /// Replacement reconstruction-pixel extent.
        actual: usize,
    },
}

impl fmt::Display for PyramidPixelImagePreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Regions(error) => error.fmt(formatter),
            Self::Normalization(error) => error.fmt(formatter),
            Self::Composition(error) => error.fmt(formatter),
            Self::OriginExtent { expected, actual } => write!(
                formatter,
                "flattened pupil-origin extent {actual} does not match {expected}"
            ),
            Self::OriginCoordinateOverflow => {
                formatter.write_str("a pupil origin does not fit usize")
            }
            Self::SelectedPixelExtent { expected, actual } => write!(
                formatter,
                "replacement mask selects {actual} pixels, expected {expected}"
            ),
        }
    }
}

impl core::error::Error for PyramidPixelImagePreparationError {}

impl From<RegionExtractionError> for PyramidPixelImagePreparationError {
    fn from(error: RegionExtractionError) -> Self {
        Self::Regions(error)
    }
}

impl From<PyramidPixelPlanError> for PyramidPixelImagePreparationError {
    fn from(error: PyramidPixelPlanError) -> Self {
        Self::Normalization(error)
    }
}

impl From<PyramidPixelImagePlanError> for PyramidPixelImagePreparationError {
    fn from(error: PyramidPixelImagePlanError) -> Self {
        Self::Composition(error)
    }
}

/// Invalid caller output or numerical processing for the portable image
/// declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPixelImageProcessError {
    /// The mean-pupil-intensity output is not one scalar.
    MeanOutputExtent,
    /// The prepared scientific operation rejected the call.
    Algorithm(PyramidPixelImageError),
}

impl fmt::Display for PyramidPixelImageProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeanOutputExtent => {
                formatter.write_str("mean-pupil-intensity output must contain one value")
            }
            Self::Algorithm(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PyramidPixelImageProcessError {}

/// Incompatible prepared pupil-region and pyramid-pixel plans.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPixelImagePlanError {
    /// Pyramid processing requires exactly four pupil image regions.
    PupilExtent {
        /// Supplied image-region count.
        actual: usize,
    },
    /// One pupil region and the shared reconstruction mask use different pixel
    /// extents.
    PupilPixelExtent {
        /// Pixels in one pupil image region.
        region_pixels: usize,
        /// Pixels in the shared pupil mask.
        mask_pixels: usize,
    },
}

impl fmt::Display for PyramidPixelImagePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PupilExtent { actual } => write!(
                formatter,
                "pyramid image geometry contains {actual} pupils, expected {PYRAMID_PUPIL_COUNT}"
            ),
            Self::PupilPixelExtent {
                region_pixels,
                mask_pixels,
            } => write!(
                formatter,
                "pupil image region extent {region_pixels} does not match mask extent {mask_pixels}"
            ),
        }
    }
}

impl core::error::Error for PyramidPixelImagePlanError {}

/// Image-view construction or pyramid-pixel processing failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPixelImageError {
    /// The detector input is incompatible with the prepared image geometry.
    Image(ProcessError),
    /// The pyramid-pixel operation rejected its input, output, or result.
    Pyramid(PyramidPixelError),
}

impl fmt::Display for PyramidPixelImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(error) => error.fmt(formatter),
            Self::Pyramid(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PyramidPixelImageError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Image(error) => Some(error),
            Self::Pyramid(error) => Some(error),
        }
    }
}

/// Prepared detector-image to pyramid reconstruction-pixel composition.
///
/// The composition is an ordinary typed [`AlgorithmPlan`]. It retains pupil
/// geometry and normalization as scientific state while borrowing pupil rows
/// directly from each detector image.
#[derive(Clone, Debug)]
pub struct PyramidPixelImagePlan<T> {
    pupil_regions: RegionExtractionPlan<T>,
    normalization: PyramidPixelPlan<T>,
    selected_runs: PupilRuns,
}

impl<T> PyramidPixelImagePlan<T>
where
    T: Float + NumCast,
{
    /// Composes compatible pupil-image geometry and normalization.
    pub fn new(
        pupil_regions: RegionExtractionPlan<T>,
        normalization: PyramidPixelPlan<T>,
    ) -> Result<Self, PyramidPixelImagePlanError> {
        if pupil_regions.len() != PYRAMID_PUPIL_COUNT {
            return Err(PyramidPixelImagePlanError::PupilExtent {
                actual: pupil_regions.len(),
            });
        }
        if pupil_regions.pixels_per_region() != normalization.pupil_pixel_count() {
            return Err(PyramidPixelImagePlanError::PupilPixelExtent {
                region_pixels: pupil_regions.pixels_per_region(),
                mask_pixels: normalization.pupil_pixel_count(),
            });
        }
        let selected_runs = PupilRuns::new(&pupil_regions, normalization.mask());
        Ok(Self {
            selected_runs,
            pupil_regions,
            normalization,
        })
    }

    /// Returns the prepared pupil-image geometry.
    pub const fn pupil_regions(&self) -> &RegionExtractionPlan<T> {
        &self.pupil_regions
    }

    /// Returns the prepared pyramid-pixel normalization.
    pub const fn normalization(&self) -> &PyramidPixelPlan<T> {
        &self.normalization
    }

    /// Creates HEART-compatible committed normalization history.
    pub fn workspace(&self) -> PyramidPixelWorkspace<T> {
        self.normalization.workspace()
    }

    /// Prepare one image-backed operation from detector pupil geometry and a
    /// shared row-major mask.
    pub fn from_detector_pupils(
        image_size: (usize, usize),
        pupil_size: (usize, usize),
        pupil_origins: &[(usize, usize)],
        pupil_mask: Vec<bool>,
    ) -> Result<Self, PyramidPixelImagePreparationError> {
        let pupil_regions = RegionExtractionPlan::new(image_size, pupil_origins, pupil_size)?;
        let normalization = PyramidPixelPlan::new(pupil_mask)?;
        Self::new(pupil_regions, normalization).map_err(Into::into)
    }

    /// Replace four zero-based pupil origins while preserving image layout and
    /// normalization.
    pub fn with_u32_pupil_origins(
        &self,
        pair_interleaved_origins: &[u32],
    ) -> Result<Self, PyramidPixelImagePreparationError> {
        let expected = PYRAMID_PUPIL_COUNT * 2;
        if pair_interleaved_origins.len() != expected {
            return Err(PyramidPixelImagePreparationError::OriginExtent {
                expected,
                actual: pair_interleaved_origins.len(),
            });
        }
        let mut origins = Vec::with_capacity(PYRAMID_PUPIL_COUNT);
        for origin in pair_interleaved_origins.chunks_exact(2) {
            origins.push((
                usize::try_from(origin[0])
                    .map_err(|_| PyramidPixelImagePreparationError::OriginCoordinateOverflow)?,
                usize::try_from(origin[1])
                    .map_err(|_| PyramidPixelImagePreparationError::OriginCoordinateOverflow)?,
            ));
        }
        let pupil_regions = RegionExtractionPlan::with_layout(
            self.pupil_regions.image_size(),
            &origins,
            self.pupil_regions.region_size(),
            self.pupil_regions.row_stride(),
            self.pupil_regions.first_pixel(),
        )?;
        Self::new(pupil_regions, self.normalization.clone()).map_err(Into::into)
    }

    /// Replace the shared pupil mask without changing the prepared output
    /// shape.
    pub fn with_pupil_mask(
        &self,
        pupil_mask: &[bool],
    ) -> Result<Self, PyramidPixelImagePreparationError> {
        let normalization = PyramidPixelPlan::new(pupil_mask.to_vec())?;
        let expected = self.normalization.output_pixel_count();
        let actual = normalization.output_pixel_count();
        if actual != expected {
            return Err(PyramidPixelImagePreparationError::SelectedPixelExtent {
                expected,
                actual,
            });
        }
        Self::new(self.pupil_regions.clone(), normalization).map_err(Into::into)
    }
}

#[derive(Clone, Debug)]
struct PupilRuns {
    bases: [usize; PYRAMID_PUPIL_COUNT],
    runs: Box<[Range<usize>]>,
}

impl PupilRuns {
    fn new<T: Float>(layout: &RegionExtractionPlan<T>, mask: &[bool]) -> Self {
        let stride = layout.row_stride();
        let columns = layout.region_size().1;
        let bases = core::array::from_fn(|pupil| {
            let (row, column) = layout.origins()[pupil];
            layout.first_pixel() + row * stride + column
        });
        let mut runs = Vec::new();
        for (row, mask_row) in mask.chunks_exact(columns).enumerate() {
            let mut start = row * stride;
            for run in mask_row.chunk_by(|left, right| left == right) {
                let end = start + run.len();
                if run[0] {
                    runs.push(start..end);
                }
                start = end;
            }
        }
        Self {
            bases,
            runs: runs.into_boxed_slice(),
        }
    }

    fn write_selected<T: Float>(
        &self,
        mut remaining: &mut [T],
        image: &[T],
        gain: T,
    ) -> Result<(), PyramidPixelError> {
        let mut index = 0;
        for base in self.bases {
            for run in &self.runs {
                let input = &image[base + run.start..base + run.end];
                let (output, rest) = remaining.split_at_mut(input.len());
                write_pupil_run(output, input, gain, index)?;
                index += input.len();
                remaining = rest;
            }
        }
        Ok(())
    }
}

fn write_pupil_run<T: Float>(
    output: &mut [T],
    input: &[T],
    gain: T,
    mut index: usize,
) -> Result<(), PyramidPixelError> {
    const LANES: usize = if cfg!(target_feature = "avx512f") {
        16
    } else if cfg!(target_feature = "avx2") {
        8
    } else {
        4
    };
    let (input_chunks, input_tail) = input.as_chunks::<LANES>();
    let (output_chunks, output_tail) = output.as_chunks_mut::<LANES>();
    for (output, input) in output_chunks.iter_mut().zip(input_chunks) {
        let products = input.map(|value| gain * value);
        let finite = products
            .iter()
            .fold(true, |finite, value| finite & value.is_finite());
        if !finite {
            // Match the scalar pass's first failure and leave its suffix untouched.
            return write_finite_pupil_pixels(output, products, index);
        }
        *output = products;
        index += LANES;
    }
    write_finite_pupil_pixels(
        output_tail,
        input_tail.iter().map(|&value| gain * value),
        index,
    )
}

fn write_finite_pupil_pixels<T: Float>(
    output: &mut [T],
    products: impl IntoIterator<Item = T>,
    start: usize,
) -> Result<(), PyramidPixelError> {
    for (offset, (output, value)) in output.iter_mut().zip(products).enumerate() {
        if !value.is_finite() {
            return Err(PyramidPixelError::NonFiniteOutput {
                index: start + offset,
            });
        }
        *output = value;
    }
    Ok(())
}

impl<T> AlgorithmPlan for PyramidPixelImagePlan<T>
where
    T: Float + NumCast + 'static,
{
    type Input<'a> = &'a [T];
    type Output<'a> = PyramidPixelOutput<'a, T>;
    type Workspace = PyramidPixelWorkspace<T>;
    type Error = PyramidPixelImageError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        let pupils = self
            .pupil_regions
            .image_regions(input)
            .map_err(PyramidPixelImageError::Image)?;
        self.normalization
            .process_pupils_with(output, workspace, &pupils, |output, gain| {
                self.selected_runs.write_selected(output, input, gain)
            })
            .map_err(PyramidPixelImageError::Pyramid)
    }
}

#[cfg(feature = "blas")]
impl PyramidPixelImagePlan<f32> {
    fn process_small_image(
        &self,
        output: PyramidPixelOutput<'_, f32>,
        workspace: &mut PyramidPixelWorkspace<f32>,
        image: &[f32],
    ) -> Result<(), PyramidPixelImageError> {
        let pupils = self
            .pupil_regions
            .image_regions(image)
            .map_err(PyramidPixelImageError::Image)?;
        self.normalization
            .process_small_image(output, workspace, pupils, |output, gain| {
                self.selected_runs.write_selected(output, image, gain)
            })
            .map_err(PyramidPixelImageError::Pyramid)
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 image-backed pyramid-pixel declaration.
    pub PyramidPupilImageF32 {
        label = "pyramid-pupil-image-f32";
        configuration PyramidPupilImageF32Configuration {
            image_size: (usize, usize) = GraphRebuild;
            pupil_size: (usize, usize) = GraphRebuild;
            pupil_origins: Vec<[usize; 2]> = GraphRebuild;
            pupil_mask: Option<Vec<bool>> = Construction;
        }
        plan = PyramidPixelImagePlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            image_rows = plan.pupil_regions().image_size().0;
            image_columns = plan.pupil_regions().image_size().1;
            pupil_rows = plan.pupil_regions().region_size().0;
            pupil_columns = plan.pupil_regions().region_size().1;
            selected_pixel_count = plan.normalization().selected_pixel_count();
        }
        inputs {
            image: f32 [image_rows, image_columns] = "image",
                schema = crate::schemas::CALIBRATED_PIXELS_V1;
        }
        outputs {
            reconstruction_pixels: f32 [PYRAMID_PUPIL_COUNT, selected_pixel_count]
                = "reconstruction-pixels",
                schema = crate::schemas::PYRAMID_RECONSTRUCTION_PIXELS_V1,
                metadata_from = "image";
            mean_pupil_intensity: f32 [1] = "mean-pupil-intensity",
                schema = crate::schemas::PYRAMID_MEAN_PUPIL_INTENSITY_V1,
                metadata_from = "image";
        }
        parameters {
            pupil_origins: u32 [PYRAMID_PUPIL_COUNT, 2] = "pupil-origins",
                schema = crate::schemas::REGION_ORIGINS_V1,
                replace |plan, values| {
                    plan.with_u32_pupil_origins(values)
                };
            pupil_mask: bool [pupil_rows, pupil_columns] = "pupil-mask",
                schema = crate::schemas::PYRAMID_PUPIL_MASK_V1,
                replace |plan, values| {
                    plan.with_pupil_mask(values)
                };
        }
        parameter_error = PyramidPixelImagePreparationError;
        prepare |configuration| {
            let origins = configuration
                .pupil_origins
                .iter()
                .map(|origin| (origin[0], origin[1]))
                .collect::<Vec<_>>();
            let plan = PyramidPixelImagePlan::from_detector_pupils(
                configuration.image_size,
                configuration.pupil_size,
                &origins,
                super::pyramid_measurement_block::configured_pyramid_mask(configuration.pupil_mask, configuration.pupil_size),
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = PyramidPixelImagePreparationError;
        process |plan, workspace| {
            let [mean] = mean_pupil_intensity else {
                return Err(PyramidPixelImageProcessError::MeanOutputExtent);
            };
            let output = PyramidPixelOutput::new(reconstruction_pixels, mean);
            #[cfg(feature = "blas")]
            let result = plan.process_small_image(output, workspace, image);
            #[cfg(not(feature = "blas"))]
            let result = plan.process(output, workspace, image);
            result.map_err(PyramidPixelImageProcessError::Algorithm)
        }
        process_error = PyramidPixelImageProcessError;
        reset |_plan, workspace| {
            *workspace = PyramidPixelWorkspace::new();
        }
    }
}
