//! Prepared pyramid-wavefront-sensor pupil-pixel normalization.

use alloc::sync::Arc;
use core::fmt;

use num_traits::{Float, NumCast};

use crate::{AlgorithmPlan, ImageRegionPixels, RegionPixels};

/// Number of pupil images produced by a pyramid wavefront sensor.
pub const PYRAMID_PUPIL_COUNT: usize = 4;

pub(crate) trait PupilPixels<T> {
    fn pixel_count(&self) -> usize;

    fn pupil_count(&self) -> usize;

    fn try_for_each_row<E>(
        &self,
        pupil: usize,
        row: impl FnMut(&[T]) -> Result<(), E>,
    ) -> Result<(), E>;
}

impl<T> PupilPixels<T> for RegionPixels<'_, T> {
    fn pixel_count(&self) -> usize {
        self.pixel_count()
    }

    fn pupil_count(&self) -> usize {
        self.region_count()
    }

    fn try_for_each_row<E>(
        &self,
        pupil: usize,
        mut row: impl FnMut(&[T]) -> Result<(), E>,
    ) -> Result<(), E> {
        let start = pupil * self.stride();
        row(&self.data()[start..start + self.pixel_count()])
    }
}

impl<T> PupilPixels<T> for ImageRegionPixels<'_, '_, T> {
    fn pixel_count(&self) -> usize {
        self.pixel_count()
    }

    fn pupil_count(&self) -> usize {
        self.region_count()
    }

    fn try_for_each_row<E>(
        &self,
        pupil: usize,
        mut row: impl FnMut(&[T]) -> Result<(), E>,
    ) -> Result<(), E> {
        for pupil_row in 0..self.region_size().0 {
            row(self.row(pupil, pupil_row))?;
        }
        Ok(())
    }
}

/// Invalid prepared pyramid-pixel configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPixelPlanError {
    /// The shared pupil mask contains no pixels.
    EmptyMask,

    /// The shared pupil mask selects no reconstruction pixels.
    NoSelectedPixels,

    /// The pyramid-pixel output extent cannot be represented by `usize`.
    OutputExtentOverflow,

    /// The full-pupil sample count cannot be represented by the scalar type.
    SampleCountNotRepresentable,
}

impl fmt::Display for PyramidPixelPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMask => formatter.write_str("pyramid pupil mask must be nonempty"),
            Self::NoSelectedPixels => {
                formatter.write_str("pyramid pupil mask must select reconstruction pixels")
            }
            Self::OutputExtentOverflow => {
                formatter.write_str("pyramid-pixel output extent overflows usize")
            }
            Self::SampleCountNotRepresentable => formatter
                .write_str("full-pupil sample count is not representable by the scalar type"),
        }
    }
}

impl core::error::Error for PyramidPixelPlanError {}

/// Invalid pyramid-pixel input, output, or numerical result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidPixelError {
    /// Each pupil image has the wrong pixel extent.
    InputPixelExtent {
        /// Required pixels per pupil image.
        expected: usize,
        /// Supplied pixels per pupil image.
        actual: usize,
    },

    /// The input does not contain exactly four pupil images.
    InputPupilExtent {
        /// Supplied pupil-image extent.
        actual: usize,
    },

    /// The reconstruction-pixel output has the wrong extent.
    OutputExtent {
        /// Required reconstruction-pixel extent.
        expected: usize,
        /// Supplied reconstruction-pixel extent.
        actual: usize,
    },

    /// A pupil image contains a non-finite pixel.
    NonFiniteInput {
        /// Zero-based pupil-image index.
        pupil: usize,
        /// Zero-based row-major pixel index within the pupil image.
        pixel: usize,
    },

    /// Full-pupil accumulation produced a non-finite mean intensity.
    NonFiniteMean,

    /// The reciprocal current-frame mean is not finite.
    NonFiniteNormalizationGain,

    /// Applying the reciprocal previous-frame mean produced a non-finite pixel.
    NonFiniteOutput {
        /// Zero-based reconstruction-pixel output index.
        index: usize,
    },
}

impl fmt::Display for PyramidPixelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputPixelExtent { expected, actual } => write!(
                formatter,
                "pyramid pupil pixel extent {actual} does not match {expected}"
            ),
            Self::InputPupilExtent { actual } => write!(
                formatter,
                "pyramid input contains {actual} pupil images, expected {PYRAMID_PUPIL_COUNT}"
            ),
            Self::OutputExtent { expected, actual } => write!(
                formatter,
                "pyramid-pixel output extent {actual} does not match {expected}"
            ),
            Self::NonFiniteInput { pupil, pixel } => write!(
                formatter,
                "pyramid pupil {pupil} pixel {pixel} is not finite"
            ),
            Self::NonFiniteMean => {
                formatter.write_str("pyramid full-pupil mean intensity is not finite")
            }
            Self::NonFiniteNormalizationGain => {
                formatter.write_str("pyramid normalization gain is not finite")
            }
            Self::NonFiniteOutput { index } => write!(
                formatter,
                "normalized pyramid pixel at output index {index} is not finite"
            ),
        }
    }
}

impl core::error::Error for PyramidPixelError {}

/// Grouped caller-owned output for one pyramid-pixel frame.
///
/// Reconstruction pixels are pupil-major. Within each pupil they retain the
/// row-major order selected by the shared mask. `mean_pupil_intensity` is the
/// current frame's mean over every pixel in all four pupil images, including
/// pixels excluded by the reconstruction mask. On the canonical AO boundary,
/// input and mean intensity are PDE and normalized reconstruction pixels are
/// dimensionless.
#[derive(Debug)]
pub struct PyramidPixelOutput<'a, T> {
    pixels: &'a mut [T],
    mean_pupil_intensity: &'a mut T,
}

impl<'a, T> PyramidPixelOutput<'a, T> {
    /// Groups caller-owned reconstruction-pixel and mean-intensity storage.
    pub fn new(pixels: &'a mut [T], mean_pupil_intensity: &'a mut T) -> Self {
        Self {
            pixels,
            mean_pupil_intensity,
        }
    }

    /// Returns normalized reconstruction pixels in pupil-major order.
    pub fn pixels(&self) -> &[T] {
        self.pixels
    }

    /// Returns the current frame's full-pupil mean intensity.
    pub fn mean_pupil_intensity(&self) -> T
    where
        T: Copy,
    {
        *self.mean_pupil_intensity
    }

    /// Returns the reconstruction-pixel extent.
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Returns whether the reconstruction-pixel output is empty.
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }
}

/// Committed normalization history for one pyramid-pixel execution owner.
///
/// New workspaces use zero gain, matching HEART's startup behavior. Therefore
/// the first successful frame measures and commits its mean intensity while
/// producing zero reconstruction pixels. The following frame uses that
/// committed reciprocal mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyramidPixelWorkspace<T> {
    reciprocal_previous_frame_mean: T,
}

impl<T> PyramidPixelWorkspace<T>
where
    T: Float,
{
    /// Creates normalization history with HEART-compatible zero startup gain.
    pub fn new() -> Self {
        Self {
            reciprocal_previous_frame_mean: T::zero(),
        }
    }

    /// Returns the normalization gain committed by the previous frame.
    pub fn reciprocal_previous_frame_mean(&self) -> T {
        self.reciprocal_previous_frame_mean
    }

    pub(crate) fn commit_frame_gain(&mut self, gain: T) {
        self.reciprocal_previous_frame_mean = gain;
    }
}

impl<T> Default for PyramidPixelWorkspace<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Prepared HEART-compatible pyramid pupil-pixel normalization.
///
/// Let `P[k][q][j]` be row-major pixel `j` in pupil image `q` for frame `k`,
/// `M[j]` the shared mask, and `g[k-1]` the committed reciprocal mean from the
/// previous completed frame. Processing emits, in pupil-major order,
/// `g[k-1] * P[k][q][j]` for every `j` selected by `M`. It also evaluates
///
/// ```text
/// mean[k] = sum(P[k][q][j], q = 0, 1, 2, 3, j = all pixels) / (4 * pixel_count)
/// g[k] = 1 / mean[k] when mean[k] > 0, otherwise 0
/// ```
///
/// The mask selects reconstruction pixels but does not affect the mean. The
/// new gain is committed only after the complete operation succeeds. This
/// plan intentionally exposes only [`AlgorithmPlan`]: no progressive pupil
/// contract has been admitted.
#[derive(Clone, Debug, PartialEq)]
pub struct PyramidPixelPlan<T> {
    mask: Arc<[bool]>,
    output_pixel_count: usize,
    mean_scale: T,
}

impl<T> PyramidPixelPlan<T>
where
    T: Float + NumCast,
{
    /// Prepares one shared row-major pupil mask.
    pub fn new(mask: impl Into<Arc<[bool]>>) -> Result<Self, PyramidPixelPlanError> {
        let mask = mask.into();
        if mask.is_empty() {
            return Err(PyramidPixelPlanError::EmptyMask);
        }
        let selected_pixel_count = mask.iter().filter(|selected| **selected).count();
        if selected_pixel_count == 0 {
            return Err(PyramidPixelPlanError::NoSelectedPixels);
        }
        let output_pixel_count = selected_pixel_count
            .checked_mul(PYRAMID_PUPIL_COUNT)
            .ok_or(PyramidPixelPlanError::OutputExtentOverflow)?;
        let sample_count = mask
            .len()
            .checked_mul(PYRAMID_PUPIL_COUNT)
            .ok_or(PyramidPixelPlanError::OutputExtentOverflow)?;
        let sample_count = <T as NumCast>::from(sample_count)
            .ok_or(PyramidPixelPlanError::SampleCountNotRepresentable)?;

        Ok(Self {
            mask,
            output_pixel_count,
            mean_scale: T::one() / sample_count,
        })
    }

    /// Returns the shared row-major pupil mask.
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// Returns shared ownership of the row-major pupil mask.
    pub fn shared_mask(&self) -> Arc<[bool]> {
        Arc::clone(&self.mask)
    }

    /// Returns the full pixel extent of one pupil image.
    pub fn pupil_pixel_count(&self) -> usize {
        self.mask.len()
    }

    /// Returns the selected reconstruction-pixel extent in one pupil image.
    pub fn selected_pixel_count(&self) -> usize {
        self.output_pixel_count / PYRAMID_PUPIL_COUNT
    }

    /// Returns the selected reconstruction-pixel extent across four pupils.
    pub fn output_pixel_count(&self) -> usize {
        self.output_pixel_count
    }

    /// Returns the prepared full-pupil mean scale.
    pub const fn mean_pupil_intensity_scale(&self) -> T {
        self.mean_scale
    }

    /// Creates HEART-compatible committed normalization history.
    pub fn workspace(&self) -> PyramidPixelWorkspace<T> {
        PyramidPixelWorkspace::new()
    }

    pub(crate) fn frame_mean_and_gain(&self, sum: T) -> Result<(T, T), PyramidPixelError> {
        let mean_pupil_intensity = sum * self.mean_scale;
        if !mean_pupil_intensity.is_finite() {
            return Err(PyramidPixelError::NonFiniteMean);
        }
        let reciprocal_current_frame_mean = if mean_pupil_intensity > T::zero() {
            T::one() / mean_pupil_intensity
        } else {
            T::zero()
        };
        if !reciprocal_current_frame_mean.is_finite() {
            return Err(PyramidPixelError::NonFiniteNormalizationGain);
        }
        Ok((mean_pupil_intensity, reciprocal_current_frame_mean))
    }

    fn validate<I>(
        &self,
        output: &PyramidPixelOutput<'_, T>,
        input: &I,
    ) -> Result<(), PyramidPixelError>
    where
        I: PupilPixels<T>,
    {
        if input.pixel_count() != self.pupil_pixel_count() {
            return Err(PyramidPixelError::InputPixelExtent {
                expected: self.pupil_pixel_count(),
                actual: input.pixel_count(),
            });
        }
        if input.pupil_count() != PYRAMID_PUPIL_COUNT {
            return Err(PyramidPixelError::InputPupilExtent {
                actual: input.pupil_count(),
            });
        }
        if output.len() != self.output_pixel_count {
            return Err(PyramidPixelError::OutputExtent {
                expected: self.output_pixel_count,
                actual: output.len(),
            });
        }
        Ok(())
    }

    fn process_pupils<I>(
        &self,
        output: PyramidPixelOutput<'_, T>,
        workspace: &mut PyramidPixelWorkspace<T>,
        input: &I,
    ) -> Result<(), PyramidPixelError>
    where
        I: PupilPixels<T>,
    {
        self.process_pupils_with(output, workspace, input, |output, gain| {
            self.write_selected(output, gain, input)
        })
    }

    pub(crate) fn process_pupils_with<I: PupilPixels<T>>(
        &self,
        output: PyramidPixelOutput<'_, T>,
        workspace: &mut PyramidPixelWorkspace<T>,
        input: &I,
        write_selected: impl FnOnce(&mut [T], T) -> Result<(), PyramidPixelError>,
    ) -> Result<(), PyramidPixelError> {
        self.validate(&output, input)?;
        let sum = self.ordered_sum(input)?;
        self.publish_pupils(output, workspace, sum, write_selected)
    }

    fn ordered_sum<I: PupilPixels<T>>(&self, input: &I) -> Result<T, PyramidPixelError> {
        let mut sum = T::zero();
        for pupil in 0..PYRAMID_PUPIL_COUNT {
            let mut pixel = 0;
            input.try_for_each_row(pupil, |row| {
                for &value in row {
                    if !value.is_finite() {
                        return Err(PyramidPixelError::NonFiniteInput { pupil, pixel });
                    }
                    sum = sum + value;
                    if !sum.is_finite() {
                        return Err(PyramidPixelError::NonFiniteMean);
                    }
                    pixel += 1;
                }
                Ok(())
            })?;
            debug_assert_eq!(pixel, self.pupil_pixel_count());
        }

        Ok(sum)
    }

    fn publish_pupils(
        &self,
        output: PyramidPixelOutput<'_, T>,
        workspace: &mut PyramidPixelWorkspace<T>,
        sum: T,
        write_selected: impl FnOnce(&mut [T], T) -> Result<(), PyramidPixelError>,
    ) -> Result<(), PyramidPixelError> {
        let (mean_pupil_intensity, reciprocal_current_frame_mean) =
            self.frame_mean_and_gain(sum)?;
        write_selected(output.pixels, workspace.reciprocal_previous_frame_mean)?;
        *output.mean_pupil_intensity = mean_pupil_intensity;
        workspace.commit_frame_gain(reciprocal_current_frame_mean);
        Ok(())
    }

    fn write_selected<I: PupilPixels<T>>(
        &self,
        output: &mut [T],
        gain: T,
        input: &I,
    ) -> Result<(), PyramidPixelError> {
        let mut output_index = 0;
        for pupil in 0..PYRAMID_PUPIL_COUNT {
            let mut pixel = 0;
            input.try_for_each_row(pupil, |row| {
                for &input in row {
                    if self.mask[pixel] {
                        let value = gain * input;
                        if !value.is_finite() {
                            return Err(PyramidPixelError::NonFiniteOutput {
                                index: output_index,
                            });
                        }
                        output[output_index] = value;
                        output_index += 1;
                    }
                    pixel += 1;
                }
                Ok(())
            })?;
            debug_assert_eq!(pixel, self.pupil_pixel_count());
        }

        Ok(())
    }

    /// Normalizes four pupil regions directly from borrowed detector-image rows.
    ///
    /// The image-backed regions must have the same pupil-pixel extent as the
    /// shared mask and contain exactly four pupils. Processing performs no
    /// allocation and does not materialize a region-major pupil-pixel slab.
    pub fn process_image_regions(
        &self,
        output: PyramidPixelOutput<'_, T>,
        workspace: &mut PyramidPixelWorkspace<T>,
        input: ImageRegionPixels<'_, '_, T>,
    ) -> Result<(), PyramidPixelError> {
        self.process_pupils(output, workspace, &input)
    }
}

#[cfg(feature = "blas")]
impl PyramidPixelPlan<f32> {
    /// Use a widened, lane-wise sum for small borrowed Float32 image regions.
    pub(crate) fn process_small_image(
        &self,
        output: PyramidPixelOutput<'_, f32>,
        workspace: &mut PyramidPixelWorkspace<f32>,
        input: ImageRegionPixels<'_, '_, f32>,
        write_selected: impl FnOnce(&mut [f32], f32) -> Result<(), PyramidPixelError>,
    ) -> Result<(), PyramidPixelError> {
        self.validate(&output, &input)?;
        let (rows, columns) = input.region_size();
        let sum = if rows > 1
            && columns >= 4
            && self.pupil_pixel_count() <= 4096 / PYRAMID_PUPIL_COUNT
            && filtergraph_blas::sgemv_default_float_mode()
        {
            bounded_image_sum(&input, self.mean_scale)
        } else {
            None
        };
        let sum = match sum {
            Some(value) => value,
            None => self.ordered_sum(&input)?,
        };
        self.publish_pupils(output, workspace, sum, write_selected)
    }
}

#[cfg(feature = "blas")]
#[inline(never)]
fn bounded_image_sum(input: &ImageRegionPixels<'_, '_, f32>, mean_scale: f32) -> Option<f32> {
    let limit = f32::MAX * mean_scale * 0.5;
    let mut absolute_lanes = [0.0_f64; 4];
    let mut lanes = [0.0_f64; 4];
    let mut valid = [true; 4];
    let mut tail = 0.0_f64;
    let mut absolute_tail = 0.0_f64;
    let mut tail_valid = true;
    for pupil in 0..PYRAMID_PUPIL_COUNT {
        for row_index in 0..input.region_size().0 {
            let (blocks, remainder) = input.row(pupil, row_index).as_chunks::<4>();
            for block in blocks {
                for lane in 0..4 {
                    let value = block[lane];
                    lanes[lane] += value as f64;
                    let magnitude = value.abs();
                    absolute_lanes[lane] += magnitude as f64;
                    valid[lane] &= magnitude < limit;
                }
            }
            for &value in remainder {
                tail += value as f64;
                let magnitude = value.abs();
                absolute_tail += magnitude as f64;
                tail_valid &= magnitude < limit;
            }
        }
    }
    let total = lanes.into_iter().sum::<f64>() + tail;
    let magnitude = absolute_lanes.into_iter().sum::<f64>() + absolute_tail;
    if !tail_valid
        || !valid.into_iter().all(|value| value)
        || total.abs() <= 0.01 * magnitude
        || total.abs() * (mean_scale as f64) < f32::MIN_POSITIVE as f64
    {
        // Keep ordered rejection and gain decisions for sensitive frames.
        None
    } else {
        Some(total as f32)
    }
}

impl<T> AlgorithmPlan for PyramidPixelPlan<T>
where
    T: Float + NumCast + 'static,
{
    type Input<'a> = RegionPixels<'a, T>;
    type Output<'a> = PyramidPixelOutput<'a, T>;
    type Workspace = PyramidPixelWorkspace<T>;
    type Error = PyramidPixelError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        self.process_pupils(output, workspace, &input)
    }
}

#[cfg(all(test, feature = "blas"))]
mod small_image_tests {
    use super::*;
    use crate::RegionExtractionPlan;
    use alloc::vec;

    fn process(
        plan: &PyramidPixelPlan<f32>,
        output: PyramidPixelOutput<'_, f32>,
        workspace: &mut PyramidPixelWorkspace<f32>,
        input: ImageRegionPixels<'_, '_, f32>,
    ) -> Result<(), PyramidPixelError> {
        plan.process_small_image(output, workspace, input, |pixels, gain| {
            plan.write_selected(pixels, gain, &input)
        })
    }

    #[test]
    fn widened_sum_falls_back_in_order_and_preserves_previous_gain() {
        let geometry =
            RegionExtractionPlan::<f32>::new((8, 16), &[(0, 0), (0, 8), (4, 0), (4, 8)], (4, 8))
                .unwrap();
        let plan = PyramidPixelPlan::new(vec![true; 32]).unwrap();
        let mut workspace = plan.workspace();
        let mut pixels = [0.0; 128];
        let mut mean = 0.0;
        let mut image = [2.0; 128];

        process(
            &plan,
            PyramidPixelOutput::new(&mut pixels, &mut mean),
            &mut workspace,
            geometry.image_regions(&image).unwrap(),
        )
        .unwrap();
        assert_eq!(mean, 2.0);

        image[..3].copy_from_slice(&[f32::MAX, f32::MAX, f32::NAN]);
        assert_eq!(
            process(
                &plan,
                PyramidPixelOutput::new(&mut pixels, &mut mean),
                &mut workspace,
                geometry.image_regions(&image).unwrap(),
            ),
            Err(PyramidPixelError::NonFiniteMean),
        );
        assert_eq!(mean, 2.0);

        image.fill(4.0);
        process(
            &plan,
            PyramidPixelOutput::new(&mut pixels, &mut mean),
            &mut workspace,
            geometry.image_regions(&image).unwrap(),
        )
        .unwrap();
        assert_eq!(mean, 4.0);
        assert_eq!(pixels, [2.0; 128]);
    }

    #[test]
    fn signed_cancellation_retains_ordered_next_frame_gain() {
        let geometry =
            RegionExtractionPlan::<f32>::new((8, 16), &[(0, 0), (0, 8), (4, 0), (4, 8)], (4, 8))
                .unwrap();
        let plan = PyramidPixelPlan::new(vec![true; 32]).unwrap();
        let large = (1_u32 << 25) as f32;
        for (signed, expected) in [
            ([large, 1.0, -large, 0.0], 0.0),
            ([large, 2.0, -large, -1.0], -1.0),
            ([-large, -2.0, large, 1.0], 1.0),
        ] {
            let mut workspace = plan.workspace();
            let mut pixels = [0.0; 128];
            let mut mean = 0.0;
            let mut image = [2.0; 128];
            let mut process = |image: &[f32]| {
                process(
                    &plan,
                    PyramidPixelOutput::new(&mut pixels, &mut mean),
                    &mut workspace,
                    geometry.image_regions(image).unwrap(),
                )
                .unwrap();
                (mean, pixels[0])
            };
            assert_eq!(process(&image).0, 2.0);
            image.fill(0.0);
            image[..4].copy_from_slice(&signed);
            assert_eq!(process(&image), (expected / 128.0, signed[0] * 0.5));
            image.fill(4.0);
            assert_eq!(
                process(&image),
                (4.0, if expected > 0.0 { 512.0 } else { 0.0 })
            );
        }
    }
}
