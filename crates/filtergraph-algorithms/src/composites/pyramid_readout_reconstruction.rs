//! Progressive pyramid reconstruction over detector readout regions.

use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
use core::fmt;

use num_traits::NumCast;

use crate::{
    DetectorReadoutBlockInfo, DetectorReadoutPlan, DetectorReadoutProfileError,
    DetectorReadoutResult, MatrixElement, PYRAMID_PUPIL_COUNT, PreparedGemv, PreparedGemvError,
    PreparedProgressiveGemv, PreparedProgressiveGemvError, ProgressiveGemvError, PyramidPixelError,
    PyramidPixelImagePlan, PyramidPixelImagePreparationError, PyramidPixelWorkspace,
};

use super::detector_readout::DetectorReadoutFrame;

/// Invalid portable progressive pyramid preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidReadoutReconstructionPreparationError {
    /// The detector readout profile is invalid.
    Readout(DetectorReadoutProfileError),
    /// Pyramid pupil geometry or normalization is invalid.
    Pixels(PyramidPixelImagePreparationError),
    /// Dense reconstructor preparation failed.
    Reconstructor(PreparedGemvError),
    /// The prepared operations are incompatible.
    Composition(PyramidReadoutReconstructionPlanError),
}

impl fmt::Display for PyramidReadoutReconstructionPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Readout(error) => error.fmt(formatter),
            Self::Pixels(error) => error.fmt(formatter),
            Self::Reconstructor(error) => error.fmt(formatter),
            Self::Composition(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PyramidReadoutReconstructionPreparationError {}

impl From<DetectorReadoutProfileError> for PyramidReadoutReconstructionPreparationError {
    fn from(error: DetectorReadoutProfileError) -> Self {
        Self::Readout(error)
    }
}

impl From<PyramidPixelImagePreparationError> for PyramidReadoutReconstructionPreparationError {
    fn from(error: PyramidPixelImagePreparationError) -> Self {
        Self::Pixels(error)
    }
}

impl From<PreparedGemvError> for PyramidReadoutReconstructionPreparationError {
    fn from(error: PreparedGemvError) -> Self {
        Self::Reconstructor(error)
    }
}

impl From<PyramidReadoutReconstructionPlanError> for PyramidReadoutReconstructionPreparationError {
    fn from(error: PyramidReadoutReconstructionPlanError) -> Self {
        Self::Composition(error)
    }
}

/// Invalid progressive detector-readout pyramid preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidReadoutReconstructionPlanError {
    /// Detector readout and pyramid image extents differ.
    ImageExtent {
        /// Prepared detector readout image size.
        readout: (usize, usize),
        /// Prepared pyramid image size.
        pupils: (usize, usize),
    },
    /// The pyramid image view is not a complete row-major detector image.
    ImageLayout,
    /// Reconstructor storage extent arithmetic overflowed.
    ReconstructorExtentOverflow,
    /// Reconstructor columns do not equal the canonical selected-pixel extent.
    ReconstructorColumns {
        /// Required pupil-major selected-pixel extent.
        expected: usize,
        /// Supplied reconstructor column count.
        actual: usize,
    },
    /// Prepared scientific readiness matrix packing is invalid.
    Reconstructor(PreparedProgressiveGemvError),
}

impl fmt::Display for PyramidReadoutReconstructionPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageExtent { readout, pupils } => write!(
                formatter,
                "detector readout image {readout:?} does not match pyramid image {pupils:?}"
            ),
            Self::ImageLayout => {
                formatter.write_str("pyramid detector image must use complete row-major layout")
            }
            Self::ReconstructorExtentOverflow => {
                formatter.write_str("pyramid reconstructor extent overflow")
            }
            Self::ReconstructorColumns { expected, actual } => write!(
                formatter,
                "reconstructor column count {actual} does not match reconstruction-pixel extent {expected}"
            ),
            Self::Reconstructor(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PyramidReadoutReconstructionPlanError {}

impl From<PreparedProgressiveGemvError> for PyramidReadoutReconstructionPlanError {
    fn from(error: PreparedProgressiveGemvError) -> Self {
        Self::Reconstructor(error)
    }
}

/// Invalid progressive detector-readout pyramid execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidReadoutReconstructionError {
    /// The input block did not carry transport-neutral sample identity.
    MissingSampleMetadata,
    /// The portable acquisition ordinal cannot be represented on this target.
    OrdinalOverflow,
    /// Mean-pupil-intensity output is not one scalar.
    MeanOutputExtent,
    /// Caller-owned normalized reconstruction pixels have the wrong extent.
    PixelOutputExtent {
        /// Required pupil-major selected-pixel extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// Caller-owned reconstructed output has the wrong extent.
    ReconstructedOutputExtent {
        /// Required reconstructed-vector extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// One detector readout block has the wrong extent.
    InputExtent {
        /// Required block extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// Pyramid normalization rejected a detector or derived value.
    Pixels(PyramidPixelError),
    /// Progressive reconstruction accumulation failed.
    Reconstruction(ProgressiveGemvError),
}

impl fmt::Display for PyramidReadoutReconstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSampleMetadata => {
                formatter.write_str("detector readout block sample metadata is required")
            }
            Self::OrdinalOverflow => {
                formatter.write_str("detector readout acquisition ordinal does not fit usize")
            }
            Self::MeanOutputExtent => {
                formatter.write_str("mean-pupil-intensity output must contain one value")
            }
            Self::PixelOutputExtent { expected, actual } => write!(
                formatter,
                "reconstruction-pixel output extent {actual} does not match {expected}"
            ),
            Self::ReconstructedOutputExtent { expected, actual } => write!(
                formatter,
                "reconstructed output extent {actual} does not match {expected}"
            ),
            Self::InputExtent { expected, actual } => write!(
                formatter,
                "detector readout block extent {actual} does not match {expected}"
            ),
            Self::Pixels(error) => error.fmt(formatter),
            Self::Reconstruction(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PyramidReadoutReconstructionError {}

/// Caller-owned terminal outputs of progressive pyramid reconstruction.
#[derive(Debug)]
pub struct PyramidReadoutReconstructionOutput<'a, T> {
    reconstruction_pixels: &'a mut [T],
    mean_pupil_intensity: &'a mut T,
    reconstructed: &'a mut [T],
}

impl<'a, T> PyramidReadoutReconstructionOutput<'a, T> {
    /// Groups normalized pixels, mean pupil intensity, and reconstructed output.
    pub fn new(
        reconstruction_pixels: &'a mut [T],
        mean_pupil_intensity: &'a mut T,
        reconstructed: &'a mut [T],
    ) -> Self {
        Self {
            reconstruction_pixels,
            mean_pupil_intensity,
            reconstructed,
        }
    }
}

/// Prepared multi-region pyramid normalization and reconstruction.
///
/// Selected pixels retain pupil-major scientific identity. Detector acquisition
/// order controls only when each selected pixel is normalized and when its
/// packed canonical reconstructor column is accumulated.
#[derive(Clone, Debug)]
pub struct PyramidReadoutReconstructionPlan<T> {
    image: PyramidPixelImagePlan<T>,
    readout: DetectorReadoutPlan,
    reconstructor: PreparedProgressiveGemv<T>,
    selected_detector_indices: Box<[usize]>,
    selected_pupil_pixels: Box<[(usize, usize)]>,
    pixel_completion_ordinals: Box<[usize]>,
}

impl<T> PyramidReadoutReconstructionPlan<T>
where
    T: MatrixElement + NumCast,
{
    /// Prepares selected-pixel readiness and reconstructor panels.
    pub fn new(
        image: PyramidPixelImagePlan<T>,
        canonical_reconstructor: PreparedGemv<T>,
        readout: DetectorReadoutPlan,
    ) -> Result<Self, PyramidReadoutReconstructionPlanError> {
        let regions = image.pupil_regions();
        if regions.image_size() != readout.detector_size() {
            return Err(PyramidReadoutReconstructionPlanError::ImageExtent {
                readout: readout.detector_size(),
                pupils: regions.image_size(),
            });
        }
        if regions.row_stride() != regions.image_size().1 || regions.first_pixel() != 0 {
            return Err(PyramidReadoutReconstructionPlanError::ImageLayout);
        }
        let pixel_count = image.normalization().output_pixel_count();
        if canonical_reconstructor.columns() != pixel_count {
            return Err(
                PyramidReadoutReconstructionPlanError::ReconstructorColumns {
                    expected: pixel_count,
                    actual: canonical_reconstructor.columns(),
                },
            );
        }

        let (pupil_rows, pupil_columns) = regions.region_size();
        let detector_columns = regions.image_size().1;
        let mask = image.normalization().mask();
        let mut selected_detector_indices = Vec::with_capacity(pixel_count);
        let mut selected_pupil_pixels = Vec::with_capacity(pixel_count);
        let mut completion_ordinals = Vec::with_capacity(pixel_count);
        for (pupil, &(first_row, first_column)) in regions.origins().iter().enumerate() {
            for pupil_row in 0..pupil_rows {
                for pupil_column in 0..pupil_columns {
                    let pixel = pupil_row * pupil_columns + pupil_column;
                    if !mask[pixel] {
                        continue;
                    }
                    let row = first_row + pupil_row;
                    let column = first_column + pupil_column;
                    selected_detector_indices.push(row * detector_columns + column);
                    selected_pupil_pixels.push((pupil, pixel));
                    completion_ordinals.push(
                        readout
                            .arrival_ordinal(row, column)
                            .expect("matching complete detector mapping"),
                    );
                }
            }
        }
        debug_assert_eq!(selected_detector_indices.len(), pixel_count);
        let reconstructor = PreparedProgressiveGemv::new(
            canonical_reconstructor,
            &completion_ordinals,
            readout.blocks(),
        )?;
        Ok(Self {
            image,
            readout,
            reconstructor,
            selected_detector_indices: selected_detector_indices.into_boxed_slice(),
            selected_pupil_pixels: selected_pupil_pixels.into_boxed_slice(),
            pixel_completion_ordinals: completion_ordinals.into_boxed_slice(),
        })
    }

    /// Returns the canonical pyramid pupil geometry and normalization.
    pub const fn image(&self) -> &PyramidPixelImagePlan<T> {
        &self.image
    }

    /// Returns the prepared detector readout mapping.
    pub const fn readout(&self) -> &DetectorReadoutPlan {
        &self.readout
    }

    /// Returns the canonical matrix and its prepared acquisition groups.
    pub const fn reconstructor(&self) -> &PreparedProgressiveGemv<T> {
        &self.reconstructor
    }

    /// Returns each canonical selected pixel's acquisition ordinal.
    pub fn pixel_completion_ordinals(&self) -> &[usize] {
        &self.pixel_completion_ordinals
    }

    /// Allocates bounded mutable storage for one execution owner.
    pub fn workspace(&self) -> PyramidReadoutReconstructionWorkspace<T> {
        PyramidReadoutReconstructionWorkspace {
            detector: vec![T::zero(); self.readout.detector_elements()].into_boxed_slice(),
            reconstruction_pixels: vec![T::zero(); self.image.normalization().output_pixel_count()]
                .into_boxed_slice(),
            normalization: self.image.normalization().workspace(),
            reconstructed: vec![T::zero(); self.reconstructor.canonical().rows()]
                .into_boxed_slice(),
            frame: DetectorReadoutFrame::new(),
        }
    }

    /// Consumes one immutable detector readout block and conditionally publishes.
    pub fn process_block(
        &self,
        output: PyramidReadoutReconstructionOutput<'_, T>,
        workspace: &mut PyramidReadoutReconstructionWorkspace<T>,
        block: &[T],
        info: DetectorReadoutBlockInfo,
    ) -> Result<DetectorReadoutResult, PyramidReadoutReconstructionError> {
        let pixel_count = self.image.normalization().output_pixel_count();
        if output.reconstruction_pixels.len() != pixel_count {
            return Err(PyramidReadoutReconstructionError::PixelOutputExtent {
                expected: pixel_count,
                actual: output.reconstruction_pixels.len(),
            });
        }
        if output.reconstructed.len() != self.reconstructor.canonical().rows() {
            return Err(
                PyramidReadoutReconstructionError::ReconstructedOutputExtent {
                    expected: self.reconstructor.canonical().rows(),
                    actual: output.reconstructed.len(),
                },
            );
        }
        if block.len() != self.readout.block_elements() {
            return Err(PyramidReadoutReconstructionError::InputExtent {
                expected: self.readout.block_elements(),
                actual: block.len(),
            });
        }
        if let Err(reason) = workspace.frame.accept(info, self.readout.blocks()) {
            workspace.clear_partial();
            return Ok(DetectorReadoutResult::Dropped(reason));
        }
        if info.ordinal == 0 {
            workspace.clear_partial();
        }
        self.readout
            .scatter_block(&mut workspace.detector, block, info.ordinal)
            .expect("validated extents and admitted ordinal");

        let gain = workspace.normalization.reciprocal_previous_frame_mean();
        for &pixel in self
            .reconstructor
            .input_indices(info.ordinal)
            .expect("admitted ordinal")
        {
            let input = workspace.detector[self.selected_detector_indices[pixel]];
            if !input.is_finite() {
                workspace.abandon();
                let (pupil, pixel) = self.selected_pupil_pixels[pixel];
                return Err(PyramidReadoutReconstructionError::Pixels(
                    PyramidPixelError::NonFiniteInput { pupil, pixel },
                ));
            }
            let normalized = gain * input;
            if !normalized.is_finite() {
                workspace.abandon();
                return Err(PyramidReadoutReconstructionError::Pixels(
                    PyramidPixelError::NonFiniteOutput { index: pixel },
                ));
            }
            workspace.reconstruction_pixels[pixel] = normalized;
        }
        if let Err(error) = self.reconstructor.accumulate_group(
            &mut workspace.reconstructed,
            &workspace.reconstruction_pixels,
            info.ordinal,
        ) {
            workspace.abandon();
            return Err(PyramidReadoutReconstructionError::Reconstruction(error));
        }

        if !info.terminal {
            return Ok(DetectorReadoutResult::Pending);
        }
        let regions = self.image.pupil_regions();
        let (pupil_rows, pupil_columns) = regions.region_size();
        let detector_columns = regions.image_size().1;
        let mut sum = T::zero();
        for pupil in 0..PYRAMID_PUPIL_COUNT {
            let (first_row, first_column) = regions.origins()[pupil];
            let mut pupil_pixel = 0;
            for pupil_row in 0..pupil_rows {
                let start = (first_row + pupil_row) * detector_columns + first_column;
                for &input in &workspace.detector[start..start + pupil_columns] {
                    if !input.is_finite() {
                        workspace.abandon();
                        return Err(PyramidReadoutReconstructionError::Pixels(
                            PyramidPixelError::NonFiniteInput {
                                pupil,
                                pixel: pupil_pixel,
                            },
                        ));
                    }
                    sum = sum + input;
                    if !sum.is_finite() {
                        workspace.abandon();
                        return Err(PyramidReadoutReconstructionError::Pixels(
                            PyramidPixelError::NonFiniteMean,
                        ));
                    }
                    pupil_pixel += 1;
                }
            }
        }
        let (mean, next_gain) = match self.image.normalization().frame_mean_and_gain(sum) {
            Ok(values) => values,
            Err(error) => {
                workspace.abandon();
                return Err(PyramidReadoutReconstructionError::Pixels(error));
            }
        };
        output
            .reconstruction_pixels
            .copy_from_slice(&workspace.reconstruction_pixels);
        *output.mean_pupil_intensity = mean;
        output
            .reconstructed
            .copy_from_slice(&workspace.reconstructed);
        workspace.normalization.commit_frame_gain(next_gain);
        let result = DetectorReadoutResult::Complete {
            sequence: info.sequence,
            discontinuity: workspace.frame.discontinuity(),
        };
        workspace.complete();
        Ok(result)
    }
}

/// Bounded mutable state for one progressive pyramid execution owner.
#[derive(Debug)]
pub struct PyramidReadoutReconstructionWorkspace<T> {
    detector: Box<[T]>,
    reconstruction_pixels: Box<[T]>,
    normalization: PyramidPixelWorkspace<T>,
    reconstructed: Box<[T]>,
    frame: DetectorReadoutFrame,
}

impl<T> PyramidReadoutReconstructionWorkspace<T>
where
    T: MatrixElement,
{
    pub(crate) fn frame_discontinuity(&self) -> bool {
        self.frame.discontinuity()
    }

    pub(crate) fn reconstruction_pixels(&self) -> &[T] {
        &self.reconstruction_pixels
    }

    /// Returns the active detector-frame sequence, if any.
    pub const fn sequence(&self) -> Option<u64> {
        self.frame.sequence()
    }

    /// Returns the reciprocal mean committed by the previous complete frame.
    pub fn reciprocal_previous_frame_mean(&self) -> T {
        self.normalization.reciprocal_previous_frame_mean()
    }

    /// Returns the private progressive reconstructed-output accumulator.
    pub fn accumulated(&self) -> &[T] {
        &self.reconstructed
    }

    /// Resets frame assembly and committed normalization history.
    pub fn reset(&mut self) {
        self.frame = DetectorReadoutFrame::new();
        self.normalization = PyramidPixelWorkspace::new();
        self.clear_partial();
    }

    /// Abandons a partial frame while preserving committed normalization.
    pub fn plan_adopted(&mut self) {
        self.abandon();
    }

    fn abandon(&mut self) {
        self.frame.abandon();
        self.clear_partial();
    }

    fn clear_partial(&mut self) {
        self.detector.fill(T::zero());
        self.reconstruction_pixels.fill(T::zero());
        self.reconstructed.fill(T::zero());
    }

    fn complete(&mut self) {
        self.frame.complete();
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 multi-region pyramid reconstruction declaration.
    pub PyramidReadoutReconstructorF32 {
        label = "pwfs-readout-reconstructor-f32";
        configuration PyramidReadoutReconstructorF32Configuration {
            detector_size: (usize, usize) = GraphRebuild;
            region_lines: usize = GraphRebuild;
            block_lines: usize = GraphRebuild;
            region_samples: usize = GraphRebuild;
            detector_origins: Vec<[usize; 2]> = GraphRebuild;
            detector_line_steps: Vec<[i32; 2]> = GraphRebuild;
            detector_sample_steps: Vec<[i32; 2]> = GraphRebuild;
            pupil_size: (usize, usize) = GraphRebuild;
            pupil_origins: Vec<[usize; 2]> = GraphRebuild;
            pupil_mask: Option<Vec<bool>> = Construction;
            reconstructed_count: usize = GraphRebuild;
            initial_reconstructor: Option<Vec<f32>> = Construction;
            reconstructed_schema: alloc::string::String = GraphRebuild;
        }
        plan = PyramidReadoutReconstructionPlan<f32>;
        workspace = PyramidReadoutReconstructionWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            readout_region_count = plan.readout().regions().len();
            block_lines = plan.readout().block_lines();
            region_samples = plan.readout().region_samples();
            selected_pixel_count = plan.image().normalization().selected_pixel_count();
            reconstructed_count = plan.reconstructor().canonical().rows();
            block_rate_multiplier = u32::try_from(plan.readout().blocks())
                .map_err(|_| crate::DeclarationPreparationError::Format(crate::FormatError::Rate))?;
        }
        inputs {
            readout_block: f32 [readout_region_count, block_lines, region_samples]
                = "readout-block",
                schema = crate::schemas::CALIBRATED_PIXEL_READOUT_BLOCK_V1,
                sample = readout_sample,
                rate = [block_rate_multiplier, 1];
        }
        outputs {
            reconstruction_pixels: f32 [PYRAMID_PUPIL_COUNT, selected_pixel_count]
                = "reconstruction-pixels",
                schema = crate::schemas::PYRAMID_RECONSTRUCTION_PIXELS_V1,
                metadata_from = "readout-block",
                publication = conditional;
            mean_pupil_intensity: f32 [1] = "mean-pupil-intensity",
                schema = crate::schemas::PYRAMID_MEAN_PUPIL_INTENSITY_V1,
                metadata_from = "readout-block",
                publication = conditional;
            reconstructed: f32 [reconstructed_count] = "reconstructed",
                schema = configuration.reconstructed_schema.as_str(),
                metadata_from = "readout-block",
                publication = conditional;
        }
        prepare |configuration| {
            let readout = DetectorReadoutPlan::from_i32_coordinate_steps(
                configuration.detector_size,
                configuration.region_lines,
                configuration.block_lines,
                configuration.region_samples,
                &configuration.detector_origins,
                &configuration.detector_line_steps,
                &configuration.detector_sample_steps,
            )?;
            let pupil_origins = configuration.pupil_origins.iter()
                .map(|origin| (origin[0], origin[1]))
                .collect::<Vec<_>>();
            let image = PyramidPixelImagePlan::from_detector_pupils(
                configuration.detector_size,
                configuration.pupil_size,
                &pupil_origins,
                super::pyramid_measurement_block::configured_pyramid_mask(configuration.pupil_mask, configuration.pupil_size),
            )?;
            let reconstruction_pixel_count = image.normalization().output_pixel_count();
            let elements = configuration.reconstructed_count
                .checked_mul(reconstruction_pixel_count)
                .ok_or(PyramidReadoutReconstructionPlanError::ReconstructorExtentOverflow)?;
            let values = configuration.initial_reconstructor
                .unwrap_or_else(|| vec![0.0; elements]);
            let plan = PyramidReadoutReconstructionPlan::new(
                image,
                PreparedGemv::new(
                    configuration.reconstructed_count,
                    reconstruction_pixel_count,
                    Arc::from(values),
                )?,
                readout,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = PyramidReadoutReconstructionPreparationError;
        process |plan, workspace| {
            let Some(sample) = readout_sample else {
                workspace.plan_adopted();
                return Err(PyramidReadoutReconstructionError::MissingSampleMetadata);
            };
            let ordinal = usize::try_from(sample.offset())
                .map_err(|_| PyramidReadoutReconstructionError::OrdinalOverflow)?;
            let [mean] = mean_pupil_intensity else {
                return Err(PyramidReadoutReconstructionError::MeanOutputExtent);
            };
            let result = plan.process_block(
                PyramidReadoutReconstructionOutput::new(
                    reconstruction_pixels,
                    mean,
                    reconstructed,
                ),
                workspace,
                readout_block,
                DetectorReadoutBlockInfo {
                    sequence: sample.sequence(),
                    ordinal,
                    terminal: sample.is_terminal(),
                    discontinuity: sample.is_discontinuous(),
                    corrupted: sample.is_corrupted(),
                },
            )?;
            Ok(match result {
                DetectorReadoutResult::Pending | DetectorReadoutResult::Dropped(_) => {
                    crate::AlgorithmProcessResult::deferred()
                }
                DetectorReadoutResult::Complete { sequence: _, discontinuity } => {
                    crate::AlgorithmProcessResult::complete_with_metadata(
                        sample.with_offset(0)
                        .with_terminal(true)
                        .with_discontinuity(discontinuity)
                        .with_corrupted(false),
                    )
                }
            })
        }
        process_error = PyramidReadoutReconstructionError;
        reset |_plan, workspace| { workspace.reset(); }
        plan_adopted |_plan, workspace| { workspace.plan_adopted(); }
    }
}
