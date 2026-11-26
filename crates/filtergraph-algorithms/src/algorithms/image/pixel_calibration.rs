//! Prepared detector pixel calibration.

use alloc::{sync::Arc, vec, vec::Vec};
use core::fmt;
use core::marker::PhantomData;
use core::ops::Range;

use num_traits::{AsPrimitive, Float};

use crate::{
    AlgorithmPlan, InputProgress, OutputProgress, ProcessError, ProgressResult, ProgressWorkspace,
};

/// Invalid prepared pixel-calibration configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelCalibrationError {
    /// Calibration arrays contain no detector pixels.
    Empty,

    /// Flat and background arrays have different extents.
    ExtentMismatch {
        /// Flat-array extent.
        flat: usize,
        /// Background-array extent.
        background: usize,
    },

    /// A flat value is not finite.
    NonFiniteFlat {
        /// Zero-based detector-pixel index.
        index: usize,
    },

    /// A background value is not finite.
    NonFiniteBackground {
        /// Zero-based detector-pixel index.
        index: usize,
    },
}

impl fmt::Display for PixelCalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("calibration arrays must be nonempty"),
            Self::ExtentMismatch { flat, background } => write!(
                formatter,
                "flat extent {flat} does not match background extent {background}"
            ),
            Self::NonFiniteFlat { index } => {
                write!(formatter, "flat value at index {index} is not finite")
            }
            Self::NonFiniteBackground { index } => {
                write!(formatter, "background value at index {index} is not finite")
            }
        }
    }
}

impl core::error::Error for PixelCalibrationError {}

/// Invalid portable U16-to-F32 pixel-calibration construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelCalibrationU16F32PreparationError {
    /// Detector rows and columns overflow the addressable pixel extent.
    ImageExtentOverflow,
    /// Initial flat/background calibration is invalid.
    Calibration(PixelCalibrationError),
}

/// Invalid portable row-block U16-to-F32 pixel-calibration construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelCalibrationRowU16F32PreparationError {
    /// Detector rows and columns overflow the addressable pixel extent.
    ImageExtentOverflow,
    /// The row block must be smaller than the detector and divide its height.
    RowBlockRows,
    /// Initial flat/background calibration is invalid.
    Calibration(PixelCalibrationError),
}

impl fmt::Display for PixelCalibrationRowU16F32PreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageExtentOverflow => formatter.write_str("detector image extent overflow"),
            Self::RowBlockRows => formatter.write_str(
                "row-block height must be positive, smaller than the detector, and divide it",
            ),
            Self::Calibration(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PixelCalibrationRowU16F32PreparationError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::ImageExtentOverflow | Self::RowBlockRows => None,
            Self::Calibration(error) => Some(error),
        }
    }
}

impl From<PixelCalibrationError> for PixelCalibrationRowU16F32PreparationError {
    fn from(error: PixelCalibrationError) -> Self {
        Self::Calibration(error)
    }
}

/// Invalid row-block calibration sample metadata or numerical execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelCalibrationRowU16F32ProcessError {
    /// The row-block input did not carry sample identity metadata.
    MissingSampleMetadata,
    /// The transport offset did not fit the native detector-row extent.
    OffsetOverflow,
    /// The first row was not aligned to the configured row-block height.
    MisalignedOffset {
        /// Supplied zero-based first detector row.
        first_row: usize,
        /// Configured number of rows per block.
        block_rows: usize,
    },
    /// The block extended beyond the prepared detector height.
    RowRange,
    /// The terminal marker did not agree with the block ending at the detector height.
    Terminal,
    /// The prepared pixel-calibration operation rejected the block.
    Process(ProcessError),
}

impl fmt::Display for PixelCalibrationRowU16F32ProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSampleMetadata => {
                formatter.write_str("row-block input requires sample metadata")
            }
            Self::OffsetOverflow => formatter.write_str("row-block offset exceeds usize"),
            Self::MisalignedOffset {
                first_row,
                block_rows,
            } => write!(
                formatter,
                "row-block first row {first_row} is not aligned to {block_rows} rows",
            ),
            Self::RowRange => formatter.write_str("row block exceeds the detector height"),
            Self::Terminal => formatter
                .write_str("row-block terminal marker does not agree with the detector height"),
            Self::Process(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PixelCalibrationRowU16F32ProcessError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Process(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ProcessError> for PixelCalibrationRowU16F32ProcessError {
    fn from(error: ProcessError) -> Self {
        Self::Process(error)
    }
}

impl fmt::Display for PixelCalibrationU16F32PreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageExtentOverflow => formatter.write_str("detector image extent overflow"),
            Self::Calibration(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for PixelCalibrationU16F32PreparationError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::ImageExtentOverflow => None,
            Self::Calibration(error) => Some(error),
        }
    }
}

impl From<PixelCalibrationError> for PixelCalibrationU16F32PreparationError {
    fn from(error: PixelCalibrationError) -> Self {
        Self::Calibration(error)
    }
}

/// Prepared detector calibration for raw representation `R` and floating
/// representation `T`.
///
/// For every detector pixel, processing evaluates, in the written order:
///
/// ```text
/// calibrated[i] = flat[i] * (T(raw[i]) - background[i])
/// ```
///
/// The plan shares ownership of immutable flat and background arrays. A
/// replacement plan can therefore reuse either unchanged plane without a
/// copy. Conversion from `R` to `T` occurs at the arithmetic boundary.
/// On the canonical AO boundary, raw and background values are detector ADU,
/// `flat` is the complete PDE/ADU calibration coefficient, and output is PDE.
#[derive(Clone, Debug)]
pub struct PixelCalibrationPlan<T, R> {
    flat: Arc<[T]>,
    background: Arc<[T]>,
    raw: PhantomData<fn(R)>,
}

impl<T, R> PixelCalibrationPlan<T, R>
where
    T: Float,
{
    /// Prepares calibration from finite, equally sized arrays.
    pub fn new(
        flat: impl Into<Arc<[T]>>,
        background: impl Into<Arc<[T]>>,
    ) -> Result<Self, PixelCalibrationError> {
        let flat = flat.into();
        let background = background.into();
        if flat.is_empty() {
            return Err(PixelCalibrationError::Empty);
        }
        if flat.len() != background.len() {
            return Err(PixelCalibrationError::ExtentMismatch {
                flat: flat.len(),
                background: background.len(),
            });
        }
        if let Some(index) = flat.iter().position(|value| !value.is_finite()) {
            return Err(PixelCalibrationError::NonFiniteFlat { index });
        }
        if let Some(index) = background.iter().position(|value| !value.is_finite()) {
            return Err(PixelCalibrationError::NonFiniteBackground { index });
        }

        Ok(Self {
            flat,
            background,
            raw: PhantomData,
        })
    }

    /// Returns shared ownership of the immutable flat calibration.
    ///
    /// This permits an authoritative execution owner to prepare a replacement
    /// plan that changes only the background without copying the unchanged
    /// flat plane.
    pub fn shared_flat(&self) -> Arc<[T]> {
        Arc::clone(&self.flat)
    }

    /// Returns shared ownership of the immutable background calibration.
    ///
    /// This permits an authoritative execution owner to prepare a replacement
    /// plan that changes only the flat without copying the unchanged
    /// background plane.
    pub fn shared_background(&self) -> Arc<[T]> {
        Arc::clone(&self.background)
    }

    /// Prepares an equal-extent replacement flat with the active background.
    pub fn with_flat(&self, flat: &[T]) -> Result<Self, PixelCalibrationError> {
        Self::new(Arc::from(flat), self.shared_background())
    }

    /// Prepares an equal-extent replacement background with the active flat.
    pub fn with_background(&self, background: &[T]) -> Result<Self, PixelCalibrationError> {
        Self::new(self.shared_flat(), Arc::from(background))
    }

    /// Returns the logically read-only flat calibration.
    pub fn flat(&self) -> &[T] {
        &self.flat
    }

    /// Returns the logically read-only background calibration.
    pub fn background(&self) -> &[T] {
        &self.background
    }

    /// Returns the prepared detector-pixel extent.
    pub fn len(&self) -> usize {
        self.flat.len()
    }

    /// Returns `false`; valid calibration is always nonempty.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Creates idle progressive sequence state.
    pub const fn workspace(&self) -> ProgressWorkspace {
        ProgressWorkspace::new()
    }

    fn validate(&self, output_len: usize, input_len: usize) -> Result<(), ProcessError> {
        if input_len != self.len() {
            return Err(ProcessError::InputExtent {
                expected: self.len(),
                actual: input_len,
            });
        }
        if output_len != self.len() {
            return Err(ProcessError::OutputExtent {
                expected: self.len(),
                actual: output_len,
            });
        }
        Ok(())
    }

    fn calibrate(&self, output: &mut [T], raw: &[R], workrange: Range<usize>)
    where
        T: 'static,
        R: Copy + AsPrimitive<T> + 'static,
    {
        for index in workrange {
            output[index] = self.flat[index] * (raw[index].as_() - self.background[index]);
        }
    }

    /// Calibrate one complete input window at its detector-pixel range.
    ///
    /// `raw` and `output` contain only the values in `workrange`. This is the
    /// ordinary block-oriented counterpart to progressive processing of a
    /// caller-owned complete detector frame.
    pub fn process_window(
        &self,
        output: &mut [T],
        raw: &[R],
        workrange: Range<usize>,
    ) -> Result<(), ProcessError>
    where
        T: 'static,
        R: Copy + AsPrimitive<T> + 'static,
    {
        if workrange.start >= workrange.end {
            return Err(ProcessError::EmptyRange);
        }
        if workrange.end > self.len() {
            return Err(ProcessError::RangeOutOfBounds {
                end: workrange.end,
                extent: self.len(),
            });
        }
        let extent = workrange.end - workrange.start;
        if raw.len() != extent {
            return Err(ProcessError::InputExtent {
                expected: extent,
                actual: raw.len(),
            });
        }
        if output.len() != extent {
            return Err(ProcessError::OutputExtent {
                expected: extent,
                actual: output.len(),
            });
        }
        for (local, detector) in workrange.enumerate() {
            output[local] = self.flat[detector] * (raw[local].as_() - self.background[detector]);
        }
        Ok(())
    }

    /// Processes one contiguous semantic range into range-local output storage.
    ///
    /// Unlike [`InputProgress::process_range`], `output` contains only the
    /// elements named by `workrange`. This form lets a progressive transport
    /// retain published output as immutable while granting the producer a
    /// mutable view of only the unpublished range. `input` remains a complete
    /// caller-owned frame, such as a private decoded detector scratch buffer.
    pub fn process_range_window(
        &self,
        output: &mut [T],
        workspace: &mut ProgressWorkspace,
        input: &[R],
        workrange: Range<usize>,
    ) -> Result<ProgressResult, ProcessError>
    where
        T: 'static,
        R: Copy + AsPrimitive<T> + 'static,
    {
        workspace.validate_range(&workrange, self.len())?;
        if input.len() != self.len() {
            return Err(ProcessError::InputExtent {
                expected: self.len(),
                actual: input.len(),
            });
        }
        if output.len() != workrange.len() {
            return Err(ProcessError::OutputExtent {
                expected: workrange.len(),
                actual: output.len(),
            });
        }

        let immutable_count = workrange.end;
        for (destination, index) in output.iter_mut().zip(workrange) {
            *destination = self.flat[index] * (input[index].as_() - self.background[index]);
        }
        workspace.advance(immutable_count);
        Ok(ProgressResult::new(immutable_count))
    }
}

impl<T, R> AlgorithmPlan for PixelCalibrationPlan<T, R>
where
    T: Float + 'static,
    R: Copy + AsPrimitive<T> + 'static,
{
    type Input<'a> = &'a [R];
    type Output<'a> = &'a mut [T];
    type Workspace = ProgressWorkspace;
    type Error = ProcessError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        workspace.ensure_idle()?;
        self.validate(output.len(), input.len())?;
        self.calibrate(output, input, 0..self.len());
        Ok(())
    }
}

impl<T, R> InputProgress for PixelCalibrationPlan<T, R>
where
    T: Float + 'static,
    R: Copy + AsPrimitive<T> + 'static,
{
    fn start(&self, workspace: &mut Self::Workspace) -> Result<(), Self::Error> {
        workspace.start()
    }

    fn process_range<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
        workrange: Range<usize>,
    ) -> Result<ProgressResult, Self::Error> {
        workspace.validate_range(&workrange, self.len())?;
        self.validate(output.len(), input.len())?;
        let window = &mut output[workrange.clone()];
        self.process_range_window(window, workspace, input, workrange)
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

impl<T, R> OutputProgress for PixelCalibrationPlan<T, R>
where
    T: Float + 'static,
    R: Copy + AsPrimitive<T> + 'static,
{
}

crate::filtergraph_algorithm! {
    /// Portable detector U16 ADU to Float32 PDE calibration declaration.
    pub PixelCalibrationU16F32 {
        label = "pixel-calibration-u16-f32";
        configuration PixelCalibrationU16F32Configuration {
            image_size: (usize, usize) = GraphRebuild;
            initial_flat: Option<Vec<f32>> = Construction;
            initial_background: Option<Vec<f32>> = Construction;
        }
        plan = PixelCalibrationPlan<f32, u16>;
        workspace = ProgressWorkspace;
        properties = none;
        dimensions |_plan, _workspace| {
            image_rows = configuration.image_size.0;
            image_columns = configuration.image_size.1;
        }
        inputs {
            raw: u16 [image_rows, image_columns] = "raw",
                schema = crate::schemas::RAW_DETECTOR_PIXELS_V1;
        }
        outputs {
            calibrated: f32 [image_rows, image_columns] = "calibrated",
                schema = crate::schemas::CALIBRATED_PIXELS_V1,
                metadata_from = "raw";
        }
        parameters {
            flat: f32 [image_rows, image_columns] = "flat",
                schema = crate::schemas::FLAT_CALIBRATION_V1,
                replace |plan, values| {
                    plan.with_flat(values)
                };
            background: f32 [image_rows, image_columns] = "background",
                schema = crate::schemas::BACKGROUND_CALIBRATION_V1,
                replace |plan, values| {
                    plan.with_background(values)
                };
        }
        parameter_error = PixelCalibrationError;
        prepare |configuration| {
            let pixel_count = configuration.image_size.0
                .checked_mul(configuration.image_size.1)
                .ok_or(PixelCalibrationU16F32PreparationError::ImageExtentOverflow)?;
            let initial_flat = configuration
                .initial_flat
                .unwrap_or_else(|| vec![1.0; pixel_count]);
            let initial_background = configuration
                .initial_background
                .unwrap_or_else(|| vec![0.0; pixel_count]);
            if initial_flat.len() != pixel_count {
                return Err(PixelCalibrationU16F32PreparationError::Calibration(
                    PixelCalibrationError::ExtentMismatch {
                        flat: initial_flat.len(),
                        background: pixel_count,
                    },
                ));
            }
            if initial_background.len() != pixel_count {
                return Err(PixelCalibrationU16F32PreparationError::Calibration(
                    PixelCalibrationError::ExtentMismatch {
                        flat: pixel_count,
                        background: initial_background.len(),
                    },
                ));
            }
            let plan = PixelCalibrationPlan::new(
                initial_flat,
                initial_background,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = PixelCalibrationU16F32PreparationError;
        process |plan, workspace| {
            plan.process(calibrated, workspace, raw)
        }
        process_error = ProcessError;
        reset |plan, workspace| {
            *workspace = plan.workspace();
        }
    }
}

/// Bounded row-calibration publication state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelCalibrationRowWorkspace {
    image_rows: usize,
    image_columns: usize,
    row_block_rows: usize,
    pending_discontinuity: bool,
}

impl PixelCalibrationRowWorkspace {
    fn new(image_rows: usize, image_columns: usize, row_block_rows: usize) -> Self {
        Self {
            image_rows,
            image_columns,
            row_block_rows,
            pending_discontinuity: false,
        }
    }

    /// Mark the next complete calibrated block discontinuous.
    pub fn mark_discontinuity(&mut self) {
        self.pending_discontinuity = true;
    }

    fn take_discontinuity(&mut self) -> bool {
        core::mem::take(&mut self.pending_discontinuity)
    }
}

crate::filtergraph_algorithm! {
    /// Portable detector-row U16 ADU to Float32 PDE calibration declaration.
    pub PixelCalibrationRowU16F32 {
        label = "pixel-calibration-row-u16-f32";
        configuration PixelCalibrationRowU16F32Configuration {
            image_size: (usize, usize) = GraphRebuild;
            row_block_rows: usize = GraphRebuild;
            initial_flat: Option<Vec<f32>> = Construction;
            initial_background: Option<Vec<f32>> = Construction;
        }
        plan = PixelCalibrationPlan<f32, u16>;
        workspace = PixelCalibrationRowWorkspace;
        properties = none;
        dimensions |_plan, _workspace| {
            image_rows = configuration.image_size.0;
            image_columns = configuration.image_size.1;
            row_block_rows = configuration.row_block_rows;
            block_rate_multiplier = u32::try_from(image_rows / row_block_rows)
                .map_err(|_| crate::DeclarationPreparationError::Format(crate::FormatError::Rate))?;
        }
        inputs {
            raw: u16 [row_block_rows, image_columns] = "raw",
                schema = crate::schemas::RAW_PIXEL_ROW_BLOCK_V1,
                sample = raw_sample,
                rate = [block_rate_multiplier, 1];
        }
        outputs {
            calibrated: f32 [row_block_rows, image_columns] = "calibrated",
                schema = crate::schemas::CALIBRATED_PIXEL_ROW_BLOCK_V1,
                metadata_from = "raw",
                rate = [block_rate_multiplier, 1];
        }
        parameters {
            flat: f32 [image_rows, image_columns] = "flat",
                schema = crate::schemas::FLAT_CALIBRATION_V1,
                replace |plan, values| {
                    plan.with_flat(values)
                };
            background: f32 [image_rows, image_columns] = "background",
                schema = crate::schemas::BACKGROUND_CALIBRATION_V1,
                replace |plan, values| {
                    plan.with_background(values)
                };
        }
        parameter_error = PixelCalibrationError;
        prepare |configuration| {
            if configuration.row_block_rows == 0
                || configuration.row_block_rows >= configuration.image_size.0
                || configuration.image_size.0 % configuration.row_block_rows != 0
            {
                return Err(PixelCalibrationRowU16F32PreparationError::RowBlockRows);
            }
            let pixel_count = configuration.image_size.0
                .checked_mul(configuration.image_size.1)
                .ok_or(PixelCalibrationRowU16F32PreparationError::ImageExtentOverflow)?;
            let initial_flat = configuration
                .initial_flat
                .unwrap_or_else(|| vec![1.0; pixel_count]);
            let initial_background = configuration
                .initial_background
                .unwrap_or_else(|| vec![0.0; pixel_count]);
            if initial_flat.len() != pixel_count {
                return Err(PixelCalibrationRowU16F32PreparationError::Calibration(
                    PixelCalibrationError::ExtentMismatch {
                        flat: initial_flat.len(),
                        background: pixel_count,
                    },
                ));
            }
            if initial_background.len() != pixel_count {
                return Err(PixelCalibrationRowU16F32PreparationError::Calibration(
                    PixelCalibrationError::ExtentMismatch {
                        flat: pixel_count,
                        background: initial_background.len(),
                    },
                ));
            }
            Ok((
                PixelCalibrationPlan::new(initial_flat, initial_background)?,
                PixelCalibrationRowWorkspace::new(
                    configuration.image_size.0,
                    configuration.image_size.1,
                    configuration.row_block_rows,
                ),
            ))
        }
        prepare_error = PixelCalibrationRowU16F32PreparationError;
        process |plan, workspace| {
            let sample = raw_sample
                .ok_or(PixelCalibrationRowU16F32ProcessError::MissingSampleMetadata)?;
            let first_row = usize::try_from(sample.offset())
                .map_err(|_| PixelCalibrationRowU16F32ProcessError::OffsetOverflow)?;
            if first_row % workspace.row_block_rows != 0 {
                return Err(PixelCalibrationRowU16F32ProcessError::MisalignedOffset {
                    first_row,
                    block_rows: workspace.row_block_rows,
                });
            }
            let end_row = first_row
                .checked_add(workspace.row_block_rows)
                .ok_or(PixelCalibrationRowU16F32ProcessError::RowRange)?;
            if end_row > workspace.image_rows {
                return Err(PixelCalibrationRowU16F32ProcessError::RowRange);
            }
            if sample.is_terminal() != (end_row == workspace.image_rows) {
                return Err(PixelCalibrationRowU16F32ProcessError::Terminal);
            }
            let start = first_row
                .checked_mul(workspace.image_columns)
                .ok_or(PixelCalibrationRowU16F32ProcessError::RowRange)?;
            let end = end_row
                .checked_mul(workspace.image_columns)
                .ok_or(PixelCalibrationRowU16F32ProcessError::RowRange)?;
            plan.process_window(calibrated, raw, start..end)?;
            let pending_discontinuity = workspace.take_discontinuity();
            let discontinuity = sample.is_discontinuous() || pending_discontinuity;
            Ok(crate::AlgorithmProcessResult::complete_with_metadata(
                sample.with_discontinuity(discontinuity),
            ))
        }
        process_error = PixelCalibrationRowU16F32ProcessError;
        reset |_plan, workspace| {
            workspace.mark_discontinuity();
        }
        plan_adopted |_plan, workspace| {
            workspace.mark_discontinuity();
        }
    }
}
