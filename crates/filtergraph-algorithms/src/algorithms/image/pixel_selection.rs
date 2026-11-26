//! Prepared arbitrary detector-pixel selection.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::ops::Range;

use crate::{
    AlgorithmPlan, InputProgress, OutputProgress, ProcessError, ProgressResult, ProgressWorkspace,
};

/// Invalid prepared pixel-selection geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelSelectionPlanError {
    /// Logical detector image has a zero dimension.
    EmptyImage,

    /// The mask does not contain one entry for every logical image pixel.
    MaskExtent {
        /// Required row-major mask extent.
        expected: usize,
        /// Supplied mask extent.
        actual: usize,
    },

    /// The mask selects no pixels.
    NoSelectedPixels,

    /// Physical row stride cannot contain the logical image width.
    RowStrideTooSmall {
        /// Logical image width.
        columns: usize,
        /// Supplied physical row stride.
        stride: usize,
    },

    /// Prepared detector geometry cannot be represented by `usize`.
    LayoutOverflow,
}

impl fmt::Display for PixelSelectionPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage => formatter.write_str("image dimensions must be positive"),
            Self::MaskExtent { expected, actual } => write!(
                formatter,
                "pixel-selection mask extent {actual} does not match {expected}"
            ),
            Self::NoSelectedPixels => {
                formatter.write_str("pixel-selection mask must select pixels")
            }
            Self::RowStrideTooSmall { columns, stride } => write!(
                formatter,
                "row stride {stride} is smaller than logical image width {columns}"
            ),
            Self::LayoutOverflow => formatter.write_str("detector layout overflows usize"),
        }
    }
}

impl core::error::Error for PixelSelectionPlanError {}

/// Invalid shape-preserving pixel-selection mask replacement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelSelectionParameterError {
    /// The replacement mask is not a valid selection for the detector shape.
    Plan(PixelSelectionPlanError),
    /// Changing the selected-pixel count requires rebuilding the graph.
    SelectedPixelCount {
        /// Selected-pixel count fixed by the active output format.
        expected: usize,
        /// Selected-pixel count produced by the replacement mask.
        actual: usize,
    },
}

impl From<PixelSelectionPlanError> for PixelSelectionParameterError {
    fn from(error: PixelSelectionPlanError) -> Self {
        Self::Plan(error)
    }
}

impl fmt::Display for PixelSelectionParameterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => error.fmt(formatter),
            Self::SelectedPixelCount { expected, actual } => write!(
                formatter,
                "replacement mask selects {actual} pixels but the graph format requires {expected}"
            ),
        }
    }
}

impl core::error::Error for PixelSelectionParameterError {}

fn image_index(
    first_pixel: usize,
    row_stride: usize,
    row: usize,
    column: usize,
) -> Result<usize, PixelSelectionPlanError> {
    row.checked_mul(row_stride)
        .and_then(|offset| offset.checked_add(column))
        .and_then(|offset| first_pixel.checked_add(offset))
        .ok_or(PixelSelectionPlanError::LayoutOverflow)
}

#[derive(Clone, Debug)]
enum PixelPacking {
    Contiguous(Range<usize>),
    Runs(Box<[Range<usize>]>),
    Indexed,
}

impl PixelPacking {
    fn prepare(indices: &[usize]) -> Self {
        let start = indices[0];
        // Validated image extent bounds every selected index plus one.
        let end = indices[indices.len() - 1] + 1;
        if end - start == indices.len() {
            return Self::Contiguous(start..end);
        }

        let runs = || indices.chunk_by(|previous, next| previous + 1 == *next);
        // In the Float32 sweep, copies broke even around four pixels per
        // run. Keep a margin for mixed runs and other storage layouts.
        const MIN_AVERAGE_RUN_LENGTH: usize = 16;
        if indices.len() / runs().count() < MIN_AVERAGE_RUN_LENGTH {
            return Self::Indexed;
        }

        Self::Runs(runs().map(|run| run[0]..run[0] + run.len()).collect())
    }
}

/// Prepared selection of an arbitrary row-major detector-pixel mask.
///
/// `image_size` uses `(rows, columns)` order and Rust's zero-based indexing.
/// The mask contains one entry per logical image pixel in row-major order.
/// Preparation converts selected entries to strictly increasing physical
/// input-storage indices while accounting for `first_pixel` and row padding.
/// This order is the output coordinate contract and must match the prepared
/// reconstructor's pixel order.
///
/// The plan copies selected values without interpreting saturation,
/// non-finite values, or pixel validity. Those policies belong to the image
/// operation that produces the selected input image.
///
/// For progressive execution, one semantic input unit is one selected pixel
/// in this canonical output order. Every completed range therefore makes the
/// corresponding output prefix immutable.
#[derive(Clone, Debug)]
pub struct PixelSelectionPlan<T> {
    mask: Arc<[bool]>,
    selected_indices: Box<[usize]>,
    packing: PixelPacking,
    image_size: (usize, usize),
    row_stride: usize,
    first_pixel: usize,
    image_extent: usize,
    pixel: PhantomData<fn(T)>,
}

impl<T> PixelSelectionPlan<T> {
    /// Prepares selection from a compact row-major detector image.
    pub fn new(
        image_size: (usize, usize),
        mask: impl Into<Arc<[bool]>>,
    ) -> Result<Self, PixelSelectionPlanError> {
        Self::with_layout(image_size, mask, image_size.1, 0)
    }

    /// Prepares selection with explicit physical detector layout.
    pub fn with_layout(
        image_size: (usize, usize),
        mask: impl Into<Arc<[bool]>>,
        row_stride: usize,
        first_pixel: usize,
    ) -> Result<Self, PixelSelectionPlanError> {
        let (image_rows, image_columns) = image_size;
        if image_rows == 0 || image_columns == 0 {
            return Err(PixelSelectionPlanError::EmptyImage);
        }
        let logical_extent = image_rows
            .checked_mul(image_columns)
            .ok_or(PixelSelectionPlanError::LayoutOverflow)?;
        let mask = mask.into();
        if mask.len() != logical_extent {
            return Err(PixelSelectionPlanError::MaskExtent {
                expected: logical_extent,
                actual: mask.len(),
            });
        }
        if row_stride < image_columns {
            return Err(PixelSelectionPlanError::RowStrideTooSmall {
                columns: image_columns,
                stride: row_stride,
            });
        }

        let image_extent = image_index(first_pixel, row_stride, image_rows - 1, image_columns - 1)?
            .checked_add(1)
            .ok_or(PixelSelectionPlanError::LayoutOverflow)?;

        let mut selected_indices = Vec::new();
        for (logical_index, selected) in mask.iter().copied().enumerate() {
            if selected {
                let row = logical_index / image_columns;
                let column = logical_index % image_columns;
                selected_indices.push(image_index(first_pixel, row_stride, row, column)?);
            }
        }
        if selected_indices.is_empty() {
            return Err(PixelSelectionPlanError::NoSelectedPixels);
        }

        let packing = PixelPacking::prepare(&selected_indices);
        Ok(Self {
            mask,
            selected_indices: selected_indices.into_boxed_slice(),
            packing,
            image_size,
            row_stride,
            first_pixel,
            image_extent,
            pixel: PhantomData,
        })
    }

    /// Returns the logical row-major selection mask.
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// Returns shared ownership of the logical row-major selection mask.
    pub fn shared_mask(&self) -> Arc<[bool]> {
        Arc::clone(&self.mask)
    }

    /// Returns selected physical input indices in canonical output order.
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }

    /// Returns logical detector size in `(rows, columns)` order.
    pub const fn image_size(&self) -> (usize, usize) {
        self.image_size
    }

    /// Returns physical detector-row stride in elements.
    pub const fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Returns the zero-based input index of logical detector pixel `(0, 0)`.
    pub const fn first_pixel(&self) -> usize {
        self.first_pixel
    }

    /// Returns the minimum caller-owned input-storage extent.
    pub const fn image_extent(&self) -> usize {
        self.image_extent
    }

    /// Returns the selected output-pixel extent.
    pub const fn len(&self) -> usize {
        self.selected_indices.len()
    }

    /// Returns `false`; valid selection contains at least one pixel.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Creates idle progressive sequence state.
    pub const fn workspace(&self) -> ProgressWorkspace {
        ProgressWorkspace::new()
    }

    fn validate(&self, output_len: usize, input_len: usize) -> Result<(), ProcessError> {
        if input_len < self.image_extent {
            return Err(ProcessError::InputExtent {
                expected: self.image_extent,
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

    fn select(&self, output: &mut [T], image: &[T], workrange: Range<usize>)
    where
        T: Copy,
    {
        for selected_index in workrange {
            output[selected_index] = image[self.selected_indices[selected_index]];
        }
    }
}

impl<T> AlgorithmPlan for PixelSelectionPlan<T>
where
    T: Copy + 'static,
{
    type Input<'a> = &'a [T];
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
        match &self.packing {
            PixelPacking::Contiguous(source) => output.copy_from_slice(&input[source.clone()]),
            PixelPacking::Runs(runs) => {
                let mut remaining = output;
                for source in runs {
                    let (dest, tail) = remaining.split_at_mut(source.len());
                    dest.copy_from_slice(&input[source.clone()]);
                    remaining = tail;
                }
            }
            PixelPacking::Indexed => self.select(output, input, 0..self.len()),
        }
        Ok(())
    }
}

impl<T> InputProgress for PixelSelectionPlan<T>
where
    T: Copy + 'static,
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

        let immutable_count = workrange.end;
        self.select(output, input, workrange);
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

impl<T> OutputProgress for PixelSelectionPlan<T> where T: Copy + 'static {}

crate::filtergraph_algorithm! {
    /// Portable Float32 packed-image pixel-selection declaration.
    pub PixelSelectionF32 {
        label = "pixel-selection-f32";
        configuration PixelSelectionF32Configuration {
            image_size: (usize, usize) = GraphRebuild;
            mask: Vec<bool> = GraphRebuild;
            input_schema: String = GraphRebuild;
            output_schema: String = GraphRebuild;
        }
        plan = PixelSelectionPlan<f32>;
        workspace = ProgressWorkspace;
        properties = none;
        dimensions |plan, _workspace| {
            image_rows = plan.image_size().0;
            image_columns = plan.image_size().1;
            selected_pixel_count = plan.len();
        }
        inputs {
            image: f32 [image_rows, image_columns] = "image",
                schema = configuration.input_schema.as_str(),
                optional_sample_metadata = ["chopped_image_kind", "trigger_state", "acquisition_identity", "exposure_duration_nanoseconds"];
        }
        outputs {
            selected_pixels: f32 [selected_pixel_count] = "selected-pixels",
                schema = configuration.output_schema.as_str(),
                metadata_from = "image";
        }
        parameters {
            mask: bool [image_rows, image_columns] = "mask",
                schema = crate::schemas::PIXEL_SELECTION_MASK_V1,
                replace |plan, values| {
                    let replacement = PixelSelectionPlan::new(
                        plan.image_size(),
                        Arc::<[bool]>::from(values),
                    )?;
                    if replacement.len() != plan.len() {
                        return Err(PixelSelectionParameterError::SelectedPixelCount {
                            expected: plan.len(),
                            actual: replacement.len(),
                        });
                    }
                    Ok(replacement)
                };
        }
        parameter_error = PixelSelectionParameterError;
        prepare |configuration| {
            let plan = PixelSelectionPlan::new(
                configuration.image_size,
                configuration.mask,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = PixelSelectionPlanError;
        process |plan, workspace| {
            plan.process(selected_pixels, workspace, image)
        }
        process_error = ProcessError;
        reset |plan, workspace| {
            *workspace = plan.workspace();
        }
    }
}
