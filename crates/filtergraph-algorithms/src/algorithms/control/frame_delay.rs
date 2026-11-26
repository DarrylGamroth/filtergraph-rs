//! Fixed frame delay measured in successful local adoptions.

use alloc::{boxed::Box, string::String, vec, vec::Vec};
use core::{fmt, marker::PhantomData, mem::size_of};
use num_traits::Float;

use crate::{AlgorithmPlan, ProcessError};

/// Invalid frame-delay construction configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameDelayPreparationError {
    /// The frame has no coordinates.
    EmptyState,
    /// A construction-state coordinate is not finite.
    NonFiniteInitialState,
    /// History size overflows or declaration history byte extent exceeds UInt32.
    HistoryExtent,
}
impl fmt::Display for FrameDelayPreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::EmptyState => "frame-delay extent must be positive",
            Self::NonFiniteInitialState => "initial delay state must be finite",
            Self::HistoryExtent => "frame-delay history byte extent is invalid",
        })
    }
}
impl core::error::Error for FrameDelayPreparationError {}

/// Numerical rejection or incompatible caller storage for a frame delay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameDelayError {
    /// Caller input or output has an incompatible extent.
    Process(ProcessError),
    /// Workspace history was prepared for another delay.
    HistoryExtent,
    /// The input contains a nonfinite coordinate.
    NonFiniteInput,
}
impl fmt::Display for FrameDelayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Process(error) => error.fmt(f),
            Self::HistoryExtent => f.write_str("frame-delay history has an incompatible extent"),
            Self::NonFiniteInput => f.write_str("frame-delay input must be finite"),
        }
    }
}
impl core::error::Error for FrameDelayError {}
impl From<ProcessError> for FrameDelayError {
    fn from(error: ProcessError) -> Self {
        Self::Process(error)
    }
}

/// A fixed delay in successfully adopted samples, independent of elapsed time.
///
/// A complete successful `process` call publishes the old history frame and
/// adopts the input once. Rejection preserves caller output and history. A
/// later algorithm rejecting the graph sample does not undo this adoption.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameDelayPlan<T> {
    delay_frames: usize,
    scalar: PhantomData<T>,
}
impl<T> FrameDelayPlan<T> {
    /// Prepare a nonnegative delay. Zero copies the current finite input.
    pub const fn new(delay_frames: usize) -> Self {
        Self {
            delay_frames,
            scalar: PhantomData,
        }
    }
    /// Return the number of successful local adoptions retained in history.
    pub const fn delay_frames(&self) -> usize {
        self.delay_frames
    }
}

fn history_extent<T>(
    extent: usize,
    delay_frames: usize,
) -> Result<usize, FrameDelayPreparationError> {
    if extent == 0 {
        return Err(FrameDelayPreparationError::EmptyState);
    }
    let initial_bytes = extent
        .checked_mul(size_of::<T>())
        .ok_or(FrameDelayPreparationError::HistoryExtent)?;
    if initial_bytes > isize::MAX as usize {
        return Err(FrameDelayPreparationError::HistoryExtent);
    }
    let elements = extent
        .checked_mul(delay_frames)
        .ok_or(FrameDelayPreparationError::HistoryExtent)?;
    let history_bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or(FrameDelayPreparationError::HistoryExtent)?;
    if history_bytes > isize::MAX as usize {
        return Err(FrameDelayPreparationError::HistoryExtent);
    }
    Ok(elements)
}

fn validate_f32_configuration(
    extent: usize,
    delay_frames: usize,
) -> Result<(), FrameDelayPreparationError> {
    let elements = history_extent::<f32>(extent, delay_frames)?;
    let bytes = elements * size_of::<f32>();
    if bytes > u32::MAX as usize || extent > u32::MAX as usize / size_of::<f32>() {
        return Err(FrameDelayPreparationError::HistoryExtent);
    }
    Ok(())
}

/// Construction state and committed history for a fixed frame delay.
///
/// History is packed by ring frame, then coordinate. The zero-based `head`
/// identifies the frame emitted on the next successful positive-delay call.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameDelayWorkspace<T> {
    initial_state: Box<[T]>,
    history: Box<[T]>,
    head: usize,
}
impl<T: Float> FrameDelayWorkspace<T> {
    /// Copy a finite nonempty construction frame into every history position.
    /// Direct callers use checked native-size history extents; the Float32
    /// declaration additionally limits history byte extent to UInt32.
    pub fn new(
        initial_state: Vec<T>,
        plan: &FrameDelayPlan<T>,
    ) -> Result<Self, FrameDelayPreparationError> {
        let elements = history_extent::<T>(initial_state.len(), plan.delay_frames)?;
        if initial_state.iter().any(|value| !value.is_finite()) {
            return Err(FrameDelayPreparationError::NonFiniteInitialState);
        }
        let mut history = vec![T::zero(); elements];
        for frame in history.chunks_exact_mut(initial_state.len()) {
            frame.copy_from_slice(&initial_state);
        }
        Ok(Self {
            initial_state: initial_state.into_boxed_slice(),
            history: history.into_boxed_slice(),
            head: 0,
        })
    }
    /// Prepare a nonempty frame whose construction coordinates share one value.
    pub fn filled(
        extent: usize,
        initial_value: T,
        plan: &FrameDelayPlan<T>,
    ) -> Result<Self, FrameDelayPreparationError> {
        history_extent::<T>(extent, plan.delay_frames)?;
        if !initial_value.is_finite() {
            return Err(FrameDelayPreparationError::NonFiniteInitialState);
        }
        Self::new(vec![initial_value; extent], plan)
    }
    /// Return the prepared coordinate count.
    pub const fn len(&self) -> usize {
        self.initial_state.len()
    }
    /// Return false; a prepared workspace is nonempty.
    pub const fn is_empty(&self) -> bool {
        false
    }
    /// Borrow committed history in ring-frame, then coordinate order.
    pub fn history(&self) -> &[T] {
        &self.history
    }
    /// Return the zero-based ring frame emitted by the next valid call.
    pub const fn head(&self) -> usize {
        self.head
    }
    /// Restore every history frame to construction state and the head to zero.
    pub fn reset(&mut self) {
        for frame in self.history.chunks_exact_mut(self.initial_state.len()) {
            frame.copy_from_slice(&self.initial_state);
        }
        self.head = 0;
    }
}
impl<T: Float + 'static> AlgorithmPlan for FrameDelayPlan<T> {
    type Input<'a> = &'a [T];
    type Output<'a> = &'a mut [T];
    type Workspace = FrameDelayWorkspace<T>;
    type Error = FrameDelayError;
    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        let extent = workspace.len();
        if input.len() != extent {
            return Err(ProcessError::InputExtent {
                expected: extent,
                actual: input.len(),
            }
            .into());
        }
        if output.len() != extent {
            return Err(ProcessError::OutputExtent {
                expected: extent,
                actual: output.len(),
            }
            .into());
        }
        if extent.checked_mul(self.delay_frames) != Some(workspace.history.len()) {
            return Err(FrameDelayError::HistoryExtent);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(FrameDelayError::NonFiniteInput);
        }
        if self.delay_frames == 0 {
            output.copy_from_slice(input);
        } else {
            let start = workspace.head * extent;
            let frame = &mut workspace.history[start..start + extent];
            output.copy_from_slice(frame);
            frame.copy_from_slice(input);
            workspace.head = if workspace.head + 1 == self.delay_frames {
                0
            } else {
                workspace.head + 1
            };
        }
        Ok(())
    }
}

crate::filtergraph_algorithm! {
    /// Delay a Float32 vector by successful local adoptions; propagate current input metadata.
    pub FrameDelayF32 {
        label = "frame-delay-f32";
        configuration FrameDelayF32Configuration {
            extent: usize = GraphRebuild;
            delay_frames: usize = GraphRebuild;
            initial_value: f32 = Construction;
            input_schema: String = GraphRebuild;
            output_schema: String = GraphRebuild;
        }
        plan = FrameDelayPlan<f32>;
        properties = none;
        dimensions |_plan, workspace| { extent = workspace.len(); }
        inputs { input: f32 [extent] = "input", schema = configuration.input_schema.as_str(); }
        outputs { output: f32 [extent] = "output", schema = configuration.output_schema.as_str(), metadata_from = "input"; }
        prepare |configuration| {
            validate_f32_configuration(configuration.extent, configuration.delay_frames)?;
            let plan = FrameDelayPlan::new(configuration.delay_frames);
            let workspace = FrameDelayWorkspace::filled(configuration.extent, configuration.initial_value, &plan)?;
            Ok((plan, workspace))
        }
        prepare_error = FrameDelayPreparationError;
        process |plan, workspace| { plan.process(output, workspace, input) }
        process_error = FrameDelayError;
        reset |_plan, workspace| { workspace.reset(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_checks_uint32_boundary_without_allocating() {
        let frames = u32::MAX as usize / size_of::<f32>();
        assert_eq!(validate_f32_configuration(1, frames), Ok(()));
        assert_eq!(
            validate_f32_configuration(1, frames + 1),
            Err(FrameDelayPreparationError::HistoryExtent)
        );
        #[cfg(target_pointer_width = "64")]
        assert!(history_extent::<f64>(1, frames + 1).is_ok());
    }
}
