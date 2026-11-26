//! Prepared acquisition-group packing for progressive matrix-vector accumulation.

use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
use core::fmt;

use crate::{MatrixElement, PreparedGemv};

use super::gemv::multiply_add;

/// Invalid progressive matrix-vector preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedProgressiveGemvError {
    /// At least one acquisition group is required.
    EmptyGroups,
    /// Completion ordinals do not contain one value per matrix input coordinate.
    CompletionExtent {
        /// Required matrix input-coordinate extent.
        expected: usize,
        /// Supplied completion-ordinal extent.
        actual: usize,
    },
    /// A completion ordinal is outside the prepared groups.
    OrdinalRange {
        /// Zero-based matrix input-coordinate index.
        input: usize,
        /// Supplied completion ordinal.
        ordinal: usize,
        /// Prepared group count.
        groups: usize,
    },
    /// Prepared storage extent arithmetic overflowed.
    ExtentOverflow,
}

impl fmt::Display for PreparedProgressiveGemvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGroups => formatter.write_str("at least one matrix input group is required"),
            Self::CompletionExtent { expected, actual } => write!(
                formatter,
                "completion-ordinal extent {actual} does not match matrix input extent {expected}"
            ),
            Self::OrdinalRange {
                input,
                ordinal,
                groups,
            } => write!(
                formatter,
                "matrix input {input} completion ordinal {ordinal} is outside {groups} groups"
            ),
            Self::ExtentOverflow => formatter.write_str("progressive matrix extent overflow"),
        }
    }
}

impl core::error::Error for PreparedProgressiveGemvError {}

/// Invalid progressive matrix-vector accumulation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgressiveGemvError {
    /// The canonical input vector has the wrong extent.
    InputExtent {
        /// Required matrix input extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// The reconstructed output vector has the wrong extent.
    OutputExtent {
        /// Required matrix output extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// The requested acquisition group is outside the prepared schedule.
    OrdinalRange {
        /// Supplied acquisition ordinal.
        ordinal: usize,
        /// Prepared group count.
        groups: usize,
    },
    /// One newly consumed canonical input coordinate is not finite.
    NonFiniteInput {
        /// Zero-based canonical matrix input-coordinate index.
        input: usize,
    },
    /// Progressive accumulation produced a non-finite output.
    NonFiniteOutput,
}

impl fmt::Display for ProgressiveGemvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent { expected, actual } => write!(
                formatter,
                "matrix input extent {actual} does not match {expected}"
            ),
            Self::OutputExtent { expected, actual } => write!(
                formatter,
                "matrix output extent {actual} does not match {expected}"
            ),
            Self::OrdinalRange { ordinal, groups } => {
                write!(
                    formatter,
                    "matrix group {ordinal} is outside {groups} groups"
                )
            }
            Self::NonFiniteInput { input } => {
                write!(formatter, "matrix input coordinate {input} is not finite")
            }
            Self::NonFiniteOutput => {
                formatter.write_str("progressive matrix accumulation produced a non-finite output")
            }
        }
    }
}

impl core::error::Error for ProgressiveGemvError {}

/// Prepared reconstructor panels grouped by scientific-coordinate readiness.
///
/// The retained canonical matrix remains row-major and unchanged. Packed
/// coefficients place all output coefficients for one input coordinate
/// contiguously in deterministic acquisition-group and canonical-coordinate
/// order.
#[derive(Clone, Debug)]
pub struct PreparedProgressiveGemv<T> {
    canonical: PreparedGemv<T>,
    group_offsets: Box<[usize]>,
    input_indices: Box<[usize]>,
    coefficients: Arc<[T]>,
}

impl<T> PreparedProgressiveGemv<T>
where
    T: MatrixElement,
{
    /// Packs canonical matrix columns by their zero-based completion ordinal.
    pub fn new(
        canonical: PreparedGemv<T>,
        completion_ordinals: &[usize],
        groups: usize,
    ) -> Result<Self, PreparedProgressiveGemvError> {
        if groups == 0 {
            return Err(PreparedProgressiveGemvError::EmptyGroups);
        }
        if completion_ordinals.len() != canonical.columns() {
            return Err(PreparedProgressiveGemvError::CompletionExtent {
                expected: canonical.columns(),
                actual: completion_ordinals.len(),
            });
        }
        let mut counts = vec![0_usize; groups];
        for (input, &ordinal) in completion_ordinals.iter().enumerate() {
            let Some(count) = counts.get_mut(ordinal) else {
                return Err(PreparedProgressiveGemvError::OrdinalRange {
                    input,
                    ordinal,
                    groups,
                });
            };
            *count = count
                .checked_add(1)
                .ok_or(PreparedProgressiveGemvError::ExtentOverflow)?;
        }
        let mut group_offsets: Vec<usize> = Vec::with_capacity(groups + 1);
        group_offsets.push(0);
        for count in counts {
            let next = group_offsets
                .last()
                .copied()
                .and_then(|offset| offset.checked_add(count))
                .ok_or(PreparedProgressiveGemvError::ExtentOverflow)?;
            group_offsets.push(next);
        }
        let mut cursors = group_offsets[..groups].to_vec();
        let mut input_indices = vec![usize::MAX; canonical.columns()];
        for (input, &ordinal) in completion_ordinals.iter().enumerate() {
            let cursor = &mut cursors[ordinal];
            input_indices[*cursor] = input;
            *cursor += 1;
        }
        let extent = canonical
            .rows()
            .checked_mul(canonical.columns())
            .ok_or(PreparedProgressiveGemvError::ExtentOverflow)?;
        let mut coefficients = Vec::with_capacity(extent);
        for &input in &input_indices {
            for output in 0..canonical.rows() {
                coefficients.push(canonical.matrix()[output * canonical.columns() + input]);
            }
        }
        Ok(Self {
            canonical,
            group_offsets: group_offsets.into_boxed_slice(),
            input_indices: input_indices.into_boxed_slice(),
            coefficients: Arc::from(coefficients),
        })
    }

    /// Returns the unchanged canonical row-major matrix operation.
    pub const fn canonical(&self) -> &PreparedGemv<T> {
        &self.canonical
    }

    /// Returns the number of prepared acquisition groups.
    pub fn groups(&self) -> usize {
        self.group_offsets.len() - 1
    }

    /// Returns canonical input-coordinate indices for one acquisition group.
    pub fn input_indices(&self, ordinal: usize) -> Option<&[usize]> {
        let start = *self.group_offsets.get(ordinal)?;
        let end = *self.group_offsets.get(ordinal + 1)?;
        self.input_indices.get(start..end)
    }

    /// Returns packed input-coordinate-major coefficients for inspection.
    pub fn coefficients(&self) -> &[T] {
        &self.coefficients
    }

    /// Accumulates one acquisition group's matrix contribution into `output`.
    pub fn accumulate_group(
        &self,
        output: &mut [T],
        input: &[T],
        ordinal: usize,
    ) -> Result<(), ProgressiveGemvError> {
        if input.len() != self.canonical.columns() {
            return Err(ProgressiveGemvError::InputExtent {
                expected: self.canonical.columns(),
                actual: input.len(),
            });
        }
        if output.len() != self.canonical.rows() {
            return Err(ProgressiveGemvError::OutputExtent {
                expected: self.canonical.rows(),
                actual: output.len(),
            });
        }
        let Some(indices) = self.input_indices(ordinal) else {
            return Err(ProgressiveGemvError::OrdinalRange {
                ordinal,
                groups: self.groups(),
            });
        };
        if let Some(&input_index) = indices.iter().find(|&&index| !input[index].is_finite()) {
            return Err(ProgressiveGemvError::NonFiniteInput { input: input_index });
        }
        let first = self.group_offsets[ordinal];
        for (packed_input, &input_index) in indices.iter().enumerate() {
            let coefficient_start = (first + packed_input) * self.canonical.rows();
            let coefficients =
                &self.coefficients[coefficient_start..coefficient_start + self.canonical.rows()];
            multiply_add(input[input_index], coefficients, output);
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(ProgressiveGemvError::NonFiniteOutput);
        }
        Ok(())
    }
}
