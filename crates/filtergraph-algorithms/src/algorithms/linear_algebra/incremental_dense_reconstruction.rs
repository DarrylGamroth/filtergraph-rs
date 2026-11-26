//! Sensor-independent dense reconstruction from indexed measurement blocks.

use alloc::{boxed::Box, string::String, sync::Arc, vec, vec::Vec};
use core::{fmt, ops::Range};

use crate::composites::row_block::{RowBlockDropReason, RowBlockInfo, RowBlockResult};
use crate::{AlgorithmProcessResult, MatrixElement, PreparedGemv, PreparedGemvError, multiply_add};

/// Invalid preparation or shape-preserving reconstructor replacement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IncrementalDenseReconstructionPreparationError {
    /// The canonical reconstruction matrix is invalid.
    Reconstructor(PreparedGemvError),
    /// Measurement capacity and block count must both be positive.
    EmptyBlockDimension,
    /// Measurement column indices must fit the declared UInt32 representation.
    ColumnIndexRange,
    /// Prepared extent arithmetic overflowed.
    ExtentOverflow,
}
impl fmt::Display for IncrementalDenseReconstructionPreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reconstructor(error) => error.fmt(f),
            Self::EmptyBlockDimension => {
                f.write_str("measurement capacity and block count must be positive")
            }
            Self::ColumnIndexRange => f.write_str("measurement column indices must fit UInt32"),
            Self::ExtentOverflow => f.write_str("incremental reconstruction extent overflow"),
        }
    }
}
impl core::error::Error for IncrementalDenseReconstructionPreparationError {}
impl From<PreparedGemvError> for IncrementalDenseReconstructionPreparationError {
    fn from(error: PreparedGemvError) -> Self {
        Self::Reconstructor(error)
    }
}

/// Invalid caller storage, measurement coverage, or numerical result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IncrementalDenseReconstructionError {
    /// One input or output has an incompatible extent.
    StorageExtent,
    /// The count exceeds the prepared block capacity.
    CountOutOfRange,
    /// One zero-based reconstructor column is outside the matrix.
    ColumnOutOfRange,
    /// A measurement column contributed more than once to this sample.
    DuplicateColumn,
    /// One consumed measurement is not finite.
    NonFiniteInput,
    /// Incremental accumulation produced a non-finite value.
    NonFiniteOutput,
    /// The terminal block did not cover every measurement column.
    IncompleteSample,
    /// A helper selected an invalid batch or output partition.
    PartitionRange,
}
impl fmt::Display for IncrementalDenseReconstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::StorageExtent => "incremental reconstruction storage extent is invalid",
            Self::CountOutOfRange => "measurement count exceeds block capacity",
            Self::ColumnOutOfRange => "measurement column is outside the reconstructor",
            Self::DuplicateColumn => "measurement column contributes more than once",
            Self::NonFiniteInput => "measurement must be finite",
            Self::NonFiniteOutput => "incremental reconstruction produced a non-finite value",
            Self::IncompleteSample => "terminal sample does not cover every measurement column",
            Self::PartitionRange => "incremental reconstruction partition is invalid",
        })
    }
}
impl core::error::Error for IncrementalDenseReconstructionError {}

/// Failure from scientific processing or its synchronous deployment executor.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IncrementalDenseReconstructionExecutionError<E> {
    /// Scientific validation or numerical processing failed.
    Algorithm(IncrementalDenseReconstructionError),
    /// The deployment-owned reconstruction executor failed.
    Executor(E),
}

/// Failure at the portable declaration boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IncrementalDenseReconstructionProcessError {
    /// Indexed blocks require transport-neutral sample identity.
    MissingSampleMetadata,
    /// The sample-piece offset does not fit the current target.
    OffsetOverflow,
    /// Scientific validation or numerical processing failed.
    Algorithm(IncrementalDenseReconstructionError),
}
impl fmt::Display for IncrementalDenseReconstructionProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSampleMetadata => f.write_str("measurement sample metadata is required"),
            Self::OffsetOverflow => {
                f.write_str("measurement sample-piece offset does not fit usize")
            }
            Self::Algorithm(error) => error.fmt(f),
        }
    }
}
impl core::error::Error for IncrementalDenseReconstructionProcessError {}
impl From<IncrementalDenseReconstructionError> for IncrementalDenseReconstructionProcessError {
    fn from(error: IncrementalDenseReconstructionError) -> Self {
        Self::Algorithm(error)
    }
}

/// Immutable dense reconstruction with contiguous coefficients per input column.
///
/// Matrix columns retain the declared scientific coordinate order. Each block
/// supplies zero-based column indices; accumulation follows the received order
/// for deterministic per-output arithmetic in serial and helper execution.
#[derive(Clone, Debug)]
pub struct IncrementalDenseReconstructionPlan<T> {
    reconstructor: PreparedGemv<T>,
    by_column: Arc<[T]>,
    block_capacity: usize,
    blocks_per_sample: usize,
}
impl<T: MatrixElement> IncrementalDenseReconstructionPlan<T> {
    /// Packs matrix columns once during preparation.
    pub fn new(
        reconstructor: PreparedGemv<T>,
        block_capacity: usize,
        blocks_per_sample: usize,
    ) -> Result<Self, IncrementalDenseReconstructionPreparationError> {
        if block_capacity == 0 || blocks_per_sample == 0 {
            return Err(IncrementalDenseReconstructionPreparationError::EmptyBlockDimension);
        }
        if reconstructor.columns() - 1 > u32::MAX as usize {
            return Err(IncrementalDenseReconstructionPreparationError::ColumnIndexRange);
        }
        let extent = reconstructor
            .rows()
            .checked_mul(reconstructor.columns())
            .ok_or(IncrementalDenseReconstructionPreparationError::ExtentOverflow)?;
        let mut by_column = Vec::with_capacity(extent);
        for column in 0..reconstructor.columns() {
            for row in 0..reconstructor.rows() {
                by_column.push(reconstructor.matrix()[row * reconstructor.columns() + column]);
            }
        }
        Ok(Self {
            reconstructor,
            by_column: Arc::from(by_column),
            block_capacity,
            blocks_per_sample,
        })
    }
    /// Returns the canonical row-major reconstructor.
    pub const fn reconstructor(&self) -> &PreparedGemv<T> {
        &self.reconstructor
    }
    /// Returns the maximum measurements published in one block.
    pub const fn block_capacity(&self) -> usize {
        self.block_capacity
    }
    /// Returns the nominal input-to-output rate multiplier.
    pub const fn blocks_per_sample(&self) -> usize {
        self.blocks_per_sample
    }
    /// Replaces a canonical row-major matrix without changing prepared extents.
    pub fn with_reconstructor(
        &self,
        values: &[T],
    ) -> Result<Self, IncrementalDenseReconstructionPreparationError> {
        Self::new(
            PreparedGemv::new(
                self.reconstructor.rows(),
                self.reconstructor.columns(),
                Arc::from(values),
            )?,
            self.block_capacity,
            self.blocks_per_sample,
        )
    }
    /// Allocates bounded single-owner candidate and column-coverage storage.
    pub fn workspace(&self) -> IncrementalDenseReconstructionWorkspace<T> {
        IncrementalDenseReconstructionWorkspace {
            candidate: vec![T::zero(); self.reconstructor.rows()].into_boxed_slice(),
            seen: vec![false; self.reconstructor.columns()].into_boxed_slice(),
            seen_count: 0,
            sequence: None,
            last_offset: 0,
            pending_discontinuity: false,
            current_discontinuity: false,
        }
    }
    /// Accumulates one indexed measurement batch into a disjoint output shard.
    ///
    /// The coordinator validates duplicate columns before delegating. This safe
    /// adapter hook checks all borrowed extents and preserves measurement order.
    #[doc(hidden)]
    pub fn accumulate_reconstruction_partition(
        &self,
        measurements: &[T],
        columns: &[u32],
        output: &mut [T],
        batch: Range<usize>,
        first_output: usize,
    ) -> Result<(), IncrementalDenseReconstructionError> {
        if measurements.len() != columns.len()
            || batch.start > batch.end
            || batch.end > measurements.len()
        {
            return Err(IncrementalDenseReconstructionError::PartitionRange);
        }
        let last_output = first_output
            .checked_add(output.len())
            .filter(|&last| last <= self.reconstructor.rows())
            .ok_or(IncrementalDenseReconstructionError::PartitionRange)?;
        for index in batch.clone() {
            if columns[index] as usize >= self.reconstructor.columns() {
                return Err(IncrementalDenseReconstructionError::ColumnOutOfRange);
            }
            if !measurements[index].is_finite() {
                return Err(IncrementalDenseReconstructionError::NonFiniteInput);
            }
        }
        let rows = self.reconstructor.rows();
        let mut first = batch.start;
        // Keep the accumulator in registers across four columns. Apply their
        // fused updates in the same order as the single-column path.
        while batch.end - first >= 4 {
            let input = |index: usize| {
                let start = columns[index] as usize * rows;
                &self.by_column[start + first_output..start + last_output]
            };
            let (a, b, c, d) = (
                input(first),
                input(first + 1),
                input(first + 2),
                input(first + 3),
            );
            let (wa, wb, wc, wd) = (
                measurements[first],
                measurements[first + 1],
                measurements[first + 2],
                measurements[first + 3],
            );
            for ((((out, &a), &b), &c), &d) in output.iter_mut().zip(a).zip(b).zip(c).zip(d) {
                let z = a.mul_add(wa, *out);
                let z = b.mul_add(wb, z);
                let z = c.mul_add(wc, z);
                *out = d.mul_add(wd, z);
            }
            first += 4;
        }
        for index in first..batch.end {
            let start = columns[index] as usize * rows;
            multiply_add(
                measurements[index],
                &self.by_column[start + first_output..start + last_output],
                output,
            );
        }
        Ok(())
    }
    /// Consumes one indexed block; `info.first_row` is a strictly increasing sample-piece offset.
    /// Caller-owned output is changed only for complete, finite sample coverage.
    pub fn process_block(
        &self,
        output: &mut [T],
        workspace: &mut IncrementalDenseReconstructionWorkspace<T>,
        measurements: &[T],
        columns: &[u32],
        count: u32,
        info: RowBlockInfo,
    ) -> Result<RowBlockResult, IncrementalDenseReconstructionError> {
        match self.process_block_with_reconstruction(
            output,
            workspace,
            measurements,
            columns,
            count,
            info,
            |plan, values, indices, candidate| {
                plan.accumulate_reconstruction_partition(
                    values,
                    indices,
                    candidate,
                    0..values.len(),
                    0,
                )
            },
        ) {
            Ok(result) => Ok(result),
            Err(
                IncrementalDenseReconstructionExecutionError::Algorithm(error)
                | IncrementalDenseReconstructionExecutionError::Executor(error),
            ) => Err(error),
        }
    }
    /// Consumes one indexed block with synchronous executor-owned output shards.
    ///
    /// The callback receives only the consumed measurement prefix and its
    /// indices. It must return every disjoint candidate shard before returning.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn process_block_with_reconstruction<E, F>(
        &self,
        output: &mut [T],
        workspace: &mut IncrementalDenseReconstructionWorkspace<T>,
        measurements: &[T],
        columns: &[u32],
        count: u32,
        info: RowBlockInfo,
        reconstruct: F,
    ) -> Result<RowBlockResult, IncrementalDenseReconstructionExecutionError<E>>
    where
        F: FnOnce(&Self, &[T], &[u32], &mut [T]) -> Result<(), E>,
    {
        use IncrementalDenseReconstructionExecutionError::{Algorithm, Executor};
        if output.len() != self.reconstructor.rows()
            || measurements.len() != self.block_capacity
            || columns.len() != self.block_capacity
            || workspace.candidate.len() != self.reconstructor.rows()
            || workspace.seen.len() != self.reconstructor.columns()
        {
            return Err(Algorithm(
                IncrementalDenseReconstructionError::StorageExtent,
            ));
        }
        let count = count as usize;
        if count > self.block_capacity {
            workspace.abandon();
            return Err(Algorithm(
                IncrementalDenseReconstructionError::CountOutOfRange,
            ));
        }
        if let Err(reason) = workspace.accept(info) {
            workspace.clear_partial();
            return Ok(RowBlockResult::Dropped(reason));
        }
        if info.first_row == 0 {
            workspace.clear_partial();
        }
        for index in 0..count {
            let column = columns[index] as usize;
            let error = if column >= self.reconstructor.columns() {
                Some(IncrementalDenseReconstructionError::ColumnOutOfRange)
            } else if workspace.seen[column] {
                Some(IncrementalDenseReconstructionError::DuplicateColumn)
            } else if !measurements[index].is_finite() {
                Some(IncrementalDenseReconstructionError::NonFiniteInput)
            } else {
                None
            };
            if let Some(error) = error {
                workspace.abandon();
                return Err(Algorithm(error));
            }
            workspace.seen[column] = true;
            workspace.seen_count += 1;
        }
        if count != 0 {
            if let Err(error) = reconstruct(
                self,
                &measurements[..count],
                &columns[..count],
                &mut workspace.candidate,
            ) {
                workspace.abandon();
                return Err(Executor(error));
            }
            if workspace.candidate.iter().any(|value| !value.is_finite()) {
                workspace.abandon();
                return Err(Algorithm(
                    IncrementalDenseReconstructionError::NonFiniteOutput,
                ));
            }
        }
        if !info.terminal {
            return Ok(RowBlockResult::Pending);
        }
        if workspace.seen_count != self.reconstructor.columns() {
            workspace.abandon();
            return Err(Algorithm(
                IncrementalDenseReconstructionError::IncompleteSample,
            ));
        }
        output.copy_from_slice(&workspace.candidate);
        let result = RowBlockResult::Complete {
            sequence: info.sequence,
            discontinuity: workspace.current_discontinuity,
        };
        workspace.sequence = None;
        workspace.last_offset = 0;
        workspace.current_discontinuity = false;
        workspace.pending_discontinuity = false;
        workspace.clear_partial();
        Ok(result)
    }
}

/// Bounded mutable candidate and scientific-column coverage for one owner.
#[derive(Debug)]
pub struct IncrementalDenseReconstructionWorkspace<T> {
    candidate: Box<[T]>,
    seen: Box<[bool]>,
    seen_count: usize,
    sequence: Option<u64>,
    last_offset: usize,
    pending_discontinuity: bool,
    current_discontinuity: bool,
}
impl<T: MatrixElement> IncrementalDenseReconstructionWorkspace<T> {
    /// Returns the active logical sample identity.
    pub const fn sequence(&self) -> Option<u64> {
        self.sequence
    }
    /// Returns the unique measurement columns consumed for the active sample.
    pub const fn seen_count(&self) -> usize {
        self.seen_count
    }
    /// Clear reconstruction history and begin a clean logical sample stream.
    pub fn reset(&mut self) {
        self.abandon();
        self.pending_discontinuity = false;
    }
    fn abandon(&mut self) {
        self.sequence = None;
        self.last_offset = 0;
        self.pending_discontinuity = true;
        self.current_discontinuity = false;
        self.clear_partial();
    }
    fn accept(&mut self, info: RowBlockInfo) -> Result<(), RowBlockDropReason> {
        let reason = if info.corrupted {
            Some(RowBlockDropReason::Corrupted)
        } else if info.first_row == 0 {
            let superseded = self.sequence.is_some();
            if superseded {
                self.abandon();
            }
            self.clear_partial();
            self.sequence = Some(info.sequence);
            self.current_discontinuity =
                self.pending_discontinuity || info.discontinuity || superseded;
            self.pending_discontinuity = false;
            None
        } else if self.sequence.is_none() {
            Some(RowBlockDropReason::MissingStart)
        } else if self.sequence != Some(info.sequence) {
            Some(RowBlockDropReason::Sequence)
        } else if info.discontinuity {
            Some(RowBlockDropReason::MidFrameDiscontinuity)
        } else if info.first_row <= self.last_offset {
            Some(RowBlockDropReason::RowGap)
        } else {
            None
        };
        if let Some(reason) = reason {
            self.abandon();
            return Err(reason);
        }
        self.last_offset = info.first_row;
        Ok(())
    }
    fn clear_partial(&mut self) {
        self.candidate.fill(T::zero());
        self.seen.fill(false);
        self.seen_count = 0;
    }
}

crate::filtergraph_algorithm! {
    /// Sensor-independent Float32 reconstruction from indexed measurement blocks.
    pub IncrementalDenseReconstructorF32 {
        label = "incremental-dense-reconstructor-f32";
        configuration IncrementalDenseReconstructorF32Configuration {
            measurement_count: usize = GraphRebuild;
            measurement_block_capacity: usize = GraphRebuild;
            blocks_per_sample: usize = GraphRebuild;
            reconstructed_count: usize = GraphRebuild;
            initial_reconstructor: Option<Vec<f32>> = Construction;
            measurements_schema: String = GraphRebuild;
            column_indices_schema: String = GraphRebuild;
            measurement_count_schema: String = GraphRebuild;
            reconstructed_schema: String = GraphRebuild;
            reconstructor_schema: String = GraphRebuild;
        }
        plan = IncrementalDenseReconstructionPlan<f32>;
        operation = IncrementalDenseReconstructionF32;
        workspace = IncrementalDenseReconstructionWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            measurement_count = plan.reconstructor().columns();
            measurement_block_capacity = plan.block_capacity();
            reconstructed_count = plan.reconstructor().rows();
            block_rate_multiplier = u32::try_from(plan.blocks_per_sample())
                .map_err(|_| crate::DeclarationPreparationError::Format(crate::FormatError::Rate))?;
        }
        inputs {
            measurements: f32 [measurement_block_capacity] = "measurements", schema = configuration.measurements_schema.as_str(), sample = measurement_sample, rate = [block_rate_multiplier, 1];
            columns: u32 [measurement_block_capacity] = "columns", schema = configuration.column_indices_schema.as_str(), rate = [block_rate_multiplier, 1];
            count: u32 [1] = "count", schema = configuration.measurement_count_schema.as_str(), rate = [block_rate_multiplier, 1];
        }
        outputs {
            reconstructed: f32 [reconstructed_count] = "reconstructed", schema = configuration.reconstructed_schema.as_str(), metadata_from = "measurements", publication = conditional;
        }
        parameters {
            reconstructor: f32 [reconstructed_count, measurement_count] = "reconstructor", schema = configuration.reconstructor_schema.as_str(), replace |plan, values| { plan.with_reconstructor(values) };
        }
        parameter_error = IncrementalDenseReconstructionPreparationError;
        prepare |configuration| {
            let extent = configuration.reconstructed_count.checked_mul(configuration.measurement_count)
                .filter(|&n| n <= u32::MAX as usize / core::mem::size_of::<f32>())
                .ok_or(IncrementalDenseReconstructionPreparationError::ExtentOverflow)?;
            let matrix = configuration.initial_reconstructor.unwrap_or_else(|| vec![0.0; extent]);
            let plan = IncrementalDenseReconstructionPlan::new(
                PreparedGemv::new(configuration.reconstructed_count, configuration.measurement_count, Arc::from(matrix))?,
                configuration.measurement_block_capacity, configuration.blocks_per_sample)?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = IncrementalDenseReconstructionPreparationError;
        process |plan, workspace| {
            let Some(sample)=measurement_sample else {
                workspace.abandon();
                return Err(IncrementalDenseReconstructionProcessError::MissingSampleMetadata);
            };
            let result = plan.process_block(reconstructed, workspace, measurements, columns, count[0], RowBlockInfo {
                sequence: sample.sequence(),
                first_row: usize::try_from(sample.offset()).map_err(|_| IncrementalDenseReconstructionProcessError::OffsetOverflow)?,
                terminal: sample.is_terminal(), discontinuity: sample.is_discontinuous(), corrupted: sample.is_corrupted(),
            })?;
            Ok(match result {
                RowBlockResult::Pending | RowBlockResult::Dropped(_) => AlgorithmProcessResult::deferred(),
                RowBlockResult::Complete { sequence: _, discontinuity } => AlgorithmProcessResult::complete_with_metadata(
                    sample.with_offset(0).with_terminal(true).with_discontinuity(discontinuity).with_corrupted(false)),
            })
        }
        process_error = IncrementalDenseReconstructionProcessError;
        reset |_plan, workspace| { workspace.reset(); }
        plan_adopted |_plan, workspace| { workspace.abandon(); }
    }
}
