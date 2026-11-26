//! Projection of PDM actuator constraint feedback into active-VDM coordinates.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
#[cfg(all(feature = "experimental-compact-projection", target_arch = "x86_64"))]
use core::ops::Range;

use crate::algorithms::linear_algebra::multiply_row_major;
use crate::{AlgorithmPlan, MatrixElement};

use super::pdm_projection::{
    PdmProjection, PdmProjectionError, PdmToVdmProjection, validate_pdm_projections,
    validate_pdm_to_vdm_projections,
};

/// Storage selected for a prepared full-to-active VDM projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VdmTruncationStorage {
    /// A square identity projection retained without matrix coefficients.
    Identity,
    /// One full-VDM input index retained for each active-VDM output.
    RowSelection,
    /// A contiguous row-major matrix used by the Level-2 BLAS backend.
    RowMajor,
}

/// Immutable representation retained for one full-to-active VDM truncation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VdmTruncationArtifacts<'a, T> {
    /// A square identity projection with no retained coefficients.
    Identity,
    /// One full-VDM input column for each active-VDM output coordinate.
    RowSelection(&'a [usize]),
    /// A complete contiguous row-major active-by-full matrix.
    RowMajor(&'a [T]),
}

#[derive(Clone, Debug)]
enum VdmTruncation<T> {
    Identity,
    RowSelection(Arc<[usize]>),
    RowMajor {
        matrix: Arc<[T]>,
        #[cfg(all(feature = "experimental-compact-projection", target_arch = "x86_64"))]
        compact: Option<Arc<CompactTruncation<T>>>,
    },
}

impl<T: PartialEq> PartialEq for VdmTruncation<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Identity, Self::Identity) => true,
            (Self::RowSelection(left), Self::RowSelection(right)) => left == right,
            (Self::RowMajor { matrix: left, .. }, Self::RowMajor { matrix: right, .. }) => {
                left == right
            }
            _ => false,
        }
    }
}

impl<T> VdmTruncation<T>
where
    T: MatrixElement,
{
    fn prepare(active_vdm_size: usize, full_vdm_size: usize, row_major: Arc<[T]>) -> Self {
        let mut selected_columns = Vec::with_capacity(active_vdm_size);
        for row in row_major.chunks_exact(full_vdm_size) {
            let mut selected = None;
            for (column, &coefficient) in row.iter().enumerate() {
                if coefficient == T::zero() {
                    continue;
                }
                if coefficient != T::one() || selected.is_some() {
                    return Self::row_major(active_vdm_size, full_vdm_size, row_major);
                }
                selected = Some(column);
            }
            let Some(column) = selected else {
                return Self::row_major(active_vdm_size, full_vdm_size, row_major);
            };
            selected_columns.push(column);
        }
        if active_vdm_size == full_vdm_size
            && selected_columns
                .iter()
                .enumerate()
                .all(|(row, &column)| row == column)
        {
            Self::Identity
        } else {
            Self::RowSelection(Arc::from(selected_columns))
        }
    }

    fn row_major(rows: usize, columns: usize, matrix: Arc<[T]>) -> Self {
        #[cfg(not(all(feature = "experimental-compact-projection", target_arch = "x86_64")))]
        let _ = (rows, columns);
        Self::RowMajor {
            #[cfg(all(feature = "experimental-compact-projection", target_arch = "x86_64"))]
            compact: CompactTruncation::prepare(rows, columns, &matrix).map(Arc::new),
            matrix,
        }
    }

    const fn storage(&self) -> VdmTruncationStorage {
        match self {
            Self::Identity => VdmTruncationStorage::Identity,
            Self::RowSelection(_) => VdmTruncationStorage::RowSelection,
            Self::RowMajor { .. } => VdmTruncationStorage::RowMajor,
        }
    }

    fn artifacts(&self) -> VdmTruncationArtifacts<'_, T> {
        match self {
            Self::Identity => VdmTruncationArtifacts::Identity,
            Self::RowSelection(columns) => VdmTruncationArtifacts::RowSelection(columns),
            Self::RowMajor { matrix, .. } => VdmTruncationArtifacts::RowMajor(matrix),
        }
    }

    fn multiply(
        &self,
        active_vdm_size: usize,
        full_vdm_size: usize,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) {
        #[cfg(not(all(feature = "experimental-compact-projection", target_arch = "x86_64")))]
        let _ = scratch;
        match self {
            Self::Identity => output.copy_from_slice(input),
            Self::RowSelection(columns) => {
                for (value, &column) in output.iter_mut().zip(columns.iter()) {
                    *value = input[column];
                }
            }
            Self::RowMajor {
                matrix,
                #[cfg(all(feature = "experimental-compact-projection", target_arch = "x86_64"))]
                compact,
            } => {
                #[cfg(all(feature = "experimental-compact-projection", target_arch = "x86_64"))]
                if let Some(compact) = compact
                    && compact.try_multiply(active_vdm_size, input, output, scratch)
                {
                    return;
                }
                multiply_row_major(matrix, active_vdm_size, full_vdm_size, input, output);
            }
        }
    }
}

#[cfg(all(feature = "experimental-compact-projection", target_arch = "x86_64"))]
#[derive(Debug)]
struct CompactTruncation<T> {
    spans: Box<[Range<usize>]>,
    columns: usize,
    matrix: Box<[T]>,
    min_input: f32,
    max_input: f32,
}

#[cfg(all(feature = "experimental-compact-projection", target_arch = "x86_64"))]
impl<T: MatrixElement> CompactTruncation<T> {
    fn prepare(rows: usize, columns: usize, matrix: &[T]) -> Option<Self> {
        // Establish that both layouts use native Float32 GEMV. Avoid keeping a
        // second dense matrix for small projections or a minor width reduction.
        if size_of::<T>() != size_of::<f32>()
            || matrix.len() < 1 << 18
            || !filtergraph_blas::supports_sgemv_dimensions(rows, columns)
        {
            return None;
        }
        let mut active = vec![false; columns];
        let mut max_row_sum = 1.0_f64;
        let mut min_coefficient = f32::INFINITY;
        for row in matrix.chunks_exact(columns) {
            let mut row_sum = 0.0_f64;
            for (active, coefficient) in active.iter_mut().zip(row) {
                // MatrixElement is sealed. The f32 type gate makes this
                // conversion exact and the bit check independent of DAZ.
                let coefficient = coefficient.to_f32().unwrap();
                if coefficient.to_bits() == 0 {
                    continue;
                }
                *active = true;
                let magnitude = coefficient.abs();
                if magnitude.is_subnormal() {
                    return None;
                }
                if magnitude != 0.0 {
                    min_coefficient = min_coefficient.min(magnitude);
                    row_sum += f64::from(magnitude);
                }
            }
            max_row_sum = max_row_sum.max(row_sum);
        }
        let spans: Vec<_> = active
            .chunk_by(|left, right| left == right)
            .scan(0, |start, span| {
                let range = *start..*start + span.len();
                *start = range.end;
                Some((range, span[0]))
            })
            .filter_map(|(range, active)| active.then_some(range))
            .collect();
        let compact_columns = spans.iter().map(Range::len).sum::<usize>();
        if compact_columns == 0
            || columns - compact_columns < columns.div_ceil(8)
            || !filtergraph_blas::supports_sgemv_dimensions(rows, compact_columns)
        {
            return None;
        }

        // A dot product's rounding error is bounded by gamma times the
        // row L1 norm times max |x|, even under cancellation. Use a conservative
        // operation count and leave margin for both reduction orders. The error
        // budget (µm wavefront) is one tenth of the feedback
        // absolute qualification tolerance. It also keeps every partial sum
        // well within the finite Float32 range.
        const MAX_ABSOLUTE_ERROR: f64 = 1.0e-8;
        let rounding = (2.0 * columns as f64 + 8.0) * f64::from(f32::EPSILON) / 2.0;
        let gamma = rounding / (1.0 - rounding);
        let max_input = (MAX_ABSOLUTE_ERROR / (4.0 * gamma * max_row_sum)) as f32;
        // Keep nonzero products in the normal range. The extra factor of two
        // makes the test conservative at the conversion boundary.
        let min_input = (2.0 * f64::from(f32::MIN_POSITIVE) / f64::from(min_coefficient)) as f32;
        if !max_input.is_normal() || min_input > max_input {
            return None;
        }
        let mut compact = Vec::with_capacity(rows * compact_columns);
        for row in matrix.chunks_exact(columns) {
            for span in &spans {
                compact.extend_from_slice(&row[span.clone()]);
            }
        }
        Some(Self {
            spans: spans.into_boxed_slice(),
            columns: compact_columns,
            matrix: compact.into_boxed_slice(),
            min_input: min_input.max(f32::MIN_POSITIVE),
            max_input,
        })
    }

    fn allows(&self, input: &[T], default_mode: bool) -> bool {
        default_mode
            && input.iter().all(|value| {
                let magnitude = value.to_f32().unwrap().abs();
                magnitude <= self.max_input && (magnitude == 0.0 || magnitude >= self.min_input)
            })
    }

    fn try_multiply(&self, rows: usize, input: &[T], output: &mut [T], scratch: &mut [T]) -> bool {
        if !self.allows(input, filtergraph_blas::sgemv_default_float_mode()) {
            return false;
        }
        let mut start = 0;
        for span in &self.spans {
            let end = start + span.len();
            scratch[start..end].copy_from_slice(&input[span.clone()]);
            start = end;
        }
        multiply_row_major(
            &self.matrix,
            rows,
            self.columns,
            &scratch[..self.columns],
            output,
        );
        // Let dense GEMV decide the result of a zero or subnormal reduction
        // for nonzero feedback, including cancellation at the underflow boundary.
        output
            .iter()
            .all(|value| value.to_f32().unwrap().is_normal())
            || input.iter().all(|value| *value == T::zero())
    }
}

/// Invalid PDM-to-VDM feedback input, output, or result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmFeedbackToVdmError {
    /// The concatenated PDM constraint feedback has the wrong extent.
    InputExtent,
    /// The VDM constraint feedback has the wrong extent.
    OutputExtent,
    /// The workspace was prepared for an incompatible full-VDM extent.
    WorkspaceExtent,
    /// A PDM constraint-feedback vector contains a non-finite value.
    NonFiniteInput,
    /// The VDM constraint-feedback calculation produced a non-finite value.
    NonFiniteOutput,
}

impl fmt::Display for PdmFeedbackToVdmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => formatter.write_str("PDM-to-VDM feedback input extent is invalid"),
            Self::OutputExtent => {
                formatter.write_str("PDM-to-VDM feedback output extent is invalid")
            }
            Self::WorkspaceExtent => {
                formatter.write_str("PDM-to-VDM feedback workspace extent is invalid")
            }
            Self::NonFiniteInput => formatter.write_str("PDM constraint feedback must be finite"),
            Self::NonFiniteOutput => {
                formatter.write_str("PDM-to-VDM feedback produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for PdmFeedbackToVdmError {}

/// Prepared PDM-to-full-VDM projections and full-to-active truncation.
#[derive(Clone, Debug, PartialEq)]
pub struct PdmFeedbackToVdmPlan<T> {
    active_vdm_size: usize,
    full_vdm_size: usize,
    physical_actuator_extent: usize,
    full_to_active: VdmTruncation<T>,
    pdms: Arc<[PdmToVdmProjection<T>]>,
}

impl<T> PdmFeedbackToVdmPlan<T>
where
    T: MatrixElement,
{
    /// Prepares every PDM-to-VDM projection and full-to-active truncation.
    pub fn new(
        active_vdm_size: usize,
        full_vdm_size: usize,
        full_to_active: Arc<[T]>,
        pdms: Arc<[PdmProjection<T>]>,
    ) -> Result<Self, PdmProjectionError> {
        validate_pdm_projections(active_vdm_size, full_vdm_size, &pdms)?;
        let pdms = pdms
            .iter()
            .map(PdmProjection::pdm_to_vdm_projection)
            .collect::<Vec<_>>()
            .into();
        Self::from_pdm_to_vdm_projections(active_vdm_size, full_vdm_size, full_to_active, pdms)
    }

    /// Prepare full-to-active truncation and only the feedback direction of
    /// every physical-DM projection.
    pub fn from_pdm_to_vdm_projections(
        active_vdm_size: usize,
        full_vdm_size: usize,
        full_to_active: Arc<[T]>,
        pdms: Arc<[PdmToVdmProjection<T>]>,
    ) -> Result<Self, PdmProjectionError> {
        let physical_actuator_extent =
            validate_pdm_to_vdm_projections(active_vdm_size, full_vdm_size, &pdms)?;
        let active_full = active_vdm_size
            .checked_mul(full_vdm_size)
            .ok_or(PdmProjectionError::ExtentOverflow)?;
        if full_to_active.len() != active_full {
            return Err(PdmProjectionError::ProjectionExtent);
        }
        if full_to_active.iter().any(|value| !value.is_finite()) {
            return Err(PdmProjectionError::NonFiniteConfiguration);
        }
        let full_to_active = VdmTruncation::prepare(active_vdm_size, full_vdm_size, full_to_active);
        Ok(Self {
            active_vdm_size,
            full_vdm_size,
            physical_actuator_extent,
            full_to_active,
            pdms,
        })
    }

    /// Returns the active-VDM output extent.
    pub const fn active_vdm_size(&self) -> usize {
        self.active_vdm_size
    }

    /// Returns the full-VDM intermediate extent.
    pub const fn full_vdm_size(&self) -> usize {
        self.full_vdm_size
    }

    /// Returns the fixed number of PDM projections.
    pub fn pdm_count(&self) -> usize {
        self.pdms.len()
    }

    /// Returns the concatenated physical-actuator input extent.
    pub const fn physical_actuator_extent(&self) -> usize {
        self.physical_actuator_extent
    }

    /// Return the ordered PDM-to-full-VDM projections retained by this plan.
    pub fn pdm_to_vdm_projections(&self) -> &[PdmToVdmProjection<T>] {
        &self.pdms
    }

    /// Returns storage selected for the full-to-active VDM projection.
    pub fn truncation_storage(&self) -> VdmTruncationStorage {
        self.full_to_active.storage()
    }

    /// Returns the complete retained full-to-active VDM truncation.
    pub fn truncation_artifacts(&self) -> VdmTruncationArtifacts<'_, T> {
        self.full_to_active.artifacts()
    }

    /// Prepares an otherwise identical plan with a new full-to-active VDM
    /// truncation.
    pub fn with_full_to_active(
        &self,
        full_to_active: Arc<[T]>,
    ) -> Result<Self, PdmProjectionError> {
        let active_full = self
            .active_vdm_size
            .checked_mul(self.full_vdm_size)
            .ok_or(PdmProjectionError::ExtentOverflow)?;
        if full_to_active.len() != active_full {
            return Err(PdmProjectionError::ProjectionExtent);
        }
        if full_to_active.iter().any(|value| !value.is_finite()) {
            return Err(PdmProjectionError::NonFiniteConfiguration);
        }
        Ok(Self {
            active_vdm_size: self.active_vdm_size,
            full_vdm_size: self.full_vdm_size,
            physical_actuator_extent: self.physical_actuator_extent,
            full_to_active: VdmTruncation::prepare(
                self.active_vdm_size,
                self.full_vdm_size,
                full_to_active,
            ),
            pdms: Arc::clone(&self.pdms),
        })
    }

    /// Prepares an otherwise identical plan with a complete replacement PDM
    /// projection set.
    pub fn with_pdm_to_vdm_projections(
        &self,
        pdms: Arc<[PdmToVdmProjection<T>]>,
    ) -> Result<Self, PdmProjectionError> {
        let physical_actuator_extent =
            validate_pdm_to_vdm_projections(self.active_vdm_size, self.full_vdm_size, &pdms)?;
        if physical_actuator_extent != self.physical_actuator_extent {
            return Err(PdmProjectionError::ProjectionExtent);
        }
        Ok(Self {
            active_vdm_size: self.active_vdm_size,
            full_vdm_size: self.full_vdm_size,
            physical_actuator_extent,
            full_to_active: self.full_to_active.clone(),
            pdms,
        })
    }

    /// Allocates fixed aggregate-feedback scratch.
    pub fn workspace(&self) -> PdmFeedbackToVdmWorkspace<T> {
        PdmFeedbackToVdmWorkspace {
            full_vdm_feedback: vec![T::zero(); self.full_vdm_size].into_boxed_slice(),
            pdm_feedback: vec![T::zero(); self.full_vdm_size].into_boxed_slice(),
        }
    }

    pub(crate) fn workspace_is_compatible(&self, workspace: &PdmFeedbackToVdmWorkspace<T>) -> bool {
        workspace.full_vdm_feedback.len() == self.full_vdm_size
            && workspace.pdm_feedback.len() == self.full_vdm_size
    }
}

impl<T> AlgorithmPlan for PdmFeedbackToVdmPlan<T>
where
    T: MatrixElement + 'static,
{
    type Input<'a>
        = &'a [T]
    where
        Self: 'a;
    type Output<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type Workspace = PdmFeedbackToVdmWorkspace<T>;
    type Error = PdmFeedbackToVdmError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        pdm_constraint_feedback: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if pdm_constraint_feedback.len() != self.physical_actuator_extent {
            return Err(PdmFeedbackToVdmError::InputExtent);
        }
        if output.len() != self.active_vdm_size {
            return Err(PdmFeedbackToVdmError::OutputExtent);
        }
        if !self.workspace_is_compatible(workspace) {
            return Err(PdmFeedbackToVdmError::WorkspaceExtent);
        }
        if pdm_constraint_feedback
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(PdmFeedbackToVdmError::NonFiniteInput);
        }
        workspace.full_vdm_feedback.fill(T::zero());
        let mut offset = 0;
        for pdm in self.pdms.iter() {
            let end = offset + pdm.pdm_size;
            pdm.project_to_vdm(
                &pdm_constraint_feedback[offset..end],
                &mut workspace.pdm_feedback,
            );
            for (sum, value) in workspace
                .full_vdm_feedback
                .iter_mut()
                .zip(workspace.pdm_feedback.iter())
            {
                *sum = *sum + *value;
            }
            offset = end;
        }
        // All per-PDM projections are consumed, so their scratch is available
        // to gather the compact truncation input.
        self.full_to_active.multiply(
            self.active_vdm_size,
            self.full_vdm_size,
            &workspace.full_vdm_feedback,
            output,
            &mut workspace.pdm_feedback,
        );
        if output.iter().any(|value| !value.is_finite()) {
            return Err(PdmFeedbackToVdmError::NonFiniteOutput);
        }
        Ok(())
    }
}

/// Fixed scratch for one PDM-to-VDM feedback owner.
#[derive(Clone, Debug, PartialEq)]
pub struct PdmFeedbackToVdmWorkspace<T> {
    full_vdm_feedback: Box<[T]>,
    pdm_feedback: Box<[T]>,
}

crate::filtergraph_algorithm! {
    /// Portable Float32 physical-DM feedback to active-VDM declaration.
    pub PdmFeedbackToVdmF32 {
        label = "pdm-feedback-to-vdm-f32";
        configuration PdmFeedbackToVdmF32Configuration {
            active_vdm_count: usize = GraphRebuild;
            full_vdm_count: usize = GraphRebuild;
            physical_actuator_count: usize = GraphRebuild;
        }
        plan = PdmFeedbackToVdmPlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            active_vdm_size = plan.active_vdm_size();
            full_vdm_size = plan.full_vdm_size();
            physical_actuator_extent = plan.physical_actuator_extent();
        }
        inputs {
            pdm_constraint_feedback: f32 [physical_actuator_extent] = "pdm-constraint-feedback",
                schema = crate::schemas::PDM_CONSTRAINT_FEEDBACK_V1;
        }
        outputs {
            vdm_constraint_feedback: f32 [active_vdm_size] = "vdm-constraint-feedback",
                schema = crate::schemas::VDM_CONSTRAINT_FEEDBACK_V1,
                metadata_from = "pdm-constraint-feedback";
        }
        parameters {
            full_to_active: f32 [active_vdm_size, full_vdm_size] = "full-to-active",
                schema = crate::schemas::FULL_TO_ACTIVE_VDM_PROJECTION_V1,
                replace |plan, values| {
                    plan.with_full_to_active(Arc::from(values))
                };
            pdm_to_vdm: f32 [full_vdm_size, physical_actuator_extent] = "pdm-to-vdm",
                schema = crate::schemas::PDM_TO_VDM_PROJECTION_V1,
                replace |plan, values| {
                    let projection = PdmToVdmProjection::new(
                        plan.physical_actuator_extent(),
                        Arc::from(values),
                    )?;
                    plan.with_pdm_to_vdm_projections(Arc::from([projection]))
                };
        }
        parameter_error = PdmProjectionError;
        prepare |configuration| {
            if configuration.active_vdm_count == 0
                || configuration.full_vdm_count == 0
                || configuration.physical_actuator_count == 0
            {
                return Err(PdmProjectionError::EmptyDimension);
            }
            let active_full_extent = super::declared_f32_matrix_extent(
                configuration.active_vdm_count,
                configuration.full_vdm_count,
            )
                .ok_or(PdmProjectionError::ExtentOverflow)?;
            let pdm_full_extent = super::declared_f32_matrix_extent(
                configuration.physical_actuator_count,
                configuration.full_vdm_count,
            )
                .ok_or(PdmProjectionError::ExtentOverflow)?;
            let projection = PdmToVdmProjection::new(
                configuration.physical_actuator_count,
                Arc::from(vec![0.0; pdm_full_extent]),
            )?;
            let plan = PdmFeedbackToVdmPlan::from_pdm_to_vdm_projections(
                configuration.active_vdm_count,
                configuration.full_vdm_count,
                Arc::from(vec![0.0; active_full_extent]),
                Arc::from([projection]),
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = PdmProjectionError;
        process |plan, workspace| {
            plan.process(vdm_constraint_feedback, workspace, pdm_constraint_feedback)
        }
        process_error = PdmFeedbackToVdmError;
    }
}

#[cfg(all(
    test,
    feature = "experimental-compact-projection",
    target_arch = "x86_64"
))]
mod tests {
    use super::*;

    #[test]
    fn truncation_equality_uses_the_original_matrix() {
        let (rows, columns) = (256, 1024);
        let mut matrix = vec![0.0_f32; rows * columns];
        for row in matrix.chunks_exact_mut(columns) {
            row[..columns - columns / 8].fill(0.5);
        }
        let compact = VdmTruncation::prepare(rows, columns, Arc::from(matrix.clone()));
        matrix[columns - 1] = -0.0;
        let dense = VdmTruncation::prepare(rows, columns, Arc::from(matrix));
        assert!(matches!(
            compact,
            VdmTruncation::RowMajor {
                compact: Some(_),
                ..
            }
        ));
        assert!(matches!(
            dense,
            VdmTruncation::RowMajor { compact: None, .. }
        ));
        assert_eq!(compact, dense);
    }

    #[test]
    fn dense_truncation_retains_original_columns_and_controls_compact_dispatch() {
        let (rows, columns) = (256, 1024);
        let mut matrix = vec![0.0_f32; rows * columns];
        for row in matrix.chunks_exact_mut(columns) {
            row[0] = 2.0;
            row[17] = -2.0;
            row[18] = 0.5;
            row[200] = -0.0;
            row[911] = 1.0e-20;
        }
        let compact = CompactTruncation::prepare(rows, columns, &matrix).unwrap();
        assert_eq!(compact.spans.as_ref(), &[0..1, 17..19, 200..201, 911..912]);
        assert_eq!(compact.matrix[3].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(compact.matrix[4].to_bits(), matrix[911].to_bits());

        let mut identity = vec![0.0_f32; columns * columns];
        for (index, row) in identity.chunks_exact_mut(columns).enumerate() {
            row[index] = 1.0;
        }
        let pdm = PdmToVdmProjection::new(columns, Arc::from(identity)).unwrap();
        let plan = PdmFeedbackToVdmPlan::from_pdm_to_vdm_projections(
            rows,
            columns,
            Arc::from(matrix.clone()),
            Arc::from([pdm]),
        )
        .unwrap();
        let VdmTruncationArtifacts::RowMajor(retained) = plan.truncation_artifacts() else {
            panic!("the plan must expose the complete projection matrix")
        };
        assert_eq!(retained.len(), rows * columns);
        assert_eq!(retained[200].to_bits(), (-0.0_f32).to_bits());

        let mut workspace = plan.workspace();
        let mut input = vec![0.0_f32; columns];
        input[0] = 1.0e-7;
        input[17] = 1.0e-7;
        input[18] = 2.0e-7;
        input[911] = 3.0e-7;
        let mut output = vec![f32::NAN; rows];
        let mut dense = output.clone();
        let mut gathered = input.clone();
        assert!(compact.try_multiply(rows, &input, &mut output, &mut gathered));
        multiply_row_major(&matrix, rows, columns, &input, &mut dense);
        plan.process(&mut output, &mut workspace, &input).unwrap();
        assert!(
            output
                .iter()
                .zip(&dense)
                .all(|(a, b)| (a - b).abs() <= 1e-12)
        );
        assert!((output[0] - 1.0e-7).abs() <= 1e-12);
        assert_eq!(gathered[..5], [1.0e-7, 1.0e-7, 2.0e-7, 0.0, 3.0e-7]);

        // The dense backend decides near-zero reductions, even on an input
        // that passes the magnitude guard.
        input[18] = 0.0;
        input[911] = 0.0;
        assert!(compact.allows(&input, true));
        assert!(!compact.allows(&input, false));
        assert!(!compact.try_multiply(rows, &input, &mut output, &mut gathered));
        plan.process(&mut output, &mut workspace, &input).unwrap();
        assert!(output.iter().all(|value| value.to_bits() == 0));

        // A tiny coefficient can affect the dense result. It must either be
        // retained above or disable compaction when it is subnormal.
        matrix[511] = f32::from_bits(1);
        let replacement = plan.with_full_to_active(Arc::from(matrix.clone())).unwrap();
        assert!(matches!(
            replacement.full_to_active,
            VdmTruncation::RowMajor { compact: None, .. }
        ));
        input.fill(0.0);
        input[511] = 1.0;
        replacement
            .process(&mut output, &mut workspace, &input)
            .unwrap();
        assert_eq!(output[0].to_bits(), 1);

        input[511] = f32::from_bits(1);
        assert!(!compact.allows(&input, true));
        input[511] = f32::NAN;
        assert_eq!(
            plan.process(&mut output, &mut workspace, &input),
            Err(PdmFeedbackToVdmError::NonFiniteInput)
        );
        assert_eq!(output[0].to_bits(), 1);
        input[511] = 0.0;
        input[0] = f32::MAX;
        assert!(!compact.allows(&input, true));
        assert_eq!(
            plan.process(&mut output, &mut workspace, &input),
            Err(PdmFeedbackToVdmError::NonFiniteOutput)
        );
        assert!(output.iter().all(|value| *value == f32::INFINITY));
    }
}
