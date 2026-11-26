//! Projection of active VDM values into one or more PDM actuator-coordinate spaces.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::mem::size_of;

use crate::algorithms::linear_algebra::multiply_row_major;
use crate::{AlgorithmPlan, MatrixElement};

use super::pdm_projection::{
    PdmProjection, PdmProjectionArtifacts, PdmProjectionError, VdmToPdmProjection,
    validate_pdm_projections, validate_vdm_to_pdm_projections,
};

// Cold metadata preparation; no command values or health are inferred.
pub(super) fn full_correction_roles<T: MatrixElement>(columns: usize, matrix: &[T]) -> Arc<[u8]> {
    let mut controlled = vec![None; columns];
    let mut roles = vec![3; matrix.len() / columns];
    for (row_index, row) in matrix.chunks_exact(columns).enumerate() {
        let mut nonzero = row
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != T::zero());
        let Some((column, value)) = nonzero.next() else {
            roles[row_index] = 0;
            continue;
        };
        if nonzero.next().is_none() && *value == T::one() && controlled[column].is_none() {
            controlled[column] = Some(row_index);
            roles[row_index] = 1;
        }
    }
    if controlled.iter().all(Option::is_some) {
        for role in &mut roles {
            if *role == 3 {
                *role = 2;
            }
        }
    } else {
        for role in &mut roles {
            if *role == 1 {
                *role = 3;
            }
        }
    }
    roles.into()
}

fn physical_correction_roles<T: MatrixElement>(
    full_roles: &[u8],
    pdms: &[VdmToPdmProjection<T>],
) -> Arc<[u8]> {
    let mut roles = Vec::new();
    for pdm in pdms {
        match pdm.artifacts() {
            PdmProjectionArtifacts::Identity => roles.extend_from_slice(full_roles),
            PdmProjectionArtifacts::RowMajor(matrix) => {
                for row in matrix.chunks_exact(full_roles.len()) {
                    let mut nonzero = row
                        .iter()
                        .enumerate()
                        .filter(|(_, value)| **value != T::zero());
                    let role = match nonzero.next() {
                        None => 0,
                        Some((column, value)) if nonzero.next().is_none() && *value == T::one() => {
                            full_roles[column]
                        }
                        _ => 3,
                    };
                    roles.push(role);
                }
            }
        }
    }
    roles.into()
}

/// Storage selected for a prepared active-to-full VDM extrapolation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VdmExtrapolationStorage {
    /// A square identity map retained without matrix coefficients.
    Identity,
    /// Controlled coordinates and a dense extrapolated-coordinate completion.
    ControlledExtrapolation,
    /// A contiguous row-major matrix used by the dense Level-2 BLAS backend.
    RowMajor,
    /// A compressed-sparse-row matrix traversed without per-frame allocation.
    CompressedSparseRow,
}

/// Immutable representation retained for one active-to-full VDM extrapolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VdmExtrapolationArtifacts<'a, T> {
    /// A square identity map with no retained coefficients.
    Identity,
    /// Direct controlled-coordinate placement plus dense extrapolated rows.
    ControlledExtrapolation {
        /// Full-VDM rows receiving controlled coordinates directly.
        controlled_rows: &'a [usize],
        /// Active-VDM input columns in controlled-coordinate order, or `None`
        /// when the active input is already in that order.
        input_columns: Option<&'a [usize]>,
        /// Full-VDM rows calculated from the controlled coordinates.
        extrapolated_rows: &'a [usize],
        /// Row-major extrapolated-row coefficients over controlled coordinates.
        coefficients: &'a [T],
    },
    /// A complete contiguous row-major full-by-active matrix.
    RowMajor(&'a [T]),
    /// A complete compressed-sparse-row full-by-active matrix.
    CompressedSparseRow {
        /// CSR row offsets with `full_vdm_size + 1` entries.
        row_offsets: &'a [usize],
        /// CSR active-VDM column indices.
        column_indices: &'a [usize],
        /// CSR coefficients aligned with `column_indices`.
        coefficients: &'a [T],
    },
}

#[derive(Clone, Debug, PartialEq)]
enum VdmExtrapolation<T> {
    Identity,
    ControlledExtrapolation {
        controlled_rows: Arc<[usize]>,
        input_columns: Option<Arc<[usize]>>,
        extrapolated_rows: Arc<[usize]>,
        coefficients: Arc<[T]>,
    },
    RowMajor(Arc<[T]>),
    CompressedSparseRow {
        row_offsets: Arc<[usize]>,
        column_indices: Arc<[usize]>,
        coefficients: Arc<[T]>,
    },
}

impl<T> VdmExtrapolation<T>
where
    T: MatrixElement,
{
    fn prepare(
        full_vdm_size: usize,
        active_vdm_size: usize,
        row_major: Arc<[T]>,
    ) -> Result<Self, PdmProjectionError> {
        let extent = full_vdm_size
            .checked_mul(active_vdm_size)
            .ok_or(PdmProjectionError::ExtentOverflow)?;
        if row_major.len() != extent {
            return Err(PdmProjectionError::ProjectionExtent);
        }
        if row_major.iter().any(|value| !value.is_finite()) {
            return Err(PdmProjectionError::NonFiniteConfiguration);
        }
        if full_vdm_size == active_vdm_size
            && row_major
                .chunks_exact(active_vdm_size)
                .enumerate()
                .all(|(row_index, row)| {
                    row.iter().enumerate().all(|(column_index, &coefficient)| {
                        coefficient
                            == if row_index == column_index {
                                T::one()
                            } else {
                                T::zero()
                            }
                    })
                })
        {
            return Ok(Self::Identity);
        }

        let nonzero_count = row_major
            .iter()
            .filter(|&&coefficient| coefficient != T::zero())
            .count();
        let dense_bytes = extent.checked_mul(size_of::<T>());
        let sparse_bytes = full_vdm_size
            .checked_add(1)
            .and_then(|offset_count| offset_count.checked_mul(size_of::<usize>()))
            .and_then(|offset_bytes| {
                nonzero_count
                    .checked_mul(size_of::<usize>() + size_of::<T>())
                    .and_then(|entry_bytes| offset_bytes.checked_add(entry_bytes))
            });
        if let Some(partitioned) = Self::prepare_controlled_extrapolation(
            full_vdm_size,
            active_vdm_size,
            &row_major,
            dense_bytes,
            sparse_bytes,
        ) {
            return Ok(partitioned);
        }
        if !matches!((dense_bytes, sparse_bytes), (Some(dense), Some(sparse)) if sparse < dense) {
            return Ok(Self::RowMajor(row_major));
        }

        let mut row_offsets = Vec::with_capacity(full_vdm_size + 1);
        let mut column_indices = Vec::with_capacity(nonzero_count);
        let mut coefficients = Vec::with_capacity(nonzero_count);
        row_offsets.push(0);
        for row in row_major.chunks_exact(active_vdm_size) {
            for (column, &coefficient) in row.iter().enumerate() {
                if coefficient != T::zero() {
                    column_indices.push(column);
                    coefficients.push(coefficient);
                }
            }
            row_offsets.push(coefficients.len());
        }
        Ok(Self::CompressedSparseRow {
            row_offsets: Arc::from(row_offsets),
            column_indices: Arc::from(column_indices),
            coefficients: Arc::from(coefficients),
        })
    }

    fn prepare_controlled_extrapolation(
        full_vdm_size: usize,
        active_vdm_size: usize,
        row_major: &[T],
        dense_bytes: Option<usize>,
        sparse_bytes: Option<usize>,
    ) -> Option<Self> {
        let mut controlled_rows = Vec::new();
        let mut controlled_columns = Vec::new();
        let mut controlled_row = vec![false; full_vdm_size];
        let mut column_position = vec![None; active_vdm_size];

        for (row_index, row) in row_major.chunks_exact(active_vdm_size).enumerate() {
            let mut standard_basis_column = None;
            for (column_index, &coefficient) in row.iter().enumerate() {
                if coefficient == T::zero() {
                    continue;
                }
                if coefficient != T::one() || standard_basis_column.is_some() {
                    standard_basis_column = None;
                    break;
                }
                standard_basis_column = Some(column_index);
            }
            let Some(column_index) = standard_basis_column else {
                continue;
            };
            if column_position[column_index].is_some() {
                continue;
            }
            column_position[column_index] = Some(controlled_columns.len());
            controlled_rows.push(row_index);
            controlled_columns.push(column_index);
            controlled_row[row_index] = true;
        }
        if controlled_rows.is_empty() {
            return None;
        }

        let mut extrapolated_rows = Vec::new();
        for (row_index, row) in row_major.chunks_exact(active_vdm_size).enumerate() {
            if controlled_row[row_index] {
                continue;
            }
            let mut has_nonzero = false;
            for (column_index, &coefficient) in row.iter().enumerate() {
                if coefficient != T::zero() {
                    has_nonzero = true;
                    let _ = column_position[column_index]?;
                }
            }
            if has_nonzero {
                extrapolated_rows.push(row_index);
            }
        }

        let input_is_packed = controlled_columns.len() == active_vdm_size
            && controlled_columns
                .iter()
                .enumerate()
                .all(|(position, &column)| position == column);
        let coefficient_count = extrapolated_rows
            .len()
            .checked_mul(controlled_columns.len())?;
        let index_count = controlled_rows
            .len()
            .checked_add(extrapolated_rows.len())?
            .checked_add(if input_is_packed {
                0
            } else {
                controlled_columns.len()
            })?;
        let partitioned_bytes = coefficient_count
            .checked_mul(size_of::<T>())?
            .checked_add(index_count.checked_mul(size_of::<usize>())?)?;
        if !matches!(
            (dense_bytes, sparse_bytes),
            (Some(dense), Some(sparse)) if partitioned_bytes < dense && partitioned_bytes < sparse
        ) {
            return None;
        }

        let mut coefficients = Vec::with_capacity(coefficient_count);
        for &row_index in &extrapolated_rows {
            let row_start = row_index * active_vdm_size;
            let row = &row_major[row_start..row_start + active_vdm_size];
            coefficients.extend(
                controlled_columns
                    .iter()
                    .map(|&column_index| row[column_index]),
            );
        }

        Some(Self::ControlledExtrapolation {
            controlled_rows: Arc::from(controlled_rows),
            input_columns: (!input_is_packed).then(|| Arc::from(controlled_columns)),
            extrapolated_rows: Arc::from(extrapolated_rows),
            coefficients: Arc::from(coefficients),
        })
    }

    fn storage(&self) -> VdmExtrapolationStorage {
        match self {
            Self::Identity => VdmExtrapolationStorage::Identity,
            Self::ControlledExtrapolation { .. } => {
                VdmExtrapolationStorage::ControlledExtrapolation
            }
            Self::RowMajor(_) => VdmExtrapolationStorage::RowMajor,
            Self::CompressedSparseRow { .. } => VdmExtrapolationStorage::CompressedSparseRow,
        }
    }

    fn coefficient_count(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::ControlledExtrapolation { coefficients, .. } => coefficients.len(),
            Self::RowMajor(matrix) => matrix.len(),
            Self::CompressedSparseRow { coefficients, .. } => coefficients.len(),
        }
    }

    fn artifacts(&self) -> VdmExtrapolationArtifacts<'_, T> {
        match self {
            Self::Identity => VdmExtrapolationArtifacts::Identity,
            Self::ControlledExtrapolation {
                controlled_rows,
                input_columns,
                extrapolated_rows,
                coefficients,
            } => VdmExtrapolationArtifacts::ControlledExtrapolation {
                controlled_rows,
                input_columns: input_columns.as_deref(),
                extrapolated_rows,
                coefficients,
            },
            Self::RowMajor(matrix) => VdmExtrapolationArtifacts::RowMajor(matrix),
            Self::CompressedSparseRow {
                row_offsets,
                column_indices,
                coefficients,
            } => VdmExtrapolationArtifacts::CompressedSparseRow {
                row_offsets,
                column_indices,
                coefficients,
            },
        }
    }

    fn multiply(
        &self,
        full_vdm_size: usize,
        active_vdm_size: usize,
        input: &[T],
        output: &mut [T],
        controlled_vdm: &mut [T],
        extrapolated_vdm: &mut [T],
    ) {
        match self {
            Self::Identity => output.copy_from_slice(input),
            Self::ControlledExtrapolation {
                controlled_rows,
                input_columns,
                extrapolated_rows,
                coefficients,
            } => {
                output.fill(T::zero());
                let controlled_input = if let Some(input_columns) = input_columns {
                    let controlled_vdm = &mut controlled_vdm[..input_columns.len()];
                    for (value, &column) in controlled_vdm.iter_mut().zip(input_columns.iter()) {
                        *value = input[column];
                    }
                    &*controlled_vdm
                } else {
                    input
                };
                for (&row, &value) in controlled_rows.iter().zip(controlled_input.iter()) {
                    output[row] = value;
                }
                if !extrapolated_rows.is_empty() {
                    let extrapolated_vdm = &mut extrapolated_vdm[..extrapolated_rows.len()];
                    multiply_row_major(
                        coefficients,
                        extrapolated_rows.len(),
                        controlled_rows.len(),
                        controlled_input,
                        extrapolated_vdm,
                    );
                    for (&row, &value) in extrapolated_rows.iter().zip(extrapolated_vdm.iter()) {
                        output[row] = value;
                    }
                }
            }
            Self::RowMajor(matrix) => {
                multiply_row_major(matrix, full_vdm_size, active_vdm_size, input, output)
            }
            Self::CompressedSparseRow {
                row_offsets,
                column_indices,
                coefficients,
            } => {
                for (row, bounds) in row_offsets.windows(2).enumerate() {
                    let mut sum = T::zero();
                    for entry in bounds[0]..bounds[1] {
                        sum = sum + coefficients[entry] * input[column_indices[entry]];
                    }
                    output[row] = sum;
                }
            }
        }
    }
}

/// Invalid VDM-to-PDM input, output, or result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VdmToPdmError {
    /// An input vector has the wrong extent.
    InputExtent,
    /// An output vector has the wrong extent.
    OutputExtent,
    /// The workspace was prepared for an incompatible full-VDM extent.
    WorkspaceExtent,
    /// An input vector contains a non-finite value.
    NonFiniteInput,
    /// The calculation produced a non-finite value.
    NonFiniteOutput,
}

impl fmt::Display for VdmToPdmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => formatter.write_str("VDM-to-PDM input extent is invalid"),
            Self::OutputExtent => formatter.write_str("VDM-to-PDM output extent is invalid"),
            Self::WorkspaceExtent => formatter.write_str("VDM-to-PDM workspace extent is invalid"),
            Self::NonFiniteInput => formatter.write_str("VDM command must be finite"),
            Self::NonFiniteOutput => {
                formatter.write_str("VDM-to-PDM projection produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for VdmToPdmError {}

/// Prepared active-to-full VDM mapping and fixed VDM-to-PDM projections.
#[derive(Clone, Debug, PartialEq)]
pub struct VdmToPdmPlan<T> {
    active_vdm_size: usize,
    full_vdm_size: usize,
    physical_actuator_extent: usize,
    active_to_full: VdmExtrapolation<T>,
    pdms: Arc<[VdmToPdmProjection<T>]>,
    full_correction_roles: Arc<[u8]>,
    correction_roles: Arc<[u8]>,
}

impl<T> VdmToPdmPlan<T>
where
    T: MatrixElement,
{
    /// Prepares active-to-full extrapolation and every VDM-to-PDM projection.
    ///
    /// Identity extrapolation is retained without coefficients. Other
    /// extrapolations retain controlled rows and dense extrapolated-coordinate
    /// completion when that exact factorization is smallest. Other sparse
    /// matrices use compressed sparse rows when smaller than the supplied
    /// dense row-major matrix; otherwise the dense matrix remains on the
    /// Level-2 BLAS path.
    pub fn new(
        active_vdm_size: usize,
        full_vdm_size: usize,
        active_to_full: Arc<[T]>,
        pdms: Arc<[PdmProjection<T>]>,
    ) -> Result<Self, PdmProjectionError> {
        validate_pdm_projections(active_vdm_size, full_vdm_size, &pdms)?;
        let pdms = pdms
            .iter()
            .map(PdmProjection::vdm_to_pdm_projection)
            .collect::<Vec<_>>()
            .into();
        Self::from_vdm_to_pdm_projections(active_vdm_size, full_vdm_size, active_to_full, pdms)
    }

    /// Prepare active-to-full extrapolation and only the command direction of
    /// every physical-DM projection.
    pub fn from_vdm_to_pdm_projections(
        active_vdm_size: usize,
        full_vdm_size: usize,
        active_to_full: Arc<[T]>,
        pdms: Arc<[VdmToPdmProjection<T>]>,
    ) -> Result<Self, PdmProjectionError> {
        let physical_actuator_extent =
            validate_vdm_to_pdm_projections(active_vdm_size, full_vdm_size, &pdms)?;
        let active_to_full_execution =
            VdmExtrapolation::prepare(full_vdm_size, active_vdm_size, Arc::clone(&active_to_full))?;
        let full_correction_roles = full_correction_roles(active_vdm_size, &active_to_full);
        let correction_roles = physical_correction_roles(&full_correction_roles, &pdms);
        let active_to_full = active_to_full_execution;
        Ok(Self {
            active_vdm_size,
            full_vdm_size,
            physical_actuator_extent,
            active_to_full,
            pdms,
            full_correction_roles,
            correction_roles,
        })
    }

    /// Physical correction provenance: 0 none, 1 controlled, 2 extrapolated, 3 unknown.
    pub fn correction_roles(&self) -> &[u8] {
        &self.correction_roles
    }

    /// Returns the fixed number of PDM projections.
    pub fn pdm_count(&self) -> usize {
        self.pdms.len()
    }

    /// Returns one PDM's actuator extent.
    pub fn pdm_size(&self, index: usize) -> Option<usize> {
        self.pdms.get(index).map(VdmToPdmProjection::pdm_size)
    }

    /// Returns the sum of all PDM actuator extents.
    pub fn physical_actuator_extent(&self) -> usize {
        self.physical_actuator_extent
    }

    /// Returns the active-VDM input extent.
    pub const fn active_vdm_size(&self) -> usize {
        self.active_vdm_size
    }

    /// Returns the full-VDM intermediate extent.
    pub const fn full_vdm_size(&self) -> usize {
        self.full_vdm_size
    }

    /// Return the ordered full-VDM-to-PDM projections retained by this plan.
    pub fn vdm_to_pdm_projections(&self) -> &[VdmToPdmProjection<T>] {
        &self.pdms
    }

    /// Returns the prepared extrapolation representation used by `process`.
    pub fn extrapolation_storage(&self) -> VdmExtrapolationStorage {
        self.active_to_full.storage()
    }

    /// Returns the number of explicitly retained extrapolation coefficients.
    pub fn extrapolation_coefficient_count(&self) -> usize {
        self.active_to_full.coefficient_count()
    }

    /// Returns the complete retained active-to-full VDM extrapolation.
    pub fn extrapolation_artifacts(&self) -> VdmExtrapolationArtifacts<'_, T> {
        self.active_to_full.artifacts()
    }

    /// Prepares an otherwise identical plan with a new active-to-full VDM
    /// extrapolation.
    pub fn with_active_to_full(
        &self,
        active_to_full: Arc<[T]>,
    ) -> Result<Self, PdmProjectionError> {
        let execution = VdmExtrapolation::prepare(
            self.full_vdm_size,
            self.active_vdm_size,
            Arc::clone(&active_to_full),
        )?;
        let full_correction_roles = full_correction_roles(self.active_vdm_size, &active_to_full);
        let correction_roles = physical_correction_roles(&full_correction_roles, &self.pdms);
        let active_to_full = execution;
        Ok(Self {
            active_vdm_size: self.active_vdm_size,
            full_vdm_size: self.full_vdm_size,
            physical_actuator_extent: self.physical_actuator_extent,
            active_to_full,
            pdms: Arc::clone(&self.pdms),
            full_correction_roles,
            correction_roles,
        })
    }

    /// Prepares an otherwise identical plan with a complete replacement PDM
    /// projection set.
    pub fn with_vdm_to_pdm_projections(
        &self,
        pdms: Arc<[VdmToPdmProjection<T>]>,
    ) -> Result<Self, PdmProjectionError> {
        let physical_actuator_extent =
            validate_vdm_to_pdm_projections(self.active_vdm_size, self.full_vdm_size, &pdms)?;
        if physical_actuator_extent != self.physical_actuator_extent {
            return Err(PdmProjectionError::ProjectionExtent);
        }
        Ok(Self {
            active_vdm_size: self.active_vdm_size,
            full_vdm_size: self.full_vdm_size,
            physical_actuator_extent,
            active_to_full: self.active_to_full.clone(),
            correction_roles: physical_correction_roles(&self.full_correction_roles, &pdms),
            full_correction_roles: Arc::clone(&self.full_correction_roles),
            pdms,
        })
    }

    /// Allocates fixed VDM-to-PDM scratch.
    pub fn workspace(&self) -> VdmToPdmWorkspace<T> {
        VdmToPdmWorkspace {
            full_vdm: vec![T::zero(); self.full_vdm_size].into_boxed_slice(),
            controlled_vdm: vec![T::zero(); self.active_vdm_size].into_boxed_slice(),
            extrapolated_vdm: vec![T::zero(); self.full_vdm_size].into_boxed_slice(),
        }
    }

    pub(crate) fn workspace_is_compatible(&self, workspace: &VdmToPdmWorkspace<T>) -> bool {
        workspace.full_vdm.len() == self.full_vdm_size
            && workspace.controlled_vdm.len() == self.active_vdm_size
            && workspace.extrapolated_vdm.len() == self.full_vdm_size
    }
}

impl<T> AlgorithmPlan for VdmToPdmPlan<T>
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
    type Workspace = VdmToPdmWorkspace<T>;
    type Error = VdmToPdmError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        vdm_command: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if vdm_command.len() != self.active_vdm_size {
            return Err(VdmToPdmError::InputExtent);
        }
        if output.len() != self.physical_actuator_extent {
            return Err(VdmToPdmError::OutputExtent);
        }
        if !self.workspace_is_compatible(workspace) {
            return Err(VdmToPdmError::WorkspaceExtent);
        }
        if vdm_command.iter().any(|value| !value.is_finite()) {
            return Err(VdmToPdmError::NonFiniteInput);
        }
        self.active_to_full.multiply(
            self.full_vdm_size,
            self.active_vdm_size,
            vdm_command,
            &mut workspace.full_vdm,
            &mut workspace.controlled_vdm,
            &mut workspace.extrapolated_vdm,
        );
        let mut offset = 0;
        for pdm in self.pdms.iter() {
            let end = offset + pdm.pdm_size;
            pdm.project_to_pdm(&workspace.full_vdm, &mut output[offset..end]);
            offset = end;
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(VdmToPdmError::NonFiniteOutput);
        }
        Ok(())
    }
}

/// Fixed scratch for one VDM-to-PDM operation owner.
#[derive(Clone, Debug, PartialEq)]
pub struct VdmToPdmWorkspace<T> {
    full_vdm: Box<[T]>,
    controlled_vdm: Box<[T]>,
    extrapolated_vdm: Box<[T]>,
}

crate::filtergraph_algorithm! {
    /// Portable Float32 active-VDM to physical-DM projection declaration.
    pub VdmToPdmF32 {
        label = "vdm-to-pdm-f32";
        configuration VdmToPdmF32Configuration {
            active_vdm_count: usize = GraphRebuild;
            full_vdm_count: usize = GraphRebuild;
            physical_actuator_count: usize = GraphRebuild;
        }
        plan = VdmToPdmPlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            active_vdm_size = plan.active_vdm_size();
            full_vdm_size = plan.full_vdm_size();
            physical_actuator_extent = plan.physical_actuator_extent();
        }
        inputs {
            vdm_command: f32 [active_vdm_size] = "vdm-command",
                schema = crate::schemas::VDM_COMMAND_V1,
                optional_sample_metadata = ["chopped_image_kind", "trigger_state",
                    "acquisition_identity", "exposure_duration_nanoseconds"];
        }
        outputs {
            requested_pdm_command: f32 [physical_actuator_extent] = "requested-pdm-command",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1,
                metadata_from = "vdm-command";
            correction_role: u8 [physical_actuator_extent] = "correction-role",
                schema = crate::schemas::PDM_CORRECTION_ROLE_V1,
                metadata_from = "vdm-command";
        }
        parameters {
            active_to_full: f32 [full_vdm_size, active_vdm_size] = "active-to-full",
                schema = crate::schemas::ACTIVE_TO_FULL_VDM_PROJECTION_V1,
                replace |plan, values| {
                    plan.with_active_to_full(Arc::from(values))
                };
            vdm_to_pdm: f32 [physical_actuator_extent, full_vdm_size] = "vdm-to-pdm",
                schema = crate::schemas::VDM_TO_PDM_PROJECTION_V1,
                replace |plan, values| {
                    let projection = VdmToPdmProjection::new(
                        plan.physical_actuator_extent(),
                        Arc::from(values),
                    )?;
                    plan.with_vdm_to_pdm_projections(Arc::from([projection]))
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
                configuration.full_vdm_count,
                configuration.active_vdm_count,
            )
                .ok_or(PdmProjectionError::ExtentOverflow)?;
            let pdm_full_extent = super::declared_f32_matrix_extent(
                configuration.physical_actuator_count,
                configuration.full_vdm_count,
            )
                .ok_or(PdmProjectionError::ExtentOverflow)?;
            let projection = VdmToPdmProjection::new(
                configuration.physical_actuator_count,
                Arc::from(vec![0.0; pdm_full_extent]),
            )?;
            let plan = VdmToPdmPlan::from_vdm_to_pdm_projections(
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
            plan.process(requested_pdm_command, workspace, vdm_command)?;
            correction_role.copy_from_slice(plan.correction_roles());
            Ok(())
        }
        process_error = VdmToPdmError;
    }
}
