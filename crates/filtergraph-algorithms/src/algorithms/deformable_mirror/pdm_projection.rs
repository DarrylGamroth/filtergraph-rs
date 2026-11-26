//! Coherent forward-command and reverse-feedback projections for physical DMs.

use alloc::sync::Arc;
use core::fmt;

use crate::MatrixElement;
use crate::algorithms::linear_algebra::multiply_row_major;

/// Storage selected for one prepared VDM/PDM projection direction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmProjectionStorage {
    /// A square identity projection retained without matrix coefficients.
    Identity,
    /// A contiguous row-major matrix used by the Level-2 BLAS backend.
    RowMajor,
}

/// Immutable coefficients retained for one prepared VDM/PDM projection direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PdmProjectionArtifacts<'a, T> {
    /// A square identity projection with no retained coefficients.
    Identity,
    /// A contiguous row-major projection matrix.
    RowMajor(&'a [T]),
}

#[derive(Clone, Debug, PartialEq)]
enum Projection<T> {
    Identity,
    RowMajor(Arc<[T]>),
}

impl<T> Projection<T>
where
    T: MatrixElement,
{
    fn prepare(rows: usize, columns: usize, matrix: Arc<[T]>) -> Self {
        if rows == columns
            && matrix
                .chunks_exact(columns)
                .enumerate()
                .all(|(row, values)| {
                    values.iter().enumerate().all(|(column, &coefficient)| {
                        coefficient == if row == column { T::one() } else { T::zero() }
                    })
                })
        {
            Self::Identity
        } else {
            Self::RowMajor(matrix)
        }
    }

    const fn storage(&self) -> PdmProjectionStorage {
        match self {
            Self::Identity => PdmProjectionStorage::Identity,
            Self::RowMajor(_) => PdmProjectionStorage::RowMajor,
        }
    }

    fn artifacts(&self) -> PdmProjectionArtifacts<'_, T> {
        match self {
            Self::Identity => PdmProjectionArtifacts::Identity,
            Self::RowMajor(matrix) => PdmProjectionArtifacts::RowMajor(matrix),
        }
    }

    fn multiply(&self, rows: usize, columns: usize, input: &[T], output: &mut [T]) {
        match self {
            Self::Identity => output.copy_from_slice(input),
            Self::RowMajor(matrix) => {
                multiply_row_major(matrix, rows, columns, input, output);
            }
        }
    }
}

/// Invalid prepared VDM/PDM projection set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmProjectionError {
    /// Active VDM, full VDM, and PDM dimensions must be nonzero.
    EmptyDimension,
    /// At least one PDM projection is required.
    EmptyPdmSet,
    /// A prepared projection has the wrong extent.
    ProjectionExtent,
    /// A projection contains a non-finite value.
    NonFiniteConfiguration,
    /// A prepared extent overflowed addressable storage.
    ExtentOverflow,
}

impl fmt::Display for PdmProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension => formatter.write_str("VDM and PDM dimensions must be nonempty"),
            Self::EmptyPdmSet => {
                formatter.write_str("VDM-to-PDM projection requires at least one PDM")
            }
            Self::ProjectionExtent => formatter.write_str("VDM/PDM projection extent is invalid"),
            Self::NonFiniteConfiguration => {
                formatter.write_str("VDM/PDM projections must be finite")
            }
            Self::ExtentOverflow => formatter.write_str("VDM/PDM projection extent overflow"),
        }
    }
}

impl core::error::Error for PdmProjectionError {}

/// Coherent forward-command and reverse-feedback projections for one PDM.
#[derive(Clone, Debug, PartialEq)]
pub struct PdmProjection<T> {
    pub(super) pdm_size: usize,
    full_vdm_size: usize,
    vdm_to_pdm: Projection<T>,
    pdm_to_vdm: Projection<T>,
}

/// Prepared full-VDM command projection into one PDM actuator order.
#[derive(Clone, Debug, PartialEq)]
pub struct VdmToPdmProjection<T> {
    pub(super) pdm_size: usize,
    full_vdm_size: usize,
    vdm_to_pdm: Projection<T>,
}

/// Prepared PDM constraint-feedback projection into full-VDM coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct PdmToVdmProjection<T> {
    pub(super) pdm_size: usize,
    full_vdm_size: usize,
    pdm_to_vdm: Projection<T>,
}

fn validate_direction<T>(pdm_size: usize, projection: &[T]) -> Result<usize, PdmProjectionError>
where
    T: MatrixElement,
{
    if pdm_size == 0 || projection.is_empty() {
        return Err(PdmProjectionError::EmptyDimension);
    }
    if !projection.len().is_multiple_of(pdm_size) {
        return Err(PdmProjectionError::ProjectionExtent);
    }
    if projection.iter().any(|value| !value.is_finite()) {
        return Err(PdmProjectionError::NonFiniteConfiguration);
    }
    Ok(projection.len() / pdm_size)
}

impl<T> VdmToPdmProjection<T>
where
    T: MatrixElement,
{
    /// Prepare one row-major PDM-by-full-VDM command projection.
    pub fn new(pdm_size: usize, vdm_to_pdm: Arc<[T]>) -> Result<Self, PdmProjectionError> {
        let full_vdm_size = validate_direction(pdm_size, &vdm_to_pdm)?;
        Ok(Self {
            pdm_size,
            full_vdm_size,
            vdm_to_pdm: Projection::prepare(pdm_size, full_vdm_size, vdm_to_pdm),
        })
    }

    /// Return the physical-actuator extent.
    pub const fn pdm_size(&self) -> usize {
        self.pdm_size
    }

    /// Return the full-VDM input extent.
    pub const fn full_vdm_size(&self) -> usize {
        self.full_vdm_size
    }

    /// Return storage selected for the command projection.
    pub const fn storage(&self) -> PdmProjectionStorage {
        self.vdm_to_pdm.storage()
    }

    /// Return the retained command-projection artifacts.
    pub fn artifacts(&self) -> PdmProjectionArtifacts<'_, T> {
        self.vdm_to_pdm.artifacts()
    }

    pub(super) fn project_to_pdm(&self, input: &[T], output: &mut [T]) {
        self.vdm_to_pdm
            .multiply(self.pdm_size, self.full_vdm_size, input, output);
    }
}

impl<T> PdmToVdmProjection<T>
where
    T: MatrixElement,
{
    /// Prepare one row-major full-VDM-by-PDM feedback projection.
    pub fn new(pdm_size: usize, pdm_to_vdm: Arc<[T]>) -> Result<Self, PdmProjectionError> {
        let full_vdm_size = validate_direction(pdm_size, &pdm_to_vdm)?;
        Ok(Self {
            pdm_size,
            full_vdm_size,
            pdm_to_vdm: Projection::prepare(full_vdm_size, pdm_size, pdm_to_vdm),
        })
    }

    /// Return the physical-actuator extent.
    pub const fn pdm_size(&self) -> usize {
        self.pdm_size
    }

    /// Return the full-VDM output extent.
    pub const fn full_vdm_size(&self) -> usize {
        self.full_vdm_size
    }

    /// Return storage selected for the feedback projection.
    pub const fn storage(&self) -> PdmProjectionStorage {
        self.pdm_to_vdm.storage()
    }

    /// Return the retained feedback-projection artifacts.
    pub fn artifacts(&self) -> PdmProjectionArtifacts<'_, T> {
        self.pdm_to_vdm.artifacts()
    }

    pub(super) fn project_to_vdm(&self, input: &[T], output: &mut [T]) {
        self.pdm_to_vdm
            .multiply(self.full_vdm_size, self.pdm_size, input, output);
    }
}

impl<T> PdmProjection<T>
where
    T: MatrixElement,
{
    /// Prepares one PDM's row-major forward and reverse projections.
    ///
    /// The containing forward and reverse plans validate their extents against
    /// the full-VDM dimension.
    pub fn new(
        pdm_size: usize,
        vdm_to_pdm: Arc<[T]>,
        pdm_to_vdm: Arc<[T]>,
    ) -> Result<Self, PdmProjectionError> {
        if pdm_size == 0 {
            return Err(PdmProjectionError::EmptyDimension);
        }
        if vdm_to_pdm.len() != pdm_to_vdm.len() || !vdm_to_pdm.len().is_multiple_of(pdm_size) {
            return Err(PdmProjectionError::ProjectionExtent);
        }
        if vdm_to_pdm
            .iter()
            .chain(pdm_to_vdm.iter())
            .any(|value| !value.is_finite())
        {
            return Err(PdmProjectionError::NonFiniteConfiguration);
        }
        let full_vdm_size = vdm_to_pdm.len() / pdm_size;
        Ok(Self {
            pdm_size,
            full_vdm_size,
            vdm_to_pdm: Projection::prepare(pdm_size, full_vdm_size, vdm_to_pdm),
            pdm_to_vdm: Projection::prepare(full_vdm_size, pdm_size, pdm_to_vdm),
        })
    }

    /// Returns this projection's physical-actuator extent.
    pub fn pdm_size(&self) -> usize {
        self.pdm_size
    }

    /// Returns the common full-VDM extent used by both directions.
    pub fn full_vdm_size(&self) -> usize {
        self.full_vdm_size
    }

    /// Returns storage selected for the forward full-VDM-to-PDM projection.
    pub fn vdm_to_pdm_storage(&self) -> PdmProjectionStorage {
        self.vdm_to_pdm.storage()
    }

    /// Returns storage selected for the reverse PDM-to-full-VDM projection.
    pub fn pdm_to_vdm_storage(&self) -> PdmProjectionStorage {
        self.pdm_to_vdm.storage()
    }

    /// Returns the retained full-VDM-to-PDM projection artifacts.
    pub fn vdm_to_pdm_artifacts(&self) -> PdmProjectionArtifacts<'_, T> {
        self.vdm_to_pdm.artifacts()
    }

    /// Returns the independently retained PDM-to-full-VDM projection artifacts.
    pub fn pdm_to_vdm_artifacts(&self) -> PdmProjectionArtifacts<'_, T> {
        self.pdm_to_vdm.artifacts()
    }

    /// Select the forward command direction without copying coefficients.
    pub fn vdm_to_pdm_projection(&self) -> VdmToPdmProjection<T> {
        VdmToPdmProjection {
            pdm_size: self.pdm_size,
            full_vdm_size: self.full_vdm_size,
            vdm_to_pdm: self.vdm_to_pdm.clone(),
        }
    }

    /// Select the reverse feedback direction without copying coefficients.
    pub fn pdm_to_vdm_projection(&self) -> PdmToVdmProjection<T> {
        PdmToVdmProjection {
            pdm_size: self.pdm_size,
            full_vdm_size: self.full_vdm_size,
            pdm_to_vdm: self.pdm_to_vdm.clone(),
        }
    }
}

pub(super) fn validate_vdm_to_pdm_projections<T>(
    active_vdm_size: usize,
    full_vdm_size: usize,
    pdms: &[VdmToPdmProjection<T>],
) -> Result<usize, PdmProjectionError>
where
    T: MatrixElement,
{
    validate_directional_pdm_projections(
        active_vdm_size,
        full_vdm_size,
        pdms.iter().map(|pdm| (pdm.pdm_size, pdm.full_vdm_size)),
    )
}

pub(super) fn validate_pdm_to_vdm_projections<T>(
    active_vdm_size: usize,
    full_vdm_size: usize,
    pdms: &[PdmToVdmProjection<T>],
) -> Result<usize, PdmProjectionError>
where
    T: MatrixElement,
{
    validate_directional_pdm_projections(
        active_vdm_size,
        full_vdm_size,
        pdms.iter().map(|pdm| (pdm.pdm_size, pdm.full_vdm_size)),
    )
}

fn validate_directional_pdm_projections(
    active_vdm_size: usize,
    full_vdm_size: usize,
    pdms: impl ExactSizeIterator<Item = (usize, usize)>,
) -> Result<usize, PdmProjectionError> {
    if active_vdm_size == 0 || full_vdm_size == 0 {
        return Err(PdmProjectionError::EmptyDimension);
    }
    if pdms.len() == 0 {
        return Err(PdmProjectionError::EmptyPdmSet);
    }
    let mut physical_actuator_extent = 0_usize;
    for (pdm_size, projection_full_vdm_size) in pdms {
        full_vdm_size
            .checked_mul(pdm_size)
            .ok_or(PdmProjectionError::ExtentOverflow)?;
        if projection_full_vdm_size != full_vdm_size {
            return Err(PdmProjectionError::ProjectionExtent);
        }
        physical_actuator_extent = physical_actuator_extent
            .checked_add(pdm_size)
            .ok_or(PdmProjectionError::ExtentOverflow)?;
    }
    Ok(physical_actuator_extent)
}

pub(super) fn validate_pdm_projections<T>(
    active_vdm_size: usize,
    full_vdm_size: usize,
    pdms: &[PdmProjection<T>],
) -> Result<usize, PdmProjectionError>
where
    T: MatrixElement,
{
    if active_vdm_size == 0 || full_vdm_size == 0 {
        return Err(PdmProjectionError::EmptyDimension);
    }
    if pdms.is_empty() {
        return Err(PdmProjectionError::EmptyPdmSet);
    }
    let mut physical_actuator_extent = 0_usize;
    for pdm in pdms {
        full_vdm_size
            .checked_mul(pdm.pdm_size)
            .ok_or(PdmProjectionError::ExtentOverflow)?;
        if pdm.full_vdm_size != full_vdm_size {
            return Err(PdmProjectionError::ProjectionExtent);
        }
        physical_actuator_extent = physical_actuator_extent
            .checked_add(pdm.pdm_size)
            .ok_or(PdmProjectionError::ExtentOverflow)?;
    }
    Ok(physical_actuator_extent)
}
