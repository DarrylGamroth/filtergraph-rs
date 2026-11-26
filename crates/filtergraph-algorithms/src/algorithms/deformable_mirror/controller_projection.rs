//! Coherent forward-command and reverse-feedback projections for controllers.

use alloc::sync::Arc;
use core::fmt;

use crate::MatrixElement;

/// Invalid prepared controller/VDM projection set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControllerProjectionError {
    /// The output VDM or one controller command has zero extent.
    EmptyDimension,
    /// At least one controller projection is required.
    EmptyControllerSet,
    /// A projection has the wrong row-major extent.
    ProjectionExtent,
    /// A projection contains a non-finite value.
    NonFiniteConfiguration,
    /// A prepared extent overflowed addressable storage.
    ExtentOverflow,
}

impl fmt::Display for ControllerProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension => {
                formatter.write_str("VDM and controller command dimensions must be nonempty")
            }
            Self::EmptyControllerSet => {
                formatter.write_str("controller-to-VDM projection requires at least one controller")
            }
            Self::ProjectionExtent => {
                formatter.write_str("controller/VDM projection extent is invalid")
            }
            Self::NonFiniteConfiguration => {
                formatter.write_str("controller/VDM projections must be finite")
            }
            Self::ExtentOverflow => {
                formatter.write_str("controller/VDM projection extent overflow")
            }
        }
    }
}

impl core::error::Error for ControllerProjectionError {}

/// Coherent forward-command and reverse-feedback projections for one controller.
#[derive(Clone, Debug, PartialEq)]
pub struct ControllerProjection<T> {
    pub(super) command_size: usize,
    pub(super) controller_to_vdm: Arc<[T]>,
    pub(super) vdm_to_controller: Arc<[T]>,
}

/// Prepared controller-command projection into the common VDM coordinates.
#[derive(Clone, Debug)]
pub struct ControllerToVdmProjection<T> {
    pub(super) command_size: usize,
    pub(super) controller_to_vdm: Arc<[T]>,
    pub(super) kind: ControllerProjectionKind,
}

/// Prepared common-VDM feedback projection into controller coordinates.
#[derive(Clone, Debug)]
pub struct VdmToControllerProjection<T> {
    pub(super) command_size: usize,
    pub(super) vdm_to_controller: Arc<[T]>,
    pub(super) kind: ControllerProjectionKind,
}

impl<T: PartialEq> PartialEq for ControllerToVdmProjection<T> {
    fn eq(&self, other: &Self) -> bool {
        self.command_size == other.command_size
            && self.controller_to_vdm == other.controller_to_vdm
            && self.kind.is_identity() == other.kind.is_identity()
    }
}

impl<T: PartialEq> PartialEq for VdmToControllerProjection<T> {
    fn eq(&self, other: &Self) -> bool {
        self.command_size == other.command_size
            && self.vdm_to_controller == other.vdm_to_controller
            && self.kind.is_identity() == other.kind.is_identity()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ControllerProjectionKind {
    Identity,
    #[cfg(all(feature = "blas", target_arch = "x86_64"))]
    Contiguous(usize),
    RowMajor,
}

impl ControllerProjectionKind {
    pub(super) fn for_current_float_mode(self) -> Self {
        #[cfg(all(feature = "blas", target_arch = "x86_64"))]
        if matches!(self, Self::Contiguous(_)) {
            return self.for_float_mode(filtergraph_blas::sgemv_default_float_mode());
        }
        self
    }

    #[cfg(all(feature = "blas", target_arch = "x86_64"))]
    fn for_float_mode(self, default_mode: bool) -> Self {
        if !default_mode && matches!(self, Self::Contiguous(_)) {
            Self::RowMajor
        } else {
            self
        }
    }

    fn is_identity(self) -> bool {
        self == Self::Identity
    }
}

fn projection_kind<T>(rows: usize, columns: usize, matrix: &[T]) -> ControllerProjectionKind
where
    T: MatrixElement,
{
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
        return ControllerProjectionKind::Identity;
    }

    #[cfg(all(feature = "blas", target_arch = "x86_64"))]
    if rows != columns
        && size_of::<T>() == size_of::<f32>()
        && filtergraph_blas::supports_sgemv_dimensions(rows, columns)
        && let Some(offset) = contiguous_offset(rows, columns, matrix)
    {
        return ControllerProjectionKind::Contiguous(offset);
    }
    ControllerProjectionKind::RowMajor
}

#[cfg(all(feature = "blas", target_arch = "x86_64"))]
fn contiguous_offset<T: MatrixElement>(rows: usize, columns: usize, matrix: &[T]) -> Option<usize> {
    // MatrixElement is sealed to f32 and f64. Called only for f32, so
    // to_f32 preserves every coefficient bit, including signed zero.
    let bits = |value: T| value.to_f32().unwrap().to_bits();
    let first = matrix.iter().position(|&value| bits(value) != 0)?;
    let first_row = first / columns;
    let first_column = first % columns;
    let length = rows.min(columns);
    if first_row > rows - length || first_column > columns - length {
        return None;
    }
    let diagonal_rows = first_row..first_row + length;
    matrix
        .chunks_exact(columns)
        .enumerate()
        .all(|(row, values)| {
            let selected = diagonal_rows
                .contains(&row)
                .then(|| first_column + row - first_row);
            values.iter().enumerate().all(|(column, &value)| {
                let expected = if selected == Some(column) {
                    1.0_f32.to_bits()
                } else {
                    0
                };
                bits(value) == expected
            })
        })
        .then_some(first_row.max(first_column))
}

fn validate_direction<T>(
    command_size: usize,
    projection: &[T],
) -> Result<(), ControllerProjectionError>
where
    T: MatrixElement,
{
    if command_size == 0 || projection.is_empty() {
        return Err(ControllerProjectionError::EmptyDimension);
    }
    if !projection.len().is_multiple_of(command_size) {
        return Err(ControllerProjectionError::ProjectionExtent);
    }
    if projection.iter().any(|value| !value.is_finite()) {
        return Err(ControllerProjectionError::NonFiniteConfiguration);
    }
    Ok(())
}

impl<T> ControllerToVdmProjection<T>
where
    T: MatrixElement,
{
    /// Prepare one row-major VDM-by-controller command projection.
    pub fn new(
        command_size: usize,
        controller_to_vdm: Arc<[T]>,
    ) -> Result<Self, ControllerProjectionError> {
        validate_direction(command_size, &controller_to_vdm)?;
        let vdm_size = controller_to_vdm.len() / command_size;
        Ok(Self {
            command_size,
            kind: projection_kind(vdm_size, command_size, &controller_to_vdm),
            controller_to_vdm,
        })
    }

    /// Return this controller's command extent.
    pub const fn command_size(&self) -> usize {
        self.command_size
    }

    /// Return the common VDM extent.
    pub fn vdm_size(&self) -> usize {
        self.controller_to_vdm.len() / self.command_size
    }

    /// Return the row-major VDM-by-controller command projection.
    pub fn controller_to_vdm(&self) -> &[T] {
        &self.controller_to_vdm
    }
}

impl<T> VdmToControllerProjection<T>
where
    T: MatrixElement,
{
    /// Prepare one row-major controller-by-VDM feedback projection.
    pub fn new(
        command_size: usize,
        vdm_to_controller: Arc<[T]>,
    ) -> Result<Self, ControllerProjectionError> {
        validate_direction(command_size, &vdm_to_controller)?;
        let vdm_size = vdm_to_controller.len() / command_size;
        Ok(Self {
            command_size,
            kind: projection_kind(command_size, vdm_size, &vdm_to_controller),
            vdm_to_controller,
        })
    }

    /// Return this controller's command extent.
    pub const fn command_size(&self) -> usize {
        self.command_size
    }

    /// Return the common VDM extent.
    pub fn vdm_size(&self) -> usize {
        self.vdm_to_controller.len() / self.command_size
    }

    /// Return the row-major controller-by-VDM feedback projection.
    pub fn vdm_to_controller(&self) -> &[T] {
        &self.vdm_to_controller
    }
}

impl<T> ControllerProjection<T>
where
    T: MatrixElement,
{
    /// Prepares one controller projection pair.
    ///
    /// The containing plan validates both matrix extents against its VDM
    /// dimension. The reverse projection is explicit and is never inferred
    /// from the forward projection.
    pub fn new(
        command_size: usize,
        controller_to_vdm: Arc<[T]>,
        vdm_to_controller: Arc<[T]>,
    ) -> Result<Self, ControllerProjectionError> {
        if command_size == 0 {
            return Err(ControllerProjectionError::EmptyDimension);
        }
        if controller_to_vdm.len() != vdm_to_controller.len()
            || !controller_to_vdm.len().is_multiple_of(command_size)
        {
            return Err(ControllerProjectionError::ProjectionExtent);
        }
        if controller_to_vdm
            .iter()
            .chain(vdm_to_controller.iter())
            .any(|value| !value.is_finite())
        {
            return Err(ControllerProjectionError::NonFiniteConfiguration);
        }
        Ok(Self {
            command_size,
            controller_to_vdm,
            vdm_to_controller,
        })
    }

    /// Returns this controller's command extent.
    pub fn command_size(&self) -> usize {
        self.command_size
    }

    /// Returns the common VDM extent used by both projection directions.
    pub fn vdm_size(&self) -> usize {
        self.controller_to_vdm.len() / self.command_size
    }

    /// Returns the row-major controller-to-VDM projection.
    pub fn controller_to_vdm(&self) -> &[T] {
        &self.controller_to_vdm
    }

    /// Returns the independently prepared row-major VDM-to-controller projection.
    pub fn vdm_to_controller(&self) -> &[T] {
        &self.vdm_to_controller
    }

    /// Select the forward command direction without retaining a second copy
    /// of either projection matrix.
    pub fn controller_to_vdm_projection(&self) -> ControllerToVdmProjection<T> {
        ControllerToVdmProjection {
            command_size: self.command_size,
            controller_to_vdm: Arc::clone(&self.controller_to_vdm),
            kind: projection_kind(self.vdm_size(), self.command_size, &self.controller_to_vdm),
        }
    }

    /// Select the reverse feedback direction without retaining a second copy
    /// of either projection matrix.
    pub fn vdm_to_controller_projection(&self) -> VdmToControllerProjection<T> {
        VdmToControllerProjection {
            command_size: self.command_size,
            vdm_to_controller: Arc::clone(&self.vdm_to_controller),
            kind: projection_kind(self.command_size, self.vdm_size(), &self.vdm_to_controller),
        }
    }
}

pub(super) fn validate_controller_to_vdm_projections<T>(
    vdm_size: usize,
    controllers: &[ControllerToVdmProjection<T>],
) -> Result<usize, ControllerProjectionError>
where
    T: MatrixElement,
{
    validate_directional_controller_projections(
        vdm_size,
        controllers
            .iter()
            .map(|controller| (controller.command_size, controller.controller_to_vdm.len())),
    )
}

pub(super) fn validate_vdm_to_controller_projections<T>(
    vdm_size: usize,
    controllers: &[VdmToControllerProjection<T>],
) -> Result<usize, ControllerProjectionError>
where
    T: MatrixElement,
{
    validate_directional_controller_projections(
        vdm_size,
        controllers
            .iter()
            .map(|controller| (controller.command_size, controller.vdm_to_controller.len())),
    )
}

fn validate_directional_controller_projections(
    vdm_size: usize,
    controllers: impl ExactSizeIterator<Item = (usize, usize)>,
) -> Result<usize, ControllerProjectionError> {
    if vdm_size == 0 {
        return Err(ControllerProjectionError::EmptyDimension);
    }
    if controllers.len() == 0 {
        return Err(ControllerProjectionError::EmptyControllerSet);
    }
    let mut controller_command_extent = 0_usize;
    for (command_size, projection_extent) in controllers {
        let expected = vdm_size
            .checked_mul(command_size)
            .ok_or(ControllerProjectionError::ExtentOverflow)?;
        if projection_extent != expected {
            return Err(ControllerProjectionError::ProjectionExtent);
        }
        controller_command_extent = controller_command_extent
            .checked_add(command_size)
            .ok_or(ControllerProjectionError::ExtentOverflow)?;
    }
    Ok(controller_command_extent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_identities_and_scaled_projections_keep_their_behavior() {
        assert_eq!(
            projection_kind(2, 2, &[1.0_f32, -0.0, 0.0, 1.0]),
            ControllerProjectionKind::Identity
        );
        assert_eq!(
            projection_kind(2, 2, &[1.0_f32, 0.0, 0.0, 2.0]),
            ControllerProjectionKind::RowMajor
        );
    }

    #[cfg(all(feature = "blas", target_arch = "x86_64"))]
    #[test]
    fn rectangular_selectors_match_dense_in_both_orientations() {
        use crate::{
            AlgorithmPlan, ControllerToVdmError, ControllerToVdmPlan, NoWorkspace,
            VdmFeedbackToControllerError, VdmFeedbackToControllerPlan,
        };
        use alloc::vec;

        let (woofer, tweeter, full) = (97, 1412, 1509);
        let mut forward = vec![0.0_f32; tweeter * full];
        let mut reverse = forward.clone();
        for index in 0..tweeter {
            forward[index * full + woofer + index] = 1.0;
            reverse[(woofer + index) * tweeter + index] = 1.0;
        }
        let forward = ControllerToVdmProjection::new(full, Arc::from(forward)).unwrap();
        let reverse = VdmToControllerProjection::new(full, Arc::from(reverse)).unwrap();
        assert_eq!(forward.kind, ControllerProjectionKind::Contiguous(woofer));
        assert_eq!(reverse.kind, ControllerProjectionKind::Contiguous(woofer));

        let mut forward_dense = forward.clone();
        forward_dense.kind = ControllerProjectionKind::RowMajor;
        let optimized = ControllerToVdmPlan::from_controller_to_vdm_projections(
            tweeter,
            Arc::from([forward.clone(), forward]),
        )
        .unwrap();
        let dense = ControllerToVdmPlan::from_controller_to_vdm_projections(
            tweeter,
            Arc::from([forward_dense.clone(), forward_dense]),
        )
        .unwrap();
        let mut input = vec![1.0; full * 2];
        input[woofer] = -0.0;
        input[full + woofer] = -0.0;
        input[woofer + 1] = -f32::from_bits(1);
        input[full + woofer + 1] = -f32::from_bits(1);
        input[woofer + 2] = f32::MAX;
        input[full + woofer + 2] = -f32::MAX;
        let mut output = vec![0.0; tweeter];
        let mut reference = output.clone();
        let mut workspace = optimized.workspace();
        optimized
            .process(&mut output, &mut workspace, &input)
            .unwrap();
        dense
            .process(&mut reference, &mut dense.workspace(), &input)
            .unwrap();
        assert!(
            output
                .iter()
                .zip(&reference)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(output[..4], [0.0, -f32::from_bits(2), 0.0, 2.0]);

        input[0] = f32::NAN;
        assert_eq!(
            optimized.process(&mut output, &mut workspace, &input),
            Err(ControllerToVdmError::NonFiniteInput)
        );
        assert_eq!(
            optimized.process(&mut output, &mut workspace, &input[..full]),
            Err(ControllerToVdmError::InputExtent)
        );

        let mut reverse_dense = reverse.clone();
        reverse_dense.kind = ControllerProjectionKind::RowMajor;
        let optimized = VdmFeedbackToControllerPlan::from_vdm_to_controller_projections(
            tweeter,
            Arc::from([reverse]),
        )
        .unwrap();
        let dense = VdmFeedbackToControllerPlan::from_vdm_to_controller_projections(
            tweeter,
            Arc::from([reverse_dense]),
        )
        .unwrap();
        let input = &input[woofer..full];
        let mut feedback = vec![f32::NAN; full];
        let mut reference = feedback.clone();
        optimized
            .process(&mut feedback, &mut NoWorkspace, input)
            .unwrap();
        dense
            .process(&mut reference, &mut NoWorkspace, input)
            .unwrap();
        assert!(
            feedback
                .iter()
                .zip(&reference)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert!(feedback[..woofer].iter().all(|value| value.to_bits() == 0));
        assert_eq!(feedback[woofer + 1].to_bits(), input[1].to_bits());

        assert_eq!(
            optimized.process(&mut feedback[..tweeter], &mut NoWorkspace, input),
            Err(VdmFeedbackToControllerError::OutputExtent)
        );
    }

    #[cfg(all(feature = "blas", target_arch = "x86_64"))]
    #[test]
    fn selector_recognition_requires_bitwise_coefficients_and_a_contiguous_diagonal() {
        let mut matrix = [0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(
            projection_kind(2, 4, &matrix),
            ControllerProjectionKind::Contiguous(1)
        );
        let projection = VdmToControllerProjection::new(2, Arc::from(matrix)).unwrap();
        assert_eq!(
            projection.kind.for_float_mode(false),
            ControllerProjectionKind::RowMajor
        );
        assert_eq!(
            projection.kind.for_float_mode(true),
            ControllerProjectionKind::Contiguous(1)
        );
        matrix[0] = -0.0;
        let dense = VdmToControllerProjection::new(2, Arc::from(matrix)).unwrap();
        assert_eq!(projection, dense);
        for coefficient in [-0.0, f32::from_bits(1), -1.0] {
            matrix[0] = coefficient;
            assert_eq!(
                projection_kind(2, 4, &matrix),
                ControllerProjectionKind::RowMajor
            );
        }
        matrix[0] = 1.0;
        matrix[1] = 0.0;
        assert_eq!(
            projection_kind(2, 4, &matrix),
            ControllerProjectionKind::RowMajor
        );
    }
}
