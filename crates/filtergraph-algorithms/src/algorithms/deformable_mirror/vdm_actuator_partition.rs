//! Control roles for coordinates in a full virtual deformable mirror.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

/// Control role of one coordinate in canonical full-VDM order.
///
/// This role is independent of physical actuator health. For example, a
/// device adapter may report a stuck physical actuator while the associated
/// full-VDM coordinate is assigned [`Self::Fixed`] by the active profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VdmActuatorRole {
    /// An independent coordinate directly produced by the controller.
    Controlled,
    /// A dependent coordinate completed from the controlled coordinates.
    Extrapolated,
    /// A coordinate assigned a prepared value rather than controller output.
    Fixed,
}

/// Invalid full-VDM actuator partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VdmActuatorPartitionError {
    /// A full VDM must contain at least one coordinate.
    Empty,
    /// An executable control partition must contain a controlled coordinate.
    NoControlledActuator,
}

impl fmt::Display for VdmActuatorPartitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("full VDM actuator partition must be nonempty"),
            Self::NoControlledActuator => {
                formatter.write_str("VDM actuator partition has no controlled actuator")
            }
        }
    }
}

impl core::error::Error for VdmActuatorPartitionError {}

/// Prepared, exhaustive control partition in canonical full-VDM order.
///
/// Every full-VDM coordinate has exactly one role. The packed controlled,
/// extrapolated, and fixed index lists preserve canonical full-VDM order; a
/// different controller order requires an explicit controller-to-VDM
/// projection rather than a permutation hidden in this descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VdmActuatorPartition {
    roles: Arc<[VdmActuatorRole]>,
    controlled_indices: Arc<[usize]>,
    extrapolated_indices: Arc<[usize]>,
    fixed_indices: Arc<[usize]>,
}

impl VdmActuatorPartition {
    /// Validates and prepares one role for every full-VDM coordinate.
    pub fn new(
        roles: impl Into<Arc<[VdmActuatorRole]>>,
    ) -> Result<Self, VdmActuatorPartitionError> {
        let roles = roles.into();
        if roles.is_empty() {
            return Err(VdmActuatorPartitionError::Empty);
        }

        let mut controlled_indices = Vec::new();
        let mut extrapolated_indices = Vec::new();
        let mut fixed_indices = Vec::new();
        for (index, role) in roles.iter().enumerate() {
            match role {
                VdmActuatorRole::Controlled => controlled_indices.push(index),
                VdmActuatorRole::Extrapolated => extrapolated_indices.push(index),
                VdmActuatorRole::Fixed => fixed_indices.push(index),
            }
        }
        if controlled_indices.is_empty() {
            return Err(VdmActuatorPartitionError::NoControlledActuator);
        }

        Ok(Self {
            roles,
            controlled_indices: Arc::from(controlled_indices),
            extrapolated_indices: Arc::from(extrapolated_indices),
            fixed_indices: Arc::from(fixed_indices),
        })
    }

    /// Returns one role for every coordinate in canonical full-VDM order.
    pub fn roles(&self) -> &[VdmActuatorRole] {
        &self.roles
    }

    /// Returns the full-VDM extent.
    pub fn full_vdm_size(&self) -> usize {
        self.roles.len()
    }

    /// Returns full-VDM indices of independently controlled coordinates.
    pub fn controlled_indices(&self) -> &[usize] {
        &self.controlled_indices
    }

    /// Returns full-VDM indices of coordinates completed by extrapolation.
    pub fn extrapolated_indices(&self) -> &[usize] {
        &self.extrapolated_indices
    }

    /// Returns full-VDM indices assigned prepared fixed values.
    pub fn fixed_indices(&self) -> &[usize] {
        &self.fixed_indices
    }
}
