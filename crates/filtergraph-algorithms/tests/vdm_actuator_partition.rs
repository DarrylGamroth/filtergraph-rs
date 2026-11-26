//! Full-VDM actuator control-partition fixtures.

use filtergraph_algorithms::{VdmActuatorPartition, VdmActuatorPartitionError, VdmActuatorRole};

#[test]
fn partition_preserves_canonical_full_vdm_order() {
    let partition = VdmActuatorPartition::new([
        VdmActuatorRole::Extrapolated,
        VdmActuatorRole::Controlled,
        VdmActuatorRole::Fixed,
        VdmActuatorRole::Controlled,
        VdmActuatorRole::Extrapolated,
    ])
    .unwrap();

    assert_eq!(partition.full_vdm_size(), 5);
    assert_eq!(partition.controlled_indices(), [1, 3]);
    assert_eq!(partition.extrapolated_indices(), [0, 4]);
    assert_eq!(partition.fixed_indices(), [2]);
    assert_eq!(
        partition.roles(),
        [
            VdmActuatorRole::Extrapolated,
            VdmActuatorRole::Controlled,
            VdmActuatorRole::Fixed,
            VdmActuatorRole::Controlled,
            VdmActuatorRole::Extrapolated,
        ]
    );
}

#[test]
fn executable_partition_requires_a_controlled_coordinate() {
    assert_eq!(
        VdmActuatorPartition::new([]),
        Err(VdmActuatorPartitionError::Empty)
    );
    assert_eq!(
        VdmActuatorPartition::new([VdmActuatorRole::Extrapolated, VdmActuatorRole::Fixed,]),
        Err(VdmActuatorPartitionError::NoControlledActuator)
    );
}
