//! Bounded physical optical-gain update tests.

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmPlan, OpticalGainAssignment, OpticalGainUpdateCriteria,
    OpticalGainUpdateError, OpticalGainUpdateF32, OpticalGainUpdateF32Configuration,
    OpticalGainUpdateInput, OpticalGainUpdateOutput, OpticalGainUpdatePlan,
    OpticalGainUpdatePlanError, OpticalGainUpdateWorkspaceError,
};

fn criteria<T>(minimum_gain: T, maximum_gain: T) -> OpticalGainUpdateCriteria<T>
where
    T: From<f32>,
{
    OpticalGainUpdateCriteria::new(
        T::from(0.1),
        T::from(0.05),
        minimum_gain,
        maximum_gain,
        T::from(0.25),
    )
}

struct OutputStorage<T> {
    optical_gains: Vec<T>,
    accepted: Vec<bool>,
    updated: Vec<bool>,
}

impl<T: Clone> OutputStorage<T> {
    fn filled(dither_modes: usize, reconstructed_modes: usize, value: T) -> Self {
        Self {
            optical_gains: vec![value; reconstructed_modes],
            accepted: vec![true; dither_modes],
            updated: vec![true; reconstructed_modes],
        }
    }

    fn output(&mut self) -> OpticalGainUpdateOutput<'_, T> {
        OpticalGainUpdateOutput::new(
            &mut self.optical_gains,
            &mut self.accepted,
            &mut self.updated,
        )
    }
}

#[test]
fn preparation_validates_criteria_and_assignments() {
    let assignment = OpticalGainAssignment::DitherMode {
        dither_mode: 0,
        update_fraction: 1.0_f32,
    };
    assert_eq!(
        OpticalGainUpdatePlan::new([], [assignment]).unwrap_err(),
        OpticalGainUpdatePlanError::EmptyCriteria
    );
    assert_eq!(
        OpticalGainUpdatePlan::<f32>::new([criteria(0.1, 2.0)], []).unwrap_err(),
        OpticalGainUpdatePlanError::EmptyAssignments
    );
    assert_eq!(
        OpticalGainUpdatePlan::new(
            [OpticalGainUpdateCriteria::new(-0.1_f32, 0.0, 0.1, 2.0, 0.1)],
            [assignment],
        )
        .unwrap_err(),
        OpticalGainUpdatePlanError::InvalidAmplitudeThreshold { dither_mode: 0 }
    );
    assert_eq!(
        OpticalGainUpdatePlan::new(
            [OpticalGainUpdateCriteria::new(0.0_f32, 0.0, 0.0, 2.0, 0.1)],
            [assignment],
        )
        .unwrap_err(),
        OpticalGainUpdatePlanError::InvalidOpticalGainBounds { dither_mode: 0 }
    );
    assert_eq!(
        OpticalGainUpdatePlan::new(
            [OpticalGainUpdateCriteria::new(
                0.0_f32,
                0.0,
                f32::from_bits(1),
                2.0,
                0.1,
            )],
            [assignment],
        )
        .unwrap_err(),
        OpticalGainUpdatePlanError::InvalidOpticalGainBounds { dither_mode: 0 }
    );
    assert_eq!(
        OpticalGainUpdatePlan::new(
            [OpticalGainUpdateCriteria::new(
                0.0_f32,
                0.0,
                0.1,
                2.0,
                core::f32::consts::PI + 0.1,
            )],
            [assignment],
        )
        .unwrap_err(),
        OpticalGainUpdatePlanError::InvalidMaximumAbsolutePhaseLag { dither_mode: 0 }
    );
    assert_eq!(
        OpticalGainUpdatePlan::new(
            [criteria(0.1_f32, 2.0)],
            [OpticalGainAssignment::DitherMode {
                dither_mode: 1,
                update_fraction: 1.0,
            }],
        )
        .unwrap_err(),
        OpticalGainUpdatePlanError::AssignmentDitherModeOutOfBounds {
            reconstructed_mode: 0,
            dither_mode: 1,
        }
    );
    assert_eq!(
        OpticalGainUpdatePlan::new(
            [criteria(0.1_f32, 2.0)],
            [OpticalGainAssignment::DitherMode {
                dither_mode: 0,
                update_fraction: 0.0,
            }],
        )
        .unwrap_err(),
        OpticalGainUpdatePlanError::InvalidUpdateFraction {
            reconstructed_mode: 0,
        }
    );
}

#[test]
fn initial_physical_gains_are_validated_against_assignments() {
    let plan = OpticalGainUpdatePlan::new(
        [criteria(0.2_f32, 1.0)],
        [
            OpticalGainAssignment::DitherMode {
                dither_mode: 0,
                update_fraction: 1.0,
            },
            OpticalGainAssignment::Hold,
        ],
    )
    .unwrap();
    assert_eq!(
        plan.workspace(&[1.0]).unwrap_err(),
        OpticalGainUpdateWorkspaceError::InitialOpticalGainExtent {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        plan.workspace(&[f32::NAN, 1.0]).unwrap_err(),
        OpticalGainUpdateWorkspaceError::InvalidInitialOpticalGain {
            reconstructed_mode: 0,
        }
    );
    assert_eq!(
        plan.workspace(&[1.1, 1.0]).unwrap_err(),
        OpticalGainUpdateWorkspaceError::InitialOpticalGainOutsideBounds {
            reconstructed_mode: 0,
            dither_mode: 0,
        }
    );
    assert_eq!(
        plan.workspace(&[1.0, 3.0]).unwrap().optical_gains(),
        &[1.0, 3.0]
    );
}

#[test]
fn incomplete_window_holds_without_reading_unpublished_values() {
    let plan = OpticalGainUpdatePlan::new(
        [criteria(0.1_f32, 2.0)],
        [OpticalGainAssignment::DitherMode {
            dither_mode: 0,
            update_fraction: 1.0,
        }],
    )
    .unwrap();
    let mut workspace = plan.workspace(&[0.8]).unwrap();
    let mut storage = OutputStorage::filled(1, 1, f32::NAN);

    plan.process(
        storage.output(),
        &mut workspace,
        OpticalGainUpdateInput::new(
            &[f32::NAN],
            &[f32::NAN],
            &[f32::NAN],
            &[f32::NAN],
            &[true],
            false,
        ),
    )
    .unwrap();

    assert_eq!(storage.optical_gains, [0.8]);
    assert_eq!(storage.accepted, [false]);
    assert_eq!(storage.updated, [false]);
    assert_eq!(workspace.optical_gains(), &[0.8]);
}

#[test]
fn accepted_estimates_support_global_grouped_and_held_assignments() {
    let plan = OpticalGainUpdatePlan::new(
        [criteria(0.2_f64, 1.0), criteria(0.2, 1.0)],
        [
            OpticalGainAssignment::DitherMode {
                dither_mode: 0,
                update_fraction: 1.0,
            },
            OpticalGainAssignment::DitherMode {
                dither_mode: 0,
                update_fraction: 0.5,
            },
            OpticalGainAssignment::DitherMode {
                dither_mode: 1,
                update_fraction: 0.25,
            },
            OpticalGainAssignment::Hold,
        ],
    )
    .unwrap();
    let mut workspace = plan.workspace(&[1.0, 0.8, 0.8, 3.0]).unwrap();
    let mut storage = OutputStorage::filled(2, 4, f64::NAN);

    plan.process(
        storage.output(),
        &mut workspace,
        OpticalGainUpdateInput::new(
            &[0.5, 0.5],
            &[0.25, 0.2],
            &[0.5, 0.4],
            &[0.1, -0.1],
            &[true, true],
            true,
        ),
    )
    .unwrap();

    assert_eq!(
        storage.optical_gains,
        [0.5, 0.65, 0.8 + 0.25 * (0.4 - 0.8), 3.0]
    );
    assert_eq!(storage.accepted, [true, true]);
    assert_eq!(storage.updated, [true, true, true, false]);
    assert_eq!(workspace.optical_gains(), storage.optical_gains);
}

#[test]
fn every_acceptance_condition_can_hold_an_assigned_gain() {
    let plan = OpticalGainUpdatePlan::new(
        [criteria(0.2_f32, 1.0)],
        [OpticalGainAssignment::DitherMode {
            dither_mode: 0,
            update_fraction: 1.0,
        }],
    )
    .unwrap();

    for (gain, phase) in [(0.2, 0.25), (1.0, -0.25)] {
        let mut workspace = plan.workspace(&[0.8]).unwrap();
        let mut storage = OutputStorage::filled(1, 1, f32::NAN);
        plan.process(
            storage.output(),
            &mut workspace,
            OpticalGainUpdateInput::new(&[0.1], &[0.05], &[gain], &[phase], &[true], true),
        )
        .unwrap();
        assert_eq!(storage.optical_gains, [gain]);
        assert_eq!(storage.accepted, [true]);
        assert_eq!(storage.updated, [true]);
    }

    let rejected = [
        (0.09, 0.5, 0.5, 0.0, true),
        (0.5, 0.04, 0.5, 0.0, true),
        (0.5, 0.5, 0.19, 0.0, true),
        (0.5, 0.5, 1.01, 0.0, true),
        (0.5, 0.5, 0.5, 0.26, true),
        (0.5, 0.5, 0.5, 0.0, false),
    ];
    for (command, response, gain, phase, valid) in rejected {
        let mut workspace = plan.workspace(&[0.8]).unwrap();
        let mut storage = OutputStorage::filled(1, 1, f32::NAN);
        plan.process(
            storage.output(),
            &mut workspace,
            OpticalGainUpdateInput::new(&[command], &[response], &[gain], &[phase], &[valid], true),
        )
        .unwrap();
        assert_eq!(storage.optical_gains, [0.8]);
        assert_eq!(storage.accepted, [false]);
        assert_eq!(storage.updated, [false]);
        assert_eq!(workspace.optical_gains(), &[0.8]);
    }
}

#[test]
fn execution_errors_preserve_committed_history_and_output() {
    let plan = OpticalGainUpdatePlan::new(
        [criteria(0.1_f64, 2.0)],
        [OpticalGainAssignment::DitherMode {
            dither_mode: 0,
            update_fraction: 0.5,
        }],
    )
    .unwrap();
    let mut workspace = plan.workspace(&[1.0]).unwrap();
    let mut storage = OutputStorage::filled(1, 1, 7.0_f64);

    assert_eq!(
        plan.process(
            storage.output(),
            &mut workspace,
            OpticalGainUpdateInput::new(&[], &[], &[], &[], &[], true),
        )
        .unwrap_err(),
        OpticalGainUpdateError::InputExtent
    );
    assert_eq!(workspace.optical_gains(), &[1.0]);
    assert_eq!(storage.optical_gains, [7.0]);

    assert_eq!(
        plan.process(
            storage.output(),
            &mut workspace,
            OpticalGainUpdateInput::new(&[1.0], &[1.0], &[f64::NAN], &[0.0], &[true], true,),
        )
        .unwrap_err(),
        OpticalGainUpdateError::NonFiniteInput { dither_mode: 0 }
    );
    assert_eq!(workspace.optical_gains(), &[1.0]);
    assert_eq!(storage.optical_gains, [7.0]);

    let mut empty_gains = [];
    let mut accepted = [false];
    let mut empty_updated = [];
    assert_eq!(
        plan.process(
            OpticalGainUpdateOutput::new(&mut empty_gains, &mut accepted, &mut empty_updated,),
            &mut workspace,
            OpticalGainUpdateInput::new(&[1.0], &[1.0], &[1.0], &[0.0], &[true], false),
        )
        .unwrap_err(),
        OpticalGainUpdateError::OutputExtent
    );
    assert_eq!(workspace.optical_gains(), &[1.0]);

    let two_mode_plan = OpticalGainUpdatePlan::new(
        [criteria(0.1_f64, 2.0)],
        [
            OpticalGainAssignment::DitherMode {
                dither_mode: 0,
                update_fraction: 0.5,
            },
            OpticalGainAssignment::Hold,
        ],
    )
    .unwrap();
    let mut two_mode_storage = OutputStorage::filled(1, 2, 7.0_f64);
    assert_eq!(
        two_mode_plan
            .process(
                two_mode_storage.output(),
                &mut workspace,
                OpticalGainUpdateInput::new(&[1.0], &[1.0], &[1.0], &[0.0], &[true], false),
            )
            .unwrap_err(),
        OpticalGainUpdateError::WorkspaceExtent
    );
    assert_eq!(workspace.optical_gains(), &[1.0]);
}

#[test]
fn unity_update_fraction_preserves_a_small_invertible_estimate() {
    let minimum = f64::MIN_POSITIVE;
    let plan = OpticalGainUpdatePlan::new(
        [criteria(minimum, f64::MAX)],
        [OpticalGainAssignment::DitherMode {
            dither_mode: 0,
            update_fraction: 1.0,
        }],
    )
    .unwrap();
    let mut workspace = plan.workspace(&[f64::MAX]).unwrap();
    let mut storage = OutputStorage::filled(1, 1, f64::NAN);

    plan.process(
        storage.output(),
        &mut workspace,
        OpticalGainUpdateInput::new(&[1.0], &[1.0], &[minimum], &[0.0], &[true], true),
    )
    .unwrap();

    assert_eq!(storage.optical_gains, [minimum]);
    assert_eq!(workspace.optical_gains(), &[minimum]);
    assert!((1.0 / storage.optical_gains[0]).is_finite());
}

#[test]
fn declaration_filters_complete_estimates_and_reset_restores_initial_gains() {
    let mut declared = OpticalGainUpdateF32::prepare(OpticalGainUpdateF32Configuration {
        criteria: vec![criteria(0.1_f32, 2.0)],
        assignments: vec![
            OpticalGainAssignment::DitherMode {
                dither_mode: 0,
                update_fraction: 1.0,
            },
            OpticalGainAssignment::Hold,
        ],
        initial_optical_gains: vec![1.0, 3.0],
    })
    .unwrap();
    assert_eq!(OpticalGainUpdateF32::NAME, "optical-gain-update-f32");
    assert_eq!(declared.formats().len(), 9);

    let mut optical_gains = [f32::NAN; 2];
    let mut accepted = [true];
    let mut updated = [true; 2];
    OpticalGainUpdateF32::process(
        &mut declared,
        &[f32::NAN],
        &[f32::NAN],
        &[f32::NAN],
        &[f32::NAN],
        &[true],
        &[false],
        &mut optical_gains,
        &mut accepted,
        &mut updated,
    )
    .unwrap();
    assert_eq!(optical_gains, [1.0, 3.0]);
    assert_eq!(accepted, [false]);
    assert_eq!(updated, [false, false]);

    OpticalGainUpdateF32::process(
        &mut declared,
        &[0.5],
        &[0.25],
        &[0.5],
        &[0.0],
        &[true],
        &[true],
        &mut optical_gains,
        &mut accepted,
        &mut updated,
    )
    .unwrap();
    assert_eq!(optical_gains, [0.5, 3.0]);
    assert_eq!(accepted, [true]);
    assert_eq!(updated, [true, false]);

    OpticalGainUpdateF32::reset(&mut declared);
    OpticalGainUpdateF32::process(
        &mut declared,
        &[f32::NAN],
        &[f32::NAN],
        &[f32::NAN],
        &[f32::NAN],
        &[true],
        &[false],
        &mut optical_gains,
        &mut accepted,
        &mut updated,
    )
    .unwrap();
    assert_eq!(optical_gains, [1.0, 3.0]);
    assert_eq!(accepted, [false]);
    assert_eq!(updated, [false, false]);
}
