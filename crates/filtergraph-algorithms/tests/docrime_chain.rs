//! Complete transport-independent DO-CRIME open-loop algorithm chain.

use filtergraph_algorithms::{
    AlgorithmPlan, DocrimeConditioningCriteria, DocrimeCovarianceInput, DocrimeCovariancePlan,
    DocrimeExcitationPlan, DocrimeInteractionMatrixDiagnostics, DocrimeInteractionMatrixOutput,
    DocrimeInteractionMatrixStatus, DocrimeOpenLoopInteractionMatrixPlan,
};

#[test]
fn deterministic_excitation_and_centered_covariance_recover_the_interaction_matrix() {
    let excitation = DocrimeExcitationPlan::new([0.5_f64, 1.0], 0x0123_4567_89ab_cdef).unwrap();
    let covariance = DocrimeCovariancePlan::new(2, 2).unwrap();
    let solve = DocrimeOpenLoopInteractionMatrixPlan::new(
        2,
        2,
        DocrimeConditioningCriteria::new(1.0e-4, 0.99, 1.0e-4).unwrap(),
    )
    .unwrap();
    let mut excitation_workspace = excitation.workspace();
    let mut covariance_workspace = covariance.workspace();
    let mut solve_workspace = solve.workspace();
    let mut calibration_excitation = [0.0; 2];

    for _ in 0..64 {
        excitation
            .process(&mut calibration_excitation, &mut excitation_workspace, ())
            .unwrap();
        let measurements = [
            10.0 + 1.5 * calibration_excitation[0] - 0.25 * calibration_excitation[1],
            -3.0 + 0.4 * calibration_excitation[0] + 2.0 * calibration_excitation[1],
        ];
        covariance
            .process(
                (),
                &mut covariance_workspace,
                DocrimeCovarianceInput::new(&measurements, &calibration_excitation),
            )
            .unwrap();
    }

    let mut interaction_matrix = [f64::NAN; 4];
    let mut diagnostics = DocrimeInteractionMatrixDiagnostics::default();
    solve
        .process(
            DocrimeInteractionMatrixOutput::new(&mut interaction_matrix, &mut diagnostics),
            &mut solve_workspace,
            covariance_workspace.covariances(),
        )
        .unwrap();

    assert_eq!(diagnostics.status(), DocrimeInteractionMatrixStatus::Valid);
    assert_eq!(diagnostics.sample_count(), 64);
    for (actual, expected) in interaction_matrix.iter().zip([1.5, -0.25, 0.4, 2.0]) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "{actual} != {expected}"
        );
    }
}
