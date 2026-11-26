//! End-to-end composition of the independent optical-gain algorithms.

use std::sync::Arc;

use filtergraph_algorithms::{
    AlgorithmPlan, NoWorkspace, OpticalGainAssignment, OpticalGainCorrectionPlan,
    OpticalGainDitherMode, OpticalGainDitherPlan, OpticalGainDitherSchedule,
    OpticalGainEstimateOutput, OpticalGainEstimatorInput, OpticalGainEstimatorPlan,
    OpticalGainUpdateCriteria, OpticalGainUpdateInput, OpticalGainUpdateOutput,
    OpticalGainUpdatePlan, PreparedGemv,
};

fn assert_near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 64.0 * f64::EPSILON * expected.abs().max(1.0),
        "{actual} is not near {expected}"
    );
}

#[test]
fn raw_response_chain_updates_and_applies_a_global_physical_gain() {
    let schedule = Arc::new(
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(0.4_f64, 8, 3)]).unwrap(),
    );
    let generator = OpticalGainDitherPlan::new(Arc::clone(&schedule));
    let dither_to_command = PreparedGemv::new(2, 1, Arc::from([1.0_f64, -1.0])).unwrap();
    let command_to_dither = PreparedGemv::new(1, 2, Arc::from([0.5_f64, -0.5])).unwrap();
    let response_to_dither = PreparedGemv::new(1, 2, Arc::from([0.5_f64, -0.5])).unwrap();
    let estimator = OpticalGainEstimatorPlan::new(Arc::clone(&schedule), [1.0]).unwrap();
    let update = OpticalGainUpdatePlan::new(
        [OpticalGainUpdateCriteria::new(0.39, 0.19, 0.4, 1.1, 0.1)],
        [
            OpticalGainAssignment::DitherMode {
                dither_mode: 0,
                update_fraction: 1.0,
            },
            OpticalGainAssignment::DitherMode {
                dither_mode: 0,
                update_fraction: 1.0,
            },
        ],
    )
    .unwrap();

    let mut generator_workspace = generator.workspace();
    let mut estimator_workspace = estimator.workspace();
    let mut update_workspace = update.workspace(&[1.0, 1.0]).unwrap();
    let mut modal_dither = [0.0_f64; 1];
    let mut command = [0.0_f64; 2];
    let mut realized_modal_command = [0.0_f64; 1];
    let mut raw_reconstructed_response = [0.0_f64; 2];
    let mut modal_response = [0.0_f64; 1];
    let mut command_amplitude = [f64::NAN; 1];
    let mut response_amplitude = [f64::NAN; 1];
    let mut estimated_gain = [f64::NAN; 1];
    let mut phase_lag = [f64::NAN; 1];
    let mut estimate_valid = [false; 1];
    let mut estimate_complete = true;
    let mut updated_gains = [f64::NAN; 2];
    let mut accepted_dither = [true; 1];
    let mut updated_reconstructed = [true; 2];

    for frame in 0..schedule.frames_per_estimate() {
        generator
            .process(&mut modal_dither, &mut generator_workspace, ())
            .unwrap();
        dither_to_command
            .multiply(&mut command, &modal_dither)
            .unwrap();
        command_to_dither
            .multiply(&mut realized_modal_command, &command)
            .unwrap();

        // This explicit plant response has a physical optical gain of 0.5.
        for (response, command) in raw_reconstructed_response.iter_mut().zip(command) {
            *response = 0.5 * command;
        }
        response_to_dither
            .multiply(&mut modal_response, &raw_reconstructed_response)
            .unwrap();

        estimator
            .process(
                OpticalGainEstimateOutput::new(
                    &mut command_amplitude,
                    &mut response_amplitude,
                    &mut estimated_gain,
                    &mut phase_lag,
                    &mut estimate_valid,
                    &mut estimate_complete,
                ),
                &mut estimator_workspace,
                OpticalGainEstimatorInput::new(&realized_modal_command, &modal_response),
            )
            .unwrap();
        update
            .process(
                OpticalGainUpdateOutput::new(
                    &mut updated_gains,
                    &mut accepted_dither,
                    &mut updated_reconstructed,
                ),
                &mut update_workspace,
                OpticalGainUpdateInput::new(
                    &command_amplitude,
                    &response_amplitude,
                    &estimated_gain,
                    &phase_lag,
                    &estimate_valid,
                    estimate_complete,
                ),
            )
            .unwrap();

        if frame + 1 < schedule.frames_per_estimate() {
            assert!(!estimate_complete);
            assert_eq!(updated_gains, [1.0, 1.0]);
            assert_eq!(accepted_dither, [false]);
            assert_eq!(updated_reconstructed, [false, false]);
        }
    }

    assert!(estimate_complete);
    assert_near(command_amplitude[0], 0.4);
    assert_near(response_amplitude[0], 0.2);
    assert_near(estimated_gain[0], 0.5);
    assert_near(phase_lag[0], 0.0);
    assert_eq!(estimate_valid, [true]);
    assert_eq!(accepted_dither, [true]);
    assert_eq!(updated_reconstructed, [true, true]);
    assert_near(updated_gains[0], 0.5);
    assert_near(updated_gains[1], 0.5);

    // Preparation and activation of a replacement correction remain outside
    // the repeated algorithm calls represented above.
    let correction = OpticalGainCorrectionPlan::new(Arc::from(updated_gains)).unwrap();
    assert_eq!(correction.reciprocal_corrections(), &[2.0, 2.0]);
    let mut corrected = [0.0_f64; 2];
    correction
        .process(&mut corrected, &mut NoWorkspace, &[0.25_f64, -0.5])
        .unwrap();
    assert_eq!(corrected, [0.5, -1.0]);
}
