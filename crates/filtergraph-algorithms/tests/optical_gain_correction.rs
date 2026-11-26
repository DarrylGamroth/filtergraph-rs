//! Reciprocal physical optical-gain correction tests.

use filtergraph_algorithms::{
    AlgorithmPlan, NoWorkspace, OpticalGainCorrectionError, OpticalGainCorrectionPlan,
    OpticalGainCorrectionPlanError,
};

#[test]
fn preparation_keeps_physical_gains_distinct_from_reciprocal_corrections() {
    let plan = OpticalGainCorrectionPlan::new([1.0_f32, 0.5, 0.25]).unwrap();

    assert_eq!(plan.optical_gains(), &[1.0, 0.5, 0.25]);
    assert_eq!(plan.reciprocal_corrections(), &[1.0, 2.0, 4.0]);
    assert_eq!(plan.shared_optical_gains().as_ref(), plan.optical_gains());
    assert_eq!(
        plan.shared_reciprocal_corrections().as_ref(),
        plan.reciprocal_corrections()
    );
}

#[test]
fn correction_divides_raw_reconstructed_modes_by_physical_gain() {
    let plan = OpticalGainCorrectionPlan::new([1.0_f64, 0.5, 0.25]).unwrap();
    let mut corrected = [0.0; 3];

    plan.process(&mut corrected, &mut NoWorkspace, &[2.0, -2.0, 0.125])
        .unwrap();

    assert_eq!(corrected, [2.0, -4.0, 0.5]);
}

#[test]
fn preparation_rejects_invalid_physical_gains_and_reciprocals() {
    assert_eq!(
        OpticalGainCorrectionPlan::<f32>::new([]).unwrap_err(),
        OpticalGainCorrectionPlanError::Empty
    );
    for optical_gain in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            OpticalGainCorrectionPlan::new([optical_gain]).unwrap_err(),
            OpticalGainCorrectionPlanError::InvalidOpticalGain { mode: 0 }
        );
    }
    assert_eq!(
        OpticalGainCorrectionPlan::new([f32::from_bits(1)]).unwrap_err(),
        OpticalGainCorrectionPlanError::NonFiniteReciprocalCorrection { mode: 0 }
    );
}

#[test]
fn extent_and_nonfinite_errors_leave_output_unchanged() {
    let plan = OpticalGainCorrectionPlan::new([0.5_f32, 0.25]).unwrap();
    let mut output = [7.0_f32, 8.0];

    assert_eq!(
        plan.process(&mut output, &mut NoWorkspace, &[1.0])
            .unwrap_err(),
        OpticalGainCorrectionError::InputExtent {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(output, [7.0, 8.0]);

    let mut short = [7.0_f32];
    assert_eq!(
        plan.process(&mut short, &mut NoWorkspace, &[1.0, 2.0])
            .unwrap_err(),
        OpticalGainCorrectionError::OutputExtent {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(short, [7.0]);

    assert_eq!(
        plan.process(&mut output, &mut NoWorkspace, &[1.0, f32::NAN])
            .unwrap_err(),
        OpticalGainCorrectionError::NonFiniteInput { mode: 1 }
    );
    assert_eq!(output, [7.0, 8.0]);
}

#[test]
fn nonfinite_result_is_rejected_before_any_output_is_written() {
    let plan = OpticalGainCorrectionPlan::new([1.0_f64, 0.5]).unwrap();
    let mut output = [7.0_f64, 8.0];

    assert_eq!(
        plan.process(&mut output, &mut NoWorkspace, &[1.0, f64::MAX])
            .unwrap_err(),
        OpticalGainCorrectionError::NonFiniteOutput { mode: 1 }
    );
    assert_eq!(output, [7.0, 8.0]);
}
