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
    use OpticalGainCorrectionPlanError::*;
    for (gains, expected, detail) in [
        (&[][..], Empty, "must be nonempty"),
        (
            &[1.0, 0.0][..],
            InvalidOpticalGain { mode: 1 },
            "mode 1 must be finite and positive",
        ),
        (
            &[-1.0][..],
            InvalidOpticalGain { mode: 0 },
            "mode 0 must be finite and positive",
        ),
        (
            &[f32::NAN][..],
            InvalidOpticalGain { mode: 0 },
            "mode 0 must be finite and positive",
        ),
        (
            &[f32::INFINITY][..],
            InvalidOpticalGain { mode: 0 },
            "mode 0 must be finite and positive",
        ),
        (
            &[1.0, f32::from_bits(1)][..],
            NonFiniteReciprocalCorrection { mode: 1 },
            "reciprocal optical-gain correction for mode 1",
        ),
    ] {
        let error = OpticalGainCorrectionPlan::new(gains).unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
    }
}

#[test]
fn extent_and_nonfinite_errors_leave_output_unchanged() {
    let plan = OpticalGainCorrectionPlan::new([0.5_f32, 0.25]).unwrap();
    let mut output = [7.0_f32, 8.0];

    use OpticalGainCorrectionError::*;
    for (input, nout, expected, detail) in [
        (
            &[1.0][..],
            2,
            InputExtent {
                expected: 2,
                actual: 1,
            },
            "input extent 1 does not match 2",
        ),
        (
            &[1.0, 2.0][..],
            1,
            OutputExtent {
                expected: 2,
                actual: 1,
            },
            "output extent 1 does not match 2",
        ),
        (
            &[1.0, f32::NAN][..],
            2,
            NonFiniteInput { mode: 1 },
            "raw reconstructed input for optical-gain mode 1",
        ),
        (
            &[1.0, f32::MAX][..],
            2,
            NonFiniteOutput { mode: 1 },
            "mode 1",
        ),
    ] {
        let error = plan
            .process(&mut output[..nout], &mut NoWorkspace, input)
            .unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
        assert_eq!(output, [7.0, 8.0]);
    }
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
