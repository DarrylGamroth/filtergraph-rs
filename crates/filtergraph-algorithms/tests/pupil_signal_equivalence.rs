//! Source-derived coordinate, normalization, signature, and transactional output obligations.
use filtergraph_algorithms::{
    AlgorithmInput, BiOEdgeImageF32 as B, BiOEdgeImageF32Configuration as BC,
    CurvaturePairedChannelF32 as C, CurvaturePairedChannelF32Configuration as CC,
    CurvaturePairedImageF32 as I, CurvaturePairedImageF32Configuration as IC, PyramidImageF32 as P,
    PyramidImageF32Configuration as PC, ZernikePupilSignalF32 as Z,
    ZernikePupilSignalF32Configuration as ZC,
};

fn pyramid(policy: &str) -> PC {
    PC {
        image_size: (4, 5),
        pupil_size: (2, 2),
        pupil_origins: vec![[0, 0], [2, 0], [2, 3], [0, 3]],
        i4q_support: Some(vec![true, false, true, true]),
        initial_reference_signal: None,
        initial_optical_gain: None,
        normalization_policy: policy.into(),
        incidence_flux: 5.0,
    }
}
fn bio(policy: &str) -> BC {
    let c = pyramid(policy);
    BC {
        image_size: c.image_size,
        pupil_size: c.pupil_size,
        pupil_origins: c.pupil_origins,
        bi_o_edge_support: c.i4q_support,
        initial_reference_signal: c.initial_reference_signal,
        initial_optical_gain: c.initial_optical_gain,
        normalization_policy: c.normalization_policy,
        incidence_flux: c.incidence_flux,
        initial_calibration_signature: 7,
    }
}
fn image() -> [f32; 20] {
    [
        1.0,
        2.0,
        f32::NAN,
        4.0,
        8.0,
        3.0,
        5.0,
        0.0,
        6.0,
        10.0,
        2.0,
        4.0,
        0.0,
        1.0,
        3.0,
        6.0,
        8.0,
        0.0,
        5.0,
        7.0,
    ]
}
#[test]
fn differential_signs_column_major_support_and_outside_pupil_policy() {
    let mut p = P::prepare(pyramid("incidence-flux")).unwrap();
    let mut signal = [71.0; 6];
    let mut divisor = [71.0];
    P::process(&mut p, &image(), &mut signal, &mut divisor).unwrap();
    assert_eq!(signal, [0.4, -0.4, 0.6, -1.0, 0.0, -0.8]);
    assert_eq!(divisor, [5.0]);
    p.reset();
    let mut invalid = image();
    invalid[5] = f32::NAN;
    let before = signal;
    assert!(P::process(&mut p, &invalid, &mut signal, &mut divisor).is_err());
    assert_eq!(signal, before);
    assert_eq!(divisor, [5.0]);
    let mut b = B::prepare(bio("incidence-flux")).unwrap();
    B::process(&mut b, &image(), &[7], &mut signal, &mut divisor).unwrap();
    assert_eq!(signal, before);
    assert!(B::process(&mut b, &invalid, &[8], &mut signal, &mut divisor).is_err());
    assert_eq!(signal, before);
}
#[test]
fn current_frame_mean_uses_exact_256_lane_stride_and_tree() {
    let columns = 513;
    let mut p = P::prepare(PC {
        image_size: (4, columns),
        pupil_size: (1, columns),
        pupil_origins: vec![[0, 0], [1, 0], [2, 0], [3, 0]],
        i4q_support: None,
        initial_reference_signal: None,
        initial_optical_gain: None,
        normalization_policy: "mean-valid-flux".into(),
        incidence_flux: 0.0,
    })
    .unwrap();
    let mut image = vec![0.0; 4 * columns];
    for (index, value) in [(0, 16777216.0), (1, 1.0), (256, 1.0), (512, -16777216.0)] {
        image[index] = value;
    }
    let mut signal = vec![71.0; 2 * columns];
    let mut divisor = [71.0];
    P::process(&mut p, &image, &mut signal, &mut divisor).unwrap();
    assert_eq!(divisor, [1.0 / 513.0]);
    image.iter_mut().for_each(|v| *v *= 2.0);
    P::process(&mut p, &image, &mut signal, &mut divisor).unwrap();
    assert_eq!(divisor, [2.0 / 513.0]);
}
#[test]
fn pyramid_and_bio_have_distinct_epsilon_normalization_boundaries() {
    let mut pc = pyramid("mean-valid-flux");
    pc.i4q_support = None;
    let mut p = P::prepare(pc).unwrap();
    let mut bc = bio("mean-valid-flux");
    bc.bi_o_edge_support = None;
    let mut b = B::prepare(bc).unwrap();
    let input = [f32::EPSILON / 16.0; 20];
    let mut signal = [71.0; 8];
    let mut divisor = [71.0];
    P::process(&mut p, &input, &mut signal, &mut divisor).unwrap();
    assert_eq!(divisor, [f32::EPSILON / 4.0]);
    assert_eq!(signal, [0.0; 8]);
    signal.fill(71.0);
    divisor.fill(71.0);
    assert!(B::process(&mut b, &input, &[7], &mut signal, &mut divisor).is_err());
    assert_eq!(signal, [71.0; 8]);
    assert_eq!(divisor, [71.0]);
    let mut pc = pyramid("incidence-flux");
    pc.incidence_flux = f32::MIN_POSITIVE;
    assert!(P::prepare(pc).is_ok());
    let mut bc = bio("incidence-flux");
    bc.incidence_flux = f32::EPSILON;
    assert!(B::prepare(bc).is_err());
}
fn zernike(policy: &str) -> ZC {
    ZC {
        image_size: (2, 3),
        zernike_pupil_support: Some(vec![true, false, false, true, true, false]),
        initial_reference_pupil_image: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        normalization_policy: policy.into(),
        incidence_flux: 2.0,
        initial_calibration_signature: 7,
    }
}
#[test]
fn zernike_reference_geometry_low_flux_and_atomic_calibration() {
    let mut p = Z::prepare(zernike("mean-valid-flux")).unwrap();
    let mut signal = [71.0; 3];
    let mut divisor = [71.0];
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    Z::process(&mut p, &input, &[7], &mut signal, &mut divisor).unwrap();
    assert_eq!(divisor, [3.0]);
    assert_eq!(signal, [1.0 / 3.0 - 0.1, 5.0 / 3.0 - 0.4, 1.0 - 0.5]);
    let before = signal;
    let mut invalid = input;
    invalid[1] = f32::NAN;
    assert!(Z::process(&mut p, &invalid, &[7], &mut signal, &mut divisor).is_err());
    assert_eq!(signal, before);
    Z::process(&mut p, &[0.0; 6], &[7], &mut signal, &mut divisor).unwrap();
    assert_eq!(signal, [0.0; 3]);
    assert_eq!(divisor, [f32::EPSILON]);
    let bad_flux = [-1.0_f32];
    let new_signature = [u64::MAX];
    assert!(
        p.replace_parameters(&[
            (7, AlgorithmInput::from(new_signature.as_slice())),
            (6, AlgorithmInput::from(bad_flux.as_slice()))
        ])
        .is_err()
    );
    Z::process(&mut p, &input, &[7], &mut signal, &mut divisor).unwrap();
    assert_eq!(signal, before);
    let new_support = [false, true, true, true, false, false];
    let new_reference = [0.0_f32; 6];
    let new_flux = [0.0_f32];
    p.replace_parameters(&[
        (4, AlgorithmInput::from(new_support.as_slice())),
        (5, AlgorithmInput::from(new_reference.as_slice())),
        (6, AlgorithmInput::from(new_flux.as_slice())),
        (7, AlgorithmInput::from(new_signature.as_slice())),
    ])
    .unwrap();
    assert!(Z::process(&mut p, &input, &[7], &mut signal, &mut divisor).is_err());
    Z::process(&mut p, &input, &new_signature, &mut signal, &mut divisor).unwrap();
    p.reset();
    Z::process(&mut p, &input, &new_signature, &mut signal, &mut divisor).unwrap();
    assert!(signal.iter().all(|v| v.is_finite()));
    let mut c = zernike("incidence-flux");
    c.incidence_flux = f32::EPSILON;
    let mut p = Z::prepare(c).unwrap();
    Z::process(&mut p, &input, &[7], &mut signal, &mut divisor).unwrap();
    assert_eq!(signal, [0.0; 3]);
}
fn channel() -> CC {
    CC {
        input_branch_rows: 2,
        input_channels: 3,
        signal_size: (1, 3),
        positive_branch_row: 0,
        negative_branch_row: 1,
        channel_offsets: vec![0, 1, 2],
        curvature_support: None,
        initial_reference_signal: Some(vec![0.1, 0.2, 0.3]),
        initial_branch_scales: None,
        initial_calibration_signature: 7,
    }
}
#[test]
fn curvature_finite_overflow_scaling_low_total_and_ignored_reference() {
    let mut p = C::prepare(channel()).unwrap();
    let mut signal = [71.0; 3];
    C::process(
        &mut p,
        &[
            f32::MAX,
            f32::MAX,
            f32::MAX,
            f32::MAX,
            -f32::MAX,
            -f32::MAX * 0.5,
        ],
        &[7],
        &mut signal,
    )
    .unwrap();
    assert_eq!(signal, [-0.1, 0.0, 3.0 - 0.3]);
    let before = signal;
    assert!(C::process(&mut p, &[f32::NAN; 6], &[7], &mut signal).is_err());
    assert_eq!(signal, before);
    C::process(&mut p, &[-1.0; 6], &[7], &mut signal).unwrap();
    assert_eq!(signal, [0.0; 3]);
    let support = [false; 3];
    p.replace_parameter(3, AlgorithmInput::from(support.as_slice()))
        .unwrap()
        .unwrap();
    C::process(&mut p, &[1.0; 6], &[7], &mut signal).unwrap();
    assert_eq!(signal, [0.0; 3]);
}
#[test]
fn curvature_tiles_accumulate_column_then_row_and_full_grid_stays_column_major() {
    let c = IC {
        image_size: (4, 8),
        signal_size: (2, 2),
        branch_size: (4, 4),
        positive_branch_origin: [0, 0],
        negative_branch_origin: [0, 4],
        curvature_support: Some(vec![true, false, true, true]),
        initial_reference_signal: None,
        initial_branch_scales: None,
        initial_calibration_signature: 7,
    };
    let mut p = I::prepare(c).unwrap();
    let mut image = [1.0; 32];
    for row in 0..4 {
        for col in 0..4 {
            image[row * 8 + col] = 2.0;
        }
    }
    image[0] = 16777216.0;
    image[8] = -16777216.0;
    image[1] = 1.0;
    image[9] = 1.0;
    let mut signal = [71.0; 4];
    I::process(&mut p, &image, &[7], &mut signal).unwrap();
    assert_eq!(
        signal,
        [
            -2.0 / (6.0 + f32::EPSILON),
            0.0,
            4.0 / (12.0 + f32::EPSILON),
            4.0 / (12.0 + f32::EPSILON)
        ]
    );
    let before = signal;
    image[0] = f32::MAX;
    image[8] = f32::MAX;
    assert!(I::process(&mut p, &image, &[7], &mut signal).is_err());
    assert_eq!(signal, before);
}
#[test]
fn invalid_calibration_and_geometry_do_not_prepare() {
    let mut c = pyramid("unknown");
    assert!(P::prepare(c.clone()).is_err());
    c.normalization_policy = "mean-valid-flux".into();
    c.i4q_support = Some(vec![false; 4]);
    assert!(P::prepare(c).is_err());
    let mut c = bio("incidence-flux");
    c.pupil_origins[0] = [usize::MAX, 0];
    assert!(B::prepare(c).is_err());
    let mut c = zernike("incidence-flux");
    c.incidence_flux = -1.0;
    assert!(Z::prepare(c).is_err());
    let mut c = channel();
    c.channel_offsets[0] = 3;
    assert!(C::prepare(c).is_err());
    let mut c = channel();
    c.initial_branch_scales = Some(vec![0.0, 1.0]);
    assert!(C::prepare(c).is_err());
    let mut c = channel();
    c.signal_size = (0, 1);
    assert!(C::prepare(c).is_err());
}
