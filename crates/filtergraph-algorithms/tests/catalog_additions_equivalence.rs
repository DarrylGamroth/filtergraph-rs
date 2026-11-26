//! Scientific contracts for the four current JFG catalog additions.
use filtergraph_algorithms::*;
fn stats(mode: &str, n: usize, window: usize) -> ArrayStatisticsPlan {
    ArrayStatisticsPlan::new(vec![n], mode, window, 2.5, 0.2, 268435456).unwrap()
}
#[test]
fn physical_factor_order_atomic_replacement_and_output() {
    let p = PhysicalOffsetReferencePlan::new(3, 2, 2, -1.0)
        .unwrap()
        .with_physical_to_modes(&[1.0, 2.0, -0.5, 3.0])
        .unwrap()
        .with_modes_to_reference(&[1.0, -2.0, 0.5, 1.0, -3.0, 0.25])
        .unwrap()
        .with_offset(&[0.25, -0.5])
        .unwrap();
    // Q*offset = [-0.75,-1.625], -C*(Q*offset) = [-2.5,2, -1.84375].
    assert_eq!(p.correction(), &[-2.5, 2.0, -1.84375]);
    let mut w = p.workspace();
    let (mut y, mut offset) = ([71.0; 3], [71.0; 2]);
    p.process((&mut y, &mut offset), &mut w, &[1.0, -2.0, 0.125])
        .unwrap();
    assert_eq!(y, [-1.5, 0.0, -1.71875]);
    assert_eq!(offset, [0.25, -0.5]);
    assert!(p.with_offset(&[f32::MAX; 2]).is_err());
    assert!(p.with_modes_to_reference(&[f32::NAN; 6]).is_err());
    assert!(p.with_offset(&[0.0]).is_err());
    let old = (y, offset);
    assert!(
        p.process((&mut y, &mut offset), &mut w, &[1.0, f32::NAN, 3.0])
            .is_err()
    );
    assert_eq!((y, offset), old);
    w.reset();
    p.process((&mut y, &mut offset), &mut w, &[1.0, -2.0, 0.125])
        .unwrap();
    assert_eq!((y, offset), old);
    let p = PhysicalOffsetReferencePlan::new(1, 1, 1, 1.0)
        .unwrap()
        .with_physical_to_modes(&[1.0])
        .unwrap()
        .with_modes_to_reference(&[1.0])
        .unwrap()
        .with_offset(&[f32::MAX])
        .unwrap();
    let mut w = p.workspace();
    let (mut y, mut o) = ([71.0], [72.0]);
    assert_eq!(
        p.process((&mut y, &mut o), &mut w, &[f32::MAX]),
        Err(VectorLinearCombinationError::NonFiniteOutput)
    );
    assert_eq!((y, o), ([71.0], [72.0]));
}
#[test]
fn statistics_finite_sliding_centered_batch_and_freeze() {
    for mode in ["finite", "sliding"] {
        for window in [1, 2, 4, 7] {
            let p = stats(mode, 2, window);
            let mut w = p.workspace();
            let (mut mean, mut var, mut sd) = ([0.0; 2], [0.0; 2], [0.0; 2]);
            let (mut count, mut complete, mut valid) = ([0], [false], [false]);
            let mut history = Vec::new();
            for k in 0..37 {
                let sample = [
                    1e6 + 0.125 * ((k * 7) % 13) as f32,
                    if k > 30 {
                        0.0
                    } else {
                        1e8 + 8.0 * ((k * 19) % 31) as f32
                    },
                ];
                history.push(sample);
                p.process(
                    (
                        &mut mean,
                        &mut var,
                        &mut sd,
                        &mut count,
                        &mut complete,
                        &mut valid,
                    ),
                    &mut w,
                    &sample,
                )
                .unwrap();
                let end = if mode == "finite" {
                    (k + 1).min(window)
                } else {
                    k + 1
                };
                let start = if mode == "finite" {
                    0
                } else {
                    end.saturating_sub(window)
                };
                let n = end - start;
                for i in 0..2 {
                    let mu =
                        history[start..end].iter().map(|x| x[i] as f64).sum::<f64>() / n as f64;
                    let v = if n > 1 {
                        history[start..end]
                            .iter()
                            .map(|x| (x[i] as f64 - mu).powi(2))
                            .sum::<f64>()
                            / (n - 1) as f64
                    } else {
                        0.0
                    };
                    assert_eq!(mean[i], mu as f32);
                    assert!((var[i] as f64 - v).abs() <= v.abs() * 2e-6 + 1e-8);
                    assert_eq!(sd[i], v.sqrt() as f32);
                }
                assert_eq!(
                    count[0],
                    if mode == "finite" {
                        end as u64
                    } else {
                        (k + 1) as u64
                    }
                );
                assert_eq!(valid, [n >= 2]);
                assert_eq!(complete, [mode == "finite" && end == window]);
            }
            if mode == "finite" {
                let old = (mean, var, sd, count, complete, valid);
                p.process(
                    (
                        &mut mean,
                        &mut var,
                        &mut sd,
                        &mut count,
                        &mut complete,
                        &mut valid,
                    ),
                    &mut w,
                    &[f32::NAN; 2],
                )
                .unwrap();
                assert_eq!((mean, var, sd, count, complete, valid), old);
            }
        }
    }
}
#[test]
fn statistics_exponential_weighted_oracle_and_candidate_lifecycle() {
    let p = stats("exponential", 1, 3);
    let mut w = p.workspace();
    let (mut mean, mut var, mut sd) = ([0.0], [0.0], [0.0]);
    let (mut count, mut complete, mut valid) = ([0], [false], [false]);
    let alpha = -(-0.2_f64 / 2.5).exp_m1();
    let mut history = Vec::new();
    for k in 0..20 {
        let x = (10.0 + (k as f32).sin()) as f32;
        history.push(x);
        p.process(
            (
                &mut mean,
                &mut var,
                &mut sd,
                &mut count,
                &mut complete,
                &mut valid,
            ),
            &mut w,
            &[x],
        )
        .unwrap();
        let weights: Vec<_> = (0..=k)
            .map(|j| {
                if j == 0 {
                    (1.0 - alpha).powi(k as i32)
                } else {
                    alpha * (1.0 - alpha).powi((k - j) as i32)
                }
            })
            .collect();
        let mu: f64 = history
            .iter()
            .zip(&weights)
            .map(|(&x, &a)| x as f64 * a)
            .sum();
        let variance: f64 = history
            .iter()
            .zip(&weights)
            .map(|(&x, &a)| a * (x as f64 - mu).powi(2))
            .sum();
        assert!((mean[0] as f64 - mu).abs() < 2e-6);
        assert!((var[0] as f64 - variance).abs() < 2e-6);
        assert_eq!(valid, [true]);
        assert_eq!(complete, [false]);
    }
    for mode in ["finite", "sliding", "exponential"] {
        let p = stats(mode, 1, 3);
        let mut w = p.workspace();
        p.stage(
            (
                &mut mean,
                &mut var,
                &mut sd,
                &mut count,
                &mut complete,
                &mut valid,
            ),
            &mut w,
            &[5.0],
        )
        .unwrap();
        assert_eq!(w.sample_count(), 0);
        w.discard();
        w.commit();
        assert_eq!(w.sample_count(), 0);
        p.stage(
            (
                &mut mean,
                &mut var,
                &mut sd,
                &mut count,
                &mut complete,
                &mut valid,
            ),
            &mut w,
            &[7.0],
        )
        .unwrap();
        w.commit();
        assert_eq!(w.sample_count(), 1);
        let old = (mean, var, sd, count, complete, valid);
        assert_eq!(
            p.process(
                (
                    &mut mean,
                    &mut var,
                    &mut sd,
                    &mut count,
                    &mut complete,
                    &mut valid
                ),
                &mut w,
                &[f32::MAX]
            ),
            Err(ArrayStatisticsError::NonFiniteOutput)
        );
        assert_eq!((mean, var, sd, count, complete, valid), old);
        assert_eq!(w.sample_count(), 1);
        w.reset();
        assert_eq!(w.sample_count(), 0);
    }
}
#[test]
fn statistics_reject_configuration_before_allocation() {
    for (size, mode, window, tau, period, bytes) in [
        (vec![], "finite", 1, 1.0, 1.0, 100),
        (vec![0], "finite", 1, 1.0, 1.0, 100),
        (vec![usize::MAX, 3], "finite", 1, 1.0, 1.0, 100),
        (vec![1], "other", 1, 1.0, 1.0, 100),
        (vec![1], "finite", 0, 1.0, 1.0, 100),
        (vec![1], "finite", 1, f64::MAX, f64::from_bits(1), 100),
        (vec![1], "finite", 1, 1.0, 1.0, 51),
        (vec![1], "sliding", 2, 1.0, 1.0, 75),
        (vec![1], "sliding", usize::MAX, 1.0, 1.0, usize::MAX),
    ] {
        assert!(ArrayStatisticsPlan::new(size, mode, window, tau, period, bytes).is_err());
    }
    assert!(ArrayStatisticsPlan::new(vec![1], "finite", 1, 1.0, 1.0, 52).is_ok());
    assert!(ArrayStatisticsPlan::new(vec![1], "sliding", 2, 1.0, 1.0, 76).is_ok());
}
#[test]
fn pixel_validity_finite_only_and_gated_feedback_hold() {
    let mut p = PixelValidityF32::prepare(PixelValidityF32Configuration {
        extent: 7,
        input_schema: "test.pixels/1".into(),
    })
    .unwrap();
    let mut validity = [false; 7];
    PixelValidityF32::process(
        &mut p,
        &[
            0.0,
            -0.0,
            f32::from_bits(1),
            f32::MAX,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
        &mut validity,
    )
    .unwrap();
    assert_eq!(validity, [true, true, true, true, false, false, false]);
    let mut p = ValidityGatedClosedLoopIntegratorF32::prepare(
        ValidityGatedClosedLoopIntegratorF32Configuration {
            state_length: 1,
            validity_length: 2,
            validity_schema: "test.validity/1".into(),
        },
    )
    .unwrap();
    p.replace_properties(&[
        PropertyAssignment {
            id: 0,
            value: PropertyValue::Float(0.5),
        },
        PropertyAssignment {
            id: 1,
            value: PropertyValue::Float(0.75),
        },
        PropertyAssignment {
            id: 2,
            value: PropertyValue::Float(0.25),
        },
    ])
    .unwrap();
    let (mut y, mut state) = ([71.0], [72.0]);
    ValidityGatedClosedLoopIntegratorF32::process(
        &mut p,
        &[f32::NAN],
        &[f32::NAN],
        &[false, true],
        &mut y,
        &mut state,
    )
    .unwrap();
    assert_eq!((y, state), ([0.0], [0.0]));
    ValidityGatedClosedLoopIntegratorF32::process(
        &mut p,
        &[4.0],
        &[2.0],
        &[true, true],
        &mut y,
        &mut state,
    )
    .unwrap();
    assert_eq!((y, state), ([1.625], [-0.5]));
    for _ in 0..2 {
        ValidityGatedClosedLoopIntegratorF32::process(
            &mut p,
            &[f32::NAN],
            &[f32::NAN],
            &[true, false],
            &mut y,
            &mut state,
        )
        .unwrap();
        assert_eq!((y, state), ([1.625], [-0.5]));
    }
    // Only the immediately preceding external delay value is supplied and consumed.
    ValidityGatedClosedLoopIntegratorF32::process(
        &mut p,
        &[8.0],
        &[4.0],
        &[true, true],
        &mut y,
        &mut state,
    )
    .unwrap();
    assert_eq!((y, state), ([4.46875], [0.625]));
    p.reset();
    ValidityGatedClosedLoopIntegratorF32::process(
        &mut p,
        &[4.0],
        &[f32::NAN],
        &[true, true],
        &mut y,
        &mut state,
    )
    .unwrap();
    assert_eq!((y, state), ([2.0], [0.0]));
}
#[test]
fn statistics_dynamic_rank_defaults_and_scientific_port_contract() {
    let c: ArrayStatisticsF32Configuration = toml::from_str(
        r#"
 size=[2,3,4]
 mode="finite"
 input_schema="test.sample/1"
 mean_schema="test.mean/1"
 variance_schema="test.variance/1"
 stddev_schema="test.stddev/1"
 "#,
    )
    .unwrap();
    assert_eq!(c.window_samples, 1);
    assert_eq!(c.time_constant_s, 1.0);
    assert_eq!(c.sample_period_s, 1.0);
    assert_eq!(c.max_state_bytes, 268435456);
    let fields = ArrayStatisticsF32::CONFIGURATION;
    assert_eq!(
        fields
            .iter()
            .filter(|f| !f.required())
            .map(|f| f.name())
            .collect::<Vec<_>>(),
        vec![
            "window_samples",
            "time_constant_s",
            "sample_period_s",
            "max_state_bytes"
        ]
    );
    let p = ArrayStatisticsF32::prepare(c).unwrap();
    for format in &p.formats()[..4] {
        assert_eq!(format.shape(), &[2, 3, 4]);
    }
    for format in &p.formats()[4..] {
        assert_eq!(format.shape(), &[1]);
    }
    assert_eq!(p.formats()[1].schema(), "test.mean/1");
    assert_eq!(p.formats()[2].schema(), "test.variance/1");
    assert!(
        ArrayStatisticsF32::PORTS[1..]
            .iter()
            .all(|p| p.metadata_source() == Some("sample"))
    );
}
