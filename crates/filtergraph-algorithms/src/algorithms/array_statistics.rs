//! Online array statistics using the JFG compensated Float64 recurrences.
use crate::AlgorithmPlan;
use alloc::{string::String, vec, vec::Vec};
use core::fmt;
#[cfg(not(feature = "blas"))]
use num_traits::Float as _;

/// Invalid statistics configuration, shape, or rejected numerical result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArrayStatisticsError {
    /// Invalid shape, mode, positive parameter, or memory bound.
    Configuration,
    /// Caller storage differs from prepared extents.
    DimensionMismatch,
    /// An accepted sample contains a nonfinite value.
    NonFiniteInput,
    /// A statistic is not representable as finite Float32 or variance is negative.
    NonFiniteOutput,
    /// Accepted-sample count cannot be incremented.
    SampleCountOverflow,
}
impl fmt::Display for ArrayStatisticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "array statistics: {:?}", self)
    }
}
impl core::error::Error for ArrayStatisticsError {}
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Finite,
    Sliding,
    Exponential,
}
/// Prepared shape, mode, window, and exponential weight; no mutable history.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayStatisticsPlan {
    size: Vec<usize>,
    mode: Mode,
    window_samples: usize,
    alpha: f64,
    elements: usize,
}
#[derive(Clone, Debug, PartialEq)]
struct State {
    sample_count: u64,
    sum: Vec<f64>,
    correction: Vec<f64>,
    m2: Vec<f64>,
}
impl State {
    fn new(n: usize) -> Self {
        Self {
            sample_count: 0,
            sum: vec![0.0; n],
            correction: vec![0.0; n],
            m2: vec![0.0; n],
        }
    }
    fn reset(&mut self) {
        self.sample_count = 0;
        self.sum.fill(0.0);
        self.correction.fill(0.0);
        self.m2.fill(0.0);
    }
}
/// Persistent committed/candidate statistics, ring history, and disposable scratch.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayStatisticsWorkspace {
    committed: State,
    candidate: State,
    history: Vec<f32>,
    sample: Vec<f32>,
    mean: Vec<f64>,
    variance_correction: Vec<f64>,
    head: usize,
    pending: bool,
}
impl ArrayStatisticsWorkspace {
    /// Adopt only the most recently staged successful sample.
    pub fn commit(&mut self) {
        if !self.pending {
            return;
        }
        if !self.history.is_empty() {
            let n = self.sample.len();
            self.history[self.head * n..(self.head + 1) * n].copy_from_slice(&self.sample);
            self.head = (self.head + 1) % (self.history.len() / n);
        }
        core::mem::swap(&mut self.committed, &mut self.candidate);
        self.pending = false;
    }
    /// Retain committed history and discard the current candidate.
    pub fn discard(&mut self) {
        self.pending = false;
    }
    /// Clear mathematical history, ring, and disposable scratch.
    pub fn reset(&mut self) {
        self.committed.reset();
        self.candidate.reset();
        self.history.fill(0.0);
        self.sample.fill(0.0);
        self.mean.fill(0.0);
        self.variance_correction.fill(0.0);
        self.head = 0;
        self.pending = false;
    }
    /// Number of committed accepted samples since reset.
    pub fn sample_count(&self) -> u64 {
        self.committed.sample_count
    }
}
/// Ordered array and scalar outputs for one statistics sample.
pub type ArrayStatisticsOutput<'a> = (
    &'a mut [f32],
    &'a mut [f32],
    &'a mut [f32],
    &'a mut [u64],
    &'a mut [bool],
    &'a mut [bool],
);
fn add(total: f64, correction: f64, x: f64) -> (f64, f64) {
    let next = total + x;
    let error = if total.abs() >= x.abs() {
        (total - next) + x
    } else {
        (x - next) + total
    };
    (next, correction + error)
}
impl ArrayStatisticsPlan {
    /// Prepare exact owned-state memory and byte-extent bounds before allocation.
    pub fn new(
        size: Vec<usize>,
        mode: &str,
        window_samples: usize,
        time_constant_s: f64,
        sample_period_s: f64,
        max_state_bytes: usize,
    ) -> Result<Self, ArrayStatisticsError> {
        let e = ArrayStatisticsError::Configuration;
        if size.is_empty()
            || size.contains(&0)
            || window_samples == 0
            || max_state_bytes == 0
            || !time_constant_s.is_finite()
            || time_constant_s <= 0.0
            || !sample_period_s.is_finite()
            || sample_period_s <= 0.0
        {
            return Err(e);
        }
        let elements = size
            .iter()
            .try_fold(1usize, |n, &d| {
                n.checked_mul(d).filter(|&n| n <= u32::MAX as usize / 4)
            })
            .ok_or(e)?;
        let mode = match mode {
            "finite" => Mode::Finite,
            "sliding" => Mode::Sliding,
            "exponential" => Mode::Exponential,
            _ => return Err(e),
        };
        let history = if mode == Mode::Sliding {
            window_samples
        } else {
            0
        };
        if elements
            .checked_mul(history)
            .is_none_or(|n| n > u32::MAX as usize / 4)
        {
            return Err(e);
        }
        let arrays = if mode == Mode::Sliding {
            8usize
        } else {
            6usize
        };
        let bytes = (elements as u128) * (arrays as u128 * 8 + 4 + history as u128 * 4);
        if bytes > max_state_bytes as u128 {
            return Err(e);
        }
        let alpha = -(-sample_period_s / time_constant_s).exp_m1();
        if !(alpha > 0.0 && alpha <= 1.0) {
            return Err(e);
        }
        Ok(Self {
            size,
            mode,
            window_samples,
            alpha,
            elements,
        })
    }
    /// Allocate fixed state; sliding storage is O(elements × window_samples).
    pub fn workspace(&self) -> ArrayStatisticsWorkspace {
        let sliding = self.mode == Mode::Sliding;
        ArrayStatisticsWorkspace {
            committed: State::new(self.elements),
            candidate: State::new(self.elements),
            history: vec![
                0.0;
                if sliding {
                    self.elements * self.window_samples
                } else {
                    0
                }
            ],
            sample: vec![0.0; self.elements],
            mean: vec![0.0; if sliding { self.elements } else { 0 }],
            variance_correction: vec![0.0; if sliding { self.elements } else { 0 }],
            head: 0,
            pending: false,
        }
    }
    /// Configured logical dimensions, retained on every array product.
    pub fn size(&self) -> &[usize] {
        &self.size
    }
    /// Stage an atomic candidate; explicit commit/discard determines history adoption.
    /// Finite/exponential processing is O(P); sliding processing is O(PW).
    pub fn stage(
        &self,
        (mean, variance, stddev, sample_count, complete, variance_valid): ArrayStatisticsOutput<'_>,
        w: &mut ArrayStatisticsWorkspace,
        sample: &[f32],
    ) -> Result<(), ArrayStatisticsError> {
        use ArrayStatisticsError as E;
        if [
            mean.len(),
            variance.len(),
            stddev.len(),
            sample.len(),
            w.sample.len(),
            w.committed.sum.len(),
            w.candidate.sum.len(),
        ]
        .iter()
        .any(|&n| n != self.elements)
            || sample_count.len() != 1
            || complete.len() != 1
            || variance_valid.len() != 1
            || w.history.len()
                != if self.mode == Mode::Sliding {
                    self.elements * self.window_samples
                } else {
                    0
                }
        {
            return Err(E::DimensionMismatch);
        }
        w.pending = false;
        let frozen =
            self.mode == Mode::Finite && w.committed.sample_count == self.window_samples as u64;
        let mut count = w.committed.sample_count;
        if !frozen {
            if sample.iter().any(|v| !v.is_finite()) {
                return Err(E::NonFiniteInput);
            }
            count = count.checked_add(1).ok_or(E::SampleCountOverflow)?;
            w.sample.copy_from_slice(sample);
            let (c, k) = (&w.committed, &mut w.candidate);
            if count == 1 {
                for (i, &x) in sample.iter().enumerate() {
                    k.sum[i] = x as f64;
                    k.correction[i] = 0.0;
                    k.m2[i] = 0.0;
                }
            } else if self.mode == Mode::Finite {
                let n = count as f64;
                let old = (count - 1) as f64;
                for (i, &x) in sample.iter().enumerate() {
                    let x = x as f64;
                    let (total, correction) = add(c.sum[i], c.correction[i], x);
                    let next_mean = total / n + correction / n;
                    let old_mean = c.sum[i] / old + c.correction[i] / old;
                    k.sum[i] = total;
                    k.correction[i] = correction;
                    k.m2[i] = num_traits::Float::mul_add(x - old_mean, x - next_mean, c.m2[i]);
                }
            } else if self.mode == Mode::Exponential {
                let alpha = self.alpha;
                let beta = 1.0 - alpha;
                for (i, &x) in sample.iter().enumerate() {
                    let x = x as f64;
                    let scaled = beta * c.sum[i];
                    let mut correction = num_traits::Float::mul_add(
                        beta,
                        c.correction[i],
                        num_traits::Float::mul_add(beta, c.sum[i], -scaled),
                    );
                    let term = alpha * x;
                    correction += num_traits::Float::mul_add(alpha, x, -term);
                    let (total, correction) = add(scaled, correction, term);
                    let delta = x - (c.sum[i] + c.correction[i]);
                    k.sum[i] = total;
                    k.correction[i] = correction;
                    k.m2[i] = beta * num_traits::Float::mul_add(alpha * delta, delta, c.m2[i]);
                }
            } else {
                let retained = c.sample_count.min(self.window_samples as u64) as usize;
                let n = count.min(self.window_samples as u64) as f64;
                let full = c.sample_count >= self.window_samples as u64;
                k.sum.fill(0.0);
                k.correction.fill(0.0);
                for column in 0..retained {
                    if full && column == w.head {
                        continue;
                    }
                    for i in 0..self.elements {
                        (k.sum[i], k.correction[i]) = add(
                            k.sum[i],
                            k.correction[i],
                            w.history[column * self.elements + i] as f64,
                        );
                    }
                }
                for (i, &x) in sample.iter().enumerate() {
                    let (total, correction) = add(k.sum[i], k.correction[i], x as f64);
                    k.sum[i] = total;
                    k.correction[i] = correction;
                    w.mean[i] = total / n + correction / n;
                    k.m2[i] = 0.0;
                    w.variance_correction[i] = 0.0;
                }
                for column in 0..retained {
                    if full && column == w.head {
                        continue;
                    }
                    for i in 0..self.elements {
                        let delta = w.history[column * self.elements + i] as f64 - w.mean[i];
                        (k.m2[i], w.variance_correction[i]) =
                            add(k.m2[i], w.variance_correction[i], delta * delta);
                    }
                }
                for (i, &x) in sample.iter().enumerate() {
                    let delta = x as f64 - w.mean[i];
                    let (m2, correction) = add(k.m2[i], w.variance_correction[i], delta * delta);
                    k.m2[i] = m2 + correction;
                }
            }
        }
        let selected = if frozen { &w.committed } else { &w.candidate };
        let n = if self.mode == Mode::Sliding {
            count.min(self.window_samples as u64)
        } else {
            count
        };
        let valid = if self.mode == Mode::Exponential {
            count >= 1
        } else {
            n >= 2
        };
        let mean_divisor = if self.mode == Mode::Exponential {
            1.0
        } else {
            n as f64
        };
        let divisor = if self.mode == Mode::Exponential {
            1.0
        } else {
            (n.max(2) - 1) as f64
        };
        for i in 0..self.elements {
            let v = if valid { selected.m2[i] / divisor } else { 0.0 };
            let mu = selected.sum[i] / mean_divisor + selected.correction[i] / mean_divisor;
            if !(mu as f32).is_finite() || !(v as f32).is_finite() || v < 0.0 {
                return Err(E::NonFiniteOutput);
            }
        }
        for i in 0..self.elements {
            let v = if valid { selected.m2[i] / divisor } else { 0.0 };
            mean[i] =
                (selected.sum[i] / mean_divisor + selected.correction[i] / mean_divisor) as f32;
            variance[i] = v as f32;
            stddev[i] = v.sqrt() as f32;
        }
        sample_count[0] = count;
        complete[0] = self.mode == Mode::Finite && count == self.window_samples as u64;
        variance_valid[0] = valid;
        if !frozen {
            w.candidate.sample_count = count;
            w.pending = true;
        }
        Ok(())
    }
}
impl AlgorithmPlan for ArrayStatisticsPlan {
    type Input<'a> = &'a [f32];
    type Output<'a> = ArrayStatisticsOutput<'a>;
    type Workspace = ArrayStatisticsWorkspace;
    type Error = ArrayStatisticsError;
    fn process<'i, 'o>(
        &self,
        output: Self::Output<'o>,
        w: &mut Self::Workspace,
        sample: Self::Input<'i>,
    ) -> Result<(), Self::Error> {
        self.stage(output, w, sample)?;
        w.commit();
        Ok(())
    }
}
crate::filtergraph_algorithm! {
    /// Online per-element statistics preserving the configured array shape and units.
    pub ArrayStatisticsF32 {
        label="array-statistics-f32";
        configuration ArrayStatisticsF32Configuration {
            size: Vec<usize> = GraphRebuild;
            mode: String = GraphRebuild;
            window_samples: usize = GraphRebuild, default=1;
            time_constant_s: f64 = GraphRebuild, default=1.0;
            sample_period_s: f64 = GraphRebuild, default=1.0;
            max_state_bytes: usize = GraphRebuild, default=268435456;
            input_schema: String = GraphRebuild;
            mean_schema: String = GraphRebuild;
            variance_schema: String = GraphRebuild;
            stddev_schema: String = GraphRebuild;
        }
        plan=ArrayStatisticsPlan;properties=none;
        dimensions|plan,_workspace|{size=plan.size();}
        inputs{sample:f32[..size]="sample",schema=configuration.input_schema.as_str();}
        outputs{
            mean:f32[..size]="mean",schema=configuration.mean_schema.as_str(),metadata_from="sample";
            variance:f32[..size]="variance",schema=configuration.variance_schema.as_str(),metadata_from="sample";
            stddev:f32[..size]="stddev",schema=configuration.stddev_schema.as_str(),metadata_from="sample";
            sample_count:u64[1]="sample-count",schema="pipewireao/array-statistics-sample-count-u64/1",metadata_from="sample";
            complete:bool[1]="complete",schema="pipewireao/array-statistics-complete-bool/1",metadata_from="sample";
            variance_valid:bool[1]="variance-valid",schema="pipewireao/array-statistics-variance-valid-bool/1",metadata_from="sample";
        }
        prepare|configuration|{let plan=ArrayStatisticsPlan::new(configuration.size,&configuration.mode,configuration.window_samples,configuration.time_constant_s,configuration.sample_period_s,configuration.max_state_bytes)?;let workspace=plan.workspace();Ok((plan,workspace))}
        prepare_error=ArrayStatisticsError;
        process|plan,workspace|{plan.process((mean,variance,stddev,sample_count,complete,variance_valid),workspace,sample)}
        process_error=ArrayStatisticsError;
        reset|_plan,workspace|{workspace.reset();}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sample_count_overflow_and_rejected_candidate_preserve_committed_history() {
        for mode in ["finite", "sliding", "exponential"] {
            let plan = ArrayStatisticsPlan::new(vec![1], mode, 3, 1.0, 1.0, 268435456).unwrap();
            let mut workspace = plan.workspace();
            let (mut mean, mut variance, mut stddev) = ([71.0], [72.0], [73.0]);
            let (mut count, mut complete, mut valid) = ([74], [false], [false]);
            workspace.committed.sample_count = u64::MAX;
            assert_eq!(
                plan.stage(
                    (
                        &mut mean,
                        &mut variance,
                        &mut stddev,
                        &mut count,
                        &mut complete,
                        &mut valid
                    ),
                    &mut workspace,
                    &[1.0]
                ),
                Err(ArrayStatisticsError::SampleCountOverflow)
            );
            assert_eq!(
                (mean, variance, stddev, count, complete, valid),
                ([71.0], [72.0], [73.0], [74], [false], [false])
            );
            workspace.commit();
            assert_eq!(workspace.sample_count(), u64::MAX);
            workspace.reset();
            plan.stage(
                (
                    &mut mean,
                    &mut variance,
                    &mut stddev,
                    &mut count,
                    &mut complete,
                    &mut valid,
                ),
                &mut workspace,
                &[1.0],
            )
            .unwrap();
            assert!(
                plan.stage(
                    (
                        &mut mean,
                        &mut variance,
                        &mut stddev,
                        &mut count,
                        &mut complete,
                        &mut valid
                    ),
                    &mut workspace,
                    &[f32::NAN]
                )
                .is_err()
            );
            workspace.commit();
            assert_eq!(workspace.sample_count(), 0);
        }
    }
}
