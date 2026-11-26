//! Hidden-to-shape projections consisting of vertically stacked diagonal blocks.

use alloc::sync::Arc;

use crate::MatrixElement;

// Large correction scaling workloads have 16 hidden modes and 277 or 4096 coordinates.
const MODES: usize = 16;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RepeatedDiagonal<T> {
    weights: Arc<[T]>,
    min_abs: f64,
    max_abs: f64,
}

impl<T: MatrixElement> RepeatedDiagonal<T> {
    pub(super) fn prepare(rows: usize, columns: usize, matrix: &[T]) -> Option<Self> {
        if size_of::<T>() != size_of::<f32>() || rows < 256 || columns != MODES {
            return None;
        }
        let mut min_abs = f64::INFINITY;
        let mut max_abs = 0.0_f64;
        for (r, row) in matrix.chunks_exact(MODES).enumerate() {
            for (c, &weight) in row.iter().enumerate() {
                // MatrixElement is sealed to f32 and f64; here weight is f32.
                let w = weight.to_f32()?;
                if c == r % MODES {
                    if !w.is_normal() {
                        return None;
                    }
                    let magnitude = f64::from(w.abs());
                    min_abs = min_abs.min(magnitude);
                    max_abs = max_abs.max(magnitude);
                } else if w.to_bits() != 0 {
                    return None;
                }
            }
        }
        let weights = matrix
            .chunks_exact(MODES)
            .enumerate()
            .map(|(r, row)| row[r % MODES])
            .collect();
        Some(Self {
            weights,
            min_abs,
            max_abs,
        })
    }

    pub(super) fn project(&self, hidden: &[T], shape: &mut [T]) -> bool {
        if !filtergraph_blas::sgemv_default_float_mode() {
            return false;
        }
        for &sample in hidden {
            // Exact binary64 products bound every f32 product, covering both
            // overflow and underflow. Unordered comparisons also reject NaNs.
            let magnitude = sample.to_f64().unwrap().abs();
            if magnitude != 0.0
                && !(magnitude * self.min_abs >= f64::from(f32::MIN_POSITIVE)
                    && magnitude * self.max_abs <= f64::from(f32::MAX))
            {
                return false;
            }
        }
        // Each dense dot has precisely one nonzero coefficient. Addition to
        // +0 reproduces the sign of an all-zero dot even with a signed input.
        let hidden: &[T; MODES] = hidden.try_into().unwrap();
        let (weights, tail_weights) = self.weights.as_chunks::<MODES>();
        let (blocks, tail) = shape.as_chunks_mut::<MODES>();
        for (block, weights) in blocks.iter_mut().zip(weights) {
            for i in 0..MODES {
                block[i] = weights[i] * hidden[i] + T::zero();
            }
        }
        for ((y, &weight), &sample) in tail.iter_mut().zip(tail_weights).zip(hidden) {
            *y = weight * sample + T::zero();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn stacked_diagonal_matches_dense_bits_and_defers_extremes() {
        let n = 277;
        let mut dense = vec![0.0_f32; n * MODES];
        for (r, row) in dense.chunks_exact_mut(MODES).enumerate() {
            row[r % MODES] = if r % 3 == 0 { -0.61 } else { 7.03 };
        }
        let prepared = RepeatedDiagonal::prepare(n, MODES, &dense).unwrap();
        let mut hidden = core::array::from_fn::<_, MODES, _>(|j| (j as f32 - 7.0) * 0.37);
        hidden[11] = -0.0;
        let mut got = vec![0.0; n];
        let mut expected = vec![0.0; n];
        assert!(prepared.project(&hidden, &mut got));
        filtergraph_blas::sgemv(n, MODES, &dense, &hidden, &mut expected);
        assert!(
            got.iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        for value in [f32::MAX, f32::from_bits(1), f32::INFINITY, f32::NAN] {
            hidden[3] = value;
            assert!(!prepared.project(&hidden, &mut got));
        }
        let plan = super::super::ClosedLoopCorrectionPlan::with_hidden_modes(
            0.5,
            0.8,
            0.1,
            n,
            MODES,
            0.01,
            Arc::from(dense.clone()),
            Arc::from(dense.clone()),
        )
        .unwrap();
        dense[1] = -0.0;
        assert_eq!(
            plan,
            plan.with_hidden_to_shape(Arc::from(dense.clone())).unwrap()
        );
        for v in [-0.0, f32::from_bits(1), 0.25] {
            dense[1] = v;
            assert!(RepeatedDiagonal::prepare(n, MODES, &dense).is_none());
        }
    }
}
