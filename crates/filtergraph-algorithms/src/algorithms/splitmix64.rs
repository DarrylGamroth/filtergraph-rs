//! Deterministic SplitMix64 sampling shared by admitted algorithms.

use num_traits::{Float, NumCast};

const SPLITMIX64_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

pub(crate) fn splitmix64_at(state: u64, draw_index: u64) -> u64 {
    let offset = SPLITMIX64_GAMMA.wrapping_mul(draw_index.wrapping_add(1));
    let mut value = state.wrapping_add(offset);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(crate) fn splitmix64_advance(state: u64, draw_count: u64) -> u64 {
    state.wrapping_add(SPLITMIX64_GAMMA.wrapping_mul(draw_count))
}

pub(crate) fn splitmix64_unit_at<T>(state: u64, draw_index: u64) -> Option<T>
where
    T: Float + NumCast,
{
    // The top 24 bits convert exactly to every admitted Rust scalar type.
    let value = splitmix64_at(state, draw_index);
    let numerator = <T as NumCast>::from(value >> 40)?;
    let denominator = <T as NumCast>::from(1_u32 << 24)?;
    Some(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::{splitmix64_advance, splitmix64_at};

    #[test]
    fn indexed_draws_and_fixed_advance_match_the_frozen_sequence() {
        let state = 0x1234_5678_9abc_def0;
        let expected = [
            0x1619_22c6_45ce_50e8,
            0xad76_0caf_a169_7b60,
            0x3501_ff44_902c_a50d,
            0x417c_b9a8_26d8_31df,
            0x99af_6f9b_0c44_76b6,
            0x5d51_f5f7_5b76_2c59,
        ];

        for (draw_index, expected) in expected.into_iter().enumerate().rev() {
            assert_eq!(splitmix64_at(state, draw_index as u64), expected);
        }
        assert_eq!(splitmix64_advance(state, 6), 0xc781_30d1_967b_c76e);
    }
}
