//! Packed Float32 operations whose floating-point status must remain observable.

use core::arch::{asm, x86_64::*};

/// Float32 lanes in the native packed subtraction.
pub const LANES: usize = if cfg!(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
)) {
    16
} else {
    8
};

/// Whether the current thread masks every SIMD floating-point exception.
#[inline]
pub fn exceptions_masked() -> bool {
    const EXCEPTION_MASKS: u32 = 0x1f80;
    let mut mxcsr = 0_u32;
    // SAFETY: The enabled SIMD target supports STMXCSR, which writes one u32.
    unsafe { asm!("stmxcsr [{0}]", in(reg) &mut mxcsr, options(nostack)) };
    mxcsr & EXCEPTION_MASKS == EXCEPTION_MASKS
}

/// Subtract the reference from each finite, valid sample. Write positive zero
/// and false for inactive, nonfinite or overflowing lanes. Return the valid count.
/// Callers with unmasked exceptions must use scalar operations if store order
/// before a trap is observable.
#[cfg(not(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
)))]
#[inline]
pub fn subtract_finite(
    output: &mut [f32; LANES],
    output_validity: &mut [bool; LANES],
    difference: &[f32; LANES],
    validity: &[bool; LANES],
    reference: &[f32; LANES],
) -> usize {
    // SAFETY: AVX2 is enabled. Unaligned loads/stores stay within the arrays.
    // The flag store writes exactly eight one-byte Bool representations (0/1).
    unsafe {
        let samples = _mm256_loadu_ps(difference.as_ptr());
        let reference = _mm256_loadu_ps(reference.as_ptr());
        let input_flags = _mm256_cvtepu8_epi32(_mm_loadl_epi64(validity.as_ptr().cast()));
        let masked = _mm256_cmpeq_epi32(input_flags, _mm256_setzero_si256());
        let inactive = _mm256_or_si256(masked, nonfinite_lanes(samples));
        let values = subtract_active(samples, reference, inactive);
        let invalid = _mm256_or_si256(inactive, nonfinite_lanes(values));
        _mm256_storeu_ps(
            output.as_mut_ptr(),
            _mm256_andnot_ps(_mm256_castsi256_ps(invalid), values),
        );
        write_validity(output_validity, invalid);
        LANES - (_mm256_movemask_ps(_mm256_castsi256_ps(invalid)) as u32).count_ones() as usize
    }
}

/// Stage the current chopped pixels and write the rolling signed difference.
///
/// The retained validity must already exclude non-finite values. Callers
/// must use scalar processing when floating-point exceptions are unmasked.
#[cfg(not(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
)))]
#[inline]
pub fn subtract_chopped<const CURRENT_FRINGED: bool>(
    (output, output_validity): (&mut [f32; LANES], &mut [bool; LANES]),
    (candidate, candidate_validity): (&mut [f32; LANES], &mut [bool; LANES]),
    (current, current_validity): (&[f32; LANES], &[bool; LANES]),
    (retained, retained_validity): (&[f32; LANES], &[bool; LANES]),
) {
    // SAFETY: The enabled target supports all intrinsics and unaligned
    // accesses stay within the arrays.
    unsafe {
        let current = _mm256_loadu_ps(current.as_ptr());
        let retained = _mm256_loadu_ps(retained.as_ptr());
        let current_flags = _mm_loadl_epi64(current_validity.as_ptr().cast());
        let retained_flags = _mm_loadl_epi64(retained_validity.as_ptr().cast());
        let zero = _mm256_setzero_si256();
        let current_invalid = _mm256_or_si256(
            _mm256_cmpeq_epi32(_mm256_cvtepu8_epi32(current_flags), zero),
            nonfinite_lanes(current),
        );
        let inactive = _mm256_or_si256(
            current_invalid,
            _mm256_cmpeq_epi32(_mm256_cvtepu8_epi32(retained_flags), zero),
        );
        let (fringed, unfringed) = if CURRENT_FRINGED {
            (current, retained)
        } else {
            (retained, current)
        };
        let values = subtract_active(fringed, unfringed, inactive);
        let invalid = _mm256_or_si256(inactive, nonfinite_lanes(values));
        _mm256_storeu_ps(
            output.as_mut_ptr(),
            _mm256_andnot_ps(_mm256_castsi256_ps(invalid), values),
        );
        write_validity(output_validity, invalid);
        _mm256_storeu_ps(candidate.as_mut_ptr(), current);
        write_validity(candidate_validity, current_invalid);
    }
}

#[cfg(not(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
)))]
#[inline]
fn subtract_active(samples: __m256, reference: __m256, inactive: __m256i) -> __m256 {
    // SAFETY: AVX2 is enabled.
    unsafe {
        let samples = _mm256_andnot_ps(_mm256_castsi256_ps(inactive), samples);
        let reference = _mm256_andnot_ps(_mm256_castsi256_ps(inactive), reference);
        // An opaque subtraction prevents LLVM from moving it before sanitization.
        let values: __m256;
        asm!(
            "vsubps {values}, {samples}, {reference}",
            samples = in(ymm_reg) samples,
            reference = in(ymm_reg) reference,
            values = lateout(ymm_reg) values,
            options(nostack, nomem),
        );
        values
    }
}

#[cfg(not(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
)))]
#[inline]
fn write_validity(output_validity: &mut [bool; LANES], invalid: __m256i) {
    // SAFETY: AVX2 is enabled. The store writes eight valid Bool bytes (0/1).
    unsafe {
        let flags = _mm256_andnot_si256(invalid, _mm256_set1_epi32(1));
        let flags = _mm_packus_epi32(
            _mm256_castsi256_si128(flags),
            _mm256_extracti128_si256::<1>(flags),
        );
        _mm_storel_epi64(
            output_validity.as_mut_ptr().cast(),
            _mm_packus_epi16(flags, flags),
        );
    }
}

#[cfg(not(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
)))]
#[inline]
fn nonfinite_lanes(values: __m256) -> __m256i {
    let mask: __m256i;
    // SAFETY: AVX2 is enabled. Comparing only integer exponent bits keeps
    // signaling NaNs and subnormals from affecting floating-point status.
    unsafe {
        let exponent = _mm256_set1_epi32(f32::INFINITY.to_bits() as i32);
        asm!(
            "vpand {mask}, {values}, {exponent}",
            "vpcmpeqd {mask}, {mask}, {exponent}",
            values = in(ymm_reg) values,
            exponent = in(ymm_reg) exponent,
            mask = out(ymm_reg) mask,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    mask
}

/// Subtract the reference from each finite, valid sample. Write positive zero
/// and false for inactive, nonfinite or overflowing lanes. Return the valid count.
/// Callers with unmasked exceptions must use scalar operations if store order
/// before a trap is observable.
#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
#[inline]
pub fn subtract_finite(
    output: &mut [f32; LANES],
    output_validity: &mut [bool; LANES],
    difference: &[f32; LANES],
    validity: &[bool; LANES],
    reference: &[f32; LANES],
) -> usize {
    // SAFETY: The required AVX-512 features are enabled. Unaligned loads/stores
    // stay within the arrays, with 0/1 in every output Bool byte.
    unsafe {
        let samples = _mm512_loadu_ps(difference.as_ptr());
        let reference = _mm512_loadu_ps(reference.as_ptr());
        let input_flags = _mm_loadu_si128(validity.as_ptr().cast());
        let active =
            _mm_cmpneq_epi8_mask(input_flags, _mm_setzero_si128()) & !nonfinite_mask(samples);
        let values = subtract_active(samples, reference, active);
        let valid = active & !nonfinite_mask(values);
        _mm512_storeu_ps(output.as_mut_ptr(), _mm512_maskz_mov_ps(valid, values));
        write_validity(output_validity, valid);
        valid.count_ones() as usize
    }
}

/// Stage the current chopped pixels and write the rolling signed difference.
///
/// The retained validity must already exclude non-finite values. Callers
/// must use scalar processing when floating-point exceptions are unmasked.
#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
#[inline]
pub fn subtract_chopped<const CURRENT_FRINGED: bool>(
    (output, output_validity): (&mut [f32; LANES], &mut [bool; LANES]),
    (candidate, candidate_validity): (&mut [f32; LANES], &mut [bool; LANES]),
    (current, current_validity): (&[f32; LANES], &[bool; LANES]),
    (retained, retained_validity): (&[f32; LANES], &[bool; LANES]),
) {
    // SAFETY: The enabled target supports all intrinsics and unaligned
    // accesses stay within the arrays.
    unsafe {
        let current = _mm512_loadu_ps(current.as_ptr());
        let retained = _mm512_loadu_ps(retained.as_ptr());
        let current_flags = _mm_loadu_si128(current_validity.as_ptr().cast());
        let retained_flags = _mm_loadu_si128(retained_validity.as_ptr().cast());
        let zero = _mm_setzero_si128();
        let current_valid = _mm_cmpneq_epi8_mask(current_flags, zero) & !nonfinite_mask(current);
        let active = current_valid & _mm_cmpneq_epi8_mask(retained_flags, zero);
        let (fringed, unfringed) = if CURRENT_FRINGED {
            (current, retained)
        } else {
            (retained, current)
        };
        let values = subtract_active(fringed, unfringed, active);
        let valid = active & !nonfinite_mask(values);
        _mm512_storeu_ps(output.as_mut_ptr(), _mm512_maskz_mov_ps(valid, values));
        write_validity(output_validity, valid);
        _mm512_storeu_ps(candidate.as_mut_ptr(), current);
        write_validity(candidate_validity, current_valid);
    }
}

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
#[inline]
fn subtract_active(samples: __m512, reference: __m512, active: u16) -> __m512 {
    // SAFETY: AVX-512F is enabled.
    unsafe {
        // Inactive lanes have no arithmetic or floating-point exceptions.
        let values: __m512;
        asm!(
            "vsubps {values}{{{active}}}{{z}}, {samples}, {reference}",
            samples = in(zmm_reg) samples,
            reference = in(zmm_reg) reference,
            active = in(kreg) active,
            values = lateout(zmm_reg) values,
            options(nostack, nomem),
        );
        values
    }
}

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
#[inline]
fn write_validity(output: &mut [bool; LANES], valid: u16) {
    // SAFETY: The required features are enabled. The store writes sixteen
    // valid Bool bytes (0/1).
    unsafe { _mm_storeu_si128(output.as_mut_ptr().cast(), _mm_maskz_set1_epi8(valid, 1)) }
}

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
#[inline]
fn nonfinite_mask(values: __m512) -> u16 {
    let mask: u16;
    // SAFETY: AVX-512F is enabled. Compare integer exponent bits without
    // raising floating-point exceptions.
    unsafe {
        let exponent = _mm512_set1_epi32(f32::INFINITY.to_bits() as i32);
        asm!(
            "vpandd {scratch}, {values}, {exponent}",
            "vpcmpeqd {mask}, {scratch}, {exponent}",
            values = in(zmm_reg) values,
            exponent = in(zmm_reg) exponent,
            scratch = out(zmm_reg) _,
            mask = lateout(kreg) mask,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    mask
}
