//! Checked CBLAS matrix/vector calls for native CPU execution.
//!
//! OpenBLAS is linked statically, independently of Julia. Its LP64 dimensions
//! match the `cblas` bindings. Cargo configuration caps BLAS threads at one and
//! enables OpenBLAS CPU dispatch. This crate contains the FFI boundary so the
//! scientific algorithms can continue to forbid unsafe code.
#![deny(unsafe_op_in_unsafe_fn)]

use cblas::{Layout, Transpose};
use core::ffi::{CStr, c_char, c_int};
use openblas_src as _;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[doc(hidden)]
pub mod f32_lanes;

#[cfg(all(
    feature = "experimental-simd-gemv",
    target_arch = "x86_64",
    any(
        target_feature = "avx512f",
        all(target_feature = "avx2", target_feature = "fma")
    ),
))]
mod simd;

#[cfg(feature = "experimental-simd-gemv")]
type SgemvKernel = fn(&[f32], &[f32], &mut [f32]);

/// Return the opt-in SIMD kernel for the specialized pupil matrix on supported targets.
///
/// The caller must bound intermediate Float32 arithmetic in both this kernel
/// and OpenBLAS, require finite operands and check the thread's floating-point
/// mode. The kernel checks slice extents and overwrites the output.
#[cfg(feature = "experimental-simd-gemv")]
#[inline]
pub fn pupil_sgemv_kernel(rows: usize, columns: usize) -> Option<SgemvKernel> {
    #[cfg(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma")
        ),
    ))]
    if (rows, columns) == (simd::PUPIL_ROWS, simd::PUPIL_COLUMNS) {
        return Some(simd::multiply);
    }
    let _ = (rows, columns);
    None
}

/// Whether compact Float64 matrix extents fit the bounded BLAS backend.
///
/// The margin bounds Float64 scratch within the private 64 KiB stack
/// interface, including Windows padding and alignment. Larger operations must
/// use the scalar backend; they never enter the native shared scratch pool.
pub fn supports_dimensions(rows: usize, columns: usize) -> bool {
    dimensions_fit(rows, columns, size_of::<f64>())
}

/// Whether compact Float32 matrix extents fit the bounded BLAS backend.
pub fn supports_sgemv_dimensions(rows: usize, columns: usize) -> bool {
    dimensions_fit(rows, columns, size_of::<f32>())
}

fn matrix_fits_lp64(rows: usize, columns: usize) -> bool {
    rows > 0
        && columns > 0
        && rows
            .checked_mul(columns)
            .is_some_and(|n| n <= i32::MAX as usize)
}

fn dimensions_fit(rows: usize, columns: usize, element_bytes: usize) -> bool {
    matrix_fits_lp64(rows, columns)
        && rows
            .checked_add(columns)
            .is_some_and(|n| n <= 64 * 1024 / element_bytes - 128)
}

#[repr(C, align(64))]
#[derive(Clone)]
struct ScratchBlock {
    _values: [f32; 16],
}

/// Private scratch for a prepared, wide row-major Float32 GEMV.
///
/// One workspace belongs to one execution owner and is allocated before replay.
pub struct SgemvWorkspace {
    rows: c_int,
    columns: c_int,
    scratch: Vec<ScratchBlock>,
}

impl SgemvWorkspace {
    /// Prepare scratch for shapes exceeding the stack budget and within LP64.
    pub fn for_wide_matrix(rows: usize, columns: usize) -> Option<Self> {
        if !matrix_fits_lp64(rows, columns) || supports_sgemv_dimensions(rows, columns) {
            return None;
        }
        // The pinned interface uses rows + columns, plus up to 288 bytes of
        // padding and rounding to four elements. Budget 128 elements and
        // round the allocation to a whole 64-byte aligned block.
        let elements = rows.checked_add(columns)?.checked_add(128)?;
        Some(Self {
            rows: rows as c_int,
            columns: columns as c_int,
            scratch: vec![ScratchBlock { _values: [0.0; 16] }; elements.div_ceil(16)],
        })
    }

    /// Overwrite the output with the prepared shape's matrix-vector product.
    pub fn multiply(&mut self, matrix: &[f32], input: &[f32], output: &mut [f32]) {
        let (rows, columns) = (self.rows as usize, self.columns as usize);
        assert_eq!(matrix.len(), rows * columns);
        assert_eq!(input.len(), columns);
        assert_eq!(output.len(), rows);
        output.fill(0.0);
        // SAFETY: shape and LP64 extents were checked at construction, lengths
        // match, and the kernel borrows disjoint slices and aligned private
        // scratch for this synchronous call. It reads matrix and input only.
        unsafe {
            filtergraph_sgemv_with_workspace(
                self.rows,
                self.columns,
                matrix.as_ptr(),
                input.as_ptr(),
                output.as_mut_ptr(),
                self.scratch.as_mut_ptr().cast(),
            );
        }
    }
}

fn dimensions(
    rows: usize,
    columns: usize,
    matrix: usize,
    input: usize,
    output: usize,
    element_bytes: usize,
) -> (i32, i32) {
    assert!(
        dimensions_fit(rows, columns, element_bytes),
        "BLAS integer or stack scratch extent limit"
    );
    assert_eq!(
        rows.checked_mul(columns),
        Some(matrix),
        "BLAS matrix extent"
    );
    assert_eq!(columns, input, "BLAS input extent");
    assert_eq!(rows, output, "BLAS output extent");
    (
        i32::try_from(rows).expect("BLAS row limit"),
        i32::try_from(columns).expect("BLAS column limit"),
    )
}

/// Compute `output = matrix × input` for a compact row-major Float32 matrix.
///
/// Panics on incompatible extents or unsupported backend integer arithmetic.
/// Output is exclusively borrowed; no Rust allocation or ownership transfer
/// occurs. The private interface uses at most 64 KiB of stack scratch and never
/// enters OpenBLAS's shared scratch pool for admitted dimensions. Calling threads
/// need sufficient stack and off-path warmup before latency-sensitive processing.
pub fn sgemv(rows: usize, columns: usize, matrix: &[f32], input: &[f32], output: &mut [f32]) {
    let (m, n) = dimensions(
        rows,
        columns,
        matrix.len(),
        input.len(),
        output.len(),
        size_of::<f32>(),
    );
    // SAFETY: the checked compact row-major matrix has m*n elements and lda=n;
    // input/output have n/m initialized elements, unit strides, disjoint safe
    // borrows and lifetimes spanning this synchronous call. Binding dimensions
    // and the statically linked OpenBLAS both use LP64 (32-bit BLAS integers).
    // The checked sum keeps Float32 within the private interface's 64 KiB
    // stack scratch, so its native-pool allocation branch is unreachable.
    unsafe {
        filtergraph_cblas_sgemv(
            Layout::RowMajor,
            Transpose::None,
            m,
            n,
            1.0,
            matrix.as_ptr(),
            n,
            input.as_ptr(),
            1,
            0.0,
            output.as_mut_ptr(),
            1,
        )
    }
}

/// Compute `output = matrix × input` for a compact row-major Float64 matrix.
///
/// Panics on incompatible extents or unsupported backend integer arithmetic.
/// Output is exclusively borrowed; no Rust allocation or ownership transfer
/// occurs. The private interface uses at most 64 KiB of stack scratch and never
/// enters OpenBLAS's shared scratch pool for admitted dimensions. Calling threads
/// need sufficient stack and off-path warmup before latency-sensitive processing.
pub fn dgemv(rows: usize, columns: usize, matrix: &[f64], input: &[f64], output: &mut [f64]) {
    let (m, n) = dimensions(
        rows,
        columns,
        matrix.len(),
        input.len(),
        output.len(),
        size_of::<f64>(),
    );
    // SAFETY: identical extent, leading-dimension, stride, lifetime and aliasing
    // proof to sgemv, with initialized Float64 elements and the matching ABI.
    unsafe {
        filtergraph_cblas_dgemv(
            Layout::RowMajor,
            Transpose::None,
            m,
            n,
            1.0,
            matrix.as_ptr(),
            n,
            input.as_ptr(),
            1,
            0.0,
            output.as_mut_ptr(),
            1,
        )
    }
}

/// Whether the current thread uses the supported default Float32 arithmetic mode.
pub fn sgemv_default_float_mode() -> bool {
    // SAFETY: this reads the current thread's floating-point control register.
    unsafe { filtergraph_sgemv_default_float_mode() != 0 }
}

unsafe extern "C" {
    fn filtergraph_sgemv_default_float_mode() -> c_int;
    fn filtergraph_sgemv_with_workspace(
        rows: c_int,
        columns: c_int,
        matrix: *const f32,
        input: *const f32,
        output: *mut f32,
        scratch: *mut f32,
    );
    // Same LP64 CBLAS ABI and enum representations as the cblas bindings.
    // These uniquely named interfaces compile the pinned vendor gemv.c with
    // bounded stack scratch; they call the unchanged dispatched BLAS kernels.
    fn filtergraph_cblas_sgemv(
        layout: Layout,
        transpose: Transpose,
        rows: c_int,
        columns: c_int,
        alpha: f32,
        matrix: *const f32,
        lda: c_int,
        input: *const f32,
        incx: c_int,
        beta: f32,
        output: *mut f32,
        incy: c_int,
    );
    fn filtergraph_cblas_dgemv(
        layout: Layout,
        transpose: Transpose,
        rows: c_int,
        columns: c_int,
        alpha: f64,
        matrix: *const f64,
        lda: c_int,
        input: *const f64,
        incx: c_int,
        beta: f64,
        output: *mut f64,
        incy: c_int,
    );
    fn openblas_get_config() -> *const c_char;
    fn openblas_get_corename() -> *const c_char;
    fn openblas_get_num_threads() -> c_int;
}

/// The linked OpenBLAS build configuration; intended for cold-path diagnostics.
pub fn configuration() -> &'static str {
    // SAFETY: OpenBLAS returns an immutable, static, NUL-terminated ASCII string.
    unsafe { CStr::from_ptr(openblas_get_config()) }
        .to_str()
        .expect("OpenBLAS configuration encoding")
}
/// CPU kernel selected by OpenBLAS; intended for cold-path diagnostics.
pub fn core_name() -> &'static str {
    // SAFETY: OpenBLAS returns an immutable, static, NUL-terminated ASCII string.
    unsafe { CStr::from_ptr(openblas_get_corename()) }
        .to_str()
        .expect("OpenBLAS CPU name encoding")
}
/// Number of BLAS threads allowed for one call.
pub fn thread_count() -> usize {
    // SAFETY: this argument-free read-only OpenBLAS query has the C-int ABI.
    usize::try_from(unsafe { openblas_get_num_threads() }).expect("positive OpenBLAS thread count")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_coordinates_and_nonvector_tail() {
        let mut f32_output = [99.0; 2];
        let mut f64_output = [99.0; 2];
        sgemv(
            2,
            3,
            &[1., 2., 3., -1., 0.5, 0.25],
            &[1., 2., 4.],
            &mut f32_output,
        );
        dgemv(
            2,
            3,
            &[1., 2., 3., -1., 0.5, 0.25],
            &[1., 2., 4.],
            &mut f64_output,
        );
        assert_eq!(f32_output, [17., 1.]);
        assert_eq!(f64_output, [17., 1.]);
        assert_eq!(thread_count(), 1);
        assert!(!configuration().contains("USE64BITINT"));
    }
    #[test]
    fn independent_concurrent_calls_use_private_stack_scratch() {
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let matrix = vec![0.5_f32; 17 * 1025];
                    let input = vec![2.0_f32; 1025];
                    let mut output = vec![0.0_f32; 17];
                    for _ in 0..100 {
                        sgemv(17, 1025, &matrix, &input, &mut output);
                        assert_eq!(output, [1025.0; 17]);
                    }
                });
            }
        });
    }

    #[test]
    fn scratch_limit_and_largest_admitted_sum() {
        let columns = 64 * 1024 / size_of::<f64>() - 128 - 1;
        assert!(supports_dimensions(1, columns));
        assert!(!supports_dimensions(1, columns + 1));
        let matrix = vec![1.0_f64; columns];
        let input = vec![1.0_f64; columns];
        let mut output = [0.0_f64];
        dgemv(1, columns, &matrix, &input, &mut output);
        assert_eq!(output, [columns as f64]);
    }

    #[test]
    fn float32_scratch_limit_admits_pupil_matrix_and_rejects_the_next_extent() {
        let limit = 64 * 1024 / size_of::<f32>() - 128;
        assert!(supports_sgemv_dimensions(1509, 11104));
        assert!(!supports_dimensions(1509, 11104));
        for (rows, columns) in [(1, limit - 1), (limit - 1, 1)] {
            assert!(supports_sgemv_dimensions(rows, columns));
            assert!(!supports_sgemv_dimensions(rows, columns + 1));
            let matrix = vec![0.5_f32; rows * columns];
            let input = vec![2.0_f32; columns];
            let mut output = vec![0.0_f32; rows];
            sgemv(rows, columns, &matrix, &input, &mut output);
            assert!(output.iter().all(|value| *value == columns as f32));
        }
    }

    #[test]
    fn wide_workspace_overwrites_and_reuses_output() {
        assert!(SgemvWorkspace::for_wide_matrix(8, 1000).is_none());
        assert!(SgemvWorkspace::for_wide_matrix(usize::MAX, 2).is_none());
        for columns in [50_176, 65_536] {
            let mut workspace = SgemvWorkspace::for_wide_matrix(8, columns).unwrap();
            let matrix = vec![0.5; 8 * columns];
            let mut input = vec![2.0; columns];
            let mut output = [f32::NAN; 8];
            workspace.multiply(&matrix, &input, &mut output);
            assert_eq!(output, [columns as f32; 8]);
            input.fill(-2.0);
            workspace.multiply(&matrix, &input, &mut output);
            assert_eq!(output, [-(columns as f32); 8]);
        }
    }

    #[test]
    fn invalid_memory_extents_reject_before_ffi() {
        for (rows, columns, matrix, input, output) in [
            (0, 1, 0, 1, 0),
            (2, 3, 5, 3, 2),
            (2, 3, 6, 2, 2),
            (2, 3, 6, 3, 1),
            (usize::MAX, 2, 0, 0, 0),
            (
                1,
                i32::MAX as usize,
                i32::MAX as usize,
                i32::MAX as usize,
                1,
            ),
            (65536, 65536, 0, 65536, 65536),
        ] {
            assert!(
                std::panic::catch_unwind(|| dimensions(
                    rows,
                    columns,
                    matrix,
                    input,
                    output,
                    size_of::<f64>()
                ))
                .is_err()
            );
        }
    }
}
