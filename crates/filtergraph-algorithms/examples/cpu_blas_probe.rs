//! Controlled CPU-only backend/layout probe, not a production backend qualification.
//! ABI authority: OpenBLAS v0.3.30 cblas.h and openblas_config_template.h:
//! https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.30/cblas.h
//! https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.30/openblas_config_template.h
//! CBLAS enums are C int; USE64BITINT blasint is C long on this Linux ABI.
use filtergraph_algorithms::PreparedGemv;
use serde::Serialize;
use std::{
    error::Error,
    ffi::{CStr, CString, c_char, c_int, c_long, c_void},
    hint::black_box,
    sync::Arc,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn dlclose(handle: *mut c_void) -> c_int;
}
type Sgemv = unsafe extern "C" fn(
    c_int,
    c_int,
    c_long,
    c_long,
    f32,
    *const f32,
    c_long,
    *const f32,
    c_long,
    f32,
    *mut f32,
    c_long,
);
struct Blas {
    handle: *mut c_void,
    gemv: Sgemv,
    config: String,
    threads: c_int,
}
impl Blas {
    fn load(path: &str) -> Result<Self> {
        if std::mem::size_of::<c_long>() != 8 {
            return Err("probe requires Linux ILP64 C-long ABI".into());
        }
        let path = CString::new(path)?;
        // RTLD_NOW resolves dependencies immediately. Every requested symbol is checked.
        let handle = unsafe { dlopen(path.as_ptr(), 2) };
        if handle.is_null() {
            let message = unsafe { CStr::from_ptr(dlerror()) }.to_string_lossy();
            return Err(message.into_owned().into());
        }
        unsafe fn symbol<T: Copy>(handle: *mut c_void, name: &str) -> Result<T> {
            let name = CString::new(name)?;
            let pointer = unsafe { dlsym(handle, name.as_ptr()) };
            if pointer.is_null() {
                return Err(format!("missing symbol {name:?}").into());
            }
            // All T signatures below follow the cited OpenBLAS header, never inferred by calls.
            Ok(unsafe { std::mem::transmute_copy(&pointer) })
        }
        let config: unsafe extern "C" fn() -> *const c_char =
            unsafe { symbol(handle, "openblas_get_config64_")? };
        let config = unsafe { CStr::from_ptr(config()) }
            .to_string_lossy()
            .into_owned();
        if !config.contains("OpenBLAS 0.3.30") || !config.contains("USE64BITINT") {
            return Err(format!("unverified ABI/config: {config}").into());
        }
        let set: unsafe extern "C" fn(c_int) =
            unsafe { symbol(handle, "openblas_set_num_threads64_")? };
        let get: unsafe extern "C" fn() -> c_int =
            unsafe { symbol(handle, "openblas_get_num_threads64_")? };
        unsafe { set(1) };
        let threads = unsafe { get() };
        if threads != 1 {
            return Err("OpenBLAS thread budget differs".into());
        }
        Ok(Self {
            handle,
            gemv: unsafe { symbol(handle, "cblas_sgemv64_")? },
            config,
            threads,
        })
    }
    fn multiply(
        &self,
        rows: usize,
        columns: usize,
        matrix: &[f32],
        column_major: bool,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        // These extra FFI guards establish the complete matrix memory extent.
        // PreparedGemv establishes its immutable matrix invariant at preparation.
        if rows == 0
            || columns == 0
            || rows > c_long::MAX as usize
            || columns > c_long::MAX as usize
            || rows.checked_mul(columns) != Some(matrix.len())
        {
            return Err("invalid matrix extent".into());
        }
        // Match PreparedGemv's per-call dimensions and finite-input/output validation.
        if input.len() != columns || output.len() != rows {
            return Err("dimension mismatch".into());
        }
        if input.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite input".into());
        }
        // Live buffers have the validated leading dimension/extent, unit strides, no aliasing.
        unsafe {
            (self.gemv)(
                if column_major { 102 } else { 101 },
                111,
                rows as c_long,
                columns as c_long,
                1.,
                matrix.as_ptr(),
                if column_major { rows } else { columns } as c_long,
                input.as_ptr(),
                1,
                0.,
                output.as_mut_ptr(),
                1,
            )
        };
        if output.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite output".into());
        }
        Ok(())
    }
}
impl Drop for Blas {
    fn drop(&mut self) {
        unsafe { dlclose(self.handle) };
    }
}
fn column_major(matrix: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut output = vec![0.; matrix.len()];
    for r in 0..rows {
        for c in 0..columns {
            output[r + rows * c] = matrix[r * columns + c];
        }
    }
    output
}
fn agrees(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(&x, &y)| {
            x.is_finite() && y.is_finite() && (x - y).abs() <= 1e-7 + 2e-6 * y.abs()
        })
}
#[derive(Serialize)]
struct Run {
    rows: usize,
    columns: usize,
    repetition: usize,
    order: usize,
    backend: String,
    layout: String,
    call_ns: Vec<u64>,
    batch_ns: Vec<u64>,
    batch_size: usize,
    batch_median_ns_per_operation: f64,
    output: Vec<f64>,
}
fn timed(mut operation: impl FnMut() -> Result<()>) -> Result<(Vec<u64>, Vec<u64>)> {
    for _ in 0..200 {
        operation()?;
    }
    let mut calls = vec![0; 2000];
    for t in &mut calls {
        let start = Instant::now();
        operation()?;
        *t = start.elapsed().as_nanos() as u64;
    }
    let mut batches = vec![0; 200];
    for t in &mut batches {
        let start = Instant::now();
        for _ in 0..32 {
            operation()?;
        }
        *t = start.elapsed().as_nanos() as u64;
    }
    Ok((calls, batches))
}
#[derive(Serialize)]
struct Probe {
    library: String,
    openblas_config: String,
    openblas_threads: c_int,
    warmup: usize,
    samples: usize,
    batches: usize,
    batch_size: usize,
    validation: String,
    cancellation_scalar: Vec<f64>,
    cancellation_row_major: Vec<f64>,
    cancellation_column_major: Vec<f64>,
    runs: Vec<Run>,
}
fn main() -> Result<()> {
    let library = std::env::args()
        .nth(1)
        .ok_or("usage: cpu_blas_probe /absolute/libopenblas64_.so")?;
    let blas = Blas::load(&library)?;
    let cancel = vec![-1., 0., 0.5, 0., -0.25, 0.];
    let cancel_column = column_major(&cancel, 1, 6);
    let input = [f32::MAX; 6];
    let plan = PreparedGemv::new(1, 6, Arc::from(cancel.clone()))?;
    let mut scalar = vec![0.; 1];
    let mut row = scalar.clone();
    let mut column = scalar.clone();
    plan.multiply(&mut scalar, &input)?;
    blas.multiply(1, 6, &cancel, false, &input, &mut row)?;
    blas.multiply(1, 6, &cancel_column, true, &input, &mut column)?;
    if !agrees(&scalar, &row) || !agrees(&scalar, &column) {
        return Err("finite-cancellation regression differs".into());
    }
    let mut runs = Vec::new();
    for (rows, columns) in [(64, 128), (256, 512)] {
        let matrix = (0usize..rows * columns)
            .map(|i| ((i * 17 % 97) as i32 - 48) as f32 / 256.)
            .collect::<Vec<_>>();
        let transposed = column_major(&matrix, rows, columns);
        let input = (0usize..columns)
            .map(|i| ((i * 13 % 101) as i32 - 50) as f32 / 64.)
            .collect::<Vec<_>>();
        let plan = PreparedGemv::new(rows, columns, Arc::from(matrix.clone()))?;
        let mut reference = vec![0.; rows];
        plan.multiply(&mut reference, &input)?;
        for repetition in 0..3 {
            let order = if repetition % 2 == 0 {
                [0, 1, 2]
            } else {
                [2, 1, 0]
            };
            for (position, backend) in order.into_iter().enumerate() {
                let mut output = vec![0.; rows];
                let (calls, batches) = timed(|| {
                    match backend {
                        0 => plan
                            .multiply(&mut output, black_box(&input))
                            .map_err(Box::<dyn Error>::from)?,
                        1 => blas.multiply(
                            rows,
                            columns,
                            black_box(&matrix),
                            false,
                            black_box(&input),
                            &mut output,
                        )?,
                        _ => blas.multiply(
                            rows,
                            columns,
                            black_box(&transposed),
                            true,
                            black_box(&input),
                            &mut output,
                        )?,
                    };
                    black_box(&output);
                    Ok(())
                })?;
                if !agrees(&reference, &output) {
                    return Err("GEMV outputs differ".into());
                }
                let mut sorted = batches.clone();
                sorted.sort_unstable();
                let median = (sorted[99] as f64 + sorted[100] as f64) / 2. / 32.;
                runs.push(Run {
                    rows,
                    columns,
                    repetition: repetition + 1,
                    order: position + 1,
                    backend: if backend == 0 {
                        "prepared-gemv-f32-scalar"
                    } else {
                        "direct-openblas-sgemv-ilp64"
                    }
                    .into(),
                    layout: if backend == 2 {
                        "column-major-pretransposed-untimed"
                    } else {
                        "row-major"
                    }
                    .into(),
                    call_ns: calls,
                    batch_ns: batches,
                    batch_size: 32,
                    batch_median_ns_per_operation: median,
                    output: output.iter().map(|&v| v as f64).collect(),
                });
            }
        }
    }
    println!("{}",toml::to_string(&Probe{library,openblas_config:blas.config.clone(),openblas_threads:blas.threads,warmup:200,samples:2000,batches:200,batch_size:32,validation:"full finite output; atol=1e-7,rtol=2e-6; includes per-call finite input/output guards; cancellation is one regression, not full profile qualification".into(),cancellation_scalar:scalar.iter().map(|&v|v as f64).collect(),cancellation_row_major:row.iter().map(|&v|v as f64).collect(),cancellation_column_major:column.iter().map(|&v|v as f64).collect(),runs})?);
    Ok(())
}
