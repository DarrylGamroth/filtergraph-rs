# BLAS and SIMD integration

Optional experimental features select a shape-specialized pupil GEMV and
compact feedback. Default builds use OpenBLAS for supported dense
matrix–vector multiplication and use the full feedback matrix.

The `blas` feature selects statically linked OpenBLAS 0.3.32 using the CBLAS
API and `cblas` 0.5.0 types for prepared Float32 and Float64 matrix/vector multiplication. It is the
default for the algorithms and production bundle packages; `oxiblas` remains a
compatibility feature alias. There is no production dependency on Julia.
Disabling default features retains ordered scalar GEMV and the algorithms
crate's `no_std` profile. Scientific source still forbids unsafe code.

## Build for AVX2 and FMA

ML LLOWFS inference, SCC pixel residual and chopped-image difference,
Shack–Hartmann centroiding, and pyramid image normalization run in Rust.
To select the tested AVX2/FMA paths in the FGN bundle on an x86-64 host that
supports both features, build with the default Cargo features and these
compiler flags.

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma" \
    cargo build --locked --release -p filtergraph-fgn-bundle
```

Every machine loading that bundle must support AVX2 and FMA. A generic build
without these flags keeps the portable Rust paths available. OpenBLAS
selects its own CPU kernel at runtime independently of the Rust build flags.

The experimental pupil GEMV uses a fixed reduction tree in safe Rust. The
compiler chooses whether to pack its independent accumulators into 256-bit or
512-bit vectors. The `experimental-simd-gemv` feature enables this kernel
for the supported shape and numerical envelope. Other pupil inputs fall back
to the ordinary matrix–vector path.

## Boundary and build policy

`filtergraph-blas` contains the native backend and its checked CBLAS calls. This name describes the established backend, without introducing
new scientific terminology or changing schemas. Dimensions, compact matrix
extent, input/output lengths, signed integer products, and scratch-size sums
are checked before FFI. Safe borrowed slices provide initialization, lifetime
and output exclusivity. Matrices outside backend integer or stack-scratch bounds use the
scientific scalar fallback. Row-major layout, unit strides, alpha=1 and beta=0
preserve the prepared operation's coordinates and overwrite semantics.

The locked `openblas-src` 0.10.16 dependency builds OpenBLAS from source using
Make, a C compiler and a Fortran compiler; the archive is cached in the user's
OpenBLAS build directory. Workspace Cargo configuration fixes LP64 integers,
one thread, CPU dispatch, and scratch-pool locking. The native crate's build
script rejects missing policy at an external build entry point. In particular,
a single-thread OpenBLAS build still needs locking for concurrent independent
callers. Native source and licenses remain in the dependency cache; dependency
versions and artifact identity accompany validation evidence.

The original OpenBLAS GEMV interface borrows a globally locked scratch pool for
larger matrices, conflicting with CAL-DECL-007. The boundary therefore compiles
the exact pinned vendor `interface/gemv.c` twice with unique private C names
and a 64 KiB stack-scratch cap. The scientific equations and dispatched kernels
are unchanged. It uses CBLAS's enum/LP64 ABI and matches the archive's four
precision build definitions and dispatch configuration; ordinary CBLAS symbols
are neither replaced nor interposed. The vendor source and license stay intact.

The checked `rows + columns` limit is 16,256 for Float32 and 8,064 for
Float64. Both limits keep scratch, including platform padding and alignment,
below 64 KiB. Admitted calls
cannot enter the native scratch-pool branch. Larger dimensions retain the
ordered scalar path unless the prepared Float32 dense reconstructor owns
workspace for them. Calling threads must provide at least the 64 KiB native
scratch budget plus caller frames, and warm stack pages and BLAS initialization
off the repeated path. This is a bounded stack cost, not a new heap workspace.
The C build enables stack-clash protection where supported.

A prepared Float32 dense reconstructor with a wide LP64 matrix allocates one
aligned workspace per execution owner. Its private C entry calls the same
dispatched OpenBLAS kernel used by the row-major stack interface. The workspace
is sized before replay from the matrix dimensions, with padding for the pinned
kernel interface. Replacing coefficients preserves the shape and reuses the
workspace. Other operation types continue to use their existing scratch policy.

The wrapper allocates no Rust storage and transfers no buffer ownership.
Rust allocator counters alone cannot establish native allocation or syscall
behavior; independent native instrumentation and source review qualify the
private interface. These checks do not establish physical timing or arbitrary
application thread-stack suitability.

## Execution and numerical behavior

BLAS owns the complete DenseF32 operation at every PipeWire worker count.
The host's existing scalar dense callback otherwise bypasses the backend even
with one lane. Selecting BLAS consistently also preserves the declaration
contract's worker-count invariance; dense helper parallelism is unavailable in
this profile. The no-BLAS profile retains the existing host callback. Other
specialized host lane routes retain their established dispatch.

Full GEMV may group or fuse reductions differently from scalar code. JFG is
the scientific authority; accepted output tolerances, whole-call rejection,
recovery, state and publication remain qualification gates. Progressive AXPY
retains separate multiply/add SIMD because fusing an overflowing product can
change JFG's accepted/rejected outcome. No gains or coefficients change.

Vector linear combination now computes into its existing disposable scratch
before checking for nonfinite results and copying to output. It retains input
validation and separate arithmetic operations. Rejected output remains
unpublished; scratch contents have no mathematical history. Leaky integration,
UInt16 pixel calibration and AXPY already emit packed SIMD. Float32 image
calibration retains its logical-coordinate rejection precedence.
