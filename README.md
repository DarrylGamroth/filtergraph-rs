# filtergraph-rs

[![CI](https://github.com/DarrylGamroth/filtergraph-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DarrylGamroth/filtergraph-rs/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/DarrylGamroth/filtergraph-rs/branch/main/graph/badge.svg)](https://app.codecov.io/gh/DarrylGamroth/filtergraph-rs)

Rust numerical algorithms and Filter-Graph Native (FGN) plugins for adaptive optics.
The workspace contains the numerical library, the OpenBLAS backend, the FGN
adapter, and the shared-library bundle that registers the numerical declarations.

## Build and test

The default build requires Rust, Make, a C compiler, and a Fortran compiler.
OpenBLAS is built from source on the first build. The workspace configuration
selects LP64, one BLAS thread, CPU dispatch, and scratch-pool locking.

```sh
cargo test --workspace --locked
cargo test --locked -p filtergraph-algorithms --no-default-features
cargo build --release --locked -p filtergraph-fgn-bundle
```

`filtergraph-algorithms` uses `no_std` and Rust's `alloc` crate for owned
configuration, prepared plans, and workspaces. The portable build uses ordered
Rust matrix-vector multiplication. See the [BLAS notes](docs/blas-simd.md) for
the default backend and scratch limits.

GitHub CI combines Rust line coverage from the default, portable, and
all-features tests and the synthetic native FGN graph tests. The Codecov project
target is above 90%.

## Use with PipeWireAO

The [FGN setup guide](docs/fgn-plugin.md) shows how to load the bundle into a
published PipeWireAO 4.x host. Use the [declaration contract](docs/algorithm-declaration-contract.md)
and [scalar property contract](docs/scalar-property-contract.md) when writing
an operation. Applications supply their graph, calibrations, and device plugins
and can pin a FilterGraph release for deployment qualification.

## License

FilterGraph source is available under the [MIT license](LICENSE). The
[third-party notices](docs/releases/THIRD_PARTY_NOTICES.md) cover the PipeWireAO
ABI declarations and dependencies used by the FGN bundle.
