# Contributor guidance

Keep numerical plans and workspaces transport-independent. The FGN adapter
maps them to PipeWireAO ABI 8. The native host owns graph execution. Keep
instrument configurations, calibrations, and hardware qualification in the
downstream deployment that owns them.

Preserve scientific names, units, signs, operation ordering, and schema IDs.
Use the [declaration contract](docs/algorithm-declaration-contract.md) and
[scalar property contract](docs/scalar-property-contract.md) when changing
configuration, ports, properties, or runtime state.

Keep the algorithms crate compatible with `no_std` plus `alloc`, and keep
OpenBLAS as the default backend. The workspace Cargo configuration fixes LP64,
one BLAS thread, native scratch locking, and CPU dispatch. See
[BLAS and SIMD integration](docs/blas-simd.md) before changing that boundary.

Run `cargo test --workspace --locked`, `cargo fmt --all --check`, and
`cargo test --locked -p filtergraph-algorithms --no-default-features` for
Rust changes. Use the [FGN guide](docs/fgn-plugin.md) for the native declaration
and shared-library tests. Software tests and native host tests provide separate
evidence from instrument and timing qualification.
