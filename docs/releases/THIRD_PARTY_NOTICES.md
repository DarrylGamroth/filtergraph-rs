# Third-party notices for FilterGraph

The FGN adapter represents the public PipeWireAO ABI 8 declarations in
`spa/filter-graph/ndarray-plugin.h` and `spa/filter-graph/ndarray-executor.h`.
Those headers carry Copyright © 2026 PipeWireAO contributors and are
licensed under MIT. The upstream license is included in
[PipeWireAO/COPYING](third-party-licenses/PipeWireAO/COPYING). The Rust adapter
and its C tests use these interfaces. The bundle does not ship the native host.

[Native ABI header](https://github.com/DarrylGamroth/PipeWireAO/blob/v4.0.10/spa/include/spa/filter-graph/ndarray-plugin.h) · [Executor header](https://github.com/DarrylGamroth/PipeWireAO/blob/v4.0.10/spa/include/spa/filter-graph/ndarray-executor.h)

The default shared library statically includes OpenBLAS 0.3.32 and Rust
libraries. Keep this file and the accompanying `third-party-licenses/` directory
with a binary distribution. The installer stages them under
`share/doc/filtergraph-rs/` alongside the separately installed library.

OpenBLAS copyright, conditions and disclaimer are reproduced verbatim in
`third-party-licenses/OpenBLAS-0.3.32/LICENSE`; bundled reference BLAS, LAPACK
and related notices are included in the same directory. Rust standard-library
attribution and license texts are in `third-party-licenses/rust-1.99.1/`.

The following resolved normal dependency closure includes procedural macros
used during compilation. Files are copied verbatim from the locked crate
sources. This inventory does not grant a license for filtergraph-rs itself.

| Crate | Version | Declared license | Included notice files |
|---|---|---|---|
| cblas | 0.5.0 | Apache-2.0 OR MIT | LICENSE.md |
| cblas-sys | 0.2.0 | Apache-2.0 OR MIT | LICENSE.md |
| itoa | 1.0.18 | MIT OR Apache-2.0 | LICENSE-APACHE,LICENSE-MIT |
| libc | 0.2.191 | MIT OR Apache-2.0 | LICENSE-APACHE,LICENSE-MIT |
| libm | 0.2.16 | MIT | LICENSE.txt |
| memchr | 2.8.3 | Unlicense OR MIT | COPYING,LICENSE-MIT |
| num-complex | 0.4.6 | MIT OR Apache-2.0 | LICENSE-APACHE,LICENSE-MIT |
| num-traits | 0.2.19 | MIT OR Apache-2.0 | LICENSE-APACHE,LICENSE-MIT |
| openblas-src | 0.10.16 | Apache-2.0/MIT | LICENSE.md |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 | LICENSE-APACHE,LICENSE-MIT |
| quote | 1.0.47 | MIT OR Apache-2.0 | LICENSE-APACHE,LICENSE-MIT |
| serde | 1.0.229 | MIT OR Apache-2.0 | LICENSE-APACHE,LICENSE-MIT |
| serde_core | 1.0.229 | MIT OR Apache-2.0 | LICENSE-APACHE,LICENSE-MIT |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 | LICENSE-APACHE,LICENSE-MIT |
| serde_json | 1.0.151 | MIT OR Apache-2.0 | LICENSE-APACHE,LICENSE-MIT |
| syn | 3.0.3 | MIT OR Apache-2.0 | LICENSE-APACHE,LICENSE-MIT |
| unicode-ident | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 | LICENSE-APACHE,LICENSE-MIT,LICENSE-UNICODE |
| zmij | 1.0.23 | MIT | LICENSE-MIT |

The openblas-src crate omits its workspace license from the registry archive;
its notice is copied from [the exact packaged Git revision](https://github.com/blas-lapack-rs/openblas-src/blob/98ff41d8b2dd8f881517a2c25c8db19a0ab0f722/LICENSE.md).
