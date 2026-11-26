# Scientific schemas

The [Rust schema definitions](../crates/filtergraph-algorithms/src/schemas.rs)
list the supported identifiers. Shared scientific IDs retain their exact
`pipewireao/<name>-<type>/<version>` spelling. Treat the full identifier as
opaque at a graph connection and validate its structural element type, shape,
layout, and rate separately.

[Pixel calibration](pixel-calibration-1.md) documents the calibrated-pixel
quantity. [Extensions](extensions.toml) records Rust-owned additions.
