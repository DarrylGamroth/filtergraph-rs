//! Independently callable numerical operations.

mod array_statistics;
mod calibration;
mod control;
mod deformable_mirror;
mod docrime;
mod image;
mod linear_algebra;
mod ndarray_copy;
mod optical_gain;
mod physical_offset_reference;
mod scc;
mod splitmix64;
mod vector;
mod wavefront_sensor;

pub(crate) use splitmix64::{splitmix64_advance, splitmix64_at, splitmix64_unit_at};

pub use array_statistics::*;
pub use calibration::*;
pub use control::*;
pub use deformable_mirror::*;
pub use docrime::*;
pub use image::*;
pub use linear_algebra::*;
pub use ndarray_copy::*;
pub use optical_gain::*;
pub use physical_offset_reference::*;
pub use scc::*;
pub use vector::*;
pub use wavefront_sensor::*;
