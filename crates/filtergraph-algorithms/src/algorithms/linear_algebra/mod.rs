//! Matrix-vector operations and their shared backend boundary.
//!
//! For an input unit `X` and output unit `Y`, matrix coefficients use `Y/X`.
//! Matrix-vector multiplication performs no implicit unit conversion.

mod gemv;
mod incremental_dense_reconstruction;
mod matrix_vector;
mod progressive_matrix_vector;

pub use gemv::MatrixElement;
pub use matrix_vector::{GemvError, PreparedGemv, PreparedGemvError};
pub use progressive_matrix_vector::{
    PreparedProgressiveGemv, PreparedProgressiveGemvError, ProgressiveGemvError,
};

pub(crate) use gemv::{multiply_add, multiply_row_major};

pub use incremental_dense_reconstruction::{
    IncrementalDenseReconstructionError, IncrementalDenseReconstructionExecutionError,
    IncrementalDenseReconstructionPlan, IncrementalDenseReconstructionPreparationError,
    IncrementalDenseReconstructionProcessError, IncrementalDenseReconstructionWorkspace,
    IncrementalDenseReconstructorF32, IncrementalDenseReconstructorF32Configuration,
};

mod dense_vector_reconstruction;
pub use dense_vector_reconstruction::*;
