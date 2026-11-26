//! Dense Shack-Hartmann slope reconstruction as a portable graph boundary.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use crate::{GemvError, NoWorkspace, PreparedGemv, PreparedGemvError};

crate::filtergraph_algorithm! {
    /// Portable Float32 dense Shack-Hartmann reconstruction declaration.
    pub ShackHartmannReconstructorF32 {
        label = "shwfs-reconstructor-f32";
        configuration ShackHartmannReconstructorF32Configuration {
            reconstructed_count: usize = GraphRebuild;
            subaperture_count: usize = GraphRebuild;
            initial_reconstructor: Option<Vec<f32>> = Construction;
            reconstructed_schema: String = GraphRebuild;
        }
        plan = PreparedGemv<f32>;
        operation = DenseF32;
        workspace = NoWorkspace;
        properties = none;
        dimensions |plan, _workspace| {
            actuator_count = plan.rows();
            slope_count = plan.columns();
            subaperture_count = slope_count / 2;
        }
        inputs {
            slopes: f32 [subaperture_count, 2] = "slopes",
                schema = crate::schemas::SHACK_HARTMANN_SLOPES_V1;
        }
        outputs {
            reconstructed: f32 [actuator_count] = "reconstructed",
                schema = configuration.reconstructed_schema.as_str(),
                metadata_from = "slopes";
        }
        parameters {
            reconstructor: f32 [actuator_count, slope_count] = "reconstructor",
                schema = crate::schemas::SHWFS_RECONSTRUCTOR_V1,
                replace |plan, values| {
                    PreparedGemv::new(plan.rows(), plan.columns(), Arc::from(values))
                };
        }
        parameter_error = PreparedGemvError;
        prepare |configuration| {
            let slope_count = configuration
                .subaperture_count
                .checked_mul(2)
                .ok_or(PreparedGemvError::ExtentOverflow)?;
            let element_count = configuration.reconstructed_count
                .checked_mul(slope_count)
                .ok_or(PreparedGemvError::ExtentOverflow)?;
            let reconstructor = configuration
                .initial_reconstructor
                .unwrap_or_else(|| vec![0.0; element_count]);
            Ok((
                PreparedGemv::new(
                    configuration.reconstructed_count,
                    slope_count,
                    Arc::from(reconstructor),
                )?,
                NoWorkspace,
            ))
        }
        prepare_error = PreparedGemvError;
        process |plan, _workspace| {
            plan.multiply(reconstructed, slopes)
        }
        process_error = GemvError;
    }
}
