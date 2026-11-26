//! Dense SCC selected-dark-hole pixels to named controller residual-error coordinates.
use crate::{GemvError, NoWorkspace, PreparedGemv, PreparedGemvError};
use alloc::{sync::Arc, vec, vec::Vec};
crate::filtergraph_algorithm! {
    /// Reconstruct controller residual error, with selected dark-hole pixels in matrix columns.
    pub SccReconstructorF32 {
        label="scc-reconstructor-f32";
        configuration SccReconstructorF32Configuration {
            controller_coordinate_count:usize=GraphRebuild;selected_pixel_count:usize=GraphRebuild;
            initial_reconstructor:Option<Vec<f32>>=Construction;
        }
        plan=PreparedGemv<f32>;operation=DenseF32;workspace=NoWorkspace;properties=none;
        dimensions |plan,_workspace| {controller_coordinate_count=plan.rows();selected_pixel_count=plan.columns();}
        inputs {selected_residuals:f32 [selected_pixel_count]="selected-residuals",schema=crate::schemas::SCC_SELECTED_PIXEL_RESIDUALS_V1;}
        outputs {controller_residual_error:f32 [controller_coordinate_count]="controller-residual-error",schema=crate::schemas::CONTROLLER_RESIDUAL_ERROR_V1,metadata_from="selected-residuals";}
        parameters {reconstructor:f32 [controller_coordinate_count,selected_pixel_count]="reconstructor",schema=crate::schemas::SCC_RECONSTRUCTOR_V1,replace |plan,values| {PreparedGemv::new(plan.rows(),plan.columns(),Arc::from(values))};}
        parameter_error=PreparedGemvError;
        prepare |configuration| {
            let rows=configuration.controller_coordinate_count;let columns=configuration.selected_pixel_count;
            if rows==0||columns==0{return Err(PreparedGemvError::EmptyDimension);}
            let count=rows.checked_mul(columns).filter(|&count|count<=u32::MAX as usize/core::mem::size_of::<f32>()).ok_or(PreparedGemvError::ExtentOverflow)?;
            let matrix=configuration.initial_reconstructor.unwrap_or_else(||vec![0.0;count]);
            Ok((PreparedGemv::new(rows,columns,Arc::from(matrix))?,NoWorkspace))
        }
        prepare_error=PreparedGemvError;
        process |plan,_workspace| {plan.multiply(controller_residual_error,selected_residuals)}
        process_error=GemvError;
    }
}
