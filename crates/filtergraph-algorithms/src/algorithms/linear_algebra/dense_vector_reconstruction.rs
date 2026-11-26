//! Explicitly ordered dense reconstruction with caller-supplied scientific schemas.
use crate::{GemvError, PreparedGemv, PreparedGemvError};
use alloc::{string::String, sync::Arc, vec, vec::Vec};
/// Execution scratch for a prepared Float32 dense reconstructor.
pub struct DenseVectorReconstructorWorkspace {
    #[cfg(feature = "blas")]
    wide: Option<filtergraph_blas::SgemvWorkspace>,
}

impl DenseVectorReconstructorWorkspace {
    fn new(_plan: &PreparedGemv<f32>) -> Self {
        Self {
            #[cfg(feature = "blas")]
            wide: filtergraph_blas::SgemvWorkspace::for_wide_matrix(_plan.rows(), _plan.columns()),
        }
    }

    fn multiply(
        &mut self,
        plan: &PreparedGemv<f32>,
        output: &mut [f32],
        input: &[f32],
    ) -> Result<(), GemvError> {
        #[cfg(feature = "blas")]
        if let Some(workspace) = &mut self.wide {
            return plan.multiply_with(output, input, |output, input| {
                workspace.multiply(plan.matrix(), input, output);
            });
        }
        plan.multiply(output, input)
    }
}

crate::filtergraph_algorithm! {
    /// Dense reconstruction without implicit ordering, sign, scale or unit conversion.
    pub DenseVectorReconstructorF32 {
        label="dense-vector-reconstructor-f32";
        configuration DenseVectorReconstructorF32Configuration {
            input_count:usize=GraphRebuild;
            output_count:usize=GraphRebuild;
            initial_reconstructor:Option<Vec<f32>>=Construction;
            input_schema:String=GraphRebuild;
            output_schema:String=GraphRebuild;
            reconstructor_schema:String=GraphRebuild;
        }
        plan=PreparedGemv<f32>;operation=DenseF32;workspace=DenseVectorReconstructorWorkspace;properties=none;
        dimensions|plan,_workspace|{inputs_count=plan.columns();outputs_count=plan.rows();}
        inputs{ordered_input:f32[inputs_count]="ordered-input",schema=configuration.input_schema.as_str(), optional_sample_metadata = ["chopped_image_kind", "trigger_state", "acquisition_identity", "exposure_duration_nanoseconds"];}
        outputs{output:f32[outputs_count]="output",schema=configuration.output_schema.as_str(),metadata_from="ordered-input";}
        parameters{reconstructor:f32[outputs_count,inputs_count]="reconstructor",schema=configuration.reconstructor_schema.as_str(),replace|plan,values|{PreparedGemv::new(plan.rows(),plan.columns(),Arc::from(values))};}
        parameter_error=PreparedGemvError;
        prepare|configuration|{
            let rows=configuration.output_count;let columns=configuration.input_count;
            if rows==0||columns==0{return Err(PreparedGemvError::EmptyDimension);}
            let count=rows.checked_mul(columns).filter(|&n|n<=u32::MAX as usize/core::mem::size_of::<f32>()).ok_or(PreparedGemvError::ExtentOverflow)?;
            let matrix=configuration.initial_reconstructor.unwrap_or_else(||vec![0.0;count]);
            let plan = PreparedGemv::new(rows, columns, Arc::from(matrix))?;
            let workspace = DenseVectorReconstructorWorkspace::new(&plan);
            Ok((plan, workspace))
        }
        prepare_error=PreparedGemvError;
        process|plan,workspace|{workspace.multiply(plan,output,ordered_input)}
        process_error=GemvError;
    }
}
