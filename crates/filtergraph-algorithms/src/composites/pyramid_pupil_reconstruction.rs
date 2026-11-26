//! Dense pupil-major reconstruction, distinct from differential I4Q sensing.
use crate::{GemvError, NoWorkspace, PYRAMID_PUPIL_COUNT, PreparedGemv, PreparedGemvError};
use alloc::{string::String, sync::Arc, vec, vec::Vec};
crate::filtergraph_algorithm! {
    /// Reconstruct an ordered four-pupil selected-pixel matrix.
    pub PyramidPupilReconstructorF32 {
        label="pwfs-pupil-reconstructor-f32";
        configuration PyramidPupilReconstructorF32Configuration {
            reconstructed_count:usize=GraphRebuild;selected_pixel_count:usize=GraphRebuild;
            initial_reconstructor:Option<Vec<f32>>=Construction;reconstructed_schema:String=GraphRebuild;
        }
        plan=PreparedGemv<f32>;operation=DenseF32;workspace=NoWorkspace;properties=none;
        dimensions|plan,_workspace|{rows=plan.rows();columns=plan.columns();selected=columns/PYRAMID_PUPIL_COUNT;}
        inputs{reconstruction_pixels:f32[PYRAMID_PUPIL_COUNT,selected]="reconstruction-pixels",schema=crate::schemas::PYRAMID_RECONSTRUCTION_PIXELS_V1;}
        outputs{reconstructed:f32[rows]="reconstructed",schema=configuration.reconstructed_schema.as_str(),metadata_from="reconstruction-pixels";}
        parameters{reconstructor:f32[rows,columns]="reconstructor",schema=crate::schemas::PWFS_RECONSTRUCTOR_V1,replace|plan,values|{PreparedGemv::new_pupil(plan.rows(),plan.columns(),Arc::from(values))};}
        parameter_error=PreparedGemvError;
        prepare|configuration|{
            let rows=configuration.reconstructed_count;
            let columns=configuration.selected_pixel_count.checked_mul(PYRAMID_PUPIL_COUNT).ok_or(PreparedGemvError::ExtentOverflow)?;
            if rows==0||columns==0{return Err(PreparedGemvError::EmptyDimension);}
            let count=rows.checked_mul(columns).filter(|&n|n<=u32::MAX as usize/core::mem::size_of::<f32>()).ok_or(PreparedGemvError::ExtentOverflow)?;
            let values=configuration.initial_reconstructor.unwrap_or_else(||vec![0.0;count]);
            Ok((PreparedGemv::new_pupil(rows,columns,Arc::from(values))?,NoWorkspace))
        }
        prepare_error=PreparedGemvError;
        process|plan,_workspace|{plan.multiply_pupil(reconstructed,reconstruction_pixels)}
        process_error=GemvError;
    }
}
