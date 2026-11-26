//! Dense reconstruction of a selected I4Q X/Y signal.
use crate::{GemvError, PreparedGemv, PreparedGemvError};
use alloc::{boxed::Box, string::String, sync::Arc, vec, vec::Vec};
/// Disposable canonical component-major ordering scratch.
#[derive(Debug)]
pub struct PyramidReconstructionWorkspace {
    ordered_signal: Box<[f32]>,
}
crate::filtergraph_algorithm! {
    /// Map selected I4Q coordinates through canonical X-then-Y matrix columns.
    pub PyramidReconstructorF32 {
        label="pwfs-reconstructor-f32";
        configuration PyramidReconstructorF32Configuration {
            reconstructed_count:usize=GraphRebuild;selected_coordinate_count:usize=GraphRebuild;
            initial_reconstructor:Option<Vec<f32>>=Construction;reconstructed_schema:String=GraphRebuild;
        }
        plan=PreparedGemv<f32>;workspace=PyramidReconstructionWorkspace;properties=none;
        dimensions|plan,_workspace|{rows=plan.rows();columns=plan.columns();selected=columns/2;}
        inputs{signal:f32[selected,2]="signal",schema=crate::schemas::PYRAMID_I4Q_SIGNAL_V1;}
        outputs{reconstructed:f32[rows]="reconstructed",schema=configuration.reconstructed_schema.as_str(),metadata_from="signal";}
        parameters{reconstructor:f32[rows,columns]="reconstructor",schema=crate::schemas::PYRAMID_I4Q_RECONSTRUCTOR_V1,replace|plan,values|{PreparedGemv::new(plan.rows(),plan.columns(),Arc::from(values))};}
        parameter_error=PreparedGemvError;
        prepare|configuration|{
            let rows=configuration.reconstructed_count;let columns=configuration.selected_coordinate_count.checked_mul(2).ok_or(PreparedGemvError::ExtentOverflow)?;
            if rows==0||columns==0{return Err(PreparedGemvError::EmptyDimension);}
            let count=rows.checked_mul(columns).filter(|&n|n<=u32::MAX as usize/core::mem::size_of::<f32>()).ok_or(PreparedGemvError::ExtentOverflow)?;
            let values=configuration.initial_reconstructor.unwrap_or_else(||vec![0.0;count]);
            Ok((PreparedGemv::new(rows,columns,Arc::from(values))?,PyramidReconstructionWorkspace{ordered_signal:vec![0.0;columns].into_boxed_slice()}))
        }
        prepare_error=PreparedGemvError;
        process|plan,workspace|{
            if signal.len()!=plan.columns(){return Err(GemvError::InputExtent);}
            if reconstructed.len()!=plan.rows(){return Err(GemvError::OutputExtent);}
            if signal.iter().any(|v|!v.is_finite()){return Err(GemvError::NonFiniteInput);}
            let selected=plan.columns()/2;
            for component in 0..2{for coordinate in 0..selected{workspace.ordered_signal[component*selected+coordinate]=signal[coordinate*2+component];}}
            plan.multiply(reconstructed,&workspace.ordered_signal)
        }
        process_error=GemvError;
    }
}
