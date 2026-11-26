//! Sparse pupil measurements in detector-readout completion order.
use super::pyramid_measurement_block::configured_pyramid_mask;
use crate::{
    AlgorithmProcessResult, DetectorReadoutBlockInfo, DetectorReadoutPlan, DetectorReadoutResult,
    PreparedGemv, PyramidPixelImagePlan, PyramidReadoutReconstructionError,
    PyramidReadoutReconstructionOutput, PyramidReadoutReconstructionPlan,
    PyramidReadoutReconstructionPlanError, PyramidReadoutReconstructionPreparationError,
    PyramidReadoutReconstructionWorkspace, SampleMetadata,
};
use alloc::{boxed::Box, string::String, sync::Arc, vec, vec::Vec};
/// Prepared detector mapping, pupil normalization, and completion groups.
#[derive(Clone, Debug)]
pub struct PyramidReadoutMeasurementBlockPlan {
    source: PyramidReadoutReconstructionPlan<f32>,
}
impl PyramidReadoutMeasurementBlockPlan {
    /// Compose canonical pyramid sensing with its zero-row terminal reconstructor.
    pub fn new(
        image: PyramidPixelImagePlan<f32>,
        readout: DetectorReadoutPlan,
    ) -> Result<Self, PyramidReadoutReconstructionPreparationError> {
        let n = image.normalization().output_pixel_count();
        Ok(Self {
            source: PyramidReadoutReconstructionPlan::new(
                image,
                PreparedGemv::new(1, n, Arc::from(vec![0.0; n]))?,
                readout,
            )?,
        })
    }
    /// Return the canonical sensing/readout preparation.
    pub const fn source(&self) -> &PyramidReadoutReconstructionPlan<f32> {
        &self.source
    }
    /// Allocate bounded mutable state for one execution owner.
    pub fn workspace(&self) -> PyramidReadoutMeasurementBlockWorkspace {
        let n = self.source.image().normalization().output_pixel_count();
        PyramidReadoutMeasurementBlockWorkspace {
            source: self.source.workspace(),
            terminal_pixels: vec![0.0; n].into_boxed_slice(),
            terminal_mean: 0.0,
            terminal_reconstructed: [0.0],
        }
    }
    /// Consume one block; only the valid measurement prefix changes on publication.
    pub fn process_block(
        &self,
        measurements: &mut [f32],
        columns: &mut [u32],
        count: &mut [u32],
        workspace: &mut PyramidReadoutMeasurementBlockWorkspace,
        input: &[f32],
        sample: SampleMetadata,
    ) -> Result<AlgorithmProcessResult, PyramidReadoutReconstructionError> {
        let n = self.source.image().normalization().output_pixel_count();
        if measurements.len() != n || columns.len() != n || count.len() != 1 {
            return Err(PyramidReadoutReconstructionError::PixelOutputExtent {
                expected: n,
                actual: measurements.len(),
            });
        }
        let ordinal = usize::try_from(sample.offset())
            .map_err(|_| PyramidReadoutReconstructionError::OrdinalOverflow)?;
        let result = self.source.process_block(
            PyramidReadoutReconstructionOutput::new(
                &mut workspace.terminal_pixels,
                &mut workspace.terminal_mean,
                &mut workspace.terminal_reconstructed,
            ),
            &mut workspace.source,
            input,
            DetectorReadoutBlockInfo {
                sequence: sample.sequence(),
                ordinal,
                terminal: sample.is_terminal(),
                discontinuity: sample.is_discontinuous(),
                corrupted: sample.is_corrupted(),
            },
        )?;
        if matches!(result, DetectorReadoutResult::Dropped(_)) {
            return Ok(AlgorithmProcessResult::deferred());
        }
        let ready = self
            .source
            .reconstructor()
            .input_indices(ordinal)
            .expect("accepted detector ordinal");
        let pixels = workspace.source.reconstruction_pixels();
        for (index, &column) in ready.iter().enumerate() {
            measurements[index] = pixels[column];
            columns[index] = column as u32;
        }
        count[0] = ready.len() as u32;
        let discontinuity =
            sample.is_discontinuous() || (ordinal == 0 && workspace.source.frame_discontinuity());
        Ok(AlgorithmProcessResult::complete_with_metadata(
            sample.with_discontinuity(discontinuity),
        ))
    }
}
/// Single-owner readout assembly and fixed terminal scratch.
#[derive(Debug)]
pub struct PyramidReadoutMeasurementBlockWorkspace {
    source: PyramidReadoutReconstructionWorkspace<f32>,
    terminal_pixels: Box<[f32]>,
    terminal_mean: f32,
    terminal_reconstructed: [f32; 1],
}
impl PyramidReadoutMeasurementBlockWorkspace {
    /// Restore construction normalization and abandon incomplete assembly.
    pub fn reset(&mut self) {
        self.source.reset();
    }
    /// Abandon incomplete assembly, retaining committed normalization history.
    pub fn plan_adopted(&mut self) {
        self.source.plan_adopted();
    }
    /// Return the gain from the previous complete detector frame.
    pub fn reciprocal_previous_frame_mean(&self) -> f32 {
        self.source.reciprocal_previous_frame_mean()
    }
}
crate::filtergraph_algorithm! {
    /// Publish newly ready normalized pixels with canonical reconstructor columns.
    pub PyramidReadoutMeasurementBlockF32 {
        label="pyramid-readout-measurement-block-f32";
        configuration PyramidReadoutMeasurementBlockF32Configuration {
            detector_size:(usize,usize)=GraphRebuild;region_lines:usize=GraphRebuild;block_lines:usize=GraphRebuild;region_samples:usize=GraphRebuild;
            detector_origins:Vec<[usize;2]>=GraphRebuild;detector_line_steps:Vec<[i32;2]>=GraphRebuild;detector_sample_steps:Vec<[i32;2]>=GraphRebuild;
            pupil_size:(usize,usize)=GraphRebuild;pupil_origins:Vec<[usize;2]>=GraphRebuild;pupil_mask:Option<Vec<bool>>=Construction;
            measurements_schema:String=GraphRebuild;column_indices_schema:String=GraphRebuild;measurement_count_schema:String=GraphRebuild;
        }
        plan=PyramidReadoutMeasurementBlockPlan;workspace=PyramidReadoutMeasurementBlockWorkspace;properties=none;
        dimensions|plan,_workspace|{regions=plan.source().readout().regions().len();lines=plan.source().readout().block_lines();samples=plan.source().readout().region_samples();capacity=plan.source().image().normalization().output_pixel_count();rate=u32::try_from(plan.source().readout().blocks()).map_err(|_|crate::DeclarationPreparationError::Format(crate::FormatError::Rate))?;}
        inputs{readout_block:f32[regions,lines,samples]="readout-block",schema=crate::schemas::CALIBRATED_PIXEL_READOUT_BLOCK_V1,sample=readout_sample,rate=[rate,1];}
        outputs{measurements:f32[capacity]="measurements",schema=configuration.measurements_schema.as_str(),metadata_from="readout-block",publication=conditional,rate=[rate,1];
            columns:u32[capacity]="columns",schema=configuration.column_indices_schema.as_str(),metadata_from="readout-block",publication=conditional,rate=[rate,1];
            count:u32[1]="count",schema=configuration.measurement_count_schema.as_str(),metadata_from="readout-block",publication=conditional,rate=[rate,1];}
        prepare|configuration|{
            configuration.detector_size.0.checked_mul(configuration.detector_size.1).filter(|&n|n<=u32::MAX as usize/4).ok_or(PyramidReadoutReconstructionPreparationError::Composition(PyramidReadoutReconstructionPlanError::ReconstructorExtentOverflow))?;
            configuration.pupil_size.0.checked_mul(configuration.pupil_size.1).filter(|&n|n<=u32::MAX as usize/16).ok_or(PyramidReadoutReconstructionPreparationError::Composition(PyramidReadoutReconstructionPlanError::ReconstructorExtentOverflow))?;
            let readout=DetectorReadoutPlan::from_i32_coordinate_steps(configuration.detector_size,configuration.region_lines,configuration.block_lines,configuration.region_samples,&configuration.detector_origins,&configuration.detector_line_steps,&configuration.detector_sample_steps)?;
            let origins=configuration.pupil_origins.iter().map(|p|(p[0],p[1])).collect::<Vec<_>>();
            let image=PyramidPixelImagePlan::from_detector_pupils(configuration.detector_size,configuration.pupil_size,&origins,configured_pyramid_mask(configuration.pupil_mask,configuration.pupil_size))?;
            let plan=PyramidReadoutMeasurementBlockPlan::new(image,readout)?;let workspace=plan.workspace();Ok((plan,workspace))
        }
        prepare_error=PyramidReadoutReconstructionPreparationError;
        process|plan,workspace|{let Some(sample)=readout_sample else{workspace.plan_adopted();return Err(PyramidReadoutReconstructionError::MissingSampleMetadata);};plan.process_block(measurements,columns,count,workspace,readout_block,sample)}
        process_error=PyramidReadoutReconstructionError;
        reset|_plan,workspace|{workspace.reset();}
        plan_adopted|_plan,workspace|{workspace.plan_adopted();}
    }
}
