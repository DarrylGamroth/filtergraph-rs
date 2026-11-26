//! Sparse measurements from successive detector-row pyramid blocks.
use super::row_block::{RowBlockInfo, RowBlockResult};
use crate::{
    AlgorithmProcessResult, PyramidPixelImagePlan, PyramidPixelRowExecutionError,
    PyramidPixelRowPlan, PyramidPixelRowPlanError, PyramidPixelRowPreparationError,
    PyramidPixelRowProcessError, PyramidPixelRowWorkspace, SampleMetadata,
};
use alloc::{boxed::Box, string::String, vec, vec::Vec};
use core::{convert::Infallible, fmt};
/// Invalid caller storage or row-block sensing failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyramidMeasurementBlockError {
    /// A caller-owned measurement, column, or count extent is incompatible.
    Extent,
    /// Canonical row-block sensing rejected the call.
    Source(PyramidPixelRowProcessError),
}
impl fmt::Display for PyramidMeasurementBlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Extent => f.write_str("pyramid measurement-block extent is invalid"),
            Self::Source(e) => e.fmt(f),
        }
    }
}
impl core::error::Error for PyramidMeasurementBlockError {}
/// Single-owner row assembly and disposable sparse-publication scratch.
#[derive(Debug)]
pub struct PyramidMeasurementBlockWorkspace {
    source: PyramidPixelRowWorkspace<f32>,
    terminal_pixels: Box<[f32]>,
    terminal_mean: f32,
    measurements: Box<[f32]>,
    columns: Box<[u32]>,
}
impl PyramidMeasurementBlockWorkspace {
    /// Allocate fixed scratch for the prepared canonical row plan.
    pub fn new(plan: &PyramidPixelRowPlan<f32>) -> Self {
        let n = plan.image().normalization().output_pixel_count();
        Self {
            source: plan.workspace(),
            terminal_pixels: vec![0.0; n].into_boxed_slice(),
            terminal_mean: 0.0,
            measurements: vec![0.0; n].into_boxed_slice(),
            columns: vec![0; n].into_boxed_slice(),
        }
    }
    /// Restore construction normalization and abandon incomplete assembly.
    pub fn reset(&mut self) {
        self.source.reset();
    }
    /// Abandon incomplete assembly, retaining committed normalization.
    pub fn plan_adopted(&mut self) {
        self.source.plan_adopted();
    }
    /// Return the committed gain from the previous complete detector frame.
    pub fn reciprocal_previous_frame_mean(&self) -> f32 {
        self.source.reciprocal_previous_frame_mean()
    }
}
pub(crate) fn configured_pyramid_mask(
    values: Option<Vec<bool>>,
    size: (usize, usize),
) -> Vec<bool> {
    // These older pyramid declarations explicitly use row-major construction masks.
    values.unwrap_or_else(|| vec![true; size.0 * size.1])
}

fn process(
    measurements: &mut [f32],
    columns: &mut [u32],
    count: &mut [u32],
    plan: &PyramidPixelRowPlan<f32>,
    workspace: &mut PyramidMeasurementBlockWorkspace,
    input: &[f32],
    sample: Option<SampleMetadata>,
) -> Result<AlgorithmProcessResult, PyramidMeasurementBlockError> {
    let n = plan.image().normalization().output_pixel_count();
    if measurements.len() != n || columns.len() != n || count.len() != 1 {
        return Err(PyramidMeasurementBlockError::Extent);
    }
    let Some(sample) = sample else {
        workspace.plan_adopted();
        return Err(PyramidMeasurementBlockError::Source(
            PyramidPixelRowProcessError::MissingSampleMetadata,
        ));
    };
    let first_row = usize::try_from(sample.offset()).map_err(|_| {
        PyramidMeasurementBlockError::Source(PyramidPixelRowProcessError::OffsetOverflow)
    })?;
    let mut staged_count = 0;
    let result = plan.process_block_with_ready_pixels(
        &mut workspace.terminal_pixels,
        &mut workspace.terminal_mean,
        &mut workspace.source,
        input,
        RowBlockInfo {
            sequence: sample.sequence(),
            first_row,
            terminal: sample.is_terminal(),
            discontinuity: sample.is_discontinuous(),
            corrupted: sample.is_corrupted(),
        },
        |_, pixels, ranges| {
            for range in ranges {
                for column in range.clone() {
                    workspace.measurements[staged_count] = pixels[column];
                    workspace.columns[staged_count] = column as u32;
                    staged_count += 1;
                }
            }
            Ok::<(), Infallible>(())
        },
    );
    let result = match result {
        Ok(result) => result,
        Err(PyramidPixelRowExecutionError::Algorithm(e)) => {
            return Err(PyramidMeasurementBlockError::Source(
                PyramidPixelRowProcessError::Algorithm(e),
            ));
        }
        Err(PyramidPixelRowExecutionError::Executor(e)) => match e {},
    };
    if matches!(result, RowBlockResult::Dropped(_)) {
        return Ok(AlgorithmProcessResult::deferred());
    }
    measurements[..staged_count].copy_from_slice(&workspace.measurements[..staged_count]);
    columns[..staged_count].copy_from_slice(&workspace.columns[..staged_count]);
    count[0] = staged_count as u32;
    let discontinuity =
        sample.is_discontinuous() || (first_row == 0 && workspace.source.frame_discontinuity());
    Ok(AlgorithmProcessResult::complete_with_metadata(
        sample.with_discontinuity(discontinuity),
    ))
}
crate::filtergraph_algorithm! {
    /// Publish newly immutable normalized pupil pixels with canonical dense columns.
    pub PyramidMeasurementBlockF32 {
        label="pyramid-measurement-block-f32";
        configuration PyramidMeasurementBlockF32Configuration {
            image_size:(usize,usize)=GraphRebuild;row_block_rows:usize=GraphRebuild;pupil_size:(usize,usize)=GraphRebuild;
            pupil_origins:Vec<[usize;2]>=GraphRebuild;pupil_mask:Option<Vec<bool>>=Construction;
            measurements_schema:String=GraphRebuild;column_indices_schema:String=GraphRebuild;measurement_count_schema:String=GraphRebuild;
        }
        plan=PyramidPixelRowPlan<f32>;workspace=PyramidMeasurementBlockWorkspace;properties=none;
        dimensions|plan,_workspace|{rows=plan.block_rows();columns=plan.image().pupil_regions().image_size().1;pupil_rows=plan.image().pupil_regions().region_size().0;pupil_columns=plan.image().pupil_regions().region_size().1;capacity=plan.image().normalization().output_pixel_count();rate=u32::try_from(plan.image().pupil_regions().image_size().0/rows).map_err(|_|crate::DeclarationPreparationError::Format(crate::FormatError::Rate))?;}
        inputs{row_block:f32[rows,columns]="row-block",schema=crate::schemas::CALIBRATED_PIXEL_ROW_BLOCK_V1,sample=row_sample,rate=[rate,1];}
        outputs{
            measurements:f32[capacity]="measurements",schema=configuration.measurements_schema.as_str(),metadata_from="row-block",publication=conditional,rate=[rate,1];
            columns:u32[capacity]="columns",schema=configuration.column_indices_schema.as_str(),metadata_from="row-block",publication=conditional,rate=[rate,1];
            count:u32[1]="count",schema=configuration.measurement_count_schema.as_str(),metadata_from="row-block",publication=conditional,rate=[rate,1];
        }
        parameters{pupil_origins:u32[4,2]="pupil-origins",schema=crate::schemas::REGION_ORIGINS_V1,replace|plan,values|{plan.with_u32_pupil_origins(values)};
            pupil_mask:bool[pupil_rows,pupil_columns]="pupil-mask",schema=crate::schemas::PYRAMID_PUPIL_MASK_V1,replace|plan,values|{plan.with_pupil_mask(values)};}
        parameter_error=PyramidPixelRowPreparationError;
        prepare|configuration|{
            for (size,bytes) in [(configuration.image_size,4),(configuration.pupil_size,16)]{size.0.checked_mul(size.1).filter(|&n|n<=u32::MAX as usize/bytes).ok_or(PyramidPixelRowPreparationError::Composition(PyramidPixelRowPlanError::ExtentOverflow))?;}
            let origins=configuration.pupil_origins.iter().map(|p|(p[0],p[1])).collect::<Vec<_>>();
            let image=PyramidPixelImagePlan::from_detector_pupils(configuration.image_size,configuration.pupil_size,&origins,configured_pyramid_mask(configuration.pupil_mask,configuration.pupil_size))?;
            let plan=PyramidPixelRowPlan::new(image,configuration.row_block_rows)?;let workspace=PyramidMeasurementBlockWorkspace::new(&plan);Ok((plan,workspace))
        }
        prepare_error=PyramidPixelRowPreparationError;
        process|plan,workspace|{process(measurements,columns,count,plan,workspace,row_block,row_sample)}
        process_error=PyramidMeasurementBlockError;
        reset|_plan,workspace|{workspace.reset();}
        plan_adopted|_plan,workspace|{workspace.plan_adopted();}
    }
}
