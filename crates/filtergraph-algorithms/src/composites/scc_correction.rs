//! Transactional rolling SCC correction with explicit acquisition identity,
//! optical role, exposure duration, controller calibration, and phase response.
//! Each accepted exposure must have used the last accepted command. If downstream
//! publication rejects a locally accepted update, stop processing and reset before
//! resuming; this primitive cannot roll back an already accepted exposure.
use crate::{
    AlgorithmPlan, ChoppedImageDifferenceInput, ChoppedImageDifferenceOutput,
    ChoppedImageDifferencePlan, ChoppedImageDifferenceStatus, ChoppedImageDifferenceWorkspace,
    ChoppedImageKind, LeakyIntegratorPlan, LeakyIntegratorWorkspace, PreparedGemv,
    PropertyConstraint, PropertyValue, SampleAcquisitionIdentity, SampleMetadata, ScalarOrVector,
    SccPixelResidualPlan,
};
use alloc::{string::String, sync::Arc, vec, vec::Vec};
use core::{fmt, mem};
/// Invalid coherent SCC controller calibration or integrator policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccCorrectionPlanError {
    /// Positive image/controller/matrix byte extents must fit UInt32.
    Extent,
    /// Calibration arrays or controller coordinate order are incompatible.
    Calibration,
    /// Command unit and image schema must be explicit.
    Units,
    /// Reconstructor sign must be command_error or correction.
    Sign,
    /// Exposure duration must be positive Int64 nanoseconds.
    Exposure,
    /// Bias and limits must be finite, ordered, and include the bias.
    Limits,
    /// Gain must be nonnegative and finite; pole must lie in [0,1].
    Integrator,
}
impl fmt::Display for SccCorrectionPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid SCC correction preparation: {self:?}")
    }
}
impl core::error::Error for SccCorrectionPlanError {}
/// Expected acquisition or numerical rejection; no command or history is adopted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccCorrectionError {
    /// Caller frame/command or execution dimensions are incompatible.
    DimensionMismatch,
    /// Acquisition identity or optical role is absent.
    MetadataMissing,
    /// Supplied exposure duration differs from calibration.
    ExposureMismatch,
    /// Acquisition sequence repeats/decreases, or disagrees with sample sequence.
    SequenceInvalid,
    /// Frame corruption, nonfinite image, or selected residual invalidates input.
    NonFiniteInput,
    /// Reconstruction, phase compensation, integration, or command is nonfinite.
    NonFiniteCorrection,
    /// The complete candidate command violates calibrated limits.
    CommandOutOfBounds,
}
impl fmt::Display for SccCorrectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SCC correction disposition: {self:?}")
    }
}
impl core::error::Error for SccCorrectionError {}
#[derive(Clone, Debug)]
struct SccCalibration {
    frame_shape: (usize, usize),
    residual: SccPixelResidualPlan<f32>,
    reconstructor: PreparedGemv<f32>,
    fringed_response: PreparedGemv<f32>,
    unfringed_response: PreparedGemv<f32>,
    correction_sign: f32,
    exposure_duration_nanoseconds: u64,
    bias_command: Vec<f32>,
    lower_limits: Vec<f32>,
    upper_limits: Vec<f32>,
}
/// Immutable coherent SCC calibration and runtime integrator coefficients.
#[derive(Clone, Debug)]
pub struct SccCorrectionPlan {
    calibration: Arc<SccCalibration>,
    difference: ChoppedImageDifferencePlan<f32>,
    integrator: LeakyIntegratorPlan<f32>,
    gain: f32,
    pole: f32,
}
#[derive(Clone, Debug)]
struct SccCorrectionState {
    difference: ChoppedImageDifferenceWorkspace<f32>,
    committed_exposure_command: Vec<f32>,
    candidate_exposure_command: Vec<f32>,
    committed_command: Vec<f32>,
    candidate_command: Vec<f32>,
    integrator: LeakyIntegratorWorkspace<f32>,
    committed_identity: Option<SampleAcquisitionIdentity>,
    candidate_identity: Option<SampleAcquisitionIdentity>,
    candidate_clear_history: bool,
    candidate_ready: bool,
}
#[derive(Clone, Debug)]
struct SccCorrectionWorkspace {
    image: Vec<f32>,
    image_validity: Vec<bool>,
    difference: Vec<f32>,
    difference_validity: Vec<bool>,
    pair_status: ChoppedImageDifferenceStatus,
    selected_residuals: Vec<f32>,
    selected_validity: Vec<bool>,
    valid_fraction: f32,
    reconstructed: Vec<f32>,
    correction: Vec<f32>,
    command_difference: Vec<f32>,
    phase_correction: Vec<f32>,
}
/// Fixed retained pairing/controller history and disposable frame scratch.
#[derive(Clone, Debug)]
pub struct SccCorrectionExecution {
    state: SccCorrectionState,
    workspace: SccCorrectionWorkspace,
    initial_command: Vec<f32>,
}
impl SccCorrectionExecution {
    /// Last successfully published complete controller command.
    pub fn command(&self) -> &[f32] {
        &self.state.committed_command
    }
    /// Command used for the most recently retained pure exposure.
    pub fn exposure_command(&self) -> &[f32] {
        &self.state.committed_exposure_command
    }
    /// Committed signed integrator correction.
    pub fn integrator_state(&self) -> &[f32] {
        self.state.integrator.state()
    }
    /// Last successfully adopted acquisition identity.
    pub const fn acquisition_identity(&self) -> Option<SampleAcquisitionIdentity> {
        self.state.committed_identity
    }
    /// Whether the rolling difference retains a pure exposure.
    pub fn is_primed(&self) -> bool {
        self.state.difference.is_primed()
    }
    fn discard(&mut self) {
        self.state.candidate_ready = false;
        self.state.difference.discard();
        self.state.integrator.discard();
    }
    fn commit(&mut self) {
        let s = &mut self.state;
        if !s.candidate_ready {
            return;
        }
        if s.candidate_clear_history {
            s.difference.reset();
        } else {
            s.difference.commit();
            mem::swap(
                &mut s.committed_exposure_command,
                &mut s.candidate_exposure_command,
            );
        }
        mem::swap(&mut s.committed_command, &mut s.candidate_command);
        s.committed_identity = s.candidate_identity;
        s.integrator.commit();
        s.candidate_ready = false;
    }
    /// Restore bias commands, zero integrator history, and unprimed acquisition state.
    pub fn reset(&mut self) {
        let s = &mut self.state;
        s.committed_identity = None;
        s.candidate_identity = None;
        s.candidate_clear_history = false;
        s.candidate_ready = false;
        s.difference.reset();
        for command in [
            &mut s.committed_exposure_command,
            &mut s.candidate_exposure_command,
            &mut s.committed_command,
            &mut s.candidate_command,
        ] {
            command.copy_from_slice(&self.initial_command);
        }
        s.integrator.reset();
        let w = &mut self.workspace;
        for values in [
            &mut w.image,
            &mut w.difference,
            &mut w.selected_residuals,
            &mut w.reconstructed,
            &mut w.correction,
            &mut w.command_difference,
            &mut w.phase_correction,
        ] {
            values.fill(0.0);
        }
        w.valid_fraction = 0.0;
        w.selected_validity.fill(false);
        w.difference_validity.fill(false);
        w.pair_status = ChoppedImageDifferenceStatus::Priming;
    }
}
fn replace_properties(
    plan: &SccCorrectionPlan,
    gain: f32,
    pole: f32,
) -> Result<SccCorrectionPlan, SccCorrectionPlanError> {
    if !gain.is_finite() || gain < 0.0 {
        return Err(SccCorrectionPlanError::Integrator);
    }
    let integrator = LeakyIntegratorPlan::new(plan.calibration.correction_sign * gain, pole)
        .map_err(|_| SccCorrectionPlanError::Integrator)?;
    Ok(SccCorrectionPlan {
        calibration: Arc::clone(&plan.calibration),
        difference: plan.difference,
        integrator,
        gain,
        pole,
    })
}
crate::algorithm_properties! {
    impl SccCorrectionPlan {
        plan_error=SccCorrectionPlanError;replace=replace_properties;
        properties {
            0 => gain:Float {name="gain";description="SCC input gain per usable rolling exposure";unit="1";update=Runtime;default=0.1;constraint=PropertyConstraint::Range{minimum:PropertyValue::Float(0.0),maximum:PropertyValue::Float(f32::MAX)};}
            1 => pole:Float {name="pole";description="SCC integrator pole per usable rolling exposure";unit="1";update=Runtime;default=1.0;constraint=PropertyConstraint::Range{minimum:PropertyValue::Float(0.0),maximum:PropertyValue::Float(1.0)};}
        }
    }
}
impl SccCorrectionPlan {
    fn stage_process(
        &self,
        command: &mut [f32],
        execution: &mut SccCorrectionExecution,
        frame: &[f32],
        metadata: Option<SampleMetadata>,
    ) -> Result<(), SccCorrectionError> {
        use SccCorrectionError as E;
        execution.discard();
        let s = &mut execution.state;
        let w = &mut execution.workspace;
        let c = &self.calibration;
        let count = c.bias_command.len();
        if command.len() != count || frame.len() != c.residual.pixel_count() {
            return Err(E::DimensionMismatch);
        }
        let metadata = metadata.ok_or(E::MetadataMissing)?;
        let kind = metadata
            .chopped_image_kind()
            .filter(|kind| *kind != ChoppedImageKind::Absent)
            .ok_or(E::MetadataMissing)?;
        if metadata
            .exposure_duration_nanoseconds()
            .ok_or(E::MetadataMissing)?
            != c.exposure_duration_nanoseconds
        {
            return Err(E::ExposureMismatch);
        }
        let identity = metadata.acquisition_identity().ok_or(E::MetadataMissing)?;
        if metadata.is_corrupted() {
            return Err(E::NonFiniteInput);
        }
        if s.committed_identity.is_some_and(|previous| {
            previous.domain() == identity.domain() && identity.generation() < previous.generation()
        }) {
            return Err(E::SequenceInvalid);
        }
        let same_cohort = s.committed_identity.is_some_and(|previous| {
            previous.domain() == identity.domain() && previous.generation() == identity.generation()
        });
        if same_cohort && identity.sequence() <= s.committed_identity.unwrap().sequence() {
            return Err(E::SequenceInvalid);
        }
        if frame.iter().any(|v| !v.is_finite()) {
            return Err(E::NonFiniteInput);
        }
        s.candidate_identity = Some(identity);
        s.candidate_clear_history = kind == ChoppedImageKind::Mixed;
        s.candidate_command.copy_from_slice(&s.committed_command);
        if !s.candidate_clear_history {
            let fringed = kind == ChoppedImageKind::Fringed;
            let adjacent = same_cohort
                && identity.sequence() - s.committed_identity.unwrap().sequence() == 1
                && !metadata.is_discontinuous();
            // Source calibration vectors are Julia vec order; native matrix input
            // is logical row-major. Convert exactly once into retained image order.
            let (rows, columns) = c.frame_shape;
            for column in 0..columns {
                for row in 0..rows {
                    w.image[row + column * rows] = frame[row * columns + column];
                }
            }
            s.candidate_exposure_command
                .copy_from_slice(&s.committed_command);
            self.difference
                .stage_process(
                    ChoppedImageDifferenceOutput::new(
                        &mut w.difference,
                        &mut w.difference_validity,
                        &mut w.pair_status,
                    ),
                    &mut s.difference,
                    ChoppedImageDifferenceInput::new(kind, &w.image, &w.image_validity),
                )
                .map_err(|_| E::NonFiniteInput)?;
            if adjacent && w.pair_status == ChoppedImageDifferenceStatus::Complete {
                if w.difference_validity.iter().any(|&v| !v) {
                    return Err(E::NonFiniteInput);
                }
                c.residual
                    .process(
                        (
                            &mut w.selected_residuals,
                            &mut w.selected_validity,
                            &mut w.valid_fraction,
                        ),
                        &mut crate::NoWorkspace,
                        (&w.difference, &w.difference_validity, 1),
                    )
                    .map_err(|_| E::NonFiniteInput)?;
                c.reconstructor
                    .multiply(&mut w.reconstructed, &w.selected_residuals)
                    .map_err(|_| E::NonFiniteCorrection)?;
                for i in 0..count {
                    w.command_difference[i] =
                        s.committed_command[i] - s.committed_exposure_command[i];
                }
                let response = if fringed {
                    &c.unfringed_response
                } else {
                    &c.fringed_response
                };
                response
                    .multiply(&mut w.phase_correction, &w.command_difference)
                    .map_err(|_| E::NonFiniteCorrection)?;
                for i in 0..count {
                    let value = if fringed {
                        w.reconstructed[i] - w.phase_correction[i]
                    } else {
                        w.reconstructed[i] + w.phase_correction[i]
                    };
                    if !value.is_finite() {
                        return Err(E::NonFiniteCorrection);
                    }
                    w.reconstructed[i] = value;
                }
                self.integrator
                    .stage_process(&mut w.correction, &mut s.integrator, &w.reconstructed)
                    .map_err(|_| E::NonFiniteCorrection)?;
                for i in 0..count {
                    let value = c.bias_command[i] + w.correction[i];
                    if !value.is_finite() {
                        return Err(E::NonFiniteCorrection);
                    }
                    if value < c.lower_limits[i] || value > c.upper_limits[i] {
                        return Err(E::CommandOutOfBounds);
                    }
                    s.candidate_command[i] = value;
                }
            }
        }
        command.copy_from_slice(&s.candidate_command);
        s.candidate_ready = true;
        Ok(())
    }
}
impl AlgorithmPlan for SccCorrectionPlan {
    type Input<'a> = (&'a [f32], Option<SampleMetadata>);
    type Output<'a> = &'a mut [f32];
    type Workspace = SccCorrectionExecution;
    type Error = SccCorrectionError;
    fn process<'i, 'o>(
        &self,
        command: Self::Output<'o>,
        execution: &mut Self::Workspace,
        (frame, metadata): Self::Input<'i>,
    ) -> Result<(), Self::Error> {
        if let Err(error) = self.stage_process(command, execution, frame, metadata) {
            execution.discard();
            return Err(error);
        }
        execution.commit();
        Ok(())
    }
}
fn column_major_matrix(
    rows: usize,
    columns: usize,
    values: &[f32],
) -> Result<PreparedGemv<f32>, SccCorrectionPlanError> {
    if values.len() != rows * columns {
        return Err(SccCorrectionPlanError::Calibration);
    }
    let mut ordered = vec![0.0; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            ordered[row * columns + column] = values[row + column * rows];
        }
    }
    PreparedGemv::new(rows, columns, ordered.into())
        .map_err(|_| SccCorrectionPlanError::Calibration)
}
fn prepare(
    c: SccCorrectionF32Configuration,
) -> Result<(SccCorrectionPlan, SccCorrectionExecution), SccCorrectionPlanError> {
    use SccCorrectionPlanError as E;
    let (rows, columns) = c.frame_size;
    let pixel_count = rows
        .checked_mul(columns)
        .filter(|&v| rows > 0 && columns > 0 && v <= u32::MAX as usize / 4)
        .ok_or(E::Extent)?;
    let count = c.controller_coordinate_count;
    if count == 0
        || count > u32::MAX as usize / 4
        || count
            .checked_mul(count)
            .is_none_or(|v| v > u32::MAX as usize / 4)
    {
        return Err(E::Extent);
    }
    if c.reference_difference.len() != pixel_count
        || c.correction_mask.len() != pixel_count
        || c.coordinate_order.iter().copied().ne(1..=count)
    {
        return Err(E::Calibration);
    }
    if c.command_unit.is_empty() || c.image_schema.is_empty() {
        return Err(E::Units);
    }
    if c.exposure_duration_nanoseconds <= 0 {
        return Err(E::Exposure);
    }
    let sign = match c.reconstructor_sign.as_str() {
        "command_error" => -1.0,
        "correction" => 1.0,
        _ => return Err(E::Sign),
    };
    let residual = SccPixelResidualPlan::new(c.reference_difference, c.correction_mask, 1.0)
        .map_err(|_| E::Calibration)?;
    let selected = residual.selected_count();
    if count
        .checked_mul(selected)
        .is_none_or(|v| v > u32::MAX as usize / 4)
    {
        return Err(E::Extent);
    }
    let reconstructor = column_major_matrix(count, selected, &c.reconstructor)?;
    let fringed_response = column_major_matrix(count, count, &c.fringed_response)?;
    let unfringed_response = column_major_matrix(count, count, &c.unfringed_response)?;
    let lower = c.command_limits.0.expand(count).map_err(|_| E::Limits)?;
    let upper = c.command_limits.1.expand(count).map_err(|_| E::Limits)?;
    if c.bias_command.len() != count
        || c.bias_command
            .iter()
            .chain(&lower)
            .chain(&upper)
            .any(|v| !v.is_finite())
        || c.bias_command
            .iter()
            .zip(lower.iter().zip(&upper))
            .any(|(&b, (&lo, &hi))| b < lo || b > hi)
    {
        return Err(E::Limits);
    }
    if !c.gain.is_finite() || c.gain < 0.0 {
        return Err(E::Integrator);
    }
    let integrator = LeakyIntegratorPlan::new(sign * c.gain, c.pole).map_err(|_| E::Integrator)?;
    let difference = ChoppedImageDifferencePlan::new(pixel_count).map_err(|_| E::Extent)?;
    let state = SccCorrectionState {
        difference: difference.workspace(),
        committed_exposure_command: c.bias_command.clone(),
        candidate_exposure_command: c.bias_command.clone(),
        committed_command: c.bias_command.clone(),
        candidate_command: c.bias_command.clone(),
        integrator: LeakyIntegratorWorkspace::new(count).map_err(|_| E::Extent)?,
        committed_identity: None,
        candidate_identity: None,
        candidate_clear_history: false,
        candidate_ready: false,
    };
    let workspace = SccCorrectionWorkspace {
        image: vec![0.0; pixel_count],
        image_validity: vec![true; pixel_count],
        difference: vec![0.0; pixel_count],
        difference_validity: vec![false; pixel_count],
        pair_status: ChoppedImageDifferenceStatus::Priming,
        selected_residuals: vec![0.0; selected],
        selected_validity: vec![false; selected],
        valid_fraction: 0.0,
        reconstructed: vec![0.0; count],
        correction: vec![0.0; count],
        command_difference: vec![0.0; count],
        phase_correction: vec![0.0; count],
    };
    let initial_command = c.bias_command.clone();
    let calibration = SccCalibration {
        frame_shape: c.frame_size,
        residual,
        reconstructor,
        fringed_response,
        unfringed_response,
        correction_sign: sign,
        exposure_duration_nanoseconds: c.exposure_duration_nanoseconds as u64,
        bias_command: c.bias_command,
        lower_limits: lower,
        upper_limits: upper,
    };
    Ok((
        SccCorrectionPlan {
            calibration: Arc::new(calibration),
            difference,
            integrator,
            gain: c.gain,
            pole: c.pole,
        },
        SccCorrectionExecution {
            state,
            workspace,
            initial_command,
        },
    ))
}
crate::filtergraph_algorithm! {
    /// Rolling SCC exposure correction with coherent calibration and bounded commands.
    pub SccCorrectionF32 {
        label="scc-correction-f32";
        configuration SccCorrectionF32Configuration {
            frame_size:(usize,usize)=GraphRebuild;controller_coordinate_count:usize=GraphRebuild;image_schema:String=GraphRebuild;
            exposure_duration_nanoseconds:i64=Construction;coordinate_order:Vec<usize>=Construction;command_unit:String=Construction;reconstructor_sign:String=Construction;
            reference_difference:Vec<f32>=Construction;correction_mask:Vec<bool>=Construction;reconstructor:Vec<f32>=Construction;fringed_response:Vec<f32>=Construction;unfringed_response:Vec<f32>=Construction;bias_command:Vec<f32>=Construction;command_limits:(ScalarOrVector<f32>,ScalarOrVector<f32>)=Construction;gain:f32=Construction;pole:f32=Construction;
        }
        plan=SccCorrectionPlan;workspace=SccCorrectionExecution;properties=runtime;
        dimensions |plan,_workspace|{rows=plan.calibration.frame_shape.0;columns=plan.calibration.frame_shape.1;coordinates=plan.calibration.bias_command.len();}
        inputs {frame:f32 [rows,columns]="frame",schema=configuration.image_schema.as_str(),sample=frame_sample, required_sample_metadata=["acquisition_identity", "exposure_duration_nanoseconds", "chopped_image_kind"];}
        outputs {command:f32 [coordinates]="command",schema=crate::schemas::CONTROLLER_COMMAND_V1,metadata_from="frame";}
        prepare|configuration|{prepare(configuration)}prepare_error=SccCorrectionPlanError;
        process|plan,workspace|{plan.process(command,workspace,(frame,frame_sample))}process_error=SccCorrectionError;
        reset|_plan,workspace|{workspace.reset();}
    }
}
