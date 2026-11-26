//! Sequential residual allocation with one transaction across all correctors.
//!
//! Authority: JFG `control/corrector_allocation.jl`. Each ordered stage computes
//! `requested = P * residual`, filters at the base rate, updates or holds its
//! conditioned command, then evaluates `residual -= H * commanded` in row order.
//! Successful processing adopts every filter and each due conditioner together.
//! The `command_limits` pair of componentwise vectors is the exact adopted JFG
//! configuration boundary; scalar expansion and alternate bound names are not added.
//! Slower stages require explicit base-rate coefficients. Filter stability alone
//! does not establish antialias qualification. A later graph operation rejecting
//! the sample does not undo this allocation's successful local adoption.

use crate::{
    AlgorithmPlan, BiquadCascadeError, BiquadCascadePlan, BiquadCascadePlanError,
    BiquadCascadeWorkspace, PdmCommandError, PdmCommandInput, PdmCommandOutput, PdmCommandPlan,
    PdmCommandPlanError, PdmCommandWorkspace, PdmCommandWorkspaceError,
};
use alloc::{boxed::Box, string::String, vec, vec::Vec};
use core::fmt;
#[cfg(not(feature = "blas"))]
use num_traits::Float as _;

/// Invalid allocation dimensions, projections, periods, filters, or bounds.
#[derive(Debug)]
pub enum CorrectorAllocationPreparationError {
    /// Counts must be positive and all declared array byte extents must fit UInt32.
    Extent,
    /// Construction arrays do not match the configured dimensions.
    ConfigurationExtent,
    /// Stage periods must be positive and match the number of stages.
    UpdatePeriod,
    /// A slower stage requires explicitly supplied base-rate coefficients.
    MissingCoefficients,
    /// Both projection matrices must contain finite values.
    NonFiniteProjection,
    /// Stage bounds must contain the zero initial command.
    LimitsExcludeZero,
    /// A stage cascade is invalid.
    Filter(BiquadCascadePlanError),
    /// A stage conditioner is invalid.
    Conditioner(PdmCommandPlanError),
    /// Zero construction state is incompatible with a stage conditioner.
    Workspace(PdmCommandWorkspaceError),
}
impl fmt::Display for CorrectorAllocationPreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Extent => {
                f.write_str("corrector extents must be positive and byte extents fit UInt32")
            }
            Self::ConfigurationExtent => {
                f.write_str("corrector construction array extent is invalid")
            }
            Self::UpdatePeriod => {
                f.write_str("corrector update periods must be positive and match stages")
            }
            Self::MissingCoefficients => {
                f.write_str("slower correctors require explicit base-rate coefficients")
            }
            Self::NonFiniteProjection => f.write_str("corrector projections must be finite"),
            Self::LimitsExcludeZero => f.write_str("corrector limits must contain zero"),
            Self::Filter(e) => e.fmt(f),
            Self::Conditioner(e) => e.fmt(f),
            Self::Workspace(e) => e.fmt(f),
        }
    }
}
impl core::error::Error for CorrectorAllocationPreparationError {}
impl From<BiquadCascadePlanError> for CorrectorAllocationPreparationError {
    fn from(e: BiquadCascadePlanError) -> Self {
        Self::Filter(e)
    }
}
impl From<PdmCommandPlanError> for CorrectorAllocationPreparationError {
    fn from(e: PdmCommandPlanError) -> Self {
        Self::Conditioner(e)
    }
}
impl From<PdmCommandWorkspaceError> for CorrectorAllocationPreparationError {
    fn from(e: PdmCommandWorkspaceError) -> Self {
        Self::Workspace(e)
    }
}

/// Expected numerical rejection or incompatible caller storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CorrectorAllocationError {
    /// Controller demand has an incompatible length.
    InputExtent,
    /// Commands or controller feedback has an incompatible length.
    OutputExtent,
    /// The workspace was prepared for different stage dimensions.
    WorkspaceExtent,
    /// Controller demand contains a nonfinite value.
    NonFiniteInput,
    /// Projection, filtering, conditioning, or residual arithmetic is nonfinite.
    NonFiniteOutput,
}
impl fmt::Display for CorrectorAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InputExtent => "corrector controller demand extent is invalid",
            Self::OutputExtent => "corrector output extent is invalid",
            Self::WorkspaceExtent => "corrector workspace extents are invalid",
            Self::NonFiniteInput => "corrector demand must be finite",
            Self::NonFiniteOutput => "corrector arithmetic must be finite",
        })
    }
}
impl core::error::Error for CorrectorAllocationError {}

#[derive(Clone, Debug)]
struct CorrectorStagePlan {
    to_command: Box<[f32]>,
    to_controller: Box<[f32]>,
    filter: BiquadCascadePlan<f32>,
    conditioner: PdmCommandPlan<f32>,
    update_period: usize,
}
/// Prepared ordered correctors sharing controller coordinates.
#[derive(Clone, Debug)]
pub struct CorrectorAllocationPlan {
    stages: Box<[CorrectorStagePlan]>,
    controller_command_count: usize,
    command_count: usize,
}
impl CorrectorAllocationPlan {
    /// Return the common controller-coordinate extent.
    pub const fn controller_command_count(&self) -> usize {
        self.controller_command_count
    }
    /// Return the concatenated device-command extent.
    pub const fn command_count(&self) -> usize {
        self.command_count
    }
    fn workspace(
        &self,
    ) -> Result<CorrectorAllocationWorkspace, CorrectorAllocationPreparationError> {
        let mut stages = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let n = stage.conditioner.len();
            stages.push(CorrectorStageWorkspace {
                filter: stage.filter.workspace(),
                conditioner: stage.conditioner.workspace(&vec![0.0; n])?,
                requested: vec![0.0; n].into_boxed_slice(),
                filtered: vec![0.0; n].into_boxed_slice(),
                commanded: vec![0.0; n].into_boxed_slice(),
                feedback: vec![0.0; n].into_boxed_slice(),
                held: vec![0.0; n].into_boxed_slice(),
                counter: 0,
                candidate_counter: 0,
            });
        }
        Ok(CorrectorAllocationWorkspace {
            stages: stages.into_boxed_slice(),
            residual: vec![0.0; self.controller_command_count].into_boxed_slice(),
            commands: vec![0.0; self.command_count].into_boxed_slice(),
        })
    }
}
#[derive(Clone, Debug)]
struct CorrectorStageWorkspace {
    filter: BiquadCascadeWorkspace<f32>,
    conditioner: PdmCommandWorkspace<f32>,
    requested: Box<[f32]>,
    filtered: Box<[f32]>,
    commanded: Box<[f32]>,
    feedback: Box<[f32]>,
    held: Box<[f32]>,
    counter: usize,
    candidate_counter: usize,
}
/// Fixed committed nested state and private scratch for one allocation.
#[derive(Clone, Debug)]
pub struct CorrectorAllocationWorkspace {
    stages: Box<[CorrectorStageWorkspace]>,
    residual: Box<[f32]>,
    commands: Box<[f32]>,
}
impl CorrectorAllocationWorkspace {
    fn discard(&mut self) {
        for stage in &mut self.stages {
            stage.filter.discard();
            stage.conditioner.discard();
            stage.commanded.copy_from_slice(&stage.held);
            stage.candidate_counter = stage.counter;
        }
    }
    fn commit(&mut self) {
        for stage in &mut self.stages {
            stage.filter.commit();
            if stage.counter == 0 {
                stage.conditioner.commit();
            }
            stage.held.copy_from_slice(&stage.commanded);
            stage.counter = stage.candidate_counter;
        }
    }
    /// Clear temporal history and obsolete scratch while retaining conditioned
    /// current, held commands and update phase. The owner must serialize recovery.
    pub fn recover(&mut self) {
        let mut offset = 0;
        for stage in &mut self.stages {
            stage.filter.recover();
            stage.conditioner.recover();
            stage.requested.fill(0.0);
            stage.filtered.fill(0.0);
            stage.feedback.fill(0.0);
            stage.commanded.copy_from_slice(&stage.held);
            stage.candidate_counter = stage.counter;
            let end = offset + stage.held.len();
            self.commands[offset..end].copy_from_slice(&stage.held);
            offset = end;
        }
        self.residual.fill(0.0);
    }

    /// Restore zero filters, zero held/current commands, and tick-zero updates.
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.filter.reset();
            stage.held.fill(0.0);
            stage.conditioner.restore_prepared(&stage.held);
            stage.requested.fill(0.0);
            stage.filtered.fill(0.0);
            stage.commanded.fill(0.0);
            stage.feedback.fill(0.0);
            stage.counter = 0;
            stage.candidate_counter = 0;
        }
        self.residual.fill(0.0);
        self.commands.fill(0.0);
    }
}
/// Caller-owned commands and final unserved controller residual.
pub struct CorrectorAllocationOutput<'a> {
    /// Device commands concatenated in configured stage order.
    pub commands: &'a mut [f32],
    /// Final residual in controller coordinates.
    pub constraint_feedback: &'a mut [f32],
}
impl CorrectorAllocationPlan {
    fn stage_process(
        &self,
        output: CorrectorAllocationOutput<'_>,
        workspace: &mut CorrectorAllocationWorkspace,
        demand: &[f32],
    ) -> Result<(), CorrectorAllocationError> {
        use CorrectorAllocationError as E;
        if demand.len() != self.controller_command_count {
            return Err(E::InputExtent);
        }
        if output.commands.len() != self.command_count
            || output.constraint_feedback.len() != self.controller_command_count
        {
            return Err(E::OutputExtent);
        }
        if workspace.stages.len() != self.stages.len()
            || workspace.residual.len() != self.controller_command_count
            || workspace.commands.len() != self.command_count
        {
            return Err(E::WorkspaceExtent);
        }
        for (stage, w) in self.stages.iter().zip(workspace.stages.iter()) {
            let n = stage.conditioner.len();
            if [
                w.requested.len(),
                w.filtered.len(),
                w.commanded.len(),
                w.feedback.len(),
                w.held.len(),
            ]
            .iter()
            .any(|&len| len != n)
            {
                return Err(E::WorkspaceExtent);
            }
        }
        if demand.iter().any(|v| !v.is_finite()) {
            return Err(E::NonFiniteInput);
        }
        workspace.residual.copy_from_slice(demand);
        let m = self.controller_command_count;
        let mut offset = 0;
        for (stage, w) in self.stages.iter().zip(workspace.stages.iter_mut()) {
            let n = stage.conditioner.len();
            for row in 0..n {
                let mut value = 0.0;
                for col in 0..m {
                    value = stage.to_command[row * m + col].mul_add(workspace.residual[col], value);
                }
                w.requested[row] = value;
            }
            stage
                .filter
                .stage_process(&mut w.filtered, &mut w.filter, &w.requested)
                .map_err(|e| match e {
                    BiquadCascadeError::WorkspaceExtent | BiquadCascadeError::Process(_) => {
                        E::WorkspaceExtent
                    }
                    _ => E::NonFiniteOutput,
                })?;
            if w.counter == 0 {
                stage
                    .conditioner
                    .stage_process(
                        PdmCommandOutput {
                            demanded: &mut w.commanded,
                            constraint_feedback: &mut w.feedback,
                        },
                        &mut w.conditioner,
                        PdmCommandInput {
                            requested: &w.filtered,
                        },
                    )
                    .map_err(|e| match e {
                        PdmCommandError::InputExtent
                        | PdmCommandError::OutputExtent
                        | PdmCommandError::WorkspaceExtent => E::WorkspaceExtent,
                        _ => E::NonFiniteOutput,
                    })?;
            } else {
                w.commanded.copy_from_slice(&w.held);
            }
            for row in 0..m {
                let mut value = workspace.residual[row];
                for col in 0..n {
                    value = (-stage.to_controller[row * n + col]).mul_add(w.commanded[col], value);
                }
                if !value.is_finite() {
                    return Err(E::NonFiniteOutput);
                }
                workspace.residual[row] = value;
            }
            workspace.commands[offset..offset + n].copy_from_slice(&w.commanded);
            offset += n;
            w.candidate_counter = if w.counter == stage.update_period - 1 {
                0
            } else {
                w.counter + 1
            };
        }
        output.commands.copy_from_slice(&workspace.commands);
        output
            .constraint_feedback
            .copy_from_slice(&workspace.residual);
        Ok(())
    }
}
impl AlgorithmPlan for CorrectorAllocationPlan {
    type Input<'a> = &'a [f32];
    type Output<'a> = CorrectorAllocationOutput<'a>;
    type Workspace = CorrectorAllocationWorkspace;
    type Error = CorrectorAllocationError;
    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if let Err(e) = self.stage_process(output, workspace, input) {
            workspace.discard();
            return Err(e);
        }
        workspace.commit();
        Ok(())
    }
}

fn prepare(
    configuration: CorrectorAllocationF32Configuration,
) -> Result<
    (CorrectorAllocationPlan, CorrectorAllocationWorkspace),
    CorrectorAllocationPreparationError,
> {
    use CorrectorAllocationPreparationError as E;
    let c = configuration;
    let m = c.controller_command_count;
    if m == 0
        || c.stage_command_counts.is_empty()
        || c.stage_command_counts.contains(&0)
        || c.section_count == 0
    {
        return Err(E::Extent);
    }
    if c.update_periods.len() != c.stage_command_counts.len() || c.update_periods.contains(&0) {
        return Err(E::UpdatePeriod);
    }
    let n = c
        .stage_command_counts
        .iter()
        .try_fold(0usize, |sum, &count| sum.checked_add(count))
        .ok_or(E::Extent)?;
    let max_elements = u32::MAX as usize / 4;
    let projection_count = n.checked_mul(m).ok_or(E::Extent)?;
    let coefficient_count = 5usize
        .checked_mul(c.section_count)
        .and_then(|v| v.checked_mul(n))
        .ok_or(E::Extent)?;
    if n > max_elements
        || m > max_elements
        || projection_count > max_elements
        || coefficient_count > max_elements
    {
        return Err(E::Extent);
    }
    if c.command_projection.len() != projection_count
        || c.realization_projection.len() != projection_count
        || c.command_limits.0.len() != n
        || c.command_limits.1.len() != n
        || c.slew_limits.as_ref().is_some_and(|v| v.len() != n)
        || c.initial_coefficients
            .as_ref()
            .is_some_and(|v| v.len() != coefficient_count)
    {
        return Err(E::ConfigurationExtent);
    }
    if c.command_projection
        .iter()
        .chain(c.realization_projection.iter())
        .any(|v| !v.is_finite())
    {
        return Err(E::NonFiniteProjection);
    }
    let bank = match c.initial_coefficients {
        Some(v) => v,
        None => {
            if c.update_periods.iter().any(|&period| period != 1) {
                return Err(E::MissingCoefficients);
            }
            let mut v = vec![0.0; coefficient_count];
            for tuple in v.as_chunks_mut::<5>().0 {
                tuple[0] = 1.0;
            }
            v
        }
    };
    let mut stages = Vec::with_capacity(c.stage_command_counts.len());
    let mut offset = 0;
    for (&count, &period) in c.stage_command_counts.iter().zip(c.update_periods.iter()) {
        let lower = &c.command_limits.0[offset..offset + count];
        let upper = &c.command_limits.1[offset..offset + count];
        if lower
            .iter()
            .zip(upper)
            .any(|(&lo, &hi)| !(lo <= 0.0 && hi >= 0.0))
        {
            return Err(E::LimitsExcludeZero);
        }
        let conditioner = PdmCommandPlan::new(
            lower.to_vec().into(),
            upper.to_vec().into(),
            c.slew_limits
                .as_ref()
                .map(|v| v[offset..offset + count].to_vec().into()),
        )?;
        let filter = BiquadCascadePlan::new(
            bank[offset * 5 * c.section_count..(offset + count) * 5 * c.section_count].to_vec(),
            count,
            c.section_count,
            count,
        )?;
        let to_command = c.command_projection[offset * m..(offset + count) * m]
            .to_vec()
            .into_boxed_slice();
        let mut to_controller = Vec::with_capacity(m * count);
        for row in 0..m {
            to_controller.extend_from_slice(
                &c.realization_projection[row * n + offset..row * n + offset + count],
            );
        }
        stages.push(CorrectorStagePlan {
            to_command,
            to_controller: to_controller.into_boxed_slice(),
            filter,
            conditioner,
            update_period: period,
        });
        offset += count;
    }
    let plan = CorrectorAllocationPlan {
        stages: stages.into_boxed_slice(),
        controller_command_count: m,
        command_count: n,
    };
    let workspace = plan.workspace()?;
    Ok((plan, workspace))
}
crate::filtergraph_algorithm! {
    /// CPU sequential residual allocation with ordered stage commands and controller feedback.
    pub CorrectorAllocationF32 {
        label = "corrector-allocation-f32";
        configuration CorrectorAllocationF32Configuration {
            controller_command_count: usize = GraphRebuild;
            stage_command_counts: Vec<usize> = GraphRebuild;
            update_periods: Vec<usize> = Construction;
            command_projection: Vec<f32> = Construction;
            realization_projection: Vec<f32> = Construction;
            section_count: usize = Construction;
            initial_coefficients: Option<Vec<f32>> = Construction;
            command_limits: (Vec<f32>,Vec<f32>) = Construction;
            slew_limits: Option<Vec<f32>> = Construction;
            command_schema: String = GraphRebuild;
        }
        plan = CorrectorAllocationPlan;
        workspace = CorrectorAllocationWorkspace;
        properties = none;
        dimensions |plan,_workspace| {controller_command_count=plan.controller_command_count();command_count=plan.command_count();}
        inputs {controller_command:f32 [controller_command_count]="controller-command",schema=crate::schemas::CONTROLLER_COMMAND_V1;}
        outputs {
            commands:f32 [command_count]="commands",schema=configuration.command_schema.as_str(),metadata_from="controller-command";
            constraint_feedback:f32 [controller_command_count]="controller-constraint-feedback",schema=crate::schemas::CONTROLLER_CONSTRAINT_FEEDBACK_V1,metadata_from="controller-command";
        }
        prepare |configuration| {prepare(configuration)}
        prepare_error = CorrectorAllocationPreparationError;
        process |plan,workspace| {plan.process(CorrectorAllocationOutput {commands,constraint_feedback},workspace,controller_command)}
        process_error = CorrectorAllocationError;
        reset |_plan,workspace| {workspace.reset();}
    }
}
