//! Compatibility composite for a fused one-VDM/one-PDM command boundary.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use core::{fmt, mem};

use num_traits::NumCast;

use crate::{
    AlgorithmPlan, InterActuatorStrokeLimitPlan, MatrixElement, PdmCommandError, PdmCommandInput,
    PdmCommandOutput, PdmCommandPlan, PdmCommandPlanError, PdmCommandPowerLimitPlan,
    PdmCommandWorkspace, PdmCommandWorkspaceError, PdmFeedbackToVdmError, PdmFeedbackToVdmPlan,
    PdmFeedbackToVdmWorkspace, PdmProjection, PdmProjectionError, PdmStrokeOptimizationPlan,
    VdmToPdmError, VdmToPdmPlan, VdmToPdmWorkspace,
};

/// Invalid prepared deformable-mirror command configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeformableMirrorCommandPlanError {
    /// Every configured vector space must contain at least one coordinate.
    EmptyDimension,
    /// Prepared projection or command coordinate spaces are incompatible.
    MatrixExtent,
    /// A matrix or limit contains a non-finite value.
    NonFiniteConfiguration,
    /// A lower limit exceeds its matching upper limit.
    InvalidActuatorRange,
    /// A per-tick slew limit is negative.
    InvalidSlewLimit,
    /// A selected command-power plan has the wrong PDM extent.
    CommandPowerLimitExtent,
    /// A selected inter-actuator stroke plan has the wrong PDM extent.
    InterActuatorStrokeLimitExtent,
    /// A selected stroke-optimization plan has the wrong PDM extent.
    StrokeOptimizationExtent,
    /// A radial constraint was selected for actuator ranges that exclude zero.
    ConstraintOriginOutsideRange,
    /// A matrix extent overflowed addressable storage.
    ExtentOverflow,
}

impl fmt::Display for DeformableMirrorCommandPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension => formatter.write_str("DM dimensions must be nonempty"),
            Self::MatrixExtent => formatter.write_str("DM projection extents are incompatible"),
            Self::NonFiniteConfiguration => {
                formatter.write_str("DM matrices and limits must be finite")
            }
            Self::InvalidActuatorRange => {
                formatter.write_str("lower actuator limit exceeds upper limit")
            }
            Self::InvalidSlewLimit => {
                formatter.write_str("per-tick slew limits must be nonnegative")
            }
            Self::CommandPowerLimitExtent => {
                formatter.write_str("DM command-power limit extent is invalid")
            }
            Self::InterActuatorStrokeLimitExtent => {
                formatter.write_str("DM inter-actuator stroke-limit extent is invalid")
            }
            Self::StrokeOptimizationExtent => {
                formatter.write_str("DM stroke-optimization extent is invalid")
            }
            Self::ConstraintOriginOutsideRange => {
                formatter.write_str("DM actuator ranges must contain zero for radial limits")
            }
            Self::ExtentOverflow => formatter.write_str("DM matrix extent overflow"),
        }
    }
}

impl core::error::Error for DeformableMirrorCommandPlanError {}

/// Invalid initial physical-actuator slew state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeformableMirrorCommandWorkspaceError {
    /// The initial command has the wrong physical extent.
    InitialCurrentExtent,
    /// The initial command contains a non-finite value.
    NonFiniteInitialCurrent,
    /// The initial command violates an actuator-range limit.
    InitialCurrentOutsideRange,
    /// The initial command exceeds the selected command-power limit.
    InitialCurrentExceedsCommandPower,
    /// The initial command exceeds a selected inter-actuator stroke limit.
    InitialCurrentExceedsInterActuatorStroke,
}

impl fmt::Display for DeformableMirrorCommandWorkspaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitialCurrentExtent => {
                formatter.write_str("initial physical-command state extent is invalid")
            }
            Self::NonFiniteInitialCurrent => {
                formatter.write_str("initial physical-command state must be finite")
            }
            Self::InitialCurrentOutsideRange => {
                formatter.write_str("initial physical-command state violates actuator limits")
            }
            Self::InitialCurrentExceedsCommandPower => formatter
                .write_str("initial physical-command state exceeds the command-power limit"),
            Self::InitialCurrentExceedsInterActuatorStroke => formatter
                .write_str("initial physical-command state exceeds an inter-actuator stroke limit"),
        }
    }
}

impl core::error::Error for DeformableMirrorCommandWorkspaceError {}

/// Invalid deformable-mirror command input, output, or result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeformableMirrorCommandError {
    /// An input vector has the wrong extent.
    InputExtent,
    /// An output vector has the wrong extent.
    OutputExtent,
    /// The workspace was prepared for incompatible command dimensions.
    WorkspaceExtent,
    /// An input vector contains a non-finite value.
    NonFiniteInput,
    /// The calculation produced a non-finite value.
    NonFiniteOutput,
}

impl fmt::Display for DeformableMirrorCommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => formatter.write_str("DM command input extent is invalid"),
            Self::OutputExtent => formatter.write_str("DM command output extent is invalid"),
            Self::WorkspaceExtent => formatter.write_str("DM command workspace extent is invalid"),
            Self::NonFiniteInput => formatter.write_str("DM command input must be finite"),
            Self::NonFiniteOutput => {
                formatter.write_str("DM command calculation produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for DeformableMirrorCommandError {}

/// Inputs for one deformable-mirror command tick.
#[derive(Clone, Copy, Debug)]
pub struct DeformableMirrorCommandInput<'a, T> {
    /// Latest active-VDM command.
    pub vdm_command: &'a [T],
    /// System-flat contribution in PDM order.
    pub best_flat: &'a [T],
    /// Engineering-offset contribution in PDM order.
    pub pdm_offset: &'a [T],
}

/// Outputs from one deformable-mirror command tick.
#[derive(Debug)]
pub struct DeformableMirrorCommandOutput<'a, T> {
    /// Full composed physical command before constraints.
    pub requested: &'a mut [T],
    /// Physical command after selected stroke optimization and actuator,
    /// device-rate slew, command-power, and inter-actuator stroke limits.
    pub demanded: &'a mut [T],
    /// Requested-minus-demanded command projected into active VDM coordinates.
    pub vdm_constraint_feedback: &'a mut [T],
}

/// Fused SCAO composition of VDM-to-PDM, PDM conditioning, and feedback plans.
#[derive(Clone, Debug, PartialEq)]
pub struct DeformableMirrorCommandPlan<T> {
    active_vdm_size: usize,
    pdm_size: usize,
    vdm_to_pdm: VdmToPdmPlan<T>,
    pdm_feedback_to_vdm: PdmFeedbackToVdmPlan<T>,
    pdm: PdmCommandPlan<T>,
}

impl<T> DeformableMirrorCommandPlan<T>
where
    T: MatrixElement + NumCast + 'static,
{
    /// Composes compatible prepared forward, reverse-feedback, and PDM plans.
    ///
    /// This compatibility composite owns one VDM and one PDM. The forward and
    /// reverse plans must therefore describe the same active-VDM, full-VDM,
    /// and physical-actuator extents and each contain exactly one PDM
    /// projection.
    pub fn from_plans(
        vdm_to_pdm: VdmToPdmPlan<T>,
        pdm_feedback_to_vdm: PdmFeedbackToVdmPlan<T>,
        pdm: PdmCommandPlan<T>,
    ) -> Result<Self, DeformableMirrorCommandPlanError> {
        if vdm_to_pdm.pdm_count() != 1
            || pdm_feedback_to_vdm.pdm_count() != 1
            || vdm_to_pdm.active_vdm_size() != pdm_feedback_to_vdm.active_vdm_size()
            || vdm_to_pdm.full_vdm_size() != pdm_feedback_to_vdm.full_vdm_size()
            || vdm_to_pdm.physical_actuator_extent()
                != pdm_feedback_to_vdm.physical_actuator_extent()
            || vdm_to_pdm.physical_actuator_extent() != pdm.len()
        {
            return Err(DeformableMirrorCommandPlanError::MatrixExtent);
        }
        Ok(Self {
            active_vdm_size: vdm_to_pdm.active_vdm_size(),
            pdm_size: pdm.len(),
            vdm_to_pdm,
            pdm_feedback_to_vdm,
            pdm,
        })
    }

    /// Prepares explicit forward, reverse, range, and optional slew operations.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        active_vdm_size: usize,
        full_vdm_size: usize,
        pdm_size: usize,
        active_to_full: Arc<[T]>,
        vdm_to_pdm: Arc<[T]>,
        pdm_to_vdm: Arc<[T]>,
        full_to_active: Arc<[T]>,
        lower_limits: Arc<[T]>,
        upper_limits: Arc<[T]>,
        slew_limits: Option<Arc<[T]>>,
    ) -> Result<Self, DeformableMirrorCommandPlanError> {
        if active_vdm_size == 0 || full_vdm_size == 0 || pdm_size == 0 {
            return Err(DeformableMirrorCommandPlanError::EmptyDimension);
        }
        let projection = PdmProjection::new(pdm_size, vdm_to_pdm, pdm_to_vdm)
            .map_err(map_pdm_projection_error)?;
        let projections: Arc<[PdmProjection<T>]> = Arc::from([projection]);
        let vdm_to_pdm = VdmToPdmPlan::new(
            active_vdm_size,
            full_vdm_size,
            active_to_full,
            Arc::clone(&projections),
        )
        .map_err(map_pdm_projection_error)?;
        let pdm_feedback_to_vdm =
            PdmFeedbackToVdmPlan::new(active_vdm_size, full_vdm_size, full_to_active, projections)
                .map_err(map_pdm_projection_error)?;
        let pdm = PdmCommandPlan::new(lower_limits, upper_limits, slew_limits)
            .map_err(map_pdm_plan_error)?;
        Self::from_plans(vdm_to_pdm, pdm_feedback_to_vdm, pdm)
    }

    /// Selects a prepared total PDM command-power projection.
    pub fn with_command_power_limit(
        mut self,
        limit: PdmCommandPowerLimitPlan<T>,
    ) -> Result<Self, DeformableMirrorCommandPlanError> {
        self.pdm = self
            .pdm
            .with_command_power_limit(limit)
            .map_err(map_pdm_plan_error)?;
        Ok(self)
    }

    /// Selects a prepared physical inter-actuator stroke projection.
    pub fn with_inter_actuator_stroke_limit(
        mut self,
        limit: InterActuatorStrokeLimitPlan<T>,
    ) -> Result<Self, DeformableMirrorCommandPlanError> {
        self.pdm = self
            .pdm
            .with_inter_actuator_stroke_limit(limit)
            .map_err(map_pdm_plan_error)?;
        Ok(self)
    }

    /// Selects calibrated physical-command null-mode stroke optimization.
    pub fn with_stroke_optimization(
        mut self,
        optimization: PdmStrokeOptimizationPlan<T>,
    ) -> Result<Self, DeformableMirrorCommandPlanError> {
        self.pdm = self
            .pdm
            .with_stroke_optimization(optimization)
            .map_err(map_pdm_plan_error)?;
        Ok(self)
    }

    /// Returns the active-VDM command extent.
    pub const fn active_vdm_size(&self) -> usize {
        self.active_vdm_size
    }

    /// Returns the physical-actuator command extent.
    pub const fn pdm_size(&self) -> usize {
        self.pdm_size
    }

    /// Returns the prepared active-VDM-to-PDM operation.
    pub const fn vdm_to_pdm(&self) -> &VdmToPdmPlan<T> {
        &self.vdm_to_pdm
    }

    /// Returns the prepared PDM-feedback-to-active-VDM operation.
    pub const fn pdm_feedback_to_vdm(&self) -> &PdmFeedbackToVdmPlan<T> {
        &self.pdm_feedback_to_vdm
    }

    /// Returns the prepared physical-actuator command conditioner.
    pub const fn pdm_command(&self) -> &PdmCommandPlan<T> {
        &self.pdm
    }

    /// Allocates fixed scratch and installs the cold-start actuator state.
    pub fn workspace(
        &self,
        initial_current: &[T],
    ) -> Result<DeformableMirrorCommandWorkspace<T>, DeformableMirrorCommandWorkspaceError> {
        let pdm = self
            .pdm
            .workspace(initial_current)
            .map_err(map_pdm_workspace_error)?;
        let next_pdm = pdm.clone();
        Ok(DeformableMirrorCommandWorkspace {
            vdm_to_pdm: self.vdm_to_pdm.workspace(),
            pdm_feedback_to_vdm: self.pdm_feedback_to_vdm.workspace(),
            pdm,
            next_pdm,
            pdm_correction: vec![T::zero(); self.pdm_size].into_boxed_slice(),
            pdm_feedback: vec![T::zero(); self.pdm_size].into_boxed_slice(),
        })
    }
}

impl<T> AlgorithmPlan for DeformableMirrorCommandPlan<T>
where
    T: MatrixElement + NumCast + 'static,
{
    type Input<'a>
        = DeformableMirrorCommandInput<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = DeformableMirrorCommandOutput<'a, T>
    where
        Self: 'a;
    type Workspace = DeformableMirrorCommandWorkspace<T>;
    type Error = DeformableMirrorCommandError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if input.vdm_command.len() != self.active_vdm_size
            || input.best_flat.len() != self.pdm_size
            || input.pdm_offset.len() != self.pdm_size
        {
            return Err(DeformableMirrorCommandError::InputExtent);
        }
        if output.requested.len() != self.pdm_size
            || output.demanded.len() != self.pdm_size
            || output.vdm_constraint_feedback.len() != self.active_vdm_size
        {
            return Err(DeformableMirrorCommandError::OutputExtent);
        }
        if workspace.pdm_correction.len() != self.pdm_size
            || workspace.pdm_feedback.len() != self.pdm_size
            || !self
                .vdm_to_pdm
                .workspace_is_compatible(&workspace.vdm_to_pdm)
            || !self
                .pdm_feedback_to_vdm
                .workspace_is_compatible(&workspace.pdm_feedback_to_vdm)
            || !self.pdm.workspace_is_compatible(&workspace.pdm)
            || !self.pdm.workspace_is_compatible(&workspace.next_pdm)
        {
            return Err(DeformableMirrorCommandError::WorkspaceExtent);
        }
        if input
            .vdm_command
            .iter()
            .chain(input.best_flat.iter())
            .chain(input.pdm_offset.iter())
            .any(|value| !value.is_finite())
        {
            return Err(DeformableMirrorCommandError::NonFiniteInput);
        }

        self.vdm_to_pdm
            .process(
                &mut workspace.pdm_correction,
                &mut workspace.vdm_to_pdm,
                input.vdm_command,
            )
            .map_err(map_vdm_to_pdm_error)?;
        for index in 0..self.pdm_size {
            output.requested[index] =
                workspace.pdm_correction[index] + input.best_flat[index] + input.pdm_offset[index];
        }
        if output.requested.iter().any(|value| !value.is_finite()) {
            return Err(DeformableMirrorCommandError::NonFiniteOutput);
        }

        workspace.next_pdm.clone_from(&workspace.pdm);
        self.pdm
            .process(
                PdmCommandOutput {
                    demanded: output.demanded,
                    constraint_feedback: &mut workspace.pdm_feedback,
                },
                &mut workspace.next_pdm,
                PdmCommandInput {
                    requested: output.requested,
                },
            )
            .map_err(map_pdm_error)?;
        self.pdm_feedback_to_vdm
            .process(
                output.vdm_constraint_feedback,
                &mut workspace.pdm_feedback_to_vdm,
                &workspace.pdm_feedback,
            )
            .map_err(map_feedback_error)?;
        // Commit PDM history only after the complete composite succeeds.
        mem::swap(&mut workspace.pdm, &mut workspace.next_pdm);
        Ok(())
    }
}

/// Fused SCAO workspace containing projection scratch and one PDM state.
#[derive(Clone, Debug, PartialEq)]
pub struct DeformableMirrorCommandWorkspace<T> {
    vdm_to_pdm: VdmToPdmWorkspace<T>,
    pdm_feedback_to_vdm: PdmFeedbackToVdmWorkspace<T>,
    pdm: PdmCommandWorkspace<T>,
    next_pdm: PdmCommandWorkspace<T>,
    pdm_correction: Box<[T]>,
    pdm_feedback: Box<[T]>,
}

impl<T> DeformableMirrorCommandWorkspace<T> {
    /// Returns the committed physical-actuator slew state.
    pub fn current(&self) -> &[T] {
        self.pdm.current()
    }
}

fn map_pdm_projection_error(error: PdmProjectionError) -> DeformableMirrorCommandPlanError {
    match error {
        PdmProjectionError::EmptyDimension | PdmProjectionError::EmptyPdmSet => {
            DeformableMirrorCommandPlanError::EmptyDimension
        }
        PdmProjectionError::ProjectionExtent => DeformableMirrorCommandPlanError::MatrixExtent,
        PdmProjectionError::NonFiniteConfiguration => {
            DeformableMirrorCommandPlanError::NonFiniteConfiguration
        }
        PdmProjectionError::ExtentOverflow => DeformableMirrorCommandPlanError::ExtentOverflow,
    }
}

fn map_pdm_plan_error(error: PdmCommandPlanError) -> DeformableMirrorCommandPlanError {
    match error {
        PdmCommandPlanError::Empty | PdmCommandPlanError::Extent => {
            DeformableMirrorCommandPlanError::MatrixExtent
        }
        PdmCommandPlanError::NonFiniteConfiguration => {
            DeformableMirrorCommandPlanError::NonFiniteConfiguration
        }
        PdmCommandPlanError::InvalidActuatorRange => {
            DeformableMirrorCommandPlanError::InvalidActuatorRange
        }
        PdmCommandPlanError::InvalidSlewLimit => DeformableMirrorCommandPlanError::InvalidSlewLimit,
        PdmCommandPlanError::CommandPowerLimitExtent => {
            DeformableMirrorCommandPlanError::CommandPowerLimitExtent
        }
        PdmCommandPlanError::InterActuatorStrokeLimitExtent => {
            DeformableMirrorCommandPlanError::InterActuatorStrokeLimitExtent
        }
        PdmCommandPlanError::StrokeOptimizationExtent => {
            DeformableMirrorCommandPlanError::StrokeOptimizationExtent
        }
        PdmCommandPlanError::ConstraintOriginOutsideRange => {
            DeformableMirrorCommandPlanError::ConstraintOriginOutsideRange
        }
    }
}

fn map_pdm_workspace_error(
    error: PdmCommandWorkspaceError,
) -> DeformableMirrorCommandWorkspaceError {
    match error {
        PdmCommandWorkspaceError::InitialCurrentExtent => {
            DeformableMirrorCommandWorkspaceError::InitialCurrentExtent
        }
        PdmCommandWorkspaceError::NonFiniteInitialCurrent => {
            DeformableMirrorCommandWorkspaceError::NonFiniteInitialCurrent
        }
        PdmCommandWorkspaceError::InitialCurrentOutsideRange => {
            DeformableMirrorCommandWorkspaceError::InitialCurrentOutsideRange
        }
        PdmCommandWorkspaceError::InitialCurrentExceedsCommandPower => {
            DeformableMirrorCommandWorkspaceError::InitialCurrentExceedsCommandPower
        }
        PdmCommandWorkspaceError::InitialCurrentExceedsInterActuatorStroke => {
            DeformableMirrorCommandWorkspaceError::InitialCurrentExceedsInterActuatorStroke
        }
    }
}

fn map_vdm_to_pdm_error(error: VdmToPdmError) -> DeformableMirrorCommandError {
    match error {
        VdmToPdmError::InputExtent => DeformableMirrorCommandError::InputExtent,
        VdmToPdmError::OutputExtent => DeformableMirrorCommandError::OutputExtent,
        VdmToPdmError::WorkspaceExtent => DeformableMirrorCommandError::WorkspaceExtent,
        VdmToPdmError::NonFiniteInput => DeformableMirrorCommandError::NonFiniteInput,
        VdmToPdmError::NonFiniteOutput => DeformableMirrorCommandError::NonFiniteOutput,
    }
}

fn map_feedback_error(error: PdmFeedbackToVdmError) -> DeformableMirrorCommandError {
    match error {
        PdmFeedbackToVdmError::InputExtent => DeformableMirrorCommandError::InputExtent,
        PdmFeedbackToVdmError::OutputExtent => DeformableMirrorCommandError::OutputExtent,
        PdmFeedbackToVdmError::WorkspaceExtent => DeformableMirrorCommandError::WorkspaceExtent,
        PdmFeedbackToVdmError::NonFiniteInput => DeformableMirrorCommandError::NonFiniteInput,
        PdmFeedbackToVdmError::NonFiniteOutput => DeformableMirrorCommandError::NonFiniteOutput,
    }
}

fn map_pdm_error(error: PdmCommandError) -> DeformableMirrorCommandError {
    match error {
        PdmCommandError::InputExtent => DeformableMirrorCommandError::InputExtent,
        PdmCommandError::OutputExtent => DeformableMirrorCommandError::OutputExtent,
        PdmCommandError::WorkspaceExtent => DeformableMirrorCommandError::WorkspaceExtent,
        PdmCommandError::NonFiniteInput => DeformableMirrorCommandError::NonFiniteInput,
        PdmCommandError::NonFiniteOutput => DeformableMirrorCommandError::NonFiniteOutput,
    }
}
