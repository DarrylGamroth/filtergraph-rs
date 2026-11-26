//! Prepared device-rate physical-DM actuator-value conditioning.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::{fmt, mem};

use num_traits::{Float, NumCast};

use crate::schemas::{ACTUATOR_COMMAND_OPD_M_V1, DEMANDED_PDM_COMMAND_V1, NATIVE_PDM_COMMAND_V1};
use crate::{
    AlgorithmPlan, InterActuatorStrokeLimitPlan, MatrixElement, NoWorkspace,
    PdmCommandPowerLimitPlan, PdmStrokeOptimizationError, PdmStrokeOptimizationPlan,
    PdmStrokeOptimizationWorkspace, ScalarOrVector, ScalarOrVectorExtentError,
};

/// Invalid prepared PDM command configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmCommandPlanError {
    /// At least one physical actuator is required.
    Empty,
    /// Prepared arrays do not have the PDM extent.
    Extent,
    /// A prepared value is not finite.
    NonFiniteConfiguration,
    /// A lower actuator limit exceeds its upper limit.
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
}

impl fmt::Display for PdmCommandPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("PDM actuator extent must be nonempty"),
            Self::Extent => formatter.write_str("PDM prepared-array extent is invalid"),
            Self::NonFiniteConfiguration => {
                formatter.write_str("PDM command configuration must be finite")
            }
            Self::InvalidActuatorRange => {
                formatter.write_str("PDM lower actuator limit exceeds upper limit")
            }
            Self::InvalidSlewLimit => {
                formatter.write_str("PDM per-tick slew limits must be nonnegative")
            }
            Self::CommandPowerLimitExtent => {
                formatter.write_str("PDM command-power limit extent is invalid")
            }
            Self::InterActuatorStrokeLimitExtent => {
                formatter.write_str("PDM inter-actuator stroke-limit extent is invalid")
            }
            Self::StrokeOptimizationExtent => {
                formatter.write_str("PDM stroke-optimization extent is invalid")
            }
            Self::ConstraintOriginOutsideRange => {
                formatter.write_str("PDM actuator ranges must contain zero for radial limits")
            }
        }
    }
}

impl core::error::Error for PdmCommandPlanError {}

/// Invalid initial PDM slew state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmCommandWorkspaceError {
    /// The initial command has the wrong PDM extent.
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

impl fmt::Display for PdmCommandWorkspaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitialCurrentExtent => {
                formatter.write_str("initial PDM command extent is invalid")
            }
            Self::NonFiniteInitialCurrent => {
                formatter.write_str("initial PDM command must be finite")
            }
            Self::InitialCurrentOutsideRange => {
                formatter.write_str("initial PDM command violates actuator limits")
            }
            Self::InitialCurrentExceedsCommandPower => {
                formatter.write_str("initial PDM command exceeds the command-power limit")
            }
            Self::InitialCurrentExceedsInterActuatorStroke => {
                formatter.write_str("initial PDM command exceeds an inter-actuator stroke limit")
            }
        }
    }
}

impl core::error::Error for PdmCommandWorkspaceError {}

/// Invalid PDM command input, output, or result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmCommandError {
    /// An input vector has the wrong PDM extent.
    InputExtent,
    /// An output vector has the wrong PDM extent.
    OutputExtent,
    /// The workspace was prepared for incompatible command dimensions.
    WorkspaceExtent,
    /// An input contains a non-finite value.
    NonFiniteInput,
    /// The calculation produced a non-finite value.
    NonFiniteOutput,
}

impl fmt::Display for PdmCommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => formatter.write_str("PDM command input extent is invalid"),
            Self::OutputExtent => formatter.write_str("PDM command output extent is invalid"),
            Self::WorkspaceExtent => formatter.write_str("PDM command workspace extent is invalid"),
            Self::NonFiniteInput => formatter.write_str("PDM command input must be finite"),
            Self::NonFiniteOutput => {
                formatter.write_str("PDM command calculation produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for PdmCommandError {}

/// Inputs for one physical-DM command tick.
#[derive(Clone, Copy, Debug)]
pub struct PdmCommandInput<'a, T> {
    /// Complete requested command in physical-actuator order.
    pub requested: &'a [T],
}

/// Outputs from one physical-DM command tick.
#[derive(Debug)]
pub struct PdmCommandOutput<'a, T> {
    /// Physical command after selected stroke optimization and actuator,
    /// slew, command-power, and inter-actuator stroke limits.
    pub demanded: &'a mut [T],
    /// Requested-minus-demanded command in physical-actuator order.
    pub constraint_feedback: &'a mut [T],
}

/// Prepared physical-actuator command conditioning.
#[derive(Clone, Debug, PartialEq)]
pub struct PdmCommandPlan<T> {
    pdm_size: usize,
    lower_limits: Arc<[T]>,
    upper_limits: Arc<[T]>,
    slew_limits: Option<Arc<[T]>>,
    stroke_optimization: Option<PdmStrokeOptimizationPlan<T>>,
    command_power_limit: Option<PdmCommandPowerLimitPlan<T>>,
    inter_actuator_stroke_limit: Option<InterActuatorStrokeLimitPlan<T>>,
}

impl<T> PdmCommandPlan<T>
where
    T: MatrixElement + NumCast + 'static,
{
    /// Prepares range and optional per-tick slew operations.
    pub fn new(
        lower_limits: Arc<[T]>,
        upper_limits: Arc<[T]>,
        slew_limits: Option<Arc<[T]>>,
    ) -> Result<Self, PdmCommandPlanError> {
        let pdm_size = lower_limits.len();
        if pdm_size == 0 {
            return Err(PdmCommandPlanError::Empty);
        }
        if upper_limits.len() != pdm_size
            || slew_limits
                .as_ref()
                .is_some_and(|limits| limits.len() != pdm_size)
        {
            return Err(PdmCommandPlanError::Extent);
        }
        if lower_limits
            .iter()
            .chain(upper_limits.iter())
            .chain(slew_limits.iter().flat_map(|limits| limits.iter()))
            .any(|value| !value.is_finite())
        {
            return Err(PdmCommandPlanError::NonFiniteConfiguration);
        }
        if lower_limits
            .iter()
            .zip(upper_limits.iter())
            .any(|(lower, upper)| lower > upper)
        {
            return Err(PdmCommandPlanError::InvalidActuatorRange);
        }
        if slew_limits
            .as_ref()
            .is_some_and(|limits| limits.iter().any(|limit| *limit < T::zero()))
        {
            return Err(PdmCommandPlanError::InvalidSlewLimit);
        }
        Ok(Self {
            pdm_size,
            lower_limits,
            upper_limits,
            slew_limits,
            stroke_optimization: None,
            command_power_limit: None,
            inter_actuator_stroke_limit: None,
        })
    }

    /// Returns the physical-actuator extent.
    pub fn len(&self) -> usize {
        self.pdm_size
    }

    /// Returns whether the physical-actuator extent is zero.
    pub fn is_empty(&self) -> bool {
        self.pdm_size == 0
    }

    /// Returns the lower physical-actuator command limits.
    pub fn lower_limits(&self) -> &[T] {
        &self.lower_limits
    }

    /// Returns the upper physical-actuator command limits.
    pub fn upper_limits(&self) -> &[T] {
        &self.upper_limits
    }

    /// Returns the optional per-tick physical-actuator slew limits.
    pub fn slew_limits(&self) -> Option<&[T]> {
        self.slew_limits.as_deref()
    }

    /// Selects a prepared total command-power projection.
    pub fn with_command_power_limit(
        mut self,
        limit: PdmCommandPowerLimitPlan<T>,
    ) -> Result<Self, PdmCommandPlanError> {
        if limit.len() != self.pdm_size {
            return Err(PdmCommandPlanError::CommandPowerLimitExtent);
        }
        self.require_constraint_origin()?;
        self.command_power_limit = Some(limit);
        Ok(self)
    }

    /// Selects a prepared physical inter-actuator stroke projection.
    pub fn with_inter_actuator_stroke_limit(
        mut self,
        limit: InterActuatorStrokeLimitPlan<T>,
    ) -> Result<Self, PdmCommandPlanError> {
        if limit.len() != self.pdm_size {
            return Err(PdmCommandPlanError::InterActuatorStrokeLimitExtent);
        }
        self.require_constraint_origin()?;
        self.inter_actuator_stroke_limit = Some(limit);
        Ok(self)
    }

    /// Selects calibrated physical-command null-mode stroke optimization.
    pub fn with_stroke_optimization(
        mut self,
        optimization: PdmStrokeOptimizationPlan<T>,
    ) -> Result<Self, PdmCommandPlanError> {
        if optimization.len() != self.pdm_size {
            return Err(PdmCommandPlanError::StrokeOptimizationExtent);
        }
        self.stroke_optimization = Some(optimization);
        Ok(self)
    }

    /// Returns the selected total command-power projection, if any.
    pub fn command_power_limit(&self) -> Option<&PdmCommandPowerLimitPlan<T>> {
        self.command_power_limit.as_ref()
    }

    /// Returns the selected inter-actuator stroke projection, if any.
    pub fn inter_actuator_stroke_limit(&self) -> Option<&InterActuatorStrokeLimitPlan<T>> {
        self.inter_actuator_stroke_limit.as_ref()
    }

    /// Returns the selected calibrated stroke optimization, if any.
    pub fn stroke_optimization(&self) -> Option<&PdmStrokeOptimizationPlan<T>> {
        self.stroke_optimization.as_ref()
    }

    fn require_constraint_origin(&self) -> Result<(), PdmCommandPlanError> {
        if self
            .lower_limits
            .iter()
            .zip(self.upper_limits.iter())
            .any(|(lower, upper)| *lower > T::zero() || *upper < T::zero())
        {
            return Err(PdmCommandPlanError::ConstraintOriginOutsideRange);
        }
        Ok(())
    }

    /// Allocates fixed state and installs the cold-start physical command.
    pub fn workspace(
        &self,
        initial_current: &[T],
    ) -> Result<PdmCommandWorkspace<T>, PdmCommandWorkspaceError> {
        if initial_current.len() != self.pdm_size {
            return Err(PdmCommandWorkspaceError::InitialCurrentExtent);
        }
        if initial_current.iter().any(|value| !value.is_finite()) {
            return Err(PdmCommandWorkspaceError::NonFiniteInitialCurrent);
        }
        if initial_current
            .iter()
            .zip(self.lower_limits.iter().zip(self.upper_limits.iter()))
            .any(|(value, (lower, upper))| value < lower || value > upper)
        {
            return Err(PdmCommandWorkspaceError::InitialCurrentOutsideRange);
        }
        let mut constraint_scratch = vec![T::zero(); self.pdm_size].into_boxed_slice();
        let mut projection_scratch = vec![T::zero(); self.pdm_size].into_boxed_slice();
        if let Some(limit) = &self.inter_actuator_stroke_limit {
            limit
                .process(&mut constraint_scratch, &mut NoWorkspace, initial_current)
                .map_err(|_| PdmCommandWorkspaceError::InitialCurrentExceedsInterActuatorStroke)?;
            if constraint_scratch.as_ref() != initial_current {
                return Err(PdmCommandWorkspaceError::InitialCurrentExceedsInterActuatorStroke);
            }
        }
        if let Some(limit) = &self.command_power_limit {
            limit
                .process(&mut projection_scratch, &mut NoWorkspace, initial_current)
                .map_err(|_| PdmCommandWorkspaceError::InitialCurrentExceedsCommandPower)?;
            if projection_scratch.as_ref() != initial_current {
                return Err(PdmCommandWorkspaceError::InitialCurrentExceedsCommandPower);
            }
        }
        Ok(PdmCommandWorkspace {
            current: initial_current.into(),
            next_current: vec![T::zero(); self.pdm_size].into_boxed_slice(),
            optimized_requested: vec![T::zero(); self.pdm_size].into_boxed_slice(),
            stroke_optimization: self
                .stroke_optimization
                .as_ref()
                .map(PdmStrokeOptimizationPlan::workspace),
            constraint_scratch,
            projection_scratch,
        })
    }

    pub(crate) fn workspace_is_compatible(&self, workspace: &PdmCommandWorkspace<T>) -> bool {
        let extent_matches = workspace.current.len() == self.pdm_size
            && workspace.next_current.len() == self.pdm_size
            && workspace.optimized_requested.len() == self.pdm_size
            && workspace.constraint_scratch.len() == self.pdm_size
            && workspace.projection_scratch.len() == self.pdm_size;
        let optimization_matches = match (&self.stroke_optimization, &workspace.stroke_optimization)
        {
            (Some(plan), Some(optimization_workspace)) => {
                plan.workspace_is_compatible(optimization_workspace)
            }
            (None, None) => true,
            _ => false,
        };
        extent_matches && optimization_matches
    }
}

impl<T> AlgorithmPlan for PdmCommandPlan<T>
where
    T: MatrixElement + NumCast + 'static,
{
    type Input<'a>
        = PdmCommandInput<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = PdmCommandOutput<'a, T>
    where
        Self: 'a;
    type Workspace = PdmCommandWorkspace<T>;
    type Error = PdmCommandError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if let Err(error) = self.stage_process(output, workspace, input) {
            workspace.discard();
            return Err(error);
        }
        workspace.commit();
        Ok(())
    }
}

impl<T: MatrixElement + NumCast + 'static> PdmCommandPlan<T> {
    pub(crate) fn stage_process(
        &self,
        output: PdmCommandOutput<'_, T>,
        workspace: &mut PdmCommandWorkspace<T>,
        input: PdmCommandInput<'_, T>,
    ) -> Result<(), PdmCommandError> {
        if input.requested.len() != self.pdm_size {
            return Err(PdmCommandError::InputExtent);
        }
        if output.demanded.len() != self.pdm_size
            || output.constraint_feedback.len() != self.pdm_size
        {
            return Err(PdmCommandError::OutputExtent);
        }
        if !self.workspace_is_compatible(workspace) {
            return Err(PdmCommandError::WorkspaceExtent);
        }
        if input.requested.iter().any(|value| !value.is_finite()) {
            return Err(PdmCommandError::NonFiniteInput);
        }
        let optimized = if let Some(optimization) = &self.stroke_optimization {
            let optimization_workspace = workspace
                .stroke_optimization
                .as_mut()
                .ok_or(PdmCommandError::WorkspaceExtent)?;
            optimization
                .process(
                    &mut workspace.optimized_requested,
                    optimization_workspace,
                    input.requested,
                )
                .map_err(|error| match error {
                    PdmStrokeOptimizationError::WorkspaceExtent => PdmCommandError::WorkspaceExtent,
                    _ => PdmCommandError::NonFiniteOutput,
                })?;
            true
        } else {
            false
        };
        for index in 0..self.pdm_size {
            let requested = if optimized {
                workspace.optimized_requested[index]
            } else {
                input.requested[index]
            };
            let safe_target = num_traits::clamp(
                requested,
                self.lower_limits[index],
                self.upper_limits[index],
            );
            let current = self.slew_limits.as_ref().map_or(safe_target, |limits| {
                move_toward(workspace.current[index], safe_target, limits[index])
            });
            if !current.is_finite() {
                return Err(PdmCommandError::NonFiniteOutput);
            }
            workspace.next_current[index] = current;
        }
        workspace
            .constraint_scratch
            .copy_from_slice(&workspace.next_current);
        if let Some(limit) = &self.inter_actuator_stroke_limit {
            limit
                .process(
                    &mut workspace.projection_scratch,
                    &mut NoWorkspace,
                    &workspace.constraint_scratch,
                )
                .map_err(|_| PdmCommandError::NonFiniteOutput)?;
            mem::swap(
                &mut workspace.constraint_scratch,
                &mut workspace.projection_scratch,
            );
        }
        if let Some(limit) = &self.command_power_limit {
            limit
                .process(
                    &mut workspace.projection_scratch,
                    &mut NoWorkspace,
                    &workspace.constraint_scratch,
                )
                .map_err(|_| PdmCommandError::NonFiniteOutput)?;
            mem::swap(
                &mut workspace.constraint_scratch,
                &mut workspace.projection_scratch,
            );
        }
        mem::swap(
            &mut workspace.next_current,
            &mut workspace.constraint_scratch,
        );

        for index in 0..self.pdm_size {
            let demanded = num_traits::clamp(
                workspace.next_current[index],
                self.lower_limits[index],
                self.upper_limits[index],
            );
            if !demanded.is_finite() {
                return Err(PdmCommandError::NonFiniteOutput);
            }
            workspace.constraint_scratch[index] = demanded;
        }
        if let Some(limit) = &self.inter_actuator_stroke_limit {
            limit
                .process(
                    &mut workspace.projection_scratch,
                    &mut NoWorkspace,
                    &workspace.constraint_scratch,
                )
                .map_err(|_| PdmCommandError::NonFiniteOutput)?;
            mem::swap(
                &mut workspace.constraint_scratch,
                &mut workspace.projection_scratch,
            );
        }
        if let Some(limit) = &self.command_power_limit {
            limit
                .process(
                    &mut workspace.projection_scratch,
                    &mut NoWorkspace,
                    &workspace.constraint_scratch,
                )
                .map_err(|_| PdmCommandError::NonFiniteOutput)?;
            mem::swap(
                &mut workspace.constraint_scratch,
                &mut workspace.projection_scratch,
            );
        }
        for ((feedback, requested), demanded) in workspace
            .projection_scratch
            .iter_mut()
            .zip(input.requested)
            .zip(workspace.constraint_scratch.iter())
        {
            *feedback = *requested - *demanded;
            if !feedback.is_finite() {
                return Err(PdmCommandError::NonFiniteOutput);
            }
        }
        output
            .demanded
            .copy_from_slice(&workspace.constraint_scratch);
        output
            .constraint_feedback
            .copy_from_slice(&workspace.projection_scratch);
        Ok(())
    }
}

fn move_toward<T>(current: T, target: T, limit: T) -> T
where
    T: Float,
{
    if target > current {
        let limited = current + limit;
        if !limited.is_finite() || limited >= target {
            target
        } else {
            limited
        }
    } else if target < current {
        let limited = current - limit;
        if !limited.is_finite() || limited <= target {
            target
        } else {
            limited
        }
    } else {
        current
    }
}

/// Device-rate PDM slew state and constraint scratch.
#[derive(Debug, PartialEq)]
pub struct PdmCommandWorkspace<T> {
    current: Box<[T]>,
    next_current: Box<[T]>,
    optimized_requested: Box<[T]>,
    stroke_optimization: Option<PdmStrokeOptimizationWorkspace<T>>,
    constraint_scratch: Box<[T]>,
    projection_scratch: Box<[T]>,
}

impl<T> Clone for PdmCommandWorkspace<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            current: self.current.clone(),
            next_current: self.next_current.clone(),
            optimized_requested: self.optimized_requested.clone(),
            stroke_optimization: self.stroke_optimization.clone(),
            constraint_scratch: self.constraint_scratch.clone(),
            projection_scratch: self.projection_scratch.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        clone_boxed_slice_from(&mut self.current, &source.current);
        clone_boxed_slice_from(&mut self.next_current, &source.next_current);
        clone_boxed_slice_from(&mut self.optimized_requested, &source.optimized_requested);
        self.stroke_optimization
            .clone_from(&source.stroke_optimization);
        clone_boxed_slice_from(&mut self.constraint_scratch, &source.constraint_scratch);
        clone_boxed_slice_from(&mut self.projection_scratch, &source.projection_scratch);
    }
}

impl<T> PdmCommandWorkspace<T> {
    /// Returns the committed physical-actuator slew state.
    pub fn current(&self) -> &[T] {
        &self.current
    }
}

impl<T> PdmCommandWorkspace<T>
where
    T: MatrixElement,
{
    pub(crate) fn commit(&mut self) {
        mem::swap(&mut self.current, &mut self.next_current);
    }
    /// Discard the candidate at a serialized recovery boundary while
    /// retaining the committed physical current.
    pub fn recover(&mut self) {
        self.discard();
    }

    pub(crate) fn discard(&mut self) {
        self.next_current.copy_from_slice(&self.current);
    }

    /// Restore a previously admitted cold-start command without replacing any
    /// prepared storage.
    ///
    /// The caller owns the invariant that `initial_current` was validated for
    /// the unchanged plan when this workspace was prepared. Every processing
    /// scratch array is overwritten before it is read, but clearing it here
    /// also makes lifecycle state deterministic for inspection and tests.
    pub(crate) fn restore_prepared(&mut self, initial_current: &[T]) {
        debug_assert_eq!(initial_current.len(), self.current.len());
        self.current.copy_from_slice(initial_current);
        self.next_current.fill(T::zero());
        self.optimized_requested.fill(T::zero());
        self.constraint_scratch.fill(T::zero());
        self.projection_scratch.fill(T::zero());
    }
}

fn clone_boxed_slice_from<T>(target: &mut Box<[T]>, source: &[T])
where
    T: Clone,
{
    if target.len() == source.len() {
        target.clone_from_slice(source);
    } else {
        *target = source.into();
    }
}

/// Invalid preparation of the portable Float32 PDM command declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmCommandF32PreparationError {
    /// A scalar-or-vector configuration has the wrong actuator extent.
    ConfigurationExtent(ScalarOrVectorExtentError),
    /// The selected PDM command plan is invalid.
    Plan(PdmCommandPlanError),
    /// The initial physical command cannot prepare compatible state.
    Workspace(PdmCommandWorkspaceError),
    /// Legacy graph configuration selected device quantization.
    UnsupportedQuantization,
    /// The demanded schema is not a supported physical actuator command schema.
    UnsupportedDemandedSchema,
}

impl fmt::Display for PdmCommandF32PreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfigurationExtent(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
            Self::Workspace(error) => error.fmt(formatter),
            Self::UnsupportedQuantization => formatter.write_str(
                "Configure DAC quantization and dither in the device driver. Omit quantization_origin and quantization_step or set both to null.",
            ),
            Self::UnsupportedDemandedSchema => {
                formatter.write_str("unsupported physical actuator command schema")
            }
        }
    }
}

impl core::error::Error for PdmCommandF32PreparationError {}

impl From<ScalarOrVectorExtentError> for PdmCommandF32PreparationError {
    fn from(error: ScalarOrVectorExtentError) -> Self {
        Self::ConfigurationExtent(error)
    }
}

impl From<PdmCommandPlanError> for PdmCommandF32PreparationError {
    fn from(error: PdmCommandPlanError) -> Self {
        Self::Plan(error)
    }
}

impl From<PdmCommandWorkspaceError> for PdmCommandF32PreparationError {
    fn from(error: PdmCommandWorkspaceError) -> Self {
        Self::Workspace(error)
    }
}

/// Prepared state and cold-start command for the portable Float32 PDM command
/// declaration.
pub struct PdmCommandF32Workspace {
    algorithm: PdmCommandWorkspace<f32>,
    initial_current: Box<[f32]>,
}

crate::filtergraph_algorithm! {
    /// Portable Float32 physical-DM command-conditioning declaration.
    pub PdmCommandF32 {
        label = "pdm-command-f32";
        configuration PdmCommandF32Configuration {
            actuator_count: usize = GraphRebuild;
            command_limits: (ScalarOrVector<f32>, ScalarOrVector<f32>) = Construction;
            slew_limit: Option<ScalarOrVector<f32>> = Construction;
            initial_current: Option<Vec<f32>> = Construction;
            quantization_origin: Option<f32> = Construction, default = None;
            quantization_step: Option<f32> = Construction, default = None;
            dither_seed: Option<u64> = Construction, default = None;
            demanded_schema: String = GraphRebuild, default = String::from(DEMANDED_PDM_COMMAND_V1);
        }
        plan = PdmCommandPlan<f32>;
        workspace = PdmCommandF32Workspace;
        properties = none;
        dimensions |plan, _workspace| {
            actuator_count = plan.len();
        }
        inputs {
            requested: f32 [actuator_count] = "requested",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1,
                optional_sample_metadata = ["chopped_image_kind", "trigger_state",
                    "acquisition_identity", "exposure_duration_nanoseconds"];
        }
        outputs {
            demanded: f32 [actuator_count] = "demanded",
                schema = configuration.demanded_schema.as_str(),
                metadata_from = "requested";
            constraint_feedback: f32 [actuator_count] = "constraint-feedback",
                schema = crate::schemas::PDM_CONSTRAINT_FEEDBACK_V1,
                metadata_from = "requested";
        }
        prepare |configuration| {
            if !matches!(
                configuration.demanded_schema.as_str(),
                DEMANDED_PDM_COMMAND_V1 | ACTUATOR_COMMAND_OPD_M_V1 | NATIVE_PDM_COMMAND_V1
            ) {
                return Err(PdmCommandF32PreparationError::UnsupportedDemandedSchema);
            }
            prepare_pdm_command(configuration)
        }
        prepare_error = PdmCommandF32PreparationError;
        process |plan, workspace| {
            plan.process(
                PdmCommandOutput {
                    demanded,
                    constraint_feedback,
                },
                &mut workspace.algorithm,
                PdmCommandInput { requested },
            )
        }
        process_error = PdmCommandError;
        reset |_plan, workspace| {
            workspace.algorithm.restore_prepared(&workspace.initial_current);
        }
    }
}

fn prepare_pdm_command(
    configuration: PdmCommandF32Configuration,
) -> Result<(PdmCommandPlan<f32>, PdmCommandF32Workspace), PdmCommandF32PreparationError> {
    if configuration.quantization_origin.is_some() || configuration.quantization_step.is_some() {
        return Err(PdmCommandF32PreparationError::UnsupportedQuantization);
    }
    let lower_limits = Arc::from(
        configuration
            .command_limits
            .0
            .expand(configuration.actuator_count)?,
    );
    let upper_limits = Arc::from(
        configuration
            .command_limits
            .1
            .expand(configuration.actuator_count)?,
    );
    let slew_limits = configuration
        .slew_limit
        .map(|limits| {
            limits
                .expand(configuration.actuator_count)
                .map(Arc::<[f32]>::from)
        })
        .transpose()?;
    let plan = PdmCommandPlan::new(lower_limits, upper_limits, slew_limits)?;
    let initial_current = configuration
        .initial_current
        .unwrap_or_else(|| vec![0.0; configuration.actuator_count])
        .into_boxed_slice();
    let algorithm = plan.workspace(&initial_current)?;
    Ok((
        plan,
        PdmCommandF32Workspace {
            algorithm,
            initial_current,
        },
    ))
}

/// Invalid generic vector-conditioning construction.
#[derive(Debug)]
pub enum VectorConditioningPreparationError {
    /// A positive Float32 vector byte extent must fit UInt32.
    Extent,
    /// Scientific schemas must contain non-whitespace text.
    Schema,
    /// Numerical command-conditioning configuration is invalid.
    Command(PdmCommandF32PreparationError),
}
impl fmt::Display for VectorConditioningPreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Extent => f.write_str("vector byte extent must be positive and fit UInt32"),
            Self::Schema => f.write_str("vector schemas must be nonempty"),
            Self::Command(e) => e.fmt(f),
        }
    }
}
impl core::error::Error for VectorConditioningPreparationError {}
impl From<PdmCommandF32PreparationError> for VectorConditioningPreparationError {
    fn from(e: PdmCommandF32PreparationError) -> Self {
        Self::Command(e)
    }
}
crate::filtergraph_algorithm! {
    /// Condition an ordered same-unit vector without inferring device semantics.
    pub VectorConditioningF32 {
        label = "vector-conditioning-f32";
        configuration VectorConditioningF32Configuration {
            count: usize = GraphRebuild;
            requested_schema: alloc::string::String = GraphRebuild;
            demanded_schema: alloc::string::String = GraphRebuild;
            feedback_schema: alloc::string::String = GraphRebuild;
            command_limits: (ScalarOrVector<f32>, ScalarOrVector<f32>) = Construction;
            slew_limit: Option<ScalarOrVector<f32>> = Construction;
            initial_current: Option<Vec<f32>> = Construction;
            quantization_origin: Option<f32> = Construction, default = None;
            quantization_step: Option<f32> = Construction, default = None;
            dither_seed: Option<u64> = Construction, default = None;
        }
        plan = PdmCommandPlan<f32>;
        workspace = PdmCommandF32Workspace;
        properties = none;
        dimensions |plan, _workspace| { count = plan.len(); }
        inputs { requested: f32 [count] = "requested", schema = configuration.requested_schema.as_str(); }
        outputs {
            demanded: f32 [count] = "demanded", schema = configuration.demanded_schema.as_str(), metadata_from = "requested";
            constraint_feedback: f32 [count] = "constraint-feedback", schema = configuration.feedback_schema.as_str(), metadata_from = "requested";
        }
        prepare |configuration| {
            if configuration.count == 0 || configuration.count > u32::MAX as usize / 4 { return Err(VectorConditioningPreparationError::Extent); }
            if [&configuration.requested_schema,&configuration.demanded_schema,&configuration.feedback_schema].iter().any(|s|s.trim().is_empty()) { return Err(VectorConditioningPreparationError::Schema); }
            Ok(prepare_pdm_command(PdmCommandF32Configuration {
            actuator_count: configuration.count,
            demanded_schema: configuration.demanded_schema,
            slew_limit: configuration.slew_limit,
            initial_current: configuration.initial_current,
            quantization_origin: configuration.quantization_origin,
            quantization_step: configuration.quantization_step,
            dither_seed: configuration.dither_seed,
            command_limits: (configuration.command_limits.0, configuration.command_limits.1),
        })?)
        }
        prepare_error = VectorConditioningPreparationError;
        process |plan, workspace| {
            plan.process(PdmCommandOutput{demanded,constraint_feedback},&mut workspace.algorithm,PdmCommandInput{requested})
        }
        process_error = PdmCommandError;
        reset |_plan, workspace| { workspace.algorithm.restore_prepared(&workspace.initial_current); }
    }
}
