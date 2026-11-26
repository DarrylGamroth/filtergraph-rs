//! Calibrated physical-DM actuator stroke optimization along prepared null modes.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::{vec, vec::Vec};
use core::fmt;

use crate::algorithms::linear_algebra::multiply_row_major;
use crate::{AlgorithmPlan, MatrixElement};

/// Invalid prepared PDM stroke-optimization configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmStrokeOptimizationPlanError {
    /// At least one physical actuator is required.
    EmptyPdm,
    /// At least one calibrated physical-command null mode is required.
    EmptyNullModes,
    /// A prepared vector or matrix has the wrong extent.
    Extent,
    /// A prepared value is not finite.
    NonFiniteConfiguration,
    /// A prepared matrix extent overflowed addressable storage.
    ExtentOverflow,
}

impl fmt::Display for PdmStrokeOptimizationPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPdm => formatter.write_str("PDM actuator extent must be nonempty"),
            Self::EmptyNullModes => {
                formatter.write_str("PDM stroke optimization requires at least one null mode")
            }
            Self::Extent => formatter.write_str("PDM stroke-optimization extent is invalid"),
            Self::NonFiniteConfiguration => {
                formatter.write_str("PDM stroke-optimization configuration must be finite")
            }
            Self::ExtentOverflow => {
                formatter.write_str("PDM stroke-optimization matrix extent overflow")
            }
        }
    }
}

impl core::error::Error for PdmStrokeOptimizationPlanError {}

/// Invalid PDM stroke-optimization input, output, or result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PdmStrokeOptimizationError {
    /// The input has the wrong PDM extent.
    InputExtent,
    /// The output has the wrong PDM extent.
    OutputExtent,
    /// The workspace was prepared for incompatible PDM or null-mode extents.
    WorkspaceExtent,
    /// The input contains a non-finite value.
    NonFiniteInput,
    /// The calculation produced a non-finite value.
    NonFiniteOutput,
}

impl fmt::Display for PdmStrokeOptimizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => {
                formatter.write_str("PDM stroke-optimization input extent is invalid")
            }
            Self::OutputExtent => {
                formatter.write_str("PDM stroke-optimization output extent is invalid")
            }
            Self::WorkspaceExtent => {
                formatter.write_str("PDM stroke-optimization workspace extent is invalid")
            }
            Self::NonFiniteInput => {
                formatter.write_str("PDM stroke-optimization input must be finite")
            }
            Self::NonFiniteOutput => {
                formatter.write_str("PDM stroke optimization produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for PdmStrokeOptimizationError {}

/// Prepared affine removal of calibrated physical-command null modes.
///
/// For requested command `c`, preferred command `b`, command-to-null-mode map
/// `H`, and independently prepared null-mode-to-command map `J`, processing
/// computes `c - J * H * (c - b)`. Matrices are row-major. Calibration owns
/// the optical-null and weighted-minimum-stroke qualifications; this plan
/// validates only the generic numerical contract.
#[derive(Clone, Debug, PartialEq)]
pub struct PdmStrokeOptimizationPlan<T> {
    pdm_size: usize,
    null_mode_count: usize,
    preferred_command: Arc<[T]>,
    command_to_null_modes: Arc<[T]>,
    null_modes_to_command: Arc<[T]>,
}

impl<T> PdmStrokeOptimizationPlan<T>
where
    T: MatrixElement,
{
    /// Prepares an affine physical-command null-mode operation.
    pub fn new(
        pdm_size: usize,
        null_mode_count: usize,
        preferred_command: Arc<[T]>,
        command_to_null_modes: Arc<[T]>,
        null_modes_to_command: Arc<[T]>,
    ) -> Result<Self, PdmStrokeOptimizationPlanError> {
        if pdm_size == 0 {
            return Err(PdmStrokeOptimizationPlanError::EmptyPdm);
        }
        if null_mode_count == 0 {
            return Err(PdmStrokeOptimizationPlanError::EmptyNullModes);
        }
        let matrix_extent = pdm_size
            .checked_mul(null_mode_count)
            .ok_or(PdmStrokeOptimizationPlanError::ExtentOverflow)?;
        if preferred_command.len() != pdm_size
            || command_to_null_modes.len() != matrix_extent
            || null_modes_to_command.len() != matrix_extent
        {
            return Err(PdmStrokeOptimizationPlanError::Extent);
        }
        if preferred_command
            .iter()
            .chain(command_to_null_modes.iter())
            .chain(null_modes_to_command.iter())
            .any(|value| !value.is_finite())
        {
            return Err(PdmStrokeOptimizationPlanError::NonFiniteConfiguration);
        }
        Ok(Self {
            pdm_size,
            null_mode_count,
            preferred_command,
            command_to_null_modes,
            null_modes_to_command,
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

    /// Returns the number of prepared physical-command null modes.
    pub fn null_mode_count(&self) -> usize {
        self.null_mode_count
    }

    /// Returns the preferred physical command `b`.
    pub fn preferred_command(&self) -> &[T] {
        &self.preferred_command
    }

    /// Returns shared ownership of the preferred physical command.
    pub fn shared_preferred_command(&self) -> Arc<[T]> {
        Arc::clone(&self.preferred_command)
    }

    /// Returns the row-major physical-command-to-null-mode map `H`.
    pub fn command_to_null_modes(&self) -> &[T] {
        &self.command_to_null_modes
    }

    /// Returns shared ownership of the command-to-null-mode map.
    pub fn shared_command_to_null_modes(&self) -> Arc<[T]> {
        Arc::clone(&self.command_to_null_modes)
    }

    /// Returns the independently prepared row-major null-mode-to-command map `J`.
    pub fn null_modes_to_command(&self) -> &[T] {
        &self.null_modes_to_command
    }

    /// Returns shared ownership of the null-mode-to-command map.
    pub fn shared_null_modes_to_command(&self) -> Arc<[T]> {
        Arc::clone(&self.null_modes_to_command)
    }

    /// Replaces the preferred command while preserving both calibrated maps.
    pub fn with_preferred_command(
        &self,
        preferred_command: Arc<[T]>,
    ) -> Result<Self, PdmStrokeOptimizationPlanError> {
        Self::new(
            self.pdm_size,
            self.null_mode_count,
            preferred_command,
            self.shared_command_to_null_modes(),
            self.shared_null_modes_to_command(),
        )
    }

    /// Replaces the command-to-null-mode map while preserving other artifacts.
    pub fn with_command_to_null_modes(
        &self,
        command_to_null_modes: Arc<[T]>,
    ) -> Result<Self, PdmStrokeOptimizationPlanError> {
        Self::new(
            self.pdm_size,
            self.null_mode_count,
            self.shared_preferred_command(),
            command_to_null_modes,
            self.shared_null_modes_to_command(),
        )
    }

    /// Replaces the null-mode-to-command map while preserving other artifacts.
    pub fn with_null_modes_to_command(
        &self,
        null_modes_to_command: Arc<[T]>,
    ) -> Result<Self, PdmStrokeOptimizationPlanError> {
        Self::new(
            self.pdm_size,
            self.null_mode_count,
            self.shared_preferred_command(),
            self.shared_command_to_null_modes(),
            null_modes_to_command,
        )
    }

    /// Allocates fixed processing scratch.
    pub fn workspace(&self) -> PdmStrokeOptimizationWorkspace<T> {
        PdmStrokeOptimizationWorkspace {
            centered_command: vec![T::zero(); self.pdm_size].into_boxed_slice(),
            null_mode_coefficients: vec![T::zero(); self.null_mode_count].into_boxed_slice(),
        }
    }

    pub(crate) fn workspace_is_compatible(
        &self,
        workspace: &PdmStrokeOptimizationWorkspace<T>,
    ) -> bool {
        workspace.centered_command.len() == self.pdm_size
            && workspace.null_mode_coefficients.len() == self.null_mode_count
    }
}

impl<T> AlgorithmPlan for PdmStrokeOptimizationPlan<T>
where
    T: MatrixElement + 'static,
{
    type Input<'a>
        = &'a [T]
    where
        Self: 'a;
    type Output<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type Workspace = PdmStrokeOptimizationWorkspace<T>;
    type Error = PdmStrokeOptimizationError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if input.len() != self.pdm_size {
            return Err(PdmStrokeOptimizationError::InputExtent);
        }
        if output.len() != self.pdm_size {
            return Err(PdmStrokeOptimizationError::OutputExtent);
        }
        if !self.workspace_is_compatible(workspace) {
            return Err(PdmStrokeOptimizationError::WorkspaceExtent);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(PdmStrokeOptimizationError::NonFiniteInput);
        }
        for ((centered, value), preferred) in workspace
            .centered_command
            .iter_mut()
            .zip(input.iter())
            .zip(self.preferred_command.iter())
        {
            *centered = *value - *preferred;
        }
        if workspace
            .centered_command
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(PdmStrokeOptimizationError::NonFiniteOutput);
        }
        multiply_row_major(
            &self.command_to_null_modes,
            self.null_mode_count,
            self.pdm_size,
            &workspace.centered_command,
            &mut workspace.null_mode_coefficients,
        );
        if workspace
            .null_mode_coefficients
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(PdmStrokeOptimizationError::NonFiniteOutput);
        }
        multiply_row_major(
            &self.null_modes_to_command,
            self.pdm_size,
            self.null_mode_count,
            &workspace.null_mode_coefficients,
            &mut workspace.centered_command,
        );
        for (candidate, requested) in workspace.centered_command.iter_mut().zip(input.iter()) {
            let value = *requested - *candidate;
            if !value.is_finite() {
                return Err(PdmStrokeOptimizationError::NonFiniteOutput);
            }
            *candidate = value;
        }
        output.copy_from_slice(&workspace.centered_command);
        Ok(())
    }
}

/// Fixed scratch for one PDM stroke-optimization owner.
#[derive(Debug, PartialEq)]
pub struct PdmStrokeOptimizationWorkspace<T> {
    centered_command: Box<[T]>,
    null_mode_coefficients: Box<[T]>,
}

impl<T> Clone for PdmStrokeOptimizationWorkspace<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            centered_command: self.centered_command.clone(),
            null_mode_coefficients: self.null_mode_coefficients.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        clone_boxed_slice_from(&mut self.centered_command, &source.centered_command);
        clone_boxed_slice_from(
            &mut self.null_mode_coefficients,
            &source.null_mode_coefficients,
        );
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

crate::filtergraph_algorithm! {
    /// Portable Float32 calibrated PDM stroke-optimization declaration.
    pub PdmStrokeOptimizationF32 {
        label = "pdm-stroke-optimization-f32";
        configuration PdmStrokeOptimizationF32Configuration {
            actuator_count: usize = GraphRebuild;
            null_mode_count: usize = GraphRebuild;
            preferred_command: Vec<f32> = Construction;
            command_to_null_modes: Vec<f32> = Construction;
            null_modes_to_command: Vec<f32> = Construction;
        }
        plan = PdmStrokeOptimizationPlan<f32>;
        workspace = PdmStrokeOptimizationWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            pdm_size = plan.len();
            null_mode_count = plan.null_mode_count();
        }
        inputs {
            requested: f32 [pdm_size] = "requested",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1;
        }
        outputs {
            optimized: f32 [pdm_size] = "optimized",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1,
                metadata_from = "requested";
        }
        parameters {
            preferred_command: f32 [pdm_size] = "preferred-command",
                schema = crate::schemas::PREFERRED_PDM_COMMAND_V1,
                replace |plan, values| {
                    plan.with_preferred_command(Arc::from(values))
                };
            command_to_null_modes: f32 [null_mode_count, pdm_size]
                = "command-to-null-modes",
                schema = crate::schemas::PDM_COMMAND_TO_NULL_MODES_V1,
                replace |plan, values| {
                    plan.with_command_to_null_modes(Arc::from(values))
                };
            null_modes_to_command: f32 [pdm_size, null_mode_count]
                = "null-modes-to-command",
                schema = crate::schemas::PDM_NULL_MODES_TO_COMMAND_V1,
                replace |plan, values| {
                    plan.with_null_modes_to_command(Arc::from(values))
                };
        }
        parameter_error = PdmStrokeOptimizationPlanError;
        prepare |configuration| {
            let plan = PdmStrokeOptimizationPlan::new(
                configuration.actuator_count,
                configuration.null_mode_count,
                Arc::from(configuration.preferred_command),
                Arc::from(configuration.command_to_null_modes),
                Arc::from(configuration.null_modes_to_command),
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = PdmStrokeOptimizationPlanError;
        process |plan, workspace| {
            plan.process(optimized, workspace, requested)
        }
        process_error = PdmStrokeOptimizationError;
    }
}
