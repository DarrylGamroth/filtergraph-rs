//! Deterministic random corrector excitation for DO-CRIME calibration.

use alloc::{boxed::Box, vec, vec::Vec};
use core::fmt;

use num_traits::{Float, NumCast};

use crate::algorithms::{splitmix64_advance, splitmix64_at, splitmix64_unit_at};
use crate::{AlgorithmPlan, MatrixElement};

/// Invalid prepared DO-CRIME calibration excitation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeExcitationPlanError {
    /// At least one corrector coordinate is required.
    Empty,
    /// One amplitude is non-finite or not positive.
    InvalidAmplitude {
        /// Zero-based corrector-coordinate index.
        corrector_coordinate: usize,
    },
}

impl fmt::Display for DocrimeExcitationPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("DO-CRIME excitation amplitudes must be nonempty"),
            Self::InvalidAmplitude {
                corrector_coordinate,
            } => write!(
                formatter,
                "DO-CRIME excitation amplitude for corrector coordinate {corrector_coordinate} must be finite and positive"
            ),
        }
    }
}

impl core::error::Error for DocrimeExcitationPlanError {}

/// Invalid DO-CRIME calibration-excitation execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocrimeExcitationError {
    /// The caller-owned output has the wrong corrector-coordinate extent.
    OutputExtent {
        /// Required extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// The supplied workspace was prepared for a different extent.
    WorkspaceExtent,
    /// A SplitMix64 sample could not be represented by the plan scalar type.
    ScalarConversion,
    /// One staged excitation was unexpectedly non-finite.
    NonFiniteCalculation,
}

impl fmt::Display for DocrimeExcitationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputExtent { expected, actual } => write!(
                formatter,
                "DO-CRIME excitation output extent {actual} does not match {expected}"
            ),
            Self::WorkspaceExtent => {
                formatter.write_str("DO-CRIME excitation workspace extent is invalid")
            }
            Self::ScalarConversion => formatter
                .write_str("DO-CRIME SplitMix64 sample is not representable by the scalar type"),
            Self::NonFiniteCalculation => {
                formatter.write_str("DO-CRIME excitation calculation produced a non-finite value")
            }
        }
    }
}

impl core::error::Error for DocrimeExcitationError {}

/// Prepared deterministic random corrector excitation for DO-CRIME.
///
/// Each successful [`AlgorithmPlan::process`] call emits one complete
/// corrector-coordinate vector. Physical projection, command superposition,
/// constraint handling, and exposure association remain caller-owned.
#[derive(Clone, Debug, PartialEq)]
pub struct DocrimeExcitationPlan<T> {
    amplitudes: Box<[T]>,
    seed: u64,
}

/// Prepared deterministic binary random corrector excitation for DO-CRIME.
///
/// Each corrector coordinate is exactly its positive or negative prepared
/// amplitude. Physical projection, command superposition, constraint handling,
/// and exposure association remain caller-owned.
#[derive(Clone, Debug, PartialEq)]
pub struct DocrimeBinaryExcitationPlan<T> {
    amplitudes: Box<[T]>,
    seed: u64,
}

impl<T> DocrimeExcitationPlan<T>
where
    T: Float,
{
    /// Validates positive per-coordinate amplitudes and retains the seed.
    pub fn new(
        amplitudes: impl Into<Box<[T]>>,
        seed: u64,
    ) -> Result<Self, DocrimeExcitationPlanError> {
        let amplitudes = amplitudes.into();
        validate_amplitudes(&amplitudes)?;
        Ok(Self { amplitudes, seed })
    }

    /// Returns the corrector-coordinate extent.
    pub fn len(&self) -> usize {
        self.amplitudes.len()
    }

    /// Returns whether the corrector-coordinate extent is zero.
    pub fn is_empty(&self) -> bool {
        self.amplitudes.is_empty()
    }

    /// Returns the prepared per-coordinate excitation amplitudes.
    pub fn amplitudes(&self) -> &[T] {
        &self.amplitudes
    }

    /// Returns the initial SplitMix64 state for a new workspace.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Allocates generator history and fixed staged storage for one owner.
    pub fn workspace(&self) -> DocrimeExcitationWorkspace<T> {
        DocrimeExcitationWorkspace {
            random_state: self.seed,
            staged_excitation: vec![T::zero(); self.len()].into_boxed_slice(),
        }
    }
}

impl<T> DocrimeBinaryExcitationPlan<T>
where
    T: Float,
{
    /// Validates positive per-coordinate amplitudes and retains the seed.
    pub fn new(
        amplitudes: impl Into<Box<[T]>>,
        seed: u64,
    ) -> Result<Self, DocrimeExcitationPlanError> {
        let amplitudes = amplitudes.into();
        validate_amplitudes(&amplitudes)?;
        Ok(Self { amplitudes, seed })
    }

    /// Returns the corrector-coordinate extent.
    pub fn len(&self) -> usize {
        self.amplitudes.len()
    }

    /// Returns whether the corrector-coordinate extent is zero.
    pub fn is_empty(&self) -> bool {
        self.amplitudes.is_empty()
    }

    /// Returns the prepared per-coordinate excitation amplitudes.
    pub fn amplitudes(&self) -> &[T] {
        &self.amplitudes
    }

    /// Returns the initial SplitMix64 state for a new workspace.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Allocates generator history and fixed staged storage for one owner.
    pub fn workspace(&self) -> DocrimeExcitationWorkspace<T> {
        DocrimeExcitationWorkspace {
            random_state: self.seed,
            staged_excitation: vec![T::zero(); self.len()].into_boxed_slice(),
        }
    }
}

impl<T> AlgorithmPlan for DocrimeExcitationPlan<T>
where
    T: MatrixElement + NumCast + 'static,
{
    type Input<'a> = ();
    type Output<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type Workspace = DocrimeExcitationWorkspace<T>;
    type Error = DocrimeExcitationError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        (): Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if output.len() != self.len() {
            return Err(DocrimeExcitationError::OutputExtent {
                expected: self.len(),
                actual: output.len(),
            });
        }
        if workspace.staged_excitation.len() != self.len() {
            return Err(DocrimeExcitationError::WorkspaceExtent);
        }

        let random_state = workspace.random_state;
        for (draw_index, (staged, amplitude)) in workspace
            .staged_excitation
            .iter_mut()
            .zip(&self.amplitudes)
            .enumerate()
        {
            let unit = splitmix64_unit_at::<T>(random_state, draw_index as u64)
                .ok_or(DocrimeExcitationError::ScalarConversion)?;
            *staged = *amplitude * (unit + unit - T::one());
            if !staged.is_finite() {
                return Err(DocrimeExcitationError::NonFiniteCalculation);
            }
        }

        output.copy_from_slice(&workspace.staged_excitation);
        workspace.random_state = splitmix64_advance(random_state, self.len() as u64);
        Ok(())
    }
}

impl<T> AlgorithmPlan for DocrimeBinaryExcitationPlan<T>
where
    T: MatrixElement + 'static,
{
    type Input<'a> = ();
    type Output<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type Workspace = DocrimeExcitationWorkspace<T>;
    type Error = DocrimeExcitationError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        (): Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if output.len() != self.len() {
            return Err(DocrimeExcitationError::OutputExtent {
                expected: self.len(),
                actual: output.len(),
            });
        }
        if workspace.staged_excitation.len() != self.len() {
            return Err(DocrimeExcitationError::WorkspaceExtent);
        }

        let random_state = workspace.random_state;
        for (draw_index, (staged, amplitude)) in workspace
            .staged_excitation
            .iter_mut()
            .zip(&self.amplitudes)
            .enumerate()
        {
            *staged = if splitmix64_at(random_state, draw_index as u64) >> 63 == 0 {
                -*amplitude
            } else {
                *amplitude
            };
        }

        output.copy_from_slice(&workspace.staged_excitation);
        workspace.random_state = splitmix64_advance(random_state, self.len() as u64);
        Ok(())
    }
}

/// Committed SplitMix64 state and staged storage for DO-CRIME excitation.
#[derive(Clone, Debug, PartialEq)]
pub struct DocrimeExcitationWorkspace<T> {
    random_state: u64,
    staged_excitation: Box<[T]>,
}

fn validate_amplitudes<T>(amplitudes: &[T]) -> Result<(), DocrimeExcitationPlanError>
where
    T: Float,
{
    if amplitudes.is_empty() {
        return Err(DocrimeExcitationPlanError::Empty);
    }
    if let Some(corrector_coordinate) = amplitudes
        .iter()
        .position(|amplitude| !amplitude.is_finite() || *amplitude <= T::zero())
    {
        return Err(DocrimeExcitationPlanError::InvalidAmplitude {
            corrector_coordinate,
        });
    }
    Ok(())
}

crate::filtergraph_algorithm! {
    /// Portable Float32 deterministic DO-CRIME excitation declaration.
    pub DocrimeExcitationF32 {
        label = "docrime-excitation-f32";
        configuration DocrimeExcitationF32Configuration {
            amplitudes: Vec<f32> = GraphRebuild;
            seed: u64 = Construction;
        }
        plan = DocrimeExcitationPlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            corrector_coordinate_count = plan.len();
        }
        inputs {}
        outputs {
            excitation: f32 [corrector_coordinate_count] = "excitation",
                schema = crate::schemas::DOCRIME_EXCITATION_V1;
        }
        parameters {
            amplitudes: f32 [corrector_coordinate_count] = "amplitudes",
                schema = crate::schemas::DOCRIME_EXCITATION_AMPLITUDE_V1,
                replace |plan, values| {
                    DocrimeExcitationPlan::new(values, plan.seed())
                };
        }
        parameter_error = DocrimeExcitationPlanError;
        prepare |configuration| {
            let plan = DocrimeExcitationPlan::new(
                configuration.amplitudes,
                configuration.seed,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = DocrimeExcitationPlanError;
        process |plan, workspace| {
            plan.process(excitation, workspace, ())
        }
        process_error = DocrimeExcitationError;
        reset |plan, workspace| {
            *workspace = plan.workspace();
        }
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 deterministic binary DO-CRIME excitation declaration.
    pub DocrimeBinaryExcitationF32 {
        label = "docrime-binary-excitation-f32";
        configuration DocrimeBinaryExcitationF32Configuration {
            amplitudes: Vec<f32> = GraphRebuild;
            seed: u64 = Construction;
        }
        plan = DocrimeBinaryExcitationPlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            corrector_coordinate_count = plan.len();
        }
        inputs {}
        outputs {
            excitation: f32 [corrector_coordinate_count] = "excitation",
                schema = crate::schemas::DOCRIME_EXCITATION_V1;
        }
        parameters {
            amplitudes: f32 [corrector_coordinate_count] = "amplitudes",
                schema = crate::schemas::DOCRIME_EXCITATION_AMPLITUDE_V1,
                replace |plan, values| {
                    DocrimeBinaryExcitationPlan::new(values, plan.seed())
                };
        }
        parameter_error = DocrimeExcitationPlanError;
        prepare |configuration| {
            let plan = DocrimeBinaryExcitationPlan::new(
                configuration.amplitudes,
                configuration.seed,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = DocrimeExcitationPlanError;
        process |plan, workspace| {
            plan.process(excitation, workspace, ())
        }
        process_error = DocrimeExcitationError;
        reset |plan, workspace| {
            *workspace = plan.workspace();
        }
    }
}
