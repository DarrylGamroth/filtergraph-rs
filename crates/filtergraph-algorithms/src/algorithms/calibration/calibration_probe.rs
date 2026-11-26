//! Deterministic structured excitation for interaction-matrix calibration.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use num_traits::Float;

use crate::{AlgorithmPlan, MatrixElement};

use super::structured_calibration::{
    CalibrationProbeExtent, CalibrationProbePlanError, checked_hadamard_order,
    hadamard_sign_is_positive, validate_probe_amplitudes,
};

/// Invalid structured calibration-probe execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationProbeError {
    /// The caller-owned excitation output has the wrong extent.
    OutputExtent {
        /// Required corrector-coordinate extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },
    /// The supplied workspace was prepared for a different plan.
    WorkspaceExtent,
}

impl fmt::Display for CalibrationProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputExtent { expected, actual } => write!(
                formatter,
                "calibration probe output extent {actual} does not match {expected}"
            ),
            Self::WorkspaceExtent => {
                formatter.write_str("calibration probe workspace extent is invalid")
            }
        }
    }
}

impl core::error::Error for CalibrationProbeError {}

/// Global sign of one structured calibration command.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CalibrationProbePolarity {
    /// Positive zonal command, Hadamard row, or modal command.
    #[default]
    Positive,
    /// Exact negative of the preceding positive command or row.
    Negative,
}

/// Metadata identifying one command in a structured calibration sequence.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CalibrationProbeStep {
    pattern_index: usize,
    polarity: CalibrationProbePolarity,
    completes_cycle: bool,
}

impl CalibrationProbeStep {
    /// Returns the zonal coordinate, Hadamard row, or modal row for this step.
    pub const fn pattern_index(self) -> usize {
        self.pattern_index
    }

    /// Returns the global command polarity.
    pub const fn polarity(self) -> CalibrationProbePolarity {
        self.polarity
    }

    /// Returns whether this command is the last one in the prepared cycle.
    pub const fn completes_cycle(self) -> bool {
        self.completes_cycle
    }
}

/// Caller-owned coherent structured excitation and sequence metadata.
#[derive(Debug)]
pub struct CalibrationProbeOutput<'a, T> {
    excitation: &'a mut [T],
    step: &'a mut CalibrationProbeStep,
}

impl<'a, T> CalibrationProbeOutput<'a, T> {
    /// Groups an excitation vector and its metadata destination.
    pub fn new(excitation: &'a mut [T], step: &'a mut CalibrationProbeStep) -> Self {
        Self { excitation, step }
    }

    /// Returns the complete corrector-coordinate excitation.
    pub fn excitation(&self) -> &[T] {
        self.excitation
    }

    /// Returns the metadata published with the excitation.
    pub const fn step(&self) -> CalibrationProbeStep {
        *self.step
    }
}

/// Invalid prepared general modal push/pull probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModalPushPullPlanError {
    /// Mode and corrector-coordinate dimensions must both be nonzero.
    EmptyDimension,

    /// A command-matrix or cycle extent overflowed addressable storage.
    ExtentOverflow,

    /// The positive-command matrix has the wrong row-major extent.
    CommandExtent {
        /// Required matrix-coefficient extent.
        expected: usize,
        /// Supplied matrix-coefficient extent.
        actual: usize,
    },

    /// A positive command contains a non-finite coefficient.
    NonFiniteCommand {
        /// Zero-based modal row.
        mode: usize,
        /// Zero-based corrector coordinate.
        corrector_coordinate: usize,
    },

    /// A positive modal command contains only zero coefficients.
    ZeroCommand {
        /// Zero-based modal row.
        mode: usize,
    },
}

impl fmt::Display for ModalPushPullPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension => {
                formatter.write_str("modal push/pull dimensions must be nonzero")
            }
            Self::ExtentOverflow => formatter.write_str("modal push/pull extent overflow"),
            Self::CommandExtent { expected, actual } => write!(
                formatter,
                "modal positive-command extent {actual} does not match {expected}"
            ),
            Self::NonFiniteCommand {
                mode,
                corrector_coordinate,
            } => write!(
                formatter,
                "modal positive command {mode} at corrector coordinate {corrector_coordinate} must be finite"
            ),
            Self::ZeroCommand { mode } => {
                write!(formatter, "modal positive command {mode} must be nonzero")
            }
        }
    }
}

impl core::error::Error for ModalPushPullPlanError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PushPullSequence {
    pattern_count: usize,
    pattern_index: usize,
    polarity: CalibrationProbePolarity,
}

impl PushPullSequence {
    const fn new(pattern_count: usize) -> Self {
        Self {
            pattern_count,
            pattern_index: 0,
            polarity: CalibrationProbePolarity::Positive,
        }
    }

    const fn is_compatible(self, pattern_count: usize) -> bool {
        self.pattern_count == pattern_count && self.pattern_index < pattern_count
    }

    const fn step(self) -> CalibrationProbeStep {
        CalibrationProbeStep {
            pattern_index: self.pattern_index,
            polarity: self.polarity,
            completes_cycle: matches!(self.polarity, CalibrationProbePolarity::Negative)
                && self.pattern_index + 1 == self.pattern_count,
        }
    }

    fn advance(&mut self) {
        match self.polarity {
            CalibrationProbePolarity::Positive => {
                self.polarity = CalibrationProbePolarity::Negative;
            }
            CalibrationProbePolarity::Negative => {
                self.polarity = CalibrationProbePolarity::Positive;
                self.pattern_index = if self.pattern_index + 1 == self.pattern_count {
                    0
                } else {
                    self.pattern_index + 1
                };
            }
        }
    }
}

/// Prepared arbitrary modal positive/negative calibration sequence.
///
/// `positive_commands` is row-major with `mode_count` rows and
/// `corrector_coordinate_count` columns. Each row is the exact positive
/// corrector command, including its selected probe amplitude. Processing emits
/// that row and then its exact negative before advancing to the next mode.
/// The plan does not infer basis normalization, orthogonality, rank, Fourier
/// meaning, or physical stroke safety.
#[derive(Clone, Debug, PartialEq)]
pub struct ModalPushPullPlan<T> {
    mode_count: usize,
    corrector_coordinate_count: usize,
    cycle_len: usize,
    positive_commands: Arc<[T]>,
}

impl<T> ModalPushPullPlan<T>
where
    T: Float,
{
    /// Validates and retains exact positive commands in modal row order.
    pub fn new(
        mode_count: usize,
        corrector_coordinate_count: usize,
        positive_commands: impl Into<Arc<[T]>>,
    ) -> Result<Self, ModalPushPullPlanError> {
        if mode_count == 0 || corrector_coordinate_count == 0 {
            return Err(ModalPushPullPlanError::EmptyDimension);
        }
        let command_count = mode_count
            .checked_mul(corrector_coordinate_count)
            .ok_or(ModalPushPullPlanError::ExtentOverflow)?;
        let cycle_len = mode_count
            .checked_mul(2)
            .ok_or(ModalPushPullPlanError::ExtentOverflow)?;
        let positive_commands = positive_commands.into();
        if positive_commands.len() != command_count {
            return Err(ModalPushPullPlanError::CommandExtent {
                expected: command_count,
                actual: positive_commands.len(),
            });
        }
        for mode in 0..mode_count {
            let row = &positive_commands
                [mode * corrector_coordinate_count..(mode + 1) * corrector_coordinate_count];
            if let Some(corrector_coordinate) =
                row.iter().position(|coefficient| !coefficient.is_finite())
            {
                return Err(ModalPushPullPlanError::NonFiniteCommand {
                    mode,
                    corrector_coordinate,
                });
            }
            if row.iter().all(|coefficient| *coefficient == T::zero()) {
                return Err(ModalPushPullPlanError::ZeroCommand { mode });
            }
        }

        Ok(Self {
            mode_count,
            corrector_coordinate_count,
            cycle_len,
            positive_commands,
        })
    }

    /// Returns the number of prepared modal rows.
    pub const fn mode_count(&self) -> usize {
        self.mode_count
    }

    /// Returns the corrector-coordinate extent of every command.
    pub const fn corrector_coordinate_count(&self) -> usize {
        self.corrector_coordinate_count
    }

    /// Returns exact positive commands in modal row-major order.
    pub fn positive_commands(&self) -> &[T] {
        &self.positive_commands
    }

    /// Replaces the modal command matrix without changing the sequence topology.
    pub fn with_positive_commands(
        &self,
        positive_commands: Arc<[T]>,
    ) -> Result<Self, ModalPushPullPlanError> {
        Self::new(
            self.mode_count,
            self.corrector_coordinate_count,
            positive_commands,
        )
    }

    /// Returns the number of commands in one complete modal cycle.
    pub const fn cycle_len(&self) -> usize {
        self.cycle_len
    }

    /// Creates positive-mode-zero sequence history for one owner.
    pub const fn workspace(&self) -> ModalPushPullWorkspace {
        ModalPushPullWorkspace {
            corrector_coordinate_count: self.corrector_coordinate_count,
            sequence: PushPullSequence::new(self.mode_count),
        }
    }
}

impl<T> AlgorithmPlan for ModalPushPullPlan<T>
where
    T: MatrixElement + 'static,
{
    type Input<'a> = ();
    type Output<'a>
        = CalibrationProbeOutput<'a, T>
    where
        Self: 'a;
    type Workspace = ModalPushPullWorkspace;
    type Error = CalibrationProbeError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        (): Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        validate_output_extent(self.corrector_coordinate_count, output.excitation.len())?;
        if workspace.corrector_coordinate_count != self.corrector_coordinate_count
            || !workspace.sequence.is_compatible(self.mode_count)
        {
            return Err(CalibrationProbeError::WorkspaceExtent);
        }

        let sequence = workspace.sequence;
        let row_start = sequence.pattern_index * self.corrector_coordinate_count;
        let positive_command =
            &self.positive_commands[row_start..row_start + self.corrector_coordinate_count];
        match sequence.polarity {
            CalibrationProbePolarity::Positive => {
                output.excitation.copy_from_slice(positive_command);
            }
            CalibrationProbePolarity::Negative => {
                for (excitation, coefficient) in output.excitation.iter_mut().zip(positive_command)
                {
                    *excitation = -*coefficient;
                }
            }
        }
        *output.step = sequence.step();
        workspace.sequence.advance();
        Ok(())
    }
}

/// Mode and polarity for the next arbitrary modal command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModalPushPullWorkspace {
    corrector_coordinate_count: usize,
    sequence: PushPullSequence,
}

impl ModalPushPullWorkspace {
    /// Restarts the prepared sequence at positive mode zero.
    pub fn reset(&mut self) {
        self.sequence = PushPullSequence::new(self.sequence.pattern_count);
    }
}

/// Prepared zonal positive/negative calibration sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct ZonalPushPullPlan<T> {
    amplitudes: Box<[T]>,
    cycle_len: usize,
}

impl<T> ZonalPushPullPlan<T>
where
    T: Float,
{
    /// Validates positive amplitudes and the two-command-per-coordinate cycle.
    pub fn new(amplitudes: impl Into<Box<[T]>>) -> Result<Self, CalibrationProbePlanError> {
        let amplitudes = amplitudes.into();
        validate_probe_amplitudes(&amplitudes)?;
        let cycle_len =
            amplitudes
                .len()
                .checked_mul(2)
                .ok_or(CalibrationProbePlanError::ExtentOverflow {
                    extent: CalibrationProbeExtent::ZonalCycle,
                })?;
        Ok(Self {
            amplitudes,
            cycle_len,
        })
    }

    /// Returns the corrector-coordinate extent.
    pub fn len(&self) -> usize {
        self.amplitudes.len()
    }

    /// Returns whether the corrector-coordinate extent is zero.
    pub fn is_empty(&self) -> bool {
        self.amplitudes.is_empty()
    }

    /// Returns the prepared per-coordinate amplitudes.
    pub fn amplitudes(&self) -> &[T] {
        &self.amplitudes
    }

    /// Returns the number of commands in one complete zonal cycle.
    pub const fn cycle_len(&self) -> usize {
        self.cycle_len
    }

    /// Creates positive-coordinate-zero sequence history for one owner.
    pub fn workspace(&self) -> ZonalPushPullWorkspace {
        ZonalPushPullWorkspace {
            sequence: PushPullSequence::new(self.len()),
        }
    }
}

impl<T> AlgorithmPlan for ZonalPushPullPlan<T>
where
    T: MatrixElement + 'static,
{
    type Input<'a> = ();
    type Output<'a>
        = CalibrationProbeOutput<'a, T>
    where
        Self: 'a;
    type Workspace = ZonalPushPullWorkspace;
    type Error = CalibrationProbeError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        (): Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        validate_output_extent(self.len(), output.excitation.len())?;
        if !workspace.sequence.is_compatible(self.len()) {
            return Err(CalibrationProbeError::WorkspaceExtent);
        }

        let sequence = workspace.sequence;
        let corrector_coordinate = sequence.pattern_index;
        output.excitation.fill(T::zero());
        output.excitation[corrector_coordinate] = match sequence.polarity {
            CalibrationProbePolarity::Positive => self.amplitudes[corrector_coordinate],
            CalibrationProbePolarity::Negative => -self.amplitudes[corrector_coordinate],
        };
        *output.step = sequence.step();
        workspace.sequence.advance();
        Ok(())
    }
}

/// Zonal coordinate and polarity for the next command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZonalPushPullWorkspace {
    sequence: PushPullSequence,
}

impl ZonalPushPullWorkspace {
    /// Restarts the prepared sequence at positive coordinate zero.
    pub fn reset(&mut self) {
        self.sequence = PushPullSequence::new(self.sequence.pattern_count);
    }
}

/// Prepared balanced Sylvester-Hadamard positive/negative sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct HadamardPushPullPlan<T> {
    amplitudes: Box<[T]>,
    order: usize,
    cycle_len: usize,
}

impl<T> HadamardPushPullPlan<T>
where
    T: Float,
{
    /// Validates amplitudes and prepares the smallest usable Sylvester order.
    pub fn new(amplitudes: impl Into<Box<[T]>>) -> Result<Self, CalibrationProbePlanError> {
        let amplitudes = amplitudes.into();
        validate_probe_amplitudes(&amplitudes)?;
        let order = checked_hadamard_order(amplitudes.len())?;
        let cycle_len = order
            .checked_mul(2)
            .ok_or(CalibrationProbePlanError::ExtentOverflow {
                extent: CalibrationProbeExtent::HadamardCycle,
            })?;
        Ok(Self {
            amplitudes,
            order,
            cycle_len,
        })
    }

    /// Returns the corrector-coordinate extent.
    pub fn len(&self) -> usize {
        self.amplitudes.len()
    }

    /// Returns whether the corrector-coordinate extent is zero.
    pub fn is_empty(&self) -> bool {
        self.amplitudes.is_empty()
    }

    /// Returns the prepared per-coordinate amplitudes.
    pub fn amplitudes(&self) -> &[T] {
        &self.amplitudes
    }

    /// Returns the prepared Sylvester-Hadamard order.
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Returns the number of commands in one complete paired-row cycle.
    pub const fn cycle_len(&self) -> usize {
        self.cycle_len
    }

    /// Creates positive-row-zero sequence history for one owner.
    pub fn workspace(&self) -> HadamardPushPullWorkspace {
        HadamardPushPullWorkspace {
            corrector_count: self.len(),
            sequence: PushPullSequence::new(self.order),
        }
    }
}

impl<T> AlgorithmPlan for HadamardPushPullPlan<T>
where
    T: MatrixElement + 'static,
{
    type Input<'a> = ();
    type Output<'a>
        = CalibrationProbeOutput<'a, T>
    where
        Self: 'a;
    type Workspace = HadamardPushPullWorkspace;
    type Error = CalibrationProbeError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        (): Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        validate_output_extent(self.len(), output.excitation.len())?;
        if workspace.corrector_count != self.len() || !workspace.sequence.is_compatible(self.order)
        {
            return Err(CalibrationProbeError::WorkspaceExtent);
        }

        let sequence = workspace.sequence;
        let row = sequence.pattern_index;
        for (corrector_coordinate, (excitation, amplitude)) in output
            .excitation
            .iter_mut()
            .zip(&self.amplitudes)
            .enumerate()
        {
            let base_positive = hadamard_sign_is_positive(row, corrector_coordinate);
            let positive =
                base_positive == (sequence.polarity == CalibrationProbePolarity::Positive);
            *excitation = if positive { *amplitude } else { -*amplitude };
        }
        *output.step = sequence.step();
        workspace.sequence.advance();
        Ok(())
    }
}

/// Hadamard row and polarity for the next command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HadamardPushPullWorkspace {
    corrector_count: usize,
    sequence: PushPullSequence,
}

impl HadamardPushPullWorkspace {
    /// Restarts the prepared sequence at positive Hadamard row zero.
    pub fn reset(&mut self) {
        self.sequence = PushPullSequence::new(self.sequence.pattern_count);
    }
}

fn validate_output_extent(expected: usize, actual: usize) -> Result<(), CalibrationProbeError> {
    if actual != expected {
        return Err(CalibrationProbeError::OutputExtent { expected, actual });
    }
    Ok(())
}

crate::filtergraph_algorithm! {
    /// Portable Float32 arbitrary modal push/pull source declaration.
    pub ModalPushPullF32 {
        label = "modal-push-pull-f32";
        configuration ModalPushPullF32Configuration {
            mode_count: usize = GraphRebuild;
            corrector_coordinate_count: usize = GraphRebuild;
            initial_positive_commands: Vec<f32> = Construction;
            excitation_schema: String = GraphRebuild;
        }
        plan = ModalPushPullPlan<f32>;
        workspace = ModalPushPullWorkspace;
        properties = none;
        dimensions |plan, _workspace| {
            mode_count = plan.mode_count();
            corrector_coordinate_count = plan.corrector_coordinate_count();
        }
        inputs {}
        outputs {
            excitation: f32 [corrector_coordinate_count] = "excitation",
                schema = configuration.excitation_schema.as_str();
            pattern_index: u64 [1] = "pattern-index",
                schema = crate::schemas::CALIBRATION_PATTERN_INDEX_V1;
            polarity: u8 [1] = "polarity",
                schema = crate::schemas::CALIBRATION_PROBE_POLARITY_V1;
            completes_cycle: bool [1] = "completes-cycle",
                schema = crate::schemas::CALIBRATION_CYCLE_COMPLETE_V1;
        }
        parameters {
            positive_commands: f32 [mode_count, corrector_coordinate_count]
                = "positive-commands",
                schema = crate::schemas::MODAL_POSITIVE_COMMANDS_V1,
                replace |plan, values| {
                    plan.with_positive_commands(Arc::from(values))
                };
        }
        parameter_error = ModalPushPullPlanError;
        prepare |configuration| {
            let plan = ModalPushPullPlan::new(
                configuration.mode_count,
                configuration.corrector_coordinate_count,
                configuration.initial_positive_commands,
            )?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = ModalPushPullPlanError;
        process |plan, workspace| {
            let mut step = CalibrationProbeStep::default();
            plan.process(
                CalibrationProbeOutput::new(excitation, &mut step),
                workspace,
                (),
            )?;
            pattern_index[0] = step.pattern_index() as u64;
            polarity[0] = match step.polarity() {
                CalibrationProbePolarity::Positive => 0,
                CalibrationProbePolarity::Negative => 1,
            };
            completes_cycle[0] = step.completes_cycle();
            Ok(())
        }
        process_error = CalibrationProbeError;
        reset |_plan, workspace| {
            workspace.reset();
        }
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 balanced Hadamard push/pull source declaration.
    pub HadamardPushPullF32 {
        label = "hadamard-push-pull-f32";
        configuration HadamardPushPullF32Configuration {
            amplitudes: Vec<f32> = GraphRebuild;
            excitation_schema: String = GraphRebuild;
        }
        plan = HadamardPushPullPlan<f32>;
        workspace = HadamardPushPullWorkspace;
        properties = none;
        dimensions |plan, _workspace| {
            corrector_coordinate_count = plan.len();
        }
        inputs {}
        outputs {
            excitation: f32 [corrector_coordinate_count] = "excitation",
                schema = configuration.excitation_schema.as_str();
            pattern_index: u64 [1] = "pattern-index",
                schema = crate::schemas::CALIBRATION_PATTERN_INDEX_V1;
            polarity: u8 [1] = "polarity",
                schema = crate::schemas::CALIBRATION_PROBE_POLARITY_V1;
            completes_cycle: bool [1] = "completes-cycle",
                schema = crate::schemas::CALIBRATION_CYCLE_COMPLETE_V1;
        }
        parameters {
            amplitudes: f32 [corrector_coordinate_count] = "amplitudes",
                schema = crate::schemas::CALIBRATION_PROBE_AMPLITUDES_V1,
                replace |_plan, values| {
                    HadamardPushPullPlan::new(values)
                };
        }
        parameter_error = CalibrationProbePlanError;
        prepare |configuration| {
            let plan = HadamardPushPullPlan::new(configuration.amplitudes)?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = CalibrationProbePlanError;
        process |plan, workspace| {
            let mut step = CalibrationProbeStep::default();
            plan.process(
                CalibrationProbeOutput::new(excitation, &mut step),
                workspace,
                (),
            )?;
            pattern_index[0] = step.pattern_index() as u64;
            polarity[0] = match step.polarity() {
                CalibrationProbePolarity::Positive => 0,
                CalibrationProbePolarity::Negative => 1,
            };
            completes_cycle[0] = step.completes_cycle();
            Ok(())
        }
        process_error = CalibrationProbeError;
        reset |_plan, workspace| {
            workspace.reset();
        }
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 zonal push/pull source declaration.
    pub ZonalPushPullF32 {
        label = "zonal-push-pull-f32";
        configuration ZonalPushPullF32Configuration {
            amplitudes: Vec<f32> = GraphRebuild;
            excitation_schema: String = GraphRebuild;
        }
        plan = ZonalPushPullPlan<f32>;
        workspace = ZonalPushPullWorkspace;
        properties = none;
        dimensions |plan, _workspace| {
            corrector_coordinate_count = plan.len();
        }
        inputs {}
        outputs {
            excitation: f32 [corrector_coordinate_count] = "excitation",
                schema = configuration.excitation_schema.as_str();
            pattern_index: u64 [1] = "pattern-index",
                schema = crate::schemas::CALIBRATION_PATTERN_INDEX_V1;
            polarity: u8 [1] = "polarity",
                schema = crate::schemas::CALIBRATION_PROBE_POLARITY_V1;
            completes_cycle: bool [1] = "completes-cycle",
                schema = crate::schemas::CALIBRATION_CYCLE_COMPLETE_V1;
        }
        parameters {
            amplitudes: f32 [corrector_coordinate_count] = "amplitudes",
                schema = crate::schemas::CALIBRATION_PROBE_AMPLITUDES_V1,
                replace |_plan, values| {
                    ZonalPushPullPlan::new(values)
                };
        }
        parameter_error = CalibrationProbePlanError;
        prepare |configuration| {
            let plan = ZonalPushPullPlan::new(configuration.amplitudes)?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = CalibrationProbePlanError;
        process |plan, workspace| {
            let mut step = CalibrationProbeStep::default();
            plan.process(
                CalibrationProbeOutput::new(excitation, &mut step),
                workspace,
                (),
            )?;
            pattern_index[0] = step.pattern_index() as u64;
            polarity[0] = match step.polarity() {
                CalibrationProbePolarity::Positive => 0,
                CalibrationProbePolarity::Negative => 1,
            };
            completes_cycle[0] = step.completes_cycle();
            Ok(())
        }
        process_error = CalibrationProbeError;
        reset |_plan, workspace| {
            workspace.reset();
        }
    }
}
