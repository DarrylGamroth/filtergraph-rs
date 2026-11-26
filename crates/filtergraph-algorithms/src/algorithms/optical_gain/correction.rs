//! Reciprocal correction of reconstructed modes by physical optical gain.

use alloc::{sync::Arc, vec::Vec};
use core::fmt;

use num_traits::Float;

use crate::{AlgorithmPlan, NoWorkspace};

/// Invalid prepared reciprocal optical-gain correction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainCorrectionPlanError {
    /// At least one reconstructed mode is required.
    Empty,

    /// A prepared physical optical gain is non-finite or nonpositive.
    InvalidOpticalGain {
        /// Zero-based reconstructed-mode index.
        mode: usize,
    },

    /// Inverting a positive physical optical gain produced a non-finite value.
    NonFiniteReciprocalCorrection {
        /// Zero-based reconstructed-mode index.
        mode: usize,
    },
}

impl fmt::Display for OpticalGainCorrectionPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("optical-gain correction must be nonempty"),
            Self::InvalidOpticalGain { mode } => write!(
                formatter,
                "physical optical gain for mode {mode} must be finite and positive"
            ),
            Self::NonFiniteReciprocalCorrection { mode } => write!(
                formatter,
                "reciprocal optical-gain correction for mode {mode} is not finite"
            ),
        }
    }
}

impl core::error::Error for OpticalGainCorrectionPlanError {}

/// Invalid reconstructed modal input, output, or corrected result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpticalGainCorrectionError {
    /// The raw reconstructed input has the wrong modal extent.
    InputExtent {
        /// Required modal extent.
        expected: usize,
        /// Supplied modal extent.
        actual: usize,
    },

    /// The corrected output has the wrong modal extent.
    OutputExtent {
        /// Required modal extent.
        expected: usize,
        /// Supplied modal extent.
        actual: usize,
    },

    /// A raw reconstructed input mode is non-finite.
    NonFiniteInput {
        /// Zero-based reconstructed-mode index.
        mode: usize,
    },

    /// Applying a reciprocal correction produced a non-finite result.
    NonFiniteOutput {
        /// Zero-based reconstructed-mode index.
        mode: usize,
    },
}

impl fmt::Display for OpticalGainCorrectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent { expected, actual } => write!(
                formatter,
                "optical-gain correction input extent {actual} does not match {expected}"
            ),
            Self::OutputExtent { expected, actual } => write!(
                formatter,
                "optical-gain correction output extent {actual} does not match {expected}"
            ),
            Self::NonFiniteInput { mode } => write!(
                formatter,
                "raw reconstructed input for optical-gain mode {mode} is not finite"
            ),
            Self::NonFiniteOutput { mode } => write!(
                formatter,
                "corrected reconstructed output for optical-gain mode {mode} is not finite"
            ),
        }
    }
}

impl core::error::Error for OpticalGainCorrectionError {}

/// Prepared reciprocal correction of reconstructed modes.
///
/// `optical_gains` are physical sensor-response ratios. Preparation stores
/// `1 / optical_gains[m]`; processing multiplies raw, uncorrected reconstructed
/// modes by those reciprocal corrections. This plan does not estimate gain or
/// infer it from frame intensity.
#[derive(Clone, Debug, PartialEq)]
pub struct OpticalGainCorrectionPlan<T> {
    optical_gains: Arc<[T]>,
    reciprocal_corrections: Arc<[T]>,
}

impl<T> OpticalGainCorrectionPlan<T>
where
    T: Float,
{
    /// Validates physical optical gains and prepares their reciprocals.
    pub fn new(optical_gains: impl Into<Arc<[T]>>) -> Result<Self, OpticalGainCorrectionPlanError> {
        let optical_gains = optical_gains.into();
        if optical_gains.is_empty() {
            return Err(OpticalGainCorrectionPlanError::Empty);
        }
        let mut reciprocal_corrections = Vec::with_capacity(optical_gains.len());
        for (mode, optical_gain) in optical_gains.iter().enumerate() {
            if !optical_gain.is_finite() || *optical_gain <= T::zero() {
                return Err(OpticalGainCorrectionPlanError::InvalidOpticalGain { mode });
            }
            let reciprocal = T::one() / *optical_gain;
            if !reciprocal.is_finite() {
                return Err(OpticalGainCorrectionPlanError::NonFiniteReciprocalCorrection { mode });
            }
            reciprocal_corrections.push(reciprocal);
        }
        Ok(Self {
            optical_gains,
            reciprocal_corrections: reciprocal_corrections.into(),
        })
    }

    /// Returns the reconstructed modal extent.
    pub fn len(&self) -> usize {
        self.optical_gains.len()
    }

    /// Returns whether the reconstructed modal extent is zero.
    pub fn is_empty(&self) -> bool {
        self.optical_gains.is_empty()
    }

    /// Returns the prepared physical optical gains.
    pub fn optical_gains(&self) -> &[T] {
        &self.optical_gains
    }

    /// Returns shared ownership of the prepared physical optical gains.
    pub fn shared_optical_gains(&self) -> Arc<[T]> {
        Arc::clone(&self.optical_gains)
    }

    /// Returns the prepared reciprocal corrections applied during processing.
    pub fn reciprocal_corrections(&self) -> &[T] {
        &self.reciprocal_corrections
    }

    /// Returns shared ownership of the prepared reciprocal corrections.
    pub fn shared_reciprocal_corrections(&self) -> Arc<[T]> {
        Arc::clone(&self.reciprocal_corrections)
    }
}

impl<T> AlgorithmPlan for OpticalGainCorrectionPlan<T>
where
    T: Float + 'static,
{
    type Input<'a>
        = &'a [T]
    where
        Self: 'a;
    type Output<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type Workspace = NoWorkspace;
    type Error = OpticalGainCorrectionError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if input.len() != self.len() {
            return Err(OpticalGainCorrectionError::InputExtent {
                expected: self.len(),
                actual: input.len(),
            });
        }
        if output.len() != self.len() {
            return Err(OpticalGainCorrectionError::OutputExtent {
                expected: self.len(),
                actual: output.len(),
            });
        }
        for (mode, input) in input.iter().enumerate() {
            if !input.is_finite() {
                return Err(OpticalGainCorrectionError::NonFiniteInput { mode });
            }
            if !(*input * self.reciprocal_corrections[mode]).is_finite() {
                return Err(OpticalGainCorrectionError::NonFiniteOutput { mode });
            }
        }
        for ((destination, input), reciprocal) in output
            .iter_mut()
            .zip(input)
            .zip(self.reciprocal_corrections.iter())
        {
            *destination = *input * *reciprocal;
        }
        Ok(())
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 reciprocal optical-gain correction declaration.
    pub OpticalGainCorrectionF32 {
        label = "optical-gain-correction-f32";
        configuration OpticalGainCorrectionF32Configuration {
            optical_gains: Vec<f32> = Construction;
        }
        plan = OpticalGainCorrectionPlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            mode_count = plan.len();
        }
        inputs {
            reconstructed: f32 [mode_count] = "reconstructed",
                schema = crate::schemas::RECONSTRUCTED_MODES_V1;
        }
        outputs {
            corrected: f32 [mode_count] = "corrected",
                schema = crate::schemas::RECONSTRUCTED_MODES_V1,
                metadata_from = "reconstructed";
        }
        parameters {
            optical_gains: f32 [mode_count] = "optical-gains",
                schema = crate::schemas::PHYSICAL_OPTICAL_GAIN_V1,
                replace |_plan, values| {
                    OpticalGainCorrectionPlan::new(values)
                };
        }
        parameter_error = OpticalGainCorrectionPlanError;
        prepare |configuration| {
            Ok((
                OpticalGainCorrectionPlan::new(configuration.optical_gains)?,
                NoWorkspace,
            ))
        }
        prepare_error = OpticalGainCorrectionPlanError;
        process |plan, workspace| {
            plan.process(corrected, workspace, reconstructed)
        }
        process_error = OpticalGainCorrectionError;
    }
}
