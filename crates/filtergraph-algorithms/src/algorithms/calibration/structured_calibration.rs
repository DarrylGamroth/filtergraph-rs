//! Shared preparation rules for structured calibration probing and decoding.

use core::fmt;

use num_traits::Float;

/// Prepared extent named by a calibration-probe error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationProbeExtent {
    /// Two commands per zonal coordinate.
    ZonalCycle,
    /// Sylvester order containing all selected non-DC columns.
    HadamardOrder,
    /// Positive and negative command for every Hadamard row.
    HadamardCycle,
}

/// Invalid prepared structured calibration probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationProbePlanError {
    /// At least one corrector coordinate is required.
    Empty,
    /// One amplitude is non-finite or not positive.
    InvalidAmplitude {
        /// Zero-based corrector-coordinate index.
        corrector_coordinate: usize,
    },
    /// A sequence extent overflowed addressable storage.
    ExtentOverflow {
        /// Extent whose calculation overflowed.
        extent: CalibrationProbeExtent,
    },
}

impl fmt::Display for CalibrationProbePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("calibration probe amplitudes must be nonempty"),
            Self::InvalidAmplitude {
                corrector_coordinate,
            } => write!(
                formatter,
                "calibration probe amplitude for corrector coordinate {corrector_coordinate} must be finite and positive"
            ),
            Self::ExtentOverflow { extent } => {
                write!(formatter, "calibration probe {extent:?} extent overflow")
            }
        }
    }
}

impl core::error::Error for CalibrationProbePlanError {}

pub(super) fn validate_probe_amplitudes<T>(
    amplitudes: &[T],
) -> Result<(), CalibrationProbePlanError>
where
    T: Float,
{
    if amplitudes.is_empty() {
        return Err(CalibrationProbePlanError::Empty);
    }
    if let Some(corrector_coordinate) = amplitudes
        .iter()
        .position(|amplitude| !amplitude.is_finite() || *amplitude <= T::zero())
    {
        return Err(CalibrationProbePlanError::InvalidAmplitude {
            corrector_coordinate,
        });
    }
    Ok(())
}

pub(super) fn checked_hadamard_order(
    corrector_count: usize,
) -> Result<usize, CalibrationProbePlanError> {
    corrector_count
        .checked_add(1)
        .and_then(usize::checked_next_power_of_two)
        .ok_or(CalibrationProbePlanError::ExtentOverflow {
            extent: CalibrationProbeExtent::HadamardOrder,
        })
}

pub(super) fn hadamard_sign_is_positive(row: usize, corrector_coordinate: usize) -> bool {
    (row & (corrector_coordinate + 1))
        .count_ones()
        .is_multiple_of(2)
}

#[cfg(test)]
mod tests {
    use super::{CalibrationProbeExtent, CalibrationProbePlanError, checked_hadamard_order};

    #[test]
    fn hadamard_order_overflow_is_rejected() {
        assert_eq!(
            checked_hadamard_order(usize::MAX),
            Err(CalibrationProbePlanError::ExtentOverflow {
                extent: CalibrationProbeExtent::HadamardOrder,
            })
        );
    }
}
