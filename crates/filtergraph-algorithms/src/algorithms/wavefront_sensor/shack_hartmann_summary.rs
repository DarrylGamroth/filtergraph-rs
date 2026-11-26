//! Eight-field Shack–Hartmann diagnostics with Float64 accumulation.
use crate::{AlgorithmPlan, NoWorkspace};
use core::fmt;
use num_traits::Float;

/// Invalid summary configuration or caller storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShackHartmannSummaryError {
    /// Counts must remain exactly representable in Float32: `1..=2^24`.
    SubapertureCount,
    /// A caller-owned input or output has the wrong packed extent.
    PortExtent {
        /// Scientific port name.
        port: &'static str,
        /// Required scalar element count.
        expected: usize,
        /// Supplied scalar element count.
        actual: usize,
    },
}
impl fmt::Display for ShackHartmannSummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SubapertureCount => {
                f.write_str("subaperture count must be in 1:2^24 for exact Float32 counts")
            }
            Self::PortExtent {
                port,
                expected,
                actual,
            } => write!(f, "{port} extent {actual} does not match {expected}"),
        }
    }
}
impl core::error::Error for ShackHartmannSummaryError {}

/// Immutable count for a complete-frame Shack–Hartmann diagnostic.
///
/// Input slopes are pair-interleaved X/Y coordinates. Summary fields are total
/// count, valid count, mean valid flux, uncentered X/Y RMS, their hypotenuse,
/// and mean X/Y slope. Invalid rows are ignored. Nonfinite valid rows or no
/// valid rows produce six NaNs after the two counts and processing succeeds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShackHartmannSummaryPlan {
    subaperture_count: usize,
}
impl ShackHartmannSummaryPlan {
    /// Validate the exact Float32 count domain without allocating frame storage.
    pub fn new(subaperture_count: usize) -> Result<Self, ShackHartmannSummaryError> {
        if subaperture_count == 0 || subaperture_count > (1 << 24) {
            return Err(ShackHartmannSummaryError::SubapertureCount);
        }
        Ok(Self { subaperture_count })
    }
    /// Return the fixed number of subapertures.
    pub const fn subaperture_count(&self) -> usize {
        self.subaperture_count
    }
}
impl AlgorithmPlan for ShackHartmannSummaryPlan {
    type Input<'a> = (&'a [f32], &'a [f32], &'a [bool]);
    type Output<'a> = &'a mut [f32];
    type Workspace = NoWorkspace;
    type Error = ShackHartmannSummaryError;
    fn process<'input, 'output>(
        &self,
        summary: Self::Output<'output>,
        _: &mut NoWorkspace,
        (slopes, flux, validity): Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        for (port, actual, expected) in [
            ("slopes", slopes.len(), 2 * self.subaperture_count),
            ("flux", flux.len(), self.subaperture_count),
            ("validity", validity.len(), self.subaperture_count),
            ("summary", summary.len(), 8),
        ] {
            if actual != expected {
                return Err(ShackHartmannSummaryError::PortExtent {
                    port,
                    expected,
                    actual,
                });
            }
        }
        let mut valid_count = 0_usize;
        let mut finite = true;
        let mut flux_sum = 0.0_f64;
        let mut x_sum = 0.0_f64;
        let mut y_sum = 0.0_f64;
        let mut x_square_sum = 0.0_f64;
        let mut y_square_sum = 0.0_f64;
        for subaperture in 0..self.subaperture_count {
            if validity[subaperture] {
                valid_count += 1;
                let f = f64::from(flux[subaperture]);
                let x = f64::from(slopes[2 * subaperture]);
                let y = f64::from(slopes[2 * subaperture + 1]);
                finite &= f.is_finite() && x.is_finite() && y.is_finite();
                flux_sum += f;
                x_sum += x;
                y_sum += y;
                x_square_sum += x * x;
                y_square_sum += y * y;
            }
        }
        summary[0] = self.subaperture_count as f32;
        summary[1] = valid_count as f32;
        if valid_count == 0 || !finite {
            summary[2..].fill(f32::NAN);
        } else {
            let n = valid_count as f64;
            let rms_x = <f64 as Float>::sqrt(x_square_sum / n);
            let rms_y = <f64 as Float>::sqrt(y_square_sum / n);
            summary[2] = (flux_sum / n) as f32;
            summary[3] = rms_x as f32;
            summary[4] = rms_y as f32;
            summary[5] = <f64 as Float>::hypot(rms_x, rms_y) as f32;
            summary[6] = (x_sum / n) as f32;
            summary[7] = (y_sum / n) as f32;
        }
        Ok(())
    }
}
crate::filtergraph_algorithm! {
    /// Float64-accumulated diagnostics; metadata follows the current slopes sample.
    pub ShackHartmannSummaryF32 {
        label = "shack-hartmann-summary-f32";
        configuration ShackHartmannSummaryF32Configuration { subaperture_count: usize = GraphRebuild; }
        plan = ShackHartmannSummaryPlan;
        workspace = NoWorkspace;
        properties = none;
        dimensions |plan, _workspace| { subaperture_count = plan.subaperture_count(); }
        inputs {
            slopes: f32 [subaperture_count,2] = "slopes", schema = crate::schemas::SHACK_HARTMANN_SLOPES_V1;
            flux: f32 [subaperture_count] = "flux", schema = crate::schemas::SHACK_HARTMANN_FLUX_V1;
            validity: bool [subaperture_count] = "validity", schema = crate::schemas::SHACK_HARTMANN_VALIDITY_V1;
        }
        outputs { summary: f32 [8] = "summary", schema = crate::schemas::SHACK_HARTMANN_SUMMARY_V1, metadata_from = "slopes"; }
        prepare |configuration| { Ok((ShackHartmannSummaryPlan::new(configuration.subaperture_count)?,NoWorkspace)) }
        prepare_error = ShackHartmannSummaryError;
        process |plan,workspace| { plan.process(summary,workspace,(slopes,flux,validity)) }
        process_error = ShackHartmannSummaryError;
    }
}
