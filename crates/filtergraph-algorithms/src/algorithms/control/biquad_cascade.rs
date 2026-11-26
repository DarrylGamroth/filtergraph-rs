//! CPU temporal cascades of strictly Schur-stable second-order sections.
//!
//! Scientific authority: FilterGraphAlgorithms `control/biquad_cascade.jl`.
//! A section stores `(b0,b1,b2,a1,a2)` with normalized `a0=1` and executes
//! transposed direct form II. Input, output, and the two DF2T states use the
//! same physical unit; coefficients are dimensionless.
use crate::{AlgorithmPlan, ProcessError};
use alloc::{boxed::Box, string::String, sync::Arc, vec, vec::Vec};
use core::{fmt, mem, ops::Range};
use num_traits::Float;

/// Invalid cascade coefficients, dimensions, or mode-region coverage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BiquadCascadePlanError {
    /// At least one independent element is required.
    EmptyExtent,
    /// Every cascade must contain at least one section.
    EmptySections,
    /// Prepared coefficient or state dimensions overflow addressable storage.
    ExtentOverflow,
    /// Coefficient set count must be one or the element extent.
    CoefficientSetCount,
    /// The coefficient buffer does not match its declared dimensions.
    CoefficientExtent,
    /// Every coefficient must be finite.
    NonFiniteCoefficient,
    /// A denominator fails strict Schur stability in the prepared precision.
    UnstableDenominator,
    /// Mode-region construction requires at least one region.
    EmptyRegions,
    /// A mode range is empty or outside the zero-based element extent.
    InvalidRegion,
    /// Two mode regions overlap.
    OverlappingRegions,
    /// Mode regions do not cover every independent element.
    UncoveredMode,
}
impl fmt::Display for BiquadCascadePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::EmptyExtent => "biquad extent must be positive",
            Self::EmptySections => "biquad section count must be positive",
            Self::ExtentOverflow => "biquad dimensions overflow addressable storage",
            Self::CoefficientSetCount => "biquad coefficient set count must be one or extent",
            Self::CoefficientExtent => "biquad coefficient extent is invalid",
            Self::NonFiniteCoefficient => "biquad coefficients must be finite",
            Self::UnstableDenominator => "biquad denominators must be strictly Schur stable",
            Self::EmptyRegions => "biquad mode regions must be nonempty",
            Self::InvalidRegion => "biquad mode ranges must be nonempty and inside the extent",
            Self::OverlappingRegions => "biquad mode regions must not overlap",
            Self::UncoveredMode => "biquad mode regions must cover every element",
        })
    }
}
impl core::error::Error for BiquadCascadePlanError {}

/// A rejected complete temporal sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BiquadCascadeError {
    /// Input or output has an incompatible extent.
    Process(ProcessError),
    /// Workspace dimensions do not match this cascade.
    WorkspaceExtent,
    /// The input contains a nonfinite coordinate.
    NonFiniteInput,
    /// A section output or either next DF2T state is nonfinite.
    NonFiniteOutput,
}
impl fmt::Display for BiquadCascadeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Process(e) => e.fmt(f),
            Self::WorkspaceExtent => f.write_str("biquad workspace dimensions are incompatible"),
            Self::NonFiniteInput => f.write_str("biquad input must be finite"),
            Self::NonFiniteOutput => f.write_str("biquad section output and state must be finite"),
        }
    }
}
impl core::error::Error for BiquadCascadeError {}
impl From<ProcessError> for BiquadCascadeError {
    fn from(e: ProcessError) -> Self {
        Self::Process(e)
    }
}

/// Owned coefficients for independent DF2T cascades.
///
/// Coefficients are tuple-contiguous: coefficient first, then section, then
/// coefficient set. This is the reference's column-major `(5,sections,sets)`
/// order. A single set broadcasts; otherwise each element selects its own set.
#[derive(Clone, Debug, PartialEq)]
pub struct BiquadCascadePlan<T> {
    coefficients: Arc<[T]>,
    // Section, coefficient, then independent mode, with mode contiguous.
    mode_coefficients: Arc<[T]>,
    extent: usize,
    section_count: usize,
    coefficient_set_count: usize,
}
impl<T: Float> BiquadCascadePlan<T> {
    /// Validate finite coefficients and strict `|a2|<1`, `|a1|<1+a2` in `T`.
    pub fn new(
        coefficients: impl Into<Arc<[T]>>,
        extent: usize,
        section_count: usize,
        coefficient_set_count: usize,
    ) -> Result<Self, BiquadCascadePlanError> {
        if extent == 0 {
            return Err(BiquadCascadePlanError::EmptyExtent);
        }
        if section_count == 0 {
            return Err(BiquadCascadePlanError::EmptySections);
        }
        if coefficient_set_count != 1 && coefficient_set_count != extent {
            return Err(BiquadCascadePlanError::CoefficientSetCount);
        }
        let count = 5_usize
            .checked_mul(section_count)
            .and_then(|v| v.checked_mul(coefficient_set_count))
            .ok_or(BiquadCascadePlanError::ExtentOverflow)?;
        let state_count = 2_usize
            .checked_mul(extent)
            .and_then(|v| v.checked_mul(section_count))
            .ok_or(BiquadCascadePlanError::ExtentOverflow)?;
        // Vec/Box storage cannot exceed isize::MAX bytes, even if usize arithmetic fits.
        if count
            .checked_mul(mem::size_of::<T>())
            .is_none_or(|v| v > isize::MAX as usize)
            || state_count
                .checked_mul(mem::size_of::<T>())
                .is_none_or(|v| v > isize::MAX as usize)
        {
            return Err(BiquadCascadePlanError::ExtentOverflow);
        }
        let coefficients = coefficients.into();
        if coefficients.len() != count {
            return Err(BiquadCascadePlanError::CoefficientExtent);
        }
        if coefficients.iter().any(|v| !v.is_finite()) {
            return Err(BiquadCascadePlanError::NonFiniteCoefficient);
        }
        for section in coefficients.as_chunks::<5>().0 {
            let a1 = section[3];
            let a2 = section[4];
            if !(a2.abs() < T::one() && a1.abs() < T::one() + a2) {
                return Err(BiquadCascadePlanError::UnstableDenominator);
            }
        }
        let mut mode_coefficients = vec![T::zero(); count];
        for section in 0..section_count {
            for coefficient in 0..5 {
                for element in 0..coefficient_set_count {
                    mode_coefficients
                        [(section * 5 + coefficient) * coefficient_set_count + element] =
                        coefficients[5 * (section + section_count * element) + coefficient];
                }
            }
        }
        Ok(Self {
            mode_coefficients: mode_coefficients.into(),
            coefficients,
            extent,
            section_count,
            coefficient_set_count,
        })
    }
    /// Construct unsorted, nonoverlapping, exhaustive zero-based mode regions.
    ///
    /// Each range is half-open and each matrix is a vector of coefficient
    /// tuples. Shorter cascades are padded with identity sections.
    pub fn from_regions(
        regions: &[(Range<usize>, Vec<[T; 5]>)],
        extent: usize,
    ) -> Result<Self, BiquadCascadePlanError> {
        if extent == 0 {
            return Err(BiquadCascadePlanError::EmptyExtent);
        }
        if regions.is_empty() {
            return Err(BiquadCascadePlanError::EmptyRegions);
        }
        if extent
            .checked_mul(2)
            .and_then(|v| v.checked_mul(mem::size_of::<T>()))
            .is_none_or(|v| v > isize::MAX as usize)
        {
            return Err(BiquadCascadePlanError::ExtentOverflow);
        }
        let mut covered = vec![false; extent];
        let mut section_count = 0;
        for (modes, coefficients) in regions {
            if modes.is_empty() || modes.end > extent {
                return Err(BiquadCascadePlanError::InvalidRegion);
            }
            if coefficients.is_empty() {
                return Err(BiquadCascadePlanError::EmptySections);
            }
            section_count = section_count.max(coefficients.len());
            for mode in modes.clone() {
                if covered[mode] {
                    return Err(BiquadCascadePlanError::OverlappingRegions);
                }
                covered[mode] = true;
            }
        }
        if covered.iter().any(|&v| !v) {
            return Err(BiquadCascadePlanError::UncoveredMode);
        }
        let count = 5_usize
            .checked_mul(section_count)
            .and_then(|v| v.checked_mul(extent))
            .ok_or(BiquadCascadePlanError::ExtentOverflow)?;
        if count
            .checked_mul(mem::size_of::<T>())
            .is_none_or(|v| v > isize::MAX as usize)
        {
            return Err(BiquadCascadePlanError::ExtentOverflow);
        }
        let mut bank = vec![T::zero(); count];
        for tuple in bank.as_chunks_mut::<5>().0 {
            tuple[0] = T::one();
        }
        for (modes, coefficients) in regions {
            for mode in modes.clone() {
                for (section, tuple) in coefficients.iter().enumerate() {
                    let start = 5 * (section + section_count * mode);
                    bank[start..start + 5].copy_from_slice(tuple);
                }
            }
        }
        Self::new(bank, extent, section_count, extent)
    }
    /// Borrow tuple-contiguous coefficients in coefficient, section, set order.
    pub fn coefficients(&self) -> &[T] {
        &self.coefficients
    }
    /// Return the independent input/output coordinate count.
    pub const fn extent(&self) -> usize {
        self.extent
    }
    /// Return the number of cascaded sections in each coefficient set.
    pub const fn section_count(&self) -> usize {
        self.section_count
    }
    /// Return one for broadcasting, otherwise the independent element extent.
    pub const fn coefficient_set_count(&self) -> usize {
        self.coefficient_set_count
    }
    /// Validate replacement tuples while retaining this plan's dimensions.
    ///
    /// A direct caller must reset its workspace when adopting the returned plan.
    /// The graph declaration performs this reset automatically.
    pub fn with_coefficients(
        &self,
        coefficients: impl Into<Arc<[T]>>,
    ) -> Result<Self, BiquadCascadePlanError> {
        Self::new(
            coefficients,
            self.extent,
            self.section_count,
            self.coefficient_set_count,
        )
    }
    /// Allocate fixed zero committed/candidate DF2T state and output staging.
    pub fn workspace(&self) -> BiquadCascadeWorkspace<T> {
        let count = 2 * self.extent * self.section_count;
        BiquadCascadeWorkspace {
            extent: self.extent,
            section_count: self.section_count,
            committed_state: vec![T::zero(); count].into_boxed_slice(),
            candidate_state: vec![T::zero(); count].into_boxed_slice(),
            candidate_output: vec![T::zero(); self.extent].into_boxed_slice(),
        }
    }
}
/// Fixed DF2T history owned by one cascade instance.
#[derive(Clone, Debug, PartialEq)]
pub struct BiquadCascadeWorkspace<T> {
    extent: usize,
    section_count: usize,
    committed_state: Box<[T]>,
    candidate_state: Box<[T]>,
    candidate_output: Box<[T]>,
}
impl<T: Float> BiquadCascadeWorkspace<T> {
    /// Borrow committed state with state fastest, then element, then section.
    pub fn state(&self) -> &[T] {
        &self.committed_state
    }
    /// Zero committed state and all private staging buffers.
    pub fn reset(&mut self) {
        self.committed_state.fill(T::zero());
        self.candidate_state.fill(T::zero());
        self.candidate_output.fill(T::zero());
    }
    /// Clear temporal history at a serialized protective-recovery boundary.
    pub fn recover(&mut self) {
        self.reset();
    }
    /// Adopt a successfully staged sample for a composite Algorithm owner.
    pub(crate) fn commit(&mut self) {
        mem::swap(&mut self.committed_state, &mut self.candidate_state);
    }
    /// Restore disposable staging after a composite Algorithm rejects a sample.
    pub(crate) fn discard(&mut self) {
        self.candidate_state.copy_from_slice(&self.committed_state);
        self.candidate_output.fill(T::zero());
    }
}
impl<T: Float> BiquadCascadePlan<T> {
    // Every iteration owns a separate mode. Tiling bounds the output working
    // set while the public state remains state, element, then section order.
    fn stage<const SHARED: bool>(&self, workspace: &mut BiquadCascadeWorkspace<T>) -> bool {
        let sets = self.coefficient_set_count;
        for first_element in (0..self.extent).step_by(4096) {
            let last_element = first_element + (self.extent - first_element).min(4096);
            for section in 0..self.section_count {
                let c = &self.mode_coefficients[5 * sets * section..5 * sets * (section + 1)];
                let b0 = &c[..sets];
                let b1 = &c[sets..2 * sets];
                let b2 = &c[2 * sets..3 * sets];
                let a1 = &c[3 * sets..4 * sets];
                let a2 = &c[4 * sets..];
                let state_offset = 2 * (self.extent * section + first_element);
                let state_end = 2 * (self.extent * section + last_element);
                let committed = &workspace.committed_state[state_offset..state_end];
                let candidate = &mut workspace.candidate_state[state_offset..state_end];
                let values = &mut workspace.candidate_output[first_element..last_element];
                let committed = committed.as_chunks::<2>().0;
                let candidate = candidate.as_chunks_mut::<2>().0;
                let valid = if SHARED {
                    let coefficients = [b0[0], b1[0], b2[0], a1[0], a2[0]];
                    let mut valid = true;
                    for ((value, state), next) in values.iter_mut().zip(committed).zip(candidate) {
                        valid &= Self::stage_value(coefficients, state, next, value);
                    }
                    valid
                } else {
                    let modes = first_element..last_element;
                    let coefficients = b0[modes.clone()]
                        .iter()
                        .zip(&b1[modes.clone()])
                        .zip(&b2[modes.clone()])
                        .zip(&a1[modes.clone()])
                        .zip(&a2[modes])
                        .map(|((((&b0, &b1), &b2), &a1), &a2)| [b0, b1, b2, a1, a2]);
                    Self::stage_section(coefficients, committed, candidate, values)
                };
                if !valid {
                    return false;
                }
            }
        }
        true
    }

    // Zipped, equally sized owned slices expose independent safe lanes without
    // coefficient bounds checks or an early-exit branch inside the reduction.
    fn stage_section(
        coefficients: impl Iterator<Item = [T; 5]>,
        committed: &[[T; 2]],
        candidate: &mut [[T; 2]],
        values: &mut [T],
    ) -> bool {
        let mut valid = true;
        for ((([b0, b1, b2, a1, a2], state), next), value) in
            coefficients.zip(committed).zip(candidate).zip(values)
        {
            valid &= Self::stage_value([b0, b1, b2, a1, a2], state, next, value);
        }
        valid
    }

    // Keep the scientific expression identical for shared and per-mode lanes.
    #[inline]
    fn stage_value(
        [b0, b1, b2, a1, a2]: [T; 5],
        state: &[T; 2],
        next: &mut [T; 2],
        value: &mut T,
    ) -> bool {
        let section_output = b0.mul_add(*value, state[0]);
        // JFG rounds b1*x and b2*x before feedback and adds s2 after
        // the first FMA. These groupings govern numerical rejection.
        let next_s1 = (-a1).mul_add(section_output, b1 * *value) + state[1];
        let next_s2 = (-a2).mul_add(section_output, b2 * *value);
        let valid = section_output.is_finite() & next_s1.is_finite() & next_s2.is_finite();
        *next = [next_s1, next_s2];
        *value = section_output;
        valid
    }

    /// Stage a complete sample for a composite Algorithm without adopting history.
    pub(crate) fn stage_process(
        &self,
        output: &mut [T],
        workspace: &mut BiquadCascadeWorkspace<T>,
        input: &[T],
    ) -> Result<(), BiquadCascadeError> {
        if input.len() != self.extent {
            return Err(ProcessError::InputExtent {
                expected: self.extent,
                actual: input.len(),
            }
            .into());
        }
        if output.len() != self.extent {
            return Err(ProcessError::OutputExtent {
                expected: self.extent,
                actual: output.len(),
            }
            .into());
        }
        if workspace.extent != self.extent || workspace.section_count != self.section_count {
            return Err(BiquadCascadeError::WorkspaceExtent);
        }
        if input.iter().any(|v| !v.is_finite()) {
            workspace.discard();
            return Err(BiquadCascadeError::NonFiniteInput);
        }
        workspace.candidate_output.copy_from_slice(input);
        let valid = if self.coefficient_set_count == 1 {
            self.stage::<true>(workspace)
        } else {
            self.stage::<false>(workspace)
        };
        if !valid {
            workspace.discard();
            return Err(BiquadCascadeError::NonFiniteOutput);
        }
        output.copy_from_slice(&workspace.candidate_output);
        Ok(())
    }
}
impl<T: Float + 'static> AlgorithmPlan for BiquadCascadePlan<T> {
    type Input<'a> = &'a [T];
    type Output<'a> = &'a mut [T];
    type Workspace = BiquadCascadeWorkspace<T>;
    type Error = BiquadCascadeError;
    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        self.stage_process(output, workspace, input)?;
        workspace.commit();
        Ok(())
    }
}

fn replace_coefficients(
    plan: &BiquadCascadePlan<f32>,
    values: &[f32],
) -> Result<BiquadCascadePlan<f32>, BiquadCascadePlanError> {
    if values.len() != plan.coefficients.len() {
        return Err(BiquadCascadePlanError::CoefficientExtent);
    }
    // Rust ndarray Port [coefficient,section,set] is row-major; internal tuples
    // and construction vectors retain the reference's coefficient-fastest order.
    let mut tuples = vec![0.0; values.len()];
    for set in 0..plan.coefficient_set_count {
        for section in 0..plan.section_count {
            for coefficient in 0..5 {
                tuples[coefficient + 5 * (section + plan.section_count * set)] =
                    values[(coefficient * plan.section_count + section)
                        * plan.coefficient_set_count
                        + set];
            }
        }
    }
    plan.with_coefficients(tuples)
}
crate::filtergraph_algorithm! {
    /// CPU Float32 DF2T cascade; valid coefficient replacement clears history.
    pub BiquadCascadeF32 {
        label = "biquad-cascade-f32";
        configuration BiquadCascadeF32Configuration {
            extent: usize = GraphRebuild;
            section_count: usize = GraphRebuild;
            per_mode: bool = GraphRebuild;
            initial_coefficients: Option<Vec<f32>> = Construction;
            input_schema: String = GraphRebuild;
            output_schema: String = GraphRebuild;
        }
        plan = BiquadCascadePlan<f32>;
        workspace = BiquadCascadeWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            extent = plan.extent();section_count = plan.section_count();coefficient_set_count = plan.coefficient_set_count();
        }
        inputs { input: f32 [extent] = "input", schema = configuration.input_schema.as_str(); }
        outputs { output: f32 [extent] = "output", schema = configuration.output_schema.as_str(), metadata_from = "input"; }
        parameters { coefficients: f32 [5,section_count,coefficient_set_count] = "coefficients", schema = crate::schemas::BIQUAD_COEFFICIENTS_V1,
            replace |plan, values| { replace_coefficients(plan, values) }; }
        parameter_error = BiquadCascadePlanError;
        prepare |configuration| {
            if configuration.extent == 0 { return Err(BiquadCascadePlanError::EmptyExtent); }
            if configuration.section_count == 0 { return Err(BiquadCascadePlanError::EmptySections); }
            let sets=if configuration.per_mode{configuration.extent}else{1};
            let count=5_usize.checked_mul(configuration.section_count).and_then(|v|v.checked_mul(sets)).ok_or(BiquadCascadePlanError::ExtentOverflow)?;
            let state_elements = 2_usize.checked_mul(configuration.extent).and_then(|v| v.checked_mul(configuration.section_count)).ok_or(BiquadCascadePlanError::ExtentOverflow)?;
            if configuration.extent > u32::MAX as usize / mem::size_of::<f32>() || count > u32::MAX as usize / mem::size_of::<f32>() || state_elements.checked_mul(mem::size_of::<f32>()).is_none_or(|v| v > isize::MAX as usize) {
                return Err(BiquadCascadePlanError::ExtentOverflow);
            }
            let coefficients=match configuration.initial_coefficients {
                Some(values)=>values,
                None=>{let mut values=vec![0.0;count];for section in values.as_chunks_mut::<5>().0{section[0]=1.0;}values}
            };
            let plan=BiquadCascadePlan::new(coefficients,configuration.extent,configuration.section_count,sets)?;
            let workspace=plan.workspace();Ok((plan,workspace))
        }
        prepare_error = BiquadCascadePlanError;
        process |plan, workspace| { plan.process(output,workspace,input) }
        process_error = BiquadCascadeError;
        reset |_plan, workspace| { workspace.reset(); }
        plan_adopted |_plan, workspace| { workspace.reset(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staged_history_requires_adoption_and_discard_restores_it() {
        let plan =
            BiquadCascadePlan::new(vec![0.5_f64, 0.25, 0.125, -0.5, 0.125], 1, 1, 1).unwrap();
        let mut workspace = plan.workspace();
        let mut output = [71.0];
        plan.stage_process(&mut output, &mut workspace, &[1.0])
            .unwrap();
        assert_eq!(output, [0.5]);
        assert_eq!(workspace.state(), [0.0, 0.0]);
        workspace.discard();
        assert_eq!(workspace.candidate_state.as_ref(), workspace.state());
        assert_eq!(workspace.candidate_output.as_ref(), [0.0]);
        plan.stage_process(&mut output, &mut workspace, &[0.0])
            .unwrap();
        assert_eq!(output, [0.0]);
        workspace.commit();
        assert_eq!(workspace.state(), [0.0, 0.0]);
        plan.stage_process(&mut output, &mut workspace, &[1.0])
            .unwrap();
        workspace.commit();
        assert_eq!(workspace.state(), [0.5, 0.0625]);
        let committed = workspace.state().to_vec();
        assert_eq!(
            plan.stage_process(&mut output, &mut workspace, &[f64::NAN]),
            Err(BiquadCascadeError::NonFiniteInput)
        );
        assert_eq!(workspace.state(), committed);
        assert_eq!(workspace.candidate_state.as_ref(), workspace.state());
    }
}
