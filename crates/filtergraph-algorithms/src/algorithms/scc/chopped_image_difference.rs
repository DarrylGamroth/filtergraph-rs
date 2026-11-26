//! Rolling labeled SCC chopped-image differencing.

use alloc::{boxed::Box, vec};
use core::{fmt, marker::PhantomData};

use num_traits::Float;

use crate::{AlgorithmPlan, SampleAcquisitionIdentity};

/// Invalid prepared chopped-image difference configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChoppedImageDifferencePlanError {
    /// Chopped images must contain at least one pixel.
    EmptyImage,
}

impl fmt::Display for ChoppedImageDifferencePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage => formatter.write_str("chopped images must be nonempty"),
        }
    }
}

impl core::error::Error for ChoppedImageDifferencePlanError {}

/// Authoritative optical role of one chopped image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ChoppedImageKind {
    /// An explicit optical record reports no classification.
    Absent = 0,
    /// Image acquired with the SCC reference fringes present.
    Fringed = 2,
    /// Image acquired without the SCC reference fringes.
    Unfringed = 1,
    /// Exposure contains both optical phases and cannot form a pure pair.
    Mixed = 3,
}

/// Publication state of one rolling chopped-image result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ChoppedImageDifferenceStatus {
    /// No adjacent opposite-labeled pair exists; output is zero and invalid.
    #[default]
    Priming,
    /// An adjacent opposite-labeled pair produced the published difference.
    Complete,
}

/// Invalid chopped-image input, output, or workspace extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChoppedImageDifferenceError {
    /// The image sample has no metadata.
    MetadataMissing,
    /// Neither a pure optical role nor a pure electrical trigger is available.
    TriggerMissing,
    /// Physical acquisition sequence or generation moved backwards or repeated.
    SequenceInvalid,
    /// Adjacent acquisitions in one cohort have different exposure durations.
    ExposureMismatch,
    /// The encoded pair status requires one output element.
    PairStatusExtent {
        /// Supplied output extent.
        actual: usize,
    },
    /// The encoded chopped-image kind is neither fringed nor unfringed.
    InvalidKind {
        /// Supplied encoded kind.
        value: u8,
    },

    /// The input image has the wrong pixel extent.
    ImageExtent {
        /// Required pixel extent.
        expected: usize,
        /// Supplied pixel extent.
        actual: usize,
    },

    /// The input validity array has the wrong pixel extent.
    InputValidityExtent {
        /// Required pixel extent.
        expected: usize,
        /// Supplied validity extent.
        actual: usize,
    },

    /// The difference output has the wrong pixel extent.
    DifferenceExtent {
        /// Required pixel extent.
        expected: usize,
        /// Supplied pixel extent.
        actual: usize,
    },

    /// The output validity array has the wrong pixel extent.
    OutputValidityExtent {
        /// Required pixel extent.
        expected: usize,
        /// Supplied validity extent.
        actual: usize,
    },

    /// The supplied workspace was prepared for a different image extent.
    WorkspaceExtent,
}

impl fmt::Display for ChoppedImageDifferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetadataMissing => formatter.write_str("chopped image metadata is missing"),
            Self::TriggerMissing => {
                formatter.write_str("chopped image optical role or trigger is missing or mixed")
            }
            Self::SequenceInvalid => {
                formatter.write_str("chopped image acquisition sequence is invalid")
            }
            Self::ExposureMismatch => {
                formatter.write_str("chopped image exposure duration changed within one cohort")
            }
            Self::PairStatusExtent { actual } => write!(
                formatter,
                "chopped image pair status extent {actual} must be one"
            ),
            Self::InvalidKind { value } => {
                write!(formatter, "chopped-image kind {value} is invalid")
            }
            Self::ImageExtent { expected, actual } => write!(
                formatter,
                "chopped image extent {actual} does not match {expected}"
            ),
            Self::InputValidityExtent { expected, actual } => write!(
                formatter,
                "chopped-image input validity extent {actual} does not match {expected}"
            ),
            Self::DifferenceExtent { expected, actual } => write!(
                formatter,
                "chopped-image difference extent {actual} does not match {expected}"
            ),
            Self::OutputValidityExtent { expected, actual } => write!(
                formatter,
                "chopped-image output validity extent {actual} does not match {expected}"
            ),
            Self::WorkspaceExtent => {
                formatter.write_str("chopped-image difference workspace extent is invalid")
            }
        }
    }
}

impl core::error::Error for ChoppedImageDifferenceError {}

/// One caller-labeled chopped image and its per-pixel validity.
///
/// `validity` carries upstream saturation, bad-pixel, and other image-policy
/// decisions. Processing also treats every non-finite image value as invalid.
#[derive(Clone, Copy, Debug)]
pub struct ChoppedImageDifferenceInput<'a, T> {
    kind: ChoppedImageKind,
    image: &'a [T],
    validity: &'a [bool],
}

impl<'a, T> ChoppedImageDifferenceInput<'a, T> {
    /// Groups one image with its authoritative label and validity.
    pub const fn new(kind: ChoppedImageKind, image: &'a [T], validity: &'a [bool]) -> Self {
        Self {
            kind,
            image,
            validity,
        }
    }

    /// Returns the authoritative chopped-image role.
    pub const fn kind(&self) -> ChoppedImageKind {
        self.kind
    }

    /// Returns the chopped image.
    pub const fn image(&self) -> &[T] {
        self.image
    }

    /// Returns upstream per-pixel validity.
    pub const fn validity(&self) -> &[bool] {
        self.validity
    }
}

/// Caller-owned rolling difference, per-pixel validity, and coherent status.
///
/// Invalid pixels are zero. [`ChoppedImageDifferenceStatus::Complete`] means
/// an adjacent opposite-labeled pair was formed; it does not promise that
/// every pixel is valid.
#[derive(Debug)]
pub struct ChoppedImageDifferenceOutput<'a, T> {
    difference: &'a mut [T],
    validity: &'a mut [bool],
    status: &'a mut ChoppedImageDifferenceStatus,
}

impl<'a, T> ChoppedImageDifferenceOutput<'a, T> {
    /// Groups the coherent chopped-image output destinations.
    pub fn new(
        difference: &'a mut [T],
        validity: &'a mut [bool],
        status: &'a mut ChoppedImageDifferenceStatus,
    ) -> Self {
        Self {
            difference,
            validity,
            status,
        }
    }

    /// Returns the rolling `fringed - unfringed` image.
    pub fn difference(&self) -> &[T] {
        self.difference
    }

    /// Returns per-pixel difference validity.
    pub fn validity(&self) -> &[bool] {
        self.validity
    }

    /// Returns whether the output is priming or contains a complete pair.
    pub const fn status(&self) -> ChoppedImageDifferenceStatus {
        *self.status
    }
}

/// Prepared rolling adjacent chopped-image difference.
///
/// Calls must follow acquisition order and supply an authoritative
/// [`ChoppedImageKind`]. The workspace retains only the preceding image. An
/// opposite-labeled current image produces:
///
/// ```text
/// difference = fringed - unfringed
/// ```
///
/// When the current image is fringed, processing evaluates `current -
/// retained`; when it is unfringed, processing evaluates `retained - current`.
/// The result sign therefore remains fixed while every correctly alternating
/// input after priming produces a new rolling difference.
///
/// A repeated label emits priming output and replaces the retained image. The
/// owning adapter remains responsible for acquisition sequence checks and for
/// resetting the workspace on a wider discontinuity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChoppedImageDifferencePlan<T> {
    pixel_count: usize,
    trigger_high_kind: ChoppedImageKind,
    require_acquisition_identity: bool,
    pixel: PhantomData<fn(T)>,
}

impl<T> ChoppedImageDifferencePlan<T> {
    /// Prepares one nonempty chopped-image extent.
    pub fn new(pixel_count: usize) -> Result<Self, ChoppedImageDifferencePlanError> {
        if pixel_count == 0 {
            return Err(ChoppedImageDifferencePlanError::EmptyImage);
        }
        Ok(Self {
            pixel_count,
            trigger_high_kind: ChoppedImageKind::Fringed,
            require_acquisition_identity: false,
            pixel: PhantomData,
        })
    }

    /// Returns the prepared image-pixel extent.
    pub const fn pixel_count(&self) -> usize {
        self.pixel_count
    }

    /// Allocates one retained image and its normalized validity.
    pub fn workspace(&self) -> ChoppedImageDifferenceWorkspace<T>
    where
        T: Float,
    {
        ChoppedImageDifferenceWorkspace {
            retained_kind: None,
            retained_image: vec![T::zero(); self.pixel_count].into_boxed_slice(),
            retained_validity: vec![false; self.pixel_count].into_boxed_slice(),
            candidate_kind: None,
            candidate_image: vec![T::zero(); self.pixel_count].into_boxed_slice(),
            candidate_validity: vec![false; self.pixel_count].into_boxed_slice(),
            candidate_ready: false,
            committed_identity: None,
            candidate_identity: None,
            committed_duration: 0,
            candidate_duration: 0,
        }
    }

    fn validate(
        &self,
        output: &ChoppedImageDifferenceOutput<'_, T>,
        workspace: &ChoppedImageDifferenceWorkspace<T>,
        input: &ChoppedImageDifferenceInput<'_, T>,
    ) -> Result<(), ChoppedImageDifferenceError> {
        validate_extent(self.pixel_count, input.image.len(), |expected, actual| {
            ChoppedImageDifferenceError::ImageExtent { expected, actual }
        })?;
        validate_extent(
            self.pixel_count,
            input.validity.len(),
            |expected, actual| ChoppedImageDifferenceError::InputValidityExtent {
                expected,
                actual,
            },
        )?;
        validate_extent(
            self.pixel_count,
            output.difference.len(),
            |expected, actual| ChoppedImageDifferenceError::DifferenceExtent { expected, actual },
        )?;
        validate_extent(
            self.pixel_count,
            output.validity.len(),
            |expected, actual| ChoppedImageDifferenceError::OutputValidityExtent {
                expected,
                actual,
            },
        )?;
        if workspace.retained_image.len() != self.pixel_count
            || workspace.retained_validity.len() != self.pixel_count
            || workspace.candidate_image.len() != self.pixel_count
            || workspace.candidate_validity.len() != self.pixel_count
        {
            return Err(ChoppedImageDifferenceError::WorkspaceExtent);
        }
        Ok(())
    }
}

impl<T> AlgorithmPlan for ChoppedImageDifferencePlan<T>
where
    T: Float + 'static,
{
    type Input<'a>
        = ChoppedImageDifferenceInput<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = ChoppedImageDifferenceOutput<'a, T>
    where
        Self: 'a;
    type Workspace = ChoppedImageDifferenceWorkspace<T>;
    type Error = ChoppedImageDifferenceError;

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

impl<T: Float + 'static> ChoppedImageDifferencePlan<T> {
    /// Stage one pure exposure without adopting its retained image.
    pub(crate) fn stage_process(
        &self,
        output: ChoppedImageDifferenceOutput<'_, T>,
        workspace: &mut ChoppedImageDifferenceWorkspace<T>,
        input: ChoppedImageDifferenceInput<'_, T>,
    ) -> Result<(), ChoppedImageDifferenceError> {
        self.stage_process_adjacent(output, workspace, input, true)
    }

    fn stage_process_adjacent(
        &self,
        output: ChoppedImageDifferenceOutput<'_, T>,
        workspace: &mut ChoppedImageDifferenceWorkspace<T>,
        input: ChoppedImageDifferenceInput<'_, T>,
        adjacent: bool,
    ) -> Result<(), ChoppedImageDifferenceError> {
        self.validate(&output, workspace, &input)?;
        if matches!(
            input.kind,
            ChoppedImageKind::Absent | ChoppedImageKind::Mixed
        ) {
            return Err(ChoppedImageDifferenceError::InvalidKind {
                value: input.kind as u8,
            });
        }
        workspace.candidate_ready = false;

        let complete = adjacent
            && workspace
                .retained_kind
                .is_some_and(|retained_kind| retained_kind != input.kind);
        let processed = if complete {
            #[cfg(all(feature = "blas", target_arch = "x86_64", target_feature = "avx2"))]
            let processed = {
                use filtergraph_blas::f32_lanes::{LANES, exceptions_masked};
                // Keep scalar store order when an active subtraction can trap.
                if self.pixel_count >= LANES
                    && core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>()
                    && exceptions_masked()
                {
                    if input.kind == ChoppedImageKind::Fringed {
                        packed::stage_complete::<T, true>(
                            &mut *output.difference,
                            &mut *output.validity,
                            workspace,
                            &input,
                        )
                    } else {
                        packed::stage_complete::<T, false>(
                            &mut *output.difference,
                            &mut *output.validity,
                            workspace,
                            &input,
                        )
                    }
                } else {
                    0
                }
            };
            #[cfg(not(all(feature = "blas", target_arch = "x86_64", target_feature = "avx2")))]
            let processed = 0;
            let (fringed, unfringed) = match input.kind {
                ChoppedImageKind::Fringed => (input.image, workspace.retained_image.as_ref()),
                ChoppedImageKind::Unfringed => (workspace.retained_image.as_ref(), input.image),
                ChoppedImageKind::Absent | ChoppedImageKind::Mixed => {
                    unreachable!("non-pure images rejected before staging")
                }
            };
            for index in processed..self.pixel_count {
                let current_valid = input.validity[index] && input.image[index].is_finite();
                let valid = current_valid && workspace.retained_validity[index];
                if valid {
                    let difference = fringed[index] - unfringed[index];
                    if difference.is_finite() {
                        output.difference[index] = difference;
                        output.validity[index] = true;
                    } else {
                        output.difference[index] = T::zero();
                        output.validity[index] = false;
                    }
                } else {
                    output.difference[index] = T::zero();
                    output.validity[index] = false;
                }
            }
            processed
        } else {
            output.difference.fill(T::zero());
            output.validity.fill(false);
            0
        };

        *output.status = if complete {
            ChoppedImageDifferenceStatus::Complete
        } else {
            ChoppedImageDifferenceStatus::Priming
        };

        workspace.candidate_image[processed..].copy_from_slice(&input.image[processed..]);
        for index in processed..self.pixel_count {
            workspace.candidate_validity[index] =
                input.validity[index] && input.image[index].is_finite();
        }
        workspace.candidate_kind = Some(input.kind);
        workspace.candidate_identity = None;
        workspace.candidate_duration = 0;
        workspace.candidate_ready = true;
        Ok(())
    }
}

#[cfg(all(feature = "blas", target_arch = "x86_64", target_feature = "avx2"))]
mod packed {
    use super::{ChoppedImageDifferenceInput, ChoppedImageDifferenceWorkspace};
    use core::any::Any;
    use filtergraph_blas::f32_lanes::{LANES, subtract_chopped};

    pub(super) fn stage_complete<T: 'static, const CURRENT_FRINGED: bool>(
        difference: &mut [T],
        validity: &mut [bool],
        workspace: &mut ChoppedImageDifferenceWorkspace<T>,
        input: &ChoppedImageDifferenceInput<'_, T>,
    ) -> usize {
        let difference = difference.as_chunks_mut::<LANES>().0;
        let processed = difference.len() * LANES;
        let differences = difference
            .iter_mut()
            .zip(validity.as_chunks_mut::<LANES>().0);
        let candidates = workspace
            .candidate_image
            .as_chunks_mut::<LANES>()
            .0
            .iter_mut()
            .zip(workspace.candidate_validity.as_chunks_mut::<LANES>().0);
        let currents = input
            .image
            .as_chunks::<LANES>()
            .0
            .iter()
            .zip(input.validity.as_chunks::<LANES>().0);
        let retained = workspace
            .retained_image
            .as_chunks::<LANES>()
            .0
            .iter()
            .zip(workspace.retained_validity.as_chunks::<LANES>().0);
        for (
            (
                ((difference, validity), (candidate, candidate_validity)),
                (current, current_validity),
            ),
            (retained, retained_validity),
        ) in differences.zip(candidates).zip(currents).zip(retained)
        {
            // The plan selects this path only for Float32.
            let difference = (difference as &mut dyn Any).downcast_mut().unwrap();
            let candidate = (candidate as &mut dyn Any).downcast_mut().unwrap();
            let current = (current as &dyn Any).downcast_ref().unwrap();
            let retained = (retained as &dyn Any).downcast_ref().unwrap();
            subtract_chopped::<CURRENT_FRINGED>(
                (difference, validity),
                (candidate, candidate_validity),
                (current, current_validity),
                (retained, retained_validity),
            );
        }
        processed
    }
}

/// One retained chopped image and its normalized validity.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoppedImageDifferenceWorkspace<T> {
    retained_kind: Option<ChoppedImageKind>,
    retained_image: Box<[T]>,
    retained_validity: Box<[bool]>,
    candidate_kind: Option<ChoppedImageKind>,
    candidate_image: Box<[T]>,
    candidate_validity: Box<[bool]>,
    candidate_ready: bool,
    committed_identity: Option<SampleAcquisitionIdentity>,
    candidate_identity: Option<SampleAcquisitionIdentity>,
    committed_duration: u64,
    candidate_duration: u64,
}

impl<T> ChoppedImageDifferenceWorkspace<T>
where
    T: Float,
{
    /// Returns whether one preceding image is retained.
    pub const fn is_primed(&self) -> bool {
        self.retained_kind.is_some()
    }

    /// Returns the label of the retained preceding image.
    pub const fn retained_kind(&self) -> Option<ChoppedImageKind> {
        self.retained_kind
    }

    /// Returns the retained preceding image when primed.
    pub fn retained_image(&self) -> Option<&[T]> {
        self.retained_kind.map(|_| self.retained_image.as_ref())
    }

    /// Returns normalized validity for the retained image when primed.
    pub fn retained_validity(&self) -> Option<&[bool]> {
        self.retained_kind.map(|_| self.retained_validity.as_ref())
    }

    pub(crate) fn commit(&mut self) {
        if !self.candidate_ready {
            return;
        }
        core::mem::swap(&mut self.retained_image, &mut self.candidate_image);
        core::mem::swap(&mut self.retained_validity, &mut self.candidate_validity);
        self.retained_kind = self.candidate_kind;
        self.committed_identity = self.candidate_identity;
        self.committed_duration = self.candidate_duration;
        self.candidate_ready = false;
    }

    pub(crate) fn discard(&mut self) {
        self.candidate_ready = false;
    }

    /// Clears the retained image and its normalized validity.
    pub fn reset(&mut self) {
        self.retained_kind = None;
        self.committed_identity = None;
        self.candidate_identity = None;
        self.committed_duration = 0;
        self.candidate_duration = 0;
        self.retained_image.fill(T::zero());
        self.retained_validity.fill(false);
        self.candidate_kind = None;
        self.candidate_image.fill(T::zero());
        self.candidate_validity.fill(false);
        self.candidate_ready = false;
    }
}

fn validate_extent(
    expected: usize,
    actual: usize,
    error: impl FnOnce(usize, usize) -> ChoppedImageDifferenceError,
) -> Result<(), ChoppedImageDifferenceError> {
    if actual == expected {
        Ok(())
    } else {
        Err(error(expected, actual))
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 rolling SCC chopped-image difference declaration.
    pub ChoppedImageDifferenceF32 {
        label = "chopped-image-difference-f32";
        configuration ChoppedImageDifferenceF32Configuration {
            pixel_count: usize = GraphRebuild;
            trigger_high_is_fringed: bool = Construction;
            input_schema: alloc::string::String = GraphRebuild, default = alloc::string::String::from("pipewireao/calibrated-pixels-f32/1");
            require_acquisition_identity: bool = Construction, default = false;
        }
        plan = ChoppedImageDifferencePlan<f32>;
        workspace = ChoppedImageDifferenceWorkspace<f32>;
        properties = none;
        required_sample_metadata |plan, port| {
            if port == 0 && plan.require_acquisition_identity {
                &["acquisition_identity", "exposure_duration_nanoseconds", "chopped_image_kind"]
            } else {
                &[]
            }
        }
        dimensions |plan, _workspace| {
            pixel_count = plan.pixel_count();
        }
        inputs {
            image: f32 [pixel_count] = "image",
                schema = configuration.input_schema.as_str(),
                sample = image_sample, optional_sample_metadata = ["chopped_image_kind", "trigger_state", "acquisition_identity", "exposure_duration_nanoseconds"];
            input_validity: bool [pixel_count] = "input-validity",
                schema = crate::schemas::CALIBRATED_PIXEL_VALIDITY_V1;
        }
        outputs {
            difference: f32 [pixel_count] = "difference",
                schema = crate::schemas::SCC_CHOPPED_IMAGE_DIFFERENCE_V1,
                metadata_from = "image";
            output_validity: bool [pixel_count] = "output-validity",
                schema = crate::schemas::SCC_CHOPPED_DIFFERENCE_VALIDITY_V1,
                metadata_from = "image";
            pair_status: u8 [1] = "pair-status",
                schema = crate::schemas::SCC_CHOPPED_DIFFERENCE_STATUS_V1,
                metadata_from = "image";
        }
        prepare |configuration| {
            let mut plan = ChoppedImageDifferencePlan::new(configuration.pixel_count)?;
            plan.trigger_high_kind = if configuration.trigger_high_is_fringed {
                ChoppedImageKind::Fringed
            } else {
                ChoppedImageKind::Unfringed
            };
            plan.require_acquisition_identity = configuration.require_acquisition_identity;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = ChoppedImageDifferencePlanError;
        process |plan, workspace| {
            let sample = image_sample.ok_or(ChoppedImageDifferenceError::MetadataMissing)?;
            let kind = sample.chopped_image_kind();
            if kind == Some(ChoppedImageKind::Absent) {
                return Err(ChoppedImageDifferenceError::TriggerMissing);
            }
            let mut adjacent = true;
            let mut identity = None;
            let mut duration = 0;
            if plan.require_acquisition_identity {
                let current = sample.acquisition_identity()
                    .ok_or(ChoppedImageDifferenceError::MetadataMissing)?;
                duration = sample.exposure_duration_nanoseconds()
                    .ok_or(ChoppedImageDifferenceError::MetadataMissing)?;
                if kind.is_none() {
                    return Err(ChoppedImageDifferenceError::TriggerMissing);
                }
                let previous = workspace.committed_identity;
                let same_domain = previous.is_some_and(|previous| previous.domain() == current.domain());
                if same_domain && current.generation() < previous.unwrap().generation() {
                    return Err(ChoppedImageDifferenceError::SequenceInvalid);
                }
                let same_cohort = same_domain && previous.unwrap().generation() == current.generation();
                if same_cohort && current.sequence() <= previous.unwrap().sequence() {
                    return Err(ChoppedImageDifferenceError::SequenceInvalid);
                }
                if same_cohort && duration != workspace.committed_duration {
                    return Err(ChoppedImageDifferenceError::ExposureMismatch);
                }
                adjacent = same_cohort && current.sequence() - previous.unwrap().sequence() == 1
                    && !sample.is_discontinuous() && !sample.is_corrupted();
                identity = Some(current);
            }
            let kind = if plan.require_acquisition_identity &&
                (sample.is_corrupted() || kind == Some(ChoppedImageKind::Mixed)) {
                plan.trigger_high_kind
            } else {
                match kind {
                    Some(ChoppedImageKind::Absent | ChoppedImageKind::Mixed) =>
                        return Err(ChoppedImageDifferenceError::TriggerMissing),
                    Some(kind) => kind,
                    None => match sample.trigger_high() {
                        Some(true) => plan.trigger_high_kind,
                        Some(false) => match plan.trigger_high_kind {
                            ChoppedImageKind::Fringed => ChoppedImageKind::Unfringed,
                            ChoppedImageKind::Unfringed => ChoppedImageKind::Fringed,
                            ChoppedImageKind::Absent | ChoppedImageKind::Mixed => unreachable!("prepared polarity is pure"),
                        },
                        None => return Err(ChoppedImageDifferenceError::TriggerMissing),
                    },
                }
            };
            if pair_status.len() != 1 {
                return Err(ChoppedImageDifferenceError::PairStatusExtent { actual: pair_status.len() });
            }
            let mut status = ChoppedImageDifferenceStatus::Priming;
            plan.stage_process_adjacent(
                ChoppedImageDifferenceOutput::new(
                    difference,
                    output_validity,
                    &mut status,
                ),
                workspace,
                ChoppedImageDifferenceInput::new(kind, image, input_validity),
                adjacent,
            )?;
            if plan.require_acquisition_identity {
                workspace.candidate_identity = identity;
                workspace.candidate_duration = duration;
                if sample.chopped_image_kind() == Some(ChoppedImageKind::Mixed) || sample.is_corrupted() {
                    workspace.candidate_kind = None;
                    difference.fill(0.0);
                    output_validity.fill(false);
                    status = ChoppedImageDifferenceStatus::Priming;
                }
            }
            workspace.commit();
            pair_status[0] = match status {
                ChoppedImageDifferenceStatus::Priming => 0,
                ChoppedImageDifferenceStatus::Complete => 1,
            };
            Ok(())
        }
        process_error = ChoppedImageDifferenceError;
        reset |_plan, workspace| {
            workspace.reset();
        }
    }
}

#[cfg(test)]
mod transaction_tests {
    use super::*;
    #[test]
    fn staging_discard_mixed_rejection_and_public_adoption_preserve_pair_history() {
        let plan = ChoppedImageDifferencePlan::new(2).unwrap();
        let mut workspace = plan.workspace();
        let mut difference = [71.0_f32; 2];
        let mut validity = [true; 2];
        let mut status = ChoppedImageDifferenceStatus::Complete;
        plan.stage_process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
            &mut workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Fringed, &[3.0, 7.0], &[true; 2]),
        )
        .unwrap();
        assert!(!workspace.is_primed());
        workspace.commit();
        assert_eq!(workspace.retained_image(), Some([3.0, 7.0].as_slice()));
        plan.stage_process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
            &mut workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Unfringed, &[1.0, 2.0], &[true; 2]),
        )
        .unwrap();
        assert_eq!(difference, [2.0, 5.0]);
        workspace.discard();
        assert_eq!(workspace.retained_kind(), Some(ChoppedImageKind::Fringed));
        plan.process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
            &mut workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Fringed, &[4.0, 9.0], &[true; 2]),
        )
        .unwrap();
        assert_eq!(status, ChoppedImageDifferenceStatus::Priming);
        plan.process(
            ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
            &mut workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Unfringed, &[2.0, 3.0], &[true; 2]),
        )
        .unwrap();
        assert_eq!(difference, [2.0, 6.0]);
        let previous = workspace.clone();
        difference.fill(71.0);
        assert_eq!(
            plan.process(
                ChoppedImageDifferenceOutput::new(&mut difference, &mut validity, &mut status),
                &mut workspace,
                ChoppedImageDifferenceInput::new(ChoppedImageKind::Mixed, &[2.0, 3.0], &[true; 2])
            ),
            Err(ChoppedImageDifferenceError::InvalidKind { value: 3 })
        );
        assert_eq!(difference, [71.0; 2]);
        assert_eq!(workspace, previous);
        workspace.reset();
        assert!(!workspace.is_primed());
    }
}
