//! Thompson four-image SCC calibration response.

use alloc::{boxed::Box, vec};
use core::{fmt, marker::PhantomData};

use num_traits::Float;

use crate::AlgorithmPlan;

/// Invalid prepared SCC push-pull response operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccPushPullResponsePlanError {
    /// A response must contain at least one image pixel.
    EmptyResponse,
}

impl fmt::Display for SccPushPullResponsePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyResponse => {
                formatter.write_str("SCC push-pull response pixel count must be nonzero")
            }
        }
    }
}

impl core::error::Error for SccPushPullResponsePlanError {}

/// Image role in one complete SCC push-pull calibration sample set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccPushPullImageKind {
    /// Positive probe acquired with the SCC reference fringes present.
    PushFringed,
    /// Positive probe acquired without the SCC reference fringes.
    PushUnfringed,
    /// Negative probe acquired with the SCC reference fringes present.
    PullFringed,
    /// Negative probe acquired without the SCC reference fringes.
    PullUnfringed,
}

/// Invalid SCC push-pull response input, output, workspace, or calculation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccPushPullResponseError {
    /// One calibration image has the wrong pixel extent.
    ImageExtent {
        /// Image with the wrong extent.
        image: SccPushPullImageKind,
        /// Required pixel extent.
        expected: usize,
        /// Supplied pixel extent.
        actual: usize,
    },
    /// The caller-owned response has the wrong pixel extent.
    ResponseExtent {
        /// Required pixel extent.
        expected: usize,
        /// Supplied pixel extent.
        actual: usize,
    },
    /// The supplied workspace was prepared for a different pixel extent.
    WorkspaceExtent,
    /// One calibration image contains a non-finite pixel.
    NonFiniteImage {
        /// Image containing the non-finite pixel.
        image: SccPushPullImageKind,
        /// Zero-based image-pixel index.
        pixel: usize,
    },
    /// One four-image double difference is not finite.
    NonFiniteCalculation {
        /// Zero-based response-pixel index.
        pixel: usize,
    },
}

impl fmt::Display for SccPushPullResponseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageExtent {
                image,
                expected,
                actual,
            } => write!(
                formatter,
                "SCC push-pull {image:?} image extent {actual} does not match {expected}"
            ),
            Self::ResponseExtent { expected, actual } => write!(
                formatter,
                "SCC push-pull response extent {actual} does not match {expected}"
            ),
            Self::WorkspaceExtent => {
                formatter.write_str("SCC push-pull response workspace extent is invalid")
            }
            Self::NonFiniteImage { image, pixel } => write!(
                formatter,
                "SCC push-pull {image:?} image pixel {pixel} must be finite"
            ),
            Self::NonFiniteCalculation { pixel } => write!(
                formatter,
                "SCC push-pull response calculation at pixel {pixel} is not finite"
            ),
        }
    }
}

impl core::error::Error for SccPushPullResponseError {}

/// Four caller-associated images for one SCC push-pull calibration response.
#[derive(Clone, Copy, Debug)]
pub struct SccPushPullResponseInput<'a, T> {
    push_fringed: &'a [T],
    push_unfringed: &'a [T],
    pull_fringed: &'a [T],
    pull_unfringed: &'a [T],
}

impl<'a, T> SccPushPullResponseInput<'a, T> {
    /// Groups a complete push/pull and fringed/unfringed image set.
    pub const fn new(
        push_fringed: &'a [T],
        push_unfringed: &'a [T],
        pull_fringed: &'a [T],
        pull_unfringed: &'a [T],
    ) -> Self {
        Self {
            push_fringed,
            push_unfringed,
            pull_fringed,
            pull_unfringed,
        }
    }

    /// Returns the positive-probe fringed image.
    pub const fn push_fringed(&self) -> &[T] {
        self.push_fringed
    }

    /// Returns the positive-probe unfringed image.
    pub const fn push_unfringed(&self) -> &[T] {
        self.push_unfringed
    }

    /// Returns the negative-probe fringed image.
    pub const fn pull_fringed(&self) -> &[T] {
        self.pull_fringed
    }

    /// Returns the negative-probe unfringed image.
    pub const fn pull_unfringed(&self) -> &[T] {
        self.pull_unfringed
    }
}

/// Prepared Thompson four-image SCC calibration response.
///
/// Each output pixel is the double difference:
///
/// ```text
/// (push_fringed - push_unfringed) - (pull_fringed - pull_unfringed)
/// ```
///
/// The four images must already be calibrated, admitted, and associated with
/// the named probe and SCC-reference states. This operation does not infer
/// acquisition order, divide by probe amplitude, or apply a spatial filter.
/// It publishes the complete response only after every staged pixel is
/// finite.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SccPushPullResponsePlan<T> {
    pixel_count: usize,
    scalar: PhantomData<fn(T)>,
}

impl<T> SccPushPullResponsePlan<T> {
    /// Prepares one nonempty SCC response extent.
    pub fn new(pixel_count: usize) -> Result<Self, SccPushPullResponsePlanError> {
        if pixel_count == 0 {
            return Err(SccPushPullResponsePlanError::EmptyResponse);
        }
        Ok(Self {
            pixel_count,
            scalar: PhantomData,
        })
    }

    /// Returns the prepared response-pixel extent.
    pub const fn pixel_count(&self) -> usize {
        self.pixel_count
    }

    /// Allocates failure-atomic staged storage for one execution owner.
    pub fn workspace(&self) -> SccPushPullResponseWorkspace<T>
    where
        T: Float,
    {
        SccPushPullResponseWorkspace {
            staged_response: vec![T::zero(); self.pixel_count].into_boxed_slice(),
        }
    }
}

impl<T> AlgorithmPlan for SccPushPullResponsePlan<T>
where
    T: Float + 'static,
{
    type Input<'a>
        = SccPushPullResponseInput<'a, T>
    where
        Self: 'a;
    type Output<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type Workspace = SccPushPullResponseWorkspace<T>;
    type Error = SccPushPullResponseError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        validate_image_extent(
            SccPushPullImageKind::PushFringed,
            self.pixel_count,
            input.push_fringed.len(),
        )?;
        validate_image_extent(
            SccPushPullImageKind::PushUnfringed,
            self.pixel_count,
            input.push_unfringed.len(),
        )?;
        validate_image_extent(
            SccPushPullImageKind::PullFringed,
            self.pixel_count,
            input.pull_fringed.len(),
        )?;
        validate_image_extent(
            SccPushPullImageKind::PullUnfringed,
            self.pixel_count,
            input.pull_unfringed.len(),
        )?;
        if output.len() != self.pixel_count {
            return Err(SccPushPullResponseError::ResponseExtent {
                expected: self.pixel_count,
                actual: output.len(),
            });
        }
        if workspace.staged_response.len() != self.pixel_count {
            return Err(SccPushPullResponseError::WorkspaceExtent);
        }

        validate_finite(SccPushPullImageKind::PushFringed, input.push_fringed)?;
        validate_finite(SccPushPullImageKind::PushUnfringed, input.push_unfringed)?;
        validate_finite(SccPushPullImageKind::PullFringed, input.pull_fringed)?;
        validate_finite(SccPushPullImageKind::PullUnfringed, input.pull_unfringed)?;

        for (pixel, ((((staged, push_fringed), push_unfringed), pull_fringed), pull_unfringed)) in
            workspace
                .staged_response
                .iter_mut()
                .zip(input.push_fringed)
                .zip(input.push_unfringed)
                .zip(input.pull_fringed)
                .zip(input.pull_unfringed)
                .enumerate()
        {
            let push_difference = *push_fringed - *push_unfringed;
            let pull_difference = *pull_fringed - *pull_unfringed;
            *staged = push_difference - pull_difference;
            if !push_difference.is_finite() || !pull_difference.is_finite() || !staged.is_finite() {
                return Err(SccPushPullResponseError::NonFiniteCalculation { pixel });
            }
        }

        output.copy_from_slice(&workspace.staged_response);
        Ok(())
    }
}

/// Fixed staged response storage for one SCC response execution owner.
#[derive(Clone, Debug, PartialEq)]
pub struct SccPushPullResponseWorkspace<T> {
    staged_response: Box<[T]>,
}

impl<T> SccPushPullResponseWorkspace<T> {
    /// Returns the response-pixel extent accepted by this workspace.
    pub const fn pixel_count(&self) -> usize {
        self.staged_response.len()
    }
}

fn validate_image_extent(
    image: SccPushPullImageKind,
    expected: usize,
    actual: usize,
) -> Result<(), SccPushPullResponseError> {
    if actual != expected {
        return Err(SccPushPullResponseError::ImageExtent {
            image,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_finite<T>(
    image: SccPushPullImageKind,
    values: &[T],
) -> Result<(), SccPushPullResponseError>
where
    T: Float,
{
    if let Some(pixel) = values.iter().position(|value| !value.is_finite()) {
        return Err(SccPushPullResponseError::NonFiniteImage { image, pixel });
    }
    Ok(())
}

crate::filtergraph_algorithm! {
    /// Portable Float32 Thompson four-image SCC response declaration.
    pub SccPushPullResponseF32 {
        label = "scc-push-pull-response-f32";
        configuration SccPushPullResponseF32Configuration {
            pixel_count: usize = GraphRebuild;
        }
        plan = SccPushPullResponsePlan<f32>;
        workspace = SccPushPullResponseWorkspace<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            pixel_count = plan.pixel_count();
        }
        inputs {
            push_fringed: f32 [pixel_count] = "push-fringed",
                schema = crate::schemas::CALIBRATED_PIXELS_V1;
            push_unfringed: f32 [pixel_count] = "push-unfringed",
                schema = crate::schemas::CALIBRATED_PIXELS_V1;
            pull_fringed: f32 [pixel_count] = "pull-fringed",
                schema = crate::schemas::CALIBRATED_PIXELS_V1;
            pull_unfringed: f32 [pixel_count] = "pull-unfringed",
                schema = crate::schemas::CALIBRATED_PIXELS_V1;
        }
        outputs {
            response: f32 [pixel_count] = "response",
                schema = crate::schemas::SCC_PUSH_PULL_RESPONSE_V1,
                metadata_from = "push-fringed";
        }
        prepare |configuration| {
            let plan = SccPushPullResponsePlan::new(configuration.pixel_count)?;
            let workspace = plan.workspace();
            Ok((plan, workspace))
        }
        prepare_error = SccPushPullResponsePlanError;
        process |plan, workspace| {
            plan.process(
                response,
                workspace,
                SccPushPullResponseInput::new(
                    push_fringed,
                    push_unfringed,
                    pull_fringed,
                    pull_unfringed,
                ),
            )
        }
        process_error = SccPushPullResponseError;
    }
}
