//! Reference-subtracted SCC dark-hole pixels in the prepared mask's linear order.
use crate::{AlgorithmPlan, NoWorkspace};
use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
use core::fmt;
use num_traits::Float;

/// Invalid SCC reference, selection mask, or required valid fraction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccPixelResidualPlanError {
    /// Reference image must contain at least one pixel.
    EmptyImage,
    /// The portable Float32 complete-image byte extent must fit UInt32.
    ImageExtent,
    /// Reference and mask must have equal complete-image extents.
    MaskExtent,
    /// Every prepared reference sample must be finite.
    NonFiniteReference,
    /// A dark-hole mask must select at least one pixel.
    EmptyMask,
    /// Required valid fraction must be finite and in (0,1].
    InvalidMinimumValidFraction,
    /// A replacement mask must retain the selected-pixel extent.
    SelectedCount,
    /// Configuration reference extent differs from its declared pixel count.
    ReferenceExtent,
}
impl fmt::Display for SccPixelResidualPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::EmptyImage => "SCC reference image must be nonempty",
            Self::ImageExtent => "SCC Float32 image byte extent must fit UInt32",
            Self::MaskExtent => "SCC mask and reference extents must match",
            Self::NonFiniteReference => "SCC reference image must be finite",
            Self::EmptyMask => "SCC mask must select pixels",
            Self::InvalidMinimumValidFraction => {
                "SCC minimum valid fraction must be finite and in (0,1]"
            }
            Self::SelectedCount => "replacement SCC mask changes selected-pixel count",
            Self::ReferenceExtent => "SCC reference extent differs from pixel count",
        })
    }
}
impl core::error::Error for SccPixelResidualPlanError {}
/// Complete-pair disposition or incompatible caller storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccPixelResidualError {
    /// Caller input/output shapes do not match the prepared complete/selected extents.
    DimensionMismatch,
    /// Chopped-image status is priming rather than a complete pair.
    IncompletePair,
    /// Chopped-image status is neither priming nor complete.
    InvalidPairStatus,
    /// Complete-pair outputs were calculated but too few selected pixels remained valid.
    InsufficientValidPixels,
}
impl fmt::Display for SccPixelResidualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DimensionMismatch => "SCC pixel-residual extents are incompatible",
            Self::IncompletePair => "SCC chopped-image pair is incomplete",
            Self::InvalidPairStatus => "SCC chopped-image pair status is invalid",
            Self::InsufficientValidPixels => "SCC selected valid-pixel count is insufficient",
        })
    }
}
impl core::error::Error for SccPixelResidualError {}

/// Immutable coherent SCC reference and ordered dark-hole selection profile.
#[derive(Clone, Debug)]
pub struct SccPixelResidualPlan<T> {
    reference_image: Arc<[T]>,
    mask: Arc<[bool]>,
    selected_indices: Box<[usize]>,
    minimum_valid_fraction: T,
    minimum_valid_count: usize,
}
impl<T: Float> SccPixelResidualPlan<T> {
    /// Prepare finite reference pixels, a nonempty selection, and fraction in (0,1].
    pub fn new(
        reference_image: impl Into<Arc<[T]>>,
        mask: impl Into<Arc<[bool]>>,
        minimum_valid_fraction: T,
    ) -> Result<Self, SccPixelResidualPlanError> {
        let reference_image = reference_image.into();
        let mask = mask.into();
        if reference_image.is_empty() {
            return Err(SccPixelResidualPlanError::EmptyImage);
        }
        if reference_image.len() != mask.len() {
            return Err(SccPixelResidualPlanError::MaskExtent);
        }
        if reference_image.iter().any(|x| !x.is_finite()) {
            return Err(SccPixelResidualPlanError::NonFiniteReference);
        }
        let selected_indices: Box<[usize]> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| selected.then_some(i))
            .collect();
        if selected_indices.is_empty() {
            return Err(SccPixelResidualPlanError::EmptyMask);
        }
        if !minimum_valid_fraction.is_finite()
            || minimum_valid_fraction <= T::zero()
            || minimum_valid_fraction > T::one()
        {
            return Err(SccPixelResidualPlanError::InvalidMinimumValidFraction);
        }
        // JFG deliberately computes this ceiling in Float64, including for f32 plans.
        let minimum_valid_count = (minimum_valid_fraction
            .to_f64()
            .expect("Float fraction converts to f64")
            * selected_indices.len() as f64)
            .ceil() as usize;
        Ok(Self {
            reference_image,
            mask,
            selected_indices,
            minimum_valid_fraction,
            minimum_valid_count,
        })
    }
    /// Complete chopped-image pixel count.
    pub fn pixel_count(&self) -> usize {
        self.reference_image.len()
    }
    /// Number of selected dark-hole pixels and reconstructor columns.
    pub fn selected_count(&self) -> usize {
        self.selected_indices.len()
    }
    /// Ordered zero-based selected complete-image coordinates.
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }
    /// Complete-image finite reference samples.
    pub fn reference_image(&self) -> &[T] {
        &self.reference_image
    }
    /// Complete-image dark-hole selection mask.
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }
    /// Required fraction of selected valid pixels.
    pub const fn minimum_valid_fraction(&self) -> T {
        self.minimum_valid_fraction
    }
    /// Smallest accepted selected valid-pixel count.
    pub const fn minimum_valid_count(&self) -> usize {
        self.minimum_valid_count
    }
    /// This operation retains no mathematical history or scratch.
    pub const fn workspace(&self) -> NoWorkspace {
        NoWorkspace
    }
    /// Prepare copied reference replacement in the same complete-image order.
    pub fn with_reference_image(&self, reference: &[T]) -> Result<Self, SccPixelResidualPlanError> {
        Self::new(
            Arc::from(reference),
            Arc::clone(&self.mask),
            self.minimum_valid_fraction,
        )
    }
    /// Prepare a copied mask retaining the same selected-pixel count.
    pub fn with_mask(&self, mask: &[bool]) -> Result<Self, SccPixelResidualPlanError> {
        let replacement = Self::new(
            Arc::clone(&self.reference_image),
            Arc::from(mask),
            self.minimum_valid_fraction,
        )?;
        if replacement.selected_count() != self.selected_count() {
            return Err(SccPixelResidualPlanError::SelectedCount);
        }
        Ok(replacement)
    }
}
impl<T: Float + 'static> AlgorithmPlan for SccPixelResidualPlan<T> {
    type Input<'a> = (&'a [T], &'a [bool], u8);
    type Output<'a> = (&'a mut [T], &'a mut [bool], &'a mut T);
    type Workspace = NoWorkspace;
    type Error = SccPixelResidualError;
    fn process<'input, 'output>(
        &self,
        (residuals, selected_validity, fraction): Self::Output<'output>,
        _workspace: &mut NoWorkspace,
        (difference, validity, pair_status): Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if difference.len() != self.pixel_count()
            || validity.len() != self.pixel_count()
            || residuals.len() != self.selected_count()
            || selected_validity.len() != self.selected_count()
        {
            return Err(SccPixelResidualError::DimensionMismatch);
        }
        match pair_status {
            0 => return Err(SccPixelResidualError::IncompletePair),
            1 => {}
            _ => return Err(SccPixelResidualError::InvalidPairStatus),
        }
        let (mut count, processed) =
            self.write_contiguous_prefix(residuals, selected_validity, difference, validity);
        for (offset, &pixel) in self.selected_indices[processed..].iter().enumerate() {
            let mut valid = validity[pixel];
            let mut value = T::zero();
            if valid {
                let sample = difference[pixel];
                valid = sample.is_finite();
                if valid {
                    value = sample - self.reference_image[pixel];
                    valid = value.is_finite();
                }
            }
            let selected = processed + offset;
            residuals[selected] = if valid { value } else { T::zero() };
            selected_validity[selected] = valid;
            count += usize::from(valid);
        }
        *fraction = T::from(count).expect("selected count converts to Float")
            / T::from(self.selected_count()).expect("selected count converts to Float");
        if count < self.minimum_valid_count {
            return Err(SccPixelResidualError::InsufficientValidPixels);
        }
        Ok(())
    }
}

impl<T: Float + 'static> SccPixelResidualPlan<T> {
    fn write_contiguous_prefix(
        &self,
        residuals: &mut [T],
        selected_validity: &mut [bool],
        difference: &[T],
        validity: &[bool],
    ) -> (usize, usize) {
        #[cfg(all(feature = "blas", target_arch = "x86_64", target_feature = "avx2"))]
        // Keep the scalar store order if a valid lane can trap before publication.
        if self.selected_count() == self.pixel_count()
            && core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>()
            && filtergraph_blas::f32_lanes::exceptions_masked()
        {
            return packed::write_all_selected(
                residuals,
                selected_validity,
                difference,
                validity,
                &self.reference_image,
            );
        }
        let _ = (residuals, selected_validity, difference, validity);
        (0, 0)
    }
}

#[cfg(all(feature = "blas", target_arch = "x86_64", target_feature = "avx2"))]
mod packed {
    use core::any::Any;
    use filtergraph_blas::f32_lanes::{LANES, subtract_finite};

    pub(super) fn write_all_selected<T: 'static>(
        residuals: &mut [T],
        selected_validity: &mut [bool],
        difference: &[T],
        validity: &[bool],
        reference_image: &[T],
    ) -> (usize, usize) {
        let (residuals, _) = residuals.as_chunks_mut::<LANES>();
        let mut count = 0;
        for ((((residuals, selected_validity), difference), validity), reference) in residuals
            .iter_mut()
            .zip(selected_validity.as_chunks_mut::<LANES>().0)
            .zip(difference.as_chunks::<LANES>().0)
            .zip(validity.as_chunks::<LANES>().0)
            .zip(reference_image.as_chunks::<LANES>().0)
        {
            // The caller selects this path only for Float32 plans.
            let residuals = (residuals as &mut dyn Any)
                .downcast_mut::<[f32; LANES]>()
                .unwrap();
            let difference = (difference as &dyn Any)
                .downcast_ref::<[f32; LANES]>()
                .unwrap();
            let reference = (reference as &dyn Any)
                .downcast_ref::<[f32; LANES]>()
                .unwrap();
            count += subtract_finite(
                residuals,
                selected_validity,
                difference,
                validity,
                reference,
            );
        }
        (count, residuals.len() * LANES)
    }
}

fn validate_f32_pixel_count(pixel_count: usize) -> Result<(), SccPixelResidualPlanError> {
    if pixel_count == 0 {
        return Err(SccPixelResidualPlanError::EmptyImage);
    }
    if pixel_count > u32::MAX as usize / core::mem::size_of::<f32>() {
        return Err(SccPixelResidualPlanError::ImageExtent);
    }
    Ok(())
}

crate::filtergraph_algorithm! {
    /// Ordered SCC reference subtraction, selected validity, and complete-pair acceptance.
    pub SccPixelResidualF32 {
        label="scc-pixel-residual-f32";
        configuration SccPixelResidualF32Configuration {
            pixel_count:usize=GraphRebuild;mask:Vec<bool>=GraphRebuild;
            initial_reference_image:Option<Vec<f32>>=Construction;minimum_valid_fraction:f32=Construction;
        }
        plan=SccPixelResidualPlan<f32>;workspace=NoWorkspace;properties=none;
        dimensions |plan,_workspace| {pixel_count=plan.pixel_count();selected_count=plan.selected_count();}
        inputs {
            difference:f32 [pixel_count]="difference",schema=crate::schemas::SCC_CHOPPED_IMAGE_DIFFERENCE_V1, optional_sample_metadata = ["chopped_image_kind", "trigger_state", "acquisition_identity", "exposure_duration_nanoseconds"];
            difference_validity:bool [pixel_count]="difference-validity",schema=crate::schemas::SCC_CHOPPED_DIFFERENCE_VALIDITY_V1;
            pair_status:u8 [1]="pair-status",schema=crate::schemas::SCC_CHOPPED_DIFFERENCE_STATUS_V1;
        }
        outputs {
            selected_residuals:f32 [selected_count]="selected-residuals",schema=crate::schemas::SCC_SELECTED_PIXEL_RESIDUALS_V1,metadata_from="difference";
            selected_validity:bool [selected_count]="selected-validity",schema=crate::schemas::SCC_SELECTED_PIXEL_VALIDITY_V1,metadata_from="difference";
            valid_fraction:f32 [1]="valid-fraction",schema=crate::schemas::SCC_SELECTED_PIXEL_VALID_FRACTION_V1,metadata_from="difference";
        }
        parameters {
            reference_image:f32 [pixel_count]="reference-image",schema=crate::schemas::SCC_REFERENCE_IMAGE_V1,replace |plan,values| {plan.with_reference_image(values)};
            mask:bool [pixel_count]="mask",schema=crate::schemas::PIXEL_SELECTION_MASK_V1,replace |plan,values| {plan.with_mask(values)};
        }
        parameter_error=SccPixelResidualPlanError;
        prepare |configuration| {
            validate_f32_pixel_count(configuration.pixel_count)?;
            if configuration.mask.len()!=configuration.pixel_count{return Err(SccPixelResidualPlanError::MaskExtent);}
            let reference=configuration.initial_reference_image.unwrap_or_else(||vec![0.0;configuration.pixel_count]);
            if reference.len()!=configuration.pixel_count{return Err(SccPixelResidualPlanError::ReferenceExtent);}
            Ok((SccPixelResidualPlan::new(reference,configuration.mask,configuration.minimum_valid_fraction)?,NoWorkspace))
        }
        prepare_error=SccPixelResidualPlanError;
        process |plan,workspace| {
            if pair_status.len()!=1||valid_fraction.len()!=1{return Err(SccPixelResidualError::DimensionMismatch);}
            plan.process((selected_residuals,selected_validity,&mut valid_fraction[0]),workspace,(difference,difference_validity,pair_status[0]))
        }
        process_error=SccPixelResidualError;
    }
}

#[cfg(test)]
mod tests {
    use super::{SccPixelResidualPlanError, validate_f32_pixel_count};

    #[test]
    fn portable_f32_image_extent_is_checked_without_allocating_planes() {
        let maximum = u32::MAX as usize / core::mem::size_of::<f32>();
        assert_eq!(
            validate_f32_pixel_count(0),
            Err(SccPixelResidualPlanError::EmptyImage)
        );
        assert_eq!(validate_f32_pixel_count(maximum), Ok(()));
        assert_eq!(
            validate_f32_pixel_count(maximum + 1),
            Err(SccPixelResidualPlanError::ImageExtent)
        );
    }
}
