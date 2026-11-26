//! Exact rank-one ndarray copies with caller-owned output storage.

use crate::{AlgorithmPlan, NoWorkspace};
use alloc::string::String;
use core::fmt;

/// Invalid copy configuration or frame extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NdarrayCopyError {
    /// At least one element is required.
    Empty,
    /// Input length differs from the prepared extent.
    InputExtent,
    /// Output length differs from the prepared extent.
    OutputExtent,
}
impl fmt::Display for NdarrayCopyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "ndarray copy extent must be nonempty",
            Self::InputExtent => "ndarray copy input extent is invalid",
            Self::OutputExtent => "ndarray copy output extent is invalid",
        })
    }
}
impl core::error::Error for NdarrayCopyError {}

/// A prepared exact Float32 vector copy. No arithmetic or finite-value filtering occurs.
#[derive(Clone, Debug, PartialEq)]
pub struct NdarrayCopyPlan {
    extent: usize,
}
impl NdarrayCopyPlan {
    /// Prepare a nonempty vector copy.
    pub fn new(extent: usize) -> Result<Self, NdarrayCopyError> {
        if extent == 0 {
            return Err(NdarrayCopyError::Empty);
        }
        Ok(Self { extent })
    }
    /// Return the number of copied elements.
    pub fn extent(&self) -> usize {
        self.extent
    }
}
impl AlgorithmPlan for NdarrayCopyPlan {
    type Input<'a> = &'a [f32];
    type Output<'a> = &'a mut [f32];
    type Workspace = NoWorkspace;
    type Error = NdarrayCopyError;
    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _: &mut NoWorkspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if input.len() != self.extent {
            return Err(NdarrayCopyError::InputExtent);
        }
        if output.len() != self.extent {
            return Err(NdarrayCopyError::OutputExtent);
        }
        output.copy_from_slice(input);
        Ok(())
    }
}
crate::filtergraph_algorithm! {
    /// Exact Float32 rank-one copy preserving the configured scientific schema and input metadata.
    pub NdarrayCopyF32 {
        label = "ndarray-copy-f32";
        configuration NdarrayCopyF32Configuration {
            extent: usize = GraphRebuild;
            schema: String = GraphRebuild;
        }
        plan = NdarrayCopyPlan;
        properties = none;
        dimensions |plan, _workspace| { extent = plan.extent(); }
        inputs { input: f32 [extent] = "input", schema = configuration.schema.as_str(); }
        outputs { output: f32 [extent] = "output", schema = configuration.schema.as_str(), metadata_from = "input"; }
        prepare |configuration| { Ok((NdarrayCopyPlan::new(configuration.extent)?, NoWorkspace)) }
        prepare_error = NdarrayCopyError;
        process |plan, workspace| { plan.process(output, workspace, input) }
        process_error = NdarrayCopyError;
    }
}
