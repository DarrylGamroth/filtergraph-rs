//! Shape-preserving physical-DM inter-actuator stroke limiting.

use alloc::{sync::Arc, vec::Vec};
use core::fmt;

use num_traits::Float;

use crate::{AlgorithmPlan, NoWorkspace};

/// One undirected physical-actuator stroke-difference constraint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterActuatorStrokeLimit<T> {
    first: usize,
    second: usize,
    max_difference: T,
}

impl<T> InterActuatorStrokeLimit<T> {
    /// Describes one physical adjacency edge and its maximum absolute command
    /// difference. [`InterActuatorStrokeLimitPlan::new`] validates the values.
    pub const fn new(first: usize, second: usize, max_difference: T) -> Self {
        Self {
            first,
            second,
            max_difference,
        }
    }

    /// Returns the first physical-actuator index.
    pub const fn first(&self) -> usize {
        self.first
    }

    /// Returns the second physical-actuator index.
    pub const fn second(&self) -> usize {
        self.second
    }
}

impl<T> InterActuatorStrokeLimit<T>
where
    T: Copy,
{
    /// Returns the maximum absolute command difference for this edge.
    pub const fn max_difference(&self) -> T {
        self.max_difference
    }
}

/// Invalid prepared inter-actuator stroke-limit graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterActuatorStrokeLimitPlanError {
    /// At least one physical actuator is required.
    EmptyActuatorSet,
    /// A selected graph must contain at least one edge.
    EmptyGraph,
    /// An edge endpoint is outside the physical-actuator extent.
    ActuatorIndex,
    /// An edge connects an actuator to itself.
    SelfEdge,
    /// The graph contains the same undirected edge more than once.
    DuplicateEdge,
    /// An edge limit is non-finite or negative.
    InvalidLimit,
    /// The edge and maximum-difference arrays have different extents.
    LimitExtent {
        /// Number of prepared adjacency edges.
        expected: usize,
        /// Number of supplied maximum differences.
        actual: usize,
    },
}

impl fmt::Display for InterActuatorStrokeLimitPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyActuatorSet => {
                formatter.write_str("inter-actuator stroke extent must be nonempty")
            }
            Self::EmptyGraph => {
                formatter.write_str("inter-actuator stroke graph must contain an edge")
            }
            Self::ActuatorIndex => {
                formatter.write_str("inter-actuator stroke edge index is out of range")
            }
            Self::SelfEdge => {
                formatter.write_str("inter-actuator stroke graph contains a self-edge")
            }
            Self::DuplicateEdge => {
                formatter.write_str("inter-actuator stroke graph contains a duplicate edge")
            }
            Self::InvalidLimit => {
                formatter.write_str("inter-actuator stroke limits must be finite and nonnegative")
            }
            Self::LimitExtent { expected, actual } => write!(
                formatter,
                "inter-actuator limit extent {actual} does not match {expected} edges"
            ),
        }
    }
}

impl core::error::Error for InterActuatorStrokeLimitPlanError {}

/// Invalid inter-actuator stroke-limit input or output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterActuatorStrokeLimitError {
    /// The input vector has the wrong physical-actuator extent.
    InputExtent,
    /// The output vector has the wrong physical-actuator extent.
    OutputExtent,
    /// The input vector contains a non-finite value.
    NonFiniteInput,
}

impl fmt::Display for InterActuatorStrokeLimitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputExtent => {
                formatter.write_str("inter-actuator stroke input extent is invalid")
            }
            Self::OutputExtent => {
                formatter.write_str("inter-actuator stroke output extent is invalid")
            }
            Self::NonFiniteInput => {
                formatter.write_str("inter-actuator stroke input must be finite")
            }
        }
    }
}

impl core::error::Error for InterActuatorStrokeLimitError {}

/// Prepared radial projection onto physical inter-actuator stroke limits.
#[derive(Clone, Debug, PartialEq)]
pub struct InterActuatorStrokeLimitPlan<T> {
    pdm_size: usize,
    limits: Arc<[InterActuatorStrokeLimit<T>]>,
}

impl<T> InterActuatorStrokeLimitPlan<T>
where
    T: Float,
{
    /// Validates separate physical adjacency edges and maximum differences.
    pub fn from_edges_and_limits(
        pdm_size: usize,
        edges: &[(usize, usize)],
        max_differences: &[T],
    ) -> Result<Self, InterActuatorStrokeLimitPlanError> {
        if edges.len() != max_differences.len() {
            return Err(InterActuatorStrokeLimitPlanError::LimitExtent {
                expected: edges.len(),
                actual: max_differences.len(),
            });
        }
        let limits = edges
            .iter()
            .zip(max_differences)
            .map(|(&(first, second), &max_difference)| {
                InterActuatorStrokeLimit::new(first, second, max_difference)
            })
            .collect::<Vec<_>>();
        Self::new(pdm_size, Arc::from(limits))
    }

    /// Validates and retains one physical adjacency graph.
    pub fn new(
        pdm_size: usize,
        limits: Arc<[InterActuatorStrokeLimit<T>]>,
    ) -> Result<Self, InterActuatorStrokeLimitPlanError> {
        if pdm_size == 0 {
            return Err(InterActuatorStrokeLimitPlanError::EmptyActuatorSet);
        }
        if limits.is_empty() {
            return Err(InterActuatorStrokeLimitPlanError::EmptyGraph);
        }
        for (index, limit) in limits.iter().enumerate() {
            if limit.first >= pdm_size || limit.second >= pdm_size {
                return Err(InterActuatorStrokeLimitPlanError::ActuatorIndex);
            }
            if limit.first == limit.second {
                return Err(InterActuatorStrokeLimitPlanError::SelfEdge);
            }
            if !limit.max_difference.is_finite() || limit.max_difference < T::zero() {
                return Err(InterActuatorStrokeLimitPlanError::InvalidLimit);
            }
            let first = limit.first.min(limit.second);
            let second = limit.first.max(limit.second);
            if limits[..index].iter().any(|existing| {
                existing.first.min(existing.second) == first
                    && existing.first.max(existing.second) == second
            }) {
                return Err(InterActuatorStrokeLimitPlanError::DuplicateEdge);
            }
        }
        Ok(Self { pdm_size, limits })
    }

    /// Returns the physical-actuator extent.
    pub fn len(&self) -> usize {
        self.pdm_size
    }

    /// Returns whether the physical-actuator extent is zero.
    pub fn is_empty(&self) -> bool {
        self.pdm_size == 0
    }

    /// Returns the prepared physical adjacency graph.
    pub fn limits(&self) -> &[InterActuatorStrokeLimit<T>] {
        &self.limits
    }

    /// Replaces all maximum differences while preserving the prepared
    /// physical adjacency topology.
    pub fn with_max_differences(
        &self,
        max_differences: &[T],
    ) -> Result<Self, InterActuatorStrokeLimitPlanError> {
        if self.limits.len() != max_differences.len() {
            return Err(InterActuatorStrokeLimitPlanError::LimitExtent {
                expected: self.limits.len(),
                actual: max_differences.len(),
            });
        }
        let limits = self
            .limits
            .iter()
            .zip(max_differences)
            .map(|(edge, &max_difference)| {
                InterActuatorStrokeLimit::new(edge.first, edge.second, max_difference)
            })
            .collect::<Vec<_>>();
        Self::new(self.pdm_size, Arc::from(limits))
    }
}

fn difference_gain<T>(first: T, second: T, limit: T) -> T
where
    T: Float,
{
    if first.is_sign_negative() == second.is_sign_negative() {
        let difference = (first - second).abs();
        if difference <= limit {
            T::one()
        } else {
            limit / difference
        }
    } else {
        let first = first.abs();
        let second = second.abs();
        let scale = first.max(second);
        if scale == T::zero() {
            return T::one();
        }
        let normalized_difference = T::one() + first.min(second) / scale;
        if scale <= limit / normalized_difference {
            T::one()
        } else {
            (limit / scale) / normalized_difference
        }
    }
}

impl<T> AlgorithmPlan for InterActuatorStrokeLimitPlan<T>
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
    type Error = InterActuatorStrokeLimitError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if input.len() != self.pdm_size {
            return Err(InterActuatorStrokeLimitError::InputExtent);
        }
        if output.len() != self.pdm_size {
            return Err(InterActuatorStrokeLimitError::OutputExtent);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(InterActuatorStrokeLimitError::NonFiniteInput);
        }
        let mut gain = T::one();
        for limit in self.limits.iter() {
            gain = gain.min(difference_gain(
                input[limit.first],
                input[limit.second],
                limit.max_difference,
            ));
        }
        for (destination, source) in output.iter_mut().zip(input) {
            *destination = *source * gain;
        }
        Ok(())
    }
}

crate::filtergraph_algorithm! {
    /// Portable Float32 physical inter-actuator stroke-limit declaration.
    pub InterActuatorStrokeLimitF32 {
        label = "inter-actuator-stroke-limit-f32";
        configuration InterActuatorStrokeLimitF32Configuration {
            actuator_count: usize = GraphRebuild;
            edges: Vec<[usize; 2]> = GraphRebuild;
            max_differences: Vec<f32> = Construction;
        }
        plan = InterActuatorStrokeLimitPlan<f32>;
        properties = none;
        dimensions |plan, _workspace| {
            actuator_count = plan.len();
            edge_count = plan.limits().len();
        }
        inputs {
            requested: f32 [actuator_count] = "requested",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1;
        }
        outputs {
            stroke_limited: f32 [actuator_count] = "stroke-limited",
                schema = crate::schemas::REQUESTED_PDM_COMMAND_V1,
                metadata_from = "requested";
        }
        parameters {
            max_differences: f32 [edge_count] = "max-differences",
                schema = crate::schemas::INTER_ACTUATOR_STROKE_LIMIT_V1,
                replace |plan, values| {
                    plan.with_max_differences(values)
                };
        }
        parameter_error = InterActuatorStrokeLimitPlanError;
        prepare |configuration| {
            let edges = configuration
                .edges
                .iter()
                .map(|edge| (edge[0], edge[1]))
                .collect::<Vec<_>>();
            Ok((
                InterActuatorStrokeLimitPlan::from_edges_and_limits(
                    configuration.actuator_count,
                    &edges,
                    &configuration.max_differences,
                )?,
                NoWorkspace,
            ))
        }
        prepare_error = InterActuatorStrokeLimitPlanError;
        process |plan, workspace| {
            plan.process(stroke_limited, workspace, requested)
        }
        process_error = InterActuatorStrokeLimitError;
    }
}
