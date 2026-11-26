//! Fixed LLOWFS model, using the pinned JFG SimpleChains parameter order.
use crate::AlgorithmPlan;
use alloc::{string::String, vec, vec::Vec};
use core::fmt;
use num_traits::Float;

/// Number of parameters in `legacy-simplechains-llowfs-v1`.
pub const ML_LLOWFS_PARAMETER_COUNT: usize = 240_584;
/// Fixed scientific model identifier.
pub const ML_LLOWFS_MODEL: &str = "legacy-simplechains-llowfs-v1";

/// Invalid model preparation, storage, input, or inference result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlLlowfsError {
    /// The model identifier is unsupported.
    Model,
    /// Parameter count or finite-value validation failed.
    Parameters,
    /// Caller storage has the wrong extent.
    Extent,
    /// Model parameters have not been adopted.
    ParametersMissing,
    /// An input pixel is nonfinite.
    NonFiniteInput,
    /// Total flux is nonpositive or nonfinite.
    InvalidFlux,
    /// Normalized sample deviation is nonpositive or nonfinite.
    InvalidStandardDeviation,
    /// At least one inferred mode is nonfinite.
    NonFiniteOutput,
}
impl fmt::Display for MlLlowfsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ML LLOWFS: {self:?}")
    }
}
impl core::error::Error for MlLlowfsError {}

/// Immutable finite model parameters; parameter adoption makes the model ready.
#[derive(Clone, Debug)]
pub struct MlLlowfsPlan {
    parameters: Vec<f32>,
    ready: bool,
}
/// Fixed inference scratch, without carried mathematical history.
#[derive(Clone, Debug)]
pub struct MlLlowfsWorkspace {
    input: Vec<f32>,
    left: Vec<f32>,
    right: Vec<f32>,
}
impl MlLlowfsWorkspace {
    /// Clear disposable inference scratch, retaining adopted parameters.
    pub fn reset(&mut self) {
        self.input.fill(0.0);
        self.left.fill(0.0);
        self.right.fill(0.0);
    }
}
impl MlLlowfsPlan {
    /// Construct the unconfigured, not-ready scientific model.
    pub fn new(model: &str) -> Result<Self, MlLlowfsError> {
        if model != ML_LLOWFS_MODEL {
            return Err(MlLlowfsError::Model);
        }
        Ok(Self {
            parameters: vec![0.0; ML_LLOWFS_PARAMETER_COUNT],
            ready: false,
        })
    }
    /// Copy a complete finite parameter replacement in SimpleChains flat order.
    pub fn with_parameters(&self, values: &[f32]) -> Result<Self, MlLlowfsError> {
        if values.len() != ML_LLOWFS_PARAMETER_COUNT || values.iter().any(|v| !v.is_finite()) {
            return Err(MlLlowfsError::Parameters);
        }
        Ok(Self {
            parameters: values.to_vec(),
            ready: true,
        })
    }
    /// Allocate the largest intermediate tensor once, during preparation.
    pub fn workspace(&self) -> MlLlowfsWorkspace {
        MlLlowfsWorkspace {
            input: vec![0.0; 400],
            left: vec![0.0; 2890],
            right: vec![0.0; 2890],
        }
    }
}

// Tensor and convolution-kernel axes follow Julia column-major order:
// (row,column,channel) and (kernel_row,kernel_column,input_channel,output_channel).
fn convolution<
    const OUT_SIDE: usize,
    const CHANNELS: usize,
    const KERNEL: usize,
    const OUTPUTS: usize,
>(
    input: &[f32],
    output: &mut [f32],
    parameters: &[f32],
) -> usize {
    let side = OUT_SIDE + KERNEL - 1;
    if !cfg!(all(feature = "blas", target_feature = "fma")) {
        return convolution_scalar(input, output, side, CHANNELS, KERNEL, OUTPUTS, parameters);
    }

    let weights = KERNEL * KERNEL * CHANNELS * OUTPUTS;
    for o in 0..OUTPUTS {
        for col in 0..OUT_SIDE {
            // Each lane accumulates one row in the original kr, kc, i order.
            let mut sums = [0.0f32; OUT_SIDE];
            for kr in 0..KERNEL {
                for kc in 0..KERNEL {
                    for i in 0..CHANNELS {
                        let start = kr + side * (col + kc + side * i);
                        let w = parameters[kr + KERNEL * (kc + KERNEL * (i + CHANNELS * o))];
                        for (sum, &x) in sums.iter_mut().zip(&input[start..start + OUT_SIDE]) {
                            *sum = x.mul_add(w, *sum);
                        }
                    }
                }
            }
            let start = OUT_SIDE * (col + OUT_SIDE * o);
            for (value, sum) in output[start..start + OUT_SIDE].iter_mut().zip(sums) {
                *value = Float::tanh(sum + parameters[weights + o]);
            }
        }
    }
    weights + OUTPUTS
}

fn convolution_scalar(
    input: &[f32],
    output: &mut [f32],
    side: usize,
    channels: usize,
    kernel: usize,
    outputs: usize,
    parameters: &[f32],
) -> usize {
    let out_side = side - kernel + 1;
    let weights = kernel * kernel * channels * outputs;
    for o in 0..outputs {
        for col in 0..out_side {
            for row in 0..out_side {
                let mut sum = 0.0;
                for kr in 0..kernel {
                    for kc in 0..kernel {
                        for i in 0..channels {
                            let x = input[row + kr + side * (col + kc + side * i)];
                            let w = parameters[kr + kernel * (kc + kernel * (i + channels * o))];
                            sum = x.mul_add(w, sum);
                        }
                    }
                }
                output[row + out_side * (col + out_side * o)] =
                    Float::tanh(sum + parameters[weights + o]);
            }
        }
    }
    weights + outputs
}
fn dense(input: &[f32], output: &mut [f32], parameters: &[f32], relu: bool) -> usize {
    if !cfg!(all(feature = "blas", target_feature = "fma")) {
        return dense_scalar(input, output, parameters, relu);
    }

    let extent = output.len();
    let weights = input.len() * extent;
    let (coefficients, biases) = parameters.split_at(weights);
    // Keep independent outputs in registers across successive input weights.
    const LANES: usize = 32;
    let (blocks, tail) = output.as_chunks_mut::<LANES>();
    for (block, output) in blocks.iter_mut().enumerate() {
        let start = block * LANES;
        let sums = dense_sums::<LANES>(input, coefficients, extent, start);
        activate(output, &sums, &biases[start..start + LANES], relu);
    }
    let start = blocks.len() * LANES;
    let (blocks, tail) = tail.as_chunks_mut::<8>();
    for (block, output) in blocks.iter_mut().enumerate() {
        let start = start + block * 8;
        let sums = dense_sums::<8>(input, coefficients, extent, start);
        activate(output, &sums, &biases[start..start + 8], relu);
    }
    for (neuron, value) in (extent - tail.len()..).zip(tail) {
        let sum = dense_sums::<1>(input, coefficients, extent, neuron)[0] + biases[neuron];
        *value = if relu { sum.max(0.0) } else { sum };
    }
    weights + extent
}

#[inline]
fn dense_sums<const LANES: usize>(
    input: &[f32],
    coefficients: &[f32],
    extent: usize,
    start: usize,
) -> [f32; LANES] {
    let mut sums = [0.0; LANES];
    for (&x, weights) in input.iter().zip(coefficients.chunks_exact(extent)) {
        for (sum, &w) in sums.iter_mut().zip(&weights[start..start + LANES]) {
            *sum = x.mul_add(w, *sum);
        }
    }
    sums
}

fn activate(output: &mut [f32], sums: &[f32], biases: &[f32], relu: bool) {
    for ((value, &sum), &bias) in output.iter_mut().zip(sums).zip(biases) {
        let sum = sum + bias;
        *value = if relu { sum.max(0.0) } else { sum };
    }
}

fn dense_scalar(input: &[f32], output: &mut [f32], parameters: &[f32], relu: bool) -> usize {
    let weights = input.len() * output.len();
    let extent = output.len();
    for (o, value) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (i, &x) in input.iter().enumerate() {
            sum = x.mul_add(parameters[o + extent * i], sum);
        }
        sum += parameters[weights + o];
        *value = if relu { sum.max(0.0) } else { sum };
    }
    weights + extent
}

// Julia 1.13.2's Array{Float32,3} mapreduce on the qualified znver3 CPU
// initializes with the first two values, accumulates four eight-lane groups,
// then a four-lane epilogue and scalar tail. Preserve that observed order:
// cancellation can change the flux/deviation rejection category. See the
// recorded code-generation evidence.
fn model_sum(input: &[f32], map: impl Fn(f32) -> f32) -> f32 {
    debug_assert_eq!(input.len(), 400);
    let mut lanes = [[-0.0f32; 8]; 4];
    lanes[0][0] = map(input[0]) + map(input[1]);
    for block in input[2..386].as_chunks::<32>().0 {
        for group in 0..4 {
            for lane in 0..8 {
                lanes[group][lane] += map(block[group * 8 + lane]);
            }
        }
    }
    let mut combined = [0.0; 8];
    for lane in 0..8 {
        combined[lane] = ((lanes[1][lane] + lanes[0][lane]) + lanes[2][lane]) + lanes[3][lane];
    }
    let reduce4 = |v: [f32; 4]| (v[0] + v[2]) + (v[1] + v[3]);
    let mut epilogue = [
        reduce4(core::array::from_fn(|i| combined[i] + combined[i + 4])),
        -0.0,
        -0.0,
        -0.0,
    ];
    for block in input[386..398].as_chunks::<4>().0 {
        for lane in 0..4 {
            epilogue[lane] += map(block[lane]);
        }
    }
    (reduce4(epilogue) + map(input[398])) + map(input[399])
}
impl AlgorithmPlan for MlLlowfsPlan {
    /// One 20-by-20 detector region in native row-major storage.
    type Input<'a> = &'a [f32];
    /// Fourteen coefficients in deployed LLOWFS mode order.
    type Output<'a> = &'a mut [f32];
    type Workspace = MlLlowfsWorkspace;
    type Error = MlLlowfsError;
    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if input.len() != 400 || output.len() != 14 {
            return Err(MlLlowfsError::Extent);
        }
        if !self.ready {
            return Err(MlLlowfsError::ParametersMissing);
        }
        for row in 0..20 {
            for col in 0..20 {
                let value = input[row * 20 + col];
                if !value.is_finite() {
                    return Err(MlLlowfsError::NonFiniteInput);
                }
                workspace.input[row + 20 * col] = value;
            }
        }
        let flux = model_sum(&workspace.input, |value| value);
        if !flux.is_finite() || flux <= 0.0 {
            return Err(MlLlowfsError::InvalidFlux);
        }
        for value in &mut workspace.input {
            *value /= flux;
        }
        let mean = model_sum(&workspace.input, |value| value) / 400.0;
        for value in &mut workspace.input {
            *value -= mean;
        }
        // Statistics.std computes a fresh mean of the centered image and uses n-1.
        let centered_mean = model_sum(&workspace.input, |value| value) / 400.0;
        let variance = model_sum(&workspace.input, |value| {
            let d = value - centered_mean;
            d * d
        }) / 399.0;
        let std = Float::sqrt(variance);
        if !std.is_finite() || std <= 0.0 {
            return Err(MlLlowfsError::InvalidStandardDeviation);
        }
        for value in &mut workspace.input {
            *value /= std;
        }
        let p = &self.parameters;
        let mut cursor = convolution::<17, 1, 4, 10>(&workspace.input, &mut workspace.left, p);
        cursor += convolution::<14, 10, 4, 10>(&workspace.left, &mut workspace.right, &p[cursor..]);
        // 14-by-14-by-10 -> 7-by-7-by-10, stride-two 2-by-2 max pooling.
        for ch in 0..10 {
            for col in 0..7 {
                for row in 0..7 {
                    let k = 2 * row + 14 * (2 * col + 14 * ch);
                    workspace.left[row + 7 * (col + 7 * ch)] = workspace.right[k]
                        .max(workspace.right[k + 1])
                        .max(workspace.right[k + 14].max(workspace.right[k + 15]));
                }
            }
        }
        cursor += convolution::<6, 10, 2, 20>(&workspace.left, &mut workspace.right, &p[cursor..]);
        cursor += convolution::<5, 20, 2, 20>(&workspace.right, &mut workspace.left, &p[cursor..]);
        // Flatten(3) preserves the 5-by-5-by-20 column-major tensor order.
        cursor += dense(
            &workspace.left[..500],
            &mut workspace.right[..300],
            &p[cursor..],
            true,
        );
        cursor += dense(
            &workspace.right[..300],
            &mut workspace.left[..200],
            &p[cursor..],
            true,
        );
        cursor += dense(
            &workspace.left[..200],
            &mut workspace.right[..100],
            &p[cursor..],
            true,
        );
        cursor += dense(
            &workspace.right[..100],
            &mut workspace.left[..50],
            &p[cursor..],
            true,
        );
        cursor += dense(
            &workspace.left[..50],
            &mut workspace.right[..14],
            &p[cursor..],
            false,
        );
        debug_assert_eq!(cursor, ML_LLOWFS_PARAMETER_COUNT);
        if workspace.right[..14].iter().any(|v| !v.is_finite()) {
            return Err(MlLlowfsError::NonFiniteOutput);
        }
        output.copy_from_slice(&workspace.right[..14]);
        Ok(())
    }
}

crate::filtergraph_algorithm! {
    /// Fixed-shape CPU inference for the configured LLOWFS model.
    pub MlLlowfsF32 {
        label = "ml-llowfs-f32";
        configuration MlLlowfsF32Configuration { model: String = Construction; }
        plan = MlLlowfsPlan;
        properties = none;
        dimensions |_plan, _workspace| { regions = 1; side = 20; modes_count = 14; parameter_count = ML_LLOWFS_PARAMETER_COUNT; }
        inputs { region_pixels: f32 [regions, side, side] = "region-pixels", schema = crate::schemas::REGION_PIXELS_V1; }
        outputs { modes: f32 [modes_count] = "modes", schema = crate::schemas::LLOWFS_MODES_V1, metadata_from = "region-pixels"; }
        parameters { model_parameters: f32 [parameter_count] = "model-parameters", schema = crate::schemas::ML_LLOWFS_PARAMETERS_V1, replace |plan, values| { plan.with_parameters(values) }; }
        parameter_error = MlLlowfsError;
        prepare |configuration| { let plan = MlLlowfsPlan::new(&configuration.model)?; let workspace = plan.workspace(); Ok((plan, workspace)) }
        prepare_error = MlLlowfsError;
        process |plan, workspace| { plan.process(modes, workspace, region_pixels) }
        process_error = MlLlowfsError;
        reset |_plan, workspace| { workspace.reset(); }
    }
}

#[cfg(all(test, feature = "blas", target_feature = "fma"))]
mod tests {
    use super::{convolution, convolution_scalar, dense, dense_scalar};

    #[test]
    fn layer_outputs_match_scalar_fma_order() {
        let convolution_input: [_; 17 * 17 * 10] =
            core::array::from_fn(|i| (i % 113) as f32 / 29.0 - 2.1);
        let convolution_parameters: [_; 4 * 4 * 10 * 10 + 10] =
            core::array::from_fn(|i| (i % 41) as f32 / 23.0 - 0.89);
        let mut output = [0.0; 14 * 14 * 10];
        let mut reference = output;
        convolution::<14, 10, 4, 10>(&convolution_input, &mut output, &convolution_parameters);
        convolution_scalar(
            &convolution_input,
            &mut reference,
            17,
            10,
            4,
            10,
            &convolution_parameters,
        );
        assert_eq!(output.map(f32::to_bits), reference.map(f32::to_bits));

        // Exercise a full tile, a full vector in the tail, and scalar neurons.
        let mut input = [0.1, 0.2, 0.3, 0.4];
        let mut parameters: [_; 43 * 5] = core::array::from_fn(|i| (i % 29) as f32 / 11.0 - 1.2);
        for relu in [true, false] {
            let mut output = [0.0; 43];
            let mut reference = output;
            dense(&input, &mut output, &parameters, relu);
            dense_scalar(&input, &mut reference, &parameters, relu);
            assert_eq!(output.map(f32::to_bits), reference.map(f32::to_bits));
        }

        input = [2.0, 2.0, 0.0, -0.0];
        for neuron in [0, 32, 42] {
            parameters[neuron] = f32::MAX;
            parameters[43 + neuron] = -f32::MAX;
        }
        let mut output = [0.0; 43];
        let mut reference = output;
        dense(&input, &mut output, &parameters, false);
        dense_scalar(&input, &mut reference, &parameters, false);
        assert_eq!(output.map(f32::to_bits), reference.map(f32::to_bits));
        assert!(
            [0, 32, 42]
                .into_iter()
                .all(|neuron| output[neuron] == f32::INFINITY)
        );
    }
}
