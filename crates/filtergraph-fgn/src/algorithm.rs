//! Generated FGN adaptation for portable FilterGraph algorithm declarations.

use core::cell::UnsafeCell;
use core::ffi::{CStr, c_void};
use core::marker::PhantomData;
use core::mem::{MaybeUninit, size_of};
use core::ptr;

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmFormat, AlgorithmInput, AlgorithmInputs, AlgorithmOperation,
    AlgorithmOperationPlan, AlgorithmOperationWorkspace, AlgorithmOutput, AlgorithmOutputs,
    AlgorithmProcessResult, AlgorithmProperties, ArrayLayout, DeclarationParameterPolicy,
    DeclaredParameters, ElementType, IncrementalDenseReconstructionExecutionError,
    IncrementalDenseReconstructionPlan, NoDeclaredParameters, NoDeclaredProperties,
    PYRAMID_PUPIL_COUNT, PortDirection, PortPublication, PortRole,
    PyramidRowReconstructionExecutionError, PyramidRowReconstructionOutput,
    PyramidRowReconstructionPlan, RowBlockInfo, RowBlockResult, RuntimeDeclaredProperties,
    SampleMetadata, ShackHartmannImagePlan,
};
use serde::de::DeserializeOwned;
use std::ffi::CString;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::{
    Bool8, FixedPlan, InputBuffers, NdarrayFormat, OutputBuffers, ParameterRuntime, PlanStore,
    PluginInstance, PropertyParameterRuntime, PropertyRuntime, abi,
};

/// Generated declarations use fixed stack storage for typed borrowed views.
/// Keep this adapter-specific admission limit independent of ndarray rank and
/// the broader C host limit.
const MAX_DECLARED_PORTS: usize = 64;
const MAX_OPERATION_DATA_PORTS: usize = 3;
const MAX_PROGRESSIVE_RANGES: usize = PYRAMID_PUPIL_COUNT;

/// Map a portable declaration's property capability to one FGN plan owner.
#[doc(hidden)]
pub trait FgnDeclarationProperties<P, Parameters> {
    /// Bounded plan and property owner used by the generated instance.
    type Plans: PlanStore<Plan = P> + Send + Sync + 'static;
}

impl<P> FgnDeclarationProperties<P, NoDeclaredParameters> for NoDeclaredProperties
where
    P: Send + Sync + 'static,
{
    type Plans = FixedPlan<P>;
}

impl<P> FgnDeclarationProperties<P, NoDeclaredParameters> for RuntimeDeclaredProperties
where
    P: AlgorithmProperties + Send + Sync + 'static,
{
    type Plans = PropertyRuntime<P>;
}

impl<P> FgnDeclarationProperties<P, DeclaredParameters> for NoDeclaredProperties
where
    P: Send + Sync + 'static,
{
    type Plans = ParameterRuntime<P>;
}

impl<P> FgnDeclarationProperties<P, DeclaredParameters> for RuntimeDeclaredProperties
where
    P: AlgorithmProperties + Send + Sync + 'static,
{
    type Plans = PropertyParameterRuntime<P>;
}

type DeclarationPlans<D> =
    <<D as AlgorithmDeclaration>::ScalarProperties as FgnDeclarationProperties<
        <D as AlgorithmDeclaration>::Plan,
        <D as AlgorithmDeclaration>::Parameters,
    >>::Plans;

fn validate_declaration<D>() -> Result<(), i32>
where
    D: AlgorithmDeclaration,
{
    if D::NAME.is_empty()
        || D::NAME.len() > abi::LOCAL_NAME_MAX
        || D::PORTS.is_empty()
        || D::PORTS.len() > MAX_DECLARED_PORTS
    {
        return Err(abi::EINVAL);
    }
    if D::PORTS
        .iter()
        .any(|port| port.role() == PortRole::Parameter)
        != D::Parameters::ENABLED
    {
        return Err(abi::EINVAL);
    }
    for (index, port) in D::PORTS.iter().enumerate() {
        if port.name().is_empty()
            || port.name().len() > abi::LOCAL_NAME_MAX
            || D::PORTS[..index]
                .iter()
                .any(|previous| previous.name() == port.name())
            || (port.role() == PortRole::Parameter && port.direction() != PortDirection::Input)
            || (port.publication() == PortPublication::Conditional
                && port.direction() != PortDirection::Output)
        {
            return Err(abi::EINVAL);
        }
        match (port.direction(), port.metadata_source()) {
            (PortDirection::Input, Some(_)) => return Err(abi::EINVAL),
            (PortDirection::Output, Some(source))
                if !D::PORTS.iter().any(|candidate| {
                    candidate.direction() == PortDirection::Input
                        && candidate.role() == PortRole::Data
                        && candidate.name() == source
                }) =>
            {
                return Err(abi::EINVAL);
            }
            _ => {}
        }
    }
    let mut data_inputs = D::PORTS
        .iter()
        .filter(|port| port.direction() == PortDirection::Input && port.role() == PortRole::Data);
    let mut data_outputs = D::PORTS
        .iter()
        .filter(|port| port.direction() == PortDirection::Output && port.role() == PortRole::Data);
    // Optimized operations bind ports by declared order and representation.
    // Names and schemas remain declaration-owned facts projected unchanged by
    // the descriptor; repeating them here would create a second registry.
    match D::OPERATION {
        AlgorithmOperation::General => {}
        AlgorithmOperation::DenseF32 => {
            let (Some(input), Some(output)) = (data_inputs.next(), data_outputs.next()) else {
                return Err(abi::EINVAL);
            };
            if data_inputs.next().is_some()
                || data_outputs.next().is_some()
                || input.element_type() != ElementType::F32
                || output.element_type() != ElementType::F32
                || output.publication() != PortPublication::Always
            {
                return Err(abi::EINVAL);
            }
        }
        AlgorithmOperation::ShackHartmannImageF32 => {
            let (Some(input), Some(slopes), Some(flux), Some(validity)) = (
                data_inputs.next(),
                data_outputs.next(),
                data_outputs.next(),
                data_outputs.next(),
            ) else {
                return Err(abi::EINVAL);
            };
            if data_inputs.next().is_some()
                || data_outputs.next().is_some()
                || input.element_type() != ElementType::F32
                || slopes.element_type() != ElementType::F32
                || flux.element_type() != ElementType::F32
                || validity.element_type() != ElementType::Bool8
                || [slopes, flux, validity]
                    .iter()
                    .any(|output| output.publication() != PortPublication::Always)
            {
                return Err(abi::EINVAL);
            }
        }
        AlgorithmOperation::IncrementalDenseReconstructionF32 => {
            let (Some(measurements), Some(columns), Some(count), Some(output)) = (
                data_inputs.next(),
                data_inputs.next(),
                data_inputs.next(),
                data_outputs.next(),
            ) else {
                return Err(abi::EINVAL);
            };
            if data_inputs.next().is_some()
                || data_outputs.next().is_some()
                || measurements.element_type() != ElementType::F32
                || columns.element_type() != ElementType::U32
                || count.element_type() != ElementType::U32
                || output.element_type() != ElementType::F32
                || output.publication() != PortPublication::Conditional
                || output.metadata_source() != Some(measurements.name())
            {
                return Err(abi::EINVAL);
            }
        }

        AlgorithmOperation::PyramidRowReconstructionF32 => {
            let (Some(input), Some(pixels), Some(mean), Some(reconstructed)) = (
                data_inputs.next(),
                data_outputs.next(),
                data_outputs.next(),
                data_outputs.next(),
            ) else {
                return Err(abi::EINVAL);
            };
            if data_inputs.next().is_some()
                || data_outputs.next().is_some()
                || input.element_type() != ElementType::F32
                || [pixels, mean, reconstructed].iter().any(|output| {
                    output.element_type() != ElementType::F32
                        || output.publication() != PortPublication::Conditional
                        || output.metadata_source() != Some(input.name())
                })
            {
                return Err(abi::EINVAL);
            }
        }
    }
    Ok(())
}

struct DeploymentFormat {
    rate: Option<(u32, u32)>,
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn port_rate(
    format: &AlgorithmFormat,
    deployment: &DeploymentFormat,
) -> Result<Option<(u32, u32)>, i32> {
    let (ratio_numerator, ratio_denominator) = format.rate_ratio();
    let Some((base_numerator, base_denominator)) = deployment.rate else {
        return if (ratio_numerator, ratio_denominator) == (1, 1) {
            Ok(None)
        } else {
            Err(abi::EINVAL)
        };
    };
    if base_numerator == 0 || base_denominator == 0 {
        return Err(abi::EINVAL);
    }
    let numerator = u64::from(base_numerator)
        .checked_mul(u64::from(ratio_numerator))
        .ok_or(abi::EOVERFLOW)?;
    let denominator = u64::from(base_denominator)
        .checked_mul(u64::from(ratio_denominator))
        .ok_or(abi::EOVERFLOW)?;
    let divisor = greatest_common_divisor(numerator, denominator);
    let numerator = u32::try_from(numerator / divisor).map_err(|_| abi::EOVERFLOW)?;
    let denominator = u32::try_from(denominator / divisor).map_err(|_| abi::EOVERFLOW)?;
    Ok(Some((numerator, denominator)))
}

fn decode_configuration<C>(config: &CStr) -> Result<(C, DeploymentFormat), i32>
where
    C: DeserializeOwned,
{
    let mut value: serde_json::Value =
        serde_json::from_slice(config.to_bytes()).map_err(|_| abi::EINVAL)?;
    let object = value.as_object_mut().ok_or(abi::EINVAL)?;
    let rate = object
        .remove("rate")
        .map(serde_json::from_value::<[u32; 2]>)
        .transpose()
        .map_err(|_| abi::EINVAL)?
        .map(|rate| (rate[0], rate[1]));
    if object.remove("profiles").is_some() {
        return Err(abi::EINVAL);
    }
    let configuration = serde_json::from_value(value).map_err(|_| abi::EINVAL)?;
    Ok((configuration, DeploymentFormat { rate }))
}

fn fgn_format(
    format: &AlgorithmFormat,
    deployment: &DeploymentFormat,
) -> Result<NdarrayFormat, i32> {
    let shape: Box<[u32]> = format.shape().into();
    let layout = match format.layout() {
        ArrayLayout::RowMajor => abi::LAYOUT_ROW_MAJOR,
        ArrayLayout::ColumnMajor => abi::LAYOUT_COLUMN_MAJOR,
    };
    let rate = port_rate(format, deployment)?;
    let result = match format.element_type() {
        ElementType::Bool8 => {
            NdarrayFormat::packed_with_rate::<Bool8>(shape, layout, rate, Some(format.schema()))
        }
        ElementType::I8 => {
            NdarrayFormat::packed_with_rate::<i8>(shape, layout, rate, Some(format.schema()))
        }
        ElementType::U8 => {
            NdarrayFormat::packed_with_rate::<u8>(shape, layout, rate, Some(format.schema()))
        }
        ElementType::I16 => {
            NdarrayFormat::packed_with_rate::<i16>(shape, layout, rate, Some(format.schema()))
        }
        ElementType::U16 => {
            NdarrayFormat::packed_with_rate::<u16>(shape, layout, rate, Some(format.schema()))
        }
        ElementType::I32 => {
            NdarrayFormat::packed_with_rate::<i32>(shape, layout, rate, Some(format.schema()))
        }
        ElementType::U32 => {
            NdarrayFormat::packed_with_rate::<u32>(shape, layout, rate, Some(format.schema()))
        }
        ElementType::I64 => {
            NdarrayFormat::packed_with_rate::<i64>(shape, layout, rate, Some(format.schema()))
        }
        ElementType::U64 => {
            NdarrayFormat::packed_with_rate::<u64>(shape, layout, rate, Some(format.schema()))
        }
        ElementType::F32 => {
            NdarrayFormat::packed_with_rate::<f32>(shape, layout, rate, Some(format.schema()))
        }
        ElementType::F64 => {
            NdarrayFormat::packed_with_rate::<f64>(shape, layout, rate, Some(format.schema()))
        }
    };
    result.map_err(|_| abi::EINVAL)
}

unsafe fn bool8_input(buffer: &abi::Buffer, elements: usize) -> Result<&[bool], i32> {
    // SAFETY: the checked FGN Bool8 storage is borrowed for this call.
    let values = unsafe { crate::typed_slice::<Bool8>(buffer, elements) }?;
    if values.iter().any(|value| value.get().is_none()) {
        return Err(abi::EINVAL);
    }
    // SAFETY: Bool8 and bool both have one-byte alignment and size. Every byte
    // was validated as zero or one before constructing the bool slice.
    Ok(unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<bool>(), elements) })
}

unsafe fn bool8_output(buffer: &mut abi::Buffer, elements: usize) -> Result<&mut [bool], i32> {
    // SAFETY: the complete output storage is validated before initialization.
    let values = unsafe { crate::typed_slice_mut::<Bool8>(buffer, elements) }?;
    values.fill(Bool8::new(false));
    // SAFETY: Bool8 and bool both have one-byte alignment and size. The fill
    // established valid false values before constructing the mutable slice;
    // Rust bool writes preserve the zero-or-one representation.
    Ok(unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<bool>(), elements) })
}

unsafe fn input_view<'a>(
    buffer: &'a abi::Buffer,
    format: &AlgorithmFormat,
) -> Result<AlgorithmInput<'a>, i32> {
    let elements = format.elements();
    Ok(match format.element_type() {
        ElementType::Bool8 => AlgorithmInput::Bool8(unsafe { bool8_input(buffer, elements) }?),
        ElementType::I8 => {
            AlgorithmInput::I8(unsafe { crate::typed_slice::<i8>(buffer, elements) }?)
        }
        ElementType::U8 => {
            AlgorithmInput::U8(unsafe { crate::typed_slice::<u8>(buffer, elements) }?)
        }
        ElementType::I16 => {
            AlgorithmInput::I16(unsafe { crate::typed_slice::<i16>(buffer, elements) }?)
        }
        ElementType::U16 => {
            AlgorithmInput::U16(unsafe { crate::typed_slice::<u16>(buffer, elements) }?)
        }
        ElementType::I32 => {
            AlgorithmInput::I32(unsafe { crate::typed_slice::<i32>(buffer, elements) }?)
        }
        ElementType::U32 => {
            AlgorithmInput::U32(unsafe { crate::typed_slice::<u32>(buffer, elements) }?)
        }
        ElementType::I64 => {
            AlgorithmInput::I64(unsafe { crate::typed_slice::<i64>(buffer, elements) }?)
        }
        ElementType::U64 => {
            AlgorithmInput::U64(unsafe { crate::typed_slice::<u64>(buffer, elements) }?)
        }
        ElementType::F32 => {
            AlgorithmInput::F32(unsafe { crate::typed_slice::<f32>(buffer, elements) }?)
        }
        ElementType::F64 => {
            AlgorithmInput::F64(unsafe { crate::typed_slice::<f64>(buffer, elements) }?)
        }
    })
}

unsafe fn output_view<'a>(
    buffer: &'a mut abi::Buffer,
    format: &AlgorithmFormat,
) -> Result<AlgorithmOutput<'a>, i32> {
    let elements = format.elements();
    Ok(match format.element_type() {
        ElementType::Bool8 => {
            AlgorithmOutput::Bool8(Some(unsafe { bool8_output(buffer, elements) }?))
        }
        ElementType::I8 => AlgorithmOutput::I8(Some(unsafe {
            crate::typed_slice_mut::<i8>(buffer, elements)
        }?)),
        ElementType::U8 => AlgorithmOutput::U8(Some(unsafe {
            crate::typed_slice_mut::<u8>(buffer, elements)
        }?)),
        ElementType::I16 => AlgorithmOutput::I16(Some(unsafe {
            crate::typed_slice_mut::<i16>(buffer, elements)
        }?)),
        ElementType::U16 => AlgorithmOutput::U16(Some(unsafe {
            crate::typed_slice_mut::<u16>(buffer, elements)
        }?)),
        ElementType::I32 => AlgorithmOutput::I32(Some(unsafe {
            crate::typed_slice_mut::<i32>(buffer, elements)
        }?)),
        ElementType::U32 => AlgorithmOutput::U32(Some(unsafe {
            crate::typed_slice_mut::<u32>(buffer, elements)
        }?)),
        ElementType::I64 => AlgorithmOutput::I64(Some(unsafe {
            crate::typed_slice_mut::<i64>(buffer, elements)
        }?)),
        ElementType::U64 => AlgorithmOutput::U64(Some(unsafe {
            crate::typed_slice_mut::<u64>(buffer, elements)
        }?)),
        ElementType::F32 => AlgorithmOutput::F32(Some(unsafe {
            crate::typed_slice_mut::<f32>(buffer, elements)
        }?)),
        ElementType::F64 => AlgorithmOutput::F64(Some(unsafe {
            crate::typed_slice_mut::<f64>(buffer, elements)
        }?)),
    })
}

fn invoke_declaration<D>(
    plan: &D::Plan,
    workspace: &mut D::Workspace,
    formats: &[AlgorithmFormat],
    inputs: &InputBuffers<'_>,
    outputs: &mut OutputBuffers<'_>,
) -> Result<AlgorithmProcessResult, i32>
where
    D: AlgorithmDeclaration,
{
    let mut input_storage: [MaybeUninit<AlgorithmInput<'_>>; MAX_DECLARED_PORTS] =
        [const { MaybeUninit::uninit() }; MAX_DECLARED_PORTS];
    let mut output_storage: [MaybeUninit<AlgorithmOutput<'_>>; MAX_DECLARED_PORTS] =
        [const { MaybeUninit::uninit() }; MAX_DECLARED_PORTS];
    let mut input_metadata = [None; MAX_DECLARED_PORTS];
    let mut input_buffer_index = 0;
    let mut data_input_count = 0;
    let mut data_output_count = 0;
    let mut output_buffers = outputs.buffers.iter_mut();

    for (port, format) in D::PORTS.iter().zip(formats) {
        match (port.direction(), port.role()) {
            (PortDirection::Input, PortRole::Data) => {
                let buffer = inputs.buffers.get(input_buffer_index).ok_or(abi::EINVAL)?;
                // SAFETY: the callback owns an immutable input buffer for this
                // call; the shared helper validates all nested storage.
                input_storage[data_input_count].write(unsafe { input_view(buffer, format) }?);
                input_metadata[data_input_count] = inputs.sample_metadata(input_buffer_index)?;
                data_input_count += 1;
                input_buffer_index += 1;
            }
            (PortDirection::Input, PortRole::Parameter) => {
                if !inputs.is_disconnected(input_buffer_index)? {
                    return Err(abi::EINVAL);
                }
                input_buffer_index += 1;
            }
            (PortDirection::Output, PortRole::Data) => {
                let buffer = output_buffers.next().ok_or(abi::EINVAL)?;
                // SAFETY: each iteration borrows a distinct output descriptor.
                // OutputSizeGuard rejected overlap between their complete
                // nested regions before this function was called.
                output_storage[data_output_count].write(unsafe { output_view(buffer, format) }?);
                data_output_count += 1;
            }
            (PortDirection::Output, PortRole::Parameter) => return Err(abi::EINVAL),
        }
    }

    // SAFETY: exactly the tracked prefixes were initialized above. Their
    // borrowed buffers outlive this invocation and no storage escapes it.
    let input_views = unsafe {
        core::slice::from_raw_parts(
            input_storage.as_ptr().cast::<AlgorithmInput<'_>>(),
            data_input_count,
        )
    };
    let output_views = unsafe {
        core::slice::from_raw_parts_mut(
            output_storage.as_mut_ptr().cast::<AlgorithmOutput<'_>>(),
            data_output_count,
        )
    };
    D::process(
        plan,
        workspace,
        AlgorithmInputs::with_metadata(input_views, &input_metadata[..data_input_count])
            .map_err(|_| abi::EINVAL)?,
        AlgorithmOutputs::new(output_views),
    )
    .map_err(|_| abi::EINVAL)
}

#[derive(Clone, Copy)]
struct BoundDataPort<'a> {
    buffer_index: usize,
    format: &'a AlgorithmFormat,
}

struct BoundOperationPorts<'a> {
    inputs: [Option<BoundDataPort<'a>>; MAX_OPERATION_DATA_PORTS],
    outputs: [Option<BoundDataPort<'a>>; MAX_OPERATION_DATA_PORTS],
}

fn bind_operation_ports<'a, D>(
    formats: &'a [AlgorithmFormat],
    inputs: &InputBuffers<'_>,
) -> Result<BoundOperationPorts<'a>, i32>
where
    D: AlgorithmDeclaration,
{
    if formats.len() != D::PORTS.len() {
        return Err(abi::EINVAL);
    }
    let mut bound = BoundOperationPorts {
        inputs: [None; MAX_OPERATION_DATA_PORTS],
        outputs: [None; MAX_OPERATION_DATA_PORTS],
    };
    let mut input_buffer_index = 0;
    let mut output_buffer_index = 0;
    let mut data_input_count = 0;
    let mut data_output_count = 0;

    for (port, format) in D::PORTS.iter().zip(formats) {
        match (port.direction(), port.role()) {
            (PortDirection::Input, PortRole::Data) => {
                let slot = bound.inputs.get_mut(data_input_count).ok_or(abi::EINVAL)?;
                *slot = Some(BoundDataPort {
                    buffer_index: input_buffer_index,
                    format,
                });
                data_input_count += 1;
                input_buffer_index += 1;
            }
            (PortDirection::Input, PortRole::Parameter) => {
                if !inputs.is_disconnected(input_buffer_index)? {
                    return Err(abi::EINVAL);
                }
                input_buffer_index += 1;
            }
            (PortDirection::Output, PortRole::Data) => {
                let slot = bound
                    .outputs
                    .get_mut(data_output_count)
                    .ok_or(abi::EINVAL)?;
                *slot = Some(BoundDataPort {
                    buffer_index: output_buffer_index,
                    format,
                });
                data_output_count += 1;
                output_buffer_index += 1;
            }
            (PortDirection::Output, PortRole::Parameter) => return Err(abi::EINVAL),
        }
    }
    Ok(bound)
}

fn invoke_dense_f32<D>(
    executor: &abi::Executor,
    plan: &D::Plan,
    formats: &[AlgorithmFormat],
    inputs: &InputBuffers<'_>,
    outputs: &mut OutputBuffers<'_>,
) -> Result<AlgorithmProcessResult, i32>
where
    D: AlgorithmDeclaration,
{
    let AlgorithmOperationPlan::DenseF32(plan) = D::operation_plan(plan) else {
        return Err(abi::EINVAL);
    };
    let BoundOperationPorts {
        inputs: [Some(input), None, None],
        outputs: [Some(output), None, None],
    } = bind_operation_ports::<D>(formats, inputs)?
    else {
        return Err(abi::EINVAL);
    };
    let input_index = input.buffer_index;
    let input_format = input.format;
    let output_index = output.buffer_index;
    let output_format = output.format;
    if input_format.element_type() != ElementType::F32
        || output_format.element_type() != ElementType::F32
        || input_format.elements() != plan.columns()
        || output_format.elements() != plan.rows()
    {
        return Err(abi::EINVAL);
    }
    let input_buffer = inputs.buffers.get(input_index).ok_or(abi::EINVAL)?;
    // SAFETY: the process callback owns this admitted immutable input.
    let AlgorithmInput::F32(input) = unsafe { input_view(input_buffer, input_format) }? else {
        return Err(abi::EINVAL);
    };
    if input.iter().any(|value| !value.is_finite()) {
        return Err(abi::EINVAL);
    }
    let output_buffer = outputs.buffers.get_mut(output_index).ok_or(abi::EINVAL)?;
    // SAFETY: the process callback owns this distinct admitted mutable output.
    let AlgorithmOutput::F32(Some(output)) = unsafe { output_view(output_buffer, output_format) }?
    else {
        return Err(abi::EINVAL);
    };
    let task = abi::DenseF32Task {
        struct_size: size_of::<abi::DenseF32Task>() as u32,
        flags: 0,
        matrix: plan.matrix().as_ptr(),
        input: input.as_ptr(),
        output: output.as_mut_ptr(),
        n_rows: u32::try_from(plan.rows()).map_err(|_| abi::EOVERFLOW)?,
        matrix_row_stride: u32::try_from(plan.columns()).map_err(|_| abi::EOVERFLOW)?,
        first_column: 0,
        n_columns: u32::try_from(plan.columns()).map_err(|_| abi::EOVERFLOW)?,
    };
    let run = executor.run_dense_f32.ok_or(abi::EINVAL)?;
    // SAFETY: instantiate validated the host interface. The task borrows
    // admitted arrays and immutable prepared matrix storage through this call.
    let result = unsafe { run(executor.data, &task) };
    if result != 0 {
        return Err(result);
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(abi::EINVAL);
    }
    Ok(AlgorithmProcessResult::complete())
}

struct ShackHartmannLaneContext {
    plan: *const ShackHartmannImagePlan<f32>,
    image: *const f32,
    image_len: usize,
    slopes: *mut f32,
    flux: *mut f32,
    validity: *mut bool,
    subapertures: usize,
    partition_quantum: usize,
    n_lanes: u32,
}

fn proportional_partition(extent: usize, lane: usize, n_lanes: usize) -> usize {
    let quotient = extent / n_lanes;
    let remainder = extent % n_lanes;
    lane * quotient + lane.min(remainder)
}

fn checked_partition_range(
    extent: usize,
    quantum: usize,
    lane: usize,
    n_lanes: usize,
) -> Option<(usize, usize)> {
    if extent == 0 || quantum == 0 || n_lanes == 0 || lane >= n_lanes {
        return None;
    }
    let chunks = extent.div_ceil(quantum);
    let first_chunk = proportional_partition(chunks, lane, n_lanes);
    let last_chunk = proportional_partition(chunks, lane + 1, n_lanes);
    let first = first_chunk.checked_mul(quantum)?;
    let last = if last_chunk == chunks {
        extent
    } else {
        last_chunk.checked_mul(quantum)?.min(extent)
    };
    (first < last).then_some((first, last))
}

unsafe extern "C" fn run_shack_hartmann_lane(data: *mut c_void, lane: u32, n_lanes: u32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() || n_lanes == 0 || lane >= n_lanes {
            return -abi::EINVAL;
        }
        // SAFETY: the synchronous executor borrows this coordinator-owned
        // context through the complete lane rendezvous.
        let context = unsafe { &*data.cast::<ShackHartmannLaneContext>() };
        if context.n_lanes != n_lanes
            || context.plan.is_null()
            || context.image.is_null()
            || context.slopes.is_null()
            || context.flux.is_null()
            || context.validity.is_null()
            || context.subapertures == 0
            || context.partition_quantum == 0
        {
            return -abi::EINVAL;
        }
        let Some((first, last)) = checked_partition_range(
            context.subapertures,
            context.partition_quantum,
            lane as usize,
            n_lanes as usize,
        ) else {
            return -abi::EINVAL;
        };
        let Some(slope_first) = first.checked_mul(2) else {
            return -abi::EOVERFLOW;
        };
        let Some(slope_len) = (last - first).checked_mul(2) else {
            return -abi::EOVERFLOW;
        };

        // SAFETY: invoke_shack_hartmann_image_f32 validated every complete
        // region. Partition boundaries are disjoint, cache-line isolated, and
        // the synchronous call prevents any borrowed slice from escaping.
        let plan = unsafe { &*context.plan };
        let image = unsafe { core::slice::from_raw_parts(context.image, context.image_len) };
        let slopes =
            unsafe { core::slice::from_raw_parts_mut(context.slopes.add(slope_first), slope_len) };
        let flux =
            unsafe { core::slice::from_raw_parts_mut(context.flux.add(first), last - first) };
        let validity =
            unsafe { core::slice::from_raw_parts_mut(context.validity.add(first), last - first) };
        match plan.process_image_partition(slopes, flux, validity, image, first) {
            Ok(()) => 0,
            Err(_) => -abi::EINVAL,
        }
    }))
    .unwrap_or(-abi::EFAULT)
}

fn byte_region<T>(pointer: *const T, elements: usize) -> Result<(usize, usize), i32> {
    let bytes = elements.checked_mul(size_of::<T>()).ok_or(abi::EOVERFLOW)?;
    let begin = pointer.addr();
    let end = begin.checked_add(bytes).ok_or(abi::EOVERFLOW)?;
    Ok((begin, end))
}

fn regions_overlap(left: (usize, usize), right: (usize, usize)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

fn validate_disjoint_byte_regions(regions: &[(usize, usize)]) -> Result<(), i32> {
    for left in 0..regions.len() {
        for right in left + 1..regions.len() {
            if regions_overlap(regions[left], regions[right]) {
                return Err(abi::EINVAL);
            }
        }
    }
    Ok(())
}

fn invoke_shack_hartmann_image_f32<D>(
    executor: &abi::Executor,
    plan: &D::Plan,
    formats: &[AlgorithmFormat],
    inputs: &InputBuffers<'_>,
    outputs: &mut OutputBuffers<'_>,
) -> Result<AlgorithmProcessResult, i32>
where
    D: AlgorithmDeclaration,
{
    let AlgorithmOperationPlan::ShackHartmannImageF32(plan) = D::operation_plan(plan) else {
        return Err(abi::EINVAL);
    };
    let BoundOperationPorts {
        inputs: [Some(image), None, None],
        outputs: [Some(slopes), Some(flux), Some(validity)],
    } = bind_operation_ports::<D>(formats, inputs)?
    else {
        return Err(abi::EINVAL);
    };
    let image_index = image.buffer_index;
    let image_format = image.format;
    let slopes_index = slopes.buffer_index;
    let slopes_format = slopes.format;
    let flux_index = flux.buffer_index;
    let flux_format = flux.format;
    let validity_index = validity.buffer_index;
    let validity_format = validity.format;

    let image_buffer = inputs.buffers.get(image_index).ok_or(abi::EINVAL)?;
    // SAFETY: the process callback owns this admitted immutable input.
    let AlgorithmInput::F32(image) = unsafe { input_view(image_buffer, image_format) }? else {
        return Err(abi::EINVAL);
    };
    let image_pointer = image.as_ptr();
    let image_len = image.len();

    let (slopes_pointer, slopes_len) = {
        let buffer = outputs.buffers.get_mut(slopes_index).ok_or(abi::EINVAL)?;
        // SAFETY: the process callback owns this admitted mutable output.
        let AlgorithmOutput::F32(Some(values)) = unsafe { output_view(buffer, slopes_format) }?
        else {
            return Err(abi::EINVAL);
        };
        (values.as_mut_ptr(), values.len())
    };
    let (flux_pointer, flux_len) = {
        let buffer = outputs.buffers.get_mut(flux_index).ok_or(abi::EINVAL)?;
        // SAFETY: this is a distinct host-admitted output region.
        let AlgorithmOutput::F32(Some(values)) = unsafe { output_view(buffer, flux_format) }?
        else {
            return Err(abi::EINVAL);
        };
        (values.as_mut_ptr(), values.len())
    };
    let (validity_pointer, validity_len) = {
        let buffer = outputs.buffers.get_mut(validity_index).ok_or(abi::EINVAL)?;
        // SAFETY: this is a distinct host-admitted output region. Bool8 is
        // initialized before a Rust bool slice is constructed.
        let AlgorithmOutput::Bool8(Some(values)) = unsafe { output_view(buffer, validity_format) }?
        else {
            return Err(abi::EINVAL);
        };
        (values.as_mut_ptr(), values.len())
    };

    let subapertures = plan.measurement().len();
    if image_len
        != plan
            .regions()
            .image_size()
            .0
            .checked_mul(plan.regions().image_size().1)
            .ok_or(abi::EOVERFLOW)?
        || slopes_len != subapertures.checked_mul(2).ok_or(abi::EOVERFLOW)?
        || flux_len != subapertures
        || validity_len != subapertures
    {
        return Err(abi::EINVAL);
    }

    validate_disjoint_byte_regions(&[
        byte_region(image_pointer, image_len)?,
        byte_region(slopes_pointer, slopes_len)?,
        byte_region(flux_pointer, flux_len)?,
        byte_region(validity_pointer, validity_len)?,
    ])?;

    let cache_line_size = executor.cache_line_size as usize;
    let aligned = cache_line_size != 0
        && slopes_pointer.addr().is_multiple_of(cache_line_size)
        && flux_pointer.addr().is_multiple_of(cache_line_size)
        && validity_pointer.addr().is_multiple_of(cache_line_size);
    let available_lanes = usize::try_from(executor.n_lanes).map_err(|_| abi::EOVERFLOW)?;
    let chunk_count = if aligned {
        subapertures.div_ceil(cache_line_size)
    } else {
        1
    };
    let n_lanes = available_lanes.min(chunk_count).max(1);
    let context = ShackHartmannLaneContext {
        plan: ptr::from_ref(plan),
        image: image_pointer,
        image_len,
        slopes: slopes_pointer,
        flux: flux_pointer,
        validity: validity_pointer,
        subapertures,
        partition_quantum: if aligned {
            cache_line_size
        } else {
            subapertures
        },
        n_lanes: u32::try_from(n_lanes).map_err(|_| abi::EOVERFLOW)?,
    };

    let result = if n_lanes == 1 {
        // SAFETY: this is the same callback contract on the coordinator only.
        unsafe {
            run_shack_hartmann_lane(
                ptr::from_ref(&context).cast_mut().cast(),
                0,
                context.n_lanes,
            )
        }
    } else {
        let run = executor.run_lanes.ok_or(abi::EINVAL)?;
        let task = abi::LanesTask {
            struct_size: size_of::<abi::LanesTask>() as u32,
            flags: 0,
            data: ptr::from_ref(&context).cast_mut().cast(),
            run: Some(run_shack_hartmann_lane),
            n_lanes: context.n_lanes,
            reserved: 0,
        };
        // SAFETY: the executor call is synchronous. Context and every array it
        // borrows remain live and are inaccessible to the coordinator until
        // all disjoint lane slices have been returned.
        unsafe { run(executor.data, &task) }
    };
    if result != 0 {
        return Err(result);
    }
    Ok(AlgorithmProcessResult::complete())
}

trait ProgressiveReconstructionF32: Sync {
    fn accumulate_partition(
        &self,
        input: &[f32],
        output: &mut [f32],
        columns: core::ops::Range<usize>,
        first_output: usize,
    ) -> Result<(), ()>;
}

impl ProgressiveReconstructionF32 for PyramidRowReconstructionPlan<f32> {
    fn accumulate_partition(
        &self,
        input: &[f32],
        output: &mut [f32],
        columns: core::ops::Range<usize>,
        first_output: usize,
    ) -> Result<(), ()> {
        self.accumulate_reconstruction_partition(input, output, columns, first_output)
            .map_err(|_| ())
    }
}

/// Safe borrowed view reusing the existing synchronous output-row partitioner.
struct IndexedMeasurementReconstruction<'a> {
    plan: &'a IncrementalDenseReconstructionPlan<f32>,
    columns: &'a [u32],
}

impl ProgressiveReconstructionF32 for IndexedMeasurementReconstruction<'_> {
    fn accumulate_partition(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: core::ops::Range<usize>,
        first_output: usize,
    ) -> Result<(), ()> {
        self.plan
            .accumulate_reconstruction_partition(input, self.columns, output, batch, first_output)
            .map_err(|_| ())
    }
}

struct ProgressiveRowLaneContext<P> {
    plan: *const P,
    input: *const f32,
    input_count: usize,
    output: *mut f32,
    output_count: usize,
    ranges: *const core::ops::Range<usize>,
    range_count: usize,
    cache_line_size: usize,
    n_lanes: u32,
}

fn checked_cache_line_partition_boundary(
    output: *mut f32,
    extent: usize,
    cache_line_size: usize,
    lane: usize,
    n_lanes: usize,
) -> Option<usize> {
    if output.is_null()
        || extent == 0
        || cache_line_size == 0
        || !cache_line_size.is_power_of_two()
        || n_lanes == 0
        || lane > n_lanes
    {
        return None;
    }
    if lane == 0 {
        return Some(0);
    }
    if lane == n_lanes {
        return Some(extent);
    }
    let row = proportional_partition(extent, lane, n_lanes);
    let row_bytes = row.checked_mul(size_of::<f32>())?;
    let address = output.addr().checked_add(row_bytes)?;
    let misalignment = address % cache_line_size;
    let adjustment = if misalignment == 0 {
        0
    } else {
        cache_line_size - misalignment
    };
    if adjustment % size_of::<f32>() != 0 {
        return None;
    }
    row.checked_add(adjustment / size_of::<f32>())
        .map(|row| row.min(extent))
}

unsafe extern "C" fn run_progressive_row_lane<P>(data: *mut c_void, lane: u32, n_lanes: u32) -> i32
where
    P: ProgressiveReconstructionF32,
{
    catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() || n_lanes == 0 || lane >= n_lanes {
            return -abi::EINVAL;
        }
        // SAFETY: the synchronous executor borrows this coordinator-owned
        // context until every lane has returned.
        let context = unsafe { &*data.cast::<ProgressiveRowLaneContext<P>>() };
        if context.n_lanes != n_lanes
            || context.plan.is_null()
            || context.input.is_null()
            || context.output.is_null()
            || context.input_count == 0
            || context.output_count == 0
            || context.ranges.is_null()
            || context.range_count == 0
            || context.range_count > MAX_PROGRESSIVE_RANGES
            || context.cache_line_size == 0
            || !context.cache_line_size.is_power_of_two()
        {
            return -abi::EINVAL;
        }
        let Some(first) = checked_cache_line_partition_boundary(
            context.output,
            context.output_count,
            context.cache_line_size,
            lane as usize,
            n_lanes as usize,
        ) else {
            return -abi::EINVAL;
        };
        let Some(last) = checked_cache_line_partition_boundary(
            context.output,
            context.output_count,
            context.cache_line_size,
            lane as usize + 1,
            n_lanes as usize,
        ) else {
            return -abi::EINVAL;
        };
        if first > last || last > context.output_count {
            return -abi::EINVAL;
        }
        if first == last {
            return 0;
        }

        // SAFETY: the adapter formed the context from valid safe borrows. Every
        // internal boundary is an address-level cache-line boundary, so these
        // mutable ranges are disjoint and do not share a mutable cache line.
        // The scientific input and plan are immutable through the rendezvous.
        let plan = unsafe { &*context.plan };
        let input = unsafe { core::slice::from_raw_parts(context.input, context.input_count) };
        let output =
            unsafe { core::slice::from_raw_parts_mut(context.output.add(first), last - first) };
        // SAFETY: the coordinator-owned bounded range list remains immutable
        // through this synchronous executor rendezvous.
        let ranges = unsafe { core::slice::from_raw_parts(context.ranges, context.range_count) };
        for columns in ranges {
            if columns.start >= columns.end || columns.end > context.input_count {
                return -abi::EINVAL;
            }
            if plan
                .accumulate_partition(input, output, columns.clone(), first)
                .is_err()
            {
                return -abi::EINVAL;
            }
        }
        0
    }))
    .unwrap_or(-abi::EFAULT)
}

fn run_progressive_reconstruction<P>(
    executor: &abi::Executor,
    plan: &P,
    input: &[f32],
    output: &mut [f32],
    ranges: &[core::ops::Range<usize>],
) -> Result<(), i32>
where
    P: ProgressiveReconstructionF32,
{
    if ranges.is_empty() {
        return Ok(());
    }
    if ranges.len() > MAX_PROGRESSIVE_RANGES
        || ranges
            .iter()
            .any(|columns| columns.start >= columns.end || columns.end > input.len())
    {
        return Err(abi::EINVAL);
    }
    let cache_line_size = executor.cache_line_size as usize;
    if cache_line_size == 0 || !cache_line_size.is_power_of_two() {
        return Err(abi::EINVAL);
    }
    let output_bytes = output
        .len()
        .checked_mul(size_of::<f32>())
        .ok_or(abi::EOVERFLOW)?;
    let spanned_bytes = (output.as_ptr().addr() % cache_line_size)
        .checked_add(output_bytes)
        .ok_or(abi::EOVERFLOW)?;
    let cache_lines = spanned_bytes.div_ceil(cache_line_size);
    let available_lanes = usize::try_from(executor.n_lanes).map_err(|_| abi::EOVERFLOW)?;
    let n_lanes = available_lanes.min(cache_lines).min(output.len()).max(1);
    if n_lanes == 1 {
        for columns in ranges {
            plan.accumulate_partition(input, output, columns.clone(), 0)
                .map_err(|_| abi::EINVAL)?;
        }
        return Ok(());
    }

    let context = ProgressiveRowLaneContext {
        plan: ptr::from_ref(plan),
        input: input.as_ptr(),
        input_count: input.len(),
        output: output.as_mut_ptr(),
        output_count: output.len(),
        ranges: ranges.as_ptr(),
        range_count: ranges.len(),
        cache_line_size,
        n_lanes: u32::try_from(n_lanes).map_err(|_| abi::EOVERFLOW)?,
    };
    let task = abi::LanesTask {
        struct_size: size_of::<abi::LanesTask>() as u32,
        flags: 0,
        data: ptr::from_ref(&context).cast_mut().cast(),
        run: Some(run_progressive_row_lane::<P>),
        n_lanes: context.n_lanes,
        reserved: 0,
    };
    let run = executor.run_lanes.ok_or(abi::EINVAL)?;
    // SAFETY: the call is synchronous. The immutable plan/input and every
    // disjoint output shard remain live and inaccessible to this coordinator
    // until all helper callbacks have returned.
    let result = unsafe { run(executor.data, &task) };
    if result == 0 { Ok(()) } else { Err(result) }
}

fn row_block_info(sample: SampleMetadata) -> Result<RowBlockInfo, i32> {
    Ok(RowBlockInfo {
        sequence: sample.sequence(),
        first_row: usize::try_from(sample.offset()).map_err(|_| abi::EOVERFLOW)?,
        terminal: sample.is_terminal(),
        discontinuity: sample.is_discontinuous(),
        corrupted: sample.is_corrupted(),
    })
}

fn row_block_process_result(
    result: RowBlockResult,
    sample: SampleMetadata,
) -> AlgorithmProcessResult {
    match result {
        RowBlockResult::Pending | RowBlockResult::Dropped(_) => AlgorithmProcessResult::deferred(),
        RowBlockResult::Complete {
            sequence: _,
            discontinuity,
        } => AlgorithmProcessResult::complete_with_metadata(
            sample
                .with_offset(0)
                .with_terminal(true)
                .with_discontinuity(discontinuity)
                .with_corrupted(false),
        ),
    }
}

fn invoke_incremental_dense_reconstruction_f32<D>(
    executor: &abi::Executor,
    declared_plan: &D::Plan,
    declared_workspace: &mut D::Workspace,
    formats: &[AlgorithmFormat],
    inputs: &InputBuffers<'_>,
    outputs: &mut OutputBuffers<'_>,
) -> Result<AlgorithmProcessResult, i32>
where
    D: AlgorithmDeclaration,
{
    let AlgorithmOperationPlan::IncrementalDenseReconstructionF32(plan) =
        D::operation_plan(declared_plan)
    else {
        return Err(abi::EINVAL);
    };
    let BoundOperationPorts {
        inputs: [Some(measurements), Some(columns), Some(count)],
        outputs: [Some(output), None, None],
    } = bind_operation_ports::<D>(formats, inputs)?
    else {
        return Err(abi::EINVAL);
    };
    if measurements.format.elements() != plan.block_capacity()
        || columns.format.elements() != plan.block_capacity()
        || count.format.elements() != 1
        || output.format.elements() != plan.reconstructor().rows()
    {
        return Err(abi::EINVAL);
    }
    let values_buffer = inputs
        .buffers
        .get(measurements.buffer_index)
        .ok_or(abi::EINVAL)?;
    let indices_buffer = inputs
        .buffers
        .get(columns.buffer_index)
        .ok_or(abi::EINVAL)?;
    let count_buffer = inputs.buffers.get(count.buffer_index).ok_or(abi::EINVAL)?;
    // SAFETY: the callback owns the admitted immutable measurement inputs;
    // existing typed-view helpers validate their nested ndarray storage.
    let AlgorithmInput::F32(values) = (unsafe { input_view(values_buffer, measurements.format) })?
    else {
        return Err(abi::EINVAL);
    };
    let AlgorithmInput::U32(indices) = (unsafe { input_view(indices_buffer, columns.format) })?
    else {
        return Err(abi::EINVAL);
    };
    let AlgorithmInput::U32(count_value) = (unsafe { input_view(count_buffer, count.format) })?
    else {
        return Err(abi::EINVAL);
    };
    let Some(sample) = inputs.sample_metadata(measurements.buffer_index)? else {
        // Required scientific metadata is an expected sample failure. Apply
        // the declaration's abandonment hook before a caller retries a block.
        D::plan_adopted(declared_plan, declared_workspace);
        return Err(abi::EINVAL);
    };
    let AlgorithmOperationWorkspace::IncrementalDenseReconstructionF32(workspace) =
        D::operation_workspace(declared_workspace)
    else {
        return Err(abi::EINVAL);
    };
    let output_buffer = outputs
        .buffers
        .get_mut(output.buffer_index)
        .ok_or(abi::EINVAL)?;
    // SAFETY: the callback owns this distinct admitted mutable output buffer.
    let AlgorithmOutput::F32(Some(output)) =
        (unsafe { output_view(output_buffer, output.format) })?
    else {
        return Err(abi::EINVAL);
    };
    let result = plan.process_block_with_reconstruction(
        output,
        workspace,
        values,
        indices,
        count_value[0],
        row_block_info(sample)?,
        |plan, values, indices, candidate| {
            let indexed = IndexedMeasurementReconstruction {
                plan,
                columns: indices,
            };
            run_progressive_reconstruction(
                executor,
                &indexed,
                values,
                candidate,
                core::slice::from_ref(&(0..values.len())),
            )
        },
    );
    match result {
        Ok(result) => Ok(row_block_process_result(result, sample)),
        Err(IncrementalDenseReconstructionExecutionError::Algorithm(_)) => Err(abi::EINVAL),
        Err(IncrementalDenseReconstructionExecutionError::Executor(error)) => Err(error),
    }
}

fn invoke_pyramid_row_reconstruction_f32<D>(
    executor: &abi::Executor,
    declared_plan: &D::Plan,
    declared_workspace: &mut D::Workspace,
    formats: &[AlgorithmFormat],
    inputs: &InputBuffers<'_>,
    outputs: &mut OutputBuffers<'_>,
) -> Result<AlgorithmProcessResult, i32>
where
    D: AlgorithmDeclaration,
{
    let AlgorithmOperationPlan::PyramidRowReconstructionF32(plan) =
        D::operation_plan(declared_plan)
    else {
        return Err(abi::EINVAL);
    };
    let BoundOperationPorts {
        inputs: [Some(input), None, None],
        outputs: [Some(pixels), Some(mean), Some(reconstructed)],
    } = bind_operation_ports::<D>(formats, inputs)?
    else {
        return Err(abi::EINVAL);
    };
    let input_index = input.buffer_index;
    let input_format = input.format;
    let pixels_index = pixels.buffer_index;
    let pixels_format = pixels.format;
    let mean_index = mean.buffer_index;
    let mean_format = mean.format;
    let reconstructed_index = reconstructed.buffer_index;
    let reconstructed_format = reconstructed.format;

    let expected_input = plan
        .pixels()
        .block_rows()
        .checked_mul(plan.pixels().image().pupil_regions().image_size().1)
        .ok_or(abi::EOVERFLOW)?;
    if input_format.element_type() != ElementType::F32
        || pixels_format.element_type() != ElementType::F32
        || mean_format.element_type() != ElementType::F32
        || reconstructed_format.element_type() != ElementType::F32
        || input_format.elements() != expected_input
        || pixels_format.elements() != plan.reconstructor().columns()
        || mean_format.elements() != 1
        || reconstructed_format.elements() != plan.reconstructor().rows()
    {
        return Err(abi::EINVAL);
    }

    let input_buffer = inputs.buffers.get(input_index).ok_or(abi::EINVAL)?;
    // SAFETY: the process callback owns this admitted immutable row block.
    let AlgorithmInput::F32(block) = unsafe { input_view(input_buffer, input_format) }? else {
        return Err(abi::EINVAL);
    };
    let Some(sample) = inputs.sample_metadata(input_index)? else {
        // Required scientific metadata is an expected sample failure. Apply
        // the declaration's abandonment hook before a caller retries a block.
        D::plan_adopted(declared_plan, declared_workspace);
        return Err(abi::EINVAL);
    };
    let AlgorithmOperationWorkspace::PyramidRowReconstructionF32(workspace) =
        D::operation_workspace(declared_workspace)
    else {
        return Err(abi::EINVAL);
    };

    let (pixels_pointer, pixels_len) = {
        let buffer = outputs.buffers.get_mut(pixels_index).ok_or(abi::EINVAL)?;
        // SAFETY: the process callback owns this admitted mutable output.
        let AlgorithmOutput::F32(Some(values)) = unsafe { output_view(buffer, pixels_format) }?
        else {
            return Err(abi::EINVAL);
        };
        (values.as_mut_ptr(), values.len())
    };
    let (mean_pointer, mean_len) = {
        let buffer = outputs.buffers.get_mut(mean_index).ok_or(abi::EINVAL)?;
        // SAFETY: this is a distinct host-admitted output region.
        let AlgorithmOutput::F32(Some(values)) = unsafe { output_view(buffer, mean_format) }?
        else {
            return Err(abi::EINVAL);
        };
        (values.as_mut_ptr(), values.len())
    };
    let (reconstructed_pointer, reconstructed_len) = {
        let buffer = outputs
            .buffers
            .get_mut(reconstructed_index)
            .ok_or(abi::EINVAL)?;
        // SAFETY: this is a distinct host-admitted output region.
        let AlgorithmOutput::F32(Some(values)) =
            unsafe { output_view(buffer, reconstructed_format) }?
        else {
            return Err(abi::EINVAL);
        };
        (values.as_mut_ptr(), values.len())
    };

    validate_disjoint_byte_regions(&[
        byte_region(block.as_ptr(), block.len())?,
        byte_region(pixels_pointer, pixels_len)?,
        byte_region(mean_pointer, mean_len)?,
        byte_region(reconstructed_pointer, reconstructed_len)?,
    ])?;

    // SAFETY: output_view validated each complete region, the explicit region
    // checks above prove they are pairwise disjoint from one another and the
    // immutable input, and none of these borrows escape this synchronous call.
    let pixels = unsafe { core::slice::from_raw_parts_mut(pixels_pointer, pixels_len) };
    let mean = unsafe { &mut *mean_pointer };
    let reconstructed =
        unsafe { core::slice::from_raw_parts_mut(reconstructed_pointer, reconstructed_len) };
    let info = row_block_info(sample)?;
    let result = plan.process_block_with_reconstruction(
        PyramidRowReconstructionOutput::new(pixels, mean, reconstructed),
        workspace,
        block,
        info,
        |plan, normalized, reconstructed, ranges| {
            run_progressive_reconstruction(executor, plan, normalized, reconstructed, ranges)
        },
    );
    match result {
        Ok(result) => Ok(row_block_process_result(result, sample)),
        Err(PyramidRowReconstructionExecutionError::Algorithm(_)) => Err(abi::EINVAL),
        Err(PyramidRowReconstructionExecutionError::Executor(error)) => Err(error),
    }
}

fn finish_output(
    outputs: &mut OutputBuffers<'_>,
    output: usize,
    format: &AlgorithmFormat,
) -> Result<(), i32> {
    let elements = format.elements();
    match format.element_type() {
        ElementType::Bool8 => outputs.finish::<Bool8>(output, elements),
        ElementType::I8 => outputs.finish::<i8>(output, elements),
        ElementType::U8 => outputs.finish::<u8>(output, elements),
        ElementType::I16 => outputs.finish::<i16>(output, elements),
        ElementType::U16 => outputs.finish::<u16>(output, elements),
        ElementType::I32 => outputs.finish::<i32>(output, elements),
        ElementType::U32 => outputs.finish::<u32>(output, elements),
        ElementType::I64 => outputs.finish::<i64>(output, elements),
        ElementType::U64 => outputs.finish::<u64>(output, elements),
        ElementType::F32 => outputs.finish::<f32>(output, elements),
        ElementType::F64 => outputs.finish::<f64>(output, elements),
    }
}

fn input_index<D>(name: &str) -> Option<usize>
where
    D: AlgorithmDeclaration,
{
    let mut input = 0;
    for port in D::PORTS {
        if port.direction() == PortDirection::Input {
            if port.role() == PortRole::Data && port.name() == name {
                return Some(input);
            }
            input += 1;
        }
    }
    None
}

/// FGN-owned instance generated from one portable algorithm declaration.
#[doc(hidden)]
pub struct AlgorithmInstance<D>
where
    D: AlgorithmDeclaration,
    D::ScalarProperties: FgnDeclarationProperties<D::Plan, D::Parameters>,
{
    plans: DeclarationPlans<D>,
    workspace: UnsafeCell<D::Workspace>,
    portable_formats: Box<[AlgorithmFormat]>,
    fgn_formats: Box<[NdarrayFormat]>,
    required_sample_metadata: Box<[u8]>,
    declaration: PhantomData<D>,
}

// SAFETY: the selected plan store owns synchronized publication. FGN provides
// one non-reentrant data-loop owner for workspace and serializes reset against
// processing.
unsafe impl<D> Sync for AlgorithmInstance<D>
where
    D: AlgorithmDeclaration + Send + Sync,
    D::Plan: Send + Sync,
    D::Workspace: Send,
    D::ScalarProperties: FgnDeclarationProperties<D::Plan, D::Parameters>,
{
}

impl<D> AlgorithmInstance<D>
where
    D: AlgorithmDeclaration,
    D::Plan: Send + Sync,
    D::Workspace: Send,
    D::ScalarProperties: FgnDeclarationProperties<D::Plan, D::Parameters>,
{
    fn process_impl(
        &self,
        executor: Option<&abi::Executor>,
        inputs: InputBuffers<'_>,
        outputs: &mut OutputBuffers<'_>,
    ) -> Result<(), i32> {
        // SAFETY: FGN serializes the data-loop call and reset for this instance.
        let workspace = unsafe { &mut *self.workspace.get() };
        if self.plans.adopt() {
            D::plan_adopted(self.plans.active_plan(), workspace);
        }
        let mut required_input_index = 0;
        for (port, &required) in D::PORTS.iter().zip(&self.required_sample_metadata) {
            if port.direction() == PortDirection::Input {
                if required != 0 {
                    crate::metadata::validate_sample(
                        inputs.sample_metadata(required_input_index)?,
                        required,
                    )?;
                }
                required_input_index += 1;
            }
        }
        let process_result = match (D::OPERATION, executor) {
            // BLAS owns the complete dense operation for every host lane count.
            // Mixing its reduction order with host row kernels would violate
            // worker-count invariance and bypass the selected backend.
            (AlgorithmOperation::DenseF32, Some(executor))
                if !cfg!(feature = "blas") && executor.run_dense_f32.is_some() =>
            {
                invoke_dense_f32::<D>(
                    executor,
                    self.plans.active_plan(),
                    &self.portable_formats,
                    &inputs,
                    outputs,
                )?
            }
            (AlgorithmOperation::ShackHartmannImageF32, Some(executor))
                if executor.n_lanes > 1 && executor.run_lanes.is_some() =>
            {
                invoke_shack_hartmann_image_f32::<D>(
                    executor,
                    self.plans.active_plan(),
                    &self.portable_formats,
                    &inputs,
                    outputs,
                )?
            }
            (AlgorithmOperation::IncrementalDenseReconstructionF32, Some(executor))
                if executor.n_lanes > 1 && executor.run_lanes.is_some() =>
            {
                invoke_incremental_dense_reconstruction_f32::<D>(
                    executor,
                    self.plans.active_plan(),
                    workspace,
                    &self.portable_formats,
                    &inputs,
                    outputs,
                )?
            }

            (AlgorithmOperation::PyramidRowReconstructionF32, Some(executor))
                if executor.n_lanes > 1 && executor.run_lanes.is_some() =>
            {
                invoke_pyramid_row_reconstruction_f32::<D>(
                    executor,
                    self.plans.active_plan(),
                    workspace,
                    &self.portable_formats,
                    &inputs,
                    outputs,
                )?
            }
            _ => invoke_declaration::<D>(
                self.plans.active_plan(),
                workspace,
                &self.portable_formats,
                &inputs,
                outputs,
            )?,
        };

        if process_result == AlgorithmProcessResult::Deferred {
            if D::PORTS.iter().any(|port| {
                port.direction() == PortDirection::Output
                    && port.publication() != PortPublication::Conditional
            }) {
                return Err(abi::EINVAL);
            }
            return Ok(());
        }

        let mut output_index = 0;
        for (port, format) in D::PORTS.iter().zip(&self.portable_formats) {
            if port.direction() != PortDirection::Output {
                continue;
            }
            if let Some(source) = port.metadata_source() {
                outputs.copy_metadata_from(
                    output_index,
                    &inputs,
                    input_index::<D>(source).ok_or(abi::EINVAL)?,
                )?;
            }
            if let Some(metadata) = process_result.metadata() {
                outputs.set_sample_metadata(output_index, metadata)?;
            }
            finish_output(outputs, output_index, format)?;
            output_index += 1;
        }
        Ok(())
    }
}

unsafe impl<D> PluginInstance for AlgorithmInstance<D>
where
    D: AlgorithmDeclaration + Send + Sync,
    D::Configuration: DeserializeOwned,
    D::Plan: Send + Sync,
    D::Workspace: Send,
    D::ScalarProperties: FgnDeclarationProperties<D::Plan, D::Parameters>,
{
    type Properties = DeclarationPlans<D>;

    // Dynamic declaration descriptors supply the actual name and port table.
    const NAME: &'static CStr = c"";
    const PORTS: &'static [abi::PortInfo] = &[];
    const PARAMETER_TRANSACTIONS: bool = D::Parameters::ENABLED;

    fn input_port_count(&self) -> usize {
        D::PORTS
            .iter()
            .filter(|port| port.direction() == PortDirection::Input)
            .count()
    }

    fn output_port_count(&self) -> usize {
        D::PORTS
            .iter()
            .filter(|port| port.direction() == PortDirection::Output)
            .count()
    }

    fn instantiate(config: &CStr) -> Result<Self, i32> {
        validate_declaration::<D>()?;
        let (configuration, deployment) = decode_configuration(config)?;
        let prepared = D::prepare(configuration).map_err(|_| abi::EINVAL)?;
        let (plan, mut workspace, portable_formats) = prepared.into_parts();
        let operation_matches_plan = matches!(
            (D::OPERATION, D::operation_plan(&plan)),
            (AlgorithmOperation::General, AlgorithmOperationPlan::General)
                | (
                    AlgorithmOperation::DenseF32,
                    AlgorithmOperationPlan::DenseF32(_)
                )
                | (
                    AlgorithmOperation::ShackHartmannImageF32,
                    AlgorithmOperationPlan::ShackHartmannImageF32(_)
                )
                | (
                    AlgorithmOperation::IncrementalDenseReconstructionF32,
                    AlgorithmOperationPlan::IncrementalDenseReconstructionF32(_)
                )
                | (
                    AlgorithmOperation::PyramidRowReconstructionF32,
                    AlgorithmOperationPlan::PyramidRowReconstructionF32(_)
                )
        );
        let operation_matches_workspace = matches!(
            (D::OPERATION, D::operation_workspace(&mut workspace)),
            (
                AlgorithmOperation::General,
                AlgorithmOperationWorkspace::General
            ) | (
                AlgorithmOperation::DenseF32,
                AlgorithmOperationWorkspace::General
            ) | (
                AlgorithmOperation::ShackHartmannImageF32,
                AlgorithmOperationWorkspace::General
            ) | (
                AlgorithmOperation::IncrementalDenseReconstructionF32,
                AlgorithmOperationWorkspace::IncrementalDenseReconstructionF32(_)
            ) | (
                AlgorithmOperation::PyramidRowReconstructionF32,
                AlgorithmOperationWorkspace::PyramidRowReconstructionF32(_)
            )
        );
        if !operation_matches_plan || !operation_matches_workspace {
            return Err(abi::EINVAL);
        }
        if portable_formats.len() != D::PORTS.len()
            || portable_formats
                .iter()
                .zip(D::PORTS)
                .any(|(format, port)| format.element_type() != port.element_type())
        {
            return Err(abi::EINVAL);
        }
        let mut fgn_formats = Vec::with_capacity(portable_formats.len());
        for format in &portable_formats {
            fgn_formats.push(fgn_format(format, &deployment)?);
        }
        let required_sample_metadata = D::PORTS
            .iter()
            .enumerate()
            .map(|(index, _port)| {
                D::required_sample_metadata(&plan, index)
                    .iter()
                    .try_fold(0, |mask, fact| {
                        crate::metadata::required_fact(fact).map(|(bit, _)| mask | bit)
                    })
            })
            .collect::<Result<Vec<u8>, i32>>()?
            .into_boxed_slice();
        Ok(Self {
            required_sample_metadata,
            plans: DeclarationPlans::<D>::from_plan(plan)?,
            workspace: UnsafeCell::new(workspace),
            portable_formats,
            fgn_formats: fgn_formats.into_boxed_slice(),
            declaration: PhantomData,
        })
    }

    fn required_sample_metadata(&self, port: u32) -> &'static [&'static str] {
        D::required_sample_metadata(self.plans.active_plan(), port as usize)
    }

    fn port_format(&self, port: u32) -> Option<&abi::Format> {
        self.fgn_formats
            .get(usize::try_from(port).ok()?)
            .map(NdarrayFormat::as_abi)
    }

    fn properties(&self) -> &Self::Properties {
        &self.plans
    }

    fn process(
        &self,
        inputs: InputBuffers<'_>,
        outputs: &mut OutputBuffers<'_>,
    ) -> Result<(), i32> {
        self.process_impl(None, inputs, outputs)
    }

    fn process_with_executor(
        &self,
        executor: &abi::Executor,
        inputs: InputBuffers<'_>,
        outputs: &mut OutputBuffers<'_>,
    ) -> Result<(), i32> {
        self.process_impl(Some(executor), inputs, outputs)
    }

    fn reset(&self) -> Result<(), i32> {
        // SAFETY: the lifecycle serializes reset outside processing.
        let workspace = unsafe { &mut *self.workspace.get() };
        D::reset(self.plans.active_plan(), workspace);
        Ok(())
    }

    unsafe fn prepare_parameter(
        &self,
        port: u32,
        buffer: &abi::Buffer,
    ) -> Result<*mut c_void, i32> {
        let port = usize::try_from(port).map_err(|_| abi::EINVAL)?;
        let port_info = D::PORTS.get(port).ok_or(abi::EINVAL)?;
        if port_info.direction() != PortDirection::Input || port_info.role() != PortRole::Parameter
        {
            return Err(abi::EINVAL);
        }
        let format = self.portable_formats.get(port).ok_or(abi::EINVAL)?;
        // SAFETY: the parameter callback lends the complete buffer for this
        // preparation call; the shared view helper validates nested storage.
        let parameter = unsafe { input_view(buffer, format) }?;
        // SAFETY: the same callback borrow covers the optional Header metadata.
        let sequence = unsafe { crate::header_sequence(buffer.buffer) }?;
        self.plans.prepare_parameter_with(sequence, |plan| {
            D::prepare_parameter(plan, port, parameter)
                .ok_or(())?
                .map_err(|_| ())
        })
    }

    unsafe fn prepare_parameters(&self, parameters: &[abi::Parameter]) -> Result<*mut c_void, i32> {
        if !D::Parameters::ENABLED || parameters.is_empty() {
            return Err(abi::EINVAL);
        }
        if parameters.len() > MAX_DECLARED_PORTS {
            return Err(abi::E2BIG);
        }

        let mut assignments = Vec::new();
        assignments
            .try_reserve_exact(parameters.len())
            .map_err(|_| abi::ENOMEM)?;
        let mut sequence = None;
        for (index, parameter) in parameters.iter().enumerate() {
            if parameter.reserved != 0 {
                return Err(abi::EINVAL);
            }
            if parameters[..index]
                .iter()
                .any(|previous| previous.port == parameter.port)
            {
                return Err(abi::EEXIST);
            }
            let port = usize::try_from(parameter.port).map_err(|_| abi::EINVAL)?;
            let port_info = D::PORTS.get(port).ok_or(abi::EINVAL)?;
            if port_info.direction() != PortDirection::Input
                || port_info.role() != PortRole::Parameter
            {
                return Err(abi::EINVAL);
            }
            let format = self.portable_formats.get(port).ok_or(abi::EINVAL)?;
            if parameter.buffer.format != self.fgn_formats[port].as_abi() {
                return Err(abi::EINVAL);
            }
            // SAFETY: the batch callback lends every complete buffer through
            // the end of this scientific replacement-plan preparation.
            let input = unsafe { input_view(&parameter.buffer, format) }?;
            // SAFETY: the same callback borrow covers Header metadata.
            if let Some(current) = unsafe { crate::header_sequence(parameter.buffer.buffer) }? {
                if sequence.is_some_and(|expected| expected != current) {
                    return Err(abi::EINVAL);
                }
                sequence = Some(current);
            }
            assignments.push((port, input));
        }

        self.plans.prepare_parameter_with(sequence, |plan| {
            D::prepare_parameters(plan, assignments.as_slice()).map_err(|_| ())
        })
    }

    fn adopt_parameters(&self) {
        // SAFETY: graph-transaction adoption runs on the same serialized data
        // owner immediately before node processing.
        let workspace = unsafe { &mut *self.workspace.get() };
        if self.plans.adopt() {
            D::plan_adopted(self.plans.active_plan(), workspace);
        }
    }

    unsafe fn commit_parameter(&self, prepared: *mut c_void) {
        // SAFETY: FGN forwards a token from this instance exactly once.
        unsafe { self.plans.commit_parameter(prepared) }
    }

    unsafe fn discard_parameter(&self, prepared: *mut c_void) {
        // SAFETY: FGN forwards an unpublished token from this instance once.
        unsafe { self.plans.discard_parameter(prepared) }
    }
}

/// Owned ABI descriptor derived solely from one portable declaration.
#[doc(hidden)]
pub struct AlgorithmDescriptor {
    descriptor: abi::Descriptor,
    name: CString,
    _port_names: Box<[CString]>,
    _ports: Box<[abi::PortInfo]>,
    _metas: Box<[Box<[abi::SpaMetaRequirement]>]>,
}

// SAFETY: every raw pointer targets immutable heap storage owned by this value.
unsafe impl Send for AlgorithmDescriptor {}
// SAFETY: descriptor storage is immutable after construction.
unsafe impl Sync for AlgorithmDescriptor {}

impl AlgorithmDescriptor {
    /// Derive one FGN descriptor from a portable declaration.
    pub fn new<D>() -> Result<Self, i32>
    where
        D: AlgorithmDeclaration + Send + Sync,
        D::Configuration: DeserializeOwned,
        D::Plan: Send + Sync,
        D::Workspace: Send,
        D::ScalarProperties: FgnDeclarationProperties<D::Plan, D::Parameters>,
    {
        Self::new_with_metadata::<D>(&[])
    }

    /// Derive a descriptor with caller-selected additional layouts in port order.
    /// Scientific requirements and inherited output layouts are merged cold.
    pub fn new_with_metadata<D>(metas: &[&[abi::SpaMetaRequirement]]) -> Result<Self, i32>
    where
        D: AlgorithmDeclaration + Send + Sync,
        D::Configuration: DeserializeOwned,
        D::Plan: Send + Sync,
        D::Workspace: Send,
        D::ScalarProperties: FgnDeclarationProperties<D::Plan, D::Parameters>,
    {
        validate_declaration::<D>()?;
        if !metas.is_empty() && metas.len() != D::PORTS.len() {
            return Err(abi::EINVAL);
        }
        let mut layouts = Vec::with_capacity(D::PORTS.len());
        for (index, port) in D::PORTS.iter().enumerate() {
            let supplied = metas.get(index).copied().unwrap_or(&[]);
            if supplied.len() >= abi::MAX_METAS {
                return Err(abi::EINVAL);
            }
            let mut records = supplied.to_vec();
            for (j, r) in records.iter().enumerate() {
                if r.type_ == abi::META_HEADER
                    || r.reserved != 0
                    || r.flags & !abi::META_REQUIREMENT_REQUIRED != 0
                    || records[..j]
                        .iter()
                        .any(|previous| previous.type_ == r.type_)
                {
                    return Err(abi::EINVAL);
                }
                // A synthetic aligned pointer permits reuse of the pure layout gate.
                crate::metadata::validate_layout(&abi::SpaMeta {
                    type_: r.type_,
                    size: r.size,
                    data: core::ptr::without_provenance_mut(8),
                })?;
            }
            if port.role() == PortRole::Parameter
                && (!records.is_empty()
                    || !port.required_sample_metadata().is_empty()
                    || !port.optional_sample_metadata().is_empty())
            {
                return Err(abi::EINVAL);
            }
            if port.direction() != PortDirection::Input
                && (!port.required_sample_metadata().is_empty()
                    || !port.optional_sample_metadata().is_empty())
            {
                return Err(abi::EINVAL);
            }
            for (facts, required) in [
                (port.optional_sample_metadata(), false),
                (port.required_sample_metadata(), true),
            ] {
                for fact in facts {
                    let (_, type_) = crate::metadata::required_fact(fact)?;
                    if type_ == abi::META_HEADER {
                        continue;
                    }
                    let flags = if required {
                        abi::META_REQUIREMENT_REQUIRED
                    } else {
                        0
                    };
                    if let Some(r) = records.iter_mut().find(|r| r.type_ == type_) {
                        r.flags |= flags;
                    } else {
                        records.push(abi::SpaMetaRequirement {
                            type_,
                            size: crate::metadata::record_size(type_).ok_or(abi::EINVAL)?,
                            flags,
                            reserved: 0,
                        });
                    }
                }
            }
            layouts.push(records);
        }
        // Native PRESERVE outputs must admit every declared input layout. Flags
        // on the output remain optional: a different triggering input may omit it.
        let inherited = D::PORTS
            .iter()
            .zip(&layouts)
            .filter(|(p, _)| p.direction() == PortDirection::Input && p.role() == PortRole::Data)
            .flat_map(|(_, records)| records.iter().copied())
            .collect::<Vec<_>>();
        for (port, records) in D::PORTS.iter().zip(&mut layouts) {
            if port.direction() != PortDirection::Output || port.metadata_source().is_none() {
                continue;
            }
            for r in &inherited {
                if crate::metadata::transport_owned(r.type_) {
                    continue;
                }
                if let Some(existing) = records.iter().find(|m| m.type_ == r.type_) {
                    if existing.size != r.size {
                        return Err(abi::EINVAL);
                    }
                } else {
                    records.push(abi::SpaMetaRequirement { flags: 0, ..*r });
                }
            }
        }
        for records in &layouts {
            if records.len() >= abi::MAX_METAS
                || records
                    .iter()
                    .try_fold(32_usize, |bytes, r| {
                        bytes.checked_add((r.size as usize + 7) & !7)
                    })
                    .is_none_or(|bytes| bytes > abi::MAX_META_BYTES)
            {
                return Err(abi::EINVAL);
            }
        }
        let metas = layouts
            .into_iter()
            .map(Vec::into_boxed_slice)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let name = CString::new(D::NAME).map_err(|_| abi::EINVAL)?;
        let mut port_names = Vec::with_capacity(D::PORTS.len());
        for port in D::PORTS {
            port_names.push(CString::new(port.name()).map_err(|_| abi::EINVAL)?);
        }
        let port_names = port_names.into_boxed_slice();
        let mut ports = Vec::with_capacity(D::PORTS.len());
        for (index, (port, name)) in D::PORTS.iter().zip(&port_names).enumerate() {
            let direction = match port.direction() {
                PortDirection::Input => abi::DIRECTION_INPUT,
                PortDirection::Output => abi::DIRECTION_OUTPUT,
            };
            let mut flags = match port.role() {
                PortRole::Data => 0,
                PortRole::Parameter => abi::PORT_OPTIONAL | abi::PORT_PARAMETER,
            };
            if port.publication() == PortPublication::Conditional {
                flags |= abi::PORT_CONDITIONAL;
            }
            let info = abi::PortInfo::new(
                u32::try_from(index).map_err(|_| abi::EOVERFLOW)?,
                direction,
                flags,
                name.as_ptr(),
            );

            let info = abi::PortInfo {
                n_metas: metas[index].len() as u32,
                metas: metas[index].as_ptr(),
                propagation: if port.direction() == PortDirection::Output
                    && port.metadata_source().is_none()
                {
                    abi::META_PROPAGATION_PRODUCE
                } else {
                    abi::META_PROPAGATION_PRESERVE
                },
                ..info
            };
            ports.push(info);
        }
        let ports = ports.into_boxed_slice();
        let descriptor = crate::descriptor_with::<AlgorithmInstance<D>>(
            name.as_ptr(),
            ports.len() as u32,
            ports.as_ptr(),
        );
        Ok(Self {
            descriptor,
            name,
            _port_names: port_names,
            _ports: ports,
            _metas: metas,
        })
    }

    /// Return the stable ABI descriptor.
    pub const fn as_abi(&self) -> &abi::Descriptor {
        &self.descriptor
    }

    fn name(&self) -> &CStr {
        &self.name
    }
}

/// Instance-lifetime registry generated from selected portable declarations.
#[doc(hidden)]
pub struct AlgorithmRegistry {
    plugin: abi::Plugin,
    plugin_name: CString,
    descriptors: Box<[AlgorithmDescriptor]>,
}

// SAFETY: owned strings and descriptor arrays are immutable after construction.
unsafe impl Send for AlgorithmRegistry {}
// SAFETY: all pointers target the registry's stable heap allocations.
unsafe impl Sync for AlgorithmRegistry {}

impl AlgorithmRegistry {
    /// Construct one checked declaration registry.
    pub fn new(
        plugin_name: &str,
        descriptors: Vec<AlgorithmDescriptor>,
        find_descriptor: abi::FindDescriptor,
    ) -> Result<Self, i32> {
        if plugin_name.is_empty()
            || plugin_name.len() > abi::LOCAL_NAME_MAX
            || descriptors.is_empty()
        {
            return Err(abi::EINVAL);
        }
        for (index, descriptor) in descriptors.iter().enumerate() {
            if descriptors[..index]
                .iter()
                .any(|previous| previous.name() == descriptor.name())
            {
                return Err(abi::EEXIST);
            }
        }
        let plugin_name = CString::new(plugin_name).map_err(|_| abi::EINVAL)?;
        let descriptors = descriptors.into_boxed_slice();
        let plugin = abi::Plugin {
            struct_size: size_of::<abi::Plugin>() as u32,
            abi_version: abi::ABI_VERSION,
            name: plugin_name.as_ptr(),
            find_descriptor: Some(find_descriptor),
            flags: abi::PLUGIN_RETAIN_LIBRARY,
        };
        Ok(Self {
            plugin,
            plugin_name,
            descriptors,
        })
    }

    /// Return the stable plugin interface.
    pub const fn plugin(&self) -> &abi::Plugin {
        &self.plugin
    }

    /// Return the number of selected declarations.
    pub const fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Return whether no declarations were selected.
    pub const fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Resolve one exact declaration label.
    pub fn find(&self, name: &CStr) -> *const abi::Descriptor {
        self.descriptors
            .iter()
            .find(|descriptor| descriptor.name() == name)
            .map_or(ptr::null(), |descriptor| ptr::from_ref(descriptor.as_abi()))
    }

    /// Return the stable plugin-family name.
    pub fn name(&self) -> &CStr {
        &self.plugin_name
    }
}

/// Export selected portable declarations as one FGN shared-library registry.
///
/// The list is a build-time deployment selection only. Every descriptor, port,
/// configuration decoder, format, property surface, and callback is derived
/// from the declaration beside the scientific algorithm.
#[macro_export]
macro_rules! export_algorithm_declarations {
    ($plugin_name:literal, [$($declaration:ty),+ $(,)?] $(,)?) => {
        static FILTERGRAPH_ALGORITHM_REGISTRY: std::sync::OnceLock<
            Result<$crate::AlgorithmRegistry, i32>,
        > = std::sync::OnceLock::new();

        fn filtergraph_algorithm_registry() -> Option<&'static $crate::AlgorithmRegistry> {
            FILTERGRAPH_ALGORITHM_REGISTRY
                .get_or_init(|| {
                    let descriptors = vec![
                        $($crate::AlgorithmDescriptor::new::<$declaration>()?),+
                    ];
                    $crate::AlgorithmRegistry::new(
                        $plugin_name,
                        descriptors,
                        filtergraph_fgn_find_algorithm_descriptor,
                    )
                })
                .as_ref()
                .ok()
        }

        unsafe extern "C" fn filtergraph_fgn_find_algorithm_descriptor(
            name: *const core::ffi::c_char,
        ) -> *const $crate::abi::Descriptor {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if name.is_null() {
                    return core::ptr::null();
                }
                let Some(registry) = filtergraph_algorithm_registry() else {
                    return core::ptr::null();
                };
                // SAFETY: FGN supplies a NUL-terminated local descriptor name.
                registry.find(unsafe { core::ffi::CStr::from_ptr(name) })
            }))
            .unwrap_or(core::ptr::null())
        }

        #[doc = "Return the number of portable declarations in this registry."]
        #[allow(dead_code)]
        pub fn filtergraph_fgn_registered_declaration_count() -> usize {
            filtergraph_algorithm_registry().map_or(0, $crate::AlgorithmRegistry::len)
        }

        #[doc = "Return the FilterGraph FGN plugin interface for the requested ABI version."]
        #[unsafe(no_mangle)]
        pub extern "C" fn spa_filter_graph_ndarray_plugin_get_interface(
            abi_version: u32,
        ) -> *const $crate::abi::Plugin {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if abi_version != $crate::abi::ABI_VERSION {
                    return core::ptr::null();
                }
                filtergraph_algorithm_registry()
                    .map_or(core::ptr::null(), |registry| core::ptr::from_ref(registry.plugin()))
            }))
            .unwrap_or(core::ptr::null())
        }
    };
}

#[cfg(test)]
mod tests {
    use super::checked_partition_range;

    #[test]
    fn lane_partition_covers_extreme_extents_without_wrapping() {
        let extent = usize::MAX;
        let mut next = 0;
        for lane in 0..4 {
            let (first, last) = checked_partition_range(extent, 64, lane, 4).unwrap();
            assert_eq!(first, next);
            assert!(last > first);
            next = last;
        }
        assert_eq!(next, extent);
    }
}
