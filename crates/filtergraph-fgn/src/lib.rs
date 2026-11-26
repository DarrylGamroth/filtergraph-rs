//! Reusable FGN ABI adaptation for transport-independent FilterGraph algorithms.
//!
//! A plugin-instance wrapper owns port formats and buffer-to-slice conversion. The
//! shared adapter owns the C callback table, schema projection, transactional
//! property preparation, bounded plan slots, generations, panic containment,
//! and process-boundary publication.

use core::ffi::{CStr, c_char, c_void};
use core::ptr;
use std::ffi::CString;
use std::panic::{AssertUnwindSafe, catch_unwind};

pub mod abi;
mod algorithm;
mod declaration;
mod metadata;
mod parameter;
mod ports;
mod properties;
mod property_parameter;
mod publication;
mod source;
mod unary;

#[doc(hidden)]
pub use algorithm::{
    AlgorithmDescriptor, AlgorithmInstance, AlgorithmRegistry, FgnDeclarationProperties,
};
pub use declaration::{DeclaredPlugin, ParameterBuffer, PluginDeclaration, PreparedInstance};
pub use parameter::{
    ACTIVE_PARAMETER_SEQUENCE_ID, ParameterRuntime, REQUESTED_PARAMETER_SEQUENCE_ID,
};
pub use ports::{DataInput, DataOutput, ParameterInput};
pub use properties::{
    ACTIVE_GENERATION_ID, PropertyAdapterError, PropertyRuntime, PropertySchema,
    REQUESTED_GENERATION_ID,
};
pub use property_parameter::PropertyParameterRuntime;
pub use source::{PreparedSourceF32, SOURCE_F32_PARAMETER_PORTS, SourceF32ParameterInstance};
pub use unary::{FixedPlan, PreparedUnaryF32, UNARY_F32_PORTS, UnaryF32Instance};
pub use unary::{UNARY_F32_PARAMETER_PORTS, UnaryF32ParameterInstance};

/// Dependencies used by exported declarative macros.
#[doc(hidden)]
pub mod __private {
    pub use filtergraph_algorithms::AlgorithmPlan;
}

mod sealed {
    pub trait NdarrayElement {}

    impl NdarrayElement for i8 {}
    impl NdarrayElement for u8 {}
    impl NdarrayElement for i16 {}
    impl NdarrayElement for u16 {}
    impl NdarrayElement for i32 {}
    impl NdarrayElement for u32 {}
    impl NdarrayElement for i64 {}
    impl NdarrayElement for u64 {}
    impl NdarrayElement for f32 {}
    impl NdarrayElement for f64 {}
    impl NdarrayElement for super::Bool8 {}
}

/// One byte in an ndarray whose element type is `SPA_ELEMENT_TYPE_BOOL8`.
///
/// Unlike Rust's `bool`, every bit pattern is valid for this storage type.
/// Call [`Self::get`] when consuming an input so values other than zero and one
/// are rejected before they enter a scientific algorithm.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Bool8(u8);

impl Bool8 {
    /// Encode one Boolean value for an ndarray port.
    pub const fn new(value: bool) -> Self {
        Self(value as u8)
    }

    /// Decode one admitted Boolean byte.
    pub const fn get(self) -> Option<bool> {
        match self.0 {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

impl From<bool> for Bool8 {
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}

/// Primitive Rust storage admitted by one packed FGN ndarray element type.
///
/// The trait is sealed because constructing a Rust slice is sound only for
/// types whose complete bit-pattern and alignment contract is known here.
pub trait NdarrayElement: sealed::NdarrayElement + Copy + Send + Sync + Sized + 'static {
    /// Matching `SPA_ELEMENT_TYPE_*` value.
    const ELEMENT_TYPE: u32;
}

macro_rules! ndarray_elements {
    ($($type:ty => $element:expr),+ $(,)?) => {
        $(impl NdarrayElement for $type {
            const ELEMENT_TYPE: u32 = $element;
        })+
    };
}

ndarray_elements! {
    Bool8 => abi::ELEMENT_BOOL8,
    i8 => abi::ELEMENT_I8,
    u8 => abi::ELEMENT_U8,
    i16 => abi::ELEMENT_I16_LE,
    u16 => abi::ELEMENT_U16_LE,
    i32 => abi::ELEMENT_I32_LE,
    u32 => abi::ELEMENT_U32_LE,
    i64 => abi::ELEMENT_I64_LE,
    u64 => abi::ELEMENT_U64_LE,
    f32 => abi::ELEMENT_F32_LE,
    f64 => abi::ELEMENT_F64_LE,
}

/// Instance-owned packed ndarray format whose ABI pointers remain stable.
pub struct NdarrayFormat {
    raw: abi::Format,
    _shape: Box<[u32]>,
    _schema: Option<CString>,
}

// SAFETY: all format storage is immutable after construction.
unsafe impl Send for NdarrayFormat {}
// SAFETY: all format storage is immutable after construction.
unsafe impl Sync for NdarrayFormat {}

impl NdarrayFormat {
    /// Construct one packed ndarray format without a declared rate.
    pub fn packed<T: NdarrayElement>(
        shape: Box<[u32]>,
        layout: u32,
        schema: Option<&str>,
    ) -> Result<Self, PropertyAdapterError> {
        Self::packed_with_rate::<T>(shape, layout, None, schema)
    }

    /// Construct one packed ndarray format with an optional exact rate.
    pub fn packed_with_rate<T: NdarrayElement>(
        shape: Box<[u32]>,
        layout: u32,
        rate: Option<(u32, u32)>,
        schema: Option<&str>,
    ) -> Result<Self, PropertyAdapterError> {
        if shape.is_empty()
            || shape.len() > abi::NDARRAY_MAX_DIMENSIONS
            || shape
                .iter()
                .any(|&extent| extent == 0 || extent > i32::MAX as u32)
            || !matches!(layout, abi::LAYOUT_ROW_MAJOR | abi::LAYOUT_COLUMN_MAJOR)
            || rate.is_some_and(|(numerator, denominator)| numerator == 0 || denominator == 0)
            || schema.is_some_and(str::is_empty)
        {
            return Err(PropertyAdapterError::InvalidFormat);
        }
        let elements = shape.iter().try_fold(1_u64, |product, &extent| {
            product.checked_mul(u64::from(extent))
        });
        if elements
            .and_then(|elements| elements.checked_mul(size_of::<T>() as u64))
            .is_none_or(|bytes| bytes > u64::from(u32::MAX))
        {
            return Err(PropertyAdapterError::InvalidFormat);
        }
        let schema = schema.map(CString::new).transpose()?;
        let (rate_num, rate_denom) = rate.unwrap_or((0, 0));
        let raw = abi::Format {
            element_type: T::ELEMENT_TYPE,
            layout,
            rate_num,
            rate_denom,
            n_dimensions: shape.len() as u32,
            shape: shape.as_ptr(),
            schema: schema.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
        };
        Ok(Self {
            raw,
            _shape: shape,
            _schema: schema,
        })
    }

    /// Construct a packed little-endian F32 ndarray format.
    pub fn f32(
        shape: Box<[u32]>,
        layout: u32,
        schema: Option<&str>,
    ) -> Result<Self, PropertyAdapterError> {
        Self::packed::<f32>(shape, layout, schema)
    }

    /// Return the immutable FGN ABI view.
    pub const fn as_abi(&self) -> &abi::Format {
        &self.raw
    }
}

/// Checked input-buffer access for one FGN process callback.
pub struct InputBuffers<'a> {
    buffers: &'a [abi::Buffer],
}

impl<'a> InputBuffers<'a> {
    /// Return one packed typed input with the required element count.
    pub fn slice<T: NdarrayElement>(&self, index: usize, elements: usize) -> Result<&'a [T], i32> {
        let buffer = self.buffers.get(index).ok_or(abi::EINVAL)?;
        // SAFETY: the callback validates the outer ABI array. This helper
        // validates the nested FGN/SPA pointers, format, chunk, and extent.
        unsafe { typed_slice(buffer, elements) }
    }

    /// Return one packed F32 input with the required element count.
    pub fn f32(&self, index: usize, elements: usize) -> Result<&'a [f32], i32> {
        self.slice(index, elements)
    }

    /// Return whether a declared input has no per-frame buffer.
    ///
    /// FGN parameter inputs are present in the process input array but remain
    /// disconnected after their sparse update has been prepared.
    pub fn is_disconnected(&self, index: usize) -> Result<bool, i32> {
        self.buffers
            .get(index)
            .map(|buffer| buffer.buffer.is_null())
            .ok_or(abi::EINVAL)
    }

    /// Return transport-neutral standard sample metadata for one input.
    pub fn sample_metadata(
        &self,
        index: usize,
    ) -> Result<Option<filtergraph_algorithms::SampleMetadata>, i32> {
        let buffer = self.buffers.get(index).ok_or(abi::EINVAL)?;
        if buffer.buffer.is_null() {
            return Ok(None);
        }
        // SAFETY: the callback lends this complete nested SPA buffer. The
        // helper validates its metadata descriptors and Header storage.
        unsafe { read_sample_metadata(buffer.buffer) }
    }

    /// Return the number of plugin input ports supplied by the host.
    pub const fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Return whether no input ports were supplied.
    pub const fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

/// Checked output-buffer access for one FGN process callback.
pub struct OutputBuffers<'a> {
    buffers: &'a mut [abi::Buffer],
}

impl OutputBuffers<'_> {
    /// Borrow one disjoint packed input/output pair.
    pub fn input_output<'input, 'output, T: NdarrayElement>(
        &'output mut self,
        inputs: &InputBuffers<'input>,
        input: usize,
        output: usize,
        elements: usize,
    ) -> Result<(&'input [T], &'output mut [T]), i32> {
        let input = inputs.buffers.get(input).ok_or(abi::EINVAL)?;
        let output = self.buffers.get_mut(output).ok_or(abi::EINVAL)?;
        // SAFETY: the FGN callback lends both complete buffers. The shared
        // helper validates every region and rejects cross-buffer overlap.
        unsafe { typed_input_output(input, output, elements) }
    }

    /// Return one packed typed output with the required element count.
    pub fn slice_mut<T: NdarrayElement>(
        &mut self,
        index: usize,
        elements: usize,
    ) -> Result<&mut [T], i32> {
        let buffer = self.buffers.get_mut(index).ok_or(abi::EINVAL)?;
        // SAFETY: the callback validates the outer ABI array. This helper
        // validates the nested FGN/SPA pointers, format, and storage extent.
        unsafe { typed_slice_mut(buffer, elements) }
    }

    /// Return one packed F32 output with the required element count.
    pub fn f32_mut(&mut self, index: usize, elements: usize) -> Result<&mut [f32], i32> {
        self.slice_mut(index, elements)
    }

    /// Mark one previously validated typed output complete.
    pub fn finish<T: NdarrayElement>(&mut self, index: usize, elements: usize) -> Result<(), i32> {
        let buffer = self.buffers.get_mut(index).ok_or(abi::EINVAL)?;
        // SAFETY: this revalidates the complete output path before publishing
        // its successful byte count.
        unsafe { finish_typed_output::<T>(buffer, elements) }
    }

    /// Mark one previously validated F32 output complete.
    pub fn finish_f32(&mut self, index: usize, elements: usize) -> Result<(), i32> {
        self.finish::<f32>(index, elements)
    }

    /// Copy matching metadata types from one admitted input to one output.
    pub fn copy_metadata_from(
        &mut self,
        output: usize,
        inputs: &InputBuffers<'_>,
        input: usize,
    ) -> Result<(), i32> {
        let source = inputs.buffers.get(input).ok_or(abi::EINVAL)?;
        let destination = self.buffers.get_mut(output).ok_or(abi::EINVAL)?;
        if source.buffer.is_null() || destination.buffer.is_null() {
            return Err(abi::EINVAL);
        }
        // SAFETY: both nested SPA buffers are borrowed for this callback. The
        // helper validates their metadata before copying disjoint regions.
        unsafe { copy_matching_metadata(source.buffer, destination.buffer) }
    }

    /// Replace standard Header metadata on one output.
    pub fn set_sample_metadata(
        &mut self,
        output: usize,
        metadata: filtergraph_algorithms::SampleMetadata,
    ) -> Result<(), i32> {
        let destination = self.buffers.get_mut(output).ok_or(abi::EINVAL)?;
        if destination.buffer.is_null() {
            return Err(abi::EINVAL);
        }
        // SAFETY: the callback lends this complete mutable SPA buffer. The
        // helper validates metadata before changing the Header value.
        unsafe { write_sample_metadata(destination.buffer, metadata) }
    }

    /// Return the number of plugin output ports supplied by the host.
    pub const fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Return whether no output ports were supplied.
    pub const fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

/// Property callback surface consumed by the shared FGN descriptor adapter.
///
/// Property-enabled FilterGraph plugin instances use [`PropertyRuntime`]. Instances
/// without scalar properties use [`NoProperties`] and do not need to implement
/// FilterGraph's optional `AlgorithmProperties` interface.
pub trait FgnProperties: Send + Sync {
    /// Return one ABI descriptor by enumeration index.
    fn property_info(&self, index: usize) -> Option<abi::PropertyInfo>;

    /// Return one current ABI value by descriptor-local identifier.
    fn property(&self, id: u32) -> Result<abi::Value, i32>;

    /// Return the monotonic externally visible revision.
    fn revision(&self) -> u64;

    /// Prepare one complete property transaction on the serial control path.
    ///
    /// # Safety
    ///
    /// The ABI assignments and result pointer must remain valid for this call.
    unsafe fn prepare(
        &self,
        properties: *const abi::Property,
        n_properties: u32,
        result: *mut *mut c_void,
    ) -> i32;

    /// Publish a token returned by [`Self::prepare`] at graph-cycle start.
    ///
    /// # Safety
    ///
    /// The token must belong to this property owner and be consumed once.
    unsafe fn commit(&self, prepared: *mut c_void);

    /// Discard a token returned by [`Self::prepare`] on the control path.
    ///
    /// # Safety
    ///
    /// The token must belong to this property owner and be consumed once.
    unsafe fn discard(&self, prepared: *mut c_void);
}

impl<P> FgnProperties for PropertyRuntime<P>
where
    P: filtergraph_algorithms::AlgorithmProperties + Send + Sync,
{
    fn property_info(&self, index: usize) -> Option<abi::PropertyInfo> {
        self.schema().info(index)
    }

    fn property(&self, id: u32) -> Result<abi::Value, i32> {
        self.get(id)
    }

    fn revision(&self) -> u64 {
        self.revision()
    }

    unsafe fn prepare(
        &self,
        properties: *const abi::Property,
        n_properties: u32,
        result: *mut *mut c_void,
    ) -> i32 {
        unsafe { self.prepare(properties, n_properties, result) }
    }

    unsafe fn commit(&self, prepared: *mut c_void) {
        unsafe { self.commit(prepared) }
    }

    unsafe fn discard(&self, prepared: *mut c_void) {
        unsafe { self.discard(prepared) }
    }
}

/// Empty FGN property surface for plugin instances with no scalar properties.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoProperties;

impl FgnProperties for NoProperties {
    fn property_info(&self, _index: usize) -> Option<abi::PropertyInfo> {
        None
    }

    fn property(&self, _id: u32) -> Result<abi::Value, i32> {
        Err(-abi::ENOENT)
    }

    fn revision(&self) -> u64 {
        0
    }

    unsafe fn prepare(
        &self,
        _properties: *const abi::Property,
        _n_properties: u32,
        _result: *mut *mut c_void,
    ) -> i32 {
        -abi::ENOENT
    }

    unsafe fn commit(&self, _prepared: *mut c_void) {}

    unsafe fn discard(&self, _prepared: *mut c_void) {}
}

/// Plan owner used by reusable plugin-shape adapters.
///
/// Property-enabled stores publish complete replacement plans. A [`FixedPlan`]
/// owns a plan whose scalar configuration cannot change at runtime.
pub trait PlanStore: FgnProperties {
    /// FilterGraph plan selected for processing.
    type Plan: Send + Sync;

    /// Construct the store from one admitted plan.
    fn from_plan(plan: Self::Plan) -> Result<Self, i32>
    where
        Self: Sized;

    /// Return the complete plan selected for the current process call.
    fn active_plan(&self) -> &Self::Plan;

    /// Adopt any completely published replacement immediately before one
    /// numerical process call.
    fn adopt(&self) -> bool {
        false
    }

    /// Prepare a complete plan replacement from one sparse ndarray parameter.
    ///
    /// Stores without a parameter publication surface return `EINVAL` without
    /// invoking `prepare`.
    fn prepare_parameter_with<E>(
        &self,
        _sequence: Option<u64>,
        _prepare: impl FnOnce(&Self::Plan) -> Result<Self::Plan, E>,
    ) -> Result<*mut c_void, i32> {
        Err(abi::EINVAL)
    }

    /// Publish one parameter token returned by [`Self::prepare_parameter_with`].
    ///
    /// # Safety
    ///
    /// The token must belong to this store and be consumed exactly once.
    unsafe fn commit_parameter(&self, _prepared: *mut c_void) {}

    /// Discard one unpublished parameter token.
    ///
    /// # Safety
    ///
    /// The token must belong to this store and be consumed exactly once.
    unsafe fn discard_parameter(&self, _prepared: *mut c_void) {}
}

impl<P> PlanStore for PropertyRuntime<P>
where
    P: filtergraph_algorithms::AlgorithmProperties + Send + Sync,
{
    type Plan = P;

    fn from_plan(plan: Self::Plan) -> Result<Self, i32> {
        Self::new(plan).map_err(|_| abi::EINVAL)
    }

    fn active_plan(&self) -> &Self::Plan {
        PropertyRuntime::active_plan(self)
    }

    fn adopt(&self) -> bool {
        self.adopt_requested()
    }
}

/// Minimal instance-specific surface needed by the shared FGN adapter.
/// Thread-safe plugin-instance surface consumed by the FGN callback adapter.
///
/// # Safety
///
/// FGN can invoke read-only lifecycle, control, parameter, and data callbacks
/// from different threads. Implementers must make every shared callback safe
/// under that overlap. The host guarantees that `process` and workspace-reset
/// callbacks for one instance do not overlap or reenter. Prefer the declaration
/// macros in this crate: they contain the `UnsafeCell` ownership proof and keep
/// this unsafe contract out of scientist-authored algorithm code.
///
/// A thread-affine instance cannot cross this safe adapter boundary:
///
/// ```compile_fail
/// use std::{ffi::CStr, rc::Rc};
/// use filtergraph_fgn::{InputBuffers, NoProperties, OutputBuffers, PluginInstance, abi};
///
/// struct ThreadAffine(Rc<()>);
///
/// unsafe impl PluginInstance for ThreadAffine {
///     type Properties = NoProperties;
///     const NAME: &'static CStr = c"thread-affine";
///     const PORTS: &'static [abi::PortInfo] = &[];
///     fn instantiate(_: &CStr) -> Result<Self, i32> {
///         Ok(Self(Rc::new(())))
///     }
///     fn port_format(&self, _: u32) -> Option<&abi::Format> { None }
///     fn properties(&self) -> &Self::Properties { &NoProperties }
///     fn process(
///         &self,
///         _: InputBuffers<'_>,
///         _: &mut OutputBuffers<'_>,
///     ) -> Result<(), i32> { Ok(()) }
/// }
/// ```
///
/// Interior mutability without synchronization is rejected independently:
///
/// ```compile_fail
/// use std::{cell::Cell, ffi::CStr};
/// use filtergraph_fgn::{InputBuffers, NoProperties, OutputBuffers, PluginInstance, abi};
///
/// struct Unsynchronized(Cell<u32>);
///
/// unsafe impl PluginInstance for Unsynchronized {
///     type Properties = NoProperties;
///     const NAME: &'static CStr = c"unsynchronized";
///     const PORTS: &'static [abi::PortInfo] = &[];
///     fn instantiate(_: &CStr) -> Result<Self, i32> { Ok(Self(Cell::new(0))) }
///     fn port_format(&self, _: u32) -> Option<&abi::Format> { None }
///     fn properties(&self) -> &Self::Properties { &NoProperties }
///     fn process(
///         &self,
///         _: InputBuffers<'_>,
///         _: &mut OutputBuffers<'_>,
///     ) -> Result<(), i32> { Ok(()) }
/// }
/// ```
pub unsafe trait PluginInstance: Send + Sync + Sized + 'static {
    /// Instance-owned scalar property surface.
    type Properties: FgnProperties;

    /// Stable descriptor label.
    const NAME: &'static CStr;
    /// Static plugin port descriptors.
    const PORTS: &'static [abi::PortInfo];
    /// Whether this instance admits coherent local parameter transactions and
    /// graph-cycle adoption through the ABI batch callbacks.
    const PARAMETER_TRANSACTIONS: bool = false;

    /// Return the number of input ports expected by the process callback.
    fn input_port_count(&self) -> usize {
        Self::PORTS
            .iter()
            .filter(|port| port.direction == abi::DIRECTION_INPUT)
            .count()
    }

    /// Return the number of output ports expected by the process callback.
    fn output_port_count(&self) -> usize {
        Self::PORTS
            .iter()
            .filter(|port| port.direction == abi::DIRECTION_OUTPUT)
            .count()
    }

    /// Construct an instance from plugin-specific configuration.
    ///
    /// Errors are positive errno values.
    fn instantiate(config: &CStr) -> Result<Self, i32>;

    /// Return the prepared instance's scientific requirements for a local port.
    /// Static descriptor layouts cover every admitted construction configuration;
    /// conditional requirements are checked during instantiation and processing.
    fn required_sample_metadata(&self, _port: u32) -> &'static [&'static str] {
        &[]
    }

    /// Return the exact packed format for one local port ID.
    fn port_format(&self, port: u32) -> Option<&abi::Format>;

    /// Return the shared property runtime owned by this instance.
    fn properties(&self) -> &Self::Properties;

    /// Process one complete set of buffers without allocation or blocking.
    ///
    /// Errors are positive errno values.
    fn process(&self, inputs: InputBuffers<'_>, outputs: &mut OutputBuffers<'_>)
    -> Result<(), i32>;

    /// Process with the host-owned executor available to an adapter override.
    ///
    /// Ordinary plugin implementations and scientific algorithms use
    /// [`Self::process`]. Generated deployment adapters override this method
    /// only when they can map a declared operation to a native executor task.
    #[doc(hidden)]
    fn process_with_executor(
        &self,
        _executor: &abi::Executor,
        inputs: InputBuffers<'_>,
        outputs: &mut OutputBuffers<'_>,
    ) -> Result<(), i32> {
        self.process(inputs, outputs)
    }

    /// Activate the instance after graph admission.
    fn activate(&self) -> Result<(), i32> {
        Ok(())
    }

    /// Deactivate the instance after processing stops.
    fn deactivate(&self) -> Result<(), i32> {
        Ok(())
    }

    /// Reset persistent plugin state outside processing.
    fn reset(&self) -> Result<(), i32> {
        Ok(())
    }

    /// Prepare one sparse parameter-port update on the serial worker.
    ///
    /// # Safety
    ///
    /// The buffer and nested pointers are borrowed only for this call.
    unsafe fn prepare_parameter(
        &self,
        _port: u32,
        _buffer: &abi::Buffer,
    ) -> Result<*mut c_void, i32> {
        Err(abi::EINVAL)
    }

    /// Publish one prepared parameter update.
    ///
    /// # Safety
    ///
    /// The token must come from `prepare_parameter()` on this instance.
    unsafe fn commit_parameter(&self, _prepared: *mut c_void) {}

    /// Discard one unpublished parameter update.
    ///
    /// # Safety
    ///
    /// The token must come from `prepare_parameter()` on this instance.
    unsafe fn discard_parameter(&self, _prepared: *mut c_void) {}

    /// Prepare one complete replacement from several unique parameter ports.
    ///
    /// # Safety
    ///
    /// Every buffer and nested pointer is borrowed only for this call.
    unsafe fn prepare_parameters(
        &self,
        _parameters: &[abi::Parameter],
    ) -> Result<*mut c_void, i32> {
        Err(abi::EINVAL)
    }

    /// Adopt a completely published replacement before graph processing.
    fn adopt_parameters(&self) {}
}

/// Construct the static FGN descriptor for one plugin-instance wrapper.
pub const fn descriptor<I: PluginInstance>() -> abi::Descriptor {
    descriptor_with::<I>(I::NAME.as_ptr(), I::PORTS.len() as u32, I::PORTS.as_ptr())
}

/// Construct an FGN descriptor with dynamically owned name and port storage.
#[doc(hidden)]
pub const fn descriptor_with<I: PluginInstance>(
    name: *const c_char,
    n_ports: u32,
    ports: *const abi::PortInfo,
) -> abi::Descriptor {
    abi::Descriptor {
        struct_size: size_of::<abi::Descriptor>() as u32,
        version: abi::ABI_VERSION,
        name,
        n_ports,
        ports,
        instantiate: Some(instantiate::<I>),
        cleanup: Some(cleanup::<I>),
        get_port_format: Some(get_port_format::<I>),
        enum_prop_info: Some(enum_prop_info::<I>),
        get_prop: Some(get_prop::<I>),
        get_prop_revision: Some(get_prop_revision::<I>),
        prepare_props: Some(prepare_props::<I>),
        commit_props: Some(commit_props::<I>),
        discard_props: Some(discard_props::<I>),
        prepare_parameter: Some(prepare_parameter::<I>),
        commit_parameter: Some(commit_parameter::<I>),
        discard_parameter: Some(discard_parameter::<I>),
        prepare_parameters: if I::PARAMETER_TRANSACTIONS {
            Some(prepare_parameters::<I>)
        } else {
            None
        },
        adopt_parameters: if I::PARAMETER_TRANSACTIONS {
            Some(adopt_parameters::<I>)
        } else {
            None
        },
        activate: Some(activate::<I>),
        deactivate: Some(deactivate::<I>),
        reset: Some(reset::<I>),
        process: Some(process::<I>),
        prepare_process_thread: None,
        get_port_metadata: Some(get_port_metadata::<I>),
    }
}

fn negative_errno(error: i32) -> i32 {
    match error {
        1.. => -error,
        ..=-1 => error,
        0 => -abi::EFAULT,
    }
}

fn contained(callback: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(callback)).unwrap_or(-abi::EFAULT)
}

struct AdapterInstance<I> {
    plugin: I,
    executor: *const abi::Executor,
    descriptor: *const abi::Descriptor,
    metadata_requirements: Box<[Box<[abi::SpaMetaRequirement]>]>,
}

// SAFETY: PluginInstance is Send + Sync, and FGN guarantees that the immutable
// executor interface outlives the adapter instance. Its mutable data is used by
// one process coordinator at a time under the host executor contract.
unsafe impl<I: PluginInstance> Send for AdapterInstance<I> {}
// SAFETY: See the Send implementation; control callbacks never mutate the
// executor interface, and only the process owner invokes its task callbacks.
unsafe impl<I: PluginInstance> Sync for AdapterInstance<I> {}

unsafe fn adapter_instance<'a, I: PluginInstance>(
    data: *mut c_void,
) -> Option<&'a AdapterInstance<I>> {
    if data.is_null() {
        None
    } else {
        // SAFETY: FGN passes the Box<AdapterInstance<I>> allocated below.
        Some(unsafe { &*data.cast::<AdapterInstance<I>>() })
    }
}

unsafe fn instance<'a, I: PluginInstance>(data: *mut c_void) -> Option<&'a I> {
    // SAFETY: The same adapter-instance pointer contract applies.
    unsafe { adapter_instance::<I>(data) }.map(|adapter| &adapter.plugin)
}

unsafe extern "C" fn instantiate<I: PluginInstance>(
    descriptor: *const abi::Descriptor,
    config: *const c_char,
    executor: *const abi::Executor,
    result: *mut *mut c_void,
) -> i32 {
    contained(|| {
        if descriptor.is_null()
            || !descriptor.is_aligned()
            || config.is_null()
            || executor.is_null()
            || result.is_null()
        {
            return -abi::EINVAL;
        }
        // SAFETY: FGN lends a complete immutable executor with instance lifetime.
        let executor_ref = unsafe { &*executor };
        if executor_ref.struct_size < size_of::<abi::Executor>() as u32
            || executor_ref.version != abi::EXECUTOR_VERSION
            || executor_ref.n_lanes == 0
            || executor_ref.reserved != 0
            || (executor_ref.run_lanes.is_some() && executor_ref.cache_line_size == 0)
        {
            return -abi::EINVAL;
        }
        // SAFETY: the FGN host supplies a NUL-terminated config for this call.
        let config = unsafe { CStr::from_ptr(config) };
        match I::instantiate(config) {
            Ok(instance) => {
                // SAFETY: the caller lends a complete immutable descriptor and
                // its port/requirement tables for the instance lifetime.
                let metadata_requirements =
                    match unsafe { instance_metadata(&instance, &*descriptor) } {
                        Ok(requirements) => requirements,
                        Err(error) => return negative_errno(error),
                    };
                // SAFETY: result is writable and ownership transfers to FGN.
                unsafe {
                    *result = Box::into_raw(Box::new(AdapterInstance {
                        plugin: instance,
                        executor,
                        descriptor,
                        metadata_requirements,
                    }))
                    .cast();
                }
                0
            }
            Err(error) => negative_errno(error),
        }
    })
}

// Descriptor metadata is immutable and may be a superset for configuration-
// dependent requirements. Validate layouts here; fact presence is data-loop work.
unsafe fn instance_metadata<I: PluginInstance>(
    instance: &I,
    descriptor: &abi::Descriptor,
) -> Result<Box<[Box<[abi::SpaMetaRequirement]>]>, i32> {
    let count = instance
        .input_port_count()
        .checked_add(instance.output_port_count())
        .ok_or(abi::EINVAL)?;
    if count != descriptor.n_ports as usize
        || count > 64
        || (count != 0 && (descriptor.ports.is_null() || !descriptor.ports.is_aligned()))
    {
        return Err(abi::EINVAL);
    }
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let port = unsafe { &*descriptor.ports.add(index) };
        if port.n_metas as usize >= abi::MAX_METAS
            || (port.n_metas != 0 && (port.metas.is_null() || !port.metas.is_aligned()))
        {
            return Err(abi::EINVAL);
        }
        let mut records = if port.n_metas == 0 {
            Vec::new()
        } else {
            unsafe { core::slice::from_raw_parts(port.metas, port.n_metas as usize) }.to_vec()
        };
        let mut bytes = size_of::<abi::SpaMetaHeader>();
        for (i, record) in records.iter().enumerate() {
            if record.type_ == abi::META_HEADER
                || record.reserved != 0
                || record.flags & !abi::META_REQUIREMENT_REQUIRED != 0
                || records[..i].iter().any(|r| r.type_ == record.type_)
            {
                return Err(abi::EINVAL);
            }
            metadata::validate_layout(&abi::SpaMeta {
                type_: record.type_,
                size: record.size,
                data: core::ptr::without_provenance_mut(8),
            })?;
            bytes = bytes
                .checked_add((record.size as usize + 7) & !7)
                .ok_or(abi::EOVERFLOW)?;
            if bytes > abi::MAX_META_BYTES {
                return Err(abi::EINVAL);
            }
        }
        for fact in instance.required_sample_metadata(port.index) {
            let (_, type_) = metadata::required_fact(fact)?;
            if type_ == abi::META_HEADER {
                continue;
            }
            let record = records
                .iter_mut()
                .find(|r| r.type_ == type_ && Some(r.size) == metadata::record_size(type_))
                .ok_or(abi::EINVAL)?;
            record.flags |= abi::META_REQUIREMENT_REQUIRED;
        }
        result.push(records.into_boxed_slice());
    }
    Ok(result.into_boxed_slice())
}

unsafe extern "C" fn get_port_metadata<I: PluginInstance>(
    data: *mut c_void,
    port: u32,
    metas: *mut *const abi::SpaMetaRequirement,
    n_metas: *mut u32,
) -> i32 {
    contained(|| {
        if metas.is_null() || !metas.is_aligned() || n_metas.is_null() || !n_metas.is_aligned() {
            return -abi::EINVAL;
        }
        let Some(adapter) = (unsafe { adapter_instance::<I>(data) }) else {
            return -abi::EINVAL;
        };
        let descriptor = unsafe { &*adapter.descriptor };
        let Some(index) = (0..descriptor.n_ports as usize)
            .find(|&i| unsafe { (*descriptor.ports.add(i)).index == port })
        else {
            return -abi::EINVAL;
        };
        let records = &adapter.metadata_requirements[index];
        // SAFETY: checked caller-owned output pointers receive immutable tables
        // held by this instance through cleanup, including across plan adoption.
        unsafe {
            *metas = if records.is_empty() {
                core::ptr::null()
            } else {
                records.as_ptr()
            };
            *n_metas = records.len() as u32;
        }
        0
    })
}

unsafe extern "C" fn cleanup<I: PluginInstance>(data: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if !data.is_null() {
            // SAFETY: cleanup consumes the adapter Box exactly once after callbacks stop.
            unsafe {
                drop(Box::from_raw(data.cast::<AdapterInstance<I>>()));
            }
        }
    }));
}

unsafe extern "C" fn get_port_format<I: PluginInstance>(
    data: *mut c_void,
    port: u32,
    format: *mut *const abi::Format,
) -> i32 {
    contained(|| {
        let Some(adapter) = (unsafe { adapter_instance::<I>(data) }) else {
            return -abi::EINVAL;
        };
        let instance = &adapter.plugin;
        if format.is_null() {
            return -abi::EINVAL;
        }
        let Some(value) = instance.port_format(port) else {
            return -abi::ENOENT;
        };
        // SAFETY: format was checked and value lives with the instance.
        unsafe {
            *format = ptr::from_ref(value);
        }
        0
    })
}

unsafe extern "C" fn enum_prop_info<I: PluginInstance>(
    data: *mut c_void,
    index: u32,
    info: *mut abi::PropertyInfo,
) -> i32 {
    contained(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return -abi::EINVAL;
        };
        if info.is_null() {
            return -abi::EINVAL;
        }
        let Some(value) = instance.properties().property_info(index as usize) else {
            return 0;
        };
        // SAFETY: info is writable for this callback.
        unsafe {
            *info = value;
        }
        1
    })
}

unsafe extern "C" fn get_prop<I: PluginInstance>(
    data: *mut c_void,
    id: u32,
    value: *mut abi::Value,
) -> i32 {
    contained(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return -abi::EINVAL;
        };
        if value.is_null() {
            return -abi::EINVAL;
        }
        match instance.properties().property(id) {
            Ok(current) => {
                // SAFETY: value is writable for this callback.
                unsafe {
                    *value = current;
                }
                0
            }
            Err(error) => error,
        }
    })
}

unsafe extern "C" fn get_prop_revision<I: PluginInstance>(data: *mut c_void) -> u64 {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return 0;
        };
        instance.properties().revision()
    }))
    .unwrap_or(0)
}

unsafe extern "C" fn prepare_props<I: PluginInstance>(
    data: *mut c_void,
    properties: *const abi::Property,
    n_properties: u32,
    result: *mut *mut c_void,
) -> i32 {
    contained(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return -abi::EINVAL;
        };
        // SAFETY: FGN borrows the ABI assignments and result for this call.
        unsafe {
            instance
                .properties()
                .prepare(properties, n_properties, result)
        }
    })
}

unsafe extern "C" fn commit_props<I: PluginInstance>(data: *mut c_void, prepared: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return;
        };
        // SAFETY: FGN supplies a token returned by prepare_props.
        unsafe {
            instance.properties().commit(prepared);
        }
    }));
}

unsafe extern "C" fn discard_props<I: PluginInstance>(data: *mut c_void, prepared: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return;
        };
        // SAFETY: FGN supplies a token returned by prepare_props.
        unsafe {
            instance.properties().discard(prepared);
        }
    }));
}

unsafe extern "C" fn prepare_parameter<I: PluginInstance>(
    data: *mut c_void,
    port: u32,
    buffer: *const abi::Buffer,
    result: *mut *mut c_void,
) -> i32 {
    contained(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return -abi::EINVAL;
        };
        if buffer.is_null() || result.is_null() {
            return -abi::EINVAL;
        }
        // SAFETY: FGN borrows buffer for this callback.
        match unsafe { instance.prepare_parameter(port, &*buffer) } {
            Ok(prepared) => {
                // SAFETY: result was checked above.
                unsafe {
                    *result = prepared;
                }
                0
            }
            Err(error) => negative_errno(error),
        }
    })
}

unsafe extern "C" fn commit_parameter<I: PluginInstance>(data: *mut c_void, prepared: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return;
        };
        // SAFETY: FGN supplies a token returned by prepare_parameter.
        unsafe {
            instance.commit_parameter(prepared);
        }
    }));
}

unsafe extern "C" fn discard_parameter<I: PluginInstance>(
    data: *mut c_void,
    prepared: *mut c_void,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return;
        };
        // SAFETY: FGN supplies a token returned by prepare_parameter.
        unsafe {
            instance.discard_parameter(prepared);
        }
    }));
}

unsafe extern "C" fn prepare_parameters<I: PluginInstance>(
    data: *mut c_void,
    parameters: *const abi::Parameter,
    n_parameters: u32,
    result: *mut *mut c_void,
) -> i32 {
    contained(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return -abi::EINVAL;
        };
        if parameters.is_null() || n_parameters == 0 || result.is_null() {
            return -abi::EINVAL;
        }
        if n_parameters as usize > abi::MAX_PARAMETER_TRANSACTION_ASSIGNMENTS {
            return -abi::E2BIG;
        }
        // SAFETY: result was checked above and is writable for this callback.
        unsafe {
            *result = ptr::null_mut();
        }
        // SAFETY: FGN lends n_parameters initialized entries for this call.
        let parameters = unsafe { core::slice::from_raw_parts(parameters, n_parameters as usize) };
        // SAFETY: the same callback borrow covers every nested buffer.
        match unsafe { instance.prepare_parameters(parameters) } {
            Ok(prepared) => {
                // SAFETY: result was checked above and is writable.
                unsafe {
                    *result = prepared;
                }
                0
            }
            Err(error) => negative_errno(error),
        }
    })
}

unsafe extern "C" fn adopt_parameters<I: PluginInstance>(data: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return;
        };
        instance.adopt_parameters();
    }));
}

unsafe extern "C" fn activate<I: PluginInstance>(data: *mut c_void) -> i32 {
    contained(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return -abi::EINVAL;
        };
        instance.activate().map_or_else(negative_errno, |()| 0)
    })
}

unsafe extern "C" fn deactivate<I: PluginInstance>(data: *mut c_void) -> i32 {
    contained(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return -abi::EINVAL;
        };
        instance.deactivate().map_or_else(negative_errno, |()| 0)
    })
}

unsafe extern "C" fn reset<I: PluginInstance>(data: *mut c_void) -> i32 {
    contained(|| {
        let Some(instance) = (unsafe { instance::<I>(data) }) else {
            return -abi::EINVAL;
        };
        instance.reset().map_or_else(negative_errno, |()| 0)
    })
}

struct OutputSizeGuard {
    outputs: *mut abi::Buffer,
    count: usize,
    armed: bool,
}

impl OutputSizeGuard {
    unsafe fn new(inputs: &[abi::Buffer], outputs: &mut [abi::Buffer]) -> Result<Self, i32> {
        // Input-only plugins still require full descriptor and extent admission.
        for input in inputs {
            unsafe { callback_buffer_regions(input, false) }?;
        }
        for (output_index, output) in outputs.iter().enumerate() {
            // SAFETY: FGN lends all nested buffer storage for this callback.
            let Some(output_view) = (unsafe { callback_buffer_regions(output, true) })? else {
                continue;
            };
            for input in inputs {
                // SAFETY: FGN lends all nested buffer storage for this callback.
                let Some(input_view) = (unsafe { callback_buffer_regions(input, false) })? else {
                    continue;
                };
                if output_view.regions[..output_view.n_regions]
                    .iter()
                    .any(|output| {
                        input_view.regions[..input_view.n_regions]
                            .iter()
                            .any(|input| regions_overlap(*output, *input))
                    })
                {
                    return Err(abi::EINVAL);
                }
            }
            for previous in &outputs[..output_index] {
                // SAFETY: FGN lends all nested buffer storage for this callback.
                let Some(previous_view) = (unsafe { callback_buffer_regions(previous, true) })?
                else {
                    continue;
                };
                if output_view.regions[..output_view.n_regions]
                    .iter()
                    .any(|output| {
                        previous_view.regions[..previous_view.n_regions]
                            .iter()
                            .any(|previous| regions_overlap(*output, *previous))
                    })
                {
                    return Err(abi::EINVAL);
                }
            }
        }
        let mut guard = Self {
            outputs: outputs.as_mut_ptr(),
            count: outputs.len(),
            armed: true,
        };
        // SAFETY: every connected output chunk was validated above.
        unsafe { guard.clear() };
        Ok(guard)
    }

    unsafe fn clear(&mut self) {
        for index in 0..self.count {
            // SAFETY: outputs points to the complete callback-owned array.
            let output = unsafe { &*self.outputs.add(index) };
            if output.buffer.is_null() {
                continue;
            }
            // SAFETY: construction validated the complete nested chunk path.
            unsafe {
                (*(*(*output.buffer).datas).chunk).size = 0;
            }
        }
    }

    fn complete(mut self) {
        self.armed = false;
    }
}

impl Drop for OutputSizeGuard {
    fn drop(&mut self) {
        if self.armed {
            // SAFETY: the callback still owns the output array and nested
            // chunks while this guard is live. This also runs during unwind.
            unsafe { self.clear() };
        }
    }
}

unsafe extern "C" fn process<I: PluginInstance>(
    data: *mut c_void,
    inputs: *const abi::Buffer,
    n_inputs: u32,
    outputs: *mut abi::Buffer,
    n_outputs: u32,
) -> i32 {
    contained(|| {
        let Some(adapter) = (unsafe { adapter_instance::<I>(data) }) else {
            return -abi::EINVAL;
        };
        let instance = &adapter.plugin;
        if (n_inputs != 0 && inputs.is_null()) || (n_outputs != 0 && outputs.is_null()) {
            return -abi::EINVAL;
        }
        let expected_inputs = instance.input_port_count();
        let expected_outputs = instance.output_port_count();
        if n_inputs as usize != expected_inputs || n_outputs as usize != expected_outputs {
            return -abi::EINVAL;
        }
        if (n_inputs != 0 && !inputs.is_aligned()) || (n_outputs != 0 && !outputs.is_aligned()) {
            return -abi::EINVAL;
        }
        let input_bytes = (n_inputs as usize).checked_mul(size_of::<abi::Buffer>());
        let output_bytes = (n_outputs as usize).checked_mul(size_of::<abi::Buffer>());
        let (Some(input_bytes), Some(output_bytes)) = (input_bytes, output_bytes) else {
            return -abi::EOVERFLOW;
        };
        if raw_ranges_overlap(
            inputs.cast::<u8>(),
            input_bytes,
            outputs.cast::<u8>(),
            output_bytes,
        ) {
            return -abi::EINVAL;
        }
        // SAFETY: non-empty FGN arrays were checked above and remain initialized
        // for this call. Rust slices still require a non-null pointer at length
        // zero, so construct their empty forms directly.
        let inputs = if n_inputs == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(inputs, n_inputs as usize) }
        };
        let outputs = if n_outputs == 0 {
            &mut []
        } else {
            unsafe { core::slice::from_raw_parts_mut(outputs, n_outputs as usize) }
        };
        let output_guard = match unsafe { OutputSizeGuard::new(inputs, outputs) } {
            Ok(guard) => guard,
            Err(error) => return negative_errno(error),
        };
        for output in outputs.iter() {
            if output.buffer.is_null() {
                continue;
            }
            let raw = unsafe { &*output.buffer };
            unsafe { metadata::clear_records(raw.metas, raw.n_metas as usize) };
        }
        // SAFETY: descriptors and ports have static shared-library lifetime.
        let descriptor = unsafe { &*adapter.descriptor };
        let ports = if descriptor.n_ports == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(descriptor.ports, descriptor.n_ports as usize) }
        };
        for (input, (index, _port)) in inputs.iter().zip(
            ports
                .iter()
                .enumerate()
                .filter(|(_, p)| p.direction == abi::DIRECTION_INPUT),
        ) {
            if input.buffer.is_null() {
                continue;
            }
            let raw = unsafe { &*input.buffer };
            if let Err(error) = unsafe {
                metadata::validate_requirements(
                    raw.metas,
                    raw.n_metas as usize,
                    adapter.metadata_requirements[index].as_ptr(),
                    adapter.metadata_requirements[index].len(),
                )
            } {
                return negative_errno(error);
            }
        }
        for input in inputs {
            if input.buffer.is_null() {
                continue;
            }
            let raw = unsafe { &*input.buffer };
            if let Err(error) =
                unsafe { metadata::validate_records(raw.metas, raw.n_metas as usize) }
            {
                return negative_errno(error);
            }
        }
        let inputs = InputBuffers { buffers: inputs };
        let mut outputs = OutputBuffers { buffers: outputs };
        // SAFETY: instantiate validated this pointer and the host keeps the
        // immutable interface live through cleanup.
        let executor = unsafe { &*adapter.executor };
        let result = instance.process_with_executor(executor, inputs, &mut outputs);
        match result {
            Ok(()) => {
                for (output, port) in outputs.buffers.iter().zip(
                    ports
                        .iter()
                        .filter(|p| p.direction == abi::DIRECTION_OUTPUT),
                ) {
                    if output.buffer.is_null() {
                        continue;
                    }
                    let raw = unsafe { &*output.buffer };
                    if unsafe { (*(*raw.datas).chunk).size } == 0 {
                        continue;
                    }
                    if let Err(error) =
                        unsafe { metadata::validate_records(raw.metas, raw.n_metas as usize) }
                            .and_then(|()| unsafe {
                                metadata::validate_requirements(
                                    raw.metas,
                                    raw.n_metas as usize,
                                    port.metas,
                                    port.n_metas as usize,
                                )
                            })
                    {
                        return negative_errno(error);
                    }
                }
                output_guard.complete();
                0
            }
            Err(error) => negative_errno(error),
        }
    })
}

unsafe fn typed_slice<T: NdarrayElement>(
    buffer: &abi::Buffer,
    elements: usize,
) -> Result<&[T], i32> {
    let view = unsafe { typed_buffer_view::<T>(buffer, elements, false) }?;
    Ok(unsafe { core::slice::from_raw_parts(view.pointer.cast_const(), elements) })
}

unsafe fn f32_slice(buffer: &abi::Buffer, elements: usize) -> Result<&[f32], i32> {
    unsafe { typed_slice(buffer, elements) }
}

unsafe fn typed_slice_mut<T: NdarrayElement>(
    buffer: &mut abi::Buffer,
    elements: usize,
) -> Result<&mut [T], i32> {
    let view = unsafe { typed_buffer_view::<T>(buffer, elements, true) }?;
    // SAFETY: typed_buffer_view checked the complete SPA data and chunk path.
    unsafe {
        (*(*buffer.buffer).datas)
            .chunk
            .as_mut()
            .expect("validated output chunk")
            .size = 0;
    }
    Ok(unsafe { core::slice::from_raw_parts_mut(view.pointer, elements) })
}

const MAX_BUFFER_REGIONS: usize = 5 + abi::MAX_METAS;

#[derive(Clone, Copy)]
struct MemoryRegion {
    pointer: *const u8,
    bytes: usize,
}

impl MemoryRegion {
    const EMPTY: Self = Self {
        pointer: ptr::null(),
        bytes: 0,
    };
}

struct TypedBufferView<T> {
    pointer: *mut T,
    bytes: usize,
    regions: [MemoryRegion; MAX_BUFFER_REGIONS],
    n_regions: usize,
}

struct CallbackBufferRegions {
    regions: [MemoryRegion; MAX_BUFFER_REGIONS],
    n_regions: usize,
}

unsafe fn callback_buffer_regions(
    buffer: &abi::Buffer,
    mutable_output: bool,
) -> Result<Option<CallbackBufferRegions>, i32> {
    if buffer.buffer.is_null() {
        return Ok(None);
    }
    if !buffer.buffer.is_aligned() {
        return Err(abi::EINVAL);
    }
    // SAFETY: the FGN caller lends the accessible aligned SPA buffer.
    let spa_buffer = unsafe { &*buffer.buffer };
    if spa_buffer.n_datas != 1
        || spa_buffer.datas.is_null()
        || !spa_buffer.datas.is_aligned()
        || spa_buffer.n_metas as usize > abi::MAX_METAS
        || (spa_buffer.n_metas != 0
            && (spa_buffer.metas.is_null() || !spa_buffer.metas.is_aligned()))
    {
        return Err(abi::ENOBUFS);
    }
    // SAFETY: exactly one aligned data descriptor was admitted.
    let data = unsafe { &*spa_buffer.datas };
    if data.data.is_null() || data.chunk.is_null() || !data.chunk.is_aligned() {
        return Err(abi::ENOBUFS);
    }
    // SAFETY: the chunk is non-null and belongs to this data descriptor.
    let chunk = unsafe { &*data.chunk };
    let offset = chunk.offset as usize;
    if offset > data.maxsize as usize {
        return Err(if mutable_output {
            abi::ENOSPC
        } else {
            abi::EMSGSIZE
        });
    }
    let payload_bytes = if mutable_output {
        data.maxsize as usize - offset
    } else {
        let end = offset
            .checked_add(chunk.size as usize)
            .ok_or(abi::EOVERFLOW)?;
        if end > data.maxsize as usize {
            return Err(abi::EMSGSIZE);
        }
        chunk.size as usize
    };
    // SAFETY: the admitted offset is within the complete data allocation.
    let payload = unsafe { data.data.cast::<u8>().add(offset) };
    let mut regions = [MemoryRegion::EMPTY; MAX_BUFFER_REGIONS];
    let mut n_regions = 0;
    push_region(
        &mut regions,
        &mut n_regions,
        buffer.buffer.cast(),
        size_of::<abi::SpaBuffer>(),
    )?;
    push_region(
        &mut regions,
        &mut n_regions,
        spa_buffer.datas.cast(),
        size_of::<abi::SpaData>(),
    )?;
    push_region(
        &mut regions,
        &mut n_regions,
        data.chunk.cast(),
        size_of::<abi::SpaChunk>(),
    )?;
    if spa_buffer.n_metas != 0 {
        push_region(
            &mut regions,
            &mut n_regions,
            spa_buffer.metas.cast(),
            spa_buffer.n_metas as usize * size_of::<abi::SpaMeta>(),
        )?;
    }
    push_region(&mut regions, &mut n_regions, payload.cast(), payload_bytes)?;
    let mut metadata_bytes = 0_usize;
    for index in 0..spa_buffer.n_metas as usize {
        // SAFETY: the aligned descriptor array contains the admitted count.
        let meta = unsafe { &*spa_buffer.metas.add(index) };
        metadata::validate_layout(meta)?;
        metadata_bytes = metadata_bytes
            .checked_add((meta.size as usize + 7) & !7)
            .filter(|bytes| *bytes <= abi::MAX_META_BYTES)
            .ok_or(abi::EOVERFLOW)?;
        for previous in 0..index {
            // SAFETY: both indices are within the admitted descriptor array.
            if unsafe { (*spa_buffer.metas.add(previous)).type_ } == meta.type_ {
                return Err(abi::EINVAL);
            }
        }
        push_region(
            &mut regions,
            &mut n_regions,
            meta.data.cast(),
            meta.size as usize,
        )?;
    }
    validate_disjoint_regions(&regions[..n_regions])?;
    Ok(Some(CallbackBufferRegions { regions, n_regions }))
}

fn push_region(
    regions: &mut [MemoryRegion; MAX_BUFFER_REGIONS],
    n_regions: &mut usize,
    pointer: *const u8,
    bytes: usize,
) -> Result<(), i32> {
    if bytes == 0 {
        return Ok(());
    }
    if pointer.is_null() || pointer.addr().checked_add(bytes).is_none() {
        return Err(abi::EINVAL);
    }
    let Some(region) = regions.get_mut(*n_regions) else {
        return Err(abi::EOVERFLOW);
    };
    *region = MemoryRegion { pointer, bytes };
    *n_regions += 1;
    Ok(())
}

fn regions_overlap(first: MemoryRegion, second: MemoryRegion) -> bool {
    raw_ranges_overlap(first.pointer, first.bytes, second.pointer, second.bytes)
}

fn validate_disjoint_regions(regions: &[MemoryRegion]) -> Result<(), i32> {
    for (index, first) in regions.iter().enumerate() {
        if regions[..index]
            .iter()
            .any(|second| regions_overlap(*first, *second))
        {
            return Err(abi::EINVAL);
        }
    }
    Ok(())
}

unsafe fn typed_buffer_view<T: NdarrayElement>(
    buffer: &abi::Buffer,
    elements: usize,
    output: bool,
) -> Result<TypedBufferView<T>, i32> {
    if !cfg!(target_endian = "little")
        || buffer.buffer.is_null()
        || !buffer.buffer.is_aligned()
        || buffer.format.is_null()
        || !buffer.format.is_aligned()
        || unsafe { (*buffer.format).element_type } != T::ELEMENT_TYPE
    {
        return Err(abi::EINVAL);
    }
    // SAFETY: pointers were checked above and are borrowed for the callback.
    let spa_buffer = unsafe { &*buffer.buffer };
    if spa_buffer.n_datas != 1
        || spa_buffer.datas.is_null()
        || !spa_buffer.datas.is_aligned()
        || spa_buffer.n_metas as usize > abi::MAX_METAS
        || (spa_buffer.n_metas != 0
            && (spa_buffer.metas.is_null() || !spa_buffer.metas.is_aligned()))
    {
        return Err(abi::ENOBUFS);
    }
    // SAFETY: exactly one aligned data descriptor was admitted.
    let data = unsafe { &*spa_buffer.datas };
    if data.data.is_null() || data.chunk.is_null() || !data.chunk.is_aligned() {
        return Err(abi::ENOBUFS);
    }
    // SAFETY: chunk is non-null and belongs to data.
    let chunk = unsafe { &*data.chunk };
    let bytes = elements.checked_mul(size_of::<T>()).ok_or(abi::EOVERFLOW)?;
    let end = (chunk.offset as usize)
        .checked_add(bytes)
        .ok_or(abi::EOVERFLOW)?;
    let supplied_end = (chunk.offset as usize)
        .checked_add(chunk.size as usize)
        .ok_or(abi::EOVERFLOW)?;
    if !output && (chunk.size as usize) < bytes {
        return Err(abi::EMSGSIZE);
    }
    if end > data.maxsize as usize || supplied_end > data.maxsize as usize {
        return Err(if output { abi::ENOSPC } else { abi::EMSGSIZE });
    }
    // SAFETY: the bounds above cover elements packed from chunk.offset.
    let pointer = unsafe {
        data.data
            .cast::<u8>()
            .add(chunk.offset as usize)
            .cast::<T>()
    };
    if !pointer.is_aligned() {
        return Err(abi::EINVAL);
    }

    let mut regions = [MemoryRegion::EMPTY; MAX_BUFFER_REGIONS];
    let mut n_regions = 0;
    push_region(
        &mut regions,
        &mut n_regions,
        buffer.buffer.cast(),
        size_of::<abi::SpaBuffer>(),
    )?;
    push_region(
        &mut regions,
        &mut n_regions,
        spa_buffer.datas.cast(),
        size_of::<abi::SpaData>(),
    )?;
    push_region(
        &mut regions,
        &mut n_regions,
        data.chunk.cast(),
        size_of::<abi::SpaChunk>(),
    )?;
    if spa_buffer.n_metas != 0 {
        push_region(
            &mut regions,
            &mut n_regions,
            spa_buffer.metas.cast(),
            spa_buffer.n_metas as usize * size_of::<abi::SpaMeta>(),
        )?;
    }
    push_region(&mut regions, &mut n_regions, pointer.cast(), bytes)?;

    let mut metadata_bytes = 0_usize;
    for index in 0..spa_buffer.n_metas as usize {
        // SAFETY: the aligned descriptor array contains the admitted count.
        let meta = unsafe { &*spa_buffer.metas.add(index) };
        metadata::validate_layout(meta)?;
        metadata_bytes = metadata_bytes
            .checked_add((meta.size as usize + 7) & !7)
            .filter(|bytes| *bytes <= abi::MAX_META_BYTES)
            .ok_or(abi::EOVERFLOW)?;
        for previous in 0..index {
            // SAFETY: both indices are within the admitted descriptor array.
            if unsafe { (*spa_buffer.metas.add(previous)).type_ } == meta.type_ {
                return Err(abi::EINVAL);
            }
        }
        push_region(
            &mut regions,
            &mut n_regions,
            meta.data.cast(),
            meta.size as usize,
        )?;
    }
    validate_disjoint_regions(&regions[..n_regions])?;
    Ok(TypedBufferView {
        pointer,
        bytes,
        regions,
        n_regions,
    })
}

pub(crate) unsafe fn f32_input_output<'input, 'output>(
    input: &'input abi::Buffer,
    output: &'output mut abi::Buffer,
    elements: usize,
) -> Result<(&'input [f32], &'output mut [f32]), i32> {
    unsafe { typed_input_output(input, output, elements) }
}

unsafe fn typed_input_output<'input, 'output, T: NdarrayElement>(
    input: &'input abi::Buffer,
    output: &'output mut abi::Buffer,
    elements: usize,
) -> Result<(&'input [T], &'output mut [T]), i32> {
    let input_view = unsafe { typed_buffer_view::<T>(input, elements, false) }?;
    let output_view = unsafe { typed_buffer_view::<T>(output, elements, true) }?;
    if input_view.regions[..input_view.n_regions]
        .iter()
        .any(|input| {
            output_view.regions[..output_view.n_regions]
                .iter()
                .any(|output| regions_overlap(*input, *output))
        })
    {
        return Err(abi::EINVAL);
    }
    // SAFETY: the output chunk was validated above. It remains incomplete
    // until the plugin instance reports success and calls finish_f32_output.
    unsafe {
        (*(*output.buffer).datas)
            .chunk
            .as_mut()
            .expect("validated output chunk")
            .size = 0;
    }
    // SAFETY: both complete regions were validated and do not overlap.
    let input = unsafe { core::slice::from_raw_parts(input_view.pointer.cast_const(), elements) };
    let output = unsafe { core::slice::from_raw_parts_mut(output_view.pointer, elements) };
    Ok((input, output))
}

pub(crate) unsafe fn finish_f32_output(
    output: &mut abi::Buffer,
    elements: usize,
) -> Result<(), i32> {
    unsafe { finish_typed_output::<f32>(output, elements) }
}

unsafe fn finish_typed_output<T: NdarrayElement>(
    output: &mut abi::Buffer,
    elements: usize,
) -> Result<(), i32> {
    let view = unsafe { typed_buffer_view::<T>(output, elements, true) }?;
    // SAFETY: typed_buffer_view validated the nested output data and chunk.
    unsafe {
        (*(*output.buffer).datas)
            .chunk
            .as_mut()
            .expect("validated output chunk")
            .size = view.bytes as u32;
    }
    Ok(())
}

pub(crate) unsafe fn copy_matching_metadata(
    input: *const abi::SpaBuffer,
    output: *mut abi::SpaBuffer,
) -> Result<(), i32> {
    let input_view = unsafe { metadata_view(input.cast_mut()) }?;
    unsafe { metadata::validate_records(input_view.metas, input_view.n_metas) }?;
    let output_view = unsafe { metadata_view(output) }?;
    if input_view.regions[..input_view.n_regions]
        .iter()
        .any(|input| {
            output_view.regions[..output_view.n_regions]
                .iter()
                .any(|output| regions_overlap(*input, *output))
        })
    {
        return Err(abi::EINVAL);
    }
    // Preflight every input fact before changing any destination byte.
    for input_index in 0..input_view.n_metas {
        let source = unsafe { &*input_view.metas.add(input_index) };
        if metadata::transport_owned(source.type_) {
            continue;
        }
        let destination = (0..output_view.n_metas)
            .map(|i| unsafe { &*output_view.metas.add(i) })
            .find(|m| m.type_ == source.type_);
        let Some(destination) = destination else {
            // The validated zero representation of an optional native record
            // carries no sample fact. A destination may omit its storage.
            if unsafe { metadata::is_absent_native_record(source) } {
                continue;
            }
            return Err(abi::ENOENT);
        };
        if source.size != destination.size {
            return Err(abi::EINVAL);
        }
    }
    unsafe { metadata::clear_records(output_view.metas, output_view.n_metas) };
    for input_index in 0..input_view.n_metas {
        let source = unsafe { &*input_view.metas.add(input_index) };
        if metadata::transport_owned(source.type_) {
            continue;
        }
        let Some(destination) = (0..output_view.n_metas)
            .map(|i| unsafe { &*output_view.metas.add(i) })
            .find(|m| m.type_ == source.type_)
        else {
            continue; // Preflight established that this native record is absent.
        };
        if source.size != 0 {
            unsafe {
                ptr::copy_nonoverlapping(
                    source.data.cast::<u8>(),
                    destination.data.cast::<u8>(),
                    source.size as usize,
                )
            };
        }
    }
    Ok(())
}

struct MetadataView {
    metas: *mut abi::SpaMeta,
    n_metas: usize,
    regions: [MemoryRegion; MAX_BUFFER_REGIONS],
    n_regions: usize,
}

unsafe fn metadata_view(buffer: *mut abi::SpaBuffer) -> Result<MetadataView, i32> {
    if buffer.is_null() || !buffer.is_aligned() {
        return Err(abi::EINVAL);
    }
    // SAFETY: the caller promises an accessible aligned SPA buffer.
    let spa_buffer = unsafe { &*buffer };
    if spa_buffer.n_metas as usize > abi::MAX_METAS
        || (spa_buffer.n_metas != 0
            && (spa_buffer.metas.is_null() || !spa_buffer.metas.is_aligned()))
    {
        return Err(abi::EINVAL);
    }
    let mut regions = [MemoryRegion::EMPTY; MAX_BUFFER_REGIONS];
    let mut n_regions = 0;
    push_region(
        &mut regions,
        &mut n_regions,
        buffer.cast(),
        size_of::<abi::SpaBuffer>(),
    )?;
    if spa_buffer.n_metas != 0 {
        push_region(
            &mut regions,
            &mut n_regions,
            spa_buffer.metas.cast(),
            spa_buffer.n_metas as usize * size_of::<abi::SpaMeta>(),
        )?;
    }
    let mut metadata_bytes = 0_usize;
    for index in 0..spa_buffer.n_metas as usize {
        // SAFETY: the aligned array contains the admitted count.
        let meta = unsafe { &*spa_buffer.metas.add(index) };
        metadata::validate_layout(meta)?;
        metadata_bytes = metadata_bytes
            .checked_add((meta.size as usize + 7) & !7)
            .filter(|bytes| *bytes <= abi::MAX_META_BYTES)
            .ok_or(abi::EOVERFLOW)?;
        for previous in 0..index {
            // SAFETY: both indices are within the admitted array.
            if unsafe { (*spa_buffer.metas.add(previous)).type_ } == meta.type_ {
                return Err(abi::EINVAL);
            }
        }
        push_region(
            &mut regions,
            &mut n_regions,
            meta.data.cast(),
            meta.size as usize,
        )?;
    }
    validate_disjoint_regions(&regions[..n_regions])?;
    Ok(MetadataView {
        metas: spa_buffer.metas,
        n_metas: spa_buffer.n_metas as usize,
        regions,
        n_regions,
    })
}

pub(crate) unsafe fn header_sequence(buffer: *mut abi::SpaBuffer) -> Result<Option<u64>, i32> {
    let view = unsafe { metadata_view(buffer) }?;
    for index in 0..view.n_metas {
        // SAFETY: metadata_view admitted this descriptor array and count.
        let meta = unsafe { &*view.metas.add(index) };
        if meta.type_ == abi::META_HEADER {
            let header = meta.data.cast::<abi::SpaMetaHeader>();
            // SAFETY: metadata_view checked size, alignment, and storage.
            return Ok(Some(unsafe { (*header).seq }));
        }
    }
    Ok(None)
}

unsafe fn read_sample_metadata(
    buffer: *mut abi::SpaBuffer,
) -> Result<Option<filtergraph_algorithms::SampleMetadata>, i32> {
    let view = unsafe { metadata_view(buffer) }?;
    unsafe { metadata::read_sample(view.metas, view.n_metas) }
}

unsafe fn write_sample_metadata(
    buffer: *mut abi::SpaBuffer,
    metadata: filtergraph_algorithms::SampleMetadata,
) -> Result<(), i32> {
    let view = unsafe { metadata_view(buffer) }?;
    unsafe { metadata::validate_publication(view.metas, view.n_metas, metadata) }?;
    for index in 0..view.n_metas {
        // SAFETY: metadata_view admitted this descriptor array and count.
        let meta = unsafe { &*view.metas.add(index) };
        if meta.type_ != abi::META_HEADER {
            continue;
        }
        let mut flags = 0;
        if metadata.is_discontinuous() {
            flags |= abi::META_HEADER_DISCONT;
        }
        if metadata.is_corrupted() {
            flags |= abi::META_HEADER_CORRUPTED;
        }
        if metadata.is_terminal() {
            flags |= abi::META_HEADER_MARKER;
        }
        // SAFETY: metadata_view checked Header size, alignment, and mutable
        // storage. This callback owns output metadata for the call.
        let header = unsafe { &mut *meta.data.cast::<abi::SpaMetaHeader>() };
        *header = abi::SpaMetaHeader {
            flags,
            offset: metadata.offset(),
            pts: metadata
                .timestamp_nanoseconds()
                .unwrap_or(abi::TIME_INVALID),
            dts_offset: 0,
            seq: metadata.sequence(),
        };
        unsafe { metadata::write_sample(view.metas, view.n_metas, metadata) };
        return Ok(());
    }
    Err(abi::ENOENT)
}

fn raw_ranges_overlap(
    first: *const u8,
    first_bytes: usize,
    second: *const u8,
    second_bytes: usize,
) -> bool {
    if first_bytes == 0 || second_bytes == 0 {
        return false;
    }
    let first_start = first.addr();
    let second_start = second.addr();
    let Some(first_end) = first_start.checked_add(first_bytes) else {
        return true;
    };
    let Some(second_end) = second_start.checked_add(second_bytes) else {
        return true;
    };
    first_start < second_end && second_start < first_end
}

/// Validate one static plugin-descriptor registry.
///
/// # Safety
///
/// Every descriptor name must point to a NUL-terminated string that remains
/// valid for the complete registry lifetime.
#[doc(hidden)]
pub unsafe fn registry_is_valid(descriptors: &[abi::Descriptor]) -> bool {
    for (index, descriptor) in descriptors.iter().enumerate() {
        if descriptor.name.is_null() {
            return false;
        }
        // SAFETY: required by this function's contract.
        let name = unsafe { CStr::from_ptr(descriptor.name) };
        if name.is_empty() || name.to_bytes().len() > abi::LOCAL_NAME_MAX {
            return false;
        }
        for other in &descriptors[..index] {
            // SAFETY: required by this function's contract.
            if name == unsafe { CStr::from_ptr(other.name) } {
                return false;
            }
        }
    }
    true
}

/// Validate a plugin-family name and its descriptor registry.
///
/// # Safety
///
/// Every descriptor name must satisfy [`registry_is_valid`].
#[doc(hidden)]
pub unsafe fn plugin_registry_is_valid(
    plugin_name: &CStr,
    descriptors: &[abi::Descriptor],
) -> bool {
    !plugin_name.is_empty() && unsafe { registry_is_valid(descriptors) }
}

/// Resolve one exact descriptor label from a validated static registry.
///
/// # Safety
///
/// Every descriptor name must satisfy [`registry_is_valid`].
#[doc(hidden)]
pub unsafe fn registry_find_descriptor(
    descriptors: &'static [abi::Descriptor],
    name: &CStr,
) -> *const abi::Descriptor {
    for descriptor in descriptors {
        // SAFETY: required by this function's contract.
        if name == unsafe { CStr::from_ptr(descriptor.name) } {
            return ptr::from_ref(descriptor);
        }
    }
    ptr::null()
}

/// Export a multi-descriptor FilterGraph FGN plugin without handwritten callbacks.
///
/// The registry must contain at least one plugin-instance type. Duplicate or
/// empty descriptor names make the plugin interface unavailable.
#[macro_export]
macro_rules! export_plugin {
    ($plugin_name:expr, [$($instance:ty),+ $(,)?] $(,)?) => {
        static FILTERGRAPH_FGN_DESCRIPTORS: &[$crate::abi::Descriptor] = &[
            $($crate::descriptor::<$instance>()),+
        ];

        unsafe extern "C" fn filtergraph_fgn_find_descriptor(
            name: *const core::ffi::c_char,
        ) -> *const $crate::abi::Descriptor {
            if name.is_null() {
                return core::ptr::null();
            }
            // SAFETY: every descriptor was generated from a PluginInstance's
            // static CStr name.
            if !unsafe { $crate::registry_is_valid(FILTERGRAPH_FGN_DESCRIPTORS) } {
                return core::ptr::null();
            }
            // SAFETY: FGN supplies a NUL-terminated descriptor name.
            let name = unsafe { core::ffi::CStr::from_ptr(name) };
            // SAFETY: the registry was validated above.
            unsafe { $crate::registry_find_descriptor(FILTERGRAPH_FGN_DESCRIPTORS, name) }
        }

        static FILTERGRAPH_FGN_PLUGIN: $crate::abi::Plugin = $crate::abi::Plugin {
            struct_size: core::mem::size_of::<$crate::abi::Plugin>() as u32,
            abi_version: $crate::abi::ABI_VERSION,
            name: $plugin_name.as_ptr(),
            find_descriptor: Some(filtergraph_fgn_find_descriptor),
            flags: 0,
        };

        #[doc = "Return the descriptors in this Rust plugin registry."]
        #[allow(dead_code)]
        pub fn filtergraph_fgn_registered_descriptors(
        ) -> &'static [$crate::abi::Descriptor] {
            FILTERGRAPH_FGN_DESCRIPTORS
        }

        #[doc = "Return the FilterGraph FGN plugin interface for the requested ABI version."]
        #[unsafe(no_mangle)]
        pub extern "C" fn spa_filter_graph_ndarray_plugin_get_interface(
            abi_version: u32,
        ) -> *const $crate::abi::Plugin {
            // SAFETY: every descriptor was generated from a PluginInstance's
            // static CStr name.
            let registry_valid = unsafe {
                $crate::plugin_registry_is_valid($plugin_name, FILTERGRAPH_FGN_DESCRIPTORS)
            };
            if abi_version == $crate::abi::ABI_VERSION && registry_valid {
                core::ptr::from_ref(&FILTERGRAPH_FGN_PLUGIN)
            } else {
                core::ptr::null()
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(panic = "unwind")]
    use filtergraph_algorithms::LeakyIntegratorPlan;

    unsafe extern "C" fn test_run_dense_f32(
        _data: *mut c_void,
        _task: *const abi::DenseF32Task,
    ) -> i32 {
        -abi::EINVAL
    }

    fn test_executor() -> abi::Executor {
        abi::Executor {
            struct_size: size_of::<abi::Executor>() as u32,
            version: abi::EXECUTOR_VERSION,
            flags: 0,
            n_lanes: 1,
            data: ptr::null_mut(),
            run_dense_f32: Some(test_run_dense_f32),
            run_lanes: None,
            cache_line_size: 64,
            reserved: 0,
        }
    }

    #[cfg(panic = "unwind")]
    struct PanickingPlugin {
        properties: PropertyRuntime<LeakyIntegratorPlan<f32>>,
    }

    struct EmptyPlugin {
        properties: NoProperties,
    }

    struct SecondEmptyPlugin {
        properties: NoProperties,
    }

    struct EmptyNamePlugin {
        properties: NoProperties,
    }

    unsafe impl PluginInstance for EmptyPlugin {
        type Properties = NoProperties;

        const NAME: &'static CStr = c"empty-plugin";
        const PORTS: &'static [abi::PortInfo] = &[];

        fn instantiate(_config: &CStr) -> Result<Self, i32> {
            Ok(Self {
                properties: NoProperties,
            })
        }

        fn port_format(&self, _port: u32) -> Option<&abi::Format> {
            None
        }

        fn properties(&self) -> &Self::Properties {
            &self.properties
        }

        fn process(
            &self,
            _inputs: InputBuffers<'_>,
            _outputs: &mut OutputBuffers<'_>,
        ) -> Result<(), i32> {
            Ok(())
        }
    }

    unsafe impl PluginInstance for SecondEmptyPlugin {
        type Properties = NoProperties;

        const NAME: &'static CStr = c"second-empty-plugin";
        const PORTS: &'static [abi::PortInfo] = &[];

        fn instantiate(_config: &CStr) -> Result<Self, i32> {
            Ok(Self {
                properties: NoProperties,
            })
        }

        fn port_format(&self, _port: u32) -> Option<&abi::Format> {
            None
        }

        fn properties(&self) -> &Self::Properties {
            &self.properties
        }

        fn process(
            &self,
            _inputs: InputBuffers<'_>,
            _outputs: &mut OutputBuffers<'_>,
        ) -> Result<(), i32> {
            Ok(())
        }
    }

    unsafe impl PluginInstance for EmptyNamePlugin {
        type Properties = NoProperties;

        const NAME: &'static CStr = c"";
        const PORTS: &'static [abi::PortInfo] = &[];

        fn instantiate(_config: &CStr) -> Result<Self, i32> {
            Ok(Self {
                properties: NoProperties,
            })
        }

        fn port_format(&self, _port: u32) -> Option<&abi::Format> {
            None
        }

        fn properties(&self) -> &Self::Properties {
            &self.properties
        }

        fn process(
            &self,
            _inputs: InputBuffers<'_>,
            _outputs: &mut OutputBuffers<'_>,
        ) -> Result<(), i32> {
            Ok(())
        }
    }

    mod invalid_export {
        use super::EmptyPlugin;

        crate::export_plugin!(c"duplicate-test", [EmptyPlugin, EmptyPlugin]);
    }

    #[cfg(panic = "unwind")]
    unsafe impl PluginInstance for PanickingPlugin {
        type Properties = PropertyRuntime<LeakyIntegratorPlan<f32>>;

        const NAME: &'static CStr = c"panicking-plugin";
        const PORTS: &'static [abi::PortInfo] = &[];

        fn instantiate(_config: &CStr) -> Result<Self, i32> {
            Ok(Self {
                properties: PropertyRuntime::new(LeakyIntegratorPlan::new(1.0_f32, 0.0).unwrap())
                    .unwrap(),
            })
        }

        fn port_format(&self, _port: u32) -> Option<&abi::Format> {
            None
        }

        fn properties(&self) -> &Self::Properties {
            &self.properties
        }

        fn process(
            &self,
            _inputs: InputBuffers<'_>,
            _outputs: &mut OutputBuffers<'_>,
        ) -> Result<(), i32> {
            panic!("callback panic must not cross the C ABI")
        }
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn process_panic_is_contained_at_the_c_boundary() {
        let descriptor = descriptor::<PanickingPlugin>();
        let executor = test_executor();
        let mut instance = ptr::null_mut();
        // SAFETY: the descriptor callbacks receive valid local pointers and
        // cleanup consumes the instance exactly once.
        unsafe {
            assert_eq!(
                descriptor.instantiate.unwrap()(
                    &descriptor,
                    c"{}".as_ptr(),
                    &executor,
                    &mut instance,
                ),
                0
            );
            assert_eq!(
                descriptor.process.unwrap()(instance, ptr::null(), 0, ptr::null_mut(), 0),
                -abi::EFAULT
            );
            descriptor.cleanup.unwrap()(instance);
        }
    }

    #[test]
    fn plugin_without_properties_needs_no_algorithm_property_implementation() {
        let descriptor = descriptor::<EmptyPlugin>();
        let executor = test_executor();
        let mut instance = ptr::null_mut();
        // SAFETY: the descriptor callbacks receive valid local pointers and
        // cleanup consumes the instance exactly once.
        unsafe {
            assert_eq!(
                descriptor.instantiate.unwrap()(
                    &descriptor,
                    c"{}".as_ptr(),
                    &executor,
                    &mut instance,
                ),
                0
            );
            let mut info = core::mem::MaybeUninit::<abi::PropertyInfo>::uninit();
            assert_eq!(
                descriptor.enum_prop_info.unwrap()(instance, 0, info.as_mut_ptr()),
                0
            );
            assert_eq!(
                descriptor.process.unwrap()(instance, ptr::null(), 0, ptr::null_mut(), 0),
                0
            );
            descriptor.cleanup.unwrap()(instance);
        }
    }

    #[test]
    fn descriptor_registry_requires_unique_nonempty_labels_and_exact_lookup() {
        static VALID: &[abi::Descriptor] = &[
            descriptor::<EmptyPlugin>(),
            descriptor::<SecondEmptyPlugin>(),
        ];
        static DUPLICATE: &[abi::Descriptor] =
            &[descriptor::<EmptyPlugin>(), descriptor::<EmptyPlugin>()];
        static EMPTY: &[abi::Descriptor] = &[descriptor::<EmptyNamePlugin>()];
        let mut long_name = [b'x'; abi::LOCAL_NAME_MAX + 2];
        long_name[abi::LOCAL_NAME_MAX + 1] = 0;
        let mut long_descriptor = descriptor::<EmptyPlugin>();
        long_descriptor.name = long_name.as_ptr().cast();

        // SAFETY: these descriptors use static PluginInstance names.
        unsafe {
            assert!(registry_is_valid(VALID));
            assert!(!registry_is_valid(DUPLICATE));
            assert!(!registry_is_valid(EMPTY));
            assert!(!registry_is_valid(core::slice::from_ref(&long_descriptor)));
            assert!(!plugin_registry_is_valid(c"", VALID));
            assert!(plugin_registry_is_valid(c"valid-family", VALID));
            assert_eq!(
                registry_find_descriptor(VALID, c"empty-plugin"),
                ptr::from_ref(&VALID[0])
            );
            assert_eq!(
                registry_find_descriptor(VALID, c"second-empty-plugin"),
                ptr::from_ref(&VALID[1])
            );
            assert!(registry_find_descriptor(VALID, c"unknown-plugin").is_null());
        }
        assert!(
            invalid_export::spa_filter_graph_ndarray_plugin_get_interface(abi::ABI_VERSION)
                .is_null()
        );
    }
}

#[cfg(test)]
mod metadata_copy_regression {
    use super::*;
    #[test]
    fn unmatched_owned_output_slot_is_cleared() {
        let mut src = [17_u64];
        let mut dst = [99_u64];
        let mut stale = [88_u64];
        let mut input_metas = [abi::SpaMeta {
            type_: 0x777,
            size: 8,
            data: src.as_mut_ptr().cast(),
        }];
        let mut output_metas = [
            abi::SpaMeta {
                type_: 0x777,
                size: 8,
                data: dst.as_mut_ptr().cast(),
            },
            abi::SpaMeta {
                type_: 0x778,
                size: 8,
                data: stale.as_mut_ptr().cast(),
            },
        ];
        let input = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 0,
            metas: input_metas.as_mut_ptr(),
            datas: core::ptr::null_mut(),
        };
        let mut output = abi::SpaBuffer {
            n_metas: 2,
            n_datas: 0,
            metas: output_metas.as_mut_ptr(),
            datas: core::ptr::null_mut(),
        };
        unsafe {
            copy_matching_metadata(&input, &mut output).unwrap();
        }
        assert_eq!(dst, [17]);
        assert_eq!(stale, [0]);
    }
}
