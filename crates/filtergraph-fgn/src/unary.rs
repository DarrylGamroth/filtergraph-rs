//! Reusable deployment for one-input, one-output F32 vector algorithms.

use core::cell::UnsafeCell;
use core::ffi::CStr;

use serde::de::DeserializeOwned;

use crate::{
    FgnProperties, InputBuffers, NdarrayFormat, NoProperties, OutputBuffers, ParameterRuntime,
    PlanStore, abi,
};

/// Fixed plan and empty property surface for an algorithm without scalar
/// runtime controls.
pub struct FixedPlan<P> {
    plan: P,
    properties: NoProperties,
}

impl<P> FgnProperties for FixedPlan<P>
where
    P: Send + Sync,
{
    fn property_info(&self, index: usize) -> Option<abi::PropertyInfo> {
        self.properties.property_info(index)
    }

    fn property(&self, id: u32) -> Result<abi::Value, i32> {
        self.properties.property(id)
    }

    fn revision(&self) -> u64 {
        self.properties.revision()
    }

    unsafe fn prepare(
        &self,
        properties: *const abi::Property,
        n_properties: u32,
        result: *mut *mut core::ffi::c_void,
    ) -> i32 {
        unsafe { self.properties.prepare(properties, n_properties, result) }
    }

    unsafe fn commit(&self, prepared: *mut core::ffi::c_void) {
        unsafe { self.properties.commit(prepared) }
    }

    unsafe fn discard(&self, prepared: *mut core::ffi::c_void) {
        unsafe { self.properties.discard(prepared) }
    }
}

impl<P> PlanStore for FixedPlan<P>
where
    P: Send + Sync,
{
    type Plan = P;

    fn from_plan(plan: Self::Plan) -> Result<Self, i32> {
        Ok(Self {
            plan,
            properties: NoProperties,
        })
    }

    fn active_plan(&self) -> &Self::Plan {
        &self.plan
    }
}

/// Prepared, transport-independent contents of a unary F32 plugin instance.
pub struct PreparedUnaryF32<P, W> {
    plan: P,
    workspace: W,
    extent: usize,
    rate: Option<(u32, u32)>,
}

impl<P, W> PreparedUnaryF32<P, W> {
    /// Combine an admitted FilterGraph plan, its workspace, and vector extent.
    pub const fn new(plan: P, workspace: W, extent: usize) -> Self {
        Self {
            plan,
            workspace,
            extent,
            rate: None,
        }
    }

    /// Attach the exact stream rate projected by both data ports.
    pub fn with_rate(mut self, rate: Option<(u32, u32)>) -> Self {
        self.rate = rate;
        self
    }
}

/// Typed cold-path preparation function for one unary F32 instance.
pub type UnaryF32Prepare<C, P, W, E> = fn(C) -> Result<PreparedUnaryF32<P, W>, E>;

/// Typed cold-path replacement-plan function for one F32 parameter vector.
pub type UnaryF32ParameterPrepare<P, E> = fn(&P, &[f32]) -> Result<P, E>;

/// Shared FGN-owned instance for a unary F32 FilterGraph algorithm.
pub struct UnaryF32Instance<S, W>
where
    S: PlanStore,
{
    format: NdarrayFormat,
    extent: usize,
    plans: S,
    workspace: UnsafeCell<W>,
}

// SAFETY: FGN serializes `process` and reset for one instance. Other callbacks
// access only immutable formats and the thread-safe plan store; the data-loop
// owner has exclusive access to the workspace for each such callback.
unsafe impl<S, W> Sync for UnaryF32Instance<S, W>
where
    S: PlanStore,
    W: Send,
{
}

impl<S, W> UnaryF32Instance<S, W>
where
    S: PlanStore,
{
    /// Decode typed JSON configuration and prepare one instance off the data
    /// loop. Algorithm preparation errors map to invalid configuration.
    pub fn from_json<C, E>(
        config: &CStr,
        schema: &str,
        prepare: UnaryF32Prepare<C, S::Plan, W, E>,
    ) -> Result<Self, i32>
    where
        C: DeserializeOwned,
    {
        let config = config.to_str().map_err(|_| abi::EINVAL)?;
        let config = serde_json::from_str(config).map_err(|_| abi::EINVAL)?;
        let prepared = prepare(config).map_err(|_| abi::EINVAL)?;
        let elements = u32::try_from(prepared.extent).map_err(|_| abi::EOVERFLOW)?;
        let format = NdarrayFormat::packed_with_rate::<f32>(
            Box::new([elements]),
            abi::LAYOUT_ROW_MAJOR,
            prepared.rate,
            Some(schema),
        )
        .map_err(|_| abi::EINVAL)?;
        let plans = S::from_plan(prepared.plan)?;
        Ok(Self {
            format,
            extent: prepared.extent,
            plans,
            workspace: UnsafeCell::new(prepared.workspace),
        })
    }

    /// Return the common input/output format for one local port ID.
    pub fn port_format(&self, port: u32) -> Option<&abi::Format> {
        (port <= 1).then(|| self.format.as_abi())
    }

    /// Return the instance-owned plan and property store.
    pub const fn plans(&self) -> &S {
        &self.plans
    }

    /// Validate one input and output buffer, then run the typed numerical call.
    ///
    /// # Safety
    ///
    /// One data-loop owner must call this method without overlap or reentry.
    pub unsafe fn process_with<E>(
        &self,
        inputs: InputBuffers<'_>,
        outputs: &mut OutputBuffers<'_>,
        process: impl FnOnce(&S::Plan, &mut [f32], &mut W, &[f32]) -> Result<(), E>,
    ) -> Result<(), i32> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(abi::EINVAL);
        }
        let input_buffer = &inputs.buffers[0];
        let output_buffer = &mut outputs.buffers[0];
        // SAFETY: the FGN host lends complete buffers to its sole data-loop
        // owner. The shared helper also rejects overlapping storage.
        let (input, output) =
            unsafe { crate::f32_input_output(input_buffer, output_buffer, self.extent) }?;
        self.plans.adopt();
        // SAFETY: the FGN lifecycle supplies one data-loop owner.
        let workspace = unsafe { &mut *self.workspace.get() };
        process(self.plans.active_plan(), output, workspace, input).map_err(|_| abi::EINVAL)?;
        // SAFETY: the buffers validated above remain lent for this call.
        unsafe {
            crate::finish_f32_output(output_buffer, self.extent)?;
            crate::copy_matching_metadata(input_buffer.buffer, output_buffer.buffer)?;
        }
        Ok(())
    }

    /// Mutate workspace during serialized lifecycle reset.
    ///
    /// # Safety
    ///
    /// Reset must not overlap or reenter processing.
    pub unsafe fn reset_with(&self, reset: impl FnOnce(&mut W)) {
        // SAFETY: reset cannot overlap processing under the FGN lifecycle.
        reset(unsafe { &mut *self.workspace.get() });
    }
}

/// Static ports shared by unary F32 plugin declarations.
#[doc(hidden)]
pub static UNARY_F32_PORTS: [abi::PortInfo; 2] = [
    abi::PortInfo::new(0, abi::DIRECTION_INPUT, 0, c"in".as_ptr()),
    abi::PortInfo::new(1, abi::DIRECTION_OUTPUT, 0, c"out".as_ptr()),
];

/// Shared FGN-owned instance for unary F32 processing with one sparse F32
/// parameter vector.
pub struct UnaryF32ParameterInstance<P, W> {
    data_format: NdarrayFormat,
    parameter_format: NdarrayFormat,
    extent: usize,
    plans: ParameterRuntime<P>,
    workspace: UnsafeCell<W>,
}

// SAFETY: FGN serializes `process` and reset for one instance. Parameter
// publication is owned by ParameterRuntime, while the single data-loop owner
// has exclusive access to the workspace.
unsafe impl<P, W> Sync for UnaryF32ParameterInstance<P, W>
where
    P: Send + Sync,
    W: Send,
{
}

impl<P, W> UnaryF32ParameterInstance<P, W>
where
    P: Send + Sync,
{
    /// Decode initial typed configuration and prepare one parameterized
    /// plugin instance off the data loop.
    pub fn from_json<C, E>(
        config: &CStr,
        data_schema: &str,
        parameter_schema: &str,
        prepare: UnaryF32Prepare<C, P, W, E>,
    ) -> Result<Self, i32>
    where
        C: DeserializeOwned,
    {
        let config = config.to_str().map_err(|_| abi::EINVAL)?;
        let config = serde_json::from_str(config).map_err(|_| abi::EINVAL)?;
        let prepared = prepare(config).map_err(|_| abi::EINVAL)?;
        let elements = u32::try_from(prepared.extent).map_err(|_| abi::EOVERFLOW)?;
        let data_format = NdarrayFormat::f32(
            Box::new([elements]),
            abi::LAYOUT_ROW_MAJOR,
            Some(data_schema),
        )
        .map_err(|_| abi::EINVAL)?;
        let parameter_format = NdarrayFormat::f32(
            Box::new([elements]),
            abi::LAYOUT_ROW_MAJOR,
            Some(parameter_schema),
        )
        .map_err(|_| abi::EINVAL)?;
        Ok(Self {
            data_format,
            parameter_format,
            extent: prepared.extent,
            plans: ParameterRuntime::new(prepared.plan),
            workspace: UnsafeCell::new(prepared.workspace),
        })
    }

    /// Return the exact format for data ports zero and one or parameter port
    /// two.
    pub fn port_format(&self, port: u32) -> Option<&abi::Format> {
        match port {
            0 | 1 => Some(self.data_format.as_abi()),
            2 => Some(self.parameter_format.as_abi()),
            _ => None,
        }
    }

    /// Return the parameter plan runtime and its read-only sequence surface.
    pub const fn plans(&self) -> &ParameterRuntime<P> {
        &self.plans
    }

    /// Prepare a replacement plan from one borrowed parameter buffer.
    pub fn prepare_parameter_with<E>(
        &self,
        port: u32,
        buffer: &abi::Buffer,
        prepare: UnaryF32ParameterPrepare<P, E>,
    ) -> Result<*mut core::ffi::c_void, i32> {
        if port != 2 {
            return Err(abi::EINVAL);
        }
        // SAFETY: the FGN callback lends a valid buffer for this call. The
        // checked helper validates every nested pointer and extent.
        let values = unsafe { crate::f32_slice(buffer, self.extent) }?;
        match unsafe { crate::header_sequence(buffer.buffer) }? {
            Some(sequence) => self
                .plans
                .prepare_with(sequence, |active| prepare(active, values)),
            None => self
                .plans
                .prepare_next_with(|active| prepare(active, values)),
        }
    }

    /// Publish one prepared parameter plan on the serial parameter path.
    ///
    /// # Safety
    ///
    /// The token must come from [`Self::prepare_parameter_with`].
    pub unsafe fn commit_parameter(&self, prepared: *mut core::ffi::c_void) {
        unsafe { self.plans.commit(prepared) }
    }

    /// Discard one unpublished parameter plan on the serial parameter path.
    ///
    /// # Safety
    ///
    /// The token must come from [`Self::prepare_parameter_with`].
    pub unsafe fn discard_parameter(&self, prepared: *mut core::ffi::c_void) {
        unsafe { self.plans.discard(prepared) }
    }

    /// Adopt a complete parameter plan, then execute one data frame.
    ///
    /// # Safety
    ///
    /// One data-loop owner must call this method without overlap or reentry.
    pub unsafe fn process_with<E>(
        &self,
        inputs: InputBuffers<'_>,
        outputs: &mut OutputBuffers<'_>,
        process: impl FnOnce(&P, &mut [f32], &mut W, &[f32]) -> Result<(), E>,
    ) -> Result<(), i32> {
        if inputs.len() != 2 || outputs.len() != 1 || !inputs.is_disconnected(1)? {
            return Err(abi::EINVAL);
        }
        let input_buffer = &inputs.buffers[0];
        let output_buffer = &mut outputs.buffers[0];
        // SAFETY: the FGN host lends complete buffers to its sole data-loop
        // owner. The shared helper also rejects overlapping storage.
        let (input, output) =
            unsafe { crate::f32_input_output(input_buffer, output_buffer, self.extent) }?;
        self.plans.adopt();
        // SAFETY: the FGN lifecycle supplies one data-loop owner.
        let workspace = unsafe { &mut *self.workspace.get() };
        process(self.plans.active_plan(), output, workspace, input).map_err(|_| abi::EINVAL)?;
        // SAFETY: the buffers validated above remain lent for this call.
        unsafe {
            crate::finish_f32_output(output_buffer, self.extent)?;
            crate::copy_matching_metadata(input_buffer.buffer, output_buffer.buffer)?;
        }
        Ok(())
    }

    /// Mutate workspace during serialized lifecycle reset.
    ///
    /// # Safety
    ///
    /// Reset must not overlap or reenter processing.
    pub unsafe fn reset_with(&self, reset: impl FnOnce(&mut W)) {
        // SAFETY: reset cannot overlap processing under the FGN lifecycle.
        reset(unsafe { &mut *self.workspace.get() });
    }
}

/// Static ports shared by unary F32 plugins with one parameter vector.
#[doc(hidden)]
pub static UNARY_F32_PARAMETER_PORTS: [abi::PortInfo; 3] = [
    abi::PortInfo::new(0, abi::DIRECTION_INPUT, 0, c"in".as_ptr()),
    abi::PortInfo::new(1, abi::DIRECTION_OUTPUT, 0, c"out".as_ptr()),
    abi::PortInfo::new(
        2,
        abi::DIRECTION_INPUT,
        abi::PORT_OPTIONAL | abi::PORT_PARAMETER,
        c"parameter".as_ptr(),
    ),
];

/// Declare a typed unary F32 FilterGraph plugin without FGN ABI code.
///
/// The preparation function receives the deserialized configuration and
/// returns [`PreparedUnaryF32`]. Select `runtime` when the plan implements
/// FilterGraph's property interface, or `none` when all scalar configuration is
/// fixed for the instance lifetime.
#[macro_export]
macro_rules! declare_unary_f32_plugin {
    (
        $(#[$meta:meta])*
        $vis:vis $name:ident {
            label = $label:literal;
            schema = $schema:expr;
            config = $config:ty;
            plan = $plan:ty;
            properties = $properties:ident;
            prepare = $prepare:path;
            $(reset = $reset:path;)?
        }
    ) => {
        $(#[$meta])*
        $vis struct $name {
            inner: $crate::UnaryF32Instance<
                $crate::declare_unary_f32_plugin!(@store $properties, $plan),
                <$plan as $crate::__private::AlgorithmPlan>::Workspace,
            >,
        }

        unsafe impl $crate::PluginInstance for $name {
            type Properties =
                $crate::declare_unary_f32_plugin!(@store $properties, $plan);

            const NAME: &'static core::ffi::CStr = match core::ffi::CStr::from_bytes_with_nul(
                concat!($label, "\0").as_bytes(),
            ) {
                Ok(name) => name,
                Err(_) => panic!("FilterGraph plugin labels cannot contain NUL bytes"),
            };
            const PORTS: &'static [$crate::abi::PortInfo] = &$crate::UNARY_F32_PORTS;

            fn instantiate(config: &core::ffi::CStr) -> Result<Self, i32> {
                $crate::UnaryF32Instance::from_json::<$config, _>(config, $schema, $prepare)
                    .map(|inner| Self { inner })
            }

            fn port_format(&self, port: u32) -> Option<&$crate::abi::Format> {
                self.inner.port_format(port)
            }

            fn properties(&self) -> &Self::Properties {
                self.inner.plans()
            }

            fn process(
                &self,
                inputs: $crate::InputBuffers<'_>,
                outputs: &mut $crate::OutputBuffers<'_>,
            ) -> Result<(), i32> {
                // SAFETY: the FGN callback contract provides one non-reentrant
                // data-loop owner for this instance.
                unsafe {
                    self.inner.process_with(inputs, outputs, |plan, output, workspace, input| {
                        <$plan as $crate::__private::AlgorithmPlan>::process(
                            plan, output, workspace, input,
                        )
                    })
                }
            }

            $crate::declare_unary_f32_plugin!(@reset_method $($reset)?);
        }
    };

    (@store runtime, $plan:ty) => {
        $crate::PropertyRuntime<$plan>
    };
    (@store none, $plan:ty) => {
        $crate::FixedPlan<$plan>
    };
    (@reset_method) => {};
    (@reset_method $reset:path) => {
        fn reset(&self) -> Result<(), i32> {
            // SAFETY: FGN serializes reset outside processing.
            unsafe { self.inner.reset_with($reset) };
            Ok(())
        }
    };
}

/// Declare a unary F32 FilterGraph plugin with one sparse F32 parameter.
///
/// The parameter preparation function receives the active immutable plan and
/// the borrowed parameter values, then returns a complete replacement plan.
/// Shared code owns copying or preparation lifetime, bounded publication,
/// sequence properties, frame-boundary adoption, and retired-plan reclamation.
#[macro_export]
macro_rules! declare_unary_f32_parameter_plugin {
    (
        $(#[$meta:meta])*
        $vis:vis $name:ident {
            label = $label:literal;
            data_schema = $data_schema:expr;
            parameter_schema = $parameter_schema:expr;
            config = $config:ty;
            plan = $plan:ty;
            prepare = $prepare:path;
            prepare_parameter = $prepare_parameter:path;
            $(reset = $reset:path;)?
        }
    ) => {
        $(#[$meta])*
        $vis struct $name {
            inner: $crate::UnaryF32ParameterInstance<
                $plan,
                <$plan as $crate::__private::AlgorithmPlan>::Workspace,
            >,
        }

        unsafe impl $crate::PluginInstance for $name {
            type Properties = $crate::ParameterRuntime<$plan>;

            const NAME: &'static core::ffi::CStr = match core::ffi::CStr::from_bytes_with_nul(
                concat!($label, "\0").as_bytes(),
            ) {
                Ok(name) => name,
                Err(_) => panic!("FilterGraph plugin labels cannot contain NUL bytes"),
            };
            const PORTS: &'static [$crate::abi::PortInfo] =
                &$crate::UNARY_F32_PARAMETER_PORTS;

            fn instantiate(config: &core::ffi::CStr) -> Result<Self, i32> {
                $crate::UnaryF32ParameterInstance::from_json::<$config, _>(
                    config,
                    $data_schema,
                    $parameter_schema,
                    $prepare,
                )
                .map(|inner| Self { inner })
            }

            fn port_format(&self, port: u32) -> Option<&$crate::abi::Format> {
                self.inner.port_format(port)
            }

            fn properties(&self) -> &Self::Properties {
                self.inner.plans()
            }

            fn process(
                &self,
                inputs: $crate::InputBuffers<'_>,
                outputs: &mut $crate::OutputBuffers<'_>,
            ) -> Result<(), i32> {
                // SAFETY: the FGN callback contract provides one non-reentrant
                // data-loop owner for this instance.
                unsafe {
                    self.inner.process_with(inputs, outputs, |plan, output, workspace, input| {
                        <$plan as $crate::__private::AlgorithmPlan>::process(
                            plan, output, workspace, input,
                        )
                    })
                }
            }

            unsafe fn prepare_parameter(
                &self,
                port: u32,
                buffer: &$crate::abi::Buffer,
            ) -> Result<*mut core::ffi::c_void, i32> {
                self.inner
                    .prepare_parameter_with(port, buffer, $prepare_parameter)
            }

            unsafe fn commit_parameter(&self, prepared: *mut core::ffi::c_void) {
                unsafe { self.inner.commit_parameter(prepared) }
            }

            unsafe fn discard_parameter(&self, prepared: *mut core::ffi::c_void) {
                unsafe { self.inner.discard_parameter(prepared) }
            }

            $crate::declare_unary_f32_parameter_plugin!(@reset_method $($reset)?);
        }
    };

    (@reset_method) => {};
    (@reset_method $reset:path) => {
        fn reset(&self) -> Result<(), i32> {
            // SAFETY: FGN serializes reset outside processing.
            unsafe { self.inner.reset_with($reset) };
            Ok(())
        }
    };
}
