//! Reusable deployment for F32 vector sources with sparse parameters.

use core::cell::UnsafeCell;
use core::ffi::CStr;

use serde::de::DeserializeOwned;

use crate::{NdarrayFormat, OutputBuffers, ParameterRuntime, PlanStore, abi};

/// Prepared, transport-independent contents of an F32 vector source.
pub struct PreparedSourceF32<P, W> {
    plan: P,
    workspace: W,
    extent: usize,
}

impl<P, W> PreparedSourceF32<P, W> {
    /// Combine an admitted FilterGraph plan, its workspace, and output extent.
    pub const fn new(plan: P, workspace: W, extent: usize) -> Self {
        Self {
            plan,
            workspace,
            extent,
        }
    }
}

/// Typed cold-path preparation function for one F32 vector source.
pub type SourceF32Prepare<C, P, W, E> = fn(C) -> Result<PreparedSourceF32<P, W>, E>;

/// Typed cold-path replacement-plan function for one F32 parameter vector.
pub type SourceF32ParameterPrepare<P, E> = fn(&P, &[f32]) -> Result<P, E>;

/// Shared FGN-owned instance for an F32 source with one sparse parameter.
pub struct SourceF32ParameterInstance<P, W> {
    output_format: NdarrayFormat,
    parameter_format: NdarrayFormat,
    extent: usize,
    plans: ParameterRuntime<P>,
    workspace: UnsafeCell<W>,
}

// SAFETY: FGN serializes `process` and reset for one instance. Parameter
// publication is owned by ParameterRuntime, while the single data-loop owner
// has exclusive access to the workspace.
unsafe impl<P, W> Sync for SourceF32ParameterInstance<P, W>
where
    P: Send + Sync,
    W: Send,
{
}

impl<P, W> SourceF32ParameterInstance<P, W>
where
    P: Send + Sync,
{
    /// Decode typed configuration and prepare one source off the data loop.
    pub fn from_json<C, E>(
        config: &CStr,
        output_schema: &str,
        parameter_schema: &str,
        prepare: SourceF32Prepare<C, P, W, E>,
    ) -> Result<Self, i32>
    where
        C: DeserializeOwned,
    {
        let config = config.to_str().map_err(|_| abi::EINVAL)?;
        let config = serde_json::from_str(config).map_err(|_| abi::EINVAL)?;
        let prepared = prepare(config).map_err(|_| abi::EINVAL)?;
        let elements = u32::try_from(prepared.extent).map_err(|_| abi::EOVERFLOW)?;
        let output_format = NdarrayFormat::f32(
            Box::new([elements]),
            abi::LAYOUT_ROW_MAJOR,
            Some(output_schema),
        )
        .map_err(|_| abi::EINVAL)?;
        let parameter_format = NdarrayFormat::f32(
            Box::new([elements]),
            abi::LAYOUT_ROW_MAJOR,
            Some(parameter_schema),
        )
        .map_err(|_| abi::EINVAL)?;
        Ok(Self {
            output_format,
            parameter_format,
            extent: prepared.extent,
            plans: ParameterRuntime::new(prepared.plan),
            workspace: UnsafeCell::new(prepared.workspace),
        })
    }

    /// Return the output format for port zero or parameter format for port one.
    pub fn port_format(&self, port: u32) -> Option<&abi::Format> {
        match port {
            0 => Some(self.output_format.as_abi()),
            1 => Some(self.parameter_format.as_abi()),
            _ => None,
        }
    }

    /// Return the parameter plan runtime and read-only sequence surface.
    pub const fn plans(&self) -> &ParameterRuntime<P> {
        &self.plans
    }

    /// Prepare a complete plan from one borrowed parameter vector.
    pub fn prepare_parameter_with<E>(
        &self,
        port: u32,
        buffer: &abi::Buffer,
        prepare: SourceF32ParameterPrepare<P, E>,
    ) -> Result<*mut core::ffi::c_void, i32> {
        if port != 1 {
            return Err(abi::EINVAL);
        }
        // SAFETY: the callback lends a valid buffer for this call. The shared
        // helper validates every nested pointer and extent.
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

    /// Adopt a complete parameter plan, then emit one vector.
    ///
    /// # Safety
    ///
    /// One data-loop owner must call this method without overlap or reentry.
    pub unsafe fn process_with<E>(
        &self,
        inputs: crate::InputBuffers<'_>,
        outputs: &mut OutputBuffers<'_>,
        process: impl FnOnce(&P, &mut [f32], &mut W) -> Result<(), E>,
    ) -> Result<(), i32> {
        if inputs.len() != 1 || !inputs.is_disconnected(0)? || outputs.len() != 1 {
            return Err(abi::EINVAL);
        }
        let output = outputs.f32_mut(0, self.extent)?;
        self.plans.adopt();
        // SAFETY: the FGN lifecycle supplies one data-loop owner.
        let workspace = unsafe { &mut *self.workspace.get() };
        process(self.plans.active_plan(), output, workspace).map_err(|_| abi::EINVAL)?;
        outputs.finish_f32(0, self.extent)
    }

    /// Mutate workspace during serialized lifecycle reset.
    ///
    /// # Safety
    ///
    /// Reset must not overlap or reenter processing.
    pub unsafe fn reset_with(&self, reset: impl FnOnce(&P, &mut W)) {
        // SAFETY: reset cannot overlap processing under the FGN lifecycle.
        reset(self.plans.active_plan(), unsafe {
            &mut *self.workspace.get()
        });
    }
}

/// Static ports shared by F32 sources with one parameter vector.
#[doc(hidden)]
pub static SOURCE_F32_PARAMETER_PORTS: [abi::PortInfo; 2] = [
    abi::PortInfo::new(0, abi::DIRECTION_OUTPUT, 0, c"out".as_ptr()),
    abi::PortInfo::new(
        1,
        abi::DIRECTION_INPUT,
        abi::PORT_OPTIONAL | abi::PORT_PARAMETER,
        c"parameter".as_ptr(),
    ),
];

/// Declare an F32 vector-source plugin with one sparse F32 parameter.
#[macro_export]
macro_rules! declare_source_f32_parameter_plugin {
    (
        $(#[$meta:meta])*
        $vis:vis $name:ident {
            label = $label:literal;
            output_schema = $output_schema:expr;
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
            inner: $crate::SourceF32ParameterInstance<
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
                &$crate::SOURCE_F32_PARAMETER_PORTS;

            fn instantiate(config: &core::ffi::CStr) -> Result<Self, i32> {
                $crate::SourceF32ParameterInstance::from_json::<$config, _>(
                    config,
                    $output_schema,
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
                    self.inner.process_with(inputs, outputs, |plan, output, workspace| {
                        <$plan as $crate::__private::AlgorithmPlan>::process(
                            plan, output, workspace, (),
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

            $crate::declare_source_f32_parameter_plugin!(@reset_method $($reset)?);
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
