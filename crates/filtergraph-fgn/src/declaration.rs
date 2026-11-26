//! One topology-independent declaration path for FilterGraph plugins.

use core::cell::UnsafeCell;
use core::ffi::{CStr, c_void};
use core::marker::PhantomData;

use serde::de::DeserializeOwned;

use crate::{
    InputBuffers, NdarrayElement, NdarrayFormat, OutputBuffers, PlanStore, PluginInstance, abi,
};

/// A prepared plan, workspace, and exact port formats.
///
/// Algorithm preparation remains ordinary typed Rust. This value is the only
/// deployment-specific grouping it must return to a [`PluginDeclaration`].
pub struct PreparedInstance<P, W> {
    plan: P,
    workspace: W,
    formats: Box<[NdarrayFormat]>,
}

impl<P, W> PreparedInstance<P, W> {
    /// Group one admitted plan and compatible workspace with descriptor-order
    /// port formats.
    pub fn new(plan: P, workspace: W, formats: Box<[NdarrayFormat]>) -> Self {
        Self {
            plan,
            workspace,
            formats,
        }
    }

    /// Return the prepared scientific plan.
    pub const fn plan(&self) -> &P {
        &self.plan
    }

    /// Return the compatible workspace.
    pub const fn workspace(&self) -> &W {
        &self.workspace
    }

    /// Return exact formats in descriptor order.
    pub const fn formats(&self) -> &[NdarrayFormat] {
        &self.formats
    }
}

/// Checked borrowed view of one sparse ndarray parameter update.
#[derive(Clone, Copy)]
pub struct ParameterBuffer<'a> {
    buffer: &'a abi::Buffer,
}

impl<'a> ParameterBuffer<'a> {
    fn new(buffer: &'a abi::Buffer) -> Self {
        Self { buffer }
    }

    /// Borrow one packed typed array for this preparation call.
    pub fn slice<T: NdarrayElement>(&self, elements: usize) -> Result<&'a [T], i32> {
        // SAFETY: the FGN parameter callback lends the complete buffer for this
        // call. The shared helper checks every nested pointer and extent.
        unsafe { crate::typed_slice(self.buffer, elements) }
    }

    /// Return the optional standard Header sequence attached to the update.
    pub fn sequence(&self) -> Result<Option<u64>, i32> {
        // SAFETY: the FGN parameter callback lends the nested SPA buffer for
        // this call and the shared helper validates its metadata first.
        unsafe { crate::header_sequence(self.buffer.buffer) }
    }
}

/// Safe, topology-independent deployment declaration for one FilterGraph plugin.
///
/// A deployment adapter supplies typed JSON configuration, ordinary
/// plan/workspace preparation, exact scientific port formats, and a small
/// mapping between checked ndarray slices and the plan's typed input/output
/// views. Scientific [`filtergraph_algorithms::AlgorithmPlan`] implementations do
/// not implement this trait. The shared [`DeclaredPlugin`] owns every C
/// callback, raw pointer, errno conversion, unwind barrier, workspace ownership
/// proof, and publication transition.
pub trait PluginDeclaration: Send + Sync + Sized + 'static {
    /// Typed construction configuration decoded from standard JSON.
    type Configuration: DeserializeOwned;
    /// Prepared scientific plan.
    type Plan: Send + Sync + 'static;
    /// Exclusively data-loop-owned execution storage.
    type Workspace: Send + 'static;
    /// Fixed, property-published, or parameter-published plan owner.
    type Plans: PlanStore<Plan = Self::Plan> + Send + Sync + 'static;
    /// Declaration-specific preparation or processing failure.
    type Error;

    /// Stable plugin descriptor label.
    const NAME: &'static CStr;
    /// Static plugin port descriptors. Formats returned by [`Self::prepare`]
    /// use this exact order.
    const PORTS: &'static [abi::PortInfo];

    /// Prepare a plan, compatible workspace, and exact port formats.
    fn prepare(
        configuration: Self::Configuration,
    ) -> Result<PreparedInstance<Self::Plan, Self::Workspace>, Self::Error>;

    /// Execute one complete scientific operation using checked ndarray views.
    fn process(
        plan: &Self::Plan,
        workspace: &mut Self::Workspace,
        inputs: InputBuffers<'_>,
        outputs: &mut OutputBuffers<'_>,
    ) -> Result<(), Self::Error>;

    /// Reset persistent mathematical history during serialized lifecycle.
    fn reset(_plan: &Self::Plan, _workspace: &mut Self::Workspace) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Prepare a complete replacement plan from one sparse ndarray parameter.
    ///
    /// Declarations whose [`Self::Plans`] do not admit parameter publication
    /// may keep this default.
    fn prepare_parameter(
        _plan: &Self::Plan,
        _port: u32,
        _parameter: ParameterBuffer<'_>,
    ) -> Option<Result<Self::Plan, Self::Error>> {
        None
    }

    /// Map a typed declaration error to a positive errno value at the ABI edge.
    /// Most declarations may keep the default mapping.
    fn errno(_error: &Self::Error) -> i32 {
        abi::EINVAL
    }
}

/// Audited FGN plugin instance generated from one safe declaration.
pub struct DeclaredPlugin<D: PluginDeclaration> {
    plans: D::Plans,
    workspace: UnsafeCell<D::Workspace>,
    formats: Box<[NdarrayFormat]>,
    declaration: PhantomData<D>,
}

// SAFETY: `plans` provides its own synchronized publication. FGN supplies one
// non-reentrant data-loop owner for `workspace`, while lifecycle reset cannot
// overlap processing. Control callbacks never access the workspace.
unsafe impl<D: PluginDeclaration> Sync for DeclaredPlugin<D> {}

fn positive_errno(error: i32) -> i32 {
    match error {
        1.. => error,
        ..=-1 => error.saturating_neg(),
        0 => abi::EFAULT,
    }
}

unsafe impl<D: PluginDeclaration> PluginInstance for DeclaredPlugin<D> {
    type Properties = D::Plans;

    const NAME: &'static CStr = D::NAME;
    const PORTS: &'static [abi::PortInfo] = D::PORTS;

    fn instantiate(config: &CStr) -> Result<Self, i32> {
        let configuration = serde_json::from_slice(config.to_bytes()).map_err(|_| abi::EINVAL)?;
        let prepared =
            D::prepare(configuration).map_err(|error| positive_errno(D::errno(&error)))?;
        if prepared.formats.len() != D::PORTS.len() {
            return Err(abi::EINVAL);
        }
        let plans = D::Plans::from_plan(prepared.plan)?;
        Ok(Self {
            plans,
            workspace: UnsafeCell::new(prepared.workspace),
            formats: prepared.formats,
            declaration: PhantomData,
        })
    }

    fn port_format(&self, port: u32) -> Option<&abi::Format> {
        D::PORTS
            .iter()
            .position(|info| info.index == port)
            .and_then(|index| self.formats.get(index))
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
        self.plans.adopt();
        // SAFETY: FGN supplies exactly one non-reentrant data-loop owner.
        let workspace = unsafe { &mut *self.workspace.get() };
        D::process(self.plans.active_plan(), workspace, inputs, outputs)
            .map_err(|error| positive_errno(D::errno(&error)))
    }

    fn reset(&self) -> Result<(), i32> {
        // SAFETY: reset is serialized outside processing by the FGN lifecycle.
        let workspace = unsafe { &mut *self.workspace.get() };
        D::reset(self.plans.active_plan(), workspace)
            .map_err(|error| positive_errno(D::errno(&error)))
    }

    unsafe fn prepare_parameter(
        &self,
        port: u32,
        buffer: &abi::Buffer,
    ) -> Result<*mut c_void, i32> {
        let admitted = D::PORTS.iter().any(|info| {
            info.index == port
                && info.direction == abi::DIRECTION_INPUT
                && info.flags & abi::PORT_PARAMETER != 0
        });
        if !admitted {
            return Err(abi::EINVAL);
        }
        let parameter = ParameterBuffer::new(buffer);
        let sequence = parameter.sequence()?;
        self.plans.prepare_parameter_with(sequence, |plan| {
            D::prepare_parameter(plan, port, parameter)
                .ok_or(())?
                .map_err(|_| ())
        })
    }

    unsafe fn commit_parameter(&self, prepared: *mut c_void) {
        unsafe { self.plans.commit_parameter(prepared) }
    }

    unsafe fn discard_parameter(&self, prepared: *mut c_void) {
        unsafe { self.plans.discard_parameter(prepared) }
    }
}
