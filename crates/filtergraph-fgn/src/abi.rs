//! Maintained Rust representation of the PipeWireAO FGN ABI.
//!
//! The upstream `ndarray-plugin.h` and `ndarray-executor.h` headers are MIT
//! licensed, Copyright © 2026 PipeWireAO contributors. Their license is
//! reproduced under `docs/releases/third-party-licenses/PipeWireAO/`.
//!
//! The C graph-host integration test checks these layouts by loading a plugin
//! built from this module and validating every public `struct_size`.

#![allow(missing_docs)]

use core::ffi::{c_char, c_void};

/// FGN ABI revision emitted by this build.
pub const ABI_VERSION: u32 = 8;
pub const EXECUTOR_VERSION: u32 = 1;
pub const LOCAL_NAME_MAX: usize = 255;
pub const MAX_PARAMETER_TRANSACTION_ASSIGNMENTS: usize = 4096;

pub const DIRECTION_INPUT: u32 = 0;
pub const DIRECTION_OUTPUT: u32 = 1;
pub const PORT_OPTIONAL: u32 = 1 << 0;
pub const PORT_PARAMETER: u32 = 1 << 1;
pub const PORT_CONDITIONAL: u32 = 1 << 2;

pub const PLUGIN_RETAIN_LIBRARY: u32 = 1 << 0;

pub const PROPERTY_READONLY: u32 = 1 << 0;
pub const PROPERTY_RANGE: u32 = 1 << 1;
pub const PROPERTY_RUNTIME: u32 = 1 << 2;
pub const PROPERTY_CONSTRUCTION: u32 = 1 << 3;
pub const PROPERTY_GRAPH_REBUILD: u32 = 1 << 4;
pub const PROPERTY_CHOICES: u32 = 1 << 5;

pub const TYPE_NONE: u32 = 1;
pub const TYPE_BOOL: u32 = 2;
pub const TYPE_ID: u32 = 3;
pub const TYPE_INT: u32 = 4;
pub const TYPE_LONG: u32 = 5;
pub const TYPE_FLOAT: u32 = 6;
pub const TYPE_DOUBLE: u32 = 7;
pub const TYPE_STRING: u32 = 8;

pub const ELEMENT_BOOL8: u32 = 1;
pub const ELEMENT_I8: u32 = 2;
pub const ELEMENT_U8: u32 = 3;
pub const ELEMENT_I16_LE: u32 = 4;
pub const ELEMENT_U16_LE: u32 = 5;
pub const ELEMENT_I32_LE: u32 = 6;
pub const ELEMENT_U32_LE: u32 = 7;
pub const ELEMENT_I64_LE: u32 = 8;
pub const ELEMENT_U64_LE: u32 = 9;
pub const ELEMENT_F32_LE: u32 = 18;
pub const ELEMENT_F64_LE: u32 = 19;
pub const LAYOUT_ROW_MAJOR: u32 = 1;
pub const LAYOUT_COLUMN_MAJOR: u32 = 2;
pub const NDARRAY_MAX_DIMENSIONS: usize = 64;
pub const MAX_METAS: usize = 64;
pub const MAX_META_BYTES: usize = 4096;
pub const META_HEADER: u32 = 1;
pub const META_ACQUISITION_IDENTITY: u32 = 0x300;
pub const META_EXPOSURE_TIMING: u32 = 0x301;
pub const META_OPTICAL_CHOPPING: u32 = 0x303;
pub const META_ELECTRICAL_TRIGGER: u32 = 0x304;
pub const META_REFERENCE_TIMESTAMP: u32 = 0x305;
pub const META_REQUIREMENT_REQUIRED: u32 = 1;
pub const META_PROPAGATION_PRODUCE: u32 = 0;
pub const META_PROPAGATION_PRESERVE: u32 = 1;
pub const META_PROPAGATION_REPLACE: u32 = 2;
pub const META_PROPAGATION_DROP: u32 = 3;
pub const META_HEADER_DISCONT: u32 = 1 << 0;
pub const META_HEADER_CORRUPTED: u32 = 1 << 1;
pub const META_HEADER_MARKER: u32 = 1 << 2;
pub const TIME_INVALID: i64 = i64::MIN;

pub const EFAULT: i32 = 14;
pub const ENOMEM: i32 = 12;
pub const E2BIG: i32 = 7;
pub const EBUSY: i32 = 16;
pub const EEXIST: i32 = 17;
pub const EINVAL: i32 = 22;
pub const ENOENT: i32 = 2;
pub const EOVERFLOW: i32 = 75;
pub const EPERM: i32 = 1;
pub const ERANGE: i32 = 34;
pub const ENOBUFS: i32 = 105;
pub const ENOSPC: i32 = 28;
pub const EMSGSIZE: i32 = 90;

#[repr(C)]
pub struct SpaChunk {
    pub offset: u32,
    pub size: u32,
    pub stride: i32,
    pub flags: i32,
}

#[repr(C)]
pub struct SpaData {
    pub type_: u32,
    pub flags: u32,
    pub fd: i64,
    pub mapoffset: u32,
    pub maxsize: u32,
    pub data: *mut c_void,
    pub chunk: *mut SpaChunk,
}

#[repr(C)]
pub struct SpaMeta {
    pub type_: u32,
    pub size: u32,
    pub data: *mut c_void,
}

#[repr(C)]
pub struct SpaMetaHeader {
    pub flags: u32,
    pub offset: u32,
    pub pts: i64,
    pub dts_offset: i64,
    pub seq: u64,
}

/// Native 4.x independent acquisition identity.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SpaMetaAcquisitionIdentity {
    pub flags: u32,
    pub reserved: u32,
    pub domain: [u8; 16],
    pub generation: u64,
    pub sequence: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SpaMetaExposureTiming {
    pub flags: u32,
    pub clock: u32,
    pub start_nsec: i64,
    pub duration_nsec: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SpaMetaReferenceTimestamp {
    pub clock: u32,
    pub reserved: u32,
    pub timestamp_nsec: u64,
    pub duration_nsec: u64,
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
pub struct SpaMetaOpticalChopping {
    pub flags: u32,
    pub classification: u32,
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
pub struct SpaMetaElectricalTrigger {
    pub flags: u32,
    pub classification: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SpaMetaRequirement {
    pub type_: u32,
    pub size: u32,
    pub flags: u32,
    pub reserved: u32,
}

#[repr(C)]
pub struct SpaBuffer {
    pub n_metas: u32,
    pub n_datas: u32,
    pub metas: *mut SpaMeta,
    pub datas: *mut SpaData,
}

#[repr(C)]
pub struct Format {
    pub element_type: u32,
    pub layout: u32,
    pub rate_num: u32,
    pub rate_denom: u32,
    pub n_dimensions: u32,
    pub shape: *const u32,
    pub schema: *const c_char,
}

// SAFETY: a format and everything it references are immutable after admission.
unsafe impl Sync for Format {}

#[repr(C)]
pub struct PortInfo {
    pub struct_size: u32,
    pub index: u32,
    pub direction: u32,
    pub flags: u32,
    pub name: *const c_char,
    pub n_metas: u32,
    pub propagation: u32,
    pub metas: *const SpaMetaRequirement,
}

// SAFETY: port descriptors have immutable static storage.
unsafe impl Sync for PortInfo {}

impl PortInfo {
    pub const fn new(index: u32, direction: u32, flags: u32, name: *const c_char) -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            index,
            direction,
            flags,
            name,
            n_metas: 0,
            propagation: META_PROPAGATION_PRESERVE,
            metas: core::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union ValueBody {
    pub boolean: i32,
    pub integer: i32,
    pub long_integer: i64,
    pub float_value: f32,
    pub double_value: f64,
    pub id: u32,
    pub string: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Value {
    pub type_: u32,
    pub reserved: u32,
    pub value: ValueBody,
}

impl Value {
    pub const NONE: Self = Self {
        type_: TYPE_NONE,
        reserved: 0,
        value: ValueBody { long_integer: 0 },
    };
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PropertyChoice {
    pub struct_size: u32,
    pub reserved: u32,
    pub value: Value,
    pub name: *const c_char,
    pub description: *const c_char,
}

// SAFETY: choices and their strings are immutable for the instance lifetime.
unsafe impl Sync for PropertyChoice {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PropertyInfo {
    pub struct_size: u32,
    pub id: u32,
    pub flags: u32,
    pub reserved: u32,
    pub name: *const c_char,
    pub description: *const c_char,
    pub unit: *const c_char,
    pub default_value: Value,
    pub minimum: Value,
    pub maximum: Value,
    pub n_choices: u32,
    pub choices_reserved: u32,
    pub choices: *const PropertyChoice,
}

#[repr(C)]
pub struct Property {
    pub id: u32,
    pub reserved: u32,
    pub value: Value,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Buffer {
    pub buffer: *mut SpaBuffer,
    pub format: *const Format,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Parameter {
    pub port: u32,
    pub reserved: u32,
    pub buffer: Buffer,
}

pub const EXECUTOR_POLLING: u32 = 1 << 0;
pub const DENSE_F32_ACCUMULATE: u32 = 1 << 0;

#[repr(C)]
pub struct DenseF32Task {
    pub struct_size: u32,
    pub flags: u32,
    pub matrix: *const f32,
    pub input: *const f32,
    pub output: *mut f32,
    pub n_rows: u32,
    pub matrix_row_stride: u32,
    pub first_column: u32,
    pub n_columns: u32,
}

pub type RunDenseF32 = unsafe extern "C" fn(*mut c_void, *const DenseF32Task) -> i32;

pub type RunLane = unsafe extern "C" fn(*mut c_void, u32, u32) -> i32;

#[repr(C)]
pub struct LanesTask {
    pub struct_size: u32,
    pub flags: u32,
    pub data: *mut c_void,
    pub run: Option<RunLane>,
    pub n_lanes: u32,
    pub reserved: u32,
}

pub type RunLanes = unsafe extern "C" fn(*mut c_void, *const LanesTask) -> i32;

#[repr(C)]
pub struct Executor {
    pub struct_size: u32,
    pub version: u32,
    pub flags: u32,
    pub n_lanes: u32,
    pub data: *mut c_void,
    pub run_dense_f32: Option<RunDenseF32>,
    pub run_lanes: Option<RunLanes>,
    pub cache_line_size: u32,
    pub reserved: u32,
}

pub type Instantiate = unsafe extern "C" fn(
    *const Descriptor,
    *const c_char,
    *const Executor,
    *mut *mut c_void,
) -> i32;
pub type Cleanup = unsafe extern "C" fn(*mut c_void);
pub type GetPortFormat = unsafe extern "C" fn(*mut c_void, u32, *mut *const Format) -> i32;
pub type EnumPropInfo = unsafe extern "C" fn(*mut c_void, u32, *mut PropertyInfo) -> i32;
pub type GetProp = unsafe extern "C" fn(*mut c_void, u32, *mut Value) -> i32;
pub type GetPortMetadata =
    unsafe extern "C" fn(*mut c_void, u32, *mut *const SpaMetaRequirement, *mut u32) -> i32;
pub type GetPropRevision = unsafe extern "C" fn(*mut c_void) -> u64;
pub type PrepareProps =
    unsafe extern "C" fn(*mut c_void, *const Property, u32, *mut *mut c_void) -> i32;
pub type PreparedAction = unsafe extern "C" fn(*mut c_void, *mut c_void);
pub type PrepareParameter =
    unsafe extern "C" fn(*mut c_void, u32, *const Buffer, *mut *mut c_void) -> i32;
pub type PrepareParameters =
    unsafe extern "C" fn(*mut c_void, *const Parameter, u32, *mut *mut c_void) -> i32;
pub type AdoptParameters = unsafe extern "C" fn(*mut c_void);
pub type Lifecycle = unsafe extern "C" fn(*mut c_void) -> i32;
pub type Process = unsafe extern "C" fn(*mut c_void, *const Buffer, u32, *mut Buffer, u32) -> i32;

#[repr(C)]
pub struct Descriptor {
    pub struct_size: u32,
    pub version: u32,
    pub name: *const c_char,
    pub n_ports: u32,
    pub ports: *const PortInfo,
    pub instantiate: Option<Instantiate>,
    pub cleanup: Option<Cleanup>,
    pub get_port_format: Option<GetPortFormat>,
    pub enum_prop_info: Option<EnumPropInfo>,
    pub get_prop: Option<GetProp>,
    pub get_prop_revision: Option<GetPropRevision>,
    pub prepare_props: Option<PrepareProps>,
    pub commit_props: Option<PreparedAction>,
    pub discard_props: Option<PreparedAction>,
    pub prepare_parameter: Option<PrepareParameter>,
    pub commit_parameter: Option<PreparedAction>,
    pub discard_parameter: Option<PreparedAction>,
    pub prepare_parameters: Option<PrepareParameters>,
    pub adopt_parameters: Option<AdoptParameters>,
    pub activate: Option<Lifecycle>,
    pub deactivate: Option<Lifecycle>,
    pub reset: Option<Lifecycle>,
    pub process: Option<Process>,
    /// Optional ABI-v7 extension invoked on the exact process coordinator.
    pub prepare_process_thread: Option<Lifecycle>,
    /// Optional preparation-time instance metadata requirements.
    pub get_port_metadata: Option<GetPortMetadata>,
}

// SAFETY: descriptors and callback tables have immutable static storage.
unsafe impl Sync for Descriptor {}

pub type FindDescriptor = unsafe extern "C" fn(*const c_char) -> *const Descriptor;

#[repr(C)]
pub struct Plugin {
    pub struct_size: u32,
    pub abi_version: u32,
    pub name: *const c_char,
    pub find_descriptor: Option<FindDescriptor>,
    pub flags: u32,
}

// SAFETY: plugin descriptors have immutable static storage.
unsafe impl Sync for Plugin {}
