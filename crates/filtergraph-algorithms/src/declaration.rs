//! Transport-neutral declarations for typed algorithm graph boundaries.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::mem;

use crate::interface::{AlgorithmProperties, PropertyAssignment};

/// Maximum rank admitted by the portable ndarray declaration contract.
pub const MAX_NDARRAY_DIMENSIONS: usize = 64;

/// Maximum UTF-8 byte length of a portable scientific-schema identifier.
pub const MAX_SCIENTIFIC_SCHEMA_BYTES: usize = 255;

/// Primitive ndarray element representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElementType {
    /// One byte restricted to the values zero and one.
    Bool8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Little-endian signed 16-bit integer.
    I16,
    /// Little-endian unsigned 16-bit integer.
    U16,
    /// Little-endian signed 32-bit integer.
    I32,
    /// Little-endian unsigned 32-bit integer.
    U32,
    /// Little-endian signed 64-bit integer.
    I64,
    /// Little-endian unsigned 64-bit integer.
    U64,
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary64.
    F64,
}

/// Packed storage order for a declared ndarray.
///
/// Logical shapes and scientific schemas are independent of storage order.
/// Rust slice-backed declarations currently use [`Self::RowMajor`], while a
/// Julia deployment can project ordinary Julia arrays as
/// [`Self::ColumnMajor`] without changing the scientific port definition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArrayLayout {
    /// The last logical axis is contiguous.
    RowMajor,
    /// The first logical axis is contiguous.
    ColumnMajor,
}

impl ElementType {
    const fn size(self) -> usize {
        match self {
            Self::Bool8 | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

/// Direction of one algorithm port.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortDirection {
    /// The algorithm consumes the value.
    Input,
    /// The algorithm produces the value.
    Output,
}

/// Update role of one algorithm port.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortRole {
    /// One value is present for every complete processing call.
    Data,
    /// A sparse ndarray replaces prepared state outside processing.
    Parameter,
}

/// Publication behavior of one algorithm output port.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortPublication {
    /// Every successful processing call publishes one complete artifact.
    Always,
    /// A successful processing call may defer this artifact.
    Conditional,
}

/// Scalar-property behavior needed by a deployment binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PropertyPolicy {
    /// The plan has no scalar property surface.
    None,
    /// The plan exposes runtime-writable scalar properties.
    Runtime,
}

/// Prepared operation shape that a deployment adapter may execute natively.
///
/// This identifies mathematical structure, not a threading policy. A native
/// caller may always use the declaration's ordinary serial process method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgorithmOperation {
    /// The declaration supplies only its ordinary typed process method.
    General,
    /// One row-major F32 operation `output = matrix * input`.
    DenseF32,
    /// Complete-image F32 Shack-Hartmann centroid measurements.
    ShackHartmannImageF32,
    /// Incremental F32 reconstruction from indexed measurement batches.
    IncrementalDenseReconstructionF32,
    /// Progressive F32 pyramid normalization and reconstruction.
    PyramidRowReconstructionF32,
}

/// Exact prepared plan selected by an optional operation marker.
///
/// Deployment adapters use this hidden projection to obtain the mathematical
/// plan without downcasting. Constructing a variant in the declaration macro
/// also makes an incompatible marker and plan type fail at compile time.
#[doc(hidden)]
pub enum AlgorithmOperationPlan<'a> {
    /// No native operation shape was declared.
    General,
    /// Prepared row-major F32 matrix-vector multiplication.
    DenseF32(&'a crate::PreparedGemv<f32>),
    /// Prepared image-backed F32 Shack-Hartmann measurement.
    ShackHartmannImageF32(&'a crate::ShackHartmannImagePlan<f32>),
    /// Prepared incremental F32 reconstruction from measurement batches.
    IncrementalDenseReconstructionF32(&'a crate::IncrementalDenseReconstructionPlan<f32>),
    /// Prepared row-block F32 pyramid normalization and reconstruction.
    PyramidRowReconstructionF32(&'a crate::PyramidRowReconstructionPlan<f32>),
}

/// Exact mutable workspace selected by an optional operation marker.
///
/// This projection is separate from [`AlgorithmOperationPlan`] so ordinary
/// declarations retain independent plan and workspace types while a generated
/// adapter can obtain a checked operation-specific workspace without a cast.
#[doc(hidden)]
pub enum AlgorithmOperationWorkspace<'a> {
    /// No operation-specific mutable workspace was declared.
    General,
    /// Incremental F32 reconstruction state.
    IncrementalDenseReconstructionF32(&'a mut crate::IncrementalDenseReconstructionWorkspace<f32>),
    /// Row-block F32 pyramid assembly, normalization, and reconstruction state.
    PyramidRowReconstructionF32(&'a mut crate::PyramidRowReconstructionWorkspace<f32>),
}

/// Type-level scalar-property capability selected by a declaration.
#[doc(hidden)]
pub trait DeclarationPropertyPolicy: sealed::DeclarationPropertyPolicy {
    /// Runtime behavior exposed to a deployment adapter.
    const POLICY: PropertyPolicy;
}

/// Marker for a declaration without scalar properties.
#[doc(hidden)]
pub struct NoDeclaredProperties;

/// Marker for a declaration backed by [`crate::AlgorithmProperties`].
#[doc(hidden)]
pub struct RuntimeDeclaredProperties;

/// Type-level sparse-parameter capability selected by a declaration.
#[doc(hidden)]
pub trait DeclarationParameterPolicy: sealed::DeclarationParameterPolicy {
    /// Whether the declaration contains at least one sparse parameter port.
    const ENABLED: bool;
}

/// Marker for a declaration without sparse ndarray parameters.
#[doc(hidden)]
pub struct NoDeclaredParameters;

/// Marker for a declaration with sparse ndarray parameters.
#[doc(hidden)]
pub struct DeclaredParameters;

impl DeclarationPropertyPolicy for NoDeclaredProperties {
    const POLICY: PropertyPolicy = PropertyPolicy::None;
}

impl DeclarationPropertyPolicy for RuntimeDeclaredProperties {
    const POLICY: PropertyPolicy = PropertyPolicy::Runtime;
}

impl DeclarationParameterPolicy for NoDeclaredParameters {
    const ENABLED: bool = false;
}

impl DeclarationParameterPolicy for DeclaredParameters {
    const ENABLED: bool = true;
}

/// Update behavior of one construction-configuration field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigurationUpdate {
    /// The value is used only while preparing one algorithm instance.
    Construction,
    /// Changing the value changes a port shape and requires graph reconstruction.
    GraphRebuild,
}

/// One configuration scalar applied uniformly or one explicit value per
/// scientific coordinate.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(untagged)]
pub enum ScalarOrVector<T> {
    /// Repeat one value over the declared extent.
    Scalar(T),
    /// Use one value for each coordinate in declaration order.
    Vector(Vec<T>),
}

/// An explicit configuration vector has the wrong scientific extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScalarOrVectorExtentError {
    expected: usize,
    actual: usize,
}

impl ScalarOrVectorExtentError {
    /// Return the required scientific extent.
    pub const fn expected(&self) -> usize {
        self.expected
    }

    /// Return the supplied vector extent.
    pub const fn actual(&self) -> usize {
        self.actual
    }
}

impl fmt::Display for ScalarOrVectorExtentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "configuration vector extent {} does not match {}",
            self.actual, self.expected
        )
    }
}

impl core::error::Error for ScalarOrVectorExtentError {}

impl<T> ScalarOrVector<T>
where
    T: Clone,
{
    /// Expand the scalar or validate and return the explicit vector.
    pub fn expand(self, extent: usize) -> Result<Vec<T>, ScalarOrVectorExtentError> {
        match self {
            Self::Scalar(value) => Ok(vec![value; extent]),
            Self::Vector(values) if values.len() == extent => Ok(values),
            Self::Vector(values) => Err(ScalarOrVectorExtentError {
                expected: extent,
                actual: values.len(),
            }),
        }
    }
}

/// A transport-neutral value returned by cold configuration inspection.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(untagged)]
pub enum ConfigurationValue {
    /// A Boolean value.
    Bool(bool),
    /// A signed integer.
    I64(i64),
    /// An unsigned integer.
    U64(u64),
    /// A floating-point value.
    F64(f64),
    /// A UTF-8 string.
    String(alloc::string::String),
    /// An ordered tuple or vector.
    Sequence(Vec<ConfigurationValue>),
    /// An explicit absent optional value.
    None,
}

macro_rules! configuration_value_from {
    ($variant:ident, $target:ty; $($type:ty),+) => {$ (
        impl From<$type> for ConfigurationValue {
            fn from(value: $type) -> Self { Self::$variant(value as $target) }
        }
    )+};
}
configuration_value_from!(I64, i64; i8, i16, i32, i64, isize);
configuration_value_from!(U64, u64; u8, u16, u32, u64, usize);
configuration_value_from!(F64, f64; f32, f64);
impl From<bool> for ConfigurationValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}
impl From<alloc::string::String> for ConfigurationValue {
    fn from(value: alloc::string::String) -> Self {
        Self::String(value)
    }
}
impl<T: Into<ConfigurationValue>> From<Option<T>> for ConfigurationValue {
    fn from(value: Option<T>) -> Self {
        value.map(Into::into).unwrap_or(Self::None)
    }
}
impl<T: Into<ConfigurationValue>> From<Vec<T>> for ConfigurationValue {
    fn from(value: Vec<T>) -> Self {
        Self::Sequence(value.into_iter().map(Into::into).collect())
    }
}
impl<T: Into<ConfigurationValue>, const N: usize> From<[T; N]> for ConfigurationValue {
    fn from(value: [T; N]) -> Self {
        Self::Sequence(value.into_iter().map(Into::into).collect())
    }
}
impl<A: Into<ConfigurationValue>, B: Into<ConfigurationValue>> From<(A, B)> for ConfigurationValue {
    fn from(value: (A, B)) -> Self {
        Self::Sequence(vec![value.0.into(), value.1.into()])
    }
}

/// Missing and explicitly supplied fields remain distinct, including `None`.
#[doc(hidden)]
pub enum ConfigurationField<T> {
    Missing,
    Present(T),
}
impl<T> Default for ConfigurationField<T> {
    fn default() -> Self {
        Self::Missing
    }
}
impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for ConfigurationField<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(Self::Present)
    }
}

/// One typed construction-configuration field.
#[derive(Clone, Copy, Debug)]
pub struct AlgorithmConfiguration {
    name: &'static str,
    rust_type: &'static str,
    update: ConfigurationUpdate,
    default: Option<fn() -> ConfigurationValue>,
}

impl PartialEq for AlgorithmConfiguration {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.rust_type == other.rust_type
            && self.update == other.update
            && match (self.default, other.default) {
                (None, None) => true,
                (Some(left), Some(right)) => core::ptr::fn_addr_eq(left, right),
                _ => false,
            }
    }
}
impl Eq for AlgorithmConfiguration {}

impl AlgorithmConfiguration {
    /// Construct one required field descriptor.
    pub const fn new(
        name: &'static str,
        rust_type: &'static str,
        update: ConfigurationUpdate,
    ) -> Self {
        Self {
            name,
            rust_type,
            update,
            default: None,
        }
    }
    /// Attach the same typed default used by configuration deserialization.
    pub const fn with_default(mut self, default: fn() -> ConfigurationValue) -> Self {
        self.default = Some(default);
        self
    }
    /// Return the canonical unqualified field name.
    pub const fn name(&self) -> &'static str {
        self.name
    }
    /// Return the declared Rust field type for authoring inspection.
    pub const fn rust_type(&self) -> &'static str {
        self.rust_type
    }
    /// Return the field's update behavior.
    pub const fn update(&self) -> ConfigurationUpdate {
        self.update
    }
    /// Return whether callers must supply this field, including optional values.
    pub const fn required(&self) -> bool {
        self.default.is_none()
    }
    /// Evaluate the declared default outside processing; `None` means required.
    pub fn default_value(&self) -> Option<ConfigurationValue> {
        self.default.map(|default| default())
    }
}

/// One statically declared algorithm port.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlgorithmPort {
    name: &'static str,
    direction: PortDirection,
    role: PortRole,
    element_type: ElementType,
    metadata_source: Option<&'static str>,
    publication: PortPublication,
    required_sample_metadata: &'static [&'static str],
    optional_sample_metadata: &'static [&'static str],
}

impl AlgorithmPort {
    /// Construct one port descriptor.
    pub const fn new(
        name: &'static str,
        direction: PortDirection,
        role: PortRole,
        element_type: ElementType,
        metadata_source: Option<&'static str>,
    ) -> Self {
        Self {
            name,
            direction,
            role,
            element_type,
            metadata_source,
            publication: PortPublication::Always,
            required_sample_metadata: &[],
            optional_sample_metadata: &[],
        }
    }

    /// Mark this output as conditionally published.
    ///
    /// Declaration admission rejects this setting on an input port.
    pub const fn conditional(mut self) -> Self {
        self.publication = PortPublication::Conditional;
        self
    }

    /// Declare scientific facts required on this input, using the JFG vocabulary.
    pub const fn with_required_sample_metadata(
        mut self,
        requirements: &'static [&'static str],
    ) -> Self {
        assert!(
            matches!(self.direction, PortDirection::Input) && matches!(self.role, PortRole::Data),
            "sample metadata facts belong to data inputs"
        );
        self.required_sample_metadata = requirements;
        self
    }

    /// Declare scientific facts consumed when present on this data input.
    pub const fn with_optional_sample_metadata(mut self, facts: &'static [&'static str]) -> Self {
        assert!(
            matches!(self.direction, PortDirection::Input) && matches!(self.role, PortRole::Data),
            "sample metadata facts belong to data inputs"
        );
        self.optional_sample_metadata = facts;
        self
    }

    /// Return scientific sample facts consumed when present on this input.
    pub const fn optional_sample_metadata(&self) -> &'static [&'static str] {
        self.optional_sample_metadata
    }

    /// Return scientific sample facts required by this port.
    pub const fn required_sample_metadata(&self) -> &'static [&'static str] {
        self.required_sample_metadata
    }

    /// Return the canonical unqualified port name.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Return the port direction.
    pub const fn direction(&self) -> PortDirection {
        self.direction
    }

    /// Return whether the port is per-frame data or a sparse parameter.
    pub const fn role(&self) -> PortRole {
        self.role
    }

    /// Return the ndarray element type.
    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }

    /// Return the input whose sample metadata is propagated to this output.
    pub const fn metadata_source(&self) -> Option<&'static str> {
        self.metadata_source
    }

    /// Return whether every successful call publishes this output.
    pub const fn publication(&self) -> PortPublication {
        self.publication
    }
}

/// Invalid exact ndarray format prepared from an algorithm declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FormatError {
    /// A format has no dimensions or exceeds the portable rank limit.
    Rank,
    /// A dimension is zero or cannot be represented by the portable ABI.
    Dimension,
    /// The complete element or byte extent overflows the portable ABI.
    Extent,
    /// The scientific schema is empty.
    Schema,
    /// A port rate ratio is zero or cannot be represented.
    Rate,
}

impl fmt::Display for FormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rank => formatter.write_str("ndarray rank must be in 1..=64"),
            Self::Dimension => formatter.write_str("ndarray dimensions must fit positive u32"),
            Self::Extent => formatter.write_str("ndarray byte extent must fit u32"),
            Self::Schema => formatter.write_str("scientific schema must be nonempty"),
            Self::Rate => formatter.write_str("port rate ratio must be positive"),
        }
    }
}

impl core::error::Error for FormatError {}

/// Exact packed row-major ndarray format prepared for one port.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AlgorithmFormat {
    element_type: ElementType,
    layout: ArrayLayout,
    shape: Box<[u32]>,
    schema: Box<str>,
    rate_numerator: u32,
    rate_denominator: u32,
}

impl AlgorithmFormat {
    /// Prepare one exact packed row-major format.
    pub fn packed<T>(shape: &[usize], schema: impl Into<Box<str>>) -> Result<Self, FormatError>
    where
        T: NdarrayElement,
    {
        Self::packed_with_layout::<T>(shape, ArrayLayout::RowMajor, schema)
    }

    /// Prepare one exact packed format with an explicit storage order.
    pub fn packed_with_layout<T>(
        shape: &[usize],
        layout: ArrayLayout,
        schema: impl Into<Box<str>>,
    ) -> Result<Self, FormatError>
    where
        T: NdarrayElement,
    {
        if shape.is_empty() || shape.len() > MAX_NDARRAY_DIMENSIONS {
            return Err(FormatError::Rank);
        }
        let schema = schema.into();
        if !valid_scientific_schema(&schema) {
            return Err(FormatError::Schema);
        }
        let mut converted = alloc::vec::Vec::with_capacity(shape.len());
        let mut elements = 1_usize;
        for &dimension in shape {
            let dimension = u32::try_from(dimension).map_err(|_| FormatError::Dimension)?;
            if dimension == 0 {
                return Err(FormatError::Dimension);
            }
            elements = elements
                .checked_mul(dimension as usize)
                .ok_or(FormatError::Extent)?;
            converted.push(dimension);
        }
        elements
            .checked_mul(T::ELEMENT_TYPE.size())
            .filter(|&bytes| bytes <= u32::MAX as usize)
            .ok_or(FormatError::Extent)?;
        Ok(Self {
            element_type: T::ELEMENT_TYPE,
            layout,
            shape: converted.into_boxed_slice(),
            schema,
            rate_numerator: 1,
            rate_denominator: 1,
        })
    }

    /// Set the exact port cadence relative to the deployment base rate.
    pub fn with_rate_ratio(
        mut self,
        numerator: u32,
        denominator: u32,
    ) -> Result<Self, FormatError> {
        if numerator == 0 || denominator == 0 {
            return Err(FormatError::Rate);
        }
        let divisor = greatest_common_divisor(numerator, denominator);
        self.rate_numerator = numerator / divisor;
        self.rate_denominator = denominator / divisor;
        Ok(self)
    }

    /// Return the ndarray element type.
    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }

    /// Return the exact packed storage order.
    pub const fn layout(&self) -> ArrayLayout {
        self.layout
    }

    /// Return the exact row-major shape.
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    /// Return the versioned scientific schema.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Return the exact port cadence relative to the deployment base rate.
    pub const fn rate_ratio(&self) -> (u32, u32) {
        (self.rate_numerator, self.rate_denominator)
    }

    /// Return the exact number of elements.
    pub fn elements(&self) -> usize {
        self.shape.iter().fold(1_usize, |elements, &dimension| {
            elements * dimension as usize
        })
    }
}

const fn greatest_common_divisor(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn valid_scientific_schema(schema: &str) -> bool {
    if schema.is_empty()
        || schema.len() > MAX_SCIENTIFIC_SCHEMA_BYTES
        || !schema.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return false;
    }
    let Some((name, version)) = schema.rsplit_once('/') else {
        return false;
    };
    !name.is_empty()
        && !version.is_empty()
        && version.bytes().all(|byte| byte.is_ascii_digit())
        && version.parse::<u32>().is_ok_and(|version| version > 0)
}

/// Source-reported electrical trigger classification, independent of optical role.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum SampleTriggerState {
    /// A present electrical record explicitly reports no classification.
    #[default]
    Absent = 0,
    /// The source reported a low trigger level.
    Low = 1,
    /// The source reported a high trigger level.
    High = 2,
    /// Both levels occurred during the acquisition.
    Mixed = 3,
}

/// Clock reference for a sample-associated timestamp.
///
/// A clock label does not assert synchronization or accuracy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum SampleClock {
    /// Acquisition-host monotonic clock.
    Monotonic = 1,
    /// Unix-epoch realtime clock.
    Realtime = 2,
    /// Unix-epoch TAI clock.
    Tai = 3,
    /// Stream or model clock.
    Model = 4,
}

/// Exposure start in the specified clock, independent of exposure duration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampleExposureStart {
    nanoseconds: i64,
    clock: SampleClock,
}
impl SampleExposureStart {
    /// Construct an exposure start. Signed stream/model times are permitted.
    pub const fn new(nanoseconds: i64, clock: SampleClock) -> Self {
        Self { nanoseconds, clock }
    }
    /// Exposure start in nanoseconds.
    pub const fn nanoseconds(self) -> i64 {
        self.nanoseconds
    }
    /// Clock qualifying this timestamp.
    pub const fn clock(self) -> SampleClock {
        self.clock
    }
}

/// A reference timestamp or duration used the native unspecified-value sentinel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleReferenceTimestampError {
    /// UINT64_MAX is not a timestamp.
    Timestamp,
    /// Use None for an unspecified duration instead of UINT64_MAX.
    Duration,
}
impl fmt::Display for SampleReferenceTimestampError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid reference timestamp: {self:?}")
    }
}
impl core::error::Error for SampleReferenceTimestampError {}

/// One sample-associated reference timestamp and optional duration in nanoseconds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampleReferenceTimestamp {
    nanoseconds: u64,
    duration_nanoseconds: Option<u64>,
    clock: SampleClock,
}
impl SampleReferenceTimestamp {
    /// Construct a reference timestamp without losing unsigned nanosecond precision.
    pub const fn new(
        nanoseconds: u64,
        duration_nanoseconds: Option<u64>,
        clock: SampleClock,
    ) -> Result<Self, SampleReferenceTimestampError> {
        if nanoseconds == u64::MAX {
            return Err(SampleReferenceTimestampError::Timestamp);
        }
        if matches!(duration_nanoseconds, Some(u64::MAX)) {
            return Err(SampleReferenceTimestampError::Duration);
        }
        Ok(Self {
            nanoseconds,
            duration_nanoseconds,
            clock,
        })
    }
    /// Exact unsigned reference timestamp in nanoseconds.
    pub const fn nanoseconds(self) -> u64 {
        self.nanoseconds
    }
    /// Reference duration; None maps to the native UINT64_MAX sentinel.
    pub const fn duration_nanoseconds(self) -> Option<u64> {
        self.duration_nanoseconds
    }
    /// Clock qualifying this timestamp.
    pub const fn clock(self) -> SampleClock {
        self.clock
    }
}

/// Physical acquisition identity within a source cohort.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampleAcquisitionIdentity {
    domain: [u8; 16],
    generation: u64,
    sequence: u64,
}

/// An acquisition domain must contain at least one nonzero byte.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleAcquisitionIdentityError {
    /// An acquisition domain cannot be all zeros.
    ZeroDomain,
}
impl fmt::Display for SampleAcquisitionIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ZeroDomain => "acquisition domain must contain a nonzero byte",
        })
    }
}
impl core::error::Error for SampleAcquisitionIdentityError {}

impl SampleAcquisitionIdentity {
    /// Validate and construct a physical acquisition identity.
    pub fn new(
        domain: [u8; 16],
        generation: u64,
        sequence: u64,
    ) -> Result<Self, SampleAcquisitionIdentityError> {
        if domain.iter().all(|byte| *byte == 0) {
            return Err(SampleAcquisitionIdentityError::ZeroDomain);
        }
        Ok(Self {
            domain,
            generation,
            sequence,
        })
    }

    /// Return the source domain.
    pub const fn domain(self) -> [u8; 16] {
        self.domain
    }
    /// Return the source generation.
    pub const fn generation(self) -> u64 {
        self.generation
    }
    /// Return the physical acquisition sequence.
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
}

/// Transport-neutral identity and state of one complete input or output sample.
///
/// The absence of this value means that the native caller or deployment did
/// not supply sample metadata. A deployment adapter maps these fields to its
/// standard metadata representation; scientific code does not depend on that
/// representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampleMetadata {
    sequence: u64,
    offset: u32,
    timestamp_nanoseconds: Option<i64>,
    acquisition_identity: Option<SampleAcquisitionIdentity>,
    chopped_image_kind: Option<crate::ChoppedImageKind>,
    trigger_state: Option<SampleTriggerState>,
    exposure_duration_nanoseconds: Option<core::num::NonZeroU64>,
    exposure_start: Option<SampleExposureStart>,
    reference_timestamp: Option<SampleReferenceTimestamp>,
    terminal: bool,
    discontinuity: bool,
    corrupted: bool,
}

impl SampleMetadata {
    /// Construct one sample identity with no timestamp or flags.
    pub const fn new(sequence: u64, offset: u32) -> Self {
        Self {
            sequence,
            offset,
            timestamp_nanoseconds: None,
            acquisition_identity: None,
            chopped_image_kind: None,
            trigger_state: None,
            exposure_duration_nanoseconds: None,
            exposure_start: None,
            reference_timestamp: None,
            terminal: false,
            discontinuity: false,
            corrupted: false,
        }
    }

    /// Return the publication sequence shared by related sample pieces.
    ///
    /// This can differ from the original physical acquisition sequence.
    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    /// Return the zero-based semantic offset of this sample piece.
    pub const fn offset(self) -> u32 {
        self.offset
    }

    /// Return a qualified physical exposure start, when supplied.
    pub const fn exposure_start(self) -> Option<SampleExposureStart> {
        self.exposure_start
    }
    /// Replace the exposure start independently of exposure duration.
    pub const fn with_exposure_start(mut self, start: Option<SampleExposureStart>) -> Self {
        self.exposure_start = start;
        self
    }
    /// Return the sample-associated reference timestamp, when supplied.
    pub const fn reference_timestamp(self) -> Option<SampleReferenceTimestamp> {
        self.reference_timestamp
    }
    /// Replace the sample-associated reference timestamp.
    pub const fn with_reference_timestamp(
        mut self,
        reference: Option<SampleReferenceTimestamp>,
    ) -> Self {
        self.reference_timestamp = reference;
        self
    }
    /// Return the electrical record classification; None means no record fact.
    pub const fn trigger_state(self) -> Option<SampleTriggerState> {
        self.trigger_state
    }
    /// Return a pure electrical trigger level, or None for absent or mixed levels.
    pub const fn trigger_high(self) -> Option<bool> {
        match self.trigger_state {
            Some(SampleTriggerState::Low) => Some(false),
            Some(SampleTriggerState::High) => Some(true),
            _ => None,
        }
    }
    /// Replace the independently optional electrical trigger classification.
    pub const fn with_trigger_state(mut self, state: Option<SampleTriggerState>) -> Self {
        self.trigger_state = state;
        self
    }

    /// Return the physical acquisition identity, when supplied.
    pub const fn acquisition_identity(self) -> Option<SampleAcquisitionIdentity> {
        self.acquisition_identity
    }
    /// Return the application-assigned optical role, when supplied.
    pub const fn chopped_image_kind(self) -> Option<crate::ChoppedImageKind> {
        self.chopped_image_kind
    }
    /// Return the positive exposure duration, when supplied.
    pub const fn exposure_duration_nanoseconds(self) -> Option<u64> {
        match self.exposure_duration_nanoseconds {
            Some(value) => Some(value.get()),
            None => None,
        }
    }
    /// Replace the original physical identity independently of publication sequence.
    pub const fn with_acquisition_identity(
        mut self,
        identity: Option<SampleAcquisitionIdentity>,
    ) -> Self {
        self.acquisition_identity = identity;
        self
    }
    /// Replace the application-assigned optical role.
    pub const fn with_chopped_image_kind(mut self, kind: Option<crate::ChoppedImageKind>) -> Self {
        self.chopped_image_kind = kind;
        self
    }
    /// Replace the positive exposure duration.
    pub const fn with_exposure_duration_nanoseconds(
        mut self,
        duration: Option<core::num::NonZeroU64>,
    ) -> Self {
        self.exposure_duration_nanoseconds = duration;
        self
    }

    /// Return the presentation or completion timestamp when known.
    pub const fn timestamp_nanoseconds(self) -> Option<i64> {
        self.timestamp_nanoseconds
    }

    /// Return whether this piece terminates its larger scientific sample.
    pub const fn is_terminal(self) -> bool {
        self.terminal
    }

    /// Return whether this sample follows a discontinuity.
    pub const fn is_discontinuous(self) -> bool {
        self.discontinuity
    }

    /// Return whether acquisition marked this sample corrupt.
    pub const fn is_corrupted(self) -> bool {
        self.corrupted
    }

    /// Replace the semantic offset.
    pub const fn with_offset(mut self, offset: u32) -> Self {
        self.offset = offset;
        self
    }

    /// Replace the optional timestamp.
    pub const fn with_timestamp_nanoseconds(mut self, timestamp: Option<i64>) -> Self {
        self.timestamp_nanoseconds = timestamp;
        self
    }

    /// Replace the terminal indication.
    pub const fn with_terminal(mut self, terminal: bool) -> Self {
        self.terminal = terminal;
        self
    }

    /// Replace the discontinuity indication.
    pub const fn with_discontinuity(mut self, discontinuity: bool) -> Self {
        self.discontinuity = discontinuity;
        self
    }

    /// Replace the corruption indication.
    pub const fn with_corrupted(mut self, corrupted: bool) -> Self {
        self.corrupted = corrupted;
        self
    }
}

/// Publication outcome of one successful portable declaration call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgorithmProcessResult {
    /// Every declared output contains one complete artifact.
    Complete {
        /// Optional metadata that replaces propagated metadata on all outputs.
        metadata: Option<SampleMetadata>,
    },
    /// Conditional outputs remain logically unpublished for this call.
    Deferred,
}

impl AlgorithmProcessResult {
    /// Construct the ordinary one-call/one-artifact outcome.
    pub const fn complete() -> Self {
        Self::Complete { metadata: None }
    }

    /// Construct a complete outcome with replacement output metadata.
    pub const fn complete_with_metadata(metadata: SampleMetadata) -> Self {
        Self::Complete {
            metadata: Some(metadata),
        }
    }

    /// Construct a successful call that does not publish an artifact yet.
    pub const fn deferred() -> Self {
        Self::Deferred
    }

    /// Return whether this call completed its declared outputs.
    pub const fn is_complete(self) -> bool {
        matches!(self, Self::Complete { .. })
    }

    /// Return replacement output metadata for a completed call.
    pub const fn metadata(self) -> Option<SampleMetadata> {
        match self {
            Self::Complete { metadata } => metadata,
            Self::Deferred => None,
        }
    }
}

/// Convert an ordinary or conditional process body into its publication result.
#[doc(hidden)]
pub trait IntoAlgorithmProcessResult {
    /// Convert the process body's successful value.
    fn into_algorithm_process_result(self) -> AlgorithmProcessResult;
}

impl IntoAlgorithmProcessResult for () {
    fn into_algorithm_process_result(self) -> AlgorithmProcessResult {
        AlgorithmProcessResult::complete()
    }
}

impl IntoAlgorithmProcessResult for AlgorithmProcessResult {
    fn into_algorithm_process_result(self) -> AlgorithmProcessResult {
        self
    }
}

/// A typed borrowed input supplied to native declaration execution.
#[derive(Clone, Copy, Debug)]
pub enum AlgorithmInput<'a> {
    /// Boolean values represented as BOOL8 by C adapters.
    Bool8(&'a [bool]),
    /// Signed 8-bit integers.
    I8(&'a [i8]),
    /// Unsigned 8-bit integers.
    U8(&'a [u8]),
    /// Signed 16-bit integers.
    I16(&'a [i16]),
    /// Unsigned 16-bit integers.
    U16(&'a [u16]),
    /// Signed 32-bit integers.
    I32(&'a [i32]),
    /// Unsigned 32-bit integers.
    U32(&'a [u32]),
    /// Signed 64-bit integers.
    I64(&'a [i64]),
    /// Unsigned 64-bit integers.
    U64(&'a [u64]),
    /// IEEE 754 binary32 values.
    F32(&'a [f32]),
    /// IEEE 754 binary64 values.
    F64(&'a [f64]),
}

/// A typed borrowed output supplied to native declaration execution.
#[derive(Debug)]
pub enum AlgorithmOutput<'a> {
    /// Boolean values represented as BOOL8 by C adapters.
    Bool8(Option<&'a mut [bool]>),
    /// Signed 8-bit integers.
    I8(Option<&'a mut [i8]>),
    /// Unsigned 8-bit integers.
    U8(Option<&'a mut [u8]>),
    /// Signed 16-bit integers.
    I16(Option<&'a mut [i16]>),
    /// Unsigned 16-bit integers.
    U16(Option<&'a mut [u16]>),
    /// Signed 32-bit integers.
    I32(Option<&'a mut [i32]>),
    /// Unsigned 32-bit integers.
    U32(Option<&'a mut [u32]>),
    /// Signed 64-bit integers.
    I64(Option<&'a mut [i64]>),
    /// Unsigned 64-bit integers.
    U64(Option<&'a mut [u64]>),
    /// IEEE 754 binary32 values.
    F32(Option<&'a mut [f32]>),
    /// IEEE 754 binary64 values.
    F64(Option<&'a mut [f64]>),
}

mod sealed {
    pub trait DeclarationParameterPolicy {}

    impl DeclarationParameterPolicy for super::NoDeclaredParameters {}
    impl DeclarationParameterPolicy for super::DeclaredParameters {}

    pub trait DeclarationPropertyPolicy {}

    impl DeclarationPropertyPolicy for super::NoDeclaredProperties {}
    impl DeclarationPropertyPolicy for super::RuntimeDeclaredProperties {}

    pub trait NdarrayElement {}

    impl NdarrayElement for bool {}
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
}

/// Primitive storage supported by portable algorithm ndarray ports.
pub trait NdarrayElement: sealed::NdarrayElement + Copy + Sized + 'static {
    /// Portable element representation.
    const ELEMENT_TYPE: ElementType;

    #[doc(hidden)]
    fn input(array: AlgorithmInput<'_>) -> Option<&[Self]>;

    #[doc(hidden)]
    fn take_output<'a>(array: &mut AlgorithmOutput<'a>) -> Option<&'a mut [Self]>;
}

macro_rules! ndarray_elements {
    ($($type:ty => $variant:ident, $element:ident);+ $(;)?) => {
        $(
            impl NdarrayElement for $type {
                const ELEMENT_TYPE: ElementType = ElementType::$element;

                fn input(array: AlgorithmInput<'_>) -> Option<&[Self]> {
                    match array {
                        AlgorithmInput::$variant(values) => Some(values),
                        _ => None,
                    }
                }

                fn take_output<'a>(array: &mut AlgorithmOutput<'a>) -> Option<&'a mut [Self]> {
                    match array {
                        AlgorithmOutput::$variant(values) => values.take(),
                        _ => None,
                    }
                }
            }

            impl<'a> From<&'a [$type]> for AlgorithmInput<'a> {
                fn from(values: &'a [$type]) -> Self {
                    Self::$variant(values)
                }
            }

            impl<'a> From<&'a mut [$type]> for AlgorithmOutput<'a> {
                fn from(values: &'a mut [$type]) -> Self {
                    Self::$variant(Some(values))
                }
            }
        )+
    };
}

ndarray_elements! {
    bool => Bool8, Bool8;
    i8 => I8, I8;
    u8 => U8, U8;
    i16 => I16, I16;
    u16 => U16, U16;
    i32 => I32, I32;
    u32 => U32, U32;
    i64 => I64, I64;
    u64 => U64, U64;
    f32 => F32, F32;
    f64 => F64, F64;
}

/// Invalid typed port binding during native or adapter execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortError {
    /// Fewer ports were supplied than the declaration requires.
    Missing,
    /// A supplied port has the wrong element representation.
    ElementType,
    /// A borrowed buffer has a different element count from its prepared format.
    Extent {
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// More ports were supplied than the declaration accepts.
    Extra,
    /// Sample metadata did not contain one entry per data input.
    MetadataExtent,
}

impl fmt::Display for PortError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => formatter.write_str("a declared algorithm port is missing"),
            Self::ElementType => formatter.write_str("algorithm port element type does not match"),
            Self::Extent { expected, actual } => write!(
                formatter,
                "algorithm port extent {actual} does not match prepared extent {expected}"
            ),
            Self::Extra => formatter.write_str("undeclared algorithm ports were supplied"),
            Self::MetadataExtent => {
                formatter.write_str("sample metadata extent does not match algorithm inputs")
            }
        }
    }
}

impl core::error::Error for PortError {}

// Prepared native bindings validate all buffers before the scientific closure runs.
// Filtering by role keeps sparse parameters out of the per-sample data port order.
struct PortFormats<'a> {
    ports: &'a [AlgorithmPort],
    formats: &'a [AlgorithmFormat],
    direction: PortDirection,
    next: usize,
}
impl PortFormats<'_> {
    fn validate_next(&mut self, actual: usize) -> Result<(), PortError> {
        while let Some(port) = self.ports.get(self.next) {
            let index = self.next;
            self.next += 1;
            if port.direction() == self.direction && port.role() == PortRole::Data {
                let expected = self
                    .formats
                    .get(index)
                    .ok_or(PortError::Missing)?
                    .elements();
                return if actual == expected {
                    Ok(())
                } else {
                    Err(PortError::Extent { expected, actual })
                };
            }
        }
        Err(PortError::Missing)
    }
}

/// Ordered typed inputs for one portable algorithm call.
pub struct AlgorithmInputs<'ports, 'data> {
    values: &'ports [AlgorithmInput<'data>],
    metadata: Option<&'ports [Option<SampleMetadata>]>,
    formats: Option<PortFormats<'ports>>,
    next: usize,
}

impl<'ports, 'data> AlgorithmInputs<'ports, 'data> {
    /// Borrow one ordered input collection.
    pub const fn new(values: &'ports [AlgorithmInput<'data>]) -> Self {
        Self {
            values,
            metadata: None,
            formats: None,
            next: 0,
        }
    }

    /// Borrow inputs and direction-local optional sample metadata.
    pub fn with_metadata(
        values: &'ports [AlgorithmInput<'data>],
        metadata: &'ports [Option<SampleMetadata>],
    ) -> Result<Self, PortError> {
        if values.len() != metadata.len() {
            return Err(PortError::MetadataExtent);
        }
        Ok(Self {
            values,
            metadata: Some(metadata),
            formats: None,
            next: 0,
        })
    }

    fn with_formats(
        mut self,
        ports: &'ports [AlgorithmPort],
        formats: &'ports [AlgorithmFormat],
    ) -> Self {
        self.formats = Some(PortFormats {
            ports,
            formats,
            direction: PortDirection::Input,
            next: 0,
        });
        self
    }

    /// Take the next declared input with its exact element representation.
    pub fn take<T>(&mut self) -> Result<&'data [T], PortError>
    where
        T: NdarrayElement,
    {
        let value = self
            .values
            .get(self.next)
            .copied()
            .ok_or(PortError::Missing)?;
        self.next += 1;
        let values = T::input(value).ok_or(PortError::ElementType)?;
        if let Some(formats) = &mut self.formats {
            formats.validate_next(values.len())?;
        }
        Ok(values)
    }

    /// Take the next typed input together with its optional sample metadata.
    pub fn take_sample<T>(&mut self) -> Result<(&'data [T], Option<SampleMetadata>), PortError>
    where
        T: NdarrayElement,
    {
        let index = self.next;
        let values = self.take::<T>()?;
        let metadata = self
            .metadata
            .and_then(|metadata| metadata.get(index).copied())
            .flatten();
        Ok((values, metadata))
    }

    /// Reject an input collection containing undeclared trailing ports.
    pub fn finish(self) -> Result<(), PortError> {
        (self.next == self.values.len()
            && self
                .metadata
                .is_none_or(|metadata| metadata.len() == self.values.len()))
        .then_some(())
        .ok_or(PortError::Extra)
    }
}

/// Ordered typed outputs for one portable algorithm call.
pub struct AlgorithmOutputs<'ports, 'data> {
    remaining: &'ports mut [AlgorithmOutput<'data>],
    formats: Option<PortFormats<'ports>>,
}

impl<'ports, 'data> AlgorithmOutputs<'ports, 'data> {
    /// Borrow one ordered output collection.
    pub fn new(values: &'ports mut [AlgorithmOutput<'data>]) -> Self {
        Self {
            remaining: values,
            formats: None,
        }
    }

    fn with_formats(
        mut self,
        ports: &'ports [AlgorithmPort],
        formats: &'ports [AlgorithmFormat],
    ) -> Self {
        self.formats = Some(PortFormats {
            ports,
            formats,
            direction: PortDirection::Output,
            next: 0,
        });
        self
    }

    /// Take the next declared output with its exact element representation.
    pub fn take<T>(&mut self) -> Result<&'data mut [T], PortError>
    where
        T: NdarrayElement,
    {
        let remaining = mem::take(&mut self.remaining);
        let (value, rest) = remaining.split_first_mut().ok_or(PortError::Missing)?;
        self.remaining = rest;
        let values = T::take_output(value).ok_or(PortError::ElementType)?;
        if let Some(formats) = &mut self.formats {
            formats.validate_next(values.len())?;
        }
        Ok(values)
    }

    /// Reject an output collection containing undeclared trailing ports.
    pub fn finish(self) -> Result<(), PortError> {
        self.remaining
            .is_empty()
            .then_some(())
            .ok_or(PortError::Extra)
    }
}

/// Preparation failure introduced by a portable declaration boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeclarationPreparationError<E> {
    /// Scientific plan or workspace preparation failed.
    Algorithm(E),
    /// A declared port format was invalid.
    Format(FormatError),
}

impl<E> fmt::Display for DeclarationPreparationError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Algorithm(error) => error.fmt(formatter),
            Self::Format(error) => error.fmt(formatter),
        }
    }
}

impl<E> core::error::Error for DeclarationPreparationError<E> where E: core::error::Error + 'static {}

/// Processing failure introduced by a portable declaration boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeclarationProcessError<E> {
    /// A supplied port did not match the declaration.
    Port(PortError),
    /// The scientific operation failed.
    Algorithm(E),
}

impl<E> From<PortError> for DeclarationProcessError<E> {
    fn from(error: PortError) -> Self {
        Self::Port(error)
    }
}

impl<E> fmt::Display for DeclarationProcessError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Port(error) => error.fmt(formatter),
            Self::Algorithm(error) => error.fmt(formatter),
        }
    }
}

impl<E> core::error::Error for DeclarationProcessError<E> where E: core::error::Error + 'static {}

/// Sparse-parameter preparation failure introduced by a declaration boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeclarationParameterError<E> {
    /// A transaction contained no parameter assignments.
    EmptyTransaction,
    /// The supplied declaration port is not a sparse parameter input.
    UnknownPort {
        /// Declaration-local port index.
        port: usize,
    },
    /// One transaction assigns the same parameter port more than once.
    DuplicatePort {
        /// Duplicated declaration-local port index.
        port: usize,
    },
    /// The supplied ndarray has the wrong element representation.
    Port(PortError),
    /// Scientific replacement-plan preparation failed.
    Algorithm(E),
}

impl<E> fmt::Display for DeclarationParameterError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTransaction => formatter.write_str("parameter transaction is empty"),
            Self::UnknownPort { port } => {
                write!(
                    formatter,
                    "declaration port {port} is not a parameter input"
                )
            }
            Self::DuplicatePort { port } => {
                write!(
                    formatter,
                    "parameter port {port} is assigned more than once"
                )
            }
            Self::Port(error) => error.fmt(formatter),
            Self::Algorithm(error) => error.fmt(formatter),
        }
    }
}

impl<E> core::error::Error for DeclarationParameterError<E> where E: core::error::Error + 'static {}

/// Native metadata binding or declared processing failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeProcessError<E> {
    /// Optional sample metadata did not align with the data inputs.
    Metadata(PortError),
    /// The declared operation rejected the call.
    Process(E),
}

impl<E> fmt::Display for NativeProcessError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metadata(error) => error.fmt(formatter),
            Self::Process(error) => error.fmt(formatter),
        }
    }
}

impl<E> core::error::Error for NativeProcessError<E> where E: core::error::Error + 'static {}

/// One transport-neutral graph declaration for an ordinary typed operation.
///
/// The prepared plan and workspace are independent associated types. This
/// keeps the graph boundary usable by operations that already implement
/// [`crate::AlgorithmPlan`] and by scientifically meaningful prepared kernels
/// that do not need that interface.
pub trait AlgorithmDeclaration: Sized + 'static {
    /// Typed construction configuration generated by the declaration.
    type Configuration;
    /// Scientific plan executed by this declaration.
    type Plan;
    /// Mutable storage owned by one prepared instance.
    type Workspace;
    /// Scientific plan or workspace preparation error.
    type PreparationError;
    /// Scientific processing error.
    type ProcessError;
    /// Type-level scalar-property capability used by deployment adapters.
    #[doc(hidden)]
    type ScalarProperties: DeclarationPropertyPolicy;
    /// Type-level sparse-parameter capability used by deployment adapters.
    #[doc(hidden)]
    type Parameters: DeclarationParameterPolicy;
    /// Scientific replacement-plan preparation error.
    type ParameterError;

    /// Stable descriptor label.
    const NAME: &'static str;
    /// Typed construction-configuration fields.
    const CONFIGURATION: &'static [AlgorithmConfiguration];
    /// Statically declared ports in descriptor order.
    const PORTS: &'static [AlgorithmPort];
    /// Scientific facts required on this prepared input port.
    ///
    /// The default uses the static declaration. Algorithms whose construction
    /// configuration changes admission requirements can override this cold query.
    /// Unknown port indices have no requirements.
    fn required_sample_metadata(_plan: &Self::Plan, port: usize) -> &'static [&'static str] {
        Self::PORTS
            .get(port)
            .map_or(&[], |port| port.required_sample_metadata())
    }

    /// Scalar-property behavior projected by deployment adapters.
    const PROPERTIES: PropertyPolicy = Self::ScalarProperties::POLICY;
    /// Optional mathematical operation shape recognized by deployment adapters.
    const OPERATION: AlgorithmOperation = AlgorithmOperation::General;

    /// Project the exact prepared plan selected by [`Self::OPERATION`].
    #[doc(hidden)]
    fn operation_plan(_plan: &Self::Plan) -> AlgorithmOperationPlan<'_> {
        AlgorithmOperationPlan::General
    }

    /// Project the exact mutable workspace selected by [`Self::OPERATION`].
    #[doc(hidden)]
    fn operation_workspace(_workspace: &mut Self::Workspace) -> AlgorithmOperationWorkspace<'_> {
        AlgorithmOperationWorkspace::General
    }

    /// Prepare one plan, compatible workspace, and exact port formats.
    fn prepare(
        configuration: Self::Configuration,
    ) -> Result<PreparedAlgorithm<Self>, Self::PreparationError>;

    /// Execute one complete operation through ordered typed array views.
    fn process(
        plan: &Self::Plan,
        workspace: &mut Self::Workspace,
        inputs: AlgorithmInputs<'_, '_>,
        outputs: AlgorithmOutputs<'_, '_>,
    ) -> Result<AlgorithmProcessResult, Self::ProcessError>;

    /// Reset persistent mathematical history at a serialized lifecycle boundary.
    fn reset(_plan: &Self::Plan, _workspace: &mut Self::Workspace) {}

    /// Reconcile workspace state after a complete replacement plan is adopted.
    ///
    /// The default preserves compatible persistent state. An operation that
    /// assembles a sample over several calls can abandon that partial sample
    /// here so one published result never mixes plan generations.
    fn plan_adopted(_plan: &Self::Plan, _workspace: &mut Self::Workspace) {}

    /// Prepare a complete replacement plan from one sparse ndarray parameter.
    fn prepare_parameter(
        _plan: &Self::Plan,
        _port: usize,
        _parameter: AlgorithmInput<'_>,
    ) -> Option<Result<Self::Plan, DeclarationParameterError<Self::ParameterError>>> {
        None
    }

    /// Failure-atomically prepare one plan from a nonempty parameter set.
    ///
    /// Assignments are applied in the supplied order to a private replacement
    /// plan. The active plan remains borrowed and unchanged if any assignment
    /// fails. Duplicate ports are rejected before scientific preparation.
    fn prepare_parameters(
        plan: &Self::Plan,
        parameters: &[(usize, AlgorithmInput<'_>)],
    ) -> Result<Self::Plan, DeclarationParameterError<Self::ParameterError>> {
        if parameters.is_empty() {
            return Err(DeclarationParameterError::EmptyTransaction);
        }
        for (index, (port, _)) in parameters.iter().enumerate() {
            if !Self::PORTS.get(*port).is_some_and(|candidate| {
                candidate.direction() == PortDirection::Input
                    && candidate.role() == PortRole::Parameter
            }) {
                return Err(DeclarationParameterError::UnknownPort { port: *port });
            }
            if parameters[..index]
                .iter()
                .any(|(previous, _)| previous == port)
            {
                return Err(DeclarationParameterError::DuplicatePort { port: *port });
            }
        }

        let mut replacement = None;
        for &(port, parameter) in parameters {
            let base = replacement.as_ref().unwrap_or(plan);
            replacement = Some(
                Self::prepare_parameter(base, port, parameter)
                    .ok_or(DeclarationParameterError::UnknownPort { port })??,
            );
        }
        replacement.ok_or(DeclarationParameterError::EmptyTransaction)
    }
}

/// Prepared native execution owner for one algorithm declaration.
pub struct PreparedAlgorithm<D>
where
    D: AlgorithmDeclaration,
{
    plan: D::Plan,
    workspace: D::Workspace,
    formats: Box<[AlgorithmFormat]>,
}

impl<D> PreparedAlgorithm<D>
where
    D: AlgorithmDeclaration,
{
    /// Group one admitted plan, compatible workspace, and exact port formats.
    pub fn new(plan: D::Plan, workspace: D::Workspace, formats: Box<[AlgorithmFormat]>) -> Self {
        Self {
            plan,
            workspace,
            formats,
        }
    }

    /// Return the prepared scientific plan.
    pub const fn plan(&self) -> &D::Plan {
        &self.plan
    }

    /// Return the compatible workspace.
    pub const fn workspace(&self) -> &D::Workspace {
        &self.workspace
    }

    /// Return exact formats in declaration order.
    pub fn formats(&self) -> &[AlgorithmFormat] {
        &self.formats
    }

    /// Return scientific facts required by one prepared input port.
    pub fn required_sample_metadata(&self, port: usize) -> &'static [&'static str] {
        D::required_sample_metadata(&self.plan, port)
    }

    /// Decompose the owner for a deployment adapter.
    pub fn into_parts(self) -> (D::Plan, D::Workspace, Box<[AlgorithmFormat]>) {
        (self.plan, self.workspace, self.formats)
    }

    /// Execute one complete native call using ordinary borrowed arrays.
    pub fn process(
        &mut self,
        inputs: &[AlgorithmInput<'_>],
        outputs: &mut [AlgorithmOutput<'_>],
    ) -> Result<AlgorithmProcessResult, D::ProcessError> {
        D::process(
            &self.plan,
            &mut self.workspace,
            AlgorithmInputs::new(inputs).with_formats(D::PORTS, &self.formats),
            AlgorithmOutputs::new(outputs).with_formats(D::PORTS, &self.formats),
        )
    }

    /// Execute one native call with optional metadata for every data input.
    pub fn process_with_metadata(
        &mut self,
        inputs: &[AlgorithmInput<'_>],
        metadata: &[Option<SampleMetadata>],
        outputs: &mut [AlgorithmOutput<'_>],
    ) -> Result<AlgorithmProcessResult, NativeProcessError<D::ProcessError>> {
        let inputs = AlgorithmInputs::with_metadata(inputs, metadata)
            .map_err(NativeProcessError::Metadata)?
            .with_formats(D::PORTS, &self.formats);
        D::process(
            &self.plan,
            &mut self.workspace,
            inputs,
            AlgorithmOutputs::new(outputs).with_formats(D::PORTS, &self.formats),
        )
        .map_err(NativeProcessError::Process)
    }

    /// Reset persistent mathematical history.
    pub fn reset(&mut self) {
        D::reset(&self.plan, &mut self.workspace);
    }

    /// Prepare and install a complete replacement plan from one sparse
    /// ndarray parameter.
    ///
    /// The caller performs this operation outside the processing path. The
    /// declaration guarantees that an accepted replacement preserves every
    /// prepared port format and remains compatible with the existing
    /// workspace.
    pub fn replace_parameter(
        &mut self,
        port: usize,
        parameter: AlgorithmInput<'_>,
    ) -> Option<Result<(), DeclarationParameterError<D::ParameterError>>> {
        D::prepare_parameter(&self.plan, port, parameter).map(|replacement| {
            replacement.map(|plan| {
                self.plan = plan;
                D::plan_adopted(&self.plan, &mut self.workspace);
            })
        })
    }

    /// Failure-atomically replace several sparse parameter artifacts.
    ///
    /// The final plan is installed only after every named port has produced a
    /// compatible replacement. This is the native equivalent of one graph
    /// parameter transaction and performs preparation outside processing.
    pub fn replace_parameters(
        &mut self,
        parameters: &[(usize, AlgorithmInput<'_>)],
    ) -> Result<(), DeclarationParameterError<D::ParameterError>> {
        self.plan = D::prepare_parameters(&self.plan, parameters)?;
        D::plan_adopted(&self.plan, &mut self.workspace);
        Ok(())
    }
}

impl<D> PreparedAlgorithm<D>
where
    D: AlgorithmDeclaration,
    D::Plan: AlgorithmProperties,
{
    /// Validate and install one complete scalar-property transaction.
    ///
    /// This native convenience performs no publication. Deployment bindings
    /// prepare and publish the same replacement plan through their bounded
    /// control/data-loop handoff.
    pub fn replace_properties(
        &mut self,
        assignments: &[PropertyAssignment<'_>],
    ) -> Result<(), <D::Plan as AlgorithmProperties>::PropertyError> {
        if assignments.is_empty() {
            return Ok(());
        }
        self.plan = self.plan.prepare_properties(assignments)?;
        D::plan_adopted(&self.plan, &mut self.workspace);
        Ok(())
    }
}

/// Declare one transport-neutral typed algorithm graph boundary.
#[macro_export]
macro_rules! filtergraph_algorithm {
    (
        $(#[$meta:meta])*
        $vis:vis $declaration:ident {
            label = $label:literal;
            configuration $configuration:ident {
                $($field:ident : $field_type:ty = $field_update:ident $(, default = $field_default:expr)?;)*
            }
            plan = $plan:ty;
            $(operation = $operation:ident;)?
            $(workspace = $workspace:ty;)?
            properties = $properties:ident;
            $(required_sample_metadata |$required_plan:ident, $required_port:ident| $required_facts:block)?
            dimensions |$format_plan:ident, $format_workspace:ident| {
                $($dimension:ident = $dimension_value:expr;)*
            }
            inputs {
                $($input:ident : $input_type:ty [$($input_shape:tt)*]
                    = $input_name:literal, schema = $input_schema:expr
                    $(, sample = $input_sample:ident)?
                    $(, required_sample_metadata = [$($required_sample_metadata:literal),* $(,)?])?
                    $(, optional_sample_metadata = [$($optional_sample_metadata:literal),* $(,)?])?
                    $(, rate = [$input_rate_numerator:expr, $input_rate_denominator:expr])?;)*
            }
            outputs {
                $($output:ident : $output_type:ty [$($output_shape:tt)*]
                    = $output_name:literal, schema = $output_schema:expr
                    $(, metadata_from = $metadata_source:literal)?
                    $(, publication = $publication:ident)?
                    $(, rate = [$output_rate_numerator:expr, $output_rate_denominator:expr])?;)*
            }
            $(
                parameters {
                    $(
                        $parameter:ident : $parameter_type:ty
                            [$($parameter_shape:expr),+ $(,)?]
                            = $parameter_name:literal, schema = $parameter_schema:expr,
                            replace |$replacement_plan:ident, $parameter_value:ident|
                                $replacement:block;
                    )+
                }
                parameter_error = $parameter_error:ty;
            )?
            prepare |$prepare_configuration:ident| $prepare:block
            prepare_error = $prepare_error:ty;
            process |$process_plan:ident, $process_workspace:ident| $process:block
            process_error = $process_error:ty;
            $(reset |$reset_plan:ident, $reset_workspace:ident| $reset:block)?
            $(plan_adopted |$adopted_plan:ident, $adopted_workspace:ident| $adopted:block)?
        }
    ) => {
        #[allow(missing_docs)]
        #[derive(Clone, Debug, PartialEq)]
        $vis struct $configuration {
            $(pub $field: $field_type,)*
        }

        impl<'de> $crate::__private::Deserialize<'de> for $configuration {
            fn deserialize<D: $crate::__private::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                #[derive($crate::__private::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Fields {
                    $(#[serde(default)] $field: $crate::__private::ConfigurationField<$field_type>,)*
                }
                let fields = Fields::deserialize(deserializer)?;
                Ok(Self {
                    $($field: match fields.$field {
                        $crate::__private::ConfigurationField::Present(value) => value,
                        $crate::__private::ConfigurationField::Missing =>
                            $crate::filtergraph_algorithm!(@missing D, $field $(, $field_default)?),
                    },)*
                })
            }
        }

        $(#[$meta])*
        $vis struct $declaration;

        impl $crate::AlgorithmDeclaration for $declaration {
            type Configuration = $configuration;
            type Plan = $plan;
            type Workspace = $crate::filtergraph_algorithm!(@workspace $plan $(, $workspace)?);
            type PreparationError = $crate::DeclarationPreparationError<$prepare_error>;
            type ProcessError = $crate::DeclarationProcessError<$process_error>;
            type ScalarProperties =
                $crate::filtergraph_algorithm!(@property_type $properties);
            type Parameters =
                $crate::filtergraph_algorithm!(@parameter_type $($parameter_error)?);
            type ParameterError =
                $crate::filtergraph_algorithm!(@parameter_error $($parameter_error)?);

            const NAME: &'static str = $label;
            const OPERATION: $crate::AlgorithmOperation =
                $crate::filtergraph_algorithm!(@operation $($operation)?);
            const CONFIGURATION: &'static [$crate::AlgorithmConfiguration] = &[
                $(
                    $crate::filtergraph_algorithm!(@configuration
                        $crate::AlgorithmConfiguration::new(
                            stringify!($field), stringify!($field_type),
                            $crate::ConfigurationUpdate::$field_update,
                        ), $field_type $(, $field_default)?
                    ),
                )*
            ];
            const PORTS: &'static [$crate::AlgorithmPort] = &[
                $(
                    $crate::AlgorithmPort::new(
                        $input_name,
                        $crate::PortDirection::Input,
                        $crate::PortRole::Data,
                        <$input_type as $crate::NdarrayElement>::ELEMENT_TYPE,
                        None,
                    ).with_required_sample_metadata(&[$($($required_sample_metadata),*)?])
                     .with_optional_sample_metadata(&[$($($optional_sample_metadata),*)?]),
                )*
                $(
                    $crate::filtergraph_algorithm!(@output_port
                        $crate::AlgorithmPort::new(
                            $output_name,
                            $crate::PortDirection::Output,
                            $crate::PortRole::Data,
                            <$output_type as $crate::NdarrayElement>::ELEMENT_TYPE,
                            $crate::filtergraph_algorithm!(@metadata_source $($metadata_source)?),
                        ) $(, $publication)?
                    ),
                )*
                $($(
                    $crate::AlgorithmPort::new(
                        $parameter_name,
                        $crate::PortDirection::Input,
                        $crate::PortRole::Parameter,
                        <$parameter_type as $crate::NdarrayElement>::ELEMENT_TYPE,
                        None,
                    ),
                )+)?
            ];

            $(fn required_sample_metadata(
                $required_plan: &Self::Plan,
                $required_port: usize,
            ) -> &'static [&'static str] {
                $required_facts
            })?

            fn operation_plan(
                plan: &Self::Plan,
            ) -> $crate::AlgorithmOperationPlan<'_> {
                $crate::filtergraph_algorithm!(@operation_plan plan $(, $operation)?)
            }

            fn operation_workspace(
                workspace: &mut Self::Workspace,
            ) -> $crate::AlgorithmOperationWorkspace<'_> {
                $crate::filtergraph_algorithm!(@operation_workspace workspace $(, $operation)?)
            }
            fn prepare(
                $prepare_configuration: Self::Configuration,
            ) -> Result<$crate::PreparedAlgorithm<Self>, Self::PreparationError> {
                let mut __filtergraph_schemas = [
                    $($crate::__private::Box::<str>::from($input_schema),)*
                    $($crate::__private::Box::<str>::from($output_schema),)*
                    $($($crate::__private::Box::<str>::from($parameter_schema),)*)?
                ].into_iter();
                let prepared: Result<
                    (
                        Self::Plan,
                        Self::Workspace,
                    ),
                    $prepare_error,
                > = $crate::__private::invoke(|| $prepare);
                let (plan, workspace) = prepared
                    .map_err($crate::DeclarationPreparationError::Algorithm)?;
                let $format_plan = &plan;
                let $format_workspace = &workspace;
                $(let $dimension = $dimension_value;)*
                let formats = $crate::__private::Box::new([
                    $(
                        $crate::filtergraph_algorithm!(@format $input_type,
                            $crate::filtergraph_algorithm!(@shape [$($input_shape)*]),
                            __filtergraph_schemas.next().expect("one schema per declared port"),
                            $($input_rate_numerator, $input_rate_denominator)?
                        ).map_err($crate::DeclarationPreparationError::Format)?,
                    )*
                    $(
                        $crate::filtergraph_algorithm!(@format $output_type,
                            $crate::filtergraph_algorithm!(@shape [$($output_shape)*]),
                            __filtergraph_schemas.next().expect("one schema per declared port"),
                            $($output_rate_numerator, $output_rate_denominator)?
                        ).map_err($crate::DeclarationPreparationError::Format)?,
                    )*
                    $($(
                        $crate::AlgorithmFormat::packed::<$parameter_type>(
                            &[$($parameter_shape),+],
                            __filtergraph_schemas.next().expect("one schema per declared port"),
                        ).map_err($crate::DeclarationPreparationError::Format)?,
                    )+)?
                ]);
                Ok($crate::PreparedAlgorithm::new(plan, workspace, formats))
            }

            #[allow(unused_mut)]
            fn process(
                $process_plan: &Self::Plan,
                $process_workspace: &mut Self::Workspace,
                mut inputs: $crate::AlgorithmInputs<'_, '_>,
                mut outputs: $crate::AlgorithmOutputs<'_, '_>,
            ) -> Result<$crate::AlgorithmProcessResult, Self::ProcessError> {
                $(
                    let ($input, __filtergraph_input_sample) =
                        inputs.take_sample::<$input_type>()?;
                    $(let $input_sample = __filtergraph_input_sample;)?
                )*
                inputs.finish()?;
                $(let $output = outputs.take::<$output_type>()?;)*
                outputs.finish()?;
                let result: Result<_, $process_error> =
                    $crate::__private::invoke(|| -> Result<_, $process_error> { $process });
                result
                    .map($crate::IntoAlgorithmProcessResult::into_algorithm_process_result)
                    .map_err($crate::DeclarationProcessError::Algorithm)
            }

            $(
                fn reset(
                    $reset_plan: &Self::Plan,
                    $reset_workspace: &mut Self::Workspace,
                ) {
                    $reset
                }
            )?

            $(
                fn plan_adopted(
                    $adopted_plan: &Self::Plan,
                    $adopted_workspace: &mut Self::Workspace,
                ) {
                    $adopted
                }
            )?

            $(
                fn prepare_parameter(
                    plan: &Self::Plan,
                    port: usize,
                    parameter: $crate::AlgorithmInput<'_>,
                ) -> Option<Result<
                    Self::Plan,
                    $crate::DeclarationParameterError<Self::ParameterError>,
                >> {
                    let mut declared_port = Self::PORTS
                        .iter()
                        .position(|port| port.role() == $crate::PortRole::Parameter)
                        .expect("a parameter-enabled declaration has a parameter port");
                    $(
                        if port == declared_port {
                            let $replacement_plan = plan;
                            let Some($parameter_value) =
                                <$parameter_type as $crate::NdarrayElement>::input(parameter)
                            else {
                                return Some(Err($crate::DeclarationParameterError::Port(
                                    $crate::PortError::ElementType,
                                )));
                            };
                            let replacement: Result<Self::Plan, Self::ParameterError> =
                                $crate::__private::invoke(|| $replacement);
                            return Some(replacement.map_err(
                                $crate::DeclarationParameterError::Algorithm,
                            ));
                        }
                        declared_port += 1;
                    )+
                    let _ = declared_port;
                    None
                }
            )?
        }

        impl $declaration {
            /// Prepare one native algorithm instance from typed configuration.
            pub fn prepare(
                configuration: $configuration,
            ) -> Result<
                $crate::PreparedAlgorithm<Self>,
                $crate::DeclarationPreparationError<$prepare_error>,
            > {
                <Self as $crate::AlgorithmDeclaration>::prepare(configuration)
            }

            /// Execute one complete native operation with typed borrowed arrays.
            #[allow(clippy::too_many_arguments)]
            pub fn process(
                prepared: &mut $crate::PreparedAlgorithm<Self>,
                $($input: &[$input_type],)*
                $($output: &mut [$output_type],)*
            ) -> Result<(), $crate::DeclarationProcessError<$process_error>> {
                let inputs = [$($crate::AlgorithmInput::from($input),)*];
                let mut outputs = [$($crate::AlgorithmOutput::from($output),)*];
                prepared.process(&inputs, &mut outputs).map(|_| ())
            }

            /// Execute one native operation with optional metadata for each input.
            #[allow(clippy::too_many_arguments)]
            pub fn process_with_metadata(
                prepared: &mut $crate::PreparedAlgorithm<Self>,
                metadata: &[Option<$crate::SampleMetadata>],
                $($input: &[$input_type],)*
                $($output: &mut [$output_type],)*
            ) -> Result<
                $crate::AlgorithmProcessResult,
                $crate::NativeProcessError<$crate::DeclarationProcessError<$process_error>>,
            > {
                let inputs = [$($crate::AlgorithmInput::from($input),)*];
                let mut outputs = [$($crate::AlgorithmOutput::from($output),)*];
                prepared.process_with_metadata(&inputs, metadata, &mut outputs)
            }

            /// Reset persistent mathematical history.
            pub fn reset(prepared: &mut $crate::PreparedAlgorithm<Self>) {
                prepared.reset();
            }
        }
    };

    (@shape [..$shape:expr]) => { &$shape };
    (@shape [$($dimension:expr),+ $(,)?]) => { &[$($dimension),+] };
    (@missing $deserializer:ident, $field:ident) => {
        return Err(<$deserializer::Error as $crate::__private::DeserializeError>::missing_field(stringify!($field)))
    };
    (@missing $deserializer:ident, $field:ident, $default:expr) => { $default };
    (@configuration $descriptor:expr, $type:ty) => { $descriptor };
    (@configuration $descriptor:expr, $type:ty, $default:expr) => {
        $descriptor.with_default(|| { let value: $type = $default; value.into() })
    };

    (@property_type none) => { $crate::NoDeclaredProperties };
    (@property_type runtime) => { $crate::RuntimeDeclaredProperties };
    (@workspace $plan:ty) => { <$plan as $crate::AlgorithmPlan>::Workspace };

    (@operation) => { $crate::AlgorithmOperation::General };
    (@operation DenseF32) => { $crate::AlgorithmOperation::DenseF32 };
    (@operation ShackHartmannImageF32) => {
        $crate::AlgorithmOperation::ShackHartmannImageF32
    };
    (@operation IncrementalDenseReconstructionF32) => {
        $crate::AlgorithmOperation::IncrementalDenseReconstructionF32
    };
    (@operation PyramidRowReconstructionF32) => {
        $crate::AlgorithmOperation::PyramidRowReconstructionF32
    };

    (@operation_plan $plan:ident) => {{
        let _ = $plan;
        $crate::AlgorithmOperationPlan::General
    }};
    (@operation_plan $plan:ident, DenseF32) => {
        $crate::AlgorithmOperationPlan::DenseF32($plan)
    };
    (@operation_plan $plan:ident, ShackHartmannImageF32) => {
        $crate::AlgorithmOperationPlan::ShackHartmannImageF32($plan)
    };
    (@operation_plan $plan:ident, IncrementalDenseReconstructionF32) => {
        $crate::AlgorithmOperationPlan::IncrementalDenseReconstructionF32($plan)
    };
    (@operation_plan $plan:ident, PyramidRowReconstructionF32) => {
        $crate::AlgorithmOperationPlan::PyramidRowReconstructionF32($plan)
    };
    (@operation_workspace $workspace:ident) => {{
        let _ = $workspace;
        $crate::AlgorithmOperationWorkspace::General
    }};
    (@operation_workspace $workspace:ident, DenseF32) => {{
        let _ = $workspace;
        $crate::AlgorithmOperationWorkspace::General
    }};
    (@operation_workspace $workspace:ident, ShackHartmannImageF32) => {{
        let _ = $workspace;
        $crate::AlgorithmOperationWorkspace::General
    }};
    (@operation_workspace $workspace:ident, IncrementalDenseReconstructionF32) => {
        $crate::AlgorithmOperationWorkspace::IncrementalDenseReconstructionF32($workspace)
    };
    (@operation_workspace $workspace:ident, PyramidRowReconstructionF32) => {
        $crate::AlgorithmOperationWorkspace::PyramidRowReconstructionF32($workspace)
    };
    (@workspace $plan:ty, $workspace:ty) => { $workspace };
    (@parameter_type) => { $crate::NoDeclaredParameters };
    (@parameter_type $parameter_error:ty) => { $crate::DeclaredParameters };
    (@parameter_error) => { $crate::__private::Infallible };
    (@parameter_error $parameter_error:ty) => { $parameter_error };
    (@metadata_source) => { None };
    (@metadata_source $metadata_source:literal) => { Some($metadata_source) };
    (@output_port $port:expr) => { $port };
    (@output_port $port:expr, conditional) => { $port.conditional() };
    (@format $type:ty, $shape:expr, $schema:expr,) => {
        $crate::AlgorithmFormat::packed::<$type>($shape, $schema)
    };
    (@format $type:ty, $shape:expr, $schema:expr, $numerator:expr, $denominator:expr) => {
        $crate::AlgorithmFormat::packed::<$type>($shape, $schema)
            .and_then(|format| format.with_rate_ratio($numerator, $denominator))
    };
}
