//! Lock-free FGN publication for immutable FilterGraph plans.

use core::ffi::{CStr, c_void};
use core::fmt;
use core::ptr;
use core::sync::atomic::{AtomicU64, Ordering};
use std::ffi::{CString, NulError};

use filtergraph_algorithms::{
    AlgorithmProperties, PropertyAssignment, PropertyConstraint, PropertyInfo, PropertySchemaError,
    PropertyUpdate, PropertyValue, property_value_satisfies, validate_property_schema,
};

use crate::parameter::{ACTIVE_PARAMETER_SEQUENCE_ID, REQUESTED_PARAMETER_SEQUENCE_ID};
use crate::{abi, publication::Publication};

/// Adapter-local ID for the latest published property generation.
pub const REQUESTED_GENERATION_ID: u32 = u32::MAX - 1;
/// Adapter-local ID for the property generation used by processing.
pub const ACTIVE_GENERATION_ID: u32 = u32::MAX;

pub(crate) const FGN_GENERATION_MAX: u64 = i64::MAX as u64;

/// Failure while admitting a FilterGraph property surface to FGN.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PropertyAdapterError {
    /// The FilterGraph schema itself is invalid.
    InvalidSchema(PropertySchemaError),
    /// An algorithm descriptor collides with an adapter-owned observation.
    ReservedProperty {
        /// Colliding algorithm-local identifier.
        id: u32,
    },
    /// A property name exceeds the FGN local-name limit.
    PropertyNameTooLong {
        /// Descriptor-local identifier.
        id: u32,
    },
    /// Text cannot be represented as a C string.
    InteriorNul,
    /// A plan's current value does not satisfy its descriptor.
    InvalidCurrentValue {
        /// Descriptor-local identifier.
        id: u32,
    },
    /// An ndarray format cannot be represented by the FGN host contract.
    InvalidFormat,
}

impl fmt::Display for PropertyAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchema(error) => error.fmt(formatter),
            Self::ReservedProperty { id } => {
                write!(formatter, "property id {id} is reserved by the FGN adapter")
            }
            Self::PropertyNameTooLong { id } => write!(
                formatter,
                "property id {id} exceeds the FGN local-name limit"
            ),
            Self::InteriorNul => formatter.write_str("FGN text contains an interior NUL byte"),
            Self::InvalidCurrentValue { id } => {
                write!(formatter, "property id {id} has an invalid current value")
            }
            Self::InvalidFormat => formatter.write_str("invalid FGN ndarray format"),
        }
    }
}

impl core::error::Error for PropertyAdapterError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::InvalidSchema(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PropertySchemaError> for PropertyAdapterError {
    fn from(value: PropertySchemaError) -> Self {
        Self::InvalidSchema(value)
    }
}

impl From<NulError> for PropertyAdapterError {
    fn from(_value: NulError) -> Self {
        Self::InteriorNul
    }
}

struct OwnedValue {
    abi: abi::Value,
    _string: Option<CString>,
}

impl OwnedValue {
    fn new(value: PropertyValue<'_>) -> Result<Self, PropertyAdapterError> {
        let string = match value {
            PropertyValue::String(value) => Some(CString::new(value)?),
            _ => None,
        };
        let abi = match value {
            PropertyValue::Bool(value) => abi::Value {
                type_: abi::TYPE_BOOL,
                reserved: 0,
                value: abi::ValueBody {
                    boolean: i32::from(value),
                },
            },
            PropertyValue::Int(value) => abi::Value {
                type_: abi::TYPE_INT,
                reserved: 0,
                value: abi::ValueBody { integer: value },
            },
            PropertyValue::Long(value) => abi::Value {
                type_: abi::TYPE_LONG,
                reserved: 0,
                value: abi::ValueBody {
                    long_integer: value,
                },
            },
            PropertyValue::Float(value) => abi::Value {
                type_: abi::TYPE_FLOAT,
                reserved: 0,
                value: abi::ValueBody { float_value: value },
            },
            PropertyValue::Double(value) => abi::Value {
                type_: abi::TYPE_DOUBLE,
                reserved: 0,
                value: abi::ValueBody {
                    double_value: value,
                },
            },
            PropertyValue::Id(value) => abi::Value {
                type_: abi::TYPE_ID,
                reserved: 0,
                value: abi::ValueBody { id: value },
            },
            PropertyValue::String(_) => abi::Value {
                type_: abi::TYPE_STRING,
                reserved: 0,
                value: abi::ValueBody {
                    string: string.as_ref().expect("string value exists").as_ptr(),
                },
            },
        };
        Ok(Self {
            abi,
            _string: string,
        })
    }
}

struct OwnedChoice {
    value: OwnedValue,
    name: CString,
    description: CString,
}

struct OwnedPropertyInfo {
    abi: abi::PropertyInfo,
    _name: CString,
    _description: CString,
    _unit: CString,
    _default_value: OwnedValue,
    _minimum: Option<OwnedValue>,
    _maximum: Option<OwnedValue>,
    _choices: Box<[OwnedChoice]>,
    _abi_choices: Box<[abi::PropertyChoice]>,
}

impl OwnedPropertyInfo {
    fn new(info: PropertyInfo<'static>) -> Result<Self, PropertyAdapterError> {
        let name = CString::new(info.name)?;
        let description = CString::new(info.description)?;
        let unit = CString::new(info.unit)?;
        let default_value = OwnedValue::new(info.default_value)?;
        let (minimum, maximum, choices, mut flags) = match info.constraint {
            PropertyConstraint::None => (None, None, Vec::new(), 0),
            PropertyConstraint::Range { minimum, maximum } => (
                Some(OwnedValue::new(minimum)?),
                Some(OwnedValue::new(maximum)?),
                Vec::new(),
                abi::PROPERTY_RANGE,
            ),
            PropertyConstraint::Choices(values) => {
                let mut choices = Vec::with_capacity(values.len());
                for choice in values {
                    choices.push(OwnedChoice {
                        value: OwnedValue::new(choice.value)?,
                        name: CString::new(choice.name)?,
                        description: CString::new(choice.description)?,
                    });
                }
                (None, None, choices, abi::PROPERTY_CHOICES)
            }
        };
        flags |= match info.update {
            PropertyUpdate::ReadOnly => abi::PROPERTY_READONLY,
            PropertyUpdate::Runtime => abi::PROPERTY_RUNTIME,
            PropertyUpdate::Construction => abi::PROPERTY_CONSTRUCTION,
            PropertyUpdate::GraphRebuild => abi::PROPERTY_GRAPH_REBUILD,
        };
        let choices = choices.into_boxed_slice();
        let abi_choices = choices
            .iter()
            .map(|choice| abi::PropertyChoice {
                struct_size: size_of::<abi::PropertyChoice>() as u32,
                reserved: 0,
                value: choice.value.abi,
                name: choice.name.as_ptr(),
                description: choice.description.as_ptr(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let abi = abi::PropertyInfo {
            struct_size: size_of::<abi::PropertyInfo>() as u32,
            id: info.id,
            flags,
            reserved: 0,
            name: name.as_ptr(),
            description: description.as_ptr(),
            unit: unit.as_ptr(),
            default_value: default_value.abi,
            minimum: minimum.as_ref().map_or(abi::Value::NONE, |value| value.abi),
            maximum: maximum.as_ref().map_or(abi::Value::NONE, |value| value.abi),
            n_choices: abi_choices.len() as u32,
            choices_reserved: 0,
            choices: if abi_choices.is_empty() {
                ptr::null()
            } else {
                abi_choices.as_ptr()
            },
        };
        Ok(Self {
            abi,
            _name: name,
            _description: description,
            _unit: unit,
            _default_value: default_value,
            _minimum: minimum,
            _maximum: maximum,
            _choices: choices,
            _abi_choices: abi_choices,
        })
    }
}

/// Instance-owned FGN projection of one validated FilterGraph schema.
pub struct PropertySchema {
    algorithm_infos: Box<[PropertyInfo<'static>]>,
    descriptors: Box<[OwnedPropertyInfo]>,
}

// SAFETY: ABI pointers reference heap allocations owned by the schema. Moving
// the schema moves only Box handles, not their allocations, and all pointees
// are immutable after construction.
unsafe impl Send for PropertySchema {}
// SAFETY: the schema and every allocation reachable through its ABI pointers
// are immutable after admission.
unsafe impl Sync for PropertySchema {}

impl PropertySchema {
    pub(crate) fn new<P>(plan: &P) -> Result<Self, PropertyAdapterError>
    where
        P: AlgorithmProperties,
    {
        validate_property_schema(plan)?;
        let mut infos = Vec::new();
        let mut index = 0;
        while let Some(info) = plan.property_info(index) {
            if info.name.len() > abi::LOCAL_NAME_MAX {
                return Err(PropertyAdapterError::PropertyNameTooLong { id: info.id });
            }
            if info.id == REQUESTED_GENERATION_ID
                || info.id == ACTIVE_GENERATION_ID
                || info.id == REQUESTED_PARAMETER_SEQUENCE_ID
                || info.id == ACTIVE_PARAMETER_SEQUENCE_ID
                || info.name == "requested-generation"
                || info.name == "active-generation"
                || info.name == "requested-parameter-sequence"
                || info.name == "active-parameter-sequence"
            {
                return Err(PropertyAdapterError::ReservedProperty { id: info.id });
            }
            infos.push(info);
            index += 1;
        }
        let algorithm_infos = infos.into_boxed_slice();
        let mut descriptors = Vec::with_capacity(algorithm_infos.len() + 2);
        for info in &algorithm_infos {
            descriptors.push(OwnedPropertyInfo::new(*info)?);
        }
        descriptors.push(OwnedPropertyInfo::new(PropertyInfo {
            id: REQUESTED_GENERATION_ID,
            name: "requested-generation",
            description: "Latest property generation published for processing",
            unit: "1",
            update: PropertyUpdate::ReadOnly,
            default_value: PropertyValue::Long(0),
            constraint: PropertyConstraint::None,
        })?);
        descriptors.push(OwnedPropertyInfo::new(PropertyInfo {
            id: ACTIVE_GENERATION_ID,
            name: "active-generation",
            description: "Property generation used by processing",
            unit: "1",
            update: PropertyUpdate::ReadOnly,
            default_value: PropertyValue::Long(0),
            constraint: PropertyConstraint::None,
        })?);
        Ok(Self {
            algorithm_infos,
            descriptors: descriptors.into_boxed_slice(),
        })
    }

    /// Return one ABI descriptor by enumeration index.
    pub fn info(&self, index: usize) -> Option<abi::PropertyInfo> {
        self.descriptors.get(index).map(|descriptor| descriptor.abi)
    }

    pub(crate) const fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub(crate) fn algorithm_infos(&self) -> &[PropertyInfo<'static>] {
        &self.algorithm_infos
    }
}

pub(crate) struct AdaptedPlan<P> {
    plan: P,
    values: Box<[(u32, OwnedValue)]>,
}

// SAFETY: each ABI string pointer refers to CString storage owned by the same
// immutable AdaptedPlan. Moving Box/CString handles does not move their
// allocations, and no callback mutates the cached values.
unsafe impl<P: Send> Send for AdaptedPlan<P> {}
// SAFETY: cached property values and their pointed-to strings are immutable.
unsafe impl<P: Sync> Sync for AdaptedPlan<P> {}

impl<P> AdaptedPlan<P>
where
    P: AlgorithmProperties,
{
    pub(crate) fn new(plan: P, schema: &PropertySchema) -> Result<Self, PropertyAdapterError> {
        let mut values = Vec::with_capacity(schema.algorithm_infos.len());
        for info in &schema.algorithm_infos {
            let value = plan
                .property(info.id)
                .map_err(|_| PropertyAdapterError::InvalidCurrentValue { id: info.id })?;
            if !property_value_satisfies(*info, value) {
                return Err(PropertyAdapterError::InvalidCurrentValue { id: info.id });
            }
            values.push((info.id, OwnedValue::new(value)?));
        }
        Ok(Self {
            plan,
            values: values.into_boxed_slice(),
        })
    }

    pub(crate) fn plan(&self) -> &P {
        &self.plan
    }

    pub(crate) fn value(&self, id: u32) -> Option<abi::Value> {
        self.values
            .iter()
            .find_map(|(candidate, value)| (*candidate == id).then_some(value.abi))
    }
}

/// Two-slot, single-control-owner property publication for an FGN plugin instance.
///
/// Preparation allocates and validates on the serial control path. FGN calls
/// `commit()` at graph-cycle start; it performs only atomic publication and
/// retains the prior plan for later control-path reclamation.
pub struct PropertyRuntime<P> {
    schema: PropertySchema,
    publication: Publication<AdaptedPlan<P>, u64>,
    requested_generation: AtomicU64,
    active_generation: AtomicU64,
    revision: AtomicU64,
}

impl<P> PropertyRuntime<P>
where
    P: AlgorithmProperties + Send + Sync,
{
    /// Admit a plan and construct its complete FGN property schema.
    pub fn new(plan: P) -> Result<Self, PropertyAdapterError> {
        let schema = PropertySchema::new(&plan)?;
        let active = AdaptedPlan::new(plan, &schema)?;
        Ok(Self {
            schema,
            publication: Publication::new(active),
            requested_generation: AtomicU64::new(0),
            active_generation: AtomicU64::new(0),
            revision: AtomicU64::new(0),
        })
    }

    /// Return the admitted FGN descriptor projection.
    pub const fn schema(&self) -> &PropertySchema {
        &self.schema
    }

    fn active_adapted_plan(&self) -> &AdaptedPlan<P> {
        self.publication.active()
    }

    /// Return the immutable plan selected for the current process call.
    pub fn active_plan(&self) -> &P {
        &self.active_adapted_plan().plan
    }

    /// Return one current algorithm or adapter-owned property value.
    pub fn get(&self, id: u32) -> Result<abi::Value, i32> {
        if id == REQUESTED_GENERATION_ID {
            return Ok(long_value(
                self.requested_generation.load(Ordering::Acquire),
            ));
        }
        if id == ACTIVE_GENERATION_ID {
            return Ok(long_value(self.active_generation.load(Ordering::Acquire)));
        }
        self.active_adapted_plan().value(id).ok_or(-abi::ENOENT)
    }

    /// Return the monotonic externally visible property revision.
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    /// Prepare one complete FGN transaction and return its opaque slot token.
    ///
    /// # Safety
    ///
    /// `properties` and every string it references must remain valid for this
    /// call. `result` must be writable. Calls are serialized by the FGN host.
    pub unsafe fn prepare(
        &self,
        properties: *const abi::Property,
        n_properties: u32,
        result: *mut *mut c_void,
    ) -> i32 {
        if result.is_null() {
            return -abi::EINVAL;
        }
        // SAFETY: result was checked above and is lent for this callback. An
        // empty transaction is a successful capacity-free no-op.
        unsafe {
            *result = ptr::null_mut();
        }
        if n_properties == 0 {
            return 0;
        }
        if properties.is_null() {
            return -abi::EINVAL;
        }
        if self.requested_generation.load(Ordering::Acquire) >= FGN_GENERATION_MAX {
            return -abi::EOVERFLOW;
        }
        // SAFETY: the caller supplies n_properties initialized ABI entries.
        let properties = unsafe { core::slice::from_raw_parts(properties, n_properties as usize) };
        match self.publication.prepare(0, |active| {
            let mut assignments = Vec::with_capacity(properties.len());
            for property in properties {
                assignments.push(unsafe { assignment_from_abi(property) }.map_err(|_| ())?);
            }
            let replacement = active
                .plan
                .prepare_properties(assignments.as_slice())
                .map_err(|_| ())?;
            AdaptedPlan::new(replacement, &self.schema).map_err(|_| ())
        }) {
            Ok(prepared) => {
                // SAFETY: result is writable for this callback.
                unsafe {
                    *result = prepared;
                }
                0
            }
            Err(error) => -error,
        }
    }

    /// Publish one prepared slot at graph-cycle start without allocation.
    ///
    /// # Safety
    ///
    /// `prepared` must be a token returned by `prepare()` for this runtime.
    pub unsafe fn commit(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance and exactly-once consumption are
        // this method's caller contract.
        unsafe {
            self.publication.commit(prepared, |generation| {
                *generation = self.requested_generation.load(Ordering::Relaxed) + 1;
                self.revision.fetch_add(1, Ordering::AcqRel);
                self.requested_generation
                    .store(*generation, Ordering::Release);
                self.revision.fetch_add(1, Ordering::Release);
            });
        }
    }

    /// Discard one unpublished prepared slot on the control path.
    ///
    /// # Safety
    ///
    /// `prepared` must be a token returned by `prepare()` for this runtime.
    pub unsafe fn discard(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance and exactly-once consumption are
        // this method's caller contract.
        unsafe { self.publication.discard(prepared) }
    }

    pub(crate) fn adopt_requested(&self) -> bool {
        self.publication.adopt(
            || {
                self.revision.fetch_add(1, Ordering::AcqRel);
            },
            |generation| {
                self.active_generation.store(*generation, Ordering::Release);
                self.revision.fetch_add(1, Ordering::Release);
            },
        )
    }
}

pub(crate) fn long_value(generation: u64) -> abi::Value {
    abi::Value {
        type_: abi::TYPE_LONG,
        reserved: 0,
        value: abi::ValueBody {
            long_integer: generation as i64,
        },
    }
}

pub(crate) unsafe fn assignment_from_abi<'a>(
    property: &'a abi::Property,
) -> Result<PropertyAssignment<'a>, i32> {
    let value = match property.value.type_ {
        abi::TYPE_BOOL => PropertyValue::Bool(unsafe { property.value.value.boolean } != 0),
        abi::TYPE_INT => PropertyValue::Int(unsafe { property.value.value.integer }),
        abi::TYPE_LONG => PropertyValue::Long(unsafe { property.value.value.long_integer }),
        abi::TYPE_FLOAT => PropertyValue::Float(unsafe { property.value.value.float_value }),
        abi::TYPE_DOUBLE => PropertyValue::Double(unsafe { property.value.value.double_value }),
        abi::TYPE_ID => PropertyValue::Id(unsafe { property.value.value.id }),
        abi::TYPE_STRING => {
            let string = unsafe { property.value.value.string };
            if string.is_null() {
                return Err(-abi::EINVAL);
            }
            let string = unsafe { CStr::from_ptr(string) }
                .to_str()
                .map_err(|_| -abi::EINVAL)?;
            PropertyValue::String(string)
        }
        _ => return Err(-abi::EINVAL),
    };
    Ok(PropertyAssignment {
        id: property.id,
        value,
    })
}
