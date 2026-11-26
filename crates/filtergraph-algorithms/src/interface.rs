//! Common execution contract for admitted algorithm plans.

use core::fmt;
use core::mem;
use core::ops::Range;

/// A zero-size workspace for an admitted algorithm with no mutable execution
/// storage.
///
/// This type preserves the common call shape. It is not, by itself, a reason
/// to wrap an operation in an algorithm plan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NoWorkspace;

/// Sequence state for a stateless algorithm with progressive-input support.
///
/// The numerical operation may be stateless while its progressive capability
/// still needs to enforce one started, contiguous, complete execution. Plans
/// using this workspace expose the same complete [`AlgorithmPlan::process`]
/// operation while preventing it from overlapping an active progressive call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProgressWorkspace {
    processed: usize,
    started: bool,
}

impl ProgressWorkspace {
    /// Creates idle progressive sequence state.
    pub const fn new() -> Self {
        Self {
            processed: 0,
            started: false,
        }
    }

    /// Returns the number of contiguous semantic units processed so far.
    pub const fn processed(&self) -> usize {
        self.processed
    }

    /// Returns whether a progressive execution has started.
    pub const fn is_started(&self) -> bool {
        self.started
    }

    pub(crate) fn ensure_idle(&self) -> Result<(), ProcessError> {
        if self.started {
            return Err(ProcessError::AlreadyStarted);
        }
        Ok(())
    }

    pub(crate) fn start(&mut self) -> Result<(), ProcessError> {
        self.ensure_idle()?;
        self.processed = 0;
        self.started = true;
        Ok(())
    }

    pub(crate) fn validate_range(
        &self,
        workrange: &Range<usize>,
        extent: usize,
    ) -> Result<(), ProcessError> {
        if !self.started {
            return Err(ProcessError::NotStarted);
        }
        if workrange.is_empty() {
            return Err(ProcessError::EmptyRange);
        }
        if workrange.start != self.processed {
            return Err(ProcessError::RangeNotContiguous {
                start: workrange.start,
                expected: self.processed,
            });
        }
        if workrange.end > extent {
            return Err(ProcessError::RangeOutOfBounds {
                end: workrange.end,
                extent,
            });
        }
        Ok(())
    }

    pub(crate) fn advance(&mut self, processed: usize) {
        self.processed = processed;
    }

    pub(crate) fn finish(&mut self, extent: usize) -> Result<(), ProcessError> {
        if !self.started {
            return Err(ProcessError::NotStarted);
        }
        if self.processed != extent {
            return Err(ProcessError::Incomplete {
                processed: self.processed,
                extent,
            });
        }
        self.processed = 0;
        self.started = false;
        Ok(())
    }

    pub(crate) fn abort(&mut self) -> Result<(), ProcessError> {
        if !self.started {
            return Err(ProcessError::NotStarted);
        }
        self.processed = 0;
        self.started = false;
        Ok(())
    }
}

/// The length of the greatest immutable output semantic-unit prefix.
///
/// A length of `n` promises that output units `0..n` are complete and will not
/// be modified by later ranged calls in the same progressive execution. The
/// unit is declared by the concrete plan and need not be a scalar-array index.
/// This value never represents input bytes, transport readiness, sample
/// validity, or terminal publication.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[must_use]
pub struct ProgressResult {
    len: usize,
}

impl ProgressResult {
    /// Constructs a result for an immutable output prefix of `len` semantic
    /// units.
    pub const fn new(len: usize) -> Self {
        Self { len }
    }

    /// Returns the immutable output-prefix length.
    pub const fn len(self) -> usize {
        self.len
    }

    /// Returns `true` when no output semantic unit is immutable yet.
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// Direct-call misuse or a broken prepared execution invariant.
///
/// Expected numerical conditions belong in concrete output validity or an
/// owning adapter outcome rather than this error type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessError {
    /// Progressive processing was started more than once.
    AlreadyStarted,

    /// A progressive operation was requested before `start`.
    NotStarted,

    /// A caller-owned input has an incompatible extent.
    InputExtent {
        /// Required extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },

    /// A caller-owned output has an incompatible extent.
    OutputExtent {
        /// Required extent.
        expected: usize,
        /// Supplied extent.
        actual: usize,
    },

    /// A ranged call supplied no semantic work.
    EmptyRange,

    /// A semantic work range exceeded the prepared extent.
    RangeOutOfBounds {
        /// Exclusive range end.
        end: usize,
        /// Prepared semantic extent.
        extent: usize,
    },

    /// A stateful progressive range did not begin at the next required unit.
    RangeNotContiguous {
        /// Supplied zero-based range start.
        start: usize,
        /// Required zero-based range start.
        expected: usize,
    },

    /// Finish was requested before all required semantic units were processed.
    Incomplete {
        /// Number of processed semantic units.
        processed: usize,
        /// Required semantic extent.
        extent: usize,
    },
}

impl fmt::Display for ProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyStarted => formatter.write_str("progressive processing already started"),
            Self::NotStarted => formatter.write_str("progressive processing has not started"),
            Self::InputExtent { expected, actual } => {
                write!(formatter, "input extent {actual} does not match {expected}")
            }
            Self::OutputExtent { expected, actual } => {
                write!(
                    formatter,
                    "output extent {actual} does not match {expected}"
                )
            }
            Self::EmptyRange => formatter.write_str("work range must be nonempty"),
            Self::RangeOutOfBounds { end, extent } => {
                write!(formatter, "work range end {end} exceeds extent {extent}")
            }
            Self::RangeNotContiguous { start, expected } => write!(
                formatter,
                "work range starts at {start}, expected {expected}"
            ),
            Self::Incomplete { processed, extent } => write!(
                formatter,
                "processed extent {processed} has not reached {extent}"
            ),
        }
    }
}

impl core::error::Error for ProcessError {}

/// Scalar kinds admitted by the transport-independent property contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PropertyType {
    /// Boolean value.
    Bool,
    /// Signed 32-bit integer.
    Int,
    /// Signed 64-bit integer.
    Long,
    /// IEEE 754 binary32 value.
    Float,
    /// IEEE 754 binary64 value.
    Double,
    /// Unsigned descriptor-local or schema-defined identifier.
    Id,
    /// UTF-8 string used outside the repeated processing path.
    String,
}

/// One scalar property value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropertyValue<'a> {
    /// Boolean value.
    Bool(bool),
    /// Signed 32-bit integer.
    Int(i32),
    /// Signed 64-bit integer.
    Long(i64),
    /// IEEE 754 binary32 value.
    Float(f32),
    /// IEEE 754 binary64 value.
    Double(f64),
    /// Unsigned descriptor-local or schema-defined identifier.
    Id(u32),
    /// Borrowed UTF-8 string.
    String(&'a str),
}

impl PropertyValue<'_> {
    /// Returns the common scalar kind of this value.
    pub const fn property_type(self) -> PropertyType {
        match self {
            Self::Bool(_) => PropertyType::Bool,
            Self::Int(_) => PropertyType::Int,
            Self::Long(_) => PropertyType::Long,
            Self::Float(_) => PropertyType::Float,
            Self::Double(_) => PropertyType::Double,
            Self::Id(_) => PropertyType::Id,
            Self::String(_) => PropertyType::String,
        }
    }
}

/// When and how one scalar property may change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PropertyUpdate {
    /// Observable value that cannot be assigned through a property transaction.
    ReadOnly,
    /// Scalar that can replace a compatible plan at a process boundary.
    Runtime,
    /// Construction input that cannot change through a runtime transaction.
    Construction,
    /// Construction input whose change also requires graph reconstruction.
    GraphRebuild,
}

/// One named value admitted by a property choice constraint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropertyChoice<'a> {
    /// Scalar value represented by this choice.
    pub value: PropertyValue<'a>,
    /// Stable canonical name for the choice.
    pub name: &'a str,
    /// Human-readable description of the choice.
    pub description: &'a str,
}

/// Constraint attached to a scalar property descriptor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropertyConstraint<'a> {
    /// No common range or choice constraint.
    None,
    /// Inclusive scalar range.
    Range {
        /// Inclusive minimum.
        minimum: PropertyValue<'a>,
        /// Inclusive maximum.
        maximum: PropertyValue<'a>,
    },
    /// Complete set of admitted named values.
    Choices(&'a [PropertyChoice<'a>]),
}

/// Transport-independent description of one scalar algorithm property.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropertyInfo<'a> {
    /// Stable identifier within one algorithm descriptor.
    pub id: u32,
    /// Canonical unqualified property name.
    pub name: &'a str,
    /// Human-readable description.
    pub description: &'a str,
    /// Unit symbol, with `"1"` denoting a dimensionless quantity.
    pub unit: &'a str,
    /// Property update class.
    pub update: PropertyUpdate,
    /// Declared default value.
    pub default_value: PropertyValue<'a>,
    /// Optional range or named choices.
    pub constraint: PropertyConstraint<'a>,
}

impl PropertyInfo<'_> {
    /// Returns the common scalar kind declared by this descriptor.
    pub const fn property_type(self) -> PropertyType {
        self.default_value.property_type()
    }
}

/// One scalar assignment in a local property transaction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropertyAssignment<'a> {
    /// Descriptor-local property identifier.
    pub id: u32,
    /// Proposed scalar value.
    pub value: PropertyValue<'a>,
}

/// Invalid transport-independent property schema.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PropertySchemaError {
    /// A canonical property name is empty.
    EmptyName {
        /// Property identifier.
        id: u32,
    },
    /// A property description is empty.
    EmptyDescription {
        /// Property identifier.
        id: u32,
    },
    /// A canonical unit is empty.
    EmptyUnit {
        /// Property identifier.
        id: u32,
    },
    /// A range or choice value has a different kind from the property default.
    ConstraintType {
        /// Property identifier.
        id: u32,
    },
    /// A range is unsupported, non-finite, or ordered incorrectly.
    InvalidRange {
        /// Property identifier.
        id: u32,
    },
    /// A default does not satisfy its declared constraint.
    InvalidDefault {
        /// Property identifier.
        id: u32,
    },
    /// A choice constraint contains no choices.
    EmptyChoices {
        /// Property identifier.
        id: u32,
    },
    /// A canonical choice name is empty.
    EmptyChoiceName {
        /// Property identifier.
        id: u32,
        /// Zero-based choice index.
        index: usize,
    },
    /// Two choices use the same canonical name.
    DuplicateChoiceName {
        /// Property identifier.
        id: u32,
        /// Zero-based index of the first choice.
        first: usize,
        /// Zero-based index of the duplicate choice.
        second: usize,
    },
    /// Two choices represent the same scalar value.
    DuplicateChoiceValue {
        /// Property identifier.
        id: u32,
        /// Zero-based index of the first choice.
        first: usize,
        /// Zero-based index of the duplicate choice.
        second: usize,
    },
    /// Two descriptors use the same local identifier.
    DuplicateId {
        /// Duplicated identifier.
        id: u32,
        /// Zero-based index of the first descriptor.
        first: usize,
        /// Zero-based index of the duplicate descriptor.
        second: usize,
    },
    /// Two descriptors use the same canonical name.
    DuplicateName {
        /// Zero-based index of the first descriptor.
        first: usize,
        /// Zero-based index of the duplicate descriptor.
        second: usize,
    },
}

impl fmt::Display for PropertySchemaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName { id } => write!(formatter, "property id {id} has an empty name"),
            Self::EmptyDescription { id } => {
                write!(formatter, "property id {id} has an empty description")
            }
            Self::EmptyUnit { id } => write!(formatter, "property id {id} has an empty unit"),
            Self::ConstraintType { id } => {
                write!(
                    formatter,
                    "property id {id} has an incompatible constraint type"
                )
            }
            Self::InvalidRange { id } => {
                write!(formatter, "property id {id} has an invalid range")
            }
            Self::InvalidDefault { id } => {
                write!(formatter, "property id {id} has an invalid default")
            }
            Self::EmptyChoices { id } => {
                write!(formatter, "property id {id} has no choices")
            }
            Self::EmptyChoiceName { id, index } => {
                write!(
                    formatter,
                    "property id {id} choice {index} has an empty name"
                )
            }
            Self::DuplicateChoiceName { id, first, second } => write!(
                formatter,
                "property id {id} choices {first} and {second} have the same name"
            ),
            Self::DuplicateChoiceValue { id, first, second } => write!(
                formatter,
                "property id {id} choices {first} and {second} have the same value"
            ),
            Self::DuplicateId { id, first, second } => write!(
                formatter,
                "property descriptors {first} and {second} use id {id}"
            ),
            Self::DuplicateName { first, second } => write!(
                formatter,
                "property descriptors {first} and {second} use the same name"
            ),
        }
    }
}

impl core::error::Error for PropertySchemaError {}

/// Optional scalar property surface for an [`AlgorithmPlan`].
///
/// Preparation validates a complete local transaction and returns a new
/// logically read-only plan. It does not mutate the active plan or publish the
/// replacement. A runtime binding or deployment adapter owns bounded staging,
/// publication, generations, and process-boundary adoption.
pub trait AlgorithmProperties: AlgorithmPlan + Sized {
    /// Direct property lookup or transaction-validation error.
    type PropertyError;

    /// Returns one property descriptor by enumeration index.
    fn property_info(&self, index: usize) -> Option<PropertyInfo<'static>>;

    /// Returns the current plan value for one descriptor-local identifier.
    fn property(&self, id: u32) -> Result<PropertyValue<'_>, Self::PropertyError>;

    /// Validates a complete transaction and constructs a compatible plan.
    fn prepare_properties(
        &self,
        assignments: &[PropertyAssignment<'_>],
    ) -> Result<Self, Self::PropertyError>;
}

/// Common scalar property lookup or transaction failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PropertyError<E> {
    /// The algorithm declared an invalid property schema.
    InvalidSchema(PropertySchemaError),
    /// No property with the supplied descriptor-local identifier exists.
    UnknownProperty {
        /// Unrecognized identifier.
        id: u32,
    },
    /// One transaction assigns the same property more than once.
    DuplicateProperty {
        /// Duplicated identifier.
        id: u32,
    },
    /// An assignment has a different scalar kind from its descriptor.
    PropertyType {
        /// Property identifier.
        id: u32,
        /// Declared scalar kind.
        expected: PropertyType,
        /// Supplied scalar kind.
        actual: PropertyType,
    },
    /// A scalar value violates its descriptor range or choice constraint.
    PropertyConstraint {
        /// Property identifier.
        id: u32,
    },
    /// A runtime transaction targets a non-runtime property.
    PropertyUpdateClass {
        /// Property identifier.
        id: u32,
    },
    /// The complete proposed property set is not an admitted plan.
    InvalidPlan(E),
}

impl<E> fmt::Display for PropertyError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchema(error) => error.fmt(formatter),
            Self::UnknownProperty { id } => write!(formatter, "unknown property id {id}"),
            Self::DuplicateProperty { id } => {
                write!(formatter, "property id {id} is assigned more than once")
            }
            Self::PropertyType {
                id,
                expected,
                actual,
            } => write!(
                formatter,
                "property id {id} requires {expected:?}, received {actual:?}"
            ),
            Self::PropertyConstraint { id } => {
                write!(
                    formatter,
                    "property id {id} violates its descriptor constraint"
                )
            }
            Self::PropertyUpdateClass { id } => {
                write!(formatter, "property id {id} is not runtime-writable")
            }
            Self::InvalidPlan(error) => error.fmt(formatter),
        }
    }
}

impl<E> core::error::Error for PropertyError<E>
where
    E: core::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::InvalidPlan(error) => Some(error),
            Self::InvalidSchema(error) => Some(error),
            _ => None,
        }
    }
}

/// Returns whether two property values have the same scalar kind and value.
///
/// Deployment adapters use this while validating that a replacement prepared
/// from a non-property input preserves the complete scalar-property state.
#[doc(hidden)]
pub fn property_values_equal(a: PropertyValue<'_>, b: PropertyValue<'_>) -> bool {
    match (a, b) {
        (PropertyValue::Bool(a), PropertyValue::Bool(b)) => a == b,
        (PropertyValue::Int(a), PropertyValue::Int(b)) => a == b,
        (PropertyValue::Long(a), PropertyValue::Long(b)) => a == b,
        (PropertyValue::Float(a), PropertyValue::Float(b)) => a == b,
        (PropertyValue::Double(a), PropertyValue::Double(b)) => a == b,
        (PropertyValue::Id(a), PropertyValue::Id(b)) => a == b,
        (PropertyValue::String(a), PropertyValue::String(b)) => a == b,
        _ => false,
    }
}

/// Returns whether a typed value satisfies one descriptor constraint.
#[doc(hidden)]
pub fn property_value_satisfies(info: PropertyInfo<'_>, value: PropertyValue<'_>) -> bool {
    if info.property_type() != value.property_type() {
        return false;
    }
    match info.constraint {
        PropertyConstraint::None => true,
        PropertyConstraint::Range { minimum, maximum } => match (value, minimum, maximum) {
            (PropertyValue::Int(value), PropertyValue::Int(min), PropertyValue::Int(max)) => {
                (min..=max).contains(&value)
            }
            (PropertyValue::Long(value), PropertyValue::Long(min), PropertyValue::Long(max)) => {
                (min..=max).contains(&value)
            }
            (PropertyValue::Float(value), PropertyValue::Float(min), PropertyValue::Float(max)) => {
                value.is_finite() && value >= min && value <= max
            }
            (
                PropertyValue::Double(value),
                PropertyValue::Double(min),
                PropertyValue::Double(max),
            ) => value.is_finite() && value >= min && value <= max,
            (PropertyValue::Id(value), PropertyValue::Id(min), PropertyValue::Id(max)) => {
                (min..=max).contains(&value)
            }
            _ => false,
        },
        PropertyConstraint::Choices(choices) => choices
            .iter()
            .any(|choice| property_values_equal(choice.value, value)),
    }
}

/// Validate one property descriptor independently of an algorithm instance.
pub fn validate_property_info(info: PropertyInfo<'_>) -> Result<(), PropertySchemaError> {
    if info.name.is_empty() {
        return Err(PropertySchemaError::EmptyName { id: info.id });
    }
    if info.description.is_empty() {
        return Err(PropertySchemaError::EmptyDescription { id: info.id });
    }
    if info.unit.is_empty() {
        return Err(PropertySchemaError::EmptyUnit { id: info.id });
    }
    match info.constraint {
        PropertyConstraint::None => Ok(()),
        PropertyConstraint::Range { minimum, maximum } => {
            if minimum.property_type() != info.property_type()
                || maximum.property_type() != info.property_type()
            {
                return Err(PropertySchemaError::ConstraintType { id: info.id });
            }
            let valid_range = match (minimum, maximum) {
                (PropertyValue::Int(minimum), PropertyValue::Int(maximum)) => minimum <= maximum,
                (PropertyValue::Long(minimum), PropertyValue::Long(maximum)) => minimum <= maximum,
                (PropertyValue::Float(minimum), PropertyValue::Float(maximum)) => {
                    minimum.is_finite() && maximum.is_finite() && minimum <= maximum
                }
                (PropertyValue::Double(minimum), PropertyValue::Double(maximum)) => {
                    minimum.is_finite() && maximum.is_finite() && minimum <= maximum
                }
                (PropertyValue::Id(minimum), PropertyValue::Id(maximum)) => minimum <= maximum,
                _ => false,
            };
            if !valid_range {
                return Err(PropertySchemaError::InvalidRange { id: info.id });
            }
            if !property_value_satisfies(info, info.default_value) {
                return Err(PropertySchemaError::InvalidDefault { id: info.id });
            }
            Ok(())
        }
        PropertyConstraint::Choices(choices) => {
            if choices.is_empty() {
                return Err(PropertySchemaError::EmptyChoices { id: info.id });
            }
            for (index, choice) in choices.iter().enumerate() {
                if choice.value.property_type() != info.property_type() {
                    return Err(PropertySchemaError::ConstraintType { id: info.id });
                }
                if choice.name.is_empty() {
                    return Err(PropertySchemaError::EmptyChoiceName { id: info.id, index });
                }
                for (previous_index, previous) in choices[..index].iter().enumerate() {
                    if choice.name == previous.name {
                        return Err(PropertySchemaError::DuplicateChoiceName {
                            id: info.id,
                            first: previous_index,
                            second: index,
                        });
                    }
                    if property_values_equal(choice.value, previous.value) {
                        return Err(PropertySchemaError::DuplicateChoiceValue {
                            id: info.id,
                            first: previous_index,
                            second: index,
                        });
                    }
                }
            }
            if !property_value_satisfies(info, info.default_value) {
                return Err(PropertySchemaError::InvalidDefault { id: info.id });
            }
            Ok(())
        }
    }
}

/// Validate descriptor metadata and schema-wide identifier/name uniqueness.
///
/// Deployment adapters call this once while admitting an algorithm instance.
/// The check performs bounded stack-only work for a finite descriptor surface.
pub fn validate_property_schema<P>(plan: &P) -> Result<(), PropertySchemaError>
where
    P: AlgorithmProperties,
{
    let mut index = 0;
    while let Some(info) = plan.property_info(index) {
        validate_property_info(info)?;
        for previous_index in 0..index {
            let Some(previous) = plan.property_info(previous_index) else {
                break;
            };
            if previous.id == info.id {
                return Err(PropertySchemaError::DuplicateId {
                    id: info.id,
                    first: previous_index,
                    second: index,
                });
            }
            if previous.name == info.name {
                return Err(PropertySchemaError::DuplicateName {
                    first: previous_index,
                    second: index,
                });
            }
        }
        index += 1;
    }
    Ok(())
}

/// Declare an algorithm's scalar property schema and current-value mapping.
///
/// The macro generates identifier lookup, duplicate rejection, type and
/// constraint validation, update-class enforcement, and transaction iteration.
/// The algorithm supplies one field mapping per property and one replacement
/// function with the signature `fn(&Plan, values...) -> Result<Plan, E>`.
/// A binding normally reads the field with the same name; `from a.b` selects a
/// nested field while retaining a concise replacement-argument name.
#[macro_export]
macro_rules! algorithm_properties {
    (
        impl $plan:ty {
            plan_error = $plan_error:ty;
            replace = $replace:path;
            properties {
                $(
                    $id:expr => $field:ident $(from $($source:ident).+)? : $kind:ident {
                        name = $name:literal;
                        description = $description:literal;
                        unit = $unit:literal;
                        update = $update:ident;
                        default = $default:expr;
                        constraint = $constraint:expr;
                    }
                )+
            }
        }
    ) => {
        impl $crate::AlgorithmProperties for $plan {
            type PropertyError = $crate::PropertyError<$plan_error>;

            fn property_info(&self, index: usize) -> Option<$crate::PropertyInfo<'static>> {
                let properties = [
                    $(
                        $crate::PropertyInfo {
                            id: $id,
                            name: $name,
                            description: $description,
                            unit: $unit,
                            update: $crate::PropertyUpdate::$update,
                            default_value: $crate::PropertyValue::$kind($default),
                            constraint: $constraint,
                        },
                    )+
                ];
                properties.get(index).copied()
            }

            fn property(
                &self,
                id: u32,
            ) -> Result<$crate::PropertyValue<'_>, Self::PropertyError> {
                match id {
                    $(
                        id if id == $id => Ok($crate::algorithm_properties!(
                            @borrow $kind,
                            $crate::algorithm_properties!(
                                @source self, $field $(, $($source).+)?
                            )
                        )),
                    )+
                    id => Err($crate::PropertyError::UnknownProperty { id }),
                }
            }

            fn prepare_properties(
                &self,
                assignments: &[$crate::PropertyAssignment<'_>],
            ) -> Result<Self, Self::PropertyError> {
                $crate::validate_property_schema(self)
                    .map_err($crate::PropertyError::InvalidSchema)?;
                $(
                    let mut $field = $crate::algorithm_properties!(
                        @source self, $field $(, $($source).+)?
                    ).clone();
                )+

                for (index, assignment) in assignments.iter().enumerate() {
                    if assignments[..index]
                        .iter()
                        .any(|previous| previous.id == assignment.id)
                    {
                        return Err($crate::PropertyError::DuplicateProperty {
                            id: assignment.id,
                        });
                    }
                    match assignment.id {
                        $(
                            id if id == $id => {
                                if $crate::PropertyUpdate::$update !=
                                    $crate::PropertyUpdate::Runtime
                                {
                                    return Err($crate::PropertyError::PropertyUpdateClass {
                                        id: assignment.id,
                                    });
                                }
                                let $crate::PropertyValue::$kind(value) = assignment.value else {
                                    return Err($crate::PropertyError::PropertyType {
                                        id: assignment.id,
                                        expected: $crate::PropertyType::$kind,
                                        actual: assignment.value.property_type(),
                                    });
                                };
                                let info = $crate::PropertyInfo {
                                    id: $id,
                                    name: $name,
                                    description: $description,
                                    unit: $unit,
                                    update: $crate::PropertyUpdate::$update,
                                    default_value: $crate::PropertyValue::$kind($default),
                                    constraint: $constraint,
                                };
                                if !$crate::property_value_satisfies(info, assignment.value) {
                                    return Err($crate::PropertyError::PropertyConstraint {
                                        id: assignment.id,
                                    });
                                }
                                $field = $crate::algorithm_properties!(@owned $kind, value);
                            }
                        )+
                        id => return Err($crate::PropertyError::UnknownProperty { id }),
                    }
                }

                ($replace)(self, $($field),+)
                    .map_err($crate::PropertyError::InvalidPlan)
            }
        }
    };
    (@borrow String, $value:expr) => {
        $crate::PropertyValue::String($value.as_str())
    };
    (@borrow $kind:ident, $value:expr) => {
        $crate::PropertyValue::$kind($value)
    };
    (@owned String, $value:expr) => {
        $value.into()
    };
    (@owned $kind:ident, $value:expr) => {
        $value
    };
    (@source $self:ident, $field:ident) => {
        $self.$field
    };
    (@source $self:ident, $field:ident, $($source:ident).+) => {
        $self.$($source).+
    };
}

/// Failure to prepare one bounded portable property update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PropertyBindingError<E> {
    /// A prepared or published update already occupies the one-update capacity.
    Busy,
    /// No further monotonic generation can be represented.
    GenerationExhausted,
    /// The algorithm rejected the proposed property transaction.
    Property(E),
}

impl<E> fmt::Display for PropertyBindingError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy => formatter.write_str("a property update is already pending"),
            Self::GenerationExhausted => formatter.write_str("property generation is exhausted"),
            Self::Property(error) => error.fmt(formatter),
        }
    }
}

impl<E> core::error::Error for PropertyBindingError<E> where E: core::error::Error + 'static {}

/// Bounded serial property publication for a portable algorithm executor.
///
/// This owner is deliberately not synchronized. One serial control owner calls
/// preparation, commit, discard, and retired-plan reclamation. The processing
/// owner calls [`Self::adopt_properties`] immediately before one complete
/// algorithm operation. Deployment adapters with concurrent control and data
/// loops provide their own synchronization while preserving these transitions.
#[derive(Clone, Debug, PartialEq)]
pub struct PropertyBinding<P> {
    active: P,
    staged: Option<P>,
    requested: Option<P>,
    retired: Option<P>,
    requested_generation: u64,
    active_generation: u64,
}

impl<P> PropertyBinding<P>
where
    P: AlgorithmProperties,
{
    /// Constructs a binding with generation zero and no pending update.
    pub const fn new(plan: P) -> Self {
        Self {
            active: plan,
            staged: None,
            requested: None,
            retired: None,
            requested_generation: 0,
            active_generation: 0,
        }
    }

    /// Returns the plan currently used for processing.
    pub const fn active_plan(&self) -> &P {
        &self.active
    }

    /// Returns the latest published plan, or the active plan when none is pending.
    pub fn requested_plan(&self) -> &P {
        self.requested.as_ref().unwrap_or(&self.active)
    }

    /// Returns the latest published property generation.
    pub const fn requested_generation(&self) -> u64 {
        self.requested_generation
    }

    /// Returns the property generation currently used for processing.
    pub const fn active_generation(&self) -> u64 {
        self.active_generation
    }

    /// Prepares one complete local transaction into the inactive slot.
    ///
    /// This cold control-path operation also releases a plan retired by an
    /// earlier adoption. It reports [`PropertyBindingError::Busy`] until a
    /// staged transaction is discarded or a published transaction is adopted.
    pub fn prepare_properties(
        &mut self,
        assignments: &[PropertyAssignment<'_>],
    ) -> Result<(), PropertyBindingError<P::PropertyError>> {
        if assignments.is_empty() {
            return Ok(());
        }
        if self.staged.is_some() || self.requested.is_some() {
            return Err(PropertyBindingError::Busy);
        }
        if self.requested_generation == u64::MAX {
            return Err(PropertyBindingError::GenerationExhausted);
        }
        self.retired = None;
        self.staged = Some(
            self.active
                .prepare_properties(assignments)
                .map_err(PropertyBindingError::Property)?,
        );
        Ok(())
    }

    /// Publishes the successfully prepared transaction.
    ///
    /// This operation is non-failing. Calling it without a staged transaction
    /// is a no-op so an FFI adapter never needs to unwind from a commit callback.
    pub fn commit_properties(&mut self) {
        let Some(plan) = self.staged.take() else {
            return;
        };
        self.requested = Some(plan);
        self.requested_generation += 1;
    }

    /// Discards a prepared transaction without changing either generation.
    pub fn discard_properties(&mut self) {
        self.staged = None;
    }

    /// Adopts one complete requested plan and returns whether adoption occurred.
    ///
    /// The previous active plan is retained rather than destroyed on this
    /// repeated-path transition. The next control-path preparation or explicit
    /// reclamation releases it.
    pub fn adopt_properties(&mut self) -> bool {
        let Some(requested) = self.requested.take() else {
            return false;
        };
        let previous = mem::replace(&mut self.active, requested);
        self.retired = Some(previous);
        self.active_generation = self.requested_generation;
        true
    }

    /// Releases a retired plan on the serial control path.
    pub fn reclaim_retired_properties(&mut self) {
        self.retired = None;
    }
}

/// A prepared implementation of one admitted numerical algorithm.
///
/// The plan is logically read-only while it is used. [`Self::Workspace`] has
/// one authoritative mutable owner. Input and output associated types are
/// short-lived caller-owned views; implementations must not retain them.
///
/// A complete call performs one accepted completed-sample operation. A
/// stateful implementation does not expose partially updated committed
/// history if the call fails. An aborted progressive operation likewise
/// preserves the history committed before that operation began. Unpublished
/// result scratch is named staged storage and is copied to caller-owned output
/// only after every operation failure check succeeds. Caller-owned output
/// storage is never exchanged with a workspace buffer. Double buffering and
/// ownership exchange may commit state only when the workspace owns both
/// buffers; neither mechanism is required by the interface. Scratch that
/// cannot affect a later operation may be mutated directly.
///
/// Complete execution is entered through [`Self::process`]. Each plan
/// represents one independently callable numerical operation. Related
/// operations use separate plans, although they may share immutable prepared
/// artifacts. Additional execution methods are supplied only by declared
/// capability traits such as [`InputProgress`]. A dependency operation that
/// does not add a scientific contract remains a direct dependency call rather
/// than acquiring a forwarding plan solely for interface uniformity.
///
/// One plan operation does not imply one implementation kernel or dependency
/// call. A backend implementation may use no kernel, one kernel, several
/// kernels, or direct dependency operations while preserving every observable
/// result, state transition, unit, coordinate space, failure outcome, ordering
/// rule, and progressive-output guarantee. A different backend uses a separate
/// concrete plan type implementing this trait; the common interface does not
/// include a backend parameter, device buffers, launch, synchronization, or
/// graph scheduling. A complete accelerator implementation returns from
/// [`Self::process`] only after the complete operation has finished.
///
/// Physical units and coordinate spaces are part of the operation contract.
/// A unit-parametric operation documents and preserves the required
/// relationships. It does not infer or introduce a unit conversion unless
/// conversion is the operation represented by the plan.
pub trait AlgorithmPlan {
    /// Caller-owned input view for one borrow lifetime.
    type Input<'a>
    where
        Self: 'a;

    /// Caller-owned output view for one borrow lifetime.
    type Output<'a>
    where
        Self: 'a;

    /// Exclusively owned mutable execution storage.
    type Workspace;

    /// Preparation misuse, broken invariants, or unexpected execution failure.
    ///
    /// Expected numerical outcomes belong in the concrete output or validity
    /// representation rather than this error type.
    type Error;

    /// Processes one complete input into caller-owned output storage.
    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error>;
}

/// Progressive input execution for an [`AlgorithmPlan`].
///
/// Implementing this trait is the Rust capability declaration corresponding
/// to FilterGraph's progressive-input support. Every successful ranged call must
/// consume its complete, nonempty `workrange`. The range uses Rust's
/// zero-based half-open convention in the plan's semantic input-unit domain.
/// The authoritative caller owns range ordering, bounds, and the decision to
/// finish or abort.
pub trait InputProgress: AlgorithmPlan {
    /// Initializes bounded in-progress state from committed mathematical
    /// history without changing that committed history.
    fn start(&self, workspace: &mut Self::Workspace) -> Result<(), Self::Error>;

    /// Processes a complete nonempty range of semantic input units.
    ///
    /// The returned prefix is monotonic within one execution and cannot exceed
    /// the plan's declared output extent. A plan without [`OutputProgress`]
    /// returns an empty result before [`Self::finish`].
    fn process_range<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
        workrange: Range<usize>,
    ) -> Result<ProgressResult, Self::Error>;

    /// Finalizes output and commits fully processed mathematical history.
    fn finish<'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
    ) -> Result<(), Self::Error>;

    /// Discards in-progress state while preserving previously committed
    /// mathematical history.
    ///
    /// Caller-owned output may remain dirty and must not be published as a
    /// completed sample.
    fn abort(&self, workspace: &mut Self::Workspace) -> Result<(), Self::Error>;
}

/// Marker for progressive plans that expose immutable output prefixes.
///
/// Implementing this trait promises that every [`ProgressResult`] returned by
/// [`InputProgress::process_range`] identifies a prefix safe for the owning
/// adapter to expose before terminal completion. The prefix cannot be
/// retracted, including when the execution later aborts; terminal sample state
/// remains the caller's responsibility.
pub trait OutputProgress: InputProgress {}
