//! FGN property projection and bounded publication conformance.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::c_void;

use filtergraph_algorithms::LeakyIntegratorPlan;
use filtergraph_fgn::{
    ACTIVE_GENERATION_ID, NdarrayFormat, PlanStore, PropertyAdapterError, PropertyRuntime,
    REQUESTED_GENERATION_ID, abi,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static DEALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if COUNTING.get() {
            DEALLOCATIONS.set(DEALLOCATIONS.get() + 1);
        }
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations(callback: impl FnOnce()) -> (usize, usize) {
    ALLOCATIONS.set(0);
    DEALLOCATIONS.set(0);
    COUNTING.set(true);
    callback();
    COUNTING.set(false);
    (ALLOCATIONS.get(), DEALLOCATIONS.get())
}

fn float_property(id: u32, value: f32) -> abi::Property {
    abi::Property {
        id,
        reserved: 0,
        value: abi::Value {
            type_: abi::TYPE_FLOAT,
            reserved: 0,
            value: abi::ValueBody { float_value: value },
        },
    }
}

fn long(runtime: &PropertyRuntime<LeakyIntegratorPlan<f32>>, id: u32) -> i64 {
    let value = runtime.get(id).unwrap();
    assert_eq!(value.type_, abi::TYPE_LONG);
    // SAFETY: the type tag was checked above.
    unsafe { value.value.long_integer }
}

#[test]
fn schema_projection_and_two_property_publication_are_bounded() {
    let runtime = PropertyRuntime::new(LeakyIntegratorPlan::new(0.5_f32, 0.75).unwrap()).unwrap();
    let schema = runtime.schema();
    assert_eq!(schema.info(0).unwrap().id, 0);
    assert_eq!(schema.info(1).unwrap().id, 1);
    assert_eq!(schema.info(3).unwrap().id, REQUESTED_GENERATION_ID);
    assert_eq!(schema.info(4).unwrap().id, ACTIVE_GENERATION_ID);
    assert_eq!(schema.info(2).unwrap().id, 2);
    assert!(schema.info(5).is_none());

    let mut empty = core::ptr::dangling_mut::<c_void>();
    // SAFETY: the empty transaction borrows no assignment storage and lends a
    // writable result pointer.
    assert_eq!(
        unsafe { runtime.prepare(core::ptr::null(), 0, &mut empty) },
        0
    );
    assert!(empty.is_null());
    // SAFETY: null is the adapter's empty-transaction token.
    unsafe { runtime.commit(empty) };
    assert_eq!(long(&runtime, REQUESTED_GENERATION_ID), 0);
    assert_eq!(long(&runtime, ACTIVE_GENERATION_ID), 0);

    let assignments = [float_property(0, -0.25), float_property(1, 0.5)];
    let mut prepared: *mut c_void = core::ptr::null_mut();
    // SAFETY: assignments and result live through each call.
    assert_eq!(
        unsafe {
            runtime.prepare(
                assignments.as_ptr(),
                assignments.len() as u32,
                &mut prepared,
            )
        },
        0
    );
    assert!(!prepared.is_null());
    assert_eq!(runtime.active_plan().gain(), 0.5);
    let mut second = core::ptr::null_mut();
    // SAFETY: assignments and result live through the call.
    assert_eq!(
        unsafe { runtime.prepare(assignments.as_ptr(), 2, &mut second) },
        -abi::EBUSY
    );
    assert_eq!(long(&runtime, REQUESTED_GENERATION_ID), 0);
    assert_eq!(long(&runtime, ACTIVE_GENERATION_ID), 0);

    assert_eq!(
        allocations(|| {
            // SAFETY: prepared came from this runtime and is consumed once.
            unsafe { runtime.commit(prepared) };
        }),
        (0, 0)
    );
    assert_eq!(runtime.active_plan().gain(), 0.5);
    assert_eq!(long(&runtime, REQUESTED_GENERATION_ID), 1);
    assert_eq!(long(&runtime, ACTIVE_GENERATION_ID), 0);
    assert_eq!(allocations(|| assert!(PlanStore::adopt(&runtime))), (0, 0));
    assert_eq!(runtime.active_plan().gain(), -0.25);
    assert_eq!(runtime.active_plan().pole(), 0.5);
    assert_eq!(long(&runtime, ACTIVE_GENERATION_ID), 1);

    let mut same_value = core::ptr::null_mut();
    // SAFETY: assignments and result live through the call.
    assert_eq!(
        unsafe { runtime.prepare(assignments.as_ptr(), 2, &mut same_value) },
        0
    );
    // SAFETY: same_value came from this runtime and is consumed once.
    unsafe { runtime.commit(same_value) };
    PlanStore::adopt(&runtime);
    assert_eq!(long(&runtime, REQUESTED_GENERATION_ID), 2);
    assert_eq!(long(&runtime, ACTIVE_GENERATION_ID), 2);

    let mut discarded = core::ptr::null_mut();
    // SAFETY: assignments and result live through the call.
    assert_eq!(
        unsafe { runtime.prepare(assignments.as_ptr(), 2, &mut discarded) },
        0
    );
    // SAFETY: discarded came from this runtime and is consumed once.
    unsafe { runtime.discard(discarded) };
    assert_eq!(long(&runtime, REQUESTED_GENERATION_ID), 2);
    assert_eq!(long(&runtime, ACTIVE_GENERATION_ID), 2);
}

#[test]
fn ndarray_formats_are_rejected_before_crossing_the_c_boundary() {
    assert!(matches!(
        NdarrayFormat::f32(Box::new([]), abi::LAYOUT_ROW_MAJOR, None),
        Err(PropertyAdapterError::InvalidFormat)
    ));
    assert!(matches!(
        NdarrayFormat::f32(Box::new([2]), u32::MAX, None),
        Err(PropertyAdapterError::InvalidFormat)
    ));
    assert!(matches!(
        NdarrayFormat::f32(Box::new([u32::MAX]), abi::LAYOUT_ROW_MAJOR, None),
        Err(PropertyAdapterError::InvalidFormat)
    ));
    assert!(matches!(
        NdarrayFormat::f32(Box::new([2]), abi::LAYOUT_ROW_MAJOR, Some("")),
        Err(PropertyAdapterError::InvalidFormat)
    ));
}

use core::ffi::CStr;
use filtergraph_algorithms::{
    AlgorithmPlan, AlgorithmProperties, NoWorkspace, PropertyAssignment, PropertyChoice,
    PropertyConstraint, PropertyInfo, PropertyUpdate, PropertyValue,
};

#[derive(Clone)]
struct ScalarPlan {
    info: PropertyInfo<'static>,
    value: Option<PropertyValue<'static>>,
    text: Option<String>,
}

impl AlgorithmPlan for ScalarPlan {
    type Input<'a> = ();
    type Output<'a> = ();
    type Workspace = NoWorkspace;
    type Error = ();

    fn process(&self, (): (), _: &mut NoWorkspace, (): ()) -> Result<(), ()> {
        Ok(())
    }
}

impl AlgorithmProperties for ScalarPlan {
    type PropertyError = ();

    fn property_info(&self, index: usize) -> Option<PropertyInfo<'static>> {
        (index == 0).then_some(self.info)
    }

    fn property(&self, id: u32) -> Result<PropertyValue<'_>, ()> {
        if id != self.info.id {
            return Err(());
        }
        if let Some(text) = &self.text {
            Ok(PropertyValue::String(text))
        } else {
            self.value.ok_or(())
        }
    }

    fn prepare_properties(&self, assignments: &[PropertyAssignment<'_>]) -> Result<Self, ()> {
        let [assignment] = assignments else {
            return Err(());
        };
        if assignment.id != self.info.id
            || self.info.update != PropertyUpdate::Runtime
            || !filtergraph_algorithms::property_value_satisfies(self.info, assignment.value)
        {
            return Err(());
        }
        let mut replacement = self.clone();
        replacement.value = Some(match assignment.value {
            PropertyValue::Bool(value) => PropertyValue::Bool(value),
            PropertyValue::Int(value) => PropertyValue::Int(value),
            PropertyValue::Long(value) => PropertyValue::Long(value),
            PropertyValue::Float(value) => PropertyValue::Float(value),
            PropertyValue::Double(value) => PropertyValue::Double(value),
            PropertyValue::Id(value) => PropertyValue::Id(value),
            PropertyValue::String(text) => {
                replacement.text = Some(text.to_owned());
                return Ok(replacement);
            }
        });
        Ok(replacement)
    }
}

fn scalar_info(value: PropertyValue<'static>) -> PropertyInfo<'static> {
    PropertyInfo {
        id: 9,
        name: "mode",
        description: "Selected detector mode",
        unit: "1",
        update: PropertyUpdate::Runtime,
        default_value: value,
        constraint: PropertyConstraint::None,
    }
}

#[test]
fn property_values_preserve_their_abi_type_and_survive_parameter_buffer_release() {
    use PropertyValue::*;
    for (type_, initial, expected, body) in [
        (
            abi::TYPE_BOOL,
            Bool(false),
            Bool(true),
            abi::ValueBody { boolean: 1 },
        ),
        (
            abi::TYPE_INT,
            Int(0),
            Int(i32::MIN),
            abi::ValueBody { integer: i32::MIN },
        ),
        (
            abi::TYPE_LONG,
            Long(0),
            Long(i64::MIN + 3),
            abi::ValueBody {
                long_integer: i64::MIN + 3,
            },
        ),
        (
            abi::TYPE_DOUBLE,
            Double(0.0),
            Double(1.0 + f64::EPSILON),
            abi::ValueBody {
                double_value: 1.0 + f64::EPSILON,
            },
        ),
        (
            abi::TYPE_ID,
            Id(0),
            Id(u32::MAX),
            abi::ValueBody { id: u32::MAX },
        ),
        (
            abi::TYPE_STRING,
            String("guided"),
            String("acquisition μs"),
            abi::ValueBody {
                string: c"guided".as_ptr(),
            },
        ),
    ] {
        let info = scalar_info(initial);
        let runtime = PropertyRuntime::new(ScalarPlan {
            info,
            value: Some(initial),
            text: None,
        })
        .unwrap();
        let mut token = core::ptr::null_mut();
        {
            // Exercise a host-owned UTF-8 value whose storage ends before adoption.
            let text = std::ffi::CString::new("acquisition μs").unwrap();
            let value = if type_ == abi::TYPE_STRING {
                abi::ValueBody {
                    string: text.as_ptr(),
                }
            } else {
                body
            };
            let assignment = abi::Property {
                id: info.id,
                reserved: 0,
                value: abi::Value {
                    type_,
                    reserved: 0,
                    value,
                },
            };
            assert_eq!(unsafe { runtime.prepare(&assignment, 1, &mut token) }, 0);
        }
        assert!(!token.is_null());
        assert_eq!(runtime.active_plan().property(info.id), Ok(initial));
        unsafe { runtime.commit(token) };
        assert_eq!(runtime.active_plan().property(info.id), Ok(initial));
        assert!(PlanStore::adopt(&runtime));
        assert_eq!(runtime.active_plan().property(info.id), Ok(expected));
        let result = runtime.get(info.id).unwrap();
        assert_eq!(result.type_, type_);
        // Each tag in this table determines the matching active union field.
        unsafe {
            match expected {
                Bool(expected) => assert_eq!(result.value.boolean, i32::from(expected)),
                Int(expected) => assert_eq!(result.value.integer, expected),
                Long(expected) => assert_eq!(result.value.long_integer, expected),
                Double(expected) => assert_eq!(result.value.double_value, expected),
                Id(expected) => assert_eq!(result.value.id, expected),
                String(_) => assert_eq!(CStr::from_ptr(result.value.string), c"acquisition μs"),
                Float(_) => unreachable!(),
            }
        }
    }
}

#[test]
fn property_schema_reports_reserved_names_and_projects_choice_metadata() {
    use PropertyUpdate::*;
    use PropertyValue::*;
    const CHOICES: &[PropertyChoice<'static>] = &[
        PropertyChoice {
            value: Id(0),
            name: "dark",
            description: "Dark exposure",
        },
        PropertyChoice {
            value: Id(1),
            name: "open",
            description: "Open exposure",
        },
    ];
    let base = scalar_info(Id(0));
    for (update, flag) in [
        (ReadOnly, abi::PROPERTY_READONLY),
        (Runtime, abi::PROPERTY_RUNTIME),
        (Construction, abi::PROPERTY_CONSTRUCTION),
        (GraphRebuild, abi::PROPERTY_GRAPH_REBUILD),
    ] {
        let info = PropertyInfo {
            update,
            constraint: PropertyConstraint::Choices(CHOICES),
            ..base
        };
        let runtime = PropertyRuntime::new(ScalarPlan {
            info,
            value: Some(Id(1)),
            text: None,
        })
        .unwrap();
        let projected = runtime.schema().info(0).unwrap();
        assert_eq!(projected.flags, flag | abi::PROPERTY_CHOICES);
        assert_eq!(projected.n_choices, 2);
        assert_eq!(projected.default_value.type_, abi::TYPE_ID);
        unsafe {
            let choices = core::slice::from_raw_parts(projected.choices, 2);
            assert_eq!(CStr::from_ptr(choices[1].name), c"open");
            assert_eq!(CStr::from_ptr(choices[1].description), c"Open exposure");
            assert_eq!(choices[1].value.type_, abi::TYPE_ID);
            assert_eq!(choices[1].value.value.id, 1);
        }
    }

    for (info, value, expected, detail) in [
        (
            PropertyInfo {
                name: "active-generation",
                ..base
            },
            Some(Id(0)),
            PropertyAdapterError::ReservedProperty { id: 9 },
            "id 9 is reserved",
        ),
        (
            PropertyInfo {
                description: "invalid\0description",
                ..base
            },
            Some(Id(0)),
            PropertyAdapterError::InteriorNul,
            "interior NUL",
        ),
        (
            PropertyInfo {
                constraint: PropertyConstraint::Choices(CHOICES),
                ..base
            },
            Some(Id(2)),
            PropertyAdapterError::InvalidCurrentValue { id: 9 },
            "id 9 has an invalid current value",
        ),
        (
            base,
            None,
            PropertyAdapterError::InvalidCurrentValue { id: 9 },
            "id 9 has an invalid current value",
        ),
    ] {
        let error = PropertyRuntime::new(ScalarPlan {
            info,
            value,
            text: None,
        })
        .err()
        .unwrap();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(detail), "{error}");
    }
}

#[test]
fn property_update_rejects_bad_abi_strings_without_occupying_publication_slot() {
    let value = PropertyValue::String("guided");
    let info = scalar_info(value);
    let runtime = PropertyRuntime::new(ScalarPlan {
        info,
        value: Some(value),
        text: None,
    })
    .unwrap();
    for pointer in [core::ptr::null(), c"\xff".as_ptr()] {
        let assignment = abi::Property {
            id: info.id,
            reserved: 0,
            value: abi::Value {
                type_: abi::TYPE_STRING,
                reserved: 0,
                value: abi::ValueBody { string: pointer },
            },
        };
        let mut token = core::ptr::null_mut();
        assert_eq!(
            unsafe { runtime.prepare(&assignment, 1, &mut token) },
            -abi::EINVAL
        );
        assert!(token.is_null());
        assert_eq!(runtime.active_plan().property(info.id), Ok(value));
        assert_eq!(runtime.revision(), 0);
    }
    let assignment = abi::Property {
        id: info.id,
        reserved: 0,
        value: abi::Value {
            type_: abi::TYPE_STRING,
            reserved: 0,
            value: abi::ValueBody {
                string: c"open".as_ptr(),
            },
        },
    };
    let mut token = core::ptr::null_mut();
    assert_eq!(unsafe { runtime.prepare(&assignment, 1, &mut token) }, 0);
    unsafe { runtime.discard(token) };
    assert_eq!(runtime.active_plan().property(info.id), Ok(value));
}
