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
