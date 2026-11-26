//! Unified scalar-property and ndarray-parameter publication conformance.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use filtergraph_algorithms::{
    AlgorithmPlan, AlgorithmProperties, NoWorkspace, PropertyAssignment, PropertyConstraint,
    PropertyInfo, PropertyUpdate, PropertyValue,
};
use filtergraph_fgn::{
    ACTIVE_GENERATION_ID, ACTIVE_PARAMETER_SEQUENCE_ID, FgnProperties, PlanStore,
    PropertyParameterRuntime, REQUESTED_GENERATION_ID, REQUESTED_PARAMETER_SEQUENCE_ID, abi,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static DEALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegation preserves the global allocator contract.
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
        // SAFETY: pointer and layout came from the delegated allocator.
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

#[derive(Clone)]
struct MixedPlan {
    gain: f32,
    parameter: Arc<[f32]>,
}

impl AlgorithmPlan for MixedPlan {
    type Input<'a> = &'a [f32];
    type Output<'a> = &'a mut [f32];
    type Workspace = NoWorkspace;
    type Error = ();

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if input.len() != output.len() || input.len() != self.parameter.len() {
            return Err(());
        }
        for ((output, input), parameter) in output.iter_mut().zip(input).zip(&*self.parameter) {
            *output = self.gain * input * parameter;
        }
        Ok(())
    }
}

impl AlgorithmProperties for MixedPlan {
    type PropertyError = ();

    fn property_info(&self, index: usize) -> Option<PropertyInfo<'static>> {
        (index == 0).then_some(PropertyInfo {
            id: 0,
            name: "gain",
            description: "Dimensionless numerical gain",
            unit: "1",
            update: PropertyUpdate::Runtime,
            default_value: PropertyValue::Float(1.0),
            constraint: PropertyConstraint::Range {
                minimum: PropertyValue::Float(-10.0),
                maximum: PropertyValue::Float(10.0),
            },
        })
    }

    fn property(&self, id: u32) -> Result<PropertyValue<'_>, Self::PropertyError> {
        match id {
            0 => Ok(PropertyValue::Float(self.gain)),
            _ => Err(()),
        }
    }

    fn prepare_properties(
        &self,
        assignments: &[PropertyAssignment<'_>],
    ) -> Result<Self, Self::PropertyError> {
        let mut gain = self.gain;
        let mut assigned = false;
        for assignment in assignments {
            if assignment.id != 0 || assigned {
                return Err(());
            }
            let PropertyValue::Float(value) = assignment.value else {
                return Err(());
            };
            if !value.is_finite() || !(-10.0..=10.0).contains(&value) {
                return Err(());
            }
            gain = value;
            assigned = true;
        }
        Ok(Self {
            gain,
            parameter: Arc::clone(&self.parameter),
        })
    }
}

fn float_property(value: f32) -> abi::Property {
    abi::Property {
        id: 0,
        reserved: 0,
        value: abi::Value {
            type_: abi::TYPE_FLOAT,
            reserved: 0,
            value: abi::ValueBody { float_value: value },
        },
    }
}

fn long(runtime: &PropertyParameterRuntime<MixedPlan>, id: u32) -> i64 {
    let value = runtime.property(id).unwrap();
    assert_eq!(value.type_, abi::TYPE_LONG);
    // SAFETY: the checked type selects the long union member.
    unsafe { value.value.long_integer }
}

#[test]
fn property_and_parameter_updates_share_one_bounded_plan_owner() {
    let initial_parameter: Arc<[f32]> = Arc::from([1.0, 2.0]);
    let runtime = PropertyParameterRuntime::new(MixedPlan {
        gain: 1.0,
        parameter: Arc::clone(&initial_parameter),
    })
    .unwrap();

    assert_eq!(runtime.property_info(0).unwrap().id, 0);
    assert_eq!(
        runtime.property_info(1).unwrap().id,
        REQUESTED_GENERATION_ID
    );
    assert_eq!(runtime.property_info(2).unwrap().id, ACTIVE_GENERATION_ID);
    assert_eq!(
        runtime.property_info(3).unwrap().id,
        REQUESTED_PARAMETER_SEQUENCE_ID
    );
    assert_eq!(
        runtime.property_info(4).unwrap().id,
        ACTIVE_PARAMETER_SEQUENCE_ID
    );
    assert!(runtime.property_info(5).is_none());

    let assignment = float_property(2.0);
    let mut property_token = core::ptr::null_mut();
    // SAFETY: assignment and result remain valid through the callback.
    assert_eq!(
        unsafe { runtime.prepare_properties(&assignment, 1, &mut property_token) },
        0
    );
    assert_eq!(
        runtime.prepare_parameter_with(Some(1), |_| {
            Ok::<_, ()>(MixedPlan {
                gain: 1.0,
                parameter: Arc::from([3.0, 4.0]),
            })
        }),
        Err(abi::EBUSY)
    );

    assert_eq!(
        allocations(|| {
            // SAFETY: property_token belongs to this runtime and is consumed once.
            unsafe { runtime.commit_properties(property_token) };
        }),
        (0, 0)
    );
    assert_eq!(long(&runtime, REQUESTED_GENERATION_ID), 1);
    assert_eq!(long(&runtime, ACTIVE_GENERATION_ID), 0);
    assert_eq!(runtime.active_plan().gain, 1.0);
    assert_eq!(
        runtime.prepare_parameter_with(Some(1), |_| -> Result<MixedPlan, ()> { unreachable!() }),
        Err(abi::EBUSY)
    );

    assert_eq!(allocations(|| assert!(PlanStore::adopt(&runtime))), (0, 0));
    assert_eq!(runtime.active_plan().gain, 2.0);
    assert!(Arc::ptr_eq(
        &runtime.active_plan().parameter,
        &initial_parameter
    ));
    assert_eq!(long(&runtime, ACTIVE_GENERATION_ID), 1);

    let replacement_parameter: Arc<[f32]> = Arc::from([3.0, 4.0]);
    let parameter_token = runtime
        .prepare_parameter_with(Some(7), |active| {
            Ok::<_, ()>(MixedPlan {
                gain: active.gain,
                parameter: Arc::clone(&replacement_parameter),
            })
        })
        .unwrap();
    let mut blocked_property = core::ptr::null_mut::<c_void>();
    // SAFETY: assignment and result remain valid through the callback.
    assert_eq!(
        unsafe { runtime.prepare_properties(&assignment, 1, &mut blocked_property) },
        -abi::EBUSY
    );
    assert!(blocked_property.is_null());
    assert_eq!(
        allocations(|| {
            // SAFETY: parameter_token belongs to this runtime and is consumed once.
            unsafe { runtime.commit_parameter(parameter_token) };
        }),
        (0, 0)
    );
    assert_eq!(long(&runtime, REQUESTED_PARAMETER_SEQUENCE_ID), 7);
    assert_eq!(long(&runtime, ACTIVE_PARAMETER_SEQUENCE_ID), 0);
    assert_eq!(long(&runtime, REQUESTED_GENERATION_ID), 1);

    assert_eq!(allocations(|| assert!(PlanStore::adopt(&runtime))), (0, 0));
    assert_eq!(runtime.active_plan().gain, 2.0);
    assert!(Arc::ptr_eq(
        &runtime.active_plan().parameter,
        &replacement_parameter
    ));
    assert_eq!(long(&runtime, ACTIVE_PARAMETER_SEQUENCE_ID), 7);
    assert_eq!(long(&runtime, ACTIVE_GENERATION_ID), 1);

    assert_eq!(
        runtime.prepare_parameter_with(Some(8), |active| {
            Ok::<_, ()>(MixedPlan {
                gain: active.gain + 1.0,
                parameter: Arc::from([5.0, 6.0]),
            })
        }),
        Err(abi::EINVAL)
    );
    assert_eq!(runtime.active_plan().gain, 2.0);
    assert_eq!(long(&runtime, REQUESTED_PARAMETER_SEQUENCE_ID), 7);
}

#[test]
fn mixed_updates_remain_coherent_during_concurrent_process_adoption() {
    const UPDATES_PER_KIND: u64 = 2_000;

    let runtime = Arc::new(
        PropertyParameterRuntime::new(MixedPlan {
            gain: 1.0,
            parameter: Arc::from([0.0]),
        })
        .unwrap(),
    );
    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(2));
    let control_runtime = Arc::clone(&runtime);
    let control_done = Arc::clone(&done);
    let control_start = Arc::clone(&start);
    let control = thread::spawn(move || {
        control_start.wait();
        for update in 1..=UPDATES_PER_KIND * 2 {
            if update & 1 == 1 {
                let property = float_property((update % 10) as f32);
                loop {
                    let mut prepared = core::ptr::null_mut();
                    // SAFETY: property and result remain valid through this callback.
                    let result =
                        unsafe { control_runtime.prepare_properties(&property, 1, &mut prepared) };
                    if result == 0 {
                        // SAFETY: this serial owner consumes its token exactly once.
                        unsafe { control_runtime.commit_properties(prepared) };
                        break;
                    }
                    assert_eq!(result, -abi::EBUSY);
                    thread::yield_now();
                }
            } else {
                let sequence = update / 2;
                loop {
                    match control_runtime.prepare_parameter_with(None, |active| {
                        Ok::<_, ()>(MixedPlan {
                            gain: active.gain,
                            parameter: Arc::from([sequence as f32]),
                        })
                    }) {
                        Ok(prepared) => {
                            // SAFETY: this serial owner consumes its token exactly once.
                            unsafe { control_runtime.commit_parameter(prepared) };
                            break;
                        }
                        Err(error) => {
                            assert_eq!(error, abi::EBUSY);
                            thread::yield_now();
                        }
                    }
                }
            }
        }
        control_done.store(true, Ordering::Release);
    });

    let mut previous_requested_generation = 0_i64;
    let mut previous_active_generation = 0_i64;
    let mut previous_requested_sequence = 0_i64;
    let mut previous_active_sequence = 0_i64;
    start.wait();
    while !done.load(Ordering::Acquire)
        || long(&runtime, ACTIVE_GENERATION_ID) != UPDATES_PER_KIND as i64
        || long(&runtime, ACTIVE_PARAMETER_SEQUENCE_ID) != UPDATES_PER_KIND as i64
    {
        PlanStore::adopt(&*runtime);
        loop {
            let before = runtime.revision();
            if before & 1 != 0 {
                continue;
            }
            let requested_generation = long(&runtime, REQUESTED_GENERATION_ID);
            let active_generation = long(&runtime, ACTIVE_GENERATION_ID);
            let requested_sequence = long(&runtime, REQUESTED_PARAMETER_SEQUENCE_ID);
            let active_sequence = long(&runtime, ACTIVE_PARAMETER_SEQUENCE_ID);
            let after = runtime.revision();
            if before == after && after & 1 == 0 {
                assert!(requested_generation >= previous_requested_generation);
                assert!(active_generation >= previous_active_generation);
                assert!(active_generation <= requested_generation);
                assert!(requested_sequence >= previous_requested_sequence);
                assert!(active_sequence >= previous_active_sequence);
                assert!(active_sequence <= requested_sequence);
                previous_requested_generation = requested_generation;
                previous_active_generation = active_generation;
                previous_requested_sequence = requested_sequence;
                previous_active_sequence = active_sequence;
                break;
            }
        }
    }
    control.join().unwrap();
    assert_eq!(runtime.active_plan().parameter[0], UPDATES_PER_KIND as f32);
}
