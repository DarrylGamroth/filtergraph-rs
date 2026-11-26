//! Bounded ndarray-parameter plan publication tests.

use core::ffi::CStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use filtergraph_fgn::{
    ACTIVE_PARAMETER_SEQUENCE_ID, FgnProperties, ParameterRuntime, PlanStore,
    REQUESTED_PARAMETER_SEQUENCE_ID, abi,
};

fn long_property<P: Send + Sync>(runtime: &ParameterRuntime<P>, id: u32) -> i64 {
    let value = runtime.property(id).unwrap();
    assert_eq!(value.type_, abi::TYPE_LONG);
    // SAFETY: the checked type selects the long union member.
    unsafe { value.value.long_integer }
}

#[test]
fn parameter_plans_publish_with_backpressure_and_process_boundary_adoption() {
    let runtime = ParameterRuntime::new(vec![1.0_f32, 1.0]);
    let requested = runtime.property_info(0).unwrap();
    let active = runtime.property_info(1).unwrap();
    // SAFETY: descriptor strings have static adapter lifetime.
    assert_eq!(
        unsafe { CStr::from_ptr(requested.name) },
        c"requested-parameter-sequence"
    );
    // SAFETY: descriptor strings have static adapter lifetime.
    assert_eq!(
        unsafe { CStr::from_ptr(active.name) },
        c"active-parameter-sequence"
    );
    assert!(runtime.property_info(2).is_none());

    let prepared = runtime
        .prepare_with(7, |_| Ok::<_, ()>(vec![2.0, 4.0]))
        .unwrap();
    // SAFETY: consume the token once on the same runtime.
    unsafe { runtime.commit(prepared) };
    assert_eq!(long_property(&runtime, REQUESTED_PARAMETER_SEQUENCE_ID), 7);
    assert_eq!(long_property(&runtime, ACTIVE_PARAMETER_SEQUENCE_ID), 0);
    assert_eq!(
        runtime.prepare_with(8, |_| Ok::<_, ()>(vec![8.0, 8.0])),
        Err(abi::EBUSY)
    );

    PlanStore::adopt(&runtime);
    assert_eq!(runtime.active_plan(), &[2.0, 4.0]);
    assert_eq!(long_property(&runtime, ACTIVE_PARAMETER_SEQUENCE_ID), 7);

    let discarded = runtime
        .prepare_with(8, |_| Ok::<_, ()>(vec![8.0, 8.0]))
        .unwrap();
    // SAFETY: discard the unpublished token once.
    unsafe { runtime.discard(discarded) };
    assert_eq!(runtime.active_plan(), &[2.0, 4.0]);
    assert_eq!(long_property(&runtime, ACTIVE_PARAMETER_SEQUENCE_ID), 7);
}

#[test]
fn rejected_or_panicking_parameter_preparation_releases_capacity() {
    let runtime = ParameterRuntime::new(vec![1.0_f32]);
    assert_eq!(
        runtime.prepare_with(1, |_| Err::<Vec<f32>, _>(())),
        Err(abi::EINVAL)
    );
    assert_eq!(
        runtime.prepare_with(i64::MAX as u64 + 1, |_| Ok::<_, ()>(vec![2.0])),
        Err(abi::EOVERFLOW)
    );
    assert_eq!(
        runtime.prepare_with(1, |_| -> Result<Vec<f32>, ()> {
            panic!("parameter preparation panic")
        }),
        Err(abi::EFAULT)
    );

    let prepared = runtime
        .prepare_with(1, |_| Ok::<_, ()>(vec![3.0]))
        .expect("capacity must recover after failure");
    // SAFETY: discard the unpublished token once.
    unsafe { runtime.discard(prepared) };
}

#[test]
fn exhausted_automatic_sequence_releases_the_claimed_slot() {
    let runtime = ParameterRuntime::new(0_u64);
    let prepared = runtime
        .prepare_with(i64::MAX as u64, |_| Ok::<_, ()>(1))
        .unwrap();
    // SAFETY: consume the token once on the same runtime.
    unsafe { runtime.commit(prepared) };
    PlanStore::adopt(&runtime);

    assert_eq!(
        runtime.prepare_next_with(|_| Ok::<_, ()>(2)),
        Err(abi::EOVERFLOW)
    );
    let prepared = runtime
        .prepare_with(1, |_| Ok::<_, ()>(3))
        .expect("sequence overflow must not strand the inactive slot");
    // SAFETY: discard the unpublished token once.
    unsafe { runtime.discard(prepared) };
}

#[test]
fn headerless_sequences_remain_monotonic_during_concurrent_adoption() {
    let runtime = Arc::new(ParameterRuntime::new(0_u64));
    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(2));
    let control_runtime = Arc::clone(&runtime);
    let control_done = Arc::clone(&done);
    let control_start = Arc::clone(&start);
    let control = thread::spawn(move || {
        control_start.wait();
        for expected in 1_u64..=10_000 {
            loop {
                match control_runtime.prepare_next_with(|_| Ok::<_, ()>(expected)) {
                    Ok(prepared) => {
                        // SAFETY: this serial control owner consumes its token once.
                        unsafe { control_runtime.commit(prepared) };
                        assert_eq!(
                            long_property(&control_runtime, REQUESTED_PARAMETER_SEQUENCE_ID),
                            expected as i64
                        );
                        break;
                    }
                    Err(error) => {
                        assert_eq!(error, abi::EBUSY);
                        thread::yield_now();
                    }
                }
            }
        }
        control_done.store(true, Ordering::Release);
    });

    let mut previous_requested = 0_i64;
    let mut previous_active = 0_i64;
    start.wait();
    while !done.load(Ordering::Acquire)
        || long_property(&runtime, ACTIVE_PARAMETER_SEQUENCE_ID) != 10_000
    {
        PlanStore::adopt(&*runtime);
        loop {
            let before = runtime.revision();
            if before & 1 != 0 {
                continue;
            }
            let requested = long_property(&runtime, REQUESTED_PARAMETER_SEQUENCE_ID);
            let active = long_property(&runtime, ACTIVE_PARAMETER_SEQUENCE_ID);
            let after = runtime.revision();
            if before == after && after & 1 == 0 {
                assert!(requested >= previous_requested);
                assert!(active >= previous_active);
                assert!(active <= requested);
                previous_requested = requested;
                previous_active = active;
                break;
            }
        }
    }
    control.join().unwrap();
    assert_eq!(*runtime.active_plan(), 10_000);
}
