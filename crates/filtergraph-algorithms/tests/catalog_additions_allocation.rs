//! Zero-allocation prepared processing for current JFG catalog additions.
use filtergraph_algorithms::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if COUNTING.load(Ordering::Relaxed) && !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if COUNTING.load(Ordering::Relaxed) && !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if COUNTING.load(Ordering::Relaxed) && !new_pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        new_pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations(operation: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    DEALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    operation();
    COUNTING.store(false, Ordering::Relaxed);
    assert_eq!(DEALLOCATIONS.load(Ordering::Relaxed), 0);
    ALLOCATIONS.load(Ordering::Relaxed)
}

#[test]
fn current_catalog_process_reject_hold_commit_discard_allocate_nothing() {
    let p = PhysicalOffsetReferencePlan::new(3, 2, 2, -1.0)
        .unwrap()
        .with_physical_to_modes(&[1.0, 2.0, -0.5, 3.0])
        .unwrap()
        .with_modes_to_reference(&[1.0, -2.0, 0.5, 1.0, -3.0, 0.25])
        .unwrap()
        .with_offset(&[0.25, -0.5])
        .unwrap();
    let mut w = p.workspace();
    let (mut y, mut offset) = ([0.0; 3], [0.0; 2]);
    p.process((&mut y, &mut offset), &mut w, &[1.0, -2.0, 0.125])
        .unwrap();
    assert_eq!(
        allocations(|| {
            for _ in 0..100 {
                p.process((&mut y, &mut offset), &mut w, &[1.0, -2.0, 0.125])
                    .unwrap();
                assert!(
                    p.process((&mut y, &mut offset), &mut w, &[f32::NAN; 3])
                        .is_err()
                );
                w.reset();
            }
        }),
        0
    );
    let mut p = PixelValidityF32::prepare(PixelValidityF32Configuration {
        extent: 3,
        input_schema: "test.pixels/1".into(),
    })
    .unwrap();
    let mut validity = [false; 3];
    PixelValidityF32::process(&mut p, &[0.0, f32::NAN, f32::INFINITY], &mut validity).unwrap();
    assert_eq!(
        allocations(|| {
            for _ in 0..100 {
                PixelValidityF32::process(&mut p, &[0.0, f32::NAN, f32::INFINITY], &mut validity)
                    .unwrap();
            }
        }),
        0
    );
    for mode in ["finite", "sliding", "exponential"] {
        let p = ArrayStatisticsPlan::new(vec![2, 2], mode, 3, 2.5, 0.2, 268435456).unwrap();
        let mut w = p.workspace();
        let (mut mean, mut variance, mut stddev) = ([0.0; 4], [0.0; 4], [0.0; 4]);
        let (mut count, mut complete, mut valid) = ([0], [false], [false]);
        p.process(
            (
                &mut mean,
                &mut variance,
                &mut stddev,
                &mut count,
                &mut complete,
                &mut valid,
            ),
            &mut w,
            &[1.0; 4],
        )
        .unwrap();
        assert_eq!(
            allocations(|| {
                for _ in 0..100 {
                    w.reset();
                    p.stage(
                        (
                            &mut mean,
                            &mut variance,
                            &mut stddev,
                            &mut count,
                            &mut complete,
                            &mut valid,
                        ),
                        &mut w,
                        &[2.0; 4],
                    )
                    .unwrap();
                    w.discard();
                    w.commit();
                    p.stage(
                        (
                            &mut mean,
                            &mut variance,
                            &mut stddev,
                            &mut count,
                            &mut complete,
                            &mut valid,
                        ),
                        &mut w,
                        &[3.0; 4],
                    )
                    .unwrap();
                    w.commit();
                    assert!(
                        p.process(
                            (
                                &mut mean,
                                &mut variance,
                                &mut stddev,
                                &mut count,
                                &mut complete,
                                &mut valid
                            ),
                            &mut w,
                            &[f32::NAN; 4]
                        )
                        .is_err()
                    );
                    assert!(
                        p.process(
                            (
                                &mut mean,
                                &mut variance,
                                &mut stddev,
                                &mut count,
                                &mut complete,
                                &mut valid
                            ),
                            &mut w,
                            &[f32::MAX; 4]
                        )
                        .is_err()
                    );
                    for sample in [1.0, 4.0, 7.0, 9.0] {
                        p.process(
                            (
                                &mut mean,
                                &mut variance,
                                &mut stddev,
                                &mut count,
                                &mut complete,
                                &mut valid,
                            ),
                            &mut w,
                            &[sample; 4],
                        )
                        .unwrap();
                    }
                }
            }),
            0
        );
    }
    let mut p = ValidityGatedClosedLoopIntegratorF32::prepare(
        ValidityGatedClosedLoopIntegratorF32Configuration {
            state_length: 2,
            validity_length: 2,
            validity_schema: "test.validity/1".into(),
        },
    )
    .unwrap();
    let (mut correction, mut state) = ([0.0; 2], [0.0; 2]);
    ValidityGatedClosedLoopIntegratorF32::process(
        &mut p,
        &[1.0; 2],
        &[0.0; 2],
        &[true; 2],
        &mut correction,
        &mut state,
    )
    .unwrap();
    assert_eq!(
        allocations(|| {
            for _ in 0..100 {
                ValidityGatedClosedLoopIntegratorF32::process(
                    &mut p,
                    &[1.0; 2],
                    &[0.0; 2],
                    &[true; 2],
                    &mut correction,
                    &mut state,
                )
                .unwrap();
                ValidityGatedClosedLoopIntegratorF32::process(
                    &mut p,
                    &[f32::NAN; 2],
                    &[f32::NAN; 2],
                    &[false, true],
                    &mut correction,
                    &mut state,
                )
                .unwrap();
                assert!(
                    ValidityGatedClosedLoopIntegratorF32::process(
                        &mut p,
                        &[f32::NAN; 2],
                        &[0.0; 2],
                        &[true; 2],
                        &mut correction,
                        &mut state
                    )
                    .is_err()
                );
                p.reset();
            }
        }),
        0
    );
}
