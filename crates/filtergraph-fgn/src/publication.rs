//! Shared two-slot publication for immutable FilterGraph plans.

use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::mem::MaybeUninit;
use core::ptr;
use core::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::abi;

const SLOT_FREE: u8 = 0;
const SLOT_ACTIVE: u8 = 1;
const SLOT_WRITING: u8 = 2;
const SLOT_PREPARED: u8 = 3;
const SLOT_REQUESTED: u8 = 4;
const SLOT_RETIRED: u8 = 5;
const SLOT_NONE: u32 = u32::MAX;

struct Slot<P, K> {
    state: AtomicU8,
    plan: UnsafeCell<MaybeUninit<P>>,
    kind: UnsafeCell<MaybeUninit<K>>,
}

impl<P, K> Slot<P, K> {
    const fn new() -> Self {
        Self {
            state: AtomicU8::new(SLOT_FREE),
            plan: UnsafeCell::new(MaybeUninit::uninit()),
            kind: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

/// One active plan plus at most one prepared or requested replacement.
///
/// The serial control owner prepares and reclaims slots. The data-loop owner
/// publishes property replacements and adopts every replacement; the serial
/// parameter owner publishes parameter replacements. Callers own the
/// observable generation or sequence counters because their meaning depends
/// on the update kind.
pub(crate) struct Publication<P, K> {
    slots: [Slot<P, K>; 2],
    active_slot: AtomicU32,
    requested_slot: AtomicU32,
}

// SAFETY: the serial owner writes only FREE/WRITING slots. Release publication
// makes the complete plan and kind visible to the data-loop owner, which reads
// ACTIVE/REQUESTED slots. Only the serial owner destroys RETIRED values.
unsafe impl<P, K> Sync for Publication<P, K>
where
    P: Send + Sync,
    K: Copy + Send + Sync,
{
}

impl<P, K> Publication<P, K>
where
    P: Send + Sync,
    K: Copy + Send + Sync,
{
    pub(crate) fn new(plan: P) -> Self {
        let publication = Self {
            slots: [Slot::new(), Slot::new()],
            active_slot: AtomicU32::new(0),
            requested_slot: AtomicU32::new(SLOT_NONE),
        };
        // SAFETY: the unpublished runtime owns uninitialized slot zero.
        unsafe {
            (*publication.slots[0].plan.get()).write(plan);
        }
        publication.slots[0]
            .state
            .store(SLOT_ACTIVE, Ordering::Relaxed);
        publication
    }

    pub(crate) fn active(&self) -> &P {
        let index = self.active_slot.load(Ordering::Acquire) as usize;
        // SAFETY: active_slot names one initialized ACTIVE slot. A retired slot
        // is not published until after active_slot changes.
        unsafe { (&*self.slots[index].plan.get()).assume_init_ref() }
    }

    fn reclaim_retired(&self) {
        for slot in &self.slots {
            if slot
                .state
                .compare_exchange(
                    SLOT_RETIRED,
                    SLOT_WRITING,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                // SAFETY: RETIRED is no longer reachable by the data loop and
                // this serial owner exclusively holds WRITING.
                unsafe {
                    (&mut *slot.plan.get()).assume_init_drop();
                }
                slot.state.store(SLOT_FREE, Ordering::Release);
            }
        }
    }

    pub(crate) fn prepare<E>(
        &self,
        kind: K,
        prepare: impl FnOnce(&P) -> Result<P, E>,
    ) -> Result<*mut c_void, i32> {
        self.reclaim_retired();
        if self.requested_slot.load(Ordering::Acquire) != SLOT_NONE {
            return Err(abi::EBUSY);
        }
        let Some(index) = self
            .slots
            .iter()
            .position(|slot| slot.state.load(Ordering::Acquire) == SLOT_FREE)
        else {
            return Err(abi::EBUSY);
        };
        let slot = &self.slots[index];
        if slot
            .state
            .compare_exchange(
                SLOT_FREE,
                SLOT_WRITING,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return Err(abi::EBUSY);
        }
        let plan = match catch_unwind(AssertUnwindSafe(|| prepare(self.active()))) {
            Ok(Ok(plan)) => plan,
            Ok(Err(_)) => {
                slot.state.store(SLOT_FREE, Ordering::Release);
                return Err(abi::EINVAL);
            }
            Err(_) => {
                slot.state.store(SLOT_FREE, Ordering::Release);
                return Err(abi::EFAULT);
            }
        };
        // SAFETY: the serial owner exclusively owns this WRITING slot.
        unsafe {
            (*slot.plan.get()).write(plan);
            (*slot.kind.get()).write(kind);
        }
        slot.state.store(SLOT_PREPARED, Ordering::Release);
        Ok(ptr::from_ref(slot).cast_mut().cast())
    }

    fn slot_index(&self, token: *mut c_void) -> Option<usize> {
        self.slots
            .iter()
            .position(|slot| ptr::from_ref(slot).cast::<c_void>() == token.cast_const())
    }

    /// Publish a prepared replacement for adoption by a later process call.
    ///
    /// `publish` updates the requested observation before the requested slot
    /// receives its release publication.
    ///
    /// # Safety
    ///
    /// `token` belongs to this publication and is consumed exactly once.
    pub(crate) unsafe fn commit(&self, token: *mut c_void, publish: impl FnOnce(&mut K)) {
        let Some(index) = self.slot_index(token) else {
            return;
        };
        let slot = &self.slots[index];
        if slot
            .state
            .compare_exchange(
                SLOT_PREPARED,
                SLOT_REQUESTED,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return;
        }
        // SAFETY: this callback owner exclusively consumes the PREPARED slot;
        // requested_slot has not published it to adoption yet.
        publish(unsafe { (&mut *slot.kind.get()).assume_init_mut() });
        self.requested_slot.store(index as u32, Ordering::Release);
    }

    /// Destroy one unpublished replacement on the serial control path.
    ///
    /// # Safety
    ///
    /// `token` belongs to this publication and is consumed exactly once.
    pub(crate) unsafe fn discard(&self, token: *mut c_void) {
        let Some(index) = self.slot_index(token) else {
            return;
        };
        let slot = &self.slots[index];
        if slot
            .state
            .compare_exchange(
                SLOT_PREPARED,
                SLOT_WRITING,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return;
        }
        // SAFETY: the serial owner exclusively owns this WRITING slot.
        unsafe {
            (&mut *slot.plan.get()).assume_init_drop();
        }
        slot.state.store(SLOT_FREE, Ordering::Release);
    }

    /// Adopt one requested replacement immediately before processing.
    ///
    /// `begin` and `finish` bracket the active-plan change for externally
    /// observable counters. They must not allocate, block, destroy, or panic.
    pub(crate) fn adopt(&self, begin: impl FnOnce(), finish: impl FnOnce(&K)) -> bool {
        let requested = self.requested_slot.swap(SLOT_NONE, Ordering::AcqRel);
        if requested == SLOT_NONE {
            return false;
        }
        let requested = requested as usize;
        let slot = &self.slots[requested];
        if slot
            .state
            .compare_exchange(
                SLOT_REQUESTED,
                SLOT_ACTIVE,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return false;
        }
        // SAFETY: acquire observation of requested_slot admits the initialized
        // kind, and the slot remains ACTIVE through this call.
        let kind = unsafe { (&*slot.kind.get()).assume_init_ref() };
        begin();
        let previous = self.active_slot.swap(requested as u32, Ordering::AcqRel) as usize;
        finish(kind);
        // Reclamation becomes visible only after the complete observation
        // bracket closes, so no plan is destroyed on this data-loop call.
        self.slots[previous]
            .state
            .store(SLOT_RETIRED, Ordering::Release);
        true
    }
}

impl<P, K> Drop for Publication<P, K> {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            if matches!(
                *slot.state.get_mut(),
                SLOT_ACTIVE | SLOT_PREPARED | SLOT_REQUESTED | SLOT_RETIRED
            ) {
                // SAFETY: cleanup runs after all publication callbacks stop.
                unsafe {
                    slot.plan.get_mut().assume_init_drop();
                }
            }
        }
    }
}
