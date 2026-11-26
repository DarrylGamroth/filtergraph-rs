//! Bounded publication of complete plans prepared from ndarray parameters.

use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{FgnProperties, PlanStore, abi, publication::Publication};

/// Read-only property ID for the latest accepted parameter sequence.
pub const REQUESTED_PARAMETER_SEQUENCE_ID: u32 = u32::MAX - 2;
/// Read-only property ID for the parameter sequence used by processing.
pub const ACTIVE_PARAMETER_SEQUENCE_ID: u32 = u32::MAX - 3;

pub(crate) const SEQUENCE_MAX: u64 = i64::MAX as u64;

/// Two-slot plan publication driven by sparse ndarray parameter updates.
pub struct ParameterRuntime<P> {
    publication: Publication<P, u64>,
    requested_sequence: AtomicU64,
    active_sequence: AtomicU64,
    revision: AtomicU64,
}

impl<P> ParameterRuntime<P>
where
    P: Send + Sync,
{
    /// Construct a runtime with one initial active plan at sequence zero.
    pub fn new(plan: P) -> Self {
        Self {
            publication: Publication::new(plan),
            requested_sequence: AtomicU64::new(0),
            active_sequence: AtomicU64::new(0),
            revision: AtomicU64::new(0),
        }
    }

    fn active_plan_ref(&self) -> &P {
        self.publication.active()
    }

    /// Prepare a complete replacement plan on the serial parameter path.
    pub fn prepare_with<E>(
        &self,
        sequence: u64,
        prepare: impl FnOnce(&P) -> Result<P, E>,
    ) -> Result<*mut c_void, i32> {
        self.prepare_optional_with(Some(sequence), prepare)
    }

    /// Prepare a replacement plan with the next latest-accepted sequence.
    pub fn prepare_next_with<E>(
        &self,
        prepare: impl FnOnce(&P) -> Result<P, E>,
    ) -> Result<*mut c_void, i32> {
        self.prepare_optional_with(None, prepare)
    }

    fn prepare_optional_with<E>(
        &self,
        sequence: Option<u64>,
        prepare: impl FnOnce(&P) -> Result<P, E>,
    ) -> Result<*mut c_void, i32> {
        if sequence.is_some_and(|sequence| sequence > SEQUENCE_MAX) {
            return Err(abi::EOVERFLOW);
        }
        let sequence = match sequence {
            Some(sequence) => sequence,
            None => {
                let Some(sequence) = self
                    .requested_sequence
                    .load(Ordering::Acquire)
                    .checked_add(1)
                    .filter(|sequence| *sequence <= SEQUENCE_MAX)
                else {
                    return Err(abi::EOVERFLOW);
                };
                sequence
            }
        };
        self.publication.prepare(sequence, prepare)
    }

    /// Publish one prepared plan for adoption by a later process call.
    ///
    /// # Safety
    ///
    /// `prepared` must be a token returned by [`Self::prepare_with`] for this
    /// runtime and must be consumed exactly once.
    pub unsafe fn commit(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance and exactly-once consumption are
        // this method's caller contract.
        unsafe {
            self.publication.commit(prepared, |sequence| {
                self.revision.fetch_add(1, Ordering::AcqRel);
                self.requested_sequence.store(*sequence, Ordering::Release);
                self.revision.fetch_add(1, Ordering::Release);
            });
        }
    }

    /// Discard one unpublished prepared plan on the serial parameter path.
    ///
    /// # Safety
    ///
    /// `prepared` must be a token returned by [`Self::prepare_with`] for this
    /// runtime and must be consumed exactly once.
    pub unsafe fn discard(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance and exactly-once consumption are
        // this method's caller contract.
        unsafe { self.publication.discard(prepared) }
    }

    fn adopt_requested(&self) -> bool {
        self.publication.adopt(
            || {
                self.revision.fetch_add(1, Ordering::AcqRel);
            },
            |sequence| {
                self.active_sequence.store(*sequence, Ordering::Release);
                self.revision.fetch_add(1, Ordering::Release);
            },
        )
    }
}

impl<P> PlanStore for ParameterRuntime<P>
where
    P: Send + Sync,
{
    type Plan = P;

    fn from_plan(plan: Self::Plan) -> Result<Self, i32> {
        Ok(Self::new(plan))
    }

    fn active_plan(&self) -> &Self::Plan {
        self.active_plan_ref()
    }

    fn adopt(&self) -> bool {
        self.adopt_requested()
    }

    fn prepare_parameter_with<E>(
        &self,
        sequence: Option<u64>,
        prepare: impl FnOnce(&Self::Plan) -> Result<Self::Plan, E>,
    ) -> Result<*mut c_void, i32> {
        match sequence {
            Some(sequence) => self.prepare_with(sequence, prepare),
            None => self.prepare_next_with(prepare),
        }
    }

    unsafe fn commit_parameter(&self, prepared: *mut c_void) {
        unsafe { self.commit(prepared) }
    }

    unsafe fn discard_parameter(&self, prepared: *mut c_void) {
        unsafe { self.discard(prepared) }
    }
}

impl<P> FgnProperties for ParameterRuntime<P>
where
    P: Send + Sync,
{
    fn property_info(&self, index: usize) -> Option<abi::PropertyInfo> {
        parameter_sequence_info(index)
    }

    fn property(&self, id: u32) -> Result<abi::Value, i32> {
        match id {
            REQUESTED_PARAMETER_SEQUENCE_ID => {
                Ok(long_value(self.requested_sequence.load(Ordering::Acquire)))
            }
            ACTIVE_PARAMETER_SEQUENCE_ID => {
                Ok(long_value(self.active_sequence.load(Ordering::Acquire)))
            }
            _ => Err(-abi::ENOENT),
        }
    }

    fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    unsafe fn prepare(
        &self,
        _properties: *const abi::Property,
        _n_properties: u32,
        _result: *mut *mut c_void,
    ) -> i32 {
        -abi::ENOENT
    }

    unsafe fn commit(&self, _prepared: *mut c_void) {}

    unsafe fn discard(&self, _prepared: *mut c_void) {}
}

pub(crate) fn parameter_sequence_info(index: usize) -> Option<abi::PropertyInfo> {
    let (id, name, description) = match index {
        0 => (
            REQUESTED_PARAMETER_SEQUENCE_ID,
            c"requested-parameter-sequence",
            c"Sequence of the latest accepted ndarray parameter",
        ),
        1 => (
            ACTIVE_PARAMETER_SEQUENCE_ID,
            c"active-parameter-sequence",
            c"Parameter sequence used by numerical processing",
        ),
        _ => return None,
    };
    Some(abi::PropertyInfo {
        struct_size: size_of::<abi::PropertyInfo>() as u32,
        id,
        flags: abi::PROPERTY_READONLY,
        reserved: 0,
        name: name.as_ptr(),
        description: description.as_ptr(),
        unit: c"1".as_ptr(),
        default_value: long_value(0),
        minimum: abi::Value::NONE,
        maximum: abi::Value::NONE,
        n_choices: 0,
        choices_reserved: 0,
        choices: ptr::null(),
    })
}

pub(crate) fn long_value(value: u64) -> abi::Value {
    abi::Value {
        type_: abi::TYPE_LONG,
        reserved: 0,
        value: abi::ValueBody {
            long_integer: value as i64,
        },
    }
}
