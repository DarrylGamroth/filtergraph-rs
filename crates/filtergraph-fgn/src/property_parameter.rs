//! Unified bounded publication for plans with scalar properties and an ndarray parameter.

use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{AtomicU64, Ordering};

use filtergraph_algorithms::{AlgorithmProperties, property_values_equal};

use crate::parameter::{
    ACTIVE_PARAMETER_SEQUENCE_ID, REQUESTED_PARAMETER_SEQUENCE_ID, SEQUENCE_MAX, long_value,
    parameter_sequence_info,
};
use crate::properties::{
    ACTIVE_GENERATION_ID, AdaptedPlan, FGN_GENERATION_MAX, PropertyAdapterError, PropertySchema,
    REQUESTED_GENERATION_ID, assignment_from_abi,
};
use crate::{FgnProperties, PlanStore, abi, publication::Publication};

#[derive(Clone, Copy)]
enum UpdateKind {
    Property(u64),
    Parameter(u64),
}

/// One bounded plan owner for scalar-property and ndarray-parameter updates.
///
/// Algorithm implementations continue to implement only [`AlgorithmProperties`]
/// and [`filtergraph_algorithms::AlgorithmPlan`]. This owner supplies FGN tokens,
/// cross-kind backpressure, observations, and frame-boundary plan adoption.
pub struct PropertyParameterRuntime<P> {
    schema: PropertySchema,
    publication: Publication<AdaptedPlan<P>, UpdateKind>,
    requested_generation: AtomicU64,
    active_generation: AtomicU64,
    requested_sequence: AtomicU64,
    active_sequence: AtomicU64,
    revision: AtomicU64,
}

impl<P> PropertyParameterRuntime<P>
where
    P: AlgorithmProperties + Send + Sync,
{
    /// Admit a property-enabled plan with an ndarray parameter surface.
    pub fn new(plan: P) -> Result<Self, PropertyAdapterError> {
        let schema = PropertySchema::new(&plan)?;
        let active = AdaptedPlan::new(plan, &schema)?;
        Ok(Self {
            schema,
            publication: Publication::new(active),
            requested_generation: AtomicU64::new(0),
            active_generation: AtomicU64::new(0),
            requested_sequence: AtomicU64::new(0),
            active_sequence: AtomicU64::new(0),
            revision: AtomicU64::new(0),
        })
    }

    /// Return the admitted scalar-property schema.
    pub const fn schema(&self) -> &PropertySchema {
        &self.schema
    }

    /// Return the immutable plan selected for the current process call.
    pub fn active_plan(&self) -> &P {
        self.publication.active().plan()
    }

    /// Return one current algorithm or adapter-owned property value.
    pub fn get(&self, id: u32) -> Result<abi::Value, i32> {
        match id {
            REQUESTED_GENERATION_ID => Ok(long_value(
                self.requested_generation.load(Ordering::Acquire),
            )),
            ACTIVE_GENERATION_ID => Ok(long_value(self.active_generation.load(Ordering::Acquire))),
            REQUESTED_PARAMETER_SEQUENCE_ID => {
                Ok(long_value(self.requested_sequence.load(Ordering::Acquire)))
            }
            ACTIVE_PARAMETER_SEQUENCE_ID => {
                Ok(long_value(self.active_sequence.load(Ordering::Acquire)))
            }
            _ => self.publication.active().value(id).ok_or(-abi::ENOENT),
        }
    }

    /// Return the monotonic externally visible property revision.
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    /// Prepare one complete scalar-property transaction.
    ///
    /// # Safety
    ///
    /// `properties` and every string it references must remain valid for this
    /// call. `result` must be writable. Calls are serialized by the FGN host.
    pub unsafe fn prepare_properties(
        &self,
        properties: *const abi::Property,
        n_properties: u32,
        result: *mut *mut c_void,
    ) -> i32 {
        if result.is_null() {
            return -abi::EINVAL;
        }
        // SAFETY: result was checked above and is lent for this callback.
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
        match self.publication.prepare(
            UpdateKind::Property(0),
            |active| -> Result<AdaptedPlan<P>, ()> {
                let mut assignments = Vec::with_capacity(properties.len());
                for property in properties {
                    assignments.push(unsafe { assignment_from_abi(property) }.map_err(|_| ())?);
                }
                let replacement = active
                    .plan()
                    .prepare_properties(assignments.as_slice())
                    .map_err(|_| ())?;
                AdaptedPlan::new(replacement, &self.schema).map_err(|_| ())
            },
        ) {
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

    /// Publish one prepared property replacement for later adoption.
    ///
    /// # Safety
    ///
    /// `prepared` must be a token returned by [`Self::prepare_properties`] for
    /// this runtime and must be consumed exactly once.
    pub unsafe fn commit_properties(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance and exactly-once consumption are
        // this method's caller contract.
        unsafe {
            self.publication.commit(prepared, |kind| {
                let generation = self.requested_generation.load(Ordering::Relaxed) + 1;
                *kind = UpdateKind::Property(generation);
                self.revision.fetch_add(1, Ordering::AcqRel);
                self.requested_generation
                    .store(generation, Ordering::Release);
                self.revision.fetch_add(1, Ordering::Release);
            });
        }
    }

    /// Discard one unpublished property replacement.
    ///
    /// # Safety
    ///
    /// `prepared` must be a token returned by [`Self::prepare_properties`] for
    /// this runtime and must be consumed exactly once.
    pub unsafe fn discard_properties(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance and exactly-once consumption are
        // this method's caller contract.
        unsafe { self.publication.discard(prepared) }
    }

    /// Prepare a complete replacement plan from one ndarray parameter.
    pub fn prepare_parameter_with<E>(
        &self,
        sequence: Option<u64>,
        prepare: impl FnOnce(&P) -> Result<P, E>,
    ) -> Result<*mut c_void, i32> {
        if sequence.is_some_and(|sequence| sequence > SEQUENCE_MAX) {
            return Err(abi::EOVERFLOW);
        }
        let sequence = match sequence {
            Some(sequence) => sequence,
            None => self
                .requested_sequence
                .load(Ordering::Acquire)
                .checked_add(1)
                .filter(|sequence| *sequence <= SEQUENCE_MAX)
                .ok_or(abi::EOVERFLOW)?,
        };
        self.publication.prepare(
            UpdateKind::Parameter(sequence),
            |active| -> Result<AdaptedPlan<P>, ()> {
                let replacement = prepare(active.plan()).map_err(|_| ())?;
                for info in self.schema.algorithm_infos() {
                    let previous = active.plan().property(info.id).map_err(|_| ())?;
                    let replacement_value = replacement.property(info.id).map_err(|_| ())?;
                    if !property_values_equal(previous, replacement_value) {
                        return Err(());
                    }
                }
                AdaptedPlan::new(replacement, &self.schema).map_err(|_| ())
            },
        )
    }

    /// Publish one prepared ndarray-parameter replacement for later adoption.
    ///
    /// # Safety
    ///
    /// `prepared` must be a token returned by [`Self::prepare_parameter_with`]
    /// for this runtime and must be consumed exactly once.
    pub unsafe fn commit_parameter(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance and exactly-once consumption are
        // this method's caller contract.
        unsafe {
            self.publication.commit(prepared, |kind| {
                let UpdateKind::Parameter(sequence) = *kind else {
                    return;
                };
                self.revision.fetch_add(1, Ordering::AcqRel);
                self.requested_sequence.store(sequence, Ordering::Release);
                self.revision.fetch_add(1, Ordering::Release);
            });
        }
    }

    /// Discard one unpublished ndarray-parameter replacement.
    ///
    /// # Safety
    ///
    /// `prepared` must be a token returned by [`Self::prepare_parameter_with`]
    /// for this runtime and must be consumed exactly once.
    pub unsafe fn discard_parameter(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance and exactly-once consumption are
        // this method's caller contract.
        unsafe { self.publication.discard(prepared) }
    }

    fn adopt_requested(&self) -> bool {
        self.publication.adopt(
            || {
                self.revision.fetch_add(1, Ordering::AcqRel);
            },
            |kind| {
                match *kind {
                    UpdateKind::Property(generation) => {
                        self.active_generation.store(generation, Ordering::Release)
                    }
                    UpdateKind::Parameter(sequence) => {
                        self.active_sequence.store(sequence, Ordering::Release)
                    }
                }
                self.revision.fetch_add(1, Ordering::Release);
            },
        )
    }
}

impl<P> FgnProperties for PropertyParameterRuntime<P>
where
    P: AlgorithmProperties + Send + Sync,
{
    fn property_info(&self, index: usize) -> Option<abi::PropertyInfo> {
        self.schema
            .info(index)
            .or_else(|| parameter_sequence_info(index.checked_sub(self.schema.len())?))
    }

    fn property(&self, id: u32) -> Result<abi::Value, i32> {
        self.get(id)
    }

    fn revision(&self) -> u64 {
        self.revision()
    }

    unsafe fn prepare(
        &self,
        properties: *const abi::Property,
        n_properties: u32,
        result: *mut *mut c_void,
    ) -> i32 {
        // SAFETY: forwarded ABI borrows are this callback's caller contract.
        unsafe { self.prepare_properties(properties, n_properties, result) }
    }

    unsafe fn commit(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance is this callback's caller contract.
        unsafe { self.commit_properties(prepared) }
    }

    unsafe fn discard(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance is this callback's caller contract.
        unsafe { self.discard_properties(prepared) }
    }
}

impl<P> PlanStore for PropertyParameterRuntime<P>
where
    P: AlgorithmProperties + Send + Sync,
{
    type Plan = P;

    fn from_plan(plan: Self::Plan) -> Result<Self, i32> {
        Self::new(plan).map_err(|_| abi::EINVAL)
    }

    fn active_plan(&self) -> &Self::Plan {
        PropertyParameterRuntime::active_plan(self)
    }

    fn adopt(&self) -> bool {
        self.adopt_requested()
    }

    fn prepare_parameter_with<E>(
        &self,
        sequence: Option<u64>,
        prepare: impl FnOnce(&Self::Plan) -> Result<Self::Plan, E>,
    ) -> Result<*mut c_void, i32> {
        PropertyParameterRuntime::prepare_parameter_with(self, sequence, prepare)
    }

    unsafe fn commit_parameter(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance is this callback's caller contract.
        unsafe { PropertyParameterRuntime::commit_parameter(self, prepared) }
    }

    unsafe fn discard_parameter(&self, prepared: *mut c_void) {
        // SAFETY: forwarded token provenance is this callback's caller contract.
        unsafe { PropertyParameterRuntime::discard_parameter(self, prepared) }
    }
}
