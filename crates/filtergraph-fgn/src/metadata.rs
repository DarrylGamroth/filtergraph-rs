//! Native 4.x SPA record validation and transport-neutral scientific values.
//!
//! Names and layouts follow spa/buffer/meta.h. Descriptor storage belongs to the
//! host; this module only reads descriptors and updates owned payload bytes.
use crate::abi;
use filtergraph_algorithms::{
    ChoppedImageKind, SampleAcquisitionIdentity, SampleClock, SampleExposureStart, SampleMetadata,
    SampleReferenceTimestamp, SampleTriggerState,
};

pub(crate) fn record_size(type_: u32) -> Option<u32> {
    Some(match type_ {
        abi::META_HEADER => 32,
        abi::META_ACQUISITION_IDENTITY => 40,
        abi::META_EXPOSURE_TIMING | abi::META_REFERENCE_TIMESTAMP => 24,
        abi::META_OPTICAL_CHOPPING | abi::META_ELECTRICAL_TRIGGER => 8,
        _ => return None,
    })
}

pub(crate) fn transport_owned(type_: u32) -> bool {
    matches!(type_, 7 | 9) || type_ >= 0x10000
}

pub(crate) fn validate_layout(meta: &abi::SpaMeta) -> Result<(), i32> {
    let retired = |type_| matches!(type_, 11..=13 | 0x302);
    if meta.type_ == 0
        || retired(meta.type_)
        || meta.size as usize > abi::MAX_META_BYTES
        || (meta.type_ >= 0x10000 && (meta.size != 0 || retired(meta.type_ >> 16)))
        || record_size(meta.type_).is_some_and(|size| size != meta.size)
        || (meta.size != 0 && (meta.data.is_null() || !meta.data.addr().is_multiple_of(8)))
    {
        return Err(abi::EINVAL);
    }
    Ok(())
}

unsafe fn zero(meta: &abi::SpaMeta) -> bool {
    meta.size == 0
        || unsafe { core::slice::from_raw_parts(meta.data.cast::<u8>(), meta.size as usize) }
            .iter()
            .all(|byte| *byte == 0)
}

/// Call after record validation. Header and opaque records have no absent
/// representation.
pub(crate) unsafe fn is_absent_native_record(meta: &abi::SpaMeta) -> bool {
    matches!(
        meta.type_,
        abi::META_ACQUISITION_IDENTITY
            | abi::META_EXPOSURE_TIMING
            | abi::META_REFERENCE_TIMESTAMP
            | abi::META_OPTICAL_CHOPPING
            | abi::META_ELECTRICAL_TRIGGER
    ) && unsafe { *meta.data.cast::<u32>() == 0 }
}

/// Caller has admitted accessible descriptor storage and nonoverlapping payloads.
pub(crate) unsafe fn validate_record(meta: &abi::SpaMeta, required: bool) -> Result<(), i32> {
    validate_layout(meta)?;
    let valid = match meta.type_ {
        abi::META_ACQUISITION_IDENTITY => {
            let m = unsafe { &*meta.data.cast::<abi::SpaMetaAcquisitionIdentity>() };
            if m.flags == 0 {
                !required && unsafe { zero(meta) }
            } else {
                m.flags == 1 && m.reserved == 0 && m.domain != [0; 16]
            }
        }
        abi::META_EXPOSURE_TIMING => {
            let m = unsafe { &*meta.data.cast::<abi::SpaMetaExposureTiming>() };
            if m.flags == 0 {
                !required && unsafe { zero(meta) }
            } else if m.flags & !3 != 0 {
                false
            } else if m.flags & 1 == 0 {
                m.flags == 2 && m.duration_nsec > 0 && m.clock == 0 && m.start_nsec == 0
            } else {
                (1..=4).contains(&m.clock)
                    && if m.flags & 2 != 0 {
                        m.duration_nsec > 0
                    } else {
                        m.duration_nsec == 0
                    }
            }
        }
        abi::META_REFERENCE_TIMESTAMP => {
            let m = unsafe { &*meta.data.cast::<abi::SpaMetaReferenceTimestamp>() };
            if m.clock == 0 {
                !required && unsafe { zero(meta) }
            } else {
                (1..=4).contains(&m.clock) && m.reserved == 0 && m.timestamp_nsec != u64::MAX
            }
        }
        abi::META_OPTICAL_CHOPPING | abi::META_ELECTRICAL_TRIGGER => {
            let m = unsafe { &*meta.data.cast::<abi::SpaMetaOpticalChopping>() };
            if m.flags == 1 {
                m.classification <= 3
            } else {
                !required && m.flags == 0 && m.classification == 0
            }
        }
        _ => true,
    };
    if valid { Ok(()) } else { Err(abi::EINVAL) }
}

pub(crate) unsafe fn validate_records(metas: *const abi::SpaMeta, count: usize) -> Result<(), i32> {
    for i in 0..count {
        unsafe { validate_record(&*metas.add(i), false) }?;
    }
    Ok(())
}

pub(crate) unsafe fn clear_records(metas: *const abi::SpaMeta, count: usize) {
    for i in 0..count {
        let meta = unsafe { &*metas.add(i) };
        if !transport_owned(meta.type_) && meta.size != 0 {
            unsafe { core::ptr::write_bytes(meta.data.cast::<u8>(), 0, meta.size as usize) };
        }
    }
}

fn clock(value: u32) -> Result<SampleClock, i32> {
    match value {
        1 => Ok(SampleClock::Monotonic),
        2 => Ok(SampleClock::Realtime),
        3 => Ok(SampleClock::Tai),
        4 => Ok(SampleClock::Model),
        _ => Err(abi::EINVAL),
    }
}

pub(crate) unsafe fn read_sample(
    metas: *const abi::SpaMeta,
    count: usize,
) -> Result<Option<SampleMetadata>, i32> {
    unsafe { validate_records(metas, count) }?;
    let records = if count == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(metas, count) }
    };
    let Some(header) = records.iter().find(|m| m.type_ == abi::META_HEADER) else {
        return Ok(None);
    };
    let h = unsafe { &*header.data.cast::<abi::SpaMetaHeader>() };
    let mut sample = SampleMetadata::new(h.seq, h.offset)
        .with_timestamp_nanoseconds((h.pts != abi::TIME_INVALID).then_some(h.pts))
        .with_terminal(h.flags & abi::META_HEADER_MARKER != 0)
        .with_discontinuity(h.flags & abi::META_HEADER_DISCONT != 0)
        .with_corrupted(h.flags & abi::META_HEADER_CORRUPTED != 0);
    for meta in records {
        match meta.type_ {
            abi::META_ACQUISITION_IDENTITY => {
                let m = unsafe { &*meta.data.cast::<abi::SpaMetaAcquisitionIdentity>() };
                if m.flags != 0 {
                    sample = sample.with_acquisition_identity(Some(
                        SampleAcquisitionIdentity::new(m.domain, m.generation, m.sequence)
                            .map_err(|_| abi::EINVAL)?,
                    ));
                }
            }
            abi::META_EXPOSURE_TIMING => {
                let m = unsafe { &*meta.data.cast::<abi::SpaMetaExposureTiming>() };
                if m.flags & 1 != 0 {
                    sample = sample.with_exposure_start(Some(SampleExposureStart::new(
                        m.start_nsec,
                        clock(m.clock)?,
                    )));
                }
                if m.flags & 2 != 0 {
                    sample = sample.with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(
                        m.duration_nsec,
                    ));
                }
            }
            abi::META_REFERENCE_TIMESTAMP => {
                let m = unsafe { &*meta.data.cast::<abi::SpaMetaReferenceTimestamp>() };
                if m.clock != 0 {
                    sample = sample.with_reference_timestamp(Some(
                        SampleReferenceTimestamp::new(
                            m.timestamp_nsec,
                            (m.duration_nsec != u64::MAX).then_some(m.duration_nsec),
                            clock(m.clock)?,
                        )
                        .map_err(|_| abi::EINVAL)?,
                    ));
                }
            }
            abi::META_OPTICAL_CHOPPING => {
                let m = unsafe { &*meta.data.cast::<abi::SpaMetaOpticalChopping>() };
                if m.flags != 0 {
                    sample = sample.with_chopped_image_kind(Some(match m.classification {
                        0 => ChoppedImageKind::Absent,
                        1 => ChoppedImageKind::Unfringed,
                        2 => ChoppedImageKind::Fringed,
                        _ => ChoppedImageKind::Mixed,
                    }));
                }
            }
            abi::META_ELECTRICAL_TRIGGER => {
                let m = unsafe { &*meta.data.cast::<abi::SpaMetaElectricalTrigger>() };
                if m.flags != 0 {
                    sample = sample.with_trigger_state(Some(match m.classification {
                        0 => SampleTriggerState::Absent,
                        1 => SampleTriggerState::Low,
                        2 => SampleTriggerState::High,
                        _ => SampleTriggerState::Mixed,
                    }));
                }
            }
            _ => {}
        }
    }
    Ok(Some(sample))
}

/// The descriptor must have negotiated storage for every published fact.
pub(crate) unsafe fn validate_publication(
    metas: *const abi::SpaMeta,
    count: usize,
    sample: SampleMetadata,
) -> Result<(), i32> {
    for (present, type_) in [
        (
            sample.acquisition_identity().is_some(),
            abi::META_ACQUISITION_IDENTITY,
        ),
        (
            sample.exposure_start().is_some() || sample.exposure_duration_nanoseconds().is_some(),
            abi::META_EXPOSURE_TIMING,
        ),
        (
            sample.reference_timestamp().is_some(),
            abi::META_REFERENCE_TIMESTAMP,
        ),
        (
            sample.chopped_image_kind().is_some(),
            abi::META_OPTICAL_CHOPPING,
        ),
        (
            sample.trigger_state().is_some(),
            abi::META_ELECTRICAL_TRIGGER,
        ),
    ] {
        if present && !(0..count).any(|i| unsafe { (*metas.add(i)).type_ == type_ }) {
            return Err(abi::ENOENT);
        }
    }
    Ok(())
}

/// Replace interpreted facts in existing owned slots, leaving opaque facts intact.
pub(crate) unsafe fn write_sample(
    metas: *const abi::SpaMeta,
    count: usize,
    sample: SampleMetadata,
) {
    for i in 0..count {
        let meta = unsafe { &*metas.add(i) };
        match meta.type_ {
            abi::META_ACQUISITION_IDENTITY => {
                let value = sample.acquisition_identity().map_or_else(
                    abi::SpaMetaAcquisitionIdentity::default,
                    |m| abi::SpaMetaAcquisitionIdentity {
                        flags: 1,
                        reserved: 0,
                        domain: m.domain(),
                        generation: m.generation(),
                        sequence: m.sequence(),
                    },
                );
                unsafe { *meta.data.cast::<abi::SpaMetaAcquisitionIdentity>() = value };
            }
            abi::META_EXPOSURE_TIMING => {
                let mut value = abi::SpaMetaExposureTiming::default();
                if let Some(m) = sample.exposure_start() {
                    value.flags |= 1;
                    value.clock = m.clock() as u32;
                    value.start_nsec = m.nanoseconds();
                }
                if let Some(duration) = sample.exposure_duration_nanoseconds() {
                    value.flags |= 2;
                    value.duration_nsec = duration;
                }
                unsafe { *meta.data.cast::<abi::SpaMetaExposureTiming>() = value };
            }
            abi::META_REFERENCE_TIMESTAMP => {
                let value = sample.reference_timestamp().map_or_else(
                    abi::SpaMetaReferenceTimestamp::default,
                    |m| abi::SpaMetaReferenceTimestamp {
                        clock: m.clock() as u32,
                        reserved: 0,
                        timestamp_nsec: m.nanoseconds(),
                        duration_nsec: m.duration_nanoseconds().unwrap_or(u64::MAX),
                    },
                );
                unsafe { *meta.data.cast::<abi::SpaMetaReferenceTimestamp>() = value };
            }
            abi::META_OPTICAL_CHOPPING => {
                let value = sample.chopped_image_kind().map_or_else(
                    abi::SpaMetaOpticalChopping::default,
                    |m| abi::SpaMetaOpticalChopping {
                        flags: 1,
                        classification: m as u32,
                    },
                );
                unsafe { *meta.data.cast::<abi::SpaMetaOpticalChopping>() = value };
            }
            abi::META_ELECTRICAL_TRIGGER => {
                let value = sample.trigger_state().map_or_else(
                    abi::SpaMetaElectricalTrigger::default,
                    |m| abi::SpaMetaElectricalTrigger {
                        flags: 1,
                        classification: m as u32,
                    },
                );
                unsafe { *meta.data.cast::<abi::SpaMetaElectricalTrigger>() = value };
            }
            _ => {}
        }
    }
}

// Transport-neutral requirement names are owned by AlgorithmPort / Julia's
// required_sample_metadata; only this boundary translates them to SPA records.
pub(crate) fn required_fact(fact: &str) -> Result<(u8, u32), i32> {
    Ok(match fact {
        "acquisition_identity" => (1, abi::META_ACQUISITION_IDENTITY),
        "exposure_duration_nanoseconds" => (2, abi::META_EXPOSURE_TIMING),
        "exposure_start" => (4, abi::META_EXPOSURE_TIMING),
        "reference_timestamp" => (8, abi::META_REFERENCE_TIMESTAMP),
        "chopped_image_kind" => (16, abi::META_OPTICAL_CHOPPING),
        "trigger_state" => (32, abi::META_ELECTRICAL_TRIGGER),
        "timestamp_nanoseconds" => (64, abi::META_HEADER),
        _ => return Err(abi::EINVAL),
    })
}

pub(crate) fn validate_sample(sample: Option<SampleMetadata>, required: u8) -> Result<(), i32> {
    if required == 0 {
        return Ok(());
    }
    let m = sample.ok_or(abi::EINVAL)?;
    if (required & 1 != 0 && m.acquisition_identity().is_none())
        || (required & 2 != 0 && m.exposure_duration_nanoseconds().is_none())
        || (required & 4 != 0 && m.exposure_start().is_none())
        || (required & 8 != 0 && m.reference_timestamp().is_none())
        || (required & 16 != 0 && m.chopped_image_kind().is_none())
        || (required & 32 != 0 && m.trigger_state().is_none())
        || (required & 64 != 0 && m.timestamp_nanoseconds().is_none())
    {
        Err(abi::EINVAL)
    } else {
        Ok(())
    }
}

/// Inputs and published outputs must satisfy their immutable port contracts.
pub(crate) unsafe fn validate_requirements(
    metas: *const abi::SpaMeta,
    count: usize,
    requirements: *const abi::SpaMetaRequirement,
    n_requirements: usize,
) -> Result<(), i32> {
    for i in 0..n_requirements {
        let r = unsafe { &*requirements.add(i) };
        let found = (0..count)
            .map(|i| unsafe { &*metas.add(i) })
            .find(|m| m.type_ == r.type_);
        let Some(m) = found else {
            if r.flags & abi::META_REQUIREMENT_REQUIRED != 0 {
                return Err(abi::ENOENT);
            }
            continue;
        };
        if m.size != r.size {
            return Err(abi::EINVAL);
        }
        if r.flags & abi::META_REQUIREMENT_REQUIRED != 0 && record_size(m.type_).is_some() {
            unsafe { validate_record(m, true) }?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{
        mem::{align_of, offset_of, size_of},
        ptr,
    };

    fn meta<T>(type_: u32, value: &mut T) -> abi::SpaMeta {
        abi::SpaMeta {
            type_,
            size: size_of::<T>() as u32,
            data: ptr::from_mut(value).cast(),
        }
    }
    fn buffer(metas: &mut [abi::SpaMeta]) -> abi::SpaBuffer {
        abi::SpaBuffer {
            n_metas: metas.len() as u32,
            n_datas: 0,
            metas: metas.as_mut_ptr(),
            datas: ptr::null_mut(),
        }
    }

    fn header() -> abi::SpaMetaHeader {
        abi::SpaMetaHeader {
            flags: 0,
            offset: 4,
            pts: 500,
            dts_offset: 0,
            seq: 901,
        }
    }

    #[test]
    fn current_native_layouts_are_exact() {
        assert_eq!(size_of::<abi::Format>(), 40);
        assert_eq!(size_of::<abi::SpaMetaAcquisitionIdentity>(), 40);
        assert_eq!(offset_of!(abi::SpaMetaAcquisitionIdentity, generation), 24);
        assert_eq!(size_of::<abi::SpaMetaExposureTiming>(), 24);
        assert_eq!(size_of::<abi::SpaMetaReferenceTimestamp>(), 24);
        assert_eq!(offset_of!(abi::SpaMetaExposureTiming, start_nsec), 8);
        assert_eq!(
            offset_of!(abi::SpaMetaReferenceTimestamp, timestamp_nsec),
            8
        );
        assert_eq!(size_of::<abi::SpaMetaOpticalChopping>(), 8);
        assert_eq!(align_of::<abi::SpaMetaOpticalChopping>(), 8);
        assert_eq!(align_of::<abi::SpaMetaElectricalTrigger>(), 8);

        {
            assert_eq!(size_of::<abi::PortInfo>(), 40);
            assert_eq!(offset_of!(abi::PortInfo, n_metas), 24);
            assert_eq!(offset_of!(abi::PortInfo, propagation), 28);
            assert_eq!(offset_of!(abi::PortInfo, metas), 32);
        }
        assert_eq!(abi::MAX_METAS, 64);
    }

    #[test]
    fn retired_exact_size_alignment_and_feature_layouts_are_rejected() {
        let mut storage = [0_u64; 64];
        for type_ in [
            0,
            11,
            12,
            13,
            0x302,
            (11 << 16) | 1,
            (12 << 16),
            (13 << 16),
            (0x302 << 16) | 2,
        ] {
            assert!(
                validate_layout(&abi::SpaMeta {
                    type_,
                    size: 0,
                    data: ptr::null_mut()
                })
                .is_err()
            );
        }
        for (type_, size) in [
            (abi::META_HEADER, 32),
            (abi::META_ACQUISITION_IDENTITY, 40),
            (abi::META_EXPOSURE_TIMING, 24),
            (abi::META_REFERENCE_TIMESTAMP, 24),
            (abi::META_OPTICAL_CHOPPING, 8),
            (abi::META_ELECTRICAL_TRIGGER, 8),
        ] {
            let mut m = abi::SpaMeta {
                type_,
                size,
                data: storage.as_mut_ptr().cast(),
            };

            assert!(validate_layout(&m).is_ok());
            m.size += 8;
            assert!(validate_layout(&m).is_err());
            m.size = size - 1;
            assert!(validate_layout(&m).is_err());
            m.size = size;
            m.data = storage.as_mut_ptr().cast::<u8>().wrapping_add(1).cast();
            assert!(validate_layout(&m).is_err());
        }
        assert!(
            validate_layout(&abi::SpaMeta {
                type_: 0x777,
                size: 7,
                data: storage.as_mut_ptr().cast()
            })
            .is_ok()
        );
        assert!(
            validate_layout(&abi::SpaMeta {
                type_: 1 << 16,
                size: 0,
                data: ptr::null_mut()
            })
            .is_ok()
        );
    }

    #[test]

    fn optional_absence_and_required_facts_match_native_semantics() {
        let mut identity = abi::SpaMetaAcquisitionIdentity::default();
        let m = meta(abi::META_ACQUISITION_IDENTITY, &mut identity);
        unsafe {
            assert!(validate_record(&m, false).is_ok());
            assert!(validate_record(&m, true).is_err());
        }
        identity.sequence = 9;
        assert_eq!(identity.sequence, 9);
        unsafe {
            assert!(validate_record(&m, false).is_err());
        }
        identity.flags = 1;
        identity.domain = [1; 16];
        assert_eq!((identity.flags, identity.domain), (1, [1; 16]));
        unsafe {
            assert!(validate_record(&m, true).is_ok());
        }
        identity.reserved = 1;
        assert_eq!(identity.reserved, 1);
        unsafe {
            assert!(validate_record(&m, false).is_err());
        }

        let mut exposure = abi::SpaMetaExposureTiming::default();
        let m = meta(abi::META_EXPOSURE_TIMING, &mut exposure);
        for (flags, clock, start, duration, valid) in [
            (0, 0, 0, 0, true),
            (0, 0, 0, 1, false),
            (2, 0, 0, 11, true),
            (2, 1, 0, 11, false),
            (1, 4, -7, 0, true),
            (1, 0, 4, 0, false),
            (1, 1, 4, 1, false),
            (3, 2, -4, 1, true),
            (3, 3, 4, 0, false),
            (4, 1, 4, 1, false),
        ] {
            exposure = abi::SpaMetaExposureTiming {
                flags,
                clock,
                start_nsec: start,
                duration_nsec: duration,
            };
            assert_eq!(exposure.flags, flags);
            unsafe {
                assert_eq!(validate_record(&m, false).is_ok(), valid);
                assert_eq!(validate_record(&m, true).is_ok(), valid && flags != 0);
            }
        }
        let mut reference = abi::SpaMetaReferenceTimestamp {
            clock: 2,
            reserved: 0,
            timestamp_nsec: u64::MAX - 1,
            duration_nsec: u64::MAX,
        };
        let m = meta(abi::META_REFERENCE_TIMESTAMP, &mut reference);
        unsafe {
            assert!(validate_record(&m, true).is_ok());
        }
        reference.timestamp_nsec = u64::MAX;
        assert_eq!(reference.timestamp_nsec, u64::MAX);
        unsafe {
            assert!(validate_record(&m, false).is_err());
        }
        reference = abi::SpaMetaReferenceTimestamp::default();
        assert_eq!(reference.clock, 0);
        unsafe {
            assert!(validate_record(&m, false).is_ok());
            assert!(validate_record(&m, true).is_err());
        }
        for type_ in [abi::META_OPTICAL_CHOPPING, abi::META_ELECTRICAL_TRIGGER] {
            let mut state = abi::SpaMetaOpticalChopping::default();
            let m = meta(type_, &mut state);
            unsafe {
                assert!(validate_record(&m, false).is_ok());
                assert!(validate_record(&m, true).is_err());
            }
            state.flags = 1;
            for classification in 0..5 {
                state.classification = classification;
                unsafe {
                    assert_eq!(validate_record(&m, true).is_ok(), classification <= 3);
                }
            }
        }
    }

    #[test]

    fn fields_roundtrip_independently_without_sequence_or_clock_inference() {
        let mut h = header();
        let mut identity = abi::SpaMetaAcquisitionIdentity {
            flags: 1,
            reserved: 0,
            domain: [3; 16],
            generation: 7,
            sequence: 19,
        };
        let mut exposure = abi::SpaMetaExposureTiming {
            flags: 3,
            clock: 4,
            start_nsec: -100,
            duration_nsec: 17,
        };
        let mut reference = abi::SpaMetaReferenceTimestamp {
            clock: 3,
            reserved: 0,
            timestamp_nsec: 18_000_000_000_000_001_123,
            duration_nsec: u64::MAX,
        };
        let mut optical = abi::SpaMetaOpticalChopping {
            flags: 1,
            classification: 2,
        };
        let mut trigger = abi::SpaMetaElectricalTrigger {
            flags: 1,
            classification: 1,
        };
        let mut records = [
            meta(abi::META_REFERENCE_TIMESTAMP, &mut reference),
            meta(abi::META_ELECTRICAL_TRIGGER, &mut trigger),
            meta(abi::META_HEADER, &mut h),
            meta(abi::META_ACQUISITION_IDENTITY, &mut identity),
            meta(abi::META_EXPOSURE_TIMING, &mut exposure),
            meta(abi::META_OPTICAL_CHOPPING, &mut optical),
        ];
        let mut raw = buffer(&mut records);
        let sample = unsafe { crate::read_sample_metadata(&mut raw).unwrap().unwrap() };
        assert_eq!(sample.sequence(), 901);
        assert_eq!(sample.acquisition_identity().unwrap().sequence(), 19);
        assert_eq!(sample.timestamp_nanoseconds(), Some(500));
        assert_eq!(sample.exposure_start().unwrap().nanoseconds(), -100);
        assert_eq!(sample.exposure_start().unwrap().clock(), SampleClock::Model);
        assert_eq!(
            sample.reference_timestamp().unwrap().clock(),
            SampleClock::Tai
        );
        assert_eq!(
            sample.reference_timestamp().unwrap().nanoseconds(),
            18_000_000_000_000_001_123
        );
        assert_eq!(
            sample.reference_timestamp().unwrap().duration_nanoseconds(),
            None
        );
        assert_eq!(sample.chopped_image_kind(), Some(ChoppedImageKind::Fringed));
        assert_eq!(sample.trigger_state(), Some(SampleTriggerState::Low));
        unsafe {
            clear_records(records.as_ptr(), records.len());
            write_sample(records.as_ptr(), records.len(), sample);
        }
        assert_eq!(identity.sequence, 19);
        assert_eq!(exposure.clock, 4);
        assert_eq!(reference.timestamp_nsec, 18_000_000_000_000_001_123);
        let absent = sample
            .with_chopped_image_kind(None)
            .with_trigger_state(Some(SampleTriggerState::Absent))
            .with_reference_timestamp(None);
        unsafe {
            write_sample(records.as_ptr(), records.len(), absent);
        }
        assert_eq!((optical.flags, optical.classification), (0, 0));
        assert_eq!((trigger.flags, trigger.classification), (1, 0));
        assert_eq!(reference.clock, 0);
        assert!(validate_sample(Some(absent), 16).is_err());
        assert!(validate_sample(Some(absent), 32).is_ok());
    }

    #[test]
    fn copy_requires_exact_preallocated_slots_and_clears_only_owned_payloads() {
        let mut source_payload = [17_u64];
        let mut destination_payload = [99_u64];
        let mut stale = [88_u64];
        let mut busy = [123_u64];
        let mut source = [meta(0x777, &mut source_payload)];
        let mut destination = [
            meta(0x777, &mut destination_payload),
            meta(0x778, &mut stale),
            meta(7, &mut busy),
        ];
        let descriptors = destination
            .iter()
            .map(|m| (m.type_, m.size, m.data))
            .collect::<Vec<_>>();
        let src = buffer(&mut source);
        let mut dst = buffer(&mut destination);
        unsafe {
            crate::copy_matching_metadata(&src, &mut dst).unwrap();
        }
        assert_eq!(destination_payload, [17]);
        assert_eq!(stale, [0]);
        assert_eq!(busy, [123]);
        assert_eq!(
            destination
                .iter()
                .map(|m| (m.type_, m.size, m.data))
                .collect::<Vec<_>>(),
            descriptors
        );
        source[0].size = 7;
        destination_payload[0] = 42;
        stale[0] = 77;
        unsafe {
            assert_eq!(
                crate::copy_matching_metadata(&src, &mut dst),
                Err(abi::EINVAL)
            );
        }
        assert_eq!(destination_payload, [42]);
        assert_eq!(stale, [77]);
        source[0].size = 8;
        assert_eq!(source[0].size, 8);
        dst.n_metas = 0;
        unsafe {
            assert_eq!(
                crate::copy_matching_metadata(&src, &mut dst),
                Err(abi::ENOENT)
            );
        }
    }

    #[test]
    fn missing_output_storage_only_accepts_absent_native_metadata() {
        let mut identity = abi::SpaMetaAcquisitionIdentity {
            flags: 0,
            reserved: 0,
            domain: [0; 16],
            generation: 0,
            sequence: 0,
        };
        let mut source = [meta(abi::META_ACQUISITION_IDENTITY, &mut identity)];
        let src = buffer(&mut source);
        let mut dst = buffer(&mut []);
        unsafe {
            assert_eq!(crate::copy_matching_metadata(&src, &mut dst), Ok(()));
        }

        identity.flags = 1;
        identity.domain = [1; 16];
        source[0] = meta(abi::META_ACQUISITION_IDENTITY, &mut identity);
        let src = buffer(&mut source);
        unsafe {
            assert_eq!(
                crate::copy_matching_metadata(&src, &mut dst),
                Err(abi::ENOENT)
            );
        }
    }

    #[test]
    fn envelope_is_bounded_disjoint_and_counts_aligned_bytes() {
        let mut payload = [0_u64; 512];
        let mut records = (0..64)
            .map(|i| abi::SpaMeta {
                type_: 0x700 + i,
                size: 0,
                data: ptr::null_mut(),
            })
            .collect::<Vec<_>>();
        let mut raw = buffer(&mut records);
        unsafe {
            assert!(crate::metadata_view(&mut raw).is_ok());
        }
        raw.n_metas = 65;
        unsafe {
            assert!(crate::metadata_view(&mut raw).is_err());
        }
        raw.n_metas = 2;
        records[0].size = 4096;
        records[0].data = payload.as_mut_ptr().cast();
        records[1].size = 1;
        records[1].data = payload.as_mut_ptr().cast();
        unsafe {
            assert!(crate::metadata_view(&mut raw).is_err());
        }
        records[0].size = 8;
        records[1].size = 8;
        unsafe {
            assert!(crate::metadata_view(&mut raw).is_err());
        }
        records[1].data = records.as_mut_ptr().cast();
        unsafe {
            assert!(crate::metadata_view(&mut raw).is_err());
        }
    }
}
