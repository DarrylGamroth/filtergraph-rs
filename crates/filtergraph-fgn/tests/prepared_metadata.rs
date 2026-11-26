//! Configuration-dependent requirements apply at native admission and before science.
use core::ffi::{CStr, c_void};
use filtergraph_algorithms::{
    AlgorithmConfiguration, AlgorithmDeclaration, AlgorithmInputs, AlgorithmOutputs, AlgorithmPort,
    AlgorithmProcessResult, ChoppedImageDifferenceF32 as Chopped, PreparedAlgorithm,
};
use filtergraph_fgn::{AlgorithmDescriptor, abi};
use std::sync::atomic::{AtomicUsize, Ordering};

const STRICT: &CStr =
    c"{\"pixel_count\":2,\"trigger_high_is_fringed\":true,\"require_acquisition_identity\":true}";
const DEFAULT: &CStr = c"{\"pixel_count\":2,\"trigger_high_is_fringed\":true}";

// This test-only forwarding declaration counts entry to the real Chopped
// scientific call, so rejection by FGN preflight is distinguishable from a
// later rejection by the algorithm's own metadata guards.
struct CountedChopped;
static SCIENTIFIC_CALLS: AtomicUsize = AtomicUsize::new(0);
impl AlgorithmDeclaration for CountedChopped {
    type Configuration = <Chopped as AlgorithmDeclaration>::Configuration;
    type Plan = <Chopped as AlgorithmDeclaration>::Plan;
    type Workspace = <Chopped as AlgorithmDeclaration>::Workspace;
    type PreparationError = <Chopped as AlgorithmDeclaration>::PreparationError;
    type ProcessError = <Chopped as AlgorithmDeclaration>::ProcessError;
    type ScalarProperties = <Chopped as AlgorithmDeclaration>::ScalarProperties;
    type Parameters = <Chopped as AlgorithmDeclaration>::Parameters;
    type ParameterError = <Chopped as AlgorithmDeclaration>::ParameterError;
    const NAME: &'static str = Chopped::NAME;
    const CONFIGURATION: &'static [AlgorithmConfiguration] = Chopped::CONFIGURATION;
    const PORTS: &'static [AlgorithmPort] = Chopped::PORTS;
    fn required_sample_metadata(plan: &Self::Plan, port: usize) -> &'static [&'static str] {
        Chopped::required_sample_metadata(plan, port)
    }
    fn prepare(c: Self::Configuration) -> Result<PreparedAlgorithm<Self>, Self::PreparationError> {
        let (plan, workspace, formats) = Chopped::prepare(c)?.into_parts();
        Ok(PreparedAlgorithm::new(plan, workspace, formats))
    }
    fn process(
        plan: &Self::Plan,
        workspace: &mut Self::Workspace,
        inputs: AlgorithmInputs<'_, '_>,
        outputs: AlgorithmOutputs<'_, '_>,
    ) -> Result<AlgorithmProcessResult, Self::ProcessError> {
        SCIENTIFIC_CALLS.fetch_add(1, Ordering::SeqCst);
        <Chopped as AlgorithmDeclaration>::process(plan, workspace, inputs, outputs)
    }
}

struct Instance<'a> {
    descriptor: &'a abi::Descriptor,
    data: *mut c_void,
    _executor: Box<abi::Executor>,
}
impl<'a> Instance<'a> {
    fn new(descriptor: &'a abi::Descriptor, config: &CStr) -> Result<Self, i32> {
        let executor = Box::new(abi::Executor {
            struct_size: size_of::<abi::Executor>() as u32,
            version: abi::EXECUTOR_VERSION,
            flags: 0,
            n_lanes: 1,
            data: core::ptr::null_mut(),
            run_dense_f32: None,
            run_lanes: None,
            cache_line_size: 64,
            reserved: 0,
        });
        let mut data = core::ptr::null_mut();
        // SAFETY: descriptor tables, executor, config and result are live and aligned.
        let result = unsafe {
            descriptor.instantiate.unwrap()(descriptor, config.as_ptr(), &*executor, &mut data)
        };
        if result != 0 {
            assert!(data.is_null(), "failed admission published an instance");
            return Err(result);
        }
        assert!(!data.is_null());
        Ok(Self {
            descriptor,
            data,
            _executor: executor,
        })
    }
    fn formats(&self) -> [*const abi::Format; 5] {
        let mut formats = [core::ptr::null(); 5];
        for (port, format) in formats.iter_mut().enumerate() {
            // SAFETY: this instance and each writable format pointer remain live.
            assert_eq!(
                unsafe { self.descriptor.get_port_format.unwrap()(self.data, port as u32, format) },
                0
            );
        }
        formats
    }
}
impl Drop for Instance<'_> {
    fn drop(&mut self) {
        // SAFETY: the unique successfully admitted instance is cleaned up once.
        unsafe { self.descriptor.cleanup.unwrap()(self.data) };
    }
}

// Clone POD ABI tables into local owned storage, never mutating descriptor globals.
struct DescriptorCopy {
    descriptor: Box<abi::Descriptor>,
    _ports: Box<[abi::PortInfo]>,
    _metas: Vec<Box<[abi::SpaMetaRequirement]>>,
}
impl DescriptorCopy {
    fn input_layouts(
        source: &abi::Descriptor,
        change: impl FnOnce(&mut Vec<abi::SpaMetaRequirement>),
    ) -> Self {
        // SAFETY: AlgorithmDescriptor owns complete immutable POD ABI tables.
        let mut descriptor = Box::new(unsafe { core::ptr::read(source) });
        let mut ports: Box<[_]> = (0..source.n_ports as usize)
            .map(|i| unsafe { core::ptr::read(source.ports.add(i)) })
            .collect();
        let mut tables = Vec::new();
        let mut change = Some(change);
        for (index, port) in ports.iter_mut().enumerate() {
            let mut records = if port.n_metas == 0 {
                Vec::new()
            } else {
                // SAFETY: the original metadata array has the declared extent.
                unsafe { core::slice::from_raw_parts(port.metas, port.n_metas as usize) }.to_vec()
            };
            if index == 0 {
                change.take().unwrap()(&mut records);
            }
            let records = records.into_boxed_slice();
            port.n_metas = records.len() as u32;
            port.metas = records.as_ptr();
            tables.push(records);
        }
        descriptor.ports = ports.as_ptr();
        Self {
            descriptor,
            _ports: ports,
            _metas: tables,
        }
    }
}

#[test]
fn prepared_requirements_admit_layouts_before_publishing_an_instance() {
    let owned = AlgorithmDescriptor::new::<Chopped>().unwrap();
    let source = owned.as_abi();
    // SAFETY: source owns its immutable metadata table for this test's lifetime.
    let original = unsafe {
        core::slice::from_raw_parts((*source.ports).metas, (*source.ports).n_metas as usize)
    };
    let snapshot: Vec<_> = original
        .iter()
        .map(|r| (r.type_, r.size, r.flags, r.reserved))
        .collect();
    assert_eq!(snapshot.len(), 4);
    assert!(snapshot.iter().all(|r| r.2 == 0));
    Instance::new(source, STRICT).unwrap();
    let kind_only = DescriptorCopy::input_layouts(source, |records| {
        records.retain(|r| r.type_ == abi::META_OPTICAL_CHOPPING)
    });
    Instance::new(&kind_only.descriptor, DEFAULT).unwrap();
    for required in [
        abi::META_ACQUISITION_IDENTITY,
        abi::META_EXPOSURE_TIMING,
        abi::META_OPTICAL_CHOPPING,
    ] {
        let omitted = DescriptorCopy::input_layouts(source, |records| {
            records.retain(|r| r.type_ != required)
        });
        assert_eq!(
            Instance::new(&omitted.descriptor, STRICT).err(),
            Some(-abi::EINVAL)
        );
        let wrong_size = DescriptorCopy::input_layouts(source, |records| {
            records
                .iter_mut()
                .find(|r| r.type_ == required)
                .unwrap()
                .size += 8
        });
        assert_eq!(
            Instance::new(&wrong_size.descriptor, STRICT).err(),
            Some(-abi::EINVAL)
        );
    }
    assert_eq!(
        original
            .iter()
            .map(|r| (r.type_, r.size, r.flags, r.reserved))
            .collect::<Vec<_>>(),
        snapshot
    );
}

// All pointers target stable owned allocations, including eight-byte metadata
// payloads; moving Packet never relocates payload, descriptors, or chunks.
struct Packet<T> {
    payload: Box<[T]>,
    chunk: Box<abi::SpaChunk>,
    header: Box<abi::SpaMetaHeader>,
    identity: Box<abi::SpaMetaAcquisitionIdentity>,
    exposure: Box<abi::SpaMetaExposureTiming>,
    optical: Box<abi::SpaMetaOpticalChopping>,
    _trigger: Box<abi::SpaMetaElectricalTrigger>,
    _metas: Box<[abi::SpaMeta]>,
    _data: Box<abi::SpaData>,
    buffer: Box<abi::SpaBuffer>,
}
impl<T> Packet<T> {
    fn new(values: Vec<T>, extras: &[u32]) -> Self {
        let mut payload = values.into_boxed_slice();
        let bytes = size_of_val(&*payload) as u32;
        let mut chunk = Box::new(abi::SpaChunk {
            offset: 0,
            size: bytes,
            stride: size_of::<T>() as i32,
            flags: 0,
        });
        let mut header = Box::new(abi::SpaMetaHeader {
            flags: 0,
            offset: 0,
            pts: 0,
            dts_offset: 0,
            seq: 100,
        });
        let mut identity = Box::<abi::SpaMetaAcquisitionIdentity>::default();
        let mut exposure = Box::<abi::SpaMetaExposureTiming>::default();
        let mut optical = Box::<abi::SpaMetaOpticalChopping>::default();
        let mut trigger = Box::<abi::SpaMetaElectricalTrigger>::default();
        fn meta<U>(type_: u32, value: &mut U) -> abi::SpaMeta {
            abi::SpaMeta {
                type_,
                size: size_of::<U>() as u32,
                data: core::ptr::from_mut(value).cast(),
            }
        }
        let mut metas = vec![meta(abi::META_HEADER, &mut *header)];
        for &type_ in extras {
            metas.push(match type_ {
                abi::META_ACQUISITION_IDENTITY => meta(type_, &mut *identity),
                abi::META_EXPOSURE_TIMING => meta(type_, &mut *exposure),
                abi::META_OPTICAL_CHOPPING => meta(type_, &mut *optical),
                abi::META_ELECTRICAL_TRIGGER => meta(type_, &mut *trigger),
                _ => panic!("unsupported test record"),
            });
        }
        let mut metas = metas.into_boxed_slice();
        let mut data = Box::new(abi::SpaData {
            type_: 0,
            flags: 0,
            fd: -1,
            mapoffset: 0,
            maxsize: bytes,
            data: payload.as_mut_ptr().cast(),
            chunk: &mut *chunk,
        });
        let buffer = Box::new(abi::SpaBuffer {
            n_metas: metas.len() as u32,
            n_datas: 1,
            metas: metas.as_mut_ptr(),
            datas: &mut *data,
        });
        Self {
            payload,
            chunk,
            header,
            identity,
            exposure,
            optical,
            _trigger: trigger,
            _metas: metas,
            _data: data,
            buffer,
        }
    }
    fn abi(&mut self, format: *const abi::Format) -> abi::Buffer {
        abi::Buffer {
            buffer: &mut *self.buffer,
            format,
        }
    }
    fn acquisition(&mut self, physical_sequence: u64, publication_sequence: u64, kind: u32) {
        self.header.seq = publication_sequence;
        *self.identity = abi::SpaMetaAcquisitionIdentity {
            flags: 1,
            reserved: 0,
            domain: [1; 16],
            generation: 1,
            sequence: physical_sequence,
        };
        *self.exposure = abi::SpaMetaExposureTiming {
            flags: 2,
            clock: 0,
            start_nsec: 0,
            duration_nsec: 20,
        };
        *self.optical = abi::SpaMetaOpticalChopping {
            flags: 1,
            classification: kind,
        };
    }
}
const ALL: &[u32] = &[
    abi::META_OPTICAL_CHOPPING,
    abi::META_ELECTRICAL_TRIGGER,
    abi::META_ACQUISITION_IDENTITY,
    abi::META_EXPOSURE_TIMING,
];
struct Frames {
    image: Packet<f32>,
    valid: Packet<u8>,
    difference: Packet<f32>,
    output_valid: Packet<u8>,
    status: Packet<u8>,
}
impl Frames {
    fn new(extras: &[u32]) -> Self {
        Self {
            image: Packet::new(vec![10.0, 20.0], extras),
            valid: Packet::new(vec![1, 1], &[]),
            difference: Packet::new(vec![71.0, 71.0], ALL),
            output_valid: Packet::new(vec![1, 1], ALL),
            status: Packet::new(vec![99], ALL),
        }
    }
    fn sentinels(&mut self) {
        self.difference.payload.fill(71.0);
        self.output_valid.payload.fill(1);
        self.status.payload.fill(99);
    }
    fn process(&mut self, instance: &Instance<'_>) -> i32 {
        let formats = instance.formats();
        let inputs = [self.image.abi(formats[0]), self.valid.abi(formats[1])];
        let mut outputs = [
            self.difference.abi(formats[2]),
            self.output_valid.abi(formats[3]),
            self.status.abi(formats[4]),
        ];
        // SAFETY: all separate packet allocations and descriptor tables remain
        // live and exclusive during this synchronous callback.
        unsafe {
            instance.descriptor.process.unwrap()(
                instance.data,
                inputs.as_ptr(),
                2,
                outputs.as_mut_ptr(),
                3,
            )
        }
    }
    fn assert_unpublished(&self) {
        assert_eq!(&*self.difference.payload, &[71.0, 71.0]);
        assert_eq!(&*self.output_valid.payload, &[1, 1]);
        assert_eq!(&*self.status.payload, &[99]);
        assert_eq!(
            (
                self.difference.chunk.size,
                self.output_valid.chunk.size,
                self.status.chunk.size
            ),
            (0, 0, 0)
        );
    }
}

#[test]
fn prepared_facts_preflight_values_before_science_and_preserve_pair_history() {
    let owned = AlgorithmDescriptor::new::<CountedChopped>().unwrap();
    let instance = Instance::new(owned.as_abi(), STRICT).unwrap();
    let mut frames = Frames::new(ALL);
    *frames.image.optical = abi::SpaMetaOpticalChopping {
        flags: 1,
        classification: 2,
    };
    SCIENTIFIC_CALLS.store(0, Ordering::SeqCst);
    assert_eq!(frames.process(&instance), -abi::EINVAL);
    assert_eq!(
        SCIENTIFIC_CALLS.load(Ordering::SeqCst),
        0,
        "missing facts reached scientific processing"
    );
    frames.assert_unpublished();
    frames.image.acquisition(10, 100, 2);
    assert_eq!(frames.process(&instance), 0);
    assert_eq!(&*frames.status.payload, &[0]);
    assert_eq!(&*frames.difference.payload, &[0.0, 0.0]);
    frames.image.payload.copy_from_slice(&[3.0, 5.0]);
    frames.image.acquisition(11, 1, 1);
    assert_eq!(frames.process(&instance), 0);
    assert_eq!(&*frames.difference.payload, &[7.0, 15.0]);
    assert_eq!(&*frames.output_valid.payload, &[1, 1]);
    assert_eq!(&*frames.status.payload, &[1]);
    assert_eq!(
        frames.difference.header.seq, 1,
        "publication identity was replaced by physical identity"
    );
    for missing in [
        abi::META_ACQUISITION_IDENTITY,
        abi::META_EXPOSURE_TIMING,
        abi::META_OPTICAL_CHOPPING,
    ] {
        frames.image.payload.copy_from_slice(&[12.0, 25.0]);
        frames.image.acquisition(12, 900, 2);
        match missing {
            abi::META_ACQUISITION_IDENTITY => *frames.image.identity = Default::default(),
            abi::META_EXPOSURE_TIMING => *frames.image.exposure = Default::default(),
            _ => *frames.image.optical = Default::default(),
        }
        frames.sentinels();
        let before = SCIENTIFIC_CALLS.load(Ordering::SeqCst);
        assert_eq!(frames.process(&instance), -abi::EINVAL);
        assert_eq!(SCIENTIFIC_CALLS.load(Ordering::SeqCst), before);
        frames.assert_unpublished();
    }
    frames.image.acquisition(12, 0, 2);
    assert_eq!(frames.process(&instance), 0);
    assert_eq!(&*frames.difference.payload, &[9.0, 20.0]);
    assert_eq!(&*frames.status.payload, &[1]);

    let ordinary = AlgorithmDescriptor::new::<Chopped>().unwrap();
    let kind_only = DescriptorCopy::input_layouts(ordinary.as_abi(), |records| {
        records.retain(|r| r.type_ == abi::META_OPTICAL_CHOPPING)
    });
    let instance = Instance::new(&kind_only.descriptor, DEFAULT).unwrap();
    let mut frames = Frames::new(&[abi::META_OPTICAL_CHOPPING]);
    *frames.image.optical = abi::SpaMetaOpticalChopping {
        flags: 1,
        classification: 2,
    };
    assert_eq!(frames.process(&instance), 0);
    assert_eq!(&*frames.status.payload, &[0]);
    frames.image.header.seq = 1;
    frames.image.payload.copy_from_slice(&[3.0, 5.0]);
    frames.image.optical.classification = 1;
    assert_eq!(frames.process(&instance), 0);
    assert_eq!(&*frames.difference.payload, &[7.0, 15.0]);
}

#[test]
fn prepared_metadata_tables_are_stable_and_instance_specific() {
    let owned = AlgorithmDescriptor::new::<Chopped>().unwrap();
    let descriptor = owned.as_abi();
    let strict = Instance::new(descriptor, STRICT).unwrap();
    let normal = Instance::new(descriptor, DEFAULT).unwrap();
    let callback = descriptor.get_port_metadata.unwrap();
    let mut strict_ptr = core::ptr::null();
    let mut count = 0;
    // SAFETY: both owned instances and the descriptor outlive all returned tables.
    unsafe {
        assert_eq!(callback(strict.data, 0, &mut strict_ptr, &mut count), 0);
        let records = core::slice::from_raw_parts(strict_ptr, count as usize);
        for record in records {
            let expected = if record.type_ == abi::META_ELECTRICAL_TRIGGER {
                0
            } else {
                abi::META_REQUIREMENT_REQUIRED
            };
            assert_eq!(record.flags, expected);
        }
        let mut repeated = core::ptr::null();
        assert_eq!(callback(strict.data, 0, &mut repeated, &mut count), 0);
        assert_eq!(strict_ptr, repeated);
        let mut normal_ptr = core::ptr::null();
        assert_eq!(callback(normal.data, 0, &mut normal_ptr, &mut count), 0);
        assert!(
            core::slice::from_raw_parts(normal_ptr, count as usize)
                .iter()
                .all(|r| r.flags == 0)
        );
        assert_ne!(strict_ptr, normal_ptr);
        let port = &*descriptor.ports;
        assert!(
            core::slice::from_raw_parts(port.metas, port.n_metas as usize)
                .iter()
                .all(|r| r.flags == 0)
        );
        assert_eq!(
            callback(strict.data, u32::MAX, &mut normal_ptr, &mut count),
            -abi::EINVAL
        );
    }
}
