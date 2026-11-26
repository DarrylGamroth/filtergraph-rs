//! Host-visible behavior of the standalone FGN adapters.

use core::ffi::{CStr, c_void};
use core::ptr::{null, null_mut};

use filtergraph_algorithms::AlgorithmPlan;
use filtergraph_fgn::{
    ACTIVE_PARAMETER_SEQUENCE_ID, DeclaredPlugin, InputBuffers, NdarrayFormat, OutputBuffers,
    ParameterBuffer, ParameterRuntime, PluginDeclaration, PluginInstance, PreparedInstance,
    PreparedSourceF32, PreparedUnaryF32, REQUESTED_PARAMETER_SEQUENCE_ID, abi, descriptor,
};

const DATA_SCHEMA: &str = "test/samples/1";
const PARAMETER_SCHEMA: &str = "test/gain/1";
const DISCONNECTED: abi::Buffer = abi::Buffer {
    buffer: null_mut(),
    format: null(),
};

fn executor() -> abi::Executor {
    abi::Executor {
        struct_size: size_of::<abi::Executor>() as u32,
        version: abi::EXECUTOR_VERSION,
        flags: 0,
        n_lanes: 1,
        data: null_mut(),
        run_dense_f32: None,
        run_lanes: None,
        cache_line_size: 0,
        reserved: 0,
    }
}

struct Instance {
    descriptor: Box<abi::Descriptor>,
    _executor: Box<abi::Executor>,
    data: *mut c_void,
}

impl Instance {
    fn new<I: PluginInstance>(configuration: &CStr) -> Result<Self, i32> {
        Self::from_descriptor(descriptor::<I>(), configuration)
    }

    fn from_descriptor(descriptor: abi::Descriptor, configuration: &CStr) -> Result<Self, i32> {
        let descriptor = Box::new(descriptor);
        let executor = Box::new(executor());
        let mut data = null_mut();
        // Both immutable tables remain at stable addresses through cleanup.
        let status = unsafe {
            (descriptor.instantiate.unwrap())(
                &*descriptor,
                configuration.as_ptr(),
                &*executor,
                &mut data,
            )
        };
        if status != 0 {
            assert!(data.is_null());
            return Err(status);
        }
        assert!(!data.is_null());
        Ok(Self {
            descriptor,
            _executor: executor,
            data,
        })
    }

    fn format(&self, port: u32) -> Result<&abi::Format, i32> {
        let mut raw = null();
        let status =
            unsafe { (self.descriptor.get_port_format.unwrap())(self.data, port, &mut raw) };
        if status != 0 {
            assert!(raw.is_null());
            return Err(status);
        }
        Ok(unsafe { &*raw })
    }

    // Callers keep all nested buffers alive and exclusively lent for this call.
    unsafe fn process(&self, inputs: &[abi::Buffer], outputs: &mut [abi::Buffer]) -> i32 {
        unsafe {
            (self.descriptor.process.unwrap())(
                self.data,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
            )
        }
    }

    unsafe fn prepare(&self, port: u32, parameter: &abi::Buffer) -> Result<*mut c_void, i32> {
        let mut token = null_mut();
        let status = unsafe {
            (self.descriptor.prepare_parameter.unwrap())(self.data, port, parameter, &mut token)
        };
        if status == 0 {
            assert!(!token.is_null());
            Ok(token)
        } else {
            assert!(token.is_null());
            Err(status)
        }
    }

    fn sequence(&self, id: u32) -> i64 {
        let mut value = abi::Value::NONE;
        assert_eq!(
            unsafe { (self.descriptor.get_prop.unwrap())(self.data, id, &mut value) },
            0
        );
        assert_eq!(value.type_, abi::TYPE_LONG);
        unsafe { value.value.long_integer }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { (self.descriptor.cleanup.unwrap())(self.data) };
    }
}

fn with_buffer(
    values: &mut [f32],
    format: &abi::Format,
    header_sequence: Option<u64>,
    callback: impl FnOnce(abi::Buffer),
) -> (u32, u64) {
    with_typed_buffer(values, format, header_sequence, callback)
}

fn with_typed_buffer<T>(
    values: &mut [T],
    format: &abi::Format,
    header_sequence: Option<u64>,
    callback: impl FnOnce(abi::Buffer),
) -> (u32, u64) {
    let bytes = size_of_val(values) as u32;
    let mut chunk = abi::SpaChunk {
        offset: 0,
        size: bytes,
        stride: bytes as i32,
        flags: 0,
    };
    let mut data = abi::SpaData {
        type_: 0,
        flags: 0,
        fd: -1,
        mapoffset: 0,
        maxsize: bytes,
        data: values.as_mut_ptr().cast(),
        chunk: &mut chunk,
    };
    let mut header = abi::SpaMetaHeader {
        flags: 0,
        offset: 0,
        pts: 123,
        dts_offset: 0,
        seq: header_sequence.unwrap_or_default(),
    };
    let mut meta = abi::SpaMeta {
        type_: abi::META_HEADER,
        size: size_of::<abi::SpaMetaHeader>() as u32,
        data: (&raw mut header).cast(),
    };
    let mut buffer = abi::SpaBuffer {
        n_metas: u32::from(header_sequence.is_some()),
        n_datas: 1,
        metas: &mut meta,
        datas: &mut data,
    };
    callback(abi::Buffer {
        buffer: &mut buffer,
        format,
    });
    (chunk.size, header.seq)
}

struct Gain(f32);

impl AlgorithmPlan for Gain {
    type Input<'a> = &'a [f32];
    type Output<'a> = &'a mut [f32];
    type Workspace = u32;
    type Error = ();

    fn process(&self, output: &mut [f32], frames: &mut u32, input: &[f32]) -> Result<(), ()> {
        if input.iter().any(|sample| !sample.is_finite()) {
            return Err(());
        }
        for (result, sample) in output.iter_mut().zip(input) {
            *result = sample * self.0 + *frames as f32;
        }
        *frames += 1;
        Ok(())
    }
}

fn prepare_gain(extent: usize) -> Result<PreparedUnaryF32<Gain, u32>, ()> {
    if extent == 1 {
        return Err(());
    }
    Ok(PreparedUnaryF32::new(Gain(2.0), 0, extent).with_rate(Some((500, 1))))
}

fn replacement_gain(_: &Gain, values: &[f32]) -> Result<Gain, ()> {
    (values[0].is_finite() && values[1] == 0.0)
        .then_some(Gain(values[0]))
        .ok_or(())
}

fn reset_frames(frames: &mut u32) {
    *frames = 0;
}

filtergraph_fgn::declare_unary_f32_plugin! {
    FixedGain {
        label = "fixed-gain";
        schema = DATA_SCHEMA;
        config = usize;
        plan = Gain;
        properties = none;
        prepare = prepare_gain;
        reset = reset_frames;
    }
}

filtergraph_fgn::declare_unary_f32_parameter_plugin! {
    TunableGain {
        label = "tunable-gain";
        data_schema = DATA_SCHEMA;
        parameter_schema = PARAMETER_SCHEMA;
        config = usize;
        plan = Gain;
        prepare = prepare_gain;
        prepare_parameter = replacement_gain;
        reset = reset_frames;
    }
}

struct Ramp([f32; 2]);

impl AlgorithmPlan for Ramp {
    type Input<'a> = ();
    type Output<'a> = &'a mut [f32];
    type Workspace = u32;
    type Error = ();

    fn process(&self, output: &mut [f32], frames: &mut u32, (): ()) -> Result<(), ()> {
        if *frames == 2 {
            return Err(());
        }
        output.copy_from_slice(&self.0.map(|value| value + *frames as f32));
        *frames += 1;
        Ok(())
    }
}

fn prepare_ramp(extent: usize) -> Result<PreparedSourceF32<Ramp, u32>, ()> {
    if extent == 1 {
        return Err(());
    }
    Ok(PreparedSourceF32::new(Ramp([4.0, 5.0]), 0, extent))
}

fn replacement_ramp(_: &Ramp, values: &[f32]) -> Result<Ramp, ()> {
    let values: [f32; 2] = values.try_into().map_err(|_| ())?;
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(Ramp(values))
        .ok_or(())
}

fn reset_ramp(_: &Ramp, frames: &mut u32) {
    reset_frames(frames);
}

filtergraph_fgn::declare_source_f32_parameter_plugin! {
    TunableRamp {
        label = "tunable-ramp";
        output_schema = DATA_SCHEMA;
        parameter_schema = PARAMETER_SCHEMA;
        config = usize;
        plan = Ramp;
        prepare = prepare_ramp;
        prepare_parameter = replacement_ramp;
        reset = reset_ramp;
    }
}

fn rejected_configuration<I: PluginInstance>() {
    for (configuration, errno) in [
        (c"\xff", abi::EINVAL),
        (c"{}", abi::EINVAL),
        (c"1", abi::EINVAL),
        (c"0", abi::EINVAL),
        (c"4294967296", abi::EOVERFLOW),
    ] {
        assert_eq!(
            Instance::new::<I>(configuration).err(),
            Some(-errno),
            "{configuration:?}"
        );
    }
}

#[test]
fn fixed_unary_publishes_complete_frames_and_reset_restarts_history() {
    rejected_configuration::<FixedGain>();
    let instance = Instance::new::<FixedGain>(c"2").unwrap();
    let input_format = instance.format(0).unwrap();
    let output_format = instance.format(1).unwrap();
    assert_eq!((input_format.rate_num, input_format.rate_denom), (500, 1));
    assert_eq!(
        unsafe { CStr::from_ptr(output_format.schema) }
            .to_str()
            .unwrap(),
        DATA_SCHEMA
    );
    assert_eq!(instance.format(2).err(), Some(-abi::ENOENT));

    let mut result = [9.0; 2];
    for (samples, expected) in [([3.0, -2.0], [6.0, -4.0]), ([3.0, -2.0], [7.0, -3.0])] {
        let (size, seq) = with_buffer(&mut result, output_format, Some(0), |output| {
            with_buffer(&mut samples.clone(), input_format, Some(77), |input| {
                assert_eq!(unsafe { instance.process(&[input], &mut [output]) }, 0);
            });
        });
        assert_eq!((size, seq), (8, 77));
        assert_eq!(result, expected);
    }

    let (size, seq) = with_buffer(&mut result, output_format, Some(80), |output| {
        with_buffer(&mut [f32::NAN, 0.0], input_format, Some(81), |input| {
            assert_eq!(
                unsafe { instance.process(&[input], &mut [output]) },
                -abi::EINVAL
            );
        });
    });
    assert_eq!((size, seq), (0, 0));
    assert_eq!(result, [7.0, -3.0]);

    assert_eq!(
        unsafe { (instance.descriptor.reset.unwrap())(instance.data) },
        0
    );
    with_buffer(&mut result, output_format, None, |output| {
        with_buffer(&mut [3.0, -2.0], input_format, None, |input| {
            assert_eq!(unsafe { instance.process(&[input], &mut [output]) }, 0);
        });
    });
    assert_eq!(result, [6.0, -4.0]);

    let mut value = abi::Value::NONE;
    let mut token = null_mut();
    let mut info = core::mem::MaybeUninit::uninit();
    unsafe {
        let d = &instance.descriptor;
        assert_eq!(
            (d.enum_prop_info.unwrap())(instance.data, 0, info.as_mut_ptr()),
            0
        );
        assert_eq!(
            (d.get_prop.unwrap())(instance.data, 0, &mut value),
            -abi::ENOENT
        );
        assert_eq!((d.get_prop_revision.unwrap())(instance.data), 0);
        assert_eq!(
            (d.prepare_props.unwrap())(instance.data, null(), 0, &mut token),
            -abi::ENOENT
        );
        assert!(token.is_null());
    }
}

fn check_parameter_publication(instance: &Instance, port: u32, values: &mut [f32]) {
    let format = instance.format(port).unwrap();
    assert_eq!(
        unsafe { CStr::from_ptr(format.schema) }.to_str().unwrap(),
        PARAMETER_SCHEMA
    );
    with_buffer(values, format, Some(3), |parameter| unsafe {
        assert_eq!(instance.prepare(0, &parameter), Err(-abi::EINVAL));
        let discarded = instance.prepare(port, &parameter).unwrap();
        (instance.descriptor.discard_parameter.unwrap())(instance.data, discarded);
    });
    assert_eq!(instance.sequence(REQUESTED_PARAMETER_SEQUENCE_ID), 0);

    with_buffer(values, format, None, |parameter| unsafe {
        let prepared = instance.prepare(port, &parameter).unwrap();
        (instance.descriptor.commit_parameter.unwrap())(instance.data, prepared);
    });
    assert_eq!(instance.sequence(REQUESTED_PARAMETER_SEQUENCE_ID), 1);
    assert_eq!(instance.sequence(ACTIVE_PARAMETER_SEQUENCE_ID), 0);
}

#[test]
fn sparse_unary_adopts_a_parameter_only_for_an_admitted_frame() {
    rejected_configuration::<TunableGain>();
    let instance = Instance::new::<TunableGain>(c"2").unwrap();
    assert_eq!(instance.format(3).err(), Some(-abi::ENOENT));
    check_parameter_publication(&instance, 2, &mut [3.0, 0.0]);

    let mut result = [0.0; 2];
    let (size, _) = with_buffer(&mut result, instance.format(1).unwrap(), None, |output| {
        with_buffer(
            &mut [2.0, 5.0],
            instance.format(0).unwrap(),
            None,
            |input| {
                assert_eq!(
                    unsafe { instance.process(&[input, input], &mut [output]) },
                    -abi::EINVAL
                );
            },
        );
    });
    assert_eq!(size, 0);
    assert_eq!(instance.sequence(ACTIVE_PARAMETER_SEQUENCE_ID), 0);
    let (size, seq) = with_buffer(
        &mut result,
        instance.format(1).unwrap(),
        Some(0),
        |output| {
            with_buffer(
                &mut [2.0, 5.0],
                instance.format(0).unwrap(),
                Some(41),
                |input| {
                    assert_eq!(
                        unsafe { instance.process(&[input, DISCONNECTED], &mut [output]) },
                        0
                    );
                },
            );
        },
    );
    assert_eq!((size, seq), (8, 41));
    assert_eq!(result, [6.0, 15.0]);
    assert_eq!(instance.sequence(ACTIVE_PARAMETER_SEQUENCE_ID), 1);

    with_buffer(
        &mut [4.0, -1.0],
        instance.format(2).unwrap(),
        Some(4),
        |parameter| unsafe {
            assert_eq!(instance.prepare(2, &parameter), Err(-abi::EINVAL));
        },
    );
    assert_eq!(instance.sequence(REQUESTED_PARAMETER_SEQUENCE_ID), 1);
}

#[test]
fn sparse_source_discards_old_tokens_and_preserves_plan_across_reset() {
    rejected_configuration::<TunableRamp>();
    let instance = Instance::new::<TunableRamp>(c"2").unwrap();
    assert_eq!(instance.format(2).err(), Some(-abi::ENOENT));
    check_parameter_publication(&instance, 1, &mut [6.0, 7.0]);

    let mut result = [0.0; 2];
    let output_format = instance.format(0).unwrap();
    let (size, _) = with_buffer(&mut result, output_format, None, |output| {
        with_buffer(
            &mut [1.0, 2.0],
            instance.format(1).unwrap(),
            None,
            |parameter| {
                assert_eq!(
                    unsafe { instance.process(&[parameter], &mut [output]) },
                    -abi::EINVAL
                );
            },
        );
    });
    assert_eq!(size, 0);
    assert_eq!(instance.sequence(ACTIVE_PARAMETER_SEQUENCE_ID), 0);
    for (status, expected, bytes) in [
        (0, [6.0, 7.0], 8),
        (0, [7.0, 8.0], 8),
        (-abi::EINVAL, [7.0, 8.0], 0),
    ] {
        let (size, _) = with_buffer(&mut result, output_format, None, |output| {
            assert_eq!(
                unsafe { instance.process(&[DISCONNECTED], &mut [output]) },
                status
            );
        });
        assert_eq!((size, result), (bytes, expected));
    }
    assert_eq!(instance.sequence(ACTIVE_PARAMETER_SEQUENCE_ID), 1);
    assert_eq!(
        unsafe { (instance.descriptor.reset.unwrap())(instance.data) },
        0
    );
    with_buffer(&mut result, output_format, None, |output| {
        assert_eq!(
            unsafe { instance.process(&[DISCONNECTED], &mut [output]) },
            0
        );
    });
    assert_eq!(result, [6.0, 7.0]);
}

filtergraph_fgn::declare_ndarray_ports! {
    Ports {
        inputs { IN: f32 = "samples"; }
        outputs { OUT: f32 = "scaled"; }
        parameters { GAIN: f32 = "gain"; }
    }
}

struct Scaler;

impl PluginDeclaration for Scaler {
    type Configuration = i32;
    type Plan = f32;
    type Workspace = [f32; 2];
    type Plans = ParameterRuntime<f32>;
    type Error = i32;

    const NAME: &'static CStr = c"scaler";
    const PORTS: &'static [abi::PortInfo] = Ports::PORTS;

    fn prepare(value: i32) -> Result<PreparedInstance<f32, [f32; 2]>, i32> {
        if value <= 0 {
            return Err(value);
        }
        let shape = Box::new([2]);
        let mut formats = Vec::new();
        for schema in [DATA_SCHEMA, DATA_SCHEMA, PARAMETER_SCHEMA] {
            formats.push(
                NdarrayFormat::f32(shape.clone(), abi::LAYOUT_ROW_MAJOR, Some(schema))
                    .map_err(|_| abi::EINVAL)?,
            );
        }
        if value == 1 {
            formats.pop();
        }
        let prepared = PreparedInstance::new(value as f32, [0.0; 2], formats.into());
        assert_eq!(*prepared.plan(), value as f32);
        assert_eq!(prepared.workspace(), &[0.0; 2]);
        assert_eq!(prepared.formats().len(), if value == 1 { 2 } else { 3 });
        Ok(prepared)
    }

    fn process(
        plan: &f32,
        workspace: &mut [f32; 2],
        inputs: InputBuffers<'_>,
        outputs: &mut OutputBuffers<'_>,
    ) -> Result<(), i32> {
        assert!(!inputs.is_empty() && !outputs.is_empty());
        if !Ports::GAIN.is_disconnected(&inputs)? {
            return Err(abi::EINVAL);
        }
        let input = Ports::IN.slice(&inputs, 2)?;
        for (destination, value) in workspace.iter_mut().zip(input) {
            *destination = plan * value;
        }
        Ports::OUT.copy_from(outputs, workspace)?;
        Ports::OUT.copy_metadata_from(outputs, &inputs, Ports::IN)?;
        if let Some(sample) = inputs.sample_metadata(Ports::IN.index())? {
            // This fixture tags the final block of a two-block sample.
            outputs.set_sample_metadata(
                Ports::OUT.index(),
                sample.with_offset(1).with_terminal(true),
            )?;
        }
        Ports::OUT.finish(outputs, 2)
    }

    fn reset(_: &f32, workspace: &mut [f32; 2]) -> Result<(), i32> {
        *workspace = [0.0; 2];
        Ok(())
    }

    fn prepare_parameter(
        _: &f32,
        port: u32,
        parameter: ParameterBuffer<'_>,
    ) -> Option<Result<f32, i32>> {
        assert_eq!(port, Ports::GAIN.port());
        Some(Ports::GAIN.slice(parameter, 2).and_then(|values| {
            if values[1] != 0.0 {
                Err(abi::EINVAL)
            } else {
                Ok(values[0])
            }
        }))
    }

    fn errno(error: &i32) -> i32 {
        *error
    }
}

#[test]
fn topology_independent_declaration_projects_ports_and_uses_typed_buffers() {
    type Plugin = DeclaredPlugin<Scaler>;
    for (config, errno) in [
        (c"{}", abi::EINVAL),
        (c"0", abi::EFAULT),
        (c"-12", abi::ENOMEM),
        (c"1", abi::EINVAL),
    ] {
        assert_eq!(Instance::new::<Plugin>(config).err(), Some(-errno));
    }

    let instance = Instance::new::<Plugin>(c"2").unwrap();
    assert_eq!(
        (
            Ports::IN.index(),
            Ports::OUT.index(),
            Ports::GAIN.input_index()
        ),
        (0, 0, 1)
    );
    assert_eq!(instance.format(3).err(), Some(-abi::ENOENT));
    check_parameter_publication(&instance, Ports::GAIN.port(), &mut [4.0, 0.0]);

    let mut result = [9.0; 2];
    with_buffer(
        &mut result,
        instance.format(1).unwrap(),
        Some(0),
        |output| {
            with_buffer(
                &mut [2.0, 3.0],
                instance.format(0).unwrap(),
                Some(45),
                |input| {
                    assert_eq!(
                        unsafe { instance.process(&[input, DISCONNECTED], &mut [output]) },
                        0
                    );
                },
            );
        },
    );
    let (size, sequence) = with_buffer(
        &mut result,
        instance.format(1).unwrap(),
        Some(0),
        |output| {
            with_buffer(
                &mut [2.0, 3.0],
                instance.format(0).unwrap(),
                Some(55),
                |input| unsafe {
                    assert_eq!(instance.process(&[input, DISCONNECTED], &mut [output]), 0);
                    let header = &*(*(*output.buffer).metas).data.cast::<abi::SpaMetaHeader>();
                    assert_eq!(header.offset, 1);
                    assert_eq!(
                        header.flags & abi::META_HEADER_MARKER,
                        abi::META_HEADER_MARKER
                    );
                    assert_eq!(header.pts, 123);
                },
            );
        },
    );
    assert_eq!((size, sequence), (8, 55));

    assert_eq!(result, [8.0, 12.0]);
    with_buffer(
        &mut result,
        instance.format(1).unwrap(),
        Some(0),
        |output| {
            with_buffer(&mut [2.0], instance.format(0).unwrap(), Some(45), |input| {
                assert_eq!(
                    unsafe { instance.process(&[input, DISCONNECTED], &mut [output]) },
                    -abi::EMSGSIZE
                );
            });
        },
    );
    assert_eq!(result, [8.0, 12.0]);
    assert_eq!(
        unsafe { (instance.descriptor.reset.unwrap())(instance.data) },
        0
    );
}

#[test]
fn descriptor_rejects_incompatible_hosts_without_returning_an_instance() {
    let d = descriptor::<FixedGain>();
    let instantiate = d.instantiate.unwrap();
    let mut data = null_mut();
    for invalid_executor in [
        abi::Executor {
            struct_size: 4,
            ..executor()
        },
        abi::Executor {
            version: abi::EXECUTOR_VERSION + 1,
            ..executor()
        },
        abi::Executor {
            n_lanes: 0,
            ..executor()
        },
        abi::Executor {
            reserved: 1,
            ..executor()
        },
    ] {
        assert_eq!(
            unsafe { instantiate(&d, c"2".as_ptr(), &invalid_executor, &mut data) },
            -abi::EINVAL
        );
        assert!(data.is_null());
    }
    let executor = executor();
    for (descriptor, config, host) in [
        (null(), c"2".as_ptr(), &raw const executor),
        (&raw const d, null(), &raw const executor),
        (&raw const d, c"2".as_ptr(), null()),
    ] {
        assert_eq!(
            unsafe { instantiate(descriptor, config, host, &mut data) },
            -abi::EINVAL
        );
        assert!(data.is_null());
    }

    let d = abi::Descriptor { n_ports: 0, ..d };
    assert_eq!(
        unsafe { instantiate(&d, c"2".as_ptr(), &executor, &mut data) },
        -abi::EINVAL
    );
    assert!(data.is_null());
}

macro_rules! numeric_loopback {
    ($name:ident, $config:ident, $element:ty, $tag:ident, $schema:literal, $samples:expr) => {
        #[test]
        fn $name() {
            use core::convert::Infallible;
            use filtergraph_algorithms::NoWorkspace;
            use filtergraph_fgn::{AlgorithmInstance, descriptor_with};

            filtergraph_algorithms::filtergraph_algorithm! {
                Loopback {
                    label = "numeric-loopback";
                    configuration $config {
                        extent: usize = GraphRebuild;
                    }
                    plan = usize;
                    workspace = NoWorkspace;
                    properties = none;
                    dimensions |plan, _workspace| { extent = *plan; }
                    inputs {
                        input: $element [extent] = "in", schema = $schema;
                    }
                    outputs {
                        output: $element [extent] = "out", schema = $schema, metadata_from = "in";
                    }
                    prepare |config| { Ok((config.extent, NoWorkspace)) }
                    prepare_error = Infallible;
                    process |_plan, _workspace| {
                        output.copy_from_slice(input);
                        Ok(())
                    }
                    process_error = Infallible;
                }
            }

            static PORTS: [abi::PortInfo; 2] = [
                abi::PortInfo::new(0, abi::DIRECTION_INPUT, 0, c"in".as_ptr()),
                abi::PortInfo::new(1, abi::DIRECTION_OUTPUT, 0, c"out".as_ptr()),
            ];
            let instance = Instance::from_descriptor(
                descriptor_with::<AlgorithmInstance<Loopback>>(
                    c"numeric-loopback".as_ptr(),
                    2,
                    PORTS.as_ptr(),
                ),
                c"{\"extent\":2}",
            )
            .unwrap();
            let input_format = instance.format(0).unwrap();
            let output_format = instance.format(1).unwrap();
            assert_eq!(input_format.element_type, abi::$tag);
            assert_eq!(output_format.element_type, input_format.element_type);
            let mut samples: [$element; 2] = $samples;
            let mut result = [0 as $element; 2];
            let (size, sequence) =
                with_typed_buffer(&mut result, output_format, Some(0), |output| {
                    with_typed_buffer(&mut samples, input_format, Some(23), |input| {
                        assert_eq!(unsafe { instance.process(&[input], &mut [output]) }, 0);
                    });
                });
            assert_eq!((size as usize, sequence), (size_of_val(&samples), 23));
            assert_eq!(result, samples);
        }
    };
}

numeric_loopback!(
    preserves_i8_extremes,
    I8Config,
    i8,
    ELEMENT_I8,
    "test/i8/1",
    [i8::MIN, i8::MAX]
);
numeric_loopback!(
    preserves_i16_extremes,
    I16Config,
    i16,
    ELEMENT_I16_LE,
    "test/i16/1",
    [i16::MIN, i16::MAX]
);
numeric_loopback!(
    preserves_i32_extremes,
    I32Config,
    i32,
    ELEMENT_I32_LE,
    "test/i32/1",
    [i32::MIN, i32::MAX]
);
numeric_loopback!(
    preserves_i64_extremes,
    I64Config,
    i64,
    ELEMENT_I64_LE,
    "test/i64/1",
    [i64::MIN, i64::MAX]
);
numeric_loopback!(
    preserves_f64_precision,
    F64Config,
    f64,
    ELEMENT_F64_LE,
    "test/f64/1",
    [1.0 + f64::EPSILON, f64::MIN_POSITIVE]
);

#[test]
fn host_rejects_malformed_frames_without_advancing_workspace_history() {
    let instance = Instance::new::<FixedGain>(c"2").unwrap();
    let input_format = instance.format(0).unwrap();
    let output_format = instance.format(1).unwrap();
    let process = instance.descriptor.process.unwrap();
    let wrong_type =
        NdarrayFormat::packed::<i32>(Box::new([2]), abi::LAYOUT_ROW_MAJOR, Some(DATA_SCHEMA))
            .unwrap();
    let mut result = [71.0_f32; 2];
    let mut samples = [3.0_f32, -2.0];
    with_buffer(&mut result, output_format, None, |mut output| {
        with_buffer(&mut samples, input_format, None, |input| unsafe {
            assert_eq!(process(null_mut(), &input, 1, &mut output, 1), -abi::EINVAL);
            assert_eq!(
                process(instance.data, null(), 1, &mut output, 1),
                -abi::EINVAL
            );
            assert_eq!(
                process(instance.data, &input, 1, null_mut(), 1),
                -abi::EINVAL
            );
            assert_eq!(
                process(instance.data, &input, 0, &mut output, 1),
                -abi::EINVAL
            );
            assert_eq!(
                process(instance.data, &input, 1, &mut output, 0),
                -abi::EINVAL
            );
            // The host ABI rejects intersecting input and output descriptors.
            let mut shared = input;
            let shared = &raw mut shared;
            assert_eq!(process(instance.data, shared, 1, shared, 1), -abi::EINVAL);
            assert_eq!(instance.process(&[input], &mut [input]), -abi::EINVAL);
            assert_eq!(
                instance.process(&[DISCONNECTED], &mut [output]),
                -abi::EINVAL
            );
            assert_eq!(
                instance.process(&[input], &mut [DISCONNECTED]),
                -abi::EINVAL
            );

            let bad_formats = [null(), wrong_type.as_abi()];
            for format in bad_formats {
                let bad = abi::Buffer { format, ..input };
                assert_eq!(instance.process(&[bad], &mut [output]), -abi::EINVAL);
                assert_eq!((*(*(*output.buffer).datas).chunk).size, 0);
            }

            let data = (*input.buffer).datas;
            let chunk = (*data).chunk;
            (*input.buffer).n_datas = 0;
            assert_eq!(instance.process(&[input], &mut [output]), -abi::ENOBUFS);
            (*input.buffer).n_datas = 1;
            (*chunk).offset = 12;
            assert_eq!(instance.process(&[input], &mut [output]), -abi::EMSGSIZE);
            (*chunk).offset = 0;
            (*chunk).size = 4;
            assert_eq!(instance.process(&[input], &mut [output]), -abi::EMSGSIZE);
            (*chunk).size = 8;
            assert_eq!(instance.process(&[input], &mut [output]), 0);
        });
    });
    assert_eq!(result, [6.0, -4.0]);
}

#[test]
fn host_checks_port_and_metadata_callback_destinations() {
    let instance = Instance::new::<FixedGain>(c"2").unwrap();
    let d = &instance.descriptor;
    let mut f = null();
    let mut records = null();
    let mut count = 99;
    let mut value = abi::Value::NONE;
    let mut prop = core::mem::MaybeUninit::uninit();
    let mut token = null_mut();
    unsafe {
        assert_eq!(
            (d.get_port_format.unwrap())(null_mut(), 0, &mut f),
            -abi::EINVAL
        );
        assert_eq!(
            (d.get_port_format.unwrap())(instance.data, 0, null_mut()),
            -abi::EINVAL
        );
        assert_eq!(
            (d.enum_prop_info.unwrap())(null_mut(), 0, prop.as_mut_ptr()),
            -abi::EINVAL
        );
        assert_eq!(
            (d.enum_prop_info.unwrap())(instance.data, 0, null_mut()),
            -abi::EINVAL
        );
        assert_eq!(
            (d.get_prop.unwrap())(null_mut(), 0, &mut value),
            -abi::EINVAL
        );
        assert_eq!(
            (d.get_prop.unwrap())(instance.data, 0, null_mut()),
            -abi::EINVAL
        );
        assert_eq!((d.get_prop_revision.unwrap())(null_mut()), 0);
        assert_eq!(
            (d.prepare_props.unwrap())(null_mut(), null(), 0, &mut token),
            -abi::EINVAL
        );
        assert_eq!(
            (d.prepare_parameter.unwrap())(null_mut(), 0, &DISCONNECTED, &mut token),
            -abi::EINVAL
        );
        assert_eq!(
            (d.prepare_parameter.unwrap())(instance.data, 0, null(), &mut token),
            -abi::EINVAL
        );
        assert_eq!(
            (d.prepare_parameter.unwrap())(instance.data, 0, &DISCONNECTED, null_mut()),
            -abi::EINVAL
        );
        let metas = d.get_port_metadata.unwrap();
        assert_eq!(metas(null_mut(), 0, &mut records, &mut count), -abi::EINVAL);
        assert_eq!(
            metas(instance.data, 0, null_mut(), &mut count),
            -abi::EINVAL
        );
        assert_eq!(
            metas(instance.data, 0, &mut records, null_mut()),
            -abi::EINVAL
        );
        assert_eq!(
            metas(instance.data, 7, &mut records, &mut count),
            -abi::EINVAL
        );
        assert_eq!(count, 99);
        assert_eq!(metas(instance.data, 0, &mut records, &mut count), 0);
        assert_eq!(count, 0);
    }
    assert!(records.is_null() && f.is_null() && token.is_null());
}

mod plugin_registry {
    use super::*;

    filtergraph_fgn::export_plugin!(c"test-adapters", [FixedGain, TunableGain, TunableRamp]);

    #[test]
    fn exported_registry_resolves_exact_labels_for_the_supported_abi() {
        assert!(spa_filter_graph_ndarray_plugin_get_interface(abi::ABI_VERSION - 1).is_null());
        let raw = spa_filter_graph_ndarray_plugin_get_interface(abi::ABI_VERSION);
        assert!(!raw.is_null());
        unsafe {
            let plugin = &*raw;
            assert_eq!(CStr::from_ptr(plugin.name), c"test-adapters");
            let find = plugin.find_descriptor.unwrap();
            for (index, name) in [c"fixed-gain", c"tunable-gain", c"tunable-ramp"]
                .into_iter()
                .enumerate()
            {
                assert_eq!(
                    find(name.as_ptr()),
                    &raw const filtergraph_fgn_registered_descriptors()[index]
                );
            }
            assert!(find(c"Fixed-gain".as_ptr()).is_null());
            assert!(find(null()).is_null());
        }
    }
}

mod duplicate_registry {
    use super::*;

    // Avoid defining another exported C symbol in the same test executable.
    #[test]
    fn duplicate_empty_and_overlong_plugin_names_fail_admission() {
        let valid = descriptor::<FixedGain>();
        let duplicate = [descriptor::<FixedGain>(), descriptor::<FixedGain>()];
        let name = std::ffi::CString::new("a".repeat(abi::LOCAL_NAME_MAX + 1)).unwrap();
        let long = abi::Descriptor {
            name: name.as_ptr(),
            ..descriptor::<FixedGain>()
        };
        let empty = abi::Descriptor {
            name: c"".as_ptr(),
            ..descriptor::<FixedGain>()
        };
        unsafe {
            assert!(filtergraph_fgn::plugin_registry_is_valid(c"one", &[valid]));
            assert!(!filtergraph_fgn::plugin_registry_is_valid(
                c"",
                &duplicate[..1]
            ));
            assert!(!filtergraph_fgn::registry_is_valid(&duplicate));
            assert!(!filtergraph_fgn::registry_is_valid(&[long]));
            assert!(!filtergraph_fgn::registry_is_valid(&[empty]));
        }
    }
}

#[test]
fn missing_output_header_fails_without_publishing_a_metadata_dependent_frame() {
    let instance = Instance::new::<DeclaredPlugin<Scaler>>(c"2").unwrap();
    let mut result = [71.0; 2];
    let (size, _) = with_buffer(&mut result, instance.format(1).unwrap(), None, |output| {
        with_buffer(
            &mut [3.0, 2.0],
            instance.format(0).unwrap(),
            Some(19),
            |input| {
                assert_eq!(
                    unsafe { instance.process(&[input, DISCONNECTED], &mut [output]) },
                    -abi::ENOENT
                );
            },
        );
    });
    assert_eq!(size, 0);
}

#[test]
fn short_output_and_misaligned_input_do_not_publish_or_advance_history() {
    let instance = Instance::new::<FixedGain>(c"2").unwrap();
    let mut short = [71.0; 1];
    let input_format = instance.format(0).unwrap();
    let output_format = instance.format(1).unwrap();
    let (size, _) = with_buffer(&mut short, output_format, None, |output| {
        with_buffer(&mut [3.0, 2.0], input_format, None, |input| {
            assert_eq!(
                unsafe { instance.process(&[input], &mut [output]) },
                -abi::ENOSPC
            );
        });
    });
    assert_eq!((size, short), (0, [71.0]));

    let mut result = [71.0; 2];
    let (size, _) = with_buffer(&mut result, output_format, None, |output| {
        with_buffer(&mut [3.0, 2.0, 0.0], input_format, None, |input| unsafe {
            let chunk = (*(*input.buffer).datas).chunk;
            (*chunk).offset = 1;
            (*chunk).size = 8;
            assert_eq!(instance.process(&[input], &mut [output]), -abi::EINVAL);
        });
    });
    assert_eq!((size, result), (0, [71.0; 2]));

    with_buffer(&mut result, output_format, None, |output| {
        with_buffer(&mut [3.0, 2.0], input_format, None, |input| {
            assert_eq!(unsafe { instance.process(&[input], &mut [output]) }, 0);
        });
    });
    assert_eq!(result, [6.0, 4.0]);
}
