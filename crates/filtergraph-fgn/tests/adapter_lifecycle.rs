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
        let descriptor = Box::new(descriptor::<I>());
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
