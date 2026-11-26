//! End-to-end ABI proof for a generated portable algorithm declaration.

use core::ffi::{CStr, c_void};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use filtergraph_algorithms::{
    AlgorithmPlan, AlgorithmProperties, ClosedLoopCorrectionF32, ClosedLoopCorrectionInput,
    ClosedLoopCorrectionPlan, IncrementalDenseReconstructorF32,
    IncrementalDenseReconstructorF32Configuration, LeakyIntegratorF32, NdarrayCopyF32,
    OpticalGainCorrectionF32, PdmCommandF32, PdmEngineeringOffsetF32, PreparedGemv,
    PropertyAssignment, PropertyValue, PyramidPupilImageF32, PyramidRowReconstructorF32,
    PyramidRowReconstructorF32Configuration, SampleMetadata, ShackHartmannImageF32,
    ShackHartmannImageF32Configuration, ShackHartmannReconstructorF32, VdmToPdmF32,
};
use filtergraph_fgn::{
    ACTIVE_GENERATION_ID, ACTIVE_PARAMETER_SEQUENCE_ID, AlgorithmDescriptor, Bool8,
    REQUESTED_GENERATION_ID, REQUESTED_PARAMETER_SEQUENCE_ID, abi,
};

unsafe extern "C" fn test_run_dense_f32(_data: *mut c_void, task: *const abi::DenseF32Task) -> i32 {
    if task.is_null() {
        return -abi::EINVAL;
    }
    // SAFETY: this test executor is called only with adapter-validated tasks.
    let task = unsafe { &*task };
    if task.struct_size < size_of::<abi::DenseF32Task>() as u32
        || task.matrix.is_null()
        || task.input.is_null()
        || task.output.is_null()
        || task.n_rows == 0
        || task.n_columns == 0
        || task.first_column > task.matrix_row_stride
        || task.n_columns > task.matrix_row_stride - task.first_column
    {
        return -abi::EINVAL;
    }
    let matrix_elements = (task.n_rows as usize - 1) * task.matrix_row_stride as usize
        + task.first_column as usize
        + task.n_columns as usize;
    // SAFETY: adapter-owned plan and admitted buffers cover these exact extents.
    let matrix = unsafe { core::slice::from_raw_parts(task.matrix, matrix_elements) };
    let input = unsafe { core::slice::from_raw_parts(task.input, task.n_columns as usize) };
    let output = unsafe { core::slice::from_raw_parts_mut(task.output, task.n_rows as usize) };
    for (row, output) in output.iter_mut().enumerate() {
        let start = row * task.matrix_row_stride as usize + task.first_column as usize;
        let coefficients = &matrix[start..start + task.n_columns as usize];
        let mut sum = if task.flags & abi::DENSE_F32_ACCUMULATE != 0 {
            *output
        } else {
            0.0
        };
        for (&coefficient, &input) in coefficients.iter().zip(input) {
            sum += coefficient * input;
        }
        *output = sum;
    }
    0
}

static LANE_CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn test_run_lanes(_data: *mut c_void, task: *const abi::LanesTask) -> i32 {
    if task.is_null() {
        return -abi::EINVAL;
    }
    // SAFETY: this test executor is called only with adapter-validated tasks.
    let task = unsafe { &*task };
    let Some(run) = task.run else {
        return -abi::EINVAL;
    };
    if task.struct_size < size_of::<abi::LanesTask>() as u32
        || task.flags != 0
        || task.n_lanes == 0
        || task.n_lanes > 4
        || task.reserved != 0
    {
        return -abi::EINVAL;
    }
    LANE_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut result = 0;
    for lane in 0..task.n_lanes {
        // SAFETY: the adapter's callback and context remain borrowed through
        // this synchronous mock dispatch.
        let status = unsafe { run(task.data, lane, task.n_lanes) };
        if result == 0 && status != 0 {
            result = status;
        }
    }
    result
}

struct TestExecutor(abi::Executor);

// SAFETY: this immutable fixture has a null data pointer and is retained for
// the complete test process.
unsafe impl Sync for TestExecutor {}

static TEST_EXECUTOR: TestExecutor = TestExecutor(abi::Executor {
    struct_size: size_of::<abi::Executor>() as u32,
    version: abi::EXECUTOR_VERSION,
    flags: 0,
    n_lanes: 4,
    data: core::ptr::null_mut(),
    run_dense_f32: Some(test_run_dense_f32),
    run_lanes: Some(test_run_lanes),
    // The sequential mock has no sharing. A one-byte quantum makes every
    // admitted Vec exercise the generated partition path deterministically.
    cache_line_size: 1,
    reserved: 0,
});

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static DEALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
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

fn float_property(id: u32, value: f32) -> abi::Property {
    abi::Property {
        id,
        reserved: 0,
        value: abi::Value {
            type_: abi::TYPE_FLOAT,
            reserved: 0,
            value: abi::ValueBody { float_value: value },
        },
    }
}

fn spa_data<T>(values: &mut [T], chunk: &mut abi::SpaChunk) -> abi::SpaData {
    let bytes = u32::try_from(size_of_val(values)).unwrap();
    *chunk = abi::SpaChunk {
        offset: 0,
        size: bytes,
        stride: i32::try_from(bytes).unwrap(),
        flags: 0,
    };
    abi::SpaData {
        type_: 0,
        flags: 0,
        fd: -1,
        mapoffset: 0,
        maxsize: bytes,
        data: values.as_mut_ptr().cast::<c_void>(),
        chunk,
    }
}

fn long_property(descriptor: &abi::Descriptor, instance: *mut c_void, id: u32) -> i64 {
    let mut value = abi::Value::NONE;
    // SAFETY: instance belongs to this descriptor and value is writable.
    assert_eq!(
        unsafe { (descriptor.get_prop.unwrap())(instance, id, &mut value) },
        0
    );
    assert_eq!(value.type_, abi::TYPE_LONG);
    // SAFETY: the checked tag selects this union member.
    unsafe { value.value.long_integer }
}

fn large_frame_origins() -> Vec<[usize; 2]> {
    let mut origins = Vec::with_capacity(16 * 16);
    for grid_row in 0..16 {
        for grid_column in 0..16 {
            origins.push([grid_row * 22, grid_column * 22]);
        }
    }
    origins
}

fn large_frame_image() -> Vec<f32> {
    let mut image = vec![0.0_f32; 352 * 352];
    for row in 0..352 {
        for column in 0..352 {
            let subaperture = (row / 22) * 16 + column / 22;
            image[row * 352 + column] = 1.0
                + 0.01 * (row % 22) as f32
                + 0.02 * (column % 22) as f32
                + 0.001 * subaperture as f32;
        }
    }
    image
}

#[test]
fn generated_descriptor_preserves_formats_properties_state_and_realtime_boundary() {
    let owned = AlgorithmDescriptor::new::<LeakyIntegratorF32>().unwrap();
    let descriptor = owned.as_abi();
    assert_eq!(descriptor.n_ports, 2);
    assert!(descriptor.prepare_parameters.is_none());
    assert!(descriptor.adopt_parameters.is_none());
    // SAFETY: the owned descriptor retains its C strings and port table.
    unsafe {
        assert_eq!(CStr::from_ptr(descriptor.name), c"leaky-integrator-f32");
        let ports = core::slice::from_raw_parts(descriptor.ports, descriptor.n_ports as usize);
        assert_eq!(CStr::from_ptr(ports[0].name), c"input");
        assert_eq!(ports[0].direction, abi::DIRECTION_INPUT);
        assert_eq!(CStr::from_ptr(ports[1].name), c"output");
        assert_eq!(ports[1].direction, abi::DIRECTION_OUTPUT);
    }

    let mut instance = core::ptr::null_mut();
    // SAFETY: every pointer in this test remains valid for its callback.
    unsafe {
        let mut rejected = core::ptr::null_mut();
        assert_eq!(
            (descriptor.instantiate.unwrap())(
                descriptor,
                c"{\"extent\":2,\"initial_state\":4.0,\"input_schema\":\"pipewireao/controller-residual-error-f32/1\",\"output_schema\":\"pipewireao/controller-command-f32/1\",\"rate\":[1000,1],\"profile\":\"retired\"}"
                    .as_ptr(),
                &TEST_EXECUTOR.0,
                &mut rejected,
            ),
            -abi::EINVAL
        );
        assert!(rejected.is_null());

        for profile_configuration in [
            c"{\"extent\":2,\"initial_state\":4.0,\"input_schema\":\"pipewireao/controller-residual-error-f32/1\",\"output_schema\":\"pipewireao/controller-command-f32/1\",\"rate\":[1000,1],\"profiles\":{\"input\":\"test.profile\"}}",
            c"{\"extent\":2,\"initial_state\":4.0,\"input_schema\":\"pipewireao/controller-residual-error-f32/1\",\"output_schema\":\"pipewireao/controller-command-f32/1\",\"rate\":[1000,1],\"profiles\":{}}",
        ] {
            let mut profile_instance = core::ptr::null_mut();
            let profile_result = (descriptor.instantiate.unwrap())(
                descriptor,
                profile_configuration.as_ptr(),
                &TEST_EXECUTOR.0,
                &mut profile_instance,
            );
            assert_eq!(profile_result, -abi::EINVAL);
            assert!(profile_instance.is_null());
        }

        assert_eq!(
            (descriptor.instantiate.unwrap())(
                descriptor,
                c"{\"extent\":2,\"initial_state\":4.0,\"input_schema\":\"pipewireao/controller-residual-error-f32/1\",\"output_schema\":\"pipewireao/controller-command-f32/1\",\"rate\":[1000,1]}"
                    .as_ptr(),
                &TEST_EXECUTOR.0,
                &mut instance,
            ),
            0
        );
        assert!(!instance.is_null());

        let mut input_format = core::ptr::null();
        let mut output_format = core::ptr::null();
        assert_eq!(
            (descriptor.get_port_format.unwrap())(instance, 0, &mut input_format),
            0
        );
        assert_eq!(
            (descriptor.get_port_format.unwrap())(instance, 1, &mut output_format),
            0
        );
        for (format, schema) in [input_format, output_format].into_iter().zip([
            c"pipewireao/controller-residual-error-f32/1",
            c"pipewireao/controller-command-f32/1",
        ]) {
            let format = &*format;
            assert_eq!(format.element_type, abi::ELEMENT_F32_LE);
            assert_eq!(format.layout, abi::LAYOUT_ROW_MAJOR);
            assert_eq!((format.rate_num, format.rate_denom), (1000, 1));
            assert_eq!(format.n_dimensions, 1);
            assert_eq!(*format.shape, 2);
            assert_eq!(CStr::from_ptr(format.schema), schema);
        }

        let mut info = core::mem::zeroed::<abi::PropertyInfo>();
        assert_eq!(
            (descriptor.enum_prop_info.unwrap())(instance, 0, &mut info),
            1
        );
        assert_eq!(info.id, 0);
        assert_eq!(CStr::from_ptr(info.name), c"gain");
        assert_eq!(
            (descriptor.enum_prop_info.unwrap())(instance, 1, &mut info),
            1
        );
        assert_eq!(info.id, 1);
        assert_eq!(CStr::from_ptr(info.name), c"pole");

        let assignments = [float_property(0, 0.5), float_property(1, 0.75)];
        let mut prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_props.unwrap())(
                instance,
                assignments.as_ptr(),
                assignments.len() as u32,
                &mut prepared,
            ),
            0
        );
        assert!(!prepared.is_null());
        (descriptor.commit_props.unwrap())(instance, prepared);
        assert_eq!(
            long_property(descriptor, instance, REQUESTED_GENERATION_ID),
            1
        );
        assert_eq!(long_property(descriptor, instance, ACTIVE_GENERATION_ID), 0);

        let mut input = [2.0_f32, 2.0];
        let mut output = [0.0_f32; 2];
        let mut input_chunk = core::mem::zeroed();
        let mut output_chunk = core::mem::zeroed();
        let mut input_data = spa_data(&mut input, &mut input_chunk);
        let mut output_data = spa_data(&mut output, &mut output_chunk);
        let mut input_header = abi::SpaMetaHeader {
            flags: 7,
            offset: 3,
            pts: 99,
            dts_offset: -2,
            seq: 42,
        };
        let mut output_header = core::mem::zeroed::<abi::SpaMetaHeader>();
        let mut input_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut input_header).cast(),
        };
        let mut output_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut output_header).cast(),
        };
        let mut input_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut input_meta,
            datas: &mut input_data,
        };
        let mut output_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut output_meta,
            datas: &mut output_data,
        };
        let inputs = [abi::Buffer {
            buffer: &mut input_buffer,
            format: input_format,
        }];
        let mut outputs = [abi::Buffer {
            buffer: &mut output_buffer,
            format: output_format,
        }];
        let process = descriptor.process.unwrap();

        assert_eq!(
            process(instance, inputs.as_ptr(), 1, outputs.as_mut_ptr(), 1),
            0
        );
        assert_eq!(output, [4.0, 4.0]);
        assert_eq!(output_chunk.size as usize, size_of_val(&output));
        assert_eq!(output_header.seq, 42);
        assert_eq!(long_property(descriptor, instance, ACTIVE_GENERATION_ID), 1);

        input.copy_from_slice(&[4.0, -4.0]);
        assert_eq!(
            allocations(|| {
                assert_eq!(
                    process(instance, inputs.as_ptr(), 1, outputs.as_mut_ptr(), 1),
                    0
                );
            }),
            (0, 0)
        );
        assert_eq!(output, [5.0, 1.0]);

        assert_eq!((descriptor.reset.unwrap())(instance), 0);
        input.fill(0.0);
        assert_eq!(
            process(instance, inputs.as_ptr(), 1, outputs.as_mut_ptr(), 1),
            0
        );
        assert_eq!(output, [3.0, 3.0]);

        (descriptor.cleanup.unwrap())(instance);
    }
}

#[test]
fn generated_sparse_parameter_replaces_one_complete_plan_at_a_process_boundary() {
    let owned = AlgorithmDescriptor::new::<OpticalGainCorrectionF32>().unwrap();
    let descriptor = owned.as_abi();
    assert_eq!(descriptor.n_ports, 3);
    assert!(descriptor.prepare_parameters.is_some());
    assert!(descriptor.adopt_parameters.is_some());

    // SAFETY: the owned descriptor retains its C strings and port table, and
    // every callback pointer below refers to local storage for the full call.
    unsafe {
        assert_eq!(
            CStr::from_ptr(descriptor.name),
            c"optical-gain-correction-f32"
        );
        let ports = core::slice::from_raw_parts(descriptor.ports, descriptor.n_ports as usize);
        assert_eq!(CStr::from_ptr(ports[2].name), c"optical-gains");
        assert_eq!(ports[2].direction, abi::DIRECTION_INPUT);
        assert_eq!(ports[2].flags, abi::PORT_OPTIONAL | abi::PORT_PARAMETER);

        let mut instance = core::ptr::null_mut();
        assert_eq!(
            (descriptor.instantiate.unwrap())(
                descriptor,
                c"{\"optical_gains\":[0.5,2.0]}".as_ptr(),
                &TEST_EXECUTOR.0,
                &mut instance,
            ),
            0
        );

        let mut input_format = core::ptr::null();
        let mut output_format = core::ptr::null();
        let mut parameter_format = core::ptr::null();
        assert_eq!(
            (descriptor.get_port_format.unwrap())(instance, 0, &mut input_format),
            0
        );
        assert_eq!(
            (descriptor.get_port_format.unwrap())(instance, 1, &mut output_format),
            0
        );
        assert_eq!(
            (descriptor.get_port_format.unwrap())(instance, 2, &mut parameter_format),
            0
        );

        let mut parameter_values = [2.0_f32, 4.0];
        let mut parameter_chunk = core::mem::zeroed();
        let mut parameter_data = spa_data(&mut parameter_values, &mut parameter_chunk);
        let mut parameter_header = abi::SpaMetaHeader {
            flags: 0,
            offset: 0,
            pts: 0,
            dts_offset: 0,
            seq: 9,
        };
        let mut parameter_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut parameter_header).cast(),
        };
        let mut parameter_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut parameter_meta,
            datas: &mut parameter_data,
        };
        let parameter = abi::Buffer {
            buffer: &mut parameter_buffer,
            format: parameter_format,
        };
        let mut prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_parameter.unwrap())(instance, 2, &parameter, &mut prepared),
            0
        );
        assert!(!prepared.is_null());
        (descriptor.commit_parameter.unwrap())(instance, prepared);
        assert_eq!(
            long_property(descriptor, instance, REQUESTED_PARAMETER_SEQUENCE_ID),
            9
        );
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            0
        );

        let mut input = [1.0_f32, 4.0];
        let mut output = [0.0_f32; 2];
        let mut input_chunk = core::mem::zeroed();
        let mut output_chunk = core::mem::zeroed();
        let mut input_data = spa_data(&mut input, &mut input_chunk);
        let mut output_data = spa_data(&mut output, &mut output_chunk);
        let mut input_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut input_data,
        };
        let mut output_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut output_data,
        };
        let inputs = [
            abi::Buffer {
                buffer: &mut input_buffer,
                format: input_format,
            },
            abi::Buffer {
                buffer: core::ptr::null_mut(),
                format: parameter_format,
            },
        ];
        let mut outputs = [abi::Buffer {
            buffer: &mut output_buffer,
            format: output_format,
        }];
        let process = descriptor.process.unwrap();

        assert_eq!(
            allocations(|| {
                assert_eq!(
                    process(
                        instance,
                        inputs.as_ptr(),
                        inputs.len() as u32,
                        outputs.as_mut_ptr(),
                        outputs.len() as u32,
                    ),
                    0
                );
            }),
            (0, 0)
        );
        assert_eq!(output, [0.5, 1.0]);
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            9
        );

        (descriptor.cleanup.unwrap())(instance);
    }
}

#[test]
fn generated_engineering_offset_reports_adopted_values_at_a_process_boundary() {
    let owned = AlgorithmDescriptor::new::<PdmEngineeringOffsetF32>().unwrap();
    let descriptor = owned.as_abi();
    assert_eq!(descriptor.n_ports, 4);
    assert!(descriptor.prepare_parameters.is_some());
    assert!(descriptor.adopt_parameters.is_some());

    // SAFETY: the owned descriptor retains its C strings and port table, and
    // every callback pointer below refers to local storage for the full call.
    unsafe {
        assert_eq!(
            CStr::from_ptr(descriptor.name),
            c"pdm-engineering-offset-f32"
        );
        let ports = core::slice::from_raw_parts(descriptor.ports, descriptor.n_ports as usize);
        assert_eq!(CStr::from_ptr(ports[3].name), c"engineering-offset");
        assert_eq!(ports[3].direction, abi::DIRECTION_INPUT);
        assert_eq!(ports[3].flags, abi::PORT_OPTIONAL | abi::PORT_PARAMETER);

        let mut instance = core::ptr::null_mut();
        assert_eq!(
            (descriptor.instantiate.unwrap())(
                descriptor,
                c"{\"actuator_count\":2}".as_ptr(),
                &TEST_EXECUTOR.0,
                &mut instance,
            ),
            0
        );

        let mut input_format = core::ptr::null();
        let mut output_format = core::ptr::null();
        let mut parameter_format = core::ptr::null();
        let mut current_format = core::ptr::null();
        assert_eq!(
            (descriptor.get_port_format.unwrap())(instance, 0, &mut input_format),
            0
        );
        assert_eq!(
            (descriptor.get_port_format.unwrap())(instance, 1, &mut output_format),
            0
        );
        assert_eq!(
            (descriptor.get_port_format.unwrap())(instance, 3, &mut parameter_format),
            0
        );

        assert_eq!(
            (descriptor.get_port_format.unwrap())(instance, 2, &mut current_format),
            0
        );
        let mut parameter_values = [2.0_f32, 4.0];
        let mut parameter_chunk = core::mem::zeroed();
        let mut parameter_data = spa_data(&mut parameter_values, &mut parameter_chunk);
        let mut parameter_header = abi::SpaMetaHeader {
            flags: 0,
            offset: 0,
            pts: 0,
            dts_offset: 0,
            seq: 9,
        };
        let mut parameter_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut parameter_header).cast(),
        };
        let mut parameter_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut parameter_meta,
            datas: &mut parameter_data,
        };
        let parameter = abi::Buffer {
            buffer: &mut parameter_buffer,
            format: parameter_format,
        };
        let mut prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_parameter.unwrap())(instance, 3, &parameter, &mut prepared),
            0
        );
        assert!(!prepared.is_null());
        (descriptor.commit_parameter.unwrap())(instance, prepared);
        assert_eq!(
            long_property(descriptor, instance, REQUESTED_PARAMETER_SEQUENCE_ID),
            9
        );
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            0
        );

        let mut input = [1.0_f32, 4.0];
        let mut output = [0.0_f32; 2];
        let mut current = [0.0_f32; 2];
        let mut current_chunk = core::mem::zeroed();
        let mut current_data = spa_data(&mut current, &mut current_chunk);
        let mut current_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut current_data,
        };
        let mut input_chunk = core::mem::zeroed();
        let mut output_chunk = core::mem::zeroed();
        let mut input_data = spa_data(&mut input, &mut input_chunk);
        let mut output_data = spa_data(&mut output, &mut output_chunk);
        let mut input_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut input_data,
        };
        let mut output_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut output_data,
        };
        let inputs = [
            abi::Buffer {
                buffer: &mut input_buffer,
                format: input_format,
            },
            abi::Buffer {
                buffer: core::ptr::null_mut(),
                format: parameter_format,
            },
        ];
        let mut outputs = [
            abi::Buffer {
                buffer: &mut output_buffer,
                format: output_format,
            },
            abi::Buffer {
                buffer: &mut current_buffer,
                format: current_format,
            },
        ];
        let process = descriptor.process.unwrap();

        assert_eq!(
            allocations(|| {
                assert_eq!(
                    process(
                        instance,
                        inputs.as_ptr(),
                        inputs.len() as u32,
                        outputs.as_mut_ptr(),
                        outputs.len() as u32,
                    ),
                    0
                );
            }),
            (0, 0)
        );
        assert_eq!(output, [3.0, 8.0]);
        assert_eq!(current, [2.0, 4.0]);
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            9
        );

        parameter_values[1] = f32::NAN;
        parameter_header.seq = 10;
        std::hint::black_box(&parameter_header);
        prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_parameter.unwrap())(instance, 3, &parameter, &mut prepared),
            -abi::EINVAL
        );
        assert!(prepared.is_null());
        assert_eq!(
            long_property(descriptor, instance, REQUESTED_PARAMETER_SEQUENCE_ID),
            9
        );
        assert_eq!(
            process(instance, inputs.as_ptr(), 2, outputs.as_mut_ptr(), 2),
            0
        );
        assert_eq!(output, [3.0, 8.0]);
        assert_eq!(current, [2.0, 4.0]);

        parameter_values.copy_from_slice(&[-0.5, 0.25]);
        parameter_header.seq = 11;
        std::hint::black_box(&parameter_header);
        assert_eq!(
            (descriptor.prepare_parameter.unwrap())(instance, 3, &parameter, &mut prepared),
            0
        );
        (descriptor.commit_parameter.unwrap())(instance, prepared);
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            9
        );
        assert_eq!(current, [2.0, 4.0]);
        assert_eq!(
            allocations(|| {
                assert_eq!(
                    process(instance, inputs.as_ptr(), 2, outputs.as_mut_ptr(), 2),
                    0
                );
            }),
            (0, 0)
        );
        assert_eq!(output, [0.5, 4.25]);
        assert_eq!(current, [-0.5, 0.25]);
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            11
        );

        input[1] = f32::NAN;
        std::hint::black_box(&input);
        assert!(process(instance, inputs.as_ptr(), 2, outputs.as_mut_ptr(), 2) < 0);
        assert_eq!(output, [0.5, 4.25]);
        assert_eq!(current, [-0.5, 0.25]);
        (descriptor.cleanup.unwrap())(instance);
    }
}

#[test]
fn generated_pyramid_image_node_uses_views_and_adopts_a_shape_preserving_mask() {
    let owned = AlgorithmDescriptor::new::<PyramidPupilImageF32>().unwrap();
    let descriptor = owned.as_abi();
    assert_eq!(descriptor.n_ports, 5);

    // SAFETY: descriptor-owned pointers and all local callback storage remain
    // alive until this instance is cleaned up exactly once.
    unsafe {
        let mut instance = core::ptr::null_mut();
        assert_eq!(
            (descriptor.instantiate.unwrap())(
                descriptor,
                c"{\"image_size\":[4,4],\"pupil_size\":[2,2],\"pupil_origins\":[[0,0],[0,2],[2,0],[2,2]],\"pupil_mask\":[true,false,true,true],\"rate\":[1000,1]}"
                    .as_ptr(),
                &TEST_EXECUTOR.0,
                &mut instance,
            ),
            0
        );
        let mut formats = [core::ptr::null(); 5];
        for (port, format) in formats.iter_mut().enumerate() {
            assert_eq!(
                (descriptor.get_port_format.unwrap())(instance, port as u32, format),
                0
            );
        }
        assert_eq!(core::slice::from_raw_parts((*formats[0]).shape, 2), &[4, 4]);
        assert_eq!(core::slice::from_raw_parts((*formats[1]).shape, 2), &[4, 3]);
        assert_eq!(core::slice::from_raw_parts((*formats[4]).shape, 2), &[2, 2]);

        let mut image = (1_u32..=16).map(|value| value as f32).collect::<Vec<_>>();
        let mut pixels = [0.0_f32; 12];
        let mut mean = [0.0_f32];
        let mut image_chunk = core::mem::zeroed();
        let mut pixels_chunk = core::mem::zeroed();
        let mut mean_chunk = core::mem::zeroed();
        let mut image_data = spa_data(&mut image, &mut image_chunk);
        let mut pixels_data = spa_data(&mut pixels, &mut pixels_chunk);
        let mut mean_data = spa_data(&mut mean, &mut mean_chunk);
        let mut image_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut image_data,
        };
        let mut pixels_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut pixels_data,
        };
        let mut mean_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut mean_data,
        };
        let disconnected = |format| abi::Buffer {
            buffer: core::ptr::null_mut(),
            format,
        };
        let inputs = [
            abi::Buffer {
                buffer: &mut image_buffer,
                format: formats[0],
            },
            disconnected(formats[3]),
            disconnected(formats[4]),
        ];
        let mut outputs = [
            abi::Buffer {
                buffer: &mut pixels_buffer,
                format: formats[1],
            },
            abi::Buffer {
                buffer: &mut mean_buffer,
                format: formats[2],
            },
        ];
        let process = descriptor.process.unwrap();
        for frame in 0..2 {
            assert_eq!(
                process(
                    instance,
                    inputs.as_ptr(),
                    inputs.len() as u32,
                    outputs.as_mut_ptr(),
                    outputs.len() as u32,
                ),
                0
            );
            assert_eq!(mean, [8.5]);
            if frame == 0 {
                assert_eq!(pixels, [0.0; 12]);
            }
        }
        assert_eq!(pixels[0..3], [1.0 / 8.5, 5.0 / 8.5, 6.0 / 8.5]);
        assert_eq!(
            allocations(|| {
                assert_eq!(
                    process(
                        instance,
                        inputs.as_ptr(),
                        inputs.len() as u32,
                        outputs.as_mut_ptr(),
                        outputs.len() as u32,
                    ),
                    0
                );
            }),
            (0, 0)
        );

        let mut mask = [
            Bool8::new(false),
            Bool8::new(true),
            Bool8::new(true),
            Bool8::new(true),
        ];
        let mut mask_chunk = core::mem::zeroed();
        let mut mask_data = spa_data(&mut mask, &mut mask_chunk);
        let mut mask_header = abi::SpaMetaHeader {
            flags: 0,
            offset: 0,
            pts: 0,
            dts_offset: 0,
            seq: 31,
        };
        let mut mask_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut mask_header).cast(),
        };
        let mut mask_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut mask_meta,
            datas: &mut mask_data,
        };
        let parameter = abi::Buffer {
            buffer: &mut mask_buffer,
            format: formats[4],
        };
        let mut prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_parameter.unwrap())(instance, 4, &parameter, &mut prepared),
            0
        );
        (descriptor.commit_parameter.unwrap())(instance, prepared);
        assert_eq!(
            allocations(|| {
                assert_eq!(
                    process(
                        instance,
                        inputs.as_ptr(),
                        inputs.len() as u32,
                        outputs.as_mut_ptr(),
                        outputs.len() as u32,
                    ),
                    0
                );
            }),
            (0, 0)
        );
        assert_eq!(pixels[0..3], [2.0 / 8.5, 5.0 / 8.5, 6.0 / 8.5]);
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            31
        );

        assert_eq!((descriptor.reset.unwrap())(instance), 0);
        assert_eq!(
            process(
                instance,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
            ),
            0
        );
        assert_eq!(pixels, [0.0; 12]);
        (descriptor.cleanup.unwrap())(instance);
    }
}

#[test]
fn generated_large_frame_shack_hartmann_node_matches_native_and_adopts_bool_parameter() {
    const SUBAPERTURES: usize = 256;
    let origins = large_frame_origins();
    let configuration = ShackHartmannImageF32Configuration {
        subaperture_count: SUBAPERTURES,
        image_schema: "pipewireao/calibrated-pixels-f32/1".into(),
        initial_subaperture_origins: Some(origins),
        coordinate_scale: 0.5,
        pixel_threshold: 0.0,
        flux_threshold: 1.0,
        reference_slopes: None,
        active: None,
        image_size: (352, 352),
        subaperture_size: (22, 22),
    };
    let mut native = ShackHartmannImageF32::prepare(configuration).unwrap();
    let mut image = large_frame_image();
    let mut native_slopes = vec![0.0_f32; SUBAPERTURES * 2];
    let mut native_flux = vec![0.0_f32; SUBAPERTURES];
    let mut native_validity = vec![false; SUBAPERTURES];
    ShackHartmannImageF32::process(
        &mut native,
        &image,
        &mut native_slopes,
        &mut native_flux,
        &mut native_validity,
    )
    .unwrap();
    assert_ne!(native_slopes[0], 0.0);

    let config = std::ffi::CString::new(
        serde_json::to_string(&serde_json::json!({
            "image_size": [352, 352],
            "subaperture_size": [22, 22],
            "subaperture_count": SUBAPERTURES,
            "image_schema": "org.adaptiveopticssim.hil-reference.shack-hartmann-frame.f32/1",
            "coordinate_scale": 0.5,
            "pixel_threshold": 0.0,
            "flux_threshold": 1.0,
            "initial_subaperture_origins": null,
            "reference_slopes": null,
            "active": null,
            "rate": [1000, 1]
        }))
        .unwrap(),
    )
    .unwrap();
    let owned = AlgorithmDescriptor::new::<ShackHartmannImageF32>().unwrap();
    let descriptor = owned.as_abi();
    assert_eq!(descriptor.n_ports, 9);

    // SAFETY: descriptor-owned pointers and all local buffers remain live for
    // every callback; cleanup consumes the instance exactly once.
    unsafe {
        let mut instance = core::ptr::null_mut();
        assert_eq!(
            (descriptor.instantiate.unwrap())(
                descriptor,
                config.as_ptr(),
                &TEST_EXECUTOR.0,
                &mut instance,
            ),
            0
        );
        let mut formats = [core::ptr::null(); 9];
        for (port, format) in formats.iter_mut().enumerate() {
            assert_eq!(
                (descriptor.get_port_format.unwrap())(instance, port as u32, format),
                0
            );
        }
        assert_eq!((*formats[0]).n_dimensions, 2);
        assert_eq!(
            core::slice::from_raw_parts((*formats[0]).shape, 2),
            &[352, 352]
        );
        assert_eq!(
            ((*formats[0]).rate_num, (*formats[0]).rate_denom),
            (1000, 1)
        );
        assert_eq!(
            CStr::from_ptr((*formats[0]).schema),
            c"org.adaptiveopticssim.hil-reference.shack-hartmann-frame.f32/1"
        );

        let mut slopes = vec![0.0_f32; SUBAPERTURES * 2];
        let mut flux = vec![0.0_f32; SUBAPERTURES];
        let mut validity = vec![Bool8::new(false); SUBAPERTURES];
        let mut image_chunk = core::mem::zeroed();
        let mut slopes_chunk = core::mem::zeroed();
        let mut flux_chunk = core::mem::zeroed();
        let mut validity_chunk = core::mem::zeroed();
        let mut image_data = spa_data(&mut image, &mut image_chunk);
        let mut slopes_data = spa_data(&mut slopes, &mut slopes_chunk);
        let mut flux_data = spa_data(&mut flux, &mut flux_chunk);
        let mut validity_data = spa_data(&mut validity, &mut validity_chunk);
        let mut image_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut image_data,
        };
        let mut slopes_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut slopes_data,
        };
        let mut flux_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut flux_data,
        };
        let mut validity_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut validity_data,
        };
        let disconnected = |format| abi::Buffer {
            buffer: core::ptr::null_mut(),
            format,
        };
        let inputs = [
            abi::Buffer {
                buffer: &mut image_buffer,
                format: formats[0],
            },
            disconnected(formats[4]),
            disconnected(formats[5]),
            disconnected(formats[6]),
            disconnected(formats[7]),
            disconnected(formats[8]),
        ];
        let mut outputs = [
            abi::Buffer {
                buffer: &mut slopes_buffer,
                format: formats[1],
            },
            abi::Buffer {
                buffer: &mut flux_buffer,
                format: formats[2],
            },
            abi::Buffer {
                buffer: &mut validity_buffer,
                format: formats[3],
            },
        ];
        let process = descriptor.process.unwrap();
        LANE_CALLS.store(0, Ordering::Relaxed);
        assert_eq!(
            process(
                instance,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
            ),
            0
        );
        assert_eq!(LANE_CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(slopes, native_slopes);
        assert_eq!(flux, native_flux);
        assert_eq!(
            validity,
            native_validity
                .iter()
                .copied()
                .map(Bool8::new)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            allocations(|| {
                assert_eq!(
                    process(
                        instance,
                        inputs.as_ptr(),
                        inputs.len() as u32,
                        outputs.as_mut_ptr(),
                        outputs.len() as u32,
                    ),
                    0
                );
            }),
            (0, 0)
        );

        let mut active = vec![Bool8::new(true); SUBAPERTURES];
        active[0] = Bool8::new(false);
        let mut active_chunk = core::mem::zeroed();
        let mut active_data = spa_data(&mut active, &mut active_chunk);
        let mut active_header = abi::SpaMetaHeader {
            flags: 0,
            offset: 0,
            pts: 0,
            dts_offset: 0,
            seq: 23,
        };
        let mut active_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut active_header).cast(),
        };
        let mut active_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut active_meta,
            datas: &mut active_data,
        };
        let parameter = abi::Buffer {
            buffer: &mut active_buffer,
            format: formats[8],
        };
        let mut prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_parameter.unwrap())(instance, 8, &parameter, &mut prepared),
            0
        );
        (descriptor.commit_parameter.unwrap())(instance, prepared);
        assert_eq!(
            long_property(descriptor, instance, REQUESTED_PARAMETER_SEQUENCE_ID),
            23
        );
        assert_eq!(
            allocations(|| {
                assert_eq!(
                    process(
                        instance,
                        inputs.as_ptr(),
                        inputs.len() as u32,
                        outputs.as_mut_ptr(),
                        outputs.len() as u32,
                    ),
                    0
                );
            }),
            (0, 0)
        );
        assert_eq!(&slopes[..2], &[0.0, 0.0]);
        assert_eq!(validity[0], Bool8::new(true));
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            23
        );

        (descriptor.cleanup.unwrap())(instance);
    }
}

#[test]
fn generated_shack_hartmann_rejects_missing_or_empty_image_schema() {
    let owned = AlgorithmDescriptor::new::<ShackHartmannImageF32>().unwrap();
    let descriptor = owned.as_abi();
    let invalid = [
        c"{\"subaperture_count\":1,\"coordinate_scale\":1.0,\"pixel_threshold\":0.0,\"flux_threshold\":0.0,\"image_size\":[1,1],\"subaperture_size\":[1,1],\"initial_subaperture_origins\":null,\"reference_slopes\":null,\"active\":null}",
        c"{\"subaperture_count\":1,\"image_schema\":\"\",\"coordinate_scale\":1.0,\"pixel_threshold\":0.0,\"flux_threshold\":0.0,\"image_size\":[1,1],\"subaperture_size\":[1,1],\"initial_subaperture_origins\":null,\"reference_slopes\":null,\"active\":null}",
    ];

    // SAFETY: descriptor-owned pointers and the configuration strings remain
    // live for each rejected construction call; no instance is returned.
    unsafe {
        for configuration in invalid {
            let mut instance = core::ptr::null_mut();
            assert_eq!(
                (descriptor.instantiate.unwrap())(
                    descriptor,
                    configuration.as_ptr(),
                    &TEST_EXECUTOR.0,
                    &mut instance,
                ),
                -abi::EINVAL
            );
            assert!(instance.is_null());
        }
    }
}

#[test]
fn generated_large_frame_reconstructor_wraps_direct_gemv_without_an_algorithm_plan() {
    unsafe extern "C" fn tracked_dense_f32(
        data: *mut c_void,
        task: *const abi::DenseF32Task,
    ) -> i32 {
        // SAFETY: this fixture lends its call counter through synchronous processing.
        let calls = unsafe { &*data.cast::<AtomicUsize>() };
        calls.fetch_add(1, Ordering::Relaxed);
        if cfg!(feature = "blas") {
            // A BLAS-enabled declaration must not reach the host's dense kernel.
            -abi::EINVAL
        } else {
            // SAFETY: the adapter validated the task and retains all borrowed buffers.
            unsafe { test_run_dense_f32(core::ptr::null_mut(), task) }
        }
    }

    const SUBAPERTURES: usize = 256;
    const SLOPES: usize = SUBAPERTURES * 2;
    const ACTUATORS: usize = 277;

    let mut sensor = ShackHartmannImageF32::prepare(ShackHartmannImageF32Configuration {
        subaperture_count: SUBAPERTURES,
        image_schema: "pipewireao/calibrated-pixels-f32/1".into(),
        initial_subaperture_origins: None,
        coordinate_scale: 0.5,
        pixel_threshold: 0.0,
        flux_threshold: 1.0,
        reference_slopes: None,
        active: None,
        image_size: (352, 352),
        subaperture_size: (22, 22),
    })
    .unwrap();
    let image = large_frame_image();
    let mut slopes = vec![0.0_f32; SLOPES];
    let mut flux = vec![0.0_f32; SUBAPERTURES];
    let mut validity = vec![false; SUBAPERTURES];
    ShackHartmannImageF32::process(&mut sensor, &image, &mut slopes, &mut flux, &mut validity)
        .unwrap();

    let mut matrix = vec![0.0_f32; ACTUATORS * SLOPES];
    for actuator in 0..ACTUATORS {
        matrix[actuator * SLOPES + (actuator * 37) % SLOPES] = 0.01;
        matrix[actuator * SLOPES + (actuator * 37 + 1) % SLOPES] = -0.005;
    }
    let direct = PreparedGemv::new(ACTUATORS, SLOPES, Arc::from(matrix.clone())).unwrap();
    let mut expected = vec![0.0_f32; ACTUATORS];
    direct.multiply(&mut expected, &slopes).unwrap();

    let owned = AlgorithmDescriptor::new::<ShackHartmannReconstructorF32>().unwrap();
    let descriptor = owned.as_abi();
    assert_eq!(descriptor.n_ports, 3);

    for n_lanes in [1, 4] {
        let calls = AtomicUsize::new(0);
        let executor = abi::Executor {
            struct_size: size_of::<abi::Executor>() as u32,
            version: abi::EXECUTOR_VERSION,
            flags: 0,
            n_lanes,
            data: core::ptr::from_ref(&calls).cast_mut().cast(),
            run_dense_f32: Some(tracked_dense_f32),
            run_lanes: Some(test_run_lanes),
            cache_line_size: 1,
            reserved: 0,
        };
        // SAFETY: the descriptor, executor, counter and callback buffers remain
        // alive until this instance is cleaned up exactly once.
        unsafe {
            let mut instance = core::ptr::null_mut();
            assert_eq!(
            (descriptor.instantiate.unwrap())(
                descriptor,
                c"{\"subaperture_count\":256,\"reconstructed_schema\":\"pipewireao/controller-residual-error-f32/1\",\"rate\":[1000,1],\"reconstructed_count\":277,\"initial_reconstructor\":null}"
                    .as_ptr(),
                &executor,
                &mut instance,
            ),
            0
        );
            let mut formats = [core::ptr::null(); 3];
            for (port, format) in formats.iter_mut().enumerate() {
                assert_eq!(
                    (descriptor.get_port_format.unwrap())(instance, port as u32, format),
                    0
                );
            }
            assert_eq!(
                core::slice::from_raw_parts((*formats[0]).shape, 2),
                &[256, 2]
            );
            assert_eq!(
                core::slice::from_raw_parts((*formats[2]).shape, 2),
                &[277, 512]
            );

            let mut matrix_chunk = core::mem::zeroed();
            let mut matrix_data = spa_data(&mut matrix, &mut matrix_chunk);
            let mut matrix_header = abi::SpaMetaHeader {
                flags: 0,
                offset: 0,
                pts: 0,
                dts_offset: 0,
                seq: 81,
            };
            let mut matrix_meta = abi::SpaMeta {
                type_: abi::META_HEADER,
                size: size_of::<abi::SpaMetaHeader>() as u32,
                data: core::ptr::from_mut(&mut matrix_header).cast(),
            };
            let mut matrix_buffer = abi::SpaBuffer {
                n_metas: 1,
                n_datas: 1,
                metas: &mut matrix_meta,
                datas: &mut matrix_data,
            };
            let parameter = abi::Buffer {
                buffer: &mut matrix_buffer,
                format: formats[2],
            };
            let mut prepared = core::ptr::null_mut();
            assert_eq!(
                (descriptor.prepare_parameter.unwrap())(instance, 2, &parameter, &mut prepared),
                0
            );
            (descriptor.commit_parameter.unwrap())(instance, prepared);

            let mut actual = vec![0.0_f32; ACTUATORS];
            let mut slopes_chunk = core::mem::zeroed();
            let mut actual_chunk = core::mem::zeroed();
            let mut slopes_data = spa_data(&mut slopes, &mut slopes_chunk);
            let mut actual_data = spa_data(&mut actual, &mut actual_chunk);
            let mut slopes_buffer = abi::SpaBuffer {
                n_metas: 0,
                n_datas: 1,
                metas: core::ptr::null_mut(),
                datas: &mut slopes_data,
            };
            let mut actual_buffer = abi::SpaBuffer {
                n_metas: 0,
                n_datas: 1,
                metas: core::ptr::null_mut(),
                datas: &mut actual_data,
            };
            let inputs = [
                abi::Buffer {
                    buffer: &mut slopes_buffer,
                    format: formats[0],
                },
                abi::Buffer {
                    buffer: core::ptr::null_mut(),
                    format: formats[2],
                },
            ];
            let mut outputs = [abi::Buffer {
                buffer: &mut actual_buffer,
                format: formats[1],
            }];
            let process = descriptor.process.unwrap();
            assert_eq!(
                allocations(|| {
                    assert_eq!(
                        process(
                            instance,
                            inputs.as_ptr(),
                            inputs.len() as u32,
                            outputs.as_mut_ptr(),
                            outputs.len() as u32,
                        ),
                        0
                    );
                }),
                (0, 0)
            );
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
                "dense result must match the selected direct backend bitwise at {n_lanes} lanes"
            );
            assert_eq!(
                calls.load(Ordering::Relaxed),
                usize::from(!cfg!(feature = "blas")),
                "BLAS owns dense execution; the scalar profile retains host dispatch"
            );
            assert_eq!(
                long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
                81
            );

            (descriptor.cleanup.unwrap())(instance);
        }
    }
}

#[test]
fn generated_closed_loop_node_preserves_one_atomic_plan_and_state_owner() {
    const STATE_LENGTH: usize = 3;
    const HIDDEN_MODE_COUNT: usize = 2;
    let shape_to_hidden_values = [1.0_f32, -0.5, 0.25, 0.0, 0.75, -1.0];
    let hidden_to_shape_values = [0.5_f32, -0.25, 0.75, 0.0, -0.5, 1.0];

    let owned = AlgorithmDescriptor::new::<ClosedLoopCorrectionF32>().unwrap();
    let descriptor = owned.as_abi();
    assert_eq!(descriptor.n_ports, 6);

    // SAFETY: the descriptor owns every C string and port entry. Each local
    // buffer and metadata object remains alive until cleanup, and prepared
    // publication tokens are consumed exactly once.
    unsafe {
        assert_eq!(
            CStr::from_ptr(descriptor.name),
            c"closed-loop-correction-f32"
        );
        let ports = core::slice::from_raw_parts(descriptor.ports, descriptor.n_ports as usize);
        assert_eq!(CStr::from_ptr(ports[0].name), c"residual-error");
        assert_eq!(CStr::from_ptr(ports[1].name), c"constraint-feedback");
        assert_eq!(CStr::from_ptr(ports[2].name), c"correction");
        assert_eq!(CStr::from_ptr(ports[3].name), c"controller-state");
        assert_eq!(CStr::from_ptr(ports[4].name), c"shape-to-hidden");
        assert_eq!(CStr::from_ptr(ports[5].name), c"hidden-to-shape");
        assert_eq!(ports[4].flags, abi::PORT_OPTIONAL | abi::PORT_PARAMETER);
        assert_eq!(ports[5].flags, abi::PORT_OPTIONAL | abi::PORT_PARAMETER);

        let mut instance = core::ptr::null_mut();
        assert_eq!(
            (descriptor.instantiate.unwrap())(
                descriptor,
                c"{\"state_length\":3,\"hidden_mode_count\":2,\"rate\":[1000,1]}".as_ptr(),
                &TEST_EXECUTOR.0,
                &mut instance,
            ),
            0
        );
        assert!(!instance.is_null());

        let mut formats = [core::ptr::null(); 6];
        for (port, format) in formats.iter_mut().enumerate() {
            assert_eq!(
                (descriptor.get_port_format.unwrap())(instance, port as u32, format),
                0
            );
        }
        assert_eq!(
            core::slice::from_raw_parts((*formats[4]).shape, 2),
            &[HIDDEN_MODE_COUNT as u32, STATE_LENGTH as u32]
        );
        assert_eq!(
            core::slice::from_raw_parts((*formats[5]).shape, 2),
            &[STATE_LENGTH as u32, HIDDEN_MODE_COUNT as u32]
        );
        for format in formats {
            assert_eq!(((*format).rate_num, (*format).rate_denom), (1000, 1));
        }

        let mut info = core::mem::zeroed::<abi::PropertyInfo>();
        for (index, name) in [c"gain", c"pole", c"anti-windup-gain", c"hidden-mode-gain"]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                (descriptor.enum_prop_info.unwrap())(instance, index as u32, &mut info),
                1
            );
            assert_eq!(CStr::from_ptr(info.name), name);
        }

        let mut residual_error = [0.5_f32, -1.0, 0.25];
        let mut constraint_feedback = [99.0_f32, 99.0, 99.0];
        let mut correction = [0.0_f32; STATE_LENGTH];
        let mut controller_state = [0.0_f32; STATE_LENGTH];
        let mut residual_chunk = core::mem::zeroed();
        let mut feedback_chunk = core::mem::zeroed();
        let mut correction_chunk = core::mem::zeroed();
        let mut state_chunk = core::mem::zeroed();
        let mut residual_data = spa_data(&mut residual_error, &mut residual_chunk);
        let mut feedback_data = spa_data(&mut constraint_feedback, &mut feedback_chunk);
        let mut correction_data = spa_data(&mut correction, &mut correction_chunk);
        let mut state_data = spa_data(&mut controller_state, &mut state_chunk);
        let mut residual_header = abi::SpaMetaHeader {
            flags: 3,
            offset: 2,
            pts: 100,
            dts_offset: -1,
            seq: 20,
        };
        let mut correction_header = core::mem::zeroed::<abi::SpaMetaHeader>();
        let mut state_header = core::mem::zeroed::<abi::SpaMetaHeader>();
        let mut residual_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut residual_header).cast(),
        };
        let mut correction_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut correction_header).cast(),
        };
        let mut state_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut state_header).cast(),
        };
        let mut residual_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut residual_meta,
            datas: &mut residual_data,
        };
        let mut feedback_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut feedback_data,
        };
        let mut correction_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut correction_meta,
            datas: &mut correction_data,
        };
        let mut state_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut state_meta,
            datas: &mut state_data,
        };
        let inputs = [
            abi::Buffer {
                buffer: &mut residual_buffer,
                format: formats[0],
            },
            abi::Buffer {
                buffer: &mut feedback_buffer,
                format: formats[1],
            },
            abi::Buffer {
                buffer: core::ptr::null_mut(),
                format: formats[4],
            },
            abi::Buffer {
                buffer: core::ptr::null_mut(),
                format: formats[5],
            },
        ];
        let mut outputs = [
            abi::Buffer {
                buffer: &mut correction_buffer,
                format: formats[2],
            },
            abi::Buffer {
                buffer: &mut state_buffer,
                format: formats[3],
            },
        ];

        let mut direct = ClosedLoopCorrectionPlan::with_hidden_modes(
            1.0,
            0.0,
            0.0,
            STATE_LENGTH,
            HIDDEN_MODE_COUNT,
            0.0,
            Arc::from([0.0_f32; STATE_LENGTH * HIDDEN_MODE_COUNT]),
            Arc::from([0.0_f32; STATE_LENGTH * HIDDEN_MODE_COUNT]),
        )
        .unwrap();
        let mut direct_workspace = direct.workspace();
        let mut expected_correction = [0.0_f32; STATE_LENGTH];

        let mut shape_to_hidden = shape_to_hidden_values;
        let mut shape_chunk = core::mem::zeroed();
        let mut shape_data = spa_data(&mut shape_to_hidden, &mut shape_chunk);
        let mut shape_header = abi::SpaMetaHeader {
            flags: 0,
            offset: 0,
            pts: 0,
            dts_offset: 0,
            seq: 11,
        };
        let mut shape_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut shape_header).cast(),
        };
        let mut shape_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut shape_meta,
            datas: &mut shape_data,
        };
        let shape_parameter = abi::Buffer {
            buffer: &mut shape_buffer,
            format: formats[4],
        };
        let mut prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_parameter.unwrap())(instance, 4, &shape_parameter, &mut prepared),
            0
        );
        (descriptor.commit_parameter.unwrap())(instance, prepared);
        direct = direct
            .with_shape_to_hidden(Arc::from(shape_to_hidden_values))
            .unwrap();

        let one_property = [float_property(0, 0.5)];
        prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_props.unwrap())(
                instance,
                one_property.as_ptr(),
                one_property.len() as u32,
                &mut prepared,
            ),
            -abi::EBUSY
        );
        assert!(prepared.is_null());

        direct
            .process(
                &mut expected_correction,
                &mut direct_workspace,
                ClosedLoopCorrectionInput {
                    x: &residual_error,
                    constraint_feedback: None,
                },
            )
            .unwrap();
        assert_eq!(
            (descriptor.process.unwrap())(
                instance,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
            ),
            0
        );
        assert_eq!(correction, expected_correction);
        assert_eq!(controller_state, direct_workspace.state());
        assert_eq!(correction_header.seq, residual_header.seq);
        assert_eq!(state_header.seq, residual_header.seq);
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            11
        );

        let mut hidden_to_shape = hidden_to_shape_values;
        let mut hidden_chunk = core::mem::zeroed();
        let mut hidden_data = spa_data(&mut hidden_to_shape, &mut hidden_chunk);
        let mut hidden_header = abi::SpaMetaHeader {
            flags: 0,
            offset: 0,
            pts: 0,
            dts_offset: 0,
            seq: 12,
        };
        let mut hidden_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut hidden_header).cast(),
        };
        let mut hidden_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut hidden_meta,
            datas: &mut hidden_data,
        };
        let hidden_parameter = abi::Buffer {
            buffer: &mut hidden_buffer,
            format: formats[5],
        };
        prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_parameter.unwrap())(instance, 5, &hidden_parameter, &mut prepared),
            0
        );
        (descriptor.commit_parameter.unwrap())(instance, prepared);
        direct = direct
            .with_hidden_to_shape(Arc::from(hidden_to_shape_values))
            .unwrap();
        residual_error.copy_from_slice(&[0.1, 0.2, -0.3]);
        constraint_feedback.copy_from_slice(&[0.05, -0.1, 0.0]);
        direct
            .process(
                &mut expected_correction,
                &mut direct_workspace,
                ClosedLoopCorrectionInput {
                    x: &residual_error,
                    constraint_feedback: Some(&constraint_feedback),
                },
            )
            .unwrap();
        assert_eq!(
            (descriptor.process.unwrap())(
                instance,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
            ),
            0
        );
        assert_eq!(correction, expected_correction);
        assert_eq!(controller_state, direct_workspace.state());
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            12
        );

        let properties = [
            float_property(0, 0.5),
            float_property(1, 0.75),
            float_property(2, 0.8),
            float_property(3, 0.25),
        ];
        prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_props.unwrap())(
                instance,
                properties.as_ptr(),
                properties.len() as u32,
                &mut prepared,
            ),
            0
        );
        (descriptor.commit_props.unwrap())(instance, prepared);
        direct = direct
            .prepare_properties(&[
                PropertyAssignment {
                    id: 0,
                    value: PropertyValue::Float(0.5),
                },
                PropertyAssignment {
                    id: 1,
                    value: PropertyValue::Float(0.75),
                },
                PropertyAssignment {
                    id: 2,
                    value: PropertyValue::Float(0.8),
                },
                PropertyAssignment {
                    id: 3,
                    value: PropertyValue::Float(0.25),
                },
            ])
            .unwrap();
        residual_error.copy_from_slice(&[-0.4, 0.6, 0.8]);
        constraint_feedback.copy_from_slice(&[0.0, 0.2, -0.25]);
        direct
            .process(
                &mut expected_correction,
                &mut direct_workspace,
                ClosedLoopCorrectionInput {
                    x: &residual_error,
                    constraint_feedback: Some(&constraint_feedback),
                },
            )
            .unwrap();
        assert_eq!(
            allocations(|| {
                assert_eq!(
                    (descriptor.process.unwrap())(
                        instance,
                        inputs.as_ptr(),
                        inputs.len() as u32,
                        outputs.as_mut_ptr(),
                        outputs.len() as u32,
                    ),
                    0
                );
            }),
            (0, 0)
        );
        assert_eq!(correction, expected_correction);
        assert_eq!(controller_state, direct_workspace.state());
        assert_eq!(long_property(descriptor, instance, ACTIVE_GENERATION_ID), 1);
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            12
        );

        assert_eq!(
            allocations(|| assert_eq!((descriptor.reset.unwrap())(instance), 0)),
            (0, 0)
        );
        (descriptor.cleanup.unwrap())(instance);
    }
}

#[test]
fn pdm_command_demanded_schema_follows_graph_configuration() {
    use std::ffi::CString;

    let owned = AlgorithmDescriptor::new::<PdmCommandF32>().unwrap();
    let descriptor = owned.as_abi();
    for (configured, expected) in [
        (None, Some(c"pipewireao/dm-command-opd-um-f32/1")),
        (
            Some("pipewireao/demanded-pdm-command-f32/1"),
            Some(c"pipewireao/demanded-pdm-command-f32/1"),
        ),
        (
            Some("pipewireao/dm-command-opd-um-f32/1"),
            Some(c"pipewireao/dm-command-opd-um-f32/1"),
        ),
        (
            Some("pipewireao/actuator-command-opd-m-f32/1"),
            Some(c"pipewireao/actuator-command-opd-m-f32/1"),
        ),
        (Some("pipewireao/requested-pdm-command-f32/1"), None),
        (Some(""), None),
    ] {
        let demanded_schema = configured
            .map(|schema| format!(r#","demanded_schema":"{schema}""#))
            .unwrap_or_default();
        let config = CString::new(format!(
            r#"{{"rate":[1000,1],"actuator_count":2,"command_limits":[-1,1],"slew_limit":null,"initial_current":null{demanded_schema}}}"#
        ))
        .unwrap();
        let mut instance = core::ptr::null_mut();
        // SAFETY: the descriptor, config, and returned instance remain live
        // for each callback. Each successful construction is cleaned up once.
        unsafe {
            let status = (descriptor.instantiate.unwrap())(
                descriptor,
                config.as_ptr(),
                &TEST_EXECUTOR.0,
                &mut instance,
            );
            if let Some(schema) = expected {
                assert_eq!(status, 0, "{configured:?}");
                let mut format = core::ptr::null();
                assert_eq!(
                    (descriptor.get_port_format.unwrap())(instance, 1, &mut format),
                    0,
                );
                assert_eq!(CStr::from_ptr((*format).schema), schema);
                (descriptor.cleanup.unwrap())(instance);
            } else {
                assert_eq!(status, -abi::EINVAL, "{configured:?}");
                assert!(instance.is_null());
            }
        }
    }
}

#[test]
fn pdm_completion_and_command_preserve_optional_sample_metadata() {
    let completion = AlgorithmDescriptor::new::<VdmToPdmF32>().unwrap();
    let command = AlgorithmDescriptor::new::<PdmCommandF32>().unwrap();
    for (descriptor, config, second_output_bytes) in [
        (
            completion.as_abi(),
            c"{\"rate\":[1000,1],\"active_vdm_count\":2,\"full_vdm_count\":2,\"physical_actuator_count\":2}",
            2,
        ),
        (
            command.as_abi(),
            c"{\"rate\":[1000,1],\"actuator_count\":2,\"command_limits\":[-1,1],\"slew_limit\":null,\"initial_current\":null,\"demanded_schema\":\"pipewireao/demanded-pdm-command-f32/1\"}",
            8,
        ),
    ] {
        // SAFETY: both descriptors own their port and metadata arrays. Each
        // live instance and all its nested ABI buffers outlive their callback.
        unsafe {
            for index in 0..3 {
                let port = &*descriptor.ports.add(index);
                assert_eq!(
                    port.n_metas,
                    4,
                    "{:?} port {index}",
                    CStr::from_ptr(descriptor.name)
                );
                let records = core::slice::from_raw_parts(port.metas, port.n_metas as usize);
                for (type_, size) in [
                    (abi::META_OPTICAL_CHOPPING, 8),
                    (abi::META_ELECTRICAL_TRIGGER, 8),
                    (abi::META_ACQUISITION_IDENTITY, 40),
                    (abi::META_EXPOSURE_TIMING, 24),
                ] {
                    let record = records.iter().find(|record| record.type_ == type_).unwrap();
                    assert_eq!((record.size, record.flags), (size, 0));
                }
                if index != 0 {
                    assert_eq!(port.propagation, abi::META_PROPAGATION_PRESERVE);
                }
            }

            let mut instance = core::ptr::null_mut();
            assert_eq!(
                (descriptor.instantiate.unwrap())(
                    descriptor,
                    config.as_ptr(),
                    &TEST_EXECUTOR.0,
                    &mut instance,
                ),
                0
            );
            let mut formats = [core::ptr::null(); 5];
            for (port, format) in formats[..descriptor.n_ports as usize].iter_mut().enumerate() {
                assert_eq!(
                    (descriptor.get_port_format.unwrap())(instance, port as u32, format),
                    0
                );
            }

            let mut optical = abi::SpaMetaOpticalChopping {
                flags: 1,
                classification: 2,
            };
            let mut output_optical = [abi::SpaMetaOpticalChopping::default(); 2];
            let [first, second] = &mut output_optical;
            let mut metadata = [&mut optical, first, second].map(|value| abi::SpaMeta {
                type_: abi::META_OPTICAL_CHOPPING,
                size: size_of::<abi::SpaMetaOpticalChopping>() as u32,
                data: core::ptr::from_mut(value).cast(),
            });
            let mut values = [[0.5_f32, -2.0], [0.0; 2], [0.0; 2]];
            let mut chunks = core::array::from_fn::<_, 3, _>(|_| abi::SpaChunk {
                offset: 0,
                size: 0,
                stride: 0,
                flags: 0,
            });
            let mut data = core::array::from_fn::<_, 3, _>(|i| spa_data(&mut values[i], &mut chunks[i]));
            let mut buffers = core::array::from_fn::<_, 3, _>(|i| abi::SpaBuffer {
                n_metas: 1,
                n_datas: 1,
                metas: &raw mut metadata[i],
                datas: &raw mut data[i],
            });
            let mut inputs = [abi::Buffer {
                buffer: core::ptr::null_mut(),
                format: core::ptr::null(),
            }; 3];
            inputs[0] = abi::Buffer {
                buffer: &raw mut buffers[0],
                format: formats[0],
            };
            inputs[1].format = formats[3];
            inputs[2].format = formats[4];
            let mut outputs = [1, 2].map(|i| abi::Buffer {
                buffer: &raw mut buffers[i],
                format: formats[i],
            });

            for has_optical in [true, false] {
                (*inputs[0].buffer).n_metas = u32::from(has_optical);
                assert_eq!(
                    (descriptor.process.unwrap())(
                        instance,
                        inputs.as_ptr(),
                        descriptor.n_ports - 2,
                        outputs.as_mut_ptr(),
                        outputs.len() as u32,
                    ),
                    0
                );
                assert_eq!(
                    ((*data[1].chunk).size, (*data[2].chunk).size),
                    (8, second_output_bytes)
                );
                let expected = if has_optical { (1, 2) } else { (0, 0) };
                for meta in &metadata[1..] {
                    let record = &*meta.data.cast::<abi::SpaMetaOpticalChopping>();
                    assert_eq!((record.flags, record.classification), expected);
                }
            }
            (descriptor.cleanup.unwrap())(instance);
        }
    }
}

#[test]
fn generated_vdm_projection_preserves_prior_parameter_and_realtime_boundary() {
    let owned = AlgorithmDescriptor::new::<VdmToPdmF32>().unwrap();
    let descriptor = owned.as_abi();
    assert_eq!(descriptor.n_ports, 5);

    // SAFETY: the descriptor owns its strings and port table. Every instance,
    // format, buffer, and prepared token remains live for the callback that
    // consumes it, and cleanup runs exactly once.
    unsafe {
        assert_eq!(CStr::from_ptr(descriptor.name), c"vdm-to-pdm-f32");
        let mut instance = core::ptr::null_mut();
        assert_eq!(
            (descriptor.instantiate.unwrap())(
                descriptor,
                c"{\"rate\":[1000,1],\"active_vdm_count\":2,\"full_vdm_count\":2,\"physical_actuator_count\":2}"
                    .as_ptr(),
                &TEST_EXECUTOR.0,
                &mut instance,
            ),
            0
        );

        let mut formats = [core::ptr::null(); 5];
        for (port, format) in formats.iter_mut().enumerate() {
            assert_eq!(
                (descriptor.get_port_format.unwrap())(instance, port as u32, format),
                0
            );
        }
        assert_eq!(
            CStr::from_ptr((*formats[0]).schema),
            c"pipewireao/vdm-command-f32/1"
        );
        assert_eq!(
            CStr::from_ptr((*formats[3]).schema),
            c"pipewireao/active-to-full-vdm-projection-f32/1"
        );
        assert_eq!(core::slice::from_raw_parts((*formats[3]).shape, 2), &[2, 2]);

        let mut input = [0.5_f32, -0.25];
        let mut output = [0.0_f32; 2];
        let mut input_chunk = core::mem::zeroed();
        let mut output_chunk = core::mem::zeroed();
        let mut input_data = spa_data(&mut input, &mut input_chunk);
        let mut output_data = spa_data(&mut output, &mut output_chunk);
        let mut input_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut input_data,
        };
        let mut output_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut output_data,
        };
        let inputs = [
            abi::Buffer {
                buffer: &mut input_buffer,
                format: formats[0],
            },
            abi::Buffer {
                buffer: core::ptr::null_mut(),
                format: formats[3],
            },
            abi::Buffer {
                buffer: core::ptr::null_mut(),
                format: formats[4],
            },
        ];
        let mut roles = [3u8; 2];
        let mut role_chunk = core::mem::zeroed();
        let mut role_data = spa_data(&mut roles, &mut role_chunk);
        let mut role_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut role_data,
        };
        let mut outputs = [
            abi::Buffer {
                buffer: &mut output_buffer,
                format: formats[1],
            },
            abi::Buffer {
                buffer: &mut role_buffer,
                format: formats[2],
            },
        ];
        let process = descriptor.process.unwrap();

        // The batch envelope rejects invalid arguments before reading the
        // parameter arrays or changing the requested plan generation.
        let prepare_batch = descriptor.prepare_parameters.unwrap();
        let empty = abi::Parameter {
            port: 3,
            reserved: 0,
            buffer: inputs[1],
        };
        let mut token = core::ptr::null_mut();
        let oversize = abi::MAX_PARAMETER_TRANSACTION_ASSIGNMENTS as u32 + 1;
        for (owner, params, count, expected) in [
            (core::ptr::null_mut(), &raw const empty, 1, -abi::EINVAL),
            (instance, core::ptr::null(), 1, -abi::EINVAL),
            (instance, &raw const empty, 0, -abi::EINVAL),
            (instance, &raw const empty, oversize, -abi::E2BIG),
        ] {
            assert_eq!(prepare_batch(owner, params, count, &mut token), expected);
            assert!(token.is_null());
        }
        assert_eq!(
            prepare_batch(instance, &empty, 1, core::ptr::null_mut()),
            -abi::EINVAL
        );
        let wrong_port = abi::Parameter { port: 0, ..empty };
        let reserved = abi::Parameter {
            reserved: 1,
            ..empty
        };
        let wrong_format = abi::Parameter {
            buffer: inputs[0],
            ..empty
        };
        let unknown = abi::Parameter { port: 33, ..empty };
        for parameter in [wrong_port, reserved, wrong_format, unknown] {
            assert_eq!(
                prepare_batch(instance, &parameter, 1, &mut token),
                -abi::EINVAL
            );
            assert!(token.is_null());
        }
        assert_eq!(
            long_property(descriptor, instance, REQUESTED_PARAMETER_SEQUENCE_ID),
            0
        );

        let mut identity = [1.0_f32, 0.0, 0.0, 1.0];
        let mut parameter_chunk = core::mem::zeroed();
        let mut parameter_data = spa_data(&mut identity, &mut parameter_chunk);
        let mut parameter_header = abi::SpaMetaHeader {
            flags: 0,
            offset: 0,
            pts: 0,
            dts_offset: 0,
            seq: 21,
        };
        let mut parameter_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut parameter_header).cast(),
        };
        let mut parameter_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut parameter_meta,
            datas: &mut parameter_data,
        };
        let mut parameter = abi::Buffer {
            buffer: &mut parameter_buffer,
            format: formats[3],
        };
        let mut prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_parameter.unwrap())(instance, 3, &parameter, &mut prepared),
            0
        );
        (descriptor.commit_parameter.unwrap())(instance, prepared);
        assert_eq!(
            process(
                instance,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
            ),
            0
        );
        assert_eq!(output, [0.0, 0.0]);

        parameter_header.seq = 22;
        assert_eq!(parameter_header.seq, 22);
        parameter.format = formats[4];
        prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_parameter.unwrap())(instance, 4, &parameter, &mut prepared),
            0
        );
        (descriptor.commit_parameter.unwrap())(instance, prepared);
        assert_eq!(
            allocations(|| {
                assert_eq!(
                    process(
                        instance,
                        inputs.as_ptr(),
                        inputs.len() as u32,
                        outputs.as_mut_ptr(),
                        outputs.len() as u32,
                    ),
                    0
                );
            }),
            (0, 0)
        );
        assert_eq!(output, input);
        assert_eq!(roles, [1, 1]);
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            22
        );

        input.copy_from_slice(&[-0.75, 0.25]);
        assert_eq!(
            allocations(|| {
                assert_eq!(
                    process(
                        instance,
                        inputs.as_ptr(),
                        inputs.len() as u32,
                        outputs.as_mut_ptr(),
                        outputs.len() as u32,
                    ),
                    0
                );
            }),
            (0, 0)
        );
        assert_eq!(output, input);
        assert_eq!(roles, [1, 1]);

        (descriptor.cleanup.unwrap())(instance);
    }
}

#[test]
fn generated_incremental_reconstructor_defers_output_and_projects_sample_metadata() {
    const BLOCKS: usize = 6;
    const CAPACITY: usize = 2;
    const ACTUATORS: usize = 3;

    let matrix = (0..ACTUATORS * BLOCKS * CAPACITY)
        .map(|index| (index as f32 + 1.0) * 0.01)
        .collect::<Vec<_>>();
    let measurement_values = (0..BLOCKS)
        .flat_map(|block| {
            (0..CAPACITY).map(move |column| 1.0 + 0.2 * block as f32 + 0.1 * column as f32)
        })
        .collect::<Vec<_>>();

    let configuration = IncrementalDenseReconstructorF32Configuration {
        measurement_count: BLOCKS * CAPACITY,
        measurement_block_capacity: CAPACITY,
        blocks_per_sample: BLOCKS,
        reconstructed_count: ACTUATORS,
        initial_reconstructor: Some(matrix.clone()),
        measurements_schema: "test.filtergraph/measurements/1".into(),
        column_indices_schema: "test.filtergraph/measurement-columns/1".into(),
        measurement_count_schema: "test.filtergraph/measurement-count/1".into(),
        reconstructed_schema: "pipewireao/controller-residual-error-f32/1".into(),
        reconstructor_schema: "test.filtergraph/reconstructor/1".into(),
    };
    let mut native = IncrementalDenseReconstructorF32::prepare(configuration).unwrap();
    let mut expected = vec![0.0_f32; ACTUATORS];
    for block in 0..BLOCKS {
        IncrementalDenseReconstructorF32::process_with_metadata(
            &mut native,
            &[
                Some(
                    SampleMetadata::new(91, block as u32)
                        .with_timestamp_nanoseconds((block + 1 == BLOCKS).then_some(456_789))
                        .with_terminal(block + 1 == BLOCKS),
                ),
                None,
                None,
            ],
            &measurement_values[block * CAPACITY..(block + 1) * CAPACITY],
            &[block as u32 * 2, block as u32 * 2 + 1],
            &[CAPACITY as u32],
            &mut expected,
        )
        .unwrap();
    }

    let config = std::ffi::CString::new(
        serde_json::json!({
            "measurement_count": BLOCKS * CAPACITY,
            "measurement_block_capacity": CAPACITY,
            "blocks_per_sample": BLOCKS,
            "reconstructed_count": ACTUATORS,
            "initial_reconstructor": matrix,
            "measurements_schema": "test.filtergraph/measurements/1",
            "column_indices_schema": "test.filtergraph/measurement-columns/1",
            "measurement_count_schema": "test.filtergraph/measurement-count/1",
            "reconstructed_schema": "pipewireao/controller-residual-error-f32/1",
            "reconstructor_schema": "test.filtergraph/reconstructor/1",
            "rate": [10, 1],
        })
        .to_string(),
    )
    .unwrap();
    let owned = AlgorithmDescriptor::new::<IncrementalDenseReconstructorF32>().unwrap();
    let descriptor = owned.as_abi();
    assert_eq!(descriptor.n_ports, 5);
    // SAFETY: the descriptor owns its immutable port table.
    let ports =
        unsafe { core::slice::from_raw_parts(descriptor.ports, descriptor.n_ports as usize) };
    assert_eq!(ports[3].flags, abi::PORT_CONDITIONAL);

    // SAFETY: all callback buffers and descriptor-owned pointers remain live,
    // and cleanup consumes the instance once after the final callback.
    unsafe {
        let mut instance = core::ptr::null_mut();
        assert_eq!(
            (descriptor.instantiate.unwrap())(
                descriptor,
                config.as_ptr(),
                &TEST_EXECUTOR.0,
                &mut instance,
            ),
            0
        );
        let mut formats = [core::ptr::null(); 5];
        for (port, format) in formats.iter_mut().enumerate() {
            assert_eq!(
                (descriptor.get_port_format.unwrap())(instance, port as u32, format),
                0
            );
        }
        assert_eq!(((*formats[0]).rate_num, (*formats[0]).rate_denom), (60, 1));
        assert_eq!(((*formats[3]).rate_num, (*formats[3]).rate_denom), (10, 1));

        let mut measurements = [0.0_f32; CAPACITY];
        let mut output = [-7.0_f32; ACTUATORS];
        let mut input_chunk = core::mem::zeroed();
        let mut output_chunk = core::mem::zeroed();
        let mut columns = [0_u32, 1_u32];
        let mut count = [CAPACITY as u32];
        let mut columns_chunk = core::mem::zeroed();
        let mut count_chunk = core::mem::zeroed();
        let mut columns_data = spa_data(&mut columns, &mut columns_chunk);
        let mut count_data = spa_data(&mut count, &mut count_chunk);
        let mut columns_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut columns_data,
        };
        let mut count_buffer = abi::SpaBuffer {
            n_metas: 0,
            n_datas: 1,
            metas: core::ptr::null_mut(),
            datas: &mut count_data,
        };
        let mut input_data = spa_data(&mut measurements, &mut input_chunk);
        let mut output_data = spa_data(&mut output, &mut output_chunk);
        let mut input_header = core::mem::zeroed::<abi::SpaMetaHeader>();
        let mut output_header = core::mem::zeroed::<abi::SpaMetaHeader>();
        let mut input_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut input_header).cast(),
        };
        let mut output_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut output_header).cast(),
        };
        let mut input_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut input_meta,
            datas: &mut input_data,
        };
        let mut output_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut output_meta,
            datas: &mut output_data,
        };
        let inputs = [
            abi::Buffer {
                buffer: &mut input_buffer,
                format: formats[0],
            },
            abi::Buffer {
                buffer: &mut columns_buffer,
                format: formats[1],
            },
            abi::Buffer {
                buffer: &mut count_buffer,
                format: formats[2],
            },
            abi::Buffer {
                buffer: core::ptr::null_mut(),
                format: formats[4],
            },
        ];
        let mut outputs = [abi::Buffer {
            buffer: &mut output_buffer,
            format: formats[3],
        }];
        let process = descriptor.process.unwrap();
        let lane_calls_before = LANE_CALLS.load(Ordering::Relaxed);

        for block in 0..BLOCKS {
            measurements
                .copy_from_slice(&measurement_values[block * CAPACITY..(block + 1) * CAPACITY]);
            columns.copy_from_slice(&[block as u32 * 2, block as u32 * 2 + 1]);
            *input_meta.data.cast::<abi::SpaMetaHeader>() = abi::SpaMetaHeader {
                flags: if block + 1 == BLOCKS {
                    abi::META_HEADER_MARKER
                } else {
                    0
                },
                offset: block as u32,
                pts: if block + 1 == BLOCKS {
                    456_789
                } else {
                    abi::TIME_INVALID
                },
                dts_offset: 0,
                seq: 91,
            };
            (*input_data.chunk).size = size_of_val(&measurements) as u32;
            assert_eq!(
                allocations(|| {
                    assert_eq!(
                        process(
                            instance,
                            inputs.as_ptr(),
                            inputs.len() as u32,
                            outputs.as_mut_ptr(),
                            1
                        ),
                        0
                    );
                }),
                (0, 0)
            );
            if block + 1 == BLOCKS {
                assert_eq!(output_chunk.size as usize, size_of_val(&output));
            } else {
                assert_eq!(output_chunk.size, 0);
                assert_eq!(output, [-7.0; ACTUATORS]);
            }
        }
        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(LANE_CALLS.load(Ordering::Relaxed) > lane_calls_before);
        assert_eq!(output_header.seq, 91);
        assert_eq!(output_header.offset, 0);
        assert_eq!(output_header.pts, 456_789);
        assert_eq!(output_header.flags, abi::META_HEADER_MARKER);

        measurements.copy_from_slice(&measurement_values[..CAPACITY]);
        columns.copy_from_slice(&[0, 1]);
        *input_meta.data.cast::<abi::SpaMetaHeader>() = abi::SpaMetaHeader {
            flags: 0,
            offset: 0,
            pts: abi::TIME_INVALID,
            dts_offset: 0,
            seq: 92,
        };
        (*input_data.chunk).size = size_of_val(&measurements) as u32;
        assert_eq!(
            process(
                instance,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                1,
            ),
            0
        );
        assert_eq!(output_chunk.size, 0);

        // A direct callback caller can omit Header. Unlike host preflight,
        // this enters the specialized callback and must abandon the partial frame.
        input_buffer.n_metas = 0;
        assert_eq!(input_buffer.n_metas, 0);
        assert_eq!(
            allocations(|| assert_eq!(
                process(
                    instance,
                    inputs.as_ptr(),
                    inputs.len() as u32,
                    outputs.as_mut_ptr(),
                    1
                ),
                -abi::EINVAL
            )),
            (0, 0)
        );
        input_buffer.n_metas = 1;
        assert_eq!(input_buffer.n_metas, 1);
        for block in 1..BLOCKS {
            measurements
                .copy_from_slice(&measurement_values[block * CAPACITY..(block + 1) * CAPACITY]);
            columns.copy_from_slice(&[block as u32 * 2, block as u32 * 2 + 1]);
            input_header.offset = block as u32;
            input_header.flags = if block + 1 == BLOCKS {
                abi::META_HEADER_MARKER
            } else {
                0
            };
            assert_eq!(input_header.offset, block as u32);
            assert_eq!(
                process(
                    instance,
                    inputs.as_ptr(),
                    inputs.len() as u32,
                    outputs.as_mut_ptr(),
                    1
                ),
                0
            );
            assert_eq!(output_chunk.size, 0);
        }
        measurements.copy_from_slice(&measurement_values[..CAPACITY]);
        columns.copy_from_slice(&[0, 1]);
        input_header.offset = 0;
        input_header.flags = 0;
        assert_eq!(input_header.offset, 0);
        assert_eq!(
            process(
                instance,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                1
            ),
            0
        );
        assert_eq!(output_chunk.size, 0);

        let mut replacement = vec![0.0_f32; ACTUATORS * BLOCKS * CAPACITY];
        let mut parameter_chunk = core::mem::zeroed();
        let mut parameter_data = spa_data(&mut replacement, &mut parameter_chunk);
        let mut parameter_header = abi::SpaMetaHeader {
            flags: 0,
            offset: 0,
            pts: abi::TIME_INVALID,
            dts_offset: 0,
            seq: 5,
        };
        let mut parameter_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut parameter_header).cast(),
        };
        let mut parameter_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut parameter_meta,
            datas: &mut parameter_data,
        };
        let parameter = abi::Buffer {
            buffer: &mut parameter_buffer,
            format: formats[4],
        };
        let mut prepared = core::ptr::null_mut();
        assert_eq!(
            (descriptor.prepare_parameter.unwrap())(instance, 4, &parameter, &mut prepared),
            0
        );
        (descriptor.commit_parameter.unwrap())(instance, prepared);

        measurements.copy_from_slice(&measurement_values[CAPACITY..2 * CAPACITY]);
        columns.copy_from_slice(&[2, 3]);
        *input_meta.data.cast::<abi::SpaMetaHeader>() = abi::SpaMetaHeader {
            flags: 0,
            offset: 1,
            pts: abi::TIME_INVALID,
            dts_offset: 0,
            seq: 92,
        };
        assert_eq!(
            process(
                instance,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_mut_ptr(),
                1,
            ),
            0
        );
        assert_eq!(output_chunk.size, 0);
        assert_eq!(
            long_property(descriptor, instance, ACTIVE_PARAMETER_SEQUENCE_ID),
            5
        );

        for block in 0..BLOCKS {
            measurements
                .copy_from_slice(&measurement_values[block * CAPACITY..(block + 1) * CAPACITY]);
            columns.copy_from_slice(&[block as u32 * 2, block as u32 * 2 + 1]);
            *input_meta.data.cast::<abi::SpaMetaHeader>() = abi::SpaMetaHeader {
                flags: if block + 1 == BLOCKS {
                    abi::META_HEADER_MARKER
                } else {
                    0
                },
                offset: block as u32,
                pts: if block + 1 == BLOCKS {
                    987_654
                } else {
                    abi::TIME_INVALID
                },
                dts_offset: 0,
                seq: 93,
            };
            assert_eq!(
                allocations(|| {
                    assert_eq!(
                        process(
                            instance,
                            inputs.as_ptr(),
                            inputs.len() as u32,
                            outputs.as_mut_ptr(),
                            1
                        ),
                        0
                    );
                }),
                (0, 0)
            );
        }
        assert_eq!(output, [0.0; ACTUATORS]);
        assert_eq!(
            output_header.flags,
            abi::META_HEADER_MARKER | abi::META_HEADER_DISCONT
        );
        assert_eq!(output_header.seq, 93);
        assert_eq!(output_header.pts, 987_654);

        (descriptor.cleanup.unwrap())(instance);
    }
}

#[test]
fn generated_pyramid_row_reconstructor_uses_workers_before_terminal_publication() {
    const ROWS: usize = 8;
    const COLUMNS: usize = 8;
    const BLOCK_ROWS: usize = 2;
    const SELECTED_PIXELS: usize = 12;
    const RECONSTRUCTED: usize = 5;

    let origins = vec![[6, 0], [0, 0], [6, 4], [0, 4]];
    let mask = vec![true, false, true, true];
    let matrix = (0..RECONSTRUCTED * SELECTED_PIXELS)
        .map(|index| ((index % 13) as f32 - 6.0) * 0.003_125)
        .collect::<Vec<_>>();

    let configuration = PyramidRowReconstructorF32Configuration {
        row_block_rows: BLOCK_ROWS,
        pupil_origins: origins.clone(),
        pupil_mask: Some(mask.clone()),
        reconstructed_count: RECONSTRUCTED,
        initial_reconstructor: Some(matrix.clone()),
        reconstructed_schema: "pipewireao/controller-residual-error-f32/1".into(),
        image_size: (ROWS, COLUMNS),
        pupil_size: (2, 2),
    };
    let mut native = PyramidRowReconstructorF32::prepare(configuration).unwrap();
    let mut expected_pixels = vec![0.0_f32; SELECTED_PIXELS];
    let mut expected_mean = [0.0_f32];
    let mut expected_reconstructed = vec![0.0_f32; RECONSTRUCTED];
    for sequence in 1..=2 {
        let image = (0..ROWS * COLUMNS)
            .map(|index| index as f32 + sequence as f32 * 0.25 + 1.0)
            .collect::<Vec<_>>();
        for first_row in (0..ROWS).step_by(BLOCK_ROWS) {
            PyramidRowReconstructorF32::process_with_metadata(
                &mut native,
                &[Some(
                    SampleMetadata::new(sequence as u64, first_row as u32)
                        .with_terminal(first_row + BLOCK_ROWS == ROWS),
                )],
                &image[first_row * COLUMNS..(first_row + BLOCK_ROWS) * COLUMNS],
                &mut expected_pixels,
                &mut expected_mean,
                &mut expected_reconstructed,
            )
            .unwrap();
        }
    }

    let config = std::ffi::CString::new(
        serde_json::json!({
            "image_size": [ROWS, COLUMNS],
            "row_block_rows": BLOCK_ROWS,
            "pupil_size": [2, 2],
            "pupil_origins": origins,
            "pupil_mask": mask,
            "reconstructed_count": RECONSTRUCTED,
            "initial_reconstructor": matrix,
            "reconstructed_schema": "pipewireao/controller-residual-error-f32/1",
            "rate": [10, 1],
        })
        .to_string(),
    )
    .unwrap();
    let owned = AlgorithmDescriptor::new::<PyramidRowReconstructorF32>().unwrap();
    let descriptor = owned.as_abi();
    assert_eq!(descriptor.n_ports, 7);

    // SAFETY: every callback buffer and descriptor-owned pointer remains live
    // through the synchronous calls and cleanup consumes the instance once.
    unsafe {
        let mut instance = core::ptr::null_mut();
        assert_eq!(
            (descriptor.instantiate.unwrap())(
                descriptor,
                config.as_ptr(),
                &TEST_EXECUTOR.0,
                &mut instance,
            ),
            0
        );
        let mut formats = [core::ptr::null(); 7];
        for (port, format) in formats.iter_mut().enumerate() {
            assert_eq!(
                (descriptor.get_port_format.unwrap())(instance, port as u32, format),
                0
            );
        }
        assert_eq!(((*formats[0]).rate_num, (*formats[0]).rate_denom), (40, 1));
        for format in &formats[1..4] {
            assert_eq!(((**format).rate_num, (**format).rate_denom), (10, 1));
        }

        let mut row_block = [0.0_f32; BLOCK_ROWS * COLUMNS];
        let mut actual_pixels = [-7.0_f32; SELECTED_PIXELS];
        let mut actual_mean = [-7.0_f32];
        let mut actual_reconstructed = [-7.0_f32; RECONSTRUCTED];
        let mut input_chunk = core::mem::zeroed();
        let mut pixels_chunk = core::mem::zeroed();
        let mut mean_chunk = core::mem::zeroed();
        let mut reconstructed_chunk = core::mem::zeroed();
        let mut input_data = spa_data(&mut row_block, &mut input_chunk);
        let mut pixels_data = spa_data(&mut actual_pixels, &mut pixels_chunk);
        let mut mean_data = spa_data(&mut actual_mean, &mut mean_chunk);
        let mut reconstructed_data = spa_data(&mut actual_reconstructed, &mut reconstructed_chunk);
        let mut input_header = core::mem::zeroed::<abi::SpaMetaHeader>();
        let mut pixels_header = core::mem::zeroed::<abi::SpaMetaHeader>();
        let mut mean_header = core::mem::zeroed::<abi::SpaMetaHeader>();
        let mut reconstructed_header = core::mem::zeroed::<abi::SpaMetaHeader>();
        let mut input_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut input_header).cast(),
        };
        let mut pixels_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut pixels_header).cast(),
        };
        let mut mean_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut mean_header).cast(),
        };
        let mut reconstructed_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut reconstructed_header).cast(),
        };
        let mut input_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut input_meta,
            datas: &mut input_data,
        };
        let mut pixels_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut pixels_meta,
            datas: &mut pixels_data,
        };
        let mut mean_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut mean_meta,
            datas: &mut mean_data,
        };
        let mut reconstructed_buffer = abi::SpaBuffer {
            n_metas: 1,
            n_datas: 1,
            metas: &mut reconstructed_meta,
            datas: &mut reconstructed_data,
        };
        let mut inputs = vec![abi::Buffer {
            buffer: &mut input_buffer,
            format: formats[0],
        }];
        for format in &formats[4..] {
            inputs.push(abi::Buffer {
                buffer: core::ptr::null_mut(),
                format: *format,
            });
        }
        let mut outputs = [
            abi::Buffer {
                buffer: &mut pixels_buffer,
                format: formats[1],
            },
            abi::Buffer {
                buffer: &mut mean_buffer,
                format: formats[2],
            },
            abi::Buffer {
                buffer: &mut reconstructed_buffer,
                format: formats[3],
            },
        ];
        let process = descriptor.process.unwrap();
        let lane_calls_before = LANE_CALLS.load(Ordering::Relaxed);

        for sequence in 1..=2 {
            let image = (0..ROWS * COLUMNS)
                .map(|index| index as f32 + sequence as f32 * 0.25 + 1.0)
                .collect::<Vec<_>>();
            for first_row in (0..ROWS).step_by(BLOCK_ROWS) {
                row_block.copy_from_slice(
                    &image[first_row * COLUMNS..(first_row + BLOCK_ROWS) * COLUMNS],
                );
                *input_meta.data.cast::<abi::SpaMetaHeader>() = abi::SpaMetaHeader {
                    flags: if first_row + BLOCK_ROWS == ROWS {
                        abi::META_HEADER_MARKER
                    } else {
                        0
                    },
                    offset: first_row as u32,
                    pts: if first_row + BLOCK_ROWS == ROWS {
                        654_321
                    } else {
                        abi::TIME_INVALID
                    },
                    dts_offset: 0,
                    seq: sequence as u64,
                };
                (*input_data.chunk).size = size_of_val(&row_block) as u32;
                assert_eq!(
                    allocations(|| {
                        assert_eq!(
                            process(
                                instance,
                                inputs.as_ptr(),
                                inputs.len() as u32,
                                outputs.as_mut_ptr(),
                                outputs.len() as u32,
                            ),
                            0
                        );
                    }),
                    (0, 0)
                );
                if first_row + BLOCK_ROWS < ROWS {
                    assert_eq!(pixels_chunk.size, 0);
                    assert_eq!(mean_chunk.size, 0);
                    assert_eq!(reconstructed_chunk.size, 0);
                }
            }
        }

        assert!(LANE_CALLS.load(Ordering::Relaxed) > lane_calls_before);
        assert_eq!(
            actual_pixels.map(f32::to_bits).as_slice(),
            expected_pixels
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
                .as_slice()
        );
        assert_eq!(actual_mean[0].to_bits(), expected_mean[0].to_bits());
        assert_eq!(
            actual_reconstructed.map(f32::to_bits).as_slice(),
            expected_reconstructed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
                .as_slice()
        );
        for header in [pixels_header, mean_header, reconstructed_header] {
            assert_eq!(header.seq, 2);
            assert_eq!(header.offset, 0);
            assert_eq!(header.pts, 654_321);
            assert_eq!(header.flags, abi::META_HEADER_MARKER);
        }

        (descriptor.cleanup.unwrap())(instance);
    }
}

#[test]
fn generated_ndarray_copy_preserves_schema_rate_bits_and_realtime_boundary() {
    let layouts = [
        (abi::META_ACQUISITION_IDENTITY, 40, 1),
        (abi::META_EXPOSURE_TIMING, 24, 0),
        (abi::META_REFERENCE_TIMESTAMP, 24, 0),
        (abi::META_OPTICAL_CHOPPING, 8, 0),
        (abi::META_ELECTRICAL_TRIGGER, 8, 0),
        (0x777, 8, 0),
    ]
    .map(|(type_, size, flags)| abi::SpaMetaRequirement {
        type_,
        size,
        flags,
        reserved: 0,
    });
    let extra = [abi::SpaMetaRequirement {
        type_: 0x778,
        size: 8,
        flags: 0,
        reserved: 0,
    }];
    let owned =
        AlgorithmDescriptor::new_with_metadata::<NdarrayCopyF32>(&[&layouts, &extra]).unwrap();
    let descriptor = owned.as_abi();
    assert_eq!(descriptor.n_ports, 2);
    // SAFETY: descriptor and all callback buffers below remain live for every call.
    unsafe {
        let mut instance = core::ptr::null_mut();
        assert_eq!((descriptor.instantiate.unwrap())(descriptor,
            c"{\"extent\":2,\"schema\":\"pipewireao/dm-command-opd-um-f32/1\",\"rate\":[500,1]}".as_ptr(),
            &TEST_EXECUTOR.0, &mut instance), 0);
        let mut formats = [core::ptr::null(); 2];
        for (port, format) in formats.iter_mut().enumerate() {
            assert_eq!(
                (descriptor.get_port_format.unwrap())(instance, port as u32, format),
                0
            );
            let format = &**format;
            assert_eq!((format.rate_num, format.rate_denom), (500, 1));
            assert_eq!(format.n_dimensions, 1);
            assert_eq!(*format.shape, 2);
            assert_eq!(
                CStr::from_ptr(format.schema),
                c"pipewireao/dm-command-opd-um-f32/1"
            );
        }
        let sentinel = f32::from_bits(0x7fc54321);
        let mut input = [
            sentinel,
            f32::from_bits(0x80000000),
            f32::from_bits(0x7fc12345),
            sentinel,
        ];
        let mut output = [sentinel; 4];
        let mut input_chunk = core::mem::zeroed();
        let mut output_chunk = core::mem::zeroed();
        let mut input_data = spa_data(&mut input, &mut input_chunk);
        let mut output_data = spa_data(&mut output, &mut output_chunk);
        input_chunk.offset = size_of::<f32>() as u32;
        input_chunk.size = 2 * size_of::<f32>() as u32;
        output_chunk.offset = size_of::<f32>() as u32;
        let mut input_header = abi::SpaMetaHeader {
            flags: 7,
            offset: 3,
            pts: 99,
            dts_offset: -2,
            seq: 42,
        };
        let mut output_header = core::mem::zeroed::<abi::SpaMetaHeader>();
        let input_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut input_header).cast(),
        };
        let output_meta = abi::SpaMeta {
            type_: abi::META_HEADER,
            size: size_of::<abi::SpaMetaHeader>() as u32,
            data: core::ptr::from_mut(&mut output_header).cast(),
        };
        let mut identity = abi::SpaMetaAcquisitionIdentity {
            flags: 1,
            reserved: 0,
            domain: [1; 16],
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
        let mut opaque = 17_u64;
        let mut output_identity = abi::SpaMetaAcquisitionIdentity::default();
        let mut output_exposure = abi::SpaMetaExposureTiming::default();
        let mut output_reference = abi::SpaMetaReferenceTimestamp::default();
        let mut output_optical = abi::SpaMetaOpticalChopping::default();
        let mut output_trigger = abi::SpaMetaElectricalTrigger::default();
        let mut output_opaque = 0_u64;
        let mut stale = 88_u64;
        fn record<T>(type_: u32, value: &mut T) -> abi::SpaMeta {
            abi::SpaMeta {
                type_,
                size: size_of::<T>() as u32,
                data: core::ptr::from_mut(value).cast(),
            }
        }
        let mut input_metas = [
            input_meta,
            record(abi::META_ACQUISITION_IDENTITY, &mut identity),
            record(abi::META_EXPOSURE_TIMING, &mut exposure),
            record(abi::META_REFERENCE_TIMESTAMP, &mut reference),
            record(abi::META_OPTICAL_CHOPPING, &mut optical),
            record(abi::META_ELECTRICAL_TRIGGER, &mut trigger),
            record(0x777, &mut opaque),
        ];
        let mut output_metas = [
            output_meta,
            record(abi::META_ACQUISITION_IDENTITY, &mut output_identity),
            record(abi::META_EXPOSURE_TIMING, &mut output_exposure),
            record(abi::META_REFERENCE_TIMESTAMP, &mut output_reference),
            record(abi::META_OPTICAL_CHOPPING, &mut output_optical),
            record(abi::META_ELECTRICAL_TRIGGER, &mut output_trigger),
            record(0x777, &mut output_opaque),
            record(0x778, &mut stale),
        ];
        let descriptor_snapshot = output_metas
            .iter()
            .map(|m| (m.type_, m.size, m.data))
            .collect::<Vec<_>>();
        let mut input_buffer = abi::SpaBuffer {
            n_metas: input_metas.len() as u32,
            n_datas: 1,
            metas: input_metas.as_mut_ptr(),
            datas: &mut input_data,
        };
        let mut output_buffer = abi::SpaBuffer {
            n_metas: output_metas.len() as u32,
            n_datas: 1,
            metas: output_metas.as_mut_ptr(),
            datas: &mut output_data,
        };
        let inputs = [abi::Buffer {
            buffer: &mut input_buffer,
            format: formats[0],
        }];
        let mut outputs = [abi::Buffer {
            buffer: &mut output_buffer,
            format: formats[1],
        }];
        let process = descriptor.process.unwrap();
        assert_eq!(
            process(instance, inputs.as_ptr(), 1, outputs.as_mut_ptr(), 1),
            0
        );
        assert_eq!(
            allocations(|| {
                for _ in 0..16 {
                    assert_eq!(
                        process(instance, inputs.as_ptr(), 1, outputs.as_mut_ptr(), 1),
                        0
                    );
                }
            }),
            (0, 0)
        );
        assert_eq!(output.map(f32::to_bits), input.map(f32::to_bits));
        assert_eq!(input_chunk.offset, size_of::<f32>() as u32);
        assert_eq!(input_chunk.size, 2 * size_of::<f32>() as u32);
        assert_eq!(output_chunk.offset, size_of::<f32>() as u32);
        assert_eq!(output_chunk.size, 2 * size_of::<f32>() as u32);
        assert_eq!(
            (
                output_header.flags,
                output_header.offset,
                output_header.pts,
                output_header.dts_offset,
                output_header.seq
            ),
            (
                input_header.flags,
                input_header.offset,
                input_header.pts,
                input_header.dts_offset,
                input_header.seq
            )
        );
        assert_eq!(output_identity.sequence, 19);
        assert_eq!(
            (output_exposure.start_nsec, output_exposure.clock),
            (-100, 4)
        );
        assert_eq!(output_reference.timestamp_nsec, 18_000_000_000_000_001_123);
        assert_eq!(
            (output_optical.classification, output_trigger.classification),
            (2, 1)
        );
        assert_eq!((output_opaque, stale), (17, 0));
        assert_eq!(
            output_metas
                .iter()
                .map(|m| (m.type_, m.size, m.data))
                .collect::<Vec<_>>(),
            descriptor_snapshot
        );
        // Required negotiated storage without a populated fact fails before copying data.
        identity.flags = 0;
        identity.domain = [0; 16];
        identity.generation = 0;
        identity.sequence = 0;
        assert_eq!(identity.flags, 0);
        assert_eq!(
            allocations(|| assert_eq!(
                process(instance, inputs.as_ptr(), 1, outputs.as_mut_ptr(), 1),
                -abi::EINVAL
            )),
            (0, 0)
        );
        assert_eq!(output_chunk.size, 0);
        assert_eq!(input_header.seq, 42);
        assert_eq!(reference.timestamp_nsec, 18_000_000_000_000_001_123);
        (descriptor.cleanup.unwrap())(instance);
    }
}

#[test]
fn metadata_contracts_follow_declarations_and_explicit_deployment_layouts() {
    use filtergraph_algorithms::{ChoppedImageDifferenceF32, SccCorrectionF32};
    let scc = AlgorithmDescriptor::new::<SccCorrectionF32>().unwrap();
    let chopped = AlgorithmDescriptor::new::<ChoppedImageDifferenceF32>().unwrap();
    // SAFETY: owned descriptors retain immutable, bounded metadata arrays.
    unsafe {
        let scc_ports =
            core::slice::from_raw_parts(scc.as_abi().ports, scc.as_abi().n_ports as usize);
        let frame = core::slice::from_raw_parts(scc_ports[0].metas, scc_ports[0].n_metas as usize);
        assert_eq!(
            frame
                .iter()
                .map(|r| (r.type_, r.size, r.flags))
                .collect::<Vec<_>>(),
            vec![
                (abi::META_ACQUISITION_IDENTITY, 40, 1),
                (abi::META_EXPOSURE_TIMING, 24, 1),
                (abi::META_OPTICAL_CHOPPING, 8, 1)
            ]
        );
        let command = &scc_ports[1];
        assert_eq!(command.propagation, abi::META_PROPAGATION_PRESERVE);
        assert_eq!(command.n_metas, 3);
        assert!(
            core::slice::from_raw_parts(command.metas, 3)
                .iter()
                .all(|r| r.flags == 0)
        );
        let chopped_ports =
            core::slice::from_raw_parts(chopped.as_abi().ports, chopped.as_abi().n_ports as usize);
        assert_eq!(chopped_ports[0].n_metas, 4);
        assert_eq!(
            core::slice::from_raw_parts(chopped_ports[0].metas, 4)
                .iter()
                .map(|record| (record.type_, record.size, record.flags))
                .collect::<Vec<_>>(),
            vec![
                (abi::META_OPTICAL_CHOPPING, 8, 0),
                (abi::META_ELECTRICAL_TRIGGER, 8, 0),
                (abi::META_ACQUISITION_IDENTITY, 40, 0),
                (abi::META_EXPOSURE_TIMING, 24, 0),
            ]
        );
        for output in &chopped_ports[2..] {
            assert_eq!(output.n_metas, 4);
            assert_eq!(output.propagation, abi::META_PROPAGATION_PRESERVE);
        }
    }
    for (type_, size, flags, reserved) in [
        (11, 96, 0, 0),
        (0x302, 24, 0, 0),
        (abi::META_EXPOSURE_TIMING, 64, 0, 0),
        (abi::META_HEADER, 32, 0, 0),
        (0x777, 8, 2, 0),
        (0x777, 8, 0, 1),
        (0x777, 4096, 0, 0),
    ] {
        let records = [abi::SpaMetaRequirement {
            type_,
            size,
            flags,
            reserved,
        }];
        assert!(
            AlgorithmDescriptor::new_with_metadata::<NdarrayCopyF32>(&[&records, &[]]).is_err()
        );
    }
    let duplicate = [abi::SpaMetaRequirement {
        type_: 0x777,
        size: 8,
        flags: 0,
        reserved: 0,
    }; 2];
    assert!(AlgorithmDescriptor::new_with_metadata::<NdarrayCopyF32>(&[&duplicate, &[]]).is_err());
}

#[test]
fn deployment_metadata_must_match_port_direction_count_and_inherited_record_size() {
    let optional = abi::SpaMetaRequirement {
        type_: 0x777,
        size: 8,
        flags: 0,
        reserved: 0,
    };
    let incompatible = abi::SpaMetaRequirement {
        size: 16,
        ..optional
    };

    assert_eq!(
        AlgorithmDescriptor::new_with_metadata::<NdarrayCopyF32>(&[&[]]).err(),
        Some(abi::EINVAL)
    );
    assert_eq!(
        AlgorithmDescriptor::new_with_metadata::<NdarrayCopyF32>(&[&[optional], &[incompatible]])
            .err(),
        Some(abi::EINVAL)
    );
    assert_eq!(
        AlgorithmDescriptor::new_with_metadata::<OpticalGainCorrectionF32>(&[
            &[],
            &[],
            &[optional]
        ])
        .err(),
        Some(abi::EINVAL)
    );
    let valid =
        AlgorithmDescriptor::new_with_metadata::<NdarrayCopyF32>(&[&[optional], &[optional]])
            .unwrap();
    unsafe {
        let ports = core::slice::from_raw_parts(valid.as_abi().ports, 2);
        assert_eq!(ports[1].n_metas, 1);
        assert_eq!((*ports[1].metas).size, 8);
        assert_eq!((*ports[1].metas).type_, optional.type_);
    }
}

#[test]
fn invalid_port_topologies_are_rejected_before_export() {
    use filtergraph_algorithms::{
        AlgorithmConfiguration, AlgorithmDeclaration, AlgorithmInputs, AlgorithmOperation,
        AlgorithmOutputs, AlgorithmPort, AlgorithmProcessResult, ElementType, NoDeclaredParameters,
        NoDeclaredProperties, NoWorkspace, PortDirection, PortRole, PreparedAlgorithm,
    };

    const INPUT: AlgorithmPort = AlgorithmPort::new(
        "data",
        PortDirection::Input,
        PortRole::Data,
        ElementType::F32,
        None,
    );
    const OUTPUT: AlgorithmPort = AlgorithmPort::new(
        "output",
        PortDirection::Output,
        PortRole::Data,
        ElementType::F32,
        None,
    );
    struct Declaration<const CASE: u8>;
    impl<const CASE: u8> AlgorithmDeclaration for Declaration<CASE> {
        type Configuration = ();
        type Plan = ();
        type Workspace = NoWorkspace;
        type PreparationError = core::convert::Infallible;
        type ProcessError = core::convert::Infallible;
        type ScalarProperties = NoDeclaredProperties;
        type Parameters = NoDeclaredParameters;
        type ParameterError = core::convert::Infallible;

        const NAME: &'static str = "test-topology";
        const CONFIGURATION: &'static [AlgorithmConfiguration] = &[];
        const PORTS: &'static [AlgorithmPort] = match CASE {
            0 => &[],
            1 => &[INPUT, INPUT],
            2 => &[INPUT.conditional(), OUTPUT],
            3 => &[
                INPUT,
                AlgorithmPort::new(
                    "output",
                    PortDirection::Output,
                    PortRole::Data,
                    ElementType::F32,
                    Some("absent"),
                ),
            ],
            4 => &[
                AlgorithmPort::new(
                    "data",
                    PortDirection::Input,
                    PortRole::Data,
                    ElementType::F32,
                    Some("data"),
                ),
                OUTPUT,
            ],
            5 => &[OUTPUT],
            6 => &[
                INPUT,
                AlgorithmPort::new(
                    "output",
                    PortDirection::Output,
                    PortRole::Data,
                    ElementType::F64,
                    None,
                ),
            ],
            _ => &[INPUT, OUTPUT],
        };
        const OPERATION: AlgorithmOperation = if CASE == 5 || CASE == 6 {
            AlgorithmOperation::DenseF32
        } else {
            AlgorithmOperation::General
        };
        fn prepare(_: ()) -> Result<PreparedAlgorithm<Self>, Self::PreparationError> {
            unreachable!("descriptor export is independent of scientific preparation")
        }
        fn process(
            _: &(),
            _: &mut NoWorkspace,
            _: AlgorithmInputs<'_, '_>,
            _: AlgorithmOutputs<'_, '_>,
        ) -> Result<AlgorithmProcessResult, Self::ProcessError> {
            unreachable!("descriptor export is independent of processing")
        }
    }
    for result in [
        AlgorithmDescriptor::new::<Declaration<0>>(),
        AlgorithmDescriptor::new::<Declaration<1>>(),
        AlgorithmDescriptor::new::<Declaration<2>>(),
        AlgorithmDescriptor::new::<Declaration<3>>(),
        AlgorithmDescriptor::new::<Declaration<4>>(),
        AlgorithmDescriptor::new::<Declaration<5>>(),
        AlgorithmDescriptor::new::<Declaration<6>>(),
    ] {
        assert_eq!(result.err(), Some(abi::EINVAL));
    }
    assert_eq!(
        AlgorithmDescriptor::new::<Declaration<7>>()
            .unwrap()
            .as_abi()
            .n_ports,
        2
    );
}
