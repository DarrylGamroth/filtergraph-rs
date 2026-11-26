//! Large-frame proof for the portable image-backed Shack-Hartmann declaration.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use filtergraph_algorithms::schemas::{CONTROLLER_RESIDUAL_ERROR_V1, REQUESTED_PDM_COMMAND_V1};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmOperation, AlgorithmOperationPlan, AlgorithmPlan,
    LeakyIntegratorF32, LeakyIntegratorF32Configuration, PdmCommandF32, PdmCommandF32Configuration,
    PdmCommandInput, PdmCommandOutput, PdmCommandPlan, PreparedGemv, RegionPixels, RegionPixelsMut,
    ScalarOrVector, ShackHartmannImageF32, ShackHartmannImageF32Configuration,
    ShackHartmannImagePlan, ShackHartmannOutput, ShackHartmannReconstructorF32,
    ShackHartmannReconstructorF32Configuration,
};

const DETECTOR_ROWS: usize = 352;
const DETECTOR_COLUMNS: usize = 352;
const SUBAPERTURE_ROWS: usize = 22;
const SUBAPERTURE_COLUMNS: usize = 22;
const SUBAPERTURE_GRID_ROWS: usize = 16;
const SUBAPERTURE_GRID_COLUMNS: usize = 16;
const SUBAPERTURE_COUNT: usize = SUBAPERTURE_GRID_ROWS * SUBAPERTURE_GRID_COLUMNS;

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

fn allocation_activity(callback: impl FnOnce()) -> (usize, usize) {
    ALLOCATIONS.set(0);
    DEALLOCATIONS.set(0);
    COUNTING.set(true);
    callback();
    COUNTING.set(false);
    (ALLOCATIONS.get(), DEALLOCATIONS.get())
}

fn origins() -> Vec<[usize; 2]> {
    let mut origins = Vec::with_capacity(SUBAPERTURE_COUNT);
    for grid_row in 0..SUBAPERTURE_GRID_ROWS {
        for grid_column in 0..SUBAPERTURE_GRID_COLUMNS {
            origins.push([
                grid_row * SUBAPERTURE_ROWS,
                grid_column * SUBAPERTURE_COLUMNS,
            ]);
        }
    }
    origins
}

fn detector_image() -> Vec<f32> {
    let mut image = vec![0.0_f32; DETECTOR_ROWS * DETECTOR_COLUMNS];
    for row in 0..DETECTOR_ROWS {
        for column in 0..DETECTOR_COLUMNS {
            let subaperture =
                (row / SUBAPERTURE_ROWS) * SUBAPERTURE_GRID_COLUMNS + column / SUBAPERTURE_COLUMNS;
            image[row * DETECTOR_COLUMNS + column] = 1.0
                + 0.01 * (row % SUBAPERTURE_ROWS) as f32
                + 0.02 * (column % SUBAPERTURE_COLUMNS) as f32
                + 0.001 * subaperture as f32;
        }
    }
    image
}

fn measured_slopes() -> Vec<f32> {
    let mut sensor = ShackHartmannImageF32::prepare(ShackHartmannImageF32Configuration {
        subaperture_count: SUBAPERTURE_COUNT,
        image_schema: "pipewireao/calibrated-pixels-f32/1".into(),
        initial_subaperture_origins: None,
        coordinate_scale: 0.5,
        pixel_threshold: 0.0,
        flux_threshold: 1.0,
        reference_slopes: None,
        active: None,
        image_size: (DETECTOR_ROWS, DETECTOR_COLUMNS),
        subaperture_size: (SUBAPERTURE_ROWS, SUBAPERTURE_COLUMNS),
    })
    .unwrap();
    let image = detector_image();
    let mut slopes = vec![0.0_f32; SUBAPERTURE_COUNT * 2];
    let mut flux = vec![0.0_f32; SUBAPERTURE_COUNT];
    let mut validity = vec![false; SUBAPERTURE_COUNT];
    ShackHartmannImageF32::process(&mut sensor, &image, &mut slopes, &mut flux, &mut validity)
        .unwrap();
    assert!(validity.iter().all(|valid| *valid));
    slopes
}

#[test]
fn large_frame_declaration_matches_materialized_and_image_view_execution_bit_exactly() {
    let direct_origins = origins()
        .into_iter()
        .map(|origin| (origin[0], origin[1]))
        .collect::<Vec<_>>();
    let configuration = ShackHartmannImageF32Configuration {
        subaperture_count: SUBAPERTURE_COUNT,
        image_schema: "pipewireao/calibrated-pixels-f32/1".into(),
        initial_subaperture_origins: None,
        coordinate_scale: 0.5,
        pixel_threshold: 0.0,
        flux_threshold: 1.0,
        reference_slopes: None,
        active: None,
        image_size: (DETECTOR_ROWS, DETECTOR_COLUMNS),
        subaperture_size: (SUBAPERTURE_ROWS, SUBAPERTURE_COLUMNS),
    };
    let direct = ShackHartmannImagePlan::from_detector_regions(
        (DETECTOR_ROWS, DETECTOR_COLUMNS),
        (SUBAPERTURE_ROWS, SUBAPERTURE_COLUMNS),
        &direct_origins,
        configuration.coordinate_scale,
        configuration.pixel_threshold,
        configuration.flux_threshold,
        None,
        None,
    )
    .unwrap();
    let mut declared = ShackHartmannImageF32::prepare(configuration).unwrap();

    let formats = declared.formats();
    assert_eq!(formats.len(), 9);
    assert_eq!(
        formats[0].shape(),
        &[DETECTOR_ROWS as u32, DETECTOR_COLUMNS as u32]
    );
    assert_eq!(formats[1].shape(), &[SUBAPERTURE_COUNT as u32, 2]);
    assert_eq!(formats[2].shape(), &[SUBAPERTURE_COUNT as u32]);
    assert_eq!(formats[3].shape(), &[SUBAPERTURE_COUNT as u32]);

    let image = detector_image();
    let mut regions = vec![0.0_f32; SUBAPERTURE_COUNT * SUBAPERTURE_ROWS * SUBAPERTURE_COLUMNS];
    direct
        .regions()
        .process(
            RegionPixelsMut::new(
                &mut regions,
                SUBAPERTURE_ROWS * SUBAPERTURE_COLUMNS,
                SUBAPERTURE_COUNT,
            )
            .unwrap(),
            &mut direct.regions().workspace(),
            &image,
        )
        .unwrap();

    let mut materialized_slopes = vec![0.0_f32; SUBAPERTURE_COUNT * 2];
    let mut materialized_flux = vec![0.0_f32; SUBAPERTURE_COUNT];
    let mut materialized_validity = vec![false; SUBAPERTURE_COUNT];
    direct
        .measurement()
        .process(
            ShackHartmannOutput::new(
                &mut materialized_slopes,
                &mut materialized_flux,
                &mut materialized_validity,
            )
            .unwrap(),
            &mut direct.measurement().workspace(),
            RegionPixels::new(
                &regions,
                SUBAPERTURE_ROWS * SUBAPERTURE_COLUMNS,
                SUBAPERTURE_COUNT,
            )
            .unwrap(),
        )
        .unwrap();

    let mut viewed_slopes = vec![0.0_f32; SUBAPERTURE_COUNT * 2];
    let mut viewed_flux = vec![0.0_f32; SUBAPERTURE_COUNT];
    let mut viewed_validity = vec![false; SUBAPERTURE_COUNT];
    direct
        .process(
            ShackHartmannOutput::new(&mut viewed_slopes, &mut viewed_flux, &mut viewed_validity)
                .unwrap(),
            &mut direct.workspace(),
            &image,
        )
        .unwrap();

    let mut declared_slopes = vec![0.0_f32; SUBAPERTURE_COUNT * 2];
    let mut declared_flux = vec![0.0_f32; SUBAPERTURE_COUNT];
    let mut declared_validity = vec![false; SUBAPERTURE_COUNT];
    ShackHartmannImageF32::process(
        &mut declared,
        &image,
        &mut declared_slopes,
        &mut declared_flux,
        &mut declared_validity,
    )
    .unwrap();

    assert_eq!(viewed_slopes, materialized_slopes);
    assert_eq!(viewed_flux, materialized_flux);
    assert_eq!(viewed_validity, materialized_validity);
    assert_eq!(declared_slopes, materialized_slopes);
    assert_eq!(declared_flux, materialized_flux);
    assert_eq!(declared_validity, materialized_validity);
    assert!(declared_validity.iter().all(|valid| *valid));

    assert_eq!(
        allocation_activity(|| {
            ShackHartmannImageF32::process(
                &mut declared,
                &image,
                &mut declared_slopes,
                &mut declared_flux,
                &mut declared_validity,
            )
            .unwrap();
        }),
        (0, 0)
    );
}

#[test]
fn large_frame_controller_declarations_match_direct_reconstruction_and_command_plans() {
    const ACTUATOR_COUNT: usize = 277;
    const SLOPE_COUNT: usize = SUBAPERTURE_COUNT * 2;
    let slopes = measured_slopes();
    let mut matrix = vec![0.0_f32; ACTUATOR_COUNT * SLOPE_COUNT];
    for actuator in 0..ACTUATOR_COUNT {
        matrix[actuator * SLOPE_COUNT + (actuator * 37) % SLOPE_COUNT] = 0.01;
        matrix[actuator * SLOPE_COUNT + (actuator * 37 + 1) % SLOPE_COUNT] = -0.005;
    }

    let direct = PreparedGemv::new(ACTUATOR_COUNT, SLOPE_COUNT, Arc::from(matrix.clone())).unwrap();
    let mut expected = vec![0.0_f32; ACTUATOR_COUNT];
    direct.multiply(&mut expected, &slopes).unwrap();

    let mut declared =
        ShackHartmannReconstructorF32::prepare(ShackHartmannReconstructorF32Configuration {
            subaperture_count: SUBAPERTURE_COUNT,
            initial_reconstructor: Some(matrix),
            reconstructed_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
            reconstructed_count: ACTUATOR_COUNT,
        })
        .unwrap();
    assert_eq!(
        ShackHartmannReconstructorF32::OPERATION,
        AlgorithmOperation::DenseF32
    );
    assert!(matches!(
        ShackHartmannReconstructorF32::operation_plan(declared.plan()),
        AlgorithmOperationPlan::DenseF32(_)
    ));
    let mut actual = vec![0.0_f32; ACTUATOR_COUNT];
    ShackHartmannReconstructorF32::process(&mut declared, &slopes, &mut actual).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(
        declared.formats()[0].shape(),
        &[SUBAPERTURE_COUNT as u32, 2]
    );
    assert_eq!(declared.formats()[1].shape(), &[ACTUATOR_COUNT as u32]);
    assert_eq!(
        declared.formats()[2].shape(),
        &[ACTUATOR_COUNT as u32, SLOPE_COUNT as u32]
    );

    let mut integrator = LeakyIntegratorF32::prepare(LeakyIntegratorF32Configuration {
        extent: ACTUATOR_COUNT,
        initial_state: 0.0,
        input_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
        output_schema: REQUESTED_PDM_COMMAND_V1.into(),
    })
    .unwrap();
    let mut command = PdmCommandF32::prepare(PdmCommandF32Configuration {
        demanded_schema: filtergraph_algorithms::schemas::DEMANDED_PDM_COMMAND_V1.into(),
        actuator_count: ACTUATOR_COUNT,
        slew_limit: None,
        initial_current: None,
        quantization_origin: None,
        quantization_step: None,
        dither_seed: None,
        command_limits: (ScalarOrVector::Scalar(-1.0), ScalarOrVector::Scalar(1.0)),
    })
    .unwrap();
    let mut requested = vec![0.0_f32; ACTUATOR_COUNT];
    let mut demanded = vec![0.0_f32; ACTUATOR_COUNT];
    let mut feedback = vec![0.0_f32; ACTUATOR_COUNT];
    LeakyIntegratorF32::process(&mut integrator, &actual, &mut requested).unwrap();
    PdmCommandF32::process(&mut command, &requested, &mut demanded, &mut feedback).unwrap();

    let direct_command = PdmCommandPlan::new(
        Arc::from(vec![-1.0_f32; ACTUATOR_COUNT]),
        Arc::from(vec![1.0_f32; ACTUATOR_COUNT]),
        None,
    )
    .unwrap();
    let mut direct_command_workspace = direct_command
        .workspace(&vec![0.0_f32; ACTUATOR_COUNT])
        .unwrap();
    let mut expected_demanded = vec![0.0_f32; ACTUATOR_COUNT];
    let mut expected_feedback = vec![0.0_f32; ACTUATOR_COUNT];
    direct_command
        .process(
            PdmCommandOutput {
                demanded: &mut expected_demanded,
                constraint_feedback: &mut expected_feedback,
            },
            &mut direct_command_workspace,
            PdmCommandInput {
                requested: &expected,
            },
        )
        .unwrap();
    assert_eq!(requested, expected);
    assert_eq!(demanded, expected_demanded);
    assert_eq!(feedback, expected_feedback);

    assert_eq!(
        allocation_activity(|| {
            ShackHartmannReconstructorF32::process(&mut declared, &slopes, &mut actual).unwrap();
            LeakyIntegratorF32::process(&mut integrator, &actual, &mut requested).unwrap();
            PdmCommandF32::process(&mut command, &requested, &mut demanded, &mut feedback).unwrap();
        }),
        (0, 0)
    );
}

#[test]
fn pdm_declaration_reset_restores_prepared_state_without_heap_activity() {
    const ACTUATOR_COUNT: usize = 277;
    let mut command = PdmCommandF32::prepare(PdmCommandF32Configuration {
        demanded_schema: filtergraph_algorithms::schemas::DEMANDED_PDM_COMMAND_V1.into(),
        actuator_count: ACTUATOR_COUNT,
        slew_limit: Some(ScalarOrVector::Scalar(0.05)),
        initial_current: None,
        quantization_origin: None,
        quantization_step: None,
        dither_seed: None,
        command_limits: (ScalarOrVector::Scalar(-1.0), ScalarOrVector::Scalar(1.0)),
    })
    .unwrap();
    let requested = vec![1.0_f32; ACTUATOR_COUNT];
    let zero = vec![0.0_f32; ACTUATOR_COUNT];
    let mut demanded = vec![0.0_f32; ACTUATOR_COUNT];
    let mut feedback = vec![0.0_f32; ACTUATOR_COUNT];

    for _ in 0..2 {
        PdmCommandF32::process(&mut command, &requested, &mut demanded, &mut feedback).unwrap();
    }
    assert_eq!(demanded, vec![0.1; ACTUATOR_COUNT]);

    assert_eq!(
        allocation_activity(|| PdmCommandF32::reset(&mut command)),
        (0, 0)
    );
    PdmCommandF32::process(&mut command, &zero, &mut demanded, &mut feedback).unwrap();
    assert_eq!(demanded, zero);
}
