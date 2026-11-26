//! Image-view pyramid declaration proof without a materialized pupil slab.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use filtergraph_algorithms::{
    AlgorithmInput, AlgorithmPlan, DeclarationParameterError, DeclarationProcessError,
    PyramidPixelError, PyramidPixelImageError, PyramidPixelImagePlan,
    PyramidPixelImagePreparationError, PyramidPixelImageProcessError, PyramidPixelOutput,
    PyramidPixelPlan, PyramidPupilImageF32, PyramidPupilImageF32Configuration,
    RegionExtractionPlan, RegionPixels, RegionPixelsMut,
};

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

#[test]
fn portable_pyramid_declaration_matches_materialized_and_image_view_execution() {
    let origins = [(0_usize, 0_usize), (0, 2), (2, 0), (2, 2)];
    let mask = vec![true, false, true, true];
    let pupil_regions = RegionExtractionPlan::new((4, 4), &origins, (2, 2)).unwrap();
    let normalization = PyramidPixelPlan::new(mask.clone()).unwrap();
    let direct = PyramidPixelImagePlan::new(pupil_regions.clone(), normalization.clone()).unwrap();
    let mut declared = PyramidPupilImageF32::prepare(PyramidPupilImageF32Configuration {
        pupil_origins: origins.map(|(row, column)| [row, column]).to_vec(),
        pupil_mask: Some(mask),
        image_size: (4, 4),
        pupil_size: (2, 2),
    })
    .unwrap();
    assert_eq!(declared.formats().len(), 5);
    assert_eq!(declared.formats()[0].shape(), &[4, 4]);
    assert_eq!(declared.formats()[1].shape(), &[4, 3]);
    assert_eq!(declared.formats()[2].shape(), &[1]);
    assert_eq!(declared.formats()[3].shape(), &[4, 2]);
    assert_eq!(declared.formats()[4].shape(), &[2, 2]);

    let image = (1_u32..=16).map(|value| value as f32).collect::<Vec<_>>();
    let mut pupils = vec![0.0_f32; 16];
    pupil_regions
        .process(
            RegionPixelsMut::new(&mut pupils, 4, 4).unwrap(),
            &mut pupil_regions.workspace(),
            &image,
        )
        .unwrap();

    let mut materialized_workspace = normalization.workspace();
    let mut viewed_workspace = direct.workspace();
    let mut materialized_pixels = vec![0.0_f32; 12];
    let mut viewed_pixels = vec![0.0_f32; 12];
    let mut declared_pixels = vec![0.0_f32; 12];
    let mut materialized_mean = 0.0_f32;
    let mut viewed_mean = 0.0_f32;
    let mut declared_mean = [0.0_f32];

    for frame in 0..2 {
        normalization
            .process(
                PyramidPixelOutput::new(&mut materialized_pixels, &mut materialized_mean),
                &mut materialized_workspace,
                RegionPixels::new(&pupils, 4, 4).unwrap(),
            )
            .unwrap();
        direct
            .process(
                PyramidPixelOutput::new(&mut viewed_pixels, &mut viewed_mean),
                &mut viewed_workspace,
                &image,
            )
            .unwrap();
        PyramidPupilImageF32::process(
            &mut declared,
            &image,
            &mut declared_pixels,
            &mut declared_mean,
        )
        .unwrap();

        assert_eq!(viewed_pixels, materialized_pixels);
        assert_eq!(declared_pixels, materialized_pixels);
        assert_eq!(viewed_mean, materialized_mean);
        assert_eq!(declared_mean[0], materialized_mean);
        assert_eq!(materialized_mean, 8.5);
        if frame == 0 {
            assert_eq!(declared_pixels, vec![0.0; 12]);
        }
    }

    assert_eq!(
        allocation_activity(|| {
            PyramidPupilImageF32::process(
                &mut declared,
                &image,
                &mut declared_pixels,
                &mut declared_mean,
            )
            .unwrap();
        }),
        (0, 0)
    );

    let replacement_mask = [false, true, true, true];
    assert_eq!(
        declared.replace_parameter(4, AlgorithmInput::from(&replacement_mask[..])),
        Some(Ok(()))
    );
    PyramidPupilImageF32::process(
        &mut declared,
        &image,
        &mut declared_pixels,
        &mut declared_mean,
    )
    .unwrap();
    assert_eq!(declared_pixels[0..3], [2.0 / 8.5, 5.0 / 8.5, 6.0 / 8.5]);

    let shape_changing_mask = [true, false, false, true];
    assert!(matches!(
        declared.replace_parameter(4, AlgorithmInput::from(&shape_changing_mask[..])),
        Some(Err(DeclarationParameterError::Algorithm(
            PyramidPixelImagePreparationError::SelectedPixelExtent { .. }
        )))
    ));
    assert_eq!(
        allocation_activity(|| PyramidPupilImageF32::reset(&mut declared)),
        (0, 0)
    );
}

#[test]
fn declaration_adopts_irregular_runs_and_new_origins_without_resetting_gain() {
    let image_size = (7, 86);
    let pupil_size = (3, 38);
    let origins = [(0, 2), (0, 42), (3, 2), (3, 42)];
    let new_origins = [(1, 43), (4, 3), (1, 3), (4, 43)];
    let mut mask = [false; 114];
    mask[12..56].fill(true); // Spans a pupil row boundary.
    mask[58] = true;
    mask[60] = true;
    mask[67..71].fill(true);
    let mut new_mask = [false; 114];
    new_mask[1..17].fill(true);
    new_mask[38..55].fill(true);
    new_mask[97..114].fill(true);
    let mut declared = PyramidPupilImageF32::prepare(PyramidPupilImageF32Configuration {
        image_size,
        pupil_size,
        pupil_origins: origins.map(|(row, column)| [row, column]).to_vec(),
        pupil_mask: Some(mask.to_vec()),
    })
    .unwrap();
    let original = PyramidPixelImagePlan::from_detector_pupils(
        image_size,
        pupil_size,
        &origins,
        mask.to_vec(),
    )
    .unwrap();
    let replacement = PyramidPixelImagePlan::from_detector_pupils(
        image_size,
        pupil_size,
        &new_origins,
        new_mask.to_vec(),
    )
    .unwrap();
    let mut workspace = original.workspace();
    let mut oracle = &original;
    let mut expected_pixels = [0.0; 200];
    let mut expected_mean = 0.0;
    let mut actual_pixels = [0.0; 200];
    let mut actual_mean = [0.0];
    for frame in 0..4 {
        if frame == 2 {
            let values = [1u32, 43, 4, 3, 1, 3, 4, 43];
            assert_eq!(
                declared.replace_parameter(3, AlgorithmInput::from(&values[..])),
                Some(Ok(()))
            );
            assert_eq!(
                declared.replace_parameter(4, AlgorithmInput::from(&new_mask[..])),
                Some(Ok(()))
            );
            oracle = &replacement;
        }
        let image = core::array::from_fn::<_, { 7 * 86 }, _>(|index| {
            (index * (frame + 1) + 1) as f32 * 0.1
        });
        oracle
            .normalization()
            .process_image_regions(
                PyramidPixelOutput::new(&mut expected_pixels, &mut expected_mean),
                &mut workspace,
                oracle.pupil_regions().image_regions(&image).unwrap(),
            )
            .unwrap();
        assert_eq!(
            allocation_activity(|| {
                PyramidPupilImageF32::process(
                    &mut declared,
                    &image,
                    &mut actual_pixels,
                    &mut actual_mean,
                )
                .unwrap();
            }),
            (0, 0)
        );
        assert_eq!(actual_pixels, expected_pixels);
        assert_eq!(actual_mean, [expected_mean]);
        assert_eq!(declared.workspace(), &workspace);
    }
}

#[test]
fn later_chunk_failure_writes_only_the_finite_prefix_and_keeps_the_gain() {
    const FAILURE: usize = 141;
    const SENTINEL: f32 = -99.0;
    let mut mask = vec![true; 80];
    mask[79] = false;
    let mut declared = PyramidPupilImageF32::prepare(PyramidPupilImageF32Configuration {
        image_size: (8, 40),
        pupil_size: (2, 40),
        pupil_origins: vec![[0, 0], [2, 0], [4, 0], [6, 0]],
        pupil_mask: Some(mask),
    })
    .unwrap();
    let mut image = [f32::MIN_POSITIVE; 320];
    let mut pixels = [SENTINEL; 316];
    let mut mean = [SENTINEL];
    PyramidPupilImageF32::process(&mut declared, &image, &mut pixels, &mut mean).unwrap();
    let committed = *declared.workspace();
    assert_eq!(
        committed.reciprocal_previous_frame_mean(),
        1.0 / f32::MIN_POSITIVE
    );
    pixels.fill(SENTINEL);
    mean.fill(SENTINEL);
    image[FAILURE] = f32::MAX;
    image[FAILURE + 1] = -f32::MAX;

    let error = |error| {
        Err(DeclarationProcessError::Algorithm(
            PyramidPixelImageProcessError::Algorithm(PyramidPixelImageError::Pyramid(error)),
        ))
    };
    // The first pass includes masked-out input and preserves input/sum error order.
    image[319] = f32::NAN;
    image[0..2].fill(f32::MAX);
    assert_eq!(
        PyramidPupilImageF32::process(&mut declared, &image, &mut pixels, &mut mean),
        error(PyramidPixelError::NonFiniteMean),
    );
    image[0..2].fill(f32::MIN_POSITIVE);
    assert_eq!(
        PyramidPupilImageF32::process(&mut declared, &image, &mut pixels, &mut mean),
        error(PyramidPixelError::NonFiniteInput {
            pupil: 3,
            pixel: 79,
        }),
    );
    assert_eq!(pixels, [SENTINEL; 316]);
    assert_eq!(mean, [SENTINEL]);
    assert_eq!(declared.workspace(), &committed);

    image[319] = f32::MIN_POSITIVE;
    let index = FAILURE - 1; // One unselected pixel in the preceding pupil.
    assert_eq!(
        PyramidPupilImageF32::process(&mut declared, &image, &mut pixels, &mut mean),
        error(PyramidPixelError::NonFiniteOutput { index }),
    );
    assert_eq!(pixels[..index], [1.0; FAILURE - 1]);
    assert_eq!(pixels[index..], [SENTINEL; 316 - FAILURE + 1]);
    assert_eq!(mean, [SENTINEL]);
    assert_eq!(declared.workspace(), &committed);
}

#[cfg(feature = "blas")]
#[test]
fn selected_runs_use_the_small_pupil_gain_from_the_previous_frame() {
    let mut pupil = PyramidPupilImageF32::prepare(PyramidPupilImageF32Configuration {
        image_size: (8, 4),
        pupil_size: (2, 4),
        pupil_origins: vec![[0, 0], [2, 0], [4, 0], [6, 0]],
        pupil_mask: Some(vec![true; 8]),
    })
    .unwrap();
    let mut image = [1.0; 32];
    image[0] = 16_777_216.0;
    let mut selected = [0.0; 32];
    let mut mean = [0.0];
    PyramidPupilImageF32::process(&mut pupil, &image, &mut selected, &mut mean).unwrap();
    let expected_mean = if filtergraph_blas::sgemv_default_float_mode() {
        524_289.0
    } else {
        524_288.0
    };
    assert_eq!(mean, [expected_mean]);
    image.fill(1.0);
    PyramidPupilImageF32::process(&mut pupil, &image, &mut selected, &mut mean).unwrap();
    assert_eq!(selected, [1.0 / expected_mean; 32]);
    assert_eq!(mean, [1.0]);
}
