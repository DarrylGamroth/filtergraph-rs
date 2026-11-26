//! Steady-state heap-allocation checks for qualified CPU processing paths.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use filtergraph_algorithms::schemas::{CONTROLLER_RESIDUAL_ERROR_V1, REQUESTED_PDM_COMMAND_V1};
use filtergraph_algorithms::{
    AlgorithmInput, AlgorithmPlan, CalibrationProbeOutput, CalibrationProbeStep,
    CenterOfGravityF32, CenterOfGravityF32Configuration, CenterOfGravityOutput,
    CenterOfGravityPlan, ChoppedImageDifferenceF32, ChoppedImageDifferenceF32Configuration,
    ChoppedImageDifferenceInput, ChoppedImageDifferenceOutput, ChoppedImageDifferencePlan,
    ChoppedImageDifferenceStatus, ChoppedImageKind, ClosedLoopCorrectionF32,
    ClosedLoopCorrectionF32Configuration, ClosedLoopCorrectionInput, ClosedLoopCorrectionPlan,
    ControllerProjection, ControllerToVdmF32, ControllerToVdmF32Configuration, ControllerToVdmPlan,
    DeformableMirrorCommandInput, DeformableMirrorCommandOutput, DeformableMirrorCommandPlan,
    DetectorFrameAssemblerPlan, DetectorReadoutBlockInfo, DetectorReadoutCalibrationPlan,
    DetectorReadoutPlan, DetectorReadoutRegion, DocrimeBinaryExcitationPlan,
    DocrimeClosedLoopInteractionMatrixPlan, DocrimeConditioningCriteria, DocrimeCovarianceF32,
    DocrimeCovarianceF32Configuration, DocrimeCovarianceInput, DocrimeCovariancePlan,
    DocrimeCovariances, DocrimeExcitationPlan, DocrimeInteractionMatrixDiagnostics,
    DocrimeInteractionMatrixOutput, DocrimeOpenLoopInteractionMatrixPlan,
    HadamardInteractionMatrixPlan, HadamardPushPullPlan, InputProgress, InterActuatorStrokeLimit,
    InterActuatorStrokeLimitPlan, InteractionMatrixStabilityCriteria,
    InteractionMatrixStabilityDiagnostics, InteractionMatrixStabilityInput,
    InteractionMatrixStabilityPlan, LeakyIntegratorF32, LeakyIntegratorF32Configuration,
    LeakyIntegratorPlan, LeakyIntegratorWorkspace, MatchedCalibrationDifferencingInput,
    MatchedCalibrationDifferencingOutput, MatchedCalibrationDifferencingPlan,
    MatchedCalibrationDifferencingStatus, ModalPushPullPlan, NdarrayCopyF32,
    NdarrayCopyF32Configuration, NoWorkspace, OpticalGainAssignment, OpticalGainCorrectionPlan,
    OpticalGainDitherF32, OpticalGainDitherF32Configuration, OpticalGainDitherMode,
    OpticalGainDitherPlan, OpticalGainDitherSchedule, OpticalGainEstimateOutput,
    OpticalGainEstimatorF32, OpticalGainEstimatorF32Configuration, OpticalGainEstimatorInput,
    OpticalGainEstimatorPlan, OpticalGainUpdateCriteria, OpticalGainUpdateF32,
    OpticalGainUpdateF32Configuration, OpticalGainUpdateInput, OpticalGainUpdateOutput,
    OpticalGainUpdatePlan, PdmCommandInput, PdmCommandOutput, PdmCommandPlan,
    PdmCommandPowerLimitPlan, PdmEngineeringOffsetF32, PdmEngineeringOffsetF32Configuration,
    PdmFeedbackToVdmF32, PdmFeedbackToVdmF32Configuration, PdmFeedbackToVdmPlan, PdmProjection,
    PdmStrokeOptimizationPlan, PixelCalibrationPlan, PixelCalibrationU16F32,
    PixelCalibrationU16F32Configuration, PixelSelectionPlan, PreparedGemv, PropertyAssignment,
    PropertyBinding, PropertyValue, PseudoOpenLoopReconstructionInput,
    PseudoOpenLoopReconstructionOutput, PseudoOpenLoopReconstructionPlan, PyramidPixelImagePlan,
    PyramidPixelOutput, PyramidPixelPlan, PyramidReadoutReconstructionOutput,
    PyramidReadoutReconstructionPlan, RegionExtractionPlan, RegionPixels, RegionPixelsMut,
    SccPushPullResponseInput, SccPushPullResponsePlan, ShackHartmannImagePlan, ShackHartmannOutput,
    ShackHartmannPlan, ShackHartmannReadoutReconstructionPlan, VdmExtrapolationStorage,
    VdmFeedbackToControllerF32, VdmFeedbackToControllerF32Configuration,
    VdmFeedbackToControllerPlan, VdmToPdmF32, VdmToPdmF32Configuration, VdmToPdmPlan,
    VdmToPdmProjection, ZonalInteractionMatrixPlan, ZonalPushPullPlan,
};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if COUNTING.load(Ordering::Relaxed) && !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if COUNTING.load(Ordering::Relaxed) && !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if COUNTING.load(Ordering::Relaxed) && !new_pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        new_pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations(operation: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    DEALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    operation();
    COUNTING.store(false, Ordering::Relaxed);
    assert_eq!(DEALLOCATIONS.load(Ordering::Relaxed), 0);
    ALLOCATIONS.load(Ordering::Relaxed)
}

#[test]
fn warmed_cpu_processing_paths_allocate_no_heap_storage() {
    let mut copy = NdarrayCopyF32::prepare(NdarrayCopyF32Configuration {
        extent: 277,
        schema: "pipewireao/dm-command-opd-um-f32/1".into(),
    })
    .unwrap();
    let input = [f32::from_bits(0x7fc00001); 277];
    let mut output = [0.0; 277];
    NdarrayCopyF32::process(&mut copy, &input, &mut output).unwrap();
    assert_eq!(
        allocations(|| {
            NdarrayCopyF32::process(&mut copy, &input, &mut output).unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| {
            assert!(NdarrayCopyF32::process(&mut copy, &input[..276], &mut output).is_err());
        }),
        0
    );
    let mut engineering = PdmEngineeringOffsetF32::prepare(PdmEngineeringOffsetF32Configuration {
        actuator_count: 277,
    })
    .unwrap();
    let mut requested_offset = [1.0_f32; 277];
    let mut offset_command = [0.0; 277];
    let mut current_offset = [0.0; 277];
    for offset in [[0.0_f32; 277], [0.5_f32; 277]] {
        engineering
            .replace_parameter(3, AlgorithmInput::from(&offset[..]))
            .unwrap()
            .unwrap();
        PdmEngineeringOffsetF32::process(
            &mut engineering,
            &requested_offset,
            &mut offset_command,
            &mut current_offset,
        )
        .unwrap();
        assert_eq!(
            allocations(|| {
                PdmEngineeringOffsetF32::process(
                    &mut engineering,
                    &requested_offset,
                    &mut offset_command,
                    &mut current_offset,
                )
                .unwrap();
            }),
            0
        );
    }
    requested_offset[276] = f32::NAN;
    assert_eq!(
        allocations(|| {
            assert!(
                PdmEngineeringOffsetF32::process(
                    &mut engineering,
                    &requested_offset,
                    &mut offset_command,
                    &mut current_offset
                )
                .is_err()
            );
        }),
        0
    );
    requested_offset[276] = f32::MAX;
    engineering
        .replace_parameter(3, AlgorithmInput::from(&[f32::MAX; 277][..]))
        .unwrap()
        .unwrap();
    assert_eq!(
        allocations(|| {
            assert!(
                PdmEngineeringOffsetF32::process(
                    &mut engineering,
                    &requested_offset,
                    &mut offset_command,
                    &mut current_offset
                )
                .is_err()
            );
        }),
        0
    );

    let calibration = PixelCalibrationPlan::<f32, u16>::new(vec![1.0; 8], vec![0.0; 8]).unwrap();
    let mut calibration_workspace = calibration.workspace();
    let raw = [1_u16; 8];
    let mut calibrated = [0.0_f32; 8];
    calibration
        .process(&mut calibrated, &mut calibration_workspace, &raw)
        .unwrap();
    assert_eq!(
        allocations(|| {
            calibration
                .process(&mut calibrated, &mut calibration_workspace, &raw)
                .unwrap();
        }),
        0
    );

    assert_eq!(
        allocations(|| {
            calibration.start(&mut calibration_workspace).unwrap();
            let _ = calibration
                .process_range(&mut calibrated, &mut calibration_workspace, &raw, 0..8)
                .unwrap();
            calibration
                .finish(&mut calibrated, &mut calibration_workspace)
                .unwrap();
        }),
        0
    );
    let mut calibration_window = [0.0_f32; 4];
    assert_eq!(
        allocations(|| {
            calibration.start(&mut calibration_workspace).unwrap();
            let _ = calibration
                .process_range_window(
                    &mut calibration_window,
                    &mut calibration_workspace,
                    &raw,
                    0..4,
                )
                .unwrap();
            let _ = calibration
                .process_range_window(
                    &mut calibration_window,
                    &mut calibration_workspace,
                    &raw,
                    4..8,
                )
                .unwrap();
            calibration
                .finish(&mut [], &mut calibration_workspace)
                .unwrap();
        }),
        0
    );
    let mut declared_calibration =
        PixelCalibrationU16F32::prepare(PixelCalibrationU16F32Configuration {
            initial_flat: Some(vec![1.0; 8]),
            initial_background: Some(vec![0.0; 8]),
            image_size: (2, 4),
        })
        .unwrap();
    PixelCalibrationU16F32::process(&mut declared_calibration, &raw, &mut calibrated).unwrap();
    assert_eq!(
        allocations(|| {
            PixelCalibrationU16F32::process(&mut declared_calibration, &raw, &mut calibrated)
                .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| PixelCalibrationU16F32::reset(&mut declared_calibration)),
        0
    );

    let readout = DetectorReadoutPlan::new(
        (4, 4),
        2,
        1,
        4,
        [
            DetectorReadoutRegion::new((0, 0), (1, 0), (0, 1)),
            DetectorReadoutRegion::new((3, 0), (-1, 0), (0, 1)),
        ],
    )
    .unwrap();
    let readout_raw = [2_u16; 16];
    let packed_raw = readout.pack_detector(&readout_raw).unwrap();
    let readout_calibration =
        DetectorReadoutCalibrationPlan::new(readout.clone(), &[1.0_f32; 16], &[0.0_f32; 16])
            .unwrap();
    let assembler = DetectorFrameAssemblerPlan::new(readout.clone());
    let mut assembler_workspace = assembler.workspace::<f32>();
    let mut readout_block = [0.0_f32; 8];
    let mut assembled = [0.0_f32; 16];
    let readout_sensing = ShackHartmannImagePlan::from_detector_regions(
        (4, 4),
        (1, 2),
        &[(0, 0), (1, 0), (2, 0), (3, 0)],
        1.0,
        0.0,
        0.0,
        None,
        None,
    )
    .unwrap();
    let readout_reconstructor = PreparedGemv::new(1, 8, Arc::from([1.0_f32; 8])).unwrap();
    let readout_shack_hartmann = ShackHartmannReadoutReconstructionPlan::new(
        readout_sensing,
        readout_reconstructor,
        readout.clone(),
    )
    .unwrap();
    let mut readout_shack_hartmann_workspace = readout_shack_hartmann.workspace();
    let mut reconstructed = [0.0_f32; 1];
    let readout_pyramid_image = PyramidPixelImagePlan::from_detector_pupils(
        (4, 4),
        (1, 1),
        &[(3, 0), (0, 0), (2, 0), (1, 0)],
        vec![true],
    )
    .unwrap();
    let readout_pyramid = PyramidReadoutReconstructionPlan::new(
        readout_pyramid_image,
        PreparedGemv::new(1, 4, Arc::from([1.0_f32; 4])).unwrap(),
        readout.clone(),
    )
    .unwrap();
    let mut readout_pyramid_workspace = readout_pyramid.workspace();
    let mut pyramid_pixels = [0.0_f32; 4];
    let mut pyramid_mean = 0.0_f32;
    let mut pyramid_reconstructed = [0.0_f32; 1];
    assert_eq!(
        allocations(|| {
            for ordinal in 0..readout.blocks() {
                let start = ordinal * readout.block_elements();
                let info = DetectorReadoutBlockInfo {
                    sequence: 1,
                    ordinal,
                    terminal: ordinal + 1 == readout.blocks(),
                    discontinuity: false,
                    corrupted: false,
                };
                readout_calibration
                    .process_block(
                        &mut readout_block,
                        &packed_raw[start..start + readout.block_elements()],
                        info,
                    )
                    .unwrap();
                assembler
                    .process_block(
                        &mut assembled,
                        &mut assembler_workspace,
                        &readout_block,
                        info,
                    )
                    .unwrap();
                readout_shack_hartmann
                    .process_block(
                        &mut reconstructed,
                        &mut readout_shack_hartmann_workspace,
                        &readout_block,
                        info,
                    )
                    .unwrap();
                readout_pyramid
                    .process_block(
                        PyramidReadoutReconstructionOutput::new(
                            &mut pyramid_pixels,
                            &mut pyramid_mean,
                            &mut pyramid_reconstructed,
                        ),
                        &mut readout_pyramid_workspace,
                        &readout_block,
                        info,
                    )
                    .unwrap();
            }
        }),
        0
    );

    let chopped_difference = ChoppedImageDifferencePlan::new(8).unwrap();
    let mut chopped_workspace = chopped_difference.workspace();
    let mut chopped_pixels = [0.0_f32; 8];
    let mut chopped_validity = [false; 8];
    let mut chopped_status = ChoppedImageDifferenceStatus::Priming;
    chopped_difference
        .process(
            ChoppedImageDifferenceOutput::new(
                &mut chopped_pixels,
                &mut chopped_validity,
                &mut chopped_status,
            ),
            &mut chopped_workspace,
            ChoppedImageDifferenceInput::new(ChoppedImageKind::Fringed, &calibrated, &[true; 8]),
        )
        .unwrap();
    assert_eq!(
        allocations(|| {
            chopped_difference
                .process(
                    ChoppedImageDifferenceOutput::new(
                        &mut chopped_pixels,
                        &mut chopped_validity,
                        &mut chopped_status,
                    ),
                    &mut chopped_workspace,
                    ChoppedImageDifferenceInput::new(
                        ChoppedImageKind::Unfringed,
                        &calibrated,
                        &[true; 8],
                    ),
                )
                .unwrap();
        }),
        0
    );
    let mut declared_chopped =
        ChoppedImageDifferenceF32::prepare(ChoppedImageDifferenceF32Configuration {
            input_schema: filtergraph_algorithms::schemas::CALIBRATED_PIXELS_V1.into(),
            require_acquisition_identity: false,
            pixel_count: 8,
            trigger_high_is_fringed: true,
        })
        .unwrap();
    let mut qualified_chopped =
        ChoppedImageDifferenceF32::prepare(ChoppedImageDifferenceF32Configuration {
            pixel_count: 8,
            trigger_high_is_fringed: true,
            input_schema: "test.image/1".into(),
            require_acquisition_identity: true,
        })
        .unwrap();
    let qualified_sample = |sequence, kind| {
        filtergraph_algorithms::SampleMetadata::new(100 - sequence, 0)
            .with_acquisition_identity(Some(
                filtergraph_algorithms::SampleAcquisitionIdentity::new([1; 16], 1, sequence)
                    .unwrap(),
            ))
            .with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(20))
            .with_chopped_image_kind(Some(kind))
    };
    let mut qualified_status = [0_u8];
    for (sequence, kind) in [
        (1, ChoppedImageKind::Fringed),
        (2, ChoppedImageKind::Unfringed),
        (3, ChoppedImageKind::Mixed),
    ] {
        assert_eq!(
            allocations(|| {
                ChoppedImageDifferenceF32::process_with_metadata(
                    &mut qualified_chopped,
                    &[Some(qualified_sample(sequence, kind)), None],
                    &calibrated,
                    &[true; 8],
                    &mut chopped_pixels,
                    &mut chopped_validity,
                    &mut qualified_status,
                )
                .unwrap();
            }),
            0
        );
    }
    let mut declared_chopped_status = [0_u8];
    ChoppedImageDifferenceF32::process_with_metadata(
        &mut declared_chopped,
        &[
            Some(
                filtergraph_algorithms::SampleMetadata::new(1, 0)
                    .with_trigger_state(Some(filtergraph_algorithms::SampleTriggerState::High)),
            ),
            None,
        ],
        &calibrated,
        &[true; 8],
        &mut chopped_pixels,
        &mut chopped_validity,
        &mut declared_chopped_status,
    )
    .unwrap();
    assert_eq!(
        allocations(|| {
            ChoppedImageDifferenceF32::process_with_metadata(
                &mut declared_chopped,
                &[
                    Some(
                        filtergraph_algorithms::SampleMetadata::new(2, 0).with_trigger_state(Some(
                            filtergraph_algorithms::SampleTriggerState::Low,
                        )),
                    ),
                    None,
                ],
                &calibrated,
                &[true; 8],
                &mut chopped_pixels,
                &mut chopped_validity,
                &mut declared_chopped_status,
            )
            .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| ChoppedImageDifferenceF32::reset(&mut declared_chopped)),
        0
    );

    let extraction = RegionExtractionPlan::new((2, 4), &[(0, 0), (0, 2)], (2, 2)).unwrap();
    let mut extraction_workspace = extraction.workspace();
    let mut subaperture_pixels = [0.0_f32; 8];
    extraction
        .process(
            RegionPixelsMut::new(&mut subaperture_pixels, 4, 2).unwrap(),
            &mut extraction_workspace,
            &calibrated,
        )
        .unwrap();
    assert_eq!(
        allocations(|| {
            extraction
                .process(
                    RegionPixelsMut::new(&mut subaperture_pixels, 4, 2).unwrap(),
                    &mut extraction_workspace,
                    &calibrated,
                )
                .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| {
            extraction.start(&mut extraction_workspace).unwrap();
            let _ = extraction
                .process_range(
                    RegionPixelsMut::new(&mut subaperture_pixels, 4, 2).unwrap(),
                    &mut extraction_workspace,
                    &calibrated,
                    0..1,
                )
                .unwrap();
            let _ = extraction
                .process_range(
                    RegionPixelsMut::new(&mut subaperture_pixels, 4, 2).unwrap(),
                    &mut extraction_workspace,
                    &calibrated,
                    1..2,
                )
                .unwrap();
            extraction
                .finish(
                    RegionPixelsMut::new(&mut subaperture_pixels, 4, 2).unwrap(),
                    &mut extraction_workspace,
                )
                .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| {
            calibration
                .process(&mut calibrated, &mut calibration_workspace, &raw)
                .unwrap();
            extraction
                .process(
                    RegionPixelsMut::new(&mut subaperture_pixels, 4, 2).unwrap(),
                    &mut extraction_workspace,
                    &calibrated,
                )
                .unwrap();
        }),
        0
    );

    let selection =
        PixelSelectionPlan::new((2, 4), [true, false, true, false, false, true, false, true])
            .unwrap();
    let mut selection_workspace = selection.workspace();
    let mut selected_pixels = [0.0_f32; 4];
    selection
        .process(&mut selected_pixels, &mut selection_workspace, &calibrated)
        .unwrap();
    assert_eq!(
        allocations(|| {
            selection
                .process(&mut selected_pixels, &mut selection_workspace, &calibrated)
                .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| {
            selection.start(&mut selection_workspace).unwrap();
            let _ = selection
                .process_range(
                    &mut selected_pixels,
                    &mut selection_workspace,
                    &calibrated,
                    0..2,
                )
                .unwrap();
            let _ = selection
                .process_range(
                    &mut selected_pixels,
                    &mut selection_workspace,
                    &calibrated,
                    2..4,
                )
                .unwrap();
            selection
                .finish(&mut selected_pixels, &mut selection_workspace)
                .unwrap();
        }),
        0
    );

    let image = [1.0_f32; 65];
    let mut selected = [0.0_f32; 64];
    for row_stride in [32, 33] {
        let selection =
            PixelSelectionPlan::with_layout((2, 32), [true; 64], row_stride, 0).unwrap();
        let mut workspace = selection.workspace();
        selection
            .process(&mut selected, &mut workspace, &image)
            .unwrap();
        assert_eq!(
            allocations(|| {
                selection
                    .process(&mut selected, &mut workspace, &image)
                    .unwrap();
            }),
            0
        );
    }

    let center_of_gravity = CenterOfGravityPlan::new(
        [-1.0_f32, 0.0, 1.0, 0.0],
        [0.0_f32, -1.0, 0.0, 1.0],
        [0.0; 2],
        None,
        0.0,
    )
    .unwrap();
    let mut centroid_offset = [0.0_f32; 2];
    let mut centroid_flux = 0.0_f32;
    let mut centroid_validity = false;
    center_of_gravity
        .process(
            CenterOfGravityOutput::new(
                &mut centroid_offset,
                &mut centroid_flux,
                &mut centroid_validity,
            ),
            &mut center_of_gravity.workspace(),
            &selected_pixels,
        )
        .unwrap();
    assert_eq!(
        allocations(|| {
            center_of_gravity
                .process(
                    CenterOfGravityOutput::new(
                        &mut centroid_offset,
                        &mut centroid_flux,
                        &mut centroid_validity,
                    ),
                    &mut center_of_gravity.workspace(),
                    &selected_pixels,
                )
                .unwrap();
        }),
        0
    );
    let mut declared_center = CenterOfGravityF32::prepare(CenterOfGravityF32Configuration {
        pixel_count: 4,
        initial_coordinates: Some(vec![[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
    })
    .unwrap();
    let center_coordinates = [-1.0_f32, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0];
    declared_center
        .replace_parameter(4, AlgorithmInput::from(&center_coordinates[..]))
        .unwrap()
        .unwrap();
    let mut declared_centroid_offset = [0.0_f32; 2];
    let mut declared_centroid_flux = [0.0_f32];
    let mut declared_centroid_validity = [false];
    CenterOfGravityF32::process(
        &mut declared_center,
        &selected_pixels,
        &mut declared_centroid_offset,
        &mut declared_centroid_flux,
        &mut declared_centroid_validity,
    )
    .unwrap();
    assert_eq!(
        allocations(|| {
            CenterOfGravityF32::process(
                &mut declared_center,
                &selected_pixels,
                &mut declared_centroid_offset,
                &mut declared_centroid_flux,
                &mut declared_centroid_validity,
            )
            .unwrap();
        }),
        0
    );

    let shack_hartmann = ShackHartmannPlan::new(
        vec![-0.5_f32, 0.5, -0.5, 0.5],
        vec![-0.5_f32, -0.5, 0.5, 0.5],
        vec![0.0_f32; 4],
        vec![0.0_f32; 2],
        vec![0.0_f32; 2],
        vec![true; 2],
    )
    .unwrap();
    let mut shack_hartmann_workspace = shack_hartmann.workspace();
    let mut slopes = [0.0_f32; 4];
    let mut flux = [0.0_f32; 2];
    let mut validity = [false; 2];
    shack_hartmann
        .process(
            ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
            &mut shack_hartmann_workspace,
            RegionPixels::new(&subaperture_pixels, 4, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(
        allocations(|| {
            shack_hartmann
                .process(
                    ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
                    &mut shack_hartmann_workspace,
                    RegionPixels::new(&subaperture_pixels, 4, 2).unwrap(),
                )
                .unwrap();
        }),
        0
    );
    let shack_hartmann_regions =
        RegionExtractionPlan::new((2, 4), &[(0, 0), (0, 2)], (2, 2)).unwrap();
    let shack_hartmann_image = [1.0_f32; 8];
    assert_eq!(
        allocations(|| {
            shack_hartmann
                .process_image_regions(
                    ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
                    &mut shack_hartmann_workspace,
                    shack_hartmann_regions
                        .image_regions(&shack_hartmann_image)
                        .unwrap(),
                )
                .unwrap();
        }),
        0
    );
    let shack_hartmann_image_plan =
        ShackHartmannImagePlan::new(shack_hartmann_regions, shack_hartmann.clone()).unwrap();
    let mut shack_hartmann_image_workspace = shack_hartmann_image_plan.workspace();
    assert_eq!(
        allocations(|| {
            shack_hartmann_image_plan
                .process(
                    ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
                    &mut shack_hartmann_image_workspace,
                    &shack_hartmann_image,
                )
                .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| {
            shack_hartmann.start(&mut shack_hartmann_workspace).unwrap();
            let _ = shack_hartmann
                .process_range(
                    ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
                    &mut shack_hartmann_workspace,
                    RegionPixels::new(&subaperture_pixels, 4, 2).unwrap(),
                    0..1,
                )
                .unwrap();
            let _ = shack_hartmann
                .process_range(
                    ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
                    &mut shack_hartmann_workspace,
                    RegionPixels::new(&subaperture_pixels, 4, 2).unwrap(),
                    1..2,
                )
                .unwrap();
            shack_hartmann
                .finish(
                    ShackHartmannOutput::new(&mut slopes, &mut flux, &mut validity).unwrap(),
                    &mut shack_hartmann_workspace,
                )
                .unwrap();
        }),
        0
    );

    let pyramid_pixel = PyramidPixelPlan::new([true, false, true, true]).unwrap();
    let mut pyramid_workspace = pyramid_pixel.workspace();
    let pupil_pixels = [1.0_f32; 16];
    let mut normalized_pixels = [0.0_f32; 12];
    let mut mean_pupil_intensity = 0.0_f32;
    pyramid_pixel
        .process(
            PyramidPixelOutput::new(&mut normalized_pixels, &mut mean_pupil_intensity),
            &mut pyramid_workspace,
            RegionPixels::new(&pupil_pixels, 4, 4).unwrap(),
        )
        .unwrap();
    assert_eq!(
        allocations(|| {
            pyramid_pixel
                .process(
                    PyramidPixelOutput::new(&mut normalized_pixels, &mut mean_pupil_intensity),
                    &mut pyramid_workspace,
                    RegionPixels::new(&pupil_pixels, 4, 4).unwrap(),
                )
                .unwrap();
        }),
        0
    );
    let pyramid_regions =
        RegionExtractionPlan::new((4, 4), &[(0, 0), (0, 2), (2, 0), (2, 2)], (2, 2)).unwrap();
    let pyramid_image = [1.0_f32; 16];
    assert_eq!(
        allocations(|| {
            pyramid_pixel
                .process_image_regions(
                    PyramidPixelOutput::new(&mut normalized_pixels, &mut mean_pupil_intensity),
                    &mut pyramid_workspace,
                    pyramid_regions.image_regions(&pyramid_image).unwrap(),
                )
                .unwrap();
        }),
        0
    );
    let pyramid_image_plan =
        PyramidPixelImagePlan::new(pyramid_regions, pyramid_pixel.clone()).unwrap();
    let mut pyramid_image_workspace = pyramid_image_plan.workspace();
    assert_eq!(
        allocations(|| {
            pyramid_image_plan
                .process(
                    PyramidPixelOutput::new(&mut normalized_pixels, &mut mean_pupil_intensity),
                    &mut pyramid_image_workspace,
                    &pyramid_image,
                )
                .unwrap();
        }),
        0
    );

    let matrix_vector = PreparedGemv::new(2, 2, Arc::from([1.0_f32, 2.0, 3.0, 4.0])).unwrap();
    let mut matrix_output = [0.0_f32; 2];
    matrix_vector
        .multiply(&mut matrix_output, &[0.5, -0.5])
        .unwrap();
    assert_eq!(
        allocations(|| {
            matrix_vector
                .multiply(&mut matrix_output, &[0.5, -0.5])
                .unwrap();
        }),
        0
    );

    let integrator = LeakyIntegratorPlan::new(0.5_f32, 0.75).unwrap();
    let mut integrator_workspace = LeakyIntegratorWorkspace::new(4).unwrap();
    let input = [2.0_f32; 4];
    let mut output = [0.0_f32; 4];
    integrator
        .process(&mut output, &mut integrator_workspace, &input)
        .unwrap();
    assert_eq!(
        allocations(|| {
            integrator
                .process(&mut output, &mut integrator_workspace, &input)
                .unwrap();
        }),
        0
    );
    let nonfinite_input = [2.0_f32, 2.0, 2.0, f32::NAN];
    assert_eq!(
        allocations(|| {
            assert!(
                integrator
                    .process(&mut output, &mut integrator_workspace, &nonfinite_input)
                    .is_err()
            );
        }),
        0
    );
    let overflow_integrator = LeakyIntegratorPlan::new(2.0_f32, 1.0).unwrap();
    assert_eq!(
        allocations(|| {
            assert!(
                overflow_integrator
                    .process(&mut output, &mut integrator_workspace, &[f32::MAX; 4])
                    .is_err()
            );
        }),
        0
    );
    assert_eq!(allocations(|| integrator_workspace.reset()), 0);
    let mut declared_integrator = LeakyIntegratorF32::prepare(LeakyIntegratorF32Configuration {
        extent: 4,
        initial_state: 0.0,
        input_schema: CONTROLLER_RESIDUAL_ERROR_V1.into(),
        output_schema: REQUESTED_PDM_COMMAND_V1.into(),
    })
    .unwrap();
    LeakyIntegratorF32::process(&mut declared_integrator, &input, &mut output).unwrap();
    assert_eq!(
        allocations(|| {
            LeakyIntegratorF32::process(&mut declared_integrator, &input, &mut output).unwrap();
        }),
        0
    );
    let mut integrator_properties = PropertyBinding::new(integrator);
    integrator_properties
        .prepare_properties(&[PropertyAssignment {
            id: 0,
            value: PropertyValue::Float(0.25),
        }])
        .unwrap();
    integrator_properties.commit_properties();
    assert_eq!(
        allocations(|| {
            assert!(integrator_properties.adopt_properties());
            integrator_properties
                .active_plan()
                .process(&mut output, &mut integrator_workspace, &input)
                .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| {
            integrator.start(&mut integrator_workspace).unwrap();
            let _ = integrator
                .process_range(&mut output, &mut integrator_workspace, &input, 0..2)
                .unwrap();
            let _ = integrator
                .process_range(&mut output, &mut integrator_workspace, &input, 2..4)
                .unwrap();
            integrator
                .finish(&mut output, &mut integrator_workspace)
                .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| {
            integrator.start(&mut integrator_workspace).unwrap();
            let _ = integrator
                .process_range(&mut output, &mut integrator_workspace, &input, 0..2)
                .unwrap();
            integrator.abort(&mut integrator_workspace).unwrap();
        }),
        0
    );

    let correction = ClosedLoopCorrectionPlan::new(-0.5_f32, 0.9, 0.9, 2).unwrap();
    let mut correction_workspace = correction.workspace();
    let mut correction_output = [0.0; 2];
    correction
        .process(
            &mut correction_output,
            &mut correction_workspace,
            ClosedLoopCorrectionInput {
                x: &[1.0, -1.0],
                constraint_feedback: None,
            },
        )
        .unwrap();
    assert_eq!(
        allocations(|| {
            correction
                .process(
                    &mut correction_output,
                    &mut correction_workspace,
                    ClosedLoopCorrectionInput {
                        x: &[1.0, -1.0],
                        constraint_feedback: Some(&[0.1, -0.1]),
                    },
                )
                .unwrap();
        }),
        0
    );

    let mut declared_correction =
        ClosedLoopCorrectionF32::prepare(ClosedLoopCorrectionF32Configuration {
            state_length: 2,
            hidden_mode_count: 1,
        })
        .unwrap();
    declared_correction
        .replace_parameter(4, AlgorithmInput::from(&[1.0_f32, -1.0][..]))
        .unwrap()
        .unwrap();
    declared_correction
        .replace_parameter(5, AlgorithmInput::from(&[0.5_f32, -0.5][..]))
        .unwrap()
        .unwrap();
    let residual_error = [1.0_f32, -1.0];
    let constraint_feedback = [0.1_f32, -0.1];
    let mut declared_correction_output = [0.0_f32; 2];
    let mut declared_controller_state = [0.0_f32; 2];
    ClosedLoopCorrectionF32::process(
        &mut declared_correction,
        &residual_error,
        &constraint_feedback,
        &mut declared_correction_output,
        &mut declared_controller_state,
    )
    .unwrap();
    assert_eq!(
        allocations(|| {
            ClosedLoopCorrectionF32::process(
                &mut declared_correction,
                &residual_error,
                &constraint_feedback,
                &mut declared_correction_output,
                &mut declared_controller_state,
            )
            .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| ClosedLoopCorrectionF32::reset(&mut declared_correction)),
        0
    );

    let pseudo_open_loop = PseudoOpenLoopReconstructionPlan::new(
        2,
        2,
        2,
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
    )
    .unwrap();
    let mut pseudo_open_loop_workspace = pseudo_open_loop.workspace();
    let mut pseudo_open_loop_measurements = [0.0; 2];
    let mut residual_error = [0.0; 2];
    pseudo_open_loop
        .process(
            PseudoOpenLoopReconstructionOutput::new(
                &mut pseudo_open_loop_measurements,
                &mut residual_error,
            ),
            &mut pseudo_open_loop_workspace,
            PseudoOpenLoopReconstructionInput::new(&[1.0, -1.0], &[0.25, -0.25]),
        )
        .unwrap();
    assert_eq!(
        allocations(|| {
            pseudo_open_loop
                .process(
                    PseudoOpenLoopReconstructionOutput::new(
                        &mut pseudo_open_loop_measurements,
                        &mut residual_error,
                    ),
                    &mut pseudo_open_loop_workspace,
                    PseudoOpenLoopReconstructionInput::new(&[1.0, -1.0], &[0.25, -0.25]),
                )
                .unwrap();
        }),
        0
    );

    let docrime_excitation = DocrimeExcitationPlan::new([0.1_f32, 0.2], 7).unwrap();
    let mut docrime_excitation_workspace = docrime_excitation.workspace();
    let mut calibration_excitation = [0.0; 2];
    docrime_excitation
        .process(
            &mut calibration_excitation,
            &mut docrime_excitation_workspace,
            (),
        )
        .unwrap();
    assert_eq!(
        allocations(|| {
            docrime_excitation
                .process(
                    &mut calibration_excitation,
                    &mut docrime_excitation_workspace,
                    (),
                )
                .unwrap();
        }),
        0
    );

    let docrime_binary_excitation = DocrimeBinaryExcitationPlan::new([0.1_f32, 0.2], 7).unwrap();
    let mut docrime_binary_workspace = docrime_binary_excitation.workspace();
    docrime_binary_excitation
        .process(
            &mut calibration_excitation,
            &mut docrime_binary_workspace,
            (),
        )
        .unwrap();
    assert_eq!(
        allocations(|| {
            docrime_binary_excitation
                .process(
                    &mut calibration_excitation,
                    &mut docrime_binary_workspace,
                    (),
                )
                .unwrap();
        }),
        0
    );

    let matched_differencing = MatchedCalibrationDifferencingPlan::new(2, 2).unwrap();
    let mut matched_differencing_workspace = matched_differencing.workspace();
    let mut filtered_measurements = [0.0_f32; 2];
    let mut filtered_corrector = [0.0_f32; 2];
    let mut filter_status = MatchedCalibrationDifferencingStatus::Priming;
    matched_differencing
        .process(
            MatchedCalibrationDifferencingOutput::new(
                &mut filtered_measurements,
                &mut filtered_corrector,
                &mut filter_status,
            ),
            &mut matched_differencing_workspace,
            MatchedCalibrationDifferencingInput::new(&[1.0, -1.0], &calibration_excitation),
        )
        .unwrap();
    assert_eq!(
        allocations(|| {
            matched_differencing
                .process(
                    MatchedCalibrationDifferencingOutput::new(
                        &mut filtered_measurements,
                        &mut filtered_corrector,
                        &mut filter_status,
                    ),
                    &mut matched_differencing_workspace,
                    MatchedCalibrationDifferencingInput::new(&[0.5, -0.5], &calibration_excitation),
                )
                .unwrap();
        }),
        0
    );

    let zonal_probe = ZonalPushPullPlan::new([0.1_f32, 0.2]).unwrap();
    let mut zonal_workspace = zonal_probe.workspace();
    let mut structured_excitation = [0.0; 2];
    let mut probe_step = CalibrationProbeStep::default();
    let modal_probe = ModalPushPullPlan::new(2, 2, [0.1_f32, -0.1, 0.2, 0.2]).unwrap();
    let mut modal_workspace = modal_probe.workspace();
    assert_eq!(
        allocations(|| {
            modal_probe
                .process(
                    CalibrationProbeOutput::new(&mut structured_excitation, &mut probe_step),
                    &mut modal_workspace,
                    (),
                )
                .unwrap();
        }),
        0
    );

    let scc_response = SccPushPullResponsePlan::new(2).unwrap();
    let mut scc_response_workspace = scc_response.workspace();
    let mut response = [0.0_f32; 2];
    assert_eq!(
        allocations(|| {
            scc_response
                .process(
                    &mut response,
                    &mut scc_response_workspace,
                    SccPushPullResponseInput::new(
                        &[2.0, 3.0],
                        &[1.0, 1.0],
                        &[0.0, 1.0],
                        &[0.0, 0.0],
                    ),
                )
                .unwrap();
        }),
        0
    );

    assert_eq!(
        allocations(|| {
            zonal_probe
                .process(
                    CalibrationProbeOutput::new(&mut structured_excitation, &mut probe_step),
                    &mut zonal_workspace,
                    (),
                )
                .unwrap();
        }),
        0
    );

    let hadamard_probe = HadamardPushPullPlan::new([0.1_f32, 0.2]).unwrap();
    let mut hadamard_workspace = hadamard_probe.workspace();
    assert_eq!(
        allocations(|| {
            hadamard_probe
                .process(
                    CalibrationProbeOutput::new(&mut structured_excitation, &mut probe_step),
                    &mut hadamard_workspace,
                    (),
                )
                .unwrap();
        }),
        0
    );

    let zonal_decoder = ZonalInteractionMatrixPlan::new(2, [0.1_f32, 0.2]).unwrap();
    let mut zonal_decoder_workspace = zonal_decoder.workspace();
    let zonal_responses = [1.0, -1.0, -1.0, 1.0, 2.0, -2.0, -2.0, 2.0];
    let mut structured_interaction_matrix = [0.0; 4];
    assert_eq!(
        allocations(|| {
            zonal_decoder
                .process(
                    &mut structured_interaction_matrix,
                    &mut zonal_decoder_workspace,
                    &zonal_responses,
                )
                .unwrap();
        }),
        0
    );

    let hadamard_decoder = HadamardInteractionMatrixPlan::new(2, [0.1_f32, 0.2]).unwrap();
    let mut hadamard_decoder_workspace = hadamard_decoder.workspace();
    let hadamard_responses = [
        1.0, -1.0, -1.0, 1.0, 2.0, -2.0, -2.0, 2.0, 1.0, -1.0, -1.0, 1.0, 2.0, -2.0, -2.0, 2.0,
    ];
    assert_eq!(
        allocations(|| {
            hadamard_decoder
                .process(
                    &mut structured_interaction_matrix,
                    &mut hadamard_decoder_workspace,
                    &hadamard_responses,
                )
                .unwrap();
        }),
        0
    );

    let stability = InteractionMatrixStabilityPlan::new(
        2,
        2,
        InteractionMatrixStabilityCriteria::new(0.1_f32, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let mut stability_diagnostics = InteractionMatrixStabilityDiagnostics::default();
    assert_eq!(
        allocations(|| {
            stability
                .process(
                    &mut stability_diagnostics,
                    &mut NoWorkspace,
                    InteractionMatrixStabilityInput::new(
                        &[1.0, 0.0, 0.0, 1.0],
                        &[1.0, 0.0, 0.0, 1.0],
                    ),
                )
                .unwrap();
        }),
        0
    );

    let docrime_covariance = DocrimeCovariancePlan::with_sample_limit(2, 2, 3).unwrap();
    let mut docrime_covariance_workspace = docrime_covariance.workspace();
    docrime_covariance
        .process(
            (),
            &mut docrime_covariance_workspace,
            DocrimeCovarianceInput::new(&[1.0, -1.0], &calibration_excitation),
        )
        .unwrap();
    assert_eq!(
        allocations(|| {
            docrime_covariance
                .process(
                    (),
                    &mut docrime_covariance_workspace,
                    DocrimeCovarianceInput::new(&[0.5, -0.5], &calibration_excitation),
                )
                .unwrap();
        }),
        0
    );
    let mut declared_docrime_covariance =
        DocrimeCovarianceF32::prepare(DocrimeCovarianceF32Configuration {
            measurement_count: 2,
            sample_limit: Some(3),
            measurements_schema: "test.filtergraph/test-measurement/1".into(),
            corrector_sample_schema: "test.filtergraph/test-corrector-sample/1".into(),
            cross_covariance_schema: "test.filtergraph/test-cross-covariance/1".into(),
            corrector_covariance_schema: "test.filtergraph/test-corrector-covariance/1".into(),
            corrector_coordinate_count: 2,
        })
        .unwrap();
    let mut declared_cross_covariance = [0.0_f32; 4];
    let mut declared_corrector_covariance = [0.0_f32; 4];
    let mut declared_sample_count = [0_u64];
    let mut declared_complete = [false];
    DocrimeCovarianceF32::process(
        &mut declared_docrime_covariance,
        &[1.0, -1.0],
        &calibration_excitation,
        &mut declared_cross_covariance,
        &mut declared_corrector_covariance,
        &mut declared_sample_count,
        &mut declared_complete,
    )
    .unwrap();
    assert_eq!(
        allocations(|| {
            DocrimeCovarianceF32::process(
                &mut declared_docrime_covariance,
                &[0.5, -0.5],
                &calibration_excitation,
                &mut declared_cross_covariance,
                &mut declared_corrector_covariance,
                &mut declared_sample_count,
                &mut declared_complete,
            )
            .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| DocrimeCovarianceF32::reset(&mut declared_docrime_covariance)),
        0
    );

    let docrime_criteria = DocrimeConditioningCriteria::new(0.1_f32, 0.9, 0.01).unwrap();
    let docrime_open_loop =
        DocrimeOpenLoopInteractionMatrixPlan::new(2, 2, docrime_criteria).unwrap();
    let docrime_closed_loop =
        DocrimeClosedLoopInteractionMatrixPlan::new(2, 2, 0.5, docrime_criteria).unwrap();
    let mut docrime_open_loop_workspace = docrime_open_loop.workspace();
    let mut docrime_closed_loop_workspace = docrime_closed_loop.workspace();
    let docrime_cross_covariance_sum = [6.0_f32, 0.0, 8.0, 13.0];
    let docrime_corrector_covariance_sum = [4.0_f32, 2.0, 2.0, 4.0];
    let mut docrime_interaction_matrix = [0.0; 4];
    let mut docrime_diagnostics = DocrimeInteractionMatrixDiagnostics::default();
    assert_eq!(
        allocations(|| {
            docrime_open_loop
                .process(
                    DocrimeInteractionMatrixOutput::new(
                        &mut docrime_interaction_matrix,
                        &mut docrime_diagnostics,
                    ),
                    &mut docrime_open_loop_workspace,
                    DocrimeCovariances::new(
                        3,
                        &docrime_cross_covariance_sum,
                        &docrime_corrector_covariance_sum,
                    ),
                )
                .unwrap();
            docrime_closed_loop
                .process(
                    DocrimeInteractionMatrixOutput::new(
                        &mut docrime_interaction_matrix,
                        &mut docrime_diagnostics,
                    ),
                    &mut docrime_closed_loop_workspace,
                    DocrimeCovariances::new(
                        3,
                        &docrime_cross_covariance_sum,
                        &docrime_corrector_covariance_sum,
                    ),
                )
                .unwrap();
        }),
        0
    );

    let stroke_optimization = PdmStrokeOptimizationPlan::new(
        2,
        1,
        Arc::from([0.0_f32; 2]),
        Arc::from([0.5_f32, 0.5]),
        Arc::from([1.0_f32, 1.0]),
    )
    .unwrap();
    let mut stroke_workspace = stroke_optimization.workspace();
    let mut optimized_command = [0.0; 2];
    assert_eq!(
        allocations(|| {
            stroke_optimization
                .process(&mut optimized_command, &mut stroke_workspace, &[0.5, -0.25])
                .unwrap();
        }),
        0
    );

    let dm = DeformableMirrorCommandPlan::new(
        2,
        2,
        2,
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::from([-1.0_f32; 2]),
        Arc::from([1.0_f32; 2]),
        Some(Arc::from([0.2_f32; 2])),
    )
    .unwrap()
    .with_stroke_optimization(stroke_optimization.clone())
    .unwrap()
    .with_inter_actuator_stroke_limit(
        InterActuatorStrokeLimitPlan::new(
            2,
            Arc::from([InterActuatorStrokeLimit::new(0, 1, 0.25)]),
        )
        .unwrap(),
    )
    .unwrap()
    .with_command_power_limit(PdmCommandPowerLimitPlan::new(2, 0.02).unwrap())
    .unwrap();
    let mut dm_workspace = dm.workspace(&[0.0; 2]).unwrap();
    let mut requested = [0.0; 2];
    let mut demanded = [0.0; 2];
    let mut feedback = [0.0; 2];
    assert_eq!(
        allocations(|| {
            dm.process(
                DeformableMirrorCommandOutput {
                    requested: &mut requested,
                    demanded: &mut demanded,
                    vdm_constraint_feedback: &mut feedback,
                },
                &mut dm_workspace,
                DeformableMirrorCommandInput {
                    vdm_command: &[0.5, -0.5],
                    best_flat: &[0.0; 2],
                    pdm_offset: &[0.0; 2],
                },
            )
            .unwrap();
        }),
        0
    );

    let pdm_projections = Arc::from([
        PdmProjection::new(1, Arc::from([1.0_f32, 0.0]), Arc::from([1.0_f32, 0.0])).unwrap(),
        PdmProjection::new(1, Arc::from([0.0_f32, 1.0]), Arc::from([0.0_f32, 1.0])).unwrap(),
    ]);
    let decomposition = VdmToPdmPlan::new(
        2,
        2,
        Arc::from([1.0_f32, 0.0, 0.0, 1.0]),
        Arc::clone(&pdm_projections),
    )
    .unwrap();
    let decomposition_feedback =
        PdmFeedbackToVdmPlan::new(2, 2, Arc::from([1.0_f32, 0.0, 0.0, 1.0]), pdm_projections)
            .unwrap();
    let mut decomposition_workspace = decomposition.workspace();
    let mut decomposition_feedback_workspace = decomposition_feedback.workspace();
    let mut physical_correction = [0.0; 2];
    let mut vdm_feedback = [0.0; 2];
    assert_eq!(
        allocations(|| {
            decomposition
                .process(
                    &mut physical_correction,
                    &mut decomposition_workspace,
                    &[0.5, -0.5],
                )
                .unwrap();
            decomposition_feedback
                .process(
                    &mut vdm_feedback,
                    &mut decomposition_feedback_workspace,
                    &[0.1, -0.1],
                )
                .unwrap();
        }),
        0
    );

    let mut compact_matrix = vec![0.0_f32; 21 * 10];
    for column in 0..10 {
        compact_matrix[(2 * column + 1) * 10 + column] = 1.0;
        for row in (0..20).step_by(2) {
            compact_matrix[row * 10 + column] = 1.0;
        }
    }
    let compact = VdmToPdmPlan::from_vdm_to_pdm_projections(
        10,
        21,
        Arc::from(compact_matrix),
        Arc::from([VdmToPdmProjection::new(1, Arc::from(vec![1.0; 21])).unwrap()]),
    )
    .unwrap();
    assert_eq!(
        compact.extrapolation_storage(),
        VdmExtrapolationStorage::ControlledExtrapolation
    );
    assert_eq!(compact.extrapolation_coefficient_count(), 100);
    let mut compact_workspace = compact.workspace();
    let mut compact_output = [0.0];
    compact
        .process(&mut compact_output, &mut compact_workspace, &[1.0; 10])
        .unwrap();
    assert_eq!(
        allocations(|| {
            compact
                .process(&mut compact_output, &mut compact_workspace, &[1.0; 10])
                .unwrap();
        }),
        0
    );

    let controller_projections = Arc::from([
        ControllerProjection::new(1, Arc::from([1.0_f32, 0.0]), Arc::from([1.0_f32, 0.0])).unwrap(),
        ControllerProjection::new(1, Arc::from([0.0_f32, 1.0]), Arc::from([0.0_f32, 1.0])).unwrap(),
    ]);
    let command_sum = ControllerToVdmPlan::new(2, Arc::clone(&controller_projections)).unwrap();
    let controller_feedback_plan =
        VdmFeedbackToControllerPlan::new(2, controller_projections).unwrap();
    let mut command_sum_workspace = command_sum.workspace();
    let mut controller_feedback_workspace = NoWorkspace;
    let mut summed_command = [0.0; 2];
    let mut controller_feedback = [0.0; 2];
    assert_eq!(
        allocations(|| {
            command_sum
                .process(
                    &mut summed_command,
                    &mut command_sum_workspace,
                    &[0.5, -0.5],
                )
                .unwrap();
            controller_feedback_plan
                .process(
                    &mut controller_feedback,
                    &mut controller_feedback_workspace,
                    &[0.1, -0.1],
                )
                .unwrap();
        }),
        0
    );

    #[cfg(all(feature = "experimental-compact-projection", target_arch = "x86_64"))]
    {
        use filtergraph_algorithms::PdmToVdmProjection;

        let (active, full) = (256, 1024);
        let mut identity = vec![0.0_f32; full * full];
        for (index, row) in identity.chunks_exact_mut(full).enumerate() {
            row[index] = 1.0;
        }
        let mut full_to_active = vec![0.0; active * full];
        for row in full_to_active.chunks_exact_mut(full) {
            row[0] = 0.5;
            row[2] = 2.0;
        }
        let projection = PdmToVdmProjection::new(full, Arc::from(identity)).unwrap();
        let plan = PdmFeedbackToVdmPlan::from_pdm_to_vdm_projections(
            active,
            full,
            Arc::from(full_to_active),
            Arc::from([projection]),
        )
        .unwrap();
        let mut workspace = plan.workspace();
        let input = vec![1.0e-7; full];
        let mut output = vec![0.0; active];
        plan.process(&mut output, &mut workspace, &input).unwrap();
        assert_eq!(
            allocations(|| plan.process(&mut output, &mut workspace, &input).unwrap()),
            0
        );
    }

    let identity = [1.0_f32, 0.0, 0.0, 1.0];
    let mut declared_controller_projection =
        ControllerToVdmF32::prepare(ControllerToVdmF32Configuration {
            controller_command_count: 2,
            active_vdm_count: 2,
        })
        .unwrap();
    declared_controller_projection
        .replace_parameter(2, AlgorithmInput::from(&identity[..]))
        .unwrap()
        .unwrap();
    let mut declared_vdm_projection = VdmToPdmF32::prepare(VdmToPdmF32Configuration {
        active_vdm_count: 2,
        full_vdm_count: 2,
        physical_actuator_count: 2,
    })
    .unwrap();
    declared_vdm_projection
        .replace_parameter(3, AlgorithmInput::from(&identity[..]))
        .unwrap()
        .unwrap();
    declared_vdm_projection
        .replace_parameter(4, AlgorithmInput::from(&identity[..]))
        .unwrap()
        .unwrap();
    let mut declared_vdm_feedback =
        PdmFeedbackToVdmF32::prepare(PdmFeedbackToVdmF32Configuration {
            active_vdm_count: 2,
            full_vdm_count: 2,
            physical_actuator_count: 2,
        })
        .unwrap();
    declared_vdm_feedback
        .replace_parameter(2, AlgorithmInput::from(&identity[..]))
        .unwrap()
        .unwrap();
    declared_vdm_feedback
        .replace_parameter(3, AlgorithmInput::from(&identity[..]))
        .unwrap()
        .unwrap();
    let mut declared_controller_feedback =
        VdmFeedbackToControllerF32::prepare(VdmFeedbackToControllerF32Configuration {
            controller_command_count: 2,
            active_vdm_count: 2,
        })
        .unwrap();
    declared_controller_feedback
        .replace_parameter(2, AlgorithmInput::from(&identity[..]))
        .unwrap()
        .unwrap();
    let controller_command = [0.5_f32, -0.5];
    let mut declared_vdm_command = [0.0_f32; 2];
    let mut declared_requested_pdm_command = [0.0_f32; 2];
    let mut declared_vdm_constraint_feedback = [0.0_f32; 2];
    let mut declared_controller_constraint_feedback = [0.0_f32; 2];
    ControllerToVdmF32::process(
        &mut declared_controller_projection,
        &controller_command,
        &mut declared_vdm_command,
    )
    .unwrap();
    VdmToPdmF32::process(
        &mut declared_vdm_projection,
        &declared_vdm_command,
        &mut declared_requested_pdm_command,
        &mut [0u8; 2],
    )
    .unwrap();
    PdmFeedbackToVdmF32::process(
        &mut declared_vdm_feedback,
        &declared_requested_pdm_command,
        &mut declared_vdm_constraint_feedback,
    )
    .unwrap();
    VdmFeedbackToControllerF32::process(
        &mut declared_controller_feedback,
        &declared_vdm_constraint_feedback,
        &mut declared_controller_constraint_feedback,
    )
    .unwrap();
    assert_eq!(
        allocations(|| {
            ControllerToVdmF32::process(
                &mut declared_controller_projection,
                &controller_command,
                &mut declared_vdm_command,
            )
            .unwrap();
            VdmToPdmF32::process(
                &mut declared_vdm_projection,
                &declared_vdm_command,
                &mut declared_requested_pdm_command,
                &mut [0u8; 2],
            )
            .unwrap();
            PdmFeedbackToVdmF32::process(
                &mut declared_vdm_feedback,
                &declared_requested_pdm_command,
                &mut declared_vdm_constraint_feedback,
            )
            .unwrap();
            VdmFeedbackToControllerF32::process(
                &mut declared_controller_feedback,
                &declared_vdm_constraint_feedback,
                &mut declared_controller_constraint_feedback,
            )
            .unwrap();
        }),
        0
    );

    let pdm = PdmCommandPlan::new(
        Arc::from([-1.0_f32; 2]),
        Arc::from([1.0_f32; 2]),
        Some(Arc::from([0.2_f32; 2])),
    )
    .unwrap()
    .with_stroke_optimization(stroke_optimization)
    .unwrap()
    .with_inter_actuator_stroke_limit(
        InterActuatorStrokeLimitPlan::new(
            2,
            Arc::from([InterActuatorStrokeLimit::new(0, 1, 0.25)]),
        )
        .unwrap(),
    )
    .unwrap()
    .with_command_power_limit(PdmCommandPowerLimitPlan::new(2, 0.02).unwrap())
    .unwrap();
    let mut pdm_workspace = pdm.workspace(&[0.0; 2]).unwrap();
    let mut pdm_demanded = [0.0; 2];
    let mut pdm_feedback = [0.0; 2];
    assert_eq!(
        allocations(|| {
            pdm.process(
                PdmCommandOutput {
                    demanded: &mut pdm_demanded,
                    constraint_feedback: &mut pdm_feedback,
                },
                &mut pdm_workspace,
                PdmCommandInput {
                    requested: &[0.5, -0.5],
                },
            )
            .unwrap();
        }),
        0
    );

    let optical_gain_schedule = Arc::new(
        OpticalGainDitherSchedule::new([OpticalGainDitherMode::new(0.5_f32, 4, 1)]).unwrap(),
    );
    let optical_gain_dither = OpticalGainDitherPlan::new(Arc::clone(&optical_gain_schedule));
    let optical_gain_estimator =
        OpticalGainEstimatorPlan::new(optical_gain_schedule, [1.0]).unwrap();
    let mut declared_estimator =
        OpticalGainEstimatorF32::prepare(OpticalGainEstimatorF32Configuration {
            modes: vec![OpticalGainDitherMode::new(0.5, 4, 1)],
            initial_calibration_responses: vec![1.0],
        })
        .unwrap();
    let optical_gain_update = OpticalGainUpdatePlan::new(
        [OpticalGainUpdateCriteria::new(0.1, 0.05, 0.1, 2.0, 0.25)],
        [OpticalGainAssignment::DitherMode {
            dither_mode: 0,
            update_fraction: 0.5,
        }],
    )
    .unwrap();
    let mut declared_update = OpticalGainUpdateF32::prepare(OpticalGainUpdateF32Configuration {
        criteria: vec![OpticalGainUpdateCriteria::new(0.1, 0.05, 0.1, 2.0, 0.25)],
        assignments: vec![OpticalGainAssignment::DitherMode {
            dither_mode: 0,
            update_fraction: 0.5,
        }],
        initial_optical_gains: vec![1.0],
    })
    .unwrap();
    let optical_gain_correction = OpticalGainCorrectionPlan::new([0.5_f32]).unwrap();
    let mut dither_workspace = optical_gain_dither.workspace();
    let mut declared_dither = OpticalGainDitherF32::prepare(OpticalGainDitherF32Configuration {
        modes: vec![OpticalGainDitherMode::new(0.5, 4, 1)],
    })
    .unwrap();
    let mut estimator_workspace = optical_gain_estimator.workspace();
    let mut update_workspace = optical_gain_update.workspace(&[1.0]).unwrap();
    let mut modal_dither = [0.0_f32];
    let mut declared_window_frame = [0_u64];
    let mut declared_phase_indices = [0_u64];
    let mut declared_window_complete = [false];
    let mut command_amplitude = [0.0_f32];
    let mut response_amplitude = [0.0_f32];
    let mut optical_gain = [0.0_f32];
    let mut phase_lag = [0.0_f32];
    let mut optical_gain_valid = [false];
    let mut estimate_complete = false;
    let mut updated_optical_gain = [0.0_f32];
    let mut accepted_dither_mode = [false];
    let mut updated_reconstructed_mode = [false];
    let mut corrected_mode = [0.0_f32];
    assert_eq!(
        allocations(|| {
            optical_gain_dither
                .process(&mut modal_dither, &mut dither_workspace, ())
                .unwrap();
            optical_gain_estimator
                .process(
                    OpticalGainEstimateOutput::new(
                        &mut command_amplitude,
                        &mut response_amplitude,
                        &mut optical_gain,
                        &mut phase_lag,
                        &mut optical_gain_valid,
                        &mut estimate_complete,
                    ),
                    &mut estimator_workspace,
                    OpticalGainEstimatorInput::new(&modal_dither, &[0.25]),
                )
                .unwrap();
            optical_gain_update
                .process(
                    OpticalGainUpdateOutput::new(
                        &mut updated_optical_gain,
                        &mut accepted_dither_mode,
                        &mut updated_reconstructed_mode,
                    ),
                    &mut update_workspace,
                    OpticalGainUpdateInput::new(&[0.5], &[0.25], &[0.5], &[0.0], &[true], true),
                )
                .unwrap();
            optical_gain_correction
                .process(&mut corrected_mode, &mut NoWorkspace, &[0.25])
                .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| {
            OpticalGainDitherF32::process(
                &mut declared_dither,
                &mut modal_dither,
                &mut declared_window_frame,
                &mut declared_phase_indices,
                &mut declared_window_complete,
            )
            .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| OpticalGainDitherF32::reset(&mut declared_dither)),
        0
    );
    let mut declared_estimate_complete = [false];
    OpticalGainEstimatorF32::process(
        &mut declared_estimator,
        &[0.5],
        &[0.25],
        &[0],
        &[0],
        &[false],
        &mut command_amplitude,
        &mut response_amplitude,
        &mut optical_gain,
        &mut phase_lag,
        &mut optical_gain_valid,
        &mut declared_estimate_complete,
    )
    .unwrap();
    assert_eq!(
        allocations(|| {
            OpticalGainEstimatorF32::process(
                &mut declared_estimator,
                &[0.0],
                &[0.0],
                &[1],
                &[1],
                &[false],
                &mut command_amplitude,
                &mut response_amplitude,
                &mut optical_gain,
                &mut phase_lag,
                &mut optical_gain_valid,
                &mut declared_estimate_complete,
            )
            .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| OpticalGainEstimatorF32::reset(&mut declared_estimator)),
        0
    );
    OpticalGainUpdateF32::process(
        &mut declared_update,
        &[f32::NAN],
        &[f32::NAN],
        &[f32::NAN],
        &[f32::NAN],
        &[false],
        &[false],
        &mut updated_optical_gain,
        &mut accepted_dither_mode,
        &mut updated_reconstructed_mode,
    )
    .unwrap();
    assert_eq!(
        allocations(|| {
            OpticalGainUpdateF32::process(
                &mut declared_update,
                &[0.5],
                &[0.25],
                &[0.5],
                &[0.0],
                &[true],
                &[true],
                &mut updated_optical_gain,
                &mut accepted_dither_mode,
                &mut updated_reconstructed_mode,
            )
            .unwrap();
        }),
        0
    );
    assert_eq!(
        allocations(|| OpticalGainUpdateF32::reset(&mut declared_update)),
        0
    );
    let combination =
        filtergraph_algorithms::VectorLinearCombinationPlan::new(3, 0.5, -2.0).unwrap();
    let mut vector_workspace = combination.workspace();
    let mut vector_output = [0.0; 3];
    assert_eq!(
        allocations(|| combination
            .process(
                &mut vector_output,
                &mut vector_workspace,
                (&[1.0; 3], &[2.0; 3])
            )
            .unwrap()),
        0
    );
    assert_eq!(
        allocations(|| {
            assert!(
                combination
                    .process(
                        &mut vector_output,
                        &mut vector_workspace,
                        (&[1.0, 2.0, f32::NAN], &[2.0; 3])
                    )
                    .is_err()
            );
        }),
        0
    );
    assert_eq!(allocations(|| vector_workspace.reset()), 0);
    for delay in [0, 2] {
        let delay_plan = filtergraph_algorithms::FrameDelayPlan::<f32>::new(delay);
        let mut delay_workspace =
            filtergraph_algorithms::FrameDelayWorkspace::filled(3, 0.25, &delay_plan).unwrap();
        assert_eq!(
            allocations(|| delay_plan
                .process(&mut vector_output, &mut delay_workspace, &[1.0; 3])
                .unwrap()),
            0
        );
        assert_eq!(
            allocations(|| {
                assert!(
                    delay_plan
                        .process(
                            &mut vector_output,
                            &mut delay_workspace,
                            &[1.0, 2.0, f32::NAN]
                        )
                        .is_err()
                );
            }),
            0
        );
        assert_eq!(allocations(|| delay_workspace.reset()), 0);
    }
    let biquad = filtergraph_algorithms::BiquadCascadePlan::<f32>::new(
        vec![0.5, 0.25, 0.0, -0.5, 0.125],
        3,
        1,
        1,
    )
    .unwrap();
    let mut biquad_workspace = biquad.workspace();
    assert_eq!(
        allocations(|| biquad
            .process(&mut vector_output, &mut biquad_workspace, &[1.0; 3])
            .unwrap()),
        0
    );
    assert_eq!(
        allocations(|| {
            assert!(
                biquad
                    .process(
                        &mut vector_output,
                        &mut biquad_workspace,
                        &[1.0, 2.0, f32::NAN]
                    )
                    .is_err()
            );
        }),
        0
    );
    assert_eq!(allocations(|| biquad_workspace.reset()), 0);
    let mut signed_calibration = filtergraph_algorithms::PixelCalibrationF32::prepare(
        filtergraph_algorithms::PixelCalibrationF32Configuration {
            image_size: (1, 3),
            input_schema: "org.example.signed/1".into(),
            initial_flat: Some(vec![2.0; 3]),
            initial_background: None,
        },
    )
    .unwrap();
    assert_eq!(
        allocations(|| filtergraph_algorithms::PixelCalibrationF32::process(
            &mut signed_calibration,
            &[-0.5, 1.0, 2.0],
            &mut vector_output
        )
        .unwrap()),
        0
    );
    assert_eq!(
        allocations(|| {
            assert!(
                filtergraph_algorithms::PixelCalibrationF32::process(
                    &mut signed_calibration,
                    &[1.0, 2.0, f32::MAX],
                    &mut vector_output
                )
                .is_err()
            );
        }),
        0
    );
    assert_eq!(allocations(|| signed_calibration.reset()), 0);
    let mut weighted = filtergraph_algorithms::WeightedCenterOfGravityF32::prepare(
        filtergraph_algorithms::WeightedCenterOfGravityF32Configuration {
            pixel_count: 3,
            initial_coordinates: vec![[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            initial_weights: vec![1.0, 2.0, 3.0],
        },
    )
    .unwrap();
    let mut weighted_offset = [0.0; 2];
    let mut weighted_flux = [0.0];
    let mut weighted_valid = [false];
    assert_eq!(
        allocations(
            || filtergraph_algorithms::WeightedCenterOfGravityF32::process(
                &mut weighted,
                &[1.0; 3],
                &mut weighted_offset,
                &mut weighted_flux,
                &mut weighted_valid
            )
            .unwrap()
        ),
        0
    );
    assert_eq!(
        allocations(
            || filtergraph_algorithms::WeightedCenterOfGravityF32::process(
                &mut weighted,
                &[1.0, 2.0, f32::NAN],
                &mut weighted_offset,
                &mut weighted_flux,
                &mut weighted_valid
            )
            .unwrap()
        ),
        0
    );
    assert_eq!(allocations(|| weighted.reset()), 0);
    let mut allocation = filtergraph_algorithms::CorrectorAllocationF32::prepare(
        filtergraph_algorithms::CorrectorAllocationF32Configuration {
            controller_command_count: 1,
            stage_command_counts: vec![1, 1],
            update_periods: vec![1, 2],
            command_projection: vec![1.0, 1.0],
            realization_projection: vec![0.5, 0.5],
            section_count: 1,
            initial_coefficients: Some(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            command_limits: (vec![-2.0; 2], vec![2.0; 2]),
            slew_limits: Some(vec![0.25; 2]),
            command_schema: "org.example.vector/1".into(),
        },
    )
    .unwrap();
    let mut allocated = [0.0; 2];
    let mut allocation_feedback = [0.0];
    assert_eq!(
        allocations(|| filtergraph_algorithms::CorrectorAllocationF32::process(
            &mut allocation,
            &[1.0],
            &mut allocated,
            &mut allocation_feedback
        )
        .unwrap()),
        0
    );
    assert_eq!(
        allocations(|| {
            assert!(
                filtergraph_algorithms::CorrectorAllocationF32::process(
                    &mut allocation,
                    &[f32::NAN],
                    &mut allocated,
                    &mut allocation_feedback
                )
                .is_err()
            );
        }),
        0
    );
    assert_eq!(allocations(|| allocation.reset()), 0);
    let mut summary = filtergraph_algorithms::ShackHartmannSummaryF32::prepare(
        filtergraph_algorithms::ShackHartmannSummaryF32Configuration {
            subaperture_count: 2,
        },
    )
    .unwrap();
    let mut summary_values = [0.0; 8];
    assert_eq!(
        allocations(|| filtergraph_algorithms::ShackHartmannSummaryF32::process(
            &mut summary,
            &[3.0, 4.0, 0.0, 0.0],
            &[2.0, 6.0],
            &[true, true],
            &mut summary_values
        )
        .unwrap()),
        0
    );
    let mut weighted_image = filtergraph_algorithms::ShackHartmannWeightedCentroidF32::prepare(
        filtergraph_algorithms::ShackHartmannWeightedCentroidF32Configuration {
            image_schema: filtergraph_algorithms::schemas::CALIBRATED_PIXELS_V1.into(),
            subaperture_count: 1,
            initial_subaperture_origins: Some(vec![[0, 0]]),
            coordinate_scale: 1.0,
            pixel_threshold: 0.0,
            flux_threshold: 0.0,
            reference_slopes: None,
            active: None,
            initial_weights: vec![vec![1.0, 3.0]],
            image_size: (1, 2),
            subaperture_size: (1, 2),
            simd_width: 0,
        },
    )
    .unwrap();
    assert_eq!(
        allocations(
            || filtergraph_algorithms::ShackHartmannWeightedCentroidF32::process(
                &mut weighted_image,
                &[1.0, 1.0],
                &mut weighted_offset,
                &mut weighted_flux,
                &mut weighted_valid
            )
            .unwrap()
        ),
        0
    );
    assert_eq!(allocations(|| weighted_image.reset()), 0);
    let mut scc_residual = filtergraph_algorithms::SccPixelResidualF32::prepare(
        filtergraph_algorithms::SccPixelResidualF32Configuration {
            pixel_count: 3,
            mask: vec![true, false, true],
            initial_reference_image: Some(vec![0.5, 0.0, -0.25]),
            minimum_valid_fraction: 1.0,
        },
    )
    .unwrap();
    let mut scc_selected = [0.0; 2];
    let mut scc_valid = [false; 2];
    let mut scc_fraction = [0.0];
    assert_eq!(
        allocations(|| filtergraph_algorithms::SccPixelResidualF32::process(
            &mut scc_residual,
            &[1.0, 2.0, 3.0],
            &[true; 3],
            &[1],
            &mut scc_selected,
            &mut scc_valid,
            &mut scc_fraction
        )
        .unwrap()),
        0
    );
    assert_eq!(
        allocations(|| {
            assert!(
                filtergraph_algorithms::SccPixelResidualF32::process(
                    &mut scc_residual,
                    &[1.0, 2.0, 3.0],
                    &[true, true, false],
                    &[1],
                    &mut scc_selected,
                    &mut scc_valid,
                    &mut scc_fraction
                )
                .is_err()
            );
        }),
        0
    );
    assert_eq!(allocations(|| scc_residual.reset()), 0);
    let mut integrator = filtergraph_algorithms::ClosedLoopIntegratorF32::prepare(
        filtergraph_algorithms::ClosedLoopIntegratorF32Configuration { state_length: 3 },
    )
    .unwrap();
    let mut controller_state = [0.0; 3];
    assert_eq!(
        allocations(|| filtergraph_algorithms::ClosedLoopIntegratorF32::process(
            &mut integrator,
            &[1.0; 3],
            &[0.0; 3],
            &mut vector_output,
            &mut controller_state
        )
        .unwrap()),
        0
    );
    assert_eq!(
        allocations(|| {
            assert!(
                filtergraph_algorithms::ClosedLoopIntegratorF32::process(
                    &mut integrator,
                    &[1.0, 2.0, f32::NAN],
                    &[0.0; 3],
                    &mut vector_output,
                    &mut controller_state
                )
                .is_err()
            );
        }),
        0
    );
    assert_eq!(allocations(|| integrator.reset()), 0);
    let mut offset = filtergraph_algorithms::VectorOffsetF32::prepare(
        filtergraph_algorithms::VectorOffsetF32Configuration {
            extent: 3,
            schema: "org.example.vector/1".into(),
        },
    )
    .unwrap();
    assert_eq!(
        allocations(|| filtergraph_algorithms::VectorOffsetF32::process(
            &mut offset,
            &[1.0; 3],
            &mut vector_output,
            &mut controller_state
        )
        .unwrap()),
        0
    );
    assert_eq!(
        allocations(|| {
            assert!(
                filtergraph_algorithms::VectorOffsetF32::process(
                    &mut offset,
                    &[1.0, 2.0, f32::NAN],
                    &mut vector_output,
                    &mut controller_state
                )
                .is_err()
            );
        }),
        0
    );
    assert_eq!(allocations(|| offset.reset()), 0);
    let mut conditioning = filtergraph_algorithms::VectorConditioningF32::prepare(
        filtergraph_algorithms::VectorConditioningF32Configuration {
            count: 3,
            requested_schema: "org.example.vector/1".into(),
            demanded_schema: "org.example.vector/1".into(),
            feedback_schema: "org.example.vector/1".into(),
            slew_limit: Some(filtergraph_algorithms::ScalarOrVector::Scalar(0.25)),
            initial_current: None,
            quantization_origin: None,
            quantization_step: None,
            dither_seed: None,
            command_limits: (
                filtergraph_algorithms::ScalarOrVector::Scalar(-1.0),
                filtergraph_algorithms::ScalarOrVector::Scalar(1.0),
            ),
        },
    )
    .unwrap();
    assert_eq!(
        allocations(|| filtergraph_algorithms::VectorConditioningF32::process(
            &mut conditioning,
            &[1.0; 3],
            &mut vector_output,
            &mut controller_state
        )
        .unwrap()),
        0
    );
    assert_eq!(
        allocations(|| {
            assert!(
                filtergraph_algorithms::VectorConditioningF32::process(
                    &mut conditioning,
                    &[1.0, 2.0, f32::NAN],
                    &mut vector_output,
                    &mut controller_state
                )
                .is_err()
            );
        }),
        0
    );
    assert_eq!(allocations(|| conditioning.reset()), 0);
    let mut scc_reconstructor = filtergraph_algorithms::SccReconstructorF32::prepare(
        filtergraph_algorithms::SccReconstructorF32Configuration {
            controller_coordinate_count: 3,
            selected_pixel_count: 3,
            initial_reconstructor: Some(vec![1.0, 2.0, 3.0, 0.0, 1.0, 0.0, -1.0, 0.5, 0.25]),
        },
    )
    .unwrap();
    assert_eq!(
        allocations(|| filtergraph_algorithms::SccReconstructorF32::process(
            &mut scc_reconstructor,
            &[1.0; 3],
            &mut vector_output
        )
        .unwrap()),
        0
    );
    assert_eq!(
        allocations(|| {
            assert!(
                filtergraph_algorithms::SccReconstructorF32::process(
                    &mut scc_reconstructor,
                    &[1.0, 2.0, f32::NAN],
                    &mut vector_output
                )
                .is_err()
            );
        }),
        0
    );
    assert_equivalence_port_allocations();
    assert_pyramid_pupil_flux_allocations();
    assert_scientific_contract_allocations();
}

fn assert_equivalence_port_allocations() {
    use filtergraph_algorithms::*;
    let ml = MlLlowfsPlan::new(ML_LLOWFS_MODEL)
        .unwrap()
        .with_parameters(&vec![0.0; ML_LLOWFS_PARAMETER_COUNT])
        .unwrap();
    let mut ml_workspace = ml.workspace();
    let input: Vec<f32> = (0..400).map(|i| 1.0 + (i % 7) as f32).collect();
    let mut modes = [0.0; 14];
    ml.process(&mut modes, &mut ml_workspace, &input).unwrap();
    assert_eq!(
        allocations(|| {
            ml.process(&mut modes, &mut ml_workspace, &input).unwrap();
            ml_workspace.reset();
            assert!(
                ml.process(&mut modes, &mut ml_workspace, &[f32::NAN; 400])
                    .is_err()
            );
        }),
        0
    );
    let shape = VdmShapeOffsetPlan::new(
        2,
        2,
        2,
        &[1.0, 0.0, 0.0, 1.0],
        &[1.0, 0.0, 0.0, 1.0],
        &[0.5, -1.0],
    )
    .unwrap();
    let mut sw = shape.workspace();
    let mut output = [0.0; 2];
    let mut current = [0.0; 2];
    assert_eq!(
        allocations(|| {
            shape
                .process((&mut output, &mut current), &mut sw, &[1.0, 2.0])
                .unwrap();
            sw.reset();
            assert!(
                shape
                    .process((&mut output, &mut current), &mut sw, &[1.0, f32::NAN])
                    .is_err()
            );
        }),
        0
    );
    let mut dense =
        DenseVectorReconstructorF32::prepare(DenseVectorReconstructorF32Configuration {
            input_count: 2,
            output_count: 2,
            initial_reconstructor: Some(vec![1.0, 2.0, 3.0, 4.0]),
            input_schema: "org.example.input/1".into(),
            output_schema: "org.example.output/1".into(),
            reconstructor_schema: "org.example.matrix/1".into(),
        })
        .unwrap();
    assert_eq!(
        allocations(|| {
            DenseVectorReconstructorF32::process(&mut dense, &[1.0, 2.0], &mut output).unwrap();
            dense.reset();
            assert!(
                DenseVectorReconstructorF32::process(&mut dense, &[f32::NAN, 2.0], &mut output)
                    .is_err()
            );
        }),
        0
    );
    let mut pyramid = PyramidImageF32::prepare(PyramidImageF32Configuration {
        image_size: (2, 2),
        pupil_size: (1, 1),
        pupil_origins: vec![[0, 0], [1, 0], [1, 1], [0, 1]],
        i4q_support: None,
        initial_reference_signal: None,
        initial_optical_gain: None,
        normalization_policy: "mean-valid-flux".into(),
        incidence_flux: 1.0,
    })
    .unwrap();
    let mut divisor = [0.0];
    assert_eq!(
        allocations(|| {
            PyramidImageF32::process(
                &mut pyramid,
                &[8.0, 5.0, 2.0, 1.0],
                &mut output,
                &mut divisor,
            )
            .unwrap();
            pyramid.reset();
            assert!(
                PyramidImageF32::process(&mut pyramid, &[f32::NAN; 4], &mut output, &mut divisor)
                    .is_err()
            );
        }),
        0
    );
    let mut bio = BiOEdgeImageF32::prepare(BiOEdgeImageF32Configuration {
        image_size: (2, 2),
        pupil_size: (1, 1),
        pupil_origins: vec![[0, 0], [1, 0], [1, 1], [0, 1]],
        bi_o_edge_support: None,
        initial_reference_signal: None,
        initial_optical_gain: None,
        normalization_policy: "mean-valid-flux".into(),
        incidence_flux: 1.0,
        initial_calibration_signature: 7,
    })
    .unwrap();
    assert_eq!(
        allocations(|| {
            BiOEdgeImageF32::process(
                &mut bio,
                &[8.0, 5.0, 2.0, 1.0],
                &[7],
                &mut output,
                &mut divisor,
            )
            .unwrap();
            bio.reset();
            assert!(
                BiOEdgeImageF32::process(&mut bio, &[f32::NAN; 4], &[7], &mut output, &mut divisor)
                    .is_err()
            );
        }),
        0
    );
    let mut z = ZernikePupilSignalF32::prepare(ZernikePupilSignalF32Configuration {
        image_size: (1, 2),
        zernike_pupil_support: None,
        initial_reference_pupil_image: None,
        normalization_policy: "incidence-flux".into(),
        incidence_flux: 4.0,
        initial_calibration_signature: 7,
    })
    .unwrap();
    assert_eq!(
        allocations(|| {
            ZernikePupilSignalF32::process(&mut z, &[2.0, 6.0], &[7], &mut output, &mut divisor)
                .unwrap();
            z.reset();
            assert!(
                ZernikePupilSignalF32::process(
                    &mut z,
                    &[2.0, 6.0],
                    &[8],
                    &mut output,
                    &mut divisor
                )
                .is_err()
            );
        }),
        0
    );
    let mut curvature = CurvaturePairedImageF32::prepare(CurvaturePairedImageF32Configuration {
        image_size: (1, 4),
        signal_size: (1, 2),
        branch_size: (1, 2),
        positive_branch_origin: [0, 0],
        negative_branch_origin: [0, 2],
        curvature_support: None,
        initial_reference_signal: None,
        initial_branch_scales: None,
        initial_calibration_signature: 7,
    })
    .unwrap();
    assert_eq!(
        allocations(|| {
            CurvaturePairedImageF32::process(
                &mut curvature,
                &[3.0, 6.0, 1.0, 2.0],
                &[7],
                &mut output,
            )
            .unwrap();
            curvature.reset();
            assert!(
                CurvaturePairedImageF32::process(&mut curvature, &[f32::NAN; 4], &[7], &mut output)
                    .is_err()
            );
        }),
        0
    );
    let mut channels = CurvaturePairedChannelF32::prepare(CurvaturePairedChannelF32Configuration {
        input_branch_rows: 2,
        input_channels: 2,
        signal_size: (1, 2),
        positive_branch_row: 0,
        negative_branch_row: 1,
        channel_offsets: vec![1, 0],
        curvature_support: None,
        initial_reference_signal: None,
        initial_branch_scales: None,
        initial_calibration_signature: 7,
    })
    .unwrap();
    assert_eq!(
        allocations(|| {
            CurvaturePairedChannelF32::process(
                &mut channels,
                &[3.0, 6.0, 1.0, 2.0],
                &[7],
                &mut output,
            )
            .unwrap();
            channels.reset();
            assert!(
                CurvaturePairedChannelF32::process(
                    &mut channels,
                    &[f32::NAN; 4],
                    &[7],
                    &mut output
                )
                .is_err()
            );
        }),
        0
    );
    let spatial = SccSpatialCarrierDemodulationPlan::new(
        (2, 3),
        &[true; 6],
        3,
        None,
        None,
        Some(&[0.0, 0.3, -0.7, 1.1, -1.4, 2.0]),
        Some(&[1.0, 2.0, 1.0]),
        0.25,
        0.5,
    )
    .unwrap();
    let mut spatial_workspace = spatial.workspace();
    let mut measurements = [0.0; 12];
    let mut weights = [0.0; 12];
    let mut validity = [false; 12];
    let mut fraction = 0.0;
    assert_eq!(
        allocations(|| {
            spatial
                .process(
                    (
                        &mut measurements,
                        &mut weights,
                        &mut validity,
                        &mut fraction,
                    ),
                    &mut spatial_workspace,
                    (&[1.0, 2.0, 4.0, 8.0, 16.0, 32.0], &[true; 6], 1),
                )
                .unwrap();
            spatial_workspace.reset();
            assert!(
                spatial
                    .process(
                        (
                            &mut measurements,
                            &mut weights,
                            &mut validity,
                            &mut fraction
                        ),
                        &mut spatial_workspace,
                        (&[0.0; 6], &[false; 6], 1)
                    )
                    .is_err()
            );
        }),
        0
    );
    let pcg = SccWeightedRegularizedReconstructionPlan::new(
        4,
        2,
        3,
        vec![1.0, 0.5, -1.0, 2.0, 2.0, -0.25, 0.75, -1.5],
        vec![1.0, -0.25, 0.5, 1.25, -0.75, 0.5],
        0.125,
        16,
        1e-5,
        1e-8,
    )
    .unwrap();
    let mut pw = pcg.workspace();
    assert_eq!(
        allocations(|| {
            pcg.process(
                &mut output,
                &mut pw,
                (&[0.75, -1.25, 999.0, 2.5], &[1.0, 0.5, 0.0, 1.5]),
            )
            .unwrap();
            pw.reset();
            assert!(
                pcg.process(
                    &mut output,
                    &mut pw,
                    (&[0.75, -1.25, 999.0, 2.5], &[1.0, -0.5, 0.0, 1.5])
                )
                .is_err()
            );
        }),
        0
    );
    let mut correction = SccCorrectionF32::prepare(SccCorrectionF32Configuration {
        frame_size: (2, 2),
        controller_coordinate_count: 2,
        image_schema: "org.example.frame/1".into(),
        exposure_duration_nanoseconds: 1000,
        coordinate_order: vec![1, 2],
        command_unit: "controller-unit".into(),
        reconstructor_sign: "command_error".into(),
        reference_difference: vec![0.0; 4],
        correction_mask: vec![true, false, false, true],
        reconstructor: vec![1.0, 0.0, 0.0, 2.0],
        fringed_response: vec![0.0; 4],
        unfringed_response: vec![0.0; 4],
        bias_command: vec![0.01; 2],
        command_limits: (ScalarOrVector::Scalar(-1.0), ScalarOrVector::Scalar(1.0)),
        gain: 0.5,
        pole: 0.8,
    })
    .unwrap();
    let metadata = |sequence, kind| {
        Some(
            SampleMetadata::new(sequence, 0)
                .with_acquisition_identity(Some(
                    SampleAcquisitionIdentity::new([1; 16], 1, sequence).unwrap(),
                ))
                .with_chopped_image_kind(Some(kind))
                .with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(1000)),
        )
    };
    assert_eq!(
        allocations(|| {
            SccCorrectionF32::process_with_metadata(
                &mut correction,
                &[metadata(1, ChoppedImageKind::Fringed)],
                &[0.1, 9.0, 8.0, -0.1],
                &mut output,
            )
            .unwrap();
            SccCorrectionF32::process_with_metadata(
                &mut correction,
                &[metadata(2, ChoppedImageKind::Unfringed)],
                &[0.0; 4],
                &mut output,
            )
            .unwrap();
            assert!(
                SccCorrectionF32::process_with_metadata(
                    &mut correction,
                    &[None],
                    &[0.0; 4],
                    &mut output
                )
                .is_err()
            );
            correction.reset();
        }),
        0
    );
    let mut weighted = ShackHartmannWeightedMeasurementBlockF32::prepare(
        ShackHartmannWeightedMeasurementBlockF32Configuration {
            row_block_rows: 2,
            subaperture_count: 1,
            initial_subaperture_origins: Some(vec![[0, 0]]),
            coordinate_scale: 1.0,
            pixel_threshold: 0.0,
            flux_threshold: 0.0,
            reference_slopes: None,
            active: None,
            initial_weights: vec![vec![1.0; 4]],
            measurements_schema: "test.measurements/1".into(),
            column_indices_schema: "test.columns/1".into(),
            measurement_count_schema: "test.count/1".into(),
            image_size: (4, 2),
            subaperture_size: (2, 2),
        },
    )
    .unwrap();
    let mut columns = [0u32; 2];
    let mut count = [0u32];
    assert_eq!(
        allocations(|| {
            for row in [0, 2] {
                ShackHartmannWeightedMeasurementBlockF32::process_with_metadata(
                    &mut weighted,
                    &[Some(SampleMetadata::new(1, row).with_terminal(row == 2))],
                    &[2.0; 4],
                    &mut output,
                    &mut columns,
                    &mut count,
                )
                .unwrap();
            }
            weighted.reset();
            assert!(
                ShackHartmannWeightedMeasurementBlockF32::process_with_metadata(
                    &mut weighted,
                    &[None],
                    &[2.0; 4],
                    &mut output,
                    &mut columns,
                    &mut count
                )
                .is_err()
            );
        }),
        0
    );
    let mut readout = ShackHartmannReadoutMeasurementBlockF32::prepare(
        ShackHartmannReadoutMeasurementBlockF32Configuration {
            region_lines: 4,
            block_lines: 2,
            region_samples: 2,
            detector_origins: vec![[0, 0]],
            detector_line_steps: vec![[1, 0]],
            detector_sample_steps: vec![[0, 1]],
            subaperture_count: 1,
            initial_subaperture_origins: Some(vec![[0, 0]]),
            coordinate_scale: 1.0,
            pixel_threshold: 0.0,
            flux_threshold: 0.0,
            reference_slopes: None,
            active: None,
            measurements_schema: "test.measurements/1".into(),
            column_indices_schema: "test.columns/1".into(),
            measurement_count_schema: "test.count/1".into(),
            subaperture_size: (2, 2),
            detector_size: (4, 2),
        },
    )
    .unwrap();
    assert_eq!(
        allocations(|| {
            for ordinal in [0, 1] {
                ShackHartmannReadoutMeasurementBlockF32::process_with_metadata(
                    &mut readout,
                    &[Some(
                        SampleMetadata::new(1, ordinal).with_terminal(ordinal == 1),
                    )],
                    &[2.0; 4],
                    &mut output,
                    &mut columns,
                    &mut count,
                )
                .unwrap();
            }
            readout.reset();
            assert!(
                ShackHartmannReadoutMeasurementBlockF32::process_with_metadata(
                    &mut readout,
                    &[None],
                    &[2.0; 4],
                    &mut output,
                    &mut columns,
                    &mut count
                )
                .is_err()
            );
        }),
        0
    );
    let mut row_reconstruction =
        ShackHartmannRowReconstructorF32::prepare(ShackHartmannRowReconstructorF32Configuration {
            row_block_rows: 2,
            subaperture_count: 1,
            initial_subaperture_origins: Some(vec![[0, 0]]),
            coordinate_scale: 1.0,
            pixel_threshold: 0.0,
            flux_threshold: 0.0,
            reference_slopes: None,
            active: None,
            reconstructed_count: 2,
            initial_reconstructor: Some(vec![1.0, 0.0, 0.0, 1.0]),
            reconstructed_schema: "test.reconstructed/1".into(),
            image_size: (4, 2),
            subaperture_size: (2, 2),
        })
        .unwrap();
    assert_eq!(
        allocations(|| {
            for row in [0, 2] {
                ShackHartmannRowReconstructorF32::process_with_metadata(
                    &mut row_reconstruction,
                    &[Some(SampleMetadata::new(1, row).with_terminal(row == 2))],
                    &[2.0; 4],
                    &mut output,
                )
                .unwrap();
            }
            row_reconstruction.reset();
            assert!(
                ShackHartmannRowReconstructorF32::process_with_metadata(
                    &mut row_reconstruction,
                    &[None],
                    &[2.0; 4],
                    &mut output
                )
                .is_err()
            );
        }),
        0
    );
    let mut pupil_reconstruction =
        PyramidPupilReconstructorF32::prepare(PyramidPupilReconstructorF32Configuration {
            selected_pixel_count: 1,
            reconstructed_count: 2,
            initial_reconstructor: Some(vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]),
            reconstructed_schema: "test.reconstructed/1".into(),
        })
        .unwrap();
    assert_eq!(
        allocations(|| {
            PyramidPupilReconstructorF32::process(
                &mut pupil_reconstruction,
                &[1.0; 4],
                &mut output,
            )
            .unwrap();
            pupil_reconstruction.reset();
            assert!(
                PyramidPupilReconstructorF32::process(
                    &mut pupil_reconstruction,
                    &[f32::NAN; 4],
                    &mut output
                )
                .is_err()
            );
        }),
        0
    );
    let mut row_measurement =
        PyramidMeasurementBlockF32::prepare(PyramidMeasurementBlockF32Configuration {
            image_size: (4, 2),
            row_block_rows: 1,
            pupil_size: (1, 1),
            pupil_origins: vec![[0, 0], [1, 0], [2, 0], [3, 0]],
            pupil_mask: None,
            measurements_schema: "test.measurements/1".into(),
            column_indices_schema: "test.columns/1".into(),
            measurement_count_schema: "test.count/1".into(),
        })
        .unwrap();
    let mut values = [0.0; 4];
    let mut cols = [0u32; 4];
    assert_eq!(
        allocations(|| {
            for frame in 0..2 {
                for row in 0..4 {
                    PyramidMeasurementBlockF32::process_with_metadata(
                        &mut row_measurement,
                        &[Some(
                            SampleMetadata::new(frame, row).with_terminal(row == 3),
                        )],
                        &[2.0; 2],
                        &mut values,
                        &mut cols,
                        &mut count,
                    )
                    .unwrap();
                }
            }
            row_measurement.reset();
            assert!(
                PyramidMeasurementBlockF32::process_with_metadata(
                    &mut row_measurement,
                    &[None],
                    &[2.0; 2],
                    &mut values,
                    &mut cols,
                    &mut count
                )
                .is_err()
            );
        }),
        0
    );
    let mut readout_measurement = PyramidReadoutMeasurementBlockF32::prepare(
        PyramidReadoutMeasurementBlockF32Configuration {
            detector_size: (4, 2),
            region_lines: 4,
            block_lines: 1,
            region_samples: 2,
            detector_origins: vec![[0, 0]],
            detector_line_steps: vec![[1, 0]],
            detector_sample_steps: vec![[0, 1]],
            pupil_size: (1, 1),
            pupil_origins: vec![[0, 0], [1, 0], [2, 0], [3, 0]],
            pupil_mask: None,
            measurements_schema: "test.measurements/1".into(),
            column_indices_schema: "test.columns/1".into(),
            measurement_count_schema: "test.count/1".into(),
        },
    )
    .unwrap();
    assert_eq!(
        allocations(|| {
            for frame in 0..2 {
                for ordinal in 0..4 {
                    PyramidReadoutMeasurementBlockF32::process_with_metadata(
                        &mut readout_measurement,
                        &[Some(
                            SampleMetadata::new(frame, ordinal).with_terminal(ordinal == 3),
                        )],
                        &[2.0; 2],
                        &mut values,
                        &mut cols,
                        &mut count,
                    )
                    .unwrap();
                }
            }
            readout_measurement.reset();
            assert!(
                PyramidReadoutMeasurementBlockF32::process_with_metadata(
                    &mut readout_measurement,
                    &[None],
                    &[2.0; 2],
                    &mut values,
                    &mut cols,
                    &mut count
                )
                .is_err()
            );
        }),
        0
    );

    let mut i4q = PyramidReconstructorF32::prepare(PyramidReconstructorF32Configuration {
        selected_coordinate_count: 2,
        reconstructed_count: 2,
        initial_reconstructor: Some(vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]),
        reconstructed_schema: "test.reconstructed/1".into(),
    })
    .unwrap();
    assert_eq!(
        allocations(|| {
            PyramidReconstructorF32::process(&mut i4q, &[1.0, 2.0, 3.0, 4.0], &mut output).unwrap();
            i4q.reset();
            assert!(
                PyramidReconstructorF32::process(&mut i4q, &[f32::NAN; 4], &mut output).is_err()
            );
        }),
        0
    );
}

fn assert_pyramid_pupil_flux_allocations() {
    use filtergraph_algorithms::{
        PyramidPupilFluxU16F64 as D, PyramidPupilFluxU16F64Configuration,
    };
    let mut d = D::prepare(PyramidPupilFluxU16F64Configuration {
        image_size: (64, 64),
        pupil_size: (30, 30),
        pupil_origins: vec![[33, 34], [0, 0], [34, 1], [0, 34]],
    })
    .unwrap();
    let raw = [u16::MAX; 4096];
    let mut out = [0.0; 4];
    D::process(&mut d, &raw, &mut out).unwrap();
    assert_eq!(
        allocations(|| {
            D::process(&mut d, &raw, &mut out).unwrap();
            d.reset();
        }),
        0
    );
}

fn assert_scientific_contract_allocations() {
    use filtergraph_algorithms::{
        PhysicalOffsetReferenceF32 as R, PhysicalOffsetReferenceF32Configuration,
        PixelValidityF32 as V, PixelValidityF32Configuration, SampleAcquisitionIdentity,
        SampleMetadata,
    };
    let mut reference = R::prepare(PhysicalOffsetReferenceF32Configuration {
        residual_count: 3,
        physical_count: 3,
        mode_count: 2,
        reference_sign: -1.0,
        residual_schema: "org.example.residual/1".into(),
        physical_schema: "org.example.physical/1".into(),
        physical_to_modes_schema: "org.example.q/1".into(),
        modes_to_reference_schema: "org.example.c/1".into(),
    })
    .unwrap();
    reference
        .replace_parameter(4, AlgorithmInput::from([1.0_f32; 6].as_slice()))
        .unwrap()
        .unwrap();
    reference
        .replace_parameter(5, AlgorithmInput::from([2.0_f32; 6].as_slice()))
        .unwrap()
        .unwrap();
    reference
        .replace_parameter(3, AlgorithmInput::from([0.5_f32; 3].as_slice()))
        .unwrap()
        .unwrap();
    let mut compensated = [0.; 3];
    let mut physical = [0.; 3];
    R::process(&mut reference, &[1.; 3], &mut compensated, &mut physical).unwrap();
    assert_eq!(
        allocations(|| {
            R::process(&mut reference, &[1.; 3], &mut compensated, &mut physical).unwrap();
            assert!(
                R::process(
                    &mut reference,
                    &[f32::NAN; 3],
                    &mut compensated,
                    &mut physical
                )
                .is_err()
            );
            R::reset(&mut reference);
            R::process(&mut reference, &[1.; 3], &mut compensated, &mut physical).unwrap();
        }),
        0
    );
    assert_eq!(compensated, [-5.; 3]);
    assert_eq!(physical, [0.5; 3]);
    let mut validity = V::prepare(PixelValidityF32Configuration {
        extent: 3,
        input_schema: "org.example.adu/1".into(),
    })
    .unwrap();
    let mut valid = [false; 3];
    V::process(&mut validity, &[1., f32::NAN, f32::INFINITY], &mut valid).unwrap();
    assert_eq!(
        allocations(|| {
            V::process(&mut validity, &[1., f32::NAN, f32::INFINITY], &mut valid).unwrap();
            V::reset(&mut validity);
        }),
        0
    );
    assert_eq!(valid, [true, false, false]);
    let mut chopped = ChoppedImageDifferenceF32::prepare(ChoppedImageDifferenceF32Configuration {
        pixel_count: 3,
        trigger_high_is_fringed: true,
        input_schema: "org.example.adu/1".into(),
        require_acquisition_identity: true,
    })
    .unwrap();
    let domain = [1; 16];
    let mut difference = [0.; 3];
    let mut status = [99];
    assert_eq!(
        allocations(|| {
            for (seq, kind) in [
                (1, ChoppedImageKind::Fringed),
                (2, ChoppedImageKind::Unfringed),
                (4, ChoppedImageKind::Fringed),
                (5, ChoppedImageKind::Mixed),
                (6, ChoppedImageKind::Unfringed),
                (7, ChoppedImageKind::Fringed),
            ] {
                let metadata = SampleMetadata::new(seq, 0)
                    .with_acquisition_identity(Some(
                        SampleAcquisitionIdentity::new(domain, 1, seq).unwrap(),
                    ))
                    .with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(20))
                    .with_chopped_image_kind(Some(kind));
                ChoppedImageDifferenceF32::process_with_metadata(
                    &mut chopped,
                    &[Some(metadata), None],
                    &[1.; 3],
                    &[true; 3],
                    &mut difference,
                    &mut valid,
                    &mut status,
                )
                .unwrap();
            }
            let metadata = SampleMetadata::new(7, 0)
                .with_acquisition_identity(Some(
                    SampleAcquisitionIdentity::new(domain, 1, 7).unwrap(),
                ))
                .with_exposure_duration_nanoseconds(core::num::NonZeroU64::new(20))
                .with_chopped_image_kind(Some(ChoppedImageKind::Unfringed));
            assert!(
                ChoppedImageDifferenceF32::process_with_metadata(
                    &mut chopped,
                    &[Some(metadata), None],
                    &[1.; 3],
                    &[true; 3],
                    &mut difference,
                    &mut valid,
                    &mut status
                )
                .is_err()
            );
            ChoppedImageDifferenceF32::reset(&mut chopped);
        }),
        0
    );
    assert_eq!(status, [1]);
}
