//! Center-of-gravity preparation, measurement, and validity tests.

use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, AlgorithmProperties, ArrayLayout,
    CenterOfGravityF32, CenterOfGravityF32Configuration, CenterOfGravityOutput,
    CenterOfGravityPlan, CenterOfGravityPlanError, ConfigurationUpdate,
    DeclarationPreparationError, ElementType, PortRole, ProcessError, PropertyAssignment,
    PropertyPolicy, PropertyValue,
};

#[test]
fn preparation_validates_coordinates_references_and_thresholds() {
    use CenterOfGravityPlanError::*;
    for (x, y, reference, pixel, flux, expected, diagnostic) in [
        (
            &[][..],
            &[][..],
            [0.0; 2],
            None,
            0.0,
            EmptyCoordinates,
            "region must contain pixels",
        ),
        (
            &[0.0][..],
            &[0.0, 1.0][..],
            [0.0; 2],
            None,
            0.0,
            CoordinateExtent { x: 1, y: 2 },
            "x-coordinate extent 1",
        ),
        (
            &[0.0, f32::NAN][..],
            &[0.0; 2][..],
            [0.0; 2],
            None,
            0.0,
            NonFiniteX { pixel: 1 },
            "x coordinate at selected pixel 1",
        ),
        (
            &[0.0; 2][..],
            &[0.0, f32::INFINITY][..],
            [0.0; 2],
            None,
            0.0,
            NonFiniteY { pixel: 1 },
            "y coordinate at selected pixel 1",
        ),
        (
            &[0.0][..],
            &[0.0][..],
            [f32::NAN, 0.0],
            None,
            0.0,
            NonFiniteReference { coordinate: 0 },
            "reference centroid coordinate 0",
        ),
        (
            &[0.0][..],
            &[0.0][..],
            [0.0; 2],
            Some(f32::NAN),
            0.0,
            NonFinitePixelThreshold,
            "pixel threshold must be finite",
        ),
        (
            &[0.0][..],
            &[0.0][..],
            [0.0; 2],
            None,
            -1.0,
            InvalidFluxThreshold,
            "flux threshold",
        ),
    ] {
        let error = CenterOfGravityPlan::new(x, y, reference, pixel, flux).unwrap_err();
        assert_eq!(error, expected);
        assert!(error.to_string().contains(diagnostic), "{error}");
    }
}

#[test]
fn measurement_uses_detector_coordinates_and_reference_centroid() {
    let plan = CenterOfGravityPlan::new(
        [-1.0_f32, 0.0, 1.0],
        [0.0_f32, 1.0, 2.0],
        [0.25, -0.5],
        None,
        4.0,
    )
    .unwrap();
    let pixels = [0.0_f32, 0.0, 4.0];
    let mut centroid_offset = [f32::NAN; 2];
    let mut flux = f32::NAN;
    let mut validity = false;

    plan.process(
        CenterOfGravityOutput::new(&mut centroid_offset, &mut flux, &mut validity),
        &mut plan.workspace(),
        &pixels,
    )
    .unwrap();

    assert_eq!(centroid_offset, [0.75, 2.5]);
    assert_eq!(flux, 4.0);
    assert!(validity);
}

#[test]
fn absent_pixel_threshold_retains_negative_calibrated_pixels() {
    let all_pixels =
        CenterOfGravityPlan::new([-1.0_f32, 1.0], [0.0; 2], [0.0; 2], None, 0.0).unwrap();
    let nonnegative_pixels =
        CenterOfGravityPlan::new([-1.0_f32, 1.0], [0.0; 2], [0.0; 2], Some(0.0), 0.0).unwrap();
    let pixels = [-1.0_f32, 2.0];
    let mut all_offset = [0.0; 2];
    let mut selected_offset = [0.0; 2];
    let mut all_flux = 0.0;
    let mut selected_flux = 0.0;
    let mut all_valid = false;
    let mut selected_valid = false;

    all_pixels
        .process(
            CenterOfGravityOutput::new(&mut all_offset, &mut all_flux, &mut all_valid),
            &mut all_pixels.workspace(),
            &pixels,
        )
        .unwrap();
    nonnegative_pixels
        .process(
            CenterOfGravityOutput::new(
                &mut selected_offset,
                &mut selected_flux,
                &mut selected_valid,
            ),
            &mut nonnegative_pixels.workspace(),
            &pixels,
        )
        .unwrap();

    assert_eq!(all_flux, 1.0);
    assert_eq!(all_offset, [3.0, 0.0]);
    assert!(all_valid);
    assert_eq!(selected_flux, 2.0);
    assert_eq!(selected_offset, [1.0, 0.0]);
    assert!(selected_valid);
}

#[test]
fn invalid_measurement_has_zero_offset_and_preserves_flux() {
    let plan = CenterOfGravityPlan::new([0.0_f32], [0.0], [0.0; 2], None, 1.0).unwrap();
    let mut centroid_offset = [f32::NAN; 2];
    let mut flux = f32::NAN;
    let mut validity = true;
    plan.process(
        CenterOfGravityOutput::new(&mut centroid_offset, &mut flux, &mut validity),
        &mut plan.workspace(),
        &[0.5],
    )
    .unwrap();
    assert_eq!(centroid_offset, [0.0; 2]);
    assert_eq!(flux, 0.5);
    assert!(!validity);

    plan.process(
        CenterOfGravityOutput::new(&mut centroid_offset, &mut flux, &mut validity),
        &mut plan.workspace(),
        &[f32::NAN],
    )
    .unwrap();
    assert_eq!(centroid_offset, [0.0; 2]);
    assert!(flux.is_nan());
    assert!(!validity);
}

#[test]
fn input_extent_is_rejected_before_output_mutation() {
    let plan = CenterOfGravityPlan::new([0.0_f32], [0.0], [0.0; 2], None, 0.0).unwrap();
    let mut centroid_offset = [-1.0; 2];
    let mut flux = -1.0;
    let mut validity = true;
    assert_eq!(
        plan.process(
            CenterOfGravityOutput::new(&mut centroid_offset, &mut flux, &mut validity),
            &mut plan.workspace(),
            &[],
        ),
        Err(ProcessError::InputExtent {
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(centroid_offset, [-1.0; 2]);
    assert_eq!(flux, -1.0);
    assert!(validity);
}

#[test]
fn declaration_keeps_coordinates_sparse_and_scalar_controls_transactional() {
    let initial_coordinates = [[-1.0_f32, 0.0], [0.0, 1.0], [1.0, 2.0]];
    let mut declared = CenterOfGravityF32::prepare(CenterOfGravityF32Configuration {
        pixel_count: 3,
        initial_coordinates: Some(initial_coordinates.to_vec()),
    })
    .unwrap();
    assert_eq!(CenterOfGravityF32::NAME, "center-of-gravity-f32");
    assert_eq!(CenterOfGravityF32::PROPERTIES, PropertyPolicy::Runtime);
    assert_eq!(CenterOfGravityF32::CONFIGURATION.len(), 2);
    assert_eq!(
        CenterOfGravityF32::CONFIGURATION[0].update(),
        ConfigurationUpdate::GraphRebuild
    );
    assert_eq!(
        CenterOfGravityF32::CONFIGURATION[1].update(),
        ConfigurationUpdate::Construction
    );
    assert_eq!(CenterOfGravityF32::PORTS.len(), 5);
    assert_eq!(CenterOfGravityF32::PORTS[4].role(), PortRole::Parameter);
    assert_eq!(
        CenterOfGravityF32::PORTS[4].element_type(),
        ElementType::F32
    );
    assert_eq!(declared.formats()[0].shape(), &[3]);
    assert_eq!(declared.formats()[1].shape(), &[2]);
    assert_eq!(declared.formats()[2].shape(), &[1]);
    assert_eq!(declared.formats()[3].shape(), &[1]);
    assert_eq!(declared.formats()[4].shape(), &[3, 2]);
    assert!(
        declared
            .formats()
            .iter()
            .all(|format| format.layout() == ArrayLayout::RowMajor)
    );

    let properties = (0..)
        .map_while(|index| declared.plan().property_info(index))
        .collect::<Vec<_>>();
    assert_eq!(
        properties
            .iter()
            .map(|property| property.name)
            .collect::<Vec<_>>(),
        [
            "reference-centroid-x",
            "reference-centroid-y",
            "pixel-threshold-enabled",
            "pixel-threshold",
            "flux-threshold",
        ]
    );
    declared
        .replace_properties(&[
            PropertyAssignment {
                id: properties[0].id,
                value: PropertyValue::Float(0.25),
            },
            PropertyAssignment {
                id: properties[1].id,
                value: PropertyValue::Float(-0.5),
            },
            PropertyAssignment {
                id: properties[4].id,
                value: PropertyValue::Float(4.0),
            },
        ])
        .unwrap();
    let coordinates = [-1.0_f32, 0.0, 0.0, 1.0, 1.0, 2.0];
    assert_eq!(
        declared.replace_parameter(4, AlgorithmInput::from(&coordinates[..])),
        Some(Ok(()))
    );
    assert_eq!(declared.plan().reference_centroid(), [0.25, -0.5]);
    assert_eq!(declared.plan().flux_threshold(), 4.0);

    let mut offset = [f32::NAN; 2];
    let mut flux = [f32::NAN];
    let mut validity = [false];
    CenterOfGravityF32::process(
        &mut declared,
        &[0.0, 0.0, 4.0],
        &mut offset,
        &mut flux,
        &mut validity,
    )
    .unwrap();
    assert_eq!(offset, [0.75, 2.5]);
    assert_eq!(flux, [4.0]);
    assert_eq!(validity, [true]);

    declared
        .replace_properties(&[
            PropertyAssignment {
                id: properties[2].id,
                value: PropertyValue::Bool(true),
            },
            PropertyAssignment {
                id: properties[3].id,
                value: PropertyValue::Float(5.0),
            },
        ])
        .unwrap();
    CenterOfGravityF32::process(
        &mut declared,
        &[0.0, 0.0, 4.0],
        &mut offset,
        &mut flux,
        &mut validity,
    )
    .unwrap();
    assert_eq!(offset, [0.0; 2]);
    assert_eq!(flux, [0.0]);
    assert_eq!(validity, [false]);
}

#[test]
fn coordinate_replacement_rejects_extent_without_changing_the_plan() {
    let plan = CenterOfGravityPlan::new(
        [-1.0_f32, 0.0, 1.0],
        [0.0_f32, 1.0, 2.0],
        [0.25, -0.5],
        None,
        4.0,
    )
    .unwrap();
    assert_eq!(
        plan.with_pair_interleaved_coordinates(&[0.0, 0.0])
            .unwrap_err(),
        CenterOfGravityPlanError::PairInterleavedCoordinateExtent {
            expected: 6,
            actual: 2,
        }
    );
    assert_eq!(plan.x_coordinates(), [-1.0, 0.0, 1.0]);
    assert_eq!(plan.reference_centroid(), [0.25, -0.5]);
}

#[test]
fn declaration_rejects_an_initial_coordinate_count_mismatch() {
    assert!(matches!(
        CenterOfGravityF32::prepare(CenterOfGravityF32Configuration {
            pixel_count: 2,
            initial_coordinates: Some(vec![[0.0, 0.0]]),
        }),
        Err(DeclarationPreparationError::Algorithm(
            CenterOfGravityPlanError::CoordinateCount {
                expected: 2,
                actual: 1,
            }
        ))
    ));
}
