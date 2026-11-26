//! Cold field inspection uses the same defaults as configuration construction.
use filtergraph_algorithms::{
    AlgorithmConfiguration, AlgorithmDeclaration, CenterOfGravityF32,
    CenterOfGravityF32Configuration, ChoppedImageDifferenceF32,
    ChoppedImageDifferenceF32Configuration, ConfigurationUpdate, ConfigurationValue,
    NdarrayCopyError, NdarrayCopyPlan, NoWorkspace, PixelCalibrationU16F32,
    PixelCalibrationU16F32Configuration, ShackHartmannWeightedCentroidF32,
    ShackHartmannWeightedCentroidF32Configuration,
};
use serde_json::{Value, json};

fn field<D: AlgorithmDeclaration>(name: &str) -> &'static AlgorithmConfiguration {
    D::CONFIGURATION
        .iter()
        .find(|field| field.name() == name)
        .unwrap()
}

#[test]
fn catalog_defaults_match_omitted_and_explicit_configuration_fields() {
    let minimum = json!({"pixel_count": 1, "trigger_high_is_fringed": true});
    let implicit: ChoppedImageDifferenceF32Configuration =
        serde_json::from_value(minimum.clone()).unwrap();
    let mut explicit = minimum;
    for descriptor in ChoppedImageDifferenceF32::CONFIGURATION {
        match descriptor.default_value() {
            Some(default) => {
                assert!(!descriptor.required());
                explicit[descriptor.name()] = serde_json::to_value(default).unwrap();
            }
            None => assert!(descriptor.required()),
        }
    }
    let explicit: ChoppedImageDifferenceF32Configuration =
        serde_json::from_value(explicit).unwrap();
    assert_eq!(implicit, explicit);
    assert_eq!(implicit.input_schema, "pipewireao/calibrated-pixels-f32/1");
    assert!(!implicit.require_acquisition_identity);
    assert_eq!(
        field::<ChoppedImageDifferenceF32>("input_schema").update(),
        ConfigurationUpdate::GraphRebuild
    );

    let minimum = json!({
        "image_size": [2, 2], "image_schema": "test.image/1", "subaperture_size": [2, 2],
        "subaperture_count": 1, "initial_subaperture_origins": null,
        "coordinate_scale": 1.0, "pixel_threshold": 0.0, "flux_threshold": 0.0,
        "reference_slopes": null, "active": null, "initial_weights": [[1.0, 1.0, 1.0, 1.0]]
    });
    let implicit: ShackHartmannWeightedCentroidF32Configuration =
        serde_json::from_value(minimum.clone()).unwrap();
    let descriptor = field::<ShackHartmannWeightedCentroidF32>("simd_width");
    let mut explicit = minimum;
    explicit[descriptor.name()] =
        serde_json::to_value(descriptor.default_value().unwrap()).unwrap();
    let explicit: ShackHartmannWeightedCentroidF32Configuration =
        serde_json::from_value(explicit).unwrap();
    assert_eq!(implicit, explicit);
    assert_eq!(implicit.simd_width, 0);
    assert_eq!(descriptor.update(), ConfigurationUpdate::Construction);
    ShackHartmannWeightedCentroidF32::prepare(explicit.clone()).unwrap();
    for width in [4, 8, 16] {
        let mut configuration = explicit.clone();
        configuration.simd_width = width;
        ShackHartmannWeightedCentroidF32::prepare(configuration).unwrap();
    }
    let mut invalid = explicit;
    invalid.simd_width = 3;
    assert!(ShackHartmannWeightedCentroidF32::prepare(invalid).is_err());
}

#[test]
fn optional_type_does_not_make_a_required_field_optional() {
    let complete = json!({"image_size": [1, 1], "initial_flat": null, "initial_background": null});
    let configuration: PixelCalibrationU16F32Configuration =
        serde_json::from_value(complete.clone()).unwrap();
    assert_eq!(configuration.initial_flat, None);
    PixelCalibrationU16F32::prepare(configuration).unwrap();
    for name in ["initial_flat", "initial_background"] {
        let descriptor = field::<PixelCalibrationU16F32>(name);
        assert!(descriptor.required());
        assert_eq!(descriptor.default_value(), None);
        let mut missing = complete.clone();
        missing.as_object_mut().unwrap().remove(name);
        let error =
            serde_json::from_value::<PixelCalibrationU16F32Configuration>(missing).unwrap_err();
        assert!(error.to_string().contains(name), "{error}");
    }
    let center: CenterOfGravityF32Configuration = serde_json::from_value(json!({
        "pixel_count": 2, "initial_coordinates": null,
    }))
    .unwrap();
    assert!(field::<CenterOfGravityF32>("initial_coordinates").required());
    let mut center = CenterOfGravityF32::prepare(center).unwrap();
    let (mut centroid, mut flux, mut validity) = ([123.0; 2], [0.0], [false]);
    CenterOfGravityF32::process(
        &mut center,
        &[1.0, 2.0],
        &mut centroid,
        &mut flux,
        &mut validity,
    )
    .unwrap();
    assert_eq!((centroid, flux, validity), ([0.0; 2], [3.0], [true]));
    let mut old_names = complete;
    old_names.as_object_mut().unwrap().remove("image_size");
    old_names["image_rows"] = json!(1);
    old_names["image_columns"] = json!(1);
    assert!(serde_json::from_value::<PixelCalibrationU16F32Configuration>(old_names).is_err());
}

filtergraph_algorithms::filtergraph_algorithm! {
    Defaults {
        label = "configuration-defaults-test";
        configuration DefaultsConfiguration {
            required_optional: Option<usize> = Construction;
            absent: Option<usize> = Construction, default = None;
            count: usize = GraphRebuild, default = 1;
        }
        plan = NdarrayCopyPlan;
        properties = none;
        dimensions |plan, _workspace| { extent = plan.extent(); }
        inputs { input: f32 [extent] = "input", schema = "test.input/1"; }
        outputs { output: f32 [extent] = "output", schema = "test.output/1"; }
        prepare |configuration| {
            let _ = (configuration.required_optional, configuration.absent);
            Ok((NdarrayCopyPlan::new(configuration.count)?, NoWorkspace))
        }
        prepare_error = NdarrayCopyError;
        process |plan, workspace| { filtergraph_algorithms::AlgorithmPlan::process(plan, output, workspace, input) }
        process_error = NdarrayCopyError;
    }
}

#[test]
fn explicit_none_default_is_distinct_from_no_default() {
    let descriptor = field::<Defaults>("absent");
    assert!(!descriptor.required());
    assert_eq!(descriptor.default_value(), Some(ConfigurationValue::None));
    assert_eq!(
        serde_json::to_value(descriptor.default_value().unwrap()).unwrap(),
        Value::Null
    );
    let parsed: DefaultsConfiguration =
        serde_json::from_value(json!({"required_optional": null})).unwrap();
    assert_eq!(parsed.absent, None);
    assert_eq!(parsed.count, 1);
    assert!(serde_json::from_value::<DefaultsConfiguration>(json!({})).is_err());
    assert!(
        serde_json::from_value::<DefaultsConfiguration>(
            json!({"required_optional": null, "count": null})
        )
        .is_err()
    );
    assert!(
        serde_json::from_value::<DefaultsConfiguration>(
            json!({"required_optional": null, "unknown": 2})
        )
        .is_err()
    );
    Defaults::prepare(parsed).unwrap();
}
