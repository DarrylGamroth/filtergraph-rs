//! Adapt the frozen TOML fixture vocabulary to current typed configurations.
//!
//! TOML cannot express null. At this test/benchmark boundary only, omitted
//! nullable fixture coefficients become explicit null inputs. Production JSON
//! still rejects missing required fields and all retired configuration names.
use filtergraph_algorithms::AlgorithmDeclaration;
use serde_json::Value;
use std::error::Error;

pub fn configuration<D: AlgorithmDeclaration>(
    value: &toml::Value,
    fields: &[&str],
) -> Result<D::Configuration, Box<dyn Error>>
where
    D::Configuration: serde::de::DeserializeOwned,
{
    let mut source = value.as_table().ok_or("fixture must be a table")?.clone();
    source.retain(|key, _| fields.contains(&key));
    let mut value = serde_json::to_value(source)?;
    let object = value.as_object_mut().ok_or("fixture must be an object")?;
    for (rows, columns, size) in [
        ("image_rows", "image_columns", "image_size"),
        ("detector_rows", "detector_columns", "detector_size"),
        (
            "subaperture_rows",
            "subaperture_columns",
            "subaperture_size",
        ),
        ("pupil_rows", "pupil_columns", "pupil_size"),
        ("region_rows", "region_columns", "region_size"),
        ("lower_limit", "upper_limit", "command_limits"),
    ] {
        match (object.remove(rows), object.remove(columns)) {
            (Some(row), Some(column)) => {
                if object
                    .insert(size.into(), Value::Array(vec![row, column]))
                    .is_some()
                {
                    return Err("fixture specifies both old and current dimensions".into());
                }
            }
            (None, None) => {}
            _ => return Err("fixture dimension pair is incomplete".into()),
        }
    }
    for (old, new) in match D::NAME {
        "shack-hartmann-reconstructor-f32" => &[("actuator_count", "reconstructed_count")][..],
        "matched-calibration-differencing-f32" | "docrime-covariance-f32" => {
            &[("corrector_sample_count", "corrector_coordinate_count")][..]
        }
        _ => &[],
    } {
        if let Some(value) = object.remove(*old) {
            if object.insert((*new).into(), value).is_some() {
                return Err("fixture specifies both old and current field names".into());
            }
        }
    }
    for field in D::CONFIGURATION {
        if field.required() && field.rust_type().starts_with("Option") {
            object.entry(field.name()).or_insert(Value::Null);
        }
    }
    Ok(serde_json::from_value(value)?)
}
