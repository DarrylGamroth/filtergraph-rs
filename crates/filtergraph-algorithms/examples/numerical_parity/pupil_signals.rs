//! Real-plan comparisons; only fixture construction and logical-layout conversion live here.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmOutput, BiOEdgeImageError as BE,
    BiOEdgeImageF32 as B, BiOEdgeImageF32Configuration as BC, CurvaturePairedChannelF32 as C,
    CurvaturePairedImageF32 as I, CurvaturePairedSignalError as CE, DeclarationProcessError as DE,
    PortDirection, PreparedAlgorithm, PyramidImageError as PE, PyramidImageF32 as P,
    PyramidImageF32Configuration as PC, ZernikePupilSignalError as ZE, ZernikePupilSignalF32 as Z,
    ZernikePupilSignalF32Configuration as ZC,
};
use serde::de::DeserializeOwned;
use std::{error::Error, fmt::Debug};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn field<T: DeserializeOwned>(v: &toml::Value, name: &str) -> Result<T> {
    Ok(v.get(name)
        .ok_or_else(|| format!("missing fixture {name}"))?
        .clone()
        .try_into()?)
}
#[path = "../support/fixture_configuration.rs"]
mod fixture_configuration;
use fixture_configuration::configuration;

const QUAD_FIELDS: &[&str] = &[
    "image_size",
    "pupil_size",
    "pupil_origins",
    "i4q_support",
    "initial_reference_signal",
    "initial_optical_gain",
    "normalization_policy",
    "incidence_flux",
];
const ZERNIKE_FIELDS: &[&str] = &[
    "image_size",
    "zernike_pupil_support",
    "initial_reference_pupil_image",
    "normalization_policy",
    "incidence_flux",
    "initial_calibration_signature",
];
const CURVATURE_FIELDS: &[&str] = &[
    "image_size",
    "signal_size",
    "branch_size",
    "positive_branch_origin",
    "negative_branch_origin",
    "input_branch_rows",
    "input_channels",
    "positive_branch_row",
    "negative_branch_row",
    "channel_offsets",
    "curvature_support",
    "initial_reference_signal",
    "initial_branch_scales",
    "initial_calibration_signature",
];
fn row_major<T: Copy>(values: &[T], size: (usize, usize)) -> Vec<T> {
    (0..size.0)
        .flat_map(|r| (0..size.1).map(move |c| values[r + c * size.0]))
        .collect()
}
fn sample<D>(
    p: &mut PreparedAlgorithm<D>,
    label: &str,
    case: &str,
    image: &[f32],
    signature: Option<u64>,
    reason: fn(&D::ProcessError) -> Option<u8>,
    results: &mut Vec<Record>,
) -> Result<()>
where
    D: AlgorithmDeclaration,
    D::ProcessError: Error + 'static,
{
    let signature = signature.map(|v| [v]);
    let mut inputs = vec![AlgorithmInput::from(image)];
    if let Some(ref v) = signature {
        inputs.push(AlgorithmInput::from(v.as_slice()));
    }
    let ports: Vec<_> = D::PORTS
        .iter()
        .zip(p.formats())
        .filter(|(port, _)| port.direction() == PortDirection::Output)
        .collect();
    let mut buffers: Vec<_> = ports
        .iter()
        .map(|(_, f)| vec![71.0_f32; f.elements()])
        .collect();
    let names: Vec<_> = ports.iter().map(|(port, _)| port.name()).collect();
    let mut outputs: Vec<_> = buffers
        .iter_mut()
        .map(|v| AlgorithmOutput::from(v.as_mut_slice()))
        .collect();
    let outcome = p.process(&inputs, &mut outputs);
    drop(outputs);
    let code = match outcome {
        Ok(_) => 0,
        Err(e) => reason(&e).ok_or_else(|| format!("{label}/{case}: unexpected {e}"))?,
    };
    let preserved = code == 0 || buffers.iter().all(|v| v.iter().all(|&v| v == 71.0));
    record(
        results,
        label,
        format!("{case}-status"),
        vec![code as f64, if preserved { 1.0 } else { 0.0 }],
    );
    for (name, values) in names.iter().zip(buffers) {
        record(
            results,
            label,
            format!("{case}-{name}"),
            values.into_iter().map(f64::from).collect(),
        );
    }
    Ok(())
}
fn parameter<D>(p: &mut PreparedAlgorithm<D>, port: usize, value: AlgorithmInput<'_>) -> Result<()>
where
    D: AlgorithmDeclaration,
    D::ParameterError: Error + 'static,
{
    p.replace_parameter(port, value)
        .ok_or("missing parameter")??;
    Ok(())
}
fn invalid_parameter<D>(
    p: &mut PreparedAlgorithm<D>,
    label: &str,
    case: &str,
    port: usize,
    value: AlgorithmInput<'_>,
    results: &mut Vec<Record>,
) -> Result<()>
where
    D: AlgorithmDeclaration,
    D::ParameterError: Debug,
{
    let rejected = p
        .replace_parameter(port, value)
        .ok_or("missing parameter")?
        .is_err();
    record(
        results,
        label,
        format!("{case}-status"),
        vec![if rejected { 0.0 } else { 1.0 }],
    );
    Ok(())
}
fn pyramid_reason(e: &DE<PE>) -> Option<u8> {
    match e {
        DE::Algorithm(PE::NonFiniteInput) => Some(1),
        DE::Algorithm(PE::InvalidNormalization) => Some(2),
        DE::Algorithm(PE::NonFiniteOutput) => Some(3),
        _ => None,
    }
}
fn bio_reason(e: &DE<BE>) -> Option<u8> {
    match e {
        DE::Algorithm(BE::CalibrationMismatch) => Some(1),
        DE::Algorithm(BE::NonFiniteInput) => Some(2),
        DE::Algorithm(BE::InvalidNormalization) => Some(3),
        DE::Algorithm(BE::NonFiniteOutput) => Some(4),
        _ => None,
    }
}
fn zernike_reason(e: &DE<ZE>) -> Option<u8> {
    match e {
        DE::Algorithm(ZE::CalibrationMismatch) => Some(1),
        DE::Algorithm(ZE::NonFiniteInput) => Some(2),
        DE::Algorithm(ZE::InvalidNormalization) => Some(3),
        DE::Algorithm(ZE::NonFiniteOutput) => Some(4),
        _ => None,
    }
}
fn curvature_reason(e: &DE<CE>) -> Option<u8> {
    match e {
        DE::Algorithm(CE::CalibrationMismatch) => Some(1),
        DE::Algorithm(CE::NonFiniteInput) => Some(2),
        DE::Algorithm(CE::NonFiniteOutput) => Some(3),
        _ => None,
    }
}

fn quad<D>(
    mut p: PreparedAlgorithm<D>,
    q: &toml::Value,
    (label, prefix): (&str, &str),
    signature: Option<u64>,
    base: usize,
    reason: fn(&D::ProcessError) -> Option<u8>,
    results: &mut Vec<Record>,
) -> Result<()>
where
    D: AlgorithmDeclaration,
    D::ProcessError: Error + 'static,
    D::ParameterError: Error + 'static,
{
    let image: Vec<f32> = field(q, "image")?;
    let size: (usize, usize) = field(q, "pupil_size")?;
    let image_size: (usize, usize) = field(q, "image_size")?;
    let selected = field::<Vec<bool>>(q, "i4q_support")?
        .into_iter()
        .filter(|&v| v)
        .count();
    macro_rules! emit {
        ($case:expr,$input:expr) => {
            sample(
                &mut p,
                label,
                &format!("{prefix}-{}", $case),
                $input,
                signature,
                reason,
                results,
            )?
        };
    }
    emit!("ordinary", &image);
    let mut outside = image.clone();
    outside[2] = f32::NAN;
    emit!("outside-nan", &outside);
    let mut inside = image.clone();
    inside[image_size.1] = f32::NAN;
    emit!("masked-pupil-nan", &inside);
    emit!("zero", &vec![0.0; image.len()]);
    emit!("tiny", &vec![f32::EPSILON / 16.0; image.len()]);
    emit!("mean-overflow", &vec![f32::MAX; image.len()]);
    let mut overflow = image.clone();
    overflow[0] = f32::MAX;
    overflow[2 * image_size.1] = -f32::MAX;
    overflow[2 * image_size.1 + 3] = 0.0;
    overflow[3] = 0.0;
    emit!("differential-overflow", &overflow);
    emit!("recovery", &image);
    if let Some(s) = signature {
        sample(
            &mut p,
            label,
            &format!("{prefix}-signature-priority"),
            &inside,
            Some(s + 1),
            reason,
            results,
        )?;
    }
    let origins: Vec<[u32; 2]> = field(q, "replacement_origins")?;
    let origins: Vec<_> = origins.into_iter().flatten().collect();
    parameter(&mut p, base, AlgorithmInput::from(origins.as_slice()))?;
    emit!("origins", &image);
    let mut bad = origins;
    bad[0] = image_size.0 as u32;
    invalid_parameter(
        &mut p,
        label,
        &format!("{prefix}-invalid-origins"),
        base,
        AlgorithmInput::from(bad.as_slice()),
        results,
    )?;
    emit!("after-invalid-origins", &image);
    let mask: Vec<bool> = field(q, "replacement_support")?;
    let mask = row_major(&mask, size);
    parameter(&mut p, base + 1, AlgorithmInput::from(mask.as_slice()))?;
    emit!("support", &image);
    invalid_parameter(
        &mut p,
        label,
        &format!("{prefix}-invalid-support"),
        base + 1,
        AlgorithmInput::from(vec![false; mask.len()].as_slice()),
        results,
    )?;
    emit!("after-invalid-support", &image);
    for (offset, name, key) in [
        (2, "reference", "replacement_reference"),
        (3, "gain", "replacement_gain"),
    ] {
        let values: Vec<f32> = field(q, key)?;
        let values = row_major(&values, (selected, 2));
        parameter(
            &mut p,
            base + offset,
            AlgorithmInput::from(values.as_slice()),
        )?;
        emit!(name, &image);
        let mut bad = values;
        bad[0] = f32::NAN;
        invalid_parameter(
            &mut p,
            label,
            &format!("{prefix}-invalid-{name}"),
            base + offset,
            AlgorithmInput::from(bad.as_slice()),
            results,
        )?;
        emit!(&format!("after-invalid-{name}"), &image);
    }
    if let Some(s) = signature {
        parameter(&mut p, base + 4, AlgorithmInput::from([s + 10].as_slice()))?;
        sample(
            &mut p,
            label,
            &format!("{prefix}-old-signature"),
            &image,
            Some(s),
            reason,
            results,
        )?;
        sample(
            &mut p,
            label,
            &format!("{prefix}-new-signature"),
            &image,
            Some(s + 10),
            reason,
            results,
        )?;
        invalid_parameter(
            &mut p,
            label,
            &format!("{prefix}-invalid-signature"),
            base + 4,
            AlgorithmInput::from([0u64; 0].as_slice()),
            results,
        )?;
        p.reset();
        sample(
            &mut p,
            label,
            &format!("{prefix}-reset"),
            &image,
            Some(s + 10),
            reason,
            results,
        )?;
    } else {
        p.reset();
        emit!("reset", &image);
    }
    Ok(())
}

pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<()> {
    let f: toml::Value = toml::from_str(source)?;
    let q = &f["quad"];
    for policy in ["mean-valid-flux", "incidence-flux"] {
        let mut c: PC = configuration::<P>(q, QUAD_FIELDS)?;
        c.normalization_policy = policy.into();
        quad(
            P::prepare(c.clone())?,
            q,
            ("pyramid-image-f32", policy),
            None,
            3,
            pyramid_reason,
            results,
        )?;
        let b = BC {
            image_size: c.image_size,
            pupil_size: c.pupil_size,
            pupil_origins: c.pupil_origins,
            bi_o_edge_support: c.i4q_support,
            initial_reference_signal: c.initial_reference_signal,
            initial_optical_gain: c.initial_optical_gain,
            normalization_policy: c.normalization_policy,
            incidence_flux: c.incidence_flux,
            initial_calibration_signature: field(q, "signature")?,
        };
        quad(
            B::prepare(b)?,
            q,
            ("bi-o-edge-image-f32", policy),
            Some(field(q, "signature")?),
            4,
            bio_reason,
            results,
        )?;
    }
    let columns: usize = field(&f["tree"], "columns")?;
    let sparse: Vec<(usize, f32)> = field(&f["tree"], "q1_sparse")?;
    let mut input = vec![0.0; 4 * columns];
    for (i, v) in sparse {
        input[i] = v;
    }
    let c = PC {
        image_size: (4, columns),
        pupil_size: (1, columns),
        pupil_origins: vec![[0, 0], [1, 0], [2, 0], [3, 0]],
        i4q_support: None,
        initial_reference_signal: None,
        initial_optical_gain: None,
        normalization_policy: "mean-valid-flux".into(),
        incidence_flux: 0.0,
    };
    sample(
        &mut P::prepare(c.clone())?,
        "pyramid-image-f32",
        "lane-tree",
        &input,
        None,
        pyramid_reason,
        results,
    )?;
    let b = BC {
        image_size: c.image_size,
        pupil_size: c.pupil_size,
        pupil_origins: c.pupil_origins,
        bi_o_edge_support: None,
        initial_reference_signal: None,
        initial_optical_gain: None,
        normalization_policy: c.normalization_policy,
        incidence_flux: 0.0,
        initial_calibration_signature: 7,
    };
    sample(
        &mut B::prepare(b)?,
        "bi-o-edge-image-f32",
        "lane-tree",
        &input,
        Some(7),
        bio_reason,
        results,
    )?;
    zernike(&f["zernike"], results)?;
    curvature::<I>(
        &f["curvature_image"],
        I::prepare(configuration::<I>(&f["curvature_image"], CURVATURE_FIELDS)?)?,
        "curvature-paired-image-f32",
        (4, 8),
        false,
        results,
    )?;
    curvature::<C>(
        &f["curvature_channel"],
        C::prepare(configuration::<C>(
            &f["curvature_channel"],
            CURVATURE_FIELDS,
        )?)?,
        "curvature-paired-channel-f32",
        (3, 5),
        true,
        results,
    )?;
    Ok(())
}

fn zernike(f: &toml::Value, results: &mut Vec<Record>) -> Result<()> {
    let image: Vec<f32> = field(f, "image")?;
    let label = "zernike-pupil-signal-f32";
    for policy in ["mean-valid-flux", "incidence-flux"] {
        let mut c: ZC = configuration::<Z>(f, ZERNIKE_FIELDS)?;
        c.normalization_policy = policy.into();
        let size = c.image_size;
        let signature = c.initial_calibration_signature;
        let mut p = Z::prepare(c)?;
        macro_rules! emit {
            ($phase:expr,$input:expr) => {
                sample(
                    &mut p,
                    label,
                    &format!("{policy}-{}", $phase),
                    $input,
                    Some(signature),
                    zernike_reason,
                    results,
                )?
            };
        }
        emit!("ordinary", &image);
        emit!("zero", &vec![0.0; image.len()]);
        emit!("negative", &vec![-1.0; image.len()]);
        let mut bad = image.clone();
        bad[1] = f32::NAN;
        emit!("masked-nan", &bad);
        sample(
            &mut p,
            label,
            &format!("{policy}-signature-priority"),
            &bad,
            Some(signature + 1),
            zernike_reason,
            results,
        )?;
        emit!("normalization-overflow", &vec![f32::MAX; image.len()]);
        emit!("recovery", &image);
        for (name, port, value) in [
            ("support", 4, AlgorithmInput::Bool8(&[false; 6])),
            ("reference", 5, AlgorithmInput::F32(&[f32::NAN; 6])),
            ("flux", 6, AlgorithmInput::F32(&[-1.0_f32])),
            ("signature", 7, AlgorithmInput::U64(&[])),
        ] {
            invalid_parameter(
                &mut p,
                label,
                &format!("{policy}-invalid-{name}"),
                port,
                value,
                results,
            )?;
            emit!(&format!("after-invalid-{name}"), &image);
        }
        let new_support: Vec<bool> = field(f, "replacement_support")?;
        let new_support = row_major(&new_support, size);
        let reference: Vec<f32> = field(f, "replacement_reference")?;
        let reference = row_major(&reference, size);
        let flux = [field::<f32>(f, "replacement_flux")?];
        let new_signature = [field::<u64>(f, "replacement_signature")?];
        let invalid_flux = [-1.0_f32];
        let rejected = p
            .replace_parameters(&[
                (7, AlgorithmInput::from(new_signature.as_slice())),
                (6, AlgorithmInput::from(invalid_flux.as_slice())),
            ])
            .is_err();
        record(
            results,
            label,
            format!("{policy}-invalid-transaction-status"),
            vec![if rejected { 0.0 } else { 1.0 }],
        );
        emit!("after-invalid-transaction", &image);
        p.replace_parameters(&[
            (4, AlgorithmInput::from(new_support.as_slice())),
            (5, AlgorithmInput::from(reference.as_slice())),
            (6, AlgorithmInput::from(flux.as_slice())),
            (7, AlgorithmInput::from(new_signature.as_slice())),
        ])?;
        emit!("old-signature", &image);
        sample(
            &mut p,
            label,
            &format!("{policy}-new-calibration"),
            &image,
            Some(new_signature[0]),
            zernike_reason,
            results,
        )?;
        p.reset();
        sample(
            &mut p,
            label,
            &format!("{policy}-reset"),
            &image,
            Some(new_signature[0]),
            zernike_reason,
            results,
        )?;
    }
    for (name, flux) in [
        ("zero", 0.0),
        ("epsilon", f32::EPSILON),
        ("above-epsilon", f32::from_bits(f32::EPSILON.to_bits() + 1)),
    ] {
        let mut c: ZC = configuration::<Z>(f, ZERNIKE_FIELDS)?;
        c.normalization_policy = "incidence-flux".into();
        c.incidence_flux = flux;
        let signature = c.initial_calibration_signature;
        sample(
            &mut Z::prepare(c)?,
            label,
            &format!("incidence-{name}"),
            &image,
            Some(signature),
            zernike_reason,
            results,
        )?;
    }
    Ok(())
}

fn curvature<D>(
    f: &toml::Value,
    mut p: PreparedAlgorithm<D>,
    label: &str,
    input_size: (usize, usize),
    channels: bool,
    results: &mut Vec<Record>,
) -> Result<()>
where
    D: AlgorithmDeclaration<ProcessError = DE<CE>>,
    D::ParameterError: Error + 'static,
{
    let image: Vec<f32> = field(f, "input")?;
    let signature: u64 = field(f, "initial_calibration_signature")?;
    let size: (usize, usize) = field(f, "signal_size")?;
    macro_rules! emit {
        ($phase:expr,$input:expr) => {
            sample(
                &mut p,
                label,
                $phase,
                $input,
                Some(signature),
                curvature_reason,
                results,
            )?
        };
    }
    emit!("ordinary", &image);
    emit!("zero", &vec![0.0; image.len()]);
    emit!("negative", &vec![-1.0; image.len()]);
    let mut bad = image.clone();
    bad[0] = f32::NAN;
    emit!("nonfinite", &bad);
    sample(
        &mut p,
        label,
        "signature-priority",
        &bad,
        Some(signature + 1),
        curvature_reason,
        results,
    )?;
    emit!("reduction-overflow", &vec![f32::MAX; image.len()]);
    emit!("recovery", &image);
    if channels {
        parameter(&mut p, 5, AlgorithmInput::from([1.0_f32; 2].as_slice()))?;
        let pr: usize = field(f, "positive_branch_row")?;
        let mr: usize = field(f, "negative_branch_row")?;
        for (name, minus) in [
            ("sum-overflow", f32::MAX),
            ("difference-overflow", -f32::MAX * 0.5),
            ("low-total", -f32::MAX),
        ] {
            let mut input = vec![0.0; input_size.0 * input_size.1];
            for col in 0..input_size.1 {
                input[pr * input_size.1 + col] = f32::MAX;
                input[mr * input_size.1 + col] = minus;
            }
            emit!(name, &input);
        }
    }
    for (name, port, value) in [
        ("support", 3, AlgorithmInput::Bool8(&[false; 3])),
        ("reference", 4, AlgorithmInput::F32(&[f32::NAN; 4])),
        ("scales", 5, AlgorithmInput::F32(&[0.0_f32, 1.0])),
        ("signature", 6, AlgorithmInput::U64(&[])),
    ] {
        invalid_parameter(
            &mut p,
            label,
            &format!("invalid-{name}"),
            port,
            value,
            results,
        )?;
        emit!(&format!("after-invalid-{name}"), &image);
    }
    parameter(
        &mut p,
        3,
        AlgorithmInput::from(vec![false; size.0 * size.1].as_slice()),
    )?;
    emit!("empty-support", &image);
    emit!("empty-support-nonfinite", &bad);
    let mask: Vec<bool> = field(f, "replacement_support")?;
    let mask = row_major(&mask, size);
    let reference: Vec<f32> = field(f, "replacement_reference")?;
    let reference = row_major(&reference, size);
    let scales: Vec<f32> = field(f, "replacement_scales")?;
    let new_signature = [field::<u64>(f, "replacement_signature")?];
    let rejected = p
        .replace_parameters(&[
            (6, AlgorithmInput::from(new_signature.as_slice())),
            (5, AlgorithmInput::from([-1.0_f32, 1.0].as_slice())),
        ])
        .is_err();
    record(
        results,
        label,
        "invalid-transaction-status".into(),
        vec![if rejected { 0.0 } else { 1.0 }],
    );
    emit!("after-invalid-transaction", &image);
    p.replace_parameters(&[
        (3, AlgorithmInput::from(mask.as_slice())),
        (4, AlgorithmInput::from(reference.as_slice())),
        (5, AlgorithmInput::from(scales.as_slice())),
        (6, AlgorithmInput::from(new_signature.as_slice())),
    ])?;
    emit!("old-signature", &image);
    sample(
        &mut p,
        label,
        "new-calibration",
        &image,
        Some(new_signature[0]),
        curvature_reason,
        results,
    )?;
    p.reset();
    sample(
        &mut p,
        label,
        "reset",
        &image,
        Some(new_signature[0]),
        curvature_reason,
        results,
    )?;
    Ok(())
}
