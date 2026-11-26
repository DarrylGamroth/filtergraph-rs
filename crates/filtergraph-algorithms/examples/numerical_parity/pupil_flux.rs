//! Exact raw detector counts in configured logical region order.
use super::{Record, record};
use filtergraph_algorithms::AlgorithmDeclaration;
use filtergraph_algorithms::{PyramidPupilFluxU16F64 as D, PyramidPupilFluxU16F64Configuration};
#[derive(serde::Deserialize)]
pub(crate) struct Fixture {
    pub image_size: (usize, usize),
    pub pupil_size: (usize, usize),
    pub pupil_origins: Vec<[usize; 2]>,
    pub frames: Vec<Vec<u16>>,
}
pub(super) fn run(
    source: &str,
    results: &mut Vec<Record>,
) -> Result<(), Box<dyn std::error::Error>> {
    let f: Fixture = toml::from_str(source)?;
    let mut p = D::prepare(PyramidPupilFluxU16F64Configuration {
        image_size: f.image_size,
        pupil_size: f.pupil_size,
        pupil_origins: f.pupil_origins,
    })?;
    let mut out = [0.0; 4];
    for (i, frame) in f.frames.iter().enumerate() {
        D::process(&mut p, frame, &mut out)?;
        record(results, D::NAME, format!("frame-{i}"), out.to_vec());
    }
    p.reset();
    D::process(&mut p, &f.frames[0], &mut out)?;
    record(results, D::NAME, "after-reset".into(), out.to_vec());
    Ok(())
}
