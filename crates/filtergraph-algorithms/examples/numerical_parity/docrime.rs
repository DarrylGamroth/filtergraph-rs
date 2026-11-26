//! DO-CRIME real-plan differential inputs; records use logical row-major matrices.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmPlan, DocrimeBinaryExcitationPlan, DocrimeClosedLoopInteractionMatrixPlan,
    DocrimeConditioningCriteria, DocrimeCovarianceError, DocrimeCovarianceInput,
    DocrimeCovarianceInputKind, DocrimeCovariancePlan, DocrimeCovariances, DocrimeExcitationPlan,
    DocrimeInteractionMatrixDiagnostics, DocrimeInteractionMatrixError,
    DocrimeInteractionMatrixInputKind, DocrimeInteractionMatrixOutput,
    DocrimeInteractionMatrixStatus, DocrimeOpenLoopInteractionMatrixPlan,
};
use serde::Deserialize;
use std::error::Error;
#[derive(Deserialize)]
struct Fixture {
    excitation: Excitation,
    covariance: Covariance,
    solve: Solve,
}
#[derive(Deserialize)]
struct Excitation {
    seed: u64,
    amplitudes: Vec<f64>,
    replacement_amplitudes: Vec<f64>,
    frames: usize,
    replace_before_frame: usize,
    sentinel: f64,
}
#[derive(Deserialize)]
struct Covariance {
    measurement_count: usize,
    corrector_count: usize,
    sample_limit: usize,
    measurements: Vec<Vec<f64>>,
    correctors: Vec<Vec<f64>>,
}
#[derive(Deserialize)]
struct Solve {
    measurement_count: usize,
    corrector_count: usize,
    cross_covariance_sum: Vec<f64>,
    minimum_corrector_variance: f64,
    maximum_absolute_corrector_correlation: f64,
    minimum_factorization_pivot_ratio: f64,
    loop_gain: f64,
    replacement_gain: f64,
    sentinel: f64,
    cases: Vec<SolveCase>,
}
#[derive(Deserialize)]
struct SolveCase {
    name: String,
    sample_count: usize,
    corrector_covariance_sum: Vec<f64>,
}
fn solve_status(s: DocrimeInteractionMatrixStatus) -> Vec<f64> {
    let (code, first, second) = match s {
        DocrimeInteractionMatrixStatus::Valid => (0, 0, 0),
        DocrimeInteractionMatrixStatus::InsufficientSamples => (1, 0, 0),
        DocrimeInteractionMatrixStatus::InsufficientCorrectorVariance {
            corrector_coordinate,
        } => (2, corrector_coordinate, 0),
        DocrimeInteractionMatrixStatus::ExcessiveCorrectorCorrelation { first, second } => {
            (3, first, second)
        }
        DocrimeInteractionMatrixStatus::NonPositiveDefinite {
            corrector_coordinate,
        } => (4, corrector_coordinate, 0),
        DocrimeInteractionMatrixStatus::InsufficientFactorizationPivot {
            corrector_coordinate,
        } => (5, corrector_coordinate, 0),
    };
    vec![code as f64, first as f64, second as f64]
}
pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    macro_rules! precision {
        ($t:ty,$suffix:literal) => {{
            let e = &f.excitation;
            macro_rules! excitation {
                ($constructor:ident,$kind:literal) => {{
                    let label = concat!($kind, "-", $suffix);
                    let mut plan = $constructor::new(
                        e.amplitudes.iter().map(|&x| x as $t).collect::<Vec<_>>(),
                        e.seed,
                    )?;
                    let mut workspace = plan.workspace();
                    let mut output = vec![e.sentinel as $t; e.amplitudes.len()];
                    for frame in 0..e.frames {
                        if frame == e.replace_before_frame {
                            plan = $constructor::new(
                                e.replacement_amplitudes
                                    .iter()
                                    .map(|&x| x as $t)
                                    .collect::<Vec<_>>(),
                                e.seed,
                            )?;
                        }
                        plan.process(&mut output, &mut workspace, ())?;
                        record(
                            results,
                            label,
                            format!("frame-{frame}-output"),
                            output.iter().map(|&x| x as f64).collect(),
                        );
                    }
                    let previous = workspace.clone();
                    let mut bad = vec![e.sentinel as $t; output.len() - 1];
                    let rejected = plan.process(&mut bad, &mut workspace, ()).is_err();
                    record(
                        results,
                        label,
                        "invalid-extent-status".into(),
                        vec![
                            (!rejected) as u8 as f64,
                            (workspace == previous) as u8 as f64,
                            bad.iter().all(|&x| x == e.sentinel as $t) as u8 as f64,
                        ],
                    );
                    plan.process(&mut output, &mut workspace, ())?;
                    record(
                        results,
                        label,
                        "after-invalid-output".into(),
                        output.iter().map(|&x| x as f64).collect(),
                    );
                    workspace = plan.workspace();
                    plan.process(&mut output, &mut workspace, ())?;
                    record(
                        results,
                        label,
                        "after-reset-output".into(),
                        output.iter().map(|&x| x as f64).collect(),
                    );
                }};
            }
            excitation!(DocrimeExcitationPlan, "docrime-excitation");
            excitation!(DocrimeBinaryExcitationPlan, "docrime-binary-excitation");
            let c = &f.covariance;
            let label = concat!("docrime-covariance-", $suffix);
            let plan = DocrimeCovariancePlan::<$t>::with_sample_limit(
                c.measurement_count,
                c.corrector_count,
                c.sample_limit,
            )?;
            let mut workspace = plan.workspace();
            macro_rules! snapshot {
                ($phase:expr) => {{
                    record(
                        results,
                        label,
                        format!("{}-measurement-mean", $phase),
                        workspace
                            .measurement_mean()
                            .iter()
                            .map(|&x| x as f64)
                            .collect(),
                    );
                    record(
                        results,
                        label,
                        format!("{}-corrector-mean", $phase),
                        workspace
                            .corrector_sample_mean()
                            .iter()
                            .map(|&x| x as f64)
                            .collect(),
                    );
                    record(
                        results,
                        label,
                        format!("{}-cross", $phase),
                        workspace
                            .measurement_corrector_cross_covariance_sum()
                            .iter()
                            .map(|&x| x as f64)
                            .collect(),
                    );
                    record(
                        results,
                        label,
                        format!("{}-corrector-covariance", $phase),
                        workspace
                            .corrector_covariance_sum()
                            .iter()
                            .map(|&x| x as f64)
                            .collect(),
                    );
                    record(
                        results,
                        label,
                        format!("{}-count-status", $phase),
                        vec![
                            workspace.sample_count() as f64,
                            (workspace.sample_count() == c.sample_limit) as u8 as f64,
                        ],
                    );
                }};
            }
            snapshot!("initial");
            for (frame, (measurement, corrector)) in
                c.measurements.iter().zip(&c.correctors).enumerate()
            {
                let measurements: Vec<$t> = measurement.iter().map(|&x| x as $t).collect();
                let correctors: Vec<$t> = corrector.iter().map(|&x| x as $t).collect();
                for (name, kind, value) in [
                    ("nan-measurement", 0, <$t>::NAN),
                    ("inf-corrector", 1, <$t>::INFINITY),
                ] {
                    let mut m = measurements.clone();
                    let mut r = correctors.clone();
                    if kind == 0 {
                        *m.last_mut().unwrap() = value;
                    } else {
                        *r.last_mut().unwrap() = value;
                    }
                    let count = workspace.sample_count();
                    let means = workspace.measurement_mean().to_vec();
                    let cross = workspace
                        .measurement_corrector_cross_covariance_sum()
                        .to_vec();
                    let rmeans = workspace.corrector_sample_mean().to_vec();
                    let rcov = workspace.corrector_covariance_sum().to_vec();
                    let outcome =
                        plan.process((), &mut workspace, DocrimeCovarianceInput::new(&m, &r));
                    let code = match outcome {
                        Ok(()) => 0,
                        Err(DocrimeCovarianceError::NonFiniteInput {
                            input: DocrimeCovarianceInputKind::Measurements,
                            ..
                        }) => 1,
                        Err(DocrimeCovarianceError::NonFiniteInput {
                            input: DocrimeCovarianceInputKind::CorrectorSample,
                            ..
                        }) => 2,
                        Err(DocrimeCovarianceError::SampleLimitReached { .. }) => 6,
                        Err(error) => return Err(error.into()),
                    };
                    record(
                        results,
                        label,
                        format!("frame-{frame}-{name}-status"),
                        vec![
                            code as f64,
                            (count == workspace.sample_count()
                                && means == workspace.measurement_mean()
                                && cross == workspace.measurement_corrector_cross_covariance_sum()
                                && rmeans == workspace.corrector_sample_mean()
                                && rcov == workspace.corrector_covariance_sum())
                                as u8 as f64,
                        ],
                    );
                }
                let outcome = plan.process(
                    (),
                    &mut workspace,
                    DocrimeCovarianceInput::new(&measurements, &correctors),
                );
                let code = match outcome {
                    Ok(()) => 0,
                    Err(DocrimeCovarianceError::SampleLimitReached { .. }) => 6,
                    Err(error) => return Err(error.into()),
                };
                record(
                    results,
                    label,
                    format!("frame-{frame}-status"),
                    vec![code as f64],
                );
                snapshot!(format!("frame-{frame}"));
            }
            workspace.reset();
            snapshot!("reset");
            let m: Vec<$t> = c.measurements[0].iter().map(|&x| x as $t).collect();
            let r: Vec<$t> = c.correctors[0].iter().map(|&x| x as $t).collect();
            plan.process((), &mut workspace, DocrimeCovarianceInput::new(&m, &r))?;
            snapshot!("after-reset");
            let previous_count=workspace.sample_count();
            let previous_mean=workspace.measurement_mean().to_vec();
            let previous_corrector=workspace.corrector_sample_mean().to_vec();
            let previous_cross=workspace.measurement_corrector_cross_covariance_sum().to_vec();
            let previous_covariance=workspace.corrector_covariance_sum().to_vec();
            let outcome=plan.process((),&mut workspace,DocrimeCovarianceInput::new(&vec![<$t>::MAX;c.measurement_count],&vec![<$t>::MAX;c.corrector_count]));
            let code=match outcome {Ok(())=>0,Err(DocrimeCovarianceError::NonFiniteCalculation)=>3,Err(error)=>return Err(error.into())};
            record(results,label,"overflow-status".into(),vec![code as f64,(workspace.sample_count()==previous_count&&workspace.measurement_mean()==previous_mean&&workspace.corrector_sample_mean()==previous_corrector&&workspace.measurement_corrector_cross_covariance_sum()==previous_cross&&workspace.corrector_covariance_sum()==previous_covariance) as u8 as f64]);
            let m:Vec<$t>=c.measurements[1].iter().map(|&x|x as $t).collect();let r:Vec<$t>=c.correctors[1].iter().map(|&x|x as $t).collect();
            plan.process((),&mut workspace,DocrimeCovarianceInput::new(&m,&r))?;snapshot!("after-overflow");
            let s = &f.solve;
            let criteria = DocrimeConditioningCriteria::new(
                s.minimum_corrector_variance as $t,
                s.maximum_absolute_corrector_correlation as $t,
                s.minimum_factorization_pivot_ratio as $t,
            )?;
            let cross: Vec<$t> = s.cross_covariance_sum.iter().map(|&x| x as $t).collect();
            macro_rules! solver {
                ($plan:expr,$kind:literal) => {{
                    let plan = $plan;
                    let label = concat!($kind, "-", $suffix);
                    let mut workspace = plan.workspace();
                    let mut output =
                        vec![s.sentinel as $t; s.measurement_count * s.corrector_count];
                    let mut diagnostics = DocrimeInteractionMatrixDiagnostics::<$t>::default();
                    for case in &s.cases {
                        let covariance: Vec<$t> = case
                            .corrector_covariance_sum
                            .iter()
                            .map(|&x| x as $t)
                            .collect();
                        plan.process(
                            DocrimeInteractionMatrixOutput::new(&mut output, &mut diagnostics),
                            &mut workspace,
                            DocrimeCovariances::new(case.sample_count, &cross, &covariance),
                        )?;
                        let mut status = solve_status(diagnostics.status());
                        status.push(diagnostics.sample_count() as f64);
                        record(results, label, format!("{}-status", case.name), status);
                        record(
                            results,
                            label,
                            format!("{}-output", case.name),
                            output.iter().map(|&x| x as f64).collect(),
                        );
                        record(
                            results,
                            label,
                            format!("{}-diagnostics", case.name),
                            vec![
                                diagnostics.minimum_corrector_variance() as f64,
                                diagnostics.maximum_corrector_variance() as f64,
                                diagnostics.maximum_absolute_corrector_correlation() as f64,
                                diagnostics.minimum_factorization_pivot_ratio() as f64,
                            ],
                        );
                    }
                    for (name, kind) in [("nan-cross", 0), ("inf-covariance", 1), ("asymmetric", 2)] {
                        let mut invalid_cross = cross.clone();
                        let mut covariance: Vec<$t> = s.cases[0].corrector_covariance_sum.iter().map(|&x|x as $t).collect();
                        match kind {0 => *invalid_cross.last_mut().unwrap() = <$t>::NAN, 1 => covariance[0] = <$t>::INFINITY, _ => covariance[1] += 0.125 as $t};
                        let previous_output = output.clone();
                        let previous_diagnostics = diagnostics;
                        let outcome = plan.process(DocrimeInteractionMatrixOutput::new(&mut output,&mut diagnostics), &mut workspace, DocrimeCovariances::new(s.cases[0].sample_count,&invalid_cross,&covariance));
                        let code=match outcome {
                            Ok(())=>0,
                            Err(DocrimeInteractionMatrixError::NonFiniteInput{input:DocrimeInteractionMatrixInputKind::MeasurementCorrectorCrossCovarianceSum,..})=>1,
                            Err(DocrimeInteractionMatrixError::NonFiniteInput{input:DocrimeInteractionMatrixInputKind::CorrectorCovarianceSum,..})=>2,
                            Err(DocrimeInteractionMatrixError::NonSymmetricCorrectorCovariance{..})=>3,
                            Err(error)=>return Err(error.into()),
                        };
                        record(results,label,format!("{name}-status"),vec![code as f64,(output==previous_output) as u8 as f64,(diagnostics==previous_diagnostics) as u8 as f64]);
                        let covariance: Vec<$t> = s.cases[0].corrector_covariance_sum.iter().map(|&x|x as $t).collect();
                        plan.process(DocrimeInteractionMatrixOutput::new(&mut output,&mut diagnostics),&mut workspace,DocrimeCovariances::new(s.cases[0].sample_count,&cross,&covariance))?;
                        record(results,label,format!("after-{name}-output"),output.iter().map(|&x|x as f64).collect());
                    }
                }};
            }
            solver!(
                DocrimeOpenLoopInteractionMatrixPlan::new(
                    s.measurement_count,
                    s.corrector_count,
                    criteria
                )?,
                "docrime-open-loop-interaction-matrix"
            );
            solver!(
                DocrimeClosedLoopInteractionMatrixPlan::new(
                    s.measurement_count,
                    s.corrector_count,
                    s.loop_gain as $t,
                    criteria
                )?,
                "docrime-closed-loop-interaction-matrix"
            );
            let replacement = DocrimeClosedLoopInteractionMatrixPlan::new(
                s.measurement_count,
                s.corrector_count,
                s.replacement_gain as $t,
                criteria,
            )?;
            let mut output = vec![s.sentinel as $t; s.measurement_count * s.corrector_count];
            let mut diagnostics = DocrimeInteractionMatrixDiagnostics::<$t>::default();
            let covariance: Vec<$t> = s.cases[0]
                .corrector_covariance_sum
                .iter()
                .map(|&x| x as $t)
                .collect();
            replacement.process(
                DocrimeInteractionMatrixOutput::new(&mut output, &mut diagnostics),
                &mut replacement.workspace(),
                DocrimeCovariances::new(s.cases[0].sample_count, &cross, &covariance),
            )?;
            record(
                results,
                concat!("docrime-closed-loop-interaction-matrix-", $suffix),
                "replacement-gain-output".into(),
                output.iter().map(|&x| x as f64).collect(),
            );
        }};
    }
    precision!(f32, "f32");
    precision!(f64, "f64");
    Ok(())
}
