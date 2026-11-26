//! Fixed frame delay: logical outputs, committed ring history, and rejected samples.
use super::{Record, record};
use filtergraph_algorithms::{
    AlgorithmPlan, FrameDelayError, FrameDelayF32, FrameDelayF32Configuration, FrameDelayPlan,
    FrameDelayWorkspace,
};
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize)]
struct Fixture {
    delays: Vec<usize>,
    initial_state: Vec<f64>,
    output_sentinel: f64,
    reset_before_frame: usize,
    reject_before_frame: usize,
    frames: Vec<Vec<f64>>,
}

pub(super) fn run(source: &str, results: &mut Vec<Record>) -> Result<(), Box<dyn Error>> {
    let f: Fixture = toml::from_str(source)?;
    macro_rules! precision {
        ($t:ty, $label:literal) => {{
            for &delay in &f.delays {
                let plan = FrameDelayPlan::<$t>::new(delay);
                let mut workspace = FrameDelayWorkspace::new(
                    f.initial_state.iter().map(|&v| v as $t).collect(),
                    &plan,
                )?;
                let mut output = vec![f.output_sentinel as $t; f.initial_state.len()];
                macro_rules! history {
                    ($phase:expr) => {{
                        // Julia history is [coordinate, ring frame] in column-major storage.
                        if delay != 0 {
                            record(
                                results,
                                $label,
                                format!("delay-{delay}-{}-history", $phase),
                                workspace.history().iter().map(|&v| v as f64).collect(),
                            );
                        }
                        record(
                            results,
                            $label,
                            format!("delay-{delay}-{}-head-status", $phase),
                            vec![workspace.head() as f64],
                        );
                    }};
                }
                history!("initial");
                for (frame, values) in f.frames.iter().enumerate() {
                    if frame == f.reset_before_frame {
                        workspace.reset();
                        history!("reset");
                    }
                    if frame == f.reject_before_frame {
                        for (name, value) in [
                            ("nan", <$t>::NAN),
                            ("positive-inf", <$t>::INFINITY),
                            ("negative-inf", <$t>::NEG_INFINITY),
                        ] {
                            let mut input: Vec<$t> = values.iter().map(|&v| v as $t).collect();
                            *input.last_mut().ok_or("empty delay frame")? = value;
                            output.fill(f.output_sentinel as $t);
                            let previous = workspace.clone();
                            let outcome = plan.process(&mut output, &mut workspace, &input);
                            record(
                                results,
                                $label,
                                format!("delay-{delay}-{name}-status"),
                                vec![
                                    if matches!(outcome, Err(FrameDelayError::NonFiniteInput)) {
                                        0.0
                                    } else {
                                        1.0
                                    },
                                    if output.iter().all(|&v| v == f.output_sentinel as $t) {
                                        1.0
                                    } else {
                                        0.0
                                    },
                                    if workspace == previous { 1.0 } else { 0.0 },
                                ],
                            );
                            history!(name);
                        }
                    }
                    let input: Vec<$t> = values.iter().map(|&v| v as $t).collect();
                    plan.process(&mut output, &mut workspace, &input)?;
                    record(
                        results,
                        $label,
                        format!("delay-{delay}-frame-{frame}-output"),
                        output.iter().map(|&v| v as f64).collect(),
                    );
                    history!(format!("frame-{frame}"));
                }
            }
        }};
    }
    precision!(f32, "frame-delay-f32");
    precision!(f64, "frame-delay-f64");
    // Rejected declaration configuration must fail before allocating huge history.
    for (name, extent, delay, initial) in [
        ("empty", 0, 1, 0.0),
        ("nonfinite", 1, 1, f32::NAN),
        ("history-limit", 1, u32::MAX as usize / 4 + 1, 0.0),
    ] {
        let rejected = FrameDelayF32::prepare(FrameDelayF32Configuration {
            extent,
            delay_frames: delay,
            initial_value: initial,
            input_schema: "org.example.parity.vector/1".into(),
            output_schema: "org.example.parity.vector/1".into(),
        })
        .is_err();
        record(
            results,
            "frame-delay-f32",
            format!("configuration-{name}-status"),
            vec![if rejected { 0.0 } else { 1.0 }],
        );
    }
    Ok(())
}
