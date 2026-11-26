//! DO-CRIME calibration-excitation tests.

use filtergraph_algorithms::schemas::{DOCRIME_EXCITATION_AMPLITUDE_V1, DOCRIME_EXCITATION_V1};
use filtergraph_algorithms::{
    AlgorithmDeclaration, AlgorithmInput, AlgorithmPlan, DocrimeBinaryExcitationF32,
    DocrimeBinaryExcitationF32Configuration, DocrimeBinaryExcitationPlan, DocrimeExcitationError,
    DocrimeExcitationF32, DocrimeExcitationF32Configuration, DocrimeExcitationPlan,
    DocrimeExcitationPlanError, PortDirection, PortRole, PropertyPolicy,
};

const SEED: u64 = 0x1234_5678_9abc_def0;

#[test]
fn fixed_seed_sequence_matches_the_declared_splitmix64_mapping() {
    let plan = DocrimeExcitationPlan::new([1.0_f64, 2.0, 0.5], SEED).unwrap();
    let mut workspace = plan.workspace();
    let mut excitation = [0.0; 3];

    plan.process(&mut excitation, &mut workspace, ()).unwrap();
    assert_eq!(
        excitation,
        [
            -0.827_358_007_431_030_3,
            0.710_330_009_460_449_2,
            -0.292_938_292_026_519_8,
        ]
    );

    plan.process(&mut excitation, &mut workspace, ()).unwrap();
    assert_eq!(
        excitation,
        [
            -0.488_381_266_593_933_1,
            0.401_332_616_806_030_3,
            -0.135_468_184_947_967_53,
        ]
    );
}

#[test]
fn f32_and_f64_use_the_same_unit_samples_and_respect_each_amplitude() {
    let plan32 = DocrimeExcitationPlan::new([0.25_f32, 2.0], SEED).unwrap();
    let plan64 = DocrimeExcitationPlan::new([0.25_f64, 2.0], SEED).unwrap();
    let mut workspace32 = plan32.workspace();
    let mut workspace64 = plan64.workspace();
    let mut excitation32 = [0.0; 2];
    let mut excitation64 = [0.0; 2];

    for _ in 0..256 {
        plan32
            .process(&mut excitation32, &mut workspace32, ())
            .unwrap();
        plan64
            .process(&mut excitation64, &mut workspace64, ())
            .unwrap();
        assert_eq!(excitation32[0] as f64, excitation64[0]);
        assert_eq!(excitation32[1] as f64, excitation64[1]);
        assert!((-0.25..0.25).contains(&excitation32[0]));
        assert!((-2.0..2.0).contains(&excitation32[1]));
    }
}

#[test]
fn preparation_rejects_empty_or_invalid_amplitudes() {
    let error = DocrimeExcitationPlan::<f32>::new([], 0).unwrap_err();
    assert_eq!(error, DocrimeExcitationPlanError::Empty);
    assert!(error.to_string().contains("amplitudes must be nonempty"));
    for amplitude in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
        let error = DocrimeExcitationPlan::new([1.0, amplitude], 0).unwrap_err();
        assert_eq!(
            error,
            DocrimeExcitationPlanError::InvalidAmplitude {
                corrector_coordinate: 1
            }
        );
        assert!(
            error
                .to_string()
                .contains("corrector coordinate 1 must be finite and positive")
        );
    }
}

#[test]
fn rejected_output_or_workspace_extent_preserves_generator_history() {
    let plan = DocrimeExcitationPlan::new([1.0_f32, 1.0], SEED).unwrap();
    let other_plan = DocrimeExcitationPlan::new([1.0_f32], SEED).unwrap();
    let mut workspace = plan.workspace();
    let mut other_workspace = other_plan.workspace();
    let initial = workspace.clone();
    let mut short_output = [7.0];

    let error = plan
        .process(&mut short_output, &mut workspace, ())
        .unwrap_err();
    assert_eq!(
        error,
        DocrimeExcitationError::OutputExtent {
            expected: 2,
            actual: 1
        }
    );
    assert!(
        error
            .to_string()
            .contains("output extent 1 does not match 2")
    );
    assert_eq!(short_output, [7.0]);
    assert_eq!(workspace, initial);

    let mut output = [8.0, 9.0];
    let error = plan
        .process(&mut output, &mut other_workspace, ())
        .unwrap_err();
    assert_eq!(error, DocrimeExcitationError::WorkspaceExtent);
    assert!(error.to_string().contains("workspace extent"));
    assert_eq!(output, [8.0, 9.0]);
    assert_eq!(other_workspace, other_plan.workspace());
}

#[test]
fn binary_fixed_seed_sequence_has_exact_prepared_magnitudes() {
    let plan = DocrimeBinaryExcitationPlan::new([1.0_f64, 2.0, 0.5], SEED).unwrap();
    let mut workspace = plan.workspace();
    let mut excitation = [0.0; 3];

    plan.process(&mut excitation, &mut workspace, ()).unwrap();
    assert_eq!(excitation, [-1.0, 2.0, -0.5]);

    plan.process(&mut excitation, &mut workspace, ()).unwrap();
    assert_eq!(excitation, [-1.0, 2.0, -0.5]);

    plan.process(&mut excitation, &mut workspace, ()).unwrap();
    assert_eq!(excitation, [-1.0, -2.0, 0.5]);
}

#[test]
fn binary_f32_and_f64_sequences_are_identical() {
    let plan32 = DocrimeBinaryExcitationPlan::new([0.25_f32, 2.0], SEED).unwrap();
    let plan64 = DocrimeBinaryExcitationPlan::new([0.25_f64, 2.0], SEED).unwrap();
    let mut workspace32 = plan32.workspace();
    let mut workspace64 = plan64.workspace();
    let mut excitation32 = [0.0; 2];
    let mut excitation64 = [0.0; 2];

    for _ in 0..256 {
        plan32
            .process(&mut excitation32, &mut workspace32, ())
            .unwrap();
        plan64
            .process(&mut excitation64, &mut workspace64, ())
            .unwrap();
        assert_eq!(excitation32[0] as f64, excitation64[0]);
        assert_eq!(excitation32[1] as f64, excitation64[1]);
        assert_eq!(excitation32[0].abs(), 0.25);
        assert_eq!(excitation32[1].abs(), 2.0);
    }
}

#[test]
fn binary_plan_reuses_amplitude_validation() {
    assert_eq!(
        DocrimeBinaryExcitationPlan::<f32>::new([], 0),
        Err(DocrimeExcitationPlanError::Empty)
    );
    assert_eq!(
        DocrimeBinaryExcitationPlan::new([1.0_f32, f32::NAN], 0),
        Err(DocrimeExcitationPlanError::InvalidAmplitude {
            corrector_coordinate: 1,
        })
    );
}

#[test]
fn rejected_binary_output_preserves_generator_history() {
    let plan = DocrimeBinaryExcitationPlan::new([1.0_f32, 1.0], SEED).unwrap();
    let other_plan = DocrimeBinaryExcitationPlan::new([1.0_f32], SEED).unwrap();
    let mut workspace = plan.workspace();
    let mut other_workspace = other_plan.workspace();
    let initial = workspace.clone();
    let mut short_output = [7.0];

    assert_eq!(
        plan.process(&mut short_output, &mut workspace, ()),
        Err(DocrimeExcitationError::OutputExtent {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(short_output, [7.0]);
    assert_eq!(workspace, initial);

    let mut output = [8.0, 9.0];
    assert_eq!(
        plan.process(&mut output, &mut other_workspace, ()),
        Err(DocrimeExcitationError::WorkspaceExtent)
    );
    assert_eq!(output, [8.0, 9.0]);
    assert_eq!(other_workspace, other_plan.workspace());
}

#[test]
fn declaration_exposes_a_source_and_one_amplitude_parameter() {
    assert_eq!(DocrimeExcitationF32::NAME, "docrime-excitation-f32");
    assert_eq!(DocrimeExcitationF32::PROPERTIES, PropertyPolicy::None);
    assert_eq!(DocrimeExcitationF32::PORTS.len(), 2);
    assert_eq!(DocrimeExcitationF32::PORTS[0].name(), "excitation");
    assert_eq!(
        DocrimeExcitationF32::PORTS[0].direction(),
        PortDirection::Output
    );
    assert_eq!(DocrimeExcitationF32::PORTS[0].role(), PortRole::Data);
    assert_eq!(DocrimeExcitationF32::PORTS[1].name(), "amplitudes");
    assert_eq!(
        DocrimeExcitationF32::PORTS[1].direction(),
        PortDirection::Input
    );
    assert_eq!(DocrimeExcitationF32::PORTS[1].role(), PortRole::Parameter);

    let prepared = DocrimeExcitationF32::prepare(DocrimeExcitationF32Configuration {
        amplitudes: vec![1.0, 2.0, 0.5],
        seed: SEED,
    })
    .unwrap();
    assert_eq!(prepared.formats()[0].shape(), &[3]);
    assert_eq!(prepared.formats()[0].schema(), DOCRIME_EXCITATION_V1);
    assert_eq!(prepared.formats()[1].shape(), &[3]);
    assert_eq!(
        prepared.formats()[1].schema(),
        DOCRIME_EXCITATION_AMPLITUDE_V1
    );
}

#[test]
fn declared_source_preserves_sequence_across_parameter_adoption_and_reset() {
    let mut declared = DocrimeExcitationF32::prepare(DocrimeExcitationF32Configuration {
        amplitudes: vec![1.0, 2.0],
        seed: SEED,
    })
    .unwrap();
    let mut direct_plan = DocrimeExcitationPlan::new([1.0_f32, 2.0], SEED).unwrap();
    let mut direct_workspace = direct_plan.workspace();
    let mut expected = [0.0_f32; 2];
    let mut actual = [0.0_f32; 2];

    direct_plan
        .process(&mut expected, &mut direct_workspace, ())
        .unwrap();
    DocrimeExcitationF32::process(&mut declared, &mut actual).unwrap();
    assert_eq!(actual, expected);

    let amplitudes = [0.5_f32, 1.0];
    assert_eq!(
        declared.replace_parameter(1, AlgorithmInput::from(&amplitudes[..])),
        Some(Ok(()))
    );
    direct_plan = DocrimeExcitationPlan::new(amplitudes, SEED).unwrap();
    direct_plan
        .process(&mut expected, &mut direct_workspace, ())
        .unwrap();
    DocrimeExcitationF32::process(&mut declared, &mut actual).unwrap();
    assert_eq!(actual, expected);

    DocrimeExcitationF32::reset(&mut declared);
    direct_workspace = direct_plan.workspace();
    direct_plan
        .process(&mut expected, &mut direct_workspace, ())
        .unwrap();
    DocrimeExcitationF32::process(&mut declared, &mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn declared_binary_source_matches_direct_processing_and_parameter_adoption() {
    assert_eq!(
        DocrimeBinaryExcitationF32::NAME,
        "docrime-binary-excitation-f32"
    );
    assert_eq!(DocrimeBinaryExcitationF32::PORTS.len(), 2);
    assert_eq!(
        DocrimeBinaryExcitationF32::PORTS[0].direction(),
        PortDirection::Output
    );
    assert_eq!(
        DocrimeBinaryExcitationF32::PORTS[1].role(),
        PortRole::Parameter
    );

    let mut declared =
        DocrimeBinaryExcitationF32::prepare(DocrimeBinaryExcitationF32Configuration {
            amplitudes: vec![1.0, 2.0],
            seed: SEED,
        })
        .unwrap();
    assert_eq!(declared.formats()[0].schema(), DOCRIME_EXCITATION_V1);
    assert_eq!(
        declared.formats()[1].schema(),
        DOCRIME_EXCITATION_AMPLITUDE_V1
    );
    let mut direct_plan = DocrimeBinaryExcitationPlan::new([1.0_f32, 2.0], SEED).unwrap();
    let mut direct_workspace = direct_plan.workspace();
    let mut expected = [0.0_f32; 2];
    let mut actual = [0.0_f32; 2];

    direct_plan
        .process(&mut expected, &mut direct_workspace, ())
        .unwrap();
    DocrimeBinaryExcitationF32::process(&mut declared, &mut actual).unwrap();
    assert_eq!(actual, expected);

    let amplitudes = [0.25_f32, 0.5];
    assert_eq!(
        declared.replace_parameter(1, AlgorithmInput::from(&amplitudes[..])),
        Some(Ok(()))
    );
    direct_plan = DocrimeBinaryExcitationPlan::new(amplitudes, SEED).unwrap();
    direct_plan
        .process(&mut expected, &mut direct_workspace, ())
        .unwrap();
    DocrimeBinaryExcitationF32::process(&mut declared, &mut actual).unwrap();
    assert_eq!(actual, expected);

    DocrimeBinaryExcitationF32::reset(&mut declared);
    direct_workspace = direct_plan.workspace();
    direct_plan
        .process(&mut expected, &mut direct_workspace, ())
        .unwrap();
    DocrimeBinaryExcitationF32::process(&mut declared, &mut actual).unwrap();
    assert_eq!(actual, expected);
}
