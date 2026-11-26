//! Native bindings reject malformed buffers before any scientific state advances.
use filtergraph_algorithms::{
    ClosedLoopCorrectionF32 as C, ClosedLoopCorrectionF32Configuration,
    ClosedLoopIntegratorF32 as I, ClosedLoopIntegratorF32Configuration,
};
#[test]
fn auxiliary_state_extent_rejection_precedes_output_and_history_publication() {
    macro_rules! check {
        ($d:ident,$prepared:expr) => {{
            let mut p = $prepared;
            let mut correction = [71.0; 2];
            let mut state = [];
            assert!(
                $d::process(&mut p, &[1.0, 2.0], &[0.0; 2], &mut correction, &mut state).is_err()
            );
            assert_eq!(correction, [71.0; 2]);
            assert_eq!(p.workspace().last_output(), None);
            let mut state = [-17.0; 2];
            $d::process(
                &mut p,
                &[1.0, 2.0],
                &[f32::NAN; 2],
                &mut correction,
                &mut state,
            )
            .unwrap();
            assert_eq!(correction, [1.0, 2.0]);
            assert_eq!(state, [0.0; 2]);
        }};
    }
    check!(
        I,
        I::prepare(ClosedLoopIntegratorF32Configuration { state_length: 2 }).unwrap()
    );
    check!(
        C,
        C::prepare(ClosedLoopCorrectionF32Configuration {
            state_length: 2,
            hidden_mode_count: 1
        })
        .unwrap()
    );
}

#[test]
fn metadata_calls_reject_extents_before_history_adoption() {
    use filtergraph_algorithms::{DeclarationProcessError, NativeProcessError, PortError};
    let mut p = I::prepare(ClosedLoopIntegratorF32Configuration { state_length: 2 }).unwrap();
    let mut correction = [71.0; 2];
    let mut state = [-17.0; 1];
    let outcome = I::process_with_metadata(
        &mut p,
        &[None, None],
        &[1.0, 2.0],
        &[0.0; 2],
        &mut correction,
        &mut state,
    );
    assert!(matches!(
        outcome,
        Err(NativeProcessError::Process(DeclarationProcessError::Port(
            PortError::Extent {
                expected: 2,
                actual: 1
            }
        )))
    ));
    assert_eq!(correction, [71.0; 2]);
    assert_eq!(state, [-17.0; 1]);
    assert_eq!(p.workspace().last_output(), None);
}
