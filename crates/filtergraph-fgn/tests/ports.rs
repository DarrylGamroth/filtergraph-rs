//! Typed port-declaration layout checks.

use std::ffi::CStr;

use filtergraph_fgn::{Bool8, abi};

filtergraph_fgn::declare_ndarray_ports! {
    /// Fixture with every supported port role.
    FixturePorts {
        inputs {
            IMAGE: f32 = "image";
            ENABLE: Bool8 = "enable";
        }
        outputs {
            RESULT: f64 = "result";
        }
        parameters {
            MATRIX: f32 = "matrix";
            MASK: Bool8 = "mask";
        }
    }
}

#[test]
fn typed_declaration_generates_stable_descriptor_and_process_indices() {
    assert_eq!(FixturePorts::PORTS.len(), 5);
    let expected = [
        (0, abi::DIRECTION_INPUT, 0, c"image"),
        (1, abi::DIRECTION_INPUT, 0, c"enable"),
        (2, abi::DIRECTION_OUTPUT, 0, c"result"),
        (
            3,
            abi::DIRECTION_INPUT,
            abi::PORT_OPTIONAL | abi::PORT_PARAMETER,
            c"matrix",
        ),
        (
            4,
            abi::DIRECTION_INPUT,
            abi::PORT_OPTIONAL | abi::PORT_PARAMETER,
            c"mask",
        ),
    ];
    for (port, (index, direction, flags, name)) in FixturePorts::PORTS.iter().zip(expected) {
        assert_eq!(port.index, index);
        assert_eq!(port.direction, direction);
        assert_eq!(port.flags, flags);
        // SAFETY: generated port names have static NUL-terminated storage.
        assert_eq!(unsafe { CStr::from_ptr(port.name) }, name);
    }

    assert_eq!(FixturePorts::IMAGE.index(), 0);
    assert_eq!(FixturePorts::ENABLE.index(), 1);
    assert_eq!(FixturePorts::RESULT.index(), 0);
    assert_eq!(FixturePorts::MATRIX.port(), 3);
    assert_eq!(FixturePorts::MATRIX.input_index(), 2);
    assert_eq!(FixturePorts::MASK.port(), 4);
    assert_eq!(FixturePorts::MASK.input_index(), 3);
}
