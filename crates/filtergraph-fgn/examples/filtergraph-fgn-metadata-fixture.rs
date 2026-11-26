//! Explicit deployment layout used to prove native-host metadata propagation.
use core::ffi::{CStr, c_char};
use filtergraph_algorithms::NdarrayCopyF32;
use filtergraph_fgn::{AlgorithmDescriptor, AlgorithmRegistry, abi};
use std::sync::OnceLock;

static REGISTRY: OnceLock<Result<AlgorithmRegistry, i32>> = OnceLock::new();

fn registry() -> Option<&'static AlgorithmRegistry> {
    REGISTRY
        .get_or_init(|| {
            let layouts = [
                (
                    abi::META_ACQUISITION_IDENTITY,
                    40,
                    abi::META_REQUIREMENT_REQUIRED,
                ),
                (abi::META_EXPOSURE_TIMING, 24, 0),
                (abi::META_REFERENCE_TIMESTAMP, 24, 0),
                (abi::META_OPTICAL_CHOPPING, 8, 0),
                (abi::META_ELECTRICAL_TRIGGER, 8, 0),
                (0x777, 8, 0),
            ]
            .map(|(type_, size, flags)| abi::SpaMetaRequirement {
                type_,
                size,
                flags,
                reserved: 0,
            });
            // The output obtains its layouts from the declared triggering input.
            let descriptor =
                AlgorithmDescriptor::new_with_metadata::<NdarrayCopyF32>(&[&layouts, &[]])?;
            AlgorithmRegistry::new(
                "filtergraph-metadata-proof",
                vec![descriptor],
                find_descriptor,
            )
        })
        .as_ref()
        .ok()
}

unsafe extern "C" fn find_descriptor(name: *const c_char) -> *const abi::Descriptor {
    std::panic::catch_unwind(|| {
        if name.is_null() {
            return core::ptr::null();
        }
        registry().map_or(core::ptr::null(), |r| {
            r.find(unsafe { CStr::from_ptr(name) })
        })
    })
    .unwrap_or(core::ptr::null())
}

/// Return the current native FGN metadata proof interface.
#[unsafe(no_mangle)]
pub extern "C" fn spa_filter_graph_ndarray_plugin_get_interface(
    version: u32,
) -> *const abi::Plugin {
    std::panic::catch_unwind(|| {
        if version != abi::ABI_VERSION {
            return core::ptr::null();
        }
        registry().map_or(core::ptr::null(), |r| core::ptr::from_ref(r.plugin()))
    })
    .unwrap_or(core::ptr::null())
}
