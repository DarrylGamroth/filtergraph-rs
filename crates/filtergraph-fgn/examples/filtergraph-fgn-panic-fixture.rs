//! Release-profile ABI fixture that deliberately panics inside `process`.

use core::ffi::CStr;

use filtergraph_fgn::{InputBuffers, NoProperties, OutputBuffers, PluginInstance, abi};

struct PanicFixture;

// SAFETY: the fixture has no state. Its deliberate process panic is contained
// by the shared callback adapter.
unsafe impl PluginInstance for PanicFixture {
    type Properties = NoProperties;

    const NAME: &'static CStr = c"panic-fixture";
    const PORTS: &'static [abi::PortInfo] = &[];

    fn instantiate(_config: &CStr) -> Result<Self, i32> {
        Ok(Self)
    }

    fn port_format(&self, _port: u32) -> Option<&abi::Format> {
        None
    }

    fn properties(&self) -> &Self::Properties {
        &NoProperties
    }

    fn process(
        &self,
        _inputs: InputBuffers<'_>,
        _outputs: &mut OutputBuffers<'_>,
    ) -> Result<(), i32> {
        panic!("release-profile panic fixture")
    }
}

filtergraph_fgn::export_plugin!(c"filtergraph-panic-fixture", [PanicFixture]);
