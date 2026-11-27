# FGN shared-library deployment

The production bundle is a shared library of scientific declarations for the
published PipeWireAO 4.0.1 or later
`libpipewire-module-ndarray-filter-chain` host (FGN ABI 8).
Rust supplies prepared numerical
operations and the FGN adapter. PipeWire supplies the graph executor. No Rust
graph compiler, scheduler, result provenance, or configuration provenance layer
is required to load this library.

PipeWireAO_jll 4.0.1+0 supplies the released native host, modules and tools.
The prepared metadata requirements extension is included in that release; no
manual host patch or local source build is required. Native 3.x also reports
ABI 8 and is incompatible. PipeWireAO 4.0.0 lacks the extension required by
configuration-dependent scientific metadata requirements.

Build with `cargo build --release --locked -p filtergraph-fgn-bundle`.
Use the absolute resulting library path in each filter-chain node:

```text
filter.graph = {
    nodes = [
        {
            type = ndarray
            name = integrate
            plugin = "/absolute/path/libfiltergraph_fgn_bundle.so"
            label = leaky-integrator-f32
            config = {
                extent = 1
                initial_state = 0.0
                input_schema = pipewireao/controller-residual-error-f32/1
                output_schema = pipewireao/requested-pdm-command-f32/1
                rate = [ 1000 1 ]
            }
            props = { gain = 1.0 pole = 0.0 }
        }
    ]
    inputs = [ "integrate:input" ]
    outputs = [ "integrate:output" ]
}
```

This is a node within the host's filter-chain configuration. `type = ndarray`
selects the host's ndarray plugin loader; `plugin` names the shared library;
`label` selects one declaration. Other host loaders use their own types for
LADSPA, LV2, SOFA, FFmpeg, and ONNX. FGN is the Rust library's existing boundary.
The complete [SHWFS example](../examples/shwfs-filter-chain.conf) shows links
and several nodes. Installation can instead stage
`lib/pipewire-ao/filter-graph/libfiltergraph-fgn.so` and render that absolute
path into `@FILTERGRAPH_FGN_PLUGIN@` templates.

Use `python3 scripts/install_fgn_bundle.py --prefix /usr --destdir /path/to/stage`
to build and stage the library with its MIT license and dependency notices.
The installer uses `readelf` to check the library's ELF compiler stamp and
copies the matching Rust standard-library notices from the selected toolchain.
For an existing release build, use `--no-build` with the same Rust toolchain
and `CARGO_TARGET_DIR`. Keep the ELF `.comment` section until staging is
complete.

The host discovers declarations, creates/configures instances on its control
path, prepares and connects buffers, activates processing, schedules calls,
and owns routing and publication. It must quiesce processing before reset,
deactivation, instance destruction, or buffer replacement as required by FGN.
The library requests `PLUGIN_RETAIN_LIBRARY`: retain it for the host process
lifetime. Prepared Plans retain numerical artifacts; Workspaces retain mutable
history and scratch. Property and parameter publication retains its private
revision, token, and retirement checks; those checks protect safe adoption and
are not software identity or scientific result provenance. Metadata, reset,
commit, failure, and numerical behavior continue to follow the existing
[declaration contract](algorithm-declaration-contract.md) and
[property contract](scalar-property-contract.md).

## Software identity

`const char *filtergraph_software_version(void)` returns immutable,
NUL-terminated UTF-8 JSON owned by the loaded library. Do not free or modify it.
It is an optional control-path query, independent of plugin instances and
processing. Example:

```json
{"version":"0.2.0","git_commit":"<40-digit Git commit>","git_dirty":true}
```

`version` is Cargo's package version. `git_commit` and `git_dirty` are captured
by the bundle's Cargo build script from the source checkout at compilation;
`git_dirty` is `true` when `git status --porcelain --untracked-files=normal`
reports changes, otherwise `false`. Ignored build output does not count.
Unavailable Git information is JSON `null`. The source file must be tracked
by the detected repository; a source archive nested in an unrelated Git checkout
does not adopt that checkout’s identity. Querying a copied artifact, changing
the caller's directory, or editing a checkout after compilation cannot change
the returned string. This identifies the built software, not a result, graph
configuration, calibration, or later checkout. It does not claim a content hash
of dirty sources or a reproducible binary. Cargo watches known source files and
Git HEAD/index/ref changes; force a rebuild after adding an otherwise unwatched
untracked file when that file must affect the dirty marker.

For example, Python can query the built library without starting PipeWire:

```python
import ctypes, json
library = ctypes.CDLL("/absolute/path/libfiltergraph_fgn_bundle.so")
library.filtergraph_software_version.restype = ctypes.c_char_p
print(json.loads(library.filtergraph_software_version()))
```

The identity names are external software metadata: Cargo owns `version`, Git
owns `git_commit` and `git_dirty`, and this document owns the single optional
`filtergraph_software_version` C query. Scientific schema names are unchanged.
The native declaration test loads the actual shared library, checks identity
against the build checkout, verifies the exact dynamic exports, and then runs
all declarations through the real C graph host. The C test uses synthetic inputs.

## Verify against the native host

Use a configured build of the public PipeWireAO 4.x sources, version 4.0.1
or later. Run the declaration, metadata, shared-library identity, and
installation checks from the repository root:

```sh
python3 scripts/test_fgn_declaration.py \
    --fgn-source /absolute/path/to/PipeWireAO \
    --fgn-build /absolute/path/to/PipeWireAO/build
python3 scripts/test_fgn_release_panic.py
```

The tests exercise the public host's ABI, graph preparation, and numerical
callbacks with synthetic data. Device delivery and real-time timing are
qualified by each downstream deployment.
