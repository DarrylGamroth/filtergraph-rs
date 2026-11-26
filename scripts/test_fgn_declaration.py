#!/usr/bin/env python3
"""Build and run the generated declaration through the FGN C graph host."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

def fgn_host_abi(fgn_root: Path) -> int:
    # ABI 8 alone does not distinguish the retired native 3.x metadata layout.
    version = re.search(r"version\s*:\s*['\"](\d+)\.", (fgn_root / "meson.build").read_text())
    if version is None or int(version.group(1)) != 4:
        raise ValueError("current buffer metadata requires a native 4.x host")
    host_header = fgn_root / "spa/include/spa/filter-graph/ndarray-plugin.h"
    match = re.search(
        r"#define SPA_FGN_PLUGIN_ABI_VERSION (\d+)u", host_header.read_text()
    )
    if match is None:
        raise ValueError(f"cannot determine FGN host ABI from {host_header}")
    host_abi = int(match.group(1))
    if host_abi != 8:
        raise ValueError(f"unsupported FGN host ABI {host_abi}")
    if "spa_fgn_descriptor_has_get_port_metadata" not in host_header.read_text():
        raise ValueError("current catalog requires the prepared port metadata query extension")
    return host_abi


def run(command: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=env)


def verify_production_selection(repository: Path) -> None:
    declarations: dict[str, str] = {}
    algorithm_source = repository / "crates/filtergraph-algorithms/src"
    declaration_start = re.compile(r"crate::filtergraph_algorithm!\s*\{")
    declaration_name = re.compile(r"\bpub\s+([A-Za-z][A-Za-z0-9_]*)\s*\{")
    declaration_label = re.compile(r'\blabel\s*=\s*"([a-z0-9-]+)"')
    for path in sorted(algorithm_source.rglob("*.rs")):
        source = path.read_text()
        for start in declaration_start.finditer(source):
            end = start.end() + 512
            name = declaration_name.search(source, start.end(), end)
            label = declaration_label.search(source, start.end(), end)
            if name is None or label is None:
                raise RuntimeError(f"cannot identify portable declaration in {path}")
            declarations[name.group(1)] = label.group(1)

    bundle_source = (
        repository / "crates/filtergraph-fgn-bundle/src/lib.rs"
    ).read_text()
    selection_start = bundle_source.index(
        "filtergraph_fgn::export_algorithm_declarations!"
    )
    selection = bundle_source[selection_start:].split("[", 1)[1].split("]", 1)[0]
    selected = re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", selection)
    if len(selected) != len(set(selected)):
        raise RuntimeError("production bundle selects a declaration more than once")
    missing = sorted(set(declarations) - set(selected))
    unknown = sorted(set(selected) - set(declarations))
    if missing or unknown:
        raise RuntimeError(
            f"production declaration selection mismatch: missing={missing}, unknown={unknown}"
        )

    graph_path = repository / "crates/filtergraph-fgn/tests/algorithm_declaration_graph.c"
    graph_test = graph_path.read_text()
    for include in re.findall(r'^#include "([^"]+)"', graph_test, re.MULTILINE):
        graph_test += "\n" + (graph_path.parent / include).read_text()
    # C joins adjacent string literals, including labels split across lines.
    graph_test = re.sub(r'"\s*"', "", graph_test)
    host_labels = set(re.findall(r"\blabel = ([a-z0-9-]+)", graph_test))
    untested = sorted(set(declarations.values()) - host_labels)
    if untested:
        raise RuntimeError(f"production declarations without C-host cases: {untested}")
    print(
        f"production bundle selects all {len(selected)} portable declarations "
        "and the C graph names every selected label"
    )


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fgn-source",
        type=Path,
        required=True,
    )
    parser.add_argument("--fgn-build", type=Path)
    arguments = parser.parse_args()

    verify_production_selection(repository)

    fgn_source = arguments.fgn_source.resolve()
    fgn_build = (
        arguments.fgn_build.resolve()
        if arguments.fgn_build is not None
        else fgn_source / "build"
    )
    fgn_host_abi(fgn_source)
    build_environment = os.environ.copy()
    bundle_command = [
        "cargo", "build", "--release", "--locked", "-p", "filtergraph-fgn-bundle"
    ]
    run(bundle_command, repository, env=build_environment)
    run(
        [
            "cargo",
            "rustc",
            "--release",
            "--locked",
            "-p",
            "filtergraph-fgn",
            "--features",
            "blas",
            "--example",
            "filtergraph-fgn-metadata-fixture",
            "--",
            "-C",
            "link-arg=-Wl,--exclude-libs,ALL",
        ],
        repository,
        env=build_environment,
    )
    target = Path(os.environ.get("CARGO_TARGET_DIR", repository / "target")).resolve()
    library = target / "release/libfiltergraph_fgn_bundle.so"
    metadata_library = (
        target / "release/examples/libfiltergraph_fgn_metadata_fixture.so"
    )
    for artifact in (library, metadata_library):
        dynamic_symbols = subprocess.run(
            ["nm", "-D", "--defined-only", str(artifact)],
            cwd=repository,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
        exported = {line.rsplit(maxsplit=1)[-1] for line in dynamic_symbols}
        expected = {"spa_filter_graph_ndarray_plugin_get_interface"}
        if artifact == library:
            expected.add("filtergraph_software_version")
        if exported != expected:
            raise RuntimeError(
                f"unexpected exports from {artifact.name}: {sorted(exported)}"
            )

    # Query the actual native artifact, rather than Rust's statically linked test copy.
    loaded = ctypes.CDLL(str(library))
    loaded.filtergraph_software_version.restype = ctypes.c_char_p
    identity = json.loads(loaded.filtergraph_software_version())
    version = tomllib.loads((repository / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    assert identity["version"] == version, identity
    tracked_source = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "crates/filtergraph-fgn-bundle/src/lib.rs"],
        cwd=repository, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    commit = (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
              if tracked_source else None)
    dirty = (bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repository))
             if tracked_source else None)
    assert identity["git_commit"] == commit, identity
    assert identity["git_dirty"] is dirty, identity
    # Loading and repeated queries do not consult the caller's current directory.
    with tempfile.TemporaryDirectory(prefix="filtergraph-identity-") as directory:
        previous = Path.cwd()
        try:
            os.chdir(directory)
            assert json.loads(loaded.filtergraph_software_version()) == identity
        finally:
            os.chdir(previous)

    with tempfile.TemporaryDirectory(prefix="filtergraph-fgn-install-") as directory:
        run(
            [
                "python3",
                "scripts/install_fgn_bundle.py",
                "--no-build",
                "--prefix",
                "/usr",
                "--destdir",
                directory,
            ],
            repository,
        )
        installed = (
            Path(directory)
            / "usr/lib/pipewire-ao/filter-graph/libfiltergraph-fgn.so"
        )
        if installed.read_bytes() != library.read_bytes():
            raise RuntimeError("installed FilterGraph FGN bundle differs from build artifact")

    run(
        [
            "ninja",
            "-C",
            str(fgn_build),
            "spa/plugins/filter-graph/libspa-filter-graph-ndarray.a",
        ],
        fgn_source,
    )
    source = (
        repository
        / "crates/filtergraph-fgn/tests/algorithm_declaration_graph.c"
    )
    with tempfile.TemporaryDirectory(prefix="filtergraph-fgn-declaration-") as directory:
        executable = Path(directory) / "algorithm-declaration-graph"
        run(
            [
                "cc",
                "-std=gnu11",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-I{fgn_build}",
                f"-I{fgn_source}",
                f"-I{fgn_source / 'spa/include'}",
                f"-I{fgn_source / 'spa/include-private'}",
                str(source),
                str(
                    fgn_build
                    / "spa/plugins/filter-graph/libspa-filter-graph-ndarray.a"
                ),
                "-ldl",
                "-lm",
                "-pthread",
                "-o",
                str(executable),
            ],
            repository,
        )
        run([str(executable), str(library)], repository)

        metadata_executable = Path(directory) / "metadata-graph"
        run(
            [
                "cc",
                "-std=gnu11",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-I{fgn_build}",
                f"-I{fgn_source}",
                f"-I{fgn_source / 'spa/include'}",
                f"-I{fgn_source / 'spa/include-private'}",
                str(repository / "crates/filtergraph-fgn/tests/metadata_graph.c"),
                str(
                    fgn_build
                    / "spa/plugins/filter-graph/libspa-filter-graph-ndarray.a"
                ),
                "-ldl",
                "-lm",
                "-pthread",
                "-o",
                str(metadata_executable),
            ],
            repository,
        )
        run([str(metadata_executable), str(metadata_library)], repository)


if __name__ == "__main__":
    main()
