#!/usr/bin/env python3
"""Build and install the production FilterGraph FGN shared library."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def installation_path(destdir: Path, prefix: Path, libdir: Path) -> Path:
    if not prefix.is_absolute():
        raise ValueError("--prefix must be absolute")
    if libdir.is_absolute() or ".." in libdir.parts:
        raise ValueError("--libdir must be a relative path without '..'")
    return destdir / Path(*prefix.parts[1:]) / libdir / "libfiltergraph-fgn.so"


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path("/usr"))
    parser.add_argument(
        "--libdir",
        type=Path,
        default=Path("lib/pipewire-ao/filter-graph"),
    )
    parser.add_argument(
        "--destdir",
        type=Path,
        default=Path(os.environ.get("DESTDIR", "/")),
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="install an existing release artifact",
    )
    options = parser.parse_args()

    if not options.no_build:
        command = [
            "cargo",
            "build",
            "--release",
            "--locked",
            "-p",
            "filtergraph-fgn-bundle",
        ]
        print(f"+ {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=repository, check=True)

    target = Path(os.environ.get("CARGO_TARGET_DIR", repository / "target"))
    artifact = target / "release/libfiltergraph_fgn_bundle.so"
    if not artifact.is_file():
        raise SystemExit(f"missing release artifact: {artifact}")

    try:
        destination = installation_path(
            options.destdir.resolve(), options.prefix, options.libdir
        )
    except ValueError as error:
        parser.error(str(error))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, destination)
    documentation = options.destdir.resolve() / Path(*options.prefix.parts[1:]) / "share/doc/filtergraph-rs"
    documentation.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repository / "LICENSE", documentation)
    shutil.copy2(repository / "docs/releases/THIRD_PARTY_NOTICES.md", documentation)
    shutil.copytree(
        repository / "docs/releases/third-party-licenses",
        documentation / "third-party-licenses",
        dirs_exist_ok=True,
    )
    print(destination)


if __name__ == "__main__":
    main()
