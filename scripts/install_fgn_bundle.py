#!/usr/bin/env python3
"""Build and install the production FilterGraph FGN shared library."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def installation_path(destdir: Path, prefix: Path, libdir: Path) -> Path:
    if not prefix.is_absolute():
        raise ValueError("--prefix must be absolute")
    if libdir.is_absolute() or ".." in libdir.parts:
        raise ValueError("--libdir must be a relative path without '..'")
    return destdir / Path(*prefix.parts[1:]) / libdir / "libfiltergraph-fgn.so"


def rust_notices(repository: Path, artifact: Path) -> tuple[str, Path]:
    rustc = os.environ.get("RUSTC", "rustc")
    version = subprocess.check_output([rustc, "--version"], cwd=repository, text=True).strip()
    _, release, *_ = version.split()
    expected = version.replace("rustc ", "rustc version ", 1)

    # Read the compiler stamp without loading the library or executing target code.
    comments = subprocess.run(
        ["readelf", "--string-dump=.comment", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    strings = (line.partition("]")[2].strip() for line in comments.splitlines())
    stamps = {value for value in strings if value.startswith("rustc version ")}
    if not stamps:
        raise ValueError(f"{artifact} has no Rust compiler stamp in its ELF .comment section")
    if stamps != {expected}:
        raise ValueError(
            f"{artifact} reports {', '.join(sorted(stamps))}, selected compiler is {version}. "
            "Select the matching Rust toolchain or rebuild the bundle."
        )

    sysroot = subprocess.check_output(
        [rustc, "--print", "sysroot"], cwd=repository, text=True
    ).strip()
    notices = Path(sysroot) / "share/doc/rust"
    copyright = notices / "COPYRIGHT-library.html"
    licenses = notices / "licenses"
    if not copyright.is_file() or not licenses.is_dir() or not any(licenses.iterdir()):
        raise ValueError(f"missing Rust standard-library notices in {notices}")
    return release, notices


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
        help="install an existing release artifact using its matching Rust toolchain",
    )
    options = parser.parse_args()

    try:
        destination = installation_path(
            options.destdir.resolve(), options.prefix, options.libdir
        )
    except ValueError as error:
        parser.error(str(error))

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

    target = repository / os.environ.get("CARGO_TARGET_DIR", "target")
    artifact = target / "release/libfiltergraph_fgn_bundle.so"
    if not artifact.is_file():
        raise SystemExit(f"missing release artifact: {artifact}")

    try:
        rust_release, notices = rust_notices(repository, artifact)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    documentation = (
        options.destdir.resolve()
        / Path(*options.prefix.parts[1:])
        / "share/doc/filtergraph-rs"
    )
    documentation.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repository / "LICENSE", documentation)
    shutil.copy2(repository / "docs/releases/THIRD_PARTY_NOTICES.md", documentation)
    licenses = documentation / "third-party-licenses"
    shutil.copytree(
        repository / "docs/releases/third-party-licenses",
        licenses,
        dirs_exist_ok=True,
    )
    rust_licenses = licenses / f"rust-{rust_release}"
    shutil.copytree(
        notices / "licenses",
        rust_licenses / "licenses",
        dirs_exist_ok=True,
    )
    shutil.copy2(notices / "COPYRIGHT-library.html", rust_licenses)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        staged_library = Path(staging) / destination.name
        shutil.copy2(artifact, staged_library)
        staged_library.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
