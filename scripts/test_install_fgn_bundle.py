"""Test installer notice selection using an ELF shared library."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path("usr/lib/pipewire-ao/filter-graph/libfiltergraph-fgn.so")
DOCUMENTATION = Path("usr/share/doc/filtergraph-rs")


class InstallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory(prefix="filtergraph-install-tests-")
        cls.addClassCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root / "fixture.rs"
        source.write_text('#[unsafe(no_mangle)] pub extern "C" fn fixture() {}\n')
        cls.fixture = root / "fixture.so"
        rustc = os.environ.get("RUSTC", "rustc")
        subprocess.run(
            [rustc, "--edition=2024", "--crate-type=cdylib", source, "-o", cls.fixture],
            cwd=ROOT,
            check=True,
        )
        version = subprocess.check_output([rustc, "--version"], cwd=ROOT, text=True)
        cls.release = version.split()[1]
        sysroot = subprocess.check_output(
            [rustc, "--print", "sysroot"], cwd=ROOT, text=True
        ).strip()
        cls.notices = Path(sysroot) / "share/doc/rust"

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="filtergraph-install-case-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.target = self.root / "target"
        self.destination = self.root / "stage"
        self.artifact = self.target / "release/libfiltergraph_fgn_bundle.so"
        self.artifact.parent.mkdir(parents=True)
        shutil.copyfile(self.fixture, self.artifact)

    def install(self):
        return subprocess.run(
            [
                sys.executable,
                ROOT / "scripts/install_fgn_bundle.py",
                "--no-build",
                "--destdir",
                self.destination,
            ],
            cwd=self.root,
            env={**os.environ, "CARGO_TARGET_DIR": str(self.target)},
            capture_output=True,
            text=True,
        )

    def test_installs_matching_toolchain_notices_with_the_bundle(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.destination / LIBRARY).read_bytes(), self.artifact.read_bytes())
        docs = self.destination / DOCUMENTATION
        self.assertEqual((docs / "LICENSE").read_bytes(), (ROOT / "LICENSE").read_bytes())
        dependencies = docs / "third-party-licenses"
        self.assertTrue((dependencies / "OpenBLAS-0.3.32/LICENSE").is_file())
        self.assertEqual(list(dependencies.glob("rust-*")), [dependencies / f"rust-{self.release}"])
        installed_rust = dependencies / f"rust-{self.release}"
        for path in (self.notices / "COPYRIGHT-library.html", *self.notices.glob("licenses/*")):
            self.assertEqual(
                (installed_rust / path.relative_to(self.notices)).read_bytes(), path.read_bytes()
            )

    def test_rejects_a_different_compiler_without_replacing_the_installed_library(self):
        stamp = self.root / "comment"
        stamp.write_bytes(b"rustc version 1.0.0 (000000000 2000-01-01)\0")
        subprocess.run(
            ["objcopy", "--update-section", f".comment={stamp}", self.artifact], check=True
        )
        installed = self.destination / LIBRARY
        installed.parent.mkdir(parents=True)
        installed.write_bytes(b"existing library")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Select the matching Rust toolchain", result.stderr)
        self.assertEqual(installed.read_bytes(), b"existing library")
        self.assertFalse((self.destination / DOCUMENTATION).exists())

    def test_rejects_a_library_without_compiler_evidence(self):
        subprocess.run(["objcopy", "--remove-section=.comment", self.artifact], check=True)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no Rust compiler stamp", result.stderr)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
