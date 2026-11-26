"""Exercise the production build stamp in real temporary Git checkouts."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SoftwareVersionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="filtergraph-build-stamp-")
        cls.build = Path(cls.directory.name) / "build-script"
        source = Path(__file__).resolve().parents[1] / "crates/filtergraph-fgn-bundle/build.rs"
        subprocess.run(["rustc", "--edition=2024", str(source), "-o", str(cls.build)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def stamp(self, root):
        manifest = root / "crates/filtergraph-fgn-bundle"
        manifest.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, CARGO_MANIFEST_DIR=str(manifest), CARGO_CFG_TARGET_OS="none")
        output = subprocess.check_output([str(self.build)], cwd=root, env=environment, text=True)
        fields = dict(line.removeprefix("cargo:rustc-env=").split("=", 1)
                      for line in output.splitlines() if line.startswith("cargo:rustc-env="))
        return {key: json.loads(value) for key, value in fields.items()}

    def git(self, root, *args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()

    def test_clean_dirty_and_changed_commit(self):
        with tempfile.TemporaryDirectory(prefix="filtergraph-version-git-") as directory:
            root = Path(directory)
            source = root / "crates/filtergraph-fgn-bundle/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text("// tracked build input\n")
            self.git(root, "init", "-q")
            self.git(root, "add", ".")
            self.git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial")
            clean = self.stamp(root)
            self.assertEqual(clean["FILTERGRAPH_GIT_COMMIT"], self.git(root, "rev-parse", "HEAD"))
            self.assertIs(clean["FILTERGRAPH_GIT_DIRTY"], False)
            source.write_text("// changed build input\n")
            self.assertIs(self.stamp(root)["FILTERGRAPH_GIT_DIRTY"], True)
            self.git(root, "add", ".")
            self.git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "changed")
            updated = self.stamp(root)
            self.assertNotEqual(clean["FILTERGRAPH_GIT_COMMIT"], updated["FILTERGRAPH_GIT_COMMIT"])
            self.assertIs(updated["FILTERGRAPH_GIT_DIRTY"], False)

    def test_missing_git_and_unrelated_parent_git_are_unknown(self):
        with tempfile.TemporaryDirectory(prefix="filtergraph-version-archive-") as directory:
            root = Path(directory)
            self.assertEqual(set(self.stamp(root).values()), {None})
            self.git(root, "init", "-q")
            (root / "README").write_text("unrelated parent repository\n")
            self.git(root, "add", "README")
            self.git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "unrelated")
            archive = root / "archive"
            archive.mkdir()
            self.assertEqual(set(self.stamp(archive).values()), {None})
