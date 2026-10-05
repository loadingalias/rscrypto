#!/usr/bin/env python3
"""Prove vector inventory and content changes cannot silently pass validation."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fixtures import verify

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


class FixtureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
        environment = patch.dict(os.environ, env, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.directory = self.root / "testdata/family"
        self.directory.mkdir(parents=True)
        self.payload = self.directory / "vector.bin"
        self.payload.write_bytes(b"abc")
        self.manifest = self.directory / "SHA256SUMS"
        self.manifest.write_text(f"{ABC_SHA256}  vector.bin\n")

    def test_nested_collections_and_ignored_scratch(self):
        nested = self.directory / "runtime"
        nested.mkdir()
        (nested / "vector.bin").write_bytes(b"abc")
        (nested / "SHA256SUMS").write_text(f"{ABC_SHA256} *vector.bin\n")
        (self.directory / "README.md").write_text("Vector provenance\n")
        (self.directory / "NIST-NOTICE.txt").write_text("Source notice\n")
        (self.root / ".gitignore").write_text("scratch/\n")
        scratch = self.directory / "scratch"
        scratch.mkdir()
        (scratch / "unmanifested").write_text("local scratch")
        self.assertEqual(verify(self.root), (2, 2))

    def test_corrupted_payload_is_rejected(self):
        self.payload.write_bytes(b"abd")
        with self.assertRaisesRegex(ValueError, "checksum mismatch.*vector.bin"):
            verify(self.root)

    def test_missing_tracked_payload_is_rejected(self):
        subprocess.run(["git", "add", "testdata"], cwd=self.root, check=True)
        self.payload.unlink()
        with self.assertRaisesRegex(ValueError, "missing vector input.*vector.bin"):
            verify(self.root)

    def test_unlisted_payloads_and_missing_manifests_are_rejected(self):
        for name in ("testdata/family/extra.bin", "testdata/new-family/vector.bin", "tests/vectors/vector.json"):
            with self.subTest(name=name):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"abc")
                with self.assertRaisesRegex(ValueError, "inventory mismatch.*unlisted"):
                    verify(self.root)
                path.unlink()
        self.manifest.unlink()
        with self.assertRaisesRegex(ValueError, "no vector checksum manifests"):
            verify(self.root)

    def test_missing_manifest_entry_target_is_rejected(self):
        self.manifest.write_text(f"{ABC_SHA256}  absent.bin\n")
        with self.assertRaisesRegex(ValueError, "inventory mismatch.*absent.bin"):
            verify(self.root)

    def test_duplicate_empty_and_invalid_manifests_are_rejected(self):
        valid = self.manifest.read_text()
        for content, message in ((valid * 2, "duplicate"), ("", "empty"), ("wrong  vector.bin\n", "invalid"),
                                 (f"{ABC_SHA256}  ../vector.bin\n", "invalid"),
                                 (f"{ABC_SHA256}  /vector.bin\n", "invalid"),
                                 (f"{ABC_SHA256}  runtime/vector.bin\n", "invalid"),
                                 (f"{ABC_SHA256}  ..\n", "invalid")):
            with self.subTest(content=content):
                self.manifest.write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    verify(self.root)

    def test_symlink_cannot_replace_a_payload(self):
        outside = self.root / "outside.bin"
        outside.write_bytes(b"abc")
        self.payload.unlink()
        try:
            self.payload.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "symlink"):
            verify(self.root)


if __name__ == "__main__":
    unittest.main()
