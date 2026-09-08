#!/usr/bin/env python3
"""Verify CT preparation preserves measurements and rejects unmatched selectors."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import binsec

ROOT = Path(__file__).resolve().parents[2]
TARGET = "x86_64-unknown-linux-gnu"


def executable(path, source):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(source)
  path.chmod(0o755)


def snapshot(root):
  return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


class PreparationTests(unittest.TestCase):
  def test_artifacts_preserve_evidence_on_success_and_failure(self):
    for failure in ("", "--lib", "--bin"):
      with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        commands = root / "commands"
        executable(commands / "git", f"#!/bin/sh\nprintf '%s\\n' '{root}'\n")
        sysroot = root / "sysroot"
        executable(commands / "rustc", f"#!/bin/sh\n" +
                   f"if [ \"$*\" = '--print sysroot' ]; then echo '{sysroot}'; else echo 'host: {TARGET}'; fi\n")
        llvm_bin = sysroot / "lib/rustlib" / TARGET / "bin"
        for tool in ("llvm-objdump", "llvm-nm", "llvm-size"):
          executable(llvm_bin / tool, f"#!/bin/sh\necho 'sysroot {tool}'\n")
        objdump = commands / "custom tools/objdump"
        nm = commands / "custom tools/nm.exe"
        executable(objdump, "#!/bin/sh\necho 'override objdump'\n")
        executable(nm, "#!/bin/sh\necho 'override nm'\n")
        executable(commands / "cc", "#!/bin/sh\necho 'fixture linker'\n")
        executable(root / "scripts/lib/python.sh", f"#!/bin/sh\necho '{commands / 'reporter'}'\n")
        executable(commands / "reporter", f"#!{sys.executable}\n" + '''
import os, sys
if sys.argv[1] == 'scripts/ct/provenance.py':
  os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
        (root / "scripts/ct").mkdir()
        shutil.copy2(ROOT / "scripts/ct/provenance.py", root / "scripts/ct/provenance.py")
        for path in ("Cargo.toml", "tools/ct-harness/Cargo.toml"):
          manifest = root / path
          manifest.parent.mkdir(parents=True, exist_ok=True)
          manifest.write_text('[package]\nname = "fixture"\nversion = "0.0.0"\n')
        for path in ("ct.toml", "Cargo.lock", "tools/ct-harness/Cargo.lock",
                     "tools/ct-dudect/Cargo.lock", "tools/ct-binsec-harness/Cargo.lock"):
          lockfile = root / path
          lockfile.parent.mkdir(parents=True, exist_ok=True)
          lockfile.touch()
        executable(commands / "cargo", f"#!{sys.executable}\n" + '''
import os, sys
from pathlib import Path
args = sys.argv[1:]
if args == ['-V']:
  print('cargo fixture')
  sys.exit(0)
if os.environ['FAIL_BUILD'] and os.environ['FAIL_BUILD'] in args:
  sys.exit(17)
build = Path(args[args.index('--target-dir') + 1])
assert not (build / 'stale-build').exists()
emit = build / args[args.index('--target') + 1] / 'release'
emit.mkdir(parents=True, exist_ok=True)
name = 'rscrypto_ct_harness' if '--lib' in args else 'rscrypto_ct_evidence'
for extension in ('ll', 's', 'o'):
  (emit / (name + '.' + extension)).write_text('fresh')
if '--bin' in args:
  (emit / 'rscrypto-ct-evidence').write_text('binary')
  flag = next(arg for arg in args if arg.startswith('link-arg=-Wl,--Map='))
  Path(flag.split('=', 2)[2]).write_text('map')
  print('linker "-o" binary')
''')
        output = root / "target/ct" / TARGET / "release"
        retained = output / "dudect/runs/historical"
        retained.mkdir(parents=True)
        (retained / "dudect-raw.csv").write_bytes(b"original measurement\n")
        (output / "binsec").mkdir()
        (output / "binsec/report.json").write_text('{"status":"secure"}')
        before = snapshot(output)
        current_reports = ("provenance.json", "artifact-hashes.txt", "evidence-index.json",
                           "asm-heuristics.json", "asm-heuristics.md")
        for name in current_reports:
          (output / name).write_text("stale report")
        artifacts = output / "artifacts"
        artifacts.mkdir()
        (artifacts / "stale-artifact").write_text("stale")
        build = root / "target/ct-build" / TARGET / "release"
        build.mkdir(parents=True)
        (build / "stale-build").write_text("stale")
        env = {key: value for key, value in os.environ.items()
               if key not in {"BASH_ENV", "ENV"} and not key.startswith("BASH_FUNC_")}
        env.update(PATH=str(commands) + os.pathsep + os.environ["PATH"], FAIL_BUILD=failure)
        env.update(LLVM_OBJDUMP=str(objdump), LLVM_NM=str(nm.with_suffix("")))
        env.pop("LLVM_SIZE", None)
        result = subprocess.run(
          [shutil.which("bash"), str(ROOT / "scripts/ct/artifacts.sh"), "--target", TARGET],
          cwd=root, env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 17 if failure else 0, result.stdout + result.stderr)
        self.assertFalse((artifacts / "stale-artifact").exists())
        self.assertFalse((build / "stale-build").exists())
        for path, contents in before.items():
          self.assertEqual((output / path).read_bytes(), contents)
        for name in current_reports:
          path = output / name
          if failure:
            self.assertFalse(path.exists(), name)
          elif path.exists():
            self.assertNotEqual(path.read_text(), "stale report", name)
        if not failure:
          self.assertEqual((artifacts / "rscrypto-ct-evidence").read_text(), "binary")
          tools = json.loads((output / "provenance.json").read_text())["tools"]
          for tool, suffix, expected in (
            ("llvm_objdump", "disasm", "override objdump"),
            ("llvm_nm", "raw-symbols", "override nm"),
            ("llvm_size", "size", "sysroot llvm-size"),
          ):
            self.assertEqual(tools[tool], expected)
            self.assertEqual((artifacts / f"rscrypto_ct_harness.o.{suffix}.txt").read_text().strip(), expected)

  def test_binsec_selection_preserves_unrelated_results(self):
    manifest = {
      "target": [{"name": TARGET, "claim": "ct-claimed", "binsec": "required"},
                 {"name": "unsupported", "binsec": "unsupported"}],
      "binsec_kernel": [{"id": "known", "symbol": "known_symbol", "targets": [TARGET]}],
    }
    for selector, target, expected in (
      ("typo", TARGET, 2), ("known", "unsupported", 2),
      ("typo", "unsupported", 2), ("", TARGET, 2),
      ("known", TARGET, 0), ("known_symbol", TARGET, 0),
      (None, TARGET, 1), (None, "unsupported", 0),
    ):
      with self.subTest(selector=selector, target=target), tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "target/ct" / target / "release/binsec"
        output.mkdir(parents=True)
        (output / "unrelated.json").write_text("retained")
        before = snapshot(output)
        ct = {**manifest, "binsec_kernel": []} if selector is None else manifest
        args = ["binsec.py", "--target", target, "--allow-missing-binsec"]
        if selector is not None:
          args += ["--kernel", selector]
        with patch.object(binsec, "ROOT", root), patch.object(binsec, "load_manifest", return_value=ct), \
             patch.object(binsec, "find_binsec", return_value=None) as probe, \
             patch.object(binsec, "candidate_identity", return_value={}), patch.object(sys, "argv", args), \
             contextlib.redirect_stderr(io.StringIO()) as stderr, contextlib.redirect_stdout(io.StringIO()):
          try:
            status = binsec.main()
          except SystemExit as error:
            status = error.code
        self.assertEqual(status, expected)
        for path, contents in before.items():
          self.assertEqual((output / path).read_bytes(), contents)
        if expected == 2:
          self.assertIn("no BINSEC kernel matches", stderr.getvalue())
          probe.assert_not_called()
          self.assertEqual(snapshot(output), before)
        if selector in ("known", "known_symbol") and target == TARGET:
          self.assertTrue((output / "known/binsec-report.json").is_file())


if __name__ == "__main__":
  unittest.main()
