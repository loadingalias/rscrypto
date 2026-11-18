#!/usr/bin/env -S uv run python
"""Measure native SLH-DSA-SHA2-128s secret residue and resources on a private, painted stack.

The harness binary runs each production call once on a fresh stack. It derives
every secret of the call with an independent FIPS 205 oracle over the `sha2`
crate and judges the stack. This records cleanup and resource evidence, never
timing evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from mldsa_residue import ROOT, build, command, sha256

BINARY = "rscrypto-slhdsa-residue-native"
INPUTS = [
  "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml",
  "tools/residue-harness/Cargo.toml", "tools/residue-harness/Cargo.lock",
  "tools/residue-harness/src/slhdsa_native.rs", "tools/residue-harness/src/native_stack.rs",
  "scripts/stack/slhdsa_residue.py", "scripts/stack/mldsa_residue.py",
]


def sources() -> dict[str, str | None]:
  files = {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "src").rglob("*.rs"))}
  return files | {name: sha256(ROOT / name) if (ROOT / name).is_file() else None for name in INPUTS}


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--cases", type=int, default=2, help="keys and messages per scenario (1..16)")
  parser.add_argument("--backend", action="append", choices=("native", "portable"))
  args = parser.parse_args(argv)
  if not 1 <= args.cases <= 16:
    parser.error("--cases must be 1..16")
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  before = sources()
  identity = {
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "rustc": subprocess.check_output(["rustc", "-vV"], cwd=ROOT, text=True),
    "platform": platform.platform(), "machine": platform.machine(),
    "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "cases": args.cases,
    "measurement": "release opt-level=3 lto=fat codegen-units=1 panic=abort overflow-checks=true",
    "features": "rscrypto std plus the harness features; portable adds portable-only",
    "inputs_sha256": before,
  }
  reports = []
  for backend in args.backend or ["native", "portable"]:
    directory = output / backend
    directory.mkdir()
    binary = build(ROOT, ROOT / "target/slhdsa-residue-native" / backend, backend, "native-slhdsa",
                   directory / "build.log", BINARY)
    shutil.copy2(binary, directory / BINARY)
    status = subprocess.run([str(directory / BINARY), str(directory), str(args.cases)], cwd=ROOT,
                            stdout=(directory / "run.log").open("w"), stderr=subprocess.STDOUT, check=False)
    if status.returncode not in (0, 1):
      raise RuntimeError(f"harness failed ({status.returncode}); see {directory / 'run.log'}")
    report = json.loads((directory / "report.json").read_text())
    report["backend"] = backend
    report["binary_sha256"] = sha256(directory / BINARY)
    reports.append(report)
    print((directory / "run.log").read_text(), end="")
    print(f"{backend}: residue={'PASS' if report['passed'] else 'FAIL'}", flush=True)
  after = sources()
  identity["completed_utc"] = datetime.now(timezone.utc).isoformat()
  identity["inputs_unchanged"] = before == after
  (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
  for report in reports:
    report["passed"] &= identity["inputs_unchanged"]
  (output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
  (output / "SHA256SUMS").write_text("".join(
    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}\n"
    for path in sorted(output.rglob("*")) if path.is_file()
  ))
  if not identity["inputs_unchanged"]:
    print("Qualification failed: source or build inputs changed during the run", file=sys.stderr)
  return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
  sys.exit(main())
