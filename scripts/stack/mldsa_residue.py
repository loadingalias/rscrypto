#!/usr/bin/env -S uv run python
"""Qualify native ML-DSA signing residue without instrumenting the measured build.

Capture intermediates in an isolated copy of the selected production source,
then match them against a separate uninstrumented signer's private stack.
This records functional cleanup evidence, never timing evidence.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BINARY = "rscrypto-mldsa-residue-native"
CRATE = Path("tools/residue-harness")
SOURCE = Path("src/auth/mldsa/portable.rs")

CAPTURE_HELPERS = '''
// Residue capture only: injected into an isolated source copy. Never present
// in the measured build. The observer borrows initialized production scratch.
fn residue_observe(attempt: usize, stage: u32, index: usize, bytes: &[u8]) {
  unsafe extern "C" {
    fn rscrypto_residue_observe(attempt: usize, stage: u32, index: usize, data: *const u8, len: usize);
  }
  // SAFETY: the capture harness synchronously borrows these initialized bytes
  // and returns before the production owner can change or drop.
  unsafe { rscrypto_residue_observe(attempt, stage, index, bytes.as_ptr(), bytes.len()) };
}

fn residue_poly(attempt: usize, stage: u32, index: usize, value: &Poly) {
  // SAFETY: a 256-element u32 array has 1024 initialized bytes and no padding.
  // This shared byte view neither mutates nor outlives the borrowed owner.
  let bytes = unsafe { core::slice::from_raw_parts(value.0.as_ptr().cast::<u8>(), 1024) };
  residue_observe(attempt, stage, index, bytes);
}
'''


def instrument(text: str) -> str:
  replacements = [
    ("fn index_byte(index: usize)", CAPTURE_HELPERS + "\nfn index_byte(index: usize)"),
    ("    for (z, y) in z.iter_mut().zip(&y) {\n      z.copy_from(y);\n      z.ntt();\n    }\n    let mut hash", "    for (i, (z, y)) in z.iter_mut().zip(&y).enumerate() {\n      z.copy_from(y);\n      z.ntt();\n      residue_poly(attempt, 1, i, y);\n      residue_poly(attempt, 2, i, z);\n    }\n    let mut hash"),
    ("    sampling::challenge(&commitment.as_array()[..p.challenge_bytes], p.tau, &mut challenge)?;", "    residue_observe(attempt, 3, 0, &commitment.as_array()[..p.challenge_bytes]);\n    sampling::challenge(&commitment.as_array()[..p.challenge_bytes], p.tau, &mut challenge)?;"),
    ("    // This is the scheme's aggregate rejection decision, never an error oracle.", "    for (i, response) in z.iter().enumerate() { residue_poly(attempt, 4, i, response); }\n    residue_observe(attempt, 5, 0, &[u8::from(invalid != 0)]);\n    // This is the scheme's aggregate rejection decision, never an error oracle."),
    ("    if invalid != 0 || count > p.omega {", "    for (i, hints) in w.iter().enumerate() { residue_poly(attempt, 6, i, hints); }\n    residue_observe(attempt, 7, 0, &[u8::from(invalid != 0 || count > p.omega)]);\n    if invalid != 0 || count > p.omega {"),
  ]
  for before, after in replacements:
    if text.count(before) != 1:
      raise ValueError(f"capture seam moved: {before!r}")
    text = text.replace(before, after)
  return text


def command(args: list[str], *, cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
  with log.open("w") as output:
    result = subprocess.run(args, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=False)
  if result.returncode:
    raise RuntimeError(f"command failed ({result.returncode}); see {log}: {args}")


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, target: Path, backend: str, capture: bool, log: Path) -> Path:
  features = ["capture-mldsa" if capture else "native-mldsa"]
  if backend == "portable":
    features.append("portable-only")
  env = dict(os.environ)
  env.pop("RUSTFLAGS", None)
  env.pop("CARGO_ENCODED_RUSTFLAGS", None)
  command([
    "cargo", "build", "--locked", "--release", "--manifest-path", str(root / CRATE / "Cargo.toml"),
    "--bin", BINARY, "--features", ",".join(features), "--target-dir", str(target),
  ], cwd=root, log=log, env=env)
  return target / "release" / BINARY


def parse(path: Path) -> dict[str, dict]:
  cases = {}
  current = None
  for line in path.read_text().splitlines():
    if line.startswith("case "):
      if current is not None:
        raise ValueError("unterminated case")
      name = line.split()[1]
      if name in cases:
        raise ValueError("duplicate case")
      current = {"capture": [], "signature": None}
      cases[name] = current
    elif line.startswith("capture ") and current is not None:
      _, attempt, stage, index, value = line.split()
      current["capture"].append((int(attempt), int(stage), int(index), bytes.fromhex(value)))
    elif line.startswith("signature ") and current is not None:
      if current["signature"] is not None:
        raise ValueError("duplicate signature")
      current["signature"] = bytes.fromhex(line.split()[1])
    elif line == "end":
      if current is None or current["signature"] is None:
        raise ValueError("missing signature")
      current = None
    elif line:
      raise ValueError(f"unexpected capture output: {line[:120]}")
  if current is not None or not cases:
    raise ValueError("incomplete run")
  return cases


def windows(data: bytes, hints: bool) -> list[tuple[int, bytes]]:
  if not hints:
    if len(data) < 16:
      raise ValueError("short needle")
    return [(offset, data[offset:offset + 16]) for offset in range(0, len(data) - 15, 16)]
  # Four u32 hint coefficients produce low-entropy 16-byte strings that also
  # occur in cleared memory. Use 64 coefficients and at least four set hints.
  selected = []
  for offset in range(0, len(data) - 255, 4):
    block = data[offset:offset + 256]
    coefficients = [int.from_bytes(block[i:i + 4], "little") for i in range(0, 256, 4)]
    if any(value not in (0, 1) for value in coefficients):
      raise ValueError("noncanonical capture hint")
    if sum(coefficients) >= 4:
      selected.append((offset, block))
  if not selected:
    # Sparse maps need their complete 256-coefficient pattern. A zero map
    # already equals the cleanup value and cannot identify retained secrets.
    return [(0, data)] if any(data) else []
  return selected


def needles(case: dict) -> tuple[list[dict], dict]:
  records = case["capture"]
  if not records:
    raise ValueError("capture missing")
  attempts = sorted({row[0] for row in records})
  if attempts != list(range(attempts[-1] + 1)):
    raise ValueError("capture attempts missing")
  if len(attempts) > 821:
    raise ValueError("capture exceeds production attempt bound")
  profiles = {2420: (4, 4, 32), 3309: (6, 5, 48), 4627: (8, 7, 64)}
  k, l, commitment_bytes = profiles[len(case["signature"])]
  keyed = {(attempt, stage, index): data for attempt, stage, index, data in records}
  if len(keyed) != len(records):
    raise ValueError("duplicate intermediate")
  norm = {attempt: data[0] for attempt, stage, _, data in records if stage == 5}
  hint = {attempt: data[0] for attempt, stage, _, data in records if stage == 7}
  if set(norm) != set(attempts) or norm[attempts[-1]] or hint.get(attempts[-1]) != 0:
    raise ValueError("capture has no final accepted attempt")
  for attempt in attempts[:-1]:
    if not norm[attempt] and hint.get(attempt) != 1:
      raise ValueError("capture did not reject a non-final attempt")
  expected = {}
  for attempt in attempts:
    for stage in (1, 2, 4):
      expected.update({(attempt, stage, index): 1024 for index in range(l)})
    expected[(attempt, 3, 0)] = commitment_bytes
    expected[(attempt, 5, 0)] = 1
    if not norm[attempt]:
      expected.update({(attempt, 6, index): 1024 for index in range(k)})
      expected[(attempt, 7, 0)] = 1
  if expected.keys() != keyed.keys() or any(len(keyed[key]) != size for key, size in expected.items()):
    raise ValueError("missing or malformed production intermediate")
  if any(value not in (0, 1) for value in [*norm.values(), *hint.values()]):
    raise ValueError("invalid rejection flag")
  output = []
  names = {1: "mask", 2: "ntt_mask", 3: "rejected_commitment", 4: "rejected_response", 6: "hint_map"}
  for attempt, stage, index, data in records:
    if stage not in names or (stage in (3, 4) and attempt == attempts[-1]):
      continue
    output.append({
      "attempt": attempt, "kind": names[stage], "index": index, "data": data,
      "windows": windows(data, stage == 6), "rejected": attempt != attempts[-1],
    })
  return output, {
    "attempts": len(attempts), "norm_rejections": sum(norm.values()),
    "hint_rejections": sum(hint.values()),
  }


def read_snapshot(path: Path) -> bytes:
  stack = path.read_bytes()
  if len(stack) != 1024 * 1024 or stack[:4096] != b"\xa5" * 4096:
    raise ValueError(f"truncated snapshot or exhausted stack: {path}")
  return stack


def judge(captured: dict[str, dict], measured: dict[str, dict], directory: Path) -> tuple[dict, dict[str, bytes]]:
  if captured.keys() != measured.keys():
    raise ValueError("capture/measurement case mismatch")
  rows = []
  controls = {}
  for name, case in captured.items():
    if case["signature"] != measured[name]["signature"] or measured[name]["capture"]:
      raise ValueError(f"signature mismatch or instrumented measurement: {name}")
    stack = read_snapshot(directory / f"{name}.stack")
    touched = len(stack.lstrip(b"\xa5"))
    values, counters = needles(case)
    if any(not window.strip(b"\xa5") for value in values for _, window in value["windows"]):
      raise ValueError("captured needle is indistinguishable from stack paint")
    # Keep the largest needle's overlap: its first bytes may equal the paint.
    # Trimming exactly at the first changed byte would miss such a fragment.
    overlap = max((len(window) for value in values for _, window in value["windows"]), default=1) - 1
    first = max(0, len(stack) - touched - overlap)
    stack = stack[first:]
    found = []
    groups = {}
    for value in values:
      group = groups.setdefault(value["kind"], {"objects": 0, "windows": 0, "bytes": 0})
      group["objects"] += 1
      group["windows"] += len(value["windows"])
      group["bytes"] += len(value["data"])
      if value["windows"]:
        control = value["windows"][0][1]
        controls.setdefault(f"{value['kind']}_{len(control)}", control)
      for offset, window in value["windows"]:
        position = stack.find(window)
        if position >= 0:
          found.append({
            "kind": value["kind"], "attempt": value["attempt"], "index": value["index"],
            "needle_offset": offset, "stack_offset": first + position, "window_bytes": len(window),
            "rejected": value["rejected"],
          })
    rows.append({"case": name, **counters, "stack_touched_bytes": touched, "groups": groups, "found": found})
  return {"cases": rows, "passed": all(not row["found"] for row in rows)}, controls


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--cases", type=int, default=8, help="message indices per parameter set and mode (1..128)")
  parser.add_argument("--start", type=int, default=0, help="first deterministic message index (0..4096)")
  parser.add_argument("--backend", action="append", choices=("native", "portable"))
  args = parser.parse_args(argv)
  if not 1 <= args.cases <= 128:
    parser.error("--cases must be 1..128")
  if not 0 <= args.start <= 4096:
    parser.error("--start must be 0..4096")
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=True)
  identity = {
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "rustc": subprocess.check_output(["rustc", "-vV"], cwd=ROOT, text=True),
    "platform": platform.platform(), "machine": platform.machine(),
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "command": sys.argv, "cases_per_mode": args.cases, "first_message_index": args.start,
    "measurement": "release opt-level=3 lto=fat codegen-units=1 panic=abort overflow-checks=true",
    "features": "alloc,ml-dsa,ml-kem,slh-dsa,std; portable adds portable-only",
    "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "src").rglob("*.rs"))},
    "build_input_sha256": {
      name: sha256(ROOT / name) for name in [
        "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml", ".config/target-matrix.json",
        "tools/residue-harness/Cargo.toml", "tools/residue-harness/Cargo.lock",
        "tools/residue-harness/src/mldsa_native.rs", "scripts/stack/mldsa_residue.py",
      ]
    },
  }
  (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
  original = (ROOT / SOURCE).read_text()
  patched = instrument(original)
  (output / "capture.patch").write_text("".join(difflib.unified_diff(
    original.splitlines(True), patched.splitlines(True), fromfile="a/" + str(SOURCE), tofile="b/" + str(SOURCE)
  )))
  reports = []
  with tempfile.TemporaryDirectory(prefix="rscrypto-mldsa-residue-capture-") as temporary:
    capture_root = Path(temporary)
    for name in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]:
      shutil.copy2(ROOT / name, capture_root / name)
    for name in ["src", "examples", "benches", ".cargo", ".config", "rscrypto-rustls", str(CRATE)]:
      shutil.copytree(ROOT / name, capture_root / name, ignore=shutil.ignore_patterns("target", ".DS_Store"))
    (capture_root / SOURCE).write_text(patched)
    for backend in args.backend or ["native", "portable"]:
      directory = output / backend
      directory.mkdir(exist_ok=True)
      capture = build(capture_root, ROOT / "target/mldsa-residue-capture" / backend, backend, True, directory / "capture-build.log")
      command([str(capture), str(directory), str(args.cases), str(args.start)], cwd=ROOT, log=directory / "capture.log")
      measured = build(ROOT, ROOT / "target/mldsa-residue-native" / backend, backend, False, directory / "measurement-build.log")
      command([str(measured), str(directory), str(args.cases), str(args.start)], cwd=ROOT, log=directory / "measurement.log")
      shutil.copy2(measured, directory / BINARY)
      captured = parse(directory / "capture.log")
      measured_cases = parse(directory / "measurement.log")
      expected_cases = {
        f"{parameter}-{form}-{mode}-{index}"
        for parameter in ("mldsa44", "mldsa65", "mldsa87")
        for form in ("compact", "prepared")
        for mode in ("deterministic", "hedged")
        for index in range(args.start, args.start + args.cases)
      }
      if captured.keys() != expected_cases:
        raise ValueError("capture omitted requested cases")
      report, control_values = judge(captured, measured_cases, directory)
      report["backend"] = backend
      report["capture_binary_sha256"] = sha256(capture)
      report["measurement_binary_sha256"] = sha256(measured)
      fixture = b"".join(control_values.values())
      (directory / "control.bin").write_bytes(fixture)
      command([str(measured), str(directory), "control=" + str(directory / "control.bin")], cwd=ROOT, log=directory / "control.log")
      planted = read_snapshot(directory / "planted.stack")
      cleared = read_snapshot(directory / "cleared.stack")
      report["controls"] = {
        kind: {"bytes": len(value), "planted_detected": value in planted, "cleared_absent": value not in cleared}
        for kind, value in control_values.items()
      }
      report["passed"] &= all(value["planted_detected"] and value["cleared_absent"] for value in report["controls"].values())
      (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
      reports.append(report)
      print(f"{backend}: {len(report['cases'])} signatures; residue={'PASS' if report['passed'] else 'FAIL'}", flush=True)
  final_sources = {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "src").rglob("*.rs"))}
  final_inputs = {
    name: sha256(ROOT / name) if (ROOT / name).is_file() else None for name in identity["build_input_sha256"]
  }
  changed = {
    name: {"before": before, "after": after}
    for original_hashes, final_hashes in [
      (identity["source_sha256"], final_sources), (identity["build_input_sha256"], final_inputs)
    ]
    for name in original_hashes.keys() | final_hashes.keys()
    if (before := original_hashes.get(name)) != (after := final_hashes.get(name))
  }
  identity["completed_utc"] = datetime.now(timezone.utc).isoformat()
  identity["inputs_unchanged"] = not changed
  identity["changed_inputs"] = changed
  (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
  for report in reports:
    report["source_unchanged"] = not changed
    report["passed"] &= not changed
    (output / report["backend"] / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  (output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
  if changed:
    print("Qualification failed: source or build inputs changed during the run; see identity.json", file=sys.stderr)
  return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
  sys.exit(main())
