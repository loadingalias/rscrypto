#!/usr/bin/env python3
"""Build a machine-readable report from dudect-bencher output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from toml_compat import tomllib
from provenance import cfg_target_features, codegen_value, codegen_values, resolved_rustflags


SEED_RE = re.compile(r"^bench\s+(?P<name>\S+)\s+seeded with (?P<seed>0x[0-9a-fA-F]+)$")
RESULT_RE = re.compile(
  r"^bench (?P<name>\S+)\s+\.\.\. : n == (?P<n>[+-]?[0-9.]+)M, "
  r"max t = (?P<t>[+-]?[0-9.]+), max tau = (?P<tau>[+-]?[0-9.]+), "
  r"\(5/tau\)\^2 = (?P<needed>[0-9]+)$"
)


def manifest_dudect_cases(ct: dict[str, Any]) -> dict[str, dict[str, Any]]:
  cases: dict[str, dict[str, Any]] = {}
  for raw_case in ct.get("dudect_case", []):
    missing = [key for key in ("name", "primitive", "filter", "left_class", "right_class") if not raw_case.get(key)]
    if missing:
      raise ValueError(f"dudect_case missing required keys {missing}: {raw_case!r}")
    name = str(raw_case["name"])
    if name in cases:
      raise ValueError(f"duplicate dudect_case name {name!r}")
    gate = str(raw_case.get("gate", "required"))
    if gate not in ("required", "diagnostic"):
      raise ValueError(f"dudect_case {name!r} has unsupported gate {gate!r}")
    case = dict(raw_case)
    case["gate"] = gate
    cases[name] = case
  return cases


def dudect_case_rows(
  results: dict[str, dict[str, Any]],
  seeds: dict[str, str],
  raw_rows: dict[str, dict[str, Any]],
  manifest_cases: dict[str, dict[str, Any]],
  *,
  threshold: float,
  requested_samples: int,
) -> list[dict[str, Any]]:
  cases = []
  for name in sorted(results):
    metadata = manifest_cases.get(name)
    if metadata is None:
      raise ValueError(f"DudeCT emitted case {name!r}, which is not declared in ct.toml")
    result = results[name]
    gate = str(metadata["gate"])
    passed = result["abs_max_t"] <= threshold
    diagnostic = gate == "diagnostic"
    cases.append(
      {
        "name": name,
        "primitive": metadata["primitive"],
        "left_class": metadata["left_class"],
        "right_class": metadata["right_class"],
        "gate": gate,
        "diagnostic_reason": metadata.get("reason") or metadata.get("notes"),
        "seed": seeds.get(name),
        "requested_samples": requested_samples,
        "raw_csv": raw_rows.get(name, {"row_count": 0, "labels": {}}),
        **result,
        "threshold_abs_max_t": threshold,
        "status": "pass" if passed else ("diagnostic-fail" if diagnostic else "fail"),
      }
    )
  return cases


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def rustc_verbose() -> str:
  try:
    return subprocess.check_output(["rustc", "-vV"], text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return "unavailable"


def parse_stdout(path: Path) -> tuple[dict[str, str], dict[str, dict]]:
  seeds: dict[str, str] = {}
  results: dict[str, dict] = {}
  for line in path.read_text(errors="replace").splitlines():
    if seed_match := SEED_RE.match(line):
      seeds[seed_match.group("name")] = seed_match.group("seed")
      continue
    if result_match := RESULT_RE.match(line):
      name = result_match.group("name")
      max_t = float(result_match.group("t"))
      max_tau = float(result_match.group("tau"))
      results[name] = {
        "samples_millions": float(result_match.group("n")),
        "max_t": max_t,
        "abs_max_t": abs(max_t),
        "max_tau": max_tau,
        "needed_samples_for_tau_threshold": int(result_match.group("needed")),
      }
  return seeds, results


def raw_csv_rows(path: Path) -> dict[str, dict]:
  rows: dict[str, dict] = {}
  if not path.exists():
    return rows

  with path.open(newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
      name = row.get("benchname", "")
      class_name = row.get("class", "")
      if not name or not class_name:
        continue
      entry = rows.setdefault(name, {"row_count": 0, "labels": {}})
      entry["row_count"] += 1
      entry["labels"].setdefault(class_name, 0)
      entry["labels"][class_name] += 1
  return rows


def owner_symbol_evidence(
  symbols_text: str,
  expected_symbols: set[str],
) -> tuple[dict[str, int], dict[int, str]]:
  counts = {symbol: 0 for symbol in sorted(expected_symbols)}
  addresses: dict[int, str] = {}
  definition = re.compile(r"^\s*([0-9a-fA-F]+)\s+\S\s+_?(ct_entry_owner_eq_[0-9]+)\s*$")
  for line in symbols_text.splitlines():
    if match := definition.match(line):
      symbol = match.group(2)
      if symbol in expected_symbols:
        counts[symbol] += 1
        addresses[int(match.group(1), 16)] = symbol
  return counts, addresses


def owner_call_site_counts(
  disassembly_text: str,
  expected_symbols: set[str],
  symbols_by_address: dict[int, str],
) -> dict[str, int]:
  counts = {symbol: 0 for symbol in sorted(expected_symbols)}
  relative_relocations: dict[int, int] = {}
  relative_relocation = re.compile(
    r"^\s*([0-9a-fA-F]+)\s+R_[A-Z0-9_]+_RELATIVE\s+\*ABS\*\+0x([0-9a-fA-F]+)\s*$"
  )
  for line in disassembly_text.splitlines():
    if match := relative_relocation.match(line):
      relative_relocations[int(match.group(1), 16)] = int(match.group(2), 16)

  current_symbol = ""
  function_label = re.compile(r"^[0-9a-fA-F]+ <(.+)>:$")
  instruction_pattern = re.compile(r"\b(?:bl|brasl|call|callq|jal|jalr)\b")
  for line in disassembly_text.splitlines():
    if match := function_label.match(line.strip()):
      current_symbol = match.group(1).removeprefix("_")
      continue
    instruction = instruction_pattern.search(line)
    if instruction is None:
      continue

    called: set[str] = set()
    for symbol in expected_symbols:
      if re.search(rf"<_?{re.escape(symbol)}(?:\+[^>]*)?>", line):
        called.add(symbol)
    if instruction.group(0) != "jalr" and (
      target := re.search(r"\b0x([0-9a-fA-F]+)\b", line[instruction.end() :])
    ):
      if symbol := symbols_by_address.get(int(target.group(1), 16)):
        called.add(symbol)
    if slot := re.search(r"\*[^#]*#\s*0x([0-9a-fA-F]+)\b", line[instruction.end() :]):
      if target_address := relative_relocations.get(int(slot.group(1), 16)):
        if symbol := symbols_by_address.get(target_address):
          called.add(symbol)
    for symbol in called:
      if current_symbol != symbol:
        counts[symbol] += 1
  return counts


def linker_driver(command: str) -> str:
  assignment = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
  tokens = shlex.split(command)
  index = 0
  while index < len(tokens) and assignment.fullmatch(tokens[index]):
    index += 1
  if index == len(tokens):
    raise ValueError("DudeCT linker command does not identify the linker driver")

  if Path(tokens[index]).name == "env":
    index += 1
    while index < len(tokens):
      token = tokens[index]
      if assignment.fullmatch(token) or token in ("-i", "--ignore-environment"):
        index += 1
        continue
      if token in ("-u", "--unset", "-C", "--chdir"):
        index += 2
        continue
      if token.startswith(("--unset=", "--chdir=")):
        index += 1
        continue
      if token == "--":
        index += 1
      break

  for token in tokens[index:]:
    if assignment.fullmatch(token):
      continue
    if token.startswith("-"):
      break
    return token
  raise ValueError("DudeCT linker command does not identify the linker driver")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stdout", required=True, type=Path)
  parser.add_argument("--csv", required=True, type=Path)
  parser.add_argument("--out", required=True, type=Path)
  parser.add_argument("--target", required=True)
  parser.add_argument("--profile", default="release")
  parser.add_argument("--threshold", type=float, default=10.0)
  parser.add_argument("--samples", type=int, required=True)
  parser.add_argument("--command", default="")
  parser.add_argument("--binary", required=True, type=Path)
  parser.add_argument("--binary-disassembly", required=True, type=Path)
  parser.add_argument("--binary-symbols", required=True, type=Path)
  parser.add_argument("--linker-command-log", required=True, type=Path)
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[2]
  with (root / "Cargo.toml").open("rb") as source:
    crate_version = tomllib.load(source)["package"]["version"]
  ct_manifest_path = root / "ct.toml"
  with ct_manifest_path.open("rb") as source:
    ct_manifest = tomllib.load(source)
  release_binary = ct_manifest["equality_evidence"]["release_binary"]
  manifest_cases = manifest_dudect_cases(ct_manifest)
  dudect_manifest_path = root / "tools" / "ct-dudect" / "Cargo.toml"
  harness_manifest_path = root / "tools" / "ct-harness" / "Cargo.toml"
  dudect_lockfile_path = root / "tools" / "ct-dudect" / "Cargo.lock"
  with dudect_manifest_path.open("rb") as source:
    dudect_manifest = tomllib.load(source)
  configured_rustflags, environment_rustflags, effective_rustflags, rustflags_source = resolved_rustflags(
    root, args.target
  )
  target_cfg = subprocess.check_output(
    ["rustc", "--print", "cfg", "--target", args.target, *effective_rustflags],
    cwd=root,
    text=True,
  )
  expected_owner_symbols = {f"ct_entry_owner_eq_{width}" for width in release_binary["formal_owner_widths"]}
  for path in (args.binary, args.binary_disassembly, args.binary_symbols, args.linker_command_log):
    if not path.is_file():
      raise ValueError(f"DudeCT evidence artifact missing: {path}")

  symbol_counts, owner_symbols_by_address = owner_symbol_evidence(
    args.binary_symbols.read_text(),
    expected_owner_symbols,
  )
  wrong_symbol_counts = {symbol: count for symbol, count in symbol_counts.items() if count != 1}
  if wrong_symbol_counts:
    raise ValueError(f"DudeCT binary owner equality symbols must occur exactly once: {wrong_symbol_counts}")
  owner_call_sites = owner_call_site_counts(
    args.binary_disassembly.read_text(errors="replace"),
    expected_owner_symbols,
    owner_symbols_by_address,
  )
  missing_call_sites = {symbol: count for symbol, count in owner_call_sites.items() if count < 1}
  if missing_call_sites:
    raise ValueError(f"DudeCT binary does not call every owner equality symbol: {missing_call_sites}")

  linker_command = next(
    (line for line in args.linker_command_log.read_text().splitlines() if '"-o"' in line),
    "",
  )
  linker = linker_driver(linker_command)
  linker_path_text = shutil.which(linker)
  if linker_path_text is None:
    raise ValueError(f"DudeCT linker driver is not resolvable: {linker}")
  linker_path = Path(linker_path_text).resolve()
  linker_version = subprocess.check_output([str(linker_path), "--version"], cwd=root, text=True, stderr=subprocess.STDOUT).strip()
  git_status = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=root, text=True).splitlines()

  seeds, results = parse_stdout(args.stdout)
  raw_rows = raw_csv_rows(args.csv)
  cases = dudect_case_rows(
    results,
    seeds,
    raw_rows,
    manifest_cases,
    threshold=args.threshold,
    requested_samples=args.samples,
  )

  report = {
    "schema_version": 2,
    "kind": "rscrypto.ct.dudect",
    "crate": "rscrypto",
    "crate_version": crate_version,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "git_dirty": bool(git_status),
    "git_status": git_status,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "target": args.target,
    "target_triple": args.target,
    "profile": args.profile,
    "profile_settings": dudect_manifest.get("profile", {}).get(args.profile, {}),
    "features": release_binary["features"],
    "default_features": release_binary["default_features"],
    "backend": release_binary["backend"],
    "ct_manifest_sha256": sha256_file(ct_manifest_path),
    "dudect_manifest_sha256": sha256_file(dudect_manifest_path),
    "harness_manifest_sha256": sha256_file(harness_manifest_path),
    "dudect_lockfile_sha256": sha256_file(dudect_lockfile_path),
    "cargo": subprocess.check_output(["cargo", "-V"], cwd=root, text=True).strip(),
    "configured_rustflags": configured_rustflags,
    "environment_rustflags": environment_rustflags,
    "effective_rustflags": effective_rustflags,
    "rustflags_source": rustflags_source,
    "target_cpu": codegen_value(effective_rustflags, "target-cpu"),
    "target_features": codegen_values(effective_rustflags, "target-feature"),
    "target_cfg_features": cfg_target_features(target_cfg),
    "linker": linker,
    "linker_path": str(linker_path),
    "linker_sha256": sha256_file(linker_path),
    "linker_version": linker_version,
    "binary": {
      "path": str(args.binary),
      "sha256": sha256_file(args.binary),
      "bytes": args.binary.stat().st_size,
      "owner_symbols": sorted(expected_owner_symbols),
      "owner_call_sites": owner_call_sites,
    },
    "binary_disassembly": {
      "path": str(args.binary_disassembly),
      "sha256": sha256_file(args.binary_disassembly),
      "bytes": args.binary_disassembly.stat().st_size,
    },
    "binary_symbols": {
      "path": str(args.binary_symbols),
      "sha256": sha256_file(args.binary_symbols),
      "bytes": args.binary_symbols.stat().st_size,
    },
    "linker_command_log": {
      "path": str(args.linker_command_log),
      "sha256": sha256_file(args.linker_command_log),
      "bytes": args.linker_command_log.stat().st_size,
    },
    "threshold_abs_max_t": args.threshold,
    "requested_samples": args.samples,
    "command": args.command,
    "rustc_verbose": rustc_verbose(),
    "host": {
      "system": platform.system(),
      "release": platform.release(),
      "machine": platform.machine(),
      "processor": platform.processor(),
      "python": platform.python_version(),
    },
    "raw_stdout": str(args.stdout),
    "raw_stdout_sha256": sha256_file(args.stdout) if args.stdout.exists() else None,
    "raw_csv": str(args.csv),
    "raw_csv_sha256": sha256_file(args.csv) if args.csv.exists() else None,
    "cases": cases,
    "case_count": len(cases),
    "failure_count": sum(1 for case in cases if case["status"] == "fail"),
    "diagnostic_failure_count": sum(1 for case in cases if case["status"] == "diagnostic-fail"),
    "notes": [
      "DudeCT is empirical timing evidence, not a proof.",
      "A pass means no leakage was detected for this configuration, host, and input classification.",
      "dudect-bencher 0.7.0 writes both raw CSV classes with label 0; raw CSV labels are recorded for traceability, not class-balance proof.",
    ],
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"dudect report: {args.out}")
  return 1 if report["failure_count"] else 0


if __name__ == "__main__":
  raise SystemExit(main())
