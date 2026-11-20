#!/usr/bin/env -S uv run python
"""Build a machine-readable report from dudect-bencher output."""

from __future__ import annotations

import argparse
import sys
import csv
import json
import math
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Embedded Windows Python omits the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import evidence_bundle as bundle
from manifest import dudect_min_median_ticks, resolve_dudect_case
from provenance import cfg_target_features, codegen_value, codegen_values, dudect_runner_sources, resolved_rustflags, sha256_file


SEED_RE = re.compile(r"^bench\s+(?P<name>\S+)\s+seeded with (?P<seed>0x[0-9a-fA-F]+)$")
TICK_RE = re.compile(r"^timer tick (?P<tick>[0-9]+) ns$")
CALLS_RE = re.compile(r"^bench (?P<name>\S+) timed (?P<calls>[0-9]+) calls per sample$")
RESULT_RE = re.compile(
  r"^bench (?P<name>\S+)\s+\.\.\. : n == (?P<n>[+-]?[0-9.]+)M, "
  r"max t = (?P<t>[+-]?[0-9.]+), max tau = (?P<tau>[+-]?[0-9.]+), "
  r"\(5/tau\)\^2 = (?P<needed>[0-9]+)$"
)


def manifest_dudect_cases(ct: dict[str, Any], target: str | None = None) -> dict[str, dict[str, Any]]:
  """Load manifest cases with each gate resolved for `target`."""
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
    cases[name] = resolve_dudect_case({**raw_case, "gate": gate}, target)
  return cases


def dudect_case_rows(
  results: dict[str, dict[str, Any]],
  seeds: dict[str, str],
  raw_rows: dict[str, dict[str, Any]],
  manifest_cases: dict[str, dict[str, Any]],
  *,
  threshold: float,
  requested_samples: int,
  tick_ns: int,
  min_median_ticks: int,
  filter_value: str = "",
) -> list[dict[str, Any]]:
  """Decide each case: its |t| must stay under the threshold, and its median sample must span
  at least `min_median_ticks` timer ticks, so that clock quantization cannot hide a difference."""
  requested = {name for name in manifest_cases if filter_value in name}
  if not requested:
    raise ValueError(f"DudeCT filter {filter_value!r} selects no manifest cases")
  completed = set(results)
  if completed != requested:
    raise ValueError(
      f"DudeCT completed case set does not match selection: "
      f"missing={sorted(requested - completed)}, unexpected={sorted(completed - requested)}"
    )
  if set(raw_rows) != requested:
    raise ValueError("DudeCT raw CSV case set does not match selection")
  cases = []
  for name in sorted(results):
    raw = raw_rows[name]
    if raw["row_count"] != requested_samples or any(raw["labels"].get(label, 0) == 0 for label in ("0", "1")):
      raise ValueError(f"DudeCT case {name!r} raw CSV must contain all {requested_samples} samples and both classes")
    metadata = manifest_cases[name]
    result = results[name]
    gate = str(metadata["gate"])
    limit = metadata.get("threshold_abs_max_t", threshold)
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
      raise ValueError(f"DudeCT case {name!r} threshold must be finite and positive")
    limit = min(threshold, limit)
    passed = result["abs_max_t"] < limit if "threshold_abs_max_t" in metadata else result["abs_max_t"] <= limit
    if tick_ns < 1 or "calls_per_sample" not in result:
      raise ValueError(f"DudeCT case {name!r} has no timer tick or calls per sample")
    median_ticks = raw["median_runtime_ns"] / tick_ns
    resolved = median_ticks >= min_median_ticks
    diagnostic = gate == "diagnostic"
    cases.append(
      {
        "name": name,
        "primitive": metadata["primitive"],
        "left_class": metadata["left_class"],
        "right_class": metadata["right_class"],
        "gate": gate,
        "diagnostic_reason": metadata.get("reason"),
        "seed": seeds.get(name),
        "requested_samples": requested_samples,
        "raw_csv": raw,
        **result,
        "threshold_abs_max_t": limit,
        "timer_tick_ns": tick_ns,
        "median_ticks": median_ticks,
        "min_median_ticks": min_median_ticks,
        "resolution": "pass" if resolved else "fail",
        "status": "pass" if passed and resolved else ("diagnostic-fail" if diagnostic else "fail"),
      }
    )
  return cases


def rustc_verbose() -> str:
  try:
    return subprocess.check_output(["rustc", "-vV"], text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return "unavailable"


def parse_stdout(path: Path) -> tuple[dict[str, str], dict[str, dict], int]:
  seeds: dict[str, str] = {}
  results: dict[str, dict] = {}
  calls: dict[str, int] = {}
  tick_ns = 0
  for line in path.read_text(errors="replace").splitlines():
    if seed_match := SEED_RE.match(line):
      seeds[seed_match.group("name")] = seed_match.group("seed")
      continue
    if tick_match := TICK_RE.match(line):
      tick_ns = int(tick_match.group("tick"))
      continue
    if calls_match := CALLS_RE.match(line):
      calls[calls_match.group("name")] = int(calls_match.group("calls"))
      continue
    if result_match := RESULT_RE.match(line):
      name = result_match.group("name")
      if name in results:
        raise ValueError(f"DudeCT emitted duplicate result for case {name!r}")
      max_t = float(result_match.group("t"))
      max_tau = float(result_match.group("tau"))
      results[name] = {
        "samples_millions": float(result_match.group("n")),
        "max_t": max_t,
        "abs_max_t": abs(max_t),
        "max_tau": max_tau,
        "needed_samples_for_tau_threshold": int(result_match.group("needed")),
      }
  for name, count in calls.items():
    if name in results:
      results[name]["calls_per_sample"] = count
  return seeds, results, tick_ns


def raw_csv_rows(path: Path) -> dict[str, dict]:
  rows: dict[str, dict] = {}
  runtimes: dict[str, list[int]] = {}
  with path.open(newline="") as fh:
    reader = csv.DictReader(fh)
    if reader.fieldnames != ["benchname", "sequence", "class", "runtime_ns"]:
      raise ValueError("DudeCT raw CSV requires benchname,sequence,class,runtime_ns columns")
    for row in reader:
      name = row.get("benchname")
      label = row.get("class")
      if not name or label not in ("0", "1") or None in row:
        raise ValueError(f"invalid DudeCT raw CSV row at line {reader.line_num}")
      if any(re.fullmatch(r"[0-9]+", row.get(key) or "") is None for key in ("sequence", "runtime_ns")):
        raise ValueError(f"invalid DudeCT raw CSV sequence or runtime at line {reader.line_num}")
      entry = rows.setdefault(name, {"row_count": 0, "labels": {"0": 0, "1": 0}})
      if int(row["sequence"]) != entry["row_count"]:
        raise ValueError(f"DudeCT raw CSV sequence for {name!r} is not contiguous at line {reader.line_num}")
      entry["row_count"] += 1
      entry["labels"][label] += 1
      runtimes.setdefault(name, []).append(int(row["runtime_ns"]))
  for name, entry in rows.items():
    entry["median_runtime_ns"] = statistics.median_low(runtimes[name])
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


# Each x86-64 operand name, mapped to the general register that it writes.
X86_REGISTERS = {
  name: register
  for register, names in {
    "rax": ("rax", "eax", "ax", "al", "ah"),
    "rbx": ("rbx", "ebx", "bx", "bl", "bh"),
    "rcx": ("rcx", "ecx", "cx", "cl", "ch"),
    "rdx": ("rdx", "edx", "dx", "dl", "dh"),
    "rsi": ("rsi", "esi", "si", "sil"),
    "rdi": ("rdi", "edi", "di", "dil"),
    "rbp": ("rbp", "ebp", "bp", "bpl"),
    "rsp": ("rsp", "esp", "sp", "spl"),
    **{f"r{n}": (f"r{n}", f"r{n}d", f"r{n}w", f"r{n}b") for n in range(8, 16)},
  }.items()
  for name in names
}
# A System V call preserves these registers.
X86_CALLEE_SAVED = {"rbx", "rbp", "rsp", "r12", "r13", "r14", "r15"}


def owner_call_site_counts(
  disassembly_text: str,
  expected_symbols: set[str],
  symbols_by_address: dict[int, str],
) -> dict[str, int]:
  counts = {symbol: 0 for symbol in sorted(expected_symbols)}
  relative_relocations: dict[int, int] = {}
  coff_call_relocations: set[int] = set()
  relative_relocation = re.compile(
    r"^\s*([0-9a-fA-F]+)\s+R_[A-Z0-9_]+_RELATIVE\s+\*ABS\*\+0x([0-9a-fA-F]+)\s*$"
  )
  for line in disassembly_text.splitlines():
    if match := relative_relocation.match(line):
      relative_relocations[int(match.group(1), 16)] = int(match.group(2), 16)

  current_symbol = ""
  # x86-64 registers that hold an owner entry, loaded from its relocated slot in this function.
  loaded: dict[str, str] = {}
  slot_load = re.compile(r"\bmovq?\s+-?0x[0-9a-fA-F]+\(%rip\),\s*%([a-z0-9]+)\s*#\s*0x([0-9a-fA-F]+)\b")
  register_call = re.compile(r"\bcallq?\s+\*%([a-z0-9]+)\s*$")
  last_register = re.compile(r"%([a-z0-9]+)\s*(?:#.*)?$")
  function_label = re.compile(r"^[0-9a-fA-F]+ <(.+)>:$")
  instruction_pattern = re.compile(r"\b(?:bl|brasl|call|callq|jal|jalr)\b")
  instruction_address = re.compile(r"^\s*([0-9a-fA-F]+):")
  coff_owner_relocation = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+IMAGE_REL_AMD64_REL32\s+_?(ct_entry_owner_eq_[0-9]+)\s*$"
  )
  for line in disassembly_text.splitlines():
    if match := function_label.match(line.strip()):
      current_symbol = match.group(1).removeprefix("_")
      loaded = {}
      continue
    if match := slot_load.search(line):
      if register := X86_REGISTERS.get(match.group(1)):
        target_address = relative_relocations.get(int(match.group(2), 16))
        if symbol := symbols_by_address.get(target_address):
          loaded[register] = symbol
        else:
          loaded.pop(register, None)
      continue
    if match := coff_owner_relocation.match(line):
      symbol = match.group(2)
      if (
        symbol in expected_symbols
        and int(match.group(1), 16) in coff_call_relocations
        and current_symbol != symbol
      ):
        counts[symbol] += 1
      continue
    instruction = instruction_pattern.search(line)
    if instruction is None:
      # The last register operand is the destination. An exchange also writes its other operands.
      written = [last.group(1)] if (last := last_register.search(line)) else []
      if re.search(r"\b(?:xchg|xadd|cmpxchg)", line):
        written = re.findall(r"%([a-z0-9]+)", line)
      for name in written:
        loaded.pop(X86_REGISTERS.get(name, name), None)
      continue

    if instruction.group(0) in ("call", "callq") and (address := instruction_address.match(line)):
      # AMD64 REL32 call relocations begin one byte after the E8 opcode.
      coff_call_relocations.add(int(address.group(1), 16) + 1)

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
    if (target := register_call.search(line)) and (register := X86_REGISTERS.get(target.group(1))):
      if symbol := loaded.get(register):
        called.add(symbol)
    for symbol in called:
      if current_symbol != symbol:
        counts[symbol] += 1
    if instruction.group(0) in ("call", "callq"):
      loaded = {register: symbol for register, symbol in loaded.items() if register in X86_CALLEE_SAVED}
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


def linker_version(linker: Path) -> str:
  version_argument = "/?" if linker.name.lower() == "link.exe" else "--version"
  result = subprocess.run(
    [str(linker), version_argument],
    cwd=linker.parent,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
  )
  version = result.stdout.strip()
  if not version:
    raise ValueError(f"DudeCT linker version output is empty: {linker}")
  return version


def arguments():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prepare", action="store_true")
  parser.add_argument("--stdout", type=Path)
  parser.add_argument("--csv", type=Path)
  parser.add_argument("--out", required=True, type=Path)
  parser.add_argument("--target", required=True)
  parser.add_argument("--profile", default="release")
  parser.add_argument("--threshold", type=float, default=10.0)
  parser.add_argument("--samples", type=int)
  parser.add_argument("--command", default="")
  parser.add_argument("--filter", default="")
  parser.add_argument("--binary", required=True, type=Path)
  parser.add_argument("--binary-object", type=Path)
  parser.add_argument("--binary-disassembly", required=True, type=Path)
  parser.add_argument("--binary-symbols", required=True, type=Path)
  parser.add_argument("--linker-command-log", required=True, type=Path)
  args = parser.parse_args()
  if not args.prepare and (args.stdout is None or args.csv is None or args.samples is None):
    parser.error("measurement reporting requires --stdout, --csv and --samples")
  return args


def prepare_report(args):
  root = Path(__file__).resolve().parents[2]
  with (root / "Cargo.toml").open("rb") as source:
    crate_version = tomllib.load(source)["package"]["version"]
  ct_manifest_path = root / "ct.toml"
  with ct_manifest_path.open("rb") as source:
    ct_manifest = tomllib.load(source)
  release_binary = ct_manifest["equality_evidence"]["release_binary"]
  manifest_cases = manifest_dudect_cases(ct_manifest, args.target)
  dudect_manifest_path = root / "tools" / "ct-dudect" / "Cargo.toml"
  harness_manifest_path = root / "tools" / "ct-harness" / "Cargo.toml"
  dudect_lockfile_path = root / "tools" / "Cargo.lock"
  with (root / "tools" / "Cargo.toml").open("rb") as source:
    tools_manifest = tomllib.load(source)
  configured_rustflags, environment_rustflags, effective_rustflags, rustflags_source = resolved_rustflags(
    root, args.target
  )
  target_cfg = subprocess.check_output(
    ["rustc", "--print", "cfg", "--target", args.target, *effective_rustflags],
    cwd=root,
    text=True,
  )
  expected_owner_symbols = {f"ct_entry_owner_eq_{width}" for width in release_binary["formal_owner_widths"]}
  evidence_paths = [args.binary, args.binary_disassembly, args.binary_symbols, args.linker_command_log]
  if args.binary_object is not None:
    evidence_paths.append(args.binary_object)
  for path in evidence_paths:
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
    (
      line
      for line in args.linker_command_log.read_text().splitlines()
      if '"-o"' in line or '"/OUT:' in line
    ),
    "",
  )
  linker = linker_driver(linker_command)
  linker_path_text = shutil.which(linker)
  if linker_path_text is None:
    raise ValueError(f"DudeCT linker driver is not resolvable: {linker}")
  linker_path = Path(linker_path_text).resolve()
  linker_version_text = linker_version(linker_path)
  git_status = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=root, text=True).splitlines()

  report = {
    "schema_version": 4,
    "kind": "rscrypto.ct.dudect",
    "crate": "rscrypto",
    "crate_version": crate_version,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "git_dirty": bool(git_status),
    "git_status": git_status,
    "source": bundle.source_identity(root),
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "target": args.target,
    "target_triple": args.target,
    "profile": args.profile,
    "profile_settings": tools_manifest.get("profile", {}).get(args.profile, {}),
    "features": release_binary["features"],
    "default_features": release_binary["default_features"],
    "backend": release_binary["backend"],
    "ct_manifest_sha256": sha256_file(ct_manifest_path),
    "dudect_manifest_sha256": sha256_file(dudect_manifest_path),
    "harness_manifest_sha256": sha256_file(harness_manifest_path),
    "dudect_lockfile_sha256": sha256_file(dudect_lockfile_path),
    "dudect_runner_sources": dudect_runner_sources(root),
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
    "linker_version": linker_version_text,
    "binary": {
      "path": str(args.binary),
      "sha256": sha256_file(args.binary),
      "bytes": args.binary.stat().st_size,
      "owner_symbols": sorted(expected_owner_symbols),
      "owner_call_sites": owner_call_sites,
    },
    "binary_object": (
      {
        "path": str(args.binary_object),
        "sha256": sha256_file(args.binary_object),
        "bytes": args.binary_object.stat().st_size,
        "purpose": "COFF symbol, relocation, and call-site evidence for the exact LTO object linked into the timed PE",
      }
      if args.binary_object is not None
      else None
    ),
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
    "rustc_verbose": rustc_verbose(),
    "host": {
      "system": platform.system(),
      "release": platform.release(),
      "machine": platform.machine(),
      "processor": platform.processor(),
      "python": platform.python_version(),
    },
    "notes": [
      "DudeCT is empirical timing evidence, not a proof.",
      "A pass means no leakage was detected for this configuration, host, and input classification.",
      "Raw CSV retains every duration in nanoseconds, with class 0=left, 1=right and a zero-based execution sequence per case.",
    ],
  }

  return {"metadata": report, "manifest_cases": manifest_cases, "min_median_ticks": dudect_min_median_ticks(ct_manifest)}


def case_report(prepared, args):
  manifest_cases = prepared["manifest_cases"]
  seeds, results, tick_ns = parse_stdout(args.stdout)
  raw_rows = raw_csv_rows(args.csv)
  cases = dudect_case_rows(
    results,
    seeds,
    raw_rows,
    manifest_cases,
    threshold=args.threshold,
    requested_samples=args.samples,
    tick_ns=tick_ns,
    min_median_ticks=prepared["min_median_ticks"],
    filter_value=args.filter,
  )

  return {
    **prepared["metadata"],
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "threshold_abs_max_t": args.threshold,
    "requested_samples": args.samples,
    "filter": args.filter,
    "command": args.command,
    "raw_stdout": str(args.stdout),
    "raw_stdout_sha256": sha256_file(args.stdout) if args.stdout.exists() else None,
    "raw_csv": str(args.csv),
    "raw_csv_sha256": sha256_file(args.csv) if args.csv.exists() else None,
    "cases": cases,
    "case_count": len(cases),
    "failure_count": sum(1 for case in cases if case["status"] == "fail"),
    "diagnostic_failure_count": sum(1 for case in cases if case["status"] == "diagnostic-fail"),
  }


def write_report(path, report):
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(".json.tmp")
  temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def main() -> int:
  args = arguments()
  if args.prepare:
    write_report(args.out, prepare_report(args))
    return 0
  # Validate measurements before collecting the (more expensive) binary provenance.
  with (Path(__file__).resolve().parents[2] / "ct.toml").open("rb") as source:
    ct = tomllib.load(source)
  measurement = case_report(
    {"metadata": {}, "manifest_cases": manifest_dudect_cases(ct, args.target), "min_median_ticks": dudect_min_median_ticks(ct)},
    args,
  )
  report = {**prepare_report(args)["metadata"], **measurement}
  write_report(args.out, report)
  print(f"dudect report: {args.out}")
  return 1 if report["failure_count"] else 0


if __name__ == "__main__":
  raise SystemExit(main())
