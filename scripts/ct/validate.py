#!/usr/bin/env python3
"""Validate rscrypto constant-time manifest and artifact structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from toml_compat import tomllib


VALID_CLAIMS = {"ct-claimed", "ct-intended", "best-effort", "unsupported"}
VALID_HARNESS_STATUSES = {"covered", "partial", "missing", "not-applicable"}
VALID_BINSEC_TIERS = {"A", "B", "C"}
VALID_BINSEC_INPUT_KINDS = {"global", "const"}
VALID_BINSEC_POLICIES = {"required", "unsupported"}
VALID_PHYSICAL_TIMING_POLICIES = {"required", "unsupported"}
VALID_OPERATION_EVIDENCE_KINDS = {"primitive", "unit", "harness", "dudect", "binsec"}
CT_REQUIRED_PROFILE_NAMES = {"tier_a", "tier_b"}
PROVENANCE_REQUIRED_KEYS = {
  "schema_version",
  "kind",
  "crate",
  "crate_version",
  "manifest",
  "ct_manifest_sha256",
  "package_manifest_sha256",
  "harness_manifest",
  "harness_manifest_sha256",
  "harness_package",
  "harness_version",
  "harness_crate_type",
  "git_commit",
  "git_dirty",
  "git_status_entries",
  "git_status",
  "rustc",
  "rustc_verbose",
  "rustc_channel",
  "rustc_commit_hash",
  "rustc_commit_date",
  "rustc_host",
  "llvm_version",
  "tools",
  "backend",
  "target",
  "target_triple",
  "target_cfg",
  "target_cfg_features",
  "target_cpu",
  "target_features",
  "configured_rustflags",
  "environment_rustflags",
  "effective_rustflags",
  "rustflags_source",
  "linker",
  "linker_source",
  "linker_path",
  "linker_sha256",
  "linker_version",
  "linker_command",
  "linker_command_sha256",
  "link_args",
  "profile",
  "opt_level",
  "lto",
  "panic",
  "codegen_units",
  "overflow_checks",
  "debug",
  "strip",
  "features",
  "default_features",
  "dependency_lockfile",
  "dependency_lockfile_sha256",
  "workspace_lockfile",
  "workspace_lockfile_sha256",
  "component_lockfiles",
  "host_runner",
  "host_uname",
  "host_os",
  "build_target_dir",
  "artifact_dir",
  "artifacts",
  "reports",
}


def load_toml(path: Path) -> dict:
  with path.open("rb") as fh:
    return tomllib.load(fh)


def fail(errors: list[str], message: str) -> None:
  errors.append(message)


def warn(warnings: list[str], message: str) -> None:
  warnings.append(message)


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def matrix_targets(matrix: dict) -> set[str]:
  groups = matrix.get("groups", {})
  return {target for values in groups.values() for target in values}


def primitive_requires_evidence(ct: dict, primitive: dict, evidence: str) -> bool:
  if evidence == "binsec" and primitive.get("binsec") == "deferred":
    return False
  for profile_name in primitive.get("required", []):
    profile = ct.get("evidence", {}).get("profile", {}).get(profile_name, {})
    if evidence in profile.get("required", []) or evidence in profile.get("required_where_practical", []):
      return True
  return False


def claimed_targets(ct: dict) -> set[str]:
  return {target.get("name", "") for target in ct.get("target", []) if target.get("claim") in {"ct-intended", "ct-claimed"}}


def binsec_required_targets(ct: dict) -> set[str]:
  return {
    target.get("name", "")
    for target in ct.get("target", [])
    if target.get("claim") in {"ct-intended", "ct-claimed"} and target.get("binsec") == "required"
  }


def binsec_kernel_targets(ct: dict, kernel: dict) -> set[str]:
  targets = kernel.get("targets", [])
  if "*" in targets:
    return binsec_required_targets(ct)
  return set(targets)


def ct_required_primitives(ct: dict) -> list[dict]:
  rows = []
  for primitive in ct.get("primitive", []):
    if primitive.get("claim") not in {"ct-intended", "ct-claimed"}:
      continue
    if not set(primitive.get("required", [])).intersection(CT_REQUIRED_PROFILE_NAMES):
      continue
    rows.append(primitive)
  return rows


def primitive_supports_physical_timing(primitive: dict, target: str | None) -> bool:
  if target is None:
    return True
  return target not in set(primitive.get("physical_timing_unsupported_targets", []))


def is_diagnostic_dudect_case(case: dict) -> bool:
  return case.get("gate") == "diagnostic"


def required_dudect_cases(ct: dict, target: str | None = None) -> list[dict]:
  primitives = {primitive.get("id", ""): primitive for primitive in ct.get("primitive", [])}
  return [
    case
    for case in ct.get("dudect_case", [])
    if not is_diagnostic_dudect_case(case)
    and primitive_supports_physical_timing(primitives.get(case.get("primitive"), {}), target)
  ]


def generated_symbols(artifact_dir: Path) -> set[str]:
  symbols: set[str] = set()
  for path in artifact_dir.glob("*.symbols.txt"):
    for line in path.read_text().splitlines():
      if match := re.search(r"\b_?(ct_entry_[A-Za-z0-9_]+)\b", line):
        symbols.add(match.group(1))
  return symbols


def symbol_counts(path: Path) -> dict[str, int]:
  counts: dict[str, int] = {}
  for line in path.read_text().splitlines():
    if match := re.search(r"\b_?(ct_entry_[A-Za-z0-9_]+)\b", line):
      symbol = match.group(1)
      counts[symbol] = counts.get(symbol, 0) + 1
  return counts


def recorded_hashes(path: Path) -> dict[str, str]:
  recorded: dict[str, str] = {}
  for line in path.read_text().splitlines():
    parts = line.split()
    if len(parts) == 2:
      recorded[parts[1]] = parts[0]
  return recorded


def dudect_registered_benches(root: Path, errors: list[str]) -> set[str]:
  path = root / "tools" / "ct-dudect" / "src" / "main.rs"
  if not path.exists():
    fail(errors, f"DudeCT source missing: {path}")
    return set()
  text = path.read_text()
  start = text.find("ctbench_main_with_seeds!(")
  if start < 0:
    fail(errors, "DudeCT source has no ctbench_main_with_seeds! registry")
    return set()
  body = text[start:]
  return set(re.findall(r"\(\s*([A-Za-z0-9_]+),\s*Some\(", body))


def compiler_public_api_snapshot(root: Path, prefixes: tuple[str, ...], errors: list[str]) -> tuple[int, str] | None:
  """Return the audited public security API count and digest from rustdoc JSON."""
  target_dir = root / "target" / "ct-api-inventory"
  command = [
    "cargo",
    "rustdoc",
    "--quiet",
    "--lib",
    "--all-features",
    "--target-dir",
    str(target_dir),
    "--",
    "-Z",
    "unstable-options",
    "--output-format",
    "json",
  ]
  completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
  if completed.returncode != 0:
    detail = completed.stderr.strip().splitlines()
    suffix = f": {detail[-1]}" if detail else ""
    fail(errors, f"compiler public-API inventory failed{suffix}")
    return None

  rustdoc_path = target_dir / "doc" / "rscrypto.json"
  if not rustdoc_path.is_file():
    fail(errors, f"compiler public-API inventory missing {rustdoc_path}")
    return None
  try:
    rustdoc = json.loads(rustdoc_path.read_text())
  except json.JSONDecodeError as exc:
    fail(errors, f"invalid compiler public-API inventory JSON: {exc}")
    return None

  index = rustdoc.get("index", {})
  paths = rustdoc.get("paths", {})
  rows: set[str] = set()
  for item_id, summary in paths.items():
    if summary.get("crate_id") != 0:
      continue
    path = "::".join(summary.get("path", []))
    if not path.startswith(prefixes):
      continue
    item = index.get(item_id)
    if not isinstance(item, dict) or item.get("visibility") != "public":
      continue
    kind = summary.get("kind")
    if kind == "function":
      rows.add(path)
      continue

    inner = item.get("inner", {})
    if kind == "trait":
      for child_id in inner.get("trait", {}).get("items", []):
        child = index.get(str(child_id), {})
        if "function" in child.get("inner", {}):
          rows.add(f"{path}::{child.get('name')}")
      continue

    type_inner = inner.get(kind, {})
    if not isinstance(type_inner, dict):
      continue
    for impl_id in type_inner.get("impls", []):
      implementation = index.get(str(impl_id), {}).get("inner", {}).get("impl", {})
      trait = implementation.get("trait")
      trait_name = trait.get("path") if isinstance(trait, dict) else None
      for child_id in implementation.get("items", []):
        child = index.get(str(child_id), {})
        if "function" not in child.get("inner", {}):
          continue
        span = child.get("span") or {}
        if not str(span.get("filename", "")).startswith("src/"):
          continue
        name = child.get("name")
        if trait_name:
          rows.add(f"<{path} as {trait_name}>::{name}")
        elif child.get("visibility") == "public":
          rows.add(f"{path}::{name}")

  encoded = ("\n".join(sorted(rows)) + "\n").encode()
  return len(rows), hashlib.sha256(encoded).hexdigest()


def validate_manifest(root: Path, errors: list[str], warnings: list[str]) -> dict:
  ct_path = root / "ct.toml"
  matrix_path = root / ".config" / "target-matrix.json"
  if not ct_path.exists():
    fail(errors, "ct.toml is missing")
    return {}
  if not matrix_path.exists():
    fail(errors, ".config/target-matrix.json is missing")
    return {}

  ct = load_toml(ct_path)
  matrix = json.loads(matrix_path.read_text())

  expected_targets = matrix_targets(matrix)
  actual_targets = {target.get("name", "") for target in ct.get("target", [])}
  missing = sorted(expected_targets - actual_targets)
  extra = sorted(actual_targets - expected_targets)
  if missing:
    fail(errors, f"ct.toml missing target(s): {', '.join(missing)}")
  if extra:
    fail(errors, f"ct.toml has target(s) not in target matrix: {', '.join(extra)}")

  for target in ct.get("target", []):
    name = target.get("name", "<unnamed>")
    claim = target.get("claim")
    if claim not in VALID_CLAIMS:
      fail(errors, f"target {name} has invalid claim {claim!r}")
    if claim == "ct-claimed":
      fail(errors, f"target {name} is ct-claimed before release evidence gates exist")
    if target.get("backend") != "llvm":
      fail(errors, f"target {name} backend must be llvm during phase 1")
    for required in ("group", "linker", "physical_timing", "ci"):
      if not target.get(required):
        fail(errors, f"target {name} missing {required}")
    if target.get("physical_timing") not in VALID_PHYSICAL_TIMING_POLICIES:
      fail(errors, f"target {name} has invalid physical_timing policy {target.get('physical_timing')!r}")
    if target.get("physical_timing") == "unsupported" and not target.get("physical_timing_reason"):
      fail(errors, f"target {name} physical_timing unsupported requires physical_timing_reason")
    if target.get("binsec") not in VALID_BINSEC_POLICIES:
      fail(errors, f"target {name} has invalid binsec policy {target.get('binsec')!r}")
    if target.get("binsec") == "unsupported" and not target.get("binsec_reason"):
      fail(errors, f"target {name} binsec unsupported requires binsec_reason")
    if target.get("binsec") == "required" and "linux" not in name:
      fail(errors, f"target {name} requires BINSEC but is not a Linux target")

  harness_sections = ct.get("harness", [])
  all_harness_symbols = {symbol for harness in harness_sections for symbol in harness.get("symbols", [])}
  if not harness_sections:
    fail(errors, "ct.toml has no [[harness]] sections")

  release_binary = ct.get("equality_evidence", {}).get("release_binary", {})
  expected_release_binary = {
    "name": "rscrypto-ct-evidence",
    "kind": "executable",
    "profile": "release",
    "backend": "llvm",
    "features": ["std", "full", "parallel", "diag"],
    "default_features": False,
  }
  for field, expected in expected_release_binary.items():
    if release_binary.get(field) != expected:
      fail(errors, f"equality release binary {field} expected {expected!r}, got {release_binary.get(field)!r}")

  harness_manifest_path = root / "tools" / "ct-harness" / "Cargo.toml"
  dudect_manifest_path = root / "tools" / "ct-dudect" / "Cargo.toml"
  with harness_manifest_path.open("rb") as source:
    harness_manifest = tomllib.load(source)
  with dudect_manifest_path.open("rb") as source:
    dudect_manifest = tomllib.load(source)
  harness_dependency = harness_manifest.get("dependencies", {}).get("rscrypto", {})
  dudect_dependency = dudect_manifest.get("dependencies", {}).get("rscrypto", {})
  for label, dependency in (("CT harness", harness_dependency), ("DudeCT", dudect_dependency)):
    if dependency.get("features") != release_binary.get("features"):
      fail(errors, f"{label} rscrypto features do not match the equality release binary")
    if dependency.get("default-features") != release_binary.get("default_features"):
      fail(errors, f"{label} rscrypto default features do not match the equality release binary")
  harness_bins = [row for row in harness_manifest.get("bin", []) if row.get("name") == release_binary.get("name")]
  if len(harness_bins) != 1:
    fail(errors, "CT harness must define exactly one equality release binary")
  if harness_manifest.get("profile", {}).get("release") != dudect_manifest.get("profile", {}).get("release"):
    fail(errors, "CT harness and DudeCT release profiles must match")
  owner_widths = release_binary.get("owner_widths", [])
  owner_symbols = release_binary.get("owner_symbols", [])
  public_len_symbols = release_binary.get("public_len_symbols", [])
  if (
    not isinstance(owner_widths, list)
    or not owner_widths
    or any(isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in owner_widths)
    or owner_widths != sorted(set(owner_widths))
  ):
    fail(errors, f"equality release binary owner_widths must be positive, unique, and ordered: {owner_widths!r}")
  required_owner_widths = {16, 32, 48, 64}
  if not required_owner_widths.issubset(owner_widths):
    fail(errors, "equality release binary must cover the required 16-, 32-, 48-, and 64-byte owners")
  expected_owner_symbols = [f"ct_entry_owner_eq_{width}" for width in owner_widths]
  if owner_symbols != expected_owner_symbols:
    fail(errors, "equality release binary owner symbols do not match owner widths")
  equality_symbols = owner_symbols + public_len_symbols
  if not equality_symbols or len(equality_symbols) != len(set(equality_symbols)):
    fail(errors, "equality release binary symbols must be non-empty and unique")
  unknown_equality_symbols = sorted(set(equality_symbols) - all_harness_symbols)
  if unknown_equality_symbols:
    fail(errors, f"equality release binary references unknown harness symbol(s): {', '.join(unknown_equality_symbols)}")
  if release_binary.get("formal_owner_widths") != [16, 32, 48, 64]:
    fail(errors, "equality release binary formal_owner_widths must cover 16, 32, 48, and 64 bytes")
  for field in ("formal_limitation", "downstream_limitation"):
    if not isinstance(release_binary.get(field), str) or not release_binary.get(field, "").strip():
      fail(errors, f"equality release binary requires {field}")

  owner_primitives = [row for row in ct.get("primitive", []) if row.get("id") == "owner_equality.fixed"]
  if len(owner_primitives) != 1:
    fail(errors, "ct.toml must contain exactly one owner_equality.fixed primitive")
  else:
    owner_primitive = owner_primitives[0]
    if owner_primitive.get("release_owner_widths") != owner_widths:
      fail(errors, "owner_equality.fixed release widths do not match the equality release binary")
    if owner_primitive.get("harness", {}).get("symbols") != owner_symbols:
      fail(errors, "owner_equality.fixed harness symbols do not match the equality release binary")

  evidence_main = root / "tools" / "ct-harness" / "src" / "main.rs"
  retain_match = re.search(r"retain!\((.*?)\n  \);", evidence_main.read_text(), re.DOTALL)
  if retain_match is None:
    fail(errors, "CT harness lacks the final equality root retention list")
  else:
    retained_symbols = re.findall(r"\bct_entry_[A-Za-z0-9_]+\b", retain_match.group(1))
    if retained_symbols != equality_symbols:
      fail(errors, "CT harness equality retention list does not exactly match ct.toml")

  primitive_ids: set[str] = set()
  for primitive in ct.get("primitive", []):
    primitive_id = primitive.get("id", "<unnamed>")
    if primitive_id in primitive_ids:
      fail(errors, f"duplicate primitive id {primitive_id}")
    primitive_ids.add(primitive_id)

    claim = primitive.get("claim")
    if claim not in VALID_CLAIMS:
      fail(errors, f"primitive {primitive_id} has invalid claim {claim!r}")
    if claim == "ct-claimed":
      fail(errors, f"primitive {primitive_id} is ct-claimed before release evidence gates exist")

    unsupported_timing_targets = primitive.get("physical_timing_unsupported_targets", [])
    if unsupported_timing_targets and (
      not isinstance(unsupported_timing_targets, list)
      or any(not isinstance(target, str) or not target for target in unsupported_timing_targets)
    ):
      fail(errors, f"primitive {primitive_id} physical_timing_unsupported_targets must be a list of target triples")
      unsupported_timing_targets = []
    if unsupported_timing_targets and not primitive.get("physical_timing_unsupported_reason"):
      fail(errors, f"primitive {primitive_id} physical_timing_unsupported_targets requires physical_timing_unsupported_reason")
    for unsupported_target in unsupported_timing_targets:
      if unsupported_target not in actual_targets:
        fail(errors, f"primitive {primitive_id} references unknown physical timing unsupported target {unsupported_target!r}")

    harness = primitive.get("harness")
    if not isinstance(harness, dict):
      fail(errors, f"primitive {primitive_id} missing [primitive.harness]")
      continue

    status = harness.get("status")
    if status not in VALID_HARNESS_STATUSES:
      fail(errors, f"primitive {primitive_id} has invalid harness status {status!r}")
      continue

    symbols = harness.get("symbols", [])
    gap = harness.get("gap", "")
    if status == "covered" and not symbols:
      fail(errors, f"primitive {primitive_id} marked covered without symbols")
    if status in {"partial", "missing", "not-applicable"} and not gap:
      fail(errors, f"primitive {primitive_id} status {status} requires an explicit gap")
    if status == "missing" and symbols:
      fail(errors, f"primitive {primitive_id} marked missing but lists symbols")
    if status == "not-applicable" and claim == "ct-intended":
      warn(warnings, f"primitive {primitive_id} is ct-intended but harness is not-applicable")

    for symbol in symbols:
      if symbol not in all_harness_symbols:
        fail(errors, f"primitive {primitive_id} references unknown harness symbol {symbol}")

  for harness in harness_sections:
    for covered in harness.get("covers", []):
      if covered not in primitive_ids:
        fail(errors, f"harness {harness.get('name', '<unnamed>')} covers unknown primitive {covered}")

  for index, rule in enumerate(ct.get("asm_public_operand", [])):
    required_fields = {"primitive", "root", "symbol", "kind", "max_count", "source", "rationale"}
    if set(rule) != required_fields:
      fail(errors, f"asm_public_operand[{index}] has incomplete or unknown fields")
      continue
    primitive_id = rule.get("primitive")
    if primitive_id not in primitive_ids:
      fail(errors, f"asm_public_operand[{index}] references unknown primitive {primitive_id!r}")
      continue
    primitive = next(row for row in ct.get("primitive", []) if row.get("id") == primitive_id)
    if rule.get("root") not in primitive.get("harness", {}).get("symbols", []):
      fail(errors, f"asm_public_operand[{index}] root is not owned by primitive {primitive_id}")
    if rule.get("kind") not in {"variable_latency_division", "variable_latency_multiply"}:
      fail(errors, f"asm_public_operand[{index}] kind is not a public-operand-classifiable instruction")
    max_count = rule.get("max_count")
    if isinstance(max_count, bool) or not isinstance(max_count, int) or max_count <= 0:
      fail(errors, f"asm_public_operand[{index}] max_count must be a positive integer")
    source, separator, line = str(rule.get("source", "")).rpartition(":")
    if not separator or not line.isdigit() or not (root / source).is_file():
      fail(errors, f"asm_public_operand[{index}] source must identify an existing source line")
    if not isinstance(rule.get("rationale"), str) or not rule.get("rationale", "").strip():
      fail(errors, f"asm_public_operand[{index}] requires a rationale")

  dudect_names: set[str] = set()
  dudect_by_name: dict[str, dict] = {}
  for case in ct.get("dudect_case", []):
    name = case.get("name", "<unnamed>")
    if name in dudect_names:
      fail(errors, f"duplicate DudeCT case name {name}")
    dudect_names.add(name)
    dudect_by_name[name] = case
    primitive = case.get("primitive")
    if primitive not in primitive_ids:
      fail(errors, f"DudeCT case {name} references unknown primitive {primitive!r}")
    if not case.get("filter"):
      fail(errors, f"DudeCT case {name} missing filter")
    if case.get("gate") == "diagnostic" and not (case.get("reason") or case.get("notes")):
      fail(errors, f"diagnostic DudeCT case {name} requires reason or notes")
    timeout_seconds = case.get("timeout_seconds")
    if timeout_seconds is not None:
      if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        fail(errors, f"DudeCT case {name} timeout_seconds must be a positive integer")
      timeout_reason = case.get("timeout_reason")
      if not isinstance(timeout_reason, str) or not timeout_reason.strip():
        fail(errors, f"DudeCT case {name} timeout_seconds requires timeout_reason")
    elif case.get("timeout_reason") is not None:
      fail(errors, f"DudeCT case {name} timeout_reason requires timeout_seconds")

  registered_dudect = dudect_registered_benches(root, errors)
  if registered_dudect:
    missing_from_manifest = sorted(registered_dudect - dudect_names)
    missing_from_runner = sorted(dudect_names - registered_dudect)
    if missing_from_manifest:
      fail(errors, f"DudeCT runner has unmanifested bench(es): {', '.join(missing_from_manifest)}")
    if missing_from_runner:
      fail(errors, f"ct.toml has DudeCT case(s) missing from runner: {', '.join(missing_from_runner)}")
    for case in ct.get("dudect_case", []):
      name = case.get("name", "<unnamed>")
      filter_value = str(case.get("filter", ""))
      matches = sorted(bench for bench in registered_dudect if filter_value in bench)
      if matches != [name]:
        matched = ", ".join(matches) if matches else "<none>"
        fail(errors, f"DudeCT case {name} filter {filter_value!r} must match exactly itself; matched: {matched}")

  unit_ids: set[str] = set()
  for unit in ct.get("evidence_unit", []):
    unit_id = unit.get("id", "<unnamed>")
    if unit_id in unit_ids:
      fail(errors, f"duplicate evidence unit id {unit_id}")
    unit_ids.add(unit_id)

    primitive = unit.get("primitive")
    if primitive not in primitive_ids:
      fail(errors, f"evidence unit {unit_id} references unknown primitive {primitive!r}")
    if not unit.get("variant"):
      fail(errors, f"evidence unit {unit_id} missing variant")

    dudect = unit.get("dudect", [])
    if not isinstance(dudect, list):
      fail(errors, f"evidence unit {unit_id} dudect must be a list")
      dudect = []
    for case_name in dudect:
      case = dudect_by_name.get(case_name)
      if case is None:
        fail(errors, f"evidence unit {unit_id} references unknown DudeCT case {case_name!r}")
        continue
      if case.get("primitive") != primitive:
        fail(errors, f"evidence unit {unit_id} references DudeCT case {case_name} for different primitive {case.get('primitive')!r}")
      if case.get("gate") == "diagnostic":
        fail(errors, f"evidence unit {unit_id} cannot rely on diagnostic DudeCT case {case_name}")
    binsec = unit.get("binsec", [])
    if binsec and not isinstance(binsec, list):
      fail(errors, f"evidence unit {unit_id} binsec must be a list")

  binsec_ids: set[str] = set()
  binsec_by_id: dict[str, dict] = {}
  for kernel in ct.get("binsec_kernel", []):
    kernel_id = kernel.get("id", "<unnamed>")
    if kernel_id in binsec_ids:
      fail(errors, f"duplicate BINSEC kernel id {kernel_id}")
    binsec_ids.add(kernel_id)
    binsec_by_id[kernel_id] = kernel

    primitive = kernel.get("primitive")
    if primitive not in primitive_ids:
      fail(errors, f"BINSEC kernel {kernel_id} references unknown primitive {primitive!r}")

    symbol = kernel.get("symbol")
    if not isinstance(symbol, str) or not symbol.startswith("ct_binsec_"):
      fail(errors, f"BINSEC kernel {kernel_id} has invalid symbol {symbol!r}")

    tier = kernel.get("tier")
    if tier not in VALID_BINSEC_TIERS:
      fail(errors, f"BINSEC kernel {kernel_id} has invalid tier {tier!r}")

    if not isinstance(kernel.get("required"), bool):
      fail(errors, f"BINSEC kernel {kernel_id} required must be boolean")

    claim = kernel.get("claim")
    if claim not in VALID_CLAIMS:
      fail(errors, f"BINSEC kernel {kernel_id} has invalid claim {claim!r}")
    if claim == "ct-claimed":
      fail(errors, f"BINSEC kernel {kernel_id} is ct-claimed before release evidence gates exist")

    targets = kernel.get("targets")
    if not isinstance(targets, list) or not targets:
      fail(errors, f"BINSEC kernel {kernel_id} targets must be a non-empty list")
    else:
      for target in targets:
        if target == "*":
          continue
        if target not in actual_targets:
          fail(errors, f"BINSEC kernel {kernel_id} references unknown target {target!r}")

    secrets = kernel.get("secrets")
    if not isinstance(secrets, list) or not secrets:
      fail(errors, f"BINSEC kernel {kernel_id} secrets must be a non-empty list")
    else:
      for secret in secrets:
        name = secret.get("name") if isinstance(secret, dict) else None
        kind = secret.get("kind") if isinstance(secret, dict) else None
        bytes_len = secret.get("bytes") if isinstance(secret, dict) else None
        if not isinstance(name, str) or not name:
          fail(errors, f"BINSEC kernel {kernel_id} has secret with invalid name")
        if kind not in VALID_BINSEC_INPUT_KINDS:
          fail(errors, f"BINSEC kernel {kernel_id} secret {name!r} has invalid kind {kind!r}")
        if kind == "global" and (not isinstance(bytes_len, int) or bytes_len <= 0):
          fail(errors, f"BINSEC kernel {kernel_id} secret {name!r} bytes must be positive")

    public = kernel.get("public", [])
    if not isinstance(public, list):
      fail(errors, f"BINSEC kernel {kernel_id} public must be a list")
    else:
      for item in public:
        is_table = isinstance(item, dict)
        name = item.get("name") if is_table else None
        kind = item.get("kind") if is_table else None
        if not isinstance(name, str) or not name:
          fail(errors, f"BINSEC kernel {kernel_id} has public input with invalid name")
        if kind not in VALID_BINSEC_INPUT_KINDS:
          fail(errors, f"BINSEC kernel {kernel_id} public input {name!r} has invalid kind {kind!r}")
        if kind == "const" and is_table and "value" not in item:
          fail(errors, f"BINSEC kernel {kernel_id} public const {name!r} missing value")

    assumptions = kernel.get("assumptions", [])
    if not isinstance(assumptions, list) or any(not isinstance(assumption, str) for assumption in assumptions):
      fail(errors, f"BINSEC kernel {kernel_id} assumptions must be a list of strings")

  for unit in ct.get("evidence_unit", []):
    unit_id = unit.get("id", "<unnamed>")
    primitive = unit.get("primitive")
    for kernel_id in unit.get("binsec", []):
      kernel = binsec_by_id.get(kernel_id)
      if kernel is None:
        fail(errors, f"evidence unit {unit_id} references unknown BINSEC kernel {kernel_id!r}")
        continue
      if kernel.get("primitive") != primitive:
        fail(errors, f"evidence unit {unit_id} references BINSEC kernel {kernel_id} for different primitive {kernel.get('primitive')!r}")
      if not kernel.get("required", False):
        fail(errors, f"evidence unit {unit_id} cannot rely on non-required BINSEC kernel {kernel_id}")

  inventory = ct.get("operation_inventory")
  if not isinstance(inventory, dict):
    fail(errors, "ct.toml missing [operation_inventory]")
  else:
    if inventory.get("schema_version") != 1:
      fail(errors, f"operation_inventory schema_version expected 1, got {inventory.get('schema_version')!r}")
    if inventory.get("authority") != "ct.toml":
      fail(errors, "operation_inventory authority must be ct.toml")
    required_scope = {"production", "feature-gated", "target-gated", "trait-defaults", "escape-hatches"}
    actual_scope = set(inventory.get("scope", []))
    if actual_scope != required_scope:
      fail(errors, f"operation_inventory scope must be exactly {', '.join(sorted(required_scope))}")
    prefixes = inventory.get("compiler_api_prefixes")
    if not isinstance(prefixes, list) or not prefixes or any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
      fail(errors, "operation_inventory compiler_api_prefixes must be a non-empty list of strings")
    elif len(prefixes) != len(set(prefixes)):
      fail(errors, "operation_inventory compiler_api_prefixes must not contain duplicates")
    else:
      snapshot = compiler_public_api_snapshot(root, tuple(prefixes), errors)
      if snapshot is not None:
        actual_count, actual_sha256 = snapshot
        expected_count = inventory.get("compiler_api_item_count")
        expected_sha256 = inventory.get("compiler_api_sha256")
        if actual_count != expected_count:
          fail(errors, f"compiler public-API inventory expected {expected_count} items, got {actual_count}")
        if actual_sha256 != expected_sha256:
          fail(
            errors,
            "compiler public-API inventory digest changed; audit and classify the public surface before updating ct.toml",
          )

  operation_ids: set[str] = set()
  operation_api: dict[str, str] = {}
  referenced_primitives: set[str] = set()
  known_evidence = {
    "primitive": primitive_ids,
    "unit": unit_ids,
    "harness": all_harness_symbols,
    "dudect": dudect_names,
    "binsec": binsec_ids,
  }
  for operation in ct.get("operation", []):
    operation_id = operation.get("id", "<unnamed>")
    if operation_id in operation_ids:
      fail(errors, f"duplicate operation id {operation_id}")
    operation_ids.add(operation_id)

    claim = operation.get("claim")
    if claim not in VALID_CLAIMS:
      fail(errors, f"operation {operation_id} has invalid claim {claim!r}")
    if claim == "ct-claimed":
      fail(errors, f"operation {operation_id} is ct-claimed before release evidence gates exist")

    for field in ("api", "features", "targets", "variable_time_components", "permitted_leakage"):
      values = operation.get(field)
      if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values):
        fail(errors, f"operation {operation_id} {field} must be a non-empty list of strings")
    for field in ("secret_inputs", "public_inputs"):
      values = operation.get(field)
      if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        fail(errors, f"operation {operation_id} {field} must be a list of strings")

    for api in operation.get("api", []):
      owner = operation_api.get(api)
      if owner is not None:
        fail(errors, f"operation API family {api!r} is listed by both {owner} and {operation_id}")
      operation_api[api] = operation_id

    evidence = operation.get("evidence", [])
    limitation = operation.get("limitation")
    if not evidence and not (isinstance(limitation, str) and limitation.strip()):
      fail(errors, f"operation {operation_id} requires linked evidence or an explicit limitation")
    if evidence and (not isinstance(evidence, list) or any(not isinstance(value, str) for value in evidence)):
      fail(errors, f"operation {operation_id} evidence must be a list of kind:id strings")
      evidence = []
    for reference in evidence:
      kind, separator, evidence_id = reference.partition(":")
      if not separator or kind not in VALID_OPERATION_EVIDENCE_KINDS or not evidence_id:
        fail(errors, f"operation {operation_id} has invalid evidence reference {reference!r}")
        continue
      if evidence_id not in known_evidence[kind]:
        fail(errors, f"operation {operation_id} references unknown {kind} evidence {evidence_id!r}")
      if kind == "primitive":
        referenced_primitives.add(evidence_id)

  if not operation_ids:
    fail(errors, "ct.toml has no [[operation]] security inventory")
  missing_operation_primitives = sorted(primitive_ids - referenced_primitives)
  if missing_operation_primitives:
    fail(errors, f"security operation inventory does not map primitive(s): {', '.join(missing_operation_primitives)}")

  comparison_ids: set[str] = set()
  expected_public_len_calls: dict[str, int] = {}
  for comparison in ct.get("public_len_comparison", []):
    comparison_id = comparison.get("id", "<unnamed>")
    if comparison_id in comparison_ids:
      fail(errors, f"duplicate public-length comparison id {comparison_id}")
    comparison_ids.add(comparison_id)

    source = comparison.get("source")
    if not isinstance(source, str) or not source.startswith("src/") or not (root / source).is_file():
      fail(errors, f"public-length comparison {comparison_id} has invalid source {source!r}")
      continue
    call_count = comparison.get("call_count")
    if isinstance(call_count, bool) or not isinstance(call_count, int) or call_count <= 0:
      fail(errors, f"public-length comparison {comparison_id} call_count must be a positive integer")
      continue
    if source in expected_public_len_calls:
      fail(errors, f"public-length comparison source {source} is listed more than once")
    expected_public_len_calls[source] = call_count

    operations = comparison.get("operations")
    if not isinstance(operations, list) or not operations:
      fail(errors, f"public-length comparison {comparison_id} operations must be a non-empty list")
    else:
      for operation_id in operations:
        if operation_id not in operation_ids:
          fail(errors, f"public-length comparison {comparison_id} references unknown operation {operation_id!r}")
    for field in ("public_length", "secret_contents", "tests"):
      if not isinstance(comparison.get(field), str) or not comparison.get(field, "").strip():
        fail(errors, f"public-length comparison {comparison_id} requires {field}")
    evidence_symbols = comparison.get("evidence_symbols")
    if not isinstance(evidence_symbols, list) or not evidence_symbols:
      fail(errors, f"public-length comparison {comparison_id} requires release evidence symbols")
      evidence_symbols = []
    for symbol in evidence_symbols:
      if symbol not in public_len_symbols:
        fail(errors, f"public-length comparison {comparison_id} references unknown release symbol {symbol!r}")
    evidenced_call_count = comparison.get("evidenced_call_count")
    limited_call_count = comparison.get("limited_call_count")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (evidenced_call_count, limited_call_count)):
      fail(errors, f"public-length comparison {comparison_id} evidence counts must be non-negative integers")
    elif evidenced_call_count + limited_call_count != call_count:
      fail(errors, f"public-length comparison {comparison_id} evidence counts do not equal call_count")
    limitation = comparison.get("limitation")
    if isinstance(limited_call_count, int) and limited_call_count > 0 and (
      not isinstance(limitation, str) or not limitation.strip()
    ):
      fail(errors, f"public-length comparison {comparison_id} limited calls require an explicit limitation")

  mapped_public_len_symbols = {
    symbol for comparison in ct.get("public_len_comparison", []) for symbol in comparison.get("evidence_symbols", [])
  }
  if mapped_public_len_symbols != set(public_len_symbols):
    fail(errors, "public-length comparison evidence does not exactly cover the release binary public-length symbols")

  actual_public_len_calls: dict[str, int] = {}
  for source_path in (root / "src").rglob("*.rs"):
    count = source_path.read_text().count("ct::public_len_eq(")
    if count:
      actual_public_len_calls[source_path.relative_to(root).as_posix()] = count
  if actual_public_len_calls != expected_public_len_calls:
    missing_sources = sorted(set(actual_public_len_calls) - set(expected_public_len_calls))
    stale_sources = sorted(set(expected_public_len_calls) - set(actual_public_len_calls))
    wrong_counts = sorted(
      source
      for source in set(actual_public_len_calls).intersection(expected_public_len_calls)
      if actual_public_len_calls[source] != expected_public_len_calls[source]
    )
    if missing_sources:
      fail(errors, f"unclassified public_len_eq caller source(s): {', '.join(missing_sources)}")
    if stale_sources:
      fail(errors, f"stale public_len_eq inventory source(s): {', '.join(stale_sources)}")
    for source in wrong_counts:
      fail(
        errors,
        f"public_len_eq call count for {source} expected {expected_public_len_calls[source]}, got {actual_public_len_calls[source]}",
      )

  return ct


def validate_strict_coverage(ct: dict, errors: list[str], target: str | None = None) -> None:
  targets = binsec_required_targets(ct)
  dudect_primitives = {case.get("primitive") for case in required_dudect_cases(ct, target)}
  dudect_case_names = {case.get("name") for case in required_dudect_cases(ct, target)}
  evidence_units_by_primitive: dict[str, list[dict]] = {}
  for unit in ct.get("evidence_unit", []):
    primitive = unit.get("primitive")
    if isinstance(primitive, str):
      evidence_units_by_primitive.setdefault(primitive, []).append(unit)

  required_binsec_targets: dict[str, set[str]] = {}
  for kernel in ct.get("binsec_kernel", []):
    if not kernel.get("required", False):
      continue
    primitive = kernel.get("primitive")
    if not isinstance(primitive, str):
      continue
    required_binsec_targets.setdefault(primitive, set()).update(binsec_kernel_targets(ct, kernel))

  for primitive in ct_required_primitives(ct):
    primitive_id = primitive.get("id", "<unnamed>")
    if not primitive_supports_physical_timing(primitive, target):
      continue
    if primitive_requires_evidence(ct, primitive, "dudect") and primitive_id not in dudect_primitives:
      fail(errors, f"primitive {primitive_id} requires at least one non-diagnostic DudeCT case")
    variants = primitive.get("variants", [])
    if primitive_requires_evidence(ct, primitive, "dudect") and variants:
      units = evidence_units_by_primitive.get(primitive_id, [])
      covered_variants = {unit.get("variant") for unit in units if unit.get("dudect")}
      missing_variants = sorted(set(variants) - covered_variants)
      extra_variants = sorted(covered_variants - set(variants))
      if missing_variants:
        fail(errors, f"primitive {primitive_id} requires DudeCT evidence for variant(s): {', '.join(missing_variants)}")
      if extra_variants:
        fail(errors, f"primitive {primitive_id} has evidence unit(s) for unknown variant(s): {', '.join(extra_variants)}")
      for unit in units:
        unit_id = unit.get("id", "<unnamed>")
        for case_name in unit.get("dudect", []):
          if case_name not in dudect_case_names:
            fail(errors, f"primitive {primitive_id} evidence unit {unit_id} lacks a non-diagnostic DudeCT case {case_name!r}")
    if primitive_requires_evidence(ct, primitive, "binsec"):
      covered_targets = required_binsec_targets.get(primitive_id, set())
      missing_targets = sorted(targets - covered_targets)
      if missing_targets:
        fail(
          errors,
          f"primitive {primitive_id} requires BINSEC but lacks required kernels for target(s): {', '.join(missing_targets)}",
        )
      variants = primitive.get("variants", [])
      if variants:
        units = evidence_units_by_primitive.get(primitive_id, [])
        units_by_variant = {unit.get("variant"): unit for unit in units}
        for variant in sorted(variants):
          unit = units_by_variant.get(variant)
          if unit is None:
            fail(errors, f"primitive {primitive_id} requires BINSEC evidence unit for variant {variant}")
            continue
          if not unit.get("binsec"):
            fail(errors, f"primitive {primitive_id} variant {variant} requires BINSEC kernel evidence")


def validate_artifacts(root: Path, target: str, profile: str, ct: dict, errors: list[str], warnings: list[str]) -> None:
  out_dir = root / "target" / "ct" / target / profile
  artifact_dir = out_dir / "artifacts"
  provenance_path = out_dir / "provenance.json"
  evidence_path = out_dir / "evidence-index.json"
  heuristics_path = out_dir / "asm-heuristics.json"
  heuristics_summary_path = out_dir / "asm-heuristics.md"
  hashes_path = out_dir / "artifact-hashes.txt"

  if not out_dir.exists():
    fail(errors, f"artifact output directory missing: {out_dir}")
    return
  if not artifact_dir.exists():
    fail(errors, f"artifact directory missing: {artifact_dir}")
    return
  if not provenance_path.exists():
    fail(errors, f"provenance missing: {provenance_path}")
  if not evidence_path.exists():
    fail(errors, f"evidence index missing: {evidence_path}")
  if not heuristics_path.exists():
    fail(errors, f"asm heuristics report missing: {heuristics_path}")
  if not heuristics_summary_path.exists():
    fail(errors, f"asm heuristics summary missing: {heuristics_summary_path}")
  if not hashes_path.exists():
    fail(errors, f"artifact hashes missing: {hashes_path}")

  files = list(artifact_dir.iterdir())
  required_suffix_groups = [
    (".ll",),
    (".s",),
    (".o", ".obj"),
    (".o.disasm.txt", ".obj.disasm.txt"),
    (".o.symbols.txt", ".obj.symbols.txt"),
    (".o.size.txt", ".obj.size.txt"),
    (".binary.disasm.txt",),
    (".binary.symbols.txt",),
    (".binary.size.txt",),
  ]
  for suffixes in required_suffix_groups:
    if not any(path.name.endswith(suffixes) for path in files):
      fail(errors, f"artifact suffix {' or '.join(suffixes)} missing in {artifact_dir}")

  provenance = {}
  if provenance_path.exists():
    try:
      provenance = json.loads(provenance_path.read_text())
    except json.JSONDecodeError as exc:
      fail(errors, f"invalid provenance JSON: {exc}")
    else:
      expected = {
        "crate": "rscrypto",
        "manifest": "ct.toml",
        "backend": "llvm",
        "target": target,
        "profile": profile,
      }
      for key, value in expected.items():
        if provenance.get(key) != value:
          fail(errors, f"provenance {key} expected {value!r}, got {provenance.get(key)!r}")
      for key in sorted(PROVENANCE_REQUIRED_KEYS):
        if key not in provenance:
          fail(errors, f"provenance missing {key}")
      if provenance.get("schema_version") != 2:
        fail(errors, f"provenance schema_version expected 2, got {provenance.get('schema_version')!r}")
      if provenance.get("kind") != "rscrypto.ct.provenance":
        fail(errors, f"provenance kind expected 'rscrypto.ct.provenance', got {provenance.get('kind')!r}")
      if provenance.get("target_triple") != target:
        fail(errors, f"provenance target_triple expected {target!r}, got {provenance.get('target_triple')!r}")
      if provenance.get("rustc_channel") not in {"stable", "beta", "nightly", "unknown"}:
        fail(errors, f"provenance rustc_channel invalid: {provenance.get('rustc_channel')!r}")
      if not isinstance(provenance.get("features"), list) or not provenance.get("features"):
        fail(errors, "provenance features must be a non-empty list")
      if not isinstance(provenance.get("effective_rustflags"), list):
        fail(errors, "provenance effective_rustflags must be a list")
      if not isinstance(provenance.get("rustflags_source"), str) or not provenance.get("rustflags_source"):
        fail(errors, "provenance rustflags_source must be a non-empty string")
      if not isinstance(provenance.get("target_cfg_features"), list):
        fail(errors, "provenance target_cfg_features must be a list")
      if not isinstance(provenance.get("target_features"), list):
        fail(errors, "provenance target_features must be a list")
      if not isinstance(provenance.get("link_args"), list):
        fail(errors, "provenance link_args must be a list")
      for field in ("linker", "linker_source", "linker_path", "linker_sha256", "linker_version"):
        if not isinstance(provenance.get(field), str) or not provenance.get(field):
          fail(errors, f"provenance {field} must be a non-empty string")
      if provenance.get("linker_command") != "artifacts/linker-command.txt":
        fail(errors, "provenance linker_command must name the archived linker command")
      linker_log = artifact_dir / "linker-command.txt"
      if not linker_log.is_file() or provenance.get("linker_command_sha256") != sha256_file(linker_log):
        fail(errors, "provenance linker_command_sha256 mismatch")
      release_binary = ct.get("equality_evidence", {}).get("release_binary", {})
      if provenance.get("features") != release_binary.get("features"):
        fail(errors, "provenance features do not match the equality release binary feature set")
      for field in ("opt_level", "lto", "panic", "codegen_units", "overflow_checks", "debug", "strip"):
        if provenance.get(field) in (None, "unknown"):
          fail(errors, f"provenance {field} must identify the resolved harness profile")
      if not isinstance(provenance.get("artifacts"), list) or not provenance.get("artifacts"):
        fail(errors, "provenance artifacts must be a non-empty list")
      if not isinstance(provenance.get("reports"), list):
        fail(errors, "provenance reports must be a list")
      tools = provenance.get("tools", {})
      if not isinstance(tools, dict):
        fail(errors, "provenance tools must be an object")
      else:
        for tool in ("python", "cargo", "rustc", "llvm_objdump", "llvm_nm", "llvm_size"):
          if not isinstance(tools.get(tool), str) or not tools.get(tool):
            fail(errors, f"provenance tools.{tool} must be a non-empty version string")
      if (root / "ct.toml").exists() and provenance.get("ct_manifest_sha256") != sha256_file(root / "ct.toml"):
        fail(errors, "provenance ct_manifest_sha256 does not match ct.toml")
      dependency_lock = root / str(provenance.get("dependency_lockfile", ""))
      if dependency_lock.exists() and provenance.get("dependency_lockfile_sha256") != sha256_file(dependency_lock):
        fail(errors, "provenance dependency_lockfile_sha256 mismatch")
      workspace_lock = root / str(provenance.get("workspace_lockfile", ""))
      if workspace_lock.exists() and provenance.get("workspace_lockfile_sha256") != sha256_file(workspace_lock):
        fail(errors, "provenance workspace_lockfile_sha256 mismatch")
      component_locks = provenance.get("component_lockfiles", [])
      if not isinstance(component_locks, list) or not component_locks:
        fail(errors, "provenance component_lockfiles must be a non-empty list")
      else:
        expected_lock_paths = {
          "Cargo.lock",
          "tools/ct-harness/Cargo.lock",
          "tools/ct-dudect/Cargo.lock",
          "tools/ct-binsec-harness/Cargo.lock",
        }
        actual_lock_paths = [record.get("path") for record in component_locks]
        if len(actual_lock_paths) != len(set(actual_lock_paths)) or set(actual_lock_paths) != expected_lock_paths:
          fail(errors, "provenance component_lockfiles does not match the required lockfile set")
        for record in component_locks:
          path = root / str(record.get("path", ""))
          if not path.is_file() or record.get("sha256") != sha256_file(path):
            fail(errors, f"provenance component lockfile mismatch: {record.get('path')!r}")

  if hashes_path.exists():
    recorded = recorded_hashes(hashes_path)
    for path in files:
      if path.is_file():
        actual = sha256_file(path)
        expected = recorded.get(path.name)
        if expected is None:
          fail(errors, f"artifact {path.name} missing from artifact-hashes.txt")
        elif expected != actual:
          fail(errors, f"artifact {path.name} hash mismatch")
    if provenance.get("artifacts"):
      provenance_hashes = {artifact.get("name"): artifact.get("sha256") for artifact in provenance.get("artifacts", [])}
      if recorded != provenance_hashes:
        fail(errors, "artifact-hashes.txt does not match provenance artifacts")

  actual_symbols = generated_symbols(artifact_dir)
  expected_symbols = {symbol for harness in ct.get("harness", []) for symbol in harness.get("symbols", [])}
  missing_symbols = sorted(expected_symbols - actual_symbols)
  if missing_symbols:
    fail(errors, f"generated artifacts missing harness symbol(s): {', '.join(missing_symbols)}")

  extra_symbols = sorted(symbol for symbol in actual_symbols if symbol not in expected_symbols)
  if extra_symbols:
    warn(warnings, f"generated artifacts include unmanifested ct_entry symbol(s): {', '.join(extra_symbols)}")

  equality_symbols = set(ct.get("equality_evidence", {}).get("release_binary", {}).get("owner_symbols", []))
  equality_symbols.update(ct.get("equality_evidence", {}).get("release_binary", {}).get("public_len_symbols", []))
  equality_object_maps = sorted(artifact_dir.glob("rscrypto_ct_evidence*.o.symbols.txt"))
  equality_object_maps.extend(sorted(artifact_dir.glob("rscrypto_ct_evidence*.obj.symbols.txt")))
  binary_maps = sorted(artifact_dir.glob("*.binary.symbols.txt"))
  binary_disassemblies = sorted(artifact_dir.glob("*.binary.disasm.txt"))
  binary_raw_disassemblies = sorted(artifact_dir.glob("*.binary.raw-disasm.txt"))
  binary_nm_maps = sorted(artifact_dir.glob("*.binary.nm-symbols.txt"))
  binary_indirect_maps = sorted(artifact_dir.glob("*.binary.indirect-symbols.txt"))
  binary_link_maps = sorted(artifact_dir.glob("rscrypto-ct-evidence.link-map.txt"))
  linked_binaries = sorted(
    path for path in artifact_dir.iterdir() if path.name in {"rscrypto-ct-evidence", "rscrypto-ct-evidence.exe"}
  )
  for role, paths in (
    ("equality pre-link object symbol map", equality_object_maps),
    ("final linked equality binary symbol map", binary_maps),
    ("final linked equality binary disassembly", binary_disassemblies),
    ("final linked equality binary raw disassembly", binary_raw_disassemblies),
    ("final linked equality binary nm symbol map", binary_nm_maps),
    ("final linked equality binary", linked_binaries),
  ):
    if len(paths) != 1:
      fail(errors, f"expected exactly one {role}; found {len(paths)}")
  if "linux" in target and len(binary_link_maps) != 1:
    fail(errors, f"expected exactly one final linked equality binary linker map; found {len(binary_link_maps)}")
  if "apple-darwin" in target and len(binary_indirect_maps) != 1:
    fail(errors, f"expected exactly one final linked equality binary indirect symbol map; found {len(binary_indirect_maps)}")
  for role, paths in (("equality pre-link object", equality_object_maps), ("final linked equality binary", binary_maps)):
    if len(paths) != 1:
      continue
    counts = symbol_counts(paths[0])
    missing = sorted(symbol for symbol in equality_symbols if counts.get(symbol) != 1)
    if missing:
      fail(errors, f"{role} missing or duplicating equality symbol(s): {', '.join(missing)}")

  heuristics = {}
  if heuristics_path.exists():
    try:
      heuristics = json.loads(heuristics_path.read_text())
    except json.JSONDecodeError as exc:
      fail(errors, f"invalid asm-heuristics JSON: {exc}")
    else:
      expected = {
        "schema_version": 2,
        "kind": "rscrypto.ct.asm-heuristics",
        "target": target,
        "profile": profile,
      }
      for key, value in expected.items():
        if heuristics.get(key) != value:
          fail(errors, f"asm-heuristics {key} expected {value!r}, got {heuristics.get(key)!r}")
      missing = heuristics.get("missing_symbols", [])
      if missing:
        fail(errors, f"asm-heuristics missing symbol(s): {', '.join(missing)}")
      if heuristics.get("unwaived_fail_count", 0) != 0:
        fail(errors, f"asm-heuristics has {heuristics.get('unwaived_fail_count')} unwaived failure finding(s)")
      if heuristics.get("needs_fix_count", 0) != 0:
        fail(errors, f"asm-heuristics has {heuristics.get('needs_fix_count')} finding(s) that need fixes")
      if heuristics.get("unclassified_count", 0) != 0:
        fail(errors, f"asm-heuristics has {heuristics.get('unclassified_count')} unclassified finding(s)")
      if heuristics.get("finding_count", 0) != (
        heuristics.get("needs_fix_count", 0)
        + heuristics.get("needs_binsec_count", 0)
        + heuristics.get("accepted_count", 0)
      ):
        fail(errors, "asm-heuristics disposition counts do not equal finding_count")
      summary = heuristics.get("symbol_summary", {})
      for symbol in sorted(expected_symbols):
        row = summary.get(symbol)
        if row is None:
          fail(errors, f"asm-heuristics missing summary for {symbol}")
        elif not row.get("present"):
          fail(errors, f"asm-heuristics symbol {symbol} not present")

      if provenance.get("reports"):
        report_hashes = {report.get("name"): report.get("sha256") for report in provenance.get("reports", [])}
        if report_hashes.get("asm-heuristics.json") != sha256_file(heuristics_path):
          fail(errors, "provenance asm-heuristics.json hash mismatch")
        if heuristics_summary_path.exists() and report_hashes.get("asm-heuristics.md") != sha256_file(
          heuristics_summary_path
        ):
          fail(errors, "provenance asm-heuristics.md hash mismatch")
      elif provenance:
        fail(errors, "provenance reports missing asm-heuristics.json")

  if evidence_path.exists():
    try:
      evidence = json.loads(evidence_path.read_text())
    except json.JSONDecodeError as exc:
      fail(errors, f"invalid evidence-index JSON: {exc}")
    else:
      expected = {
        "schema_version": 2,
        "kind": "rscrypto.ct.evidence-index",
        "crate": "rscrypto",
        "backend": "llvm",
        "target": target,
        "target_triple": target,
        "profile": profile,
      }
      for key, value in expected.items():
        if evidence.get(key) != value:
          fail(errors, f"evidence-index {key} expected {value!r}, got {evidence.get(key)!r}")
      if evidence.get("provenance") != "provenance.json":
        fail(errors, "evidence-index provenance must be provenance.json")
      if provenance_path.exists() and evidence.get("provenance_sha256") != sha256_file(provenance_path):
        fail(errors, "evidence-index provenance_sha256 mismatch")
      evidence_artifacts = {artifact.get("name"): artifact.get("sha256") for artifact in evidence.get("artifacts", [])}
      if hashes_path.exists() and evidence_artifacts != recorded_hashes(hashes_path):
        fail(errors, "evidence-index artifacts do not match artifact-hashes.txt")
      evidence_reports = {report.get("name"): report.get("sha256") for report in evidence.get("reports", [])}
      if heuristics_path.exists() and evidence_reports.get("asm-heuristics.json") != sha256_file(heuristics_path):
        fail(errors, "evidence-index asm-heuristics.json hash mismatch")
      if heuristics_summary_path.exists() and evidence_reports.get("asm-heuristics.md") != sha256_file(
        heuristics_summary_path
      ):
        fail(errors, "evidence-index asm-heuristics.md hash mismatch")

      primitive_ids = {primitive.get("id") for primitive in ct.get("primitive", [])}
      evidence_primitives = {primitive.get("id"): primitive for primitive in evidence.get("primitives", [])}
      missing_primitives = sorted(primitive_id for primitive_id in primitive_ids if primitive_id not in evidence_primitives)
      if missing_primitives:
        fail(errors, f"evidence-index missing primitive(s): {', '.join(missing_primitives)}")
      extra_primitives = sorted(primitive_id for primitive_id in evidence_primitives if primitive_id not in primitive_ids)
      if extra_primitives:
        fail(errors, f"evidence-index has unknown primitive(s): {', '.join(extra_primitives)}")

      for primitive in ct.get("primitive", []):
        primitive_id = primitive.get("id")
        evidence_primitive = evidence_primitives.get(primitive_id, {})
        expected_primitive_symbols = primitive.get("harness", {}).get("symbols", [])
        symbol_rows = evidence_primitive.get("harness", {}).get("symbols", [])
        symbol_by_name = {symbol.get("name"): symbol for symbol in symbol_rows}
        for symbol in expected_primitive_symbols:
          row = symbol_by_name.get(symbol)
          if row is None:
            fail(errors, f"evidence-index primitive {primitive_id} missing symbol {symbol}")
          elif not row.get("present"):
            fail(errors, f"evidence-index primitive {primitive_id} symbol {symbol} not present")
          elif not row.get("locations"):
            fail(errors, f"evidence-index primitive {primitive_id} symbol {symbol} has no locations")

      equality = ct.get("equality_evidence", {}).get("release_binary", {})
      equality_symbols = set(equality.get("owner_symbols", [])) | set(equality.get("public_len_symbols", []))
      equality_locations: dict[str, list[dict]] = {}
      for primitive in evidence.get("primitives", []):
        for row in primitive.get("harness", {}).get("symbols", []):
          name = row.get("name")
          if isinstance(name, str):
            equality_locations.setdefault(name, []).extend(row.get("locations", []))
      for symbol in equality_symbols:
        object_names = [str(row.get("object", "")) for row in equality_locations.get(symbol, [])]
        if sum(name.startswith("rscrypto_ct_evidence") and name.endswith((".o", ".obj")) for name in object_names) != 1:
          fail(errors, f"evidence-index lacks one equality pre-link location for {symbol}")
        if sum(name == "rscrypto-ct-evidence.binary" for name in object_names) != 1:
          fail(errors, f"evidence-index lacks one final linked location for {symbol}")

      final_closure = heuristics.get("final_equality_call_closure", {})
      if set(final_closure.get("root_symbols", [])) != equality_symbols:
        fail(errors, "final equality call closure roots do not match ct.toml")
      if final_closure.get("missing_root_symbols"):
        fail(errors, "final equality call closure has missing roots")
      if final_closure.get("unresolved_internal_calls"):
        fail(errors, "final equality call closure has unresolved internal calls")
      terminal_call_sites = final_closure.get("terminal_call_sites")
      if not isinstance(terminal_call_sites, list):
        fail(errors, "final equality call closure lacks terminal call sites")
      else:
        terminal_call_locators = [row.get("locator") for row in terminal_call_sites if isinstance(row, dict)]
        if len(terminal_call_locators) != len(terminal_call_sites) or any(
          not isinstance(locator, str) or not locator for locator in terminal_call_locators
        ):
          fail(errors, "final equality call closure has an invalid terminal call site")
        elif len(set(terminal_call_locators)) != len(terminal_call_locators):
          fail(errors, "final equality call closure has duplicate terminal call sites")
      final_disassemblies = [
        row for row in provenance.get("artifacts", []) if row.get("kind") == "linked_binary_disassembly"
      ]
      if len(final_disassemblies) != 1:
        fail(errors, "provenance lacks exactly one final linked binary disassembly")
      else:
        final_disassembly = final_disassemblies[0]
        if final_closure.get("artifact") != final_disassembly.get("path") or final_closure.get(
          "artifact_sha256"
        ) != final_disassembly.get("sha256"):
          fail(errors, "final equality call closure is not bound to the linked binary disassembly")

      unmanifested = evidence.get("unmanifested_ct_symbols", [])
      if unmanifested:
        warn(warnings, f"evidence-index includes unmanifested ct_entry symbol(s): {', '.join(unmanifested)}")


def validate_dudect(root: Path, target: str, profile: str, ct: dict, errors: list[str], warnings: list[str]) -> None:
  report_path = root / "target" / "ct" / target / profile / "dudect" / "dudect-report.json"
  if not report_path.exists():
    fail(errors, f"dudect report missing: {report_path}")
    return

  try:
    report = json.loads(report_path.read_text())
  except json.JSONDecodeError as exc:
    fail(errors, f"invalid dudect report JSON: {exc}")
    return

  expected = {
    "schema_version": 2,
    "kind": "rscrypto.ct.dudect",
    "crate": "rscrypto",
    "target": target,
    "target_triple": target,
    "profile": profile,
  }
  for key, value in expected.items():
    if report.get(key) != value:
      fail(errors, f"dudect {key} expected {value!r}, got {report.get(key)!r}")

  release_binary = ct.get("equality_evidence", {}).get("release_binary", {})
  for key, value in {
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "features": release_binary.get("features"),
    "default_features": release_binary.get("default_features"),
    "backend": release_binary.get("backend"),
  }.items():
    if report.get(key) != value:
      fail(errors, f"dudect {key} expected {value!r}, got {report.get(key)!r}")
  for key in ("crate_version", "linker", "linker_path", "linker_sha256", "linker_version"):
    if not isinstance(report.get(key), str) or not report.get(key):
      fail(errors, f"dudect {key} must be a non-empty string")

  provenance_path = root / "target" / "ct" / target / profile / "provenance.json"
  try:
    provenance = json.loads(provenance_path.read_text())
  except (OSError, json.JSONDecodeError) as exc:
    fail(errors, f"dudect cannot load lane provenance: {exc}")
    provenance = {}
  for key in (
    "configured_rustflags",
    "environment_rustflags",
    "effective_rustflags",
    "rustflags_source",
    "target_cpu",
    "target_features",
    "target_cfg_features",
  ):
    if report.get(key) != provenance.get(key):
      fail(errors, f"dudect {key} does not match lane provenance")
  if report.get("rustc_verbose") != provenance.get("tools", {}).get("rustc"):
    fail(errors, "dudect rustc does not match lane provenance")
  if report.get("cargo") != provenance.get("tools", {}).get("cargo"):
    fail(errors, "dudect cargo does not match lane provenance")
  dudect_manifest = root / "tools" / "ct-dudect" / "Cargo.toml"
  harness_manifest = root / "tools" / "ct-harness" / "Cargo.toml"
  dudect_lockfile = root / "tools" / "ct-dudect" / "Cargo.lock"
  for key, path in (
    ("dudect_manifest_sha256", dudect_manifest),
    ("harness_manifest_sha256", harness_manifest),
    ("dudect_lockfile_sha256", dudect_lockfile),
  ):
    if report.get(key) != sha256_file(path):
      fail(errors, f"dudect {key} mismatch")
  with dudect_manifest.open("rb") as source:
    expected_profile_settings = tomllib.load(source).get("profile", {}).get(profile, {})
  if report.get("profile_settings") != expected_profile_settings:
    fail(errors, "dudect profile settings do not match its manifest")

  dudect_dir = report_path.parent
  timing_artifacts = {
    "binary": dudect_dir / ("rscrypto-ct-dudect.exe" if (dudect_dir / "rscrypto-ct-dudect.exe").exists() else "rscrypto-ct-dudect"),
    "binary_disassembly": dudect_dir / "rscrypto-ct-dudect.binary.disasm.txt",
    "binary_symbols": dudect_dir / "rscrypto-ct-dudect.binary.symbols.txt",
    "linker_command_log": dudect_dir / "dudect-linker-command.txt",
  }
  for field, path in timing_artifacts.items():
    record = report.get(field, {})
    if not path.is_file():
      fail(errors, f"dudect {field} artifact missing: {path}")
    elif record.get("sha256") != sha256_file(path) or record.get("bytes") != path.stat().st_size:
      fail(errors, f"dudect {field} artifact identity mismatch")
  expected_owner_symbols = {
    f"ct_entry_owner_eq_{width}"
    for width in release_binary.get("formal_owner_widths", [])
  }
  if set(report.get("binary", {}).get("owner_symbols", [])) != expected_owner_symbols:
    fail(errors, "dudect binary owner symbol set must cover 16, 32, 48, and 64 bytes")
  call_sites = report.get("binary", {}).get("owner_call_sites", {})
  if set(call_sites) != expected_owner_symbols or any(
    isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in call_sites.values()
  ):
    fail(errors, "dudect binary must call every owner equality symbol")
  if timing_artifacts["binary_symbols"].is_file():
    counts = symbol_counts(timing_artifacts["binary_symbols"])
    wrong = sorted(symbol for symbol in expected_owner_symbols if counts.get(symbol) != 1)
    if wrong:
      fail(errors, f"dudect binary missing or duplicating owner symbol(s): {', '.join(wrong)}")

  cases = report.get("cases")
  if not isinstance(cases, list) or not cases:
    fail(errors, "dudect report cases must be a non-empty list")
    return

  failure_count = report.get("failure_count")
  actual_failures = sum(1 for case in cases if case.get("status") != "pass")
  if failure_count != actual_failures:
    fail(errors, f"dudect failure_count expected {actual_failures}, got {failure_count!r}")
  if actual_failures:
    failed = ", ".join(case.get("name", "<unnamed>") for case in cases if case.get("status") != "pass")
    fail(errors, f"dudect detected timing leakage candidate(s): {failed}")

  threshold = report.get("threshold_abs_max_t")
  if not isinstance(threshold, (int, float)) or threshold <= 0:
    fail(errors, "dudect threshold_abs_max_t must be a positive number")

  for case in cases:
    name = case.get("name", "<unnamed>")
    for key in ("primitive", "left_class", "right_class", "seed", "requested_samples", "max_t", "abs_max_t", "max_tau"):
      if key not in case:
        fail(errors, f"dudect case {name} missing {key}")
    if case.get("abs_max_t", 0) > threshold:
      fail(errors, f"dudect case {name} abs_max_t exceeds threshold {threshold}")
    raw_csv = case.get("raw_csv", {})
    if not isinstance(raw_csv, dict) or raw_csv.get("row_count", 0) <= 0:
      warn(warnings, f"dudect case {name} has no raw CSV rows")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", default=None, help="target triple to validate artifacts for")
  parser.add_argument("--profile", default="release", help="profile to validate artifacts for")
  parser.add_argument("--manifest-only", action="store_true", help="validate ct.toml only")
  parser.add_argument("--require-dudect", action="store_true", help="require a passing dudect report")
  parser.add_argument("--strict-coverage", action="store_true", help="require DudeCT and BINSEC manifest coverage for every claimed primitive/target")
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[2]
  errors: list[str] = []
  warnings: list[str] = []

  ct = validate_manifest(root, errors, warnings)
  if args.strict_coverage:
    validate_strict_coverage(ct, errors, args.target)
  if not args.manifest_only:
    target = args.target
    if target is None:
      import subprocess

      target = subprocess.check_output(["rustc", "-vV"], text=True)
      target = next(line.split(":", 1)[1].strip() for line in target.splitlines() if line.startswith("host:"))
    validate_artifacts(root, target, args.profile, ct, errors, warnings)
    if args.require_dudect:
      validate_dudect(root, target, args.profile, ct, errors, warnings)

  for message in warnings:
    print(f"warning: {message}", file=sys.stderr)
  for message in errors:
    print(f"error: {message}", file=sys.stderr)

  if errors:
    return 1

  print("ct validation ok")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
