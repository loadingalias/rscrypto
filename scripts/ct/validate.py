#!/usr/bin/env python3
"""Validate rscrypto constant-time manifest and artifact structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path


VALID_CLAIMS = {"ct-claimed", "ct-intended", "best-effort", "unsupported"}
VALID_HARNESS_STATUSES = {"covered", "partial", "missing", "not-applicable"}
VALID_BINSEC_TIERS = {"A", "B", "C"}
VALID_BINSEC_INPUT_KINDS = {"global", "const"}
VALID_BINSEC_POLICIES = {"required", "unsupported"}
VALID_PHYSICAL_TIMING_POLICIES = {"required", "unsupported"}
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
  "linker",
  "linker_source",
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


def ci_targets(matrix: dict) -> set[str]:
  return {row.get("name", "") for row in matrix.get("ci", []) if row.get("name")}


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


def generated_symbols(artifact_dir: Path) -> set[str]:
  symbols: set[str] = set()
  for path in artifact_dir.glob("*.symbols.txt"):
    for line in path.read_text().splitlines():
      if match := re.search(r"\b_?(ct_entry_[A-Za-z0-9_]+)\b", line):
        symbols.add(match.group(1))
  return symbols


def recorded_hashes(path: Path) -> dict[str, str]:
  recorded: dict[str, str] = {}
  for line in path.read_text().splitlines():
    parts = line.split()
    if len(parts) == 2:
      recorded[parts[1]] = parts[0]
  return recorded


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
  ci_target_set = ci_targets(matrix)
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
    if claim in {"ct-intended", "ct-claimed"} and name not in ci_target_set:
      fail(errors, f"target {name} is {claim} but is missing from .config/target-matrix.json ci")

  harness_sections = ct.get("harness", [])
  all_harness_symbols = {symbol for harness in harness_sections for symbol in harness.get("symbols", [])}
  if not harness_sections:
    fail(errors, "ct.toml has no [[harness]] sections")

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

  return ct


def validate_strict_coverage(ct: dict, errors: list[str]) -> None:
  targets = binsec_required_targets(ct)
  dudect_primitives = {case.get("primitive") for case in ct.get("dudect_case", [])}
  dudect_case_names = {case.get("name") for case in ct.get("dudect_case", []) if case.get("gate") != "diagnostic"}
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
    if primitive_requires_evidence(ct, primitive, "dudect") and primitive_id not in dudect_primitives:
      fail(errors, f"primitive {primitive_id} requires DudeCT but has no [[dudect_case]]")
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
      if provenance.get("schema_version") != 1:
        fail(errors, f"provenance schema_version expected 1, got {provenance.get('schema_version')!r}")
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
      if not isinstance(provenance.get("target_cfg_features"), list):
        fail(errors, "provenance target_cfg_features must be a list")
      if not isinstance(provenance.get("target_features"), list):
        fail(errors, "provenance target_features must be a list")
      if not isinstance(provenance.get("link_args"), list):
        fail(errors, "provenance link_args must be a list")
      if not isinstance(provenance.get("artifacts"), list) or not provenance.get("artifacts"):
        fail(errors, "provenance artifacts must be a non-empty list")
      if not isinstance(provenance.get("reports"), list):
        fail(errors, "provenance reports must be a list")
      if (root / "ct.toml").exists() and provenance.get("ct_manifest_sha256") != sha256_file(root / "ct.toml"):
        fail(errors, "provenance ct_manifest_sha256 does not match ct.toml")
      dependency_lock = root / str(provenance.get("dependency_lockfile", ""))
      if dependency_lock.exists() and provenance.get("dependency_lockfile_sha256") != sha256_file(dependency_lock):
        fail(errors, "provenance dependency_lockfile_sha256 mismatch")
      workspace_lock = root / str(provenance.get("workspace_lockfile", ""))
      if workspace_lock.exists() and provenance.get("workspace_lockfile_sha256") != sha256_file(workspace_lock):
        fail(errors, "provenance workspace_lockfile_sha256 mismatch")

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

  heuristics = {}
  if heuristics_path.exists():
    try:
      heuristics = json.loads(heuristics_path.read_text())
    except json.JSONDecodeError as exc:
      fail(errors, f"invalid asm-heuristics JSON: {exc}")
    else:
      expected = {
        "schema_version": 1,
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
      elif provenance:
        fail(errors, "provenance reports missing asm-heuristics.json")

  if evidence_path.exists():
    try:
      evidence = json.loads(evidence_path.read_text())
    except json.JSONDecodeError as exc:
      fail(errors, f"invalid evidence-index JSON: {exc}")
    else:
      expected = {
        "schema_version": 1,
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

      unmanifested = evidence.get("unmanifested_ct_symbols", [])
      if unmanifested:
        warn(warnings, f"evidence-index includes unmanifested ct_entry symbol(s): {', '.join(unmanifested)}")


def validate_dudect(root: Path, target: str, profile: str, errors: list[str], warnings: list[str]) -> None:
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
    "schema_version": 1,
    "kind": "rscrypto.ct.dudect",
    "crate": "rscrypto",
    "target": target,
    "target_triple": target,
    "profile": profile,
  }
  for key, value in expected.items():
    if report.get(key) != value:
      fail(errors, f"dudect {key} expected {value!r}, got {report.get(key)!r}")

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
    validate_strict_coverage(ct, errors)
  if not args.manifest_only:
    target = args.target
    if target is None:
      import subprocess

      target = subprocess.check_output(["rustc", "-vV"], text=True)
      target = next(line.split(":", 1)[1].strip() for line in target.splitlines() if line.startswith("host:"))
    validate_artifacts(root, target, args.profile, ct, errors, warnings)
    if args.require_dudect:
      validate_dudect(root, target, args.profile, errors, warnings)

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
