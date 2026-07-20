#!/usr/bin/env python3
"""Fail closed unless downloaded CT lanes form release-bound evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from toml_compat import tomllib


def fail(message: str) -> None:
  raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError) as exc:
    fail(f"cannot read {path}: {exc}")
  if not isinstance(value, dict):
    fail(f"{path} must contain a JSON object")
  return value


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def sha256_git_file(root: Path, commit: str, path: str) -> str:
  try:
    contents = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
  except subprocess.CalledProcessError as exc:
    fail(f"cannot read {path} from evidence commit {commit}: {exc}")
  return hashlib.sha256(contents).hexdigest()


def load_git_toml(root: Path, commit: str, path: str) -> dict[str, Any]:
  try:
    contents = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
    value = tomllib.loads(contents.decode())
  except (subprocess.CalledProcessError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
    fail(f"cannot parse {path} from evidence commit {commit}: {exc}")
  if not isinstance(value, dict):
    fail(f"{path} from evidence commit {commit} must contain a TOML table")
  return value


def unique_file(root: Path, name: str) -> Path:
  matches = sorted(path for path in root.rglob(name) if path.is_file())
  if len(matches) != 1:
    fail(f"expected exactly one {name} under {root}, found {len(matches)}")
  return matches[0]


def expected_lanes(root: Path, matrix_script: Path) -> dict[str, str]:
  env = dict(os.environ)
  env.pop("GITHUB_OUTPUT", None)
  env.update({"GH_RUN_ID": "release-evidence-validation", "CT_PLATFORMS": "all,riscv"})
  try:
    output = subprocess.check_output([str(matrix_script), "ct"], cwd=root, env=env, text=True)
    matrix = json.loads(output)
  except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
    fail(f"cannot resolve the release CT matrix: {exc}")
  lanes: dict[str, str] = {}
  for row in matrix:
    suffix = row.get("artifact_suffix")
    target = row.get("target")
    if not isinstance(suffix, str) or not isinstance(target, str) or suffix in lanes:
      fail("release CT matrix contains an invalid or duplicate artifact suffix")
    lanes[suffix] = target
  if not lanes:
    fail("release CT matrix is empty")
  return lanes


def target_policy(ct: dict[str, Any], target: str) -> dict[str, Any]:
  matches = [row for row in ct.get("target", []) if row.get("name") == target]
  if len(matches) != 1:
    fail(f"ct.toml must contain exactly one target policy for {target}")
  return matches[0]


def expected_dudect_cases(ct: dict[str, Any], target: str) -> set[str]:
  primitives = {row.get("id"): row for row in ct.get("primitive", [])}
  expected = set()
  for case in ct.get("dudect_case", []):
    if case.get("gate", "required") == "diagnostic":
      continue
    primitive = primitives.get(case.get("primitive"), {})
    if target in primitive.get("physical_timing_unsupported_targets", []):
      continue
    name = case.get("name")
    if isinstance(name, str):
      expected.add(name)
  return expected


def expected_binsec_kernels(ct: dict[str, Any], target: str) -> set[str]:
  expected = set()
  for kernel in ct.get("binsec_kernel", []):
    targets = kernel.get("targets", ["*"])
    if kernel.get("required", False) and ("*" in targets or target in targets):
      kernel_id = kernel.get("id")
      if isinstance(kernel_id, str):
        expected.add(kernel_id)
  return expected


def records_by_name(records: Any, source: str) -> dict[str, dict[str, Any]]:
  if not isinstance(records, list):
    fail(f"{source} must be a list")
  result = {}
  for row in records:
    if not isinstance(row, dict) or not isinstance(row.get("name"), str):
      fail(f"{source} contains an invalid record")
    name = row["name"]
    if name in result:
      fail(f"{source} contains duplicate record {name}")
    result[name] = row
  return result


def records_by_path(records: Any, source: str) -> dict[str, dict[str, Any]]:
  if not isinstance(records, list):
    fail(f"{source} must be a list")
  result = {}
  for row in records:
    if not isinstance(row, dict) or not isinstance(row.get("path"), str):
      fail(f"{source} contains an invalid record")
    path = row["path"]
    if path in result:
      fail(f"{source} contains duplicate record {path}")
    result[path] = row
  return result


def parse_hashes(path: Path) -> dict[str, str]:
  result = {}
  for line_number, line in enumerate(path.read_text().splitlines(), 1):
    match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
    if match is None:
      fail(f"{path}:{line_number} is not a canonical SHA-256 record")
    digest, name = match.groups()
    if name in result:
      fail(f"{path} contains duplicate artifact {name}")
    result[name] = digest
  if not result:
    fail(f"{path} contains no generated artifact hashes")
  return result


def tar_member_record(archive: tarfile.TarFile, relative: str) -> tuple[int, str]:
  suffix = f"/{relative}"
  matches = [member for member in archive.getmembers() if member.isfile() and (member.name == relative or member.name.endswith(suffix))]
  if len(matches) != 1:
    fail(f"raw CT archive must contain exactly one {relative}, found {len(matches)}")
  source = archive.extractfile(matches[0])
  if source is None:
    fail(f"cannot read {relative} from raw CT archive")
  digest = hashlib.sha256()
  size = 0
  for chunk in iter(lambda: source.read(1024 * 1024), b""):
    size += len(chunk)
    digest.update(chunk)
  return size, digest.hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
  if actual != expected:
    fail(f"{label}: expected {expected!r}, got {actual!r}")


def validate_exact_candidate(version: str, commit: str, evidence_version: str, evidence_commit: str) -> None:
  require_equal(evidence_version, version, "release CT evidence version")
  require_equal(evidence_commit, commit, "release CT evidence commit")


def validate_report(
  report: dict[str, Any],
  suffix: str,
  target: str,
  version: str,
  commit: str,
  provenance: dict[str, Any],
  provenance_path: Path,
  ct: dict[str, Any],
) -> None:
  for key, expected in {
    "schema_version": 1,
    "kind": "rscrypto.ct.full-report",
    "crate": "rscrypto",
    "target": target,
    "target_triple": target,
    "profile": "release",
    "status": "pass",
    "failure_count": 0,
  }.items():
    require_equal(report.get(key), expected, f"{suffix} report {key}")
  for key, expected in {
    "crate_version": version,
    "git_commit": commit,
    "git_dirty": False,
    "provenance": "provenance.json",
    "provenance_sha256": sha256_file(provenance_path),
  }.items():
    require_equal(report.get(key), expected, f"{suffix} report {key}")
  require_equal(report.get("dudect", {}).get("filter"), None, f"{suffix} DudeCT filter")
  require_equal(report.get("dudect", {}).get("gate"), "required", f"{suffix} DudeCT gate")
  require_equal(report.get("dudect", {}).get("coverage_mode"), "required", f"{suffix} DudeCT coverage mode")
  require_equal(report.get("dudect", {}).get("target_skipped_case_count"), 0, f"{suffix} skipped DudeCT cases")
  case_rows = report.get("dudect", {}).get("cases", [])
  executed_case_names = [case.get("name") for case in case_rows]
  if len(executed_case_names) != len(set(executed_case_names)):
    fail(f"{suffix} contains duplicate DudeCT case records")
  require_equal(set(executed_case_names), expected_dudect_cases(ct, target), f"{suffix} required DudeCT cases")
  if any(case.get("status") != "pass" for case in case_rows):
    fail(f"{suffix} contains a non-passing required DudeCT case")

  equality = ct.get("equality_evidence", {}).get("release_binary", {})
  root = Path(__file__).resolve().parents[2]
  dudect_manifest_path = "tools/ct-dudect/Cargo.toml"
  harness_manifest_path = "tools/ct-harness/Cargo.toml"
  dudect_lockfile_path = "tools/ct-dudect/Cargo.lock"
  dudect_manifest = load_git_toml(root, commit, dudect_manifest_path)
  expected_owner_symbols = {f"ct_entry_owner_eq_{width}" for width in equality.get("formal_owner_widths", [])}
  binary_hashes = set()
  timing_linkers = set()
  for case in case_rows:
    for key, expected in {
      "crate_version": version,
      "git_commit": commit,
      "git_dirty": False,
      "features": equality.get("features"),
      "default_features": equality.get("default_features"),
      "backend": equality.get("backend"),
      "profile_settings": dudect_manifest.get("profile", {}).get("release", {}),
      "dudect_manifest_sha256": sha256_git_file(root, commit, dudect_manifest_path),
      "harness_manifest_sha256": sha256_git_file(root, commit, harness_manifest_path),
      "dudect_lockfile_sha256": sha256_git_file(root, commit, dudect_lockfile_path),
      "rustc_verbose": provenance.get("tools", {}).get("rustc"),
      "cargo": provenance.get("tools", {}).get("cargo"),
      "configured_rustflags": provenance.get("configured_rustflags"),
      "environment_rustflags": provenance.get("environment_rustflags"),
      "effective_rustflags": provenance.get("effective_rustflags"),
      "rustflags_source": provenance.get("rustflags_source"),
      "target_cpu": provenance.get("target_cpu"),
      "target_features": provenance.get("target_features"),
      "target_cfg_features": provenance.get("target_cfg_features"),
    }.items():
      require_equal(case.get(key), expected, f"{suffix} DudeCT {case.get('name')} {key}")
    for key in ("linker", "linker_path", "linker_sha256", "linker_version"):
      if not isinstance(case.get(key), str) or not case.get(key):
        fail(f"{suffix} DudeCT {case.get('name')} lacks {key}")
    timing_linkers.add(tuple(case.get(key) for key in ("linker", "linker_path", "linker_sha256", "linker_version")))
    binary = case.get("binary", {})
    require_equal(set(binary.get("owner_symbols", [])), expected_owner_symbols, f"{suffix} DudeCT owner symbols")
    call_sites = binary.get("owner_call_sites", {})
    require_equal(set(call_sites), expected_owner_symbols, f"{suffix} DudeCT owner call sites")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in call_sites.values()):
      fail(f"{suffix} DudeCT timing executable does not call every owner root")
    if not isinstance(binary.get("sha256"), str) or not isinstance(binary.get("bytes"), int):
      fail(f"{suffix} DudeCT {case.get('name')} lacks binary identity")
    binary_hashes.add(binary.get("sha256"))
    for artifact_key in ("binary_disassembly", "binary_symbols", "linker_command_log"):
      artifact = case.get(artifact_key, {})
      if not isinstance(artifact.get("sha256"), str) or not isinstance(artifact.get("bytes"), int):
        fail(f"{suffix} DudeCT {case.get('name')} lacks {artifact_key} identity")
  if len(binary_hashes) != 1:
    fail(f"{suffix} DudeCT cases do not bind one exact timing executable")
  if len(timing_linkers) != 1:
    fail(f"{suffix} DudeCT cases do not bind one exact linker")

  report_artifacts = records_by_path(report.get("artifacts"), f"{suffix} full report artifacts")
  for path, kind in (
    ("dudect/rscrypto-ct-dudect", "dudect_binary"),
    ("dudect/rscrypto-ct-dudect.exe", "dudect_binary"),
  ):
    if path in report_artifacts:
      timing_binary = report_artifacts[path]
      break
  else:
    fail(f"{suffix} full report lacks the DudeCT timing executable")
  require_equal(timing_binary.get("kind"), kind, f"{suffix} DudeCT binary kind")
  require_equal(timing_binary.get("sha256"), next(iter(binary_hashes)), f"{suffix} DudeCT binary hash")
  for path, kind, case_field in (
    ("dudect/rscrypto-ct-dudect.binary.disasm.txt", "dudect_binary_disassembly", "binary_disassembly"),
    ("dudect/rscrypto-ct-dudect.binary.symbols.txt", "dudect_binary_symbol_map", "binary_symbols"),
    ("dudect/dudect-linker-command.txt", "dudect_linker_command", "linker_command_log"),
  ):
    artifact = report_artifacts.get(path)
    if artifact is None or artifact.get("kind") != kind:
      fail(f"{suffix} full report lacks {kind}")
    expected_hashes = {case.get(case_field, {}).get("sha256") for case in case_rows}
    require_equal(expected_hashes, {artifact.get("sha256")}, f"{suffix} {kind} hash")

  validate_binsec(report, suffix, target, version, commit, provenance, ct)


def validate_binsec(
  report: dict[str, Any],
  suffix: str,
  target: str,
  version: str,
  commit: str,
  provenance: dict[str, Any],
  ct: dict[str, Any],
) -> None:
  binsec = report.get("binsec", {})
  policy = target_policy(ct, target)
  if policy.get("binsec") == "required":
    require_equal(binsec.get("enabled"), True, f"{suffix} BINSEC enabled")
    require_equal(binsec.get("policy"), "required", f"{suffix} BINSEC policy")
    required_kernel_rows = [row for row in binsec.get("kernels", []) if row.get("required")]
    kernel_names = [row.get("kernel") for row in required_kernel_rows]
    if len(kernel_names) != len(set(kernel_names)):
      fail(f"{suffix} contains duplicate required BINSEC kernel records")
    require_equal(set(kernel_names), expected_binsec_kernels(ct, target), f"{suffix} required BINSEC kernels")
    if any(row.get("status") != "secure" for row in required_kernel_rows):
      fail(f"{suffix} contains a non-secure required BINSEC kernel")
    report_artifacts = records_by_path(report.get("artifacts"), f"{suffix} full report artifacts")
    root = Path(__file__).resolve().parents[2]
    harness_manifest_path = "tools/ct-binsec-harness/Cargo.toml"
    harness_lockfile_path = "tools/ct-binsec-harness/Cargo.lock"
    harness = load_git_toml(root, commit, harness_manifest_path)
    expected_profile = harness.get("profile", {}).get("release", {})
    dependency = harness.get("dependencies", {}).get("rscrypto", {})
    expected_features = dependency.get("features", [])
    expected_default_features = dependency.get("default-features", True)
    expected_hashes = {
      "ct_manifest_sha256": sha256_git_file(root, commit, "ct.toml"),
      "harness_manifest_sha256": sha256_git_file(root, commit, harness_manifest_path),
      "harness_lockfile_sha256": sha256_git_file(root, commit, harness_lockfile_path),
    }
    for row in required_kernel_rows:
      for key, expected in {
        "backend": "llvm",
        "target": target,
        "target_triple": target,
        "profile": "release",
        "crate_version": version,
        "git_commit": commit,
        "git_dirty": False,
        "default_features": expected_default_features,
        "features": expected_features,
        "profile_settings": expected_profile,
        "rustc_verbose": provenance.get("tools", {}).get("rustc"),
        "cargo": provenance.get("tools", {}).get("cargo"),
        **expected_hashes,
      }.items():
        require_equal(row.get(key), expected, f"{suffix} BINSEC {row.get('kernel')} {key}")
      if not isinstance(row.get("rustflags"), list):
        fail(f"{suffix} BINSEC {row.get('kernel')} lacks resolved rustflags")
      if row.get("harness_elf_type") != "exec":
        fail(f"{suffix} BINSEC {row.get('kernel')} does not use a static executable proof driver")
      if not isinstance(row.get("binsec_version"), str) or not row.get("binsec_version"):
        fail(f"{suffix} BINSEC {row.get('kernel')} lacks the BINSEC version")
      if ".text" not in row.get("load_sections", []):
        fail(f"{suffix} BINSEC {row.get('kernel')} does not load the proof driver's text section")
      component_artifacts = row.get("artifacts", {})
      for required_name in ("driver.elf", "driver.disasm", "checkct.cfg", "binsec.log"):
        digest = component_artifacts.get(required_name)
        path = f"{row.get('artifact_dir')}/{required_name}"
        record = report_artifacts.get(path)
        if not isinstance(digest, str) or record is None or record.get("sha256") != digest:
          fail(f"{suffix} BINSEC {row.get('kernel')} does not bind {required_name}")
  else:
    require_equal(binsec.get("enabled"), False, f"{suffix} BINSEC enabled")
    require_equal(binsec.get("policy"), "unsupported", f"{suffix} BINSEC policy")
    for row in report.get("primitive_evidence", []):
      if row.get("binsec", {}).get("required_kernel_count", 0):
        require_equal(row.get("binsec", {}).get("status"), "not_applicable", f"{suffix} {row.get('id')} BINSEC status")


def validate_provenance(
  provenance: dict[str, Any],
  suffix: str,
  target: str,
  version: str,
  commit: str,
  root: Path,
  ct: dict[str, Any],
) -> None:
  equality = ct.get("equality_evidence", {}).get("release_binary", {})
  harness_path = "tools/ct-harness/Cargo.toml"
  harness = load_git_toml(root, commit, harness_path)
  harness_profile = harness.get("profile", {}).get("release", {})
  harness_library = harness.get("lib", {})
  for key, expected in {
    "schema_version": 2,
    "kind": "rscrypto.ct.provenance",
    "crate": "rscrypto",
    "crate_version": version,
    "git_commit": commit,
    "git_dirty": False,
    "git_status_entries": 0,
    "git_status": [],
    "target": target,
    "target_triple": target,
    "profile": "release",
    "backend": equality.get("backend"),
    "features": equality.get("features"),
    "default_features": equality.get("default_features"),
    "harness_package": harness.get("package", {}).get("name"),
    "harness_version": harness.get("package", {}).get("version"),
    "harness_crate_type": harness_library.get("crate-type"),
    "opt_level": str(harness_profile.get("opt-level")),
    "lto": harness_profile.get("lto"),
    "panic": harness_profile.get("panic"),
    "codegen_units": harness_profile.get("codegen-units"),
    "overflow_checks": harness_profile.get("overflow-checks"),
    "debug": harness_profile.get("debug"),
    "strip": harness_profile.get("strip"),
  }.items():
    require_equal(provenance.get(key), expected, f"{suffix} provenance {key}")
  require_equal(provenance.get("ct_manifest_sha256"), sha256_git_file(root, commit, "ct.toml"), f"{suffix} ct.toml hash")
  require_equal(
    provenance.get("package_manifest_sha256"),
    sha256_git_file(root, commit, "Cargo.toml"),
    f"{suffix} Cargo.toml hash",
  )
  require_equal(
    provenance.get("harness_manifest_sha256"),
    sha256_git_file(root, commit, harness_path),
    f"{suffix} CT harness manifest hash",
  )
  tools = provenance.get("tools", {})
  for tool in ("python", "cargo", "rustc", "llvm_objdump", "llvm_nm", "llvm_size"):
    if not isinstance(tools.get(tool), str) or not tools[tool]:
      fail(f"{suffix} provenance lacks the {tool} version")
  if not isinstance(provenance.get("rustc"), str) or not provenance["rustc"]:
    fail(f"{suffix} provenance lacks the rustc version")
  for field in (
    "linker",
    "linker_source",
    "linker_path",
    "linker_sha256",
    "linker_version",
    "linker_command",
    "linker_command_sha256",
  ):
    if not isinstance(provenance.get(field), str) or not provenance.get(field):
      fail(f"{suffix} provenance lacks {field}")
  if not isinstance(provenance.get("effective_rustflags"), list):
    fail(f"{suffix} provenance effective_rustflags must be a list")
  if not isinstance(provenance.get("rustflags_source"), str) or not provenance.get("rustflags_source"):
    fail(f"{suffix} provenance rustflags_source must be a non-empty string")
  if not isinstance(provenance.get("target_features"), list):
    fail(f"{suffix} provenance target_features must be a list")
  if not isinstance(provenance.get("target_cfg_features"), list):
    fail(f"{suffix} provenance target_cfg_features must be a list")
  if provenance.get("target_cpu") is not None and not isinstance(provenance.get("target_cpu"), str):
    fail(f"{suffix} provenance target_cpu must be a string or null")
  component_locks = provenance.get("component_lockfiles", [])
  if not isinstance(component_locks, list) or not component_locks:
    fail(f"{suffix} provenance lacks component lockfile hashes")
  expected_lock_paths = {
    "Cargo.lock",
    "tools/ct-harness/Cargo.lock",
    "tools/ct-dudect/Cargo.lock",
    "tools/ct-binsec-harness/Cargo.lock",
  }
  actual_lock_paths = [record.get("path") for record in component_locks]
  if len(actual_lock_paths) != len(set(actual_lock_paths)) or set(actual_lock_paths) != expected_lock_paths:
    fail(f"{suffix} provenance component lockfile set is incomplete or duplicated")
  for record in component_locks:
    path = str(record.get("path", ""))
    require_equal(record.get("sha256"), sha256_git_file(root, commit, path), f"{suffix} component lockfile {path}")


def validate_indexes(
  provenance: dict[str, Any],
  evidence: dict[str, Any],
  suffix: str,
  target: str,
  version: str,
  provenance_path: Path,
  hashes_path: Path,
  heuristics_path: Path,
  heuristics_md_path: Path,
) -> dict[str, dict[str, Any]]:
  for key, expected in {
    "schema_version": 2,
    "kind": "rscrypto.ct.evidence-index",
    "crate": "rscrypto",
    "crate_version": version,
    "target": target,
    "target_triple": target,
    "profile": "release",
    "provenance": "provenance.json",
    "provenance_sha256": sha256_file(provenance_path),
  }.items():
    require_equal(evidence.get(key), expected, f"{suffix} evidence index {key}")

  artifacts = records_by_name(provenance.get("artifacts"), f"{suffix} provenance artifacts")
  evidence_artifacts = records_by_name(evidence.get("artifacts"), f"{suffix} evidence artifacts")
  require_equal(evidence_artifacts, artifacts, f"{suffix} evidence/provenance artifact records")
  recorded_hashes = parse_hashes(hashes_path)
  require_equal(recorded_hashes, {name: row.get("sha256") for name, row in artifacts.items()}, f"{suffix} artifact hash list")
  artifact_kinds = [row.get("kind") for row in artifacts.values()]
  for kind in (
    "linked_binary",
    "linked_binary_disassembly",
    "linked_binary_raw_disassembly",
    "linked_binary_symbol_map",
    "linked_binary_nm_symbol_map",
    "linked_binary_link_map",
    "linked_binary_size",
    "linker_command",
  ):
    require_equal(artifact_kinds.count(kind), 1, f"{suffix} {kind} artifact count")
  linker_commands = [row for row in artifacts.values() if row.get("kind") == "linker_command"]
  linker_command = linker_commands[0]
  require_equal(provenance.get("linker_command"), linker_command.get("path"), f"{suffix} linker command path")
  require_equal(
    provenance.get("linker_command_sha256"),
    linker_command.get("sha256"),
    f"{suffix} linker command hash",
  )

  provenance_reports = records_by_name(provenance.get("reports"), f"{suffix} provenance reports")
  evidence_reports = records_by_name(evidence.get("reports"), f"{suffix} evidence reports")
  require_equal(evidence_reports, provenance_reports, f"{suffix} evidence/provenance report records")
  for name, path in (("asm-heuristics.json", heuristics_path), ("asm-heuristics.md", heuristics_md_path)):
    row = provenance_reports.get(name)
    if row is None:
      fail(f"{suffix} provenance lacks {name}")
    require_equal(row.get("sha256"), sha256_file(path), f"{suffix} {name} hash")
  return artifacts


def validate_heuristics(heuristics: dict[str, Any], suffix: str, target: str, ct: dict[str, Any]) -> None:
  for key, expected in {
    "schema_version": 2,
    "kind": "rscrypto.ct.asm-heuristics",
    "target": target,
    "profile": "release",
    "needs_fix_count": 0,
    "unclassified_count": 0,
    "unwaived_fail_count": 0,
    "missing_symbols": [],
  }.items():
    require_equal(heuristics.get(key), expected, f"{suffix} heuristics {key}")
  require_equal(
    heuristics.get("finding_count"),
    heuristics.get("needs_fix_count", 0) + heuristics.get("needs_binsec_count", 0) + heuristics.get("accepted_count", 0),
    f"{suffix} heuristic disposition count",
  )
  equality = ct.get("equality_evidence", {}).get("release_binary", {})
  equality_symbols = set(equality.get("owner_symbols", [])) | set(equality.get("public_len_symbols", []))
  summary = heuristics.get("symbol_summary", {})
  for symbol in equality_symbols:
    if not summary.get(symbol, {}).get("present"):
      fail(f"{suffix} heuristics lack final equality symbol {symbol}")
  owner_summary = heuristics.get("ct_intended_call_closure", {}).get("primitive_summary", {}).get(
    "owner_equality.fixed", {}
  )
  require_equal(set(owner_summary.get("root_symbols", [])), set(equality.get("owner_symbols", [])), f"{suffix} owner roots")
  if owner_summary.get("missing_root_symbols"):
    fail(f"{suffix} owner equality closure has missing roots")
  final_closure = heuristics.get("final_equality_call_closure", {})
  require_equal(set(final_closure.get("root_symbols", [])), equality_symbols, f"{suffix} final equality roots")
  require_equal(final_closure.get("missing_root_symbols"), [], f"{suffix} final equality missing roots")
  require_equal(final_closure.get("unresolved_internal_calls"), [], f"{suffix} final equality unresolved calls")
  terminal_call_sites = final_closure.get("terminal_call_sites")
  if not isinstance(terminal_call_sites, list):
    fail(f"{suffix} final equality closure lacks terminal call sites")
  terminal_call_locators = [row.get("locator") for row in terminal_call_sites if isinstance(row, dict)]
  if len(terminal_call_locators) != len(terminal_call_sites) or any(
    not isinstance(locator, str) or not locator for locator in terminal_call_locators
  ):
    fail(f"{suffix} final equality closure has an invalid terminal call site")
  if len(set(terminal_call_locators)) != len(terminal_call_locators):
    fail(f"{suffix} final equality closure has duplicate terminal call sites")
  disassemblies = [
    row for row in heuristics.get("disassembly_files", []) if row.get("path", "").endswith(".binary.disasm.txt")
  ]
  if len(disassemblies) != 1:
    fail(f"{suffix} heuristics do not identify one final binary disassembly")
  require_equal(final_closure.get("artifact"), disassemblies[0].get("path"), f"{suffix} final equality artifact")
  require_equal(
    final_closure.get("artifact_sha256"),
    disassemblies[0].get("sha256"),
    f"{suffix} final equality artifact hash",
  )


def validate_equality_index(evidence: dict[str, Any], suffix: str, ct: dict[str, Any]) -> None:
  equality = ct.get("equality_evidence", {}).get("release_binary", {})
  equality_symbols = set(equality.get("owner_symbols", [])) | set(equality.get("public_len_symbols", []))
  locations: dict[str, list[dict[str, Any]]] = {}
  for primitive in evidence.get("primitives", []):
    for row in primitive.get("harness", {}).get("symbols", []):
      name = row.get("name")
      if isinstance(name, str):
        locations.setdefault(name, []).extend(row.get("locations", []))
  for symbol in equality_symbols:
    object_names = [str(row.get("object", "")) for row in locations.get(symbol, [])]
    if sum(name.startswith("rscrypto_ct_evidence") and name.endswith((".o", ".obj")) for name in object_names) != 1:
      fail(f"{suffix} evidence index lacks one equality pre-link location for {symbol}")
    if sum(name == "rscrypto-ct-evidence.binary" for name in object_names) != 1:
      fail(f"{suffix} evidence index lacks one final linked location for {symbol}")


def validate_raw_archive(
  raw_path: Path,
  suffix: str,
  report: dict[str, Any],
  artifacts: dict[str, dict[str, Any]],
  compact_files: tuple[tuple[str, Path], ...],
) -> None:
  with tarfile.open(raw_path, "r:gz") as archive:
    for relative, compact_path in compact_files:
      _, digest = tar_member_record(archive, relative)
      require_equal(digest, sha256_file(compact_path), f"{suffix} raw {relative}")
    for name, row in artifacts.items():
      size, digest = tar_member_record(archive, f"artifacts/{name}")
      require_equal(size, row.get("bytes"), f"{suffix} raw artifact {name} size")
      require_equal(digest, row.get("sha256"), f"{suffix} raw artifact {name} hash")
    for row in report.get("artifacts", []):
      relative = row.get("path")
      if not isinstance(relative, str):
        fail(f"{suffix} report contains an artifact without a path")
      size, digest = tar_member_record(archive, relative)
      require_equal(size, row.get("bytes"), f"{suffix} report artifact {relative} size")
      require_equal(digest, row.get("sha256"), f"{suffix} report artifact {relative} hash")


def validate_lane(
  input_dir: Path,
  suffix: str,
  target: str,
  version: str,
  commit: str,
  root: Path,
  ct: dict[str, Any],
) -> dict[str, Any]:
  report_path = unique_file(input_dir, f"ct-report-{suffix}.json")
  report_md_path = unique_file(input_dir, f"ct-report-{suffix}.md")
  provenance_path = unique_file(input_dir, f"provenance-{suffix}.json")
  evidence_path = unique_file(input_dir, f"evidence-index-{suffix}.json")
  hashes_path = unique_file(input_dir, f"artifact-hashes-{suffix}.txt")
  heuristics_path = unique_file(input_dir, f"asm-heuristics-{suffix}.json")
  heuristics_md_path = unique_file(input_dir, f"asm-heuristics-{suffix}.md")
  raw_path = unique_file(input_dir, f"target-ct-raw-{suffix}.tar.gz")

  report = load_json(report_path)
  provenance = load_json(provenance_path)
  evidence = load_json(evidence_path)
  heuristics = load_json(heuristics_path)
  validate_report(report, suffix, target, version, commit, provenance, provenance_path, ct)
  validate_provenance(provenance, suffix, target, version, commit, root, ct)
  artifacts = validate_indexes(
    provenance,
    evidence,
    suffix,
    target,
    version,
    provenance_path,
    hashes_path,
    heuristics_path,
    heuristics_md_path,
  )
  validate_equality_index(evidence, suffix, ct)
  validate_heuristics(heuristics, suffix, target, ct)
  validate_raw_archive(
    raw_path,
    suffix,
    report,
    artifacts,
    (
      ("provenance.json", provenance_path),
      ("evidence-index.json", evidence_path),
      ("artifact-hashes.txt", hashes_path),
      ("asm-heuristics.json", heuristics_path),
      ("asm-heuristics.md", heuristics_md_path),
      ("ct-report.json", report_path),
      ("ct-report.md", report_md_path),
    ),
  )

  return {
    "suffix": suffix,
    "target": target,
    "report_sha256": sha256_file(report_path),
    "provenance_sha256": sha256_file(provenance_path),
    "evidence_index_sha256": sha256_file(evidence_path),
    "raw_artifacts_sha256": sha256_file(raw_path),
    "rustc": provenance.get("rustc"),
    "tools": provenance.get("tools"),
    "features": provenance.get("features"),
    "target_cpu": provenance.get("target_cpu"),
    "target_features": provenance.get("target_features"),
    "target_cfg_features": provenance.get("target_cfg_features"),
    "effective_rustflags": provenance.get("effective_rustflags"),
    "rustflags_source": provenance.get("rustflags_source"),
    "linker": provenance.get("linker"),
    "linker_path": provenance.get("linker_path"),
    "linker_sha256": provenance.get("linker_sha256"),
    "linker_version": provenance.get("linker_version"),
    "linker_command_sha256": provenance.get("linker_command_sha256"),
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--version", required=True)
  parser.add_argument("--commit", required=True)
  parser.add_argument("--evidence-version")
  parser.add_argument("--evidence-commit")
  parser.add_argument("--input", required=True, type=Path)
  parser.add_argument("--metadata-out", required=True, type=Path)
  args = parser.parse_args()

  if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", args.version) is None:
    fail("--version must be an unprefixed SemVer version")
  if re.fullmatch(r"[0-9a-f]{40}", args.commit) is None:
    fail("--commit must be a full lowercase Git commit")
  evidence_version = args.evidence_version or args.version
  evidence_commit = args.evidence_commit or args.commit
  if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", evidence_version) is None:
    fail("--evidence-version must be an unprefixed SemVer version")
  if re.fullmatch(r"[0-9a-f]{40}", evidence_commit) is None:
    fail("--evidence-commit must be a full lowercase Git commit")
  validate_exact_candidate(args.version, args.commit, evidence_version, evidence_commit)
  if not args.input.is_dir():
    fail(f"CT evidence artifact directory missing: {args.input}")

  root = Path(__file__).resolve().parents[2]
  matrix_script = root / "scripts" / "ci" / "emit-manual-matrix.sh"
  lanes = expected_lanes(root, matrix_script)
  actual_suffixes = {
    match.group(1)
    for path in args.input.rglob("ct-report-*.json")
    if (match := re.fullmatch(r"ct-report-(.+)\.json", path.name))
  }
  require_equal(actual_suffixes, set(lanes), "release CT lane set")

  with (root / "ct.toml").open("rb") as source:
    ct = tomllib.load(source)
  metadata = {
    "schema_version": 1,
    "kind": "rscrypto.ct.release-evidence",
    "crate": "rscrypto",
    "crate_version": args.version,
    "git_commit": args.commit,
    "evidence_crate_version": evidence_version,
    "evidence_git_commit": evidence_commit,
    "evidence_mode": "exact_commit",
    "profile": "release",
    "lanes": [
      validate_lane(args.input, suffix, target, evidence_version, evidence_commit, root, ct)
      for suffix, target in lanes.items()
    ],
  }
  args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
  args.metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
  print(f"validated {len(lanes)} release CT lanes for rscrypto {args.version} at {args.commit}")
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except ValueError as exc:
    print(f"CT release evidence validation failed: {exc}", file=__import__("sys").stderr)
    raise SystemExit(1) from exc
