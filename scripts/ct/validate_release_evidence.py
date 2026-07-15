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


def unique_file(root: Path, name: str) -> Path:
  matches = sorted(path for path in root.rglob(name) if path.is_file())
  if len(matches) != 1:
    fail(f"expected exactly one {name} under {root}, found {len(matches)}")
  return matches[0]


def expected_lanes(root: Path, matrix_script: Path) -> dict[str, str]:
  env = dict(os.environ)
  env.pop("GITHUB_OUTPUT", None)
  env.update({"GH_RUN_ID": "release-evidence-validation", "CT_PLATFORMS": "all"})
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


def validate_report(report: dict[str, Any], suffix: str, target: str, ct: dict[str, Any]) -> None:
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

  validate_binsec(report, suffix, target, ct)


def validate_binsec(report: dict[str, Any], suffix: str, target: str, ct: dict[str, Any]) -> None:
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
) -> None:
  for key, expected in {
    "schema_version": 1,
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
    "features": ["std", "full", "parallel"],
  }.items():
    require_equal(provenance.get(key), expected, f"{suffix} provenance {key}")
  require_equal(provenance.get("ct_manifest_sha256"), sha256_git_file(root, commit, "ct.toml"), f"{suffix} ct.toml hash")
  tools = provenance.get("tools", {})
  for tool in ("python", "cargo", "rustc", "llvm_objdump", "llvm_nm", "llvm_size"):
    if not isinstance(tools.get(tool), str) or not tools[tool]:
      fail(f"{suffix} provenance lacks the {tool} version")
  if not isinstance(provenance.get("rustc"), str) or not provenance["rustc"]:
    fail(f"{suffix} provenance lacks the rustc version")
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
    "schema_version": 1,
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

  provenance_reports = records_by_name(provenance.get("reports"), f"{suffix} provenance reports")
  evidence_reports = records_by_name(evidence.get("reports"), f"{suffix} evidence reports")
  require_equal(evidence_reports, provenance_reports, f"{suffix} evidence/provenance report records")
  for name, path in (("asm-heuristics.json", heuristics_path), ("asm-heuristics.md", heuristics_md_path)):
    row = provenance_reports.get(name)
    if row is None:
      fail(f"{suffix} provenance lacks {name}")
    require_equal(row.get("sha256"), sha256_file(path), f"{suffix} {name} hash")
  return artifacts


def validate_heuristics(heuristics: dict[str, Any], suffix: str, target: str) -> None:
  for key, expected in {
    "schema_version": 2,
    "kind": "rscrypto.ct.asm-heuristics",
    "target": target,
    "profile": "release",
    "needs_fix_count": 0,
    "unclassified_count": 0,
  }.items():
    require_equal(heuristics.get(key), expected, f"{suffix} heuristics {key}")
  require_equal(
    heuristics.get("finding_count"),
    heuristics.get("needs_fix_count", 0) + heuristics.get("needs_binsec_count", 0) + heuristics.get("accepted_count", 0),
    f"{suffix} heuristic disposition count",
  )


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
  validate_report(report, suffix, target, ct)
  validate_provenance(provenance, suffix, target, version, commit, root)
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
  validate_heuristics(heuristics, suffix, target)
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
    "effective_rustflags": provenance.get("effective_rustflags"),
    "linker": provenance.get("linker"),
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
    "evidence_mode": "exact_commit" if evidence_commit == args.commit else "release_only_delta",
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
