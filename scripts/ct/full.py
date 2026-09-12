#!/usr/bin/env python3
"""Run the full rscrypto CT evidence pipeline and emit release-style reports."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Embedded Windows Python omits the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance import load_toml, sha256_file
from manifest import (
  dudect_sample_count,
  is_diagnostic_dudect_case, primitive_supports_physical_timing, target_record,
  required_dudect_cases as select_required_dudect_cases,
)

@dataclass
class CommandResult:
  name: str
  command: list[str]
  status: str
  returncode: int | None
  stdout_path: str
  stderr_path: str
  started_at_utc: str
  finished_at_utc: str
  duration_seconds: float


def host_target(root: Path) -> str:
  verbose = subprocess.check_output(["rustc", "-vV"], cwd=root, text=True)
  return next(line.split(":", 1)[1].strip() for line in verbose.splitlines() if line.startswith("host:"))


def is_host_executable_target(target: str, host: str) -> bool:
  if target == host:
    return True
  compatible_linux_musl = {
    "x86_64-unknown-linux-gnu": {"x86_64-unknown-linux-musl"},
    "aarch64-unknown-linux-gnu": {"aarch64-unknown-linux-musl"},
  }
  return target in compatible_linux_musl.get(host, set())


def configure_target_environment(target: str, environment: dict[str, str]) -> None:
  if target == "s390x-unknown-linux-gnu":
    environment.setdefault(
      "CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_RUSTFLAGS",
      "-C target-feature=+vector",
    )


def now_utc() -> str:
  return datetime.now(UTC).isoformat()


def command_log_name(name: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def run_command(
  root: Path,
  logs_dir: Path,
  name: str,
  command: list[str],
  *,
  env: dict[str, str] | None = None,
  timeout: int | None = None,
) -> CommandResult:
  started = datetime.now(UTC)
  stem = command_log_name(name)
  stdout_path = logs_dir / f"{stem}.stdout.txt"
  stderr_path = logs_dir / f"{stem}.stderr.txt"
  merged_env = os.environ.copy()
  if env:
    merged_env.update(env)

  try:
    proc = subprocess.run(
      command,
      cwd=root,
      env=merged_env,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      timeout=timeout,
      check=False,
    )
    status = "pass" if proc.returncode == 0 else "fail"
    returncode: int | None = proc.returncode
    stdout = proc.stdout
    stderr = proc.stderr
  except subprocess.TimeoutExpired as exc:
    status = "timeout"
    returncode = None
    stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
    stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
    stderr += f"\ncommand timed out after {timeout} seconds\n"

  finished = datetime.now(UTC)
  stdout_path.parent.mkdir(parents=True, exist_ok=True)
  stdout_path.write_text(stdout)
  stderr_path.write_text(stderr)
  return CommandResult(
    name=name,
    command=command,
    status=status,
    returncode=returncode,
    stdout_path=str(stdout_path),
    stderr_path=str(stderr_path),
    started_at_utc=started.isoformat(),
    finished_at_utc=finished.isoformat(),
    duration_seconds=round((finished - started).total_seconds(), 3),
  )


def result_record(result: CommandResult) -> dict[str, Any]:
  return {
    "name": result.name,
    "command": result.command,
    "status": result.status,
    "returncode": result.returncode,
    "stdout": result.stdout_path,
    "stderr": result.stderr_path,
    "started_at_utc": result.started_at_utc,
    "finished_at_utc": result.finished_at_utc,
    "duration_seconds": result.duration_seconds,
  }


def skipped_step(name: str, reason: str) -> dict[str, Any]:
  timestamp = now_utc()
  return {
    "name": name,
    "command": [],
    "status": "not_applicable",
    "returncode": None,
    "stdout": "",
    "stderr": "",
    "started_at_utc": timestamp,
    "finished_at_utc": timestamp,
    "duration_seconds": 0.0,
    "reason": reason,
  }


def shell_script(root: Path, relative: str, *args: str) -> list[str]:
  script = str(root / relative)
  if os.name == "nt":
    bash = shutil.which("bash")
    if bash is None:
      raise FileNotFoundError("Git Bash is missing from PATH; run scripts/tooling/x86_64-win.ps1 -CiCt")
    return [bash, relative, *args]
  return [script, *args]


def python_script(root: Path, relative: str, *args: str) -> list[str]:
  return [sys.executable, "-X", "utf8", str(root / relative), *args]


def primitives_by_id(ct: dict[str, Any]) -> dict[str, dict[str, Any]]:
  return {primitive.get("id", ""): primitive for primitive in ct.get("primitive", []) if primitive.get("id")}


def dudect_case_supported_on_target(ct: dict[str, Any], case: dict[str, Any], target: str | None) -> bool:
  return primitive_supports_physical_timing(primitives_by_id(ct).get(str(case.get("primitive", "")), {}), target)


def filter_dudect_cases_by_target(
  ct: dict[str, Any],
  cases: list[dict[str, Any]],
  target: str | None,
) -> list[dict[str, Any]]:
  return [case for case in cases if dudect_case_supported_on_target(ct, case, target)]


def primitive_ids_requiring_dudect(ct: dict[str, Any], target: str | None = None) -> set[str]:
  ids = set()
  for primitive in ct.get("primitive", []):
    if primitive.get("claim") != "ct-intended":
      continue
    primitive_id = primitive.get("id", "")
    if not primitive_id or not primitive_supports_physical_timing(primitives_by_id(ct).get(primitive_id, {}), target):
      continue
    required = set()
    for profile_name in primitive.get("required", []):
      profile = ct.get("evidence", {}).get("profile", {}).get(profile_name, {})
      required.update(profile.get("required", []))
    if "dudect" in required:
      ids.add(primitive_id)
  return ids


def required_dudect_cases(ct: dict[str, Any], target: str | None = None) -> list[dict[str, Any]]:
  return select_required_dudect_cases(ct, target, cases=manifest_dudect_cases(ct))


def required_dudect_primitives(ct: dict[str, Any], target: str | None = None) -> set[str]:
  required_primitives = primitive_ids_requiring_dudect(ct, target)
  return {case["primitive"] for case in required_dudect_cases(ct, target) if case["primitive"] in required_primitives}


def binsec_policy(ct: dict[str, Any], target: str) -> tuple[str, str]:
  row = target_record(ct, target)
  if row is None:
    return "unsupported", "target is not listed in ct.toml"
  policy = str(row.get("binsec", "unsupported"))
  if policy == "required":
    return policy, "native BINSEC evidence is required for this target"
  if policy == "unsupported":
    return policy, str(row.get("binsec_reason", "native BINSEC loading is not supported for this target"))
  return "unsupported", f"unknown BINSEC policy {policy!r}; treating as unsupported"


def manifest_dudect_cases(ct: dict[str, Any]) -> list[dict[str, Any]]:
  cases = []
  for case in ct.get("dudect_case", []):
    missing = [key for key in ("name", "primitive", "filter") if not case.get(key)]
    if missing:
      raise ValueError(f"dudect_case missing required keys {missing}: {case!r}")
    cases.append(case)
  return cases


def dudect_filter_tokens(value: str | None) -> list[str]:
  if value is None:
    return []
  return [token.strip().lower() for token in value.split(",") if token.strip()]


def case_matches_dudect_filter(case: dict[str, Any], tokens: list[str]) -> bool:
  if not tokens:
    return True
  haystack = " ".join(
    str(case.get(key, "")).lower()
    for key in (
      "name",
      "filter",
      "primitive",
      "left_class",
      "right_class",
    )
  )
  return any(token in haystack for token in tokens)


def filter_dudect_cases(cases: list[dict[str, Any]], filter_value: str | None) -> tuple[list[dict[str, Any]], list[str]]:
  tokens = dudect_filter_tokens(filter_value)
  if not tokens:
    return cases, []
  return [case for case in cases if case_matches_dudect_filter(case, tokens)], tokens


def filter_dudect_cases_by_gate(cases: list[dict[str, Any]], gate: str) -> list[dict[str, Any]]:
  if gate == "all":
    return cases
  if gate == "required":
    return [case for case in cases if not is_diagnostic_dudect_case(case)]
  if gate == "diagnostic":
    return [case for case in cases if is_diagnostic_dudect_case(case)]
  raise ValueError(f"unsupported DudeCT gate {gate!r}")


def case_sample_count(case: dict[str, Any], *, fallback: int) -> int:
  return dudect_sample_count(case, fallback=fallback)


def case_timeout_seconds(case: dict[str, Any], fallback: int) -> int:
  value = case.get("timeout_seconds", fallback)
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise ValueError(f"DudeCT case {case.get('name', '<unnamed>')} timeout_seconds must be a positive integer")
  return value


def file_record(path: Path, base: Path, kind: str) -> dict[str, Any]:
  return {
    "path": str(path.relative_to(base)),
    "kind": kind,
    "sha256": sha256_file(path),
    "bytes": path.stat().st_size,
  }


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
  if not path.exists():
    return None
  try:
    return json.loads(path.read_text())
  except json.JSONDecodeError:
    return None


def validate_dudect_case_report(manifest_case: dict[str, Any], reported_case: dict[str, Any]) -> None:
  name = str(manifest_case["name"])
  manifest_gate = str(manifest_case.get("gate", "required"))
  reported_gate = str(reported_case.get("gate", "required"))
  if reported_case.get("name") != name:
    raise ValueError(
      f"DudeCT child report returned case {reported_case.get('name')!r} while {name!r} was requested"
    )
  if reported_gate != manifest_gate:
    raise ValueError(
      f"DudeCT case {name!r} gate disagrees with ct.toml: report={reported_gate!r}, manifest={manifest_gate!r}"
    )
  if manifest_gate != "diagnostic" and reported_case.get("status") == "diagnostic-fail":
    raise ValueError(f"manifest-required DudeCT case {name!r} cannot report diagnostic-fail")


def validated_dudect_case_report(manifest_case: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
  name = str(manifest_case["name"])
  if not isinstance(report, dict) or not isinstance(report.get("cases"), list):
    raise ValueError("DudeCT child report must contain a cases array")
  if any(not isinstance(row, dict) for row in report["cases"]):
    raise ValueError("DudeCT child report cases must be objects")
  matches = [row for row in report["cases"] if row.get("name") == name]
  if len(report["cases"]) != 1 or len(matches) != 1:
    raise ValueError(f"DudeCT child report must contain exactly one result for requested case {name!r}")
  if matches[0].get("status") not in ("pass", "fail", "diagnostic-fail"):
    raise ValueError(f"DudeCT case {name!r} has an invalid or missing status")
  validate_dudect_case_report(manifest_case, matches[0])
  return matches[0]


def candidate_identity(out_dir: Path) -> dict[str, Any]:
  provenance_path = out_dir / "provenance.json"
  provenance = load_json_if_exists(provenance_path)
  if provenance is None:
    return {
      "crate_version": None,
      "git_commit": None,
      "git_dirty": None,
      "provenance": "provenance.json",
      "provenance_sha256": None,
    }
  return {
    "crate_version": provenance.get("crate_version"),
    "git_commit": provenance.get("git_commit"),
    "git_dirty": provenance.get("git_dirty"),
    "provenance": "provenance.json",
    "provenance_sha256": sha256_file(provenance_path),
  }


def collect_artifact_records(out_dir: Path, dudect_run: Path | None = None) -> list[dict[str, Any]]:
  records = []
  for relative, kind in (
    ("provenance.json", "provenance"),
    ("evidence-index.json", "evidence_index"),
    ("asm-heuristics.json", "asm_heuristics"),
    ("asm-heuristics.md", "asm_heuristics_summary"),
  ):
    path = out_dir / relative
    if path.exists():
      records.append(file_record(path, out_dir, kind))

  for path in sorted(dudect_run.rglob("*")) if dudect_run is not None else []:
    if not path.is_file():
      continue
    kind = {
      "dudect-report.json": "dudect_report",
      "dudect-raw.csv": "dudect_raw_samples",
      "dudect.stdout.txt": "dudect_stdout",
      "rscrypto-ct-dudect": "dudect_binary",
      "rscrypto-ct-dudect.exe": "dudect_binary",
      "rscrypto-ct-dudect.binary.disasm.txt": "dudect_binary_disassembly",
      "rscrypto-ct-dudect.binary.symbols.txt": "dudect_binary_symbol_map",
      "dudect-linker-command.txt": "dudect_linker_command",
      "prepared.json": "dudect_preparation",
    }.get(path.name, "dudect_component")
    records.append(file_record(path, out_dir, kind))

  for path in sorted((out_dir / "binsec").glob("*/*")):
    if not path.is_file():
      continue
    kind = {
      "binsec-report.json": "binsec_report",
      "driver.elf": "binsec_driver",
      "driver.disasm": "binsec_driver_disassembly",
      "checkct.cfg": "binsec_configuration",
      "binsec.log": "binsec_log",
      "binsec-stats.toml": "binsec_statistics",
    }.get(path.name, "binsec_component")
    records.append(file_record(path, out_dir, kind))

  return records


def dudect_case_result(
  root: Path,
  logs_dir: Path,
  samples: int,
  threshold: float,
  case: dict[str, Any],
  timeout: int | None,
  prepared: Path,
) -> dict[str, Any]:
  cases_dir = prepared.parent.parent / "cases"
  cases_dir.mkdir(parents=True, exist_ok=True)
  evidence_dir = Path(tempfile.mkdtemp(prefix=f"{case['name']}-", dir=cases_dir)).resolve()
  command = [
    *python_script(root, "scripts/ct/dudect_execute.py"),
    "--prepared", str(prepared), "--evidence-dir", str(evidence_dir),
    "--samples", str(samples), "--threshold", str(threshold), "--filter", case["filter"],
  ]
  if timeout is not None:
    command += ["--timeout", str(timeout)]
  result = run_command(
    root, logs_dir, f"dudect-{evidence_dir.name}", command,
  )
  if result.returncode == 124:
    result.status = "timeout"
  artifacts = [{"name": path.name, "path": str(path)} for path in sorted(evidence_dir.iterdir()) if path.is_file()]
  report_path = evidence_dir / "dudect-report.json"
  report = None
  case_report = None
  report_error = None
  try:
    report = json.loads(report_path.read_text())
    case_report = validated_dudect_case_report(case, report)
  except (OSError, UnicodeError, ValueError) as exc:
    report_error = f"Missing or invalid current DudeCT report: {exc}"
    report = None

  statistical_status = case_report["status"] if case_report is not None else None
  expected_exit = 1 if statistical_status == "fail" else 0
  command_matches_report = result.returncode == expected_exit and result.status != "timeout"
  if result.status == "timeout":
    status = "timeout"
  elif case_report is None or not command_matches_report:
    status = "tooling-fail"
  else:
    status = statistical_status
  failure_count = int(statistical_status == "fail") if case_report is not None else None

  row = {
    "name": case["name"],
    "primitive": case["primitive"],
    "filter": case["filter"],
    "gate": case.get("gate", "required"),
    "diagnostic_reason": case.get("reason") or case.get("notes"),
    "left_class": case.get("left_class"),
    "right_class": case.get("right_class"),
    "status": status,
    "statistical_status": statistical_status,
    "report_error": report_error,
    "requested_samples": samples,
    "timeout_seconds": timeout,
    "timeout_reason": case.get("timeout_reason"),
    "failure_count": failure_count,
    "command_result": result_record(result),
    "artifacts": artifacts,
    "report": str(report_path) if report_path.exists() else None,
  }
  if case_report is not None:
    for key in (
      "gate",
      "diagnostic_reason",
      "left_class",
      "right_class",
      "seed",
      "max_t",
      "abs_max_t",
      "max_tau",
      "needed_samples_for_tau_threshold",
      "threshold_abs_max_t",
      "samples_millions",
      "raw_csv",
    ):
      if key in case_report:
        row[key] = case_report[key]
  if report is not None:
    for key in (
      "crate_version",
      "git_commit",
      "git_dirty",
      "features",
      "default_features",
      "backend",
      "ct_manifest_sha256",
      "profile_settings",
      "dudect_manifest_sha256",
      "harness_manifest_sha256",
      "dudect_lockfile_sha256",
      "dudect_runner_sources",
      "rustc_verbose",
      "cargo",
      "configured_rustflags",
      "environment_rustflags",
      "effective_rustflags",
      "rustflags_source",
      "target_cpu",
      "target_features",
      "target_cfg_features",
      "linker",
      "linker_path",
      "linker_sha256",
      "linker_version",
      "binary",
      "binary_disassembly",
      "binary_symbols",
      "linker_command_log",
      "build_host",
      "host",
      "transfer",
    ):
      if key in report:
        row[key] = report[key]
  row["primitive"] = case["primitive"]
  row["gate"] = case.get("gate", "required")
  row["diagnostic_reason"] = case.get("reason") or case.get("notes") or row.get("diagnostic_reason")
  row["left_class"] = case.get("left_class", row.get("left_class"))
  row["right_class"] = case.get("right_class", row.get("right_class"))
  if row["gate"] == "diagnostic" and row["status"] == "fail":
    row["status"] = "diagnostic-fail"
  return row


def run_dudect_cases(root, out_dir, logs_dir, target, profile, manifest_cases, threshold, dudect_timeout, transferred=None):
  dudect_cases = []
  if transferred is None:
    dudect_runs = out_dir / "dudect" / "runs"
    dudect_runs.mkdir(parents=True, exist_ok=True)
    dudect_run = Path(tempfile.mkdtemp(prefix="run-", dir=dudect_runs)).resolve()
    prepared = dudect_run / "shared" / "prepared.json"
    preparation = run_command(root, logs_dir, f"dudect-prepare-{dudect_run.name}", [
      *shell_script(root, "scripts/ct/dudect.sh"), "--prepare-only",
      "--shared-dir", str(prepared.parent), "--target", target, "--profile", profile,
    ])
  else:
    prepared = transferred
    dudect_run = prepared.parent.parent
    timestamp = now_utc()
    preparation = CommandResult("ct-dudect-import", [], "pass", 0, "", "", timestamp, timestamp, 0.0)
  fallback_samples = int(os.environ.get("RSCRYPTO_CT_DUDECT_SAMPLES", "20000"))
  for case in manifest_cases if preparation.status == "pass" else []:
    samples = case_sample_count(case, fallback=fallback_samples)
    timeout_seconds = case_timeout_seconds(case, dudect_timeout)
    print(f"ct-full: dudect {case['name']}", flush=True)
    dudect_cases.append(
      dudect_case_result(
        root,
        logs_dir,
        samples,
        threshold,
        case,
        timeout_seconds,
        prepared,
      )
    )
    row = dudect_cases[-1]
    result = row["command_result"]
    print(
      f"ct-full: dudect {row['name']}: {row['status']} "
      f"({result['duration_seconds']:.1f}s, exit={result['returncode']}, "
      f"timeout={row['timeout_seconds']}s, abs_max_t={row.get('abs_max_t')}, "
      f"threshold={row.get('threshold_abs_max_t', threshold)})",
      flush=True,
    )
    if row["status"] not in ("pass", "diagnostic-fail"):
      if row.get("report_error"):
        print(row["report_error"], file=sys.stderr, flush=True)
      for key in ("stdout", "stderr"):
        path = Path(result[key])
        if path.is_file():
          print(f"ct-full: {key}: {path}\n{path.read_text(errors='replace')[-16384:]}",
                file=sys.stderr, flush=True)
      break

  return dudect_run, preparation, dudect_cases


def binsec_result_category(kernel: dict[str, Any]) -> str:
  status = str(kernel.get("status", "unknown"))
  reason = str(kernel.get("reason", "")).lower()
  if status == "secure":
    return "pass"
  if status == "insecure":
    return "ct_failure"
  if status in {"unknown", "blocked"} and (
    "timeout" in reason or "incomplete" in reason or "unknown" in reason or status == "blocked"
  ):
    return "proof_inconclusive"
  return "tooling_failure"


def summarize_status(rows: list[dict[str, Any]], key: str = "status") -> dict[str, int]:
  summary: dict[str, int] = {}
  for row in rows:
    status = str(row.get(key, "unknown"))
    summary[status] = summary.get(status, 0) + 1
  return summary


def summarize_findings(findings: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
  by_category: dict[str, int] = {}
  for finding in findings:
    category = str(finding.get("category", finding.get("kind", "unknown")))
    by_category[category] = by_category.get(category, 0) + 1
  return {
    "blockers": len(findings),
    "diagnostics": len(diagnostics),
    "by_category": by_category,
  }


def primitive_manifest_dudect_cases(ct: dict[str, Any], primitive_id: str, target: str | None) -> list[dict[str, Any]]:
  return [case for case in manifest_dudect_cases(ct) if case["primitive"] == primitive_id and dudect_case_supported_on_target(ct, case, target)]


def primitive_binsec_kernels(ct: dict[str, Any], primitive_id: str) -> list[dict[str, Any]]:
  return [kernel for kernel in ct.get("binsec_kernel", []) if kernel.get("primitive") == primitive_id]


def evidence_status(failures: int, warnings: int, present: bool = True) -> str:
  if not present:
    return "missing"
  if failures:
    return "fail"
  if warnings:
    return "review"
  return "pass"


def build_primitive_evidence(
  ct: dict[str, Any],
  target: str,
  dudect_cases: list[dict[str, Any]],
  binsec_kernels: list[dict[str, Any]],
  asm_report: dict[str, Any] | None,
  *,
  binsec_enabled: bool,
  coverage_limited: bool,
) -> list[dict[str, Any]]:
  asm_by_primitive = (
    (asm_report or {})
    .get("ct_intended_call_closure", {})
    .get("primitive_summary", {})
  )
  executed_dudect_by_primitive: dict[str, list[dict[str, Any]]] = {}
  for case in dudect_cases:
    executed_dudect_by_primitive.setdefault(case["primitive"], []).append(case)

  binsec_by_primitive: dict[str, list[dict[str, Any]]] = {}
  for kernel in binsec_kernels:
    primitive = kernel.get("primitive")
    if isinstance(primitive, str):
      binsec_by_primitive.setdefault(primitive, []).append(kernel)

  rows = []
  for primitive in ct.get("primitive", []):
    primitive_id = primitive.get("id")
    if not isinstance(primitive_id, str):
      continue

    claim = primitive.get("claim", "not-claimed")
    physical_supported = primitive_supports_physical_timing(primitives_by_id(ct).get(primitive_id, {}), target)
    manifest_cases = primitive_manifest_dudect_cases(ct, primitive_id, target)
    required_cases = [case for case in manifest_cases if not is_diagnostic_dudect_case(case)]
    executed_cases = executed_dudect_by_primitive.get(primitive_id, [])
    executed_required = [case for case in executed_cases if case.get("gate") != "diagnostic"]
    failing_required = [case for case in executed_required if case.get("status") != "pass"]
    required_case_names = {case["name"] for case in required_cases}
    executed_required_names = {case["name"] for case in executed_required}
    missing_required_names = sorted(required_case_names - executed_required_names)

    if not physical_supported:
      dudect_status = "unsupported"
    elif failing_required:
      dudect_status = "fail"
    elif required_cases and not missing_required_names:
      dudect_status = "pass"
    elif required_cases and executed_required:
      dudect_status = "partial"
    elif required_cases and coverage_limited:
      dudect_status = "not_selected"
    elif required_cases:
      dudect_status = "missing"
    else:
      dudect_status = "not_required"

    asm_summary = asm_by_primitive.get(primitive_id)
    if claim == "ct-intended":
      asm_status = evidence_status(
        int((asm_summary or {}).get("unwaived_fail_count", 0)),
        int((asm_summary or {}).get("unwaived_warn_count", 0)),
        asm_summary is not None,
      )
    else:
      asm_status = "entry_only"

    manifest_binsec = primitive_binsec_kernels(ct, primitive_id)
    required_binsec = [kernel for kernel in manifest_binsec if kernel.get("required", False)]
    executed_binsec = binsec_by_primitive.get(primitive_id, [])
    failing_binsec = [kernel for kernel in executed_binsec if kernel.get("required", False) and kernel.get("status") != "secure"]
    if failing_binsec:
      binsec_status = "fail"
    elif required_binsec and executed_binsec and all(
      kernel.get("status") == "secure" for kernel in executed_binsec if kernel.get("required", False)
    ):
      binsec_status = "pass"
    elif required_binsec and not binsec_enabled:
      binsec_status = "not_applicable"
    elif required_binsec:
      binsec_status = "not_run"
    else:
      binsec_status = "not_required"

    blockers = []
    if claim == "ct-intended" and physical_supported:
      if asm_status in {"fail", "missing"}:
        blockers.append("asm")
      if dudect_status == "fail" or (dudect_status == "missing" and not coverage_limited):
        blockers.append("dudect")
      if binsec_status == "fail":
        blockers.append("binsec")

    if claim != "ct-intended":
      status = "classified"
    elif not physical_supported:
      status = "unsupported_on_target"
    elif coverage_limited and dudect_status == "not_selected":
      status = "not_evaluated"
    elif coverage_limited and dudect_status == "partial":
      status = "partial"
    else:
      status = "fail" if blockers else "pass"

    rows.append(
      {
        "id": primitive_id,
        "tier": primitive.get("tier"),
        "claim": claim,
        "status": status,
        "blockers": blockers,
        "physical_timing_supported": physical_supported,
        "physical_timing_unsupported_reason": primitive.get("physical_timing_unsupported_reason"),
        "entrypoints": primitive.get("entrypoints", []),
        "secrets": primitive.get("secrets", []),
        "public": primitive.get("public", []),
        "may_leak": primitive.get("may_leak", []),
        "harness": {
          "status": primitive.get("harness", {}).get("status"),
          "symbols": primitive.get("harness", {}).get("symbols", []),
          "coverage": primitive.get("harness", {}).get("coverage"),
        },
        "asm": {
          "status": asm_status,
          "reachable_symbol_count": (asm_summary or {}).get("reachable_symbol_count"),
          "finding_count": (asm_summary or {}).get("finding_count", 0),
          "unwaived_fail_count": (asm_summary or {}).get("unwaived_fail_count", 0),
          "unwaived_warn_count": (asm_summary or {}).get("unwaived_warn_count", 0),
        },
        "dudect": {
          "status": dudect_status,
          "required_case_count": len(required_cases),
          "executed_required_case_count": len(executed_required),
          "passing_required_case_count": sum(1 for case in executed_required if case.get("status") == "pass"),
          "missing_required_cases": missing_required_names,
          "executed_cases": [case["name"] for case in executed_cases],
          "status_counts": summarize_status(executed_cases),
        },
        "binsec": {
          "status": binsec_status,
          "required_kernel_count": len(required_binsec),
          "executed_kernel_count": len(executed_binsec),
          "secure_kernel_count": sum(1 for kernel in executed_binsec if kernel.get("status") == "secure"),
          "status_counts": summarize_status(executed_binsec),
        },
        "notes": primitive.get("notes"),
      }
    )
  return rows


def format_primitive_evidence_cell(row: dict[str, Any], key: str) -> str:
  evidence = row.get(key, {})
  status = evidence.get("status", "unknown")
  if key == "asm":
    fails = evidence.get("unwaived_fail_count", 0)
    warns = evidence.get("unwaived_warn_count", 0)
    reachable = evidence.get("reachable_symbol_count")
    suffix = f", reachable={reachable}" if reachable is not None else ""
    return f"`{status}` (fail={fails}, warn={warns}{suffix})"
  if key == "dudect":
    return (
      f"`{status}` "
      f"({evidence.get('passing_required_case_count', 0)}/"
      f"{evidence.get('required_case_count', 0)} required)"
    )
  if key == "binsec":
    return (
      f"`{status}` "
      f"({evidence.get('secure_kernel_count', 0)}/"
      f"{evidence.get('required_kernel_count', 0)} required)"
    )
  return f"`{status}`"


def markdown_escape_cell(value: Any) -> str:
  return str(value).replace("|", "\\|")


def markdown_report(report: dict[str, Any]) -> str:
  summary = report.get("summary", {})
  lines = [
    "# rscrypto CT Report",
    "",
    f"- Generated: `{report['generated_at_utc']}`",
    f"- Target: `{report['target']}`",
    f"- Profile: `{report['profile']}`",
    f"- Status: `{report['status']}`",
    f"- Blocking findings: `{summary.get('blockers', report['failure_count'])}`",
    f"- Non-blocking diagnostics: `{summary.get('diagnostics', 0)}`",
    "",
    "## What To Fix",
    "",
  ]
  if report["findings"]:
    for finding in report["findings"]:
      reason = f" - {finding['reason']}" if finding.get("reason") else ""
      category = finding.get("category", finding["kind"])
      lines.append(f"- `{category}` `{finding['kind']}`: {finding['summary']}{reason}")
  else:
    lines.append("- No blocking CT evidence findings.")

  if report.get("diagnostics"):
    lines.extend(["", "## Diagnostics"])
    for finding in report["diagnostics"]:
      reason = f" - {finding['reason']}" if finding.get("reason") else ""
      lines.append(f"- `{finding['kind']}`: {finding['summary']}{reason}")

  lines.extend(["", "## Gate Summary", ""])
  for step in report["steps"]:
    reason = f" - {step['reason']}" if step.get("reason") else ""
    lines.append(f"- `{step['name']}`: `{step['status']}`{reason}")

  primitive_evidence = report.get("primitive_evidence", [])
  if primitive_evidence:
    lines.extend(["", "## Primitive Evidence", ""])
    lines.append("| Primitive | Claim | Status | ASM | DudeCT | BINSEC |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in primitive_evidence:
      if row.get("claim") not in {"ct-intended", "best-effort"}:
        continue
      lines.append(
        "| "
        + " | ".join(
          [
            f"`{markdown_escape_cell(row['id'])}`",
            f"`{markdown_escape_cell(row.get('claim', 'unknown'))}`",
            f"`{markdown_escape_cell(row.get('status', 'unknown'))}`",
            format_primitive_evidence_cell(row, "asm"),
            format_primitive_evidence_cell(row, "dudect"),
            format_primitive_evidence_cell(row, "binsec"),
          ]
        )
        + " |"
      )

  lines.extend(["", "## BINSEC Summary", ""])
  if report["binsec"]["enabled"]:
    counts = report["binsec"].get("status_counts", {})
    lines.append(", ".join(f"`{status}`={count}" for status, count in sorted(counts.items())) or "- no kernels")
    non_secure = [kernel for kernel in report["binsec"]["kernels"] if kernel.get("status") != "secure"]
    if non_secure:
      lines.append("")
      for kernel in non_secure:
        category = kernel.get("category", "unknown")
        reason = f" - {kernel['reason']}" if kernel.get("reason") else ""
        lines.append(f"- `{category}` `{kernel['kernel']}` (`{kernel['primitive']}`): `{kernel['status']}`{reason}")
    else:
      lines.append("")
      lines.append("- All required BINSEC kernels reported `secure`.")
  else:
    lines.append(f"- `{report['binsec']['policy']}`: {report['binsec']['reason']}")

  lines.extend(["", "## DudeCT Summary", ""])
  coverage_mode = report["dudect"].get("coverage_mode", "full")
  selected = report["dudect"].get("selected_case_count", len(report["dudect"].get("cases", [])))
  manifest_total = report["dudect"].get("manifest_case_count", selected)
  gate_total = report["dudect"].get("gate_case_count", selected)
  gate = report["dudect"].get("gate", "all")
  lines.append(f"- Gate: `{gate}`")
  lines.append(f"- Coverage mode: `{coverage_mode}`")
  lines.append(f"- Selected cases: `{selected}` / `{gate_total}` gate cases (`{manifest_total}` total manifest cases)")
  if report["dudect"].get("filter"):
    lines.append(f"- Filter: `{report['dudect']['filter']}`")
  skipped_cases = report["dudect"].get("target_skipped_cases", [])
  if skipped_cases:
    lines.append(f"- Target-unsupported cases skipped: `{len(skipped_cases)}`")
  coverage = report.get("coverage", {})
  if coverage:
    required = len(coverage.get("required_dudect_primitives", []))
    executed_required = len(coverage.get("executed_required_dudect_primitives", []))
    passing_required = len(coverage.get("passing_required_dudect_primitives", []))
    lines.append(f"- Required primitive coverage: `{passing_required}` passing / `{executed_required}` executed / `{required}` required")
  lines.append("")
  counts = report["dudect"].get("status_counts", {})
  lines.append(", ".join(f"`{status}`={count}" for status, count in sorted(counts.items())) or "- no cases")
  non_pass = [case for case in report["dudect"]["cases"] if case.get("status") != "pass"]
  if non_pass:
    lines.append("")
    for case in non_pass:
      detail = f", failures={case['failure_count']}" if case.get("failure_count") is not None else ""
      reason = f" - {case['diagnostic_reason']}" if case.get("diagnostic_reason") else ""
      lines.append(f"- `{case['name']}` (`{case['primitive']}`): `{case['status']}`{detail}{reason}")
  if report["coverage"]["missing_dudect_primitives"]:
    lines.extend(["", "## Missing DudeCT Coverage", ""])
    for primitive in report["coverage"]["missing_dudect_primitives"]:
      lines.append(f"- `{primitive}`")
  if report["known_findings"]:
    lines.extend(["", "## Known Findings", ""])
    for finding in report["known_findings"]:
      lines.append(f"- `{finding['severity']}` `{finding['id']}`: {finding['summary']}")
  lines.extend(["", "## Artifacts", ""])
  artifact_counts = summarize_status(report["artifacts"], "kind")
  for kind, count in sorted(artifact_counts.items()):
    lines.append(f"- `{kind}`: `{count}` file(s)")
  lines.append("")
  return "\n".join(lines)


def build_findings(
  steps: list[dict[str, Any]],
  dudect_cases: list[dict[str, Any]],
  binsec_kernels: list[dict[str, Any]],
  missing_dudect: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  findings: list[dict[str, Any]] = []
  diagnostics: list[dict[str, Any]] = []

  for step in steps:
    if step["status"] not in {"fail", "timeout"}:
      continue
    findings.append(
      {
        "kind": "gate_failure",
        "category": "tooling_failure",
        "severity": "blocker",
        "summary": f"{step['name']} failed before complete evidence could be collected",
      }
    )

  for case in dudect_cases:
    if case["status"] == "pass":
      continue
    if case.get("gate") == "diagnostic" and case["status"] in {"fail", "diagnostic-fail"}:
      diagnostics.append(
        {
          "kind": "dudect_diagnostic",
          "severity": "diagnostic",
          "summary": f"{case['name']} showed timing separation outside the release CT claim",
          "primitive": case["primitive"],
          "reason": case.get("diagnostic_reason"),
        }
      )
      continue
    command_status = case.get("command_result", {}).get("status")
    if case["status"] in {"tooling-fail", "timeout"} or (case.get("report") is None and command_status in {"fail", "timeout"}):
      timed_out = command_status == "timeout"
      findings.append(
        {
          "kind": "dudect_inconclusive",
          "category": "evidence_inconclusive" if timed_out else "tooling_failure",
          "severity": "blocker",
          "summary": (
            f"{case['name']} exceeded its {case.get('timeout_seconds')}-second DudeCT evidence budget"
            if timed_out
            else f"{case['name']} did not complete with valid current DudeCT evidence"
          ),
          "primitive": case["primitive"],
          "timeout_seconds": case.get("timeout_seconds"),
        }
      )
    else:
      findings.append(
        {
          "kind": "dudect_failure",
          "category": "timing_failure",
          "severity": "blocker",
          "summary": f"{case['name']} produced a timing result above the DudeCT threshold",
          "primitive": case["primitive"],
        }
      )

  for kernel in binsec_kernels:
    if kernel.get("status") == "secure":
      continue
    category = str(kernel.get("category", binsec_result_category(kernel)))
    kind = "binsec_inconclusive" if category == "proof_inconclusive" else "binsec_failure"
    summary = (
      f"{kernel.get('kernel')} did not complete a BINSEC proof"
      if category == "proof_inconclusive"
      else f"{kernel.get('kernel')} did not pass BINSEC"
    )
    if not kernel.get("required", False) and category != "ct_failure":
      diagnostics.append(
        {
          "kind": "binsec_diagnostic",
          "severity": "diagnostic",
          "summary": summary,
          "primitive": kernel.get("primitive"),
          "reason": kernel.get("reason"),
        }
      )
      continue
    findings.append(
      {
        "kind": kind,
        "category": category,
        "severity": "blocker",
        "summary": summary,
        "primitive": kernel.get("primitive"),
        "reason": kernel.get("reason"),
      }
    )

  for primitive in missing_dudect:
    findings.append(
      {
        "kind": "missing_dudect",
        "category": "coverage_gap",
        "severity": "blocker",
        "summary": f"{primitive} requires DudeCT evidence but has no executed manifest case",
        "primitive": primitive,
      }
    )

  return findings, diagnostics


def write_full_report(out_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
  json_path = out_dir / "ct-report.json"
  md_path = out_dir / "ct-report.md"
  json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  md_path.write_text(markdown_report(report))
  print(f"ct-full: {report['status']}; {report['failure_count']} blocking findings", flush=True)
  for finding in report["findings"]:
    print(f"ct-full: {finding['category']}: {finding['summary']}", flush=True)
  if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
    with Path(summary_path).open("a", encoding="utf-8") as summary:
      summary.write(md_path.read_text())
  return json_path, md_path


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  transfer_args = parser.add_mutually_exclusive_group()
  transfer_args.add_argument("--prepare-archive", type=Path, help="prepare sealed cross-compiled CT evidence without timing")
  transfer_args.add_argument("--run-archive", type=Path, help="measure sealed cross-compiled CT evidence without rebuilding")
  parser.add_argument("--target", default=None)
  parser.add_argument("--profile", default="release")
  parser.add_argument("--threshold", type=float, default=float(os.environ.get("RSCRYPTO_CT_DUDECT_THRESHOLD", "10.0")))
  parser.add_argument("--binsec-timeout", type=int, default=120)
  parser.add_argument(
    "--binsec-smt-solver",
    default=os.environ.get("BINSEC_SMT_SOLVER", "bitwuzla:builtin"),
    help="BINSEC SMT solver backend; default: BINSEC_SMT_SOLVER or bitwuzla:builtin",
  )
  parser.add_argument("--dudect-timeout", type=int, default=300)
  parser.add_argument(
    "--dudect-filter",
    default="",
    help="comma-separated DudeCT case/name/filter substrings; empty runs every case in the selected gate",
  )
  parser.add_argument(
    "--dudect-gate",
    choices=("required", "diagnostic", "all"),
    default=os.environ.get("RSCRYPTO_CT_DUDECT_GATE", "required"),
    help="DudeCT gate class to execute; required is the release evidence gate, diagnostic is non-blocking trace evidence",
  )
  args = parser.parse_args()
  if args.prepare_archive and args.prepare_archive.exists():
    parser.error(f"refusing to overwrite existing evidence: {args.prepare_archive}")

  root = Path(__file__).resolve().parents[2]
  os.environ["RUSTUP_TOOLCHAIN"] = subprocess.check_output(
    python_script(root, "scripts/lib/toolchain.py", "--host"), text=True, cwd=root,
  ).strip()
  target = args.target or host_target(root)
  transferred = None
  transfer_identity = None
  if args.prepare_archive or args.run_archive:
    from transfer import bundle
    from cross_build import TARGETS
    if target not in TARGETS or args.profile != "release" or args.dudect_filter or args.dudect_gate != "required":
      parser.error("CT transfer requires the complete supported cross-compiled release lane")
    if args.threshold != 10.0 or "RSCRYPTO_CT_DUDECT_SAMPLES" in os.environ:
      parser.error("CT transfer requires unchanged manifest sampling and threshold")
    if args.prepare_archive:
      from cross_build import environment
      os.environ.update(environment(target))
      transfer_identity = bundle.source_identity(root)
  host = host_target(root)
  if not args.prepare_archive and not is_host_executable_target(target, host):
    print(
      f"ct-full target must match the physical runner target: requested {target}, host is {host}",
      file=sys.stderr,
    )
    return 2
  configure_target_environment(target, os.environ)

  profile = args.profile
  out_dir = root / "target" / "ct" / target / profile
  full_dir = out_dir / "full"
  logs_dir = full_dir / "logs"
  logs_dir.mkdir(parents=True, exist_ok=True)
  if args.prepare_archive or args.run_archive:
    for name in ("ct-report.json", "ct-report.md"):
      (out_dir / name).unlink(missing_ok=True)

  ct = load_toml(root / "ct.toml")
  all_manifest_cases = manifest_dudect_cases(ct)
  all_gate_manifest_cases = filter_dudect_cases_by_gate(all_manifest_cases, args.dudect_gate)
  filtered_gate_manifest_cases, dudect_filter = filter_dudect_cases(all_gate_manifest_cases, args.dudect_filter)
  target_skipped_cases = [
    case for case in filtered_gate_manifest_cases if not dudect_case_supported_on_target(ct, case, target)
  ]
  gate_manifest_cases = filter_dudect_cases_by_target(ct, all_gate_manifest_cases, target)
  manifest_cases = filter_dudect_cases_by_target(ct, filtered_gate_manifest_cases, target)
  filtered_dudect = bool(dudect_filter)
  if filtered_dudect and not manifest_cases:
    skipped = ", ".join(case["name"] for case in target_skipped_cases)
    target_note = f"; target-unsupported match(es): {skipped}" if skipped else ""
    print(
      f"ct-full: --dudect-filter {args.dudect_filter!r} matched no supported {args.dudect_gate} DudeCT cases"
      f" for {target}{target_note}",
      file=sys.stderr,
    )
    return 2
  binsec_mode, binsec_reason = binsec_policy(ct, target)
  binsec_enabled = binsec_mode == "required"
  steps = []

  if args.run_archive:
    from transfer import consume
    steps, transferred = consume(root, out_dir, args.run_archive.resolve(), target)
  else:
    artifacts_result = run_command(
      root,
      logs_dir,
      "ct-artifacts",
      shell_script(root, "scripts/ct/artifacts.sh", "--target", target, "--profile", profile),
      timeout=None,
    )
    steps.append(result_record(artifacts_result))

    validate_result = run_command(
      root,
      logs_dir,
      "ct-validate-artifacts",
      python_script(root, "scripts/ct/validate.py", "--target", target, "--profile", profile, "--strict-coverage"),
      timeout=None,
    )
    steps.append(result_record(validate_result))
    if artifacts_result.status == "pass" and validate_result.status == "pass":
      cleanup_result = run_command(
        root, logs_dir, "ct-zeroization-sentinel",
        python_script(root, "scripts/ct/zeroization.py", "--artifact-dir", str(out_dir / "artifacts"),
                      "--out", str(out_dir / "zeroization.json")),
      )
      steps.append(result_record(cleanup_result))
  identity = candidate_identity(out_dir)
  if any(step["status"] != "pass" for step in steps):
    for step in steps:
      if step["status"] != "pass":
        print(f"ct-full: stopping after failed gate-one step {step['name']}", file=sys.stderr)
    findings, diagnostics = build_findings(steps, [], [], [])
    asm_report = load_json_if_exists(out_dir / "asm-heuristics.json")
    report = {
      "schema_version": 1,
      "kind": "rscrypto.ct.full-report",
      "crate": "rscrypto",
      **identity,
      "generated_at_utc": now_utc(),
      "target": target,
      "target_triple": target,
      "profile": profile,
      "status": "fail",
      "failure_count": len(findings),
      "summary": summarize_findings(findings, diagnostics),
      "host": {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
      },
      "steps": steps,
      "dudect": {
        "enabled": True,
        "manifest_case_count": len(all_manifest_cases),
        "gate_case_count": len(gate_manifest_cases),
        "selected_case_count": len(manifest_cases),
        "target_skipped_case_count": len(target_skipped_cases),
        "target_skipped_cases": [
          {
            "name": case["name"],
            "primitive": case["primitive"],
            "reason": primitives_by_id(ct)
            .get(case["primitive"], {})
            .get("physical_timing_unsupported_reason", "physical timing evidence unsupported on this target"),
          }
          for case in target_skipped_cases
        ],
        "filter": args.dudect_filter or None,
        "filter_tokens": dudect_filter,
        "gate": args.dudect_gate,
        "coverage_mode": "filtered" if filtered_dudect else args.dudect_gate,
        "samples": "manifest",
        "default_timeout_seconds": args.dudect_timeout,
        "threshold_abs_max_t": args.threshold,
        "smoke": False,
        "cases": [],
        "status_counts": {},
      },
      "binsec": {
        "enabled": binsec_enabled,
        "policy": binsec_mode,
        "reason": binsec_reason,
        "timeout_seconds": args.binsec_timeout,
        "smt_solver": args.binsec_smt_solver,
        "kernels": [],
        "status_counts": {},
      },
      "coverage": {
        "required_dudect_primitives": sorted(primitive_ids_requiring_dudect(ct, target)),
        "manifest_required_dudect_primitives": sorted(required_dudect_primitives(ct, target)),
        "executed_dudect_primitives": [],
        "executed_required_dudect_primitives": [],
        "passing_dudect_primitives": [],
        "passing_required_dudect_primitives": [],
        "missing_dudect_primitives": [],
      },
      "findings": findings,
      "diagnostics": diagnostics,
      "primitive_evidence": build_primitive_evidence(
        ct,
        target,
        [],
        [],
        asm_report,
        binsec_enabled=binsec_enabled,
        coverage_limited=False,
      ),
      "known_findings": [],
      "artifacts": collect_artifact_records(out_dir),
      "notes": [
        "Gate one failed before timing and proof evidence could be completed.",
        "Inspect the listed gate logs first; later gates were not run.",
      ],
    }
    json_path, md_path = write_full_report(out_dir, report)
    print(f"ct-full report: {json_path}")
    print(f"ct-full summary: {md_path}")
    return 1

  if args.prepare_archive:
    from transfer import export
    if binsec_enabled:
      parser.error("CT transfer does not replace a required native BINSEC lane")
    shared = Path(tempfile.mkdtemp(prefix="prepare-", dir=out_dir)) / "shared"
    preparation = run_command(root, logs_dir, "ct-dudect-prepare-transfer", [
      *shell_script(root, "scripts/ct/dudect.sh"), "--prepare-only", "--target", target,
      "--shared-dir", str(shared),
    ])
    if preparation.status != "pass":
      print(f"DudeCT preparation failed: {preparation.stderr_path}", file=sys.stderr)
      return 1
    export(root, out_dir, shared, steps, transfer_identity, args.prepare_archive.resolve(), target)
    print(f"CT preparation complete; native timing remains required: {args.prepare_archive}")
    return 0

  binsec_kernels = []
  if binsec_enabled:
    print("ct-full: binsec", flush=True)
    binsec_result = run_command(
      root,
      logs_dir,
      "ct-binsec",
      [
        *python_script(root, "scripts/ct/binsec.py"),
        "--target",
        target,
        "--profile",
        profile,
        "--timeout",
        str(args.binsec_timeout),
        "--smt-solver",
        args.binsec_smt_solver,
      ],
      timeout=None,
    )
    steps.append(result_record(binsec_result))

    binsec_root = out_dir / "binsec"
    for report_path in sorted(binsec_root.glob("*/binsec-report.json")):
      report = load_json_if_exists(report_path)
      if report is None:
        binsec_kernels.append(
          {
            "kernel": report_path.parent.name,
            "primitive": None,
            "required": True,
            "status": "unknown",
            "report": str(report_path),
            "reason": "invalid or unreadable report",
            "category": "tooling_failure",
          }
        )
        continue
      binsec_kernels.append(
        {
          "kernel": report.get("kernel"),
          "primitive": report.get("primitive"),
          "required": bool(report.get("required", False)),
          "status": report.get("status"),
          "report": str(report_path),
          "reason": report.get("reason"),
          "category": binsec_result_category(report),
          "backend": report.get("backend"),
          "target": report.get("target"),
          "target_triple": report.get("target_triple"),
          "profile": report.get("profile"),
          "crate_version": report.get("crate_version"),
          "git_commit": report.get("git_commit"),
          "git_dirty": report.get("git_dirty"),
          "ct_manifest_sha256": report.get("ct_manifest_sha256"),
          "harness_manifest_sha256": report.get("harness_manifest_sha256"),
          "harness_lockfile_sha256": report.get("harness_lockfile_sha256"),
          "rustc_verbose": report.get("rustc_verbose"),
          "cargo": report.get("cargo"),
          "features": report.get("features"),
          "default_features": report.get("default_features"),
          "profile_settings": report.get("profile_settings"),
          "rustflags": report.get("rustflags"),
          "harness_elf_type": report.get("harness_elf_type"),
          "load_sections": report.get("load_sections"),
          "binsec_version": report.get("binsec_version"),
          "binsec_sha256": report.get("binsec_sha256"),
          "artifacts": report.get("artifacts", {}),
          "artifact_dir": str(report_path.parent.relative_to(out_dir)),
        }
      )
  else:
    steps.append(skipped_step("ct-binsec", binsec_reason))

  dudect_run = None
  dudect_cases = []
  if all(step["status"] in ("pass", "not_applicable") for step in steps):
    dudect_run, preparation, dudect_cases = run_dudect_cases(
      root, out_dir, logs_dir, target, profile, manifest_cases, args.threshold, args.dudect_timeout, transferred,
    )
    steps.append(result_record(preparation))
  else:
    steps.append(skipped_step("ct-dudect", "proof gate failed; timing was not started"))

  executed_dudect = {case["primitive"] for case in dudect_cases}
  executed_required_dudect = {case["primitive"] for case in dudect_cases if case.get("gate") != "diagnostic"}
  passing_dudect = {case["primitive"] for case in dudect_cases if case["status"] == "pass"}
  passing_required_dudect = {
    case["primitive"] for case in dudect_cases if case["status"] == "pass" and case.get("gate") != "diagnostic"
  }
  required_dudect = primitive_ids_requiring_dudect(ct, target)
  manifest_required_dudect = required_dudect_primitives(ct, target)
  missing_dudect = (
    []
    if filtered_dudect or args.dudect_gate != "required"
    else sorted(required_dudect - executed_required_dudect)
  )
  missing_manifest_required_dudect = (
    [] if filtered_dudect or args.dudect_gate != "required" else sorted(required_dudect - manifest_required_dudect)
  )
  missing_dudect = sorted(set(missing_dudect) | set(missing_manifest_required_dudect))

  artifact_records = collect_artifact_records(out_dir, dudect_run)
  if transferred is not None:
    from transfer import KIND, bundle
    original = Path(json.loads(transferred.read_text())["metadata"]["transfer"]["original"])
    bundle.verify(root, original, KIND, target)
  findings, diagnostics = build_findings(steps, dudect_cases, binsec_kernels, missing_dudect)
  asm_report = load_json_if_exists(out_dir / "asm-heuristics.json")

  failure_count = len(findings)
  status = "pass" if failure_count == 0 else "fail"
  report = {
    "schema_version": 1,
    "kind": "rscrypto.ct.full-report",
    "crate": "rscrypto",
    **identity,
    "generated_at_utc": now_utc(),
    "target": target,
    "target_triple": target,
    "profile": profile,
    "status": status,
    "failure_count": failure_count,
    "summary": summarize_findings(findings, diagnostics),
    "host": {
      "system": platform.system(),
      "release": platform.release(),
      "machine": platform.machine(),
      "processor": platform.processor(),
    },
    "steps": steps,
    "dudect": {
      "enabled": True,
      "manifest_case_count": len(all_manifest_cases),
      "gate_case_count": len(gate_manifest_cases),
      "selected_case_count": len(manifest_cases),
      "target_skipped_case_count": len(target_skipped_cases),
      "target_skipped_cases": [
        {
          "name": case["name"],
          "primitive": case["primitive"],
          "reason": primitives_by_id(ct)
          .get(case["primitive"], {})
          .get("physical_timing_unsupported_reason", "physical timing evidence unsupported on this target"),
        }
        for case in target_skipped_cases
      ],
      "filter": args.dudect_filter or None,
      "filter_tokens": dudect_filter,
      "gate": args.dudect_gate,
      "coverage_mode": "filtered" if filtered_dudect else args.dudect_gate,
      "samples": "manifest",
      "default_timeout_seconds": args.dudect_timeout,
      "threshold_abs_max_t": args.threshold,
      "smoke": False,
      "cases": dudect_cases,
      "status_counts": summarize_status(dudect_cases),
    },
    "binsec": {
      "enabled": binsec_enabled,
      "policy": binsec_mode,
      "reason": binsec_reason,
      "timeout_seconds": args.binsec_timeout,
      "smt_solver": args.binsec_smt_solver,
      "kernels": binsec_kernels,
      "status_counts": summarize_status(binsec_kernels),
    },
    "coverage": {
      "required_dudect_primitives": sorted(required_dudect),
      "manifest_required_dudect_primitives": sorted(manifest_required_dudect),
      "executed_dudect_primitives": sorted(executed_dudect),
      "executed_required_dudect_primitives": sorted(executed_required_dudect),
      "passing_dudect_primitives": sorted(passing_dudect),
      "passing_required_dudect_primitives": sorted(passing_required_dudect),
      "missing_dudect_primitives": missing_dudect,
    },
    "findings": findings,
    "diagnostics": diagnostics,
    "primitive_evidence": build_primitive_evidence(
      ct,
      target,
      dudect_cases,
      binsec_kernels,
      asm_report,
      binsec_enabled=binsec_enabled,
      coverage_limited=filtered_dudect or args.dudect_gate != "required",
    ),
    "known_findings": [],
    "artifacts": artifact_records,
    "notes": [
      "This report is an evidence index, not a constant-time proof.",
      "DudeCT passes mean no leakage was detected for the sampled classes and host configuration.",
      "Native BINSEC evidence is required only for targets whose ct.toml binsec policy is required.",
      (
        "Filtered DudeCT runs are diagnostic evidence; skipped manifest cases are not release coverage."
        if filtered_dudect
        else "Uncovered CT-intended primitives remain blockers for ct-claimed release status."
      ),
    ],
  }

  full_dir.mkdir(parents=True, exist_ok=True)
  json_path, md_path = write_full_report(out_dir, report)
  print(f"ct-full report: {json_path}")
  print(f"ct-full summary: {md_path}")
  return 0 if status == "pass" else 1


if __name__ == "__main__":
  raise SystemExit(main())
