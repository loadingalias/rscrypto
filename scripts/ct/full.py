#!/usr/bin/env python3
"""Run the full rscrypto CT evidence pipeline and emit release-style reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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


def load_toml(path: Path) -> dict[str, Any]:
  with path.open("rb") as fh:
    return tomllib.load(fh)


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def host_target(root: Path) -> str:
  verbose = subprocess.check_output(["rustc", "-vV"], cwd=root, text=True)
  return next(line.split(":", 1)[1].strip() for line in verbose.splitlines() if line.startswith("host:"))


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


def primitive_ids_requiring_dudect(ct: dict[str, Any]) -> set[str]:
  ids = set()
  for primitive in ct.get("primitive", []):
    if primitive.get("claim") != "ct-intended":
      continue
    required = set()
    for profile_name in primitive.get("required", []):
      profile = ct.get("evidence", {}).get("profile", {}).get(profile_name, {})
      required.update(profile.get("required", []))
    if "dudect" in required:
      ids.add(primitive.get("id", ""))
  ids.discard("")
  return ids


def manifest_dudect_cases(ct: dict[str, Any]) -> list[dict[str, Any]]:
  cases = []
  for case in ct.get("dudect_case", []):
    missing = [key for key in ("name", "primitive", "filter") if not case.get(key)]
    if missing:
      raise ValueError(f"dudect_case missing required keys {missing}: {case!r}")
    cases.append(case)
  return cases


def case_sample_count(case: dict[str, Any], *, smoke: bool, override: int | None, fallback: int) -> int:
  if override is not None:
    return override
  key = "smoke_samples" if smoke else "samples"
  value = case.get(key, fallback)
  return int(value)


def file_record(path: Path, base: Path, kind: str) -> dict[str, Any]:
  return {
    "path": str(path.relative_to(base)),
    "kind": kind,
    "sha256": sha256_file(path),
    "bytes": path.stat().st_size,
  }


def copy_latest_dudect_outputs(out_dir: Path, case_name: str) -> list[dict[str, str]]:
  dudect_dir = out_dir / "dudect"
  cases_dir = dudect_dir / "cases" / case_name
  cases_dir.mkdir(parents=True, exist_ok=True)
  copied = []
  for source_name, target_name in (
    ("dudect-report.json", "dudect-report.json"),
    ("dudect-raw.csv", "dudect-raw.csv"),
    ("dudect.stdout.txt", "dudect.stdout.txt"),
  ):
    source = dudect_dir / source_name
    if source.exists():
      target = cases_dir / target_name
      shutil.copy2(source, target)
      copied.append({"name": target_name, "path": str(target)})
  return copied


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
  if not path.exists():
    return None
  try:
    return json.loads(path.read_text())
  except json.JSONDecodeError:
    return None


def dudect_case_result(
  root: Path,
  out_dir: Path,
  logs_dir: Path,
  target: str,
  profile: str,
  samples: int,
  threshold: float,
  smoke: bool,
  case: dict[str, Any],
  timeout: int | None,
) -> dict[str, Any]:
  command = [
    str(root / "scripts" / "ct" / "dudect.sh"),
    "--target",
    target,
    "--profile",
    profile,
    "--samples",
    str(samples),
    "--threshold",
    str(threshold),
    "--filter",
    case["filter"],
  ]
  if smoke:
    command.append("--smoke")
  result = run_command(root, logs_dir, f"dudect-{case['name']}", command, timeout=timeout)
  copied = copy_latest_dudect_outputs(out_dir, case["name"])
  report_path = out_dir / "dudect" / "cases" / case["name"] / "dudect-report.json"
  report = load_json_if_exists(report_path)
  status = result.status
  failure_count = None
  if report is not None:
    failure_count = report.get("failure_count")
    if failure_count:
      status = "fail"

  return {
    "name": case["name"],
    "primitive": case["primitive"],
    "filter": case["filter"],
    "status": status,
    "requested_samples": samples,
    "failure_count": failure_count,
    "command_result": result_record(result),
    "artifacts": copied,
    "report": str(report_path) if report_path.exists() else None,
  }


def known_findings(target: str) -> list[dict[str, str]]:
  return []


def markdown_report(report: dict[str, Any]) -> str:
  lines = [
    "# rscrypto CT Report",
    "",
    f"- Generated: `{report['generated_at_utc']}`",
    f"- Target: `{report['target']}`",
    f"- Profile: `{report['profile']}`",
    f"- Status: `{report['status']}`",
    f"- Failure count: `{report['failure_count']}`",
    "",
    "## Evidence Steps",
    "",
  ]
  for step in report["steps"]:
    lines.append(f"- `{step['name']}`: `{step['status']}`")
  lines.extend(["", "## DudeCT Cases", ""])
  for case in report["dudect"]["cases"]:
    detail = f", failures={case['failure_count']}" if case.get("failure_count") is not None else ""
    lines.append(f"- `{case['name']}` (`{case['primitive']}`): `{case['status']}`{detail}")
  if report["coverage"]["missing_dudect_primitives"]:
    lines.extend(["", "## Missing DudeCT Coverage", ""])
    for primitive in report["coverage"]["missing_dudect_primitives"]:
      lines.append(f"- `{primitive}`")
  if report["known_findings"]:
    lines.extend(["", "## Known Findings", ""])
    for finding in report["known_findings"]:
      lines.append(f"- `{finding['severity']}` `{finding['id']}`: {finding['summary']}")
  lines.extend(["", "## Artifacts", ""])
  for artifact in report["artifacts"]:
    lines.append(f"- `{artifact['kind']}`: `{artifact['path']}`")
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", default=None)
  parser.add_argument("--profile", default="release")
  parser.add_argument("--samples", type=int, default=None, help="override manifest dudect sample counts")
  parser.add_argument("--threshold", type=float, default=float(os.environ.get("RSCRYPTO_CT_DUDECT_THRESHOLD", "10.0")))
  parser.add_argument("--smoke", action="store_true", help="run dudect with smoke sample count")
  parser.add_argument("--skip-dudect", action="store_true")
  parser.add_argument("--dudect-timeout", type=int, default=300)
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[2]
  target = args.target or host_target(root)
  profile = args.profile
  out_dir = root / "target" / "ct" / target / profile
  full_dir = out_dir / "full"
  logs_dir = full_dir / "logs"
  logs_dir.mkdir(parents=True, exist_ok=True)

  ct = load_toml(root / "ct.toml")
  manifest_cases = manifest_dudect_cases(ct)
  steps = []

  artifacts_result = run_command(
    root,
    logs_dir,
    "ct-artifacts",
    [str(root / "scripts" / "ct" / "artifacts.sh"), "--target", target, "--profile", profile],
    timeout=None,
  )
  steps.append(result_record(artifacts_result))

  validate_result = run_command(
    root,
    logs_dir,
    "ct-validate-artifacts",
    [str(root / "scripts" / "ct" / "validate.py"), "--target", target, "--profile", profile],
    timeout=None,
  )
  steps.append(result_record(validate_result))

  dudect_cases = []
  if not args.skip_dudect:
    fallback_samples = int(os.environ.get("RSCRYPTO_CT_DUDECT_SAMPLES", "20000"))
    for case in manifest_cases:
      samples = case_sample_count(case, smoke=args.smoke, override=args.samples, fallback=fallback_samples)
      print(f"ct-full: dudect {case['name']}", flush=True)
      dudect_cases.append(
        dudect_case_result(
          root,
          out_dir,
          logs_dir,
          target,
          profile,
          samples,
          args.threshold,
          args.smoke,
          case,
          args.dudect_timeout,
        )
      )

  executed_dudect = {case["primitive"] for case in dudect_cases}
  passing_dudect = {case["primitive"] for case in dudect_cases if case["status"] == "pass"}
  required_dudect = primitive_ids_requiring_dudect(ct)
  missing_dudect = sorted(required_dudect - executed_dudect)

  artifact_records = []
  for relative, kind in (
    ("provenance.json", "provenance"),
    ("evidence-index.json", "evidence_index"),
    ("asm-heuristics.json", "asm_heuristics"),
  ):
    path = out_dir / relative
    if path.exists():
      artifact_records.append(file_record(path, out_dir, kind))

  for path in sorted((out_dir / "dudect" / "cases").glob("*/dudect-report.json")):
    artifact_records.append(file_record(path, out_dir, "dudect_report"))

  findings = []
  for step in steps:
    if step["status"] != "pass":
      findings.append({"kind": "step_failure", "severity": "blocker", "summary": f"{step['name']} failed"})
  for case in dudect_cases:
    if case["status"] != "pass":
      findings.append(
        {
          "kind": "dudect_failure",
          "severity": "blocker",
          "summary": f"{case['name']} did not pass",
          "primitive": case["primitive"],
        }
      )
  for primitive in missing_dudect:
    findings.append(
      {
        "kind": "missing_dudect",
        "severity": "blocker",
        "summary": f"{primitive} requires dudect evidence but has no executed manifest case",
        "primitive": primitive,
      }
    )

  known = known_findings(target)
  failure_count = len(findings)
  status = "pass" if failure_count == 0 else "fail"
  report = {
    "schema_version": 1,
    "kind": "rscrypto.ct.full-report",
    "crate": "rscrypto",
    "generated_at_utc": now_utc(),
    "target": target,
    "target_triple": target,
    "profile": profile,
    "status": status,
    "failure_count": failure_count,
    "host": {
      "system": platform.system(),
      "release": platform.release(),
      "machine": platform.machine(),
      "processor": platform.processor(),
    },
    "steps": steps,
    "dudect": {
      "enabled": not args.skip_dudect,
      "manifest_case_count": len(manifest_cases),
      "samples": args.samples,
      "threshold_abs_max_t": args.threshold,
      "smoke": args.smoke,
      "cases": dudect_cases,
    },
    "coverage": {
      "required_dudect_primitives": sorted(required_dudect),
      "executed_dudect_primitives": sorted(executed_dudect),
      "passing_dudect_primitives": sorted(passing_dudect),
      "missing_dudect_primitives": missing_dudect,
    },
    "findings": findings,
    "known_findings": known,
    "artifacts": artifact_records,
    "notes": [
      "This report is an evidence index, not a constant-time proof.",
      "DudeCT passes mean no leakage was detected for the sampled classes and host configuration.",
      "Uncovered CT-intended primitives remain blockers for ct-claimed release status.",
    ],
  }

  full_dir.mkdir(parents=True, exist_ok=True)
  json_path = out_dir / "ct-report.json"
  md_path = out_dir / "ct-report.md"
  json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  md_path.write_text(markdown_report(report))
  print(f"ct-full report: {json_path}")
  print(f"ct-full summary: {md_path}")
  return 0 if status == "pass" else 1


if __name__ == "__main__":
  raise SystemExit(main())
