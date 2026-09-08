#!/usr/bin/env python3
"""Execute a prepared DudeCT binary without rebuilding or copying its artifacts."""

import argparse
import copy
import json
import os
import subprocess
from pathlib import Path

from dudect_report import case_report, write_report
from manifest import dudect_sample_count


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prepared", required=True, type=Path)
  parser.add_argument("--evidence-dir", required=True, type=Path)
  parser.add_argument("--samples", type=int)
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--threshold", type=float, default=10.0)
  parser.add_argument("--filter", default="")
  parser.add_argument("--timeout", type=int)
  parser.add_argument("--latest", type=Path)
  args = parser.parse_args()
  prepared = json.loads(args.prepared.read_text())
  if args.smoke:
    selected = {name: case for name, case in prepared["manifest_cases"].items() if args.filter in name}
    if not selected:
      parser.error("DudeCT filter selects no manifest cases")
    budgets = {name: dudect_sample_count(case, smoke=True, override=args.samples) for name, case in selected.items()}
    reports = []
    status = 0
    for index, (name, samples) in enumerate(budgets.items()):
      case_args = copy.copy(args)
      case_args.filter = name
      case_args.samples = samples
      case_args.evidence_dir = args.evidence_dir / str(index)
      case_args.latest = None
      result = measure(prepared, case_args)
      if result > 1:
        return result
      status |= result
      reports.append(json.loads((case_args.evidence_dir / "dudect-report.json").read_text()))
    summary = {**prepared["metadata"], "smoke": True, "requested_samples_by_case": budgets,
               "threshold_abs_max_t": args.threshold, "filter": args.filter,
               "cases": [case for report in reports for case in report["cases"]],
               "case_count": sum(report["case_count"] for report in reports),
               "failure_count": sum(report["failure_count"] for report in reports),
               "diagnostic_failure_count": sum(report["diagnostic_failure_count"] for report in reports),
               "measurements": [str(args.evidence_dir / str(index) / "dudect-report.json") for index in range(len(reports))]}
    write_report(args.evidence_dir / "dudect-report.json", summary)
    if args.latest is not None:
      write_report(args.latest, summary)
    return status
  args.samples = dudect_sample_count({}, override=args.samples)
  return measure(prepared, args)


def measure(prepared, args):
  args.evidence_dir.mkdir(parents=True, exist_ok=True)
  args.stdout = args.evidence_dir / "dudect.stdout.txt"
  args.csv = args.evidence_dir / "dudect-raw.csv"
  report_path = args.evidence_dir / "dudect-report.json"
  for path in (args.stdout, args.csv, report_path):
    if path.exists():
      raise ValueError(f"refusing to reuse measurement evidence: {path}")
  command = [prepared["metadata"]["binary"]["path"], "--out", str(args.csv)]
  if args.filter:
    command += ["--filter", args.filter]
  args.command = f"RSCRYPTO_CT_DUDECT_SAMPLES={args.samples} " + " ".join(command)
  print(args.command, flush=True)
  with args.stdout.open("w") as stdout:
    try:
      result = subprocess.run(
        command, stdout=stdout, check=False, timeout=args.timeout,
        env={**os.environ, "RSCRYPTO_CT_DUDECT_SAMPLES": str(args.samples)},
      )
    except subprocess.TimeoutExpired:
      print("DudeCT measurement timed out", flush=True)
      return 124
  print(args.stdout.read_text(), end="", flush=True)
  # A binary failure is a tooling failure, even if it emitted complete samples.
  if result.returncode != 0:
    print(f"DudeCT measurement command failed: exit {result.returncode}", flush=True)
    return 2
  report = case_report(prepared, args)
  report["prepared"] = str(args.prepared)
  report["smoke"] = args.smoke
  report["measurement_returncode"] = result.returncode
  write_report(report_path, report)
  if args.latest is not None:
    write_report(args.latest, report)
  print(f"dudect report: {report_path}", flush=True)
  return 1 if report["failure_count"] else 0


if __name__ == "__main__":
  raise SystemExit(main())
