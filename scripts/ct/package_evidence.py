#!/usr/bin/env python3
"""Package compact CT evidence for CI artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
  if not path.exists():
    return None
  try:
    return json.loads(path.read_text())
  except json.JSONDecodeError:
    return None


def copy_file(source: Path, destination: Path) -> bool:
  if not source.exists() or not source.is_file():
    return False
  destination.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(source, destination)
  return True


def resolve_path(root: Path, value: str | None) -> Path | None:
  if not value:
    return None
  path = Path(value)
  return path if path.is_absolute() else root / path


def copy_command_logs(root: Path, row: dict[str, Any], destination: Path, stem: str) -> None:
  command = row.get("command_result", row)
  for key in ("stdout", "stderr"):
    source = resolve_path(root, command.get(key))
    if source is not None:
      copy_file(source, destination / "logs" / f"{stem}.{key}.txt")


def package_component_reports(root: Path, report: dict[str, Any], destination: Path) -> None:
  for step in report.get("steps", []):
    if step.get("status") in {"fail", "timeout"}:
      copy_command_logs(root, step, destination, str(step.get("name", "step")))

  for case in report.get("dudect", {}).get("cases", []):
    if case.get("status") == "pass":
      continue
    name = str(case.get("name", "unknown"))
    source = resolve_path(root, case.get("report"))
    if source is not None:
      copy_file(source, destination / "failed-reports" / "dudect" / f"{name}.json")
    copy_command_logs(root, case, destination, f"dudect-{name}")

  for kernel in report.get("binsec", {}).get("kernels", []):
    if kernel.get("status") == "secure":
      continue
    name = str(kernel.get("kernel") or "unknown").replace("/", "_").replace(":", "_")
    source = resolve_path(root, kernel.get("report"))
    if source is not None:
      copy_file(source, destination / "failed-reports" / "binsec" / f"{name}.json")


def write_readme(destination: Path, suffix: str, report: dict[str, Any] | None, raw: bool) -> None:
  lines = [
    "# rscrypto CT Evidence",
    "",
    f"- Lane: `{suffix}`",
    f"- Raw target artifacts: `{'included' if raw else 'not included'}`",
  ]
  if report is not None:
    summary = report.get("summary", {})
    lines.extend(
      [
        f"- Target: `{report.get('target')}`",
        f"- Status: `{report.get('status')}`",
        f"- Blocking findings: `{summary.get('blockers', report.get('failure_count', 0))}`",
        f"- Non-blocking diagnostics: `{summary.get('diagnostics', 0)}`",
        "",
        f"Read `ct-report-{suffix}.md` first. Failed or inconclusive component reports are under `failed-reports/`.",
      ]
    )
  else:
    lines.extend(["", "No `ct-report.json` was produced. Inspect `logs/` and the top-level `ct-full` log."])
  lines.append("")
  (destination / "README.md").write_text("\n".join(lines))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", required=True)
  parser.add_argument("--profile", default="release")
  parser.add_argument("--suffix", required=True)
  parser.add_argument("--out-dir", required=True, type=Path)
  parser.add_argument("--raw", action="store_true", help="include raw target/ct artifacts")
  args = parser.parse_args()

  root = Path.cwd()
  destination = args.out_dir
  destination.mkdir(parents=True, exist_ok=True)
  target_dir = root / "target" / "ct" / args.target / args.profile
  report_json = target_dir / "ct-report.json"
  report_md = target_dir / "ct-report.md"
  report = load_json(report_json)

  copy_file(report_json, destination / f"ct-report-{args.suffix}.json")
  copy_file(report_md, destination / f"ct-report-{args.suffix}.md")

  if report is not None:
    package_component_reports(root, report, destination)
  else:
    logs_dir = target_dir / "full" / "logs"
    if logs_dir.exists():
      for path in sorted(logs_dir.glob("*.txt")):
        copy_file(path, destination / "logs" / path.name)

  if args.raw and target_dir.exists():
    archive_base = destination / f"target-ct-raw-{args.suffix}"
    shutil.make_archive(str(archive_base), "gztar", root_dir=root, base_dir=target_dir.relative_to(root))

  write_readme(destination, args.suffix, report, args.raw)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
