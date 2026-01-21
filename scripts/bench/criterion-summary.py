#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


GIB = 1024.0 * 1024.0 * 1024.0


@dataclass(frozen=True)
class CriterionPoint:
  group_id: str
  function_id: str
  value_str: str
  throughput_kind: str
  throughput_units: int
  mean_ns: float

  @property
  def units_per_sec(self) -> float:
    if self.mean_ns <= 0.0:
      return 0.0
    return (float(self.throughput_units) * 1_000_000_000.0) / self.mean_ns

  @property
  def gib_per_sec(self) -> float:
    if self.throughput_kind != "Bytes":
      return 0.0
    return self.units_per_sec / GIB


def _read_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def _extract_throughput(throughput: Any) -> Optional[tuple[str, int]]:
  if not isinstance(throughput, dict):
    return None
  if "Bytes" in throughput:
    v = throughput["Bytes"]
    if isinstance(v, (int, float)):
      return ("Bytes", int(v))
  if "Elements" in throughput:
    v = throughput["Elements"]
    if isinstance(v, (int, float)):
      return ("Elements", int(v))
  return None


def _extract_mean_ns(estimates: Any) -> Optional[float]:
  if not isinstance(estimates, dict):
    return None
  mean = estimates.get("mean")
  if not isinstance(mean, dict):
    return None
  pe = mean.get("point_estimate")
  if not isinstance(pe, (int, float)):
    return None
  return float(pe)


def iter_criterion_points(root: Path) -> Iterator[CriterionPoint]:
  # Criterion output layout differs across versions:
  # - Old (Criterion <= 0.4-ish): <bench>/benchmark.json and <bench>/{new,base}/estimates.json
  # - New (Criterion 0.5+): <group>/<function>/<value>/{new,base}/benchmark.json + estimates.json
  #
  # We support both and prefer `new` when both exist.
  points_by_key: dict[tuple[str, str, str], tuple[int, CriterionPoint]] = {}

  for dirpath, _dirnames, filenames in os.walk(root):
    if "benchmark.json" not in filenames:
      continue

    bench_dir = Path(dirpath)
    bench_json_path = bench_dir / "benchmark.json"

    # New layout: benchmark.json + estimates.json live together (under new/ or base/).
    estimates_path = bench_dir / "estimates.json"
    priority: int | None = None
    if estimates_path.exists():
      if bench_dir.name == "new":
        priority = 2
      elif bench_dir.name == "base":
        priority = 1
      else:
        priority = 0
    else:
      # Old layout: estimates live in subdirs.
      new_estimates_path = bench_dir / "new" / "estimates.json"
      base_estimates_path = bench_dir / "base" / "estimates.json"
      if new_estimates_path.exists():
        estimates_path = new_estimates_path
        priority = 2
      elif base_estimates_path.exists():
        estimates_path = base_estimates_path
        priority = 1
      else:
        continue

    bench = _read_json(bench_json_path)
    estimates = _read_json(estimates_path)

    group_id = bench.get("group_id")
    function_id = bench.get("function_id")
    value_str = bench.get("value_str")
    throughput = _extract_throughput(bench.get("throughput"))
    mean_ns = _extract_mean_ns(estimates)

    if not isinstance(group_id, str) or not isinstance(function_id, str) or not isinstance(value_str, str):
      continue
    if throughput is None or mean_ns is None:
      continue

    throughput_kind, throughput_units = throughput
    point = CriterionPoint(
      group_id=group_id,
      function_id=function_id,
      value_str=value_str,
      throughput_kind=throughput_kind,
      throughput_units=throughput_units,
      mean_ns=mean_ns,
    )

    key = (group_id, function_id, value_str)
    prev = points_by_key.get(key)
    if prev is None or priority > prev[0]:
      points_by_key[key] = (priority, point)

  for _priority, point in points_by_key.values():
    yield point


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
  for p in patterns:
    if p and p in text:
      return True
  return False


def format_tsv(points: list[CriterionPoint]) -> str:
  lines: list[str] = []
  lines.append("group\tcase\timpl\tthrpt_kind\tthrpt_units\tgib_per_s\tunits_per_s")
  for p in points:
    lines.append(
      f"{p.group_id}\t{p.value_str}\t{p.function_id}\t{p.throughput_kind}\t{p.throughput_units}\t{p.gib_per_sec:.6f}\t{p.units_per_sec:.3f}"
    )
  return "\n".join(lines) + "\n"


def format_non_wins(
  points: list[CriterionPoint],
  ours_prefix: str,
  exclude_patterns: list[str],
  min_improvement_pct: float,
) -> str:
  by_case: dict[tuple[str, str], list[CriterionPoint]] = {}
  for p in points:
    by_case.setdefault((p.group_id, p.value_str), []).append(p)

  lines: list[str] = []
  for (group_id, case), entries in sorted(by_case.items()):
    entries = [e for e in entries if not _matches_any(e.function_id, exclude_patterns)]
    ours = [e for e in entries if e.function_id.startswith(ours_prefix)]
    if not ours:
      continue

    ours_best = max(ours, key=lambda e: e.units_per_sec)
    faster = [
      e
      for e in entries
      if e.units_per_sec > ours_best.units_per_sec and not e.function_id.startswith(ours_prefix)
    ]
    if not faster:
      continue

    faster.sort(key=lambda e: e.units_per_sec, reverse=True)
    top = faster[0]
    need_pct = (top.units_per_sec / ours_best.units_per_sec - 1.0) * 100.0 if ours_best.units_per_sec > 0.0 else 0.0
    if need_pct < min_improvement_pct:
      continue

    ours_unit = "GiB/s" if ours_best.throughput_kind == "Bytes" else "Melem/s"
    top_unit = "GiB/s" if top.throughput_kind == "Bytes" else "Melem/s"
    ours_rate = ours_best.gib_per_sec if ours_best.throughput_kind == "Bytes" else (ours_best.units_per_sec / 1_000_000.0)
    top_rate = top.gib_per_sec if top.throughput_kind == "Bytes" else (top.units_per_sec / 1_000_000.0)

    lines.append(
      f"{group_id}\t{case}\tours={ours_best.function_id}\t{ours_rate:.3f} {ours_unit}\t"
      f"best={top.function_id}\t{top_rate:.3f} {top_unit}\tneed=+{need_pct:.1f}%"
    )
    for e in faster:
      pct = (e.units_per_sec / ours_best.units_per_sec - 1.0) * 100.0 if ours_best.units_per_sec > 0.0 else 0.0
      e_unit = "GiB/s" if e.throughput_kind == "Bytes" else "Melem/s"
      e_rate = e.gib_per_sec if e.throughput_kind == "Bytes" else (e.units_per_sec / 1_000_000.0)
      lines.append(f"  vs {e.function_id}\t{e_rate:.3f} {e_unit}\t(+{pct:.1f}%)")

  return "\n".join(lines) + ("\n" if lines else "")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description="Summarize Criterion throughput and report non-wins.")
  parser.add_argument("--root", default="target/criterion", help="Criterion output directory (default: target/criterion)")
  parser.add_argument("--group-prefix", default="", help="Only include groups containing this string (e.g. crc64/xz)")
  parser.add_argument(
    "--only",
    default="oneshot",
    choices=["oneshot", "all"],
    help="Filter set (default: oneshot excludes rscrypto/buffered)",
  )
  parser.add_argument("--non-wins", action="store_true", help="Print non-wins report instead of TSV")
  parser.add_argument("--ours", default="rscrypto/checksum", help="Prefix identifying our implementation")
  parser.add_argument(
    "--exclude",
    action="append",
    default=[],
    help="Exclude implementations containing this substring (can be repeated)",
  )
  parser.add_argument(
    "--min-improvement-pct",
    type=float,
    default=0.0,
    help="Only report cases needing at least this improvement percent",
  )
  args = parser.parse_args(argv)

  root = Path(args.root)
  if not root.exists():
    print(f"error: criterion root does not exist: {root}", file=sys.stderr)
    return 2

  exclude_patterns = list(args.exclude)
  if args.only == "oneshot":
    exclude_patterns.append("rscrypto/buffered")

  points = [
    p
    for p in iter_criterion_points(root)
    if (not args.group_prefix or args.group_prefix in p.group_id)
  ]
  points.sort(key=lambda p: (p.group_id, p.value_str, p.function_id))

  if args.non_wins:
    sys.stdout.write(format_non_wins(points, args.ours, exclude_patterns, args.min_improvement_pct))
  else:
    sys.stdout.write(format_tsv(points))
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
