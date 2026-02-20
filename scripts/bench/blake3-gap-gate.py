#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ratcheted thresholds: close to current baseline so we lock wins while
# preserving enough headroom for normal runner variance.
DEFAULT_MAX_GAP_PCT = {
  "256": 4.8,
  "1024": 6.8,
  "4096": 7.8,
  "16384": 13.0,
  "65536": 4.5,
}


@dataclass(frozen=True)
class Point:
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
    return (self.throughput_units * 1_000_000_000.0) / self.mean_ns


def _read_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def _extract_throughput(throughput: Any):
  if not isinstance(throughput, dict):
    return None
  if "Bytes" in throughput and isinstance(throughput["Bytes"], (int, float)):
    return ("Bytes", int(throughput["Bytes"]))
  if "Elements" in throughput and isinstance(throughput["Elements"], (int, float)):
    return ("Elements", int(throughput["Elements"]))
  return None


def _extract_mean_ns(estimates: Any):
  if not isinstance(estimates, dict):
    return None
  mean = estimates.get("mean")
  if not isinstance(mean, dict):
    return None
  pe = mean.get("point_estimate")
  if not isinstance(pe, (int, float)):
    return None
  return float(pe)


def iter_points(root: Path):
  by_key: dict[tuple[str, str, str], tuple[int, Point]] = {}
  for dirpath, _d, files in os.walk(root):
    if "benchmark.json" not in files:
      continue

    bench_dir = Path(dirpath)
    bench_json = bench_dir / "benchmark.json"
    estimates_path = bench_dir / "estimates.json"
    priority = None
    if estimates_path.exists():
      if bench_dir.name == "new":
        priority = 2
      elif bench_dir.name == "base":
        priority = 1
      else:
        priority = 0
    else:
      new_est = bench_dir / "new" / "estimates.json"
      base_est = bench_dir / "base" / "estimates.json"
      if new_est.exists():
        estimates_path = new_est
        priority = 2
      elif base_est.exists():
        estimates_path = base_est
        priority = 1
      else:
        continue

    bench = _read_json(bench_json)
    estimates = _read_json(estimates_path)
    tp = _extract_throughput(bench.get("throughput"))
    mean_ns = _extract_mean_ns(estimates)
    if tp is None or mean_ns is None:
      continue

    group_id = bench.get("group_id")
    function_id = bench.get("function_id")
    value_str = bench.get("value_str")
    if not isinstance(group_id, str) or not isinstance(function_id, str) or not isinstance(value_str, str):
      continue

    p = Point(
      group_id=group_id,
      function_id=function_id,
      value_str=value_str,
      throughput_kind=tp[0],
      throughput_units=tp[1],
      mean_ns=mean_ns,
    )

    key = (group_id, function_id, value_str)
    prev = by_key.get(key)
    if prev is None or priority > prev[0]:
      by_key[key] = (priority, p)

  for _prio, p in by_key.values():
    yield p


def parse_case_thresholds(raw: str) -> dict[str, float]:
  out: dict[str, float] = {}
  if not raw.strip():
    return out
  for token in raw.split(","):
    item = token.strip()
    if not item:
      continue
    if "=" not in item:
      raise ValueError(f"invalid threshold token '{item}' (expected SIZE=PCT)")
    case, pct = item.split("=", 1)
    case = case.strip()
    pct = pct.strip()
    if not case or not pct:
      raise ValueError(f"invalid threshold token '{item}' (expected SIZE=PCT)")
    out[case] = float(pct)
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description="BLAKE3 oneshot performance gap gate")
  ap.add_argument("--root", default="target/criterion")
  ap.add_argument("--group", default="blake3/oneshot")
  ap.add_argument("--ours", default="rscrypto")
  ap.add_argument("--ours-prefix", default="")
  ap.add_argument("--rival", default="official")
  ap.add_argument("--max-gap-case", default="")
  ap.add_argument("--label", default="")
  args = ap.parse_args()

  root = Path(args.root)
  if not root.exists():
    print(f"error: missing criterion root: {root}", file=sys.stderr)
    return 2

  points = [p for p in iter_points(root) if p.group_id == args.group]
  if not points:
    print(f"error: no points for group '{args.group}' in {root}", file=sys.stderr)
    return 2

  by_case: dict[str, dict[str, Point]] = {}
  for p in points:
    by_case.setdefault(p.value_str, {})[p.function_id] = p

  failures: list[str] = []
  gate_label = args.label.strip() if args.label.strip() else args.group
  lines: list[str] = [f"BLAKE3 gap gate ({gate_label}):"]
  thresholds = dict(DEFAULT_MAX_GAP_PCT)
  if args.max_gap_case.strip():
    try:
      thresholds.update(parse_case_thresholds(args.max_gap_case))
    except ValueError as e:
      print(f"error: {e}", file=sys.stderr)
      return 2

  for case, max_gap in sorted(thresholds.items(), key=lambda kv: int(kv[0])):
    case_points = by_case.get(case, {})
    ours = None
    ours_name = args.ours
    if args.ours_prefix:
      prefix = args.ours_prefix
      candidates = [p for name, p in case_points.items() if name.startswith(prefix)]
      if candidates:
        ours = max(candidates, key=lambda p: p.units_per_sec)
        ours_name = ours.function_id
    if ours is None:
      ours = case_points.get(args.ours)
    rival = case_points.get(args.rival)
    if ours is None or rival is None:
      ours_need = f"{args.ours_prefix}*" if args.ours_prefix else args.ours
      failures.append(f"missing data for size={case} (need {ours_need} and {args.rival})")
      continue

    if ours.units_per_sec <= 0.0:
      failures.append(f"invalid throughput for size={case} ({ours_name})")
      continue

    need_pct = (rival.units_per_sec / ours.units_per_sec - 1.0) * 100.0
    lines.append(
      f"  size={case}: ours={ours_name} need=+{need_pct:.2f}% (limit +{max_gap:.2f}%)"
    )
    if need_pct > max_gap:
      failures.append(
        f"size={case} gap too large: need +{need_pct:.2f}% vs allowed +{max_gap:.2f}%"
      )

  print("\n".join(lines))

  if failures:
    print("\nGate failed:", file=sys.stderr)
    for f in failures:
      print(f"  - {f}", file=sys.stderr)
    return 1

  print("Gate passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
