#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Metric:
  mid: float
  unit: str
  normalized: float


@dataclass(frozen=True)
class Entry:
  bench_id: str
  case_key: str
  impl: str
  metric: Metric


_BENCH_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+$")
_TRIPLE_RE = re.compile(r"\[\s*([0-9.]+)\s*([^\s\]]+)\s+([0-9.]+)\s*([^\s\]]+)\s+([0-9.]+)\s*([^\s\]]+)\s*\]")


def _throughput_factor(unit: str) -> Optional[float]:
  u = unit.strip()
  if u.endswith("/s"):
    u = u[:-2]

  if u == "B":
    return 1.0
  if u == "KB":
    return 1_000.0
  if u == "MB":
    return 1_000_000.0
  if u == "GB":
    return 1_000_000_000.0
  if u == "TB":
    return 1_000_000_000_000.0

  if u == "KiB":
    return 1024.0
  if u == "MiB":
    return 1024.0**2
  if u == "GiB":
    return 1024.0**3
  if u == "TiB":
    return 1024.0**4

  if u == "elem":
    return 1.0
  if u == "Kelem":
    return 1_000.0
  if u == "Melem":
    return 1_000_000.0
  if u == "Gelem":
    return 1_000_000_000.0
  if u == "Telem":
    return 1_000_000_000_000.0

  return None


def _time_factor_to_ns(unit: str) -> Optional[float]:
  u = unit.strip()
  if u == "ns":
    return 1.0
  if u in ("µs", "us"):
    return 1_000.0
  if u == "ms":
    return 1_000_000.0
  if u == "s":
    return 1_000_000_000.0
  return None


def _parse_metric_line(line: str, kind: str) -> Optional[Metric]:
  m = _TRIPLE_RE.search(line)
  if not m:
    return None

  mid = float(m.group(3))
  unit = m.group(4)

  if kind == "thrpt":
    factor = _throughput_factor(unit)
    if factor is None:
      return None
    return Metric(mid=mid, unit=unit, normalized=mid * factor)

  if kind == "time":
    factor = _time_factor_to_ns(unit)
    if factor is None:
      return None
    mid_ns = mid * factor
    normalized = 0.0 if mid_ns <= 0.0 else (1.0 / mid_ns)
    return Metric(mid=mid, unit=unit, normalized=normalized)

  return None


def _case_key_and_impl(bench_id: str) -> Optional[tuple[str, str]]:
  parts = bench_id.split("/")
  if len(parts) < 3:
    return None
  impl = parts[-2]
  case_key = "/".join(parts[:-2] + [parts[-1]])
  return (case_key, impl)


def iter_entries(text: str, prefer: str) -> list[Entry]:
  entries: list[Entry] = []

  current_id: Optional[str] = None
  current_thrpt: Optional[Metric] = None
  current_time: Optional[Metric] = None

  def flush() -> None:
    nonlocal current_id, current_thrpt, current_time
    if not current_id:
      return
    parsed = _case_key_and_impl(current_id)
    if not parsed:
      current_id = None
      current_thrpt = None
      current_time = None
      return

    case_key, impl = parsed
    metric = None
    if prefer == "thrpt":
      metric = current_thrpt or current_time
    else:
      metric = current_time or current_thrpt
    if metric is None:
      current_id = None
      current_thrpt = None
      current_time = None
      return

    entries.append(Entry(bench_id=current_id, case_key=case_key, impl=impl, metric=metric))
    current_id = None
    current_thrpt = None
    current_time = None

  for raw in text.splitlines():
    line = raw.strip()
    if not line:
      continue

    if _BENCH_ID_RE.match(line):
      flush()
      current_id = line
      continue

    if current_id:
      if line.startswith("thrpt:"):
        m = _parse_metric_line(line, "thrpt")
        if m:
          current_thrpt = m
      elif line.startswith("time:"):
        m = _parse_metric_line(line, "time")
        if m:
          current_time = m

  flush()
  return entries


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description="Report cases where rscrypto is not the fastest in a Criterion bench stdout dump.")
  parser.add_argument("path", help="Path to saved `cargo bench` output (Criterion stdout)")
  parser.add_argument("--ours", default="rscrypto", help="Implementation name to treat as ours (default: rscrypto)")
  parser.add_argument(
    "--prefer",
    default="thrpt",
    choices=["thrpt", "time"],
    help="Prefer comparing throughput or time if both are present (default: thrpt)",
  )
  args = parser.parse_args(argv)

  path = Path(args.path)
  if not path.exists():
    print(f"error: file does not exist: {path}", file=sys.stderr)
    return 2

  text = path.read_text(encoding="utf-8", errors="replace")
  entries = iter_entries(text, prefer=args.prefer)

  by_case: dict[str, list[Entry]] = {}
  for e in entries:
    by_case.setdefault(e.case_key, []).append(e)

  losses: list[str] = []
  for case_key in sorted(by_case.keys()):
    case_entries = by_case[case_key]
    ours = [e for e in case_entries if e.impl == args.ours]
    if not ours:
      continue

    ours_best = max(ours, key=lambda e: e.metric.normalized)
    best = max(case_entries, key=lambda e: e.metric.normalized)
    if best.impl == args.ours:
      continue

    ours_v = ours_best.metric.normalized
    best_v = best.metric.normalized
    need_pct = (best_v / ours_v - 1.0) * 100.0 if ours_v > 0.0 else 0.0

    losses.append(
      f"{case_key} {args.ours} {ours_best.metric.mid:g} {ours_best.metric.unit} < {best.impl} {best.metric.mid:g} {best.metric.unit} (+{need_pct:.1f}%)"
    )

  if losses:
    sys.stdout.write("\n".join(losses) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))

