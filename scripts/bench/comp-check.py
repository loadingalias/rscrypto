#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

KNOWN_IMPLS = frozenset({"rscrypto", "official", "official-rayon"})


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


@dataclass(frozen=True)
class Loss:
  case_key: str
  ours_impl: str
  ours_metric: Metric
  other_impl: str
  other_metric: Metric
  need_pct: float


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
  for i, p in enumerate(parts):
    if p in KNOWN_IMPLS:
      case_parts = parts[:i] + parts[i + 1 :]
      return ("/".join(case_parts), p)
  return None


def _parse_size_suffix(segment: str) -> Optional[int]:
  """Try to parse a trailing size from a bench-id segment (e.g. '1048576' or '65536')."""
  try:
    return int(segment)
  except ValueError:
    return None


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


def _best_for_impl(case_entries: list[Entry], impl: str) -> Optional[Entry]:
  candidates = [e for e in case_entries if e.impl == impl]
  if not candidates:
    return None
  return max(candidates, key=lambda e: e.metric.normalized)


def _parse_case_size(case_key: str) -> Optional[int]:
  last_seg = case_key.rsplit("/", 1)[-1]
  return _parse_size_suffix(last_seg)


def _loss_against_impl(case_entries: list[Entry], ours_impl: str, target_impl: str) -> Optional[Loss]:
  ours_best = _best_for_impl(case_entries, ours_impl)
  target_best = _best_for_impl(case_entries, target_impl)
  if ours_best is None or target_best is None:
    return None
  if target_best.metric.normalized <= ours_best.metric.normalized:
    return None

  ours_v = ours_best.metric.normalized
  target_v = target_best.metric.normalized
  need_pct = (target_v / ours_v - 1.0) * 100.0 if ours_v > 0.0 else 0.0
  return Loss(
    case_key=ours_best.case_key,
    ours_impl=ours_impl,
    ours_metric=ours_best.metric,
    other_impl=target_impl,
    other_metric=target_best.metric,
    need_pct=need_pct,
  )


def _print_losses(label: str, losses: list[Loss]) -> None:
  if not losses:
    print(f"{label}: PASS (no losses)")
    return
  print(f"{label}: {len(losses)} loss(es)")
  for loss in losses:
    print(
      f"  {loss.case_key} {loss.ours_impl} {loss.ours_metric.mid:g} {loss.ours_metric.unit} "
      f"< {loss.other_impl} {loss.other_metric.mid:g} {loss.other_metric.unit} (+{loss.need_pct:.1f}%)"
    )


def _gate_exit_code(mode: str, losses: list[Loss]) -> int:
  if mode == "fail" and losses:
    return 1
  return 0


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(
    description="Compare rscrypto against official baselines with separate single-thread and parallel gates."
  )
  parser.add_argument("path", help="Path to saved `cargo bench` output (Criterion stdout)")
  parser.add_argument("--ours", default="rscrypto", help="Implementation name to treat as ours (default: rscrypto)")
  parser.add_argument(
    "--prefer",
    default="thrpt",
    choices=["thrpt", "time"],
    help="Prefer comparing throughput or time if both are present (default: thrpt)",
  )
  parser.add_argument(
    "--parallel-threshold",
    type=int,
    default=524288,
    help="Minimum input size (bytes) to consider parallel comparisons (default: 524288 = 512 KiB)",
  )
  parser.add_argument(
    "--gate-single-thread",
    default="fail",
    choices=["off", "warn", "fail"],
    help="Single-thread gate mode against 'official' (default: fail)",
  )
  parser.add_argument(
    "--gate-parallel",
    default="warn",
    choices=["off", "warn", "fail"],
    help="Parallel gate mode against 'official-rayon' for size >= --parallel-threshold (default: warn)",
  )
  parser.add_argument(
    "--quiet-pass",
    action="store_true",
    help="Suppress PASS lines and print only losses/warnings",
  )
  parser.add_argument(
    "--require-groups",
    default=None,
    help="Comma-separated group prefixes; exit 1 if any prefix has no rscrypto entries",
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

  # --require-groups validation
  if args.require_groups:
    required = [p.strip() for p in args.require_groups.split(",") if p.strip()]
    all_rscrypto_keys = {e.case_key for e in entries if e.impl == args.ours}
    missing: list[str] = []
    for prefix in required:
      if not any(k.startswith(prefix) for k in all_rscrypto_keys):
        missing.append(prefix)
    if missing:
      for m in missing:
        print(f"error: --require-groups: no {args.ours} entries for group prefix '{m}'", file=sys.stderr)
      return 1

  single_thread_losses: list[Loss] = []
  parallel_losses: list[Loss] = []
  parallel_missing_baseline: list[str] = []
  threshold = args.parallel_threshold

  for case_key in sorted(by_case.keys()):
    case_entries = by_case[case_key]
    impls_present = {e.impl for e in case_entries}
    if args.ours not in impls_present:
      continue

    single_loss = _loss_against_impl(case_entries, args.ours, "official")
    if single_loss is not None:
      single_thread_losses.append(single_loss)

    size = _parse_case_size(case_key)
    if size is None or size < threshold:
      continue

    if "official-rayon" not in impls_present:
      if "official" in impls_present:
        parallel_missing_baseline.append(case_key)
      continue

    parallel_loss = _loss_against_impl(case_entries, args.ours, "official-rayon")
    if parallel_loss is not None:
      parallel_losses.append(parallel_loss)

  if not args.quiet_pass or single_thread_losses:
    _print_losses("single-thread gate (official)", single_thread_losses)
  if not args.quiet_pass or parallel_losses:
    _print_losses(
      f"parallel gate (official-rayon, size >= {threshold}B)",
      parallel_losses,
    )

  if parallel_missing_baseline:
    for case_key in sorted(parallel_missing_baseline):
      print(
        f"WARNING: {case_key}: missing official-rayon at eligible parallel size",
        file=sys.stderr,
      )

  if args.gate_single_thread == "warn" and single_thread_losses:
    print(f"WARNING: single-thread gate has {len(single_thread_losses)} loss(es)", file=sys.stderr)
  if args.gate_parallel == "warn" and parallel_losses:
    print(f"WARNING: parallel gate has {len(parallel_losses)} loss(es)", file=sys.stderr)

  exit_code = 0
  exit_code = max(exit_code, _gate_exit_code(args.gate_single_thread, single_thread_losses))
  exit_code = max(exit_code, _gate_exit_code(args.gate_parallel, parallel_losses))
  return exit_code


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
