#!/usr/bin/env python3
"""Report BLAKE3 rscrypto-vs-official benchmark gaps from raw CI artifacts."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TIME_RE = re.compile(
  r"^(?:(?P<name>blake3/[^\s]+(?:/[^\s]+){1,2})\s+)?"
  r"time:\s+\[(?P<lo>[0-9.]+)\s+(?P<lo_unit>\S+)\s+"
  r"(?P<mean>[0-9.]+)\s+(?P<mean_unit>\S+)\s+"
  r"(?P<hi>[0-9.]+)\s+(?P<hi_unit>\S+)\]"
)
NAME_RE = re.compile(r"^(?P<name>blake3/[^\s]+(?:/[^\s]+){1,2})\s*$")
PLATFORM_RE = re.compile(r"^platform=(?P<platform>[A-Za-z0-9_-]+)\s*$")

X86_PLATFORMS = {"amd-zen4", "amd-zen5", "intel-icl", "intel-spr"}
AARCH64_PLATFORMS = {"graviton3", "graviton4"}

PLATFORM_LABELS = {
  "amd-zen4": "AMD Zen4",
  "amd-zen5": "AMD Zen5",
  "graviton3": "AWS Graviton3",
  "graviton4": "AWS Graviton4",
  "ibm-power10": "IBM Power10",
  "ibm-s390x": "IBM z16/s390x",
  "intel-icl": "Intel Ice Lake",
  "intel-spr": "Intel Sapphire Rapids",
  "rise-riscv": "RISE RISC-V",
}

OP_ORDER = {
  "oneshot": 0,
  "keyed": 1,
  "derive-key": 2,
  "streaming": 3,
  "xof": 4,
}


@dataclass(frozen=True)
class Row:
  platform: str
  op: str
  size_label: str
  size_sort: int
  ours_ns: float
  official_ns: float

  @property
  def ratio(self) -> float:
    return self.official_ns / self.ours_ns

  @property
  def class_name(self) -> str:
    if self.ratio > 1.05:
      return "win"
    if self.ratio < 0.95:
      return "loss"
    return "tie"

  @property
  def needed_reduction_pct(self) -> float:
    return max(0.0, (1.0 / self.ratio - 1.0) * 100.0)


def parse_time_to_ns(value: str, unit: str) -> float:
  scalar = float(value)
  if unit == "ns":
    return scalar
  if unit in {"us", "\N{MICRO SIGN}s"}:
    return scalar * 1_000.0
  if unit == "ms":
    return scalar * 1_000_000.0
  if unit == "s":
    return scalar * 1_000_000_000.0
  raise ValueError(f"unsupported time unit: {unit}")


def parse_bench_name(name: str) -> tuple[str, str, str, int] | None:
  parts = name.split("/")
  if len(parts) == 3 and parts[0] == "blake3":
    op = "oneshot"
    impl = parts[1]
    size_label = parts[2]
  elif len(parts) == 4 and parts[0] == "blake3":
    op = parts[1]
    impl = parts[2]
    size_label = parts[3]
  else:
    return None

  if impl not in {"rscrypto", "blake3"}:
    return None

  numeric = size_label[:-1] if size_label.endswith("B") else size_label
  try:
    size_sort = int(numeric)
  except ValueError:
    return None

  return op, impl, size_label, size_sort


def platform_from_text(path: Path, text: str) -> str:
  for line in text.splitlines()[:20]:
    match = PLATFORM_RE.match(line)
    if match:
      return match.group("platform")
  return path.parent.name


def parse_results_file(path: Path) -> list[Row]:
  text = path.read_text(encoding="utf-8")
  platform = platform_from_text(path, text)
  current_name: str | None = None
  times: dict[tuple[str, str, str], tuple[int, float]] = {}

  for line in text.splitlines():
    if line.startswith("Benchmarking "):
      continue

    time_match = TIME_RE.match(line.strip())
    if time_match:
      name = time_match.group("name") or current_name
      current_name = None
      if name is None:
        continue
      parsed = parse_bench_name(name)
      if parsed is None:
        continue
      op, impl, size_label, size_sort = parsed
      mean_ns = parse_time_to_ns(time_match.group("mean"), time_match.group("mean_unit"))
      times[(op, size_label, impl)] = (size_sort, mean_ns)
      continue

    name_match = NAME_RE.match(line.strip())
    if name_match:
      current_name = name_match.group("name")

  rows: list[Row] = []
  seen_cases = {(op, size_label) for op, size_label, _ in times.keys()}
  for op, size_label in sorted(seen_cases, key=lambda item: (OP_ORDER.get(item[0], 99), item[1])):
    ours = times.get((op, size_label, "rscrypto"))
    official = times.get((op, size_label, "blake3"))
    if ours is None or official is None:
      continue
    rows.append(
      Row(
        platform=platform,
        op=op,
        size_label=size_label,
        size_sort=ours[0],
        ours_ns=ours[1],
        official_ns=official[1],
      )
    )
  return rows


def selected_files(root: Path) -> list[Path]:
  if root.is_file():
    return [root]
  return sorted(root.glob("*/results.txt"))


def filter_rows(rows: list[Row], arch: str) -> list[Row]:
  if arch == "all":
    return rows
  if arch == "x86":
    return [row for row in rows if row.platform in X86_PLATFORMS]
  if arch == "aarch64":
    return [row for row in rows if row.platform in AARCH64_PLATFORMS]
  raise ValueError(f"unsupported arch filter: {arch}")


def stats(rows: list[Row]) -> tuple[int, int, int, float, float]:
  if not rows:
    return 0, 0, 0, float("nan"), float("nan")
  wins = sum(1 for row in rows if row.class_name == "win")
  ties = sum(1 for row in rows if row.class_name == "tie")
  losses = sum(1 for row in rows if row.class_name == "loss")
  geomean = math.exp(sum(math.log(row.ratio) for row in rows) / len(rows))
  median = statistics.median(row.ratio for row in rows)
  return wins, ties, losses, geomean, median


def fmt_ratio(value: float) -> str:
  return f"{value:.3f}x"


def fmt_ns(value: float) -> str:
  if value < 1_000.0:
    return f"{value:.2f} ns"
  if value < 1_000_000.0:
    return f"{value / 1_000.0:.2f} us"
  if value < 1_000_000_000.0:
    return f"{value / 1_000_000.0:.2f} ms"
  return f"{value / 1_000_000_000.0:.2f} s"


def group_by_platform(rows: list[Row]) -> list[tuple[str, list[Row]]]:
  grouped: dict[str, list[Row]] = defaultdict(list)
  for row in rows:
    grouped[row.platform].append(row)
  return sorted(grouped.items(), key=lambda item: item[0])


def group_by_op(rows: list[Row]) -> list[tuple[str, list[Row]]]:
  grouped: dict[str, list[Row]] = defaultdict(list)
  for row in rows:
    grouped[row.op].append(row)
  return sorted(grouped.items(), key=lambda item: OP_ORDER.get(item[0], 99))


def render_summary_table(title: str, rows: list[tuple[str, list[Row]]]) -> list[str]:
  out = [f"### {title}", "", "| Scope | Rows | W/T/L | Geomean | Median |", "| --- | ---: | ---: | ---: | ---: |"]
  for label, group_rows in rows:
    wins, ties, losses, geomean, median = stats(group_rows)
    out.append(f"| {label} | {len(group_rows)} | {wins}/{ties}/{losses} | {fmt_ratio(geomean)} | {fmt_ratio(median)} |")
  out.append("")
  return out


def render_worst(rows: list[Row], top: int) -> list[str]:
  out = [
    f"### Worst x86 Rows",
    "",
    "| Platform | Op | Size | Ratio | Needed Reduction | rscrypto | official blake3 |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
  ]
  worst = sorted(rows, key=lambda row: (row.ratio, row.platform, row.op, row.size_sort))[:top]
  for row in worst:
    platform = PLATFORM_LABELS.get(row.platform, row.platform)
    out.append(
      f"| {platform} | `{row.op}` | {row.size_label} | {fmt_ratio(row.ratio)} | "
      f"{row.needed_reduction_pct:.1f}% | {fmt_ns(row.ours_ns)} | {fmt_ns(row.official_ns)} |"
    )
  out.append("")
  return out


def render_markdown(rows: list[Row], root: Path, top: int) -> str:
  all_stats = [("All parsed BLAKE3 rows", rows), ("x86_64 rows", filter_rows(rows, "x86")), ("AArch64 rows", filter_rows(rows, "aarch64"))]
  x86_rows = filter_rows(rows, "x86")

  out = [
    "# BLAKE3 Gap Report",
    "",
    f"Source: `{root}`",
    "",
    "Ratio is `official blake3 time / rscrypto time`; higher means rscrypto is faster.",
    "Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`.",
    "",
  ]
  out.extend(render_summary_table("Overall", all_stats))
  out.extend(
    render_summary_table(
      "x86_64 By Platform",
      [(PLATFORM_LABELS.get(platform, platform), group_rows) for platform, group_rows in group_by_platform(x86_rows)],
    )
  )
  out.extend(render_summary_table("x86_64 By Operation", group_by_op(x86_rows)))
  out.extend(render_worst(x86_rows, top))
  return "\n".join(out)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=Path("benchmark_results/2026-06-22/linux"))
  parser.add_argument("--arch", choices=["all", "x86", "aarch64"], default="all")
  parser.add_argument("--top", type=int, default=16)
  args = parser.parse_args()

  rows: list[Row] = []
  for path in selected_files(args.root):
    rows.extend(parse_results_file(path))

  rows = filter_rows(rows, args.arch)
  if not rows:
    raise SystemExit(f"error: no BLAKE3 benchmark rows parsed from {args.root}")

  print(render_markdown(rows, args.root, args.top))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
