#!/usr/bin/env python3
"""Report rscrypto AEAD gaps against the fastest external benchmark row."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TIME_RE = re.compile(
  r"^(?:(?P<name>[a-z0-9-]+/[a-z0-9-]+/[^\s/]+/[^\s/]+)\s+)?"
  r"time:\s+\[(?P<lo>[0-9.]+)\s+(?P<lo_unit>\S+)\s+"
  r"(?P<mean>[0-9.]+)\s+(?P<mean_unit>\S+)\s+"
  r"(?P<hi>[0-9.]+)\s+(?P<hi_unit>\S+)\]"
)
NAME_RE = re.compile(r"^(?P<name>[a-z0-9-]+/[a-z0-9-]+/[^\s/]+/[^\s/]+)\s*$")
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


@dataclass(frozen=True)
class Row:
  platform: str
  algorithm: str
  operation: str
  size_label: str
  size_sort: int
  ours_ns: float
  fastest_external_ns: float
  fastest_external: str

  @property
  def ratio(self) -> float:
    return self.fastest_external_ns / self.ours_ns

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


def parse_size_label(size_label: str) -> int | None:
  numeric = size_label[:-1] if size_label.endswith("B") else size_label
  try:
    return int(numeric)
  except ValueError:
    return None


def parse_bench_name(name: str, algorithm: str, operation: str) -> tuple[str, str, int] | None:
  parts = name.split("/")
  if len(parts) != 4:
    return None
  row_algorithm, row_operation, implementation, size_label = parts
  if row_algorithm != algorithm or row_operation != operation:
    return None

  size_sort = parse_size_label(size_label)
  if size_sort is None:
    return None
  return implementation, size_label, size_sort


def platform_from_text(path: Path, text: str) -> str:
  for line in text.splitlines()[:20]:
    match = PLATFORM_RE.match(line)
    if match:
      return match.group("platform")
  return path.parent.name


def parse_results_file(path: Path, algorithm: str, operation: str) -> list[Row]:
  text = path.read_text(encoding="utf-8")
  platform = platform_from_text(path, text)
  current_name: str | None = None
  times: dict[tuple[str, str], tuple[int, float]] = {}

  for line in text.splitlines():
    if line.startswith("Benchmarking "):
      continue

    time_match = TIME_RE.match(line.strip())
    if time_match:
      name = time_match.group("name") or current_name
      current_name = None
      if name is None:
        continue
      parsed = parse_bench_name(name, algorithm, operation)
      if parsed is None:
        continue
      implementation, size_label, size_sort = parsed
      mean_ns = parse_time_to_ns(time_match.group("mean"), time_match.group("mean_unit"))
      times[(size_label, implementation)] = (size_sort, mean_ns)
      continue

    name_match = NAME_RE.match(line.strip())
    if name_match:
      current_name = name_match.group("name")

  rows: list[Row] = []
  size_labels = sorted({size_label for size_label, _ in times.keys()}, key=lambda label: times[(label, "rscrypto")][0] if (label, "rscrypto") in times else 0)
  for size_label in size_labels:
    ours = times.get((size_label, "rscrypto"))
    if ours is None:
      continue

    externals = [
      (implementation, value)
      for (row_size, implementation), value in times.items()
      if row_size == size_label and not implementation.startswith("rscrypto")
    ]
    if not externals:
      continue

    fastest_external, (_, fastest_ns) = min(externals, key=lambda item: item[1][1])
    rows.append(
      Row(
        platform=platform,
        algorithm=algorithm,
        operation=operation,
        size_label=size_label,
        size_sort=ours[0],
        ours_ns=ours[1],
        fastest_external_ns=fastest_ns,
        fastest_external=fastest_external,
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


def group_by_size(rows: list[Row]) -> list[tuple[str, list[Row]]]:
  grouped: dict[str, list[Row]] = defaultdict(list)
  size_order: dict[str, int] = {}
  for row in rows:
    grouped[row.size_label].append(row)
    size_order[row.size_label] = row.size_sort
  return sorted(grouped.items(), key=lambda item: size_order[item[0]])


def render_summary_table(title: str, rows: list[tuple[str, list[Row]]]) -> list[str]:
  out = [f"### {title}", "", "| Scope | Rows | W/T/L | Geomean | Median |", "| --- | ---: | ---: | ---: | ---: |"]
  for label, group_rows in rows:
    wins, ties, losses, geomean, median = stats(group_rows)
    out.append(f"| {label} | {len(group_rows)} | {wins}/{ties}/{losses} | {fmt_ratio(geomean)} | {fmt_ratio(median)} |")
  out.append("")
  return out


def render_pressure(rows: list[Row]) -> list[str]:
  pressure = Counter(row.fastest_external for row in rows if row.class_name == "loss")
  out = ["### Loss Pressure", "", "| External | Loss Rows |", "| --- | ---: |"]
  for implementation, count in pressure.most_common():
    out.append(f"| `{implementation}` | {count} |")
  if not pressure:
    out.append("| none | 0 |")
  out.append("")
  return out


def render_worst(rows: list[Row], top: int) -> list[str]:
  out = [
    "### Worst Rows",
    "",
    "| Platform | Size | External | Ratio | Needed Reduction | rscrypto | fastest external |",
    "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
  ]
  worst = sorted(rows, key=lambda row: (row.ratio, row.platform, row.size_sort))[:top]
  for row in worst:
    platform = PLATFORM_LABELS.get(row.platform, row.platform)
    out.append(
      f"| {platform} | {row.size_label} | `{row.fastest_external}` | {fmt_ratio(row.ratio)} | "
      f"{row.needed_reduction_pct:.1f}% | {fmt_ns(row.ours_ns)} | {fmt_ns(row.fastest_external_ns)} |"
    )
  out.append("")
  return out


def render_markdown(rows: list[Row], root: Path, algorithm: str, operation: str, top: int) -> str:
  all_stats = [
    (f"All parsed {algorithm}/{operation} rows", rows),
    ("x86_64 rows", filter_rows(rows, "x86")),
    ("AArch64 rows", filter_rows(rows, "aarch64")),
  ]

  out = [
    "# AEAD Gap Report",
    "",
    f"Source: `{root}`",
    f"Target: `{algorithm}/{operation}`",
    "",
    "Ratio is `fastest external time / rscrypto time`; higher means rscrypto is faster.",
    "Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`.",
    "",
  ]
  out.extend(render_summary_table("Overall", all_stats))
  out.extend(
    render_summary_table(
      "By Platform",
      [(PLATFORM_LABELS.get(platform, platform), group_rows) for platform, group_rows in group_by_platform(rows)],
    )
  )
  out.extend(render_summary_table("By Size", group_by_size(rows)))
  out.extend(render_pressure(rows))
  out.extend(render_worst(rows, top))
  return "\n".join(out)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--algorithm", default="chacha20-poly1305")
  parser.add_argument("--operation", default="encrypt")
  parser.add_argument("--arch", choices=["all", "x86", "aarch64"], default="all")
  parser.add_argument("--top", type=int, default=16)
  args = parser.parse_args()

  rows: list[Row] = []
  for path in selected_files(args.root):
    rows.extend(parse_results_file(path, args.algorithm, args.operation))

  rows = filter_rows(rows, args.arch)
  if not rows:
    raise SystemExit(f"error: no AEAD benchmark rows parsed from {args.root}")

  print(render_markdown(rows, args.root, args.algorithm, args.operation, args.top))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
