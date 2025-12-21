#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Record:
    group_id: str
    function_id: str
    size: int
    bytes: int | None
    mean_ns: float

    @property
    def gib_per_s(self) -> float | None:
        if self.bytes is None:
            return None
        if self.mean_ns == 0:
            return None
        bytes_per_s = self.bytes / (self.mean_ns * 1e-9)
        return bytes_per_s / (1024**3)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_records(root: Path) -> Iterable[Record]:
    for benchmark_path in root.rglob("benchmark.json"):
        estimates_path = benchmark_path.with_name("estimates.json")
        if not estimates_path.exists():
            continue

        bench = _read_json(benchmark_path)
        estimates = _read_json(estimates_path)

        group_id = bench.get("group_id", "")
        function_id = bench.get("function_id", "")
        value_str = bench.get("value_str", "")
        try:
            size = int(value_str)
        except ValueError:
            continue

        throughput = bench.get("throughput") or {}
        bytes_value = throughput.get("Bytes")
        if bytes_value is not None:
            try:
                bytes_value = int(bytes_value)
            except (TypeError, ValueError):
                bytes_value = None

        mean_ns = float(estimates["mean"]["point_estimate"])

        yield Record(
            group_id=group_id,
            function_id=function_id,
            size=size,
            bytes=bytes_value,
            mean_ns=mean_ns,
        )


def _parse_sizes(value: str) -> set[int]:
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Criterion results (mean + GiB/s).")
    parser.add_argument("--criterion-dir", default="target/criterion", help="Criterion output directory")
    parser.add_argument("--group-prefix", default="", help="Only include group_id starting with this prefix")
    parser.add_argument(
        "--only",
        choices=["oneshot", "stream", "buffered", "all"],
        default="oneshot",
        help="Filter by function_id prefix",
    )
    parser.add_argument(
        "--sizes",
        default="",
        help="Comma-separated sizes to include (default: all)",
    )
    args = parser.parse_args()

    root = Path(args.criterion_dir)
    if not root.exists():
        raise SystemExit(f"criterion dir not found: {root}")

    only_prefix: str | None = None
    if args.only == "oneshot":
        only_prefix = "oneshot/"
    elif args.only == "stream":
        only_prefix = "stream/"
    elif args.only == "buffered":
        only_prefix = "buffered/"

    sizes: set[int] | None = None
    if args.sizes.strip():
        sizes = _parse_sizes(args.sizes)

    records = []
    for rec in _iter_records(root):
        if args.group_prefix and not rec.group_id.startswith(args.group_prefix):
            continue
        if only_prefix is not None and not rec.function_id.startswith(only_prefix):
            continue
        if sizes is not None and rec.size not in sizes:
            continue
        records.append(rec)

    records.sort(key=lambda r: (r.group_id, r.function_id, r.size))

    if not records:
        print("No matching benchmarks found.")
        return 0

    # Header
    print("group_id\tfunction_id\tsize\tmean_ns\tGiB/s")
    for rec in records:
        gib = rec.gib_per_s
        gib_str = "" if gib is None else f"{gib:.2f}"
        print(f"{rec.group_id}\t{rec.function_id}\t{rec.size}\t{rec.mean_ns:.3f}\t{gib_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

