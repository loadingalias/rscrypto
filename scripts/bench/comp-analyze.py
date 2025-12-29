#!/usr/bin/env python3
"""
Analyze `crates/checksum/benches/comp.rs` output (or the checked-in
`crates/checksum/bench_baseline/*.txt`) and report where rscrypto loses.

This is intentionally lightweight: it parses the human-readable Criterion output
and the "Kernel selection by size" table printed at the start of the bench.

Usage:
  python3 scripts/bench/comp-analyze.py crates/checksum/bench_baseline/linux_x86-64.txt
  python3 scripts/bench/comp-analyze.py crates/checksum/bench_baseline/macos_arm64.txt
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BenchPoint:
    algo: str  # e.g. "crc64/xz"
    impl: str  # e.g. "rscrypto/checksum"
    size: str  # e.g. "xs"
    gib_s: float


KERNEL_ROW_RE = re.compile(
    r"^\s*║\s*(?P<size>\w+)\s*\(\s*(?P<bytes>\d+)\s*B\):\s*(?P<rest>.*)$"
)

BENCH_HEAD_RE = re.compile(r"^(?P<algo>crc(?:16|24|32c?|64)/(?:xz|nvme|ieee|castagnoli|ccitt|ibm|openpgp))/(?P<impl>[^/]+)/(?P<size>\w+)\s*$")

THRPT_RE = re.compile(r"^\s*thrpt:\s*\[\s*(?P<lo>[0-9.]+)\s+GiB/s\s+(?P<mid>[0-9.]+)\s+GiB/s\s+(?P<hi>[0-9.]+)\s+GiB/s\]\s*$")


def parse_kernel_selection(lines: List[str]) -> Dict[Tuple[str, str], str]:
    """
    Returns mapping: (algo, size_label) -> kernel_name
    algo keys match the `comp.rs` printed labels:
      - crc64/xz, crc64/nvme
      - crc32, crc32c
      - crc16/ccitt, crc16/ibm
      - crc24/openpgp
    """
    out: Dict[Tuple[str, str], str] = {}

    in_table = False
    for line in lines:
        if "║ Kernel selection by size:" in line:
            in_table = True
            continue
        if in_table and line.startswith("╚"):
            break
        if not in_table:
            continue

        m = KERNEL_ROW_RE.match(line)
        if not m:
            continue
        size = m.group("size")
        rest = m.group("rest")

        # Example rest contains: "crc64/xz=...  crc64/nvme=...  crc32=...  ..."
        for part in rest.split("  "):
            part = part.strip()
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            out[(k, size)] = v

    return out


def parse_points(lines: List[str]) -> List[BenchPoint]:
    points: List[BenchPoint] = []
    i = 0
    while i < len(lines):
        head = BENCH_HEAD_RE.match(lines[i].strip())
        if not head:
            i += 1
            continue
        algo = head.group("algo")
        impl = head.group("impl")
        size = head.group("size")

        # Find the next "thrpt:" line (Criterion prints "time:" then "thrpt:").
        j = i + 1
        thrpt_mid: Optional[float] = None
        while j < len(lines) and j < i + 12:
            tm = THRPT_RE.match(lines[j])
            if tm:
                thrpt_mid = float(tm.group("mid"))
                break
            j += 1

        if thrpt_mid is not None:
            points.append(BenchPoint(algo=algo, impl=impl, size=size, gib_s=thrpt_mid))
        i = j + 1
    return points


def best_by_algo_size(points: List[BenchPoint]) -> Dict[Tuple[str, str], BenchPoint]:
    best: Dict[Tuple[str, str], BenchPoint] = {}
    for p in points:
        key = (p.algo, p.size)
        cur = best.get(key)
        if cur is None or p.gib_s > cur.gib_s:
            best[key] = p
    return best


def fmt_ratio(a: float, b: float) -> str:
    if b <= 0:
        return "n/a"
    return f"{a / b:.2f}x"


def main(argv: List[str]) -> int:
    if len(argv) != 2 or argv[1] in {"-h", "--help"}:
        print(__doc__.strip())
        return 2

    path = Path(argv[1])
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    kernel_map = parse_kernel_selection(lines)
    points = parse_points(lines)
    best = best_by_algo_size(points)

    # Index rscrypto/checksum points for comparisons.
    rs: Dict[Tuple[str, str], BenchPoint] = {
        (p.algo, p.size): p for p in points if p.impl == "rscrypto/checksum"
    }

    losing: List[Tuple[str, str, BenchPoint, BenchPoint]] = []
    for key, winner in best.items():
        ours = rs.get(key)
        if ours is None:
            continue
        if winner.impl == ours.impl:
            continue
        losing.append((key[0], key[1], ours, winner))

    losing.sort(key=lambda t: (t[0], t[1]))

    if not losing:
        print(f"{path}: rscrypto/checksum is best for all parsed algo/size points.")
        return 0

    print(f"{path}: rscrypto/checksum losing cases (higher GiB/s is better):")
    for algo, size, ours, winner in losing:
        kernel = kernel_map.get((algo, size))
        kinfo = f" kernel={kernel}" if kernel else ""
        print(
            f"- {algo}/{size}{kinfo}: ours={ours.gib_s:.2f} GiB/s vs best={winner.impl} {winner.gib_s:.2f} GiB/s ({fmt_ratio(winner.gib_s, ours.gib_s)} faster)"
        )

    print("\nHeuristic root-cause hints:")
    print("- If our kernel is `portable/*` while best is SIMD: thresholds/caps gating.")
    print("- If our kernel is SIMD but still loses: kernel µarch mismatch (often AVX-512 downclock) or per-call overhead.")
    print("- Confirm by forcing tiers via env vars (e.g. `RSCRYPTO_CRC64_FORCE=portable|pclmul|vpclmul`).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

