#!/usr/bin/env python3
"""
Summarize rscrypto-blake3-boundary CSV artifacts and suggest x86 boundaries.

Input files are produced by:
  cargo run -p tune --release --bin rscrypto-blake3-boundary -- --output <file>.csv [--force-kernel <k>]

This script groups by (variant, size), finds the fastest effective kernel, and emits:
  - best kernel table
  - simple boundary suggestion for plain mode (xs/s/m/l)
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict


def normalize_kernel(name: str) -> str:
    n = name.strip().lower()
    aliases = {
        "sse41": "x86_64/sse4.1",
        "avx2": "x86_64/avx2",
        "avx512": "x86_64/avx512",
        "portable": "portable",
    }
    return aliases.get(n, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="+", help="boundary CSV files")
    args = ap.parse_args()

    rows = []
    for p in args.csv:
        path = pathlib.Path(p)
        if not path.exists():
            print(f"missing file: {path}", file=sys.stderr)
            return 2
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))

    by_variant_size: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        try:
            size = int(r["size"])
            tp = float(r["throughput_gib_s"])
        except (KeyError, ValueError):
            continue
        if size <= 0 or tp <= 0:
            continue
        r["_size"] = str(size)
        r["_tp"] = str(tp)
        r["_effective_norm"] = normalize_kernel(r.get("effective_kernel", ""))
        by_variant_size[(r.get("variant", ""), size)].append(r)

    if not by_variant_size:
        print("no usable rows found", file=sys.stderr)
        return 1

    # Best kernel per (variant,size)
    best = {}
    for key, items in by_variant_size.items():
        items_sorted = sorted(items, key=lambda x: float(x["_tp"]), reverse=True)
        best[key] = items_sorted[0]

    print("== Best Kernel By Variant/Size ==")
    print("variant,size,best_kernel,throughput_gib_s")
    for (variant, size) in sorted(best):
        b = best[(variant, size)]
        print(f"{variant},{size},{b['_effective_norm']},{float(b['_tp']):.6f}")

    # Plain-mode boundary suggestion (x86 tiers)
    plain = [(size, rec["_effective_norm"]) for (variant, size), rec in best.items() if variant == "plain"]
    if not plain:
        return 0
    plain.sort()

    # Heuristic tier assignment by kernel speed winner.
    # xs: <=64, s: next tier, m: next tier, l: largest tier.
    xs_kernel = next((k for s, k in plain if s <= 64), "portable")
    after64 = [(s, k) for s, k in plain if s > 64]
    if not after64:
        return 0

    # Collapse contiguous winner regions.
    regions: list[tuple[int, int, str]] = []
    start = after64[0][0]
    prev = after64[0][0]
    cur = after64[0][1]
    for s, k in after64[1:]:
        if k == cur:
            prev = s
            continue
        regions.append((start, prev, cur))
        start = prev = s
        cur = k
    regions.append((start, prev, cur))

    s_kernel = regions[0][2]
    m_kernel = regions[1][2] if len(regions) > 1 else s_kernel
    l_kernel = regions[-1][2]
    s_max = regions[0][1]
    m_max = regions[1][1] if len(regions) > 1 else s_max

    print("\n== Suggested Plain Dispatch (Heuristic) ==")
    print(f"boundaries: [64, {s_max}, {m_max}]")
    print(f"xs: {xs_kernel}")
    print(f"s:  {s_kernel}")
    print(f"m:  {m_kernel}")
    print(f"l:  {l_kernel}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
