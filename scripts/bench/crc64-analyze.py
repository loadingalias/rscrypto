#!/usr/bin/env python3
"""
CRC64 Tuning Analysis Script

Analyzes benchmark results to determine optimal:
- Kernel selection per buffer size
- Stream count (parallel_streams) for each kernel
- Threshold crossovers between kernels

Usage:
    python3 scripts/bench/crc64-analyze.py
    python3 scripts/bench/crc64-analyze.py --summary tune-results/summary.tsv
    python3 scripts/bench/crc64-analyze.py --output analysis.json

Output:
    - Console: Human-readable recommendations
    - JSON: Machine-readable optimal settings for tune.rs updates
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class BenchResult:
    """Single benchmark result."""

    group_id: str  # e.g., "crc64/xz(force=pmull-eor3,streams=3)"
    function_id: str  # e.g., "oneshot/aarch64/pmull-eor3-2way"
    kernel: str  # e.g., "pmull-eor3"
    streams: int  # e.g., 1, 2, 3 (inferred from kernel name)
    size_bytes: int  # Buffer size in bytes
    mean_ns: float  # Mean time in nanoseconds
    throughput_gib_s: float  # Throughput in GiB/s

    @property
    def config_key(self) -> str:
        """Unique key for this kernel+streams configuration."""
        return f"{self.kernel}/{self.streams}-way"


@dataclass
class SizeAnalysis:
    """Analysis results for a single buffer size."""

    size_bytes: int
    size_label: str
    best_kernel: str
    best_streams: int
    best_throughput: float
    rankings: list[tuple[str, int, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "size_bytes": self.size_bytes,
            "size_label": self.size_label,
            "best_kernel": self.best_kernel,
            "best_streams": self.best_streams,
            "best_throughput_gib_s": round(self.best_throughput, 2),
            "rankings": [
                {
                    "kernel": k,
                    "streams": s,
                    "throughput_gib_s": round(t, 2),
                    "vs_best": round(t / self.best_throughput * 100, 1)
                    if self.best_throughput > 0
                    else 0,
                }
                for k, s, t in self.rankings[:5]
            ],
        }


@dataclass
class KernelAnalysis:
    """Analysis results for a single kernel type."""

    kernel: str
    best_streams: int
    avg_throughput_by_streams: dict[int, float] = field(default_factory=dict)
    best_streams_by_size: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "best_streams": self.best_streams,
            "avg_throughput_by_streams": {
                str(k): round(v, 2) for k, v in self.avg_throughput_by_streams.items()
            },
            "best_streams_by_size": {
                _size_label(k): v for k, v in self.best_streams_by_size.items()
            },
        }


@dataclass
class PlatformAnalysis:
    """Complete analysis for a platform."""

    platform: str
    cpu_name: str
    total_results: int
    by_size: dict[int, SizeAnalysis] = field(default_factory=dict)
    by_kernel: dict[str, KernelAnalysis] = field(default_factory=dict)
    recommended_streams: int = 1
    recommended_kernel: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "cpu_name": self.cpu_name,
            "total_results": self.total_results,
            "recommendations": {
                "parallel_streams": self.recommended_streams,
                "preferred_kernel": self.recommended_kernel,
            },
            "by_size": {
                _size_label(k): v.to_dict() for k, v in sorted(self.by_size.items())
            },
            "by_kernel": {k: v.to_dict() for k, v in sorted(self.by_kernel.items())},
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _size_label(size_bytes: int) -> str:
    """Convert bytes to human-readable label."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024}KiB"
    else:
        return f"{size_bytes // (1024 * 1024)}MiB"


def _parse_kernel_from_function_id(function_id: str) -> tuple[str, int]:
    """
    Extract kernel name and stream count from function_id.

    Examples:
        "oneshot/aarch64/pmull-eor3" -> ("pmull-eor3", 1)
        "oneshot/aarch64/pmull-eor3-2way" -> ("pmull-eor3", 2)
        "oneshot/aarch64/pmull-eor3-3way" -> ("pmull-eor3", 3)
        "oneshot/x86_64/vpclmul-4way" -> ("vpclmul", 4)
        "oneshot/portable/slice16" -> ("portable", 1)
    """
    # Get the last component
    parts = function_id.split("/")
    if len(parts) < 2:
        return ("unknown", 1)

    if "portable" in parts:
        return ("portable", 1)

    kernel_part = parts[-1]  # e.g., "pmull-eor3-2way" or "pmull-small"

    # Check for N-way suffix
    way_match = re.search(r"-(\d+)way$", kernel_part)
    if way_match:
        streams = int(way_match.group(1))
        kernel = kernel_part[: way_match.start()]
    else:
        streams = 1
        kernel = kernel_part

    return (kernel, streams)


def _parse_force_from_group_id(group_id: str) -> tuple[str, int]:
    """
    Extract force and streams settings from group_id.

    Example:
        "crc64/xz(force=pmull-eor3,eff=pmull-eor3,streams=3)" -> ("pmull-eor3", 3)
        "crc64/nvme(force=auto,eff=auto,streams=3)" -> ("auto", 3)
    """
    force = "auto"
    streams = 1

    force_match = re.search(r"force=([^,)]+)", group_id)
    if force_match:
        force = force_match.group(1)

    streams_match = re.search(r"streams=(\d+)", group_id)
    if streams_match:
        streams = int(streams_match.group(1))

    return (force, streams)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result Parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def parse_summary_tsv(tsv_path: Path) -> list[BenchResult]:
    """
    Parse the summary.tsv file generated by criterion-summary.py.

    Format:
        group_id<TAB>function_id<TAB>size<TAB>mean_ns<TAB>GiB/s
    """
    results: list[BenchResult] = []
    seen: set[tuple[str, str, int]] = set()  # Deduplicate

    with open(tsv_path, "r") as f:
        header = f.readline()  # Skip header
        if not header.startswith("group_id"):
            print(f"Warning: Unexpected TSV header: {header}", file=sys.stderr)

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 5:
                continue

            group_id, function_id, size_str, mean_ns_str, throughput_str = parts[:5]

            # Skip non-CRC64 or non-oneshot entries
            if not group_id.startswith("crc64"):
                continue
            if not function_id.startswith("oneshot"):
                continue

            try:
                size_bytes = int(size_str)
                mean_ns = float(mean_ns_str)
                throughput = float(throughput_str) if throughput_str else 0.0
            except ValueError:
                continue

            # Deduplicate (criterion can report same result twice)
            key = (function_id, group_id, size_bytes)
            if key in seen:
                continue
            seen.add(key)

            # Parse kernel and streams from function_id
            kernel, streams = _parse_kernel_from_function_id(function_id)

            results.append(
                BenchResult(
                    group_id=group_id,
                    function_id=function_id,
                    kernel=kernel,
                    streams=streams,
                    size_bytes=size_bytes,
                    mean_ns=mean_ns,
                    throughput_gib_s=throughput,
                )
            )

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# Kernels to include in analysis (exclude portable, etc.)
SIMD_KERNELS = {
    "pclmul",
    "vpclmul",
    "pmull",
    "pmull-eor3",
    "pmull-small",
    "sve2-pmull",
}


def analyze_results(
    results: list[BenchResult], platform_info: dict[str, Any] | None = None
) -> PlatformAnalysis:
    """Analyze benchmark results and find optimal settings."""

    platform = platform_info.get("platform", "unknown") if platform_info else "unknown"
    cpu_name = platform_info.get("cpu_name", "unknown") if platform_info else "unknown"

    analysis = PlatformAnalysis(
        platform=platform, cpu_name=cpu_name, total_results=len(results)
    )

    if not results:
        return analysis

    # Group results by size
    by_size: dict[int, list[BenchResult]] = defaultdict(list)
    for r in results:
        by_size[r.size_bytes].append(r)

    # Group results by kernel (base kernel without streams)
    by_kernel: dict[str, list[BenchResult]] = defaultdict(list)
    for r in results:
        # Get base kernel (pmull-eor3 from pmull-eor3-2way)
        base_kernel = r.kernel
        by_kernel[base_kernel].append(r)

    # ─────────────────────────────────────────────────────────────────────
    # Analyze each size: find best kernel+streams combo
    # ─────────────────────────────────────────────────────────────────────

    for size_bytes, size_results in sorted(by_size.items()):
        if not size_results:
            continue

        # Find best result for this size
        best = max(size_results, key=lambda r: r.throughput_gib_s)

        # Get rankings (sorted by throughput descending)
        rankings = sorted(size_results, key=lambda r: -r.throughput_gib_s)
        ranking_tuples = [(r.kernel, r.streams, r.throughput_gib_s) for r in rankings]

        analysis.by_size[size_bytes] = SizeAnalysis(
            size_bytes=size_bytes,
            size_label=_size_label(size_bytes),
            best_kernel=best.kernel,
            best_streams=best.streams,
            best_throughput=best.throughput_gib_s,
            rankings=ranking_tuples,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Analyze each kernel: find optimal stream count
    # ─────────────────────────────────────────────────────────────────────

    for kernel, kernel_results in by_kernel.items():
        if kernel not in SIMD_KERNELS:
            continue

        # Group by streams
        by_streams: dict[int, list[BenchResult]] = defaultdict(list)
        for r in kernel_results:
            by_streams[r.streams].append(r)

        # Calculate average throughput per stream count (across all sizes)
        avg_by_streams: dict[int, float] = {}
        for streams, stream_results in by_streams.items():
            if stream_results:
                # Weight by size (larger sizes matter more)
                weighted_sum = sum(
                    r.throughput_gib_s * r.size_bytes for r in stream_results
                )
                weight_total = sum(r.size_bytes for r in stream_results)
                avg_by_streams[streams] = (
                    weighted_sum / weight_total if weight_total > 0 else 0
                )

        # Find best overall stream count
        best_streams = (
            max(avg_by_streams, key=lambda s: avg_by_streams[s])
            if avg_by_streams
            else 1
        )

        # Find best streams per size
        best_by_size: dict[int, int] = {}
        for size_bytes in by_size:
            size_kernel_results = [
                r for r in kernel_results if r.size_bytes == size_bytes
            ]
            if size_kernel_results:
                best_r = max(size_kernel_results, key=lambda r: r.throughput_gib_s)
                best_by_size[size_bytes] = best_r.streams

        analysis.by_kernel[kernel] = KernelAnalysis(
            kernel=kernel,
            best_streams=best_streams,
            avg_throughput_by_streams=avg_by_streams,
            best_streams_by_size=best_by_size,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Overall recommendations
    # ─────────────────────────────────────────────────────────────────────

    # Recommended streams: mode of best_streams across large sizes (>= 4KB)
    large_size_streams: list[int] = []
    for size_bytes, size_analysis in analysis.by_size.items():
        if size_bytes >= 4096:
            large_size_streams.append(size_analysis.best_streams)

    if large_size_streams:
        from collections import Counter

        stream_counts = Counter(large_size_streams)
        analysis.recommended_streams = stream_counts.most_common(1)[0][0]

    # Recommended kernel: the one that wins most at large sizes
    kernel_wins: dict[str, int] = defaultdict(int)
    for size_bytes, size_analysis in analysis.by_size.items():
        if size_bytes >= 4096 and size_analysis.best_kernel in SIMD_KERNELS:
            kernel_wins[size_analysis.best_kernel] += 1

    if kernel_wins:
        analysis.recommended_kernel = max(kernel_wins, key=lambda k: kernel_wins[k])

    return analysis


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def print_analysis(analysis: PlatformAnalysis) -> None:
    """Print human-readable analysis to console."""

    print()
    print("=" * 78)
    print("CRC64 TUNING ANALYSIS")
    print("=" * 78)
    print(f"Platform:      {analysis.platform}")
    print(f"CPU:           {analysis.cpu_name}")
    print(f"Total results: {analysis.total_results}")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Recommendations
    # ─────────────────────────────────────────────────────────────────────

    print("-" * 78)
    print("RECOMMENDATIONS FOR tune.rs")
    print("-" * 78)
    print(f"  parallel_streams: {analysis.recommended_streams}")
    print(f"  preferred_kernel: {analysis.recommended_kernel}")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Best configuration per size
    # ─────────────────────────────────────────────────────────────────────

    print("-" * 78)
    print("BEST CONFIGURATION BY BUFFER SIZE")
    print("-" * 78)
    print(f"{'Size':<10} {'Kernel':<18} {'Streams':<8} {'Throughput':<12}")
    print("-" * 50)

    for size_bytes in sorted(analysis.by_size.keys()):
        sa = analysis.by_size[size_bytes]
        streams_str = f"{sa.best_streams}-way" if sa.best_streams > 1 else "1-way"
        print(
            f"{sa.size_label:<10} {sa.best_kernel:<18} {streams_str:<8} {sa.best_throughput:>8.2f} GiB/s"
        )

    print()

    # ─────────────────────────────────────────────────────────────────────
    # Optimal streams per kernel
    # ─────────────────────────────────────────────────────────────────────

    print("-" * 78)
    print("OPTIMAL STREAM COUNT BY KERNEL")
    print("-" * 78)

    for kernel in sorted(analysis.by_kernel.keys()):
        ka = analysis.by_kernel[kernel]
        print(f"\n{kernel}:")
        print(f"  Best overall: {ka.best_streams}-way")

        if ka.avg_throughput_by_streams:
            print(f"  Average throughput by stream count:")
            for streams in sorted(ka.avg_throughput_by_streams.keys()):
                throughput = ka.avg_throughput_by_streams[streams]
                marker = " <-- best" if streams == ka.best_streams else ""
                print(f"    {streams}-way: {throughput:>8.2f} GiB/s{marker}")

    print()
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze CRC64 benchmark results for optimal tuning."
    )
    parser.add_argument(
        "--summary",
        default="tune-results/summary.tsv",
        help="Path to summary.tsv from criterion-summary.py",
    )
    parser.add_argument(
        "--platform-file",
        default="tune-results/platform.json",
        help="Platform info JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="",
        help="Output JSON file for analysis results",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON to stdout instead of human-readable",
    )

    args = parser.parse_args()

    # Load platform info
    platform_info: dict[str, Any] = {}
    platform_path = Path(args.platform_file)
    if platform_path.exists():
        try:
            with open(platform_path, "r") as f:
                platform_info = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load platform info: {e}", file=sys.stderr)

    # Parse results
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}", file=sys.stderr)
        print("Run the tuning script first:", file=sys.stderr)
        print(
            "  RUSTFLAGS='-C target-cpu=native' scripts/bench/crc64-tune.sh --quick",
            file=sys.stderr,
        )
        return 1

    results = parse_summary_tsv(summary_path)
    if not results:
        print(f"Error: No results parsed from {summary_path}", file=sys.stderr)
        return 1

    print(f"Parsed {len(results)} benchmark results", file=sys.stderr)

    # Run analysis
    analysis = analyze_results(results, platform_info)

    # Output
    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        print_analysis(analysis)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(analysis.to_dict(), f, indent=2)
        print(f"\nAnalysis saved to: {output_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
