#!/usr/bin/env python3
"""
Generate kernel tables from Criterion benchmark results.

This script parses *_kernels.txt files from bench_baseline/ and generates
crates/checksum/src/generated/kernel_tables.rs with optimal kernel selections.

Usage:
    python scripts/gen/kernel_tables.py

The generated file contains:
- KernelTable structs for each benchmarked platform
- Size class boundaries (xs/s/m/l/xl mapped to byte thresholds)
- Direct function pointer assignments for each (variant, size_class)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Size class definitions (must match benchmark naming)
SIZE_CLASSES = ["xs", "s", "m", "l", "xl"]

# Size class byte thresholds (used for dispatch boundaries)
# These are the UPPER bounds for each class
SIZE_CLASS_BYTES = {
    "xs": 64,      # 0-64 bytes
    "s": 256,      # 65-256 bytes
    "m": 4096,     # 257-4KB
    "l": 65536,    # 4KB-64KB
    "xl": 1048576, # 64KB+ (1MB benchmark point)
}

# CRC variants we support
CRC_VARIANTS = [
    ("crc16", "ccitt"),
    ("crc16", "ibm"),
    ("crc24", "openpgp"),
    ("crc32", "ieee"),
    ("crc32c", "castagnoli"),
    ("crc64", "xz"),
    ("crc64", "nvme"),
]

# Map variant to Rust function type and width
VARIANT_INFO = {
    ("crc16", "ccitt"): ("Crc16Fn", "u16", "crc16_ccitt"),
    ("crc16", "ibm"): ("Crc16Fn", "u16", "crc16_ibm"),
    ("crc24", "openpgp"): ("Crc24Fn", "u32", "crc24_openpgp"),
    ("crc32", "ieee"): ("Crc32Fn", "u32", "crc32_ieee"),
    ("crc32c", "castagnoli"): ("Crc32Fn", "u32", "crc32c"),
    ("crc64", "xz"): ("Crc64Fn", "u64", "crc64_xz"),
    ("crc64", "nvme"): ("Crc64Fn", "u64", "crc64_nvme"),
}

# Map TuneKind names to Rust enum variants
TUNE_KIND_MAP = {
    "AppleM1M3": "AppleM1M3",
    "AppleM4": "AppleM4",
    "AppleM5": "AppleM5",
    "Graviton2": "Graviton2",
    "Graviton3": "Graviton3",
    "Graviton4": "Graviton4",
    "Graviton5": "Graviton5",
    "NeoverseN2": "NeoverseN2",
    "NeoverseN3": "NeoverseN3",
    "NeoverseV3": "NeoverseV3",
    "NvidiaGrace": "NvidiaGrace",
    "AmpereAltra": "AmpereAltra",
    "Zen4": "Zen4",
    "Zen5": "Zen5",
    "Zen5c": "Zen5c",
    "IntelSpr": "IntelSpr",
    "IntelGnr": "IntelGnr",
    "IntelIcl": "IntelIcl",
    "Default": "Default",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    """Single benchmark result."""
    kernel_path: str      # e.g., "aarch64/pmull-eor3-3way"
    size_class: str       # xs, s, m, l, xl
    time_ns: float        # median time in nanoseconds
    throughput_gib_s: float  # throughput in GiB/s

@dataclass
class VariantResults:
    """All benchmark results for one CRC variant."""
    crc_type: str         # e.g., "crc64"
    variant: str          # e.g., "xz"
    results: dict[str, list[BenchResult]] = field(default_factory=dict)
    # results[size_class] = [BenchResult, ...]

@dataclass
class PlatformResults:
    """All benchmark results for one platform."""
    tune_kind: str        # e.g., "AppleM1M3"
    arch: str             # e.g., "aarch64" or "x86_64"
    variants: dict[tuple[str, str], VariantResults] = field(default_factory=dict)

@dataclass
class KernelSelection:
    """Optimal kernel for one (variant, size_class) combination."""
    kernel_path: str      # e.g., "aarch64/pmull-eor3-3way"
    throughput_gib_s: float

@dataclass
class PlatformTable:
    """Complete kernel table for one platform."""
    tune_kind: str
    arch: str
    # selections[(crc_type, variant)][size_class] = KernelSelection
    selections: dict[tuple[str, str], dict[str, KernelSelection]] = field(default_factory=dict)

# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_tune_kind(content: str) -> Optional[str]:
    """Extract TuneKind from benchmark file header."""
    match = re.search(r"Tune Kind:\s*(\w+)", content)
    if match:
        return match.group(1)
    return None

def parse_arch_from_caps(content: str) -> Optional[str]:
    """Extract architecture from Caps line."""
    if "aarch64" in content:
        return "aarch64"
    elif "x86_64" in content or "x86-64" in content:
        return "x86_64"
    return None

def parse_benchmark_line(line: str) -> Optional[tuple[str, float, float]]:
    """
    Parse a Criterion benchmark result line.

    Returns (bench_name, time_ns, throughput_gib_s) or None.
    """
    # Match time line: "time:   [31.494 ns 32.155 ns 33.047 ns]"
    time_match = re.match(r"\s*time:\s*\[[\d.]+ \w+ ([\d.]+) (\w+)", line)
    if time_match:
        value = float(time_match.group(1))
        unit = time_match.group(2)
        # Convert to nanoseconds
        if unit == "ns":
            return ("time", value, 0.0)
        elif unit == "µs" or unit == "us":
            return ("time", value * 1000, 0.0)
        elif unit == "ms":
            return ("time", value * 1_000_000, 0.0)

    # Match throughput line: "thrpt:  [1.8036 GiB/s 1.8537 GiB/s 1.8926 GiB/s]"
    thrpt_match = re.match(r"\s*thrpt:\s*\[[\d.]+ \S+ ([\d.]+) (\S+)", line)
    if thrpt_match:
        value = float(thrpt_match.group(1))
        unit = thrpt_match.group(2)
        if "GiB/s" in unit:
            return ("thrpt", 0.0, value)
        elif "MiB/s" in unit:
            return ("thrpt", 0.0, value / 1024)

    return None

def parse_benchmark_file(path: Path) -> Optional[PlatformResults]:
    """Parse a *_kernels.txt benchmark file."""
    content = path.read_text()

    tune_kind = parse_tune_kind(content)
    arch = parse_arch_from_caps(content)

    if not tune_kind or not arch:
        print(f"Warning: Could not detect platform from {path}", file=sys.stderr)
        return None

    results = PlatformResults(tune_kind=tune_kind, arch=arch)

    current_bench: Optional[str] = None
    current_time: Optional[float] = None

    for line in content.split("\n"):
        # Match benchmark name: "kernels/crc16/ccitt/aarch64/pmull-small/xs@vec"
        bench_match = re.match(r"^(kernels/\S+)", line)
        if bench_match:
            current_bench = bench_match.group(1)
            current_time = None
            continue

        if current_bench:
            parsed = parse_benchmark_line(line)
            if parsed:
                kind, time_ns, throughput = parsed
                if kind == "time":
                    current_time = time_ns
                elif kind == "thrpt" and current_time is not None:
                    # We have both time and throughput, record the result
                    # Parse: kernels/{crc_type}/{variant}/{arch}/{kernel}/{size}@{align}
                    parts = current_bench.split("/")
                    if len(parts) >= 6:
                        crc_type = parts[1]  # e.g., "crc16"
                        variant = parts[2]   # e.g., "ccitt"
                        kern_arch = parts[3] # e.g., "aarch64"
                        kernel = parts[4]    # e.g., "pmull-small"
                        size_align = parts[5] # e.g., "xs@vec"
                        size_class = size_align.split("@")[0]

                        key = (crc_type, variant)
                        if key not in results.variants:
                            results.variants[key] = VariantResults(crc_type, variant)

                        if size_class not in results.variants[key].results:
                            results.variants[key].results[size_class] = []

                        kernel_path = f"{kern_arch}/{kernel}"
                        results.variants[key].results[size_class].append(
                            BenchResult(kernel_path, size_class, current_time, throughput)
                        )

                    current_bench = None
                    current_time = None

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Analysis: Find Optimal Kernels
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_kernels(platform: PlatformResults) -> PlatformTable:
    """
    For each (variant, size_class), find the kernel with highest throughput.
    """
    table = PlatformTable(
        tune_kind=platform.tune_kind,
        arch=platform.arch,
    )

    for (crc_type, variant), var_results in platform.variants.items():
        table.selections[(crc_type, variant)] = {}

        for size_class in SIZE_CLASSES:
            if size_class not in var_results.results:
                continue

            results = var_results.results[size_class]
            if not results:
                continue

            # Find highest throughput (best kernel)
            best = max(results, key=lambda r: r.throughput_gib_s)
            table.selections[(crc_type, variant)][size_class] = KernelSelection(
                kernel_path=best.kernel_path,
                throughput_gib_s=best.throughput_gib_s,
            )

    return table

# ─────────────────────────────────────────────────────────────────────────────
# Code Generation
# ─────────────────────────────────────────────────────────────────────────────

def kernel_path_to_rust_fn(kernel_path: str, crc_type: str, variant: str) -> str:
    """
    Convert kernel path to Rust function reference.

    e.g., "aarch64/pmull-eor3-3way" + "crc64" + "xz"
          -> "aarch64::crc64_xz_pmull_eor3_3way"
    """
    arch, kernel = kernel_path.split("/", 1)

    # Normalize kernel name for Rust identifier
    kernel_rust = kernel.replace("-", "_")

    # Build function name based on variant
    _, _, rust_variant = VARIANT_INFO[(crc_type, variant)]

    if arch == "portable":
        # Portable kernels: portable::{variant}_{kernel}
        return f"portable::{rust_variant}_{kernel_rust}"
    else:
        # SIMD kernels: {arch}::{variant}_{kernel}
        return f"{arch}::{rust_variant}_{kernel_rust}"

def generate_kernel_set_struct() -> str:
    """Generate the KernelSet struct definition."""
    fields = []
    for crc_type, variant in CRC_VARIANTS:
        fn_type, _, rust_name = VARIANT_INFO[(crc_type, variant)]
        fields.append(f"  pub {rust_name}: {fn_type},")

    return f"""\
/// All kernel function pointers for one size class.
#[derive(Clone, Copy)]
pub struct KernelSet {{
{chr(10).join(fields)}
}}
"""

def generate_kernel_table_struct() -> str:
    """Generate the KernelTable struct definition."""
    return """\
/// Complete kernel table for one platform.
///
/// Contains pre-selected optimal kernels for each (variant, size_class) pair.
/// Size class boundaries define when to transition between kernel tiers.
#[derive(Clone, Copy)]
pub struct KernelTable {
  /// Size class boundaries: [xs_max, s_max, m_max]
  /// - data.len() <= xs_max -> use tiny kernels
  /// - data.len() <= s_max -> use small kernels
  /// - data.len() <= m_max -> use medium kernels
  /// - else -> use large kernels
  pub boundaries: [usize; 3],

  /// Kernels for tiny buffers (0 to xs_max bytes)
  pub xs: KernelSet,
  /// Kernels for small buffers (xs_max+1 to s_max bytes)
  pub s: KernelSet,
  /// Kernels for medium buffers (s_max+1 to m_max bytes)
  pub m: KernelSet,
  /// Kernels for large buffers (m_max+1+ bytes)
  pub l: KernelSet,
}
"""

def generate_platform_table(table: PlatformTable) -> str:
    """Generate Rust code for one platform's kernel table."""
    tune_kind = table.tune_kind

    def get_kernel(crc_type: str, variant: str, size_class: str) -> str:
        key = (crc_type, variant)
        if key in table.selections and size_class in table.selections[key]:
            sel = table.selections[key][size_class]
            return kernel_path_to_rust_fn(sel.kernel_path, crc_type, variant)
        # Fallback to portable
        _, _, rust_variant = VARIANT_INFO[(crc_type, variant)]
        return f"portable::{rust_variant}_slice"

    def kernel_set(size_class: str) -> str:
        lines = []
        for crc_type, variant in CRC_VARIANTS:
            _, _, rust_name = VARIANT_INFO[(crc_type, variant)]
            fn_ref = get_kernel(crc_type, variant, size_class)
            lines.append(f"      {rust_name}: {fn_ref},")
        return "\n".join(lines)

    # Use xs boundary for tiny, s for small, m for medium
    # "l" and "xl" both use "large" kernels (the "l" selection)
    boundaries = f"[{SIZE_CLASS_BYTES['xs']}, {SIZE_CLASS_BYTES['s']}, {SIZE_CLASS_BYTES['m']}]"

    return f"""\
/// Kernel table for {tune_kind}.
///
/// Generated from benchmark data. Do not edit manually.
pub static {tune_kind.upper()}_TABLE: KernelTable = KernelTable {{
  boundaries: {boundaries},
  xs: KernelSet {{
{kernel_set('xs')}
  }},
  s: KernelSet {{
{kernel_set('s')}
  }},
  m: KernelSet {{
{kernel_set('m')}
  }},
  l: KernelSet {{
{kernel_set('l')}
  }},
}};
"""

def generate_dispatch_fn() -> str:
    """Generate the dispatch selection function."""
    return """\
/// Select the appropriate kernel table for the current platform.
///
/// Resolution order:
/// 1. Exact TuneKind match (benchmarked platform)
/// 2. Family match (inferred from similar hardware)
/// 3. Capability-based fallback (conservative defaults)
/// 4. Portable fallback (no SIMD)
#[inline]
pub fn select_table(tune_kind: TuneKind, caps: Caps) -> &'static KernelTable {
  // 1. Exact match
  if let Some(table) = exact_match(tune_kind) {
    return table;
  }

  // 2. Family match
  if let Some(table) = family_match(tune_kind) {
    return table;
  }

  // 3. Capability match
  if let Some(table) = capability_match(caps) {
    return table;
  }

  // 4. Portable fallback
  &PORTABLE_TABLE
}

#[inline]
fn exact_match(tune_kind: TuneKind) -> Option<&'static KernelTable> {
  match tune_kind {
    // Benchmarked platforms - add entries as benchmark data is collected
    TuneKind::AppleM1M3 => Some(&APPLEM1M3_TABLE),
    // TuneKind::Zen4 => Some(&ZEN4_TABLE),
    // TuneKind::IntelSpr => Some(&INTELSPR_TABLE),
    _ => None,
  }
}

#[inline]
fn family_match(tune_kind: TuneKind) -> Option<&'static KernelTable> {
  match tune_kind {
    // Apple Silicon family
    TuneKind::AppleM4 | TuneKind::AppleM5 => Some(&APPLEM1M3_TABLE),

    // AWS Graviton / ARM Neoverse family
    // TuneKind::Graviton3 | TuneKind::Graviton4 | TuneKind::Graviton5 => Some(&GRAVITON3_TABLE),
    // TuneKind::NeoverseN2 | TuneKind::NeoverseN3 | TuneKind::NeoverseV3 => Some(&GRAVITON3_TABLE),

    // AMD Zen family
    // TuneKind::Zen5 | TuneKind::Zen5c => Some(&ZEN4_TABLE),

    // Intel family
    // TuneKind::IntelGnr => Some(&INTELSPR_TABLE),

    _ => None,
  }
}

#[inline]
fn capability_match(caps: Caps) -> Option<&'static KernelTable> {
  #[cfg(target_arch = "aarch64")]
  {
    if caps.has(platform::caps::aarch64::SHA3) && caps.has(platform::caps::aarch64::PMULL_READY) {
      return Some(&GENERIC_ARM_PMULL_EOR3_TABLE);
    }
    if caps.has(platform::caps::aarch64::PMULL_READY) {
      return Some(&GENERIC_ARM_PMULL_TABLE);
    }
  }

  #[cfg(target_arch = "x86_64")]
  {
    if caps.has(platform::caps::x86::VPCLMUL_READY) {
      return Some(&GENERIC_X86_VPCLMUL_TABLE);
    }
    if caps.has(platform::caps::x86::PCLMUL_READY) {
      return Some(&GENERIC_X86_PCLMUL_TABLE);
    }
  }

  None
}
"""

def generate_file_header() -> str:
    """Generate the file header."""
    return """\
//! Auto-generated kernel dispatch tables.
//!
//! This file is generated by `scripts/gen/kernel_tables.py` from benchmark data
//! in `crates/checksum/bench_baseline/`. Do not edit manually.
//!
//! To regenerate: `python scripts/gen/kernel_tables.py`
//!
//! # Design
//!
//! Each platform has a `KernelTable` containing optimal kernels for each
//! (CRC variant, size class) combination. At runtime, we:
//!
//! 1. Detect the platform once (via `platform::tune()`)
//! 2. Select the appropriate `KernelTable`
//! 3. For each CRC call, branch on buffer size and call the optimal kernel
//!
//! This eliminates the per-call policy computation overhead (~5ns -> ~1.5ns).

#![allow(unused_imports)]
#![allow(dead_code)]

use platform::{Caps, TuneKind};

use crate::dispatchers::{Crc16Fn, Crc24Fn, Crc32Fn, Crc64Fn};

// Import kernel modules
#[cfg(target_arch = "aarch64")]
use crate::{
  crc16::aarch64 as crc16_aarch64,
  crc24::aarch64 as crc24_aarch64,
  crc32::aarch64 as crc32_aarch64,
  crc64::aarch64 as crc64_aarch64,
};

#[cfg(target_arch = "x86_64")]
use crate::{
  crc16::x86_64 as crc16_x86_64,
  crc24::x86_64 as crc24_x86_64,
  crc32::x86_64 as crc32_x86_64,
  crc64::x86_64 as crc64_x86_64,
};

use crate::{
  crc16::portable as crc16_portable,
  crc24::portable as crc24_portable,
  crc32::portable as crc32_portable,
  crc64::portable as crc64_portable,
};

"""

def main():
    repo_root = Path(__file__).parent.parent.parent
    bench_dir = repo_root / "crates" / "checksum" / "bench_baseline"
    output_dir = repo_root / "crates" / "checksum" / "src" / "generated"

    # Find all kernel benchmark files
    kernel_files = list(bench_dir.glob("*_kernels.txt"))
    if not kernel_files:
        print(f"No *_kernels.txt files found in {bench_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(kernel_files)} benchmark files:")
    for f in kernel_files:
        print(f"  - {f.name}")

    # Parse all files
    platforms: list[PlatformResults] = []
    for path in kernel_files:
        result = parse_benchmark_file(path)
        if result:
            platforms.append(result)
            print(f"  Parsed {path.name}: {result.tune_kind} ({result.arch})")
            print(f"    Variants: {list(result.variants.keys())}")

    if not platforms:
        print("No valid benchmark data found", file=sys.stderr)
        sys.exit(1)

    # Analyze and find optimal kernels
    tables: list[PlatformTable] = []
    for platform in platforms:
        table = find_optimal_kernels(platform)
        tables.append(table)
        print(f"\nOptimal kernels for {table.tune_kind}:")
        for (crc_type, variant), selections in table.selections.items():
            print(f"  {crc_type}/{variant}:")
            for size_class, sel in selections.items():
                print(f"    {size_class}: {sel.kernel_path} ({sel.throughput_gib_s:.2f} GiB/s)")

    # Generate output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kernel_tables.rs"

    with open(output_path, "w") as f:
        f.write(generate_file_header())
        f.write("\n")
        f.write(generate_kernel_set_struct())
        f.write("\n")
        f.write(generate_kernel_table_struct())
        f.write("\n")

        for table in tables:
            f.write(generate_platform_table(table))
            f.write("\n")

        # TODO: Generate fallback tables and dispatch function
        # f.write(generate_dispatch_fn())

    print(f"\nGenerated: {output_path}")

if __name__ == "__main__":
    main()
