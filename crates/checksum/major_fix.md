# Major Fix: Kernel Selection & Dispatch Overhead

## Executive Summary

Deep analysis of the benchmark data revealed two distinct problems:

1. **Wrong Kernel Selection** - The policy system selects suboptimal kernels (up to 36% slower than best available)
2. **Dispatch Overhead** - ~5ns per-call overhead from dynamic dispatch

These issues affect **all 7 CRC variants** across **all platforms**. This document outlines a comprehensive fix that:
- Guarantees optimal kernel selection based on empirical benchmarks
- Reduces dispatch overhead to <2ns
- Handles unknown platforms gracefully
- Scales to future algorithms (hashes, AEAD)

---

## Scope: Everything Is Equal

This fix treats all dimensions equally:

### All 7 Variants
| Variant | Width | Use Cases |
|---------|-------|-----------|
| `crc64_xz` | 64-bit | XZ compression, 7-Zip, LZMA |
| `crc64_nvme` | 64-bit | NVMe storage specification |
| `crc32` | 32-bit | Ethernet, PNG, ZIP, gzip |
| `crc32c` | 32-bit | iSCSI, SCTP, Btrfs, ext4 |
| `crc16_ccitt` | 16-bit | X.25, HDLC, Bluetooth |
| `crc16_ibm` | 16-bit | USB, Modbus, legacy systems |
| `crc24_openpgp` | 24-bit | OpenPGP/GPG armor |

### All Platforms
| Platform | Architecture | Examples |
|----------|--------------|----------|
| Apple Silicon | aarch64 | M1, M2, M3, M4 MacBooks/iMacs |
| AWS Graviton | aarch64 | Graviton2, Graviton3, Graviton4 |
| ARM Neoverse | aarch64 | Ampere Altra, cloud instances |
| AMD Zen | x86-64 | Zen3, Zen4, Zen5, EPYC |
| Intel Server | x86-64 | Ice Lake, Sapphire Rapids, Granite Rapids |
| Intel Client | x86-64 | Alder Lake, Raptor Lake |
| Generic ARM | aarch64 | Any with PMULL |
| Generic x86 | x86-64 | Any with PCLMUL |
| Portable | any | Fallback for unknown |

### All Buffer Sizes
| Class | Range | Use Cases |
|-------|-------|-----------|
| Tiny | 0-128B | Network headers, small packets |
| Small | 128B-2KB | Typical network MTU, small files |
| Medium | 2KB-64KB | Database pages, medium files |
| Large | 64KB+ | Bulk data, large file hashing |

---

## Problem 1: Wrong Kernel Selection

### Evidence

| Platform | Variant | Size | Gap | Current | Optimal |
|----------|---------|------|-----|---------|---------|
| Linux x86-64 | crc16_ccitt | 256B | **36%** | pclmul-4way | vpclmul-2way |
| Linux x86-64 | crc64_xz | 64KB | **29%** | vpclmul-4x512 | vpclmul-2way |
| macOS ARM64 | crc16_ibm | 256B | **22%** | pmull | pmull-small |
| macOS ARM64 | crc16_ccitt | 256B | **21%** | pmull | pmull-small |
| Linux ARM64 | crc16_ccitt | 256B | **16%** | pmull | pmull-small |
| Linux ARM64 | crc16_ibm | 256B | **15%** | pmull | pmull-small |

### Root Cause

The policy system computes kernel selection at runtime using hand-tuned thresholds that don't match reality:
- Stream count selection based on assumptions, not measurements
- Crossover thresholds are approximate
- Per-variant tuning is incomplete (CRC16/CRC24 use generic defaults)

---

## Problem 2: Dispatch Overhead

### Evidence (macOS ARM64, CRC64/XZ)

| Size | Raw Kernel | Through API | Overhead | Impact |
|------|------------|-------------|----------|--------|
| 64B | 5.4ns | 10ns | 4.6ns | **85%** |
| 256B | 8.3ns | 13.3ns | 5ns | **60%** |
| 4KB | 66ns | 75ns | 9ns | **13%** |
| 64KB | 950ns | 986ns | 36ns | **4%** |

### Root Cause

Every call executes:
```
update() → self.kernel() → auto() → get_or_init() → policy_dispatch() → kernel()
```

The `policy_dispatch()` function has 6+ branches, stream computation, and array indexing.

---

## Solution: Empirical Kernel Tables

### Core Principle

**Replace runtime computation with pre-computed lookup tables derived from actual benchmarks.**

```
Current:  caps → policy → thresholds → streams → kernel_name → lookup → kernel
Proposed: platform → size_class → kernel (direct)
```

### The 3D Matrix

Every cell has a specific, benchmarked-optimal kernel:

```
KERNEL_TABLES[platform][size_class][variant] → kernel_fn

Example for Apple M1-M3:
┌─────────────┬──────────────┬──────────────┬──────────────┬─────────────────┐
│ Variant     │ tiny (<128B) │ small (<2KB) │ medium(<64KB)│ large (64KB+)   │
├─────────────┼──────────────┼──────────────┼──────────────┼─────────────────┤
│ crc64_xz    │ pmull-small  │ pmull-eor3   │ pmull-eor3   │ pmull-eor3-3way │
│ crc64_nvme  │ pmull-small  │ pmull-eor3   │ pmull-eor3   │ pmull-eor3-3way │
│ crc32       │ pmull-small  │ pmull-small  │ pmull-small  │ pmull-eor3-v9   │
│ crc32c      │ pmull-small  │ pmull-small  │ pmull-small  │ pmull-eor3-v9   │
│ crc16_ccitt │ pmull-small  │ pmull-small  │ pmull-2way   │ pmull-3way      │
│ crc16_ibm   │ pmull-small  │ pmull-small  │ pmull-3way   │ pmull-3way      │
│ crc24       │ pmull-small  │ pmull        │ pmull        │ pmull           │
└─────────────┴──────────────┴──────────────┴──────────────┴─────────────────┘
```

---

## Runtime Flow

### Phase 1: Compile Time (Our Build)

All kernel tables for all platforms are compiled into the binary:

```rust
static KERNEL_TABLES: &[(TuneKind, KernelTable)] = &[
    // Benchmarked platforms (exact data)
    (TuneKind::AppleM1M3, KernelTable { ... }),
    (TuneKind::AppleM4, KernelTable { ... }),
    (TuneKind::Graviton3, KernelTable { ... }),
    (TuneKind::Graviton4, KernelTable { ... }),
    (TuneKind::Zen4, KernelTable { ... }),
    (TuneKind::Zen5, KernelTable { ... }),
    (TuneKind::IntelSPR, KernelTable { ... }),
    (TuneKind::IntelGNR, KernelTable { ... }),

    // Inferred platforms (extrapolated from similar hardware)
    (TuneKind::AppleM5, KernelTable { ... }),      // From M4 data
    (TuneKind::Graviton5, KernelTable { ... }),    // From Graviton4 data
    (TuneKind::NeoverseN3, KernelTable { ... }),   // From Graviton3 data

    // Capability-based fallbacks (for unknown specific platforms)
    (TuneKind::GenericArmPmullEor3, KernelTable { ... }),
    (TuneKind::GenericArmPmull, KernelTable { ... }),
    (TuneKind::GenericX86Vpclmul, KernelTable { ... }),
    (TuneKind::GenericX86Pclmul, KernelTable { ... }),

    // Ultimate fallback (portable, no SIMD)
    (TuneKind::Portable, KernelTable { ... }),
];
```

### Phase 2: Runtime - First Call (Once)

```rust
fn init_kernel_table() -> &'static KernelTable {
    // 1. Detect CPU capabilities
    let caps = platform::caps();
    // → Caps { pmull: true, sha3: true, sve2: false, vpclmul: false, ... }

    // 2. Detect microarchitecture
    let tune_kind = platform::tune().kind;
    // → TuneKind::AppleM1M3

    // 3. Find matching table (see "Platform Resolution" below)
    let table = resolve_platform_table(tune_kind, caps);

    // 4. Verify capabilities (safety check)
    assert!(table.is_safe_for(caps));

    table
}

// Cached in OnceLock - resolved exactly once
static ACTIVE_TABLE: OnceLock<&'static KernelTable> = OnceLock::new();
```

### Phase 3: Runtime - Every Call (~1-2ns)

```rust
#[inline]
pub fn crc64_xz(data: &[u8]) -> u64 {
    let table = ACTIVE_TABLE.get_or_init(init_kernel_table);
    let kernel = table.select_kernel(Variant::Crc64Xz, data.len());
    kernel(!0, data) ^ !0
}

impl KernelTable {
    #[inline]
    fn select_kernel(&self, variant: Variant, len: usize) -> KernelFn {
        let class = if len <= self.boundaries[0] {
            &self.tiny
        } else if len <= self.boundaries[1] {
            &self.small
        } else if len <= self.boundaries[2] {
            &self.medium
        } else {
            &self.large
        };
        class.get(variant)
    }
}
```

---

## Handling Unknown Platforms

### The Challenge

We can only benchmark platforms we have access to. Users will run on:
- Future hardware (M5, Graviton5, Zen6, etc.)
- Niche hardware (Ampere Altra, Fujitsu A64FX, etc.)
- Cloud instances with unknown microarchitecture
- Embedded/IoT ARM devices

### Resolution Strategy (Ordered)

```rust
fn resolve_platform_table(tune_kind: TuneKind, caps: Caps) -> &'static KernelTable {
    // 1. EXACT MATCH: We have benchmark data for this specific platform
    if let Some(table) = find_exact_match(tune_kind) {
        return table;
    }

    // 2. FAMILY MATCH: Use data from same microarchitecture family
    if let Some(table) = find_family_match(tune_kind) {
        return table;
    }

    // 3. CAPABILITY MATCH: Use conservative table based on CPU features
    if let Some(table) = find_capability_match(caps) {
        return table;
    }

    // 4. PORTABLE: Ultimate fallback, no SIMD
    &PORTABLE_TABLE
}
```

### Level 1: Exact Match (Benchmarked)

Platforms we have actual benchmark data for:

| TuneKind | Source | Confidence |
|----------|--------|------------|
| AppleM1M3 | Benchmarked on M1/M2/M3 | **High** |
| AppleM4 | Benchmarked on M4 | **High** |
| Graviton3 | Benchmarked on AWS | **High** |
| Zen4 | Benchmarked on Ryzen 7000 | **High** |
| IntelSPR | Benchmarked on Xeon | **High** |

### Level 2: Family Match (Inferred)

For platforms in the same family, we extrapolate:

| Unknown Platform | Use Data From | Rationale |
|-----------------|---------------|-----------|
| AppleM5 | AppleM4 | Same ISA, similar μarch |
| Graviton4/5 | Graviton3 | Neoverse evolution |
| NeoverseN2/N3/V2/V3 | Graviton3 | Same Neoverse family |
| Zen5/Zen5c | Zen4 | Same ISA, similar μarch |
| IntelGNR | IntelSPR | Same AVX-512 profile |
| AmpereAltra | Graviton2 | Similar Neoverse N1 base |

```rust
fn find_family_match(tune_kind: TuneKind) -> Option<&'static KernelTable> {
    match tune_kind {
        // Apple Silicon family
        TuneKind::AppleM5 => Some(&TABLES[AppleM4]),
        TuneKind::AppleM6 => Some(&TABLES[AppleM4]),

        // AWS Graviton / ARM Neoverse family
        TuneKind::Graviton4 => Some(&TABLES[Graviton3]),
        TuneKind::Graviton5 => Some(&TABLES[Graviton3]),
        TuneKind::NeoverseN2 => Some(&TABLES[Graviton3]),
        TuneKind::NeoverseN3 => Some(&TABLES[Graviton3]),
        TuneKind::NeoverseV2 => Some(&TABLES[Graviton3]),
        TuneKind::NeoverseV3 => Some(&TABLES[Graviton3]),
        TuneKind::NvidiaGrace => Some(&TABLES[Graviton3]),
        TuneKind::AmpereAltra => Some(&TABLES[Graviton2]),

        // AMD Zen family
        TuneKind::Zen5 => Some(&TABLES[Zen4]),
        TuneKind::Zen5c => Some(&TABLES[Zen4]),
        TuneKind::Zen6 => Some(&TABLES[Zen4]),

        // Intel family
        TuneKind::IntelGNR => Some(&TABLES[IntelSPR]),
        TuneKind::IntelMTL => Some(&TABLES[IntelICL]),
        TuneKind::IntelLNL => Some(&TABLES[IntelICL]),

        _ => None,
    }
}
```

### Level 3: Capability Match (Conservative)

For completely unknown platforms, use capability-based tables:

```rust
fn find_capability_match(caps: Caps) -> Option<&'static KernelTable> {
    // ARM platforms
    if caps.has(Caps::SVE2_PMULL) {
        // Has SVE2 + PMULL: use aggressive wide kernels
        return Some(&GENERIC_ARM_SVE2);
    }
    if caps.has(Caps::PMULL) && caps.has(Caps::SHA3) {
        // Has PMULL + EOR3: use EOR3-accelerated kernels
        return Some(&GENERIC_ARM_PMULL_EOR3);
    }
    if caps.has(Caps::PMULL) {
        // Has basic PMULL: use standard PMULL kernels
        return Some(&GENERIC_ARM_PMULL);
    }
    if caps.has(Caps::ARM_CRC) {
        // Has CRC32 instruction but no PMULL
        return Some(&GENERIC_ARM_CRC);
    }

    // x86 platforms
    if caps.has(Caps::VPCLMULQDQ) && caps.has(Caps::AVX512F) {
        // Has wide VPCLMUL: use AVX-512 kernels
        return Some(&GENERIC_X86_VPCLMUL_AVX512);
    }
    if caps.has(Caps::VPCLMULQDQ) {
        // Has VPCLMUL (AVX2): use 256-bit kernels
        return Some(&GENERIC_X86_VPCLMUL);
    }
    if caps.has(Caps::PCLMULQDQ) {
        // Has basic PCLMUL: use SSE kernels
        return Some(&GENERIC_X86_PCLMUL);
    }
    if caps.has(Caps::SSE42) {
        // Has CRC32C instruction
        return Some(&GENERIC_X86_CRC);
    }

    None  // Fall through to portable
}
```

### Level 4: Portable Fallback

When nothing else matches:

```rust
static PORTABLE_TABLE: KernelTable = KernelTable {
    boundaries: [64, 512, 4096],
    tiny:   KernelSet::all_portable(),
    small:  KernelSet::all_portable(),
    medium: KernelSet::all_slice16(),
    large:  KernelSet::all_slice16(),
};
```

### Capability Tables: Conservative by Design

The generic capability tables use **conservative** kernel selections:

```rust
// Generic ARM with PMULL+EOR3 (unknown specific microarchitecture)
static GENERIC_ARM_PMULL_EOR3: KernelTable = KernelTable {
    // Conservative boundaries (don't assume fast wide ops)
    boundaries: [128, 2048, 32768],

    tiny: KernelSet {
        // Small kernels are safe - low setup cost
        crc64_xz: pmull_small,
        crc64_nvme: pmull_small,
        crc32: pmull_small,
        crc32c: pmull_small,
        crc16_ccitt: pmull_small,
        crc16_ibm: pmull_small,
        crc24_openpgp: pmull_small,
    },
    small: KernelSet {
        // 1-way kernels - safe, no ILP assumptions
        crc64_xz: pmull_eor3,      // 1-way
        crc64_nvme: pmull_eor3,
        crc32: pmull_eor3_v9,
        crc32c: pmull_eor3_v9,
        crc16_ccitt: pmull,
        crc16_ibm: pmull,
        crc24_openpgp: pmull,
    },
    medium: KernelSet {
        // Still conservative - 1-way or 2-way max
        crc64_xz: pmull_eor3,
        crc64_nvme: pmull_eor3,
        crc32: pmull_eor3_v9,
        crc32c: pmull_eor3_v9,
        crc16_ccitt: pmull,
        crc16_ibm: pmull,
        crc24_openpgp: pmull,
    },
    large: KernelSet {
        // Allow 2-way for large buffers (safe ILP bet)
        crc64_xz: pmull_eor3_2way,
        crc64_nvme: pmull_eor3_2way,
        crc32: pmull_eor3_v9,
        crc32c: pmull_eor3_v9,
        crc16_ccitt: pmull_2way,
        crc16_ibm: pmull_2way,
        crc24_openpgp: pmull,
    },
};
```

### Why Conservative?

The capability-based fallbacks are intentionally conservative because:

1. **Unknown ILP characteristics** - Multi-way kernels assume the CPU can execute multiple CLMUL operations in parallel. Unknown CPUs might have narrower execution units.

2. **Unknown cache behavior** - Aggressive prefetching/streaming might hurt on some microarchitectures.

3. **Safe > Fast for unknown** - A 10% slower but reliable kernel is better than a kernel that's sometimes 20% faster but sometimes 30% slower.

4. **Users can tune** - If users know their platform, they can use `RSCRYPTO_FORCE_*` env vars to select specific kernels.

---

## Data Structures

```rust
/// Complete kernel table for one platform
pub struct KernelTable {
    /// Size class boundaries: [tiny_max, small_max, medium_max]
    /// Everything above medium_max uses large kernels
    pub boundaries: [usize; 3],

    /// Kernels for each size class
    pub tiny: KernelSet,
    pub small: KernelSet,
    pub medium: KernelSet,
    pub large: KernelSet,
}

/// All kernel function pointers for one size class
#[derive(Clone, Copy)]
pub struct KernelSet {
    pub crc64_xz: Crc64Fn,
    pub crc64_nvme: Crc64Fn,
    pub crc32: Crc32Fn,
    pub crc32c: Crc32Fn,
    pub crc16_ccitt: Crc16Fn,
    pub crc16_ibm: Crc16Fn,
    pub crc24_openpgp: Crc24Fn,
}

/// Function pointer types
pub type Crc64Fn = fn(u64, &[u8]) -> u64;
pub type Crc32Fn = fn(u32, &[u8]) -> u32;
pub type Crc16Fn = fn(u16, &[u8]) -> u16;
pub type Crc24Fn = fn(u32, &[u8]) -> u32;  // 24-bit in low bits of u32
```

---

## User Scenarios

### Scenario 1: Linux Distributed Team (Graviton3, mixed buffer sizes)

```rust
// Their code
let checksum = crc64_xz(&packet);  // 256 bytes typically

// What happens:
// 1. First call detects Graviton3
// 2. Loads GRAVITON3_TABLE (benchmarked on their exact hardware)
// 3. 256B → small class → pmull-eor3 kernel
// 4. Every subsequent call: ~1-2ns dispatch + kernel time
```

### Scenario 2: macOS Team (M3, huge buffers)

```rust
// Their code
let checksum = crc64_xz(&large_file);  // 100MB

// What happens:
// 1. First call detects AppleM1M3
// 2. Loads APPLE_M1M3_TABLE (benchmarked)
// 3. 100MB → large class → pmull-eor3-3way kernel
// 4. Gets ~62 GiB/s throughput
```

### Scenario 3: Windows Team (Unknown Intel, tiny buffers)

```rust
// Their code
for packet in packets {
    let checksum = crc32c(&packet);  // 64 bytes
}

// What happens:
// 1. First call detects "IntelUnknown" + VPCLMUL capability
// 2. No exact match → capability match → GENERIC_X86_VPCLMUL table
// 3. 64B → tiny class → pclmul-small kernel
// 4. Conservative but still SIMD-accelerated
```

### Scenario 4: Embedded ARM (Unknown, no PMULL)

```rust
// Their code
let checksum = crc16_ccitt(&sensor_data);

// What happens:
// 1. First call detects no SIMD capabilities
// 2. Falls back to PORTABLE_TABLE
// 3. Uses slice16 kernel (table-based, no SIMD)
// 4. Still correct, just slower
```

---

## Implementation Plan

### Phase 1: Benchmark Infrastructure

**Goal:** Machine-readable benchmark results for all (platform, variant, size) combinations.

```
scripts/
├── bench/
│   ├── run-kernel-matrix.sh      # Runs all kernel benchmarks
│   ├── parse-criterion-output.py # Extracts structured data
│   └── find-optimal-kernels.py   # Determines winners per cell
│
crates/checksum/bench_baseline/
├── macos_arm64.json              # Structured results
├── linux_arm64.json
├── linux_x86-64.json
└── windows_x86-64.json
```

**JSON Schema:**
```json
{
  "platform": "macos_arm64",
  "tune_kind": "AppleM1M3",
  "timestamp": "2024-01-15T10:30:00Z",
  "results": {
    "crc64_xz": {
      "xs": { "winner": "aarch64/pmull-small", "throughput_gib_s": 10.95 },
      "s":  { "winner": "aarch64/pmull-eor3", "throughput_gib_s": 28.66 },
      "m":  { "winner": "aarch64/pmull-eor3", "throughput_gib_s": 57.53 },
      "l":  { "winner": "aarch64/pmull-eor3-3way", "throughput_gib_s": 61.32 },
      "xl": { "winner": "aarch64/pmull-eor3-3way", "throughput_gib_s": 61.78 }
    },
    "crc64_nvme": { ... },
    "crc32": { ... },
    "crc32c": { ... },
    "crc16_ccitt": { ... },
    "crc16_ibm": { ... },
    "crc24_openpgp": { ... }
  }
}
```

### Phase 2: Code Generation

**Goal:** Generate Rust source from benchmark JSON.

```
crates/tune/src/
├── codegen.rs                    # Table generation logic
└── templates/
    └── kernel_tables.rs.template

crates/checksum/src/generated/
└── kernel_tables.rs              # Generated, committed
```

**Generated Code:**
```rust
// AUTO-GENERATED from bench_baseline/*.json
// Do not edit manually. Run `just regen-kernel-tables` to update.

pub static APPLE_M1M3_TABLE: KernelTable = KernelTable {
    boundaries: [128, 2048, 16384],
    tiny: KernelSet {
        crc64_xz: aarch64::crc64_xz_pmull_small,
        crc64_nvme: aarch64::crc64_nvme_pmull_small,
        crc32: aarch64::crc32_pmull_small,
        // ...
    },
    // ...
};

pub static GRAVITON3_TABLE: KernelTable = KernelTable { ... };
pub static ZEN4_TABLE: KernelTable = KernelTable { ... };
// ...
```

### Phase 3: New Dispatch System

**Goal:** Replace policy-based dispatch with table lookup.

**New Files:**
```
crates/checksum/src/
├── dispatch.rs      # New table-based dispatch
├── generated/
│   └── kernel_tables.rs
└── api/
    ├── oneshot.rs   # crc64_xz(data), crc32(data), etc.
    └── streaming.rs # Crc64Xz::new/update/finalize
```

**API:**
```rust
// Oneshot (new, recommended for single buffers)
pub fn crc64_xz(data: &[u8]) -> u64;
pub fn crc64_nvme(data: &[u8]) -> u64;
pub fn crc32(data: &[u8]) -> u32;
pub fn crc32c(data: &[u8]) -> u32;
pub fn crc16_ccitt(data: &[u8]) -> u16;
pub fn crc16_ibm(data: &[u8]) -> u16;
pub fn crc24_openpgp(data: &[u8]) -> u32;

// Streaming (existing, for incremental hashing)
pub struct Crc64Xz { ... }
pub struct Crc32 { ... }
// etc.
```

### Phase 4: Migration

**Goal:** All variants using new dispatch, old policy system deprecated.

- [ ] CRC64/XZ and CRC64/NVME
- [ ] CRC32 and CRC32C
- [ ] CRC16/CCITT and CRC16/IBM
- [ ] CRC24/OpenPGP
- [ ] Update streaming API internals
- [ ] Mark old policy API as `#[deprecated]`

### Phase 5: Validation

**Goal:** Prove we win everywhere.

- [ ] Benchmark all platforms with new system
- [ ] Compare against crc64fast, crc32fast, crc-fast
- [ ] Verify dispatch overhead <2ns
- [ ] Document any remaining gaps

### Phase 6: Release

- [ ] Update documentation
- [ ] Changelog
- [ ] Release

---

## Architecture Changes: What Happens to Existing Code

### Public API (v1 Release)

```rust
// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API - What users import and use
// ═══════════════════════════════════════════════════════════════════════════

// Oneshot functions (NEW - recommended)
pub fn crc64_xz(data: &[u8]) -> u64;
pub fn crc64_nvme(data: &[u8]) -> u64;
pub fn crc32(data: &[u8]) -> u32;
pub fn crc32c(data: &[u8]) -> u32;
pub fn crc16_ccitt(data: &[u8]) -> u16;
pub fn crc16_ibm(data: &[u8]) -> u16;
pub fn crc24_openpgp(data: &[u8]) -> u32;

// Streaming hashers (EXISTING - for incremental use)
pub struct Crc64Xz;      // impl: new(), update(&[u8]), finalize() -> u64
pub struct Crc64Nvme;
pub struct Crc32;
pub struct Crc32C;
pub struct Crc16Ccitt;
pub struct Crc16Ibm;
pub struct Crc24OpenPgp;

// Traits (EXISTING)
pub trait Hasher { ... }
pub trait Checksum { ... }
```

### Internal Code (Testing/Tuning Only)

```rust
// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL - Not exported, used for benchmarking and tuning only
// ═══════════════════════════════════════════════════════════════════════════

// Location: crates/checksum/src/internal/
mod internal {
    // Old policy system - INTERNAL ONLY
    // Used by: `cargo bench --bench kernels`, `just tune`
    // NOT used by: public API, user code

    pub(crate) mod policy;           // Policy computation (for analysis)
    pub(crate) mod tuned_defaults;   // Old threshold data (reference)
    pub(crate) mod dispatchers;      // Old dispatcher macros (deprecated)
}

// Location: crates/tune/
// The tuning crate uses internal APIs to:
// 1. Run kernel benchmarks
// 2. Analyze crossover points
// 3. Generate new kernel tables
```

### What Gets Removed from Hot Path

```
BEFORE (current):
┌─────────────────────────────────────────────────────────────────────────┐
│ update(data)                                                            │
│   → self.kernel(state, data)        [self.kernel = auto fn]            │
│       → get_or_init()               [OnceLock check: ~2-3ns]           │
│           → policy_dispatch()        [6+ branches: ~2ns]               │
│               → streams_for_len()    [division + compare]              │
│               → kernels[idx]         [array index]                     │
│                   → actual_kernel()  [indirect call]                   │
│                                                                         │
│ TOTAL OVERHEAD: ~5ns per call                                          │
└─────────────────────────────────────────────────────────────────────────┘

AFTER (new):
┌─────────────────────────────────────────────────────────────────────────┐
│ crc64_xz(data)                                                          │
│   → ACTIVE_TABLE.get()              [OnceLock load: ~0.5ns]            │
│   → if len <= boundaries[0]         [1-2 branches: ~0.5ns]             │
│       → kernel(state, data)         [direct call: ~0.5ns]              │
│                                                                         │
│ TOTAL OVERHEAD: ~1.5ns per call                                        │
└─────────────────────────────────────────────────────────────────────────┘

ELIMINATED:
- policy_dispatch() branches          [GONE - not in hot path]
- streams_for_len() computation       [GONE - pre-computed]
- Runtime threshold checks            [GONE - pre-computed]
- Kernel name → function lookup       [GONE - direct fn ptr]
- Double indirection                  [GONE - single call]
```

### File Organization (Post-Migration)

```
crates/checksum/src/
├── lib.rs                      # Public API re-exports
├── api/
│   ├── mod.rs
│   ├── oneshot.rs              # NEW: crc64_xz(), crc32(), etc.
│   └── streaming.rs            # Crc64Xz::new/update/finalize
├── dispatch.rs                 # NEW: Table-based dispatch (~50 lines)
├── generated/
│   └── kernel_tables.rs        # AUTO-GENERATED: All platform tables
├── kernels/                    # Kernel implementations (unchanged)
│   ├── aarch64.rs
│   ├── x86_64.rs
│   └── portable.rs
├── internal/                   # INTERNAL ONLY - not exported
│   ├── mod.rs                  # #![doc(hidden)]
│   ├── policy.rs               # Old policy (for tune crate)
│   ├── tuned_defaults.rs       # Old defaults (reference)
│   └── dispatchers.rs          # Old macros (deprecated)
└── tests/                      # Tests use public API only
```

### Migration Checklist

```
[ ] Move policy.rs → internal/policy.rs
[ ] Move tuned_defaults.rs → internal/tuned_defaults.rs
[ ] Move dispatchers.rs → internal/dispatchers.rs
[ ] Add #![doc(hidden)] to internal/mod.rs
[ ] Add #[deprecated] to old dispatch functions
[ ] Create api/oneshot.rs with new functions
[ ] Update api/streaming.rs to use new dispatch
[ ] Update lib.rs exports (only public API)
[ ] Update tune crate to use internal:: path
[ ] Verify benchmarks still work (use internal::)
[ ] Verify tests pass (use public API)
```

---

## Dispatch Overhead: Before vs After

### Current Overhead Breakdown (measured)

| Component | Time | Notes |
|-----------|------|-------|
| `get_or_init()` | ~2-3ns | OnceLock atomic check + branch |
| `policy_dispatch()` entry | ~0.5ns | Function call |
| Length threshold checks | ~1ns | 3-4 conditional branches |
| `streams_for_len()` | ~0.5ns | Division + comparison |
| Kernel array index | ~0.5ns | Bounds check + load |
| Indirect call | ~0.5ns | Function pointer call |
| **Total** | **~5-6ns** | |

### New Overhead Breakdown (target)

| Component | Time | Notes |
|-----------|------|-------|
| `ACTIVE_TABLE.get()` | ~0.5ns | Already initialized, just load |
| Size class branch | ~0.5ns | 2-3 predictable branches |
| Direct kernel call | ~0.5ns | Single function pointer |
| **Total** | **~1.5ns** | |

### Overhead Reduction

```
Current:  ~5ns overhead per call
New:      ~1.5ns overhead per call
Savings:  ~3.5ns per call (70% reduction)

Impact at different buffer sizes:
- 64B buffer (5ns kernel): 50% faster API throughput
- 256B buffer (8ns kernel): 30% faster API throughput
- 4KB buffer (66ns kernel): 5% faster API throughput
- 64KB+ buffer: <1% difference (kernel dominates)
```

---

## V1 Release Checklist

Upon completing this plan, you will have:

### Correctness
- [ ] All existing tests pass
- [ ] All variants produce correct checksums
- [ ] No undefined behavior (Miri clean)
- [ ] No panics on any input

### Performance
- [ ] Every (platform, variant, size) uses optimal kernel
- [ ] Dispatch overhead <2ns (measured)
- [ ] Beat or tie competitors at every benchmark point
- [ ] No regressions from current performance

### API Quality
- [ ] Clean public API (oneshot + streaming)
- [ ] Internal code clearly separated
- [ ] Documentation complete
- [ ] Examples compile and run

### Platform Support
- [ ] macOS ARM64 (Apple Silicon) - benchmarked
- [ ] Linux ARM64 (Graviton) - benchmarked
- [ ] Linux x86-64 (Zen4/Intel) - benchmarked
- [ ] Windows x86-64 - benchmarked
- [ ] Unknown platforms - safe fallback tested

### Maintenance
- [ ] Benchmark → JSON → Codegen pipeline works
- [ ] `just regen-kernel-tables` regenerates correctly
- [ ] Adding new platform is documented

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Correctness | 100% | Existing test suite passes |
| Optimal Selection | 100% | Every (platform, variant, size) uses fastest kernel |
| Dispatch Overhead | <2ns | Micro-benchmark |
| Win Rate | 100% | Beat or tie competitors at every benchmark point |
| Unknown Platform | Safe | Capability fallback works, no crashes |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Benchmark variance | Use median of 5+ runs, require >3% margin |
| New platforms | Capability-based fallback + family inference |
| Compiler changes | Re-benchmark on each release |
| Code size | ~50KB for all tables (acceptable) |
| Maintenance burden | Automated generation, single source of truth |

---

## Future: Hashes & AEAD

The same pattern scales:

```rust
pub struct KernelSet {
    // CRC (current)
    pub crc64_xz: Crc64Fn,
    // ...

    // Hashes (future)
    pub blake3_hash: HashFn,
    pub blake3_keyed: KeyedHashFn,
    pub sha256: HashFn,
    pub sha512: HashFn,
    pub sha3_256: HashFn,

    // AEAD (future)
    pub aes_gcm_256_seal: AeadSealFn,
    pub aes_gcm_256_open: AeadOpenFn,
    pub chacha20_poly1305_seal: AeadSealFn,
    pub chacha20_poly1305_open: AeadOpenFn,
}
```

Each new algorithm:
1. Implements kernels for each platform
2. Gets benchmarked by same infrastructure
3. Optimal kernel added to tables
4. Benefits from same low-overhead dispatch

---

## Summary

**The Contract:**

For any user, on any platform, with any buffer size, calling any CRC variant:
1. We detect their platform once
2. We select the empirically-fastest kernel for their exact situation
3. We dispatch with minimal overhead (<2ns)
4. If we don't know their platform, we fall back safely

**Generated from benchmarks. Verified by benchmarks. No assumptions.**
