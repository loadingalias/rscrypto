# Blake3 Final Work: Comprehensive Audit & Action Plan

Consolidated from five parallel deep-dive explorations of the rscrypto Blake3
implementation, tuning engine, bench baselines, CRC64 reference patterns, and
workspace topology.

---

## 1. Architecture Overview

### Three-Tier Dispatch System

```
Tier 1: Dispatch Layer (dispatch.rs)
  dispatch::resolved() → OnceCache<ResolvedDispatch>
    ├── Platform caps  (platform::caps())
    ├── Tuning hints   (platform::tune())
    └── Effective TuneKind (Intel SPR vs ICL split for AVX-512)

Tier 2: Dispatch Tables (dispatch_tables.rs)
  FamilyProfile per microarchitecture:
    ├── DispatchTable    (size-class kernel selection: XS/S/M/L)
    ├── StreamingTable   (stream vs bulk kernel + threshold)
    ├── ParallelTable    (spawn/merge costs, bytes-per-core tiers)
    └── StreamingParallelTable

Tier 3: Kernel Registry (kernels.rs)
  Blake3KernelId → Kernel { compress, hash_many_contiguous, simd_degree, name }
```

### Kernel Inventory

| ID | Arch | SIMD Degree | Compress | hash_many | Status |
|----|------|-------------|----------|-----------|--------|
| `Portable` | all | 1 | scalar 7-round | per-chunk loop | Complete |
| `X86Ssse3` | x86_64 | 1 (single-block) | pshufb rotations | delegates to portable | Dead in dispatch |
| `X86Sse41` | x86_64 | 4 | shift-OR rotations | 4-way transpose | Complete |
| `X86Avx2` | x86_64 | 8 | 256-bit G-function | 8-way + ASM backend | Complete |
| `X86Avx512` | x86_64 | 16 | native `rol_epi32` | 16-way + ASM backend | Complete |
| `Aarch64Neon` | aarch64 | 4 | `vsriq` rotations | 4-way + ASM fast paths | Complete |

### Dispatch Profiles (22 total)

**x86_64 (6 profiles):**
- Zen4, Zen5, Zen5c, Intel SPR, Intel GNR, Intel ICL

**aarch64 (5 profiles):**
- Apple M1/M3, Apple M4, Apple M5, Graviton2, Server NEON
- Graviton3/4/5, NeoverseN2/N3/V3, NvidiaGrace, AmpereAltra, Aarch64Pmull
  all map to Server NEON

**Scalar platforms (7 profiles):**
- Z13, Z14, Z15, Power7, Power8, Power9, Power10

**Meta:**
- Custom, Default, Portable

### Streaming State Machine

```
ChunkState {
  chaining_value: [u32; 8],  // running hash state
  chunk_counter: u64,
  block: [u8; 64],           // partial block buffer
  block_len: u8,             // 0-63
  blocks_compressed: u8,     // 0-15
  flags: u32,
  kernel: Kernel,
}

Update flow:
  Phase 1: Fill partial block buffer → compress if full
  Phase 2: Compress full blocks directly from caller slice
           Reserve final block for CHUNK_END flag
           Buffer remainder

aarch64 fast path: full 1024B chunk at boundary → ASM
```

### CV Stack & Tree Reduction

- Max depth: 54 (2^54 chunks = 64 EiB)
- Binary tree merging via `chunk_counter.trailing_zeros()`
- Batched CV reduction: `reduce_power_of_two_chunk_cvs()`
- Parallel tree output: Rayon-based with `SendPtr` for disjoint partitions

### Parallel Cost Model

```
ParallelTable {
  min_bytes, min_chunks, max_threads,
  spawn_cost_bytes, merge_cost_bytes,
  bytes_per_core_small, bytes_per_core_medium, bytes_per_core_large,
  small_limit_bytes, medium_limit_bytes,
}
```

Tuned per-profile via regression on measured parallel speedup curves.

---

## 2. Bench Baseline Assessment

### Baselines Present (7 files)

| File | Arch | CPU | Lines |
|------|------|-----|-------|
| `mac_arm_m1.txt` | aarch64 | Apple M1 | ~1207 |
| `linux_x86_zen4.txt` | x86_64 | AMD Zen 4 | ~2233 |
| `linux_x86_zen5.txt` | x86_64 | AMD Zen 5 | ~2173 |
| `linux_x86_intelspr.txt` | x86_64 | Intel SPR | ~2402 |
| `linux_x86_intelicl.txt` | x86_64 | Intel ICL | ~2235 |
| `linux_arm_graviton3.txt` | aarch64 | Graviton 3 | ~2222 |
| `linux_arm_graviton4.txt` | aarch64 | Graviton 4 | ~2191 |

`current_results.txt` and `previous_result.txt` are **empty (0 bytes)** — loss-tracking
workflow wired up but not actively used.

### Baselines Missing

| Platform | Has Profile? | Has Baseline? | Notes |
|----------|-------------|---------------|-------|
| Zen5c | Yes | **No** | Profile exists, no bench data |
| Intel GNR | Yes | **No** | Profile exists, no bench data |
| Apple M4 | Yes | **No** | Profile exists, no bench data |
| Apple M5 | Yes | **No** | Profile exists, no bench data |
| Graviton2 | Yes | **No** | Profile exists, no bench data |
| Z13/Z14/Z15 | Yes | **No** | Portable-only, no SIMD kernels |
| Power7-10 | Yes | **No** | Portable-only, no SIMD kernels |
| wasm32/64 | **No** | **No** | No kernel, no profile, no baseline |
| RISC-V | **No** | **No** | No kernel exists |
| Windows x86_64 | N/A | **No** | No Windows baselines at all |

### Benchmark Groups Covered

1. `blake3/oneshot/{rscrypto,official,official-rayon}` — 17 input sizes (0B–1MiB)
2. `blake3/streaming/{rscrypto,official}` — chunk sizes 64B–65536B over 1MiB
3. `blake3/update-overhead` — raw streaming overhead per chunk size
4. `blake3/parent-folding` — kernel variants on 1MiB of CVs
5. `blake3/xof` — init+read with input/output size combinations
6. `blake3/xof-sized-comparison` — standard vs sized API
7. `blake3/keyed` — streaming keyed hashing
8. `blake3/derive-key` — derive-key streaming
9. `blake3/active-kernel` — reports selected kernel per size
10. `blake3/rscrypto-threads` — multi-threaded (t1/t2/t4/t8)
11. `blake3/streaming-dispatch` — dispatch details (plain/keyed/derive)

### Criterion Settings

| Group | Samples | Warmup | Measurement |
|-------|---------|--------|-------------|
| Oneshot | 40 | 2s | 4s |
| Streaming | 30 | 2s | 4s |
| XOF | 20 | 1s | 3s |
| Keyed/derive | 25 | 1s | 3s |
| Dispatch details | 10 | 200ms | 500ms |
| Threads | 20 | 2s | 4s |

### Comparison Tooling

`scripts/bench/comp-check.py`:
- Parses Criterion output, normalizes units
- Identifies losses (rscrypto slower than official/official-rayon)
- Supports parallel-size thresholding (rayon comparison only for >= 512KiB)
- Gates: `--gate-single-thread fail`, `--gate-parallel warn`
- Exit codes: 0 (pass), 1 (failure), 2 (parse error)

---

## 3. Tuning Engine Pipeline

### Full Pipeline: Measure -> Derive -> Apply

```
Step 1: MEASUREMENT (crates/tune/src/bin/rscrypto-tune.rs)
  just tune blake3 [--apply] [--quick]
  ├── BenchRunner collects throughput at multiple buffer sizes
  ├── Per-size-class best kernel selection
  ├── Blake3 parallel speedup curves (1 -> max threads)
  └── Output: raw-results.json

Step 2: DERIVATION (crates/tune/src/engine.rs, analysis.rs)
  ├── Kernel selection per size class (best throughput wins)
  ├── Blake3 parallel policy derivation (blake3_adapter.rs)
  │   ├── Fit crossover points: where parallel > single-threaded
  │   ├── Robust linear regression for bytes-per-core
  │   ├── Outlier rejection, fit quality validation (<=60% error)
  │   └── Output: ParallelTable (10 parameters)
  └── Streaming tuning: stream vs bulk crossover -> bulk_sizeclass_threshold

Step 3: APPLY (crates/tune/src/apply.rs)
  ├── Generate per-TuneKind Rust code
  ├── Find marker comment in dispatch_tables.rs
  ├── Replace section between markers
  └── Write atomically to disk
```

### Blake3 Tuning Corpus (17 surfaces)

- 3 oneshot: default, keyed, derive
- 3 primitives: chunk, parent, parent-fold
- 11 streaming: 64B/4K updates x base + keyed + derive + XOF, plus mixed

### Parallel Policy Tuning Algorithm

1. Measure single-threaded at 13 sizes (64K-8M)
2. Measure multi-threaded (2, 4, 8, 12, 16, max) at same sizes
3. Find crossover: where `parallel_tp * 1.02 > single_tp`
4. Fit line: `bytes_per_thread = slope * threads + intercept`
5. Derive spawn/merge costs + 3-tier thresholds
6. Validate fit quality; fallback to defaults if outlier-heavy

### Platform Detection -> Runtime Dispatch

```
platform::caps() -> Caps (CPU features, 256-bit bitset)
platform::tune() -> Tune {
  kind: TuneKind,
  simd_threshold, pclmul_threshold, hwcrc_threshold,
  effective_simd_width, fast_wide_ops, parallel_streams,
  prefer_hybrid, cache_line, prefetch_distance,
  sve_vlen, sme_tile, memory_bound_hwcrc,
}

Dispatch pattern:
  dispatch(|caps, tune| { ... })  // ~3ns cached lookup
  dispatch_static(|caps, tune| { ... })  // compile-time only, zero overhead
```

---

## 4. Gap Analysis: Blake3 vs CRC64 Gold Standard

CRC64 is the canonical reference (per CLAUDE.md). Five patterns define its
gold-standard status:

### Pattern 1: Empirical Dispatch Tables

| Aspect | CRC64 | Blake3 |
|--------|-------|--------|
| Dispatch resolution | exact_match -> family_match -> capability_match -> portable (4 tiers) | flat `match` on TuneKind (2 tiers: exact or default) |
| KernelTable | Pre-computed `KernelSet { func, name }` per (variant, size_class) | `DispatchTable { xs, s, m, l: KernelId }` + separate streaming/parallel |
| Capability gate | `requires: Caps` on each table | Validated in `dispatch::resolve()` |
| Runtime cost | ~1.5ns (OnceCell lookup) | ~1.5ns (OnceCache lookup) |

**Gap**: Blake3 `select_profile()` is a flat match with no intermediate
capability-based fallback. CRC has 4-tier resolution.

### Pattern 2: Per-Kernel Benchmarks

| Aspect | CRC64 | Blake3 |
|--------|-------|--------|
| Bench file | `benches/kernels.rs` — each kernel x each size x alignment | `benches/blake3.rs` — rscrypto vs official comparison only |
| Baseline granularity | Per-kernel throughput | Aggregate auto-dispatched throughput |
| Can identify kernel regression? | Yes | **No** |

**Gap**: No per-kernel isolation benchmarks for Blake3.

### Pattern 3: Cross-Check Tests

| Aspect | CRC64 | Blake3 |
|--------|-------|--------|
| Test count | 60+ (all kernels x all lengths x streaming x combine x alignment) | `kernel_test.rs` exists but tests kernels at limited sizes |
| Reference oracle | `crc64_bitwise()` — obviously correct | Portable compress — correct but cross-check scope unclear |
| Streaming chunk sizes | `[1, 3, 7, 13, 17, 31, 37, 61, 127, 251]` | Standard crypto test vectors |
| Single-byte streaming | Yes | Unknown |
| Unaligned access tests | Yes (offsets 0-15) | Unknown |

**Gap**: Blake3 has `kernel_test.rs` but scope is narrower than CRC64's
exhaustive cross-check pattern.

### Pattern 4: Forced Kernel Mode

| Aspect | CRC64 | Blake3 |
|--------|-------|--------|
| Env var | `RSCRYPTO_CRC64_FORCE=portable\|pclmul\|vpclmul` | **None** |
| Programmatic | Available | **None** |
| Use case | A/B testing, regression diagnosis | Cannot debug dispatch vs kernel issues |

**Gap**: No way to force a specific Blake3 kernel without code changes.

### Pattern 5: Formal Kernel Tiers

| Aspect | CRC64 | Blake3 |
|--------|-------|--------|
| Tier 0 (Reference) | `crc64_bitwise()` | N/A (no bitwise Blake3) |
| Tier 1 (Portable) | `crc64_slice16_xz()` | `Blake3KernelId::Portable` |
| Tier 3 (HW Accel) | PCLMUL, PMULL, VPMSUMD, VGFM, Zbc | SSE4.1, NEON |
| Tier 4 (Wide SIMD) | VPCLMUL, PMULL+EOR3, SVE2, ZVBC | AVX2, AVX-512 |
| Documentation | Explicit tier comments | Flat enum, no tier annotations |

**Gap**: Blake3 kernels have no formal tier structure. The `backend` crate
already defines `KernelTier` (Reference, Portable, HwCrc, Folding, Wide) that
CRC uses but Blake3 does not leverage.

---

## 5. Code Quality Issues

### 5a. Arithmetic Rule Violations

**`wrapping_add` for a length counter** (`mod.rs:2141`):
```rust
self.block_len = self.block_len.wrapping_add(take as u8);
```
Per CLAUDE.md: counters/lengths/indices must use `strict_*`.
`block_len` is a length (0-64). Overflow would be a bug, not intentional
wraparound. Fix: `strict_add`.

**`saturating_sub` for a counter** (`mod.rs:2190`):
```rust
blocks_to_compress = blocks_to_compress.saturating_sub(1);
```
Per CLAUDE.md: `saturating_*` should be avoided for counters.
Fix: guard with `if blocks_to_compress > 0` then `strict_sub`.

### 5b. Dead Kernel: X86Ssse3

- `simd_degree: 1` (single-block only, no multi-lane hash_many)
- Never selected by any dispatch table profile
- Not used in any hot path
- Adds code size and maintenance burden

Options:
- Remove entirely (preferred — pre-release mindset)
- Gate behind `#[cfg(test)]` as test-only reference

### 5c. Big-Endian Hot Path Branching

`mod.rs:2323-2340` — `single_chunk_output` uses `cfg!(target_endian = "little")`
in the hot path. Little-endian uses `ptr::copy_nonoverlapping` (zero-cost);
big-endian does `words16_from_le_bytes_64`. Not a bug, but:
- Verify big-endian path is tested (s390x, Power)
- Consider extracting to helper to keep hot path clean

### 5d. Suspicious Tuning Value: Zen5 `medium_limit_bytes = 262145`

This is 256K + 1, creating a medium range of exactly `[256K+1, ...]`.
Could be an off-by-one in the tuning derivation, or it could mean there's
effectively no medium class (small and large only). Worth verifying the
tuning fit quality for Zen5.

---

## 6. Dispatch Table Observations

### Apple M1M3: Portable for XS/S

```rust
boundaries: [64, 4095, 4096],
xs: KernelId::Portable,    // NEON setup overhead > savings
s: KernelId::Portable,     // Same — sub-chunk inputs stay scalar
m: KernelId::Aarch64Neon,
l: KernelId::Aarch64Neon,
```

Boundary at 4095 (not 4096) means the s->m transition is at exactly one
BLAKE3 chunk. Data-driven and correct.

### Intel SPR: AVX-512 at XS

```rust
xs: KernelId::X86Avx512,   // Wins at <=64B on SPR
s: KernelId::X86Avx2,      // Loses to AVX2 at 65-64B
m: KernelId::X86Avx512,
l: KernelId::X86Avx512,
```

AVX-512 winning at xs but losing to AVX2 at s is unusual. Reflects SPR's
efficient AVX-512 single-block compress vs. AVX2's lower-overhead multi-block
pipeline. Worth a second measurement pass to confirm.

### streaming_parallel Disabled on Most Profiles

Most profiles set `streaming_parallel.min_bytes = u64::MAX` — parallel
streaming completely disabled. Only Zen5c, Apple M4/M5, and Graviton2/ServerNeon
have finite thresholds.

Impact: streaming large inputs on Zen4/Zen5/SPR/ICL never goes parallel.
Conservative-correct but potentially leaving performance on the table.

---

## 7. Performance Observations

### Apple M1 (from mac_arm_m1.txt)

- **Small (0-64B)**: rscrypto and official within noise. rscrypto wins at 31B
  (375 vs 347 MiB/s) due to less oneshot overhead.
- **Medium (1K-64K)**: rscrypto matches or beats official single-threaded.
- **Large (1MiB)**: ~3.6 GiB/s both. Rayon doesn't help at 1MB on M1.
- **Parallel wins** show at 4MB+ where thread spawning amortizes.

### Known Blockers (from TASK.md)

- Graviton one-shot tiny inputs: **+56-65% slower** than official
- Keyed/derive small inputs (16-64B): ~20-30% remaining overhead
- XOF large-squeeze gap: reduced from +500% to +20-30% (recent progress)

---

## 8. Missing Kernel Coverage

### Architecture Gaps

| Arch | What's Missing | Competitive Impact |
|------|---------------|-------------------|
| **wasm32** | No `wasm_simd128` kernel | Official blake3 crate has this. CLAUDE.md promises wasm coverage. **Highest priority gap.** |
| **aarch64 SVE/SVE2** | No SVE kernel | Could give degree-8 on Graviton3+/Neoverse (matching AVX2 vs NEON's degree-4) |
| **s390x** | No VGFM/VSX kernel | CRC has s390x SIMD kernels; Blake3 is scalar-only |
| **POWER** | No VSX kernel | CRC has powerpc64 SIMD kernels; Blake3 is scalar-only |
| **RISC-V** | No RVV kernel | CRC has riscv64 Zbc/Zvbc kernels; Blake3 is scalar-only |

---

## 9. Workspace Context

### Crate Dependency Graph

```
rscrypto (facade)
  ├── checksum ──┬── traits
  │              ├── backend ── platform
  │              └── platform
  └── hashes ───┬── traits
                ├── backend ── platform
                └── platform

tune (dev-only)
  ├── checksum
  ├── hashes
  └── platform
```

### Key Traits

- `Checksum` — CRC types (update/finalize/checksum)
- `ChecksumCombine` — parallel combine (O(log n))
- `Digest` — Blake3, SHA-2, SHA-3 (update/finalize/digest)
- `Xof` — Blake3 XOF mode (squeeze)
- `FastHash` — XXH3, rapidhash (NOT crypto)

### Backend Crate: KernelTier (unused by Blake3)

```rust
pub enum KernelTier {
  Reference,   // bitwise, obviously correct
  Portable,    // table-based, no SIMD
  HwCrc,       // hardware CRC instructions
  Folding,     // carryless multiply / SIMD
  Wide,        // AVX-512 / SVE2 / wide SIMD
}
```

Blake3 defines its own flat `Blake3KernelId` without leveraging this.

---

## 10. Prioritized Action Items

### Tier 1: Close Before Moving On (Today)

#### 1.1 Fix Arithmetic Violations
- **File**: `crates/hashes/src/crypto/blake3/mod.rs`
- **Line 2141**: `wrapping_add` -> `strict_add` for `block_len`
- **Line 2190**: `saturating_sub` -> guarded `strict_sub` for `blocks_to_compress`
- **Scope**: ~5 lines changed
- **Why**: Direct CLAUDE.md violation. Correctness signal.

#### 1.2 Add Per-Kernel Isolation Benchmarks
- **New file**: `crates/hashes/benches/blake3_kernels.rs`
- **Pattern**: Match CRC64's `benches/kernels.rs`
- **Structure**:
  ```
  for each kernel in Blake3KernelId::ALL:
    for each size in [1, 64, 256, 1024, 4096, 16384, 65536, 1048576]:
      bench kernel.compress()
      bench kernel.hash_many_contiguous()
  ```
- **Why**: Foundation for trustworthy tuning. Can't identify kernel regressions
  without this.

#### 1.3 Add Forced Kernel Mode
- **Files**: `dispatch.rs`, `dispatch_tables.rs` (or new `config.rs`)
- **Env var**: `RSCRYPTO_BLAKE3_KERNEL=portable|x86_sse41|x86_avx2|x86_avx512|neon`
- **Pattern**: Match CRC64's `RSCRYPTO_CRC64_FORCE`
- **Why**: Essential for debugging dispatch vs kernel issues.

### Tier 2: High Value

#### 2.1 Remove or Gate X86Ssse3 Kernel
- **File**: `crates/hashes/src/crypto/blake3/kernels.rs`
- **Action**: Remove `X86Ssse3` variant entirely, or gate behind `#[cfg(test)]`
- **Why**: Dead code in dispatch. simd_degree=1. Pre-release mindset = no legacy.

#### 2.2 wasm32 SIMD Kernel
- **New files**: `crates/hashes/src/crypto/blake3/wasm32.rs` (or similar)
- **Target**: `wasm_simd128` intrinsics for 4-way compress
- **Why**: CLAUDE.md promises wasm coverage. Official blake3 crate has this.
  Competitive gap.

#### 2.3 Baseline Files for Unbaselined Profiles
- Add baselines for: Apple M4/M5, Graviton2, Zen5c, Intel GNR
- Even if run on CI, having baseline files prevents silent regressions.

#### 2.4 Formalize Kernel Tiers
- **File**: `crates/hashes/src/crypto/blake3/kernels.rs`
- Add tier doc comments to each `Blake3KernelId` variant:
  ```rust
  /// Portable: scalar compress, 1x SIMD degree. [Tier 1: Portable]
  Portable = 0,
  /// x86 SSE4.1: 4-way throughput. [Tier 3: SIMD Folding]
  X86Sse41 = 2,
  /// x86 AVX-512: 16-way throughput. [Tier 4: Wide SIMD]
  X86Avx512 = 4,
  ```
- **Why**: Makes kernel progression explicit. Matches CRC pattern.

### Tier 3: Polish & Completeness

#### 3.1 Validate Zen5 `medium_limit_bytes = 262145`
- Check if the +1 is intentional or off-by-one in tuning derivation.
- Re-run `just tune blake3` on Zen5 to verify.

#### 3.2 Enable Streaming Parallel on More Profiles
- Currently disabled (`min_bytes = u64::MAX`) on Zen4, Zen5, SPR, ICL.
- Evaluate whether streaming-API users on these platforms should get parallelism.
- Requires measuring streaming parallel overhead specifically.

#### 3.3 aarch64 SVE2 Kernel
- Target: Graviton3+/Neoverse V3 with 256-bit SVE2
- Benefit: degree-8 (matching AVX2) vs NEON's degree-4
- Lower priority: NEON is "good enough" for now.

#### 3.4 s390x / POWER / RISC-V SIMD Kernels
- Match CRC's architecture breadth (VGFM, VSX, Zbc/Zvbc)
- Lower priority: niche platforms, portable fallback works.

#### 3.5 Platform Detection Header in Baselines
- CRC baselines include a platform detection box:
  ```
  Platform: Caps(aarch64, [neon, aes, pmull, ...])
  Tune Kind: AppleM1M3
  Kernel selection by size: ...
  ```
- Blake3 baselines are raw Criterion output without this header.
- Add header generation to bench runner for reproducibility.

#### 3.6 Automated Regression Tracking
- `current_results.txt` / `previous_result.txt` are empty.
- Wire up `comp-check.py` to CI so losses are tracked per commit.

---

## 11. File Reference

### Blake3 Core Implementation

| File | Purpose | Lines |
|------|---------|-------|
| `crates/hashes/src/crypto/blake3/mod.rs` | Core impl: compress, ChunkState, OutputState, CV stack, tree | Large |
| `crates/hashes/src/crypto/blake3/kernels.rs` | Kernel registry, Blake3KernelId enum, function tables | ~300 |
| `crates/hashes/src/crypto/blake3/dispatch.rs` | Runtime dispatch resolution, HasherDispatch | ~300 |
| `crates/hashes/src/crypto/blake3/dispatch_tables.rs` | FamilyProfile per TuneKind, parallel tables | ~1022 |
| `crates/hashes/src/crypto/blake3/x86_64.rs` | x86 SSSE3/SSE4.1/AVX2/AVX-512 kernels | Large |
| `crates/hashes/src/crypto/blake3/x86_64/asm.rs` | x86 ASM backend (AVX2, AVX-512) | |
| `crates/hashes/src/crypto/blake3/x86_64/avx2.rs` | AVX2 8-way intrinsics | |
| `crates/hashes/src/crypto/blake3/x86_64/avx512.rs` | AVX-512 16-way intrinsics | |
| `crates/hashes/src/crypto/blake3/aarch64.rs` | NEON 4-way kernel | |
| `crates/hashes/src/crypto/blake3/aarch64/asm.rs` | aarch64 ASM backend | |
| `crates/hashes/src/crypto/blake3/kernel_test.rs` | Cross-kernel verification | |

### Benchmarks & Baselines

| File | Purpose |
|------|---------|
| `crates/hashes/benches/blake3.rs` | Main Blake3 Criterion bench (rscrypto vs official) |
| `crates/hashes/benches/common/` | Shared bench utilities (sized_inputs, pseudo_random) |
| `crates/hashes/src/crypto/blake3/bench_baseline/*.txt` | 7 platform baselines |
| `scripts/bench/comp-check.py` | Baseline comparison / gating tool |
| `scripts/gen_blake3_x86_asm_ports.py` | Blake3 x86 ASM codegen |

### Tuning Engine

| File | Purpose |
|------|---------|
| `crates/tune/src/bin/rscrypto-tune.rs` | CLI: measure, derive, apply |
| `crates/tune/src/engine.rs` | TuneEngine: measure -> derive pipeline |
| `crates/tune/src/engine/blake3_adapter.rs` | Blake3 parallel policy derivation |
| `crates/tune/src/apply.rs` | Generate Rust code, patch dispatch_tables.rs |
| `crates/tune/src/hash.rs` | Hash corpus definitions (17 Blake3 surfaces) |
| `crates/tune/src/runner.rs` | BenchRunner: measurement loop, sampling |
| `crates/tune/src/analysis.rs` | Crossover detection, statistical analysis |
| `scripts/tune/apply.sh` | Multi-artifact apply with overlap detection |
| `scripts/ci/run-tune.sh` | CI pipeline orchestration |

### Platform & Backend

| File | Purpose |
|------|---------|
| `crates/platform/src/tune.rs` | TuneKind enum (28 variants), Tune struct (14 fields) |
| `crates/platform/src/detect.rs` | CPU detection (CPUID, HWCAP, etc.) |
| `crates/platform/src/caps.rs` | CPU capability bitset (256-bit) |
| `crates/platform/src/dispatch.rs` | dispatch(), dispatch_static(), dispatch_auto() |
| `crates/backend/src/tier.rs` | KernelTier enum (Reference/Portable/HwCrc/Folding/Wide) |
| `crates/backend/src/cache.rs` | OnceCache, PolicyCache |

### CRC64 Reference (Gold Standard)

| File | Purpose |
|------|---------|
| `crates/checksum/src/dispatch.rs` | KernelTable, select_table() — 4-tier resolution |
| `crates/checksum/benches/kernels.rs` | Per-kernel, per-size, per-alignment benchmarks |
| `crates/checksum/bench_baseline/kernels/*.txt` | Kernel-level baseline files |
| `crates/checksum/src/crc64/mod.rs` | 60+ cross-check tests (`mod cross_check`) |
| `scripts/gen/kernel_tables.py` | Baseline analysis -> dispatch table generation |

---

## 12. Unsafe Code Inventory (Blake3)

All unsafe in Blake3 falls into these categories:

1. **Unaligned loads**: `ptr::read_unaligned()` for `[u8] -> [u32]` conversion.
   Justified: `[u8]` is 1-byte aligned, `read_unaligned` handles any alignment.

2. **SIMD intrinsics**: `_mm_loadu_si128`, `_mm256_*`, `_mm512_*`, NEON intrinsics.
   Gated by `#[target_feature(enable = "...")]`. Dispatch validates caps before use.

3. **`hash_many_contiguous` fn pointers**: Caller must guarantee input/output
   buffer sizes. Verified in wrapper functions.

4. **`ptr::copy_nonoverlapping`**: LE byte conversion in OutputState. Fixed-size
   arrays, known alignment.

5. **`slice::from_raw_parts_mut`**: Rayon task partitions. `SendPtr` wrapper
   ensures disjoint ranges.

6. **ASM backends**: `extern "C"` functions. Platform-specific calling conventions.
   Correctness validated by kernel_test suite.

All unsafe blocks have `// SAFETY:` comments. Miri fallbacks provided via
`#[cfg(miri)]` where needed.
