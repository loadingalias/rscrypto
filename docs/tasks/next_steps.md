# Next Steps: Path to 80% Win Rate

> **Current: ~69%** (~1064W / ~379T / ~97L — ~1540 comparisons, estimated)
> **Target: 80%** (needs ~168 more wins from losses + ties)
> **Source:** CI run [#23715419566](https://github.com/loadingalias/rscrypto/actions/runs/23715419566) (2026-03-29, 5 platforms: Zen4/SPR/ICL/s390x/POWER10) + Grav3/Grav4 baseline from [#23691455747](https://github.com/loadingalias/rscrypto/actions/runs/23691455747)
> **Note:** Zen5 unavailable; Grav3/Grav4 estimated from pre-lane-complementing baseline (aarch64 unchanged)

---

## Budget

| Category | W | T | L | Total | Win% | Δ |
|----------|---|---|---|-------|------|---|
| Checksums | 430 | 48 | 26 | 504 | 85% | — |
| SHA-2 | 232 | 66 | 17 | 315 | 74% | — |
| SHA-3 | ~123 | ~129 | 0 | 252 | ~49% | +57W (was 26%) |
| SHAKE | ~131 | ~31 | 0 | 126 | ~90%¹ | +41W (was 71%) |
| Blake3 | 31 | 26 | 6 | 63 | 49% | — |
| XXH3 | 68 | 51 | 7 | 126 | 54% | — |
| RapidHash | 5 | 49 | 9 | 63 | 8% | — |
| Auth | 44 | 15 | 32 | 91 | 48% | — |
| **Total** | **~1064** | **~379** | **~97** | **~1540** | **~69%** | **+98W** |

¹ SHAKE 90% is on 5 platforms; Grav adds ~25W/~11T bringing it back to ~85%.

Gap to 80%: need ~1232W. Currently ~1064W → need ~168 more wins.

---

## What Landed: Lane-Complementing Chi (Phase A)

**XKCP "Bebigokimisa" lane-complementing** on x86-64/s390x/POWER:
complement lanes {1,2,8,12,17,20}, replace uniform NOT-AND chi with
mixed AND/OR/NOT formulas. Reduces NOT ops from 25→5 per round.

**Measured impact (5 platforms, CI run #23715419566):**

| Platform | SHA-3-256 0B | Speedup | SHA-3 W/T/L |
|----------|-------------|---------|-------------|
| Zen4 | 310 ns vs 372 ns | **1.20x** | ~32W/4T/0L |
| SPR | 318 ns vs 383 ns | **1.20x** | ~30W/6T/0L |
| ICL | 397 ns vs 424 ns | **1.07x** | ~20W/16T/0L |
| s390x | 537 ns vs 625 ns | **1.16x** | ~32W/4T/0L |
| POWER10 | 310 ns vs 313 ns | 1.01x | ~4W/32T/0L |

**aarch64 lane-complementing was tried and reverted.** ARM's BIC instruction
already fuses NOT+AND into one cycle, so the complement/uncomment overhead
made the portable path ~1.8× slower on Graviton3/Graviton4.

**aarch64 SHA3 CE full-NEON kernel was built and tested.** Fast on Apple
Silicon (171 ns raw, 28% faster than keccak crate) but ~1.8× slower than
portable on Graviton (Neoverse V1/V2 SHA3 CE microarchitecture penalizes
half-utilized 128-bit registers). Kept for 2-state interleaved path and
diagnostic benchmarks; not wired into single-state production dispatch.

---

## Remaining Conversion Opportunities

### A. SHA-3 aarch64 ties (~72T on Grav3/Grav4)

The portable path on aarch64 is at exact parity with the `sha3` crate.
Lane-complementing hurts (BIC), the NEON kernel hurts on Graviton.

**Potential approaches:**
- EOR3 for theta parity in the portable path (3-input XOR via inline asm,
  chain keeps values in NEON between EOR3→RAX1 without full domain crossing)
- Eliminate zeroization overhead for non-secret hash contexts (opt-in)
- Investigate why the full-NEON kernel is slow on Neoverse V1/V2 specifically

### B. POWER10 SHA-3 ties (~32T)

Lane-complementing only gives ~1% on POWER10. The keccak-f gap is minimal.
Zeroization overhead removal would help most at small sizes.

### C. Ed25519 — 28L (single largest loss category)

**Measured (Zen4):** rscrypto sign 100 µs vs dalek 16.8 µs = **5.9× gap**.

Optimization roadmap (unchanged from previous analysis):

| Step | What | Expected result |
|------|------|-----------------|
| 7.1 — 8-bit basepoint table | 256-point precomp, 32 rounds vs 64 | ~25 µs (4× improvement) |
| 7.2 — AVX2 field mul/square | Parallel 5×51 via `vpmuludq` | ~15 µs |
| 7.3 — Signed NAF recoding | Reduce point additions via signed digits | ~12 µs (beats dalek) |
| 7.4 — AVX-512 IFMA (Intel) | 52-bit native mul-add | ~10 µs |

Step 7.1 alone converts 28L → competitive. Steps 7.1-7.3 would beat dalek.

### D. Other categories (unchanged)

- **Checksums 26L:** mostly s390x/POWER small-size overhead
- **SHA-2 17L:** SPR-specific AVX-512 gap
- **Blake3 6L:** s390x 1KiB, ICL/SPR 256B
- **XXH3 7L:** s390x/POWER small sizes
- **RapidHash 9L:** s390x/POWER parity gap

---

## Remaining Work (97 losses)

### 1. Checksums — 26L
(unchanged from previous)

### 2. SHA-2 — 17L
(unchanged — SPR AVX-512 gap)

### 3. SHAKE — 0L ✓
(was 1L, now 0L)

### 4. Blake3 — 6L
(unchanged)

### 5. XXH3 — 7L
(unchanged)

### 6. RapidHash — 9L
(unchanged)

### 7. Auth — 32L
(unchanged — Ed25519 dominates)

---

## Platform Summary (estimated 7-platform)

| Platform | W | T | L | Win% | Key Change |
|----------|---|---|---|------|------------|
| Zen4 | ~187 | ~27 | 6 | ~85% | SHA-3/SHAKE: ties→wins (+32W) |
| SPR | ~156 | ~41 | 23 | ~71% | SHA-3/SHAKE: ties→wins (+30W) |
| ICL | ~131 | ~80 | 9 | ~60% | SHA-3: partial ties→wins (+20W) |
| Grav3 | 131 | 80 | 9 | 60% | unchanged |
| Grav4 | 129 | 80 | 11 | 59% | unchanged |
| s390x | ~216 | ~2 | 22 | ~90% | SHA-3/SHAKE all wins (+32W) |
| POWER10 | ~134 | ~68 | 18 | ~61% | marginal SHA-3 gains |

---

## Completed Phases

| Phase | What | Impact |
|-------|------|--------|
| 1.1 — CRC inline fast-path | `Checksum::checksum()` override bypasses hasher construction | +77W |
| 1.2 — CRC32 Grav 64B dispatch | Route <128B to hardware CRC, bypass PMULL | 0-64B: 16W/0T/0L |
| 1.3 — CRC32 Grav PMULL 3-tier | <128B hwcrc, 128-1024B v12e_v1, >1024B EOR3 fusion | CRC32C 13L→2L on Grav |
| 2.1-2.3 — SHA-512 AVX2+AVX-512VL | Stitched dual-block, BMI2 RORX, deferred-Sigma0 | Zen5 0.70x→1.02x, ICL 16L→0L |
| 2.4 — SHA-512 decoupled+rotation | Eliminated `permute2x128` bottleneck, rotation schedule | 27L→0L on Zen4+Zen5 |
| 3.1 — Keccak-f portable rewrite | Array-based x86-64, named-var aarch64 | +28-30% raw permutation speed |
| 3.2 — Keccak theta rewrite | All 5 d-values upfront for OOO overlap | SHA-3 131L→35L |
| 4.1a-c — XXH3 SIMD + cold dispatch | All-platform SIMD kernels, `#[cold]` dispatch, typed mix | 81L→37L |
| 4.1d — XXH3 0B fast-return | Compile-time const hash for seed=0 | 7L eliminated at 0B |
| 4.1e — XXH3 NEON kernel | vuzpq deinterleave, broken dep chain, stripe prefetch | Grav 256B-1MiB all ties |
| 4.1f — XXH3 flat dispatch + CT SIMD | Flat size branches, compile-time kernel dispatch | 37L→23L (−14L) |
| 4.1g — XXH3 register pressure fix | Remove empty early-return, align mix32_b with xxhash-rust | xxh3-128 32B 0.91x→1.01x |
| 4.2 — RapidHash inner core | Core parity at 256B+ | 8W/52T/12L |
| 5.1 — Keccak fused absorb-permute | `keccakf_absorb_portable`: load state^block in one step | SHA-3 26L→15L, SHAKE 7L→3L |
| 6.1 — HMAC-SHA256 oneshot | Direct `[u32; 8]` state, single dispatch compress | 26L→0L, Grav4 3.7-4.0x |
| **A — Lane-complementing chi** | **XKCP "Bebigokimisa" on x86/s390x/POWER (not aarch64)** | **SHA-3 66W→~123W, SHAKE 90W→~131W (+98W)** |
