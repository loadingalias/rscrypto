# Blake3 Comp Gaps — 2026-03-15

Source: CI run [23122515388](https://github.com/loadingalias/rscrypto/actions/runs/23122515388)
on AMD Zen4 (AVX-512), Intel Sapphire Rapids (AVX-512), AWS Graviton4 (NEON).
Criterion 30 samples, `RUSTFLAGS="-C target-cpu=native"`, `--bench hashes_comp -- blake3/`.

Only losses listed (cases where official beats rscrypto). Mid-point estimate used.

## Critical: XOF (3-4x slower at small inputs, every platform)

Losses on ALL platforms, ALL sizes. Worst case: 265-320% slower.

| Case | Zen4 gap | SPR gap | Graviton4 gap |
|------|----------|---------|---------------|
| 1B-in / 32B-out | **-264%** | **-320%** | **-320%** |
| 64B-in / 32B-out | **-266%** | **-302%** | **-319%** |
| 1KiB-in / 32B-out | **-22%** | **-25%** | **-27%** |
| 64KiB-in / 32B-out | **-16%** | **-16%** | +7% win |
| 1B-in / 1KiB-out | **-142%** | **-179%** | **-16%** |
| 64B-in / 1KiB-out | **-105%** | **-165%** | **-13%** |
| 1KiB-in / 1KiB-out | **-26%** | **-27%** | **-14%** |
| 64KiB-in / 1KiB-out | **-16%** | **-16%** | +7% win |

Root cause: XOF squeeze/output-extend path is not using SIMD. The official
crate parallelizes output block generation; rscrypto appears to do it serially.

## Critical: Streaming (10-17% gap on x86-64, mixed on ARM)

Losses on Zen4 and SPR at ALL chunk sizes. Graviton4 loses small, wins large.

| Chunk size | Zen4 gap | SPR gap | Graviton4 gap |
|------------|----------|---------|---------------|
| 64B | **-13.5%** | **-13.5%** | -2.7% |
| 128B | **-14.5%** | **-14.9%** | **-6.0%** |
| 256B | **-14.4%** | **-15.6%** | **-6.0%** |
| 512B | **-14.6%** | **-15.6%** | **-4.4%** |
| 1KiB | -1.3% | ~0% | -2.2% |
| 4KiB | **-8.3%** | **-6.2%** | +6.4% win |
| 16KiB | **-16.8%** | **-14.1%** | +5.8% win |
| 64KiB | **-13.3%** | **-12.5%** | +8.0% win |

Pattern: x86-64 has a systemic streaming regression. Graviton4 wins at
large chunks but loses at small. The 1KiB chunk boundary is a transition point.

## Significant: 1 MiB hash/keyed-hash (~10% gap, x86-64 only)

| Mode | Zen4 gap | SPR gap | Graviton4 gap |
|------|----------|---------|---------------|
| hash 1MiB | **-10.8%** | **-10.3%** | +8.6% win |
| keyed 1MiB | **-10.3%** | **-11.1%** | +8.5% win |
| derive 1MiB | **-10.2%** | **-10.6%** | +8.2% win |

x86-64 specific. Graviton4 wins by ~8.5% at the same size. Likely a large-buffer
hot-loop issue: prefetch strategy, cache line handling, or AVX-512 utilization.

## Minor: Small-input hash/keyed overhead

Size 65 (first byte of second chunk) is a consistent loss across all platforms:

| Mode | Zen4 gap | SPR gap | Graviton4 gap |
|------|----------|---------|---------------|
| hash 65B | -4.9% | **-7.0%** | -1.6% |
| keyed 65B | -4.2% | **-7.6%** | -3.7% |

Sizes 0-1B also lose on Graviton4 (~5%) but win on Zen4/SPR (~17%).

## Priority ranking

1. **XOF**: 3-4x slower at small inputs on every platform. Architectural gap
   in output-extend logic. Biggest absolute improvement available.
2. **Streaming (x86-64)**: 13-17% slower across the board on Zen4/SPR. This
   is the update hot path — likely something in buffer management, state
   machine overhead, or the compression function dispatch.
3. **1 MiB bulk (x86-64)**: ~10% gap for hash/keyed/derive at the largest
   single-thread workload. Graviton4 does NOT have this issue.
4. **Size 65B (chunk boundary)**: Consistent 2-8% penalty at the first byte
   of the second chunk. Likely chunk finalization + new chunk setup overhead.

## Strengths (where rscrypto wins)

**derive-key: blowout win everywhere**
- Empty to 128B: 30-60% faster (official double-hashes for key derivation)
- Wins every size except 1MiB on x86-64

**Small/medium hash and keyed-hash on x86-64**
- Zen4: 32B +19%, 64B +22%, 128B +15% (huge wins)
- SPR: 128B +5%, 1KiB +3%

**Large sizes on Graviton4**
- 4KiB-1MiB: +7-11% for hash/keyed/derive
- Streaming 4KiB-64KiB: +6-8%
