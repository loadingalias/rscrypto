# Blake3 Comp Gaps — 2026-03-15 (local, Apple Silicon, noisy)

Source: `just bench-blake3-compare` on laptop. CI numbers pending.
Criterion 30 samples, default warmup/measurement. Not native RUSTFLAGS.

Only losses listed (cases where official beats rscrypto).

## Critical: XOF (4-5x slower at small inputs)

| Case | rscrypto | official | gap |
|------|----------|----------|-----|
| 1B-in / 32B-out | 87 MiB/s | 401 MiB/s | **-78%** |
| 64B-in / 32B-out | 253 MiB/s | 1.12 GiB/s | **-78%** |
| 1KiB-in / 32B-out | 674 MiB/s | 857 MiB/s | **-21%** |
| 64KiB-in / 32B-out | 1.65 GiB/s | 1.73 GiB/s | -4.7% |
| 1B-in / 1KiB-out | 698 MiB/s | 879 MiB/s | **-21%** |
| 64B-in / 1KiB-out | 740 MiB/s | 931 MiB/s | **-20%** |
| 1KiB-in / 1KiB-out | 772 MiB/s | 882 MiB/s | **-12%** |
| 64KiB-in / 1KiB-out | 1.59 GiB/s | 1.65 GiB/s | -3.8% |

Root cause likely: XOF squeeze path not using SIMD, or excessive per-call overhead.

## Streaming (consistent 2-4% gap)

| Chunk size | rscrypto | official | gap |
|------------|----------|----------|-----|
| 64B | 742 MiB/s | 757 MiB/s | -2.0% |
| 128B | 744 MiB/s | 769 MiB/s | -3.2% |
| 256B | 752 MiB/s | 784 MiB/s | **-4.1%** |
| 512B | 762 MiB/s | 789 MiB/s | **-3.4%** |
| 1KiB | 763 MiB/s | 778 MiB/s | -2.0% |
| 4KiB | 1.55 GiB/s | 1.61 GiB/s | **-3.8%** |
| 16KiB | 1.59 GiB/s | 1.70 GiB/s | **-6.4%** |
| 64KiB | 1.65 GiB/s | 1.72 GiB/s | **-4.1%** |

Pattern: gap widens at medium-large chunk sizes. Suggests overhead in
update/finalize path or suboptimal buffer management.

## Oneshot hash (small gaps at medium-large sizes)

| Size | rscrypto | official | gap |
|------|----------|----------|-----|
| 0 (empty) | 13.45 Melem/s | 13.90 Melem/s | -3.3% |
| 64B | 798 MiB/s | 831 MiB/s | **-4.0%** |
| 1KiB | 806 MiB/s | 822 MiB/s | -1.9% |
| 4KiB | 1.60 GiB/s | 1.63 GiB/s | -1.6% |
| 64KiB | 1.72 GiB/s | 1.73 GiB/s | -0.4% |
| 1MiB | 1.64 GiB/s | 1.70 GiB/s | **-3.4%** |

## Keyed hash (mirrors oneshot pattern)

| Size | rscrypto | official | gap |
|------|----------|----------|-----|
| 0 (empty) | 13.40 Melem/s | 13.91 Melem/s | -3.6% |
| 1B | 12.24 MiB/s | 12.50 MiB/s | -2.1% |
| 64B | 795 MiB/s | 838 MiB/s | **-5.1%** |
| 65B | 401 MiB/s | 413 MiB/s | -2.9% |
| 128B | 799 MiB/s | 815 MiB/s | -2.0% |
| 1KiB | 801 MiB/s | 819 MiB/s | -2.2% |
| 4KiB | 1.58 GiB/s | 1.66 GiB/s | **-4.8%** |
| 64KiB | 1.72 GiB/s | 1.73 GiB/s | -0.6% |
| 1MiB | 1.66 GiB/s | 1.73 GiB/s | **-3.8%** |

## Derive-key (only 2 losses, large sizes)

| Size | rscrypto | official | gap |
|------|----------|----------|-----|
| 4KiB | 1.60 GiB/s | 1.62 GiB/s | -0.8% |
| 1MiB | 1.62 GiB/s | 1.69 GiB/s | **-4.2%** |

Note: rscrypto **dominates** derive-key at small sizes (2x faster).

## Priority ranking

1. **XOF**: 4-5x slower at small inputs. Likely a missing SIMD path or
   architectural issue in the squeeze/output-extend logic. Biggest win.
2. **Streaming**: Consistent 3-6% gap. Buffer management or update hot path.
3. **Oneshot/keyed 1MiB**: 3-5% gap at bulk sizes. Compression loop or
   finalization overhead.
4. **Oneshot/keyed 64B**: 4-5% gap at exactly 1 chunk. Possibly setup cost.

## Wins (not losing)

- Oneshot hash 1-63B: rscrypto faster by 2-3%
- Oneshot hash 128B: rscrypto faster
- Keyed hash 31-63B: rscrypto faster
- Derive-key 0B-1KiB: rscrypto **2x faster**
- Derive-key 16KiB-64KiB: rscrypto faster
