# Benchmark Overview

Sources:

- Linux: AEAD bench CI run [#25460469827](https://github.com/loadingalias/rscrypto/actions/runs/25460469827), created 2026-05-06 20:52:28 UTC.
- macOS: local AEAD run `benchmark_results/2026-05-06/macos/aarch64/results.txt`, created 2026-05-06 16:54:36 local time.
- Raw files: `benchmark_results/2026-05-06/linux/*/results.txt` and `benchmark_results/2026-05-06/macos/aarch64/results.txt`.

Commit: `31339f2cc954ab3e3a066b0bb7c48e5b28a60659`.

Scope: AEAD only. Linux CI is the platform-matrix source of truth; local macOS is a proof run for Apple Silicon.
Speedup is `external_crate_time / rscrypto_time`; above `1.00x` means `rscrypto` is faster.
Wins are `>1.05x`, ties are `0.95x..1.05x`, losses are `<0.95x`.
All-pair comparisons count every matched external implementation. Fastest-external comparisons keep only the fastest external implementation for each platform, operation, mode, and input size.

## Headline

Linux AEAD all-pair headline: `rscrypto` wins **1864 / 2970** matched Linux CI external comparisons.
Against only the fastest external implementation per case, it wins **650 / 1386** comparisons.

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Linux CI, all external pairs | 2970 | 1864/299/807 | 63% | 1.80x | 1.30x |
| Linux CI, fastest external per case | 1386 | 650/176/560 | 47% | 1.18x | 1.03x |
| macOS local, all external pairs | 330 | 256/14/60 | 78% | 2.85x | 1.52x |
| macOS local, fastest external per case | 154 | 107/7/40 | 69% | 1.33x | 1.24x |
| Linux CI + macOS local, all external pairs | 3300 | 2120/313/867 | 64% | 1.88x | 1.31x |
| Linux CI + macOS local, fastest external per case | 1540 | 757/183/600 | 49% | 1.19x | 1.04x |

What matters:

- Clean-sweep status: not done. Required Linux large rows are **90/1/269** against the fastest external implementation, with **0.96x** geomean.
- AES-GCM is the primary blocker: Linux large AES-GCM rows are **20/0/160** at **0.75x** geomean; macOS large AES-GCM rows are **0/0/20** at **0.88x**.
- GCM-SIV is strong on Apple Silicon, Graviton, and s390x, but it is not a Linux clean sweep: Linux large GCM-SIV rows are **70/1/109** at **1.23x** geomean.
- Local macOS GCM-SIV remains clean: large GCM-SIV rows are **20/0/0** at **2.79x** geomean.
- Against all external pairs, AEAD looks healthy because RustCrypto is much slower on many rows. Against the fastest competitor, AWS-LC and `ring` still apply real pressure.

## Linux AEAD Detail

| Primitive | All pairs | W/T/L | Geomean | Median | Fastest-only pairs | W/T/L | Geomean | Median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AES-128-GCM | 594 | 371/23/200 | 2.01x | 1.38x | 198 | 76/11/111 | 1.04x | 0.87x |
| AES-256-GCM | 594 | 374/38/182 | 2.13x | 1.39x | 198 | 75/11/112 | 1.08x | 0.88x |
| AES-128-GCM-SIV | 396 | 278/11/107 | 2.42x | 1.67x | 198 | 93/2/103 | 1.30x | 0.94x |
| AES-256-GCM-SIV | 396 | 289/4/103 | 2.70x | 1.73x | 198 | 91/4/103 | 1.38x | 0.93x |
| ChaCha20-Poly1305 | 594 | 300/103/191 | 1.02x | 1.05x | 198 | 63/28/107 | 0.85x | 0.94x |
| XChaCha20-Poly1305 | 198 | 144/52/2 | 1.28x | 1.17x | 198 | 144/52/2 | 1.28x | 1.17x |
| AEGIS-256 | 198 | 108/68/22 | 1.45x | 1.07x | 198 | 108/68/22 | 1.45x | 1.07x |

## macOS AEAD Detail

| Primitive | All pairs | W/T/L | Geomean | Median | Fastest-only pairs | W/T/L | Geomean | Median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AES-128-GCM | 66 | 52/4/10 | 3.68x | 1.76x | 22 | 12/0/10 | 1.20x | 1.09x |
| AES-256-GCM | 66 | 55/1/10 | 3.70x | 1.62x | 22 | 11/1/10 | 1.13x | 1.08x |
| AES-128-GCM-SIV | 44 | 44/0/0 | 7.99x | 11.36x | 22 | 22/0/0 | 2.67x | 2.70x |
| AES-256-GCM-SIV | 44 | 44/0/0 | 8.77x | 12.24x | 22 | 22/0/0 | 2.62x | 2.73x |
| ChaCha20-Poly1305 | 66 | 23/3/40 | 0.72x | 0.82x | 22 | 2/0/20 | 0.54x | 0.47x |
| XChaCha20-Poly1305 | 22 | 18/4/0 | 1.18x | 1.23x | 22 | 18/4/0 | 1.18x | 1.23x |
| AEGIS-256 | 22 | 20/2/0 | 1.20x | 1.11x | 22 | 20/2/0 | 1.20x | 1.11x |

## Clean-Sweep Gate: Full Matrix

Fastest-external comparisons only. Linux rows have 99 cases each; macOS rows have 11 cases each.

| Required primitive | Linux W/T/L | Linux geomean | Linux median | macOS W/T/L | macOS geomean | macOS median |
|---|---:|---:|---:|---:|---:|---:|
| AES-128-GCM encrypt | 34/5/60 | 1.00x | 0.85x | 6/0/5 | 1.12x | 1.08x |
| AES-128-GCM decrypt | 42/6/51 | 1.08x | 0.92x | 6/0/5 | 1.28x | 1.10x |
| AES-256-GCM encrypt | 33/4/62 | 1.03x | 0.86x | 5/1/5 | 1.07x | 1.04x |
| AES-256-GCM decrypt | 42/7/50 | 1.14x | 0.93x | 6/0/5 | 1.19x | 1.12x |
| AES-128-GCM-SIV encrypt | 49/0/50 | 1.33x | 0.95x | 11/0/0 | 2.56x | 2.69x |
| AES-128-GCM-SIV decrypt | 44/2/53 | 1.27x | 0.93x | 11/0/0 | 2.78x | 2.72x |
| AES-256-GCM-SIV encrypt | 49/2/48 | 1.44x | 1.05x | 11/0/0 | 2.55x | 2.63x |
| AES-256-GCM-SIV decrypt | 42/2/55 | 1.33x | 0.93x | 11/0/0 | 2.69x | 2.76x |

## Clean-Sweep Gate: Large Rows

Fastest-external comparisons only for input sizes `>=4096`. Linux rows have 45 cases each; macOS rows have 5 cases each.

| Required primitive | Linux W/T/L | Linux geomean | Linux median | macOS W/T/L | macOS geomean | macOS median |
|---|---:|---:|---:|---:|---:|---:|
| AES-128-GCM encrypt | 5/0/40 | 0.73x | 0.54x | 0/0/5 | 0.86x | 0.86x |
| AES-128-GCM decrypt | 5/0/40 | 0.72x | 0.52x | 0/0/5 | 0.86x | 0.85x |
| AES-256-GCM encrypt | 5/0/40 | 0.77x | 0.50x | 0/0/5 | 0.88x | 0.87x |
| AES-256-GCM decrypt | 5/0/40 | 0.78x | 0.58x | 0/0/5 | 0.91x | 0.90x |
| AES-128-GCM-SIV encrypt | 20/0/25 | 1.31x | 0.90x | 5/0/0 | 2.71x | 2.70x |
| AES-128-GCM-SIV decrypt | 15/0/30 | 1.07x | 0.85x | 5/0/0 | 2.71x | 2.72x |
| AES-256-GCM-SIV encrypt | 20/1/24 | 1.44x | 0.89x | 5/0/0 | 2.89x | 2.89x |
| AES-256-GCM-SIV decrypt | 15/0/30 | 1.15x | 0.89x | 5/0/0 | 2.87x | 2.86x |

## Required Large Rows By Platform

Fastest-external comparisons only for AES-GCM and GCM-SIV, 128-bit and 256-bit keys, encrypt and decrypt, input sizes `>=4096`.

| Platform | AES-GCM W/T/L | AES-GCM geomean | AES-GCM median | GCM-SIV W/T/L | GCM-SIV geomean | GCM-SIV median |
|---|---:|---:|---:|---:|---:|---:|
| AMD Zen 4 | 0/0/20 | 0.51x | 0.50x | 0/0/20 | 0.70x | 0.70x |
| AMD Zen 5 | 0/0/20 | 0.31x | 0.29x | 0/0/20 | 0.70x | 0.70x |
| Intel Ice Lake | 0/0/20 | 0.47x | 0.47x | 0/0/20 | 0.51x | 0.51x |
| Intel Sapphire Rapids | 0/0/20 | 0.48x | 0.48x | 0/0/20 | 0.58x | 0.58x |
| AWS Graviton3 | 0/0/20 | 0.68x | 0.70x | 20/0/0 | 2.53x | 2.53x |
| AWS Graviton4 | 0/0/20 | 0.72x | 0.75x | 20/0/0 | 2.48x | 2.48x |
| IBM POWER10 | 0/0/20 | 0.38x | 0.37x | 10/0/10 | 1.46x | 1.61x |
| IBM Z / s390x | 20/0/0 | 12.45x | 12.51x | 20/0/0 | 5.46x | 5.33x |
| RISE RISC-V | 0/0/20 | 0.86x | 0.86x | 0/1/19 | 0.91x | 0.91x |
| macOS AArch64 | 0/0/20 | 0.88x | 0.87x | 20/0/0 | 2.79x | 2.76x |

## External Competitor Pressure

All-pair comparisons. This answers which external implementations are applying pressure in the current AEAD run.

### Linux

| External | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| `rustcrypto` | 1188 | 987/117/84 | 83% | 2.79x | 1.62x |
| `aegis-crate` | 198 | 108/68/22 | 55% | 1.45x | 1.07x |
| `ring` | 594 | 283/73/238 | 48% | 1.35x | 1.03x |
| `aws-lc-rs` | 990 | 486/41/463 | 49% | 1.32x | 1.03x |

### macOS

| External | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| `rustcrypto` | 132 | 125/7/0 | 95% | 9.84x | 21.85x |
| `aws-lc-rs` | 110 | 69/1/40 | 63% | 1.46x | 1.94x |
| `aegis-crate` | 22 | 20/2/0 | 91% | 1.20x | 1.11x |
| `ring` | 66 | 42/4/20 | 64% | 0.97x | 1.09x |

## Raw Results

| Runner | Result |
|---|---|
| AMD Zen 4 | `benchmark_results/2026-05-06/linux/amd-zen4/results.txt` |
| AMD Zen 5 | `benchmark_results/2026-05-06/linux/amd-zen5/results.txt` |
| Intel Ice Lake | `benchmark_results/2026-05-06/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `benchmark_results/2026-05-06/linux/intel-spr/results.txt` |
| AWS Graviton3 | `benchmark_results/2026-05-06/linux/graviton3/results.txt` |
| AWS Graviton4 | `benchmark_results/2026-05-06/linux/graviton4/results.txt` |
| IBM POWER10 | `benchmark_results/2026-05-06/linux/ibm-power10/results.txt` |
| IBM Z / s390x | `benchmark_results/2026-05-06/linux/ibm-s390x/results.txt` |
| RISE RISC-V | `benchmark_results/2026-05-06/linux/rise-riscv/results.txt` |
| macOS AArch64 | `benchmark_results/2026-05-06/macos/aarch64/results.txt` |

## Methodology Notes

- Parsed 4840 Criterion median timings from 10 structured AEAD result files.
- Every platform has 484 parsed AEAD timings; no platform result file is missing from this run.
- CI result headers were normalized to the workflow creation time and runner platform ID after artifact extraction.
- A comparison is matched when a Criterion ID has a `rscrypto` path component and an external ID with the same platform, primitive, operation, and input size, differing only in the implementation component.
- The local macOS run is included for Apple Silicon signal, but it is not the final platform-matrix gate.
