# Benchmark Overview

Sources:

- Linux: Bench CI run [#25400626998](https://github.com/loadingalias/rscrypto/actions/runs/25400626998), created 2026-05-05 20:30:36 UTC.
- Raw files: `benchmark_results/2026-05-05/linux/*/results.txt`.

Commit: `67b10410a17c23ef2cb33aa2454b43622d861079`.

Scope: nine Linux CI runners. Local macOS files are excluded.
Speedup is `external_crate_time / rscrypto_time`; above `1.00x` means `rscrypto` is faster.
Wins are `>1.05x`, ties are `0.95x..1.05x`, losses are `<0.95x`.
All-pair comparisons count every matched external implementation. Fastest-external comparisons keep only the fastest external implementation for each platform, operation, mode, and input size.

## Headline

Linux all-pair headline: `rscrypto` wins **6108 / 10395** matched Linux CI external comparisons.
Against only the fastest external implementation per case, it wins **3139 / 5895** comparisons.

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Linux CI, all external pairs | 10395 | 6108/2297/1990 | 59% | 1.46x | 1.12x |
| Linux CI, fastest external per case | 5895 | 3139/1530/1226 | 53% | 1.43x | 1.07x |

What matters:

- This run fixes the previous AES-128 coverage gap. AEAD now contributes 2574 all-pair comparisons and 1386 fastest-external comparisons.
- The overall shape is solid but not universal: 1.46x all-pair geomean and 1.43x fastest-external geomean across Linux CI.
- SHA-3 and SHAKE remain clean wins: 2.18x and 2.61x geomean vs RustCrypto `sha3`, with no losses in this run.
- Checksums remain the strongest public story: 4.29x all-pair geomean and 5.00x against the fastest external implementation per case.
- AEAD is healthier with AES-128 included: 1.46x all-pair geomean and 1.36x fastest-external geomean. AES-GCM still loses to the fastest external backend on most cases; AES-GCM-SIV remains a decisive win.
- BLAKE3 is still strongest on large inputs: 2.23x geomean for `>=64 KiB` inputs across all measured BLAKE3 modes.
- Fast non-crypto hashes are not a headline win against the expanded field: XXH3 and RapidHash both lose to the fastest external implementation on geomean.
- Auth/KDF/signing is mixed against the fastest external implementation: 0.97x geomean, with Ed25519 signing still strong at 1.47x all-pair geomean.

## Linux All-Pair Scorecard

| Area | Baseline | Pairs | W/T/L | Win % | Geomean | Median |
|---|---|---:|---:|---:|---:|---:|
| Checksums | `crc`, `crc32fast`, `crc32c`, `crc64fast`, `crc-fast`, `crc64fast-nvme` | 990 | 785/139/66 | 79% | 4.29x | 2.28x |
| SHA-3 | RustCrypto `sha3` | 414 | 387/27/0 | 93% | 2.18x | 2.15x |
| SHAKE | RustCrypto `sha3` | 198 | 190/8/0 | 96% | 2.61x | 2.22x |
| AEAD | RustCrypto AEADs, `aws-lc-rs`, `ring`, `dryoc`, `aegis-crate` | 2574 | 1472/307/795 | 57% | 1.46x | 1.15x |
| BLAKE3 | `blake3` | 432 | 219/175/38 | 51% | 1.41x | 1.05x |
| Auth / KDF / Sign | RustCrypto, `aws-lc-rs`, `ring`, `dryoc`, `dalek` | 1809 | 1051/381/377 | 58% | 1.21x | 1.09x |
| Ascon hash/XOF | `ascon-hash` | 198 | 128/70/0 | 65% | 1.31x | 1.18x |
| BLAKE2 | RustCrypto `blake2`, `dryoc` | 1071 | 648/417/6 | 61% | 1.24x | 1.09x |
| SHA-2 | RustCrypto `sha2`, `aws-lc-rs`, `ring` | 1125 | 657/405/63 | 58% | 1.41x | 1.09x |
| XXH3 | `xxhash-rust`, `gxhash`, `ahash`, `foldhash` | 528 | 251/84/193 | 48% | 0.98x | 1.01x |
| RapidHash | `rapidhash`, `gxhash`, `ahash`, `foldhash` | 1056 | 320/284/452 | 30% | 0.85x | 0.99x |

## Linux Fastest-External Scorecard

| Area | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Checksums | 693 | 514/120/59 | 74% | 5.00x | 2.27x |
| SHA-3 | 414 | 387/27/0 | 93% | 2.18x | 2.15x |
| SHAKE | 198 | 190/8/0 | 96% | 2.61x | 2.22x |
| AEAD | 1386 | 756/191/439 | 55% | 1.36x | 1.10x |
| BLAKE3 | 432 | 219/175/38 | 51% | 1.41x | 1.05x |
| Auth / KDF / Sign | 603 | 200/180/223 | 33% | 0.97x | 1.00x |
| Ascon hash/XOF | 198 | 128/70/0 | 65% | 1.31x | 1.18x |
| BLAKE2 | 846 | 431/409/6 | 51% | 1.08x | 1.06x |
| SHA-2 | 531 | 221/270/40 | 42% | 1.32x | 1.04x |
| XXH3 | 198 | 50/24/124 | 25% | 0.66x | 0.73x |
| RapidHash | 396 | 43/56/297 | 11% | 0.51x | 0.52x |

## AEAD Detail

| Primitive | All pairs | W/T/L | Geomean | Median | Fastest-only pairs | W/T/L | Geomean | Median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AES-128-GCM | 594 | 279/36/279 | 1.33x | 1.00x | 198 | 39/15/144 | 0.69x | 0.63x |
| AES-128-GCM encrypt | 297 | 124/16/157 | 1.27x | 0.91x | 99 | 15/4/80 | 0.65x | 0.56x |
| AES-128-GCM decrypt | 297 | 155/20/122 | 1.40x | 1.12x | 99 | 24/11/64 | 0.73x | 0.79x |
| AES-256-GCM | 594 | 277/25/292 | 1.38x | 0.96x | 198 | 39/4/155 | 0.70x | 0.62x |
| AES-256-GCM encrypt | 297 | 125/12/160 | 1.33x | 0.91x | 99 | 15/1/83 | 0.67x | 0.54x |
| AES-256-GCM decrypt | 297 | 152/13/132 | 1.43x | 1.08x | 99 | 24/3/72 | 0.73x | 0.74x |
| AES-128-GCM-SIV | 198 | 173/12/13 | 3.10x | 1.77x | 198 | 173/12/13 | 3.10x | 1.77x |
| AES-128-GCM-SIV encrypt | 99 | 84/8/7 | 3.36x | 1.83x | 99 | 84/8/7 | 3.36x | 1.83x |
| AES-128-GCM-SIV decrypt | 99 | 89/4/6 | 2.86x | 1.71x | 99 | 89/4/6 | 2.86x | 1.71x |
| AES-256-GCM-SIV | 198 | 187/4/7 | 3.56x | 1.79x | 198 | 187/4/7 | 3.56x | 1.79x |
| AES-256-GCM-SIV encrypt | 99 | 93/3/3 | 3.88x | 1.80x | 99 | 93/3/3 | 3.88x | 1.80x |
| AES-256-GCM-SIV decrypt | 99 | 94/1/4 | 3.27x | 1.79x | 99 | 94/1/4 | 3.27x | 1.79x |
| ChaCha20-Poly1305 | 594 | 301/110/183 | 1.02x | 1.05x | 198 | 63/36/99 | 0.85x | 0.95x |
| ChaCha20-Poly1305 encrypt | 297 | 150/54/93 | 1.01x | 1.05x | 99 | 31/16/52 | 0.84x | 0.92x |
| ChaCha20-Poly1305 decrypt | 297 | 151/56/90 | 1.03x | 1.05x | 99 | 32/20/47 | 0.86x | 0.96x |
| XChaCha20-Poly1305 | 198 | 143/55/0 | 1.28x | 1.17x | 198 | 143/55/0 | 1.28x | 1.17x |
| XChaCha20-Poly1305 encrypt | 99 | 68/31/0 | 1.28x | 1.15x | 99 | 68/31/0 | 1.28x | 1.15x |
| XChaCha20-Poly1305 decrypt | 99 | 75/24/0 | 1.29x | 1.19x | 99 | 75/24/0 | 1.29x | 1.19x |
| AEGIS-256 | 198 | 112/65/21 | 1.45x | 1.07x | 198 | 112/65/21 | 1.45x | 1.07x |
| AEGIS-256 encrypt | 99 | 46/34/19 | 1.36x | 1.03x | 99 | 46/34/19 | 1.36x | 1.03x |
| AEGIS-256 decrypt | 99 | 66/31/2 | 1.55x | 1.27x | 99 | 66/31/2 | 1.55x | 1.27x |

## External Competitor Pressure

This table is all-pair only and answers which competitors are actually applying pressure.

| External | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| `rustcrypto` | 2529 | 1613/776/140 | 64% | 1.59x | 1.12x |
| `aws-lc-rs` | 1458 | 691/214/553 | 47% | 1.03x | 1.03x |
| `ring` | 1440 | 765/169/506 | 53% | 1.16x | 1.09x |
| `sha3` | 612 | 577/35/0 | 94% | 2.31x | 2.15x |
| `sha2` | 531 | 269/255/7 | 51% | 1.44x | 1.05x |
| `blake3` | 432 | 219/175/38 | 51% | 1.41x | 1.05x |
| `gxhash` | 396 | 33/9/354 | 8% | 0.40x | 0.41x |
| `rapidhash` | 396 | 124/210/62 | 31% | 1.10x | 1.01x |
| `dryoc` | 315 | 290/18/7 | 92% | 1.81x | 1.71x |
| `crc` | 297 | 291/2/4 | 98% | 19.35x | 29.93x |
| `crc-fast` | 297 | 175/99/23 | 59% | 1.99x | 1.13x |
| `ahash` | 297 | 196/15/86 | 66% | 1.32x | 1.34x |
| `foldhash` | 297 | 112/66/119 | 38% | 1.08x | 0.98x |
| `xxhash-rust` | 198 | 106/68/24 | 54% | 1.19x | 1.07x |
| `ascon-hash` | 198 | 128/70/0 | 65% | 1.31x | 1.18x |
| `aegis-crate` | 198 | 112/65/21 | 57% | 1.45x | 1.07x |
| `dalek` | 108 | 88/13/7 | 81% | 1.35x | 1.31x |
| `crc32fast` | 99 | 82/9/8 | 83% | 2.44x | 2.02x |
| `crc32c` | 99 | 90/4/5 | 91% | 2.73x | 2.17x |
| `crc64fast` | 99 | 74/12/13 | 75% | 2.35x | 1.97x |
| `crc64fast-nvme` | 99 | 73/13/13 | 74% | 2.35x | 1.96x |

## Platform Scorecard

All-pair comparisons; POWER10, s390x, and RISC-V have fewer external pairs because some competitors do not support every target.

| Platform | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| AMD Zen 4 | 1177 | 744/185/248 | 63% | 1.23x | 1.10x |
| AMD Zen 5 | 1177 | 663/291/223 | 56% | 1.25x | 1.09x |
| Intel Ice Lake | 1177 | 700/199/278 | 59% | 1.22x | 1.12x |
| Intel Sapphire Rapids | 1177 | 689/190/298 | 59% | 1.20x | 1.10x |
| AWS Graviton3 | 1177 | 556/334/287 | 47% | 1.36x | 1.02x |
| AWS Graviton4 | 1177 | 542/365/270 | 46% | 1.36x | 1.04x |
| IBM POWER10 | 1111 | 720/310/81 | 65% | 1.86x | 1.23x |
| IBM Z / s390x | 1111 | 950/95/66 | 86% | 3.43x | 2.65x |
| RISE RISC-V | 1111 | 544/328/239 | 49% | 1.16x | 1.05x |

## README Numbers

These are the public-facing claims worth carrying into `README.md`:

- **6108 faster comparisons** across **10395** matched Linux CI external comparisons.
- **1.46x geomean** Linux CI speedup overall across all external pairs; **1.43x** against the fastest external implementation per case.
- **SHA-3 / SHAKE:** 2.18x / 2.61x geomean vs RustCrypto `sha3`.
- **BLAKE3:** 2.23x geomean for `>=64 KiB` inputs vs `blake3` across all measured BLAKE3 modes.
- **AEAD:** 1.46x all-pair geomean; AES-128-GCM-SIV is 3.10x, AES-256-GCM-SIV is 3.56x, while AES-128-GCM and AES-256-GCM are 0.69x / 0.70x against the fastest external backend.
- **Ed25519 / X25519:** Ed25519 signing 1.47x all-pair geomean; X25519 1.16x all-pair geomean.
- **Checksums:** 4.29x all-pair geomean; 5.00x against the fastest external implementation per case.

## Raw Results

| Runner | Result |
|---|---|
| AMD Zen 4 | `benchmark_results/2026-05-05/linux/amd-zen4/results.txt` |
| AMD Zen 5 | `benchmark_results/2026-05-05/linux/amd-zen5/results.txt` |
| Intel Ice Lake | `benchmark_results/2026-05-05/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `benchmark_results/2026-05-05/linux/intel-spr/results.txt` |
| AWS Graviton3 | `benchmark_results/2026-05-05/linux/graviton3/results.txt` |
| AWS Graviton4 | `benchmark_results/2026-05-05/linux/graviton4/results.txt` |
| IBM POWER10 | `benchmark_results/2026-05-05/linux/ibm-power10/results.txt` |
| IBM Z / s390x | `benchmark_results/2026-05-05/linux/ibm-s390x/results.txt` |
| RISE RISC-V | `benchmark_results/2026-05-05/linux/rise-riscv/results.txt` |

## Methodology Notes

- Parsed 16542 Criterion median timings from the nine structured result files.
- A comparison is matched when a Criterion ID has a `rscrypto` path component and an external ID with the same platform, bench binary, path length, operation/mode components, and input size, differing only in that implementation component.
- Fastest-external rows collapse multiple external implementations for the same case down to the fastest observed external median.
- The raw result files were normalized after extraction to use the workflow creation time in their header and to strip ANSI color codes from Cargo output.
- 216 diagnostic timings were parsed but excluded because they do not have a matched `rscrypto` vs external implementation shape.
