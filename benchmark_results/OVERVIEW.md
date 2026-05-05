# Benchmark Overview

Sources:

- Linux: Bench CI run [#25375704946](https://github.com/loadingalias/rscrypto/actions/runs/25375704946), created 2026-05-05 12:14:28 UTC.
- Raw files: `benchmark_results/2026-05-05/linux/*/results.txt`.

Commit: `dc3cf94f11e0c73f50ad989039c16b617cf1d414`.

Scope: nine Linux CI runners. The 2026-05-04 local macOS file is a failed local compile log, so it is excluded.
Speedup is `external_crate_time / rscrypto_time`; above `1.00x` means `rscrypto` is faster.
Wins are `>1.05x`, ties are `0.95x..1.05x`, losses are `<0.95x`.
All-pair comparisons count every matched external implementation. Fastest-external comparisons keep only the fastest external implementation for each platform, operation, mode, and input size.

## Headline

Linux all-pair headline: `rscrypto` wins **5679 / 9603** matched Linux CI external comparisons.
Against only the fastest external implementation per case, it wins **2941 / 5499** comparisons.

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Linux CI, all external pairs | 9603 | 5679/2232/1692 | 59% | 1.43x | 1.12x |
| Linux CI, fastest external per case | 5499 | 2941/1489/1069 | 53% | 1.42x | 1.07x |

What matters:

- The expanded competitor set changed the headline from a broad RustCrypto-style comparison to a harder field: 9603 all-pair comparisons and 5499 fastest-external comparisons.
- SHA-3 and SHAKE remain clean wins: **2.18x** and **2.62x** geomean vs RustCrypto `sha3`, with no losses in this run.
- Checksums remain the strongest public story: **4.32x** all-pair geomean.
- AEAD is mixed after adding `aws-lc-rs` and `ring`: **1.28x** all-pair geomean, but AES-256-GCM loses to the fastest external backend on most cases while AES-256-GCM-SIV remains a decisive win.
- BLAKE3 is still strong on large inputs: **2.23x** geomean for `>=64 KiB` inputs across all measured BLAKE3 modes.
- Fast non-crypto hashes are not a headline win against the expanded field: RapidHash and XXH3 lose to the fastest external implementation on geomean.
- Coverage gap: this CI run contains AES-256-GCM and AES-256-GCM-SIV results, but no AES-128-GCM or AES-128-GCM-SIV result IDs. The CI selector has been fixed after this run; another Bench run is needed for AES-128 numbers.

## Linux All-Pair Scorecard

| Area | Baseline | Pairs | W/T/L | Win % | Geomean | Median |
|---|---|---:|---:|---:|---:|---:|
| Checksums | `crc`, `crc32fast`, `crc32c`, `crc64fast`, `crc-fast`, `crc64fast-nvme` | 990 | 792/134/64 | 80% | 4.32x | 2.31x |
| SHA-3 | RustCrypto `sha3` | 414 | 388/26/0 | 94% | 2.18x | 2.16x |
| SHAKE | RustCrypto `sha3` | 198 | 191/7/0 | 96% | 2.62x | 2.17x |
| AEAD | RustCrypto AEADs, `aws-lc-rs`, `ring`, `dryoc`, `aegis-crate` | 1782 | 1006/264/512 | 56% | 1.28x | 1.12x |
| BLAKE3 | `blake3` | 432 | 237/165/30 | 55% | 1.43x | 1.09x |
| Auth / KDF / Sign | RustCrypto, `aws-lc-rs`, `ring`, `dryoc`, `dalek` | 1809 | 1046/379/384 | 58% | 1.21x | 1.09x |
| Ascon hash/XOF | `ascon-hash` | 198 | 128/68/2 | 65% | 1.31x | 1.22x |
| BLAKE2 | RustCrypto `blake2` | 1071 | 653/414/4 | 61% | 1.24x | 1.09x |
| SHA-2 | RustCrypto `sha2`, `aws-lc-rs`, `ring` | 1125 | 660/404/61 | 59% | 1.43x | 1.09x |
| XXH3 | `xxhash-rust`, `gxhash`, `ahash`, `foldhash` | 528 | 253/85/190 | 48% | 0.99x | 1.01x |
| RapidHash | `rapidhash`, `gxhash`, `ahash`, `foldhash` | 1056 | 325/286/445 | 31% | 0.85x | 0.99x |

## Linux Fastest-External Scorecard

| Area | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Checksums | 693 | 517/118/58 | 75% | 5.02x | 2.31x |
| SHA-3 | 414 | 388/26/0 | 94% | 2.18x | 2.16x |
| SHAKE | 198 | 191/7/0 | 96% | 2.62x | 2.17x |
| AEAD | 990 | 528/174/288 | 53% | 1.25x | 1.07x |
| BLAKE3 | 432 | 237/165/30 | 55% | 1.43x | 1.09x |
| Auth / KDF / Sign | 603 | 197/176/230 | 33% | 0.97x | 1.00x |
| Ascon hash/XOF | 198 | 128/68/2 | 65% | 1.31x | 1.22x |
| BLAKE2 | 846 | 434/408/4 | 51% | 1.08x | 1.05x |
| SHA-2 | 531 | 226/267/38 | 43% | 1.33x | 1.04x |
| XXH3 | 198 | 51/25/122 | 26% | 0.66x | 0.73x |
| RapidHash | 396 | 44/55/297 | 11% | 0.51x | 0.52x |

## AEAD Detail

| Primitive | All pairs | W/T/L | Geomean | Median | Fastest-only pairs | W/T/L | Geomean | Median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AES-256-GCM | 594 | 276/29/289 | 1.12x | 0.97x | 198 | 37/9/152 | 0.56x | 0.57x |
| AES-256-GCM encrypt | 297 | 122/16/159 | 1.08x | 0.91x | 99 | 14/2/83 | 0.54x | 0.52x |
| AES-256-GCM decrypt | 297 | 154/13/130 | 1.16x | 1.10x | 99 | 23/7/69 | 0.59x | 0.71x |
| AES-256-GCM-SIV | 198 | 187/3/8 | 3.54x | 1.81x | 198 | 187/3/8 | 3.54x | 1.81x |
| AES-256-GCM-SIV encrypt | 99 | 93/2/4 | 3.85x | 1.81x | 99 | 93/2/4 | 3.85x | 1.81x |
| AES-256-GCM-SIV decrypt | 99 | 94/1/4 | 3.26x | 1.81x | 99 | 94/1/4 | 3.26x | 1.81x |
| ChaCha20-Poly1305 | 594 | 299/104/191 | 1.01x | 1.05x | 198 | 60/34/104 | 0.84x | 0.94x |
| XChaCha20-Poly1305 | 198 | 139/58/1 | 1.27x | 1.17x | 198 | 139/58/1 | 1.27x | 1.17x |
| AEGIS-256 | 198 | 105/70/23 | 1.43x | 1.06x | 198 | 105/70/23 | 1.43x | 1.06x |

## External Competitor Pressure

This table is all-pair only and answers which newly added competitors are actually applying pressure.

| External | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| `rustcrypto` | 2133 | 1289/758/86 | 60% | 1.39x | 1.09x |
| `aws-lc-rs` | 1260 | 622/195/443 | 49% | 1.05x | 1.05x |
| `ring` | 1242 | 698/157/387 | 56% | 1.14x | 1.11x |
| `sha3` | 612 | 579/33/0 | 95% | 2.31x | 2.16x |
| `sha2` | 531 | 270/254/7 | 51% | 1.46x | 1.05x |
| `blake3` | 432 | 237/165/30 | 55% | 1.43x | 1.09x |
| `gxhash` | 396 | 34/8/354 | 9% | 0.40x | 0.41x |
| `rapidhash` | 396 | 128/206/62 | 32% | 1.10x | 1.01x |
| `dryoc` | 315 | 292/16/7 | 93% | 1.81x | 1.70x |
| `ahash` | 297 | 196/13/88 | 66% | 1.33x | 1.36x |
| `crc` | 297 | 291/2/4 | 98% | 19.35x | 29.93x |
| `crc-fast` | 297 | 180/94/23 | 61% | 2.03x | 1.17x |
| `foldhash` | 297 | 113/76/108 | 38% | 1.08x | 0.99x |
| `aegis-crate` | 198 | 105/70/23 | 53% | 1.43x | 1.06x |
| `ascon-hash` | 198 | 128/68/2 | 65% | 1.31x | 1.22x |
| `xxhash-rust` | 198 | 107/68/23 | 54% | 1.19x | 1.09x |
| `dalek` | 108 | 89/11/8 | 82% | 1.35x | 1.32x |
| `crc32c` | 99 | 91/3/5 | 92% | 2.74x | 2.19x |
| `crc32fast` | 99 | 81/9/9 | 82% | 2.43x | 2.00x |
| `crc64fast` | 99 | 75/12/12 | 76% | 2.38x | 2.04x |
| `crc64fast-nvme` | 99 | 74/14/11 | 75% | 2.37x | 1.96x |

## Platform Scorecard

All-pair comparisons; POWER10, s390x, and RISC-V have fewer external pairs because some competitors do not support every target.

| Platform | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| AMD Zen 4 | 1089 | 701/182/206 | 64% | 1.28x | 1.10x |
| AMD Zen 5 | 1089 | 622/286/181 | 57% | 1.29x | 1.10x |
| Intel Ice Lake | 1089 | 669/184/236 | 61% | 1.27x | 1.13x |
| Intel Sapphire Rapids | 1089 | 656/182/251 | 60% | 1.26x | 1.11x |
| AWS Graviton3 | 1089 | 501/326/262 | 46% | 1.25x | 1.02x |
| AWS Graviton4 | 1089 | 488/363/238 | 45% | 1.25x | 1.03x |
| IBM POWER10 | 1023 | 651/308/64 | 64% | 1.76x | 1.18x |
| IBM Z / s390x | 1023 | 871/86/66 | 85% | 3.02x | 2.38x |
| RISE RISC-V | 1023 | 520/315/188 | 51% | 1.17x | 1.05x |

## README Numbers

These are the public-facing claims worth carrying into `README.md` after the next AES-128-inclusive run confirms the missing AEAD coverage:

- **5679 faster comparisons** across **9603** matched Linux CI external comparisons.
- **1.43x geomean** Linux CI speedup overall across all external pairs; **1.42x** against the fastest external implementation per case.
- **SHA-3 / SHAKE:** 2.18x / 2.62x geomean vs RustCrypto `sha3`.
- **BLAKE3:** 2.23x geomean for `>=64 KiB` inputs vs `blake3` across all measured BLAKE3 modes.
- **AEAD:** 1.28x all-pair geomean; AES-256-GCM-SIV is 3.54x, while AES-256-GCM is 0.56x against the fastest external backend.
- **Ed25519 / X25519:** Ed25519 signing 1.47x all-pair geomean; X25519 1.15x all-pair geomean.
- **Checksums:** 4.32x all-pair geomean.
- **Coverage warning:** do not publish AES-128 AEAD claims from this run; rerun Bench after the selector fix.

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

- Parsed 15354 Criterion median timings from the nine structured result files.
- A comparison is matched when a Criterion ID has a `rscrypto` path component and an external ID with the same platform, bench binary, path length, operation/mode components, and input size, differing only in that implementation component.
- Fastest-external rows collapse multiple external implementations for the same case down to the fastest observed external median.
- The raw result files were normalized after extraction to use the workflow creation time in their header and to strip ANSI color codes from Cargo output.
