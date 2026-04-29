# Benchmark Overview

Sources:

- Linux: Bench CI run [#25069224636](https://github.com/loadingalias/rscrypto/actions/runs/25069224636), created 2026-04-28 17:57:49 UTC.
- macOS: local Apple Silicon run at `benchmark_results/2026-04-28/macos/aarch64/results.txt`, started 2026-04-28 `23_13_22`.

Commit: `0ea1b73b2a40fd164b702e8b927d5930a996f846`.

Scope: nine Linux CI runners plus one local macOS Apple Silicon run.
The macOS run includes KMAC, cSHAKE, and password-hashing comparisons that are not in the Linux CI scope, so Linux and macOS totals are reported separately.
Speedup is `external_crate_time / rscrypto_time`; above `1.00x` means `rscrypto` is faster.
Wins are `>1.05x`, ties are `0.95x..1.05x`, losses are `<0.95x`.
Linux headline numbers below were recomputed from raw Criterion median times on 2026-04-29.
The macOS scorecard is a public-facing subset of the local log; the raw file contains extra benchmark groups outside that scorecard.

## Headline

Linux CI remains the broad headline: `rscrypto` wins **3717 / 5796** matched external comparisons.
The macOS scorecard remains separate at **357 / 681** faster listed comparisons.

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Linux CI | 5796 | 3717/1792/287 | 64% | 1.75x | 1.14x |
| macOS Apple Silicon | 681 | 357/281/43 | 52% | 1.54x | 1.06x |

What matters:

- SHA-3 and SHAKE are the cleanest cryptographic wins: **2.18x** and **2.60x** geomean vs RustCrypto `sha3`.
- AEAD is broadly ahead: **1.84x** geomean across RustCrypto AEADs and `aegis`, with AES-256-GCM and AES-256-GCM-SIV carrying the strongest user-facing story.
- BLAKE3 is not universally ahead on tiny inputs, but large one-shot/keyed/derive-key inputs are strong: **2.37x** geomean for `>=64 KiB` vs the `blake3` crate.
- Ed25519 signing wins everywhere measured: **1.57x** geomean vs `ed25519-dalek`.
- Checksums are decisive: **4.41x** geomean across `crc`, `crc32fast`, `crc64fast`, `crc-fast`, and related baselines.
- macOS Apple Silicon is strongest in AEAD (**2.60x**) and checksums (**4.18x**); SHA-2 and SHA-3 are mostly parity there (**1.02x** and **1.01x**).

## Linux Category Scorecard

| Area | Baseline | Pairs | W/T/L | Win % | Geomean | Median | Strongest useful result |
|---|---|---:|---:|---:|---:|---:|---|
| Checksums | `crc`, `crc32fast`, `crc64fast`, `crc-fast` | 990 | 801/135/54 | 81% | 4.41x | 2.41x | CRC32C 2.24x geomean; CRC64/NVMe 2.35x geomean |
| SHA-3 | RustCrypto `sha3` | 414 | 385/27/2 | 93% | 2.18x | 2.15x | up to 25.83x on IBM Z |
| SHAKE/cSHAKE | RustCrypto `sha3` | 198 | 189/8/1 | 95% | 2.60x | 2.23x | up to 22.41x on IBM Z |
| AEAD | RustCrypto AEADs, `aegis` | 990 | 736/184/70 | 74% | 1.84x | 1.43x | AES-256-GCM decrypt 2.51x geomean; AES-256-GCM-SIV encrypt 3.79x geomean |
| BLAKE3 | `blake3` | 432 | 223/183/26 | 52% | 1.42x | 1.06x | one-shot/keyed/derive-key `>=64 KiB` inputs 2.37x geomean; up to 7.70x |
| Auth / Sign / KDF | RustCrypto, `ed25519-dalek`, `x25519-dalek` | 603 | 327/236/40 | 54% | 1.26x | 1.06x | Ed25519 sign 1.57x geomean; X25519 1.37x geomean |
| Ascon | `ascon-hash` | 198 | 124/73/1 | 63% | 1.31x | 1.18x | up to 2.52x |
| XXH3 | `xxhash-rust` | 198 | 107/67/24 | 54% | 1.18x | 1.08x | up to 2.41x |
| RapidHash | `rapidhash` | 396 | 130/205/61 | 33% | 1.09x | 1.00x | up to 4.98x; broad parity remains |
| BLAKE2 | RustCrypto `blake2` | 846 | 424/421/1 | 50% | 1.08x | 1.05x | up to 1.70x |
| SHA-2 | RustCrypto `sha2` | 531 | 271/253/7 | 51% | 1.45x | 1.05x | up to 121.01x on SPR large-buffer SHA-224 |

## macOS Apple Silicon Scorecard

| Area | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Checksums | 110 | 85/21/4 | 77% | 4.18x | 1.85x |
| AEAD | 110 | 105/5/0 | 95% | 2.60x | 1.27x |
| BLAKE3 | 48 | 16/32/0 | 33% | 1.23x | 1.01x |
| Auth / Sign / KDF | 78 | 30/45/3 | 38% | 1.07x | 1.01x |
| Password hashing | 15 | 5/7/3 | 33% | 0.77x | 0.99x |
| SHAKE/cSHAKE | 33 | 22/6/5 | 67% | 1.11x | 1.10x |
| Ascon | 22 | 16/6/0 | 73% | 1.09x | 1.06x |
| RapidHash | 44 | 16/23/5 | 36% | 1.18x | 1.00x |
| XXH3 | 22 | 7/8/7 | 32% | 1.04x | 0.99x |
| BLAKE2 | 94 | 33/61/0 | 35% | 1.04x | 1.02x |
| SHA-3 | 46 | 11/26/9 | 24% | 1.01x | 1.02x |
| SHA-2 | 59 | 11/41/7 | 19% | 1.02x | 1.00x |

## Platform Scorecard

| Platform | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| AMD Zen 4 | 644 | 500/128/16 | 78% | 1.62x | 1.20x |
| AMD Zen 5 | 644 | 394/220/30 | 61% | 1.65x | 1.12x |
| AWS Graviton3 | 644 | 341/259/44 | 53% | 1.65x | 1.07x |
| AWS Graviton4 | 644 | 336/288/20 | 52% | 1.66x | 1.06x |
| IBM POWER10 | 644 | 331/296/17 | 51% | 1.94x | 1.06x |
| IBM Z / s390x | 644 | 529/78/37 | 82% | 3.03x | 2.36x |
| Intel Ice Lake | 644 | 467/152/25 | 73% | 1.62x | 1.23x |
| Intel Sapphire Rapids | 644 | 501/100/43 | 78% | 1.92x | 1.27x |
| RISE RISC-V | 644 | 318/271/55 | 49% | 1.18x | 1.05x |
| macOS Apple Silicon | 681 | 357/281/43 | 52% | 1.54x | 1.06x |

## README Numbers

These are the public-facing claims worth carrying into `README.md`:

- **3717 faster comparisons** across **5796** matched Linux CI comparisons.
- **1.75x geomean** Linux CI speedup overall.
- **357 faster comparisons** across **681** matched macOS Apple Silicon comparisons; **1.54x geomean** speedup.
- **SHA-3 / SHAKE:** 2.18x / 2.60x geomean vs RustCrypto `sha3`; up to 25.83x / 22.41x.
- **BLAKE3:** 2.37x geomean on one-shot/keyed/derive-key `>=64 KiB` inputs vs `blake3`; up to 7.70x.
- **AEAD:** 1.84x geomean; AES-256-GCM encrypt/decrypt at 2.25x / 2.51x geomean; AES-256-GCM-SIV encrypt at 3.79x geomean.
- **Ed25519 / X25519:** Ed25519 signing 1.57x geomean vs `ed25519-dalek`; X25519 1.37x geomean vs `x25519-dalek`.
- **Checksums:** 4.41x geomean; CRC32C 2.24x geomean; CRC64/NVMe 2.35x geomean.
- **macOS Apple Silicon:** AEAD 2.60x geomean; checksums 4.18x geomean; SHA-2/SHA-3 are parity, not headline wins.

## Raw Results

| Runner | Result |
|---|---|
| AMD Zen 4 | `benchmark_results/2026-04-28/linux/amd-zen4/results.txt` |
| AMD Zen 5 | `benchmark_results/2026-04-28/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `benchmark_results/2026-04-28/linux/graviton3/results.txt` |
| AWS Graviton4 | `benchmark_results/2026-04-28/linux/graviton4/results.txt` |
| IBM POWER10 | `benchmark_results/2026-04-28/linux/ibm-power10/results.txt` |
| IBM Z / s390x | `benchmark_results/2026-04-28/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `benchmark_results/2026-04-28/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `benchmark_results/2026-04-28/linux/intel-spr/results.txt` |
| RISE RISC-V | `benchmark_results/2026-04-28/linux/rise-riscv/results.txt` |
| macOS Apple Silicon | `benchmark_results/2026-04-28/macos/aarch64/results.txt` |
