# Benchmark Overview

Sources:

- Linux: Bench CI run [#25069224636](https://github.com/loadingalias/rscrypto/actions/runs/25069224636), created 2026-04-28 17:57:49 UTC.
- macOS: local Apple Silicon run at `benchmark_results/2026-04-28/macos/aarch64/results.txt`, started 2026-04-28 `23_13_22`.

Commit: `0ea1b73b2a40fd164b702e8b927d5930a996f846`.

Scope: nine Linux CI runners plus one local macOS Apple Silicon run.
The macOS run includes KMAC, cSHAKE, and password-hashing comparisons that are not in the Linux CI scope, so Linux and macOS totals are reported separately.
Speedup is `external_crate_time / rscrypto_time`; above `1.00x` means `rscrypto` is faster.
Wins are `>1.05x`, ties are `0.95x..1.05x`, losses are `<0.95x`.

## Headline

Linux CI remains the broad headline: `rscrypto` wins **3716 / 5796** matched external comparisons.
Adding the local macOS run gives a count-only total of **4073 / 6477** faster listed comparisons.

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Linux CI | 5796 | 3716/1796/284 | 64% | 1.76x | 1.14x |
| macOS Apple Silicon | 681 | 357/281/43 | 52% | 1.54x | 1.06x |

What matters:

- SHA-3 and SHAKE are the cleanest cryptographic wins: **2.18x** and **2.60x** geomean vs RustCrypto `sha3`.
- AEAD is broadly ahead: **1.83x** geomean across RustCrypto AEADs and `aegis`, with AES-256-GCM and AES-256-GCM-SIV carrying the strongest user-facing story.
- BLAKE3 is not universally ahead on tiny inputs, but large buffers are strong: **2.37x** geomean for `>=64 KiB` vs the `blake3` crate.
- Ed25519 signing wins everywhere measured: **1.57x** geomean vs `ed25519-dalek`.
- Checksums are decisive: **4.42x** geomean across `crc`, `crc32fast`, `crc64fast`, `crc-fast`, and related baselines.
- macOS Apple Silicon is strongest in AEAD (**2.60x**) and checksums (**4.18x**); SHA-2 and SHA-3 are mostly parity there (**1.02x** and **1.01x**).

## Linux Category Scorecard

| Area | Baseline | Pairs | W/T/L | Win % | Geomean | Median | Strongest useful result |
|---|---|---:|---:|---:|---:|---:|---|
| Checksums | `crc`, `crc32fast`, `crc64fast`, `crc-fast` | 990 | 800/137/53 | 81% | 4.42x | 2.42x | CRC32C 2.25x geomean; CRC64/NVMe 2.36x geomean |
| SHA-3 | RustCrypto `sha3` | 414 | 384/29/1 | 93% | 2.18x | 2.15x | up to 25.66x on IBM Z |
| SHAKE/cSHAKE | RustCrypto `sha3` | 198 | 189/8/1 | 95% | 2.60x | 2.23x | up to 22.27x on IBM Z |
| AEAD | RustCrypto AEADs, `aegis` | 990 | 737/183/70 | 74% | 1.83x | 1.43x | AES-256-GCM decrypt 2.51x geomean; AES-256-GCM-SIV encrypt 3.79x geomean |
| BLAKE3 | `blake3` | 432 | 221/185/26 | 51% | 1.42x | 1.05x | `>=64 KiB` buffers 2.37x geomean; up to 7.88x |
| Auth / Sign / KDF | RustCrypto, `ed25519-dalek`, `x25519-dalek` | 702 | 380/283/39 | 54% | 1.26x | 1.06x | Ed25519 sign 1.57x geomean; X25519 1.37x geomean |
| Ascon | `ascon-hash` | 198 | 122/76/0 | 62% | 1.31x | 1.18x | up to 2.52x |
| XXH3 | `xxhash-rust` | 198 | 107/68/23 | 54% | 1.18x | 1.07x | up to 2.45x |
| RapidHash | `rapidhash` | 396 | 131/203/62 | 33% | 1.09x | 1.00x | up to 4.99x; broad parity remains |
| BLAKE2 | RustCrypto `blake2` | 846 | 426/420/0 | 50% | 1.08x | 1.05x | up to 1.71x; no loss bucket in this run |
| SHA-2 | RustCrypto `sha2` | 432 | 219/204/9 | 51% | 1.50x | 1.05x | up to 120.39x on SPR large-buffer SHA-224 |

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
| AMD Zen 4 | 644 | 498/129/17 | 77% | 1.62x | 1.20x |
| AMD Zen 5 | 644 | 393/221/30 | 61% | 1.65x | 1.12x |
| AWS Graviton3 | 644 | 341/260/43 | 53% | 1.65x | 1.07x |
| AWS Graviton4 | 644 | 337/286/21 | 52% | 1.66x | 1.06x |
| IBM POWER10 | 644 | 332/296/16 | 52% | 1.94x | 1.05x |
| IBM Z / s390x | 644 | 528/81/35 | 82% | 3.03x | 2.38x |
| Intel Ice Lake | 644 | 467/155/22 | 73% | 1.62x | 1.23x |
| Intel Sapphire Rapids | 644 | 499/99/46 | 77% | 1.93x | 1.28x |
| RISE RISC-V | 644 | 321/269/54 | 50% | 1.18x | 1.05x |
| macOS Apple Silicon | 681 | 357/281/43 | 52% | 1.54x | 1.06x |

## README Numbers

These are the public-facing claims worth carrying into `README.md`:

- **3716 faster comparisons** across **5796** matched Linux CI comparisons.
- **1.76x geomean** Linux CI speedup overall.
- **357 faster comparisons** across **681** matched macOS Apple Silicon comparisons; **1.54x geomean** speedup.
- **4073 faster comparisons** across **6477** listed matched comparisons when Linux CI and local macOS counts are added together.
- **SHA-3 / SHAKE:** 2.18x / 2.60x geomean vs RustCrypto `sha3`; up to 25.66x / 22.27x.
- **BLAKE3:** 2.37x geomean on `>=64 KiB` buffers vs `blake3`; up to 7.88x.
- **AEAD:** 1.83x geomean; AES-256-GCM encrypt/decrypt at 2.25x / 2.51x geomean; AES-256-GCM-SIV encrypt at 3.79x geomean.
- **Ed25519 / X25519:** Ed25519 signing 1.57x geomean vs `ed25519-dalek`; X25519 1.37x geomean vs `x25519-dalek`.
- **Checksums:** 4.42x geomean; CRC32C 2.25x geomean; CRC64/NVMe 2.36x geomean.
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
