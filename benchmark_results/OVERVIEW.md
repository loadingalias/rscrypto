# Benchmark Overview

Source: Bench CI run [#24962985907](https://github.com/loadingalias/rscrypto/actions/runs/24962985907) created 2026-04-26 17:43:45 UTC, plus local macOS Apple Silicon results from `benchmark_results/2026-04-26/macos/aarch64/results.txt`.
Commit: `f5c6ff7cb80b3cd168998dd152e934751817ec0d`

Scope: nine Linux CI runners plus local macOS aarch64. RISE RISC-V is included in this snapshot. Speedup is `competitor_time / rscrypto_time`; values above `1.00x` mean `rscrypto` is faster. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`.

## Executive Read

Overall: **3883W / 2163T / 467L = 60% win rate** across 6513 paired external comparisons. Geomean speedup is **1.64x** and median speedup is **1.11x**.
Strongest category by geomean: **Checksums** at 4.40x. Weakest category by geomean: **Password Hashing** at 0.76x.
Checksums: 885W/158T/57L, geomean 4.40x, worst 0.35x on macOS-A64 `crc32 vs crc32fast` case `64`.
SHA-3: 390W/64T/6L, geomean 2.02x, worst 0.92x on macOS-A64 `sha3-384 vs sha3` case `0`.
Ascon: 138W/81T/1L, geomean 1.28x, worst 0.92x on RISC-V `ascon-hash256 vs ascon-hash` case `0`.
AEAD: 797W/167T/136L, geomean 1.47x, worst 0.01x on RISC-V `aes-256-gcm-siv/decrypt vs rustcrypto` case `16384`.
Password Hashing: 5W/6T/4L, geomean 0.76x, worst 0.20x on macOS-A64 `argon2d-small vs rustcrypto` case `m=64_t=3_p=2`.
RapidHash: 119W/238T/83L, geomean 1.05x, worst 0.54x on s390x `rapidhash-v3-64 vs rapidhash` case `1`.
XXH3: 113W/72T/35L, geomean 1.16x, worst 0.37x on Zen5 `xxh3-64 vs xxhash-rust` case `256`.
BLAKE3: 241W/199T/40L, geomean 1.40x, worst 0.73x on s390x `blake3/xof vs blake3` case `1024`.

## Platforms

| Short | Platform | Architecture | Source | Results |
|---|---|---|---|---|
| Zen4 | AMD EPYC Zen 4 | x86-64 | ci | `benchmark_results/2026-04-26/linux/amd-zen4/results.txt` |
| Zen5 | AMD EPYC Zen 5 | x86-64 | ci | `benchmark_results/2026-04-26/linux/amd-zen5/results.txt` |
| SPR | Intel Xeon Sapphire Rapids | x86-64 | ci | `benchmark_results/2026-04-26/linux/intel-spr/results.txt` |
| ICL | Intel Xeon Ice Lake | x86-64 | ci | `benchmark_results/2026-04-26/linux/intel-icl/results.txt` |
| Grav3 | AWS Graviton3 | aarch64 | ci | `benchmark_results/2026-04-26/linux/graviton3/results.txt` |
| Grav4 | AWS Graviton4 | aarch64 | ci | `benchmark_results/2026-04-26/linux/graviton4/results.txt` |
| s390x | IBM Z | s390x | ci | `benchmark_results/2026-04-26/linux/ibm-s390x/results.txt` |
| POWER10 | IBM POWER10 | ppc64le | ci | `benchmark_results/2026-04-26/linux/ibm-power10/results.txt` |
| RISC-V | RISE RISC-V | riscv64 | ci | `benchmark_results/2026-04-26/linux/rise-riscv/results.txt` |
| macOS-A64 | macOS Apple Silicon | aarch64 | local | `benchmark_results/2026-04-26/macos/aarch64/results.txt` |

## Overall Scoreboard

| W | T | L | Total | Win % | Geomean | Median | Worst | Best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3883 | 2163 | 467 | 6513 | 60% | 1.64x | 1.11x | 0.01x | 228.06x |

## By Category

| Category | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Platform | Worst Case | Best | Best Platform | Best Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Checksums | 885 | 158 | 57 | 1100 | 80% | 4.40x | 2.31x | 0.35x | macOS-A64 | `crc32 vs crc32fast` `64` | 228.06x | SPR | `crc16-ccitt vs crc` `16384` |
| SHA-2 | 284 | 296 | 10 | 590 | 48% | 1.40x | 1.05x | 0.01x | SPR | `sha256/streaming vs sha2` `64B` | 123.02x | SPR | `sha224 vs sha2` `262144` |
| SHA-3 | 390 | 64 | 6 | 460 | 85% | 2.02x | 2.00x | 0.92x | macOS-A64 | `sha3-384 vs sha3` `0` | 24.84x | s390x | `sha3-512 vs sha3` `1048576` |
| SHAKE | 201 | 23 | 7 | 231 | 87% | 2.29x | 2.05x | 0.90x | macOS-A64 | `shake128 vs sha3` `262144` | 16.46x | s390x | `shake256 vs sha3` `1048576` |
| BLAKE2 | 363 | 564 | 49 | 976 | 37% | 1.04x | 1.02x | 0.74x | RISC-V | `blake2/blake2b512 vs rustcrypto` `1048576` | 1.32x | SPR | `blake2/keyed/blake2s256 vs rustcrypto` `0` |
| BLAKE3 | 241 | 199 | 40 | 480 | 50% | 1.40x | 1.05x | 0.73x | s390x | `blake3/xof vs blake3` `1024` | 9.58x | POWER10 | `blake3/derive-key vs blake3` `262144` |
| Ascon | 138 | 81 | 1 | 220 | 63% | 1.28x | 1.12x | 0.92x | RISC-V | `ascon-hash256 vs ascon-hash` `0` | 2.51x | Zen5 | `ascon-xof128 vs ascon-hash` `0` |
| XXH3 | 113 | 72 | 35 | 220 | 51% | 1.16x | 1.06x | 0.37x | Zen5 | `xxh3-64 vs xxhash-rust` `256` | 2.62x | macOS-A64 | `xxh3-64 vs xxhash-rust` `0` |
| RapidHash | 119 | 238 | 83 | 440 | 27% | 1.05x | 1.00x | 0.54x | s390x | `rapidhash-v3-64 vs rapidhash` `1` | 2.28x | Grav3 | `rapidhash-v3-128 vs rapidhash` `32` |
| Auth/KDF | 347 | 295 | 39 | 681 | 51% | 1.24x | 1.05x | 0.01x | SPR | `hmac-sha256/streaming vs rustcrypto` `64B` | 87.77x | SPR | `hkdf-sha256/expand vs rustcrypto` `32` |
| Password Hashing | 5 | 6 | 4 | 15 | 33% | 0.76x | 0.99x | 0.20x | macOS-A64 | `argon2d-small vs rustcrypto` `m=64_t=3_p=2` | 1.16x | macOS-A64 | `scrypt-small vs rustcrypto` `log_n=10_r=8_p=4` |
| AEAD | 797 | 167 | 136 | 1100 | 72% | 1.47x | 1.36x | 0.01x | RISC-V | `aes-256-gcm-siv/decrypt vs rustcrypto` `16384` | 31.33x | s390x | `aes-256-gcm-siv/decrypt vs rustcrypto` `0` |

## By Platform

| Platform | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Category | Worst Case | Best | Best Category | Best Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Zen4 | 466 | 156 | 22 | 644 | 72% | 1.61x | 1.16x | 0.38x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `64` | 129.35x | Checksums | `crc16-ibm vs crc` `262144` |
| Zen5 | 378 | 228 | 38 | 644 | 59% | 1.63x | 1.13x | 0.37x | XXH3 | `xxh3-64 vs xxhash-rust` `256` | 129.27x | Checksums | `crc16-ccitt vs crc` `262144` |
| SPR | 484 | 96 | 64 | 644 | 75% | 1.88x | 1.25x | 0.01x | SHA-2 | `sha256/streaming vs sha2` `64B` | 228.06x | Checksums | `crc16-ccitt vs crc` `16384` |
| ICL | 467 | 152 | 25 | 644 | 73% | 1.61x | 1.21x | 0.43x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `64` | 119.74x | Checksums | `crc16-ibm vs crc` `262144` |
| Grav3 | 324 | 276 | 44 | 644 | 50% | 1.63x | 1.05x | 0.77x | AEAD | `aegis-256/encrypt vs aegis-crate` `0` | 126.74x | Checksums | `crc16-ibm vs crc` `262144` |
| Grav4 | 323 | 297 | 24 | 644 | 50% | 1.65x | 1.05x | 0.77x | XXH3 | `xxh3-128 vs xxhash-rust` `256` | 104.99x | Checksums | `crc16-ibm vs crc` `262144` |
| s390x | 526 | 72 | 46 | 644 | 82% | 3.03x | 2.36x | 0.54x | RapidHash | `rapidhash-v3-64 vs rapidhash` `1` | 100.17x | Checksums | `crc16-ccitt vs crc` `65536` |
| POWER10 | 323 | 294 | 27 | 644 | 50% | 1.93x | 1.05x | 0.70x | RapidHash | `rapidhash-v3-128 vs rapidhash` `1` | 177.74x | Checksums | `crc16-ibm vs crc` `262144` |
| RISC-V | 256 | 271 | 117 | 644 | 40% | 0.76x | 1.02x | 0.01x | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` `16384` | 11.71x | Checksums | `crc64-nvme vs crc-fast` `0` |
| macOS-A64 | 336 | 321 | 60 | 717 | 47% | 1.49x | 1.05x | 0.20x | Password Hashing | `argon2d-small vs rustcrypto` `m=64_t=3_p=2` | 167.67x | Checksums | `crc16-ibm vs crc` `1048576` |

## Platform x Category

| Platform | Category | W | T | L | Total | Win % | Geomean | Worst | Worst Case |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zen4 | Checksums | 90 | 19 | 1 | 110 | 82% | 4.43x | 0.95x | `crc32 vs crc-fast` `1024` |
| Zen4 | SHA-2 | 38 | 21 | 0 | 59 | 64% | 1.12x | 1.00x | `sha256/streaming vs sha2` `64B` |
| Zen4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.29x | 2.23x | `sha3-224 vs sha3` `262144` |
| Zen4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.06x | 2.25x | `shake256 vs sha3` `262144` |
| Zen4 | BLAKE2 | 55 | 38 | 1 | 94 | 59% | 1.06x | 0.94x | `blake2/blake2s256 vs rustcrypto` `0` |
| Zen4 | BLAKE3 | 25 | 23 | 0 | 48 | 52% | 1.36x | 0.95x | `blake3/xof vs blake3` `0` |
| Zen4 | Ascon | 22 | 0 | 0 | 22 | 100% | 1.93x | 1.67x | `ascon-hash256 vs ascon-hash` `1` |
| Zen4 | XXH3 | 14 | 3 | 5 | 22 | 64% | 1.04x | 0.74x | `xxh3-64 vs xxhash-rust` `256` |
| Zen4 | RapidHash | 14 | 30 | 0 | 44 | 32% | 1.07x | 0.95x | `rapidhash-v3-64 vs rapidhash` `16384` |
| Zen4 | Auth/KDF | 56 | 7 | 4 | 67 | 84% | 1.18x | 0.92x | `pbkdf2-sha256/iters=1000 vs rustcrypto` `32` |
| Zen4 | AEAD | 84 | 15 | 11 | 110 | 76% | 1.23x | 0.38x | `aes-256-gcm/encrypt vs rustcrypto` `64` |
| Zen5 | Checksums | 91 | 18 | 1 | 110 | 83% | 4.79x | 0.93x | `crc32c vs crc-fast` `1024` |
| Zen5 | SHA-2 | 27 | 32 | 0 | 59 | 46% | 1.09x | 1.00x | `sha256/streaming vs sha2` `64B` |
| Zen5 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.17x | 2.11x | `sha3-256/streaming vs sha3` `64B` |
| Zen5 | SHAKE | 22 | 0 | 0 | 22 | 100% | 2.93x | 2.17x | `shake256 vs sha3` `1048576` |
| Zen5 | BLAKE2 | 7 | 87 | 0 | 94 | 7% | 1.00x | 0.95x | `blake2/blake2s256 vs rustcrypto` `32` |
| Zen5 | BLAKE3 | 22 | 21 | 5 | 48 | 46% | 1.34x | 0.84x | `blake3/streaming vs blake3` `64B` |
| Zen5 | Ascon | 22 | 0 | 0 | 22 | 100% | 2.17x | 1.84x | `ascon-hash256 vs ascon-hash` `1` |
| Zen5 | XXH3 | 13 | 4 | 5 | 22 | 59% | 1.18x | 0.37x | `xxh3-64 vs xxhash-rust` `256` |
| Zen5 | RapidHash | 12 | 24 | 8 | 44 | 27% | 1.05x | 0.92x | `rapidhash-v3-64 vs rapidhash` `16384` |
| Zen5 | Auth/KDF | 29 | 31 | 7 | 67 | 43% | 1.10x | 0.91x | `ed25519/verify vs dalek` `0` |
| Zen5 | AEAD | 87 | 11 | 12 | 110 | 79% | 1.36x | 0.42x | `aes-256-gcm/encrypt vs rustcrypto` `64` |
| SPR | Checksums | 96 | 8 | 6 | 110 | 87% | 4.91x | 0.82x | `crc32 vs crc-fast` `16384` |
| SPR | SHA-2 | 51 | 6 | 2 | 59 | 86% | 3.27x | 0.01x | `sha256/streaming vs sha2` `64B` |
| SPR | SHA-3 | 46 | 0 | 0 | 46 | 100% | 3.60x | 3.28x | `sha3-256 vs sha3` `1048576` |
| SPR | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.81x | 3.23x | `shake128 vs sha3` `65536` |
| SPR | BLAKE2 | 80 | 5 | 9 | 94 | 85% | 1.13x | 0.82x | `blake2/blake2b256 vs rustcrypto` `0` |
| SPR | BLAKE3 | 22 | 20 | 6 | 48 | 46% | 1.21x | 0.88x | `blake3/xof vs blake3` `0` |
| SPR | Ascon | 22 | 0 | 0 | 22 | 100% | 1.45x | 1.27x | `ascon-hash256 vs ascon-hash` `1` |
| SPR | XXH3 | 14 | 5 | 3 | 22 | 64% | 1.13x | 0.55x | `xxh3-64 vs xxhash-rust` `64` |
| SPR | RapidHash | 10 | 20 | 14 | 44 | 23% | 1.03x | 0.79x | `rapidhash-64 vs rapidhash` `0` |
| SPR | Auth/KDF | 39 | 18 | 10 | 67 | 58% | 1.21x | 0.01x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| SPR | AEAD | 82 | 14 | 14 | 110 | 75% | 1.25x | 0.44x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| ICL | Checksums | 87 | 22 | 1 | 110 | 79% | 3.98x | 0.51x | `crc32 vs crc32fast` `256` |
| ICL | SHA-2 | 35 | 22 | 2 | 59 | 59% | 1.12x | 0.88x | `sha256/streaming vs sha2` `4096B` |
| ICL | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.99x | 2.88x | `sha3-256/streaming vs sha3` `64B` |
| ICL | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.02x | 2.98x | `shake128 vs sha3` `1048576` |
| ICL | BLAKE2 | 92 | 2 | 0 | 94 | 98% | 1.16x | 1.05x | `blake2/blake2s256 vs rustcrypto` `0` |
| ICL | BLAKE3 | 14 | 30 | 4 | 48 | 29% | 1.16x | 0.83x | `blake3/keyed vs blake3` `256` |
| ICL | Ascon | 22 | 0 | 0 | 22 | 100% | 1.40x | 1.24x | `ascon-hash256 vs ascon-hash` `1` |
| ICL | XXH3 | 13 | 5 | 4 | 22 | 59% | 1.27x | 0.53x | `xxh3-64 vs xxhash-rust` `64` |
| ICL | RapidHash | 15 | 28 | 1 | 44 | 34% | 1.09x | 0.89x | `rapidhash-64 vs rapidhash` `32` |
| ICL | Auth/KDF | 36 | 28 | 3 | 67 | 54% | 1.10x | 0.86x | `hmac-sha256/streaming vs rustcrypto` `4096B` |
| ICL | AEAD | 85 | 15 | 10 | 110 | 77% | 1.24x | 0.43x | `aes-256-gcm/encrypt vs rustcrypto` `64` |
| Grav3 | Checksums | 78 | 25 | 7 | 110 | 71% | 3.69x | 0.92x | `crc32 vs crc-fast` `65536` |
| Grav3 | SHA-2 | 20 | 38 | 1 | 59 | 34% | 1.06x | 0.90x | `sha256/streaming vs sha2` `64B` |
| Grav3 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.81x | 1.73x | `sha3-256/streaming vs sha3` `64B` |
| Grav3 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.82x | 1.77x | `shake256 vs sha3` `1048576` |
| Grav3 | BLAKE2 | 22 | 72 | 0 | 94 | 23% | 1.03x | 0.99x | `blake2/streaming/blake2b256 vs rustcrypto` `64B` |
| Grav3 | BLAKE3 | 21 | 22 | 5 | 48 | 44% | 1.40x | 0.91x | `blake3/keyed vs blake3` `1` |
| Grav3 | Ascon | 4 | 18 | 0 | 22 | 18% | 1.03x | 0.99x | `ascon-xof128 vs ascon-hash` `1048576` |
| Grav3 | XXH3 | 6 | 12 | 4 | 22 | 27% | 1.07x | 0.86x | `xxh3-128 vs xxhash-rust` `256` |
| Grav3 | RapidHash | 11 | 19 | 14 | 44 | 25% | 1.05x | 0.90x | `rapidhash-v3-64 vs rapidhash` `256` |
| Grav3 | Auth/KDF | 29 | 31 | 7 | 67 | 43% | 1.08x | 0.89x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| Grav3 | AEAD | 65 | 39 | 6 | 110 | 59% | 2.48x | 0.77x | `aegis-256/encrypt vs aegis-crate` `0` |
| Grav4 | Checksums | 79 | 28 | 3 | 110 | 72% | 3.68x | 0.82x | `crc32c vs crc-fast` `256` |
| Grav4 | SHA-2 | 20 | 38 | 1 | 59 | 34% | 1.05x | 0.90x | `sha256/streaming vs sha2` `64B` |
| Grav4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.97x | 1.89x | `sha3-256/streaming vs sha3` `64B` |
| Grav4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.99x | 1.93x | `shake256 vs sha3` `262144` |
| Grav4 | BLAKE2 | 18 | 76 | 0 | 94 | 19% | 1.02x | 0.98x | `blake2/streaming/blake2b256 vs rustcrypto` `64B` |
| Grav4 | BLAKE3 | 22 | 25 | 1 | 48 | 46% | 1.47x | 0.93x | `blake3/keyed vs blake3` `1` |
| Grav4 | Ascon | 12 | 10 | 0 | 22 | 55% | 1.08x | 1.01x | `ascon-hash256 vs ascon-hash` `1` |
| Grav4 | XXH3 | 5 | 13 | 4 | 22 | 23% | 1.03x | 0.77x | `xxh3-128 vs xxhash-rust` `256` |
| Grav4 | RapidHash | 13 | 28 | 3 | 44 | 30% | 1.07x | 0.89x | `rapidhash-v3-128 vs rapidhash` `256` |
| Grav4 | Auth/KDF | 20 | 45 | 2 | 67 | 30% | 1.07x | 0.88x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| Grav4 | AEAD | 66 | 34 | 10 | 110 | 60% | 2.45x | 0.77x | `aegis-256/encrypt vs aegis-crate` `0` |
| s390x | Checksums | 98 | 2 | 10 | 110 | 89% | 7.25x | 0.63x | `crc24-openpgp vs crc` `64` |
| s390x | SHA-2 | 59 | 0 | 0 | 59 | 100% | 5.26x | 1.90x | `sha224 vs sha2` `0` |
| s390x | SHA-3 | 46 | 0 | 0 | 46 | 100% | 4.69x | 1.09x | `sha3-256/streaming vs sha3` `64B` |
| s390x | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.93x | 1.08x | `shake128 vs sha3` `64` |
| s390x | BLAKE2 | 56 | 38 | 0 | 94 | 60% | 1.11x | 0.96x | `blake2/streaming/blake2b256 vs rustcrypto` `64B` |
| s390x | BLAKE3 | 39 | 5 | 4 | 48 | 81% | 1.91x | 0.73x | `blake3/xof vs blake3` `1024` |
| s390x | Ascon | 4 | 18 | 0 | 22 | 18% | 1.04x | 1.00x | `ascon-xof128 vs ascon-hash` `262144` |
| s390x | XXH3 | 17 | 3 | 2 | 22 | 77% | 1.70x | 0.68x | `xxh3-64 vs xxhash-rust` `32` |
| s390x | RapidHash | 11 | 3 | 30 | 44 | 25% | 0.97x | 0.54x | `rapidhash-v3-64 vs rapidhash` `1` |
| s390x | Auth/KDF | 64 | 3 | 0 | 67 | 96% | 3.48x | 1.00x | `ed25519/verify vs dalek` `1024` |
| s390x | AEAD | 110 | 0 | 0 | 110 | 100% | 4.30x | 1.16x | `xchacha20-poly1305/encrypt vs rustcrypto` `1` |
| POWER10 | Checksums | 104 | 3 | 3 | 110 | 95% | 10.87x | 0.78x | `crc64-xz vs crc64fast` `64` |
| POWER10 | SHA-2 | 6 | 53 | 0 | 59 | 10% | 1.01x | 0.96x | `sha256 vs sha2` `4096` |
| POWER10 | SHA-3 | 17 | 29 | 0 | 46 | 37% | 1.04x | 0.96x | `sha3-256/streaming vs sha3` `64B` |
| POWER10 | SHAKE | 13 | 9 | 0 | 22 | 59% | 1.39x | 1.00x | `shake256 vs sha3` `65536` |
| POWER10 | BLAKE2 | 7 | 79 | 8 | 94 | 7% | 0.99x | 0.92x | `blake2/keyed/blake2b256 vs rustcrypto` `32` |
| POWER10 | BLAKE3 | 35 | 11 | 2 | 48 | 73% | 2.01x | 0.94x | `blake3 vs blake3` `1` |
| POWER10 | Ascon | 5 | 17 | 0 | 22 | 23% | 1.06x | 1.01x | `ascon-hash256 vs ascon-hash` `65536` |
| POWER10 | XXH3 | 19 | 3 | 0 | 22 | 86% | 1.29x | 0.98x | `xxh3-64 vs xxhash-rust` `1` |
| POWER10 | RapidHash | 11 | 26 | 7 | 44 | 25% | 1.06x | 0.70x | `rapidhash-v3-128 vs rapidhash` `1` |
| POWER10 | Auth/KDF | 11 | 56 | 0 | 67 | 16% | 1.05x | 0.97x | `hmac-sha256 vs rustcrypto` `4096` |
| POWER10 | AEAD | 95 | 8 | 7 | 110 | 86% | 2.62x | 0.93x | `aegis-256/encrypt vs aegis-crate` `262144` |
| RISC-V | Checksums | 71 | 18 | 21 | 110 | 65% | 1.47x | 0.56x | `crc32c vs crc32c` `32` |
| RISC-V | SHA-2 | 16 | 43 | 0 | 59 | 27% | 1.04x | 0.96x | `sha256/streaming vs sha2` `4096B` |
| RISC-V | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.17x | 1.08x | `sha3-256/streaming vs sha3` `4096B` |
| RISC-V | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.60x | 1.08x | `shake256 vs sha3` `16384` |
| RISC-V | BLAKE2 | 3 | 79 | 12 | 94 | 3% | 0.98x | 0.74x | `blake2/blake2b512 vs rustcrypto` `1048576` |
| RISC-V | BLAKE3 | 24 | 16 | 8 | 48 | 50% | 1.14x | 0.76x | `blake3/streaming vs blake3` `65536B` |
| RISC-V | Ascon | 8 | 13 | 1 | 22 | 36% | 1.06x | 0.92x | `ascon-hash256 vs ascon-hash` `0` |
| RISC-V | XXH3 | 7 | 14 | 1 | 22 | 32% | 1.04x | 0.74x | `xxh3-64 vs xxhash-rust` `0` |
| RISC-V | RapidHash | 10 | 29 | 5 | 44 | 23% | 1.06x | 0.83x | `rapidhash-64 vs rapidhash` `262144` |
| RISC-V | Auth/KDF | 30 | 34 | 3 | 67 | 45% | 1.08x | 0.93x | `hmac-sha384 vs rustcrypto` `4096` |
| RISC-V | AEAD | 19 | 25 | 66 | 110 | 17% | 0.10x | 0.01x | `aes-256-gcm-siv/decrypt vs rustcrypto` `16384` |
| macOS-A64 | Checksums | 91 | 15 | 4 | 110 | 83% | 4.18x | 0.35x | `crc32 vs crc32fast` `64` |
| macOS-A64 | SHA-2 | 12 | 43 | 4 | 59 | 20% | 1.03x | 0.82x | `sha224 vs sha2` `32` |
| macOS-A64 | SHA-3 | 5 | 35 | 6 | 46 | 11% | 1.02x | 0.92x | `sha3-384 vs sha3` `0` |
| macOS-A64 | SHAKE | 12 | 14 | 7 | 33 | 36% | 1.08x | 0.90x | `shake128 vs sha3` `262144` |
| macOS-A64 | BLAKE2 | 23 | 88 | 19 | 130 | 18% | 1.01x | 0.80x | `blake2/host-overhead/blake2s256 vs rustcrypto` `1` |
| macOS-A64 | BLAKE3 | 17 | 26 | 5 | 48 | 35% | 1.23x | 0.90x | `blake3/keyed vs blake3` `65536` |
| macOS-A64 | Ascon | 17 | 5 | 0 | 22 | 77% | 1.10x | 1.01x | `ascon-hash256 vs ascon-hash` `0` |
| macOS-A64 | XXH3 | 5 | 10 | 7 | 22 | 23% | 1.05x | 0.75x | `xxh3-64 vs xxhash-rust` `32` |
| macOS-A64 | RapidHash | 12 | 31 | 1 | 44 | 27% | 1.09x | 0.81x | `rapidhash-64 vs rapidhash` `1` |
| macOS-A64 | Auth/KDF | 33 | 42 | 3 | 78 | 42% | 1.07x | 0.81x | `hkdf-sha256/expand vs rustcrypto` `32` |
| macOS-A64 | Password Hashing | 5 | 6 | 4 | 15 | 33% | 0.76x | 0.20x | `argon2d-small vs rustcrypto` `m=64_t=3_p=2` |
| macOS-A64 | AEAD | 104 | 6 | 0 | 110 | 95% | 2.58x | 0.99x | `aegis-256/encrypt vs aegis-crate` `0` |

## By Primitive Group

| Category | Group | W | T | L | Total | Win % | Geomean | Median | Worst | Platform | Case | Best | Platform | Case |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Checksums | `crc16-ccitt vs crc` | 109 | 0 | 1 | 110 | 99% | 25.09x | 52.09x | 0.70x | s390x | `1` | 228.06x | SPR | `16384` |
| Checksums | `crc16-ibm vs crc` | 109 | 0 | 1 | 110 | 99% | 25.61x | 49.04x | 0.79x | s390x | `1` | 224.60x | SPR | `262144` |
| Checksums | `crc24-openpgp vs crc` | 108 | 1 | 1 | 110 | 98% | 14.88x | 23.67x | 0.63x | s390x | `64` | 127.05x | Zen5 | `262144` |
| Checksums | `crc32 vs crc-fast` | 66 | 36 | 8 | 110 | 60% | 1.99x | 1.16x | 0.37x | macOS-A64 | `64` | 27.36x | POWER10 | `262144` |
| Checksums | `crc32 vs crc32fast` | 93 | 9 | 8 | 110 | 85% | 2.49x | 2.04x | 0.35x | macOS-A64 | `64` | 24.04x | POWER10 | `262144` |
| Checksums | `crc32c vs crc-fast` | 61 | 39 | 10 | 110 | 55% | 1.72x | 1.09x | 0.82x | Grav4 | `256` | 27.63x | POWER10 | `1048576` |
| Checksums | `crc32c vs crc32c` | 102 | 4 | 4 | 110 | 93% | 2.79x | 2.25x | 0.56x | RISC-V | `32` | 41.06x | POWER10 | `1048576` |
| Checksums | `crc64-nvme vs crc-fast` | 68 | 37 | 5 | 110 | 62% | 2.15x | 1.15x | 0.61x | RISC-V | `1048576` | 34.36x | POWER10 | `1048576` |
| Checksums | `crc64-nvme vs crc64fast-nvme` | 81 | 17 | 12 | 110 | 74% | 2.34x | 2.00x | 0.69x | RISC-V | `1048576` | 29.54x | POWER10 | `1048576` |
| Checksums | `crc64-xz vs crc64fast` | 88 | 15 | 7 | 110 | 80% | 2.39x | 2.02x | 0.76x | s390x | `1` | 29.21x | POWER10 | `262144` |
| SHA-2 | `sha256/streaming vs sha2` | 2 | 12 | 6 | 20 | 10% | 0.72x | 0.99x | 0.01x | SPR | `64B` | 9.61x | s390x | `4096B` |
| SHA-2 | `sha512/streaming vs sha2` | 7 | 13 | 0 | 20 | 35% | 1.21x | 1.01x | 0.96x | Grav3 | `64B` | 10.94x | s390x | `4096B` |
| SHA-3 | `sha3-224 vs sha3` | 97 | 13 | 0 | 110 | 88% | 2.02x | 2.01x | 1.01x | macOS-A64 | `1048576` | 15.63x | s390x | `262144` |
| SHA-3 | `sha3-256 vs sha3` | 92 | 18 | 0 | 110 | 84% | 1.99x | 1.96x | 0.96x | macOS-A64 | `0` | 17.13x | s390x | `1048576` |
| SHA-3 | `sha3-256/streaming vs sha3` | 16 | 2 | 2 | 20 | 80% | 1.89x | 1.91x | 0.92x | macOS-A64 | `4096B` | 11.94x | s390x | `4096B` |
| SHA-3 | `sha3-384 vs sha3` | 92 | 15 | 3 | 110 | 84% | 2.03x | 1.95x | 0.92x | macOS-A64 | `0` | 20.71x | s390x | `262144` |
| SHA-3 | `sha3-512 vs sha3` | 93 | 16 | 1 | 110 | 85% | 2.05x | 1.95x | 0.94x | macOS-A64 | `0` | 24.84x | s390x | `1048576` |
| SHAKE | `cshake256 vs tiny-keccak` | 11 | 0 | 0 | 11 | 100% | 1.27x | 1.30x | 1.18x | macOS-A64 | `64` | 1.35x | macOS-A64 | `262144` |
| SHAKE | `shake128 vs sha3` | 96 | 7 | 7 | 110 | 87% | 2.34x | 2.09x | 0.90x | macOS-A64 | `262144` | 14.26x | s390x | `1048576` |
| SHAKE | `shake256 vs sha3` | 94 | 16 | 0 | 110 | 85% | 2.38x | 2.09x | 0.98x | macOS-A64 | `1` | 16.46x | s390x | `1048576` |
| BLAKE2 | `blake2/blake2b256 vs rustcrypto` | 38 | 65 | 7 | 110 | 35% | 1.02x | 1.02x | 0.74x | RISC-V | `1048576` | 1.20x | macOS-A64 | `64` |
| BLAKE2 | `blake2/blake2b512 vs rustcrypto` | 46 | 62 | 2 | 110 | 42% | 1.04x | 1.03x | 0.74x | RISC-V | `1048576` | 1.17x | ICL | `32` |
| BLAKE2 | `blake2/blake2s128 vs rustcrypto` | 45 | 64 | 1 | 110 | 41% | 1.06x | 1.03x | 0.94x | RISC-V | `262144` | 1.28x | SPR | `1024` |
| BLAKE2 | `blake2/blake2s256 vs rustcrypto` | 35 | 70 | 5 | 110 | 32% | 1.05x | 1.01x | 0.85x | SPR | `1` | 1.30x | SPR | `4096` |
| BLAKE2 | `blake2/host-keyed-overhead/blake2b256 vs rustcrypto` | 0 | 6 | 0 | 6 | 0% | 0.99x | 0.98x | 0.97x | macOS-A64 | `1` | 1.04x | macOS-A64 | `0` |
| BLAKE2 | `blake2/host-keyed-overhead/blake2s256 vs rustcrypto` | 1 | 5 | 0 | 6 | 17% | 1.05x | 1.03x | 1.03x | macOS-A64 | `128` | 1.11x | macOS-A64 | `0` |
| BLAKE2 | `blake2/host-overhead/blake2b256 vs rustcrypto` | 0 | 2 | 4 | 6 | 0% | 0.91x | 0.94x | 0.84x | macOS-A64 | `1` | 0.96x | macOS-A64 | `64` |
| BLAKE2 | `blake2/host-overhead/blake2s256 vs rustcrypto` | 0 | 2 | 4 | 6 | 0% | 0.91x | 0.92x | 0.80x | macOS-A64 | `1` | 1.00x | macOS-A64 | `64` |
| BLAKE2 | `blake2/host-stream-overhead/blake2b256 vs rustcrypto` | 0 | 1 | 5 | 6 | 0% | 0.91x | 0.90x | 0.89x | macOS-A64 | `32` | 0.96x | macOS-A64 | `128` |
| BLAKE2 | `blake2/host-stream-overhead/blake2s256 vs rustcrypto` | 0 | 0 | 6 | 6 | 0% | 0.84x | 0.84x | 0.82x | macOS-A64 | `0` | 0.88x | macOS-A64 | `128` |
| BLAKE2 | `blake2/keyed/blake2b256 vs rustcrypto` | 46 | 61 | 3 | 110 | 42% | 1.04x | 1.03x | 0.92x | POWER10 | `32` | 1.22x | ICL | `0` |
| BLAKE2 | `blake2/keyed/blake2b512 vs rustcrypto` | 47 | 60 | 3 | 110 | 43% | 1.04x | 1.02x | 0.92x | POWER10 | `32` | 1.22x | ICL | `0` |
| BLAKE2 | `blake2/keyed/blake2s128 vs rustcrypto` | 43 | 63 | 4 | 110 | 39% | 1.07x | 1.03x | 0.92x | RISC-V | `1` | 1.29x | s390x | `0` |
| BLAKE2 | `blake2/keyed/blake2s256 vs rustcrypto` | 45 | 62 | 3 | 110 | 41% | 1.07x | 1.04x | 0.95x | RISC-V | `1048576` | 1.32x | SPR | `0` |
| BLAKE2 | `blake2/streaming/blake2b256 vs rustcrypto` | 8 | 20 | 2 | 30 | 27% | 1.02x | 1.01x | 0.95x | RISC-V | `64B` | 1.14x | ICL | `64B` |
| BLAKE2 | `blake2/streaming/blake2s256 vs rustcrypto` | 9 | 21 | 0 | 30 | 30% | 1.07x | 1.02x | 0.96x | POWER10 | `64B` | 1.25x | s390x | `65536B` |
| BLAKE3 | `blake3 vs blake3` | 50 | 51 | 9 | 110 | 45% | 1.32x | 1.04x | 0.76x | s390x | `1024` | 8.44x | s390x | `262144` |
| BLAKE3 | `blake3/derive-key vs blake3` | 84 | 22 | 4 | 110 | 76% | 1.80x | 1.91x | 0.83x | s390x | `1024` | 9.58x | POWER10 | `262144` |
| BLAKE3 | `blake3/keyed vs blake3` | 46 | 55 | 9 | 110 | 42% | 1.32x | 1.02x | 0.77x | s390x | `1024` | 8.55x | s390x | `262144` |
| BLAKE3 | `blake3/streaming vs blake3` | 7 | 26 | 7 | 40 | 18% | 1.12x | 1.00x | 0.76x | RISC-V | `65536B` | 3.10x | POWER10 | `65536B` |
| BLAKE3 | `blake3/xof vs blake3` | 54 | 45 | 11 | 110 | 49% | 1.31x | 1.05x | 0.73x | s390x | `1024` | 8.70x | s390x | `262144` |
| Ascon | `ascon-hash256 vs ascon-hash` | 57 | 52 | 1 | 110 | 52% | 1.24x | 1.05x | 0.92x | RISC-V | `0` | 2.18x | Zen5 | `65536` |
| Ascon | `ascon-xof128 vs ascon-hash` | 81 | 29 | 0 | 110 | 74% | 1.33x | 1.26x | 0.99x | Grav3 | `1048576` | 2.51x | Zen5 | `0` |
| XXH3 | `xxh3-128 vs xxhash-rust` | 56 | 39 | 15 | 110 | 51% | 1.17x | 1.05x | 0.68x | SPR | `256` | 2.44x | s390x | `65536` |
| XXH3 | `xxh3-64 vs xxhash-rust` | 57 | 33 | 20 | 110 | 52% | 1.16x | 1.06x | 0.37x | Zen5 | `256` | 2.62x | macOS-A64 | `0` |
| RapidHash | `rapidhash-128 vs rapidhash` | 36 | 67 | 7 | 110 | 33% | 1.09x | 1.02x | 0.89x | s390x | `262144` | 2.16x | POWER10 | `0` |
| RapidHash | `rapidhash-64 vs rapidhash` | 20 | 72 | 18 | 110 | 18% | 1.02x | 1.00x | 0.68x | s390x | `32` | 1.61x | POWER10 | `32` |
| RapidHash | `rapidhash-v3-128 vs rapidhash` | 36 | 46 | 28 | 110 | 33% | 1.07x | 1.00x | 0.70x | POWER10 | `1` | 2.28x | Grav3 | `32` |
| RapidHash | `rapidhash-v3-64 vs rapidhash` | 27 | 53 | 30 | 110 | 25% | 1.04x | 1.00x | 0.54x | s390x | `1` | 2.00x | Zen4 | `32` |
| Auth/KDF | `ed25519/keypair-from-secret vs dalek` | 10 | 0 | 0 | 10 | 100% | 1.62x | 1.53x | 1.27x | Grav4 | `rscrypto` | 2.46x | Zen4 | `rscrypto` |
| Auth/KDF | `ed25519/public-key-from-secret vs dalek` | 10 | 0 | 0 | 10 | 100% | 1.65x | 1.51x | 1.29x | Grav4 | `rscrypto` | 2.55x | ICL | `rscrypto` |
| Auth/KDF | `ed25519/sign vs dalek` | 40 | 0 | 0 | 40 | 100% | 1.59x | 1.52x | 1.12x | ICL | `16384` | 3.73x | s390x | `16384` |
| Auth/KDF | `ed25519/verify vs dalek` | 23 | 14 | 3 | 40 | 57% | 1.09x | 1.07x | 0.91x | Zen5 | `0` | 1.41x | s390x | `16384` |
| Auth/KDF | `hkdf-sha256/expand vs rustcrypto` | 28 | 11 | 1 | 40 | 70% | 1.70x | 1.08x | 0.81x | macOS-A64 | `32` | 87.77x | SPR | `32` |
| Auth/KDF | `hkdf-sha384/expand vs rustcrypto` | 19 | 21 | 0 | 40 | 48% | 1.18x | 1.05x | 0.99x | RISC-V | `48` | 2.99x | s390x | `96` |
| Auth/KDF | `hmac-sha256 vs rustcrypto` | 42 | 61 | 7 | 110 | 38% | 1.25x | 1.01x | 0.71x | SPR | `1024` | 9.40x | s390x | `1048576` |
| Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | 2 | 12 | 6 | 20 | 10% | 0.72x | 1.00x | 0.01x | SPR | `64B` | 9.54x | s390x | `4096B` |
| Auth/KDF | `hmac-sha384 vs rustcrypto` | 59 | 50 | 1 | 110 | 54% | 1.25x | 1.05x | 0.93x | RISC-V | `4096` | 10.60x | s390x | `1048576` |
| Auth/KDF | `hmac-sha512 vs rustcrypto` | 48 | 58 | 4 | 110 | 44% | 1.24x | 1.04x | 0.92x | macOS-A64 | `256` | 10.58x | s390x | `1048576` |
| Auth/KDF | `kmac256 vs tiny-keccak` | 11 | 0 | 0 | 11 | 100% | 1.23x | 1.26x | 1.12x | macOS-A64 | `0` | 1.35x | macOS-A64 | `262144` |
| Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | 6 | 9 | 5 | 20 | 30% | 1.04x | 1.00x | 0.68x | SPR | `64` | 1.99x | s390x | `64` |
| Auth/KDF | `pbkdf2-sha256/iters=100 vs rustcrypto` | 2 | 14 | 4 | 20 | 10% | 1.06x | 1.00x | 0.92x | Zen4 | `64` | 2.22x | s390x | `32` |
| Auth/KDF | `pbkdf2-sha256/iters=1000 vs rustcrypto` | 2 | 14 | 4 | 20 | 10% | 1.06x | 1.00x | 0.92x | Zen4 | `32` | 2.13x | s390x | `64` |
| Auth/KDF | `pbkdf2-sha512/iters=1 vs rustcrypto` | 10 | 10 | 0 | 20 | 50% | 1.16x | 1.05x | 0.96x | RISC-V | `64` | 2.64x | s390x | `128` |
| Auth/KDF | `pbkdf2-sha512/iters=100 vs rustcrypto` | 9 | 9 | 2 | 20 | 45% | 1.14x | 1.03x | 0.95x | Grav3 | `128` | 2.81x | s390x | `64` |
| Auth/KDF | `pbkdf2-sha512/iters=1000 vs rustcrypto` | 8 | 10 | 2 | 20 | 40% | 1.14x | 1.03x | 0.95x | Grav3 | `64` | 2.85x | s390x | `64` |
| Auth/KDF | `x25519/diffie-hellman vs dalek` | 8 | 2 | 0 | 10 | 80% | 1.16x | 1.15x | 1.04x | SPR | `rscrypto` | 1.47x | RISC-V | `rscrypto` |
| Auth/KDF | `x25519/public-key-from-secret vs dalek` | 10 | 0 | 0 | 10 | 100% | 1.56x | 1.52x | 1.25x | Grav4 | `rscrypto` | 2.07x | Zen4 | `rscrypto` |
| Password Hashing | `argon2d-small vs rustcrypto` | 0 | 2 | 1 | 3 | 0% | 0.58x | 0.97x | 0.20x | macOS-A64 | `m=64_t=3_p=2` | 0.99x | macOS-A64 | `m=32_t=2_p=1` |
| Password Hashing | `argon2i-small vs rustcrypto` | 0 | 1 | 2 | 3 | 0% | 0.60x | 0.95x | 0.23x | macOS-A64 | `m=64_t=3_p=2` | 0.97x | macOS-A64 | `m=32_t=2_p=1` |
| Password Hashing | `argon2id-owasp vs rustcrypto` | 0 | 1 | 0 | 1 | 0% | 1.00x | 1.00x | 1.00x | macOS-A64 | `m=19MiB_t=2_p=1` | 1.00x | macOS-A64 | `m=19MiB_t=2_p=1` |
| Password Hashing | `argon2id-small vs rustcrypto` | 0 | 2 | 1 | 3 | 0% | 0.58x | 0.96x | 0.20x | macOS-A64 | `m=64_t=3_p=2` | 0.99x | macOS-A64 | `m=32_t=2_p=1` |
| Password Hashing | `scrypt-owasp vs rustcrypto` | 1 | 0 | 0 | 1 | 100% | 1.11x | 1.11x | 1.11x | macOS-A64 | `log_n=17_r=8_p=1` | 1.11x | macOS-A64 | `log_n=17_r=8_p=1` |
| Password Hashing | `scrypt-small vs rustcrypto` | 4 | 0 | 0 | 4 | 100% | 1.15x | 1.15x | 1.13x | macOS-A64 | `log_n=10_r=8_p=1` | 1.16x | macOS-A64 | `log_n=10_r=8_p=4` |
| AEAD | `aegis-256/decrypt vs aegis-crate` | 64 | 31 | 15 | 110 | 58% | 1.10x | 1.10x | 0.08x | RISC-V | `0` | 8.27x | s390x | `1` |
| AEAD | `aegis-256/encrypt vs aegis-crate` | 33 | 44 | 33 | 110 | 30% | 0.96x | 1.01x | 0.08x | RISC-V | `16384` | 8.08x | s390x | `1` |
| AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | 99 | 0 | 11 | 110 | 90% | 2.22x | 3.48x | 0.01x | RISC-V | `16384` | 31.33x | s390x | `0` |
| AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | 99 | 0 | 11 | 110 | 90% | 2.80x | 3.85x | 0.01x | RISC-V | `262144` | 30.88x | s390x | `0` |
| AEAD | `aes-256-gcm/decrypt vs rustcrypto` | 85 | 2 | 23 | 110 | 77% | 1.70x | 2.79x | 0.01x | RISC-V | `16384` | 21.76x | macOS-A64 | `32` |
| AEAD | `aes-256-gcm/encrypt vs rustcrypto` | 82 | 1 | 27 | 110 | 75% | 1.56x | 2.81x | 0.01x | RISC-V | `65536` | 11.84x | macOS-A64 | `32` |
| AEAD | `chacha20-poly1305/decrypt vs rustcrypto` | 90 | 16 | 4 | 110 | 82% | 1.32x | 1.28x | 0.62x | Zen5 | `256` | 2.48x | s390x | `16384` |
| AEAD | `chacha20-poly1305/encrypt vs rustcrypto` | 80 | 26 | 4 | 110 | 73% | 1.29x | 1.26x | 0.61x | Zen5 | `256` | 2.45x | s390x | `1048576` |
| AEAD | `xchacha20-poly1305/decrypt vs rustcrypto` | 87 | 19 | 4 | 110 | 79% | 1.26x | 1.21x | 0.65x | Zen5 | `256` | 2.52x | s390x | `1048576` |
| AEAD | `xchacha20-poly1305/encrypt vs rustcrypto` | 78 | 28 | 4 | 110 | 71% | 1.24x | 1.17x | 0.66x | Zen5 | `256` | 2.52x | s390x | `65536` |
| Other | `sha224 vs sha2` | 56 | 53 | 1 | 110 | 51% | 1.76x | 1.06x | 0.82x | macOS-A64 | `32` | 123.02x | SPR | `262144` |
| Other | `sha256 vs sha2` | 51 | 59 | 0 | 110 | 46% | 1.76x | 1.04x | 0.96x | POWER10 | `4096` | 118.99x | SPR | `262144` |
| Other | `sha384 vs sha2` | 58 | 51 | 1 | 110 | 53% | 1.27x | 1.05x | 0.92x | macOS-A64 | `256` | 11.06x | s390x | `262144` |
| Other | `sha512 vs sha2` | 57 | 52 | 1 | 110 | 52% | 1.26x | 1.05x | 0.93x | macOS-A64 | `256` | 10.80x | s390x | `1048576` |
| Other | `sha512-256 vs sha2` | 53 | 56 | 1 | 110 | 48% | 1.25x | 1.05x | 0.92x | macOS-A64 | `256` | 10.69x | s390x | `262144` |

## 1 MiB End-to-End Throughput

These rows use the `rscrypto` midpoint for the 1 MiB case where that benchmark has a byte-size input. Throughput is computed from the parsed Criterion midpoint.

| Primitive | Zen4 | Zen5 | SPR | ICL | Grav3 | Grav4 | s390x | POWER10 | RISC-V | macOS-A64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| crc32c vs crc32c | 53.31 GiB/s | 66.18 GiB/s | 92.32 GiB/s | 47.55 GiB/s | 37.55 GiB/s | 44.87 GiB/s | 33.69 GiB/s | 57.49 GiB/s | 619 MiB/s | 75.35 GiB/s |
| sha256 vs sha2 | 1.69 GiB/s | 2.06 GiB/s | 1.66 GiB/s | 1.21 GiB/s | 1.33 GiB/s | 1.43 GiB/s | 1.78 GiB/s | 317 MiB/s | 56 MiB/s | 2.05 GiB/s |
| sha512 vs sha2 | 639 MiB/s | 1.02 GiB/s | 645 MiB/s | 520 MiB/s | 923 MiB/s | 997 MiB/s | 2.79 GiB/s | 506 MiB/s | 76 MiB/s | 1.32 GiB/s |
| sha3-256 vs sha3 | 449 MiB/s | 625 MiB/s | 448 MiB/s | 388 MiB/s | 384 MiB/s | 453 MiB/s | 3.59 GiB/s | 452 MiB/s | 82 MiB/s | 766 MiB/s |
| shake256 vs sha3 | 451 MiB/s | 627 MiB/s | 512 MiB/s | 387 MiB/s | 385 MiB/s | 451 MiB/s | 3.60 GiB/s | 455 MiB/s | 82 MiB/s | 767 MiB/s |
| blake2/blake2b512 vs rustcrypto | 878 MiB/s | 766 MiB/s | 0.99 GiB/s | 1.00 GiB/s | 767 MiB/s | 860 MiB/s | 552 MiB/s | 685 MiB/s | 110 MiB/s | 1.01 GiB/s |
| blake2/blake2s256 vs rustcrypto | 601 MiB/s | 497 MiB/s | 726 MiB/s | 745 MiB/s | 461 MiB/s | 518 MiB/s | 385 MiB/s | 484 MiB/s | 78 MiB/s | 615 MiB/s |
| blake3 vs blake3 | 20.14 GiB/s | 35.64 GiB/s | 20.65 GiB/s | 12.03 GiB/s | 4.37 GiB/s | 5.51 GiB/s | 2.26 GiB/s | 2.51 GiB/s | 348 MiB/s | 6.00 GiB/s |
| ascon-hash256 vs ascon-hash | 220 MiB/s | 296 MiB/s | 166 MiB/s | 146 MiB/s | 190 MiB/s | 216 MiB/s | 84 MiB/s | 123 MiB/s | 36 MiB/s | 164 MiB/s |
| xxh3-64 vs xxhash-rust | 61.14 GiB/s | 117.31 GiB/s | 65.70 GiB/s | 58.54 GiB/s | 18.04 GiB/s | 20.43 GiB/s | 10.88 GiB/s | 19.68 GiB/s | 858 MiB/s | 32.36 GiB/s |
| rapidhash-64 vs rapidhash | 27.52 GiB/s | 50.80 GiB/s | 39.17 GiB/s | 27.75 GiB/s | 20.29 GiB/s | 29.70 GiB/s | 9.33 GiB/s | 21.46 GiB/s | 352 MiB/s | 44.29 GiB/s |
| hmac-sha256 vs rustcrypto | 1.69 GiB/s | 2.06 GiB/s | 1.52 GiB/s | 1.21 GiB/s | 1.33 GiB/s | 1.43 GiB/s | 1.78 GiB/s | 316 MiB/s | 56 MiB/s | 2.05 GiB/s |
| aes-256-gcm/encrypt vs rustcrypto | 2.49 GiB/s | 3.18 GiB/s | 2.33 GiB/s | 2.11 GiB/s | 760 MiB/s | 823 MiB/s | 234 MiB/s | 387 MiB/s | 0 MiB/s | 543 MiB/s |
| aes-256-gcm-siv/encrypt vs rustcrypto | 2.56 GiB/s | 3.29 GiB/s | 2.54 GiB/s | 2.23 GiB/s | 1.56 GiB/s | 1.86 GiB/s | 329 MiB/s | 1.37 GiB/s | 0 MiB/s | 2.77 GiB/s |
| chacha20-poly1305/encrypt vs rustcrypto | 2.18 GiB/s | 3.07 GiB/s | 1.91 GiB/s | 1.85 GiB/s | 354 MiB/s | 397 MiB/s | 486 MiB/s | 470 MiB/s | 61 MiB/s | 548 MiB/s |
| xchacha20-poly1305/encrypt vs rustcrypto | 2.17 GiB/s | 3.07 GiB/s | 2.01 GiB/s | 1.81 GiB/s | 354 MiB/s | 398 MiB/s | 495 MiB/s | 472 MiB/s | 60 MiB/s | 545 MiB/s |
| aegis-256/encrypt vs aegis-crate | 8.22 GiB/s | 8.97 GiB/s | 6.17 GiB/s | 5.75 GiB/s | 4.83 GiB/s | 5.22 GiB/s | 296 MiB/s | 4.63 GiB/s | 0 MiB/s | 7.17 GiB/s |

## 64-Byte Latency

These rows use the `rscrypto` midpoint for the 64-byte case. This captures short-message call overhead and dispatch cost better than the 1 MiB throughput table.

| Primitive | Zen4 | Zen5 | SPR | ICL | Grav3 | Grav4 | s390x | POWER10 | RISC-V | macOS-A64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| crc32c vs crc32c | 3.17 ns | 2.16 ns | 2.44 ns | 3.27 ns | 4.23 ns | 3.69 ns | 36.52 ns | 36.36 ns | 169.37 ns | 3.11 ns |
| sha256 vs sha2 | 73.56 ns | 62.94 ns | 79.75 ns | 101.22 ns | 89.76 ns | 73.79 ns | 342.01 ns | 411.03 ns | 2.35 us | 49.26 ns |
| sha512 vs sha2 | 191.68 ns | 126.91 ns | 205.32 ns | 225.02 ns | 143.30 ns | 136.29 ns | 187.78 ns | 270.23 ns | 1.77 us | 67.08 ns |
| sha3-256 vs sha3 | 292.17 ns | 213.47 ns | 256.86 ns | 348.16 ns | 342.01 ns | 291.05 ns | 538.20 ns | 303.09 ns | 1.56 us | 159.95 ns |
| shake256 vs sha3 | 291.38 ns | 211.35 ns | 259.37 ns | 357.10 ns | 341.82 ns | 291.60 ns | 565.86 ns | 298.63 ns | 1.53 us | 164.47 ns |
| blake2/blake2b512 vs rustcrypto | 176.79 ns | 187.71 ns | 150.56 ns | 138.92 ns | 173.69 ns | 155.86 ns | 235.28 ns | 197.77 ns | 1.02 us | 120.55 ns |
| blake2/blake2s256 vs rustcrypto | 128.46 ns | 143.23 ns | 135.85 ns | 108.01 ns | 147.86 ns | 130.97 ns | 186.02 ns | 144.39 ns | 784.33 ns | 99.97 ns |
| blake3 vs blake3 | 57.86 ns | 58.40 ns | 45.44 ns | 52.87 ns | 110.58 ns | 101.64 ns | 131.26 ns | 115.38 ns | 525.46 ns | 74.42 ns |
| ascon-hash256 vs ascon-hash | 441.35 ns | 324.78 ns | 569.16 ns | 643.63 ns | 492.15 ns | 436.84 ns | 1.11 us | 733.23 ns | 2.63 us | 561.17 ns |
| xxh3-64 vs xxhash-rust | 3.92 ns | 2.64 ns | 5.46 ns | 7.68 ns | 5.39 ns | 3.88 ns | 11.35 ns | 6.04 ns | 78.75 ns | 3.62 ns |
| rapidhash-64 vs rapidhash | 3.28 ns | 2.49 ns | 3.19 ns | 4.43 ns | 5.71 ns | 4.08 ns | 16.97 ns | 6.47 ns | 95.95 ns | 2.45 ns |
| hmac-sha256 vs rustcrypto | 178.30 ns | 148.52 ns | 10.18 us | 250.14 ns | 193.30 ns | 165.03 ns | 495.79 ns | 1.04 us | 5.90 us | 126.96 ns |
| aes-256-gcm/encrypt vs rustcrypto | 190.58 ns | 160.10 ns | 203.21 ns | 198.64 ns | 164.68 ns | 151.74 ns | 398.59 ns | 262.35 ns | 508.29 us | 133.22 ns |
| aes-256-gcm-siv/encrypt vs rustcrypto | 198.97 ns | 162.70 ns | 221.96 ns | 225.66 ns | 302.53 ns | 304.54 ns | 451.49 ns | 444.79 ns | 1.02 ms | 237.25 ns |
| chacha20-poly1305/encrypt vs rustcrypto | 300.14 ns | 317.94 ns | 333.51 ns | 314.06 ns | 410.44 ns | 356.67 ns | 482.24 ns | 448.72 ns | 2.12 us | 291.80 ns |
| xchacha20-poly1305/encrypt vs rustcrypto | 405.83 ns | 450.46 ns | 388.93 ns | 412.71 ns | 524.56 ns | 449.21 ns | 624.46 ns | 591.39 ns | 2.71 us | 381.06 ns |
| aegis-256/encrypt vs aegis-crate | 42.48 ns | 34.75 ns | 45.55 ns | 48.46 ns | 93.33 ns | 79.39 ns | 1.35 us | 86.38 ns | 1.06 ms | 48.14 ns |

## Losses Over 5 Percent

Complete list: 467 cases below the 0.95x budget.

| Speedup | Platform | Category | Group | Case | rscrypto | Competitor | Competitor Time |
|---:|---|---|---|---|---:|---|---:|
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `16384` | 103.82 ms | rustcrypto | 693.50 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `262144` | 1.62 s | rustcrypto | 10.96 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `65536` | 408.88 ms | rustcrypto | 2.77 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `1048576` | 6.41 s | rustcrypto | 43.56 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `262144` | 1.61 s | rustcrypto | 10.97 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `16384` | 102.52 ms | rustcrypto | 699.51 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `16384` | 102.91 ms | rustcrypto | 702.99 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `65536` | 413.41 ms | rustcrypto | 2.83 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `4096` | 25.40 ms | rustcrypto | 176.23 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `1048576` | 6.36 s | rustcrypto | 44.48 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `1024` | 6.56 ms | rustcrypto | 46.46 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `4096` | 25.34 ms | rustcrypto | 179.84 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `1048576` | 6.19 s | rustcrypto | 43.92 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1048576` | 6.43 s | rustcrypto | 46.02 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1024` | 6.49 ms | rustcrypto | 46.50 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `262144` | 1.54 s | rustcrypto | 11.07 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `65536` | 378.54 ms | rustcrypto | 2.73 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `65536` | 383.06 ms | rustcrypto | 2.77 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `16384` | 95.35 ms | rustcrypto | 694.33 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `4096` | 25.56 ms | rustcrypto | 190.56 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `262144` | 1.54 s | rustcrypto | 11.69 ms |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `4096` | 24.07 ms | rustcrypto | 189.94 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `256` | 1.70 ms | rustcrypto | 14.07 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `1024` | 6.94 ms | rustcrypto | 60.57 us |
| 0.01x | SPR | SHA-2 | `sha256/streaming vs sha2` | `64B` | 71.12 ms | sha2 | 627.02 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `256` | 1.55 ms | rustcrypto | 13.85 us |
| 0.01x | SPR | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `64B` | 71.41 ms | rustcrypto | 646.67 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `1024` | 6.54 ms | rustcrypto | 59.49 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `64` | 508.29 us | rustcrypto | 5.87 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `256` | 2.39 ms | rustcrypto | 28.31 us |
| 0.01x | SPR | SHA-2 | `sha256/streaming vs sha2` | `4096B` | 70.34 ms | sha2 | 841.31 us |
| 0.01x | SPR | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `4096B` | 71.50 ms | rustcrypto | 867.84 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `64` | 468.24 us | rustcrypto | 5.94 us |
| 0.01x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `256` | 2.11 ms | rustcrypto | 27.17 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `64` | 1.11 ms | rustcrypto | 20.24 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `64` | 1.02 ms | rustcrypto | 19.21 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `0` | 749.94 us | rustcrypto | 16.64 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `0` | 745.55 us | rustcrypto | 16.73 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `32` | 295.43 us | rustcrypto | 6.70 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `0` | 101.46 us | rustcrypto | 2.34 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `32` | 912.97 us | rustcrypto | 21.14 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `1` | 806.87 us | rustcrypto | 19.01 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `1` | 812.43 us | rustcrypto | 19.29 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `32` | 276.76 us | rustcrypto | 6.66 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `32` | 840.65 us | rustcrypto | 20.36 us |
| 0.02x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1` | 198.43 us | rustcrypto | 4.85 us |
| 0.03x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `1` | 190.22 us | rustcrypto | 4.89 us |
| 0.03x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `0` | 93.19 us | rustcrypto | 2.47 us |
| 0.08x | RISC-V | AEAD | `aegis-256/decrypt vs aegis-crate` | `0` | 947.55 us | aegis-crate | 76.76 us |
| 0.08x | RISC-V | AEAD | `aegis-256/encrypt vs aegis-crate` | `16384` | 42.58 ms | aegis-crate | 3.45 ms |
| 0.08x | RISC-V | AEAD | `aegis-256/decrypt vs aegis-crate` | `32` | 1.03 ms | aegis-crate | 83.67 us |
| 0.08x | RISC-V | AEAD | `aegis-256/encrypt vs aegis-crate` | `4096` | 10.90 ms | aegis-crate | 891.45 us |
| 0.08x | RISC-V | AEAD | `aegis-256/decrypt vs aegis-crate` | `1` | 976.86 us | aegis-crate | 80.40 us |
| 0.08x | RISC-V | AEAD | `aegis-256/encrypt vs aegis-crate` | `1024` | 3.41 ms | aegis-crate | 281.69 us |
| 0.08x | RISC-V | AEAD | `aegis-256/decrypt vs aegis-crate` | `64` | 1.08 ms | aegis-crate | 89.60 us |
| 0.08x | RISC-V | AEAD | `aegis-256/encrypt vs aegis-crate` | `65536` | 166.25 ms | aegis-crate | 13.81 ms |
| 0.08x | RISC-V | AEAD | `aegis-256/encrypt vs aegis-crate` | `256` | 1.52 ms | aegis-crate | 126.80 us |
| 0.08x | RISC-V | AEAD | `aegis-256/encrypt vs aegis-crate` | `262144` | 660.47 ms | aegis-crate | 55.15 ms |
| 0.08x | RISC-V | AEAD | `aegis-256/encrypt vs aegis-crate` | `64` | 1.06 ms | aegis-crate | 88.49 us |
| 0.08x | RISC-V | AEAD | `aegis-256/encrypt vs aegis-crate` | `1` | 943.74 us | aegis-crate | 78.90 us |
| 0.08x | RISC-V | AEAD | `aegis-256/encrypt vs aegis-crate` | `32` | 977.78 us | aegis-crate | 82.13 us |
| 0.08x | RISC-V | AEAD | `aegis-256/encrypt vs aegis-crate` | `1048576` | 2.61 s | aegis-crate | 220.45 ms |
| 0.08x | RISC-V | AEAD | `aegis-256/decrypt vs aegis-crate` | `256` | 1.53 ms | aegis-crate | 129.29 us |
| 0.08x | RISC-V | AEAD | `aegis-256/encrypt vs aegis-crate` | `0` | 895.80 us | aegis-crate | 75.71 us |
| 0.09x | RISC-V | AEAD | `aegis-256/decrypt vs aegis-crate` | `16384` | 38.44 ms | aegis-crate | 3.37 ms |
| 0.09x | RISC-V | AEAD | `aegis-256/decrypt vs aegis-crate` | `1024` | 3.27 ms | aegis-crate | 289.13 us |
| 0.09x | RISC-V | AEAD | `aegis-256/decrypt vs aegis-crate` | `4096` | 10.36 ms | aegis-crate | 925.72 us |
| 0.09x | RISC-V | AEAD | `aegis-256/decrypt vs aegis-crate` | `65536` | 148.05 ms | aegis-crate | 13.42 ms |
| 0.09x | RISC-V | AEAD | `aegis-256/decrypt vs aegis-crate` | `1048576` | 2.38 s | aegis-crate | 218.71 ms |
| 0.10x | RISC-V | AEAD | `aegis-256/decrypt vs aegis-crate` | `262144` | 589.46 ms | aegis-crate | 58.12 ms |
| 0.20x | macOS-A64 | Password Hashing | `argon2d-small vs rustcrypto` | `m=64_t=3_p=2` | 395.94 us | rustcrypto | 79.63 us |
| 0.20x | macOS-A64 | Password Hashing | `argon2id-small vs rustcrypto` | `m=64_t=3_p=2` | 406.67 us | rustcrypto | 82.15 us |
| 0.23x | macOS-A64 | Password Hashing | `argon2i-small vs rustcrypto` | `m=64_t=3_p=2` | 411.91 us | rustcrypto | 94.55 us |
| 0.35x | macOS-A64 | Checksums | `crc32 vs crc32fast` | `64` | 8.63 ns | crc32fast | 3.02 ns |
| 0.37x | macOS-A64 | Checksums | `crc32 vs crc-fast` | `64` | 8.63 ns | crc-fast | 3.17 ns |
| 0.37x | Zen5 | XXH3 | `xxh3-64 vs xxhash-rust` | `256` | 16.76 ns | xxhash-rust | 6.21 ns |
| 0.38x | Zen4 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `64` | 190.58 ns | rustcrypto | 72.30 ns |
| 0.42x | Zen5 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `64` | 160.10 ns | rustcrypto | 67.38 ns |
| 0.43x | ICL | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `64` | 198.64 ns | rustcrypto | 85.33 ns |
| 0.44x | SPR | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `32` | 148.12 ns | rustcrypto | 65.80 ns |
| 0.45x | SPR | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `64` | 203.21 ns | rustcrypto | 91.97 ns |
| 0.46x | Zen4 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `32` | 130.28 ns | rustcrypto | 59.68 ns |
| 0.48x | Zen5 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `32` | 107.45 ns | rustcrypto | 51.26 ns |
| 0.48x | ICL | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `32` | 141.53 ns | rustcrypto | 67.99 ns |
| 0.50x | ICL | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `0` | 78.88 ns | rustcrypto | 39.69 ns |
| 0.51x | Zen5 | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `64` | 148.52 ns | rustcrypto | 75.47 ns |
| 0.51x | ICL | Checksums | `crc32 vs crc32fast` | `256` | 23.45 ns | crc32fast | 12.08 ns |
| 0.52x | Zen4 | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `64` | 175.78 ns | rustcrypto | 92.03 ns |
| 0.53x | ICL | XXH3 | `xxh3-64 vs xxhash-rust` | `64` | 7.68 ns | xxhash-rust | 4.04 ns |
| 0.53x | SPR | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `0` | 78.59 ns | rustcrypto | 41.65 ns |
| 0.54x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1` | 15.36 ns | rapidhash | 8.33 ns |
| 0.55x | SPR | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `64` | 183.81 ns | rustcrypto | 100.22 ns |
| 0.55x | SPR | XXH3 | `xxh3-64 vs xxhash-rust` | `64` | 5.46 ns | xxhash-rust | 2.99 ns |
| 0.56x | RISC-V | Checksums | `crc32c vs crc32c` | `32` | 124.42 ns | crc32c | 69.80 ns |
| 0.58x | SPR | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `32` | 128.19 ns | rustcrypto | 74.14 ns |
| 0.60x | Zen4 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `0` | 67.57 ns | rustcrypto | 40.73 ns |
| 0.61x | RISC-V | Checksums | `crc64-nvme vs crc-fast` | `1048576` | 3.70 ms | crc-fast | 2.25 ms |
| 0.61x | ICL | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `64` | 178.49 ns | rustcrypto | 108.96 ns |
| 0.61x | Zen5 | AEAD | `chacha20-poly1305/encrypt vs rustcrypto` | `256` | 660.32 ns | rustcrypto | 404.59 ns |
| 0.62x | Zen5 | AEAD | `chacha20-poly1305/decrypt vs rustcrypto` | `256` | 646.04 ns | rustcrypto | 400.24 ns |
| 0.62x | SPR | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1` | 125.95 ns | rustcrypto | 78.30 ns |
| 0.63x | Zen5 | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `32` | 96.15 ns | rustcrypto | 60.26 ns |
| 0.63x | s390x | Checksums | `crc24-openpgp vs crc` | `64` | 280.18 ns | crc | 175.85 ns |
| 0.63x | Zen5 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `0` | 52.85 ns | rustcrypto | 33.40 ns |
| 0.65x | RISC-V | Checksums | `crc32 vs crc32fast` | `64` | 169.24 ns | crc32fast | 109.47 ns |
| 0.65x | Zen5 | AEAD | `xchacha20-poly1305/decrypt vs rustcrypto` | `256` | 775.05 ns | rustcrypto | 504.79 ns |
| 0.65x | ICL | AEAD | `chacha20-poly1305/encrypt vs rustcrypto` | `256` | 647.77 ns | rustcrypto | 422.46 ns |
| 0.66x | ICL | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1` | 115.04 ns | rustcrypto | 75.36 ns |
| 0.66x | SPR | AEAD | `xchacha20-poly1305/decrypt vs rustcrypto` | `256` | 761.26 ns | rustcrypto | 498.76 ns |
| 0.66x | Zen5 | AEAD | `xchacha20-poly1305/encrypt vs rustcrypto` | `256` | 792.28 ns | rustcrypto | 520.65 ns |
| 0.67x | SPR | AEAD | `chacha20-poly1305/decrypt vs rustcrypto` | `256` | 659.74 ns | rustcrypto | 440.51 ns |
| 0.68x | SPR | Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | `64` | 26.18 us | rustcrypto | 17.68 us |
| 0.68x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `32` | 16.44 ns | rapidhash | 11.16 ns |
| 0.68x | s390x | XXH3 | `xxh3-64 vs xxhash-rust` | `32` | 17.24 ns | xxhash-rust | 11.71 ns |
| 0.68x | ICL | AEAD | `xchacha20-poly1305/encrypt vs rustcrypto` | `256` | 736.89 ns | rustcrypto | 500.99 ns |
| 0.68x | Zen4 | AEAD | `chacha20-poly1305/encrypt vs rustcrypto` | `256` | 617.35 ns | rustcrypto | 421.07 ns |
| 0.68x | Zen4 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1` | 108.27 ns | rustcrypto | 73.85 ns |
| 0.68x | SPR | XXH3 | `xxh3-128 vs xxhash-rust` | `256` | 18.26 ns | xxhash-rust | 12.47 ns |
| 0.68x | ICL | AEAD | `chacha20-poly1305/decrypt vs rustcrypto` | `256` | 640.61 ns | rustcrypto | 437.84 ns |
| 0.69x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `1048576` | 3.70 ms | crc64fast-nvme | 2.55 ms |
| 0.69x | Zen5 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1` | 86.31 ns | rustcrypto | 59.71 ns |
| 0.69x | SPR | AEAD | `xchacha20-poly1305/encrypt vs rustcrypto` | `256` | 687.84 ns | rustcrypto | 476.73 ns |
| 0.70x | s390x | Checksums | `crc16-ccitt vs crc` | `1` | 12.96 ns | crc | 9.03 ns |
| 0.70x | Zen4 | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `32` | 113.23 ns | rustcrypto | 79.00 ns |
| 0.70x | POWER10 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1` | 9.75 ns | rapidhash | 6.86 ns |
| 0.71x | Zen5 | XXH3 | `xxh3-64 vs xxhash-rust` | `1024` | 19.99 ns | xxhash-rust | 14.10 ns |
| 0.71x | SPR | Auth/KDF | `hmac-sha256 vs rustcrypto` | `1024` | 15.01 us | rustcrypto | 10.64 us |
| 0.71x | ICL | AEAD | `xchacha20-poly1305/decrypt vs rustcrypto` | `256` | 736.40 ns | rustcrypto | 522.50 ns |
| 0.71x | Zen4 | AEAD | `xchacha20-poly1305/encrypt vs rustcrypto` | `256` | 726.79 ns | rustcrypto | 516.96 ns |
| 0.72x | SPR | AEAD | `chacha20-poly1305/encrypt vs rustcrypto` | `256` | 663.68 ns | rustcrypto | 474.84 ns |
| 0.72x | RISC-V | Checksums | `crc32c vs crc32c` | `64` | 169.37 ns | crc32c | 121.33 ns |
| 0.72x | POWER10 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `0` | 6.70 ns | rapidhash | 4.82 ns |
| 0.72x | POWER10 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `256` | 58.08 ns | rapidhash | 41.90 ns |
| 0.73x | ICL | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `32` | 123.04 ns | rustcrypto | 89.49 ns |
| 0.73x | s390x | BLAKE3 | `blake3/xof vs blake3` | `1024` | 2.59 us | blake3 | 1.90 us |
| 0.74x | RISC-V | BLAKE2 | `blake2/blake2b512 vs rustcrypto` | `1048576` | 9.08 ms | rustcrypto | 6.69 ms |
| 0.74x | RISC-V | BLAKE2 | `blake2/blake2b256 vs rustcrypto` | `1048576` | 9.05 ms | rustcrypto | 6.68 ms |
| 0.74x | RISC-V | XXH3 | `xxh3-64 vs xxhash-rust` | `0` | 9.03 ns | xxhash-rust | 6.70 ns |
| 0.74x | Zen4 | XXH3 | `xxh3-64 vs xxhash-rust` | `256` | 12.74 ns | xxhash-rust | 9.49 ns |
| 0.75x | macOS-A64 | XXH3 | `xxh3-64 vs xxhash-rust` | `32` | 2.17 ns | xxhash-rust | 1.63 ns |
| 0.76x | s390x | BLAKE3 | `blake3 vs blake3` | `1024` | 2.46 us | blake3 | 1.85 us |
| 0.76x | SPR | Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | `32` | 17.21 us | rustcrypto | 13.01 us |
| 0.76x | Zen4 | AEAD | `chacha20-poly1305/decrypt vs rustcrypto` | `256` | 602.75 ns | rustcrypto | 456.78 ns |
| 0.76x | Zen5 | XXH3 | `xxh3-128 vs xxhash-rust` | `256` | 12.34 ns | xxhash-rust | 9.38 ns |
| 0.76x | RISC-V | BLAKE3 | `blake3/streaming vs blake3` | `65536B` | 10.64 ms | blake3 | 8.12 ms |
| 0.76x | s390x | Checksums | `crc64-xz vs crc64fast` | `1` | 12.15 ns | crc64fast | 9.27 ns |
| 0.77x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `0` | 83.51 ns | aegis-crate | 64.05 ns |
| 0.77x | s390x | Checksums | `crc64-nvme vs crc64fast-nvme` | `1` | 12.11 ns | crc64fast-nvme | 9.31 ns |
| 0.77x | Zen4 | AEAD | `xchacha20-poly1305/decrypt vs rustcrypto` | `256` | 712.82 ns | rustcrypto | 547.84 ns |
| 0.77x | RISC-V | BLAKE3 | `blake3/streaming vs blake3` | `16384B` | 10.57 ms | blake3 | 8.13 ms |
| 0.77x | RISC-V | BLAKE3 | `blake3/streaming vs blake3` | `4096B` | 10.59 ms | blake3 | 8.16 ms |
| 0.77x | s390x | BLAKE3 | `blake3/keyed vs blake3` | `1024` | 2.46 us | blake3 | 1.90 us |
| 0.77x | Grav4 | XXH3 | `xxh3-128 vs xxhash-rust` | `256` | 26.00 ns | xxhash-rust | 20.11 ns |
| 0.77x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `0` | 69.22 ns | aegis-crate | 53.60 ns |
| 0.77x | SPR | Auth/KDF | `hmac-sha256 vs rustcrypto` | `4096` | 17.07 us | rustcrypto | 13.22 us |
| 0.78x | RISC-V | Checksums | `crc64-xz vs crc64fast` | `32` | 151.53 ns | crc64fast | 117.66 ns |
| 0.78x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `32` | 150.63 ns | crc64fast-nvme | 116.97 ns |
| 0.78x | POWER10 | Checksums | `crc64-xz vs crc64fast` | `64` | 41.05 ns | crc64fast | 32.01 ns |
| 0.79x | macOS-A64 | XXH3 | `xxh3-64 vs xxhash-rust` | `256` | 14.07 ns | xxhash-rust | 11.06 ns |
| 0.79x | SPR | RapidHash | `rapidhash-64 vs rapidhash` | `0` | 1.76 ns | rapidhash | 1.38 ns |
| 0.79x | macOS-A64 | XXH3 | `xxh3-64 vs xxhash-rust` | `1` | 2.05 ns | xxhash-rust | 1.61 ns |
| 0.79x | POWER10 | Checksums | `crc64-nvme vs crc64fast-nvme` | `64` | 40.80 ns | crc64fast-nvme | 32.27 ns |
| 0.79x | s390x | Checksums | `crc16-ibm vs crc` | `1` | 11.30 ns | crc | 8.96 ns |
| 0.79x | SPR | XXH3 | `xxh3-64 vs xxhash-rust` | `256` | 12.41 ns | xxhash-rust | 9.84 ns |
| 0.79x | POWER10 | Checksums | `crc32 vs crc32fast` | `64` | 36.64 ns | crc32fast | 29.12 ns |
| 0.80x | macOS-A64 | BLAKE2 | `blake2/host-overhead/blake2s256 vs rustcrypto` | `1` | 125.70 ns | rustcrypto | 100.67 ns |
| 0.80x | SPR | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `1` | 103.99 ns | rustcrypto | 83.39 ns |
| 0.80x | s390x | Checksums | `crc64-xz vs crc64fast` | `64` | 51.77 ns | crc64fast | 41.60 ns |
| 0.81x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1` | 90.13 ns | aegis-crate | 72.59 ns |
| 0.81x | macOS-A64 | Auth/KDF | `hkdf-sha256/expand vs rustcrypto` | `32` | 79.06 ns | rustcrypto | 63.69 ns |
| 0.81x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `32` | 88.67 ns | aegis-crate | 71.49 ns |
| 0.81x | macOS-A64 | RapidHash | `rapidhash-64 vs rapidhash` | `1` | 1.56 ns | rapidhash | 1.26 ns |
| 0.81x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `64` | 209.83 ns | crc64fast-nvme | 169.99 ns |
| 0.81x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `32` | 75.05 ns | aegis-crate | 61.03 ns |
| 0.81x | RISC-V | BLAKE3 | `blake3/xof vs blake3` | `262144` | 2.50 ms | blake3 | 2.04 ms |
| 0.82x | macOS-A64 | SHA-2 | `sha224 vs sha2` | `32` | 35.77 ns | sha2 | 29.17 ns |
| 0.82x | SPR | BLAKE2 | `blake2/blake2b256 vs rustcrypto` | `0` | 173.08 ns | rustcrypto | 141.16 ns |
| 0.82x | SPR | Checksums | `crc32 vs crc-fast` | `16384` | 190.04 ns | crc-fast | 155.21 ns |
| 0.82x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1` | 75.14 ns | aegis-crate | 61.41 ns |
| 0.82x | Grav4 | Checksums | `crc32c vs crc-fast` | `256` | 14.71 ns | crc-fast | 12.02 ns |
| 0.82x | ICL | XXH3 | `xxh3-128 vs xxhash-rust` | `256` | 21.83 ns | xxhash-rust | 17.89 ns |
| 0.82x | macOS-A64 | BLAKE2 | `blake2/host-stream-overhead/blake2s256 vs rustcrypto` | `0` | 126.06 ns | rustcrypto | 103.53 ns |
| 0.82x | SPR | BLAKE2 | `blake2/blake2b256 vs rustcrypto` | `64` | 172.39 ns | rustcrypto | 141.62 ns |
| 0.82x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `64` | 93.33 ns | aegis-crate | 76.81 ns |
| 0.82x | RISC-V | Checksums | `crc64-xz vs crc64fast` | `1048576` | 3.50 ms | crc64fast | 2.88 ms |
| 0.83x | s390x | BLAKE3 | `blake3/derive-key vs blake3` | `1024` | 2.46 us | blake3 | 2.03 us |
| 0.83x | ICL | BLAKE3 | `blake3/keyed vs blake3` | `256` | 257.28 ns | blake3 | 212.44 ns |
| 0.83x | macOS-A64 | BLAKE2 | `blake2/host-stream-overhead/blake2s256 vs rustcrypto` | `16` | 126.35 ns | rustcrypto | 104.46 ns |
| 0.83x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `64` | 79.39 ns | aegis-crate | 65.64 ns |
| 0.83x | RISC-V | Checksums | `crc64-xz vs crc64fast` | `64` | 206.93 ns | crc64fast | 171.72 ns |
| 0.83x | s390x | XXH3 | `xxh3-128 vs xxhash-rust` | `0` | 7.71 ns | xxhash-rust | 6.41 ns |
| 0.83x | RISC-V | RapidHash | `rapidhash-64 vs rapidhash` | `262144` | 497.40 us | rapidhash | 413.75 us |
| 0.83x | macOS-A64 | BLAKE2 | `blake2/host-stream-overhead/blake2s256 vs rustcrypto` | `32` | 126.66 ns | rustcrypto | 105.41 ns |
| 0.83x | Zen4 | XXH3 | `xxh3-128 vs xxhash-rust` | `256` | 15.96 ns | xxhash-rust | 13.32 ns |
| 0.84x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `262144` | 631.67 us | crc64fast-nvme | 527.64 us |
| 0.84x | POWER10 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `256` | 27.35 ns | rapidhash | 22.86 ns |
| 0.84x | Zen5 | BLAKE3 | `blake3/streaming vs blake3` | `64B` | 1.63 ms | blake3 | 1.36 ms |
| 0.84x | s390x | Checksums | `crc32c vs crc32c` | `1` | 11.43 ns | crc32c | 9.57 ns |
| 0.84x | macOS-A64 | BLAKE2 | `blake2/host-overhead/blake2b256 vs rustcrypto` | `1` | 143.48 ns | rustcrypto | 120.28 ns |
| 0.84x | macOS-A64 | BLAKE2 | `blake2/host-overhead/blake2b256 vs rustcrypto` | `16` | 142.89 ns | rustcrypto | 120.20 ns |
| 0.85x | SPR | Checksums | `crc32c vs crc-fast` | `1024` | 18.28 ns | crc-fast | 15.45 ns |
| 0.85x | RISC-V | Checksums | `crc64-nvme vs crc-fast` | `262144` | 631.67 us | crc-fast | 535.72 us |
| 0.85x | macOS-A64 | BLAKE2 | `blake2/host-stream-overhead/blake2s256 vs rustcrypto` | `64` | 125.62 ns | rustcrypto | 106.63 ns |
| 0.85x | macOS-A64 | BLAKE2 | `blake2/host-overhead/blake2s256 vs rustcrypto` | `0` | 123.14 ns | rustcrypto | 104.53 ns |
| 0.85x | s390x | Checksums | `crc32 vs crc32fast` | `64` | 36.73 ns | crc32fast | 31.18 ns |
| 0.85x | Zen5 | BLAKE3 | `blake3/keyed vs blake3` | `256` | 348.39 ns | blake3 | 296.13 ns |
| 0.85x | POWER10 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1024` | 122.66 ns | rapidhash | 104.27 ns |
| 0.85x | SPR | BLAKE2 | `blake2/blake2s256 vs rustcrypto` | `1` | 131.66 ns | rustcrypto | 112.23 ns |
| 0.85x | s390x | Checksums | `crc64-nvme vs crc64fast-nvme` | `64` | 48.84 ns | crc64fast-nvme | 41.65 ns |
| 0.86x | macOS-A64 | XXH3 | `xxh3-64 vs xxhash-rust` | `64` | 3.62 ns | xxhash-rust | 3.10 ns |
| 0.86x | RISC-V | RapidHash | `rapidhash-64 vs rapidhash` | `64` | 95.95 ns | rapidhash | 82.26 ns |
| 0.86x | macOS-A64 | BLAKE2 | `blake2/host-stream-overhead/blake2s256 vs rustcrypto` | `1` | 126.07 ns | rustcrypto | 108.09 ns |
| 0.86x | SPR | Auth/KDF | `hmac-sha256 vs rustcrypto` | `16384` | 24.79 us | rustcrypto | 21.31 us |
| 0.86x | ICL | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `4096B` | 917.45 us | rustcrypto | 789.07 us |
| 0.86x | Zen4 | XXH3 | `xxh3-64 vs xxhash-rust` | `64` | 3.92 ns | xxhash-rust | 3.37 ns |
| 0.86x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `262144` | 54.61 us | rapidhash | 47.14 us |
| 0.86x | Grav3 | XXH3 | `xxh3-128 vs xxhash-rust` | `256` | 33.43 ns | xxhash-rust | 28.88 ns |
| 0.86x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `0` | 3.82 ns | rapidhash | 3.30 ns |
| 0.86x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `1048576` | 104.72 us | rapidhash | 90.51 us |
| 0.86x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `65536` | 13.67 us | rapidhash | 11.82 us |
| 0.87x | SPR | BLAKE2 | `blake2/blake2s256 vs rustcrypto` | `0` | 122.73 ns | rustcrypto | 106.33 ns |
| 0.87x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `256` | 78.53 ns | rapidhash | 68.08 ns |
| 0.87x | POWER10 | RapidHash | `rapidhash-64 vs rapidhash` | `1` | 5.42 ns | rapidhash | 4.70 ns |
| 0.87x | SPR | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `0` | 57.78 ns | rustcrypto | 50.16 ns |
| 0.87x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1048576` | 108.21 us | rapidhash | 94.05 us |
| 0.87x | ICL | BLAKE3 | `blake3/streaming vs blake3` | `64B` | 1.12 ms | blake3 | 971.39 us |
| 0.87x | s390x | Checksums | `crc32c vs crc32c` | `0` | 3.12 ns | crc32c | 2.71 ns |
| 0.87x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `1024` | 116.70 ns | rapidhash | 101.58 ns |
| 0.87x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1048576` | 215.71 us | rapidhash | 188.06 us |
| 0.87x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `256` | 128.30 ns | aegis-crate | 111.95 ns |
| 0.87x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `16384` | 1.62 us | rapidhash | 1.42 us |
| 0.87x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `65536` | 6.78 us | rapidhash | 5.92 us |
| 0.87x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1024` | 232.81 ns | rapidhash | 203.53 ns |
| 0.87x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `16384` | 1.70 us | rapidhash | 1.49 us |
| 0.88x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `4096` | 440.48 ns | rapidhash | 385.54 ns |
| 0.88x | macOS-A64 | XXH3 | `xxh3-64 vs xxhash-rust` | `4096` | 136.09 ns | xxhash-rust | 119.13 ns |
| 0.88x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `16384` | 3.41 us | rapidhash | 2.98 us |
| 0.88x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `4096` | 875.33 ns | rapidhash | 766.94 ns |
| 0.88x | RISC-V | Checksums | `crc32 vs crc32fast` | `256` | 432.85 ns | crc32fast | 379.44 ns |
| 0.88x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `262144` | 27.03 us | rapidhash | 23.70 us |
| 0.88x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `256` | 113.21 ns | aegis-crate | 99.32 ns |
| 0.88x | ICL | SHA-2 | `sha256/streaming vs sha2` | `4096B` | 915.76 us | sha2 | 803.75 us |
| 0.88x | Grav4 | Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | `32` | 189.46 ns | rustcrypto | 166.34 ns |
| 0.88x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `262144` | 25.99 us | rapidhash | 22.82 us |
| 0.88x | SPR | BLAKE3 | `blake3/xof vs blake3` | `0` | 58.55 ns | blake3 | 51.41 ns |
| 0.88x | RISC-V | Checksums | `crc64-xz vs crc64fast` | `256` | 567.54 ns | crc64fast | 498.51 ns |
| 0.88x | macOS-A64 | BLAKE2 | `blake2/host-stream-overhead/blake2s256 vs rustcrypto` | `128` | 226.94 ns | rustcrypto | 199.66 ns |
| 0.88x | RISC-V | BLAKE3 | `blake3/derive-key vs blake3` | `262144` | 2.27 ms | blake3 | 2.01 ms |
| 0.89x | SPR | Checksums | `crc32c vs crc-fast` | `1048576` | 10.58 us | crc-fast | 9.36 us |
| 0.89x | Grav3 | Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | `32` | 198.13 ns | rustcrypto | 175.41 ns |
| 0.89x | SPR | Auth/KDF | `hmac-sha256 vs rustcrypto` | `65536` | 57.94 us | rustcrypto | 51.34 us |
| 0.89x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `262144` | 51.33 us | rapidhash | 45.50 us |
| 0.89x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `1024` | 2.13 us | crc64fast-nvme | 1.89 us |
| 0.89x | SPR | RapidHash | `rapidhash-v3-128 vs rapidhash` | `16384` | 795.55 ns | rapidhash | 706.08 ns |
| 0.89x | SPR | Checksums | `crc32 vs crc32fast` | `256` | 12.54 ns | crc32fast | 11.13 ns |
| 0.89x | SPR | BLAKE2 | `blake2/blake2b256 vs rustcrypto` | `1` | 165.00 ns | rustcrypto | 146.70 ns |
| 0.89x | macOS-A64 | BLAKE2 | `blake2/host-stream-overhead/blake2b256 vs rustcrypto` | `32` | 136.70 ns | rustcrypto | 121.63 ns |
| 0.89x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `4096` | 413.95 ns | rapidhash | 368.47 ns |
| 0.89x | Zen4 | XXH3 | `xxh3-128 vs xxhash-rust` | `1` | 7.96 ns | xxhash-rust | 7.10 ns |
| 0.89x | Grav4 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `256` | 29.20 ns | rapidhash | 26.03 ns |
| 0.89x | macOS-A64 | XXH3 | `xxh3-64 vs xxhash-rust` | `1024` | 37.16 ns | xxhash-rust | 33.16 ns |
| 0.89x | macOS-A64 | BLAKE2 | `blake2/host-stream-overhead/blake2b256 vs rustcrypto` | `1` | 132.20 ns | rustcrypto | 118.08 ns |
| 0.89x | macOS-A64 | BLAKE2 | `blake2/host-stream-overhead/blake2b256 vs rustcrypto` | `0` | 131.55 ns | rustcrypto | 117.52 ns |
| 0.89x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1024` | 117.09 ns | rapidhash | 104.62 ns |
| 0.89x | ICL | RapidHash | `rapidhash-64 vs rapidhash` | `32` | 2.59 ns | rapidhash | 2.32 ns |
| 0.89x | macOS-A64 | Checksums | `crc32c vs crc-fast` | `256` | 7.03 ns | crc-fast | 6.29 ns |
| 0.89x | SPR | BLAKE2 | `blake2/blake2s256 vs rustcrypto` | `64` | 135.85 ns | rustcrypto | 121.54 ns |
| 0.90x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `64` | 16.97 ns | rapidhash | 15.19 ns |
| 0.90x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `256` | 20.62 ns | rapidhash | 18.48 ns |
| 0.90x | macOS-A64 | BLAKE2 | `blake2/host-overhead/blake2s256 vs rustcrypto` | `128` | 228.37 ns | rustcrypto | 204.80 ns |
| 0.90x | Grav4 | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `64B` | 805.28 us | rustcrypto | 722.24 us |
| 0.90x | SPR | BLAKE3 | `blake3/streaming vs blake3` | `64B` | 1.21 ms | blake3 | 1.08 ms |
| 0.90x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `1024` | 217.69 ns | rapidhash | 195.32 ns |
| 0.90x | macOS-A64 | BLAKE2 | `blake2/host-stream-overhead/blake2b256 vs rustcrypto` | `16` | 134.29 ns | rustcrypto | 120.55 ns |
| 0.90x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `1048576` | 204.86 us | rapidhash | 183.90 us |
| 0.90x | Grav4 | SHA-2 | `sha256/streaming vs sha2` | `64B` | 804.43 us | sha2 | 722.21 us |
| 0.90x | macOS-A64 | BLAKE3 | `blake3/keyed vs blake3` | `65536` | 39.53 us | blake3 | 35.50 us |
| 0.90x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `16384` | 395.27 ns | rapidhash | 355.10 ns |
| 0.90x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `65536` | 12.82 us | rapidhash | 11.53 us |
| 0.90x | macOS-A64 | SHAKE | `shake128 vs sha3` | `262144` | 270.93 us | sha3 | 243.62 us |
| 0.90x | macOS-A64 | SHAKE | `shake128 vs sha3` | `1048576` | 1.08 ms | sha3 | 973.57 us |
| 0.90x | Grav3 | SHA-2 | `sha256/streaming vs sha2` | `64B` | 879.89 us | sha2 | 791.57 us |
| 0.90x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `256` | 552.29 ns | crc64fast-nvme | 496.99 ns |
| 0.90x | ICL | Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | `32` | 273.31 ns | rustcrypto | 246.08 ns |
| 0.90x | SPR | BLAKE3 | `blake3 vs blake3` | `1` | 76.34 ns | blake3 | 68.74 ns |
| 0.90x | macOS-A64 | SHAKE | `shake128 vs sha3` | `65536` | 67.72 us | sha3 | 60.98 us |
| 0.90x | macOS-A64 | Checksums | `crc32 vs crc-fast` | `256` | 6.96 ns | crc-fast | 6.27 ns |
| 0.90x | SPR | AEAD | `aegis-256/encrypt vs aegis-crate` | `0` | 36.95 ns | aegis-crate | 33.32 ns |
| 0.90x | macOS-A64 | BLAKE3 | `blake3/derive-key vs blake3` | `65536` | 39.58 us | blake3 | 35.72 us |
| 0.90x | SPR | BLAKE3 | `blake3/xof vs blake3` | `65536` | 11.42 us | blake3 | 10.31 us |
| 0.90x | SPR | RapidHash | `rapidhash-v3-128 vs rapidhash` | `262144` | 12.54 us | rapidhash | 11.33 us |
| 0.90x | ICL | BLAKE3 | `blake3 vs blake3` | `256` | 255.53 ns | blake3 | 231.03 ns |
| 0.90x | SPR | RapidHash | `rapidhash-v3-128 vs rapidhash` | `65536` | 3.14 us | rapidhash | 2.84 us |
| 0.91x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `256` | 42.81 ns | rapidhash | 38.75 ns |
| 0.91x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `65536` | 157.56 us | crc64fast-nvme | 142.61 us |
| 0.91x | macOS-A64 | SHAKE | `shake128 vs sha3` | `4096` | 4.32 us | sha3 | 3.91 us |
| 0.91x | Grav4 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `256` | 14.85 ns | rapidhash | 13.45 ns |
| 0.91x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `65536` | 6.43 us | rapidhash | 5.82 us |
| 0.91x | Grav3 | BLAKE3 | `blake3/keyed vs blake3` | `1` | 119.87 ns | blake3 | 108.57 ns |
| 0.91x | SPR | RapidHash | `rapidhash-64 vs rapidhash` | `1` | 1.92 ns | rapidhash | 1.74 ns |
| 0.91x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `262144` | 6.24 us | rapidhash | 5.66 us |
| 0.91x | Grav3 | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `64B` | 873.84 us | rustcrypto | 792.94 us |
| 0.91x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `16384` | 3.26 us | rapidhash | 2.96 us |
| 0.91x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `256` | 40.20 ns | rapidhash | 36.52 ns |
| 0.91x | RISC-V | RapidHash | `rapidhash-128 vs rapidhash` | `65536` | 211.75 us | rapidhash | 192.59 us |
| 0.91x | Grav4 | XXH3 | `xxh3-128 vs xxhash-rust` | `1` | 2.77 ns | xxhash-rust | 2.53 ns |
| 0.91x | SPR | BLAKE2 | `blake2/blake2s256 vs rustcrypto` | `32` | 124.95 ns | rustcrypto | 113.85 ns |
| 0.91x | SPR | BLAKE2 | `blake2/blake2b256 vs rustcrypto` | `256` | 275.88 ns | rustcrypto | 251.59 ns |
| 0.91x | macOS-A64 | SHAKE | `shake128 vs sha3` | `1024` | 1.20 us | sha3 | 1.09 us |
| 0.91x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1024` | 27.89 ns | rapidhash | 25.47 ns |
| 0.91x | macOS-A64 | BLAKE3 | `blake3 vs blake3` | `65536` | 38.88 us | blake3 | 35.52 us |
| 0.91x | Zen5 | Auth/KDF | `ed25519/verify vs dalek` | `0` | 20.24 us | dalek | 18.50 us |
| 0.91x | Zen4 | XXH3 | `xxh3-128 vs xxhash-rust` | `0` | 6.87 ns | xxhash-rust | 6.28 ns |
| 0.91x | macOS-A64 | BLAKE2 | `blake2/host-stream-overhead/blake2b256 vs rustcrypto` | `64` | 134.88 ns | rustcrypto | 123.38 ns |
| 0.92x | macOS-A64 | SHA-3 | `sha3-384 vs sha3` | `0` | 168.28 ns | sha3 | 154.01 ns |
| 0.92x | Zen5 | Auth/KDF | `ed25519/verify vs dalek` | `1024` | 21.12 us | dalek | 19.33 us |
| 0.92x | SPR | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1048576` | 50.04 us | rapidhash | 45.82 us |
| 0.92x | Zen5 | XXH3 | `xxh3-64 vs xxhash-rust` | `64` | 2.64 ns | xxhash-rust | 2.42 ns |
| 0.92x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `16384` | 32.98 us | crc64fast-nvme | 30.22 us |
| 0.92x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `65536` | 1.58 us | rapidhash | 1.44 us |
| 0.92x | Zen5 | Auth/KDF | `ed25519/verify vs dalek` | `32` | 20.02 us | dalek | 18.36 us |
| 0.92x | SPR | RapidHash | `rapidhash-v3-128 vs rapidhash` | `4096` | 212.45 ns | rapidhash | 194.91 ns |
| 0.92x | macOS-A64 | SHA-2 | `sha384 vs sha2` | `256` | 269.28 ns | sha2 | 247.07 ns |
| 0.92x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1024` | 251.12 ns | aegis-crate | 230.47 ns |
| 0.92x | Grav4 | XXH3 | `xxh3-128 vs xxhash-rust` | `1024` | 56.89 ns | xxhash-rust | 52.21 ns |
| 0.92x | Zen5 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `16384` | 308.46 ns | rapidhash | 283.32 ns |
| 0.92x | RISC-V | Checksums | `crc64-nvme vs crc-fast` | `1024` | 2.13 us | crc-fast | 1.96 us |
| 0.92x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1048576` | 24.88 us | rapidhash | 22.86 us |
| 0.92x | SPR | Checksums | `crc32c vs crc-fast` | `4096` | 55.27 ns | crc-fast | 50.80 ns |
| 0.92x | SPR | BLAKE3 | `blake3/xof vs blake3` | `64` | 65.77 ns | blake3 | 60.45 ns |
| 0.92x | RISC-V | BLAKE3 | `blake3/xof vs blake3` | `65536` | 562.28 us | blake3 | 516.90 us |
| 0.92x | Grav3 | Checksums | `crc32 vs crc-fast` | `65536` | 1.73 us | crc-fast | 1.59 us |
| 0.92x | Grav3 | Checksums | `crc32c vs crc-fast` | `1048576` | 26.01 us | crc-fast | 23.91 us |
| 0.92x | macOS-A64 | SHA-2 | `sha512-256 vs sha2` | `256` | 268.60 ns | sha2 | 246.97 ns |
| 0.92x | Zen4 | Auth/KDF | `pbkdf2-sha256/iters=1000 vs rustcrypto` | `32` | 83.57 us | rustcrypto | 76.86 us |
| 0.92x | Grav3 | Checksums | `crc32 vs crc-fast` | `262144` | 6.50 us | crc-fast | 5.98 us |
| 0.92x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1024` | 276.31 ns | aegis-crate | 254.17 ns |
| 0.92x | Zen4 | Auth/KDF | `pbkdf2-sha256/iters=1000 vs rustcrypto` | `64` | 167.01 us | rustcrypto | 153.63 us |
| 0.92x | POWER10 | BLAKE2 | `blake2/keyed/blake2b256 vs rustcrypto` | `32` | 406.94 ns | rustcrypto | 374.34 ns |
| 0.92x | Zen5 | XXH3 | `xxh3-128 vs xxhash-rust` | `1024` | 15.77 ns | xxhash-rust | 14.51 ns |
| 0.92x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `4096` | 827.61 ns | rapidhash | 761.76 ns |
| 0.92x | Grav3 | Checksums | `crc32c vs crc-fast` | `262144` | 6.49 us | crc-fast | 5.98 us |
| 0.92x | Grav4 | XXH3 | `xxh3-128 vs xxhash-rust` | `32` | 4.54 ns | xxhash-rust | 4.18 ns |
| 0.92x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `262144` | 24.68 us | rapidhash | 22.73 us |
| 0.92x | SPR | Checksums | `crc64-nvme vs crc-fast` | `16384` | 171.96 ns | crc-fast | 158.41 ns |
| 0.92x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `262144` | 12.33 us | rapidhash | 11.36 us |
| 0.92x | macOS-A64 | Auth/KDF | `hmac-sha512 vs rustcrypto` | `256` | 528.82 ns | rustcrypto | 487.32 ns |
| 0.92x | RISC-V | RapidHash | `rapidhash-v3-64 vs rapidhash` | `262144` | 447.54 us | rapidhash | 412.49 us |
| 0.92x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1048576` | 50.02 us | rapidhash | 46.11 us |
| 0.92x | Zen4 | Auth/KDF | `pbkdf2-sha256/iters=100 vs rustcrypto` | `64` | 16.82 us | rustcrypto | 15.52 us |
| 0.92x | RISC-V | BLAKE2 | `blake2/keyed/blake2s128 vs rustcrypto` | `1` | 1.55 us | rustcrypto | 1.43 us |
| 0.92x | Zen4 | Auth/KDF | `pbkdf2-sha256/iters=100 vs rustcrypto` | `32` | 8.46 us | rustcrypto | 7.80 us |
| 0.92x | SPR | Auth/KDF | `hmac-sha256 vs rustcrypto` | `262144` | 182.92 us | rustcrypto | 168.76 us |
| 0.92x | RISC-V | BLAKE2 | `blake2/blake2b256 vs rustcrypto` | `262144` | 2.23 ms | rustcrypto | 2.06 ms |
| 0.92x | POWER10 | BLAKE2 | `blake2/keyed/blake2b512 vs rustcrypto` | `32` | 408.08 ns | rustcrypto | 376.55 ns |
| 0.92x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `65536` | 3.08 us | rapidhash | 2.84 us |
| 0.92x | Grav3 | Checksums | `crc32c vs crc-fast` | `65536` | 1.72 us | crc-fast | 1.59 us |
| 0.92x | RISC-V | RapidHash | `rapidhash-64 vs rapidhash` | `0` | 20.26 ns | rapidhash | 18.71 ns |
| 0.92x | RISC-V | Ascon | `ascon-hash256 vs ascon-hash` | `0` | 973.50 ns | ascon-hash | 899.49 ns |
| 0.92x | macOS-A64 | SHA-3 | `sha3-256/streaming vs sha3` | `4096B` | 1.30 ms | sha3 | 1.21 ms |
| 0.92x | Grav3 | XXH3 | `xxh3-128 vs xxhash-rust` | `1024` | 68.31 ns | xxhash-rust | 63.14 ns |
| 0.92x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `65536` | 6.16 us | rapidhash | 5.70 us |
| 0.92x | macOS-A64 | SHA-3 | `sha3-256/streaming vs sha3` | `64B` | 1.30 ms | sha3 | 1.21 ms |
| 0.93x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1` | 1.55 ns | rapidhash | 1.44 ns |
| 0.93x | macOS-A64 | SHA-2 | `sha512 vs sha2` | `256` | 269.82 ns | sha2 | 250.02 ns |
| 0.93x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `16384` | 773.62 ns | rapidhash | 717.00 ns |
| 0.93x | SPR | BLAKE3 | `blake3/xof vs blake3` | `32` | 58.44 ns | blake3 | 54.16 ns |
| 0.93x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `16384` | 1.55 us | rapidhash | 1.43 us |
| 0.93x | Grav4 | BLAKE3 | `blake3/keyed vs blake3` | `1` | 110.86 ns | blake3 | 102.80 ns |
| 0.93x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1048576` | 99.94 us | rapidhash | 92.72 us |
| 0.93x | RISC-V | BLAKE3 | `blake3/derive-key vs blake3` | `65536` | 561.54 us | blake3 | 520.98 us |
| 0.93x | Grav3 | BLAKE3 | `blake3/streaming vs blake3` | `4096B` | 956.26 us | blake3 | 887.37 us |
| 0.93x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `262144` | 51.21 us | aegis-crate | 47.60 us |
| 0.93x | Grav3 | Auth/KDF | `hmac-sha256 vs rustcrypto` | `256` | 380.58 ns | rustcrypto | 353.73 ns |
| 0.93x | Zen5 | Auth/KDF | `pbkdf2-sha256/iters=1000 vs rustcrypto` | `64` | 142.67 us | rustcrypto | 132.81 us |
| 0.93x | RISC-V | Auth/KDF | `hmac-sha384 vs rustcrypto` | `4096` | 67.75 us | rustcrypto | 63.07 us |
| 0.93x | Zen5 | Auth/KDF | `pbkdf2-sha256/iters=1000 vs rustcrypto` | `32` | 71.31 us | rustcrypto | 66.40 us |
| 0.93x | Zen5 | Auth/KDF | `pbkdf2-sha256/iters=100 vs rustcrypto` | `64` | 14.36 us | rustcrypto | 13.37 us |
| 0.93x | Zen5 | BLAKE3 | `blake3 vs blake3` | `256` | 346.62 ns | blake3 | 322.81 ns |
| 0.93x | POWER10 | BLAKE2 | `blake2/keyed/blake2s128 vs rustcrypto` | `16384` | 33.97 us | rustcrypto | 31.64 us |
| 0.93x | RISC-V | Checksums | `crc64-nvme vs crc-fast` | `65536` | 157.56 us | crc-fast | 146.82 us |
| 0.93x | Zen5 | Auth/KDF | `pbkdf2-sha256/iters=100 vs rustcrypto` | `32` | 7.21 us | rustcrypto | 6.72 us |
| 0.93x | POWER10 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `4096` | 404.45 ns | rapidhash | 377.00 ns |
| 0.93x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1048576` | 210.79 us | aegis-crate | 196.51 us |
| 0.93x | Zen5 | Checksums | `crc32c vs crc-fast` | `1024` | 16.47 ns | crc-fast | 15.37 ns |
| 0.93x | POWER10 | BLAKE2 | `blake2/keyed/blake2b256 vs rustcrypto` | `64` | 401.61 ns | rustcrypto | 374.95 ns |
| 0.93x | Zen5 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `16384` | 610.11 ns | rapidhash | 569.61 ns |
| 0.93x | macOS-A64 | SHA-3 | `sha3-384 vs sha3` | `1` | 166.53 ns | sha3 | 155.57 ns |
| 0.93x | Grav3 | BLAKE3 | `blake3 vs blake3` | `1` | 116.40 ns | blake3 | 108.78 ns |
| 0.93x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1` | 2.72 ns | rapidhash | 2.54 ns |
| 0.94x | macOS-A64 | SHA-3 | `sha3-512 vs sha3` | `0` | 165.27 ns | sha3 | 154.54 ns |
| 0.94x | Grav4 | Checksums | `crc32 vs crc-fast` | `256` | 12.95 ns | crc-fast | 12.11 ns |
| 0.94x | SPR | AEAD | `aegis-256/encrypt vs aegis-crate` | `32` | 40.73 ns | aegis-crate | 38.10 ns |
| 0.94x | macOS-A64 | BLAKE3 | `blake3/xof vs blake3` | `65536` | 37.98 us | blake3 | 35.55 us |
| 0.94x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `4096` | 873.65 ns | aegis-crate | 817.99 ns |
| 0.94x | Grav3 | BLAKE3 | `blake3/keyed vs blake3` | `4096` | 3.52 us | blake3 | 3.30 us |
| 0.94x | Zen5 | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `0` | 40.72 ns | rustcrypto | 38.14 ns |
| 0.94x | macOS-A64 | SHAKE | `shake128 vs sha3` | `16384` | 16.96 us | sha3 | 15.88 us |
| 0.94x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `32` | 80.98 ns | aegis-crate | 75.86 ns |
| 0.94x | macOS-A64 | BLAKE2 | `blake2/host-overhead/blake2b256 vs rustcrypto` | `32` | 130.40 ns | rustcrypto | 122.17 ns |
| 0.94x | Grav3 | BLAKE3 | `blake3 vs blake3` | `4096` | 3.52 us | blake3 | 3.30 us |
| 0.94x | ICL | XXH3 | `xxh3-64 vs xxhash-rust` | `1` | 2.60 ns | xxhash-rust | 2.43 ns |
| 0.94x | RISC-V | Auth/KDF | `hmac-sha512 vs rustcrypto` | `65536` | 950.21 us | rustcrypto | 891.14 us |
| 0.94x | Zen5 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `262144` | 4.81 us | rapidhash | 4.51 us |
| 0.94x | Zen5 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `65536` | 1.21 us | rapidhash | 1.14 us |
| 0.94x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `4096` | 201.95 ns | rapidhash | 189.54 ns |
| 0.94x | Grav4 | AEAD | `aegis-256/decrypt vs aegis-crate` | `65536` | 13.30 us | aegis-crate | 12.49 us |
| 0.94x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `16384` | 3.27 us | aegis-crate | 3.07 us |
| 0.94x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `65536` | 12.73 us | aegis-crate | 11.95 us |
| 0.94x | macOS-A64 | XXH3 | `xxh3-64 vs xxhash-rust` | `16384` | 496.84 ns | xxhash-rust | 466.64 ns |
| 0.94x | ICL | XXH3 | `xxh3-64 vs xxhash-rust` | `256` | 14.68 ns | xxhash-rust | 13.79 ns |
| 0.94x | Zen5 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `262144` | 9.62 us | rapidhash | 9.03 us |
| 0.94x | Grav4 | RapidHash | `rapidhash-64 vs rapidhash` | `1` | 1.69 ns | rapidhash | 1.59 ns |
| 0.94x | Zen4 | BLAKE2 | `blake2/blake2s256 vs rustcrypto` | `0` | 130.32 ns | rustcrypto | 122.52 ns |
| 0.94x | macOS-A64 | BLAKE2 | `blake2/host-overhead/blake2s256 vs rustcrypto` | `32` | 109.21 ns | rustcrypto | 102.68 ns |
| 0.94x | POWER10 | BLAKE2 | `blake2/keyed/blake2b512 vs rustcrypto` | `64` | 403.91 ns | rustcrypto | 379.81 ns |
| 0.94x | Grav4 | AEAD | `aegis-256/decrypt vs aegis-crate` | `1048576` | 209.22 us | aegis-crate | 196.83 us |
| 0.94x | Zen5 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `65536` | 2.43 us | rapidhash | 2.29 us |
| 0.94x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `4096` | 104.00 ns | rapidhash | 97.87 ns |
| 0.94x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1024` | 284.74 ns | aegis-crate | 267.98 ns |
| 0.94x | ICL | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `64B` | 956.85 us | rustcrypto | 900.98 us |
| 0.94x | RISC-V | BLAKE2 | `blake2/keyed/blake2b256 vs rustcrypto` | `262144` | 2.14 ms | rustcrypto | 2.02 ms |
| 0.94x | Zen5 | BLAKE3 | `blake3/xof vs blake3` | `1` | 102.54 ns | blake3 | 96.59 ns |
| 0.94x | Zen5 | BLAKE3 | `blake3/xof vs blake3` | `32` | 102.53 ns | blake3 | 96.61 ns |
| 0.94x | POWER10 | BLAKE3 | `blake3 vs blake3` | `1` | 123.26 ns | blake3 | 116.21 ns |
| 0.94x | Grav3 | Checksums | `crc32c vs crc-fast` | `16384` | 505.24 ns | crc-fast | 476.50 ns |
| 0.94x | Grav4 | AEAD | `aegis-256/decrypt vs aegis-crate` | `262144` | 51.71 us | aegis-crate | 48.77 us |
| 0.94x | SPR | BLAKE2 | `blake2/blake2b256 vs rustcrypto` | `32` | 159.44 ns | rustcrypto | 150.38 ns |
| 0.94x | POWER10 | BLAKE3 | `blake3/keyed vs blake3` | `1` | 123.13 ns | blake3 | 116.15 ns |
| 0.94x | RISC-V | BLAKE2 | `blake2/keyed/blake2s128 vs rustcrypto` | `32` | 1.55 us | rustcrypto | 1.47 us |
| 0.94x | macOS-A64 | Auth/KDF | `hmac-sha512 vs rustcrypto` | `1024` | 1.07 us | rustcrypto | 1.01 us |
| 0.94x | SPR | Auth/KDF | `hmac-sha256 vs rustcrypto` | `1048576` | 640.89 us | rustcrypto | 605.31 us |
| 0.94x | RISC-V | BLAKE2 | `blake2/blake2s128 vs rustcrypto` | `262144` | 3.12 ms | rustcrypto | 2.95 ms |
| 0.95x | macOS-A64 | Password Hashing | `argon2i-small vs rustcrypto` | `m=8_t=1_p=1` | 14.44 us | rustcrypto | 13.65 us |
| 0.95x | macOS-A64 | BLAKE2 | `blake2/host-overhead/blake2b256 vs rustcrypto` | `0` | 127.18 ns | rustcrypto | 120.21 ns |
| 0.95x | ICL | BLAKE3 | `blake3/xof vs blake3` | `0` | 56.82 ns | blake3 | 53.73 ns |
| 0.95x | Zen5 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `4096` | 79.41 ns | rapidhash | 75.12 ns |
| 0.95x | Grav4 | Checksums | `crc64-nvme vs crc64fast-nvme` | `4096` | 106.37 ns | crc64fast-nvme | 100.67 ns |
| 0.95x | ICL | SHA-2 | `sha256/streaming vs sha2` | `64B` | 959.55 us | sha2 | 908.20 us |
| 0.95x | RISC-V | BLAKE2 | `blake2/streaming/blake2b256 vs rustcrypto` | `64B` | 7.96 ms | rustcrypto | 7.54 ms |
| 0.95x | POWER10 | BLAKE2 | `blake2/blake2b512 vs rustcrypto` | `65536` | 93.65 us | rustcrypto | 88.67 us |
| 0.95x | RISC-V | BLAKE3 | `blake3 vs blake3` | `4096` | 32.68 us | blake3 | 30.94 us |
| 0.95x | Grav3 | Auth/KDF | `pbkdf2-sha512/iters=1000 vs rustcrypto` | `64` | 287.80 us | rustcrypto | 272.52 us |
| 0.95x | Zen5 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `4096` | 161.80 ns | rapidhash | 153.22 ns |
| 0.95x | Grav3 | Auth/KDF | `pbkdf2-sha512/iters=100 vs rustcrypto` | `128` | 57.87 us | rustcrypto | 54.81 us |
| 0.95x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1024` | 55.71 ns | rapidhash | 52.77 ns |
| 0.95x | Zen4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `0` | 35.68 ns | aegis-crate | 33.80 ns |
| 0.95x | macOS-A64 | SHAKE | `shake128 vs sha3` | `256` | 333.46 ns | sha3 | 315.95 ns |
| 0.95x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `4096` | 402.22 ns | rapidhash | 381.16 ns |
| 0.95x | RISC-V | Checksums | `crc32 vs crc32fast` | `65536` | 103.67 us | crc32fast | 98.26 us |
| 0.95x | RISC-V | BLAKE2 | `blake2/keyed/blake2s128 vs rustcrypto` | `64` | 1.56 us | rustcrypto | 1.48 us |
| 0.95x | Grav3 | Checksums | `crc32 vs crc-fast` | `1048576` | 25.97 us | crc-fast | 24.62 us |
| 0.95x | Grav3 | Auth/KDF | `pbkdf2-sha512/iters=1000 vs rustcrypto` | `128` | 575.22 us | rustcrypto | 545.42 us |
| 0.95x | macOS-A64 | SHA-3 | `sha3-384 vs sha3` | `32` | 168.48 ns | sha3 | 159.78 ns |
| 0.95x | Grav3 | Auth/KDF | `pbkdf2-sha512/iters=100 vs rustcrypto` | `64` | 29.07 us | rustcrypto | 27.57 us |
| 0.95x | RISC-V | Auth/KDF | `hmac-sha512 vs rustcrypto` | `262144` | 3.71 ms | rustcrypto | 3.52 ms |
| 0.95x | RISC-V | BLAKE2 | `blake2/keyed/blake2s256 vs rustcrypto` | `1048576` | 10.61 ms | rustcrypto | 10.06 ms |
| 0.95x | Grav4 | AEAD | `aegis-256/decrypt vs aegis-crate` | `16384` | 3.23 us | aegis-crate | 3.07 us |
| 0.95x | Zen5 | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `1` | 74.64 ns | rustcrypto | 70.83 ns |
| 0.95x | RISC-V | BLAKE2 | `blake2/keyed/blake2s256 vs rustcrypto` | `262144` | 3.14 ms | rustcrypto | 2.98 ms |
| 0.95x | Zen4 | Checksums | `crc32 vs crc-fast` | `1024` | 31.66 ns | crc-fast | 30.06 ns |
| 0.95x | macOS-A64 | BLAKE3 | `blake3/keyed vs blake3` | `64` | 75.02 ns | blake3 | 71.23 ns |
| 0.95x | Grav3 | XXH3 | `xxh3-128 vs xxhash-rust` | `4096` | 233.06 ns | xxhash-rust | 221.32 ns |
| 0.95x | POWER10 | BLAKE2 | `blake2/streaming/blake2b256 vs rustcrypto` | `64B` | 1.57 ms | rustcrypto | 1.49 ms |
| 0.95x | Grav3 | XXH3 | `xxh3-64 vs xxhash-rust` | `65536` | 3.46 us | xxhash-rust | 3.29 us |
| 0.95x | POWER10 | BLAKE2 | `blake2/keyed/blake2b512 vs rustcrypto` | `1` | 396.50 ns | rustcrypto | 376.62 ns |
| 0.95x | RISC-V | BLAKE2 | `blake2/keyed/blake2s256 vs rustcrypto` | `1` | 1.55 us | rustcrypto | 1.47 us |

## Absolute-Only Benches

These Criterion groups do not have a same-path external competitor in every run, so they are reported as raw `rscrypto` midpoints rather than speedups.

| Platform | Category | Benchmark | Time |
|---|---|---|---:|
| macOS-A64 | Ascon | `ascon-hash256/many/rscrypto/batch-auto` | 1.67 ms |
| macOS-A64 | Ascon | `ascon-hash256/streaming/rscrypto/4096B` | 6.12 ms |
| macOS-A64 | Ascon | `ascon-hash256/streaming/rscrypto/64B` | 6.11 ms |
| macOS-A64 | Ascon | `ascon-xof128/many/rscrypto/batch-auto` | 1.69 ms |
| macOS-A64 | Auth/KDF | `pbkdf2-sha256-state/iters=1/rscrypto/32` | 60.23 ns |
| macOS-A64 | Auth/KDF | `pbkdf2-sha256-state/iters=1/rscrypto/64` | 118.76 ns |
| macOS-A64 | Auth/KDF | `pbkdf2-sha256-state/iters=100/rscrypto/32` | 7.07 us |
| macOS-A64 | Auth/KDF | `pbkdf2-sha256-state/iters=100/rscrypto/64` | 14.14 us |
| macOS-A64 | Auth/KDF | `pbkdf2-sha256-state/iters=1000/rscrypto/32` | 70.17 us |
| macOS-A64 | Auth/KDF | `pbkdf2-sha256-state/iters=1000/rscrypto/64` | 141.04 us |
| macOS-A64 | Auth/KDF | `pbkdf2-sha512-state/iters=1/rscrypto/128` | 358.43 ns |
| macOS-A64 | Auth/KDF | `pbkdf2-sha512-state/iters=1/rscrypto/64` | 199.41 ns |
| macOS-A64 | Auth/KDF | `pbkdf2-sha512-state/iters=100/rscrypto/128` | 38.34 us |
| macOS-A64 | Auth/KDF | `pbkdf2-sha512-state/iters=100/rscrypto/64` | 19.11 us |
| macOS-A64 | Auth/KDF | `pbkdf2-sha512-state/iters=1000/rscrypto/128` | 381.37 us |
| macOS-A64 | Auth/KDF | `pbkdf2-sha512-state/iters=1000/rscrypto/64` | 190.70 us |
| macOS-A64 | BLAKE2 | `blake2/params/rscrypto/blake2b256/plain/64` | 120.95 ns |
| macOS-A64 | BLAKE2 | `blake2/params/rscrypto/blake2b256/salt+personal/64` | 126.35 ns |
| macOS-A64 | BLAKE2 | `blake2/params/rscrypto/blake2s256/plain/64` | 100.12 ns |
| macOS-A64 | BLAKE2 | `blake2/params/rscrypto/blake2s256/salt+personal/64` | 105.71 ns |
| macOS-A64 | Password Hashing | `argon2id-parallel-owasp/rscrypto/p=1` | 13.78 ms |
| macOS-A64 | Password Hashing | `argon2id-parallel-owasp/rscrypto/p=16` | 3.97 ms |
| macOS-A64 | Password Hashing | `argon2id-parallel-owasp/rscrypto/p=4` | 5.03 ms |
| macOS-A64 | Password Hashing | `argon2id-parallel-owasp/rscrypto/p=8` | 3.78 ms |
| macOS-A64 | Password Hashing | `argon2id-parallel/rscrypto/p=1` | 2.79 ms |
| macOS-A64 | Password Hashing | `argon2id-parallel/rscrypto/p=16` | 1.38 ms |
| macOS-A64 | Password Hashing | `argon2id-parallel/rscrypto/p=4` | 1.50 ms |
| macOS-A64 | Password Hashing | `argon2id-parallel/rscrypto/p=8` | 1.34 ms |

## Internal Benches

These compare two rscrypto-internal benchmark modes, not rscrypto against an external crate; they are excluded from the scoreboard above. Ratio is `alternate_time / primary_time`.

| Platform | Benchmark | Primary | Alternate | Ratio |
|---|---|---:|---:|---:|
| Zen4 | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 4.35 ms | 4.42 ms | 1.02x |
| Zen4 | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 571.39 ns | 594.33 ns | 1.04x |
| Zen5 | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 3.14 ms | 3.21 ms | 1.02x |
| Zen5 | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 418.34 ns | 427.16 ns | 1.02x |
| Grav3 | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 4.80 ms | 5.20 ms | 1.08x |
| Grav3 | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 623.89 ns | 689.22 ns | 1.10x |
| Grav4 | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 4.41 ms | 4.42 ms | 1.00x |
| Grav4 | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 572.04 ns | 615.61 ns | 1.08x |
| POWER10 | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 4.39 ms | 4.43 ms | 1.01x |
| POWER10 | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 593.55 ns | 603.49 ns | 1.02x |
| s390x | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 7.82 ms | 540.85 us | 0.07x |
| s390x | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 1.05 us | 1.07 us | 1.02x |
| ICL | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 5.11 ms | 5.16 ms | 1.01x |
| ICL | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 674.06 ns | 692.51 ns | 1.03x |
| SPR | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 4.00 ms | 3.96 ms | 0.99x |
| SPR | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 533.26 ns | 539.63 ns | 1.01x |
| RISC-V | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 23.86 ms | 24.41 ms | 1.02x |
| RISC-V | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 3.00 us | 3.14 us | 1.05x |
| macOS-A64 | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 1.40 ms | 2.61 ms | 1.87x |
| macOS-A64 | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 181.05 ns | 320.47 ns | 1.77x |

## Raw Results

- `benchmark_results/2026-04-26/linux/amd-zen4/results.txt`
- `benchmark_results/2026-04-26/linux/amd-zen5/results.txt`
- `benchmark_results/2026-04-26/linux/intel-spr/results.txt`
- `benchmark_results/2026-04-26/linux/intel-icl/results.txt`
- `benchmark_results/2026-04-26/linux/graviton3/results.txt`
- `benchmark_results/2026-04-26/linux/graviton4/results.txt`
- `benchmark_results/2026-04-26/linux/ibm-s390x/results.txt`
- `benchmark_results/2026-04-26/linux/ibm-power10/results.txt`
- `benchmark_results/2026-04-26/linux/rise-riscv/results.txt`
- `benchmark_results/2026-04-26/macos/aarch64/results.txt`
