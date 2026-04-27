# Benchmark Overview

Source: Bench CI run [#24974478214](https://github.com/loadingalias/rscrypto/actions/runs/24974478214) created 2026-04-27 03:01:46 UTC.
Commit: `e65388d85f72ac813d0fd29a28e01c8f1a623f70`

Scope: nine Linux CI runners plus the local macOS Apple Silicon result at `benchmark_results/2026-04-27/macos/aarch64/results.txt`. The macOS run is included because it is on the same commit as CI. Speedup is `competitor_time / rscrypto_time`; values above `1.00x` mean `rscrypto` is faster. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`.

## Executive Read

Overall: **3969W / 2174T / 348L = 61% win rate** across 6491 paired external comparisons. Geomean speedup is **1.72x** and median speedup is **1.12x**.
Linux CI alone: **3642W / 1865T / 289L**, geomean **1.75x**, median **1.14x**. macOS Apple Silicon: **327W / 309T / 59L**, geomean **1.52x**, median **1.04x**.
Strongest category by geomean: **Checksums** at 4.40x. Weakest category by geomean: **Password Hashing** at 0.76x.
Checksums: 889W/158T/53L, geomean 4.40x, worst 0.56x on RISC-V `crc32c vs crc32c` `32`.
SHA-2: 279W/299T/12L, geomean 1.39x, worst 0.01x on SPR `sha256/streaming vs sha2` `64B`.
SHA-3: 397W/54T/9L, geomean 2.03x, worst 0.82x on macOS `sha3-512 vs sha3` `32`.
SHAKE: 197W/15T/8L, geomean 2.38x, worst 0.65x on macOS `shake256 vs sha3` `1`.
BLAKE2: 411W/539T/26L, geomean 1.06x, worst 0.49x on macOS `blake2 vs rustcrypto` `blake2b256/256`.
BLAKE3: 242W/203T/35L, geomean 1.39x, worst 0.73x on macOS `blake3/keyed vs blake3` `64`.
Ascon: 142W/78T/0L, geomean 1.29x, worst 0.95x on RISC-V `ascon-hash256 vs ascon-hash` `1`.
XXH3: 108W/79T/33L, geomean 1.16x, worst 0.38x on SPR `xxh3-64 vs xxhash-rust` `256`.
RapidHash: 120W/257T/63L, geomean 1.06x, worst 0.52x on s390x `rapidhash-v3-64 vs rapidhash` `1`.
Auth/KDF: 329W/302T/39L, geomean 1.23x, worst 0.01x on SPR `hmac-sha256/streaming vs rustcrypto` `64B`.
Password Hashing: 5W/6T/4L, geomean 0.76x, worst 0.21x on macOS `argon2d-small vs rustcrypto` `m=64_t=3_p=2`.
AEAD: 850W/184T/66L, geomean 1.90x, worst 0.45x on ICL `aes-256-gcm/encrypt vs rustcrypto` `0`.

## Platforms

| Short | Platform | Architecture | Source | Header Time | Pairs | Results |
|---|---|---|---|---|---:|---|
| Zen4 | AMD EPYC Zen 4 | x86-64 | ci | `03_01_46` | 644 | `benchmark_results/2026-04-27/linux/amd-zen4/results.txt` |
| Zen5 | AMD EPYC Zen 5 | x86-64 | ci | `03_01_46` | 644 | `benchmark_results/2026-04-27/linux/amd-zen5/results.txt` |
| SPR | Intel Xeon Sapphire Rapids | x86-64 | ci | `03_01_46` | 644 | `benchmark_results/2026-04-27/linux/intel-spr/results.txt` |
| ICL | Intel Xeon Ice Lake | x86-64 | ci | `03_01_46` | 644 | `benchmark_results/2026-04-27/linux/intel-icl/results.txt` |
| Grav3 | AWS Graviton3 | aarch64 | ci | `03_01_46` | 644 | `benchmark_results/2026-04-27/linux/graviton3/results.txt` |
| Grav4 | AWS Graviton4 | aarch64 | ci | `03_01_46` | 644 | `benchmark_results/2026-04-27/linux/graviton4/results.txt` |
| s390x | IBM Z | s390x | ci | `03_01_46` | 644 | `benchmark_results/2026-04-27/linux/ibm-s390x/results.txt` |
| POWER10 | IBM POWER10 | ppc64le | ci | `03_01_46` | 644 | `benchmark_results/2026-04-27/linux/ibm-power10/results.txt` |
| RISC-V | RISE RISC-V | riscv64 | ci | `03_01_46` | 644 | `benchmark_results/2026-04-27/linux/rise-riscv/results.txt` |
| macOS | Apple Silicon local | aarch64 | local | `07_57_38` | 695 | `benchmark_results/2026-04-27/macos/aarch64/results.txt` |

## Overall Scoreboard

| W | T | L | Total | Win % | Geomean | Median | Worst | Worst Platform | Worst Case | Best | Best Platform | Best Case |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| 3969 | 2174 | 348 | 6491 | 61% | 1.72x | 1.12x | 0.01x | SPR | `sha256/streaming vs sha2` `64B` | 213.39x | SPR | `crc16-ibm vs crc` `262144` |

## By Category

| Category | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Platform | Worst Case | Best | Best Platform | Best Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Checksums | 889 | 158 | 53 | 1100 | 81% | 4.40x | 2.28x | 0.56x | RISC-V | `crc32c vs crc32c` `32` | 213.39x | SPR | `crc16-ibm vs crc` `262144` |
| SHA-2 | 279 | 299 | 12 | 590 | 47% | 1.39x | 1.05x | 0.01x | SPR | `sha256/streaming vs sha2` `64B` | 119.18x | SPR | `sha224 vs sha2` `65536` |
| SHA-3 | 397 | 54 | 9 | 460 | 86% | 2.03x | 2.00x | 0.82x | macOS | `sha3-512 vs sha3` `32` | 24.99x | s390x | `sha3-512 vs sha3` `262144` |
| SHAKE | 197 | 15 | 8 | 220 | 90% | 2.38x | 2.12x | 0.65x | macOS | `shake256 vs sha3` `1` | 16.53x | s390x | `shake256 vs sha3` `1048576` |
| BLAKE2 | 411 | 539 | 26 | 976 | 42% | 1.06x | 1.03x | 0.49x | macOS | `blake2 vs rustcrypto` `blake2b256/256` | 4.09x | macOS | `blake2 vs rustcrypto` `blake2s128/64` |
| BLAKE3 | 242 | 203 | 35 | 480 | 50% | 1.39x | 1.05x | 0.73x | macOS | `blake3/keyed vs blake3` `64` | 8.38x | s390x | `blake3/keyed vs blake3` `262144` |
| Ascon | 142 | 78 | 0 | 220 | 65% | 1.29x | 1.12x | 0.95x | RISC-V | `ascon-hash256 vs ascon-hash` `1` | 2.51x | Zen5 | `ascon-xof128 vs ascon-hash` `0` |
| XXH3 | 108 | 79 | 33 | 220 | 49% | 1.16x | 1.04x | 0.38x | SPR | `xxh3-64 vs xxhash-rust` `256` | 3.19x | s390x | `xxh3-64 vs xxhash-rust` `65536` |
| RapidHash | 120 | 257 | 63 | 440 | 27% | 1.06x | 1.00x | 0.52x | s390x | `rapidhash-v3-64 vs rapidhash` `1` | 2.28x | Grav3 | `rapidhash-v3-128 vs rapidhash` `32` |
| Auth/KDF | 329 | 302 | 39 | 670 | 49% | 1.23x | 1.05x | 0.01x | SPR | `hmac-sha256/streaming vs rustcrypto` `64B` | 82.78x | SPR | `hkdf-sha256/expand vs rustcrypto` `32` |
| Password Hashing | 5 | 6 | 4 | 15 | 33% | 0.76x | 0.99x | 0.21x | macOS | `argon2d-small vs rustcrypto` `m=64_t=3_p=2` | 1.15x | macOS | `scrypt-small vs rustcrypto` `log_n=10_r=8_p=4` |
| AEAD | 850 | 184 | 66 | 1100 | 77% | 1.90x | 1.42x | 0.45x | ICL | `aes-256-gcm/encrypt vs rustcrypto` `0` | 32.10x | s390x | `aes-256-gcm-siv/encrypt vs rustcrypto` `0` |

## By Platform

| Platform | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Category | Worst Case | Best | Best Category | Best Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Zen4 | 476 | 150 | 18 | 644 | 74% | 1.62x | 1.20x | 0.49x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `32` | 129.45x | Checksums | `crc16-ibm vs crc` `262144` |
| Zen5 | 383 | 235 | 26 | 644 | 59% | 1.65x | 1.12x | 0.50x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `32` | 129.43x | Checksums | `crc16-ccitt vs crc` `262144` |
| SPR | 495 | 104 | 45 | 644 | 77% | 1.91x | 1.27x | 0.01x | SHA-2 | `sha256/streaming vs sha2` `64B` | 213.39x | Checksums | `crc16-ibm vs crc` `262144` |
| ICL | 477 | 146 | 21 | 644 | 74% | 1.62x | 1.22x | 0.45x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `0` | 119.74x | Checksums | `crc16-ibm vs crc` `262144` |
| Grav3 | 334 | 267 | 43 | 644 | 52% | 1.64x | 1.06x | 0.77x | AEAD | `aegis-256/encrypt vs aegis-crate` `0` | 127.48x | Checksums | `crc16-ibm vs crc` `262144` |
| Grav4 | 323 | 297 | 24 | 644 | 50% | 1.65x | 1.05x | 0.78x | AEAD | `aegis-256/encrypt vs aegis-crate` `0` | 102.60x | Checksums | `crc16-ibm vs crc` `262144` |
| s390x | 527 | 89 | 28 | 644 | 82% | 3.02x | 2.34x | 0.52x | RapidHash | `rapidhash-v3-64 vs rapidhash` `1` | 101.36x | Checksums | `crc16-ccitt vs crc` `65536` |
| POWER10 | 328 | 286 | 30 | 644 | 51% | 1.93x | 1.06x | 0.74x | RapidHash | `rapidhash-v3-128 vs rapidhash` `1` | 179.03x | Checksums | `crc16-ibm vs crc` `262144` |
| RISC-V | 299 | 291 | 54 | 644 | 46% | 1.17x | 1.04x | 0.56x | Checksums | `crc32c vs crc32c` `32` | 10.56x | Checksums | `crc64-nvme vs crc-fast` `0` |
| macOS | 327 | 309 | 59 | 695 | 47% | 1.52x | 1.04x | 0.21x | Password Hashing | `argon2d-small vs rustcrypto` `m=64_t=3_p=2` | 171.15x | Checksums | `crc16-ccitt vs crc` `1048576` |

## Platform x Category

| Platform | Category | W | T | L | Total | Win % | Geomean | Worst | Worst Case |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zen4 | Checksums | 90 | 18 | 2 | 110 | 82% | 4.43x | 0.95x | `crc32 vs crc-fast` `1024` |
| Zen4 | SHA-2 | 33 | 26 | 0 | 59 | 56% | 1.12x | 0.99x | `sha256/streaming vs sha2` `64B` |
| Zen4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.29x | 2.24x | `sha3-256 vs sha3` `65536` |
| Zen4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.06x | 2.24x | `shake256 vs sha3` `1048576` |
| Zen4 | BLAKE2 | 64 | 30 | 0 | 94 | 68% | 1.09x | 1.00x | `blake2/streaming vs rustcrypto` `blake2s256/64B` |
| Zen4 | BLAKE3 | 24 | 22 | 2 | 48 | 50% | 1.34x | 0.94x | `blake3/xof vs blake3` `64` |
| Zen4 | Ascon | 22 | 0 | 0 | 22 | 100% | 1.93x | 1.67x | `ascon-hash256 vs ascon-hash` `1` |
| Zen4 | XXH3 | 13 | 4 | 5 | 22 | 59% | 1.03x | 0.70x | `xxh3-64 vs xxhash-rust` `256` |
| Zen4 | RapidHash | 12 | 32 | 0 | 44 | 27% | 1.07x | 0.95x | `rapidhash-v3-64 vs rapidhash` `16384` |
| Zen4 | Auth/KDF | 56 | 7 | 4 | 67 | 84% | 1.20x | 0.92x | `pbkdf2-sha256/iters=1000 vs rustcrypto` `32` |
| Zen4 | AEAD | 94 | 11 | 5 | 110 | 85% | 1.27x | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| Zen5 | Checksums | 91 | 18 | 1 | 110 | 83% | 4.81x | 0.94x | `crc32c vs crc-fast` `1024` |
| Zen5 | SHA-2 | 25 | 34 | 0 | 59 | 42% | 1.09x | 1.00x | `sha256/streaming vs sha2` `64B` |
| Zen5 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.18x | 2.12x | `sha3-256/streaming vs sha3` `64B` |
| Zen5 | SHAKE | 22 | 0 | 0 | 22 | 100% | 2.91x | 2.15x | `shake128 vs sha3` `65536` |
| Zen5 | BLAKE2 | 13 | 81 | 0 | 94 | 14% | 1.01x | 0.97x | `blake2/keyed vs rustcrypto` `blake2s256/65536` |
| Zen5 | BLAKE3 | 21 | 24 | 3 | 48 | 44% | 1.35x | 0.85x | `blake3/keyed vs blake3` `256` |
| Zen5 | Ascon | 22 | 0 | 0 | 22 | 100% | 2.17x | 1.84x | `ascon-hash256 vs ascon-hash` `1` |
| Zen5 | XXH3 | 13 | 6 | 3 | 22 | 59% | 1.26x | 0.76x | `xxh3-128 vs xxhash-rust` `256` |
| Zen5 | RapidHash | 13 | 24 | 7 | 44 | 30% | 1.05x | 0.94x | `rapidhash-v3-64 vs rapidhash` `65536` |
| Zen5 | Auth/KDF | 27 | 33 | 7 | 67 | 40% | 1.09x | 0.92x | `ed25519/verify vs dalek` `0` |
| Zen5 | AEAD | 90 | 15 | 5 | 110 | 82% | 1.41x | 0.50x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| SPR | Checksums | 95 | 11 | 4 | 110 | 86% | 4.84x | 0.84x | `crc32 vs crc32fast` `256` |
| SPR | SHA-2 | 53 | 4 | 2 | 59 | 90% | 3.26x | 0.01x | `sha256/streaming vs sha2` `64B` |
| SPR | SHA-3 | 46 | 0 | 0 | 46 | 100% | 3.58x | 3.38x | `sha3-256/streaming vs sha3` `64B` |
| SPR | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.79x | 3.49x | `shake128 vs sha3` `1048576` |
| SPR | BLAKE2 | 92 | 2 | 0 | 94 | 98% | 1.22x | 1.01x | `blake2 vs rustcrypto` `blake2b512/1` |
| SPR | BLAKE3 | 19 | 23 | 6 | 48 | 40% | 1.26x | 0.87x | `blake3/xof vs blake3` `64` |
| SPR | Ascon | 22 | 0 | 0 | 22 | 100% | 1.47x | 1.28x | `ascon-hash256 vs ascon-hash` `1` |
| SPR | XXH3 | 12 | 6 | 4 | 22 | 55% | 1.04x | 0.38x | `xxh3-64 vs xxhash-rust` `256` |
| SPR | RapidHash | 10 | 21 | 13 | 44 | 23% | 1.03x | 0.78x | `rapidhash-64 vs rapidhash` `0` |
| SPR | Auth/KDF | 41 | 18 | 8 | 67 | 61% | 1.21x | 0.01x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| SPR | AEAD | 83 | 19 | 8 | 110 | 75% | 1.29x | 0.48x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| ICL | Checksums | 89 | 19 | 2 | 110 | 81% | 4.00x | 0.74x | `crc32 vs crc32fast` `256` |
| ICL | SHA-2 | 36 | 21 | 2 | 59 | 61% | 1.12x | 0.82x | `sha256/streaming vs sha2` `4096B` |
| ICL | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.99x | 2.88x | `sha3-256/streaming vs sha3` `64B` |
| ICL | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.94x | 2.91x | `shake128 vs sha3` `262144` |
| ICL | BLAKE2 | 94 | 0 | 0 | 94 | 100% | 1.20x | 1.10x | `blake2/keyed vs rustcrypto` `blake2b512/1048576` |
| ICL | BLAKE3 | 20 | 24 | 4 | 48 | 42% | 1.17x | 0.83x | `blake3/keyed vs blake3` `256` |
| ICL | Ascon | 22 | 0 | 0 | 22 | 100% | 1.40x | 1.25x | `ascon-hash256 vs ascon-hash` `1` |
| ICL | XXH3 | 13 | 5 | 4 | 22 | 59% | 1.26x | 0.52x | `xxh3-64 vs xxhash-rust` `64` |
| ICL | RapidHash | 13 | 30 | 1 | 44 | 30% | 1.08x | 0.89x | `rapidhash-64 vs rapidhash` `32` |
| ICL | Auth/KDF | 36 | 28 | 3 | 67 | 54% | 1.10x | 0.83x | `hmac-sha256/streaming vs rustcrypto` `4096B` |
| ICL | AEAD | 86 | 19 | 5 | 110 | 78% | 1.25x | 0.45x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| Grav3 | Checksums | 80 | 23 | 7 | 110 | 73% | 3.68x | 0.92x | `crc32c vs crc-fast` `262144` |
| Grav3 | SHA-2 | 22 | 36 | 1 | 59 | 37% | 1.06x | 0.91x | `sha256/streaming vs sha2` `64B` |
| Grav3 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.81x | 1.74x | `sha3-256/streaming vs sha3` `64B` |
| Grav3 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.82x | 1.77x | `shake256 vs sha3` `262144` |
| Grav3 | BLAKE2 | 29 | 65 | 0 | 94 | 31% | 1.04x | 1.00x | `blake2/streaming vs rustcrypto` `blake2s256/64B` |
| Grav3 | BLAKE3 | 21 | 22 | 5 | 48 | 44% | 1.41x | 0.93x | `blake3/keyed vs blake3` `1` |
| Grav3 | Ascon | 4 | 18 | 0 | 22 | 18% | 1.03x | 0.99x | `ascon-xof128 vs ascon-hash` `262144` |
| Grav3 | XXH3 | 5 | 14 | 3 | 22 | 23% | 1.07x | 0.87x | `xxh3-128 vs xxhash-rust` `256` |
| Grav3 | RapidHash | 12 | 18 | 14 | 44 | 27% | 1.06x | 0.90x | `rapidhash-v3-64 vs rapidhash` `256` |
| Grav3 | Auth/KDF | 29 | 31 | 7 | 67 | 43% | 1.09x | 0.90x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| Grav3 | AEAD | 64 | 40 | 6 | 110 | 58% | 2.48x | 0.77x | `aegis-256/encrypt vs aegis-crate` `0` |
| Grav4 | Checksums | 79 | 28 | 3 | 110 | 72% | 3.66x | 0.81x | `crc32c vs crc-fast` `256` |
| Grav4 | SHA-2 | 20 | 37 | 2 | 59 | 34% | 1.05x | 0.90x | `sha256/streaming vs sha2` `64B` |
| Grav4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.97x | 1.88x | `sha3-256/streaming vs sha3` `64B` |
| Grav4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.99x | 1.94x | `shake256 vs sha3` `65536` |
| Grav4 | BLAKE2 | 21 | 73 | 0 | 94 | 22% | 1.02x | 0.99x | `blake2/keyed vs rustcrypto` `blake2b512/1024` |
| Grav4 | BLAKE3 | 22 | 25 | 1 | 48 | 46% | 1.47x | 0.93x | `blake3/keyed vs blake3` `1` |
| Grav4 | Ascon | 12 | 10 | 0 | 22 | 55% | 1.08x | 1.01x | `ascon-hash256 vs ascon-hash` `1` |
| Grav4 | XXH3 | 5 | 13 | 4 | 22 | 23% | 1.03x | 0.79x | `xxh3-128 vs xxhash-rust` `256` |
| Grav4 | RapidHash | 12 | 29 | 3 | 44 | 27% | 1.07x | 0.89x | `rapidhash-v3-128 vs rapidhash` `256` |
| Grav4 | Auth/KDF | 18 | 47 | 2 | 67 | 27% | 1.07x | 0.87x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| Grav4 | AEAD | 66 | 35 | 9 | 110 | 60% | 2.46x | 0.78x | `aegis-256/encrypt vs aegis-crate` `0` |
| s390x | Checksums | 98 | 3 | 9 | 110 | 89% | 7.21x | 0.63x | `crc24-openpgp vs crc` `64` |
| s390x | SHA-2 | 59 | 0 | 0 | 59 | 100% | 5.21x | 1.92x | `sha256 vs sha2` `64` |
| s390x | SHA-3 | 46 | 0 | 0 | 46 | 100% | 4.69x | 1.13x | `sha3-256/streaming vs sha3` `64B` |
| s390x | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.93x | 1.10x | `shake128 vs sha3` `64` |
| s390x | BLAKE2 | 53 | 41 | 0 | 94 | 56% | 1.09x | 0.97x | `blake2 vs rustcrypto` `blake2b256/262144` |
| s390x | BLAKE3 | 41 | 3 | 4 | 48 | 85% | 1.87x | 0.73x | `blake3/xof vs blake3` `1024` |
| s390x | Ascon | 4 | 18 | 0 | 22 | 18% | 1.05x | 0.99x | `ascon-hash256 vs ascon-hash` `16384` |
| s390x | XXH3 | 17 | 2 | 3 | 22 | 77% | 1.70x | 0.75x | `xxh3-64 vs xxhash-rust` `32` |
| s390x | RapidHash | 13 | 19 | 12 | 44 | 30% | 1.02x | 0.52x | `rapidhash-v3-64 vs rapidhash` `1` |
| s390x | Auth/KDF | 64 | 3 | 0 | 67 | 96% | 3.48x | 1.01x | `ed25519/verify vs dalek` `0` |
| s390x | AEAD | 110 | 0 | 0 | 110 | 100% | 4.32x | 1.17x | `xchacha20-poly1305/encrypt vs rustcrypto` `1` |
| POWER10 | Checksums | 103 | 4 | 3 | 110 | 94% | 10.75x | 0.78x | `crc64-nvme vs crc64fast-nvme` `64` |
| POWER10 | SHA-2 | 5 | 54 | 0 | 59 | 8% | 1.01x | 0.98x | `sha384 vs sha2` `262144` |
| POWER10 | SHA-3 | 17 | 28 | 1 | 46 | 37% | 1.04x | 0.95x | `sha3-256/streaming vs sha3` `64B` |
| POWER10 | SHAKE | 15 | 7 | 0 | 22 | 68% | 1.39x | 1.00x | `shake256 vs sha3` `1048576` |
| POWER10 | BLAKE2 | 11 | 73 | 10 | 94 | 12% | 1.00x | 0.92x | `blake2/keyed vs rustcrypto` `blake2b512/32` |
| POWER10 | BLAKE3 | 36 | 12 | 0 | 48 | 75% | 2.03x | 0.95x | `blake3/keyed vs blake3` `1` |
| POWER10 | Ascon | 5 | 17 | 0 | 22 | 23% | 1.06x | 1.01x | `ascon-hash256 vs ascon-hash` `16384` |
| POWER10 | XXH3 | 19 | 3 | 0 | 22 | 86% | 1.29x | 0.98x | `xxh3-128 vs xxhash-rust` `64` |
| POWER10 | RapidHash | 9 | 26 | 9 | 44 | 20% | 1.04x | 0.74x | `rapidhash-v3-128 vs rapidhash` `1` |
| POWER10 | Auth/KDF | 12 | 55 | 0 | 67 | 18% | 1.05x | 0.96x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| POWER10 | AEAD | 96 | 7 | 7 | 110 | 87% | 2.64x | 0.92x | `aegis-256/encrypt vs aegis-crate` `16384` |
| RISC-V | Checksums | 75 | 16 | 19 | 110 | 68% | 1.49x | 0.56x | `crc32c vs crc32c` `32` |
| RISC-V | SHA-2 | 14 | 45 | 0 | 59 | 24% | 1.03x | 0.97x | `sha256/streaming vs sha2` `4096B` |
| RISC-V | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.17x | 1.07x | `sha3-256 vs sha3` `65536` |
| RISC-V | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.61x | 1.16x | `shake256 vs sha3` `16384` |
| RISC-V | BLAKE2 | 10 | 79 | 5 | 94 | 11% | 1.00x | 0.85x | `blake2 vs rustcrypto` `blake2b256/1048576` |
| RISC-V | BLAKE3 | 21 | 23 | 4 | 48 | 44% | 1.10x | 0.75x | `blake3/streaming vs blake3` `16384B` |
| RISC-V | Ascon | 12 | 10 | 0 | 22 | 55% | 1.06x | 0.95x | `ascon-hash256 vs ascon-hash` `1` |
| RISC-V | XXH3 | 5 | 17 | 0 | 22 | 23% | 1.05x | 0.97x | `xxh3-64 vs xxhash-rust` `256` |
| RISC-V | RapidHash | 14 | 27 | 3 | 44 | 32% | 1.08x | 0.86x | `rapidhash-64 vs rapidhash` `64` |
| RISC-V | Auth/KDF | 23 | 42 | 2 | 67 | 34% | 1.08x | 0.93x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| RISC-V | AEAD | 57 | 32 | 21 | 110 | 52% | 1.20x | 0.81x | `aes-256-gcm/encrypt vs rustcrypto` `1048576` |
| macOS | Checksums | 89 | 18 | 3 | 110 | 81% | 4.27x | 0.88x | `crc32c vs crc-fast` `256` |
| macOS | SHA-2 | 12 | 42 | 5 | 59 | 20% | 1.02x | 0.57x | `sha512/streaming vs sha2` `64B` |
| macOS | SHA-3 | 12 | 26 | 8 | 46 | 26% | 1.09x | 0.82x | `sha3-512 vs sha3` `32` |
| macOS | SHAKE | 6 | 8 | 8 | 22 | 27% | 1.08x | 0.65x | `shake256 vs sha3` `1` |
| macOS | BLAKE2 | 24 | 95 | 11 | 130 | 18% | 1.03x | 0.49x | `blake2 vs rustcrypto` `blake2b256/256` |
| macOS | BLAKE3 | 17 | 25 | 6 | 48 | 35% | 1.20x | 0.73x | `blake3/keyed vs blake3` `64` |
| macOS | Ascon | 17 | 5 | 0 | 22 | 77% | 1.10x | 1.02x | `ascon-hash256 vs ascon-hash` `0` |
| macOS | XXH3 | 6 | 9 | 7 | 22 | 27% | 1.04x | 0.76x | `xxh3-64 vs xxhash-rust` `32` |
| macOS | RapidHash | 12 | 31 | 1 | 44 | 27% | 1.09x | 0.80x | `rapidhash-64 vs rapidhash` `1` |
| macOS | Auth/KDF | 23 | 38 | 6 | 67 | 34% | 1.04x | 0.49x | `ed25519/sign vs dalek` `0` |
| macOS | Password Hashing | 5 | 6 | 4 | 15 | 33% | 0.76x | 0.21x | `argon2d-small vs rustcrypto` `m=64_t=3_p=2` |
| macOS | AEAD | 104 | 6 | 0 | 110 | 95% | 2.59x | 0.99x | `aegis-256/encrypt vs aegis-crate` `0` |

## Worst Losses

| Rank | Platform | Category | Speedup | Case |
|---:|---|---|---:|---|
| 1 | SPR | SHA-2 | 0.01x | `sha256/streaming vs sha2` `64B` |
| 2 | SPR | Auth/KDF | 0.01x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| 3 | SPR | Auth/KDF | 0.01x | `hmac-sha256/streaming vs rustcrypto` `4096B` |
| 4 | SPR | SHA-2 | 0.01x | `sha256/streaming vs sha2` `4096B` |
| 5 | macOS | Password Hashing | 0.21x | `argon2d-small vs rustcrypto` `m=64_t=3_p=2` |
| 6 | macOS | Password Hashing | 0.21x | `argon2id-small vs rustcrypto` `m=64_t=3_p=2` |
| 7 | macOS | Password Hashing | 0.24x | `argon2i-small vs rustcrypto` `m=64_t=3_p=2` |
| 8 | SPR | XXH3 | 0.38x | `xxh3-64 vs xxhash-rust` `256` |
| 9 | ICL | AEAD | 0.45x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 10 | SPR | AEAD | 0.48x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 11 | ICL | AEAD | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 12 | Zen4 | AEAD | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 13 | SPR | AEAD | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 14 | macOS | BLAKE2 | 0.49x | `blake2 vs rustcrypto` `blake2b256/256` |
| 15 | macOS | Auth/KDF | 0.49x | `ed25519/sign vs dalek` `0` |
| 16 | Zen5 | AEAD | 0.50x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 17 | s390x | RapidHash | 0.52x | `rapidhash-v3-64 vs rapidhash` `1` |
| 18 | ICL | XXH3 | 0.52x | `xxh3-64 vs xxhash-rust` `64` |
| 19 | Zen4 | AEAD | 0.52x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 20 | Zen5 | AEAD | 0.53x | `aes-256-gcm/encrypt vs rustcrypto` `0` |

## Strongest Wins

| Rank | Platform | Category | Speedup | Case |
|---:|---|---|---:|---|
| 1 | SPR | Checksums | 213.39x | `crc16-ibm vs crc` `262144` |
| 2 | SPR | Checksums | 211.43x | `crc16-ccitt vs crc` `262144` |
| 3 | SPR | Checksums | 204.72x | `crc16-ibm vs crc` `16384` |
| 4 | SPR | Checksums | 196.89x | `crc16-ccitt vs crc` `16384` |
| 5 | SPR | Checksums | 188.16x | `crc16-ibm vs crc` `1048576` |
| 6 | SPR | Checksums | 186.89x | `crc16-ibm vs crc` `65536` |
| 7 | SPR | Checksums | 185.44x | `crc16-ccitt vs crc` `65536` |
| 8 | SPR | Checksums | 183.97x | `crc16-ccitt vs crc` `4096` |
| 9 | POWER10 | Checksums | 179.03x | `crc16-ibm vs crc` `262144` |
| 10 | POWER10 | Checksums | 178.42x | `crc16-ibm vs crc` `1048576` |
| 11 | POWER10 | Checksums | 177.31x | `crc16-ccitt vs crc` `262144` |
| 12 | POWER10 | Checksums | 176.23x | `crc16-ccitt vs crc` `1048576` |
| 13 | SPR | Checksums | 174.80x | `crc16-ccitt vs crc` `1048576` |
| 14 | macOS | Checksums | 171.15x | `crc16-ccitt vs crc` `1048576` |
| 15 | macOS | Checksums | 170.62x | `crc16-ibm vs crc` `262144` |
| 16 | POWER10 | Checksums | 168.87x | `crc16-ibm vs crc` `65536` |
| 17 | macOS | Checksums | 168.84x | `crc16-ccitt vs crc` `262144` |
| 18 | POWER10 | Checksums | 168.60x | `crc16-ccitt vs crc` `65536` |
| 19 | macOS | Checksums | 168.36x | `crc16-ccitt vs crc` `65536` |
| 20 | macOS | Checksums | 166.94x | `crc16-ibm vs crc` `1048576` |

## Notes

- CI artifact headers were normalized to the GitHub run creation time (`03_01_46`) so all Linux files share one timestamp, as required by the benchmark extraction convention.
- The local macOS result keeps its local host time (`07_57_38`) and is not a CI artifact.
- Unpaired internal microbenchmarks, such as SHA3 digest-pair and Ascon many/scalar-loop tests, are intentionally excluded from the external-comparison scoreboards.
- The real work is in the losses: password hashing on local Apple Silicon, tiny-message AES-GCM, SPR streaming SHA-2/HMAC, RISC-V CRC32C, and the BLAKE2/RapidHash parity zones are the places to investigate before celebrating aggregate wins.

## Input Files

- `benchmark_results/2026-04-27/linux/amd-zen4/results.txt`
- `benchmark_results/2026-04-27/linux/amd-zen5/results.txt`
- `benchmark_results/2026-04-27/linux/intel-spr/results.txt`
- `benchmark_results/2026-04-27/linux/intel-icl/results.txt`
- `benchmark_results/2026-04-27/linux/graviton3/results.txt`
- `benchmark_results/2026-04-27/linux/graviton4/results.txt`
- `benchmark_results/2026-04-27/linux/ibm-s390x/results.txt`
- `benchmark_results/2026-04-27/linux/ibm-power10/results.txt`
- `benchmark_results/2026-04-27/linux/rise-riscv/results.txt`
- `benchmark_results/2026-04-27/macos/aarch64/results.txt`
