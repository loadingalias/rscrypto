# Benchmark Overview

Source: Bench CI run [#24968459490](https://github.com/loadingalias/rscrypto/actions/runs/24968459490) created 2026-04-26 22:18:11 UTC.
Commit: `6703515f25eb36a85764d94fdede4fb88079f55a`

Scope: nine Linux CI runners. The older local macOS Apple Silicon result under `benchmark_results/2026-04-26/macos/aarch64/results.txt` is intentionally excluded because it was captured at commit `f5c6ff7c`, not this CI commit. Speedup is `competitor_time / rscrypto_time`; values above `1.00x` mean `rscrypto` is faster. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`.

## Executive Read

Overall: **3606W / 1848T / 342L = 62% win rate** across 5796 paired external comparisons. Geomean speedup is **1.74x** and median speedup is **1.13x**.
Strongest category by geomean: **Checksums** at 4.41x. Weakest category by geomean: **RapidHash** at 1.05x.
Checksums: 794W/148T/48L, geomean 4.41x, worst 0.54x on RISC-V `crc64-nvme vs crc-fast` case `1048576`.
SHA-3: 384W/30T/0L, geomean 2.17x, worst 0.97x on POWER10 `sha3-256/streaming vs sha3` case `64B`.
Ascon: 126W/68T/4L, geomean 1.31x, worst 0.88x on s390x `ascon-hash256 vs ascon-hash` case `0`.
AEAD: 725W/179T/86L, geomean 1.82x, worst 0.42x on SPR `aes-256-gcm/encrypt vs rustcrypto` case `32`.
RapidHash: 105W/209T/82L, geomean 1.05x, worst 0.53x on s390x `rapidhash-v3-64 vs rapidhash` case `1`.
XXH3: 107W/68T/23L, geomean 1.18x, worst 0.29x on Zen5 `xxh3-64 vs xxhash-rust` case `256`.
BLAKE3: 229W/172T/31L, geomean 1.41x, worst 0.73x on s390x `blake3/xof vs blake3` case `1024`.
BLAKE2: 353W/465T/28L, geomean 1.05x, worst 0.75x on RISC-V `blake2/blake2b256 vs rustcrypto` case `1048576`.

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

## Overall Scoreboard

| W | T | L | Total | Win % | Geomean | Median | Worst | Best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3606 | 1848 | 342 | 5796 | 62% | 1.74x | 1.13x | 0.01x | 210.51x |

## By Category

| Category | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Platform | Worst Case | Best | Best Platform | Best Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Checksums | 794 | 148 | 48 | 990 | 80% | 4.41x | 2.42x | 0.54x | RISC-V | `crc64-nvme vs crc-fast` `1048576` | 210.51x | SPR | `crc16-ccitt vs crc` `262144` |
| SHA-2 | 274 | 249 | 8 | 531 | 52% | 1.44x | 1.05x | 0.01x | SPR | `sha256/streaming vs sha2` `64B` | 119.15x | SPR | `sha224 vs sha2` `1048576` |
| SHA-3 | 384 | 30 | 0 | 414 | 93% | 2.17x | 2.15x | 0.97x | POWER10 | `sha3-256/streaming vs sha3` `64B` | 24.95x | s390x | `sha3-512 vs sha3` `1048576` |
| SHAKE | 190 | 8 | 0 | 198 | 96% | 2.60x | 2.23x | 0.99x | POWER10 | `shake256 vs sha3` `262144` | 16.34x | s390x | `shake256 vs sha3` `1048576` |
| BLAKE2 | 353 | 465 | 28 | 846 | 42% | 1.05x | 1.03x | 0.75x | RISC-V | `blake2/blake2b256 vs rustcrypto` `1048576` | 1.38x | s390x | `blake2/blake2s128 vs rustcrypto` `64` |
| BLAKE3 | 229 | 172 | 31 | 432 | 53% | 1.41x | 1.06x | 0.73x | s390x | `blake3/xof vs blake3` `1024` | 8.43x | s390x | `blake3/xof vs blake3` `262144` |
| Ascon | 126 | 68 | 4 | 198 | 64% | 1.31x | 1.18x | 0.88x | s390x | `ascon-hash256 vs ascon-hash` `0` | 2.51x | Zen5 | `ascon-xof128 vs ascon-hash` `0` |
| XXH3 | 107 | 68 | 23 | 198 | 54% | 1.18x | 1.08x | 0.29x | Zen5 | `xxh3-64 vs xxhash-rust` `256` | 2.48x | s390x | `xxh3-64 vs xxhash-rust` `4096` |
| RapidHash | 105 | 209 | 82 | 396 | 27% | 1.05x | 1.00x | 0.53x | s390x | `rapidhash-v3-64 vs rapidhash` `1` | 2.29x | Grav3 | `rapidhash-v3-128 vs rapidhash` `32` |
| Auth/KDF | 319 | 252 | 32 | 603 | 53% | 1.26x | 1.06x | 0.01x | SPR | `hmac-sha256/streaming vs rustcrypto` `64B` | 84.39x | SPR | `hkdf-sha256/expand vs rustcrypto` `32` |
| AEAD | 725 | 179 | 86 | 990 | 73% | 1.82x | 1.43x | 0.42x | SPR | `aes-256-gcm/encrypt vs rustcrypto` `32` | 32.50x | s390x | `aes-256-gcm-siv/decrypt vs rustcrypto` `0` |

## By Platform

| Platform | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Category | Worst Case | Best | Best Category | Best Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Zen4 | 467 | 155 | 22 | 644 | 73% | 1.61x | 1.17x | 0.42x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `32` | 128.61x | Checksums | `crc16-ibm vs crc` `262144` |
| Zen5 | 376 | 234 | 34 | 644 | 58% | 1.63x | 1.11x | 0.29x | XXH3 | `xxh3-64 vs xxhash-rust` `256` | 129.27x | Checksums | `crc16-ccitt vs crc` `262144` |
| SPR | 498 | 97 | 49 | 644 | 77% | 1.89x | 1.25x | 0.01x | SHA-2 | `sha256/streaming vs sha2` `64B` | 210.51x | Checksums | `crc16-ccitt vs crc` `262144` |
| ICL | 472 | 149 | 23 | 644 | 73% | 1.61x | 1.21x | 0.44x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `32` | 119.91x | Checksums | `crc16-ibm vs crc` `262144` |
| Grav3 | 330 | 271 | 43 | 644 | 51% | 1.64x | 1.05x | 0.77x | AEAD | `aegis-256/encrypt vs aegis-crate` `0` | 126.73x | Checksums | `crc16-ibm vs crc` `262144` |
| Grav4 | 321 | 300 | 23 | 644 | 50% | 1.65x | 1.05x | 0.77x | XXH3 | `xxh3-128 vs xxhash-rust` `256` | 105.20x | Checksums | `crc16-ibm vs crc` `262144` |
| s390x | 525 | 70 | 49 | 644 | 82% | 3.00x | 2.36x | 0.53x | RapidHash | `rapidhash-v3-64 vs rapidhash` `1` | 99.35x | Checksums | `crc16-ccitt vs crc` `65536` |
| POWER10 | 324 | 295 | 25 | 644 | 50% | 1.92x | 1.05x | 0.70x | RapidHash | `rapidhash-v3-128 vs rapidhash` `1` | 177.55x | Checksums | `crc16-ibm vs crc` `262144` |
| RISC-V | 293 | 277 | 74 | 644 | 45% | 1.16x | 1.04x | 0.54x | Checksums | `crc64-nvme vs crc-fast` `1048576` | 10.96x | Checksums | `crc64-nvme vs crc-fast` `0` |

## Platform x Category

| Platform | Category | W | T | L | Total | Win % | Geomean | Worst | Worst Case |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zen4 | Checksums | 90 | 19 | 1 | 110 | 82% | 4.45x | 0.95x | `crc32 vs crc-fast` `1024` |
| Zen4 | SHA-2 | 35 | 24 | 0 | 59 | 59% | 1.12x | 0.99x | `sha256/streaming vs sha2` `64B` |
| Zen4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.28x | 2.15x | `sha3-224 vs sha3` `1024` |
| Zen4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.06x | 2.26x | `shake256 vs sha3` `262144` |
| Zen4 | BLAKE2 | 56 | 37 | 1 | 94 | 60% | 1.06x | 0.94x | `blake2/blake2s256 vs rustcrypto` `0` |
| Zen4 | BLAKE3 | 24 | 22 | 2 | 48 | 50% | 1.34x | 0.95x | `blake3/xof vs blake3` `64` |
| Zen4 | Ascon | 22 | 0 | 0 | 22 | 100% | 1.93x | 1.67x | `ascon-hash256 vs ascon-hash` `1` |
| Zen4 | XXH3 | 14 | 3 | 5 | 22 | 64% | 1.04x | 0.75x | `xxh3-64 vs xxhash-rust` `256` |
| Zen4 | RapidHash | 14 | 30 | 0 | 44 | 32% | 1.07x | 0.95x | `rapidhash-v3-64 vs rapidhash` `4096` |
| Zen4 | Auth/KDF | 56 | 7 | 4 | 67 | 84% | 1.20x | 0.92x | `pbkdf2-sha256/iters=1000 vs rustcrypto` `64` |
| Zen4 | AEAD | 88 | 13 | 9 | 110 | 80% | 1.24x | 0.42x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| Zen5 | Checksums | 91 | 18 | 1 | 110 | 83% | 4.80x | 0.93x | `crc32c vs crc-fast` `1024` |
| Zen5 | SHA-2 | 27 | 32 | 0 | 59 | 46% | 1.09x | 0.99x | `sha256/streaming vs sha2` `64B` |
| Zen5 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.18x | 2.11x | `sha3-256/streaming vs sha3` `64B` |
| Zen5 | SHAKE | 22 | 0 | 0 | 22 | 100% | 2.91x | 2.15x | `shake256 vs sha3` `1048576` |
| Zen5 | BLAKE2 | 6 | 87 | 1 | 94 | 6% | 1.00x | 0.94x | `blake2/streaming/blake2b256 vs rustcrypto` `64B` |
| Zen5 | BLAKE3 | 19 | 25 | 4 | 48 | 40% | 1.33x | 0.85x | `blake3/keyed vs blake3` `256` |
| Zen5 | Ascon | 22 | 0 | 0 | 22 | 100% | 2.17x | 1.84x | `ascon-hash256 vs ascon-hash` `1` |
| Zen5 | XXH3 | 13 | 6 | 3 | 22 | 59% | 1.19x | 0.29x | `xxh3-64 vs xxhash-rust` `256` |
| Zen5 | RapidHash | 11 | 25 | 8 | 44 | 25% | 1.05x | 0.94x | `rapidhash-v3-64 vs rapidhash` `16384` |
| Zen5 | Auth/KDF | 32 | 28 | 7 | 67 | 48% | 1.09x | 0.93x | `pbkdf2-sha256/iters=1000 vs rustcrypto` `64` |
| Zen5 | AEAD | 87 | 13 | 10 | 110 | 79% | 1.37x | 0.43x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| SPR | Checksums | 96 | 12 | 2 | 110 | 87% | 4.85x | 0.86x | `crc32 vs crc32fast` `256` |
| SPR | SHA-2 | 57 | 0 | 2 | 59 | 97% | 3.28x | 0.01x | `sha256/streaming vs sha2` `64B` |
| SPR | SHA-3 | 46 | 0 | 0 | 46 | 100% | 3.58x | 3.46x | `sha3-256/streaming vs sha3` `64B` |
| SPR | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.78x | 3.51x | `shake256 vs sha3` `262144` |
| SPR | BLAKE2 | 86 | 8 | 0 | 94 | 91% | 1.15x | 0.96x | `blake2/blake2b512 vs rustcrypto` `256` |
| SPR | BLAKE3 | 24 | 16 | 8 | 48 | 50% | 1.21x | 0.87x | `blake3/keyed vs blake3` `16384` |
| SPR | Ascon | 22 | 0 | 0 | 22 | 100% | 1.46x | 1.28x | `ascon-hash256 vs ascon-hash` `1` |
| SPR | XXH3 | 12 | 7 | 3 | 22 | 55% | 1.09x | 0.55x | `xxh3-64 vs xxhash-rust` `64` |
| SPR | RapidHash | 10 | 20 | 14 | 44 | 23% | 1.03x | 0.79x | `rapidhash-64 vs rapidhash` `0` |
| SPR | Auth/KDF | 44 | 15 | 8 | 67 | 66% | 1.21x | 0.01x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| SPR | AEAD | 79 | 19 | 12 | 110 | 72% | 1.26x | 0.42x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| ICL | Checksums | 87 | 21 | 2 | 110 | 79% | 4.00x | 0.74x | `crc32 vs crc32fast` `256` |
| ICL | SHA-2 | 35 | 22 | 2 | 59 | 59% | 1.12x | 0.83x | `sha256/streaming vs sha2` `4096B` |
| ICL | SHA-3 | 46 | 0 | 0 | 46 | 100% | 3.00x | 2.87x | `sha3-256/streaming vs sha3` `64B` |
| ICL | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.03x | 2.99x | `shake128 vs sha3` `1048576` |
| ICL | BLAKE2 | 92 | 2 | 0 | 94 | 98% | 1.16x | 1.05x | `blake2/blake2s256 vs rustcrypto` `1` |
| ICL | BLAKE3 | 21 | 24 | 3 | 48 | 44% | 1.19x | 0.82x | `blake3/keyed vs blake3` `256` |
| ICL | Ascon | 22 | 0 | 0 | 22 | 100% | 1.41x | 1.25x | `ascon-hash256 vs ascon-hash` `1` |
| ICL | XXH3 | 13 | 6 | 3 | 22 | 59% | 1.26x | 0.53x | `xxh3-64 vs xxhash-rust` `64` |
| ICL | RapidHash | 15 | 28 | 1 | 44 | 34% | 1.08x | 0.89x | `rapidhash-64 vs rapidhash` `32` |
| ICL | Auth/KDF | 36 | 28 | 3 | 67 | 54% | 1.10x | 0.82x | `hmac-sha256/streaming vs rustcrypto` `4096B` |
| ICL | AEAD | 83 | 18 | 9 | 110 | 75% | 1.24x | 0.44x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| Grav3 | Checksums | 79 | 24 | 7 | 110 | 72% | 3.69x | 0.92x | `crc32 vs crc-fast` `262144` |
| Grav3 | SHA-2 | 21 | 37 | 1 | 59 | 36% | 1.06x | 0.91x | `sha256/streaming vs sha2` `64B` |
| Grav3 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.82x | 1.74x | `sha3-256/streaming vs sha3` `64B` |
| Grav3 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.82x | 1.77x | `shake256 vs sha3` `262144` |
| Grav3 | BLAKE2 | 22 | 72 | 0 | 94 | 23% | 1.03x | 0.99x | `blake2/streaming/blake2b256 vs rustcrypto` `64B` |
| Grav3 | BLAKE3 | 22 | 22 | 4 | 48 | 46% | 1.41x | 0.93x | `blake3/streaming vs blake3` `4096B` |
| Grav3 | Ascon | 4 | 18 | 0 | 22 | 18% | 1.03x | 0.99x | `ascon-xof128 vs ascon-hash` `1048576` |
| Grav3 | XXH3 | 6 | 13 | 3 | 22 | 27% | 1.07x | 0.86x | `xxh3-128 vs xxhash-rust` `256` |
| Grav3 | RapidHash | 11 | 18 | 15 | 44 | 25% | 1.05x | 0.90x | `rapidhash-v3-64 vs rapidhash` `256` |
| Grav3 | Auth/KDF | 32 | 28 | 7 | 67 | 48% | 1.09x | 0.88x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| Grav3 | AEAD | 65 | 39 | 6 | 110 | 59% | 2.48x | 0.77x | `aegis-256/encrypt vs aegis-crate` `0` |
| Grav4 | Checksums | 79 | 30 | 1 | 110 | 72% | 3.68x | 0.81x | `crc32c vs crc-fast` `256` |
| Grav4 | SHA-2 | 19 | 38 | 2 | 59 | 32% | 1.05x | 0.90x | `sha256/streaming vs sha2` `64B` |
| Grav4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.97x | 1.88x | `sha3-256/streaming vs sha3` `64B` |
| Grav4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.99x | 1.94x | `shake256 vs sha3` `16384` |
| Grav4 | BLAKE2 | 18 | 76 | 0 | 94 | 19% | 1.02x | 0.98x | `blake2/streaming/blake2b256 vs rustcrypto` `64B` |
| Grav4 | BLAKE3 | 23 | 24 | 1 | 48 | 48% | 1.47x | 0.93x | `blake3/keyed vs blake3` `1` |
| Grav4 | Ascon | 12 | 10 | 0 | 22 | 55% | 1.08x | 1.01x | `ascon-hash256 vs ascon-hash` `1` |
| Grav4 | XXH3 | 5 | 13 | 4 | 22 | 23% | 1.03x | 0.77x | `xxh3-128 vs xxhash-rust` `256` |
| Grav4 | RapidHash | 12 | 29 | 3 | 44 | 27% | 1.07x | 0.89x | `rapidhash-v3-128 vs rapidhash` `256` |
| Grav4 | Auth/KDF | 20 | 45 | 2 | 67 | 30% | 1.07x | 0.87x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| Grav4 | AEAD | 65 | 35 | 10 | 110 | 59% | 2.46x | 0.78x | `aegis-256/encrypt vs aegis-crate` `0` |
| s390x | Checksums | 99 | 3 | 8 | 110 | 90% | 7.25x | 0.62x | `crc24-openpgp vs crc` `64` |
| s390x | SHA-2 | 59 | 0 | 0 | 59 | 100% | 5.18x | 1.93x | `sha256 vs sha2` `32` |
| s390x | SHA-3 | 46 | 0 | 0 | 46 | 100% | 4.69x | 1.10x | `sha3-256/streaming vs sha3` `64B` |
| s390x | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.94x | 1.06x | `shake256 vs sha3` `1` |
| s390x | BLAKE2 | 54 | 40 | 0 | 94 | 57% | 1.08x | 0.98x | `blake2/streaming/blake2b256 vs rustcrypto` `64B` |
| s390x | BLAKE3 | 40 | 4 | 4 | 48 | 83% | 1.82x | 0.73x | `blake3/xof vs blake3` `1024` |
| s390x | Ascon | 4 | 15 | 3 | 22 | 18% | 1.03x | 0.88x | `ascon-hash256 vs ascon-hash` `0` |
| s390x | XXH3 | 18 | 2 | 2 | 22 | 82% | 1.73x | 0.69x | `xxh3-64 vs xxhash-rust` `32` |
| s390x | RapidHash | 9 | 3 | 32 | 44 | 20% | 0.95x | 0.53x | `rapidhash-v3-64 vs rapidhash` `1` |
| s390x | Auth/KDF | 64 | 3 | 0 | 67 | 96% | 3.46x | 1.02x | `ed25519/verify vs dalek` `32` |
| s390x | AEAD | 110 | 0 | 0 | 110 | 100% | 4.32x | 1.16x | `xchacha20-poly1305/encrypt vs rustcrypto` `1` |
| POWER10 | Checksums | 105 | 2 | 3 | 110 | 95% | 10.78x | 0.78x | `crc64-xz vs crc64fast` `64` |
| POWER10 | SHA-2 | 5 | 54 | 0 | 59 | 8% | 1.01x | 0.98x | `sha256 vs sha2` `4096` |
| POWER10 | SHA-3 | 17 | 29 | 0 | 46 | 37% | 1.04x | 0.97x | `sha3-256/streaming vs sha3` `64B` |
| POWER10 | SHAKE | 14 | 8 | 0 | 22 | 64% | 1.39x | 0.99x | `shake256 vs sha3` `262144` |
| POWER10 | BLAKE2 | 4 | 81 | 9 | 94 | 4% | 0.98x | 0.93x | `blake2/keyed/blake2b512 vs rustcrypto` `32` |
| POWER10 | BLAKE3 | 36 | 12 | 0 | 48 | 75% | 1.99x | 0.96x | `blake3/keyed vs blake3` `1` |
| POWER10 | Ascon | 5 | 17 | 0 | 22 | 23% | 1.05x | 1.00x | `ascon-hash256 vs ascon-hash` `1` |
| POWER10 | XXH3 | 19 | 3 | 0 | 22 | 86% | 1.29x | 0.97x | `xxh3-64 vs xxhash-rust` `1` |
| POWER10 | RapidHash | 11 | 26 | 7 | 44 | 25% | 1.06x | 0.70x | `rapidhash-v3-128 vs rapidhash` `1` |
| POWER10 | Auth/KDF | 12 | 55 | 0 | 67 | 18% | 1.05x | 0.98x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| POWER10 | AEAD | 96 | 8 | 6 | 110 | 87% | 2.63x | 0.93x | `aegis-256/encrypt vs aegis-crate` `4096` |
| RISC-V | Checksums | 68 | 19 | 23 | 110 | 62% | 1.44x | 0.54x | `crc64-nvme vs crc-fast` `1048576` |
| RISC-V | SHA-2 | 16 | 42 | 1 | 59 | 27% | 1.03x | 0.95x | `sha512 vs sha2` `4096` |
| RISC-V | SHA-3 | 45 | 1 | 0 | 46 | 98% | 1.17x | 1.02x | `sha3-256/streaming vs sha3` `4096B` |
| RISC-V | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.61x | 1.15x | `shake256 vs sha3` `16384` |
| RISC-V | BLAKE2 | 15 | 62 | 17 | 94 | 16% | 0.98x | 0.75x | `blake2/blake2b256 vs rustcrypto` `1048576` |
| RISC-V | BLAKE3 | 20 | 23 | 5 | 48 | 42% | 1.16x | 0.74x | `blake3/streaming vs blake3` `65536B` |
| RISC-V | Ascon | 13 | 8 | 1 | 22 | 59% | 1.06x | 0.95x | `ascon-hash256 vs ascon-hash` `0` |
| RISC-V | XXH3 | 7 | 15 | 0 | 22 | 32% | 1.04x | 0.95x | `xxh3-128 vs xxhash-rust` `1024` |
| RISC-V | RapidHash | 12 | 30 | 2 | 44 | 27% | 1.08x | 0.85x | `rapidhash-64 vs rapidhash` `64` |
| RISC-V | Auth/KDF | 23 | 43 | 1 | 67 | 34% | 1.07x | 0.94x | `pbkdf2-sha512/iters=1 vs rustcrypto` `64` |
| RISC-V | AEAD | 52 | 34 | 24 | 110 | 47% | 1.19x | 0.85x | `aes-256-gcm/encrypt vs rustcrypto` `262144` |

## By Primitive Group

| Category | Group | W | T | L | Total | Win % | Geomean | Median | Worst | Platform | Case | Best | Platform | Case |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Checksums | `crc16-ccitt vs crc` | 98 | 0 | 1 | 99 | 99% | 23.64x | 37.37x | 0.68x | s390x | `1` | 210.51x | SPR | `262144` |
| Checksums | `crc16-ibm vs crc` | 98 | 0 | 1 | 99 | 99% | 24.26x | 38.96x | 0.79x | s390x | `1` | 209.58x | SPR | `262144` |
| Checksums | `crc24-openpgp vs crc` | 97 | 1 | 1 | 99 | 98% | 13.55x | 15.03x | 0.62x | s390x | `64` | 126.95x | Zen5 | `262144` |
| Checksums | `crc32 vs crc-fast` | 62 | 32 | 5 | 99 | 63% | 2.13x | 1.30x | 0.92x | Grav3 | `262144` | 27.46x | POWER10 | `1048576` |
| Checksums | `crc32 vs crc32fast` | 83 | 8 | 8 | 99 | 84% | 2.42x | 2.00x | 0.65x | RISC-V | `64` | 24.02x | POWER10 | `262144` |
| Checksums | `crc32c vs crc-fast` | 55 | 36 | 8 | 99 | 56% | 1.81x | 1.12x | 0.81x | Grav4 | `256` | 27.33x | POWER10 | `262144` |
| Checksums | `crc32c vs crc32c` | 93 | 3 | 3 | 99 | 94% | 2.80x | 2.18x | 0.57x | RISC-V | `32` | 40.92x | POWER10 | `262144` |
| Checksums | `crc64-nvme vs crc-fast` | 59 | 37 | 3 | 99 | 60% | 2.21x | 1.17x | 0.54x | RISC-V | `1048576` | 33.67x | POWER10 | `262144` |
| Checksums | `crc64-nvme vs crc64fast-nvme` | 74 | 16 | 9 | 99 | 75% | 2.49x | 2.22x | 0.61x | RISC-V | `1048576` | 28.94x | POWER10 | `262144` |
| Checksums | `crc64-xz vs crc64fast` | 75 | 15 | 9 | 99 | 76% | 2.51x | 2.26x | 0.67x | RISC-V | `1048576` | 28.76x | POWER10 | `262144` |
| SHA-2 | `sha256/streaming vs sha2` | 2 | 10 | 6 | 18 | 11% | 0.69x | 0.99x | 0.01x | SPR | `64B` | 9.63x | s390x | `4096B` |
| SHA-2 | `sha512/streaming vs sha2` | 7 | 11 | 0 | 18 | 39% | 1.23x | 1.03x | 0.95x | Grav3 | `64B` | 10.88x | s390x | `4096B` |
| SHA-3 | `sha3-224 vs sha3` | 93 | 6 | 0 | 99 | 94% | 2.16x | 2.15x | 1.01x | POWER10 | `16384` | 15.59x | s390x | `1048576` |
| SHA-3 | `sha3-256 vs sha3` | 92 | 7 | 0 | 99 | 93% | 2.15x | 2.15x | 1.01x | POWER10 | `262144` | 17.25x | s390x | `1048576` |
| SHA-3 | `sha3-256/streaming vs sha3` | 15 | 3 | 0 | 18 | 83% | 2.04x | 2.02x | 0.97x | POWER10 | `64B` | 11.84x | s390x | `4096B` |
| SHA-3 | `sha3-384 vs sha3` | 92 | 7 | 0 | 99 | 93% | 2.19x | 2.16x | 0.98x | POWER10 | `262144` | 20.55x | s390x | `1048576` |
| SHA-3 | `sha3-512 vs sha3` | 92 | 7 | 0 | 99 | 93% | 2.22x | 2.15x | 1.01x | POWER10 | `1048576` | 24.95x | s390x | `1048576` |
| SHAKE | `shake128 vs sha3` | 95 | 4 | 0 | 99 | 96% | 2.59x | 2.23x | 1.01x | POWER10 | `262144` | 14.73x | s390x | `1048576` |
| SHAKE | `shake256 vs sha3` | 95 | 4 | 0 | 99 | 96% | 2.61x | 2.22x | 0.99x | POWER10 | `262144` | 16.34x | s390x | `1048576` |
| BLAKE2 | `blake2/blake2b256 vs rustcrypto` | 42 | 55 | 2 | 99 | 42% | 1.03x | 1.02x | 0.75x | RISC-V | `1048576` | 1.16x | ICL | `64` |
| BLAKE2 | `blake2/blake2b512 vs rustcrypto` | 42 | 54 | 3 | 99 | 42% | 1.03x | 1.02x | 0.76x | RISC-V | `1048576` | 1.17x | ICL | `32` |
| BLAKE2 | `blake2/blake2s128 vs rustcrypto` | 46 | 52 | 1 | 99 | 46% | 1.08x | 1.04x | 0.95x | RISC-V | `65536` | 1.38x | s390x | `64` |
| BLAKE2 | `blake2/blake2s256 vs rustcrypto` | 35 | 61 | 3 | 99 | 35% | 1.05x | 1.03x | 0.94x | POWER10 | `1048576` | 1.26x | SPR | `4096` |
| BLAKE2 | `blake2/keyed/blake2b256 vs rustcrypto` | 43 | 51 | 5 | 99 | 43% | 1.04x | 1.02x | 0.81x | RISC-V | `262144` | 1.29x | RISC-V | `0` |
| BLAKE2 | `blake2/keyed/blake2b512 vs rustcrypto` | 41 | 54 | 4 | 99 | 41% | 1.03x | 1.02x | 0.83x | RISC-V | `262144` | 1.23x | ICL | `0` |
| BLAKE2 | `blake2/keyed/blake2s128 vs rustcrypto` | 43 | 51 | 5 | 99 | 43% | 1.06x | 1.04x | 0.86x | RISC-V | `1` | 1.27x | ICL | `0` |
| BLAKE2 | `blake2/keyed/blake2s256 vs rustcrypto` | 43 | 52 | 4 | 99 | 43% | 1.07x | 1.04x | 0.88x | RISC-V | `1` | 1.28x | SPR | `256` |
| BLAKE2 | `blake2/streaming/blake2b256 vs rustcrypto` | 8 | 18 | 1 | 27 | 30% | 1.02x | 1.00x | 0.94x | Zen5 | `64B` | 1.15x | ICL | `64B` |
| BLAKE2 | `blake2/streaming/blake2s256 vs rustcrypto` | 10 | 17 | 0 | 27 | 37% | 1.07x | 1.03x | 0.97x | RISC-V | `64B` | 1.25x | SPR | `65536B` |
| BLAKE3 | `blake3 vs blake3` | 45 | 49 | 5 | 99 | 45% | 1.34x | 1.04x | 0.75x | s390x | `1024` | 8.33x | s390x | `262144` |
| BLAKE3 | `blake3/derive-key vs blake3` | 79 | 18 | 2 | 99 | 80% | 1.80x | 1.90x | 0.81x | s390x | `1024` | 5.06x | POWER10 | `1048576` |
| BLAKE3 | `blake3/keyed vs blake3` | 40 | 52 | 7 | 99 | 40% | 1.31x | 1.02x | 0.76x | s390x | `1024` | 5.21x | POWER10 | `1048576` |
| BLAKE3 | `blake3/streaming vs blake3` | 10 | 19 | 7 | 36 | 28% | 1.14x | 1.00x | 0.74x | RISC-V | `65536B` | 3.12x | POWER10 | `65536B` |
| BLAKE3 | `blake3/xof vs blake3` | 55 | 34 | 10 | 99 | 56% | 1.36x | 1.07x | 0.73x | s390x | `1024` | 8.43x | s390x | `262144` |
| Ascon | `ascon-hash256 vs ascon-hash` | 54 | 41 | 4 | 99 | 55% | 1.26x | 1.05x | 0.88x | s390x | `0` | 2.18x | Zen5 | `16384` |
| Ascon | `ascon-xof128 vs ascon-hash` | 72 | 27 | 0 | 99 | 73% | 1.35x | 1.29x | 0.99x | Grav3 | `1048576` | 2.51x | Zen5 | `0` |
| XXH3 | `xxh3-128 vs xxhash-rust` | 50 | 37 | 12 | 99 | 51% | 1.18x | 1.06x | 0.69x | SPR | `256` | 2.39x | s390x | `1048576` |
| XXH3 | `xxh3-64 vs xxhash-rust` | 57 | 31 | 11 | 99 | 58% | 1.17x | 1.10x | 0.29x | Zen5 | `256` | 2.48x | s390x | `4096` |
| RapidHash | `rapidhash-128 vs rapidhash` | 33 | 60 | 6 | 99 | 33% | 1.09x | 1.02x | 0.88x | s390x | `1048576` | 2.19x | POWER10 | `0` |
| RapidHash | `rapidhash-64 vs rapidhash` | 18 | 64 | 17 | 99 | 18% | 1.02x | 1.00x | 0.68x | s390x | `32` | 1.60x | POWER10 | `32` |
| RapidHash | `rapidhash-v3-128 vs rapidhash` | 32 | 39 | 28 | 99 | 32% | 1.06x | 1.00x | 0.70x | POWER10 | `1` | 2.29x | Grav3 | `32` |
| RapidHash | `rapidhash-v3-64 vs rapidhash` | 22 | 46 | 31 | 99 | 22% | 1.03x | 1.00x | 0.53x | s390x | `1` | 2.00x | Zen4 | `32` |
| Auth/KDF | `ed25519/keypair-from-secret vs dalek` | 9 | 0 | 0 | 9 | 100% | 1.62x | 1.61x | 1.27x | Grav4 | `rscrypto` | 2.49x | Zen4 | `rscrypto` |
| Auth/KDF | `ed25519/public-key-from-secret vs dalek` | 9 | 0 | 0 | 9 | 100% | 1.67x | 1.62x | 1.28x | Grav4 | `rscrypto` | 3.03x | Zen4 | `rscrypto` |
| Auth/KDF | `ed25519/sign vs dalek` | 36 | 0 | 0 | 36 | 100% | 1.59x | 1.55x | 1.12x | ICL | `16384` | 3.72x | s390x | `16384` |
| Auth/KDF | `ed25519/verify vs dalek` | 18 | 15 | 3 | 36 | 50% | 1.08x | 1.05x | 0.93x | Zen5 | `0` | 1.40x | s390x | `16384` |
| Auth/KDF | `hkdf-sha256/expand vs rustcrypto` | 31 | 5 | 0 | 36 | 86% | 1.85x | 1.09x | 1.00x | POWER10 | `256` | 84.39x | SPR | `32` |
| Auth/KDF | `hkdf-sha384/expand vs rustcrypto` | 25 | 11 | 0 | 36 | 69% | 1.21x | 1.08x | 1.00x | RISC-V | `1024` | 2.94x | s390x | `256` |
| Auth/KDF | `hmac-sha256 vs rustcrypto` | 38 | 56 | 5 | 99 | 38% | 1.27x | 1.01x | 0.72x | SPR | `1024` | 9.35x | s390x | `1048576` |
| Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | 2 | 10 | 6 | 18 | 11% | 0.69x | 0.99x | 0.01x | SPR | `64B` | 9.54x | s390x | `4096B` |
| Auth/KDF | `hmac-sha384 vs rustcrypto` | 52 | 47 | 0 | 99 | 53% | 1.28x | 1.05x | 0.96x | RISC-V | `65536` | 10.52x | s390x | `1048576` |
| Auth/KDF | `hmac-sha512 vs rustcrypto` | 48 | 51 | 0 | 99 | 48% | 1.27x | 1.05x | 0.95x | RISC-V | `65536` | 10.75x | s390x | `1048576` |
| Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | 5 | 8 | 5 | 18 | 28% | 1.03x | 0.99x | 0.68x | SPR | `64` | 1.97x | s390x | `64` |
| Auth/KDF | `pbkdf2-sha256/iters=100 vs rustcrypto` | 2 | 12 | 4 | 18 | 11% | 1.06x | 0.99x | 0.92x | Zen4 | `64` | 2.06x | s390x | `64` |
| Auth/KDF | `pbkdf2-sha256/iters=1000 vs rustcrypto` | 2 | 12 | 4 | 18 | 11% | 1.06x | 1.00x | 0.92x | Zen4 | `64` | 2.09x | s390x | `32` |
| Auth/KDF | `pbkdf2-sha512/iters=1 vs rustcrypto` | 8 | 9 | 1 | 18 | 44% | 1.16x | 1.04x | 0.94x | RISC-V | `64` | 2.58x | s390x | `128` |
| Auth/KDF | `pbkdf2-sha512/iters=100 vs rustcrypto` | 8 | 8 | 2 | 18 | 44% | 1.15x | 1.04x | 0.94x | Grav3 | `64` | 2.88x | s390x | `64` |
| Auth/KDF | `pbkdf2-sha512/iters=1000 vs rustcrypto` | 8 | 8 | 2 | 18 | 44% | 1.15x | 1.04x | 0.94x | Grav3 | `64` | 2.82x | s390x | `128` |
| Auth/KDF | `x25519/diffie-hellman vs dalek` | 9 | 0 | 0 | 9 | 100% | 1.17x | 1.18x | 1.06x | POWER10 | `rscrypto` | 1.47x | RISC-V | `rscrypto` |
| Auth/KDF | `x25519/public-key-from-secret vs dalek` | 9 | 0 | 0 | 9 | 100% | 1.61x | 1.53x | 1.25x | Grav4 | `rscrypto` | 2.58x | Zen4 | `rscrypto` |
| AEAD | `aegis-256/decrypt vs aegis-crate` | 65 | 30 | 4 | 99 | 66% | 1.54x | 1.16x | 0.94x | Grav4 | `65536` | 8.25x | s390x | `1` |
| AEAD | `aegis-256/encrypt vs aegis-crate` | 37 | 43 | 19 | 99 | 37% | 1.36x | 1.02x | 0.77x | Grav3 | `0` | 8.02x | s390x | `0` |
| AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | 94 | 1 | 4 | 99 | 95% | 3.22x | 1.78x | 0.88x | RISC-V | `1048576` | 32.50x | s390x | `0` |
| AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | 94 | 1 | 4 | 99 | 95% | 3.82x | 1.79x | 0.92x | RISC-V | `262144` | 30.78x | s390x | `0` |
| AEAD | `aes-256-gcm/decrypt vs rustcrypto` | 77 | 8 | 14 | 99 | 78% | 2.50x | 1.78x | 0.54x | SPR | `32` | 12.21x | Grav3 | `32` |
| AEAD | `aes-256-gcm/encrypt vs rustcrypto` | 71 | 4 | 24 | 99 | 72% | 2.26x | 1.73x | 0.42x | SPR | `32` | 11.05x | Grav4 | `32` |
| AEAD | `chacha20-poly1305/decrypt vs rustcrypto` | 75 | 20 | 4 | 99 | 76% | 1.33x | 1.38x | 0.60x | Zen5 | `256` | 2.46x | s390x | `65536` |
| AEAD | `chacha20-poly1305/encrypt vs rustcrypto` | 72 | 23 | 4 | 99 | 73% | 1.30x | 1.29x | 0.62x | Zen5 | `256` | 2.45x | s390x | `65536` |
| AEAD | `xchacha20-poly1305/decrypt vs rustcrypto` | 75 | 20 | 4 | 99 | 76% | 1.27x | 1.20x | 0.67x | Zen5 | `256` | 2.51x | s390x | `262144` |
| AEAD | `xchacha20-poly1305/encrypt vs rustcrypto` | 65 | 29 | 5 | 99 | 66% | 1.25x | 1.17x | 0.66x | Zen5 | `256` | 2.49x | s390x | `1048576` |
| Other | `sha224 vs sha2` | 55 | 44 | 0 | 99 | 56% | 1.87x | 1.09x | 0.98x | RISC-V | `1048576` | 119.15x | SPR | `1048576` |
| Other | `sha256 vs sha2` | 50 | 48 | 1 | 99 | 51% | 1.86x | 1.06x | 0.94x | Grav4 | `256` | 117.74x | SPR | `1048576` |
| Other | `sha384 vs sha2` | 57 | 42 | 0 | 99 | 58% | 1.29x | 1.06x | 0.98x | ICL | `262144` | 10.75x | s390x | `65536` |
| Other | `sha512 vs sha2` | 52 | 46 | 1 | 99 | 53% | 1.27x | 1.05x | 0.95x | RISC-V | `4096` | 10.70x | s390x | `262144` |
| Other | `sha512-256 vs sha2` | 51 | 48 | 0 | 99 | 52% | 1.28x | 1.06x | 0.98x | ICL | `65536` | 10.56x | s390x | `65536` |

## 1 MiB End-to-End Throughput

These rows use the `rscrypto` midpoint for the 1 MiB case where that benchmark has a byte-size input. Throughput is computed from the parsed Criterion midpoint.

| Primitive | Zen4 | Zen5 | SPR | ICL | Grav3 | Grav4 | s390x | POWER10 | RISC-V |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| crc32c vs crc32c | 54.30 GiB/s | 66.19 GiB/s | 79.47 GiB/s | 47.07 GiB/s | 37.83 GiB/s | 43.57 GiB/s | 34.09 GiB/s | 50.18 GiB/s | 613 MiB/s |
| sha256 vs sha2 | 1.69 GiB/s | 2.04 GiB/s | 1.45 GiB/s | 1.28 GiB/s | 1.33 GiB/s | 1.43 GiB/s | 1.78 GiB/s | 316 MiB/s | 56 MiB/s |
| sha512 vs sha2 | 639 MiB/s | 1.02 GiB/s | 604 MiB/s | 523 MiB/s | 924 MiB/s | 996 MiB/s | 2.79 GiB/s | 510 MiB/s | 77 MiB/s |
| sha3-256 vs sha3 | 449 MiB/s | 623 MiB/s | 437 MiB/s | 388 MiB/s | 385 MiB/s | 453 MiB/s | 3.60 GiB/s | 454 MiB/s | 82 MiB/s |
| shake256 vs sha3 | 452 MiB/s | 624 MiB/s | 432 MiB/s | 388 MiB/s | 385 MiB/s | 453 MiB/s | 3.60 GiB/s | 454 MiB/s | 82 MiB/s |
| blake2/blake2b512 vs rustcrypto | 878 MiB/s | 749 MiB/s | 937 MiB/s | 1.01 GiB/s | 769 MiB/s | 860 MiB/s | 550 MiB/s | 672 MiB/s | 114 MiB/s |
| blake2/blake2s256 vs rustcrypto | 601 MiB/s | 482 MiB/s | 684 MiB/s | 743 MiB/s | 461 MiB/s | 518 MiB/s | 404 MiB/s | 465 MiB/s | 79 MiB/s |
| blake3 vs blake3 | 19.95 GiB/s | 35.66 GiB/s | 18.16 GiB/s | 12.15 GiB/s | 4.39 GiB/s | 5.53 GiB/s | 2.20 GiB/s | 2.68 GiB/s | 365 MiB/s |
| ascon-hash256 vs ascon-hash | 220 MiB/s | 297 MiB/s | 144 MiB/s | 147 MiB/s | 190 MiB/s | 216 MiB/s | 84 MiB/s | 123 MiB/s | 36 MiB/s |
| xxh3-64 vs xxhash-rust | 61.80 GiB/s | 113.08 GiB/s | 48.74 GiB/s | 60.27 GiB/s | 17.85 GiB/s | 20.35 GiB/s | 10.81 GiB/s | 19.53 GiB/s | 861 MiB/s |
| rapidhash-64 vs rapidhash | 27.68 GiB/s | 50.72 GiB/s | 33.74 GiB/s | 27.89 GiB/s | 20.30 GiB/s | 28.93 GiB/s | 9.50 GiB/s | 21.37 GiB/s | 360 MiB/s |
| hmac-sha256 vs rustcrypto | 1.69 GiB/s | 2.06 GiB/s | 1.43 GiB/s | 1.30 GiB/s | 1.33 GiB/s | 1.43 GiB/s | 1.78 GiB/s | 315 MiB/s | 56 MiB/s |
| aes-256-gcm/encrypt vs rustcrypto | 2.52 GiB/s | 3.20 GiB/s | 2.35 GiB/s | 2.10 GiB/s | 763 MiB/s | 825 MiB/s | 236 MiB/s | 394 MiB/s | 19 MiB/s |
| aes-256-gcm-siv/encrypt vs rustcrypto | 2.56 GiB/s | 3.26 GiB/s | 2.43 GiB/s | 2.22 GiB/s | 1.56 GiB/s | 1.87 GiB/s | 331 MiB/s | 1.39 GiB/s | 20 MiB/s |
| chacha20-poly1305/encrypt vs rustcrypto | 2.17 GiB/s | 3.05 GiB/s | 1.92 GiB/s | 1.84 GiB/s | 354 MiB/s | 397 MiB/s | 492 MiB/s | 436 MiB/s | 60 MiB/s |
| xchacha20-poly1305/encrypt vs rustcrypto | 2.17 GiB/s | 3.08 GiB/s | 1.94 GiB/s | 1.83 GiB/s | 354 MiB/s | 398 MiB/s | 495 MiB/s | 469 MiB/s | 60 MiB/s |
| aegis-256/encrypt vs aegis-crate | 8.31 GiB/s | 9.03 GiB/s | 6.14 GiB/s | 5.58 GiB/s | 4.86 GiB/s | 5.22 GiB/s | 301 MiB/s | 4.69 GiB/s | 8 MiB/s |

## 64-Byte Latency

These rows use the `rscrypto` midpoint for the 64-byte case. This captures short-message call overhead and dispatch cost better than the 1 MiB throughput table.

| Primitive | Zen4 | Zen5 | SPR | ICL | Grav3 | Grav4 | s390x | POWER10 | RISC-V |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| crc32c vs crc32c | 3.17 ns | 2.16 ns | 2.91 ns | 3.39 ns | 4.45 ns | 3.75 ns | 36.54 ns | 36.25 ns | 164.11 ns |
| sha256 vs sha2 | 73.42 ns | 63.46 ns | 91.30 ns | 101.23 ns | 88.59 ns | 73.82 ns | 342.21 ns | 412.98 ns | 2.31 us |
| sha512 vs sha2 | 192.27 ns | 126.19 ns | 214.06 ns | 225.24 ns | 141.69 ns | 135.74 ns | 192.23 ns | 270.93 ns | 1.80 us |
| sha3-256 vs sha3 | 292.23 ns | 210.10 ns | 303.66 ns | 348.91 ns | 342.26 ns | 291.24 ns | 533.41 ns | 297.28 ns | 1.56 us |
| shake256 vs sha3 | 290.86 ns | 211.17 ns | 303.99 ns | 356.72 ns | 344.37 ns | 291.06 ns | 563.24 ns | 300.75 ns | 1.51 us |
| blake2/blake2b512 vs rustcrypto | 176.67 ns | 188.69 ns | 173.72 ns | 137.32 ns | 173.67 ns | 155.83 ns | 239.23 ns | 194.71 ns | 1.01 us |
| blake2/blake2s256 vs rustcrypto | 128.41 ns | 144.38 ns | 129.42 ns | 107.51 ns | 147.84 ns | 132.91 ns | 167.73 ns | 139.54 ns | 772.49 ns |
| blake3 vs blake3 | 57.45 ns | 58.69 ns | 50.65 ns | 52.76 ns | 110.43 ns | 101.61 ns | 128.80 ns | 114.91 ns | 514.46 ns |
| ascon-hash256 vs ascon-hash | 441.36 ns | 324.92 ns | 658.44 ns | 641.36 ns | 492.05 ns | 437.25 ns | 1.19 us | 736.61 ns | 2.63 us |
| xxh3-64 vs xxhash-rust | 3.90 ns | 2.67 ns | 6.39 ns | 7.66 ns | 5.41 ns | 3.90 ns | 11.58 ns | 5.95 ns | 79.09 ns |
| rapidhash-64 vs rapidhash | 3.27 ns | 2.49 ns | 3.58 ns | 4.38 ns | 5.67 ns | 4.10 ns | 17.07 ns | 6.49 ns | 96.21 ns |
| hmac-sha256 vs rustcrypto | 178.56 ns | 169.54 ns | 11.28 us | 249.70 ns | 193.33 ns | 165.00 ns | 506.98 ns | 1.04 us | 5.89 us |
| aes-256-gcm/encrypt vs rustcrypto | 111.08 ns | 89.09 ns | 121.72 ns | 122.75 ns | 170.73 ns | 153.15 ns | 405.16 ns | 253.08 ns | 5.67 us |
| aes-256-gcm-siv/encrypt vs rustcrypto | 198.91 ns | 164.69 ns | 222.42 ns | 224.37 ns | 303.65 ns | 303.15 ns | 473.44 ns | 438.71 ns | 12.89 us |
| chacha20-poly1305/encrypt vs rustcrypto | 300.39 ns | 317.65 ns | 325.13 ns | 315.10 ns | 412.81 ns | 356.38 ns | 473.82 ns | 467.51 ns | 2.11 us |
| xchacha20-poly1305/encrypt vs rustcrypto | 407.65 ns | 453.57 ns | 435.79 ns | 412.37 ns | 526.01 ns | 449.02 ns | 621.00 ns | 590.05 ns | 2.65 us |
| aegis-256/encrypt vs aegis-crate | 42.01 ns | 34.66 ns | 45.41 ns | 48.40 ns | 93.68 ns | 84.16 ns | 1.31 us | 86.53 ns | 44.30 us |

## Losses Over 5 Percent

Complete list: 342 cases below the 0.95x budget.

| Speedup | Platform | Category | Group | Case | rscrypto | Competitor | Competitor Time |
|---:|---|---|---|---|---:|---|---:|
| 0.01x | SPR | SHA-2 | `sha256/streaming vs sha2` | `64B` | 81.06 ms | sha2 | 724.50 us |
| 0.01x | SPR | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `64B` | 81.07 ms | rustcrypto | 735.48 us |
| 0.01x | SPR | SHA-2 | `sha256/streaming vs sha2` | `4096B` | 81.44 ms | sha2 | 975.80 us |
| 0.01x | SPR | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `4096B` | 81.70 ms | rustcrypto | 993.97 us |
| 0.29x | Zen5 | XXH3 | `xxh3-64 vs xxhash-rust` | `256` | 21.84 ns | xxhash-rust | 6.26 ns |
| 0.42x | SPR | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `32` | 155.82 ns | rustcrypto | 64.79 ns |
| 0.42x | Zen4 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `32` | 144.03 ns | rustcrypto | 60.37 ns |
| 0.43x | Zen5 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `32` | 119.17 ns | rustcrypto | 51.02 ns |
| 0.44x | ICL | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `32` | 145.15 ns | rustcrypto | 63.47 ns |
| 0.50x | ICL | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `0` | 78.55 ns | rustcrypto | 39.20 ns |
| 0.52x | SPR | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `0` | 77.81 ns | rustcrypto | 40.75 ns |
| 0.53x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1` | 16.25 ns | rapidhash | 8.54 ns |
| 0.53x | ICL | XXH3 | `xxh3-64 vs xxhash-rust` | `64` | 7.66 ns | xxhash-rust | 4.04 ns |
| 0.54x | RISC-V | Checksums | `crc64-nvme vs crc-fast` | `1048576` | 4.02 ms | crc-fast | 2.16 ms |
| 0.54x | SPR | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `32` | 135.60 ns | rustcrypto | 73.63 ns |
| 0.55x | SPR | XXH3 | `xxh3-64 vs xxhash-rust` | `64` | 6.39 ns | xxhash-rust | 3.50 ns |
| 0.56x | Zen5 | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `32` | 107.57 ns | rustcrypto | 60.11 ns |
| 0.56x | Zen5 | XXH3 | `xxh3-64 vs xxhash-rust` | `1024` | 25.34 ns | xxhash-rust | 14.26 ns |
| 0.57x | RISC-V | Checksums | `crc32c vs crc32c` | `32` | 121.05 ns | crc32c | 69.17 ns |
| 0.59x | Zen4 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `0` | 67.38 ns | rustcrypto | 39.55 ns |
| 0.60x | Zen5 | AEAD | `chacha20-poly1305/decrypt vs rustcrypto` | `256` | 642.08 ns | rustcrypto | 386.11 ns |
| 0.61x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `1048576` | 4.02 ms | crc64fast-nvme | 2.47 ms |
| 0.62x | Zen5 | AEAD | `chacha20-poly1305/encrypt vs rustcrypto` | `256` | 659.36 ns | rustcrypto | 406.19 ns |
| 0.62x | SPR | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1` | 122.38 ns | rustcrypto | 76.12 ns |
| 0.62x | s390x | Checksums | `crc24-openpgp vs crc` | `64` | 282.35 ns | crc | 176.07 ns |
| 0.62x | ICL | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1` | 114.03 ns | rustcrypto | 71.19 ns |
| 0.63x | Zen5 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `0` | 52.52 ns | rustcrypto | 33.00 ns |
| 0.64x | Zen4 | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `32` | 123.64 ns | rustcrypto | 79.21 ns |
| 0.65x | RISC-V | Checksums | `crc32 vs crc32fast` | `64` | 166.46 ns | crc32fast | 108.44 ns |
| 0.65x | Zen4 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `64` | 111.08 ns | rustcrypto | 72.62 ns |
| 0.66x | Zen5 | AEAD | `xchacha20-poly1305/encrypt vs rustcrypto` | `256` | 795.17 ns | rustcrypto | 522.14 ns |
| 0.66x | ICL | AEAD | `chacha20-poly1305/encrypt vs rustcrypto` | `256` | 639.99 ns | rustcrypto | 420.90 ns |
| 0.66x | ICL | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `64` | 122.75 ns | rustcrypto | 81.38 ns |
| 0.67x | Zen5 | AEAD | `xchacha20-poly1305/decrypt vs rustcrypto` | `256` | 772.00 ns | rustcrypto | 515.69 ns |
| 0.67x | RISC-V | Checksums | `crc64-xz vs crc64fast` | `1048576` | 4.00 ms | crc64fast | 2.68 ms |
| 0.67x | SPR | AEAD | `chacha20-poly1305/decrypt vs rustcrypto` | `256` | 657.78 ns | rustcrypto | 441.48 ns |
| 0.67x | Zen4 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1` | 108.18 ns | rustcrypto | 72.67 ns |
| 0.67x | SPR | AEAD | `xchacha20-poly1305/decrypt vs rustcrypto` | `256` | 765.07 ns | rustcrypto | 515.51 ns |
| 0.68x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `32` | 16.50 ns | rapidhash | 11.14 ns |
| 0.68x | SPR | Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | `64` | 30.00 us | rustcrypto | 20.27 us |
| 0.68x | ICL | AEAD | `chacha20-poly1305/decrypt vs rustcrypto` | `256` | 641.88 ns | rustcrypto | 436.95 ns |
| 0.68x | s390x | Checksums | `crc16-ccitt vs crc` | `1` | 13.15 ns | crc | 8.98 ns |
| 0.69x | s390x | XXH3 | `xxh3-64 vs xxhash-rust` | `32` | 17.66 ns | xxhash-rust | 12.17 ns |
| 0.69x | SPR | XXH3 | `xxh3-128 vs xxhash-rust` | `256` | 21.18 ns | xxhash-rust | 14.69 ns |
| 0.69x | SPR | AEAD | `xchacha20-poly1305/encrypt vs rustcrypto` | `256` | 764.76 ns | rustcrypto | 530.22 ns |
| 0.69x | ICL | AEAD | `xchacha20-poly1305/encrypt vs rustcrypto` | `256` | 737.95 ns | rustcrypto | 511.68 ns |
| 0.70x | ICL | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `32` | 128.77 ns | rustcrypto | 89.76 ns |
| 0.70x | Zen5 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1` | 85.36 ns | rustcrypto | 59.52 ns |
| 0.70x | Zen4 | AEAD | `chacha20-poly1305/encrypt vs rustcrypto` | `256` | 619.80 ns | rustcrypto | 434.45 ns |
| 0.70x | POWER10 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1` | 9.74 ns | rapidhash | 6.86 ns |
| 0.72x | Zen4 | AEAD | `xchacha20-poly1305/encrypt vs rustcrypto` | `256` | 726.12 ns | rustcrypto | 519.86 ns |
| 0.72x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `32` | 19.99 ns | rapidhash | 14.35 ns |
| 0.72x | SPR | Auth/KDF | `hmac-sha256 vs rustcrypto` | `1024` | 16.71 us | rustcrypto | 12.01 us |
| 0.72x | ICL | AEAD | `xchacha20-poly1305/decrypt vs rustcrypto` | `256` | 737.45 ns | rustcrypto | 531.80 ns |
| 0.73x | s390x | BLAKE3 | `blake3/xof vs blake3` | `1024` | 2.58 us | blake3 | 1.87 us |
| 0.73x | POWER10 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `0` | 6.70 ns | rapidhash | 4.87 ns |
| 0.73x | SPR | AEAD | `chacha20-poly1305/encrypt vs rustcrypto` | `256` | 653.23 ns | rustcrypto | 476.61 ns |
| 0.73x | POWER10 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `256` | 57.75 ns | rapidhash | 42.30 ns |
| 0.73x | RISC-V | Checksums | `crc32c vs crc32c` | `64` | 164.11 ns | crc32c | 120.38 ns |
| 0.74x | ICL | Checksums | `crc32 vs crc32fast` | `256` | 16.34 ns | crc32fast | 12.05 ns |
| 0.74x | RISC-V | BLAKE3 | `blake3/streaming vs blake3` | `65536B` | 10.92 ms | blake3 | 8.10 ms |
| 0.75x | RISC-V | BLAKE3 | `blake3/streaming vs blake3` | `4096B` | 10.98 ms | blake3 | 8.23 ms |
| 0.75x | Zen4 | XXH3 | `xxh3-64 vs xxhash-rust` | `256` | 12.66 ns | xxhash-rust | 9.50 ns |
| 0.75x | SPR | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `64` | 121.72 ns | rustcrypto | 91.31 ns |
| 0.75x | RISC-V | BLAKE2 | `blake2/blake2b256 vs rustcrypto` | `1048576` | 8.85 ms | rustcrypto | 6.64 ms |
| 0.75x | s390x | BLAKE3 | `blake3 vs blake3` | `1024` | 2.45 us | blake3 | 1.84 us |
| 0.75x | SPR | Auth/KDF | `hmac-sha256 vs rustcrypto` | `4096` | 18.59 us | rustcrypto | 13.96 us |
| 0.75x | SPR | Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | `32` | 19.93 us | rustcrypto | 15.02 us |
| 0.75x | Zen5 | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `64` | 89.09 ns | rustcrypto | 67.15 ns |
| 0.76x | s390x | BLAKE3 | `blake3/keyed vs blake3` | `1024` | 2.45 us | blake3 | 1.85 us |
| 0.76x | RISC-V | BLAKE2 | `blake2/blake2b512 vs rustcrypto` | `1048576` | 8.79 ms | rustcrypto | 6.66 ms |
| 0.76x | Zen4 | AEAD | `xchacha20-poly1305/decrypt vs rustcrypto` | `256` | 721.83 ns | rustcrypto | 547.83 ns |
| 0.76x | RISC-V | BLAKE3 | `blake3/streaming vs blake3` | `16384B` | 10.70 ms | blake3 | 8.13 ms |
| 0.76x | s390x | Checksums | `crc64-nvme vs crc64fast-nvme` | `1` | 12.10 ns | crc64fast-nvme | 9.24 ns |
| 0.77x | Zen4 | AEAD | `chacha20-poly1305/decrypt vs rustcrypto` | `256` | 604.03 ns | rustcrypto | 465.01 ns |
| 0.77x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `0` | 82.76 ns | aegis-crate | 63.93 ns |
| 0.77x | Grav4 | XXH3 | `xxh3-128 vs xxhash-rust` | `256` | 26.01 ns | xxhash-rust | 20.10 ns |
| 0.78x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `0` | 3.82 ns | rapidhash | 2.96 ns |
| 0.78x | RISC-V | Checksums | `crc64-xz vs crc64fast` | `32` | 148.74 ns | crc64fast | 115.60 ns |
| 0.78x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `0` | 69.16 ns | aegis-crate | 54.01 ns |
| 0.78x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `64` | 84.16 ns | aegis-crate | 65.78 ns |
| 0.78x | POWER10 | Checksums | `crc64-xz vs crc64fast` | `64` | 41.06 ns | crc64fast | 32.22 ns |
| 0.79x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `32` | 148.30 ns | crc64fast-nvme | 116.55 ns |
| 0.79x | SPR | RapidHash | `rapidhash-64 vs rapidhash` | `0` | 2.09 ns | rapidhash | 1.65 ns |
| 0.79x | s390x | Checksums | `crc16-ibm vs crc` | `1` | 11.39 ns | crc | 9.03 ns |
| 0.79x | RISC-V | Checksums | `crc64-xz vs crc64fast` | `262144` | 722.08 us | crc64fast | 573.87 us |
| 0.80x | SPR | XXH3 | `xxh3-64 vs xxhash-rust` | `256` | 14.53 ns | xxhash-rust | 11.55 ns |
| 0.80x | POWER10 | Checksums | `crc32 vs crc32fast` | `64` | 36.31 ns | crc32fast | 28.90 ns |
| 0.80x | POWER10 | Checksums | `crc64-nvme vs crc64fast-nvme` | `64` | 40.93 ns | crc64fast-nvme | 32.66 ns |
| 0.80x | SPR | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `1` | 103.41 ns | rustcrypto | 82.65 ns |
| 0.80x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `32` | 89.06 ns | aegis-crate | 71.40 ns |
| 0.81x | Grav4 | Checksums | `crc32c vs crc-fast` | `256` | 14.94 ns | crc-fast | 12.03 ns |
| 0.81x | RISC-V | BLAKE2 | `blake2/keyed/blake2b256 vs rustcrypto` | `262144` | 2.08 ms | rustcrypto | 1.68 ms |
| 0.81x | s390x | BLAKE3 | `blake3/derive-key vs blake3` | `1024` | 2.44 us | blake3 | 1.98 us |
| 0.81x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1` | 89.71 ns | aegis-crate | 72.84 ns |
| 0.81x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `32` | 75.05 ns | aegis-crate | 61.05 ns |
| 0.82x | ICL | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `4096B` | 916.36 us | rustcrypto | 749.73 us |
| 0.82x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `64` | 93.68 ns | aegis-crate | 76.69 ns |
| 0.82x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1` | 75.13 ns | aegis-crate | 61.52 ns |
| 0.82x | ICL | BLAKE3 | `blake3/keyed vs blake3` | `256` | 257.55 ns | blake3 | 211.94 ns |
| 0.82x | RISC-V | BLAKE3 | `blake3/xof vs blake3` | `262144` | 2.45 ms | blake3 | 2.02 ms |
| 0.82x | Zen4 | XXH3 | `xxh3-128 vs xxhash-rust` | `256` | 15.69 ns | xxhash-rust | 12.93 ns |
| 0.83x | RISC-V | BLAKE2 | `blake2/keyed/blake2b512 vs rustcrypto` | `262144` | 2.00 ms | rustcrypto | 1.65 ms |
| 0.83x | ICL | SHA-2 | `sha256/streaming vs sha2` | `4096B` | 916.51 us | sha2 | 759.11 us |
| 0.83x | SPR | Auth/KDF | `hmac-sha256 vs rustcrypto` | `16384` | 26.38 us | rustcrypto | 21.91 us |
| 0.83x | RISC-V | Checksums | `crc64-xz vs crc64fast` | `64` | 202.89 ns | crc64fast | 168.99 ns |
| 0.84x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `64` | 203.11 ns | crc64fast-nvme | 171.05 ns |
| 0.85x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `262144` | 13.15 ms | rustcrypto | 11.13 ms |
| 0.85x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1048576` | 52.47 ms | rustcrypto | 44.42 ms |
| 0.85x | ICL | XXH3 | `xxh3-128 vs xxhash-rust` | `256` | 21.73 ns | xxhash-rust | 18.41 ns |
| 0.85x | s390x | Checksums | `crc64-nvme vs crc64fast-nvme` | `64` | 48.28 ns | crc64fast-nvme | 41.00 ns |
| 0.85x | s390x | XXH3 | `xxh3-128 vs xxhash-rust` | `0` | 5.88 ns | xxhash-rust | 5.00 ns |
| 0.85x | Zen5 | BLAKE3 | `blake3/keyed vs blake3` | `256` | 347.50 ns | blake3 | 295.58 ns |
| 0.85x | POWER10 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1024` | 122.37 ns | rapidhash | 104.37 ns |
| 0.85x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `65536` | 3.23 ms | rustcrypto | 2.75 ms |
| 0.85x | RISC-V | RapidHash | `rapidhash-64 vs rapidhash` | `64` | 96.21 ns | rapidhash | 82.26 ns |
| 0.86x | Zen5 | BLAKE3 | `blake3/streaming vs blake3` | `64B` | 1.60 ms | blake3 | 1.37 ms |
| 0.86x | RISC-V | BLAKE2 | `blake2/keyed/blake2s128 vs rustcrypto` | `1` | 1.70 us | rustcrypto | 1.46 us |
| 0.86x | s390x | Checksums | `crc32 vs crc32fast` | `64` | 36.56 ns | crc32fast | 31.39 ns |
| 0.86x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `65536` | 6.91 us | rapidhash | 5.95 us |
| 0.86x | Grav3 | XXH3 | `xxh3-128 vs xxhash-rust` | `256` | 33.42 ns | xxhash-rust | 28.82 ns |
| 0.86x | POWER10 | RapidHash | `rapidhash-64 vs rapidhash` | `1` | 5.44 ns | rapidhash | 4.70 ns |
| 0.86x | SPR | Checksums | `crc32 vs crc32fast` | `256` | 12.99 ns | crc32fast | 11.21 ns |
| 0.86x | RISC-V | BLAKE2 | `blake2/keyed/blake2s128 vs rustcrypto` | `32` | 1.71 us | rustcrypto | 1.48 us |
| 0.86x | Zen4 | XXH3 | `xxh3-64 vs xxhash-rust` | `64` | 3.90 ns | xxhash-rust | 3.37 ns |
| 0.86x | SPR | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `0` | 57.64 ns | rustcrypto | 49.85 ns |
| 0.87x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `16384` | 1.62 us | rapidhash | 1.41 us |
| 0.87x | ICL | SHA-2 | `sha256/streaming vs sha2` | `64B` | 962.54 us | sha2 | 835.45 us |
| 0.87x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `256` | 128.85 ns | aegis-crate | 111.90 ns |
| 0.87x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1048576` | 216.63 us | rapidhash | 188.15 us |
| 0.87x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `65536` | 13.54 us | rapidhash | 11.77 us |
| 0.87x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `16384` | 3.42 us | rapidhash | 2.97 us |
| 0.87x | RISC-V | BLAKE2 | `blake2/keyed/blake2s128 vs rustcrypto` | `64` | 1.69 us | rustcrypto | 1.47 us |
| 0.87x | Grav4 | Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | `32` | 190.65 ns | rustcrypto | 166.14 ns |
| 0.87x | POWER10 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `256` | 28.27 ns | rapidhash | 24.64 ns |
| 0.87x | ICL | BLAKE3 | `blake3/streaming vs blake3` | `64B` | 1.11 ms | blake3 | 970.23 us |
| 0.87x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `262144` | 54.10 us | rapidhash | 47.20 us |
| 0.87x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `262144` | 27.18 us | rapidhash | 23.74 us |
| 0.87x | SPR | BLAKE3 | `blake3/keyed vs blake3` | `16384` | 3.12 us | blake3 | 2.73 us |
| 0.88x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `4096` | 410.22 ns | rapidhash | 359.01 ns |
| 0.88x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1048576` | 108.84 us | rapidhash | 95.28 us |
| 0.88x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `256` | 78.51 ns | rapidhash | 68.73 ns |
| 0.88x | SPR | BLAKE3 | `blake3/xof vs blake3` | `0` | 58.98 ns | blake3 | 51.65 ns |
| 0.88x | RISC-V | Checksums | `crc64-xz vs crc64fast` | `65536` | 161.23 us | crc64fast | 141.21 us |
| 0.88x | SPR | BLAKE3 | `blake3/xof vs blake3` | `16384` | 3.13 us | blake3 | 2.74 us |
| 0.88x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `0` | 4.25 ns | rapidhash | 3.74 ns |
| 0.88x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `256` | 112.98 ns | aegis-crate | 99.26 ns |
| 0.88x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `0` | 2.67 us | rustcrypto | 2.35 us |
| 0.88x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `65536` | 3.26 ms | rustcrypto | 2.87 ms |
| 0.88x | RISC-V | BLAKE2 | `blake2/keyed/blake2s256 vs rustcrypto` | `1` | 1.69 us | rustcrypto | 1.49 us |
| 0.88x | Grav3 | Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | `32` | 198.18 ns | rustcrypto | 174.39 ns |
| 0.88x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `1048576` | 204.44 us | rapidhash | 179.99 us |
| 0.88x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `16384` | 1.72 us | rapidhash | 1.51 us |
| 0.88x | RISC-V | Checksums | `crc32 vs crc32fast` | `1024` | 1.66 us | crc32fast | 1.46 us |
| 0.88x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `4096` | 445.19 ns | rapidhash | 392.11 ns |
| 0.88x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `1048576` | 49.90 ms | rustcrypto | 43.96 ms |
| 0.88x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `4096` | 872.78 ns | rapidhash | 769.04 ns |
| 0.88x | s390x | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1024` | 229.55 ns | rapidhash | 202.43 ns |
| 0.88x | s390x | Ascon | `ascon-hash256 vs ascon-hash` | `0` | 439.19 ns | ascon-hash | 388.05 ns |
| 0.89x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `1048576` | 102.78 us | rapidhash | 90.98 us |
| 0.89x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `64` | 17.07 ns | rapidhash | 15.13 ns |
| 0.89x | SPR | BLAKE3 | `blake3 vs blake3` | `65536` | 11.62 us | blake3 | 10.31 us |
| 0.89x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `16384` | 778.61 us | rustcrypto | 691.00 us |
| 0.89x | SPR | BLAKE3 | `blake3/streaming vs blake3` | `64B` | 1.20 ms | blake3 | 1.07 ms |
| 0.89x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `16384` | 794.24 us | rustcrypto | 705.48 us |
| 0.89x | SPR | Checksums | `crc32c vs crc-fast` | `1024` | 19.05 ns | crc-fast | 16.93 ns |
| 0.89x | ICL | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `64B` | 954.72 us | rustcrypto | 848.74 us |
| 0.89x | Grav4 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `256` | 29.22 ns | rapidhash | 26.02 ns |
| 0.89x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1024` | 118.97 ns | rapidhash | 105.97 ns |
| 0.89x | s390x | Checksums | `crc32c vs crc32c` | `1` | 11.41 ns | crc32c | 10.17 ns |
| 0.89x | Zen4 | XXH3 | `xxh3-128 vs xxhash-rust` | `1` | 7.96 ns | xxhash-rust | 7.09 ns |
| 0.89x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `262144` | 25.74 us | rapidhash | 22.97 us |
| 0.89x | RISC-V | Checksums | `crc64-nvme vs crc-fast` | `65536` | 144.75 us | crc-fast | 129.19 us |
| 0.89x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `262144` | 13.07 ms | rustcrypto | 11.69 ms |
| 0.89x | ICL | RapidHash | `rapidhash-64 vs rapidhash` | `32` | 2.59 ns | rapidhash | 2.32 ns |
| 0.90x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `256` | 20.63 ns | rapidhash | 18.49 ns |
| 0.90x | Grav4 | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `64B` | 805.51 us | rustcrypto | 722.13 us |
| 0.90x | RISC-V | Checksums | `crc32 vs crc32fast` | `256` | 427.58 ns | crc32fast | 383.55 ns |
| 0.90x | ICL | Auth/KDF | `pbkdf2-sha256/iters=1 vs rustcrypto` | `32` | 273.78 ns | rustcrypto | 245.73 ns |
| 0.90x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `1048576` | 51.81 ms | rustcrypto | 46.51 ms |
| 0.90x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `1024` | 217.73 ns | rapidhash | 195.46 ns |
| 0.90x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `262144` | 51.78 us | rapidhash | 46.52 us |
| 0.90x | ICL | BLAKE3 | `blake3 vs blake3` | `256` | 256.36 ns | blake3 | 230.55 ns |
| 0.90x | RISC-V | BLAKE3 | `blake3/xof vs blake3` | `65536` | 571.77 us | blake3 | 514.46 us |
| 0.90x | Grav4 | XXH3 | `xxh3-128 vs xxhash-rust` | `1` | 2.78 ns | xxhash-rust | 2.51 ns |
| 0.90x | Grav4 | XXH3 | `xxh3-128 vs xxhash-rust` | `32` | 4.64 ns | xxhash-rust | 4.18 ns |
| 0.90x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `4096` | 198.55 us | rustcrypto | 179.47 us |
| 0.90x | s390x | RapidHash | `rapidhash-v3-64 vs rapidhash` | `256` | 43.13 ns | rapidhash | 39.00 ns |
| 0.90x | Grav4 | SHA-2 | `sha256/streaming vs sha2` | `64B` | 798.40 us | sha2 | 722.08 us |
| 0.90x | s390x | Checksums | `crc64-xz vs crc64fast` | `64` | 48.32 ns | crc64fast | 43.72 ns |
| 0.91x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `256` | 544.32 ns | crc64fast-nvme | 492.67 ns |
| 0.91x | SPR | Auth/KDF | `hmac-sha256 vs rustcrypto` | `65536` | 58.23 us | rustcrypto | 52.73 us |
| 0.91x | Grav4 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `256` | 14.85 ns | rapidhash | 13.45 ns |
| 0.91x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `65536` | 12.89 us | rapidhash | 11.70 us |
| 0.91x | RISC-V | Checksums | `crc64-nvme vs crc-fast` | `262144` | 581.69 us | crc-fast | 528.12 us |
| 0.91x | Zen5 | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `1` | 74.42 ns | rustcrypto | 67.60 ns |
| 0.91x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `16384` | 3.25 us | rapidhash | 2.95 us |
| 0.91x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `256` | 40.16 ns | rapidhash | 36.53 ns |
| 0.91x | RISC-V | AEAD | `xchacha20-poly1305/encrypt vs rustcrypto` | `1` | 2.50 us | rustcrypto | 2.28 us |
| 0.91x | RISC-V | Checksums | `crc64-xz vs crc64fast` | `256` | 541.83 ns | crc64fast | 493.28 ns |
| 0.91x | Grav4 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1024` | 252.39 ns | aegis-crate | 229.97 ns |
| 0.91x | Grav3 | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` | `64B` | 868.56 us | rustcrypto | 791.83 us |
| 0.91x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `1024` | 1.96 us | crc64fast-nvme | 1.79 us |
| 0.91x | Grav3 | SHA-2 | `sha256/streaming vs sha2` | `64B` | 868.90 us | sha2 | 793.17 us |
| 0.91x | Zen4 | XXH3 | `xxh3-128 vs xxhash-rust` | `0` | 6.87 ns | xxhash-rust | 6.27 ns |
| 0.91x | RISC-V | BLAKE2 | `blake2/keyed/blake2s128 vs rustcrypto` | `0` | 902.57 ns | rustcrypto | 824.62 ns |
| 0.91x | Zen5 | XXH3 | `xxh3-64 vs xxhash-rust` | `64` | 2.67 ns | xxhash-rust | 2.44 ns |
| 0.91x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `65536` | 6.40 us | rapidhash | 5.86 us |
| 0.91x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1` | 1.83 ns | rapidhash | 1.67 ns |
| 0.92x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `262144` | 12.29 ms | rustcrypto | 11.25 ms |
| 0.92x | SPR | BLAKE3 | `blake3/xof vs blake3` | `32` | 58.87 ns | blake3 | 53.91 ns |
| 0.92x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `4096` | 193.06 us | rustcrypto | 176.97 us |
| 0.92x | RISC-V | BLAKE2 | `blake2/keyed/blake2s256 vs rustcrypto` | `32` | 1.68 us | rustcrypto | 1.55 us |
| 0.92x | SPR | RapidHash | `rapidhash-v3-128 vs rapidhash` | `4096` | 247.38 ns | rapidhash | 227.33 ns |
| 0.92x | Grav4 | XXH3 | `xxh3-128 vs xxhash-rust` | `1024` | 56.85 ns | xxhash-rust | 52.27 ns |
| 0.92x | Grav3 | Checksums | `crc32 vs crc-fast` | `262144` | 6.50 us | crc-fast | 5.98 us |
| 0.92x | Grav3 | Checksums | `crc32c vs crc-fast` | `262144` | 6.50 us | crc-fast | 5.98 us |
| 0.92x | Zen4 | Auth/KDF | `pbkdf2-sha256/iters=1000 vs rustcrypto` | `64` | 166.93 us | rustcrypto | 153.57 us |
| 0.92x | Zen4 | Auth/KDF | `pbkdf2-sha256/iters=1000 vs rustcrypto` | `32` | 83.52 us | rustcrypto | 76.85 us |
| 0.92x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `262144` | 12.35 us | rapidhash | 11.36 us |
| 0.92x | Grav3 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1024` | 276.78 ns | aegis-crate | 254.90 ns |
| 0.92x | SPR | AEAD | `aegis-256/encrypt vs aegis-crate` | `0` | 36.44 ns | aegis-crate | 33.56 ns |
| 0.92x | SPR | RapidHash | `rapidhash-v3-128 vs rapidhash` | `16384` | 921.61 ns | rapidhash | 848.85 ns |
| 0.92x | Zen4 | Auth/KDF | `pbkdf2-sha256/iters=100 vs rustcrypto` | `64` | 16.82 us | rustcrypto | 15.51 us |
| 0.92x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `262144` | 24.66 us | rapidhash | 22.73 us |
| 0.92x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `262144` | 7.31 us | rapidhash | 6.74 us |
| 0.92x | s390x | RapidHash | `rapidhash-64 vs rapidhash` | `1024` | 109.07 ns | rapidhash | 100.62 ns |
| 0.92x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `65536` | 3.08 us | rapidhash | 2.85 us |
| 0.92x | SPR | BLAKE3 | `blake3/xof vs blake3` | `64` | 65.38 ns | blake3 | 60.36 ns |
| 0.92x | Grav3 | XXH3 | `xxh3-128 vs xxhash-rust` | `1024` | 68.41 ns | xxhash-rust | 63.16 ns |
| 0.92x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `16384` | 764.40 us | rustcrypto | 706.08 us |
| 0.92x | Zen4 | Auth/KDF | `pbkdf2-sha256/iters=100 vs rustcrypto` | `32` | 8.45 us | rustcrypto | 7.80 us |
| 0.92x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `16384` | 766.76 us | rustcrypto | 708.36 us |
| 0.92x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `65536` | 6.17 us | rapidhash | 5.70 us |
| 0.92x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1048576` | 49.95 us | rapidhash | 46.16 us |
| 0.92x | Grav3 | Checksums | `crc32 vs crc-fast` | `65536` | 1.72 us | crc-fast | 1.59 us |
| 0.92x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `65536` | 2.98 ms | rustcrypto | 2.76 ms |
| 0.92x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `1024` | 50.28 us | rustcrypto | 46.49 us |
| 0.93x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `16384` | 462.07 ns | rapidhash | 427.79 ns |
| 0.93x | POWER10 | BLAKE2 | `blake2/keyed/blake2b512 vs rustcrypto` | `32` | 408.45 ns | rustcrypto | 378.52 ns |
| 0.93x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1048576` | 99.43 us | rapidhash | 92.16 us |
| 0.93x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `16384` | 773.88 ns | rapidhash | 717.43 ns |
| 0.93x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `65536` | 1.85 us | rapidhash | 1.71 us |
| 0.93x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `16384` | 1.55 us | rapidhash | 1.43 us |
| 0.93x | Grav3 | BLAKE3 | `blake3/streaming vs blake3` | `4096B` | 956.80 us | blake3 | 887.24 us |
| 0.93x | Grav3 | Checksums | `crc32c vs crc-fast` | `65536` | 1.72 us | crc-fast | 1.59 us |
| 0.93x | SPR | BLAKE3 | `blake3/derive-key vs blake3` | `16384` | 3.12 us | blake3 | 2.90 us |
| 0.93x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `4096` | 883.02 ns | aegis-crate | 819.41 ns |
| 0.93x | Grav4 | BLAKE3 | `blake3/keyed vs blake3` | `1` | 110.77 ns | blake3 | 102.80 ns |
| 0.93x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `65536` | 3.13 ms | rustcrypto | 2.91 ms |
| 0.93x | Zen5 | BLAKE3 | `blake3 vs blake3` | `256` | 346.93 ns | blake3 | 322.54 ns |
| 0.93x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `1024` | 51.01 us | rustcrypto | 47.43 us |
| 0.93x | RISC-V | AEAD | `aes-256-gcm-siv/encrypt vs rustcrypto` | `1048576` | 49.73 ms | rustcrypto | 46.25 ms |
| 0.93x | Zen5 | Auth/KDF | `pbkdf2-sha256/iters=1000 vs rustcrypto` | `64` | 142.58 us | rustcrypto | 132.60 us |
| 0.93x | Zen5 | Auth/KDF | `pbkdf2-sha256/iters=1000 vs rustcrypto` | `32` | 71.31 us | rustcrypto | 66.33 us |
| 0.93x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `65536` | 12.75 us | aegis-crate | 11.86 us |
| 0.93x | SPR | RapidHash | `rapidhash-v3-128 vs rapidhash` | `65536` | 3.67 us | rapidhash | 3.42 us |
| 0.93x | RISC-V | AEAD | `aes-256-gcm-siv/decrypt vs rustcrypto` | `262144` | 12.46 ms | rustcrypto | 11.60 ms |
| 0.93x | Grav3 | BLAKE3 | `blake3/keyed vs blake3` | `1` | 118.97 ns | blake3 | 110.74 ns |
| 0.93x | RISC-V | RapidHash | `rapidhash-64 vs rapidhash` | `0` | 20.24 ns | rapidhash | 18.84 ns |
| 0.93x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1048576` | 208.19 us | aegis-crate | 193.95 us |
| 0.93x | Zen5 | Auth/KDF | `pbkdf2-sha256/iters=100 vs rustcrypto` | `64` | 14.35 us | rustcrypto | 13.37 us |
| 0.93x | Zen5 | Auth/KDF | `pbkdf2-sha256/iters=100 vs rustcrypto` | `32` | 7.21 us | rustcrypto | 6.72 us |
| 0.93x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1048576` | 29.16 us | rapidhash | 27.18 us |
| 0.93x | Grav3 | Auth/KDF | `hmac-sha256 vs rustcrypto` | `256` | 380.26 ns | rustcrypto | 354.48 ns |
| 0.93x | SPR | RapidHash | `rapidhash-64 vs rapidhash` | `1` | 2.15 ns | rapidhash | 2.01 ns |
| 0.93x | Zen5 | Checksums | `crc32c vs crc-fast` | `1024` | 16.45 ns | crc-fast | 15.34 ns |
| 0.93x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `16384` | 3.27 us | aegis-crate | 3.05 us |
| 0.93x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `262144` | 50.58 us | aegis-crate | 47.19 us |
| 0.93x | SPR | RapidHash | `rapidhash-v3-128 vs rapidhash` | `1048576` | 58.51 us | rapidhash | 54.61 us |
| 0.93x | Grav3 | Checksums | `crc32 vs crc-fast` | `1048576` | 25.82 us | crc-fast | 24.10 us |
| 0.93x | Zen5 | Auth/KDF | `ed25519/verify vs dalek` | `0` | 20.08 us | dalek | 18.75 us |
| 0.93x | POWER10 | BLAKE2 | `blake2/keyed/blake2s256 vs rustcrypto` | `64` | 296.22 ns | rustcrypto | 276.86 ns |
| 0.93x | Zen5 | Auth/KDF | `ed25519/verify vs dalek` | `1024` | 20.95 us | dalek | 19.58 us |
| 0.94x | RISC-V | Checksums | `crc32 vs crc-fast` | `1024` | 1.66 us | crc-fast | 1.55 us |
| 0.94x | ICL | XXH3 | `xxh3-64 vs xxhash-rust` | `1` | 2.60 ns | xxhash-rust | 2.43 ns |
| 0.94x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1` | 2.72 ns | rapidhash | 2.55 ns |
| 0.94x | POWER10 | BLAKE2 | `blake2/keyed/blake2b512 vs rustcrypto` | `64` | 404.27 ns | rustcrypto | 378.31 ns |
| 0.94x | POWER10 | BLAKE2 | `blake2/keyed/blake2b512 vs rustcrypto` | `1` | 403.48 ns | rustcrypto | 377.64 ns |
| 0.94x | POWER10 | BLAKE2 | `blake2/keyed/blake2b256 vs rustcrypto` | `32` | 406.28 ns | rustcrypto | 380.53 ns |
| 0.94x | SPR | RapidHash | `rapidhash-v3-128 vs rapidhash` | `262144` | 14.71 us | rapidhash | 13.78 us |
| 0.94x | Grav3 | BLAKE3 | `blake3 vs blake3` | `4096` | 3.52 us | blake3 | 3.29 us |
| 0.94x | Grav3 | Checksums | `crc32c vs crc-fast` | `1048576` | 25.81 us | crc-fast | 24.19 us |
| 0.94x | RISC-V | Checksums | `crc64-nvme vs crc64fast-nvme` | `65536` | 144.75 us | crc64fast-nvme | 135.66 us |
| 0.94x | POWER10 | BLAKE2 | `blake2/blake2s256 vs rustcrypto` | `1048576` | 2.15 ms | rustcrypto | 2.01 ms |
| 0.94x | RISC-V | Checksums | `crc32 vs crc32fast` | `262144` | 423.89 us | crc32fast | 397.49 us |
| 0.94x | Zen5 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `16384` | 302.87 ns | rapidhash | 284.01 ns |
| 0.94x | Zen5 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `65536` | 1.21 us | rapidhash | 1.14 us |
| 0.94x | RISC-V | BLAKE2 | `blake2/keyed/blake2s256 vs rustcrypto` | `64` | 1.62 us | rustcrypto | 1.52 us |
| 0.94x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1024` | 32.60 ns | rapidhash | 30.60 ns |
| 0.94x | Grav3 | BLAKE3 | `blake3/keyed vs blake3` | `4096` | 3.51 us | blake3 | 3.30 us |
| 0.94x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `4096` | 201.98 ns | rapidhash | 189.64 ns |
| 0.94x | Grav4 | AEAD | `aegis-256/decrypt vs aegis-crate` | `65536` | 13.30 us | aegis-crate | 12.48 us |
| 0.94x | Zen5 | Auth/KDF | `ed25519/verify vs dalek` | `32` | 19.82 us | dalek | 18.61 us |
| 0.94x | Zen5 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `262144` | 9.62 us | rapidhash | 9.03 us |
| 0.94x | Grav4 | RapidHash | `rapidhash-64 vs rapidhash` | `1` | 1.69 ns | rapidhash | 1.59 ns |
| 0.94x | Zen5 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `16384` | 607.47 ns | rapidhash | 570.45 ns |
| 0.94x | RISC-V | AEAD | `aes-256-gcm/decrypt vs rustcrypto` | `0` | 2.64 us | rustcrypto | 2.48 us |
| 0.94x | Grav3 | Checksums | `crc32c vs crc-fast` | `16384` | 504.46 ns | crc-fast | 474.20 ns |
| 0.94x | s390x | Ascon | `ascon-hash256 vs ascon-hash` | `4096` | 49.84 us | ascon-hash | 46.87 us |
| 0.94x | Zen5 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `65536` | 2.43 us | rapidhash | 2.29 us |
| 0.94x | Grav4 | AEAD | `aegis-256/decrypt vs aegis-crate` | `1048576` | 208.17 us | aegis-crate | 195.84 us |
| 0.94x | Zen4 | BLAKE2 | `blake2/blake2s256 vs rustcrypto` | `0` | 130.34 ns | rustcrypto | 122.62 ns |
| 0.94x | Zen5 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `262144` | 4.81 us | rapidhash | 4.53 us |
| 0.94x | Zen5 | BLAKE2 | `blake2/streaming/blake2b256 vs rustcrypto` | `64B` | 1.43 ms | rustcrypto | 1.35 ms |
| 0.94x | RISC-V | Auth/KDF | `pbkdf2-sha512/iters=1 vs rustcrypto` | `64` | 8.44 us | rustcrypto | 7.94 us |
| 0.94x | SPR | RapidHash | `rapidhash-v3-64 vs rapidhash` | `4096` | 121.43 ns | rapidhash | 114.39 ns |
| 0.94x | POWER10 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `4096` | 400.21 ns | rapidhash | 377.10 ns |
| 0.94x | RISC-V | Checksums | `crc64-xz vs crc64fast` | `1024` | 1.91 us | crc64fast | 1.80 us |
| 0.94x | Grav3 | Auth/KDF | `pbkdf2-sha512/iters=100 vs rustcrypto` | `64` | 29.16 us | rustcrypto | 27.49 us |
| 0.94x | Zen5 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `4096` | 79.88 ns | rapidhash | 75.34 ns |
| 0.94x | Grav4 | AEAD | `aegis-256/decrypt vs aegis-crate` | `262144` | 51.70 us | aegis-crate | 48.78 us |
| 0.94x | s390x | RapidHash | `rapidhash-128 vs rapidhash` | `4096` | 824.09 ns | rapidhash | 777.63 ns |
| 0.94x | Grav3 | Auth/KDF | `pbkdf2-sha512/iters=1000 vs rustcrypto` | `64` | 288.06 us | rustcrypto | 271.85 us |
| 0.94x | POWER10 | BLAKE2 | `blake2/keyed/blake2b256 vs rustcrypto` | `1` | 401.90 ns | rustcrypto | 379.30 ns |
| 0.94x | Grav4 | SHA-2 | `sha256 vs sha2` | `256` | 222.85 ns | sha2 | 210.43 ns |
| 0.94x | s390x | Ascon | `ascon-hash256 vs ascon-hash` | `64` | 1.19 us | ascon-hash | 1.13 us |
| 0.94x | POWER10 | BLAKE2 | `blake2/keyed/blake2b256 vs rustcrypto` | `64` | 398.23 ns | rustcrypto | 376.18 ns |
| 0.94x | POWER10 | BLAKE2 | `blake2/keyed/blake2s128 vs rustcrypto` | `1` | 292.92 ns | rustcrypto | 276.73 ns |
| 0.95x | POWER10 | AEAD | `aegis-256/encrypt vs aegis-crate` | `1024` | 284.08 ns | aegis-crate | 268.49 ns |
| 0.95x | Grav3 | RapidHash | `rapidhash-v3-64 vs rapidhash` | `1024` | 55.70 ns | rapidhash | 52.65 ns |
| 0.95x | RISC-V | BLAKE2 | `blake2/keyed/blake2b256 vs rustcrypto` | `16384` | 111.76 us | rustcrypto | 105.67 us |
| 0.95x | RISC-V | AEAD | `aes-256-gcm/encrypt vs rustcrypto` | `256` | 14.63 us | rustcrypto | 13.84 us |
| 0.95x | RISC-V | BLAKE2 | `blake2/blake2b512 vs rustcrypto` | `262144` | 1.89 ms | rustcrypto | 1.79 ms |
| 0.95x | Grav3 | RapidHash | `rapidhash-64 vs rapidhash` | `1` | 2.02 ns | rapidhash | 1.91 ns |
| 0.95x | Grav3 | Auth/KDF | `pbkdf2-sha512/iters=1000 vs rustcrypto` | `128` | 575.39 us | rustcrypto | 544.65 us |
| 0.95x | ICL | Checksums | `crc32c vs crc-fast` | `1024` | 30.73 ns | crc-fast | 29.09 ns |
| 0.95x | Grav3 | Auth/KDF | `pbkdf2-sha512/iters=100 vs rustcrypto` | `128` | 57.73 us | rustcrypto | 54.66 us |
| 0.95x | RISC-V | BLAKE2 | `blake2/blake2b256 vs rustcrypto` | `16384` | 110.56 us | rustcrypto | 104.70 us |
| 0.95x | RISC-V | BLAKE2 | `blake2/blake2b512 vs rustcrypto` | `4096` | 27.37 us | rustcrypto | 25.94 us |
| 0.95x | Grav3 | XXH3 | `xxh3-64 vs xxhash-rust` | `1` | 2.82 ns | xxhash-rust | 2.67 ns |
| 0.95x | Zen5 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `4096` | 161.94 ns | rapidhash | 153.55 ns |
| 0.95x | Zen4 | Checksums | `crc32 vs crc-fast` | `1024` | 31.65 ns | crc-fast | 30.01 ns |
| 0.95x | Zen4 | BLAKE3 | `blake3/xof vs blake3` | `64` | 90.61 ns | blake3 | 85.94 ns |
| 0.95x | Grav3 | RapidHash | `rapidhash-v3-128 vs rapidhash` | `4096` | 402.43 ns | rapidhash | 381.70 ns |
| 0.95x | Zen5 | BLAKE3 | `blake3/xof vs blake3` | `1` | 101.51 ns | blake3 | 96.28 ns |
| 0.95x | RISC-V | Ascon | `ascon-hash256 vs ascon-hash` | `0` | 948.24 ns | ascon-hash | 899.46 ns |
| 0.95x | Zen4 | BLAKE3 | `blake3/xof vs blake3` | `1` | 96.23 ns | blake3 | 91.36 ns |
| 0.95x | Grav4 | AEAD | `aegis-256/decrypt vs aegis-crate` | `16384` | 3.23 us | aegis-crate | 3.07 us |
| 0.95x | RISC-V | BLAKE2 | `blake2/blake2s256 vs rustcrypto` | `262144` | 2.82 ms | rustcrypto | 2.68 ms |
| 0.95x | RISC-V | SHA-2 | `sha512 vs sha2` | `4096` | 56.25 us | sha2 | 53.42 us |
| 0.95x | RISC-V | BLAKE2 | `blake2/blake2s128 vs rustcrypto` | `65536` | 674.65 us | rustcrypto | 640.89 us |

## Internal Benches

These compare two rscrypto-internal benchmark modes, not rscrypto against an external crate; they are excluded from the scoreboard above. Ratio is `alternate_time / primary_time`.

| Platform | Benchmark | Primary | Alternate | Ratio |
|---|---|---:|---:|---:|
| Zen4 | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 4.34 ms | 4.42 ms | 1.02x |
| Zen4 | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 570.56 ns | 591.68 ns | 1.04x |
| Zen5 | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 3.18 ms | 3.21 ms | 1.01x |
| Zen5 | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 412.37 ns | 427.79 ns | 1.04x |
| Grav3 | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 4.81 ms | 5.20 ms | 1.08x |
| Grav3 | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 624.73 ns | 687.65 ns | 1.10x |
| Grav4 | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 4.40 ms | 4.42 ms | 1.00x |
| Grav4 | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 571.97 ns | 583.96 ns | 1.02x |
| POWER10 | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 4.36 ms | 4.39 ms | 1.01x |
| POWER10 | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 589.66 ns | 601.61 ns | 1.02x |
| s390x | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 7.74 ms | 544.85 us | 0.07x |
| s390x | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 1.03 us | 1.09 us | 1.05x |
| ICL | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 5.09 ms | 5.14 ms | 1.01x |
| ICL | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 674.54 ns | 691.46 ns | 1.03x |
| SPR | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 4.52 ms | 4.49 ms | 0.99x |
| SPR | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 591.17 ns | 601.76 ns | 1.02x |
| RISC-V | `sha3-256/digest_pair/pair/1048576 vs 2x_sequential` | 23.15 ms | 24.37 ms | 1.05x |
| RISC-V | `sha3-256/digest_pair/pair/64 vs 2x_sequential` | 2.99 us | 3.12 us | 1.04x |

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
