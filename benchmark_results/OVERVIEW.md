# Benchmark Overview

Source: Bench CI run [#25017467666](https://github.com/loadingalias/rscrypto/actions/runs/25017467666) created 2026-04-27 20:20:33 UTC.
Commit: `0da9bca9fa61ac0b4dae37541c17e7b4027a4f1d`

Scope: nine Linux CI runners from the benchmark run above. Speedup is `competitor_time / rscrypto_time`; values above `1.00x` mean `rscrypto` is faster. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`.

The existing macOS Apple Silicon result at `benchmark_results/2026-04-27/macos/aarch64/results.txt` is intentionally excluded because it is from commit `e65388d85f72ac813d0fd29a28e01c8f1a623f70`, not this CI commit. Re-run macOS benches before folding Apple Silicon back into the aggregate.

## Executive Read

Overall Linux CI: **3635W / 1848T / 313L = 63% win rate** across 5796 paired external comparisons. Geomean speedup is **1.75x** and median speedup is **1.14x**.
Delta vs previous Linux CI run #24974478214 at `e65388d`: **-7W / -17T / +24L**, geomean +0.00x, median -0.00x.
Strongest category by geomean: **Checksums** at 4.44x. Weakest category by geomean: **RapidHash** at 1.06x.
Checksums: 799W/136T/55L, geomean 4.44x, worst 0.59x on RISC-V `crc32c vs crc32c` `32`.
SHA-2: 269W/256T/6L, geomean 1.44x, worst 0.01x on SPR `sha256/streaming vs sha2` `64B`.
SHA-3: 384W/30T/0L, geomean 2.18x, worst 0.98x on POWER10 `sha3-512 vs sha3` `16384`.
SHAKE: 189W/9T/0L, geomean 2.60x, worst 1.01x on POWER10 `shake128 vs sha3` `1048576`.
BLAKE2: 363W/468T/15L, geomean 1.06x, worst 0.87x on RISC-V `blake2 vs rustcrypto` `blake2b256/1048576`.
BLAKE3: 229W/165T/38L, geomean 1.42x, worst 0.63x on ICL `blake3/keyed vs blake3` `32`.
Ascon: 126W/72T/0L, geomean 1.31x, worst 0.95x on RISC-V `ascon-hash256 vs ascon-hash` `1`.
XXH3: 108W/65T/25L, geomean 1.19x, worst 0.53x on ICL `xxh3-64 vs xxhash-rust` `64`.
RapidHash: 110W/216T/70L, geomean 1.06x, worst 0.50x on s390x `rapidhash-v3-64 vs rapidhash` `1`.
Auth/KDF: 326W/242T/35L, geomean 1.26x, worst 0.01x on SPR `hmac-sha256/streaming vs rustcrypto` `64B`.
AEAD: 732W/189T/69L, geomean 1.84x, worst 0.46x on ICL `aes-256-gcm/encrypt vs rustcrypto` `0`.

## Delta vs Previous Linux CI

Baseline: Bench CI run [#24974478214](https://github.com/loadingalias/rscrypto/actions/runs/24974478214) created 2026-04-27 03:01:46 UTC at `e65388d85f72ac813d0fd29a28e01c8f1a623f70`. New run time is `20_20_33`; old run time was `03_01_46`.

| Scope | W | T | L | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Overall Linux | -7 | -17 | +24 | +0.00x | -0.00x |
| Checksums | -1 | -4 | +5 | +0.02x | +0.08x |
| SHA-2 | +2 | -1 | -1 | +0.00x | +0.00x |
| SHA-3 | -1 | +2 | -1 | +0.00x | +0.01x |
| SHAKE | -2 | +2 | +0 | +0.00x | +0.01x |
| BLAKE2 | -24 | +24 | +0 | -0.01x | -0.00x |
| BLAKE3 | +4 | -13 | +9 | +0.00x | +0.01x |
| Ascon | +1 | -1 | +0 | +0.00x | -0.00x |
| XXH3 | +6 | -5 | -1 | +0.01x | +0.02x |
| RapidHash | +2 | -10 | +8 | +0.01x | -0.00x |
| Auth/KDF | +20 | -22 | +2 | +0.00x | +0.01x |
| AEAD | -14 | +11 | +3 | -0.00x | +0.00x |

## Flippable Ties

These are not the closest ties. They are broad tie surfaces where a realistic localized change can turn many neutral comparisons into wins.

| Rank | Opportunity | Tie Surface | Median | Worst Tie | Why It Is Flippable |
|---:|---|---:|---:|---:|---|
| 1 | `blake2/keyed vs rustcrypto` | 220 ties / 8 platforms | 1.01x | 0.95x | BLAKE2 keyed constructors and compression are almost exactly parity across eight CI platforms; a small keyed/finalization win flips a broad surface. |
| 2 | `blake2 vs rustcrypto` | 210 ties / 8 platforms | 1.00x | 0.96x | Unkeyed BLAKE2 is the same story: broad parity, tiny median gap, and enough volume that one hot-path cleanup pays off everywhere. |
| 3 | `rapidhash-128 vs rapidhash` | 62 ties / 9 platforms | 1.00x | 0.96x | RapidHash 128-bit has wide parity on all CI platforms; short/medium mixer scheduling has room without needing a new backend. |

## Flippable Losses

These are loss clusters with enough spread to matter, but narrow enough that they look fixable without a broad architecture rewrite.

| Rank | Opportunity | Loss Surface | Median Loss | Worst Loss | Why It Is Flippable |
|---:|---|---:|---:|---:|---|
| 1 | `aes-256-gcm/encrypt vs rustcrypto` | 24 losses / 5 platforms | 0.69x | 0.46x | Repeated tiny-message AES-GCM encrypt losses point to fixed overhead, not throughput limits; a one-block/tiny path can move these to ties or wins. |
| 2 | `xxh3-64 vs xxhash-rust` | 14 losses / 8 platforms | 0.83x | 0.53x | XXH3-64 losses cluster on small and medium inputs across many platforms; the target is narrow enough for threshold/vector-path work. |
| 3 | `rapidhash-64 vs rapidhash` | 17 losses / 7 platforms | 0.87x | 0.67x | RapidHash-64 loses mostly on short inputs while the broader family already ties; focused small-input specialization should flip it. |

## Platforms

| Short | Platform | Architecture | Source | Header Time | Pairs | Results |
|---|---|---|---|---|---:|---|
| Zen4 | AMD EPYC Zen 4 | x86-64 | ci | `20_20_33` | 644 | `benchmark_results/2026-04-27/linux/amd-zen4/results.txt` |
| Zen5 | AMD EPYC Zen 5 | x86-64 | ci | `20_20_33` | 644 | `benchmark_results/2026-04-27/linux/amd-zen5/results.txt` |
| SPR | Intel Xeon Sapphire Rapids | x86-64 | ci | `20_20_33` | 644 | `benchmark_results/2026-04-27/linux/intel-spr/results.txt` |
| ICL | Intel Xeon Ice Lake | x86-64 | ci | `20_20_33` | 644 | `benchmark_results/2026-04-27/linux/intel-icl/results.txt` |
| Grav3 | AWS Graviton3 | aarch64 | ci | `20_20_33` | 644 | `benchmark_results/2026-04-27/linux/graviton3/results.txt` |
| Grav4 | AWS Graviton4 | aarch64 | ci | `20_20_33` | 644 | `benchmark_results/2026-04-27/linux/graviton4/results.txt` |
| s390x | IBM Z | s390x | ci | `20_20_33` | 644 | `benchmark_results/2026-04-27/linux/ibm-s390x/results.txt` |
| POWER10 | IBM POWER10 | ppc64le | ci | `20_20_33` | 644 | `benchmark_results/2026-04-27/linux/ibm-power10/results.txt` |
| RISC-V | RISE RISC-V | riscv64 | ci | `20_20_33` | 644 | `benchmark_results/2026-04-27/linux/rise-riscv/results.txt` |

## Overall Scoreboard

| W | T | L | Total | Win % | Geomean | Median | Worst | Worst Platform | Worst Case | Best | Best Platform | Best Case |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| 3635 | 1848 | 313 | 5796 | 63% | 1.75x | 1.14x | 0.01x | SPR | `sha256/streaming vs sha2` `64B` | 234.72x | SPR | `crc16-ccitt vs crc` `262144` |

## By Category

| Category | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Platform | Worst Case | Best | Best Platform | Best Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Checksums | 799 | 136 | 55 | 990 | 81% | 4.44x | 2.48x | 0.59x | RISC-V | `crc32c vs crc32c` `32` | 234.72x | SPR | `crc16-ccitt vs crc` `262144` |
| SHA-2 | 269 | 256 | 6 | 531 | 51% | 1.44x | 1.05x | 0.01x | SPR | `sha256/streaming vs sha2` `64B` | 123.35x | SPR | `sha256 vs sha2` `65536` |
| SHA-3 | 384 | 30 | 0 | 414 | 93% | 2.18x | 2.15x | 0.98x | POWER10 | `sha3-512 vs sha3` `16384` | 25.13x | s390x | `sha3-512 vs sha3` `1048576` |
| SHAKE | 189 | 9 | 0 | 198 | 95% | 2.60x | 2.24x | 1.01x | POWER10 | `shake128 vs sha3` `1048576` | 16.42x | s390x | `shake256 vs sha3` `1048576` |
| BLAKE2 | 363 | 468 | 15 | 846 | 43% | 1.06x | 1.04x | 0.87x | RISC-V | `blake2 vs rustcrypto` `blake2b256/1048576` | 1.67x | SPR | `blake2 vs rustcrypto` `blake2s128/32` |
| BLAKE3 | 229 | 165 | 38 | 432 | 53% | 1.42x | 1.07x | 0.63x | ICL | `blake3/keyed vs blake3` `32` | 8.99x | POWER10 | `blake3 vs blake3` `262144` |
| Ascon | 126 | 72 | 0 | 198 | 64% | 1.31x | 1.18x | 0.95x | RISC-V | `ascon-hash256 vs ascon-hash` `1` | 2.51x | Zen5 | `ascon-xof128 vs ascon-hash` `0` |
| XXH3 | 108 | 65 | 25 | 198 | 55% | 1.19x | 1.08x | 0.53x | ICL | `xxh3-64 vs xxhash-rust` `64` | 2.48x | s390x | `xxh3-64 vs xxhash-rust` `65536` |
| RapidHash | 110 | 216 | 70 | 396 | 28% | 1.06x | 1.00x | 0.50x | s390x | `rapidhash-v3-64 vs rapidhash` `1` | 2.28x | Grav3 | `rapidhash-v3-128 vs rapidhash` `32` |
| Auth/KDF | 326 | 242 | 35 | 603 | 54% | 1.26x | 1.06x | 0.01x | SPR | `hmac-sha256/streaming vs rustcrypto` `64B` | 84.37x | SPR | `hkdf-sha256/expand vs rustcrypto` `32` |
| AEAD | 732 | 189 | 69 | 990 | 74% | 1.84x | 1.43x | 0.46x | ICL | `aes-256-gcm/encrypt vs rustcrypto` `0` | 32.18x | s390x | `aes-256-gcm-siv/decrypt vs rustcrypto` `0` |

## By Platform

| Platform | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Category | Worst Case | Best | Best Category | Best Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Zen4 | 464 | 166 | 14 | 644 | 72% | 1.62x | 1.17x | 0.49x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `32` | 129.42x | Checksums | `crc16-ibm vs crc` `262144` |
| Zen5 | 383 | 228 | 33 | 644 | 59% | 1.65x | 1.12x | 0.56x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `32` | 129.22x | Checksums | `crc16-ccitt vs crc` `262144` |
| SPR | 495 | 103 | 46 | 644 | 77% | 1.92x | 1.28x | 0.01x | SHA-2 | `sha256/streaming vs sha2` `64B` | 234.72x | Checksums | `crc16-ccitt vs crc` `262144` |
| ICL | 477 | 140 | 27 | 644 | 74% | 1.62x | 1.22x | 0.46x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `0` | 120.14x | Checksums | `crc16-ibm vs crc` `262144` |
| Grav3 | 328 | 275 | 41 | 644 | 51% | 1.64x | 1.06x | 0.72x | RapidHash | `rapidhash-64 vs rapidhash` `0` | 127.43x | Checksums | `crc16-ibm vs crc` `262144` |
| Grav4 | 333 | 288 | 23 | 644 | 52% | 1.65x | 1.06x | 0.67x | RapidHash | `rapidhash-64 vs rapidhash` `0` | 105.09x | Checksums | `crc16-ibm vs crc` `262144` |
| s390x | 522 | 85 | 37 | 644 | 81% | 3.01x | 2.37x | 0.50x | RapidHash | `rapidhash-v3-64 vs rapidhash` `1` | 100.97x | Checksums | `crc16-ibm vs crc` `65536` |
| POWER10 | 330 | 286 | 28 | 644 | 51% | 1.93x | 1.06x | 0.72x | RapidHash | `rapidhash-v3-128 vs rapidhash` `0` | 176.48x | Checksums | `crc16-ibm vs crc` `1048576` |
| RISC-V | 303 | 277 | 64 | 644 | 47% | 1.17x | 1.04x | 0.57x | XXH3 | `xxh3-64 vs xxhash-rust` `0` | 8.74x | Checksums | `crc64-nvme vs crc-fast` `0` |

## Platform x Category

| Platform | Category | W | T | L | Total | Win % | Geomean | Worst | Worst Case |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zen4 | Checksums | 90 | 20 | 0 | 110 | 82% | 4.50x | 0.95x | `crc32 vs crc-fast` `1024` |
| Zen4 | SHA-2 | 29 | 30 | 0 | 59 | 49% | 1.11x | 1.00x | `sha256/streaming vs sha2` `64B` |
| Zen4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.29x | 2.25x | `sha3-256/streaming vs sha3` `64B` |
| Zen4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.03x | 2.23x | `shake256 vs sha3` `1048576` |
| Zen4 | BLAKE2 | 57 | 37 | 0 | 94 | 61% | 1.07x | 0.98x | `blake2 vs rustcrypto` `blake2b256/1048576` |
| Zen4 | BLAKE3 | 27 | 21 | 0 | 48 | 56% | 1.37x | 0.97x | `blake3/streaming vs blake3` `64B` |
| Zen4 | Ascon | 22 | 0 | 0 | 22 | 100% | 1.94x | 1.67x | `ascon-hash256 vs ascon-hash` `1` |
| Zen4 | XXH3 | 12 | 6 | 4 | 22 | 55% | 1.02x | 0.74x | `xxh3-64 vs xxhash-rust` `256` |
| Zen4 | RapidHash | 13 | 31 | 0 | 44 | 30% | 1.08x | 0.96x | `rapidhash-v3-64 vs rapidhash` `16384` |
| Zen4 | Auth/KDF | 56 | 7 | 4 | 67 | 84% | 1.20x | 0.92x | `pbkdf2-sha256/iters=1000 vs rustcrypto` `64` |
| Zen4 | AEAD | 90 | 14 | 6 | 110 | 82% | 1.26x | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| Zen5 | Checksums | 91 | 19 | 0 | 110 | 83% | 4.87x | 0.97x | `crc32 vs crc-fast` `1024` |
| Zen5 | SHA-2 | 28 | 31 | 0 | 59 | 47% | 1.10x | 1.00x | `sha256/streaming vs sha2` `64B` |
| Zen5 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.18x | 2.15x | `sha3-256/streaming vs sha3` `4096B` |
| Zen5 | SHAKE | 22 | 0 | 0 | 22 | 100% | 2.89x | 2.16x | `shake256 vs sha3` `1048576` |
| Zen5 | BLAKE2 | 9 | 85 | 0 | 94 | 10% | 1.01x | 0.97x | `blake2 vs rustcrypto` `blake2s128/1048576` |
| Zen5 | BLAKE3 | 21 | 20 | 7 | 48 | 44% | 1.29x | 0.79x | `blake3/keyed vs blake3` `0` |
| Zen5 | Ascon | 22 | 0 | 0 | 22 | 100% | 2.17x | 1.84x | `ascon-hash256 vs ascon-hash` `1` |
| Zen5 | XXH3 | 12 | 7 | 3 | 22 | 55% | 1.27x | 0.72x | `xxh3-128 vs xxhash-rust` `256` |
| Zen5 | RapidHash | 12 | 21 | 11 | 44 | 27% | 1.06x | 0.82x | `rapidhash-v3-64 vs rapidhash` `0` |
| Zen5 | Auth/KDF | 31 | 29 | 7 | 67 | 46% | 1.10x | 0.92x | `ed25519/verify vs dalek` `0` |
| Zen5 | AEAD | 89 | 16 | 5 | 110 | 81% | 1.42x | 0.56x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| SPR | Checksums | 96 | 8 | 6 | 110 | 87% | 4.88x | 0.87x | `crc32c vs crc-fast` `1024` |
| SPR | SHA-2 | 51 | 6 | 2 | 59 | 86% | 3.28x | 0.01x | `sha256/streaming vs sha2` `64B` |
| SPR | SHA-3 | 46 | 0 | 0 | 46 | 100% | 3.60x | 3.40x | `sha3-512 vs sha3` `1` |
| SPR | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.80x | 3.44x | `shake128 vs sha3` `1048576` |
| SPR | BLAKE2 | 88 | 6 | 0 | 94 | 94% | 1.19x | 0.99x | `blake2/keyed vs rustcrypto` `blake2s256/0` |
| SPR | BLAKE3 | 21 | 21 | 6 | 48 | 44% | 1.26x | 0.90x | `blake3/xof vs blake3` `32` |
| SPR | Ascon | 22 | 0 | 0 | 22 | 100% | 1.47x | 1.29x | `ascon-hash256 vs ascon-hash` `1` |
| SPR | XXH3 | 12 | 7 | 3 | 22 | 55% | 1.11x | 0.56x | `xxh3-64 vs xxhash-rust` `64` |
| SPR | RapidHash | 13 | 19 | 12 | 44 | 30% | 1.06x | 0.75x | `rapidhash-64 vs rapidhash` `0` |
| SPR | Auth/KDF | 43 | 16 | 8 | 67 | 64% | 1.21x | 0.01x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| SPR | AEAD | 81 | 20 | 9 | 110 | 74% | 1.29x | 0.48x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| ICL | Checksums | 88 | 20 | 2 | 110 | 80% | 4.09x | 0.63x | `crc32 vs crc32fast` `256` |
| ICL | SHA-2 | 36 | 21 | 2 | 59 | 61% | 1.12x | 0.82x | `sha256/streaming vs sha2` `4096B` |
| ICL | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.99x | 2.86x | `sha3-256/streaming vs sha3` `64B` |
| ICL | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.02x | 2.98x | `shake128 vs sha3` `1048576` |
| ICL | BLAKE2 | 94 | 0 | 0 | 94 | 100% | 1.20x | 1.08x | `blake2 vs rustcrypto` `blake2b256/4096` |
| ICL | BLAKE3 | 19 | 21 | 8 | 48 | 40% | 1.13x | 0.63x | `blake3/keyed vs blake3` `32` |
| ICL | Ascon | 22 | 0 | 0 | 22 | 100% | 1.40x | 1.25x | `ascon-hash256 vs ascon-hash` `1` |
| ICL | XXH3 | 15 | 3 | 4 | 22 | 68% | 1.29x | 0.53x | `xxh3-64 vs xxhash-rust` `64` |
| ICL | RapidHash | 12 | 30 | 2 | 44 | 27% | 1.07x | 0.75x | `rapidhash-64 vs rapidhash` `32` |
| ICL | Auth/KDF | 37 | 26 | 4 | 67 | 55% | 1.10x | 0.83x | `hmac-sha256/streaming vs rustcrypto` `4096B` |
| ICL | AEAD | 86 | 19 | 5 | 110 | 78% | 1.25x | 0.46x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| Grav3 | Checksums | 78 | 26 | 6 | 110 | 71% | 3.73x | 0.89x | `crc32c vs crc-fast` `16384` |
| Grav3 | SHA-2 | 19 | 39 | 1 | 59 | 32% | 1.06x | 0.91x | `sha256/streaming vs sha2` `64B` |
| Grav3 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.81x | 1.74x | `sha3-256/streaming vs sha3` `64B` |
| Grav3 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.82x | 1.77x | `shake256 vs sha3` `1048576` |
| Grav3 | BLAKE2 | 23 | 71 | 0 | 94 | 24% | 1.03x | 1.00x | `blake2/streaming vs rustcrypto` `blake2b256/64B` |
| Grav3 | BLAKE3 | 21 | 22 | 5 | 48 | 44% | 1.40x | 0.90x | `blake3/keyed vs blake3` `1` |
| Grav3 | Ascon | 4 | 18 | 0 | 22 | 18% | 1.03x | 0.99x | `ascon-xof128 vs ascon-hash` `1048576` |
| Grav3 | XXH3 | 8 | 10 | 4 | 22 | 36% | 1.08x | 0.79x | `xxh3-128 vs xxhash-rust` `256` |
| Grav3 | RapidHash | 13 | 17 | 14 | 44 | 30% | 1.06x | 0.72x | `rapidhash-64 vs rapidhash` `0` |
| Grav3 | Auth/KDF | 31 | 31 | 5 | 67 | 46% | 1.09x | 0.89x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| Grav3 | AEAD | 63 | 41 | 6 | 110 | 57% | 2.48x | 0.77x | `aegis-256/encrypt vs aegis-crate` `0` |
| Grav4 | Checksums | 82 | 26 | 2 | 110 | 75% | 3.67x | 0.82x | `crc32c vs crc-fast` `64` |
| Grav4 | SHA-2 | 21 | 37 | 1 | 59 | 36% | 1.06x | 0.90x | `sha256/streaming vs sha2` `64B` |
| Grav4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.97x | 1.88x | `sha3-256/streaming vs sha3` `64B` |
| Grav4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.98x | 1.93x | `shake256 vs sha3` `1048576` |
| Grav4 | BLAKE2 | 21 | 73 | 0 | 94 | 22% | 1.02x | 1.00x | `blake2/keyed vs rustcrypto` `blake2b512/1024` |
| Grav4 | BLAKE3 | 23 | 24 | 1 | 48 | 48% | 1.48x | 0.95x | `blake3/keyed vs blake3` `1` |
| Grav4 | Ascon | 13 | 9 | 0 | 22 | 59% | 1.08x | 1.01x | `ascon-hash256 vs ascon-hash` `1` |
| Grav4 | XXH3 | 5 | 13 | 4 | 22 | 23% | 1.02x | 0.71x | `xxh3-128 vs xxhash-rust` `256` |
| Grav4 | RapidHash | 14 | 27 | 3 | 44 | 32% | 1.07x | 0.67x | `rapidhash-64 vs rapidhash` `0` |
| Grav4 | Auth/KDF | 20 | 45 | 2 | 67 | 30% | 1.07x | 0.87x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| Grav4 | AEAD | 66 | 34 | 10 | 110 | 60% | 2.46x | 0.77x | `aegis-256/encrypt vs aegis-crate` `0` |
| s390x | Checksums | 95 | 4 | 11 | 110 | 86% | 7.13x | 0.63x | `crc24-openpgp vs crc` `64` |
| s390x | SHA-2 | 59 | 0 | 0 | 59 | 100% | 5.20x | 1.89x | `sha224 vs sha2` `0` |
| s390x | SHA-3 | 46 | 0 | 0 | 46 | 100% | 4.71x | 1.12x | `sha3-256/streaming vs sha3` `64B` |
| s390x | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.95x | 1.10x | `shake256 vs sha3` `64` |
| s390x | BLAKE2 | 53 | 41 | 0 | 94 | 56% | 1.09x | 0.97x | `blake2/streaming vs rustcrypto` `blake2s256/64B` |
| s390x | BLAKE3 | 41 | 3 | 4 | 48 | 85% | 1.90x | 0.73x | `blake3/xof vs blake3` `1024` |
| s390x | Ascon | 4 | 18 | 0 | 22 | 18% | 1.05x | 1.00x | `ascon-xof128 vs ascon-hash` `65536` |
| s390x | XXH3 | 18 | 2 | 2 | 22 | 82% | 1.70x | 0.62x | `xxh3-64 vs xxhash-rust` `32` |
| s390x | RapidHash | 10 | 14 | 20 | 44 | 23% | 1.00x | 0.50x | `rapidhash-v3-64 vs rapidhash` `1` |
| s390x | Auth/KDF | 64 | 3 | 0 | 67 | 96% | 3.44x | 1.02x | `ed25519/verify vs dalek` `0` |
| s390x | AEAD | 110 | 0 | 0 | 110 | 100% | 4.27x | 1.16x | `xchacha20-poly1305/encrypt vs rustcrypto` `1` |
| POWER10 | Checksums | 104 | 3 | 3 | 110 | 95% | 10.77x | 0.78x | `crc64-nvme vs crc64fast-nvme` `64` |
| POWER10 | SHA-2 | 10 | 49 | 0 | 59 | 17% | 1.01x | 0.96x | `sha512-256 vs sha2` `256` |
| POWER10 | SHA-3 | 16 | 30 | 0 | 46 | 35% | 1.03x | 0.98x | `sha3-512 vs sha3` `16384` |
| POWER10 | SHAKE | 13 | 9 | 0 | 22 | 59% | 1.40x | 1.01x | `shake128 vs sha3` `1048576` |
| POWER10 | BLAKE2 | 8 | 75 | 11 | 94 | 9% | 0.99x | 0.91x | `blake2/keyed vs rustcrypto` `blake2b512/32` |
| POWER10 | BLAKE3 | 36 | 10 | 2 | 48 | 75% | 1.99x | 0.94x | `blake3/keyed vs blake3` `1` |
| POWER10 | Ascon | 5 | 17 | 0 | 22 | 23% | 1.05x | 1.01x | `ascon-hash256 vs ascon-hash` `262144` |
| POWER10 | XXH3 | 18 | 4 | 0 | 22 | 82% | 1.29x | 0.97x | `xxh3-64 vs xxhash-rust` `1` |
| POWER10 | RapidHash | 11 | 27 | 6 | 44 | 25% | 1.08x | 0.72x | `rapidhash-v3-128 vs rapidhash` `0` |
| POWER10 | Auth/KDF | 13 | 54 | 0 | 67 | 19% | 1.06x | 0.97x | `hmac-sha384 vs rustcrypto` `64` |
| POWER10 | AEAD | 96 | 8 | 6 | 110 | 87% | 2.63x | 0.93x | `aegis-256/encrypt vs aegis-crate` `65536` |
| RISC-V | Checksums | 75 | 10 | 25 | 110 | 68% | 1.45x | 0.59x | `crc32c vs crc32c` `32` |
| RISC-V | SHA-2 | 16 | 43 | 0 | 59 | 27% | 1.04x | 0.99x | `sha224 vs sha2` `262144` |
| RISC-V | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.18x | 1.07x | `sha3-256/streaming vs sha3` `4096B` |
| RISC-V | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.62x | 1.13x | `shake256 vs sha3` `16384` |
| RISC-V | BLAKE2 | 10 | 80 | 4 | 94 | 11% | 1.00x | 0.87x | `blake2 vs rustcrypto` `blake2b256/1048576` |
| RISC-V | BLAKE3 | 20 | 23 | 5 | 48 | 42% | 1.18x | 0.80x | `blake3/xof vs blake3` `262144` |
| RISC-V | Ascon | 12 | 10 | 0 | 22 | 55% | 1.07x | 0.95x | `ascon-hash256 vs ascon-hash` `1` |
| RISC-V | XXH3 | 8 | 13 | 1 | 22 | 36% | 1.05x | 0.57x | `xxh3-64 vs xxhash-rust` `0` |
| RISC-V | RapidHash | 12 | 30 | 2 | 44 | 27% | 1.08x | 0.84x | `rapidhash-64 vs rapidhash` `64` |
| RISC-V | Auth/KDF | 31 | 31 | 5 | 67 | 46% | 1.08x | 0.93x | `pbkdf2-sha256/iters=100 vs rustcrypto` `32` |
| RISC-V | AEAD | 51 | 37 | 22 | 110 | 46% | 1.19x | 0.81x | `aes-256-gcm/encrypt vs rustcrypto` `1048576` |

## Worst Losses

| Rank | Platform | Category | Speedup | Case |
|---:|---|---|---:|---|
| 1 | SPR | SHA-2 | 0.01x | `sha256/streaming vs sha2` `64B` |
| 2 | SPR | Auth/KDF | 0.01x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| 3 | SPR | SHA-2 | 0.01x | `sha256/streaming vs sha2` `4096B` |
| 4 | SPR | Auth/KDF | 0.01x | `hmac-sha256/streaming vs rustcrypto` `4096B` |
| 5 | ICL | AEAD | 0.46x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 6 | SPR | AEAD | 0.48x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 7 | ICL | AEAD | 0.48x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 8 | Zen4 | AEAD | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 9 | s390x | RapidHash | 0.50x | `rapidhash-v3-64 vs rapidhash` `1` |
| 10 | SPR | AEAD | 0.51x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 11 | ICL | XXH3 | 0.53x | `xxh3-64 vs xxhash-rust` `64` |
| 12 | Zen4 | AEAD | 0.55x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 13 | SPR | XXH3 | 0.56x | `xxh3-64 vs xxhash-rust` `64` |
| 14 | Zen5 | AEAD | 0.56x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 15 | RISC-V | XXH3 | 0.57x | `xxh3-64 vs xxhash-rust` `0` |
| 16 | ICL | AEAD | 0.59x | `aes-256-gcm/encrypt vs rustcrypto` `1` |
| 17 | RISC-V | Checksums | 0.59x | `crc32c vs crc32c` `32` |
| 18 | RISC-V | Checksums | 0.61x | `crc64-xz vs crc64fast` `4096` |
| 19 | RISC-V | Checksums | 0.62x | `crc32 vs crc32fast` `64` |
| 20 | s390x | XXH3 | 0.62x | `xxh3-64 vs xxhash-rust` `32` |

## Strongest Wins

| Rank | Platform | Category | Speedup | Case |
|---:|---|---|---:|---|
| 1 | SPR | Checksums | 234.72x | `crc16-ccitt vs crc` `262144` |
| 2 | SPR | Checksums | 227.49x | `crc16-ibm vs crc` `262144` |
| 3 | SPR | Checksums | 215.40x | `crc16-ibm vs crc` `16384` |
| 4 | SPR | Checksums | 212.42x | `crc16-ccitt vs crc` `16384` |
| 5 | SPR | Checksums | 205.00x | `crc16-ibm vs crc` `1048576` |
| 6 | SPR | Checksums | 202.84x | `crc16-ibm vs crc` `65536` |
| 7 | SPR | Checksums | 193.26x | `crc16-ccitt vs crc` `1048576` |
| 8 | SPR | Checksums | 187.35x | `crc16-ccitt vs crc` `4096` |
| 9 | SPR | Checksums | 184.16x | `crc16-ccitt vs crc` `65536` |
| 10 | SPR | Checksums | 181.34x | `crc16-ibm vs crc` `4096` |
| 11 | POWER10 | Checksums | 176.48x | `crc16-ibm vs crc` `1048576` |
| 12 | POWER10 | Checksums | 175.46x | `crc16-ibm vs crc` `262144` |
| 13 | POWER10 | Checksums | 174.32x | `crc16-ccitt vs crc` `1048576` |
| 14 | POWER10 | Checksums | 174.23x | `crc16-ccitt vs crc` `262144` |
| 15 | POWER10 | Checksums | 167.77x | `crc16-ibm vs crc` `65536` |
| 16 | POWER10 | Checksums | 164.22x | `crc16-ccitt vs crc` `65536` |
| 17 | POWER10 | Checksums | 156.78x | `crc16-ibm vs crc` `16384` |
| 18 | POWER10 | Checksums | 155.25x | `crc16-ccitt vs crc` `16384` |
| 19 | Zen4 | Checksums | 129.42x | `crc16-ibm vs crc` `262144` |
| 20 | Zen5 | Checksums | 129.22x | `crc16-ccitt vs crc` `262144` |

## Notes

- CI artifact headers were normalized to the GitHub run creation time (`20_20_33`) so all Linux files share one timestamp, as required by the benchmark extraction convention.
- The previous same-date Linux CI files were replaced because `benchmark_results/` is date-based; time lives in the file header, not the path.
- The stale macOS Apple Silicon result remains on disk but is excluded from this overview until rerun on the new benchmark commit.
- Unpaired internal diagnostics, such as forced-kernel CRC/XXH3 probes and digest-pair microbenchmarks, are intentionally excluded from the external-comparison scoreboards.
- The pathological SPR streaming SHA-2/HMAC losses still deserve investigation, but they are omitted from the flippable-loss top three because their 0.01x result smells like fixed benchmark/setup overhead rather than a normal kernel race.

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
