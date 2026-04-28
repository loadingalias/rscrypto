# Benchmark Overview

Source: Bench CI run [#25057894407](https://github.com/loadingalias/rscrypto/actions/runs/25057894407) created 2026-04-28 14:10:38 UTC.
Commit: `90043b7aeb9d55819ce54f13b22a90c8f0cc10aa`

Scope: nine Linux CI runners from the benchmark run above. The local macOS Apple Silicon result at `benchmark_results/2026-04-28/macos/aarch64/results.txt` is intentionally excluded because it is from commit `f1a6e967a4653abdf66b55d712ec6f61e2e4c6b7`, not this CI commit. Speedup is `competitor_time / rscrypto_time`; values above `1.00x` mean `rscrypto` is faster. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`.

## Executive Read

Overall Linux CI: **3627W / 1838T / 331L = 63% win rate** across 5796 paired external comparisons. Geomean speedup is **1.75x** and median speedup is **1.14x**.
Delta vs previous Linux CI run [#25027369331](https://github.com/loadingalias/rscrypto/actions/runs/25027369331) at `f1a6e967`: **-9W / -26T / +35L**, geomean +0.00x, median +0.00x.
Strongest category by geomean: **Checksums** at 4.38x. Weakest category by geomean: **RapidHash** at 1.06x.
Checksums: 786W/142T/62L, geomean 4.38x, worst 0.54x on RISC-V `crc32c vs crc32c` `32`.
SHA-2: 264W/261T/6L, geomean 1.46x, worst 0.01x on SPR `sha256/streaming vs sha2` `64B`.
SHA-3: 385W/29T/0L, geomean 2.18x, worst 0.97x on POWER10 `sha3-512 vs sha3` `65536`.
SHAKE: 189W/9T/0L, geomean 2.62x, worst 1.00x on POWER10 `shake256 vs sha3` `262144`.
BLAKE2: 373W/449T/24L, geomean 1.07x, worst 0.72x on s390x `blake2/blake2b256 vs rustcrypto` `0`.
BLAKE3: 221W/174T/37L, geomean 1.41x, worst 0.72x on s390x `blake3/xof vs blake3` `1024`.
Ascon: 127W/70T/1L, geomean 1.31x, worst 0.93x on s390x `ascon-hash256 vs ascon-hash` `256`.
XXH3: 105W/70T/23L, geomean 1.18x, worst 0.52x on ICL `xxh3-64 vs xxhash-rust` `64`.
RapidHash: 111W/216T/69L, geomean 1.06x, worst 0.67x on Grav4 `rapidhash-64 vs rapidhash` `0`.
Auth/KDF: 324W/240T/39L, geomean 1.27x, worst 0.01x on SPR `hmac-sha256/streaming vs rustcrypto` `64B`.
AEAD: 742W/178T/70L, geomean 1.84x, worst 0.45x on SPR `aes-256-gcm/encrypt vs rustcrypto` `0`.

## Delta vs Previous Linux CI

Baseline: Bench CI run [#25027369331](https://github.com/loadingalias/rscrypto/actions/runs/25027369331) created 2026-04-28 00:39:00 UTC at `f1a6e967a4653abdf66b55d712ec6f61e2e4c6b7`. New run time is `14_10_38`; old run time was `00_39_00`.

| Scope | W | T | L | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Linux CI | -9 | -26 | +35 | +0.00x | +0.00x |
| Checksums | -3 | -7 | +10 | -0.01x | +0.03x |
| SHA-2 | -10 | +10 | +0 | +0.01x | -0.00x |
| SHA-3 | +1 | -1 | +0 | +0.01x | +0.01x |
| SHAKE | -1 | +1 | +0 | +0.02x | -0.01x |
| BLAKE2 | -6 | -8 | +14 | -0.01x | -0.00x |
| BLAKE3 | +0 | -3 | +3 | -0.00x | -0.01x |
| Ascon | +2 | -3 | +1 | -0.00x | +0.03x |
| XXH3 | -1 | -1 | +2 | -0.00x | +0.01x |
| RapidHash | +5 | -7 | +2 | -0.00x | -0.00x |
| Auth/KDF | -3 | +2 | +1 | +0.01x | +0.01x |
| AEAD | +7 | -9 | +2 | -0.00x | -0.01x |

## Flippable Ties

These are broad neutral surfaces where small local wins are likely to move many comparisons from tie to win.

| Rank | Opportunity | Tie Surface | Median | Worst Tie | Why It Is Flippable |
|---:|---|---:|---:|---:|---|
| 1 | `rapidhash-128 vs rapidhash` | 59 ties / 9 platforms | 1.00x | 0.96x | Losses/ties cluster by family, pointing at short-input overhead and threshold work rather than a redesign. |
| 2 | `rapidhash-64 vs rapidhash` | 58 ties / 9 platforms | 1.00x | 0.96x | Losses/ties cluster by family, pointing at short-input overhead and threshold work rather than a redesign. |
| 3 | `blake2/keyed/blake2b512 vs rustcrypto` | 58 ties / 8 platforms | 1.00x | 0.96x | Broad parity remains; small constructor/finalization wins can still flip many ties without a new backend. |

## Flippable Losses

These are loss clusters with enough surface area to matter but narrow enough to attack directly.

| Rank | Opportunity | Loss Surface | Median Loss | Worst Loss | Why It Is Flippable |
|---:|---|---:|---:|---:|---|
| 1 | `rapidhash-v3-128 vs rapidhash` | 24 losses / 6 platforms | 0.93x | 0.73x | Losses/ties cluster by family, pointing at short-input overhead and threshold work rather than a redesign. |
| 2 | `aes-256-gcm/encrypt vs rustcrypto` | 23 losses / 5 platforms | 0.68x | 0.45x | Tiny-message AES-GCM losses point to fixed overhead; a narrow one-block path is the likely lever. |
| 3 | `rapidhash-v3-64 vs rapidhash` | 21 losses / 6 platforms | 0.93x | 0.69x | Losses/ties cluster by family, pointing at short-input overhead and threshold work rather than a redesign. |

## Platforms

| Short | Platform | Architecture | Source | Header Time | Pairs | Results |
|---|---|---|---|---|---:|---|
| Zen4 | AMD EPYC Zen 4 | x86-64 | ci | `14_10_38` | 644 | `benchmark_results/2026-04-28/linux/amd-zen4/results.txt` |
| Zen5 | AMD EPYC Zen 5 | x86-64 | ci | `14_10_38` | 644 | `benchmark_results/2026-04-28/linux/amd-zen5/results.txt` |
| SPR | Intel Xeon Sapphire Rapids | x86-64 | ci | `14_10_38` | 644 | `benchmark_results/2026-04-28/linux/intel-spr/results.txt` |
| ICL | Intel Xeon Ice Lake | x86-64 | ci | `14_10_38` | 644 | `benchmark_results/2026-04-28/linux/intel-icl/results.txt` |
| Grav3 | AWS Graviton3 | aarch64 | ci | `14_10_38` | 644 | `benchmark_results/2026-04-28/linux/graviton3/results.txt` |
| Grav4 | AWS Graviton4 | aarch64 | ci | `14_10_38` | 644 | `benchmark_results/2026-04-28/linux/graviton4/results.txt` |
| s390x | IBM Z | s390x | ci | `14_10_38` | 644 | `benchmark_results/2026-04-28/linux/ibm-s390x/results.txt` |
| POWER10 | IBM POWER10 | ppc64le | ci | `14_10_38` | 644 | `benchmark_results/2026-04-28/linux/ibm-power10/results.txt` |
| RISC-V | RISE RISC-V | riscv64 | ci | `14_10_38` | 644 | `benchmark_results/2026-04-28/linux/rise-riscv/results.txt` |

## Overall Scoreboard

| W | T | L | Total | Win % | Geomean | Median | Worst | Worst Platform | Worst Case | Best | Best Platform | Best Case |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| 3627 | 1838 | 331 | 5796 | 63% | 1.75x | 1.14x | 0.01x | SPR | `sha256/streaming vs sha2` `64B` | 213.76x | SPR | `crc16-ibm vs crc` `262144` |

## By Category

| Category | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Platform | Worst Case | Best | Best Platform | Best Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Checksums | 786 | 142 | 62 | 990 | 79% | 4.38x | 2.42x | 0.54x | RISC-V | `crc32c vs crc32c` `32` | 213.76x | SPR | `crc16-ibm vs crc` `262144` |
| SHA-2 | 264 | 261 | 6 | 531 | 50% | 1.46x | 1.05x | 0.01x | SPR | `sha256/streaming vs sha2` `64B` | 122.42x | SPR | `sha224 vs sha2` `262144` |
| SHA-3 | 385 | 29 | 0 | 414 | 93% | 2.18x | 2.14x | 0.97x | POWER10 | `sha3-512 vs sha3` `65536` | 26.41x | s390x | `sha3-512 vs sha3` `1048576` |
| SHAKE | 189 | 9 | 0 | 198 | 95% | 2.62x | 2.22x | 1.00x | POWER10 | `shake256 vs sha3` `262144` | 17.56x | s390x | `shake256 vs sha3` `262144` |
| BLAKE2 | 373 | 449 | 24 | 846 | 44% | 1.07x | 1.04x | 0.72x | s390x | `blake2/blake2b256 vs rustcrypto` `0` | 1.64x | SPR | `blake2/blake2s128 vs rustcrypto` `64` |
| BLAKE3 | 221 | 174 | 37 | 432 | 51% | 1.41x | 1.05x | 0.72x | s390x | `blake3/xof vs blake3` `1024` | 6.46x | POWER10 | `blake3/xof vs blake3` `262144` |
| Ascon | 127 | 70 | 1 | 198 | 64% | 1.31x | 1.22x | 0.93x | s390x | `ascon-hash256 vs ascon-hash` `256` | 2.51x | Zen5 | `ascon-xof128 vs ascon-hash` `0` |
| XXH3 | 105 | 70 | 23 | 198 | 53% | 1.18x | 1.07x | 0.52x | ICL | `xxh3-64 vs xxhash-rust` `64` | 2.84x | s390x | `xxh3-64 vs xxhash-rust` `16384` |
| RapidHash | 111 | 216 | 69 | 396 | 28% | 1.06x | 1.00x | 0.67x | Grav4 | `rapidhash-64 vs rapidhash` `0` | 2.27x | Grav3 | `rapidhash-v3-128 vs rapidhash` `32` |
| Auth/KDF | 324 | 240 | 39 | 603 | 54% | 1.27x | 1.07x | 0.01x | SPR | `hmac-sha256/streaming vs rustcrypto` `64B` | 86.94x | SPR | `hkdf-sha256/expand vs rustcrypto` `32` |
| AEAD | 742 | 178 | 70 | 990 | 75% | 1.84x | 1.42x | 0.45x | SPR | `aes-256-gcm/encrypt vs rustcrypto` `0` | 34.46x | s390x | `aes-256-gcm-siv/decrypt vs rustcrypto` `0` |

## By Platform

| Platform | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Zen4 | 473 | 157 | 14 | 644 | 73% | 1.62x | 1.20x | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| Zen5 | 381 | 225 | 38 | 644 | 59% | 1.64x | 1.12x | 0.56x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| SPR | 499 | 99 | 46 | 644 | 77% | 1.91x | 1.27x | 0.01x | `sha256/streaming vs sha2` `64B` |
| ICL | 467 | 154 | 23 | 644 | 73% | 1.62x | 1.22x | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| Grav3 | 328 | 273 | 43 | 644 | 51% | 1.64x | 1.06x | 0.71x | `rapidhash-64 vs rapidhash` `0` |
| Grav4 | 329 | 290 | 25 | 644 | 51% | 1.65x | 1.05x | 0.67x | `rapidhash-64 vs rapidhash` `0` |
| s390x | 532 | 66 | 46 | 644 | 83% | 3.04x | 2.38x | 0.62x | `crc24-openpgp vs crc` `64` |
| POWER10 | 334 | 293 | 17 | 644 | 52% | 1.94x | 1.06x | 0.73x | `rapidhash-v3-128 vs rapidhash` `0` |
| RISC-V | 284 | 281 | 79 | 644 | 44% | 1.15x | 1.03x | 0.54x | `crc32c vs crc32c` `32` |

## By Platform And Category

| Platform | Category | W | T | L | Total | Win % | Geomean | Worst | Worst Case |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zen4 | Checksums | 90 | 20 | 0 | 110 | 82% | 4.44x | 0.96x | `crc32 vs crc-fast` `1024` |
| Zen4 | SHA-2 | 33 | 26 | 0 | 59 | 56% | 1.12x | 1.00x | `sha224 vs sha2` `65536` |
| Zen4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.30x | 2.24x | `sha3-256/streaming vs sha3` `4096B` |
| Zen4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.06x | 2.25x | `shake128 vs sha3` `1048576` |
| Zen4 | BLAKE2 | 60 | 34 | 0 | 94 | 64% | 1.08x | 1.01x | `blake2/blake2b512 vs rustcrypto` `0` |
| Zen4 | BLAKE3 | 25 | 23 | 0 | 48 | 52% | 1.36x | 0.96x | `blake3/xof vs blake3` `0` |
| Zen4 | Ascon | 22 | 0 | 0 | 22 | 100% | 1.93x | 1.67x | `ascon-hash256 vs ascon-hash` `1` |
| Zen4 | XXH3 | 12 | 5 | 5 | 22 | 55% | 1.03x | 0.70x | `xxh3-64 vs xxhash-rust` `256` |
| Zen4 | RapidHash | 13 | 31 | 0 | 44 | 30% | 1.07x | 0.95x | `rapidhash-v3-128 vs rapidhash` `65536` |
| Zen4 | Auth/KDF | 56 | 7 | 4 | 67 | 84% | 1.20x | 0.92x | `pbkdf2-sha256/iters=1000 vs rustcrypto` `32` |
| Zen4 | AEAD | 94 | 11 | 5 | 110 | 85% | 1.27x | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| Zen5 | Checksums | 91 | 18 | 1 | 110 | 83% | 4.85x | 0.94x | `crc32c vs crc-fast` `1024` |
| Zen5 | SHA-2 | 26 | 33 | 0 | 59 | 44% | 1.09x | 1.00x | `sha224 vs sha2` `262144` |
| Zen5 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.18x | 2.11x | `sha3-256/streaming vs sha3` `64B` |
| Zen5 | SHAKE | 22 | 0 | 0 | 22 | 100% | 2.90x | 2.15x | `shake128 vs sha3` `1048576` |
| Zen5 | BLAKE2 | 7 | 79 | 8 | 94 | 7% | 1.00x | 0.87x | `blake2/blake2s256 vs rustcrypto` `1` |
| Zen5 | BLAKE3 | 21 | 20 | 7 | 48 | 44% | 1.34x | 0.83x | `blake3/streaming vs blake3` `64B` |
| Zen5 | Ascon | 22 | 0 | 0 | 22 | 100% | 2.17x | 1.85x | `ascon-hash256 vs ascon-hash` `1` |
| Zen5 | XXH3 | 16 | 5 | 1 | 22 | 73% | 1.30x | 0.91x | `xxh3-64 vs xxhash-rust` `64` |
| Zen5 | RapidHash | 14 | 21 | 9 | 44 | 32% | 1.05x | 0.81x | `rapidhash-v3-64 vs rapidhash` `0` |
| Zen5 | Auth/KDF | 27 | 33 | 7 | 67 | 40% | 1.10x | 0.92x | `ed25519/verify vs dalek` `1024` |
| Zen5 | AEAD | 89 | 16 | 5 | 110 | 81% | 1.39x | 0.56x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| SPR | Checksums | 97 | 10 | 3 | 110 | 88% | 4.87x | 0.88x | `crc32 vs crc32fast` `256` |
| SPR | SHA-2 | 51 | 6 | 2 | 59 | 86% | 3.29x | 0.01x | `sha256/streaming vs sha2` `64B` |
| SPR | SHA-3 | 46 | 0 | 0 | 46 | 100% | 3.60x | 3.44x | `sha3-512 vs sha3` `1` |
| SPR | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.80x | 3.50x | `shake128 vs sha3` `1048576` |
| SPR | BLAKE2 | 89 | 4 | 1 | 94 | 95% | 1.19x | 0.87x | `blake2/blake2b512 vs rustcrypto` `1` |
| SPR | BLAKE3 | 20 | 20 | 8 | 48 | 42% | 1.22x | 0.88x | `blake3 vs blake3` `16384` |
| SPR | Ascon | 22 | 0 | 0 | 22 | 100% | 1.45x | 1.29x | `ascon-hash256 vs ascon-hash` `1` |
| SPR | XXH3 | 12 | 6 | 4 | 22 | 55% | 1.10x | 0.55x | `xxh3-64 vs xxhash-rust` `64` |
| SPR | RapidHash | 13 | 19 | 12 | 44 | 30% | 1.06x | 0.71x | `rapidhash-64 vs rapidhash` `0` |
| SPR | Auth/KDF | 45 | 14 | 8 | 67 | 67% | 1.21x | 0.01x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| SPR | AEAD | 82 | 20 | 8 | 110 | 75% | 1.29x | 0.45x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| ICL | Checksums | 89 | 19 | 2 | 110 | 81% | 4.01x | 0.76x | `crc32 vs crc32fast` `256` |
| ICL | SHA-2 | 35 | 22 | 2 | 59 | 59% | 1.12x | 0.83x | `sha256/streaming vs sha2` `4096B` |
| ICL | SHA-3 | 46 | 0 | 0 | 46 | 100% | 3.00x | 2.86x | `sha3-256/streaming vs sha3` `64B` |
| ICL | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.03x | 2.98x | `shake128 vs sha3` `65536` |
| ICL | BLAKE2 | 94 | 0 | 0 | 94 | 100% | 1.20x | 1.10x | `blake2/blake2s256 vs rustcrypto` `1` |
| ICL | BLAKE3 | 15 | 28 | 5 | 48 | 31% | 1.15x | 0.82x | `blake3/keyed vs blake3` `256` |
| ICL | Ascon | 22 | 0 | 0 | 22 | 100% | 1.40x | 1.24x | `ascon-hash256 vs ascon-hash` `1` |
| ICL | XXH3 | 13 | 6 | 3 | 22 | 59% | 1.25x | 0.52x | `xxh3-64 vs xxhash-rust` `64` |
| ICL | RapidHash | 12 | 30 | 2 | 44 | 27% | 1.07x | 0.75x | `rapidhash-64 vs rapidhash` `32` |
| ICL | Auth/KDF | 36 | 27 | 4 | 67 | 54% | 1.11x | 0.86x | `hmac-sha256/streaming vs rustcrypto` `4096B` |
| ICL | AEAD | 83 | 22 | 5 | 110 | 75% | 1.25x | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| Grav3 | Checksums | 77 | 27 | 6 | 110 | 70% | 3.69x | 0.92x | `crc32c vs crc-fast` `262144` |
| Grav3 | SHA-2 | 18 | 40 | 1 | 59 | 31% | 1.06x | 0.91x | `sha256/streaming vs sha2` `64B` |
| Grav3 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.81x | 1.73x | `sha3-256/streaming vs sha3` `64B` |
| Grav3 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.81x | 1.77x | `shake256 vs sha3` `262144` |
| Grav3 | BLAKE2 | 27 | 67 | 0 | 94 | 29% | 1.04x | 1.00x | `blake2/streaming/blake2b256 vs rustcrypto` `64B` |
| Grav3 | BLAKE3 | 21 | 22 | 5 | 48 | 44% | 1.40x | 0.90x | `blake3/keyed vs blake3` `1` |
| Grav3 | Ascon | 4 | 18 | 0 | 22 | 18% | 1.03x | 0.99x | `ascon-xof128 vs ascon-hash` `65536` |
| Grav3 | XXH3 | 6 | 12 | 4 | 22 | 27% | 1.07x | 0.86x | `xxh3-128 vs xxhash-rust` `256` |
| Grav3 | RapidHash | 12 | 18 | 14 | 44 | 27% | 1.06x | 0.71x | `rapidhash-64 vs rapidhash` `0` |
| Grav3 | Auth/KDF | 30 | 30 | 7 | 67 | 45% | 1.08x | 0.83x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| Grav3 | AEAD | 65 | 39 | 6 | 110 | 59% | 2.48x | 0.77x | `aegis-256/encrypt vs aegis-crate` `0` |
| Grav4 | Checksums | 79 | 28 | 3 | 110 | 72% | 3.69x | 0.82x | `crc32c vs crc-fast` `256` |
| Grav4 | SHA-2 | 20 | 38 | 1 | 59 | 34% | 1.05x | 0.90x | `sha256/streaming vs sha2` `64B` |
| Grav4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.97x | 1.88x | `sha3-256/streaming vs sha3` `64B` |
| Grav4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.98x | 1.94x | `shake256 vs sha3` `65536` |
| Grav4 | BLAKE2 | 23 | 71 | 0 | 94 | 24% | 1.03x | 0.98x | `blake2/streaming/blake2b256 vs rustcrypto` `64B` |
| Grav4 | BLAKE3 | 23 | 23 | 2 | 48 | 48% | 1.47x | 0.94x | `blake3/keyed vs blake3` `1` |
| Grav4 | Ascon | 12 | 10 | 0 | 22 | 55% | 1.08x | 1.02x | `ascon-hash256 vs ascon-hash` `32` |
| Grav4 | XXH3 | 5 | 13 | 4 | 22 | 23% | 1.03x | 0.78x | `xxh3-128 vs xxhash-rust` `256` |
| Grav4 | RapidHash | 14 | 27 | 3 | 44 | 32% | 1.07x | 0.67x | `rapidhash-64 vs rapidhash` `0` |
| Grav4 | Auth/KDF | 19 | 46 | 2 | 67 | 28% | 1.07x | 0.87x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| Grav4 | AEAD | 66 | 34 | 10 | 110 | 60% | 2.46x | 0.78x | `aegis-256/encrypt vs aegis-crate` `0` |
| s390x | Checksums | 98 | 2 | 10 | 110 | 89% | 7.02x | 0.62x | `crc24-openpgp vs crc` `64` |
| s390x | SHA-2 | 59 | 0 | 0 | 59 | 100% | 5.63x | 1.92x | `sha224 vs sha2` `64` |
| s390x | SHA-3 | 46 | 0 | 0 | 46 | 100% | 4.73x | 1.12x | `sha3-256/streaming vs sha3` `64B` |
| s390x | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.08x | 1.09x | `shake256 vs sha3` `64` |
| s390x | BLAKE2 | 62 | 26 | 6 | 94 | 66% | 1.08x | 0.72x | `blake2/blake2b256 vs rustcrypto` `0` |
| s390x | BLAKE3 | 40 | 3 | 5 | 48 | 83% | 1.81x | 0.72x | `blake3/xof vs blake3` `1024` |
| s390x | Ascon | 6 | 15 | 1 | 22 | 27% | 1.06x | 0.93x | `ascon-hash256 vs ascon-hash` `256` |
| s390x | XXH3 | 16 | 4 | 2 | 22 | 73% | 1.69x | 0.73x | `xxh3-64 vs xxhash-rust` `32` |
| s390x | RapidHash | 10 | 14 | 20 | 44 | 23% | 0.99x | 0.69x | `rapidhash-v3-64 vs rapidhash` `1` |
| s390x | Auth/KDF | 63 | 2 | 2 | 67 | 94% | 3.69x | 0.71x | `ed25519/verify vs dalek` `1024` |
| s390x | AEAD | 110 | 0 | 0 | 110 | 100% | 4.36x | 1.16x | `xchacha20-poly1305/encrypt vs rustcrypto` `32` |
| POWER10 | Checksums | 104 | 3 | 3 | 110 | 95% | 10.77x | 0.78x | `crc64-nvme vs crc64fast-nvme` `64` |
| POWER10 | SHA-2 | 9 | 50 | 0 | 59 | 15% | 1.01x | 0.96x | `sha384 vs sha2` `1024` |
| POWER10 | SHA-3 | 17 | 29 | 0 | 46 | 37% | 1.04x | 0.97x | `sha3-512 vs sha3` `65536` |
| POWER10 | SHAKE | 13 | 9 | 0 | 22 | 59% | 1.39x | 1.00x | `shake256 vs sha3` `262144` |
| POWER10 | BLAKE2 | 9 | 84 | 1 | 94 | 10% | 1.01x | 0.93x | `blake2/blake2b256 vs rustcrypto` `256` |
| POWER10 | BLAKE3 | 35 | 12 | 1 | 48 | 73% | 2.02x | 0.95x | `blake3 vs blake3` `1` |
| POWER10 | Ascon | 5 | 17 | 0 | 22 | 23% | 1.05x | 1.00x | `ascon-hash256 vs ascon-hash` `65536` |
| POWER10 | XXH3 | 19 | 3 | 0 | 22 | 86% | 1.30x | 0.97x | `xxh3-64 vs xxhash-rust` `1` |
| POWER10 | RapidHash | 11 | 27 | 6 | 44 | 25% | 1.08x | 0.73x | `rapidhash-v3-128 vs rapidhash` `0` |
| POWER10 | Auth/KDF | 16 | 51 | 0 | 67 | 24% | 1.06x | 0.99x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| POWER10 | AEAD | 96 | 8 | 6 | 110 | 87% | 2.65x | 0.93x | `aegis-256/encrypt vs aegis-crate` `262144` |
| RISC-V | Checksums | 61 | 15 | 34 | 110 | 55% | 1.37x | 0.54x | `crc32c vs crc32c` `32` |
| RISC-V | SHA-2 | 13 | 46 | 0 | 59 | 22% | 1.02x | 0.97x | `sha256 vs sha2` `16384` |
| RISC-V | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.18x | 1.06x | `sha3-256/streaming vs sha3` `4096B` |
| RISC-V | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.65x | 1.11x | `shake256 vs sha3` `16384` |
| RISC-V | BLAKE2 | 2 | 84 | 8 | 94 | 2% | 0.99x | 0.87x | `blake2/blake2s128 vs rustcrypto` `0` |
| RISC-V | BLAKE3 | 21 | 23 | 4 | 48 | 44% | 1.17x | 0.80x | `blake3/xof vs blake3` `262144` |
| RISC-V | Ascon | 12 | 10 | 0 | 22 | 55% | 1.07x | 0.95x | `ascon-hash256 vs ascon-hash` `0` |
| RISC-V | XXH3 | 6 | 16 | 0 | 22 | 27% | 1.03x | 0.95x | `xxh3-128 vs xxhash-rust` `1024` |
| RISC-V | RapidHash | 12 | 29 | 3 | 44 | 27% | 1.08x | 0.84x | `rapidhash-64 vs rapidhash` `64` |
| RISC-V | Auth/KDF | 32 | 30 | 5 | 67 | 48% | 1.09x | 0.91x | `pbkdf2-sha256/iters=100 vs rustcrypto` `32` |
| RISC-V | AEAD | 57 | 28 | 25 | 110 | 52% | 1.19x | 0.74x | `aes-256-gcm/decrypt vs rustcrypto` `1048576` |

## Worst Losses

| Rank | Platform | Category | Speedup | Case |
|---:|---|---|---:|---|
| 1 | SPR | SHA-2 | 0.01x | `sha256/streaming vs sha2` `64B` |
| 2 | SPR | Auth/KDF | 0.01x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| 3 | SPR | Auth/KDF | 0.01x | `hmac-sha256/streaming vs rustcrypto` `4096B` |
| 4 | SPR | SHA-2 | 0.01x | `sha256/streaming vs sha2` `4096B` |
| 5 | SPR | AEAD | 0.45x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 6 | ICL | AEAD | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 7 | Zen4 | AEAD | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 8 | ICL | AEAD | 0.50x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 9 | SPR | AEAD | 0.50x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 10 | ICL | XXH3 | 0.52x | `xxh3-64 vs xxhash-rust` `64` |
| 11 | Zen4 | AEAD | 0.54x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 12 | RISC-V | Checksums | 0.54x | `crc32c vs crc32c` `32` |
| 13 | SPR | XXH3 | 0.55x | `xxh3-64 vs xxhash-rust` `64` |
| 14 | RISC-V | Checksums | 0.56x | `crc64-nvme vs crc64fast-nvme` `16384` |
| 15 | Zen5 | AEAD | 0.56x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 16 | RISC-V | Checksums | 0.59x | `crc64-nvme vs crc-fast` `262144` |
| 17 | RISC-V | Checksums | 0.59x | `crc64-nvme vs crc64fast-nvme` `256` |
| 18 | RISC-V | Checksums | 0.61x | `crc64-nvme vs crc-fast` `16384` |
| 19 | RISC-V | Checksums | 0.61x | `crc32 vs crc32fast` `64` |
| 20 | ICL | AEAD | 0.62x | `aes-256-gcm/encrypt vs rustcrypto` `1` |

## Strongest Wins

| Rank | Platform | Category | Speedup | Case |
|---:|---|---|---:|---|
| 1 | SPR | Checksums | 213.76x | `crc16-ibm vs crc` `262144` |
| 2 | SPR | Checksums | 208.71x | `crc16-ccitt vs crc` `262144` |
| 3 | SPR | Checksums | 200.47x | `crc16-ccitt vs crc` `16384` |
| 4 | SPR | Checksums | 196.17x | `crc16-ccitt vs crc` `1048576` |
| 5 | SPR | Checksums | 191.39x | `crc16-ibm vs crc` `16384` |
| 6 | SPR | Checksums | 184.66x | `crc16-ibm vs crc` `4096` |
| 7 | SPR | Checksums | 184.51x | `crc16-ibm vs crc` `65536` |
| 8 | SPR | Checksums | 179.26x | `crc16-ibm vs crc` `1048576` |
| 9 | POWER10 | Checksums | 176.98x | `crc16-ibm vs crc` `1048576` |
| 10 | POWER10 | Checksums | 175.66x | `crc16-ibm vs crc` `262144` |
| 11 | POWER10 | Checksums | 175.29x | `crc16-ccitt vs crc` `1048576` |
| 12 | POWER10 | Checksums | 175.12x | `crc16-ccitt vs crc` `262144` |
| 13 | SPR | Checksums | 172.21x | `crc16-ccitt vs crc` `65536` |
| 14 | POWER10 | Checksums | 167.35x | `crc16-ibm vs crc` `65536` |
| 15 | POWER10 | Checksums | 166.64x | `crc16-ccitt vs crc` `65536` |
| 16 | SPR | Checksums | 165.23x | `crc16-ccitt vs crc` `4096` |
| 17 | POWER10 | Checksums | 156.91x | `crc16-ibm vs crc` `16384` |
| 18 | POWER10 | Checksums | 155.56x | `crc16-ccitt vs crc` `16384` |
| 19 | Zen5 | Checksums | 129.64x | `crc16-ibm vs crc` `262144` |
| 20 | Zen4 | Checksums | 129.23x | `crc16-ibm vs crc` `262144` |

## Notes

- CI artifact headers were normalized to the GitHub run creation time (`14_10_38`) so all Linux files share one timestamp, as required by the benchmark extraction convention.
- The previous same-date Linux CI files were replaced because `benchmark_results/` is date-based; time lives in the file header, not the path.
- Local macOS Apple Silicon is excluded until it is rerun at the CI commit.
- Unpaired internal diagnostics, such as digest-pair and forced-kernel probes, are intentionally excluded from the external-comparison scoreboards.
- The pathological SPR streaming SHA-2/HMAC losses still deserve investigation, but they are treated as setup/streaming overhead until the benchmark path is isolated.

## Input Files

- `benchmark_results/2026-04-28/linux/amd-zen4/results.txt`
- `benchmark_results/2026-04-28/linux/amd-zen5/results.txt`
- `benchmark_results/2026-04-28/linux/intel-spr/results.txt`
- `benchmark_results/2026-04-28/linux/intel-icl/results.txt`
- `benchmark_results/2026-04-28/linux/graviton3/results.txt`
- `benchmark_results/2026-04-28/linux/graviton4/results.txt`
- `benchmark_results/2026-04-28/linux/ibm-s390x/results.txt`
- `benchmark_results/2026-04-28/linux/ibm-power10/results.txt`
- `benchmark_results/2026-04-28/linux/rise-riscv/results.txt`
