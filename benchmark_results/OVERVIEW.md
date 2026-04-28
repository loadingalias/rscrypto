# Benchmark Overview

Source: Bench CI run [#25027369331](https://github.com/loadingalias/rscrypto/actions/runs/25027369331) created 2026-04-28 00:39:00 UTC plus local macOS Apple Silicon results from `benchmark_results/2026-04-28/macos/aarch64/results.txt`.
Commit: `f1a6e967a4653abdf66b55d712ec6f61e2e4c6b7`

Scope: nine Linux CI runners plus one local macOS Apple Silicon run at the same commit. The macOS source file was a generic `bench=all` run, so this overview includes only the same 644-comparison surface used by Linux CI and excludes generic-only password-hashing, KMAC/cSHAKE, BLAKE2 host-overhead, and params rows. Speedup is `competitor_time / rscrypto_time`; values above `1.00x` mean `rscrypto` is faster. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`.

## Executive Read

Overall all platforms: **3940W / 2159T / 341L = 61% win rate** across 6440 paired external comparisons. Geomean speedup is **1.73x** and median speedup is **1.12x**.
Linux CI only: **3636W / 1864T / 296L**, geomean **1.75x**, median **1.14x**.
Linux-only delta vs previous CI run #25017467666 at `0da9bca9`: **+1W / +16T / -17L**, geomean -0.00x, median -0.00x.
Strongest category by geomean: **Checksums** at 4.38x. Weakest category by geomean: **BLAKE2** at 1.07x.
Checksums: 882W/163T/55L, geomean 4.38x, worst 0.56x on RISC-V `crc32c vs crc32c` `32`.
SHA-2: 285W/295T/10L, geomean 1.40x, worst 0.01x on SPR `sha256/streaming vs sha2` `64B`.
SHA-3: 390W/65T/5L, geomean 2.01x, worst 0.91x on macOS `sha3-384 vs sha3` `0`.
SHAKE: 197W/17T/6L, geomean 2.36x, worst 0.90x on macOS `shake128 vs sha3` `262144`.
BLAKE2: 392W/538T/10L, geomean 1.07x, worst 0.87x on s390x `blake2/keyed vs rustcrypto` `blake2b256/1024`.
BLAKE3: 238W/204T/38L, geomean 1.39x, worst 0.62x on macOS `blake3 vs blake3` `65536`.
Ascon: 142W/78T/0L, geomean 1.29x, worst 0.95x on RISC-V `ascon-hash256 vs ascon-hash` `0`.
XXH3: 112W/80T/28L, geomean 1.17x, worst 0.52x on ICL `xxh3-64 vs xxhash-rust` `64`.
RapidHash: 122W/246T/72L, geomean 1.06x, worst 0.16x on macOS `rapidhash-v3-64 vs rapidhash` `1`.
Auth/KDF: 341W/280T/49L, geomean 1.23x, worst 0.01x on SPR `hmac-sha256/streaming vs rustcrypto` `64B`.
AEAD: 839W/193T/68L, geomean 1.90x, worst 0.45x on ICL `aes-256-gcm/encrypt vs rustcrypto` `0`.

## Linux Delta vs Previous CI

Baseline: Bench CI run [#25017467666](https://github.com/loadingalias/rscrypto/actions/runs/25017467666) created 2026-04-27 20:20:33 UTC at `0da9bca9fa61ac0b4dae37541c17e7b4027a4f1d`. New CI run time is `00_39_00`; old run time was `20_20_33`. The local macOS row is excluded from this delta.

| Scope | W | T | L | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Linux CI | +1 | +16 | -17 | -0.00x | -0.00x |
| Checksums | -10 | +13 | -3 | -0.04x | -0.08x |
| SHA-2 | +5 | -5 | +0 | +0.00x | -0.00x |
| SHA-3 | +0 | +0 | +0 | -0.00x | -0.03x |
| SHAKE | +1 | -1 | +0 | +0.00x | -0.01x |
| BLAKE2 | +16 | -11 | -5 | +0.01x | +0.00x |
| BLAKE3 | -8 | +12 | -4 | -0.00x | -0.00x |
| Ascon | -1 | +1 | +0 | +0.00x | +0.00x |
| XXH3 | -2 | +6 | -4 | -0.00x | -0.02x |
| RapidHash | -4 | +7 | -3 | -0.00x | +0.00x |
| Auth/KDF | +1 | -4 | +3 | +0.00x | -0.00x |
| AEAD | +3 | -2 | -1 | +0.00x | -0.01x |

## Flippable Ties

These are broad neutral surfaces where small local wins are likely to move many comparisons from tie to win.

| Rank | Opportunity | Tie Surface | Median | Worst Tie | Why It Is Flippable |
|---:|---|---:|---:|---:|---|
| 1 | `blake2/keyed vs rustcrypto` | 251 ties / 9 platforms | 1.01x | 0.95x | Wide parity across measured platforms; a focused constructor, finalization, or threshold cleanup can flip many cases without a new backend. |
| 2 | `blake2 vs rustcrypto` | 243 ties / 9 platforms | 1.01x | 0.96x | Wide parity across measured platforms; a focused constructor, finalization, or threshold cleanup can flip many cases without a new backend. |
| 3 | `rapidhash-v3-64 vs rapidhash` | 64 ties / 10 platforms | 1.00x | 0.95x | Wide parity across measured platforms; a focused constructor, finalization, or threshold cleanup can flip many cases without a new backend. |

## Flippable Losses

These are loss clusters with enough surface area to matter but narrow enough to attack directly.

| Rank | Opportunity | Loss Surface | Median Loss | Worst Loss | Why It Is Flippable |
|---:|---|---:|---:|---:|---|
| 1 | `rapidhash-v3-64 vs rapidhash` | 23 losses / 8 platforms | 0.92x | 0.16x | The losses cluster by operation family, which points to overhead or dispatch thresholds rather than a full algorithm rewrite. |
| 2 | `aes-256-gcm/encrypt vs rustcrypto` | 23 losses / 5 platforms | 0.66x | 0.45x | The losses cluster by operation family, which points to overhead or dispatch thresholds rather than a full algorithm rewrite. |
| 3 | `rapidhash-v3-128 vs rapidhash` | 22 losses / 7 platforms | 0.92x | 0.73x | The losses cluster by operation family, which points to overhead or dispatch thresholds rather than a full algorithm rewrite. |

## Platforms

| Short | Platform | Architecture | Source | Header Time | Pairs | Results |
|---|---|---|---|---|---:|---|
| Zen4 | AMD EPYC Zen 4 | x86-64 | ci | `00_39_00` | 644 | `benchmark_results/2026-04-28/linux/amd-zen4/results.txt` |
| Zen5 | AMD EPYC Zen 5 | x86-64 | ci | `00_39_00` | 644 | `benchmark_results/2026-04-28/linux/amd-zen5/results.txt` |
| SPR | Intel Xeon Sapphire Rapids | x86-64 | ci | `00_39_00` | 644 | `benchmark_results/2026-04-28/linux/intel-spr/results.txt` |
| ICL | Intel Xeon Ice Lake | x86-64 | ci | `00_39_00` | 644 | `benchmark_results/2026-04-28/linux/intel-icl/results.txt` |
| Grav3 | AWS Graviton3 | aarch64 | ci | `00_39_00` | 644 | `benchmark_results/2026-04-28/linux/graviton3/results.txt` |
| Grav4 | AWS Graviton4 | aarch64 | ci | `00_39_00` | 644 | `benchmark_results/2026-04-28/linux/graviton4/results.txt` |
| s390x | IBM Z | s390x | ci | `00_39_00` | 644 | `benchmark_results/2026-04-28/linux/ibm-s390x/results.txt` |
| POWER10 | IBM POWER10 | ppc64le | ci | `00_39_00` | 644 | `benchmark_results/2026-04-28/linux/ibm-power10/results.txt` |
| RISC-V | RISE RISC-V | riscv64 | ci | `00_39_00` | 644 | `benchmark_results/2026-04-28/linux/rise-riscv/results.txt` |
| macOS | Apple Silicon local | aarch64 | local | `08_37_19` | 644 | `benchmark_results/2026-04-28/macos/aarch64/results.txt` |

## Overall Scoreboard

| W | T | L | Total | Win % | Geomean | Median | Worst | Worst Platform | Worst Case | Best | Best Platform | Best Case |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| 3940 | 2159 | 341 | 6440 | 61% | 1.73x | 1.12x | 0.01x | SPR | `hmac-sha256/streaming vs rustcrypto` `64B` | 225.37x | SPR | `crc16-ccitt vs crc` `16384` |

## By Category

| Category | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Platform | Worst Case | Best | Best Platform | Best Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Checksums | 882 | 163 | 55 | 1100 | 80% | 4.38x | 2.29x | 0.56x | RISC-V | `crc32c vs crc32c` `32` | 225.37x | SPR | `crc16-ccitt vs crc` `16384` |
| SHA-2 | 285 | 295 | 10 | 590 | 48% | 1.40x | 1.05x | 0.01x | SPR | `sha256/streaming vs sha2` `64B` | 121.09x | SPR | `sha256 vs sha2` `262144` |
| SHA-3 | 390 | 65 | 5 | 460 | 85% | 2.01x | 2.00x | 0.91x | macOS | `sha3-384 vs sha3` `0` | 24.88x | s390x | `sha3-512 vs sha3` `262144` |
| SHAKE | 197 | 17 | 6 | 220 | 90% | 2.36x | 2.11x | 0.90x | macOS | `shake128 vs sha3` `262144` | 16.77x | s390x | `shake256 vs sha3` `1048576` |
| BLAKE2 | 392 | 538 | 10 | 940 | 42% | 1.07x | 1.03x | 0.87x | s390x | `blake2/keyed vs rustcrypto` `blake2b256/1024` | 1.61x | SPR | `blake2 vs rustcrypto` `blake2s256/32` |
| BLAKE3 | 238 | 204 | 38 | 480 | 50% | 1.39x | 1.05x | 0.62x | macOS | `blake3 vs blake3` `65536` | 8.56x | s390x | `blake3/keyed vs blake3` `262144` |
| Ascon | 142 | 78 | 0 | 220 | 65% | 1.29x | 1.12x | 0.95x | RISC-V | `ascon-hash256 vs ascon-hash` `0` | 2.52x | Zen5 | `ascon-xof128 vs ascon-hash` `0` |
| XXH3 | 112 | 80 | 28 | 220 | 51% | 1.17x | 1.05x | 0.52x | ICL | `xxh3-64 vs xxhash-rust` `64` | 2.58x | s390x | `xxh3-64 vs xxhash-rust` `16384` |
| RapidHash | 122 | 246 | 72 | 440 | 28% | 1.06x | 1.00x | 0.16x | macOS | `rapidhash-v3-64 vs rapidhash` `1` | 3.61x | macOS | `rapidhash-128 vs rapidhash` `262144` |
| Auth/KDF | 341 | 280 | 49 | 670 | 51% | 1.23x | 1.05x | 0.01x | SPR | `hmac-sha256/streaming vs rustcrypto` `64B` | 85.40x | SPR | `hkdf-sha256/expand vs rustcrypto` `32` |
| AEAD | 839 | 193 | 68 | 1100 | 76% | 1.90x | 1.42x | 0.45x | ICL | `aes-256-gcm/encrypt vs rustcrypto` `0` | 32.36x | s390x | `aes-256-gcm-siv/encrypt vs rustcrypto` `0` |

## By Platform

| Platform | W | T | L | Total | Win % | Geomean | Median | Worst | Worst Category | Worst Case | Best | Best Category | Best Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| Zen4 | 470 | 159 | 15 | 644 | 73% | 1.62x | 1.18x | 0.49x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `32` | 129.37x | Checksums | `crc16-ibm vs crc` `262144` |
| Zen5 | 386 | 230 | 28 | 644 | 60% | 1.65x | 1.12x | 0.56x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `32` | 129.33x | Checksums | `crc16-ibm vs crc` `262144` |
| SPR | 498 | 100 | 46 | 644 | 77% | 1.92x | 1.28x | 0.01x | Auth/KDF | `hmac-sha256/streaming vs rustcrypto` `64B` | 225.37x | Checksums | `crc16-ccitt vs crc` `16384` |
| ICL | 472 | 151 | 21 | 644 | 73% | 1.62x | 1.22x | 0.45x | AEAD | `aes-256-gcm/encrypt vs rustcrypto` `0` | 119.74x | Checksums | `crc16-ibm vs crc` `262144` |
| Grav3 | 331 | 273 | 40 | 644 | 51% | 1.63x | 1.06x | 0.71x | RapidHash | `rapidhash-64 vs rapidhash` `0` | 126.96x | Checksums | `crc16-ibm vs crc` `262144` |
| Grav4 | 327 | 294 | 23 | 644 | 51% | 1.65x | 1.05x | 0.67x | RapidHash | `rapidhash-64 vs rapidhash` `0` | 105.01x | Checksums | `crc16-ibm vs crc` `262144` |
| s390x | 519 | 86 | 39 | 644 | 81% | 3.01x | 2.33x | 0.50x | RapidHash | `rapidhash-v3-64 vs rapidhash` `1` | 100.74x | Checksums | `crc16-ccitt vs crc` `65536` |
| POWER10 | 328 | 295 | 21 | 644 | 51% | 1.94x | 1.06x | 0.73x | RapidHash | `rapidhash-v3-128 vs rapidhash` `0` | 179.42x | Checksums | `crc16-ibm vs crc` `1048576` |
| RISC-V | 305 | 276 | 63 | 644 | 47% | 1.17x | 1.04x | 0.56x | Checksums | `crc32c vs crc32c` `32` | 11.10x | Checksums | `crc64-nvme vs crc-fast` `0` |
| macOS | 304 | 295 | 45 | 644 | 47% | 1.56x | 1.05x | 0.16x | RapidHash | `rapidhash-v3-64 vs rapidhash` `1` | 171.31x | Checksums | `crc16-ibm vs crc` `1048576` |

## Platform x Category

| Platform | Category | W | T | L | Total | Win % | Geomean | Worst | Worst Case |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zen4 | Checksums | 90 | 20 | 0 | 110 | 82% | 4.46x | 0.95x | `crc32 vs crc-fast` `1024` |
| Zen4 | SHA-2 | 32 | 27 | 0 | 59 | 54% | 1.12x | 1.00x | `sha256/streaming vs sha2` `64B` |
| Zen4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.28x | 2.24x | `sha3-256/streaming vs sha3` `4096B` |
| Zen4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.05x | 2.24x | `shake128 vs sha3` `1048576` |
| Zen4 | BLAKE2 | 62 | 32 | 0 | 94 | 66% | 1.08x | 1.00x | `blake2 vs rustcrypto` `blake2s256/0` |
| Zen4 | BLAKE3 | 24 | 23 | 1 | 48 | 50% | 1.33x | 0.95x | `blake3/xof vs blake3` `64` |
| Zen4 | Ascon | 22 | 0 | 0 | 22 | 100% | 1.93x | 1.67x | `ascon-hash256 vs ascon-hash` `1` |
| Zen4 | XXH3 | 13 | 4 | 5 | 22 | 59% | 1.03x | 0.72x | `xxh3-64 vs xxhash-rust` `256` |
| Zen4 | RapidHash | 13 | 31 | 0 | 44 | 30% | 1.07x | 0.95x | `rapidhash-v3-128 vs rapidhash` `65536` |
| Zen4 | Auth/KDF | 56 | 7 | 4 | 67 | 84% | 1.17x | 0.92x | `pbkdf2-sha256/iters=1000 vs rustcrypto` `64` |
| Zen4 | AEAD | 90 | 15 | 5 | 110 | 82% | 1.27x | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| Zen5 | Checksums | 91 | 18 | 1 | 110 | 83% | 4.79x | 0.94x | `crc32c vs crc-fast` `1024` |
| Zen5 | SHA-2 | 26 | 33 | 0 | 59 | 44% | 1.09x | 1.00x | `sha256/streaming vs sha2` `64B` |
| Zen5 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 2.16x | 2.12x | `sha3-256/streaming vs sha3` `64B` |
| Zen5 | SHAKE | 22 | 0 | 0 | 22 | 100% | 2.89x | 2.15x | `shake256 vs sha3` `1048576` |
| Zen5 | BLAKE2 | 14 | 80 | 0 | 94 | 15% | 1.02x | 0.98x | `blake2/streaming vs rustcrypto` `blake2b256/64B` |
| Zen5 | BLAKE3 | 21 | 23 | 4 | 48 | 44% | 1.35x | 0.85x | `blake3/keyed vs blake3` `256` |
| Zen5 | Ascon | 22 | 0 | 0 | 22 | 100% | 2.17x | 1.84x | `ascon-hash256 vs ascon-hash` `1` |
| Zen5 | XXH3 | 15 | 6 | 1 | 22 | 68% | 1.29x | 0.91x | `xxh3-64 vs xxhash-rust` `64` |
| Zen5 | RapidHash | 13 | 21 | 10 | 44 | 30% | 1.05x | 0.82x | `rapidhash-v3-64 vs rapidhash` `0` |
| Zen5 | Auth/KDF | 30 | 30 | 7 | 67 | 45% | 1.11x | 0.92x | `ed25519/verify vs dalek` `0` |
| Zen5 | AEAD | 86 | 19 | 5 | 110 | 78% | 1.39x | 0.56x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| SPR | Checksums | 95 | 11 | 4 | 110 | 86% | 4.86x | 0.85x | `crc32 vs crc32fast` `256` |
| SPR | SHA-2 | 52 | 5 | 2 | 59 | 88% | 3.28x | 0.01x | `sha256/streaming vs sha2` `64B` |
| SPR | SHA-3 | 46 | 0 | 0 | 46 | 100% | 3.61x | 3.38x | `sha3-512 vs sha3` `64` |
| SPR | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.81x | 3.47x | `shake256 vs sha3` `65536` |
| SPR | BLAKE2 | 88 | 6 | 0 | 94 | 94% | 1.20x | 0.99x | `blake2 vs rustcrypto` `blake2b512/1` |
| SPR | BLAKE3 | 22 | 17 | 9 | 48 | 46% | 1.24x | 0.86x | `blake3/xof vs blake3` `64` |
| SPR | Ascon | 22 | 0 | 0 | 22 | 100% | 1.47x | 1.28x | `ascon-hash256 vs ascon-hash` `1` |
| SPR | XXH3 | 11 | 8 | 3 | 22 | 50% | 1.10x | 0.55x | `xxh3-64 vs xxhash-rust` `64` |
| SPR | RapidHash | 12 | 22 | 10 | 44 | 27% | 1.06x | 0.77x | `rapidhash-64 vs rapidhash` `0` |
| SPR | Auth/KDF | 43 | 16 | 8 | 67 | 64% | 1.21x | 0.01x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| SPR | AEAD | 85 | 15 | 10 | 110 | 77% | 1.29x | 0.47x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| ICL | Checksums | 89 | 20 | 1 | 110 | 81% | 3.98x | 0.58x | `crc32 vs crc32fast` `256` |
| ICL | SHA-2 | 38 | 19 | 2 | 59 | 64% | 1.12x | 0.82x | `sha256/streaming vs sha2` `4096B` |
| ICL | SHA-3 | 46 | 0 | 0 | 46 | 100% | 3.00x | 2.84x | `sha3-256/streaming vs sha3` `64B` |
| ICL | SHAKE | 22 | 0 | 0 | 22 | 100% | 4.02x | 2.99x | `shake128 vs sha3` `1048576` |
| ICL | BLAKE2 | 94 | 0 | 0 | 94 | 100% | 1.20x | 1.10x | `blake2 vs rustcrypto` `blake2s256/1` |
| ICL | BLAKE3 | 14 | 29 | 5 | 48 | 29% | 1.16x | 0.82x | `blake3/keyed vs blake3` `256` |
| ICL | Ascon | 22 | 0 | 0 | 22 | 100% | 1.41x | 1.25x | `ascon-hash256 vs ascon-hash` `1` |
| ICL | XXH3 | 13 | 6 | 3 | 22 | 59% | 1.25x | 0.52x | `xxh3-64 vs xxhash-rust` `64` |
| ICL | RapidHash | 12 | 30 | 2 | 44 | 27% | 1.07x | 0.75x | `rapidhash-64 vs rapidhash` `32` |
| ICL | Auth/KDF | 37 | 27 | 3 | 67 | 55% | 1.10x | 0.83x | `hmac-sha256/streaming vs rustcrypto` `4096B` |
| ICL | AEAD | 85 | 20 | 5 | 110 | 77% | 1.25x | 0.45x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| Grav3 | Checksums | 78 | 26 | 6 | 110 | 71% | 3.66x | 0.92x | `crc32 vs crc-fast` `1048576` |
| Grav3 | SHA-2 | 21 | 37 | 1 | 59 | 36% | 1.06x | 0.91x | `sha256/streaming vs sha2` `64B` |
| Grav3 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.81x | 1.74x | `sha3-256/streaming vs sha3` `64B` |
| Grav3 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.82x | 1.77x | `shake256 vs sha3` `1048576` |
| Grav3 | BLAKE2 | 25 | 69 | 0 | 94 | 27% | 1.03x | 0.99x | `blake2/streaming vs rustcrypto` `blake2b256/64B` |
| Grav3 | BLAKE3 | 21 | 22 | 5 | 48 | 44% | 1.40x | 0.90x | `blake3/keyed vs blake3` `1` |
| Grav3 | Ascon | 4 | 18 | 0 | 22 | 18% | 1.03x | 0.99x | `ascon-xof128 vs ascon-hash` `262144` |
| Grav3 | XXH3 | 6 | 14 | 2 | 22 | 27% | 1.07x | 0.86x | `xxh3-128 vs xxhash-rust` `256` |
| Grav3 | RapidHash | 12 | 19 | 13 | 44 | 27% | 1.06x | 0.71x | `rapidhash-64 vs rapidhash` `0` |
| Grav3 | Auth/KDF | 31 | 29 | 7 | 67 | 46% | 1.09x | 0.90x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| Grav3 | AEAD | 65 | 39 | 6 | 110 | 59% | 2.48x | 0.77x | `aegis-256/encrypt vs aegis-crate` `0` |
| Grav4 | Checksums | 79 | 29 | 2 | 110 | 72% | 3.68x | 0.82x | `crc32c vs crc-fast` `256` |
| Grav4 | SHA-2 | 20 | 38 | 1 | 59 | 34% | 1.05x | 0.90x | `sha256/streaming vs sha2` `64B` |
| Grav4 | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.97x | 1.88x | `sha3-256/streaming vs sha3` `64B` |
| Grav4 | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.98x | 1.94x | `shake256 vs sha3` `262144` |
| Grav4 | BLAKE2 | 21 | 73 | 0 | 94 | 22% | 1.03x | 0.99x | `blake2/streaming vs rustcrypto` `blake2b256/64B` |
| Grav4 | BLAKE3 | 24 | 22 | 2 | 48 | 50% | 1.47x | 0.94x | `blake3 vs blake3` `1` |
| Grav4 | Ascon | 12 | 10 | 0 | 22 | 55% | 1.08x | 1.01x | `ascon-hash256 vs ascon-hash` `32` |
| Grav4 | XXH3 | 6 | 12 | 4 | 22 | 27% | 1.03x | 0.77x | `xxh3-128 vs xxhash-rust` `256` |
| Grav4 | RapidHash | 12 | 29 | 3 | 44 | 27% | 1.07x | 0.67x | `rapidhash-64 vs rapidhash` `0` |
| Grav4 | Auth/KDF | 20 | 45 | 2 | 67 | 30% | 1.07x | 0.87x | `pbkdf2-sha256/iters=1 vs rustcrypto` `32` |
| Grav4 | AEAD | 65 | 36 | 9 | 110 | 59% | 2.46x | 0.75x | `aegis-256/encrypt vs aegis-crate` `1` |
| s390x | Checksums | 94 | 6 | 10 | 110 | 85% | 7.06x | 0.63x | `crc24-openpgp vs crc` `64` |
| s390x | SHA-2 | 59 | 0 | 0 | 59 | 100% | 5.21x | 1.87x | `sha224 vs sha2` `64` |
| s390x | SHA-3 | 46 | 0 | 0 | 46 | 100% | 4.67x | 1.10x | `sha3-512 vs sha3` `0` |
| s390x | SHAKE | 22 | 0 | 0 | 22 | 100% | 3.95x | 1.09x | `shake256 vs sha3` `64` |
| s390x | BLAKE2 | 54 | 38 | 2 | 94 | 57% | 1.09x | 0.87x | `blake2/keyed vs rustcrypto` `blake2b256/1024` |
| s390x | BLAKE3 | 39 | 5 | 4 | 48 | 81% | 1.85x | 0.74x | `blake3/xof vs blake3` `1024` |
| s390x | Ascon | 4 | 18 | 0 | 22 | 18% | 1.04x | 0.99x | `ascon-xof128 vs ascon-hash` `262144` |
| s390x | XXH3 | 18 | 1 | 3 | 22 | 82% | 1.71x | 0.63x | `xxh3-64 vs xxhash-rust` `32` |
| s390x | RapidHash | 9 | 16 | 19 | 44 | 20% | 0.99x | 0.50x | `rapidhash-v3-64 vs rapidhash` `1` |
| s390x | Auth/KDF | 64 | 2 | 1 | 67 | 96% | 3.51x | 0.90x | `ed25519/verify vs dalek` `32` |
| s390x | AEAD | 110 | 0 | 0 | 110 | 100% | 4.33x | 1.17x | `chacha20-poly1305/encrypt vs rustcrypto` `32` |
| POWER10 | Checksums | 104 | 3 | 3 | 110 | 95% | 10.80x | 0.79x | `crc64-nvme vs crc64fast-nvme` `64` |
| POWER10 | SHA-2 | 8 | 51 | 0 | 59 | 14% | 1.01x | 0.99x | `sha256 vs sha2` `256` |
| POWER10 | SHA-3 | 16 | 30 | 0 | 46 | 35% | 1.04x | 0.97x | `sha3-256/streaming vs sha3` `64B` |
| POWER10 | SHAKE | 14 | 8 | 0 | 22 | 64% | 1.39x | 1.00x | `shake128 vs sha3` `65536` |
| POWER10 | BLAKE2 | 8 | 80 | 6 | 94 | 9% | 1.00x | 0.94x | `blake2/keyed vs rustcrypto` `blake2s128/1` |
| POWER10 | BLAKE3 | 35 | 13 | 0 | 48 | 73% | 2.01x | 0.95x | `blake3 vs blake3` `1` |
| POWER10 | Ascon | 5 | 17 | 0 | 22 | 23% | 1.06x | 1.01x | `ascon-hash256 vs ascon-hash` `64` |
| POWER10 | XXH3 | 19 | 3 | 0 | 22 | 86% | 1.29x | 0.97x | `xxh3-64 vs xxhash-rust` `1` |
| POWER10 | RapidHash | 11 | 27 | 6 | 44 | 25% | 1.08x | 0.73x | `rapidhash-v3-128 vs rapidhash` `0` |
| POWER10 | Auth/KDF | 12 | 55 | 0 | 67 | 18% | 1.06x | 0.98x | `hmac-sha384 vs rustcrypto` `256` |
| POWER10 | AEAD | 96 | 8 | 6 | 110 | 87% | 2.63x | 0.93x | `aegis-256/encrypt vs aegis-crate` `16384` |
| RISC-V | Checksums | 69 | 16 | 25 | 110 | 63% | 1.44x | 0.56x | `crc32c vs crc32c` `32` |
| RISC-V | SHA-2 | 18 | 41 | 0 | 59 | 31% | 1.04x | 0.98x | `sha256 vs sha2` `16384` |
| RISC-V | SHA-3 | 46 | 0 | 0 | 46 | 100% | 1.18x | 1.07x | `sha3-256/streaming vs sha3` `4096B` |
| RISC-V | SHAKE | 22 | 0 | 0 | 22 | 100% | 1.62x | 1.13x | `shake256 vs sha3` `65536` |
| RISC-V | BLAKE2 | 13 | 79 | 2 | 94 | 14% | 1.01x | 0.94x | `blake2 vs rustcrypto` `blake2b512/262144` |
| RISC-V | BLAKE3 | 21 | 23 | 4 | 48 | 44% | 1.18x | 0.89x | `blake3/xof vs blake3` `262144` |
| RISC-V | Ascon | 12 | 10 | 0 | 22 | 55% | 1.06x | 0.95x | `ascon-hash256 vs ascon-hash` `0` |
| RISC-V | XXH3 | 5 | 17 | 0 | 22 | 23% | 1.03x | 0.95x | `xxh3-128 vs xxhash-rust` `1024` |
| RISC-V | RapidHash | 12 | 28 | 4 | 44 | 27% | 1.08x | 0.84x | `rapidhash-64 vs rapidhash` `64` |
| RISC-V | Auth/KDF | 34 | 27 | 6 | 67 | 51% | 1.08x | 0.93x | `pbkdf2-sha256/iters=100 vs rustcrypto` `64` |
| RISC-V | AEAD | 53 | 35 | 22 | 110 | 48% | 1.20x | 0.79x | `aes-256-gcm/encrypt vs rustcrypto` `1048576` |
| macOS | Checksums | 93 | 14 | 3 | 110 | 85% | 4.23x | 0.88x | `crc32 vs crc-fast` `256` |
| macOS | SHA-2 | 11 | 44 | 4 | 59 | 19% | 1.04x | 0.81x | `sha224 vs sha2` `32` |
| macOS | SHA-3 | 6 | 35 | 5 | 46 | 13% | 1.02x | 0.91x | `sha3-384 vs sha3` `0` |
| macOS | SHAKE | 7 | 9 | 6 | 22 | 32% | 1.00x | 0.90x | `shake128 vs sha3` `262144` |
| macOS | BLAKE2 | 13 | 81 | 0 | 94 | 14% | 1.02x | 0.96x | `blake2 vs rustcrypto` `blake2s128/0` |
| macOS | BLAKE3 | 17 | 27 | 4 | 48 | 35% | 1.20x | 0.62x | `blake3 vs blake3` `65536` |
| macOS | Ascon | 17 | 5 | 0 | 22 | 77% | 1.09x | 1.01x | `ascon-hash256 vs ascon-hash` `0` |
| macOS | XXH3 | 6 | 9 | 7 | 22 | 27% | 1.04x | 0.75x | `xxh3-64 vs xxhash-rust` `32` |
| macOS | RapidHash | 16 | 23 | 5 | 44 | 36% | 1.10x | 0.16x | `rapidhash-v3-64 vs rapidhash` `1` |
| macOS | Auth/KDF | 14 | 42 | 11 | 67 | 21% | 1.01x | 0.32x | `ed25519/verify vs dalek` `1024` |
| macOS | AEAD | 104 | 6 | 0 | 110 | 95% | 2.59x | 0.99x | `aegis-256/encrypt vs aegis-crate` `0` |

## Worst Losses

| Rank | Platform | Category | Speedup | Case |
|---:|---|---|---:|---|
| 1 | SPR | Auth/KDF | 0.01x | `hmac-sha256/streaming vs rustcrypto` `64B` |
| 2 | SPR | SHA-2 | 0.01x | `sha256/streaming vs sha2` `64B` |
| 3 | SPR | SHA-2 | 0.01x | `sha256/streaming vs sha2` `4096B` |
| 4 | SPR | Auth/KDF | 0.01x | `hmac-sha256/streaming vs rustcrypto` `4096B` |
| 5 | macOS | RapidHash | 0.16x | `rapidhash-v3-64 vs rapidhash` `1` |
| 6 | macOS | Auth/KDF | 0.32x | `ed25519/verify vs dalek` `1024` |
| 7 | ICL | AEAD | 0.45x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 8 | SPR | AEAD | 0.47x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 9 | ICL | AEAD | 0.48x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 10 | Zen4 | AEAD | 0.49x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 11 | SPR | AEAD | 0.50x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 12 | s390x | RapidHash | 0.50x | `rapidhash-v3-64 vs rapidhash` `1` |
| 13 | ICL | XXH3 | 0.52x | `xxh3-64 vs xxhash-rust` `64` |
| 14 | Zen4 | AEAD | 0.54x | `aes-256-gcm/encrypt vs rustcrypto` `0` |
| 15 | SPR | XXH3 | 0.55x | `xxh3-64 vs xxhash-rust` `64` |
| 16 | RISC-V | Checksums | 0.56x | `crc32c vs crc32c` `32` |
| 17 | Zen5 | AEAD | 0.56x | `aes-256-gcm/encrypt vs rustcrypto` `32` |
| 18 | macOS | Auth/KDF | 0.57x | `ed25519/sign vs dalek` `16384` |
| 19 | ICL | Checksums | 0.58x | `crc32 vs crc32fast` `256` |
| 20 | ICL | AEAD | 0.60x | `aes-256-gcm/encrypt vs rustcrypto` `1` |
