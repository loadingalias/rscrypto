# Benchmark Overview

Sources:

- Linux benchmark CI run [#25689759367](https://github.com/loadingalias/rscrypto/actions/runs/25689759367), created 2026-05-11 18:37:10 UTC.
- Local macOS aarch64 bench file: `benchmark_results/2026-05-11/macos/aarch64/results.txt`.
- Commit: `819da767dd28ebe0302089b9205d09452e386811`.

Scope: current 2026-05-11 global benchmark results: nine Linux dedicated runners plus one local macOS aarch64 run. Ratios are `rscrypto` divided by the external implementation. For throughput rows, higher throughput is better. For latency-only rows, the ratio is external time divided by `rscrypto` time, so higher is still better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and size.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| All matched performance pairs | 11731 | 7125/2674/1932 | 61% | 1.62x | 1.14x |
| Fastest external per case | 6487 | 3428/1766/1293 | 53% | 1.40x | 1.06x |
| All matched throughput pairs | 10579 | 6244/2558/1777 | 59% | 1.63x | 1.12x |
| Fastest external, throughput only | 5869 | 3002/1694/1173 | 51% | 1.42x | 1.05x |
| All matched latency-only pairs | 1152 | 881/116/155 | 76% | 1.53x | 1.31x |
| Fastest external, latency only | 618 | 426/72/120 | 69% | 1.25x | 1.18x |

What matters:

- AES-GCM sustained throughput is now at parity on x86-64 and local macOS aarch64, but still materially behind the fastest external implementation on Graviton3, Graviton4, Power10, and RISC-V.
- GCM-SIV sustained throughput is broadly strong: every platform wins except RISC-V, with Power10 mixed but positive on geomean.
- The obvious non-AEAD problem is non-cryptographic hashing. The 128-bit Rapidhash/XXH3 family is not close to GXHash-class competitors.

## Clear Losses

Collapsed fastest-external groups. This is the useful root-cause view, not just the worst individual rows.

| Place | Fastest rows | W/T/L | Geomean | Median | Main pressure |
|---|---:|---:|---:|---:|---|
| 128-bit non-crypto hashes | 330 | 59/44/227 | 0.46x | 0.36x | `gxhash` 231, `rapidhash` 66, `xxhash-rust` 33 |
| 64-bit non-crypto hashes | 330 | 44/35/251 | 0.65x | 0.64x | `gxhash` 207, `foldhash` 61, `rapidhash` 30 |
| ChaCha20-Poly1305 | 220 | 69/23/128 | 0.81x | 0.92x | `ring` 121, `aws-lc-rs` 85, `rustcrypto` 14 |

Top measured primitive/operation losses, fastest external only:

| Primitive/op | Rows | W/T/L | Geomean | Median |
|---|---:|---:|---:|---:|
| `rapidhash-v3-128` / `hash` | 110 | 10/18/82 | 0.38x | 0.27x |
| `rapidhash-128` / `hash` | 110 | 17/15/78 | 0.41x | 0.27x |
| `xxh3-128` / `hash` | 110 | 32/11/67 | 0.61x | 0.68x |
| `rapidhash-v3-64` / `hash` | 110 | 5/9/96 | 0.62x | 0.60x |
| `rapidhash-64` / `hash` | 110 | 18/14/78 | 0.67x | 0.62x |
| `xxh3-64` / `hash` | 110 | 21/12/77 | 0.68x | 0.72x |
| `hmac-sha256` / `streaming` | 20 | 2/12/6 | 0.71x | 0.99x |
| `sha256` / `streaming` | 20 | 2/13/5 | 0.72x | 0.99x |

## Sustained AEAD

Fastest-external throughput comparisons only, sizes `65536`, `262144`, and `1048576`, both encrypt and decrypt, 128-bit and 256-bit keys.

| Platform | AES-GCM W/T/L | AES-GCM geomean | GCM-SIV W/T/L | GCM-SIV geomean |
|---|---:|---:|---:|---:|
| AMD Zen4 | 0/12/0 | 1.00x | 12/0/0 | 1.74x |
| AMD Zen5 | 0/8/4 | 0.98x | 12/0/0 | 2.05x |
| Intel Ice Lake | 3/8/1 | 1.01x | 12/0/0 | 1.35x |
| Intel Sapphire Rapids | 0/9/3 | 0.97x | 12/0/0 | 1.51x |
| AWS Graviton3 | 0/0/12 | 0.63x | 12/0/0 | 2.53x |
| AWS Graviton4 | 0/0/12 | 0.64x | 12/0/0 | 2.54x |
| IBM Power10 | 0/0/12 | 0.36x | 6/0/6 | 1.46x |
| IBM z16/s390x | 12/0/0 | 12.46x | 12/0/0 | 5.35x |
| RISE RISC-V | 0/0/12 | 0.85x | 0/0/12 | 0.89x |
| macOS aarch64 | 0/12/0 | 1.01x | 12/0/0 | 2.80x |

Primitive-level sustained AEAD geomeans:

| Platform | AES-128-GCM | AES-256-GCM | AES-128-GCM-SIV | AES-256-GCM-SIV |
|---|---:|---:|---:|---:|
| AMD Zen4 | 1.01x | 0.99x | 1.77x | 1.72x |
| AMD Zen5 | 1.02x | 0.95x | 2.05x | 2.05x |
| Intel Ice Lake | 1.02x | 1.00x | 1.32x | 1.37x |
| Intel Sapphire Rapids | 0.98x | 0.96x | 1.49x | 1.53x |
| AWS Graviton3 | 0.64x | 0.62x | 2.40x | 2.67x |
| AWS Graviton4 | 0.66x | 0.62x | 2.39x | 2.69x |
| IBM Power10 | 0.36x | 0.36x | 1.44x | 1.48x |
| IBM z16/s390x | 11.90x | 13.05x | 5.25x | 5.46x |
| RISE RISC-V | 0.84x | 0.86x | 0.88x | 0.90x |
| macOS aarch64 | 1.02x | 1.00x | 2.72x | 2.87x |

## Platform Summary

Fastest-external comparisons across every matched primitive, operation, and size.

| Platform | Rows | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| AMD Zen4 | 643 | 379/137/127 | 59% | 1.31x | 1.08x |
| AMD Zen5 | 643 | 321/198/124 | 50% | 1.33x | 1.05x |
| Intel Ice Lake | 643 | 386/109/148 | 60% | 1.32x | 1.13x |
| Intel Sapphire Rapids | 643 | 386/109/148 | 60% | 1.36x | 1.11x |
| AWS Graviton3 | 643 | 290/196/157 | 45% | 1.24x | 1.02x |
| AWS Graviton4 | 643 | 295/191/157 | 46% | 1.26x | 1.03x |
| IBM Power10 | 643 | 297/261/85 | 46% | 1.48x | 1.03x |
| IBM z16/s390x | 643 | 519/78/46 | 81% | 3.11x | 2.31x |
| RISE RISC-V | 643 | 264/220/159 | 41% | 1.10x | 1.02x |
| macOS aarch64 | 700 | 291/267/142 | 42% | 1.22x | 1.03x |

## Primitive Summary

All primitives with matched `rscrypto` comparisons. The fastest-only columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
|---|---:|---:|---:|---:|---:|---:|
| `rapidhash-v3-128` | 110 | 10/18/82 | 0.38x | 187 | 36/52/99 | 0.58x |
| `rapidhash-128` | 110 | 17/15/78 | 0.41x | 187 | 46/61/80 | 0.63x |
| `xxh3-128` | 110 | 32/11/67 | 0.61x | 187 | 65/40/82 | 0.78x |
| `rapidhash-v3-64` | 110 | 5/9/96 | 0.62x | 407 | 131/101/175 | 0.97x |
| `rapidhash-64` | 110 | 18/14/78 | 0.67x | 407 | 165/112/130 | 1.05x |
| `xxh3-64` | 110 | 21/12/77 | 0.68x | 407 | 205/50/152 | 1.07x |
| `x25519` | 20 | 6/1/13 | 0.75x | 60 | 44/3/13 | 1.15x |
| `chacha20-poly1305` | 220 | 69/23/128 | 0.81x | 660 | 324/103/233 | 0.98x |
| `hmac-sha256` | 130 | 29/52/49 | 0.86x | 350 | 144/124/82 | 1.04x |
| `hmac-sha512` | 110 | 16/31/63 | 0.97x | 330 | 109/104/117 | 1.11x |
| `hmac-sha384` | 110 | 17/30/63 | 0.98x | 330 | 121/95/114 | 1.12x |
| `ed25519` | 100 | 44/11/45 | 1.03x | 340 | 239/29/72 | 1.27x |
| `blake2/blake2b256` | 268 | 113/150/5 | 1.05x | 419 | 256/158/5 | 1.33x |
| `blake2/blake2b512` | 220 | 102/116/2 | 1.06x | 341 | 218/121/2 | 1.36x |
| `blake2/blake2s256` | 268 | 121/147/0 | 1.09x | 268 | 121/147/0 | 1.09x |
| `sha256` | 130 | 43/63/24 | 1.10x | 350 | 193/114/43 | 1.41x |
| `blake2/blake2s128` | 220 | 123/96/1 | 1.11x | 220 | 123/96/1 | 1.11x |
| `hkdf-sha256` | 40 | 27/11/2 | 1.18x | 120 | 107/11/2 | 1.85x |
| `hkdf-sha384` | 40 | 24/16/0 | 1.21x | 120 | 104/16/0 | 2.74x |
| `sha512` | 130 | 47/72/11 | 1.22x | 350 | 201/132/17 | 1.31x |
| `sha384` | 110 | 41/58/11 | 1.22x | 330 | 193/120/17 | 1.32x |
| `kmac256` | 11 | 11/0/0 | 1.23x | 11 | 11/0/0 | 1.23x |
| `ascon-hash256` | 110 | 62/47/1 | 1.24x | 110 | 62/47/1 | 1.24x |
| `sha512-256` | 110 | 54/55/1 | 1.25x | 110 | 54/55/1 | 1.25x |
| `cshake256` | 10 | 10/0/0 | 1.25x | 10 | 10/0/0 | 1.25x |
| `aes-256-gcm` | 220 | 105/41/74 | 1.26x | 660 | 479/62/119 | 2.61x |
| `aes-128-gcm` | 220 | 113/42/65 | 1.27x | 660 | 497/61/102 | 2.59x |
| `xchacha20-poly1305` | 220 | 165/53/2 | 1.27x | 220 | 165/53/2 | 1.27x |
| `ascon-xof128` | 110 | 83/26/1 | 1.33x | 110 | 83/26/1 | 1.33x |
| `blake3` | 480 | 242/206/32 | 1.40x | 480 | 242/206/32 | 1.40x |
| `aegis-256` | 220 | 131/66/23 | 1.42x | 220 | 131/66/23 | 1.42x |
| `crc32c` | 110 | 53/45/12 | 1.57x | 220 | 163/45/12 | 2.18x |
| `crc32` | 110 | 51/39/20 | 1.59x | 220 | 158/42/20 | 2.24x |
| `aes-128-gcm-siv` | 220 | 154/3/63 | 1.66x | 440 | 351/18/71 | 3.24x |
| `crc64-nvme` | 110 | 60/36/14 | 1.66x | 220 | 146/52/22 | 2.13x |
| `aes-256-gcm-siv` | 220 | 154/6/60 | 1.72x | 440 | 373/7/60 | 3.55x |
| `sha224` | 110 | 55/53/2 | 1.76x | 110 | 55/53/2 | 1.76x |
| `sha3-256` | 130 | 111/17/2 | 1.98x | 130 | 111/17/2 | 1.98x |
| `sha3-224` | 110 | 97/13/0 | 2.01x | 110 | 97/13/0 | 2.01x |
| `sha3-384` | 110 | 94/13/3 | 2.03x | 110 | 94/13/3 | 2.03x |
| `sha3-512` | 110 | 94/14/2 | 2.06x | 110 | 94/14/2 | 2.06x |
| `crc64-xz` | 110 | 84/15/11 | 2.27x | 110 | 84/15/11 | 2.27x |
| `shake128` | 110 | 98/6/6 | 2.35x | 110 | 98/6/6 | 2.35x |
| `shake256` | 110 | 96/13/1 | 2.37x | 110 | 96/13/1 | 2.37x |
| `crc24-openpgp` | 110 | 108/1/1 | 14.62x | 110 | 108/1/1 | 14.62x |
| `crc16-ccitt` | 110 | 109/0/1 | 24.78x | 110 | 109/0/1 | 24.78x |
| `crc16-ibm` | 110 | 109/0/1 | 25.23x | 110 | 109/0/1 | 25.23x |

## External Pressure

All-pair comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| `rustcrypto` | 2726 | 1755/860/111 | 64% | 1.96x | 1.12x |
| `aws-lc-rs` | 1940 | 1092/282/566 | 56% | 1.38x | 1.12x |
| `ring` | 1480 | 896/210/374 | 61% | 1.40x | 1.15x |
| `sha3` | 680 | 590/76/14 | 87% | 2.12x | 2.01x |
| `sha2` | 590 | 284/296/10 | 48% | 1.40x | 1.05x |
| `blake3` | 480 | 242/206/32 | 50% | 1.40x | 1.05x |
| `gxhash` | 462 | 43/10/409 | 9% | 0.42x | 0.42x |
| `rapidhash` | 440 | 142/235/63 | 32% | 1.10x | 1.01x |
| `dryoc` | 372 | 343/22/7 | 92% | 1.83x | 1.70x |
| `ahash` | 330 | 223/17/90 | 68% | 1.38x | 1.40x |
| `crc` | 330 | 326/1/3 | 99% | 20.91x | 32.17x |
| `crc-fast` | 330 | 191/114/25 | 58% | 1.91x | 1.09x |
| `foldhash` | 330 | 128/78/124 | 39% | 1.08x | 1.00x |
| `aegis-crate` | 220 | 131/66/23 | 60% | 1.42x | 1.10x |
| `ascon-hash` | 220 | 145/73/2 | 66% | 1.28x | 1.12x |
| `xxhash-rust` | 220 | 112/76/32 | 51% | 1.18x | 1.06x |
| `dalek` | 120 | 101/12/7 | 84% | 1.35x | 1.32x |
| `crc32c` | 110 | 102/4/4 | 93% | 2.78x | 2.21x |
| `crc32fast` | 110 | 91/8/11 | 83% | 2.51x | 2.08x |
| `crc64fast` | 110 | 84/15/11 | 76% | 2.27x | 1.91x |
| `crc64fast-nvme` | 110 | 83/13/14 | 75% | 2.24x | 1.90x |
| `tiny-keccak` | 21 | 21/0/0 | 100% | 1.24x | 1.25x |

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
|---|---|---|---:|---|
| AMD Zen4 | `ci` | `2026-05-11 18_37_10` | 1904 | `benchmark_results/2026-05-11/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-05-11 18_37_10` | 1904 | `benchmark_results/2026-05-11/linux/amd-zen5/results.txt` |
| Intel Ice Lake | `ci` | `2026-05-11 18_37_10` | 1904 | `benchmark_results/2026-05-11/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-05-11 18_37_10` | 1904 | `benchmark_results/2026-05-11/linux/intel-spr/results.txt` |
| AWS Graviton3 | `ci` | `2026-05-11 18_37_10` | 1904 | `benchmark_results/2026-05-11/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-05-11 18_37_10` | 1904 | `benchmark_results/2026-05-11/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-05-11 18_37_10` | 1838 | `benchmark_results/2026-05-11/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-05-11 18_37_10` | 1838 | `benchmark_results/2026-05-11/linux/ibm-s390x/results.txt` |
| RISE RISC-V | `ci` | `2026-05-11 18_37_10` | 1838 | `benchmark_results/2026-05-11/linux/rise-riscv/results.txt` |
| macOS aarch64 | `local` | `2026-05-11 22_07_03` | 2113 | `benchmark_results/2026-05-11/macos/aarch64/results.txt` |

## Methodology Notes

- Parsed 19051 Criterion rows: 17234 throughput rows and 1817 latency-only rows.
- Matched 11731 `rscrypto` external pairs across 6487 fastest-external cases.
- Input files include 10 result files. CI result headers use `time=18_37_10`, the GitHub Actions run creation time, per extraction policy. The local macOS file keeps its local run time.
- Matching is structural: a comparison exists when an external Criterion ID can be produced by replacing the `rscrypto` path component while keeping the same platform, primitive path, operation path, and size suffix.
- Unmatched external-only rows: 28. These are internal comparisons or benches without a corresponding `rscrypto` row, so they are not part of the ratio tables.
