# Benchmark Overview

Sources:

- Linux benchmark CI baseline run [#27973944251](https://github.com/loadingalias/rscrypto/actions/runs/27973944251), created 2026-06-22 18:11:11 UTC.
- Linux baseline commit: `b978c2ca45611325850d7f1af94718e497acde50`.
- Baseline artifacts: nine successful `benchmark-*` artifacts extracted into `benchmark_results/2026-06-22/linux/*/results.txt`.
- BLAKE3 Linux refresh run [#28142543540](https://github.com/loadingalias/rscrypto/actions/runs/28142543540), created 2026-06-25 02:19:45 UTC at commit `3f209aa03cc0b8ac6875e7780f161174460e2253`; artifacts extracted into `benchmark_results/2026-06-25/linux/*/results.txt`.
- Local macOS run: `benchmark_results/2026-06-22/macos/aarch64/results.txt` at commit `b978c2ca45611325850d7f1af94718e497acde50`.

Scope: the 2026-06-22 nine-runner Linux CI benchmark matrix for commit `b978c2c`, with BLAKE3 rows refreshed from the 2026-06-25 Linux CI run for commit `3f209aa`. Ratios are `external_crate_time / rscrypto_time`; higher is better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and input shape. Internal kernel, scratch-buffer, padding-only, cold-path, PHC roundtrip, parallel-scaling, threshold-selection, public-overhead, and phase-attribution microbenches are parsed as raw rows but excluded from external win/loss claims. The macOS local run is listed separately and is not mixed into Linux CI claims.

Coverage note: this is a full Linux CI public benchmark pass. It includes checksum, hash, XOF, MAC, KDF, password-hashing, BLAKE2/BLAKE3, RSA import/verification, ECDSA P-256/P-384 signing and verification, Ed25519, X25519, AEAD, and ML-KEM-512/768/1024 keygen, encapsulation, and decapsulation rows. ML-KEM phase/arithmetic microbenches are present in the raw artifacts and intentionally excluded from release-level competitor claims.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Linux CI: all matched performance pairs | 10,781 | 7,544/2,381/856 | 70% | 1.79x | 1.22x |
| Linux CI: fastest external per case | 6,750 | 4,089/1,992/669 | 61% | 1.59x | 1.11x |

Shareable release summary:

- **Headline:** 4,089 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,081 are wins or ties. Linux CI fastest-external geomean is 1.59x.
- **Checksums:** 5.20x geomean across 693 fastest-external rows; W/T/L is 520/119/54.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,726 fastest-external rows; W/T/L is 2,074/1,435/217.
- **Auth/KDF:** 1.25x geomean across 180 fastest-external rows; W/T/L is 160/20/0.
- **Password hashing:** 1.08x geomean across 135 fastest-external rows; W/T/L is 68/28/39.
- **Public-key:** 1.21x geomean across 333 fastest-external rows; W/T/L is 195/80/58.
- **RSA:** 1.53x geomean across 99 fastest-external rows; W/T/L is 85/6/8.
- **AEAD:** 1.56x geomean across 1,584 fastest-external rows; W/T/L is 987/304/293.
- **ML-KEM:** 1.06x geomean across 81 fastest-external rows; W/T/L is 45/10/26.
- **ECDSA P-256/P-384:** Linux CI 1.41x geomean across 144 fastest-external rows; W/T/L is 114/10/20.
- **Top current loss areas:** `mlkem512` / `keygen`: 0.91x geomean across 9 rows; W/T/L 1/2/6; pressure `aws-lc-rs` 6, `libcrux` 2; `mlkem1024` / `keygen`: 0.95x geomean across 9 rows; W/T/L 4/1/4; pressure `aws-lc-rs` 3, `libcrux` 2; `ecdsa-p384` / `sign`: 0.95x geomean across 36 rows; W/T/L 16/0/20; pressure `aws-lc-rs` 20; `argon2id-owasp` / `hash`: 0.96x geomean across 9 rows; W/T/L 2/2/5; pressure `rustcrypto` 7; `ed25519` / `verify`: 0.99x geomean across 36 rows; W/T/L 9/18/9; pressure `dalek` 9, `aws-lc-rs` 9, `ring` 9.

## Coverage Matrix

| Platform | Raw Criterion rows | All pairs | Fastest rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 2,244 | 1,251 | 750 | 526/161/63 | 70% | 1.48x | 1.15x |
| AMD Zen5 | 2,244 | 1,251 | 750 | 447/233/70 | 60% | 1.50x | 1.11x |
| AWS Graviton3 | 2,250 | 1,251 | 750 | 370/289/91 | 49% | 1.37x | 1.05x |
| AWS Graviton4 | 2,250 | 1,251 | 750 | 366/326/58 | 49% | 1.38x | 1.05x |
| IBM Power10 | 1,995 | 1,012 | 750 | 387/318/45 | 52% | 1.88x | 1.06x |
| IBM z16/s390x | 1,995 | 1,012 | 750 | 609/97/44 | 81% | 3.15x | 2.34x |
| Intel Ice Lake | 2,244 | 1,251 | 750 | 519/150/81 | 69% | 1.47x | 1.17x |
| Intel Sapphire Rapids | 2,244 | 1,251 | 750 | 535/137/78 | 71% | 1.62x | 1.20x |
| RISE RISC-V | 2,244 | 1,251 | 750 | 336/274/140 | 45% | 1.09x | 1.03x |

## Category Summary

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 693 | 520/119/54 | 75% | 5.20x | 2.37x |
| Hashes/MACs/XOFs | 3,726 | 2,074/1,435/217 | 56% | 1.35x | 1.08x |
| Auth/KDF | 180 | 160/20/0 | 89% | 1.25x | 1.13x |
| Password hashing | 135 | 68/28/39 | 50% | 1.08x | 1.05x |
| Public-key | 333 | 195/80/58 | 59% | 1.21x | 1.10x |
| RSA | 99 | 85/6/8 | 86% | 1.53x | 1.17x |
| AEAD | 1,584 | 987/304/293 | 62% | 1.56x | 1.14x |

## BLAKE3 Summary

BLAKE3 rows are refreshed from Linux CI run [#28142543540](https://github.com/loadingalias/rscrypto/actions/runs/28142543540). The row set is the same 432 matched rscrypto-vs-official-`blake3` comparisons as the previous overview. All-pair and fastest-external BLAKE3 metrics are identical because official `blake3` is the only external implementation in this bench.

| Scope | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| All Linux BLAKE3 rows | 432 | 227/173/32 | 1.40x | 1.06x |
| x86_64 | 192 | 80/97/15 | 1.24x | 1.03x |
| AArch64 | 96 | 44/46/6 | 1.44x | 1.05x |

| Platform | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| AMD Zen4 | 48 | 21/24/3 | 1.31x | 1.03x |
| AMD Zen5 | 48 | 21/25/2 | 1.32x | 1.03x |
| AWS Graviton3 | 48 | 22/21/5 | 1.40x | 0.99x |
| AWS Graviton4 | 48 | 22/25/1 | 1.47x | 1.05x |
| IBM Power10 | 48 | 39/7/2 | 2.02x | 1.81x |
| IBM z16/s390x | 48 | 41/2/5 | 1.86x | 2.07x |
| Intel Ice Lake | 48 | 19/26/3 | 1.17x | 1.02x |
| Intel Sapphire Rapids | 48 | 19/22/7 | 1.16x | 1.03x |
| RISE RISC-V | 48 | 23/21/4 | 1.12x | 1.03x |

| Operation | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `oneshot` | 99 | 45/46/8 | 1.32x | 1.04x |
| `keyed` | 99 | 40/49/10 | 1.30x | 1.01x |
| `derive-key` | 99 | 76/22/1 | 1.78x | 1.87x |
| `streaming` | 36 | 10/21/5 | 1.15x | 1.01x |
| `xof` | 99 | 56/35/8 | 1.33x | 1.07x |

## ML-KEM Summary

ML-KEM public coverage is complete for the CI-selected primitive set: ML-KEM-512, ML-KEM-768, and ML-KEM-1024 each include keygen, encapsulate, and decapsulate on all nine Linux platforms. POWER10 and s390x do not have `aws-lc-rs` ML-KEM rows in this artifact set, but still have rscrypto plus `libcrux`, `fips203`, and RustCrypto comparison rows for every public operation.

| Platform | Raw ML-KEM rows | Fastest rows | W/T/L | Geomean | Median | Fastest external split |
| --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 45 | 9 | 8/0/1 | 1.17x | 1.10x | `libcrux` 7, `aws-lc-rs` 2 |
| AMD Zen5 | 45 | 9 | 8/1/0 | 1.21x | 1.12x | `libcrux` 9 |
| AWS Graviton3 | 45 | 9 | 0/1/8 | 0.82x | 0.83x | `aws-lc-rs` 9 |
| AWS Graviton4 | 45 | 9 | 0/1/8 | 0.81x | 0.80x | `aws-lc-rs` 9 |
| IBM Power10 | 36 | 9 | 6/2/1 | 1.12x | 1.10x | `libcrux` 9 |
| IBM z16/s390x | 36 | 9 | 9/0/0 | 1.37x | 1.38x | `libcrux` 9 |
| Intel Ice Lake | 45 | 9 | 5/3/1 | 1.13x | 1.08x | `libcrux` 8, `aws-lc-rs` 1 |
| Intel Sapphire Rapids | 45 | 9 | 8/0/1 | 1.19x | 1.16x | `aws-lc-rs` 5, `libcrux` 4 |
| RISE RISC-V | 45 | 9 | 1/2/6 | 0.89x | 0.86x | `aws-lc-rs` 9 |

| Primitive/op | Rows | W/T/L | Win % | Geomean | Median | Pressure |
| --- | --- | --- | --- | --- | --- | --- |
| `mlkem1024` / `decapsulate` | 9 | 6/0/3 | 67% | 1.03x | 1.10x | `aws-lc-rs` 3 |
| `mlkem1024` / `encapsulate` | 9 | 6/1/2 | 67% | 1.19x | 1.40x | `aws-lc-rs` 3 |
| `mlkem1024` / `keygen` | 9 | 4/1/4 | 44% | 0.95x | 1.01x | `aws-lc-rs` 3, `libcrux` 2 |
| `mlkem512` / `decapsulate` | 9 | 5/1/3 | 56% | 1.02x | 1.08x | `aws-lc-rs` 3, `libcrux` 1 |
| `mlkem512` / `encapsulate` | 9 | 7/0/2 | 78% | 1.22x | 1.32x | `aws-lc-rs` 2 |
| `mlkem512` / `keygen` | 9 | 1/2/6 | 11% | 0.91x | 0.93x | `aws-lc-rs` 6, `libcrux` 2 |
| `mlkem768` / `decapsulate` | 9 | 6/0/3 | 67% | 1.04x | 1.11x | `aws-lc-rs` 3 |
| `mlkem768` / `encapsulate` | 9 | 6/3/0 | 67% | 1.27x | 1.42x | `aws-lc-rs` 3 |
| `mlkem768` / `keygen` | 9 | 4/2/3 | 44% | 0.99x | 1.01x | `aws-lc-rs` 3, `libcrux` 2 |

## ECDSA Summary

ECDSA signing includes both deterministic and blinded rscrypto rows in raw results; aggregate fastest-external comparisons use the fastest rscrypto row for the exact case. Constant-time release evidence is tracked separately by `ct.toml` and CT workflow artifacts.

| Operation | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `ecdsa-p256` / `sign` | 36 | 36/0/0 | 1.45x | 1.34x |
| `ecdsa-p256` / `verify` | 36 | 26/10/0 | 1.58x | 1.13x |
| `ecdsa-p384` / `sign` | 36 | 16/0/20 | 0.95x | 0.91x |
| `ecdsa-p384` / `verify` | 36 | 36/0/0 | 1.80x | 1.39x |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are strongest-external comparisons; all-pair columns include every matched external implementation.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `argon2id-owasp` | 9 | 2/2/5 | 0.96x | 18 | 9/3/6 | 1.25x |
| `chacha20-poly1305` | 198 | 70/59/69 | 0.99x | 550 | 296/137/117 | 1.15x |
| `argon2i-small` | 27 | 13/2/12 | 1.03x | 45 | 30/2/13 | 1.36x |
| `x25519` | 18 | 4/13/1 | 1.03x | 50 | 35/13/2 | 1.54x |
| `argon2id-small` | 27 | 13/3/11 | 1.03x | 45 | 30/4/11 | 1.37x |
| `argon2d-small` | 27 | 13/5/9 | 1.04x | 27 | 13/5/9 | 1.04x |
| `mlkem512` | 27 | 13/3/11 | 1.04x | 102 | 85/5/12 | 1.98x |
| `rapidhash-v3-64` | 99 | 30/50/19 | 1.05x | 99 | 30/50/19 | 1.05x |
| `mlkem1024` | 27 | 16/2/9 | 1.05x | 102 | 91/2/9 | 2.19x |
| `rapidhash-v3-128` | 99 | 30/48/21 | 1.07x | 99 | 30/48/21 | 1.07x |
| `blake2b256` | 225 | 119/103/3 | 1.07x | 351 | 244/104/3 | 1.32x |
| `blake2b512` | 198 | 115/81/2 | 1.08x | 297 | 214/81/2 | 1.33x |
| `blake2s256` | 225 | 115/109/1 | 1.09x | 225 | 115/109/1 | 1.09x |
| `mlkem768` | 27 | 16/5/6 | 1.10x | 102 | 91/5/6 | 2.23x |
| `ed25519` | 90 | 32/47/11 | 1.10x | 290 | 202/60/28 | 1.35x |
| `blake2s128` | 198 | 119/79/0 | 1.11x | 198 | 119/79/0 | 1.11x |
| `scrypt-owasp` | 9 | 5/2/2 | 1.11x | 9 | 5/2/2 | 1.11x |
| `rapidhash-64` | 99 | 33/57/9 | 1.13x | 99 | 33/57/9 | 1.13x |
| `rapidhash-128` | 99 | 36/55/8 | 1.14x | 99 | 36/55/8 | 1.14x |
| `xxh3-128` | 99 | 50/34/15 | 1.17x | 99 | 50/34/15 | 1.17x |
| `xxh3-64` | 99 | 53/34/12 | 1.18x | 99 | 53/34/12 | 1.18x |
| `rsa-8192` | 18 | 14/2/2 | 1.19x | 32 | 28/2/2 | 1.25x |
| `scrypt-small` | 36 | 22/14/0 | 1.21x | 36 | 22/14/0 | 1.21x |
| `sha256` | 117 | 43/57/17 | 1.22x | 293 | 161/99/33 | 1.52x |
| `hmac-sha384` | 99 | 37/54/8 | 1.23x | 275 | 166/96/13 | 1.28x |
| `hkdf-sha256` | 36 | 30/6/0 | 1.24x | 100 | 94/6/0 | 1.86x |
| `sha512` | 117 | 45/62/10 | 1.24x | 293 | 170/107/16 | 1.27x |
| `sha384` | 99 | 43/45/11 | 1.24x | 275 | 171/87/17 | 1.28x |
| `pbkdf2-sha256` | 54 | 47/7/0 | 1.24x | 150 | 143/7/0 | 1.64x |
| `hmac-sha512` | 99 | 40/52/7 | 1.25x | 275 | 169/94/12 | 1.30x |
| `pbkdf2-sha512` | 54 | 47/7/0 | 1.26x | 150 | 143/7/0 | 1.33x |
| `ascon-hash256` | 99 | 58/40/1 | 1.27x | 99 | 58/40/1 | 1.27x |
| `hkdf-sha384` | 36 | 36/0/0 | 1.27x | 100 | 100/0/0 | 1.58x |
| `sha512-256` | 99 | 57/42/0 | 1.28x | 99 | 57/42/0 | 1.28x |
| `ecdsa-p384` | 72 | 52/0/20 | 1.31x | 200 | 180/0/20 | 2.96x |
| `ascon-aead128` | 198 | 148/50/0 | 1.33x | 198 | 148/50/0 | 1.33x |
| `hmac-sha256` | 117 | 64/38/15 | 1.35x | 293 | 194/74/25 | 1.70x |
| `ascon-xof128` | 99 | 74/25/0 | 1.36x | 99 | 74/25/0 | 1.36x |
| `aegis-256` | 198 | 91/83/24 | 1.38x | 198 | 91/83/24 | 1.38x |
| `xchacha20-poly1305` | 198 | 176/22/0 | 1.38x | 198 | 176/22/0 | 1.38x |
| `blake3` | 432 | 227/173/32 | 1.40x | 432 | 227/173/32 | 1.40x |
| `ecdsa-p256` | 72 | 62/10/0 | 1.51x | 200 | 186/14/0 | 2.41x |
| `rsa-4096` | 27 | 23/2/2 | 1.57x | 59 | 55/2/2 | 2.50x |
| `crc32c` | 99 | 45/39/15 | 1.61x | 198 | 141/42/15 | 2.22x |
| `crc32` | 99 | 47/30/22 | 1.62x | 198 | 141/32/25 | 2.22x |
| `rsa-3072` | 27 | 23/2/2 | 1.63x | 59 | 55/2/2 | 2.54x |
| `rsa-2048` | 27 | 25/0/2 | 1.68x | 59 | 57/0/2 | 2.60x |
| `kmac256` | 99 | 62/20/17 | 1.73x | 99 | 62/20/17 | 1.73x |
| `aes-128-gcm` | 198 | 117/42/39 | 1.77x | 550 | 431/51/68 | 2.55x |
| `cshake256` | 99 | 69/24/6 | 1.79x | 99 | 69/24/6 | 1.79x |
| `aes-256-gcm` | 198 | 110/41/47 | 1.81x | 550 | 421/52/77 | 2.60x |
| `sha224` | 99 | 52/45/2 | 1.83x | 99 | 52/45/2 | 1.83x |
| `shake128` | 99 | 70/29/0 | 1.83x | 99 | 70/29/0 | 1.83x |
| `shake256` | 99 | 67/31/1 | 1.85x | 99 | 67/31/1 | 1.85x |
| `aes-128-gcm-siv` | 198 | 137/4/57 | 2.02x | 352 | 269/18/65 | 2.79x |
| `sha3-256` | 117 | 102/15/0 | 2.10x | 117 | 102/15/0 | 2.10x |
| `sha3-224` | 99 | 88/11/0 | 2.12x | 99 | 88/11/0 | 2.12x |
| `sha3-384` | 99 | 88/11/0 | 2.15x | 99 | 88/11/0 | 2.15x |
| `aes-256-gcm-siv` | 198 | 138/3/57 | 2.16x | 352 | 291/4/57 | 3.05x |
| `sha3-512` | 99 | 88/11/0 | 2.19x | 99 | 88/11/0 | 2.19x |
| `crc64-nvme` | 99 | 63/35/1 | 2.22x | 99 | 63/35/1 | 2.22x |
| `crc64-xz` | 99 | 75/13/11 | 2.42x | 99 | 75/13/11 | 2.42x |
| `crc24-openpgp` | 99 | 94/2/3 | 13.21x | 99 | 94/2/3 | 13.21x |
| `crc16-ccitt` | 99 | 98/0/1 | 23.10x | 99 | 98/0/1 | 23.10x |
| `crc16-ibm` | 99 | 98/0/1 | 24.06x | 99 | 98/0/1 | 24.06x |

## Linux Worst Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| RISE RISC-V | `xxh3-64 / 0` | `xxhash-rust` | 0.45x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.47x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.48x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 0` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `aes-128-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 1024` | `aws-lc-rs` | 0.49x |
| Intel Ice Lake | `aes-128-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.50x |
| Intel Sapphire Rapids | `aes-128-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.50x |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.50x |
| AMD Zen4 | `ecdsa-p384 / sign / 0` | `aws-lc-rs` | 0.50x |

## Linux Strongest Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt / 262144` | `crc` | 212.12x |
| Intel Sapphire Rapids | `crc16-ibm / 262144` | `crc` | 210.47x |
| Intel Sapphire Rapids | `crc16-ccitt / 16384` | `crc` | 199.21x |
| Intel Sapphire Rapids | `crc16-ibm / 16384` | `crc` | 197.78x |
| Intel Sapphire Rapids | `crc16-ibm / 65536` | `crc` | 180.91x |
| Intel Sapphire Rapids | `crc16-ccitt / 1048576` | `crc` | 179.47x |
| IBM Power10 | `crc16-ibm / 1048576` | `crc` | 177.07x |
| Intel Sapphire Rapids | `crc16-ccitt / 65536` | `crc` | 176.35x |
| IBM Power10 | `crc16-ccitt / 262144` | `crc` | 176.29x |
| IBM Power10 | `crc16-ibm / 262144` | `crc` | 176.01x |
| IBM Power10 | `crc16-ccitt / 1048576` | `crc` | 174.53x |
| Intel Sapphire Rapids | `crc16-ibm / 1048576` | `crc` | 172.31x |

## Top Five Loss Areas

- `mlkem512` / `keygen`: 0.91x geomean across 9 rows; W/T/L 1/2/6; pressure `aws-lc-rs` 6, `libcrux` 2.
- `mlkem1024` / `keygen`: 0.95x geomean across 9 rows; W/T/L 4/1/4; pressure `aws-lc-rs` 3, `libcrux` 2.
- `ecdsa-p384` / `sign`: 0.95x geomean across 36 rows; W/T/L 16/0/20; pressure `aws-lc-rs` 20.
- `argon2id-owasp` / `hash`: 0.96x geomean across 9 rows; W/T/L 2/2/5; pressure `rustcrypto` 7.
- `ed25519` / `verify`: 0.99x geomean across 36 rows; W/T/L 9/18/9; pressure `dalek` 9, `aws-lc-rs` 9, `ring` 9.

## External Pressure

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `rapidhash` | 396 | 129/210/57 | 33% | 1.09x | 1.01x |
| `xxhash-rust` | 198 | 103/68/27 | 52% | 1.17x | 1.06x |
| `aws-lc-rs` | 1,673 | 999/348/326 | 60% | 1.22x | 1.12x |
| `libcrux` | 81 | 71/8/2 | 88% | 1.28x | 1.29x |
| `ascon-hash` | 198 | 132/65/1 | 67% | 1.31x | 1.22x |
| `ascon-aead` | 198 | 148/50/0 | 75% | 1.33x | 1.15x |
| `aegis-crate` | 198 | 91/83/24 | 46% | 1.38x | 1.04x |
| `blake3` | 432 | 227/173/32 | 53% | 1.40x | 1.06x |
| `dalek` | 108 | 84/15/9 | 78% | 1.45x | 1.25x |
| `sha2` | 531 | 288/236/7 | 54% | 1.50x | 1.06x |
| `ring` | 1,656 | 1,305/202/149 | 79% | 1.63x | 1.29x |
| `tiny-keccak` | 396 | 268/104/24 | 68% | 1.80x | 1.99x |
| `dryoc` | 360 | 335/14/11 | 93% | 1.81x | 1.86x |
| `crc-fast` | 297 | 177/97/23 | 60% | 2.03x | 1.17x |
| `rustcrypto` | 2,889 | 2,126/633/130 | 74% | 2.07x | 1.21x |
| `sha3` | 414 | 366/48/0 | 88% | 2.14x | 2.11x |
| `crc32fast` | 99 | 81/6/12 | 82% | 2.37x | 1.98x |
| `crc64fast` | 99 | 75/13/11 | 76% | 2.42x | 2.06x |
| `crc32c` | 99 | 87/6/6 | 88% | 2.74x | 2.19x |
| `fips203` | 81 | 81/0/0 | 100% | 3.32x | 3.23x |
| `rustcrypto-rsa` | 81 | 81/0/0 | 100% | 5.64x | 4.62x |
| `crc` | 297 | 290/2/5 | 98% | 19.43x | 29.80x |

## macOS Local Snapshot

The macOS Apple Silicon run is local evidence for the same commit. It is useful for Apple Silicon planning but is not folded into Linux CI release claims. The ML-KEM row reflects the newer local `just bench-quick ml-kem` public API run from this workspace.

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| macOS local: all matched performance pairs | 1,309 | 828/435/46 | 63% | 1.81x | 1.15x |
| macOS local: fastest external per case | 786 | 386/362/38 | 49% | 1.37x | 1.05x |
| macOS local: ML-KEM fastest external | 9 | 7/2/0 | 78% | 1.46x | 1.43x |

## README Numbers

- **Headline:** 4,089 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,081 are wins or ties. Linux CI geomean is 1.59x.
- **Checksums:** 5.20x geomean across 693 Linux CI fastest-external rows; W/T/L 520/119/54.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,726 Linux CI fastest-external rows; W/T/L 2,074/1,435/217.
- **Auth/KDF:** 1.25x geomean across 180 Linux CI fastest-external rows; W/T/L 160/20/0.
- **Password hashing:** 1.08x geomean across 135 Linux CI fastest-external rows; W/T/L 68/28/39.
- **Public-key:** 1.21x geomean across 333 Linux CI fastest-external rows; W/T/L 195/80/58.
- **RSA:** 1.53x geomean across 99 Linux CI fastest-external rows; W/T/L 85/6/8.
- **AEAD:** 1.56x geomean across 1,584 Linux CI fastest-external rows; W/T/L 987/304/293.
- **ML-KEM:** 1.06x geomean across 81 Linux CI fastest-external rows; W/T/L 45/10/26.
- **ECDSA P-256/P-384:** 1.41x Linux CI geomean across 144 fastest-external rows.
- **Current top losses:** `mlkem512` / `keygen`: 0.91x geomean across 9 rows; W/T/L 1/2/6; pressure `aws-lc-rs` 6, `libcrux` 2; `mlkem1024` / `keygen`: 0.95x geomean across 9 rows; W/T/L 4/1/4; pressure `aws-lc-rs` 3, `libcrux` 2; `ecdsa-p384` / `sign`: 0.95x geomean across 36 rows; W/T/L 16/0/20; pressure `aws-lc-rs` 20; `argon2id-owasp` / `hash`: 0.96x geomean across 9 rows; W/T/L 2/2/5; pressure `rustcrypto` 7; `ed25519` / `verify`: 0.99x geomean across 36 rows; W/T/L 9/18/9; pressure `dalek` 9, `aws-lc-rs` 9, `ring` 9.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-06-22 18_11_11` | 2,244 | `benchmark_results/2026-06-22/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-06-22 18_11_11` | 2,244 | `benchmark_results/2026-06-22/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-06-22 18_11_11` | 2,250 | `benchmark_results/2026-06-22/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-06-22 18_11_11` | 2,250 | `benchmark_results/2026-06-22/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-06-22 18_11_11` | 1,995 | `benchmark_results/2026-06-22/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-06-22 18_11_11` | 1,995 | `benchmark_results/2026-06-22/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-06-22 18_11_11` | 2,244 | `benchmark_results/2026-06-22/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-06-22 18_11_11` | 2,244 | `benchmark_results/2026-06-22/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-06-22 18_11_11` | 2,244 | `benchmark_results/2026-06-22/linux/rise-riscv/results.txt` |
| macOS Apple Silicon | `local` | `2026-06-22 15_11_40` | 2,277 | `benchmark_results/2026-06-22/macos/aarch64/results.txt` |
