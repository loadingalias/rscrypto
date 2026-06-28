# Benchmark Overview

Sources:

- Linux benchmark CI run [#28276488752](https://github.com/loadingalias/rscrypto/actions/runs/28276488752), created 2026-06-27 02:54:44 UTC.
- Linux commit: `c31d40564a9723c9301369f7a7a88c7bb70dbf69`.
- Linux artifacts: nine successful `benchmark-*` artifacts extracted into `benchmark_results/2026-06-27/linux/*/results.txt`.
- Local macOS run: `benchmark_results/2026-06-22/macos/aarch64/results.txt` at commit `b978c2ca45611325850d7f1af94718e497acde50`.
- Local macOS ML-KEM follow-up: `benchmark_results/2026-06-23/macos/aarch64/results.txt` at commit `b978c2ca45611325850d7f1af94718e497acde50`.

Scope: the 2026-06-27 nine-runner Linux CI benchmark matrix for commit `c31d405`. Ratios are `external_crate_time / rscrypto_time`; higher is better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and input shape. Internal kernel, scratch-buffer, padding-only, cold-path, PHC roundtrip, parallel-scaling, threshold-selection, public-overhead, and phase-attribution microbenches are parsed as raw rows but excluded from external win/loss claims. The macOS local run is listed separately and is not mixed into Linux CI claims.

Coverage note: this is a full Linux CI public benchmark pass. It includes checksum, hash, XOF, MAC, KDF, password-hashing, BLAKE2/BLAKE3, RSA import/verification, ECDSA P-256/P-384 signing and verification, Ed25519, X25519, AEAD, and ML-KEM-512/768/1024 keygen, encapsulation, and decapsulation rows. ML-KEM phase/arithmetic microbenches are present in the raw artifacts and intentionally excluded from release-level competitor claims.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Linux CI: all matched performance pairs | 10,781 | 7,537/2,412/832 | 70% | 1.82x | 1.23x |
| Linux CI: fastest external per case | 6,750 | 4,103/1,998/649 | 61% | 1.60x | 1.11x |

Shareable release summary:

- **Headline:** 4,103 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,101 are wins or ties. Linux CI fastest-external geomean is 1.60x.
- **Checksums:** 5.19x geomean across 693 fastest-external rows; W/T/L is 515/122/56.
- **Hashes/MACs/XOFs:** 1.36x geomean across 3,726 fastest-external rows; W/T/L is 2,057/1,457/212.
- **Auth/KDF:** 1.25x geomean across 180 fastest-external rows; W/T/L is 161/18/1.
- **Password hashing:** 1.07x geomean across 135 fastest-external rows; W/T/L is 68/30/37.
- **Public-key:** 1.36x geomean across 333 fastest-external rows; W/T/L is 225/72/36.
- **RSA:** 1.55x geomean across 99 fastest-external rows; W/T/L is 89/2/8.
- **AEAD:** 1.56x geomean across 1,584 fastest-external rows; W/T/L is 988/297/299.
- **ML-KEM:** 1.44x geomean across 81 fastest-external rows; W/T/L is 58/4/19.
- **ECDSA P-256/P-384:** Linux CI 1.56x geomean across 144 fastest-external rows; W/T/L is 131/9/4.
- **Top current loss areas:** `mlkem1024` / `keygen`: 0.84x geomean across 9 rows; W/T/L 1/0/8; pressure `aws-lc-rs` 4, `libcrux` 4; `mlkem768` / `keygen`: 0.94x geomean across 9 rows; W/T/L 1/1/7; pressure `aws-lc-rs` 4, `libcrux` 4; `argon2id-owasp` / `hash`: 0.96x geomean across 9 rows; W/T/L 3/1/5; pressure `rustcrypto` 6; `chacha20-poly1305` / `encrypt`: 0.98x geomean across 99 rows; W/T/L 33/32/34; pressure `ring` 31, `aws-lc-rs` 12; `ed25519` / `verify`: 0.99x geomean across 36 rows; W/T/L 9/18/9; pressure `ring` 7, `dalek` 5, `aws-lc-rs` 3.

## Coverage Matrix

| Platform | Raw Criterion rows | All pairs | Fastest rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 2,316 | 1,251 | 750 | 534/155/61 | 71% | 1.49x | 1.15x |
| AMD Zen5 | 2,316 | 1,251 | 750 | 443/241/66 | 59% | 1.51x | 1.10x |
| AWS Graviton3 | 2,320 | 1,251 | 750 | 364/297/89 | 49% | 1.37x | 1.04x |
| AWS Graviton4 | 2,320 | 1,251 | 750 | 366/334/50 | 49% | 1.39x | 1.05x |
| IBM Power10 | 2,067 | 1,012 | 750 | 388/317/45 | 52% | 1.88x | 1.06x |
| IBM z16/s390x | 2,067 | 1,012 | 750 | 605/96/49 | 81% | 3.24x | 2.37x |
| Intel Ice Lake | 2,316 | 1,251 | 750 | 532/139/79 | 71% | 1.49x | 1.20x |
| Intel Sapphire Rapids | 2,316 | 1,251 | 750 | 531/150/69 | 71% | 1.64x | 1.23x |
| RISE RISC-V | 2,316 | 1,251 | 750 | 340/269/141 | 45% | 1.09x | 1.03x |

## Category Summary

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 693 | 515/122/56 | 74% | 5.19x | 2.41x |
| Hashes/MACs/XOFs | 3,726 | 2,057/1,457/212 | 55% | 1.36x | 1.07x |
| Auth/KDF | 180 | 161/18/1 | 89% | 1.25x | 1.13x |
| Password hashing | 135 | 68/30/37 | 50% | 1.07x | 1.05x |
| Public-key | 333 | 225/72/36 | 68% | 1.36x | 1.23x |
| RSA | 99 | 89/2/8 | 90% | 1.55x | 1.17x |
| AEAD | 1,584 | 988/297/299 | 62% | 1.56x | 1.14x |

## BLAKE3 Summary

BLAKE3 rows come from Linux CI run [#28276488752](https://github.com/loadingalias/rscrypto/actions/runs/28276488752). All-pair and fastest-external BLAKE3 metrics are identical because official `blake3` is the only external implementation in this bench.

| Scope | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| All Linux BLAKE3 rows | 432 | 227/179/26 | 1.40x | 1.06x |
| x86_64 | 192 | 86/91/15 | 1.24x | 1.03x |
| AArch64 | 96 | 44/47/5 | 1.44x | 1.05x |

| Platform | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| AMD Zen4 | 48 | 21/24/3 | 1.31x | 1.02x |
| AMD Zen5 | 48 | 21/22/5 | 1.31x | 1.03x |
| AWS Graviton3 | 48 | 22/22/4 | 1.41x | 1.00x |
| AWS Graviton4 | 48 | 22/25/1 | 1.48x | 1.05x |
| IBM Power10 | 48 | 39/9/0 | 2.00x | 1.80x |
| IBM z16/s390x | 48 | 36/7/5 | 1.85x | 1.90x |
| Intel Ice Lake | 48 | 19/26/3 | 1.17x | 1.02x |
| Intel Sapphire Rapids | 48 | 25/19/4 | 1.18x | 1.05x |
| RISE RISC-V | 48 | 22/25/1 | 1.16x | 1.00x |

| Operation | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `oneshot` | 99 | 45/44/10 | 1.31x | 1.03x |
| `keyed` | 99 | 39/51/9 | 1.32x | 1.01x |
| `derive-key` | 99 | 78/21/0 | 1.78x | 1.86x |
| `streaming` | 36 | 11/24/1 | 1.18x | 1.01x |
| `xof` | 99 | 54/39/6 | 1.34x | 1.07x |

## ML-KEM Summary

ML-KEM public coverage is complete for the CI-selected primitive set: ML-KEM-512, ML-KEM-768, and ML-KEM-1024 each include keygen, encapsulate, and decapsulate on all nine Linux platforms. POWER10 and s390x do not have `aws-lc-rs` ML-KEM rows in this artifact set, but still have rscrypto plus `libcrux`, `fips203`, and RustCrypto comparison rows for every public operation.

| Platform | Raw ML-KEM rows | Fastest rows | W/T/L | Geomean | Median | Fastest external split |
| --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 45 | 9 | 7/0/2 | 1.69x | 1.70x | `libcrux` 7, `aws-lc-rs` 2 |
| AMD Zen5 | 45 | 9 | 7/0/2 | 1.79x | 1.74x | `libcrux` 9 |
| AWS Graviton3 | 45 | 9 | 5/0/4 | 1.09x | 1.14x | `aws-lc-rs` 9 |
| AWS Graviton4 | 45 | 9 | 5/0/4 | 1.09x | 1.14x | `aws-lc-rs` 9 |
| IBM Power10 | 36 | 9 | 6/2/1 | 1.40x | 1.58x | `libcrux` 9 |
| IBM z16/s390x | 36 | 9 | 9/0/0 | 1.72x | 1.70x | `libcrux` 8, `fips203` 1 |
| Intel Ice Lake | 45 | 9 | 7/0/2 | 1.65x | 1.72x | `libcrux` 7, `aws-lc-rs` 2 |
| Intel Sapphire Rapids | 45 | 9 | 7/0/2 | 1.72x | 1.87x | `aws-lc-rs` 6, `libcrux` 3 |
| RISE RISC-V | 45 | 9 | 5/2/2 | 1.11x | 1.05x | `aws-lc-rs` 9 |

| Primitive/op | Rows | W/T/L | Win % | Geomean | Median | Pressure |
| --- | --- | --- | --- | --- | --- | --- |
| `mlkem1024` / `decapsulate` | 9 | 9/0/0 | 100% | 1.63x | 1.85x | none |
| `mlkem1024` / `encapsulate` | 9 | 9/0/0 | 100% | 2.38x | 2.15x | none |
| `mlkem1024` / `keygen` | 9 | 1/0/8 | 11% | 0.84x | 0.78x | `aws-lc-rs` 4, `libcrux` 4 |
| `mlkem512` / `decapsulate` | 9 | 6/1/2 | 67% | 1.32x | 1.51x | `aws-lc-rs` 2 |
| `mlkem512` / `encapsulate` | 9 | 9/0/0 | 100% | 1.88x | 1.73x | none |
| `mlkem512` / `keygen` | 9 | 5/2/2 | 56% | 1.09x | 1.17x | `aws-lc-rs` 3 |
| `mlkem768` / `decapsulate` | 9 | 9/0/0 | 100% | 1.49x | 1.70x | none |
| `mlkem768` / `encapsulate` | 9 | 9/0/0 | 100% | 2.24x | 1.77x | none |
| `mlkem768` / `keygen` | 9 | 1/1/7 | 11% | 0.94x | 0.89x | `aws-lc-rs` 4, `libcrux` 4 |

## ECDSA Summary

ECDSA signing includes both deterministic and blinded rscrypto rows in raw results; aggregate fastest-external comparisons use the fastest rscrypto row for the exact case. Constant-time release evidence is tracked separately by `ct.toml` and CT workflow artifacts.

| Operation | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `ecdsa-p256` / `sign` | 36 | 36/0/0 | 1.45x | 1.35x |
| `ecdsa-p256` / `verify` | 36 | 27/9/0 | 1.59x | 1.15x |
| `ecdsa-p384` / `sign` | 36 | 32/0/4 | 1.42x | 1.30x |
| `ecdsa-p384` / `verify` | 36 | 36/0/0 | 1.80x | 1.39x |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are strongest-external comparisons; all-pair columns include every matched external implementation.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `argon2id-owasp` | 9 | 3/1/5 | 0.96x | 18 | 10/2/6 | 1.26x |
| `chacha20-poly1305` | 198 | 67/62/69 | 0.99x | 550 | 288/143/119 | 1.15x |
| `argon2i-small` | 27 | 12/3/12 | 1.01x | 45 | 29/3/13 | 1.33x |
| `x25519` | 18 | 3/15/0 | 1.03x | 50 | 34/16/0 | 1.55x |
| `argon2id-small` | 27 | 13/4/10 | 1.04x | 45 | 30/5/10 | 1.38x |
| `argon2d-small` | 27 | 13/6/8 | 1.04x | 27 | 13/6/8 | 1.04x |
| `rapidhash-v3-64` | 99 | 26/50/23 | 1.05x | 99 | 26/50/23 | 1.05x |
| `blake2b256` | 225 | 116/103/6 | 1.06x | 351 | 231/114/6 | 1.31x |
| `rapidhash-v3-128` | 99 | 36/39/24 | 1.07x | 99 | 36/39/24 | 1.07x |
| `blake2b512` | 198 | 118/76/4 | 1.08x | 297 | 210/82/5 | 1.32x |
| `blake2s256` | 225 | 107/118/0 | 1.09x | 225 | 107/118/0 | 1.09x |
| `ed25519` | 90 | 33/44/13 | 1.10x | 290 | 208/54/28 | 1.36x |
| `scrypt-owasp` | 9 | 5/2/2 | 1.11x | 9 | 5/2/2 | 1.11x |
| `blake2s128` | 198 | 120/76/2 | 1.11x | 198 | 120/76/2 | 1.11x |
| `rapidhash-128` | 99 | 36/59/4 | 1.14x | 99 | 36/59/4 | 1.14x |
| `rapidhash-64` | 99 | 31/64/4 | 1.15x | 99 | 31/64/4 | 1.15x |
| `xxh3-128` | 99 | 52/32/15 | 1.16x | 99 | 52/32/15 | 1.16x |
| `xxh3-64` | 99 | 54/33/12 | 1.18x | 99 | 54/33/12 | 1.18x |
| `scrypt-small` | 36 | 22/14/0 | 1.19x | 36 | 22/14/0 | 1.19x |
| `rsa-8192` | 18 | 14/2/2 | 1.20x | 32 | 28/2/2 | 1.26x |
| `hmac-sha512` | 99 | 38/54/7 | 1.23x | 275 | 163/100/12 | 1.28x |
| `pbkdf2-sha512` | 54 | 46/7/1 | 1.23x | 150 | 142/7/1 | 1.31x |
| `hmac-sha384` | 99 | 41/51/7 | 1.23x | 275 | 169/94/12 | 1.29x |
| `hkdf-sha256` | 36 | 31/5/0 | 1.25x | 100 | 95/5/0 | 1.88x |
| `sha512` | 117 | 43/65/9 | 1.25x | 293 | 169/109/15 | 1.28x |
| `sha256` | 117 | 39/63/15 | 1.25x | 293 | 159/106/28 | 1.55x |
| `pbkdf2-sha256` | 54 | 48/6/0 | 1.26x | 150 | 144/6/0 | 1.65x |
| `hkdf-sha384` | 36 | 36/0/0 | 1.26x | 100 | 100/0/0 | 1.57x |
| `sha384` | 99 | 39/50/10 | 1.26x | 275 | 166/94/15 | 1.29x |
| `ascon-hash256` | 99 | 59/39/1 | 1.27x | 99 | 59/39/1 | 1.27x |
| `sha512-256` | 99 | 55/44/0 | 1.31x | 99 | 55/44/0 | 1.31x |
| `ascon-aead128` | 198 | 157/41/0 | 1.34x | 198 | 157/41/0 | 1.34x |
| `ascon-xof128` | 99 | 75/23/1 | 1.35x | 99 | 75/23/1 | 1.35x |
| `aegis-256` | 198 | 88/72/38 | 1.36x | 198 | 88/72/38 | 1.36x |
| `hmac-sha256` | 117 | 65/38/14 | 1.37x | 293 | 195/74/24 | 1.72x |
| `mlkem512` | 27 | 20/3/4 | 1.39x | 102 | 95/3/4 | 2.66x |
| `blake3` | 432 | 227/179/26 | 1.40x | 432 | 227/179/26 | 1.40x |
| `xchacha20-poly1305` | 198 | 172/26/0 | 1.41x | 198 | 172/26/0 | 1.41x |
| `mlkem768` | 27 | 19/1/7 | 1.46x | 102 | 91/4/7 | 2.98x |
| `mlkem1024` | 27 | 19/0/8 | 1.48x | 102 | 91/0/11 | 3.08x |
| `ecdsa-p256` | 72 | 63/9/0 | 1.52x | 200 | 187/13/0 | 2.41x |
| `rsa-4096` | 27 | 25/0/2 | 1.56x | 59 | 57/0/2 | 2.50x |
| `ecdsa-p384` | 72 | 68/0/4 | 1.60x | 200 | 196/0/4 | 3.65x |
| `crc32c` | 99 | 47/40/12 | 1.61x | 198 | 143/43/12 | 2.20x |
| `crc32` | 99 | 48/33/18 | 1.64x | 198 | 142/37/19 | 2.25x |
| `rsa-3072` | 27 | 25/0/2 | 1.66x | 59 | 57/0/2 | 2.57x |
| `rsa-2048` | 27 | 25/0/2 | 1.71x | 59 | 57/0/2 | 2.63x |
| `kmac256` | 99 | 63/20/16 | 1.75x | 99 | 63/20/16 | 1.75x |
| `aes-128-gcm` | 198 | 119/41/38 | 1.78x | 550 | 435/50/65 | 2.57x |
| `aes-256-gcm` | 198 | 110/45/43 | 1.80x | 550 | 421/54/75 | 2.60x |
| `cshake256` | 99 | 61/29/9 | 1.80x | 99 | 61/29/9 | 1.80x |
| `sha224` | 99 | 51/46/2 | 1.87x | 99 | 51/46/2 | 1.87x |
| `shake128` | 99 | 71/28/0 | 1.88x | 99 | 71/28/0 | 1.88x |
| `shake256` | 99 | 69/29/1 | 1.89x | 99 | 69/29/1 | 1.89x |
| `aes-128-gcm-siv` | 198 | 137/6/55 | 2.04x | 352 | 269/20/63 | 2.81x |
| `sha3-256` | 117 | 101/16/0 | 2.14x | 117 | 101/16/0 | 2.14x |
| `sha3-224` | 99 | 88/11/0 | 2.15x | 99 | 88/11/0 | 2.15x |
| `aes-256-gcm-siv` | 198 | 138/4/56 | 2.15x | 352 | 291/5/56 | 3.03x |
| `sha3-384` | 99 | 88/11/0 | 2.18x | 99 | 88/11/0 | 2.18x |
| `crc64-nvme` | 99 | 56/35/8 | 2.18x | 99 | 56/35/8 | 2.18x |
| `sha3-512` | 99 | 88/11/0 | 2.22x | 99 | 88/11/0 | 2.22x |
| `crc64-xz` | 99 | 74/13/12 | 2.41x | 99 | 74/13/12 | 2.41x |
| `crc24-openpgp` | 99 | 94/1/4 | 13.20x | 99 | 94/1/4 | 13.20x |
| `crc16-ccitt` | 99 | 98/0/1 | 23.34x | 99 | 98/0/1 | 23.34x |
| `crc16-ibm` | 99 | 98/0/1 | 23.82x | 99 | 98/0/1 | 23.82x |

## Linux Worst Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| RISE RISC-V | `xxh3-64 / 0` | `xxhash-rust` | 0.36x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.47x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.48x |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.48x |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.49x |
| Intel Ice Lake | `aes-128-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `aes-128-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `aes-128-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.50x |
| Intel Ice Lake | `aes-128-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.51x |
| RISE RISC-V | `crc64-xz / 1048576` | `crc64fast` | 0.51x |
| Intel Ice Lake | `xxh3-64 / 64` | `xxhash-rust` | 0.51x |
| AMD Zen4 | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.51x |

## Linux Strongest Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ibm / 262144` | `crc` | 241.31x |
| Intel Sapphire Rapids | `crc16-ccitt / 262144` | `crc` | 240.50x |
| Intel Sapphire Rapids | `crc16-ccitt / 16384` | `crc` | 233.26x |
| Intel Sapphire Rapids | `crc16-ibm / 16384` | `crc` | 222.91x |
| Intel Sapphire Rapids | `crc16-ibm / 1048576` | `crc` | 212.46x |
| Intel Sapphire Rapids | `crc16-ccitt / 1048576` | `crc` | 206.30x |
| Intel Sapphire Rapids | `crc16-ibm / 65536` | `crc` | 204.30x |
| Intel Sapphire Rapids | `crc16-ccitt / 65536` | `crc` | 197.26x |
| Intel Sapphire Rapids | `crc16-ibm / 4096` | `crc` | 190.21x |
| Intel Sapphire Rapids | `crc16-ccitt / 4096` | `crc` | 189.94x |
| IBM Power10 | `crc16-ibm / 1048576` | `crc` | 177.67x |
| IBM Power10 | `crc16-ccitt / 1048576` | `crc` | 176.43x |

## Top Five Loss Areas

- `mlkem1024` / `keygen`: 0.84x geomean across 9 rows; W/T/L 1/0/8; pressure `aws-lc-rs` 4, `libcrux` 4.
- `mlkem768` / `keygen`: 0.94x geomean across 9 rows; W/T/L 1/1/7; pressure `aws-lc-rs` 4, `libcrux` 4.
- `argon2id-owasp` / `hash`: 0.96x geomean across 9 rows; W/T/L 3/1/5; pressure `rustcrypto` 6.
- `chacha20-poly1305` / `encrypt`: 0.98x geomean across 99 rows; W/T/L 33/32/34; pressure `ring` 31, `aws-lc-rs` 12.
- `ed25519` / `verify`: 0.99x geomean across 36 rows; W/T/L 9/18/9; pressure `ring` 7, `dalek` 5, `aws-lc-rs` 3.

## External Pressure

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `rapidhash` | 396 | 129/212/55 | 33% | 1.10x | 1.01x |
| `xxhash-rust` | 198 | 106/65/27 | 54% | 1.17x | 1.08x |
| `aws-lc-rs` | 1,673 | 1,023/356/294 | 61% | 1.24x | 1.14x |
| `ascon-hash` | 198 | 134/62/2 | 68% | 1.31x | 1.23x |
| `ascon-aead` | 198 | 157/41/0 | 79% | 1.34x | 1.16x |
| `aegis-crate` | 198 | 88/72/38 | 44% | 1.36x | 1.02x |
| `blake3` | 432 | 227/179/26 | 53% | 1.40x | 1.06x |
| `dalek` | 108 | 85/16/7 | 79% | 1.46x | 1.20x |
| `sha2` | 531 | 283/244/4 | 53% | 1.53x | 1.06x |
| `ring` | 1,656 | 1,305/203/148 | 79% | 1.65x | 1.30x |
| `libcrux` | 81 | 70/3/8 | 86% | 1.74x | 1.72x |
| `dryoc` | 360 | 320/29/11 | 89% | 1.80x | 1.84x |
| `tiny-keccak` | 396 | 264/106/26 | 67% | 1.83x | 2.00x |
| `crc-fast` | 297 | 167/107/23 | 56% | 2.02x | 1.17x |
| `rustcrypto` | 2,889 | 2,114/646/129 | 73% | 2.10x | 1.21x |
| `sha3` | 414 | 365/49/0 | 88% | 2.17x | 2.13x |
| `crc32fast` | 99 | 83/4/12 | 84% | 2.40x | 2.03x |
| `crc64fast` | 99 | 74/13/12 | 75% | 2.41x | 2.01x |
| `crc32c` | 99 | 91/4/4 | 92% | 2.70x | 2.10x |
| `fips203` | 81 | 81/0/0 | 100% | 4.50x | 4.74x |
| `rustcrypto-rsa` | 81 | 81/0/0 | 100% | 5.70x | 4.78x |
| `crc` | 297 | 290/1/6 | 98% | 19.43x | 28.80x |

## macOS Local Snapshot

The macOS Apple Silicon run is local evidence for the earlier 2026-06-22 baseline commit. It is useful for Apple Silicon planning but is not folded into Linux CI release claims. The ML-KEM row uses the 2026-06-23 local `just bench-quick ml-kem` public API follow-up from the same commit.

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| macOS local: all matched performance pairs | 1,309 | 828/435/46 | 63% | 1.81x | 1.15x |
| macOS local: fastest external per case | 786 | 386/362/38 | 49% | 1.37x | 1.05x |
| macOS local: ML-KEM fastest external | 9 | 7/2/0 | 78% | 1.46x | 1.43x |

## README Numbers

- **Headline:** 4,103 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,101 are wins or ties. Linux CI geomean is 1.60x.
- **Checksums:** 5.19x geomean across 693 Linux CI fastest-external rows; W/T/L 515/122/56.
- **Hashes/MACs/XOFs:** 1.36x geomean across 3,726 Linux CI fastest-external rows; W/T/L 2,057/1,457/212.
- **Auth/KDF:** 1.25x geomean across 180 Linux CI fastest-external rows; W/T/L 161/18/1.
- **Password hashing:** 1.07x geomean across 135 Linux CI fastest-external rows; W/T/L 68/30/37.
- **Public-key:** 1.36x geomean across 333 Linux CI fastest-external rows; W/T/L 225/72/36.
- **RSA:** 1.55x geomean across 99 Linux CI fastest-external rows; W/T/L 89/2/8.
- **AEAD:** 1.56x geomean across 1,584 Linux CI fastest-external rows; W/T/L 988/297/299.
- **ML-KEM:** 1.44x geomean across 81 Linux CI fastest-external rows; W/T/L 58/4/19.
- **ECDSA P-256/P-384:** 1.56x Linux CI geomean across 144 fastest-external rows; W/T/L 131/9/4.
- **Current top losses:** `mlkem1024` / `keygen`: 0.84x geomean across 9 rows; W/T/L 1/0/8; pressure `aws-lc-rs` 4, `libcrux` 4; `mlkem768` / `keygen`: 0.94x geomean across 9 rows; W/T/L 1/1/7; pressure `aws-lc-rs` 4, `libcrux` 4; `argon2id-owasp` / `hash`: 0.96x geomean across 9 rows; W/T/L 3/1/5; pressure `rustcrypto` 6; `chacha20-poly1305` / `encrypt`: 0.98x geomean across 99 rows; W/T/L 33/32/34; pressure `ring` 31, `aws-lc-rs` 12; `ed25519` / `verify`: 0.99x geomean across 36 rows; W/T/L 9/18/9; pressure `ring` 7, `dalek` 5, `aws-lc-rs` 3.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-06-27 02_54_44` | 2,316 | `benchmark_results/2026-06-27/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-06-27 02_54_44` | 2,316 | `benchmark_results/2026-06-27/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-06-27 02_54_44` | 2,320 | `benchmark_results/2026-06-27/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-06-27 02_54_44` | 2,320 | `benchmark_results/2026-06-27/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-06-27 02_54_44` | 2,067 | `benchmark_results/2026-06-27/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-06-27 02_54_44` | 2,067 | `benchmark_results/2026-06-27/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-06-27 02_54_44` | 2,316 | `benchmark_results/2026-06-27/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-06-27 02_54_44` | 2,316 | `benchmark_results/2026-06-27/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-06-27 02_54_44` | 2,316 | `benchmark_results/2026-06-27/linux/rise-riscv/results.txt` |
| macOS Apple Silicon | `local` | `2026-06-22 15_11_40` | 2,277 | `benchmark_results/2026-06-22/macos/aarch64/results.txt` |
| macOS Apple Silicon ML-KEM follow-up | `local` | `2026-06-23 14_37_24` | 97 | `benchmark_results/2026-06-23/macos/aarch64/results.txt` |
