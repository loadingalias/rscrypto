# Benchmark Overview

Sources:

- Linux benchmark CI run [#27227999222](https://github.com/loadingalias/rscrypto/actions/runs/27227999222), created 2026-06-09 18:43:21 UTC.
- Linux commit: `7dbf097a8db537fc85943f5ae6b2fd2dcc06342b`.
- Apple Silicon local run: `benchmark_results/2026-06-09/macos/aarch64/results.txt`, created 2026-06-09 14:02:43 local time.
- Apple Silicon commit: `7dbf097a8db537fc85943f5ae6b2fd2dcc06342b`.
- Previous Linux comparison baseline: run [#27164564672](https://github.com/loadingalias/rscrypto/actions/runs/27164564672), created 2026-06-08 20:21:15 UTC.
- Previous Linux commit: `8a0ac692ab68f09b47ca65b4cb6ecca7fc333f31`.
- Previous Apple Silicon local comparison baseline: `benchmark_results/2026-06-01/macos/aarch64/results.txt`, created 2026-06-01 11:45:23 local time on an MBP M1.

Scope: the 2026-06-09 nine-runner Linux CI benchmark matrix and the 2026-06-09 local Apple Silicon full run for commit `7dbf097`. Ratios are `external_crate_time / rscrypto_time`; higher is always better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and size. Internal kernel, scratch-buffer, padding-only, cold-path, PHC roundtrip, parallel-scaling, and threshold-selection microbenches are parsed as raw rows but excluded from external win/loss claims.

Coverage note: the current Linux and Apple Silicon passes are both full benchmark runs. They include checksum, hash, XOF, MAC, KDF, password-hashing, RSA import/verification, Ed25519, X25519, and AEAD rows, including Argon2, scrypt, Ascon-AEAD, BLAKE3, and CRC-32C. Apple Silicon is a separate local reference and is not combined into Linux aggregate totals.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Linux CI: all matched performance pairs | 10,174 | 7024/2328/822 | 69% | 1.76x | 1.20x |
| Linux CI: fastest external per case | 6,525 | 3958/1920/647 | 61% | 1.60x | 1.11x |
| Apple Silicon: all matched performance pairs | 1,200 | 774/392/34 | 64% | 1.82x | 1.14x |
| Apple Silicon: fastest external per case | 725 | 373/320/32 | 51% | 1.41x | 1.05x |
| Previous Linux CI: fastest external per case | 6,525 | 3908/1920/697 | 60% | 1.58x | 1.10x |

Shareable release summary:

- **Headline:** 3,958 of 6,525 matched Linux CI fastest-external comparisons are wins; 5,878 are wins or ties. Linux CI fastest-external geomean is 1.60x.
- **Apple Silicon headline:** 373 of 725 matched fastest-external comparisons are wins; 693 are wins or ties. Apple Silicon fastest-external geomean is 1.41x.
- **Checksums:** 2.52x geomean across 1,287 fastest-external rows; W/T/L is 744/400/143.
- **Hashes/MACs/XOFs:** 1.41x geomean across 3,132 fastest-external rows; W/T/L is 1870/1134/128.
- **Auth/KDF:** 1.24x geomean across 180 fastest-external rows; W/T/L is 161/19/0.
- **Password hashing:** 0.96x geomean across 135 fastest-external rows; W/T/L is 64/14/57.
- **Public-key:** 1.10x geomean across 108 fastest-external rows; W/T/L is 41/54/13.
- **RSA:** 1.55x geomean across 99 fastest-external rows; W/T/L is 85/6/8.
- **AEAD:** 1.56x geomean across 1,584 fastest-external rows; W/T/L is 993/293/298.
- **Apple Silicon AEAD:** 1.48x geomean across 176 fastest-external rows. ChaCha20-Poly1305 is near parity at 1.02x encrypt and 1.03x decrypt; the only AEAD fastest-external loss is one AES-256-GCM decrypt row.
- **Top current loss areas:** Linux scrypt/password hashing, Linux ChaCha20-Poly1305 rows against `ring`/`aws-lc-rs`, near-parity Ed25519/X25519 operations, RISE RISC-V checksum point losses, and Apple Silicon SHA3/XXH3 plus localized BLAKE3/CRC32(C) rows.

## What Changed Since 2026-06-08

- Fastest-external coverage is unchanged at 6,525 rows. This pass uses the full row set, including BLAKE3 and CRC-32C.
- Headline fastest-external geomean moved from 1.58x to 1.60x; wins rose from 3,908 to 3,958, and losses fell from 697 to 647.
- Linux and Apple Silicon evidence now point at the same validated constant-time-secure commit, `7dbf097`; Apple remains a separate local reference.

| Scope | Previous | Current | Delta |
| --- | --- | --- | --- |
| Fastest rows | 6,525 | 6,525 | 0 |
| Wins | 3,908 | 3,958 | +50 |
| Wins or ties | 5,828 | 5,878 | +50 |
| Losses | 697 | 647 | -50 |
| Geomean | 1.58x | 1.60x | +0.01x |
| Median | 1.10x | 1.11x | +0.01x |

Largest apples-to-apples drops among cases that existed in both Linux runs:

These are ratio movements, not necessarily current losses. Several rows below are still dominant rscrypto wins, but their margin narrowed.

| Platform | Case | Fastest external | Previous | Current | Delta |
| --- | --- | --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ibm` / `hash` / `262144` | `crc` | 215.26x | 211.21x | -4.05x |
| Intel Ice Lake | `crc16-ibm` / `hash` / `1048576` | `crc` | 118.58x | 114.76x | -3.82x |
| IBM z16/s390x | `blake3` / `keyed` / `262144` | `blake3` | 8.43x | 4.91x | -3.52x |
| IBM z16/s390x | `blake3` / `hash` / `262144` | `blake3` | 8.68x | 5.53x | -3.15x |
| IBM z16/s390x | `aegis-256` / `decrypt` / `4096` | `aegis-crate` | 9.70x | 6.81x | -2.89x |
| IBM Power10 | `crc16-ccitt` / `hash` / `4096` | `crc` | 104.22x | 101.50x | -2.73x |
| IBM z16/s390x | `aegis-256` / `decrypt` / `64` | `aegis-crate` | 10.22x | 7.52x | -2.70x |
| Intel Sapphire Rapids | `crc16-ibm` / `hash` / `1048576` | `crc` | 186.82x | 184.18x | -2.65x |
| AMD Zen4 | `crc16-ibm` / `hash` / `16384` | `crc` | 126.52x | 124.10x | -2.42x |
| AWS Graviton3 | `crc16-ibm` / `hash` / `1048576` | `crc` | 123.31x | 120.95x | -2.36x |

Largest apples-to-apples gains among cases that existed in both Linux runs:

| Platform | Case | Fastest external | Previous | Current | Delta |
| --- | --- | --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt` / `hash` / `4096` | `crc` | 168.86x | 183.86x | +15.00x |
| Intel Ice Lake | `crc16-ccitt` / `hash` / `1024` | `crc` | 64.59x | 74.86x | +10.27x |
| AWS Graviton3 | `crc16-ccitt` / `hash` / `1048576` | `crc` | 118.59x | 124.31x | +5.72x |
| IBM z16/s390x | `aegis-256` / `decrypt` / `65536` | `aegis-crate` | 9.52x | 14.50x | +4.98x |
| IBM z16/s390x | `crc16-ibm` / `hash` / `1024` | `crc` | 52.29x | 56.88x | +4.59x |
| IBM z16/s390x | `blake3` / `derive-key` / `262144` | `blake3` | 4.19x | 8.53x | +4.34x |
| IBM z16/s390x | `crc16-ibm` / `hash` / `1048576` | `crc` | 96.80x | 100.97x | +4.17x |
| IBM Power10 | `blake3` / `derive-key` / `262144` | `blake3` | 5.59x | 9.64x | +4.04x |
| IBM z16/s390x | `hmac-sha256` / `hash` / `65536` | `rustcrypto` | 9.24x | 12.83x | +3.59x |
| IBM z16/s390x | `hmac-sha256` / `hash` / `262144` | `rustcrypto` | 9.27x | 12.84x | +3.56x |

## Coverage Matrix

| Platform | Raw Criterion rows | All pairs | Fastest rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 2,096 | 1,178 | 725 | 515/147/63 | 71% | 1.48x | 1.14x |
| AMD Zen5 | 2,096 | 1,178 | 725 | 426/231/68 | 59% | 1.50x | 1.10x |
| AWS Graviton3 | 2,096 | 1,178 | 725 | 358/285/82 | 49% | 1.38x | 1.04x |
| AWS Graviton4 | 2,096 | 1,178 | 725 | 364/314/47 | 50% | 1.39x | 1.05x |
| IBM Power10 | 1,872 | 964 | 725 | 382/300/43 | 53% | 1.91x | 1.06x |
| IBM z16/s390x | 1,872 | 964 | 725 | 594/90/41 | 82% | 3.14x | 2.28x |
| Intel Ice Lake | 2,096 | 1,178 | 725 | 506/134/85 | 70% | 1.48x | 1.19x |
| Intel Sapphire Rapids | 2,096 | 1,178 | 725 | 479/165/81 | 66% | 1.62x | 1.18x |
| RISE RISC-V | 2,096 | 1,178 | 725 | 334/254/137 | 46% | 1.09x | 1.03x |
| Apple Silicon macOS/aarch64 | 2,171 | 1,200 | 725 | 373/320/32 | 51% | 1.41x | 1.05x |

## Category Summary

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 1,287 | 744/400/143 | 58% | 2.52x | 1.17x |
| Hashes/MACs/XOFs | 3,132 | 1870/1134/128 | 60% | 1.41x | 1.10x |
| Auth/KDF | 180 | 161/19/0 | 89% | 1.24x | 1.12x |
| Password hashing | 135 | 64/14/57 | 47% | 0.96x | 0.99x |
| Public-key | 108 | 41/54/13 | 38% | 1.10x | 1.02x |
| RSA | 99 | 85/6/8 | 86% | 1.55x | 1.16x |
| AEAD | 1,584 | 993/293/298 | 63% | 1.56x | 1.15x |

## Apple Silicon Category Summary

The 2026-06-09 macOS/aarch64 run is local Apple Silicon evidence from `benchmark_results/2026-06-09/macos/aarch64/results.txt`. It is not combined with Linux totals.

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 143 | 79/54/10 | 55% | 2.66x | 1.07x |
| Hashes/MACs/XOFs | 348 | 142/187/19 | 41% | 1.10x | 1.02x |
| Auth/KDF | 20 | 4/14/2 | 20% | 1.01x | 1.01x |
| Password hashing | 15 | 5/10/0 | 33% | 1.07x | 1.03x |
| Public-key | 12 | 3/9/0 | 25% | 1.03x | 1.02x |
| RSA | 11 | 11/0/0 | 100% | 1.46x | 1.21x |
| AEAD | 176 | 129/46/1 | 73% | 1.48x | 1.16x |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `scrypt-small` | 36 | 20/0/16 | 0.89x | 36 | 20/0/16 | 0.89x |
| `scrypt-owasp` | 9 | 5/0/4 | 0.92x | 9 | 5/0/4 | 0.92x |
| `argon2id-owasp` | 9 | 3/2/4 | 0.98x | 18 | 10/3/5 | 1.27x |
| `argon2i-small` | 27 | 12/3/12 | 0.99x | 45 | 29/3/13 | 1.36x |
| `argon2id-small` | 27 | 12/3/12 | 1.00x | 45 | 29/3/13 | 1.38x |
| `chacha20-poly1305` | 198 | 72/57/69 | 1.00x | 550 | 301/130/119 | 1.16x |
| `argon2d-small` | 27 | 12/6/9 | 1.01x | 27 | 12/6/9 | 1.01x |
| `x25519` | 18 | 5/12/1 | 1.04x | 50 | 36/13/1 | 1.54x |
| `rapidhash-v3-64` | 99 | 26/52/21 | 1.05x | 99 | 26/52/21 | 1.05x |
| `rapidhash-v3-128` | 99 | 34/42/23 | 1.07x | 99 | 34/42/23 | 1.07x |
| `blake2` | 846 | 466/378/2 | 1.09x | 1,071 | 677/392/2 | 1.24x |
| `ed25519` | 90 | 36/42/12 | 1.11x | 290 | 212/63/15 | 1.37x |
| `rapidhash-64` | 99 | 33/56/10 | 1.12x | 99 | 33/56/10 | 1.12x |
| `rapidhash-128` | 99 | 40/57/2 | 1.15x | 99 | 40/57/2 | 1.15x |
| `xxh3-128` | 99 | 49/36/14 | 1.16x | 99 | 49/36/14 | 1.16x |
| `xxh3-64` | 99 | 51/35/13 | 1.17x | 99 | 51/35/13 | 1.17x |
| `rsa-8192` | 18 | 14/2/2 | 1.19x | 32 | 28/2/2 | 1.26x |
| `hkdf-sha256` | 36 | 32/4/0 | 1.21x | 100 | 96/4/0 | 1.81x |
| `sha256` | 117 | 45/56/16 | 1.23x | 293 | 165/98/30 | 1.46x |
| `sha512` | 117 | 43/63/11 | 1.24x | 293 | 170/106/17 | 1.28x |
| `pbkdf2-sha256` | 54 | 52/2/0 | 1.25x | 150 | 148/2/0 | 1.59x |
| `hmac-sha512` | 99 | 52/40/7 | 1.25x | 275 | 183/80/12 | 1.29x |
| `sha384` | 99 | 45/43/11 | 1.25x | 275 | 175/83/17 | 1.28x |
| `pbkdf2-sha512` | 54 | 44/10/0 | 1.25x | 150 | 139/11/0 | 1.31x |
| `hmac-sha384` | 99 | 49/42/8 | 1.25x | 275 | 179/83/13 | 1.30x |
| `hkdf-sha384` | 36 | 33/3/0 | 1.26x | 100 | 97/3/0 | 1.58x |
| `ascon-hash256` | 99 | 56/42/1 | 1.27x | 99 | 56/42/1 | 1.27x |
| `sha512-256` | 99 | 60/39/0 | 1.30x | 99 | 60/39/0 | 1.30x |
| `ascon-aead128` | 198 | 153/44/1 | 1.33x | 198 | 153/44/1 | 1.33x |
| `hmac-sha256` | 117 | 64/37/16 | 1.37x | 293 | 194/73/26 | 1.69x |
| `xchacha20-poly1305` | 198 | 170/27/1 | 1.37x | 198 | 170/27/1 | 1.37x |
| `ascon-xof128` | 99 | 75/24/0 | 1.38x | 99 | 75/24/0 | 1.38x |
| `aegis-256` | 198 | 97/78/23 | 1.39x | 198 | 97/78/23 | 1.39x |
| `blake3` | 432 | 229/169/34 | 1.41x | 432 | 229/169/34 | 1.41x |
| `rsa-4096` | 27 | 23/2/2 | 1.60x | 59 | 55/2/2 | 2.53x |
| `rsa-3072` | 27 | 23/2/2 | 1.64x | 59 | 55/2/2 | 2.56x |
| `crc32c` | 99 | 51/36/12 | 1.64x | 198 | 149/37/12 | 2.25x |
| `crc32` | 99 | 45/36/18 | 1.68x | 198 | 141/39/18 | 2.29x |
| `rsa-2048` | 27 | 25/0/2 | 1.69x | 59 | 57/0/2 | 2.60x |
| `kmac256` | 99 | 59/26/14 | 1.74x | 99 | 59/26/14 | 1.74x |
| `crc64-nvme` | 99 | 49/36/14 | 1.75x | 198 | 127/53/18 | 2.28x |
| `aes-128-gcm` | 198 | 113/41/44 | 1.76x | 550 | 428/47/75 | 2.54x |
| `cshake256` | 99 | 61/30/8 | 1.79x | 99 | 61/30/8 | 1.79x |
| `aes-256-gcm` | 198 | 113/39/46 | 1.79x | 550 | 423/51/76 | 2.59x |
| `shake128` | 99 | 76/23/0 | 1.85x | 99 | 76/23/0 | 1.85x |
| `shake256` | 99 | 70/29/0 | 1.86x | 99 | 70/29/0 | 1.86x |
| `sha224` | 99 | 54/45/0 | 1.91x | 99 | 54/45/0 | 1.91x |
| `aes-128-gcm-siv` | 198 | 137/4/57 | 2.03x | 352 | 268/19/65 | 2.81x |
| `sha3-256` | 117 | 101/16/0 | 2.10x | 117 | 101/16/0 | 2.10x |
| `sha3-224` | 99 | 88/11/0 | 2.12x | 99 | 88/11/0 | 2.12x |
| `sha3-384` | 99 | 89/10/0 | 2.16x | 99 | 89/10/0 | 2.16x |
| `aes-256-gcm-siv` | 198 | 138/3/57 | 2.16x | 352 | 291/4/57 | 3.06x |
| `sha3-512` | 99 | 88/11/0 | 2.19x | 99 | 88/11/0 | 2.19x |
| `crc64-xz` | 99 | 75/12/12 | 2.41x | 99 | 75/12/12 | 2.41x |
| `crc24-openpgp` | 99 | 95/2/2 | 13.20x | 99 | 95/2/2 | 13.20x |
| `crc16-ccitt` | 99 | 98/0/1 | 22.99x | 99 | 98/0/1 | 22.99x |
| `crc16-ibm` | 99 | 98/0/1 | 24.20x | 99 | 98/0/1 | 24.20x |

## Linux Clear Losses

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | --- | --- | --- | --- | --- |
| `scrypt-small` / `password-hash` | 36 | 20/0/16 | 0.89x | 1.16x | `rustcrypto` 16 |
| `scrypt-owasp` / `password-hash` | 9 | 5/0/4 | 0.92x | 1.08x | `rustcrypto` 4 |
| `argon2id-owasp` / `password-hash` | 9 | 3/2/4 | 0.98x | 0.96x | `rustcrypto` 6 |
| `chacha20-poly1305` / `encrypt` | 99 | 33/32/34 | 0.99x | 1.01x | `ring` 49, `aws-lc-rs` 16, `rustcrypto` 1 |
| `argon2i-small` / `password-hash` | 27 | 12/3/12 | 0.99x | 0.98x | `rustcrypto` 15 |
| `argon2id-small` / `password-hash` | 27 | 12/3/12 | 1.00x | 0.97x | `rustcrypto` 15 |
| `ed25519` / `verify` | 36 | 9/18/9 | 1.00x | 1.00x | `dalek` 9, `aws-lc-rs` 9, `ring` 9 |
| `chacha20-poly1305` / `decrypt` | 99 | 39/25/35 | 1.01x | 1.00x | `aws-lc-rs` 34, `ring` 26 |
| `argon2d-small` / `password-hash` | 27 | 12/6/9 | 1.01x | 0.99x | `rustcrypto` 15 |
| `x25519` / `public-key-from-secret` | 9 | 2/6/1 | 1.02x | 1.01x | `aws-lc-rs` 6, `dalek` 1 |
| `blake2` / `streaming` | 54 | 16/38/0 | 1.04x | 1.01x | `rustcrypto` 36, `dryoc` 2 |
| `x25519` / `diffie-hellman` | 9 | 3/6/0 | 1.05x | 1.01x | `aws-lc-rs` 6 |
| `rapidhash-v3-64` / `hash` | 99 | 26/52/21 | 1.05x | 1.00x | `rapidhash` 73 |
| `ed25519` / `sign` | 36 | 13/21/2 | 1.07x | 1.03x | `aws-lc-rs` 20, `dalek` 2, `dryoc` 1 |
| `rapidhash-v3-128` / `hash` | 99 | 34/42/23 | 1.07x | 1.01x | `rapidhash` 65 |
| `blake2` / `keyed` | 396 | 210/186/0 | 1.08x | 1.06x | `rustcrypto` 186 |
| `blake2` / `hash` | 396 | 240/154/2 | 1.11x | 1.10x | `rustcrypto` 156 |
| `rapidhash-64` / `hash` | 99 | 33/56/10 | 1.12x | 1.01x | `rapidhash` 66 |
| `rapidhash-128` / `hash` | 99 | 40/57/2 | 1.15x | 1.02x | `rapidhash` 59 |
| `blake3` / `streaming` | 36 | 8/22/6 | 1.16x | 1.00x | `blake3` 28 |
| `xxh3-128` / `hash` | 99 | 49/36/14 | 1.16x | 1.05x | `xxhash-rust` 50 |
| `xxh3-64` / `hash` | 99 | 51/35/13 | 1.17x | 1.07x | `xxhash-rust` 48 |
| `sha256` / `streaming` | 18 | 3/12/3 | 1.18x | 1.00x | `sha2` 15 |
| `rsa-4096` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.19x | 1.13x | `ring` 1, `aws-lc-rs` 1 |
| `rsa-8192` / `verify-pss-sha256` | 9 | 7/1/1 | 1.19x | 1.09x | `ring` 1, `aws-lc-rs` 1 |
| `rsa-3072` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.19x | 1.15x | `ring` 1, `aws-lc-rs` 1 |
| `rsa-3072` / `verify-pss-sha256` | 9 | 7/1/1 | 1.19x | 1.15x | `ring` 1, `aws-lc-rs` 1 |
| `rsa-8192` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.20x | 1.09x | `ring` 1, `aws-lc-rs` 1 |
| `hmac-sha256` / `streaming` | 18 | 3/11/4 | 1.22x | 1.00x | `rustcrypto` 15 |
| `rsa-4096` / `verify-pss-sha256` | 9 | 7/1/1 | 1.22x | 1.13x | `ring` 1, `aws-lc-rs` 1 |
| `sha256` / `hash` | 99 | 42/44/13 | 1.24x | 1.01x | `aws-lc-rs` 32, `sha2` 23, `ring` 2 |
| `sha512` / `hash` | 99 | 38/50/11 | 1.24x | 1.04x | `aws-lc-rs` 32, `sha2` 18, `ring` 11 |
| `hmac-sha512` / `hash` | 99 | 52/40/7 | 1.25x | 1.05x | `rustcrypto` 19, `ring` 18, `aws-lc-rs` 10 |
| `rsa-2048` / `verify-pkcs1v15-sha256` | 9 | 8/0/1 | 1.25x | 1.23x | `aws-lc-rs` 1 |
| `sha384` / `hash` | 99 | 45/43/11 | 1.25x | 1.05x | `aws-lc-rs` 34, `sha2` 14, `ring` 6 |
| `hmac-sha384` / `hash` | 99 | 49/42/8 | 1.25x | 1.05x | `rustcrypto` 24, `ring` 17, `aws-lc-rs` 9 |
| `rsa-2048` / `verify-pss-sha256` | 9 | 8/0/1 | 1.26x | 1.21x | `aws-lc-rs` 1 |
| `ascon-hash256` / `hash` | 99 | 56/42/1 | 1.27x | 1.06x | `ascon-hash` 43 |
| `aegis-256` / `encrypt` | 99 | 38/40/21 | 1.31x | 1.01x | `aegis-crate` 61 |
| `blake3` / `keyed` | 99 | 44/47/8 | 1.33x | 1.02x | `blake3` 55 |
| `blake3` / `hash` | 99 | 48/42/9 | 1.34x | 1.05x | `blake3` 51 |
| `blake3` / `xof` | 99 | 52/40/7 | 1.34x | 1.05x | `blake3` 47 |
| `ascon-aead128` / `decrypt` | 99 | 76/22/1 | 1.34x | 1.17x | `ascon-aead` 23 |
| `xchacha20-poly1305` / `encrypt` | 99 | 82/16/1 | 1.36x | 1.41x | `rustcrypto` 17 |
| `hmac-sha256` / `hash` | 99 | 61/26/12 | 1.40x | 1.12x | `ring` 25, `rustcrypto` 12, `aws-lc-rs` 1 |
| `aegis-256` / `decrypt` | 99 | 59/38/2 | 1.47x | 1.12x | `aegis-crate` 40 |
| `ed25519` / `public-key-from-secret` | 9 | 7/1/1 | 1.49x | 1.29x | `dalek` 2 |
| `crc32c` / `hash` | 99 | 51/36/12 | 1.64x | 1.05x | `crc-fast` 41, `crc32c` 7 |
| `aes-128-gcm` / `encrypt` | 99 | 53/20/26 | 1.67x | 1.09x | `aws-lc-rs` 24, `rustcrypto` 19, `ring` 3 |
| `crc32` / `hash` | 99 | 45/36/18 | 1.68x | 1.04x | `crc-fast` 38, `crc32fast` 16 |
| `aes-256-gcm` / `encrypt` | 99 | 53/20/26 | 1.71x | 1.07x | `aws-lc-rs` 23, `rustcrypto` 19, `ring` 4 |
| `kmac256` / `hash` | 99 | 59/26/14 | 1.74x | 1.96x | `tiny-keccak` 40 |
| `crc64-nvme` / `hash` | 99 | 49/36/14 | 1.75x | 1.05x | `crc-fast` 27, `crc64fast-nvme` 23 |
| `cshake256` / `hash` | 99 | 61/30/8 | 1.79x | 1.98x | `tiny-keccak` 38 |
| `blake3` / `derive-key` | 99 | 77/18/4 | 1.80x | 1.90x | `blake3` 22 |
| `aes-128-gcm` / `decrypt` | 99 | 60/21/18 | 1.84x | 1.21x | `aws-lc-rs` 26, `rustcrypto` 11, `ring` 2 |
| `aes-256-gcm` / `decrypt` | 99 | 60/19/20 | 1.87x | 1.20x | `aws-lc-rs` 27, `rustcrypto` 10, `ring` 2 |
| `aes-128-gcm-siv` / `decrypt` | 99 | 68/2/29 | 1.98x | 1.62x | `aws-lc-rs` 25, `rustcrypto` 6 |
| `aes-128-gcm-siv` / `encrypt` | 99 | 69/2/28 | 2.08x | 1.79x | `aws-lc-rs` 24, `rustcrypto` 6 |
| `aes-256-gcm-siv` / `decrypt` | 99 | 69/1/29 | 2.09x | 1.53x | `aws-lc-rs` 24, `rustcrypto` 6 |
| `aes-256-gcm-siv` / `encrypt` | 99 | 69/2/28 | 2.24x | 1.87x | `aws-lc-rs` 24, `rustcrypto` 6 |
| `crc64-xz` / `hash` | 99 | 75/12/12 | 2.41x | 2.18x | `crc64fast` 24 |
| `crc24-openpgp` / `hash` | 99 | 95/2/2 | 13.20x | 14.68x | `crc` 4 |
| `crc16-ccitt` / `hash` | 99 | 98/0/1 | 22.99x | 37.01x | `crc` 1 |
| `crc16-ibm` / `hash` | 99 | 98/0/1 | 24.20x | 38.66x | `crc` 1 |

## Apple Silicon Clear Losses

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | --- | --- | --- | --- | --- |
| `sha3-256` / `streaming` | 2 | 0/0/2 | 0.93x | 0.93x | `sha3` 2 |
| `sha3-256` / `hash` | 11 | 0/4/7 | 0.94x | 0.93x | `sha3` 7 |
| `xxh3-64` / `hash` | 11 | 1/3/7 | 0.96x | 0.85x | `xxhash-rust` 7 |
| `hkdf-sha256` / `expand` | 4 | 0/3/1 | 0.96x | 0.98x | `rustcrypto` 1 |
| `sha3-512` / `hash` | 11 | 0/8/3 | 0.97x | 0.99x | `sha3` 3 |

## Linux Worst Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Ice Lake | `scrypt-small` / `password-hash` / `log_n=10_r=8_p=4` | `rustcrypto` | 0.34x |
| Intel Ice Lake | `scrypt-small` / `password-hash` / `log_n=10_r=8_p=1` | `rustcrypto` | 0.36x |
| AMD Zen4 | `scrypt-small` / `password-hash` / `log_n=14_r=8_p=1` | `rustcrypto` | 0.38x |
| AMD Zen4 | `scrypt-small` / `password-hash` / `log_n=12_r=8_p=1` | `rustcrypto` | 0.38x |
| Intel Ice Lake | `scrypt-small` / `password-hash` / `log_n=14_r=8_p=1` | `rustcrypto` | 0.39x |
| Intel Ice Lake | `scrypt-small` / `password-hash` / `log_n=12_r=8_p=1` | `rustcrypto` | 0.39x |
| AMD Zen4 | `scrypt-small` / `password-hash` / `log_n=10_r=8_p=1` | `rustcrypto` | 0.39x |
| AMD Zen4 | `scrypt-small` / `password-hash` / `log_n=10_r=8_p=4` | `rustcrypto` | 0.39x |
| Intel Sapphire Rapids | `scrypt-small` / `password-hash` / `log_n=14_r=8_p=1` | `rustcrypto` | 0.41x |
| Intel Sapphire Rapids | `scrypt-small` / `password-hash` / `log_n=12_r=8_p=1` | `rustcrypto` | 0.42x |
| RISE RISC-V | `xxh3-64` / `hash` / `0` | `xxhash-rust` | 0.43x |
| Intel Sapphire Rapids | `scrypt-small` / `password-hash` / `log_n=10_r=8_p=1` | `rustcrypto` | 0.44x |

## Linux Strongest Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt` / `hash` / `16384` | `crc` | 229.44x |
| Intel Sapphire Rapids | `crc16-ccitt` / `hash` / `262144` | `crc` | 212.47x |
| Intel Sapphire Rapids | `crc16-ibm` / `hash` / `262144` | `crc` | 211.21x |
| Intel Sapphire Rapids | `crc16-ibm` / `hash` / `16384` | `crc` | 198.70x |
| Intel Sapphire Rapids | `crc16-ibm` / `hash` / `1048576` | `crc` | 184.18x |
| Intel Sapphire Rapids | `crc16-ccitt` / `hash` / `4096` | `crc` | 183.86x |
| Intel Sapphire Rapids | `crc16-ibm` / `hash` / `65536` | `crc` | 181.43x |
| Intel Sapphire Rapids | `crc16-ccitt` / `hash` / `1048576` | `crc` | 179.97x |
| IBM Power10 | `crc16-ccitt` / `hash` / `262144` | `crc` | 177.06x |
| IBM Power10 | `crc16-ccitt` / `hash` / `1048576` | `crc` | 176.80x |
| IBM Power10 | `crc16-ibm` / `hash` / `1048576` | `crc` | 176.73x |
| IBM Power10 | `crc16-ibm` / `hash` / `262144` | `crc` | 176.68x |

## Top Five Loss Areas

1. **Linux scrypt/password hashing:** `scrypt-small` is 0.89x geomean and `scrypt-owasp` is 0.92x. The worst current rows are x86 scrypt cases against `rustcrypto` at 0.34x..0.49x. Argon2 is near parity overall, but small/OWASP variants still carry row losses.
2. **Linux ChaCha20-Poly1305:** The aggregate is close to parity, but encrypt has 34 losses at 0.99x geomean and decrypt has 35 losses at 1.01x geomean, mostly against `ring` and `aws-lc-rs`. This is the most visible AEAD gap after scrypt.
3. **Near-parity public-key operations:** Public-key overall improved to 1.10x, but Ed25519 verify is still 1.00x with 9 losses, Ed25519 sign is 1.07x with 2 losses, and X25519 public-key derivation remains only 1.02x. These are no longer category failures, but they are not decisive wins.
4. **RISE RISC-V and small-input checksum point losses:** The RISE RISC-V platform remains the weakest Linux runner at 1.09x geomean. Its worst current row is `xxh3-64` size 0 at 0.43x against `xxhash-rust`, and several checksum/hash rows are effectively parity instead of clear wins.
5. **Apple Silicon SHA3/XXH3 plus localized BLAKE3/CRC32(C) rows:** SHA3-256 streaming is 0.93x, SHA3-256 hash is 0.94x, XXH3-64 is 0.96x, and SHA3-512 hash is 0.97x. BLAKE3 64 KiB rows and 256-byte CRC32/CRC32C rows remain localized point losses, not category-level failures.

## External Pressure

All-pair Linux CI comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `rapidhash` | 396 | 133/207/56 | 34% | 1.10x | 1.01x |
| `xxhash-rust` | 198 | 100/71/27 | 51% | 1.17x | 1.05x |
| `aws-lc-rs` | 1,498 | 897/326/275 | 60% | 1.24x | 1.12x |
| `ascon-hash` | 198 | 131/66/1 | 66% | 1.32x | 1.26x |
| `ascon-aead` | 198 | 153/44/1 | 77% | 1.33x | 1.15x |
| `aegis-crate` | 198 | 97/78/23 | 49% | 1.39x | 1.04x |
| `blake3` | 432 | 229/169/34 | 53% | 1.41x | 1.06x |
| `dalek` | 108 | 87/16/5 | 81% | 1.47x | 1.28x |
| `sha2` | 531 | 301/226/4 | 57% | 1.49x | 1.07x |
| `ring` | 1,512 | 1162/201/149 | 77% | 1.55x | 1.26x |
| `tiny-keccak` | 396 | 266/108/22 | 67% | 1.81x | 1.97x |
| `dryoc` | 360 | 325/29/6 | 90% | 1.87x | 1.82x |
| `rustcrypto` | 2,664 | 1913/596/155 | 72% | 1.92x | 1.17x |
| `crc-fast` | 297 | 170/105/22 | 57% | 2.04x | 1.15x |
| `sha3` | 414 | 366/48/0 | 88% | 2.14x | 2.12x |
| `crc64fast-nvme` | 99 | 73/14/12 | 74% | 2.41x | 2.06x |
| `crc64fast` | 99 | 75/12/12 | 76% | 2.41x | 2.18x |
| `crc32fast` | 99 | 82/7/10 | 83% | 2.44x | 2.01x |
| `crc32c` | 99 | 92/3/4 | 93% | 2.79x | 2.17x |
| `rustcrypto-rsa` | 81 | 81/0/0 | 100% | 5.71x | 4.63x |
| `crc` | 297 | 291/2/4 | 98% | 19.44x | 29.79x |

## README Numbers

- **Headline:** 3,958 of 6,525 matched Linux CI fastest-external comparisons are wins; 5,878 are wins or ties. Linux CI geomean is 1.60x.
- **Apple Silicon headline:** 373 of 725 matched fastest-external comparisons are wins; 693 are wins or ties. Apple Silicon geomean is 1.41x.
- **Checksums:** 2.52x geomean across 1,287 Linux CI fastest-external rows; W/T/L 744/400/143.
- **Hashes/MACs/XOFs:** 1.41x geomean across 3,132 Linux CI fastest-external rows; W/T/L 1870/1134/128.
- **Auth/KDF:** 1.24x geomean across 180 Linux CI fastest-external rows; W/T/L 161/19/0.
- **Password hashing:** 0.96x geomean across 135 Linux CI fastest-external rows; W/T/L 64/14/57.
- **Public-key:** 1.10x geomean across 108 Linux CI fastest-external rows; W/T/L 41/54/13.
- **RSA:** 1.55x geomean across 99 Linux CI fastest-external rows; W/T/L 85/6/8.
- **AEAD:** 1.56x geomean across 1,584 Linux CI fastest-external rows; W/T/L 993/293/298.
- **Apple Silicon AEAD:** 1.48x geomean across 176 fastest-external rows; ChaCha20-Poly1305 is near parity at 1.02x encrypt and 1.03x decrypt.
- **Current top losses:** Linux scrypt/password hashing, Linux ChaCha20-Poly1305 rows against `ring`/`aws-lc-rs`, near-parity Ed25519/X25519 operations, RISE RISC-V checksum point losses, and Apple Silicon SHA3/XXH3 plus localized BLAKE3/CRC32(C) rows.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-06-09 18_43_21` | 2,096 | `benchmark_results/2026-06-09/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-06-09 18_43_21` | 2,096 | `benchmark_results/2026-06-09/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-06-09 18_43_21` | 2,096 | `benchmark_results/2026-06-09/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-06-09 18_43_21` | 2,096 | `benchmark_results/2026-06-09/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-06-09 18_43_21` | 1,872 | `benchmark_results/2026-06-09/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-06-09 18_43_21` | 1,872 | `benchmark_results/2026-06-09/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-06-09 18_43_21` | 2,096 | `benchmark_results/2026-06-09/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-06-09 18_43_21` | 2,096 | `benchmark_results/2026-06-09/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-06-09 18_43_21` | 2,096 | `benchmark_results/2026-06-09/linux/rise-riscv/results.txt` |
| Apple Silicon macOS/aarch64 | `local` | `2026-06-09 14_02_43` | 2,171 | `benchmark_results/2026-06-09/macos/aarch64/results.txt` |
