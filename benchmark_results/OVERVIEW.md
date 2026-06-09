# Benchmark Overview

Sources:

- Linux benchmark CI run [#27164564672](https://github.com/loadingalias/rscrypto/actions/runs/27164564672), created 2026-06-08 20:21:15 UTC.
- Linux commit: `8a0ac692ab68f09b47ca65b4cb6ecca7fc333f31`.
- Apple Silicon local run: `benchmark_results/2026-06-08/macos/aarch64/results.txt`, created 2026-06-08 18:48:04 local time.
- Apple Silicon commit: `8a0ac692ab68f09b47ca65b4cb6ecca7fc333f31`.
- Previous Linux comparison baseline: run [#26537297244](https://github.com/loadingalias/rscrypto/actions/runs/26537297244), created 2026-05-27 20:37:14 UTC.
- Previous Linux commit: `26845c85530d90725d3ad90c7871d7a474d80d27`.
- Previous Apple Silicon local comparison baseline: `benchmark_results/2026-06-01/macos/aarch64/results.txt`, created 2026-06-01 11:45:23 local time on an MBP M1.

Scope: the 2026-06-08 nine-runner Linux CI benchmark matrix and the 2026-06-08 local Apple Silicon full run for commit `8a0ac69`. Ratios are `external_crate_time / rscrypto_time`; higher is always better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and size. Internal kernel, scratch-buffer, padding-only, cold-path, and threshold-selection microbenches are parsed as raw rows but excluded from external win/loss claims.

Coverage note: the current Linux and Apple Silicon passes are both full benchmark runs. They include checksum, hash, XOF, MAC, KDF, password-hashing, RSA import/verification, Ed25519, X25519, and AEAD rows, including Argon2, scrypt, and Ascon-AEAD. Apple Silicon remains a separate local reference and is not combined into Linux aggregate totals.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Linux CI: all matched performance pairs | 9,544 | 6566/2142/836 | 69% | 1.75x | 1.19x |
| Linux CI: fastest external per case | 5,994 | 3627/1718/649 | 61% | 1.60x | 1.11x |
| Apple Silicon: all matched performance pairs | 1,166 | 736/340/90 | 63% | 1.79x | 1.12x |
| Apple Silicon: fastest external per case | 702 | 368/275/59 | 52% | 1.39x | 1.06x |
| Previous Linux CI: fastest external per case | 5,463 | 3325/1560/578 | 61% | 1.62x | 1.11x |

Shareable release summary:

- **Headline:** 3,627 of 5,994 matched Linux CI fastest-external comparisons are wins; 5,345 are wins or ties. Linux CI fastest-external geomean is 1.60x.
- **Apple Silicon headline:** 368 of 702 matched fastest-external comparisons are wins; 643 are wins or ties. Apple Silicon fastest-external geomean is 1.39x.
- **Checksums:** 2.62x geomean across 1,188 fastest-external rows; W/T/L is 693/351/144.
- **Hashes/MACs/XOFs:** 1.40x geomean across 2,700 fastest-external rows; W/T/L is 1634/969/97.
- **Auth/KDF:** 1.17x geomean across 180 fastest-external rows; W/T/L is 160/16/4.
- **Password hashing:** 0.97x geomean across 135 fastest-external rows; W/T/L is 62/15/58.
- **Public-key:** 0.99x geomean across 108 fastest-external rows; W/T/L is 29/53/26.
- **RSA:** 1.30x geomean across 99 fastest-external rows; W/T/L is 75/0/24.
- **AEAD:** 1.56x geomean across 1,584 fastest-external rows; W/T/L is 974/314/296.
- **Apple Silicon AEAD:** 1.44x geomean across 176 fastest-external rows, but ChaCha20-Poly1305 is the local AEAD weak spot at 0.86x.
- **Top current loss areas:** Linux public-key/RSA pressure, Apple Silicon ChaCha20-Poly1305, password/KDF front-end overhead, Apple Silicon SHA/XXH3 holes, and PBKDF2-SHA256 `iters=1`.

## What Changed Since 2026-05-27

- Fastest-external coverage increased from 5,463 to 5,994 rows because the current run includes password hashing, Ascon-AEAD, and additional full-run cases.
- Headline fastest-external geomean moved from 1.62x to 1.60x. That is not a pure regression signal because the scope is broader; use the per-case delta tables below for apples-to-apples movement.
- Apple Silicon is now refreshed on the same commit as Linux. It adds 702 fastest-external rows at 1.39x geomean, up from the old local reference's 463 fastest-external rows, but the added full-run coverage exposes a clear local ChaCha20-Poly1305 loss.

| New/expanded category | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| Hashes/MACs/XOFs | 198 | 126/50/22 | 1.75x | 1.98x |
| Password hashing | 135 | 62/15/58 | 0.97x | 1.01x |
| AEAD | 198 | 149/49/0 | 1.33x | 1.15x |

Largest apples-to-apples drops among cases that existed in both Linux runs:

These are ratio movements, not necessarily current losses. Several rows below are
still dominant rscrypto wins, but their margin narrowed.

| Platform | Case | Fastest external | Previous | Current | Delta |
| --- | --- | --- | --- | --- | --- |
| Intel Ice Lake | `crc16-ccitt` / `hash` / `1024` | `crc` | 75.67x | 64.59x | -11.08x |
| Intel Sapphire Rapids | `crc16-ibm` / `hash` / `4096` | `crc` | 175.50x | 167.01x | -8.50x |
| IBM z16/s390x | `sha256` / `hash` / `1048576` | `sha2` | 14.72x | 9.26x | -5.46x |
| AWS Graviton4 | `crc24-openpgp` / `hash` / `1048576` | `crc` | 92.95x | 88.53x | -4.41x |
| IBM z16/s390x | `sha256` / `hash` / `262144` | `sha2` | 12.78x | 9.34x | -3.44x |
| IBM z16/s390x | `aes-256-gcm-siv` / `decrypt` / `1` | `rustcrypto` | 28.12x | 24.74x | -3.37x |
| IBM z16/s390x | `sha256` / `hash` / `65536` | `sha2` | 12.16x | 9.24x | -2.92x |
| Intel Ice Lake | `crc16-ibm` / `hash` / `1024` | `crc` | 74.10x | 71.21x | -2.89x |
| IBM z16/s390x | `sha512-256` / `hash` / `262144` | `sha2` | 13.36x | 10.63x | -2.73x |
| IBM z16/s390x | `sha3-512` / `hash` / `16384` | `sha3` | 23.52x | 20.80x | -2.72x |

Largest apples-to-apples gains among cases that existed in both Linux runs:

| Platform | Case | Fastest external | Previous | Current | Delta |
| --- | --- | --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt` / `hash` / `16384` | `crc` | 199.04x | 229.74x | +30.70x |
| Intel Sapphire Rapids | `crc24-openpgp` / `hash` / `4096` | `crc` | 76.65x | 91.05x | +14.40x |
| Intel Sapphire Rapids | `sha224` / `hash` / `64` | `sha2` | 40.52x | 54.03x | +13.52x |
| Intel Sapphire Rapids | `sha224` / `hash` / `256` | `sha2` | 79.58x | 88.08x | +8.50x |
| IBM z16/s390x | `aes-256-gcm-siv` / `decrypt` / `65536` | `rustcrypto` | 5.76x | 13.87x | +8.11x |
| IBM z16/s390x | `aes-256-gcm-siv` / `decrypt` / `4096` | `rustcrypto` | 6.00x | 13.71x | +7.70x |
| IBM z16/s390x | `aes-256-gcm-siv` / `decrypt` / `16384` | `rustcrypto` | 5.73x | 13.32x | +7.59x |
| IBM z16/s390x | `aes-256-gcm-siv` / `decrypt` / `1024` | `rustcrypto` | 7.16x | 14.58x | +7.43x |
| IBM z16/s390x | `aes-128-gcm-siv` / `encrypt` / `16384` | `rustcrypto` | 4.06x | 11.21x | +7.15x |
| IBM z16/s390x | `aes-256-gcm-siv` / `decrypt` / `262144` | `rustcrypto` | 6.08x | 13.21x | +7.13x |

## Coverage Matrix

| Platform | Raw Criterion rows | All pairs | Fastest rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 2,096 | 1,108 | 666 | 473/128/65 | 71% | 1.50x | 1.15x |
| AMD Zen5 | 2,096 | 1,108 | 666 | 394/205/67 | 59% | 1.51x | 1.10x |
| AWS Graviton3 | 2,096 | 1,108 | 666 | 321/259/86 | 48% | 1.36x | 1.04x |
| AWS Graviton4 | 2,096 | 1,108 | 666 | 326/283/57 | 49% | 1.37x | 1.05x |
| IBM Power10 | 1,872 | 894 | 666 | 341/280/45 | 51% | 1.86x | 1.06x |
| IBM z16/s390x | 1,872 | 894 | 666 | 536/89/41 | 80% | 3.21x | 2.32x |
| Intel Ice Lake | 2,096 | 1,108 | 666 | 478/107/81 | 72% | 1.51x | 1.21x |
| Intel Sapphire Rapids | 2,096 | 1,108 | 666 | 463/132/71 | 70% | 1.65x | 1.17x |
| RISE RISC-V | 2,096 | 1,108 | 666 | 295/235/136 | 44% | 1.08x | 1.02x |
| Apple Silicon MBP M1 | 2,171 | 1,166 | 702 | 368/275/59 | 52% | 1.39x | 1.06x |

## Category Summary

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 1,188 | 693/351/144 | 58% | 2.62x | 1.19x |
| Hashes/MACs/XOFs | 2,700 | 1634/969/97 | 61% | 1.40x | 1.10x |
| Auth/KDF | 180 | 160/16/4 | 89% | 1.17x | 1.10x |
| Password hashing | 135 | 62/15/58 | 46% | 0.97x | 1.01x |
| Public-key | 108 | 29/53/26 | 27% | 0.99x | 1.01x |
| RSA | 99 | 75/0/24 | 76% | 1.30x | 1.15x |
| AEAD | 1,584 | 974/314/296 | 61% | 1.56x | 1.15x |

## Apple Silicon Category Summary

The 2026-06-08 macOS/aarch64 run is local Apple Silicon evidence on the same commit as the Linux CI run. It is not combined with Linux totals.

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 132 | 78/45/9 | 59% | 2.85x | 1.08x |
| Hashes/MACs/XOFs | 336 | 138/170/28 | 41% | 1.07x | 1.02x |
| Auth/KDF | 20 | 3/15/2 | 15% | 1.01x | 1.01x |
| Password hashing | 15 | 5/10/0 | 33% | 1.07x | 1.01x |
| Public-key | 12 | 2/10/0 | 17% | 1.02x | 1.01x |
| RSA | 11 | 11/0/0 | 100% | 1.45x | 1.24x |
| AEAD | 176 | 131/25/20 | 74% | 1.44x | 1.17x |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `x25519` | 18 | 5/9/4 | 0.89x | 50 | 34/12/4 | 1.31x |
| `scrypt-small` | 36 | 20/0/16 | 0.90x | 36 | 20/0/16 | 0.90x |
| `rsa-8192` | 18 | 12/0/6 | 0.91x | 32 | 20/0/12 | 0.91x |
| `scrypt-owasp` | 9 | 5/0/4 | 0.93x | 9 | 5/0/4 | 0.93x |
| `argon2id-owasp` | 9 | 3/2/4 | 0.97x | 18 | 10/3/5 | 1.26x |
| `argon2i-small` | 27 | 12/3/12 | 0.98x | 45 | 29/3/13 | 1.35x |
| `chacha20-poly1305` | 198 | 65/62/71 | 0.99x | 550 | 288/142/120 | 1.15x |
| `argon2d-small` | 27 | 11/6/10 | 1.00x | 27 | 11/6/10 | 1.00x |
| `ed25519` | 90 | 24/44/22 | 1.01x | 290 | 199/66/25 | 1.32x |
| `argon2id-small` | 27 | 11/4/12 | 1.01x | 45 | 28/4/13 | 1.39x |
| `pbkdf2-sha256` | 54 | 45/5/4 | 1.04x | 150 | 136/7/7 | 1.34x |
| `rapidhash-v3-64` | 99 | 27/49/23 | 1.05x | 99 | 27/49/23 | 1.05x |
| `rapidhash-v3-128` | 99 | 31/47/21 | 1.07x | 99 | 31/47/21 | 1.07x |
| `blake2` | 846 | 483/361/2 | 1.09x | 1,071 | 699/370/2 | 1.25x |
| `rapidhash-64` | 99 | 33/56/10 | 1.12x | 99 | 33/56/10 | 1.12x |
| `rapidhash-128` | 99 | 37/53/9 | 1.14x | 99 | 37/53/9 | 1.14x |
| `xxh3-128` | 99 | 51/36/12 | 1.18x | 99 | 51/36/12 | 1.18x |
| `xxh3-64` | 99 | 52/32/15 | 1.19x | 99 | 52/32/15 | 1.19x |
| `hkdf-sha256` | 36 | 33/3/0 | 1.22x | 100 | 97/3/0 | 1.81x |
| `sha256` | 117 | 43/57/17 | 1.22x | 293 | 164/98/31 | 1.46x |
| `pbkdf2-sha512` | 54 | 46/8/0 | 1.22x | 150 | 140/10/0 | 1.30x |
| `hmac-sha512` | 99 | 42/48/9 | 1.24x | 275 | 171/90/14 | 1.28x |
| `sha512` | 117 | 44/61/12 | 1.24x | 293 | 170/105/18 | 1.27x |
| `sha384` | 99 | 38/50/11 | 1.24x | 275 | 166/93/16 | 1.28x |
| `hmac-sha384` | 99 | 49/42/8 | 1.24x | 275 | 177/85/13 | 1.29x |
| `hkdf-sha384` | 36 | 36/0/0 | 1.25x | 100 | 100/0/0 | 1.56x |
| `ascon-hash256` | 99 | 55/42/2 | 1.26x | 99 | 55/42/2 | 1.26x |
| `sha512-256` | 99 | 52/47/0 | 1.28x | 99 | 52/47/0 | 1.28x |
| `ascon-aead128` | 198 | 149/49/0 | 1.33x | 198 | 149/49/0 | 1.33x |
| `hmac-sha256` | 117 | 64/39/14 | 1.34x | 293 | 194/75/24 | 1.67x |
| `rsa-4096` | 27 | 21/0/6 | 1.35x | 59 | 49/0/10 | 2.02x |
| `ascon-xof128` | 99 | 73/26/0 | 1.37x | 99 | 73/26/0 | 1.37x |
| `aegis-256` | 198 | 90/83/25 | 1.38x | 198 | 90/83/25 | 1.38x |
| `xchacha20-poly1305` | 198 | 171/26/1 | 1.38x | 198 | 171/26/1 | 1.38x |
| `rsa-3072` | 27 | 21/0/6 | 1.41x | 59 | 49/0/10 | 2.07x |
| `rsa-2048` | 27 | 21/0/6 | 1.46x | 59 | 49/0/10 | 2.14x |
| `crc32` | 99 | 48/29/22 | 1.68x | 198 | 144/32/22 | 2.30x |
| `crc64-nvme` | 99 | 51/33/15 | 1.71x | 198 | 133/48/17 | 2.26x |
| `kmac256` | 99 | 60/23/16 | 1.73x | 99 | 60/23/16 | 1.73x |
| `aes-128-gcm` | 198 | 115/41/42 | 1.76x | 550 | 431/50/69 | 2.55x |
| `cshake256` | 99 | 66/27/6 | 1.78x | 99 | 66/27/6 | 1.78x |
| `aes-256-gcm` | 198 | 108/44/46 | 1.78x | 550 | 419/56/75 | 2.59x |
| `shake128` | 99 | 76/23/0 | 1.85x | 99 | 76/23/0 | 1.85x |
| `shake256` | 99 | 70/29/0 | 1.87x | 99 | 70/29/0 | 1.87x |
| `sha224` | 99 | 53/46/0 | 1.90x | 99 | 53/46/0 | 1.90x |
| `aes-128-gcm-siv` | 198 | 137/4/57 | 2.03x | 352 | 267/20/65 | 2.81x |
| `sha3-256` | 117 | 102/15/0 | 2.10x | 117 | 102/15/0 | 2.10x |
| `sha3-224` | 99 | 88/11/0 | 2.12x | 99 | 88/11/0 | 2.12x |
| `sha3-384` | 99 | 88/11/0 | 2.15x | 99 | 88/11/0 | 2.15x |
| `aes-256-gcm-siv` | 198 | 139/5/54 | 2.16x | 352 | 292/6/54 | 3.07x |
| `sha3-512` | 99 | 88/11/0 | 2.18x | 99 | 88/11/0 | 2.18x |
| `crc64-xz` | 99 | 75/12/12 | 2.44x | 99 | 75/12/12 | 2.44x |
| `crc24-openpgp` | 99 | 92/4/3 | 13.12x | 99 | 92/4/3 | 13.12x |
| `crc16-ccitt` | 99 | 98/0/1 | 23.00x | 99 | 98/0/1 | 23.00x |
| `crc16-ibm` | 99 | 98/0/1 | 24.14x | 99 | 98/0/1 | 24.14x |

## Linux Clear Losses

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | --- | --- | --- | --- | --- |
| `pbkdf2-sha256` / `iters=1` | 18 | 14/2/2 | 0.81x | 1.12x | `rustcrypto` 2, `aws-lc-rs` 2 |
| `x25519` / `public-key-from-secret` | 9 | 2/5/2 | 0.85x | 1.01x | `aws-lc-rs` 6, `dalek` 1 |
| `scrypt-small` / `password-hash` | 36 | 20/0/16 | 0.90x | 1.16x | `rustcrypto` 16 |
| `rsa-8192` / `verify-pkcs1v15-sha256` | 9 | 6/0/3 | 0.90x | 1.10x | `aws-lc-rs` 3 |
| `rsa-8192` / `verify-pss-sha256` | 9 | 6/0/3 | 0.91x | 1.10x | `aws-lc-rs` 3 |
| `ed25519` / `keypair-from-secret` | 9 | 3/1/5 | 0.91x | 0.94x | `dalek` 6 |
| `ed25519` / `public-key-from-secret` | 9 | 3/1/5 | 0.92x | 0.94x | `dalek` 6 |
| `x25519` / `diffie-hellman` | 9 | 3/4/2 | 0.92x | 1.01x | `aws-lc-rs` 6 |
| `rsa-4096` / `verify-pss-sha256` | 9 | 6/0/3 | 0.93x | 1.12x | `aws-lc-rs` 3 |
| `rsa-4096` / `verify-pkcs1v15-sha256` | 9 | 6/0/3 | 0.93x | 1.13x | `aws-lc-rs` 3 |
| `scrypt-owasp` / `password-hash` | 9 | 5/0/4 | 0.93x | 1.07x | `rustcrypto` 4 |
| `rsa-3072` / `verify-pkcs1v15-sha256` | 9 | 6/0/3 | 0.95x | 1.15x | `aws-lc-rs` 3 |
| `rsa-3072` / `verify-pss-sha256` | 9 | 6/0/3 | 0.96x | 1.14x | `aws-lc-rs` 3 |
| `argon2id-owasp` / `password-hash` | 9 | 3/2/4 | 0.97x | 0.96x | `rustcrypto` 6 |
| `chacha20-poly1305` / `encrypt` | 99 | 29/35/35 | 0.98x | 1.00x | `ring` 58, `aws-lc-rs` 9, `rustcrypto` 3 |
| `argon2i-small` / `password-hash` | 27 | 12/3/12 | 0.98x | 0.97x | `rustcrypto` 15 |
| `chacha20-poly1305` / `decrypt` | 99 | 36/27/36 | 1.00x | 1.00x | `aws-lc-rs` 37, `ring` 24, `rustcrypto` 2 |
| `ed25519` / `verify` | 36 | 6/20/10 | 1.00x | 1.00x | `dalek` 12, `aws-lc-rs` 9, `ring` 9 |
| `argon2d-small` / `password-hash` | 27 | 11/6/10 | 1.00x | 0.99x | `rustcrypto` 16 |
| `argon2id-small` / `password-hash` | 27 | 11/4/12 | 1.01x | 1.01x | `rustcrypto` 16 |
| `rsa-2048` / `verify-pkcs1v15-sha256` | 9 | 6/0/3 | 1.02x | 1.21x | `aws-lc-rs` 3 |
| `rsa-2048` / `verify-pss-sha256` | 9 | 6/0/3 | 1.02x | 1.21x | `aws-lc-rs` 3 |
| `blake2` / `streaming` | 54 | 17/37/0 | 1.04x | 1.01x | `rustcrypto` 37 |
| `rapidhash-v3-64` / `hash` | 99 | 27/49/23 | 1.05x | 1.00x | `rapidhash` 72 |
| `ed25519` / `sign` | 36 | 12/22/2 | 1.06x | 1.04x | `aws-lc-rs` 21, `dalek` 2, `dryoc` 1 |
| `rapidhash-v3-128` / `hash` | 99 | 31/47/21 | 1.07x | 1.01x | `rapidhash` 68 |
| `blake2` / `keyed` | 396 | 226/170/0 | 1.08x | 1.07x | `rustcrypto` 170 |
| `blake2` / `hash` | 396 | 240/154/2 | 1.11x | 1.09x | `rustcrypto` 156 |

## Apple Silicon Clear Losses

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | --- | --- | --- | --- | --- |
| `chacha20-poly1305` / `encrypt` | 11 | 1/0/10 | 0.83x | 0.80x | `ring` 8, `aws-lc-rs` 2 |
| `chacha20-poly1305` / `decrypt` | 11 | 1/1/9 | 0.88x | 0.83x | `ring` 7, `aws-lc-rs` 3 |
| `sha3-256` / `streaming` | 2 | 0/0/2 | 0.94x | 0.94x | `sha3` 2 |
| `sha3-256` / `hash` | 11 | 0/3/8 | 0.94x | 0.93x | `sha3` 11 |
| `hkdf-sha256` / `expand` | 4 | 0/3/1 | 0.96x | 0.97x | `rustcrypto` 4 |
| `sha256` / `hash` | 11 | 3/2/6 | 0.96x | 0.89x | `ring` 6, `sha2` 1, `aws-lc-rs` 1 |
| `xxh3-64` / `hash` | 11 | 1/3/7 | 0.97x | 0.85x | `xxhash-rust` 10 |
| `sha3-512` / `hash` | 11 | 0/8/3 | 0.97x | 0.99x | `sha3` 11 |

## Linux Worst Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `pbkdf2-sha256` / `iters=1` / `32` | `aws-lc-rs` | 0.03x |
| Intel Sapphire Rapids | `pbkdf2-sha256` / `iters=1` / `64` | `aws-lc-rs` | 0.04x |
| AWS Graviton4 | `rsa-8192` / `verify-pkcs1v15-sha256` | `aws-lc-rs` | 0.32x |
| AWS Graviton4 | `rsa-8192` / `verify-pss-sha256` | `aws-lc-rs` | 0.32x |
| AWS Graviton3 | `rsa-8192` / `verify-pkcs1v15-sha256` | `aws-lc-rs` | 0.33x |
| AWS Graviton3 | `rsa-8192` / `verify-pss-sha256` | `aws-lc-rs` | 0.33x |
| AWS Graviton3 | `rsa-4096` / `verify-pkcs1v15-sha256` | `aws-lc-rs` | 0.36x |
| AWS Graviton3 | `rsa-4096` / `verify-pss-sha256` | `aws-lc-rs` | 0.36x |
| Intel Ice Lake | `scrypt-small` / `password-hash` / `log_n=10_r=8_p=1` | `rustcrypto` | 0.37x |
| AWS Graviton3 | `rsa-3072` / `verify-pkcs1v15-sha256` | `aws-lc-rs` | 0.37x |
| Intel Ice Lake | `scrypt-small` / `password-hash` / `log_n=10_r=8_p=4` | `rustcrypto` | 0.37x |
| AWS Graviton4 | `rsa-4096` / `verify-pkcs1v15-sha256` | `aws-lc-rs` | 0.38x |

## Linux Strongest Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt` / `hash` / `16384` | `crc` | 229.74x |
| Intel Sapphire Rapids | `crc16-ibm` / `hash` / `262144` | `crc` | 215.26x |
| Intel Sapphire Rapids | `crc16-ccitt` / `hash` / `262144` | `crc` | 211.09x |
| Intel Sapphire Rapids | `crc16-ibm` / `hash` / `16384` | `crc` | 198.94x |
| Intel Sapphire Rapids | `crc16-ibm` / `hash` / `1048576` | `crc` | 186.82x |
| Intel Sapphire Rapids | `crc16-ccitt` / `hash` / `1048576` | `crc` | 181.52x |
| Intel Sapphire Rapids | `crc16-ibm` / `hash` / `65536` | `crc` | 180.39x |
| IBM Power10 | `crc16-ccitt` / `hash` / `1048576` | `crc` | 176.72x |
| IBM Power10 | `crc16-ibm` / `hash` / `1048576` | `crc` | 176.10x |
| IBM Power10 | `crc16-ccitt` / `hash` / `262144` | `crc` | 175.42x |
| IBM Power10 | `crc16-ibm` / `hash` / `262144` | `crc` | 175.34x |
| Intel Sapphire Rapids | `crc16-ccitt` / `hash` / `65536` | `crc` | 175.30x |

## Top Five Loss Areas

1. **RSA verification on Arm/RISC-V Linux:** RSA-8192 verify is 0.90x..0.91x geomean, RSA-4096 verify is 0.93x, and the worst Graviton3/4 rows against `aws-lc-rs` are 0.32x..0.38x. This is the highest-priority asymmetric gap.
2. **X25519/Ed25519 derivation on Linux:** X25519 public-key derivation is 0.85x, X25519 DH is 0.92x, and Ed25519 keypair/public-key derivation is 0.91x..0.92x. Apple Silicon is near parity here, so the pressure is mainly Linux Arm/RISC-V/x86 path quality.
3. **Apple Silicon ChaCha20-Poly1305:** encrypt is 0.83x and decrypt is 0.88x, with 19 losses across 22 fastest-external rows. This is the clearest macOS AEAD gap.
4. **Password/KDF front-end work:** Linux `scrypt-small` is 0.90x and `scrypt-owasp` is 0.93x; PBKDF2-SHA256 `iters=1` is 0.81x, with Intel Sapphire Rapids 32-byte and 64-byte rows at 0.03x..0.04x against `aws-lc-rs`. High-iteration PBKDF2 recovers, so that part points at setup overhead.
5. **Apple Silicon hash/checksum holes:** SHA3-256 hash/streaming is 0.94x, SHA256 hash is 0.96x, XXH3-64 is 0.97x, and HMAC-SHA256 has individual bulk rows down to 0.87x. These are not catastrophic, but they keep Apple Silicon from looking dominant.

## External Pressure

All-pair Linux CI comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `rapidhash` | 396 | 128/205/63 | 32% | 1.09x | 1.01x |
| `xxhash-rust` | 198 | 103/68/27 | 52% | 1.18x | 1.07x |
| `aws-lc-rs` | 1,498 | 866/332/300 | 58% | 1.21x | 1.10x |
| `dalek` | 108 | 75/20/13 | 69% | 1.31x | 1.14x |
| `ascon-hash` | 198 | 128/68/2 | 65% | 1.32x | 1.25x |
| `ascon-aead` | 198 | 149/49/0 | 75% | 1.33x | 1.15x |
| `aegis-crate` | 198 | 90/83/25 | 45% | 1.38x | 1.03x |
| `sha2` | 531 | 285/242/4 | 54% | 1.48x | 1.06x |
| `ring` | 1,512 | 1145/201/166 | 76% | 1.52x | 1.23x |
| `tiny-keccak` | 396 | 272/102/22 | 69% | 1.80x | 1.99x |
| `dryoc` | 360 | 328/26/6 | 91% | 1.86x | 1.81x |
| `rustcrypto` | 2,664 | 1910/602/152 | 72% | 1.91x | 1.17x |
| `crc-fast` | 198 | 118/62/18 | 60% | 2.12x | 1.16x |
| `sha3` | 414 | 366/48/0 | 88% | 2.14x | 2.11x |
| `crc32fast` | 99 | 82/3/14 | 83% | 2.44x | 2.01x |
| `crc64fast` | 99 | 75/12/12 | 76% | 2.44x | 2.11x |
| `crc64fast-nvme` | 99 | 77/15/7 | 78% | 2.45x | 2.20x |
| `rustcrypto-rsa` | 81 | 81/0/0 | 100% | 4.88x | 3.83x |
| `crc` | 297 | 288/4/5 | 97% | 19.38x | 29.82x |

## README Numbers

- **Headline:** 3,627 of 5,994 matched Linux CI fastest-external comparisons are wins; 5,345 are wins or ties. Linux CI geomean is 1.60x.
- **Apple Silicon headline:** 368 of 702 matched fastest-external comparisons are wins; 643 are wins or ties. Apple Silicon geomean is 1.39x.
- **Checksums:** 2.62x geomean across 1,188 Linux CI fastest-external rows; W/T/L 693/351/144.
- **Hashes/MACs/XOFs:** 1.40x geomean across 2,700 Linux CI fastest-external rows; W/T/L 1634/969/97.
- **Auth/KDF:** 1.17x geomean across 180 Linux CI fastest-external rows; W/T/L 160/16/4.
- **Password hashing:** 0.97x geomean across 135 Linux CI fastest-external rows; W/T/L 62/15/58.
- **Public-key:** 0.99x geomean across 108 Linux CI fastest-external rows; W/T/L 29/53/26.
- **RSA:** 1.30x geomean across 99 Linux CI fastest-external rows; W/T/L 75/0/24.
- **AEAD:** 1.56x geomean across 1,584 Linux CI fastest-external rows; W/T/L 974/314/296.
- **Apple Silicon AEAD:** 1.44x geomean across 176 fastest-external rows; ChaCha20-Poly1305 is the local loss at 0.86x.
- **Current top losses:** RSA verification on Arm/RISC-V Linux, X25519/Ed25519 derivation on Linux, Apple Silicon ChaCha20-Poly1305, password/KDF front-end overhead, and Apple Silicon SHA/XXH3 holes.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-06-08 20_21_15` | 2,096 | `benchmark_results/2026-06-08/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-06-08 20_21_15` | 2,096 | `benchmark_results/2026-06-08/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-06-08 20_21_15` | 2,096 | `benchmark_results/2026-06-08/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-06-08 20_21_15` | 2,096 | `benchmark_results/2026-06-08/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-06-08 20_21_15` | 1,872 | `benchmark_results/2026-06-08/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-06-08 20_21_15` | 1,872 | `benchmark_results/2026-06-08/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-06-08 20_21_15` | 2,096 | `benchmark_results/2026-06-08/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-06-08 20_21_15` | 2,096 | `benchmark_results/2026-06-08/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-06-08 20_21_15` | 2,096 | `benchmark_results/2026-06-08/linux/rise-riscv/results.txt` |
| Apple Silicon MBP M1 | `local` | `2026-06-08 18_48_04` | 2,171 | `benchmark_results/2026-06-08/macos/aarch64/results.txt` |
