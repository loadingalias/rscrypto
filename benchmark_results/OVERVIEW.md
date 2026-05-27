# Benchmark Overview

Sources:

- Linux benchmark CI run [#26479146090](https://github.com/loadingalias/rscrypto/actions/runs/26479146090), created 2026-05-26 22:37:20 UTC.
- Commit: `8e6403872c481503ef0513c1a3abe6f938aa8f84`.

Scope: current nine-runner Linux CI benchmark matrix for commit `8e64038`. Ratios are `rscrypto` throughput divided by external throughput for throughput rows. For latency-only rows, ratios are external time divided by `rscrypto` time. Higher is always better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and size.

Important coverage note: this run uses the full `bench=all` corpus on all nine Linux runners and includes RSA verification/import rows in addition to checksums, hashes, MACs, KDFs, password hashing, key exchange, signatures, and AEAD. IBM POWER10 and IBM Z exclude `aws-lc-rs` rows because that dependency is not enabled for those target architectures.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| All matched performance pairs | 10,498 | 7189/2335/974 | 68% | 1.71x | 1.19x |
| Fastest external per case | 6,651 | 3984/1900/767 | 60% | 1.53x | 1.10x |
| All matched throughput pairs | 9,123 | 6181/2208/734 | 68% | 1.77x | 1.18x |
| Fastest external, throughput only | 5,796 | 3438/1795/563 | 59% | 1.62x | 1.09x |
| All matched latency-only pairs | 1,375 | 1008/127/240 | 73% | 1.35x | 1.26x |
| Fastest external, latency only | 855 | 546/105/204 | 64% | 1.04x | 1.14x |

What matters:

- Linux CI fastest-external scorecard: 3,984 wins / 1,900 ties / 767 losses across 6,651 comparisons, with a 1.53x geomean.
- Checksums: 5.05x fastest-external geomean across 693 rows (525/114/54 W/T/L).
- Hashes/MACs: 1.35x fastest-external geomean across 3,735 rows (2183/1343/209 W/T/L).
- Auth/KDF/password: 1.14x fastest-external geomean across 738 rows (390/202/146 W/T/L).
- RSA: 0.20x fastest-external geomean across 99 rows (39/1/59 W/T/L).
- AEAD: 1.59x fastest-external geomean across 1,386 rows (847/240/299 W/T/L).
- `rsa-2048`: 0.20x fastest-external geomean across 27 rows (12/0/15).
- `rsa-3072`: 0.15x fastest-external geomean across 27 rows (12/0/15).
- `rsa-4096`: 0.13x fastest-external geomean across 27 rows (11/1/15).
- `rsa-8192`: 0.58x fastest-external geomean across 18 rows (4/0/14).
- `ed25519`: 1.15x fastest-external geomean across 90 rows (44/33/13).
- `x25519`: 0.96x fastest-external geomean across 18 rows (6/8/4).
- `pbkdf2-sha256`: 1.08x fastest-external geomean across 54 rows (46/4/4).
- `argon2id-small`: 0.86x fastest-external geomean across 27 rows (8/2/17).
- `scrypt-small`: 0.91x fastest-external geomean across 36 rows (20/0/16).
- `aes-128-gcm`: 1.75x fastest-external geomean across 198 rows (116/36/46).
- `aes-256-gcm`: 1.81x fastest-external geomean across 198 rows (109/43/46).
- `blake3`: 1.42x fastest-external geomean across 432 rows (226/169/37).
- `sha3-256`: 2.13x fastest-external geomean across 117 rows (103/14/0).
- `crc32c`: 1.59x fastest-external geomean across 99 rows (50/34/15).

## Coverage Matrix

| Platform | Checksums | Hashes/MACs | Auth/KDF/password | RSA | AEAD | Parsed rows |
| --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 187 | 993 | 375 | 71 | 484 | 2,110 |
| AMD Zen5 | 187 | 993 | 375 | 71 | 484 | 2,110 |
| AWS Graviton3 | 187 | 993 | 375 | 71 | 484 | 2,110 |
| AWS Graviton4 | 187 | 993 | 375 | 71 | 484 | 2,110 |
| IBM Power10 | 187 | 960 | 303 | 63 | 374 | 1,887 |
| IBM z16/s390x | 187 | 960 | 303 | 63 | 374 | 1,887 |
| Intel Ice Lake | 187 | 993 | 375 | 71 | 484 | 2,110 |
| Intel Sapphire Rapids | 187 | 993 | 375 | 71 | 484 | 2,110 |
| RISE RISC-V | 187 | 993 | 375 | 71 | 484 | 2,110 |

## Platform Summary

| Platform | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 739 | 517/151/71 | 70% | 1.40x | 1.12x |
| AMD Zen5 | 739 | 425/233/81 | 58% | 1.40x | 1.07x |
| AWS Graviton3 | 739 | 364/274/101 | 49% | 1.32x | 1.05x |
| AWS Graviton4 | 739 | 369/305/65 | 50% | 1.33x | 1.05x |
| IBM Power10 | 739 | 393/297/49 | 53% | 1.84x | 1.07x |
| IBM z16/s390x | 739 | 591/84/64 | 80% | 3.02x | 2.26x |
| Intel Ice Lake | 739 | 489/158/92 | 66% | 1.41x | 1.15x |
| Intel Sapphire Rapids | 739 | 499/142/98 | 68% | 1.54x | 1.16x |
| RISE RISC-V | 739 | 337/256/146 | 46% | 1.08x | 1.03x |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. The fastest-only columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `chacha20-poly1305` | 198 | 69/56/73 | 0.99x | 550 | 301/129/120 | 1.15x |
| `xchacha20-poly1305` | 198 | 168/29/1 | 1.38x | 198 | 168/29/1 | 1.38x |
| `aegis-256` | 198 | 109/67/22 | 1.46x | 198 | 109/67/22 | 1.46x |
| `aes-128-gcm` | 198 | 116/36/46 | 1.75x | 550 | 430/46/74 | 2.54x |
| `aes-256-gcm` | 198 | 109/43/46 | 1.81x | 550 | 421/51/78 | 2.63x |
| `aes-128-gcm-siv` | 198 | 138/3/57 | 1.94x | 352 | 273/15/64 | 2.74x |
| `aes-256-gcm-siv` | 198 | 138/6/54 | 2.08x | 352 | 291/7/54 | 3.01x |
| `argon2d-small` | 27 | 8/4/15 | 0.86x | 27 | 8/4/15 | 0.86x |
| `argon2id-small` | 27 | 8/2/17 | 0.86x | 45 | 25/2/18 | 1.27x |
| `argon2i-small` | 27 | 8/3/16 | 0.86x | 45 | 25/3/17 | 1.25x |
| `scrypt-small` | 36 | 20/0/16 | 0.91x | 36 | 20/0/16 | 0.91x |
| `scrypt-owasp` | 9 | 5/0/4 | 0.94x | 9 | 5/0/4 | 0.94x |
| `argon2id-owasp` | 9 | 2/1/6 | 0.96x | 18 | 9/2/7 | 1.23x |
| `x25519` | 18 | 6/8/4 | 0.96x | 50 | 38/8/4 | 1.42x |
| `pbkdf2-sha256` | 54 | 46/4/4 | 1.08x | 150 | 137/6/7 | 1.36x |
| `ed25519` | 90 | 44/33/13 | 1.15x | 290 | 218/56/16 | 1.41x |
| `hmac-sha256` | 117 | 29/52/36 | 1.17x | 293 | 132/105/56 | 1.42x |
| `pbkdf2-sha512` | 54 | 46/8/0 | 1.24x | 150 | 141/9/0 | 1.30x |
| `hkdf-sha384` | 36 | 35/1/0 | 1.27x | 100 | 99/1/0 | 1.59x |
| `hmac-sha512` | 99 | 48/43/8 | 1.27x | 275 | 178/84/13 | 1.31x |
| `hmac-sha384` | 99 | 49/43/7 | 1.28x | 275 | 178/85/12 | 1.32x |
| `hkdf-sha256` | 36 | 36/0/0 | 1.28x | 100 | 100/0/0 | 1.86x |
| `crc32c` | 99 | 50/34/15 | 1.59x | 198 | 149/34/15 | 2.21x |
| `crc32` | 99 | 49/33/17 | 1.67x | 198 | 146/35/17 | 2.31x |
| `crc64-nvme` | 99 | 60/32/7 | 1.79x | 198 | 142/46/10 | 2.35x |
| `crc64-xz` | 99 | 76/12/11 | 2.39x | 99 | 76/12/11 | 2.39x |
| `crc24-openpgp` | 99 | 93/3/3 | 13.08x | 99 | 93/3/3 | 13.08x |
| `crc16-ccitt` | 99 | 98/0/1 | 23.34x | 99 | 98/0/1 | 23.34x |
| `crc16-ibm` | 99 | 99/0/0 | 24.08x | 99 | 99/0/0 | 24.08x |
| `rapidhash-v3-64` | 99 | 23/53/23 | 1.05x | 99 | 23/53/23 | 1.05x |
| `rapidhash-v3-128` | 99 | 35/41/23 | 1.07x | 99 | 35/41/23 | 1.07x |
| `blake2/blake2b512` | 198 | 112/81/5 | 1.08x | 396 | 303/88/5 | 1.54x |
| `blake2/blake2b256` | 387 | 248/129/10 | 1.09x | 612 | 457/144/11 | 1.39x |
| `blake2/blake2s256` | 387 | 237/143/7 | 1.11x | 387 | 237/143/7 | 1.11x |
| `blake2/blake2s128` | 198 | 118/79/1 | 1.12x | 198 | 118/79/1 | 1.12x |
| `rapidhash-64` | 99 | 33/58/8 | 1.13x | 99 | 33/58/8 | 1.13x |
| `rapidhash-128` | 99 | 37/56/6 | 1.14x | 99 | 37/56/6 | 1.14x |
| `xxh3-128` | 99 | 51/34/14 | 1.17x | 99 | 51/34/14 | 1.17x |
| `xxh3-64` | 99 | 53/33/13 | 1.18x | 99 | 53/33/13 | 1.18x |
| `sha256` | 117 | 44/57/16 | 1.26x | 293 | 164/99/30 | 1.48x |
| `ascon-hash256` | 99 | 55/44/0 | 1.27x | 99 | 55/44/0 | 1.27x |
| `sha512` | 117 | 49/57/11 | 1.27x | 293 | 173/103/17 | 1.29x |
| `sha384` | 99 | 46/43/10 | 1.29x | 275 | 174/85/16 | 1.31x |
| `sha512-256` | 99 | 54/45/0 | 1.32x | 99 | 54/45/0 | 1.32x |
| `ascon-xof128` | 99 | 75/22/2 | 1.37x | 99 | 75/22/2 | 1.37x |
| `blake3` | 432 | 226/169/37 | 1.42x | 432 | 226/169/37 | 1.42x |
| `kmac256` | 99 | 59/26/14 | 1.75x | 99 | 59/26/14 | 1.75x |
| `cshake256` | 99 | 62/32/5 | 1.82x | 99 | 62/32/5 | 1.82x |
| `shake128` | 99 | 76/23/0 | 1.85x | 99 | 76/23/0 | 1.85x |
| `shake256` | 99 | 70/29/0 | 1.87x | 99 | 70/29/0 | 1.87x |
| `sha224` | 99 | 57/40/2 | 1.93x | 99 | 57/40/2 | 1.93x |
| `sha3-256` | 117 | 103/14/0 | 2.13x | 117 | 103/14/0 | 2.13x |
| `sha3-384` | 99 | 86/12/1 | 2.16x | 99 | 86/12/1 | 2.16x |
| `sha3-224` | 99 | 87/11/1 | 2.18x | 99 | 87/11/1 | 2.18x |
| `sha3-512` | 99 | 87/12/0 | 2.26x | 99 | 87/12/0 | 2.26x |
| `rsa-4096` | 27 | 11/1/15 | 0.13x | 59 | 39/1/19 | 0.69x |
| `rsa-3072` | 27 | 12/0/15 | 0.15x | 59 | 40/0/19 | 0.75x |
| `rsa-2048` | 27 | 12/0/15 | 0.20x | 59 | 40/0/19 | 0.86x |
| `rsa-8192` | 18 | 4/0/14 | 0.58x | 32 | 6/0/26 | 0.55x |

## Clear Losses

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | --- | --- | --- | --- | --- |
| `rsa-4096` / `parse-spki` | 9 | 0/0/9 | <0.01x | <0.01x | `rustcrypto-rsa` 9 |
| `rsa-3072` / `parse-spki` | 9 | 0/0/9 | <0.01x | <0.01x | `rustcrypto-rsa` 9 |
| `rsa-2048` / `parse-spki` | 9 | 0/0/9 | <0.01x | <0.01x | `rustcrypto-rsa` 9 |
| `rsa-8192` / `verify-pkcs1v15-sha256` | 9 | 2/0/7 | 0.57x | 0.37x | `aws-lc-rs` 7, `ring` 2 |
| `rsa-8192` / `verify-pss-sha256` | 9 | 2/0/7 | 0.58x | 0.37x | `aws-lc-rs` 7, `ring` 2 |
| `pbkdf2-sha256` / `iters=1` | 18 | 13/3/2 | 0.83x | 1.14x | `rustcrypto` 16, `aws-lc-rs` 2 |
| `argon2d-small` / `hash` | 27 | 8/4/15 | 0.86x | 0.93x | `rustcrypto` 27 |
| `argon2id-small` / `hash` | 27 | 8/2/17 | 0.86x | 0.90x | `rustcrypto` 27 |
| `argon2i-small` / `hash` | 27 | 8/3/16 | 0.86x | 0.86x | `rustcrypto` 27 |
| `scrypt-small` / `hash` | 36 | 20/0/16 | 0.91x | 1.23x | `rustcrypto` 36 |
| `rsa-4096` / `verify-pss-sha256` | 9 | 6/0/3 | 0.93x | 1.13x | `aws-lc-rs` 7, `ring` 1, `rustcrypto-rsa` 1 |
| `scrypt-owasp` / `hash` | 9 | 5/0/4 | 0.94x | 1.13x | `rustcrypto` 9 |
| `rsa-4096` / `verify-pkcs1v15-sha256` | 9 | 5/1/3 | 0.94x | 1.13x | `aws-lc-rs` 7, `ring` 1, `rustcrypto-rsa` 1 |
| `x25519` / `diffie-hellman` | 9 | 3/4/2 | 0.95x | 1.01x | `aws-lc-rs` 7, `dryoc` 2 |
| `argon2id-owasp` / `hash` | 9 | 2/1/6 | 0.96x | 0.91x | `rustcrypto` 9 |
| `chacha20-poly1305` / `encrypt` | 99 | 30/33/36 | 0.98x | 1.00x | `ring` 75, `aws-lc-rs` 15, `rustcrypto` 9 |
| `x25519` / `public-key-from-secret` | 9 | 3/4/2 | 0.98x | 1.01x | `aws-lc-rs` 7, `dryoc` 1, `dalek` 1 |
| `rsa-3072` / `verify-pkcs1v15-sha256` | 9 | 6/0/3 | 0.98x | 1.15x | `aws-lc-rs` 7, `ring` 1, `rustcrypto-rsa` 1 |
| `rsa-3072` / `verify-pss-sha256` | 9 | 6/0/3 | 0.99x | 1.14x | `aws-lc-rs` 7, `ring` 1, `rustcrypto-rsa` 1 |
| `chacha20-poly1305` / `decrypt` | 99 | 39/23/37 | 0.99x | 1.00x | `ring` 52, `aws-lc-rs` 39, `rustcrypto` 8 |

## RSA

| Modulus | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `rsa-2048` | 27 | 12/0/15 | 0.20x | 0.82x |
| `rsa-3072` | 27 | 12/0/15 | 0.15x | 0.78x |
| `rsa-4096` | 27 | 11/1/15 | 0.13x | 0.72x |
| `rsa-8192` | 18 | 4/0/14 | 0.58x | 0.37x |

| Operation | Rows | W/T/L | Geomean | Median | Fastest external pressure |
| --- | --- | --- | --- | --- | --- |
| `parse-spki` | 27 | 0/0/27 | <0.01x | <0.01x | `rustcrypto-rsa` 27 |
| `verify-pkcs1v15-sha256` | 36 | 19/1/16 | 0.86x | 1.09x | `aws-lc-rs` 28, `ring` 5, `rustcrypto-rsa` 3 |
| `verify-pss-sha256` | 36 | 20/0/16 | 0.86x | 1.10x | `aws-lc-rs` 28, `ring` 5, `rustcrypto-rsa` 3 |

RSA diagnostic/internal rows parsed but excluded from headline fastest-external claims: 414. This includes public-exponent, hash-component, one-shot/cold verification, scratch setup, and Montgomery diagnostic rows where there is no shape-compatible external competitor.

## Non-Crypto Hash Truth Split

| Comparison | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `rapidhash-128` | 99 | 37/56/6 | 1.14x | 1.02x |
| `rapidhash-64` | 99 | 33/58/8 | 1.13x | 1.01x |
| `rapidhash-v3-128` | 99 | 35/41/23 | 1.07x | 1.00x |
| `rapidhash-v3-64` | 99 | 23/53/23 | 1.05x | 1.00x |
| `xxh3-128` | 99 | 51/34/14 | 1.17x | 1.06x |
| `xxh3-64` | 99 | 53/33/13 | 1.18x | 1.07x |

## Sustained AEAD

Fastest-external throughput comparisons only, sizes `65536`, `262144`, and `1048576`, both encrypt and decrypt, 128-bit and 256-bit keys where applicable.

| Platform | AES-GCM W/T/L | AES-GCM geomean | GCM-SIV W/T/L | GCM-SIV geomean | ChaCha20-Poly1305 W/T/L | ChaCha20-Poly1305 geomean |
| --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 0/12/0 | 1.00x | 12/0/0 | 1.75x | 0/0/6 | 0.93x |
| AMD Zen5 | 1/7/4 | 0.98x | 12/0/0 | 2.05x | 6/0/0 | 1.17x |
| AWS Graviton3 | 0/0/12 | 0.91x | 12/0/0 | 2.54x | 0/6/0 | 1.00x |
| AWS Graviton4 | 0/10/2 | 0.97x | 12/0/0 | 2.54x | 0/6/0 | 1.00x |
| IBM Power10 | 12/0/0 | 13.55x | 12/0/0 | 8.77x | 6/0/0 | 1.14x |
| IBM z16/s390x | 12/0/0 | 12.13x | 12/0/0 | 6.45x | 6/0/0 | 2.03x |
| Intel Ice Lake | 1/11/0 | 1.01x | 12/0/0 | 1.37x | 4/2/0 | 1.05x |
| Intel Sapphire Rapids | 0/8/4 | 0.95x | 12/0/0 | 1.47x | 4/2/0 | 1.06x |
| RISE RISC-V | 0/0/12 | 0.82x | 0/0/12 | 0.89x | 0/0/6 | 0.93x |

## External Pressure

All-pair Linux CI comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `rustcrypto-rsa` | 81 | 54/0/27 | 67% | 0.55x | 3.78x |
| `rapidhash` | 396 | 128/208/60 | 32% | 1.10x | 1.01x |
| `xxhash-rust` | 198 | 104/67/27 | 53% | 1.18x | 1.07x |
| `aws-lc-rs` | 1,498 | 874/319/305 | 58% | 1.20x | 1.10x |
| `ascon-hash` | 198 | 130/66/2 | 66% | 1.32x | 1.25x |
| `blake3` | 432 | 226/169/37 | 52% | 1.42x | 1.06x |
| `aegis-crate` | 198 | 109/67/22 | 55% | 1.46x | 1.06x |
| `dalek` | 108 | 92/13/3 | 85% | 1.49x | 1.44x |
| `ring` | 1,512 | 1118/195/199 | 74% | 1.52x | 1.21x |
| `sha2` | 531 | 295/229/7 | 56% | 1.52x | 1.06x |
| `rustcrypto` | 2,988 | 2111/677/200 | 71% | 1.78x | 1.15x |
| `tiny-keccak` | 396 | 267/110/19 | 67% | 1.82x | 1.98x |
| `dryoc` | 558 | 515/36/7 | 92% | 2.01x | 1.88x |
| `crc-fast` | 297 | 185/92/20 | 62% | 2.07x | 1.20x |
| `sha3` | 414 | 363/49/2 | 88% | 2.18x | 2.12x |
| `crc64fast` | 99 | 76/12/11 | 77% | 2.39x | 2.12x |
| `crc32fast` | 99 | 83/7/9 | 84% | 2.44x | 2.05x |
| `crc64fast-nvme` | 99 | 78/14/7 | 79% | 2.47x | 2.17x |
| `crc32c` | 99 | 91/2/6 | 92% | 2.69x | 2.17x |
| `crc` | 297 | 290/3/4 | 98% | 19.45x | 30.11x |

## README Numbers

- **Headline:** 3,984 of 6,651 matched Linux CI fastest-external comparisons are wins; Linux CI geomean is 1.53x.
- **Checksums:** 5.05x geomean across Linux CI fastest-external checksum comparisons.
- **SHA-3 / SHAKE:** 2.18x SHA-3 geomean and 1.86x SHAKE geomean across Linux CI fastest-external comparisons.
- **BLAKE3:** 2.33x geomean for Linux CI fastest-external rows at `>=64 KiB`.
- **AEAD:** 1.59x geomean across Linux CI fastest-external AEAD comparisons.
- **RSA:** 0.20x geomean across Linux CI fastest-external RSA import and verification comparisons.
- **Ed25519 / X25519:** 1.12x Ed25519 sign geomean, 1.00x Ed25519 verify geomean, and 0.96x X25519 geomean across Linux CI fastest-external comparisons.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-05-26 22_37_20` | 2,110 | `benchmark_results/2026-05-26/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-05-26 22_37_20` | 2,110 | `benchmark_results/2026-05-26/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-05-26 22_37_20` | 2,110 | `benchmark_results/2026-05-26/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-05-26 22_37_20` | 2,110 | `benchmark_results/2026-05-26/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-05-26 22_37_20` | 1,887 | `benchmark_results/2026-05-26/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-05-26 22_37_20` | 1,887 | `benchmark_results/2026-05-26/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-05-26 22_37_20` | 2,110 | `benchmark_results/2026-05-26/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-05-26 22_37_20` | 2,110 | `benchmark_results/2026-05-26/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-05-26 22_37_20` | 2,110 | `benchmark_results/2026-05-26/linux/rise-riscv/results.txt` |
