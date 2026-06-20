# Benchmark Overview

Sources:

- Linux benchmark CI run [#27805069694](https://github.com/loadingalias/rscrypto/actions/runs/27805069694), created 2026-06-19 04:17:15 UTC.
- Linux commit: `446e3d46fc2b3298b79d502904bab4bb0131e1c8`.
- Artifacts: nine successful `benchmark-*` artifacts copied into `benchmark_results/2026-06-19/linux/*/results.txt`.

Scope: the 2026-06-19 nine-runner Linux CI benchmark matrix for commit `446e3d4`. Ratios are `external_crate_time / rscrypto_time`; higher is better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and input shape. Internal kernel, scratch-buffer, padding-only, cold-path, PHC roundtrip, parallel-scaling, threshold-selection, and public-overhead microbenches are parsed as raw rows but excluded from external win/loss claims.

Coverage note: this is a full Linux CI benchmark pass. It includes checksum, hash, XOF, MAC, KDF, password-hashing, BLAKE2/BLAKE3, RSA import/verification, ECDSA P-256/P-384 signing and verification, Ed25519, X25519, AEAD, and ML-KEM-512/768/1024 keygen, encapsulation, and decapsulation rows.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Linux CI: all matched performance pairs | 10,781 | 7527/2360/894 | 70% | 1.78x | 1.22x |
| Linux CI: fastest external per case | 6,750 | 4102/1954/694 | 61% | 1.58x | 1.11x |

Shareable release summary:

- **Headline:** 4,102 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,056 are wins or ties. Linux CI fastest-external geomean is 1.58x.
- **Checksums:** 5.20x geomean across 693 fastest-external rows; W/T/L is 514/125/54.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,726 fastest-external rows; W/T/L is 2110/1392/224.
- **Auth/KDF:** 1.23x geomean across 180 fastest-external rows; W/T/L is 165/15/0.
- **Password hashing:** 1.11x geomean across 135 fastest-external rows; W/T/L is 66/29/40.
- **Public-key:** 1.12x geomean across 333 fastest-external rows; W/T/L is 188/74/71.
- **RSA:** 1.54x geomean across 99 fastest-external rows; W/T/L is 85/6/8.
- **AEAD:** 1.56x geomean across 1,584 fastest-external rows; W/T/L is 974/313/297.
- **ML-KEM:** 0.78x geomean across 81 fastest-external rows; W/T/L is 37/7/37.
- **ECDSA P-256/P-384:** Linux CI 1.40x geomean across 144 fastest-external rows; W/T/L is 115/9/20.
- **Top current loss areas:** `mlkem512` / `keygen`: 0.69x geomean across 9 rows; W/T/L 1/2/6; pressure `aws-lc-rs` 6, `libcrux` 2; `mlkem1024` / `keygen`: 0.70x geomean across 9 rows; W/T/L 3/1/5; pressure `aws-lc-rs` 3, `libcrux` 3; `mlkem768` / `keygen`: 0.73x geomean across 9 rows; W/T/L 3/2/4; pressure `aws-lc-rs` 3, `libcrux` 3; `mlkem1024` / `decapsulate`: 0.75x geomean across 9 rows; W/T/L 5/0/4; pressure `aws-lc-rs` 3, `libcrux` 1; `mlkem768` / `decapsulate`: 0.75x geomean across 9 rows; W/T/L 5/0/4; pressure `aws-lc-rs` 3, `libcrux` 1.

## Coverage Matrix

| Platform | Raw Criterion rows | All pairs | Fastest rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 2,202 | 1,251 | 750 | 526/158/66 | 70% | 1.48x | 1.14x |
| AMD Zen5 | 2,202 | 1,251 | 750 | 444/235/71 | 59% | 1.49x | 1.09x |
| AWS Graviton3 | 2,202 | 1,251 | 750 | 373/285/92 | 50% | 1.37x | 1.05x |
| AWS Graviton4 | 2,202 | 1,251 | 750 | 371/324/55 | 49% | 1.38x | 1.05x |
| IBM Power10 | 1,953 | 1,012 | 750 | 414/292/44 | 55% | 1.93x | 1.07x |
| IBM z16/s390x | 1,953 | 1,012 | 750 | 600/91/59 | 80% | 3.04x | 2.34x |
| Intel Ice Lake | 2,202 | 1,251 | 750 | 515/152/83 | 69% | 1.48x | 1.19x |
| Intel Sapphire Rapids | 2,202 | 1,251 | 750 | 527/150/73 | 70% | 1.62x | 1.22x |
| RISE RISC-V | 2,202 | 1,251 | 750 | 332/267/151 | 44% | 1.09x | 1.02x |

## Category Summary

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 693 | 514/125/54 | 74% | 5.20x | 2.48x |
| Hashes/MACs/XOFs | 3,726 | 2110/1392/224 | 57% | 1.35x | 1.08x |
| Auth/KDF | 180 | 165/15/0 | 92% | 1.23x | 1.11x |
| Password hashing | 135 | 66/29/40 | 49% | 1.11x | 1.01x |
| Public-key | 333 | 188/74/71 | 56% | 1.12x | 1.09x |
| RSA | 99 | 85/6/8 | 86% | 1.54x | 1.16x |
| AEAD | 1,584 | 974/313/297 | 61% | 1.56x | 1.14x |

## ML-KEM Summary

ML-KEM coverage is complete for the CI-selected primitive set: ML-KEM-512, ML-KEM-768, and ML-KEM-1024 each include keygen, encapsulate, and decapsulate on all nine Linux platforms. POWER10 and s390x do not have `aws-lc-rs` ML-KEM rows in this artifact set, but still have rscrypto plus `libcrux`, `fips203`, and RustCrypto comparison rows for every operation.

| Platform | Raw ML-KEM rows | Fastest rows | W/T/L | Geomean | Median | Fastest external split |
| --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 45 | 9 | 8/0/1 | 1.17x | 1.13x | `libcrux` 7, `aws-lc-rs` 2 |
| AMD Zen5 | 45 | 9 | 8/1/0 | 1.22x | 1.14x | `libcrux` 9 |
| AWS Graviton3 | 45 | 9 | 0/0/9 | 0.64x | 0.65x | `aws-lc-rs` 9 |
| AWS Graviton4 | 45 | 9 | 0/0/9 | 0.62x | 0.64x | `aws-lc-rs` 9 |
| IBM Power10 | 36 | 9 | 7/1/1 | 1.14x | 1.11x | `libcrux` 9 |
| IBM z16/s390x | 36 | 9 | 0/0/9 | 0.13x | 0.12x | `libcrux` 9 |
| Intel Ice Lake | 45 | 9 | 5/3/1 | 1.14x | 1.09x | `libcrux` 7, `aws-lc-rs` 2 |
| Intel Sapphire Rapids | 45 | 9 | 8/1/0 | 1.21x | 1.16x | `aws-lc-rs` 6, `libcrux` 3 |
| RISE RISC-V | 45 | 9 | 1/1/7 | 0.88x | 0.85x | `aws-lc-rs` 9 |

| Primitive/op | Rows | W/T/L | Win % | Geomean | Median | Pressure |
| --- | --- | --- | --- | --- | --- | --- |
| `mlkem512` / `keygen` | 9 | 1/2/6 | 11% | 0.69x | 0.91x | `aws-lc-rs` 6, `libcrux` 2 |
| `mlkem512` / `encapsulate` | 9 | 6/0/3 | 67% | 0.90x | 1.31x | `aws-lc-rs` 2, `libcrux` 1 |
| `mlkem512` / `decapsulate` | 9 | 4/1/4 | 44% | 0.75x | 1.04x | `aws-lc-rs` 3, `libcrux` 2 |
| `mlkem768` / `keygen` | 9 | 3/2/4 | 33% | 0.73x | 1.01x | `aws-lc-rs` 3, `libcrux` 3 |
| `mlkem768` / `encapsulate` | 9 | 5/1/3 | 56% | 0.89x | 1.21x | `aws-lc-rs` 3, `libcrux` 1 |
| `mlkem768` / `decapsulate` | 9 | 5/0/4 | 56% | 0.75x | 1.09x | `aws-lc-rs` 3, `libcrux` 1 |
| `mlkem1024` / `keygen` | 9 | 3/1/5 | 33% | 0.70x | 0.94x | `aws-lc-rs` 3, `libcrux` 3 |
| `mlkem1024` / `encapsulate` | 9 | 5/0/4 | 56% | 0.86x | 1.11x | `aws-lc-rs` 3, `libcrux` 1 |
| `mlkem1024` / `decapsulate` | 9 | 5/0/4 | 56% | 0.75x | 1.11x | `aws-lc-rs` 3, `libcrux` 1 |

## ECDSA Summary

ECDSA signing includes both deterministic and blinded rscrypto rows in raw results; aggregate fastest-external comparisons use the fastest rscrypto row for the exact case. Constant-time release evidence is tracked separately by `ct.toml` and CT workflow artifacts.

| Operation | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `ecdsa-p256` / `sign` | 36 | 36/0/0 | 1.45x | 1.34x |
| `ecdsa-p256` / `verify` | 36 | 27/9/0 | 1.58x | 1.14x |
| `ecdsa-p384` / `sign` | 36 | 16/0/20 | 0.94x | 0.90x |
| `ecdsa-p384` / `verify` | 36 | 36/0/0 | 1.79x | 1.39x |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `mlkem1024` | 27 | 13/1/13 | 0.76x | 102 | 80/3/19 | 1.68x |
| `mlkem512` | 27 | 11/3/13 | 0.78x | 102 | 74/7/21 | 1.57x |
| `mlkem768` | 27 | 13/3/11 | 0.79x | 102 | 82/3/17 | 1.69x |
| `argon2id-owasp` | 9 | 3/2/4 | 0.97x | 18 | 10/3/5 | 1.25x |
| `chacha20-poly1305` | 198 | 73/56/69 | 1.00x | 550 | 302/130/118 | 1.18x |
| `argon2i-small` | 27 | 12/3/12 | 1.02x | 45 | 29/4/12 | 1.38x |
| `argon2id-small` | 27 | 12/3/12 | 1.02x | 45 | 29/4/12 | 1.40x |
| `argon2d-small` | 27 | 12/5/10 | 1.03x | 27 | 12/5/10 | 1.03x |
| `x25519` | 18 | 5/13/0 | 1.03x | 50 | 36/14/0 | 1.55x |
| `rapidhash-v3-64` | 99 | 27/52/20 | 1.05x | 99 | 27/52/20 | 1.05x |
| `blake2b256` | 225 | 127/96/2 | 1.07x | 351 | 245/104/2 | 1.33x |
| `rapidhash-v3-128` | 99 | 29/49/21 | 1.07x | 99 | 29/49/21 | 1.07x |
| `blake2b512` | 198 | 117/79/2 | 1.09x | 297 | 211/84/2 | 1.36x |
| `ed25519` | 90 | 31/45/14 | 1.10x | 290 | 200/63/27 | 1.34x |
| `blake2s256` | 225 | 120/103/2 | 1.10x | 225 | 120/103/2 | 1.10x |
| `rapidhash-64` | 99 | 33/55/11 | 1.12x | 99 | 33/55/11 | 1.12x |
| `blake2s128` | 198 | 123/75/0 | 1.13x | 198 | 123/75/0 | 1.13x |
| `rapidhash-128` | 99 | 39/51/9 | 1.14x | 99 | 39/51/9 | 1.14x |
| `xxh3-128` | 99 | 50/34/15 | 1.17x | 99 | 50/34/15 | 1.17x |
| `xxh3-64` | 99 | 52/35/12 | 1.18x | 99 | 52/35/12 | 1.18x |
| `rsa-8192` | 18 | 14/2/2 | 1.19x | 32 | 28/2/2 | 1.26x |
| `hkdf-sha256` | 36 | 34/2/0 | 1.22x | 100 | 98/2/0 | 1.81x |
| `sha256` | 117 | 42/60/15 | 1.22x | 293 | 161/103/29 | 1.45x |
| `pbkdf2-sha256` | 54 | 49/5/0 | 1.22x | 150 | 145/5/0 | 1.57x |
| `hkdf-sha384` | 36 | 35/1/0 | 1.24x | 100 | 99/1/0 | 1.55x |
| `hmac-sha512` | 99 | 43/48/8 | 1.24x | 275 | 173/89/13 | 1.29x |
| `pbkdf2-sha512` | 54 | 47/7/0 | 1.24x | 150 | 142/8/0 | 1.31x |
| `hmac-sha384` | 99 | 49/41/9 | 1.24x | 275 | 173/88/14 | 1.29x |
| `scrypt-owasp` | 9 | 5/2/2 | 1.24x | 9 | 5/2/2 | 1.24x |
| `sha384` | 99 | 41/47/11 | 1.25x | 275 | 169/89/17 | 1.28x |
| `sha512` | 117 | 51/55/11 | 1.25x | 293 | 178/98/17 | 1.28x |
| `ascon-hash256` | 99 | 53/45/1 | 1.27x | 99 | 53/45/1 | 1.27x |
| `sha512-256` | 99 | 61/38/0 | 1.29x | 99 | 61/38/0 | 1.29x |
| `ecdsa-p384` | 72 | 52/0/20 | 1.30x | 200 | 180/0/20 | 2.95x |
| `ascon-aead128` | 198 | 148/50/0 | 1.33x | 198 | 148/50/0 | 1.33x |
| `scrypt-small` | 36 | 22/14/0 | 1.33x | 36 | 22/14/0 | 1.33x |
| `hmac-sha256` | 117 | 64/38/15 | 1.34x | 293 | 194/74/25 | 1.66x |
| `aegis-256` | 198 | 89/83/26 | 1.37x | 198 | 89/83/26 | 1.37x |
| `ascon-xof128` | 99 | 74/25/0 | 1.37x | 99 | 74/25/0 | 1.37x |
| `blake3` | 432 | 228/170/34 | 1.40x | 432 | 228/170/34 | 1.40x |
| `xchacha20-poly1305` | 198 | 171/27/0 | 1.45x | 198 | 171/27/0 | 1.45x |
| `ecdsa-p256` | 72 | 63/9/0 | 1.51x | 200 | 187/13/0 | 2.40x |
| `rsa-4096` | 27 | 23/2/2 | 1.56x | 59 | 55/2/2 | 2.51x |
| `crc32c` | 99 | 44/42/13 | 1.61x | 198 | 140/45/13 | 2.20x |
| `rsa-3072` | 27 | 23/2/2 | 1.63x | 59 | 55/2/2 | 2.55x |
| `crc32` | 99 | 48/36/15 | 1.64x | 198 | 145/38/15 | 2.25x |
| `rsa-2048` | 27 | 25/0/2 | 1.69x | 59 | 57/0/2 | 2.61x |
| `kmac256` | 99 | 62/21/16 | 1.73x | 99 | 62/21/16 | 1.73x |
| `aes-128-gcm` | 198 | 113/45/40 | 1.76x | 550 | 429/52/69 | 2.55x |
| `aes-256-gcm` | 198 | 105/44/49 | 1.77x | 550 | 417/49/84 | 2.56x |
| `cshake256` | 99 | 63/27/9 | 1.78x | 99 | 63/27/9 | 1.78x |
| `shake128` | 99 | 73/26/0 | 1.84x | 99 | 73/26/0 | 1.84x |
| `shake256` | 99 | 68/31/0 | 1.86x | 99 | 68/31/0 | 1.86x |
| `sha224` | 99 | 55/44/0 | 1.91x | 99 | 55/44/0 | 1.91x |
| `aes-128-gcm-siv` | 198 | 137/4/57 | 2.02x | 352 | 268/20/64 | 2.79x |
| `sha3-256` | 117 | 102/15/0 | 2.10x | 117 | 102/15/0 | 2.10x |
| `sha3-224` | 99 | 87/11/1 | 2.11x | 99 | 87/11/1 | 2.11x |
| `aes-256-gcm-siv` | 198 | 138/4/56 | 2.12x | 352 | 291/5/56 | 3.01x |
| `sha3-384` | 99 | 89/10/0 | 2.16x | 99 | 89/10/0 | 2.16x |
| `crc64-nvme` | 99 | 57/33/9 | 2.17x | 99 | 57/33/9 | 2.17x |
| `sha3-512` | 99 | 88/11/0 | 2.19x | 99 | 88/11/0 | 2.19x |
| `crc64-xz` | 99 | 75/12/12 | 2.44x | 99 | 75/12/12 | 2.44x |
| `crc24-openpgp` | 99 | 94/2/3 | 13.26x | 99 | 94/2/3 | 13.26x |
| `crc16-ccitt` | 99 | 98/0/1 | 23.14x | 99 | 98/0/1 | 23.14x |
| `crc16-ibm` | 99 | 98/0/1 | 24.13x | 99 | 98/0/1 | 24.13x |

## Linux Clear Losses

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | --- | --- | --- | --- | --- |
| `mlkem512` / `keygen` | 9 | 1/2/6 | 0.69x | 0.91x | `aws-lc-rs` 6, `libcrux` 2 |
| `mlkem1024` / `keygen` | 9 | 3/1/5 | 0.70x | 0.94x | `aws-lc-rs` 3, `libcrux` 3 |
| `mlkem768` / `keygen` | 9 | 3/2/4 | 0.73x | 1.01x | `aws-lc-rs` 3, `libcrux` 3 |
| `mlkem1024` / `decapsulate` | 9 | 5/0/4 | 0.75x | 1.11x | `aws-lc-rs` 3, `libcrux` 1 |
| `mlkem768` / `decapsulate` | 9 | 5/0/4 | 0.75x | 1.09x | `aws-lc-rs` 3, `libcrux` 1 |
| `mlkem512` / `decapsulate` | 9 | 4/1/4 | 0.75x | 1.04x | `aws-lc-rs` 3, `libcrux` 2 |
| `mlkem1024` / `encapsulate` | 9 | 5/0/4 | 0.86x | 1.11x | `aws-lc-rs` 3, `libcrux` 1 |
| `mlkem768` / `encapsulate` | 9 | 5/1/3 | 0.89x | 1.21x | `aws-lc-rs` 3, `libcrux` 1 |
| `mlkem512` / `encapsulate` | 9 | 6/0/3 | 0.90x | 1.31x | `aws-lc-rs` 2, `libcrux` 1 |
| `ecdsa-p384` / `sign` | 36 | 16/0/20 | 0.94x | 0.90x | `aws-lc-rs` 20 |
| `argon2id-owasp` / `hash` | 9 | 3/2/4 | 0.97x | 0.96x | `rustcrypto` 6 |
| `ed25519` / `verify` | 36 | 8/17/11 | 0.98x | 1.00x | `dalek` 10, `aws-lc-rs` 9, `ring` 9 |
| `chacha20-poly1305` / `encrypt` | 99 | 32/33/34 | 0.99x | 1.00x | `ring` 50, `aws-lc-rs` 15, `rustcrypto` 2 |
| `chacha20-poly1305` / `decrypt` | 99 | 41/23/35 | 1.01x | 1.00x | `aws-lc-rs` 33, `ring` 25 |
| `argon2i-small` / `hash` | 27 | 12/3/12 | 1.02x | 0.99x | `rustcrypto` 15 |
| `blake2b256` / `streaming` | 27 | 7/20/0 | 1.02x | 1.00x | `rustcrypto` 20 |
| `x25519` / `public-key-from-secret` | 9 | 2/7/0 | 1.02x | 1.01x | `aws-lc-rs` 6, `dalek` 1 |
| `argon2id-small` / `hash` | 27 | 12/3/12 | 1.02x | 0.98x | `rustcrypto` 15 |
| `argon2d-small` / `hash` | 27 | 12/5/10 | 1.03x | 0.99x | `rustcrypto` 15 |
| `ed25519` / `sign` | 36 | 9/24/3 | 1.05x | 1.02x | `aws-lc-rs` 24, `dryoc` 2, `dalek` 1 |
| `x25519` / `diffie-hellman` | 9 | 3/6/0 | 1.05x | 1.01x | `aws-lc-rs` 6 |
| `rapidhash-v3-64` / `hash` | 99 | 27/52/20 | 1.05x | 1.00x | `rapidhash` 72 |
| `blake2b256` / `keyed` | 99 | 60/38/1 | 1.07x | 1.06x | `rustcrypto` 39 |
| `blake2b512` / `keyed` | 99 | 55/43/1 | 1.07x | 1.07x | `rustcrypto` 44 |
| `rapidhash-v3-128` / `hash` | 99 | 29/49/21 | 1.07x | 1.01x | `rapidhash` 70 |
| `blake2s256` / `streaming` | 27 | 10/17/0 | 1.07x | 1.01x | `rustcrypto` 17 |
| `blake2b256` / `hash` | 99 | 60/38/1 | 1.09x | 1.08x | `rustcrypto` 39 |
| `blake2s256` / `keyed` | 99 | 55/44/0 | 1.09x | 1.08x | `rustcrypto` 44 |
| `blake2s128` / `keyed` | 99 | 59/40/0 | 1.10x | 1.09x | `rustcrypto` 40 |
| `blake2b512` / `hash` | 99 | 62/36/1 | 1.11x | 1.10x | `rustcrypto` 37 |
| `rapidhash-64` / `hash` | 99 | 33/55/11 | 1.12x | 1.01x | `rapidhash` 66 |
| `blake2s256` / `hash` | 99 | 55/42/2 | 1.12x | 1.11x | `rustcrypto` 44 |
| `rapidhash-128` / `hash` | 99 | 39/51/9 | 1.14x | 1.02x | `rapidhash` 60 |
| `blake2s128` / `hash` | 99 | 64/35/0 | 1.15x | 1.14x | `rustcrypto` 35 |
| `blake3` / `streaming` | 36 | 8/24/4 | 1.17x | 1.00x | `blake3` 28 |
| `xxh3-128` / `hash` | 99 | 50/34/15 | 1.17x | 1.08x | `xxhash-rust` 49 |
| `rsa-4096` / `verify-pss-sha256` | 9 | 7/1/1 | 1.17x | 1.12x | `ring` 1, `aws-lc-rs` 1 |
| `rsa-4096` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.17x | 1.13x | `ring` 1, `aws-lc-rs` 1 |
| `xxh3-64` / `hash` | 99 | 52/35/12 | 1.18x | 1.09x | `xxhash-rust` 47 |
| `sha256` / `streaming` | 18 | 3/13/2 | 1.19x | 1.00x | `sha2` 15 |
| `rsa-3072` / `verify-pss-sha256` | 9 | 7/1/1 | 1.19x | 1.15x | `ring` 1, `aws-lc-rs` 1 |
| `hmac-sha256` / `streaming` | 18 | 3/12/3 | 1.19x | 1.00x | `rustcrypto` 15 |
| `rsa-8192` / `verify-pss-sha256` | 9 | 7/1/1 | 1.19x | 1.09x | `ring` 1, `aws-lc-rs` 1 |
| `rsa-3072` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.19x | 1.15x | `ring` 1, `aws-lc-rs` 1 |
| `rsa-8192` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.20x | 1.09x | `ring` 1, `aws-lc-rs` 1 |
| `hkdf-sha256` / `expand` | 36 | 34/2/0 | 1.22x | 1.10x | `rustcrypto` 2 |
| `pbkdf2-sha256` / `hash` | 54 | 49/5/0 | 1.22x | 1.14x | `aws-lc-rs` 3, `rustcrypto` 2 |
| `sha256` / `hash` | 99 | 39/47/13 | 1.23x | 1.00x | `aws-lc-rs` 31, `sha2` 24, `ring` 5 |

## Linux Worst Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| IBM z16/s390x | `mlkem512 / decapsulate` | `libcrux` | 0.11x |
| IBM z16/s390x | `mlkem768 / decapsulate` | `libcrux` | 0.12x |
| IBM z16/s390x | `mlkem768 / encapsulate` | `libcrux` | 0.12x |
| IBM z16/s390x | `mlkem512 / encapsulate` | `libcrux` | 0.12x |
| IBM z16/s390x | `mlkem1024 / decapsulate` | `libcrux` | 0.12x |
| IBM z16/s390x | `mlkem1024 / encapsulate` | `libcrux` | 0.13x |
| IBM z16/s390x | `mlkem512 / keygen` | `libcrux` | 0.14x |
| IBM z16/s390x | `mlkem768 / keygen` | `libcrux` | 0.15x |
| IBM z16/s390x | `mlkem1024 / keygen` | `libcrux` | 0.15x |
| RISE RISC-V | `xxh3-64 / hash / 0` | `xxhash-rust` | 0.44x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 32` | `aws-lc-rs` | 0.48x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |

## Linux Strongest Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt / hash / 16384` | `crc` | 228.97x |
| Intel Sapphire Rapids | `crc16-ibm / hash / 262144` | `crc` | 210.97x |
| Intel Sapphire Rapids | `crc16-ccitt / hash / 262144` | `crc` | 210.73x |
| Intel Sapphire Rapids | `crc16-ibm / hash / 16384` | `crc` | 198.96x |
| Intel Sapphire Rapids | `crc16-ibm / hash / 1048576` | `crc` | 185.67x |
| Intel Sapphire Rapids | `crc16-ibm / hash / 4096` | `crc` | 182.67x |
| Intel Sapphire Rapids | `crc16-ccitt / hash / 4096` | `crc` | 181.26x |
| Intel Sapphire Rapids | `crc16-ibm / hash / 65536` | `crc` | 180.27x |
| IBM Power10 | `crc16-ccitt / hash / 262144` | `crc` | 177.67x |
| IBM Power10 | `crc16-ibm / hash / 262144` | `crc` | 176.43x |
| IBM Power10 | `crc16-ccitt / hash / 1048576` | `crc` | 175.99x |
| IBM Power10 | `crc16-ibm / hash / 1048576` | `crc` | 175.90x |

## Top Five Loss Areas

- `mlkem512` / `keygen`: 0.69x geomean across 9 rows; W/T/L 1/2/6; pressure `aws-lc-rs` 6, `libcrux` 2.
- `mlkem1024` / `keygen`: 0.70x geomean across 9 rows; W/T/L 3/1/5; pressure `aws-lc-rs` 3, `libcrux` 3.
- `mlkem768` / `keygen`: 0.73x geomean across 9 rows; W/T/L 3/2/4; pressure `aws-lc-rs` 3, `libcrux` 3.
- `mlkem1024` / `decapsulate`: 0.75x geomean across 9 rows; W/T/L 5/0/4; pressure `aws-lc-rs` 3, `libcrux` 1.
- `mlkem768` / `decapsulate`: 0.75x geomean across 9 rows; W/T/L 5/0/4; pressure `aws-lc-rs` 3, `libcrux` 1.

## External Pressure

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `libcrux` | 81 | 58/11/12 | 72% | 0.94x | 1.13x |
| `rapidhash` | 396 | 128/207/61 | 32% | 1.09x | 1.01x |
| `xxhash-rust` | 198 | 102/69/27 | 52% | 1.18x | 1.09x |
| `aws-lc-rs` | 1,673 | 1002/346/325 | 60% | 1.21x | 1.13x |
| `ascon-hash` | 198 | 127/70/1 | 64% | 1.32x | 1.25x |
| `ascon-aead` | 198 | 148/50/0 | 75% | 1.33x | 1.15x |
| `aegis-crate` | 198 | 89/83/26 | 45% | 1.37x | 1.03x |
| `blake3` | 432 | 228/170/34 | 53% | 1.40x | 1.07x |
| `dalek` | 108 | 83/18/7 | 77% | 1.45x | 1.24x |
| `sha2` | 531 | 304/224/3 | 57% | 1.49x | 1.06x |
| `ring` | 1,656 | 1303/198/155 | 79% | 1.62x | 1.29x |
| `tiny-keccak` | 396 | 266/105/25 | 67% | 1.80x | 1.97x |
| `dryoc` | 360 | 322/32/6 | 89% | 1.86x | 1.83x |
| `crc-fast` | 297 | 172/101/24 | 58% | 2.01x | 1.15x |
| `rustcrypto` | 2,889 | 2141/600/148 | 74% | 2.06x | 1.21x |
| `sha3` | 414 | 366/47/1 | 88% | 2.14x | 2.11x |
| `crc32fast` | 99 | 83/9/7 | 84% | 2.40x | 1.98x |
| `crc64fast` | 99 | 75/12/12 | 76% | 2.44x | 2.01x |
| `fips203` | 81 | 72/0/9 | 89% | 2.48x | 2.54x |
| `crc32c` | 99 | 87/6/6 | 88% | 2.72x | 2.14x |
| `rustcrypto-rsa` | 81 | 81/0/0 | 100% | 5.64x | 4.65x |
| `crc` | 297 | 290/2/5 | 98% | 19.49x | 29.89x |

## README Numbers

- **Headline:** 4,102 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,056 are wins or ties. Linux CI geomean is 1.58x.
- **Checksums:** 5.20x geomean across 693 Linux CI fastest-external rows; W/T/L 514/125/54.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,726 Linux CI fastest-external rows; W/T/L 2110/1392/224.
- **Auth/KDF:** 1.23x geomean across 180 Linux CI fastest-external rows; W/T/L 165/15/0.
- **Password hashing:** 1.11x geomean across 135 Linux CI fastest-external rows; W/T/L 66/29/40.
- **Public-key:** 1.12x geomean across 333 Linux CI fastest-external rows; W/T/L 188/74/71.
- **RSA:** 1.54x geomean across 99 Linux CI fastest-external rows; W/T/L 85/6/8.
- **AEAD:** 1.56x geomean across 1,584 Linux CI fastest-external rows; W/T/L 974/313/297.
- **ML-KEM:** 0.78x geomean across 81 Linux CI fastest-external rows; W/T/L 37/7/37.
- **ECDSA P-256/P-384:** 1.40x Linux CI geomean across 144 fastest-external rows.
- **Current top losses:** `mlkem512` / `keygen`: 0.69x geomean across 9 rows; W/T/L 1/2/6; pressure `aws-lc-rs` 6, `libcrux` 2; `mlkem1024` / `keygen`: 0.70x geomean across 9 rows; W/T/L 3/1/5; pressure `aws-lc-rs` 3, `libcrux` 3; `mlkem768` / `keygen`: 0.73x geomean across 9 rows; W/T/L 3/2/4; pressure `aws-lc-rs` 3, `libcrux` 3; `mlkem1024` / `decapsulate`: 0.75x geomean across 9 rows; W/T/L 5/0/4; pressure `aws-lc-rs` 3, `libcrux` 1; `mlkem768` / `decapsulate`: 0.75x geomean across 9 rows; W/T/L 5/0/4; pressure `aws-lc-rs` 3, `libcrux` 1.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-06-19 04_17_57` | 2,202 | `benchmark_results/2026-06-19/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-06-19 04_17_55` | 2,202 | `benchmark_results/2026-06-19/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-06-19 04_17_58` | 2,202 | `benchmark_results/2026-06-19/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-06-19 04_17_58` | 2,202 | `benchmark_results/2026-06-19/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-06-19 04_18_10` | 1,953 | `benchmark_results/2026-06-19/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-06-19 04_17_47` | 1,953 | `benchmark_results/2026-06-19/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-06-19 04_18_00` | 2,202 | `benchmark_results/2026-06-19/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-06-19 04_18_01` | 2,202 | `benchmark_results/2026-06-19/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-06-19 04_18_48` | 2,202 | `benchmark_results/2026-06-19/linux/rise-riscv/results.txt` |
