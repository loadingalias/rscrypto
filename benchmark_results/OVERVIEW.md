# Benchmark Overview

Sources:

- Linux benchmark CI run [#27449945480](https://github.com/loadingalias/rscrypto/actions/runs/27449945480), created 2026-06-12 23:57:49 UTC.
- Linux commit: `a33fc67613dde3640139863b3668d808ad53d71f`.
- Apple Silicon local run: `benchmark_results/2026-06-12/macos/aarch64/results.txt`, created 2026-06-12 19:58:05 local time.
- Apple Silicon commit: `a33fc67613dde3640139863b3668d808ad53d71f`.
- Previous Linux comparison baseline: run [#27387537624](https://github.com/loadingalias/rscrypto/actions/runs/27387537624), created 2026-06-12 01:00:42 UTC.
- Previous Linux commit: `62be628d1ceb3d347ffcbb7ef3af67f046bc22ac`.

Scope: the 2026-06-12 nine-runner Linux CI benchmark matrix and the 2026-06-12 local Apple Silicon full run for commit `a33fc67`. Ratios are `external_crate_time / rscrypto_time`; higher is better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and input shape. Internal kernel, scratch-buffer, padding-only, cold-path, PHC roundtrip, parallel-scaling, threshold-selection, and public-overhead microbenches are parsed as raw rows but excluded from external win/loss claims.

Coverage note: the current Linux and Apple Silicon passes are both full benchmark runs. They include checksum, hash, XOF, MAC, KDF, password-hashing, BLAKE2/BLAKE3, RSA import/verification, ECDSA P-256/P-384 signing and verification, Ed25519, X25519, and AEAD rows. Apple Silicon is a separate local reference and is not combined into Linux aggregate totals.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Linux CI: all matched performance pairs | 10,574 | 7383/2345/846 | 70% | 1.78x | 1.22x |
| Linux CI: fastest external per case | 6,669 | 4078/1931/660 | 61% | 1.59x | 1.12x |
| Apple Silicon: all matched performance pairs | 1,248 | 805/416/27 | 65% | 1.85x | 1.15x |
| Apple Silicon: fastest external per case | 741 | 376/340/25 | 51% | 1.41x | 1.05x |
| Previous Linux CI: fastest external per case | 6,669 | 4051/1955/663 | 61% | 1.59x | 1.11x |

Shareable release summary:

- **Headline:** 4,078 of 6,669 matched Linux CI fastest-external comparisons are wins; 6,009 are wins or ties. Linux CI fastest-external geomean is 1.59x.
- **Apple Silicon headline:** 376 of 741 matched fastest-external comparisons are wins; 716 are wins or ties. Apple Silicon fastest-external geomean is 1.41x.
- **Checksums:** 5.00x geomean across 693 fastest-external rows; W/T/L is 515/120/58.
- **Hashes/MACs/XOFs:** 1.36x geomean across 3,726 fastest-external rows; W/T/L is 2127/1377/222.
- **Auth/KDF:** 1.23x geomean across 180 fastest-external rows; W/T/L is 159/20/1.
- **Password hashing:** 1.10x geomean across 135 fastest-external rows; W/T/L is 63/31/41.
- **Public-key:** 1.26x geomean across 252 fastest-external rows; W/T/L is 150/67/35.
- **RSA:** 1.54x geomean across 99 fastest-external rows; W/T/L is 87/4/8.
- **AEAD:** 1.56x geomean across 1,584 fastest-external rows; W/T/L is 977/312/295.
- **ECDSA P-256/P-384:** Linux CI 1.41x geomean across 144 fastest-external rows; W/T/L is 115/8/21. Apple Silicon is 1.25x across 16 rows; W/T/L is 12/4/0.
- **Apple Silicon AEAD:** 1.48x geomean across 176 fastest-external rows.
- **Top current loss areas:** `argon2id-owasp` / `hash`: 0.95x geomean across 9 rows; W/T/L 2/3/4; pressure `rustcrypto` 7; `ecdsa-p384` / `sign`: 0.95x geomean across 36 rows; W/T/L 16/0/20; pressure `aws-lc-rs` 20; `ed25519` / `verify`: 0.98x geomean across 36 rows; W/T/L 8/19/9; pressure `dalek` 10, `aws-lc-rs` 9, `ring` 9; `chacha20-poly1305` / `encrypt`: 0.99x geomean across 99 rows; W/T/L 30/35/34; pressure `ring` 55, `aws-lc-rs` 11, `rustcrypto` 3; `argon2i-small` / `hash`: 1.00x geomean across 27 rows; W/T/L 12/3/12; pressure `rustcrypto` 15.

## What Changed Since 2026-06-12 01:00 UTC

- Fastest-external Linux coverage stayed at 6,669 rows. This rewrite reports the full matched row set present in the raw artifacts, including BLAKE2/BLAKE3 and CRC32C rows.
- Headline fastest-external geomean stayed at 1.59x; wins moved from 4,051 to 4,078, and losses moved from 663 to 660.
- On 6,669 overlapping platform/case rows, geomean moved from 1.59x to 1.59x.
- Linux and Apple Silicon evidence now point at commit `a33fc67`; Apple remains a separate local reference.

| Scope | Previous | Current | Delta |
| --- | --- | --- | --- |
| Fastest rows | 6,669 | 6,669 | +0 |
| Wins | 4,051 | 4,078 | +27 |
| Wins or ties | 6,006 | 6,009 | +3 |
| Losses | 663 | 660 | -3 |
| Geomean | 1.59x | 1.59x | +0.00x |
| Median | 1.11x | 1.12x | +0.00x |

### Largest apples-to-apples drops

| Platform | Case | Fastest external | Previous | Current | Delta |
| --- | --- | --- | --- | --- | --- |
| AMD Zen4 | `crc16-ibm / hash / 256` | `crc` | 31.55x | 19.22x | -12.34x |
| Intel Ice Lake | `crc16-ccitt / hash / 256` | `crc` | 36.08x | 24.41x | -11.68x |
| Intel Sapphire Rapids | `crc16-ccitt / hash / 262144` | `crc` | 220.23x | 210.70x | -9.54x |
| Intel Ice Lake | `crc24-openpgp / hash / 1024` | `crc` | 54.14x | 46.97x | -7.17x |
| Intel Sapphire Rapids | `sha224 / hash / 65536` | `sha2` | 124.16x | 117.59x | -6.57x |
| Intel Ice Lake | `crc16-ibm / hash / 1048576` | `crc` | 119.50x | 113.18x | -6.32x |
| Intel Sapphire Rapids | `crc16-ccitt / hash / 65536` | `crc` | 179.59x | 173.68x | -5.91x |
| IBM Power10 | `crc16-ccitt / hash / 16384` | `crc` | 161.78x | 156.43x | -5.35x |
| Intel Sapphire Rapids | `crc24-openpgp / hash / 16384` | `crc` | 89.23x | 84.09x | -5.13x |
| AWS Graviton3 | `crc24-openpgp / hash / 1048576` | `crc` | 73.54x | 68.72x | -4.82x |

### Largest apples-to-apples gains

| Platform | Case | Fastest external | Previous | Current | Delta |
| --- | --- | --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt / hash / 4096` | `crc` | 160.62x | 169.61x | +8.99x |
| Intel Sapphire Rapids | `crc24-openpgp / hash / 4096` | `crc` | 82.69x | 91.28x | +8.59x |
| Intel Sapphire Rapids | `crc16-ibm / hash / 1048576` | `crc` | 179.59x | 186.92x | +7.33x |
| IBM z16/s390x | `crc16-ibm / hash / 16384` | `crc` | 91.59x | 98.73x | +7.14x |
| IBM z16/s390x | `crc16-ibm / hash / 65536` | `crc` | 98.18x | 102.39x | +4.21x |
| IBM z16/s390x | `crc16-ibm / hash / 4096` | `crc` | 77.00x | 81.14x | +4.14x |
| Intel Ice Lake | `crc16-ccitt / hash / 1048576` | `crc` | 112.42x | 116.55x | +4.13x |
| AMD Zen5 | `crc24-openpgp / hash / 1048576` | `crc` | 123.05x | 126.54x | +3.49x |
| Intel Sapphire Rapids | `crc16-ccitt / hash / 16384` | `crc` | 225.43x | 228.58x | +3.15x |
| IBM Power10 | `crc16-ibm / hash / 65536` | `crc` | 165.83x | 168.96x | +3.13x |

## Coverage Matrix

| Platform | Raw Criterion rows | All pairs | Fastest rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 2,168 | 1,226 | 741 | 522/156/63 | 70% | 1.47x | 1.14x |
| AMD Zen5 | 2,168 | 1,226 | 741 | 435/241/65 | 59% | 1.49x | 1.09x |
| AWS Graviton3 | 2,168 | 1,226 | 741 | 377/281/83 | 51% | 1.38x | 1.05x |
| AWS Graviton4 | 2,168 | 1,226 | 741 | 371/323/47 | 50% | 1.39x | 1.05x |
| IBM Power10 | 1,928 | 996 | 741 | 407/289/45 | 55% | 1.94x | 1.07x |
| IBM z16/s390x | 1,928 | 996 | 741 | 603/91/47 | 81% | 3.14x | 2.35x |
| Intel Ice Lake | 2,168 | 1,226 | 741 | 509/150/82 | 69% | 1.48x | 1.19x |
| Intel Sapphire Rapids | 2,168 | 1,226 | 741 | 515/148/78 | 70% | 1.62x | 1.19x |
| RISE RISC-V | 2,168 | 1,226 | 741 | 339/252/150 | 46% | 1.09x | 1.03x |
| Apple Silicon macOS/aarch64 | 2,243 | 1,248 | 741 | 376/340/25 | 51% | 1.41x | 1.05x |

## Category Summary

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 693 | 515/120/58 | 74% | 5.00x | 2.28x |
| Hashes/MACs/XOFs | 3,726 | 2127/1377/222 | 57% | 1.36x | 1.08x |
| Auth/KDF | 180 | 159/20/1 | 88% | 1.23x | 1.12x |
| Password hashing | 135 | 63/31/41 | 47% | 1.10x | 1.01x |
| Public-key | 252 | 150/67/35 | 60% | 1.26x | 1.13x |
| RSA | 99 | 87/4/8 | 88% | 1.54x | 1.16x |
| AEAD | 1,584 | 977/312/295 | 62% | 1.56x | 1.14x |

## Apple Silicon Category Summary

The 2026-06-12 macOS/aarch64 run is local Apple Silicon evidence from `benchmark_results/2026-06-12/macos/aarch64/results.txt`. It is not combined with Linux totals.

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 77 | 59/16/2 | 77% | 5.38x | 1.66x |
| Hashes/MACs/XOFs | 414 | 162/232/20 | 39% | 1.11x | 1.02x |
| Auth/KDF | 20 | 1/17/2 | 5% | 1.01x | 1.00x |
| Password hashing | 15 | 5/10/0 | 33% | 1.07x | 1.02x |
| Public-key | 28 | 14/14/0 | 50% | 1.15x | 1.06x |
| RSA | 11 | 11/0/0 | 100% | 1.44x | 1.20x |
| AEAD | 176 | 124/51/1 | 70% | 1.48x | 1.17x |

## ECDSA Summary

ECDSA signing includes both deterministic and blinded rscrypto rows in raw results; aggregate fastest-external comparisons use the fastest rscrypto row for the exact case. Constant-time release evidence is tracked separately by `ct.toml` and CT workflow artifacts.

| Operation | Scope | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `ecdsa-p256` / `sign` | Apple Silicon | 4 | 4/0/0 | 1.70x | 1.78x |
| `ecdsa-p256` / `sign` | Linux CI | 36 | 36/0/0 | 1.45x | 1.34x |
| `ecdsa-p256` / `verify` | Apple Silicon | 4 | 0/4/0 | 1.02x | 1.02x |
| `ecdsa-p256` / `verify` | Linux CI | 36 | 27/8/1 | 1.57x | 1.14x |
| `ecdsa-p384` / `sign` | Apple Silicon | 4 | 4/0/0 | 1.11x | 1.11x |
| `ecdsa-p384` / `sign` | Linux CI | 36 | 16/0/20 | 0.95x | 0.91x |
| `ecdsa-p384` / `verify` | Apple Silicon | 4 | 4/0/0 | 1.27x | 1.27x |
| `ecdsa-p384` / `verify` | Linux CI | 36 | 36/0/0 | 1.80x | 1.39x |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `argon2id-owasp` | 9 | 2/3/4 | 0.95x | 18 | 9/4/5 | 1.24x |
| `argon2i-small` | 27 | 12/3/12 | 1.00x | 45 | 29/3/13 | 1.37x |
| `chacha20-poly1305` | 198 | 68/62/68 | 1.00x | 550 | 296/138/116 | 1.18x |
| `argon2d-small` | 27 | 10/7/10 | 1.01x | 27 | 10/7/10 | 1.01x |
| `argon2id-small` | 27 | 12/3/12 | 1.02x | 45 | 29/3/13 | 1.40x |
| `x25519` | 18 | 4/13/1 | 1.03x | 50 | 34/14/2 | 1.53x |
| `rapidhash-v3-64` | 99 | 25/53/21 | 1.04x | 99 | 25/53/21 | 1.04x |
| `rapidhash-v3-128` | 99 | 30/48/21 | 1.07x | 99 | 30/48/21 | 1.07x |
| `blake2b256` | 225 | 129/94/2 | 1.07x | 351 | 249/100/2 | 1.34x |
| `blake2b512` | 198 | 119/78/1 | 1.09x | 297 | 213/83/1 | 1.35x |
| `ed25519` | 90 | 31/46/13 | 1.10x | 290 | 201/61/28 | 1.35x |
| `blake2s256` | 225 | 122/102/1 | 1.11x | 225 | 122/102/1 | 1.11x |
| `rapidhash-64` | 99 | 32/56/11 | 1.12x | 99 | 32/56/11 | 1.12x |
| `blake2s128` | 198 | 123/75/0 | 1.13x | 198 | 123/75/0 | 1.13x |
| `rapidhash-128` | 99 | 37/54/8 | 1.13x | 99 | 37/54/8 | 1.13x |
| `xxh3-128` | 99 | 51/34/14 | 1.17x | 99 | 51/34/14 | 1.17x |
| `rsa-8192` | 18 | 14/2/2 | 1.18x | 32 | 28/2/2 | 1.25x |
| `xxh3-64` | 99 | 54/34/11 | 1.19x | 99 | 54/34/11 | 1.19x |
| `pbkdf2-sha256` | 54 | 46/7/1 | 1.22x | 150 | 142/7/1 | 1.57x |
| `hkdf-sha256` | 36 | 31/5/0 | 1.22x | 100 | 95/5/0 | 1.81x |
| `sha256` | 117 | 46/52/19 | 1.22x | 293 | 165/95/33 | 1.45x |
| `scrypt-owasp` | 9 | 5/2/2 | 1.23x | 9 | 5/2/2 | 1.23x |
| `pbkdf2-sha512` | 54 | 46/8/0 | 1.23x | 150 | 141/9/0 | 1.31x |
| `hmac-sha512` | 99 | 48/43/8 | 1.24x | 275 | 178/84/13 | 1.29x |
| `sha512` | 117 | 50/56/11 | 1.24x | 293 | 176/101/16 | 1.27x |
| `hkdf-sha384` | 36 | 36/0/0 | 1.24x | 100 | 100/0/0 | 1.56x |
| `hmac-sha384` | 99 | 49/42/8 | 1.24x | 275 | 181/82/12 | 1.29x |
| `sha384` | 99 | 41/46/12 | 1.25x | 275 | 168/89/18 | 1.29x |
| `ascon-hash256` | 99 | 53/44/2 | 1.27x | 99 | 53/44/2 | 1.27x |
| `sha512-256` | 99 | 59/40/0 | 1.29x | 99 | 59/40/0 | 1.29x |
| `ecdsa-p384` | 72 | 52/0/20 | 1.31x | 200 | 180/0/20 | 2.97x |
| `hmac-sha256` | 117 | 64/39/14 | 1.32x | 293 | 192/77/24 | 1.64x |
| `ascon-aead128` | 198 | 148/50/0 | 1.33x | 198 | 148/50/0 | 1.33x |
| `scrypt-small` | 36 | 22/13/1 | 1.34x | 36 | 22/13/1 | 1.34x |
| `aegis-256` | 198 | 90/88/20 | 1.37x | 198 | 90/88/20 | 1.37x |
| `ascon-xof128` | 99 | 73/26/0 | 1.37x | 99 | 73/26/0 | 1.37x |
| `blake3` | 432 | 233/164/35 | 1.41x | 432 | 233/164/35 | 1.41x |
| `xchacha20-poly1305` | 198 | 169/28/1 | 1.44x | 198 | 169/28/1 | 1.44x |
| `ecdsa-p256` | 72 | 63/8/1 | 1.51x | 200 | 183/16/1 | 2.40x |
| `rsa-4096` | 27 | 24/1/2 | 1.56x | 59 | 56/1/2 | 2.50x |
| `crc32c` | 99 | 51/34/14 | 1.61x | 198 | 150/34/14 | 2.22x |
| `rsa-3072` | 27 | 24/1/2 | 1.64x | 59 | 56/1/2 | 2.55x |
| `rsa-2048` | 27 | 25/0/2 | 1.69x | 59 | 57/0/2 | 2.60x |
| `crc64-nvme` | 99 | 50/36/13 | 1.70x | 198 | 127/51/20 | 2.20x |
| `crc32` | 99 | 49/36/14 | 1.70x | 198 | 145/39/14 | 2.30x |
| `kmac256` | 99 | 63/21/15 | 1.74x | 99 | 63/21/15 | 1.74x |
| `aes-128-gcm` | 198 | 116/38/44 | 1.76x | 550 | 433/46/71 | 2.55x |
| `aes-256-gcm` | 198 | 109/40/49 | 1.78x | 550 | 419/49/82 | 2.57x |
| `cshake256` | 99 | 62/31/6 | 1.78x | 99 | 62/31/6 | 1.78x |
| `shake128` | 99 | 73/26/0 | 1.84x | 99 | 73/26/0 | 1.84x |
| `shake256` | 99 | 70/29/0 | 1.86x | 99 | 70/29/0 | 1.86x |
| `sha224` | 99 | 55/42/2 | 1.90x | 99 | 55/42/2 | 1.90x |
| `aes-128-gcm-siv` | 198 | 138/3/57 | 2.03x | 352 | 270/17/65 | 2.81x |
| `sha3-256` | 117 | 102/15/0 | 2.10x | 117 | 102/15/0 | 2.10x |
| `sha3-224` | 99 | 88/11/0 | 2.11x | 99 | 88/11/0 | 2.11x |
| `sha3-384` | 99 | 88/11/0 | 2.15x | 99 | 88/11/0 | 2.15x |
| `aes-256-gcm-siv` | 198 | 139/3/56 | 2.15x | 352 | 291/5/56 | 3.05x |
| `sha3-512` | 99 | 88/11/0 | 2.19x | 99 | 88/11/0 | 2.19x |
| `crc64-xz` | 99 | 76/12/11 | 2.39x | 99 | 76/12/11 | 2.39x |
| `crc24-openpgp` | 99 | 93/2/4 | 12.99x | 99 | 93/2/4 | 12.99x |
| `crc16-ccitt` | 99 | 98/0/1 | 22.95x | 99 | 98/0/1 | 22.95x |
| `crc16-ibm` | 99 | 98/0/1 | 23.72x | 99 | 98/0/1 | 23.72x |

## Linux Clear Losses

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | --- | --- | --- | --- | --- |
| `argon2id-owasp` / `hash` | 9 | 2/3/4 | 0.95x | 0.95x | `rustcrypto` 7 |
| `ecdsa-p384` / `sign` | 36 | 16/0/20 | 0.95x | 0.91x | `aws-lc-rs` 20 |
| `ed25519` / `verify` | 36 | 8/19/9 | 0.98x | 1.00x | `dalek` 10, `aws-lc-rs` 9, `ring` 9 |
| `chacha20-poly1305` / `encrypt` | 99 | 30/35/34 | 0.99x | 1.00x | `ring` 55, `aws-lc-rs` 11, `rustcrypto` 3 |
| `argon2i-small` / `hash` | 27 | 12/3/12 | 1.00x | 0.98x | `rustcrypto` 15 |
| `argon2d-small` / `hash` | 27 | 10/7/10 | 1.01x | 0.99x | `rustcrypto` 17 |
| `x25519` / `public-key-from-secret` | 9 | 1/7/1 | 1.01x | 1.01x | `aws-lc-rs` 6, `dalek` 1, `dryoc` 1 |
| `chacha20-poly1305` / `decrypt` | 99 | 38/27/34 | 1.01x | 1.01x | `aws-lc-rs` 33, `ring` 28 |
| `blake2b256` / `streaming` | 27 | 7/19/1 | 1.02x | 1.00x | `rustcrypto` 20 |
| `argon2id-small` / `hash` | 27 | 12/3/12 | 1.02x | 0.99x | `rustcrypto` 15 |
| `rapidhash-v3-64` / `hash` | 99 | 25/53/21 | 1.04x | 1.00x | `rapidhash` 74 |
| `x25519` / `diffie-hellman` | 9 | 3/6/0 | 1.05x | 1.02x | `aws-lc-rs` 6 |
| `ed25519` / `sign` | 36 | 9/25/2 | 1.05x | 1.02x | `aws-lc-rs` 24, `dalek` 2, `dryoc` 1 |
| `blake2s256` / `streaming` | 27 | 10/16/1 | 1.07x | 1.01x | `rustcrypto` 17 |
| `rapidhash-v3-128` / `hash` | 99 | 30/48/21 | 1.07x | 1.00x | `rapidhash` 69 |
| `blake2b512` / `keyed` | 99 | 56/43/0 | 1.07x | 1.07x | `rustcrypto` 43 |
| `blake2b256` / `keyed` | 99 | 60/39/0 | 1.07x | 1.06x | `rustcrypto` 39 |
| `blake2b256` / `hash` | 99 | 62/36/1 | 1.09x | 1.08x | `rustcrypto` 37 |
| `blake2s256` / `keyed` | 99 | 52/47/0 | 1.10x | 1.08x | `rustcrypto` 47 |
| `blake2s128` / `keyed` | 99 | 59/40/0 | 1.11x | 1.08x | `rustcrypto` 40 |
| `blake2b512` / `hash` | 99 | 63/35/1 | 1.11x | 1.10x | `rustcrypto` 36 |
| `rapidhash-64` / `hash` | 99 | 32/56/11 | 1.12x | 1.01x | `rapidhash` 67 |
| `blake2s256` / `hash` | 99 | 60/39/0 | 1.13x | 1.12x | `rustcrypto` 39 |
| `rapidhash-128` / `hash` | 99 | 37/54/8 | 1.13x | 1.02x | `rapidhash` 62 |
| `blake3` / `streaming` | 36 | 9/18/9 | 1.14x | 1.00x | `blake3` 27 |
| `blake2s128` / `hash` | 99 | 64/35/0 | 1.16x | 1.14x | `rustcrypto` 35 |
| `rsa-4096` / `verify-pss-sha256` | 9 | 8/0/1 | 1.17x | 1.12x | `aws-lc-rs` 1 |
| `rsa-4096` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.17x | 1.13x | `aws-lc-rs` 1, `ring` 1 |
| `xxh3-128` / `hash` | 99 | 51/34/14 | 1.17x | 1.05x | `xxhash-rust` 48 |
| `rsa-8192` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.18x | 1.09x | `aws-lc-rs` 1, `ring` 1 |
| `sha256` / `streaming` | 18 | 3/13/2 | 1.18x | 1.00x | `sha2` 15 |
| `hmac-sha256` / `streaming` | 18 | 3/13/2 | 1.18x | 1.00x | `rustcrypto` 15 |
| `rsa-3072` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.19x | 1.15x | `aws-lc-rs` 1, `ring` 1 |
| `xxh3-64` / `hash` | 99 | 54/34/11 | 1.19x | 1.10x | `xxhash-rust` 45 |
| `rsa-8192` / `verify-pss-sha256` | 9 | 7/1/1 | 1.19x | 1.09x | `aws-lc-rs` 1, `ring` 1 |
| `rsa-3072` / `verify-pss-sha256` | 9 | 8/0/1 | 1.20x | 1.15x | `aws-lc-rs` 1 |
| `pbkdf2-sha256` / `hash` | 54 | 46/7/1 | 1.22x | 1.13x | `aws-lc-rs` 5, `rustcrypto` 3 |
| `hkdf-sha256` / `expand` | 36 | 31/5/0 | 1.22x | 1.11x | `rustcrypto` 5 |
| `sha256` / `hash` | 99 | 43/39/17 | 1.23x | 1.00x | `sha2` 28, `aws-lc-rs` 22, `ring` 6 |
| `scrypt-owasp` / `hash` | 9 | 5/2/2 | 1.23x | 1.07x | `rustcrypto` 4 |
| `sha512` / `streaming` | 18 | 5/13/0 | 1.23x | 1.02x | `sha2` 13 |
| `pbkdf2-sha512` / `hash` | 54 | 46/8/0 | 1.23x | 1.11x | `aws-lc-rs` 5, `rustcrypto` 3 |
| `hmac-sha512` / `hash` | 99 | 48/43/8 | 1.24x | 1.05x | `rustcrypto` 27, `ring` 19, `aws-lc-rs` 5 |
| `sha512` / `hash` | 99 | 45/43/11 | 1.24x | 1.04x | `aws-lc-rs` 30, `sha2` 15, `ring` 9 |
| `hkdf-sha384` / `expand` | 36 | 36/0/0 | 1.24x | 1.08x | - |
| `hmac-sha384` / `hash` | 99 | 49/42/8 | 1.24x | 1.05x | `rustcrypto` 25, `ring` 16, `aws-lc-rs` 9 |
| `sha384` / `hash` | 99 | 41/46/12 | 1.25x | 1.05x | `aws-lc-rs` 37, `sha2` 16, `ring` 5 |
| `rsa-2048` / `verify-pkcs1v15-sha256` | 9 | 8/0/1 | 1.26x | 1.23x | `aws-lc-rs` 1 |
| `rsa-2048` / `verify-pss-sha256` | 9 | 8/0/1 | 1.27x | 1.22x | `aws-lc-rs` 1 |
| `ascon-hash256` / `hash` | 99 | 53/44/2 | 1.27x | 1.06x | `ascon-hash` 46 |
| `aegis-256` / `encrypt` | 99 | 29/50/20 | 1.27x | 1.00x | `aegis-crate` 70 |
| `sha512-256` / `hash` | 99 | 59/40/0 | 1.29x | 1.06x | `sha2` 40 |
| `ascon-aead128` / `encrypt` | 99 | 74/25/0 | 1.32x | 1.10x | `ascon-aead` 25 |
| `blake3` / `keyed` | 99 | 44/48/7 | 1.33x | 1.03x | `blake3` 55 |
| `scrypt-small` / `hash` | 36 | 22/13/1 | 1.34x | 1.16x | `rustcrypto` 14 |
| `ascon-aead128` / `decrypt` | 99 | 74/25/0 | 1.34x | 1.18x | `ascon-aead` 25 |
| `blake3` / `hash` | 99 | 47/44/8 | 1.34x | 1.04x | `blake3` 52 |
| `hmac-sha256` / `hash` | 99 | 61/26/12 | 1.35x | 1.11x | `ring` 23, `rustcrypto` 11, `aws-lc-rs` 4 |
| `blake3` / `xof` | 99 | 53/38/8 | 1.35x | 1.08x | `blake3` 46 |
| `ascon-xof128` / `hash` | 99 | 73/26/0 | 1.37x | 1.34x | `ascon-hash` 26 |
| `xchacha20-poly1305` / `encrypt` | 99 | 81/18/0 | 1.43x | 1.43x | `rustcrypto` 18 |
| `ecdsa-p256` / `sign` | 36 | 36/0/0 | 1.45x | 1.34x | - |
| `xchacha20-poly1305` / `decrypt` | 99 | 88/10/1 | 1.45x | 1.43x | `rustcrypto` 11 |
| `aegis-256` / `decrypt` | 99 | 61/38/0 | 1.48x | 1.12x | `aegis-crate` 38 |
| `ed25519` / `public-key-from-secret` | 9 | 7/1/1 | 1.51x | 1.26x | `dalek` 2 |
| `ed25519` / `keypair-from-secret` | 9 | 7/1/1 | 1.52x | 1.27x | `dalek` 2 |
| `ecdsa-p256` / `verify` | 36 | 27/8/1 | 1.57x | 1.14x | `aws-lc-rs` 9 |
| `crc32c` / `hash` | 99 | 51/34/14 | 1.61x | 1.05x | `crc-fast` 40, `crc32c` 8 |
| `aes-128-gcm` / `encrypt` | 99 | 56/18/25 | 1.68x | 1.09x | `aws-lc-rs` 23, `rustcrypto` 16, `ring` 4 |
| `crc64-nvme` / `hash` | 99 | 50/36/13 | 1.70x | 1.05x | `crc-fast` 26, `crc64fast-nvme` 23 |
| `crc32` / `hash` | 99 | 49/36/14 | 1.70x | 1.02x | `crc-fast` 35, `crc32fast` 15 |
| `aes-256-gcm` / `encrypt` | 99 | 52/18/29 | 1.70x | 1.07x | `aws-lc-rs` 23, `rustcrypto` 21, `ring` 3 |
| `kmac256` / `hash` | 99 | 63/21/15 | 1.74x | 1.96x | `tiny-keccak` 36 |
| `blake3` / `derive-key` | 99 | 80/16/3 | 1.77x | 1.91x | `blake3` 19 |
| `cshake256` / `hash` | 99 | 62/31/6 | 1.78x | 2.07x | `tiny-keccak` 36 |
| `ecdsa-p384` / `verify` | 36 | 36/0/0 | 1.80x | 1.39x | - |
| `shake128` / `hash` | 99 | 73/26/0 | 1.84x | 2.01x | `tiny-keccak` 26 |
| `aes-128-gcm` / `decrypt` | 99 | 60/20/19 | 1.84x | 1.18x | `aws-lc-rs` 25, `rustcrypto` 11, `ring` 3 |
| `shake256` / `hash` | 99 | 70/29/0 | 1.86x | 2.15x | `tiny-keccak` 29 |
| `aes-256-gcm` / `decrypt` | 99 | 57/22/20 | 1.86x | 1.18x | `aws-lc-rs` 28, `rustcrypto` 12, `ring` 2 |
| `sha224` / `hash` | 99 | 55/42/2 | 1.90x | 1.11x | `sha2` 44 |
| `aes-128-gcm-siv` / `decrypt` | 99 | 69/1/29 | 1.98x | 1.63x | `aws-lc-rs` 24, `rustcrypto` 6 |
| `sha3-256` / `streaming` | 18 | 15/3/0 | 2.05x | 2.02x | `sha3` 3 |
| `aes-128-gcm-siv` / `encrypt` | 99 | 69/2/28 | 2.08x | 1.82x | `aws-lc-rs` 24, `rustcrypto` 6 |
| `aes-256-gcm-siv` / `decrypt` | 99 | 69/1/29 | 2.09x | 1.52x | `aws-lc-rs` 24, `rustcrypto` 6 |
| `sha3-224` / `hash` | 99 | 88/11/0 | 2.11x | 2.13x | `sha3` 11 |
| `sha3-256` / `hash` | 99 | 87/12/0 | 2.11x | 2.14x | `sha3` 12 |
| `sha3-384` / `hash` | 99 | 88/11/0 | 2.15x | 2.14x | `sha3` 11 |
| `sha3-512` / `hash` | 99 | 88/11/0 | 2.19x | 2.14x | `sha3` 11 |
| `aes-256-gcm-siv` / `encrypt` | 99 | 70/2/27 | 2.21x | 1.87x | `aws-lc-rs` 24, `rustcrypto` 5 |
| `crc64-xz` / `hash` | 99 | 76/12/11 | 2.39x | 2.08x | `crc64fast` 23 |
| `rsa-4096` / `parse-spki` | 9 | 9/0/0 | 2.80x | 2.85x | - |
| `rsa-2048` / `parse-spki` | 9 | 9/0/0 | 3.02x | 3.05x | - |
| `rsa-3072` / `parse-spki` | 9 | 9/0/0 | 3.09x | 3.13x | - |
| `crc24-openpgp` / `hash` | 99 | 93/2/4 | 12.99x | 14.32x | `crc` 6 |
| `crc16-ccitt` / `hash` | 99 | 98/0/1 | 22.95x | 37.00x | `crc` 1 |
| `crc16-ibm` / `hash` | 99 | 98/0/1 | 23.72x | 37.05x | `crc` 1 |

## Apple Silicon Clear Losses

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | --- | --- | --- | --- | --- |
| `sha3-512` / `hash` | 11 | 0/7/4 | 0.93x | 1.01x | `sha3` 11 |
| `xxh3-64` / `hash` | 11 | 1/3/7 | 0.96x | 0.85x | `xxhash-rust` 10 |
| `hkdf-sha256` / `expand` | 4 | 0/3/1 | 0.97x | 0.98x | `rustcrypto` 4 |
| `sha3-256` / `streaming` | 2 | 0/1/1 | 0.98x | 0.98x | `sha3` 2 |
| `blake3` / `streaming` | 4 | 0/4/0 | 0.98x | 0.99x | `blake3` 4 |
| `sha512` / `streaming` | 2 | 0/2/0 | 0.99x | 0.99x | `sha2` 2 |
| `argon2id-owasp` / `hash` | 1 | 0/1/0 | 1.00x | 1.00x | `rustcrypto` 1 |
| `pbkdf2-sha256` / `hash` | 6 | 0/5/1 | 1.00x | 1.00x | `rustcrypto` 6 |
| `argon2i-small` / `hash` | 3 | 0/3/0 | 1.00x | 1.00x | `rustcrypto` 3 |
| `ed25519` / `verify` | 4 | 0/4/0 | 1.00x | 1.00x | `aws-lc-rs` 4 |
| `argon2id-small` / `hash` | 3 | 0/3/0 | 1.00x | 1.00x | `rustcrypto` 3 |
| `sha3-256` / `hash` | 11 | 0/10/1 | 1.00x | 1.02x | `sha3` 11 |
| `hkdf-sha384` / `expand` | 4 | 0/4/0 | 1.01x | 1.00x | `rustcrypto` 4 |
| `x25519` / `public-key-from-secret` | 1 | 0/1/0 | 1.01x | 1.01x | `aws-lc-rs` 1 |
| `argon2d-small` / `hash` | 3 | 0/3/0 | 1.01x | 1.00x | `rustcrypto` 3 |
| `x25519` / `diffie-hellman` | 1 | 0/1/0 | 1.01x | 1.01x | `aws-lc-rs` 1 |
| `blake2b256` / `streaming` | 3 | 0/3/0 | 1.01x | 1.01x | `rustcrypto` 3 |
| `blake2s256` / `streaming` | 3 | 0/3/0 | 1.01x | 1.01x | `rustcrypto` 3 |
| `ed25519` / `sign` | 4 | 0/4/0 | 1.01x | 1.01x | `aws-lc-rs` 4 |
| `sha3-384` / `hash` | 11 | 0/11/0 | 1.01x | 1.02x | `sha3` 11 |
| `chacha20-poly1305` / `encrypt` | 11 | 1/10/0 | 1.02x | 1.00x | `ring` 9, `aws-lc-rs` 1 |
| `blake2b256` / `keyed` | 11 | 1/10/0 | 1.02x | 1.02x | `rustcrypto` 10 |
| `ecdsa-p256` / `verify` | 4 | 0/4/0 | 1.02x | 1.02x | `aws-lc-rs` 4 |
| `sha3-224` / `hash` | 11 | 0/11/0 | 1.02x | 1.03x | `sha3` 11 |
| `blake2s128` / `keyed` | 11 | 4/7/0 | 1.03x | 1.02x | `rustcrypto` 7 |
| `chacha20-poly1305` / `decrypt` | 11 | 1/10/0 | 1.03x | 1.00x | `ring` 6, `aws-lc-rs` 4 |
| `hmac-sha384` / `hash` | 11 | 4/7/0 | 1.03x | 1.02x | `rustcrypto` 7 |
| `blake2s128` / `hash` | 11 | 2/9/0 | 1.03x | 1.02x | `rustcrypto` 9 |
| `blake2b512` / `keyed` | 11 | 1/10/0 | 1.03x | 1.02x | `rustcrypto` 10 |
| `blake2s256` / `hash` | 11 | 3/8/0 | 1.04x | 1.02x | `rustcrypto` 8 |

## Linux Worst Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.45x |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.46x |
| RISE RISC-V | `xxh3-64 / hash / 0` | `xxhash-rust` | 0.47x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 1024` | `aws-lc-rs` | 0.49x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 0` | `aws-lc-rs` | 0.49x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.49x |
| IBM z16/s390x | `rapidhash-v3-64 / hash / 1` | `rapidhash` | 0.49x |
| AMD Zen4 | `ecdsa-p384 / sign / 0` | `aws-lc-rs` | 0.51x |
| AMD Zen4 | `ecdsa-p384 / sign / 32` | `aws-lc-rs` | 0.51x |
| Intel Ice Lake | `ecdsa-p384 / sign / 0` | `aws-lc-rs` | 0.51x |

## Linux Strongest Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt / hash / 16384` | `crc` | 228.58x |
| Intel Sapphire Rapids | `crc16-ibm / hash / 262144` | `crc` | 217.09x |
| Intel Sapphire Rapids | `crc16-ccitt / hash / 262144` | `crc` | 210.70x |
| Intel Sapphire Rapids | `crc16-ibm / hash / 16384` | `crc` | 200.27x |
| Intel Sapphire Rapids | `crc16-ibm / hash / 1048576` | `crc` | 186.92x |
| Intel Sapphire Rapids | `crc16-ibm / hash / 65536` | `crc` | 180.46x |
| Intel Sapphire Rapids | `crc16-ccitt / hash / 1048576` | `crc` | 178.67x |
| IBM Power10 | `crc16-ibm / hash / 1048576` | `crc` | 176.40x |
| IBM Power10 | `crc16-ibm / hash / 262144` | `crc` | 175.64x |
| IBM Power10 | `crc16-ccitt / hash / 262144` | `crc` | 175.61x |
| IBM Power10 | `crc16-ccitt / hash / 1048576` | `crc` | 174.93x |
| Intel Sapphire Rapids | `crc16-ccitt / hash / 65536` | `crc` | 173.68x |

## Top Five Loss Areas

- `argon2id-owasp` / `hash`: 0.95x geomean across 9 rows; W/T/L 2/3/4; pressure `rustcrypto` 7.
- `ecdsa-p384` / `sign`: 0.95x geomean across 36 rows; W/T/L 16/0/20; pressure `aws-lc-rs` 20.
- `ed25519` / `verify`: 0.98x geomean across 36 rows; W/T/L 8/19/9; pressure `dalek` 10, `aws-lc-rs` 9, `ring` 9.
- `chacha20-poly1305` / `encrypt`: 0.99x geomean across 99 rows; W/T/L 30/35/34; pressure `ring` 55, `aws-lc-rs` 11, `rustcrypto` 3.
- `argon2i-small` / `hash`: 1.00x geomean across 27 rows; W/T/L 12/3/12; pressure `rustcrypto` 15.

## External Pressure

All-pair Linux CI comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `rapidhash` | 396 | 124/211/61 | 31% | 1.09x | 1.01x |
| `xxhash-rust` | 198 | 105/68/25 | 53% | 1.18x | 1.09x |
| `aws-lc-rs` | 1,610 | 978/333/299 | 61% | 1.23x | 1.14x |
| `ascon-hash` | 198 | 126/70/2 | 64% | 1.32x | 1.25x |
| `ascon-aead` | 198 | 148/50/0 | 75% | 1.33x | 1.15x |
| `aegis-crate` | 198 | 90/88/20 | 45% | 1.37x | 1.04x |
| `blake3` | 432 | 233/164/35 | 54% | 1.41x | 1.07x |
| `dalek` | 108 | 82/16/10 | 76% | 1.45x | 1.26x |
| `sha2` | 531 | 296/226/9 | 56% | 1.49x | 1.06x |
| `ring` | 1,656 | 1304/204/148 | 79% | 1.62x | 1.28x |
| `tiny-keccak` | 396 | 268/107/21 | 68% | 1.80x | 1.98x |
| `dryoc` | 360 | 324/25/11 | 90% | 1.85x | 1.85x |
| `crc-fast` | 297 | 176/95/26 | 59% | 1.99x | 1.12x |
| `rustcrypto` | 2,808 | 2071/597/140 | 74% | 2.04x | 1.20x |
| `sha3` | 414 | 366/48/0 | 88% | 2.14x | 2.14x |
| `crc64fast-nvme` | 99 | 73/15/11 | 74% | 2.34x | 2.06x |
| `crc64fast` | 99 | 76/12/11 | 77% | 2.39x | 2.08x |
| `crc32fast` | 99 | 82/10/7 | 83% | 2.48x | 2.09x |
| `crc32c` | 99 | 91/4/4 | 92% | 2.77x | 2.19x |
| `rustcrypto-rsa` | 81 | 81/0/0 | 100% | 5.66x | 4.49x |
| `crc` | 297 | 289/2/6 | 97% | 19.19x | 28.69x |

## README Numbers

- **Headline:** 4,078 of 6,669 matched Linux CI fastest-external comparisons are wins; 6,009 are wins or ties. Linux CI geomean is 1.59x.
- **Apple Silicon headline:** 376 of 741 matched fastest-external comparisons are wins; 716 are wins or ties. Apple Silicon geomean is 1.41x.
- **Checksums:** 5.00x geomean across 693 Linux CI fastest-external rows; W/T/L 515/120/58.
- **Hashes/MACs/XOFs:** 1.36x geomean across 3,726 Linux CI fastest-external rows; W/T/L 2127/1377/222.
- **Auth/KDF:** 1.23x geomean across 180 Linux CI fastest-external rows; W/T/L 159/20/1.
- **Password hashing:** 1.10x geomean across 135 Linux CI fastest-external rows; W/T/L 63/31/41.
- **Public-key:** 1.26x geomean across 252 Linux CI fastest-external rows; W/T/L 150/67/35.
- **RSA:** 1.54x geomean across 99 Linux CI fastest-external rows; W/T/L 87/4/8.
- **AEAD:** 1.56x geomean across 1,584 Linux CI fastest-external rows; W/T/L 977/312/295.
- **ECDSA P-256/P-384:** 1.41x Linux CI geomean across 144 fastest-external rows; Apple Silicon 1.25x across 16 rows.
- **Apple Silicon AEAD:** 1.48x geomean across 176 fastest-external rows.
- **Current top losses:** `argon2id-owasp` / `hash`: 0.95x geomean across 9 rows; W/T/L 2/3/4; pressure `rustcrypto` 7; `ecdsa-p384` / `sign`: 0.95x geomean across 36 rows; W/T/L 16/0/20; pressure `aws-lc-rs` 20; `ed25519` / `verify`: 0.98x geomean across 36 rows; W/T/L 8/19/9; pressure `dalek` 10, `aws-lc-rs` 9, `ring` 9; `chacha20-poly1305` / `encrypt`: 0.99x geomean across 99 rows; W/T/L 30/35/34; pressure `ring` 55, `aws-lc-rs` 11, `rustcrypto` 3; `argon2i-small` / `hash`: 1.00x geomean across 27 rows; W/T/L 12/3/12; pressure `rustcrypto` 15.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-06-12 23_58_51` | 2,168 | `benchmark_results/2026-06-12/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-06-12 23_58_35` | 2,168 | `benchmark_results/2026-06-12/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-06-12 23_58_40` | 2,168 | `benchmark_results/2026-06-12/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-06-12 23_58_55` | 2,168 | `benchmark_results/2026-06-12/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-06-12 23_58_42` | 1,928 | `benchmark_results/2026-06-12/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-06-12 23_58_42` | 1,928 | `benchmark_results/2026-06-12/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-06-12 23_58_43` | 2,168 | `benchmark_results/2026-06-12/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-06-12 23_58_44` | 2,168 | `benchmark_results/2026-06-12/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-06-12 23_59_34` | 2,168 | `benchmark_results/2026-06-12/linux/rise-riscv/results.txt` |
| Apple Silicon macOS/aarch64 | `local` | `2026-06-12 19_58_05` | 2,243 | `benchmark_results/2026-06-12/macos/aarch64/results.txt` |
