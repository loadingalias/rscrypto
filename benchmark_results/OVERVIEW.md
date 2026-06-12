# Benchmark Overview

Sources:

- Linux benchmark CI run [#27387537624](https://github.com/loadingalias/rscrypto/actions/runs/27387537624), created 2026-06-12 01:00:42 UTC.
- Linux commit: `62be628d1ceb3d347ffcbb7ef3af67f046bc22ac`.
- Apple Silicon local run: `benchmark_results/2026-06-11/macos/aarch64/results.txt`, created 2026-06-11 23:06:36 local time.
- Apple Silicon commit: `62be628d1ceb3d347ffcbb7ef3af67f046bc22ac`.
- Previous Linux comparison baseline: run [#27227999222](https://github.com/loadingalias/rscrypto/actions/runs/27227999222), created 2026-06-09 18:43:21 UTC.
- Previous Linux commit: `7dbf097a8db537fc85943f5ae6b2fd2dcc06342b`.

Scope: the 2026-06-12 nine-runner Linux CI benchmark matrix and the 2026-06-11 local Apple Silicon full run for commit `62be628`. Ratios are `external_crate_time / rscrypto_time`; higher is better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and input shape. Internal kernel, scratch-buffer, padding-only, cold-path, PHC roundtrip, parallel-scaling, and threshold-selection microbenches are parsed as raw rows but excluded from external win/loss claims.

Coverage note: the current Linux and Apple Silicon passes are both full benchmark runs. They include checksum, hash, XOF, MAC, KDF, password-hashing, RSA import/verification, ECDSA P-256/P-384 signing and verification, Ed25519, X25519, and AEAD rows. Apple Silicon is a separate local reference and is not combined into Linux aggregate totals.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Linux CI: all matched performance pairs | 8,873 | 6287/1789/797 | 71% | 1.88x | 1.26x |
| Linux CI: fastest external per case | 5,292 | 3287/1390/615 | 62% | 1.71x | 1.14x |
| Apple Silicon: all matched performance pairs | 1,037 | 724/282/31 | 70% | 1.97x | 1.18x |
| Apple Silicon: fastest external per case | 588 | 346/214/28 | 59% | 1.49x | 1.09x |
| Previous Linux CI: fastest external per case | 5,148 | 3212/1337/599 | 62% | 1.72x | 1.14x |

Shareable release summary:

- **Headline:** 3,287 of 5,292 matched Linux CI fastest-external comparisons are wins; 4,677 are wins or ties. Linux CI fastest-external geomean is 1.71x.
- **Apple Silicon headline:** 346 of 588 matched fastest-external comparisons are wins; 560 are wins or ties. Apple Silicon fastest-external geomean is 1.49x.
- **Checksums:** 6.05x geomean across 594 fastest-external rows; W/T/L is 461/84/49.
- **Hashes/MACs/XOFs:** 1.45x geomean across 2,448 fastest-external rows; W/T/L is 1385/879/184.
- **Auth/KDF:** 1.23x geomean across 180 fastest-external rows; W/T/L is 158/22/0.
- **Password hashing:** 1.09x geomean across 135 fastest-external rows; W/T/L is 65/27/43.
- **Public-key:** 1.26x geomean across 252 fastest-external rows; W/T/L is 148/72/32.
- **RSA:** 1.54x geomean across 99 fastest-external rows; W/T/L is 86/5/8.
- **AEAD:** 1.56x geomean across 1,584 fastest-external rows; W/T/L is 984/301/299.
- **ECDSA P-256/P-384:** Linux CI 1.41x geomean across 144 fastest-external rows; W/T/L is 113/12/19. Apple Silicon is 1.26x across 16 rows; W/T/L is 13/3/0.
- **Apple Silicon AEAD:** 1.48x geomean across 176 fastest-external rows.
- **Top current loss areas:** `ecdsa-p384` / `sign`: 0.95x geomean across 36 rows; W/T/L 16/1/19; pressure `aws-lc-rs` 20; `argon2id-owasp` / `hash`: 0.95x geomean across 9 rows; W/T/L 3/1/5; pressure `rustcrypto` 6; `ed25519` / `verify`: 0.98x geomean across 36 rows; W/T/L 8/18/10; pressure `dalek` 10, `aws-lc-rs` 9, `ring` 9; `chacha20-poly1305` / `encrypt`: 0.99x geomean across 99 rows; W/T/L 31/33/35; pressure `ring` 53, `aws-lc-rs` 12, `rustcrypto` 3; `chacha20-poly1305` / `decrypt`: 1.00x geomean across 99 rows; W/T/L 39/26/34; pressure `aws-lc-rs` 34, `ring` 26.

## What Changed Since 2026-06-09

- Fastest-external Linux coverage changed from 5,148 rows to 5,292 rows. This pass adds ECDSA P-256/P-384 sign/verify rows and keeps the full checksum/hash/MAC/KDF/password/RSA/Ed25519/X25519/AEAD row set.
- Headline fastest-external geomean moved from 1.72x to 1.71x; wins moved from 3,212 to 3,287, and losses moved from 599 to 615.
- Linux and Apple Silicon evidence now point at commit `62be628`; Apple remains a separate local reference.

| Scope | Previous | Current | Delta |
| --- | --- | --- | --- |
| Fastest rows | 5,148 | 5,292 | +144 |
| Wins | 3,212 | 3,287 | +75 |
| Wins or ties | 4,549 | 4,677 | +128 |
| Losses | 599 | 615 | +16 |
| Geomean | 1.72x | 1.71x | -0.01x |
| Median | 1.14x | 1.14x | +0.00x |

### Largest apples-to-apples drops

| Platform | Case | Fastest external | Previous | Current | Delta |
| --- | --- | --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt / 4096` | `crc` | 183.86x | 160.62x | -23.24x |
| Intel Sapphire Rapids | `crc24-openpgp / 4096` | `crc` | 90.98x | 82.69x | -8.29x |
| Intel Ice Lake | `crc16-ccitt / 1048576` | `crc` | 118.85x | 112.42x | -6.43x |
| IBM z16/s390x | `crc16-ibm / 16384` | `crc` | 97.74x | 91.59x | -6.15x |
| AWS Graviton3 | `crc16-ccitt / 1048576` | `crc` | 124.31x | 118.52x | -5.79x |
| IBM z16/s390x | `aegis-256 / decrypt / 65536` | `aegis-crate` | 14.50x | 9.59x | -4.91x |
| Intel Sapphire Rapids | `crc16-ibm / 1048576` | `crc` | 184.18x | 179.59x | -4.58x |
| IBM z16/s390x | `crc16-ibm / 65536` | `crc` | 102.46x | 98.18x | -4.28x |
| IBM z16/s390x | `sha3-384 / 4096` | `sha3` | 18.27x | 14.10x | -4.17x |
| IBM z16/s390x | `crc16-ibm / 1024` | `crc` | 56.88x | 52.81x | -4.07x |

### Largest apples-to-apples gains

| Platform | Case | Fastest external | Previous | Current | Delta |
| --- | --- | --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt / 262144` | `crc` | 212.47x | 220.23x | +7.76x |
| Intel Sapphire Rapids | `sha224 / 65536` | `sha2` | 117.88x | 124.16x | +6.28x |
| AWS Graviton3 | `crc24-openpgp / 1048576` | `crc` | 68.21x | 73.54x | +5.32x |
| Intel Ice Lake | `crc16-ibm / 1048576` | `crc` | 114.76x | 119.50x | +4.74x |
| AWS Graviton3 | `crc16-ibm / 65536` | `crc` | 120.35x | 125.02x | +4.67x |
| AWS Graviton4 | `crc24-openpgp / 1048576` | `crc` | 90.10x | 94.77x | +4.67x |
| Intel Sapphire Rapids | `crc16-ibm / 4096` | `crc` | 168.52x | 173.17x | +4.65x |
| Intel Sapphire Rapids | `crc24-openpgp / 16384` | `crc` | 84.77x | 89.23x | +4.45x |
| Intel Sapphire Rapids | `crc16-ibm / 262144` | `crc` | 211.21x | 215.65x | +4.44x |
| Intel Sapphire Rapids | `crc16-ibm / 1024` | `crc` | 117.77x | 121.96x | +4.19x |

## Coverage Matrix

| Platform | Raw Criterion rows | All pairs | Fastest rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 1,999 | 1,037 | 588 | 398/127/63 | 68% | 1.56x | 1.23x |
| AMD Zen5 | 1,999 | 1,037 | 588 | 389/134/65 | 66% | 1.60x | 1.16x |
| AWS Graviton3 | 1,999 | 1,037 | 588 | 309/202/77 | 53% | 1.44x | 1.06x |
| AWS Graviton4 | 1,999 | 1,037 | 588 | 316/228/44 | 54% | 1.45x | 1.06x |
| IBM Power10 | 1,769 | 807 | 588 | 322/221/45 | 55% | 2.08x | 1.07x |
| IBM z16/s390x | 1,769 | 807 | 588 | 488/60/40 | 83% | 3.81x | 2.86x |
| Intel Ice Lake | 1,999 | 1,037 | 588 | 395/114/79 | 67% | 1.56x | 1.20x |
| Intel Sapphire Rapids | 1,999 | 1,037 | 588 | 399/119/70 | 68% | 1.75x | 1.24x |
| RISE RISC-V | 1,999 | 1,037 | 588 | 271/185/132 | 46% | 1.09x | 1.03x |
| Apple Silicon macOS/aarch64 | 2,045 | 1,037 | 588 | 346/214/28 | 59% | 1.49x | 1.09x |

## Category Summary

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 594 | 461/84/49 | 78% | 6.05x | 3.00x |
| Hashes/MACs/XOFs | 2,448 | 1385/879/184 | 57% | 1.45x | 1.09x |
| Auth/KDF | 180 | 158/22/0 | 88% | 1.23x | 1.12x |
| Password hashing | 135 | 65/27/43 | 48% | 1.09x | 1.01x |
| Public-key | 252 | 148/72/32 | 59% | 1.26x | 1.11x |
| RSA | 99 | 86/5/8 | 87% | 1.54x | 1.18x |
| AEAD | 1,584 | 984/301/299 | 62% | 1.56x | 1.14x |

## Apple Silicon Category Summary

The 2026-06-11 macOS/aarch64 run is local Apple Silicon evidence from `benchmark_results/2026-06-11/macos/aarch64/results.txt`. It is not combined with Linux totals.

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 66 | 54/11/1 | 82% | 6.91x | 1.71x |
| Hashes/MACs/XOFs | 272 | 125/124/23 | 46% | 1.11x | 1.03x |
| Auth/KDF | 20 | 1/17/2 | 5% | 1.01x | 1.01x |
| Password hashing | 15 | 5/10/0 | 33% | 1.07x | 1.02x |
| Public-key | 28 | 16/11/1 | 57% | 1.15x | 1.10x |
| RSA | 11 | 11/0/0 | 100% | 1.46x | 1.28x |
| AEAD | 176 | 134/41/1 | 76% | 1.48x | 1.17x |

## ECDSA Summary

ECDSA signing includes both deterministic and blinded rscrypto rows in raw results; aggregate fastest-external comparisons use the fastest rscrypto row for the exact case. Constant-time release evidence is tracked separately by `ct.toml` and CT workflow artifacts.

| Operation | Scope | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `ecdsa-p256` / `sign` | Apple Silicon | 4 | 4/0/0 | 1.69x | 1.73x |
| `ecdsa-p256` / `sign` | Linux CI | 36 | 36/0/0 | 1.45x | 1.34x |
| `ecdsa-p256` / `verify` | Apple Silicon | 4 | 1/3/0 | 1.04x | 1.02x |
| `ecdsa-p256` / `verify` | Linux CI | 36 | 25/11/0 | 1.59x | 1.14x |
| `ecdsa-p384` / `sign` | Apple Silicon | 4 | 4/0/0 | 1.12x | 1.13x |
| `ecdsa-p384` / `sign` | Linux CI | 36 | 16/1/19 | 0.95x | 0.91x |
| `ecdsa-p384` / `verify` | Apple Silicon | 4 | 4/0/0 | 1.26x | 1.29x |
| `ecdsa-p384` / `verify` | Linux CI | 36 | 36/0/0 | 1.80x | 1.39x |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `argon2id-owasp` | 9 | 3/1/5 | 0.95x | 18 | 10/2/6 | 1.24x |
| `chacha20-poly1305` | 198 | 70/59/69 | 1.00x | 550 | 295/137/118 | 1.18x |
| `argon2id-small` | 27 | 12/3/12 | 1.00x | 45 | 29/3/13 | 1.38x |
| `argon2i-small` | 27 | 12/3/12 | 1.01x | 45 | 29/3/13 | 1.37x |
| `argon2d-small` | 27 | 11/6/10 | 1.01x | 27 | 11/6/10 | 1.01x |
| `x25519` | 18 | 3/15/0 | 1.03x | 50 | 33/17/0 | 1.54x |
| `rapidhash-v3-64` | 99 | 22/55/22 | 1.04x | 99 | 22/55/22 | 1.04x |
| `rapidhash-v3-128` | 99 | 31/49/19 | 1.07x | 99 | 31/49/19 | 1.07x |
| `ed25519` | 90 | 32/45/13 | 1.09x | 290 | 202/60/28 | 1.35x |
| `rapidhash-64` | 99 | 31/56/12 | 1.12x | 99 | 31/56/12 | 1.12x |
| `rapidhash-128` | 99 | 36/54/9 | 1.13x | 99 | 36/54/9 | 1.13x |
| `xxh3-128` | 99 | 50/32/17 | 1.16x | 99 | 50/32/17 | 1.16x |
| `xxh3-64` | 99 | 51/34/14 | 1.17x | 99 | 51/34/14 | 1.17x |
| `rsa-8192` | 18 | 14/2/2 | 1.19x | 32 | 28/2/2 | 1.26x |
| `hkdf-sha256` | 36 | 30/6/0 | 1.22x | 100 | 94/6/0 | 1.79x |
| `pbkdf2-sha256` | 54 | 49/5/0 | 1.22x | 150 | 145/5/0 | 1.57x |
| `pbkdf2-sha512` | 54 | 45/9/0 | 1.23x | 150 | 140/10/0 | 1.30x |
| `sha256` | 117 | 43/59/15 | 1.23x | 293 | 162/102/29 | 1.47x |
| `hmac-sha512` | 99 | 49/42/8 | 1.24x | 275 | 180/83/12 | 1.29x |
| `sha384` | 99 | 38/51/10 | 1.24x | 275 | 169/90/16 | 1.28x |
| `scrypt-owasp` | 9 | 5/2/2 | 1.25x | 9 | 5/2/2 | 1.25x |
| `hkdf-sha384` | 36 | 34/2/0 | 1.25x | 100 | 98/2/0 | 1.57x |
| `sha512` | 117 | 48/60/9 | 1.25x | 293 | 175/103/15 | 1.28x |
| `hmac-sha384` | 99 | 46/45/8 | 1.25x | 275 | 174/88/13 | 1.29x |
| `ascon-hash256` | 99 | 56/43/0 | 1.27x | 99 | 56/43/0 | 1.27x |
| `sha512-256` | 99 | 54/45/0 | 1.30x | 99 | 54/45/0 | 1.30x |
| `ecdsa-p384` | 72 | 52/1/19 | 1.30x | 200 | 180/1/19 | 2.97x |
| `scrypt-small` | 36 | 22/12/2 | 1.33x | 36 | 22/12/2 | 1.33x |
| `ascon-aead128` | 198 | 151/47/0 | 1.33x | 198 | 151/47/0 | 1.33x |
| `hmac-sha256` | 117 | 64/39/14 | 1.35x | 293 | 193/76/24 | 1.67x |
| `aegis-256` | 198 | 89/84/25 | 1.37x | 198 | 89/84/25 | 1.37x |
| `ascon-xof128` | 99 | 73/26/0 | 1.37x | 99 | 73/26/0 | 1.37x |
| `xchacha20-poly1305` | 198 | 171/27/0 | 1.45x | 198 | 171/27/0 | 1.45x |
| `ecdsa-p256` | 72 | 61/11/0 | 1.52x | 200 | 186/14/0 | 2.41x |
| `rsa-4096` | 27 | 23/2/2 | 1.57x | 59 | 55/2/2 | 2.52x |
| `rsa-3072` | 27 | 24/1/2 | 1.64x | 59 | 56/1/2 | 2.56x |
| `crc32` | 99 | 46/34/19 | 1.68x | 198 | 140/38/20 | 2.30x |
| `rsa-2048` | 27 | 25/0/2 | 1.70x | 59 | 57/0/2 | 2.61x |
| `crc64-nvme` | 99 | 50/36/13 | 1.70x | 198 | 133/51/14 | 2.25x |
| `kmac256` | 99 | 61/23/15 | 1.74x | 99 | 61/23/15 | 1.74x |
| `aes-128-gcm` | 198 | 117/41/40 | 1.75x | 550 | 431/52/67 | 2.54x |
| `aes-256-gcm` | 198 | 111/34/53 | 1.77x | 550 | 423/41/86 | 2.57x |
| `cshake256` | 99 | 61/28/10 | 1.78x | 99 | 61/28/10 | 1.78x |
| `shake128` | 99 | 77/22/0 | 1.84x | 99 | 77/22/0 | 1.84x |
| `shake256` | 99 | 70/29/0 | 1.87x | 99 | 70/29/0 | 1.87x |
| `sha224` | 99 | 54/43/2 | 1.89x | 99 | 54/43/2 | 1.89x |
| `aes-128-gcm-siv` | 198 | 137/4/57 | 2.03x | 352 | 268/19/65 | 2.80x |
| `sha3-256` | 117 | 104/13/0 | 2.10x | 117 | 104/13/0 | 2.10x |
| `sha3-224` | 99 | 89/10/0 | 2.12x | 99 | 89/10/0 | 2.12x |
| `sha3-384` | 99 | 89/10/0 | 2.14x | 99 | 89/10/0 | 2.14x |
| `aes-256-gcm-siv` | 198 | 138/5/55 | 2.15x | 352 | 291/6/55 | 3.06x |
| `sha3-512` | 99 | 88/11/0 | 2.19x | 99 | 88/11/0 | 2.19x |
| `crc64-xz` | 99 | 75/12/12 | 2.38x | 99 | 75/12/12 | 2.38x |
| `crc24-openpgp` | 99 | 94/2/3 | 13.08x | 99 | 94/2/3 | 13.08x |
| `crc16-ccitt` | 99 | 98/0/1 | 23.10x | 99 | 98/0/1 | 23.10x |
| `crc16-ibm` | 99 | 98/0/1 | 23.83x | 99 | 98/0/1 | 23.83x |

## Linux Clear Losses

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | --- | --- | --- | --- | --- |
| `ecdsa-p384` / `sign` | 36 | 16/1/19 | 0.95x | 0.91x | `aws-lc-rs` 20 |
| `argon2id-owasp` / `hash` | 9 | 3/1/5 | 0.95x | 0.92x | `rustcrypto` 6 |
| `ed25519` / `verify` | 36 | 8/18/10 | 0.98x | 1.00x | `dalek` 10, `aws-lc-rs` 9, `ring` 9 |
| `chacha20-poly1305` / `encrypt` | 99 | 31/33/35 | 0.99x | 1.00x | `ring` 53, `aws-lc-rs` 12, `rustcrypto` 3 |
| `chacha20-poly1305` / `decrypt` | 99 | 39/26/34 | 1.00x | 1.00x | `aws-lc-rs` 34, `ring` 26 |
| `argon2id-small` / `hash` | 27 | 12/3/12 | 1.00x | 0.99x | `rustcrypto` 15 |
| `argon2i-small` / `hash` | 27 | 12/3/12 | 1.01x | 0.98x | `rustcrypto` 15 |
| `x25519` / `public-key-from-secret` | 9 | 0/9/0 | 1.01x | 1.01x | `aws-lc-rs` 7, `dryoc` 2 |
| `argon2d-small` / `hash` | 27 | 11/6/10 | 1.01x | 0.99x | `rustcrypto` 16 |
| `rapidhash-v3-64` / `hash` | 99 | 22/55/22 | 1.04x | 1.00x | `rapidhash` 77 |
| `x25519` / `diffie-hellman` | 9 | 3/6/0 | 1.05x | 1.01x | `aws-lc-rs` 6 |
| `ed25519` / `sign` | 36 | 10/24/2 | 1.05x | 1.02x | `aws-lc-rs` 23, `dalek` 2, `dryoc` 1 |
| `rapidhash-v3-128` / `hash` | 99 | 31/49/19 | 1.07x | 1.00x | `rapidhash` 68 |
| `rapidhash-64` / `hash` | 99 | 31/56/12 | 1.12x | 1.01x | `rapidhash` 68 |
| `rapidhash-128` / `hash` | 99 | 36/54/9 | 1.13x | 1.02x | `rapidhash` 63 |
| `xxh3-128` / `hash` | 99 | 50/32/17 | 1.16x | 1.06x | `xxhash-rust` 49 |
| `xxh3-64` / `hash` | 99 | 51/34/14 | 1.17x | 1.06x | `xxhash-rust` 48 |
| `rsa-4096` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.17x | 1.13x | `ring` 1, `aws-lc-rs` 1 |
| `rsa-4096` / `verify-pss-sha256` | 9 | 7/1/1 | 1.18x | 1.12x | `ring` 1, `aws-lc-rs` 1 |
| `rsa-8192` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.18x | 1.09x | `ring` 1, `aws-lc-rs` 1 |
| `hmac-sha256` / `streaming` | 18 | 3/13/2 | 1.19x | 1.00x | `rustcrypto` 15 |
| `sha256` / `streaming` | 18 | 3/13/2 | 1.19x | 1.00x | `sha2` 15 |
| `rsa-8192` / `verify-pss-sha256` | 9 | 7/1/1 | 1.19x | 1.09x | `ring` 1, `aws-lc-rs` 1 |
| `rsa-3072` / `verify-pss-sha256` | 9 | 8/0/1 | 1.20x | 1.14x | `aws-lc-rs` 1 |
| `rsa-3072` / `verify-pkcs1v15-sha256` | 9 | 7/1/1 | 1.20x | 1.16x | `ring` 1, `aws-lc-rs` 1 |
| `hkdf-sha256` / `expand` | 36 | 30/6/0 | 1.22x | 1.11x | `rustcrypto` 6 |
| `pbkdf2-sha256` / `hash` | 54 | 49/5/0 | 1.22x | 1.14x | `rustcrypto` 4, `aws-lc-rs` 1 |
| `pbkdf2-sha512` / `hash` | 54 | 45/9/0 | 1.23x | 1.11x | `aws-lc-rs` 6, `rustcrypto` 3 |
| `hmac-sha512` / `hash` | 99 | 49/42/8 | 1.24x | 1.05x | `rustcrypto` 27, `ring` 14, `aws-lc-rs` 9 |
| `sha256` / `hash` | 99 | 40/46/13 | 1.24x | 1.01x | `aws-lc-rs` 31, `sha2` 21, `ring` 7 |
| `sha384` / `hash` | 99 | 38/51/10 | 1.24x | 1.04x | `aws-lc-rs` 36, `sha2` 17, `ring` 8 |
| `scrypt-owasp` / `hash` | 9 | 5/2/2 | 1.25x | 1.05x | `rustcrypto` 4 |
| `sha512` / `streaming` | 18 | 7/11/0 | 1.25x | 1.04x | `sha2` 11 |
| `hkdf-sha384` / `expand` | 36 | 34/2/0 | 1.25x | 1.11x | `rustcrypto` 2 |
| `sha512` / `hash` | 99 | 41/49/9 | 1.25x | 1.05x | `aws-lc-rs` 33, `sha2` 16, `ring` 9 |
| `hmac-sha384` / `hash` | 99 | 46/45/8 | 1.25x | 1.05x | `rustcrypto` 29, `ring` 15, `aws-lc-rs` 9 |
| `rsa-2048` / `verify-pkcs1v15-sha256` | 9 | 8/0/1 | 1.26x | 1.22x | `aws-lc-rs` 1 |
| `rsa-2048` / `verify-pss-sha256` | 9 | 8/0/1 | 1.27x | 1.24x | `aws-lc-rs` 1 |
| `aegis-256` / `encrypt` | 99 | 32/43/24 | 1.27x | 1.00x | `aegis-crate` 67 |
| `scrypt-small` / `hash` | 36 | 22/12/2 | 1.33x | 1.18x | `rustcrypto` 14 |
| `hmac-sha256` / `hash` | 99 | 61/26/12 | 1.38x | 1.12x | `ring` 21, `rustcrypto` 10, `aws-lc-rs` 7 |
| `ed25519` / `public-key-from-secret` | 9 | 7/1/1 | 1.47x | 1.22x | `dalek` 2 |
| `aegis-256` / `decrypt` | 99 | 57/41/1 | 1.47x | 1.11x | `aegis-crate` 42 |
| `aes-128-gcm` / `encrypt` | 99 | 56/19/24 | 1.68x | 1.08x | `aws-lc-rs` 21, `rustcrypto` 18, `ring` 4 |
| `crc32` / `hash` | 99 | 46/34/19 | 1.68x | 1.04x | `crc-fast` 37, `crc32fast` 16 |
| `aes-256-gcm` / `encrypt` | 99 | 54/16/29 | 1.70x | 1.07x | `aws-lc-rs` 22, `rustcrypto` 19, `ring` 4 |
| `crc64-nvme` / `hash` | 99 | 50/36/13 | 1.70x | 1.05x | `crc-fast` 27, `crc64fast-nvme` 22 |
| `kmac256` / `hash` | 99 | 61/23/15 | 1.74x | 1.93x | `tiny-keccak` 38 |
| `cshake256` / `hash` | 99 | 61/28/10 | 1.78x | 2.00x | `tiny-keccak` 38 |
| `aes-128-gcm` / `decrypt` | 99 | 61/22/16 | 1.84x | 1.20x | `aws-lc-rs` 24, `rustcrypto` 11, `ring` 3 |
| `aes-256-gcm` / `decrypt` | 99 | 57/18/24 | 1.85x | 1.17x | `aws-lc-rs` 29, `rustcrypto` 11, `ring` 2 |
| `sha224` / `hash` | 99 | 54/43/2 | 1.89x | 1.10x | `sha2` 45 |
| `aes-128-gcm-siv` / `decrypt` | 99 | 68/2/29 | 1.98x | 1.63x | `aws-lc-rs` 25, `rustcrypto` 6 |
| `aes-256-gcm-siv` / `decrypt` | 99 | 69/2/28 | 2.07x | 1.53x | `aws-lc-rs` 24, `rustcrypto` 6 |
| `aes-128-gcm-siv` / `encrypt` | 99 | 69/2/28 | 2.08x | 1.78x | `aws-lc-rs` 24, `rustcrypto` 6 |
| `aes-256-gcm-siv` / `encrypt` | 99 | 69/3/27 | 2.24x | 1.87x | `aws-lc-rs` 24, `rustcrypto` 6 |
| `crc64-xz` / `hash` | 99 | 75/12/12 | 2.38x | 2.06x | `crc64fast` 24 |
| `crc24-openpgp` / `hash` | 99 | 94/2/3 | 13.08x | 15.05x | `crc` 5 |
| `crc16-ccitt` / `hash` | 99 | 98/0/1 | 23.10x | 37.00x | `crc` 1 |
| `crc16-ibm` / `hash` | 99 | 98/0/1 | 23.83x | 37.10x | `crc` 1 |

## Apple Silicon Clear Losses

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | --- | --- | --- | --- | --- |
| `sha3-256` / `streaming` | 2 | 0/0/2 | 0.93x | 0.93x | `sha3` 2 |
| `sha3-256` / `hash` | 11 | 0/4/7 | 0.94x | 0.93x | `sha3` 11 |
| `hkdf-sha256` / `expand` | 4 | 0/3/1 | 0.96x | 0.97x | `rustcrypto` 4 |
| `xxh3-64` / `hash` | 11 | 1/4/6 | 0.97x | 0.85x | `xxhash-rust` 10 |
| `sha3-512` / `hash` | 11 | 0/8/3 | 0.97x | 0.99x | `sha3` 11 |
| `ed25519` / `verify` | 4 | 0/3/1 | 0.99x | 1.01x | `aws-lc-rs` 4 |
| `sha3-384` / `hash` | 11 | 0/11/0 | 0.99x | 1.00x | `sha3` 11 |
| `argon2i-small` / `hash` | 3 | 0/3/0 | 1.00x | 1.00x | `rustcrypto` 3 |
| `pbkdf2-sha256` / `hash` | 6 | 0/5/1 | 1.00x | 1.01x | `rustcrypto` 6 |
| `sha512` / `streaming` | 2 | 0/2/0 | 1.00x | 1.00x | `sha2` 2 |
| `hkdf-sha384` / `expand` | 4 | 0/4/0 | 1.00x | 1.00x | `rustcrypto` 4 |
| `argon2id-small` / `hash` | 3 | 0/3/0 | 1.00x | 1.00x | `rustcrypto` 3 |
| `chacha20-poly1305` / `encrypt` | 11 | 1/9/1 | 1.00x | 1.01x | `ring` 8, `aws-lc-rs` 2 |
| `argon2d-small` / `hash` | 3 | 0/3/0 | 1.00x | 1.00x | `rustcrypto` 3 |
| `x25519` / `public-key-from-secret` | 1 | 0/1/0 | 1.00x | 1.00x | `aws-lc-rs` 1 |
| `sha3-224` / `hash` | 11 | 1/10/0 | 1.01x | 1.01x | `sha3` 10 |
| `argon2id-owasp` / `hash` | 1 | 0/1/0 | 1.01x | 1.01x | `rustcrypto` 1 |
| `x25519` / `diffie-hellman` | 1 | 0/1/0 | 1.02x | 1.02x | `aws-lc-rs` 1 |
| `chacha20-poly1305` / `decrypt` | 11 | 1/10/0 | 1.03x | 1.01x | `ring` 6, `aws-lc-rs` 4 |
| `hmac-sha384` / `hash` | 11 | 4/7/0 | 1.03x | 1.01x | `rustcrypto` 5, `ring` 2 |
| `ed25519` / `sign` | 4 | 1/3/0 | 1.04x | 1.04x | `aws-lc-rs` 3 |
| `hmac-sha512` / `hash` | 11 | 5/6/0 | 1.04x | 1.04x | `rustcrypto` 6 |
| `ecdsa-p256` / `verify` | 4 | 1/3/0 | 1.04x | 1.02x | `aws-lc-rs` 3 |
| `sha256` / `hash` | 11 | 3/8/0 | 1.04x | 1.00x | `ring` 5, `aws-lc-rs` 2, `sha2` 1 |
| `sha256` / `streaming` | 2 | 1/1/0 | 1.05x | 1.05x | `sha2` 1 |
| `pbkdf2-sha512` / `hash` | 6 | 1/5/0 | 1.05x | 1.04x | `rustcrypto` 5 |
| `ascon-aead128` / `encrypt` | 11 | 7/4/0 | 1.06x | 1.06x | `ascon-aead` 4 |
| `sha384` / `hash` | 11 | 4/6/1 | 1.06x | 0.99x | `sha2` 6, `ring` 1 |
| `ascon-hash256` / `hash` | 11 | 8/3/0 | 1.06x | 1.07x | `ascon-hash` 3 |
| `rapidhash-v3-64` / `hash` | 11 | 3/7/1 | 1.06x | 0.99x | `rapidhash` 8 |

## Linux Worst Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| AMD Zen4 | `xxh3-64 / 256` | `xxhash-rust` | 0.35x |
| AMD Zen4 | `xxh3-128 / 256` | `xxhash-rust` | 0.43x |
| IBM z16/s390x | `rapidhash-v3-64 / 0` | `rapidhash` | 0.43x |
| RISE RISC-V | `xxh3-64 / 0` | `xxhash-rust` | 0.46x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 1024` | `aws-lc-rs` | 0.48x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 32` | `aws-lc-rs` | 0.48x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 0` | `aws-lc-rs` | 0.48x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.50x |
| AMD Zen4 | `ecdsa-p384 / sign / 32` | `aws-lc-rs` | 0.50x |
| AMD Zen4 | `ecdsa-p384 / sign / 0` | `aws-lc-rs` | 0.50x |
| Intel Ice Lake | `aes-128-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.51x |

## Linux Strongest Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt / 16384` | `crc` | 225.43x |
| Intel Sapphire Rapids | `crc16-ccitt / 262144` | `crc` | 220.23x |
| Intel Sapphire Rapids | `crc16-ibm / 262144` | `crc` | 215.65x |
| Intel Sapphire Rapids | `crc16-ibm / 16384` | `crc` | 197.21x |
| Intel Sapphire Rapids | `crc16-ibm / 65536` | `crc` | 181.87x |
| Intel Sapphire Rapids | `crc16-ibm / 1048576` | `crc` | 179.59x |
| Intel Sapphire Rapids | `crc16-ccitt / 65536` | `crc` | 179.59x |
| Intel Sapphire Rapids | `crc16-ccitt / 1048576` | `crc` | 177.90x |
| IBM Power10 | `crc16-ibm / 1048576` | `crc` | 177.05x |
| IBM Power10 | `crc16-ccitt / 1048576` | `crc` | 176.78x |
| IBM Power10 | `crc16-ibm / 262144` | `crc` | 175.85x |
| IBM Power10 | `crc16-ccitt / 262144` | `crc` | 175.39x |

## Top Five Loss Areas

- `ecdsa-p384` / `sign`: 0.95x geomean across 36 rows; W/T/L 16/1/19; pressure `aws-lc-rs` 20.
- `argon2id-owasp` / `hash`: 0.95x geomean across 9 rows; W/T/L 3/1/5; pressure `rustcrypto` 6.
- `ed25519` / `verify`: 0.98x geomean across 36 rows; W/T/L 8/18/10; pressure `dalek` 10, `aws-lc-rs` 9, `ring` 9.
- `chacha20-poly1305` / `encrypt`: 0.99x geomean across 99 rows; W/T/L 31/33/35; pressure `ring` 53, `aws-lc-rs` 12, `rustcrypto` 3.
- `chacha20-poly1305` / `decrypt`: 1.00x geomean across 99 rows; W/T/L 39/26/34; pressure `aws-lc-rs` 34, `ring` 26.

## External Pressure

All-pair Linux CI comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `rapidhash` | 396 | 120/214/62 | 30% | 1.09x | 1.01x |
| `xxhash-rust` | 198 | 101/66/31 | 51% | 1.17x | 1.06x |
| `aws-lc-rs` | 1,610 | 973/343/294 | 60% | 1.23x | 1.13x |
| `ascon-hash` | 198 | 129/69/0 | 65% | 1.32x | 1.25x |
| `ascon-aead` | 198 | 151/47/0 | 76% | 1.33x | 1.15x |
| `aegis-crate` | 198 | 89/84/25 | 45% | 1.37x | 1.03x |
| `dalek` | 108 | 82/18/8 | 76% | 1.45x | 1.22x |
| `sha2` | 531 | 293/233/5 | 55% | 1.49x | 1.07x |
| `dryoc` | 135 | 110/15/10 | 81% | 1.61x | 1.52x |
| `ring` | 1,656 | 1309/195/152 | 79% | 1.62x | 1.29x |
| `tiny-keccak` | 396 | 269/102/25 | 68% | 1.81x | 1.93x |
| `sha3` | 414 | 370/44/0 | 89% | 2.13x | 2.11x |
| `crc-fast` | 198 | 117/69/12 | 59% | 2.15x | 1.19x |
| `crc64fast-nvme` | 99 | 74/13/12 | 75% | 2.35x | 2.03x |
| `crc64fast` | 99 | 75/12/12 | 76% | 2.38x | 2.06x |
| `crc32fast` | 99 | 82/7/10 | 83% | 2.47x | 2.08x |
| `rustcrypto` | 1,962 | 1572/256/134 | 80% | 2.66x | 1.53x |
| `rustcrypto-rsa` | 81 | 81/0/0 | 100% | 5.68x | 4.57x |
| `crc` | 297 | 290/2/5 | 98% | 19.31x | 29.41x |

## README Numbers

- **Headline:** 3,287 of 5,292 matched Linux CI fastest-external comparisons are wins; 4,677 are wins or ties. Linux CI geomean is 1.71x.
- **Apple Silicon headline:** 346 of 588 matched fastest-external comparisons are wins; 560 are wins or ties. Apple Silicon geomean is 1.49x.
- **Checksums:** 6.05x geomean across 594 Linux CI fastest-external rows; W/T/L 461/84/49.
- **Hashes/MACs/XOFs:** 1.45x geomean across 2,448 Linux CI fastest-external rows; W/T/L 1385/879/184.
- **Auth/KDF:** 1.23x geomean across 180 Linux CI fastest-external rows; W/T/L 158/22/0.
- **Password hashing:** 1.09x geomean across 135 Linux CI fastest-external rows; W/T/L 65/27/43.
- **Public-key:** 1.26x geomean across 252 Linux CI fastest-external rows; W/T/L 148/72/32.
- **RSA:** 1.54x geomean across 99 Linux CI fastest-external rows; W/T/L 86/5/8.
- **AEAD:** 1.56x geomean across 1,584 Linux CI fastest-external rows; W/T/L 984/301/299.
- **ECDSA P-256/P-384:** 1.41x Linux CI geomean across 144 fastest-external rows; Apple Silicon 1.26x across 16 rows.
- **Apple Silicon AEAD:** 1.48x geomean across 176 fastest-external rows.
- **Current top losses:** `ecdsa-p384` / `sign`: 0.95x geomean across 36 rows; W/T/L 16/1/19; pressure `aws-lc-rs` 20; `argon2id-owasp` / `hash`: 0.95x geomean across 9 rows; W/T/L 3/1/5; pressure `rustcrypto` 6; `ed25519` / `verify`: 0.98x geomean across 36 rows; W/T/L 8/18/10; pressure `dalek` 10, `aws-lc-rs` 9, `ring` 9; `chacha20-poly1305` / `encrypt`: 0.99x geomean across 99 rows; W/T/L 31/33/35; pressure `ring` 53, `aws-lc-rs` 12, `rustcrypto` 3; `chacha20-poly1305` / `decrypt`: 1.00x geomean across 99 rows; W/T/L 39/26/34; pressure `aws-lc-rs` 34, `ring` 26.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-06-12 01_00_42` | 1,999 | `benchmark_results/2026-06-12/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-06-12 01_00_42` | 1,999 | `benchmark_results/2026-06-12/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-06-12 01_00_42` | 1,999 | `benchmark_results/2026-06-12/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-06-12 01_00_42` | 1,999 | `benchmark_results/2026-06-12/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-06-12 01_00_42` | 1,769 | `benchmark_results/2026-06-12/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-06-12 01_00_42` | 1,769 | `benchmark_results/2026-06-12/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-06-12 01_00_42` | 1,999 | `benchmark_results/2026-06-12/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-06-12 01_00_42` | 1,999 | `benchmark_results/2026-06-12/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-06-12 01_00_42` | 1,999 | `benchmark_results/2026-06-12/linux/rise-riscv/results.txt` |
| Apple Silicon macOS/aarch64 | `local` | `2026-06-11 23_06_36` | 2,045 | `benchmark_results/2026-06-11/macos/aarch64/results.txt` |
