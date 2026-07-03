# Benchmark Overview

Sources:

- Linux benchmark CI run [#28461694270](https://github.com/loadingalias/rscrypto/actions/runs/28461694270), created 2026-06-30 16:58:33 UTC.
- Linux commit: `1bcc2d19cf7b2df93ad3e814a1763ab9ce7e84b4`.
- Linux artifacts: nine successful `benchmark-*` artifacts extracted into `benchmark_results/2026-06-30/linux/*/results.txt`.
- Local macOS run: `benchmark_results/2026-07-03/macos/aarch64/results.txt` at commit `6e0a87146c9a187cf5c9d6225ae4452d06504edf`.

Scope: the 2026-06-30 nine-runner Linux CI benchmark matrix for commit `1bcc2d1`. Ratios are `external_crate_time / rscrypto_time`; higher is better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and input shape. Internal kernel, scratch-buffer, padding-only, cold-path, PHC roundtrip, parallel-scaling, threshold-selection, public-overhead, and phase-attribution microbenches are parsed as raw rows but excluded from external win/loss claims. The macOS local run is listed separately and is not mixed into Linux CI claims.

Coverage note: this is a full Linux CI public benchmark pass. It includes checksum, hash, XOF, MAC, KDF, password-hashing, BLAKE2/BLAKE3, RSA import/verification, ECDSA P-256/P-384 signing and verification, Ed25519, X25519, AEAD, and ML-KEM-512/768/1024 keygen, encapsulation, and decapsulation rows. ML-KEM phase/arithmetic microbenches are present in the raw artifacts and intentionally excluded from release-level competitor claims.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Linux CI: all matched performance pairs | 10,781 | 7,477/2,434/870 | 69% | 1.75x | 1.22x |
| Linux CI: fastest external per case | 6,750 | 4,040/2,007/703 | 60% | 1.59x | 1.11x |

Shareable release summary:

- **Headline:** 4,040 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,047 are wins or ties. Linux CI fastest-external geomean is 1.59x.
- **Checksums:** 5.20x geomean across 693 fastest-external rows; W/T/L is 509/128/56.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,726 fastest-external rows; W/T/L is 2,060/1,440/226.
- **Auth/KDF:** 1.25x geomean across 180 fastest-external rows; W/T/L is 162/18/0.
- **Password hashing:** 1.08x geomean across 135 fastest-external rows; W/T/L is 69/27/39.
- **Public-key:** 1.32x geomean across 333 fastest-external rows; W/T/L is 210/72/51.
- **RSA:** 1.55x geomean across 99 fastest-external rows; W/T/L is 88/3/8.
- **AEAD:** 1.54x geomean across 1,584 fastest-external rows; W/T/L is 942/319/323.
- **ML-KEM:** 1.45x geomean across 81 fastest-external rows; W/T/L is 58/4/19.
- **ECDSA P-256/P-384:** Linux CI 1.45x geomean across 144 fastest-external rows; W/T/L is 113/11/20.
- **Top current loss areas:** `mlkem1024` / `keygen`: 0.82x geomean across 9 rows; W/T/L 1/0/8; pressure `libcrux` 4, `aws-lc-rs` 4; `mlkem768` / `keygen`: 0.92x geomean across 9 rows; W/T/L 1/1/7; pressure `aws-lc-rs` 4, `libcrux` 3; `argon2id-owasp` / `hash`: 0.96x geomean across 9 rows; W/T/L 3/2/4; pressure `rustcrypto` 4; `blake2b256` / `streaming`: 0.99x geomean across 27 rows; W/T/L 8/16/3; pressure `rustcrypto` 3; `chacha20-poly1305` / `encrypt`: 0.99x geomean across 99 rows; W/T/L 30/35/34; pressure `ring` 27, `aws-lc-rs` 7.

## Coverage Matrix

| Platform | Raw Criterion rows | All pairs | Fastest rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 2,332 | 1,251 | 750 | 522/158/70 | 70% | 1.48x | 1.15x |
| AMD Zen5 | 2,332 | 1,251 | 750 | 438/242/70 | 58% | 1.50x | 1.09x |
| AWS Graviton3 | 2,340 | 1,251 | 750 | 353/303/94 | 47% | 1.37x | 1.04x |
| AWS Graviton4 | 2,340 | 1,251 | 750 | 364/329/57 | 49% | 1.38x | 1.05x |
| IBM Power10 | 2,083 | 1,012 | 750 | 385/319/46 | 51% | 1.88x | 1.06x |
| IBM z16/s390x | 2,083 | 1,012 | 750 | 602/94/54 | 80% | 3.15x | 2.35x |
| Intel Ice Lake | 2,332 | 1,251 | 750 | 519/144/87 | 69% | 1.49x | 1.20x |
| Intel Sapphire Rapids | 2,332 | 1,251 | 750 | 529/146/75 | 71% | 1.62x | 1.22x |
| RISE RISC-V | 2,332 | 1,251 | 750 | 328/272/150 | 44% | 1.09x | 1.02x |

## Category Summary

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 693 | 509/128/56 | 73% | 5.20x | 2.43x |
| Hashes/MACs/XOFs | 3,726 | 2,060/1,440/226 | 55% | 1.35x | 1.08x |
| Auth/KDF | 180 | 162/18/0 | 90% | 1.25x | 1.13x |
| Password hashing | 135 | 69/27/39 | 51% | 1.08x | 1.06x |
| Public-key | 333 | 210/72/51 | 63% | 1.32x | 1.18x |
| RSA | 99 | 88/3/8 | 89% | 1.55x | 1.16x |
| AEAD | 1,584 | 942/319/323 | 59% | 1.54x | 1.13x |

## BLAKE3 Summary

BLAKE3 rows come from Linux CI run [#28461694270](https://github.com/loadingalias/rscrypto/actions/runs/28461694270). All-pair and fastest-external BLAKE3 metrics are identical because official `blake3` is the only external implementation in this bench.

| Scope | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| All Linux BLAKE3 rows | 432 | 228/172/32 | 1.40x | 1.06x |
| x86_64 | 192 | 82/96/14 | 1.24x | 1.03x |
| AArch64 | 96 | 44/46/6 | 1.44x | 1.05x |

| Platform | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| AMD Zen4 | 48 | 21/24/3 | 1.32x | 1.03x |
| AMD Zen5 | 48 | 21/25/2 | 1.32x | 1.03x |
| AWS Graviton3 | 48 | 22/21/5 | 1.40x | 0.99x |
| AWS Graviton4 | 48 | 22/25/1 | 1.47x | 1.05x |
| IBM Power10 | 48 | 39/7/2 | 1.99x | 1.83x |
| IBM z16/s390x | 48 | 40/3/5 | 1.84x | 2.05x |
| Intel Ice Lake | 48 | 17/28/3 | 1.16x | 1.02x |
| Intel Sapphire Rapids | 48 | 23/19/6 | 1.16x | 1.05x |
| RISE RISC-V | 48 | 23/20/5 | 1.13x | 1.02x |

| Operation | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `oneshot` | 99 | 43/48/8 | 1.32x | 1.03x |
| `keyed` | 99 | 42/45/12 | 1.32x | 1.02x |
| `derive-key` | 99 | 78/21/0 | 1.78x | 1.84x |
| `streaming` | 36 | 12/19/5 | 1.15x | 1.01x |
| `xof` | 99 | 53/39/7 | 1.31x | 1.06x |

## ML-KEM Summary

ML-KEM public coverage is complete for the CI-selected primitive set: ML-KEM-512, ML-KEM-768, and ML-KEM-1024 each include keygen, encapsulate, and decapsulate on all nine Linux platforms. POWER10 and s390x do not have `aws-lc-rs` ML-KEM rows in this artifact set, but still have rscrypto plus `libcrux`, `fips203`, and RustCrypto comparison rows for every public operation.

| Platform | Raw ML-KEM rows | Fastest rows | W/T/L | Geomean | Median | Fastest external split |
| --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 45 | 9 | 7/0/2 | 1.68x | 1.72x | `libcrux` 7, `aws-lc-rs` 2 |
| AMD Zen5 | 45 | 9 | 7/0/2 | 1.80x | 1.73x | `libcrux` 9 |
| AWS Graviton3 | 45 | 9 | 5/1/3 | 1.10x | 1.14x | `aws-lc-rs` 9 |
| AWS Graviton4 | 45 | 9 | 5/0/4 | 1.08x | 1.14x | `aws-lc-rs` 9 |
| IBM Power10 | 36 | 9 | 6/2/1 | 1.41x | 1.59x | `libcrux` 9 |
| IBM z16/s390x | 36 | 9 | 9/0/0 | 1.69x | 1.60x | `libcrux` 9 |
| Intel Ice Lake | 45 | 9 | 7/0/2 | 1.65x | 1.73x | `libcrux` 7, `aws-lc-rs` 2 |
| Intel Sapphire Rapids | 45 | 9 | 7/0/2 | 1.73x | 1.82x | `aws-lc-rs` 6, `libcrux` 3 |
| RISE RISC-V | 45 | 9 | 5/1/3 | 1.12x | 1.09x | `aws-lc-rs` 9 |

| Primitive/op | Rows | W/T/L | Win % | Geomean | Median | Pressure |
| --- | --- | --- | --- | --- | --- | --- |
| `mlkem1024` / `decapsulate` | 9 | 9/0/0 | 100% | 1.62x | 1.84x | none |
| `mlkem1024` / `encapsulate` | 9 | 9/0/0 | 100% | 2.43x | 2.57x | none |
| `mlkem1024` / `keygen` | 9 | 1/0/8 | 11% | 0.82x | 0.78x | `libcrux` 4, `aws-lc-rs` 4 |
| `mlkem512` / `decapsulate` | 9 | 6/2/1 | 67% | 1.32x | 1.50x | `aws-lc-rs` 1 |
| `mlkem512` / `encapsulate` | 9 | 9/0/0 | 100% | 1.88x | 1.73x | none |
| `mlkem512` / `keygen` | 9 | 5/1/3 | 56% | 1.05x | 1.18x | `aws-lc-rs` 3 |
| `mlkem768` / `decapsulate` | 9 | 9/0/0 | 100% | 1.55x | 1.72x | none |
| `mlkem768` / `encapsulate` | 9 | 9/0/0 | 100% | 2.31x | 2.00x | none |
| `mlkem768` / `keygen` | 9 | 1/1/7 | 11% | 0.92x | 0.89x | `aws-lc-rs` 4, `libcrux` 3 |

## ECDSA Summary

ECDSA signing includes both deterministic and blinded rscrypto rows in raw results; aggregate fastest-external comparisons use the fastest rscrypto row for the exact case. Constant-time release evidence is tracked separately by `ct.toml` and CT workflow artifacts.

| Operation | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `ecdsa-p256` / `sign` | 36 | 36/0/0 | 1.44x | 1.34x |
| `ecdsa-p256` / `verify` | 36 | 25/11/0 | 1.59x | 1.14x |
| `ecdsa-p384` / `sign` | 36 | 16/0/20 | 1.07x | 0.92x |
| `ecdsa-p384` / `verify` | 36 | 36/0/0 | 1.80x | 1.39x |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are strongest-external comparisons; all-pair columns include every matched external implementation.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `argon2id-owasp` | 9 | 3/2/4 | 0.96x | 18 | 10/3/5 | 1.25x |
| `chacha20-poly1305` | 198 | 63/66/69 | 0.99x | 550 | 282/148/120 | 1.15x |
| `argon2i-small` | 27 | 13/2/12 | 1.03x | 45 | 30/3/12 | 1.35x |
| `argon2d-small` | 27 | 13/4/10 | 1.04x | 27 | 13/4/10 | 1.04x |
| `x25519` | 18 | 5/13/0 | 1.04x | 50 | 36/14/0 | 1.56x |
| `argon2id-small` | 27 | 13/4/10 | 1.05x | 45 | 30/4/11 | 1.39x |
| `rapidhash-v3-64` | 99 | 26/52/21 | 1.05x | 99 | 26/52/21 | 1.05x |
| `blake2b256` | 225 | 120/98/7 | 1.06x | 351 | 237/106/8 | 1.32x |
| `rapidhash-v3-128` | 99 | 33/46/20 | 1.08x | 99 | 33/46/20 | 1.08x |
| `blake2b512` | 198 | 116/81/1 | 1.08x | 297 | 212/84/1 | 1.34x |
| `blake2s256` | 225 | 112/111/2 | 1.10x | 225 | 112/111/2 | 1.10x |
| `scrypt-owasp` | 9 | 5/2/2 | 1.11x | 9 | 5/2/2 | 1.11x |
| `ed25519` | 90 | 34/44/12 | 1.11x | 290 | 206/69/15 | 1.37x |
| `blake2s128` | 198 | 119/79/0 | 1.12x | 198 | 119/79/0 | 1.12x |
| `rapidhash-64` | 99 | 32/57/10 | 1.12x | 99 | 32/57/10 | 1.12x |
| `rapidhash-128` | 99 | 37/54/8 | 1.13x | 99 | 37/54/8 | 1.13x |
| `xxh3-128` | 99 | 51/34/14 | 1.17x | 99 | 51/34/14 | 1.17x |
| `xxh3-64` | 99 | 54/33/12 | 1.18x | 99 | 54/33/12 | 1.18x |
| `rsa-8192` | 18 | 14/2/2 | 1.20x | 32 | 28/2/2 | 1.26x |
| `scrypt-small` | 36 | 22/13/1 | 1.20x | 36 | 22/13/1 | 1.20x |
| `hmac-sha512` | 99 | 37/55/7 | 1.22x | 275 | 167/96/12 | 1.28x |
| `hmac-sha384` | 99 | 37/53/9 | 1.23x | 275 | 165/96/14 | 1.28x |
| `pbkdf2-sha512` | 54 | 47/7/0 | 1.24x | 150 | 143/7/0 | 1.31x |
| `sha256` | 117 | 45/57/15 | 1.24x | 293 | 164/100/29 | 1.54x |
| `pbkdf2-sha256` | 54 | 50/4/0 | 1.24x | 150 | 146/4/0 | 1.64x |
| `sha384` | 99 | 40/47/12 | 1.24x | 275 | 166/92/17 | 1.27x |
| `sha512` | 117 | 49/57/11 | 1.25x | 293 | 176/101/16 | 1.27x |
| `hkdf-sha384` | 36 | 36/0/0 | 1.25x | 100 | 100/0/0 | 1.57x |
| `ascon-hash256` | 99 | 55/42/2 | 1.26x | 99 | 55/42/2 | 1.26x |
| `hkdf-sha256` | 36 | 29/7/0 | 1.26x | 100 | 93/7/0 | 1.88x |
| `sha512-256` | 99 | 53/46/0 | 1.29x | 99 | 53/46/0 | 1.29x |
| `ascon-aead128` | 198 | 149/49/0 | 1.33x | 198 | 149/49/0 | 1.33x |
| `hmac-sha256` | 117 | 65/38/14 | 1.34x | 293 | 195/74/24 | 1.69x |
| `ascon-xof128` | 99 | 73/26/0 | 1.35x | 99 | 73/26/0 | 1.35x |
| `xchacha20-poly1305` | 198 | 170/26/2 | 1.36x | 198 | 170/26/2 | 1.36x |
| `mlkem512` | 27 | 20/3/4 | 1.37x | 102 | 95/3/4 | 2.64x |
| `aegis-256` | 198 | 92/81/25 | 1.38x | 198 | 92/81/25 | 1.38x |
| `ecdsa-p384` | 72 | 52/0/20 | 1.39x | 200 | 180/0/20 | 3.19x |
| `blake3` | 432 | 228/172/32 | 1.40x | 432 | 228/172/32 | 1.40x |
| `mlkem1024` | 27 | 19/0/8 | 1.48x | 102 | 91/0/11 | 3.08x |
| `mlkem768` | 27 | 19/1/7 | 1.49x | 102 | 91/4/7 | 3.02x |
| `ecdsa-p256` | 72 | 61/11/0 | 1.51x | 200 | 185/15/0 | 2.40x |
| `rsa-4096` | 27 | 24/1/2 | 1.58x | 59 | 56/1/2 | 2.52x |
| `crc32c` | 99 | 45/42/12 | 1.61x | 198 | 140/46/12 | 2.21x |
| `rsa-3072` | 27 | 25/0/2 | 1.64x | 59 | 57/0/2 | 2.57x |
| `crc32` | 99 | 48/31/20 | 1.66x | 198 | 144/34/20 | 2.26x |
| `aes-128-gcm` | 198 | 99/44/55 | 1.68x | 550 | 414/54/82 | 1.91x |
| `rsa-2048` | 27 | 25/0/2 | 1.69x | 59 | 57/0/2 | 2.62x |
| `aes-256-gcm` | 198 | 92/45/61 | 1.72x | 550 | 400/60/90 | 1.92x |
| `kmac256` | 99 | 61/22/16 | 1.74x | 99 | 61/22/16 | 1.74x |
| `cshake256` | 99 | 64/26/9 | 1.80x | 99 | 64/26/9 | 1.80x |
| `shake128` | 99 | 71/28/0 | 1.85x | 99 | 71/28/0 | 1.85x |
| `shake256` | 99 | 67/32/0 | 1.85x | 99 | 67/32/0 | 1.85x |
| `sha224` | 99 | 51/46/2 | 1.91x | 99 | 51/46/2 | 1.91x |
| `aes-128-gcm-siv` | 198 | 138/4/56 | 2.01x | 352 | 271/18/63 | 2.79x |
| `sha3-224` | 99 | 87/11/1 | 2.11x | 99 | 87/11/1 | 2.11x |
| `sha3-256` | 117 | 102/15/0 | 2.13x | 117 | 102/15/0 | 2.13x |
| `aes-256-gcm-siv` | 198 | 139/4/55 | 2.14x | 352 | 292/5/55 | 3.03x |
| `sha3-384` | 99 | 88/11/0 | 2.16x | 99 | 88/11/0 | 2.16x |
| `crc64-nvme` | 99 | 53/40/6 | 2.16x | 99 | 53/40/6 | 2.16x |
| `sha3-512` | 99 | 87/11/1 | 2.20x | 99 | 87/11/1 | 2.20x |
| `crc64-xz` | 99 | 73/14/12 | 2.40x | 99 | 73/14/12 | 2.40x |
| `crc24-openpgp` | 99 | 94/1/4 | 13.30x | 99 | 94/1/4 | 13.30x |
| `crc16-ccitt` | 99 | 98/0/1 | 23.16x | 99 | 98/0/1 | 23.16x |
| `crc16-ibm` | 99 | 98/0/1 | 24.08x | 99 | 98/0/1 | 24.08x |

## Linux Worst Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| RISE RISC-V | `xxh3-64 / 0` | `xxhash-rust` | 0.43x |
| Intel Sapphire Rapids | `aes-256-gcm / encrypt / 32` | `rustcrypto` | 0.46x |
| Intel Sapphire Rapids | `aes-128-gcm / encrypt / 32` | `rustcrypto` | 0.47x |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.47x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.48x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.48x |
| RISE RISC-V | `crc64-xz / 1048576` | `crc64fast` | 0.49x |
| Intel Sapphire Rapids | `aes-128-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Ice Lake | `aes-128-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| AMD Zen4 | `aes-256-gcm / encrypt / 32` | `rustcrypto` | 0.49x |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.50x |
| Intel Sapphire Rapids | `aes-128-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.50x |

## Linux Strongest Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ibm / 262144` | `crc` | 232.56x |
| Intel Sapphire Rapids | `crc16-ccitt / 262144` | `crc` | 229.85x |
| Intel Sapphire Rapids | `crc16-ccitt / 16384` | `crc` | 226.96x |
| Intel Sapphire Rapids | `crc16-ibm / 16384` | `crc` | 204.27x |
| Intel Sapphire Rapids | `crc16-ibm / 65536` | `crc` | 195.20x |
| Intel Sapphire Rapids | `crc16-ccitt / 65536` | `crc` | 193.31x |
| Intel Sapphire Rapids | `crc16-ibm / 1048576` | `crc` | 187.31x |
| Intel Sapphire Rapids | `crc16-ccitt / 4096` | `crc` | 184.65x |
| Intel Sapphire Rapids | `crc16-ibm / 4096` | `crc` | 182.82x |
| IBM Power10 | `crc16-ccitt / 1048576` | `crc` | 176.61x |
| IBM Power10 | `crc16-ibm / 1048576` | `crc` | 176.35x |
| IBM Power10 | `crc16-ccitt / 262144` | `crc` | 175.66x |

## Top Five Loss Areas

- `mlkem1024` / `keygen`: 0.82x geomean across 9 rows; W/T/L 1/0/8; pressure `libcrux` 4, `aws-lc-rs` 4.
- `mlkem768` / `keygen`: 0.92x geomean across 9 rows; W/T/L 1/1/7; pressure `aws-lc-rs` 4, `libcrux` 3.
- `argon2id-owasp` / `hash`: 0.96x geomean across 9 rows; W/T/L 3/2/4; pressure `rustcrypto` 4.
- `blake2b256` / `streaming`: 0.99x geomean across 27 rows; W/T/L 8/16/3; pressure `rustcrypto` 3.
- `chacha20-poly1305` / `encrypt`: 0.99x geomean across 99 rows; W/T/L 30/35/34; pressure `ring` 27, `aws-lc-rs` 7.

## External Pressure

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `rapidhash` | 396 | 128/209/59 | 32% | 1.10x | 1.01x |
| `xxhash-rust` | 198 | 105/67/26 | 53% | 1.18x | 1.09x |
| `aws-lc-rs` | 1,673 | 1,014/353/306 | 61% | 1.24x | 1.14x |
| `ascon-hash` | 198 | 128/68/2 | 65% | 1.31x | 1.18x |
| `ascon-aead` | 198 | 149/49/0 | 75% | 1.33x | 1.16x |
| `aegis-crate` | 198 | 92/81/25 | 46% | 1.38x | 1.04x |
| `blake3` | 432 | 228/172/32 | 53% | 1.40x | 1.06x |
| `dalek` | 108 | 87/18/3 | 81% | 1.48x | 1.28x |
| `sha2` | 531 | 284/241/6 | 53% | 1.52x | 1.06x |
| `ring` | 1,656 | 1,303/207/146 | 79% | 1.64x | 1.29x |
| `libcrux` | 81 | 70/3/8 | 86% | 1.74x | 1.73x |
| `tiny-keccak` | 396 | 263/108/25 | 66% | 1.81x | 1.97x |
| `dryoc` | 360 | 325/29/6 | 90% | 1.83x | 1.85x |
| `rustcrypto` | 2,889 | 2,075/646/168 | 72% | 1.86x | 1.20x |
| `crc-fast` | 297 | 169/105/23 | 57% | 2.01x | 1.15x |
| `sha3` | 414 | 364/48/2 | 88% | 2.15x | 2.14x |
| `crc64fast` | 99 | 73/14/12 | 74% | 2.40x | 2.01x |
| `crc32fast` | 99 | 82/6/11 | 83% | 2.42x | 2.01x |
| `crc32c` | 99 | 86/9/4 | 87% | 2.73x | 2.20x |
| `fips203` | 81 | 81/0/0 | 100% | 4.50x | 4.69x |
| `rustcrypto-rsa` | 81 | 81/0/0 | 100% | 5.68x | 4.82x |
| `crc` | 297 | 290/1/6 | 98% | 19.50x | 29.77x |

## macOS Local Snapshot

The macOS Apple Silicon run is local evidence from the 2026-07-03 full benchmark at commit `6e0a871`. It is useful for Apple Silicon planning but is not folded into Linux CI release claims. The ML-KEM row uses the same artifact's public ML-KEM rows.

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| macOS local: all matched performance pairs | 1,309 | 803/447/59 | 61% | 1.65x | 1.14x |
| macOS local: fastest external per case | 786 | 372/366/48 | 47% | 1.37x | 1.04x |
| macOS local: ML-KEM fastest external | 9 | 6/1/2 | 67% | 1.35x | 1.33x |

## README Numbers

- **Headline:** 4,040 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,047 are wins or ties. Linux CI geomean is 1.59x.
- **Checksums:** 5.20x geomean across 693 Linux CI fastest-external rows; W/T/L 509/128/56.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,726 Linux CI fastest-external rows; W/T/L 2,060/1,440/226.
- **Auth/KDF:** 1.25x geomean across 180 Linux CI fastest-external rows; W/T/L 162/18/0.
- **Password hashing:** 1.08x geomean across 135 Linux CI fastest-external rows; W/T/L 69/27/39.
- **Public-key:** 1.32x geomean across 333 Linux CI fastest-external rows; W/T/L 210/72/51.
- **RSA:** 1.55x geomean across 99 Linux CI fastest-external rows; W/T/L 88/3/8.
- **AEAD:** 1.54x geomean across 1,584 Linux CI fastest-external rows; W/T/L 942/319/323.
- **ML-KEM:** 1.45x geomean across 81 Linux CI fastest-external rows; W/T/L 58/4/19.
- **ECDSA P-256/P-384:** 1.45x Linux CI geomean across 144 fastest-external rows; W/T/L 113/11/20.
- **Current top losses:** `mlkem1024` / `keygen`: 0.82x geomean across 9 rows; W/T/L 1/0/8; pressure `libcrux` 4, `aws-lc-rs` 4; `mlkem768` / `keygen`: 0.92x geomean across 9 rows; W/T/L 1/1/7; pressure `aws-lc-rs` 4, `libcrux` 3; `argon2id-owasp` / `hash`: 0.96x geomean across 9 rows; W/T/L 3/2/4; pressure `rustcrypto` 4; `blake2b256` / `streaming`: 0.99x geomean across 27 rows; W/T/L 8/16/3; pressure `rustcrypto` 3; `chacha20-poly1305` / `encrypt`: 0.99x geomean across 99 rows; W/T/L 30/35/34; pressure `ring` 27, `aws-lc-rs` 7.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-06-30 16_58_33` | 2,332 | `benchmark_results/2026-06-30/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-06-30 16_58_33` | 2,332 | `benchmark_results/2026-06-30/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-06-30 16_58_33` | 2,340 | `benchmark_results/2026-06-30/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-06-30 16_58_33` | 2,340 | `benchmark_results/2026-06-30/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-06-30 16_58_33` | 2,083 | `benchmark_results/2026-06-30/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-06-30 16_58_33` | 2,083 | `benchmark_results/2026-06-30/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-06-30 16_58_33` | 2,332 | `benchmark_results/2026-06-30/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-06-30 16_58_33` | 2,332 | `benchmark_results/2026-06-30/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-06-30 16_58_33` | 2,332 | `benchmark_results/2026-06-30/linux/rise-riscv/results.txt` |
| macOS Apple Silicon | `local` | `2026-07-03 12_20_35` | 2,277 | `benchmark_results/2026-07-03/macos/aarch64/results.txt` |
