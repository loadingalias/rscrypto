# Benchmark Overview

Sources:

- Linux benchmark CI run [#28710784737](https://github.com/loadingalias/rscrypto/actions/runs/28710784737), created 2026-07-04 15:26:12 UTC.
- Linux commit: `596498f0e07e869eac71fd31c157aa1b22186239`.
- Linux artifacts: nine successful `benchmark-*` artifacts extracted into `benchmark_results/2026-07-04/linux/*/results.txt`.
- Local macOS run: `benchmark_results/2026-07-04/macos/aarch64/results.txt` at commit `596498f0e07e869eac71fd31c157aa1b22186239`.

Scope: the 2026-07-04 nine-runner Linux CI benchmark matrix for commit `596498f`. Ratios are `external_crate_time / rscrypto_time`; higher is better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and input shape. Internal kernel, scratch-buffer, padding-only, cold-path, PHC roundtrip, parallel-scaling, threshold-selection, public-overhead, and phase-attribution microbenches are parsed as raw rows but excluded from external win/loss claims. The macOS local run is listed separately and is not mixed into Linux CI claims.

Coverage note: this is a full Linux CI public benchmark pass. It includes checksum, hash, XOF, MAC, KDF, password-hashing, BLAKE2/BLAKE3, RSA import/verification, ECDSA P-256/P-384 signing and verification, Ed25519, X25519, AEAD, and ML-KEM-512/768/1024 keygen, encapsulation, and decapsulation rows. ML-KEM phase/arithmetic microbenches are present in the raw artifacts and intentionally excluded from release-level competitor claims.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Linux CI: all matched performance pairs | 10,781 | 7,542/2,470/769 | 70% | 1.76x | 1.22x |
| Linux CI: fastest external per case | 6,750 | 4,052/2,049/649 | 60% | 1.59x | 1.11x |

Shareable release summary:

- **Headline:** 4,052 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,101 are wins or ties. Linux CI fastest-external geomean is 1.59x.
- **Checksums:** 5.18x geomean across 693 fastest-external rows; W/T/L is 517/115/61.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,726 fastest-external rows; W/T/L is 2,028/1,457/241.
- **Auth/KDF:** 1.25x geomean across 180 fastest-external rows; W/T/L is 159/20/1.
- **Password hashing:** 1.07x geomean across 135 fastest-external rows; W/T/L is 69/28/38.
- **Public-key:** 1.33x geomean across 333 fastest-external rows; W/T/L is 217/73/43.
- **RSA:** 1.55x geomean across 99 fastest-external rows; W/T/L is 89/2/8.
- **AEAD:** 1.56x geomean across 1,584 fastest-external rows; W/T/L is 973/354/257.
- **ML-KEM:** 1.49x geomean across 81 fastest-external rows; W/T/L is 68/4/9.
- **ECDSA P-256/P-384:** Linux CI 1.45x geomean across 144 fastest-external rows; W/T/L is 116/7/21.
- **Top current loss areas:** `argon2id-owasp` / `hash`: 0.97x geomean across 9 rows; W/T/L 4/1/4; pressure `rustcrypto` 4; `ed25519` / `verify`: 1.00x geomean across 36 rows; W/T/L 7/20/9; pressure `ring` 6, `dalek` 3; `mlkem1024` / `keygen`: 1.00x geomean across 9 rows; W/T/L 6/0/3; pressure `aws-lc-rs` 3; `blake2b256` / `streaming`: 1.02x geomean across 27 rows; W/T/L 10/16/1; pressure `rustcrypto` 1; `argon2id-small` / `hash`: 1.02x geomean across 27 rows; W/T/L 12/4/11; pressure `rustcrypto` 11.

## Coverage Matrix

| Platform | Raw Criterion rows | All pairs | Fastest rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 2,356 | 1,251 | 750 | 513/189/48 | 68% | 1.50x | 1.15x |
| AMD Zen5 | 2,356 | 1,251 | 750 | 434/249/67 | 58% | 1.49x | 1.09x |
| AWS Graviton3 | 2,367 | 1,251 | 750 | 360/298/92 | 48% | 1.37x | 1.04x |
| AWS Graviton4 | 2,367 | 1,251 | 750 | 362/332/56 | 48% | 1.37x | 1.05x |
| IBM Power10 | 2,107 | 1,012 | 750 | 389/324/37 | 52% | 1.89x | 1.06x |
| IBM z16/s390x | 2,107 | 1,012 | 750 | 603/87/60 | 80% | 3.15x | 2.34x |
| Intel Ice Lake | 2,356 | 1,251 | 750 | 523/154/73 | 70% | 1.49x | 1.19x |
| Intel Sapphire Rapids | 2,356 | 1,251 | 750 | 533/149/68 | 71% | 1.63x | 1.20x |
| RISE RISC-V | 2,356 | 1,251 | 750 | 335/267/148 | 45% | 1.09x | 1.03x |

## Category Summary

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 693 | 517/115/61 | 75% | 5.18x | 2.44x |
| Hashes/MACs/XOFs | 3,726 | 2,028/1,457/241 | 54% | 1.35x | 1.07x |
| Auth/KDF | 180 | 159/20/1 | 88% | 1.25x | 1.13x |
| Password hashing | 135 | 69/28/38 | 51% | 1.07x | 1.05x |
| Public-key | 333 | 217/73/43 | 65% | 1.33x | 1.17x |
| RSA | 99 | 89/2/8 | 90% | 1.55x | 1.18x |
| AEAD | 1,584 | 973/354/257 | 61% | 1.56x | 1.15x |

## BLAKE3 Summary

BLAKE3 rows come from Linux CI run [#28710784737](https://github.com/loadingalias/rscrypto/actions/runs/28710784737). All-pair and fastest-external BLAKE3 metrics are identical because official `blake3` is the only external implementation in this bench.

| Scope | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| All Linux BLAKE3 rows | 432 | 234/168/30 | 1.41x | 1.08x |
| x86_64 | 192 | 92/90/10 | 1.26x | 1.05x |
| AArch64 | 96 | 44/46/6 | 1.44x | 1.05x |

| Platform | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| AMD Zen4 | 48 | 23/23/2 | 1.34x | 1.05x |
| AMD Zen5 | 48 | 21/25/2 | 1.32x | 1.03x |
| AWS Graviton3 | 48 | 22/21/5 | 1.40x | 0.99x |
| AWS Graviton4 | 48 | 22/25/1 | 1.47x | 1.05x |
| IBM Power10 | 48 | 39/9/0 | 1.98x | 1.85x |
| IBM z16/s390x | 48 | 36/5/7 | 1.81x | 2.10x |
| Intel Ice Lake | 48 | 19/26/3 | 1.17x | 1.02x |
| Intel Sapphire Rapids | 48 | 29/16/3 | 1.24x | 1.08x |
| RISE RISC-V | 48 | 23/18/7 | 1.15x | 1.02x |

| Operation | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `oneshot` | 99 | 45/47/7 | 1.32x | 1.02x |
| `keyed` | 99 | 46/45/8 | 1.34x | 1.03x |
| `derive-key` | 99 | 77/20/2 | 1.76x | 1.86x |
| `streaming` | 36 | 11/21/4 | 1.16x | 1.01x |
| `xof` | 99 | 55/35/9 | 1.35x | 1.07x |

## ML-KEM Summary

ML-KEM public coverage is complete for the CI-selected primitive set: ML-KEM-512, ML-KEM-768, and ML-KEM-1024 each include keygen, encapsulate, and decapsulate on all nine Linux platforms. POWER10 and s390x do not have `aws-lc-rs` ML-KEM rows in this artifact set, but still have rscrypto plus `libcrux`, `fips203`, and RustCrypto comparison rows for every public operation.

| Platform | Raw ML-KEM rows | Fastest rows | W/T/L | Geomean | Median | Fastest external split |
| --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 45 | 9 | 9/0/0 | 1.81x | 1.70x | `libcrux` 7, `aws-lc-rs` 2 |
| AMD Zen5 | 45 | 9 | 9/0/0 | 1.93x | 1.75x | `libcrux` 9 |
| AWS Graviton3 | 45 | 9 | 5/0/4 | 1.08x | 1.13x | `aws-lc-rs` 9 |
| AWS Graviton4 | 45 | 9 | 5/0/4 | 1.08x | 1.14x | `aws-lc-rs` 9 |
| IBM Power10 | 36 | 9 | 9/0/0 | 1.47x | 1.58x | `libcrux` 9 |
| IBM z16/s390x | 36 | 9 | 9/0/0 | 1.58x | 1.64x | `libcrux` 9 |
| Intel Ice Lake | 45 | 9 | 9/0/0 | 1.79x | 1.74x | `libcrux` 7, `aws-lc-rs` 2 |
| Intel Sapphire Rapids | 45 | 9 | 9/0/0 | 1.85x | 1.82x | `aws-lc-rs` 5, `libcrux` 4 |
| RISE RISC-V | 45 | 9 | 4/4/1 | 1.13x | 1.03x | `aws-lc-rs` 9 |

| Primitive/op | Rows | W/T/L | Win % | Geomean | Median | Pressure |
| --- | --- | --- | --- | --- | --- | --- |
| `mlkem1024` / `decapsulate` | 9 | 9/0/0 | 100% | 1.59x | 1.67x | none |
| `mlkem1024` / `encapsulate` | 9 | 9/0/0 | 100% | 2.39x | 1.98x | none |
| `mlkem1024` / `keygen` | 9 | 6/0/3 | 67% | 1.00x | 1.09x | `aws-lc-rs` 3 |
| `mlkem512` / `decapsulate` | 9 | 6/1/2 | 67% | 1.33x | 1.51x | `aws-lc-rs` 2 |
| `mlkem512` / `encapsulate` | 9 | 9/0/0 | 100% | 1.86x | 1.70x | none |
| `mlkem512` / `keygen` | 9 | 6/1/2 | 67% | 1.08x | 1.18x | `aws-lc-rs` 2 |
| `mlkem768` / `decapsulate` | 9 | 8/1/0 | 89% | 1.49x | 1.70x | none |
| `mlkem768` / `encapsulate` | 9 | 9/0/0 | 100% | 2.27x | 2.02x | none |
| `mlkem768` / `keygen` | 9 | 6/1/2 | 67% | 1.04x | 1.12x | `aws-lc-rs` 2 |

## ECDSA Summary

ECDSA signing includes both deterministic and blinded rscrypto rows in raw results; aggregate fastest-external comparisons use the fastest rscrypto row for the exact case. Constant-time release evidence is tracked separately by `ct.toml` and CT workflow artifacts.

| Operation | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `ecdsa-p256` / `sign` | 36 | 36/0/0 | 1.45x | 1.35x |
| `ecdsa-p256` / `verify` | 36 | 28/7/1 | 1.59x | 1.15x |
| `ecdsa-p384` / `sign` | 36 | 16/0/20 | 1.07x | 0.90x |
| `ecdsa-p384` / `verify` | 36 | 36/0/0 | 1.81x | 1.39x |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are strongest-external comparisons; all-pair columns include every matched external implementation.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `argon2id-owasp` | 9 | 4/1/4 | 0.97x | 18 | 11/2/5 | 1.27x |
| `argon2id-small` | 27 | 12/4/11 | 1.02x | 45 | 29/5/11 | 1.35x |
| `argon2d-small` | 27 | 13/6/8 | 1.03x | 27 | 13/6/8 | 1.03x |
| `x25519` | 18 | 2/16/0 | 1.03x | 50 | 33/17/0 | 1.54x |
| `argon2i-small` | 27 | 13/2/12 | 1.03x | 45 | 30/2/13 | 1.35x |
| `rapidhash-v3-64` | 99 | 27/53/19 | 1.05x | 99 | 27/53/19 | 1.05x |
| `blake2b256` | 225 | 120/100/5 | 1.07x | 351 | 238/108/5 | 1.31x |
| `rapidhash-v3-128` | 99 | 34/44/21 | 1.07x | 99 | 34/44/21 | 1.07x |
| `blake2b512` | 198 | 109/82/7 | 1.07x | 297 | 207/83/7 | 1.31x |
| `xxh3-128` | 99 | 29/51/19 | 1.08x | 99 | 29/51/19 | 1.08x |
| `scrypt-owasp` | 9 | 5/2/2 | 1.09x | 9 | 5/2/2 | 1.09x |
| `xxh3-64` | 99 | 33/47/19 | 1.10x | 99 | 33/47/19 | 1.10x |
| `rapidhash-64` | 99 | 32/51/16 | 1.10x | 99 | 32/51/16 | 1.10x |
| `chacha20-poly1305` | 198 | 83/108/7 | 1.10x | 550 | 344/198/8 | 1.29x |
| `blake2s256` | 225 | 117/107/1 | 1.10x | 225 | 117/107/1 | 1.10x |
| `ed25519` | 90 | 31/46/13 | 1.11x | 290 | 204/69/17 | 1.36x |
| `blake2s128` | 198 | 118/79/1 | 1.11x | 198 | 118/79/1 | 1.11x |
| `rapidhash-128` | 99 | 38/56/5 | 1.14x | 99 | 38/56/5 | 1.14x |
| `rsa-8192` | 18 | 14/2/2 | 1.19x | 32 | 28/2/2 | 1.25x |
| `scrypt-small` | 36 | 22/13/1 | 1.21x | 36 | 22/13/1 | 1.21x |
| `hmac-sha512` | 99 | 37/54/8 | 1.23x | 275 | 165/97/13 | 1.29x |
| `hmac-sha384` | 99 | 38/52/9 | 1.24x | 275 | 167/94/14 | 1.29x |
| `hkdf-sha384` | 36 | 33/2/1 | 1.24x | 100 | 96/3/1 | 1.53x |
| `sha256` | 117 | 41/59/17 | 1.24x | 293 | 164/99/30 | 1.54x |
| `sha512` | 117 | 45/61/11 | 1.25x | 293 | 172/104/17 | 1.27x |
| `pbkdf2-sha512` | 54 | 47/7/0 | 1.25x | 150 | 143/7/0 | 1.33x |
| `sha384` | 99 | 43/45/11 | 1.25x | 275 | 169/89/17 | 1.28x |
| `pbkdf2-sha256` | 54 | 50/4/0 | 1.25x | 150 | 146/4/0 | 1.65x |
| `hkdf-sha256` | 36 | 29/7/0 | 1.26x | 100 | 93/7/0 | 1.87x |
| `ascon-hash256` | 99 | 54/43/2 | 1.27x | 99 | 54/43/2 | 1.27x |
| `sha512-256` | 99 | 53/46/0 | 1.29x | 99 | 53/46/0 | 1.29x |
| `ascon-aead128` | 198 | 148/50/0 | 1.33x | 198 | 148/50/0 | 1.33x |
| `hmac-sha256` | 117 | 64/39/14 | 1.34x | 293 | 193/76/24 | 1.68x |
| `aegis-256` | 198 | 89/88/21 | 1.37x | 198 | 89/88/21 | 1.37x |
| `ascon-xof128` | 99 | 75/24/0 | 1.37x | 99 | 75/24/0 | 1.37x |
| `mlkem512` | 27 | 21/2/4 | 1.39x | 102 | 96/2/4 | 2.67x |
| `xchacha20-poly1305` | 198 | 182/16/0 | 1.39x | 198 | 182/16/0 | 1.39x |
| `ecdsa-p384` | 72 | 52/0/20 | 1.39x | 200 | 180/0/20 | 3.17x |
| `blake3` | 432 | 234/168/30 | 1.41x | 432 | 234/168/30 | 1.41x |
| `ecdsa-p256` | 72 | 64/7/1 | 1.52x | 200 | 188/11/1 | 2.40x |
| `mlkem768` | 27 | 23/2/2 | 1.52x | 102 | 98/2/2 | 3.11x |
| `mlkem1024` | 27 | 24/0/3 | 1.56x | 102 | 99/0/3 | 3.27x |
| `rsa-4096` | 27 | 25/0/2 | 1.56x | 59 | 57/0/2 | 2.52x |
| `crc32c` | 99 | 45/37/17 | 1.61x | 198 | 140/40/18 | 2.19x |
| `crc32` | 99 | 49/31/19 | 1.66x | 198 | 144/35/19 | 2.26x |
| `rsa-3072` | 27 | 25/0/2 | 1.66x | 59 | 57/0/2 | 2.59x |
| `aes-128-gcm` | 198 | 97/45/56 | 1.68x | 550 | 413/54/83 | 1.89x |
| `rsa-2048` | 27 | 25/0/2 | 1.71x | 59 | 57/0/2 | 2.63x |
| `aes-256-gcm` | 198 | 97/40/61 | 1.71x | 550 | 405/52/93 | 1.92x |
| `kmac256` | 99 | 65/18/16 | 1.75x | 99 | 65/18/16 | 1.75x |
| `cshake256` | 99 | 63/28/8 | 1.78x | 99 | 63/28/8 | 1.78x |
| `shake128` | 99 | 69/29/1 | 1.85x | 99 | 69/29/1 | 1.85x |
| `shake256` | 99 | 70/29/0 | 1.87x | 99 | 70/29/0 | 1.87x |
| `sha224` | 99 | 52/46/1 | 1.87x | 99 | 52/46/1 | 1.87x |
| `aes-128-gcm-siv` | 198 | 138/3/57 | 2.02x | 352 | 271/16/65 | 2.79x |
| `sha3-256` | 117 | 104/13/0 | 2.11x | 117 | 104/13/0 | 2.11x |
| `sha3-224` | 99 | 87/12/0 | 2.13x | 99 | 87/12/0 | 2.13x |
| `crc64-nvme` | 99 | 56/33/10 | 2.14x | 99 | 56/33/10 | 2.14x |
| `aes-256-gcm-siv` | 198 | 139/4/55 | 2.15x | 352 | 292/5/55 | 3.04x |
| `sha3-384` | 99 | 89/10/0 | 2.16x | 99 | 89/10/0 | 2.16x |
| `sha3-512` | 99 | 88/11/0 | 2.20x | 99 | 88/11/0 | 2.20x |
| `crc64-xz` | 99 | 76/12/11 | 2.39x | 99 | 76/12/11 | 2.39x |
| `crc24-openpgp` | 99 | 95/2/2 | 13.25x | 99 | 95/2/2 | 13.25x |
| `crc16-ccitt` | 99 | 98/0/1 | 23.12x | 99 | 98/0/1 | 23.12x |
| `crc16-ibm` | 99 | 98/0/1 | 24.01x | 99 | 98/0/1 | 24.01x |

## Linux Worst Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| RISE RISC-V | `xxh3-64 / 0` | `xxhash-rust` | 0.43x |
| Intel Sapphire Rapids | `aes-256-gcm / encrypt / 32` | `rustcrypto` | 0.46x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.48x |
| Intel Sapphire Rapids | `aes-128-gcm / encrypt / 32` | `rustcrypto` | 0.48x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.48x |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `aes-128-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.49x |
| Intel Ice Lake | `aes-128-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `aes-128-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| AMD Zen4 | `aes-256-gcm / encrypt / 32` | `rustcrypto` | 0.49x |
| Intel Ice Lake | `aes-256-gcm / encrypt / 32` | `rustcrypto` | 0.49x |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.49x |

## Linux Strongest Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt / 262144` | `crc` | 214.89x |
| Intel Sapphire Rapids | `crc16-ibm / 262144` | `crc` | 209.87x |
| Intel Sapphire Rapids | `crc16-ibm / 16384` | `crc` | 208.50x |
| Intel Sapphire Rapids | `crc16-ccitt / 16384` | `crc` | 207.55x |
| Intel Sapphire Rapids | `crc16-ccitt / 4096` | `crc` | 184.38x |
| Intel Sapphire Rapids | `crc16-ccitt / 65536` | `crc` | 183.70x |
| Intel Sapphire Rapids | `crc16-ibm / 65536` | `crc` | 179.68x |
| Intel Sapphire Rapids | `crc16-ccitt / 1048576` | `crc` | 178.10x |
| IBM Power10 | `crc16-ibm / 262144` | `crc` | 176.65x |
| IBM Power10 | `crc16-ibm / 1048576` | `crc` | 176.64x |
| IBM Power10 | `crc16-ccitt / 262144` | `crc` | 176.33x |
| IBM Power10 | `crc16-ccitt / 1048576` | `crc` | 176.09x |

## Top Five Loss Areas

- `argon2id-owasp` / `hash`: 0.97x geomean across 9 rows; W/T/L 4/1/4; pressure `rustcrypto` 4.
- `ed25519` / `verify`: 1.00x geomean across 36 rows; W/T/L 7/20/9; pressure `ring` 6, `dalek` 3.
- `mlkem1024` / `keygen`: 1.00x geomean across 9 rows; W/T/L 6/0/3; pressure `aws-lc-rs` 3.
- `blake2b256` / `streaming`: 1.02x geomean across 27 rows; W/T/L 10/16/1; pressure `rustcrypto` 1.
- `argon2id-small` / `hash`: 1.02x geomean across 27 rows; W/T/L 12/4/11; pressure `rustcrypto` 11.

## External Pressure

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `xxhash-rust` | 198 | 62/98/38 | 31% | 1.09x | 1.00x |
| `rapidhash` | 396 | 131/204/61 | 33% | 1.09x | 1.01x |
| `aws-lc-rs` | 1,673 | 1,044/384/245 | 62% | 1.25x | 1.15x |
| `ascon-hash` | 198 | 129/67/2 | 65% | 1.32x | 1.19x |
| `ascon-aead` | 198 | 148/50/0 | 75% | 1.33x | 1.15x |
| `aegis-crate` | 198 | 89/88/21 | 45% | 1.37x | 1.04x |
| `blake3` | 432 | 234/168/30 | 54% | 1.41x | 1.08x |
| `dalek` | 108 | 85/18/5 | 79% | 1.47x | 1.27x |
| `sha2` | 531 | 284/243/4 | 53% | 1.52x | 1.06x |
| `ring` | 1,656 | 1,333/228/95 | 80% | 1.66x | 1.29x |
| `rustcrypto` | 2,745 | 1,949/621/175 | 71% | 1.78x | 1.19x |
| `libcrux` | 81 | 81/0/0 | 100% | 1.79x | 1.70x |
| `dryoc` | 360 | 325/29/6 | 90% | 1.79x | 1.83x |
| `tiny-keccak` | 396 | 267/104/25 | 67% | 1.81x | 1.96x |
| `crc-fast` | 297 | 170/95/32 | 57% | 2.00x | 1.15x |
| `sha3` | 414 | 368/46/0 | 89% | 2.15x | 2.12x |
| `crc64fast` | 99 | 76/12/11 | 77% | 2.39x | 1.95x |
| `crc32fast` | 99 | 81/7/11 | 82% | 2.41x | 2.01x |
| `crc32c` | 99 | 89/6/4 | 90% | 2.71x | 2.19x |
| `fips203` | 81 | 81/0/0 | 100% | 4.66x | 5.11x |
| `rustcrypto-p384` | 72 | 72/0/0 | 100% | 4.79x | 5.26x |
| `rustcrypto-p256` | 72 | 72/0/0 | 100% | 5.65x | 4.91x |
| `rustcrypto-rsa` | 81 | 81/0/0 | 100% | 5.68x | 4.70x |
| `crc` | 297 | 291/2/4 | 98% | 19.45x | 28.96x |

## macOS Local Snapshot

The macOS Apple Silicon run is local evidence from the 2026-07-04 full benchmark at commit `596498f`. It is useful for Apple Silicon planning but is not folded into Linux CI release claims. The ML-KEM row uses the same artifact's public ML-KEM rows.

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| macOS local: all matched performance pairs | 1,297 | 815/404/78 | 63% | 1.66x | 1.16x |
| macOS local: fastest external per case | 774 | 382/326/66 | 49% | 1.37x | 1.05x |
| macOS local: ML-KEM fastest external | 9 | 6/1/2 | 67% | 1.35x | 1.39x |

## README Numbers

- **Headline:** 4,052 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,101 are wins or ties. Linux CI geomean is 1.59x.
- **Checksums:** 5.18x geomean across 693 Linux CI fastest-external rows; W/T/L 517/115/61.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,726 Linux CI fastest-external rows; W/T/L 2,028/1,457/241.
- **Auth/KDF:** 1.25x geomean across 180 Linux CI fastest-external rows; W/T/L 159/20/1.
- **Password hashing:** 1.07x geomean across 135 Linux CI fastest-external rows; W/T/L 69/28/38.
- **Public-key:** 1.33x geomean across 333 Linux CI fastest-external rows; W/T/L 217/73/43.
- **RSA:** 1.55x geomean across 99 Linux CI fastest-external rows; W/T/L 89/2/8.
- **AEAD:** 1.56x geomean across 1,584 Linux CI fastest-external rows; W/T/L 973/354/257.
- **ML-KEM:** 1.49x geomean across 81 Linux CI fastest-external rows; W/T/L 68/4/9.
- **ECDSA P-256/P-384:** 1.45x Linux CI geomean across 144 fastest-external rows; W/T/L 116/7/21.
- **Current top losses:** `argon2id-owasp` / `hash`: 0.97x geomean across 9 rows; W/T/L 4/1/4; pressure `rustcrypto` 4; `ed25519` / `verify`: 1.00x geomean across 36 rows; W/T/L 7/20/9; pressure `ring` 6, `dalek` 3; `mlkem1024` / `keygen`: 1.00x geomean across 9 rows; W/T/L 6/0/3; pressure `aws-lc-rs` 3; `blake2b256` / `streaming`: 1.02x geomean across 27 rows; W/T/L 10/16/1; pressure `rustcrypto` 1; `argon2id-small` / `hash`: 1.02x geomean across 27 rows; W/T/L 12/4/11; pressure `rustcrypto` 11.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-07-04 15_26_12` | 2,356 | `benchmark_results/2026-07-04/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-07-04 15_26_12` | 2,356 | `benchmark_results/2026-07-04/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-07-04 15_26_12` | 2,367 | `benchmark_results/2026-07-04/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-07-04 15_26_12` | 2,367 | `benchmark_results/2026-07-04/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-07-04 15_26_12` | 2,107 | `benchmark_results/2026-07-04/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-07-04 15_26_12` | 2,107 | `benchmark_results/2026-07-04/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-07-04 15_26_12` | 2,356 | `benchmark_results/2026-07-04/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-07-04 15_26_12` | 2,356 | `benchmark_results/2026-07-04/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-07-04 15_26_12` | 2,356 | `benchmark_results/2026-07-04/linux/rise-riscv/results.txt` |
| macOS Apple Silicon | `local` | `2026-07-04 12_28_04` | 2,277 | `benchmark_results/2026-07-04/macos/aarch64/results.txt` |
