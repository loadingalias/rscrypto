# Benchmark Overview

Sources:

- Linux benchmark CI run [#25840411675](https://github.com/loadingalias/rscrypto/actions/runs/25840411675), created 2026-05-14 03:45:15 UTC.
- Local macOS aarch64 bench file: `benchmark_results/2026-05-13/macos/aarch64/results.txt`.
- Commit: `654f70440c6fe60ab0ac302be066a94e3d430d84`.

Scope: current benchmark results for commit `654f704`: nine Linux dedicated runners from the 2026-05-14 CI run plus one local macOS aarch64 run from 2026-05-13. Ratios are `rscrypto` divided by the external implementation for throughput rows. For latency-only rows, the ratio is external time divided by `rscrypto` time, so higher is still better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and size.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| All matched performance pairs | 10990 | 7118/2679/1193 | 65% | 1.74x | 1.17x |
| Fastest external per case | 6623 | 3666/2097/860 | 55% | 1.51x | 1.08x |
| All matched throughput pairs | 9920 | 6308/2557/1055 | 64% | 1.76x | 1.16x |
| Fastest external, throughput only | 5990 | 3228/2006/756 | 54% | 1.53x | 1.07x |
| All matched latency-only pairs | 1070 | 810/122/138 | 76% | 1.58x | 1.29x |
| Fastest external, latency only | 633 | 438/91/104 | 69% | 1.34x | 1.17x |

What matters:

- Removing the retired AES-backed fast-hash/GXHash lane leaves 5,895 Linux CI fastest-external comparisons. The Linux fastest-external geomean is now 1.54x, with 3,354 wins.
- Strict non-crypto hashes are no longer a root problem after removing the retired comparison lane: 128-bit RapidHash/XXH3 is 1.13x; 64-bit RapidHash/XXH3 is 1.12x.
- PBKDF2-SHA256 (0.70x) and HMAC-SHA256 (0.85x) are the only broad fastest-external clear losses. X25519 (0.96x) and ChaCha20-Poly1305 (0.98x) are near parity but still below the win threshold, so they need backend work before we claim wins.
- AES-GCM-SIV remains a strong area at 1.69x fastest-external geomean across all platforms; checksums and SHA-3/SHAKE remain the strongest broad wins.

## Clear Losses

Collapsed fastest-external groups. This is the useful root-cause view, not just the worst individual rows.

| Place | Fastest rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| PBKDF2-SHA256 | 60 | 15/27/18 | 0.70x | 1.00x | `rustcrypto` 48, `aws-lc-rs` 12 |
| HMAC-SHA256 | 130 | 24/57/49 | 0.85x | 0.99x | `ring` 71, `rustcrypto` 39, `aws-lc-rs` 20 |

Top measured primitive/operation losses, fastest external only, excluding tiny sample groups (`rows < 10`):

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `pbkdf2-sha256` / `iters=1000` | 20 | 4/9/7 | 0.70x | 1.00x | `rustcrypto` 16, `aws-lc-rs` 4 |
| `pbkdf2-sha256` / `iters=100` | 20 | 4/10/6 | 0.70x | 1.00x | `rustcrypto` 16, `aws-lc-rs` 4 |
| `pbkdf2-sha256` / `iters=1` | 20 | 7/8/5 | 0.71x | 1.00x | `rustcrypto` 16, `aws-lc-rs` 4 |
| `sha256` / `streaming` | 20 | 2/12/6 | 0.72x | 1.00x | `sha2` 20 |
| `hmac-sha256` / `streaming` | 20 | 2/12/6 | 0.72x | 0.99x | `rustcrypto` 20 |
| `hmac-sha256` / `hash` | 110 | 22/45/43 | 0.88x | 0.99x | `ring` 71, `aws-lc-rs` 20, `rustcrypto` 19 |
| `ed25519` / `verify` | 40 | 9/7/24 | 0.89x | 0.93x | `dalek` 18, `aws-lc-rs` 13, `ring` 9 |
| `x25519` / `diffie-hellman` | 10 | 3/5/2 | 0.93x | 1.01x | `aws-lc-rs` 8, `dryoc` 2 |
| `ed25519` / `sign` | 40 | 15/3/22 | 0.95x | 0.92x | `aws-lc-rs` 33, `dalek` 4, `dryoc` 3 |
| `hmac-sha384` / `hash` | 110 | 16/31/63 | 0.97x | 0.90x | `ring` 62, `rustcrypto` 26, `aws-lc-rs` 22 |
| `hmac-sha512` / `hash` | 110 | 16/33/61 | 0.97x | 0.91x | `ring` 63, `rustcrypto` 26, `aws-lc-rs` 21 |
| `chacha20-poly1305` / `encrypt` | 110 | 30/41/39 | 0.98x | 1.00x | `ring` 78, `aws-lc-rs` 24, `rustcrypto` 8 |

Small-sample losses are listed separately because they should not dominate release triage:

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `argon2d-small` / `m=64_t=3_p=2` | 1 | 0/0/1 | 0.21x | 0.21x | `rustcrypto` 1 |
| `argon2id-small` / `m=64_t=3_p=2` | 1 | 0/0/1 | 0.21x | 0.21x | `rustcrypto` 1 |
| `argon2i-small` / `m=64_t=3_p=2` | 1 | 0/0/1 | 0.24x | 0.24x | `rustcrypto` 1 |
| `argon2i-small` / `m=8_t=1_p=1` | 1 | 0/0/1 | 0.94x | 0.94x | `rustcrypto` 1 |
| `argon2id-small` / `m=8_t=1_p=1` | 1 | 0/1/0 | 0.96x | 0.96x | `rustcrypto` 1 |
| `argon2i-small` / `m=32_t=2_p=1` | 1 | 0/1/0 | 0.97x | 0.97x | `rustcrypto` 1 |
| `argon2d-small` / `m=8_t=1_p=1` | 1 | 0/1/0 | 0.98x | 0.98x | `rustcrypto` 1 |

## Non-Crypto Hash Truth Split

Same-algorithm rows answer whether the strict implementation is competitive with its reference crate. Fastest-class rows answer whether the strict RapidHash/XXH3 families can beat the fastest external non-crypto hash in their class.

Same-algorithm comparisons:

| Comparison | Rows | W/T/L | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `xxh3-64` vs `xxhash-rust` | 110 | 55/36/19 | 1.17x | 1.05x |
| `xxh3-128` vs `xxhash-rust` | 110 | 57/40/13 | 1.17x | 1.07x |
| `rapidhash-64` vs `rapidhash` | 110 | 38/61/11 | 1.13x | 1.01x |
| `rapidhash-128` vs `rapidhash` | 110 | 42/60/8 | 1.15x | 1.02x |
| `rapidhash-v3-64` vs `rapidhash` | 110 | 30/59/21 | 1.06x | 1.00x |
| `rapidhash-v3-128` vs `rapidhash` | 110 | 38/50/22 | 1.07x | 1.00x |

Fastest-class comparisons:

| Comparison | Rows | W/T/L | All-platform geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| 64-bit RapidHash/XXH3 fastest-class | 330 | 123/156/51 | 1.12x | 1.01x | `rapidhash` 220, `xxhash-rust` 110 |
| 128-bit RapidHash/XXH3 fastest-class | 330 | 137/150/43 | 1.13x | 1.02x | `rapidhash` 220, `xxhash-rust` 110 |

## Sustained AEAD

Fastest-external throughput comparisons only, sizes `65536`, `262144`, and `1048576`, both encrypt and decrypt, 128-bit and 256-bit keys where applicable.

| Platform | AES-GCM W/T/L | AES-GCM geomean | GCM-SIV W/T/L | GCM-SIV geomean | ChaCha20-Poly1305 W/T/L | ChaCha20-Poly1305 geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 0/12/0 | 1.00x | 12/0/0 | 1.74x | 0/0/6 | 0.91x |
| AMD Zen5 | 0/8/4 | 0.99x | 12/0/0 | 2.05x | 6/0/0 | 1.15x |
| Intel Ice Lake | 2/8/2 | 1.00x | 12/0/0 | 1.36x | 0/6/0 | 1.03x |
| Intel Sapphire Rapids | 1/8/3 | 0.97x | 12/0/0 | 1.48x | 6/0/0 | 1.06x |
| AWS Graviton3 | 0/0/12 | 0.91x | 12/0/0 | 2.54x | 0/6/0 | 1.00x |
| AWS Graviton4 | 0/8/4 | 0.97x | 12/0/0 | 2.53x | 0/6/0 | 1.00x |
| IBM Power10 | 0/0/12 | 0.37x | 6/0/6 | 1.46x | 6/0/0 | 1.15x |
| IBM z16/s390x | 12/0/0 | 12.12x | 12/0/0 | 5.56x | 6/0/0 | 2.06x |
| RISE RISC-V | 0/0/12 | 0.82x | 0/0/12 | 0.90x | 0/0/6 | 0.92x |
| macOS aarch64 | 0/12/0 | 1.00x | 12/0/0 | 2.79x | 0/6/0 | 1.00x |

Primitive-level sustained AEAD geomeans:

| Platform | AES-128-GCM | AES-256-GCM | AES-128-GCM-SIV | AES-256-GCM-SIV | ChaCha20-Poly1305 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 1.01x | 0.98x | 1.78x | 1.71x | 0.91x |
| AMD Zen5 | 1.02x | 0.95x | 2.05x | 2.05x | 1.15x |
| Intel Ice Lake | 1.01x | 0.99x | 1.34x | 1.37x | 1.03x |
| Intel Sapphire Rapids | 0.98x | 0.95x | 1.44x | 1.51x | 1.06x |
| AWS Graviton3 | 0.91x | 0.91x | 2.41x | 2.67x | 1.00x |
| AWS Graviton4 | 0.96x | 0.97x | 2.39x | 2.69x | 1.00x |
| IBM Power10 | 0.37x | 0.37x | 1.43x | 1.49x | 1.15x |
| IBM z16/s390x | 11.55x | 12.71x | 5.26x | 5.88x | 2.06x |
| RISE RISC-V | 0.81x | 0.83x | 0.90x | 0.90x | 0.92x |
| macOS aarch64 | 1.01x | 1.00x | 2.71x | 2.87x | 1.00x |

## Platform Summary

Fastest-external comparisons across every matched primitive, operation, and size.

| Platform | Rows | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 655 | 422/157/76 | 64% | 1.46x | 1.09x |
| AMD Zen5 | 655 | 335/242/78 | 51% | 1.47x | 1.05x |
| Intel Ice Lake | 655 | 413/146/96 | 63% | 1.46x | 1.14x |
| Intel Sapphire Rapids | 655 | 416/127/112 | 64% | 1.44x | 1.12x |
| AWS Graviton3 | 655 | 323/238/94 | 49% | 1.42x | 1.04x |
| AWS Graviton4 | 655 | 324/263/68 | 49% | 1.44x | 1.05x |
| IBM Power10 | 655 | 300/277/78 | 46% | 1.48x | 1.03x |
| IBM z16/s390x | 655 | 543/76/36 | 83% | 3.14x | 2.37x |
| RISE RISC-V | 655 | 278/238/139 | 42% | 1.11x | 1.02x |
| macOS aarch64 | 728 | 312/333/83 | 43% | 1.34x | 1.03x |

## Primitive Summary

All primitives with matched `rscrypto` comparisons. The fastest-only columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `argon2id-small` | 3 | 0/2/1 | 0.58x | 5 | 2/2/1 | 1.02x |
| `argon2d-small` | 3 | 0/2/1 | 0.59x | 3 | 0/2/1 | 0.59x |
| `argon2i-small` | 3 | 0/1/2 | 0.60x | 5 | 2/1/2 | 1.01x |
| `pbkdf2-sha256` | 60 | 15/27/18 | 0.70x | 180 | 123/31/26 | 0.99x |
| `hmac-sha256` | 130 | 24/57/49 | 0.85x | 350 | 138/130/82 | 1.03x |
| `x25519` | 20 | 6/10/4 | 0.96x | 60 | 46/10/4 | 1.46x |
| `hmac-sha384` | 110 | 16/31/63 | 0.97x | 330 | 115/100/115 | 1.12x |
| `hmac-sha512` | 110 | 16/33/61 | 0.97x | 330 | 111/106/113 | 1.11x |
| `chacha20-poly1305` | 220 | 68/75/77 | 0.98x | 660 | 361/173/126 | 1.19x |
| `argon2id-owasp` | 1 | 0/1/0 | 1.00x | 2 | 1/1/0 | 1.24x |
| `ed25519` | 100 | 44/10/46 | 1.03x | 340 | 238/26/76 | 1.27x |
| `blake2/blake2b256` | 268 | 105/154/9 | 1.04x | 419 | 247/162/10 | 1.32x |
| `rapidhash-v3-64` | 110 | 30/59/21 | 1.06x | 110 | 30/59/21 | 1.06x |
| `blake2/blake2b512` | 220 | 113/105/2 | 1.07x | 341 | 230/109/2 | 1.37x |
| `rapidhash-v3-128` | 110 | 38/50/22 | 1.07x | 110 | 38/50/22 | 1.07x |
| `blake2/blake2s256` | 268 | 124/144/0 | 1.08x | 268 | 124/144/0 | 1.08x |
| `sha256` | 130 | 40/65/25 | 1.10x | 350 | 188/118/44 | 1.40x |
| `scrypt-owasp` | 1 | 1/0/0 | 1.10x | 1 | 1/0/0 | 1.10x |
| `blake2/blake2s128` | 220 | 124/96/0 | 1.11x | 220 | 124/96/0 | 1.11x |
| `pbkdf2-sha512` | 60 | 21/28/11 | 1.11x | 180 | 130/34/16 | 1.27x |
| `scrypt-small` | 4 | 4/0/0 | 1.12x | 4 | 4/0/0 | 1.12x |
| `rapidhash-64` | 110 | 38/61/11 | 1.13x | 110 | 38/61/11 | 1.13x |
| `rapidhash-128` | 110 | 42/60/8 | 1.15x | 110 | 42/60/8 | 1.15x |
| `hkdf-sha256` | 40 | 29/10/1 | 1.16x | 120 | 108/11/1 | 1.83x |
| `xxh3-64` | 110 | 55/36/19 | 1.17x | 110 | 55/36/19 | 1.17x |
| `xxh3-128` | 110 | 57/40/13 | 1.17x | 110 | 57/40/13 | 1.17x |
| `hkdf-sha384` | 40 | 29/10/1 | 1.20x | 120 | 107/11/2 | 1.63x |
| `kmac256` | 11 | 10/1/0 | 1.20x | 11 | 10/1/0 | 1.20x |
| `sha512` | 130 | 43/77/10 | 1.21x | 350 | 196/138/16 | 1.30x |
| `sha384` | 110 | 39/58/13 | 1.22x | 330 | 193/118/19 | 1.32x |
| `ascon-hash256` | 110 | 60/49/1 | 1.24x | 110 | 60/49/1 | 1.24x |
| `cshake256` | 11 | 11/0/0 | 1.25x | 11 | 11/0/0 | 1.25x |
| `sha512-256` | 110 | 56/52/2 | 1.26x | 110 | 56/52/2 | 1.26x |
| `aes-256-gcm` | 220 | 111/45/64 | 1.30x | 660 | 513/51/96 | 2.70x |
| `aes-128-gcm` | 220 | 115/52/53 | 1.32x | 660 | 518/63/79 | 2.67x |
| `ascon-xof128` | 110 | 86/24/0 | 1.33x | 110 | 86/24/0 | 1.33x |
| `blake3` | 480 | 243/205/32 | 1.40x | 480 | 243/205/32 | 1.40x |
| `xchacha20-poly1305` | 220 | 191/29/0 | 1.42x | 220 | 191/29/0 | 1.42x |
| `aegis-256` | 220 | 130/69/21 | 1.42x | 220 | 130/69/21 | 1.42x |
| `crc32c` | 110 | 55/40/15 | 1.56x | 220 | 163/42/15 | 2.17x |
| `crc32` | 110 | 52/37/21 | 1.59x | 220 | 158/41/21 | 2.25x |
| `crc64-nvme` | 110 | 61/36/13 | 1.65x | 220 | 149/53/18 | 2.12x |
| `aes-128-gcm-siv` | 220 | 153/4/63 | 1.66x | 440 | 351/18/71 | 3.24x |
| `aes-256-gcm-siv` | 220 | 154/5/61 | 1.73x | 440 | 373/6/61 | 3.56x |
| `sha224` | 110 | 52/56/2 | 1.76x | 110 | 52/56/2 | 1.76x |
| `sha3-256` | 130 | 110/18/2 | 1.98x | 130 | 110/18/2 | 1.98x |
| `sha3-224` | 110 | 97/13/0 | 2.02x | 110 | 97/13/0 | 2.02x |
| `sha3-384` | 110 | 96/11/3 | 2.03x | 110 | 96/11/3 | 2.03x |
| `sha3-512` | 110 | 94/14/2 | 2.05x | 110 | 94/14/2 | 2.05x |
| `crc64-xz` | 110 | 87/15/8 | 2.28x | 110 | 87/15/8 | 2.28x |
| `shake128` | 110 | 99/5/6 | 2.35x | 110 | 99/5/6 | 2.35x |
| `shake256` | 110 | 97/13/0 | 2.38x | 110 | 97/13/0 | 2.38x |
| `crc24-openpgp` | 110 | 107/2/1 | 14.40x | 110 | 107/2/1 | 14.40x |
| `crc16-ccitt` | 110 | 109/0/1 | 24.79x | 110 | 109/0/1 | 24.79x |
| `crc16-ibm` | 110 | 109/0/1 | 25.37x | 110 | 109/0/1 | 25.37x |

## External Pressure

All-pair comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rapidhash` | 440 | 148/230/62 | 34% | 1.10x | 1.01x |
| `xxhash-rust` | 220 | 112/76/32 | 51% | 1.17x | 1.06x |
| `tiny-keccak` | 22 | 21/1/0 | 95% | 1.22x | 1.27x |
| `ascon-hash` | 220 | 146/73/1 | 66% | 1.28x | 1.11x |
| `ring` | 1600 | 1053/245/302 | 66% | 1.38x | 1.17x |
| `sha2` | 590 | 278/300/12 | 47% | 1.39x | 1.05x |
| `aws-lc-rs` | 2060 | 1211/358/491 | 59% | 1.40x | 1.13x |
| `blake3` | 480 | 243/205/32 | 51% | 1.40x | 1.05x |
| `dalek` | 120 | 101/11/8 | 84% | 1.40x | 1.38x |
| `aegis-crate` | 220 | 130/69/21 | 59% | 1.42x | 1.09x |
| `dryoc` | 377 | 348/20/9 | 92% | 1.87x | 1.74x |
| `crc-fast` | 330 | 196/108/26 | 59% | 1.91x | 1.11x |
| `rustcrypto` | 2861 | 1852/864/145 | 65% | 1.95x | 1.12x |
| `sha3` | 680 | 593/74/13 | 87% | 2.12x | 2.01x |
| `crc64fast-nvme` | 110 | 82/16/12 | 75% | 2.22x | 1.92x |
| `crc64fast` | 110 | 87/15/8 | 79% | 2.28x | 1.91x |
| `crc32fast` | 110 | 91/7/12 | 83% | 2.50x | 2.03x |
| `crc32c` | 110 | 101/5/4 | 92% | 2.76x | 2.22x |
| `crc` | 330 | 325/2/3 | 98% | 20.85x | 32.18x |

## README Numbers

- **Headline:** 3,354 of 5,895 matched Linux CI fastest-external comparisons are wins; Linux CI geomean is 1.54x.
- **Checksums:** 5.03x geomean across Linux CI fastest-external checksum comparisons.
- **SHA-3 / SHAKE:** 2.18x SHA-3 geomean and 2.60x SHAKE/cSHAKE/KMAC geomean across Linux CI fastest-external comparisons.
- **BLAKE3:** 2.23x geomean for Linux CI fastest-external rows at `>=64 KiB`.
- **AEAD:** 1.37x geomean across Linux CI fastest-external AEAD comparisons.
- **Ed25519 / X25519:** 1.01x Ed25519 sign geomean and 0.95x X25519 geomean across Linux CI fastest-external comparisons.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | ---: | --- |
| AMD Zen4 | `ci` | `2026-05-14 03_45_15` | 1772 | `benchmark_results/2026-05-14/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-05-14 03_45_15` | 1772 | `benchmark_results/2026-05-14/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-05-14 03_45_15` | 1772 | `benchmark_results/2026-05-14/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-05-14 03_45_15` | 1772 | `benchmark_results/2026-05-14/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-05-14 03_45_15` | 1772 | `benchmark_results/2026-05-14/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-05-14 03_45_15` | 1772 | `benchmark_results/2026-05-14/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-05-14 03_45_15` | 1772 | `benchmark_results/2026-05-14/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-05-14 03_45_15` | 1772 | `benchmark_results/2026-05-14/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-05-14 03_45_15` | 1772 | `benchmark_results/2026-05-14/linux/rise-riscv/results.txt` |
| macOS aarch64 | `local` | `2026-05-13 23_45_23` | 1981 | `benchmark_results/2026-05-13/macos/aarch64/results.txt` |

## Methodology Notes

- Parsed 17,929 Criterion rows: 16,214 throughput rows and 1,715 latency-only rows.
- Matched 10,990 `rscrypto` external pairs across 6,623 fastest-external cases.
- Input files include 10 result files. CI result headers use `time=03_45_15`, the GitHub Actions run creation time, per extraction policy. The local macOS file keeps its local run time.
- Matching is structural: a comparison exists when an external Criterion ID can be produced by replacing the `rscrypto` path component while keeping the same platform, primitive path, operation path, and size suffix.
- Unmatched external-only rows: 244. These are internal comparisons or benches without a corresponding `rscrypto` row, so they are not part of the ratio tables.
