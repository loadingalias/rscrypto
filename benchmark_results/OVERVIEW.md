# Benchmark Overview

Sources:

- Linux benchmark CI run [#25898560527](https://github.com/loadingalias/rscrypto/actions/runs/25898560527), created 2026-05-15 03:26:57 UTC.
- macOS Apple Silicon local run: `benchmark_results/2026-05-15/macos/aarch64/results.txt`, created 2026-05-15 21_06_15 local time.
- Commit: `e29c83747b41a829d9ad447b1d36641ae35c9cb7`.

Scope: current benchmark results for commit `e29c837`. The primary release scorecard is the nine-runner Linux CI matrix because it is one reproducible CI run. The full corpus additionally includes one local macOS Apple Silicon run from the same commit; it is useful host evidence, not a CI-wide claim. Ratios are `rscrypto` throughput divided by external throughput for throughput rows. For latency-only rows, ratios are external time divided by `rscrypto` time. Higher is always better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and size.

Important coverage note: the 2026-05-15 Linux CI plan does not contain `password_hashing` or `kmac_cshake` bench sections. Password-hashing and KMAC/cSHAKE observations below come only from the local macOS `bench=all` run.

## Headline

Linux CI scorecard:

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| All matched performance pairs | 9,801 | 6,501/2,247/1,053 | 66% | 1.77x | 1.19x |
| Fastest external per case | 5,895 | 3,369/1,760/766 | 57% | 1.55x | 1.09x |
| All matched throughput pairs | 8,865 | 5,784/2,144/937 | 65% | 1.79x | 1.17x |
| Fastest external, throughput only | 5,346 | 2,986/1,682/678 | 56% | 1.57x | 1.08x |
| All matched latency-only pairs | 936 | 717/103/116 | 77% | 1.60x | 1.31x |
| Fastest external, latency only | 549 | 383/78/88 | 70% | 1.38x | 1.19x |

Full current corpus, including local macOS Apple Silicon:

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| All matched performance pairs | 10,990 | 7,158/2,667/1,165 | 65% | 1.76x | 1.17x |
| Fastest external per case | 6,623 | 3,686/2,089/848 | 56% | 1.53x | 1.08x |
| All matched throughput pairs | 9,920 | 6,349/2,538/1,033 | 64% | 1.78x | 1.16x |
| Fastest external, throughput only | 5,990 | 3,253/1,990/747 | 54% | 1.55x | 1.07x |
| All matched latency-only pairs | 1,070 | 809/129/132 | 76% | 1.59x | 1.29x |
| Fastest external, latency only | 633 | 433/99/101 | 68% | 1.35x | 1.17x |

What matters:

- Linux CI fastest-external scorecard: 3,369 wins / 1,760 ties / 766 losses across 5,895 comparisons, with a 1.55x geomean.
- Adding the local Apple Silicon run gives 3,686 wins / 2,089 ties / 848 losses across 6,623 fastest-external comparisons, with a 1.53x geomean.
- Checksums remain the cleanest broad win: 5.07x fastest-external Linux CI geomean.
- SHA-3 and SHAKE remain strong: 2.18x and 2.59x Linux CI geomeans respectively.
- BLAKE3 sustained rows (`>=64 KiB`) improved to 2.39x Linux CI geomean against the `blake3` crate.
- AEAD is a split story: GCM-SIV, XChaCha20-Poly1305, AEGIS-256, and IBM Z are strong; AES-GCM on POWER10 and RISC-V is not.
- The biggest broad Linux CI losses are SHA-256 streaming, PBKDF2-SHA256 at `iters=1`, Ed25519 verification, and one-shot HMAC-SHA384/SHA512.
- The local Apple Silicon run exposes additional password-hashing losses: Argon2 small lanes sit at 0.60x..0.65x geomean, but only on one local host with three rows per variant.

## Clear Losses

Collapsed fastest-external Linux CI groups. This is the root-cause view; it does not include local-only password hashing.

| Place | Fastest rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `x25519` | 18 | 6/8/4 | 0.95x | 1.01x | `aws-lc-rs` 14, `dryoc` 3, `dalek` 1 |
| `pbkdf2-sha256` | 54 | 23/17/14 | 0.96x | 1.03x | `rustcrypto` 42, `aws-lc-rs` 12 |
| `chacha20-poly1305` | 198 | 67/58/73 | 0.98x | 1.00x | `ring` 101, `aws-lc-rs` 83, `rustcrypto` 14 |
| `hmac-sha512` | 99 | 16/27/56 | 0.99x | 0.92x | `ring` 58, `rustcrypto` 22, `aws-lc-rs` 19 |
| `hmac-sha384` | 99 | 17/25/57 | 0.99x | 0.91x | `ring` 60, `rustcrypto` 22, `aws-lc-rs` 17 |

Top measured Linux CI primitive/operation losses, fastest external only, excluding tiny sample groups (`rows < 10`):

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `sha256` / `streaming` | 18 | 2/10/6 | 0.68x | 0.99x | `sha2` 18 |
| `pbkdf2-sha256` / `iters=1` | 18 | 9/5/4 | 0.77x | 1.04x | `rustcrypto` 14, `aws-lc-rs` 4 |
| `ed25519` / `verify` | 36 | 8/11/17 | 0.93x | 0.96x | `dalek` 16, `aws-lc-rs` 9, `ring` 9 |
| `chacha20-poly1305` / `encrypt` | 99 | 29/34/36 | 0.97x | 1.00x | `ring` 67, `aws-lc-rs` 24, `rustcrypto` 8 |
| `hmac-sha512` / `hash` | 99 | 16/27/56 | 0.99x | 0.92x | `ring` 58, `rustcrypto` 22, `aws-lc-rs` 19 |
| `hmac-sha384` / `hash` | 99 | 17/25/57 | 0.99x | 0.91x | `ring` 60, `rustcrypto` 22, `aws-lc-rs` 17 |
| `chacha20-poly1305` / `decrypt` | 99 | 38/24/37 | 0.99x | 1.00x | `aws-lc-rs` 59, `ring` 34, `rustcrypto` 6 |

Small-sample Linux CI losses are listed separately because they should not dominate release triage:

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `x25519` / `diffie-hellman` | 9 | 3/4/2 | 0.92x | 1.01x | `aws-lc-rs` 7, `dryoc` 2 |
| `x25519` / `public-key-from-secret` | 9 | 3/4/2 | 0.98x | 1.01x | `aws-lc-rs` 7, `dryoc` 1, `dalek` 1 |

Local-only Apple Silicon losses from `bench=all`:

| Primitive | Rows | W/T/L | Geomean | Median | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `argon2id-small` | 3 | 0/2/1 | 0.60x | 0.97x | local-only password-hashing row set |
| `argon2i-small` | 3 | 0/1/2 | 0.63x | 0.89x | local-only password-hashing row set |
| `argon2d-small` | 3 | 0/2/1 | 0.65x | 0.98x | local-only password-hashing row set |
| `ed25519` | 10 | 2/0/8 | 0.76x | 0.66x | Apple Silicon local host, not CI |
| `hmac-sha512` | 11 | 0/4/7 | 0.82x | 0.87x | Apple Silicon local host, not CI |
| `hmac-sha384` | 11 | 0/5/6 | 0.82x | 0.84x | Apple Silicon local host, not CI |

## Non-Crypto Hash Truth Split

Same-algorithm rows answer whether the strict implementation is competitive with its reference crate. Fastest-class rows answer whether the strict RapidHash/XXH3 families can beat the fastest external non-crypto hash in their class.

Same-algorithm Linux CI comparisons:

| Comparison | Rows | W/T/L | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `xxh3-64` vs `xxhash-rust` | 99 | 51/35/13 | 1.19x | 1.07x |
| `xxh3-128` vs `xxhash-rust` | 99 | 51/34/14 | 1.17x | 1.09x |
| `rapidhash-64` vs `rapidhash` | 99 | 33/56/10 | 1.12x | 1.01x |
| `rapidhash-128` vs `rapidhash` | 99 | 37/55/7 | 1.14x | 1.02x |
| `rapidhash-v3-64` vs `rapidhash` | 99 | 27/52/20 | 1.05x | 1.00x |
| `rapidhash-v3-128` vs `rapidhash` | 99 | 34/43/22 | 1.08x | 1.00x |

Fastest-class Linux CI comparisons:

| Comparison | Rows | W/T/L | All-platform geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| 64-bit RapidHash/XXH3 fastest-class | 297 | 111/143/43 | 1.12x | 1.02x | `rapidhash` 198, `xxhash-rust` 99 |
| 128-bit RapidHash/XXH3 fastest-class | 297 | 122/132/43 | 1.13x | 1.02x | `rapidhash` 198, `xxhash-rust` 99 |

## Sustained AEAD

Fastest-external throughput comparisons only, sizes `65536`, `262144`, and `1048576`, both encrypt and decrypt, 128-bit and 256-bit keys where applicable.

| Platform | AES-GCM W/T/L | AES-GCM geomean | GCM-SIV W/T/L | GCM-SIV geomean | ChaCha20-Poly1305 W/T/L | ChaCha20-Poly1305 geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 0/12/0 | 1.00x | 12/0/0 | 1.75x | 0/0/6 | 0.91x |
| AMD Zen5 | 2/5/5 | 0.97x | 12/0/0 | 2.05x | 6/0/0 | 1.15x |
| AWS Graviton3 | 0/0/12 | 0.91x | 12/0/0 | 2.53x | 0/6/0 | 1.00x |
| AWS Graviton4 | 0/9/3 | 0.96x | 12/0/0 | 2.54x | 0/6/0 | 1.00x |
| IBM Power10 | 0/0/12 | 0.37x | 6/0/6 | 1.46x | 6/0/0 | 1.14x |
| IBM z16/s390x | 12/0/0 | 12.53x | 12/0/0 | 5.12x | 6/0/0 | 2.11x |
| Intel Ice Lake | 2/9/1 | 1.00x | 12/0/0 | 1.35x | 0/6/0 | 1.03x |
| Intel Sapphire Rapids | 2/5/5 | 0.98x | 12/0/0 | 1.50x | 4/2/0 | 1.05x |
| RISE RISC-V | 0/0/12 | 0.82x | 0/0/12 | 0.91x | 0/0/6 | 0.93x |
| macOS Apple Silicon | 0/12/0 | 1.01x | 12/0/0 | 2.77x | 0/6/0 | 1.00x |

Primitive-level sustained AEAD geomeans:

| Platform | AES-128-GCM | AES-256-GCM | AES-128-GCM-SIV | AES-256-GCM-SIV | ChaCha20-Poly1305 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 1.00x | 0.99x | 1.78x | 1.72x | 0.91x |
| AMD Zen5 | 1.03x | 0.91x | 2.05x | 2.05x | 1.15x |
| AWS Graviton3 | 0.90x | 0.91x | 2.40x | 2.67x | 1.00x |
| AWS Graviton4 | 0.96x | 0.96x | 2.39x | 2.70x | 1.00x |
| IBM Power10 | 0.36x | 0.37x | 1.43x | 1.49x | 1.14x |
| IBM z16/s390x | 11.91x | 13.17x | 4.48x | 5.84x | 2.11x |
| Intel Ice Lake | 1.02x | 0.98x | 1.33x | 1.38x | 1.03x |
| Intel Sapphire Rapids | 0.96x | 0.99x | 1.48x | 1.53x | 1.05x |
| RISE RISC-V | 0.81x | 0.83x | 0.92x | 0.91x | 0.93x |
| macOS Apple Silicon | 1.01x | 1.01x | 2.72x | 2.82x | 1.00x |

## Platform Summary

Fastest-external comparisons across every matched primitive, operation, and size. The macOS row is local; every other row is Linux CI.

| Platform | Rows | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 655 | 423/156/76 | 65% | 1.46x | 1.09x |
| AMD Zen5 | 655 | 338/233/84 | 52% | 1.47x | 1.05x |
| AWS Graviton3 | 655 | 321/235/99 | 49% | 1.42x | 1.04x |
| AWS Graviton4 | 655 | 323/263/69 | 49% | 1.44x | 1.05x |
| IBM Power10 | 655 | 296/280/79 | 45% | 1.48x | 1.03x |
| IBM z16/s390x | 655 | 543/78/34 | 83% | 3.13x | 2.36x |
| Intel Ice Lake | 655 | 422/140/93 | 64% | 1.46x | 1.14x |
| Intel Sapphire Rapids | 655 | 425/140/90 | 65% | 1.58x | 1.13x |
| RISE RISC-V | 655 | 278/235/142 | 42% | 1.11x | 1.02x |
| macOS Apple Silicon | 728 | 317/329/82 | 44% | 1.34x | 1.03x |

## Primitive Summary

Linux CI primitives with matched `rscrypto` comparisons. The fastest-only columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `x25519` | 18 | 6/8/4 | 0.95x | 54 | 42/8/4 | 1.40x |
| `pbkdf2-sha256` | 54 | 23/17/14 | 0.96x | 162 | 121/23/18 | 1.32x |
| `chacha20-poly1305` | 198 | 67/58/73 | 0.98x | 594 | 332/140/122 | 1.18x |
| `hmac-sha512` | 99 | 16/27/56 | 0.99x | 297 | 110/85/102 | 1.14x |
| `hmac-sha384` | 99 | 17/25/57 | 0.99x | 297 | 112/82/103 | 1.14x |
| `rapidhash-v3-64` | 99 | 27/52/20 | 1.05x | 99 | 27/52/20 | 1.05x |
| `blake2/blake2b256` | 225 | 103/119/3 | 1.05x | 351 | 221/127/3 | 1.32x |
| `blake2/blake2b512` | 198 | 107/89/2 | 1.07x | 297 | 201/94/2 | 1.34x |
| `ed25519` | 90 | 41/15/34 | 1.07x | 306 | 220/29/57 | 1.30x |
| `rapidhash-v3-128` | 99 | 34/43/22 | 1.08x | 99 | 34/43/22 | 1.08x |
| `blake2/blake2s256` | 225 | 105/119/1 | 1.09x | 225 | 105/119/1 | 1.09x |
| `sha256` | 117 | 42/55/20 | 1.11x | 315 | 183/99/33 | 1.45x |
| `rapidhash-64` | 99 | 33/56/10 | 1.12x | 99 | 33/56/10 | 1.12x |
| `blake2/blake2s128` | 198 | 122/76/0 | 1.12x | 198 | 122/76/0 | 1.12x |
| `hmac-sha256` | 117 | 26/52/39 | 1.13x | 315 | 138/118/59 | 1.33x |
| `rapidhash-128` | 99 | 37/55/7 | 1.14x | 99 | 37/55/7 | 1.14x |
| `pbkdf2-sha512` | 54 | 24/29/1 | 1.14x | 162 | 125/36/1 | 1.30x |
| `xxh3-128` | 99 | 51/34/14 | 1.17x | 99 | 51/34/14 | 1.17x |
| `xxh3-64` | 99 | 51/35/13 | 1.19x | 99 | 51/35/13 | 1.19x |
| `hkdf-sha256` | 36 | 28/8/0 | 1.19x | 108 | 100/8/0 | 1.90x |
| `sha512` | 117 | 44/62/11 | 1.24x | 315 | 188/110/17 | 1.34x |
| `hkdf-sha384` | 36 | 26/10/0 | 1.24x | 108 | 98/10/0 | 1.68x |
| `sha384` | 99 | 39/49/11 | 1.24x | 297 | 186/94/17 | 1.35x |
| `ascon-hash256` | 99 | 54/44/1 | 1.26x | 99 | 54/44/1 | 1.26x |
| `sha512-256` | 99 | 51/48/0 | 1.28x | 99 | 51/48/0 | 1.28x |
| `aes-128-gcm` | 198 | 105/36/57 | 1.32x | 594 | 464/44/86 | 2.57x |
| `aes-256-gcm` | 198 | 99/40/59 | 1.32x | 594 | 457/48/89 | 2.62x |
| `ascon-xof128` | 99 | 74/25/0 | 1.35x | 99 | 74/25/0 | 1.35x |
| `xchacha20-poly1305` | 198 | 171/25/2 | 1.38x | 198 | 171/25/2 | 1.38x |
| `blake3` | 432 | 225/174/33 | 1.42x | 432 | 225/174/33 | 1.42x |
| `aegis-256` | 198 | 107/70/21 | 1.44x | 198 | 107/70/21 | 1.44x |
| `aes-128-gcm-siv` | 198 | 131/4/63 | 1.55x | 396 | 309/16/71 | 2.89x |
| `crc32` | 99 | 49/33/17 | 1.63x | 198 | 146/35/17 | 2.24x |
| `crc32c` | 99 | 49/38/12 | 1.64x | 198 | 148/38/12 | 2.24x |
| `aes-256-gcm-siv` | 198 | 133/5/60 | 1.65x | 396 | 330/6/60 | 3.23x |
| `crc64-nvme` | 99 | 53/33/13 | 1.73x | 198 | 129/49/20 | 2.25x |
| `sha224` | 99 | 57/42/0 | 1.88x | 99 | 57/42/0 | 1.88x |
| `sha3-256` | 117 | 109/8/0 | 2.13x | 117 | 109/8/0 | 2.13x |
| `sha3-224` | 99 | 93/6/0 | 2.16x | 99 | 93/6/0 | 2.16x |
| `sha3-384` | 99 | 92/7/0 | 2.19x | 99 | 92/7/0 | 2.19x |
| `sha3-512` | 99 | 92/7/0 | 2.22x | 99 | 92/7/0 | 2.22x |
| `crc64-xz` | 99 | 74/12/13 | 2.40x | 99 | 74/12/13 | 2.40x |
| `shake128` | 99 | 95/4/0 | 2.57x | 99 | 95/4/0 | 2.57x |
| `shake256` | 99 | 94/5/0 | 2.61x | 99 | 94/5/0 | 2.61x |
| `crc24-openpgp` | 99 | 97/1/1 | 13.46x | 99 | 97/1/1 | 13.46x |
| `crc16-ccitt` | 99 | 98/0/1 | 23.78x | 99 | 98/0/1 | 23.78x |
| `crc16-ibm` | 99 | 98/0/1 | 24.33x | 99 | 98/0/1 | 24.33x |

## External Pressure

All-pair Linux CI comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rapidhash` | 396 | 131/206/59 | 33% | 1.09x | 1.01x |
| `xxhash-rust` | 198 | 102/69/27 | 52% | 1.18x | 1.07x |
| `ascon-hash` | 198 | 128/69/1 | 65% | 1.31x | 1.22x |
| `dalek` | 108 | 88/13/7 | 81% | 1.40x | 1.38x |
| `blake3` | 432 | 225/174/33 | 52% | 1.42x | 1.06x |
| `aws-lc-rs` | 1,854 | 1,110/300/444 | 60% | 1.43x | 1.14x |
| `aegis-crate` | 198 | 107/70/21 | 54% | 1.44x | 1.06x |
| `sha2` | 531 | 277/247/7 | 52% | 1.44x | 1.05x |
| `ring` | 1,440 | 981/195/264 | 68% | 1.46x | 1.18x |
| `dryoc` | 315 | 287/21/7 | 91% | 1.83x | 1.72x |
| `rustcrypto` | 2,529 | 1,700/711/118 | 67% | 1.91x | 1.14x |
| `crc-fast` | 297 | 176/97/24 | 59% | 2.00x | 1.14x |
| `sha3` | 612 | 575/37/0 | 94% | 2.30x | 2.15x |
| `crc64fast-nvme` | 99 | 73/13/13 | 74% | 2.39x | 2.02x |
| `crc32fast` | 99 | 82/8/9 | 83% | 2.39x | 2.00x |
| `crc64fast` | 99 | 74/12/13 | 75% | 2.40x | 2.05x |
| `crc32c` | 99 | 92/4/3 | 93% | 2.80x | 2.19x |
| `crc` | 297 | 293/1/3 | 99% | 19.82x | 28.87x |

## README Numbers

- **Headline:** 3,369 of 5,895 matched Linux CI fastest-external comparisons are wins; Linux CI geomean is 1.55x.
- **Checksums:** 5.07x geomean across Linux CI fastest-external checksum comparisons.
- **SHA-3 / SHAKE:** 2.18x SHA-3 geomean and 2.59x SHAKE geomean across Linux CI fastest-external comparisons.
- **BLAKE3:** 2.39x geomean for Linux CI fastest-external rows at `>=64 KiB`.
- **AEAD:** 1.36x geomean across Linux CI fastest-external AEAD comparisons.
- **Ed25519 / X25519:** 1.01x Ed25519 sign geomean and 0.95x X25519 geomean across Linux CI fastest-external comparisons.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | ---: | --- |
| AMD Zen4 | `ci` | `2026-05-15 03_26_57` | 1,833 | `benchmark_results/2026-05-15/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-05-15 03_26_57` | 1,833 | `benchmark_results/2026-05-15/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-05-15 03_26_57` | 1,833 | `benchmark_results/2026-05-15/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-05-15 03_26_57` | 1,833 | `benchmark_results/2026-05-15/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-05-15 03_26_57` | 1,833 | `benchmark_results/2026-05-15/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-05-15 03_26_57` | 1,833 | `benchmark_results/2026-05-15/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-05-15 03_26_57` | 1,833 | `benchmark_results/2026-05-15/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-05-15 03_26_57` | 1,833 | `benchmark_results/2026-05-15/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-05-15 03_26_57` | 1,833 | `benchmark_results/2026-05-15/linux/rise-riscv/results.txt` |
| macOS Apple Silicon | `local` | `2026-05-15 21_06_15` | 2,039 | `benchmark_results/2026-05-15/macos/aarch64/results.txt` |

## Methodology Notes

- Parsed 18,536 Criterion rows: 16,497 Linux CI rows and 2,039 local macOS rows.
- Matched 10,990 `rscrypto` external pairs across 6,623 fastest-external cases in the full current corpus.
- Matched 9,801 `rscrypto` external pairs across 5,895 fastest-external cases in Linux CI.
- Linux CI result headers were normalized to `time=03_26_57`, the GitHub Actions run creation time, per extraction policy. Local macOS keeps its local run time.
- CI artifacts shipped with ANSI escape sequences; the 2026-05-15 CI result files were cleaned during extraction.
- Matching is structural: a comparison exists when an external Criterion ID can be produced by replacing the exact `rscrypto` path component while keeping the same platform, primitive path, operation path, and size suffix.
- When an external crate name also appears as a primitive path component, such as `blake3/blake3` or `crc32c/crc32c`, the implementation component is selected by the matching `rscrypto` key rather than by token name alone.
- Diagnostic implementation labels such as `rscrypto-oneshot`, `rscrypto-state`, and `rscrypto-stream-new` are parsed as raw rows but are not counted as public `rscrypto` comparison rows because they are not the exact `rscrypto` implementation component.
- Unmatched external-only rows: 765 in Linux CI, 851 in the full corpus. These are external or diagnostic rows without a corresponding exact `rscrypto` row, so they are not part of the ratio tables.
