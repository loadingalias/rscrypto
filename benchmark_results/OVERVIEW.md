# Benchmark Overview

Sources:

- Linux benchmark CI run [#26537297244](https://github.com/loadingalias/rscrypto/actions/runs/26537297244), created 2026-05-27 20:37:14 UTC.
- Linux commit: `26845c85530d90725d3ad90c7871d7a474d80d27`.
- Apple Silicon local run: `benchmark_results/2026-05-31/macos/aarch64/results.txt`, created 2026-05-31 22:48:16 local time on an MBP M1.
- Apple Silicon commit: `63ba9f3d35c31078a0c3a3f6ab86640db6c4a97c`.

Scope: the 2026-05-27 nine-runner Linux CI benchmark matrix for commit `26845c8`, plus the 2026-05-31 local Apple Silicon full run for commit `63ba9f3`. Ratios are `external_crate_time / rscrypto_time`; higher is always better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and size.

Coverage note: the Linux CI run is filter-based and does not include Argon2, scrypt, or Ascon-AEAD rows. Do not publish password-hashing or Ascon-AEAD claims from that Linux pass except PBKDF2, which is present under `bench=auth`. The Apple Silicon local run is a full run and includes Argon2, scrypt, and Ascon-AEAD rows. Do not combine Linux and Apple Silicon totals as one aggregate because they were collected on different commits and benchmark scopes. IBM POWER10 and IBM Z still exclude `aws-lc-rs` rows where that dependency is not enabled for the target.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Linux CI: all matched performance pairs | 9,409 | 6479/2105/825 | 69% | 1.77x | 1.20x |
| Linux CI: fastest external per case | 5,832 | 3545/1665/622 | 61% | 1.61x | 1.11x |
| Apple Silicon: all matched performance pairs | 1,156 | 699/380/77 | 60% | 1.80x | 1.12x |
| Apple Silicon: fastest external per case | 703 | 344/300/59 | 49% | 1.39x | 1.05x |

Shareable release summary:

- **Headline:** 3,545 of 5,832 matched Linux CI fastest-external comparisons are wins; 5,210 are wins or ties. Linux CI fastest-external geomean is 1.61x.
- **Apple Silicon headline:** 344 of 703 matched MBP M1 fastest-external comparisons are wins; 644 are wins or ties. Apple Silicon fastest-external geomean is 1.39x across the full local run.
- **Checksums:** 5.03x geomean across Linux CI fastest-external checksum comparisons.
- **Apple Silicon checksums:** 5.39x geomean across fastest-external checksum comparisons.
- **SHA-3 / SHAKE:** 2.15x SHA-3 geomean and 1.86x SHAKE geomean across Linux CI fastest-external comparisons.
- **Apple Silicon SHA-3 / SHAKE:** 1.03x SHA-3 all-pair geomean and 1.37x SHAKE fastest-external geomean.
- **BLAKE3:** 2.31x geomean for Linux CI fastest-external rows at `>=64 KiB`.
- **Apple Silicon BLAKE3:** 1.86x geomean for fastest-external rows at `>=64 KiB`; the 64 KiB rows lose, while larger rows recover.
- **AEAD:** 1.57x geomean across Linux CI fastest-external AEAD comparisons.
- **Apple Silicon AEAD:** 1.52x geomean across fastest-external AEAD comparisons, with 152 wins-or-ties out of 154 rows.
- **RSA:** 75 wins and 24 losses across 99 Linux CI fastest-external RSA import and verification comparisons; geomean is 1.32x. Verification-only is effectively tied at 0.98x.
- **Apple Silicon RSA:** 11 wins out of 11 fastest-external RSA import and verification comparisons; geomean is 1.44x. Verification-only is 1.19x.
- **Ed25519 / X25519:** 1.14x Ed25519 sign geomean, 1.00x Ed25519 verify geomean, and 0.95x X25519 geomean across Linux CI fastest-external comparisons.
- **Apple Silicon Ed25519 / X25519:** 1.02x Ed25519 sign geomean, 1.00x Ed25519 verify geomean, and 1.00x X25519 geomean across fastest-external comparisons.
- **Linux top losses:** PBKDF2-SHA256 `iters=1` is 0.81x, X25519 DH is 0.92x, and RSA-4096 verification is 0.94x.
- **Apple Silicon top losses:** Argon2 small-memory rows are 0.19x..0.25x, SHA-512 1-byte hashing is 0.59x, HMAC-SHA256 setup rows are below parity, and 64 KiB BLAKE3 rows lose before larger inputs recover.

Use the headline and category bullets for README, release notes, and social
posts. Use the loss table below when discussing tradeoffs; it prevents the
claim from sounding like a synthetic benchmark victory lap.

## Coverage Matrix

| Platform | Parsed Criterion rows | Fastest rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 1,944 | 648 | 448/144/56 | 69% | 1.48x | 1.13x |
| AMD Zen5 | 1,944 | 648 | 375/207/66 | 58% | 1.48x | 1.08x |
| AWS Graviton3 | 1,944 | 648 | 324/245/79 | 50% | 1.41x | 1.05x |
| AWS Graviton4 | 1,944 | 648 | 333/265/50 | 51% | 1.42x | 1.05x |
| IBM Power10 | 1,721 | 648 | 359/262/27 | 55% | 2.03x | 1.08x |
| IBM z16/s390x | 1,721 | 648 | 546/61/41 | 84% | 3.21x | 2.41x |
| Intel Ice Lake | 1,944 | 648 | 431/128/89 | 67% | 1.47x | 1.16x |
| Intel Sapphire Rapids | 1,944 | 648 | 434/141/73 | 67% | 1.60x | 1.14x |
| RISE RISC-V | 1,944 | 648 | 295/212/141 | 46% | 1.11x | 1.03x |
| Apple Silicon MBP M1 | 2,090 | 703 | 344/300/59 | 49% | 1.39x | 1.05x |

## Category Summary

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 693 | 514/114/65 | 74% | 5.03x | 2.33x |
| Hashes/MACs | 3,438 | 1972/1258/208 | 57% | 1.33x | 1.08x |
| Auth/KDF/password | 216 | 142/51/23 | 66% | 1.13x | 1.09x |
| RSA | 99 | 75/0/24 | 76% | 1.32x | 1.16x |
| AEAD | 1,386 | 842/242/302 | 61% | 1.57x | 1.15x |

## Apple Silicon Category Summary

The 2026-05-31 macOS/aarch64 run is local MBP M1 evidence, not CI. It is a full run, so its totals include Argon2, scrypt, and Ascon-AEAD rows that are absent from the Linux CI pass above.

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 77 | 59/16/2 | 77% | 5.39x | 1.66x |
| Hashes/MACs | 403 | 145/209/49 | 36% | 1.10x | 1.01x |
| Auth/KDF/password | 47 | 13/28/6 | 28% | 0.94x | 1.01x |
| RSA | 11 | 11/0/0 | 100% | 1.44x | 1.21x |
| AEAD | 154 | 105/47/2 | 68% | 1.52x | 1.19x |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `x25519` | 18 | 6/8/4 | 0.95x | 50 | 38/8/4 | 1.40x |
| `rsa-8192` | 18 | 12/0/6 | 0.98x | 32 | 20/2/10 | 0.98x |
| `chacha20-poly1305` | 198 | 67/58/73 | 0.99x | 550 | 294/135/121 | 1.15x |
| `rapidhash-v3-64` | 99 | 25/53/21 | 1.05x | 99 | 25/53/21 | 1.05x |
| `pbkdf2-sha256` | 54 | 46/4/4 | 1.05x | 150 | 137/6/7 | 1.33x |
| `rapidhash-v3-128` | 99 | 33/43/23 | 1.07x | 99 | 33/43/23 | 1.07x |
| `blake2` | 792 | 476/312/4 | 1.10x | 990 | 664/322/4 | 1.25x |
| `rapidhash-64` | 99 | 32/57/10 | 1.12x | 99 | 32/57/10 | 1.12x |
| `rapidhash-128` | 99 | 36/56/7 | 1.14x | 99 | 36/56/7 | 1.14x |
| `hmac-sha256` | 99 | 23/43/33 | 1.14x | 275 | 117/104/54 | 1.42x |
| `ed25519` | 90 | 45/30/15 | 1.15x | 290 | 217/52/21 | 1.41x |
| `xxh3-128` | 99 | 51/35/13 | 1.18x | 99 | 51/35/13 | 1.18x |
| `xxh3-64` | 99 | 51/34/14 | 1.19x | 99 | 51/34/14 | 1.19x |
| `hkdf-sha256` | 36 | 33/3/0 | 1.22x | 100 | 97/3/0 | 1.81x |
| `pbkdf2-sha512` | 54 | 45/9/0 | 1.23x | 150 | 140/10/0 | 1.30x |
| `sha512` | 99 | 41/47/11 | 1.24x | 275 | 168/92/15 | 1.28x |
| `hkdf-sha384` | 36 | 36/0/0 | 1.24x | 100 | 100/0/0 | 1.56x |
| `hmac-sha512` | 99 | 48/44/7 | 1.24x | 275 | 178/86/11 | 1.30x |
| `sha384` | 99 | 42/46/11 | 1.25x | 275 | 170/89/16 | 1.29x |
| `hmac-sha384` | 99 | 49/42/8 | 1.25x | 275 | 179/83/13 | 1.30x |
| `sha256` | 99 | 39/47/13 | 1.25x | 275 | 160/88/27 | 1.49x |
| `ascon-hash256` | 99 | 56/42/1 | 1.27x | 99 | 56/42/1 | 1.27x |
| `sha512-256` | 99 | 58/41/0 | 1.30x | 99 | 58/41/0 | 1.30x |
| `rsa-4096` | 27 | 21/0/6 | 1.35x | 59 | 49/0/10 | 2.03x |
| `ascon-xof128` | 99 | 75/24/0 | 1.37x | 99 | 75/24/0 | 1.37x |
| `xchacha20-poly1305` | 198 | 172/26/0 | 1.38x | 198 | 172/26/0 | 1.38x |
| `rsa-3072` | 27 | 21/0/6 | 1.41x | 59 | 49/0/10 | 2.09x |
| `blake3` | 396 | 216/150/30 | 1.43x | 396 | 216/150/30 | 1.43x |
| `aegis-256` | 198 | 106/70/22 | 1.44x | 198 | 106/70/22 | 1.44x |
| `rsa-2048` | 27 | 21/0/6 | 1.48x | 59 | 49/0/10 | 2.15x |
| `crc32c` | 99 | 49/32/18 | 1.63x | 198 | 144/34/20 | 2.23x |
| `crc32` | 99 | 48/34/17 | 1.66x | 198 | 144/37/17 | 2.29x |
| `crc64-nvme` | 99 | 52/33/14 | 1.74x | 198 | 130/47/21 | 2.26x |
| `aes-128-gcm` | 198 | 112/43/43 | 1.75x | 550 | 427/52/71 | 2.55x |
| `aes-256-gcm` | 198 | 109/38/51 | 1.78x | 550 | 420/47/83 | 2.58x |
| `shake128` | 99 | 76/23/0 | 1.84x | 99 | 76/23/0 | 1.84x |
| `shake256` | 99 | 72/27/0 | 1.87x | 99 | 72/27/0 | 1.87x |
| `sha224` | 99 | 53/44/2 | 1.89x | 99 | 53/44/2 | 1.89x |
| `aes-128-gcm-siv` | 198 | 138/2/58 | 1.90x | 352 | 269/17/66 | 2.70x |
| `aes-256-gcm-siv` | 198 | 138/5/55 | 2.05x | 352 | 291/6/55 | 2.97x |
| `sha3-256` | 99 | 87/12/0 | 2.11x | 99 | 87/12/0 | 2.11x |
| `sha3-224` | 99 | 89/10/0 | 2.13x | 99 | 89/10/0 | 2.13x |
| `sha3-384` | 99 | 88/11/0 | 2.15x | 99 | 88/11/0 | 2.15x |
| `sha3-512` | 99 | 87/12/0 | 2.20x | 99 | 87/12/0 | 2.20x |
| `crc64-xz` | 99 | 75/12/12 | 2.44x | 99 | 75/12/12 | 2.44x |
| `crc24-openpgp` | 99 | 94/3/2 | 13.11x | 99 | 94/3/2 | 13.11x |
| `crc16-ccitt` | 99 | 98/0/1 | 22.81x | 99 | 98/0/1 | 22.81x |
| `crc16-ibm` | 99 | 98/0/1 | 23.86x | 99 | 98/0/1 | 23.86x |

## Clear Losses

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | --- | --- | --- | --- | --- |
| `pbkdf2-sha256` / `iters=1` | 18 | 13/3/2 | 0.81x | 1.12x | `rustcrypto` 16, `aws-lc-rs` 2 |
| `x25519` / `diffie-hellman` | 9 | 3/4/2 | 0.92x | 1.01x | `aws-lc-rs` 7, `dryoc` 1, `dalek` 1 |
| `rsa-4096` / `verify-pkcs1v15-sha256` | 9 | 6/0/3 | 0.94x | 1.13x | `aws-lc-rs` 7, `ring` 1, `rustcrypto-rsa` 1 |
| `rsa-4096` / `verify-pss-sha256` | 9 | 6/0/3 | 0.94x | 1.13x | `aws-lc-rs` 7, `ring` 1, `rustcrypto-rsa` 1 |
| `rsa-3072` / `verify-pkcs1v15-sha256` | 9 | 6/0/3 | 0.96x | 1.15x | `aws-lc-rs` 7, `ring` 1, `rustcrypto-rsa` 1 |
| `rsa-3072` / `verify-pss-sha256` | 9 | 6/0/3 | 0.96x | 1.15x | `aws-lc-rs` 7, `ring` 1, `rustcrypto-rsa` 1 |
| `rsa-8192` / `verify-pkcs1v15-sha256` | 9 | 6/0/3 | 0.97x | 1.10x | `aws-lc-rs` 7, `ring` 2 |
| `x25519` / `public-key-from-secret` | 9 | 3/4/2 | 0.98x | 1.01x | `aws-lc-rs` 7, `dryoc` 1, `dalek` 1 |
| `rsa-8192` / `verify-pss-sha256` | 9 | 6/0/3 | 0.98x | 1.10x | `aws-lc-rs` 7, `ring` 2 |
| `chacha20-poly1305` / `encrypt` | 99 | 30/33/36 | 0.98x | 1.00x | `ring` 65, `aws-lc-rs` 23, `rustcrypto` 11 |
| `ed25519` / `verify` | 36 | 10/13/13 | 1.00x | 1.00x | `dalek` 17, `aws-lc-rs` 9, `ring` 9, `dryoc` 1 |
| `chacha20-poly1305` / `decrypt` | 99 | 37/25/37 | 1.00x | 1.00x | `ring` 47, `aws-lc-rs` 46, `rustcrypto` 6 |
| `rsa-2048` / `verify-pkcs1v15-sha256` | 9 | 6/0/3 | 1.03x | 1.23x | `aws-lc-rs` 7, `ring` 1, `rustcrypto-rsa` 1 |
| `rsa-2048` / `verify-pss-sha256` | 9 | 6/0/3 | 1.03x | 1.22x | `aws-lc-rs` 7, `ring` 1, `rustcrypto-rsa` 1 |
| `rapidhash-v3-64` / `hash` | 99 | 25/53/21 | 1.05x | 1.00x | `rapidhash` 99 |
| `hmac-sha256` / `hash` | 99 | 23/43/33 | 1.14x | 0.99x | `ring` 72, `rustcrypto` 22, `aws-lc-rs` 5 |
| `aes-128-gcm` / `encrypt` | 99 | 54/20/25 | 1.67x | 1.10x | `rustcrypto` 36, `aws-lc-rs` 32, `ring` 31 |
| `aes-256-gcm` / `encrypt` | 99 | 50/18/31 | 1.69x | 1.06x | `rustcrypto` 36, `ring` 32, `aws-lc-rs` 31 |
| `aes-128-gcm-siv` / `decrypt` | 99 | 69/1/29 | 1.88x | 1.63x | `aws-lc-rs` 66, `rustcrypto` 33 |
| `aes-128-gcm-siv` / `encrypt` | 99 | 69/1/29 | 1.93x | 1.77x | `aws-lc-rs` 66, `rustcrypto` 33 |
| `aes-256-gcm-siv` / `decrypt` | 99 | 69/2/28 | 1.98x | 1.55x | `aws-lc-rs` 66, `rustcrypto` 33 |
| `aes-256-gcm-siv` / `encrypt` | 99 | 69/3/27 | 2.12x | 1.89x | `aws-lc-rs` 66, `rustcrypto` 33 |

## Top Three Loss Areas

1. **PBKDF2-SHA256 setup cost at `iters=1`**: 0.81x geomean across 18 fastest-external rows. The catastrophic individual losses are on Intel Sapphire Rapids against `aws-lc-rs` for 32-byte and 64-byte outputs (0.03x worst row), which points at fixed overhead rather than long-run HMAC throughput.
2. **Arm/RISC-V public-key paths against `aws-lc-rs`**: X25519 DH is 0.92x overall, with Graviton3/4 rows around 0.55x. RSA verification has the same platform shape: Graviton3/4 3072/4096/8192-bit verification rows cluster around 0.36x..0.43x, and RISE RISC-V is 0.56x..0.81x.
3. **Small-message AEAD frontend overhead**: Chacha20-Poly1305 is 0.99x overall with 73 losses; the worst rows are 1-byte and 32-byte encrypt/decrypt against `ring` or `aws-lc-rs`. AES-GCM-SIV and AES-GCM have similar small-input losses even though larger sustained rows recover strongly on most x86 and IBM platforms.

## External Pressure

All-pair Linux CI comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `rapidhash` | 396 | 126/209/61 | 32% | 1.09x | 1.01x |
| `xxhash-rust` | 198 | 102/69/27 | 52% | 1.19x | 1.07x |
| `aws-lc-rs` | 1,498 | 858/337/303 | 57% | 1.20x | 1.09x |
| `ascon-hash` | 198 | 131/66/1 | 66% | 1.32x | 1.25x |
| `blake3` | 396 | 216/150/30 | 55% | 1.43x | 1.09x |
| `aegis-crate` | 198 | 106/70/22 | 54% | 1.44x | 1.06x |
| `dalek` | 108 | 92/11/5 | 85% | 1.49x | 1.44x |
| `ring` | 1,512 | 1112/207/193 | 74% | 1.51x | 1.22x |
| `sha2` | 495 | 289/203/3 | 58% | 1.51x | 1.07x |
| `tiny-keccak` | 198 | 148/50/0 | 75% | 1.86x | 2.03x |
| `dryoc` | 288 | 262/24/2 | 91% | 1.89x | 1.80x |
| `rustcrypto` | 2,457 | 1822/531/104 | 74% | 1.98x | 1.18x |
| `crc-fast` | 297 | 176/95/26 | 59% | 2.02x | 1.15x |
| `sha3` | 396 | 351/45/0 | 89% | 2.15x | 2.11x |
| `crc64fast-nvme` | 99 | 73/13/13 | 74% | 2.41x | 2.21x |
| `crc32fast` | 99 | 82/6/11 | 83% | 2.43x | 2.02x |
| `crc64fast` | 99 | 75/12/12 | 76% | 2.44x | 2.07x |
| `crc32c` | 99 | 87/4/8 | 88% | 2.76x | 2.19x |
| `rustcrypto-rsa` | 81 | 81/0/0 | 100% | 4.91x | 3.90x |
| `crc` | 297 | 290/3/4 | 98% | 19.25x | 29.47x |

## README Numbers

- **Headline:** 3,545 of 5,832 matched Linux CI fastest-external comparisons are wins; 5,210 are wins or ties. Linux CI geomean is 1.61x.
- **Apple Silicon headline:** 344 of 703 matched MBP M1 fastest-external comparisons are wins; 644 are wins or ties. Apple Silicon geomean is 1.39x across the full local run.
- **Checksums:** 5.03x geomean across Linux CI fastest-external checksum comparisons.
- **Apple Silicon checksums:** 5.39x geomean across fastest-external checksum comparisons.
- **SHA-3 / SHAKE:** 2.15x SHA-3 geomean and 1.86x SHAKE geomean across Linux CI fastest-external comparisons.
- **Apple Silicon SHA-3 / SHAKE:** 1.03x SHA-3 all-pair geomean and 1.37x SHAKE fastest-external geomean.
- **BLAKE3:** 2.31x geomean for Linux CI fastest-external rows at `>=64 KiB`.
- **Apple Silicon BLAKE3:** 1.86x geomean for fastest-external rows at `>=64 KiB`.
- **AEAD:** 1.57x geomean across Linux CI fastest-external AEAD comparisons.
- **Apple Silicon AEAD:** 1.52x geomean across fastest-external AEAD comparisons.
- **RSA:** 1.32x geomean and 76% wins across Linux CI fastest-external RSA import and verification comparisons; verification-only is 0.98x.
- **Apple Silicon RSA:** 1.44x geomean and 100% wins across fastest-external RSA import and verification comparisons; verification-only is 1.19x.
- **Ed25519 / X25519:** 1.14x Ed25519 sign geomean, 1.00x Ed25519 verify geomean, and 0.95x X25519 geomean across Linux CI fastest-external comparisons.
- **Apple Silicon Ed25519 / X25519:** 1.02x Ed25519 sign geomean, 1.00x Ed25519 verify geomean, and 1.00x X25519 geomean across fastest-external comparisons.
- **Linux top losses:** PBKDF2-SHA256 `iters=1` is 0.81x, X25519 DH is 0.92x, and RSA-4096 verification is 0.94x.
- **Apple Silicon top losses:** Argon2 small-memory rows are 0.19x..0.25x, SHA-512 1-byte hashing is 0.59x, HMAC-SHA256 setup rows are below parity, and 64 KiB BLAKE3 rows lose before larger inputs recover.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-05-27 20_37_14` | 1,944 | `benchmark_results/2026-05-27/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-05-27 20_37_14` | 1,944 | `benchmark_results/2026-05-27/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-05-27 20_37_14` | 1,944 | `benchmark_results/2026-05-27/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-05-27 20_37_14` | 1,944 | `benchmark_results/2026-05-27/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-05-27 20_37_14` | 1,721 | `benchmark_results/2026-05-27/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-05-27 20_37_14` | 1,721 | `benchmark_results/2026-05-27/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-05-27 20_37_14` | 1,944 | `benchmark_results/2026-05-27/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-05-27 20_37_14` | 1,944 | `benchmark_results/2026-05-27/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-05-27 20_37_14` | 1,944 | `benchmark_results/2026-05-27/linux/rise-riscv/results.txt` |
| Apple Silicon MBP M1 | `local` | `2026-05-31 22_48_16` | 2,090 | `benchmark_results/2026-05-31/macos/aarch64/results.txt` |
