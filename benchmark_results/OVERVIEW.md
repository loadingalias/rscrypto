# Benchmark Overview

Sources:

- Linux benchmark CI run [#25979567024](https://github.com/loadingalias/rscrypto/actions/runs/25979567024), created 2026-05-17 02:51:52 UTC.
- Weekly CI run [#25979569515](https://github.com/loadingalias/rscrypto/actions/runs/25979569515) validated the same commit; benchmark artifacts came from the adjacent Bench workflow run above.
- macOS Apple Silicon local run: `benchmark_results/2026-05-17/macos/aarch64/results.txt`, created 2026-05-17 15_57_41 local time.
- Commit: `1c30a68b667f964982a519a938065f98989322b2`.

Scope: current benchmark results for commit `1c30a68`. The primary release scorecard is the nine-runner Linux CI benchmark matrix because it is one reproducible CI run. The full corpus additionally includes one local macOS Apple Silicon run from the same commit; it is useful host evidence, not a CI-wide claim. Ratios are `rscrypto` throughput divided by external throughput for throughput rows. For latency-only rows, ratios are external time divided by `rscrypto` time. Higher is always better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and size.

Important coverage note: the 2026-05-17 Linux CI run uses the full `bench=all` corpus on every runner. Unlike the 2026-05-15 scorecard, it includes password hashing and KMAC/cSHAKE rows in the reproducible Linux CI matrix.

## Headline

Linux CI scorecard:

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| All matched performance pairs | 10,701 | 7,300/2,437/964 | 68% | 1.76x | 1.20x |
| Fastest external per case | 6,552 | 3,807/1,966/779 | 58% | 1.52x | 1.09x |
| All matched throughput pairs | 9,495 | 6,392/2,313/790 | 67% | 1.79x | 1.19x |
| Fastest external, throughput only | 5,796 | 3,314/1,866/616 | 57% | 1.56x | 1.08x |
| All matched latency-only pairs | 1,206 | 908/124/174 | 75% | 1.52x | 1.27x |
| Fastest external, latency only | 756 | 493/100/163 | 65% | 1.25x | 1.13x |

Full current corpus, including local macOS Apple Silicon:

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| All matched performance pairs | 11,890 | 7,978/2,862/1,050 | 67% | 1.75x | 1.19x |
| Fastest external per case | 7,280 | 4,120/2,310/850 | 57% | 1.51x | 1.08x |
| All matched throughput pairs | 10,550 | 6,976/2,710/864 | 66% | 1.79x | 1.17x |
| Fastest external, throughput only | 6,440 | 3,579/2,184/677 | 56% | 1.54x | 1.08x |
| All matched latency-only pairs | 1,340 | 1,002/152/186 | 75% | 1.52x | 1.27x |
| Fastest external, latency only | 840 | 541/126/173 | 64% | 1.24x | 1.13x |

What matters:

- Linux CI fastest-external scorecard: 3,807 wins / 1,966 ties / 779 losses across 6,552 comparisons, with a 1.52x geomean.
- Adding the local Apple Silicon run gives 4,120 wins / 2,310 ties / 850 losses across 7,280 fastest-external comparisons, with a 1.51x geomean.
- Checksums remain the cleanest broad win: 5.10x fastest-external Linux CI geomean.
- SHA-3 and SHAKE remain strong: 2.17x and 2.60x Linux CI geomeans respectively.
- BLAKE3 sustained rows (`>=64 KiB`) hold at 2.38x Linux CI geomean against the `blake3` crate.
- AEAD is still broadly positive at 1.37x Linux CI geomean; GCM-SIV, XChaCha20-Poly1305, AEGIS-256, and IBM Z remain the strongest areas.
- The SHA-512-family work paid off: HMAC-SHA384 and HMAC-SHA512 now both sit at 1.23x / 1.23x Linux CI geomean, and local Apple Silicon HMAC-SHA512 is 1.02x.
- SHA-256 streaming is no longer the broad loss it was on 2026-05-15: it moved to 1.20x Linux CI geomean.
- Ed25519 verification improved but is not closed on Linux CI: `verify` is 0.95x geomean. The local Apple Silicon verify path is now 1.01x, while local Ed25519 signing remains weak.
- New full-corpus pressure is password hashing: Linux CI password-hashing fastest-external rows are 0.81x geomean, led by Argon2 and scrypt small/OWASP losses.

## Clear Losses

Collapsed fastest-external Linux CI groups. This is the root-cause view across the full 2026-05-17 benchmark corpus.

| Place | Fastest rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `scrypt-small` | 36 | 19/1/16 | 0.75x | 1.07x | `rustcrypto` 36 |
| `scrypt-owasp` | 9 | 4/1/4 | 0.79x | 0.97x | `rustcrypto` 9 |
| `argon2id-small` | 27 | 7/2/18 | 0.82x | 0.83x | `rustcrypto` 27 |
| `argon2d-small` | 27 | 7/2/18 | 0.82x | 0.86x | `rustcrypto` 27 |
| `argon2i-small` | 27 | 7/3/17 | 0.82x | 0.80x | `rustcrypto` 27 |
| `argon2id-owasp` | 9 | 2/1/6 | 0.94x | 0.94x | `rustcrypto` 9 |
| `pbkdf2-sha256` | 54 | 23/17/14 | 0.95x | 1.03x | `rustcrypto` 42, `aws-lc-rs` 12 |
| `x25519` | 18 | 6/8/4 | 0.95x | 1.01x | `aws-lc-rs` 14, `dryoc` 3, `dalek` 1 |
| `chacha20-poly1305` | 198 | 68/57/73 | 0.99x | 1.00x | `ring` 103, `aws-lc-rs` 81, `rustcrypto` 14 |

Top measured Linux CI primitive/operation losses, fastest external only, excluding tiny sample groups (`rows < 10`):

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `scrypt-small` / `hash` | 36 | 19/1/16 | 0.75x | 1.07x | `rustcrypto` 36 |
| `pbkdf2-sha256` / `iters=1` | 18 | 9/5/4 | 0.77x | 1.05x | `rustcrypto` 14, `aws-lc-rs` 4 |
| `argon2id-small` / `hash` | 27 | 7/2/18 | 0.82x | 0.83x | `rustcrypto` 27 |
| `argon2d-small` / `hash` | 27 | 7/2/18 | 0.82x | 0.86x | `rustcrypto` 27 |
| `argon2i-small` / `hash` | 27 | 7/3/17 | 0.82x | 0.80x | `rustcrypto` 27 |
| `ed25519` / `verify` | 36 | 10/8/18 | 0.95x | 0.95x | `dalek` 18, `aws-lc-rs` 9, `ring` 9 |
| `chacha20-poly1305` / `encrypt` | 99 | 33/30/36 | 0.98x | 1.00x | `ring` 63, `aws-lc-rs` 28, `rustcrypto` 8 |
| `chacha20-poly1305` / `decrypt` | 99 | 35/27/37 | 0.99x | 1.00x | `aws-lc-rs` 53, `ring` 40, `rustcrypto` 6 |

Small-sample Linux CI losses are listed separately because they should not dominate release triage:

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `scrypt-owasp` / `hash` | 9 | 4/1/4 | 0.79x | 0.97x | `rustcrypto` 9 |
| `x25519` / `diffie-hellman` | 9 | 3/4/2 | 0.93x | 1.01x | `aws-lc-rs` 7, `dryoc` 2 |
| `argon2id-owasp` / `hash` | 9 | 2/1/6 | 0.94x | 0.94x | `rustcrypto` 9 |
| `x25519` / `public-key-from-secret` | 9 | 3/4/2 | 0.98x | 1.02x | `aws-lc-rs` 7, `dryoc` 1, `dalek` 1 |

Local Apple Silicon notable losses from `bench=all`:

| Primitive | Rows | W/T/L | Geomean | Median | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `argon2id-small` | 3 | 0/2/1 | 0.58x | 0.96x | local-only host evidence |
| `argon2d-small` | 3 | 0/2/1 | 0.59x | 0.97x | local-only host evidence |
| `argon2i-small` | 3 | 0/2/1 | 0.60x | 0.95x | local-only host evidence |
| `ed25519` | 10 | 2/4/4 | 0.90x | 1.00x | local-only host evidence |
| `hkdf-sha256` | 4 | 0/2/2 | 0.92x | 0.95x | local-only host evidence |
| `hmac-sha256` | 13 | 3/2/8 | 0.93x | 0.89x | local-only host evidence |
| `sha256` | 13 | 1/5/7 | 0.96x | 0.95x | local-only host evidence |
| `pbkdf2-sha512` | 6 | 0/4/2 | 0.96x | 0.95x | local-only host evidence |
| `pbkdf2-sha256` | 6 | 0/3/3 | 0.96x | 0.95x | local-only host evidence |
| `shake128` | 11 | 4/1/6 | 0.97x | 0.92x | local-only host evidence |
| `xxh3-64` | 11 | 1/4/6 | 0.97x | 0.88x | local-only host evidence |
| `sha224` | 11 | 0/10/1 | 0.98x | 1.01x | local-only host evidence |
| `hkdf-sha384` | 4 | 0/4/0 | 0.99x | 0.99x | local-only host evidence |

## Non-Crypto Hash Truth Split

Same-algorithm rows answer whether the strict implementation is competitive with its reference crate. Fastest-class rows answer whether the strict RapidHash/XXH3 families can beat the fastest external non-crypto hash in their class.

Same-algorithm Linux CI comparisons:

| Comparison | Rows | W/T/L | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `xxh3-64` vs `xxhash-rust` | 99 | 52/34/13 | 1.19x | 1.08x |
| `xxh3-128` vs `xxhash-rust` | 99 | 49/39/11 | 1.18x | 1.05x |
| `rapidhash-64` vs `rapidhash` | 99 | 34/56/9 | 1.12x | 1.01x |
| `rapidhash-128` vs `rapidhash` | 99 | 37/55/7 | 1.14x | 1.02x |
| `rapidhash-v3-64` vs `rapidhash` | 99 | 26/53/20 | 1.05x | 1.00x |
| `rapidhash-v3-128` vs `rapidhash` | 99 | 34/43/22 | 1.07x | 1.00x |

Fastest-class Linux CI comparisons:

| Comparison | Rows | W/T/L | All-platform geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| 64-bit RapidHash/XXH3 fastest-class | 297 | 112/143/42 | 1.12x | 1.01x | `rapidhash` 198, `xxhash-rust` 99 |
| 128-bit RapidHash/XXH3 fastest-class | 297 | 120/137/40 | 1.13x | 1.02x | `rapidhash` 198, `xxhash-rust` 99 |

## Sustained AEAD

Fastest-external throughput comparisons only, sizes `65536`, `262144`, and `1048576`, both encrypt and decrypt, 128-bit and 256-bit keys where applicable.

| Platform | AES-GCM W/T/L | AES-GCM geomean | GCM-SIV W/T/L | GCM-SIV geomean | ChaCha20-Poly1305 W/T/L | ChaCha20-Poly1305 geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 0/12/0 | 1.00x | 12/0/0 | 1.74x | 0/0/6 | 0.93x |
| AMD Zen5 | 0/8/4 | 0.98x | 12/0/0 | 2.02x | 6/0/0 | 1.17x |
| AWS Graviton3 | 0/0/12 | 0.90x | 12/0/0 | 2.53x | 0/6/0 | 1.00x |
| AWS Graviton4 | 0/10/2 | 0.96x | 12/0/0 | 2.53x | 0/6/0 | 1.00x |
| IBM Power10 | 0/0/12 | 0.37x | 6/0/6 | 1.47x | 6/0/0 | 1.15x |
| IBM z16/s390x | 12/0/0 | 12.59x | 12/0/0 | 5.58x | 6/0/0 | 2.08x |
| Intel Ice Lake | 3/9/0 | 1.02x | 12/0/0 | 1.34x | 3/3/0 | 1.05x |
| Intel Sapphire Rapids | 1/8/3 | 0.96x | 12/0/0 | 1.49x | 3/3/0 | 1.07x |
| RISE RISC-V | 0/0/12 | 0.81x | 0/1/11 | 0.90x | 0/0/6 | 0.94x |
| macOS Apple Silicon | 3/8/1 | 1.10x | 12/0/0 | 2.78x | 2/3/1 | 1.06x |

Primitive-level sustained AEAD geomeans:

| Platform | AES-128-GCM | AES-256-GCM | AES-128-GCM-SIV | AES-256-GCM-SIV | ChaCha20-Poly1305 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 1.01x | 0.99x | 1.78x | 1.69x | 0.93x |
| AMD Zen5 | 1.01x | 0.94x | 2.02x | 2.02x | 1.17x |
| AWS Graviton3 | 0.90x | 0.90x | 2.40x | 2.67x | 1.00x |
| AWS Graviton4 | 0.96x | 0.97x | 2.38x | 2.69x | 1.00x |
| IBM Power10 | 0.36x | 0.38x | 1.45x | 1.49x | 1.15x |
| IBM z16/s390x | 11.99x | 13.21x | 5.30x | 5.87x | 2.08x |
| Intel Ice Lake | 1.03x | 1.01x | 1.31x | 1.37x | 1.05x |
| Intel Sapphire Rapids | 0.98x | 0.94x | 1.46x | 1.53x | 1.07x |
| RISE RISC-V | 0.80x | 0.81x | 0.89x | 0.91x | 0.94x |
| macOS Apple Silicon | 1.19x | 1.02x | 2.78x | 2.78x | 1.06x |

## Platform Summary

Fastest-external comparisons across every matched primitive, operation, and size. The macOS row is local; every other row is Linux CI.

| Platform | Rows | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 728 | 506/151/71 | 70% | 1.45x | 1.10x |
| AMD Zen5 | 728 | 378/271/79 | 52% | 1.46x | 1.05x |
| AWS Graviton3 | 728 | 357/268/103 | 49% | 1.38x | 1.04x |
| AWS Graviton4 | 728 | 355/309/64 | 49% | 1.40x | 1.05x |
| IBM Power10 | 728 | 316/316/96 | 43% | 1.43x | 1.03x |
| IBM z16/s390x | 728 | 586/94/48 | 80% | 2.93x | 2.07x |
| Intel Ice Lake | 728 | 498/142/88 | 68% | 1.48x | 1.18x |
| Intel Sapphire Rapids | 728 | 487/154/87 | 67% | 1.59x | 1.15x |
| RISE RISC-V | 728 | 324/261/143 | 45% | 1.11x | 1.02x |
| macOS Apple Silicon | 728 | 313/344/71 | 43% | 1.35x | 1.04x |

## Primitive Summary

Linux CI primitives with matched `rscrypto` comparisons. The fastest-only columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `scrypt-small` | 36 | 19/1/16 | 0.75x | 36 | 19/1/16 | 0.75x |
| `scrypt-owasp` | 9 | 4/1/4 | 0.79x | 9 | 4/1/4 | 0.79x |
| `argon2id-small` | 27 | 7/2/18 | 0.82x | 45 | 24/2/19 | 1.23x |
| `argon2d-small` | 27 | 7/2/18 | 0.82x | 27 | 7/2/18 | 0.82x |
| `argon2i-small` | 27 | 7/3/17 | 0.82x | 45 | 24/3/18 | 1.21x |
| `argon2id-owasp` | 9 | 2/1/6 | 0.94x | 18 | 8/3/7 | 1.22x |
| `pbkdf2-sha256` | 54 | 23/17/14 | 0.95x | 162 | 126/18/18 | 1.32x |
| `x25519` | 18 | 6/8/4 | 0.95x | 54 | 42/8/4 | 1.40x |
| `chacha20-poly1305` | 198 | 68/57/73 | 0.99x | 594 | 341/134/119 | 1.18x |
| `rapidhash-v3-64` | 99 | 26/53/20 | 1.05x | 99 | 26/53/20 | 1.05x |
| `blake2/blake2b256` | 387 | 203/169/15 | 1.06x | 612 | 415/182/15 | 1.36x |
| `rapidhash-v3-128` | 99 | 34/43/22 | 1.07x | 99 | 34/43/22 | 1.07x |
| `ed25519` | 90 | 43/11/36 | 1.07x | 306 | 224/35/47 | 1.31x |
| `blake2/blake2b512` | 198 | 108/89/1 | 1.08x | 396 | 302/93/1 | 1.53x |
| `blake2/blake2s256` | 387 | 203/175/9 | 1.10x | 387 | 203/175/9 | 1.10x |
| `blake2/blake2s128` | 198 | 120/78/0 | 1.11x | 198 | 120/78/0 | 1.11x |
| `hmac-sha256` | 117 | 22/52/43 | 1.12x | 315 | 130/124/61 | 1.32x |
| `rapidhash-64` | 99 | 34/56/9 | 1.12x | 99 | 34/56/9 | 1.12x |
| `rapidhash-128` | 99 | 37/55/7 | 1.14x | 99 | 37/55/7 | 1.14x |
| `pbkdf2-sha512` | 54 | 26/24/4 | 1.15x | 162 | 129/29/4 | 1.31x |
| `xxh3-128` | 99 | 49/39/11 | 1.18x | 99 | 49/39/11 | 1.18x |
| `xxh3-64` | 99 | 52/34/13 | 1.19x | 99 | 52/34/13 | 1.19x |
| `sha256` | 117 | 42/58/17 | 1.21x | 315 | 183/101/31 | 1.49x |
| `hkdf-sha256` | 36 | 31/5/0 | 1.21x | 108 | 103/5/0 | 1.91x |
| `hkdf-sha384` | 36 | 29/7/0 | 1.22x | 108 | 101/7/0 | 1.67x |
| `hmac-sha512` | 99 | 45/47/7 | 1.23x | 297 | 196/89/12 | 1.35x |
| `hmac-sha384` | 99 | 45/47/7 | 1.23x | 297 | 196/89/12 | 1.36x |
| `sha512` | 117 | 46/60/11 | 1.24x | 315 | 193/105/17 | 1.34x |
| `sha384` | 99 | 44/45/10 | 1.24x | 297 | 190/92/15 | 1.35x |
| `ascon-hash256` | 99 | 56/42/1 | 1.27x | 99 | 56/42/1 | 1.27x |
| `sha512-256` | 99 | 54/45/0 | 1.28x | 99 | 54/45/0 | 1.28x |
| `aes-256-gcm` | 198 | 103/33/62 | 1.33x | 594 | 457/46/91 | 2.64x |
| `aes-128-gcm` | 198 | 102/46/50 | 1.33x | 594 | 461/54/79 | 2.58x |
| `ascon-xof128` | 99 | 74/25/0 | 1.35x | 99 | 74/25/0 | 1.35x |
| `xchacha20-poly1305` | 198 | 166/29/3 | 1.38x | 198 | 166/29/3 | 1.38x |
| `blake3` | 432 | 223/176/33 | 1.42x | 432 | 223/176/33 | 1.42x |
| `aegis-256` | 198 | 113/62/23 | 1.46x | 198 | 113/62/23 | 1.46x |
| `aes-128-gcm-siv` | 198 | 131/4/63 | 1.57x | 396 | 307/18/71 | 2.92x |
| `crc32c` | 99 | 49/38/12 | 1.63x | 198 | 147/38/13 | 2.24x |
| `crc32` | 99 | 47/32/20 | 1.63x | 198 | 143/35/20 | 2.25x |
| `aes-256-gcm-siv` | 198 | 131/8/59 | 1.64x | 396 | 328/9/59 | 3.21x |
| `kmac256` | 99 | 60/24/15 | 1.74x | 99 | 60/24/15 | 1.74x |
| `crc64-nvme` | 99 | 53/38/8 | 1.77x | 198 | 136/53/9 | 2.31x |
| `cshake256` | 99 | 62/31/6 | 1.78x | 99 | 62/31/6 | 1.78x |
| `sha224` | 99 | 54/44/1 | 1.87x | 99 | 54/44/1 | 1.87x |
| `sha3-256` | 117 | 108/9/0 | 2.13x | 117 | 108/9/0 | 2.13x |
| `sha3-224` | 99 | 94/5/0 | 2.16x | 99 | 94/5/0 | 2.16x |
| `sha3-384` | 99 | 91/8/0 | 2.19x | 99 | 91/8/0 | 2.19x |
| `sha3-512` | 99 | 92/7/0 | 2.22x | 99 | 92/7/0 | 2.22x |
| `crc64-xz` | 99 | 81/12/6 | 2.50x | 99 | 81/12/6 | 2.50x |
| `shake128` | 99 | 95/4/0 | 2.59x | 99 | 95/4/0 | 2.59x |
| `shake256` | 99 | 94/5/0 | 2.61x | 99 | 94/5/0 | 2.61x |
| `crc24-openpgp` | 99 | 96/0/3 | 13.31x | 99 | 96/0/3 | 13.31x |
| `crc16-ccitt` | 99 | 98/0/1 | 23.62x | 99 | 98/0/1 | 23.62x |
| `crc16-ibm` | 99 | 98/0/1 | 24.27x | 99 | 98/0/1 | 24.27x |

## External Pressure

All-pair Linux CI comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rapidhash` | 396 | 131/207/58 | 33% | 1.10x | 1.01x |
| `xxhash-rust` | 198 | 101/73/24 | 51% | 1.18x | 1.08x |
| `ascon-hash` | 198 | 130/67/1 | 66% | 1.31x | 1.19x |
| `dalek` | 108 | 92/13/3 | 85% | 1.40x | 1.38x |
| `blake3` | 432 | 223/176/33 | 52% | 1.42x | 1.06x |
| `aegis-crate` | 198 | 113/62/23 | 57% | 1.46x | 1.07x |
| `sha2` | 531 | 286/239/6 | 54% | 1.47x | 1.06x |
| `aws-lc-rs` | 1,854 | 1,204/313/337 | 65% | 1.48x | 1.18x |
| `ring` | 1,440 | 1,056/201/183 | 73% | 1.51x | 1.20x |
| `rustcrypto` | 2,988 | 1,948/823/217 | 65% | 1.73x | 1.12x |
| `tiny-keccak` | 198 | 122/55/21 | 62% | 1.76x | 1.99x |
| `dryoc` | 558 | 521/32/5 | 93% | 1.98x | 1.85x |
| `crc-fast` | 297 | 178/101/18 | 60% | 2.02x | 1.17x |
| `sha3` | 612 | 574/38/0 | 94% | 2.30x | 2.15x |
| `crc32fast` | 99 | 80/7/12 | 81% | 2.37x | 1.97x |
| `crc64fast-nvme` | 99 | 78/14/7 | 79% | 2.48x | 1.98x |
| `crc64fast` | 99 | 81/12/6 | 82% | 2.50x | 1.95x |
| `crc32c` | 99 | 90/4/5 | 91% | 2.79x | 2.18x |
| `crc` | 297 | 292/0/5 | 98% | 19.69x | 30.67x |

## README Numbers

- **Headline:** 3,807 of 6,552 matched Linux CI fastest-external comparisons are wins; Linux CI geomean is 1.52x.
- **Checksums:** 5.10x geomean across Linux CI fastest-external checksum comparisons.
- **SHA-3 / SHAKE:** 2.17x SHA-3 geomean and 2.60x SHAKE geomean across Linux CI fastest-external comparisons.
- **BLAKE3:** 2.38x geomean for Linux CI fastest-external rows at `>=64 KiB`.
- **AEAD:** 1.37x geomean across Linux CI fastest-external AEAD comparisons.
- **Ed25519 / X25519:** 1.01x Ed25519 sign geomean, 0.95x Ed25519 verify geomean, and 0.95x X25519 geomean across Linux CI fastest-external comparisons.
- **macOS Apple Silicon:** 1.35x geomean across local fastest-external comparisons; Ed25519 verify is 1.01x.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | ---: | --- |
| AMD Zen4 | `ci` | `2026-05-17 02_51_52` | 2,039 | `benchmark_results/2026-05-17/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-05-17 02_51_52` | 2,039 | `benchmark_results/2026-05-17/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-05-17 02_51_52` | 2,039 | `benchmark_results/2026-05-17/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-05-17 02_51_52` | 2,039 | `benchmark_results/2026-05-17/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-05-17 02_51_52` | 2,039 | `benchmark_results/2026-05-17/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-05-17 02_51_52` | 2,039 | `benchmark_results/2026-05-17/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-05-17 02_51_52` | 2,039 | `benchmark_results/2026-05-17/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-05-17 02_51_52` | 2,039 | `benchmark_results/2026-05-17/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-05-17 02_51_52` | 2,039 | `benchmark_results/2026-05-17/linux/rise-riscv/results.txt` |
| macOS Apple Silicon | `local` | `2026-05-17 15_57_41` | 2,039 | `benchmark_results/2026-05-17/macos/aarch64/results.txt` |
