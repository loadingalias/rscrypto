# Benchmark Overview

Sources:

- Linux benchmark CI run [#25890356700](https://github.com/loadingalias/rscrypto/actions/runs/25890356700), created 2026-05-14 22:55:31 UTC.
- Commit: `538e0bedd3f450da2eb29789521f98cf5b0d552d`.

Scope: current Linux benchmark results for commit `538e0be`: nine dedicated Linux runners from the 2026-05-14 CI run. Ratios are `rscrypto` divided by the external implementation for throughput rows. For latency-only rows, the ratio is external time divided by `rscrypto` time, so higher is still better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and size.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| All matched performance pairs | 10701 | 7086/2444/1171 | 66% | 1.73x | 1.18x |
| Fastest external per case | 6552 | 3719/1945/888 | 57% | 1.50x | 1.09x |
| All matched throughput pairs | 9495 | 6223/2307/965 | 66% | 1.76x | 1.17x |
| Fastest external, throughput only | 5796 | 3245/1839/712 | 56% | 1.54x | 1.08x |
| All matched latency-only pairs | 1206 | 863/137/206 | 72% | 1.48x | 1.25x |
| Fastest external, latency only | 756 | 474/106/176 | 63% | 1.23x | 1.13x |

What matters:

- Linux CI fastest-external scorecard: 3,719 wins / 1,945 ties / 888 losses across 6,552 comparisons, with a 1.50x geomean.
- PBKDF2-SHA256 is no longer the catastrophic broad loss from the prior SPR run, but it is not clean yet: 0.95x fastest-external geomean, 1.03x median, with `iters=1` still at 0.77x.
- HMAC-SHA256 recovered as a broad claim: 1.12x fastest-external geomean overall and 1.19x for streaming. The one-shot `hash` shape is still noisy at 0.99x median.
- SHA-256 one-shot remains healthy at 1.21x, while SHA-256 streaming is the clearest SHA-2 issue at 0.69x fastest-external geomean.
- Small-parameter password hashing is the broadest remaining loss cluster: `scrypt-small` and the Argon2 small lanes are below parity against RustCrypto on several Linux hosts.
- Checksums, SHA-3/SHAKE, BLAKE3 at large sizes, AES-GCM-SIV, and XChaCha20-Poly1305 remain the strongest broad wins.

## Clear Losses

Collapsed fastest-external groups. This is the useful root-cause view, not just the worst individual rows.

| Place | Fastest rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `scrypt-small` | 36 | 19/1/16 | 0.76x | 1.08x | `rustcrypto` 36 |
| `argon2id-small` | 27 | 6/4/17 | 0.81x | 0.82x | `rustcrypto` 27 |
| `argon2d-small` | 27 | 6/4/17 | 0.82x | 0.86x | `rustcrypto` 27 |
| `argon2i-small` | 27 | 6/4/17 | 0.82x | 0.79x | `rustcrypto` 27 |
| `pbkdf2-sha256` | 54 | 23/18/13 | 0.95x | 1.03x | `rustcrypto` 42, `aws-lc-rs` 12 |
| `x25519` | 18 | 6/8/4 | 0.95x | 1.01x | `aws-lc-rs` 14, `dryoc` 2, `dalek` 2 |

Top measured primitive/operation losses, fastest external only, excluding tiny sample groups (`rows < 10`):

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `sha256` / `streaming` | 18 | 2/10/6 | 0.69x | 1.00x | `sha2` 18 |
| `scrypt-small` / `hash` | 36 | 19/1/16 | 0.76x | 1.08x | `rustcrypto` 36 |
| `pbkdf2-sha256` / `iters=1` | 18 | 9/6/3 | 0.77x | 1.05x | `rustcrypto` 14, `aws-lc-rs` 4 |
| `argon2id-small` / `hash` | 27 | 6/4/17 | 0.81x | 0.82x | `rustcrypto` 27 |
| `argon2d-small` / `hash` | 27 | 6/4/17 | 0.82x | 0.86x | `rustcrypto` 27 |
| `argon2i-small` / `hash` | 27 | 6/4/17 | 0.82x | 0.79x | `rustcrypto` 27 |
| `ed25519` / `verify` | 36 | 10/6/20 | 0.92x | 0.94x | `dalek` 17, `ring` 10, `aws-lc-rs` 9 |

Small-sample losses are listed separately because they should not dominate release triage:

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `scrypt-owasp` / `hash` | 9 | 4/1/4 | 0.79x | 0.98x | `rustcrypto` 9 |
| `x25519` / `diffie-hellman` | 9 | 3/4/2 | 0.92x | 1.01x | `aws-lc-rs` 7, `dryoc` 1, `dalek` 1 |
| `argon2id-owasp` / `hash` | 9 | 3/2/4 | 0.98x | 0.96x | `rustcrypto` 9 |
| `x25519` / `public-key-from-secret` | 9 | 3/4/2 | 0.98x | 1.01x | `aws-lc-rs` 7, `dryoc` 1, `dalek` 1 |

## Non-Crypto Hash Truth Split

Same-algorithm rows answer whether the strict implementation is competitive with its reference crate. Fastest-class rows answer whether the strict RapidHash/XXH3 families can beat the fastest external non-crypto hash in their class.

Same-algorithm comparisons:

| Comparison | Rows | W/T/L | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `xxh3-64` vs `xxhash-rust` | 99 | 51/35/13 | 1.18x | 1.06x |
| `xxh3-128` vs `xxhash-rust` | 99 | 51/36/12 | 1.18x | 1.05x |
| `rapidhash-64` vs `rapidhash` | 99 | 34/56/9 | 1.12x | 1.01x |
| `rapidhash-128` vs `rapidhash` | 99 | 38/54/7 | 1.14x | 1.02x |
| `rapidhash-v3-64` vs `rapidhash` | 99 | 27/50/22 | 1.05x | 1.00x |
| `rapidhash-v3-128` vs `rapidhash` | 99 | 32/44/23 | 1.07x | 1.00x |

Fastest-class comparisons:

| Comparison | Rows | W/T/L | All-platform geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| 64-bit RapidHash/XXH3 fastest-class | 297 | 112/141/44 | 1.12x | 1.01x | `rapidhash` 198, `xxhash-rust` 99 |
| 128-bit RapidHash/XXH3 fastest-class | 297 | 121/134/42 | 1.13x | 1.02x | `rapidhash` 198, `xxhash-rust` 99 |

## Sustained AEAD

Fastest-external throughput comparisons only, sizes `65536`, `262144`, and `1048576`, both encrypt and decrypt, 128-bit and 256-bit keys where applicable.

| Platform | AES-GCM W/T/L | AES-GCM geomean | GCM-SIV W/T/L | GCM-SIV geomean | ChaCha20-Poly1305 W/T/L | ChaCha20-Poly1305 geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 0/12/0 | 1.00x | 12/0/0 | 1.75x | 0/0/6 | 0.93x |
| AMD Zen5 | 1/7/4 | 0.98x | 12/0/0 | 2.05x | 6/0/0 | 1.17x |
| AWS Graviton3 | 0/0/12 | 0.91x | 12/0/0 | 2.53x | 0/6/0 | 1.00x |
| AWS Graviton4 | 0/11/1 | 0.97x | 12/0/0 | 2.53x | 0/6/0 | 1.00x |
| IBM Power10 | 0/0/12 | 0.37x | 6/0/6 | 1.46x | 5/1/0 | 1.12x |
| IBM z16/s390x | 12/0/0 | 12.21x | 12/0/0 | 5.55x | 6/0/0 | 1.90x |
| Intel Ice Lake | 2/10/0 | 1.01x | 12/0/0 | 1.34x | 5/1/0 | 1.05x |
| Intel Sapphire Rapids | 1/8/3 | 0.97x | 12/0/0 | 1.50x | 5/1/0 | 1.06x |
| RISE RISC-V | 0/0/12 | 0.82x | 0/0/12 | 0.91x | 0/0/6 | 0.93x |

Primitive-level sustained AEAD geomeans:

| Platform | AES-128-GCM | AES-256-GCM | AES-128-GCM-SIV | AES-256-GCM-SIV | ChaCha20-Poly1305 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 1.01x | 0.99x | 1.78x | 1.71x | 0.93x |
| AMD Zen5 | 1.02x | 0.94x | 2.06x | 2.05x | 1.17x |
| AWS Graviton3 | 0.90x | 0.91x | 2.40x | 2.67x | 1.00x |
| AWS Graviton4 | 0.97x | 0.97x | 2.38x | 2.70x | 1.00x |
| IBM Power10 | 0.37x | 0.37x | 1.43x | 1.48x | 1.12x |
| IBM z16/s390x | 11.94x | 12.49x | 5.25x | 5.88x | 1.90x |
| Intel Ice Lake | 1.03x | 0.99x | 1.32x | 1.36x | 1.05x |
| Intel Sapphire Rapids | 0.98x | 0.96x | 1.47x | 1.52x | 1.06x |
| RISE RISC-V | 0.81x | 0.83x | 0.91x | 0.91x | 0.93x |

## Platform Summary

Fastest-external comparisons across every matched primitive, operation, and size.

| Platform | Rows | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 728 | 471/175/82 | 65% | 1.44x | 1.09x |
| AMD Zen5 | 728 | 374/267/87 | 51% | 1.45x | 1.05x |
| AWS Graviton3 | 728 | 346/271/111 | 48% | 1.37x | 1.04x |
| AWS Graviton4 | 728 | 339/304/85 | 47% | 1.34x | 1.04x |
| IBM Power10 | 728 | 303/318/107 | 42% | 1.42x | 1.02x |
| IBM z16/s390x | 728 | 591/90/47 | 81% | 2.92x | 2.03x |
| Intel Ice Lake | 728 | 492/134/102 | 68% | 1.46x | 1.17x |
| Intel Sapphire Rapids | 728 | 485/139/104 | 67% | 1.56x | 1.14x |
| RISE RISC-V | 728 | 318/247/163 | 44% | 1.10x | 1.02x |

## Primitive Summary

All primitives with matched `rscrypto` comparisons. The fastest-only columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `scrypt-small` | 36 | 19/1/16 | 0.76x | 36 | 19/1/16 | 0.76x |
| `scrypt-owasp` | 9 | 4/1/4 | 0.79x | 9 | 4/1/4 | 0.79x |
| `argon2id-small` | 27 | 6/4/17 | 0.81x | 45 | 23/4/18 | 1.22x |
| `argon2d-small` | 27 | 6/4/17 | 0.82x | 27 | 6/4/17 | 0.82x |
| `argon2i-small` | 27 | 6/4/17 | 0.82x | 45 | 23/4/18 | 1.21x |
| `pbkdf2-sha256` | 54 | 23/18/13 | 0.95x | 162 | 124/22/16 | 1.32x |
| `x25519` | 18 | 6/8/4 | 0.95x | 54 | 42/8/4 | 1.40x |
| `argon2id-owasp` | 9 | 3/2/4 | 0.98x | 18 | 8/5/5 | 1.27x |
| `chacha20-poly1305` | 198 | 72/53/73 | 0.98x | 594 | 347/127/120 | 1.18x |
| `hmac-sha512` | 99 | 15/28/56 | 0.99x | 297 | 107/88/102 | 1.14x |
| `hmac-sha384` | 99 | 16/26/57 | 0.99x | 297 | 110/84/103 | 1.14x |
| `rapidhash-v3-64` | 99 | 27/50/22 | 1.05x | 99 | 27/50/22 | 1.05x |
| `blake2/blake2b256` | 387 | 194/180/13 | 1.06x | 612 | 408/191/13 | 1.36x |
| `ed25519` | 90 | 43/9/38 | 1.06x | 306 | 221/25/60 | 1.30x |
| `rapidhash-v3-128` | 99 | 32/44/23 | 1.07x | 99 | 32/44/23 | 1.07x |
| `blake2/blake2b512` | 198 | 110/87/1 | 1.07x | 396 | 303/92/1 | 1.53x |
| `blake2/blake2s256` | 387 | 198/179/10 | 1.10x | 387 | 198/179/10 | 1.10x |
| `sha256` | 117 | 42/56/19 | 1.11x | 315 | 183/100/32 | 1.45x |
| `blake2/blake2s128` | 198 | 119/79/0 | 1.12x | 198 | 119/79/0 | 1.12x |
| `hmac-sha256` | 117 | 21/55/41 | 1.12x | 315 | 129/124/62 | 1.32x |
| `rapidhash-64` | 99 | 34/56/9 | 1.12x | 99 | 34/56/9 | 1.12x |
| `rapidhash-128` | 99 | 38/54/7 | 1.14x | 99 | 38/54/7 | 1.14x |
| `pbkdf2-sha512` | 54 | 25/25/4 | 1.15x | 162 | 127/31/4 | 1.31x |
| `xxh3-128` | 99 | 51/36/12 | 1.18x | 99 | 51/36/12 | 1.18x |
| `xxh3-64` | 99 | 51/35/13 | 1.18x | 99 | 51/35/13 | 1.18x |
| `hkdf-sha256` | 36 | 31/5/0 | 1.21x | 108 | 103/5/0 | 1.92x |
| `hkdf-sha384` | 36 | 25/11/0 | 1.22x | 108 | 97/11/0 | 1.67x |
| `sha512` | 117 | 47/59/11 | 1.24x | 315 | 191/107/17 | 1.34x |
| `sha384` | 99 | 39/49/11 | 1.24x | 297 | 186/94/17 | 1.35x |
| `ascon-hash256` | 99 | 56/43/0 | 1.27x | 99 | 56/43/0 | 1.27x |
| `sha512-256` | 99 | 57/42/0 | 1.28x | 99 | 57/42/0 | 1.28x |
| `aes-256-gcm` | 198 | 98/40/60 | 1.32x | 594 | 453/53/88 | 2.62x |
| `aes-128-gcm` | 198 | 103/45/50 | 1.33x | 594 | 462/56/76 | 2.57x |
| `ascon-xof128` | 99 | 74/25/0 | 1.35x | 99 | 74/25/0 | 1.35x |
| `xchacha20-poly1305` | 198 | 172/23/3 | 1.38x | 198 | 172/23/3 | 1.38x |
| `blake3` | 432 | 224/170/38 | 1.43x | 432 | 224/170/38 | 1.43x |
| `aegis-256` | 198 | 107/69/22 | 1.45x | 198 | 107/69/22 | 1.45x |
| `aes-128-gcm-siv` | 198 | 132/3/63 | 1.57x | 396 | 307/18/71 | 2.93x |
| `crc32` | 99 | 46/37/16 | 1.63x | 198 | 142/39/17 | 2.22x |
| `crc32c` | 99 | 51/39/9 | 1.64x | 198 | 149/40/9 | 2.31x |
| `aes-256-gcm-siv` | 198 | 132/6/60 | 1.64x | 396 | 329/7/60 | 3.22x |
| `crc64-nvme` | 99 | 50/33/16 | 1.67x | 198 | 129/46/23 | 2.15x |
| `kmac256` | 99 | 60/27/12 | 1.74x | 99 | 60/27/12 | 1.74x |
| `cshake256` | 99 | 62/28/9 | 1.77x | 99 | 62/28/9 | 1.77x |
| `sha224` | 99 | 54/45/0 | 1.87x | 99 | 54/45/0 | 1.87x |
| `sha3-256` | 117 | 108/9/0 | 2.13x | 117 | 108/9/0 | 2.13x |
| `sha3-224` | 99 | 93/6/0 | 2.16x | 99 | 93/6/0 | 2.16x |
| `sha3-384` | 99 | 92/7/0 | 2.19x | 99 | 92/7/0 | 2.19x |
| `sha3-512` | 99 | 92/7/0 | 2.22x | 99 | 92/7/0 | 2.22x |
| `crc64-xz` | 99 | 75/13/11 | 2.29x | 99 | 75/13/11 | 2.29x |
| `shake128` | 99 | 95/4/0 | 2.59x | 99 | 95/4/0 | 2.59x |
| `shake256` | 99 | 94/5/0 | 2.60x | 99 | 94/5/0 | 2.60x |
| `crc24-openpgp` | 99 | 96/0/3 | 12.38x | 99 | 96/0/3 | 12.38x |
| `crc16-ccitt` | 99 | 96/1/2 | 22.64x | 99 | 96/1/2 | 22.64x |
| `crc16-ibm` | 99 | 97/0/2 | 23.19x | 99 | 97/0/2 | 23.19x |

## External Pressure

All-pair comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rapidhash` | 396 | 131/204/61 | 33% | 1.09x | 1.01x |
| `xxhash-rust` | 198 | 102/71/25 | 52% | 1.18x | 1.06x |
| `ascon-hash` | 198 | 130/68/0 | 66% | 1.31x | 1.18x |
| `dalek` | 108 | 90/11/7 | 83% | 1.40x | 1.38x |
| `blake3` | 432 | 224/170/38 | 52% | 1.43x | 1.06x |
| `aws-lc-rs` | 1854 | 1113/306/435 | 60% | 1.43x | 1.13x |
| `sha2` | 531 | 281/244/6 | 53% | 1.44x | 1.05x |
| `aegis-crate` | 198 | 107/69/22 | 54% | 1.45x | 1.06x |
| `ring` | 1440 | 979/194/267 | 68% | 1.45x | 1.18x |
| `rustcrypto` | 2988 | 1929/846/213 | 65% | 1.73x | 1.11x |
| `tiny-keccak` | 198 | 122/55/21 | 62% | 1.76x | 1.83x |
| `crc-fast` | 297 | 176/100/21 | 59% | 1.96x | 1.13x |
| `dryoc` | 558 | 520/29/9 | 93% | 1.99x | 1.85x |
| `crc64fast-nvme` | 99 | 72/12/15 | 73% | 2.28x | 1.88x |
| `crc64fast` | 99 | 75/13/11 | 76% | 2.29x | 1.91x |
| `sha3` | 612 | 574/38/0 | 94% | 2.30x | 2.16x |
| `crc32fast` | 99 | 81/8/10 | 82% | 2.37x | 1.98x |
| `crc32c` | 99 | 91/5/3 | 92% | 2.97x | 2.20x |
| `crc` | 297 | 289/1/7 | 97% | 18.66x | 27.66x |

## README Numbers

- **Headline:** 3,719 of 6,552 matched Linux CI fastest-external comparisons are wins; Linux CI geomean is 1.50x.
- **Checksums:** 4.88x geomean across Linux CI fastest-external checksum comparisons.
- **SHA-3 / SHAKE:** 2.17x SHA-3 geomean and 2.13x SHAKE/cSHAKE/KMAC geomean across Linux CI fastest-external comparisons.
- **BLAKE3:** 2.29x geomean for Linux CI fastest-external rows at `>=64 KiB`.
- **AEAD:** 1.37x geomean across Linux CI fastest-external AEAD comparisons.
- **Ed25519 / X25519:** 1.01x Ed25519 sign geomean and 0.95x X25519 geomean across Linux CI fastest-external comparisons.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | ---: | --- |
| AMD Zen4 | `ci` | `2026-05-14 22_55_31` | 2039 | `benchmark_results/2026-05-14/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-05-14 22_55_31` | 2039 | `benchmark_results/2026-05-14/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-05-14 22_55_31` | 2039 | `benchmark_results/2026-05-14/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-05-14 22_55_31` | 2039 | `benchmark_results/2026-05-14/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-05-14 22_55_31` | 2039 | `benchmark_results/2026-05-14/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-05-14 22_55_31` | 2039 | `benchmark_results/2026-05-14/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-05-14 22_55_31` | 2039 | `benchmark_results/2026-05-14/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-05-14 22_55_31` | 2039 | `benchmark_results/2026-05-14/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-05-14 22_55_31` | 2039 | `benchmark_results/2026-05-14/linux/rise-riscv/results.txt` |

## Methodology Notes

- Parsed 18,351 Criterion rows: 16,281 throughput rows and 2,070 latency-only rows.
- Matched 10,701 `rscrypto` external pairs across 6,552 fastest-external cases.
- Input files include 9 Linux CI result files. CI result headers use `time=22_55_31`, the GitHub Actions run creation time, per extraction policy.
- Matching is structural: a comparison exists when an external Criterion ID can be produced by replacing the exact `rscrypto` path component while keeping the same platform, primitive path, operation path, and size suffix.
- When an external crate name also appears as a primitive path component, such as `blake3/blake3` or `crc32c/crc32c`, the implementation component is selected by the matching `rscrypto` key rather than by token name alone.
- Diagnostic implementation labels such as `rscrypto-oneshot`, `rscrypto-state`, and `rscrypto-stream-new` are parsed as raw rows but are not counted as public `rscrypto` comparison rows because they are not the exact `rscrypto` implementation component.
- Unmatched external-only rows: 216. These are internal diagnostic comparisons without a corresponding exact `rscrypto` row, so they are not part of the ratio tables.
