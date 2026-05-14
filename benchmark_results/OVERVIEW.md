# Benchmark Overview

Sources:

- Linux benchmark CI run [#25813605247](https://github.com/loadingalias/rscrypto/actions/runs/25813605247), created 2026-05-13 16:53:23 UTC.
- Local macOS aarch64 bench file: `benchmark_results/2026-05-13/macos/aarch64/results.txt`.
- Commit: `7609841719881f8387f8453fe51581e42ffecbf1`.

Scope: current 2026-05-13 global benchmark results: nine Linux dedicated runners plus one local macOS aarch64 run. Ratios are `rscrypto` divided by the external implementation. For throughput rows, higher throughput is better. For latency-only rows, the ratio is external time divided by `rscrypto` time, so higher is still better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and size.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| All matched performance pairs | 12926 | 8043/2733/2150 | 62% | 1.58x | 1.16x |
| Fastest external per case | 6843 | 3607/1827/1409 | 53% | 1.37x | 1.06x |
| All matched throughput pairs | 11680 | 7107/2596/1977 | 61% | 1.59x | 1.14x |
| Fastest external, throughput only | 6190 | 3174/1745/1271 | 51% | 1.38x | 1.05x |
| All matched latency-only pairs | 1246 | 936/137/173 | 75% | 1.51x | 1.30x |
| Fastest external, latency only | 653 | 433/82/138 | 66% | 1.22x | 1.15x |

What matters:

- Same-algorithm XXH3 and RapidHash are now broadly positive, but strict non-crypto hash families still lose hard against GXHash-class fastest-external pressure: 128-bit RapidHash/XXH3 is 0.46x geomean and 64-bit RapidHash/XXH3 is 0.66x geomean.
- The new AES-backed 64-bit fast hash does clear the x86/aarch64 AES class view (1.13x geomean), but the 128-bit class path does not (0.82x geomean) because short x86 rows against `gxhash` are still too expensive.
- ChaCha20-Poly1305 improved out of the catastrophic bucket: all-size fastest-external geomean is 0.91x and sustained throughput is 0.97x. It is not a clean win yet.
- Outside the original three pressure areas, PBKDF2-SHA256 (0.70x) and X25519 (0.74x) are the next broad losses.

## Clear Losses

Collapsed fastest-external groups. This is the useful root-cause view, not just the worst individual rows.

| Place | Fastest rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| Strict 128-bit RapidHash/XXH3 | 330 | 60/44/226 | 0.46x | 0.35x | `gxhash` 231, `rapidhash` 66, `xxhash-rust` 33 |
| Argon2 small password hashes | 9 | 0/5/4 | 0.60x | 0.96x | `rustcrypto` 9 |
| Strict 64-bit RapidHash/XXH3 | 330 | 43/33/254 | 0.66x | 0.64x | `gxhash` 208, `foldhash` 61, `rapidhash` 27, `ahash` 25 |
| AES-backed fast hash class, all platforms | 220 | 110/25/85 | 0.70x | 1.04x | `gxhash` 151, `xxhash-rust` 28, `rapidhash` 23, `foldhash` 17 |
| PBKDF2-SHA256 | 60 | 14/29/17 | 0.70x | 1.00x | `rustcrypto` 49, `aws-lc-rs` 11 |
| X25519 | 20 | 5/1/14 | 0.74x | 0.64x | `aws-lc-rs` 16, `dryoc` 3, `dalek` 1 |
| HMAC-SHA256 | 130 | 30/48/52 | 0.86x | 0.99x | `ring` 64, `rustcrypto` 41, `aws-lc-rs` 25 |
| ChaCha20-Poly1305 | 220 | 63/44/113 | 0.91x | 0.95x | `ring` 121, `aws-lc-rs` 84, `rustcrypto` 15 |

Top measured primitive/operation losses, fastest external only, excluding tiny sample groups (`rows < 10`):

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `rapidhash-v3-128` / `hash` | 110 | 10/18/82 | 0.38x | 0.27x | `gxhash` 77, `rapidhash` 33 |
| `rapidhash-128` / `hash` | 110 | 17/16/77 | 0.42x | 0.28x | `gxhash` 77, `rapidhash` 33 |
| `xxh3-128` / `hash` | 110 | 33/10/67 | 0.61x | 0.70x | `gxhash` 77, `xxhash-rust` 33 |
| `rapidhash-v3-64` / `hash` | 110 | 4/9/97 | 0.62x | 0.60x | `gxhash` 69, `foldhash` 21, `rapidhash` 11, `ahash` 9 |
| `x25519` / `diffie-hellman` | 10 | 2/0/8 | 0.65x | 0.54x | `aws-lc-rs` 8, `dryoc` 2 |
| `fast-hash-128` / `hash` | 110 | 55/4/51 | 0.66x | 1.04x | `gxhash` 77, `xxhash-rust` 23, `rapidhash` 10 |
| `rapidhash-64` / `hash` | 110 | 18/13/79 | 0.67x | 0.63x | `gxhash` 70, `rapidhash` 16, `foldhash` 16, `ahash` 8 |
| `xxh3-64` / `hash` | 110 | 21/11/78 | 0.68x | 0.70x | `gxhash` 69, `foldhash` 24, `xxhash-rust` 9, `ahash` 8 |
| `pbkdf2-sha256` / `iters=1000` | 20 | 4/10/6 | 0.70x | 1.00x | `rustcrypto` 17, `aws-lc-rs` 3 |
| `pbkdf2-sha256` / `iters=100` | 20 | 4/10/6 | 0.70x | 1.00x | `rustcrypto` 16, `aws-lc-rs` 4 |
| `pbkdf2-sha256` / `iters=1` | 20 | 6/9/5 | 0.71x | 1.01x | `rustcrypto` 16, `aws-lc-rs` 4 |
| `sha256` / `streaming` | 20 | 2/12/6 | 0.72x | 0.99x | `sha2` 20 |
| `hmac-sha256` / `streaming` | 20 | 2/12/6 | 0.72x | 0.99x | `rustcrypto` 20 |
| `fast-hash-64` / `hash` | 110 | 55/21/34 | 0.73x | 1.04x | `gxhash` 74, `foldhash` 17, `rapidhash` 13, `xxhash-rust` 5 |
| `x25519` / `public-key-from-secret` | 10 | 3/1/6 | 0.85x | 0.86x | `aws-lc-rs` 8, `dryoc` 1, `dalek` 1 |
| `hmac-sha256` / `hash` | 110 | 28/36/46 | 0.89x | 0.99x | `ring` 64, `aws-lc-rs` 25, `rustcrypto` 21 |
| `ed25519` / `verify` | 40 | 10/8/22 | 0.90x | 0.93x | `dalek` 16, `aws-lc-rs` 13, `ring` 9, `dryoc` 2 |
| `chacha20-poly1305` / `encrypt` | 110 | 31/23/56 | 0.91x | 0.94x | `ring` 75, `aws-lc-rs` 26, `rustcrypto` 9 |

Small-sample losses are listed separately because they should not dominate release triage:

| Primitive/op | Rows | W/T/L | Geomean | Median | Main pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| `argon2d-small` / `hash` | 3 | 0/2/1 | 0.60x | 0.98x | `rustcrypto` 3 |
| `argon2id-small` / `hash` | 3 | 0/2/1 | 0.60x | 0.96x | `rustcrypto` 3 |
| `argon2i-small` / `hash` | 3 | 0/1/2 | 0.61x | 0.94x | `rustcrypto` 3 |

## Non-Crypto Hash Truth Split

Same-algorithm rows answer whether the strict implementation is competitive with its reference crate. Fastest-class rows answer whether the library has a primitive that can beat the fastest non-crypto hash class on the platform.

Same-algorithm comparisons:

| Comparison | Rows | W/T/L | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| `xxh3-64` vs `xxhash-rust` | 110 | 58/35/17 | 1.18x | 1.06x |
| `xxh3-128` vs `xxhash-rust` | 110 | 56/42/12 | 1.18x | 1.06x |
| `rapidhash-64` vs `rapidhash` | 110 | 37/60/13 | 1.13x | 1.00x |
| `rapidhash-128` vs `rapidhash` | 110 | 41/62/7 | 1.16x | 1.01x |
| `rapidhash-v3-64` vs `rapidhash` | 110 | 27/62/21 | 1.05x | 1.00x |
| `rapidhash-v3-128` vs `rapidhash` | 110 | 35/53/22 | 1.08x | 1.00x |

Fastest-class comparisons:

| Comparison | Rows | W/T/L | All-platform geomean | Median | x86/aarch64 AES geomean | x86/aarch64 AES >=256B geomean | Main pressure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fast-hash-64` fastest-class | 110 | 55/21/34 | 0.73x | 1.04x | 1.13x | 1.19x | `gxhash` 74, `foldhash` 17, `rapidhash` 13, `xxhash-rust` 5 |
| `fast-hash-128` fastest-class | 110 | 55/4/51 | 0.66x | 1.04x | 0.82x | 1.15x | `gxhash` 77, `xxhash-rust` 23, `rapidhash` 10 |

## Sustained AEAD

Fastest-external throughput comparisons only, sizes `65536`, `262144`, and `1048576`, both encrypt and decrypt, 128-bit and 256-bit keys where applicable.

| Platform | AES-GCM W/T/L | AES-GCM geomean | GCM-SIV W/T/L | GCM-SIV geomean | ChaCha20-Poly1305 W/T/L | ChaCha20-Poly1305 geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 0/12/0 | 1.00x | 12/0/0 | 1.75x | 0/0/6 | 0.91x |
| AMD Zen5 | 0/8/4 | 0.98x | 12/0/0 | 2.05x | 6/0/0 | 1.15x |
| Intel Ice Lake | 3/7/2 | 1.01x | 12/0/0 | 1.35x | 1/5/0 | 1.03x |
| Intel Sapphire Rapids | 1/9/2 | 0.99x | 12/0/0 | 1.49x | 4/2/0 | 1.07x |
| AWS Graviton3 | 0/0/12 | 0.63x | 12/0/0 | 2.54x | 0/0/6 | 0.52x |
| AWS Graviton4 | 0/0/12 | 0.64x | 12/0/0 | 2.53x | 0/0/6 | 0.56x |
| IBM Power10 | 0/0/12 | 0.36x | 6/0/6 | 1.45x | 6/0/0 | 1.14x |
| IBM z16/s390x | 12/0/0 | 12.18x | 12/0/0 | 5.65x | 6/0/0 | 1.97x |
| RISE RISC-V | 0/0/12 | 0.82x | 0/0/12 | 0.90x | 0/0/6 | 0.92x |
| macOS aarch64 | 0/12/0 | 1.01x | 12/0/0 | 2.79x | 0/6/0 | 1.00x |

Primitive-level sustained AEAD geomeans:

| Platform | AES-128-GCM | AES-256-GCM | AES-128-GCM-SIV | AES-256-GCM-SIV | ChaCha20-Poly1305 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 1.01x | 1.00x | 1.78x | 1.72x | 0.91x |
| AMD Zen5 | 1.02x | 0.94x | 2.05x | 2.05x | 1.15x |
| Intel Ice Lake | 1.03x | 0.99x | 1.33x | 1.37x | 1.03x |
| Intel Sapphire Rapids | 0.98x | 1.00x | 1.46x | 1.52x | 1.07x |
| AWS Graviton3 | 0.64x | 0.62x | 2.41x | 2.68x | 0.52x |
| AWS Graviton4 | 0.66x | 0.62x | 2.38x | 2.69x | 0.56x |
| IBM Power10 | 0.36x | 0.37x | 1.43x | 1.48x | 1.14x |
| IBM z16/s390x | 11.28x | 13.14x | 5.33x | 5.98x | 1.97x |
| RISE RISC-V | 0.78x | 0.85x | 0.88x | 0.91x | 0.92x |
| macOS aarch64 | 1.01x | 1.00x | 2.71x | 2.87x | 1.00x |

## Platform Summary

Fastest-external comparisons across every matched primitive, operation, and size.

| Platform | Rows | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD Zen4 | 677 | 401/138/138 | 59% | 1.29x | 1.08x |
| AMD Zen5 | 677 | 346/199/132 | 51% | 1.31x | 1.06x |
| Intel Ice Lake | 677 | 401/110/166 | 59% | 1.30x | 1.13x |
| Intel Sapphire Rapids | 677 | 398/121/158 | 59% | 1.28x | 1.10x |
| AWS Graviton3 | 677 | 324/188/165 | 48% | 1.25x | 1.03x |
| AWS Graviton4 | 677 | 326/191/160 | 48% | 1.27x | 1.05x |
| IBM Power10 | 677 | 297/277/103 | 44% | 1.39x | 1.03x |
| IBM z16/s390x | 677 | 537/70/70 | 79% | 2.84x | 2.17x |
| RISE RISC-V | 677 | 262/236/179 | 39% | 1.07x | 1.01x |
| macOS aarch64 | 750 | 315/297/138 | 42% | 1.24x | 1.03x |

## Primitive Summary

All primitives with matched `rscrypto` comparisons. The fastest-only columns are the release-claim view; all-pair columns show total competitor pressure.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rapidhash-v3-128` | 110 | 10/18/82 | 0.38x | 187 | 35/53/99 | 0.58x |
| `rapidhash-128` | 110 | 17/16/77 | 0.42x | 187 | 46/63/78 | 0.64x |
| `argon2d-small` | 3 | 0/2/1 | 0.60x | 3 | 0/2/1 | 0.60x |
| `argon2id-small` | 3 | 0/2/1 | 0.60x | 5 | 2/2/1 | 1.03x |
| `argon2i-small` | 3 | 0/1/2 | 0.61x | 5 | 2/1/2 | 1.02x |
| `xxh3-128` | 110 | 33/10/67 | 0.61x | 187 | 67/42/78 | 0.78x |
| `rapidhash-v3-64` | 110 | 4/9/97 | 0.62x | 407 | 128/100/179 | 0.97x |
| `fast-hash-128` | 110 | 55/4/51 | 0.66x | 297 | 215/11/71 | 1.36x |
| `rapidhash-64` | 110 | 18/13/79 | 0.67x | 407 | 164/102/141 | 1.05x |
| `xxh3-64` | 110 | 21/11/78 | 0.68x | 407 | 206/47/154 | 1.07x |
| `pbkdf2-sha256` | 60 | 14/29/17 | 0.70x | 180 | 123/32/25 | 0.99x |
| `fast-hash-64` | 110 | 55/21/34 | 0.73x | 517 | 376/35/106 | 1.29x |
| `x25519` | 20 | 5/1/14 | 0.74x | 60 | 42/4/14 | 1.12x |
| `hmac-sha256` | 130 | 30/48/52 | 0.86x | 350 | 146/118/86 | 1.04x |
| `chacha20-poly1305` | 220 | 63/44/113 | 0.91x | 660 | 352/108/200 | 1.11x |
| `hmac-sha512` | 110 | 16/32/62 | 0.98x | 330 | 110/103/117 | 1.11x |
| `hmac-sha384` | 110 | 15/33/62 | 0.98x | 330 | 118/99/113 | 1.12x |
| `argon2id-owasp` | 1 | 0/1/0 | 1.01x | 2 | 1/1/0 | 1.27x |
| `ed25519` | 100 | 46/10/44 | 1.03x | 340 | 245/23/72 | 1.28x |
| `blake2/blake2b256` | 268 | 109/153/6 | 1.05x | 419 | 253/160/6 | 1.33x |
| `blake2/blake2b512` | 220 | 111/107/2 | 1.06x | 341 | 229/110/2 | 1.37x |
| `scrypt-owasp` | 1 | 1/0/0 | 1.07x | 1 | 1/0/0 | 1.07x |
| `blake2/blake2s256` | 268 | 123/143/2 | 1.08x | 268 | 123/143/2 | 1.08x |
| `sha256` | 130 | 41/64/25 | 1.10x | 350 | 188/118/44 | 1.41x |
| `blake2/blake2s128` | 220 | 123/97/0 | 1.11x | 220 | 123/97/0 | 1.11x |
| `pbkdf2-sha512` | 60 | 23/27/10 | 1.12x | 180 | 133/33/14 | 1.27x |
| `scrypt-small` | 4 | 4/0/0 | 1.13x | 4 | 4/0/0 | 1.13x |
| `hkdf-sha256` | 40 | 31/7/2 | 1.17x | 120 | 111/7/2 | 1.85x |
| `kmac256` | 11 | 11/0/0 | 1.20x | 11 | 11/0/0 | 1.20x |
| `hkdf-sha384` | 40 | 26/14/0 | 1.21x | 120 | 106/14/0 | 1.63x |
| `sha512` | 130 | 46/73/11 | 1.22x | 350 | 198/135/17 | 1.31x |
| `sha384` | 110 | 39/61/10 | 1.23x | 330 | 191/123/16 | 1.33x |
| `ascon-hash256` | 110 | 60/48/2 | 1.24x | 110 | 60/48/2 | 1.24x |
| `cshake256` | 11 | 11/0/0 | 1.25x | 11 | 11/0/0 | 1.25x |
| `aes-256-gcm` | 220 | 103/40/77 | 1.25x | 660 | 479/58/123 | 2.61x |
| `sha512-256` | 110 | 55/53/2 | 1.25x | 110 | 55/53/2 | 1.25x |
| `aes-128-gcm` | 220 | 111/42/67 | 1.27x | 660 | 495/59/106 | 2.58x |
| `ascon-xof128` | 110 | 85/24/1 | 1.33x | 110 | 85/24/1 | 1.33x |
| `blake3` | 480 | 244/209/27 | 1.39x | 480 | 244/209/27 | 1.39x |
| `aegis-256` | 220 | 126/73/21 | 1.41x | 220 | 126/73/21 | 1.41x |
| `xchacha20-poly1305` | 220 | 188/30/2 | 1.42x | 220 | 188/30/2 | 1.42x |
| `crc32c` | 110 | 56/40/14 | 1.57x | 220 | 165/41/14 | 2.19x |
| `crc32` | 110 | 52/41/17 | 1.60x | 220 | 158/45/17 | 2.25x |
| `crc64-nvme` | 110 | 61/34/15 | 1.63x | 220 | 146/50/24 | 2.10x |
| `aes-128-gcm-siv` | 220 | 154/2/64 | 1.66x | 440 | 352/16/72 | 3.24x |
| `aes-256-gcm-siv` | 220 | 153/6/61 | 1.73x | 440 | 372/7/61 | 3.57x |
| `sha224` | 110 | 55/52/3 | 1.76x | 110 | 55/52/3 | 1.76x |
| `sha3-256` | 130 | 112/15/3 | 1.98x | 130 | 112/15/3 | 1.98x |
| `sha3-224` | 110 | 96/13/1 | 2.01x | 110 | 96/13/1 | 2.01x |
| `sha3-384` | 110 | 95/12/3 | 2.03x | 110 | 95/12/3 | 2.03x |
| `sha3-512` | 110 | 97/9/4 | 2.05x | 110 | 97/9/4 | 2.05x |
| `crc64-xz` | 110 | 83/14/13 | 2.16x | 110 | 83/14/13 | 2.16x |
| `shake128` | 110 | 99/5/6 | 2.35x | 110 | 99/5/6 | 2.35x |
| `shake256` | 110 | 97/13/0 | 2.38x | 110 | 97/13/0 | 2.38x |
| `crc24-openpgp` | 110 | 106/1/3 | 14.23x | 110 | 106/1/3 | 14.23x |
| `crc16-ccitt` | 110 | 109/0/1 | 24.44x | 110 | 109/0/1 | 24.44x |
| `crc16-ibm` | 110 | 109/0/1 | 24.69x | 110 | 109/0/1 | 24.69x |

## External Pressure

All-pair comparisons by external implementation.

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gxhash` | 616 | 142/34/440 | 23% | 0.52x | 0.52x |
| `foldhash` | 440 | 205/77/158 | 47% | 1.10x | 1.03x |
| `tiny-keccak` | 22 | 22/0/0 | 100% | 1.23x | 1.28x |
| `xxhash-rust` | 440 | 274/84/82 | 62% | 1.23x | 1.26x |
| `rapidhash` | 660 | 306/244/110 | 46% | 1.24x | 1.03x |
| `ascon-hash` | 220 | 145/72/3 | 66% | 1.29x | 1.12x |
| `ring` | 1600 | 1010/226/364 | 63% | 1.35x | 1.15x |
| `dalek` | 120 | 100/13/7 | 83% | 1.35x | 1.32x |
| `aws-lc-rs` | 2060 | 1199/296/565 | 58% | 1.38x | 1.13x |
| `blake3` | 480 | 244/209/27 | 51% | 1.39x | 1.05x |
| `sha2` | 590 | 279/298/13 | 47% | 1.40x | 1.05x |
| `aegis-crate` | 220 | 126/73/21 | 57% | 1.41x | 1.08x |
| `ahash` | 440 | 310/14/116 | 70% | 1.43x | 1.50x |
| `dryoc` | 377 | 352/16/9 | 93% | 1.85x | 1.69x |
| `crc-fast` | 330 | 194/107/29 | 59% | 1.91x | 1.11x |
| `rustcrypto` | 2861 | 1857/859/145 | 65% | 1.93x | 1.12x |
| `sha3` | 680 | 596/67/17 | 88% | 2.12x | 2.00x |
| `crc64fast` | 110 | 83/14/13 | 75% | 2.16x | 1.87x |
| `crc64fast-nvme` | 110 | 82/14/14 | 75% | 2.20x | 1.91x |
| `crc32fast` | 110 | 91/11/8 | 83% | 2.51x | 2.07x |
| `crc32c` | 110 | 102/4/4 | 93% | 2.76x | 2.21x |
| `crc` | 330 | 324/1/5 | 98% | 20.48x | 32.16x |

## README Numbers

- **Headline:** 3,292 of 6,093 matched Linux CI fastest-external comparisons are wins; Linux CI geomean is 1.38x.
- **Checksums:** 4.94x geomean across Linux CI fastest-external checksum comparisons.
- **SHA-3 / SHAKE:** 2.18x SHA-3 geomean and 2.60x SHAKE/cSHAKE/KMAC geomean across Linux CI fastest-external comparisons.
- **BLAKE3:** 2.23x geomean for Linux CI fastest-external rows at `>=64 KiB`.
- **AEAD:** 1.33x geomean across Linux CI fastest-external AEAD comparisons.
- **Ed25519 / X25519:** 1.01x Ed25519 sign geomean and 0.78x X25519 geomean across Linux CI fastest-external comparisons.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | ---: | --- |
| AMD Zen4 | `ci` | `2026-05-13 16_53_23` | 2014 | `benchmark_results/2026-05-13/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-05-13 16_53_23` | 2014 | `benchmark_results/2026-05-13/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-05-13 16_53_23` | 2014 | `benchmark_results/2026-05-13/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-05-13 16_53_23` | 2014 | `benchmark_results/2026-05-13/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-05-13 16_53_23` | 1926 | `benchmark_results/2026-05-13/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-05-13 16_53_23` | 1926 | `benchmark_results/2026-05-13/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-05-13 16_53_23` | 2014 | `benchmark_results/2026-05-13/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-05-13 16_53_23` | 2014 | `benchmark_results/2026-05-13/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-05-13 16_53_23` | 1926 | `benchmark_results/2026-05-13/linux/rise-riscv/results.txt` |
| macOS aarch64 | `local` | `2026-05-13 15_00_07` | 2223 | `benchmark_results/2026-05-13/macos/aarch64/results.txt` |

## Methodology Notes

- Parsed 20085 Criterion rows: 18174 throughput rows and 1911 latency-only rows.
- Matched 12926 `rscrypto` external pairs across 6843 fastest-external cases.
- Input files include 10 result files. CI result headers use `time=16_53_23`, the GitHub Actions run creation time, per extraction policy. The local macOS file keeps its local run time.
- Matching is structural: a comparison exists when an external Criterion ID can be produced by replacing the `rscrypto` path component while keeping the same platform, primitive path, operation path, and size suffix.
- Unmatched external-only rows: 244. These are internal comparisons or benches without a corresponding `rscrypto` row, so they are not part of the ratio tables.
