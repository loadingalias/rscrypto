# Benchmark Overview

Sources:

- Linux benchmark CI run [#32185659553](https://github.com/loadingalias/rscrypto/actions/runs/32185659553), created 2026-08-18 21:03:07 UTC.
- Linux commit: `7eb44e9a38ef7a031d9181dc8c4c0fad38f46504`.
- Linux artifacts: eight successful `benchmark-*` artifacts extracted into `benchmark_results/2026-08-18/linux/*/results.txt`.
- Local macOS run: `benchmark_results/2026-07-04/macos/aarch64/results.txt` at commit `596498f0e07e869eac71fd31c157aa1b22186239`, carried forward unchanged.
- Local Ed25519 direct-secret before/after diagnostic, recorded below.

Scope: the 2026-08-18 eight-runner Linux CI benchmark matrix for commit `7eb44e9`. Ratios are `external_crate_time / rscrypto_time`; higher is better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and input shape. Internal kernel, scratch-buffer, padding-only, cold-path, PHC roundtrip, parallel-scaling, threshold-selection, public-overhead, and phase-attribution microbenches are parsed as raw rows but excluded from external win/loss claims. The macOS local run is listed separately and is not mixed into Linux CI claims.

This is a historical snapshot of commit `7eb44e9`, not an inventory of the
current public API. Primitive rows remain as measured even when a later commit
changes or removes that surface.

Runner coverage change: this run has eight Linux runners. The RISE RISC-V
runner did not execute in run #32185659553, so every aggregate below is over
eight platforms rather than the nine in the 2026-07-04 snapshot. Row counts are
therefore not directly comparable to that snapshot; ratios and geomeans are.

Equivalence correction resolved: the historical RustCrypto HMAC-SHA-256 rows
included key setup inside the timed loop. The current benchmark source hoists
`RustCryptoHmacSha256::new_from_slice` out of the timed loop and clones the
keyed state per iteration, matching the reusable-keyed-state treatment given to
rscrypto, ring, and AWS-LC. This artifact is a complete regenerated benchmark
pass, so the HMAC-SHA-256 rows and the aggregates that include them are
equivalent-work performance claims.

Surface change since 2026-07-04: the rapidhash benchmark surface was collapsed.
The former `rapidhash-64`, `rapidhash-128`, and `rapidhash-v3-128` primitives no
longer exist; `rapidhash-v3-64`, `rapidhash-stream`, `rapidhash-buildhasher`,
`rapidhash-hash-one`, and `rapidhash-hashmap` are the current rows.

Coverage note: this is a full Linux CI public benchmark pass. It includes checksum, hash, XOF, MAC, KDF, password-hashing, BLAKE2/BLAKE3, RSA import/verification, ECDSA P-256/P-384 signing and verification, Ed25519, X25519, AEAD, and ML-KEM-512/768/1024 keygen, encapsulation, and decapsulation rows. ML-KEM phase/arithmetic microbenches are present in the raw artifacts and intentionally excluded from release-level competitor claims.

## 2026-07-28 Ed25519 Direct-Secret Diagnostic

This local diagnostic compares the exact 1 KiB
`ed25519/sign/rscrypto-direct-secret/1024` Criterion case before and after the
maintenance remediation that removed duplicate secret expansion. The baseline
source is repository commit `c7338116bf8155566f9a028db1b28b5f0665e370`
with only the identical benchmark row added. The current source is that commit
plus the maintenance working-tree diff.

Both runs used the pinned `rustc 1.97.0-nightly (ca9a134e0 2026-04-26)`
toolchain on the same Apple Silicon macOS host. Criterion used 50 samples, a
1-second warm-up, and a 3-second measurement window.

| Source   |    Median | 95% confidence interval |      Mean |
| -------- | --------: | ----------------------: | --------: |
| Baseline | 21.892 µs |        21.874–21.919 µs | 21.885 µs |
| Current  | 21.754 µs |        21.704–21.799 µs | 21.757 µs |

The observed current/baseline median ratio is 0.9937. This check found no
regression. It was not an interleaved release benchmark, so it does not support
a speedup claim.

The repository policy retains only this curated overview. The local Criterion
metadata, estimates, and raw 50-sample files were distinct and hashed before
curation:

| Artifact         | Baseline SHA-256                                                   | Current SHA-256                                                    |
| ---------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `benchmark.json` | `6d27e19fd2a9563ecea5328345420c12b79f9924d3ecde179bc0166f5a62e6dd` | `6d27e19fd2a9563ecea5328345420c12b79f9924d3ecde179bc0166f5a62e6dd` |
| `estimates.json` | `728945652c3ec804ec064e9888fc431a5fa3528e885edf76e350392ae95ea2fc` | `3b987405f949847972740cb549826d46f2529caa1187bc13786f7d662ca63e03` |
| `sample.json`    | `f36052bcf65362d6203a6be768e251822dc3182ce8fab75dd9bba20097db30f9` | `a92a7e9fcc1af048c2bb5dcfd8782d07b8727e46477a27bc7948cd02c7a8a6bc` |

## Headline

| Scope                                   | Pairs | W/T/L           | Win % | Geomean | Median |
| --------------------------------------- | ----- | --------------- | ----- | ------- | ------ |
| Linux CI: all matched performance pairs | 9,674 | 6,831/2,085/758 | 71%   | 1.78x   | 1.24x  |
| Linux CI: fastest external per case     | 6,144 | 3,780/1,695/669 | 62%   | 1.62x   | 1.12x  |

Snapshot summary:

- **Headline:** 3,780 of 6,144 matched Linux CI fastest-external comparisons are wins; 5,475 are wins or ties. Linux CI fastest-external geomean is 1.62x.
- **Checksums:** 6.18x geomean across 616 fastest-external rows; W/T/L is 476/118/22.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,456 fastest-external rows; W/T/L is 1,926/1,181/349.
- **Auth/KDF:** 1.28x geomean across 160 fastest-external rows; W/T/L is 140/20/0.
- **Password hashing:** 1.07x geomean across 120 fastest-external rows; W/T/L is 55/27/38.
- **Public-key:** 1.09x geomean across 296 fastest-external rows; W/T/L is 187/59/50.
- **RSA:** 1.65x geomean across 88 fastest-external rows; W/T/L is 86/2/0.
- **AEAD:** 1.61x geomean across 1,408 fastest-external rows; W/T/L is 910/288/210.
- **ML-KEM:** 1.55x geomean across 72 fastest-external rows; W/T/L is 64/0/8.
- **ECDSA P-256/P-384:** Linux CI 0.87x geomean across 128 fastest-external rows; W/T/L is 88/7/33.
- **Top current loss areas:** `ecdsa-p384` / `sign`: 0.70x geomean across 32 rows; W/T/L is 12/0/20; pressure `aws-lc-rs` 16, `rustcrypto-p384` 4; `ecdsa-p256` / `verify`: 0.84x geomean across 32 rows; W/T/L is 20/7/5; pressure `rustcrypto-p256` 4, `aws-lc-rs` 1; `rapidhash-stream` / `one-write`: 0.87x geomean across 88 rows; W/T/L is 27/25/36; pressure `rapidhash` 36; `ecdsa-p256` / `sign`: 0.91x geomean across 32 rows; W/T/L is 28/0/4; pressure `ring` 4; `argon2id-owasp` / `hash`: 0.98x geomean across 8 rows; W/T/L is 3/1/4; pressure `rustcrypto` 3, `dryoc` 1.

## Coverage Matrix

| Platform              | Raw Criterion rows | All pairs | Fastest rows | W/T/L       | Win % | Geomean | Median |
| --------------------- | ------------------ | --------- | ------------ | ----------- | ----- | ------- | ------ |
| AMD Zen4              | 2,304              | 1,269     | 768          | 525/171/72  | 68%   | 1.47x   | 1.14x  |
| AMD Zen5              | 2,304              | 1,269     | 768          | 447/245/76  | 58%   | 1.47x   | 1.10x  |
| AWS Graviton3         | 2,308              | 1,269     | 768          | 367/287/114 | 48%   | 1.36x   | 1.04x  |
| AWS Graviton4         | 2,308              | 1,269     | 768          | 366/337/65  | 48%   | 1.37x   | 1.04x  |
| IBM Power10           | 2,055              | 1,030     | 768          | 400/302/66  | 52%   | 1.83x   | 1.06x  |
| IBM z16/s390x         | 2,055              | 1,030     | 768          | 620/67/81   | 81%   | 2.77x   | 2.19x  |
| Intel Ice Lake        | 2,304              | 1,269     | 768          | 517/137/114 | 67%   | 1.45x   | 1.17x  |
| Intel Sapphire Rapids | 2,304              | 1,269     | 768          | 538/149/81  | 70%   | 1.60x   | 1.18x  |

## Category Summary

| Category         | Rows  | W/T/L           | Win % | Geomean | Median |
| ---------------- | ----- | --------------- | ----- | ------- | ------ |
| Checksums        | 616   | 476/118/22      | 77%   | 6.18x   | 3.17x  |
| Hashes/MACs/XOFs | 3,456 | 1,926/1,181/349 | 56%   | 1.35x   | 1.08x  |
| Auth/KDF         | 160   | 140/20/0        | 88%   | 1.28x   | 1.13x  |
| Password hashing | 120   | 55/27/38        | 46%   | 1.07x   | 1.02x  |
| Public-key       | 296   | 187/59/50       | 63%   | 1.09x   | 1.14x  |
| RSA              | 88    | 86/2/0          | 98%   | 1.65x   | 1.20x  |
| AEAD             | 1,408 | 910/288/210     | 65%   | 1.61x   | 1.21x  |

## BLAKE3 Summary

BLAKE3 rows come from Linux CI run [#32185659553](https://github.com/loadingalias/rscrypto/actions/runs/32185659553). All-pair and fastest-external BLAKE3 metrics are identical because official `blake3` is the only external implementation in this bench.

| Scope                 | Rows | W/T/L      | Geomean | Median |
| --------------------- | ---- | ---------- | ------- | ------ |
| All Linux BLAKE3 rows | 384  | 187/134/63 | 1.35x   | 1.04x  |
| x86_64                | 192  | 79/89/24   | 1.18x   | 1.02x  |
| AArch64               | 96   | 44/36/16   | 1.40x   | 1.04x  |

| Platform              | Rows | W/T/L    | Geomean | Median |
| --------------------- | ---- | -------- | ------- | ------ |
| AMD Zen4              | 48   | 20/22/6  | 1.24x   | 1.01x  |
| AMD Zen5              | 48   | 18/27/3  | 1.27x   | 1.02x  |
| AWS Graviton3         | 48   | 22/15/11 | 1.36x   | 0.98x  |
| AWS Graviton4         | 48   | 22/21/5  | 1.44x   | 1.04x  |
| IBM Power10           | 48   | 32/6/10  | 1.76x   | 1.12x  |
| IBM z16/s390x         | 48   | 32/3/13  | 1.69x   | 1.69x  |
| Intel Ice Lake        | 48   | 19/21/8  | 1.09x   | 1.00x  |
| Intel Sapphire Rapids | 48   | 22/19/7  | 1.13x   | 1.03x  |

| Operation    | Rows | W/T/L    | Geomean | Median |
| ------------ | ---- | -------- | ------- | ------ |
| `oneshot`    | 88   | 35/35/18 | 1.33x   | 1.00x  |
| `keyed`      | 88   | 27/21/40 | 1.20x   | 0.95x  |
| `derive-key` | 88   | 65/21/2  | 1.59x   | 1.53x  |
| `streaming`  | 32   | 10/21/1  | 1.21x   | 1.02x  |
| `xof`        | 88   | 50/36/2  | 1.37x   | 1.07x  |

## ML-KEM Summary

ML-KEM public coverage is complete for the CI-selected primitive set: ML-KEM-512, ML-KEM-768, and ML-KEM-1024 each include keygen, encapsulate, and decapsulate on all eight Linux platforms. POWER10 and s390x do not have `aws-lc-rs` ML-KEM rows in this artifact set, but still have rscrypto plus `libcrux`, `fips203`, and RustCrypto comparison rows for every public operation.

| Platform              | Raw ML-KEM rows | Fastest rows | W/T/L | Geomean | Median | Fastest external split     |
| --------------------- | --------------- | ------------ | ----- | ------- | ------ | -------------------------- |
| AMD Zen4              | 45              | 9            | 9/0/0 | 1.83x   | 1.82x  | `libcrux` 7, `aws-lc-rs` 2 |
| AMD Zen5              | 45              | 9            | 9/0/0 | 1.95x   | 1.91x  | `libcrux` 9                |
| AWS Graviton3         | 45              | 9            | 5/0/4 | 1.09x   | 1.12x  | `aws-lc-rs` 9              |
| AWS Graviton4         | 45              | 9            | 5/0/4 | 1.08x   | 1.18x  | `aws-lc-rs` 9              |
| IBM Power10           | 36              | 9            | 9/0/0 | 1.41x   | 1.53x  | `libcrux` 9                |
| IBM z16/s390x         | 36              | 9            | 9/0/0 | 1.68x   | 1.74x  | `libcrux` 9                |
| Intel Ice Lake        | 45              | 9            | 9/0/0 | 1.80x   | 1.75x  | `libcrux` 7, `aws-lc-rs` 2 |
| Intel Sapphire Rapids | 45              | 9            | 9/0/0 | 1.84x   | 1.80x  | `aws-lc-rs` 7, `libcrux` 2 |

| Primitive/op                | Rows | W/T/L | Win % | Geomean | Median | Pressure      |
| --------------------------- | ---- | ----- | ----- | ------- | ------ | ------------- |
| `mlkem1024` / `decapsulate` | 8    | 8/0/0 | 100%  | 1.70x   | 1.86x  | none          |
| `mlkem1024` / `encapsulate` | 8    | 8/0/0 | 100%  | 2.51x   | 2.63x  | none          |
| `mlkem1024` / `keygen`      | 8    | 6/0/2 | 75%   | 1.02x   | 1.13x  | `aws-lc-rs` 2 |
| `mlkem512` / `decapsulate`  | 8    | 6/0/2 | 75%   | 1.41x   | 1.59x  | `aws-lc-rs` 2 |
| `mlkem512` / `encapsulate`  | 8    | 8/0/0 | 100%  | 1.94x   | 2.17x  | none          |
| `mlkem512` / `keygen`       | 8    | 6/0/2 | 75%   | 1.09x   | 1.22x  | `aws-lc-rs` 2 |
| `mlkem768` / `decapsulate`  | 8    | 8/0/0 | 100%  | 1.58x   | 1.75x  | none          |
| `mlkem768` / `encapsulate`  | 8    | 8/0/0 | 100%  | 2.33x   | 2.54x  | none          |
| `mlkem768` / `keygen`       | 8    | 6/0/2 | 75%   | 1.06x   | 1.13x  | `aws-lc-rs` 2 |

## ECDSA Summary

ECDSA signing includes both deterministic and blinded rscrypto rows in raw results; aggregate fastest-external comparisons use the fastest rscrypto row for the exact case. Constant-time release evidence is tracked separately by `ct.toml` and CT workflow artifacts.

Regression: every ECDSA aggregate in this snapshot is dominated by a single
platform. On IBM z16/s390x, P-256 signing went from 137.10 µs (2026-07-04) to
8,889.30 µs, and P-384 signing from 562.91 µs to 34,557.00 µs, while the
external crates on the same runner moved by less than 1.4x. Excluding s390x, the
seven-runner geomeans are `ecdsa-p256` / `sign` 1.33x, `ecdsa-p256` / `verify`
1.19x, `ecdsa-p384` / `sign` 1.01x, and `ecdsa-p384` / `verify` 1.53x.

| Operation               | Rows | W/T/L   | Geomean | Median |
| ----------------------- | ---- | ------- | ------- | ------ |
| `ecdsa-p256` / `sign`   | 32   | 28/0/4  | 0.91x   | 1.30x  |
| `ecdsa-p256` / `verify` | 32   | 20/7/5  | 0.84x   | 1.08x  |
| `ecdsa-p384` / `sign`   | 32   | 12/0/20 | 0.70x   | 0.83x  |
| `ecdsa-p384` / `verify` | 32   | 28/0/4  | 1.08x   | 1.36x  |

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are strongest-external comparisons; all-pair columns include every matched external implementation.

| Primitive               | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L  | All geomean |
| ----------------------- | ------------ | ------------- | --------------- | --------- | ---------- | ----------- |
| `ecdsa-p384`            | 64           | 40/0/24       | 0.87x           | 176       | 144/0/32   | 2.27x       |
| `ecdsa-p256`            | 64           | 48/7/9        | 0.87x           | 176       | 148/11/17  | 1.57x       |
| `rapidhash-stream`      | 176          | 61/33/82      | 0.92x           | 176       | 61/33/82   | 0.92x       |
| `argon2id-owasp`        | 8            | 3/1/4         | 0.98x           | 16        | 7/4/5      | 1.25x       |
| `xxh3-buildhasher`      | 88           | 41/12/35      | 0.99x           | 88        | 41/12/35   | 0.99x       |
| `x25519`                | 16           | 3/13/0        | 1.02x           | 44        | 31/13/0    | 1.58x       |
| `argon2i-small`         | 24           | 10/3/11       | 1.03x           | 40        | 26/3/11    | 1.34x       |
| `argon2id-small`        | 24           | 10/3/11       | 1.03x           | 40        | 25/4/11    | 1.35x       |
| `argon2d-small`         | 24           | 10/5/9        | 1.04x           | 24        | 10/5/9     | 1.04x       |
| `rapidhash-v3-64`       | 88           | 21/45/22      | 1.05x           | 88        | 21/45/22   | 1.05x       |
| `blake2b256`            | 200          | 101/99/0      | 1.07x           | 312       | 204/108/0  | 1.31x       |
| `scrypt-owasp`          | 8            | 4/2/2         | 1.08x           | 8         | 4/2/2      | 1.08x       |
| `blake2b512`            | 176          | 106/69/1      | 1.08x           | 264       | 194/69/1   | 1.33x       |
| `blake2s256`            | 200          | 114/86/0      | 1.11x           | 200       | 114/86/0   | 1.11x       |
| `chacha20-poly1305`     | 176          | 75/101/0      | 1.12x           | 484       | 304/180/0  | 1.32x       |
| `xxh3-128`              | 88           | 34/42/12      | 1.13x           | 88        | 34/42/12   | 1.13x       |
| `xxh3-64`               | 88           | 34/34/20      | 1.13x           | 88        | 34/34/20   | 1.13x       |
| `blake2s128`            | 176          | 113/63/0      | 1.13x           | 176       | 113/63/0   | 1.13x       |
| `ed25519`               | 80           | 32/39/9       | 1.14x           | 256       | 194/48/14  | 1.41x       |
| `xxh3-hashmap`          | 8            | 7/1/0         | 1.15x           | 8         | 7/1/0      | 1.15x       |
| `scrypt-small`          | 32           | 18/13/1       | 1.18x           | 32        | 18/13/1    | 1.18x       |
| `rapidhash-buildhasher` | 88           | 44/29/15      | 1.19x           | 88        | 44/29/15   | 1.19x       |
| `aegis-256`             | 176          | 81/65/30      | 1.23x           | 176       | 81/65/30   | 1.23x       |
| `hmac-sha256`           | 104          | 42/36/26      | 1.24x           | 258       | 144/78/36  | 1.60x       |
| `hmac-sha384`           | 88           | 28/49/11      | 1.24x           | 242       | 133/93/16  | 1.29x       |
| `hmac-sha512`           | 88           | 32/44/12      | 1.27x           | 242       | 137/88/17  | 1.31x       |
| `sha256`                | 104          | 44/46/14      | 1.27x           | 258       | 143/89/26  | 1.60x       |
| `hkdf-sha384`           | 32           | 29/3/0        | 1.27x           | 88        | 85/3/0     | 1.59x       |
| `rsa-8192`              | 16           | 14/2/0        | 1.28x           | 28        | 26/2/0     | 1.33x       |
| `hkdf-sha256`           | 32           | 27/5/0        | 1.28x           | 88        | 83/5/0     | 1.93x       |
| `pbkdf2-sha256`         | 48           | 43/5/0        | 1.28x           | 132       | 127/5/0    | 1.71x       |
| `pbkdf2-sha512`         | 48           | 41/7/0        | 1.28x           | 132       | 125/7/0    | 1.34x       |
| `sha512`                | 104          | 48/51/5       | 1.29x           | 258       | 160/88/10  | 1.31x       |
| `sha384`                | 88           | 43/39/6       | 1.30x           | 242       | 151/80/11  | 1.32x       |
| `ascon-hash256`         | 88           | 56/31/1       | 1.30x           | 88        | 56/31/1    | 1.30x       |
| `sha512-256`            | 88           | 50/38/0       | 1.33x           | 88        | 50/38/0    | 1.33x       |
| `blake3`                | 384          | 187/134/63    | 1.35x           | 384       | 187/134/63 | 1.35x       |
| `ascon-aead128`         | 176          | 136/39/1      | 1.39x           | 176       | 136/39/1   | 1.39x       |
| `ascon-xof128`          | 88           | 66/20/2       | 1.39x           | 88        | 66/20/2    | 1.39x       |
| `xchacha20-poly1305`    | 176          | 173/3/0       | 1.43x           | 176       | 173/3/0    | 1.43x       |
| `mlkem512`              | 24           | 20/0/4        | 1.44x           | 90        | 86/0/4     | 2.90x       |
| `rapidhash-hash-one`    | 24           | 18/4/2        | 1.47x           | 24        | 18/4/2     | 1.47x       |
| `mlkem768`              | 24           | 22/0/2        | 1.57x           | 90        | 88/0/2     | 3.38x       |
| `rapidhash-hashmap`     | 24           | 24/0/0        | 1.61x           | 24        | 24/0/0     | 1.61x       |
| `mlkem1024`             | 24           | 22/0/2        | 1.63x           | 90        | 88/0/2     | 3.60x       |
| `rsa-4096`              | 24           | 24/0/0        | 1.70x           | 52        | 52/0/0     | 2.69x       |
| `crc32c`                | 88           | 42/38/8       | 1.73x           | 176       | 130/38/8   | 2.41x       |
| `rsa-3072`              | 24           | 24/0/0        | 1.75x           | 52        | 52/0/0     | 2.73x       |
| `rsa-2048`              | 24           | 24/0/0        | 1.79x           | 52        | 52/0/0     | 2.77x       |
| `aes-128-gcm`           | 176          | 96/42/38      | 1.80x           | 484       | 390/50/44  | 2.01x       |
| `crc32`                 | 88           | 47/33/8       | 1.80x           | 176       | 133/35/8   | 2.51x       |
| `aes-256-gcm`           | 176          | 94/36/46      | 1.83x           | 484       | 382/44/58  | 2.02x       |
| `kmac256`               | 88           | 58/19/11      | 1.86x           | 88        | 58/19/11   | 1.86x       |
| `cshake256`             | 88           | 58/21/9       | 1.90x           | 88        | 58/21/9    | 1.90x       |
| `shake128`              | 88           | 58/30/0       | 1.94x           | 88        | 58/30/0    | 1.94x       |
| `shake256`              | 88           | 63/25/0       | 1.98x           | 88        | 63/25/0    | 1.98x       |
| `sha224`                | 88           | 51/37/0       | 2.01x           | 88        | 51/37/0    | 2.01x       |
| `aes-128-gcm-siv`       | 176          | 127/1/48      | 2.20x           | 308       | 237/16/55  | 2.92x       |
| `sha3-224`              | 88           | 77/11/0       | 2.27x           | 88        | 77/11/0    | 2.27x       |
| `sha3-256`              | 104          | 91/13/0       | 2.28x           | 104       | 91/13/0    | 2.28x       |
| `aes-256-gcm-siv`       | 176          | 128/1/47      | 2.34x           | 308       | 259/2/47   | 3.16x       |
| `crc64-nvme`            | 88           | 52/35/1       | 2.34x           | 88        | 52/35/1    | 2.34x       |
| `sha3-384`              | 88           | 79/9/0        | 2.35x           | 88        | 79/9/0     | 2.35x       |
| `sha3-512`              | 88           | 77/11/0       | 2.38x           | 88        | 77/11/0    | 2.38x       |
| `crc64-xz`              | 88           | 73/12/3       | 2.78x           | 88        | 73/12/3    | 2.78x       |
| `crc24-openpgp`         | 88           | 86/0/2        | 17.62x          | 88        | 86/0/2     | 17.62x      |
| `crc16-ccitt`           | 88           | 88/0/0        | 30.24x          | 88        | 88/0/0     | 30.24x      |
| `crc16-ibm`             | 88           | 88/0/0        | 32.07x          | 88        | 88/0/0     | 32.07x      |

## Linux Worst Individual Rows

| Platform      | Case                         | Fastest external  | Ratio |
| ------------- | ---------------------------- | ----------------- | ----- |
| IBM z16/s390x | `ecdsa-p256 / sign / 1024`   | `ring`            | 0.05x |
| IBM z16/s390x | `ecdsa-p384 / sign / 16384`  | `rustcrypto-p384` | 0.05x |
| IBM z16/s390x | `ecdsa-p384 / sign / 1024`   | `rustcrypto-p384` | 0.05x |
| IBM z16/s390x | `ecdsa-p384 / sign / 0`      | `rustcrypto-p384` | 0.06x |
| IBM z16/s390x | `ecdsa-p384 / sign / 32`     | `rustcrypto-p384` | 0.06x |
| IBM z16/s390x | `ecdsa-p256 / sign / 0`      | `ring`            | 0.06x |
| IBM z16/s390x | `ecdsa-p256 / sign / 32`     | `ring`            | 0.06x |
| IBM z16/s390x | `ecdsa-p256 / verify / 32`   | `rustcrypto-p256` | 0.06x |
| IBM z16/s390x | `ecdsa-p256 / verify / 1024` | `rustcrypto-p256` | 0.07x |
| IBM z16/s390x | `ecdsa-p256 / sign / 16384`  | `ring`            | 0.07x |
| IBM z16/s390x | `ecdsa-p256 / verify / 0`    | `rustcrypto-p256` | 0.07x |
| IBM z16/s390x | `ecdsa-p384 / verify / 1024` | `rustcrypto-p384` | 0.09x |

## Linux Strongest Individual Rows

| Platform              | Case                    | Fastest external | Ratio   |
| --------------------- | ----------------------- | ---------------- | ------- |
| Intel Sapphire Rapids | `crc16-ibm / 262144`    | `crc`            | 212.60x |
| Intel Sapphire Rapids | `crc16-ccitt / 262144`  | `crc`            | 209.27x |
| Intel Sapphire Rapids | `crc16-ccitt / 16384`   | `crc`            | 206.40x |
| Intel Sapphire Rapids | `crc16-ibm / 16384`     | `crc`            | 198.48x |
| Intel Sapphire Rapids | `crc16-ibm / 1048576`   | `crc`            | 187.52x |
| Intel Sapphire Rapids | `crc16-ibm / 4096`      | `crc`            | 178.55x |
| Intel Sapphire Rapids | `crc16-ibm / 65536`     | `crc`            | 178.28x |
| Intel Sapphire Rapids | `crc16-ccitt / 4096`    | `crc`            | 178.15x |
| IBM Power10           | `crc16-ccitt / 1048576` | `crc`            | 176.67x |
| IBM Power10           | `crc16-ibm / 1048576`   | `crc`            | 176.60x |
| Intel Sapphire Rapids | `crc16-ccitt / 1048576` | `crc`            | 176.46x |
| IBM Power10           | `crc16-ccitt / 262144`  | `crc`            | 175.61x |

## Top Five Loss Areas

- `ecdsa-p384` / `sign`: 0.70x geomean across 32 rows; W/T/L 12/0/20; pressure `aws-lc-rs` 16, `rustcrypto-p384` 4.
- `ecdsa-p256` / `verify`: 0.84x geomean across 32 rows; W/T/L 20/7/5; pressure `rustcrypto-p256` 4, `aws-lc-rs` 1.
- `rapidhash-stream` / `one-write`: 0.87x geomean across 88 rows; W/T/L 27/25/36; pressure `rapidhash` 36.
- `ecdsa-p256` / `sign`: 0.91x geomean across 32 rows; W/T/L 28/0/4; pressure `ring` 4.
- `argon2id-owasp` / `hash`: 0.98x geomean across 8 rows; W/T/L 3/1/4; pressure `rustcrypto` 3, `dryoc` 1.

## External Pressure

| External          | Pairs | W/T/L         | Win % | Geomean | Median |
| ----------------- | ----- | ------------- | ----- | ------- | ------ |
| `rapidhash`       | 400   | 168/111/121   | 42%   | 1.07x   | 1.01x  |
| `xxhash-rust`     | 272   | 116/89/67     | 43%   | 1.08x   | 1.00x  |
| `aws-lc-rs`       | 1,434 | 896/343/195   | 62%   | 1.21x   | 1.13x  |
| `aegis-crate`     | 176   | 81/65/30      | 46%   | 1.23x   | 1.04x  |
| `ascon-hash`      | 176   | 122/51/3      | 69%   | 1.34x   | 1.32x  |
| `blake3`          | 384   | 187/134/63    | 49%   | 1.35x   | 1.04x  |
| `ascon-aead`      | 176   | 136/39/1      | 77%   | 1.39x   | 1.38x  |
| `dalek`           | 96    | 80/12/4       | 83%   | 1.52x   | 1.49x  |
| `sha2`            | 472   | 276/194/2     | 58%   | 1.60x   | 1.07x  |
| `ring`            | 1,472 | 1,154/237/81  | 78%   | 1.63x   | 1.28x  |
| `libcrux`         | 72    | 72/0/0        | 100%  | 1.79x   | 1.72x  |
| `dryoc`           | 320   | 293/22/5      | 92%   | 1.81x   | 1.85x  |
| `rustcrypto`      | 2,440 | 1,783/529/128 | 73%   | 1.87x   | 1.21x  |
| `tiny-keccak`     | 352   | 237/95/20     | 67%   | 1.92x   | 2.10x  |
| `crc-fast`        | 264   | 153/101/10    | 58%   | 2.15x   | 1.20x  |
| `sha3`            | 368   | 324/44/0      | 88%   | 2.32x   | 2.15x  |
| `crc32fast`       | 88    | 79/4/5        | 90%   | 2.75x   | 2.06x  |
| `crc64fast`       | 88    | 73/12/3       | 83%   | 2.78x   | 2.49x  |
| `rustcrypto-p256` | 64    | 56/0/8        | 88%   | 3.03x   | 3.10x  |
| `rustcrypto-p384` | 64    | 56/0/8        | 88%   | 3.06x   | 5.50x  |
| `crc32c`          | 88    | 83/3/2        | 94%   | 3.13x   | 2.29x  |
| `fips203`         | 72    | 72/0/0        | 100%  | 5.28x   | 6.07x  |
| `rustcrypto-rsa`  | 72    | 72/0/0        | 100%  | 6.07x   | 6.50x  |
| `crc`             | 264   | 262/0/2       | 99%   | 25.76x  | 46.98x |

## macOS Local Snapshot

The macOS Apple Silicon run is local evidence from the 2026-07-04 full benchmark at commit `596498f`, carried forward unchanged in this refresh. It is useful for Apple Silicon planning but is not folded into Linux CI release claims. The ML-KEM row uses the same artifact's public ML-KEM rows.

| Scope                                      | Pairs | W/T/L      | Win % | Geomean | Median |
| ------------------------------------------ | ----- | ---------- | ----- | ------- | ------ |
| macOS local: all matched performance pairs | 1,297 | 815/404/78 | 63%   | 1.66x   | 1.16x  |
| macOS local: fastest external per case     | 774   | 382/326/66 | 49%   | 1.37x   | 1.05x  |
| macOS local: ML-KEM fastest external       | 9     | 6/1/2      | 67%   | 1.35x   | 1.39x  |

## README Numbers

- **Headline:** 3,780 of 6,144 matched Linux CI fastest-external comparisons are wins; 5,475 are wins or ties. Linux CI geomean is 1.62x.
- **Checksums:** 6.18x geomean across 616 Linux CI fastest-external rows; W/T/L 476/118/22.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,456 Linux CI fastest-external rows; W/T/L 1,926/1,181/349.
- **Auth/KDF:** 1.28x geomean across 160 Linux CI fastest-external rows; W/T/L 140/20/0.
- **Password hashing:** 1.07x geomean across 120 Linux CI fastest-external rows; W/T/L 55/27/38.
- **Public-key:** 1.09x geomean across 296 Linux CI fastest-external rows; W/T/L 187/59/50.
- **RSA:** 1.65x geomean across 88 Linux CI fastest-external rows; W/T/L 86/2/0.
- **AEAD:** 1.61x geomean across 1,408 Linux CI fastest-external rows; W/T/L 910/288/210.
- **ML-KEM:** 1.55x geomean across 72 Linux CI fastest-external rows; W/T/L 64/0/8.
- **ECDSA P-256/P-384:** 0.87x Linux CI geomean across 128 fastest-external rows; W/T/L 88/7/33.
- **Current top losses:** `ecdsa-p384` / `sign`: 0.70x geomean across 32 rows; W/T/L 12/0/20; pressure `aws-lc-rs` 16, `rustcrypto-p384` 4; `ecdsa-p256` / `verify`: 0.84x geomean across 32 rows; W/T/L 20/7/5; pressure `rustcrypto-p256` 4, `aws-lc-rs` 1; `rapidhash-stream` / `one-write`: 0.87x geomean across 88 rows; W/T/L 27/25/36; pressure `rapidhash` 36; `ecdsa-p256` / `sign`: 0.91x geomean across 32 rows; W/T/L 28/0/4; pressure `ring` 4; `argon2id-owasp` / `hash`: 0.98x geomean across 8 rows; W/T/L 3/1/4; pressure `rustcrypto` 3, `dryoc` 1.

## Raw Results

| Platform              | Mode    | Date/time             | Parsed rows | Result                                                       |
| --------------------- | ------- | --------------------- | ----------- | ------------------------------------------------------------ |
| AMD Zen4              | `ci`    | `2026-08-18 21_03_07` | 2,304       | `benchmark_results/2026-08-18/linux/amd-zen4/results.txt`    |
| AMD Zen5              | `ci`    | `2026-08-18 21_03_07` | 2,304       | `benchmark_results/2026-08-18/linux/amd-zen5/results.txt`    |
| AWS Graviton3         | `ci`    | `2026-08-18 21_03_07` | 2,308       | `benchmark_results/2026-08-18/linux/graviton3/results.txt`   |
| AWS Graviton4         | `ci`    | `2026-08-18 21_03_07` | 2,308       | `benchmark_results/2026-08-18/linux/graviton4/results.txt`   |
| IBM Power10           | `ci`    | `2026-08-18 21_03_07` | 2,055       | `benchmark_results/2026-08-18/linux/ibm-power10/results.txt` |
| IBM z16/s390x         | `ci`    | `2026-08-18 21_03_07` | 2,055       | `benchmark_results/2026-08-18/linux/ibm-s390x/results.txt`   |
| Intel Ice Lake        | `ci`    | `2026-08-18 21_03_07` | 2,304       | `benchmark_results/2026-08-18/linux/intel-icl/results.txt`   |
| Intel Sapphire Rapids | `ci`    | `2026-08-18 21_03_07` | 2,304       | `benchmark_results/2026-08-18/linux/intel-spr/results.txt`   |
| macOS Apple Silicon   | `local` | `2026-07-04 12_28_04` | 2,277       | `benchmark_results/2026-07-04/macos/aarch64/results.txt`     |
