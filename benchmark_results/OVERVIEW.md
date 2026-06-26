# Benchmark Overview

Sources:

- Linux benchmark CI run [#28202726811](https://github.com/loadingalias/rscrypto/actions/runs/28202726811), created 2026-06-25 21:50:27 UTC.
- Linux commit: `c5ece53efd98980831de728974d1f588cf055ca5`.
- Linux artifacts: nine successful `benchmark-*` artifacts extracted into `benchmark_results/2026-06-25/linux/*/results.txt`.
- Targeted P-384 follow-up CI run [#28212427930](https://github.com/loadingalias/rscrypto/actions/runs/28212427930), created 2026-06-26 02:02:44 UTC, commit `128c76e546243ed7d048553573c4047099dab56b`.
- Targeted P-384 fixed-base follow-up CI run [#28216163852](https://github.com/loadingalias/rscrypto/actions/runs/28216163852), created 2026-06-26 03:58:13 UTC, commit `d8481124c5b69cfba61f8135bfcec03763200530`.
- Targeted P-384 reduction-cleanup follow-up CI run [#28245803704](https://github.com/loadingalias/rscrypto/actions/runs/28245803704), created 2026-06-26 14:51:50 UTC, commit `562f3f85dba6ca21745a3189c83b53c2027d3009`.
- Local macOS run: `benchmark_results/2026-06-22/macos/aarch64/results.txt` at commit `b978c2ca45611325850d7f1af94718e497acde50`.

Scope: the 2026-06-25 nine-runner Linux CI benchmark matrix for commit `c5ece53`. Ratios are `external_crate_time / rscrypto_time`; higher is better. Wins are `>1.05x`, ties are `0.95x..1.05x`, and losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, primitive, operation, and input shape. Internal kernel, scratch-buffer, padding-only, cold-path, PHC roundtrip, parallel-scaling, threshold-selection, public-overhead, and phase-attribution microbenches are parsed as raw rows but excluded from external win/loss claims. The macOS local run is listed separately and is not mixed into Linux CI claims.

Coverage note: this is a full Linux CI public benchmark pass. It includes checksum, hash, XOF, MAC, KDF, password-hashing, BLAKE2/BLAKE3, RSA import/verification, ECDSA P-256/P-384 signing and verification, Ed25519, X25519, AEAD, and ML-KEM-512/768/1024 keygen, encapsulation, and decapsulation rows. ML-KEM phase/arithmetic microbenches are present in the raw artifacts and intentionally excluded from release-level competitor claims.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Linux CI: all matched performance pairs | 10,781 | 7,554/2,391/836 | 70% | 1.80x | 1.22x |
| Linux CI: fastest external per case | 6,750 | 4,103/1,993/654 | 61% | 1.59x | 1.11x |

Shareable release summary:

- **Headline:** 4,103 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,096 are wins or ties. Linux CI fastest-external geomean is 1.59x.
- **Checksums:** 5.22x geomean across 693 fastest-external rows; W/T/L is 525/119/49.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,726 fastest-external rows; W/T/L is 2,063/1,443/220.
- **Auth/KDF:** 1.24x geomean across 180 fastest-external rows; W/T/L is 162/17/1.
- **Password hashing:** 1.07x geomean across 135 fastest-external rows; W/T/L is 67/28/40.
- **Public-key:** 1.31x geomean across 333 fastest-external rows; W/T/L is 214/77/42.
- **RSA:** 1.56x geomean across 99 fastest-external rows; W/T/L is 88/3/8.
- **AEAD:** 1.56x geomean across 1,584 fastest-external rows; W/T/L is 984/306/294.
- **ML-KEM:** 1.49x geomean across 81 fastest-external rows; W/T/L is 63/8/10.
- **ECDSA P-256/P-384:** Linux CI 1.42x geomean across 144 fastest-external rows; W/T/L is 115/13/16.
- **Top current loss areas:** `argon2id-owasp` / `hash`: 0.95x geomean across 9 rows; W/T/L 2/1/6; pressure `rustcrypto` 6, `dryoc` 1; `mlkem1024` / `keygen`: 0.95x geomean across 9 rows; W/T/L 3/2/4; pressure `aws-lc-rs` 3, `libcrux` 1; `ecdsa-p384` / `sign`: 0.96x geomean across 36 rows; W/T/L 16/4/16; pressure `aws-lc-rs` 16; `ed25519` / `verify`: 0.98x geomean across 36 rows; W/T/L 8/16/12; pressure `ring` 9, `dalek` 7, `dryoc` 7, `aws-lc-rs` 4; `mlkem768` / `keygen`: 0.98x geomean across 9 rows; W/T/L 4/2/3; pressure `aws-lc-rs` 3.
- **P-384 follow-up:** scoped x86_64 run [#28245803704](https://github.com/loadingalias/rscrypto/actions/runs/28245803704) kept the P-384 signing rows at W/T/L 16/0/0 vs AWS-LC after deleting the x86_64 P-384 order-reduction assembly, with 1.34x geomean and 1.35x median. This targeted run is not folded into the full nine-runner totals above.

## Coverage Matrix

| Platform | Raw Criterion rows | All pairs | Fastest rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 2,269 | 1,251 | 750 | 517/162/71 | 69% | 1.49x | 1.16x |
| AMD Zen5 | 2,269 | 1,251 | 750 | 451/234/65 | 60% | 1.50x | 1.10x |
| AWS Graviton3 | 2,275 | 1,251 | 750 | 366/300/84 | 49% | 1.37x | 1.05x |
| AWS Graviton4 | 2,275 | 1,251 | 750 | 369/333/48 | 49% | 1.39x | 1.05x |
| IBM Power10 | 2,020 | 1,012 | 750 | 386/319/45 | 51% | 1.88x | 1.06x |
| IBM z16/s390x | 2,020 | 1,012 | 750 | 611/93/46 | 81% | 3.13x | 2.33x |
| Intel Ice Lake | 2,269 | 1,251 | 750 | 533/135/82 | 71% | 1.49x | 1.19x |
| Intel Sapphire Rapids | 2,269 | 1,251 | 750 | 530/146/74 | 71% | 1.63x | 1.22x |
| RISE RISC-V | 2,269 | 1,251 | 750 | 340/271/139 | 45% | 1.09x | 1.03x |

## Category Summary

| Category | Rows | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| Checksums | 693 | 525/119/49 | 76% | 5.22x | 2.43x |
| Hashes/MACs/XOFs | 3,726 | 2,063/1,443/220 | 55% | 1.35x | 1.07x |
| Auth/KDF | 180 | 162/17/1 | 90% | 1.24x | 1.12x |
| Password hashing | 135 | 67/28/40 | 50% | 1.07x | 1.03x |
| Public-key | 333 | 214/77/42 | 64% | 1.31x | 1.19x |
| RSA | 99 | 88/3/8 | 89% | 1.56x | 1.18x |
| AEAD | 1,584 | 984/306/294 | 62% | 1.56x | 1.14x |

## BLAKE3 Summary

BLAKE3 rows come from Linux CI run [#28202726811](https://github.com/loadingalias/rscrypto/actions/runs/28202726811). All-pair and fastest-external BLAKE3 metrics are identical because official `blake3` is the only external implementation in this bench.

| Scope | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| All Linux BLAKE3 rows | 432 | 225/179/28 | 1.40x | 1.06x |
| x86_64 | 192 | 84/93/15 | 1.24x | 1.03x |
| AArch64 | 96 | 44/47/5 | 1.44x | 1.05x |

| Platform | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| AMD Zen4 | 48 | 20/22/6 | 1.30x | 1.02x |
| AMD Zen5 | 48 | 21/25/2 | 1.31x | 1.03x |
| AWS Graviton3 | 48 | 22/22/4 | 1.41x | 1.00x |
| AWS Graviton4 | 48 | 22/25/1 | 1.47x | 1.05x |
| IBM Power10 | 48 | 39/7/2 | 1.98x | 1.83x |
| IBM z16/s390x | 48 | 41/3/4 | 1.88x | 2.06x |
| Intel Ice Lake | 48 | 20/26/2 | 1.17x | 1.03x |
| Intel Sapphire Rapids | 48 | 23/20/5 | 1.19x | 1.05x |
| RISE RISC-V | 48 | 17/29/2 | 1.11x | 1.00x |

| Operation | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `oneshot` | 99 | 42/50/7 | 1.32x | 1.02x |
| `keyed` | 99 | 41/49/9 | 1.30x | 1.01x |
| `derive-key` | 99 | 78/18/3 | 1.76x | 1.84x |
| `streaming` | 36 | 10/24/2 | 1.18x | 1.01x |
| `xof` | 99 | 54/38/7 | 1.35x | 1.07x |

## ML-KEM Summary

ML-KEM public coverage is complete for the CI-selected primitive set: ML-KEM-512, ML-KEM-768, and ML-KEM-1024 each include keygen, encapsulate, and decapsulate on all nine Linux platforms. POWER10 and s390x do not have `aws-lc-rs` ML-KEM rows in this artifact set, but still have rscrypto plus `libcrux`, `fips203`, and RustCrypto comparison rows for every public operation.

| Platform | Raw ML-KEM rows | Fastest rows | W/T/L | Geomean | Median | Fastest external split |
| --- | --- | --- | --- | --- | --- | --- |
| AMD Zen4 | 45 | 9 | 8/1/0 | 1.80x | 1.71x | `libcrux` 7, `aws-lc-rs` 2 |
| AMD Zen5 | 45 | 9 | 9/0/0 | 1.89x | 1.75x | `libcrux` 9 |
| AWS Graviton3 | 45 | 9 | 5/1/3 | 1.13x | 1.23x | `aws-lc-rs` 9 |
| AWS Graviton4 | 45 | 9 | 5/1/3 | 1.13x | 1.23x | `aws-lc-rs` 9 |
| IBM Power10 | 36 | 9 | 6/2/1 | 1.41x | 1.60x | `libcrux` 9 |
| IBM z16/s390x | 36 | 9 | 9/0/0 | 1.69x | 1.78x | `libcrux` 9 |
| Intel Ice Lake | 45 | 9 | 7/2/0 | 1.74x | 1.72x | `libcrux` 7, `aws-lc-rs` 2 |
| Intel Sapphire Rapids | 45 | 9 | 9/0/0 | 1.81x | 1.87x | `aws-lc-rs` 6, `libcrux` 3 |
| RISE RISC-V | 45 | 9 | 5/1/3 | 1.11x | 1.07x | `aws-lc-rs` 9 |

| Primitive/op | Rows | W/T/L | Win % | Geomean | Median | Pressure |
| --- | --- | --- | --- | --- | --- | --- |
| `mlkem1024` / `decapsulate` | 9 | 9/0/0 | 100% | 1.65x | 1.85x | none |
| `mlkem1024` / `encapsulate` | 9 | 9/0/0 | 100% | 2.45x | 2.24x | none |
| `mlkem1024` / `keygen` | 9 | 3/2/4 | 33% | 0.95x | 1.03x | `aws-lc-rs` 3, `libcrux` 1 |
| `mlkem512` / `decapsulate` | 9 | 6/3/0 | 67% | 1.35x | 1.50x | none |
| `mlkem512` / `encapsulate` | 9 | 9/0/0 | 100% | 1.91x | 1.78x | none |
| `mlkem512` / `keygen` | 9 | 5/1/3 | 56% | 1.05x | 1.19x | `aws-lc-rs` 3 |
| `mlkem768` / `decapsulate` | 9 | 9/0/0 | 100% | 1.53x | 1.71x | none |
| `mlkem768` / `encapsulate` | 9 | 9/0/0 | 100% | 2.31x | 2.03x | none |
| `mlkem768` / `keygen` | 9 | 4/2/3 | 44% | 0.98x | 1.02x | `aws-lc-rs` 3 |

## ECDSA Summary

ECDSA signing includes both deterministic and blinded rscrypto rows in raw results; aggregate fastest-external comparisons use the fastest rscrypto row for the exact case. Constant-time release evidence is tracked separately by `ct.toml` and CT workflow artifacts.

| Operation | Rows | W/T/L | Geomean | Median |
| --- | --- | --- | --- | --- |
| `ecdsa-p256` / `sign` | 36 | 36/0/0 | 1.45x | 1.36x |
| `ecdsa-p256` / `verify` | 36 | 27/9/0 | 1.60x | 1.13x |
| `ecdsa-p384` / `sign` | 36 | 16/4/16 | 0.96x | 0.98x |
| `ecdsa-p384` / `verify` | 36 | 36/0/0 | 1.81x | 1.39x |

### P-384 x86_64 Follow-Ups

Run [#28212427930](https://github.com/loadingalias/rscrypto/actions/runs/28212427930) is a quick, targeted
`ecdsa-p384` auth bench on the four x86_64 lanes that carried the worst signing losses after deleting
`src/auth/asm/rscrypto_p384_montjscalarmul_alt_x86_64_unknown_linux.S`. It is not folded into the full
Linux release metrics above because it is a scoped quick run, but it is the decision record for the
s2n-bignum deletion slice.

Run [#28216163852](https://github.com/loadingalias/rscrypto/actions/runs/28216163852) is the follow-up
after specializing the owned fixed-base P-384 signing path on x86_64.

Run [#28245803704](https://github.com/loadingalias/rscrypto/actions/runs/28245803704) is the follow-up
after deleting `src/auth/asm/rscrypto_bignum_mod_n384_x86_64_unknown_linux.S` and routing x86_64
P-384 nonce reduction through the owned reducer.

| Scope | Rows | W/T/L | Geomean | Median |
| --- | ---: | ---: | ---: | ---: |
| Pre-delete x86_64 P-384 sign rows from run #28202726811 | 16 | 0/0/16 | 0.53x | 0.52x |
| Post-delete x86_64 P-384 sign rows from run #28212427930 | 16 | 0/0/16 | 0.70x | 0.68x |
| Post-delete x86_64 P-384 verify guard rows from run #28212427930 | 16 | 16/0/0 | 1.33x | 1.34x |
| Post-specialization x86_64 P-384 sign rows from run #28216163852 | 16 | 16/0/0 | 1.38x | 1.40x |
| Post-reduction-cleanup x86_64 P-384 sign rows from run #28245803704 | 16 | 16/0/0 | 1.34x | 1.35x |
| Blended estimate if non-x86 rows remain unchanged | 36 | 16/4/16 | 1.08x | 0.98x |

| Platform | Pre-delete sign geomean | Post-delete sign geomean | Post-specialization sign geomean | Post-reduction-cleanup sign geomean |
| --- | ---: | ---: | ---: | ---: |
| AMD Zen4 | 0.53x | 0.70x | 1.38x | 1.33x |
| AMD Zen5 | 0.57x | 0.66x | 1.41x | 1.38x |
| Intel Ice Lake | 0.53x | 0.80x | 1.40x | 1.36x |
| Intel Sapphire Rapids | 0.51x | 0.66x | 1.31x | 1.28x |

Verdict: the deletion removed one vendored s2n-bignum assembly file and materially improved the worst
P-384 signing rows; the later owned fixed-base specialization closed the x86_64 AWS-LC gap. Removing the
x86_64 P-384 order-reduction helper costs about 3% of the signing ratio geomean but keeps every targeted
row ahead of AWS-LC. Keep both deletions, and remove narrower live s2n-bignum helpers only when the
signing margin stays intact.

## Primitive Summary

Linux CI primitives with matched exact `rscrypto` comparisons. Fastest columns are strongest-external comparisons; all-pair columns include every matched external implementation.

| Primitive | Fastest rows | Fastest W/T/L | Fastest geomean | All pairs | All W/T/L | All geomean |
| --- | --- | --- | --- | --- | --- | --- |
| `argon2id-owasp` | 9 | 2/1/6 | 0.95x | 18 | 9/2/7 | 1.24x |
| `chacha20-poly1305` | 198 | 64/65/69 | 1.00x | 550 | 292/140/118 | 1.16x |
| `argon2i-small` | 27 | 12/3/12 | 1.02x | 45 | 29/3/13 | 1.35x |
| `argon2id-small` | 27 | 13/3/11 | 1.03x | 45 | 30/3/12 | 1.36x |
| `x25519` | 18 | 4/13/1 | 1.03x | 50 | 35/13/2 | 1.53x |
| `argon2d-small` | 27 | 13/5/9 | 1.04x | 27 | 13/5/9 | 1.04x |
| `rapidhash-v3-64` | 99 | 28/51/20 | 1.04x | 99 | 28/51/20 | 1.04x |
| `blake2b256` | 225 | 118/106/1 | 1.07x | 351 | 237/113/1 | 1.32x |
| `rapidhash-v3-128` | 99 | 31/49/19 | 1.08x | 99 | 31/49/19 | 1.08x |
| `blake2b512` | 198 | 114/82/2 | 1.08x | 297 | 210/85/2 | 1.33x |
| `ed25519` | 90 | 32/43/15 | 1.10x | 290 | 200/59/31 | 1.35x |
| `blake2s128` | 198 | 117/77/4 | 1.10x | 198 | 117/77/4 | 1.10x |
| `blake2s256` | 225 | 115/109/1 | 1.11x | 225 | 115/109/1 | 1.11x |
| `scrypt-owasp` | 9 | 5/2/2 | 1.11x | 9 | 5/2/2 | 1.11x |
| `rapidhash-64` | 99 | 32/55/12 | 1.13x | 99 | 32/55/12 | 1.13x |
| `rapidhash-128` | 99 | 37/54/8 | 1.14x | 99 | 37/54/8 | 1.14x |
| `xxh3-128` | 99 | 51/35/13 | 1.17x | 99 | 51/35/13 | 1.17x |
| `xxh3-64` | 99 | 53/34/12 | 1.19x | 99 | 53/34/12 | 1.19x |
| `scrypt-small` | 36 | 22/14/0 | 1.20x | 36 | 22/14/0 | 1.20x |
| `rsa-8192` | 18 | 14/2/2 | 1.21x | 32 | 28/2/2 | 1.27x |
| `hkdf-sha256` | 36 | 30/6/0 | 1.22x | 100 | 94/6/0 | 1.85x |
| `hmac-sha512` | 99 | 42/49/8 | 1.23x | 275 | 171/91/13 | 1.28x |
| `sha256` | 117 | 42/59/16 | 1.23x | 293 | 161/103/29 | 1.53x |
| `hmac-sha384` | 99 | 43/49/7 | 1.23x | 275 | 171/92/12 | 1.29x |
| `pbkdf2-sha512` | 54 | 47/7/0 | 1.24x | 150 | 143/7/0 | 1.31x |
| `sha512` | 117 | 45/58/14 | 1.24x | 293 | 170/103/20 | 1.26x |
| `pbkdf2-sha256` | 54 | 50/3/1 | 1.24x | 150 | 146/3/1 | 1.64x |
| `sha384` | 99 | 43/45/11 | 1.24x | 275 | 170/89/16 | 1.28x |
| `hkdf-sha384` | 36 | 35/1/0 | 1.26x | 100 | 99/1/0 | 1.57x |
| `ascon-hash256` | 99 | 55/44/0 | 1.26x | 99 | 55/44/0 | 1.26x |
| `sha512-256` | 99 | 56/43/0 | 1.28x | 99 | 56/43/0 | 1.28x |
| `ecdsa-p384` | 72 | 52/4/16 | 1.31x | 200 | 180/4/16 | 2.96x |
| `ascon-aead128` | 198 | 146/52/0 | 1.33x | 198 | 146/52/0 | 1.33x |
| `hmac-sha256` | 117 | 65/38/14 | 1.34x | 293 | 195/74/24 | 1.69x |
| `ascon-xof128` | 99 | 73/25/1 | 1.35x | 99 | 73/25/1 | 1.35x |
| `aegis-256` | 198 | 93/80/25 | 1.37x | 198 | 93/80/25 | 1.37x |
| `mlkem512` | 27 | 20/4/3 | 1.39x | 102 | 95/4/3 | 2.68x |
| `blake3` | 432 | 225/179/28 | 1.40x | 432 | 225/179/28 | 1.40x |
| `xchacha20-poly1305` | 198 | 173/20/5 | 1.40x | 198 | 173/20/5 | 1.40x |
| `mlkem768` | 27 | 22/2/3 | 1.51x | 102 | 97/2/3 | 3.09x |
| `ecdsa-p256` | 72 | 63/9/0 | 1.52x | 200 | 187/13/0 | 2.42x |
| `mlkem1024` | 27 | 21/2/4 | 1.57x | 102 | 96/2/4 | 3.29x |
| `rsa-4096` | 27 | 24/1/2 | 1.58x | 59 | 56/1/2 | 2.53x |
| `crc32c` | 99 | 47/38/14 | 1.62x | 198 | 143/41/14 | 2.22x |
| `crc32` | 99 | 50/34/15 | 1.65x | 198 | 144/38/16 | 2.24x |
| `rsa-3072` | 27 | 25/0/2 | 1.66x | 59 | 57/0/2 | 2.58x |
| `rsa-2048` | 27 | 25/0/2 | 1.72x | 59 | 57/0/2 | 2.65x |
| `kmac256` | 99 | 62/22/15 | 1.73x | 99 | 62/22/15 | 1.73x |
| `aes-128-gcm` | 198 | 119/42/37 | 1.77x | 550 | 435/49/66 | 2.54x |
| `cshake256` | 99 | 62/27/10 | 1.79x | 99 | 62/27/10 | 1.79x |
| `aes-256-gcm` | 198 | 115/38/45 | 1.79x | 550 | 426/47/77 | 2.58x |
| `sha224` | 99 | 49/47/3 | 1.84x | 99 | 49/47/3 | 1.84x |
| `shake128` | 99 | 71/28/0 | 1.84x | 99 | 71/28/0 | 1.84x |
| `shake256` | 99 | 70/29/0 | 1.86x | 99 | 70/29/0 | 1.86x |
| `aes-128-gcm-siv` | 198 | 136/5/57 | 2.00x | 352 | 269/19/64 | 2.77x |
| `sha3-256` | 117 | 102/15/0 | 2.10x | 117 | 102/15/0 | 2.10x |
| `sha3-224` | 99 | 86/12/1 | 2.10x | 99 | 86/12/1 | 2.10x |
| `aes-256-gcm-siv` | 198 | 138/4/56 | 2.13x | 352 | 291/5/56 | 3.02x |
| `crc64-nvme` | 99 | 57/31/11 | 2.13x | 99 | 57/31/11 | 2.13x |
| `sha3-384` | 99 | 88/11/0 | 2.15x | 99 | 88/11/0 | 2.15x |
| `sha3-512` | 99 | 88/11/0 | 2.19x | 99 | 88/11/0 | 2.19x |
| `crc64-xz` | 99 | 82/13/4 | 2.54x | 99 | 82/13/4 | 2.54x |
| `crc24-openpgp` | 99 | 93/3/3 | 13.15x | 99 | 93/3/3 | 13.15x |
| `crc16-ccitt` | 99 | 98/0/1 | 23.12x | 99 | 98/0/1 | 23.12x |
| `crc16-ibm` | 99 | 98/0/1 | 24.07x | 99 | 98/0/1 | 24.07x |

## Linux Worst Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| IBM z16/s390x | `rapidhash-v3-64 / 0` | `rapidhash` | 0.45x |
| RISE RISC-V | `xxh3-64 / 0` | `xxhash-rust` | 0.46x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.48x |
| Intel Ice Lake | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.48x |
| Intel Ice Lake | `aes-128-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `aes-128-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 32` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 1024` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `aes-128-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 0` | `aws-lc-rs` | 0.49x |
| Intel Sapphire Rapids | `ecdsa-p384 / sign / 32` | `aws-lc-rs` | 0.50x |
| Intel Sapphire Rapids | `aes-256-gcm-siv / encrypt / 0` | `aws-lc-rs` | 0.50x |

## Linux Strongest Individual Rows

| Platform | Case | Fastest external | Ratio |
| --- | --- | --- | --- |
| Intel Sapphire Rapids | `crc16-ccitt / 16384` | `crc` | 228.69x |
| Intel Sapphire Rapids | `crc16-ccitt / 262144` | `crc` | 211.31x |
| Intel Sapphire Rapids | `crc16-ibm / 16384` | `crc` | 206.65x |
| Intel Sapphire Rapids | `crc16-ibm / 262144` | `crc` | 206.14x |
| Intel Sapphire Rapids | `crc16-ibm / 65536` | `crc` | 184.09x |
| Intel Sapphire Rapids | `crc16-ibm / 1048576` | `crc` | 184.03x |
| Intel Sapphire Rapids | `crc16-ccitt / 65536` | `crc` | 180.12x |
| Intel Sapphire Rapids | `crc16-ccitt / 1048576` | `crc` | 179.62x |
| IBM Power10 | `crc16-ccitt / 1048576` | `crc` | 175.75x |
| IBM Power10 | `crc16-ibm / 262144` | `crc` | 174.98x |
| IBM Power10 | `crc16-ccitt / 262144` | `crc` | 174.43x |
| IBM Power10 | `crc16-ibm / 1048576` | `crc` | 174.07x |

## Top Five Loss Areas

- `argon2id-owasp` / `hash`: 0.95x geomean across 9 rows; W/T/L 2/1/6; pressure `rustcrypto` 6, `dryoc` 1.
- `mlkem1024` / `keygen`: 0.95x geomean across 9 rows; W/T/L 3/2/4; pressure `aws-lc-rs` 3, `libcrux` 1.
- `ecdsa-p384` / `sign`: 0.96x geomean across 36 rows; W/T/L 16/4/16; pressure `aws-lc-rs` 16.
- `ed25519` / `verify`: 0.98x geomean across 36 rows; W/T/L 8/16/12; pressure `ring` 9, `dalek` 7, `dryoc` 7, `aws-lc-rs` 4.
- `mlkem768` / `keygen`: 0.98x geomean across 9 rows; W/T/L 4/2/3; pressure `aws-lc-rs` 3.

The P-384 signing line above is from the 2026-06-25 full nine-runner release
matrix. The scoped x86_64 follow-up in run
[#28245803704](https://github.com/loadingalias/rscrypto/actions/runs/28245803704)
kept the x86_64 P-384 signing rows at W/T/L 16/0/0, 1.34x geomean vs AWS-LC after
the x86_64 P-384 order-reduction assembly deletion.

## External Pressure

| External | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| `rapidhash` | 396 | 128/209/59 | 32% | 1.10x | 1.01x |
| `xxhash-rust` | 198 | 104/69/25 | 53% | 1.18x | 1.07x |
| `aws-lc-rs` | 1,673 | 1,031/335/307 | 62% | 1.23x | 1.14x |
| `ascon-hash` | 198 | 128/69/1 | 65% | 1.31x | 1.17x |
| `ascon-aead` | 198 | 146/52/0 | 74% | 1.33x | 1.15x |
| `aegis-crate` | 198 | 93/80/25 | 47% | 1.37x | 1.04x |
| `blake3` | 432 | 225/179/28 | 52% | 1.40x | 1.06x |
| `dalek` | 108 | 83/16/9 | 77% | 1.45x | 1.27x |
| `sha2` | 531 | 284/241/6 | 53% | 1.51x | 1.06x |
| `ring` | 1,656 | 1,302/203/151 | 79% | 1.62x | 1.28x |
| `libcrux` | 81 | 75/5/1 | 93% | 1.80x | 1.75x |
| `tiny-keccak` | 396 | 265/106/25 | 67% | 1.80x | 1.99x |
| `dryoc` | 360 | 324/24/12 | 90% | 1.81x | 1.85x |
| `crc-fast` | 297 | 172/97/28 | 58% | 2.00x | 1.16x |
| `rustcrypto` | 2,889 | 2,125/628/136 | 74% | 2.09x | 1.21x |
| `sha3` | 414 | 364/49/1 | 88% | 2.13x | 2.12x |
| `crc32fast` | 99 | 84/7/8 | 85% | 2.41x | 1.99x |
| `crc64fast` | 99 | 82/13/4 | 83% | 2.54x | 2.02x |
| `crc32c` | 99 | 88/6/5 | 89% | 2.73x | 2.17x |
| `fips203` | 81 | 81/0/0 | 100% | 4.66x | 5.50x |
| `rustcrypto-rsa` | 81 | 81/0/0 | 100% | 5.74x | 4.83x |
| `crc` | 297 | 289/3/5 | 97% | 19.42x | 29.79x |

## macOS Local Snapshot

The macOS Apple Silicon run is local evidence for the earlier 2026-06-22 baseline commit. It is useful for Apple Silicon planning but is not folded into Linux CI release claims. The ML-KEM row reflects the newer local `just bench-quick ml-kem` public API run from this workspace.

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
| --- | --- | --- | --- | --- | --- |
| macOS local: all matched performance pairs | 1,309 | 828/435/46 | 63% | 1.81x | 1.15x |
| macOS local: fastest external per case | 786 | 386/362/38 | 49% | 1.37x | 1.05x |
| macOS local: ML-KEM fastest external | 9 | 7/2/0 | 78% | 1.46x | 1.43x |

## README Numbers

- **Headline:** 4,103 of 6,750 matched Linux CI fastest-external comparisons are wins; 6,096 are wins or ties. Linux CI geomean is 1.59x.
- **Checksums:** 5.22x geomean across 693 Linux CI fastest-external rows; W/T/L 525/119/49.
- **Hashes/MACs/XOFs:** 1.35x geomean across 3,726 Linux CI fastest-external rows; W/T/L 2,063/1,443/220.
- **Auth/KDF:** 1.24x geomean across 180 Linux CI fastest-external rows; W/T/L 162/17/1.
- **Password hashing:** 1.07x geomean across 135 Linux CI fastest-external rows; W/T/L 67/28/40.
- **Public-key:** 1.31x geomean across 333 Linux CI fastest-external rows; W/T/L 214/77/42.
- **RSA:** 1.56x geomean across 99 Linux CI fastest-external rows; W/T/L 88/3/8.
- **AEAD:** 1.56x geomean across 1,584 Linux CI fastest-external rows; W/T/L 984/306/294.
- **ML-KEM:** 1.49x geomean across 81 Linux CI fastest-external rows; W/T/L 63/8/10.
- **ECDSA P-256/P-384:** 1.42x Linux CI geomean across 144 fastest-external rows.
- **Current top losses:** `argon2id-owasp` / `hash`: 0.95x geomean across 9 rows; W/T/L 2/1/6; pressure `rustcrypto` 6, `dryoc` 1; `mlkem1024` / `keygen`: 0.95x geomean across 9 rows; W/T/L 3/2/4; pressure `aws-lc-rs` 3, `libcrux` 1; `ecdsa-p384` / `sign`: 0.96x geomean across 36 rows; W/T/L 16/4/16; pressure `aws-lc-rs` 16; `ed25519` / `verify`: 0.98x geomean across 36 rows; W/T/L 8/16/12; pressure `ring` 9, `dalek` 7, `dryoc` 7, `aws-lc-rs` 4; `mlkem768` / `keygen`: 0.98x geomean across 9 rows; W/T/L 4/2/3; pressure `aws-lc-rs` 3.
- **Targeted P-384 follow-up:** x86_64 P-384 signing is W/T/L 16/0/0, 1.34x geomean vs AWS-LC after the order-reduction cleanup in run #28245803704; this is a scoped follow-up, not part of the full-release totals above.

## Raw Results

| Platform | Mode | Date/time | Parsed rows | Result |
| --- | --- | --- | --- | --- |
| AMD Zen4 | `ci` | `2026-06-25 21_50_27` | 2,269 | `benchmark_results/2026-06-25/linux/amd-zen4/results.txt` |
| AMD Zen5 | `ci` | `2026-06-25 21_50_27` | 2,269 | `benchmark_results/2026-06-25/linux/amd-zen5/results.txt` |
| AWS Graviton3 | `ci` | `2026-06-25 21_50_27` | 2,275 | `benchmark_results/2026-06-25/linux/graviton3/results.txt` |
| AWS Graviton4 | `ci` | `2026-06-25 21_50_27` | 2,275 | `benchmark_results/2026-06-25/linux/graviton4/results.txt` |
| IBM Power10 | `ci` | `2026-06-25 21_50_27` | 2,020 | `benchmark_results/2026-06-25/linux/ibm-power10/results.txt` |
| IBM z16/s390x | `ci` | `2026-06-25 21_50_27` | 2,020 | `benchmark_results/2026-06-25/linux/ibm-s390x/results.txt` |
| Intel Ice Lake | `ci` | `2026-06-25 21_50_27` | 2,269 | `benchmark_results/2026-06-25/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `ci` | `2026-06-25 21_50_27` | 2,269 | `benchmark_results/2026-06-25/linux/intel-spr/results.txt` |
| RISE RISC-V | `ci` | `2026-06-25 21_50_27` | 2,269 | `benchmark_results/2026-06-25/linux/rise-riscv/results.txt` |
| AMD Zen4 P-384 follow-up | `ci` | `2026-06-26 02_02_44` | 36 | `benchmark_results/2026-06-26/linux/amd-zen4/results.txt` |
| AMD Zen5 P-384 follow-up | `ci` | `2026-06-26 02_02_44` | 36 | `benchmark_results/2026-06-26/linux/amd-zen5/results.txt` |
| Intel Ice Lake P-384 follow-up | `ci` | `2026-06-26 02_02_44` | 36 | `benchmark_results/2026-06-26/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids P-384 follow-up | `ci` | `2026-06-26 02_02_44` | 36 | `benchmark_results/2026-06-26/linux/intel-spr/results.txt` |
| macOS Apple Silicon | `local` | `2026-06-22 15_11_40` | 2,277 | `benchmark_results/2026-06-22/macos/aarch64/results.txt` |
