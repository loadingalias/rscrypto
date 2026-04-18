# Benchmarks

`rscrypto` is measured against established Rust crates for every algorithm it ships. Benchmarks run on 9 production-grade CI platforms spanning x86-64, aarch64, s390x, ppc64le, and riscv64.

**Latest run:** [`#24611633843`](https://github.com/loadingalias/rscrypto/actions/runs/24611633843) &middot; 2026-04-18 &middot; commit `b3fc2bb29b95`

## Headline

**2693W / 704T / 230L = 74% win rate** across 3627 size-indexed comparisons.

- **WIN**: speedup >= `1.03x`. **TIE**: `0.97x`--`1.03x`. **LOSS**: < `0.97x`.
- Speedup = competitor_time / rscrypto_time (median of Criterion's `[low mid high]`).

## By Category

| Category | W | T | L | Total | Win % |
|----------|--:|--:|--:|------:|------:|
| Checksums | 678 | 69 | 63 | 810 | 84% |
| SHA-2 | 260 | 125 | 20 | 405 | 64% |
| SHA-3 | 303 | 21 | 0 | 324 | 94% |
| SHAKE | 160 | 2 | 0 | 162 | 99% |
| Ascon | 58 | 95 | 9 | 162 | 36% |
| Blake3 | 192 | 78 | 54 | 324 | 59% |
| XXH3 | 82 | 67 | 13 | 162 | 51% |
| RapidHash | 7 | 70 | 4 | 81 | 9% |
| Auth | 285 | 73 | 29 | 387 | 74% |
| AEAD | 668 | 104 | 38 | 810 | 82% |

## By Platform

| Platform | Arch | W | T | L | Total | Win % |
|----------|------|--:|--:|--:|------:|------:|
| Zen4 | x86-64 | 323 | 50 | 3 | 376 | 86% |
| Zen5 | x86-64 | 295 | 62 | 19 | 376 | 78% |
| SPR | x86-64 | 304 | 57 | 15 | 376 | 81% |
| ICL | x86-64 | 300 | 65 | 11 | 376 | 80% |
| Grav3 | aarch64 | 260 | 88 | 28 | 376 | 69% |
| Grav4 | aarch64 | 258 | 90 | 28 | 376 | 69% |
| s390x | s390x | 324 | 37 | 15 | 376 | 86% |
| POWER10 | ppc64le | 238 | 119 | 19 | 376 | 63% |
| RISE | riscv64 | 220 | 80 | 76 | 376 | 59% |

## Scope

- **Checksums:** CRC-16 (CCITT / IBM), CRC-24 (OpenPGP), CRC-32 (IEEE / Castagnoli), CRC-64 (XZ / NVMe).
- **Hashes:** SHA-2 (224/256/384/512/512-256), SHA-3 (224/256/384/512), SHAKE (128/256), Blake3 (hash/keyed/derive-key/xof), Ascon-Hash256, Ascon-Xof128, XXH3-64, XXH3-128, RapidHash-64.
- **Authentication:** HMAC (SHA-256/384/512), HKDF (SHA-256/384), Ed25519 (sign/verify).
- **AEAD:** AES-256-GCM, AES-256-GCM-SIV, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256.

## Detailed Tables

Per-algorithm, per-platform, per-size speedup tables live in [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md). Raw Criterion output per platform is under [`benchmark_results/$DATE/linux/$ARCH/results.txt`](benchmark_results/).

## Reproducing

```
just bench               # local full sweep
just bench-native        # with -C target-cpu=native
```

CI benchmarks run on dedicated self-hosted and cloud runners; see `.github/workflows/bench.yml`.
