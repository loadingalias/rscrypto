# Benchmark Overview

Sources:

- Linux x86-64 AEAD bench CI run [#25607157508](https://github.com/loadingalias/rscrypto/actions/runs/25607157508), created 2026-05-09 17:23:10 UTC.
- Imported x86-64 files: `benchmark_results/2026-05-09/linux/{amd-zen4,amd-zen5,intel-icl,intel-spr}/results.txt`.
- Run status at extraction time: AMD Zen4, AMD Zen5, Intel Ice Lake, and Intel Sapphire Rapids completed successfully. IBM Z s390x and RISE RISC-V were still running. Graviton and POWER artifacts from this run are intentionally excluded here because this update is x86-64 only.
- Older non-x86 files under `benchmark_results/2026-05-09/linux/` from commit `56f75c5f8679325bc6eba89cdf69d5cf5f050ded` are not part of this overview.

Linux x86-64 commit: `18187854effdc0d56ad7d5f627ff6ec622c62c3c`.

Scope: AEAD only, x86-64 Linux only. Ratios are `rscrypto` throughput divided by external throughput. Above `1.00x` means `rscrypto` is faster. Wins are `>1.05x`, ties are `0.95x..1.05x`, losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, operation, primitive, and input size.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Linux x86-64, all external pairs | 1200 | 718/158/324 | 60% | 1.29x | 1.12x |
| Linux x86-64, fastest external per case | 560 | 219/98/243 | 39% | 0.97x | 1.00x |

What matters:

- AES-GCM is improved but not closed. Latest x86-64 sustained AES-GCM rows are **0/0/48** against the fastest external implementation, with **0.79x** geomean and **0.79x** median.
- This is still a real step forward. On overlapping platforms versus the previous extracted run, Zen4 AES-GCM operation geomeans improved roughly `14%..45%`, Ice Lake seal improved roughly `25%..33%`, and Sapphire Rapids AES-256 improved roughly `7%..13%`.
- Zen5 is no longer missing. It lands in the same losing band as the other x86 runners: sustained AES-GCM operation geomeans are `0.74x..0.77x`.
- The remaining x86 AES-GCM gap is still against `aws-lc-rs` on every sustained row. The current row range is `0.72x..0.90x`, so parity needs roughly `11%..38%` more throughput depending on row; the `>=1.05x` noise-resistant target needs more.
- GCM-SIV is mixed but much healthier: x86-64 sustained GCM-SIV rows are **26/5/17** at **1.11x** geomean. AMD is clean. Intel open remains below the gate.

## Required Sustained Rows

Fastest-external comparisons only for AES-GCM and GCM-SIV, 128-bit and 256-bit keys, encrypt and decrypt, input sizes `65536`, `262144`, and `1048576`.

| Platform | AES-GCM W/T/L | AES-GCM geomean | GCM-SIV W/T/L | GCM-SIV geomean |
|---|---:|---:|---:|---:|
| AMD Zen4 | 0/0/12 | 0.79x | 12/0/0 | 1.27x |
| AMD Zen5 | 0/0/12 | 0.76x | 12/0/0 | 1.55x |
| Intel Ice Lake | 0/0/12 | 0.80x | 0/3/9 | 0.84x |
| Intel Sapphire Rapids | 0/0/12 | 0.83x | 2/2/8 | 0.92x |

## AES-GCM Sustained Throughput

Ratios are geometric means across `65536`, `262144`, and `1048576` byte rows.

| Platform | AES-128 seal | AES-128 open | AES-256 seal | AES-256 open |
|---|---:|---:|---:|---:|
| AMD Zen4 | 0.78 | 0.77 | 0.81 | 0.80 |
| AMD Zen5 | 0.74 | 0.77 | 0.76 | 0.76 |
| Intel Ice Lake | 0.81 | 0.78 | 0.82 | 0.78 |
| Intel Sapphire Rapids | 0.77 | 0.80 | 0.86 | 0.87 |

## AES-GCM-SIV Sustained Throughput

Ratios are geometric means across `65536`, `262144`, and `1048576` byte rows.

| Platform | AES-128 seal | AES-128 open | AES-256 seal | AES-256 open |
|---|---:|---:|---:|---:|
| AMD Zen4 | 1.32 | 1.12 | 1.49 | 1.19 |
| AMD Zen5 | 1.67 | 1.30 | 1.91 | 1.39 |
| Intel Ice Lake | 0.83 | 0.74 | 0.97 | 0.84 |
| Intel Sapphire Rapids | 0.91 | 0.79 | 1.06 | 0.93 |

## Linux x86-64 AEAD Detail

| Primitive | All pairs | W/T/L | Geomean | Fastest-only pairs | W/T/L | Geomean |
|---|---:|---:|---:|---:|---:|---:|
| AES-128-GCM | 240 | 162/24/54 | 1.54x | 80 | 30/9/41 | 0.96x |
| AES-256-GCM | 240 | 156/23/61 | 1.51x | 80 | 25/11/44 | 0.95x |
| AES-128-GCM-SIV | 160 | 76/17/67 | 1.22x | 80 | 18/2/60 | 0.83x |
| AES-256-GCM-SIV | 160 | 100/8/52 | 1.31x | 80 | 21/7/52 | 0.87x |
| ChaCha20-Poly1305 | 240 | 117/34/89 | 0.99x | 80 | 18/17/45 | 0.82x |
| XChaCha20-Poly1305 | 80 | 70/10/0 | 1.33x | 80 | 70/10/0 | 1.33x |
| AEGIS-256 | 80 | 37/42/1 | 1.12x | 80 | 37/42/1 | 1.12x |

## External Competitor Pressure

Linux x86-64 all-pair comparisons. This answers which external implementations are applying pressure in the current AEAD run.

| External | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| `rustcrypto` | 480 | 421/35/24 | 88% | 1.94x | 1.53x |
| `aws-lc-rs` | 400 | 130/44/226 | 33% | 0.93x | 0.90x |
| `ring` | 240 | 130/37/73 | 54% | 1.03x | 1.06x |
| `aegis-crate` | 80 | 37/42/1 | 46% | 1.12x | 1.05x |

## Raw Results

| Runner | Result |
|---|---|
| AMD Zen4 | `benchmark_results/2026-05-09/linux/amd-zen4/results.txt` |
| AMD Zen5 | `benchmark_results/2026-05-09/linux/amd-zen5/results.txt` |
| Intel Ice Lake | `benchmark_results/2026-05-09/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `benchmark_results/2026-05-09/linux/intel-spr/results.txt` |

## Methodology Notes

- Parsed 1760 Criterion throughput rows from the four current x86-64 result files.
- Every current x86-64 artifact has commit `18187854effdc0d56ad7d5f627ff6ec622c62c3c` and 440 parsed AEAD throughput rows.
- A comparison is matched when a Criterion ID has a `rscrypto` path component and an external ID with the same platform, primitive, operation, and input size, differing only in the implementation component.
- This overview intentionally does not merge old non-x86 result files with the current x86-64 run. Use the complete CI matrix after IBM Z and RISC-V finish if a full-platform release claim is needed.
