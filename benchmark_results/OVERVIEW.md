# Benchmark Overview

Sources:

- Linux: AEAD bench CI run [#25528038273](https://github.com/loadingalias/rscrypto/actions/runs/25528038273), created 2026-05-07 23:38:35 UTC.
- Linux status: 8 of 9 benchmark runners produced current artifacts. AMD Zen5 job [#74928027097](https://github.com/loadingalias/rscrypto/actions/runs/25528038273/job/74928027097) failed in `Setup`; its `Run` and artifact steps were skipped.
- macOS: local AEAD proof run `benchmark_results/2026-05-07/macos/aarch64/results.txt`, created 2026-05-07 12:03:09 local time.
- Current Linux raw files: `benchmark_results/2026-05-07/linux/{amd-zen4,graviton3,graviton4,ibm-power10,ibm-s390x,intel-icl,intel-spr,rise-riscv}/results.txt`.

Linux commit: `ff3060e2c0a696884e531a6c18616df828457a5a`.
macOS proof commit: `cc9c8f84db7fd13985b720afce34228134b9f982`.

Scope: AEAD only. Linux CI is the platform-matrix source of truth; this run is partial because Zen5 never reached the benchmark step. The existing `benchmark_results/2026-05-07/linux/amd-zen5/results.txt` file is from stale commit `d592060512f9f627d20a602bc8d7f216806e29b2` and is excluded from every current table below.

Ratios are `rscrypto` throughput divided by external throughput. Above `1.00x` means `rscrypto` is faster. Wins are `>1.05x`, ties are `0.95x..1.05x`, losses are `<0.95x`. Fastest-external comparisons keep only the fastest external implementation for each platform, operation, primitive, and input size.

## Headline

| Scope | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| Current Linux, all external pairs | 2400 | 1476/279/645 | 62% | 1.76x | 1.25x |
| Current Linux, fastest external per case | 1120 | 521/156/443 | 47% | 1.09x | 1.02x |
| macOS local, all external pairs | 300 | 219/16/65 | 73% | 2.42x | 1.28x |
| macOS local, fastest external per case | 140 | 92/5/43 | 66% | 1.09x | 1.17x |
| Current Linux + macOS local, all external pairs | 2700 | 1695/295/710 | 63% | 1.82x | 1.27x |
| Current Linux + macOS local, fastest external per case | 1260 | 613/161/486 | 49% | 1.09x | 1.04x |

What matters:

- Clean-sweep status: not done. Current Linux sustained required rows are **67/5/120** against the fastest external implementation at **1.22x** geomean, with Zen5 missing.
- AES-GCM is still the blocker. Current Linux sustained AES-GCM rows are **12/0/84** at **0.92x** geomean and **0.69x** median; only s390x sweeps the sustained AES-GCM rows.
- The x86 pass helped Intel, especially Sapphire Rapids, but it did not close the gate. Ice Lake is `0.61x..0.78x`; Sapphire Rapids is `0.75x..0.83x`.
- AMD is not closed. Zen4 is `0.56x..0.68x` on sustained AES-GCM in this snapshot, and Zen5 needs a rerun because the current job failed before benches.
- GCM-SIV is strong but not clean. Zen4, Graviton, macOS, and s390x sweep sustained GCM-SIV; Intel open, POWER10 open, and RISC-V remain below the gate.

## Required Sustained Rows

Fastest-external comparisons only for AES-GCM and GCM-SIV, 128-bit and 256-bit keys, encrypt and decrypt, input sizes `65536`, `262144`, and `1048576`.

| Platform | AES-GCM W/T/L | AES-GCM geomean | GCM-SIV W/T/L | GCM-SIV geomean |
|---|---:|---:|---:|---:|
| AMD Zen4 | 0/0/12 | 0.61x | 12/0/0 | 1.29x |
| AMD Zen5 | missing | missing | missing | missing |
| Intel Ice Lake | 0/0/12 | 0.69x | 0/3/9 | 0.85x |
| Intel Sapphire Rapids | 0/0/12 | 0.79x | 1/2/9 | 0.91x |
| AWS Graviton3 | 0/0/12 | 0.64x | 12/0/0 | 2.53x |
| AWS Graviton4 | 0/0/12 | 0.65x | 12/0/0 | 2.53x |
| macOS AArch64 | 0/0/12 | 0.83x | 12/0/0 | 2.71x |
| IBM POWER10 | 0/0/12 | 0.37x | 6/0/6 | 1.46x |
| IBM Z / s390x | 12/0/0 | 12.21x | 12/0/0 | 5.56x |
| RISE RISC-V | 0/0/12 | 0.83x | 0/0/12 | 0.90x |

## AES-GCM Sustained Throughput

Ratios are geometric means across `65536`, `262144`, and `1048576` byte rows.

| Platform | AES-128 seal | AES-128 open | AES-256 seal | AES-256 open |
|---|---:|---:|---:|---:|
| AMD Zen4 | 0.59 | 0.68 | 0.56 | 0.60 |
| AMD Zen5 | missing | missing | missing | missing |
| Intel Ice Lake | 0.61 | 0.78 | 0.65 | 0.74 |
| Intel Sapphire Rapids | 0.75 | 0.83 | 0.76 | 0.82 |
| AWS Graviton3 | 0.68 | 0.59 | 0.70 | 0.58 |
| AWS Graviton4 | 0.70 | 0.63 | 0.68 | 0.58 |
| macOS AArch64 | 0.78 | 0.77 | 0.89 | 0.88 |
| IBM POWER10 | 0.36 | 0.36 | 0.37 | 0.37 |
| IBM Z / s390x | 10.92 | 11.74 | 13.14 | 13.23 |
| RISE RISC-V | 0.80 | 0.78 | 0.87 | 0.85 |

## AES-GCM-SIV Sustained Throughput

Ratios are geometric means across `65536`, `262144`, and `1048576` byte rows.

| Platform | AES-128 seal | AES-128 open | AES-256 seal | AES-256 open |
|---|---:|---:|---:|---:|
| AMD Zen4 | 1.32 | 1.12 | 1.54 | 1.21 |
| AMD Zen5 | missing | missing | missing | missing |
| Intel Ice Lake | 0.83 | 0.74 | 0.97 | 0.85 |
| Intel Sapphire Rapids | 0.90 | 0.78 | 1.05 | 0.91 |
| AWS Graviton3 | 2.42 | 2.39 | 2.71 | 2.62 |
| AWS Graviton4 | 2.46 | 2.31 | 2.77 | 2.62 |
| macOS AArch64 | 2.58 | 2.67 | 2.76 | 2.82 |
| IBM POWER10 | 2.40 | 0.85 | 2.51 | 0.88 |
| IBM Z / s390x | 5.18 | 5.24 | 5.81 | 6.06 |
| RISE RISC-V | 0.91 | 0.90 | 0.89 | 0.91 |

## Linux AEAD Detail

| Primitive | All pairs | W/T/L | Geomean | Fastest-only pairs | W/T/L | Geomean |
|---|---:|---:|---:|---:|---:|---:|
| AES-128-GCM | 480 | 307/22/151 | 2.11x | 160 | 61/5/94 | 1.01x |
| AES-256-GCM | 480 | 291/34/155 | 2.05x | 160 | 55/12/93 | 0.96x |
| AES-128-GCM-SIV | 320 | 224/15/81 | 2.33x | 160 | 83/3/74 | 1.14x |
| AES-256-GCM-SIV | 320 | 239/12/69 | 2.53x | 160 | 82/11/67 | 1.16x |
| ChaCha20-Poly1305 | 480 | 224/91/165 | 0.99x | 160 | 49/20/91 | 0.83x |
| XChaCha20-Poly1305 | 160 | 108/49/3 | 1.27x | 160 | 108/49/3 | 1.27x |
| AEGIS-256 | 160 | 83/56/21 | 1.35x | 160 | 83/56/21 | 1.35x |

## External Competitor Pressure

Linux all-pair comparisons. This answers which external implementations are applying pressure in the current AEAD run.

| External | Pairs | W/T/L | Win % | Geomean | Median |
|---|---:|---:|---:|---:|---:|
| `rustcrypto` | 960 | 749/116/95 | 78% | 2.67x | 1.69x |
| `aws-lc-rs` | 800 | 414/35/351 | 52% | 1.34x | 1.11x |
| `ring` | 480 | 230/72/178 | 48% | 1.30x | 1.04x |
| `aegis-crate` | 160 | 83/56/21 | 52% | 1.35x | 1.06x |

## Raw Results

| Runner | Result |
|---|---|
| AMD Zen4 | `benchmark_results/2026-05-07/linux/amd-zen4/results.txt` |
| AMD Zen5 | missing from run `25528038273`; stale `d592060` file excluded |
| Intel Ice Lake | `benchmark_results/2026-05-07/linux/intel-icl/results.txt` |
| Intel Sapphire Rapids | `benchmark_results/2026-05-07/linux/intel-spr/results.txt` |
| AWS Graviton3 | `benchmark_results/2026-05-07/linux/graviton3/results.txt` |
| AWS Graviton4 | `benchmark_results/2026-05-07/linux/graviton4/results.txt` |
| IBM POWER10 | `benchmark_results/2026-05-07/linux/ibm-power10/results.txt` |
| IBM Z / s390x | `benchmark_results/2026-05-07/linux/ibm-s390x/results.txt` |
| RISE RISC-V | `benchmark_results/2026-05-07/linux/rise-riscv/results.txt` |
| macOS AArch64 | `benchmark_results/2026-05-07/macos/aarch64/results.txt` |

## Methodology Notes

- Parsed 3520 Criterion throughput rows from the 8 current Linux result files and 440 rows from the local macOS proof file.
- Every successful current Linux artifact has commit `ff3060e2c0a696884e531a6c18616df828457a5a` and 440 parsed AEAD throughput rows.
- The stale Zen5 file under `benchmark_results/2026-05-07/linux/amd-zen5/results.txt` has commit `d592060512f9f627d20a602bc8d7f216806e29b2`; it is historical only for this snapshot.
- A comparison is matched when a Criterion ID has a `rscrypto` path component and an external ID with the same platform, primitive, operation, and input size, differing only in the implementation component.
- The local macOS run is included for Apple Silicon signal, but it is not the final platform-matrix gate for commit `ff3060e`.
