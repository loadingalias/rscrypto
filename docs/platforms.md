# Platforms

`rscrypto` is built around a three-tier dispatch model so the same source compiles to a portable Rust path on every target and a hardware-accelerated path on every supported CPU.

## Dispatch Model

1. **Compile-time** — `#[cfg(target_feature = "...")]` selects the strongest backend permitted by `RUSTFLAGS` / `target-feature`.
2. **Runtime detection** (`std` only) — cached `platform::caps()` probes CPU features once at startup via `is_x86_feature_detected!` and the aarch64 / s390x / ppc64le / riscv64 equivalents, then dispatches to the strongest available kernel.
3. **Portable Rust fallback** — always present. The portable implementation is the source of truth; SIMD and ASM kernels are differentially tested against it.

In `no_std` builds, only the compile-time tier runs.

With `std` enabled, the `portable-only` feature makes `platform::caps()` report
no runtime SIMD/ASM capabilities, so dispatchers that consult runtime caps fall
through to portable backends. It does not remove SIMD code from the binary or
override compile-time `target_feature` selection. See
[`features.md`](features.md#portable-only).

## Acceleration Matrix

Availability of any specific backend depends on what the target CPU advertises and what `target-feature` permits. The portable Rust fallback is present on every target listed below.

| Target family | Backends used (when CPU advertises them) |
|---|---|
| x86_64 | SSE4.2 CRC32; SSSE3 / PCLMULQDQ; AVX2; AES-NI; SHA-NI; AVX-512F / VL / BW / DQ; AVX-512IFMA; VPCLMULQDQ; VAES |
| aarch64 / Apple Silicon | NEON; AES; PMULL; CRC; SHA2; SHA3 / EOR3; SHA512; SVE2-PMULL where available |
| s390x (IBM Z) | z/Vector; vector enhancements; CPACF / MSA; VGFM |
| ppc64le (POWER) | AltiVec; VSX; POWER8 vector / crypto; POWER9 / POWER10 vector; VPMSUMD |
| riscv64 | V / RVV; Zbc; Zvbc; Zbkc; Zkne / Zknd; Zvkned; Zkt / Zvkt |
| wasm32 | SIMD128 where enabled |

ECDSA P-256/P-384 always has a portable Rust path. x86_64 and aarch64 targets
also use assembly helpers for selected scalar, field, and basepoint operations
when those helpers are compiled for the target. Other target families fall back
to the portable implementation unless a future measured backend justifies
additional code.

## `no_std` Targets

The following `no_std` targets are built in CI:

- `thumbv6m-none-eabi`
- `riscv32imac-unknown-none-elf`
- `aarch64-unknown-none`
- `x86_64-unknown-none`
- `wasm32-unknown-unknown`
- `wasm32-wasip1`

Other `no_std` targets in the same families (e.g. `thumbv7em-*`, larger RISC-V profiles) generally work; open an issue if a target you care about is missing.

## Per-Platform Benchmark Scorecard

Current geomean speedups by platform live in
[`benchmark_results/OVERVIEW.md`](../benchmark_results/OVERVIEW.md#coverage-matrix).
The current public set includes the 2026-06-12 nine-runner Linux CI matrix and
a 2026-06-12 local Apple Silicon macOS/aarch64 full run.
