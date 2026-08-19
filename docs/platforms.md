# Platforms

Every supported target retains a portable Rust implementation. Compile-time
configuration and, with `std`, runtime CPU detection may select an eligible
accelerated backend.

## Dispatch model

1. **Compile-time**: `#[cfg(target_feature = "...")]` selects the strongest backend permitted by `RUSTFLAGS` / `target-feature`.
2. **Runtime detection** (`std` only): cached `platform::caps()` probes CPU features once via detection intrinsics and, on supported Linux/Android targets, OS capability files such as `/proc/self/auxv`; it then dispatches to the strongest available kernel.
3. **Portable Rust fallback**: always present. The portable implementation is the source of truth; SIMD and ASM kernels are differentially tested against it.

In `no_std` builds, only the compile-time tier runs.

With `std` enabled, the `portable-only` feature makes `platform::caps()` report
no runtime SIMD/ASM capabilities, so dispatchers that consult runtime caps fall
through to portable backends. It does not remove SIMD code from the binary or
override compile-time `target_feature` selection. See
[`features.md`](features.md#portable-only).

## Acceleration matrix

Backend availability depends on what the target CPU advertises and what
`target-feature` permits. The portable Rust fallback is present on every target
listed below.

This is a crate-wide capability matrix, not a promise that every primitive has
every listed backend. A target-specific candidate is eligible only when it wins
representative target-native measurements. Otherwise dispatch uses a proven
lower tier, including portable Rust.

| Target family | Backends used (when CPU advertises them) |
|---|---|
| x86_64 | SSE4.2 CRC32; SSSE3 / PCLMULQDQ; AVX2; AES-NI; SHA-NI; AVX-512F / VL / BW / DQ; AVX-512IFMA; VPCLMULQDQ; VAES |
| aarch64 / Apple Silicon | NEON; AES; PMULL; CRC; SHA2; SHA3 / EOR3; SHA512; SVE2-PMULL where available |
| s390x (IBM Z) | z/Vector; vector enhancements; CPACF / MSA; VGFM; fixed-work ML-KEM arithmetic |
| ppc64le (POWER) | AltiVec; VSX; POWER8 vector / crypto and atomics; POWER9 / POWER10 vector; VPMSUMD |
| riscv64 | V / RVV; Zbc; Zvbc; Zbkc; Zkne / Zknd; Zvkned; Zkt / Zvkt |
| wasm32 | SIMD128 where enabled |

ECDSA P-256/P-384 always has a portable Rust path. x86_64 and aarch64 targets
also use assembly helpers for selected scalar, field, and basepoint operations
when those helpers are compiled for the target. Other target families use the
portable implementation.

ML-KEM-512/768/1024 always have a portable Rust path. On s390x, secret-fed
ML-KEM arithmetic uses fixed-work z/Vector kernels where those kernels are
compiled and selected. The implementation does not replace constant-time
hardening with native scalar multiply or divide on secret-fed arithmetic.

## `no_std` targets

The following `no_std` targets are built in CI:

- `thumbv6m-none-eabi`
- `riscv32imac-unknown-none-elf`
- `aarch64-unknown-none`
- `x86_64-unknown-none`
- `wasm32-unknown-unknown`
- `wasm32-wasip1`

Targets outside this list are not part of the CI contract. Open an issue with
the exact target triple and feature set when a required target is missing.
Build coverage does not establish a constant-time claim; use
[`constant-time.md`](constant-time.md) for release-evidenced configurations.

## Per-platform benchmark evidence

The 2026-08-18 per-platform results live in
[`benchmark_results/OVERVIEW.md`](../benchmark_results/OVERVIEW.md#coverage-matrix),
covering eight Linux CI runners; the RISE RISC-V runner did not execute in that
run. Per-platform results diverge: the same commit measures `2.77x` on IBM
z16/s390x and `1.36x` on AWS Graviton3, and s390x carries a large ECDSA
regression in this snapshot. Benchmark the deployment workload on its target CPU
before choosing a performance-sensitive backend or feature set.
