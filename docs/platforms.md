# Platforms

Portable Rust defines every supported primitive.
SIMD and assembly backends make it faster; they are never a second specification.

32-bit x86 targets, including i586 and i686, are unsupported and fail compilation.
Use an x86-64 target for x86 deployments. The platform API no longer exposes `Arch::X86`.

## Backend selection

Dispatch has three tiers:

1. Compile-time target features can select an eligible backend.
1. With `std`, cached runtime detection selects a backend from the capabilities
   that the CPU and the operating system allow.
1. If neither selects a backend, the portable implementation runs.

`no_std` builds use only compile-time selection.
`portable-only` makes runtime detection report no accelerated capabilities.
It also turns off assembly that the crate selects at compile time from the target alone,
such as the RSA, elliptic-curve, ML-KEM, and ML-DSA backends.
Some hash backends that a compile-time `target_feature` setting enables still run,
for example SHA-256 on Apple Silicon.
It excludes the nightly-only POWER, IBM Z, and RISC-V backends from compilation.
It does not promise to remove every accelerated backend from the binary.

Detection runs once and is cached.
Set capability overrides, and get process permissions such as Linux AMX authorization,
before the first `platform::caps()` call.

Every accelerated path must give the same output as portable Rust for representative lengths,
alignments, tails, and state transitions.
Cross-compilation proves only that a target builds.
Runtime behavior needs execution on the target.

### Notes by primitive

- **SHA-224 and SHA-256** use the same SHA-256 compression capability policy.
  The x86-64 SHA-NI backend needs both `sha` and `sse4.1`.
  Other CPUs use the portable fallback.
- **Scalar WebAssembly:** when `simd128` is disabled, hash, AEAD, and Argon2 dispatch exclude SIMD backends.
  This compile-time boundary is separate from `portable-only` runtime dispatch.
- **BLAKE3 wasm32:** `simd128` enables four-lane chunk hashing, parent reduction, and equal-length
  digest batching through `core::arch::wasm32`. Serial compression remains portable. The artifact requires a
  SIMD128-capable engine. Builds without `simd128`, or with `portable-only`, use the portable backend.
- **P-256 ECDH** is a standalone leaf feature with a safe Rust reference implementation on every
  supported target.
  - Apple and Linux AArch64 select embedded s2n-bignum fixed-base and arbitrary-point assembly
    at compile time, unless `portable-only` or Miri is active.
  - Linux x86-64 selects the baseline or ADX/BMI2 ELF kernels after cached runtime detection.
  - Windows x86-64 uses the same baseline or ADX/BMI2 arithmetic behind Microsoft x64 wrappers.
    Public SEC1 validation crosses one batch boundary
    for the target instead of five field-call wrappers.
  - A deterministic provenance transform keeps these backends independent of the ECDSA feature.
    It also clears their secret-derived frames, saved-register spill slots,
    and volatile integer registers.
- **P-384 ECDH** is a standalone leaf feature with a safe Rust reference implementation on every
  supported target.
  - AArch64 selects these backends at compile time, unless `portable-only` or Miri is active.
    They are owned inline-assembly field multiplication, squaring, addition, subtraction, and small multiples,
    NEON fixed-base comb selection, and Bernstein–Yang divstep field inversion.
  - x86-64 selects inline-assembly field addition, subtraction,
    and divstep inversion at compile time.
    After cached runtime detection,
    it selects BMI2/ADX field multiplication and squaring and fused in-place point doubling.
    CPUs without BMI2 and ADX use the portable multiplication and point formulas.
  - Other targets use the safe Rust reference implementation.

The [P-256 ECDH development snapshot](../benchmark_results/OVERVIEW.md#p-256-ecdh-development-snapshot) records the September 2026 results for Graviton3, Graviton4,
and Intel Granite Rapids on Linux and Windows, with their source identities.
Its Linux timing and cleanup bundles are older than later shared-source changes.
It does not supply Windows timing or cleanup evidence for the exact candidate.
Native runtime tests and benchmarks do not replace those gates.
Evidence from one CPU does not qualify another CPU.

## Toolchain

`rscrypto` needs Rust 1.100 (`rust-version = "1.100.0"`).
Rust 1.100 becomes stable on 2026-11-12.
Until then, build with the 1.100 beta or a newer nightly.
The repository tests with the nightly pinned in [`rust-toolchain.toml`](../rust-toolchain.toml).
It checks the MSRV with the exact 1.100 beta pinned in [`scripts/lib/toolchain.py`](../scripts/lib/toolchain.py).

Every target in the catalog supports `portable-only` on the declared MSRV.
The compatibility lane checks core-only and allocation-enabled builds and generates release code
for POWER, IBM Z, RV64, and RV32 with this feature on the MSRV compiler.
Their accelerated backends still use the unstable compiler features below and need nightly Rust.

| Target                                     | Unstable features |
| ------------------------------------------ | ----------------- |
| `powerpc64le-unknown-linux-gnu`            | `portable_simd`, `powerpc_target_feature` |
| `s390x-unknown-linux-gnu`                  | `asm_experimental_reg`, `portable_simd`, `stdarch_s390x` |
| `riscv64gc-unknown-linux-gnu`              | `asm_experimental_reg`, `portable_simd`, `riscv_ext_intrinsics`, `riscv_target_feature` |
| `riscv32imac-unknown-none-elf` with `sha2` | `riscv_ext_intrinsics` |

Unstable features can change between nightlies.
For accelerated builds on these targets, the tested contract is the pinned nightly.
A newer nightly can fail to build them until `rscrypto` adapts.

Allocator-aware APIs, such as the ML-KEM and ML-DSA `*_in` constructors,
use only the allocator API that is stable in Rust 1.100.
`rscrypto` does not enable `allocator_api`.
Fallible allocator extensions, such as `Box::try_new_in`, stay out of the API until they are stable.

## Supported targets

[`.config/target-matrix.json`](../.config/target-matrix.json) is the catalog of supported targets.
Targets outside it can compile, but they are not part of the tested support contract.
Each target needs its own evidence.

The [CI workflow](../.github/workflows/ci.yml) and the [repository recipes](../scripts/README.md) define current validation:

| Check              | Scope |
| ------------------ | ----- |
| `just check`       | Compilation and lint checks for the host and every catalog cross-target. |
| Native CI          | Native and portable suites, plus doctests, on Linux x86-64, AArch64, POWER, IBM Z, and RISC-V, and on Windows x86-64. POWER, IBM Z, and RISC-V build on x86-64 and run the transferred artifacts on native hardware. |
| `just check-macos` | Full local Apple Silicon qualification: native and portable release suites, doctests, internal evidence regressions, and physical RSA assembly checks. The push hook queues this suite in the background. |
| `just test-musl`   | Native and portable suites, plus doctests, on a matching x86-64 or AArch64 Linux host. |
| `just ci-compat`   | Feature, MSRV, bare-metal, and MSRV `portable-only` compilation on POWER, IBM Z, RV64, and RV32. Scalar and SIMD vector execution for `wasm32-unknown-unknown` and `wasm32-wasip1` in Wasmtime. |

A configured check is not a passing result for the current revision.
Inspect the matching run artifacts before you qualify a release.
Bare-metal checks do not run on devices.
Wasmtime results do not show browser-engine behavior.
Windows AArch64 runtime CI is deferred. The pre-push hook queues macOS checks and tests locally.
Releases require the passing `rscrypto/macos` status for the commit's tree and compiler.
Timing qualification on physical Apple Silicon is a separate local requirement.

[`.github/runs-on.yml`](../.github/runs-on.yml) defines the AWS runner shapes and the Spot policy.
Native CI, cross-build preparation, fuzzing, and CT measurement use separate profiles.
The [runner guidance](../scripts/README.md#native-tooling) explains how to select a profile and when catalog changes take effect.
A smaller runner does not reduce the required test or evidence surface.

Performance and constant-time claims need retained evidence for the exact operation
and configuration.
A target in the catalog, or a passing compile check, does not supply that evidence.
See the [benchmark record](../benchmark_results/OVERVIEW.md) and the [constant-time evidence model](constant-time.md).

Retained POWER, IBM Z, and RISC-V evidence covers native unit and backend behavior
and focused portable-versus-accelerated tests.
Windows AArch64 has compile-only evidence.
Windows x86-64 has native runtime evidence.

Apple Silicon is the only supported macOS architecture.
`x86_64-apple-darwin` is not in the catalog, and it is not tested or maintained.
It can compile by chance, but that does not make it a supported target.

## Inspect one build

Backend availability changes with the primitive, the target, the compiler, and the CPU.
Use `rscrypto::platform` and the `introspect` example to inspect one build:

```bash
cargo run --example introspect --features 'crc32,sha2,chacha20poly1305,diag'
```

For target-specific timing claims, see [`constant-time.md`](constant-time.md).
For performance evidence, see [`benchmarking.md`](benchmarking.md).
