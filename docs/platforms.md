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
The Arm DIT guard is a timing mitigation, not a backend.
Without `std`, it still detects `FEAT_DIT` at run time on Linux, Android, and Apple targets.
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

On Intel hybrid CPUs (Alder Lake through Arrow Lake), detection removes AVX-512 and AVX10
even when CPUID reports them.
The E-cores lack those instructions, so a thread that moves to an E-core during an AVX-512 kernel would fault.
If every core implements AVX-512, for example with the E-cores disabled in firmware,
call the `unsafe` `platform::expert::allow_hybrid_avx512()` before the first `platform::caps()` call to keep them.
The former `RSCRYPTO_FORCE_AVX512` environment variable has no effect.

x86-64 detection reports a capability only when the CPU also reports every feature that Rust's target feature
implies for it. A CPUID mask, such as a hypervisor can set, therefore cannot select a kernel
whose implied instructions the CPU lacks.

Every accelerated path must give the same output as portable Rust for representative lengths,
alignments, tails, and state transitions.
Cross-compilation proves only that a target builds.
Runtime behavior needs execution on the target.

### Process forks

`platform::ProcessOrigin` records the process that created a value, so that an owner of unique state can
refuse a copy that a fork made. `NonceCounter`, the HPKE sender, and `RsaSigningWorkspace` use it, and
`ChaCha20Rng::origin` reports it.

- With `std` on 64-bit Linux and Android (x86-64, AArch64, POWER, RISC-V, and IBM Z), detection uses one
  page that the kernel gives as zeros to a fork child (`MADV_WIPEONFORK`, Linux 4.14).
- With `std` on Apple targets, detection uses one word that a fork handler (`pthread_atfork`) clears in
  each fork child. A fork that does not go through `fork`, such as a raw system call, is not detected.
- The page or word holds a generation above the generations of all ancestors, so a descendant never
  matches an ancestor's origin, even with the same process ID.
- If the kernel refuses the page or Apple refuses the handler, detection compares process IDs.
  Other Unix targets with `std` always compare process IDs.
  A process-ID comparison accepts a descendant that receives the recycled ID of a dead ancestor.
- Targets that cannot fork, such as Windows, WebAssembly, and bare-metal targets, always report the current process.
- Without `std`, no fork is detected.
- `ProcessOrigin::try_current` returns `ProcessOriginUnavailable` where detection compares process IDs, or
  where a Unix build has no `std`. Use it for state that no descendant may continue.

A restored virtual-machine snapshot or another copied memory image is not detected.

### Notes by primitive

- **SHA-224 and SHA-256** use the same SHA-256 compression capability policy.
  The x86-64 SHA-NI backend needs both `sha` and `sse4.1`.
  Other CPUs use the portable fallback.
- **Adler-32** has a portable Rust kernel on every target: zlib's scalar loop, which reduces both
  sums once per 5,552 bytes.
  - AArch64 selects a NEON kernel when cached runtime detection reports NEON, and a NEON kernel
    with the dot-product extension (`UDOT`) when it reports `FEAT_DotProd`.
  - x86-64 selects, by cached runtime detection, the first kernel that the CPU supports among
    AVX-512 VNNI (with AVX-512F and AVX-512BW), AVX2, and SSSE3. Below 256 bytes, the AVX2 and VNNI
    kernels share one short AVX2 path. AVX-512 without VNNI uses the AVX2 kernel: no measured host selects a
    512-bit kernel without VNNI, and such CPUs can lower their clock for 512-bit code.
  - POWER selects an AltiVec kernel (`vmsumubm`) when cached runtime detection reports AltiVec.
    IBM Z selects a vector-facility kernel when detection reports the vector facility, from z13.
    Earlier IBM Z CPUs use the portable kernel.
  - riscv64 selects an RVV kernel when cached runtime detection reports the V extension. It runs
    on every vector length and uses at most 256 lanes of 32 bits per row.
  - wasm32 selects a SIMD128 kernel when the artifact is built with `simd128`, unless `portable-only`
    is set. Builds without `simd128` use the portable kernel.
  - `portable-only` and Miri select the portable kernel.
  - `RSCRYPTO_ADLER32_FORCE` (`portable`, `neon`, `neon-dotprod`, `ssse3`, `avx2`, `avx512-vnni`,
    `vmsum`, `vsum`, or `rvv`) forces a kernel, as the CRC force variables do; a kernel that
    the CPU lacks falls back to automatic selection.
    `Adler32::config()` reports the request and the effective choice.
  - Other targets use the portable kernel.
- **X-Wing** has no backend or dispatch of its own. It runs the ML-KEM-768, X25519, and SHA-3
  backends that each target selects for those primitives. X25519 assembly runs on Apple and Linux
  AArch64 and on Linux and Windows x86-64; other targets use portable X25519.
- **HPKE** has no backend or dispatch of its own. Each suite runs the KEM, HKDF-SHA256, and AEAD
  backends that each target selects for those primitives.
- **XXH32 and XXH64** are portable Rust on every target, with no accelerated backend or dispatch.
  They read input words little-endian on every target, as the specification requires.
- **SipHash** is portable Rust on every target, with no runtime dispatch.
  Apple AArch64 rotates each word with a separate `ror` instruction, unless `portable-only` or Miri is active.
  Other AArch64 targets absorb two input words in each loop iteration.
- **ChaCha20Rng** has no backend of its own. It runs the ChaCha20 keystream kernel that cached detection
  selects for ChaCha20-Poly1305 on the target.
- **QUIC-LB** runs on the AES-128 backends. On AES-NI and AES-CE, all Feistel passes run in one target-feature
  scope with the round keys in registers. Single-pass decryption uses the AES-128 inverse cipher of each backend:
  AES-NI, AES-CE, CPACF KM, POWER8, RISC-V Zvkned and Zknd, the portable cipher, and a bitsliced inverse
  on RV64 and IBM Z without AES hardware.
- **Composite ML-DSA** has no backend or dispatch of its own. It runs the ML-DSA-65, Ed25519, ECDSA P-256,
  and SHA-512 backends that each target selects for those primitives.
- **Scalar WebAssembly:** when `simd128` is disabled, hash, AEAD, and Argon2 dispatch exclude SIMD backends.
  This compile-time boundary is separate from `portable-only` runtime dispatch.
- **BLAKE3 wasm32:** `simd128` enables four-lane chunk hashing, parent reduction, and equal-length
  digest batching through `core::arch::wasm32`. Serial compression remains portable. The artifact requires a
  SIMD128-capable engine. Builds without `simd128`, or with `portable-only`, use the portable backend.
- **P-256 ECDH** is a standalone leaf feature with a safe Rust reference implementation on every
  supported target.
  - Apple and Linux AArch64 select embedded s2n-bignum fixed-base and arbitrary-point assembly
    at compile time, unless `portable-only` or Miri is active.
  - Linux and Windows x86-64 select the baseline or ADX/BMI2 System V kernels after cached runtime
    detection. Windows calls them through `extern "sysv64"`, so both targets run the same bodies.
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
- **ECDSA P-256** signing and key derivation run the P-256 fixed-base assembly on Apple and Linux AArch64 and on
  Linux and Windows x86-64, unless `portable-only` or Miri is active. Verification of public inputs runs owned field
  kernels: flag-chained assembly on AArch64, and on x86-64 BMI2/ADX `MULX` assembly after cached runtime detection.
  Other CPUs and targets use the portable field.
- **ECDSA P-384** signing and key derivation run the AArch64 fixed-base assembly on Apple and Linux AArch64, and the
  P-384 ECDH fixed-base kernel on Linux and Windows x86-64. Other targets, `portable-only`, and Miri use the portable
  comb.

The [P-256 ECDH development snapshot](https://github.com/loadingalias/rscrypto/blob/50c708b29905bb3221e7b22b9c7cce8fcc700818/benchmark_results/OVERVIEW.md#p-256-ecdh-development-snapshot) records the September 2026 results for Graviton3, Graviton4,
and Intel Granite Rapids on Linux and Windows, with their source identities.
Its Linux timing and cleanup bundles are older than later shared-source changes.
It does not supply Windows timing or cleanup evidence for the exact candidate.
Native runtime tests and benchmarks do not replace those gates.
Evidence from one CPU does not qualify another CPU.

## Toolchain

`rscrypto` needs Rust 1.100 (`rust-version = "1.100.0"`).
Rust 1.100 became stable on 2026-11-12.
The repository tests with the nightly pinned in [`rust-toolchain.toml`](../rust-toolchain.toml).
It checks the MSRV with the released Rust 1.100.0 toolchain.

Every target in the catalog supports `portable-only` on the declared MSRV.
The compatibility lane checks core-only and allocation-enabled builds and generates release code
for POWER, IBM Z, RV64, and RV32 with this feature on the MSRV compiler.
On these targets, the accelerated backends of the features below use unstable compiler features and need nightly Rust.
A build that selects none of these features compiles on the MSRV without `portable-only`.
The compatibility lane checks every other feature for each target on the MSRV compiler.
[`src/lib.rs`](../src/lib.rs) owns the feature lists.

| Target | Features with a nightly backend | Unstable features |
| --- | --- | --- |
| `powerpc64le-unknown-linux-gnu` | `adler32`, `aegis256`, `aes-gcm`, `aes-gcm-siv`, `aes-siv`, `argon2`, `blake3`, `chacha20-rng`, `chacha20poly1305`, `crc16`, `crc24`, `crc32`, `crc64`, `ml-dsa`, `poly1305`, `xchacha20poly1305`, `xxh3` | `portable_simd`, `powerpc_target_feature` |
| `s390x-unknown-linux-gnu` | `adler32`, `aegis256`, `aes-gcm`, `aes-gcm-siv`, `argon2`, `blake3`, `chacha20-rng`, `chacha20poly1305`, `crc16`, `crc24`, `crc32`, `crc64`, `ml-dsa`, `ml-kem`, `poly1305`, `xchacha20poly1305`, `xxh3` | `asm_experimental_reg`, `portable_simd`, `stdarch_s390x` |
| `riscv64gc-unknown-linux-gnu` | `adler32`, `aegis256`, `aes-gcm`, `aes-gcm-siv`, `aes-siv`, `blake2b`, `blake2s`, `blake3`, `chacha20-rng`, `chacha20poly1305`, `crc16`, `crc24`, `crc32`, `crc64`, `poly1305`, `sha2`, `xchacha20poly1305` | `asm_experimental_reg`, `portable_simd`, `riscv_ext_intrinsics`, `riscv_target_feature` |
| `riscv32imac-unknown-none-elf` | `sha2` | `riscv_ext_intrinsics` |

A feature that enables one of these features needs nightly Rust too, for example `hmac` on RV64 through `sha2`.

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
| `just check`       | Repairs formatting and lints, checks this Mac, and runs the CI tooling checks: script tests, CT tooling tests, and dependency policy. CI checks every other catalog target. |
| Native CI          | Native and portable suites on Linux x86-64, AArch64, POWER, IBM Z, and RISC-V, and on Windows x86-64. Doctests run on the x86-64, AArch64, and Windows rows. POWER, IBM Z, and RISC-V build on x86-64 and run the transferred artifacts on native hardware, without doctests or the SLH-DSA small-set differentials. RISC-V is skipped on fork pull requests and when no donated runner starts it within 30 minutes; releases wait for it. |
| `just check-macos` | Full local Apple Silicon qualification: native and portable release suites, doctests, internal evidence regressions, and physical RSA assembly checks. `just qualify-macos` queues it in the background before a push or release. |
| `just test-musl`   | Native and portable suites, plus doctests, on a matching x86-64 or AArch64 Linux host. |
| CI compatibility   | Feature, MSRV, bare-metal, and MSRV `portable-only` compilation on POWER, IBM Z, RV64, and RV32. Scalar and SIMD vector execution for `wasm32-unknown-unknown` and `wasm32-wasip1` in Wasmtime. |

A configured check is not a passing result for the current revision.
Inspect the matching run artifacts before you qualify a release.
Bare-metal checks do not run on devices.
Wasmtime results do not show browser-engine behavior.
Windows AArch64 runtime CI is deferred. The RISC-V QEMU workflow (`.github/workflows/qemu.yml`) runs the RISC-V
suites under QEMU with the vector, bit-manipulation, and crypto extensions on, so the extension kernels
execute. It proves function, not timing or speed.
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
