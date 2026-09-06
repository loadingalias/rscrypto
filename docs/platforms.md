# Platforms

Portable Rust defines every supported primitive. SIMD and assembly are
accelerators, never separate specifications.

## Backend selection

Dispatch has three tiers:

1. Compile-time target features may select an eligible backend.
2. With `std`, cached runtime detection selects from CPU- and OS-authorized
   capabilities.
3. Otherwise, the portable implementation runs.

`no_std` builds use compile-time selection only. `portable-only` makes runtime
detection return no accelerated capabilities, but it does not override
compile-time target features or remove code from the binary.

Capability overrides and process authorization such as Linux AMX permission
must occur before the first `platform::caps()` call because detection is cached.

Every accelerated path must match portable Rust for representative lengths,
alignments, tails, and state transitions. Cross-compilation proves only that a
target builds; runtime behavior requires target execution.

P-256 ECDH remains a standalone leaf with a safe Rust authority on every
supported target. Apple and Linux AArch64 builds select embedded s2n-bignum
fixed-base and arbitrary-point assembly at compile time unless `portable-only`
or Miri is active. Linux x86-64 selects the corresponding baseline or ADX/BMI2
ELF kernels after cached runtime capability detection. Windows x86-64 uses the
same baseline or ADX/BMI2 arithmetic behind Microsoft x64 wrappers; public SEC1
validation crosses one target-shaped batch boundary instead of five field-call
wrappers. The deterministic provenance transform keeps those backends
independent of the ECDSA feature and clears their secret-derived frames,
saved-register spill slots, and volatile integer registers. Physical
Graviton3, Graviton4, Intel Granite Rapids Linux, and Intel Granite Rapids
Windows development evidence covers the applicable native ABI, direct portable
differentials, independent vectors and implementations, and equivalent-work
performance for the measured Phase 4 candidates. The sealed Linux bundles
retain complete operation-level timing artifacts and optimized cleanup
evidence, but later shared-source edits mean they are not exact-final-source
release evidence. Exact-final-source Windows timing and cleanup artifacts are
not available, and dedicated physical timing is unavailable; the native runtime
and benchmark do not stand in for those gates. Other targets and
microarchitectures retain their portable fallback or remain without native
evidence. Evidence from one CPU is never substituted for another.

## Supported targets

[`.config/target-matrix.json`](../.config/target-matrix.json) is the target
support catalog. Targets outside it may compile, but are not part of the tested
support contract. Target-specific evidence must be collected independently.

| Target | Compile proof | Runtime proof | Perf | CT |
| --- | --- | --- | --- | --- |
| `aarch64-apple-darwin` | Native | Virtual native | No | No |
| `aarch64-pc-windows-msvc` | Hosted | None | No | No |
| `aarch64-unknown-linux-gnu` | Native | Virtual native | Yes | Yes |
| `aarch64-unknown-linux-musl` | Generic cross | None | No | No |
| `aarch64-unknown-none` | Generic cross | None | No | No |
| `powerpc64le-unknown-linux-gnu` | Native | Physical native | Yes | Yes |
| `riscv32imac-unknown-none-elf` | Generic cross | None | No | No |
| `riscv64gc-unknown-linux-gnu` | Native | Physical native | Yes | Yes |
| `s390x-unknown-linux-gnu` | Native | Physical native | Yes | Yes |
| `thumbv6m-none-eabi` | Generic cross | None | No | No |
| `wasm32-unknown-unknown` | Generic cross | None | No | No |
| `wasm32-wasip1` | Generic cross | Wasmtime emulation | No | No |
| `x86_64-pc-windows-msvc` | Native | Virtual native | No | No |
| `x86_64-unknown-linux-gnu` | Native | Virtual native | Yes | Yes |
| `x86_64-unknown-linux-musl` | Generic cross | None | No | No |
| `x86_64-unknown-none` | Generic cross | None | No | No |

Performance and CT entries refer to retained target-specific evidence; a
compile proof never supplies those claims. A separate physical Intel Sapphire
Rapids run covers Linux AMX process authorization.

Retained POWER, IBM Z, and RISC-V evidence covers native unit/backend behavior
and focused portable-versus-accelerated tests. Windows AArch64 has compile-only
evidence; Windows x86-64 has native runtime evidence. Apple Silicon is the only
supported macOS architecture.
`x86_64-apple-darwin` is not catalogued, tested, or maintained; it may compile
incidentally, but that does not make it a supported target.

Backend availability varies by primitive, target, compiler, and CPU. Use
`rscrypto::platform` and the `introspect` example to inspect one build:

```sh
cargo run --example introspect --features 'crc32,sha2,chacha20poly1305,diag'
```

Use [`constant-time.md`](constant-time.md) for target-specific timing claims and
[`benchmarking.md`](benchmarking.md) for performance evidence.
