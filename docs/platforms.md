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
release evidence. Exact-final-source Windows timing and cleanup
artifacts remain awaiting the first successful wired qualification run, and
dedicated physical timing is unavailable; the native runtime and benchmark do
not stand in for those gates.
Other targets and microarchitectures retain their portable fallback or remain
awaiting their own native qualification row. Evidence from one CPU is never
substituted for another. The target catalog and qualification workflows own
those per-row obligations.

## Supported targets

[`.config/target-matrix.json`](../.config/target-matrix.json) is both the target
support catalog and the Cargo Rail variant catalog. Targets outside it may
compile, but are not part of the tested support contract. `ci.yaml` executes
only affected rows; `qualification.yaml` executes every row. The ordinary
Linux x86-64 row is already owned by the core Rust job and is not duplicated in
the platform matrix.

| Target | Compile proof | Runtime proof | Perf | CT | Release |
| --- | --- | --- | --- | --- | --- |
| `aarch64-apple-darwin` | Native | Virtual native | No | No | Yes |
| `aarch64-pc-windows-msvc` | Hosted | None | No | No | Yes |
| `aarch64-unknown-linux-gnu` | Native | Virtual native | Yes | Yes | Yes |
| `aarch64-unknown-linux-musl` | Generic cross | None | No | No | Yes |
| `aarch64-unknown-none` | Generic cross | None | No | No | Yes |
| `powerpc64le-unknown-linux-gnu` | Native | Physical native | Yes | Yes | Yes |
| `riscv32imac-unknown-none-elf` | Generic cross | None | No | No | Yes |
| `riscv64gc-unknown-linux-gnu` | Native | Physical native | Yes | Yes | Yes |
| `s390x-unknown-linux-gnu` | Native | Physical native | Yes | Yes | Yes |
| `thumbv6m-none-eabi` | Generic cross | None | No | No | Yes |
| `wasm32-unknown-unknown` | Generic cross | None | No | No | Yes |
| `wasm32-wasip1` | Generic cross | Wasmtime emulation | No | No | Yes |
| `x86_64-apple-darwin` | Native | Virtual native | No | No | Yes |
| `x86_64-pc-windows-msvc` | Native | Virtual native | No | No | Yes |
| `x86_64-unknown-linux-gnu` | Core job | Virtual native | Yes | Yes | Yes |
| `x86_64-unknown-linux-musl` | Generic cross | None | No | No | Yes |
| `x86_64-unknown-none` | Generic cross | None | No | No | Yes |

Performance and CT entries refer to their separate hardware workflows; a
compile row never supplies those claims. The catalog also contains a separate
physical Intel Sapphire Rapids proof for Linux AMX process authorization.

Native platform rows fail before Cargo work if the Rust host triple or machine
architecture does not match the catalog. Donated POWER, IBM Z, and RISC-V
machines run only native unit/backend evidence and focused portable-versus-
accelerated tests. Windows AArch64 compiles but does not claim runtime evidence;
Windows x86-64 does. Both declared Apple targets have routine native ownership.

Local and remote machines reproduce a row with
`just target-contract ROW [shallow|deep]` or
`just ssh-just MACHINE target-contract ROW deep`. `ssh-list` remains the
authority for development-machine names; rscrypto does not duplicate that
provider catalog.

Backend availability varies by primitive, target, compiler, and CPU. Use
`rscrypto::platform` and the `introspect` example to inspect one build:

```sh
cargo run --example introspect --features 'crc32,sha2,chacha20poly1305,diag'
```

Use [`constant-time.md`](constant-time.md) for target-specific timing claims and
[`benchmarking.md`](benchmarking.md) for performance evidence.
