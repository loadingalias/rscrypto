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
