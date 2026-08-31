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

[`.config/target-matrix.json`](../.config/target-matrix.json) owns the target
groups and CI execution lanes. Targets outside that matrix may compile, but are
not part of the tested support contract.

The matrix includes native Windows, macOS, Linux, IBM, bare-metal `no_std`, and
WASM targets. Only the listed GitHub Actions and native runner lanes provide
runtime evidence.

Backend availability varies by primitive, target, compiler, and CPU. Use
`rscrypto::platform` and the `introspect` example to inspect one build:

```sh
cargo run --example introspect --features 'crc32,sha2,chacha20poly1305,diag'
```

Use [`constant-time.md`](constant-time.md) for target-specific timing claims and
[`benchmarking.md`](benchmarking.md) for performance evidence.
