# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![Docs.rs](https://docs.rs/rscrypto/badge.svg)](https://docs.rs/rscrypto)
[![MSRV 1.100.0](https://img.shields.io/badge/MSRV-1.100.0-blue)](Cargo.toml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/rscrypto)](#license)

`rscrypto` is a Rust library of cryptographic primitives, hashes, password hashing, and checksums.
One feature model controls all of them.
Portable Rust is the reference implementation.
SIMD and assembly backends make it faster on supported targets.
Production builds do not depend on C, FFI, OpenSSL, or system libraries.

`rscrypto` supplies primitives only.
It is not a TLS stack, a PKI toolkit, a key store, or a protocol implementation.

## Install

`rscrypto` needs Rust 1.100.
Until Rust 1.100 is stable, use the 1.100 beta or a newer nightly.
POWER, IBM Z, and RISC-V builds need a nightly compiler.
See [Toolchain](docs/platforms.md#toolchain).

Minimal `no_std` build with SHA-2 only:

```toml
[dependencies]
rscrypto = { version = "0.10", default-features = false, features = ["sha2"] }
```

All primitives, with operating-system randomness:

```toml
[dependencies]
rscrypto = { version = "0.10", features = ["full", "getrandom"] }
```

The default feature is `std`.
Set `default-features = false` to remove it.
Enable `getrandom` only for the APIs that get salts, keys, nonces,
or RSA key-generation entropy from the operating system.
The [feature guide](docs/features.md) explains how to select features.
[`Cargo.toml`](Cargo.toml) defines the exact feature graph.

## Quick start

```rust
use rscrypto::Sha256;

let one_shot = Sha256::digest(b"hello world");

let mut hasher = Sha256::new();
hasher.update(b"hello ");
hasher.update(b"world");

assert_eq!(hasher.finalize(), one_shot);
```

Hash types support one-shot and streaming use.
[`examples/README.md`](examples/README.md) has runnable examples for AEAD, signatures, RSA, P-256 and P-384 ECDH,
X25519, ML-KEM, password hashing, and backend introspection.

## Primitives and features

| Family | Primitives | Feature |
| --- | --- | --- |
| Checksums | CRC-16, CRC-24, CRC-32, CRC-32C, CRC-64/XZ, CRC-64/NVMe | `checksums` or a leaf feature |
| Cryptographic hashes | SHA-2, SHA-3, SHAKE, cSHAKE, BLAKE2, BLAKE3, Ascon-Hash/XOF/CXOF | `crypto-hashes` or a leaf feature |
| Fast hashes | XXH3-64/128, RapidHash V3-64 | `fast-hashes` or a leaf feature |
| MACs and KDFs | HMAC-SHA-2/SHA-3, KMAC128/256, Poly1305, HKDF-SHA-2, PBKDF2-HMAC-SHA-2 | `macs`, `kdfs`, or a leaf feature |
| Password hashing | Argon2d/i/id, scrypt, bounded PHC password records | `password-hashing` or a leaf feature |
| Signatures and RSA | ECDSA P-256/P-384, Ed25519, [ML-DSA-44/65/87](docs/mldsa.md), RSA signing, verification, encryption, and key generation | `signatures` or a leaf feature |
| Key exchange and KEMs | P-256 ECDH, P-384 ECDH, X25519, ML-KEM-512/768/1024 | `key-exchange` or a leaf feature |
| AEADs | AES-GCM, AES-GCM-SIV, AES-SIV-CMAC, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, Ascon-AEAD128 | `aead` or a leaf feature |

The WebSocket accept digest exists only for protocol compatibility.
It needs the `websocket-sha1` feature.
No umbrella feature, including `full`, enables it.

See [docs.rs](https://docs.rs/rscrypto) for exact types and methods.

## Platforms and dispatch

The portable Rust implementation defines the correct output, byte for byte.
Every accelerated backend must give the same output.
At compile time, the target sets which backends are available.
With `std`, `rscrypto` also detects CPU features at run time and selects a backend.
If no accelerated backend is available, the portable implementation runs.

The [platform guide](docs/platforms.md) lists the supported targets.
It also explains dispatch, `no_std` support, and the limits of `portable-only`.

## Assurance

A security claim exists only while its evidence passes.
Missing or old evidence removes the claim; it never weakens a gate.

- **Correctness:** NIST, RFC, upstream, and Wycheproof vectors, independent implementations,
  property tests, negative tests, and Miri.
- **Fuzzing:** fuzz targets run the production code across primitives, parsers, state machines,
  and trait boundaries.
  Minimized inputs replay as tests.
  A separate lane runs them with sanitizers.
- **Backends:** differential tests compare each accelerated backend with portable Rust.
  They cover lengths, alignments, tails, state transitions, dispatch,
  and fallback on native targets.
- **Constant time:** [`ct.toml`](ct.toml) lists the exact operations under test.
  The evidence combines inspection of optimized linked binaries,
  BINSEC proofs for fixed-shape kernels, and DudeCT timing tests for end-to-end operations.
- **Secrets:** types that own secrets hide their contents in `Debug` output
  and clear their initialized storage on drop.
  Copy rules vary by type; for example, keyed BLAKE2 and BLAKE3 state supports `Clone`.
  The [secret ownership inventory](docs/secret-ownership.md) lists each type.
- **Failures:** verification failures do not tell why they failed.
  A failed AEAD open clears the unauthenticated plaintext.

A constant-time claim applies only to the target, features, compiler, profile,
and operation that its evidence covers.
Source code that looks branchless is not proof.

For details, see the [test evidence](docs/test-vector-coverage.md), the [constant-time model](docs/constant-time.md), the [secret lifecycle](docs/secret-lifecycle.md),
and the [threat model](THREAT_MODEL.md).

`rscrypto` has not had a third-party security audit.
The project cannot pay for one now, and automated evidence does not replace it.
`rscrypto` does not claim to be audited, FIPS 140-3 validated, formally verified,
or constant time as a whole crate.

Report a suspected vulnerability through [GitHub Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new).
Follow the [`SECURITY.md`](SECURITY.md) process.
Do not open a public issue.

## Performance

A performance claim applies only to a retained benchmark campaign with equivalent workloads.
The September 2026 campaign has 19,614 completed cases,
including corrected ML-KEM and Argon2 comparisons.
It does not have a summary scorecard yet.
Older summary tables mixed different workloads: entropy, key preparation, output format,
or salt length.
They are historical records, not current claims.

On AArch64 Linux (Graviton5, October 2026), `rscrypto` was faster than the fastest external crate
in 147 of 346 directly comparable hash, checksum, and scrypt rows, within 3% in 121, and slower in 78.
Most of the slower rows are SHA-3 and HMAC-SHA-2.
That snapshot covers one host and one run.

The [benchmark overview](benchmark_results/OVERVIEW.md) records each campaign, its target results, and its limits.
The [comparison contracts](docs/benchmarking.md#ml-kem-and-argon2-comparison-contracts) define equivalent ML-KEM and Argon2 workloads.

## Project

The guides and examples describe the source in this repository.
For a published version, use the matching [API documentation](https://docs.rs/rscrypto).

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before you change code.
[`CHANGELOG.md`](CHANGELOG.md) lists published changes.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
