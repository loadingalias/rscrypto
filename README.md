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
Accelerated POWER, IBM Z, and RISC-V builds need a nightly compiler;
`portable-only` builds support the declared MSRV.
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
The [API compatibility notes](docs/api-compatibility.md) explain output types, collection seeds,
and retained method names.
The [cryptographic resilience guide](docs/cryptographic-resilience.md) explains security assumptions,
migration boundaries, and the evidence to review when choosing stronger cryptography.

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

Bench run [#37092266645](https://github.com/loadingalias/rscrypto/actions/runs/37092266645) measured the v0.10.0 source
on eight platforms on 2026-10-03.
Each platform has 346 like-for-like comparisons of hashes, checksums, MACs, XOFs, and scrypt.
Each comparison divides the median time of the fastest external crate by the median time of `rscrypto`.
A ratio above 1.00x means `rscrypto` is faster.

| Platform | Faster | Within 3% | Slower | Median ratio |
| --- | ---: | ---: | ---: | ---: |
| Intel Linux (`c8i`) | 211 | 88 | 47 | 1.09x |
| AMD Linux (`c8a`) | 256 | 81 | 9 | 1.13x |
| Intel Windows (`c8i`) | 186 | 105 | 55 | 1.05x |
| AMD Windows (`c8a`) | 262 | 68 | 16 | 1.09x |
| Graviton5 Linux (`c9g`) | 147 | 121 | 78 | 1.01x |
| POWER10 Linux | 206 | 122 | 18 | 1.07x |
| IBM Z Linux | 301 | 19 | 26 | 2.77x |
| RISC-V Linux | 197 | 79 | 70 | 1.05x |

These results are from one run on cloud and shared hosts.
They do not include AEAD, signature, key-exchange, ML-KEM, ML-DSA, Argon2, or RSA cases,
and they do not include macOS.
The [benchmark overview](benchmark_results/OVERVIEW.md#2026-10-03-full-benchmark-run-v0100) gives the results by family,
the method, and the limits.
The [comparison contracts](docs/benchmarking.md#ml-kem-and-argon2-comparison-contracts) define equivalent ML-KEM and Argon2 workloads.
Older summary tables mixed different workloads. They are historical records, not current claims.

## Project

The guides and examples describe the source in this repository.
For a published version, use the matching [API documentation](https://docs.rs/rscrypto).

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before you change code.
[`CHANGELOG.md`](CHANGELOG.md) lists published changes.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
