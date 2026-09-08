# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![Docs.rs](https://docs.rs/rscrypto/badge.svg)](https://docs.rs/rscrypto)
[![MSRV 1.91.0](https://img.shields.io/badge/MSRV-1.91.0-blue)](Cargo.toml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/rscrypto)](#license)

`rscrypto` puts cryptographic primitives, cryptographic and fast hashes,
password hashing, and checksums behind one feature model. Target-gated SIMD and
assembly accelerate portable Rust backends without a production C/FFI,
OpenSSL, or system-library dependency.

`rscrypto` is a primitives crate, not a TLS stack, PKI toolkit, key store, or
protocol implementation.

## Measured performance

The historical benchmark snapshots contain comparisons that measured different
work: ML-KEM rows mixed entropy sources and key preparation, and Argon2 rows
passed different salt lengths to dryoc. The affected ratios and aggregates that
include them are withdrawn as performance claims pending corrected measurements.
Their numerical impact has not been measured.

The [`benchmark overview`](benchmark_results/OVERVIEW.md) retains the historical
record. See the [`comparison contracts`](docs/benchmarking.md#ml-kem-and-argon2-comparison-contracts)
for the corrected workloads and validation requirements.

## Assurance

Security claims fail closed: missing or stale evidence removes the claim rather
than weakening the gate.

- Correctness evidence combines NIST, RFC, upstream, and Wycheproof vectors
  with separate implementations, properties, negative tests, and Miri.
- Fuzz targets exercise production implementations across primitive, parser,
  state-machine, and trait boundaries. Minimized seeds replay as tests, with a
  separate sanitizer lane.
- Portable-versus-accelerated differential tests cover lengths, alignments,
  tails, state transitions, dispatch, and fallback behavior on native targets.
- The constant-time harness inventories exact operations in [`ct.toml`](ct.toml)
  and combines optimized linked-binary inspection, BINSEC proofs for declared
  fixed-shape kernels, and DudeCT timing tests for declared end-to-end cases.
- Secret owners mask `Debug`, omit ordinary equality and implicit duplication,
  and overwrite their initialized bytes on drop. Verification failures are opaque; failed AEAD opens
  clear caller output buffers.

A constant-time claim exists only when evidence for the required target,
feature, compiler, profile, and operation passes. Source that looks branchless
is not treated as proof.

Inspect the [`test evidence`](docs/test-vector-coverage.md),
[`constant-time model`](docs/constant-time.md),
[`secret lifecycle`](docs/secret-lifecycle.md), and
[`threat model`](THREAT_MODEL.md).

The remaining independent-review gap is a third-party security audit. The
project cannot currently fund one. Automated evidence does not replace that
review, so `rscrypto` does not claim to be audited, FIPS 140-3 validated,
formally verified, or constant time as a whole crate.

Report suspected vulnerabilities through
[GitHub Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new)
under the [`SECURITY.md`](SECURITY.md) process, not a public issue.

## Install only what you use

Minimal `no_std` SHA-2 build:

```toml
[dependencies]
rscrypto = { version = "0.9", default-features = false, features = ["sha2"] }
```

Full primitive stack with OS randomness enabled:

```toml
[dependencies]
rscrypto = { version = "0.9", features = ["full", "getrandom"] }
```

The default feature is `std`; `default-features = false` removes it. Enable
`getrandom` only for APIs that obtain salts, keys, nonces, or RSA key-generation
entropy from the operating system. The [`feature guide`](docs/features.md)
owns exact dependencies and deployment controls.

## Quick start

```rust
use rscrypto::Sha256;

let one_shot = Sha256::digest(b"hello world");

let mut hasher = Sha256::new();
hasher.update(b"hello ");
hasher.update(b"world");

assert_eq!(hasher.finalize(), one_shot);
```

Hash APIs support one-shot and streaming use. Runnable workflows for AEAD,
signatures, RSA, P-256 ECDH, X25519, ML-KEM, password hashing, and backend introspection are
in [`examples/README.md`](examples/README.md).

## Primitive and feature map

| Family | Included | Enable |
| --- | --- | --- |
| Checksums | CRC-16, CRC-24, CRC-32, CRC-32C, CRC-64/XZ, CRC-64/NVMe | `checksums` or leaf features |
| Cryptographic hashes | SHA-2, SHA-3, SHAKE, cSHAKE, BLAKE2, BLAKE3, Ascon-Hash/XOF/CXOF | `crypto-hashes` or leaf features |
| Fast hashes | XXH3-64/128, RapidHash V3-64 | `fast-hashes` or leaf features |
| MACs and KDFs | HMAC-SHA-2/SHA-3, KMAC128/256, Poly1305, HKDF-SHA-2, PBKDF2-HMAC-SHA-2 | `macs`, `kdfs`, or leaf features |
| Password hashing | Argon2d/i/id, scrypt, bounded PHC password records | `password-hashing` or leaf features |
| Signatures and RSA | ECDSA P-256/P-384, Ed25519, RSA signing, verification, encryption, and key generation | `signatures` or leaf features |
| Key exchange and KEMs | P-256 ECDH, X25519, ML-KEM-512/768/1024 | `key-exchange` or leaf features |
| AEADs | AES-GCM, AES-GCM-SIV, AES-SIV-CMAC, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, Ascon-AEAD128 | `aead` or leaf features |

Use [docs.rs](https://docs.rs/rscrypto) for exact types and methods. Use the
[`migration guide`](docs/migration.md) when replacing another library.

## Platforms and dispatch

The portable Rust implementation is the byte-for-byte authority. Compile-time
target support and, with `std`, detected runtime CPU capabilities select
eligible SIMD or assembly kernels. Unsupported acceleration falls back to
portable Rust.

The [`platform guide`](docs/platforms.md) owns the target matrix, dispatch
model, `no_std` coverage, and the limits of `portable-only`.

## Project

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing code. Published changes
live in [`CHANGELOG.md`](CHANGELOG.md).

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your
option.
