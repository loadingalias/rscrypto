# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![Docs.rs](https://docs.rs/rscrypto/badge.svg)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml)
[![RSA Gates](https://github.com/loadingalias/rscrypto/actions/workflows/rsa.yaml/badge.svg?branch=main)](https://github.com/loadingalias/rscrypto/actions/workflows/rsa.yaml)
[![MSRV 1.91.0](https://img.shields.io/badge/MSRV-1.91.0-blue)](Cargo.toml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/rscrypto)](#license)

**Pure Rust Cryptography: RSA, ECDSA, Ed25519, X25519, ML-KEM, AEADs, crypto/fast hashes, KDFs, password hashing, CRCs, `no_std`/WASM, and hardware acceleration in one dependency.**

`rscrypto` is a single primitive stack for projects that care about binary size, deployment control, and speed without dragging in C/FFI, OpenSSL, or system library coupling.

Use one leaf feature for one primitive, a group for a subset of primitives, or `full` for the full crate surface. The portable Rust backend is always present. SIMD and ASM are only accelerators.

**Current Benchmark Evidence:** `1.59x` geomean across the Linux runners vs the fastest-external competitors with `4,052 / 6,750` wins and `6,101 / 6,750` wins-or-ties.

macOS Apple Silicon local evidence: `1.37x` geomean vs fastest-external competitors with `382 / 774` wins and `708 / 774` wins-or-ties.

Raw runs, methodology, and known losses are in
[`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md).

<p align="center">
  <img alt="rscrypto benchmark chart: 1.59x Linux and 1.37x Apple Silicon fastest-matched geomeans, checksums at 5.18x against crc-fast, crc, crc32fast, crc32c, and crc64fast, plus primitive geomean bars and M1 MBP Apple Silicon notes."
       src="assets/readme/perf.svg"
       width="640">
</p>

<p align="center">
  <i>Chart: benchmark scorecard. Values above <code>1.00x</code> mean <code>rscrypto</code> is faster than the fastest matched external implementation.</i>
</p>

## Why rscrypto?

- One feature model for hashes, MACs, KDFs, password hashing, AEADs,
  signatures, key exchange, ML-KEM, RSA, and checksums.
- No OpenSSL or production C/FFI dependency.
- Concrete types, scoped errors, typed keys/nonces/tags, and opaque
  verification failures across the supported primitives.
- Portable Rust implementations are the reference path; SIMD and ASM are
  accelerators tested against that path.
- `no_std`, WASM, server, CLI, embedded, and audit-constrained builds use the
  same leaf-feature model.
- Public validation evidence covers vectors, differential tests, fuzz corpus
  replay, Miri, backend equivalence, and scoped constant-time release gates.

`rscrypto` is a primitives crate. It is not a TLS stack, PKI toolkit, key store, or protocol implementation. It is not a FIPS 140-3 validated module, third-party audited, formally verified, or a whole-crate constant-time claim today.

## Install

Minimal `no_std` SHA-2 build:

```toml
[dependencies]
rscrypto = { version = "0.6.4", default-features = false, features = ["sha2"] }
```

Full primitive stack with OS randomness enabled:

```toml
[dependencies]
rscrypto = { version = "0.6.4", features = ["full", "getrandom"] }
```

Use `default-features = false` for `no_std` builds. Enable `getrandom` only when you need APIs that generate salts, keys, nonces, or RSA key-gen entropy from the operating system.

## Quick Start

```rust
use rscrypto::Sha256;

let one_shot = Sha256::digest(b"hello world");

let mut h = Sha256::new();
h.update(b"hello ");
h.update(b"world");

assert_eq!(h.finalize(), one_shot);
```

The common API shape is one-shot when convenient and streaming when needed.

## Common Workflows

| Task | Feature | Start Here |
|---|---|---|
| AEAD seal/open | `chacha20poly1305,getrandom` | [`examples/aead_seal_open.rs`](examples/aead_seal_open.rs) |
| Ed25519 and ECDSA signatures | `ed25519,ecdsa-p256,getrandom` | [`examples/signatures.rs`](examples/signatures.rs) |
| RSA-PSS verification | `rsa` | [`examples/rsa_pss_verify.rs`](examples/rsa_pss_verify.rs) |
| ML-KEM shared secret | `ml-kem,getrandom` | [`examples/mlkem_encapsulation.rs`](examples/mlkem_encapsulation.rs) |
| Argon2id and scrypt password hashing | `password-hashing,getrandom` | [`examples/password_hashing.rs`](examples/password_hashing.rs) |

Use [`docs/types.md`](docs/types.md) when you need the full type map, and
[`docs/features.md`](docs/features.md) when you need the smallest feature set.

## What You Get

| Need | Included | Feature Path |
|---|---|---|
| Cryptographic Hashes | SHA-2, SHA-3, SHAKE, cSHAKE128/256, BLAKE2, BLAKE3, Ascon-Hash/XOF/CXOF | `hashes` or leaf features |
| MACs & KDFs | HMAC-SHA-2/SHA-3, KMAC128/256, standalone Poly1305, HKDF-SHA-2, PBKDF2-HMAC-SHA-2 | `auth` or leaf features |
| Password Hashing | Argon2d/i/id, scrypt, PHC string encode/verify | `auth`, `argon2`, `scrypt`, `phc-strings` |
| Public-Key Primitives | ECDSA P-256/P-384 signing/verification, Ed25519 signatures, RSA signing/verification/OAEP/RSAES-PKCS1-v1_5/key generation, X25519 key exchange, ML-KEM-512/768/1024 KEMs | `auth`, `signatures`, `key-exchange`, `ecdsa`, `ecdsa-p256`, `ecdsa-p384`, `ed25519`, `rsa`, `x25519`, `ml-kem` |
| AEAD Encryption | AES-128/256-GCM, AES-128/256-GCM-SIV, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, Ascon-AEAD128 | `aead` or leaf features |
| Checksums | CRC-16, CRC-24, CRC-32, CRC-32C, CRC-64/XZ, CRC-64/NVMe | `checksums` or leaf features |
| Fast Hashes | XXH3-64/128, RapidHash 64/128 | `xxh3`, `rapidhash` |

Flags are layered by use:

- **Leaf Primitives:** `sha2`, `blake3`, `aes-gcm`, `ed25519`, `x25519`, `ml-kem`, `crc32`, etc.
- **Families/Groups:** `hashes`, `checksums`, `macs`, `kdfs`, `password-hashing`, `aead`, `signatures`, `key-exchange`.
- **Deployment Controls:** `std`, `alloc`, `getrandom`, `parallel`, `serde`, `portable-only`; `serde-secrets` explicitly opts secret material into `serde`.

Full Feature Inventory: [`docs/features.md`](docs/features.md).
Public Type Inventory: [`docs/types.md`](docs/types.md).

## Constant-Time Boundaries

`rscrypto` makes only release-bound, scoped constant-time claims for
secret-bearing operations, not for every function in the crate. `ct.toml`
records the candidate primitive/configuration set; it does not create a public
claim by itself. A claim exists only where the matching signed GitHub release
includes an attested `rscrypto-X.Y.Z-ct-evidence.tar.gz` bundle that passes all
required gates for that exact version, commit, target, profile, and feature set.

The main candidate secret-bearing surfaces in [`ct.toml`](ct.toml) are
MAC/tag verification, AEAD authentication failure shape, X25519 scalar
multiplication, Ed25519 signing and secret public-key derivation, ECDSA
P-256/P-384 blinded signing, ML-KEM-512/768/1024 key gen,
encapsulation, decapsulation secret surfaces, RSA private sign/decrypt leaves,
and selected password-verification comparisons.

Public parsing, unlisted key gen, OS randomness, raw hashes, checksums,
non-cryptographic hashes, benchmark paths, and public-key verification math are
not blanket constant-time claims. See [`docs/constant-time.md`](docs/constant-time.md)
for the exact claim and verification model and [`docs/compliance.md`](docs/compliance.md)
for review boundaries. Releases through `v0.6.4` do not contain this bundle and
therefore carry no release-bound constant-time claim.

## Portability & Accel

`rscrypto` keeps the portable Rust path as the byte-for-byte authority. ISA kernels are selected only when the target and runtime CPU support them.

| Target family | Acceleration examples |
|---|---|
| x86 / x86_64 | SSE4.2, AVX2, AVX-512, AES-NI, SHA-NI, VAES, VPCLMULQDQ |
| Arm / AArch64 / Apple Silicon | NEON, AES, PMULL, SHA2, SHA3, SVE2-PMULL |
| IBM Z | CPACF, MSA, VGFM, z/Vector ML-KEM arithmetic |
| POWER / ppc64le | POWER8/9/10 vector and crypto extensions |
| RISC-V | RVV, Zbc, Zvkned, Zvbc |
| WASM | SIMD128 where available, portable fallback everywhere |

Full platform matrix: [`docs/platforms.md`](docs/platforms.md).

## Security

`rscrypto` makes scoped constant-time claims only when a matching release
publishes the required evidence bundle, never for every API or build.
Secret-bearing types zeroize on drop and
mask `Debug`; verification failures use opaque errors; failed AEAD opens wipe
output buffers. Release artifacts are signed-tag gated, published through
crates.io Trusted Publishing, and covered by GitHub build provenance
attestations.

No third-party audit, FIPS 140-3 certificate, or formal whole-crate proof is
claimed today. Report vulnerabilities through
[GitHub Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new)
or [`SECURITY.md`](SECURITY.md), not public issues.

## Docs

- Start: [docs.rs](https://docs.rs/rscrypto), [`examples/`](examples/),
  [`docs/features.md`](docs/features.md), [`docs/types.md`](docs/types.md)
- Security and review: [`SECURITY.md`](SECURITY.md),
  [`THREAT_MODEL.md`](THREAT_MODEL.md), [`docs/constant-time.md`](docs/constant-time.md),
  [`docs/compliance.md`](docs/compliance.md)
- Evidence: [`docs/test-vector-coverage.md`](docs/test-vector-coverage.md),
  [`docs/platforms.md`](docs/platforms.md),
  [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md)
- Switching crates: [`docs/migration/`](docs/migration/)
- Releases: [`CHANGELOG.md`](CHANGELOG.md)

## MSRV

Rust **1.91.0**.

The pinned nightly in [`rust-toolchain.toml`](rust-toolchain.toml) is used for Miri, fuzzing, and exotic-architecture checks.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
