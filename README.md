# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![Docs.rs](https://docs.rs/rscrypto/badge.svg)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml)
[![MSRV 1.91.0](https://img.shields.io/badge/MSRV-1.91.0-blue)](Cargo.toml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/rscrypto)](#license)

**A Rust crypto crate for systems that need performance, portability, and small dependency graphs.**

`rscrypto` gives you hashes, MACs, KDFs, password hashing, signatures, key exchange, AEAD encryption, checksums, and fast non-cryptographic hashes behind one consistent API.

Use one leaf feature for a tiny build. Use `full` for the whole toolbox. No C. No FFI. No OpenSSL. No forced third-party dependency stack.

<p align="center">
  <img alt="rscrypto benchmark scorecard: 1.75x geomean speedup across Linux with 3,717 wins out of 5,796 matched benchmark comparisons."
       src="assets/readme/perf.svg"
       width="640">
</p>

<p align="center">
  <i>Latest public benchmark pass. Values above 1.00x mean <code>rscrypto</code> is faster than the compared Rust baseline.</i>
</p>

## Why rscrypto?

- **One crate instead of a stitched-together crypto stack.** Covers the common primitive families without making you compose half a dozen APIs.
- **Small when you want small.** Leaf features like `sha2`, `blake3`, `aes-gcm`, `chacha20poly1305`, `ed25519`, `x25519`, and `argon2` keep builds focused.
- **Portable Rust first.** The portable implementation is the source of truth; optimized kernels are accelerators, not separate logic forks.
- **Hardware acceleration where it matters.** Runtime and compile-time dispatch use ISA-specific kernels on supported x86/x86_64, Arm/AArch64, IBM Z, POWER, RISC-V, Apple Silicon, and WASM targets.
- **`no_std` ready.** Built for servers, CLIs, embedded targets, bare-metal experiments, and WASM.
- **Zero default third-party dependencies.** `getrandom`, `serde`, and `rayon` are opt-in.
- **Security hygiene by default.** Constant-time verification, opaque errors, zeroized secrets, strict arithmetic, official vectors, fuzzing, Miri, and CI across multiple CPU targets.

`rscrypto` is a primitives crate. It is **not** a TLS stack, PKI toolkit, protocol implementation, or FIPS 140-3 validated module.

## Install

Minimal `no_std` SHA-2 build:

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["sha2"] }
```

Full toolbox with OS randomness enabled:

```toml
[dependencies]
rscrypto = { version = "0.1", features = ["full", "getrandom"] }
```

Use `default-features = false` for constrained `no_std` builds. Enable `getrandom` only when you need APIs that generate salts, keys, or nonces from the operating system.

## Quick start

```rust
use rscrypto::{Sha256, prelude::*};

let one_shot = Sha256::digest(b"hello world");

let mut h = Sha256::new();
h.update(b"hello ");
h.update(b"world");

assert_eq!(h.finalize(), one_shot);
```

That is the basic API shape: one-shot when convenient, streaming when needed.

## Encrypt data

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["chacha20poly1305"] }
```

```rust
use rscrypto::{aead::Nonce96, Aead, ChaCha20Poly1305, ChaCha20Poly1305Key};

let key = ChaCha20Poly1305Key::from_bytes([0x11; 32]);
let nonce = Nonce96::from_bytes([0x22; Nonce96::LENGTH]);
let cipher = ChaCha20Poly1305::new(&key);

let aad = b"transfer:v1";
let mut message = *b"pay bob 10";

let tag = cipher
    .encrypt_in_place(&nonce, aad, &mut message)
    .expect("encryption succeeds");

cipher
    .decrypt_in_place(&nonce, aad, &mut message, &tag)
    .expect("authentication succeeds");

assert_eq!(&message, b"pay bob 10");
```

## Hash passwords

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["argon2", "phc-strings", "getrandom"] }
```

```rust
use rscrypto::{Argon2Params, Argon2VerifyPolicy, Argon2id};

let password = b"correct horse battery staple";
let params = Argon2Params::new().build().expect("valid Argon2 params");
let encoded = Argon2id::hash_string(&params, password).expect("password hash created");

assert!(
    Argon2id::verify_string_with_policy(
        password,
        &encoded,
        &Argon2VerifyPolicy::default(),
    )
    .is_ok()
);
```

## What you get

| Need | Included | Feature path |
|---|---|---|
| Cryptographic hashes | SHA-2, SHA-3, SHAKE, cSHAKE256, BLAKE2, BLAKE3, Ascon-Hash/XOF/CXOF | `hashes` or leaf features |
| MACs and KDFs | HMAC-SHA-2, KMAC256, HKDF-SHA-2, PBKDF2-HMAC-SHA-2 | `auth` or leaf features |
| Password hashing | Argon2d/i/id, scrypt, PHC string encode/verify | `auth`, `argon2`, `scrypt`, `phc-strings` |
| Public-key primitives | Ed25519 signatures, X25519 key exchange | `auth`, `ed25519`, `x25519` |
| AEAD encryption | AES-128/256-GCM, AES-128/256-GCM-SIV, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, Ascon-AEAD128 | `aead` or leaf features |
| Checksums | CRC-16, CRC-24, CRC-32, CRC-32C, CRC-64/XZ, CRC-64/NVMe | `checksums` or leaf features |
| Fast non-crypto hashes | XXH3-64/128, RapidHash 64/128 | `xxh3`, `rapidhash` |

Fast non-cryptographic hashes and CRCs are for indexing, checksumming, deduplication, and integrity plumbing. Do not use them for passwords, signatures, MACs, key derivation, or authentication.

BLAKE3 uses the standard BLAKE3 construction, but under the `rscrypto` deployment model: leaf-feature builds, a portable source of truth, optional parallel and ISA acceleration, differential testing, and `portable-only` for audit-constrained builds.

Feature flags are layered deliberately:

- **Leaf primitives:** `sha2`, `blake3`, `aes-gcm`, `ed25519`, `x25519`, `crc32`, and similar.
- **Families:** `hashes`, `checksums`, `macs`, `kdfs`, `password-hashing`, `aead`, `signatures`, `key-exchange`.
- **Deployment controls:** `std`, `alloc`, `getrandom`, `parallel`, `serde`, `serde-secrets`, `portable-only`.

Full feature inventory: [`docs/features.md`](docs/features.md). Public type inventory: [`docs/types.md`](docs/types.md).

## Performance

Latest public benchmark pass: Linux plus macOS Apple Silicon, compared against established Rust baselines. Speedup is `external_crate_time / rscrypto_time`; values above `1.00x` mean `rscrypto` is faster.

| Area | Compared against | Result |
|---|---|---:|
| **Linux overall** | strongest matched Rust baselines | **1.75x geomean** |
| Matched comparisons | Linux | **3,717 wins / 5,796 pairs** |
| Checksums | `crc`, `crc32fast`, `crc64fast`, `crc-fast` | **4.41x geomean** |
| SHA-3 / SHAKE | RustCrypto `sha3` | **2.18x / 2.60x geomean** |
| BLAKE3, `>=64 KiB` | `blake3` | **2.37x geomean** |
| AEAD | RustCrypto AEADs, `aegis` | **1.84x geomean** |
| Ed25519 signing | `ed25519-dalek` | **1.57x geomean** |

macOS Apple Silicon results include AEAD at **2.60x** and checksums at **4.18x**, with broad parity on SHA-2 and SHA-3.

See [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md) for raw runs, methodology, host details, and per-platform scorecards.

## Portability and acceleration

`rscrypto` keeps the portable Rust path as the byte-for-byte authority. ISA-specific kernels are selected only when the target and runtime CPU support them.

| Target family | Acceleration examples |
|---|---|
| x86 / x86_64 | SSE4.2, AVX2, AVX-512, AES-NI, SHA-NI, VAES, VPCLMULQDQ |
| Arm / AArch64 / Apple Silicon | NEON, AES, PMULL, SHA2, SHA3, SVE2-PMULL |
| IBM Z | CPACF, MSA, VGFM |
| POWER / ppc64le | POWER8/9/10 vector and crypto extensions |
| RISC-V | RVV, Zbc, Zvkned, Zvbc |
| WASM | SIMD128 where available, portable fallback everywhere |

Use `portable-only` when you need deterministic dispatch, audit-constrained builds, or a portable backend only.

Full platform matrix: [`docs/platforms.md`](docs/platforms.md). Architecture notes: [`docs/architecture.md`](docs/architecture.md).

## Security posture

- Constant-time MAC, AEAD, and signature verification.
- Opaque verification errors that avoid leaking failure details.
- Secret-bearing types zeroize on drop and mask `Debug`.
- Strict arithmetic for counters, lengths, offsets, and indices.
- AEAD failed-open paths wipe output buffers.
- Portable and accelerated backends are differentially tested for byte-identical output.
- Official test vectors, fuzz corpus replay, Miri, `cargo deny`, and `cargo audit` run in CI.

Read [`docs/security.md`](docs/security.md) before shipping cryptographic code. For compliance posture, see [`docs/compliance.md`](docs/compliance.md).

Vulnerabilities should be reported through [GitHub Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new) or the process in [`SECURITY.md`](SECURITY.md).

## Documentation

- API reference: [docs.rs/rscrypto](https://docs.rs/rscrypto)
- Examples: [`examples/`](examples/)
- Feature flags: [`docs/features.md`](docs/features.md)
- Platform matrix: [`docs/platforms.md`](docs/platforms.md)
- Security guidance: [`docs/security.md`](docs/security.md)
- Migration guides: [`docs/migration/`](docs/migration/)
- Benchmark methodology: [`docs/benchmarking.md`](docs/benchmarking.md)
- Benchmarks: [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md)
- Release history: [`CHANGELOG.md`](CHANGELOG.md)

## MSRV

Rust **1.91.0**.

The pinned nightly in [`rust-toolchain.toml`](rust-toolchain.toml) is used for Miri, fuzzing, and exotic-architecture checks.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
