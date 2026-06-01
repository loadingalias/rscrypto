# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![Docs.rs](https://docs.rs/rscrypto/badge.svg)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml)
[![RSA Gates](https://github.com/loadingalias/rscrypto/actions/workflows/rsa.yaml/badge.svg?branch=main)](https://github.com/loadingalias/rscrypto/actions/workflows/rsa.yaml)
[![MSRV 1.91.0](https://img.shields.io/badge/MSRV-1.91.0-blue)](Cargo.toml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/rscrypto)](#license)

**Pure Rust cryptography: RSA, Ed25519, X25519, AEADs, hashes, KDFs, password hashing, CRCs, `no_std`, WASM, and hardware acceleration in one dependency.**

`rscrypto` is a single primitive stack for projects that care about binary size, deployment control, and speed without a mandatory C, OpenSSL, or system lib story.

Use one leaf feature for one primitive, a group for a subset of primitives, or `full` for the whole shebang. The portable Rust backend is always present. SIMD and assembly are only accelerators.

**Current benchmark scorecards:** Linux CI is `1.61x` fastest-external geomean with `3,545 / 5,832` wins and `5,210 / 5,832` wins-or-ties. Apple Silicon (MBP M1, macOS/aarch64 local full run) is `1.25x` fastest-external geomean with `235 / 463` wins and `450 / 463` wins-or-ties.

<p align="center">
  <img alt="rscrypto benchmark scorecard: 1.61x fastest-external geomean across Linux CI with 3,545 wins and 5,210 wins-or-ties out of 5,832 matched benchmark comparisons."
       src="assets/readme/perf.svg"
       width="640">
</p>

<p align="center">
  <i>Chart: 2026-05-27 Linux CI benchmark pass. Apple Silicon numbers from the 2026-06-01 MBP M1 local full run are listed below. Values above <code>1.00x</code> mean <code>rscrypto</code> is faster than the fastest matched Rust baseline.</i>
</p>

## Why rscrypto?

- **RSA is now a first class citizen.** Strict DER import/export, RSA-PSS, RSASSA-PKCS1-v1_5, OAEP, RSAES-PKCS1-v1_5, key generation, X.509/JWT/COSE/TLS profile mapping, blinded private operations, and reusable scratch APIs.
- **One coherent primitive stack.** Avoid composing half a dozen crates with different APIs, feature models, and security conventions.
- **Small builds stay small.** Enable `sha2`, `blake3`, `aes-gcm`, `chacha20poly1305`, `ed25519`, `x25519`, `argon2`, or any other leaf without pulling in the world.
- **Portable Rust is the source of truth.** SIMD and ASM paths are accelerators; the portable backend remains the reference implementation.
- **Hardware dispatch is built in.** x86/x86_64, Arm/AArch64, Apple Silicon, IBM Z, IBM POWER, RISC-V, and WASM all have portable fallbacks, with optimized kernels where they pay.
- **`no_std` is a first-class target.** Server, CLI, embedded, bare-metal, and WASM builds use the same crate and feature model.
- **Audit knobs are explicit.** `portable-only` forces runtime dispatch to the constant-time portable backend; `getrandom`, `serde`, and `rayon` are opt-in.
- **Security hygiene is part of the API.** Opaque verification errors, constant-time equality, zeroized secret types, strict arithmetic, official vectors, fuzzing, Miri, and cross-CPU CI are built into the project discipline.

`rscrypto` is a primitives crate. It is **not** a TLS stack, PKI toolkit, protocol implementation, or FIPS 140-3 validated module.

## Install

Minimal `no_std` SHA-2 build:

```toml
[dependencies]
rscrypto = { version = "0.3.1", default-features = false, features = ["sha2"] }
```

Full primitive stack with OS randomness enabled:

```toml
[dependencies]
rscrypto = { version = "0.3.1", features = ["full", "getrandom"] }
```

Use `default-features = false` for constrained `no_std` builds. Enable `getrandom` only when you need APIs that generate salts, keys, or nonces from the operating system.

## Quick Start

```rust
use rscrypto::{Digest, Sha256};

let one_shot = Sha256::digest(b"hello world");

let mut h = Sha256::new();
h.update(b"hello ");
h.update(b"world");

assert_eq!(h.finalize(), one_shot);
```

The common API shape is deliberately simple/boring: one-shot when convenient, streaming when it's needed.

## Verify RSA Signatures

```toml
[dependencies]
rscrypto = { version = "0.3.1", default-features = false, features = ["rsa"] }
```

```rust
use rscrypto::{RsaPssProfile, RsaPublicKey};

fn verify_release_signature(public_key_der: &[u8], message: &[u8], signature: &[u8]) -> bool {
  let Ok(key) = RsaPublicKey::from_spki_der(public_key_der) else {
    return false;
  };

  key.verify_pss(RsaPssProfile::Sha256, message, signature).is_ok()
}
```

For repeated verification with the same key, allocate scratch once:

```rust
use rscrypto::{RsaPssProfile, RsaPublicKey, RsaSignatureProfile};

fn verify_batch(public_key_der: &[u8], signed_messages: &[(&[u8], &[u8])]) -> bool {
  let Ok(key) = RsaPublicKey::from_spki_der(public_key_der) else {
    return false;
  };
  let mut scratch = key.public_scratch();

  signed_messages.iter().all(|(message, signature)| {
    key
      .verify_signature_with_scratch(
        RsaSignatureProfile::pss(RsaPssProfile::Sha256),
        message,
        signature,
        &mut scratch,
      )
      .is_ok()
  })
}
```

Enable `getrandom` for OS-backed RSA key generation, signing salt/blinding, OAEP encryption randomness, and private-operation blinding:

```toml
[dependencies]
rscrypto = { version = "0.3.1", default-features = false, features = ["rsa", "getrandom"] }
```

## Encrypt Data

```toml
[dependencies]
rscrypto = { version = "0.3.1", default-features = false, features = ["chacha20poly1305"] }
```

```rust
use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key, aead::Nonce96};

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

## Hash Passwords

```toml
[dependencies]
rscrypto = { version = "0.3.1", default-features = false, features = ["argon2", "phc-strings", "getrandom"] }
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

## What You Get

| Need | Included | Feature path |
|---|---|---|
| Cryptographic hashes | SHA-2, SHA-3, SHAKE, cSHAKE256, BLAKE2, BLAKE3, Ascon-Hash/XOF/CXOF | `hashes` or leaf features |
| MACs and KDFs | HMAC-SHA-2, KMAC256, HKDF-SHA-2, PBKDF2-HMAC-SHA-2 | `auth` or leaf features |
| Password hashing | Argon2d/i/id, scrypt, PHC string encode/verify | `auth`, `argon2`, `scrypt`, `phc-strings` |
| Public-key primitives | Ed25519 signatures, RSA signing/verification/OAEP/RSAES-PKCS1-v1_5/key generation, X25519 key exchange | `auth`, `signatures`, `ed25519`, `rsa`, `x25519` |
| AEAD encryption | AES-128/256-GCM, AES-128/256-GCM-SIV, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, Ascon-AEAD128 | `aead` or leaf features |
| Checksums | CRC-16, CRC-24, CRC-32, CRC-32C, CRC-64/XZ, CRC-64/NVMe | `checksums` or leaf features |
| Fast non-crypto hashes | XXH3-64/128, RapidHash 64/128 | `xxh3`, `rapidhash` |

Fast non-cryptographic hashes and CRCs are for indexing, checksumming, dedup, and integrity plumbing. Do not use them for passwords, signatures, MACs, key derivation, or authentication... it's obviously not safe.

Feature flags are layered deliberately:

- **Leaf primitives:** `sha2`, `blake3`, `aes-gcm`, `ed25519`, `x25519`, `crc32`, etc.
- **Families/Groups:** `hashes`, `checksums`, `macs`, `kdfs`, `password-hashing`, `aead`, `signatures`, `key-exchange`.
- **Deployment controls:** `std`, `alloc`, `getrandom`, `parallel`, `serde`, `serde-secrets`, `portable-only`.

Full feature inventory: [`docs/features.md`](docs/features.md). Public type inventory: [`docs/types.md`](docs/types.md).

## Performance

Current public benchmark evidence comes from two passes:

- Linux CI: 2026-05-27, commit `26845c8`, nine Linux runners. This run is filter-based and does not include Argon2, scrypt, or Ascon-AEAD rows.
- Apple Silicon: 2026-06-01, commit `b06b946`, local MBP M1 macOS/aarch64 full run, including Argon2, scrypt, and Ascon-AEAD rows.

Speedup is `external_crate_time / rscrypto_time`; values above `1.00x` mean `rscrypto` is faster. Do not combine the Linux and Apple Silicon totals as one aggregate because they were collected on different commits and benchmark scopes.

| Area | Compared against | Result |
|---|---|---:|
| **Linux CI fastest external** | strongest matched Rust baseline per case | **1.61x geomean** |
| Linux CI scorecard | fastest external | **3,545 wins / 5,832 pairs** |
| Linux CI wins or ties | fastest external | **5,210 / 5,832 pairs** |
| **Apple Silicon fastest external** | strongest matched Rust baseline per case | **1.25x geomean** |
| Apple Silicon scorecard | fastest external | **235 wins / 463 pairs** |
| Apple Silicon wins or ties | fastest external | **450 / 463 pairs** |
| Checksums | Linux CI / Apple Silicon | **5.03x / 2.76x geomean** |
| SHA-3 / SHAKE | Linux CI / Apple Silicon | **Linux: 2.15x / 1.86x; Apple Silicon: 0.94x / 1.32x geomean** (platform-sensitive) |
| BLAKE3, `>=64 KiB` | Linux CI / Apple Silicon | **2.31x / 1.80x geomean** |
| AEAD | Linux CI / Apple Silicon | **1.57x / 1.47x geomean** |
| RSA import + verify | Linux CI / Apple Silicon | **1.32x, 76% wins / 1.45x, 100% wins** |
| RSA verify only | Linux CI / Apple Silicon | **0.98x / 1.19x geomean** |
| Ed25519 sign / verify | Linux CI / Apple Silicon | **Linux: 1.14x / 1.00x; Apple Silicon: 1.02x / 1.00x geomean** |
| X25519 | Linux CI / Apple Silicon | **0.95x / 1.00x geomean** |

The honest weak spots right now: Linux CI still shows PBKDF2-SHA256 at `iters=1` at 0.81x, X25519 Diffie-Hellman at 0.92x, RSA-4096 verification at 0.94x, and small-message AEAD overhead on plenty of 1-byte and 32-byte rows. Apple Silicon still has BLAKE3 64 KiB losses, HMAC-SHA256 bulk pressure against `aws-lc-rs`, empty-message ChaCha20-Poly1305 overhead, and SHA3-256 streaming losses; SHA-3/SHAKE should be described per-platform because the Linux 2.15x SHA-3 result does not carry to the MBP M1 run. See [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md) for raw runs, methodology, platform scorecards, and loss tables.

## Portability And Acceleration

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

## Security

- Constant-time MAC, AEAD, and signature verification.
- Opaque verification errors that avoid leaking failure details.
- Secret-bearing types zeroize on drop and mask `Debug`.
- Strict arithmetic for counters, lengths, offsets, and indices.
- AEAD failed-open paths wipe output buffers.
- Portable and accelerated backends are differentially tested for byte-identical output.
- Official test vectors, fuzz corpus replay, Miri, `cargo deny`, and `cargo audit` run in CI.
- RSA private-operation release claims require the manual Miri and first-order leakage gates in
  [`docs/security/rsa-side-channel-audit.md`](docs/security/rsa-side-channel-audit.md).

Read [`docs/security.md`](docs/security.md) before shipping cryptographic code. For compliance posture, see [`docs/compliance.md`](docs/compliance.md).

Vulnerabilities should be reported through [GitHub Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new) or the process in [`SECURITY.md`](SECURITY.md).

## Docs

- API reference: [docs.rs/rscrypto](https://docs.rs/rscrypto)
- Examples: [`examples/`](examples/)
- Feature flags: [`docs/features.md`](docs/features.md)
- Public type inventory: [`docs/types.md`](docs/types.md)
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
