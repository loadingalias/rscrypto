# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![Docs.rs](https://docs.rs/rscrypto/badge.svg)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml)
[![RSA Gates](https://github.com/loadingalias/rscrypto/actions/workflows/rsa.yaml/badge.svg?branch=main)](https://github.com/loadingalias/rscrypto/actions/workflows/rsa.yaml)
[![MSRV 1.91.0](https://img.shields.io/badge/MSRV-1.91.0-blue)](Cargo.toml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/rscrypto)](#license)

**Pure Rust cryptography: RSA, ECDSA, Ed25519, X25519, AEADs, hashes, KDFs, password hashing, CRCs, `no_std`, WASM, and hardware acceleration in one dependency.**

`rscrypto` is a single primitive stack for projects that care about binary size, deployment control, and speed without dragging in mandatory C, OpenSSL, or system library coupling.

Use one leaf feature for one primitive, a group for a subset of primitives, or `full` for the full crate surface. The portable Rust backend is always present. SIMD and ASM are only accelerators.

**Latest published benchmark evidence:** Linux CI reports `1.58x` fastest-external geomean with `4,102 / 6,750` wins and `6,056 / 6,750` wins-or-ties, including ML-KEM-512/768/1024 keygen, encapsulation, and decapsulation rows.

<p align="center">
  <img alt="rscrypto benchmark scorecard: 1.58x fastest-external geomean across Linux CI with 4,102 wins and 6,056 wins-or-ties out of 6,750 matched benchmark comparisons."
       src="assets/readme/perf.svg"
       width="640">
</p>

<p align="center">
  <i>Chart: 06/19/2026 Linux CI bench pass. Values above <code>1.00x</code> mean <code>rscrypto</code> is faster than the fastest matched Rust baseline.</i>
</p>

## Why rscrypto?

- **RSA is a first class citizen.** Strict DER import/export, RSA-PSS, RSASSA-PKCS1-v1_5, OAEP, RSAES-PKCS1-v1_5, FIPS 186-5 A.1.3 probable-prime key generation in code, X.509/JWT/COSE/TLS profile mapping, blinded private operations, and reusable scratch APIs.
- **One coherent primitive stack.** Avoid composing a dozen crates with different APIs, feature models, and security conventions.
- **Small builds stay small.** Enable `sha2`, `blake3`, `aes-gcm`, `chacha20poly1305`, `ed25519`, `x25519`, `argon2`, or any other leaf without pulling in the world.
- **Portable Rust is the source of truth.** SIMD and ASM paths are accelerators; the portable backend remains the reference impl.
- **Hardware dispatch is built in.** x86/x86_64, Arm/AArch64, Apple Silicon, IBM Z, IBM POWER, RISC-V, and WASM all have portable fallbacks, with optimized kernels where they pay.
- **`no_std` is a first-class target.** Server, CLI, embedded, bare-metal, and WASM builds use the same crate and feature model.
- **Audit knobs are explicit.** `portable-only` collapses runtime capability detection to the portable backend; `getrandom`, `serde`, and `rayon` are opt-in.
- **Security hygiene is part of the API.** Opaque verification errors, constant-time equality, zeroized secret types, strict arithmetic, official vectors, fuzzing, Miri, and cross-CPU CI are built into the project discipline.

`rscrypto` is a primitives crate. It is **not** a TLS stack, PKI toolkit, protocol implementation, or FIPS 140-3 validated module. No third-party security audit, FIPS 140-3 validation, or formal proof is claimed today.

## Install

Minimal `no_std` SHA-2 build:

```toml
[dependencies]
rscrypto = { version = "0.5.0", default-features = false, features = ["sha2"] }
```

Full primitive stack with OS randomness enabled:

```toml
[dependencies]
rscrypto = { version = "0.5.0", features = ["full", "getrandom"] }
```

Use `default-features = false` for constrained `no_std` builds. Enable `getrandom` only when you need APIs that generate salts, keys, nonces, or RSA key-gen entropy from the operating system.

## Quick Start

```rust
use rscrypto::{Digest, Sha256};

let one_shot = Sha256::digest(b"hello world");

let mut h = Sha256::new();
h.update(b"hello ");
h.update(b"world");

assert_eq!(h.finalize(), one_shot);
```

The common API shape is one-shot when convenient and streaming when needed.

## Verify RSA Signatures

```toml
[dependencies]
rscrypto = { version = "0.5.0", default-features = false, features = ["rsa"] }
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

Default RSA imports accept modern verification keys (RSA-3072 through RSA-8192,
exponent `65537`). RSA-2048 compatibility imports must opt in with
`RsaPublicKeyPolicy::legacy_verification()` and the `*_with_policy` parser.

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

Enable `getrandom` for RSA key gen, signing salt/blinding, encryption randomness, and private-op blinding. no-std RSA encryption callers can use the `*_with_random_fill` methods with a platform RNG. RSA key generation uses `getrandom` to seed its key-generation HMAC_DRBG, then follows the crate's FIPS 186-5 Appendix A.1.3 probable-prime generation contract:

```toml
[dependencies]
rscrypto = { version = "0.5.0", default-features = false, features = ["rsa", "getrandom"] }
```

## Sign ECDSA Messages

```toml
[dependencies]
rscrypto = { version = "0.5.0", default-features = false, features = ["ecdsa-p256"] }
```

```rust
use rscrypto::{EcdsaP256PublicKey, EcdsaP256SecretKey};

fn sign_and_verify(secret_bytes: [u8; 32], public_sec1: &[u8], message: &[u8]) -> bool {
  let Ok(secret) = EcdsaP256SecretKey::from_bytes(secret_bytes) else {
    return false;
  };
  let Ok(public) = EcdsaP256PublicKey::from_sec1_bytes(public_sec1) else {
    return false;
  };

  let Ok(signature) = secret.try_sign(message) else {
    return false;
  };

  public.verify(message, &signature).is_ok()
}
```

For P-384, enable `ecdsa-p384` and use `EcdsaP384SecretKey`,
`EcdsaP384PublicKey`, and `EcdsaP384Signature`. ECDSA supports fixed
P-256/SHA-256 and P-384/SHA-384 profiles, raw `r || s` signatures, DER
signature import, SEC1/SPKI public-key import, deterministic signing, and
caller-blinded signing APIs.

## Encrypt Data

```toml
[dependencies]
rscrypto = { version = "0.5.0", default-features = false, features = ["chacha20poly1305", "getrandom"] }
```

```rust
use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key};

let key = ChaCha20Poly1305Key::from_bytes([0x11; 32]);
let cipher = ChaCha20Poly1305::new(&key);

let aad = b"transfer:v1";
let mut sealed = [0u8; 10 + ChaCha20Poly1305::TAG_SIZE];
let nonce = cipher
  .seal_random(aad, b"pay bob 10", &mut sealed)
  .expect("encryption succeeds");

let mut message = [0u8; 10];
cipher
  .decrypt(&nonce, aad, &sealed, &mut message)
  .expect("authentication succeeds");

assert_eq!(&message, b"pay bob 10");
```

For high-volume AES-GCM streams, use `aead::NonceCounter` instead of random
96-bit nonces. It issues a monotonic nonce per seal and refuses to run past the
deterministic invocation budget.

## Hash Passwords

```toml
[dependencies]
rscrypto = { version = "0.5.0", default-features = false, features = ["argon2", "phc-strings", "getrandom"] }
```

```rust
use rscrypto::{Argon2Params, Argon2id};

let password = b"correct horse battery staple";
let params = Argon2Params::new().build().expect("valid Argon2 params");
let encoded = Argon2id::hash_string(&params, password).expect("password hash created");

assert!(
  Argon2id::verify_string(password, &encoded).is_ok()
);
```

## What You Get

| Need | Included | Feature path |
|---|---|---|
| Cryptographic Hashes | SHA-2, SHA-3, SHAKE, cSHAKE256, BLAKE2, BLAKE3, Ascon-Hash/XOF/CXOF | `hashes` or leaf features |
| MACs and KDFs | HMAC-SHA-2, KMAC256, HKDF-SHA-2, PBKDF2-HMAC-SHA-2 | `auth` or leaf features |
| Password Hashing | Argon2d/i/id, scrypt, PHC string encode/verify | `auth`, `argon2`, `scrypt`, `phc-strings` |
| Public-key Primitives | ECDSA P-256/P-384 signing/verification, Ed25519 signatures, RSA signing/verification/OAEP/RSAES-PKCS1-v1_5/key generation, X25519 key exchange | `auth`, `signatures`, `ecdsa`, `ecdsa-p256`, `ecdsa-p384`, `ed25519`, `rsa`, `x25519` |
| AEAD Encryption | AES-128/256-GCM, AES-128/256-GCM-SIV, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, Ascon-AEAD128 | `aead` or leaf features |
| Checksums | CRC-16, CRC-24, CRC-32, CRC-32C, CRC-64/XZ, CRC-64/NVMe | `checksums` or leaf features |
| Fast Non-crypto Hashes | XXH3-64/128, RapidHash 64/128 | `xxh3`, `rapidhash` |

Fast non-cryptographic hashes and CRCs are for indexing, checksumming, dedup, and integrity plumbing. Do not use them for passwords, signatures, MACs, key derivation, or authentication.

Flags are layered by use:

- **Leaf Primitives:** `sha2`, `blake3`, `aes-gcm`, `ed25519`, `x25519`, `crc32`, etc.
- **Families/Groups:** `hashes`, `checksums`, `macs`, `kdfs`, `password-hashing`, `aead`, `signatures`, `key-exchange`.
- **Deployment Controls:** `std`, `alloc`, `getrandom`, `parallel`, `serde`, `serde-secrets`, `portable-only`.

Full Feature Inventory: [`docs/features.md`](docs/features.md).
Public Type Inventory: [`docs/types.md`](docs/types.md).

## Constant-Time Boundaries

`rscrypto` makes scoped constant-time claims for secret-bearing operations, not
for every function in the crate.

The exact release claim is the set of primitive/configuration pairs marked
`ct_claimed` in [`ct.toml`](ct.toml). The main secret-bearing surfaces are
MAC/tag verification, AEAD authentication failure shape, X25519 scalar
multiplication, Ed25519 signing and secret public-key derivation, ECDSA
P-256/P-384 blinded signing, RSA private sign/decrypt leaves, and selected
password-verification comparisons.

Public parsing, key generation, OS randomness, raw hashes, checksums,
non-cryptographic hashes, benchmark paths, and public-key verification math are
not blanket constant-time claims. See [`docs/security.md`](docs/security.md)
for application guidance and [`docs/constant-time.md`](docs/constant-time.md)
for the exact claim model.

## Performance

Latest public bench evidence comes from a generated full Linux CI pass:

- Linux (CI): Nine Linux runners across Intel/ARM x86/x86_64, ARM/aarch64, IBM Power/ppc64le, IBM Z/s390x, and RISC-V.

Speedup is `external_crate_time / rscrypto_time`; values above `1.00x` mean `rscrypto` is faster.

| Area | Compared Against | Result |
|---|---|---:|
| **Linux CI fastest external** | strongest matched Rust baseline per case | **1.58x geomean** |
| Linux CI scorecard | fastest external | **4,102 wins / 6,750 pairs** |
| Linux CI wins or ties | fastest external | **6,056 / 6,750 pairs** |
| Linux CI all matched pairs | every external comparison row | **1.78x geomean; 9,887 / 10,781 wins-or-ties** |
| Checksums | Linux CI | **5.20x geomean** |
| Hashes, MACs, XOFs | Linux CI | **1.35x geomean** |
| Auth/KDF | Linux CI | **1.23x geomean** |
| Password hashing | Linux CI | **1.11x geomean** |
| Public-key | Linux CI, including ML-KEM | **1.12x geomean** |
| ML-KEM-512/768/1024 | Linux CI keygen/encapsulate/decapsulate | **0.78x geomean** |
| ECDSA P-256/P-384 | Linux CI | **1.40x geomean** |
| RSA import + verify | Linux CI | **1.54x geomean** |
| AEAD | Linux CI | **1.56x geomean** |

The measured weak spots in the latest published benchmark set are ML-KEM keygen,
encapsulation, and decapsulation rows, especially on IBM Z/s390x, Graviton, and
RISC-V, followed by ECDSA P-384 signing and Argon2id OWASP pressure. See
[`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md) for raw runs,
methodology, platform scorecards, and loss tables.

## Portability And Acceleration

`rscrypto` keeps the portable Rust path as the byte-for-byte authority. ISA kernels are selected only when the target and runtime CPU support them.

| Target family | Acceleration examples |
|---|---|
| x86 / x86_64 | SSE4.2, AVX2, AVX-512, AES-NI, SHA-NI, VAES, VPCLMULQDQ |
| Arm / AArch64 / Apple Silicon | NEON, AES, PMULL, SHA2, SHA3, SVE2-PMULL |
| IBM Z | CPACF, MSA, VGFM |
| POWER / ppc64le | POWER8/9/10 vector and crypto extensions |
| RISC-V | RVV, Zbc, Zvkned, Zvbc |
| WASM | SIMD128 where available, portable fallback everywhere |

Use `portable-only` when you need deterministic dispatch, audit-constrained builds, or a portable backend only.

Full platform matrix: [`docs/platforms.md`](docs/platforms.md).

## Security

- Scoped constant-time claims for secret-bearing operations; [`docs/security.md`](docs/security.md) names the boundary.
- Opaque verification errors that avoid leaking failure details.
- Secret-bearing types zeroize on drop and mask `Debug`.
- Strict arithmetic for counters, lengths, offsets, and indices.
- AEAD failed-open paths wipe output buffers.
- Portable and accelerated backends are differentially tested for byte-identical output.
- Official test vectors, Wycheproof coverage where applicable, fuzz corpus replay, and Miri run in CI.
- RSA private operations have extra regression coverage for memory safety and
  first-order timing leakage.
- No third-party audit or FIPS certificate is claimed today.

Start with [`docs/trust.md`](docs/trust.md) for the evidence map. Read
[`docs/security.md`](docs/security.md) before shipping cryptographic code. For
compliance posture, see [`docs/compliance.md`](docs/compliance.md).

Vulnerabilities should be reported through [GitHub Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new) or the process in [`SECURITY.md`](SECURITY.md).

Do not report real-world vulnerabilities through public GitHub issues.

## Docs

- API reference: [docs.rs/rscrypto](https://docs.rs/rscrypto)
- Examples: [`examples/`](examples/)
- Feature flags: [`docs/features.md`](docs/features.md)
- Public type inventory: [`docs/types.md`](docs/types.md)
- Trust profile: [`docs/trust.md`](docs/trust.md)
- Platform matrix: [`docs/platforms.md`](docs/platforms.md)
- Security guidance: [`docs/security.md`](docs/security.md)
- Test vector coverage: [`docs/test-vector-coverage.md`](docs/test-vector-coverage.md)
- Migration guides: [`docs/migration/`](docs/migration/)
- Benchmark methodology: [`docs/benchmarking.md`](docs/benchmarking.md)
- Benchmarks: [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md)
- Release history: [`CHANGELOG.md`](CHANGELOG.md)

## MSRV

Rust **1.91.0**.

The pinned nightly in [`rust-toolchain.toml`](rust-toolchain.toml) is used for Miri, fuzzing, and exotic-architecture checks.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
