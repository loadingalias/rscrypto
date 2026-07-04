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

<p align="center">
  <img alt="rscrypto benchmark chart: 1.59x Linux and 1.37x Apple Silicon fastest-matched geomeans, checksums at 5.18x against crc-fast, crc, crc32fast, crc32c, and crc64fast, plus primitive geomean bars and M1 MBP Apple Silicon notes."
       src="assets/readme/perf.svg"
       width="640">
</p>

<p align="center">
  <i>Chart: benchmark scorecard. Values above <code>1.00x</code> mean <code>rscrypto</code> is faster than the fastest matched external implementation.</i>
</p>

## Why rscrypto?

- **Pure Rust Primitives:** Hashes, MACs, KDFs, password hashing, AEADs, signatures, key exchange, ML-KEM, RSA, and checksums live behind one crate, one feature model, and no OpenSSL or C/FFI dependency.
- **Build Only What You Need:** Enable a leaf such as `sha2`, `aes-gcm`, `ed25519`, `x25519`, `ml-kem`, `rsa`, or `argon2`, use a family feature such as `aead` or `signatures`, or choose `full` for the complete surface.
- **Consistent API Shape:** Supported primitives use concrete types, scoped errors, opaque verification failures, and the same verification convention instead of making every algorithm feel like a separate crate.
- **First-Class Public-Key Support:** RSA includes strict DER import/export, PSS, PKCS#1 v1.5 signatures, OAEP, RSAES-PKCS1-v1_5, FIPS 186-5 A.1.3 probable-prime key generation, profile mapping, private-operation blinding, and reusable scratch APIs.
- **Typed & Tested Post-Quantum KEMs:** `ml-kem` exposes ML-KEM-512/768/1024 key, ciphertext, and shared-secret types with prepared-key paths, ACVP vectors, `fips203` differential tests, scoped constant-time claims, and selected architecture kernels.
- **Portable Rust Authority** SIMD and ASM backends are accelerators. The portable implementation remains the reference path and is differentially tested against accelerated kernels.
- **Server, CLI, Embedded, Bare-Metal, and WASM.** `no_std` builds use the same feature model, with portable fallbacks across x86/x86_64, Arm/AArch64, Apple Silicon, IBM Z, IBM POWER, RISC-V, and WASM.
- **Explicit Deployment Controls** `portable-only` forces runtime dispatch toward portable backends. The only optional external dependencies are `getrandom`, `serde`, and `rayon`; `rayon` is reached through the `parallel` feature.
- **Deep Validation Coverage** Supported surfaces are backed by official vectors, Wycheproof where it maps to the API, ACVP ML-KEM vectors, differential tests against established crates, fuzz corpus replay, Miri, portable-vs-accelerated equivalence tests, cross-CPU CI, and RSA-specific leakage and memory-safety regression coverage.
- **Constant-Time Validation Release Gate** Secret-bearing paths are tracked in `ct.toml`; claimed releases require the matching CT evidence, including harness coverage, build provenance, LLVM IR/assembly/object artifacts, generated-code heuristics, native DudeCT timing tests, binary checks where supported via BINSEC, and Miri/unsafe validation for relevant paths.

`rscrypto` is a primitives crate. It is not a TLS stack, PKI toolkit, key store, or protocol implementation. It is not a FIPS 140-3 validated module, third-party audited, formally verified, or a whole-crate constant-time claim today.

## Install

Minimal `no_std` SHA-2 build:

```toml
[dependencies]
rscrypto = { version = "0.6.0", default-features = false, features = ["sha2"] }
```

Full primitive stack with OS randomness enabled:

```toml
[dependencies]
rscrypto = { version = "0.6.0", features = ["full", "getrandom"] }
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

## Verify RSA Signatures

```toml
[dependencies]
rscrypto = { version = "0.6.0", default-features = false, features = ["rsa"] }
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
exponent `65537`). RSA-2048 compatibility imports MUST opt in with
`RsaPublicKeyPolicy::legacy_verification()` and the `*_with_policy` parser.

For repeated verification with the same key, reuse `key.public_scratch()` with
the `*_with_scratch` APIs instead of allocating per call.

Add `getrandom` when `rscrypto` should generate RSA key material, PSS salt,
encryption randomness, or private-operation blinding randomness. `no_std`
encryption callers can use the `*_with_random_fill` methods with a platform
RNG. RSA key generation seeds its HMAC_DRBG from `getrandom` and follows the
crate's FIPS 186-5 Appendix A.1.3 probable-prime generation contract:

```toml
[dependencies]
rscrypto = { version = "0.6.0", default-features = false, features = ["rsa", "getrandom"] }
```

## Sign ECDSA Messages

```toml
[dependencies]
rscrypto = { version = "0.6.0", default-features = false, features = ["ecdsa-p256"] }
```

```rust
use rscrypto::EcdsaP256SecretKey;

fn sign_and_verify(secret_bytes: [u8; 32], message: &[u8]) -> bool {
  let Ok(secret) = EcdsaP256SecretKey::from_bytes(secret_bytes) else {
    return false;
  };
  let public = secret.public_key();

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

## Establish An ML-KEM Shared Secret

```toml
[dependencies]
rscrypto = { version = "0.6.0", default-features = false, features = ["ml-kem"] }
```

```rust
use rscrypto::{Kem, MlKem768, MlKemError};

fn deterministic_fill(seed: u8) -> impl FnMut(&mut [u8]) -> Result<(), MlKemError> {
  move |out| {
    for (i, b) in out.iter_mut().enumerate() {
      *b = seed.wrapping_add(i as u8);
    }
    Ok(())
  }
}

fn round_trip_mlkem768() -> Result<(), MlKemError> {
  let (encapsulation_key, decapsulation_key) =
    MlKem768::generate_keypair(deterministic_fill(0x40))?;
  let (ciphertext, sender_secret) =
    MlKem768::encapsulate(&encapsulation_key, deterministic_fill(0x90))?;
  let receiver_secret = MlKem768::decapsulate(&decapsulation_key, &ciphertext)?;

  assert_eq!(sender_secret, receiver_secret);
  Ok(())
}

assert!(round_trip_mlkem768().is_ok());
```

Production callers should use a real entropy source for key generation and
encapsulation randomness. The API accepts caller-supplied random-fill closures,
so `ml-kem` works in `no_std` deployments that own their entropy boundary.

## Encrypt Data

```toml
[dependencies]
rscrypto = { version = "0.6.0", default-features = false, features = ["chacha20poly1305", "getrandom"] }
```

```rust
use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key};

fn encrypt_round_trip() -> bool {
  let key = ChaCha20Poly1305Key::from_bytes([0x11; 32]);
  let cipher = ChaCha20Poly1305::new(&key);

  let aad = b"transfer:v1";
  let mut sealed = [0u8; 10 + ChaCha20Poly1305::TAG_SIZE];
  let Ok(nonce) = cipher.seal_random(aad, b"pay bob 10", &mut sealed) else {
    return false;
  };

  let mut message = [0u8; 10];
  let Ok(()) = cipher.decrypt(&nonce, aad, &sealed, &mut message) else {
    return false;
  };

  &message == b"pay bob 10"
}

assert!(encrypt_round_trip());
```

For high-volume AES-GCM streams, use `aead::NonceCounter` instead of random
96-bit nonces. It issues a monotonic nonce per seal and refuses to run past the
deterministic invocation budget.

## Hash Passwords

```toml
[dependencies]
rscrypto = { version = "0.6.0", default-features = false, features = ["argon2", "phc-strings", "getrandom"] }
```

```rust
use rscrypto::{Argon2Params, Argon2id};

fn hash_password_round_trip() -> bool {
  let password = b"correct horse battery staple";
  let params = Argon2Params::default();
  let Ok(encoded) = Argon2id::hash_string(&params, password) else {
    return false;
  };

  Argon2id::verify_string(password, &encoded).is_ok()
}

assert!(hash_password_round_trip());
```

Tune `Argon2Params` for your latency and memory budget before shipping a
password store.

## What You Get

| Need | Included | Feature Path |
|---|---|---|
| Cryptographic Hashes | SHA-2, SHA-3, SHAKE, cSHAKE256, BLAKE2, BLAKE3, Ascon-Hash/XOF/CXOF | `hashes` or leaf features |
| MACs & KDFs | HMAC-SHA-2, KMAC256, HKDF-SHA-2, PBKDF2-HMAC-SHA-2 | `auth` or leaf features |
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

`rscrypto` makes scoped constant-time claims for secret-bearing operations, not
for every single function in the crate.

The exact release gate is the set of primitive/config pairs marked
`ct_claimed` in [`ct.toml`](ct.toml). The main secret-bearing surfaces are
MAC/tag verification, AEAD authentication failure shape, X25519 scalar
multiplication, Ed25519 signing and secret public-key derivation, ECDSA
P-256/P-384 blinded signing, ML-KEM-512/768/1024 key gen,
encapsulation, decapsulation secret surfaces, RSA private sign/decrypt leaves,
and selected password-verification comparisons.

Public parsing, unlisted key gen, OS randomness, raw hashes, checksums,
non-cryptographic hashes, benchmark paths, and public-key verification math are
not blanket constant-time claims. See [`docs/constant-time.md`](docs/constant-time.md)
for the exact claim model and [`docs/compliance.md`](docs/compliance.md) for
review boundaries.

## Performance

Linux: Nine Linux runners across Intel/ARM x86/x86_64, ARM/aarch64, IBM Power/ppc64le, IBM Z/s390x, and RISC-V are used to benchmark `rscrypto` performance.

Speedup is `external_crate_time / rscrypto_time`; values above `1.00x` mean `rscrypto` is faster.

| Area | Compared Against | Result |
|---|---|---:|
| **Linux vs Fastest External** | strongest known Rust competitor per case (usually aws-lc-rs) | **1.59x Geomean** |
| Linux Scorecard | Fastest External | **4,052 wins / 6,750 Pairs** |
| Linux Wins or Ties | Fastest External | **6,101 / 6,750 Pairs** |
| Linux All Matched Pairs | Every Comparison Row | **1.76x Geomean; 10,012 / 10,781 Wins or Ties** |
| macOS Apple Silicon vs Fastest External | Local Apple Silicon Run | **1.37x Geomean; 708 / 774 Wins or Ties** |
| macOS Apple Silicon All Matched Pairs | Every Comparison Row | **1.66x Geomean; 1,219 / 1,297 Wins or Ties** |
| macOS Apple Silicon ML-KEM | Fastest External | **1.35x Geomean; 7 / 9 Wins or Ties** |
| Checksums | Linux | **5.18x Geomean** |
| Hashes, MACs, XOFs | Linux | **1.35x Geomean** |
| Auth/KDF | Linux | **1.25x Geomean** |
| Password hashing | Linux | **1.07x Geomean** |
| Public-key | Linux, including ML-KEM | **1.33x Geomean** |
| ML-KEM-512/768/1024 | Linux keygen/encapsulate/decapsulate | **1.49x Geomean** |
| ECDSA P-256/P-384 | Linux | **1.45x Geomean** |
| RSA import + verify | Linux | **1.55x Geomean** |
| AEAD | Linux | **1.56x Geomean** |

Use [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md) for raw runs,
methodology, platform-specific scorecards, and loss tables.

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

- Scoped constant-time claims for secret-bearing operations; see [`docs/constant-time.md`](docs/constant-time.md) for the details.
- Opaque verification errors that avoid leaking failure details.
- Secret-bearing types zeroize on drop and mask `Debug`.
- Strict arithmetic for counters, lengths, offsets, and indices.
- AEAD failed-open paths wipe output buffers.
- ML-KEM-512/768/1024 have FIPS 203 ACVP vectors, `fips203` differential coverage, and CT evidence for declared secret-bearing operations.
- Portable and accelerated backends are differentially tested for byte-identical output.
- Official test vectors, Wycheproof coverage where applicable, fuzz corpus replay, and Miri run in CI.
- RSA private operations have extra regression coverage for memory safety and first-order timing leakage.
- No third-party audit or FIPS certificate as of now.

Vulnerabilities should be reported through [GitHub Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new) or the process in [`SECURITY.md`](SECURITY.md).

Please, do not report real-world vulnerabilities through public GitHub issues.

## Docs

- API reference: [docs.rs/rscrypto](https://docs.rs/rscrypto)
- Examples: [`examples/`](examples/)
- Feature flags: [`docs/features.md`](docs/features.md)
- Public type inventory: [`docs/types.md`](docs/types.md)
- Constant-time policy: [`docs/constant-time.md`](docs/constant-time.md)
- Compliance posture: [`docs/compliance.md`](docs/compliance.md)
- Platform matrix: [`docs/platforms.md`](docs/platforms.md)
- Test vector coverage: [`docs/test-vector-coverage.md`](docs/test-vector-coverage.md)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Migration guides: [`docs/migration/`](docs/migration/)
- Benchmark methodology: [`docs/benchmarking.md`](docs/benchmarking.md)
- Benchmarks: [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md)
- Release history: [`CHANGELOG.md`](CHANGELOG.md)

## MSRV

Rust **1.91.0**.

The pinned nightly in [`rust-toolchain.toml`](rust-toolchain.toml) is used for Miri, fuzzing, and exotic-architecture checks.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
