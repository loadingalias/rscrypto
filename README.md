# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![Docs.rs](https://docs.rs/rscrypto/badge.svg)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml)
[![RSA Gates](https://github.com/loadingalias/rscrypto/actions/workflows/rsa.yaml/badge.svg?branch=main)](https://github.com/loadingalias/rscrypto/actions/workflows/rsa.yaml)
[![MSRV 1.91.0](https://img.shields.io/badge/MSRV-1.91.0-blue)](Cargo.toml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/rscrypto)](#license)

**Pure Rust cryptography: RSA, Ed25519, X25519, AEADs, hashes, KDFs, password hashing, CRCs, `no_std`, WASM, and hardware acceleration in one dependency.**

`rscrypto` is a single primitive stack for projects that care about binary size, deployment control, and speed w/o dragging in mandatory C, OpenSSL, or system lib story.

Use one leaf feature for one primitive, a group for a subset of primitives, or `full` for the whole shebang. The portable Rust backend is always present. SIMD and ASM are only accelerators.

**Current benchmark evidence:** Linux is currently `1.60x` fastest-external geomean w/ `3,627 / 5,994` wins and `5,345 / 5,994` wins-or-ties. Apple Silicon (MBP M1 local full run) is `1.39x` fastest-external geomean w/ `368 / 702` wins and `643 / 702` wins-or-ties.

<p align="center">
  <img alt="rscrypto benchmark scorecard: 1.60x fastest-external geomean across Linux CI with 3,627 wins and 5,345 wins-or-ties out of 5,994 matched benchmark comparisons."
       src="assets/readme/perf.svg"
       width="640">
</p>

<p align="center">
  <i>Chart: 2026-06-08 Linux CI benchmark pass. Apple Silicon numbers from the 2026-06-08 MBP M1 local full run are listed below. Values above <code>1.00x</code> mean <code>rscrypto</code> is faster than the fastest matched Rust baseline.</i>
</p>

## Why rscrypto?

- **RSA is a first class citizen.** Strict DER import/export, RSA-PSS, RSASSA-PKCS1-v1_5, OAEP, RSAES-PKCS1-v1_5, FIPS 186-5 A.1.3 probable-prime key generation in code, X.509/JWT/COSE/TLS profile mapping, blinded private operations, and reusable scratch APIs.
- **One coherent primitive stack.** Avoid composing a dozen crates with different APIs, feature models, and security conventions.
- **Small builds stay small.** Enable `sha2`, `blake3`, `aes-gcm`, `chacha20poly1305`, `ed25519`, `x25519`, `argon2`, or any other leaf without pulling in the world.
- **Portable Rust is the source of truth.** SIMD and ASM paths are accelerators; the portable backend remains the reference impl.
- **Hardware dispatch is built in.** x86/x86_64, Arm/AArch64, Apple Silicon, IBM Z, IBM POWER, RISC-V, and WASM all have portable fallbacks, w/ optimized kernels where they pay.
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

Full primitive stack w/ OS randomness enabled:

```toml
[dependencies]
rscrypto = { version = "0.3.1", features = ["full", "getrandom"] }
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

The common API shape is deliberately simple: one-shot when convenient, streaming when it's needed.

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

Enable `getrandom` for RSA key gen, signing salt/blinding, OAEP encryption randomness, and private-op blinding. RSA key generation uses `getrandom` to seed an internal HMAC_DRBG, then follows the crate's FIPS 186-5 Appendix A.1.3 probable-prime generation contract:

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
| Cryptographic Hashes | SHA-2, SHA-3, SHAKE, cSHAKE256, BLAKE2, BLAKE3, Ascon-Hash/XOF/CXOF | `hashes` or leaf features |
| MACs and KDFs | HMAC-SHA-2, KMAC256, HKDF-SHA-2, PBKDF2-HMAC-SHA-2 | `auth` or leaf features |
| Password Hashing | Argon2d/i/id, scrypt, PHC string encode/verify | `auth`, `argon2`, `scrypt`, `phc-strings` |
| Public-key Primitives | Ed25519 signatures, RSA signing/verification/OAEP/RSAES-PKCS1-v1_5/key generation, X25519 key exchange | `auth`, `signatures`, `ed25519`, `rsa`, `x25519` |
| AEAD Encryption | AES-128/256-GCM, AES-128/256-GCM-SIV, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, Ascon-AEAD128 | `aead` or leaf features |
| Checksums | CRC-16, CRC-24, CRC-32, CRC-32C, CRC-64/XZ, CRC-64/NVMe | `checksums` or leaf features |
| Fast Non-crypto Hashes | XXH3-64/128, RapidHash 64/128 | `xxh3`, `rapidhash` |

Fast non-cryptographic hashes and CRCs are for indexing, checksumming, dedup, and integrity plumbing. Do not use them for passwords, signatures, MACs, key derivation, or authentication... it's obviously not safe.

Flags are layered deliberately:

- **Leaf Primitives:** `sha2`, `blake3`, `aes-gcm`, `ed25519`, `x25519`, `crc32`, etc.
- **Families/Groups:** `hashes`, `checksums`, `macs`, `kdfs`, `password-hashing`, `aead`, `signatures`, `key-exchange`.
- **Deployment Controls:** `std`, `alloc`, `getrandom`, `parallel`, `serde`, `serde-secrets`, `portable-only`.

Full Feature Inventory: [`docs/features.md`](docs/features.md).
Public Type Inventory: [`docs/types.md`](docs/types.md).

## Constant-Time Verification

`rscrypto` treats constant-time as a release evidence claim, not a style claim.
For a release commit, [`ct.yaml`](.github/workflows/ct.yaml) must finish green
with the complete artifact set below, and [`ct.toml`](ct.toml) is the
machine-readable source of truth for which primitives, kernels, targets, and
gates are required. The policy and invalidation rules live in
[`docs/constant-time.md`](docs/constant-time.md).

A green CT release run uploads compact evidence for the physical runner lanes
that mirror the public benchmark matrix:

| Physical lane | Target evidence | Artifact |
|---|---|---|
| AMD Zen4, AMD Zen5, Intel Ice Lake, Intel Sapphire Rapids | Linux `x86_64` artifact/provenance review, LLVM IR/ASM/object heuristics, DudeCT, and BINSEC | `ct-amd-zen4`, `ct-amd-zen5`, `ct-intel-icl`, `ct-intel-spr` |
| AWS Graviton3 and Graviton4 | Linux `aarch64` artifact/provenance review, LLVM IR/ASM/object heuristics, DudeCT, and BINSEC | `ct-graviton3`, `ct-graviton4` |
| RISE RISC-V | Linux `riscv64gc` artifact/provenance review, LLVM IR/ASM/object heuristics, and DudeCT. BINSEC is not claimed for RISC-V today because the current BINSEC/RISC-V workflow does not complete the release-sized HMAC/HKDF/KMAC/PBKDF2/RSA leaf proofs within the CI proof budget. | `ct-rise-riscv` |
| IBM z16 / s390x | Linux `s390x` artifact/provenance review, LLVM IR/ASM/object heuristics, and DudeCT. BINSEC is not claimed for s390x today. | `ct-ibm-s390x` |
| IBM Power10 / ppc64le | Linux `powerpc64le` artifact/provenance review, LLVM IR/ASM/object heuristics, and DudeCT. BINSEC is not claimed for little-endian POWER today. | `ct-ibm-power10` |
| Apple Silicon | Local macOS `aarch64` artifact/provenance review, LLVM IR/ASM/object heuristics, and DudeCT through `just ct-full`. BINSEC is not claimed for Mach-O today. | local `ct-evidence/` package |

Each CT artifact contains a short `README.md`, `ct-report-<lane>.md`,
`ct-report-<lane>.json`, host provenance, the full CT log, and any failed or
inconclusive component reports. For public release notes, pin the exact green
run URL and commit next to these artifact names.

The CT tooling is checked in under [`tools/`](tools/) and [`scripts/ct/`](scripts/ct/):
stable harness entrypoints, the DudeCT runner, the BINSEC harness generator,
manifest validation, full-run orchestration, and artifact packaging.

Secret-bearing primitive coverage is deliberately explicit:

| Surface | Required CT evidence |
|---|---|
| Equality and verification leaves | Constant-time equality, secret-byte equality, HMAC-SHA-2 verification, KMAC256 verification, keyed BLAKE2/BLAKE3 leaves, HKDF output derivation, and PBKDF2 verification leaves. |
| AEAD authentication and symmetric transforms | AES-128/256-GCM, AES-128/256-GCM-SIV, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, and Ascon-AEAD128 required leaves, including AES rounds, GHASH, POLYVAL, Poly1305, tag generation, and open/authentication failure shape. |
| Public-key secret operations | X25519 scalar multiplication, Ed25519 signing and secret public-key derivation, RSA private signing/decryption leaves, RSA private-operation window selection, and bounded private-component validation leaves. |
| Password hashing | Argon2i secret-bearing hash leaves and final verification comparisons are CT-gated. Argon2d, Argon2id, and scrypt are classified as best-effort for local side-channel CT because their algorithms use data-dependent memory access; their final comparisons and parser/failure boundaries still run through the security test/fuzz evidence. |
| Public-only work | Raw hashes, checksums, non-cryptographic hashes, public-key verification math, DER/PHC parsing, serialization, key generation, OS randomness, and benchmark-only paths are not blanket constant-time claims unless `ct.toml` promotes a specific leaf. |

RSA has a second dedicated gate because private-key code deserves extra scrutiny:
[`rsa.yaml`](.github/workflows/rsa.yaml) uploads `rsa-miri-linux-x64`,
`rsa-leakage-linux-x64`, and `rsa-leakage-linux-arm64` artifacts for release
review.

## Performance

Current public benchmark evidence comes from two passes that are both updated regularly and programmatically:

- Linux (CI): Nine Linux runners across Intel/ARM x86/x86_64, ARM/aarch64, IBM Power/ppc64le, IBM Z/s390x, and RISC-V.
- Apple Silicon: My local MBP M1, Apple Silicon aarch64.

Speedup is `external_crate_time / rscrypto_time`; values above `1.00x` mean `rscrypto` is faster.

| Area | Compared Against | Result |
|---|---|---:|
| **Linux CI fastest external** | strongest matched Rust baseline per case | **1.60x geomean** |
| Linux CI scorecard | fastest external | **3,627 wins / 5,994 pairs** |
| Linux CI wins or ties | fastest external | **5,345 / 5,994 pairs** |
| **Apple Silicon fastest external** | strongest matched Rust baseline per case | **1.39x geomean** |
| Apple Silicon scorecard | fastest external | **368 wins / 702 pairs** |
| Apple Silicon wins or ties | fastest external | **643 / 702 pairs** |
| Linux CI all matched pairs | every external comparison row | **1.75x geomean; 8,708 / 9,544 wins-or-ties** |
| Checksums | Linux CI / Apple Silicon | **2.62x / 2.85x geomean** |
| Hashes, MACs, XOFs | Linux CI / Apple Silicon | **1.40x / 1.07x geomean** |
| Auth/KDF | Linux CI / Apple Silicon | **1.17x / 1.01x geomean** |
| Password hashing | Linux CI / Apple Silicon | **0.97x / 1.07x geomean** |
| Public-key | Linux CI / Apple Silicon | **0.99x / 1.02x geomean** |
| RSA import + verify | Linux CI / Apple Silicon | **1.30x / 1.45x geomean** |
| AEAD | Linux CI / Apple Silicon | **1.56x / 1.44x geomean** |

The honest weak spots right now: Linux public-key/RSA pressure, X25519 and
Ed25519 derivation cases, password/KDF front-end overhead, Apple Silicon
ChaCha20-Poly1305, Apple Silicon SHA/XXH3 holes, and PBKDF2-SHA256 at
`iters=1`. See [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md)
for raw runs, methodology, platform scorecards, and loss tables.

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

Full platform matrix: [`docs/platforms.md`](docs/platforms.md). Architecture notes: [`docs/architecture.md`](docs/architecture.md).

## Security

- Scoped constant-time verification and secret-bearing operations; [`docs/security.md`](docs/security.md) names the boundary.
- Opaque verification errors that avoid leaking failure details.
- Secret-bearing types zeroize on drop and mask `Debug`.
- Strict arithmetic for counters, lengths, offsets, and indices.
- AEAD failed-open paths wipe output buffers.
- Portable and accelerated backends are differentially tested for byte-identical output.
- Official test vectors, Wycheproof coverage where applicable, fuzz corpus replay, and Miri run in CI.
- RSA private-operation release claims require the dedicated RSA Miri and first-order leakage gates.

Read [`docs/security.md`](docs/security.md) before shipping cryptographic code. For compliance posture, see [`docs/compliance.md`](docs/compliance.md).

Vulnerabilities should be reported through [GitHub Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new) or the process in [`SECURITY.md`](SECURITY.md).

## Docs

- API reference: [docs.rs/rscrypto](https://docs.rs/rscrypto)
- Examples: [`examples/`](examples/)
- Feature flags: [`docs/features.md`](docs/features.md)
- Public type inventory: [`docs/types.md`](docs/types.md)
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
