# rscrypto

> Zero-dependency Rust crypto, hardware-accelerated across x86_64, ARM64, RISC-V, IBM Z, IBM POWER, and Apple Silicon.

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![docs.rs](https://img.shields.io/docsrs/rscrypto)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/loadingalias/rscrypto/graph/badge.svg)](https://codecov.io/gh/loadingalias/rscrypto)

Most Rust crypto stacks force a bad choice: wire together a pile of single-primitive crates, or accept a larger package w/ dependencies, external C-libs, and supply-chain tradeoffs.
That risk is not one I'm willing to take with the state of supply-chain attacks.

`rscrypto` is a single-crate crypto stack: checksums, cryptographic hashes, fast hashes, MACs, KDFs, password hashing, signatures, key exchange, and AEADs behind leaf-selectable features.
It can be a tiny SHA-2 dependency or the full primitives package; either way, primitive implementations stay in-tree as Rust, SIMD intrinsics, and targeted ASM kernels.
No C FFI, no vendored C/C++, no OpenSSL/libcrypto dependency, no external crypto crate stack.

Hardware acceleration is not just a single platform. I've taken great care to accelerate widely.
The benchmark matrix covers x86_64 Intel/AMD, ARM64 Graviton3/4, RISE RISC-V, IBM Z/s390x, IBM POWER10, and macOS Apple Silicon.
Every benchmarked architecture has hardware, SIMD, or ASM acceleration in-tree; portable fallbacks remain the correctness floor for unsupported configurations and constrained builds.

Proof points:

| Claim | Evidence |
|-------|----------|
| Faster than the official BLAKE3 crate on large buffers | **2.37x** geomean faster than `blake3` for one-shot/keyed/derive-key `>=64 KiB` Linux CI inputs; up to **7.70x** |
| Faster Ed25519 signing without `ed25519-dalek` / `curve25519-dalek` | **1.57x** geomean faster than `ed25519-dalek` signing across Linux CI |
| Broad hardware-accelerated portability | x86_64, ARM64, RISC-V, IBM Z, IBM POWER, Apple Silicon, plus portable `no_std`/WASM fallbacks |
| Zero default dependency surface | `full` enables the primitive stack without pulling OpenSSL, C FFI, RustCrypto, `dalek`, `blake3`, or `crc*` crates |

Use it as a single primitive:

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["sha2"] }
```

Or bring the whole toolbox without pulling in a C library:

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["full"] }
```

## Quick Start

```rust
use rscrypto::{
  Aead, Blake3, Checksum, ChaCha20Poly1305, ChaCha20Poly1305Key, Crc32C,
  Digest, Ed25519Keypair, Ed25519SecretKey, HkdfSha256, HmacSha256, Kmac256,
  Mac, Sha256, Shake256, X25519SecretKey, Xof, Xxh3, aead::Nonce96,
};

let crc = Crc32C::checksum(b"data");

let hash = Sha256::digest(b"data");
let mut h = Sha256::new();
h.update(b"da"); h.update(b"ta");
assert_eq!(h.finalize(), hash);

let tag = HmacSha256::mac(b"key", b"data");
assert!(HmacSha256::verify_tag(b"key", b"data", &tag).is_ok());

let mut okm = [0u8; 32];
HkdfSha256::new(b"salt", b"ikm").expand(b"info", &mut okm)?;

let mut xof = Shake256::xof(b"data");
let mut out = [0u8; 64];
xof.squeeze(&mut out);

let kp = Ed25519Keypair::from_secret_key(Ed25519SecretKey::from_bytes([7u8; 32]));
let sig = kp.sign(b"data");
assert!(kp.public_key().verify(b"data", &sig).is_ok());

let alice = X25519SecretKey::from_bytes([7u8; 32]);
let bob = X25519SecretKey::from_bytes([9u8; 32]);
assert_eq!(alice.diffie_hellman(&bob.public_key())?, bob.diffie_hellman(&alice.public_key())?);

let cipher = ChaCha20Poly1305::new(&ChaCha20Poly1305Key::from_bytes([0x11; 32]));
let nonce = Nonce96::from_bytes([0x22; 12]);
let mut buf = *b"data";
let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf)?;
cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &tag)?;
assert_eq!(&buf, b"data");

let _ = Xxh3::hash(b"data");
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Password Hashing Quick Start

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["password-hashing", "getrandom"] }
```

```rust
use rscrypto::{Argon2Params, Argon2VerifyPolicy, Argon2id, Scrypt, ScryptParams, ScryptVerifyPolicy};

let password = b"correct horse battery staple";

let argon2 = Argon2Params::new().build()?;
let encoded = Argon2id::hash_string(&argon2, password)?;
assert!(Argon2id::verify_string_with_policy(password, &encoded, &Argon2VerifyPolicy::default()).is_ok());
assert!(Argon2id::verify_string_with_policy(b"wrong", &encoded, &Argon2VerifyPolicy::default()).is_err());

let scrypt = ScryptParams::new().build()?;
let encoded = Scrypt::hash_string(&scrypt, password)?;
assert!(Scrypt::verify_string_with_policy(password, &encoded, &ScryptVerifyPolicy::default()).is_ok());
assert!(Scrypt::verify_string_with_policy(b"wrong", &encoded, &ScryptVerifyPolicy::default()).is_err());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Why rscrypto

`rscrypto` was built as a necessity for Rail Industries, currently in stealth. I needed reproducible builds, small dependency graphs, platform coverage, and throughput under real workloads without dragging in c-libs and weakening my supply chain story.

| What You Need | What `rscrypto` Does |
|---------------|----------------------|
| One primitive without dependency sprawl | Enable one leaf feature, such as `sha2`, `aes-gcm`, or `x25519` |
| A full crypto toolbox | Enable `full` for hashes, checksums, auth, key exchange, signatures, and AEADs |
| Optional ecosystem hooks | Add `getrandom`, `serde`, or `parallel` only when you need them |
| No C toolchain in your build | No C FFI, no vendored C/C++, no OpenSSL/libcrypto dependency |
| Hardware speed without platform lock-in | ISA-specific kernels across x86_64, ARM64, RISC-V, IBM Z, IBM POWER, and Apple Silicon |
| Embedded, WASM, and kernel-adjacent targets | Portable fallbacks for every enabled primitive, `alloc` and `std` opt-in by feature |
| Auditable failure behavior | Opaque verification errors, typed keys/nonces/tags, zeroized secrets |

The design goal is simple: replace both single-primitive dependency piles and native crypto bindings with one auditable Rust crate that's more performant that either option above.

## Performance

The current benchmarks compare `rscrypto` against the crates I would normally reach for: RustCrypto hashes and AEADs, `blake3`, `ed25519-dalek`, `x25519-dalek`, `crc*`, `xxhash-rust`, `rapidhash`, `aegis`, and related baselines.

Linux: **3717 faster comparisons** across **5796** matched comparisons on nine architectures, with a **1.75x** geomean speedup from raw Criterion medians.

| Primitive | Baseline | Result |
|---|---|---|
| SHA-3 / SHAKE | RustCrypto `sha3` | 2.18x / 2.60x geomean; up to 25.83x / 22.41x |
| BLAKE3 | `blake3` | 2.37x geomean on one-shot/keyed/derive-key `>=64 KiB`; up to 7.70x |
| AEAD | RustCrypto AEADs, `aegis` | 1.84x geomean; AES-256-GCM decrypt 2.51x; AES-256-GCM-SIV encrypt 3.79x |
| Ed25519 / X25519 | `ed25519-dalek`, `x25519-dalek` | Ed25519 signing 1.57x geomean; X25519 1.37x |
| Checksums | `crc`, `crc32fast`, `crc64fast`, `crc-fast` | 4.41x geomean; CRC32C 2.24x; CRC64/NVMe 2.35x |

On macOS Apple Silicon, AEAD is the strongest area at 2.60x geomean and checksums land at 4.18x; SHA-2 and SHA-3 are essentially parity at 1.02x and 1.01x.

Numbers: [`benchmark_results/OVERVIEW.md`](benchmark_results/OVERVIEW.md)

## Examples And Docs

| Path | Purpose |
|------|---------|
| [`examples/`](examples/) | Runnable examples with their feature sets |
| [`docs/security.md`](docs/security.md) | Nonce, verification, randomness, and fallback guidance |

## API Conventions

| Family | One-Shot | Streaming | Verify |
|--------|----------|-----------|--------|
| Checksums | `Type::checksum(data)` | `new` / `update` / `finalize` / `reset` | -- |
| Digests | `Type::digest(data)` | `new` / `update` / `finalize` / `reset` | -- |
| XOFs | `Type::xof(data)` | `new` / `update` / `finalize_xof` / `squeeze` | -- |
| MACs | `Type::mac(key, data)` | `new(key)` / `update` / `finalize` / `reset` | `verify` / `verify_tag` |
| KMAC256 | `mac_into(key, cust, data, out)` | `new(key, cust)` / `update` / `finalize_into(out)` | `verify` / `verify_tag` |
| HKDF | `derive(salt, ikm, info, out)` | `new(salt, ikm)` / `expand(info, out)` | -- |
| AEAD | `encrypt(nonce, aad, pt, out)` | -- | `decrypt(nonce, aad, ct, out)` |
| AEAD detached | `encrypt_in_place(nonce, aad, buf)` | -- | `decrypt_in_place(nonce, aad, buf, tag)` |
| Fast hashes | `Type::hash(data)` | -- | -- |

Key, nonce, tag, and signature types round-trip through `from_bytes` / `to_bytes` / `as_bytes`.
Secret key types mask `Debug` output; explicit secret display/export APIs are opt-in and should not be logged.

## Types

Top-level API families: traits, checksums, cryptographic hashes/XOFs, fast hashes, MACs, KDFs, password hashing, Ed25519/X25519, AEADs, typed keys/nonces/tags/signatures, error types, and constant-time utilities.

Full type inventory: [`docs/types.md`](docs/types.md).

## Security Properties

| Property | Implementation |
|----------|---------------|
| Constant-time verification | `ct::constant_time_eq` with `black_box` barrier on all MAC/AEAD/signature paths |
| Zeroize on drop | All secret key types use volatile writes + compiler fence |
| Opaque errors | `VerificationError` is zero-size, leaks no failure details |
| No secret-dependent memory lookups | AES and AEGIS fallbacks use hardware or constant-time portable code |
| Overflow safety | `strict_*` arithmetic + `overflow-checks = true` in release |
| Buffer zeroize on auth failure | All AEAD decrypt paths wipe the output buffer before returning errors |

See [docs/security.md](docs/security.md) for nonce lifecycle, verification handling, PHC verification limits, and RISC-V backend guidance.

## Compliance Posture

`rscrypto` provides FIPS-oriented building blocks, not a FIPS 140-3 validated module.

NIST-aligned primitives include AES-256-GCM, SHA-2, SHA-3/SHAKE, HMAC, KMAC, HKDF, and PBKDF2.
Validation requires a defined module boundary, operational environment, self-tests, documentation, and lab review; this crate does not claim that certificate today.

Roadmap: add a `nist-approved` feature bundle for FIPS-oriented deployments.
It should select approved primitives only; it will not be a validation claim.

Compliance details: [`docs/compliance.md`](docs/compliance.md).

## Feature Flags

| Feature | Default | Enables |
|---------|---------|---------|
| `std` | **Yes** | Runtime CPU detection, I/O adapters. Implies `alloc` |
| `alloc` | **Yes** | Buffered wrappers, `to_vec` methods |
| `checksums` | No | `crc16` + `crc24` + `crc32` + `crc64` |
| `crypto-hashes` | No | `sha2` + `sha3` + `blake2b` + `blake2s` + `blake3` + `ascon-hash` |
| `fast-hashes` | No | `xxh3` + `rapidhash` |
| `hashes` | No | `crypto-hashes` + `fast-hashes` |
| `macs` | No | `hmac` + `kmac` |
| `kdfs` | No | `hkdf` + `pbkdf2` (implies `hmac`); pure key-derivation only |
| `password-hashing` | No | `argon2` + `scrypt` + `phc-strings` |
| `auth` | No | `macs` + `kdfs` + `password-hashing` + `signatures` + `key-exchange` |
| `aead` | No | All 6 AEAD leaves |
| `full` | No | `checksums` + `hashes` + `auth` + `aead` |
| `parallel` | No | Rayon-backed parallel Blake3 and Argon2 lane parallelism. Implies `std` + `blake3` + `argon2` |
| `portable-only` | No | Force portable backends (FIPS / DO-178C / ISO 26262 deployment posture); suppresses runtime SIMD invocation |
| `getrandom` | No | `random()` constructors on key/nonce types and random-salt PHC password hashing |
| `serde` | No | `Serialize`/`Deserialize` on non-secret byte wrappers: nonces, tags, public keys, signatures |
| `serde-secrets` | No | Explicit raw-byte `Serialize`/`Deserialize` for secret keys and shared secrets. Implies `serde` |
| `diag` | No | Dispatch introspection. Implies `std` |

Leaf features: `crc16`, `crc24`, `crc32`, `crc64`, `sha2`, `sha3`, `blake2b`, `blake2s`, `blake3`, `ascon-hash`, `xxh3`, `rapidhash`, `hmac`, `hkdf`, `pbkdf2`, `kmac`, `ed25519`, `x25519`, `argon2`, `scrypt`, `phc-strings`, `aes-gcm`, `aes-gcm-siv`, `chacha20poly1305`, `xchacha20poly1305`, `aegis256`, `ascon-aead`.

## Platform Support

Three-tier SIMD dispatch: compile-time `#[cfg]` --> runtime detection (with `std`) --> portable fallback. Without `std`, only compile-time detection is used.

### CI-Tested Architectures

| Architecture | Key ISA Extensions |
|-------------|-------------------|
| x86_64 (Intel SPR) | AVX-512, VPCLMULQDQ, AES-NI, SHA-NI |
| x86_64 (Intel ICL) | AVX-512, VPCLMULQDQ, AES-NI |
| x86_64 (AMD Zen4/Zen5) | AVX-512, VPCLMULQDQ, AES-NI |
| aarch64 (Graviton3/4) | NEON, PMULL, AES-CE, SHA2-CE |
| aarch64 (macOS, Apple Silicon) | NEON, PMULL, AES-CE, SHA2-CE, SHA3-CE |
| s390x (IBM Z) | z/Vector, VGFM |
| ppc64le (POWER10) | AltiVec, VSX |
| riscv64 (RISE) | V, Zbc |

**no_std build targets**: `thumbv6m-none-eabi`, `riscv32imac-unknown-none-elf`, `aarch64-unknown-none`, `x86_64-unknown-none`, `wasm32-unknown-unknown`, `wasm32-wasip1`.

**NOTE**: RISE RISC-V runners are the only RISC-V runners that I could find that were reliable and worked. I'd like to expand the RISC-V work here, but I simply don't have access to the hardware currently. Also, IBM accepted this repository for thier IBM POWER/Z runners, which are 100% free and without them... I'd never have been able to code for IBM arches... so thank you, IBM.

## Correctness Model

Portable implementations are the byte-for-byte authority.
Hardware, SIMD, and ASM backends are accelerators and are differential-tested against the portable path and official vectors.
Verification errors are opaque, secret keys zeroize on drop, and release builds keep overflow checks enabled.

## Testing

| Layer | What | Command |
|-------|------|---------|
| Unit + integration | Official vectors, differential oracles, API invariants | `just test` |
| Feature matrix | Leaf and bundle reduced-feature combinations | `just test-feature-matrix` |
| Property tests | 256 cases per proptest, run alongside unit + integration | `just test` (nextest) |
| Miri | Memory safety under Stacked Borrows | `just test-miri` |
| Fuzz | Full fuzz suite plus scoped package harnesses with differential oracles | `just test-fuzz` |
| Coverage | Nextest + fuzz corpus LCOV | `just test-all-coverage` |
| Supply chain | `cargo deny` + `cargo audit` | Weekly CI |

## Internals

Module hierarchy and advanced modules: [`docs/architecture.md`](docs/architecture.md).

## MSRV

1.95.0 (edition 2024). Tested on stable and nightly.

## License

MIT OR Apache-2.0
