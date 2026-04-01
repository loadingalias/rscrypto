# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![docs.rs](https://img.shields.io/docsrs/rscrypto)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml)
[![MSRV](https://img.shields.io/badge/MSRV-1.94.0-blue.svg)](https://blog.rust-lang.org/)

> Pure Rust checksums, digests, XOFs, MACs, HKDF, Ed25519, and fast hashes with portable fallbacks and ISA dispatch.

## Quick Start

```rust
use rscrypto::{
  Blake3, Checksum, Crc32C, Digest, Ed25519Keypair, Ed25519SecretKey, HkdfSha256, HmacSha256, Mac, RapidHash, Sha256,
  Shake256, Xof, Xxh3,
};

let checksum = Crc32C::checksum(b"data");
let digest = Blake3::digest(b"data");
let tag = HmacSha256::mac(b"key", b"data");

let mut okm = [0u8; 32];
HkdfSha256::new(b"salt", b"ikm").expand(b"info", &mut okm)?;

let mut xof = Shake256::xof(b"data");
let mut out = [0u8; 32];
xof.squeeze(&mut out);

let mut sha = Sha256::new();
sha.update(b"da");
sha.update(b"ta");
assert_eq!(sha.finalize(), Sha256::digest(b"data"));

let keypair = Ed25519Keypair::from_secret_key(Ed25519SecretKey::from_bytes([7u8; 32]));
let sig = keypair.sign(b"data");
assert!(keypair.public_key().verify(b"data", &sig).is_ok());

let _ = Xxh3::hash(b"data");
let _ = RapidHash::hash(b"data");
assert_eq!(checksum, Crc32C::checksum(b"data"));
# Ok::<(), rscrypto::auth::HkdfOutputLengthError>(())
```

## Purpose

`rscrypto` is a single-crate crypto and checksum toolbox with a deliberately curated root surface. The root exports cover the default algorithms and traits; more specialized controls stay under explicit modules such as `checksum::config`, `checksum::introspect`, `hashes::introspect`, `hashes::fast`, and `platform`.

**Provides**: CRC families, SHA-2, SHA-3, SHAKE, BLAKE3, Ascon hash/XOF, HMAC-SHA256, HKDF-SHA256, Ed25519, XXH3, rapidhash, I/O adapters, dispatch introspection, and platform capability reporting.

**Design**: no C FFI, no vendored C/C++, `no_std` first, portable fallback is authoritative, and ISA-specific kernels are accelerators rather than separate APIs.

## Performance Posture

The canonical competitive report is [`docs/bench/BENCHMARKS.md`](docs/bench/BENCHMARKS.md), currently based on CI run `#23822408700` from March 31, 2026.

- Release gate 1, non-loss rate `((W + T) / total)`: `1831 / 2155 = 84.97%` — passes the `80%` gate.
- Release gate 2, pure win rate `(W / total)`: `1396 / 2155 = 64.78%` — below the `70%` public-release bar.
- Current claim: `rscrypto` is broadly competitive and often faster, but it does not yet clear this project's public-release performance bar.
- Current Ed25519 reality: signing is strong in the canonical sweep (`27W / 1T / 0L`), verification is not (`3W / 4T / 21L`). Do not describe Ed25519 as uniformly behind or uniformly ahead.

## Invariants

| Invariant | What it guarantees | What breaks if violated |
|-----------|--------------------|-------------------------|
| Portable fallback is the authority | SIMD and ISA dispatch only change speed | Wrong digest or checksum output |
| Backends are byte-for-byte equivalent | Dispatch is transparent to callers | Silent data corruption |
| Verification errors stay opaque | MAC and signature checks do not leak extra detail | Timing and oracle risk |
| Secret material is zeroized on drop where stored | Keying material does not linger in owned buffers | Key retention in memory |
| Root exports stay small and uniform | Default UX is easy to discover | Public API turns into a junk drawer |
| Official vectors and differential tests stay green | Algorithms match published behavior and reference crates | Interop and correctness failures |

Repository-level evidence:

- Official vector coverage: `tests/sha256_official_vectors.rs`, `tests/sha3_official_vectors.rs`, `tests/blake3_official_vectors.rs`, `tests/ascon_official_vectors.rs`, `tests/hmac_sha256_vectors.rs`, `tests/hkdf_sha256_vectors.rs`, `tests/ed25519_rfc8032_vectors.rs`
- Differential coverage: `tests/sha256_differential.rs`, `tests/sha512_differential.rs`, `tests/sha3_differential.rs`, `tests/shake128_differential.rs`, `tests/shake256_differential.rs`, `tests/blake3_differential.rs`, `tests/ascon_differential.rs`, `tests/xxh3_differential.rs`, `tests/rapidhash_differential.rs`
- Surface and contract guards: `tests/root_surface.rs`, `tests/api_consistency.rs`, `tests/portable_fallback.rs`, `tests/vectored_dispatch.rs`

## Complete Root Inventory

### Public Modules

| Module | Purpose | Source |
|--------|---------|--------|
| `auth` | HMAC-SHA256, HKDF-SHA256, Ed25519 | `src/lib.rs:172`, `src/auth/mod.rs:35` |
| `checksum` | CRC algorithms, buffered wrappers, config, introspection, I/O | `src/lib.rs:179`, `src/checksum/mod.rs:80` |
| `hashes` | Cryptographic hashes, XOFs, fast hashes, introspection, I/O | `src/lib.rs:182`, `src/hashes/mod.rs:20` |
| `platform` | CPU capability detection and override control | `src/lib.rs:175`, `src/platform/mod.rs:35` |
| `traits` | Core contracts and constant-time helpers | `src/lib.rs:176`, `src/traits/mod.rs:32` |
| `ct` | Constant-time equality and zeroization helpers | `src/lib.rs:200`, `src/traits/ct.rs:24` |

### Traits and Core Utility

| Item | Kind | Purpose | Source |
|------|------|---------|--------|
| `Checksum` | trait | Stateful and one-shot checksum API | `src/traits/checksum.rs:51` |
| `ChecksumCombine` | trait | Parallel CRC combine contract | `src/traits/checksum.rs:280` |
| `Digest` | trait | Stateful and one-shot fixed-output digest API | `src/traits/digest.rs:11` |
| `Xof` | trait | Arbitrary-output squeeze API | `src/traits/xof.rs:7` |
| `Mac` | trait | Keyed streaming MAC API | `src/traits/mac.rs:13` |
| `FastHash` | trait | One-shot seeded fast hash API | `src/traits/fast_hash.rs:13` |
| `VerificationError` | error type | Opaque verification failure | `src/traits/error.rs:38` |
| `ct::constant_time_eq` | fn | Constant-time byte comparison | `src/traits/ct.rs:24` |
| `ct::zeroize` | fn | Best-effort buffer wiping | `src/traits/ct.rs:50` |

### Root Checksums

| Type | Kind | Purpose | Source |
|------|------|---------|--------|
| `Crc16Ccitt` | struct | CRC-16/CCITT | `src/checksum/crc16/mod.rs:411` |
| `Crc16Ibm` | struct | CRC-16/IBM | `src/checksum/crc16/mod.rs:553` |
| `Crc24OpenPgp` | struct | CRC-24/OpenPGP | `src/checksum/crc24/mod.rs:250` |
| `Crc32` | struct | CRC-32/IEEE default | `src/checksum/crc32/mod.rs:585` |
| `Crc32C` | struct | CRC-32C/Castagnoli default | `src/checksum/crc32/mod.rs:728` |
| `Crc64` | struct | CRC-64/XZ default | `src/checksum/crc64/mod.rs:530` |
| `Crc64Nvme` | struct | CRC-64/NVME | `src/checksum/crc64/mod.rs:691` |

### Root Cryptographic Hashes and XOFs

| Type | Kind | Purpose | Source |
|------|------|---------|--------|
| `Sha224` | struct | SHA-224 digest | `src/hashes/crypto/sha224.rs:42` |
| `Sha256` | struct | SHA-256 digest | `src/hashes/crypto/sha256/mod.rs:304` |
| `Sha384` | struct | SHA-384 digest | `src/hashes/crypto/sha384.rs:36` |
| `Sha512` | struct | SHA-512 digest | `src/hashes/crypto/sha512/mod.rs:180` |
| `Sha512_256` | struct | SHA-512/256 digest | `src/hashes/crypto/sha512_256.rs:36` |
| `Sha3_224` | struct | SHA3-224 digest | `src/hashes/crypto/sha3.rs:18` |
| `Sha3_256` | struct | SHA3-256 digest | `src/hashes/crypto/sha3.rs:12` |
| `Sha3_384` | struct | SHA3-384 digest | `src/hashes/crypto/sha3.rs:143` |
| `Sha3_512` | struct | SHA3-512 digest | `src/hashes/crypto/sha3.rs:137` |
| `Shake128` | struct | SHAKE128 state | `src/hashes/crypto/sha3.rs:257` |
| `Shake128XofReader` | struct | SHAKE128 XOF reader | `src/hashes/crypto/sha3.rs:325` |
| `Shake256` | struct | SHAKE256 state | `src/hashes/crypto/sha3.rs:251` |
| `Shake256XofReader` | struct | SHAKE256 XOF reader | `src/hashes/crypto/sha3.rs:394` |
| `Blake3` | struct | BLAKE3 digest and XOF entry point | `src/hashes/crypto/blake3/mod.rs:2318` |
| `Blake3XofReader` | struct | BLAKE3 XOF reader | `src/hashes/crypto/blake3/mod.rs:3122` |
| `AsconHash256` | struct | Ascon hash | `src/hashes/crypto/ascon.rs:383` |
| `AsconXof` | struct | Ascon XOF state | `src/hashes/crypto/ascon.rs:578` |
| `AsconXofReader` | struct | Ascon XOF reader | `src/hashes/crypto/ascon.rs:787` |

### Root Authentication and KDF Types

| Type | Kind | Purpose | Source |
|------|------|---------|--------|
| `HmacSha256` | struct | HMAC-SHA256 MAC state | `src/auth/hmac.rs:34` |
| `HkdfSha256` | struct | HKDF-SHA256 extract/expand state | `src/auth/hkdf.rs:59` |
| `Ed25519SecretKey` | struct | Typed Ed25519 secret key | `src/auth/ed25519.rs:97` |
| `Ed25519PublicKey` | struct | Typed Ed25519 public key | `src/auth/ed25519.rs:160` |
| `Ed25519Signature` | struct | Typed Ed25519 signature | `src/auth/ed25519.rs:210` |
| `Ed25519Keypair` | struct | Typed Ed25519 keypair | `src/auth/ed25519.rs:253` |
| `verify_ed25519` | fn | Free-function Ed25519 verification | `src/auth/ed25519.rs:305` |

### Root Fast Hashes

| Type | Kind | Purpose | Source |
|------|------|---------|--------|
| `Xxh3` | re-export | Canonical 64-bit XXH3 root name | `src/hashes/fast/mod.rs:11` |
| `Xxh3_128` | struct | 128-bit XXH3 | `src/hashes/fast/xxh3.rs:31` |
| `RapidHash` | re-export | Canonical 64-bit rapidhash root name | `src/hashes/fast/mod.rs:10` |
| `RapidHash128` | struct | 128-bit rapidhash | `src/hashes/fast/rapidhash.rs:30` |

`RapidHashFast64` and `RapidHashFast128` are public but intentionally module-only under `rscrypto::hashes::fast::*`. They are tuned for in-process hashing, not the default root surface.

## Public Errors

| Error | When it occurs | Recovery | Source |
|-------|----------------|----------|--------|
| `VerificationError` | MAC or signature verification fails | Reject the input; error is intentionally opaque | `src/traits/error.rs:38` |
| `HkdfOutputLengthError` | HKDF expand request exceeds RFC 5869's 8160-byte limit | Request less output or split derivation | `src/auth/hkdf.rs:13` |
| `platform::OverrideError::AlreadyInitialized` | Detection override changed after cache initialization | Set overrides before first detection | `src/platform/detect.rs:24` |
| `platform::OverrideError::Unsupported` | Override support is unavailable on the current target | Fall back to normal detection | `src/platform/detect.rs:24` |

## Advanced Public Modules

### `checksum`

| Surface | Public items | Source |
|---------|--------------|--------|
| `checksum::config` | `Crc16Config`, `Crc16Force`, `Crc24Config`, `Crc24Force`, `Crc32Config`, `Crc32Force`, `Crc64Config`, `Crc64Force` | `src/checksum/mod.rs:94` |
| `checksum::buffered` | `BufferedCrc16Ccitt`, `BufferedCrc16Ibm`, `BufferedCrc24OpenPgp`, `BufferedCrc32`, `BufferedCrc32C`, `BufferedCrc64`, `BufferedCrc64Nvme` | `src/checksum/mod.rs:84` |
| `checksum::introspect` | `DispatchInfo`, `kernel_for`, `KernelIntrospect`, `is_hardware_accelerated` | `src/checksum/introspect.rs:25`, `src/checksum/introspect.rs:38`, `src/checksum/introspect.rs:90`, `src/checksum/introspect.rs:106` |
| `checksum::io` | `ChecksumReader`, `ChecksumWriter` | `src/traits/io.rs:214`, `src/traits/io.rs:356` |
| Compatibility aliases | `Crc32Ieee`, `Crc32Castagnoli`, `Crc64Xz` | `src/checksum/crc32/mod.rs:592`, `src/checksum/crc32/mod.rs:735`, `src/checksum/crc64/mod.rs:537` |

### `hashes`

| Surface | Public items | Source |
|---------|--------------|--------|
| `hashes::fast` | `Xxh3_64`, `Xxh3_128`, `RapidHash64`, `RapidHash128`, `RapidHashFast64`, `RapidHashFast128` | `src/hashes/fast/xxh3.rs:28`, `src/hashes/fast/xxh3.rs:31`, `src/hashes/fast/rapidhash.rs:26`, `src/hashes/fast/rapidhash.rs:30`, `src/hashes/fast/rapidhash.rs:34`, `src/hashes/fast/rapidhash.rs:38` |
| `hashes::introspect` | `kernel_for`, `HashKernelIntrospect` | `src/hashes/introspect.rs:29`, `src/hashes/introspect.rs:37` |
| `hashes::io` | `DigestReader`, `DigestWriter` | `src/traits/io.rs:490`, `src/traits/io.rs:619` |
| Compatibility aliases | `AsconXof128`, `AsconXof128Reader` | `src/hashes/crypto/ascon.rs:838`, `src/hashes/crypto/ascon.rs:841` |

### `platform`

| Item | Purpose | Source |
|------|---------|--------|
| `Arch` | Architecture identifier | `src/platform/caps.rs:242` |
| `Caps` | 256-bit feature bitset | `src/platform/caps.rs:47` |
| `Detected` | Cached architecture + capabilities snapshot | `src/platform/detect.rs:53` |
| `OverrideError` | Override configuration failure | `src/platform/detect.rs:24` |
| `Description` | Zero-allocation display wrapper for detected state | `src/platform/mod.rs:164` |
| `get`, `caps`, `arch`, `caps_static` | Capability queries | `src/platform/mod.rs:70`, `src/platform/mod.rs:79`, `src/platform/mod.rs:88`, `src/platform/mod.rs:152` |
| `set_override`, `try_set_override`, `clear_override`, `has_override` | Detection override control | `src/platform/mod.rs:112`, `src/platform/mod.rs:120`, `src/platform/mod.rs:133`, `src/platform/mod.rs:140` |
| `describe` | Human-readable detection summary | `src/platform/mod.rs:190` |

## Critical Operations

| Operation | Use when | Source |
|-----------|----------|--------|
| `Checksum::checksum` / `Checksum::new` | One-shot or streaming CRC computation | `src/traits/checksum.rs:120`, `src/traits/checksum.rs:66` |
| `ChecksumCombine::combine` | Parallel CRC chunk folding | `src/traits/checksum.rs:291` |
| `Digest::digest` / `Digest::new` | One-shot or streaming digest computation | `src/traits/digest.rs:56`, `src/traits/digest.rs:22` |
| `Mac::mac` / `Mac::verify_tag` | One-shot MAC and constant-time verification | `src/traits/mac.rs:59`, `src/traits/mac.rs:77` |
| `HkdfSha256::expand` / `HkdfSha256::derive_array` | KDF expand and one-shot derive | `src/auth/hkdf.rs:97`, `src/auth/hkdf.rs:141` |
| `Ed25519Keypair::sign` / `verify_ed25519` | Sign and verify messages | `src/auth/ed25519.rs:286`, `src/auth/ed25519.rs:305` |
| `Xof::squeeze` | Read arbitrary-length output | `src/traits/xof.rs:9` |
| `checksum::introspect::kernel_for` / `hashes::introspect::kernel_for` | Inspect dispatch decisions | `src/checksum/introspect.rs:98`, `src/hashes/introspect.rs:32` |
| `platform::describe` | Report detected capabilities | `src/platform/mod.rs:190` |

## Module Hierarchy

```text
src/
├── lib.rs
├── auth/
│   ├── mod.rs
│   ├── ed25519.rs
│   ├── hkdf.rs
│   └── hmac.rs
├── checksum/
│   ├── mod.rs
│   ├── crc16/
│   ├── crc24/
│   ├── crc32/
│   ├── crc64/
│   ├── common/
│   ├── introspect.rs
│   └── io.rs
├── hashes/
│   ├── mod.rs
│   ├── crypto/
│   │   ├── ascon.rs
│   │   ├── blake3/
│   │   ├── sha224.rs
│   │   ├── sha256/
│   │   ├── sha3.rs
│   │   ├── sha384.rs
│   │   ├── sha512/
│   │   └── sha512_256.rs
│   ├── fast/
│   │   ├── rapidhash.rs
│   │   └── xxh3.rs
│   ├── introspect.rs
│   └── io.rs
├── platform/
│   ├── mod.rs
│   ├── caps.rs
│   └── detect.rs
└── traits/
    ├── mod.rs
    ├── checksum.rs
    ├── digest.rs
    ├── fast_hash.rs
    ├── mac.rs
    ├── xof.rs
    ├── error.rs
    ├── ct.rs
    └── io.rs
```

## Feature Flags

| Feature | Default | Enables | Notes |
|---------|---------|---------|-------|
| `std` | Yes | Runtime CPU detection, I/O adapters | Implies `alloc` |
| `alloc` | Yes via `std` | Buffered checksum wrappers and heap-backed helpers | Can be enabled without `std` |
| `checksums` | Yes | CRC-16, CRC-24, CRC-32, CRC-64 families | Root checksum exports |
| `hashes` | Yes | Digests, XOFs, fast hashes | Root digest and fast-hash exports |
| `auth` | Yes | HMAC-SHA256, HKDF-SHA256, Ed25519 | Depends on `hashes` |
| `parallel` | No | Rayon-backed BLAKE3 parallel work | Requires `std` and `rayon` |
| `diag` | No | Checksum dispatch diagnostics | Advanced debugging surface |
| `testing` | No | Internal validation hooks | Not part of the main release UX |

## `no_std`

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["checksums"] }
```

Without `std`, runtime detection and I/O adapters are unavailable. Compile-time feature detection and portable implementations remain available.

## Examples

| Command | What it covers |
|---------|----------------|
| `cargo run --example basic` | Canonical checksum, digest, MAC, KDF, XOF, fast-hash, and I/O usage |
| `cargo run --example introspect` | Checksum and hash dispatch reporting |
| `cargo run --example parallel --features parallel` | CRC combine-based chunked processing |

## Testing and Development

```bash
just check-all
just test
cargo test --doc
```

## License

MIT OR Apache-2.0
