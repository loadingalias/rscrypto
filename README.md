# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![docs.rs](https://img.shields.io/docsrs/rscrypto)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml)
[![codecov](https://codecov.io/gh/loadingalias/rscrypto/graph/badge.svg)](https://codecov.io/gh/loadingalias/rscrypto)
[![MSRV](https://img.shields.io/badge/MSRV-1.94.1-blue.svg)](https://blog.rust-lang.org/)

> Pure Rust cryptography. Hardware-accelerated. Zero dependencies.

## Quick Start

Minimal install:

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["crc32"] }
```

Targeted bundle installs:

```toml
[dependencies]
# MACs only
rscrypto = { version = "0.1", default-features = false, features = ["macs"] }

# Signatures only
rscrypto = { version = "0.1", default-features = false, features = ["signatures"] }

# X25519 only
rscrypto = { version = "0.1", default-features = false, features = ["key-exchange"] }
```

Full install:

```toml
[dependencies]
rscrypto = { version = "0.1", features = ["full"] }
```

Feature selection rules:

- Pick a leaf when size matters most: `crc32`, `sha2`, `hmac`, `hkdf`, `ed25519`, `x25519`, `chacha20poly1305`, etc.
- Pick a bundle when you want the whole category: `checksums`, `crypto-hashes`, `fast-hashes`, `hashes`, `macs`, `kdfs`, `signatures`, `key-exchange`, `auth`, `aead`, `full`.
- Keep `default = ["std"]` unless you explicitly need `no_std`.

```rust
use rscrypto::{
  Aead, Blake3, Checksum, ChaCha20Poly1305, ChaCha20Poly1305Key, Crc32C, Digest,
  Ed25519Keypair, Ed25519SecretKey, HkdfSha256, HmacSha256, Kmac256, Mac,
  Sha256, Shake256, X25519SecretKey, Xof, Xxh3, aead::Nonce96,
};

// Checksum
let crc = Crc32C::checksum(b"data");

// Digest (one-shot and streaming)
let hash = Sha256::digest(b"data");
let mut h = Sha256::new();
h.update(b"da"); h.update(b"ta");
assert_eq!(h.finalize(), hash);

// HMAC
let tag = HmacSha256::mac(b"key", b"data");
assert!(HmacSha256::verify_tag(b"key", b"data", &tag).is_ok());

// HKDF
let mut okm = [0u8; 32];
HkdfSha256::new(b"salt", b"ikm").expand(b"info", &mut okm)?;

// XOF
let mut xof = Shake256::xof(b"data");
let mut out = [0u8; 64];
xof.squeeze(&mut out);

// KMAC256
let mut kmac = Kmac256::new(b"key", b"domain=v1");
kmac.update(b"data");
let mut tag32 = [0u8; 32];
kmac.finalize_into(&mut tag32);

// Ed25519
let kp = Ed25519Keypair::from_secret_key(Ed25519SecretKey::from_bytes([7u8; 32]));
let sig = kp.sign(b"data");
assert!(kp.public_key().verify(b"data", &sig).is_ok());

// X25519
let alice = X25519SecretKey::from_bytes([7u8; 32]);
let bob = X25519SecretKey::from_bytes([9u8; 32]);
assert_eq!(alice.diffie_hellman(&bob.public_key())?, bob.diffie_hellman(&alice.public_key())?);

// AEAD
let cipher = ChaCha20Poly1305::new(&ChaCha20Poly1305Key::from_bytes([0x11; 32]));
let nonce = Nonce96::from_bytes([0x22; 12]);
let mut buf = *b"data";
let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf);
cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &tag)?;
assert_eq!(&buf, b"data");

// Fast hashes (non-cryptographic)
let _ = Xxh3::hash(b"data");
let _ = rscrypto::RapidHash::hash(b"data");
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Purpose

Single-crate crypto and checksum toolbox. No C FFI, no vendored C/C++, `no_std` first. Portable fallback is authoritative; ISA-specific kernels are accelerators, not separate APIs.

## API Conventions

- Checksums use `Type::checksum(data)` plus `new` / `update` / `finalize` / `reset`.
- Fixed-output digests use `Type::digest(data)`; XOFs use `Type::xof(data)` and `finalize_xof()`.
- MACs use `Type::mac(key, data)` and `Type::verify_tag(key, data, tag)`; KMAC uses `finalize_into` because output length is caller-chosen.
- HKDF uses `new(salt, ikm)` followed by `expand` / `expand_array`, with one-shot `derive` / `derive_array`.
- Signature, key-exchange, nonce, key, and tag wrappers round-trip through `from_bytes` / `to_bytes` / `as_bytes`.
- AEADs use `encrypt` / `decrypt` for combined buffers and `encrypt_in_place` / `decrypt_in_place` for detached-tag flows.

## Invariants

| Invariant | What breaks if violated |
|-----------|-------------------------|
| Portable fallback is the authority | Wrong digest/checksum/ciphertext output |
| All backends produce identical bytes | Silent data corruption |
| Verification errors are opaque | Timing and oracle attacks |
| Secret material zeroized on drop | Key retention in memory |
| Official vectors and differential tests stay green | Interoperability failures |

## Complete Root Inventory

### Traits

| Trait | Purpose | Source |
|-------|---------|--------|
| `Checksum` | Stateful and one-shot checksum | `traits/checksum.rs:51` |
| `ChecksumCombine` | Parallel CRC combine | `traits/checksum.rs:280` |
| `Digest` | Fixed-output hash | `traits/digest.rs:11` |
| `Xof` | Variable-output squeeze | `traits/xof.rs:7` |
| `Mac` | Keyed streaming MAC | `traits/mac.rs:13` |
| `FastHash` | One-shot seeded hash | `traits/fast_hash.rs:13` |
| `Aead` | Authenticated encryption | `traits/aead.rs:20` |

### AEAD (feature: `aead` or AEAD leaves)

| Cipher | Key | Nonce | Tag | Source |
|--------|-----|-------|-----|--------|
| `Aes256Gcm` | 32B | `Nonce96` (12B) | 16B | `aead/aes256gcm.rs` |
| `Aes256GcmSiv` | 32B | `Nonce96` (12B) | 16B | `aead/aes256gcmsiv.rs` |
| `ChaCha20Poly1305` | 32B | `Nonce96` (12B) | 16B | `aead/chacha20poly1305.rs` |
| `XChaCha20Poly1305` | 32B | `Nonce192` (24B) | 16B | `aead/xchacha20poly1305.rs` |
| `AsconAead128` | 16B | `Nonce128` (16B) | 16B | `aead/ascon128.rs` |
| `Aegis256` | 32B | `Nonce256` (32B) | 16B | `aead/aegis256.rs` |

Each cipher has typed `*Key` and `*Tag` wrappers. Nonces: `Nonce96`, `Nonce128`, `Nonce192`, `Nonce256` in `rscrypto::aead`.

### MACs (feature: `macs` or `hmac` / `kmac`)

| Type | Purpose | Source |
|------|---------|--------|
| `HmacSha256` / `HmacSha384` / `HmacSha512` | HMAC-SHA2 family (RFC 2104) | `auth/hmac.rs` |
| `Kmac256` | Variable-output MAC (SP 800-185) | `auth/kmac.rs` |

### KDFs (feature: `kdfs` or `hkdf`)

| Type | Purpose | Source |
|------|---------|--------|
| `HkdfSha256` / `HkdfSha384` | HKDF extract-expand (RFC 5869) | `auth/hkdf.rs` |

### Signatures (feature: `signatures` or `ed25519`)

| Type | Purpose | Source |
|------|---------|--------|
| `Ed25519SecretKey` / `PublicKey` / `Signature` / `Keypair` | Ed25519 signatures (RFC 8032) | `auth/ed25519.rs` |

### Key Exchange (feature: `key-exchange` or `x25519`)

| Type | Purpose | Source |
|------|---------|--------|
| `X25519SecretKey` / `PublicKey` / `SharedSecret` | X25519 key exchange (RFC 7748) | `auth/x25519.rs` |

### Authentication Umbrella (feature: `auth`)

| Type | Purpose | Source |
|------|---------|--------|
| `verify_ed25519` | Free-function signature verification | `auth/ed25519.rs` |

`auth` enables `macs`, `kdfs`, `signatures`, and `key-exchange`.

### Cryptographic Hashes (feature: `crypto-hashes` or `hashes`)

| Type | Output | Source |
|------|--------|--------|
| `Sha224` / `Sha256` / `Sha384` / `Sha512` / `Sha512_256` | 28-64B | `hashes/crypto/sha*.rs` |
| `Sha3_224` / `Sha3_256` / `Sha3_384` / `Sha3_512` | 28-64B | `hashes/crypto/sha3.rs` |
| `Shake128` / `Shake256` | XOF | `hashes/crypto/sha3.rs` |
| `Blake3` | 32B / XOF | `hashes/crypto/blake3/mod.rs` |
| `AsconHash256` / `AsconXof` / `AsconCxof128` | 32B / XOF | `hashes/crypto/ascon.rs` |
| `Cshake256` | XOF (SP 800-185) | `hashes/crypto/cshake.rs` |

XOF readers: `Shake128XofReader`, `Shake256XofReader`, `Blake3XofReader`, `AsconXofReader`, `AsconCxof128Reader`, `Cshake256XofReader`.

### Checksums (feature: `checksums` or checksum leaves)

| Type | Output | Source |
|------|--------|--------|
| `Crc16Ccitt` / `Crc16Ibm` | `u16` | `checksum/crc16/` |
| `Crc24OpenPgp` | `u32` | `checksum/crc24/` |
| `Crc32` / `Crc32C` | `u32` | `checksum/crc32/` |
| `Crc64` / `Crc64Nvme` | `u64` | `checksum/crc64/` |

All implement `ChecksumCombine` for parallel CRC folding.

### Fast Hashes (feature: `fast-hashes` or `hashes`)

| Type | Output | Source |
|------|--------|--------|
| `Xxh3` / `Xxh3_128` | `u64` / `u128` | `hashes/fast/xxh3.rs` |
| `RapidHash` / `RapidHash128` | `u64` / `u128` | `hashes/fast/rapidhash.rs` |

`RapidHashFast64` / `RapidHashFast128` available under `rscrypto::hashes::fast`.

### Error Types

| Error | When | Recovery | Source |
|-------|------|----------|--------|
| `VerificationError` | MAC/AEAD/signature check fails | Reject input (opaque) | `traits/error.rs:38` |
| `HkdfOutputLengthError` | HKDF expand > 8160B | Request less output | `auth/hkdf.rs:24` |
| `X25519Error` | Low-order DH point | Reject peer key | `auth/x25519.rs:47` |
| `AeadBufferError` | Output buffer wrong size | Fix buffer length | `aead/mod.rs:121` |
| `OpenError` | AEAD decryption failure | Buffer or verification | `aead/mod.rs:152` |
| `AsconCxofCustomizationError` | Customization > 256B | Shorten string | `hashes/crypto/ascon.rs:109` |
| `InvalidHexError` | Hex decode failure | Fix input | `hex.rs:14` |
| `platform::OverrideError` | Override after init | Set before first detection | `platform/detect.rs:24` |

### Utility

| Item | Purpose | Source |
|------|---------|--------|
| `ct::constant_time_eq` | Constant-time byte comparison | `traits/ct.rs` |
| `ct::zeroize` | Best-effort buffer wiping | `traits/ct.rs` |
| `DisplaySecret` | Opt-in secret key hex display | `hex.rs` |
| `InvalidHexError` | Hex decode error | `hex.rs` |

## Advanced Modules

| Module | Purpose |
|--------|---------|
| `checksum::config` | Force-dispatch and configuration (`Crc{16,24,32,64}Config`, `*Force`) |
| `checksum::buffered` | Alloc-backed buffered wrappers (`BufferedCrc32C`, etc.) |
| `checksum::introspect` | Kernel selection reporting (`DispatchInfo`, `kernel_for`) with `diag` |
| `hashes::fast` | Explicit fast-hash access (`RapidHashFast64`, `Xxh3_64`, etc.) |
| `hashes::introspect` | Hash kernel reporting (`HashKernelIntrospect`, `kernel_for`) with `diag` |
| `aead::introspect` | AEAD backend reporting with `diag` |
| `platform` | CPU detection, override control, capability queries |

## Feature Flags

| Feature | Default | Enables | Notes |
|---------|---------|---------|-------|
| `std` | Yes | Runtime CPU detection, I/O adapters | Implies `alloc` |
| `alloc` | Yes | Buffered wrappers | Can enable without `std` |
| `crc16`, `crc24`, `crc32`, `crc64` | No | Individual checksum families | Pick the minimum leaf you need |
| `sha2`, `sha3`, `blake3`, `ascon-hash` | No | Cryptographic hash leaves | |
| `xxh3`, `rapidhash` | No | Fast-hash leaves | Non-cryptographic |
| `hmac`, `hkdf`, `kmac`, `ed25519`, `x25519` | No | Individual auth/KDF leaves | `hkdf` pulls `hmac`; `x25519` stands alone |
| `aes-gcm`, `aes-gcm-siv`, `chacha20poly1305`, `xchacha20poly1305`, `aegis256`, `ascon-aead` | No | AEAD leaves | |
| `checksums` | No | All checksum leaves | Convenience bundle |
| `crypto-hashes` | No | `sha2`, `sha3`, `blake3`, `ascon-hash` | Convenience bundle |
| `fast-hashes` | No | `xxh3`, `rapidhash` | Convenience bundle |
| `hashes` | No | `crypto-hashes` + `fast-hashes` | Convenience bundle |
| `macs` | No | `hmac`, `kmac` | Convenience bundle |
| `kdfs` | No | `hkdf` | Convenience bundle |
| `signatures` | No | `ed25519` | Convenience bundle |
| `key-exchange` | No | `x25519` | Convenience bundle |
| `auth` | No | `macs`, `kdfs`, `signatures`, `key-exchange` | Convenience umbrella |
| `aead` | No | All AEAD leaves | Convenience bundle |
| `full` | No | `checksums`, `hashes`, `auth`, `aead` | Opt into the whole surface |
| `parallel` | No | Rayon-backed Blake3 parallelism | Requires `std` |
| `diag` | No | Advanced checksum/hash/AEAD introspection | Requires `std`; off by default to keep normal builds lean |

## `no_std`

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["crc32"] }
```

Without `std`, runtime detection and I/O adapters are unavailable. Compile-time feature detection and portable fallbacks remain.

## Module Hierarchy

```text
src/
├── lib.rs
├── aead/               # AEAD ciphers (AES-GCM, ChaCha20, AEGIS, Ascon)
├── auth/               # HMAC, HKDF, KMAC, Ed25519, X25519
├── checksum/           # CRC families, config, buffered, introspection
│   ├── crc16/
│   ├── crc24/
│   ├── crc32/          # CRC-32 IEEE + Castagnoli
│   └── crc64/          # CRC-64/XZ + NVME (canonical reference)
├── hashes/
│   ├── crypto/         # SHA-2, SHA-3, SHAKE, Blake3, Ascon, cSHAKE
│   │   ├── blake3/
│   │   ├── sha256/
│   │   └── sha512/
│   └── fast/           # XXH3, RapidHash (non-cryptographic)
├── platform/           # CPU detection, SIMD dispatch
└── traits/             # Checksum, Digest, Mac, Xof, FastHash, Aead, ct
```

## Testing

| Layer | What | Command |
|-------|------|---------|
| Unit + integration | 785 tests, official vectors, differential oracles | `just test` |
| Feature matrix | Leaf and bundle reduced-feature combinations | `just test-feature-matrix` |
| Property tests | 10,000 cases per proptest | `just test-proptests` |
| Miri | Memory safety under Stacked Borrows | `just test-miri` |
| Fuzz | 20 targets with differential oracles (15 oracle crates) | `just test-fuzz` |
| Coverage | Nextest + fuzz corpus LCOV, uploaded to Codecov | `just coverage` |
| Supply chain | `cargo deny` + `cargo audit` | Weekly CI |

**CI platforms**: x86_64 + aarch64 Linux, x86_64 + aarch64 Windows, IBM Z (s390x), IBM POWER10 (ppc64le). **no_std targets**: thumbv6m, riscv32, aarch64-none, x86_64-none, wasm32.

**Fuzz targets by category**: 6 AEAD (roundtrip + forgery + oracle), 4 auth (Ed25519/HMAC/HKDF/KMAC with oracle differential), 5 hash (chunk-split + reset + oracle), 2 Blake3 keyed/derive, 1 CRC combine, 2 fast hash.

## Examples

| Command | Covers |
|---------|--------|
| `cargo run --example basic` | Checksum, digest, MAC, KDF, XOF, fast hash, I/O adapters |
| `cargo run --example introspect --features checksums,hashes,diag` | Dispatch reporting |
| `cargo run --example parallel --features parallel` | CRC combine-based chunked processing |

## License

MIT OR Apache-2.0
