# rscrypto

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![docs.rs](https://img.shields.io/docsrs/rscrypto)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml)
[![codecov](https://codecov.io/gh/loadingalias/rscrypto/graph/badge.svg)](https://codecov.io/gh/loadingalias/rscrypto)
[![MSRV](https://img.shields.io/badge/MSRV-1.94.1-blue.svg)](https://blog.rust-lang.org/)

> Pure Rust cryptography. Hardware-accelerated. Zero dependencies.

## Quick Start

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

### AEAD (feature: `aead`)

| Cipher | Key | Nonce | Tag | Source |
|--------|-----|-------|-----|--------|
| `Aes256Gcm` | 32B | `Nonce96` (12B) | 16B | `aead/aes256gcm.rs` |
| `Aes256GcmSiv` | 32B | `Nonce96` (12B) | 16B | `aead/aes256gcmsiv.rs` |
| `ChaCha20Poly1305` | 32B | `Nonce96` (12B) | 16B | `aead/chacha20poly1305.rs` |
| `XChaCha20Poly1305` | 32B | `Nonce192` (24B) | 16B | `aead/xchacha20poly1305.rs` |
| `AsconAead128` | 16B | `Nonce128` (16B) | 16B | `aead/ascon128.rs` |
| `Aegis256` | 32B | `Nonce256` (32B) | 16B | `aead/aegis256.rs` |

Each cipher has typed `*Key` and `*Tag` wrappers. Nonces: `Nonce96`, `Nonce128`, `Nonce192`, `Nonce256` in `rscrypto::aead`.

### Authentication (feature: `auth`)

| Type | Purpose | Source |
|------|---------|--------|
| `HmacSha256` / `HmacSha384` / `HmacSha512` | HMAC-SHA2 family (RFC 2104) | `auth/hmac.rs` |
| `HkdfSha256` / `HkdfSha384` | HKDF extract-expand (RFC 5869) | `auth/hkdf.rs` |
| `Kmac256` | Variable-output MAC (SP 800-185) | `auth/kmac.rs` |
| `Ed25519SecretKey` / `PublicKey` / `Signature` / `Keypair` | Ed25519 signatures (RFC 8032) | `auth/ed25519.rs` |
| `X25519SecretKey` / `PublicKey` / `SharedSecret` | X25519 key exchange (RFC 7748) | `auth/x25519.rs` |
| `verify_ed25519` | Free-function signature verification | `auth/ed25519.rs` |

### Cryptographic Hashes (feature: `hashes`)

| Type | Output | Source |
|------|--------|--------|
| `Sha224` / `Sha256` / `Sha384` / `Sha512` / `Sha512_256` | 28-64B | `hashes/crypto/sha*.rs` |
| `Sha3_224` / `Sha3_256` / `Sha3_384` / `Sha3_512` | 28-64B | `hashes/crypto/sha3.rs` |
| `Shake128` / `Shake256` | XOF | `hashes/crypto/sha3.rs` |
| `Blake3` | 32B / XOF | `hashes/crypto/blake3/mod.rs` |
| `AsconHash256` / `AsconXof` / `AsconCxof128` | 32B / XOF | `hashes/crypto/ascon.rs` |
| `Cshake256` | XOF (SP 800-185) | `hashes/crypto/cshake.rs` |

XOF readers: `Shake128XofReader`, `Shake256XofReader`, `Blake3XofReader`, `AsconXofReader`, `AsconCxof128Reader`, `Cshake256XofReader`.

### Checksums (feature: `checksums`)

| Type | Output | Source |
|------|--------|--------|
| `Crc16Ccitt` / `Crc16Ibm` | `u16` | `checksum/crc16/` |
| `Crc24OpenPgp` | `u32` | `checksum/crc24/` |
| `Crc32` / `Crc32C` | `u32` | `checksum/crc32/` |
| `Crc64` / `Crc64Nvme` | `u64` | `checksum/crc64/` |

All implement `ChecksumCombine` for parallel CRC folding.

### Fast Hashes (feature: `hashes`)

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
| `checksum::introspect` | Kernel selection reporting (`DispatchInfo`, `kernel_for`) |
| `hashes::fast` | Explicit fast-hash access (`RapidHashFast64`, `Xxh3_64`, etc.) |
| `hashes::introspect` | Hash kernel reporting (`HashKernelIntrospect`, `kernel_for`) |
| `aead::introspect` | AEAD backend reporting |
| `platform` | CPU detection, override control, capability queries |

## Feature Flags

| Feature | Default | Enables | Notes |
|---------|---------|---------|-------|
| `std` | Yes | Runtime CPU detection, I/O adapters | Implies `alloc` |
| `alloc` | Yes | Buffered wrappers | Can enable without `std` |
| `checksums` | Yes | CRC-16/24/32/64 families | |
| `hashes` | Yes | SHA-2, SHA-3, SHAKE, Blake3, Ascon, XXH3, RapidHash | |
| `auth` | Yes | HMAC, HKDF, KMAC, Ed25519, X25519 | Implies `hashes` |
| `aead` | Yes | AES-GCM, GCM-SIV, ChaCha20-Poly1305, AEGIS, Ascon | |
| `parallel` | No | Rayon-backed Blake3 parallelism | Requires `std` |

## `no_std`

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["checksums", "hashes"] }
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
| Feature matrix | 4 feature combinations on both x86_64 and aarch64 | `just test-feature-matrix` |
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
| `cargo run --example introspect` | Dispatch reporting |
| `cargo run --example parallel --features parallel` | CRC combine-based chunked processing |

## License

MIT OR Apache-2.0
