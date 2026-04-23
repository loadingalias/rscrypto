# rscrypto

> Pure Rust cryptography. Hardware-accelerated. Zero default dependencies.

[![Crates.io](https://img.shields.io/crates/v/rscrypto.svg)](https://crates.io/crates/rscrypto)
[![docs.rs](https://img.shields.io/docsrs/rscrypto)](https://docs.rs/rscrypto)
[![CI](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml/badge.svg)](https://github.com/loadingalias/rscrypto/actions/workflows/commit.yaml)
[![codecov](https://codecov.io/gh/loadingalias/rscrypto/graph/badge.svg)](https://codecov.io/gh/loadingalias/rscrypto)

Single-crate cryptography toolbox. No C FFI, no vendored C/C++, `no_std` first.
Pick only the features you need — one leaf feature, one dependency line.

```toml
[dependencies]
rscrypto = { version = "0.1", default-features = false, features = ["sha2"] }
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
Secret key types mask `Debug` output; use `display_secret()` for hex.

## Complete Type Inventory

### Traits

| Trait | Purpose |
|-------|---------|
| `Checksum` | Stateful + one-shot checksums |
| `ChecksumCombine` | O(log n) parallel CRC combine |
| `Digest` | Fixed-output cryptographic hash |
| `Xof` | Variable-output extendable function |
| `Mac` | Keyed streaming MAC |
| `FastHash` | One-shot seeded non-crypto hash |
| `Aead` | Authenticated encryption |
| `ConstantTimeEq` | Constant-time byte equality |

### Checksums (features: `checksums` or `crc16` / `crc24` / `crc32` / `crc64`)

| Type | Output | Standard |
|------|--------|----------|
| `Crc16Ccitt` / `Crc16Ibm` | `u16` | X.25/HDLC, ARC/IBM |
| `Crc24OpenPgp` | `u32` | RFC 4880 |
| `Crc32` / `Crc32C` | `u32` | Ethernet/gzip, iSCSI/ext4 |
| `Crc64` / `Crc64Nvme` | `u64` | XZ Utils, NVMe |

### Cryptographic Hashes (features: `crypto-hashes` or `sha2` / `sha3` / `blake2b` / `blake2s` / `blake3` / `ascon-hash`)

| Type | Output | Standard |
|------|--------|----------|
| `Sha224` / `Sha256` / `Sha384` / `Sha512` / `Sha512_256` | 28-64B | FIPS 180-4 |
| `Sha3_224` / `Sha3_256` / `Sha3_384` / `Sha3_512` | 28-64B | FIPS 202 |
| `Shake128` / `Shake256` | XOF | FIPS 202 |
| `Cshake256` | XOF | SP 800-185 |
| `Blake2b256` / `Blake2b512` | 32B / 64B | RFC 7693 |
| `Blake2s128` / `Blake2s256` | 16B / 32B | RFC 7693 |
| `Blake3` | 32B / XOF | BLAKE3 spec |
| `AsconHash256` / `AsconXof` / `AsconCxof128` | 32B / XOF | Ascon v1.2 |

XOF readers: `Shake128XofReader`, `Shake256XofReader`, `Cshake256XofReader`, `Blake3XofReader`, `AsconXofReader`, `AsconCxof128Reader`.

### Fast Hashes (features: `fast-hashes` or `xxh3` / `rapidhash`)

| Type | Output |
|------|--------|
| `Xxh3` / `Xxh3_128` | `u64` / `u128` |
| `RapidHash` / `RapidHash128` | `u64` / `u128` |

`BuildHasher` support (requires `alloc`): `Xxh3BuildHasher`, `RapidBuildHasher`.

### MACs & KDFs (features: `macs` / `kdfs` or `hmac` / `hkdf` / `pbkdf2` / `kmac`)

| Type | Tag/Output | Standard |
|------|------------|----------|
| `HmacSha256` / `HmacSha384` / `HmacSha512` | 32-64B | RFC 2104 |
| `Kmac256` | variable | SP 800-185 |
| `HkdfSha256` / `HkdfSha384` | 32-48B PRK | RFC 5869 |
| `Pbkdf2Sha256` / `Pbkdf2Sha512` | variable | RFC 2898 / SP 800-132 |

### Signatures & Key Exchange (features: `signatures` / `key-exchange` or `ed25519` / `x25519`)

| Type | Size | Standard |
|------|------|----------|
| `Ed25519SecretKey` / `Ed25519PublicKey` / `Ed25519Signature` | 32/32/64B | RFC 8032 |
| `Ed25519Keypair` | -- | RFC 8032 |
| `X25519SecretKey` / `X25519PublicKey` / `X25519SharedSecret` | 32B each | RFC 7748 |

### AEAD (feature: `aead` or individual leaves)

| Cipher | Key | Nonce | Tag | Standard |
|--------|-----|-------|-----|----------|
| `Aes256Gcm` | `Aes256GcmKey` 32B | `Nonce96` 12B | `Aes256GcmTag` 16B | SP 800-38D |
| `Aes256GcmSiv` | `Aes256GcmSivKey` 32B | `Nonce96` 12B | `Aes256GcmSivTag` 16B | RFC 8452 |
| `ChaCha20Poly1305` | `ChaCha20Poly1305Key` 32B | `Nonce96` 12B | `ChaCha20Poly1305Tag` 16B | RFC 8439 |
| `XChaCha20Poly1305` | `XChaCha20Poly1305Key` 32B | `Nonce192` 24B | `XChaCha20Poly1305Tag` 16B | draft-irtf |
| `AsconAead128` | `AsconAead128Key` 16B | `Nonce128` 16B | `AsconAead128Tag` 16B | Ascon v1.2 |
| `Aegis256` | `Aegis256Key` 32B | `Nonce256` 32B | `Aegis256Tag` 16B | draft-irtf |

Nonce types: `Nonce96` (12B), `Nonce128` (16B), `Nonce192` (24B), `Nonce256` (32B).

### Error Types

| Error | When | Recovery |
|-------|------|----------|
| `VerificationError` | MAC/AEAD/signature check fails | Reject input (intentionally opaque) |
| `AeadBufferError` | Output buffer wrong size | Fix buffer length |
| `OpenError` | AEAD combined decrypt failure | Buffer or verification |
| `HkdfOutputLengthError` | HKDF expand exceeds max | Request less output |
| `X25519Error` | Low-order DH point | Reject peer key |
| `AsconCxofCustomizationError` | Customization > 256 bytes | Shorten string |
| `InvalidHexError` | Hex decode failure | Fix input |
| `platform::OverrideError` | Override after detection init | Set before first call |

### Utility

| Item | Purpose |
|------|---------|
| `ct::constant_time_eq` | Constant-time byte comparison |
| `ct::zeroize` | Volatile-write buffer wipe |
| `DisplaySecret` | Opt-in hex display for secret keys |

## Security Properties

| Property | Implementation |
|----------|---------------|
| Constant-time verification | `ct::constant_time_eq` with `black_box` barrier on all MAC/AEAD/signature paths |
| Zeroize on drop | All secret key types use volatile writes + compiler fence |
| Opaque errors | `VerificationError` is zero-size, leaks no failure details |
| No secret-dependent memory lookups | AES and AEGIS fallbacks use hardware or constant-time portable code |
| Overflow safety | `strict_*` arithmetic + `overflow-checks = true` in release |
| Buffer zeroize on auth failure | All AEAD decrypt paths wipe the output buffer before returning errors |

See [SECURITY_NOTES.md](SECURITY_NOTES.md) for nonce lifecycle, verification handling, and RISC-V backend guidance.

## Compliance Posture

`rscrypto` is a **primitives crate**, not a FIPS-validated module.
It exposes approved, non-approved, and non-crypto components in one crate for flexibility.
If you need a FIPS-validated module boundary, keep `rscrypto` inside your module or a higher-level `fips`
wrapper; do not claim module validation from this crate alone.

### Approved / boundary examples

Inside a typical boundary, these are the NIST-aligned items:

| Category | Examples |
|---------|----------|
| Symmetric AEAD | `Aes256Gcm` (SP 800-38D) |
| Hash / XOF | `Sha*`, `Shake*` (`FIPS 180-4`, `FIPS 202`) |
| MAC / KDF | `HmacSha*`, `Kmac256`, `HkdfSha256`, `HkdfSha384` |
| Signature / KEX | `Ed25519*`, `X25519*` |

Non-approved in this release:

| Category | Examples |
|----------|----------|
| Cipher variants / AEAD | `Aes256GcmSiv`, `Aegis256`, `ChaCha20Poly1305`, `XChaCha20Poly1305` |
| Hashes | `Blake*`, `Ascon*` (`SHA`/FIPS boundary not yet established) |
| Non-crypto | `Crc*`, `Xxh3`, `RapidHash` |

For the full boundary matrix and FIPS-roadmap, see `docs/tasks/fips.md`.

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
| `kdfs` | No | `hkdf` + `pbkdf2` (implies `hmac`) |
| `auth` | No | `macs` + `kdfs` + `signatures` + `key-exchange` |
| `aead` | No | All 6 AEAD leaves |
| `full` | No | `checksums` + `hashes` + `auth` + `aead` |
| `parallel` | No | Rayon-backed parallel Blake3. Implies `std` + `blake3` |
| `getrandom` | No | `random()` constructors on key/nonce types |
| `serde` | No | `Serialize`/`Deserialize` on keys, nonces, tags, signatures |
| `diag` | No | Dispatch introspection. Implies `std` |

Leaf features: `crc16`, `crc24`, `crc32`, `crc64`, `sha2`, `sha3`, `blake2b`, `blake2s`, `blake3`, `ascon-hash`, `xxh3`, `rapidhash`, `hmac`, `hkdf`, `pbkdf2`, `kmac`, `ed25519`, `x25519`, `aes-gcm`, `aes-gcm-siv`, `chacha20poly1305`, `xchacha20poly1305`, `aegis256`, `ascon-aead`.

## Platform Support

Three-tier SIMD dispatch: compile-time `#[cfg]` --> runtime detection (with `std`) --> portable fallback. Without `std`, only compile-time detection is used.

### CI-Tested Architectures

| Architecture | Key ISA Extensions |
|-------------|-------------------|
| x86_64 (Intel SPR) | AVX-512, VPCLMULQDQ, AES-NI, SHA-NI |
| x86_64 (Intel ICL) | AVX-512, VPCLMULQDQ, AES-NI |
| x86_64 (AMD Zen4/Zen5) | AVX-512, VPCLMULQDQ, AES-NI |
| aarch64 (Graviton3/4) | NEON, PMULL, AES-CE, SHA2-CE |
| s390x (IBM Z) | z/Vector, VGFM |
| ppc64le (POWER10) | AltiVec, VSX |
| riscv64 (RISE) | V, Zbc |

**no_std build targets**: `thumbv6m-none-eabi`, `riscv32imac-unknown-none-elf`, `aarch64-unknown-none`, `x86_64-unknown-none`, `wasm32-unknown-unknown`, `wasm32-wasip1`.

## Invariants

| Invariant | What Breaks If Violated |
|-----------|-------------------------|
| Portable fallback is the authority | Wrong output on any platform |
| All backends produce identical bytes | Silent data corruption |
| Verification errors are opaque | Timing and oracle attacks |
| Secret material zeroized on drop | Key retention in memory |
| Official vectors and differential tests stay green | Interoperability failures |

## Testing

| Layer | What | Command |
|-------|------|---------|
| Unit + integration | 912 tests, official vectors, differential oracles | `just test` |
| Feature matrix | Leaf and bundle reduced-feature combinations | `just test-feature-matrix` |
| Property tests | 256 cases per proptest across 22 files | `just test-proptests` |
| Miri | Memory safety under Stacked Borrows | `just test-miri` |
| Fuzz | 32 targets with differential oracles (15 oracle crates) | `just test-fuzz` |
| Coverage | Nextest + fuzz corpus LCOV | `just coverage` |
| Supply chain | `cargo deny` + `cargo audit` | Weekly CI |

## Module Hierarchy

```text
src/
+-- lib.rs              # Public API, re-exports
+-- aead/               # AES-GCM, AES-GCM-SIV, ChaCha20, XChaCha20, AEGIS, Ascon
+-- auth/               # HMAC, HKDF, KMAC, Ed25519, X25519
+-- checksum/           # CRC families, config, buffered, introspection
+-- hashes/
|   +-- crypto/         # SHA-2, SHA-3, SHAKE, cSHAKE, Blake3, Ascon
|   +-- fast/           # XXH3, RapidHash
+-- hex.rs              # Hex encoding, DisplaySecret
+-- platform/           # CPU detection, SIMD dispatch
+-- backend/            # Internal dispatch infrastructure
+-- traits/             # Checksum, Digest, Mac, Xof, FastHash, Aead, ct, io
```

## Advanced Modules

| Module | Gate | Purpose |
|--------|------|---------|
| `checksum::config` | -- | Force-dispatch controls |
| `checksum::buffered` | `alloc` | Buffered CRC wrappers |
| `checksum::introspect` | `diag` | Kernel selection reporting |
| `hashes::introspect` | `diag` | Hash kernel reporting |
| `aead::introspect` | `diag` | AEAD backend reporting |
| `platform` | -- | CPU detection, override control |
| `traits::io` | `std` | `ChecksumReader/Writer`, `DigestReader/Writer` |

## Coming Soon

**Platforms**: loongarch64, arm32, wasm32 compilation targets.

**Algorithms**:
- AES-128, AES-128-GCM-SIV
- AEGIS-128L / AEGIS-X2 / AEGIS-X4
- HMAC and HKDF for all SHA-2/SHA-3 variants
- Argon2id, scrypt
- RSA, ECDSA (P-256, P-384)
- ML-KEM and post-quantum primitives

**Security**:
- Tighten `unsafe` discipline across all SIMD modules
- Formal third-party audit (Cure53 or equivalent)

## MSRV

1.95.0 (edition 2024). Tested on stable and nightly.

## License

MIT OR Apache-2.0
