//! Pure Rust cryptography. Hardware-accelerated. `no_std` first.
//!
//! Enable only the primitive families you use. The default feature set is
//! just `std`.
//!
//! ```toml
//! [dependencies]
//! rscrypto = { version = "0.1", default-features = false, features = ["sha2"] }
//! ```
//!
//! # Guides
//!
//! - Repository README: <https://github.com/loadingalias/rscrypto#readme>
//! - Runnable examples: <https://github.com/loadingalias/rscrypto/tree/main/examples>
//! - Additional docs: <https://github.com/loadingalias/rscrypto/tree/main/docs>
//! - Security guidance: <https://github.com/loadingalias/rscrypto/blob/main/docs/security.md>
//!
//! # API Shape
//!
//! - Checksums: `Type::checksum(data)` or `new` / `update` / `finalize`.
//! - Digests: `Type::digest(data)` or `new` / `update` / `finalize`.
//! - XOFs: `Type::xof(data)` or `new` / `update` / `finalize_xof`.
//! - MACs: `Type::mac(key, data)` and `Type::verify_tag(key, data, tag)`.
//! - AEADs: typed keys and nonces, with combined and detached APIs.
#![cfg_attr(
  feature = "sha2",
  doc = r#"
# Quick Start

```rust
use rscrypto::{Digest, Sha256};

let digest = Sha256::digest(b"hello world");

let mut h = Sha256::new();
h.update(b"hello ");
h.update(b"world");
assert_eq!(h.finalize(), digest);
```
"#
)]
#![cfg_attr(
  feature = "chacha20poly1305",
  doc = r#"
# AEAD

```rust
use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key, aead::Nonce96};

let key = ChaCha20Poly1305Key::from_bytes([0x11; 32]);
let nonce = Nonce96::from_bytes([0x22; Nonce96::LENGTH]);
let cipher = ChaCha20Poly1305::new(&key);

let mut buffer = *b"data";
let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buffer)?;
cipher.decrypt_in_place(&nonce, b"aad", &mut buffer, &tag)?;
assert_eq!(&buffer, b"data");
# Ok::<(), Box<dyn std::error::Error>>(())
```
"#
)]
#![cfg_attr(
  all(feature = "password-hashing", feature = "getrandom"),
  doc = r#"
# Password Hashing

```rust
use rscrypto::{Argon2Params, Argon2VerifyPolicy, Argon2id};

let params = Argon2Params::new().build()?;
let encoded = Argon2id::hash_string(&params, b"correct horse battery staple")?;

assert!(
  Argon2id::verify_string_with_policy(
    b"correct horse battery staple",
    &encoded,
    &Argon2VerifyPolicy::default(),
  )
  .is_ok()
);
# Ok::<(), Box<dyn std::error::Error>>(())
```
"#
)]
//! # Feature Groups
//!
//! - `checksums`: CRC families.
//! - `hashes`: SHA-2, SHA-3, BLAKE2, BLAKE3, Ascon, XXH3, RapidHash.
//! - `auth`: MACs, KDFs, password hashing, Ed25519, X25519.
//! - `aead`: AES-GCM, AES-GCM-SIV, ChaCha20-Poly1305, XChaCha20-Poly1305, AEGIS-256, Ascon-AEAD128.
//! - `full`: all public primitive families.
//!
//! Leaf features are available for size-conscious builds.
//!
//! # Security Posture
//!
//! `rscrypto` is a primitives crate, not a FIPS-validated module. It exposes
//! FIPS-aligned and non-FIPS primitives in the same crate. See the repository
//! security guidance for nonce lifecycle, PHC verification limits, and
//! platform fallback notes.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]
// Exotic-architecture backends require nightly-only features (inline asm +
// portable_simd + unstable target-feature flags). Primary targets (x86_64,
// aarch64, wasm) compile on stable Rust 1.95.0.
#![cfg_attr(target_arch = "powerpc64", feature(portable_simd, powerpc_target_feature))]
// s390x VGFM backend uses vector asm + portable SIMD, and
// target_feature_inline_always is still required for inline vector helpers.
#![cfg_attr(
  target_arch = "s390x",
  feature(asm_experimental_reg, portable_simd, target_feature_inline_always)
)]
// riscv64 backends use nightly target-feature flags; individual backend
// families opt into asm register classes, crypto intrinsics, or portable SIMD.
#![cfg_attr(target_arch = "riscv64", feature(riscv_target_feature))]
#![cfg_attr(
  all(
    target_arch = "riscv64",
    any(
      feature = "crc16",
      feature = "crc24",
      feature = "crc32",
      feature = "crc64",
      feature = "xxh3",
      feature = "aes-gcm",
      feature = "aes-gcm-siv",
      feature = "aegis256"
    )
  ),
  feature(asm_experimental_reg)
)]
#![cfg_attr(
  all(
    target_arch = "riscv64",
    any(feature = "sha2", feature = "aes-gcm", feature = "aes-gcm-siv", feature = "aegis256")
  ),
  feature(riscv_ext_intrinsics)
)]
#![cfg_attr(
  all(
    target_arch = "riscv64",
    any(feature = "blake3", feature = "chacha20poly1305", feature = "xchacha20poly1305")
  ),
  feature(portable_simd)
)]
#![cfg_attr(target_arch = "riscv32", feature(riscv_ext_intrinsics, riscv_target_feature))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

// Tests use alloc types (Vec, String) for constructing inputs regardless of feature flags.
// The alloc crate is always in the sysroot; this brings the name into scope for test builds
// when the `alloc` feature is off.
#[cfg(all(test, not(feature = "alloc")))]
extern crate alloc;

// Tests use std-backed runtime feature detection and the test harness regardless
// of whether the crate's `std` feature is enabled.
#[cfg(any(feature = "std", test))]
extern crate std;

#[macro_use]
mod macros;

// Internal modules (not published as separate crates)
#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead",
  feature = "ed25519",
  feature = "x25519"
))]
#[macro_use]
mod hex;

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead"
))]
pub mod aead;
#[cfg(any(
  feature = "hmac",
  feature = "hkdf",
  feature = "kmac",
  feature = "ed25519",
  feature = "x25519",
  feature = "phc-strings",
  feature = "argon2",
  feature = "scrypt"
))]
pub mod auth;
#[doc(hidden)]
mod backend;
pub mod platform;
pub mod traits;

#[cfg(any(feature = "crc16", feature = "crc24", feature = "crc32", feature = "crc64"))]
pub mod checksum;

/// Implement [`std::io::Read`] for an [`Xof`](crate::traits::Xof) type by
/// delegating to `squeeze`.
#[cfg(any(feature = "sha3", feature = "blake3", feature = "ascon-hash"))]
macro_rules! impl_xof_read {
  ($type:ty) => {
    #[cfg(feature = "std")]
    impl std::io::Read for $type {
      #[inline]
      fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.squeeze(buf);
        Ok(buf.len())
      }
    }
  };
}

mod secret;

#[cfg(any(
  feature = "sha2",
  feature = "sha3",
  feature = "blake2b",
  feature = "blake2s",
  feature = "blake3",
  feature = "ascon-hash",
  feature = "xxh3",
  feature = "rapidhash"
))]
pub mod hashes;

#[cfg_attr(not(any(feature = "kmac", feature = "ascon-hash")), allow(dead_code))]
#[inline]
pub(crate) fn bytes_to_bits_saturating(len: usize) -> u64 {
  match u64::try_from(len) {
    Ok(value) => value.saturating_mul(8),
    Err(_) => u64::MAX,
  }
}

// Checksum re-exports.

#[cfg(feature = "aead")]
pub use aead::{AeadBufferError, OpenError};
#[cfg(feature = "aegis256")]
pub use aead::{Aegis256, Aegis256Key, Aegis256Tag};
#[cfg(feature = "aes-gcm")]
pub use aead::{Aes256Gcm, Aes256GcmKey, Aes256GcmTag};
#[cfg(feature = "aes-gcm-siv")]
pub use aead::{Aes256GcmSiv, Aes256GcmSivKey, Aes256GcmSivTag};
#[cfg(feature = "ascon-aead")]
pub use aead::{AsconAead128, AsconAead128Key, AsconAead128Tag};
#[cfg(feature = "chacha20poly1305")]
pub use aead::{ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag};
#[cfg(feature = "xchacha20poly1305")]
pub use aead::{XChaCha20Poly1305, XChaCha20Poly1305Key, XChaCha20Poly1305Tag};
#[cfg(feature = "hkdf")]
pub use auth::HkdfOutputLengthError;
#[cfg(feature = "kmac")]
pub use auth::Kmac256;
#[cfg(feature = "phc-strings")]
pub use auth::PhcError;
#[cfg(feature = "argon2")]
pub use auth::{Argon2Error, Argon2Params, Argon2VerifyPolicy, Argon2Version, Argon2d, Argon2i, Argon2id};
#[cfg(feature = "ed25519")]
pub use auth::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature};
#[cfg(feature = "hkdf")]
pub use auth::{HkdfSha256, HkdfSha384};
#[cfg(feature = "hmac")]
pub use auth::{HmacSha256, HmacSha384, HmacSha512};
#[cfg(feature = "pbkdf2")]
pub use auth::{Pbkdf2Error, Pbkdf2Sha256, Pbkdf2Sha512};
#[cfg(feature = "scrypt")]
pub use auth::{Scrypt, ScryptError, ScryptParams, ScryptVerifyPolicy};
#[cfg(feature = "x25519")]
pub use auth::{X25519Error, X25519PublicKey, X25519SecretKey, X25519SharedSecret};
#[cfg(feature = "crc24")]
pub use checksum::Crc24OpenPgp;
#[cfg(feature = "crc16")]
pub use checksum::{Crc16Ccitt, Crc16Ibm};
#[cfg(feature = "crc32")]
pub use checksum::{Crc32, Crc32C};
#[cfg(feature = "crc64")]
pub use checksum::{Crc64, Crc64Nvme};
// Hash re-exports.
#[cfg(feature = "ascon-hash")]
pub use hashes::crypto::ascon::AsconCxofCustomizationError;
#[cfg(feature = "ascon-hash")]
pub use hashes::crypto::{AsconCxof128, AsconCxof128Reader, AsconHash256, AsconXof, AsconXofReader};
#[cfg(feature = "blake2b")]
pub use hashes::crypto::{Blake2b, Blake2b256, Blake2b512, Blake2bParams};
#[cfg(feature = "blake2s")]
pub use hashes::crypto::{Blake2s128, Blake2s256, Blake2sParams};
#[cfg(feature = "blake3")]
pub use hashes::crypto::{Blake3, Blake3XofReader};
#[cfg(feature = "sha3")]
pub use hashes::crypto::{
  Cshake256, Cshake256XofReader, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Shake128, Shake128XofReader, Shake256,
  Shake256XofReader,
};
#[cfg(feature = "sha2")]
pub use hashes::crypto::{Sha224, Sha256, Sha384, Sha512, Sha512_256};
#[cfg(all(feature = "rapidhash", feature = "alloc"))]
pub use hashes::fast::{RapidBuildHasher, RapidHasher};
#[cfg(feature = "rapidhash")]
pub use hashes::fast::{RapidHash, RapidHash128, RapidHashFast64, RapidHashFast128};
#[cfg(feature = "xxh3")]
pub use hashes::fast::{Xxh3, Xxh3_128};
#[cfg(all(feature = "xxh3", feature = "alloc"))]
pub use hashes::fast::{Xxh3BuildHasher, Xxh3Hasher};
// Hex re-exports.
#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead",
  feature = "ed25519",
  feature = "x25519"
))]
pub use hex::{DisplaySecret, InvalidHexError};
pub use secret::SecretBytes;
// Trait re-exports.
#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead"
))]
pub use traits::Aead;
pub use traits::{Checksum, ChecksumCombine, ConstantTimeEq, Mac, VerificationError, ct};
#[cfg(any(
  feature = "sha2",
  feature = "sha3",
  feature = "blake2b",
  feature = "blake2s",
  feature = "blake3",
  feature = "ascon-hash",
  feature = "xxh3",
  feature = "rapidhash"
))]
pub use traits::{Digest, FastHash, Xof};

#[cfg(all(doctest, feature = "full", feature = "diag"))]
#[doc(hidden)]
#[doc = r#"
```compile_fail
use rscrypto::Crc32Config;
```

```compile_fail
use rscrypto::DispatchInfo;
```

```compile_fail
use rscrypto::kernel_for;
```

```compile_fail
use rscrypto::backend_for;
```

```compile_fail
use rscrypto::backend;
```

```compile_fail
use rscrypto::Crc32Ieee;
```

```compile_fail
use rscrypto::Crc32Castagnoli;
```

```compile_fail
use rscrypto::Crc64Xz;
```

```compile_fail
use rscrypto::AsconXof128;
```

```compile_fail
use rscrypto::AsconXof128Reader;
```

```compile_fail
use rscrypto::BufferedCrc32C;
```

```compile_fail
use rscrypto::Xxh3_64;
```

```compile_fail
use rscrypto::RapidHash64;
```

```compile_fail
use rscrypto::checksum::BufferedCrc32C;
```

```compile_fail
use rscrypto::platform_describe;
```

```compile_fail
use rscrypto::DigestReader;
```

```rust
use rscrypto::checksum::config::Crc32Config;
use rscrypto::checksum::buffered::BufferedCrc32C;
use rscrypto::checksum::introspect::{DispatchInfo, kernel_for};
use rscrypto::checksum::{Crc32Castagnoli, Crc32Ieee, Crc64Xz};
use rscrypto::hashes::fast::{RapidHash64, RapidHashFast128, RapidHashFast64, Xxh3_64};
use rscrypto::hashes::introspect::{KernelIntrospect, kernel_for as hash_kernel_for};
use rscrypto::hashes::DigestReader;
use rscrypto::{AsconXof, AsconXofReader, RapidHash, Xxh3};

fn assert_hash_introspect<T: KernelIntrospect>() {}

let _ = rscrypto::platform::describe();
let _: Crc32Config = rscrypto::Crc32::config();
let _ = kernel_for::<rscrypto::Crc32>(64);
let _ = DispatchInfo::current();
let _ = hash_kernel_for::<rscrypto::Sha256>(1024);
assert_hash_introspect::<rscrypto::Sha256>();
let _ = (core::any::TypeId::of::<Crc32Ieee>(), core::any::TypeId::of::<Crc32Castagnoli>(), core::any::TypeId::of::<Crc64Xz>());
let _ = (core::any::TypeId::of::<AsconXof>(), core::any::TypeId::of::<AsconXofReader>());
let _ = core::any::TypeId::of::<BufferedCrc32C>();
let _ = (core::any::TypeId::of::<Xxh3>(), core::any::TypeId::of::<Xxh3_64>());
let _ = (core::any::TypeId::of::<RapidHash>(), core::any::TypeId::of::<RapidHash64>());
let _ = (core::any::TypeId::of::<RapidHashFast64>(), core::any::TypeId::of::<RapidHashFast128>());
```
"#]
pub struct __RootSurfaceAudit;

#[cfg(all(doctest, feature = "full"))]
#[doc(hidden)]
#[doc = r#"
```rust
use rscrypto::{
  Blake3, Digest, Sha224, Sha256, Sha384, Sha512, Sha512_256, Sha3_224, Sha3_256, Sha3_384, Sha3_512,
};

fn assert_digest_api<D>()
where
  D: Digest,
  D::Output: PartialEq + core::fmt::Debug,
{
  let mut h = D::new();
  h.update(b"abc");
  let expected = h.finalize();
  h.reset();
  h.update(b"abc");
  assert_eq!(h.finalize(), expected);
}

assert_digest_api::<Sha224>();
assert_digest_api::<Sha256>();
assert_digest_api::<Sha384>();
assert_digest_api::<Sha512>();
assert_digest_api::<Sha512_256>();
assert_digest_api::<Sha3_224>();
assert_digest_api::<Sha3_256>();
assert_digest_api::<Sha3_384>();
assert_digest_api::<Sha3_512>();
assert_digest_api::<Blake3>();
```

```rust
use rscrypto::{AsconXof, Blake3, Digest, Shake128, Shake256, Xof};

fn squeeze_32(mut reader: impl Xof) -> [u8; 32] {
  let mut out = [0u8; 32];
  reader.squeeze(&mut out);
  out
}

macro_rules! assert_xof_api {
  ($ty:ty) => {{
    let data = b"abc";
    let mut h = <$ty>::new();
    h.update(data);
    let streaming = squeeze_32(h.clone().finalize_xof());
    h.reset();
    let oneshot = squeeze_32(<$ty>::xof(data));
    assert_eq!(streaming, oneshot);
  }};
}

assert_xof_api!(Shake128);
assert_xof_api!(Shake256);
assert_xof_api!(Blake3);
assert_xof_api!(AsconXof);
```

```rust
use std::io::{Cursor, Read, Write};

use rscrypto::{Checksum as _, Crc32C};

let mut reader = Crc32C::reader(Cursor::new(b"abc".to_vec()));
std::io::copy(&mut reader, &mut std::io::sink())?;
assert_eq!(reader.checksum(), Crc32C::checksum(b"abc"));

let mut writer = Crc32C::writer(Vec::new());
writer.write_all(b"abc")?;
assert_eq!(writer.checksum(), Crc32C::checksum(b"abc"));
# Ok::<(), std::io::Error>(())
```

```compile_fail
use std::io::Cursor;

use rscrypto::{Checksum as _, Crc32C};

let reader = Crc32C::reader(Cursor::new(b"abc".to_vec()));
let _ = reader.crc();
```

```compile_fail
use rscrypto::{Checksum as _, Crc32C};

let writer = Crc32C::writer(Vec::<u8>::new());
let _ = writer.crc();
```
"#]
pub struct __ApiPatternAudit;

// Compile-time trait assertions.
//
// Every public type must be Send + Sync + Debug.  Most must also be Clone.
// These static assertions fail the build if any contract is broken.

#[cfg(all(test, miri))]
mod miri_shadow_tests;

#[cfg(test)]
mod send_sync_assertions {
  #![allow(unused_imports)]
  use super::*;

  fn assert_send_sync<T: Send + Sync>() {}
  fn assert_clone<T: Clone>() {}
  fn assert_debug<T: core::fmt::Debug>() {}

  #[test]
  fn public_types_are_send_and_sync() {
    // Traits. Object safety is separate; this checks the types.
    assert_send_sync::<traits::error::VerificationError>();

    // Platform.
    assert_send_sync::<platform::Caps>();
    assert_send_sync::<platform::Arch>();
    assert_send_sync::<platform::Detected>();
    assert_send_sync::<platform::OverrideError>();
    assert_send_sync::<platform::Description>();
  }

  #[test]
  #[cfg(feature = "checksums")]
  fn checksum_types_are_send_and_sync() {
    // CRC-16
    assert_send_sync::<Crc16Ccitt>();
    assert_send_sync::<Crc16Ibm>();
    assert_send_sync::<checksum::config::Crc16Force>();
    assert_send_sync::<checksum::config::Crc16Config>();

    // CRC-24
    assert_send_sync::<Crc24OpenPgp>();
    assert_send_sync::<checksum::config::Crc24Force>();
    assert_send_sync::<checksum::config::Crc24Config>();

    // CRC-32
    assert_send_sync::<Crc32>();
    assert_send_sync::<Crc32C>();
    assert_send_sync::<checksum::config::Crc32Force>();
    assert_send_sync::<checksum::config::Crc32Config>();

    // CRC-64
    assert_send_sync::<Crc64>();
    assert_send_sync::<Crc64Nvme>();
    assert_send_sync::<checksum::config::Crc64Force>();
    assert_send_sync::<checksum::config::Crc64Config>();

    #[cfg(feature = "diag")]
    {
      assert_send_sync::<checksum::introspect::DispatchInfo>();
      assert_send_sync::<checksum::diag::SelectionReason>();
      assert_send_sync::<checksum::diag::Crc32Polynomial>();
      assert_send_sync::<checksum::diag::Crc64Polynomial>();
      assert_send_sync::<checksum::diag::Crc32SelectionDiag>();
      assert_send_sync::<checksum::diag::Crc64SelectionDiag>();
    }
  }

  #[test]
  #[cfg(all(feature = "checksums", feature = "alloc"))]
  fn buffered_checksum_types_are_send_and_sync() {
    assert_send_sync::<checksum::buffered::BufferedCrc16Ccitt>();
    assert_send_sync::<checksum::buffered::BufferedCrc16Ibm>();
    assert_send_sync::<checksum::buffered::BufferedCrc24OpenPgp>();
    assert_send_sync::<checksum::buffered::BufferedCrc32>();
    assert_send_sync::<checksum::buffered::BufferedCrc32C>();
    assert_send_sync::<checksum::buffered::BufferedCrc64>();
    assert_send_sync::<checksum::buffered::BufferedCrc64Nvme>();
  }

  #[test]
  #[cfg(feature = "hashes")]
  fn hash_types_are_send_and_sync() {
    // SHA-2
    assert_send_sync::<Sha256>();
    assert_send_sync::<Sha224>();
    assert_send_sync::<Sha512>();
    assert_send_sync::<Sha384>();
    assert_send_sync::<Sha512_256>();

    // SHA-3
    assert_send_sync::<Sha3_256>();
    assert_send_sync::<Sha3_224>();
    assert_send_sync::<Sha3_512>();
    assert_send_sync::<Sha3_384>();
    assert_send_sync::<Shake128>();
    assert_send_sync::<Shake256>();
    assert_send_sync::<Shake128XofReader>();
    assert_send_sync::<Shake256XofReader>();
    assert_send_sync::<Cshake256>();
    assert_send_sync::<Cshake256XofReader>();

    // ASCON
    assert_send_sync::<AsconHash256>();
    assert_send_sync::<AsconXof>();
    assert_send_sync::<AsconXofReader>();
    assert_send_sync::<AsconCxof128>();
    assert_send_sync::<AsconCxof128Reader>();

    // BLAKE3
    assert_send_sync::<Blake3>();
    assert_send_sync::<Blake3XofReader>();

    // Fast hashes
    assert_send_sync::<Xxh3>();
    assert_send_sync::<Xxh3_128>();
    assert_send_sync::<RapidHash>();
    assert_send_sync::<RapidHash128>();
    assert_send_sync::<hashes::fast::RapidHashFast64>();
    assert_send_sync::<hashes::fast::RapidHashFast128>();

    // BuildHasher types
    #[cfg(feature = "alloc")]
    {
      assert_send_sync::<Xxh3BuildHasher>();
      assert_send_sync::<Xxh3Hasher>();
      assert_send_sync::<RapidBuildHasher>();
      assert_send_sync::<RapidHasher>();
    }
  }

  #[test]
  #[cfg(all(feature = "checksums", feature = "std"))]
  fn io_adapter_types_are_send_and_sync() {
    // ChecksumReader/Writer are Send+Sync when their inner types are
    assert_send_sync::<traits::io::ChecksumReader<std::io::Cursor<Vec<u8>>, Crc32C>>();
    assert_send_sync::<traits::io::ChecksumWriter<Vec<u8>, Crc32C>>();
  }

  #[test]
  #[cfg(all(feature = "hashes", feature = "std"))]
  fn digest_io_adapter_types_are_send_and_sync() {
    assert_send_sync::<hashes::DigestReader<std::io::Cursor<Vec<u8>>, Sha256>>();
    assert_send_sync::<hashes::DigestWriter<Vec<u8>, Sha256>>();
  }

  // Clone + Debug assertions.

  #[test]
  fn platform_types_are_clone_and_debug() {
    assert_clone::<platform::Caps>();
    assert_clone::<platform::Arch>();
    assert_clone::<platform::Detected>();
    assert_clone::<platform::OverrideError>();
    assert_clone::<platform::Description>();
    assert_clone::<traits::error::VerificationError>();

    assert_debug::<platform::Caps>();
    assert_debug::<platform::Arch>();
    assert_debug::<platform::Detected>();
    assert_debug::<platform::OverrideError>();
    assert_debug::<platform::Description>();
    assert_debug::<traits::error::VerificationError>();
  }

  #[test]
  #[cfg(feature = "checksums")]
  fn checksum_types_are_clone_and_debug() {
    assert_clone::<Crc16Ccitt>();
    assert_clone::<Crc16Ibm>();
    assert_clone::<Crc24OpenPgp>();
    assert_clone::<Crc32>();
    assert_clone::<Crc32C>();
    assert_clone::<Crc64>();
    assert_clone::<Crc64Nvme>();
    assert_clone::<checksum::config::Crc16Force>();
    assert_clone::<checksum::config::Crc16Config>();
    assert_clone::<checksum::config::Crc24Force>();
    assert_clone::<checksum::config::Crc24Config>();
    assert_clone::<checksum::config::Crc32Force>();
    assert_clone::<checksum::config::Crc32Config>();
    assert_clone::<checksum::config::Crc64Force>();
    assert_clone::<checksum::config::Crc64Config>();
    assert_debug::<Crc16Ccitt>();
    assert_debug::<Crc16Ibm>();
    assert_debug::<Crc24OpenPgp>();
    assert_debug::<Crc32>();
    assert_debug::<Crc32C>();
    assert_debug::<Crc64>();
    assert_debug::<Crc64Nvme>();
    assert_debug::<checksum::config::Crc16Force>();
    assert_debug::<checksum::config::Crc16Config>();
    assert_debug::<checksum::config::Crc24Force>();
    assert_debug::<checksum::config::Crc24Config>();
    assert_debug::<checksum::config::Crc32Force>();
    assert_debug::<checksum::config::Crc32Config>();
    assert_debug::<checksum::config::Crc64Force>();
    assert_debug::<checksum::config::Crc64Config>();
    #[cfg(feature = "diag")]
    {
      assert_clone::<checksum::introspect::DispatchInfo>();
      assert_debug::<checksum::introspect::DispatchInfo>();
    }
  }

  #[test]
  #[cfg(all(feature = "checksums", feature = "alloc"))]
  fn buffered_checksum_types_are_clone_and_debug() {
    assert_debug::<checksum::buffered::BufferedCrc16Ccitt>();
    assert_debug::<checksum::buffered::BufferedCrc16Ibm>();
    assert_debug::<checksum::buffered::BufferedCrc24OpenPgp>();
    assert_debug::<checksum::buffered::BufferedCrc32>();
    assert_debug::<checksum::buffered::BufferedCrc32C>();
    assert_debug::<checksum::buffered::BufferedCrc64>();
    assert_debug::<checksum::buffered::BufferedCrc64Nvme>();
  }

  #[test]
  #[cfg(feature = "hashes")]
  fn hash_types_are_clone_and_debug() {
    assert_clone::<Sha256>();
    assert_clone::<Sha224>();
    assert_clone::<Sha512>();
    assert_clone::<Sha384>();
    assert_clone::<Sha512_256>();
    assert_clone::<Sha3_256>();
    assert_clone::<Sha3_224>();
    assert_clone::<Sha3_512>();
    assert_clone::<Sha3_384>();
    assert_clone::<Shake128>();
    assert_clone::<Shake256>();
    assert_clone::<Shake128XofReader>();
    assert_clone::<Shake256XofReader>();
    assert_clone::<Cshake256>();
    assert_clone::<Cshake256XofReader>();
    assert_clone::<AsconHash256>();
    assert_clone::<AsconXof>();
    assert_clone::<AsconXofReader>();
    assert_clone::<AsconCxof128>();
    assert_clone::<AsconCxof128Reader>();
    assert_clone::<Blake3>();
    assert_clone::<Blake3XofReader>();
    assert_clone::<Xxh3>();
    assert_clone::<Xxh3_128>();
    assert_clone::<RapidHash>();
    assert_clone::<RapidHash128>();
    assert_clone::<hashes::fast::RapidHashFast64>();
    assert_clone::<hashes::fast::RapidHashFast128>();

    assert_debug::<Sha256>();
    assert_debug::<Sha224>();
    assert_debug::<Sha512>();
    assert_debug::<Sha384>();
    assert_debug::<Sha512_256>();
    assert_debug::<Sha3_256>();
    assert_debug::<Sha3_224>();
    assert_debug::<Sha3_512>();
    assert_debug::<Sha3_384>();
    assert_debug::<Shake128>();
    assert_debug::<Shake256>();
    assert_debug::<Shake128XofReader>();
    assert_debug::<Shake256XofReader>();
    assert_debug::<Cshake256>();
    assert_debug::<Cshake256XofReader>();
    assert_debug::<AsconHash256>();
    assert_debug::<AsconXof>();
    assert_debug::<AsconXofReader>();
    assert_debug::<AsconCxof128>();
    assert_debug::<AsconCxof128Reader>();
    assert_debug::<Blake3>();
    assert_debug::<Blake3XofReader>();
    assert_debug::<Xxh3>();
    assert_debug::<Xxh3_128>();
    assert_debug::<RapidHash>();
    assert_debug::<RapidHash128>();
    assert_debug::<hashes::fast::RapidHashFast64>();
    assert_debug::<hashes::fast::RapidHashFast128>();

    // BuildHasher types
    #[cfg(feature = "alloc")]
    {
      assert_clone::<Xxh3BuildHasher>();
      assert_clone::<RapidBuildHasher>();
      assert_debug::<Xxh3BuildHasher>();
      assert_debug::<Xxh3Hasher>();
      assert_debug::<RapidBuildHasher>();
      assert_debug::<RapidHasher>();
    }
  }

  #[test]
  #[cfg(all(feature = "checksums", feature = "std"))]
  fn io_adapter_types_are_debug() {
    assert_debug::<traits::io::ChecksumReader<std::io::Cursor<Vec<u8>>, Crc32C>>();
    assert_debug::<traits::io::ChecksumWriter<Vec<u8>, Crc32C>>();
  }

  #[test]
  #[cfg(all(feature = "hashes", feature = "std"))]
  fn digest_io_adapter_types_are_debug() {
    assert_debug::<hashes::DigestReader<std::io::Cursor<Vec<u8>>, Sha256>>();
    assert_debug::<hashes::DigestWriter<Vec<u8>, Sha256>>();
  }
}
