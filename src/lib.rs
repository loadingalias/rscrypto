//! Zero-dependency-by-default Rust checksums and hashes.
//!
//! # Quick Start
//!
//! ## Checksum
//!
//! ```rust
//! use rscrypto::{Checksum, Crc32C};
//!
//! let data = b"hello world";
//!
//! let checksum = Crc32C::checksum(data);
//!
//! let mut h = Crc32C::new();
//! h.update(b"hello ");
//! h.update(b"world");
//! assert_eq!(h.finalize(), checksum);
//! h.reset();
//! ```
//!
//! ## Digest
//!
//! ```rust
//! use rscrypto::{Digest, Sha256};
//!
//! let data = b"hello world";
//!
//! let digest = Sha256::digest(data);
//!
//! let mut h = Sha256::new();
//! h.update(b"hello ");
//! h.update(b"world");
//! assert_eq!(h.finalize(), digest);
//! h.reset();
//! ```
//!
//! ## Auth
//!
//! ```rust
//! use rscrypto::{Ed25519Keypair, Ed25519SecretKey, HkdfSha256, HmacSha256, Mac};
//!
//! let key = b"shared-secret";
//! let data = b"hello world";
//!
//! let tag = HmacSha256::mac(key, data);
//!
//! let mut mac = HmacSha256::new(key);
//! mac.update(b"hello ");
//! mac.update(b"world");
//! assert_eq!(mac.finalize(), tag);
//! assert!(mac.verify(&tag).is_ok());
//!
//! let mut okm = [0u8; 32];
//! HkdfSha256::new(b"salt", b"input key material").expand(b"context", &mut okm)?;
//! assert_ne!(okm, [0u8; 32]);
//!
//! let keypair = Ed25519Keypair::from_secret_key(Ed25519SecretKey::from_bytes([7u8; 32]));
//! let sig = keypair.sign(b"auth");
//! assert!(keypair.public_key().verify(b"auth", &sig).is_ok());
//! # Ok::<(), rscrypto::auth::HkdfOutputLengthError>(())
//! ```
//!
//! ## XOF
//!
//! ```rust
//! use rscrypto::{Shake256, Xof};
//!
//! let data = b"hello world";
//!
//! let mut h = Shake256::new();
//! h.update(data);
//! let mut xof = h.finalize_xof();
//! let mut out = [0u8; 64];
//! xof.squeeze(&mut out);
//! h.reset();
//!
//! let mut oneshot = Shake256::xof(data);
//! let mut same = [0u8; 64];
//! oneshot.squeeze(&mut same);
//! assert_eq!(out, same);
//! assert_ne!(out, [0u8; 64]);
//! ```
//!
//! `rscrypto` keeps the shipping library inside this repository:
//!
//! - no C FFI
//! - no vendored C/C++ dependency chain
//! - no mandatory runtime dependencies
//!
//! The only optional runtime dependency is `rayon`, behind the `parallel`
//! feature.
//!
//! # Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | Yes | Enables runtime CPU detection for optimal dispatch |
//! | `alloc` | Yes | Enables buffered types (implied by `std`) |
//! | `checksums` | Yes | CRC-16, CRC-24, CRC-32, and CRC-64 algorithms |
//! | `hashes` | Yes | Cryptographic and fast hash families |
//! | `auth` | Yes | HMAC-SHA256, HKDF-SHA256, and Ed25519 |
//! | `parallel` | No | Rayon-based parallel hashing (Blake3) |
//!
//! # Examples
//!
//! - `cargo run --example basic` is the canonical checksum, digest, MAC, KDF, XOF, fast hash, and
//!   I/O adapter specimen.
//! - `cargo run --example introspect` is the advanced dispatch introspection example.
//! - `cargo run --example parallel --features parallel` shows CRC combine-based chunked processing.
//!
//! # Advanced Surfaces
//!
//! - `rscrypto::checksum::config` for checksum force/config controls
//! - `rscrypto::checksum::introspect` for checksum dispatch reporting
//! - `rscrypto::hashes::introspect` for hash kernel reporting
//! - `rscrypto::hashes::fast` for explicit fast-hash family access
//! - `rscrypto::platform` for platform detection and override control
//!
//! ## `no_std` Usage
//!
//! ```toml
//! [dependencies]
//! rscrypto = { version = "0.1", default-features = false, features = ["checksums"] }
//! ```
//!
//! Without `std`, hardware acceleration uses compile-time feature detection only.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]
// Power SIMD backends require nightly-only SIMD/asm + target-feature support.
#![cfg_attr(
  target_arch = "powerpc64",
  feature(asm_experimental_arch, portable_simd, powerpc_target_feature)
)]
// s390x VGFM backend uses vector asm + portable SIMD.
#![cfg_attr(
  target_arch = "s390x",
  feature(asm_experimental_arch, asm_experimental_reg, portable_simd)
)]
// riscv64 ZVBC backend uses vector target features + inline asm.
// NIGHTLY: riscv_ext_intrinsics provides sha256{sum,sig}{0,1} Zknh intrinsics.
#![cfg_attr(
  target_arch = "riscv64",
  feature(
    asm_experimental_arch,
    asm_experimental_reg,
    riscv_target_feature,
    portable_simd,
    riscv_ext_intrinsics
  )
)]
// NIGHTLY: riscv32 SHA-256 Zknh kernel uses scalar crypto intrinsics.
#![cfg_attr(target_arch = "riscv32", feature(riscv_ext_intrinsics))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

// Internal modules (not published as separate crates)
#[cfg(feature = "auth")]
pub mod auth;
#[doc(hidden)]
mod backend;
pub mod platform;
pub mod traits;

#[cfg(feature = "checksums")]
pub mod checksum;

#[cfg(feature = "hashes")]
pub mod hashes;

// ─── Checksum re-exports ────────────────────────────────────────────────────

#[cfg(feature = "auth")]
pub use auth::{
  Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature, HkdfSha256, HmacSha256, verify_ed25519,
};
#[cfg(feature = "checksums")]
pub use checksum::{Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};
// ─── Hash re-exports ────────────────────────────────────────────────────────
#[cfg(feature = "hashes")]
pub use hashes::crypto::{
  AsconHash256, AsconXof, AsconXofReader, Blake3, Blake3Xof, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Sha224, Sha256,
  Sha384, Sha512, Sha512_256, Shake128, Shake128Xof, Shake256, Shake256Xof,
};
#[cfg(feature = "hashes")]
pub use hashes::fast::{RapidHash, RapidHash128, Xxh3, Xxh3_128};
pub use traits::ct;
// ─── Trait re-exports ───────────────────────────────────────────────────────
pub use traits::{Checksum, ChecksumCombine, Mac, VerificationError};
#[cfg(feature = "hashes")]
pub use traits::{Digest, FastHash, Xof};

#[cfg(doctest)]
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
use rscrypto::HashKernelIntrospect;
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
use rscrypto::AsconXof128Xof;
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
use rscrypto::hashes::fast::{RapidHash64, Xxh3_64};
use rscrypto::hashes::introspect::{HashKernelIntrospect, kernel_for as hash_kernel_for};
use rscrypto::hashes::DigestReader;
use rscrypto::{AsconXof, AsconXofReader, RapidHash, Xxh3};

fn assert_hash_introspect<T: HashKernelIntrospect>() {}

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
```
"#]
pub struct __RootSurfaceAudit;

#[cfg(doctest)]
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
    let streaming = squeeze_32(h.finalize_xof());
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

// ─── Compile-time trait assertions ─────────────────────────────────────────
//
// Every public type must be Send + Sync + Debug.  Most must also be Clone.
// These static assertions fail the build if any contract is broken.

#[cfg(test)]
mod send_sync_assertions {
  #![allow(unused_imports)]
  use super::*;

  fn assert_send_sync<T: Send + Sync>() {}
  fn assert_clone<T: Clone>() {}
  fn assert_debug<T: core::fmt::Debug>() {}

  #[test]
  fn public_types_are_send_and_sync() {
    // ── Traits (object-safety is separate; this checks the types) ──
    assert_send_sync::<traits::error::VerificationError>();

    // ── Platform ──
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

    // Dispatch / Diagnostics
    assert_send_sync::<checksum::introspect::DispatchInfo>();

    #[cfg(feature = "diag")]
    {
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
    assert_send_sync::<Shake128Xof>();
    assert_send_sync::<Shake256Xof>();

    // ASCON
    assert_send_sync::<AsconHash256>();
    assert_send_sync::<AsconXof>();
    assert_send_sync::<AsconXofReader>();

    // BLAKE3
    assert_send_sync::<Blake3>();
    assert_send_sync::<Blake3Xof>();

    // Fast hashes
    assert_send_sync::<Xxh3>();
    assert_send_sync::<Xxh3_128>();
    assert_send_sync::<RapidHash>();
    assert_send_sync::<RapidHash128>();
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

  // ── Clone + Debug assertions ──────────────────────────────────────────

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
    assert_clone::<checksum::introspect::DispatchInfo>();

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
    assert_debug::<checksum::introspect::DispatchInfo>();
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
    assert_clone::<Shake128Xof>();
    assert_clone::<Shake256Xof>();
    assert_clone::<AsconHash256>();
    assert_clone::<AsconXof>();
    assert_clone::<AsconXofReader>();
    assert_clone::<Blake3>();
    assert_clone::<Blake3Xof>();
    assert_clone::<Xxh3>();
    assert_clone::<Xxh3_128>();
    assert_clone::<RapidHash>();
    assert_clone::<RapidHash128>();

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
    assert_debug::<Shake128Xof>();
    assert_debug::<Shake256Xof>();
    assert_debug::<AsconHash256>();
    assert_debug::<AsconXof>();
    assert_debug::<AsconXofReader>();
    assert_debug::<Blake3>();
    assert_debug::<Blake3Xof>();
    assert_debug::<Xxh3>();
    assert_debug::<Xxh3_128>();
    assert_debug::<RapidHash>();
    assert_debug::<RapidHash128>();
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
