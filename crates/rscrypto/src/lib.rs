//! Pure Rust cryptography with hardware acceleration.
//!
//! `rscrypto` provides high-performance cryptographic primitives with automatic
//! CPU feature detection and optimal kernel selection. Zero dependencies, `no_std`
//! compatible, and hardware-accelerated on x86_64, aarch64, and more.
//!
//! # Quick Start
//!
//! ```
//! use rscrypto::{Checksum, Crc32C};
//!
//! // One-shot computation
//! let crc = Crc32C::checksum(b"hello world");
//! assert_eq!(crc, 0xC99465AA);
//!
//! // Streaming computation
//! let mut hasher = Crc32C::new();
//! hasher.update(b"hello ");
//! hasher.update(b"world");
//! assert_eq!(hasher.finalize(), crc);
//! ```
//!
//! Hash algorithms follow the same top-level pattern when the `hashes` feature
//! is enabled:
//!
//! ```
//! # #[cfg(feature = "hashes")]
//! # {
//! use rscrypto::{Digest, Sha256};
//!
//! let digest = Sha256::digest(b"hello world");
//! assert_eq!(digest.len(), 32);
//! # }
//! ```
//!
//! # Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | Yes | Enables runtime CPU detection for optimal dispatch |
//! | `alloc` | Yes | Enables buffered types (implied by `std`) |
//! | `checksums` | Yes | CRC-16, CRC-24, CRC-32, and CRC-64 algorithms |
//! | `hashes` | No | Cryptographic and fast hash families |
//!
//! ## `no_std` Usage
//!
//! ```toml
//! [dependencies]
//! rscrypto = { version = "0.1", default-features = false, features = ["checksums"] }
//! ```
//!
//! Without `std`, hardware acceleration uses compile-time feature detection only.
//! Final artifacts remain responsible for choosing their panic strategy and,
//! on bare-metal targets, supplying any required panic handler.
#![cfg_attr(not(feature = "std"), no_std)]

// =============================================================================
// Checksums
// =============================================================================

#[cfg(feature = "hashes")]
pub use ::hashes::crypto::{
  AsconHash256, AsconXof128, AsconXof128Xof, Blake3, Blake3Xof, Sha3_224, Sha3_256, Sha3_384, Sha3_512, Sha224, Sha256,
  Sha384, Sha512, Sha512_256, Shake128, Shake128Xof, Shake256, Shake256Xof,
};
#[cfg(feature = "hashes")]
pub use ::hashes::fast::{RapidHash64, RapidHash128, Xxh3_64, Xxh3_128};
// =============================================================================
// Hashes
// =============================================================================
#[cfg(feature = "hashes")]
pub use ::hashes::{Digest, FastHash, Xof, crypto, fast};
#[cfg(all(feature = "hashes", feature = "std"))]
pub use ::hashes::{DigestReader, DigestWriter};
#[cfg(all(feature = "checksums", feature = "alloc"))]
pub use checksum::{
  BufferedCrc16Ccitt, BufferedCrc16Ibm, BufferedCrc24OpenPgp, BufferedCrc32, BufferedCrc32C, BufferedCrc32Castagnoli,
  BufferedCrc32Ieee, BufferedCrc64, BufferedCrc64Nvme, BufferedCrc64Xz,
};
#[cfg(feature = "checksums")]
pub use checksum::{
  // Traits
  Checksum,
  ChecksumCombine,
  // CRC-16
  Crc16Ccitt,
  Crc16Config,
  Crc16Force,
  Crc16Ibm,
  // CRC-24
  Crc24Config,
  Crc24Force,
  Crc24OpenPgp,
  // CRC-32
  Crc32,
  Crc32C,
  Crc32Castagnoli,
  Crc32Config,
  Crc32Force,
  Crc32Ieee,
  // CRC-64
  Crc64,
  Crc64Config,
  Crc64Force,
  Crc64Nvme,
  Crc64Xz,
  // Introspection
  DispatchInfo,
  KernelIntrospect,
  is_hardware_accelerated,
  kernel_for,
  platform_describe,
};

/// Compatibility namespace for callers that already use `rscrypto::hashes::*`.
#[cfg(feature = "hashes")]
pub mod hashes {
  pub use ::hashes::{Digest, FastHash, Xof, crypto, fast};
  #[cfg(feature = "std")]
  pub use ::hashes::{DigestReader, DigestWriter};
}

#[cfg(all(test, feature = "hashes"))]
mod tests {
  use super::{Blake3, FastHash, RapidHash64, Sha512, Xxh3_128, crypto, fast, hashes};

  #[test]
  fn root_hash_reexports_match_underlying_modules() {
    assert_eq!(Blake3::digest(b"abc"), crypto::Blake3::digest(b"abc"));
    assert_eq!(Sha512::digest(b"abc"), crypto::Sha512::digest(b"abc"));
    assert_eq!(RapidHash64::hash(b"abc"), fast::RapidHash64::hash(b"abc"));
    assert_eq!(Xxh3_128::hash(b"abc"), fast::Xxh3_128::hash(b"abc"));
    assert_eq!(hashes::crypto::Sha256::digest(b"abc"), crypto::Sha256::digest(b"abc"));
    assert_eq!(hashes::fast::Xxh3_64::hash(b"abc"), fast::Xxh3_64::hash(b"abc"));
  }
}
