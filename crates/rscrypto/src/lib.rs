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
//! # Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | Yes | Enables runtime CPU detection for optimal dispatch |
//! | `alloc` | Yes | Enables buffered types (implied by `std`) |
//! | `checksums` | Yes | CRC-16, CRC-24, CRC-32, and CRC-64 algorithms |
//!
//! ## `no_std` Usage
//!
//! ```toml
//! [dependencies]
//! rscrypto = { version = "0.1", default-features = false, features = ["checksums"] }
//! ```
//!
//! Without `std`, hardware acceleration uses compile-time feature detection only.
#![cfg_attr(not(feature = "std"), no_std)]

// =============================================================================
// Checksums
// =============================================================================

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

// =============================================================================
// Hashes
// =============================================================================

#[cfg(feature = "hashes")]
pub mod hashes {
  pub use ::hashes::{
    Digest, FastHash, Xof, crypto,
    crypto::{Blake3, Blake3Xof, Sha3_256, Sha3_512, Sha256},
    fast,
  };
}

#[cfg(feature = "hashes")]
pub use hashes::crypto::{Blake3, Blake3Xof, Sha3_256, Sha3_512, Sha256};
#[cfg(feature = "hashes")]
pub use hashes::{Digest, Xof};
