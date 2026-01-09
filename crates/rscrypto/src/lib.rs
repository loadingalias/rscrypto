//! Pure Rust crypto w/ hw-accel.
//!
//! rscrypto provides high-performance cryptographic primitives with automatic
//! CPU feature detection and optimal kernel selection.
//!
//! # Features
//!
//! - **Zero dependencies** — pure Rust, no C, no system libraries
//! - **Hardware acceleration** — AVX-512, AVX2, NEON, PMULL, and more
//! - **Automatic dispatch** — detects CPU features at runtime, selects fastest path
//! - **`no_std` support** — works on embedded and WASM targets
//!
//! # Quick Start
//!
//! ```
//! use rscrypto::{Checksum, Crc32C};
//!
//! // One-shot
//! let crc = Crc32C::checksum(b"hello world");
//!
//! // Streaming
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
//! | `std` | ✓ | Enables standard library (runtime CPU detection) |
//! | `alloc` | ✓ | Enables allocator (via `std`) |
//! | `checksums` | ✓ | CRC16, CRC24, CRC32, CRC64 algorithms |
//!
//! For `no_std` without allocator:
//!
//! ```toml
//! [dependencies]
//! rscrypto = { version = "0.1", default-features = false, features = ["checksums"] }
//! ```
#![cfg_attr(not(feature = "std"), no_std)]

// ─────────────────────────────────────────────────────────────────────────────
// Checksums
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "checksums", feature = "alloc"))]
pub use checksum::{
  BufferedCrc16Ccitt, BufferedCrc16Ibm, BufferedCrc24OpenPgp, BufferedCrc32, BufferedCrc32C, BufferedCrc64Nvme,
  BufferedCrc64Xz,
};
#[cfg(feature = "checksums")]
pub use checksum::{
  // Traits
  Checksum,
  ChecksumCombine,
  // CRC-16
  Crc16Ccitt,
  Crc16Ibm,
  // CRC-24
  Crc24OpenPgp,
  // CRC-32
  Crc32,
  Crc32C,
  // CRC-64
  Crc64,
  Crc64Nvme,
};
