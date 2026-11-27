//! High-performance CRC and Adler-32 checksums (**NOT CRYPTO**).
//!
//! This module provides implementations of common CRC algorithms with automatic
//! hardware acceleration on supported platforms, and portable Adler-32.
//! Checksums detect accidental corruption only; they give no authentication.
//!
//! # Supported Algorithms
//!
//! | Type | Polynomial | Output | Use Cases |
//! |------|------------|--------|-----------|
//! | `Crc16Ccitt` | 0x1021 | `u16` | X.25, HDLC, PACTOR, SD |
//! | `Crc16Ibm` | 0x8005 | `u16` | Legacy protocols, ARC/IBM |
//! | `Crc24OpenPgp` | 0x864CFB | `u32` (24-bit) | OpenPGP (RFC 4880) |
//! | `Crc32` | 0x04C11DB7 | `u32` | Ethernet, gzip, zip, PNG |
//! | `Crc32C` | 0x1EDC6F41 | `u32` | iSCSI, SCTP, ext4, Btrfs |
//! | `Crc64` | 0x42F0E1EBA9EA3693 | `u64` | XZ Utils, 7-Zip |
//! | `Crc64Nvme` | 0xAD93D23594C93659 | `u64` | NVMe specification |
//! | `Adler32` | none (sums modulo 65521) | `u32` | zlib (RFC 1950), HTTP `deflate` |
//!
//! # Examples
//!
//! ```rust
//! # #[cfg(feature = "crc32")]
//! # {
//! use rscrypto::checksum::{Checksum, ChecksumCombine, Crc32};
//!
//! // One-shot computation (fastest for complete data)
//! let data = b"123456789";
//! let crc = Crc32::checksum(data);
//! assert_eq!(crc, 0xCBF4_3926);
//!
//! // Streaming computation
//! let mut hasher = Crc32::new();
//! hasher.update(b"1234");
//! hasher.update(b"56789");
//! assert_eq!(hasher.finalize(), crc);
//!
//! // Parallel combine (useful for multi-threaded processing)
//! let (a, b) = data.split_at(4);
//! let crc_a = Crc32::checksum(a);
//! let crc_b = Crc32::checksum(b);
//! let combined = Crc32::combine(crc_a, crc_b, b.len());
//! assert_eq!(combined, crc);
//! # }
//! ```
//!
//! # Feature Selection
//!
//! ```toml
//! [dependencies]
//! # Smallest CRC-32-only build
//! rscrypto = { version = "0.13", default-features = false, features = ["crc32"] }
//!
//! # All checksum families
//! rscrypto = { version = "0.13", default-features = false, features = ["checksums"] }
//! ```
//!
//! # API Conventions
//!
//! - All checksum types implement [`crate::traits::Checksum`] with `new` / `update` / `finalize` /
//!   `reset`.
//! - One-shot checksum entry points use `Type::checksum(data)`.
//! - The CRC families and Adler-32 also implement [`crate::traits::ChecksumCombine`] with
//!   `Type::combine(left, right, right_len)`.
//!
//! # Advanced
//!
//! The process capabilities select the kernel.
//!
//! - Use [`crate::checksum::introspect`] for the selected kernel.
//! - Use [`crate::platform`] for platform detection and override control.
//!
//! # no_std Support
//!
//! This module is `no_std` compatible. Disable the `std` feature for embedded use:
//!
//! ```toml
//! [dependencies]
//! rscrypto = { version = "0.13", default-features = false, features = ["crc32"] }
//! ```

// Internal macros must be declared before modules that use them.
#[cfg(all(
  feature = "alloc",
  any(
    feature = "crc16",
    feature = "crc24",
    feature = "crc32",
    feature = "crc64",
    feature = "adler32"
  )
))]
#[macro_use]
mod macros;

// The CRC families share this engine; Adler-32 needs none of it.
#[cfg(any(feature = "crc16", feature = "crc24", feature = "crc32", feature = "crc64"))]
mod crc;

#[cfg(feature = "adler32")]
mod adler32;
#[cfg(feature = "crc16")]
mod crc16;
#[cfg(feature = "crc24")]
mod crc24;
#[cfg(feature = "crc32")]
mod crc32;
#[cfg(feature = "crc64")]
mod crc64;
pub mod introspect;
#[cfg(feature = "std")]
pub mod io;
#[cfg(feature = "alloc")]
/// Buffered checksum adapters for workloads composed of many short updates.
pub mod buffered {
  #[cfg(feature = "adler32")]
  pub use crate::checksum::adler32::BufferedAdler32;
  #[cfg(feature = "crc16")]
  pub use crate::checksum::crc16::{BufferedCrc16Ccitt, BufferedCrc16Ibm};
  #[cfg(feature = "crc24")]
  pub use crate::checksum::crc24::BufferedCrc24OpenPgp;
  #[cfg(feature = "crc32")]
  pub use crate::checksum::crc32::{BufferedCrc32, BufferedCrc32C};
  #[cfg(feature = "crc64")]
  pub use crate::checksum::crc64::{BufferedCrc64, BufferedCrc64Nvme};
}

// Re-export public types
#[cfg(feature = "adler32")]
pub use adler32::Adler32;
#[cfg(feature = "crc16")]
pub use crc16::{Crc16Ccitt, Crc16Ibm};
#[cfg(feature = "crc24")]
pub use crc24::Crc24OpenPgp;
#[cfg(feature = "crc32")]
pub use crc32::{Crc32, Crc32C, Crc32Castagnoli, Crc32Ieee};
#[cfg(feature = "crc64")]
pub use crc64::{Crc64, Crc64Nvme, Crc64Xz};
// Re-export I/O adapters (requires std)
#[cfg(feature = "std")]
pub use io::{ChecksumReader, ChecksumWriter};

// Re-export traits for convenience
pub use crate::traits::{Checksum, ChecksumCombine};
