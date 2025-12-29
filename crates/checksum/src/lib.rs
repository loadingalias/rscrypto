//! High-performance CRC checksums with hardware acceleration.
//!
//! This crate provides implementations of common CRC algorithms with automatic
//! hardware acceleration on supported platforms.
//!
//! # Supported Algorithms
//!
//! | Type | Polynomial | Output | Use Cases |
//! |------|------------|--------|-----------|
//! | [`Crc32`] | 0x04C11DB7 | `u32` | Ethernet, gzip, zip, PNG |
//! | [`Crc32C`] | 0x1EDC6F41 | `u32` | iSCSI, SCTP, ext4, Btrfs |
//! | [`Crc64`] | 0x42F0E1EBA9EA3693 | `u64` | XZ Utils, 7-Zip |
//! | [`Crc64Nvme`] | 0xAD93D23594C93659 | `u64` | NVMe specification |
//!
//! # Hardware Acceleration
//!
//! The following hardware acceleration paths are automatically selected based on
//! detected CPU features:
//!
//! ## x86_64
//!
//! | Feature | Algorithms | Throughput |
//! |---------|------------|------------|
//! | SSE4.2 CRC32 | CRC-32C | ~20 GB/s |
//! | VPCLMULQDQ | CRC-64 | ~35-40 GB/s |
//! | PCLMULQDQ | CRC-64 | ~15 GB/s |
//!
//! ## aarch64
//!
//! | Feature | Algorithms | Throughput |
//! |---------|------------|------------|
//! | CRC extension | CRC-32/CRC-32C | ~15-25 GB/s |
//! | PMULL + EOR3 | CRC-64 | ~15 GB/s |
//! | PMULL | CRC-64 | ~12 GB/s |
//!
//! # Example
//!
//! ```ignore
//! use checksum::{Crc64, Checksum, ChecksumCombine};
//!
//! // One-shot computation (fastest for complete data)
//! let crc = Crc64::checksum(b"hello world");
//!
//! // Streaming computation
//! let mut hasher = Crc64::new();
//! hasher.update(b"hello ");
//! hasher.update(b"world");
//! assert_eq!(hasher.finalize(), crc);
//!
//! // Parallel combine (useful for multi-threaded processing)
//! let crc_a = Crc64::checksum(b"hello ");
//! let crc_b = Crc64::checksum(b"world");
//! let combined = Crc64::combine(crc_a, crc_b, 5);
//! assert_eq!(combined, crc);
//! ```
//!
//! # no_std Support
//!
//! This crate is `no_std` compatible. Disable the `std` feature for embedded use:
//!
//! ```toml
//! [dependencies]
//! checksum = { version = "0.1", default-features = false }
//! ```

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]
// powerpc64 SIMD backends currently require nightly-only SIMD/asm + target-feature support.
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
#![cfg_attr(
  target_arch = "riscv64",
  feature(asm_experimental_arch, asm_experimental_reg, riscv_target_feature)
)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod common;

// Internal macros must be declared before modules that use them.
#[macro_use]
mod macros;

mod crc32;
mod crc64;
pub mod dispatchers;
pub mod tune;

// Re-export public types
// Re-export buffered types (requires alloc)
#[cfg(feature = "alloc")]
pub use crc32::{BufferedCrc32, BufferedCrc32C};
#[cfg(feature = "alloc")]
pub use crc32::{BufferedCrc32Castagnoli, BufferedCrc32Ieee};
pub use crc32::{Crc32, Crc32C, Crc32Castagnoli, Crc32Ieee};
#[cfg(feature = "alloc")]
pub use crc64::{BufferedCrc64, BufferedCrc64Nvme, BufferedCrc64Xz};
pub use crc64::{Crc64, Crc64Nvme, Crc64Xz};
// Re-export traits for convenience
pub use traits::{Checksum, ChecksumCombine};
