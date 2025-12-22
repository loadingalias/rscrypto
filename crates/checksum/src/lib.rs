//! High-performance CRC checksums with hardware acceleration.
//!
//! This crate provides implementations of common CRC algorithms with automatic
//! hardware acceleration on supported platforms.
//!
//! # Supported Algorithms
//!
//! | Type | Polynomial | Output | Use Cases |
//! |------|------------|--------|-----------|
//! | [`Crc16Ccitt`] | 0x1021 | `u16` | X.25, HDLC, Bluetooth |
//! | [`Crc16Ibm`] | 0x8005 | `u16` | USB, Modbus |
//! | [`Crc24`] | 0x864CFB | `u32` | OpenPGP Radix-64 |
//! | [`Crc32`] | 0x04C11DB7 | `u32` | Ethernet, ZIP, PNG |
//! | [`Crc32c`] | 0x1EDC6F41 | `u32` | iSCSI, ext4, Btrfs |
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
//! | VPCLMULQDQ | CRC-32, CRC-32C, CRC-64 | ~40 GB/s |
//! | SSE4.2 crc32 | CRC-32C | ~20 GB/s |
//! | PCLMULQDQ | CRC-32, CRC-32C, CRC-64 | ~15 GB/s |
//!
//! ## aarch64
//!
//! | Feature | Algorithms | Throughput |
//! |---------|------------|------------|
//! | CRC32 extension | CRC-32, CRC-32C | ~20 GB/s |
//! | PMULL + EOR3 | CRC-32, CRC-32C, CRC-64 | ~15 GB/s |
//! | PMULL | CRC-32, CRC-32C, CRC-64 | ~12 GB/s |
//!
//! # Example
//!
//! ```ignore
//! use checksum::{Crc32c, Checksum, ChecksumCombine};
//!
//! // One-shot computation (fastest for complete data)
//! let crc = Crc32c::checksum(b"hello world");
//!
//! // Streaming computation
//! let mut hasher = Crc32c::new();
//! hasher.update(b"hello ");
//! hasher.update(b"world");
//! assert_eq!(hasher.finalize(), crc);
//!
//! // Parallel combine (useful for multi-threaded processing)
//! let crc_a = Crc32c::checksum(b"hello ");
//! let crc_b = Crc32c::checksum(b"world");
//! let combined = Crc32c::combine(crc_a, crc_b, 5);
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
mod crc16;
mod crc24;
mod crc32;
mod crc64;
pub mod dispatchers;
pub mod tune;

// Re-export public types
pub use crc16::{Crc16Ccitt, Crc16Ibm};
pub use crc24::Crc24;
pub use crc32::{Crc32, Crc32c};
// Re-export buffered types (requires alloc)
#[cfg(feature = "alloc")]
pub use crc64::{BufferedCrc64, BufferedCrc64Nvme, BufferedCrc64Xz};
pub use crc64::{Crc64, Crc64Nvme, Crc64Xz};
// Re-export traits for convenience
pub use traits::{Checksum, ChecksumCombine};
