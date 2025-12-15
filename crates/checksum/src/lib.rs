//! High-performance checksum algorithms with automatic SIMD acceleration.
//!
//! This crate provides world-class CRC implementations that automatically
//! use the fastest available hardware acceleration on your platform.
//!
//! # Algorithms
//!
//! | Algorithm | Polynomial | Use Cases |
//! |-----------|------------|-----------|
//! | [`Crc32c`] | 0x1EDC6F41 (Castagnoli) | iSCSI, SCTP, Btrfs, ext4, RocksDB |
//! | [`Crc32`] | 0x04C11DB7 (ISO 3309) | Ethernet, gzip, PNG, zip |
//! | [`Crc64`] | 0x42F0E1EBA9EA3693 (ECMA polynomial) | XZ, storage |
//! | [`crc16::Crc16Ibm`] | 0x8005 (IBM) | Modbus, USB, legacy protocols |
//! | [`crc16::Crc16CcittFalse`] | 0x1021 (CCITT-FALSE) | Bluetooth, SD cards |
//! | [`Crc24`] | 0x864CFB (OpenPGP) | OpenPGP, IETF protocols |
//!
//! # Performance
//!
//! Throughput on 64KB buffers:
//!
//! | Platform | CRC32-C | Notes |
//! |----------|---------|-------|
//! | Sapphire Rapids | ~100 GB/s | VPCLMULQDQ + hybrid |
//! | Apple M1/M2/M3 | ~85 GB/s | PMULL + EOR3 |
//! | Ice Lake | ~64 GB/s | VPCLMULQDQ |
//! | Haswell | ~25 GB/s | PCLMULQDQ |
//! | Portable | ~500 MB/s | Slicing-by-8 |
//!
//! # Quick Start
//!
//! ```
//! use checksum::Crc32c;
//!
//! // One-shot (fastest for single buffers)
//! let crc = Crc32c::checksum(b"hello world");
//! assert_eq!(crc, 0xC99465AA);
//!
//! // Streaming (for large data or incremental updates)
//! let mut hasher = Crc32c::new();
//! hasher.update(b"hello ");
//! hasher.update(b"world");
//! assert_eq!(hasher.finalize(), crc);
//! ```
//!
//! # Streaming with `std::io::Write`
//!
//! All checksum types implement [`std::io::Write`] when the `std` feature is enabled,
//! allowing seamless integration with the standard I/O ecosystem:
//!
//! ```
//! # #[cfg(feature = "std")]
//! # fn main() -> std::io::Result<()> {
//! use std::io::Write;
//!
//! use checksum::Crc32c;
//!
//! let mut hasher = Crc32c::new();
//! write!(hasher, "hello ")?;
//! write!(hasher, "world")?;
//! assert_eq!(hasher.finalize(), Crc32c::checksum(b"hello world"));
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "std"))]
//! # fn main() {}
//! ```
//!
//! For file checksums, use [`std::io::copy`]:
//!
//! ```ignore
//! use std::io;
//! use checksum::Crc32c;
//!
//! let mut file = std::fs::File::open("data.bin")?;
//! let mut hasher = Crc32c::new();
//! io::copy(&mut file, &mut hasher)?;
//! let checksum = hasher.finalize();
//! ```
//!
//! # Parallel Processing
//!
//! The [`parallel`] module provides utilities for chunked checksum computation.
//! Users bring their own parallelism (rayon, threads, etc.):
//!
//! ```
//! use checksum::{Crc32c, parallel::checksum_chunks};
//!
//! let data = b"The quick brown fox jumps over the lazy dog";
//! let chunks: Vec<&[u8]> = data.chunks(16).collect();
//!
//! // Sequential combination (works everywhere including no_std)
//! let crc = checksum_chunks::<Crc32c>(&chunks);
//! assert_eq!(crc, Crc32c::checksum(data));
//! ```
//!
//! For true parallelism with rayon (user adds dependency):
//!
//! ```ignore
//! use rayon::prelude::*;
//! use checksum::{Crc32c, parallel::combine_checksums};
//!
//! let chunks: Vec<&[u8]> = large_data.chunks(1024 * 1024).collect();
//!
//! // Compute in parallel
//! let checksums: Vec<(u32, usize)> = chunks
//!     .par_iter()
//!     .map(|chunk| (Crc32c::checksum(chunk), chunk.len()))
//!     .collect();
//!
//! // Combine results in O(n × log(max_len))
//! let final_crc = combine_checksums::<Crc32c>(&checksums);
//! ```
//!
//! # Feature Flags
//!
//! - `std` (default): Enables runtime SIMD detection and `std::io::Write`
//! - `alloc`: Enables features requiring allocation (without full std)
//!
//! # no_std Support
//!
//! This crate is `no_std` compatible. Without `std`:
//! - SIMD is only used if target features are enabled at compile time
//! - Falls back to portable slicing-by-8 implementation

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod combine;
mod constants;
pub mod parallel;
mod params;

// Make simd module accessible for benchmark re-exports
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub mod simd;

#[cfg(not(target_arch = "x86_64"))]
mod simd;

/// Internal module for benchmarks. Not part of public API.
///
/// This module exposes internal implementations that are useful for
/// benchmarking specific backends in isolation. Do not use in production code.
#[doc(hidden)]
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub mod __bench {
  pub mod hybrid {
    pub use crate::simd::x86_64::hybrid::{compute_hybrid_zen4_unchecked, compute_hybrid_zen5_unchecked};
  }
}

/// Table-less CRC implementations using branchless bitwise computation.
///
/// This module provides zero-table CRC implementations for:
/// - Embedded systems with limited memory
/// - WebAssembly targets where binary size matters
/// - Environments without SIMD where 8KB tables are too expensive
///
/// For most use cases, prefer the main [`Crc32c`] and [`Crc32`] APIs which
/// automatically select the fastest implementation for your platform.
pub mod bitwise;

pub mod crc16;
pub mod crc24;
pub mod crc32;
pub mod crc32c;
pub mod crc64;

// Re-exports for convenient access
pub use crc16::{Crc16, Crc16CcittFalse, Crc16Ibm};
pub use crc24::Crc24;
pub use crc32::Crc32;
pub use crc32c::Crc32c;
pub use crc64::{Crc64, Crc64Nvme};
pub use params::CrcParams;
// Re-export traits for convenience
pub use traits::{Checksum, ChecksumCombine};
