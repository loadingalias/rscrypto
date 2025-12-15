//! Precomputed constants for CRC algorithms.
//!
//! This module contains lookup tables and folding constants for various CRC
//! polynomials. All tables are computed at compile time.
//!
//! # Cache Alignment
//!
//! Lookup tables are 64-byte (cache line) aligned using [`Aligned64`] to prevent
//! cache line splits during table lookups. This provides a 5-10% performance
//! improvement on modern CPUs.

pub mod gf2;
#[cfg(not(feature = "no-tables"))]
pub mod tables;

pub mod crc16_ccitt_false;
pub mod crc16_ibm;
pub mod crc24_openpgp;
pub mod crc32;
pub mod crc32c;
pub mod crc64;
pub mod crc64_nvme;

/// Wrapper type to force 64-byte (cache line) alignment.
///
/// Used to align lookup tables for optimal cache behavior.
/// The inner type `T` is accessible via `.0`.
#[cfg(not(feature = "no-tables"))]
#[repr(align(64))]
pub struct Aligned64<T>(pub T);
