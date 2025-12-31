//! Static kernel name tables for CRC-24.
//!
//! This mirrors the structure used by CRC-32/CRC-64: keep all kernel names in
//! one place so both introspection and dispatch share identifiers.
//!
//! # Kernel Tiers
//!
//! CRC-24 supports only Tiers 0 and 1:
//! - Tier 0 (Reference): Bitwise implementation
//! - Tier 1 (Portable): Slice-by-4/8 table lookup
//!
//! ## Why No SIMD Acceleration?
//!
//! CRC-24/OPENPGP uses a non-reflected (MSB-first) polynomial. This means:
//! - Data bits are processed high-to-low instead of low-to-high
//! - Carryless multiply folding requires additional byte-reversal operations
//! - The performance gain would be marginal for typical OpenPGP message sizes
//! - The OpenPGP use case (ASCII armor integrity) doesn't require extreme throughput
//!
//! If SIMD acceleration is needed in the future, it can be added following the
//! CRC-16/CRC-64 patterns, but the portable implementation is likely sufficient
//! for the foreseeable future.

/// Reference (bitwise) kernel name.
pub use kernels::REFERENCE;

use crate::common::kernels;

/// Portable slice-by-4 kernel name.
pub const PORTABLE_SLICE4: &str = kernels::PORTABLE_SLICE4;
/// Portable slice-by-8 kernel name.
pub const PORTABLE_SLICE8: &str = kernels::PORTABLE_SLICE8;
/// Portable auto-selection kernel name (slice4 vs slice8 by length).
#[allow(dead_code)] // Reserved for introspection
pub const PORTABLE_AUTO: &str = "portable/auto";

/// Portable kernel name table (ordered by increasing work per byte).
#[allow(dead_code)]
pub const PORTABLE_NAMES: &[&str] = &[PORTABLE_SLICE4, PORTABLE_SLICE8];

#[inline]
#[must_use]
pub const fn portable_name_for_len(len: usize, slice4_to_slice8: usize) -> &'static str {
  if len < slice4_to_slice8 {
    PORTABLE_SLICE4
  } else {
    PORTABLE_SLICE8
  }
}
