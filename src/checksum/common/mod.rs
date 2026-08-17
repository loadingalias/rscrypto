//! Common utilities for CRC computation.
//!
//! This module provides:
//! - Bitwise reference implementations for correctness verification
//! - Portable slice-by-N implementations for all CRC widths
//! - Const-fn lookup table generation for all CRC sizes
//! - GF(2) matrix operations for `combine()` implementation
//! - Generic kernel selection and dispatch infrastructure
//! - Generic test harnesses for CRC property testing
//! - PCLMULQDQ/PMULL folding constants for hardware acceleration
//! - Software prefetch helpers for large-buffer kernels

// CLMUL folding constants and helpers (used by SIMD CRC backends).
#[cfg(all(
  feature = "crc64",
  any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x"
  )
))]
pub(in crate::checksum) mod clmul;
pub(in crate::checksum) mod combine;
pub(in crate::checksum) mod kernels;
pub(in crate::checksum) mod portable;
#[cfg(any(
  all(
    target_arch = "x86_64",
    any(feature = "crc16", feature = "crc24", feature = "crc32", feature = "crc64")
  ),
  all(
    target_arch = "aarch64",
    any(feature = "crc16", feature = "crc24", feature = "crc64")
  ),
  test
))]
pub(in crate::checksum) mod prefetch;
pub(in crate::checksum) mod reference;
pub(in crate::checksum) mod tables;
#[cfg(test)]
pub(in crate::checksum) mod tests;

#[inline]
#[cfg(all(
  feature = "crc16",
  any(target_arch = "powerpc64", target_arch = "riscv64", target_arch = "s390x")
))]
pub(in crate::checksum) const fn low_u16(value: u32) -> u16 {
  let [b0, b1, ..] = value.to_le_bytes();
  u16::from_le_bytes([b0, b1])
}

#[inline]
#[cfg(all(
  any(feature = "crc16", feature = "crc24", feature = "crc32"),
  any(target_arch = "powerpc64", target_arch = "riscv64", target_arch = "s390x")
))]
pub(in crate::checksum) const fn low_u32(value: u64) -> u32 {
  let [b0, b1, b2, b3, ..] = value.to_le_bytes();
  u32::from_le_bytes([b0, b1, b2, b3])
}
