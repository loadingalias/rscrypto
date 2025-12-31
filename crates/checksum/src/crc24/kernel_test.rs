//! Kernel testing utilities for CRC-24.
//!
//! This module provides functions to run ALL available CRC-24 kernels on the
//! current platform and return their results. Used by fuzz targets and tests
//! to verify cross-kernel equivalence.
//!
//! # Design
//!
//! The oracle is the bitwise reference implementation, which is obviously
//! correct by inspection. All production kernels (portable slice-by-N)
//! must produce identical results to the reference for any input.
//!
//! Note: CRC-24 currently has portable-only implementations (no SIMD).
//! This module is provided for uniformity and to support future SIMD backends.

/// Result from running a kernel.
#[derive(Debug, Clone, Copy)]
pub struct KernelResult {
  /// Kernel name (e.g., "reference", "portable/slice8")
  pub name: &'static str,
  /// Finalized checksum value (24-bit, masked)
  pub checksum: u32,
}

const INIT: u32 = 0x00B7_04CE;
const MASK: u32 = 0x00FF_FFFF;

/// Run all available CRC-24/OPENPGP kernels on the given data.
///
/// Returns a vector of (kernel_name, checksum) pairs. All checksums should
/// be identical if the kernels are correct. The first entry is always the
/// bitwise reference implementation.
#[must_use]
pub fn run_all_crc24_openpgp_kernels(data: &[u8]) -> alloc::vec::Vec<KernelResult> {
  use alloc::vec::Vec;

  use crate::common::{reference::crc24_bitwise, tables::CRC24_OPENPGP_POLY};

  let mut results = Vec::new();

  // Oracle: bitwise reference
  let reference = crc24_bitwise(CRC24_OPENPGP_POLY, INIT, data) & MASK;
  results.push(KernelResult {
    name: "reference",
    checksum: reference,
  });

  // Portable slice-by-4
  let portable4 = super::portable::crc24_openpgp_slice4(INIT, data) & MASK;
  results.push(KernelResult {
    name: "portable/slice4",
    checksum: portable4,
  });

  // Portable slice-by-8
  let portable8 = super::portable::crc24_openpgp_slice8(INIT, data) & MASK;
  results.push(KernelResult {
    name: "portable/slice8",
    checksum: portable8,
  });

  // Future: Architecture-specific kernels would be added here

  results
}

/// Verify all CRC-24/OPENPGP kernels produce the same result.
///
/// Returns `Ok(checksum)` if all agree, or `Err` with details of mismatches.
pub fn verify_crc24_openpgp_kernels(data: &[u8]) -> Result<u32, alloc::string::String> {
  use alloc::{format, string::ToString};

  let results = run_all_crc24_openpgp_kernels(data);

  let first = results.first().ok_or_else(|| "no kernels available".to_string())?;
  let expected = first.checksum;

  for result in results.iter().skip(1) {
    if result.checksum != expected {
      return Err(format!(
        "kernel mismatch: {} produced 0x{:06X}, but {} produced 0x{:06X}",
        first.name, expected, result.name, result.checksum
      ));
    }
  }

  Ok(expected)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_all_openpgp_kernels_agree_empty() {
    verify_crc24_openpgp_kernels(&[]).expect("kernels should agree on empty input");
  }

  #[test]
  fn test_all_openpgp_kernels_agree_small() {
    verify_crc24_openpgp_kernels(b"123456789").expect("kernels should agree on small input");
  }

  #[test]
  fn test_all_openpgp_kernels_agree_medium() {
    let data: alloc::vec::Vec<u8> = (0..1024).map(|i| (i as u8).wrapping_mul(17)).collect();
    verify_crc24_openpgp_kernels(&data).expect("kernels should agree on medium input");
  }

  #[test]
  fn test_all_openpgp_kernels_agree_large() {
    let data: alloc::vec::Vec<u8> = (0..65536).map(|i| (i as u8).wrapping_mul(31)).collect();
    verify_crc24_openpgp_kernels(&data).expect("kernels should agree on large input");
  }
}
