//! Kernel testing utilities for CRC-16.
//!
//! This module provides functions to run ALL available CRC-16 kernels on the
//! current platform and return their results. Used by fuzz targets and tests
//! to verify cross-kernel equivalence.
//!
//! # Design
//!
//! The oracle is the bitwise reference implementation, which is obviously
//! correct by inspection. All production kernels (portable slice-by-N, SIMD)
//! must produce identical results to the reference for any input.

use crate::dispatchers::Crc16Fn;

/// Result from running a kernel.
#[derive(Debug, Clone, Copy)]
pub struct KernelResult {
  /// Kernel name (e.g., "reference", "portable/slice8", "x86_64/pclmul")
  pub name: &'static str,
  /// Finalized checksum value
  pub checksum: u16,
}

/// Run all available CRC-16/CCITT (X25) kernels on the given data.
///
/// Returns a vector of (kernel_name, checksum) pairs. All checksums should
/// be identical if the kernels are correct. The first entry is always the
/// bitwise reference implementation.
#[must_use]
pub fn run_all_crc16_ccitt_kernels(data: &[u8]) -> alloc::vec::Vec<KernelResult> {
  use alloc::vec::Vec;

  use crate::common::{reference::crc16_bitwise, tables::CRC16_CCITT_POLY};

  let mut results = Vec::new();

  // Oracle: bitwise reference
  // CRC-16/CCITT uses init=0xFFFF, xorout=0xFFFF
  let reference = crc16_bitwise(CRC16_CCITT_POLY, !0u16, data) ^ !0u16;
  results.push(KernelResult {
    name: "reference",
    checksum: reference,
  });

  // Portable slice-by-4
  let portable4 = super::portable::crc16_ccitt_slice4(!0u16, data) ^ !0u16;
  results.push(KernelResult {
    name: "portable/slice4",
    checksum: portable4,
  });

  // Portable slice-by-8
  let portable8 = super::portable::crc16_ccitt_slice8(!0u16, data) ^ !0u16;
  results.push(KernelResult {
    name: "portable/slice8",
    checksum: portable8,
  });

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    run_x86_64_kernels_ccitt(data, &mut results);
  }

  #[cfg(target_arch = "aarch64")]
  {
    run_aarch64_kernels_ccitt(data, &mut results);
  }

  #[cfg(target_arch = "powerpc64")]
  {
    run_power_kernels_ccitt(data, &mut results);
  }

  #[cfg(target_arch = "s390x")]
  {
    run_s390x_kernels_ccitt(data, &mut results);
  }

  #[cfg(target_arch = "riscv64")]
  {
    run_riscv64_kernels_ccitt(data, &mut results);
  }

  results
}

/// Run all available CRC-16/IBM (ARC) kernels on the given data.
#[must_use]
pub fn run_all_crc16_ibm_kernels(data: &[u8]) -> alloc::vec::Vec<KernelResult> {
  use alloc::vec::Vec;

  use crate::common::{reference::crc16_bitwise, tables::CRC16_IBM_POLY};

  let mut results = Vec::new();

  // Oracle: bitwise reference
  // CRC-16/IBM uses init=0x0000, xorout=0x0000
  let reference = crc16_bitwise(CRC16_IBM_POLY, 0u16, data);
  results.push(KernelResult {
    name: "reference",
    checksum: reference,
  });

  // Portable slice-by-4
  let portable4 = super::portable::crc16_ibm_slice4(0u16, data);
  results.push(KernelResult {
    name: "portable/slice4",
    checksum: portable4,
  });

  // Portable slice-by-8
  let portable8 = super::portable::crc16_ibm_slice8(0u16, data);
  results.push(KernelResult {
    name: "portable/slice8",
    checksum: portable8,
  });

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    run_x86_64_kernels_ibm(data, &mut results);
  }

  #[cfg(target_arch = "aarch64")]
  {
    run_aarch64_kernels_ibm(data, &mut results);
  }

  #[cfg(target_arch = "powerpc64")]
  {
    run_power_kernels_ibm(data, &mut results);
  }

  #[cfg(target_arch = "s390x")]
  {
    run_s390x_kernels_ibm(data, &mut results);
  }

  #[cfg(target_arch = "riscv64")]
  {
    run_riscv64_kernels_ibm(data, &mut results);
  }

  results
}

/// Verify all CRC-16/CCITT kernels produce the same result.
///
/// Returns `Ok(checksum)` if all agree, or `Err` with details of mismatches.
pub fn verify_crc16_ccitt_kernels(data: &[u8]) -> Result<u16, alloc::string::String> {
  let results = run_all_crc16_ccitt_kernels(data);
  verify_kernel_agreement(&results)
}

/// Verify all CRC-16/IBM kernels produce the same result.
pub fn verify_crc16_ibm_kernels(data: &[u8]) -> Result<u16, alloc::string::String> {
  let results = run_all_crc16_ibm_kernels(data);
  verify_kernel_agreement(&results)
}

fn verify_kernel_agreement(results: &[KernelResult]) -> Result<u16, alloc::string::String> {
  use alloc::{format, string::ToString};

  let first = results.first().ok_or_else(|| "no kernels available".to_string())?;
  let expected = first.checksum;

  for result in results.iter().skip(1) {
    if result.checksum != expected {
      return Err(format!(
        "kernel mismatch: {} produced 0x{:04X}, but {} produced 0x{:04X}",
        first.name, expected, result.name, result.checksum
      ));
    }
  }

  Ok(expected)
}

// ─────────────────────────────────────────────────────────────────────────────
// Architecture-specific kernel runners
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn run_x86_64_kernels_ccitt(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::x86_64::*;
  let caps = platform::caps();

  // PCLMUL kernel
  if caps.has(platform::caps::x86::PCLMUL_READY) {
    for (&name, &func) in PCLMUL_NAMES.iter().zip(CCITT_PCLMUL.iter()) {
      run_single_kernel_ccitt(data, func, name, results);
    }
    run_single_kernel_ccitt(data, CCITT_PCLMUL_SMALL_KERNEL, PCLMUL_SMALL, results);
  }

  // VPCLMUL kernel
  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    for (&name, &func) in VPCLMUL_NAMES.iter().zip(CCITT_VPCLMUL.iter()) {
      run_single_kernel_ccitt(data, func, name, results);
    }
  }
}

#[cfg(target_arch = "x86_64")]
fn run_x86_64_kernels_ibm(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::x86_64::*;
  let caps = platform::caps();

  // PCLMUL kernel
  if caps.has(platform::caps::x86::PCLMUL_READY) {
    for (&name, &func) in PCLMUL_NAMES.iter().zip(IBM_PCLMUL.iter()) {
      run_single_kernel_ibm(data, func, name, results);
    }
    run_single_kernel_ibm(data, IBM_PCLMUL_SMALL_KERNEL, PCLMUL_SMALL, results);
  }

  // VPCLMUL kernel
  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    for (&name, &func) in VPCLMUL_NAMES.iter().zip(IBM_VPCLMUL.iter()) {
      run_single_kernel_ibm(data, func, name, results);
    }
  }
}

#[cfg(target_arch = "aarch64")]
fn run_aarch64_kernels_ccitt(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::aarch64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    for (&name, &func) in PMULL_NAMES.iter().zip(CCITT_PMULL.iter()).take(3) {
      run_single_kernel_ccitt(data, func, name, results);
    }
    run_single_kernel_ccitt(data, CCITT_PMULL_SMALL_KERNEL, PMULL_SMALL, results);
  }
}

#[cfg(target_arch = "aarch64")]
fn run_aarch64_kernels_ibm(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::aarch64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    for (&name, &func) in PMULL_NAMES.iter().zip(IBM_PMULL.iter()).take(3) {
      run_single_kernel_ibm(data, func, name, results);
    }
    run_single_kernel_ibm(data, IBM_PMULL_SMALL_KERNEL, PMULL_SMALL, results);
  }
}

#[cfg(target_arch = "powerpc64")]
fn run_power_kernels_ccitt(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::power::*;
  let caps = platform::caps();

  if caps.has(platform::caps::power::VPMSUM_READY) {
    for (&name, &func) in VPMSUM_NAMES.iter().zip(CCITT_VPMSUM.iter()) {
      run_single_kernel_ccitt(data, func, name, results);
    }
  }
}

#[cfg(target_arch = "powerpc64")]
fn run_power_kernels_ibm(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::power::*;
  let caps = platform::caps();

  if caps.has(platform::caps::power::VPMSUM_READY) {
    for (&name, &func) in VPMSUM_NAMES.iter().zip(IBM_VPMSUM.iter()) {
      run_single_kernel_ibm(data, func, name, results);
    }
  }
}

#[cfg(target_arch = "s390x")]
fn run_s390x_kernels_ccitt(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::s390x::*;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    run_single_kernel_ccitt(data, CCITT_VGFM, VGFM, results);
  }
}

#[cfg(target_arch = "s390x")]
fn run_s390x_kernels_ibm(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::s390x::*;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    run_single_kernel_ibm(data, IBM_VGFM, VGFM, results);
  }
}

#[cfg(target_arch = "riscv64")]
fn run_riscv64_kernels_ccitt(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use platform::caps::riscv;

  use super::kernels::riscv64::*;
  let caps = platform::caps();

  if caps.has(riscv::ZBC) {
    run_single_kernel_ccitt(data, CCITT_ZBC, ZBC, results);
  }

  if caps.has(riscv::ZVBC) {
    run_single_kernel_ccitt(data, CCITT_ZVBC, ZVBC, results);
  }
}

#[cfg(target_arch = "riscv64")]
fn run_riscv64_kernels_ibm(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use platform::caps::riscv;

  use super::kernels::riscv64::*;
  let caps = platform::caps();

  if caps.has(riscv::ZBC) {
    run_single_kernel_ibm(data, IBM_ZBC, ZBC, results);
  }

  if caps.has(riscv::ZVBC) {
    run_single_kernel_ibm(data, IBM_ZVBC, ZVBC, results);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Run a single CCITT kernel (init=0xFFFF, xorout=0xFFFF).
#[allow(dead_code)] // Used on x86_64/aarch64 only
fn run_single_kernel_ccitt(
  data: &[u8],
  kernel: Crc16Fn,
  name: &'static str,
  results: &mut alloc::vec::Vec<KernelResult>,
) {
  let checksum = kernel(!0u16, data) ^ !0u16;
  results.push(KernelResult { name, checksum });
}

/// Run a single IBM kernel (init=0x0000, xorout=0x0000).
#[allow(dead_code)] // Used on x86_64/aarch64 only
fn run_single_kernel_ibm(
  data: &[u8],
  kernel: Crc16Fn,
  name: &'static str,
  results: &mut alloc::vec::Vec<KernelResult>,
) {
  let checksum = kernel(0u16, data);
  results.push(KernelResult { name, checksum });
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_all_ccitt_kernels_agree_empty() {
    verify_crc16_ccitt_kernels(&[]).expect("kernels should agree on empty input");
  }

  #[test]
  fn test_all_ccitt_kernels_agree_small() {
    verify_crc16_ccitt_kernels(b"123456789").expect("kernels should agree on small input");
  }

  #[test]
  fn test_all_ccitt_kernels_agree_medium() {
    let data: alloc::vec::Vec<u8> = (0..1024).map(|i| (i as u8).wrapping_mul(17)).collect();
    verify_crc16_ccitt_kernels(&data).expect("kernels should agree on medium input");
  }

  #[test]
  fn test_all_ccitt_kernels_agree_large() {
    let data: alloc::vec::Vec<u8> = (0..65536).map(|i| (i as u8).wrapping_mul(31)).collect();
    verify_crc16_ccitt_kernels(&data).expect("kernels should agree on large input");
  }

  #[test]
  fn test_all_ibm_kernels_agree_empty() {
    verify_crc16_ibm_kernels(&[]).expect("kernels should agree on empty input");
  }

  #[test]
  fn test_all_ibm_kernels_agree_small() {
    verify_crc16_ibm_kernels(b"123456789").expect("kernels should agree on small input");
  }

  #[test]
  fn test_all_ibm_kernels_agree_medium() {
    let data: alloc::vec::Vec<u8> = (0..1024).map(|i| (i as u8).wrapping_mul(17)).collect();
    verify_crc16_ibm_kernels(&data).expect("kernels should agree on medium input");
  }

  #[test]
  fn test_all_ibm_kernels_agree_large() {
    let data: alloc::vec::Vec<u8> = (0..65536).map(|i| (i as u8).wrapping_mul(31)).collect();
    verify_crc16_ibm_kernels(&data).expect("kernels should agree on large input");
  }
}
