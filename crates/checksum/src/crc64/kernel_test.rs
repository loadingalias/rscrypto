//! Kernel testing utilities for CRC-64.
//!
//! This module provides functions to run ALL available CRC-64 kernels on the
//! current platform and return their results. Used by fuzz targets and tests
//! to verify cross-kernel equivalence.
//!
//! # Design
//!
//! The oracle is the bitwise reference implementation, which is obviously
//! correct by inspection. All production kernels (portable slice-by-N, SIMD)
//! must produce identical results to the reference for any input.

use crate::dispatchers::Crc64Fn;

/// Result from running a kernel.
#[derive(Debug, Clone, Copy)]
pub struct KernelResult {
  /// Kernel name (e.g., "reference", "portable/slice16", "x86_64/pclmul-8way")
  pub name: &'static str,
  /// Finalized checksum value
  pub checksum: u64,
}

/// Run all available CRC-64-XZ kernels on the given data.
///
/// Returns a vector of (kernel_name, checksum) pairs. All checksums should
/// be identical if the kernels are correct. The first entry is always the
/// bitwise reference implementation.
#[must_use]
pub fn run_all_crc64_xz_kernels(data: &[u8]) -> alloc::vec::Vec<KernelResult> {
  use alloc::vec::Vec;

  use crate::common::{reference::crc64_bitwise, tables::CRC64_XZ_POLY};

  let mut results = Vec::with_capacity(12);

  // Oracle: bitwise reference
  let reference = crc64_bitwise(CRC64_XZ_POLY, !0u64, data) ^ !0u64;
  results.push(KernelResult {
    name: "reference",
    checksum: reference,
  });

  // Portable slice-by-16
  let portable = super::portable::crc64_slice16_xz(!0u64, data) ^ !0u64;
  results.push(KernelResult {
    name: "portable/slice16",
    checksum: portable,
  });

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    run_x86_64_kernels_xz(data, &mut results);
  }

  #[cfg(target_arch = "aarch64")]
  {
    run_aarch64_kernels_xz(data, &mut results);
  }

  #[cfg(target_arch = "powerpc64")]
  {
    run_power_kernels_xz(data, &mut results);
  }

  #[cfg(target_arch = "s390x")]
  {
    run_s390x_kernels_xz(data, &mut results);
  }

  #[cfg(target_arch = "riscv64")]
  {
    run_riscv64_kernels_xz(data, &mut results);
  }

  results
}

/// Run all available CRC-64-NVME kernels on the given data.
#[must_use]
pub fn run_all_crc64_nvme_kernels(data: &[u8]) -> alloc::vec::Vec<KernelResult> {
  use alloc::vec::Vec;

  use crate::common::{reference::crc64_bitwise, tables::CRC64_NVME_POLY};

  let mut results = Vec::with_capacity(12);

  // Oracle: bitwise reference
  let reference = crc64_bitwise(CRC64_NVME_POLY, !0u64, data) ^ !0u64;
  results.push(KernelResult {
    name: "reference",
    checksum: reference,
  });

  // Portable slice-by-16
  let portable = super::portable::crc64_slice16_nvme(!0u64, data) ^ !0u64;
  results.push(KernelResult {
    name: "portable/slice16",
    checksum: portable,
  });

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    run_x86_64_kernels_nvme(data, &mut results);
  }

  #[cfg(target_arch = "aarch64")]
  {
    run_aarch64_kernels_nvme(data, &mut results);
  }

  #[cfg(target_arch = "powerpc64")]
  {
    run_power_kernels_nvme(data, &mut results);
  }

  #[cfg(target_arch = "s390x")]
  {
    run_s390x_kernels_nvme(data, &mut results);
  }

  #[cfg(target_arch = "riscv64")]
  {
    run_riscv64_kernels_nvme(data, &mut results);
  }

  results
}

/// Verify all kernels produce the same result.
///
/// Returns `Ok(checksum)` if all agree, or `Err` with details of mismatches.
pub fn verify_crc64_xz_kernels(data: &[u8]) -> Result<u64, alloc::string::String> {
  let results = run_all_crc64_xz_kernels(data);
  verify_kernel_agreement(&results)
}

/// Verify all NVME kernels produce the same result.
pub fn verify_crc64_nvme_kernels(data: &[u8]) -> Result<u64, alloc::string::String> {
  let results = run_all_crc64_nvme_kernels(data);
  verify_kernel_agreement(&results)
}

fn verify_kernel_agreement(results: &[KernelResult]) -> Result<u64, alloc::string::String> {
  use alloc::{format, string::ToString};

  let first = results.first().ok_or_else(|| "no kernels available".to_string())?;
  let expected = first.checksum;

  for result in results.iter().skip(1) {
    if result.checksum != expected {
      return Err(format!(
        "kernel mismatch: {} produced 0x{:016X}, but {} produced 0x{:016X}",
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
fn run_x86_64_kernels_xz(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::x86_64::*;
  let caps = platform::caps();

  // PCLMUL kernels
  if caps.has(platform::caps::x86::PCLMUL_READY) {
    run_kernel_array(data, &XZ_PCLMUL, PCLMUL_NAMES, results);
    run_single_kernel(data, XZ_PCLMUL_SMALL, PCLMUL_SMALL, results);
  }

  // VPCLMUL kernels
  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    run_kernel_array(data, &XZ_VPCLMUL, VPCLMUL_NAMES, results);
    run_single_kernel(data, XZ_VPCLMUL_4X512, VPCLMUL_4X512, results);
  }
}

#[cfg(target_arch = "x86_64")]
fn run_x86_64_kernels_nvme(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::x86_64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::x86::PCLMUL_READY) {
    run_kernel_array(data, &NVME_PCLMUL, PCLMUL_NAMES, results);
    run_single_kernel(data, NVME_PCLMUL_SMALL, PCLMUL_SMALL, results);
  }

  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    run_kernel_array(data, &NVME_VPCLMUL, VPCLMUL_NAMES, results);
    run_single_kernel(data, NVME_VPCLMUL_4X512, VPCLMUL_4X512, results);
  }
}

#[cfg(target_arch = "aarch64")]
fn run_aarch64_kernels_xz(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::aarch64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    // Only run unique kernels (array has duplicates for index consistency)
    run_kernel_array_unique(data, &XZ_PMULL, PMULL_NAMES, results);
    run_single_kernel(data, XZ_PMULL_SMALL, PMULL_SMALL, results);
  }

  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
    run_kernel_array_unique(data, &XZ_PMULL_EOR3, PMULL_EOR3_NAMES, results);
  }

  if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
    run_kernel_array_unique(data, &XZ_SVE2_PMULL, SVE2_PMULL_NAMES, results);
    run_single_kernel(data, XZ_SVE2_PMULL_SMALL, SVE2_PMULL_SMALL, results);
  }
}

#[cfg(target_arch = "aarch64")]
fn run_aarch64_kernels_nvme(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::aarch64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    run_kernel_array_unique(data, &NVME_PMULL, PMULL_NAMES, results);
    run_single_kernel(data, NVME_PMULL_SMALL, PMULL_SMALL, results);
  }

  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
    run_kernel_array_unique(data, &NVME_PMULL_EOR3, PMULL_EOR3_NAMES, results);
  }

  if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
    run_kernel_array_unique(data, &NVME_SVE2_PMULL, SVE2_PMULL_NAMES, results);
    run_single_kernel(data, NVME_SVE2_PMULL_SMALL, SVE2_PMULL_SMALL, results);
  }
}

#[cfg(target_arch = "powerpc64")]
fn run_power_kernels_xz(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::power::*;
  let caps = platform::caps();

  if caps.has(platform::caps::power::VPMSUM_READY) {
    run_kernel_array_unique(data, &XZ_VPMSUM, VPMSUM_NAMES, results);
  }
}

#[cfg(target_arch = "powerpc64")]
fn run_power_kernels_nvme(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::power::*;
  let caps = platform::caps();

  if caps.has(platform::caps::power::VPMSUM_READY) {
    run_kernel_array_unique(data, &NVME_VPMSUM, VPMSUM_NAMES, results);
  }
}

#[cfg(target_arch = "s390x")]
fn run_s390x_kernels_xz(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::s390x::*;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    run_kernel_array_unique(data, &XZ_VGFM, VGFM_NAMES, results);
  }
}

#[cfg(target_arch = "s390x")]
fn run_s390x_kernels_nvme(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::s390x::*;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    run_kernel_array_unique(data, &NVME_VGFM, VGFM_NAMES, results);
  }
}

#[cfg(target_arch = "riscv64")]
fn run_riscv64_kernels_xz(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::riscv64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::riscv::ZBC) {
    run_kernel_array_unique(data, &XZ_ZBC, ZBC_NAMES, results);
  }

  if caps.has(platform::caps::riscv::ZVBC) {
    run_kernel_array_unique(data, &XZ_ZVBC, ZVBC_NAMES, results);
  }
}

#[cfg(target_arch = "riscv64")]
fn run_riscv64_kernels_nvme(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::riscv64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::riscv::ZBC) {
    run_kernel_array_unique(data, &NVME_ZBC, ZBC_NAMES, results);
  }

  if caps.has(platform::caps::riscv::ZVBC) {
    run_kernel_array_unique(data, &NVME_ZVBC, ZVBC_NAMES, results);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (conditionally used based on architecture)
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)] // Used on x86_64 only
fn run_kernel_array(
  data: &[u8],
  kernels: &[Crc64Fn],
  names: &[&'static str],
  results: &mut alloc::vec::Vec<KernelResult>,
) {
  for (i, kernel) in kernels.iter().enumerate() {
    let checksum = kernel(!0u64, data) ^ !0u64;
    results.push(KernelResult {
      name: names.get(i).copied().unwrap_or("unknown"),
      checksum,
    });
  }
}

/// Run kernel array but skip duplicate function pointers (arrays pad with dups for index
/// consistency).
#[allow(dead_code)] // Used on aarch64/powerpc64/s390x/riscv64 only
fn run_kernel_array_unique(
  data: &[u8],
  kernels: &[Crc64Fn],
  names: &[&'static str],
  results: &mut alloc::vec::Vec<KernelResult>,
) {
  let mut seen_ptrs = alloc::vec::Vec::new();
  for (i, kernel) in kernels.iter().enumerate() {
    let ptr = *kernel as usize;
    if seen_ptrs.contains(&ptr) {
      continue;
    }
    seen_ptrs.push(ptr);

    let checksum = kernel(!0u64, data) ^ !0u64;
    results.push(KernelResult {
      name: names.get(i).copied().unwrap_or("unknown"),
      checksum,
    });
  }
}

#[allow(dead_code)] // Used on x86_64/aarch64 only
fn run_single_kernel(data: &[u8], kernel: Crc64Fn, name: &'static str, results: &mut alloc::vec::Vec<KernelResult>) {
  let checksum = kernel(!0u64, data) ^ !0u64;
  results.push(KernelResult { name, checksum });
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_all_xz_kernels_agree_empty() {
    verify_crc64_xz_kernels(&[]).expect("kernels should agree on empty input");
  }

  #[test]
  fn test_all_xz_kernels_agree_small() {
    verify_crc64_xz_kernels(b"123456789").expect("kernels should agree on small input");
  }

  #[test]
  fn test_all_xz_kernels_agree_medium() {
    let data: alloc::vec::Vec<u8> = (0..1024).map(|i| (i as u8).wrapping_mul(17)).collect();
    verify_crc64_xz_kernels(&data).expect("kernels should agree on medium input");
  }

  #[test]
  fn test_all_xz_kernels_agree_large() {
    let data: alloc::vec::Vec<u8> = (0..65536).map(|i| (i as u8).wrapping_mul(31)).collect();
    verify_crc64_xz_kernels(&data).expect("kernels should agree on large input");
  }

  #[test]
  fn test_all_nvme_kernels_agree_empty() {
    verify_crc64_nvme_kernels(&[]).expect("kernels should agree on empty input");
  }

  #[test]
  fn test_all_nvme_kernels_agree_small() {
    verify_crc64_nvme_kernels(b"123456789").expect("kernels should agree on small input");
  }

  #[test]
  fn test_all_nvme_kernels_agree_medium() {
    let data: alloc::vec::Vec<u8> = (0..1024).map(|i| (i as u8).wrapping_mul(17)).collect();
    verify_crc64_nvme_kernels(&data).expect("kernels should agree on medium input");
  }

  #[test]
  fn test_all_nvme_kernels_agree_large() {
    let data: alloc::vec::Vec<u8> = (0..65536).map(|i| (i as u8).wrapping_mul(31)).collect();
    verify_crc64_nvme_kernels(&data).expect("kernels should agree on large input");
  }
}
