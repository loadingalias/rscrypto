//! Kernel testing utilities for CRC-32.
//!
//! This module provides functions to run ALL available CRC-32 kernels on the
//! current platform and return their results. Used by fuzz targets and tests
//! to verify cross-kernel equivalence.
//!
//! # Design
//!
//! The oracle is the bitwise reference implementation, which is obviously
//! correct by inspection. All production kernels (portable slice-by-N, SIMD)
//! must produce identical results to the reference for any input.

use crate::dispatchers::Crc32Fn;

/// Result from running a kernel.
#[derive(Debug, Clone, Copy)]
pub struct KernelResult {
  /// Kernel name (e.g., "reference", "portable/slice16", "x86_64/pclmul-8way")
  pub name: &'static str,
  /// Finalized checksum value
  pub checksum: u32,
}

/// Run all available CRC-32 (IEEE) kernels on the given data.
///
/// Returns a vector of (kernel_name, checksum) pairs. All checksums should
/// be identical if the kernels are correct. The first entry is always the
/// bitwise reference implementation.
#[must_use]
pub fn run_all_crc32_ieee_kernels(data: &[u8]) -> alloc::vec::Vec<KernelResult> {
  use alloc::vec::Vec;

  use crate::common::{reference::crc32_bitwise, tables::CRC32_IEEE_POLY};

  let mut results = Vec::new();

  // Oracle: bitwise reference
  let reference = crc32_bitwise(CRC32_IEEE_POLY, !0u32, data) ^ !0u32;
  results.push(KernelResult {
    name: "reference",
    checksum: reference,
  });

  // Portable slice-by-16
  let portable = super::portable::crc32_slice16_ieee(!0u32, data) ^ !0u32;
  results.push(KernelResult {
    name: "portable/slice16",
    checksum: portable,
  });

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    run_x86_64_kernels_ieee(data, &mut results);
  }

  #[cfg(target_arch = "aarch64")]
  {
    run_aarch64_kernels_ieee(data, &mut results);
  }

  #[cfg(target_arch = "powerpc64")]
  {
    run_powerpc64_kernels_ieee(data, &mut results);
  }

  #[cfg(target_arch = "s390x")]
  {
    run_s390x_kernels_ieee(data, &mut results);
  }

  #[cfg(target_arch = "riscv64")]
  {
    run_riscv64_kernels_ieee(data, &mut results);
  }

  results
}

/// Run all available CRC-32C (Castagnoli) kernels on the given data.
#[must_use]
pub fn run_all_crc32c_kernels(data: &[u8]) -> alloc::vec::Vec<KernelResult> {
  use alloc::vec::Vec;

  use crate::common::{reference::crc32_bitwise, tables::CRC32C_POLY};

  let mut results = Vec::new();

  // Oracle: bitwise reference
  let reference = crc32_bitwise(CRC32C_POLY, !0u32, data) ^ !0u32;
  results.push(KernelResult {
    name: "reference",
    checksum: reference,
  });

  // Portable slice-by-16
  let portable = super::portable::crc32c_slice16(!0u32, data) ^ !0u32;
  results.push(KernelResult {
    name: "portable/slice16",
    checksum: portable,
  });

  // Architecture-specific kernels
  #[cfg(target_arch = "x86_64")]
  {
    run_x86_64_kernels_castagnoli(data, &mut results);
  }

  #[cfg(target_arch = "aarch64")]
  {
    run_aarch64_kernels_castagnoli(data, &mut results);
  }

  #[cfg(target_arch = "powerpc64")]
  {
    run_powerpc64_kernels_castagnoli(data, &mut results);
  }

  #[cfg(target_arch = "s390x")]
  {
    run_s390x_kernels_castagnoli(data, &mut results);
  }

  #[cfg(target_arch = "riscv64")]
  {
    run_riscv64_kernels_castagnoli(data, &mut results);
  }

  results
}

/// Verify all CRC-32 (IEEE) kernels produce the same result.
///
/// Returns `Ok(checksum)` if all agree, or `Err` with details of mismatches.
pub fn verify_crc32_ieee_kernels(data: &[u8]) -> Result<u32, alloc::string::String> {
  let results = run_all_crc32_ieee_kernels(data);
  verify_kernel_agreement(&results)
}

/// Verify all CRC-32C (Castagnoli) kernels produce the same result.
pub fn verify_crc32c_kernels(data: &[u8]) -> Result<u32, alloc::string::String> {
  let results = run_all_crc32c_kernels(data);
  verify_kernel_agreement(&results)
}

fn verify_kernel_agreement(results: &[KernelResult]) -> Result<u32, alloc::string::String> {
  use alloc::{format, string::ToString};

  let first = results.first().ok_or_else(|| "no kernels available".to_string())?;
  let expected = first.checksum;

  for result in results.iter().skip(1) {
    if result.checksum != expected {
      return Err(format!(
        "kernel mismatch: {} produced 0x{:08X}, but {} produced 0x{:08X}",
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
fn run_x86_64_kernels_ieee(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::x86_64::*;
  let caps = platform::caps();

  // PCLMUL kernels
  if caps.has(platform::caps::x86::PCLMUL_READY) {
    run_kernel_array(data, &CRC32_PCLMUL, CRC32_PCLMUL_NAMES, results);
    run_single_kernel(data, CRC32_PCLMUL_SMALL_KERNEL, "x86_64/pclmul-small", results);
  }

  // VPCLMUL kernels
  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    run_kernel_array(data, &CRC32_VPCLMUL, CRC32_VPCLMUL_NAMES, results);
  }
}

#[cfg(target_arch = "x86_64")]
fn run_x86_64_kernels_castagnoli(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::x86_64::*;
  let caps = platform::caps();

  // SSE4.2 CRC32 instruction kernels
  if caps.has(platform::caps::x86::CRC32C_READY) {
    run_kernel_array(data, &CRC32C_HWCRC, CRC32C_HWCRC_NAMES, results);
  }

  // Fusion (SSE4.2 + PCLMUL) kernels
  if caps.has(platform::caps::x86::CRC32C_READY) && caps.has(platform::caps::x86::PCLMUL_READY) {
    run_kernel_array(data, &CRC32C_FUSION_SSE, CRC32C_FUSION_SSE_NAMES, results);
  }

  // Fusion (AVX-512 + PCLMUL) kernels
  if caps.has(platform::caps::x86::CRC32C_READY) && caps.has(platform::caps::x86::AVX512_READY) {
    run_kernel_array(data, &CRC32C_FUSION_AVX512, CRC32C_FUSION_AVX512_NAMES, results);
  }

  // Fusion (AVX-512 + VPCLMUL) kernels
  if caps.has(platform::caps::x86::CRC32C_READY) && caps.has(platform::caps::x86::VPCLMUL_READY) {
    run_kernel_array(data, &CRC32C_FUSION_VPCLMUL, CRC32C_FUSION_VPCLMUL_NAMES, results);
  }
}

#[cfg(target_arch = "aarch64")]
fn run_aarch64_kernels_ieee(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::aarch64::*;
  let caps = platform::caps();

  // ARMv8 CRC extension kernels
  if caps.has(platform::caps::aarch64::CRC_READY) {
    run_kernel_array_unique(data, &CRC32_HWCRC, CRC32_HWCRC_NAMES, results);
  }

  // PMULL kernels
  if caps.has(platform::caps::aarch64::PMULL_READY) {
    run_kernel_array_unique(data, &CRC32_PMULL, CRC32_PMULL_NAMES, results);
    run_single_kernel(data, CRC32_PMULL_SMALL_KERNEL, "aarch64/pmull-small", results);
  }

  // PMULL+EOR3 kernels
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
    run_kernel_array_unique(data, &CRC32_PMULL_EOR3, CRC32_PMULL_EOR3_NAMES, results);
  }

  // SVE2 PMULL kernels
  if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
    run_kernel_array_unique(data, &CRC32_SVE2_PMULL, CRC32_SVE2_PMULL_NAMES, results);
  }
}

#[cfg(target_arch = "aarch64")]
fn run_aarch64_kernels_castagnoli(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::aarch64::*;
  let caps = platform::caps();

  // ARMv8 CRC extension kernels
  if caps.has(platform::caps::aarch64::CRC_READY) {
    run_kernel_array_unique(data, &CRC32C_HWCRC, CRC32C_HWCRC_NAMES, results);
  }

  // PMULL kernels
  if caps.has(platform::caps::aarch64::PMULL_READY) {
    run_kernel_array_unique(data, &CRC32C_PMULL, CRC32C_PMULL_NAMES, results);
    run_single_kernel(data, CRC32C_PMULL_SMALL_KERNEL, "aarch64/pmull-small", results);
  }

  // PMULL+EOR3 kernels
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
    run_kernel_array_unique(data, &CRC32C_PMULL_EOR3, CRC32C_PMULL_EOR3_NAMES, results);
  }

  // SVE2 PMULL kernels
  if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
    run_kernel_array_unique(data, &CRC32C_SVE2_PMULL, CRC32C_SVE2_PMULL_NAMES, results);
  }
}

#[cfg(target_arch = "powerpc64")]
fn run_powerpc64_kernels_ieee(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::powerpc64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
    run_kernel_array_unique(data, &CRC32_VPMSUM, CRC32_VPMSUM_NAMES, results);
  }
}

#[cfg(target_arch = "powerpc64")]
fn run_powerpc64_kernels_castagnoli(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::powerpc64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
    run_kernel_array_unique(data, &CRC32C_VPMSUM, CRC32C_VPMSUM_NAMES, results);
  }
}

#[cfg(target_arch = "s390x")]
fn run_s390x_kernels_ieee(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::s390x::*;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    run_kernel_array_unique(data, &CRC32_VGFM, CRC32_VGFM_NAMES, results);
  }
}

#[cfg(target_arch = "s390x")]
fn run_s390x_kernels_castagnoli(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::s390x::*;
  let caps = platform::caps();

  if caps.has(platform::caps::s390x::VECTOR) {
    run_kernel_array_unique(data, &CRC32C_VGFM, CRC32C_VGFM_NAMES, results);
  }
}

#[cfg(target_arch = "riscv64")]
fn run_riscv64_kernels_ieee(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::riscv64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::riscv::ZBC) {
    run_kernel_array_unique(data, &CRC32_ZBC, CRC32_ZBC_NAMES, results);
  }

  if caps.has(platform::caps::riscv::ZVBC) {
    run_kernel_array_unique(data, &CRC32_ZVBC, CRC32_ZVBC_NAMES, results);
  }
}

#[cfg(target_arch = "riscv64")]
fn run_riscv64_kernels_castagnoli(data: &[u8], results: &mut alloc::vec::Vec<KernelResult>) {
  use super::kernels::riscv64::*;
  let caps = platform::caps();

  if caps.has(platform::caps::riscv::ZBC) {
    run_kernel_array_unique(data, &CRC32C_ZBC, CRC32C_ZBC_NAMES, results);
  }

  if caps.has(platform::caps::riscv::ZVBC) {
    run_kernel_array_unique(data, &CRC32C_ZVBC, CRC32C_ZVBC_NAMES, results);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (conditionally used based on architecture)
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)] // Used on x86_64 only
fn run_kernel_array(
  data: &[u8],
  kernels: &[Crc32Fn],
  names: &[&'static str],
  results: &mut alloc::vec::Vec<KernelResult>,
) {
  for (i, kernel) in kernels.iter().enumerate() {
    let checksum = kernel(!0u32, data) ^ !0u32;
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
  kernels: &[Crc32Fn],
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

    let checksum = kernel(!0u32, data) ^ !0u32;
    results.push(KernelResult {
      name: names.get(i).copied().unwrap_or("unknown"),
      checksum,
    });
  }
}

#[allow(dead_code)] // Used on x86_64/aarch64 only
fn run_single_kernel(data: &[u8], kernel: Crc32Fn, name: &'static str, results: &mut alloc::vec::Vec<KernelResult>) {
  let checksum = kernel(!0u32, data) ^ !0u32;
  results.push(KernelResult { name, checksum });
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_all_ieee_kernels_agree_empty() {
    verify_crc32_ieee_kernels(&[]).expect("kernels should agree on empty input");
  }

  #[test]
  fn test_all_ieee_kernels_agree_small() {
    verify_crc32_ieee_kernels(b"123456789").expect("kernels should agree on small input");
  }

  #[test]
  fn test_all_ieee_kernels_agree_medium() {
    let data: alloc::vec::Vec<u8> = (0..1024).map(|i| (i as u8).wrapping_mul(17)).collect();
    verify_crc32_ieee_kernels(&data).expect("kernels should agree on medium input");
  }

  #[test]
  fn test_all_ieee_kernels_agree_large() {
    let data: alloc::vec::Vec<u8> = (0..65536).map(|i| (i as u8).wrapping_mul(31)).collect();
    verify_crc32_ieee_kernels(&data).expect("kernels should agree on large input");
  }

  #[test]
  fn test_all_castagnoli_kernels_agree_empty() {
    verify_crc32c_kernels(&[]).expect("kernels should agree on empty input");
  }

  #[test]
  fn test_all_castagnoli_kernels_agree_small() {
    verify_crc32c_kernels(b"123456789").expect("kernels should agree on small input");
  }

  #[test]
  fn test_all_castagnoli_kernels_agree_medium() {
    let data: alloc::vec::Vec<u8> = (0..1024).map(|i| (i as u8).wrapping_mul(17)).collect();
    verify_crc32c_kernels(&data).expect("kernels should agree on medium input");
  }

  #[test]
  fn test_all_castagnoli_kernels_agree_large() {
    let data: alloc::vec::Vec<u8> = (0..65536).map(|i| (i as u8).wrapping_mul(31)).collect();
    verify_crc32c_kernels(&data).expect("kernels should agree on large input");
  }
}
