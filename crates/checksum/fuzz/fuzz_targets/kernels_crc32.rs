//! Cross-kernel equivalence fuzzing for CRC-32.
//!
//! Verifies that ALL available CRC-32 kernels on the current platform produce
//! identical results for any input. This catches:
//!
//! - SIMD kernel bugs (boundary conditions, alignment handling, endianness)
//! - Forced backend selection issues
//! - Kernel-specific edge cases (small buffers, unaligned data, etc.)
//!
//! The oracle is the bitwise reference implementation, which is obviously
//! correct by inspection. All production kernels must match it exactly.

#![no_main]

use checksum::__internal::kernel_test::{
  run_all_crc32_ieee_kernels, run_all_crc32c_kernels, verify_crc32_ieee_kernels, verify_crc32c_kernels,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
  // Test CRC-32/IEEE kernels
  test_crc32_ieee_kernel_equivalence(data);

  // Test CRC-32C/Castagnoli kernels
  test_crc32c_kernel_equivalence(data);
});

fn test_crc32_ieee_kernel_equivalence(data: &[u8]) {
  let results = run_all_crc32_ieee_kernels(data);

  // All kernels must produce identical results
  if results.len() >= 2 {
    let expected = results[0].checksum;
    for result in &results[1..] {
      assert_eq!(
        result.checksum, expected,
        "CRC32-IEEE kernel mismatch: {} produced 0x{:08X}, but {} produced 0x{:08X}, len={}",
        result.name, result.checksum, results[0].name, expected, data.len()
      );
    }
  }

  // Paranoid check: verify against the verification function
  verify_crc32_ieee_kernels(data).expect("CRC32-IEEE kernel verification failed");
}

fn test_crc32c_kernel_equivalence(data: &[u8]) {
  let results = run_all_crc32c_kernels(data);

  // All kernels must produce identical results
  if results.len() >= 2 {
    let expected = results[0].checksum;
    for result in &results[1..] {
      assert_eq!(
        result.checksum, expected,
        "CRC32C kernel mismatch: {} produced 0x{:08X}, but {} produced 0x{:08X}, len={}",
        result.name, result.checksum, results[0].name, expected, data.len()
      );
    }
  }

  // Paranoid check: verify against the verification function
  verify_crc32c_kernels(data).expect("CRC32C kernel verification failed");
}
