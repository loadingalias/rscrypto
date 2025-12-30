//! Cross-kernel equivalence fuzzing.
//!
//! Verifies that ALL available CRC-64 kernels on the current platform produce
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
  run_all_crc64_nvme_kernels, run_all_crc64_xz_kernels, verify_crc64_nvme_kernels, verify_crc64_xz_kernels,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
  // Test CRC-64-XZ kernels
  test_crc64_xz_kernel_equivalence(data);

  // Test CRC-64-NVME kernels
  test_crc64_nvme_kernel_equivalence(data);
});

fn test_crc64_xz_kernel_equivalence(data: &[u8]) {
  let results = run_all_crc64_xz_kernels(data);

  // All kernels must produce identical results
  if results.len() >= 2 {
    let expected = results[0].checksum;
    for result in &results[1..] {
      assert_eq!(
        result.checksum, expected,
        "CRC64-XZ kernel mismatch: {} produced 0x{:016X}, but {} produced 0x{:016X}, len={}",
        result.name, result.checksum, results[0].name, expected, data.len()
      );
    }
  }

  // Paranoid check: verify against the verification function
  verify_crc64_xz_kernels(data).expect("CRC64-XZ kernel verification failed");
}

fn test_crc64_nvme_kernel_equivalence(data: &[u8]) {
  let results = run_all_crc64_nvme_kernels(data);

  // All kernels must produce identical results
  if results.len() >= 2 {
    let expected = results[0].checksum;
    for result in &results[1..] {
      assert_eq!(
        result.checksum, expected,
        "CRC64-NVME kernel mismatch: {} produced 0x{:016X}, but {} produced 0x{:016X}, len={}",
        result.name, result.checksum, results[0].name, expected, data.len()
      );
    }
  }

  // Paranoid check: verify against the verification function
  verify_crc64_nvme_kernels(data).expect("CRC64-NVME kernel verification failed");
}
