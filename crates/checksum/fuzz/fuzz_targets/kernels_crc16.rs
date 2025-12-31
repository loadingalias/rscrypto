//! Cross-kernel equivalence fuzzing for CRC-16.
//!
//! Verifies that ALL available CRC-16 kernels on the current platform produce
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
  run_all_crc16_ccitt_kernels, run_all_crc16_ibm_kernels, verify_crc16_ccitt_kernels, verify_crc16_ibm_kernels,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
  // Test CRC-16/CCITT (X25) kernels
  test_crc16_ccitt_kernel_equivalence(data);

  // Test CRC-16/IBM (ARC) kernels
  test_crc16_ibm_kernel_equivalence(data);
});

fn test_crc16_ccitt_kernel_equivalence(data: &[u8]) {
  let results = run_all_crc16_ccitt_kernels(data);

  // All kernels must produce identical results
  if results.len() >= 2 {
    let expected = results[0].checksum;
    for result in &results[1..] {
      assert_eq!(
        result.checksum, expected,
        "CRC16-CCITT kernel mismatch: {} produced 0x{:04X}, but {} produced 0x{:04X}, len={}",
        result.name, result.checksum, results[0].name, expected, data.len()
      );
    }
  }

  // Paranoid check: verify against the verification function
  verify_crc16_ccitt_kernels(data).expect("CRC16-CCITT kernel verification failed");
}

fn test_crc16_ibm_kernel_equivalence(data: &[u8]) {
  let results = run_all_crc16_ibm_kernels(data);

  // All kernels must produce identical results
  if results.len() >= 2 {
    let expected = results[0].checksum;
    for result in &results[1..] {
      assert_eq!(
        result.checksum, expected,
        "CRC16-IBM kernel mismatch: {} produced 0x{:04X}, but {} produced 0x{:04X}, len={}",
        result.name, result.checksum, results[0].name, expected, data.len()
      );
    }
  }

  // Paranoid check: verify against the verification function
  verify_crc16_ibm_kernels(data).expect("CRC16-IBM kernel verification failed");
}
