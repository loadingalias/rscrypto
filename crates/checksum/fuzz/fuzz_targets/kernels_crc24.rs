//! Cross-kernel equivalence fuzzing for CRC-24.
//!
//! Verifies that ALL available CRC-24 kernels on the current platform produce
//! identical results for any input. This catches:
//!
//! - Portable kernel bugs (slice-by-N boundary conditions)
//! - Future SIMD kernel bugs when added
//!
//! The oracle is the bitwise reference implementation, which is obviously
//! correct by inspection. All production kernels must match it exactly.

#![no_main]

use checksum::__internal::kernel_test::{run_all_crc24_openpgp_kernels, verify_crc24_openpgp_kernels};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
  test_crc24_openpgp_kernel_equivalence(data);
});

fn test_crc24_openpgp_kernel_equivalence(data: &[u8]) {
  let results = run_all_crc24_openpgp_kernels(data);

  // All kernels must produce identical results
  if results.len() >= 2 {
    let expected = results[0].checksum;
    for result in &results[1..] {
      assert_eq!(
        result.checksum, expected,
        "CRC24-OPENPGP kernel mismatch: {} produced 0x{:06X}, but {} produced 0x{:06X}, len={}",
        result.name, result.checksum, results[0].name, expected, data.len()
      );
    }
  }

  // Paranoid check: verify against the verification function
  verify_crc24_openpgp_kernels(data).expect("CRC24-OPENPGP kernel verification failed");
}
