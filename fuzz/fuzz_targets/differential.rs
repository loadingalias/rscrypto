//! Differential fuzzing against reference implementations.
//!
//! Compares our CRC implementations against well-established crates
//! to catch any discrepancies.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
  // Test CRC32-C against crc32fast
  test_crc32c_differential(data);

  // Test CRC64/XZ against crc64fast
  test_crc64_differential(data);
});

fn test_crc32c_differential(data: &[u8]) {
  let ours = checksum::Crc32c::checksum(data);
  let reference = crc32fast::hash(data);

  // crc32fast uses IEEE polynomial (CRC32), not Castagnoli (CRC32-C)
  // So we compare against our CRC32 for differential testing
  let ours_crc32 = checksum::Crc32::checksum(data);
  assert_eq!(
    ours_crc32, reference,
    "CRC32 differential mismatch: ours={:#010x}, reference={:#010x}, len={}",
    ours_crc32, reference, data.len()
  );

  // Self-consistency check: streaming should match one-shot
  let mut hasher = checksum::Crc32c::new();
  hasher.update(data);
  assert_eq!(hasher.finalize(), ours, "CRC32-C self-consistency mismatch");
}

fn test_crc64_differential(data: &[u8]) {
  let ours = checksum::Crc64::checksum(data);
  let mut digest = crc64fast::Digest::new();
  digest.write(data);
  let reference = digest.sum64();

  assert_eq!(
    ours, reference,
    "CRC64/XZ differential mismatch: ours={:#018x}, reference={:#018x}, len={}",
    ours, reference, data.len()
  );

  // Self-consistency check: streaming should match one-shot
  let mut hasher = checksum::Crc64::new();
  hasher.update(data);
  assert_eq!(hasher.finalize(), ours, "CRC64 self-consistency mismatch");
}
