//! Differential fuzzing against reference implementations.
//!
//! Compares our CRC implementations against well-established crates
//! to catch any discrepancies.

#![no_main]

use checksum::Checksum;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
  // Test CRC64/XZ against crc64fast
  test_crc64_differential(data);

  // Test CRC64/NVME against crc64fast-nvme
  test_crc64_nvme_differential(data);

  // Test CRC32/IEEE against crc32fast
  test_crc32_ieee_differential(data);

  // Test CRC32C against crc32c (oneshot) + crc-fast
  test_crc32c_differential(data);
});

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

fn test_crc64_nvme_differential(data: &[u8]) {
  let ours = checksum::Crc64Nvme::checksum(data);
  let mut digest = crc64fast_nvme::Digest::new();
  digest.write(data);
  let reference = digest.sum64();

  assert_eq!(
    ours, reference,
    "CRC64/NVME differential mismatch: ours={:#018x}, reference={:#018x}, len={}",
    ours, reference, data.len()
  );

  // Self-consistency check: streaming should match one-shot
  let mut hasher = checksum::Crc64Nvme::new();
  hasher.update(data);
  assert_eq!(hasher.finalize(), ours, "CRC64/NVME self-consistency mismatch");
}

fn test_crc32_ieee_differential(data: &[u8]) {
  let ours = checksum::Crc32::checksum(data);

  let mut hasher = crc32fast::Hasher::new();
  hasher.update(data);
  let reference = hasher.finalize();

  assert_eq!(
    ours, reference,
    "CRC32/IEEE differential mismatch: ours={:#010x}, reference={:#010x}, len={}",
    ours, reference, data.len()
  );

  // Self-consistency check: streaming should match one-shot
  let mut streaming = checksum::Crc32::new();
  streaming.update(data);
  assert_eq!(streaming.finalize(), ours, "CRC32 self-consistency mismatch");
}

fn test_crc32c_differential(data: &[u8]) {
  let ours = checksum::Crc32C::checksum(data);

  let reference = crc32c::crc32c(data);
  assert_eq!(
    ours, reference,
    "CRC32C differential mismatch: ours={:#010x}, reference={:#010x}, len={}",
    ours, reference, data.len()
  );

  // Cross-check against crc-fast (independent implementation).
  let reference2 = crc_fast::checksum(crc_fast::CrcAlgorithm::Crc32Iscsi, data) as u32;
  assert_eq!(
    ours, reference2,
    "CRC32C crc-fast mismatch: ours={:#010x}, reference={:#010x}, len={}",
    ours, reference2, data.len()
  );

  // Self-consistency check: streaming should match one-shot
  let mut streaming = checksum::Crc32C::new();
  streaming.update(data);
  assert_eq!(streaming.finalize(), ours, "CRC32C self-consistency mismatch");
}
