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
