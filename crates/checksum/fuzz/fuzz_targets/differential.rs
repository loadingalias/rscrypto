//! Differential fuzzing against reference implementations.
//!
//! Compares our CRC implementations against well-established crates
//! to catch any discrepancies.

#![no_main]

use checksum::{Checksum, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp};
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

  // Test CRC16/CCITT (X25) against crc-fast
  test_crc16_ccitt_differential(data);

  // Test CRC16/IBM (ARC) against crc-fast
  test_crc16_ibm_differential(data);

  // Test CRC24/OPENPGP against a bitwise reference
  test_crc24_openpgp_differential(data);
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

fn test_crc16_ccitt_differential(data: &[u8]) {
  let ours = Crc16Ccitt::checksum(data);
  let reference = crc_fast::checksum(crc_fast::CrcAlgorithm::Crc16IbmSdlc, data) as u16;

  assert_eq!(
    ours, reference,
    "CRC16/CCITT differential mismatch: ours={:#06x}, reference={:#06x}, len={}",
    ours, reference, data.len()
  );

  let mut streaming = Crc16Ccitt::new();
  streaming.update(data);
  assert_eq!(streaming.finalize(), ours, "CRC16/CCITT self-consistency mismatch");
}

fn test_crc16_ibm_differential(data: &[u8]) {
  let ours = Crc16Ibm::checksum(data);
  let reference = crc_fast::checksum(crc_fast::CrcAlgorithm::Crc16Arc, data) as u16;

  assert_eq!(
    ours, reference,
    "CRC16/IBM differential mismatch: ours={:#06x}, reference={:#06x}, len={}",
    ours, reference, data.len()
  );

  let mut streaming = Crc16Ibm::new();
  streaming.update(data);
  assert_eq!(streaming.finalize(), ours, "CRC16/IBM self-consistency mismatch");
}

fn crc24_openpgp_reference(data: &[u8]) -> u32 {
  // MSB-first OpenPGP using a 32-bit expanded register (top 24 bits).
  const POLY: u32 = 0x0086_4CFB;
  const INIT: u32 = 0x00B7_04CE;
  let poly_aligned = POLY << 8;

  let mut state: u32 = INIT << 8;
  for &byte in data {
    state ^= (byte as u32) << 24;
    for _ in 0..8 {
      if state & 0x8000_0000 != 0 {
        state = (state << 1) ^ poly_aligned;
      } else {
        state <<= 1;
      }
    }
  }
  (state >> 8) & 0x00FF_FFFF
}

fn test_crc24_openpgp_differential(data: &[u8]) {
  let ours = Crc24OpenPgp::checksum(data);
  let reference = crc24_openpgp_reference(data);

  assert_eq!(
    ours, reference,
    "CRC24/OPENPGP differential mismatch: ours={:#08x}, reference={:#08x}, len={}",
    ours, reference, data.len()
  );

  let mut streaming = Crc24OpenPgp::new();
  streaming.update(data);
  assert_eq!(streaming.finalize(), ours, "CRC24/OPENPGP self-consistency mismatch");
}
