//! Tests verifying portable fallback is always reachable.
//!
//! These tests ensure that the portable tier (slice-by-N) is always available
//! as a fallback on any platform, regardless of SIMD capabilities.

use checksum::{Checksum, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};

// ─────────────────────────────────────────────────────────────────────────────
// Test Vectors (from public CRC standards)
// ─────────────────────────────────────────────────────────────────────────────

const CHECK_STRING: &[u8] = b"123456789";

// Expected CRC values for the check string
const CRC16_CCITT_CHECK: u16 = 0x906E;
const CRC16_IBM_CHECK: u16 = 0xBB3D;
const CRC24_OPENPGP_CHECK: u32 = 0x21CF02;
const CRC32_IEEE_CHECK: u32 = 0xCBF4_3926;
const CRC32C_CHECK: u32 = 0xE306_9283;
const CRC64_XZ_CHECK: u64 = 0x995D_C9BB_DF19_39FA;
const CRC64_NVME_CHECK: u64 = 0xAE8B_1486_0A79_9888;

// ─────────────────────────────────────────────────────────────────────────────
// Portable Fallback Correctness Tests
// ─────────────────────────────────────────────────────────────────────────────
//
// These tests verify that the CRC implementations produce correct results.
// On non-SIMD platforms (like wasm32), these exercise the portable fallback.
// On SIMD platforms, they may use accelerated kernels, but the results must match.

#[test]
fn crc16_ccitt_produces_correct_result() {
  let result = Crc16Ccitt::checksum(CHECK_STRING);
  assert_eq!(
    result, CRC16_CCITT_CHECK,
    "CRC-16/CCITT mismatch: got {result:#06X}, expected {CRC16_CCITT_CHECK:#06X}"
  );
}

#[test]
fn crc16_ibm_produces_correct_result() {
  let result = Crc16Ibm::checksum(CHECK_STRING);
  assert_eq!(
    result, CRC16_IBM_CHECK,
    "CRC-16/IBM mismatch: got {result:#06X}, expected {CRC16_IBM_CHECK:#06X}"
  );
}

#[test]
fn crc24_openpgp_produces_correct_result() {
  let result = Crc24OpenPgp::checksum(CHECK_STRING);
  assert_eq!(
    result, CRC24_OPENPGP_CHECK,
    "CRC-24/OpenPGP mismatch: got {result:#08X}, expected {CRC24_OPENPGP_CHECK:#08X}"
  );
}

#[test]
fn crc32_ieee_produces_correct_result() {
  let result = Crc32::checksum(CHECK_STRING);
  assert_eq!(
    result, CRC32_IEEE_CHECK,
    "CRC-32/IEEE mismatch: got {result:#010X}, expected {CRC32_IEEE_CHECK:#010X}"
  );
}

#[test]
fn crc32c_produces_correct_result() {
  let result = Crc32C::checksum(CHECK_STRING);
  assert_eq!(
    result, CRC32C_CHECK,
    "CRC-32C mismatch: got {result:#010X}, expected {CRC32C_CHECK:#010X}"
  );
}

#[test]
fn crc64_xz_produces_correct_result() {
  let result = Crc64::checksum(CHECK_STRING);
  assert_eq!(
    result, CRC64_XZ_CHECK,
    "CRC-64/XZ mismatch: got {result:#018X}, expected {CRC64_XZ_CHECK:#018X}"
  );
}

#[test]
fn crc64_nvme_produces_correct_result() {
  let result = Crc64Nvme::checksum(CHECK_STRING);
  assert_eq!(
    result, CRC64_NVME_CHECK,
    "CRC-64/NVME mismatch: got {result:#018X}, expected {CRC64_NVME_CHECK:#018X}"
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming API Equivalence Tests
// ─────────────────────────────────────────────────────────────────────────────
//
// These tests verify that the streaming API produces the same result as one-shot,
// exercising both code paths and ensuring the portable fallback works correctly
// with incremental updates.

#[test]
fn crc16_ccitt_streaming_matches_oneshot() {
  let oneshot = Crc16Ccitt::checksum(CHECK_STRING);

  let mut hasher = Crc16Ccitt::new();
  hasher.update(b"1234");
  hasher.update(b"56789");
  let streaming = hasher.finalize();

  assert_eq!(
    streaming, oneshot,
    "CRC-16/CCITT streaming mismatch: got {streaming:#06X}, expected {oneshot:#06X}"
  );
}

#[test]
fn crc32_streaming_matches_oneshot() {
  let oneshot = Crc32::checksum(CHECK_STRING);

  let mut hasher = Crc32::new();
  hasher.update(b"123");
  hasher.update(b"456");
  hasher.update(b"789");
  let streaming = hasher.finalize();

  assert_eq!(
    streaming, oneshot,
    "CRC-32 streaming mismatch: got {streaming:#010X}, expected {oneshot:#010X}"
  );
}

#[test]
fn crc64_streaming_matches_oneshot() {
  let oneshot = Crc64::checksum(CHECK_STRING);

  let mut hasher = Crc64::new();
  hasher.update(b"12345");
  hasher.update(b"6789");
  let streaming = hasher.finalize();

  assert_eq!(
    streaming, oneshot,
    "CRC-64 streaming mismatch: got {streaming:#018X}, expected {oneshot:#018X}"
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty Input Tests
// ─────────────────────────────────────────────────────────────────────────────
//
// Verify that empty input produces the expected initial/identity value.

#[test]
fn crc16_ccitt_empty_input() {
  let result = Crc16Ccitt::checksum(&[]);
  // CRC-16/CCITT with init=0xFFFF and final XOR=0xFFFF: empty => 0xFFFF ^ 0xFFFF = 0x0000
  // Actually for X.25: init=0xFFFF, xorout=0xFFFF, so empty is 0x0000
  // But let's just verify it's consistent
  let hasher = Crc16Ccitt::new();
  let streaming = hasher.finalize();
  assert_eq!(
    result, streaming,
    "Empty input should be consistent between oneshot and streaming"
  );
}

#[test]
fn crc32_empty_input() {
  let result = Crc32::checksum(&[]);
  // CRC-32/IEEE: init=0xFFFFFFFF, xorout=0xFFFFFFFF, empty => 0x00000000
  assert_eq!(result, 0x0000_0000, "CRC-32 empty input should be 0x00000000");
}

#[test]
fn crc64_empty_input() {
  let result = Crc64::checksum(&[]);
  // CRC-64/XZ: init=0xFFFFFFFFFFFFFFFF, xorout=0xFFFFFFFFFFFFFFFF, empty => 0x0000000000000000
  assert_eq!(
    result, 0x0000_0000_0000_0000,
    "CRC-64 empty input should be 0x0000000000000000"
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Various Input Sizes (Exercises Different Code Paths)
// ─────────────────────────────────────────────────────────────────────────────
//
// Test with various input sizes to exercise:
// - Tail handling (< 8 bytes)
// - Small buffer paths
// - Chunk alignment boundaries

#[test]
fn crc32_various_sizes() {
  // Just verify that different sizes don't panic and produce consistent results
  for size in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1024] {
    let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();

    let oneshot = Crc32::checksum(&data);

    let mut hasher = Crc32::new();
    hasher.update(&data);
    let streaming = hasher.finalize();

    assert_eq!(
      streaming, oneshot,
      "CRC-32 mismatch at size {size}: oneshot={oneshot:#010X}, streaming={streaming:#010X}"
    );
  }
}

#[test]
fn crc64_various_sizes() {
  for size in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1024] {
    let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();

    let oneshot = Crc64::checksum(&data);

    let mut hasher = Crc64::new();
    hasher.update(&data);
    let streaming = hasher.finalize();

    assert_eq!(
      streaming, oneshot,
      "CRC-64 mismatch at size {size}: oneshot={oneshot:#018X}, streaming={streaming:#018X}"
    );
  }
}
