//! aarch64 SIMD dispatch for CRC algorithms.
//!
//! Uses the `platform` crate for feature detection to select
//! optimal implementations based on CPU capabilities.

#[cfg(any(
  all(target_feature = "aes", target_feature = "crc"),
  all(feature = "std", not(target_feature = "crc"))
))]
pub(crate) mod pmull;

/// Select the best available CRC32C implementation at runtime (std only).
///
/// The returned function computes the *raw* CRC state (no final XOR).
#[cfg(all(feature = "std", not(target_feature = "crc")))]
pub(crate) fn detect_crc32c_best() -> fn(u32, &[u8]) -> u32 {
  use platform::aarch64::Features;

  let features = Features::detect();

  if features.pmull && features.crc {
    if features.has_pmull_eor3() {
      return pmull::compute_pmull_eor3_runtime;
    }
    return pmull::compute_pmull_runtime;
  }

  if features.has_crc() {
    return crate::crc32c::aarch64::compute_crc_runtime;
  }

  crate::crc32c::portable::compute
}

/// Select the best available CRC32 implementation at runtime (std only).
///
/// The returned function computes the *raw* CRC state (no final XOR).
#[cfg(all(feature = "std", not(target_feature = "crc")))]
pub(crate) fn detect_crc32_best() -> fn(u32, &[u8]) -> u32 {
  use platform::aarch64::Features;

  let features = Features::detect();

  if features.pmull && features.crc {
    if features.has_pmull_eor3() {
      return pmull::compute_pmull_eor3_crc32_runtime;
    }
    return pmull::compute_pmull_crc32_runtime;
  }

  if features.has_crc() {
    return crate::crc32::aarch64::compute_crc_runtime;
  }

  crate::crc32::portable::compute
}

// ============================================================================
// CRC64 Dispatch
// ============================================================================

/// Select the best available CRC64/XZ implementation at runtime (std only).
///
/// The returned function computes the *raw* CRC state (no final XOR).
#[cfg(all(feature = "std", not(target_feature = "aes")))]
pub(crate) fn detect_crc64_xz_best() -> fn(u64, &[u8]) -> u64 {
  use platform::aarch64::Features;

  let features = Features::detect();

  if features.pmull {
    if features.has_pmull_eor3() {
      return pmull::compute_pmull_crc64_xz_eor3_runtime;
    }
    return pmull::compute_pmull_crc64_xz_runtime;
  }

  crate::crc64::xz::compute_portable
}

/// Select the best available CRC64/NVME implementation at runtime (std only).
///
/// The returned function computes the *raw* CRC state (no final XOR).
#[cfg(all(feature = "std", not(target_feature = "aes")))]
pub(crate) fn detect_crc64_nvme_best() -> fn(u64, &[u8]) -> u64 {
  use platform::aarch64::Features;

  let features = Features::detect();

  if features.pmull {
    if features.has_pmull_eor3() {
      return pmull::compute_pmull_crc64_nvme_eor3_runtime;
    }
    return pmull::compute_pmull_crc64_nvme_runtime;
  }

  crate::crc64::nvme::compute_portable
}
