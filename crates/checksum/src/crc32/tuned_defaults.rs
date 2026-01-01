//! CRC32 tuned defaults (baked-in).
//!
//! This table lets us "apply" `crc32-tune` results into the repo so users get
//! tuned defaults automatically (while still allowing `RSCRYPTO_CRC32_*` env
//! overrides to win).
//!
//! Update via:
//! - `just tune-crc32 --apply` (writes this file)

use platform::TuneKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32TunedDefaults {
  pub streams_crc32: u8,
  pub streams_crc32c: u8,
  pub portable_to_hwcrc: usize,
  pub hwcrc_to_fusion: usize,
  pub fusion_to_avx512: usize,
  pub fusion_to_vpclmul: usize,
  /// Minimum bytes per lane for CRC-32 (IEEE) multi-stream folding.
  ///
  /// When `None`, uses `KernelFamily::min_bytes_per_lane()` as the default.
  pub min_bytes_per_lane_crc32: Option<usize>,
  /// Minimum bytes per lane for CRC-32C (Castagnoli) multi-stream folding.
  ///
  /// When `None`, uses `KernelFamily::min_bytes_per_lane()` as the default.
  pub min_bytes_per_lane_crc32c: Option<usize>,
}

#[rustfmt::skip]
pub const CRC32_TUNED_DEFAULTS: &[(TuneKind, Crc32TunedDefaults)] = &[
  // BEGIN GENERATED (rscrypto-tune)
  // Zen4: CRC32C is often memory-bound; 1-way avoids the multi-stream cliff.
  (TuneKind::Zen4, Crc32TunedDefaults { streams_crc32: 4, streams_crc32c: 1, portable_to_hwcrc: 64, hwcrc_to_fusion: 64, fusion_to_avx512: 64, fusion_to_vpclmul: 64, min_bytes_per_lane_crc32: None, min_bytes_per_lane_crc32c: None }),
  // Apple M1-M3: pmull-eor3-v9s3x2e-s3 is fastest after 32K threshold.
  (TuneKind::AppleM1M3, Crc32TunedDefaults { streams_crc32: 1, streams_crc32c: 1, portable_to_hwcrc: 64, hwcrc_to_fusion: 32768, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane_crc32: None, min_bytes_per_lane_crc32c: None }),
  // Graviton2: Previous tuning, not refreshed.
  (TuneKind::Graviton2, Crc32TunedDefaults { streams_crc32: 1, streams_crc32c: 1, portable_to_hwcrc: 8, hwcrc_to_fusion: 512, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane_crc32: None, min_bytes_per_lane_crc32c: None }),
  // END GENERATED (rscrypto-tune)
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc32TunedDefaults> {
  CRC32_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}
