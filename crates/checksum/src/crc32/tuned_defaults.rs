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
}

#[rustfmt::skip]
pub const CRC32_TUNED_DEFAULTS: &[(TuneKind, Crc32TunedDefaults)] = &[
  // Zen4: CRC32C is often memory-bound; 1-way avoids the multi-stream cliff.
  (TuneKind::Zen4, Crc32TunedDefaults { streams_crc32: 4, streams_crc32c: 1, portable_to_hwcrc: 8, hwcrc_to_fusion: 64, fusion_to_avx512: 64, fusion_to_vpclmul: 64 }),
  // BEGIN GENERATED (crc32-tune --apply)
  (TuneKind::AppleM1M3, Crc32TunedDefaults { streams_crc32: 3, streams_crc32c: 3, portable_to_hwcrc: 16, hwcrc_to_fusion: 512, fusion_to_avx512: 64, fusion_to_vpclmul: usize::MAX }),
  (TuneKind::Graviton2, Crc32TunedDefaults { streams_crc32: 1, streams_crc32c: 1, portable_to_hwcrc: 8, hwcrc_to_fusion: 512, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX }),
  // END GENERATED (crc32-tune --apply)
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc32TunedDefaults> {
  CRC32_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}
