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
pub struct Crc32VariantTunedDefaults {
  pub streams: u8,
  pub portable_to_hwcrc: usize,
  pub hwcrc_to_fusion: usize,
  pub fusion_to_avx512: usize,
  pub fusion_to_vpclmul: usize,
  /// Minimum bytes per lane for multi-stream folding.
  ///
  /// When `None`, uses `KernelFamily::min_bytes_per_lane()` as the default.
  pub min_bytes_per_lane: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32TunedDefaults {
  pub crc32: Crc32VariantTunedDefaults,
  pub crc32c: Crc32VariantTunedDefaults,
}

#[rustfmt::skip]
pub const CRC32_TUNED_DEFAULTS: &[(TuneKind, Crc32TunedDefaults)] = &[
  // BEGIN GENERATED (rscrypto-tune)
  // Zen4: VPCLMUL dominates CRC32, VPCLMUL fusion dominates CRC32C.
  (TuneKind::Zen4, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 7, portable_to_hwcrc: 128, hwcrc_to_fusion: 128, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: 128, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_to_hwcrc:  64, hwcrc_to_fusion:  64, fusion_to_avx512:  64, fusion_to_vpclmul:  64, min_bytes_per_lane: None },
  }),
  // Apple M1-M3: PMULL dominates early; EOR3 wins after moderate sizes.
  (TuneKind::AppleM1M3, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_to_hwcrc: 64, hwcrc_to_fusion: 256,  fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_to_hwcrc: 64, hwcrc_to_fusion: 1024, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  // Graviton2: pmull-eor3-v9s3x2e-s3 is the fastest tier; use it immediately.
  (TuneKind::Graviton2, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 2, portable_to_hwcrc: 64, hwcrc_to_fusion: 64, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 2, portable_to_hwcrc: 64, hwcrc_to_fusion: 64, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  // END GENERATED (rscrypto-tune)
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc32TunedDefaults> {
  CRC32_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}
