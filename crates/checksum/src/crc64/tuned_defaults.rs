//! CRC64 tuned defaults (baked-in).
//!
//! This table lets us "apply" `crc64-tune` results into the repo so users get
//! tuned defaults automatically (while still allowing `RSCRYPTO_CRC64_*` env
//! overrides to win).
//!
//! Update via:
//! - `just tune-apply` (writes this file)

use platform::TuneKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc64VariantTunedDefaults {
  pub streams: u8,
  pub portable_to_clmul: usize,
  pub pclmul_to_vpclmul: usize,
  /// Minimum bytes per lane for multi-stream folding.
  ///
  /// When `None`, uses `KernelFamily::min_bytes_per_lane()` as the default.
  pub min_bytes_per_lane: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc64TunedDefaults {
  pub xz: Crc64VariantTunedDefaults,
  pub nvme: Crc64VariantTunedDefaults,
}

#[rustfmt::skip]
pub const CRC64_TUNED_DEFAULTS: &[(TuneKind, Crc64TunedDefaults)] = &[
  // BEGIN GENERATED (rscrypto-tune)
  // Zen4: vpclmul-2way is fastest; per-variant MIN_BYTES_PER_LANE differs.
  (TuneKind::Zen4, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 2, portable_to_clmul: 128, pclmul_to_vpclmul: 128, min_bytes_per_lane: Some(128) },
    nvme: Crc64VariantTunedDefaults { streams: 2, portable_to_clmul: 128, pclmul_to_vpclmul: 128, min_bytes_per_lane: Some(16384) },
  }),
  // Apple M1-M3: pmull-eor3-3way is fastest; XZ benefits from a higher per-lane minimum.
  (TuneKind::AppleM1M3, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 3, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: Some(349525) },
    nvme: Crc64VariantTunedDefaults { streams: 3, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  // Graviton2: pmull-eor3-3way is fastest; XZ benefits from a higher per-lane minimum.
  (TuneKind::Graviton2, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 3, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: Some(349525) },
    nvme: Crc64VariantTunedDefaults { streams: 3, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  // END GENERATED (rscrypto-tune)
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc64TunedDefaults> {
  CRC64_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}
