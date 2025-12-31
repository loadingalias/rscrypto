//! CRC64 tuned defaults (baked-in).
//!
//! This table lets us "apply" `crc64-tune` results into the repo so users get
//! tuned defaults automatically (while still allowing `RSCRYPTO_CRC64_*` env
//! overrides to win).
//!
//! Update via:
//! - `just tune-crc64 --apply` (writes this file)

use platform::TuneKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc64TunedDefaults {
  pub streams: u8,
  pub portable_to_clmul: usize,
  pub pclmul_to_vpclmul: usize,
  /// Minimum bytes per lane for multi-stream folding.
  ///
  /// When `None`, uses `KernelFamily::min_bytes_per_lane()` as the default.
  pub min_bytes_per_lane: Option<usize>,
}

#[rustfmt::skip]
pub const CRC64_TUNED_DEFAULTS: &[(TuneKind, Crc64TunedDefaults)] = &[
  // BEGIN GENERATED (crc64-tune --apply)
  (TuneKind::AppleM1M3, Crc64TunedDefaults { streams: 3, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None }),
  // END GENERATED (crc64-tune --apply)
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc64TunedDefaults> {
  CRC64_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}
