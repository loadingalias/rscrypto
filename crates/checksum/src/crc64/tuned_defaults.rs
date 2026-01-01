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
  // BEGIN GENERATED (rscrypto-tune)
  // Zen4: vpclmul-2way is fastest; MIN_BYTES_PER_LANE=524288 for XZ multi-stream.
  (TuneKind::Zen4, Crc64TunedDefaults { streams: 2, portable_to_clmul: 128, pclmul_to_vpclmul: 128, min_bytes_per_lane: Some(524288) }),
  // Apple M1-M3: pmull-3way is fastest.
  (TuneKind::AppleM1M3, Crc64TunedDefaults { streams: 3, portable_to_clmul: 128, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None }),
  // END GENERATED (rscrypto-tune)
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc64TunedDefaults> {
  CRC64_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}
