//! CRC-24 tuned defaults (baked-in).
//!
//! This table is intentionally minimal today; CRC-24 has SIMD acceleration but
//! we haven't yet tuned per-microarchitecture thresholds.
//!
//! Env overrides always win.

use platform::TuneKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc24TunedDefaults {
  pub slice4_to_slice8: usize,
  pub portable_to_clmul: usize,
  pub streams: u8,
  pub min_bytes_per_lane: Option<usize>,
}

#[rustfmt::skip]
pub const CRC24_TUNED_DEFAULTS: &[(TuneKind, Crc24TunedDefaults)] = &[
  // BEGIN GENERATED (rscrypto-tune)
  // Apple M1-M3: pmull is fastest; use 64-byte threshold for dispatch.
  (TuneKind::AppleM1M3, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, streams: 3, min_bytes_per_lane: None }),
  // END GENERATED (rscrypto-tune)
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc24TunedDefaults> {
  CRC24_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}
