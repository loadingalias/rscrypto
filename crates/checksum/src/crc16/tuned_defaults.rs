//! CRC-16 tuned defaults (baked-in).
//!
//! This table is intentionally minimal today; CRC-16 is portable-only and
//! slice-by selection tends to be far less sensitive than SIMD tiers.
//!
//! Env overrides always win.

use platform::TuneKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc16TunedDefaults {
  pub slice4_to_slice8: usize,
  pub portable_to_clmul: usize,
}

#[rustfmt::skip]
pub const CRC16_TUNED_DEFAULTS: &[(TuneKind, Crc16TunedDefaults)] = &[
  // BEGIN GENERATED (rscrypto-tune)
  // Zen4: pclmul is fastest; use 64-byte threshold for CLMUL dispatch.
  (TuneKind::Zen4, Crc16TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64 }),
  // Apple M1-M3: pmull is fastest; use 64-byte threshold for CLMUL dispatch.
  (TuneKind::AppleM1M3, Crc16TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64 }),
  // END GENERATED (rscrypto-tune)
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc16TunedDefaults> {
  CRC16_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}
