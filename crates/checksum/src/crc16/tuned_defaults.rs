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
}

#[rustfmt::skip]
pub const CRC16_TUNED_DEFAULTS: &[(TuneKind, Crc16TunedDefaults)] = &[
  // Intentionally empty (no tuned data applied yet).
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc16TunedDefaults> {
  CRC16_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}
