//! CRC-24 tuned defaults (baked-in).
//!
//! This table is intentionally minimal today; CRC-24 is portable-only and
//! slice-by selection tends to be far less sensitive than SIMD tiers.
//!
//! Env overrides always win.

use platform::TuneKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc24TunedDefaults {
  pub slice4_to_slice8: usize,
}

#[rustfmt::skip]
pub const CRC24_TUNED_DEFAULTS: &[(TuneKind, Crc24TunedDefaults)] = &[
  // Intentionally empty (no tuned data applied yet).
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc24TunedDefaults> {
  CRC24_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}
