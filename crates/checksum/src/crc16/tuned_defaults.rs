//! CRC-16 tuned defaults (baked-in).
//!
//! This table is intentionally small today; CRC-16 has carryless-multiply
//! acceleration, but we only bake in a few high-confidence presets.
//!
//! Env overrides always win.

use platform::TuneKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc16TunedDefaults {
  pub slice4_to_slice8: usize,
  pub portable_to_clmul: usize,
  pub streams: u8,
  pub min_bytes_per_lane: Option<usize>,
}

#[rustfmt::skip]
pub const CRC16_CCITT_TUNED_DEFAULTS: &[(TuneKind, Crc16TunedDefaults)] = &[
  // BEGIN GENERATED (rscrypto-tune)
  // Zen4: pclmul is fastest; use 64-byte threshold for CLMUL dispatch.
  (TuneKind::Zen4, Crc16TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, streams: 1, min_bytes_per_lane: None }),
  // Apple M1-M3: pmull is fastest; 3 streams on large buffers.
  (TuneKind::AppleM1M3, Crc16TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, streams: 3, min_bytes_per_lane: Some(2730) }),
  // END GENERATED (rscrypto-tune)
];

#[inline]
#[must_use]
pub fn for_tune_kind_ccitt(kind: TuneKind) -> Option<Crc16TunedDefaults> {
  CRC16_CCITT_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}

#[rustfmt::skip]
pub const CRC16_IBM_TUNED_DEFAULTS: &[(TuneKind, Crc16TunedDefaults)] = &[
  // BEGIN GENERATED (rscrypto-tune)
  // Zen4: pclmul is fastest; use 64-byte threshold for CLMUL dispatch.
  (TuneKind::Zen4, Crc16TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, streams: 1, min_bytes_per_lane: None }),
  // Apple M1-M3: pmull is fastest; 3 streams on large buffers.
  (TuneKind::AppleM1M3, Crc16TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, streams: 3, min_bytes_per_lane: Some(682) }),
  // END GENERATED (rscrypto-tune)
];

#[inline]
#[must_use]
pub fn for_tune_kind_ibm(kind: TuneKind) -> Option<Crc16TunedDefaults> {
  CRC16_IBM_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}
