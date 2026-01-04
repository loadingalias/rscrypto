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
  pub pclmul_to_vpclmul: Option<usize>,
  pub streams: u8,
  pub min_bytes_per_lane: Option<usize>,
}

#[rustfmt::skip]
pub const CRC24_TUNED_DEFAULTS: &[(TuneKind, Crc24TunedDefaults)] = &[
  // BEGIN GENERATED (rscrypto-tune)
  // Default: conservative x86_64 tuning (used when microarch is unknown).
  (TuneKind::Default, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 2, min_bytes_per_lane: Some(256) }),

  // Zen4: VPCLMUL is fastest; use it immediately. Multi-stream only pays off at very large sizes.
  (TuneKind::Zen4,  Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: Some(256), streams: 8, min_bytes_per_lane: Some(2_048) }),
  // Zen5 / Zen5c: extrapolate from Zen4 (same instruction set + wide CLMUL).
  (TuneKind::Zen5,  Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: Some(256), streams: 8, min_bytes_per_lane: Some(2_048) }),
  (TuneKind::Zen5c, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: Some(256), streams: 8, min_bytes_per_lane: Some(2_048) }),

  // Intel baseline: VPCLMUL has a large warmup cost; delay wide tier.
  // These are conservative placeholders until rscrypto-tune results land.
  (TuneKind::IntelSpr, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: Some(2_048), streams: 3, min_bytes_per_lane: None }),
  (TuneKind::IntelGnr, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: Some(2_048), streams: 4, min_bytes_per_lane: None }),
  // Ice Lake prefers 256-bit operations; disable VPCLMUL selection by default.
  (TuneKind::IntelIcl, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: Some(usize::MAX), streams: 3, min_bytes_per_lane: None }),

  // Apple M1-M3: PMULL is fastest; 3 streams is best.
  (TuneKind::AppleM1M3, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 3, min_bytes_per_lane: None }),
  // Apple M4/M5: extrapolate from Apple M1-M3.
  (TuneKind::AppleM4,   Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 3, min_bytes_per_lane: None }),
  (TuneKind::AppleM5,   Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 3, min_bytes_per_lane: None }),

  // Graviton/Neoverse class: PMULL is fastest; single-stream is best.
  (TuneKind::Graviton2,  Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 1, min_bytes_per_lane: None }),
  (TuneKind::Graviton3,  Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 1, min_bytes_per_lane: None }),
  (TuneKind::Graviton4,  Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 1, min_bytes_per_lane: None }),
  (TuneKind::Graviton5,  Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 1, min_bytes_per_lane: None }),
  (TuneKind::NeoverseN2, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 1, min_bytes_per_lane: None }),
  (TuneKind::NeoverseN3, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 1, min_bytes_per_lane: None }),
  (TuneKind::NeoverseV3, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 1, min_bytes_per_lane: None }),
  (TuneKind::NvidiaGrace, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 1, min_bytes_per_lane: None }),
  (TuneKind::AmpereAltra, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 1, min_bytes_per_lane: None }),
  (TuneKind::Aarch64Pmull, Crc24TunedDefaults { slice4_to_slice8: 64, portable_to_clmul: 64, pclmul_to_vpclmul: None, streams: 1, min_bytes_per_lane: None }),
  // END GENERATED (rscrypto-tune)
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc24TunedDefaults> {
  CRC24_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}

#[cfg(test)]
mod invariants {
  use super::*;

  #[test]
  fn crc24_tuned_defaults_includes_intel_presets() {
    assert!(for_tune_kind(TuneKind::IntelSpr).is_some());
    assert!(for_tune_kind(TuneKind::IntelGnr).is_some());
    assert!(for_tune_kind(TuneKind::IntelIcl).is_some());
  }

  #[test]
  fn crc24_tuned_defaults_invariants() {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    for (kind, d) in CRC24_TUNED_DEFAULTS {
      assert!(seen.insert(*kind), "duplicate TuneKind entry: {kind:?}");

      assert!(d.slice4_to_slice8 >= 1);
      assert!(d.portable_to_clmul >= 1);
      assert!(d.slice4_to_slice8 <= d.portable_to_clmul);
      assert!((1..=16).contains(&d.streams));

      if let Some(v) = d.pclmul_to_vpclmul {
        assert!(v >= d.portable_to_clmul);
      }
      if let Some(v) = d.min_bytes_per_lane {
        assert!(v >= 1);
      }
    }
  }
}
