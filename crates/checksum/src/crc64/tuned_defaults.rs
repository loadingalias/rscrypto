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
    xz:   Crc64VariantTunedDefaults { streams: 2, portable_to_clmul: 64, pclmul_to_vpclmul: 128, min_bytes_per_lane: Some(8_192) },
    nvme: Crc64VariantTunedDefaults { streams: 2, portable_to_clmul: 64, pclmul_to_vpclmul: 128, min_bytes_per_lane: None },
  }),
  // Zen5 / Zen5c: extrapolate from Zen4 (same instruction set + wide CLMUL).
  (TuneKind::Zen5, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 2, portable_to_clmul: 64, pclmul_to_vpclmul: 128, min_bytes_per_lane: Some(8_192) },
    nvme: Crc64VariantTunedDefaults { streams: 2, portable_to_clmul: 64, pclmul_to_vpclmul: 128, min_bytes_per_lane: None },
  }),
  (TuneKind::Zen5c, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 2, portable_to_clmul: 64, pclmul_to_vpclmul: 128, min_bytes_per_lane: Some(8_192) },
    nvme: Crc64VariantTunedDefaults { streams: 2, portable_to_clmul: 64, pclmul_to_vpclmul: 128, min_bytes_per_lane: None },
  }),

  // Intel baseline: VPCLMUL has a large warmup cost; delay wide tier.
  // These are conservative placeholders until rscrypto-tune results land.
  (TuneKind::IntelSpr, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 4, portable_to_clmul: 64, pclmul_to_vpclmul: 2_048, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 4, portable_to_clmul: 64, pclmul_to_vpclmul: 2_048, min_bytes_per_lane: None },
  }),
  (TuneKind::IntelGnr, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 7, portable_to_clmul: 64, pclmul_to_vpclmul: 2_048, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 7, portable_to_clmul: 64, pclmul_to_vpclmul: 2_048, min_bytes_per_lane: None },
  }),
  // Ice Lake prefers 256-bit operations; disable VPCLMUL selection by default.
  (TuneKind::IntelIcl, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 4, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 4, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),

  // Apple M1-M3: PMULL is fastest; prefer 3 streams with a modest per-lane minimum.
  (TuneKind::AppleM1M3, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 3, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: Some(2_730) },
    nvme: Crc64VariantTunedDefaults { streams: 3, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: Some(2_730) },
  }),
  // Apple M4/M5: extrapolate from Apple M1-M3.
  (TuneKind::AppleM4, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 3, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: Some(2_730) },
    nvme: Crc64VariantTunedDefaults { streams: 3, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: Some(2_730) },
  }),
  (TuneKind::AppleM5, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 3, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: Some(2_730) },
    nvme: Crc64VariantTunedDefaults { streams: 3, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: Some(2_730) },
  }),

  // Graviton2: PMULL(+EOR3 when available) is fastest; single-stream is best.
  (TuneKind::Graviton2, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  // Graviton/Neoverse class: extrapolate from Graviton2.
  (TuneKind::Graviton3, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::Graviton4, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::Graviton5, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::NeoverseN2, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::NeoverseN3, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::NeoverseV3, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::NvidiaGrace, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::AmpereAltra, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::Aarch64Pmull, Crc64TunedDefaults {
    xz:   Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    nvme: Crc64VariantTunedDefaults { streams: 1, portable_to_clmul: 64, pclmul_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
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

#[cfg(test)]
mod invariants {
  use super::*;

  #[test]
  fn crc64_tuned_defaults_includes_intel_presets() {
    assert!(for_tune_kind(TuneKind::IntelSpr).is_some());
    assert!(for_tune_kind(TuneKind::IntelGnr).is_some());
    assert!(for_tune_kind(TuneKind::IntelIcl).is_some());
  }

  fn assert_variant(v: Crc64VariantTunedDefaults) {
    assert!((1..=16).contains(&v.streams));
    assert!(v.portable_to_clmul >= 1);
    assert!(v.pclmul_to_vpclmul >= v.portable_to_clmul);
    if let Some(m) = v.min_bytes_per_lane {
      assert!(m >= 1);
    }
  }

  #[test]
  fn crc64_tuned_defaults_invariants() {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    for (kind, d) in CRC64_TUNED_DEFAULTS {
      assert!(seen.insert(*kind), "duplicate TuneKind entry: {kind:?}");
      assert_variant(d.xz);
      assert_variant(d.nvme);
    }
  }
}
