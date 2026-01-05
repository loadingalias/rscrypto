//! CRC32 tuned defaults (baked-in).
//!
//! This table lets us "apply" `crc32-tune` results into the repo so users get
//! tuned defaults automatically (while still allowing `RSCRYPTO_CRC32_*` env
//! overrides to win).
//!
//! Update via:
//! - `just tune-apply` (writes this file)

use platform::TuneKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32VariantTunedDefaults {
  pub streams: u8,
  /// Bytes where portable bytewise becomes slower than slice-by-16.
  pub portable_bytewise_to_slice16: usize,
  pub portable_to_hwcrc: usize,
  pub hwcrc_to_fusion: usize,
  pub fusion_to_avx512: usize,
  pub fusion_to_vpclmul: usize,
  /// Minimum bytes per lane for multi-stream folding.
  ///
  /// When `None`, uses `KernelFamily::min_bytes_per_lane()` as the default.
  pub min_bytes_per_lane: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32TunedDefaults {
  pub crc32: Crc32VariantTunedDefaults,
  pub crc32c: Crc32VariantTunedDefaults,
}

#[rustfmt::skip]
pub const CRC32_TUNED_DEFAULTS: &[(TuneKind, Crc32TunedDefaults)] = &[
  // BEGIN GENERATED (rscrypto-tune)
  // Default: conservative x86_64 tuning (used when microarch is unknown).
  (TuneKind::Default, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: usize::MAX, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048,     fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),

  // Zen4: VPCLMUL dominates CRC32; VPCLMUL fusion dominates CRC32C after a modest crossover.
  (TuneKind::Zen4, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 4, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: usize::MAX, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: 512, min_bytes_per_lane: Some(64) },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 512,      fusion_to_avx512: usize::MAX, fusion_to_vpclmul: 64,  min_bytes_per_lane: None },
  }),
  // Zen5 / Zen5c: extrapolate from Zen4 (same instruction set + wide CLMUL).
  (TuneKind::Zen5, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 4, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: usize::MAX, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: 512, min_bytes_per_lane: Some(64) },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 512,      fusion_to_avx512: usize::MAX, fusion_to_vpclmul: 64,  min_bytes_per_lane: None },
  }),
  (TuneKind::Zen5c, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 4, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: usize::MAX, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: 512, min_bytes_per_lane: Some(64) },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 512,      fusion_to_avx512: usize::MAX, fusion_to_vpclmul: 64,  min_bytes_per_lane: None },
  }),

  // Intel baseline: wide tiers have a large warmup cost; delay VPCLMUL/AVX-512 fusion.
  // These are conservative placeholders until rscrypto-tune results land.
  (TuneKind::IntelSpr, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 4, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: usize::MAX, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: 2_048, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2_048,     fusion_to_avx512: 2_048,       fusion_to_vpclmul: 2_048, min_bytes_per_lane: None },
  }),
  (TuneKind::IntelGnr, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 7, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: usize::MAX, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: 2_048, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2_048,     fusion_to_avx512: 2_048,       fusion_to_vpclmul: 2_048, min_bytes_per_lane: None },
  }),
  // Ice Lake prefers 256-bit operations; disable wide tiers by default.
  (TuneKind::IntelIcl, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 4, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: usize::MAX, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2_048,     fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),

  // AppleM1M3: generated by rscrypto-tune
  (TuneKind::AppleM1M3, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 512, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 512, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  // Apple M4/M5: extrapolate from Apple M1-M3.
  (TuneKind::AppleM4, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64,  hwcrc_to_fusion: 512, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64,  hwcrc_to_fusion: 512, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::AppleM5, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64,  hwcrc_to_fusion: 512, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64,  hwcrc_to_fusion: 512, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),

  // Graviton2: HWCRC dominates early; switch to PMULL+EOR3 fusion at 2048 bytes.
  (TuneKind::Graviton2, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64,  hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64,  hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  // Graviton/Neoverse class: extrapolate from Graviton2.
  (TuneKind::Graviton3, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::Graviton4, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::Graviton5, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::NeoverseN2, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::NeoverseN3, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::NeoverseV3, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::NvidiaGrace, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::AmpereAltra, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  (TuneKind::Aarch64Pmull, Crc32TunedDefaults {
    crc32:  Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
    crc32c: Crc32VariantTunedDefaults { streams: 1, portable_bytewise_to_slice16: 64, portable_to_hwcrc: 64, hwcrc_to_fusion: 2048, fusion_to_avx512: usize::MAX, fusion_to_vpclmul: usize::MAX, min_bytes_per_lane: None },
  }),
  // END GENERATED (rscrypto-tune)
];

#[inline]
#[must_use]
pub fn for_tune_kind(kind: TuneKind) -> Option<Crc32TunedDefaults> {
  CRC32_TUNED_DEFAULTS
    .iter()
    .find_map(|(k, v)| if *k == kind { Some(*v) } else { None })
}

#[cfg(test)]
mod invariants {
  use super::*;

  #[test]
  fn crc32_tuned_defaults_includes_intel_presets() {
    assert!(for_tune_kind(TuneKind::IntelSpr).is_some());
    assert!(for_tune_kind(TuneKind::IntelGnr).is_some());
    assert!(for_tune_kind(TuneKind::IntelIcl).is_some());
  }

  fn assert_variant(v: Crc32VariantTunedDefaults) {
    assert!((1..=16).contains(&v.streams));
    assert!(v.portable_bytewise_to_slice16 >= 1);
    assert!(v.portable_to_hwcrc >= 1);
    assert!(v.hwcrc_to_fusion >= v.portable_to_hwcrc);
    assert!(v.fusion_to_vpclmul >= 1);
    if v.fusion_to_avx512 != usize::MAX {
      assert!(v.fusion_to_avx512 >= v.fusion_to_vpclmul);
    }
    if let Some(m) = v.min_bytes_per_lane {
      assert!(m >= 1);
    }
  }

  #[test]
  fn crc32_tuned_defaults_invariants() {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    for (kind, d) in CRC32_TUNED_DEFAULTS {
      assert!(seen.insert(*kind), "duplicate TuneKind entry: {kind:?}");
      assert_variant(d.crc32);
      assert_variant(d.crc32c);
    }
  }
}
