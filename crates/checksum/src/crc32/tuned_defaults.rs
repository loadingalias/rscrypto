//! Pre-computed CRC-32 tuning defaults by microarchitecture.
//!
//! These defaults are derived from benchmarks on various CPUs.
//! Users can override via environment variables.

use platform::TuneKind;

/// CRC-32 tuned defaults for a specific microarchitecture.
#[derive(Clone, Copy, Debug)]
pub struct Crc32TunedDefaults {
  /// Threshold for portable → HW CRC.
  pub portable_to_hw: usize,
  /// Threshold for HW CRC → CLMUL.
  pub hw_to_clmul: usize,
  /// Threshold for PCLMUL → VPCLMUL.
  pub pclmul_to_vpclmul: usize,
  /// Preferred stream count.
  pub streams: u8,
}

/// Get tuned defaults for a specific tune kind.
#[must_use]
pub const fn for_tune_kind(kind: TuneKind) -> Option<Crc32TunedDefaults> {
  match kind {
    // Intel Ice Lake/Sapphire Rapids/Granite Rapids
    TuneKind::IntelIcl | TuneKind::IntelSpr | TuneKind::IntelGnr => Some(Crc32TunedDefaults {
      portable_to_hw: 8,
      hw_to_clmul: 1024, // SSE4.2 crc32 is very fast
      pclmul_to_vpclmul: 8192,
      streams: 4,
    }),

    // AMD Zen 4/5
    TuneKind::Zen4 | TuneKind::Zen5 | TuneKind::Zen5c => Some(Crc32TunedDefaults {
      portable_to_hw: 8,
      hw_to_clmul: 512, // CLMUL catches up faster on Zen
      pclmul_to_vpclmul: 1024,
      streams: 4,
    }),

    // Apple Silicon
    TuneKind::AppleM1M3 | TuneKind::AppleM4 | TuneKind::AppleM5 => Some(Crc32TunedDefaults {
      portable_to_hw: 8,
      hw_to_clmul: 256, // ARM CRC32 + PMULL both very efficient
      pclmul_to_vpclmul: usize::MAX,
      streams: 3,
    }),

    // Ampere Altra
    TuneKind::AmpereAltra => Some(Crc32TunedDefaults {
      portable_to_hw: 8,
      hw_to_clmul: 512,
      pclmul_to_vpclmul: usize::MAX,
      streams: 2,
    }),

    // AWS Graviton
    TuneKind::Graviton2 | TuneKind::Graviton3 | TuneKind::Graviton4 | TuneKind::Graviton5 => Some(Crc32TunedDefaults {
      portable_to_hw: 8,
      hw_to_clmul: 384,
      pclmul_to_vpclmul: usize::MAX,
      streams: 2,
    }),

    // ARM Neoverse
    TuneKind::NeoverseN2 | TuneKind::NeoverseN3 | TuneKind::NeoverseV3 => Some(Crc32TunedDefaults {
      portable_to_hw: 8,
      hw_to_clmul: 384,
      pclmul_to_vpclmul: usize::MAX,
      streams: 2,
    }),

    // No specific tuning
    _ => None,
  }
}
