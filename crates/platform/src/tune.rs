//! Microarchitecture-derived tuning hints.
//!
//! `Tune` answers: "What should I *prefer* on this machine?"
//!
//! Unlike `CpuCaps` (which describes what's *possible*), `Tune` describes
//! what's *optimal*. This includes:
//!
//! - SIMD threshold (minimum buffer size for SIMD to be worthwhile)
//! - Strategy preferences (hybrid vs pure folding, etc.)
//! - Microarch-specific knobs
//!
//! # Design
//!
//! Tuning hints are derived from:
//! 1. Detected microarchitecture (x86_64: family/model, aarch64: feature combo)
//! 2. Known performance characteristics from benchmarks
//!
//! # Usage
//!
//! ```ignore
//! let tune = platform::tune();
//!
//! if data.len() < tune.simd_threshold {
//!     // Use scalar/small-buffer handler
//! } else {
//!     // Use SIMD kernel
//! }
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// TuneKind: Identity discriminant for O(1) name lookup
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies which microarchitecture tuning preset is in use.
///
/// This discriminant enables O(1) `name()` lookup and disambiguates
/// presets that happen to have identical tuning values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TuneKind {
  Custom = 0,
  Default,
  Portable,
  // x86_64
  Zen4,
  Zen5,
  IntelSpr,
  IntelIcl,
  // Apple Silicon
  AppleM1M3,
  AppleM4,
  // AWS Graviton
  Graviton2,
  Graviton3,
  Graviton4,
  // ARM Neoverse
  NeoverseN2,
  Aarch64Pmull,
}

impl TuneKind {
  /// Returns the human-readable name for this tuning preset.
  #[must_use]
  pub const fn name(self) -> &'static str {
    match self {
      Self::Custom => "Custom",
      Self::Default => "Default",
      Self::Portable => "Portable",
      Self::Zen4 => "Zen4",
      Self::Zen5 => "Zen5",
      Self::IntelSpr => "Intel SPR",
      Self::IntelIcl => "Intel ICL",
      Self::AppleM1M3 => "Apple M1-M3",
      Self::AppleM4 => "Apple M4",
      Self::Graviton2 => "Graviton 2",
      Self::Graviton3 => "Graviton 3",
      Self::Graviton4 => "Graviton 4",
      Self::NeoverseN2 => "Neoverse N2",
      Self::Aarch64Pmull => "AArch64 PMULL",
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tune: Microarchitecture tuning hints
// ─────────────────────────────────────────────────────────────────────────────

/// Microarchitecture-derived tuning hints.
///
/// These hints guide algorithm selection and threshold decisions.
/// They are derived from the detected CPU microarchitecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tune {
  /// Which preset this tuning came from (enables O(1) name lookup).
  pub(crate) kind: TuneKind,
  /// Minimum buffer size (bytes) where SIMD becomes faster than scalar.
  ///
  /// Below this threshold, scalar code or hardware CRC instructions
  /// (when available) are often faster due to SIMD setup overhead.
  ///
  /// Typical values:
  /// - AMD Zen 4/5: 64 bytes (very low ZMM warmup)
  /// - Intel SPR/ICL: 256 bytes (higher ZMM warmup latency)
  /// - Apple M-series: 64-256 bytes (depends on path)
  pub simd_threshold: usize,

  /// Whether to prefer hybrid scalar+SIMD strategies for CRC.
  ///
  /// AMD Zen 4/5 benefit from interleaving scalar CRC32 instructions
  /// with SIMD folding due to their excellent CRC32 instruction throughput
  /// and low ZMM warmup latency.
  pub prefer_hybrid_crc: bool,

  /// Number of parallel CRC32 instruction streams the CPU can sustain.
  ///
  /// - Zen 5: 7-way parallel CRC32Q
  /// - Zen 4/Intel: 3-way parallel CRC32Q
  pub crc_parallelism: u8,

  /// Whether the CPU has fast ZMM register warmup.
  ///
  /// Intel CPUs have ~2000ns warmup when first using ZMM registers.
  /// AMD Zen 4/5 have ~60ns warmup.
  pub fast_zmm: bool,

  /// SVE vector length in bits (0 if no SVE).
  ///
  /// Common values:
  /// - 0: No SVE support
  /// - 128: Graviton 4, Neoverse N2 (smaller SVE for more cores)
  /// - 256: Graviton 3, Neoverse V1
  /// - 512: Some HPC implementations
  pub sve_vlen: u16,

  /// Whether SME (Scalable Matrix Extension) is available.
  ///
  /// Apple M4 has SME with Streaming SVE mode but not full SVE.
  pub has_sme: bool,
}

impl Tune {
  /// Conservative defaults for unknown CPUs.
  pub const DEFAULT: Self = Self {
    kind: TuneKind::Default,
    simd_threshold: 256,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
    sve_vlen: 0,
    has_sme: false,
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // x86_64 Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for AMD Zen 4.
  pub const ZEN4: Self = Self {
    kind: TuneKind::Zen4,
    simd_threshold: 64,
    prefer_hybrid_crc: true,
    crc_parallelism: 3,
    fast_zmm: true,
    sve_vlen: 0,
    has_sme: false,
  };

  /// Tuning for AMD Zen 5.
  pub const ZEN5: Self = Self {
    kind: TuneKind::Zen5,
    simd_threshold: 64,
    prefer_hybrid_crc: true,
    crc_parallelism: 7,
    fast_zmm: true,
    sve_vlen: 0,
    has_sme: false,
  };

  /// Tuning for Intel Sapphire Rapids / Emerald Rapids / Granite Rapids.
  pub const INTEL_SPR: Self = Self {
    kind: TuneKind::IntelSpr,
    simd_threshold: 256,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
    sve_vlen: 0,
    has_sme: false,
  };

  /// Tuning for Intel Ice Lake.
  pub const INTEL_ICL: Self = Self {
    kind: TuneKind::IntelIcl,
    simd_threshold: 256,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
    sve_vlen: 0,
    has_sme: false,
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // Apple Silicon Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for Apple M1/M2/M3 (PMULL+EOR3, no SVE, no SME).
  pub const APPLE_M1_M3: Self = Self {
    kind: TuneKind::AppleM1M3,
    simd_threshold: 64,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
    sve_vlen: 0,
    has_sme: false,
  };

  /// Tuning for Apple M4 (PMULL+EOR3, SME with Streaming SVE, no full SVE).
  pub const APPLE_M4: Self = Self {
    kind: TuneKind::AppleM4,
    simd_threshold: 64,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
    sve_vlen: 0,
    has_sme: true,
  };

  /// Alias for backwards compatibility: Apple M-series (M1-M3).
  pub const APPLE_M: Self = Self::APPLE_M1_M3;

  // ─────────────────────────────────────────────────────────────────────────────
  // AWS Graviton Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for AWS Graviton 2 (Neoverse N1, no SVE).
  pub const GRAVITON2: Self = Self {
    kind: TuneKind::Graviton2,
    simd_threshold: 64,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
    sve_vlen: 0,
    has_sme: false,
  };

  /// Tuning for AWS Graviton 3 (Neoverse V1, 256-bit SVE).
  pub const GRAVITON3: Self = Self {
    kind: TuneKind::Graviton3,
    simd_threshold: 64,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
    sve_vlen: 256,
    has_sme: false,
  };

  /// Tuning for AWS Graviton 4 (Neoverse V2, 128-bit SVE for more cores).
  pub const GRAVITON4: Self = Self {
    kind: TuneKind::Graviton4,
    simd_threshold: 64,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
    sve_vlen: 128,
    has_sme: false,
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // ARM Neoverse Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for ARM Neoverse N2 (128-bit SVE).
  pub const NEOVERSE_N2: Self = Self {
    kind: TuneKind::NeoverseN2,
    simd_threshold: 64,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
    sve_vlen: 128,
    has_sme: false,
  };

  /// Tuning for generic aarch64 with PMULL (no SVE).
  pub const AARCH64_PMULL: Self = Self {
    kind: TuneKind::Aarch64Pmull,
    simd_threshold: 256,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
    sve_vlen: 0,
    has_sme: false,
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // Fallback Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for portable/scalar code.
  pub const PORTABLE: Self = Self {
    kind: TuneKind::Portable,
    simd_threshold: 64,
    prefer_hybrid_crc: false,
    crc_parallelism: 1,
    fast_zmm: false,
    sve_vlen: 0,
    has_sme: false,
  };
}

impl Default for Tune {
  #[inline]
  fn default() -> Self {
    Self::DEFAULT
  }
}

impl core::fmt::Display for Tune {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "Tune(threshold={}, hybrid={}, parallelism={}, fast_zmm={}, sve_vlen={}, sme={})",
      self.simd_threshold, self.prefer_hybrid_crc, self.crc_parallelism, self.fast_zmm, self.sve_vlen, self.has_sme
    )
  }
}

impl Tune {
  /// Returns the kind discriminant for this tuning configuration.
  #[inline]
  #[must_use]
  pub const fn kind(&self) -> TuneKind {
    self.kind
  }

  /// Returns a descriptive name for this tuning configuration.
  ///
  /// O(1) lookup via the stored `TuneKind` discriminant.
  #[inline]
  #[must_use]
  pub const fn name(&self) -> &'static str {
    self.kind.name()
  }

  /// Create a custom tuning configuration.
  ///
  /// Use this when the built-in presets don't match your needs.
  #[must_use]
  pub const fn custom(
    simd_threshold: usize,
    prefer_hybrid_crc: bool,
    crc_parallelism: u8,
    fast_zmm: bool,
    sve_vlen: u16,
    has_sme: bool,
  ) -> Self {
    Self {
      kind: TuneKind::Custom,
      simd_threshold,
      prefer_hybrid_crc,
      crc_parallelism,
      fast_zmm,
      sve_vlen,
      has_sme,
    }
  }
}

extern crate alloc;

#[cfg(test)]
mod tests {
  use super::*;

  // ─────────────────────────────────────────────────────────────────────────────
  // Basic Tune Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_tune_defaults() {
    let tune = Tune::default();
    assert_eq!(tune.simd_threshold, 256);
    assert!(!tune.prefer_hybrid_crc);
    assert_eq!(tune.sve_vlen, 0);
    assert!(!tune.has_sme);
    assert_eq!(tune, Tune::DEFAULT);
  }

  #[test]
  fn test_tune_zen5() {
    let tune = Tune::ZEN5;
    assert_eq!(tune.simd_threshold, 64);
    assert!(tune.prefer_hybrid_crc);
    assert_eq!(tune.crc_parallelism, 7);
    assert!(tune.fast_zmm);
    assert_eq!(tune.sve_vlen, 0);
    assert!(!tune.has_sme);
  }

  #[test]
  fn test_tune_zen4() {
    let tune = Tune::ZEN4;
    assert_eq!(tune.simd_threshold, 64);
    assert!(tune.prefer_hybrid_crc);
    assert_eq!(tune.crc_parallelism, 3);
    assert!(tune.fast_zmm);
    assert_eq!(tune.sve_vlen, 0);
    assert!(!tune.has_sme);
  }

  #[test]
  fn test_tune_intel_spr() {
    let tune = Tune::INTEL_SPR;
    assert_eq!(tune.simd_threshold, 256);
    assert!(!tune.prefer_hybrid_crc);
    assert!(!tune.fast_zmm);
    assert_eq!(tune.sve_vlen, 0);
  }

  #[test]
  fn test_tune_intel_icl() {
    let tune = Tune::INTEL_ICL;
    assert_eq!(tune.simd_threshold, 256);
    assert!(!tune.prefer_hybrid_crc);
    assert_eq!(tune.crc_parallelism, 3);
    assert!(!tune.fast_zmm);
  }

  #[test]
  fn test_tune_apple_m1_m3() {
    let tune = Tune::APPLE_M1_M3;
    assert_eq!(tune.simd_threshold, 64);
    assert!(!tune.prefer_hybrid_crc);
    assert!(!tune.fast_zmm);
    assert_eq!(tune.sve_vlen, 0);
    assert!(!tune.has_sme);
    // APPLE_M is an alias
    assert_eq!(Tune::APPLE_M, Tune::APPLE_M1_M3);
  }

  #[test]
  fn test_tune_apple_m4() {
    let tune = Tune::APPLE_M4;
    assert_eq!(tune.simd_threshold, 64);
    assert!(!tune.prefer_hybrid_crc);
    assert!(!tune.fast_zmm);
    assert_eq!(tune.sve_vlen, 0); // M4 has SME, not full SVE
    assert!(tune.has_sme);
  }

  #[test]
  fn test_tune_graviton2() {
    let tune = Tune::GRAVITON2;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.sve_vlen, 0); // Neoverse N1 has no SVE
    assert!(!tune.has_sme);
  }

  #[test]
  fn test_tune_graviton3() {
    let tune = Tune::GRAVITON3;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.sve_vlen, 256); // Neoverse V1 has 256-bit SVE
    assert!(!tune.has_sme);
  }

  #[test]
  fn test_tune_graviton4() {
    let tune = Tune::GRAVITON4;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.sve_vlen, 128); // Neoverse V2 reduced to 128-bit
    assert!(!tune.has_sme);
  }

  #[test]
  fn test_tune_neoverse_n2() {
    let tune = Tune::NEOVERSE_N2;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.sve_vlen, 128);
    assert!(!tune.has_sme);
  }

  #[test]
  fn test_tune_aarch64_pmull() {
    let tune = Tune::AARCH64_PMULL;
    assert_eq!(tune.simd_threshold, 256);
    assert!(!tune.prefer_hybrid_crc);
    assert_eq!(tune.sve_vlen, 0);
  }

  #[test]
  fn test_tune_portable() {
    let tune = Tune::PORTABLE;
    assert_eq!(tune.simd_threshold, 64);
    assert!(!tune.prefer_hybrid_crc);
    assert_eq!(tune.crc_parallelism, 1);
    assert!(!tune.fast_zmm);
    assert_eq!(tune.sve_vlen, 0);
    assert!(!tune.has_sme);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Tune Name Identification (O(1) via TuneKind discriminant)
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_tune_name_presets() {
    // Each preset now returns its distinct name via TuneKind discriminant
    assert_eq!(Tune::ZEN5.name(), "Zen5");
    assert_eq!(Tune::ZEN4.name(), "Zen4");
    assert_eq!(Tune::INTEL_SPR.name(), "Intel SPR");
    assert_eq!(Tune::INTEL_ICL.name(), "Intel ICL");
    assert_eq!(Tune::APPLE_M1_M3.name(), "Apple M1-M3");
    assert_eq!(Tune::APPLE_M4.name(), "Apple M4");
    assert_eq!(Tune::GRAVITON2.name(), "Graviton 2");
    assert_eq!(Tune::GRAVITON3.name(), "Graviton 3");
    assert_eq!(Tune::GRAVITON4.name(), "Graviton 4");
    assert_eq!(Tune::NEOVERSE_N2.name(), "Neoverse N2");
    assert_eq!(Tune::AARCH64_PMULL.name(), "AArch64 PMULL");
    assert_eq!(Tune::PORTABLE.name(), "Portable");
    assert_eq!(Tune::DEFAULT.name(), "Default");
  }

  #[test]
  fn test_tune_kind_accessor() {
    assert_eq!(Tune::ZEN5.kind(), TuneKind::Zen5);
    assert_eq!(Tune::INTEL_SPR.kind(), TuneKind::IntelSpr);
    assert_eq!(Tune::APPLE_M4.kind(), TuneKind::AppleM4);
    assert_eq!(Tune::GRAVITON3.kind(), TuneKind::Graviton3);
  }

  #[test]
  fn test_tune_name_custom() {
    let custom = Tune::custom(128, true, 5, true, 512, true);
    assert_eq!(custom.name(), "Custom");
    assert_eq!(custom.kind(), TuneKind::Custom);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Display Formatting
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_tune_display() {
    let tune = Tune::ZEN5;
    let s = alloc::format!("{tune}");
    assert!(s.contains("threshold=64"));
    assert!(s.contains("hybrid=true"));
    assert!(s.contains("parallelism=7"));
    assert!(s.contains("fast_zmm=true"));
    assert!(s.contains("sve_vlen=0"));
    assert!(s.contains("sme=false"));
  }

  #[test]
  fn test_tune_display_graviton3() {
    let tune = Tune::GRAVITON3;
    let s = alloc::format!("{tune}");
    assert!(s.contains("sve_vlen=256"));
    assert!(s.contains("sme=false"));
  }

  #[test]
  fn test_tune_display_apple_m4() {
    let tune = Tune::APPLE_M4;
    let s = alloc::format!("{tune}");
    assert!(s.contains("sve_vlen=0"));
    assert!(s.contains("sme=true"));
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // All Presets Have Reasonable Values
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_all_presets_have_reasonable_values() {
    let presets = [
      Tune::DEFAULT,
      Tune::ZEN4,
      Tune::ZEN5,
      Tune::INTEL_SPR,
      Tune::INTEL_ICL,
      Tune::APPLE_M1_M3,
      Tune::APPLE_M4,
      Tune::GRAVITON2,
      Tune::GRAVITON3,
      Tune::GRAVITON4,
      Tune::NEOVERSE_N2,
      Tune::AARCH64_PMULL,
      Tune::PORTABLE,
    ];

    for tune in presets {
      // All thresholds should be positive
      assert!(tune.simd_threshold > 0, "simd_threshold should be > 0");
      // Parallelism should be at least 1
      assert!(tune.crc_parallelism >= 1, "crc_parallelism should be >= 1");
      // Threshold should be a power of 2 or common value
      assert!(
        tune.simd_threshold == 64 || tune.simd_threshold == 128 || tune.simd_threshold == 256,
        "unexpected simd_threshold: {}",
        tune.simd_threshold
      );
      // SVE vector length should be 0, 128, 256, or 512
      assert!(
        tune.sve_vlen == 0 || tune.sve_vlen == 128 || tune.sve_vlen == 256 || tune.sve_vlen == 512,
        "unexpected sve_vlen: {}",
        tune.sve_vlen
      );
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Equality and Clone
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_tune_equality() {
    assert_eq!(Tune::ZEN5, Tune::ZEN5);
    assert_ne!(Tune::ZEN5, Tune::ZEN4);
    assert_ne!(Tune::INTEL_SPR, Tune::APPLE_M);
    assert_ne!(Tune::GRAVITON3, Tune::GRAVITON4); // Different SVE vector lengths
    assert_ne!(Tune::APPLE_M1_M3, Tune::APPLE_M4); // Different SME support
  }

  #[test]
  fn test_tune_clone() {
    let tune = Tune::ZEN5;
    let cloned = tune;
    assert_eq!(tune, cloned);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // SVE and SME Field Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_sve_vlen_differentiation() {
    // Graviton 3 and 4 differ only by SVE vector length
    assert_eq!(Tune::GRAVITON3.sve_vlen, 256);
    assert_eq!(Tune::GRAVITON4.sve_vlen, 128);
    assert_ne!(Tune::GRAVITON3, Tune::GRAVITON4);
  }

  #[test]
  fn test_sme_differentiation() {
    // Apple M1-M3 and M4 differ by SME support
    // Use const blocks for compile-time assertions on const values
    const { assert!(!Tune::APPLE_M1_M3.has_sme) };
    const { assert!(Tune::APPLE_M4.has_sme) };
    assert_ne!(Tune::APPLE_M1_M3, Tune::APPLE_M4);
  }
}
