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

/// Microarchitecture-derived tuning hints.
///
/// These hints guide algorithm selection and threshold decisions.
/// They are derived from the detected CPU microarchitecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tune {
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
}

impl Tune {
  /// Conservative defaults for unknown CPUs.
  pub const DEFAULT: Self = Self {
    simd_threshold: 256,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
  };

  /// Tuning for AMD Zen 4.
  pub const ZEN4: Self = Self {
    simd_threshold: 64,
    prefer_hybrid_crc: true,
    crc_parallelism: 3,
    fast_zmm: true,
  };

  /// Tuning for AMD Zen 5.
  pub const ZEN5: Self = Self {
    simd_threshold: 64,
    prefer_hybrid_crc: true,
    crc_parallelism: 7,
    fast_zmm: true,
  };

  /// Tuning for Intel Sapphire Rapids / Emerald Rapids / Granite Rapids.
  pub const INTEL_SPR: Self = Self {
    simd_threshold: 256,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
  };

  /// Tuning for Intel Ice Lake.
  pub const INTEL_ICL: Self = Self {
    simd_threshold: 256,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
  };

  /// Tuning for Apple M-series (aarch64 with PMULL+EOR3).
  pub const APPLE_M: Self = Self {
    simd_threshold: 64,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false, // N/A for ARM
  };

  /// Tuning for generic aarch64 with PMULL.
  pub const AARCH64_PMULL: Self = Self {
    simd_threshold: 256,
    prefer_hybrid_crc: false,
    crc_parallelism: 3,
    fast_zmm: false,
  };

  /// Tuning for portable/scalar code.
  pub const PORTABLE: Self = Self {
    simd_threshold: 64,
    prefer_hybrid_crc: false,
    crc_parallelism: 1,
    fast_zmm: false,
  };
}

impl Default for Tune {
  #[inline]
  fn default() -> Self {
    Self::DEFAULT
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_tune_defaults() {
    let tune = Tune::default();
    assert_eq!(tune.simd_threshold, 256);
    assert!(!tune.prefer_hybrid_crc);
  }

  #[test]
  fn test_tune_zen5() {
    let tune = Tune::ZEN5;
    assert_eq!(tune.simd_threshold, 64);
    assert!(tune.prefer_hybrid_crc);
    assert_eq!(tune.crc_parallelism, 7);
    assert!(tune.fast_zmm);
  }

  #[test]
  fn test_tune_intel() {
    let tune = Tune::INTEL_SPR;
    assert_eq!(tune.simd_threshold, 256);
    assert!(!tune.prefer_hybrid_crc);
    assert!(!tune.fast_zmm);
  }
}
