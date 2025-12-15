//! x86_64 CPU detection and microarchitecture identification.
//!
//! This module provides:
//! - Vendor detection (Intel vs AMD)
//! - Microarchitecture identification (Sapphire Rapids, Ice Lake, Zen 4, etc.)
//! - Feature capability queries
//! - Optimal configuration hints for SIMD algorithms
//!
//! # Microarchitecture-Specific Optimizations
//!
//! Different microarchitectures have different optimal configurations:
//!
//! | Microarch | VPCLMULQDQ | ZMM Warmup | Optimal CRC Config |
//! |-----------|------------|------------|-------------------|
//! | Sapphire Rapids | 1×p5 | ~2000ns | v3s1_s3 |
//! | Ice Lake | 1×p05+2×p5 | ~2000ns | v4s5x3 |
//! | Cascade Lake | N/A | N/A | v9s3x4e |
//! | Zen 4 | Good | ~60ns | v3s2x4 |
//! | Zen 5 | Good | ~60ns | v3s7 (7-way crc32q) |

#![allow(unsafe_code)] // Required for CPUID intrinsics

use core::arch::x86_64::__cpuid;

/// CPU vendor identification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Vendor {
  Intel,
  Amd,
  Unknown,
}

/// x86_64 microarchitecture identification.
///
/// Used to select optimal algorithm configurations per CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MicroArch {
  // === Intel (different VPCLMULQDQ throughput characteristics) ===
  /// Cascade Lake (Family 6, Model 85) - No VPCLMULQDQ
  CascadeLake,

  /// Ice Lake Server (Family 6, Model 106/108) - VPCLMULQDQ 1×p05+2×p5
  IceLake,

  /// Sapphire Rapids (Family 6, Model 143) - VPCLMULQDQ 1×p5 (best Intel)
  SapphireRapids,

  /// Emerald Rapids (Family 6, Model 207) - Similar to Sapphire Rapids
  EmeraldRapids,

  /// Granite Rapids (Family 6, Model 173) - Next-gen
  GraniteRapids,

  // === AMD (virtually zero ZMM warmup) ===
  /// Zen 3 (Family 25, various models) - No AVX-512
  Zen3,

  /// Zen 4 (Family 25, Model 96-111) - AVX-512 + VPCLMULQDQ, ~60ns warmup
  Zen4,

  /// Zen 5 (Family 26) - AVX-512 + 7-way crc32q parallelism
  Zen5,

  // === Fallbacks ===
  /// Has AVX-512 + VPCLMULQDQ but unknown specific microarch
  GenericAvx512Vpclmul,

  /// Has PCLMULQDQ but no AVX-512
  GenericPclmul,

  /// Generic x86_64 (portable fallback)
  Generic,
}

impl MicroArch {
  /// Returns the CPU vendor for this microarchitecture.
  #[inline]
  #[must_use]
  pub const fn vendor(self) -> Vendor {
    match self {
      Self::CascadeLake | Self::IceLake | Self::SapphireRapids | Self::EmeraldRapids | Self::GraniteRapids => {
        Vendor::Intel
      }

      Self::Zen3 | Self::Zen4 | Self::Zen5 => Vendor::Amd,

      Self::GenericAvx512Vpclmul | Self::GenericPclmul | Self::Generic => Vendor::Unknown,
    }
  }

  /// Returns `true` if this microarch supports VPCLMULQDQ (AVX-512 carryless multiply).
  #[inline]
  #[must_use]
  pub const fn has_vpclmulqdq(self) -> bool {
    matches!(
      self,
      Self::IceLake
        | Self::SapphireRapids
        | Self::EmeraldRapids
        | Self::GraniteRapids
        | Self::Zen4
        | Self::Zen5
        | Self::GenericAvx512Vpclmul
    )
  }

  /// Returns `true` if this microarch supports PCLMULQDQ (128-bit carryless multiply).
  ///
  /// All known x86_64 microarchitectures have PCLMULQDQ except generic/unknown CPUs.
  #[inline]
  #[must_use]
  pub const fn has_pclmulqdq(self) -> bool {
    !matches!(self, Self::Generic)
  }

  /// Returns `true` if this microarch has low ZMM register warmup latency.
  ///
  /// AMD Zen 4/5 have ~60ns warmup vs Intel's ~2000ns.
  /// This affects whether AVX-512 is beneficial for small buffers.
  #[inline]
  #[must_use]
  pub const fn has_fast_zmm_warmup(self) -> bool {
    matches!(self, Self::Zen4 | Self::Zen5)
  }

  /// Returns the recommended small-buffer threshold for SIMD.
  ///
  /// Below this size, scalar/portable code may be faster due to SIMD setup overhead.
  /// AMD's low ZMM warmup allows lower thresholds.
  #[inline]
  #[must_use]
  pub const fn simd_threshold(self) -> usize {
    match self {
      // AMD: low warmup, use SIMD aggressively
      Self::Zen4 | Self::Zen5 => 64,

      // Intel with VPCLMULQDQ: higher warmup, need larger buffers
      Self::SapphireRapids | Self::EmeraldRapids | Self::GraniteRapids => 256,
      Self::IceLake => 256,

      // No VPCLMULQDQ or unknown
      Self::CascadeLake | Self::Zen3 => 256,
      Self::GenericAvx512Vpclmul => 256,
      Self::GenericPclmul => 64,
      Self::Generic => 64,
    }
  }

  /// Returns the number of parallel crc32q instructions this microarch can sustain.
  ///
  /// Zen 5 can do 7-way parallel crc32q, others typically 3-way.
  #[inline]
  #[must_use]
  pub const fn crc32q_parallelism(self) -> u8 {
    match self {
      Self::Zen5 => 7,
      Self::Zen4 | Self::Zen3 => 3,
      Self::SapphireRapids | Self::EmeraldRapids | Self::GraniteRapids | Self::IceLake | Self::CascadeLake => 3,
      Self::GenericAvx512Vpclmul | Self::GenericPclmul | Self::Generic => 3,
    }
  }
}

/// Raw CPUID result for leaf 0 (vendor string).
#[derive(Clone, Copy, Debug)]
struct CpuidVendor {
  vendor: Vendor,
}

impl CpuidVendor {
  #[inline]
  fn query() -> Self {
    // SAFETY: CPUID is always available on x86_64
    let result = unsafe { __cpuid(0) };

    // Vendor string is in EBX-EDX-ECX order
    let vendor = if result.ebx == 0x756e_6547    // "Genu"
            && result.edx == 0x4965_6e69             // "ineI"
            && result.ecx == 0x6c65_746e
    // "ntel"
    {
      Vendor::Intel
    } else if result.ebx == 0x6874_7541  // "Auth"
            && result.edx == 0x6974_6e65     // "enti"
            && result.ecx == 0x444d_4163
    // "cAMD"
    {
      Vendor::Amd
    } else {
      Vendor::Unknown
    };

    Self { vendor }
  }
}

/// Raw CPUID result for leaf 1 (family/model/stepping).
#[derive(Clone, Copy, Debug)]
struct CpuidSignature {
  family: u32,
  model: u32,
  #[allow(dead_code)]
  stepping: u32,
}

impl CpuidSignature {
  #[inline]
  fn query() -> Self {
    // SAFETY: CPUID is always available on x86_64
    let result = unsafe { __cpuid(1) };

    let stepping = result.eax & 0xF;
    let base_model = (result.eax >> 4) & 0xF;
    let base_family = (result.eax >> 8) & 0xF;
    let ext_model = (result.eax >> 16) & 0xF;
    let ext_family = (result.eax >> 20) & 0xFF;

    // Family calculation per Intel/AMD spec
    let family = if base_family == 0xF {
      base_family.strict_add(ext_family)
    } else {
      base_family
    };

    // Model calculation per Intel/AMD spec
    let model = if base_family == 0x6 || base_family == 0xF {
      (ext_model << 4).strict_add(base_model)
    } else {
      base_model
    };

    Self {
      family,
      model,
      stepping,
    }
  }
}

/// Detect the current CPU's microarchitecture.
///
/// This function queries CPUID to determine the specific microarchitecture,
/// enabling optimal algorithm selection.
///
/// # Performance
///
/// With the `std` feature, results are cached in a `OnceLock`.
/// Without `std`, CPUID is called each time (~100 cycles).
///
/// # Example
///
/// ```ignore
/// use platform::x86_64::detect_microarch;
///
/// let arch = detect_microarch();
/// if arch.has_vpclmulqdq() {
///     // Use AVX-512 carryless multiply path
/// }
/// ```
#[cfg(feature = "std")]
#[inline]
#[must_use]
pub fn detect_microarch() -> MicroArch {
  use std::sync::OnceLock;
  static CACHED: OnceLock<MicroArch> = OnceLock::new();
  *CACHED.get_or_init(detect_microarch_uncached)
}

/// Detect microarchitecture without caching.
///
/// Prefer [`detect_microarch`] when the `std` feature is available.
#[inline]
#[must_use]
pub fn detect_microarch_uncached() -> MicroArch {
  let vendor = CpuidVendor::query();
  let sig = CpuidSignature::query();

  match vendor.vendor {
    Vendor::Intel => detect_intel(sig.family, sig.model),
    Vendor::Amd => detect_amd(sig.family, sig.model),
    Vendor::Unknown => detect_by_features(),
  }
}

#[inline]
fn detect_intel(family: u32, model: u32) -> MicroArch {
  if family != 6 {
    return detect_by_features();
  }

  match model {
    // Cascade Lake (no VPCLMULQDQ)
    85 => MicroArch::CascadeLake,

    // Ice Lake Server
    106 | 108 => MicroArch::IceLake,

    // Sapphire Rapids
    143 => MicroArch::SapphireRapids,

    // Emerald Rapids
    207 => MicroArch::EmeraldRapids,

    // Granite Rapids
    173 => MicroArch::GraniteRapids,

    // Unknown Intel - fall back to feature detection
    _ => detect_by_features(),
  }
}

#[inline]
fn detect_amd(family: u32, model: u32) -> MicroArch {
  match family {
    // Zen 3 and Zen 4 share Family 25 (0x19)
    25 => {
      // Zen 4 models are typically 96-111 (Raphael, Genoa, etc.)
      // Zen 3 models are typically 0-95
      if model >= 96 { MicroArch::Zen4 } else { MicroArch::Zen3 }
    }

    // Zen 5 is Family 26 (0x1A)
    26 => MicroArch::Zen5,

    // Older or unknown AMD
    _ => detect_by_features(),
  }
}

/// Fall back to feature detection when microarch is unknown.
#[cfg(feature = "std")]
#[inline]
fn detect_by_features() -> MicroArch {
  let has_vpclmulqdq = std::arch::is_x86_feature_detected!("vpclmulqdq")
    && std::arch::is_x86_feature_detected!("avx512f")
    && std::arch::is_x86_feature_detected!("avx512vl")
    && std::arch::is_x86_feature_detected!("avx512bw");

  let has_pclmulqdq = std::arch::is_x86_feature_detected!("pclmulqdq") && std::arch::is_x86_feature_detected!("ssse3");

  if has_vpclmulqdq {
    MicroArch::GenericAvx512Vpclmul
  } else if has_pclmulqdq {
    MicroArch::GenericPclmul
  } else {
    MicroArch::Generic
  }
}

/// Fall back when no std - just return Generic.
#[cfg(not(feature = "std"))]
#[inline]
fn detect_by_features() -> MicroArch {
  // Without std, we can't do runtime feature detection.
  // Compile-time detection via target_feature will be used instead.
  MicroArch::Generic
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_detect_runs() {
    let arch = detect_microarch_uncached();
    // Just verify it doesn't panic and returns something reasonable
    let _ = arch.vendor();
    let _ = arch.has_vpclmulqdq();
    let _ = arch.simd_threshold();
  }

  #[test]
  fn test_vendor_consistency() {
    let arch = detect_microarch_uncached();
    match arch {
      MicroArch::SapphireRapids
      | MicroArch::IceLake
      | MicroArch::EmeraldRapids
      | MicroArch::GraniteRapids
      | MicroArch::CascadeLake => {
        assert_eq!(arch.vendor(), Vendor::Intel);
      }
      MicroArch::Zen3 | MicroArch::Zen4 | MicroArch::Zen5 => {
        assert_eq!(arch.vendor(), Vendor::Amd);
      }
      _ => {}
    }
  }

  #[test]
  fn test_intel_model_detection() {
    // Sapphire Rapids
    assert_eq!(detect_intel(6, 143), MicroArch::SapphireRapids);
    // Ice Lake
    assert_eq!(detect_intel(6, 106), MicroArch::IceLake);
    assert_eq!(detect_intel(6, 108), MicroArch::IceLake);
    // Cascade Lake
    assert_eq!(detect_intel(6, 85), MicroArch::CascadeLake);
    // Emerald Rapids
    assert_eq!(detect_intel(6, 207), MicroArch::EmeraldRapids);
  }

  #[test]
  fn test_amd_family_detection() {
    // Zen 3 (Family 25, low models)
    assert_eq!(detect_amd(25, 0), MicroArch::Zen3);
    assert_eq!(detect_amd(25, 50), MicroArch::Zen3);
    // Zen 4 (Family 25, high models)
    assert_eq!(detect_amd(25, 96), MicroArch::Zen4);
    assert_eq!(detect_amd(25, 111), MicroArch::Zen4);
    // Zen 5 (Family 26)
    assert_eq!(detect_amd(26, 0), MicroArch::Zen5);
  }

  #[test]
  fn test_microarch_properties() {
    // Zen 4/5 have fast ZMM warmup
    assert!(MicroArch::Zen4.has_fast_zmm_warmup());
    assert!(MicroArch::Zen5.has_fast_zmm_warmup());
    assert!(!MicroArch::SapphireRapids.has_fast_zmm_warmup());

    // Zen 5 has 7-way crc32q
    assert_eq!(MicroArch::Zen5.crc32q_parallelism(), 7);
    assert_eq!(MicroArch::Zen4.crc32q_parallelism(), 3);

    // VPCLMULQDQ support
    assert!(MicroArch::SapphireRapids.has_vpclmulqdq());
    assert!(MicroArch::Zen4.has_vpclmulqdq());
    assert!(!MicroArch::CascadeLake.has_vpclmulqdq());
    assert!(!MicroArch::Zen3.has_vpclmulqdq());
  }
}
