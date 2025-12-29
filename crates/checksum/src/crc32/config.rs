//! CRC-32 runtime configuration (overrides + thresholds).
//!
//! This module centralizes selection knobs for CRC-32/CRC-32C:
//! - portable vs hardware instruction thresholds
//! - hardware instruction vs fusion thresholds
//! - optional forced backend selection
//!
//! Safety note: forced modes are always clamped to detected CPU capabilities.

use platform::{Caps, Tune};

use super::tuned_defaults;

/// Forced backend selection for CRC-32/CRC-32C.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Crc32Force {
  /// Use the default auto selector.
  #[default]
  Auto,
  /// Force the portable table-based implementation.
  Portable,
  /// Force hardware CRC instructions (if available).
  ///
  /// - x86_64: SSE4.2 `crc32` (CRC-32C only)
  /// - aarch64: ARMv8 CRC extension (CRC-32 + CRC-32C)
  Hwcrc,
  /// Force x86_64 CRC-32C fusion kernels (SSE4.2 + PCLMULQDQ) if supported.
  Pclmul,
  /// Force x86_64 CRC-32C fusion kernels (AVX-512 + VPCLMULQDQ) if supported.
  Vpclmul,
  /// Force aarch64 CRC fusion kernels (CRC + PMULL) if supported.
  Pmull,
  /// Force aarch64 CRC fusion kernels (CRC + PMULL + EOR3/SHA3) if supported.
  PmullEor3,
  /// Force aarch64 "SVE2 PMULL" tier (2/3-way striping) if supported.
  ///
  /// This mirrors the CRC64 selector: it is an ILP-oriented tier intended for
  /// Armv9/SVE2-class CPUs. Implementations are still NEON+PMULL based.
  Sve2Pmull,
}

impl Crc32Force {
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Portable => "portable",
      Self::Hwcrc => "hwcrc",
      Self::Pclmul => "pclmul",
      Self::Vpclmul => "vpclmul",
      Self::Pmull => "pmull",
      Self::PmullEor3 => "pmull-eor3",
      Self::Sve2Pmull => "sve2-pmull",
    }
  }
}

/// CRC-32 selection tunables (thresholds + parallelism).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32Tunables {
  /// Bytes where hardware CRC becomes faster than portable.
  pub portable_to_hwcrc: usize,
  /// Bytes where fusion becomes faster than HWCRC.
  pub hwcrc_to_fusion: usize,
  /// Bytes where AVX-512 fusion becomes worthwhile (x86_64 only).
  pub fusion_to_avx512: usize,
  /// Bytes where VPCLMUL fusion becomes worthwhile (x86_64 only).
  pub fusion_to_vpclmul: usize,
  /// Preferred number of independent streams (used by multi-stream kernels where available).
  pub streams: u8,
}

/// Full CRC-32 runtime configuration (after applying overrides).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32Config {
  /// Requested force mode (env/programmatic).
  pub requested_force: Crc32Force,
  /// Force mode clamped to detected CPU capabilities.
  pub effective_force: Crc32Force,
  /// Thresholds and stream settings used by the selector.
  pub tunables: Crc32Tunables,
}

#[derive(Clone, Copy, Debug, Default)]
struct Overrides {
  force: Crc32Force,
  portable_to_hwcrc: Option<usize>,
  hwcrc_to_fusion: Option<usize>,
  fusion_to_avx512: Option<usize>,
  fusion_to_vpclmul: Option<usize>,
  streams: Option<u8>,
}

#[cfg(feature = "std")]
fn read_env_overrides() -> Overrides {
  fn parse_usize(name: &str) -> Option<usize> {
    let value = std::env::var(name).ok()?;
    let value = value.trim();
    if value.is_empty() {
      return None;
    }
    value.parse::<usize>().ok()
  }

  fn parse_u8(name: &str) -> Option<u8> {
    let value = std::env::var(name).ok()?;
    let value = value.trim();
    if value.is_empty() {
      return None;
    }
    value.parse::<u8>().ok()
  }

  fn parse_force(name: &str) -> Option<Crc32Force> {
    let value = std::env::var(name).ok()?;
    let value = value.trim();
    if value.is_empty() {
      return None;
    }

    if value.eq_ignore_ascii_case("auto") {
      return Some(Crc32Force::Auto);
    }
    if value.eq_ignore_ascii_case("portable")
      || value.eq_ignore_ascii_case("scalar")
      || value.eq_ignore_ascii_case("table")
    {
      return Some(Crc32Force::Portable);
    }
    if value.eq_ignore_ascii_case("hwcrc")
      || value.eq_ignore_ascii_case("crc")
      || value.eq_ignore_ascii_case("crc32")
      || value.eq_ignore_ascii_case("crc32c")
    {
      return Some(Crc32Force::Hwcrc);
    }
    if value.eq_ignore_ascii_case("pclmul") || value.eq_ignore_ascii_case("clmul") {
      return Some(Crc32Force::Pclmul);
    }
    if value.eq_ignore_ascii_case("vpclmul") {
      return Some(Crc32Force::Vpclmul);
    }
    if value.eq_ignore_ascii_case("pmull") || value.eq_ignore_ascii_case("pmull-neon") {
      return Some(Crc32Force::Pmull);
    }
    if value.eq_ignore_ascii_case("pmull-eor3")
      || value.eq_ignore_ascii_case("eor3")
      || value.eq_ignore_ascii_case("pmull-sha3")
    {
      return Some(Crc32Force::PmullEor3);
    }
    if value.eq_ignore_ascii_case("sve2-pmull") || value.eq_ignore_ascii_case("sve2pmull") {
      return Some(Crc32Force::Sve2Pmull);
    }

    None
  }

  Overrides {
    force: parse_force("RSCRYPTO_CRC32_FORCE").unwrap_or(Crc32Force::Auto),
    portable_to_hwcrc: parse_usize("RSCRYPTO_CRC32_THRESHOLD_PORTABLE_TO_HWCRC"),
    hwcrc_to_fusion: parse_usize("RSCRYPTO_CRC32_THRESHOLD_HWCRC_TO_FUSION"),
    fusion_to_avx512: parse_usize("RSCRYPTO_CRC32_THRESHOLD_FUSION_TO_AVX512"),
    fusion_to_vpclmul: parse_usize("RSCRYPTO_CRC32_THRESHOLD_FUSION_TO_VPCLMUL"),
    streams: parse_u8("RSCRYPTO_CRC32_STREAMS"),
  }
}

#[cfg(feature = "std")]
fn overrides() -> Overrides {
  use std::sync::OnceLock;
  static OVERRIDES: OnceLock<Overrides> = OnceLock::new();
  *OVERRIDES.get_or_init(read_env_overrides)
}

#[cfg(not(feature = "std"))]
fn overrides() -> Overrides {
  Overrides::default()
}

#[inline]
#[must_use]
#[allow(unused_variables)] // `caps` only used on x86_64/aarch64
fn clamp_force_to_caps(requested: Crc32Force, caps: Caps) -> Crc32Force {
  match requested {
    Crc32Force::Auto | Crc32Force::Portable => requested,
    Crc32Force::Hwcrc => {
      #[cfg(target_arch = "x86_64")]
      {
        if caps.has(platform::caps::x86::CRC32C_READY) {
          return Crc32Force::Hwcrc;
        }
      }
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::CRC_READY) {
          return Crc32Force::Hwcrc;
        }
      }
      Crc32Force::Auto
    }
    Crc32Force::Pclmul => {
      #[cfg(target_arch = "x86_64")]
      {
        if caps.has(platform::caps::x86::PCLMUL_READY) {
          return Crc32Force::Pclmul;
        }
      }
      Crc32Force::Auto
    }
    Crc32Force::Vpclmul => {
      #[cfg(target_arch = "x86_64")]
      {
        if caps.has(platform::caps::x86::VPCLMUL_READY) {
          return Crc32Force::Vpclmul;
        }
      }
      Crc32Force::Auto
    }
    Crc32Force::Pmull => {
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_READY) {
          return Crc32Force::Pmull;
        }
      }
      Crc32Force::Auto
    }
    Crc32Force::PmullEor3 => {
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
          return Crc32Force::PmullEor3;
        }
      }
      Crc32Force::Auto
    }
    Crc32Force::Sve2Pmull => {
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::CRC_READY)
          && caps.has(platform::caps::aarch64::PMULL_READY)
          && caps.has(platform::caps::aarch64::SVE2_PMULL)
        {
          return Crc32Force::Sve2Pmull;
        }
      }
      Crc32Force::Auto
    }
  }
}

#[inline]
#[must_use]
fn tuned_defaults(caps: Caps, tune: Tune) -> Crc32Tunables {
  let tuned = tuned_defaults::for_tune_kind(tune.kind);

  let default_streams = default_crc32_streams(caps, tune);
  let default_fusion_to_avx512 = tune.simd_threshold;
  let default_fusion_to_vpclmul = if tune.fast_wide_ops {
    tune.simd_threshold
  } else {
    tune.simd_threshold.saturating_mul(8)
  };

  let streams = tuned.map(|t| t.streams).unwrap_or(default_streams);
  let fusion_to_avx512 = tuned.map(|t| t.fusion_to_avx512).unwrap_or(default_fusion_to_avx512);
  let fusion_to_vpclmul = tuned.map(|t| t.fusion_to_vpclmul).unwrap_or(default_fusion_to_vpclmul);
  Crc32Tunables {
    // If HWCRC isn't available (e.g. POWER/s390x/RISC-V), treat this as the
    // portable→accelerated crossover by falling back to the CLMUL threshold.
    portable_to_hwcrc: tuned
      .map(|t| t.portable_to_hwcrc)
      .unwrap_or(tune.hwcrc_threshold.min(tune.pclmul_threshold)),
    hwcrc_to_fusion: tuned.map(|t| t.hwcrc_to_fusion).unwrap_or(tune.pclmul_threshold),
    fusion_to_avx512,
    fusion_to_vpclmul,
    streams,
  }
}

#[inline]
#[must_use]
#[allow(unused_variables)] // arch-specific
fn default_crc32_streams(caps: Caps, tune: Tune) -> u8 {
  // Keep CRC32 stream defaults in parity with CRC64.
  #[cfg(target_arch = "x86_64")]
  {
    let _ = caps;
    match tune.parallel_streams {
      0 | 1 => 1,
      2 => 2,
      3 => 4,
      4..=6 => 7,
      _ => 8,
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    let _ = caps;
    tune.parallel_streams.clamp(1, 3)
  }

  #[cfg(target_arch = "powerpc64")]
  {
    if !caps.has(platform::caps::powerpc64::VPMSUM_READY) {
      return 1;
    }
    tune.parallel_streams.saturating_mul(2).clamp(1, 8)
  }

  #[cfg(target_arch = "s390x")]
  {
    if !caps.has(platform::caps::s390x::VECTOR) {
      return 1;
    }
    tune.parallel_streams.saturating_mul(2).clamp(1, 4)
  }

  #[cfg(target_arch = "riscv64")]
  {
    if !(caps.has(platform::caps::riscv::ZVBC) || caps.has(platform::caps::riscv::ZBC)) {
      return 1;
    }
    match tune.parallel_streams {
      0 | 1 => 1,
      2 => 2,
      _ => 4,
    }
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  )))]
  {
    let _ = (caps, tune);
    1
  }
}

/// Get the effective CRC-32 configuration for the current platform.
#[inline]
#[must_use]
pub fn get() -> Crc32Config {
  let caps = platform::caps();
  let tune = platform::tune();
  let ov = overrides();
  let base = tuned_defaults(caps, tune);

  let requested_force = ov.force;
  let effective_force = clamp_force_to_caps(requested_force, caps);

  let mut portable_to_hwcrc = base.portable_to_hwcrc;
  if let Some(v) = ov.portable_to_hwcrc {
    portable_to_hwcrc = v;
  }

  let mut hwcrc_to_fusion = base.hwcrc_to_fusion;
  if let Some(v) = ov.hwcrc_to_fusion {
    hwcrc_to_fusion = v;
  }

  let mut fusion_to_avx512 = base.fusion_to_avx512;
  if let Some(v) = ov.fusion_to_avx512 {
    fusion_to_avx512 = v;
  }

  let mut fusion_to_vpclmul = base.fusion_to_vpclmul;
  if let Some(v) = ov.fusion_to_vpclmul {
    fusion_to_vpclmul = v;
  }

  let mut streams = base.streams;
  if let Some(v) = ov.streams {
    streams = v;
  }

  // Clamp streams to the supported name mapping.
  if streams == 0 {
    streams = 1;
  }
  if streams > 8 {
    streams = 8;
  }

  Crc32Config {
    requested_force,
    effective_force,
    tunables: Crc32Tunables {
      portable_to_hwcrc,
      hwcrc_to_fusion,
      fusion_to_avx512,
      fusion_to_vpclmul,
      streams,
    },
  }
}
