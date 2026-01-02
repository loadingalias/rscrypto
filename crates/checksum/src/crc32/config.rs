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
  /// Force the bitwise reference implementation (slow, obviously correct).
  ///
  /// This is the canonical reference against which all optimizations are
  /// verified. Use for correctness testing or when absolute auditability
  /// is required over performance.
  Reference,
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
  /// Force powerpc64 VPMSUMD folding (if supported).
  Vpmsum,
  /// Force s390x VGFM folding (if supported).
  Vgfm,
  /// Force riscv64 Zbc carryless multiply folding (if supported).
  Zbc,
  /// Force riscv64 Zvbc (vector carryless multiply) folding (if supported).
  Zvbc,
}

impl Crc32Force {
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Reference => "reference",
      Self::Portable => "portable",
      Self::Hwcrc => "hwcrc",
      Self::Pclmul => "pclmul",
      Self::Vpclmul => "vpclmul",
      Self::Pmull => "pmull",
      Self::PmullEor3 => "pmull-eor3",
      Self::Sve2Pmull => "sve2-pmull",
      Self::Vpmsum => "vpmsum",
      Self::Vgfm => "vgfm",
      Self::Zbc => "zbc",
      Self::Zvbc => "zvbc",
    }
  }

  /// Map to kernel family (None means Auto selection).
  #[must_use]
  pub const fn to_family(self) -> Option<backend::KernelFamily> {
    use backend::KernelFamily;
    match self {
      Self::Auto => None,
      Self::Reference => Some(KernelFamily::Reference),
      Self::Portable => Some(KernelFamily::Portable),
      Self::Hwcrc => {
        // Architecture-specific: x86_64 uses X86Crc32, aarch64 uses ArmCrc32
        #[cfg(target_arch = "x86_64")]
        {
          Some(KernelFamily::X86Crc32)
        }
        #[cfg(target_arch = "aarch64")]
        {
          Some(KernelFamily::ArmCrc32)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
          Some(KernelFamily::Portable)
        }
      }
      Self::Pclmul => Some(KernelFamily::X86Pclmul),
      Self::Vpclmul => Some(KernelFamily::X86Vpclmul),
      Self::Pmull => Some(KernelFamily::ArmPmull),
      Self::PmullEor3 => Some(KernelFamily::ArmPmullEor3),
      Self::Sve2Pmull => Some(KernelFamily::ArmSve2Pmull),
      Self::Vpmsum => Some(KernelFamily::PowerVpmsum),
      Self::Vgfm => Some(KernelFamily::S390xVgfm),
      Self::Zbc => Some(KernelFamily::RiscvZbc),
      Self::Zvbc => Some(KernelFamily::RiscvZvbc),
    }
  }

  /// Map from kernel family to force mode.
  #[must_use]
  pub const fn from_family(family: backend::KernelFamily) -> Self {
    use backend::KernelFamily;
    match family {
      KernelFamily::Reference => Self::Reference,
      KernelFamily::Portable => Self::Portable,
      KernelFamily::X86Crc32 | KernelFamily::ArmCrc32 => Self::Hwcrc,
      KernelFamily::X86Pclmul => Self::Pclmul,
      KernelFamily::X86Vpclmul => Self::Vpclmul,
      KernelFamily::ArmPmull => Self::Pmull,
      KernelFamily::ArmPmullEor3 => Self::PmullEor3,
      KernelFamily::ArmSve2Pmull => Self::Sve2Pmull,
      KernelFamily::PowerVpmsum => Self::Vpmsum,
      KernelFamily::S390xVgfm => Self::Vgfm,
      KernelFamily::RiscvZbc => Self::Zbc,
      KernelFamily::RiscvZvbc => Self::Zvbc,
      // Non-exhaustive fallback
      _ => Self::Portable,
    }
  }
}

/// CRC-32 selection tunables (thresholds + parallelism).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32VariantTunables {
  /// Bytes where hardware CRC becomes faster than portable.
  pub portable_to_hwcrc: usize,
  /// Bytes where fusion becomes faster than HWCRC.
  pub hwcrc_to_fusion: usize,
  /// Bytes where AVX-512 fusion becomes worthwhile (x86_64 only).
  pub fusion_to_avx512: usize,
  /// Bytes where VPCLMUL fusion becomes worthwhile (x86_64 only).
  pub fusion_to_vpclmul: usize,
  /// Preferred number of independent streams.
  ///
  /// Used by multi-stream kernels where available.
  pub streams: u8,
  /// Minimum bytes per lane for multi-stream folding.
  ///
  /// When `None`, uses `KernelFamily::min_bytes_per_lane()` as the default.
  pub min_bytes_per_lane: Option<usize>,
}

/// CRC-32 selection tunables (per-variant thresholds + parallelism).
///
/// CRC-32 and CRC-32C use different instruction sets and have different
/// crossovers on many machines. Keep their tunables separate so tuning can
/// be applied losslessly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32Tunables {
  pub crc32: Crc32VariantTunables,
  pub crc32c: Crc32VariantTunables,
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
struct VariantOverrides {
  portable_to_hwcrc: Option<usize>,
  hwcrc_to_fusion: Option<usize>,
  fusion_to_avx512: Option<usize>,
  fusion_to_vpclmul: Option<usize>,
  streams: Option<u8>,
  min_bytes_per_lane: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Overrides {
  force: Crc32Force,
  crc32: VariantOverrides,
  crc32c: VariantOverrides,
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
    if value.eq_ignore_ascii_case("reference") || value.eq_ignore_ascii_case("bitwise") {
      return Some(Crc32Force::Reference);
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
    if value.eq_ignore_ascii_case("vpmsum") || value.eq_ignore_ascii_case("vpmsumd") {
      return Some(Crc32Force::Vpmsum);
    }
    if value.eq_ignore_ascii_case("vgfm") || value.eq_ignore_ascii_case("gfmsum") {
      return Some(Crc32Force::Vgfm);
    }
    if value.eq_ignore_ascii_case("zbc") {
      return Some(Crc32Force::Zbc);
    }
    if value.eq_ignore_ascii_case("zvbc") || value.eq_ignore_ascii_case("vclmul") {
      return Some(Crc32Force::Zvbc);
    }

    None
  }

  let crc32 = VariantOverrides {
    portable_to_hwcrc: parse_usize("RSCRYPTO_CRC32_THRESHOLD_PORTABLE_TO_HWCRC"),
    hwcrc_to_fusion: parse_usize("RSCRYPTO_CRC32_THRESHOLD_HWCRC_TO_FUSION"),
    fusion_to_avx512: parse_usize("RSCRYPTO_CRC32_THRESHOLD_FUSION_TO_AVX512"),
    fusion_to_vpclmul: parse_usize("RSCRYPTO_CRC32_THRESHOLD_FUSION_TO_VPCLMUL"),
    streams: parse_u8("RSCRYPTO_CRC32_STREAMS"),
    min_bytes_per_lane: parse_usize("RSCRYPTO_CRC32_MIN_BYTES_PER_LANE"),
  };

  let crc32c = VariantOverrides {
    portable_to_hwcrc: parse_usize("RSCRYPTO_CRC32C_THRESHOLD_PORTABLE_TO_HWCRC"),
    hwcrc_to_fusion: parse_usize("RSCRYPTO_CRC32C_THRESHOLD_HWCRC_TO_FUSION"),
    fusion_to_avx512: parse_usize("RSCRYPTO_CRC32C_THRESHOLD_FUSION_TO_AVX512"),
    fusion_to_vpclmul: parse_usize("RSCRYPTO_CRC32C_THRESHOLD_FUSION_TO_VPCLMUL"),
    streams: parse_u8("RSCRYPTO_CRC32C_STREAMS"),
    min_bytes_per_lane: parse_usize("RSCRYPTO_CRC32C_MIN_BYTES_PER_LANE"),
  };

  Overrides {
    force: parse_force("RSCRYPTO_CRC32_FORCE").unwrap_or(Crc32Force::Auto),
    crc32,
    crc32c,
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
#[allow(unused_variables)] // `caps` used on arch-specific cfg paths
fn clamp_force_to_caps(requested: Crc32Force, caps: Caps) -> Crc32Force {
  match requested {
    Crc32Force::Auto | Crc32Force::Reference | Crc32Force::Portable => requested,
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
    Crc32Force::Vpmsum => {
      #[cfg(target_arch = "powerpc64")]
      {
        if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
          return Crc32Force::Vpmsum;
        }
      }
      Crc32Force::Auto
    }
    Crc32Force::Vgfm => {
      #[cfg(target_arch = "s390x")]
      {
        if caps.has(platform::caps::s390x::VECTOR) {
          return Crc32Force::Vgfm;
        }
      }
      Crc32Force::Auto
    }
    Crc32Force::Zbc => {
      #[cfg(target_arch = "riscv64")]
      {
        if caps.has(platform::caps::riscv::ZBC) {
          return Crc32Force::Zbc;
        }
      }
      Crc32Force::Auto
    }
    Crc32Force::Zvbc => {
      #[cfg(target_arch = "riscv64")]
      {
        if caps.has(platform::caps::riscv::ZVBC) {
          return Crc32Force::Zvbc;
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
  // Fusion kernels (HWCRC + carryless-multiply folding) have non-trivial setup
  // costs vs pure HWCRC/table, especially on Intel parts. Treat this threshold
  // as a "SIMD-ish" crossover, not the raw CLMUL crossover.
  let default_hwcrc_to_fusion = tune.pclmul_threshold.max(if tune.fast_wide_ops {
    tune.simd_threshold
  } else {
    tune.simd_threshold.strict_mul(2)
  });
  let default_fusion_to_avx512 = tune.simd_threshold;
  let default_fusion_to_vpclmul = if tune.fast_wide_ops {
    tune.simd_threshold
  } else {
    tune.simd_threshold.saturating_mul(8)
  };

  let default_portable_to_hwcrc = tune.hwcrc_threshold.min(tune.pclmul_threshold);

  let crc32 = tuned
    .map(|t| t.crc32)
    .unwrap_or(tuned_defaults::Crc32VariantTunedDefaults {
      streams: default_streams,
      portable_to_hwcrc: default_portable_to_hwcrc,
      hwcrc_to_fusion: default_hwcrc_to_fusion,
      fusion_to_avx512: default_fusion_to_avx512,
      fusion_to_vpclmul: default_fusion_to_vpclmul,
      min_bytes_per_lane: None,
    });

  let crc32c = tuned
    .map(|t| t.crc32c)
    .unwrap_or(tuned_defaults::Crc32VariantTunedDefaults {
      streams: default_streams,
      portable_to_hwcrc: default_portable_to_hwcrc,
      hwcrc_to_fusion: default_hwcrc_to_fusion,
      fusion_to_avx512: default_fusion_to_avx512,
      fusion_to_vpclmul: default_fusion_to_vpclmul,
      min_bytes_per_lane: None,
    });

  Crc32Tunables {
    crc32: Crc32VariantTunables {
      portable_to_hwcrc: crc32.portable_to_hwcrc,
      hwcrc_to_fusion: crc32.hwcrc_to_fusion,
      fusion_to_avx512: crc32.fusion_to_avx512,
      fusion_to_vpclmul: crc32.fusion_to_vpclmul,
      streams: crc32.streams,
      min_bytes_per_lane: crc32.min_bytes_per_lane,
    },
    crc32c: Crc32VariantTunables {
      portable_to_hwcrc: crc32c.portable_to_hwcrc,
      hwcrc_to_fusion: crc32c.hwcrc_to_fusion,
      fusion_to_avx512: crc32c.fusion_to_avx512,
      fusion_to_vpclmul: crc32c.fusion_to_vpclmul,
      streams: crc32c.streams,
      min_bytes_per_lane: crc32c.min_bytes_per_lane,
    },
  }
}

#[inline]
#[must_use]
fn clamp_streams(mut streams: u8) -> u8 {
  // Clamp streams to the supported name mapping.
  if streams == 0 {
    streams = 1;
  }
  if streams > 8 {
    streams = 8;
  }
  streams
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

  Crc32Config {
    requested_force,
    effective_force,
    tunables: {
      let mut crc32 = base.crc32;
      let mut crc32c = base.crc32c;

      if let Some(v) = ov.crc32.portable_to_hwcrc {
        crc32.portable_to_hwcrc = v;
      }
      if let Some(v) = ov.crc32.hwcrc_to_fusion {
        crc32.hwcrc_to_fusion = v;
      }
      if let Some(v) = ov.crc32.fusion_to_avx512 {
        crc32.fusion_to_avx512 = v;
      }
      if let Some(v) = ov.crc32.fusion_to_vpclmul {
        crc32.fusion_to_vpclmul = v;
      }
      if let Some(v) = ov.crc32.streams {
        crc32.streams = v;
      }
      crc32.streams = clamp_streams(crc32.streams);
      crc32.min_bytes_per_lane = ov.crc32.min_bytes_per_lane.or(crc32.min_bytes_per_lane);

      if let Some(v) = ov.crc32c.portable_to_hwcrc {
        crc32c.portable_to_hwcrc = v;
      }
      if let Some(v) = ov.crc32c.hwcrc_to_fusion {
        crc32c.hwcrc_to_fusion = v;
      }
      if let Some(v) = ov.crc32c.fusion_to_avx512 {
        crc32c.fusion_to_avx512 = v;
      }
      if let Some(v) = ov.crc32c.fusion_to_vpclmul {
        crc32c.fusion_to_vpclmul = v;
      }
      if let Some(v) = ov.crc32c.streams {
        crc32c.streams = v;
      }
      crc32c.streams = clamp_streams(crc32c.streams);
      crc32c.min_bytes_per_lane = ov.crc32c.min_bytes_per_lane.or(crc32c.min_bytes_per_lane);

      Crc32Tunables { crc32, crc32c }
    },
  }
}
