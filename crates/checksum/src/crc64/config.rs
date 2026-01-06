//! CRC-64 runtime configuration.
//!
//! This module handles force mode selection for CRC-64. The dispatch module
//! handles optimal kernel selection automatically; this module only provides
//! the ability to force specific backends for testing/debugging.

use platform::Caps;

/// Forced backend selection for CRC-64.
///
/// This is a *request* that is clamped to the detected CPU capabilities to
/// guarantee safety (e.g., forcing VPCLMUL on a CPU without VPCLMUL will fall
/// back to `Auto` selection).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Crc64Force {
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
  /// Force x86_64 PCLMULQDQ (if supported).
  Pclmul,
  /// Force x86_64 VPCLMULQDQ (if supported).
  Vpclmul,
  /// Force aarch64 PMULL (if supported).
  Pmull,
  /// Force aarch64 PMULL+EOR3 (if supported, requires SHA3).
  PmullEor3,
  /// Force aarch64 SVE2 PMULL (if supported).
  Sve2Pmull,
  /// Force Power VPMSUMD (if supported).
  Vpmsum,
  /// Force s390x VGFM (if supported).
  Vgfm,
  /// Force riscv64 Zbc carryless multiply (if supported).
  Zbc,
  /// Force riscv64 Zvbc (vector carryless multiply) folding (if supported).
  Zvbc,
}

impl Crc64Force {
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Reference => "reference",
      Self::Portable => "portable",
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

  /// Convert to the corresponding [`backend::KernelFamily`], if applicable.
  ///
  /// Returns `None` for `Auto` since it doesn't map to a specific family.
  #[must_use]
  pub const fn to_family(self) -> Option<backend::KernelFamily> {
    use backend::KernelFamily;
    match self {
      Self::Auto => None,
      Self::Reference => Some(KernelFamily::Reference),
      Self::Portable => Some(KernelFamily::Portable),
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

  /// Create from a [`backend::KernelFamily`].
  #[must_use]
  pub const fn from_family(family: backend::KernelFamily) -> Self {
    use backend::KernelFamily;
    match family {
      KernelFamily::Reference => Self::Reference,
      KernelFamily::Portable => Self::Portable,
      KernelFamily::X86Pclmul => Self::Pclmul,
      KernelFamily::X86Vpclmul => Self::Vpclmul,
      KernelFamily::ArmPmull => Self::Pmull,
      KernelFamily::ArmPmullEor3 => Self::PmullEor3,
      KernelFamily::ArmSve2Pmull => Self::Sve2Pmull,
      KernelFamily::PowerVpmsum => Self::Vpmsum,
      KernelFamily::S390xVgfm => Self::Vgfm,
      KernelFamily::RiscvZbc => Self::Zbc,
      KernelFamily::RiscvZvbc => Self::Zvbc,
      // HW CRC families don't apply to CRC-64, fallback to portable
      KernelFamily::X86Crc32 | KernelFamily::ArmCrc32 => Self::Portable,
      // Future families - fallback to portable (non_exhaustive)
      _ => Self::Portable,
    }
  }
}

#[cfg(feature = "std")]
fn parse_force_env() -> Crc64Force {
  let Ok(value) = std::env::var("RSCRYPTO_CRC64_FORCE") else {
    return Crc64Force::Auto;
  };
  let value = value.trim();
  if value.is_empty() {
    return Crc64Force::Auto;
  }

  if value.eq_ignore_ascii_case("auto") {
    return Crc64Force::Auto;
  }
  if value.eq_ignore_ascii_case("reference") || value.eq_ignore_ascii_case("bitwise") {
    return Crc64Force::Reference;
  }
  if value.eq_ignore_ascii_case("portable")
    || value.eq_ignore_ascii_case("scalar")
    || value.eq_ignore_ascii_case("table")
  {
    return Crc64Force::Portable;
  }
  if value.eq_ignore_ascii_case("pclmul") || value.eq_ignore_ascii_case("clmul") {
    return Crc64Force::Pclmul;
  }
  if value.eq_ignore_ascii_case("vpclmul") {
    return Crc64Force::Vpclmul;
  }
  if value.eq_ignore_ascii_case("pmull") || value.eq_ignore_ascii_case("pmull-neon") {
    return Crc64Force::Pmull;
  }
  if value.eq_ignore_ascii_case("pmull-eor3")
    || value.eq_ignore_ascii_case("eor3")
    || value.eq_ignore_ascii_case("pmull-sha3")
  {
    return Crc64Force::PmullEor3;
  }
  if value.eq_ignore_ascii_case("sve2-pmull")
    || value.eq_ignore_ascii_case("pmull-sve2")
    || value.eq_ignore_ascii_case("sve2")
  {
    return Crc64Force::Sve2Pmull;
  }
  if value.eq_ignore_ascii_case("vpmsum") || value.eq_ignore_ascii_case("vpmsumd") {
    return Crc64Force::Vpmsum;
  }
  if value.eq_ignore_ascii_case("vgfm") || value.eq_ignore_ascii_case("gfmsum") {
    return Crc64Force::Vgfm;
  }
  if value.eq_ignore_ascii_case("zbc") {
    return Crc64Force::Zbc;
  }
  if value.eq_ignore_ascii_case("zvbc") || value.eq_ignore_ascii_case("vclmul") {
    return Crc64Force::Zvbc;
  }

  Crc64Force::Auto
}

#[inline]
#[must_use]
#[allow(unused_variables)] // `caps` only used on x86_64/aarch64
fn clamp_force_to_caps(requested: Crc64Force, caps: Caps) -> Crc64Force {
  match requested {
    Crc64Force::Auto | Crc64Force::Reference | Crc64Force::Portable => requested,
    Crc64Force::Pclmul => {
      #[cfg(target_arch = "x86_64")]
      {
        if caps.has(platform::caps::x86::PCLMUL_READY) {
          return Crc64Force::Pclmul;
        }
      }
      Crc64Force::Auto
    }
    Crc64Force::Vpclmul => {
      #[cfg(target_arch = "x86_64")]
      {
        if caps.has(platform::caps::x86::VPCLMUL_READY) {
          return Crc64Force::Vpclmul;
        }
      }
      Crc64Force::Auto
    }
    Crc64Force::Pmull => {
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::PMULL_READY) {
          return Crc64Force::Pmull;
        }
      }
      Crc64Force::Auto
    }
    Crc64Force::PmullEor3 => {
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
          return Crc64Force::PmullEor3;
        }
      }
      Crc64Force::Auto
    }
    Crc64Force::Sve2Pmull => {
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
          return Crc64Force::Sve2Pmull;
        }
      }
      Crc64Force::Auto
    }
    Crc64Force::Vpmsum => {
      #[cfg(target_arch = "powerpc64")]
      {
        if caps.has(platform::caps::power::VPMSUM_READY) {
          return Crc64Force::Vpmsum;
        }
      }
      Crc64Force::Auto
    }
    Crc64Force::Vgfm => {
      #[cfg(target_arch = "s390x")]
      {
        if caps.has(platform::caps::s390x::VECTOR) {
          return Crc64Force::Vgfm;
        }
      }
      Crc64Force::Auto
    }
    Crc64Force::Zbc => {
      #[cfg(target_arch = "riscv64")]
      {
        if caps.has(platform::caps::riscv::ZBC) {
          return Crc64Force::Zbc;
        }
      }
      Crc64Force::Auto
    }
    Crc64Force::Zvbc => {
      #[cfg(target_arch = "riscv64")]
      {
        if caps.has(platform::caps::riscv::ZVBC) {
          return Crc64Force::Zvbc;
        }
      }
      Crc64Force::Auto
    }
  }
}

/// CRC-64 runtime configuration (force mode only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc64Config {
  /// Requested force mode (env/programmatic).
  pub requested_force: Crc64Force,
  /// Force mode clamped to detected CPU capabilities.
  pub effective_force: Crc64Force,
}

/// Compute the effective CRC-64 config for the current process.
///
/// This reads `RSCRYPTO_CRC64_FORCE` and clamps to available CPU features.
#[inline]
#[must_use]
fn config(caps: Caps) -> Crc64Config {
  #[cfg(feature = "std")]
  let requested_force = parse_force_env();
  #[cfg(not(feature = "std"))]
  let requested_force = Crc64Force::Auto;

  let effective_force = clamp_force_to_caps(requested_force, caps);

  Crc64Config {
    requested_force,
    effective_force,
  }
}

/// Cached process-wide CRC-64 configuration.
#[inline]
#[must_use]
pub fn get() -> Crc64Config {
  #[cfg(feature = "std")]
  {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Crc64Config> = OnceLock::new();
    *CACHED.get_or_init(|| config(platform::caps()))
  }

  #[cfg(not(feature = "std"))]
  {
    config(platform::caps())
  }
}
