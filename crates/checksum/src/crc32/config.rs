//! CRC-32 runtime configuration.
//!
//! This module handles force mode selection for CRC-32. The dispatch module
//! handles optimal kernel selection automatically; this module only provides
//! the ability to force specific backends for testing/debugging.

use platform::Caps;

/// Forced backend selection for CRC-32/CRC-32C.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Crc32Force {
  /// Use the default auto selector.
  #[default]
  Auto,
  /// Force the bitwise reference implementation (slow, obviously correct).
  Reference,
  /// Force the portable table-based implementation.
  Portable,
  /// Force hardware CRC instructions (if available).
  Hwcrc,
  /// Force x86_64 CRC-32C fusion kernels (SSE4.2 + PCLMULQDQ) if supported.
  Pclmul,
  /// Force x86_64 CRC-32C fusion kernels (AVX-512 + VPCLMULQDQ) if supported.
  Vpclmul,
  /// Force aarch64 CRC fusion kernels (CRC + PMULL) if supported.
  Pmull,
  /// Force aarch64 CRC fusion kernels (CRC + PMULL + EOR3/SHA3) if supported.
  PmullEor3,
  /// Force aarch64 "SVE2 PMULL" tier if supported.
  Sve2Pmull,
  /// Force Power VPMSUMD folding (if supported).
  Vpmsum,
  /// Force s390x VGFM folding (if supported).
  Vgfm,
  /// Force riscv64 Zbc carryless multiply folding (if supported).
  Zbc,
  /// Force riscv64 Zvbc folding (if supported).
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
      _ => Self::Portable,
    }
  }
}

#[cfg(feature = "std")]
fn parse_force_env() -> Crc32Force {
  let Ok(value) = std::env::var("RSCRYPTO_CRC32_FORCE") else {
    return Crc32Force::Auto;
  };
  let value = value.trim();
  if value.is_empty() {
    return Crc32Force::Auto;
  }

  if value.eq_ignore_ascii_case("auto") {
    return Crc32Force::Auto;
  }
  if value.eq_ignore_ascii_case("reference") || value.eq_ignore_ascii_case("bitwise") {
    return Crc32Force::Reference;
  }
  if value.eq_ignore_ascii_case("portable")
    || value.eq_ignore_ascii_case("scalar")
    || value.eq_ignore_ascii_case("table")
  {
    return Crc32Force::Portable;
  }
  if value.eq_ignore_ascii_case("hwcrc") || value.eq_ignore_ascii_case("crc") {
    return Crc32Force::Hwcrc;
  }
  if value.eq_ignore_ascii_case("pclmul") || value.eq_ignore_ascii_case("clmul") {
    return Crc32Force::Pclmul;
  }
  if value.eq_ignore_ascii_case("vpclmul") {
    return Crc32Force::Vpclmul;
  }
  if value.eq_ignore_ascii_case("pmull") {
    return Crc32Force::Pmull;
  }
  if value.eq_ignore_ascii_case("pmull-eor3") || value.eq_ignore_ascii_case("eor3") {
    return Crc32Force::PmullEor3;
  }
  if value.eq_ignore_ascii_case("sve2-pmull") || value.eq_ignore_ascii_case("sve2") {
    return Crc32Force::Sve2Pmull;
  }
  if value.eq_ignore_ascii_case("vpmsum") {
    return Crc32Force::Vpmsum;
  }
  if value.eq_ignore_ascii_case("vgfm") {
    return Crc32Force::Vgfm;
  }
  if value.eq_ignore_ascii_case("zbc") {
    return Crc32Force::Zbc;
  }
  if value.eq_ignore_ascii_case("zvbc") {
    return Crc32Force::Zvbc;
  }

  Crc32Force::Auto
}

#[inline]
#[must_use]
#[allow(unused_variables)]
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
        if caps.has(platform::caps::power::VPMSUM_READY) {
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

/// CRC-32 runtime configuration (force mode only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32Config {
  /// Requested force mode (env/programmatic).
  pub requested_force: Crc32Force,
  /// Force mode clamped to detected CPU capabilities.
  pub effective_force: Crc32Force,
}

/// Compute the effective CRC-32 config for the current process.
#[inline]
#[must_use]
fn config(caps: Caps) -> Crc32Config {
  #[cfg(feature = "std")]
  let requested_force = parse_force_env();
  #[cfg(not(feature = "std"))]
  let requested_force = Crc32Force::Auto;

  let effective_force = clamp_force_to_caps(requested_force, caps);

  Crc32Config {
    requested_force,
    effective_force,
  }
}

/// Cached process-wide CRC-32 configuration.
#[inline]
#[must_use]
pub fn get() -> Crc32Config {
  #[cfg(feature = "std")]
  {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Crc32Config> = OnceLock::new();
    *CACHED.get_or_init(|| config(platform::caps()))
  }

  #[cfg(not(feature = "std"))]
  {
    config(platform::caps())
  }
}
