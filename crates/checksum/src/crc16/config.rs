//! CRC-16 runtime configuration.
//!
//! This module handles force mode selection for CRC-16. The dispatch module
//! handles optimal kernel selection automatically; this module only provides
//! the ability to force specific backends for testing/debugging.

use platform::Caps;

/// Forced backend selection for CRC-16.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Crc16Force {
  /// Use the default auto selector.
  #[default]
  Auto,
  /// Force the bitwise reference implementation (slow, obviously correct).
  Reference,
  /// Force the portable tier (slice-by-8).
  Portable,
  /// Force the carryless-multiply tier (PCLMULQDQ/PMULL) if supported.
  Clmul,
  /// Force the portable slice-by-4 kernel.
  Slice4,
  /// Force the portable slice-by-8 kernel.
  Slice8,
}

impl Crc16Force {
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Reference => "reference",
      Self::Portable => "portable",
      Self::Clmul => "clmul",
      Self::Slice4 => "slice4",
      Self::Slice8 => "slice8",
    }
  }

  /// Map to `KernelFamily` for dispatch.
  #[must_use]
  pub const fn to_family(self) -> Option<backend::KernelFamily> {
    match self {
      Self::Auto | Self::Portable | Self::Slice4 | Self::Slice8 => None,
      Self::Reference => Some(backend::KernelFamily::Reference),
      Self::Clmul => {
        #[cfg(target_arch = "x86_64")]
        {
          Some(backend::KernelFamily::X86Pclmul)
        }
        #[cfg(target_arch = "aarch64")]
        {
          Some(backend::KernelFamily::ArmPmull)
        }
        #[cfg(target_arch = "powerpc64")]
        {
          Some(backend::KernelFamily::PowerVpmsum)
        }
        #[cfg(target_arch = "s390x")]
        {
          Some(backend::KernelFamily::S390xVgfm)
        }
        #[cfg(target_arch = "riscv64")]
        {
          Some(backend::KernelFamily::RiscvZbc)
        }
        #[cfg(not(any(
          target_arch = "x86_64",
          target_arch = "aarch64",
          target_arch = "powerpc64",
          target_arch = "s390x",
          target_arch = "riscv64"
        )))]
        {
          None
        }
      }
    }
  }

  /// Create from `KernelFamily`.
  #[must_use]
  pub const fn from_family(family: backend::KernelFamily) -> Self {
    match family {
      backend::KernelFamily::Reference => Self::Reference,
      backend::KernelFamily::Portable => Self::Portable,
      backend::KernelFamily::X86Pclmul | backend::KernelFamily::X86Vpclmul => Self::Clmul,
      backend::KernelFamily::ArmPmull | backend::KernelFamily::ArmPmullEor3 | backend::KernelFamily::ArmSve2Pmull => {
        Self::Clmul
      }
      backend::KernelFamily::PowerVpmsum
      | backend::KernelFamily::S390xVgfm
      | backend::KernelFamily::RiscvZbc
      | backend::KernelFamily::RiscvZvbc => Self::Clmul,
      _ => Self::Auto,
    }
  }
}

#[cfg(feature = "std")]
fn parse_force_env_ccitt() -> Crc16Force {
  let Ok(value) = std::env::var("RSCRYPTO_CRC16_CCITT_FORCE") else {
    return Crc16Force::Auto;
  };
  parse_force_value(&value)
}

#[cfg(feature = "std")]
fn parse_force_env_ibm() -> Crc16Force {
  let Ok(value) = std::env::var("RSCRYPTO_CRC16_IBM_FORCE") else {
    return Crc16Force::Auto;
  };
  parse_force_value(&value)
}

#[cfg(feature = "std")]
fn parse_force_value(value: &str) -> Crc16Force {
  let value = value.trim();
  if value.is_empty() {
    return Crc16Force::Auto;
  }

  if value.eq_ignore_ascii_case("auto") {
    return Crc16Force::Auto;
  }
  if value.eq_ignore_ascii_case("reference") || value.eq_ignore_ascii_case("bitwise") {
    return Crc16Force::Reference;
  }
  if value.eq_ignore_ascii_case("portable")
    || value.eq_ignore_ascii_case("scalar")
    || value.eq_ignore_ascii_case("table")
  {
    return Crc16Force::Portable;
  }
  if value.eq_ignore_ascii_case("clmul")
    || value.eq_ignore_ascii_case("pclmul")
    || value.eq_ignore_ascii_case("pmull")
    || value.eq_ignore_ascii_case("vpmsum")
    || value.eq_ignore_ascii_case("vgfm")
    || value.eq_ignore_ascii_case("zbc")
    || value.eq_ignore_ascii_case("zvbc")
  {
    return Crc16Force::Clmul;
  }
  if value.eq_ignore_ascii_case("slice4") {
    return Crc16Force::Slice4;
  }
  if value.eq_ignore_ascii_case("slice8") {
    return Crc16Force::Slice8;
  }

  Crc16Force::Auto
}

#[inline]
#[must_use]
#[allow(unused_variables)]
fn clamp_force_to_caps(requested: Crc16Force, caps: Caps) -> Crc16Force {
  match requested {
    Crc16Force::Auto | Crc16Force::Reference | Crc16Force::Portable | Crc16Force::Slice4 | Crc16Force::Slice8 => {
      requested
    }
    Crc16Force::Clmul => {
      #[cfg(target_arch = "x86_64")]
      {
        if caps.has(platform::caps::x86::PCLMUL_READY) {
          return Crc16Force::Clmul;
        }
      }
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::PMULL_READY) {
          return Crc16Force::Clmul;
        }
      }
      #[cfg(target_arch = "powerpc64")]
      {
        if caps.has(platform::caps::power::VPMSUM_READY) {
          return Crc16Force::Clmul;
        }
      }
      #[cfg(target_arch = "s390x")]
      {
        if caps.has(platform::caps::s390x::VECTOR) {
          return Crc16Force::Clmul;
        }
      }
      #[cfg(target_arch = "riscv64")]
      {
        use platform::caps::riscv;
        if caps.has(riscv::ZBC) || caps.has(riscv::ZVBC) {
          return Crc16Force::Clmul;
        }
      }
      Crc16Force::Auto
    }
  }
}

/// CRC-16 runtime configuration (force mode only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc16Config {
  /// Requested force mode (env/programmatic).
  pub requested_force: Crc16Force,
  /// Force mode clamped to detected CPU capabilities.
  pub effective_force: Crc16Force,
}

/// Compute the effective CRC-16/CCITT config for the current process.
#[inline]
#[must_use]
fn config_ccitt(caps: Caps) -> Crc16Config {
  #[cfg(feature = "std")]
  let requested_force = parse_force_env_ccitt();
  #[cfg(not(feature = "std"))]
  let requested_force = Crc16Force::Auto;

  let effective_force = clamp_force_to_caps(requested_force, caps);

  Crc16Config {
    requested_force,
    effective_force,
  }
}

/// Compute the effective CRC-16/IBM config for the current process.
#[inline]
#[must_use]
fn config_ibm(caps: Caps) -> Crc16Config {
  #[cfg(feature = "std")]
  let requested_force = parse_force_env_ibm();
  #[cfg(not(feature = "std"))]
  let requested_force = Crc16Force::Auto;

  let effective_force = clamp_force_to_caps(requested_force, caps);

  Crc16Config {
    requested_force,
    effective_force,
  }
}

/// Cached process-wide CRC-16/CCITT configuration.
#[inline]
#[must_use]
pub fn get_ccitt() -> Crc16Config {
  #[cfg(feature = "std")]
  {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Crc16Config> = OnceLock::new();
    *CACHED.get_or_init(|| config_ccitt(platform::caps()))
  }

  #[cfg(not(feature = "std"))]
  {
    config_ccitt(platform::caps())
  }
}

/// Cached process-wide CRC-16/IBM configuration.
#[inline]
#[must_use]
pub fn get_ibm() -> Crc16Config {
  #[cfg(feature = "std")]
  {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Crc16Config> = OnceLock::new();
    *CACHED.get_or_init(|| config_ibm(platform::caps()))
  }

  #[cfg(not(feature = "std"))]
  {
    config_ibm(platform::caps())
  }
}
