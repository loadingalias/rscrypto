//! CRC-16 runtime configuration.
//!
//! This module handles force mode selection for CRC-16. The dispatch module
//! handles optimal kernel selection automatically; this module only provides
//! the ability to force specific backends for testing/debugging.

use crate::platform::Caps;

/// Requested backend override for CRC-16.
///
/// Requests are clamped to detected CPU capabilities before use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Crc16Force {
  /// Use the default auto selector.
  #[default]
  Auto,
  /// Force the bitwise reference implementation.
  ///
  /// This is the correctness oracle used to validate optimized kernels.
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
  /// Return the stable selector name used by debug and configuration surfaces.
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
        if caps.has(crate::platform::caps::x86::PCLMUL_READY) {
          return Crc16Force::Clmul;
        }
      }
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(crate::platform::caps::aarch64::PMULL_READY) {
          return Crc16Force::Clmul;
        }
      }
      #[cfg(target_arch = "powerpc64")]
      {
        if caps.has(crate::platform::caps::power::VPMSUM_READY) {
          return Crc16Force::Clmul;
        }
      }
      #[cfg(target_arch = "s390x")]
      {
        if caps.has(crate::platform::caps::s390x::VECTOR) {
          return Crc16Force::Clmul;
        }
      }
      #[cfg(target_arch = "riscv64")]
      {
        use crate::platform::caps::riscv;
        if caps.has(riscv::ZBC) || caps.has(riscv::ZVBC) {
          return Crc16Force::Clmul;
        }
      }
      Crc16Force::Auto
    }
  }
}

/// Resolved CRC-16 force configuration.
///
/// `requested_force` is the user or environment request. `effective_force` is
/// the same request clamped to the current platform capabilities.
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

/// Return the cached process-wide CRC-16/CCITT configuration.
///
/// Under `std`, this reflects `RSCRYPTO_CRC16_CCITT_FORCE` clamped to the
/// detected platform capabilities.
#[inline]
#[must_use]
pub fn get_ccitt() -> Crc16Config {
  #[cfg(feature = "std")]
  {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Crc16Config> = OnceLock::new();
    *CACHED.get_or_init(|| config_ccitt(crate::platform::caps()))
  }

  #[cfg(not(feature = "std"))]
  {
    config_ccitt(crate::platform::caps())
  }
}

/// Return the cached process-wide CRC-16/IBM configuration.
///
/// Under `std`, this reflects `RSCRYPTO_CRC16_IBM_FORCE` clamped to the
/// detected platform capabilities.
#[inline]
#[must_use]
pub fn get_ibm() -> Crc16Config {
  #[cfg(feature = "std")]
  {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Crc16Config> = OnceLock::new();
    *CACHED.get_or_init(|| config_ibm(crate::platform::caps()))
  }

  #[cfg(not(feature = "std"))]
  {
    config_ibm(crate::platform::caps())
  }
}
