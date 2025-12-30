//! CRC-16 runtime configuration (portable tunables + overrides).
//!
//! CRC-16 is currently portable-only, but this module mirrors the CRC-32/CRC-64
//! configuration surface so we can add accelerated tiers without breaking API.

use platform::{Caps, Tune};

use super::tuned_defaults;

/// Forced backend selection for CRC-16.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Crc16Force {
  /// Use the default auto selector.
  #[default]
  Auto,
  /// Force the bitwise reference implementation (slow, obviously correct).
  Reference,
  /// Force the portable tier (reserved for future accelerated backends).
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
}

/// CRC-16 selection tunables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc16Tunables {
  /// Minimum `len` in bytes to use slice-by-8 (otherwise slice-by-4).
  pub slice4_to_slice8: usize,
  /// Minimum `len` in bytes to use the CLMUL/PMULL tier (otherwise portable).
  pub portable_to_clmul: usize,
}

/// Full CRC-16 runtime configuration (after applying overrides).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc16Config {
  /// Requested force mode (env/programmatic).
  pub requested_force: Crc16Force,
  /// Force mode clamped to detected CPU capabilities.
  ///
  /// Today this is identical to `requested_force` because there are no
  /// accelerated tiers to clamp against.
  pub effective_force: Crc16Force,
  /// Tunables used by the selector.
  pub tunables: Crc16Tunables,
}

#[derive(Clone, Copy, Debug, Default)]
struct Overrides {
  force: Crc16Force,
  slice4_to_slice8: Option<usize>,
  portable_to_clmul: Option<usize>,
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

  fn parse_force(name: &str) -> Option<Crc16Force> {
    let value = std::env::var(name).ok()?;
    let value = value.trim();
    if value.is_empty() {
      return None;
    }

    if value.eq_ignore_ascii_case("auto") {
      return Some(Crc16Force::Auto);
    }
    if value.eq_ignore_ascii_case("reference") || value.eq_ignore_ascii_case("bitwise") {
      return Some(Crc16Force::Reference);
    }
    if value.eq_ignore_ascii_case("portable")
      || value.eq_ignore_ascii_case("scalar")
      || value.eq_ignore_ascii_case("table")
    {
      return Some(Crc16Force::Portable);
    }
    if value.eq_ignore_ascii_case("clmul")
      || value.eq_ignore_ascii_case("pclmul")
      || value.eq_ignore_ascii_case("pclmulqdq")
      || value.eq_ignore_ascii_case("pmull")
    {
      return Some(Crc16Force::Clmul);
    }
    if value.eq_ignore_ascii_case("slice4") || value.eq_ignore_ascii_case("slice-4") {
      return Some(Crc16Force::Slice4);
    }
    if value.eq_ignore_ascii_case("slice8") || value.eq_ignore_ascii_case("slice-8") {
      return Some(Crc16Force::Slice8);
    }

    None
  }

  Overrides {
    force: parse_force("RSCRYPTO_CRC16_FORCE").unwrap_or(Crc16Force::Auto),
    slice4_to_slice8: parse_usize("RSCRYPTO_CRC16_THRESHOLD_SLICE4_TO_SLICE8"),
    portable_to_clmul: parse_usize("RSCRYPTO_CRC16_THRESHOLD_PORTABLE_TO_CLMUL"),
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
      Crc16Force::Auto
    }
  }
}

#[inline]
#[must_use]
fn default_slice4_to_slice8(tune: Tune) -> usize {
  tuned_defaults::for_tune_kind(tune.kind)
    .map(|d| d.slice4_to_slice8)
    .unwrap_or(tune.cache_line as usize)
    .max(1)
}

#[inline]
#[must_use]
fn default_portable_to_clmul(tune: Tune) -> usize {
  tuned_defaults::for_tune_kind(tune.kind)
    .map(|d| d.portable_to_clmul)
    .unwrap_or(tune.pclmul_threshold)
    .max(1)
}

#[inline]
#[must_use]
fn config(caps: Caps, tune: Tune) -> Crc16Config {
  let ov = overrides();

  let mut slice4_to_slice8 = default_slice4_to_slice8(tune);
  if let Some(v) = ov.slice4_to_slice8 {
    slice4_to_slice8 = v.max(1);
  }

  let mut portable_to_clmul = default_portable_to_clmul(tune);
  if let Some(v) = ov.portable_to_clmul {
    portable_to_clmul = v.max(1);
  }

  let requested_force = ov.force;
  let effective_force = clamp_force_to_caps(requested_force, caps);

  Crc16Config {
    requested_force,
    effective_force,
    tunables: Crc16Tunables {
      slice4_to_slice8,
      portable_to_clmul,
    },
  }
}

/// Cached process-wide CRC-16 configuration.
#[inline]
#[must_use]
pub fn get() -> Crc16Config {
  #[cfg(feature = "std")]
  {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Crc16Config> = OnceLock::new();
    *CACHED.get_or_init(|| platform::dispatch_auto(config))
  }

  #[cfg(not(feature = "std"))]
  {
    config(platform::caps(), platform::tune())
  }
}
