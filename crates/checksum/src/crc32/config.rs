//! CRC-32 runtime configuration (overrides + thresholds).
//!
//! This module centralizes selection knobs for CRC-32/CRC-32C:
//! - portable vs hardware instruction thresholds
//! - optional forced backend selection
//!
//! Safety note: forced modes are always clamped to detected CPU capabilities.

use platform::{Caps, Tune};

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
}

impl Crc32Force {
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Portable => "portable",
      Self::Hwcrc => "hwcrc",
    }
  }
}

/// CRC-32 selection tunables (thresholds + parallelism).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32Tunables {
  /// Bytes where hardware CRC becomes faster than portable.
  pub portable_to_hwcrc: usize,
  /// Preferred number of independent streams (reserved for future fusion kernels).
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

    None
  }

  Overrides {
    force: parse_force("RSCRYPTO_CRC32_FORCE").unwrap_or(Crc32Force::Auto),
    portable_to_hwcrc: parse_usize("RSCRYPTO_CRC32_THRESHOLD_PORTABLE_TO_HWCRC"),
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
  }
}

#[inline]
#[must_use]
fn tuned_defaults(caps: Caps, tune: Tune) -> Crc32Tunables {
  let _ = caps;
  Crc32Tunables {
    portable_to_hwcrc: tune.hwcrc_threshold,
    streams: 1,
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
      streams,
    },
  }
}
