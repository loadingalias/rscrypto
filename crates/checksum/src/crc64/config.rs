//! CRC-64 runtime configuration (overrides + thresholds).
//!
//! This module centralizes all CRC-64 selection knobs so:
//! - Dispatch remains safe (never executes unsupported instructions)
//! - Benchmarks/tests can force specific tiers
//! - Introspection can report the active configuration without allocation

use platform::{Caps, Tune};

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
  /// Force the portable table-based implementation.
  Portable,
  /// Force x86_64 PCLMULQDQ (if supported).
  Pclmul,
  /// Force x86_64 VPCLMULQDQ (if supported).
  Vpclmul,
  /// Force aarch64 PMULL (if supported).
  Pmull,
  /// Force aarch64 SVE2 PMULL (if supported).
  Sve2Pmull,
}

impl Crc64Force {
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Portable => "portable",
      Self::Pclmul => "pclmul",
      Self::Vpclmul => "vpclmul",
      Self::Pmull => "pmull",
      Self::Sve2Pmull => "sve2-pmull",
    }
  }
}

/// CRC-64 selection tunables (thresholds + parallelism).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc64Tunables {
  /// Bytes where CLMUL/PMULL becomes faster than portable.
  pub portable_to_clmul: usize,
  /// Bytes where VPCLMUL becomes faster than PCLMUL on wide-SIMD CPUs.
  pub pclmul_to_vpclmul: usize,
  /// Preferred number of independent folding streams for large buffers.
  pub streams: u8,
}

/// Full CRC-64 runtime configuration (after applying overrides).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc64Config {
  /// Requested force mode (env/programmatic).
  pub requested_force: Crc64Force,
  /// Force mode clamped to detected CPU capabilities.
  pub effective_force: Crc64Force,
  /// Thresholds and stream settings used by the selector.
  pub tunables: Crc64Tunables,
}

#[derive(Clone, Copy, Debug, Default)]
struct Overrides {
  force: Crc64Force,
  portable_to_clmul: Option<usize>,
  pclmul_to_vpclmul: Option<usize>,
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

  fn parse_force(name: &str) -> Option<Crc64Force> {
    let value = std::env::var(name).ok()?;
    let value = value.trim();
    if value.is_empty() {
      return None;
    }

    if value.eq_ignore_ascii_case("auto") {
      return Some(Crc64Force::Auto);
    }
    if value.eq_ignore_ascii_case("portable")
      || value.eq_ignore_ascii_case("scalar")
      || value.eq_ignore_ascii_case("table")
    {
      return Some(Crc64Force::Portable);
    }
    if value.eq_ignore_ascii_case("pclmul") || value.eq_ignore_ascii_case("clmul") {
      return Some(Crc64Force::Pclmul);
    }
    if value.eq_ignore_ascii_case("vpclmul") {
      return Some(Crc64Force::Vpclmul);
    }
    if value.eq_ignore_ascii_case("pmull") || value.eq_ignore_ascii_case("pmull-neon") {
      return Some(Crc64Force::Pmull);
    }
    if value.eq_ignore_ascii_case("sve2-pmull")
      || value.eq_ignore_ascii_case("pmull-sve2")
      || value.eq_ignore_ascii_case("sve2")
    {
      return Some(Crc64Force::Sve2Pmull);
    }

    None
  }

  Overrides {
    force: parse_force("RSCRYPTO_CRC64_FORCE").unwrap_or(Crc64Force::Auto),
    portable_to_clmul: parse_usize("RSCRYPTO_CRC64_THRESHOLD_PORTABLE_TO_CLMUL"),
    pclmul_to_vpclmul: parse_usize("RSCRYPTO_CRC64_THRESHOLD_PCLMUL_TO_VPCLMUL"),
    streams: parse_u8("RSCRYPTO_CRC64_STREAMS"),
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
fn clamp_force_to_caps(requested: Crc64Force, caps: Caps) -> Crc64Force {
  match requested {
    Crc64Force::Auto | Crc64Force::Portable => requested,
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
    Crc64Force::Sve2Pmull => {
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
          return Crc64Force::Sve2Pmull;
        }
      }
      Crc64Force::Auto
    }
  }
}

#[inline]
#[must_use]
fn default_pclmul_to_vpclmul_threshold(caps: Caps, tune: Tune) -> usize {
  #[cfg(target_arch = "x86_64")]
  {
    // If VPCLMUL isn't available or the tune prefers narrow vectors, disable the tier.
    if !caps.has(platform::caps::x86::VPCLMUL_READY) || tune.effective_simd_width < 512 {
      return usize::MAX;
    }

    // Heuristic default:
    // - Fast wide ops (Zen4/5): low crossover.
    // - Slow warmup (many Intel parts): require much larger buffers.
    if tune.fast_wide_ops {
      return 1024;
    }
    return 16 * 1024;
  }

  #[cfg(not(target_arch = "x86_64"))]
  {
    let _ = (caps, tune);
    usize::MAX
  }
}

#[inline]
#[must_use]
fn clamp_streams(streams: u8) -> u8 {
  // Keep this conservative: callers can override higher, but the selector is
  // free to clamp further per-backend.
  streams.clamp(1, 16)
}

/// Compute the effective CRC-64 config for the current process.
///
/// This merges:
/// - the detected [`Tune`] preset
/// - env var overrides (`RSCRYPTO_CRC64_*`) when `std` is enabled
/// - and clamps unsafe force requests to available CPU features
#[inline]
#[must_use]
pub fn config(caps: Caps, tune: Tune) -> Crc64Config {
  let ov = overrides();

  let portable_to_clmul = ov.portable_to_clmul.unwrap_or(tune.pclmul_threshold);
  let pclmul_to_vpclmul = ov
    .pclmul_to_vpclmul
    .unwrap_or_else(|| default_pclmul_to_vpclmul_threshold(caps, tune));

  let streams = clamp_streams(ov.streams.unwrap_or(tune.parallel_streams));

  let requested_force = ov.force;
  let effective_force = clamp_force_to_caps(requested_force, caps);

  Crc64Config {
    requested_force,
    effective_force,
    tunables: Crc64Tunables {
      portable_to_clmul,
      pclmul_to_vpclmul,
      streams,
    },
  }
}

/// Cached process-wide CRC-64 configuration.
///
/// The configuration depends on:
/// - detected CPU capabilities/tuning (fixed for the lifetime of the process)
/// - env var overrides (expected to be set before process start)
#[inline]
#[must_use]
pub fn get() -> Crc64Config {
  #[cfg(feature = "std")]
  {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Crc64Config> = OnceLock::new();
    *CACHED.get_or_init(|| config(platform::caps(), platform::tune()))
  }

  #[cfg(not(feature = "std"))]
  {
    config(platform::caps(), platform::tune())
  }
}
