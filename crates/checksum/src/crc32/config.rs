//! CRC-32 runtime configuration (overrides + thresholds).
//!
//! This module centralizes all CRC-32 selection knobs so:
//! - Dispatch remains safe (never executes unsupported instructions)
//! - Benchmarks/tests can force specific tiers
//! - Introspection can report the active configuration without allocation

use platform::{Caps, Tune};

use super::tuned_defaults;

/// Forced backend selection for CRC-32.
///
/// This is a *request* that is clamped to the detected CPU capabilities to
/// guarantee safety (e.g., forcing SSE4.2 on a CPU without SSE4.2 will fall
/// back to `Auto` selection).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Crc32Force {
  /// Use the default auto selector.
  #[default]
  Auto,
  /// Force the portable table-based implementation.
  Portable,
  /// Force x86_64 SSE4.2 crc32 instruction (CRC32C only).
  Sse42,
  /// Force x86_64 PCLMULQDQ.
  Pclmul,
  /// Force x86_64 VPCLMULQDQ.
  Vpclmul,
  /// Force aarch64 CRC32 extension (CRC32C only).
  ArmCrc,
  /// Force aarch64 PMULL.
  Pmull,
  /// Force aarch64 PMULL+EOR3 (requires SHA3).
  PmullEor3,
}

impl Crc32Force {
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Portable => "portable",
      Self::Sse42 => "sse4.2",
      Self::Pclmul => "pclmul",
      Self::Vpclmul => "vpclmul",
      Self::ArmCrc => "arm-crc",
      Self::Pmull => "pmull",
      Self::PmullEor3 => "pmull-eor3",
    }
  }
}

/// CRC-32 selection tunables (thresholds + parallelism).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32Tunables {
  /// Bytes where hardware CRC instruction becomes faster than portable.
  pub portable_to_hw: usize,
  /// Bytes where CLMUL/PMULL becomes faster than hardware CRC instruction.
  pub hw_to_clmul: usize,
  /// Bytes where VPCLMUL becomes faster than PCLMUL on wide-SIMD CPUs.
  pub pclmul_to_vpclmul: usize,
  /// Preferred number of independent folding streams for large buffers.
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
  portable_to_hw: Option<usize>,
  hw_to_clmul: Option<usize>,
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
    if value.eq_ignore_ascii_case("sse42")
      || value.eq_ignore_ascii_case("sse4.2")
      || value.eq_ignore_ascii_case("crc32")
    {
      return Some(Crc32Force::Sse42);
    }
    if value.eq_ignore_ascii_case("pclmul") || value.eq_ignore_ascii_case("clmul") {
      return Some(Crc32Force::Pclmul);
    }
    if value.eq_ignore_ascii_case("vpclmul") {
      return Some(Crc32Force::Vpclmul);
    }
    if value.eq_ignore_ascii_case("arm-crc") || value.eq_ignore_ascii_case("armcrc") {
      return Some(Crc32Force::ArmCrc);
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

    None
  }

  Overrides {
    force: parse_force("RSCRYPTO_CRC32_FORCE").unwrap_or(Crc32Force::Auto),
    portable_to_hw: parse_usize("RSCRYPTO_CRC32_THRESHOLD_PORTABLE_TO_HW"),
    hw_to_clmul: parse_usize("RSCRYPTO_CRC32_THRESHOLD_HW_TO_CLMUL"),
    pclmul_to_vpclmul: parse_usize("RSCRYPTO_CRC32_THRESHOLD_PCLMUL_TO_VPCLMUL"),
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
    Crc32Force::Sse42 => {
      #[cfg(target_arch = "x86_64")]
      {
        if caps.has(platform::caps::x86::SSE42) {
          return Crc32Force::Sse42;
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
    Crc32Force::ArmCrc => {
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::CRC) {
          return Crc32Force::ArmCrc;
        }
      }
      Crc32Force::Auto
    }
    Crc32Force::Pmull => {
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::PMULL_READY) {
          return Crc32Force::Pmull;
        }
      }
      Crc32Force::Auto
    }
    Crc32Force::PmullEor3 => {
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
          return Crc32Force::PmullEor3;
        }
      }
      Crc32Force::Auto
    }
  }
}

#[inline]
#[must_use]
fn default_pclmul_to_vpclmul_threshold(caps: Caps, tune: Tune) -> usize {
  #[cfg(target_arch = "x86_64")]
  {
    if !caps.has(platform::caps::x86::VPCLMUL_READY) || tune.effective_simd_width < 512 {
      return usize::MAX;
    }
    if tune.fast_wide_ops { 1024 } else { 16 * 1024 }
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
  streams.clamp(1, 8)
}

#[inline]
#[must_use]
#[allow(unused_variables)]
fn default_crc32_streams(caps: Caps, tune: Tune) -> u8 {
  #[cfg(target_arch = "x86_64")]
  {
    let _ = caps;
    match tune.parallel_streams {
      0 | 1 => 1,
      2 => 2,
      3 | 4 => 4,
      _ => 4,
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    let _ = caps;
    tune.parallel_streams.clamp(1, 3)
  }

  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  {
    let _ = (caps, tune);
    1
  }
}

/// Compute the effective CRC-32 config for the current process.
#[inline]
#[must_use]
pub fn config(caps: Caps, tune: Tune) -> Crc32Config {
  let ov = overrides();

  let tuned = tuned_defaults::for_tune_kind(tune.kind);

  // Default thresholds: HW CRC is very fast for small buffers
  // CLMUL only wins for larger buffers due to setup overhead
  let portable_to_hw = ov.portable_to_hw.or(tuned.map(|t| t.portable_to_hw)).unwrap_or(8);

  // HW CRC instruction is ~20GB/s, CLMUL can be faster for large buffers
  let hw_to_clmul = ov.hw_to_clmul.or(tuned.map(|t| t.hw_to_clmul)).unwrap_or(512);

  let pclmul_to_vpclmul = ov
    .pclmul_to_vpclmul
    .or(tuned.map(|t| t.pclmul_to_vpclmul))
    .unwrap_or_else(|| default_pclmul_to_vpclmul_threshold(caps, tune));

  let streams = clamp_streams(
    ov.streams
      .or(tuned.map(|t| t.streams))
      .unwrap_or_else(|| default_crc32_streams(caps, tune)),
  );

  let requested_force = ov.force;
  let effective_force = clamp_force_to_caps(requested_force, caps);

  Crc32Config {
    requested_force,
    effective_force,
    tunables: Crc32Tunables {
      portable_to_hw,
      hw_to_clmul,
      pclmul_to_vpclmul,
      streams,
    },
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
    *CACHED.get_or_init(|| platform::dispatch_auto(config))
  }

  #[cfg(not(feature = "std"))]
  {
    config(platform::caps(), platform::tune())
  }
}
