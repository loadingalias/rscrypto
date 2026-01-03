//! CRC-64 runtime configuration (overrides + thresholds).
//!
//! This module centralizes all CRC-64 selection knobs so:
//! - Dispatch remains safe (never executes unsupported instructions)
//! - Benchmarks/tests can force specific tiers
//! - Introspection can report the active configuration without allocation

use platform::{Caps, Tune};

use super::tuned_defaults;

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

  /// Convert to the corresponding [`KernelFamily`], if applicable.
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

  /// Create from a [`KernelFamily`].
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

/// CRC-64 selection tunables (thresholds + parallelism).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc64VariantTunables {
  /// Bytes where CLMUL/PMULL becomes faster than portable.
  pub portable_to_clmul: usize,
  /// Bytes where VPCLMUL becomes faster than PCLMUL on wide-SIMD CPUs.
  pub pclmul_to_vpclmul: usize,
  /// Preferred number of independent folding streams for large buffers.
  pub streams: u8,
  /// Minimum bytes per lane for multi-stream folding.
  ///
  /// When `None`, uses `KernelFamily::min_bytes_per_lane()` as the default.
  pub min_bytes_per_lane: Option<usize>,
}

/// CRC-64 selection tunables (per-variant thresholds + parallelism).
///
/// CRC-64-XZ and CRC-64-NVME can have different stream and crossover behavior
/// on the same CPU. Keep tunables separate so tuning can be applied losslessly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc64Tunables {
  pub xz: Crc64VariantTunables,
  pub nvme: Crc64VariantTunables,
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
  min_bytes_per_lane: Option<usize>,
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
    if value.eq_ignore_ascii_case("reference") || value.eq_ignore_ascii_case("bitwise") {
      return Some(Crc64Force::Reference);
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
    if value.eq_ignore_ascii_case("pmull-eor3")
      || value.eq_ignore_ascii_case("eor3")
      || value.eq_ignore_ascii_case("pmull-sha3")
    {
      return Some(Crc64Force::PmullEor3);
    }
    if value.eq_ignore_ascii_case("sve2-pmull")
      || value.eq_ignore_ascii_case("pmull-sve2")
      || value.eq_ignore_ascii_case("sve2")
    {
      return Some(Crc64Force::Sve2Pmull);
    }
    if value.eq_ignore_ascii_case("vpmsum") || value.eq_ignore_ascii_case("vpmsumd") {
      return Some(Crc64Force::Vpmsum);
    }
    if value.eq_ignore_ascii_case("vgfm") || value.eq_ignore_ascii_case("gfmsum") {
      return Some(Crc64Force::Vgfm);
    }
    if value.eq_ignore_ascii_case("zbc") {
      return Some(Crc64Force::Zbc);
    }
    if value.eq_ignore_ascii_case("zvbc") || value.eq_ignore_ascii_case("vclmul") {
      return Some(Crc64Force::Zvbc);
    }

    None
  }

  Overrides {
    force: parse_force("RSCRYPTO_CRC64_FORCE").unwrap_or(Crc64Force::Auto),
    portable_to_clmul: parse_usize("RSCRYPTO_CRC64_THRESHOLD_PORTABLE_TO_CLMUL"),
    pclmul_to_vpclmul: parse_usize("RSCRYPTO_CRC64_THRESHOLD_PCLMUL_TO_VPCLMUL"),
    streams: parse_u8("RSCRYPTO_CRC64_STREAMS"),
    min_bytes_per_lane: parse_usize("RSCRYPTO_CRC64_MIN_BYTES_PER_LANE"),
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

#[inline]
#[must_use]
fn default_pclmul_to_vpclmul_threshold(caps: Caps, tune: Tune) -> usize {
  #[cfg(target_arch = "x86_64")]
  {
    // If VPCLMUL isn't available or the tune prefers narrow vectors, disable the tier.
    if !caps.has(platform::caps::x86::VPCLMUL_READY) || tune.effective_simd_width < 512 {
      return usize::MAX;
    }

    // Heuristic default based on benchmarks vs crc-fast-rust:
    // - Fast wide ops (Zen4/5): low crossover (512B is sufficient).
    // - Slow warmup (Intel SPR/GNR): 2KB is the practical crossover where VPCLMUL wins despite ZMM
    //   warmup overhead. At 4KB, VPCLMUL is clearly faster.
    if tune.fast_wide_ops { 512 } else { 2 * 1024 }
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

#[inline]
#[must_use]
#[allow(unused_variables)] // arch-specific
fn default_crc64_streams(caps: Caps, tune: Tune) -> u8 {
  // Default stream preference is architecture- and backend-dependent. We bias
  // slightly toward higher ILP on x86 (where our supported stream counts are
  // 1/2/4/7/8) to avoid mapping common "3" presets down to 2-way.
  // 8-way is the Intel white paper / Linux kernel standard.
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
    // POWER VPMSUMD benefits from higher ILP; default to a higher stream count
    // when the crypto/vector facility is present.
    if !caps.has(platform::caps::power::VPMSUM_READY) {
      return 1;
    }
    tune.parallel_streams.saturating_mul(2).clamp(1, 8)
  }

  #[cfg(target_arch = "s390x")]
  {
    if !caps.has(platform::caps::s390x::VECTOR) {
      return 1;
    }
    // Similar strategy to POWER: use extra ILP to hide GF multiply latency.
    tune.parallel_streams.saturating_mul(2).clamp(1, 4)
  }

  #[cfg(target_arch = "riscv64")]
  {
    if !(caps.has(platform::caps::riscv::ZVBC) || caps.has(platform::caps::riscv::ZBC)) {
      return 1;
    }

    // Our RISC-V backends support 1/2/4-way folding; bias slightly upward
    // (like x86) so the default tune preset (3 streams) doesn't map down to 2-way.
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

  let tuned = tuned_defaults::for_tune_kind(tune.kind);

  let default_portable_to_clmul = tune.pclmul_threshold;
  let default_pclmul_to_vpclmul = default_pclmul_to_vpclmul_threshold(caps, tune);
  let default_streams = default_crc64_streams(caps, tune);

  let xz_streams = clamp_streams(ov.streams.or(tuned.map(|t| t.xz.streams)).unwrap_or(default_streams));
  let nvme_streams = clamp_streams(ov.streams.or(tuned.map(|t| t.nvme.streams)).unwrap_or(default_streams));

  let xz = Crc64VariantTunables {
    portable_to_clmul: ov
      .portable_to_clmul
      .or(tuned.map(|t| t.xz.portable_to_clmul))
      .unwrap_or(default_portable_to_clmul),
    pclmul_to_vpclmul: ov
      .pclmul_to_vpclmul
      .or(tuned.map(|t| t.xz.pclmul_to_vpclmul))
      .unwrap_or(default_pclmul_to_vpclmul),
    streams: xz_streams,
    // min_bytes_per_lane: env var > tuned defaults > None (use family default in policy)
    min_bytes_per_lane: ov.min_bytes_per_lane.or(tuned.and_then(|t| t.xz.min_bytes_per_lane)),
  };

  let nvme = Crc64VariantTunables {
    portable_to_clmul: ov
      .portable_to_clmul
      .or(tuned.map(|t| t.nvme.portable_to_clmul))
      .unwrap_or(default_portable_to_clmul),
    pclmul_to_vpclmul: ov
      .pclmul_to_vpclmul
      .or(tuned.map(|t| t.nvme.pclmul_to_vpclmul))
      .unwrap_or(default_pclmul_to_vpclmul),
    streams: nvme_streams,
    // min_bytes_per_lane: env var > tuned defaults > None (use family default in policy)
    min_bytes_per_lane: ov.min_bytes_per_lane.or(tuned.and_then(|t| t.nvme.min_bytes_per_lane)),
  };

  let requested_force = ov.force;
  let effective_force = clamp_force_to_caps(requested_force, caps);

  Crc64Config {
    requested_force,
    effective_force,
    tunables: Crc64Tunables { xz, nvme },
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
    *CACHED.get_or_init(|| platform::dispatch_auto(config))
  }

  #[cfg(not(feature = "std"))]
  {
    config(platform::caps(), platform::tune())
  }
}
