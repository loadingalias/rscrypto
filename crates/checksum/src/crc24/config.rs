//! CRC-24 runtime configuration (tunables + overrides).

use platform::{Caps, Tune};

use super::tuned_defaults;

/// Forced backend selection for CRC-24.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Crc24Force {
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

impl Crc24Force {
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

  /// Map to `KernelFamily` for policy-based dispatch.
  ///
  /// Returns `None` for `Auto` (let policy decide) and portable tiers.
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

/// CRC-24 selection tunables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc24Tunables {
  /// Minimum `len` in bytes to use slice-by-8 (otherwise slice-by-4).
  pub slice4_to_slice8: usize,
  /// Minimum `len` in bytes to use the CLMUL/PMULL tier (otherwise portable).
  pub portable_to_clmul: usize,
  /// Minimum `len` in bytes to use the wide tier (VPCLMUL/Zvbc) when available.
  pub pclmul_to_vpclmul: usize,
  /// Preferred maximum parallel streams for multi-way folding.
  pub streams: u8,
  /// Minimum bytes per lane for multi-stream folding.
  ///
  /// When `None`, uses `KernelFamily::min_bytes_per_lane()` as the default.
  pub min_bytes_per_lane: Option<usize>,
}

/// Full CRC-24 runtime configuration (after applying overrides).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc24Config {
  /// Requested force mode (env/programmatic).
  pub requested_force: Crc24Force,
  /// Force mode clamped to detected CPU capabilities.
  pub effective_force: Crc24Force,
  /// Tunables used by the selector.
  pub tunables: Crc24Tunables,
}

#[derive(Clone, Copy, Debug, Default)]
struct Overrides {
  force: Crc24Force,
  slice4_to_slice8: Option<usize>,
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

  fn parse_force(name: &str) -> Option<Crc24Force> {
    let value = std::env::var(name).ok()?;
    let value = value.trim();
    if value.is_empty() {
      return None;
    }

    if value.eq_ignore_ascii_case("auto") {
      return Some(Crc24Force::Auto);
    }
    if value.eq_ignore_ascii_case("reference") || value.eq_ignore_ascii_case("bitwise") {
      return Some(Crc24Force::Reference);
    }
    if value.eq_ignore_ascii_case("portable")
      || value.eq_ignore_ascii_case("scalar")
      || value.eq_ignore_ascii_case("table")
    {
      return Some(Crc24Force::Portable);
    }
    if value.eq_ignore_ascii_case("clmul")
      || value.eq_ignore_ascii_case("pclmul")
      || value.eq_ignore_ascii_case("pclmulqdq")
      || value.eq_ignore_ascii_case("vpclmul")
      || value.eq_ignore_ascii_case("vpclmulqdq")
      || value.eq_ignore_ascii_case("pmull")
      || value.eq_ignore_ascii_case("vpmsum")
      || value.eq_ignore_ascii_case("vgfm")
      || value.eq_ignore_ascii_case("zbc")
      || value.eq_ignore_ascii_case("zvbc")
    {
      return Some(Crc24Force::Clmul);
    }
    if value.eq_ignore_ascii_case("slice4") || value.eq_ignore_ascii_case("slice-4") {
      return Some(Crc24Force::Slice4);
    }
    if value.eq_ignore_ascii_case("slice8") || value.eq_ignore_ascii_case("slice-8") {
      return Some(Crc24Force::Slice8);
    }

    None
  }

  Overrides {
    force: parse_force("RSCRYPTO_CRC24_FORCE").unwrap_or(Crc24Force::Auto),
    slice4_to_slice8: parse_usize("RSCRYPTO_CRC24_THRESHOLD_SLICE4_TO_SLICE8"),
    portable_to_clmul: parse_usize("RSCRYPTO_CRC24_THRESHOLD_PORTABLE_TO_CLMUL"),
    pclmul_to_vpclmul: parse_usize("RSCRYPTO_CRC24_THRESHOLD_PCLMUL_TO_VPCLMUL"),
    streams: parse_u8("RSCRYPTO_CRC24_STREAMS"),
    min_bytes_per_lane: parse_usize("RSCRYPTO_CRC24_MIN_BYTES_PER_LANE"),
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
fn clamp_force_to_caps(requested: Crc24Force, caps: Caps) -> Crc24Force {
  match requested {
    Crc24Force::Auto | Crc24Force::Reference | Crc24Force::Portable | Crc24Force::Slice4 | Crc24Force::Slice8 => {
      requested
    }
    Crc24Force::Clmul => {
      #[cfg(target_arch = "x86_64")]
      {
        if caps.has(platform::caps::x86::PCLMUL_READY) {
          return Crc24Force::Clmul;
        }
      }
      #[cfg(target_arch = "aarch64")]
      {
        if caps.has(platform::caps::aarch64::PMULL_READY) {
          return Crc24Force::Clmul;
        }
      }
      #[cfg(target_arch = "powerpc64")]
      {
        if caps.has(platform::caps::power::VPMSUM_READY) {
          return Crc24Force::Clmul;
        }
      }
      #[cfg(target_arch = "s390x")]
      {
        if caps.has(platform::caps::s390x::VECTOR) {
          return Crc24Force::Clmul;
        }
      }
      #[cfg(target_arch = "riscv64")]
      {
        use platform::caps::riscv;
        if caps.has(riscv::ZBC) || caps.has(riscv::ZVBC) {
          return Crc24Force::Clmul;
        }
      }
      Crc24Force::Auto
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
fn default_pclmul_to_vpclmul(tune: Tune) -> usize {
  let simd_bytes = (tune.effective_simd_width as usize).strict_div(8).max(1);
  tuned_defaults::for_tune_kind(tune.kind)
    .and_then(|d| d.pclmul_to_vpclmul)
    .unwrap_or_else(|| simd_bytes.strict_mul(4).max(tune.pclmul_threshold))
    .max(1)
}

#[inline]
#[must_use]
fn clamp_streams(streams: u8) -> u8 {
  streams.clamp(1, 16)
}

#[inline]
#[must_use]
fn config(caps: Caps, tune: Tune) -> Crc24Config {
  let ov = overrides();

  let mut slice4_to_slice8 = default_slice4_to_slice8(tune);
  if let Some(v) = ov.slice4_to_slice8 {
    slice4_to_slice8 = v.max(1);
  }

  let mut portable_to_clmul = default_portable_to_clmul(tune);
  if let Some(v) = ov.portable_to_clmul {
    portable_to_clmul = v.max(1);
  }

  let mut pclmul_to_vpclmul = default_pclmul_to_vpclmul(tune);
  if let Some(v) = ov.pclmul_to_vpclmul {
    pclmul_to_vpclmul = v.max(1);
  }

  let requested_force = ov.force;
  let effective_force = clamp_force_to_caps(requested_force, caps);

  let tuned = tuned_defaults::for_tune_kind(tune.kind);

  let streams = clamp_streams(
    ov.streams
      .or_else(|| tuned.map(|d| d.streams))
      .unwrap_or(tune.parallel_streams),
  );

  let min_bytes_per_lane = ov
    .min_bytes_per_lane
    .or_else(|| tuned.and_then(|d| d.min_bytes_per_lane));

  Crc24Config {
    requested_force,
    effective_force,
    tunables: Crc24Tunables {
      slice4_to_slice8,
      portable_to_clmul,
      pclmul_to_vpclmul,
      streams,
      min_bytes_per_lane,
    },
  }
}

/// Cached process-wide CRC-24 configuration.
#[inline]
#[must_use]
pub fn get() -> Crc24Config {
  #[cfg(feature = "std")]
  {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Crc24Config> = OnceLock::new();
    *CACHED.get_or_init(|| platform::dispatch_auto(config))
  }

  #[cfg(not(feature = "std"))]
  {
    config(platform::caps(), platform::tune())
  }
}
