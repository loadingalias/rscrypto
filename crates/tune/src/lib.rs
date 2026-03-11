//! Unified tuning engine for rscrypto algorithms.
//!
//! This crate provides a framework for benchmarking and tuning cryptographic
//! algorithms across different platforms and CPU architectures.
//!
//! # Overview
//!
//! The tuning engine works with any algorithm that implements the [`Tunable`] trait.
//! It benchmarks available kernels across a range of buffer sizes to determine:
//!
//! - Optimal kernel for each buffer size tier
//! - Crossover points between kernel tiers (e.g., portable to SIMD)
//! - Optimal parallel stream count for ILP
//! - Per-lane minimum bytes thresholds
//!
//! # Usage
//!
//! Active frontdoor: run `just tune` (Blake3 boundary capture).
//!
//! ```text
//! # Active path: Blake3 boundary capture
//! just tune
//!
//! # Alternate window sizes
//! just tune 120 200
//! ```
//!
//! # Architecture
//!
//! The crate is organized around these key components:
//!
//! - [`TuneEngine`]: Orchestrates benchmarking for multiple algorithms
//! - [`BenchRunner`]: Configures warmup and measurement durations
//! - [`Sampler`]: Collects statistically valid throughput samples
//! - [`Tunable`]: Trait implemented by algorithms to enable tuning

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "alloc")]
extern crate alloc;

// Core infrastructure (std-only)
#[cfg(feature = "std")]
mod analysis;
#[cfg(feature = "std")]
mod engine;
#[cfg(feature = "std")]
mod runner;

// Public modules (std-only)
#[cfg(feature = "std")]
pub mod apply;
#[cfg(feature = "std")]
pub mod raw;
#[cfg(feature = "std")]
pub mod report;
#[cfg(feature = "std")]
pub mod sampler;
#[cfg(feature = "std")]
pub mod stats;
#[cfg(feature = "std")]
pub mod targets;

// CRC tunable implementations (std-only)
#[cfg(feature = "std")]
pub mod crc16;
#[cfg(feature = "std")]
pub mod crc24;
#[cfg(feature = "std")]
pub mod crc32;
#[cfg(feature = "std")]
pub mod crc64;
#[cfg(feature = "std")]
pub mod hash;

use core::fmt;

#[cfg(feature = "std")]
pub use analysis::{
  AnalysisResult, BestConfig, Crossover, CrossoverType, TypedCrossover, estimate_min_bytes_per_lane,
  find_best_config_across_sizes, find_best_large_config, find_best_large_kernel, find_tier_crossover,
  select_best_streams,
};
// ─────────────────────────────────────────────────────────────────────────────
// Core Types
// ─────────────────────────────────────────────────────────────────────────────
/// Kernel tier classification used by the tuning engine.
///
/// This is shared with the runtime dispatch subsystem in `backend` to ensure
/// tiers mean the same thing across the entire repository.
pub use backend::KernelTier;
#[cfg(feature = "std")]
pub use engine::TuneEngine;
use platform::Caps;
#[cfg(feature = "std")]
pub use raw::{
  AggregationMode, RAW_SCHEMA_VERSION, RawAlgorithmMeasurements, RawBenchPoint, RawBlake3ParallelCurve,
  RawBlake3ParallelData, RawKernelSpec, RawPlatformInfo, RawRunnerConfig, RawThroughputPoint, RawTuneResults,
  aggregate_raw_results, read_raw_results, write_raw_results,
};
#[cfg(feature = "std")]
pub use report::{OutputFormat, Report};
#[cfg(feature = "std")]
pub use runner::BenchRunner;
#[cfg(feature = "std")]
pub use sampler::{SampledResult, Sampler, SamplerConfig};
#[cfg(feature = "std")]
pub use stats::{DEFAULT_CV_THRESHOLD, SampleStats, VarianceQuality};

/// High-level tuning domain.
///
/// Checksums and hashes have different performance profiles and measurement
/// noise characteristics. The tuning engine may use different benchmarking
/// settings per domain (e.g. longer measurement windows for hashes).
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuningDomain {
  Checksum,
  Hash,
}

/// Specification of an available kernel for an algorithm.
#[derive(Clone, Debug)]
pub struct KernelSpec {
  /// Kernel name (e.g., "x86_64/vpclmul-4way").
  pub name: &'static str,

  /// Kernel tier classification.
  pub tier: KernelTier,

  /// Required CPU capabilities.
  pub requires: Caps,

  /// Available stream counts for this kernel (None = no stream variants).
  ///
  /// For kernels with ILP variants (e.g., 1-way, 2-way, 4-way folding),
  /// this specifies the range of available stream counts.
  pub streams: Option<(u8, u8)>,
}

impl KernelSpec {
  /// Create a new kernel specification.
  #[inline]
  #[must_use]
  pub const fn new(name: &'static str, tier: KernelTier, requires: Caps) -> Self {
    Self {
      name,
      tier,
      requires,
      streams: None,
    }
  }

  /// Create a kernel specification with stream variants.
  #[inline]
  #[must_use]
  pub const fn with_streams(
    name: &'static str,
    tier: KernelTier,
    requires: Caps,
    min_streams: u8,
    max_streams: u8,
  ) -> Self {
    Self {
      name,
      tier,
      requires,
      streams: Some((min_streams, max_streams)),
    }
  }
}

/// A tunable parameter for an algorithm.
#[derive(Clone, Debug)]
pub struct TunableParam {
  /// Parameter name (e.g., "portable_to_clmul").
  pub name: &'static str,

  /// Human-readable description.
  pub description: &'static str,

  /// Minimum valid value.
  pub min: usize,

  /// Maximum valid value.
  pub max: usize,

  /// Sensible default value.
  pub default: usize,
}

impl TunableParam {
  /// Create a new tunable parameter.
  #[inline]
  #[must_use]
  pub const fn new(name: &'static str, description: &'static str, min: usize, max: usize, default: usize) -> Self {
    Self {
      name,
      description,
      min,
      max,
      default,
    }
  }
}

/// Result of a single benchmark measurement.
#[derive(Clone, Debug)]
pub struct BenchResult {
  /// Kernel name used for this measurement.
  pub kernel: &'static str,

  /// Buffer size in bytes.
  pub buffer_size: usize,

  /// Number of iterations performed.
  pub iterations: u64,

  /// Total bytes processed.
  pub bytes_processed: u64,

  /// Mean throughput in GiB/s.
  pub throughput_gib_s: f64,

  /// Elapsed time in seconds.
  pub elapsed_secs: f64,

  /// Number of samples collected (for statistical validation).
  ///
  /// If `None`, the measurement was taken using aggregate timing (legacy mode).
  /// If `Some`, per-batch samples were collected for variance analysis.
  pub sample_count: Option<usize>,

  /// Standard deviation of throughput samples (GiB/s).
  pub std_dev: Option<f64>,

  /// Coefficient of variation (std_dev / mean).
  ///
  /// CV > 5% indicates noisy measurements that may be unreliable.
  pub cv: Option<f64>,

  /// Number of outliers rejected during statistical analysis.
  pub outliers_rejected: Option<usize>,

  /// Minimum throughput observed (after outlier rejection).
  pub min_throughput_gib_s: Option<f64>,

  /// Maximum throughput observed (after outlier rejection).
  pub max_throughput_gib_s: Option<f64>,
}

impl BenchResult {
  /// Returns `true` if this result includes statistical validation data.
  #[inline]
  #[must_use]
  pub fn has_stats(&self) -> bool {
    self.sample_count.is_some()
  }

  /// Returns `true` if the measurement has high variance.
  ///
  /// Returns `false` if no variance data is available.
  #[inline]
  #[must_use]
  pub fn is_high_variance(&self, threshold: f64) -> bool {
    self.cv.map(|cv| cv > threshold).unwrap_or(false)
  }

  /// Returns the CV as a percentage string, or "N/A" if not available.
  #[cfg(feature = "alloc")]
  #[must_use]
  pub fn cv_percent_str(&self) -> alloc::string::String {
    match self.cv {
      Some(cv) => alloc::format!("{:.1}%", cv * 100.0),
      None => alloc::string::String::from("N/A"),
    }
  }
}

/// Error type for tuning operations.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum TuneError {
  /// Requested kernel is not available on this platform.
  KernelNotAvailable(&'static str),

  /// Invalid stream count for the kernel.
  InvalidStreamCount(u8),

  /// Benchmark measurement failed.
  BenchmarkFailed(&'static str),

  /// I/O error during tuning.
  #[cfg(feature = "std")]
  Io(String),
}

impl fmt::Display for TuneError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::KernelNotAvailable(name) => write!(f, "kernel not available: {name}"),
      Self::InvalidStreamCount(count) => write!(f, "invalid stream count: {count}"),
      Self::BenchmarkFailed(msg) => write!(f, "benchmark failed: {msg}"),
      #[cfg(feature = "std")]
      Self::Io(msg) => write!(f, "I/O error: {msg}"),
    }
  }
}

#[cfg(feature = "std")]
impl core::error::Error for TuneError {}

#[cfg(feature = "std")]
impl From<std::io::Error> for TuneError {
  fn from(err: std::io::Error) -> Self {
    Self::Io(err.to_string())
  }
}

/// Trait for algorithms that can be tuned.
///
/// Implement this trait to enable benchmarking by the tuning engine.
/// See `crates/tune/src/crc64.rs` for reference implementations.
#[cfg(feature = "std")]
pub trait Tunable: Send + Sync {
  /// Algorithm name (e.g., "crc64-xz", "blake3", "aes-gcm-256").
  fn name(&self) -> &'static str;

  /// List available kernels for the current platform.
  ///
  /// Returns all kernels that could potentially be used on this platform,
  /// filtered by the provided CPU capabilities.
  fn available_kernels(&self, caps: &Caps) -> Vec<KernelSpec>;

  /// Force a specific kernel to be used.
  ///
  /// # Errors
  ///
  /// Returns `TuneError::KernelNotAvailable` if the kernel is not available
  /// on this platform.
  fn force_kernel(&mut self, name: &str) -> Result<(), TuneError>;

  /// Force a specific stream count.
  ///
  /// # Errors
  ///
  /// Returns `TuneError::InvalidStreamCount` if the count is not valid
  /// for the current kernel.
  fn force_streams(&mut self, count: u8) -> Result<(), TuneError>;

  /// Reset to auto-selection mode.
  fn reset(&mut self);

  /// Run benchmark with the given data.
  ///
  /// Returns the benchmark result including throughput measurement.
  /// The provided sampler config controls warmup/measurement durations.
  fn benchmark(&self, data: &[u8], config: &sampler::SamplerConfig) -> BenchResult;

  /// Get the current kernel name (after forcing or auto-selection).
  fn current_kernel(&self) -> &'static str;

  /// Get tunable parameters for this algorithm.
  fn tunable_params(&self) -> &[TunableParam];

  /// Environment variable prefix for this algorithm (e.g., "RSCRYPTO_CRC64").
  fn env_prefix(&self) -> &'static str;

  /// Which tuning domain this algorithm belongs to.
  ///
  /// The engine uses this to select domain-appropriate benchmarking settings.
  #[inline]
  fn tuning_domain(&self) -> TuningDomain {
    TuningDomain::Checksum
  }

  /// Map a generic threshold name to its environment variable suffix.
  ///
  /// This maps analysis-generated threshold names (like "portable_to_simd")
  /// to the algorithm-specific env var suffix (like "THRESHOLD_PORTABLE_TO_CLMUL").
  ///
  /// Returns `None` if the threshold name is not recognized, in which case
  /// the engine will use a default uppercase conversion.
  fn threshold_to_env_suffix(&self, threshold_name: &str) -> Option<&'static str> {
    // Default implementation: no mapping, let caller use generic conversion
    let _ = threshold_name;
    None
  }
}

/// Platform metadata collected during tuning.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct PlatformInfo {
  /// CPU architecture (e.g., "x86_64", "aarch64").
  pub arch: &'static str,

  /// Operating system.
  pub os: &'static str,

  /// Detected CPU capabilities.
  pub caps: Caps,

  /// Platform tune kind.
  pub tune_kind: platform::tune::TuneKind,

  /// Full platform description.
  pub description: String,
}

#[cfg(feature = "std")]
impl PlatformInfo {
  /// Collect platform information.
  #[must_use]
  pub fn collect() -> Self {
    let detected = platform::get();
    Self {
      arch: detected.arch.name(),
      os: std::env::consts::OS,
      caps: detected.caps,
      tune_kind: infer_tune_kind(detected),
      description: platform::describe().to_string(),
    }
  }
}

#[cfg(feature = "std")]
fn infer_tune_kind(detected: platform::Detected) -> platform::tune::TuneKind {
  use platform::{Arch, tune::TuneKind};

  match detected.arch {
    Arch::X86 | Arch::X86_64 => infer_x86_tune_kind(detected.caps),
    Arch::Aarch64 => infer_aarch64_tune_kind(detected.caps),
    Arch::S390x => infer_s390x_tune_kind(detected.caps),
    Arch::Power => infer_power_tune_kind(detected.caps),
    Arch::Riscv32 | Arch::Riscv64 => infer_riscv_tune_kind(detected.caps),
    Arch::Wasm32 | Arch::Wasm64 => infer_wasm_tune_kind(detected.caps),
    Arch::Other => TuneKind::Portable,
    _ => TuneKind::Portable,
  }
}

#[cfg(feature = "std")]
fn infer_riscv_tune_kind(caps: Caps) -> platform::tune::TuneKind {
  use platform::{caps::riscv, tune::TuneKind};

  if caps.has(riscv::ZBC) || caps.has(riscv::ZVBC) {
    TuneKind::Default
  } else {
    TuneKind::Portable
  }
}

#[cfg(feature = "std")]
fn infer_wasm_tune_kind(caps: Caps) -> platform::tune::TuneKind {
  use platform::{caps::wasm, tune::TuneKind};

  if caps.has(wasm::SIMD128) {
    TuneKind::Default
  } else {
    TuneKind::Portable
  }
}

#[cfg(feature = "std")]
fn infer_s390x_tune_kind(caps: Caps) -> platform::tune::TuneKind {
  use platform::{caps::s390x, tune::TuneKind};

  if caps.has(s390x::VECTOR_ENH2) {
    TuneKind::Z15
  } else if caps.has(s390x::VECTOR_ENH1) {
    TuneKind::Z14
  } else if caps.has(s390x::VECTOR) {
    TuneKind::Z13
  } else {
    TuneKind::Portable
  }
}

#[cfg(feature = "std")]
fn infer_power_tune_kind(caps: Caps) -> platform::tune::TuneKind {
  use platform::{caps::power, tune::TuneKind};

  if caps.has(power::POWER10_VECTOR) {
    TuneKind::Power10
  } else if caps.has(power::POWER9_VECTOR) {
    TuneKind::Power9
  } else if caps.has(power::POWER8_VECTOR) {
    TuneKind::Power8
  } else if caps.has(power::VSX) {
    TuneKind::Power7
  } else {
    TuneKind::Portable
  }
}

#[cfg(feature = "std")]
fn infer_aarch64_tune_kind(caps: Caps) -> platform::tune::TuneKind {
  use platform::{caps::aarch64, tune::TuneKind};

  #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos"))]
  {
    if caps.has(aarch64::PMULL_EOR3_READY) {
      if caps.has(aarch64::SME2) {
        return TuneKind::AppleM5;
      }
      if caps.has(aarch64::SME) {
        return TuneKind::AppleM4;
      }
      return TuneKind::AppleM1M3;
    }
  }

  if caps.has(aarch64::SME2P1) {
    return TuneKind::Graviton5;
  }

  if caps.has(aarch64::SVE2) {
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    if let Some((implementer, part)) = read_midr_parts() {
      return match (implementer, part) {
        (MIDR_IMPL_NVIDIA, _) => TuneKind::NvidiaGrace,
        (MIDR_IMPL_AMPERE, _) => TuneKind::AmpereAltra,
        (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_N3) => TuneKind::NeoverseN3,
        (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_N2) => TuneKind::NeoverseN2,
        (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_V2) => TuneKind::Graviton4,
        (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_V1) => TuneKind::Graviton3,
        _ => infer_aarch64_sve2_from_vlen(),
      };
    }
    return infer_aarch64_sve2_from_vlen();
  }

  if caps.has(aarch64::SVE) {
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    if let Some((implementer, part)) = read_midr_parts() {
      return match (implementer, part) {
        (MIDR_IMPL_NVIDIA, _) => TuneKind::NvidiaGrace,
        (MIDR_IMPL_AMPERE, _) => TuneKind::AmpereAltra,
        (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_N2) => TuneKind::NeoverseN2,
        (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_V1) => TuneKind::Graviton3,
        _ => infer_aarch64_sve_from_vlen(),
      };
    }
    return infer_aarch64_sve_from_vlen();
  }

  if caps.has(aarch64::PMULL_EOR3_READY) {
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    if let Some((implementer, part)) = read_midr_parts() {
      return match (implementer, part) {
        (MIDR_IMPL_AMPERE, _) => TuneKind::AmpereAltra,
        (MIDR_IMPL_ARM, MIDR_PART_NEOVERSE_N1) => TuneKind::Graviton2,
        _ => TuneKind::Aarch64Pmull,
      };
    }
    return TuneKind::Aarch64Pmull;
  }

  if caps.has(aarch64::PMULL_READY) {
    TuneKind::Aarch64Pmull
  } else if caps.has(aarch64::NEON) {
    TuneKind::Default
  } else {
    TuneKind::Portable
  }
}

#[cfg(feature = "std")]
fn infer_aarch64_sve2_from_vlen() -> platform::tune::TuneKind {
  use platform::tune::TuneKind;

  let vlen = detect_sve_vlen();
  if vlen > 128 {
    TuneKind::NeoverseV3
  } else if vlen > 0 {
    TuneKind::Graviton4
  } else {
    TuneKind::Default
  }
}

#[cfg(feature = "std")]
fn infer_aarch64_sve_from_vlen() -> platform::tune::TuneKind {
  use platform::tune::TuneKind;

  let vlen = detect_sve_vlen();
  if vlen >= 256 {
    TuneKind::Graviton3
  } else if vlen > 0 {
    TuneKind::NeoverseN2
  } else {
    TuneKind::Default
  }
}

#[cfg(feature = "std")]
fn infer_x86_tune_kind(caps: Caps) -> platform::tune::TuneKind {
  use platform::{caps::x86, tune::TuneKind};

  let Some((is_amd, family, model)) = x86_identity() else {
    return if caps.has(x86::PCLMUL_READY) {
      TuneKind::Default
    } else {
      TuneKind::Portable
    };
  };

  if is_intel_hybrid(is_amd, family, model) && !hybrid_avx512_override() && caps.has(x86::AVX2) {
    return TuneKind::IntelIcl;
  }

  let has_avx512 = caps.has(x86::AVX512F) && caps.has(x86::AVX512VL);
  if has_avx512 {
    if is_amd {
      return if family == 26 {
        TuneKind::Zen5
      } else if family == 25 {
        TuneKind::Zen4
      } else {
        TuneKind::Default
      };
    }

    return if is_intel_icl_model(family, model) {
      TuneKind::IntelIcl
    } else if is_intel_gnr_model(family, model) {
      TuneKind::IntelGnr
    } else if is_intel_spr_model(family, model)
      || caps.has(x86::AVX512BF16)
      || caps.has(x86::AMX_TILE)
      || caps.has(x86::AMX_BF16)
      || caps.has(x86::AMX_INT8)
    {
      TuneKind::IntelSpr
    } else if caps.has(x86::AMX_FP16) || caps.has(x86::AMX_COMPLEX) {
      TuneKind::IntelGnr
    } else {
      TuneKind::IntelIcl
    };
  }

  if caps.has(x86::AVX2) {
    if is_amd {
      return if family == 26 {
        TuneKind::Zen5
      } else if family == 25 {
        TuneKind::Zen4
      } else {
        TuneKind::Default
      };
    }
    return TuneKind::IntelIcl;
  }

  if caps.has(x86::PCLMUL_READY) {
    if is_amd { TuneKind::Default } else { TuneKind::IntelIcl }
  } else {
    TuneKind::Portable
  }
}

#[cfg(feature = "std")]
fn hybrid_avx512_override() -> bool {
  std::env::var("RSCRYPTO_FORCE_AVX512")
    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    .unwrap_or(false)
}

#[cfg(feature = "std")]
fn is_intel_hybrid(is_amd: bool, family: u32, model: u32) -> bool {
  if is_amd || family != 6 {
    return false;
  }

  matches!(
    model,
    0x97 | 0x9A | 0x9C | 0xB7 | 0xBA | 0xBF | 0xAA | 0xAC | 0xBD | 0xC5 | 0xC6
  )
}

#[cfg(feature = "std")]
fn is_intel_icl_model(family: u32, model: u32) -> bool {
  family == 6 && matches!(model, 0x6A | 0x6C | 0x7D | 0x7E)
}

#[cfg(feature = "std")]
fn is_intel_spr_model(family: u32, model: u32) -> bool {
  family == 6 && matches!(model, 0x8F | 0xCF)
}

#[cfg(feature = "std")]
fn is_intel_gnr_model(family: u32, model: u32) -> bool {
  family == 6 && matches!(model, 0xAD)
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn x86_identity() -> Option<(bool, u32, u32)> {
  use core::arch::x86_64::__cpuid;

  let leaf0 = __cpuid(0);
  let is_amd = leaf0.ebx == 0x6874_7541 && leaf0.edx == 0x6974_6E65 && leaf0.ecx == 0x444D_4163;

  let leaf1 = __cpuid(1);
  let family_id = (leaf1.eax >> 8) & 0xF;
  let model_id = (leaf1.eax >> 4) & 0xF;
  let ext_family = (leaf1.eax >> 20) & 0xFF;
  let ext_model = (leaf1.eax >> 16) & 0xF;

  let family = if family_id == 0xF {
    family_id + ext_family
  } else {
    family_id
  };
  let model = if family_id == 0x6 || family_id == 0xF {
    (ext_model << 4) | model_id
  } else {
    model_id
  };

  Some((is_amd, family, model))
}

#[cfg(all(feature = "std", target_arch = "x86"))]
fn x86_identity() -> Option<(bool, u32, u32)> {
  use core::arch::x86::__cpuid;

  let leaf0 = __cpuid(0);
  let is_amd = leaf0.ebx == 0x6874_7541 && leaf0.edx == 0x6974_6E65 && leaf0.ecx == 0x444D_4163;

  let leaf1 = __cpuid(1);
  let family_id = (leaf1.eax >> 8) & 0xF;
  let model_id = (leaf1.eax >> 4) & 0xF;
  let ext_family = (leaf1.eax >> 20) & 0xFF;
  let ext_model = (leaf1.eax >> 16) & 0xF;

  let family = if family_id == 0xF {
    family_id + ext_family
  } else {
    family_id
  };
  let model = if family_id == 0x6 || family_id == 0xF {
    (ext_model << 4) | model_id
  } else {
    model_id
  };

  Some((is_amd, family, model))
}

#[cfg(all(feature = "std", not(any(target_arch = "x86_64", target_arch = "x86"))))]
fn x86_identity() -> Option<(bool, u32, u32)> {
  None
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
const MIDR_IMPL_ARM: u8 = 0x41;
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
const MIDR_IMPL_AMPERE: u8 = 0xC0;
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
const MIDR_IMPL_NVIDIA: u8 = 0x4E;
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
const MIDR_PART_NEOVERSE_N1: u16 = 0xD0C;
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
const MIDR_PART_NEOVERSE_N2: u16 = 0xD49;
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
const MIDR_PART_NEOVERSE_N3: u16 = 0xD8E;
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
const MIDR_PART_NEOVERSE_V1: u16 = 0xD40;
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
const MIDR_PART_NEOVERSE_V2: u16 = 0xD4F;

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
fn read_midr_parts() -> Option<(u8, u16)> {
  let midr = read_midr_el1()?;
  Some((((midr >> 24) & 0xFF) as u8, ((midr >> 4) & 0xFFF) as u16))
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
fn read_midr_el1() -> Option<u64> {
  use std::fs;

  if let Ok(contents) = fs::read_to_string("/sys/devices/system/cpu/cpu0/regs/identification/midr_el1")
    && let Ok(midr) = u64::from_str_radix(contents.trim().trim_start_matches("0x"), 16)
  {
    return Some(midr);
  }

  None
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
fn detect_sve_vlen() -> u16 {
  const SYS_PRCTL: u64 = 167;
  const PR_SVE_GET_VL: u64 = 51;
  const PR_SVE_VL_LEN_MASK: u64 = 0xFFFF;

  let result: i64;
  // SAFETY: `prctl(PR_SVE_GET_VL)` is a side-effect-free kernel query.
  unsafe {
    core::arch::asm!(
      "svc #0",
      in("x8") SYS_PRCTL,
      in("x0") PR_SVE_GET_VL,
      in("x1") 0u64,
      in("x2") 0u64,
      in("x3") 0u64,
      in("x4") 0u64,
      lateout("x0") result,
      options(nostack)
    );
  }

  if result < 0 {
    0
  } else {
    (((result as u64) & PR_SVE_VL_LEN_MASK).saturating_mul(8)) as u16
  }
}

#[cfg(all(feature = "std", not(all(target_arch = "aarch64", target_os = "linux"))))]
fn detect_sve_vlen() -> u16 {
  0
}

/// Complete tuning results for all algorithms.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct TuneResults {
  /// Platform information.
  pub platform: PlatformInfo,

  /// Results per algorithm.
  pub algorithms: Vec<AlgorithmResult>,

  /// Timestamp of the tuning run.
  pub timestamp: String,
}

/// Best kernel selection for a particular size class.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct SizeClassBest {
  /// Size class label (e.g., "xs", "s", "m", "l").
  pub class: &'static str,
  /// Selected base kernel name (e.g., "x86_64/vpclmul").
  pub kernel: String,
  /// Selected stream count (1 if the kernel has no stream variants).
  pub streams: u8,
  /// Throughput at the representative class size (GiB/s).
  pub throughput_gib_s: f64,
}

/// Tuning results for a single algorithm.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct AlgorithmResult {
  /// Algorithm name.
  pub name: &'static str,

  /// Environment variable prefix for this algorithm (e.g., "RSCRYPTO_CRC64").
  pub env_prefix: &'static str,

  /// Best kernel for large buffers.
  pub best_kernel: &'static str,

  /// Recommended stream count.
  pub recommended_streams: u8,

  /// Peak throughput in GiB/s.
  pub peak_throughput_gib_s: f64,

  /// Best kernel per default size class (xs/s/m/l).
  pub size_class_best: Vec<SizeClassBest>,

  /// Recommended thresholds as (env_suffix, value) pairs.
  ///
  /// The env_suffix is the part after the prefix, e.g., "THRESHOLD_PORTABLE_TO_CLMUL"
  /// for the full env var "RSCRYPTO_CRC64_THRESHOLD_PORTABLE_TO_CLMUL".
  pub thresholds: Vec<(String, usize)>,

  /// Detailed analysis.
  pub analysis: AnalysisResult,
}
