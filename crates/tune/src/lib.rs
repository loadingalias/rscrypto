//! Unified tuning engine for rscrypto algorithms.
//!
//! This crate provides a generic framework for benchmarking and tuning
//! cryptographic algorithms across different platforms and CPU architectures.
//!
//! # Overview
//!
//! The tuning engine works with any algorithm that implements the [`Tunable`] trait.
//! It benchmarks available kernels across a range of buffer sizes to determine:
//!
//! - Optimal kernel for each buffer size tier
//! - Crossover points between kernel tiers (e.g., portable → SIMD)
//! - Optimal parallel stream count for ILP
//! - Per-lane minimum bytes thresholds
//!
//! # Quick Start
//!
//! Run `just tune` to benchmark all algorithms and print optimal settings.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
mod analysis;
#[cfg(feature = "std")]
pub mod crc16;
#[cfg(feature = "std")]
pub mod crc24;
#[cfg(feature = "std")]
pub mod crc32;
#[cfg(feature = "std")]
pub mod crc64;
#[cfg(feature = "std")]
mod engine;
#[cfg(feature = "std")]
pub mod report;
#[cfg(feature = "std")]
mod runner;
#[cfg(feature = "std")]
pub mod sampler;
#[cfg(feature = "std")]
pub mod stats;

use core::fmt;

#[cfg(feature = "std")]
pub use analysis::{
  AnalysisResult, BestConfig, Crossover, CrossoverType, TypedCrossover, estimate_min_bytes_per_lane,
  find_best_config_across_sizes, find_best_large_config, find_best_large_kernel, find_tier_crossover,
  select_best_streams,
};
#[cfg(feature = "std")]
pub use engine::TuneEngine;
#[cfg(feature = "std")]
pub mod apply;
use platform::Caps;
#[cfg(feature = "std")]
pub use report::{OutputFormat, Report};
#[cfg(feature = "std")]
pub use runner::BenchRunner;
#[cfg(feature = "std")]
pub use sampler::{SampledResult, Sampler, SamplerConfig};
#[cfg(feature = "std")]
pub use stats::{DEFAULT_CV_THRESHOLD, SampleStats, VarianceQuality};

/// Kernel tier classification.
///
/// Kernels are organized into tiers based on their complexity and overhead.
/// Higher tiers generally provide better throughput for larger buffers but
/// have higher setup overhead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum KernelTier {
  /// Bitwise reference implementation (slowest, always available).
  Reference,
  /// Table-based portable implementation (no SIMD).
  Portable,
  /// Hardware-accelerated instructions (e.g., CRC32 instruction).
  Hardware,
  /// SIMD folding/carryless-multiply (e.g., PCLMUL, PMULL).
  Folding,
  /// Wide SIMD with extended instructions (e.g., VPCLMUL, EOR3).
  Wide,
}

impl fmt::Display for KernelTier {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Reference => write!(f, "reference"),
      Self::Portable => write!(f, "portable"),
      Self::Hardware => write!(f, "hardware"),
      Self::Folding => write!(f, "folding"),
      Self::Wide => write!(f, "wide"),
    }
  }
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
  /// The `iterations` parameter suggests how many times to run the algorithm,
  /// but implementations may adjust this for statistical validity.
  fn benchmark(&self, data: &[u8], iterations: usize) -> BenchResult;

  /// Get the current kernel name (after forcing or auto-selection).
  fn current_kernel(&self) -> &'static str;

  /// Get tunable parameters for this algorithm.
  fn tunable_params(&self) -> &[TunableParam];

  /// Environment variable prefix for this algorithm (e.g., "RSCRYPTO_CRC64").
  fn env_prefix(&self) -> &'static str;

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
  pub tune_kind: platform::TuneKind,

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
      tune_kind: detected.tune.kind,
      description: platform::describe().to_string(),
    }
  }
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

  /// Recommended thresholds as (env_suffix, value) pairs.
  ///
  /// The env_suffix is the part after the prefix, e.g., "THRESHOLD_PORTABLE_TO_CLMUL"
  /// for the full env var "RSCRYPTO_CRC64_THRESHOLD_PORTABLE_TO_CLMUL".
  pub thresholds: Vec<(String, usize)>,

  /// Detailed analysis.
  pub analysis: AnalysisResult,
}
