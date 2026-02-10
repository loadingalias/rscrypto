//! Benchmark runner with warmup, measurement, and statistical validation.

use core::time::Duration;

use crate::{BenchResult, Tunable, TuneError, stats::DEFAULT_CV_THRESHOLD};

/// Default warmup duration.
const DEFAULT_WARMUP_MS: u64 = 150;

/// Default measurement duration.
const DEFAULT_MEASURE_MS: u64 = 250;

/// Quick mode warmup duration.
const QUICK_WARMUP_MS: u64 = 75;

/// Quick mode measurement duration.
const QUICK_MEASURE_MS: u64 = 125;

/// Hash tuning warmup duration.
///
/// Hash kernels are often sensitive to instruction cache, front-end effects,
/// and inlining/dispatch overhead. A slightly longer window reduces flip-flops
/// in CI without requiring algorithm-specific special-casing in the engine.
const HASH_WARMUP_MS: u64 = 100;

/// Hash tuning measurement duration.
const HASH_MEASURE_MS: u64 = 250;

/// Benchmark runner configuration.
#[derive(Clone, Debug)]
pub struct BenchRunner {
  /// Warmup duration before measurement.
  warmup: Duration,

  /// Measurement duration.
  measure: Duration,

  /// Whether to warn about high variance.
  warn_high_variance: bool,

  /// Coefficient of variation threshold for warnings.
  cv_threshold: f64,
}

impl Default for BenchRunner {
  fn default() -> Self {
    Self {
      warmup: Duration::from_millis(DEFAULT_WARMUP_MS),
      measure: Duration::from_millis(DEFAULT_MEASURE_MS),
      warn_high_variance: true,
      cv_threshold: DEFAULT_CV_THRESHOLD,
    }
  }
}

impl BenchRunner {
  /// Create a new benchmark runner with default settings.
  #[must_use]
  pub fn new() -> Self {
    Self::default()
  }

  /// Create a runner with quick mode settings (faster, noisier).
  #[must_use]
  pub fn quick() -> Self {
    Self {
      warmup: Duration::from_millis(QUICK_WARMUP_MS),
      measure: Duration::from_millis(QUICK_MEASURE_MS),
      ..Default::default()
    }
  }

  /// Create a runner with hash-oriented defaults (more stable than `quick()`).
  #[must_use]
  pub fn quick_hash() -> Self {
    Self {
      warmup: Duration::from_millis(HASH_WARMUP_MS),
      measure: Duration::from_millis(HASH_MEASURE_MS),
      ..Default::default()
    }
  }

  /// Create a runner with hash-oriented defaults (slightly longer windows).
  #[must_use]
  pub fn hash() -> Self {
    // Scale from the non-quick defaults to keep scheduled tuning runs stable.
    Self {
      warmup: Duration::from_millis(DEFAULT_WARMUP_MS + 50),
      measure: Duration::from_millis(DEFAULT_MEASURE_MS + 150),
      ..Default::default()
    }
  }

  /// Set warmup duration.
  #[must_use]
  pub fn with_warmup(mut self, warmup: Duration) -> Self {
    self.warmup = warmup;
    self
  }

  /// Set measurement duration.
  #[must_use]
  pub fn with_measure(mut self, measure: Duration) -> Self {
    self.measure = measure;
    self
  }

  /// Set warmup duration in milliseconds.
  #[must_use]
  pub fn with_warmup_ms(mut self, ms: u64) -> Self {
    self.warmup = Duration::from_millis(ms);
    self
  }

  /// Set measurement duration in milliseconds.
  #[must_use]
  pub fn with_measure_ms(mut self, ms: u64) -> Self {
    self.measure = Duration::from_millis(ms);
    self
  }

  /// Disable high variance warnings.
  #[must_use]
  pub fn without_variance_warnings(mut self) -> Self {
    self.warn_high_variance = false;
    self
  }

  /// Set the coefficient of variation threshold for warnings.
  #[must_use]
  pub fn with_cv_threshold(mut self, threshold: f64) -> Self {
    self.cv_threshold = threshold;
    self
  }

  /// Returns `true` if high variance warnings are enabled.
  #[inline]
  #[must_use]
  pub fn warns_high_variance(&self) -> bool {
    self.warn_high_variance
  }

  /// Returns warmup duration in milliseconds.
  #[inline]
  #[must_use]
  pub fn warmup_ms(&self) -> u64 {
    self.warmup.as_millis() as u64
  }

  /// Returns measurement duration in milliseconds.
  #[inline]
  #[must_use]
  pub fn measure_ms(&self) -> u64 {
    self.measure.as_millis() as u64
  }

  /// Returns the CV threshold for warnings.
  #[inline]
  #[must_use]
  pub fn cv_threshold(&self) -> f64 {
    self.cv_threshold
  }

  /// Check if a benchmark result has high variance.
  ///
  /// Returns `Some(cv)` if the result has high variance, `None` otherwise.
  #[must_use]
  pub fn check_variance(&self, result: &BenchResult) -> Option<f64> {
    if !self.warn_high_variance {
      return None;
    }
    if let Some(cv) = result.cv
      && cv > self.cv_threshold
    {
      return Some(cv);
    }
    None
  }

  /// Emit a warning if the result has high variance.
  ///
  /// Returns `true` if a warning was emitted.
  pub fn warn_if_high_variance(&self, result: &BenchResult, kernel: &str, size: usize) -> bool {
    if let Some(cv) = self.check_variance(result) {
      eprintln!(
        "warning: high variance in benchmark for {} @ {} bytes: CV = {:.1}% (threshold: {:.1}%)",
        kernel,
        size,
        cv * 100.0,
        self.cv_threshold * 100.0
      );
      true
    } else {
      false
    }
  }

  /// Run a complete benchmark suite for an algorithm.
  ///
  /// Benchmarks the algorithm at various buffer sizes and returns
  /// all measurements.
  pub fn run_suite(&self, algorithm: &dyn Tunable, sizes: &[usize]) -> Result<Vec<BenchResult>, TuneError> {
    let max_size = sizes.iter().copied().max().unwrap_or(0);
    if max_size == 0 {
      return Err(TuneError::BenchmarkFailed("no sizes specified"));
    }

    let mut buffer = vec![0u8; max_size];
    fill_data(&mut buffer);

    let mut results = Vec::with_capacity(sizes.len());
    for &size in sizes {
      if size == 0 || size > buffer.len() {
        continue;
      }

      let data = &buffer[..size];
      let result = self.measure_single(algorithm, data)?;
      results.push(result);
    }

    Ok(results)
  }

  /// Measure throughput for a single buffer size.
  ///
  /// Delegates timing to the algorithm's `benchmark()` method, which handles
  /// warmup and measurement internally. This avoids nested timing loops.
  ///
  /// If variance warnings are enabled and the result has high CV, a warning
  /// is emitted to stderr.
  pub fn measure_single(&self, algorithm: &dyn Tunable, data: &[u8]) -> Result<BenchResult, TuneError> {
    let config = self.sampler_config();
    let result = algorithm.benchmark(data, &config);

    // Emit warning if high variance
    self.warn_if_high_variance(&result, result.kernel, data.len());

    Ok(result)
  }

  #[inline]
  #[must_use]
  pub(crate) fn sampler_config(&self) -> crate::sampler::SamplerConfig {
    crate::sampler::SamplerConfig {
      warmup: self.warmup,
      measure: self.measure,
      cv_threshold: self.cv_threshold,
      ..crate::sampler::SamplerConfig::default()
    }
  }

  /// Run benchmarks for multiple kernel/stream configurations.
  pub fn run_matrix<F>(
    &self,
    algorithm: &mut dyn Tunable,
    configs: &[(Option<&str>, Option<u8>)],
    sizes: &[usize],
    mut callback: F,
  ) -> Result<Vec<BenchResult>, TuneError>
  where
    F: FnMut(&str, u8, &BenchResult),
  {
    let max_size = sizes.iter().copied().max().unwrap_or(0);
    if max_size == 0 {
      return Err(TuneError::BenchmarkFailed("no sizes specified"));
    }

    let mut buffer = vec![0u8; max_size];
    fill_data(&mut buffer);

    let mut all_results = Vec::new();

    for &(kernel, streams) in configs {
      // Apply configuration
      algorithm.reset();
      if let Some(k) = kernel {
        algorithm.force_kernel(k)?;
      }
      if let Some(s) = streams {
        algorithm.force_streams(s)?;
      }

      let effective_streams = streams.unwrap_or(1);

      // Benchmark each size
      for &size in sizes {
        if size == 0 || size > buffer.len() {
          continue;
        }

        let data = &buffer[..size];
        let result = self.measure_single(algorithm, data)?;

        callback(algorithm.current_kernel(), effective_streams, &result);
        all_results.push(result);
      }
    }

    // Reset to auto mode
    algorithm.reset();

    Ok(all_results)
  }
}

/// Fill buffer with deterministic pseudo-random data.
///
/// Uses a simple but effective pattern that produces varied byte values
/// without expensive computation. The pattern is reproducible for consistent
/// benchmarking across runs.
pub fn fill_data(buf: &mut [u8]) {
  for (i, b) in buf.iter_mut().enumerate() {
    // Mix low and high bits for better distribution
    *b = (i as u8).wrapping_mul(31).wrapping_add((i >> 8) as u8);
  }
}

/// Standard buffer sizes for threshold detection.
pub const THRESHOLD_SIZES: &[usize] = &[
  64,
  128,
  256,
  512,
  1024,
  2048,
  4096,
  8192,
  16 * 1024,
  32 * 1024,
  64 * 1024,
  1024 * 1024,
];

/// Representative sizes for the default dispatch size classes.
///
/// These sizes are intended to be "pivot points" for selecting a best kernel
/// per class while keeping benchmarking cost bounded.
pub const SIZE_CLASS_SIZES: &[usize; 4] = &[64, 256, 4096, 1024 * 1024];

/// Names for the default dispatch size classes, aligned with [`SIZE_CLASS_SIZES`].
pub const SIZE_CLASS_NAMES: &[&str; 4] = &["xs", "s", "m", "l"];

/// Buffer sizes for stream count selection.
///
/// We intentionally benchmark multiple "large-ish" sizes to avoid overfitting
/// stream selection to a single point (1 MiB). This better captures when
/// additional ILP stops being beneficial (or becomes beneficial later).
pub const STREAM_SIZES: &[usize] = &[32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024];

/// Buffer sizes for stream benchmarking.
///
/// This superset includes the `m` size-class pivot (4096) so size-class winners
/// can select different stream counts for `m` vs `l` without adding a second
/// stream-benchmark pass.
pub const STREAM_SIZES_BENCH: &[usize] = &[
  4096,
  32 * 1024,
  64 * 1024,
  128 * 1024,
  256 * 1024,
  512 * 1024,
  1024 * 1024,
];

/// Get stream candidates for the current architecture.
#[must_use]
pub fn stream_candidates() -> &'static [u8] {
  #[cfg(target_arch = "x86_64")]
  {
    // x86_64 supports 1 / 2 / 4 / 7 / 8-way folding
    &[1, 2, 4, 7, 8]
  }
  #[cfg(target_arch = "aarch64")]
  {
    &[1, 2, 3]
  }
  #[cfg(target_arch = "powerpc64")]
  {
    // Power (powerpc64) supports 1 / 2 / 4 / 8-way folding
    &[1, 2, 4, 8]
  }
  #[cfg(target_arch = "s390x")]
  {
    &[1, 2, 4]
  }
  #[cfg(target_arch = "riscv64")]
  {
    &[1, 2, 4]
  }
  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  )))]
  {
    &[1]
  }
}
