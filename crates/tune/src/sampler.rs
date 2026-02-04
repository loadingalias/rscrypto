//! Generic sampling infrastructure for benchmarks.
//!
//! This module provides a reusable sampling loop that collects per-batch
//! throughput measurements and computes statistics. Any [`Tunable`](crate::Tunable)
//! implementation can use this to get statistically valid results.
//!
//! # Design
//!
//! The sampler:
//! 1. Runs warmup iterations to stabilize CPU state
//! 2. Collects throughput samples over a measurement window
//! 3. Applies outlier rejection to filter system noise
//! 4. Computes CV to validate measurement quality
//!
//! Used internally by the tuning engine. See `rscrypto-tune` binary for usage.

use alloc::vec::Vec;
use core::time::Duration;
use std::time::Instant;

use crate::stats::{self, DEFAULT_CV_THRESHOLD, MIN_SAMPLES, SampleStats};

/// Bytes in one GiB.
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Default warmup duration.
const DEFAULT_WARMUP_MS: u64 = 50;

/// Default measurement duration.
const DEFAULT_MEASURE_MS: u64 = 100;

/// Default batch duration target (how long each sample should take).
const DEFAULT_BATCH_TARGET_US: u64 = 1000; // 1ms per batch

/// Sampler configuration.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
  /// Duration for warmup phase.
  pub warmup: Duration,

  /// Duration for measurement phase.
  pub measure: Duration,

  /// Target duration for each batch/sample.
  ///
  /// The sampler will adjust batch size to hit this target.
  /// Shorter = more samples but higher timing overhead.
  /// Longer = fewer samples but more stable individual measurements.
  pub batch_target: Duration,

  /// Coefficient of variation threshold for warnings.
  pub cv_threshold: f64,

  /// Minimum samples required for valid measurement.
  pub min_samples: usize,
}

impl Default for SamplerConfig {
  fn default() -> Self {
    Self {
      warmup: Duration::from_millis(DEFAULT_WARMUP_MS),
      measure: Duration::from_millis(DEFAULT_MEASURE_MS),
      batch_target: Duration::from_micros(DEFAULT_BATCH_TARGET_US),
      cv_threshold: DEFAULT_CV_THRESHOLD,
      min_samples: MIN_SAMPLES,
    }
  }
}

impl SamplerConfig {
  /// Create a quick configuration for faster (but noisier) measurements.
  #[must_use]
  pub fn quick() -> Self {
    Self {
      warmup: Duration::from_millis(25),
      measure: Duration::from_millis(50),
      ..Default::default()
    }
  }

  /// Create a thorough configuration for more reliable measurements.
  #[must_use]
  pub fn thorough() -> Self {
    Self {
      warmup: Duration::from_millis(100),
      measure: Duration::from_millis(250),
      batch_target: Duration::from_micros(2000),
      ..Default::default()
    }
  }
}

/// Result of a sampled benchmark run.
#[derive(Clone, Debug)]
pub struct SampledResult {
  /// Mean throughput in GiB/s.
  pub throughput_gib_s: f64,

  /// Total bytes processed.
  pub bytes_processed: u64,

  /// Total iterations performed.
  pub iterations: u64,

  /// Elapsed time in seconds.
  pub elapsed_secs: f64,

  /// Number of samples collected.
  pub sample_count: usize,

  /// Number of outliers rejected.
  pub outliers_rejected: usize,

  /// Standard deviation of throughput samples (GiB/s).
  pub std_dev: f64,

  /// Coefficient of variation (std_dev / mean).
  pub cv: f64,

  /// Minimum throughput observed (after outlier rejection).
  pub min_throughput_gib_s: f64,

  /// Maximum throughput observed (after outlier rejection).
  pub max_throughput_gib_s: f64,

  /// Full statistics (for detailed analysis).
  pub stats: SampleStats,
}

impl SampledResult {
  /// Returns `true` if the measurement has high variance.
  #[inline]
  #[must_use]
  pub fn is_high_variance(&self, threshold: f64) -> bool {
    self.cv > threshold
  }
}

/// Benchmark sampler for collecting statistically valid measurements.
pub struct Sampler<'a> {
  config: &'a SamplerConfig,
}

impl<'a> Sampler<'a> {
  /// Create a new sampler with the given configuration.
  #[inline]
  #[must_use]
  pub const fn new(config: &'a SamplerConfig) -> Self {
    Self { config }
  }

  /// Run a sampled benchmark.
  ///
  /// The closure `run` should process the entire data buffer once.
  /// It will be called multiple times for warmup and measurement.
  #[must_use]
  pub fn run<F>(&self, data: &[u8], mut run: F) -> SampledResult
  where
    F: FnMut(&[u8]),
  {
    let buffer_size = data.len();
    if buffer_size == 0 {
      return SampledResult {
        throughput_gib_s: 0.0,
        bytes_processed: 0,
        iterations: 0,
        elapsed_secs: 0.0,
        sample_count: 0,
        outliers_rejected: 0,
        std_dev: 0.0,
        cv: 0.0,
        min_throughput_gib_s: 0.0,
        max_throughput_gib_s: 0.0,
        stats: SampleStats::default(),
      };
    }

    // Choose a batch size that roughly hits `batch_target`.
    //
    // We start with a coarse estimate, but then calibrate it against the
    // actual closure cost. This avoids pathologically large batches when
    // per-call overhead dominates (tiny buffers / heavy setup), which can
    // otherwise make tuning appear to "hang" by overshooting the warmup/measure
    // windows by seconds.
    let batch_est = self.estimate_batch_size(buffer_size);
    let batch_size = self.calibrate_batch_size(data, &mut run, batch_est);

    // Warmup phase
    self.warmup(data, &mut run, batch_size);

    // Measurement phase: collect samples
    let (samples, total_iterations, total_elapsed) = self.measure(data, &mut run, batch_size);

    // Compute statistics with outlier rejection
    let stats = stats::compute_stats(&samples);

    let bytes_processed = total_iterations.strict_mul(buffer_size as u64);

    SampledResult {
      throughput_gib_s: stats.mean,
      bytes_processed,
      iterations: total_iterations,
      elapsed_secs: total_elapsed.as_secs_f64(),
      sample_count: stats.sample_count,
      outliers_rejected: stats.outliers_rejected,
      std_dev: stats.std_dev,
      cv: stats.cv,
      min_throughput_gib_s: stats.min,
      max_throughput_gib_s: stats.max,
      stats,
    }
  }

  /// Estimate batch size to hit target batch duration.
  fn estimate_batch_size(&self, buffer_size: usize) -> u32 {
    // Assume ~10 GiB/s throughput as baseline (conservative estimate)
    let assumed_throughput: f64 = 10.0 * GIB;
    let bytes_per_batch = assumed_throughput * self.config.batch_target.as_secs_f64();
    let iters = bytes_per_batch / buffer_size as f64;
    (iters as u32).clamp(1, 10_000)
  }

  /// Calibrate `batch_size` against the observed per-iteration cost.
  ///
  /// This is a small preflight run that chooses a batch size targeting
  /// `config.batch_target` in wall-clock time. It is intentionally conservative
  /// and bounded; it should not materially affect overall measurement time.
  fn calibrate_batch_size<F>(&self, data: &[u8], run: &mut F, batch_est: u32) -> u32
  where
    F: FnMut(&[u8]),
  {
    let target = self.config.batch_target;
    if target.is_zero() {
      return batch_est.max(1);
    }

    // Probe: grow iterations until we get a measurable elapsed time, but keep
    // it bounded so huge buffers don't add noticeable extra work.
    const MIN_ELAPSED: Duration = Duration::from_micros(50);
    const MAX_ITERS: u32 = 64;

    let mut iters: u32 = 1;
    let elapsed = loop {
      let start = Instant::now();
      for _ in 0..iters {
        run(core::hint::black_box(data));
        core::hint::black_box(());
      }
      let elapsed = start.elapsed();
      if elapsed >= MIN_ELAPSED || iters >= MAX_ITERS {
        break elapsed;
      }
      iters = iters.saturating_mul(2);
    };

    // If the probe timer resolution is too coarse to measure, keep the estimate.
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
      return batch_est.max(1);
    }

    let per_iter = secs / (iters as f64);
    if per_iter <= 0.0 {
      return batch_est.max(1);
    }

    let want = (target.as_secs_f64() / per_iter).round() as u32;
    want.clamp(1, 10_000)
  }

  /// Run warmup iterations.
  fn warmup<F>(&self, data: &[u8], run: &mut F, batch_size: u32)
  where
    F: FnMut(&[u8]),
  {
    let start = Instant::now();
    while start.elapsed() < self.config.warmup {
      for _ in 0..batch_size {
        run(data);
      }
    }
  }

  /// Collect measurement samples.
  ///
  /// Returns (throughput_samples, total_iterations, total_elapsed).
  fn measure<F>(&self, data: &[u8], run: &mut F, batch_size: u32) -> (Vec<f64>, u64, Duration)
  where
    F: FnMut(&[u8]),
  {
    let buffer_size = data.len();
    let mut samples = Vec::with_capacity(64);
    let mut total_iterations: u64 = 0;

    let measure_start = Instant::now();
    while measure_start.elapsed() < self.config.measure {
      // Time one batch
      let batch_start = Instant::now();
      for _ in 0..batch_size {
        run(core::hint::black_box(data));
        core::hint::black_box(());
      }
      let batch_elapsed = batch_start.elapsed();

      // Calculate throughput for this batch
      let batch_bytes = u64::from(batch_size).strict_mul(buffer_size as u64);
      let batch_secs = batch_elapsed.as_secs_f64();
      if batch_secs > 0.0 {
        let throughput = (batch_bytes as f64) / batch_secs / GIB;
        samples.push(throughput);
      }

      total_iterations = total_iterations.strict_add(u64::from(batch_size));
    }

    let total_elapsed = measure_start.elapsed();
    (samples, total_iterations, total_elapsed)
  }
}

/// Convenience function to run a sampled benchmark with default configuration.
#[must_use]
pub fn sample_benchmark<F>(data: &[u8], run: F) -> SampledResult
where
  F: FnMut(&[u8]),
{
  let config = SamplerConfig::default();
  Sampler::new(&config).run(data, run)
}

/// Convenience function to run a quick sampled benchmark.
#[must_use]
pub fn sample_benchmark_quick<F>(data: &[u8], run: F) -> SampledResult
where
  F: FnMut(&[u8]),
{
  let config = SamplerConfig::quick();
  Sampler::new(&config).run(data, run)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sampler_basic() {
    let data = vec![0u8; 4096];
    let config = SamplerConfig {
      warmup: Duration::from_millis(10),
      measure: Duration::from_millis(50),
      ..Default::default()
    };

    let result = Sampler::new(&config).run(&data, |buf| {
      // Simple checksum to prevent optimization
      let sum: u8 = buf.iter().fold(0u8, |a, &b| a.wrapping_add(b));
      core::hint::black_box(sum);
    });

    assert!(result.iterations > 0);
    assert!(result.sample_count >= 1);
    assert!(result.throughput_gib_s > 0.0);
  }

  #[test]
  fn test_sampler_collects_multiple_samples() {
    let data = vec![0u8; 1024];
    let config = SamplerConfig {
      warmup: Duration::from_millis(5),
      measure: Duration::from_millis(100),
      batch_target: Duration::from_micros(500), // Short batches = more samples
      ..Default::default()
    };

    let result = Sampler::new(&config).run(&data, |buf| {
      let sum: u8 = buf.iter().fold(0u8, |a, &b| a.wrapping_add(b));
      core::hint::black_box(sum);
    });

    // Should have collected multiple samples (threshold is low to handle slow CI runners
    // and outlier rejection reducing the count)
    assert!(result.sample_count >= 3, "sample_count = {}", result.sample_count);
  }

  #[test]
  fn test_empty_data() {
    let config = SamplerConfig::default();
    let result = Sampler::new(&config).run(&[], |_| {});

    assert_eq!(result.iterations, 0);
    assert_eq!(result.throughput_gib_s, 0.0);
  }
}
