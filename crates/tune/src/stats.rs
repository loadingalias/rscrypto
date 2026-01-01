//! Statistical utilities for benchmark analysis.
//!
//! This module provides functions for computing variance metrics,
//! coefficient of variation, and outlier rejection using the IQR method.
//!
//! # Why This Matters
//!
//! Accurate tuning requires reliable measurements. System noise (interrupts,
//! context switches, thermal throttling) can skew results and lead to incorrect
//! kernel selection. This module provides:
//!
//! - **Outlier rejection** via IQR to filter noisy samples
//! - **Coefficient of variation (CV)** to quantify measurement noise
//! - **Sample statistics** for confidence in results
//!
//! # Example
//!
//! ```ignore
//! use tune::stats::{compute_stats, DEFAULT_CV_THRESHOLD};
//!
//! let throughput_samples = vec![10.1, 10.2, 10.0, 10.3, 10.1, 15.0]; // 15.0 is an outlier
//! let stats = compute_stats(&throughput_samples);
//!
//! assert_eq!(stats.outliers_rejected, 1); // 15.0 removed
//! assert!(stats.cv < DEFAULT_CV_THRESHOLD); // Low variance after filtering
//! ```

use alloc::vec::Vec;

/// Minimum number of samples required for meaningful statistical analysis.
pub const MIN_SAMPLES: usize = 5;

/// Default coefficient of variation threshold (5%).
///
/// Measurements with CV > 5% are considered noisy and should trigger a warning.
pub const DEFAULT_CV_THRESHOLD: f64 = 0.05;

/// IQR multiplier for outlier detection.
///
/// Values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are considered outliers.
/// This is the standard "Tukey fence" used in box plots.
pub const IQR_MULTIPLIER: f64 = 1.5;

/// Summary statistics for a set of benchmark samples.
#[derive(Clone, Debug, Default)]
pub struct SampleStats {
  /// Number of samples after outlier rejection.
  pub sample_count: usize,

  /// Number of samples rejected as outliers.
  pub outliers_rejected: usize,

  /// Mean value.
  pub mean: f64,

  /// Standard deviation (with Bessel's correction).
  pub std_dev: f64,

  /// Coefficient of variation (std_dev / mean).
  pub cv: f64,

  /// Minimum value (after outlier rejection).
  pub min: f64,

  /// Maximum value (after outlier rejection).
  pub max: f64,

  /// First quartile (Q1 / 25th percentile).
  pub q1: f64,

  /// Median (Q2 / 50th percentile).
  pub median: f64,

  /// Third quartile (Q3 / 75th percentile).
  pub q3: f64,
}

impl SampleStats {
  /// Returns `true` if the coefficient of variation exceeds the threshold.
  #[inline]
  #[must_use]
  pub fn is_high_variance(&self, threshold: f64) -> bool {
    self.cv > threshold
  }

  /// Returns a human-readable variance assessment.
  #[must_use]
  pub fn variance_quality(&self, threshold: f64) -> VarianceQuality {
    if self.sample_count < MIN_SAMPLES {
      VarianceQuality::InsufficientSamples
    } else if self.cv <= threshold / 2.0 {
      VarianceQuality::Excellent
    } else if self.cv <= threshold {
      VarianceQuality::Good
    } else if self.cv <= threshold * 2.0 {
      VarianceQuality::Moderate
    } else {
      VarianceQuality::High
    }
  }
}

/// Qualitative assessment of measurement variance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarianceQuality {
  /// Not enough samples to assess.
  InsufficientSamples,
  /// CV ≤ threshold/2: Excellent measurement quality.
  Excellent,
  /// CV ≤ threshold: Good measurement quality.
  Good,
  /// CV ≤ threshold*2: Moderate noise, results may be less reliable.
  Moderate,
  /// CV > threshold*2: High noise, measurements should be repeated.
  High,
}

impl VarianceQuality {
  /// Returns a short string representation.
  #[must_use]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::InsufficientSamples => "insufficient",
      Self::Excellent => "excellent",
      Self::Good => "good",
      Self::Moderate => "moderate",
      Self::High => "high",
    }
  }
}

/// Compute the mean of a slice.
#[inline]
#[must_use]
pub fn mean(samples: &[f64]) -> f64 {
  if samples.is_empty() {
    return 0.0;
  }
  samples.iter().sum::<f64>() / samples.len() as f64
}

/// Compute the standard deviation of a slice using Bessel's correction.
#[must_use]
pub fn std_dev(samples: &[f64], mean_value: f64) -> f64 {
  if samples.len() < 2 {
    return 0.0;
  }

  let variance = samples
    .iter()
    .map(|&x| {
      let diff = x - mean_value;
      diff * diff
    })
    .sum::<f64>()
    / (samples.len().strict_sub(1)) as f64;

  variance.sqrt()
}

/// Compute the coefficient of variation (CV = std_dev / mean).
///
/// Returns 0.0 if mean is effectively zero to avoid division issues.
#[inline]
#[must_use]
pub fn coefficient_of_variation(std_dev: f64, mean: f64) -> f64 {
  if mean.abs() < f64::EPSILON {
    return 0.0;
  }
  std_dev / mean
}

/// Compute a percentile value from a sorted slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
  if sorted.is_empty() {
    return 0.0;
  }
  if sorted.len() == 1 {
    return sorted[0];
  }

  let n = sorted.len();
  let idx = p * (n.strict_sub(1)) as f64;
  let lower = idx.floor() as usize;
  let upper = idx.ceil() as usize;

  if lower == upper {
    sorted[lower]
  } else {
    let frac = idx - lower as f64;
    sorted[lower] * (1.0 - frac) + sorted[upper] * frac
  }
}

/// Compute quartiles (Q1, Q2/median, Q3) from samples.
///
/// Returns (Q1, median, Q3).
#[must_use]
pub fn quartiles(samples: &[f64]) -> (f64, f64, f64) {
  if samples.is_empty() {
    return (0.0, 0.0, 0.0);
  }

  let mut sorted: Vec<f64> = samples.to_vec();
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

  let q1 = percentile_sorted(&sorted, 0.25);
  let median = percentile_sorted(&sorted, 0.50);
  let q3 = percentile_sorted(&sorted, 0.75);

  (q1, median, q3)
}

/// Reject outliers using the IQR method (Tukey fences).
///
/// Removes values outside [Q1 - k*IQR, Q3 + k*IQR] where k = `iqr_multiplier`.
/// Returns a new vector with outliers removed.
#[must_use]
pub fn reject_outliers_iqr(samples: &[f64], iqr_multiplier: f64) -> Vec<f64> {
  if samples.len() < 4 {
    // Not enough samples to compute meaningful IQR
    return samples.to_vec();
  }

  let (q1, _, q3) = quartiles(samples);
  let iqr = q3 - q1;
  let lower_fence = q1 - iqr_multiplier * iqr;
  let upper_fence = q3 + iqr_multiplier * iqr;

  samples
    .iter()
    .copied()
    .filter(|&x| x >= lower_fence && x <= upper_fence)
    .collect()
}

/// Compute complete sample statistics with outlier rejection.
///
/// This is the main entry point for statistical analysis:
/// 1. Rejects outliers using the IQR method
/// 2. Computes mean, std_dev, CV, min, max, quartiles
/// 3. Returns a `SampleStats` struct with all metrics
#[must_use]
pub fn compute_stats(samples: &[f64]) -> SampleStats {
  compute_stats_with_iqr(samples, IQR_MULTIPLIER)
}

/// Compute sample statistics with a custom IQR multiplier.
#[must_use]
pub fn compute_stats_with_iqr(samples: &[f64], iqr_multiplier: f64) -> SampleStats {
  if samples.is_empty() {
    return SampleStats::default();
  }

  // Reject outliers
  let filtered = reject_outliers_iqr(samples, iqr_multiplier);
  let outliers_rejected = samples.len().saturating_sub(filtered.len());

  if filtered.is_empty() {
    // All samples were outliers? Fall back to original
    return compute_stats_raw(samples, 0);
  }

  compute_stats_raw(&filtered, outliers_rejected)
}

/// Compute statistics without outlier rejection (internal helper).
fn compute_stats_raw(samples: &[f64], outliers_rejected: usize) -> SampleStats {
  if samples.is_empty() {
    return SampleStats::default();
  }

  let mean_val = mean(samples);
  let std_dev_val = std_dev(samples, mean_val);
  let cv = coefficient_of_variation(std_dev_val, mean_val);
  let (q1, median, q3) = quartiles(samples);

  let min = samples
    .iter()
    .copied()
    .min_by(|a, b| a.partial_cmp(b).unwrap())
    .unwrap_or(0.0);
  let max = samples
    .iter()
    .copied()
    .max_by(|a, b| a.partial_cmp(b).unwrap())
    .unwrap_or(0.0);

  SampleStats {
    sample_count: samples.len(),
    outliers_rejected,
    mean: mean_val,
    std_dev: std_dev_val,
    cv,
    min,
    max,
    q1,
    median,
    q3,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_mean() {
    assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-10);
    assert!((mean(&[10.0]) - 10.0).abs() < 1e-10);
    assert!((mean(&[]) - 0.0).abs() < 1e-10);
  }

  #[test]
  fn test_std_dev() {
    let samples = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let m = mean(&samples);
    let sd = std_dev(&samples, m);
    // Expected std_dev with Bessel's correction ~= 2.138
    assert!(sd > 2.0 && sd < 2.3, "std_dev = {sd}");
  }

  #[test]
  fn test_coefficient_of_variation() {
    assert!((coefficient_of_variation(2.0, 10.0) - 0.2).abs() < 1e-10);
    assert!((coefficient_of_variation(0.5, 10.0) - 0.05).abs() < 1e-10);
    assert!((coefficient_of_variation(1.0, 0.0) - 0.0).abs() < 1e-10);
  }

  #[test]
  fn test_quartiles() {
    let samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let (q1, median, q3) = quartiles(&samples);
    assert!((median - 5.0).abs() < 1e-10);
    // Q1 and Q3 depend on interpolation method; accept reasonable range
    assert!((2.0..=3.5).contains(&q1), "q1 = {q1}");
    assert!((6.5..=8.0).contains(&q3), "q3 = {q3}");
  }

  #[test]
  fn test_outlier_rejection() {
    // Data with clear outliers
    let samples = [10.0, 11.0, 12.0, 11.5, 10.5, 100.0, 0.1];
    let filtered = reject_outliers_iqr(&samples, 1.5);

    // The outliers (100.0 and 0.1) should be rejected
    assert!(filtered.len() < samples.len());
    assert!(!filtered.contains(&100.0));
    assert!(!filtered.contains(&0.1));
  }

  #[test]
  fn test_compute_stats() {
    // Use a tighter distribution with small variance around mean of 10.0
    let samples: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64 - 9.5) * 0.02).collect();
    let stats = compute_stats(&samples);

    assert_eq!(stats.sample_count, 20);
    assert!(stats.cv < 0.05, "cv = {}", stats.cv);
    assert!(!stats.is_high_variance(0.05));
  }

  #[test]
  fn test_variance_quality() {
    let mut stats = SampleStats::default();
    stats.sample_count = 10;

    stats.cv = 0.02;
    assert_eq!(stats.variance_quality(0.05), VarianceQuality::Excellent);

    stats.cv = 0.04;
    assert_eq!(stats.variance_quality(0.05), VarianceQuality::Good);

    stats.cv = 0.08;
    assert_eq!(stats.variance_quality(0.05), VarianceQuality::Moderate);

    stats.cv = 0.15;
    assert_eq!(stats.variance_quality(0.05), VarianceQuality::High);

    stats.sample_count = 3;
    assert_eq!(stats.variance_quality(0.05), VarianceQuality::InsufficientSamples);
  }

  #[test]
  fn test_real_world_scenario() {
    // Simulate a noisy benchmark with one system interrupt
    let mut samples = vec![10.5, 10.4, 10.6, 10.5, 10.3, 10.5, 10.4, 10.6];
    samples.push(5.0); // System interrupt caused one slow sample

    let stats = compute_stats(&samples);

    // Outlier should be rejected
    assert_eq!(stats.outliers_rejected, 1);

    // Mean should be around 10.5, not pulled down by the outlier
    assert!((stats.mean - 10.5).abs() < 0.2, "mean = {}", stats.mean);

    // CV should be low after outlier rejection
    assert!(stats.cv < 0.03, "cv = {}", stats.cv);
  }
}
