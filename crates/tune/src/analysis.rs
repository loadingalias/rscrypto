//! Crossover detection and threshold analysis.

use crate::BenchResult;

/// Complete analysis result for an algorithm.
#[derive(Clone, Debug, Default)]
pub struct AnalysisResult {
  /// Detected crossover points.
  pub crossovers: Vec<Crossover>,

  /// Recommended stream count.
  pub recommended_streams: u8,

  /// Recommended thresholds as (name, value) pairs.
  pub recommended_thresholds: Vec<(String, usize)>,

  /// Overall confidence score (0.0 to 1.0).
  pub confidence: f64,

  /// Best kernel at large buffer sizes.
  pub best_large_kernel: Option<&'static str>,

  /// Peak throughput in GiB/s.
  pub peak_throughput_gib_s: f64,
}

/// A crossover point where one kernel starts outperforming another.
#[derive(Clone, Debug)]
pub struct Crossover {
  /// Kernel that was faster for smaller buffers.
  pub from_kernel: String,

  /// Kernel that becomes faster at the crossover point.
  pub to_kernel: String,

  /// Buffer size at which the crossover occurs.
  pub crossover_size: usize,

  /// Margin of victory at 2x the crossover size (percentage).
  pub margin_percent: f64,

  /// Statistical confidence in this crossover (0.0 to 1.0).
  pub confidence: f64,
}

/// Measurement data for analysis.
#[derive(Clone, Debug)]
pub struct Measurement {
  /// Kernel name.
  pub kernel: String,

  /// Stream count used.
  pub streams: u8,

  /// Buffer size.
  pub size: usize,

  /// Measured throughput in GiB/s.
  pub throughput_gib_s: f64,
}

impl From<&BenchResult> for Measurement {
  fn from(r: &BenchResult) -> Self {
    Self {
      kernel: r.kernel.to_string(),
      streams: 1, // Default; can be overridden
      size: r.buffer_size,
      throughput_gib_s: r.throughput_gib_s,
    }
  }
}

impl Measurement {
  /// Create a measurement with a specific stream count.
  #[must_use]
  pub fn with_streams(result: &BenchResult, streams: u8) -> Self {
    Self {
      kernel: result.kernel.to_string(),
      streams,
      size: result.buffer_size,
      throughput_gib_s: result.throughput_gib_s,
    }
  }
}

/// Find the crossover point where kernel B starts consistently beating kernel A.
///
/// Returns the smallest buffer size at which B beats A by at least `margin` percent,
/// AND B continues to beat A at all larger sizes in the measurement set.
pub fn find_crossover(
  measurements: &[Measurement],
  from_kernel: &str,
  from_streams: u8,
  to_kernel: &str,
  to_streams: u8,
  margin: f64,
) -> Option<Crossover> {
  // Filter measurements for each kernel
  let from_data: Vec<_> = measurements
    .iter()
    .filter(|m| m.kernel == from_kernel && m.streams == from_streams)
    .collect();

  let to_data: Vec<_> = measurements
    .iter()
    .filter(|m| m.kernel == to_kernel && m.streams == to_streams)
    .collect();

  if from_data.is_empty() || to_data.is_empty() {
    return None;
  }

  // Get all sizes where we have both measurements
  let mut common_sizes: Vec<usize> = from_data
    .iter()
    .map(|m| m.size)
    .filter(|&size| to_data.iter().any(|m| m.size == size))
    .collect();
  common_sizes.sort_unstable();

  if common_sizes.is_empty() {
    return None;
  }

  // Find the crossover point using "sustained by" logic:
  // Scan from largest to smallest, find smallest size where B wins consistently
  let mut threshold: Option<usize> = None;
  let mut suffix_ok = true;

  for &size in common_sizes.iter().rev() {
    let from_tp = from_data
      .iter()
      .find(|m| m.size == size)
      .map(|m| m.throughput_gib_s)
      .unwrap_or(0.0);

    let to_tp = to_data
      .iter()
      .find(|m| m.size == size)
      .map(|m| m.throughput_gib_s)
      .unwrap_or(0.0);

    // Check if 'to' beats 'from' by at least margin
    let ratio = if from_tp > 0.0 { to_tp / from_tp } else { 1.0 };

    if ratio >= margin {
      if suffix_ok {
        threshold = Some(size);
      }
    } else {
      suffix_ok = false;
    }
  }

  let crossover_size = threshold?;

  // Calculate margin at 2x the crossover size
  let double_size = crossover_size.strict_mul(2);
  let margin_percent = calculate_margin_at_size(&from_data, &to_data, double_size);

  // Calculate confidence based on consistency
  let confidence = calculate_confidence(&from_data, &to_data, crossover_size);

  Some(Crossover {
    from_kernel: from_kernel.to_string(),
    to_kernel: to_kernel.to_string(),
    crossover_size,
    margin_percent,
    confidence,
  })
}

/// Calculate the margin of victory at a specific size.
fn calculate_margin_at_size(from_data: &[&Measurement], to_data: &[&Measurement], size: usize) -> f64 {
  let from_tp = from_data
    .iter()
    .filter(|m| m.size >= size)
    .map(|m| m.throughput_gib_s)
    .next()
    .unwrap_or(0.0);

  let to_tp = to_data
    .iter()
    .filter(|m| m.size >= size)
    .map(|m| m.throughput_gib_s)
    .next()
    .unwrap_or(0.0);

  if from_tp > 0.0 {
    ((to_tp / from_tp) - 1.0) * 100.0
  } else {
    0.0
  }
}

/// Calculate confidence score based on consistency of results.
fn calculate_confidence(from_data: &[&Measurement], to_data: &[&Measurement], crossover_size: usize) -> f64 {
  // Count how many sizes above crossover have consistent wins
  let mut wins: u32 = 0;
  let mut total: u32 = 0;

  for from_m in from_data.iter().filter(|m| m.size >= crossover_size) {
    if let Some(to_m) = to_data.iter().find(|m| m.size == from_m.size) {
      total = total.strict_add(1);
      if to_m.throughput_gib_s > from_m.throughput_gib_s {
        wins = wins.strict_add(1);
      }
    }
  }

  if total == 0 {
    return 0.0;
  }

  f64::from(wins) / f64::from(total)
}

/// Select the best stream count from measurements.
///
/// Returns the stream count that achieves the highest throughput at large buffer sizes.
pub fn select_best_streams(measurements: &[Measurement], kernel: &str) -> u8 {
  let relevant: Vec<_> = measurements.iter().filter(|m| m.kernel == kernel).collect();

  if relevant.is_empty() {
    return 1;
  }

  // Find the largest buffer size
  let max_size = relevant.iter().map(|m| m.size).max().unwrap_or(0);

  // Find best throughput at max size
  let best = relevant
    .iter()
    .filter(|m| m.size == max_size)
    .max_by(|a, b| a.throughput_gib_s.partial_cmp(&b.throughput_gib_s).unwrap());

  best.map(|m| m.streams).unwrap_or(1)
}

/// Find the best kernel for large buffers.
///
/// Returns the kernel name with highest throughput at the largest buffer size in the dataset.
pub fn find_best_large_kernel(measurements: &[Measurement]) -> Option<&str> {
  if measurements.is_empty() {
    return None;
  }

  let max_size = measurements.iter().map(|m| m.size).max()?;

  measurements
    .iter()
    .filter(|m| m.size == max_size)
    .max_by(|a, b| a.throughput_gib_s.partial_cmp(&b.throughput_gib_s).unwrap())
    .map(|m| m.kernel.as_str())
}

/// Estimate min_bytes_per_lane threshold.
///
/// Finds the smallest buffer size where multi-stream outperforms single-stream,
/// then divides by the stream count to get a per-lane threshold.
///
/// Returns `None` if:
/// - `target_streams` is 1 or less (single-stream doesn't need a threshold)
/// - No measurements exist for the kernel
/// - Multi-stream never outperforms single-stream
pub fn estimate_min_bytes_per_lane(measurements: &[Measurement], kernel: &str, target_streams: u8) -> Option<usize> {
  if target_streams <= 1 {
    return None;
  }

  let single: Vec<_> = measurements
    .iter()
    .filter(|m| m.kernel == kernel && m.streams == 1)
    .collect();

  let multi: Vec<_> = measurements
    .iter()
    .filter(|m| m.kernel == kernel && m.streams == target_streams)
    .collect();

  if single.is_empty() || multi.is_empty() {
    return None;
  }

  // Find crossover using sustained logic
  let mut threshold: Option<usize> = None;
  let mut suffix_ok = true;

  let mut sizes: Vec<usize> = single.iter().map(|m| m.size).collect();
  sizes.sort_unstable();

  for &size in sizes.iter().rev() {
    let single_tp = single
      .iter()
      .find(|m| m.size == size)
      .map(|m| m.throughput_gib_s)
      .unwrap_or(0.0);

    let multi_tp = multi
      .iter()
      .find(|m| m.size == size)
      .map(|m| m.throughput_gib_s)
      .unwrap_or(0.0);

    if multi_tp >= single_tp {
      if suffix_ok {
        threshold = Some(size);
      }
    } else {
      suffix_ok = false;
    }
  }

  // Convert total threshold to per-lane
  threshold.map(|total| total / (target_streams as usize))
}

/// Best configuration result: kernel name and stream count.
#[derive(Clone, Debug)]
pub struct BestConfig<'a> {
  /// Kernel name with highest throughput.
  pub kernel: &'a str,
  /// Stream count used for that measurement.
  pub streams: u8,
  /// Throughput achieved in GiB/s.
  pub throughput_gib_s: f64,
}

/// Find the best kernel and stream configuration for large buffers.
///
/// Combines `find_best_large_kernel` and `select_best_streams` into a single lookup.
/// Returns the configuration with highest throughput at the largest buffer size.
pub fn find_best_large_config(measurements: &[Measurement]) -> Option<BestConfig<'_>> {
  if measurements.is_empty() {
    return None;
  }

  let max_size = measurements.iter().map(|m| m.size).max()?;

  measurements
    .iter()
    .filter(|m| m.size == max_size)
    .max_by(|a, b| a.throughput_gib_s.partial_cmp(&b.throughput_gib_s).unwrap())
    .map(|m| BestConfig {
      kernel: m.kernel.as_str(),
      streams: m.streams,
      throughput_gib_s: m.throughput_gib_s,
    })
}

/// Crossover type classification for threshold naming.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossoverType {
  /// Portable → SIMD (folding tier)
  PortableToSimd,
  /// SIMD (folding tier) → Wide SIMD
  SimdToWide,
  /// Single-stream → Multi-stream
  SingleToMulti,
}

impl CrossoverType {
  /// Get the threshold name for this crossover type.
  #[must_use]
  pub fn threshold_name(&self) -> &'static str {
    match self {
      Self::PortableToSimd => "portable_to_simd",
      Self::SimdToWide => "simd_to_wide",
      Self::SingleToMulti => "single_to_multi",
    }
  }
}

/// Extended crossover with type classification.
#[derive(Clone, Debug)]
pub struct TypedCrossover {
  /// Crossover type.
  pub crossover_type: CrossoverType,
  /// The underlying crossover data.
  pub crossover: Crossover,
}

/// Find crossover between kernel tiers.
///
/// This is a higher-level function that detects tier transitions:
/// - Portable → Folding (SIMD)
/// - Folding → Wide
///
/// Use this when you have tier information about your kernels.
pub fn find_tier_crossover(
  measurements: &[Measurement],
  from_kernel: &str,
  from_streams: u8,
  to_kernel: &str,
  to_streams: u8,
  crossover_type: CrossoverType,
  margin: f64,
) -> Option<TypedCrossover> {
  find_crossover(measurements, from_kernel, from_streams, to_kernel, to_streams, margin).map(|crossover| {
    TypedCrossover {
      crossover_type,
      crossover,
    }
  })
}

/// Analyze measurements and produce complete analysis result.
pub fn analyze(measurements: &[Measurement], portable_kernel: &str) -> AnalysisResult {
  if measurements.is_empty() {
    return AnalysisResult::default();
  }

  let mut result = AnalysisResult::default();

  // Find peak throughput
  if let Some(peak) = measurements
    .iter()
    .max_by(|a, b| a.throughput_gib_s.partial_cmp(&b.throughput_gib_s).unwrap())
  {
    result.peak_throughput_gib_s = peak.throughput_gib_s;
  }

  // Find best kernel for large buffers
  let max_size = measurements.iter().map(|m| m.size).max().unwrap_or(0);
  if let Some(best) = measurements
    .iter()
    .filter(|m| m.size == max_size)
    .max_by(|a, b| a.throughput_gib_s.partial_cmp(&b.throughput_gib_s).unwrap())
  {
    // Find the static kernel name - this is a workaround since we store String
    // but need &'static str. In practice, kernel names are static.
    result.best_large_kernel = None; // Will be set by caller with static str
    result.recommended_streams = best.streams;
  }

  // Find portable → SIMD crossover
  let simd_kernels: Vec<_> = measurements
    .iter()
    .filter(|m| m.kernel != portable_kernel)
    .map(|m| (m.kernel.as_str(), m.streams))
    .collect::<std::collections::HashSet<_>>()
    .into_iter()
    .collect();

  for (simd_kernel, streams) in simd_kernels {
    if let Some(crossover) = find_crossover(
      measurements,
      portable_kernel,
      1,
      simd_kernel,
      streams,
      1.0, // 0% margin (just needs to be equal or better)
    ) {
      result.crossovers.push(crossover);
    }
  }

  // Sort crossovers by size
  result.crossovers.sort_by_key(|c| c.crossover_size);

  // Calculate overall confidence
  if !result.crossovers.is_empty() {
    result.confidence = result.crossovers.iter().map(|c| c.confidence).sum::<f64>() / result.crossovers.len() as f64;
  }

  result
}
