//! Raw measurement artifacts for offline policy derivation.
//!
//! The raw artifact is the stable boundary between:
//! - **measurement** (expensive benchmarking on target hardware)
//! - **derivation** (deterministic policy computation offline)

use alloc::collections::{BTreeMap, BTreeSet};
use std::{
  fs::File,
  io::{self, BufReader, BufWriter},
  path::Path,
};

use backend::KernelTier;
use serde::{Deserialize, Serialize};

use crate::{BenchResult, KernelSpec, analysis::Measurement};

/// Current on-disk schema version for raw tune artifacts.
pub const RAW_SCHEMA_VERSION: u32 = 2;

/// Aggregation mode for combining repeated measurement runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregationMode {
  Auto,
  Median,
  TrimmedMean,
}

impl AggregationMode {
  #[must_use]
  pub fn parse(s: &str) -> Option<Self> {
    match s.trim().to_ascii_lowercase().as_str() {
      "auto" => Some(Self::Auto),
      "median" => Some(Self::Median),
      "trimmed-mean" | "trimmed_mean" | "trimmedmean" | "trim" => Some(Self::TrimmedMean),
      _ => None,
    }
  }

  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Median => "median",
      Self::TrimmedMean => "trimmed-mean",
    }
  }
}

/// Raw platform metadata persisted with measurement artifacts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawPlatformInfo {
  pub arch: String,
  pub os: String,
  pub tune_kind: u8,
  pub description: String,
  pub caps: String,
}

/// Runner settings captured at measurement time.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RawRunnerConfig {
  pub warmup_ms: u64,
  pub measure_ms: u64,
  pub warn_high_variance: bool,
  pub cv_threshold: f64,
}

/// Raw kernel metadata for one algorithm.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RawKernelSpec {
  pub name: String,
  pub tier: String,
  pub min_streams: u8,
  pub max_streams: u8,
}

impl RawKernelSpec {
  #[must_use]
  pub fn from_kernel_spec(spec: &KernelSpec) -> Self {
    let (min_streams, max_streams) = spec.streams.unwrap_or((1, 1));
    Self {
      name: spec.name.to_string(),
      tier: spec.tier.name().to_string(),
      min_streams,
      max_streams,
    }
  }

  #[must_use]
  pub fn tier(&self) -> Option<KernelTier> {
    match self.tier.as_str() {
      "reference" => Some(KernelTier::Reference),
      "portable" => Some(KernelTier::Portable),
      "hwcrc" => Some(KernelTier::Hardware),
      "folding" => Some(KernelTier::Folding),
      "wide" => Some(KernelTier::Wide),
      _ => None,
    }
  }
}

/// Raw benchmark point captured during measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawBenchPoint {
  pub kernel: String,
  pub streams: u8,
  pub size: usize,
  pub throughput_gib_s: f64,
  pub iterations: u64,
  pub bytes_processed: u64,
  pub elapsed_secs: f64,
  pub sample_count: Option<usize>,
  pub std_dev: Option<f64>,
  pub cv: Option<f64>,
  pub outliers_rejected: Option<usize>,
  pub min_throughput_gib_s: Option<f64>,
  pub max_throughput_gib_s: Option<f64>,
}

impl RawBenchPoint {
  #[must_use]
  pub fn from_result(kernel: &str, streams: u8, result: &BenchResult) -> Self {
    Self {
      kernel: kernel.to_string(),
      streams,
      size: result.buffer_size,
      throughput_gib_s: result.throughput_gib_s,
      iterations: result.iterations,
      bytes_processed: result.bytes_processed,
      elapsed_secs: result.elapsed_secs,
      sample_count: result.sample_count,
      std_dev: result.std_dev,
      cv: result.cv,
      outliers_rejected: result.outliers_rejected,
      min_throughput_gib_s: result.min_throughput_gib_s,
      max_throughput_gib_s: result.max_throughput_gib_s,
    }
  }

  #[must_use]
  pub fn to_measurement(&self) -> Measurement {
    Measurement {
      kernel: self.kernel.clone(),
      streams: self.streams,
      size: self.size,
      throughput_gib_s: self.throughput_gib_s,
    }
  }
}

/// Throughput point used by BLAKE3 parallel policy fitting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawThroughputPoint {
  pub size: usize,
  pub throughput_gib_s: f64,
  pub sample_count: Option<usize>,
  pub std_dev: Option<f64>,
  pub cv: Option<f64>,
  pub outliers_rejected: Option<usize>,
  pub min_throughput_gib_s: Option<f64>,
  pub max_throughput_gib_s: Option<f64>,
}

impl RawThroughputPoint {
  #[must_use]
  pub fn from_result(result: &BenchResult) -> Self {
    Self {
      size: result.buffer_size,
      throughput_gib_s: result.throughput_gib_s,
      sample_count: result.sample_count,
      std_dev: result.std_dev,
      cv: result.cv,
      outliers_rejected: result.outliers_rejected,
      min_throughput_gib_s: result.min_throughput_gib_s,
      max_throughput_gib_s: result.max_throughput_gib_s,
    }
  }
}

/// Throughput curve for one BLAKE3 parallel thread cap.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawBlake3ParallelCurve {
  pub max_threads: usize,
  pub throughput: Vec<RawThroughputPoint>,
}

/// Raw BLAKE3 parallel-policy measurement set.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawBlake3ParallelData {
  pub available_parallelism: usize,
  pub single: Vec<RawThroughputPoint>,
  pub curves: Vec<RawBlake3ParallelCurve>,
}

/// Raw per-algorithm measurement bundle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawAlgorithmMeasurements {
  pub name: String,
  pub env_prefix: String,
  pub domain: String,
  pub kernels: Vec<RawKernelSpec>,
  pub stream_measurements: Vec<RawBenchPoint>,
  pub threshold_measurements: Vec<RawBenchPoint>,
  pub size_class_probe_measurements: Vec<RawBenchPoint>,
  pub blake3_parallel: Option<RawBlake3ParallelData>,
}

/// Top-level raw tune artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawTuneResults {
  pub schema_version: u32,
  pub timestamp: String,
  pub quick_mode: bool,
  pub run_count: usize,
  pub aggregation: String,
  pub platform: RawPlatformInfo,
  pub checksum_runner: RawRunnerConfig,
  pub hash_runner: RawRunnerConfig,
  pub algorithms: Vec<RawAlgorithmMeasurements>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct BenchKey {
  kernel: String,
  streams: u8,
  size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ThroughputKey {
  size: usize,
}

fn collect_bench_keys(points: &[RawBenchPoint]) -> BTreeSet<BenchKey> {
  points
    .iter()
    .map(|point| BenchKey {
      kernel: point.kernel.clone(),
      streams: point.streams,
      size: point.size,
    })
    .collect()
}

fn collect_throughput_keys(points: &[RawThroughputPoint]) -> BTreeSet<ThroughputKey> {
  points.iter().map(|point| ThroughputKey { size: point.size }).collect()
}

fn aggregate_f64(mut values: Vec<f64>, mode: AggregationMode) -> f64 {
  if values.is_empty() {
    return 0.0;
  }
  values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
  let n = values.len();
  let effective = match mode {
    AggregationMode::Auto => {
      if n >= 5 {
        AggregationMode::TrimmedMean
      } else {
        AggregationMode::Median
      }
    }
    m => m,
  };
  match effective {
    AggregationMode::Auto => unreachable!("resolved above"),
    AggregationMode::Median => values[n / 2],
    AggregationMode::TrimmedMean => {
      if n < 5 {
        return values[n / 2];
      }
      let trim = (n / 5).max(1);
      let keep = &values[trim..n.saturating_sub(trim)];
      if keep.is_empty() {
        return values[n / 2];
      }
      keep.iter().sum::<f64>() / keep.len() as f64
    }
  }
}

fn aggregate_u64_median(mut values: Vec<u64>) -> u64 {
  if values.is_empty() {
    return 0;
  }
  values.sort_unstable();
  values[values.len() / 2]
}

fn aggregate_usize_median(mut values: Vec<usize>) -> usize {
  if values.is_empty() {
    return 0;
  }
  values.sort_unstable();
  values[values.len() / 2]
}

fn aggregate_opt_f64(values: impl Iterator<Item = Option<f64>>, mode: AggregationMode) -> Option<f64> {
  let collected: Vec<f64> = values.flatten().collect();
  if collected.is_empty() {
    None
  } else {
    Some(aggregate_f64(collected, mode))
  }
}

fn aggregate_opt_usize_median(values: impl Iterator<Item = Option<usize>>) -> Option<usize> {
  let collected: Vec<usize> = values.flatten().collect();
  if collected.is_empty() {
    None
  } else {
    Some(aggregate_usize_median(collected))
  }
}

fn aggregate_bench_points(point_sets: &[&[RawBenchPoint]], mode: AggregationMode) -> Vec<RawBenchPoint> {
  let mut grouped: BTreeMap<BenchKey, Vec<&RawBenchPoint>> = BTreeMap::new();
  for points in point_sets {
    for point in *points {
      grouped
        .entry(BenchKey {
          kernel: point.kernel.clone(),
          streams: point.streams,
          size: point.size,
        })
        .or_default()
        .push(point);
    }
  }

  grouped
    .into_iter()
    .map(|(key, group)| RawBenchPoint {
      kernel: key.kernel,
      streams: key.streams,
      size: key.size,
      throughput_gib_s: aggregate_f64(group.iter().map(|p| p.throughput_gib_s).collect(), mode),
      iterations: aggregate_u64_median(group.iter().map(|p| p.iterations).collect()),
      bytes_processed: aggregate_u64_median(group.iter().map(|p| p.bytes_processed).collect()),
      elapsed_secs: aggregate_f64(group.iter().map(|p| p.elapsed_secs).collect(), mode),
      sample_count: aggregate_opt_usize_median(group.iter().map(|p| p.sample_count)),
      std_dev: aggregate_opt_f64(group.iter().map(|p| p.std_dev), mode),
      cv: aggregate_opt_f64(group.iter().map(|p| p.cv), mode),
      outliers_rejected: aggregate_opt_usize_median(group.iter().map(|p| p.outliers_rejected)),
      min_throughput_gib_s: aggregate_opt_f64(group.iter().map(|p| p.min_throughput_gib_s), mode),
      max_throughput_gib_s: aggregate_opt_f64(group.iter().map(|p| p.max_throughput_gib_s), mode),
    })
    .collect()
}

fn aggregate_throughput_points(point_sets: &[&[RawThroughputPoint]], mode: AggregationMode) -> Vec<RawThroughputPoint> {
  let mut grouped: BTreeMap<ThroughputKey, Vec<&RawThroughputPoint>> = BTreeMap::new();
  for points in point_sets {
    for point in *points {
      grouped
        .entry(ThroughputKey { size: point.size })
        .or_default()
        .push(point);
    }
  }

  grouped
    .into_iter()
    .map(|(key, group)| RawThroughputPoint {
      size: key.size,
      throughput_gib_s: aggregate_f64(group.iter().map(|p| p.throughput_gib_s).collect(), mode),
      sample_count: aggregate_opt_usize_median(group.iter().map(|p| p.sample_count)),
      std_dev: aggregate_opt_f64(group.iter().map(|p| p.std_dev), mode),
      cv: aggregate_opt_f64(group.iter().map(|p| p.cv), mode),
      outliers_rejected: aggregate_opt_usize_median(group.iter().map(|p| p.outliers_rejected)),
      min_throughput_gib_s: aggregate_opt_f64(group.iter().map(|p| p.min_throughput_gib_s), mode),
      max_throughput_gib_s: aggregate_opt_f64(group.iter().map(|p| p.max_throughput_gib_s), mode),
    })
    .collect()
}

fn aggregate_blake3_parallel(
  data_sets: &[&RawBlake3ParallelData],
  mode: AggregationMode,
) -> io::Result<RawBlake3ParallelData> {
  let first = data_sets
    .first()
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no blake3 parallel data sets to aggregate"))?;
  let first_single_keys = collect_throughput_keys(&first.single);
  let first_curve_threads: BTreeSet<usize> = first.curves.iter().map(|curve| curve.max_threads).collect();
  let mut first_curve_keys: BTreeMap<usize, BTreeSet<ThroughputKey>> = BTreeMap::new();
  for curve in &first.curves {
    first_curve_keys.insert(curve.max_threads, collect_throughput_keys(&curve.throughput));
  }

  for data in &data_sets[1..] {
    let single_keys = collect_throughput_keys(&data.single);
    if single_keys != first_single_keys {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "blake3 single-curve point set mismatch during aggregation",
      ));
    }

    let curve_threads: BTreeSet<usize> = data.curves.iter().map(|curve| curve.max_threads).collect();
    if curve_threads != first_curve_threads {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "blake3 thread-curve set mismatch during aggregation",
      ));
    }

    for curve in &data.curves {
      let Some(expected_keys) = first_curve_keys.get(&curve.max_threads) else {
        return Err(io::Error::new(
          io::ErrorKind::InvalidData,
          "blake3 curve key mismatch during aggregation",
        ));
      };
      let keys = collect_throughput_keys(&curve.throughput);
      if &keys != expected_keys {
        return Err(io::Error::new(
          io::ErrorKind::InvalidData,
          format!(
            "blake3 throughput point set mismatch for thread cap {} during aggregation",
            curve.max_threads
          ),
        ));
      }
    }
  }

  let available_parallelism = data_sets.iter().map(|d| d.available_parallelism).min().unwrap_or(1);

  let single_sets: Vec<&[RawThroughputPoint]> = data_sets.iter().map(|d| d.single.as_slice()).collect();
  let single = aggregate_throughput_points(&single_sets, mode);

  let mut max_threads_set = BTreeSet::new();
  for data in data_sets {
    for curve in &data.curves {
      max_threads_set.insert(curve.max_threads);
    }
  }

  let mut curves = Vec::new();
  for max_threads in max_threads_set {
    let mut throughput_sets = Vec::new();
    for data in data_sets {
      if let Some(curve) = data.curves.iter().find(|c| c.max_threads == max_threads) {
        throughput_sets.push(curve.throughput.as_slice());
      }
    }
    if throughput_sets.is_empty() {
      continue;
    }
    curves.push(RawBlake3ParallelCurve {
      max_threads,
      throughput: aggregate_throughput_points(&throughput_sets, mode),
    });
  }

  if !single.is_empty() && curves.is_empty() {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      "cannot aggregate blake3 parallel data: no thread curves available",
    ));
  }

  Ok(RawBlake3ParallelData {
    available_parallelism,
    single,
    curves,
  })
}

fn aggregate_algorithm_runs(
  runs: &[&RawAlgorithmMeasurements],
  mode: AggregationMode,
) -> io::Result<RawAlgorithmMeasurements> {
  let first = runs
    .first()
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no algorithm runs to aggregate"))?;
  let first_stream_keys = collect_bench_keys(&first.stream_measurements);
  let first_threshold_keys = collect_bench_keys(&first.threshold_measurements);
  let first_probe_keys = collect_bench_keys(&first.size_class_probe_measurements);

  for run in &runs[1..] {
    if run.name != first.name {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("algorithm mismatch during aggregation: {} != {}", run.name, first.name),
      ));
    }
    if run.env_prefix != first.env_prefix || run.domain != first.domain {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("metadata mismatch during aggregation for algorithm {}", first.name),
      ));
    }
    if run.kernels != first.kernels {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("kernel set mismatch during aggregation for algorithm {}", first.name),
      ));
    }
    if collect_bench_keys(&run.stream_measurements) != first_stream_keys {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
          "stream measurement point set mismatch during aggregation for algorithm {}",
          first.name
        ),
      ));
    }
    if collect_bench_keys(&run.threshold_measurements) != first_threshold_keys {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
          "threshold measurement point set mismatch during aggregation for algorithm {}",
          first.name
        ),
      ));
    }
    if collect_bench_keys(&run.size_class_probe_measurements) != first_probe_keys {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
          "size-class probe measurement point set mismatch during aggregation for algorithm {}",
          first.name
        ),
      ));
    }
  }

  let stream_sets: Vec<&[RawBenchPoint]> = runs.iter().map(|r| r.stream_measurements.as_slice()).collect();
  let threshold_sets: Vec<&[RawBenchPoint]> = runs.iter().map(|r| r.threshold_measurements.as_slice()).collect();
  let probe_sets: Vec<&[RawBenchPoint]> = runs
    .iter()
    .map(|r| r.size_class_probe_measurements.as_slice())
    .collect();

  let blake3_parallel = match runs.iter().map(|r| r.blake3_parallel.as_ref()).collect::<Vec<_>>() {
    values if values.iter().all(|v| v.is_none()) => None,
    values if values.iter().all(|v| v.is_some()) => {
      let data_sets: Vec<&RawBlake3ParallelData> = values.into_iter().flatten().collect();
      Some(aggregate_blake3_parallel(&data_sets, mode)?)
    }
    _ => {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
          "blake3_parallel presence mismatch during aggregation for algorithm {}",
          first.name
        ),
      ));
    }
  };

  Ok(RawAlgorithmMeasurements {
    name: first.name.clone(),
    env_prefix: first.env_prefix.clone(),
    domain: first.domain.clone(),
    kernels: first.kernels.clone(),
    stream_measurements: aggregate_bench_points(&stream_sets, mode),
    threshold_measurements: aggregate_bench_points(&threshold_sets, mode),
    size_class_probe_measurements: aggregate_bench_points(&probe_sets, mode),
    blake3_parallel,
  })
}

/// Aggregate repeated raw measurement runs into one deterministic artifact.
pub fn aggregate_raw_results(runs: &[RawTuneResults], mode: AggregationMode) -> io::Result<RawTuneResults> {
  let first = runs
    .first()
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no raw runs to aggregate"))?;
  if runs.len() == 1 {
    return Ok(first.clone());
  }

  for run in &runs[1..] {
    if run.schema_version != first.schema_version {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
          "schema mismatch during aggregation: {} != {}",
          run.schema_version, first.schema_version
        ),
      ));
    }
    if run.quick_mode != first.quick_mode {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "quick_mode mismatch during aggregation",
      ));
    }
    if run.platform.arch != first.platform.arch
      || run.platform.os != first.platform.os
      || run.platform.tune_kind != first.platform.tune_kind
      || run.platform.caps != first.platform.caps
    {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "platform mismatch during aggregation",
      ));
    }
    if run.checksum_runner != first.checksum_runner || run.hash_runner != first.hash_runner {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "runner configuration mismatch during aggregation",
      ));
    }
  }

  let mut algorithms = Vec::with_capacity(first.algorithms.len());
  for algo in &first.algorithms {
    let mut matching = Vec::with_capacity(runs.len());
    for run in runs {
      let found = run
        .algorithms
        .iter()
        .find(|candidate| candidate.name == algo.name)
        .ok_or_else(|| {
          io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing algorithm {} in one or more runs", algo.name),
          )
        })?;
      matching.push(found);
    }
    algorithms.push(aggregate_algorithm_runs(&matching, mode)?);
  }

  Ok(RawTuneResults {
    schema_version: first.schema_version,
    timestamp: runs
      .last()
      .map(|r| r.timestamp.clone())
      .unwrap_or_else(|| first.timestamp.clone()),
    quick_mode: first.quick_mode,
    run_count: runs.len(),
    aggregation: mode.as_str().to_string(),
    platform: first.platform.clone(),
    checksum_runner: first.checksum_runner.clone(),
    hash_runner: first.hash_runner.clone(),
    algorithms,
  })
}

/// Write raw tune results to JSON on disk.
pub fn write_raw_results(path: &Path, raw: &RawTuneResults) -> io::Result<()> {
  let file = File::create(path)?;
  let writer = BufWriter::new(file);
  serde_json::to_writer_pretty(writer, raw).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

/// Read raw tune results from JSON on disk.
pub fn read_raw_results(path: &Path) -> io::Result<RawTuneResults> {
  let file = File::open(path)?;
  let reader = BufReader::new(file);
  let raw: RawTuneResults =
    serde_json::from_reader(reader).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

  if raw.schema_version != RAW_SCHEMA_VERSION {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      format!(
        "unsupported raw schema version {} (expected {})",
        raw.schema_version, RAW_SCHEMA_VERSION
      ),
    ));
  }

  Ok(raw)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn run_with_throughput(throughput: f64, size_override: Option<usize>) -> RawTuneResults {
    let size = size_override.unwrap_or(1024);
    let point = RawBenchPoint {
      kernel: "portable/slice16".to_string(),
      streams: 1,
      size,
      throughput_gib_s: throughput,
      iterations: 100,
      bytes_processed: 100 * size as u64,
      elapsed_secs: 0.01,
      sample_count: Some(8),
      std_dev: Some(0.1),
      cv: Some(0.01),
      outliers_rejected: Some(0),
      min_throughput_gib_s: Some(throughput * 0.95),
      max_throughput_gib_s: Some(throughput * 1.05),
    };

    RawTuneResults {
      schema_version: RAW_SCHEMA_VERSION,
      timestamp: "2026-02-10T00:00:00Z".to_string(),
      quick_mode: false,
      run_count: 1,
      aggregation: "single".to_string(),
      platform: RawPlatformInfo {
        arch: "aarch64".to_string(),
        os: "macos".to_string(),
        tune_kind: 9,
        description: "test".to_string(),
        caps: "none".to_string(),
      },
      checksum_runner: RawRunnerConfig {
        warmup_ms: 100,
        measure_ms: 200,
        warn_high_variance: false,
        cv_threshold: 0.05,
      },
      hash_runner: RawRunnerConfig {
        warmup_ms: 100,
        measure_ms: 300,
        warn_high_variance: false,
        cv_threshold: 0.05,
      },
      algorithms: vec![RawAlgorithmMeasurements {
        name: "crc32c".to_string(),
        env_prefix: "RSCRYPTO_CRC32C".to_string(),
        domain: "checksum".to_string(),
        kernels: vec![RawKernelSpec {
          name: "portable/slice16".to_string(),
          tier: "portable".to_string(),
          min_streams: 1,
          max_streams: 1,
        }],
        stream_measurements: vec![point.clone()],
        threshold_measurements: vec![point.clone()],
        size_class_probe_measurements: vec![point],
        blake3_parallel: None,
      }],
    }
  }

  #[test]
  fn aggregate_auto_uses_median_for_three_runs() {
    let runs = vec![
      run_with_throughput(10.0, None),
      run_with_throughput(100.0, None),
      run_with_throughput(12.0, None),
    ];
    let aggregated = aggregate_raw_results(&runs, AggregationMode::Auto).expect("aggregation should succeed");
    assert_eq!(aggregated.run_count, 3);
    assert_eq!(aggregated.aggregation, "auto");
    let value = aggregated.algorithms[0].stream_measurements[0].throughput_gib_s;
    assert_eq!(value, 12.0);
  }

  #[test]
  fn aggregate_auto_uses_trimmed_mean_for_five_runs() {
    let runs = vec![
      run_with_throughput(10.0, None),
      run_with_throughput(11.0, None),
      run_with_throughput(100.0, None),
      run_with_throughput(12.0, None),
      run_with_throughput(13.0, None),
    ];
    let aggregated = aggregate_raw_results(&runs, AggregationMode::Auto).expect("aggregation should succeed");
    let value = aggregated.algorithms[0].stream_measurements[0].throughput_gib_s;
    assert!((value - 12.0).abs() < f64::EPSILON);
  }

  #[test]
  fn aggregate_rejects_point_set_mismatch() {
    let runs = vec![run_with_throughput(10.0, None), run_with_throughput(11.0, Some(2048))];
    let err = aggregate_raw_results(&runs, AggregationMode::Median).expect_err("aggregation should fail");
    assert!(err.to_string().contains("point set mismatch"));
  }
}
