//! Raw measurement artifacts for offline policy derivation.
//!
//! The raw artifact is the stable boundary between:
//! - **measurement** (expensive benchmarking on target hardware)
//! - **derivation** (deterministic policy computation offline)

use std::{
  fs::File,
  io::{self, BufReader, BufWriter},
  path::Path,
};

use backend::KernelTier;
use serde::{Deserialize, Serialize};

use crate::{BenchResult, KernelSpec, analysis::Measurement};

/// Current on-disk schema version for raw tune artifacts.
pub const RAW_SCHEMA_VERSION: u32 = 1;

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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawRunnerConfig {
  pub warmup_ms: u64,
  pub measure_ms: u64,
  pub warn_high_variance: bool,
  pub cv_threshold: f64,
}

/// Raw kernel metadata for one algorithm.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
  pub platform: RawPlatformInfo,
  pub checksum_runner: RawRunnerConfig,
  pub hash_runner: RawRunnerConfig,
  pub algorithms: Vec<RawAlgorithmMeasurements>,
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
