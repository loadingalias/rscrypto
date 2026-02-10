//! Tuning engine orchestrator.

mod blake3_adapter;
mod crc_adapter;

use std::{collections::HashSet, time::Instant};

use crate::{
  AlgorithmResult, KernelSpec, KernelTier, PlatformInfo, RawAlgorithmMeasurements, RawBenchPoint, RawKernelSpec,
  RawPlatformInfo, RawRunnerConfig, RawTuneResults, SizeClassBest, Tunable, TuneError, TuneResults, TuningDomain,
  analysis::{self, CrossoverType, Measurement},
  crc16::crc16_threshold_to_env_suffix,
  crc24::crc24_threshold_to_env_suffix,
  crc32::crc32_threshold_to_env_suffix,
  crc64::crc64_threshold_to_env_suffix,
  hash::hash_threshold_to_env_suffix,
  raw::RAW_SCHEMA_VERSION,
  runner::{
    BenchRunner, SIZE_CLASS_NAMES, SIZE_CLASS_SIZES, STREAM_SIZES, STREAM_SIZES_BENCH, THRESHOLD_SIZES, fill_data,
    stream_candidates,
  },
};

/// Main tuning engine that orchestrates benchmarking for multiple algorithms.
pub struct TuneEngine {
  /// Algorithms to tune.
  algorithms: Vec<Box<dyn Tunable>>,

  /// Benchmark runner configuration for checksum-style algorithms.
  checksum_runner: BenchRunner,

  /// Benchmark runner configuration for hash-style algorithms.
  hash_runner: BenchRunner,

  /// Whether to run in verbose mode.
  verbose: bool,
}

impl Default for TuneEngine {
  fn default() -> Self {
    Self::new()
  }
}

impl TuneEngine {
  /// Create a new tuning engine with default settings.
  #[must_use]
  pub fn new() -> Self {
    Self {
      algorithms: Vec::new(),
      checksum_runner: BenchRunner::default(),
      hash_runner: BenchRunner::hash(),
      verbose: false,
    }
  }

  /// Create a tuning engine with quick mode settings.
  #[must_use]
  pub fn quick() -> Self {
    Self {
      algorithms: Vec::new(),
      checksum_runner: BenchRunner::quick(),
      hash_runner: BenchRunner::quick_hash(),
      verbose: false,
    }
  }

  /// Set the benchmark runner for all domains.
  #[must_use]
  pub fn with_runner(mut self, runner: BenchRunner) -> Self {
    self.checksum_runner = runner.clone();
    self.hash_runner = runner;
    self
  }

  /// Set the benchmark runner for checksum-style algorithms.
  #[must_use]
  pub fn with_checksum_runner(mut self, runner: BenchRunner) -> Self {
    self.checksum_runner = runner;
    self
  }

  /// Set the benchmark runner for hash-style algorithms.
  #[must_use]
  pub fn with_hash_runner(mut self, runner: BenchRunner) -> Self {
    self.hash_runner = runner;
    self
  }

  /// Enable verbose output.
  #[must_use]
  pub fn with_verbose(mut self, verbose: bool) -> Self {
    self.verbose = verbose;
    self
  }

  /// Add an algorithm to tune.
  pub fn add(&mut self, algorithm: Box<dyn Tunable>) {
    self.algorithms.push(algorithm);
  }

  /// Retain only the requested algorithms by name.
  ///
  /// `only` is a list of algorithm names as reported by `Tunable::name()`
  /// (e.g., `"blake3"`, `"blake3-chunk"`, `"blake3-stream4k"`).
  ///
  /// Returns the number of algorithms remaining after filtering.
  #[must_use]
  pub fn retain_only(&mut self, only: &[String]) -> usize {
    if only.is_empty() {
      return self.algorithms.len();
    }

    let wanted: HashSet<String> = only.iter().cloned().collect();
    self.algorithms.retain(|a| wanted.contains(a.name()));
    self.algorithms.len()
  }

  /// Run the complete tuning suite.
  pub fn run(&mut self) -> Result<TuneResults, TuneError> {
    let raw = self.measure(false)?;
    Self::derive_from_raw(&raw)
  }

  /// Run measurement phase and return raw per-point artifacts.
  pub fn measure(&mut self, quick_mode: bool) -> Result<RawTuneResults, TuneError> {
    let platform = PlatformInfo::collect();

    if self.verbose {
      eprintln!("Platform: {}", platform.description);
      eprintln!("Tune preset: {:?}", platform.tune_kind);
      eprintln!();
    }

    let mut raw_algorithms = Vec::with_capacity(self.algorithms.len());
    let verbose = self.verbose;
    let total = self.algorithms.len();

    for i in 0..self.algorithms.len() {
      let algorithm = &mut self.algorithms[i];

      let start = Instant::now();
      eprintln!("Tuning {} ({}/{})...", algorithm.name(), i + 1, total);

      let runner = match algorithm.tuning_domain() {
        TuningDomain::Checksum => &self.checksum_runner,
        TuningDomain::Hash => &self.hash_runner,
      };
      let raw = measure_algorithm_impl(runner, algorithm.as_mut(), &platform)?;

      if verbose {
        let stream_measurements = to_measurements(&raw.stream_measurements);
        let (best_kernel, best_streams) = find_best_config(&stream_measurements);
        eprintln!("  Best kernel: {best_kernel} ({best_streams} stream(s))");
      }
      eprintln!("  Done in {:.1}s", start.elapsed().as_secs_f64());
      if verbose {
        eprintln!();
      }

      raw_algorithms.push(raw);
    }

    let timestamp = get_timestamp();

    Ok(RawTuneResults {
      schema_version: RAW_SCHEMA_VERSION,
      timestamp,
      quick_mode,
      platform: RawPlatformInfo {
        arch: platform.arch.to_string(),
        os: platform.os.to_string(),
        tune_kind: platform.tune_kind as u8,
        description: platform.description,
        caps: platform.caps.to_string(),
      },
      checksum_runner: RawRunnerConfig {
        warmup_ms: self.checksum_runner.warmup_ms(),
        measure_ms: self.checksum_runner.measure_ms(),
        warn_high_variance: self.checksum_runner.warns_high_variance(),
        cv_threshold: self.checksum_runner.cv_threshold(),
      },
      hash_runner: RawRunnerConfig {
        warmup_ms: self.hash_runner.warmup_ms(),
        measure_ms: self.hash_runner.measure_ms(),
        warn_high_variance: self.hash_runner.warns_high_variance(),
        cv_threshold: self.hash_runner.cv_threshold(),
      },
      algorithms: raw_algorithms,
    })
  }

  /// Derive dispatch policy deterministically from raw measurements.
  pub fn derive_from_raw(raw: &RawTuneResults) -> Result<TuneResults, TuneError> {
    let platform = raw_platform_to_platform_info(&raw.platform)?;
    let mut algorithms = Vec::with_capacity(raw.algorithms.len());
    for algo in &raw.algorithms {
      algorithms.push(derive_algorithm_from_raw(algo)?);
    }

    Ok(TuneResults {
      platform,
      algorithms,
      timestamp: raw.timestamp.clone(),
    })
  }
}

/// Measure one algorithm and capture all raw data needed for offline derivation.
fn measure_algorithm_impl(
  runner: &BenchRunner,
  algorithm: &mut dyn Tunable,
  platform: &PlatformInfo,
) -> Result<RawAlgorithmMeasurements, TuneError> {
  let caps = platform.caps;
  let available_kernels = algorithm.available_kernels(&caps);

  if available_kernels.is_empty() {
    return Err(TuneError::BenchmarkFailed("no kernels available"));
  }

  // Classify kernels by tier
  let portable_kernels: Vec<&KernelSpec> = available_kernels
    .iter()
    .filter(|k| k.tier == KernelTier::Portable)
    .collect();
  let folding_kernel = find_kernel_by_tier(&available_kernels, KernelTier::Folding);
  let wide_kernel = find_kernel_by_tier(&available_kernels, KernelTier::Wide);

  // Phase A: stream scan for each kernel.
  let stream_measurements = benchmark_streams(runner, algorithm, &available_kernels)?;
  let stream_measurements_view = to_measurements(&stream_measurements);

  // Select best kernel/streams from stream measurements.
  let (best_kernel, best_streams) = find_best_config(&stream_measurements_view);

  // Phase A2: collect explicit class probes (xs/s fallbacks).
  let size_class_probe_measurements =
    benchmark_size_class_probes(runner, algorithm, &available_kernels, &stream_measurements_view)?;

  // Phase B: threshold curves.
  let portable_names: Vec<&'static str> = portable_kernels.iter().map(|k| k.name).collect();
  let mut threshold_measurements = benchmark_thresholds(runner, algorithm, &portable_names, best_kernel, best_streams)?;

  // CRC-64: benchmark the small-buffer SIMD kernel explicitly so:
  // - portable→SIMD crossovers reflect the real small-kernel behavior
  // - we can tune the small-kernel window (small→best crossover)
  if matches!(algorithm.name(), "crc64-xz" | "crc64-nvme") {
    // Ensure we have a single-stream curve for the best kernel (small kernel is single-stream).
    if best_streams != 1 && !has_measurement(&threshold_measurements, best_kernel, 1) {
      let curve = benchmark_single_kernel(runner, algorithm, best_kernel, 1)?;
      threshold_measurements.extend(curve);
    }

    let small_name = if best_kernel.starts_with("x86_64/") {
      Some("x86_64/pclmul-small")
    } else if best_kernel.starts_with("aarch64/sve2-pmull") {
      Some("aarch64/sve2-pmull-small")
    } else if best_kernel.starts_with("aarch64/") {
      Some("aarch64/pmull-small")
    } else {
      None
    };

    if let Some(name) = small_name
      && !has_measurement(&threshold_measurements, name, 1)
      && let Ok(curve) = benchmark_single_kernel(runner, algorithm, name, 1)
    {
      threshold_measurements.extend(curve);
    }
  }

  // Phase C: Additional threshold curves for tier-based crossovers
  // If we have both folding and wide kernels, benchmark the folding kernel
  // to detect SIMD→wide crossover
  if let (Some(folding), Some(_wide)) = (&folding_kernel, &wide_kernel) {
    // Only benchmark folding if it's not the same as best_kernel
    if folding.name != best_kernel {
      let folding_streams = analysis::select_best_streams(&stream_measurements_view, folding.name);
      let folding_measurements = benchmark_single_kernel(runner, algorithm, folding.name, folding_streams)?;
      threshold_measurements.extend(folding_measurements);
    }
  }

  // CRC-32: benchmark all non-portable tiers across THRESHOLD_SIZES so we can
  // compute policy-specific crossovers (portable→hwcrc, hwcrc→fusion, fusion→wide).
  if crc_adapter::is_crc32_algorithm(algorithm.name()) {
    for kernel in &available_kernels {
      if matches!(kernel.tier, KernelTier::Reference | KernelTier::Portable) {
        continue;
      }

      // CRC-32 crossovers are tier boundaries, not stream-selection boundaries.
      // Benchmark all tiers at single-stream so crossover points match the policy:
      // multi-stream enters via `min_bytes_per_lane`, not via the tier thresholds.
      let streams = 1;
      if has_measurement(&threshold_measurements, kernel.name, streams) {
        continue;
      }

      let curve = benchmark_single_kernel(runner, algorithm, kernel.name, streams)?;
      threshold_measurements.extend(curve);
    }
  }

  // Phase D: Single-stream vs multi-stream comparison for min_bytes_per_lane
  // Only if best_streams > 1
  if best_streams > 1 && !has_measurement(&threshold_measurements, best_kernel, 1) {
    let single_stream_measurements = benchmark_single_kernel(runner, algorithm, best_kernel, 1)?;
    threshold_measurements.extend(single_stream_measurements);
  }

  let blake3_parallel = if algorithm.name() == "blake3" || algorithm.name().starts_with("blake3-stream4k") {
    blake3_adapter::measure_blake3_parallel_data(runner, algorithm, best_kernel)?
  } else {
    None
  };

  algorithm.reset();

  Ok(RawAlgorithmMeasurements {
    name: algorithm.name().to_string(),
    env_prefix: algorithm.env_prefix().to_string(),
    domain: match algorithm.tuning_domain() {
      TuningDomain::Checksum => "checksum",
      TuningDomain::Hash => "hash",
    }
    .to_string(),
    kernels: available_kernels.iter().map(RawKernelSpec::from_kernel_spec).collect(),
    stream_measurements,
    threshold_measurements,
    size_class_probe_measurements,
    blake3_parallel,
  })
}

/// Deterministically derive one algorithm result from raw measurements.
fn derive_algorithm_from_raw(raw: &RawAlgorithmMeasurements) -> Result<AlgorithmResult, TuneError> {
  let stream_measurements = to_measurements(&raw.stream_measurements);
  let threshold_measurements = to_measurements(&raw.threshold_measurements);

  if stream_measurements.is_empty() {
    return Err(TuneError::BenchmarkFailed("raw stream measurements are empty"));
  }

  let (best_kernel, best_streams) = find_best_config(&stream_measurements);
  let size_class_best = derive_size_class_best(&stream_measurements, &raw.size_class_probe_measurements);

  let portable_names: Vec<&str> = raw
    .kernels
    .iter()
    .filter(|k| matches!(k.tier(), Some(KernelTier::Portable)))
    .map(|k| k.name.as_str())
    .collect();

  let folding_kernel = find_raw_kernel_by_tier(&raw.kernels, KernelTier::Folding);
  let wide_kernel = find_raw_kernel_by_tier(&raw.kernels, KernelTier::Wide);

  let min_bytes_per_lane = if best_streams > 1 {
    analysis::estimate_min_bytes_per_lane(&threshold_measurements, best_kernel, best_streams)
  } else {
    None
  };

  let max_threshold_size = THRESHOLD_SIZES.iter().copied().max().unwrap_or(0);
  let portable_name = if portable_names.is_empty() {
    "portable".to_string()
  } else {
    threshold_measurements
      .iter()
      .filter(|m| m.streams == 1 && portable_names.contains(&m.kernel.as_str()))
      .filter(|m| m.size == max_threshold_size)
      .max_by(|a, b| a.throughput_gib_s.partial_cmp(&b.throughput_gib_s).unwrap())
      .map(|m| m.kernel.clone())
      .unwrap_or_else(|| portable_names[0].to_string())
  };

  let mut result_analysis = analysis::analyze(&threshold_measurements, portable_name.as_str());
  result_analysis.best_large_kernel = Some(best_kernel);
  result_analysis.recommended_streams = best_streams;

  let mut thresholds = Vec::new();

  if crc_adapter::is_crc32_algorithm(raw.name.as_str()) {
    let hw = find_raw_kernel_by_tier(&raw.kernels, KernelTier::Hardware);
    let folding = find_raw_kernel_by_tier(&raw.kernels, KernelTier::Folding);
    let wide: Vec<&RawKernelSpec> = raw
      .kernels
      .iter()
      .filter(|k| matches!(k.tier(), Some(KernelTier::Wide)))
      .collect();

    let portable_to_target = hw.or(folding).or(wide.first().copied());
    if let Some(target) = portable_to_target
      && let Some(crossover) = analysis::find_crossover(
        &threshold_measurements,
        portable_name.as_str(),
        1,
        target.name.as_str(),
        1,
        1.0,
      )
    {
      thresholds.push(("portable_to_hwcrc".to_string(), crossover.crossover_size));
      result_analysis.crossovers.push(crossover);
    }

    if portable_names.contains(&"portable/bytewise")
      && portable_names.contains(&"portable/slice16")
      && let Some(crossover) = analysis::find_crossover(
        &threshold_measurements,
        "portable/bytewise",
        1,
        "portable/slice16",
        1,
        1.0,
      )
    {
      thresholds.push(("portable_bytewise_to_slice16".to_string(), crossover.crossover_size));
      result_analysis.crossovers.push(crossover);
    }

    if let (Some(hw), Some(folding)) = (hw, folding)
      && let Some(crossover) = analysis::find_crossover(
        &threshold_measurements,
        hw.name.as_str(),
        1,
        folding.name.as_str(),
        1,
        1.0,
      )
    {
      thresholds.push(("hwcrc_to_fusion".to_string(), crossover.crossover_size));
      result_analysis.crossovers.push(crossover);
    }

    if raw.name == "crc32-ieee"
      && let (Some(folding), Some(wide)) = (folding, wide.first().copied())
      && let Some(crossover) = analysis::find_crossover(
        &threshold_measurements,
        folding.name.as_str(),
        1,
        wide.name.as_str(),
        1,
        1.0,
      )
    {
      thresholds.push(("fusion_to_vpclmul".to_string(), crossover.crossover_size));
      result_analysis.crossovers.push(crossover);
    }

    if raw.name == "crc32c"
      && let Some(base) = folding
    {
      let avx512 = raw.kernels.iter().find(|k| k.name.starts_with("x86_64/fusion-avx512-"));
      if let Some(avx512) = avx512
        && let Some(crossover) = analysis::find_crossover(
          &threshold_measurements,
          base.name.as_str(),
          1,
          avx512.name.as_str(),
          1,
          1.0,
        )
      {
        thresholds.push(("fusion_to_avx512".to_string(), crossover.crossover_size));
        result_analysis.crossovers.push(crossover);
      }

      let vpclmul = raw
        .kernels
        .iter()
        .find(|k| k.name.starts_with("x86_64/fusion-vpclmul-"));
      if let Some(vpclmul) = vpclmul
        && let Some(crossover) = analysis::find_crossover(
          &threshold_measurements,
          base.name.as_str(),
          1,
          vpclmul.name.as_str(),
          1,
          1.0,
        )
      {
        thresholds.push(("fusion_to_vpclmul".to_string(), crossover.crossover_size));
        result_analysis.crossovers.push(crossover);
      }
    }

    if let Some(min_bpl) = min_bytes_per_lane {
      thresholds.push(("min_bytes_per_lane".to_string(), min_bpl));
    }

    thresholds.sort_by_key(|(_, size)| *size);

    return Ok(AlgorithmResult {
      name: Box::leak(raw.name.clone().into_boxed_str()),
      env_prefix: Box::leak(raw.env_prefix.clone().into_boxed_str()),
      best_kernel,
      recommended_streams: best_streams,
      peak_throughput_gib_s: result_analysis.peak_throughput_gib_s,
      size_class_best,
      thresholds: map_threshold_names(raw.name.as_str(), thresholds),
      analysis: result_analysis,
    });
  }

  if let Some(crossover) = result_analysis
    .crossovers
    .iter()
    .find(|c| c.from_kernel == portable_name)
  {
    thresholds.push((
      CrossoverType::PortableToSimd.threshold_name().to_string(),
      crossover.crossover_size,
    ));
  }

  if matches!(raw.name.as_str(), "crc64-xz" | "crc64-nvme")
    && let Some(small) = detect_crc64_small_kernel(best_kernel)
    && best_kernel != small
    && let Some(crossover) = analysis::find_crossover(&threshold_measurements, small, 1, best_kernel, 1, 1.0)
  {
    thresholds.push(("small_kernel_max_bytes".to_string(), crossover.crossover_size));
    result_analysis.crossovers.push(crossover);
  }

  if portable_names.contains(&"portable/slice4")
    && portable_names.contains(&"portable/slice8")
    && let Some(crossover) =
      analysis::find_crossover(&threshold_measurements, "portable/slice4", 1, "portable/slice8", 1, 1.0)
  {
    thresholds.push(("slice4_to_slice8".to_string(), crossover.crossover_size));
    result_analysis.crossovers.push(crossover);
  }

  if let (Some(folding), Some(wide)) = (folding_kernel, wide_kernel) {
    let folding_streams = analysis::select_best_streams(&stream_measurements, folding.name.as_str());
    let wide_streams = analysis::select_best_streams(&stream_measurements, wide.name.as_str());

    if let Some(typed_crossover) = analysis::find_tier_crossover(
      &threshold_measurements,
      folding.name.as_str(),
      folding_streams,
      wide.name.as_str(),
      wide_streams,
      CrossoverType::SimdToWide,
      1.0,
    ) {
      thresholds.push((
        CrossoverType::SimdToWide.threshold_name().to_string(),
        typed_crossover.crossover.crossover_size,
      ));
      result_analysis.crossovers.push(typed_crossover.crossover);
    }
  }

  if let Some(min_bpl) = min_bytes_per_lane {
    thresholds.push(("min_bytes_per_lane".to_string(), min_bpl));
  }

  if (raw.name == "blake3" || raw.name.starts_with("blake3-stream4k"))
    && let Some(parallel_data) = raw.blake3_parallel.as_ref()
    && let Some(policy) = blake3_adapter::derive_blake3_parallel_policy(parallel_data)
  {
    thresholds.push(("parallel_min_bytes".to_string(), policy.min_bytes));
    thresholds.push(("parallel_min_chunks".to_string(), policy.min_chunks));
    thresholds.push(("parallel_max_threads".to_string(), policy.max_threads));
    thresholds.push(("parallel_spawn_cost_bytes".to_string(), policy.spawn_cost_bytes));
    thresholds.push(("parallel_merge_cost_bytes".to_string(), policy.merge_cost_bytes));
    thresholds.push(("parallel_bytes_per_core_small".to_string(), policy.bytes_per_core_small));
    thresholds.push((
      "parallel_bytes_per_core_medium".to_string(),
      policy.bytes_per_core_medium,
    ));
    thresholds.push(("parallel_bytes_per_core_large".to_string(), policy.bytes_per_core_large));
    thresholds.push(("parallel_small_limit_bytes".to_string(), policy.small_limit_bytes));
    thresholds.push(("parallel_medium_limit_bytes".to_string(), policy.medium_limit_bytes));
  }

  thresholds.sort_by_key(|(_, size)| *size);

  Ok(AlgorithmResult {
    name: Box::leak(raw.name.clone().into_boxed_str()),
    env_prefix: Box::leak(raw.env_prefix.clone().into_boxed_str()),
    best_kernel,
    recommended_streams: best_streams,
    peak_throughput_gib_s: result_analysis.peak_throughput_gib_s,
    size_class_best,
    thresholds: map_threshold_names(raw.name.as_str(), thresholds),
    analysis: result_analysis,
  })
}

/// Map generic threshold names to algorithm-specific env var suffixes.
fn map_threshold_names(algo_name: &str, thresholds: Vec<(String, usize)>) -> Vec<(String, usize)> {
  thresholds
    .into_iter()
    .map(|(name, value)| {
      let env_suffix = threshold_to_env_suffix_for_algo(algo_name, &name)
        .map(String::from)
        .unwrap_or_else(|| format!("THRESHOLD_{}", name.to_uppercase()));
      (env_suffix, value)
    })
    .collect()
}

#[must_use]
fn threshold_to_env_suffix_for_algo(algo_name: &str, threshold_name: &str) -> Option<&'static str> {
  match algo_name {
    "crc16-ccitt" | "crc16-ibm" => crc16_threshold_to_env_suffix(threshold_name),
    "crc24-openpgp" => crc24_threshold_to_env_suffix(threshold_name),
    "crc32-ieee" | "crc32c" => crc32_threshold_to_env_suffix(threshold_name),
    "crc64-xz" | "crc64-nvme" => crc64_threshold_to_env_suffix(threshold_name),
    _ => hash_threshold_to_env_suffix(algo_name, threshold_name),
  }
}

fn raw_platform_to_platform_info(raw: &RawPlatformInfo) -> Result<PlatformInfo, TuneError> {
  let tune_kind = tune_kind_from_u8(raw.tune_kind).ok_or_else(|| {
    TuneError::Io(format!(
      "invalid raw artifact: unknown tune kind discriminant {}",
      raw.tune_kind
    ))
  })?;

  Ok(PlatformInfo {
    arch: Box::leak(raw.arch.clone().into_boxed_str()),
    os: Box::leak(raw.os.clone().into_boxed_str()),
    caps: platform::Caps::NONE,
    tune_kind,
    description: raw.description.clone(),
  })
}

#[must_use]
fn tune_kind_from_u8(value: u8) -> Option<platform::TuneKind> {
  use platform::TuneKind;
  Some(match value {
    0 => TuneKind::Custom,
    1 => TuneKind::Default,
    2 => TuneKind::Portable,
    3 => TuneKind::Zen4,
    4 => TuneKind::Zen5,
    5 => TuneKind::Zen5c,
    6 => TuneKind::IntelSpr,
    7 => TuneKind::IntelGnr,
    8 => TuneKind::IntelIcl,
    9 => TuneKind::AppleM1M3,
    10 => TuneKind::AppleM4,
    11 => TuneKind::AppleM5,
    12 => TuneKind::Graviton2,
    13 => TuneKind::Graviton3,
    14 => TuneKind::Graviton4,
    15 => TuneKind::Graviton5,
    16 => TuneKind::NeoverseN2,
    17 => TuneKind::NeoverseN3,
    18 => TuneKind::NeoverseV3,
    19 => TuneKind::NvidiaGrace,
    20 => TuneKind::AmpereAltra,
    21 => TuneKind::Aarch64Pmull,
    22 => TuneKind::Z13,
    23 => TuneKind::Z14,
    24 => TuneKind::Z15,
    25 => TuneKind::Power7,
    26 => TuneKind::Power8,
    27 => TuneKind::Power9,
    28 => TuneKind::Power10,
    _ => return None,
  })
}

/// Find the first kernel of a specific tier.
fn find_kernel_by_tier(kernels: &[KernelSpec], tier: KernelTier) -> Option<&KernelSpec> {
  kernels.iter().find(|k| k.tier == tier)
}

fn find_raw_kernel_by_tier(kernels: &[RawKernelSpec], tier: KernelTier) -> Option<&RawKernelSpec> {
  kernels.iter().find(|k| matches!(k.tier(), Some(t) if t == tier))
}

#[must_use]
fn detect_crc64_small_kernel(best_kernel: &str) -> Option<&'static str> {
  if best_kernel.starts_with("x86_64/") {
    Some("x86_64/pclmul-small")
  } else if best_kernel.starts_with("aarch64/sve2-pmull") {
    Some("aarch64/sve2-pmull-small")
  } else if best_kernel.starts_with("aarch64/") {
    Some("aarch64/pmull-small")
  } else {
    None
  }
}

#[must_use]
fn has_measurement(measurements: &[RawBenchPoint], kernel: &str, streams: u8) -> bool {
  measurements.iter().any(|m| m.kernel == kernel && m.streams == streams)
}

#[must_use]
fn to_measurements(points: &[RawBenchPoint]) -> Vec<Measurement> {
  points.iter().map(RawBenchPoint::to_measurement).collect()
}

/// Benchmark different stream counts for each kernel.
fn benchmark_streams(
  runner: &BenchRunner,
  algorithm: &mut dyn Tunable,
  kernels: &[crate::KernelSpec],
) -> Result<Vec<RawBenchPoint>, TuneError> {
  let mut measurements = Vec::new();
  let stream_candidates = stream_candidates();

  for kernel in kernels {
    // Skip reference kernel for stream benchmarks
    if kernel.tier == crate::KernelTier::Reference {
      continue;
    }

    let streams_to_test = match kernel.streams {
      Some((min, max)) => stream_candidates
        .iter()
        .filter(|&&s| s >= min && s <= max)
        .copied()
        .collect::<Vec<_>>(),
      None => vec![1],
    };

    for &streams in &streams_to_test {
      algorithm.reset();
      if let Err(err) = algorithm.force_kernel(kernel.name) {
        // Don't fail the whole tuning run if a kernel spec is stale or the
        // bench registry doesn't expose it for this build/target.
        eprintln!(
          "warning: skipping {} kernel {} (streams={streams}): {err}",
          algorithm.name(),
          kernel.name
        );
        continue;
      }
      if streams > 1
        && let Err(err) = algorithm.force_streams(streams)
      {
        eprintln!(
          "warning: skipping {} kernel {} (streams={streams}): {err}",
          algorithm.name(),
          kernel.name
        );
        continue;
      }

      for &size in STREAM_SIZES_BENCH {
        let result = benchmark_at_size(runner, algorithm, size)?;
        measurements.push(RawBenchPoint::from_result(kernel.name, streams, &result));
      }
    }
  }

  algorithm.reset();
  Ok(measurements)
}

/// Benchmark threshold curve (portable vs SIMD across sizes).
fn benchmark_thresholds(
  runner: &BenchRunner,
  algorithm: &mut dyn Tunable,
  portable_kernels: &[&'static str],
  best_kernel: &str,
  best_streams: u8,
) -> Result<Vec<RawBenchPoint>, TuneError> {
  let mut measurements = Vec::new();

  // Benchmark portable kernels
  for &portable in portable_kernels {
    algorithm.reset();
    if algorithm.force_kernel(portable).is_err() {
      continue;
    }

    for &size in THRESHOLD_SIZES {
      if let Ok(result) = benchmark_at_size(runner, algorithm, size) {
        measurements.push(RawBenchPoint::from_result(portable, 1, &result));
      }
    }
  }

  // Fallback if the algorithm doesn't expose portable kernel names.
  if portable_kernels.is_empty() {
    algorithm.reset();
    algorithm.force_kernel("portable").ok(); // May not exist, that's ok

    for &size in THRESHOLD_SIZES {
      if let Ok(result) = benchmark_at_size(runner, algorithm, size) {
        measurements.push(RawBenchPoint::from_result("portable", 1, &result));
      }
    }
  }

  // Benchmark best SIMD kernel
  algorithm.reset();
  algorithm.force_kernel(best_kernel)?;
  if best_streams > 1 {
    algorithm.force_streams(best_streams)?;
  }

  for &size in THRESHOLD_SIZES {
    let result = benchmark_at_size(runner, algorithm, size)?;
    measurements.push(RawBenchPoint::from_result(best_kernel, best_streams, &result));
  }

  algorithm.reset();
  Ok(measurements)
}

/// Benchmark at a single buffer size.
fn benchmark_at_size(
  runner: &BenchRunner,
  algorithm: &dyn Tunable,
  size: usize,
) -> Result<crate::BenchResult, TuneError> {
  let mut buffer = vec![0u8; size];
  fill_data(&mut buffer);
  runner.measure_single(algorithm, &buffer)
}

/// Benchmark a single kernel across all threshold sizes.
///
/// Used for additional tier-based crossover detection (e.g., folding vs wide)
/// and single-stream vs multi-stream comparison.
fn benchmark_single_kernel(
  runner: &BenchRunner,
  algorithm: &mut dyn Tunable,
  kernel: &str,
  streams: u8,
) -> Result<Vec<RawBenchPoint>, TuneError> {
  let mut measurements = Vec::new();

  algorithm.reset();
  if let Err(err) = algorithm.force_kernel(kernel) {
    eprintln!(
      "warning: skipping {} kernel {} (streams={streams}): {err}",
      algorithm.name(),
      kernel
    );
    algorithm.reset();
    return Ok(measurements);
  }
  if streams > 1
    && let Err(err) = algorithm.force_streams(streams)
  {
    eprintln!(
      "warning: skipping {} kernel {} (streams={streams}): {err}",
      algorithm.name(),
      kernel
    );
    algorithm.reset();
    return Ok(measurements);
  }

  for &size in THRESHOLD_SIZES {
    if let Ok(result) = benchmark_at_size(runner, algorithm, size) {
      measurements.push(RawBenchPoint::from_result(kernel, streams, &result));
    }
  }

  algorithm.reset();
  Ok(measurements)
}

fn benchmark_size_class_probes(
  runner: &BenchRunner,
  algorithm: &mut dyn Tunable,
  kernels: &[KernelSpec],
  stream_measurements: &[Measurement],
) -> Result<Vec<RawBenchPoint>, TuneError> {
  let mut probes = Vec::new();

  for (_class, &size) in SIZE_CLASS_NAMES.iter().copied().zip(SIZE_CLASS_SIZES.iter()) {
    // m/l are covered by stream measurements. Keep only explicit probe points.
    if stream_measurements.iter().any(|m| m.size == size) {
      continue;
    }

    // Otherwise (xs/s), benchmark each kernel at the pivot size, including
    // stream variants when available. This is important: many CRC kernels have
    // different optimal stream counts for `s` vs `m/l`.
    for kernel in kernels {
      if kernel.tier == crate::KernelTier::Reference {
        continue;
      }

      let streams_to_test = match kernel.streams {
        Some((min, max)) => stream_candidates()
          .iter()
          .filter(|&&s| s >= min && s <= max)
          .copied()
          .collect::<Vec<_>>(),
        None => vec![1],
      };

      for &streams in &streams_to_test {
        algorithm.reset();
        if algorithm.force_kernel(kernel.name).is_err() {
          continue;
        }
        if streams > 1 && algorithm.force_streams(streams).is_err() {
          continue;
        }

        let result = benchmark_at_size(runner, algorithm, size)?;
        probes.push(RawBenchPoint::from_result(kernel.name, streams, &result));
      }
    }

    algorithm.reset();
  }

  Ok(probes)
}

fn derive_size_class_best(stream_measurements: &[Measurement], probes: &[RawBenchPoint]) -> Vec<SizeClassBest> {
  fn winner_from_measurements(measurements: &[Measurement], size: usize) -> Option<(String, u8, f64)> {
    measurements
      .iter()
      .filter(|m| m.size == size)
      .max_by(|a, b| a.throughput_gib_s.partial_cmp(&b.throughput_gib_s).unwrap())
      .map(|m| (m.kernel.clone(), m.streams, m.throughput_gib_s))
  }

  let probe_measurements = to_measurements(probes);
  let mut best = Vec::with_capacity(SIZE_CLASS_SIZES.len());

  for (class, &size) in SIZE_CLASS_NAMES.iter().copied().zip(SIZE_CLASS_SIZES.iter()) {
    if let Some((kernel, streams, throughput_gib_s)) = winner_from_measurements(stream_measurements, size) {
      best.push(SizeClassBest {
        class,
        kernel,
        streams,
        throughput_gib_s,
      });
      continue;
    }

    if let Some((kernel, streams, throughput_gib_s)) = winner_from_measurements(&probe_measurements, size) {
      best.push(SizeClassBest {
        class,
        kernel,
        streams,
        throughput_gib_s,
      });
      continue;
    }

    best.push(SizeClassBest {
      class,
      kernel: "portable".to_string(),
      streams: 1,
      throughput_gib_s: 0.0,
    });
  }

  best
}

/// Find the best kernel/stream configuration from measurements.
///
/// Delegates to `analysis::find_best_large_config` and converts the result
/// to static strings for use in the engine.
fn find_best_config(measurements: &[Measurement]) -> (&'static str, u8) {
  match analysis::find_best_config_across_sizes(measurements, STREAM_SIZES) {
    Some(config) => {
      // Convert borrowed str to 'static by leaking (one-time operation during tuning)
      let name: &'static str = Box::leak(config.kernel.to_string().into_boxed_str());
      (name, config.streams)
    }
    None => ("portable", 1),
  }
}

/// Generate an ISO 8601 timestamp for the current time.
///
/// Uses a simplified civil calendar calculation. For precise timestamps,
/// consider using the `time` or `chrono` crate.
fn get_timestamp() -> String {
  use std::time::{SystemTime, UNIX_EPOCH};

  let secs = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs();

  // Days and time-of-day
  let days_since_epoch = secs / 86400;
  let time_of_day = secs % 86400;
  let hours = time_of_day / 3600;
  let mins = (time_of_day % 3600) / 60;
  let seconds = time_of_day % 60;

  // Civil calendar calculation (handles leap years correctly)
  let (year, month, day) = days_to_civil(days_since_epoch);

  format!("{year:04}-{month:02}-{day:02}T{hours:02}:{mins:02}:{seconds:02}Z")
}

/// Convert days since Unix epoch to (year, month, day).
///
/// Algorithm from Howard Hinnant's date library.
fn days_to_civil(days: u64) -> (u32, u32, u32) {
  // Shift epoch from 1970-01-01 to 0000-03-01
  let z = days.strict_add(719468);
  let era = z / 146097;
  let doe = z.strict_sub(era.strict_mul(146097)); // day of era [0, 146096]
  let yoe = doe
    .strict_sub(doe / 1460)
    .strict_add(doe / 36524)
    .strict_sub(doe / 146096)
    / 365; // year of era [0, 399]
  let y = yoe.strict_add(era.strict_mul(400));
  let doy = doe
    .strict_sub(365u64.strict_mul(yoe))
    .strict_sub(yoe / 4)
    .strict_add(yoe / 100); // day of year [0, 365]
  let mp = (5u64.strict_mul(doy).strict_add(2)) / 153; // month in [0, 11]
  let d = doy.strict_sub((153u64.strict_mul(mp).strict_add(2)) / 5).strict_add(1); // day [1, 31]

  let m = if mp < 10 { mp.strict_add(3) } else { mp.strict_sub(9) };
  let y = if m <= 2 { y.strict_add(1) } else { y };

  (y as u32, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
  use platform::Caps;

  use super::*;
  use crate::{BenchResult, SamplerConfig, TunableParam};

  struct BrokenKernelTunable {
    current: &'static str,
  }

  impl BrokenKernelTunable {
    fn new() -> Self {
      Self { current: "auto" }
    }
  }

  impl Tunable for BrokenKernelTunable {
    fn name(&self) -> &'static str {
      "broken-kernel"
    }

    fn available_kernels(&self, _caps: &Caps) -> Vec<KernelSpec> {
      vec![
        KernelSpec::new("portable", KernelTier::Portable, Caps::NONE),
        KernelSpec::new("broken", KernelTier::Wide, Caps::NONE),
      ]
    }

    fn force_kernel(&mut self, name: &str) -> Result<(), TuneError> {
      match name {
        "portable" => {
          self.current = "portable";
          Ok(())
        }
        "broken" => Err(TuneError::KernelNotAvailable(
          "kernel name did not resolve to a bench kernel",
        )),
        _ => Err(TuneError::KernelNotAvailable("kernel not available on this platform")),
      }
    }

    fn force_streams(&mut self, count: u8) -> Result<(), TuneError> {
      if count == 1 {
        Ok(())
      } else {
        Err(TuneError::InvalidStreamCount(count))
      }
    }

    fn reset(&mut self) {
      self.current = "auto";
    }

    fn benchmark(&self, data: &[u8], _config: &SamplerConfig) -> BenchResult {
      BenchResult {
        kernel: self.current,
        buffer_size: data.len(),
        iterations: 1,
        bytes_processed: data.len() as u64,
        throughput_gib_s: 1.0,
        elapsed_secs: 0.001,
        sample_count: None,
        std_dev: None,
        cv: None,
        outliers_rejected: None,
        min_throughput_gib_s: None,
        max_throughput_gib_s: None,
      }
    }

    fn current_kernel(&self) -> &'static str {
      self.current
    }

    fn tunable_params(&self) -> &[TunableParam] {
      &[]
    }

    fn env_prefix(&self) -> &'static str {
      "RSCRYPTO_BROKEN_KERNEL"
    }

    fn tuning_domain(&self) -> TuningDomain {
      TuningDomain::Checksum
    }
  }

  #[test]
  fn tuning_skips_kernels_that_dont_resolve() {
    let mut engine = TuneEngine::new().with_runner(BenchRunner::quick()).with_verbose(true);
    engine.add(Box::new(BrokenKernelTunable::new()));

    let results = engine.run().expect("tuning should not fail");
    assert_eq!(results.algorithms.len(), 1);
    assert_eq!(results.algorithms[0].name, "broken-kernel");
    assert_eq!(results.algorithms[0].best_kernel, "portable");
  }
}
