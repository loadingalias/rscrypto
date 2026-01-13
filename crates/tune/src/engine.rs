//! Tuning engine orchestrator.

use crate::{
  AlgorithmResult, KernelSpec, KernelTier, PlatformInfo, Tunable, TuneError, TuneResults,
  analysis::{self, CrossoverType, Measurement},
  runner::{
    BenchRunner, SIZE_CLASS_NAMES, SIZE_CLASS_SIZES, STREAM_SIZES, THRESHOLD_SIZES, fill_data, stream_candidates,
  },
};

fn is_crc32_algorithm(name: &str) -> bool {
  matches!(name, "crc32-ieee" | "crc32c")
}

/// Main tuning engine that orchestrates benchmarking for multiple algorithms.
pub struct TuneEngine {
  /// Algorithms to tune.
  algorithms: Vec<Box<dyn Tunable>>,

  /// Benchmark runner configuration.
  runner: BenchRunner,

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
      runner: BenchRunner::default(),
      verbose: false,
    }
  }

  /// Create a tuning engine with quick mode settings.
  #[must_use]
  pub fn quick() -> Self {
    Self {
      algorithms: Vec::new(),
      runner: BenchRunner::quick(),
      verbose: false,
    }
  }

  /// Set the benchmark runner.
  #[must_use]
  pub fn with_runner(mut self, runner: BenchRunner) -> Self {
    self.runner = runner;
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

  /// Run the complete tuning suite.
  pub fn run(&mut self) -> Result<TuneResults, TuneError> {
    let platform = PlatformInfo::collect();

    if self.verbose {
      eprintln!("Platform: {}", platform.description);
      eprintln!("Tune preset: {:?}", platform.tune_kind);
      eprintln!();
    }

    let mut algorithm_results = Vec::with_capacity(self.algorithms.len());
    let verbose = self.verbose;

    for i in 0..self.algorithms.len() {
      let algorithm = &mut self.algorithms[i];

      if verbose {
        eprintln!("Tuning {}...", algorithm.name());
      }

      let result = tune_algorithm_impl(&self.runner, algorithm.as_mut(), &platform)?;

      if verbose {
        eprintln!("  Best kernel: {}", result.best_kernel);
        eprintln!("  Peak throughput: {:.2} GiB/s", result.peak_throughput_gib_s);
        eprintln!();
      }

      algorithm_results.push(result);
    }

    let timestamp = get_timestamp();

    Ok(TuneResults {
      platform,
      algorithms: algorithm_results,
      timestamp,
    })
  }
}

/// Tune a single algorithm (standalone function to avoid borrow issues).
fn tune_algorithm_impl(
  runner: &BenchRunner,
  algorithm: &mut dyn Tunable,
  platform: &PlatformInfo,
) -> Result<AlgorithmResult, TuneError> {
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
  let portable_kernel = portable_kernels.first().copied();
  let folding_kernel = find_kernel_by_tier(&available_kernels, KernelTier::Folding);
  let wide_kernel = find_kernel_by_tier(&available_kernels, KernelTier::Wide);

  // Phase A: Find best stream count for each kernel at large buffer sizes
  let stream_measurements = benchmark_streams(runner, algorithm, &available_kernels)?;

  // Find best kernel/stream combination at large sizes
  let (best_kernel, best_streams) = find_best_config(&stream_measurements);

  // Phase A2: Pick best kernel per size class (xs/s/m/l).
  let size_class_best = benchmark_size_class_best(runner, algorithm, &available_kernels, &stream_measurements)?;

  // Phase B: Threshold curves - portable vs best SIMD
  let portable_names: Vec<&'static str> = portable_kernels.iter().map(|k| k.name).collect();
  let portable_names_owned: Vec<String> = portable_names.iter().map(|&n| n.to_string()).collect();
  let mut threshold_measurements = benchmark_thresholds(runner, algorithm, &portable_names, best_kernel, best_streams)?;

  // CRC-64: benchmark the small-buffer SIMD kernel explicitly so:
  // - portable→SIMD crossovers reflect the real small-kernel behavior
  // - we can tune the small-kernel window (small→best crossover)
  let mut crc64_small_kernel: Option<&'static str> = None;
  if matches!(algorithm.name(), "crc64-xz" | "crc64-nvme") {
    // Ensure we have a single-stream curve for the best kernel (small kernel is single-stream).
    if best_streams != 1
      && !threshold_measurements
        .iter()
        .any(|m| m.kernel == best_kernel && m.streams == 1)
    {
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
      && !threshold_measurements
        .iter()
        .any(|m| m.kernel == name && m.streams == 1)
      && let Ok(curve) = benchmark_single_kernel(runner, algorithm, name, 1)
      && !curve.is_empty()
    {
      crc64_small_kernel = Some(name);
      threshold_measurements.extend(curve);
    }
  }

  // Phase C: Additional threshold curves for tier-based crossovers
  // If we have both folding and wide kernels, benchmark the folding kernel
  // to detect SIMD→wide crossover
  if let (Some(folding), Some(_wide)) = (&folding_kernel, &wide_kernel) {
    // Only benchmark folding if it's not the same as best_kernel
    if folding.name != best_kernel {
      let folding_streams = analysis::select_best_streams(&stream_measurements, folding.name);
      let folding_measurements = benchmark_single_kernel(runner, algorithm, folding.name, folding_streams)?;
      threshold_measurements.extend(folding_measurements);
    }
  }

  // CRC-32: benchmark all non-portable tiers across THRESHOLD_SIZES so we can
  // compute policy-specific crossovers (portable→hwcrc, hwcrc→fusion, fusion→wide).
  if is_crc32_algorithm(algorithm.name()) {
    for kernel in &available_kernels {
      if matches!(kernel.tier, KernelTier::Reference | KernelTier::Portable) {
        continue;
      }

      // CRC-32 crossovers are tier boundaries, not stream-selection boundaries.
      // Benchmark all tiers at single-stream so crossover points match the policy:
      // multi-stream enters via `min_bytes_per_lane`, not via the tier thresholds.
      let streams = 1;
      if threshold_measurements
        .iter()
        .any(|m| m.kernel == kernel.name && m.streams == streams)
      {
        continue;
      }

      let curve = benchmark_single_kernel(runner, algorithm, kernel.name, streams)?;
      threshold_measurements.extend(curve);
    }
  }

  // Phase D: Single-stream vs multi-stream comparison for min_bytes_per_lane
  // Only if best_streams > 1
  let min_bytes_per_lane = if best_streams > 1 {
    let single_stream_measurements = benchmark_single_kernel(runner, algorithm, best_kernel, 1)?;
    threshold_measurements.extend(single_stream_measurements.clone());

    // Estimate min_bytes_per_lane
    analysis::estimate_min_bytes_per_lane(&threshold_measurements, best_kernel, best_streams)
  } else {
    None
  };

  // Analyze results
  let max_threshold_size = THRESHOLD_SIZES.iter().copied().max().unwrap_or(0);
  let portable_name = if portable_names.is_empty() {
    portable_kernel.map(|k| k.name).unwrap_or("portable")
  } else {
    threshold_measurements
      .iter()
      .filter(|m| m.streams == 1 && portable_names_owned.contains(&m.kernel))
      .filter(|m| m.size == max_threshold_size)
      .max_by(|a, b| a.throughput_gib_s.partial_cmp(&b.throughput_gib_s).unwrap())
      .map(|m| Box::leak(m.kernel.clone().into_boxed_str()) as &'static str)
      .unwrap_or_else(|| portable_kernel.map(|k| k.name).unwrap_or("portable"))
  };
  let mut result_analysis = analysis::analyze(&threshold_measurements, portable_name);
  result_analysis.best_large_kernel = Some(best_kernel);
  result_analysis.recommended_streams = best_streams;

  // Build thresholds
  let mut thresholds = Vec::new();

  if is_crc32_algorithm(algorithm.name()) {
    let hw = find_kernel_by_tier(&available_kernels, KernelTier::Hardware);
    let folding = find_kernel_by_tier(&available_kernels, KernelTier::Folding);
    let wide: Vec<&KernelSpec> = available_kernels
      .iter()
      .filter(|k| k.tier == KernelTier::Wide)
      .collect();

    // portable_to_hwcrc: portable → first accelerated tier (HWCRC if present, else Folding).
    let portable_to_target = hw.or(folding).or(wide.first().copied());
    if let Some(target) = portable_to_target
      && let Some(crossover) = analysis::find_crossover(&threshold_measurements, portable_name, 1, target.name, 1, 1.0)
    {
      thresholds.push(("portable_to_hwcrc".to_string(), crossover.crossover_size));
      result_analysis.crossovers.push(crossover);
    }

    // portable_bytewise_to_slice16: within the portable tier, select bytewise for tiny buffers.
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

    // hwcrc_to_fusion: HWCRC → baseline folding kernel.
    //
    // This matches the `checksum::crc32` policy semantics: hwcrc_to_fusion gates
    // entry into the folding/fusion family; further x86-only thresholds handle
    // selecting wider fusion kernels.
    if let (Some(hw), Some(folding)) = (hw, folding)
      && let Some(crossover) = analysis::find_crossover(&threshold_measurements, hw.name, 1, folding.name, 1, 1.0)
    {
      thresholds.push(("hwcrc_to_fusion".to_string(), crossover.crossover_size));
      result_analysis.crossovers.push(crossover);
    }

    // x86_64: fusion→wide thresholds for the policy's x86-only tiers.
    #[cfg(target_arch = "x86_64")]
    {
      if algorithm.name() == "crc32-ieee"
        && let (Some(folding), Some(wide)) = (folding, wide.first().copied())
        && let Some(crossover) = analysis::find_crossover(&threshold_measurements, folding.name, 1, wide.name, 1, 1.0)
      {
        thresholds.push(("fusion_to_vpclmul".to_string(), crossover.crossover_size));
        result_analysis.crossovers.push(crossover);
      }

      if algorithm.name() == "crc32c"
        && let Some(base) = folding
      {
        let avx512 = available_kernels
          .iter()
          .find(|k| k.name.starts_with("x86_64/fusion-avx512-"));
        if let Some(avx512) = avx512
          && let Some(crossover) = analysis::find_crossover(&threshold_measurements, base.name, 1, avx512.name, 1, 1.0)
        {
          thresholds.push(("fusion_to_avx512".to_string(), crossover.crossover_size));
          result_analysis.crossovers.push(crossover);
        }

        let vpclmul = available_kernels
          .iter()
          .find(|k| k.name.starts_with("x86_64/fusion-vpclmul-"));
        if let Some(vpclmul) = vpclmul
          && let Some(crossover) = analysis::find_crossover(&threshold_measurements, base.name, 1, vpclmul.name, 1, 1.0)
        {
          thresholds.push(("fusion_to_vpclmul".to_string(), crossover.crossover_size));
          result_analysis.crossovers.push(crossover);
        }
      }
    }

    // Min bytes per lane threshold
    if let Some(min_bpl) = min_bytes_per_lane {
      thresholds.push(("min_bytes_per_lane".to_string(), min_bpl));
    }

    // Sort thresholds by size for consistent output
    thresholds.sort_by_key(|(_, size)| *size);

    let mapped_thresholds = map_threshold_names(algorithm, thresholds);
    return Ok(AlgorithmResult {
      name: algorithm.name(),
      env_prefix: algorithm.env_prefix(),
      best_kernel,
      recommended_streams: best_streams,
      peak_throughput_gib_s: result_analysis.peak_throughput_gib_s,
      size_class_best,
      thresholds: mapped_thresholds,
      analysis: result_analysis,
    });
  }

  // 1. Portable → SIMD crossover
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

  // CRC-64: small SIMD kernel window (small→best crossover).
  if matches!(algorithm.name(), "crc64-xz" | "crc64-nvme")
    && let Some(small) = crc64_small_kernel
    && best_kernel != small
    && let Some(crossover) = analysis::find_crossover(&threshold_measurements, small, 1, best_kernel, 1, 1.0)
  {
    thresholds.push(("small_kernel_max_bytes".to_string(), crossover.crossover_size));
    result_analysis.crossovers.push(crossover);
  }

  // 1b. Portable slice4 → slice8 crossover (if present).
  if portable_names.contains(&"portable/slice4")
    && portable_names.contains(&"portable/slice8")
    && let Some(crossover) =
      analysis::find_crossover(&threshold_measurements, "portable/slice4", 1, "portable/slice8", 1, 1.0)
  {
    thresholds.push(("slice4_to_slice8".to_string(), crossover.crossover_size));
    result_analysis.crossovers.push(crossover);
  }

  // 2. SIMD → Wide crossover (if applicable)
  if let (Some(folding), Some(wide)) = (&folding_kernel, &wide_kernel) {
    let folding_streams = analysis::select_best_streams(&stream_measurements, folding.name);
    let wide_streams = analysis::select_best_streams(&stream_measurements, wide.name);

    if let Some(typed_crossover) = analysis::find_tier_crossover(
      &threshold_measurements,
      folding.name,
      folding_streams,
      wide.name,
      wide_streams,
      CrossoverType::SimdToWide,
      1.0, // No margin required
    ) {
      thresholds.push((
        CrossoverType::SimdToWide.threshold_name().to_string(),
        typed_crossover.crossover.crossover_size,
      ));
      result_analysis.crossovers.push(typed_crossover.crossover);
    }
  }

  // 3. Min bytes per lane threshold
  if let Some(min_bpl) = min_bytes_per_lane {
    thresholds.push(("min_bytes_per_lane".to_string(), min_bpl));
  }

  // Sort thresholds by size for consistent output
  thresholds.sort_by_key(|(_, size)| *size);

  // Map threshold names to env var suffixes
  let mapped_thresholds = map_threshold_names(algorithm, thresholds);

  Ok(AlgorithmResult {
    name: algorithm.name(),
    env_prefix: algorithm.env_prefix(),
    best_kernel,
    recommended_streams: best_streams,
    peak_throughput_gib_s: result_analysis.peak_throughput_gib_s,
    size_class_best,
    thresholds: mapped_thresholds,
    analysis: result_analysis,
  })
}

/// Map generic threshold names to algorithm-specific env var suffixes.
fn map_threshold_names(algorithm: &dyn Tunable, thresholds: Vec<(String, usize)>) -> Vec<(String, usize)> {
  thresholds
    .into_iter()
    .map(|(name, value)| {
      let env_suffix = algorithm
        .threshold_to_env_suffix(&name)
        .map(String::from)
        .unwrap_or_else(|| {
          // Default: uppercase with THRESHOLD_ prefix
          format!("THRESHOLD_{}", name.to_uppercase())
        });
      (env_suffix, value)
    })
    .collect()
}

/// Find the first kernel of a specific tier.
fn find_kernel_by_tier(kernels: &[KernelSpec], tier: KernelTier) -> Option<&KernelSpec> {
  kernels.iter().find(|k| k.tier == tier)
}

/// Benchmark different stream counts for each kernel.
fn benchmark_streams(
  runner: &BenchRunner,
  algorithm: &mut dyn Tunable,
  kernels: &[crate::KernelSpec],
) -> Result<Vec<Measurement>, TuneError> {
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
      algorithm.force_kernel(kernel.name)?;
      if streams > 1 {
        algorithm.force_streams(streams)?;
      }

      for &size in STREAM_SIZES {
        let result = benchmark_at_size(runner, algorithm, size)?;
        measurements.push(Measurement::with_streams(&result, streams));
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
) -> Result<Vec<Measurement>, TuneError> {
  let mut measurements = Vec::new();

  // Benchmark portable kernels
  for &portable in portable_kernels {
    algorithm.reset();
    if algorithm.force_kernel(portable).is_err() {
      continue;
    }

    for &size in THRESHOLD_SIZES {
      if let Ok(result) = benchmark_at_size(runner, algorithm, size) {
        measurements.push(Measurement::with_streams(&result, 1));
      }
    }
  }

  // Fallback if the algorithm doesn't expose portable kernel names.
  if portable_kernels.is_empty() {
    algorithm.reset();
    algorithm.force_kernel("portable").ok(); // May not exist, that's ok

    for &size in THRESHOLD_SIZES {
      if let Ok(result) = benchmark_at_size(runner, algorithm, size) {
        measurements.push(Measurement::with_streams(&result, 1));
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
    measurements.push(Measurement::with_streams(&result, best_streams));
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
) -> Result<Vec<Measurement>, TuneError> {
  let mut measurements = Vec::new();

  algorithm.reset();
  algorithm.force_kernel(kernel)?;
  if streams > 1 {
    algorithm.force_streams(streams)?;
  }

  for &size in THRESHOLD_SIZES {
    if let Ok(result) = benchmark_at_size(runner, algorithm, size) {
      measurements.push(Measurement::with_streams(&result, streams));
    }
  }

  algorithm.reset();
  Ok(measurements)
}

fn benchmark_size_class_best(
  runner: &BenchRunner,
  algorithm: &mut dyn Tunable,
  kernels: &[KernelSpec],
  stream_measurements: &[Measurement],
) -> Result<Vec<crate::SizeClassBest>, TuneError> {
  struct Point {
    size: usize,
    kernel: String,
    streams: u8,
    throughput_gib_s: f64,
  }

  let mut points: Vec<Point> = Vec::new();

  for kernel in kernels {
    // Determine best stream count for this kernel from the stream-selection phase.
    let streams = match kernel.streams {
      Some(_) => analysis::select_best_streams(stream_measurements, kernel.name),
      None => 1,
    };

    algorithm.reset();
    if algorithm.force_kernel(kernel.name).is_err() {
      continue;
    }
    if streams > 1 && algorithm.force_streams(streams).is_err() {
      continue;
    }

    for &size in SIZE_CLASS_SIZES.iter() {
      let result = benchmark_at_size(runner, algorithm, size)?;
      points.push(Point {
        size,
        kernel: kernel.name.to_string(),
        streams,
        throughput_gib_s: result.throughput_gib_s,
      });
    }
  }

  algorithm.reset();

  let mut best = Vec::with_capacity(SIZE_CLASS_SIZES.len());
  for (class, &size) in SIZE_CLASS_NAMES.iter().copied().zip(SIZE_CLASS_SIZES.iter()) {
    let Some(winner) = points
      .iter()
      .filter(|p| p.size == size)
      .max_by(|a, b| a.throughput_gib_s.partial_cmp(&b.throughput_gib_s).unwrap())
    else {
      best.push(crate::SizeClassBest {
        class,
        kernel: "portable".to_string(),
        streams: 1,
        throughput_gib_s: 0.0,
      });
      continue;
    };

    best.push(crate::SizeClassBest {
      class,
      kernel: winner.kernel.clone(),
      streams: winner.streams,
      throughput_gib_s: winner.throughput_gib_s,
    });
  }

  Ok(best)
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
