use crate::{
  Tunable, TuneError, analysis,
  runner::{BenchRunner, fill_data},
};

const BLAKE3_CHUNK_BYTES: usize = 1024;
const BLAKE3_PAR_MARGIN: f64 = 1.02;
const BLAKE3_THREAD_PICK_MARGIN: f64 = 1.01;
const BLAKE3_DEFAULT_PAR_SPAWN_COST_BYTES: usize = 24 * 1024;
const BLAKE3_DEFAULT_PAR_MERGE_COST_BYTES: usize = 16 * 1024;
const BLAKE3_DEFAULT_PAR_BYTES_PER_CORE_SMALL: usize = 256 * 1024;
const BLAKE3_DEFAULT_PAR_BYTES_PER_CORE_MEDIUM: usize = 128 * 1024;
const BLAKE3_DEFAULT_PAR_BYTES_PER_CORE_LARGE: usize = 64 * 1024;
const BLAKE3_DEFAULT_PAR_SMALL_LIMIT_BYTES: usize = 256 * 1024;
const BLAKE3_DEFAULT_PAR_MEDIUM_LIMIT_BYTES: usize = 2 * 1024 * 1024;
const BLAKE3_PAR_SIZES: &[usize] = &[
  64 * 1024,
  96 * 1024,
  128 * 1024,
  192 * 1024,
  256 * 1024,
  384 * 1024,
  512 * 1024,
  768 * 1024,
  1024 * 1024,
  1536 * 1024,
  2 * 1024 * 1024,
  4 * 1024 * 1024,
  8 * 1024 * 1024,
];

#[derive(Clone, Debug)]
struct Blake3ParallelFit {
  max_threads: usize,
  min_bytes: usize,
  min_chunks: usize,
  weighted_ratio: f64,
  peak_tp: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Blake3ParallelPolicy {
  pub(crate) min_bytes: usize,
  pub(crate) min_chunks: usize,
  pub(crate) max_threads: usize,
  pub(crate) spawn_cost_bytes: usize,
  pub(crate) merge_cost_bytes: usize,
  pub(crate) bytes_per_core_small: usize,
  pub(crate) bytes_per_core_medium: usize,
  pub(crate) bytes_per_core_large: usize,
  pub(crate) small_limit_bytes: usize,
  pub(crate) medium_limit_bytes: usize,
}

#[derive(Clone, Debug)]
struct Blake3ParallelCurve {
  fit: Blake3ParallelFit,
  throughput: Vec<(usize, f64)>,
}

fn build_blake3_crossover_measurements(
  single: &[(usize, f64)],
  parallel: &[(usize, f64)],
  size_scale: usize,
) -> Vec<analysis::Measurement> {
  let mut measurements = Vec::with_capacity(single.len().strict_add(parallel.len()));
  for &(size, tp) in single {
    measurements.push(analysis::Measurement {
      kernel: "single".to_string(),
      streams: 1,
      size: size / size_scale,
      throughput_gib_s: tp,
    });
  }
  for &(size, tp) in parallel {
    measurements.push(analysis::Measurement {
      kernel: "parallel".to_string(),
      streams: 1,
      size: size / size_scale,
      throughput_gib_s: tp,
    });
  }
  measurements
}

fn weighted_parallel_ratio(single: &[(usize, f64)], parallel: &[(usize, f64)], min_size: usize) -> f64 {
  let mut weighted = 0.0f64;
  let mut total_weight = 0.0f64;

  for ((single_size, single_tp), (parallel_size, parallel_tp)) in single.iter().zip(parallel.iter()) {
    if single_size != parallel_size || *single_size < min_size {
      continue;
    }

    let ratio = if *single_tp > 0.0 { parallel_tp / single_tp } else { 1.0 };
    let weight = *single_size as f64;
    weighted += ratio * weight;
    total_weight += weight;
  }

  if total_weight > 0.0 {
    weighted / total_weight
  } else {
    1.0
  }
}

#[inline]
#[must_use]
fn default_blake3_parallel_policy() -> Blake3ParallelPolicy {
  Blake3ParallelPolicy {
    min_bytes: 128 * 1024,
    min_chunks: 64,
    max_threads: 0,
    spawn_cost_bytes: BLAKE3_DEFAULT_PAR_SPAWN_COST_BYTES,
    merge_cost_bytes: BLAKE3_DEFAULT_PAR_MERGE_COST_BYTES,
    bytes_per_core_small: BLAKE3_DEFAULT_PAR_BYTES_PER_CORE_SMALL,
    bytes_per_core_medium: BLAKE3_DEFAULT_PAR_BYTES_PER_CORE_MEDIUM,
    bytes_per_core_large: BLAKE3_DEFAULT_PAR_BYTES_PER_CORE_LARGE,
    small_limit_bytes: BLAKE3_DEFAULT_PAR_SMALL_LIMIT_BYTES,
    medium_limit_bytes: BLAKE3_DEFAULT_PAR_MEDIUM_LIMIT_BYTES,
  }
}

#[inline]
fn median_usize(values: &mut [usize]) -> Option<usize> {
  if values.is_empty() {
    return None;
  }
  values.sort_unstable();
  Some(values[values.len() / 2])
}

#[inline]
fn fit_line_thread_bytes(samples: &[(usize, usize)]) -> Option<(f64, f64)> {
  if samples.len() < 2 {
    return None;
  }
  let n = samples.len() as f64;
  let mut sum_t = 0.0;
  let mut sum_y = 0.0;
  let mut sum_tt = 0.0;
  let mut sum_ty = 0.0;
  for (threads, bytes) in samples {
    let t = *threads as f64;
    let y = *bytes as f64;
    sum_t += t;
    sum_y += y;
    sum_tt += t * t;
    sum_ty += t * y;
  }
  let denom = n * sum_tt - sum_t * sum_t;
  if denom.abs() < f64::EPSILON {
    return None;
  }
  let slope = (n * sum_ty - sum_t * sum_y) / denom;
  let intercept = (sum_y - slope * sum_t) / n;
  Some((slope, intercept))
}

fn pick_best_threads_by_size(single: &[(usize, f64)], curves: &[Blake3ParallelCurve]) -> Vec<(usize, usize)> {
  let mut out = Vec::with_capacity(single.len());
  for (idx, &(size, single_tp)) in single.iter().enumerate() {
    let mut best_threads = 1usize;
    let mut best_tp = single_tp;
    for curve in curves {
      let Some((parallel_size, parallel_tp)) = curve.throughput.get(idx).copied() else {
        continue;
      };
      if parallel_size != size {
        continue;
      }
      if parallel_tp > best_tp * BLAKE3_THREAD_PICK_MARGIN {
        best_tp = parallel_tp;
        best_threads = curve.fit.max_threads;
      }
    }
    out.push((size, best_threads));
  }
  out
}

fn bytes_per_core_from_threads(
  best_threads_by_size: &[(usize, usize)],
  min_size: usize,
  max_size: usize,
  spawn_cost_bytes: usize,
  merge_cost_bytes: usize,
) -> Option<usize> {
  let mut candidates = Vec::new();
  for (size, threads) in best_threads_by_size {
    if *threads <= 1 || *size < min_size || *size > max_size {
      continue;
    }
    let overhead = merge_cost_bytes.saturating_add(spawn_cost_bytes.saturating_mul((*threads).saturating_sub(1)));
    let effective = (*size).saturating_sub(overhead) / *threads;
    if effective != 0 {
      candidates.push(effective);
    }
  }
  median_usize(&mut candidates)
}

fn blake3_policy_from_curves(
  single: &[(usize, f64)],
  curves: &[Blake3ParallelCurve],
  best_fit: Blake3ParallelFit,
) -> Blake3ParallelPolicy {
  let defaults = default_blake3_parallel_policy();
  let best_threads_by_size = pick_best_threads_by_size(single, curves);

  let mut crossover_samples = Vec::new();
  for curve in curves {
    crossover_samples.push((curve.fit.max_threads, curve.fit.min_bytes));
  }

  let mut small_bpc_samples = Vec::new();
  for (size, threads) in &best_threads_by_size {
    if *threads > 1 && *size <= BLAKE3_DEFAULT_PAR_SMALL_LIMIT_BYTES {
      small_bpc_samples.push(*size / *threads);
    }
  }
  let bpc_small = median_usize(&mut small_bpc_samples).unwrap_or(defaults.bytes_per_core_small);

  let (spawn_cost_bytes, merge_cost_bytes) = if let Some((slope, intercept)) = fit_line_thread_bytes(&crossover_samples)
  {
    let slope_usize = slope.max(0.0).round() as usize;
    let mut spawn = slope_usize.saturating_sub(bpc_small);
    let spawn_cap = defaults.spawn_cost_bytes.saturating_mul(4);
    spawn = spawn.min(spawn_cap);
    let merge = (intercept.max(0.0).round() as usize).saturating_add(spawn);
    (spawn.max(1), merge.max(1))
  } else {
    (
      defaults.spawn_cost_bytes,
      best_fit
        .min_bytes
        .saturating_sub(bpc_small.saturating_mul(2))
        .max(defaults.merge_cost_bytes / 2),
    )
  };

  let mut small_limit_bytes = defaults.small_limit_bytes;
  let mut medium_limit_bytes = defaults.medium_limit_bytes;
  let mut seen_small = false;
  let mut seen_medium = false;
  for (size, threads) in &best_threads_by_size {
    if *threads > 1 && *threads <= 2 {
      small_limit_bytes = *size;
      seen_small = true;
    }
    if *threads > 2 && *threads <= 4 {
      medium_limit_bytes = *size;
      seen_medium = true;
    }
  }
  if !seen_small {
    small_limit_bytes = defaults.small_limit_bytes.max(best_fit.min_bytes);
  }
  if !seen_medium {
    medium_limit_bytes = defaults.medium_limit_bytes.max(small_limit_bytes.saturating_mul(2));
  }
  medium_limit_bytes = medium_limit_bytes.max(small_limit_bytes.saturating_add(1));

  let bpc_medium = bytes_per_core_from_threads(
    &best_threads_by_size,
    small_limit_bytes.saturating_add(1),
    medium_limit_bytes,
    spawn_cost_bytes,
    merge_cost_bytes,
  )
  .unwrap_or(defaults.bytes_per_core_medium);

  let bpc_large = bytes_per_core_from_threads(
    &best_threads_by_size,
    medium_limit_bytes.saturating_add(1),
    usize::MAX,
    spawn_cost_bytes,
    merge_cost_bytes,
  )
  .unwrap_or(defaults.bytes_per_core_large);

  Blake3ParallelPolicy {
    min_bytes: best_fit.min_bytes,
    min_chunks: best_fit.min_chunks,
    max_threads: best_fit.max_threads,
    spawn_cost_bytes,
    merge_cost_bytes,
    bytes_per_core_small: bpc_small.max(1),
    bytes_per_core_medium: bpc_medium.max(1),
    bytes_per_core_large: bpc_large.max(1),
    small_limit_bytes,
    medium_limit_bytes,
  }
}

fn fit_blake3_parallel_curve(
  single: &[(usize, f64)],
  parallel: &[(usize, f64)],
  max_threads: usize,
) -> Option<Blake3ParallelFit> {
  let byte_measurements = build_blake3_crossover_measurements(single, parallel, 1);
  let byte_crossover = analysis::find_crossover(&byte_measurements, "single", 1, "parallel", 1, BLAKE3_PAR_MARGIN)?;

  let chunk_measurements = build_blake3_crossover_measurements(single, parallel, BLAKE3_CHUNK_BYTES);
  let chunk_crossover = analysis::find_crossover(&chunk_measurements, "single", 1, "parallel", 1, BLAKE3_PAR_MARGIN)?;

  let weighted_ratio = weighted_parallel_ratio(single, parallel, byte_crossover.crossover_size);
  let peak_tp = parallel.last().map(|(_, tp)| *tp).unwrap_or(0.0);

  Some(Blake3ParallelFit {
    max_threads,
    min_bytes: byte_crossover.crossover_size,
    min_chunks: chunk_crossover.crossover_size.max(1),
    weighted_ratio,
    peak_tp,
  })
}

fn select_best_blake3_parallel_fit(fits: &[Blake3ParallelFit]) -> Option<Blake3ParallelFit> {
  let mut best: Option<Blake3ParallelFit> = None;
  for fit in fits {
    let replace = if let Some(current) = &best {
      const EPS: f64 = 1e-6;
      if fit.weighted_ratio > current.weighted_ratio + EPS {
        true
      } else if fit.weighted_ratio + EPS < current.weighted_ratio {
        false
      } else if fit.peak_tp > current.peak_tp + EPS {
        true
      } else if fit.peak_tp + EPS < current.peak_tp {
        false
      } else if fit.min_bytes < current.min_bytes {
        true
      } else if fit.min_bytes > current.min_bytes {
        false
      } else {
        fit.max_threads < current.max_threads
      }
    } else {
      true
    };

    if replace {
      best = Some(fit.clone());
    }
  }
  best
}

fn measure_blake3_curve(
  runner: &BenchRunner,
  algorithm: &mut dyn Tunable,
  buffer: &[u8],
  sizes: &[usize],
  max_threads: usize,
) -> Result<Vec<(usize, f64)>, TuneError> {
  use hashes::crypto::blake3::{dispatch_tables::ParallelTable, tune::override_blake3_parallel_policy};

  let mut out = Vec::with_capacity(sizes.len());
  let _g = override_blake3_parallel_policy(ParallelTable {
    min_bytes: 0,
    min_chunks: 0,
    max_threads: max_threads.min(u8::MAX as usize) as u8,
    spawn_cost_bytes: 0,
    merge_cost_bytes: 0,
    bytes_per_core_small: 0,
    bytes_per_core_medium: 0,
    bytes_per_core_large: 0,
    small_limit_bytes: 0,
    medium_limit_bytes: 0,
  });
  for &size in sizes {
    let r = runner.measure_single(algorithm, &buffer[..size])?;
    out.push((size, r.throughput_gib_s));
  }
  Ok(out)
}

pub(crate) fn tune_blake3_parallel_policy(
  runner: &BenchRunner,
  algorithm: &mut dyn Tunable,
  best_kernel: &str,
) -> Result<Option<Blake3ParallelPolicy>, TuneError> {
  use hashes::crypto::blake3::{dispatch_tables::ParallelTable, tune::override_blake3_parallel_policy};

  let Ok(ap) = std::thread::available_parallelism() else {
    return Ok(None);
  };
  let avail = ap.get();
  if avail <= 1 {
    let mut policy = default_blake3_parallel_policy();
    policy.min_bytes = usize::MAX;
    policy.min_chunks = usize::MAX;
    policy.max_threads = 1;
    return Ok(Some(policy));
  }

  algorithm.reset();
  let _ = algorithm.force_kernel(best_kernel);

  let max_size = *BLAKE3_PAR_SIZES.last().unwrap_or(&0);
  let mut buffer = vec![0u8; max_size];
  fill_data(&mut buffer);

  let mut single_tp: Vec<(usize, f64)> = Vec::with_capacity(BLAKE3_PAR_SIZES.len());
  {
    let _g = override_blake3_parallel_policy(ParallelTable {
      min_bytes: usize::MAX,
      min_chunks: usize::MAX,
      max_threads: 1,
      spawn_cost_bytes: 0,
      merge_cost_bytes: 0,
      bytes_per_core_small: 0,
      bytes_per_core_medium: 0,
      bytes_per_core_large: 0,
      small_limit_bytes: 0,
      medium_limit_bytes: 0,
    });
    for &size in BLAKE3_PAR_SIZES {
      let r = runner.measure_single(algorithm, &buffer[..size])?;
      single_tp.push((size, r.throughput_gib_s));
    }
  }

  let max_cap = avail.min(32);
  let mut candidates: Vec<usize> = if max_cap <= 8 {
    (2..=max_cap).collect()
  } else {
    vec![2, 4, 8, 12, 16, max_cap]
  };
  candidates.sort_unstable();
  candidates.dedup();

  let mut curves = Vec::new();
  for threads in candidates {
    let parallel_tp = measure_blake3_curve(runner, algorithm, &buffer, BLAKE3_PAR_SIZES, threads)?;
    if let Some(fit) = fit_blake3_parallel_curve(&single_tp, &parallel_tp, threads) {
      curves.push(Blake3ParallelCurve {
        fit,
        throughput: parallel_tp,
      });
    }
  }

  let fits: Vec<Blake3ParallelFit> = curves.iter().map(|c| c.fit.clone()).collect();
  let Some(best_fit) = select_best_blake3_parallel_fit(&fits) else {
    let mut policy = default_blake3_parallel_policy();
    policy.min_bytes = usize::MAX;
    policy.min_chunks = usize::MAX;
    policy.max_threads = 1;
    return Ok(Some(policy));
  };

  Ok(Some(blake3_policy_from_curves(&single_tp, &curves, best_fit)))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn blake3_parallel_curve_fit_uses_curve_for_min_chunks() {
    let single = vec![
      (64 * 1024, 4.00),
      (96 * 1024, 4.05),
      (128 * 1024, 4.10),
      (192 * 1024, 4.15),
      (256 * 1024, 4.20),
    ];
    let parallel = vec![
      (64 * 1024, 3.40),
      (96 * 1024, 3.95),
      (128 * 1024, 4.30),
      (192 * 1024, 4.55),
      (256 * 1024, 4.75),
    ];

    let fit = fit_blake3_parallel_curve(&single, &parallel, 8).expect("parallel should win at larger sizes");

    assert_eq!(fit.min_bytes, 128 * 1024);
    assert_eq!(fit.min_chunks, 128);
    assert_eq!(fit.max_threads, 8);
  }

  #[test]
  fn blake3_parallel_curve_fit_returns_none_when_parallel_never_wins() {
    let single = vec![(64 * 1024, 4.0), (128 * 1024, 4.1), (256 * 1024, 4.2)];
    let parallel = vec![(64 * 1024, 3.0), (128 * 1024, 3.2), (256 * 1024, 3.5)];

    assert!(fit_blake3_parallel_curve(&single, &parallel, 4).is_none());
  }

  #[test]
  fn select_best_blake3_parallel_fit_prefers_lower_threshold_on_tie() {
    let fits = vec![
      Blake3ParallelFit {
        max_threads: 8,
        min_bytes: 256 * 1024,
        min_chunks: 256,
        weighted_ratio: 1.100,
        peak_tp: 9.0,
      },
      Blake3ParallelFit {
        max_threads: 12,
        min_bytes: 128 * 1024,
        min_chunks: 128,
        weighted_ratio: 1.100,
        peak_tp: 9.0,
      },
    ];

    let best = select_best_blake3_parallel_fit(&fits).expect("expected best fit");
    assert_eq!(best.min_bytes, 128 * 1024);
    assert_eq!(best.min_chunks, 128);
  }
}
