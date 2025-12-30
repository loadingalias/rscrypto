//! Lightweight CRC32 tuning helper.
//!
//! This binary is intentionally small and dependency-free:
//! - It spawns itself with different `RSCRYPTO_CRC32_*` env overrides.
//! - Each child process benchmarks CRC32 (IEEE) and CRC32C (Castagnoli) for one configuration.
//! - The parent selects recommended settings and prints ready-to-use `export` lines.

use core::{cmp::Ordering, hint::black_box, str::FromStr, time::Duration};
use std::{
  env, fs, io,
  path::PathBuf,
  process::{Command, ExitCode},
  time::Instant,
};

use checksum::{Checksum, Crc32, Crc32C};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Force {
  Auto,
  Portable,
  Hwcrc,
  Pclmul,
  Vpclmul,
  Pmull,
  PmullEor3,
  Sve2Pmull,
}

impl Force {
  const fn as_env_value(self) -> Option<&'static str> {
    match self {
      Self::Auto => None,
      Self::Portable => Some("portable"),
      Self::Hwcrc => Some("hwcrc"),
      Self::Pclmul => Some("pclmul"),
      Self::Vpclmul => Some("vpclmul"),
      Self::Pmull => Some("pmull"),
      Self::PmullEor3 => Some("pmull-eor3"),
      Self::Sve2Pmull => Some("sve2-pmull"),
    }
  }

  const fn name(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Portable => "portable",
      Self::Hwcrc => "hwcrc",
      Self::Pclmul => "pclmul",
      Self::Vpclmul => "vpclmul",
      Self::Pmull => "pmull",
      Self::PmullEor3 => "pmull-eor3",
      Self::Sve2Pmull => "sve2-pmull",
    }
  }
}

#[derive(Clone, Copy, Debug)]
struct RunConfig {
  force: Force,
  streams: u8,
}

#[derive(Clone, Debug)]
struct Args {
  worker: bool,
  apply: bool,
  warmup_ms: u64,
  measure_ms: u64,
}

impl Default for Args {
  fn default() -> Self {
    Self {
      worker: false,
      apply: false,
      warmup_ms: 150,
      measure_ms: 250,
    }
  }
}

fn parse_args() -> Result<Args, String> {
  let mut args = Args::default();
  let mut it = env::args().skip(1);
  while let Some(arg) = it.next() {
    match arg.as_str() {
      "--" => continue,
      "--worker" => args.worker = true,
      "--apply" => args.apply = true,
      "--quick" => {
        args.warmup_ms = 75;
        args.measure_ms = 125;
      }
      "--warmup-ms" => {
        let Some(value) = it.next() else {
          return Err("--warmup-ms requires a value".to_owned());
        };
        args.warmup_ms = parse_u64("--warmup-ms", &value)?;
      }
      "--measure-ms" => {
        let Some(value) = it.next() else {
          return Err("--measure-ms requires a value".to_owned());
        };
        args.measure_ms = parse_u64("--measure-ms", &value)?;
      }
      "--help" | "-h" => {
        print_help();
        return Err(String::new());
      }
      other => return Err(format!("Unknown arg: {other}")),
    }
  }
  Ok(args)
}

fn print_help() {
  eprintln!(
    "\
crc32-tune: fast CRC32 tuner for rscrypto

USAGE:
  cargo run -p checksum --release --bin crc32-tune -- [--quick] [--apply]

OPTIONS:
  --quick                 Shorter warmup/measurement (noisier)
  --apply                 Update baked defaults in the repo (writes tuned_defaults.rs)
  --warmup-ms <ms>        Warmup time per size (default 150)
  --measure-ms <ms>       Measurement time per size (default 250)
"
  );
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
  u64::from_str(value).map_err(|_| format!("Invalid value for {flag}: {value}"))
}

fn sizes_for_thresholds() -> &'static [usize] {
  &[
    16,
    32,
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
  ]
}

fn sizes_for_streams() -> &'static [usize] {
  &[1024 * 1024]
}

fn stream_candidates_for_arch() -> &'static [u8] {
  #[cfg(target_arch = "x86_64")]
  {
    &[1, 2, 4, 7, 8]
  }
  #[cfg(target_arch = "aarch64")]
  {
    &[1, 2, 3]
  }
  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  {
    &[1]
  }
}

#[derive(Clone, Debug)]
struct BenchRow {
  requested_force: String,
  effective_force: String,
  streams: u8,
  size: usize,
  crc32_kernel: String,
  crc32_gib_s: f64,
  crc32c_kernel: String,
  crc32c_gib_s: f64,
}

fn print_env_exports(
  streams_crc32: u8,
  streams_crc32c: u8,
  portable_to_hwcrc: usize,
  hwcrc_to_fusion: usize,
  fusion_to_avx512: usize,
  fusion_to_vpclmul: usize,
) {
  println!("# rscrypto CRC32 tuning (paste into your shell env)");
  println!("export RSCRYPTO_CRC32_STREAMS_CRC32={streams_crc32}");
  println!("export RSCRYPTO_CRC32_STREAMS_CRC32C={streams_crc32c}");
  println!("export RSCRYPTO_CRC32_THRESHOLD_PORTABLE_TO_HWCRC={portable_to_hwcrc}");
  println!("export RSCRYPTO_CRC32_THRESHOLD_HWCRC_TO_FUSION={hwcrc_to_fusion}");
  println!("export RSCRYPTO_CRC32_THRESHOLD_FUSION_TO_AVX512={fusion_to_avx512}");
  println!("export RSCRYPTO_CRC32_THRESHOLD_FUSION_TO_VPCLMUL={fusion_to_vpclmul}");
}

fn main() -> ExitCode {
  let args = match parse_args() {
    Ok(args) => args,
    Err(msg) => {
      if msg.is_empty() {
        return ExitCode::SUCCESS;
      }
      eprintln!("{msg}");
      return ExitCode::FAILURE;
    }
  };

  if args.worker {
    match worker_main(args) {
      Ok(()) => ExitCode::SUCCESS,
      Err(err) => {
        eprintln!("crc32-tune worker failed: {err}");
        ExitCode::FAILURE
      }
    }
  } else {
    match parent_main(args) {
      Ok(()) => ExitCode::SUCCESS,
      Err(err) => {
        eprintln!("crc32-tune failed: {err}");
        ExitCode::FAILURE
      }
    }
  }
}

fn forces_for_caps(det: platform::Detected) -> Vec<Force> {
  let caps = det.caps;
  let mut forces: Vec<Force> = vec![Force::Portable];

  #[cfg(target_arch = "x86_64")]
  {
    use platform::caps::x86;
    if caps.has(x86::CRC32C_READY) {
      forces.push(Force::Hwcrc);
    }
    if caps.has(x86::PCLMUL_READY) {
      forces.push(Force::Pclmul);
    }
    if caps.has(x86::VPCLMUL_READY) {
      forces.push(Force::Vpclmul);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use platform::caps::aarch64;
    if caps.has(aarch64::CRC_READY) {
      forces.push(Force::Hwcrc);
      if caps.has(aarch64::PMULL_READY) {
        forces.push(Force::Pmull);
      }
      if caps.has(aarch64::PMULL_EOR3_READY) {
        forces.push(Force::PmullEor3);
      }
      if caps.has(aarch64::PMULL_READY) && caps.has(aarch64::SVE2_PMULL) {
        forces.push(Force::Sve2Pmull);
      }
    }
  }

  forces.sort_by(|a, b| a.name().cmp(b.name()));
  forces.dedup();
  forces
}

fn baseline_auto_simd_force(forces: &[Force]) -> Force {
  #[cfg(target_arch = "aarch64")]
  {
    if forces.contains(&Force::Pmull) {
      return Force::Pmull;
    }
    if forces.contains(&Force::Hwcrc) {
      return Force::Hwcrc;
    }
    Force::Portable
  }

  #[cfg(target_arch = "x86_64")]
  {
    if forces.contains(&Force::Pclmul) {
      return Force::Pclmul;
    }
    if forces.contains(&Force::Hwcrc) {
      return Force::Hwcrc;
    }
    Force::Portable
  }

  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  {
    let _ = forces;
    Force::Portable
  }
}

fn preferred_auto_simd_force(forces: &[Force]) -> Force {
  #[cfg(target_arch = "aarch64")]
  {
    if forces.contains(&Force::PmullEor3) {
      return Force::PmullEor3;
    }
    if forces.contains(&Force::Sve2Pmull) {
      return Force::Sve2Pmull;
    }
    if forces.contains(&Force::Pmull) {
      return Force::Pmull;
    }
    if forces.contains(&Force::Hwcrc) {
      return Force::Hwcrc;
    }
    Force::Portable
  }

  #[cfg(target_arch = "x86_64")]
  {
    if forces.contains(&Force::Vpclmul) {
      return Force::Vpclmul;
    }
    if forces.contains(&Force::Pclmul) {
      return Force::Pclmul;
    }
    if forces.contains(&Force::Hwcrc) {
      return Force::Hwcrc;
    }
    Force::Portable
  }

  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  {
    let _ = forces;
    Force::Portable
  }
}

fn parent_main(args: Args) -> io::Result<()> {
  let det = platform::get();
  let tune_kind = det.tune.kind;
  println!("platform: {}", platform::describe());
  println!("crc32 backend: {}", Crc32::backend_name());
  println!("crc32c backend: {}", Crc32C::backend_name());
  println!("tune: {}", det.tune);
  println!();

  let forces = forces_for_caps(det);
  let baseline_simd_force = baseline_auto_simd_force(&forces);
  let preferred_simd_force = preferred_auto_simd_force(&forces);
  #[cfg(not(target_arch = "x86_64"))]
  let _ = baseline_simd_force;

  if preferred_simd_force == Force::Portable {
    println!("No SIMD CRC32 backend detected; portable only.");
    return Ok(());
  }

  // Phase A: pick best streams for each available force.
  let stream_candidates = stream_candidates_for_arch();
  let mut stream_runs: Vec<RunConfig> = Vec::new();
  for &force in &forces {
    let candidates: &[u8] = if force == Force::Portable {
      &[1]
    } else {
      stream_candidates
    };
    for &streams in candidates {
      stream_runs.push(RunConfig { force, streams });
    }
  }
  stream_runs.sort_by(|a, b| (a.force.name(), a.streams).cmp(&(b.force.name(), b.streams)));
  stream_runs.dedup_by(|a, b| a.force == b.force && a.streams == b.streams);

  let stream_sizes = sizes_for_streams();
  let stream_rows = run_matrix(&stream_runs, stream_sizes, args.warmup_ms, args.measure_ms)?;
  let best_by_force = select_best_by_force(&stream_rows, stream_sizes);

  let Some((_, preferred_simd)) = best_by_force.iter().find(|(force, _)| *force == preferred_simd_force) else {
    println!("No SIMD CRC32 backend detected; portable only.");
    return Ok(());
  };
  let chosen_streams = preferred_simd.streams;

  // Phase B: threshold curves.
  let mut threshold_runs: Vec<RunConfig> = vec![
    RunConfig {
      force: Force::Portable,
      streams: 1,
    },
    RunConfig {
      force: preferred_simd_force,
      streams: chosen_streams,
    },
  ];

  if forces.contains(&Force::Hwcrc) {
    threshold_runs.push(RunConfig {
      force: Force::Hwcrc,
      streams: chosen_streams,
    });
  }

  #[cfg(target_arch = "x86_64")]
  {
    if forces.contains(&Force::Pclmul) && forces.contains(&Force::Vpclmul) {
      threshold_runs.push(RunConfig {
        force: Force::Pclmul,
        streams: chosen_streams,
      });
      threshold_runs.push(RunConfig {
        force: Force::Vpclmul,
        streams: chosen_streams,
      });
    }
  }

  threshold_runs.sort_by(|a, b| (a.force.name(), a.streams).cmp(&(b.force.name(), b.streams)));
  threshold_runs.dedup_by(|a, b| a.force == b.force && a.streams == b.streams);

  let threshold_sizes = sizes_for_thresholds();
  let threshold_rows = run_matrix(&threshold_runs, threshold_sizes, args.warmup_ms, args.measure_ms)?;

  const MARGIN: f64 = 1.00;

  // portable->hwcrc (CRC32C) and portable->first-accel (CRC32)
  let portable_to_hwcrc_crc32c = if forces.contains(&Force::Hwcrc) {
    estimate_threshold_sustained_by(
      &threshold_rows,
      Force::Portable,
      1,
      Force::Hwcrc,
      chosen_streams,
      |r| r.crc32c_gib_s,
      MARGIN,
    )
  } else {
    None
  };

  #[cfg(target_arch = "x86_64")]
  let portable_to_hwcrc_crc32 = estimate_threshold_sustained_by(
    &threshold_rows,
    Force::Portable,
    1,
    baseline_simd_force,
    chosen_streams,
    |r| r.crc32_gib_s,
    MARGIN,
  );

  #[cfg(target_arch = "aarch64")]
  let portable_to_hwcrc_crc32 = if forces.contains(&Force::Hwcrc) {
    estimate_threshold_sustained_by(
      &threshold_rows,
      Force::Portable,
      1,
      Force::Hwcrc,
      chosen_streams,
      |r| r.crc32_gib_s,
      MARGIN,
    )
  } else {
    None
  };

  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  let portable_to_hwcrc_crc32: Option<usize> = None;

  let portable_to_hwcrc = match (portable_to_hwcrc_crc32, portable_to_hwcrc_crc32c) {
    (Some(a), Some(b)) => a.max(b),
    (Some(a), None) => a,
    (None, Some(b)) => b,
    (None, None) => det.tune.hwcrc_threshold,
  };

  // hwcrc->fusion (CRC32C), portable->fusion (CRC32 on x86), hwcrc->fusion (CRC32 on aarch64)
  let hwcrc_to_fusion_crc32c = if forces.contains(&Force::Hwcrc) {
    estimate_threshold_sustained_by(
      &threshold_rows,
      Force::Hwcrc,
      chosen_streams,
      preferred_simd_force,
      chosen_streams,
      |r| r.crc32c_gib_s,
      MARGIN,
    )
  } else {
    None
  };

  #[cfg(target_arch = "x86_64")]
  let hwcrc_to_fusion_crc32 = estimate_threshold_sustained_by(
    &threshold_rows,
    Force::Portable,
    1,
    preferred_simd_force,
    chosen_streams,
    |r| r.crc32_gib_s,
    MARGIN,
  );

  #[cfg(target_arch = "aarch64")]
  let hwcrc_to_fusion_crc32 = if forces.contains(&Force::Hwcrc) {
    estimate_threshold_sustained_by(
      &threshold_rows,
      Force::Hwcrc,
      chosen_streams,
      preferred_simd_force,
      chosen_streams,
      |r| r.crc32_gib_s,
      MARGIN,
    )
  } else {
    None
  };

  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  let hwcrc_to_fusion_crc32: Option<usize> = None;

  let hwcrc_to_fusion = match (hwcrc_to_fusion_crc32, hwcrc_to_fusion_crc32c) {
    (Some(a), Some(b)) => a.max(b),
    (Some(a), None) => a,
    (None, Some(b)) => b,
    (None, None) => det.tune.pclmul_threshold,
  };

  // fusion->vpclmul (x86_64 only)
  #[cfg(target_arch = "x86_64")]
  let fusion_to_vpclmul = if forces.contains(&Force::Pclmul) && forces.contains(&Force::Vpclmul) {
    let crc32 = estimate_threshold_sustained_by(
      &threshold_rows,
      Force::Pclmul,
      chosen_streams,
      Force::Vpclmul,
      chosen_streams,
      |r| r.crc32_gib_s,
      MARGIN,
    );
    let crc32c = estimate_threshold_sustained_by(
      &threshold_rows,
      Force::Pclmul,
      chosen_streams,
      Force::Vpclmul,
      chosen_streams,
      |r| r.crc32c_gib_s,
      MARGIN,
    );
    match (crc32, crc32c) {
      (Some(a), Some(b)) => a.max(b),
      _ => usize::MAX,
    }
  } else {
    usize::MAX
  };
  #[cfg(not(target_arch = "x86_64"))]
  let fusion_to_vpclmul: usize = usize::MAX;

  // AVX-512 fusion threshold is currently best-effort (no dedicated force mode).
  let fusion_to_avx512 = det.tune.simd_threshold;

  let Some((_, best_simd)) = best_by_force.iter().find(|(force, _)| *force == preferred_simd_force) else {
    println!("No SIMD CRC32 backend detected; portable only.");
    return Ok(());
  };

  println!(
    "best large-buffer config: force={} eff={} streams={} crc32={:.2} GiB/s crc32c={:.2} GiB/s",
    best_simd.requested_force, best_simd.effective_force, chosen_streams, best_simd.crc32_gib_s, best_simd.crc32c_gib_s
  );
  println!(
    "kernels: crc32={} crc32c={}",
    best_simd.crc32_kernel, best_simd.crc32c_kernel
  );
  println!(
    "portable->accel crossover (sustained): crc32={:?} crc32c={:?} chosen={portable_to_hwcrc}",
    portable_to_hwcrc_crc32, portable_to_hwcrc_crc32c
  );
  println!(
    "hwcrc/portable->fusion crossover (sustained): crc32={:?} crc32c={:?} chosen={hwcrc_to_fusion}",
    hwcrc_to_fusion_crc32, hwcrc_to_fusion_crc32c
  );
  #[cfg(target_arch = "x86_64")]
  {
    println!(
      "fusion->vpclmul crossover (sustained): chosen={}",
      if fusion_to_vpclmul == usize::MAX {
        "disabled".to_owned()
      } else {
        fusion_to_vpclmul.to_string()
      }
    );
  }
  println!();

  print_env_exports(
    chosen_streams,
    chosen_streams,
    portable_to_hwcrc,
    hwcrc_to_fusion,
    fusion_to_avx512,
    fusion_to_vpclmul,
  );
  if args.apply {
    apply_tuned_defaults(
      tune_kind,
      chosen_streams,
      chosen_streams,
      portable_to_hwcrc,
      hwcrc_to_fusion,
      fusion_to_avx512,
      fusion_to_vpclmul,
    )?;
    println!(
      "# applied baked defaults for {:?} in {}",
      tune_kind,
      tuned_defaults_path().display()
    );
  }
  Ok(())
}

fn tuned_defaults_path() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/crc32/tuned_defaults.rs")
}

fn apply_tuned_defaults(
  kind: platform::TuneKind,
  streams_crc32: u8,
  streams_crc32c: u8,
  portable_to_hwcrc: usize,
  hwcrc_to_fusion: usize,
  fusion_to_avx512: usize,
  fusion_to_vpclmul: usize,
) -> io::Result<()> {
  const BEGIN: &str = "// BEGIN GENERATED (crc32-tune --apply)";
  const END: &str = "// END GENERATED (crc32-tune --apply)";

  let path = tuned_defaults_path();
  let src = fs::read_to_string(&path)?;
  let lines: Vec<&str> = src.lines().collect();

  let Some(begin_idx) = lines.iter().position(|l| l.contains(BEGIN)) else {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      format!("Missing marker in {}: {BEGIN}", path.display()),
    ));
  };
  let Some(end_idx) = lines.iter().position(|l| l.contains(END)) else {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      format!("Missing marker in {}: {END}", path.display()),
    ));
  };
  if end_idx <= begin_idx {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      format!("Invalid marker order in {}", path.display()),
    ));
  }

  #[derive(Clone, Copy, Debug)]
  struct TunedEntry {
    streams_crc32: u8,
    streams_crc32c: u8,
    portable_to_hwcrc: usize,
    hwcrc_to_fusion: usize,
    fusion_to_avx512: usize,
    fusion_to_vpclmul: usize,
  }

  let mut entries: Vec<(String, TunedEntry)> = Vec::new();

  let mut current = String::new();
  let mut depth: i32 = 0;
  let mut in_entry = false;
  for line in &lines[begin_idx + 1..end_idx] {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with("//") {
      continue;
    }

    if !in_entry {
      if !line.contains('(') {
        continue;
      }
      in_entry = true;
      current.clear();
      depth = 0;
    }

    current.push_str(line);
    current.push('\n');
    depth += paren_delta(line);

    if in_entry && depth == 0 {
      in_entry = false;

      let Some(kind_ident) = parse_tune_kind_ident(&current) else {
        continue;
      };
      let streams_crc32 = parse_u8_field(&current, "streams_crc32:")
        .or_else(|| parse_u8_field(&current, "streams:"))
        .ok_or_else(|| {
          io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Missing/invalid streams_crc32 for {kind_ident}"),
          )
        })?;
      let streams_crc32c = parse_u8_field(&current, "streams_crc32c:")
        .or_else(|| parse_u8_field(&current, "streams:"))
        .ok_or_else(|| {
          io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Missing/invalid streams_crc32c for {kind_ident}"),
          )
        })?;
      let portable_to_hwcrc = parse_usize_field(&current, "portable_to_hwcrc:").ok_or_else(|| {
        io::Error::new(
          io::ErrorKind::InvalidData,
          format!("Missing/invalid portable_to_hwcrc for {kind_ident}"),
        )
      })?;
      let hwcrc_to_fusion = parse_usize_field(&current, "hwcrc_to_fusion:").ok_or_else(|| {
        io::Error::new(
          io::ErrorKind::InvalidData,
          format!("Missing/invalid hwcrc_to_fusion for {kind_ident}"),
        )
      })?;
      let fusion_to_avx512 = parse_usize_field(&current, "fusion_to_avx512:").ok_or_else(|| {
        io::Error::new(
          io::ErrorKind::InvalidData,
          format!("Missing/invalid fusion_to_avx512 for {kind_ident}"),
        )
      })?;
      let fusion_to_vpclmul = parse_usize_field(&current, "fusion_to_vpclmul:").ok_or_else(|| {
        io::Error::new(
          io::ErrorKind::InvalidData,
          format!("Missing/invalid fusion_to_vpclmul for {kind_ident}"),
        )
      })?;

      let values = TunedEntry {
        streams_crc32,
        streams_crc32c,
        portable_to_hwcrc,
        hwcrc_to_fusion,
        fusion_to_avx512,
        fusion_to_vpclmul,
      };

      if let Some(existing) = entries.iter_mut().find(|(k, _)| *k == kind_ident) {
        *existing = (kind_ident, values);
      } else {
        entries.push((kind_ident, values));
      }
    }
  }

  let kind_ident = format!("{kind:?}");
  let values = TunedEntry {
    streams_crc32,
    streams_crc32c,
    portable_to_hwcrc,
    hwcrc_to_fusion,
    fusion_to_avx512,
    fusion_to_vpclmul,
  };
  if let Some(existing) = entries.iter_mut().find(|(k, _)| *k == kind_ident) {
    *existing = (kind_ident, values);
  } else {
    entries.push((kind_ident, values));
  }

  entries.sort_by(|a, b| a.0.cmp(&b.0));

  let mut emitted: Vec<String> = Vec::new();
  for (kind_ident, values) in entries {
    let streams_crc32 = values.streams_crc32;
    let streams_crc32c = values.streams_crc32c;
    let portable_to_hwcrc = fmt_usize(values.portable_to_hwcrc);
    let hwcrc_to_fusion = fmt_usize(values.hwcrc_to_fusion);
    let fusion_to_avx512 = fmt_usize(values.fusion_to_avx512);
    let fusion_to_vpclmul = fmt_usize(values.fusion_to_vpclmul);
    emitted.push(format!(
      "  (TuneKind::{kind_ident}, Crc32TunedDefaults {{ streams_crc32: {streams_crc32}, streams_crc32c: \
       {streams_crc32c}, portable_to_hwcrc: {portable_to_hwcrc}, hwcrc_to_fusion: {hwcrc_to_fusion}, \
       fusion_to_avx512: {fusion_to_avx512}, fusion_to_vpclmul: {fusion_to_vpclmul} }}),"
    ));
  }

  let mut out = String::new();
  for line in &lines[..=begin_idx] {
    out.push_str(line);
    out.push('\n');
  }
  for line in emitted {
    out.push_str(&line);
    out.push('\n');
  }
  for line in &lines[end_idx..] {
    out.push_str(line);
    out.push('\n');
  }

  fs::write(&path, out)?;
  Ok(())
}

fn paren_delta(line: &str) -> i32 {
  let mut delta = 0;
  for ch in line.chars() {
    match ch {
      '(' => delta += 1,
      ')' => delta -= 1,
      _ => {}
    }
  }
  delta
}

fn parse_tune_kind_ident(line: &str) -> Option<String> {
  let start = line.find("TuneKind::")?;
  let rest = &line[start + "TuneKind::".len()..];
  let end = rest
    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
    .unwrap_or(rest.len());
  let ident = rest[..end].trim();
  if ident.is_empty() {
    return None;
  }
  Some(ident.to_owned())
}

fn parse_u8_field(src: &str, needle: &str) -> Option<u8> {
  let idx = src.find(needle)?;
  let rest = &src[idx + needle.len()..];
  let value = rest
    .split(|c: char| c == ',' || c == '}' || c.is_whitespace())
    .find(|s| !s.is_empty())?;
  value.parse::<u8>().ok()
}

fn parse_usize_field(src: &str, needle: &str) -> Option<usize> {
  let idx = src.find(needle)?;
  let rest = &src[idx + needle.len()..];
  let value = rest
    .split(|c: char| c == ',' || c == '}' || c.is_whitespace())
    .find(|s| !s.is_empty())?;
  if value == "usize::MAX" {
    return Some(usize::MAX);
  }
  value.parse::<usize>().ok()
}

fn fmt_usize(v: usize) -> String {
  if v == usize::MAX {
    "usize::MAX".to_owned()
  } else {
    v.to_string()
  }
}

fn run_matrix(configs: &[RunConfig], sizes: &[usize], warmup_ms: u64, measure_ms: u64) -> io::Result<Vec<BenchRow>> {
  let exe = env::current_exe()?;
  let mut rows = Vec::new();

  for cfg in configs {
    let mut cmd = Command::new(&exe);
    cmd.arg("--worker");
    cmd.arg("--warmup-ms").arg(cfg_value(warmup_ms));
    cmd.arg("--measure-ms").arg(cfg_value(measure_ms));

    for key in [
      "RSCRYPTO_CRC32_FORCE",
      "RSCRYPTO_CRC32_STREAMS_CRC32",
      "RSCRYPTO_CRC32_STREAMS_CRC32C",
      "RSCRYPTO_CRC32_THRESHOLD_PORTABLE_TO_HWCRC",
      "RSCRYPTO_CRC32_THRESHOLD_HWCRC_TO_FUSION",
      "RSCRYPTO_CRC32_THRESHOLD_FUSION_TO_AVX512",
      "RSCRYPTO_CRC32_THRESHOLD_FUSION_TO_VPCLMUL",
    ] {
      cmd.env_remove(key);
    }

    if let Some(force) = cfg.force.as_env_value() {
      cmd.env("RSCRYPTO_CRC32_FORCE", force);
    }
    cmd.env("RSCRYPTO_CRC32_STREAMS_CRC32", cfg_value(cfg.streams));
    cmd.env("RSCRYPTO_CRC32_STREAMS_CRC32C", cfg_value(cfg.streams));
    cmd.env("RSCRYPTO_CRC32_TUNE_SIZES", sizes_csv(sizes));

    let output = cmd.output()?;
    if !output.status.success() {
      return Err(io::Error::other(format!(
        "worker failed for force={} streams={} (status={})",
        cfg.force.name(),
        cfg.streams,
        output.status
      )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
      rows.push(parse_row(line).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?);
    }
  }

  Ok(rows)
}

fn sizes_csv(sizes: &[usize]) -> String {
  let mut out = String::new();
  for (i, size) in sizes.iter().enumerate() {
    if i != 0 {
      out.push(',');
    }
    out.push_str(&size.to_string());
  }
  out
}

fn cfg_value<T: core::fmt::Display>(value: T) -> String {
  value.to_string()
}

fn parse_row(line: &str) -> Result<BenchRow, String> {
  // Format:
  // requested_force \t effective_force \t streams \t size \t crc32_kernel \t crc32_gib_s \t
  // crc32c_kernel \t crc32c_gib_s
  let fields: Vec<&str> = line.split('\t').collect();
  if fields.len() != 8 {
    return Err(format!("Invalid worker output (expected 8 fields): {line}"));
  }

  let requested_force = fields[0].to_owned();
  let effective_force = fields[1].to_owned();
  let streams = u8::from_str(fields[2]).map_err(|_| format!("Invalid streams: {}", fields[2]))?;
  let size = usize::from_str(fields[3]).map_err(|_| format!("Invalid size: {}", fields[3]))?;
  let crc32_kernel = fields[4].to_owned();
  let crc32_gib_s = f64::from_str(fields[5]).map_err(|_| format!("Invalid crc32 GiB/s: {}", fields[5]))?;
  let crc32c_kernel = fields[6].to_owned();
  let crc32c_gib_s = f64::from_str(fields[7]).map_err(|_| format!("Invalid crc32c GiB/s: {}", fields[7]))?;

  Ok(BenchRow {
    requested_force,
    effective_force,
    streams,
    size,
    crc32_kernel,
    crc32_gib_s,
    crc32c_kernel,
    crc32c_gib_s,
  })
}

fn worker_main(args: Args) -> io::Result<()> {
  let sizes = env::var("RSCRYPTO_CRC32_TUNE_SIZES")
    .ok()
    .map(parse_sizes_csv)
    .transpose()
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?
    .unwrap_or_else(|| sizes_for_thresholds().to_vec());

  let max_size = sizes.iter().copied().max().unwrap_or(0);
  let mut buffer = vec![0u8; max_size];
  fill_data(&mut buffer);

  // Ensure config is initialized once (and consistent) before measuring.
  let cfg = Crc32C::config();
  let requested_force = cfg.requested_force.as_str().to_owned();
  let effective_force = cfg.effective_force.as_str().to_owned();
  let streams = cfg.tunables.streams_crc32;

  for size in sizes {
    if size == 0 || size > buffer.len() {
      continue;
    }
    let data = &buffer[..size];

    let crc32_kernel = Crc32::kernel_name_for_len(size).to_owned();
    let crc32c_kernel = Crc32C::kernel_name_for_len(size).to_owned();

    let crc32_gib_s = measure_gib_s_u32(
      Crc32::checksum,
      data,
      Duration::from_millis(args.warmup_ms),
      Duration::from_millis(args.measure_ms),
    );
    let crc32c_gib_s = measure_gib_s_u32(
      Crc32C::checksum,
      data,
      Duration::from_millis(args.warmup_ms),
      Duration::from_millis(args.measure_ms),
    );

    println!(
      "{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{:.6}",
      requested_force, effective_force, streams, size, crc32_kernel, crc32_gib_s, crc32c_kernel, crc32c_gib_s
    );
  }

  Ok(())
}

fn parse_sizes_csv(value: String) -> Result<Vec<usize>, String> {
  let mut sizes = Vec::new();
  for part in value.split(',') {
    let part = part.trim();
    if part.is_empty() {
      continue;
    }
    let size = usize::from_str(part).map_err(|_| format!("Invalid size: {part}"))?;
    sizes.push(size);
  }
  if sizes.is_empty() {
    return Err("No sizes specified".to_owned());
  }
  Ok(sizes)
}

fn fill_data(buf: &mut [u8]) {
  for (i, b) in buf.iter_mut().enumerate() {
    let x = (i as u8).wrapping_mul(31).wrapping_add(i.strict_shr(8) as u8);
    *b = x;
  }
}

fn measure_gib_s_u32<F>(mut f: F, data: &[u8], warmup: Duration, measure: Duration) -> f64
where
  F: FnMut(&[u8]) -> u32,
{
  let mut acc: u32 = 0;

  let warmup_deadline = Instant::now() + warmup;
  let warm_batch = batch_size_for_len(data.len());
  while Instant::now() < warmup_deadline {
    for _ in 0..warm_batch {
      acc ^= black_box(f(black_box(data)));
    }
  }

  let start = Instant::now();
  let deadline = start + measure;
  let batch = batch_size_for_len(data.len());
  let mut bytes: u64 = 0;
  while Instant::now() < deadline {
    for _ in 0..batch {
      acc ^= black_box(f(black_box(data)));
    }
    bytes = bytes.saturating_add((data.len() as u64).saturating_mul(batch as u64));
  }

  black_box(acc);

  let elapsed = start.elapsed();
  if elapsed.as_secs_f64() <= 0.0 {
    return 0.0;
  }

  let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
  gib / elapsed.as_secs_f64()
}

fn batch_size_for_len(len: usize) -> u32 {
  if len == 0 {
    return 1;
  }
  if len <= 64 {
    return 50_000;
  }
  if len <= 1024 {
    return 10_000;
  }
  if len <= 16usize.strict_mul(1024) {
    return 2000;
  }
  if len <= 256usize.strict_mul(1024) {
    return 256;
  }
  32
}

fn select_best_by_force(rows: &[BenchRow], sizes: &[usize]) -> Vec<(Force, BenchRow)> {
  let mut out: Vec<(Force, BenchRow)> = Vec::new();

  let mut forces: Vec<Force> = Vec::new();
  for row in rows {
    let f = parse_force_name(&row.requested_force);
    if !forces.contains(&f) {
      forces.push(f);
    }
  }
  forces.sort_by(|a, b| a.name().cmp(b.name()));

  for force in forces {
    let mut best: Option<&BenchRow> = None;
    for row in rows.iter().filter(|r| parse_force_name(&r.requested_force) == force) {
      if !sizes.contains(&row.size) {
        continue;
      }
      match best {
        None => best = Some(row),
        Some(prev) => {
          let prev_score = prev.crc32_gib_s.min(prev.crc32c_gib_s);
          let score = row.crc32_gib_s.min(row.crc32c_gib_s);
          if score.partial_cmp(&prev_score).unwrap_or(Ordering::Equal) == Ordering::Greater {
            best = Some(row);
          }
        }
      }
    }
    if let Some(best) = best {
      out.push((force, best.clone()));
    }
  }

  out
}

fn parse_force_name(name: &str) -> Force {
  match name {
    "portable" => Force::Portable,
    "hwcrc" => Force::Hwcrc,
    "pclmul" => Force::Pclmul,
    "vpclmul" => Force::Vpclmul,
    "pmull" => Force::Pmull,
    "pmull-eor3" => Force::PmullEor3,
    "sve2-pmull" => Force::Sve2Pmull,
    _ => Force::Auto,
  }
}

fn estimate_threshold_sustained_by<F>(
  rows: &[BenchRow],
  left_force: Force,
  left_streams: u8,
  right_force: Force,
  right_streams: u8,
  metric: F,
  margin: f64,
) -> Option<usize>
where
  F: Fn(&BenchRow) -> f64,
{
  let mut pairs: Vec<(&BenchRow, &BenchRow)> = Vec::new();
  for size in sizes_for_thresholds() {
    let left = rows
      .iter()
      .find(|r| parse_force_name(&r.requested_force) == left_force && r.streams == left_streams && r.size == *size);
    let right = rows
      .iter()
      .find(|r| parse_force_name(&r.requested_force) == right_force && r.streams == right_streams && r.size == *size);
    if let (Some(l), Some(r)) = (left, right) {
      pairs.push((l, r));
    }
  }

  // We want the smallest size such that RHS is >= LHS for all larger sizes.
  // Scan from large→small and keep a running "suffix ok" flag.
  let mut threshold: Option<usize> = None;
  let mut suffix_ok = true;

  for (l, r) in pairs.into_iter().rev() {
    let lv = metric(l);
    let rv = metric(r);
    let ok_here = rv >= lv * margin;
    suffix_ok = suffix_ok && ok_here;
    if suffix_ok {
      threshold = Some(l.size);
    }
  }

  threshold
}
