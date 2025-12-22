//! Lightweight CRC64 tuning helper.
//!
//! This binary is intentionally small and dependency-free:
//! - It spawns itself with different `RSCRYPTO_CRC64_*` env overrides.
//! - Each child process benchmarks CRC64/XZ and CRC64/NVME for one configuration.
//! - The parent selects recommended settings and prints ready-to-use `export` lines.

use core::{cmp::Ordering, hint::black_box, str::FromStr, time::Duration};
use std::{
  env, fs, io,
  path::PathBuf,
  process::{Command, ExitCode},
  time::Instant,
};

use checksum::{Checksum, Crc64, Crc64Nvme};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Force {
  Auto,
  Portable,
  Pclmul,
  Vpclmul,
  Pmull,
  PmullEor3,
  Sve2Pmull,
  Vpmsum,
}

impl Force {
  const fn as_env_value(self) -> Option<&'static str> {
    match self {
      Self::Auto => None,
      Self::Portable => Some("portable"),
      Self::Pclmul => Some("pclmul"),
      Self::Vpclmul => Some("vpclmul"),
      Self::Pmull => Some("pmull"),
      Self::PmullEor3 => Some("pmull-eor3"),
      Self::Sve2Pmull => Some("sve2-pmull"),
      Self::Vpmsum => Some("vpmsum"),
    }
  }

  const fn name(self) -> &'static str {
    match self {
      Self::Auto => "auto",
      Self::Portable => "portable",
      Self::Pclmul => "pclmul",
      Self::Vpclmul => "vpclmul",
      Self::Pmull => "pmull",
      Self::PmullEor3 => "pmull-eor3",
      Self::Sve2Pmull => "sve2-pmull",
      Self::Vpmsum => "vpmsum",
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
crc64-tune: fast CRC64 tuner for rscrypto

USAGE:
  cargo run -p checksum --release --bin crc64-tune -- [--quick] [--apply]

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
    return &[1, 2, 4, 7];
  }
  #[cfg(target_arch = "aarch64")]
  {
    &[1, 2, 3]
  }
  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  {
    &[1, 2, 4, 8]
  }
  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "powerpc64", target_endian = "little")
  )))]
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
  xz_kernel: String,
  xz_gib_s: f64,
  nvme_kernel: String,
  nvme_gib_s: f64,
}

fn print_env_exports(streams: u8, portable_to_clmul: usize, pclmul_to_vpclmul: Option<usize>) {
  println!("# rscrypto CRC64 tuning (paste into your shell env)");
  println!("export RSCRYPTO_CRC64_STREAMS={streams}");
  println!("export RSCRYPTO_CRC64_THRESHOLD_PORTABLE_TO_CLMUL={portable_to_clmul}");
  if let Some(threshold) = pclmul_to_vpclmul {
    println!("export RSCRYPTO_CRC64_THRESHOLD_PCLMUL_TO_VPCLMUL={threshold}");
  }
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
        eprintln!("crc64-tune worker failed: {err}");
        ExitCode::FAILURE
      }
    }
  } else {
    match parent_main(args) {
      Ok(()) => ExitCode::SUCCESS,
      Err(err) => {
        eprintln!("crc64-tune failed: {err}");
        ExitCode::FAILURE
      }
    }
  }
}

fn parent_main(args: Args) -> io::Result<()> {
  let det = platform::get();
  let tune_kind = det.tune.kind;
  println!("platform: {}", platform::describe());
  println!("crc64/xz backend: {}", Crc64::backend_name());
  println!("crc64/nvme backend: {}", Crc64Nvme::backend_name());
  println!("tune: {}", det.tune);
  println!();

  // `--quick` can be noisy enough to pick the wrong stream count.
  // If we're going to write results back into the repo, enforce sane minima.
  let mut warmup_ms = args.warmup_ms;
  let mut measure_ms = args.measure_ms;
  if args.apply {
    let next_warmup = warmup_ms.max(150);
    let next_measure = measure_ms.max(250);
    if next_warmup != warmup_ms || next_measure != measure_ms {
      warmup_ms = next_warmup;
      measure_ms = next_measure;
      println!("# note: --apply uses --warmup-ms {warmup_ms} --measure-ms {measure_ms} for stability");
      println!();
    }
  }

  #[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "powerpc64", target_endian = "little")
  ))]
  let forces: Vec<Force> = {
    let caps = det.caps;
    let mut forces: Vec<Force> = vec![Force::Portable];

    #[cfg(target_arch = "x86_64")]
    {
      use platform::caps::x86;
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
      if caps.has(aarch64::PMULL_READY) {
        forces.push(Force::Pmull);
      }
      if caps.has(aarch64::PMULL_EOR3_READY) {
        forces.push(Force::PmullEor3);
      }
      if caps.has(aarch64::SVE2_PMULL) && caps.has(aarch64::PMULL_READY) {
        forces.push(Force::Sve2Pmull);
      }
    }

    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    {
      use platform::caps::powerpc64;
      if caps.has(powerpc64::VPMSUM_READY) {
        forces.push(Force::Vpmsum);
      }
    }

    forces
  };

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "powerpc64", target_endian = "little")
  )))]
  let forces: Vec<Force> = vec![Force::Portable];

  let stream_candidates = stream_candidates_for_arch();
  let mut stream_runs: Vec<RunConfig> = Vec::new();
  for &force in &forces {
    let candidates = if force == Force::Portable {
      &[1u8][..]
    } else {
      stream_candidates
    };
    for &streams in candidates {
      stream_runs.push(RunConfig { force, streams });
    }
  }

  let stream_sizes = sizes_for_streams();
  let stream_rows = run_matrix(&stream_runs, stream_sizes, warmup_ms, measure_ms)?;

  // Pick the best streams per force across the selected large-buffer sizes.
  let best_by_force = select_best_by_force(&stream_rows, stream_sizes);

  let baseline_simd_force = baseline_auto_simd_force(&forces);
  if baseline_simd_force == Force::Portable {
    println!("No SIMD CRC64 backend detected; portable only.");
    return Ok(());
  }

  // Decide which SIMD force auto is expected to use for large buffers.
  let preferred_simd_force = preferred_auto_simd_force(&forces);
  let Some((_, preferred_simd)) = best_by_force.iter().find(|(force, _)| *force == preferred_simd_force) else {
    println!("No SIMD CRC64 backend detected; portable only.");
    return Ok(());
  };

  let best_force = preferred_simd_force;
  let chosen_streams = preferred_simd.streams;

  // Phase B: threshold curves (portable vs baseline SIMD; pclmul vs vpclmul on x86).
  let mut threshold_runs: Vec<RunConfig> = vec![
    RunConfig {
      force: Force::Portable,
      streams: 1,
    },
    RunConfig {
      force: baseline_simd_force,
      streams: chosen_streams,
    },
  ];

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
  let threshold_rows = run_matrix(&threshold_runs, threshold_sizes, warmup_ms, measure_ms)?;

  const MARGIN: f64 = 1.00;

  #[cfg(target_arch = "x86_64")]
  let (best_force, chosen_streams, pclmul_to_vpclmul, threshold_rows) = {
    let mut best_force = best_force;
    let mut chosen_streams = chosen_streams;
    let mut pclmul_to_vpclmul: Option<usize> = None;
    let mut threshold_rows = threshold_rows;

    if forces.contains(&Force::Pclmul) && forces.contains(&Force::Vpclmul) {
      let pclmul_to_vpclmul_xz = estimate_threshold_sustained_by(
        &threshold_rows,
        Force::Pclmul,
        chosen_streams,
        Force::Vpclmul,
        chosen_streams,
        |r| r.xz_gib_s,
        MARGIN,
      );
      let pclmul_to_vpclmul_nvme = estimate_threshold_sustained_by(
        &threshold_rows,
        Force::Pclmul,
        chosen_streams,
        Force::Vpclmul,
        chosen_streams,
        |r| r.nvme_gib_s,
        MARGIN,
      );

      pclmul_to_vpclmul = match (pclmul_to_vpclmul_xz, pclmul_to_vpclmul_nvme) {
        (Some(xz), Some(nvme)) => Some(xz.max(nvme)),
        _ => None,
      };

      // If VPCLMUL never wins consistently, disable it and tune streams
      // for PCLMUL instead.
      if best_force == Force::Vpclmul && pclmul_to_vpclmul.is_none() {
        if let Some((_, best_pclmul)) = best_by_force.iter().find(|(force, _)| *force == Force::Pclmul) {
          best_force = Force::Pclmul;
          chosen_streams = best_pclmul.streams;

          // Re-run the portable<->pclmul crossover with the final stream setting.
          threshold_runs = vec![
            RunConfig {
              force: Force::Portable,
              streams: 1,
            },
            RunConfig {
              force: Force::Pclmul,
              streams: chosen_streams,
            },
          ];
          threshold_rows = run_matrix(&threshold_runs, threshold_sizes, warmup_ms, measure_ms)?;
        }
      }
    }

    Ok::<_, io::Error>((best_force, chosen_streams, pclmul_to_vpclmul, threshold_rows))
  }?;

  #[cfg(not(target_arch = "x86_64"))]
  let pclmul_to_vpclmul: Option<usize> = None;

  let portable_to_clmul_xz = estimate_threshold_sustained_by(
    &threshold_rows,
    Force::Portable,
    1,
    baseline_simd_force,
    chosen_streams,
    |r| r.xz_gib_s,
    MARGIN,
  );
  let portable_to_clmul_nvme = estimate_threshold_sustained_by(
    &threshold_rows,
    Force::Portable,
    1,
    baseline_simd_force,
    chosen_streams,
    |r| r.nvme_gib_s,
    MARGIN,
  );
  let portable_to_clmul = match (portable_to_clmul_xz, portable_to_clmul_nvme) {
    (Some(xz), Some(nvme)) => xz.max(nvme),
    _ => {
      println!(
        "# warning: portable->clmul crossover not found; using tune preset {}",
        det.tune.pclmul_threshold
      );
      det.tune.pclmul_threshold
    }
  };

  let Some((_, best_simd)) = best_by_force.iter().find(|(force, _)| *force == best_force) else {
    println!("No SIMD CRC64 backend detected; portable only.");
    return Ok(());
  };

  println!(
    "best large-buffer config: force={} eff={} streams={} nvme={:.2} GiB/s xz={:.2} GiB/s",
    best_simd.requested_force, best_simd.effective_force, chosen_streams, best_simd.nvme_gib_s, best_simd.xz_gib_s
  );
  println!("kernels: nvme={} xz={}", best_simd.nvme_kernel, best_simd.xz_kernel);
  println!(
    "portable->clmul crossover (sustained): xz={:?} nvme={:?} chosen={portable_to_clmul}",
    portable_to_clmul_xz, portable_to_clmul_nvme
  );
  #[cfg(target_arch = "x86_64")]
  {
    println!(
      "pclmul->vpclmul crossover (sustained): chosen={}",
      pclmul_to_vpclmul.map_or("disabled".to_owned(), |v| v.to_string())
    );
  }
  println!();

  print_env_exports(chosen_streams, portable_to_clmul, pclmul_to_vpclmul);
  if args.apply {
    #[cfg(target_arch = "x86_64")]
    let pclmul_to_vpclmul_value = pclmul_to_vpclmul.unwrap_or(usize::MAX);
    #[cfg(not(target_arch = "x86_64"))]
    let pclmul_to_vpclmul_value = usize::MAX;
    apply_tuned_defaults(tune_kind, chosen_streams, portable_to_clmul, pclmul_to_vpclmul_value)?;
    println!(
      "# applied baked defaults for {:?} in {}",
      tune_kind,
      tuned_defaults_path().display()
    );
  }
  Ok(())
}

fn tuned_defaults_path() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/crc64/tuned_defaults.rs")
}

fn apply_tuned_defaults(
  kind: platform::TuneKind,
  streams: u8,
  portable_to_clmul: usize,
  pclmul_to_vpclmul: usize,
) -> io::Result<()> {
  const BEGIN: &str = "// BEGIN GENERATED (crc64-tune --apply)";
  const END: &str = "// END GENERATED (crc64-tune --apply)";

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

  // Parse the existing generated block so we can re-emit it in a stable,
  // rustfmt-proof one-entry-per-line format.
  let mut entries: Vec<(String, (u8, usize, usize))> = Vec::new();

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
        current.clear();
        continue;
      };

      let streams = parse_u8_field(&current, "streams:").ok_or_else(|| {
        io::Error::new(
          io::ErrorKind::InvalidData,
          format!("Missing/invalid streams for {kind_ident}"),
        )
      })?;
      let portable_to_clmul = parse_usize_field(&current, "portable_to_clmul:").ok_or_else(|| {
        io::Error::new(
          io::ErrorKind::InvalidData,
          format!("Missing/invalid portable_to_clmul for {kind_ident}"),
        )
      })?;
      let pclmul_to_vpclmul = parse_usize_field(&current, "pclmul_to_vpclmul:").ok_or_else(|| {
        io::Error::new(
          io::ErrorKind::InvalidData,
          format!("Missing/invalid pclmul_to_vpclmul for {kind_ident}"),
        )
      })?;

      if let Some((_, existing)) = entries.iter_mut().find(|(k, _)| k == &kind_ident) {
        *existing = (streams, portable_to_clmul, pclmul_to_vpclmul);
      } else {
        entries.push((kind_ident, (streams, portable_to_clmul, pclmul_to_vpclmul)));
      }
      current.clear();
    }
  }

  let kind_ident = format!("{kind:?}");
  if let Some((_, existing)) = entries.iter_mut().find(|(k, _)| k == &kind_ident) {
    *existing = (streams, portable_to_clmul, pclmul_to_vpclmul);
  } else {
    entries.push((kind_ident, (streams, portable_to_clmul, pclmul_to_vpclmul)));
  }
  entries.sort_by(|a, b| a.0.cmp(&b.0));

  let mut emitted: Vec<String> = Vec::new();
  for (kind_ident, (streams, portable_to_clmul, pclmul_to_vpclmul)) in entries {
    let portable_to_clmul = fmt_usize(portable_to_clmul);
    let pclmul_to_vpclmul = fmt_usize(pclmul_to_vpclmul);
    emitted.push(format!(
      "  (TuneKind::{kind_ident}, Crc64TunedDefaults {{ streams: {streams}, portable_to_clmul: {portable_to_clmul}, \
       pclmul_to_vpclmul: {pclmul_to_vpclmul} }}),"
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

fn fmt_usize(value: usize) -> String {
  if value == usize::MAX {
    "usize::MAX".to_owned()
  } else {
    value.to_string()
  }
}

fn paren_delta(value: &str) -> i32 {
  let mut delta = 0i32;
  for ch in value.chars() {
    match ch {
      '(' => delta += 1,
      ')' => delta -= 1,
      _ => {}
    }
  }
  delta
}

fn token_after<'a>(haystack: &'a str, key: &str) -> Option<&'a str> {
  let idx = haystack.find(key)?;
  let rest = haystack[idx + key.len()..].trim_start();
  if rest.is_empty() {
    return None;
  }
  let end = rest
    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))
    .unwrap_or(rest.len());
  let token = rest[..end].trim();
  if token.is_empty() { None } else { Some(token) }
}

fn parse_u8_field(haystack: &str, key: &str) -> Option<u8> {
  let token = token_after(haystack, key)?;
  u8::from_str(token).ok()
}

fn parse_usize_field(haystack: &str, key: &str) -> Option<usize> {
  let token = token_after(haystack, key)?;
  if token == "usize::MAX" {
    return Some(usize::MAX);
  }
  usize::from_str(token).ok()
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

fn preferred_auto_simd_force(available: &[Force]) -> Force {
  // Mirror the library's auto preference order for CRC64:
  // - x86_64: VPCLMUL if available, else PCLMUL
  // - aarch64: PMULL+EOR3, else SVE2-PMULL, else PMULL
  // - powerpc64le: VPMSUMD
  #[cfg(target_arch = "x86_64")]
  {
    if available.iter().any(|f| *f == Force::Vpclmul) {
      return Force::Vpclmul;
    }
    if available.iter().any(|f| *f == Force::Pclmul) {
      return Force::Pclmul;
    }
    return Force::Portable;
  }

  #[cfg(target_arch = "aarch64")]
  {
    if available.contains(&Force::PmullEor3) {
      return Force::PmullEor3;
    }
    if available.contains(&Force::Sve2Pmull) {
      return Force::Sve2Pmull;
    }
    if available.contains(&Force::Pmull) {
      return Force::Pmull;
    }
    Force::Portable
  }

  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  {
    if available.contains(&Force::Vpmsum) {
      return Force::Vpmsum;
    }
    Force::Portable
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "powerpc64", target_endian = "little")
  )))]
  {
    let _ = available;
    Force::Portable
  }
}

fn baseline_auto_simd_force(available: &[Force]) -> Force {
  // Which backend should we compare against portable when tuning the
  // portable->SIMD threshold?
  //
  // - x86_64: PCLMUL (VPCLMUL is a higher tier with different overhead)
  // - aarch64: PMULL (base tier; EOR3/SVE2 build on top of it)
  // - powerpc64le: VPMSUMD
  #[cfg(target_arch = "x86_64")]
  {
    if available.contains(&Force::Pclmul) {
      return Force::Pclmul;
    }
    if available.contains(&Force::Vpclmul) {
      return Force::Vpclmul;
    }
    return Force::Portable;
  }

  #[cfg(target_arch = "aarch64")]
  {
    if available.contains(&Force::Pmull) {
      return Force::Pmull;
    }
    if available.contains(&Force::PmullEor3) {
      return Force::PmullEor3;
    }
    if available.contains(&Force::Sve2Pmull) {
      return Force::Sve2Pmull;
    }
    Force::Portable
  }

  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  {
    if available.contains(&Force::Vpmsum) {
      return Force::Vpmsum;
    }
    Force::Portable
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "powerpc64", target_endian = "little")
  )))]
  {
    let _ = available;
    Force::Portable
  }
}

fn parse_force_name(name: &str) -> Force {
  match name {
    "portable" => Force::Portable,
    "pclmul" => Force::Pclmul,
    "vpclmul" => Force::Vpclmul,
    "pmull" => Force::Pmull,
    "pmull-eor3" => Force::PmullEor3,
    "sve2-pmull" => Force::Sve2Pmull,
    "vpmsum" => Force::Vpmsum,
    _ => Force::Auto,
  }
}

fn lookup_row(rows: &[BenchRow], force: Force, streams: u8, size: usize) -> Option<&BenchRow> {
  rows
    .iter()
    .find(|r| r.requested_force == force.name() && r.streams == streams && r.size == size)
}

fn combined_metric(row: &BenchRow) -> f64 {
  (row.xz_gib_s * row.nvme_gib_s).sqrt()
}

fn select_best_by_force(rows: &[BenchRow], sizes: &[usize]) -> Vec<(Force, BenchRow)> {
  let max_size = sizes.iter().copied().max().unwrap_or(0);
  let mut pairs: Vec<(Force, u8)> = Vec::new();
  for row in rows {
    let force = parse_force_name(row.requested_force.as_str());
    if force == Force::Auto {
      continue;
    }
    let pair = (force, row.streams);
    if !pairs.contains(&pair) {
      pairs.push(pair);
    }
  }

  let mut best: Vec<(Force, BenchRow, f64)> = Vec::new();
  for (force, streams) in pairs {
    let Some(row_max) = lookup_row(rows, force, streams, max_size) else {
      continue;
    };

    let mut log_sum = 0.0f64;
    let mut count = 0usize;
    for &size in sizes {
      let Some(row) = lookup_row(rows, force, streams, size) else {
        count = 0;
        break;
      };
      let metric = combined_metric(row);
      if metric.partial_cmp(&0.0) != Some(Ordering::Greater) {
        count = 0;
        break;
      }
      log_sum += metric.ln();
      count += 1;
    }

    let score = if count == 0 {
      0.0
    } else {
      (log_sum / count as f64).exp()
    };
    let candidate_row = row_max.clone();

    match best.iter_mut().find(|(best_force, _, _)| *best_force == force) {
      Some((_, best_row, best_score)) => {
        if score > *best_score || (score == *best_score && better_row(&candidate_row, best_row)) {
          *best_row = candidate_row;
          *best_score = score;
        }
      }
      None => best.push((force, candidate_row, score)),
    }
  }

  best.into_iter().map(|(f, r, _)| (f, r)).collect()
}

fn better_row(a: &BenchRow, b: &BenchRow) -> bool {
  // Tiebreak used for equal aggregate scores:
  // - Higher NVME throughput at the largest size
  // - Then higher XZ throughput
  // - Then fewer streams (less register pressure)
  match a.nvme_gib_s.partial_cmp(&b.nvme_gib_s).unwrap_or(Ordering::Equal) {
    Ordering::Greater => true,
    Ordering::Less => false,
    Ordering::Equal => match a.xz_gib_s.partial_cmp(&b.xz_gib_s).unwrap_or(Ordering::Equal) {
      Ordering::Greater => true,
      Ordering::Less => false,
      Ordering::Equal => a.streams < b.streams,
    },
  }
}

fn estimate_threshold_sustained_by<F>(
  rows: &[BenchRow],
  lhs_force: Force,
  lhs_streams: u8,
  rhs_force: Force,
  rhs_streams: u8,
  mut metric: F,
  margin: f64,
) -> Option<usize>
where
  F: FnMut(&BenchRow) -> f64,
{
  let mut threshold: Option<usize> = None;
  let mut suffix_ok = true;
  let sizes = sizes_for_thresholds();

  for &size in sizes.iter().rev() {
    let lhs_row = lookup_row(rows, lhs_force, lhs_streams, size)?;
    let rhs_row = lookup_row(rows, rhs_force, rhs_streams, size)?;

    let lhs = metric(lhs_row);
    let rhs = metric(rhs_row);
    let ok_here = rhs >= lhs * margin;
    suffix_ok = suffix_ok && ok_here;
    if suffix_ok {
      threshold = Some(size);
    }
  }

  threshold
}

fn run_matrix(configs: &[RunConfig], sizes: &[usize], warmup_ms: u64, measure_ms: u64) -> io::Result<Vec<BenchRow>> {
  let exe = env::current_exe()?;
  let mut rows = Vec::new();

  for cfg in configs {
    let mut cmd = Command::new(&exe);
    cmd.arg("--worker");
    cmd.arg("--warmup-ms").arg(cfg_value(warmup_ms));
    cmd.arg("--measure-ms").arg(cfg_value(measure_ms));

    // Ensure we don't inherit overrides from the parent environment.
    for key in [
      "RSCRYPTO_CRC64_FORCE",
      "RSCRYPTO_CRC64_STREAMS",
      "RSCRYPTO_CRC64_THRESHOLD_PORTABLE_TO_CLMUL",
      "RSCRYPTO_CRC64_THRESHOLD_PCLMUL_TO_VPCLMUL",
    ] {
      cmd.env_remove(key);
    }

    if let Some(force) = cfg.force.as_env_value() {
      cmd.env("RSCRYPTO_CRC64_FORCE", force);
    }
    cmd.env("RSCRYPTO_CRC64_STREAMS", cfg_value(cfg.streams));

    // Pass sizes list via an env var to keep the arg parser minimal.
    cmd.env("RSCRYPTO_CRC64_TUNE_SIZES", sizes_csv(sizes));

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
  // requested_force \t effective_force \t streams \t size \t xz_kernel \t xz_gib_s \t nvme_kernel \t
  // nvme_gib_s
  let fields: Vec<&str> = line.split('\t').collect();
  if fields.len() != 8 {
    return Err(format!("Invalid worker output (expected 8 fields): {line}"));
  }

  let requested_force = fields[0].to_owned();
  let effective_force = fields[1].to_owned();
  let streams = u8::from_str(fields[2]).map_err(|_| format!("Invalid streams: {}", fields[2]))?;
  let size = usize::from_str(fields[3]).map_err(|_| format!("Invalid size: {}", fields[3]))?;
  let xz_kernel = fields[4].to_owned();
  let xz_gib_s = f64::from_str(fields[5]).map_err(|_| format!("Invalid xz GiB/s: {}", fields[5]))?;
  let nvme_kernel = fields[6].to_owned();
  let nvme_gib_s = f64::from_str(fields[7]).map_err(|_| format!("Invalid nvme GiB/s: {}", fields[7]))?;

  Ok(BenchRow {
    requested_force,
    effective_force,
    streams,
    size,
    xz_kernel,
    xz_gib_s,
    nvme_kernel,
    nvme_gib_s,
  })
}

fn worker_main(args: Args) -> io::Result<()> {
  let sizes = env::var("RSCRYPTO_CRC64_TUNE_SIZES")
    .ok()
    .map(parse_sizes_csv)
    .transpose()
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?
    .unwrap_or_else(|| sizes_for_thresholds().to_vec());

  let max_size = sizes.iter().copied().max().unwrap_or(0);
  let mut buffer = vec![0u8; max_size];
  fill_data(&mut buffer);

  // Ensure config is initialized once (and consistent) before measuring.
  let cfg = Crc64Nvme::config();
  let requested_force = cfg.requested_force.as_str().to_owned();
  let effective_force = cfg.effective_force.as_str().to_owned();
  let streams = cfg.tunables.streams;

  for size in sizes {
    if size == 0 || size > buffer.len() {
      continue;
    }
    let data = &buffer[..size];

    let xz_kernel = Crc64::kernel_name_for_len(size).to_owned();
    let nvme_kernel = Crc64Nvme::kernel_name_for_len(size).to_owned();

    let xz_gib_s = measure_gib_s(
      Crc64::checksum,
      data,
      Duration::from_millis(args.warmup_ms),
      Duration::from_millis(args.measure_ms),
    );
    let nvme_gib_s = measure_gib_s(
      Crc64Nvme::checksum,
      data,
      Duration::from_millis(args.warmup_ms),
      Duration::from_millis(args.measure_ms),
    );

    println!(
      "{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{:.6}",
      requested_force, effective_force, streams, size, xz_kernel, xz_gib_s, nvme_kernel, nvme_gib_s
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
    let x = (i as u8).wrapping_mul(31).wrapping_add((i >> 8) as u8);
    *b = x;
  }
}

fn measure_gib_s<F>(mut f: F, data: &[u8], warmup: Duration, measure: Duration) -> f64
where
  F: FnMut(&[u8]) -> u64,
{
  let mut acc: u64 = 0;

  // Warm-up: run for at least `warmup` duration to amortize cold-start effects.
  let warmup_deadline = Instant::now() + warmup;
  let warm_batch = batch_size_for_len(data.len());
  while Instant::now() < warmup_deadline {
    for _ in 0..warm_batch {
      acc ^= black_box(f(black_box(data)));
    }
  }

  // Measurement: time-based loop for stable-ish throughput.
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
  // Target ~64KiB processed per time-check to reduce Instant overhead.
  let target = 64 * 1024usize;
  let batch = target / len;
  batch.clamp(1, 4096) as u32
}
