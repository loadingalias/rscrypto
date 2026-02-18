//! Capture BLAKE3 boundary-size diagnostics into a single CSV artifact.

use std::{
  env,
  fs::{self, File},
  io::{self, Write},
  path::PathBuf,
  process,
};

use hashes::bench as microbench;
use tune::{BenchRunner, hash::HashTunable};

const BOUNDARY_SIZES: &[usize] = &[63, 64, 65, 96, 128, 192, 256, 512, 1024, 4096, 16384, 65536];
const BLAKE3_BOUNDARY_VARIANTS: &[(&str, &str)] = &[
  ("plain", "blake3"),
  ("keyed", "blake3-keyed"),
  ("derive", "blake3-derive"),
  ("xof-init-read-32", "blake3-latency-xof"),
];

#[derive(Debug)]
struct Args {
  output: Option<PathBuf>,
  warmup_ms: u64,
  measure_ms: u64,
  force_kernel: Option<String>,
}

impl Default for Args {
  fn default() -> Self {
    Self {
      output: None,
      warmup_ms: 100,
      measure_ms: 180,
      force_kernel: None,
    }
  }
}

fn usage() {
  eprintln!(
    "Usage: rscrypto-blake3-boundary [--output <path>] [--warmup-ms <ms>] [--measure-ms <ms>] [--force-kernel \
     <kernel>]"
  );
  eprintln!("Kernels: portable, sse41, avx2, avx512, neon (plus full names like x86_64/avx2)");
}

fn parse_args() -> Result<Args, String> {
  let mut args = Args::default();
  let mut it = env::args().skip(1);
  while let Some(arg) = it.next() {
    match arg.as_str() {
      "-h" | "--help" => {
        usage();
        process::exit(0);
      }
      "--output" => {
        let Some(value) = it.next() else {
          return Err("--output requires a path".to_string());
        };
        args.output = Some(PathBuf::from(value));
      }
      "--warmup-ms" => {
        let Some(value) = it.next() else {
          return Err("--warmup-ms requires an integer".to_string());
        };
        args.warmup_ms = value
          .parse::<u64>()
          .map_err(|_| format!("invalid --warmup-ms value: {value}"))?;
      }
      "--measure-ms" => {
        let Some(value) = it.next() else {
          return Err("--measure-ms requires an integer".to_string());
        };
        args.measure_ms = value
          .parse::<u64>()
          .map_err(|_| format!("invalid --measure-ms value: {value}"))?;
      }
      "--force-kernel" => {
        let Some(value) = it.next() else {
          return Err("--force-kernel requires a kernel name".to_string());
        };
        args.force_kernel = Some(value);
      }
      other => {
        return Err(format!("unknown argument: {other}"));
      }
    }
  }
  Ok(args)
}

fn fill_data(buf: &mut [u8]) {
  for (i, b) in buf.iter_mut().enumerate() {
    *b = (i as u8).wrapping_mul(31).wrapping_add((i >> 8) as u8);
  }
}

fn writer_from_path(path: Option<&PathBuf>) -> Result<Box<dyn Write>, String> {
  if let Some(path) = path {
    if let Some(parent) = path.parent()
      && !parent.as_os_str().is_empty()
    {
      fs::create_dir_all(parent).map_err(|e| format!("failed to create output directory {}: {e}", parent.display()))?;
    }
    let file = File::create(path).map_err(|e| format!("failed to create {}: {e}", path.display()))?;
    Ok(Box::new(file))
  } else {
    Ok(Box::new(io::stdout()))
  }
}

fn run() -> Result<(), String> {
  let args = parse_args()?;
  if let Some(force) = args.force_kernel.as_deref() {
    // SAFETY: We set the environment override exactly once at process startup,
    // before any worker threads are created by the benchmark path.
    unsafe { env::set_var("RSCRYPTO_BLAKE3_FORCE_KERNEL", force) };
  }

  let runner = BenchRunner::new()
    .with_warmup_ms(args.warmup_ms)
    .with_measure_ms(args.measure_ms)
    .without_variance_warnings();

  let mut out = writer_from_path(args.output.as_ref())?;
  writeln!(
    out,
    "variant,algo,size,kernel_name_for_len,effective_kernel,throughput_gib_s,latency_ns,iterations,samples,cv"
  )
  .map_err(|e| format!("failed to write header: {e}"))?;

  for (variant, algo) in BLAKE3_BOUNDARY_VARIANTS {
    let tunable = HashTunable::new(algo, "RSCRYPTO_BLAKE3_BOUNDARY");
    for size in BOUNDARY_SIZES {
      let mut data = vec![0u8; *size];
      fill_data(&mut data);

      let result = runner
        .measure_single(&tunable, &data)
        .map_err(|e| format!("benchmark failed for {algo}@{size}: {e}"))?;

      let kernel_name_for_len = microbench::kernel_name_for_len(algo, *size).unwrap_or("unknown");
      let latency_ns = if result.iterations == 0 {
        0.0
      } else {
        result.elapsed_secs * 1_000_000_000.0 / result.iterations as f64
      };
      let samples = result.sample_count.unwrap_or(0);
      let cv = result.cv.unwrap_or(0.0);

      writeln!(
        out,
        "{variant},{algo},{size},{kernel_name_for_len},{effective},{throughput:.6},{latency_ns:.3},{iters},{samples},\
         {cv:.6}",
        effective = result.kernel,
        throughput = result.throughput_gib_s,
        iters = result.iterations,
      )
      .map_err(|e| format!("failed to write csv row: {e}"))?;
    }
  }

  Ok(())
}

fn main() {
  if let Err(err) = run() {
    eprintln!("error: {err}");
    process::exit(2);
  }
}
