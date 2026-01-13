//! Unified tuning binary for rscrypto algorithms.
//!
//! Usage:
//!   cargo run --release -p tune --bin rscrypto-tune
//!   cargo run --release -p tune --bin rscrypto-tune -- --quick
//!   cargo run --release -p tune --bin rscrypto-tune -- --format env

use core::time::Duration;
use std::{env, process::ExitCode};

use tune::{
  BenchRunner, OutputFormat, PlatformInfo, TuneEngine,
  crc16::{Crc16CcittTunable, Crc16IbmTunable},
  crc24::Crc24OpenPgpTunable,
  crc32::{Crc32IeeeTunable, Crc32cTunable},
  crc64::{Crc64NvmeTunable, Crc64XzTunable},
  hash::HashTunable,
};

/// CLI arguments.
#[derive(Debug)]
struct Args {
  /// Quick mode (faster, noisier).
  quick: bool,

  /// Output format.
  format: OutputFormat,

  /// Verbose output.
  verbose: bool,

  /// Apply results into dispatch.rs kernel tables.
  apply: bool,

  /// Custom warmup duration in ms.
  warmup_ms: Option<u64>,

  /// Custom measurement duration in ms.
  measure_ms: Option<u64>,

  /// Show help.
  help: bool,
}

impl Default for Args {
  fn default() -> Self {
    Self {
      quick: false,
      format: OutputFormat::Summary,
      verbose: false,
      apply: false,
      warmup_ms: None,
      measure_ms: None,
      help: false,
    }
  }
}

fn parse_args() -> Result<Args, String> {
  let mut args = Args::default();
  let mut iter = env::args().skip(1);

  while let Some(arg) = iter.next() {
    match arg.as_str() {
      "--" => continue,
      "--quick" | "-q" => args.quick = true,
      "--verbose" | "-v" => args.verbose = true,
      "--help" | "-h" => args.help = true,
      "--apply" => args.apply = true,
      "--format" | "-f" => {
        let Some(value) = iter.next() else {
          return Err("--format requires a value".to_string());
        };
        args.format = OutputFormat::parse(&value).ok_or_else(|| format!("Unknown format: {value}"))?;
      }
      "--warmup-ms" => {
        let Some(value) = iter.next() else {
          return Err("--warmup-ms requires a value".to_string());
        };
        args.warmup_ms = Some(value.parse().map_err(|_| format!("Invalid warmup-ms: {value}"))?);
      }
      "--measure-ms" => {
        let Some(value) = iter.next() else {
          return Err("--measure-ms requires a value".to_string());
        };
        args.measure_ms = Some(value.parse().map_err(|_| format!("Invalid measure-ms: {value}"))?);
      }
      other => {
        return Err(format!("Unknown argument: {other}"));
      }
    }
  }

  Ok(args)
}

fn print_help() {
  eprintln!(
    "\
rscrypto-tune: Unified tuning engine for rscrypto algorithms

USAGE:
    cargo run --release -p tune --bin rscrypto-tune -- [OPTIONS]

    For best results, build with native CPU flags:
    RUSTFLAGS='-C target-cpu=native' cargo run --release -p tune --bin rscrypto-tune

OPTIONS:
    -q, --quick           Quick mode (faster, noisier measurements)
    -v, --verbose         Verbose output during tuning
    -f, --format FORMAT   Output format: summary (default), env, json, tsv, contribute
    --apply               Generate dispatch.rs table entry for this TuneKind
    --warmup-ms MS        Custom warmup duration (default: 150, quick: 75)
    --measure-ms MS       Custom measurement duration (default: 250, quick: 125)
    -h, --help            Show this help message

FORMATS:
    summary     Human-readable summary (default)
    env         Shell environment variable exports
    json        JSON for programmatic use
    tsv         Tab-separated values
    contribute  Markdown ready for GitHub issue submission

EXAMPLES:
    # Standard tuning run
    just tune

    # Quick run for development
    just tune-quick

    # Generate markdown for contributing your results
    just tune -- --format contribute

    # Output as JSON
    just tune -- --format json

    # Generate dispatch table entry (outputs to crates/checksum/src/dispatch.rs)
    just tune-apply
"
  );
}

fn main() -> ExitCode {
  let args = match parse_args() {
    Ok(args) => args,
    Err(msg) => {
      eprintln!("Error: {msg}");
      eprintln!("Run with --help for usage information.");
      return ExitCode::FAILURE;
    }
  };

  if args.help {
    print_help();
    return ExitCode::SUCCESS;
  }

  // Build the runner
  let runner = if args.quick {
    BenchRunner::quick()
  } else {
    BenchRunner::new()
  };

  let runner = match (args.warmup_ms, args.measure_ms) {
    (Some(w), Some(m)) => runner
      .with_warmup(Duration::from_millis(w))
      .with_measure(Duration::from_millis(m)),
    (Some(w), None) => runner.with_warmup(Duration::from_millis(w)),
    (None, Some(m)) => runner.with_measure(Duration::from_millis(m)),
    (None, None) => runner,
  };

  // Build the engine
  let mut engine = TuneEngine::new().with_runner(runner).with_verbose(args.verbose);

  // Register all checksum tunables
  engine.add(Box::new(Crc16CcittTunable::new()));
  engine.add(Box::new(Crc16IbmTunable::new()));
  engine.add(Box::new(Crc24OpenPgpTunable::new()));
  engine.add(Box::new(Crc32IeeeTunable::new()));
  engine.add(Box::new(Crc32cTunable::new()));
  engine.add(Box::new(Crc64XzTunable::new()));
  engine.add(Box::new(Crc64NvmeTunable::new()));
  engine.add(Box::new(HashTunable::new("sha224", "RSCRYPTO_SHA224")));
  engine.add(Box::new(HashTunable::new("sha256", "RSCRYPTO_SHA256")));
  engine.add(Box::new(HashTunable::new("sha384", "RSCRYPTO_SHA384")));
  engine.add(Box::new(HashTunable::new("sha512", "RSCRYPTO_SHA512")));
  engine.add(Box::new(HashTunable::new("sha512-224", "RSCRYPTO_SHA512_224")));
  engine.add(Box::new(HashTunable::new("sha512-256", "RSCRYPTO_SHA512_256")));
  engine.add(Box::new(HashTunable::new("blake2b-512", "RSCRYPTO_BLAKE2B_512")));
  engine.add(Box::new(HashTunable::new("blake2s-256", "RSCRYPTO_BLAKE2S_256")));
  engine.add(Box::new(HashTunable::new("blake3", "RSCRYPTO_BLAKE3")));
  engine.add(Box::new(HashTunable::new("sha3-224", "RSCRYPTO_SHA3_224")));
  engine.add(Box::new(HashTunable::new("sha3-256", "RSCRYPTO_SHA3_256")));
  engine.add(Box::new(HashTunable::new("sha3-384", "RSCRYPTO_SHA3_384")));
  engine.add(Box::new(HashTunable::new("sha3-512", "RSCRYPTO_SHA3_512")));
  engine.add(Box::new(HashTunable::new("shake128", "RSCRYPTO_SHAKE128")));
  engine.add(Box::new(HashTunable::new("shake256", "RSCRYPTO_SHAKE256")));
  engine.add(Box::new(HashTunable::new("cshake128", "RSCRYPTO_CSHAKE128")));
  engine.add(Box::new(HashTunable::new("cshake256", "RSCRYPTO_CSHAKE256")));
  engine.add(Box::new(HashTunable::new("kmac128", "RSCRYPTO_KMAC128")));
  engine.add(Box::new(HashTunable::new("kmac256", "RSCRYPTO_KMAC256")));
  engine.add(Box::new(HashTunable::new("xxh3", "RSCRYPTO_XXH3")));
  engine.add(Box::new(HashTunable::new("rapidhash", "RSCRYPTO_RAPIDHASH")));
  engine.add(Box::new(HashTunable::new("siphash", "RSCRYPTO_SIPHASH")));
  engine.add(Box::new(HashTunable::new("keccakf1600", "RSCRYPTO_KECCAKF1600")));
  engine.add(Box::new(HashTunable::new("ascon-hash256", "RSCRYPTO_ASCON_HASH256")));
  engine.add(Box::new(HashTunable::new("ascon-xof128", "RSCRYPTO_ASCON_XOF128")));

  // ────────────────────────────────────────────────────────────────────────────
  // Kernel-loop microbenches (drive SIMD work; no effect on runtime env vars)
  //
  // These are intentionally registered as separate "algorithms" so tuning output
  // can report steady-state kernel throughput, tail/finalization, and different
  // update chunking profiles independently.
  // ────────────────────────────────────────────────────────────────────────────
  engine.add(Box::new(HashTunable::new(
    "sha224-compress",
    "RSCRYPTO_BENCH_SHA224_COMPRESS",
  )));
  engine.add(Box::new(HashTunable::new(
    "sha256-compress",
    "RSCRYPTO_BENCH_SHA256_COMPRESS",
  )));
  engine.add(Box::new(HashTunable::new(
    "sha256-compress-unaligned",
    "RSCRYPTO_BENCH_SHA256_COMPRESS_UNALIGNED",
  )));
  engine.add(Box::new(HashTunable::new(
    "sha384-compress",
    "RSCRYPTO_BENCH_SHA384_COMPRESS",
  )));
  engine.add(Box::new(HashTunable::new(
    "sha512-compress",
    "RSCRYPTO_BENCH_SHA512_COMPRESS",
  )));
  engine.add(Box::new(HashTunable::new(
    "sha512-compress-unaligned",
    "RSCRYPTO_BENCH_SHA512_COMPRESS_UNALIGNED",
  )));
  engine.add(Box::new(HashTunable::new(
    "sha512-224-compress",
    "RSCRYPTO_BENCH_SHA512_224_COMPRESS",
  )));
  engine.add(Box::new(HashTunable::new(
    "sha512-256-compress",
    "RSCRYPTO_BENCH_SHA512_256_COMPRESS",
  )));

  engine.add(Box::new(HashTunable::new(
    "blake2b-512-compress",
    "RSCRYPTO_BENCH_BLAKE2B_512_COMPRESS",
  )));
  engine.add(Box::new(HashTunable::new(
    "blake2s-256-compress",
    "RSCRYPTO_BENCH_BLAKE2S_256_COMPRESS",
  )));

  engine.add(Box::new(HashTunable::new(
    "blake3-chunk",
    "RSCRYPTO_BENCH_BLAKE3_CHUNK",
  )));
  engine.add(Box::new(HashTunable::new(
    "blake3-parent",
    "RSCRYPTO_BENCH_BLAKE3_PARENT",
  )));

  engine.add(Box::new(HashTunable::new(
    "keccakf1600-permute",
    "RSCRYPTO_BENCH_KECCAKF1600_PERMUTE",
  )));

  // Chunking pattern profiles (many small updates vs few large updates).
  engine.add(Box::new(HashTunable::new(
    "sha256-stream64",
    "RSCRYPTO_BENCH_SHA256_STREAM64",
  )));
  engine.add(Box::new(HashTunable::new(
    "sha256-stream4k",
    "RSCRYPTO_BENCH_SHA256_STREAM4K",
  )));
  engine.add(Box::new(HashTunable::new(
    "sha512-stream64",
    "RSCRYPTO_BENCH_SHA512_STREAM64",
  )));
  engine.add(Box::new(HashTunable::new(
    "sha512-stream4k",
    "RSCRYPTO_BENCH_SHA512_STREAM4K",
  )));
  engine.add(Box::new(HashTunable::new(
    "blake2b-512-stream64",
    "RSCRYPTO_BENCH_BLAKE2B_512_STREAM64",
  )));
  engine.add(Box::new(HashTunable::new(
    "blake2b-512-stream4k",
    "RSCRYPTO_BENCH_BLAKE2B_512_STREAM4K",
  )));
  engine.add(Box::new(HashTunable::new(
    "blake2s-256-stream64",
    "RSCRYPTO_BENCH_BLAKE2S_256_STREAM64",
  )));
  engine.add(Box::new(HashTunable::new(
    "blake2s-256-stream4k",
    "RSCRYPTO_BENCH_BLAKE2S_256_STREAM4K",
  )));
  engine.add(Box::new(HashTunable::new(
    "blake3-stream64",
    "RSCRYPTO_BENCH_BLAKE3_STREAM64",
  )));
  engine.add(Box::new(HashTunable::new(
    "blake3-stream4k",
    "RSCRYPTO_BENCH_BLAKE3_STREAM4K",
  )));

  // Print header
  let platform = PlatformInfo::collect();
  eprintln!("rscrypto-tune");
  eprintln!("=============");
  eprintln!();
  eprintln!("Platform: {}", platform.description);
  eprintln!("Tune preset: {:?}", platform.tune_kind);
  eprintln!();

  // Run the tuning engine
  let results = match engine.run() {
    Ok(results) => results,
    Err(err) => {
      eprintln!("Tuning failed: {err}");
      return ExitCode::FAILURE;
    }
  };

  // Output results in requested format
  match args.format {
    OutputFormat::Summary => {
      if let Err(err) = tune::report::print_summary(&results) {
        eprintln!("Failed to print summary: {err}");
        return ExitCode::FAILURE;
      }
    }
    OutputFormat::Env => {
      if let Err(err) = tune::report::print_env(&results) {
        eprintln!("Failed to print env: {err}");
        return ExitCode::FAILURE;
      }
    }
    OutputFormat::Json => {
      if let Err(err) = tune::report::print_json(&results) {
        eprintln!("Failed to print json: {err}");
        return ExitCode::FAILURE;
      }
    }
    OutputFormat::Tsv => {
      if let Err(err) = tune::report::print_tsv(&results) {
        eprintln!("Failed to print tsv: {err}");
        return ExitCode::FAILURE;
      }
    }
    OutputFormat::Contribute => {
      if let Err(err) = tune::report::print_contribute(&results) {
        eprintln!("Failed to print contribute: {err}");
        return ExitCode::FAILURE;
      }
    }
  }

  if args.apply {
    let cwd = match std::env::current_dir() {
      Ok(p) => p,
      Err(err) => {
        eprintln!("Failed to resolve current directory for --apply: {err}");
        return ExitCode::FAILURE;
      }
    };

    if let Err(err) = tune::apply::apply_tuned_defaults(&cwd, &results) {
      eprintln!("Failed to apply tuned defaults: {err}");
      return ExitCode::FAILURE;
    }
  }

  ExitCode::SUCCESS
}
