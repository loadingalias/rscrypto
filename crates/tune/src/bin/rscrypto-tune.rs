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

  /// Apply results into tuned_defaults.rs tables.
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
    -f, --format FORMAT   Output format: summary (default), env, json, tsv
    --apply               Update tuned_defaults.rs for this TuneKind
    --warmup-ms MS        Custom warmup duration (default: 150, quick: 75)
    --measure-ms MS       Custom measurement duration (default: 250, quick: 125)
    -h, --help            Show this help message

	EXAMPLES:
	    # Standard tuning run
	    just tune

	    # Quick run for development
	    just tune-quick

	    # Output as shell exports
	    just tune -- --format env

	    # Output as JSON
	    just tune -- --format json

	    # Apply to tuned defaults (writes into crates/checksum)
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
