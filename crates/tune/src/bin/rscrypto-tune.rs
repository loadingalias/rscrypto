//! Unified tuning binary for rscrypto algorithms.
//!
//! Usage:
//!   cargo run --release -p tune --bin rscrypto-tune
//!   cargo run --release -p tune --bin rscrypto-tune -- --quick
//!   cargo run --release -p tune --bin rscrypto-tune -- --format env

use core::time::Duration;
use std::{
  collections::HashSet,
  env,
  fs::{self, File},
  io::BufWriter,
  path::Path,
  process::ExitCode,
};

use tune::{
  BenchRunner, OutputFormat, PlatformInfo, Report, TuneEngine, TuneResults,
  crc16::{Crc16CcittTunable, Crc16IbmTunable},
  crc24::Crc24OpenPgpTunable,
  crc32::{Crc32IeeeTunable, Crc32cTunable},
  crc64::{Crc64NvmeTunable, Crc64XzTunable},
  hash::{
    BLAKE3_TUNING_CORPUS, HASH_CORE_TUNING_CORPUS, HASH_MICRO_TUNING_CORPUS, HASH_STREAM_PROFILE_TUNING_CORPUS,
    HashTunable,
  },
};

const CRATE_CHECKSUM: &str = "checksum";
const CRATE_HASHES: &str = "hashes";
const CRATE_HASHES_BENCH: &str = "hashes-bench";
const VALID_CRATE_FILTERS: &[&str] = &[CRATE_CHECKSUM, CRATE_HASHES, CRATE_HASHES_BENCH];

const CHECKSUM_ALGOS: &[&str] = &[
  "crc16-ccitt",
  "crc16-ibm",
  "crc24-openpgp",
  "crc32-ieee",
  "crc32c",
  "crc64-xz",
  "crc64-nvme",
];

/// CLI arguments.
#[derive(Debug)]
struct Args {
  /// Quick mode (faster, noisier).
  quick: bool,

  /// Only run a subset of algorithms (repeatable, comma-separated).
  only: Vec<String>,

  /// Only run selected algorithm crates/surfaces.
  crate_filters: Vec<String>,

  /// List available algorithm crate filters and algorithms.
  list: bool,

  /// Output format.
  format: OutputFormat,

  /// Optional directory for writing report artifacts.
  report_dir: Option<String>,

  /// Verbose output.
  verbose: bool,

  /// Apply results into dispatch.rs kernel tables.
  apply: bool,

  /// Validate that tuned results can be safely applied on this host.
  self_check: bool,

  /// Enforce throughput targets and fail when any class misses.
  enforce_targets: bool,

  /// Custom warmup duration in ms.
  warmup_ms: Option<u64>,

  /// Custom measurement duration in ms.
  measure_ms: Option<u64>,

  /// Custom checksum warmup duration in ms.
  checksum_warmup_ms: Option<u64>,

  /// Custom checksum measurement duration in ms.
  checksum_measure_ms: Option<u64>,

  /// Custom hash warmup duration in ms.
  hash_warmup_ms: Option<u64>,

  /// Custom hash measurement duration in ms.
  hash_measure_ms: Option<u64>,

  /// Show help.
  help: bool,
}

impl Default for Args {
  fn default() -> Self {
    Self {
      quick: false,
      only: Vec::new(),
      crate_filters: Vec::new(),
      list: false,
      format: OutputFormat::Summary,
      report_dir: None,
      verbose: false,
      apply: false,
      self_check: false,
      enforce_targets: false,
      warmup_ms: None,
      measure_ms: None,
      checksum_warmup_ms: None,
      checksum_measure_ms: None,
      hash_warmup_ms: None,
      hash_measure_ms: None,
      help: false,
    }
  }
}

fn parse_csv_values(value: &str, out: &mut Vec<String>, lowercase: bool) {
  for part in value.split(',') {
    let mut item = part.trim();
    if let Some((_, rhs)) = item.rsplit_once('=')
      && !rhs.is_empty()
    {
      item = rhs;
    }
    if item.is_empty() {
      continue;
    }
    if lowercase {
      out.push(item.to_ascii_lowercase());
    } else {
      out.push(item.to_string());
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
      "--list" => args.list = true,
      "--apply" => args.apply = true,
      "--self-check" => args.self_check = true,
      "--enforce-targets" => args.enforce_targets = true,
      "--only" => {
        let Some(value) = iter.next() else {
          return Err("--only requires a value".to_string());
        };
        parse_csv_values(&value, &mut args.only, false);
      }
      "--crate" => {
        let Some(value) = iter.next() else {
          return Err("--crate requires a value".to_string());
        };
        parse_csv_values(&value, &mut args.crate_filters, true);
      }
      "--format" | "-f" => {
        let Some(value) = iter.next() else {
          return Err("--format requires a value".to_string());
        };
        args.format = OutputFormat::parse(&value).ok_or_else(|| format!("Unknown format: {value}"))?;
      }
      "--report-dir" => {
        let Some(value) = iter.next() else {
          return Err("--report-dir requires a value".to_string());
        };
        if value.trim().is_empty() {
          return Err("--report-dir requires a non-empty value".to_string());
        }
        args.report_dir = Some(value);
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
      "--checksum-warmup-ms" => {
        let Some(value) = iter.next() else {
          return Err("--checksum-warmup-ms requires a value".to_string());
        };
        args.checksum_warmup_ms = Some(
          value
            .parse()
            .map_err(|_| format!("Invalid checksum-warmup-ms: {value}"))?,
        );
      }
      "--checksum-measure-ms" => {
        let Some(value) = iter.next() else {
          return Err("--checksum-measure-ms requires a value".to_string());
        };
        args.checksum_measure_ms = Some(
          value
            .parse()
            .map_err(|_| format!("Invalid checksum-measure-ms: {value}"))?,
        );
      }
      "--hash-warmup-ms" => {
        let Some(value) = iter.next() else {
          return Err("--hash-warmup-ms requires a value".to_string());
        };
        args.hash_warmup_ms = Some(value.parse().map_err(|_| format!("Invalid hash-warmup-ms: {value}"))?);
      }
      "--hash-measure-ms" => {
        let Some(value) = iter.next() else {
          return Err("--hash-measure-ms requires a value".to_string());
        };
        args.hash_measure_ms = Some(value.parse().map_err(|_| format!("Invalid hash-measure-ms: {value}"))?);
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
    -q, --quick           Developer preview mode (faster, noisier; cannot be used with --apply)
    -v, --verbose         Verbose output during tuning
    --list                List registered algorithms and exit
    --crate NAME(S)       Restrict tuning corpus by crate/surface (checksum, hashes, hashes-bench)
        --only ALGO(S)    Only run selected algorithm(s). Repeatable; value may be comma-separated.
    -f, --format FORMAT   Output format: summary (default), env, json, tsv, contribute
    --report-dir DIR      Write summary/env/json/tsv/contribute artifacts into DIR
    --enforce-targets     Fail if any defined per-class throughput target is missed
    --apply               Generate dispatch.rs table entry for this TuneKind
    --self-check          Validate that tuned results can be safely applied on this host
    --warmup-ms MS        Override warmup duration for all domains
    --measure-ms MS       Override measurement duration for all domains
    --checksum-warmup-ms MS   Override checksum warmup duration (defaults: 150, quick: 75)
    --checksum-measure-ms MS  Override checksum measurement duration (defaults: 250, quick: 125)
    --hash-warmup-ms MS       Override hash warmup duration (defaults: 200, quick: 100)
    --hash-measure-ms MS      Override hash measurement duration (defaults: 400, quick: 250)
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

    # Tune only hashes crate algorithms
    just tune -- --crate hashes

    # Tune only BLAKE3 corpus and write report artifacts
    just tune-quick -- --only blake3,blake3-chunk,blake3-parent,blake3-parent-fold,blake3-stream64,blake3-stream4k \
     --report-dir target/tune

    # Enforce throughput targets in CI
    just tune -- --enforce-targets

    # Generate dispatch table entry (writes into dispatch tables)
    just tune-apply
"
  );
}

fn print_catalog() {
  println!("Available crate filters:");
  for name in VALID_CRATE_FILTERS {
    println!("  - {name}");
  }

  println!();
  println!("[{CRATE_CHECKSUM}]");
  for algo in CHECKSUM_ALGOS {
    println!("  - {algo}");
  }

  println!();
  println!("[{CRATE_HASHES}] core");
  for (algo, _) in HASH_CORE_TUNING_CORPUS {
    println!("  - {algo}");
  }
  println!("[{CRATE_HASHES}] blake3 corpus");
  for (algo, _) in BLAKE3_TUNING_CORPUS {
    println!("  - {algo}");
  }

  println!();
  println!("[{CRATE_HASHES_BENCH}] micro");
  for (algo, _) in HASH_MICRO_TUNING_CORPUS {
    println!("  - {algo}");
  }
  println!("[{CRATE_HASHES_BENCH}] stream profiles");
  for (algo, _) in HASH_STREAM_PROFILE_TUNING_CORPUS {
    println!("  - {algo}");
  }
}

fn wants_crate(crate_filters: &[String], name: &str) -> bool {
  crate_filters.is_empty() || crate_filters.iter().any(|v| v == name)
}

fn add_hash_corpus(engine: &mut TuneEngine, seen: &mut HashSet<&'static str>, corpus: &[(&'static str, &'static str)]) {
  for &(algo, env_prefix) in corpus {
    if seen.insert(algo) {
      engine.add(Box::new(HashTunable::new(algo, env_prefix)));
    }
  }
}

fn register_algorithms(engine: &mut TuneEngine, crate_filters: &[String]) {
  if wants_crate(crate_filters, CRATE_CHECKSUM) {
    engine.add(Box::new(Crc16CcittTunable::new()));
    engine.add(Box::new(Crc16IbmTunable::new()));
    engine.add(Box::new(Crc24OpenPgpTunable::new()));
    engine.add(Box::new(Crc32IeeeTunable::new()));
    engine.add(Box::new(Crc32cTunable::new()));
    engine.add(Box::new(Crc64XzTunable::new()));
    engine.add(Box::new(Crc64NvmeTunable::new()));
  }

  let mut seen = HashSet::new();
  if wants_crate(crate_filters, CRATE_HASHES) {
    add_hash_corpus(engine, &mut seen, HASH_CORE_TUNING_CORPUS);
    add_hash_corpus(engine, &mut seen, BLAKE3_TUNING_CORPUS);
  }
  if wants_crate(crate_filters, CRATE_HASHES_BENCH) {
    add_hash_corpus(engine, &mut seen, HASH_MICRO_TUNING_CORPUS);
    add_hash_corpus(engine, &mut seen, HASH_STREAM_PROFILE_TUNING_CORPUS);
  }
}

fn print_output(results: &TuneResults, format: OutputFormat) -> Result<(), String> {
  match format {
    OutputFormat::Summary => tune::report::print_summary(results).map_err(|e| format!("Failed to print summary: {e}")),
    OutputFormat::Env => tune::report::print_env(results).map_err(|e| format!("Failed to print env: {e}")),
    OutputFormat::Json => tune::report::print_json(results).map_err(|e| format!("Failed to print json: {e}")),
    OutputFormat::Tsv => tune::report::print_tsv(results).map_err(|e| format!("Failed to print tsv: {e}")),
    OutputFormat::Contribute => {
      tune::report::print_contribute(results).map_err(|e| format!("Failed to print contribute: {e}"))
    }
  }
}

fn write_report_file(path: &Path, format: OutputFormat, results: &TuneResults) -> Result<(), String> {
  let file = File::create(path).map_err(|e| format!("Failed to create {}: {e}", path.display()))?;
  let mut report = Report::new(BufWriter::new(file), format);
  report
    .write(results)
    .map_err(|e| format!("Failed to write {}: {e}", path.display()))
}

fn write_report_artifacts(path: &Path, results: &TuneResults) -> Result<(), String> {
  fs::create_dir_all(path).map_err(|e| format!("Failed to create {}: {e}", path.display()))?;

  write_report_file(&path.join("summary.txt"), OutputFormat::Summary, results)?;
  write_report_file(&path.join("env.sh"), OutputFormat::Env, results)?;
  write_report_file(&path.join("results.json"), OutputFormat::Json, results)?;
  write_report_file(&path.join("results.tsv"), OutputFormat::Tsv, results)?;
  write_report_file(&path.join("contribute.md"), OutputFormat::Contribute, results)?;

  Ok(())
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

  if args.list {
    print_catalog();
    return ExitCode::SUCCESS;
  }

  for filter in &args.crate_filters {
    if !VALID_CRATE_FILTERS.contains(&filter.as_str()) {
      eprintln!("Error: unknown --crate filter '{filter}'");
      eprintln!("Valid values: {}", VALID_CRATE_FILTERS.join(", "));
      return ExitCode::FAILURE;
    }
  }

  if args.quick && args.apply {
    eprintln!("Error: --quick is developer preview mode and cannot be used with --apply.");
    eprintln!("Run a full-quality tune (without --quick) before applying dispatch defaults.");
    return ExitCode::FAILURE;
  }

  // Build runners (checksums and hashes can use different measurement windows).
  let (mut checksum_runner, mut hash_runner) = if args.quick {
    (BenchRunner::quick(), BenchRunner::quick_hash())
  } else {
    (BenchRunner::new(), BenchRunner::hash())
  };

  // Global overrides apply to both domains.
  match (args.warmup_ms, args.measure_ms) {
    (Some(w), Some(m)) => {
      checksum_runner = checksum_runner
        .with_warmup(Duration::from_millis(w))
        .with_measure(Duration::from_millis(m));
      hash_runner = hash_runner
        .with_warmup(Duration::from_millis(w))
        .with_measure(Duration::from_millis(m));
    }
    (Some(w), None) => {
      checksum_runner = checksum_runner.with_warmup(Duration::from_millis(w));
      hash_runner = hash_runner.with_warmup(Duration::from_millis(w));
    }
    (None, Some(m)) => {
      checksum_runner = checksum_runner.with_measure(Duration::from_millis(m));
      hash_runner = hash_runner.with_measure(Duration::from_millis(m));
    }
    (None, None) => {}
  }

  // Domain-specific overrides apply after the global overrides.
  match (args.checksum_warmup_ms, args.checksum_measure_ms) {
    (Some(w), Some(m)) => {
      checksum_runner = checksum_runner
        .with_warmup(Duration::from_millis(w))
        .with_measure(Duration::from_millis(m));
    }
    (Some(w), None) => checksum_runner = checksum_runner.with_warmup(Duration::from_millis(w)),
    (None, Some(m)) => checksum_runner = checksum_runner.with_measure(Duration::from_millis(m)),
    (None, None) => {}
  }

  match (args.hash_warmup_ms, args.hash_measure_ms) {
    (Some(w), Some(m)) => {
      hash_runner = hash_runner
        .with_warmup(Duration::from_millis(w))
        .with_measure(Duration::from_millis(m));
    }
    (Some(w), None) => hash_runner = hash_runner.with_warmup(Duration::from_millis(w)),
    (None, Some(m)) => hash_runner = hash_runner.with_measure(Duration::from_millis(m)),
    (None, None) => {}
  };

  // Build and populate the engine.
  let mut engine = TuneEngine::new()
    .with_checksum_runner(checksum_runner)
    .with_hash_runner(hash_runner)
    .with_verbose(args.verbose);
  register_algorithms(&mut engine, &args.crate_filters);

  // Print header.
  let platform = PlatformInfo::collect();
  eprintln!("rscrypto-tune");
  eprintln!("=============");
  eprintln!();
  eprintln!("Platform: {}", platform.description);
  eprintln!("Tune preset: {:?}", platform.tune_kind);
  if args.quick {
    eprintln!("Mode: quick developer preview (not dispatch-eligible)");
  } else {
    eprintln!("Mode: full-quality (dispatch-eligible)");
  }
  eprintln!();

  // Run the tuning engine.
  if !args.only.is_empty() {
    let kept = engine.retain_only(&args.only);
    if kept == 0 {
      eprintln!("Error: --only did not match any registered algorithms.");
      eprintln!("Tip: run with --list to see available algorithms.");
      return ExitCode::FAILURE;
    }
  }

  let results = match engine.run() {
    Ok(results) => results,
    Err(err) => {
      eprintln!("Tuning failed: {err}");
      return ExitCode::FAILURE;
    }
  };

  if let Err(err) = print_output(&results, args.format) {
    eprintln!("{err}");
    return ExitCode::FAILURE;
  }

  if let Some(dir) = &args.report_dir {
    if let Err(err) = write_report_artifacts(Path::new(dir), &results) {
      eprintln!("{err}");
      return ExitCode::FAILURE;
    }
    eprintln!("Wrote tune report artifacts: {dir}");
  }

  if args.enforce_targets {
    let misses = tune::targets::collect_target_misses(&results);
    if !misses.is_empty() {
      eprintln!("Target check failed ({} miss(es)):", misses.len());
      for miss in misses {
        eprintln!(
          "  {} {}: {:.2} GiB/s < target {:.2} GiB/s",
          miss.algo, miss.class, miss.measured_gib_s, miss.target_gib_s
        );
      }
      return ExitCode::FAILURE;
    }
    eprintln!("Target check passed.");
  }

  if args.self_check
    && let Err(err) = tune::apply::self_check(&results)
  {
    eprintln!("Self-check failed: {err}");
    return ExitCode::FAILURE;
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
