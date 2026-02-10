//! Unified tuning binary for rscrypto algorithms.
//!
//! Usage:
//!   cargo run --release -p tune --bin rscrypto-tune
//!   cargo run --release -p tune --bin rscrypto-tune -- --quick
//!   cargo run --release -p tune --bin rscrypto-tune -- --format env

extern crate alloc;

use alloc::collections::BTreeMap;
use core::time::Duration;
use std::{
  collections::HashSet,
  env,
  fs::{self, File},
  io::BufWriter,
  path::{Path, PathBuf},
  process::ExitCode,
};

use tune::{
  AggregationMode, BenchRunner, OutputFormat, PlatformInfo, Report, TuneEngine, TuneResults, aggregate_raw_results,
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
const DEFAULT_CRATE_FILTERS: &[&str] = &[CRATE_CHECKSUM, CRATE_HASHES];
const DEFAULT_APPLY_CRATE_FILTERS: &[&str] = &[CRATE_CHECKSUM, CRATE_HASHES, CRATE_HASHES_BENCH];

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

  /// Path to write raw measurement artifact JSON.
  raw_output: Option<String>,

  /// Derive policy/results from existing raw artifact JSON.
  derive_from: Option<String>,

  /// Run only measurement phase, emit raw artifact, and exit.
  measure_only: bool,

  /// Number of repeated measurement runs before aggregation.
  repeats: Option<usize>,

  /// Aggregation mode for repeated runs.
  aggregate: AggregationMode,

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
      raw_output: None,
      derive_from: None,
      measure_only: false,
      repeats: None,
      aggregate: AggregationMode::Auto,
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
      "--measure-only" => args.measure_only = true,
      "--repeats" => {
        let Some(value) = iter.next() else {
          return Err("--repeats requires a value".to_string());
        };
        let repeats: usize = value.parse().map_err(|_| format!("Invalid repeats: {value}"))?;
        if repeats == 0 {
          return Err("--repeats must be >= 1".to_string());
        }
        args.repeats = Some(repeats);
      }
      "--aggregate" => {
        let Some(value) = iter.next() else {
          return Err("--aggregate requires a value".to_string());
        };
        args.aggregate = AggregationMode::parse(&value).ok_or_else(|| format!("Unknown aggregation mode: {value}"))?;
      }
      "--only" => {
        let Some(value) = iter.next() else {
          return Err("--only requires a value".to_string());
        };
        parse_csv_values(&value, &mut args.only, true);
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
      "--raw-output" => {
        let Some(value) = iter.next() else {
          return Err("--raw-output requires a value".to_string());
        };
        if value.trim().is_empty() {
          return Err("--raw-output requires a non-empty value".to_string());
        }
        args.raw_output = Some(value);
      }
      "--derive-from" => {
        let Some(value) = iter.next() else {
          return Err("--derive-from requires a value".to_string());
        };
        if value.trim().is_empty() {
          return Err("--derive-from requires a non-empty value".to_string());
        }
        args.derive_from = Some(value);
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
        if other.starts_with('-') {
          return Err(format!("Unknown argument: {other}"));
        }
        parse_csv_values(other, &mut args.only, true);
      }
    }
  }

  Ok(args)
}

fn normalize_selector(value: &str) -> String {
  value
    .chars()
    .filter(|ch| ch.is_ascii_alphanumeric())
    .map(|ch| ch.to_ascii_lowercase())
    .collect()
}

fn push_unique(values: &mut Vec<String>, value: &str) {
  if !values.iter().any(|current| current == value) {
    values.push(value.to_string());
  }
}

fn all_algorithms() -> Vec<&'static str> {
  let mut all = Vec::new();
  let mut seen = HashSet::new();

  for &algo in CHECKSUM_ALGOS {
    if seen.insert(algo) {
      all.push(algo);
    }
  }
  for &(algo, _) in HASH_CORE_TUNING_CORPUS {
    if seen.insert(algo) {
      all.push(algo);
    }
  }
  for &(algo, _) in BLAKE3_TUNING_CORPUS {
    if seen.insert(algo) {
      all.push(algo);
    }
  }
  for &(algo, _) in HASH_MICRO_TUNING_CORPUS {
    if seen.insert(algo) {
      all.push(algo);
    }
  }
  for &(algo, _) in HASH_STREAM_PROFILE_TUNING_CORPUS {
    if seen.insert(algo) {
      all.push(algo);
    }
  }

  all
}

fn family_key_for_algorithm(algo: &str) -> String {
  if algo.starts_with("blake3") {
    return "blake3".to_string();
  }
  if algo.starts_with("crc64-") {
    return "crc64".to_string();
  }
  if algo == "crc32c" || algo.starts_with("crc32-") {
    return "crc32".to_string();
  }
  if algo.starts_with("crc16-") {
    return "crc16".to_string();
  }
  if algo.starts_with("crc24-") {
    return "crc24".to_string();
  }

  let mut key = algo;
  for suffix in ["-compress-unaligned", "-compress", "-permute"] {
    if let Some(base) = key.strip_suffix(suffix) {
      key = base;
      break;
    }
  }
  if let Some((base, _)) = key.split_once("-stream") {
    key = base;
  }
  if let Some(base) = key.strip_suffix("-keyed") {
    key = base;
  }
  if let Some(base) = key.strip_suffix("-derive") {
    key = base;
  }
  if let Some(base) = key.strip_suffix("-xof") {
    key = base;
  }

  key.to_string()
}

fn checksum_contains(algo: &str) -> bool {
  CHECKSUM_ALGOS.contains(&algo)
}

fn hashes_contains(algo: &str) -> bool {
  HASH_CORE_TUNING_CORPUS.iter().any(|(name, _)| *name == algo)
    || BLAKE3_TUNING_CORPUS.iter().any(|(name, _)| *name == algo)
}

fn hashes_bench_contains(algo: &str) -> bool {
  HASH_MICRO_TUNING_CORPUS.iter().any(|(name, _)| *name == algo)
    || HASH_STREAM_PROFILE_TUNING_CORPUS.iter().any(|(name, _)| *name == algo)
}

fn resolve_selector(raw: &str, family_map: &BTreeMap<String, Vec<&'static str>>) -> Result<Vec<&'static str>, String> {
  let key = normalize_selector(raw);
  if key.is_empty() {
    return Ok(Vec::new());
  }

  let explicit = match key.as_str() {
    "all" => Some(all_algorithms()),
    "checksum" | "checksums" => Some(CHECKSUM_ALGOS.to_vec()),
    "hash" | "hashes" => {
      let mut out: Vec<&'static str> = HASH_CORE_TUNING_CORPUS.iter().map(|(name, _)| *name).collect();
      out.extend(BLAKE3_TUNING_CORPUS.iter().map(|(name, _)| *name));
      Some(out)
    }
    "hashesbench" | "hashbench" | "bench" => {
      let mut out: Vec<&'static str> = HASH_MICRO_TUNING_CORPUS.iter().map(|(name, _)| *name).collect();
      out.extend(HASH_STREAM_PROFILE_TUNING_CORPUS.iter().map(|(name, _)| *name));
      Some(out)
    }
    "blake3" => Some(BLAKE3_TUNING_CORPUS.iter().map(|(name, _)| *name).collect()),
    "crc64" | "crc64nvme" | "crc64xz" => Some(vec!["crc64-xz", "crc64-nvme"]),
    "crc32" => Some(vec!["crc32-ieee", "crc32c"]),
    "crc16" => Some(vec!["crc16-ccitt", "crc16-ibm"]),
    _ => None,
  };
  if let Some(expanded) = explicit {
    return Ok(expanded);
  }

  if let Some(family) = family_map.get(&key) {
    return Ok(family.clone());
  }

  for algo in all_algorithms() {
    if normalize_selector(algo) == key {
      return Ok(vec![algo]);
    }
  }

  Err(format!("Unknown algorithm selector: {raw}"))
}

fn resolve_algorithm_selection(raw_only: &[String], apply: bool) -> Result<(Vec<String>, Vec<String>), String> {
  if raw_only.is_empty() {
    return Ok((Vec::new(), Vec::new()));
  }

  let mut family_map: BTreeMap<String, Vec<&'static str>> = BTreeMap::new();
  for algo in all_algorithms() {
    family_map
      .entry(normalize_selector(family_key_for_algorithm(algo).as_str()))
      .or_default()
      .push(algo);
  }

  let mut resolved = Vec::new();
  for raw in raw_only {
    let expanded = resolve_selector(raw, &family_map)?;
    for algo in expanded {
      push_unique(&mut resolved, algo);
    }
  }

  let mut notes = Vec::new();

  if apply && resolved.iter().any(|algo| checksum_contains(algo.as_str())) {
    let mut added = false;
    for &algo in CHECKSUM_ALGOS {
      if !resolved.iter().any(|current| current == algo) {
        resolved.push(algo.to_string());
        added = true;
      }
    }
    if added {
      notes.push(
        "--apply + checksum selector requires full checksum corpus; auto-added remaining checksum algorithms"
          .to_string(),
      );
    }
  }

  if apply && resolved.iter().any(|algo| algo.starts_with("blake3")) {
    let mut added = false;
    for &(algo, _) in BLAKE3_TUNING_CORPUS {
      if !resolved.iter().any(|current| current == algo) {
        resolved.push(algo.to_string());
        added = true;
      }
    }
    if added {
      notes.push("--apply + blake3 selector requires full blake3 corpus; auto-added missing surfaces".to_string());
    }
  }

  Ok((resolved, notes))
}

fn infer_crate_filters(only: &[String], apply: bool) -> Vec<String> {
  if only.is_empty() {
    return if apply {
      DEFAULT_APPLY_CRATE_FILTERS
        .iter()
        .map(|value| (*value).to_string())
        .collect()
    } else {
      DEFAULT_CRATE_FILTERS.iter().map(|value| (*value).to_string()).collect()
    };
  }

  let mut inferred = Vec::new();
  if only.iter().any(|algo| checksum_contains(algo.as_str())) {
    inferred.push(CRATE_CHECKSUM.to_string());
  }
  if only.iter().any(|algo| hashes_contains(algo.as_str())) {
    inferred.push(CRATE_HASHES.to_string());
  }
  if only.iter().any(|algo| hashes_bench_contains(algo.as_str())) {
    inferred.push(CRATE_HASHES_BENCH.to_string());
  }

  if inferred.is_empty() {
    DEFAULT_CRATE_FILTERS.iter().map(|value| (*value).to_string()).collect()
  } else {
    inferred
  }
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
    ALGO(S)               Optional positional selector(s), e.g. `blake3` or `crc64nvme`
    --crate NAME(S)       Restrict tuning corpus by crate/surface (checksum, hashes, hashes-bench)
    --only ALGO(S)        Only run selected algorithm(s). Repeatable; value may be comma-separated.
    -f, --format FORMAT   Output format: summary (default), env, json, tsv, contribute
    --report-dir DIR      Write summary/env/json/tsv/contribute artifacts into DIR
    --raw-output PATH     Write raw measurement artifact to PATH (default: <report-dir>/raw-results.json or \
     target/tune/raw-results.json)
    --derive-from PATH    Skip measurement and derive policy/results from PATH
    --measure-only        Run measurement only, emit raw artifact, skip derivation/apply
    --repeats N           Repeat measurement N times before derivation (default: 1)
    --aggregate MODE      Repeated-run aggregation mode: auto (default), median, trimmed-mean
    --enforce-targets     Fail if any defined per-class throughput target is missed
    --apply               Generate dispatch.rs table entry for this TuneKind
    --self-check          Validate that tuned results can be safely applied on this host
    --warmup-ms MS        Override warmup duration for all domains
    --measure-ms MS       Override measurement duration for all domains
    --checksum-warmup-ms MS   Override checksum warmup duration (defaults: 150, quick: 75)
    --checksum-measure-ms MS  Override checksum measurement duration (defaults: 250, quick: 125)
    --hash-warmup-ms MS       Override hash warmup duration (defaults: 100, quick: 50)
    --hash-measure-ms MS      Override hash measurement duration (defaults: 180, quick: 100)
    -h, --help            Show this help message

FORMATS:
    summary     Human-readable summary (default)
    env         Shell environment variable exports
    json        JSON for programmatic use
    tsv         Tab-separated values
    contribute  Markdown ready for GitHub issue submission

EXAMPLES:
    # Standard end-to-end run (measure -> derive)
    just tune

    # Selector-first run: all CRC64 variants, apply results
    just tune crc64nvme --apply

    # Selector-first run: full BLAKE3 corpus
    just tune blake3 --apply

    # Measure once, derive later
    just tune-measure
    just tune-derive raw=target/tune/raw-results.json

    # Explicit crate scope when needed
    just tune -- --crate hashes-bench

    # Write report artifacts to a specific directory
    just tune blake3 --report-dir target/tune

    # Derive from existing raw artifact without rerunning benches
    just tune -- --derive-from target/tune/raw-results.json --report-dir target/tune

    # Increase stability with more repeated runs
    just tune -- --repeats 5 --aggregate trimmed-mean

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

  println!();
  println!("Common selector families:");
  println!("  - blake3 (all blake3 surfaces)");
  println!("  - crc64 / crc64nvme / crc64xz (crc64-xz + crc64-nvme)");
  println!("  - crc32 (crc32-ieee + crc32c)");
  println!("  - crc16 (crc16-ccitt + crc16-ibm)");
  println!("  - checksum, hashes, hashes-bench");
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

fn resolve_raw_output_path(args: &Args) -> PathBuf {
  if let Some(path) = &args.raw_output {
    return PathBuf::from(path);
  }
  if let Some(dir) = &args.report_dir {
    return Path::new(dir).join("raw-results.json");
  }
  PathBuf::from("target/tune/raw-results.json")
}

fn has_measurement_overrides(args: &Args) -> bool {
  args.quick
    || args.repeats.is_some()
    || args.aggregate != AggregationMode::Auto
    || !args.only.is_empty()
    || !args.crate_filters.is_empty()
    || args.warmup_ms.is_some()
    || args.measure_ms.is_some()
    || args.checksum_warmup_ms.is_some()
    || args.checksum_measure_ms.is_some()
    || args.hash_warmup_ms.is_some()
    || args.hash_measure_ms.is_some()
}

fn effective_repeats(args: &Args) -> usize {
  args.repeats.unwrap_or(1)
}

fn main() -> ExitCode {
  let mut args = match parse_args() {
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

  let explicit_crate_filters = !args.crate_filters.is_empty();
  let raw_selectors = args.only.clone();

  for filter in &args.crate_filters {
    if !VALID_CRATE_FILTERS.contains(&filter.as_str()) {
      eprintln!("Error: unknown --crate filter '{filter}'");
      eprintln!("Valid values: {}", VALID_CRATE_FILTERS.join(", "));
      return ExitCode::FAILURE;
    }
  }

  if args.measure_only && args.derive_from.is_some() {
    eprintln!("Error: --measure-only and --derive-from are mutually exclusive.");
    return ExitCode::FAILURE;
  }

  if args.measure_only && (args.apply || args.self_check || args.enforce_targets) {
    eprintln!("Error: --measure-only cannot be combined with --apply/--self-check/--enforce-targets.");
    return ExitCode::FAILURE;
  }

  if args.derive_from.is_some() && has_measurement_overrides(&args) {
    eprintln!(
      "Error: --derive-from cannot be combined with measurement options \
       (--quick/--repeats/--crate/--only/--aggregate/measurement overrides)."
    );
    return ExitCode::FAILURE;
  }

  if args.derive_from.is_none() {
    if !args.only.is_empty() {
      let (resolved, notes) = match resolve_algorithm_selection(&args.only, args.apply) {
        Ok(value) => value,
        Err(err) => {
          eprintln!("Error: {err}");
          eprintln!("Tip: run with --list to see available selectors.");
          return ExitCode::FAILURE;
        }
      };
      args.only = resolved;
      if args.only != raw_selectors {
        eprintln!("Algorithm selection expanded to: {}", args.only.join(", "));
      }
      for note in notes {
        eprintln!("note: {note}");
      }
    }

    if !explicit_crate_filters {
      args.crate_filters = infer_crate_filters(&args.only, args.apply);
    }
  }

  let raw_output_path = resolve_raw_output_path(&args);

  let results = if let Some(raw_input) = &args.derive_from {
    let raw = match tune::read_raw_results(Path::new(raw_input)) {
      Ok(raw) => raw,
      Err(err) => {
        eprintln!("Failed to read raw artifact {}: {err}", raw_input);
        return ExitCode::FAILURE;
      }
    };
    if raw.quick_mode && args.apply {
      eprintln!("Error: raw artifact was measured in quick mode and cannot be used with --apply.");
      eprintln!("Run a full-quality measurement (without --quick) before applying dispatch defaults.");
      return ExitCode::FAILURE;
    }
    match TuneEngine::derive_from_raw(&raw) {
      Ok(results) => results,
      Err(err) => {
        eprintln!("Derivation failed: {err}");
        return ExitCode::FAILURE;
      }
    }
  } else {
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
    if !args.crate_filters.is_empty() {
      eprintln!("Crates: {}", args.crate_filters.join(", "));
    }
    if !args.only.is_empty() {
      eprintln!("Algorithms: {}", args.only.join(", "));
    }
    let repeats = effective_repeats(&args);
    eprintln!("Repeats: {repeats}");
    eprintln!("Aggregation: {}", args.aggregate.as_str());
    eprintln!();

    if !args.only.is_empty() {
      let kept = engine.retain_only(&args.only);
      if kept == 0 {
        eprintln!("Error: --only did not match any registered algorithms.");
        eprintln!("Tip: run with --list to see available algorithms.");
        return ExitCode::FAILURE;
      }
    }

    let repeats = effective_repeats(&args);
    let mut runs = Vec::with_capacity(repeats);
    for run_idx in 0..repeats {
      if repeats > 1 {
        eprintln!("Measurement run {}/{}...", run_idx + 1, repeats);
      }
      let raw = match engine.measure(args.quick) {
        Ok(raw) => raw,
        Err(err) => {
          eprintln!("Measurement failed on run {}: {err}", run_idx + 1);
          return ExitCode::FAILURE;
        }
      };
      runs.push(raw);
    }

    let raw = if repeats == 1 {
      runs.remove(0)
    } else {
      match aggregate_raw_results(&runs, args.aggregate) {
        Ok(aggregated) => {
          eprintln!(
            "Aggregated {} measurement runs with mode {}.",
            repeats,
            args.aggregate.as_str()
          );
          aggregated
        }
        Err(err) => {
          eprintln!("Failed to aggregate repeated measurements: {err}");
          return ExitCode::FAILURE;
        }
      }
    };

    if let Some(parent) = raw_output_path.parent()
      && let Err(err) = fs::create_dir_all(parent)
    {
      eprintln!("Failed to create {}: {err}", parent.display());
      return ExitCode::FAILURE;
    }
    if let Err(err) = tune::write_raw_results(raw_output_path.as_path(), &raw) {
      eprintln!("Failed to write raw artifact {}: {err}", raw_output_path.display());
      return ExitCode::FAILURE;
    }
    eprintln!("Wrote raw measurement artifact: {}", raw_output_path.display());

    if args.measure_only {
      return ExitCode::SUCCESS;
    }

    match TuneEngine::derive_from_raw(&raw) {
      Ok(results) => results,
      Err(err) => {
        eprintln!("Derivation failed: {err}");
        return ExitCode::FAILURE;
      }
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn selector_blake3_expands_full_family() {
    let (resolved, notes) =
      resolve_algorithm_selection(&["blake3".to_string()], false).expect("selector should resolve");
    assert!(notes.is_empty());
    assert_eq!(resolved.len(), BLAKE3_TUNING_CORPUS.len());
    assert!(resolved.iter().any(|name| name == "blake3"));
    assert!(resolved.iter().any(|name| name == "blake3-stream4k-xof"));
  }

  #[test]
  fn selector_crc64nvme_expands_crc64_pair() {
    let (resolved, _notes) =
      resolve_algorithm_selection(&["crc64nvme".to_string()], false).expect("selector should resolve");
    assert_eq!(resolved, vec!["crc64-xz".to_string(), "crc64-nvme".to_string()]);
  }

  #[test]
  fn apply_with_checksum_selector_expands_full_checksum_corpus() {
    let (resolved, notes) =
      resolve_algorithm_selection(&["crc64nvme".to_string()], true).expect("selector should resolve");
    assert!(resolved.iter().any(|name| name == "crc64-nvme"));
    assert!(resolved.iter().any(|name| name == "crc32c"));
    assert_eq!(resolved.len(), CHECKSUM_ALGOS.len());
    assert!(notes.iter().any(|note| note.contains("full checksum corpus")));
  }

  #[test]
  fn infer_crates_from_blake3_is_hashes_only() {
    let crates = infer_crate_filters(&["blake3".to_string()], false);
    assert_eq!(crates, vec![CRATE_HASHES.to_string()]);
  }

  #[test]
  fn default_repeats_is_one() {
    let args = Args::default();
    assert_eq!(effective_repeats(&args), 1);
  }
}
