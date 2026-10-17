//! Shared defaults for every Criterion harness, including direct Cargo invocations.
//!
//! Every measured row registers its routine through [`checked`], [`checked_with`],
//! or [`checked_routine`]. Each checks the row's output once, outside timing,
//! when Criterion first runs the row; listing runs no check. [`run`] fails a
//! planned run in which a measured row skipped its check.

use core::{
  sync::atomic::{AtomicUsize, Ordering},
  time::Duration,
};

use criterion::Bencher;

/// Rows of this process whose output passed its check.
static CHECKED_ROWS: AtomicUsize = AtomicUsize::new(0);

/// Returns a routine that times `operation` after `check` accepts its output.
///
/// The check receives the second output, so any state that the operation
/// reuses has already served one call.
pub(crate) fn checked<T>(mut operation: impl FnMut() -> T, check: impl FnOnce(T)) -> impl FnMut(&mut Bencher<'_>) {
  checked_with((), move |_| operation(), move |output, _| check(output))
}

/// [`checked`] for an operation that writes into state the row owns, such as a
/// restored buffer. The check also receives that state.
pub(crate) fn checked_with<S, T>(
  state: S,
  operation: impl FnMut(&mut S) -> T,
  check: impl FnOnce(T, &S),
) -> impl FnMut(&mut Bencher<'_>) {
  checked_routine(
    (state, operation),
    |(state, operation)| {
      drop(operation(&mut *state));
      let output = operation(&mut *state);
      check(output, state);
    },
    |b, (state, operation)| b.iter(|| operation(&mut *state)),
  )
}

/// Wraps a routine whose timing loop [`checked`] cannot express, such as
/// `iter_batched`. `check` must run the routine's production operation; both
/// share `state`.
pub(crate) fn checked_routine<S>(
  mut state: S,
  check: impl FnOnce(&mut S),
  mut routine: impl FnMut(&mut Bencher<'_>, &mut S),
) -> impl FnMut(&mut Bencher<'_>) {
  let mut check = Some(check);
  move |b| {
    if let Some(check) = check.take() {
      check(&mut state);
      CHECKED_ROWS.fetch_add(1, Ordering::Relaxed);
    }
    routine(b, &mut state);
  }
}

fn criterion() -> criterion::Criterion {
  let config: serde_json::Value =
    serde_json::from_str(include_str!("../../.config/criterion.json")).expect("valid shared Criterion configuration");
  let integer = |name: &str| config[name].as_u64().expect("integer Criterion setting");
  let fraction = |name: &str| config[name].as_f64().expect("fractional Criterion setting");
  let count = |name: &str| usize::try_from(integer(name)).expect("Criterion count fits usize");
  let budget = integer("max_run_seconds");
  assert!(
    (1..=600).contains(&budget),
    "benchmark budget must be at most 10 minutes"
  );
  // The orchestration watchdog bounds the entire multi-binary run. This also
  // bounds an individual harness invoked directly through Cargo.
  // WASI has no thread spawn; its engine process needs the external watchdog.
  #[cfg(not(target_arch = "wasm32"))]
  std::thread::spawn(move || {
    std::thread::sleep(Duration::from_secs(budget));
    eprintln!("benchmark harness exhausted its {budget}s run budget");
    std::process::exit(124);
  });
  criterion::Criterion::default()
    .sample_size(count("sample_size"))
    .warm_up_time(Duration::from_millis(integer("warmup_ms")))
    .measurement_time(Duration::from_millis(integer("measure_ms")))
    .nresamples(count("nresamples"))
    .confidence_level(fraction("confidence_level"))
    .significance_level(fraction("significance_level"))
    .noise_threshold(fraction("noise_threshold"))
}

fn cases() -> Option<&'static [String]> {
  static CASES: std::sync::OnceLock<Option<Vec<String>>> = std::sync::OnceLock::new();
  CASES
    .get_or_init(|| {
      std::env::var_os("RSCRYPTO_BENCH_CASES").map(|path| {
        let data = std::fs::read(path).expect("read resolved benchmark cases");
        let cases: Vec<String> = serde_json::from_slice(&data).expect("resolved benchmark case list");
        assert!(!cases.is_empty(), "resolved case list must not be empty");
        cases
      })
    })
    .as_deref()
}

pub(crate) fn selected(prefix: &str) -> bool {
  cases().is_none_or(|cases| cases.iter().any(|case| case.starts_with(prefix)))
}

fn match_cases(path: &std::ffi::OsStr) {
  let request: serde_json::Value = serde_json::from_slice(&std::fs::read(path).expect("read case matching request"))
    .expect("valid case matching request");
  let cases = request["cases"].as_array().expect("discovered case array");
  let mut matches = serde_json::Map::new();
  for pattern in request["patterns"].as_array().expect("pattern array") {
    let pattern = pattern.as_str().expect("pattern string");
    // Criterion's public filter type keeps matching on its exact regex engine.
    let filter = criterion::BenchmarkFilter::Regex(pattern.parse().expect("valid Criterion regex"));
    if let criterion::BenchmarkFilter::Regex(regex) = filter {
      let selected = cases
        .iter()
        .filter(|case| regex.is_match(case.as_str().expect("case string")))
        .cloned()
        .collect();
      matches.insert(pattern.to_owned(), serde_json::Value::Array(selected));
    }
  }
  println!("{}", serde_json::Value::Object(matches));
}

/// CPU features that this binary was compiled to assume but this host lacks.
///
/// A catalog configuration can compile every crate for a newer instruction-set
/// level, such as `xxh3_v4`. Running that binary on an older CPU would fault
/// partway through a run, so the harness refuses to start instead.
fn missing_compiled_features() -> Vec<&'static str> {
  #[cfg(target_arch = "x86_64")]
  {
    macro_rules! missing {
      ($($feature:tt),+ $(,)?) => {
        [$((cfg!(target_feature = $feature) && !std::arch::is_x86_feature_detected!($feature)).then_some($feature)),+]
      };
    }
    missing!(
      "cmpxchg16b",
      "popcnt",
      "sse3",
      "ssse3",
      "sse4.1",
      "sse4.2",
      "avx",
      "avx2",
      "bmi1",
      "bmi2",
      "f16c",
      "fma",
      "lzcnt",
      "movbe",
      "xsave",
      "avx512f",
      "avx512bw",
      "avx512cd",
      "avx512dq",
      "avx512vl",
      "aes",
      "pclmulqdq",
      "sha",
      "adx",
      "gfni",
      "vaes",
      "vpclmulqdq",
      "avx512ifma",
      "avx512vbmi",
      "avx512vbmi2",
      "avx512vnni",
    )
    .into_iter()
    .flatten()
    .collect()
  }
  #[cfg(not(target_arch = "x86_64"))]
  Vec::new()
}

pub(crate) fn run(targets: &[fn(&mut criterion::Criterion)]) {
  let missing = missing_compiled_features();
  assert!(
    missing.is_empty(),
    "this benchmark binary was compiled for CPU features this host lacks: {}",
    missing.join(", ")
  );
  let mut args = std::env::args_os().skip(1);
  if args.next().is_some_and(|arg| arg == "--rscrypto-filter-cases") {
    match_cases(&args.next().expect("case matching request path"));
    return;
  }
  let mut criterion = criterion().configure_from_args();
  if let Some(cases) = cases() {
    let mut filter = String::from("^(?:");
    for (index, case) in cases.iter().enumerate() {
      if index != 0 {
        filter.push('|');
      }
      for ch in case.chars() {
        if ".^$*+?{}[]\\|()".contains(ch) {
          filter.push('\\');
        }
        filter.push(ch);
      }
    }
    filter.push_str(")$");
    criterion = criterion.with_filter(filter);
  }
  // Every target can cheaply skip unrelated fixture construction before registration.
  for target in targets {
    target(&mut criterion);
  }
  criterion.final_summary();
  let checked = CHECKED_ROWS.load(Ordering::Relaxed);
  if let Some(cases) = cases() {
    // The case filter is exact, so Criterion measured each listed case once.
    println!("Output checks: {checked} of {} measured rows passed", cases.len());
    assert_eq!(checked, cases.len(), "every measured row must check its output");
  } else if checked != 0 {
    println!("Output checks: {checked} rows passed");
  }
}
