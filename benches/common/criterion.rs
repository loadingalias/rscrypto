//! Shared defaults for every Criterion harness, including direct Cargo invocations.

use core::time::Duration;

fn criterion() -> criterion::Criterion {
  let config: serde_json::Value =
    serde_json::from_str(include_str!("../../.config/criterion.json")).expect("valid shared Criterion configuration");
  let integer = |name: &str| config[name].as_u64().expect("integer Criterion setting");
  let fraction = |name: &str| config[name].as_f64().expect("fractional Criterion setting");
  let count = |name: &str| usize::try_from(integer(name)).expect("Criterion count fits usize");
  let budget = integer("max_run_seconds");
  assert!(
    (1..=3600).contains(&budget),
    "benchmark budget must be at most one hour"
  );
  // The orchestration watchdog bounds the entire multi-binary run. This also
  // bounds an individual harness invoked directly through Cargo.
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

pub(crate) fn run(targets: &[fn(&mut criterion::Criterion)]) {
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
}
