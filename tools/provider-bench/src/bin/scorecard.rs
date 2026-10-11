//! Resource and individual-latency observations for the catalog's TLS workload.

extern crate alloc;

use alloc::sync::Arc;
use core::time::Duration;
use rscrypto_provider_bench::{BACKENDS, CREDENTIALS, Fixture, GROUPS, Profile, SUITES};
use serde_json::json;
use std::{
  sync::Barrier,
  time::Instant,
};

#[cfg(unix)]
fn usage() -> (u128, u64) {
  let mut value = core::mem::MaybeUninit::<libc::rusage>::uninit();
  // SAFETY: getrusage writes one rusage into a valid, aligned output allocation.
  // Only a successful return permits reading its initialized fields.
  let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, value.as_mut_ptr()) };
  assert_eq!(status, 0, "process resource counters");
  // SAFETY: the successful getrusage call initialized the output.
  let value = unsafe { value.assume_init() };
  let nanos = |time: libc::timeval| {
    u128::try_from(time.tv_sec)
      .expect("nonnegative CPU seconds")
      .strict_mul(1_000_000_000)
      .strict_add(
        u128::try_from(time.tv_usec)
          .expect("nonnegative CPU microseconds")
          .strict_mul(1000),
      )
  };
  let peak = u64::try_from(value.ru_maxrss).expect("nonnegative peak RSS");
  #[cfg(not(target_os = "macos"))]
  let peak = peak.strict_mul(1024);
  (nanos(value.ru_utime).strict_add(nanos(value.ru_stime)), peak)
}

fn main() {
  let args = std::env::args().skip(1).collect::<Vec<_>>();
  assert_eq!(
    args.len(),
    7,
    "usage: scorecard BACKEND CREDENTIAL GROUP SUITE full|resumed ITERATIONS WORKERS"
  );
  assert!(BACKENDS.contains(&args[0].as_str()));
  assert!(CREDENTIALS.contains(&args[1].as_str()));
  assert!(GROUPS.contains(&args[2].as_str()));
  assert!(SUITES.contains(&args[3].as_str()));
  let resumed = match args[4].as_str() {
    "full" => false,
    "resumed" => true,
    _ => {
      eprintln!("mode must be full or resumed");
      std::process::exit(2);
    }
  };
  let iterations: usize = args[5].parse().expect("iteration count");
  let workers: usize = args[6].parse().expect("worker count");
  assert!(
    (1..=10_000).contains(&iterations) && (1..=16).contains(&workers),
    "bounded scorecard"
  );
  std::thread::spawn(|| {
    std::thread::sleep(Duration::from_secs(600));
    std::process::exit(124);
  });
  let ready = Arc::new(Barrier::new(workers.strict_add(1)));
  let start = Arc::new(Barrier::new(workers.strict_add(1)));
  let mut jobs = Vec::new();
  for _ in 0..workers {
    let args = args.clone();
    let ready = ready.clone();
    let start = start.clone();
    jobs.push(std::thread::spawn(move || {
      let fixture = Fixture::new(
        &args[0],
        Profile {
          credential: &args[1],
          group: &args[2],
          suite: &args[3],
          resumed,
        },
      );
      let pair = fixture.connect();
      pair.verify();
      let wire = pair.wire_report();
      drop(pair);
      let mut samples = Vec::with_capacity(iterations);
      ready.wait();
      start.wait();
      for _ in 0..iterations {
        let began = Instant::now();
        drop(core::hint::black_box(fixture.connect()));
        samples.push(u64::try_from(began.elapsed().as_nanos()).expect("bounded latency"));
      }
      (samples, wire)
    }));
  }
  ready.wait();
  #[cfg(unix)]
  let (cpu_before, rss_before) = usage();
  let began = Instant::now();
  start.wait();
  let rows = jobs
    .into_iter()
    .map(|job| job.join().expect("scorecard worker"))
    .collect::<Vec<_>>();
  let elapsed = began.elapsed();
  #[cfg(unix)]
  let (cpu_after, rss_after) = usage();
  let samples = rows.iter().flat_map(|row| row.0.iter().copied()).collect::<Vec<_>>();
  let mut sorted = samples.clone();
  sorted.sort_unstable();
  let percentile = |percent: usize| sorted[sorted.len().strict_sub(1).strict_mul(percent).strict_div(100)];
  let mut report = json!({
    "backend": args[0], "credential": args[1], "group": args[2], "suite": args[3], "mode": args[4],
    "workers": workers, "connections": samples.len(), "elapsed_ns": elapsed.as_nanos(),
    "latency_ns": {"p50": percentile(50), "p95": percentile(95), "p99": percentile(99), "samples": samples},
    "wire": rows.iter().map(|row| &row.1).collect::<Vec<_>>(),
    "boundaries": "reused keys/configuration per worker; mutual authentication; connection construction, TLS, transport copies, ticket processing, and destruction timed; no network RTT; empirical percentiles",
    "memory_boundary": "whole-process high-water RSS, including TLS libraries, harness, worker stacks, and key/configuration owners; not allocated heap or a per-connection bound",
    "openssl": openssl::version::version(),
    "openssl_compiler": openssl::version::c_flags(),
    "openssl_platform": openssl::version::platform(),
  });
  #[cfg(unix)]
  {
    report["cpu_ns"] = json!(cpu_after.strict_sub(cpu_before));
    report["peak_rss_bytes"] = json!(rss_after);
    report["warmup_peak_rss_bytes"] = json!(rss_before);
  }
  #[cfg(not(unix))]
  {
    report["resource_counters"] = json!("unavailable on this host; latency and wire observations only");
  }
  println!("{}", serde_json::to_string_pretty(&report).expect("scorecard JSON"));
}
