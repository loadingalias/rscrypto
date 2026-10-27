//! Live allocation observations for the production provider and TLS fixture.

extern crate alloc;

use alloc::sync::Arc;
use core::{
  alloc::{GlobalAlloc, Layout},
  sync::atomic::{AtomicUsize, Ordering},
};
use rscrypto_provider_bench::{CREDENTIALS, Fixture, GROUPS, Profile, RSA_CREDENTIALS, SUITES, Workload};
use serde_json::json;
use std::{
  alloc::System,
  sync::{Barrier, Mutex, mpsc},
};

struct ObservedSystem;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static BLOCKS: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: ObservedSystem = ObservedSystem;

/// System's successful allocations are malloc-compatible on these native hosts.
///
/// # Safety
///
/// `pointer` must be non-null and identify a live allocation returned by System.
/// It must not be freed or reallocated during this query.
unsafe fn usable(pointer: *mut u8) -> usize {
  #[cfg(target_os = "macos")]
  {
    // SAFETY: the caller supplies a live, non-null System allocation.
    unsafe { libc::malloc_size(pointer.cast()) }
  }
  #[cfg(target_os = "linux")]
  {
    // SAFETY: the caller supplies a live, non-null System allocation.
    unsafe { libc::malloc_usable_size(pointer.cast()) }
  }
}

fn added(bytes: usize) {
  let live = LIVE.fetch_add(bytes, Ordering::SeqCst).strict_add(bytes);
  PEAK.fetch_max(live, Ordering::SeqCst);
  CALLS.fetch_add(1, Ordering::SeqCst);
  BYTES.fetch_add(bytes, Ordering::SeqCst);
}

// SAFETY: every allocation/deallocation and its Layout are forwarded to System.
// Counters contain no allocation, locking, user callbacks, or pointer ownership.
// A failed allocation leaves the counters and the default realloc allocation intact.
unsafe impl GlobalAlloc for ObservedSystem {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    // SAFETY: GlobalAlloc's caller supplies a valid nonzero Layout.
    let pointer = unsafe { System.alloc(layout) };
    if !pointer.is_null() {
      // SAFETY: System just returned this live allocation.
      added(unsafe { usable(pointer) });
      BLOCKS.fetch_add(1, Ordering::SeqCst);
    }
    pointer
  }

  unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
    // SAFETY: GlobalAlloc's caller supplies a valid nonzero Layout.
    let pointer = unsafe { System.alloc_zeroed(layout) };
    if !pointer.is_null() {
      // SAFETY: System just returned this live allocation.
      added(unsafe { usable(pointer) });
      BLOCKS.fetch_add(1, Ordering::SeqCst);
    }
    pointer
  }

  unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
    // SAFETY: GlobalAlloc requires a live pointer from this allocator.
    let bytes = unsafe { usable(pointer) };
    LIVE.fetch_sub(bytes, Ordering::SeqCst);
    BLOCKS.fetch_sub(1, Ordering::SeqCst);
    // SAFETY: the pointer and its original Layout are forwarded unchanged.
    unsafe { System.dealloc(pointer, layout) };
  }

  // Use GlobalAlloc's default allocate/copy/deallocate reallocation. Both
  // payloads then pass through these counters while simultaneously live. This
  // conservatively forgoes System's possible in-place reuse and avoids hiding
  // a moving realloc's transient allocation inside System::realloc.
}

#[derive(Clone, Copy)]
struct Interval {
  live: usize,
  blocks: usize,
  calls: usize,
  bytes: usize,
}

impl Interval {
  fn begin() -> Self {
    let live = LIVE.load(Ordering::SeqCst);
    PEAK.store(live, Ordering::SeqCst);
    Self {
      live,
      blocks: BLOCKS.load(Ordering::SeqCst),
      calls: CALLS.load(Ordering::SeqCst),
      bytes: BYTES.load(Ordering::SeqCst),
    }
  }

  fn end(self) -> serde_json::Value {
    // Snapshot before allocating the JSON report itself.
    let live = LIVE.load(Ordering::SeqCst);
    let peak = PEAK.load(Ordering::SeqCst);
    let blocks = BLOCKS.load(Ordering::SeqCst);
    let calls = CALLS.load(Ordering::SeqCst);
    let bytes = BYTES.load(Ordering::SeqCst);
    json!({
      "baseline_live_bytes": self.live,
      "peak_increment_bytes": peak.strict_sub(self.live),
      "retained_increment_bytes": i128::try_from(live).expect("live bytes fit the signed report").strict_sub(i128::try_from(self.live).expect("baseline bytes fit the signed report")),
      "retained_increment_blocks": i128::try_from(blocks).expect("live allocation count fits the signed report").strict_sub(i128::try_from(self.blocks).expect("baseline allocation count fits the signed report")),
      "allocation_and_reallocation_calls": calls.strict_sub(self.calls),
      "allocated_usable_bytes": bytes.strict_sub(self.bytes),
    })
  }
}

const STACK_BYTES: usize = 8 * 1024 * 1024;
const CONNECTION_BYTES: u64 = 1024 * 1024;
const CONFIGURATION_BYTES: u64 = 8 * 1024 * 1024;

fn invalid_argument(message: &str) -> ! {
  eprintln!("{message}");
  std::process::exit(2);
}

fn main() {
  let args = std::env::args().skip(1).collect::<Vec<_>>();
  assert_eq!(
    args.len(),
    7,
    "resources CREDENTIAL GROUP SUITE full|resumed mutual|intermediate|compact|stateful WORKERS REPETITIONS"
  );
  assert!(CREDENTIALS.contains(&args[0].as_str()) || RSA_CREDENTIALS.contains(&args[0].as_str()));
  assert!(GROUPS.contains(&args[1].as_str()));
  assert!(SUITES.contains(&args[2].as_str()));
  let resumed = match args[3].as_str() {
    "full" => false,
    "resumed" => true,
    _ => invalid_argument("unknown connection kind"),
  };
  let workload = match args[4].as_str() {
    "mutual" => Workload::Mutual,
    "intermediate" => Workload::Intermediate,
    "compact" => Workload::CompactKey,
    "stateful" => Workload::Stateful,
    _ => invalid_argument("unknown workload"),
  };
  let workers: usize = args[5].parse().expect("worker count");
  let repetitions: usize = args[6].parse().expect("repetition count");
  assert!((1..=64).contains(&workers) && (1..=64).contains(&repetitions));
  let profile = Profile {
    credential: &args[0],
    group: &args[1],
    suite: &args[2],
    resumed,
  };

  let construction = Interval::begin();
  let fixture = Arc::new(Fixture::with_workload("rscrypto", profile, workload));
  let construction = construction.end();

  // Pair heap includes both TLS endpoints and the fixture's bounded transport.
  // It conservatively bounds either individual endpoint; never divide it by two.
  let single = Interval::begin();
  for _ in 0..repetitions {
    let mut pair = fixture.checked_connect();
    let input = vec![37; 16_384];
    let mut output = vec![0; input.len()];
    pair.transfer(&input, &mut output);
    assert_eq!(input, output);
    drop(pair);
  }
  let single = single.end();

  let ready = Arc::new(Barrier::new(workers.strict_add(1)));
  let start = Arc::new(Barrier::new(workers.strict_add(1)));
  let hold = Arc::new(Barrier::new(workers.strict_add(1)));
  let release = Arc::new(Barrier::new(workers.strict_add(1)));
  // The caller owns admission. No TLS connection is constructed until its
  // worker takes one permit from this bounded standard-library channel.
  let (return_permit, permits) = mpsc::sync_channel(workers);
  for _ in 0..workers {
    return_permit.send(()).expect("initialize admission permits");
  }
  let permits = Arc::new(Mutex::new(permits));
  let mut jobs = Vec::with_capacity(workers);
  let simultaneous = Interval::begin();
  for _ in 0..workers {
    let fixture = fixture.clone();
    let permits = permits.clone();
    let return_permit = return_permit.clone();
    let (ready, start, hold, release) = (ready.clone(), start.clone(), hold.clone(), release.clone());
    jobs.push(
      std::thread::Builder::new()
        .stack_size(STACK_BYTES)
        .spawn(move || {
          permits
            .lock()
            .expect("admission lock")
            .try_recv()
            .expect("admitted worker permit");
          let fixture = if resumed {
            Arc::new(fixture.fork_with_resumption_cache())
          } else {
            fixture
          };
          let pending = fixture.begin();
          ready.wait();
          start.wait();
          let pair = fixture.finish(pending);
          hold.wait();
          release.wait();
          drop(pair);
          drop(fixture);
          return_permit
            .try_send(())
            .expect("return admission permit after connection destruction");
        })
        .expect("resource worker creation"),
    );
  }
  ready.wait();
  assert!(
    matches!(
      permits.lock().expect("admission lock").try_recv(),
      Err(mpsc::TryRecvError::Empty)
    ),
    "reject overload before constructing another connection"
  );
  let pending_live = LIVE.load(Ordering::SeqCst).strict_sub(simultaneous.live);
  start.wait();
  hold.wait();
  let simultaneous = simultaneous.end();
  release.wait();
  for job in jobs {
    job.join().expect("resource worker");
  }
  for _ in 0..workers {
    permits
      .lock()
      .expect("admission lock")
      .try_recv()
      .expect("destroyed connection returns its permit");
  }
  assert!(matches!(
    permits.lock().expect("admission lock").try_recv(),
    Err(mpsc::TryRecvError::Empty)
  ));

  let construction_peak = construction["peak_increment_bytes"]
    .as_u64()
    .expect("construction peak is unsigned bytes");
  let single_peak = single["peak_increment_bytes"]
    .as_u64()
    .expect("single-pair peak is unsigned bytes");
  let simultaneous_peak = simultaneous["peak_increment_bytes"]
    .as_u64()
    .expect("concurrent peak is unsigned bytes");
  let passed = construction_peak <= CONFIGURATION_BYTES
    && single_peak <= CONNECTION_BYTES
    && simultaneous_peak <= CONNECTION_BYTES.strict_mul(u64::try_from(workers).expect("bounded worker count fits u64"));
  let report = json!({
    "credential": args[0], "group": args[1], "suite": args[2], "mode": args[3], "workload": args[4],
    "workers": workers, "single_pair_repetitions": repetitions,
    "admission": {"capacity": workers, "overload_rejected_before_TLS_construction": true, "permits_returned_after_destruction": workers, "owner": "caller harness bounded std channel; no provider-global connection limit"},
    "configuration_and_preparation": construction,
    "single_pair": single,
    "simultaneous_pairs": simultaneous,
    "all_pending_pairs_live_bytes": pending_live,
    "stack_bytes_requested_per_worker": STACK_BYTES,
    "allocation_caps_passed": passed,
    "caps": {"single_pair_bytes": CONNECTION_BYTES, "configuration_and_preparation_bytes": CONFIGURATION_BYTES, "simultaneous_pairs_bytes": CONNECTION_BYTES.strict_mul(u64::try_from(workers).expect("bounded worker count fits u64"))},
    "memory_boundary": "System allocator usable payload bytes for all Rust allocations, with GlobalAlloc default moving reallocation so both old and new payloads are counted; in-place production reuse may use less memory, including both production endpoints, shared configurations, preparation, transport queues, observer worker owners and data buffers. Excludes OS thread stacks, allocator arena metadata and C-library allocations. Configuration construction also includes its checked warmup connection. Pair peak conservatively bounds one endpoint, with no division by two. Concurrent peak includes every pair held live before and after handshake, and independently seeded per-worker session stores for resumed rows; key resolvers and trust-anchor verifiers remain Arc-shared. Not a universal hostile-input or static stack bound.",
    "provenance": rscrypto_provider_bench::provenance(),
  });
  println!("{}", serde_json::to_string_pretty(&report).expect("resource JSON"));
  assert!(passed, "approved allocation cap exceeded");
}
