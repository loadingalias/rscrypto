//! A fork child must not draw a copied stream or seal under a copied ticket key.
//!
//! The kernel's fork is the oracle. Each test forks once and relies on
//! cargo-nextest, which runs each test in its own process. libtest still runs
//! the test on its own thread, so another thread can hold a lock at the fork.
//! A lock that another thread holds at the fork stays held forever in the
//! child. So the child allocates nothing, and it takes no global, rustls, or
//! `std` lock such as stdout. It takes only one lock that the test builds and
//! never shares with another thread: the stream's mutex or the ticketer's key
//! lock. Each fork site proves that no thread holds that lock at the fork. The
//! child ends with `_exit`, and its exit status reports the result. The
//! counting allocator proves that the child allocates nothing.
#![cfg(all(unix, not(miri), feature = "chacha20-rng", feature = "aes-gcm"))]

extern crate alloc;

use alloc::sync::Arc;

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_int;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::System;

use rscrypto::ChaCha20Rng;
use rscrypto_rustls::replay::{ScopedRandom, Stream, scope};
use rustls::crypto::SecureRandom as _;

/// Allocations and reallocations of this process. The child has no other thread.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct CountingAlloc;

// SAFETY: CountingAlloc preserves the global allocator contract because:
// 1. Every allocation operation delegates to `System` with the original `Layout`.
// 2. Returned pointers and deallocations are exactly those that `System` produces and accepts.
// 3. The only added behavior is an atomic counter update, which touches no allocation memory.
unsafe impl GlobalAlloc for CountingAlloc {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: `layout` is forwarded unchanged to the platform allocator.
    unsafe { System.alloc(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: `ptr` and `layout` came from this allocator's delegated `System` allocation.
    unsafe { System.dealloc(ptr, layout) };
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: `ptr` and `layout` identify an allocation of this allocator; `new_size` is forwarded.
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

// POSIX `<unistd.h>` and `<sys/wait.h>` in glibc, musl, and Apple's libSystem: `pid_t fork(void)`,
// `pid_t waitpid(pid_t, int *, int)`, and `void _exit(int)`, where `pid_t` is `int`.
unsafe extern "C" {
  fn fork() -> c_int;
  fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
  fn _exit(status: c_int) -> !;
}

/// The child exit code when the child allocated.
const ALLOCATED: u8 = 255;

/// Fork, run `child` in the child, and return the child's exit code.
fn fork_child(child: impl FnOnce() -> u8) -> u8 {
  // SAFETY: `fork` takes no arguments. The child runs only `child`, atomics, and `_exit`, and it
  // never returns into libtest.
  let pid = unsafe { fork() };
  assert!(pid >= 0, "fork failed");
  if pid == 0 {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let code = child();
    let code = if ALLOCATIONS.load(Ordering::Relaxed) == before {
      code
    } else {
      ALLOCATED
    };
    // SAFETY: `_exit` ends the child at once and runs no destructor or `atexit` handler.
    unsafe { _exit(c_int::from(code)) }
  }
  let mut status: c_int = 0;
  // SAFETY: `status` is a live `int` that `waitpid` writes. `pid` is a child of this process.
  let waited = unsafe { waitpid(pid, &raw mut status, 0) };
  assert_eq!(waited, pid, "waitpid");
  // glibc, musl, and Apple: the low 7 bits hold the signal number (zero for a normal exit), and
  // bits 8 to 15 hold the exit code.
  assert_eq!(status & 0x7f, 0, "the child exits normally");
  u8::try_from((status >> 8) & 0xff).expect("an 8-bit exit code")
}

#[test]
fn a_stream_refuses_draws_in_a_fork_child() {
  let stream = Stream::new(ChaCha20Rng::from_seed([0x07; ChaCha20Rng::SEED_SIZE]));
  let mut bytes = [0; 16];
  // The first scope on this thread sets up its thread-local slot before the fork.
  scope(&stream, || ScopedRandom.fill(&mut bytes)).expect("the parent draws");
  // The stream exists only in this test, and no other thread got it or an `Arc` clone of it. Only
  // this thread took its mutex, in the draw above, and released it before the fork. The child's
  // path takes no other lock: `scope` swaps an `Arc` in the thread-local slot, `ScopedRandom`
  // reads two thread-local cells, and the draw locks the stream's mutex and checks the
  // generator's origin with one atomic load (page mode) or `getpid` (process-ID mode).
  let code = fork_child(|| {
    let mut copy = [0x5a; 16];
    match scope(&stream, || ScopedRandom.fill(&mut copy)) {
      Ok(()) => 1,
      Err(_) if copy != [0x5a; 16] => 2,
      Err(_) => 0,
    }
  });
  assert_eq!(
    code, 0,
    "1: the child drew a copied stream; 2: the child wrote the output; 255: the child allocated"
  );
  scope(&stream, || ScopedRandom.fill(&mut bytes)).expect("the parent still draws");
}

#[test]
fn a_generator_ticketer_seals_nothing_in_a_fork_child() {
  #[derive(Debug)]
  struct FixtureTime;

  impl rustls::time_provider::TimeProvider for FixtureTime {
    fn current_time(&self) -> Option<rustls::pki_types::UnixTime> {
      Some(rustls::pki_types::UnixTime::since_unix_epoch(
        core::time::Duration::from_secs(1_792_627_200),
      ))
    }
  }

  let root = ChaCha20Rng::from_seed([0x07; ChaCha20Rng::SEED_SIZE]);
  let ticketer =
    rscrypto_rustls::ticket::ticketer_with_rng(root.child(b"tickets\0", 0), Arc::new(FixtureTime)).expect("ticketer");
  let plain = [0x33; 64];
  ticketer.encrypt(&plain).expect("the parent seals");
  // The ticketer exists only in this test, and no other thread got it or an `Arc` clone of it. Only
  // this thread took its key lock, at construction and in the seal above, and released it before
  // the fork. The child's path takes no other lock: `encrypt` reads `FixtureTime`, raises the
  // atomic latest reading, takes the key lock for reading, and returns `None` at the key's origin
  // check, before any nonce, allocation, or AES-GCM work. No rustls code runs.
  let code = fork_child(|| match ticketer.encrypt(&plain) {
    Some(_) => 1,
    None => 0,
  });
  assert_eq!(
    code, 0,
    "1: the child sealed under a copied key and nonce counter; 255: the child allocated"
  );
  let ticket = ticketer.encrypt(&plain).expect("the parent still seals");
  assert_eq!(ticketer.decrypt(&ticket).as_deref(), Some(plain.as_slice()));
}
