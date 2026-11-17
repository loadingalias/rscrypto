//! A fork child must not draw a copied stream or seal under a copied ticket key.
//!
//! The kernel's fork is the oracle. Each test forks once and relies on
//! cargo-nextest, which runs each test in its own process. libtest still runs
//! the test on its own thread, so another thread can hold a lock at the fork.
//! A lock that another thread holds at the fork stays held forever in the
//! child. So the child takes no global, rustls, or `std` lock such as stdout,
//! prints nothing, and cannot panic. It takes only one lock that the test
//! builds and never shares with another thread: the stream's mutex or the
//! ticketer's key lock. Each fork site proves that no thread holds that lock at
//! the fork. The child ends with `_exit`, and its exit status reports the
//! result. The counting allocator proves that the child allocates nothing,
//! except in the one test that says why it must allocate.
#![cfg(all(
  unix,
  not(miri),
  feature = "aes-gcm",
  any(feature = "chacha20-rng", feature = "getrandom")
))]

extern crate alloc;

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_int;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::System;

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

/// Fork, run `child` in the child, and return the child's exit code. With `count_allocations`,
/// the child exits with [`ALLOCATED`] if it allocated.
fn fork_child(count_allocations: bool, child: impl FnOnce() -> u8) -> u8 {
  // SAFETY: `fork` takes no arguments. The child runs only `child`, atomics, and `_exit`, and it
  // never returns into libtest.
  let pid = unsafe { fork() };
  assert!(pid >= 0, "fork failed");
  if pid == 0 {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let code = child();
    let code = if !count_allocations || ALLOCATIONS.load(Ordering::Relaxed) == before {
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

#[cfg(feature = "chacha20-rng")]
#[test]
fn a_stream_refuses_draws_in_a_fork_child() {
  use rscrypto::ChaCha20Rng;
  use rscrypto_rustls::replay::{ScopedRandom, Stream, scope};
  use rustls::crypto::SecureRandom as _;

  let stream = Stream::new(ChaCha20Rng::from_seed([0x07; ChaCha20Rng::SEED_SIZE]));
  let mut bytes = [0; 16];
  // The first scope on this thread sets up its thread-local slot before the fork.
  scope(&stream, || ScopedRandom.fill(&mut bytes)).expect("the parent draws");
  // The stream exists only in this test, and no other thread got it or an `Arc` clone of it. Only
  // this thread took its mutex, in the draw above, and released it before the fork. The child's
  // path takes no other lock: `scope` swaps an `Arc` in the thread-local slot, `ScopedRandom`
  // reads two thread-local cells, and the draw locks the stream's mutex and checks the
  // generator's origin with one atomic load (page mode) or `getpid` (process-ID mode).
  let code = fork_child(true, || {
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

#[cfg(feature = "chacha20-rng")]
#[test]
fn a_generator_ticketer_seals_nothing_in_a_fork_child() {
  use alloc::sync::Arc;

  use rscrypto::ChaCha20Rng;

  /// The fixture clock: 2026-10-22 UTC.
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
  // this thread took its key lock and its generator's mutex, at construction and in the seal above,
  // and released them before the fork. The child's path takes no other lock: `encrypt` reads
  // `FixtureTime`, raises the atomic latest reading, finds that this process did not create the
  // current key, and takes the key lock for writing to rotate. The rotation locks the generator's
  // mutex, whose origin check refuses the draw, so no key, nonce, allocation, or AES-GCM work
  // follows, and `encrypt` returns `None`. No rustls code runs.
  let code = fork_child(true, || match ticketer.encrypt(&plain) {
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

/// A ticketer over operating-system entropy replaces a copied key at its first use in a fork child.
#[cfg(feature = "getrandom")]
#[test]
fn an_os_ticketer_rotates_in_a_fork_child() {
  let ticketer = rscrypto_rustls::ticket::ticketer().expect("ticketer");
  let plain = [0x33; 64];
  // A ticket starts with its 12-byte nonce: 8 zero bytes and a big-endian count of the key's
  // tickets.
  let first = ticketer.encrypt(&plain).expect("the parent seals");
  assert_eq!(first[..12], [0; 12], "a key's first ticket has nonce 0");
  // Locks: the ticketer exists only in this test, and no other thread got it or an `Arc` clone of
  // it. Only this thread took its key lock, at construction and in the seal above, and released it
  // before the fork. The child takes it for writing to rotate and for reading to seal and open.
  //
  // Entropy: the rotation draws 32 bytes from the operating system, through getrandom 0.4.3. On
  // Apple, `src/backends.rs:52-59` selects `src/backends/getentropy.rs:23-37`, which calls
  // `getentropy(2)` with no lock. On Linux, `src/backends.rs:76-107` selects
  // `src/backends/linux_android_with_fallback.rs:70-86`, which reads the function pointer that an
  // atomic caches and calls the C library's `getrandom`; on riscv64, `src/backends/getrandom.rs:26-31`
  // calls it directly. The parent filled that cache when it drew the first key. musl's `getrandom`
  // is a plain system call. In glibc 2.39, `getrandom` is a plain system call. In glibc 2.41 and later, this
  // thread reuses its vDSO state with one atomic load
  // (`sysdeps/unix/sysv/linux/getrandom.c:186-202` in glibc 2.42), and the fork child resets the
  // state allocator's lock (`sysdeps/nptl/fork.h:50`, `getrandom.c:330-335`).
  //
  // Allocation: the new key and each returned ticket allocate, so this child skips the
  // allocation count. The C libraries make malloc safe in a fork child: glibc 2.39 takes every
  // malloc lock around the fork (`posix/fork.c:71, 88, 120`), musl 1.2.5 calls `__malloc_atfork`
  // (`src/process/fork.c:60, 76`), and Apple libmalloc-812.100.31 provides `_malloc_fork_prepare`
  // and `_malloc_fork_child` (`src/malloc.c:3723, 3737`), which libSystem runs around the fork.
  let code = fork_child(false, || {
    let Some(second) = ticketer.encrypt(&plain) else {
      return 1;
    };
    // The copied key has issued one ticket, so a second ticket under it would have nonce 1.
    if second[..12] != [0; 12] {
      return 2;
    }
    // AES-GCM is deterministic: one key, nonce, plaintext, and AAD give one ticket.
    if second == first {
      return 3;
    }
    match ticketer.decrypt(&first) {
      Some(opened) if opened == plain => 0,
      _ => 4,
    }
  });
  assert_eq!(
    code, 0,
    "1: the child sealed nothing; 2: the child sealed under the copied key; 3: the child's new key \
     equals the copied key; 4: the copied key no longer opens the parent's ticket"
  );
  let third = ticketer.encrypt(&plain).expect("the parent still seals");
  assert_eq!(
    third[..12],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "the parent continues with nonce 1"
  );
}
