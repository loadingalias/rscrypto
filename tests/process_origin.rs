//! A fork child must not continue an owner of unique state that its parent created.
//!
//! The kernel's fork is the oracle. Each test forks once and relies on
//! cargo-nextest, which runs each test in its own process. libtest still runs
//! the test on its own thread, and the parent can have other threads that hold
//! a lock at the fork. So the child uses only atomics, the call under test,
//! `waitpid`, and `_exit`, and it reports through its exit status. The counting
//! allocator proves that the child allocates nothing.
#![cfg(all(unix, not(miri)))]

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_int;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::System;

use rscrypto::platform::ProcessOrigin;

/// Allocations and reallocations of this process. A fork copies the count, and
/// the child has no other thread, so the child sees only its own allocations.
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

/// End this fork child with `code`, or [`ALLOCATED`] if it allocated after `before`.
fn finish(before: usize, code: u8) -> ! {
  let code = if ALLOCATIONS.load(Ordering::Relaxed) == before {
    code
  } else {
    ALLOCATED
  };
  // SAFETY: `_exit` ends this process at once. It runs no destructor and no `atexit` handler,
  // so the child never touches a lock that a thread of the parent held at the fork.
  unsafe { _exit(c_int::from(code)) }
}

/// Fork, and return the exit code of the child, which runs `child` and exits.
///
/// The child code comes only from `child` or from [`finish`]. It never panics.
fn fork_child(child: impl FnOnce() -> u8) -> u8 {
  // SAFETY: `fork` takes no arguments. The child runs only `child` and `finish`, which use
  // atomics and the calls under test, and it never returns into libtest.
  let pid = unsafe { fork() };
  assert!(pid >= 0, "fork failed");
  if pid == 0 {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    finish(before, child());
  }
  wait(pid).expect("the child exits normally")
}

/// Wait for `pid` and return its exit code, or `None` if a signal ended it.
fn wait(pid: c_int) -> Option<u8> {
  let mut status: c_int = 0;
  // SAFETY: `status` is a live `int` that `waitpid` writes. `pid` is a child of this process.
  let waited = unsafe { waitpid(pid, &raw mut status, 0) };
  // glibc, musl, and Apple: the low 7 bits hold the signal number (zero for a normal exit), and
  // bits 8 to 15 hold the exit code.
  (waited == pid && status & 0x7f == 0)
    .then(|| u8::try_from((status >> 8) & 0xff).ok())
    .flatten()
}

#[test]
fn a_forked_child_reports_another_process_origin() {
  let origin = ProcessOrigin::current();
  let code = fork_child(|| {
    if origin.is_current() {
      return 1;
    }
    if !ProcessOrigin::current().is_current() {
      return 2;
    }
    0
  });
  assert_eq!(
    code, 0,
    "1: the child accepts its parent's origin; 2: the child rejects its own origin; 255: the child allocated"
  );
  assert!(origin.is_current(), "the parent keeps its origin");
}

#[test]
fn a_grandchild_differs_from_its_parent_child() {
  let code = fork_child(|| {
    let child = ProcessOrigin::current();
    // SAFETY: as in `fork_child`. The grandchild runs only atomics, the calls under test, and
    // `finish`.
    let pid = unsafe { fork() };
    if pid < 0 {
      return 3;
    }
    if pid == 0 {
      let before = ALLOCATIONS.load(Ordering::Relaxed);
      let code = if child.is_current() {
        1
      } else if !ProcessOrigin::current().is_current() {
        2
      } else {
        0
      };
      finish(before, code);
    }
    match wait(pid) {
      Some(0) if child.is_current() => 0,
      Some(0) => 4,
      Some(code) => code,
      None => 5,
    }
  });
  assert_eq!(
    code, 0,
    "1: the grandchild accepts the child's origin; 2: the grandchild rejects its own origin; 3: fork failed; 4: \
     the child lost its origin; 5: a signal ended the grandchild; 255: a child allocated"
  );
}

#[cfg(feature = "aes-gcm")]
#[test]
fn a_forked_nonce_counter_issues_no_nonce() {
  use rscrypto::aead::{Nonce96, NonceCounter, NonceCounterSealError};
  use rscrypto::{Aes256Gcm, Aes256GcmKey};

  let cipher = Aes256Gcm::new(&Aes256GcmKey::from_bytes([0x42; 32]));
  let mut counter = NonceCounter::<Aes256Gcm>::new(*b"fork");
  let code = fork_child(|| {
    if counter.next_nonce().is_ok() {
      return 1;
    }
    let mut buffer = [0x5a; 16];
    if !matches!(
      counter.encrypt_in_place(&cipher, b"aad", &mut buffer),
      Err(NonceCounterSealError::Exhausted(_))
    ) {
      return 2;
    }
    if buffer != [0x5a; 16] || counter.next_counter() != 0 {
      return 3;
    }
    0
  });
  assert_eq!(
    code, 0,
    "1: the child issued a nonce; 2: the child sealed; 3: the child consumed a nonce or wrote the buffer; 255: the \
     child allocated"
  );
  assert_eq!(
    counter.next_nonce().expect("the parent issues nonces"),
    Nonce96::from_bytes(*b"fork\0\0\0\0\0\0\0\0"),
    "the parent issues the first nonce"
  );
}

#[cfg(all(feature = "hpke", feature = "x25519", feature = "aes-gcm"))]
#[test]
fn a_forked_hpke_sender_seals_nothing() {
  use rscrypto::X25519SecretKey;
  use rscrypto::auth::{HpkeError, HpkeSender, HpkeX25519HkdfSha256Aes128Gcm, HpkeX25519PrivateKey};

  let recipient = HpkeX25519PrivateKey::new(X25519SecretKey::from_bytes([0x11; 32]));
  let (_, mut sender) =
    HpkeSender::<HpkeX25519HkdfSha256Aes128Gcm>::setup(&recipient.public_key(), b"fork", |random| {
      random.fill(0x22);
      Ok(())
    })
    .expect("setup");
  let plaintext = [0x33; 16];
  let mut out = [0; 32];
  let code = fork_child(|| {
    if sender.seal(b"aad", &plaintext, &mut out) != Err(HpkeError::MessageLimitReached) {
      return 1;
    }
    if sender.sequence() != 0 {
      return 2;
    }
    0
  });
  assert_eq!(
    code, 0,
    "1: the child sealed; 2: the child changed the sequence; 255: the child allocated"
  );
  sender.seal(b"aad", &plaintext, &mut out).expect("the parent seals");
  assert_eq!(sender.sequence(), 1);
}

#[cfg(feature = "chacha20-rng")]
#[test]
fn a_child_generator_keeps_its_root_origin() {
  use rscrypto::ChaCha20Rng;

  let root = ChaCha20Rng::from_seed([0x07; ChaCha20Rng::SEED_SIZE]);
  let code = fork_child(|| {
    if root.child(b"worker\0\0", 0).origin().is_current() {
      return 1;
    }
    if !ChaCha20Rng::from_seed([0x08; ChaCha20Rng::SEED_SIZE])
      .origin()
      .is_current()
    {
      return 2;
    }
    0
  });
  assert_eq!(
    code, 0,
    "1: a child of a copied root reports this process; 2: a root seeded in the child does not; 255: the child \
     allocated"
  );
  assert!(root.child(b"worker\0\0", 0).origin().is_current());
}
