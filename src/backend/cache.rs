//! Lock-free lazy cache for runtime dispatch choices.
//!
//! [`OnceCache<T>`] publishes one value per process, with the pattern of the
//! capability cache in `platform::detect`. Each first caller computes the value
//! on its own stack before it touches shared state, so a slow or panicking
//! initializer never blocks another thread. One compare-and-swap then elects
//! the publisher; the other first callers wait only across its single slot
//! store. Every caller returns the published value, even when racing
//! initializers return different values.
//!
//! Racing first callers can each run the initializer, so an initializer must
//! not have side effects that other callers depend on. It is a type parameter,
//! not a function pointer, so the cold call is direct and statically boundable.
//!
//! Targets without 8-bit compare-and-swap run the initializer on every call.

#[cfg(target_has_atomic = "8")]
use core::{
  cell::UnsafeCell,
  mem::MaybeUninit,
  sync::atomic::{AtomicU8, Ordering},
};

#[cfg(target_has_atomic = "8")]
const STATE_UNINIT: u8 = 0;
#[cfg(target_has_atomic = "8")]
const STATE_PUBLISHING: u8 = 1;
#[cfg(target_has_atomic = "8")]
const STATE_READY: u8 = 2;

/// A lazy cache for a single `Copy` value.
///
/// See the module documentation for the publication rule.
pub(crate) struct OnceCache<T: Copy> {
  #[cfg(target_has_atomic = "8")]
  state: AtomicU8,

  #[cfg(target_has_atomic = "8")]
  value: UnsafeCell<MaybeUninit<T>>,

  #[cfg(not(target_has_atomic = "8"))]
  _marker: core::marker::PhantomData<T>,
}

#[cfg(target_has_atomic = "8")]
// SAFETY: Shared access is synchronized because:
// 1. Only the caller that moves the state from `STATE_UNINIT` to `STATE_PUBLISHING` writes the slot, once.
// 2. Readers read the slot only after an Acquire observation of the writer's Release store of `STATE_READY`.
// 3. `T: Send + Sync`, so a published copy may move to, and be shared by, any reading thread.
unsafe impl<T: Copy + Send + Sync> Sync for OnceCache<T> {}

impl<T: Copy> OnceCache<T> {
  /// Create a new empty cache.
  #[must_use]
  pub(crate) const fn new() -> Self {
    Self {
      #[cfg(target_has_atomic = "8")]
      state: AtomicU8::new(STATE_UNINIT),

      #[cfg(target_has_atomic = "8")]
      value: UnsafeCell::new(MaybeUninit::uninit()),

      #[cfg(not(target_has_atomic = "8"))]
      _marker: core::marker::PhantomData,
    }
  }

  /// Return the published value, publishing `f()` on the first call.
  ///
  /// A panicking initializer leaves the cache unpublished. On targets without
  /// 8-bit compare-and-swap, this calls `f` every time.
  ///
  /// Both paths end in one read of the published slot, so an inlined caller
  /// loads only the fields it uses.
  #[inline]
  pub(crate) fn get_or_init<F: FnOnce() -> T>(&self, f: F) -> T {
    #[cfg(target_has_atomic = "8")]
    {
      if self.state.load(Ordering::Acquire) != STATE_READY {
        self.initialize(f);
      }
      // SAFETY: Published-slot read because:
      // 1. This thread observed `STATE_READY` with an Acquire load, here or in `initialize`, or it stored
      //    `STATE_READY` itself after its own slot write.
      // 2. The elected writer's one slot write happens-before its Release store of `STATE_READY`.
      // 3. No write follows publication, so the slot holds an initialized `T`.
      unsafe { (*self.value.get()).assume_init() }
    }

    #[cfg(not(target_has_atomic = "8"))]
    {
      f()
    }
  }

  /// Publish `f()` unless another caller already did, and return once the
  /// slot is published.
  #[cfg(target_has_atomic = "8")]
  #[cold]
  #[inline(never)]
  fn initialize<F: FnOnce() -> T>(&self, f: F) {
    // Shared state is untouched until `f` returns: a panic leaves the cache
    // unpublished, and no other caller waits for this initializer.
    let value = f();
    if self
      .state
      .compare_exchange(STATE_UNINIT, STATE_PUBLISHING, Ordering::Acquire, Ordering::Acquire)
      .is_ok()
    {
      // SAFETY: Elected slot write because:
      // 1. Only the caller that moves `STATE_UNINIT` to `STATE_PUBLISHING` writes, and the state never returns to
      //    `STATE_UNINIT`, so there is exactly one writer.
      // 2. Readers access the slot only after observing `STATE_READY`, which is stored below.
      // 3. The Acquire exchange keeps this write after the election.
      unsafe {
        (*self.value.get()).write(value);
      }
      self.state.store(STATE_READY, Ordering::Release);
      return;
    }
    // The elected caller stores one value and then publishes; wait only across that.
    while self.state.load(Ordering::Acquire) != STATE_READY {
      core::hint::spin_loop();
    }
  }
}

impl<T: Copy> Default for OnceCache<T> {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_once_cache_basic() {
    static CACHE: OnceCache<u64> = OnceCache::new();

    let mut call_count = 0;
    let value = CACHE.get_or_init(|| {
      call_count += 1;
      42u64
    });

    assert_eq!(value, 42);

    // Second call should return cached value
    let value2 = CACHE.get_or_init(|| {
      call_count += 1;
      99u64
    });

    assert_eq!(value2, 42);

    // Without racing callers, a published value is never recomputed.
    #[cfg(target_has_atomic = "8")]
    assert_eq!(call_count, 1);
  }

  #[test]
  fn test_once_cache_default() {
    let cache: OnceCache<u32> = OnceCache::default();
    let value = cache.get_or_init(|| 123);
    assert_eq!(value, 123);
  }

  #[cfg(all(feature = "std", target_has_atomic = "8"))]
  mod threading_tests {
    use alloc::{boxed::Box, sync::Arc, vec::Vec};
    use core::{
      panic::AssertUnwindSafe,
      sync::atomic::{AtomicUsize, Ordering},
      time::Duration,
    };
    use std::{
      sync::{Barrier, mpsc},
      thread,
    };

    use super::*;

    const THREADS: usize = 10;

    /// Racing first callers whose initializers disagree all return one published value.
    #[test]
    fn test_once_cache_racing_callers_agree() {
      static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
      static CACHE: OnceCache<usize> = OnceCache::new();

      // Every thread is spawned before any is joined: the barrier releases them together.
      let start = Arc::new(Barrier::new(THREADS));
      let mut handles: Vec<thread::JoinHandle<Vec<usize>>> = Vec::with_capacity(THREADS);
      for thread_index in 0..THREADS {
        let start = Arc::clone(&start);
        handles.push(thread::spawn(move || {
          start.wait();
          (0..100)
            .map(|_| {
              CACHE.get_or_init(|| {
                CALL_COUNT.fetch_add(1, Ordering::SeqCst);
                thread_index
              })
            })
            .collect()
        }));
      }

      let observed: Vec<usize> = handles
        .into_iter()
        .flat_map(|handle| handle.join().expect("cache worker thread must not panic"))
        .collect();
      let published = observed[0];
      assert!(published < THREADS, "the published value must come from an initializer");
      assert!(
        observed.iter().all(|&value| value == published),
        "every caller must return the published value"
      );
      // Each thread runs the initializer at most once: a loser waits for publication and then only reads.
      let calls = CALL_COUNT.load(Ordering::SeqCst);
      assert!((1..=THREADS).contains(&calls), "initializer ran {calls} times");
    }

    #[test]
    fn test_once_cache_recovers_after_initializer_panic() {
      let cache = OnceCache::<u64>::new();

      let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        cache.get_or_init(|| std::panic::resume_unwind(Box::new("controlled initializer panic")));
      }));

      result.expect_err("panicking initializer must unwind");
      assert_eq!(cache.get_or_init(|| 23), 23);
    }

    /// A caller never waits for another caller's initializer: it publishes its own value while the first is
    /// still running, and the first caller then returns that published value.
    #[test]
    fn test_once_cache_slow_initializer_does_not_block() {
      let cache = Arc::new(OnceCache::<u64>::new());
      let initializer_entered = Arc::new(Barrier::new(2));
      let release_initializer = Arc::new(Barrier::new(2));

      let slow_cache = Arc::clone(&cache);
      let slow_entered = Arc::clone(&initializer_entered);
      let slow_release = Arc::clone(&release_initializer);
      let slow = thread::spawn(move || {
        slow_cache.get_or_init(|| {
          slow_entered.wait();
          slow_release.wait();
          17
        })
      });

      initializer_entered.wait();
      let fast_cache = Arc::clone(&cache);
      let (sender, receiver) = mpsc::channel();
      let fast = thread::spawn(move || {
        sender
          .send(fast_cache.get_or_init(|| 29))
          .expect("cache caller result receiver must remain connected");
      });
      let fast_value = receiver.recv_timeout(Duration::from_secs(2));

      release_initializer.wait();
      let slow_value = slow.join().expect("slow cache caller must not panic");
      fast.join().expect("fast cache caller must not panic");
      assert_eq!(
        fast_value.expect("a caller must publish while another initializer is still running"),
        29
      );
      assert_eq!(slow_value, 29, "the slow caller must return the published value");
    }
  }
}
