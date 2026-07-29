//! Lazy caching primitives for runtime dispatch.
//!
//! - [`OnceCache<T>`]: Single-value lazy cache for selected dispatch entries
//!
//! # Platform Behavior
//!
//! | Platform | Implementation | Behavior |
//! |----------|----------------|----------|
//! | std | `OnceLock` | Thread-safe, initialized once |
//! | no_std + atomics | Atomic state machine | Thread-safe, initialized once |
//! | no_std - atomics | Direct call | Per-call computation |

#[cfg(any(test, all(not(feature = "std"), target_has_atomic = "ptr")))]
use core::cell::UnsafeCell;
#[cfg(any(test, all(not(feature = "std"), target_has_atomic = "ptr")))]
use core::mem::MaybeUninit;

#[cfg(any(test, all(not(feature = "std"), target_has_atomic = "ptr")))]
struct AtomicOnceCache<T: Copy> {
  state: core::sync::atomic::AtomicU8,
  value: UnsafeCell<MaybeUninit<T>>,
}

#[cfg(any(test, all(not(feature = "std"), target_has_atomic = "ptr")))]
struct AtomicInitGuard<'a, T: Copy> {
  cache: &'a AtomicOnceCache<T>,
  armed: bool,
}

#[cfg(any(test, all(not(feature = "std"), target_has_atomic = "ptr")))]
impl<T: Copy> AtomicInitGuard<'_, T> {
  fn disarm(&mut self) {
    self.armed = false;
  }
}

#[cfg(any(test, all(not(feature = "std"), target_has_atomic = "ptr")))]
impl<T: Copy> Drop for AtomicInitGuard<'_, T> {
  fn drop(&mut self) {
    if self.armed {
      self
        .cache
        .state
        .store(AtomicOnceCache::<T>::UNINIT, core::sync::atomic::Ordering::Release);
    }
  }
}

#[cfg(any(test, all(not(feature = "std"), target_has_atomic = "ptr")))]
// SAFETY: `get_or_init` publishes the single initialized value with a Release
// store, and every reader observes that publication with an Acquire load.
unsafe impl<T: Copy + Sync> Sync for AtomicOnceCache<T> {}

#[cfg(any(test, all(not(feature = "std"), target_has_atomic = "ptr")))]
impl<T: Copy> AtomicOnceCache<T> {
  const UNINIT: u8 = 0;
  const INITING: u8 = 1;
  const READY: u8 = 2;

  const fn new() -> Self {
    Self {
      state: core::sync::atomic::AtomicU8::new(Self::UNINIT),
      value: UnsafeCell::new(MaybeUninit::uninit()),
    }
  }

  fn get_or_init(&self, f: impl FnOnce() -> T) -> T {
    use core::sync::atomic::Ordering;

    let mut initializer = Some(f);

    loop {
      match self.state.load(Ordering::Acquire) {
        Self::READY => {
          // SAFETY: READY is published only after the value is initialized. The
          // Acquire load observes the initializing thread's Release store.
          return unsafe { (*self.value.get()).assume_init() };
        }
        Self::UNINIT => {
          if self
            .state
            .compare_exchange(Self::UNINIT, Self::INITING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
          {
            continue;
          }

          let mut guard = AtomicInitGuard {
            cache: self,
            armed: true,
          };
          let value = initializer
            .take()
            .expect("initializer is consumed only by the CAS winner")();
          // SAFETY: The successful UNINIT-to-INITING transition gives this
          // thread exclusive write access until it publishes READY.
          unsafe {
            (*self.value.get()).write(value);
          }
          self.state.store(Self::READY, Ordering::Release);
          guard.disarm();
          return value;
        }
        Self::INITING => {
          while self.state.load(Ordering::Acquire) == Self::INITING {
            core::hint::spin_loop();
          }
        }
        _ => unreachable!("atomic cache state is private and has three values"),
      }
    }
  }
}

/// A lazy cache for a single `Copy` value.
///
/// Building block for dispatcher caching with proper synchronization.
/// See module documentation for platform-specific behavior.
pub struct OnceCache<T: Copy> {
  #[cfg(feature = "std")]
  inner: std::sync::OnceLock<T>,

  #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
  inner: AtomicOnceCache<T>,

  #[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
  _marker: core::marker::PhantomData<T>,
}

impl<T: Copy> OnceCache<T> {
  /// Create a new empty cache.
  #[must_use]
  pub const fn new() -> Self {
    Self {
      #[cfg(feature = "std")]
      inner: std::sync::OnceLock::new(),

      #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
      inner: AtomicOnceCache::new(),

      #[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
      _marker: core::marker::PhantomData,
    }
  }

  /// Get the cached value, initializing with `f` if not yet set.
  ///
  /// On targets with atomics, this is thread-safe and publishes one successful
  /// initialization. A panicking initializer leaves the cache retryable. On
  /// targets without atomics, the initializer is called on every invocation.
  ///
  /// Returns the cached value by copy (since T is Copy).
  #[inline]
  pub fn get_or_init(&self, f: impl FnOnce() -> T) -> T {
    #[cfg(feature = "std")]
    {
      *self.inner.get_or_init(f)
    }

    #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
    {
      self.inner.get_or_init(f)
    }

    #[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
    {
      // No atomics: compute every time on single-threaded embedded targets.
      f()
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

    // On std/atomic targets, initializer should only be called once
    #[cfg(any(feature = "std", target_has_atomic = "ptr"))]
    assert_eq!(call_count, 1);
  }

  #[test]
  fn test_once_cache_default() {
    let cache: OnceCache<u32> = OnceCache::default();
    let value = cache.get_or_init(|| 123);
    assert_eq!(value, 123);
  }

  #[cfg(feature = "std")]
  #[allow(clippy::std_instead_of_core, clippy::std_instead_of_alloc)]
  mod threading_tests {
    use std::{
      sync::atomic::{AtomicUsize, Ordering},
      thread,
      vec::Vec,
    };

    use super::*;

    #[test]
    fn test_atomic_once_cache_concurrent_init() {
      static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
      static CACHE: AtomicOnceCache<u64> = AtomicOnceCache::new();

      let handles: Vec<thread::JoinHandle<()>> = (0..10)
        .map(|_| {
          thread::spawn(|| {
            for _ in 0..100 {
              let value = CACHE.get_or_init(|| {
                CALL_COUNT.fetch_add(1, Ordering::SeqCst);
                17
              });
              assert_eq!(value, 17);
            }
          })
        })
        .collect();

      for handle in handles {
        handle.join().unwrap();
      }

      assert_eq!(CALL_COUNT.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_atomic_once_cache_recovers_after_initializer_panic() {
      let cache = AtomicOnceCache::<u64>::new();

      let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.get_or_init(|| panic!("controlled initializer panic"));
      }));

      assert!(result.is_err());
      assert_eq!(cache.get_or_init(|| 23), 23);
    }

    #[test]
    fn test_atomic_once_cache_waiter_retries_after_initializer_panic() {
      use std::{
        sync::{Arc, Barrier, mpsc},
        time::Duration,
      };

      let cache = Arc::new(AtomicOnceCache::<u64>::new());
      let initializer_entered = Arc::new(Barrier::new(2));
      let release_initializer = Arc::new(Barrier::new(2));

      let panicking_cache = Arc::clone(&cache);
      let panicking_entered = Arc::clone(&initializer_entered);
      let panicking_release = Arc::clone(&release_initializer);
      let panicking = thread::spawn(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
          panicking_cache.get_or_init(|| {
            panicking_entered.wait();
            panicking_release.wait();
            panic!("controlled initializer panic");
          });
        }))
      });

      initializer_entered.wait();
      let waiting_cache = Arc::clone(&cache);
      let (sender, receiver) = mpsc::channel();
      let waiting = thread::spawn(move || {
        sender.send(waiting_cache.get_or_init(|| 29)).unwrap();
      });

      release_initializer.wait();
      assert!(panicking.join().unwrap().is_err());
      assert_eq!(receiver.recv_timeout(Duration::from_secs(2)).unwrap(), 29);
      waiting.join().unwrap();
    }

    #[test]
    fn test_once_cache_concurrent_init() {
      static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
      static CACHE: OnceCache<u64> = OnceCache::new();

      let handles: Vec<thread::JoinHandle<()>> = (0..10)
        .map(|_| {
          thread::spawn(|| {
            for _ in 0..100 {
              let value = CACHE.get_or_init(|| {
                CALL_COUNT.fetch_add(1, Ordering::SeqCst);
                42u64
              });
              assert_eq!(value, 42);
            }
          })
        })
        .collect();

      for handle in handles {
        handle.join().unwrap();
      }

      // Selector called exactly once
      assert_eq!(CALL_COUNT.load(Ordering::SeqCst), 1);
    }
  }
}
