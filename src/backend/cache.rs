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

#[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
use core::cell::UnsafeCell;
#[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
use core::mem::MaybeUninit;

// ─────────────────────────────────────────────────────────────────────────────
// OnceCache<T> - Single-value lazy cache
// ─────────────────────────────────────────────────────────────────────────────

/// A lazy cache for a single `Copy` value.
///
/// Building block for dispatcher caching with proper synchronization.
/// See module documentation for platform-specific behavior.
pub struct OnceCache<T: Copy> {
  #[cfg(feature = "std")]
  inner: std::sync::OnceLock<T>,

  #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
  state: core::sync::atomic::AtomicU8,
  #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
  value: UnsafeCell<MaybeUninit<T>>,

  // PhantomData<*const T> makes this !Send + !Sync on no-atomic targets.
  // This is correct: no-atomic targets (e.g., thumbv6m) are single-threaded,
  // so the linker will reject any attempt to use this across threads.
  #[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
  _marker: core::marker::PhantomData<*const T>,
}

// ─── Send/Sync for OnceCache ─────────────────────────────────────────────────
//
// std path: OnceLock<T> auto-derives Send/Sync when T: Send/Sync.
// Since OnceCache only contains OnceLock<T> on std, we inherit those bounds.
// No manual unsafe impl needed!
//
// no_std+atomics: UnsafeCell<T> is !Sync by design. We need unsafe impl Sync
// because our atomic state machine makes concurrent access safe.
// Send is auto-derived since T: Send and AtomicU8 is Send.

#[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
// SAFETY: OnceCache uses an atomic state machine (UNINIT -> INITING -> READY)
// to synchronize access to the UnsafeCell:
// - Only one thread can win the CAS from UNINIT to INITING
// - That thread has exclusive write access until it stores READY
// - All reads after READY are synchronized via Acquire/Release ordering
// - The state machine prevents data races on the inner value
#[allow(unsafe_code)]
unsafe impl<T: Copy + Sync> Sync for OnceCache<T> {}

// no_std without atomics: PhantomData<*const T> makes this !Send + !Sync by default.
// However, these targets (thumbv6m, etc.) are single-threaded, so Sync is trivially safe.
// We need Sync for static dispatchers to compile.
#[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
// SAFETY: Targets without atomics (thumbv6m, etc.) are single-threaded by definition.
// There is no concurrent access possible, so Sync is trivially satisfied.
// The linker will reject any attempt to use threading primitives on these targets.
#[allow(unsafe_code)]
unsafe impl<T: Copy + Sync> Sync for OnceCache<T> {}

impl<T: Copy> OnceCache<T> {
  /// State constants for the atomic state machine
  #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
  const UNINIT: u8 = 0;
  #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
  const INITING: u8 = 1;
  #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
  const READY: u8 = 2;

  /// Create a new empty cache.
  #[must_use]
  pub const fn new() -> Self {
    Self {
      #[cfg(feature = "std")]
      inner: std::sync::OnceLock::new(),

      #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
      state: core::sync::atomic::AtomicU8::new(0),
      #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
      value: UnsafeCell::new(MaybeUninit::uninit()),

      #[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
      _marker: core::marker::PhantomData,
    }
  }

  /// Get the cached value, initializing with `f` if not yet set.
  ///
  /// On targets with atomics, this is thread-safe and the initializer
  /// is called at most once. On targets without atomics, the initializer
  /// is called on every invocation.
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
      use core::sync::atomic::Ordering;

      // Fast path: already initialized
      let state = self.state.load(Ordering::Acquire);
      if state == Self::READY {
        // SAFETY: Value is fully initialized when state is READY.
        // The Acquire load synchronizes with the Release store after initialization.
        #[allow(unsafe_code)]
        return unsafe { (*self.value.get()).assume_init() };
      }

      // Slow path: try to initialize
      if state == Self::UNINIT
        && self
          .state
          .compare_exchange(Self::UNINIT, Self::INITING, Ordering::AcqRel, Ordering::Acquire)
          .is_ok()
      {
        // We won the race - initialize the value
        let value = f();
        // SAFETY: We hold exclusive access during INITING state.
        // No other thread can observe or write to the value until we publish READY.
        #[allow(unsafe_code)]
        unsafe {
          (*self.value.get()).write(value);
        }
        self.state.store(Self::READY, Ordering::Release);
        return value;
      }

      // Another thread is initializing - spin wait
      while self.state.load(Ordering::Acquire) != Self::READY {
        core::hint::spin_loop();
      }
      // SAFETY: State is READY, value is fully initialized.
      // Acquire ordering ensures we see the write.
      #[allow(unsafe_code)]
      unsafe {
        (*self.value.get()).assume_init()
      }
    }

    #[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
    {
      // No caching available - compute every time
      // This is acceptable for single-threaded embedded targets
      f()
    }
  }
}

impl<T: Copy> Default for OnceCache<T> {
  fn default() -> Self {
    Self::new()
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  // ─── OnceCache Tests ───────────────────────────────────────────────────────

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

  // ─── Threading Tests (std only) ────────────────────────────────────────────

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
