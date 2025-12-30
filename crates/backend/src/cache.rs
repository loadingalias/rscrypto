//! Lazy caching for policy and kernel tables.
//!
//! This module provides a `PolicyCache` type that caches (Policy, Kernels) pairs
//! with the same semantics as `std::sync::OnceLock` but works on no_std targets.
//!
//! # Caching Strategy
//!
//! - **std**: Uses `OnceLock` for thread-safe lazy initialization
//! - **no_std with atomics**: Uses atomic state machine (similar to `Dispatcher`)
//! - **no_std without atomics**: Per-call computation (unavoidable for single-threaded embedded)

#[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
use core::cell::UnsafeCell;
#[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
use core::mem::MaybeUninit;

/// A cache for (Policy, Kernels) pairs.
///
/// This provides lazy initialization with the following properties:
/// - Zero-cost after first initialization (just a pointer load)
/// - Thread-safe on targets with atomics
/// - Falls back to per-call computation on targets without atomics
///
/// # Type Parameters
///
/// - `P`: Policy type (must be `Copy + Send + Sync`)
/// - `K`: Kernels type (must be `Copy + Send + Sync`)
pub struct PolicyCache<P: Copy, K: Copy> {
  #[cfg(feature = "std")]
  inner: std::sync::OnceLock<(P, K)>,

  #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
  state: core::sync::atomic::AtomicU8,
  #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
  value: UnsafeCell<MaybeUninit<(P, K)>>,

  // Marker to consume type parameters and make the struct !Send/!Sync on no-atomic targets
  // (they're single-threaded anyway, so this is fine)
  #[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
  _marker: core::marker::PhantomData<*const (P, K)>,
}

// SAFETY: The cache is safe to share between threads because:
// - On std: OnceLock handles synchronization
// - On no_std with atomics: We use atomic operations for synchronization
// - On no_std without atomics: Target is single-threaded
#[allow(unsafe_code)]
#[cfg(feature = "std")]
unsafe impl<P: Copy + Send + Sync, K: Copy + Send + Sync> Send for PolicyCache<P, K> {}
#[allow(unsafe_code)]
#[cfg(feature = "std")]
unsafe impl<P: Copy + Send + Sync, K: Copy + Send + Sync> Sync for PolicyCache<P, K> {}

#[allow(unsafe_code)]
#[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
unsafe impl<P: Copy + Send + Sync, K: Copy + Send + Sync> Send for PolicyCache<P, K> {}
#[allow(unsafe_code)]
#[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
unsafe impl<P: Copy + Send + Sync, K: Copy + Send + Sync> Sync for PolicyCache<P, K> {}

// SAFETY: On no-atomic targets (thumbv6m), the target is single-threaded,
// so Send/Sync are trivially satisfied.
#[allow(unsafe_code)]
#[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
unsafe impl<P: Copy + Send + Sync, K: Copy + Send + Sync> Send for PolicyCache<P, K> {}
#[allow(unsafe_code)]
#[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
unsafe impl<P: Copy + Send + Sync, K: Copy + Send + Sync> Sync for PolicyCache<P, K> {}

impl<P: Copy, K: Copy> PolicyCache<P, K> {
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
  /// is called on every invocation (unavoidable for single-threaded embedded).
  ///
  /// Returns the cached value by copy (since P and K are Copy).
  #[inline]
  pub fn get_or_init(&self, f: impl FnOnce() -> (P, K)) -> (P, K) {
    #[cfg(feature = "std")]
    {
      *self.inner.get_or_init(f)
    }

    #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
    {
      use core::sync::atomic::Ordering;

      let state = self.state.load(Ordering::Acquire);
      if state == Self::READY {
        // SAFETY: Value is initialized when state is READY
        #[allow(unsafe_code)]
        return unsafe { (*self.value.get()).assume_init() };
      }

      if state == Self::UNINIT {
        if self
          .state
          .compare_exchange(Self::UNINIT, Self::INITING, Ordering::AcqRel, Ordering::Acquire)
          .is_ok()
        {
          let value = f();
          // SAFETY: We hold exclusive access during INITING state
          #[allow(unsafe_code)]
          unsafe {
            (*self.value.get()).write(value);
          }
          self.state.store(Self::READY, Ordering::Release);
          return value;
        }
      }

      // Another thread is initializing - spin wait
      while self.state.load(Ordering::Acquire) != Self::READY {
        core::hint::spin_loop();
      }
      // SAFETY: Value is initialized when state is READY
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

  /// Get a reference to the cached value, initializing with `f` if not yet set.
  ///
  /// This is the preferred API when you need to pass references to the cached values.
  /// On std and no_std with atomics, this returns a reference to the cached data.
  /// On no_std without atomics, this computes and returns by value (caller must handle).
  #[inline]
  pub fn get_or_init_ref(&self, f: impl FnOnce() -> (P, K)) -> (&P, &K) {
    #[cfg(feature = "std")]
    {
      let cached = self.inner.get_or_init(f);
      (&cached.0, &cached.1)
    }

    #[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
    {
      use core::sync::atomic::Ordering;

      let state = self.state.load(Ordering::Acquire);
      if state != Self::READY {
        if state == Self::UNINIT {
          if self
            .state
            .compare_exchange(Self::UNINIT, Self::INITING, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
          {
            let value = f();
            // SAFETY: We hold exclusive access during INITING state
            #[allow(unsafe_code)]
            unsafe {
              (*self.value.get()).write(value);
            }
            self.state.store(Self::READY, Ordering::Release);
          } else {
            // Another thread is initializing - spin wait
            while self.state.load(Ordering::Acquire) != Self::READY {
              core::hint::spin_loop();
            }
          }
        } else {
          // Another thread is initializing - spin wait
          while self.state.load(Ordering::Acquire) != Self::READY {
            core::hint::spin_loop();
          }
        }
      }

      // SAFETY: Value is initialized when state is READY
      #[allow(unsafe_code)]
      unsafe {
        let ptr = (*self.value.get()).as_ptr();
        (&(*ptr).0, &(*ptr).1)
      }
    }

    // Note: This arm is cfg'd out because thumbv6m targets don't have CRC SIMD.
    // If future no-atomic targets need caching, we'd need a different approach.
    #[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
    {
      // Unreachable in practice - no-atomic targets are thumbv6m which don't have SIMD
      let _ = f;
      unreachable!("PolicyCache::get_or_init_ref not supported on targets without atomics");
    }
  }
}

impl<P: Copy, K: Copy> Default for PolicyCache<P, K> {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_policy_cache_basic() {
    static CACHE: PolicyCache<u32, u64> = PolicyCache::new();

    let mut call_count = 0;
    let (p, k) = CACHE.get_or_init(|| {
      call_count += 1;
      (42u32, 123u64)
    });

    assert_eq!(p, 42);
    assert_eq!(k, 123);

    // Second call should return cached value
    let (p2, k2) = CACHE.get_or_init(|| {
      call_count += 1;
      (99u32, 999u64)
    });

    assert_eq!(p2, 42);
    assert_eq!(k2, 123);

    // On std/atomic targets, initializer should only be called once
    #[cfg(any(feature = "std", target_has_atomic = "ptr"))]
    assert_eq!(call_count, 1);
  }
}
