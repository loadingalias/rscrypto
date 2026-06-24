// Override System

#[cfg(all(feature = "std", not(miri)))]
use std::sync::OnceLock;

#[cfg(all(feature = "std", not(miri)))]
static STD_CACHE: OnceLock<Detected> = OnceLock::new();

/// Set detection override.
///
/// Must be called **before** the first call to [`get()`]. After caching occurs,
/// updates are rejected.
///
/// # Panics
///
/// Panics if detection has already been initialized or overrides are unsupported
/// on the current target. Use [`try_set_override()`] for a fallible path.
#[cold]
#[track_caller]
pub fn set_override(value: Option<Detected>) {
  if let Err(err) = try_set_override(value) {
    panic!("platform::set_override failed: {err}");
  }
}

/// Try to set detection override.
///
/// Contract: pre-init only. Once [`get()`] has initialized detection state,
/// this returns [`OverrideError::AlreadyInitialized`].
#[cold]
pub fn try_set_override(value: Option<Detected>) -> Result<(), OverrideError> {
  #[cfg(any(feature = "std", all(not(feature = "std"), target_has_atomic = "64")))]
  let value = validate_override(value)?;

  #[cfg(feature = "std")]
  {
    #[cfg(not(miri))]
    {
      atomic_cache::try_set_override(value, STD_CACHE.get().is_some())
    }

    #[cfg(miri)]
    {
      atomic_cache::try_set_override(value, false)
    }
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  {
    atomic_cache::try_set_override(value, false)
  }

  #[cfg(all(not(feature = "std"), not(target_has_atomic = "64")))]
  {
    let _ = value;
    Err(OverrideError::Unsupported)
  }
}

/// Clear detection override.
#[cold]
#[track_caller]
pub fn clear_override() {
  set_override(None);
}

/// Check if an override is set.
#[inline]
#[must_use]
pub fn has_override() -> bool {
  #[cfg(feature = "std")]
  {
    atomic_cache::has_override()
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  {
    atomic_cache::has_override()
  }

  #[cfg(all(not(feature = "std"), not(target_has_atomic = "64")))]
  {
    false
  }
}

#[cold]
#[cfg(not(miri))]
fn detect_with_override() -> Detected {
  #[cfg(feature = "std")]
  {
    if let Some(ov) = atomic_cache::seal_and_get_override() {
      return ov;
    }
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  {
    if let Some(ov) = atomic_cache::seal_and_get_override() {
      return ov;
    }
  }

  detect_uncached()
}

// Atomic Cache (no_std with 64-bit atomics)

#[cfg(any(feature = "std", all(not(feature = "std"), target_has_atomic = "64")))]
mod atomic_cache {
  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  use core::sync::atomic::AtomicU8;
  use core::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, Ordering},
  };

  use super::*;

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  const STATE_UNINIT: u8 = 0;
  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  const STATE_INITING: u8 = 1;
  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  const STATE_READY: u8 = 2;

  struct Slot<T>(UnsafeCell<T>);

  // SAFETY: Slot access is synchronized by explicit atomic state transitions in this module.
  unsafe impl<T: Sync> Sync for Slot<T> {}

  impl<T> Slot<T> {
    const fn new(value: T) -> Self {
      Self(UnsafeCell::new(value))
    }
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  static STATE: AtomicU8 = AtomicU8::new(STATE_UNINIT);
  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  static CACHED: Slot<Detected> = Slot::new(Detected::portable());

  static OVERRIDE_LOCK: AtomicBool = AtomicBool::new(false);
  static OVERRIDE_SEALED: AtomicBool = AtomicBool::new(false);
  static OVERRIDE_SET: AtomicBool = AtomicBool::new(false);
  static OVERRIDE_VALUE: Slot<Option<Detected>> = Slot::new(None);

  struct OverrideGuard;

  impl Drop for OverrideGuard {
    fn drop(&mut self) {
      OVERRIDE_LOCK.store(false, Ordering::Release);
    }
  }

  fn lock_override() -> OverrideGuard {
    while OVERRIDE_LOCK
      .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
      .is_err()
    {
      core::hint::spin_loop();
    }
    OverrideGuard
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  pub fn get_or_init(f: fn() -> Detected) -> Detected {
    if STATE.load(Ordering::Acquire) == STATE_READY {
      return load_cached();
    }

    match STATE.compare_exchange(STATE_UNINIT, STATE_INITING, Ordering::AcqRel, Ordering::Acquire) {
      Ok(_) => {
        let result = f();
        store_cached(&result);
        STATE.store(STATE_READY, Ordering::Release);
        result
      }
      Err(STATE_INITING) => {
        while STATE.load(Ordering::Acquire) == STATE_INITING {
          core::hint::spin_loop();
        }
        load_cached()
      }
      Err(_) => load_cached(),
    }
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  fn load_cached() -> Detected {
    // SAFETY: Readers only access after STATE_READY with Acquire ordering.
    unsafe { *CACHED.0.get() }
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  fn store_cached(det: &Detected) {
    // SAFETY: Single writer while STATE_INITING; readers are blocked until STATE_READY.
    unsafe {
      *CACHED.0.get() = *det;
    }
  }

  pub fn try_set_override(value: Option<Detected>, already_initialized: bool) -> Result<(), OverrideError> {
    let _guard = lock_override();

    if OVERRIDE_SEALED.load(Ordering::Relaxed) {
      return Err(OverrideError::AlreadyInitialized);
    }

    #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
    if already_initialized || STATE.load(Ordering::Acquire) != STATE_UNINIT {
      return Err(OverrideError::AlreadyInitialized);
    }

    #[cfg(feature = "std")]
    if already_initialized {
      return Err(OverrideError::AlreadyInitialized);
    }

    // SAFETY: Override payload write because:
    // 1. `OVERRIDE_LOCK` serializes every read and write of `OVERRIDE_VALUE`.
    // 2. Detection seals override state under the same lock before reading this payload.
    // 3. The initialization checks above reject writes after detection has started.
    unsafe {
      *OVERRIDE_VALUE.0.get() = value;
    }
    OVERRIDE_SET.store(value.is_some(), Ordering::Release);
    Ok(())
  }

  pub fn has_override() -> bool {
    OVERRIDE_SET.load(Ordering::Acquire)
  }

  #[cfg(not(miri))]
  pub fn seal_and_get_override() -> Option<Detected> {
    let _guard = lock_override();
    OVERRIDE_SEALED.store(true, Ordering::Relaxed);

    if !OVERRIDE_SET.load(Ordering::Acquire) {
      return None;
    }
    // SAFETY: Override payload read because:
    // 1. `OVERRIDE_LOCK` serializes every read and write of `OVERRIDE_VALUE`.
    // 2. `OVERRIDE_SEALED` is set while holding the lock, so future writers are rejected.
    // 3. `OVERRIDE_SET` is observed before reading, so `None` is not read as a set override.
    unsafe { *OVERRIDE_VALUE.0.get() }
  }
}
