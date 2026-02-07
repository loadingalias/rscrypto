// ─────────────────────────────────────────────────────────────────────────────
// Override System
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "std")]
use std::sync::{OnceLock, RwLock};

#[cfg(all(feature = "std", not(miri)))]
static STD_CACHE: OnceLock<Detected> = OnceLock::new();

#[cfg(feature = "std")]
static OVERRIDE: RwLock<Option<Detected>> = RwLock::new(None);

/// Set detection override.
///
/// Must be called **before** the first call to [`get()`]. After caching occurs,
/// updates are rejected.
#[cold]
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
  #[cfg(feature = "std")]
  {
    #[cfg(not(miri))]
    if STD_CACHE.get().is_some() {
      return Err(OverrideError::AlreadyInitialized);
    }

    if let Ok(mut guard) = OVERRIDE.write() {
      *guard = value;
      return Ok(());
    }
    Err(OverrideError::Unsupported)
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  {
    return atomic_cache::try_set_override(value);
  }

  #[cfg(all(not(feature = "std"), not(target_has_atomic = "64")))]
  {
    let _ = value;
    Err(OverrideError::Unsupported)
  }
}

/// Clear detection override.
#[cold]
pub fn clear_override() {
  set_override(None);
}

/// Check if an override is set.
#[inline]
#[must_use]
pub fn has_override() -> bool {
  #[cfg(feature = "std")]
  {
    OVERRIDE.read().map(|g| g.is_some()).unwrap_or(false)
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
    if let Ok(guard) = OVERRIDE.read()
      && let Some(ov) = *guard
    {
      return ov;
    }
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  {
    if let Some(ov) = atomic_cache::get_override() {
      return ov;
    }
  }

  detect_uncached()
}

// ─────────────────────────────────────────────────────────────────────────────
// Atomic Cache (no_std with 64-bit atomics)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
mod atomic_cache {
  use core::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, AtomicU8, Ordering},
  };

  use super::*;

  const STATE_UNINIT: u8 = 0;
  const STATE_INITING: u8 = 1;
  const STATE_READY: u8 = 2;

  struct Slot<T>(UnsafeCell<T>);

  // SAFETY: Slot access is synchronized by explicit atomic state transitions in this module.
  unsafe impl<T> Sync for Slot<T> {}

  impl<T> Slot<T> {
    const fn new(value: T) -> Self {
      Self(UnsafeCell::new(value))
    }
  }

  static STATE: AtomicU8 = AtomicU8::new(STATE_UNINIT);
  static CACHED: Slot<Detected> = Slot::new(Detected::portable());

  static OVERRIDE_SET: AtomicBool = AtomicBool::new(false);
  static OVERRIDE_VALUE: Slot<Option<Detected>> = Slot::new(None);

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

  fn load_cached() -> Detected {
    // SAFETY: Readers only access after STATE_READY with Acquire ordering.
    unsafe { *CACHED.0.get() }
  }

  fn store_cached(det: &Detected) {
    // SAFETY: Single writer while STATE_INITING; readers are blocked until STATE_READY.
    unsafe {
      *CACHED.0.get() = *det;
    }
  }

  pub fn try_set_override(value: Option<Detected>) -> Result<(), OverrideError> {
    if STATE.load(Ordering::Acquire) == STATE_READY {
      return Err(OverrideError::AlreadyInitialized);
    }

    // SAFETY: Override writes are pre-init only; readers gate on OVERRIDE_SET.
    unsafe {
      *OVERRIDE_VALUE.0.get() = value;
    }
    OVERRIDE_SET.store(value.is_some(), Ordering::Release);
    Ok(())
  }

  pub fn has_override() -> bool {
    OVERRIDE_SET.load(Ordering::Acquire)
  }

  pub fn get_override() -> Option<Detected> {
    if !OVERRIDE_SET.load(Ordering::Acquire) {
      return None;
    }
    // SAFETY: OVERRIDE_SET is observed true with Acquire before reading payload.
    unsafe { *OVERRIDE_VALUE.0.get() }
  }
}
