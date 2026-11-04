// Override System

/// Try to set detection override.
///
/// Contract: pre-init only. Once [`crate::platform::get`] has initialized
/// detection state, this returns [`OverrideError::AlreadyInitialized`].
#[cold]
pub fn try_set_override(value: Option<Detected>) -> Result<(), OverrideError> {
  #[cfg(any(feature = "std", target_has_atomic = "64"))]
  {
    let value = validate_override(value)?;
    atomic_cache::try_set_override(value)
  }

  #[cfg(not(any(feature = "std", target_has_atomic = "64")))]
  {
    let _ = value;
    Err(OverrideError::Unsupported)
  }
}

/// Keep the AVX-512 capabilities that CPUID reports on Intel hybrid CPUs.
///
/// Detection removes every AVX-512 and AVX10 capability on Intel hybrid models
/// (Alder Lake through Arrow Lake), even when CPUID reports them. Their E-cores
/// lack AVX-512, so a thread that the operating system moves to an E-core
/// during an AVX-512 kernel faults with `SIGILL`. A system can still expose
/// AVX-512 safely, for example with the E-cores disabled in firmware. This call
/// keeps the AVX-512 capabilities that CPUID reports on such a system.
///
/// It adds no capability that CPUID does not report, and it does not change
/// detection on other CPUs. Call it before the first [`crate::platform::get()`],
/// [`crate::platform::caps()`], or [`crate::platform::arch()`] call, and before
/// any rscrypto operation that dispatches on them.
///
/// # Safety
///
/// Every logical processor that this process can ever run on, including after
/// affinity changes and processor hotplug, must implement the AVX-512 and AVX10
/// instructions that CPUID reports during detection. Otherwise rscrypto can
/// execute an instruction that the processor does not implement, which is
/// undefined behavior.
///
/// # Errors
///
/// - [`OverrideError::AlreadyInitialized`] when detection has started. The
///   cached capabilities are unchanged.
/// - [`OverrideError::Unsupported`] when this build has no runtime AVX-512
///   detection to change: targets other than x86-64, and builds without `std`.
///
/// # Examples
///
/// ```no_run
/// use rscrypto::platform::expert;
///
/// // SAFETY: this deployment disables the E-cores in firmware, so every
/// // logical processor implements the AVX-512 that CPUID reports.
/// unsafe { expert::allow_hybrid_avx512() }?;
/// # Ok::<(), expert::OverrideError>(())
/// ```
#[cold]
pub unsafe fn allow_hybrid_avx512() -> Result<(), OverrideError> {
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  {
    atomic_cache::allow_hybrid_avx512()
  }

  #[cfg(not(all(target_arch = "x86_64", feature = "std")))]
  {
    Err(OverrideError::Unsupported)
  }
}

/// Check if an override is set.
#[inline]
#[must_use]
pub fn has_override() -> bool {
  #[cfg(any(feature = "std", target_has_atomic = "64"))]
  {
    atomic_cache::has_override()
  }

  #[cfg(not(any(feature = "std", target_has_atomic = "64")))]
  {
    false
  }
}

#[cold]
#[cfg(not(miri))]
fn detect_with_override() -> Detected {
  #[cfg(any(feature = "std", target_has_atomic = "64"))]
  {
    if let Some(ov) = atomic_cache::seal_and_get_override() {
      return ov;
    }
  }

  detect_uncached()
}

// Atomic Cache
//
// Every build with `std` or 64-bit atomics caches detection here; there is no
// std `OnceLock`. Each first caller detects locally, before it touches shared
// state, so a slow or panicking probe never blocks another thread. One
// compare-and-swap then elects the publisher; the other first callers wait only
// across its single slot store, never across detection.

#[cfg(any(feature = "std", target_has_atomic = "64"))]
mod atomic_cache {
  use core::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, AtomicU8, Ordering},
  };

  use super::*;

  const STATE_UNINIT: u8 = 0;
  const STATE_PUBLISHING: u8 = 1;
  const STATE_READY: u8 = 2;

  struct Slot<T>(UnsafeCell<T>);

  // SAFETY: Slot access is synchronized by explicit atomic state transitions in this module:
  // one elected writer before a Release publication, and readers only after an Acquire
  // observation of that publication (or under `OVERRIDE_LOCK` for the override payload).
  unsafe impl<T: Send + Sync> Sync for Slot<T> {}

  impl<T> Slot<T> {
    const fn new(value: T) -> Self {
      Self(UnsafeCell::new(value))
    }
  }

  /// One published detection result.
  pub(super) struct Cache {
    state: AtomicU8,
    value: Slot<Detected>,
  }

  impl Cache {
    pub(super) const fn new() -> Self {
      Self {
        state: AtomicU8::new(STATE_UNINIT),
        value: Slot::new(Detected::portable()),
      }
    }

    /// Return the published result, publishing `detect()` on the first call.
    ///
    /// Every caller returns the same published value, even if `detect` gives
    /// different results to racing first callers. `detect` is a type parameter,
    /// not a function pointer, so the cold call is direct and statically boundable.
    ///
    /// Both paths end in one read of the published slot. An inlined
    /// `caps().has(..)` therefore loads only the bits it tests; returning the
    /// cold path's value instead would merge the two paths in a stack copy of
    /// the whole `Detected`.
    #[inline]
    #[cfg_attr(miri, allow(dead_code))] // Miri runs `get()` without detection.
    pub(super) fn get_or_init<F: FnOnce() -> Detected>(&self, detect: F) -> Detected {
      if self.state.load(Ordering::Acquire) != STATE_READY {
        self.initialize(detect);
      }
      // SAFETY: Published-slot read because:
      // 1. This thread observed `STATE_READY` with an Acquire load, here or in `initialize`, or it
      //    stored `STATE_READY` itself after its own slot write.
      // 2. The elected writer's one slot write happens-before its Release store of `STATE_READY`.
      // 3. No write follows publication.
      unsafe { *self.value.0.get() }
    }

    /// Publish `detect()` unless another caller already did, and return once
    /// the slot is published.
    #[cold]
    #[inline(never)]
    #[cfg_attr(miri, allow(dead_code))]
    fn initialize<F: FnOnce() -> Detected>(&self, detect: F) {
      // Shared state is untouched until `detect` returns: a panic leaves the
      // cache uninitialized, and no other caller waits for this probe.
      let detected = detect();
      if self
        .state
        .compare_exchange(STATE_UNINIT, STATE_PUBLISHING, Ordering::Acquire, Ordering::Acquire)
        .is_ok()
      {
        // SAFETY: Elected slot write because:
        // 1. Only the caller that moves `STATE_UNINIT` to `STATE_PUBLISHING` writes, and the state
        //    never returns to `STATE_UNINIT`, so there is exactly one writer.
        // 2. Readers access the slot only after observing `STATE_READY`, which is stored below.
        // 3. The Acquire exchange keeps this write after the election.
        unsafe {
          *self.value.0.get() = detected;
        }
        self.state.store(STATE_READY, Ordering::Release);
        return;
      }
      // The elected caller stores one `Detected` and then publishes; wait only across that.
      while self.state.load(Ordering::Acquire) != STATE_READY {
        core::hint::spin_loop();
      }
    }

    fn is_uninit(&self) -> bool {
      self.state.load(Ordering::Acquire) == STATE_UNINIT
    }
  }

  /// The process-wide detection cache behind [`crate::platform::get`].
  pub(super) static CACHE: Cache = Cache::new();

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

  pub(super) fn try_set_override(value: Option<Detected>) -> Result<(), OverrideError> {
    let _guard = lock_override();

    // Every detecting caller seals before it detects, so a sealed state also
    // covers detection that has started but not yet published.
    if OVERRIDE_SEALED.load(Ordering::Relaxed) || !CACHE.is_uninit() {
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

  pub(super) fn has_override() -> bool {
    OVERRIDE_SET.load(Ordering::Acquire)
  }

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  static HYBRID_AVX512_ALLOWED: AtomicBool = AtomicBool::new(false);

  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  pub(super) fn allow_hybrid_avx512() -> Result<(), OverrideError> {
    let _guard = lock_override();

    // The same seal as `try_set_override`: detection seals under this lock
    // before it reads the setting, so a later call cannot change a cached result.
    if OVERRIDE_SEALED.load(Ordering::Relaxed) || !CACHE.is_uninit() {
      return Err(OverrideError::AlreadyInitialized);
    }
    HYBRID_AVX512_ALLOWED.store(true, Ordering::Release);
    Ok(())
  }

  /// Whether [`super::allow_hybrid_avx512`] has run. Read on every detection,
  /// including uncached detection.
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  pub(super) fn hybrid_avx512_allowed() -> bool {
    HYBRID_AVX512_ALLOWED.load(Ordering::Acquire)
  }

  #[cfg(not(miri))]
  pub(super) fn seal_and_get_override() -> Option<Detected> {
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
