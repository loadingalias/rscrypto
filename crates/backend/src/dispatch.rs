//! Kernel dispatch: selection and caching.
//!
//! This module provides the core dispatch primitives for rscrypto:
//!
//! - [`Candidate`]: A kernel with capability requirements
//! - [`Selected`]: The result of kernel selection
//! - [`select`]: Choose the best kernel from a candidate list
//! - Newtype dispatchers: `Crc32Dispatcher`, `Crc64Dispatcher`, etc.
//!
//! # Design
//!
//! The dispatch system has two paths:
//!
//! 1. **Compile-time selection** (zero-cost): When target features are known at compile time,
//!    dispatch can be resolved to a direct function call with no overhead. This is handled in the
//!    algorithm crate's dispatch function using `cfg!` guards.
//!
//! 2. **Runtime selection** (cached): For generic binaries, the dispatcher detects CPU features
//!    once and caches the selected kernel. Subsequent calls are a single indirect call.
//!
//! # Usage
//!
//! Algorithm crates register kernels as an ordered list of `Candidate`s:
//!
//! ```ignore
//! use backend::dispatch::{Candidate, Selected, select};
//! use platform::caps::{CpuCaps, x86};
//!
//! fn select_crc32c() -> Selected<fn(u32, &[u8]) -> u32> {
//!     let caps = platform::caps();
//!     let candidates = &[
//!         Candidate::new("x86_64/vpclmul", x86::VPCLMUL_READY, vpclmul_kernel),
//!         Candidate::new("x86_64/pclmul", x86::PCLMUL_READY, pclmul_kernel),
//!         Candidate::new("portable", Bits256::NONE, portable_kernel),
//!     ];
//!     select(caps, candidates)
//! }
//! ```

use platform::{Bits256, CpuCaps};

// ─────────────────────────────────────────────────────────────────────────────
// Core Types
// ─────────────────────────────────────────────────────────────────────────────

/// A candidate kernel with capability requirements.
///
/// Candidates are ordered from best to worst. The dispatcher selects the
/// first candidate whose requirements are satisfied by the detected capabilities.
#[derive(Clone, Copy, Debug)]
pub struct Candidate<F> {
  /// Human-readable name for diagnostics (e.g., "x86_64/vpclmul").
  pub name: &'static str,
  /// Required CPU capabilities. Must be a subset of detected caps.
  pub requires: Bits256,
  /// The kernel function pointer.
  pub func: F,
}

impl<F> Candidate<F> {
  /// Create a new candidate.
  #[inline]
  #[must_use]
  pub const fn new(name: &'static str, requires: Bits256, func: F) -> Self {
    Self { name, requires, func }
  }
}

/// The result of kernel selection.
///
/// Contains the selected kernel's name and function pointer.
#[derive(Clone, Copy, Debug)]
pub struct Selected<F> {
  /// Human-readable name of the selected kernel.
  pub name: &'static str,
  /// The selected kernel function.
  pub func: F,
}

impl<F> Selected<F> {
  /// Create a new selected result.
  #[inline]
  #[must_use]
  pub const fn new(name: &'static str, func: F) -> Self {
    Self { name, func }
  }
}

/// Select the best kernel from a candidate list.
///
/// Returns the first candidate whose `requires` is satisfied by `caps`.
/// Panics if no candidate matches (the last candidate should always have
/// `requires = Bits256::NONE` as a fallback).
///
/// # Panics
///
/// Panics if `candidates` is empty or no candidate matches.
///
/// # Performance
///
/// This function is called once during dispatcher initialization, not on
/// every kernel invocation. However, it's marked `#[inline(always)]` to
/// enable the compiler to optimize away the loop when candidates are known
/// at compile time.
#[inline(always)]
#[must_use]
pub fn select<F: Copy>(caps: CpuCaps, candidates: &[Candidate<F>]) -> Selected<F> {
  for candidate in candidates {
    if caps.has(candidate.requires) {
      return Selected::new(candidate.name, candidate.func);
    }
  }

  // This should never happen if the candidate list has a portable fallback.
  // But we need to handle it gracefully.
  panic!("No matching kernel found! Candidate list must include a portable fallback.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Newtype Dispatchers
// ─────────────────────────────────────────────────────────────────────────────
//
// Each algorithm family gets its own dispatcher type for type safety and
// clarity. This prevents accidentally mixing CRC32 and CRC64 dispatchers.

/// Signature for CRC32/CRC32C kernels: `fn(crc: u32, data: &[u8]) -> u32`
pub type Crc32Fn = fn(u32, &[u8]) -> u32;

/// Signature for CRC64 kernels: `fn(crc: u64, data: &[u8]) -> u64`
pub type Crc64Fn = fn(u64, &[u8]) -> u64;

/// Dispatcher for CRC32/CRC32C kernels.
///
/// This dispatcher caches the selected kernel on first access.
/// Under `std`, it uses `OnceLock` for thread-safe initialization.
/// Without `std`, it uses atomic operations.
///
/// # Example
///
/// ```ignore
/// static DISPATCH: Crc32Dispatcher = Crc32Dispatcher::new(select_crc32c);
///
/// fn compute(crc: u32, data: &[u8]) -> u32 {
///     let selected = DISPATCH.get();
///     (selected.func)(crc, data)
/// }
/// ```
pub struct Crc32Dispatcher {
  #[cfg(feature = "std")]
  inner: std::sync::OnceLock<Selected<Crc32Fn>>,

  #[cfg(not(feature = "std"))]
  func: core::sync::atomic::AtomicPtr<()>,
  #[cfg(not(feature = "std"))]
  name_ptr: core::sync::atomic::AtomicPtr<u8>,
  #[cfg(not(feature = "std"))]
  name_len: core::sync::atomic::AtomicUsize,

  /// The selector function that chooses the best kernel.
  selector: fn() -> Selected<Crc32Fn>,
}

impl Crc32Dispatcher {
  /// Create a new dispatcher with the given selector function.
  ///
  /// The selector is called once on first access to choose the best kernel.
  #[must_use]
  pub const fn new(selector: fn() -> Selected<Crc32Fn>) -> Self {
    Self {
      #[cfg(feature = "std")]
      inner: std::sync::OnceLock::new(),

      #[cfg(not(feature = "std"))]
      func: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
      #[cfg(not(feature = "std"))]
      name_ptr: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
      #[cfg(not(feature = "std"))]
      name_len: core::sync::atomic::AtomicUsize::new(0),

      selector,
    }
  }

  /// Get the selected kernel, initializing on first call.
  #[inline]
  #[must_use]
  pub fn get(&self) -> Selected<Crc32Fn> {
    #[cfg(feature = "std")]
    {
      *self.inner.get_or_init(|| (self.selector)())
    }

    #[cfg(not(feature = "std"))]
    {
      use core::sync::atomic::Ordering;

      let func_ptr = self.func.load(Ordering::Acquire);
      if func_ptr.is_null() {
        // First access: run selector and store result
        let selected = (self.selector)();

        // Store function pointer
        let new_func_ptr = selected.func as *mut ();
        self.func.store(new_func_ptr, Ordering::Release);

        // Store name pointer and length separately (Rust strings are NOT null-terminated)
        let name_ptr = selected.name.as_ptr() as *mut u8;
        self.name_ptr.store(name_ptr, Ordering::Release);
        self.name_len.store(selected.name.len(), Ordering::Release);

        selected
      } else {
        // Already initialized: reconstruct Selected from cached values
        // SAFETY: func_ptr was stored from a valid Crc32Fn
        #[allow(unsafe_code)]
        let func: Crc32Fn = unsafe { core::mem::transmute(func_ptr) };

        let name_ptr = self.name_ptr.load(Ordering::Acquire);
        let name_len = self.name_len.load(Ordering::Acquire);

        let name = if name_ptr.is_null() || name_len == 0 {
          "unknown"
        } else {
          // SAFETY: name_ptr and name_len were stored from a valid &'static str
          #[allow(unsafe_code)]
          unsafe {
            core::str::from_utf8_unchecked(core::slice::from_raw_parts(name_ptr, name_len))
          }
        };
        Selected { name, func }
      }
    }
  }

  /// Get the name of the selected backend.
  #[inline]
  #[must_use]
  pub fn backend_name(&self) -> &'static str {
    self.get().name
  }

  /// Call the selected kernel.
  #[inline]
  #[must_use]
  pub fn call(&self, crc: u32, data: &[u8]) -> u32 {
    (self.get().func)(crc, data)
  }
}

// SAFETY: Crc32Dispatcher uses OnceLock (std) or atomic operations (no_std),
// both of which are thread-safe. The stored function pointers are read-only after init.
#[allow(unsafe_code)]
unsafe impl Sync for Crc32Dispatcher {}
#[allow(unsafe_code)]
unsafe impl Send for Crc32Dispatcher {}

/// Dispatcher for CRC64 kernels.
///
/// Similar to `Crc32Dispatcher` but for 64-bit CRC algorithms.
pub struct Crc64Dispatcher {
  #[cfg(feature = "std")]
  inner: std::sync::OnceLock<Selected<Crc64Fn>>,

  #[cfg(not(feature = "std"))]
  func: core::sync::atomic::AtomicPtr<()>,
  #[cfg(not(feature = "std"))]
  name_ptr: core::sync::atomic::AtomicPtr<u8>,
  #[cfg(not(feature = "std"))]
  name_len: core::sync::atomic::AtomicUsize,

  selector: fn() -> Selected<Crc64Fn>,
}

impl Crc64Dispatcher {
  /// Create a new dispatcher with the given selector function.
  #[must_use]
  pub const fn new(selector: fn() -> Selected<Crc64Fn>) -> Self {
    Self {
      #[cfg(feature = "std")]
      inner: std::sync::OnceLock::new(),

      #[cfg(not(feature = "std"))]
      func: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
      #[cfg(not(feature = "std"))]
      name_ptr: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
      #[cfg(not(feature = "std"))]
      name_len: core::sync::atomic::AtomicUsize::new(0),

      selector,
    }
  }

  /// Get the selected kernel, initializing on first call.
  #[inline]
  #[must_use]
  pub fn get(&self) -> Selected<Crc64Fn> {
    #[cfg(feature = "std")]
    {
      *self.inner.get_or_init(|| (self.selector)())
    }

    #[cfg(not(feature = "std"))]
    {
      use core::sync::atomic::Ordering;

      let func_ptr = self.func.load(Ordering::Acquire);
      if func_ptr.is_null() {
        let selected = (self.selector)();
        let new_func_ptr = selected.func as *mut ();
        self.func.store(new_func_ptr, Ordering::Release);

        // Store name pointer and length separately (Rust strings are NOT null-terminated)
        let name_ptr = selected.name.as_ptr() as *mut u8;
        self.name_ptr.store(name_ptr, Ordering::Release);
        self.name_len.store(selected.name.len(), Ordering::Release);

        selected
      } else {
        // SAFETY: func_ptr was stored from a valid Crc64Fn
        #[allow(unsafe_code)]
        let func: Crc64Fn = unsafe { core::mem::transmute(func_ptr) };

        let name_ptr = self.name_ptr.load(Ordering::Acquire);
        let name_len = self.name_len.load(Ordering::Acquire);

        let name = if name_ptr.is_null() || name_len == 0 {
          "unknown"
        } else {
          // SAFETY: name_ptr and name_len were stored from a valid &'static str
          #[allow(unsafe_code)]
          unsafe {
            core::str::from_utf8_unchecked(core::slice::from_raw_parts(name_ptr, name_len))
          }
        };
        Selected { name, func }
      }
    }
  }

  /// Get the name of the selected backend.
  #[inline]
  #[must_use]
  pub fn backend_name(&self) -> &'static str {
    self.get().name
  }

  /// Call the selected kernel.
  #[inline]
  #[must_use]
  pub fn call(&self, crc: u64, data: &[u8]) -> u64 {
    (self.get().func)(crc, data)
  }
}

// SAFETY: Crc64Dispatcher uses OnceLock (std) or atomic operations (no_std),
// both of which are thread-safe. The stored function pointers are read-only after init.
#[allow(unsafe_code)]
unsafe impl Sync for Crc64Dispatcher {}
#[allow(unsafe_code)]
unsafe impl Send for Crc64Dispatcher {}

// ─────────────────────────────────────────────────────────────────────────────
// Generic Dispatcher (for future use)
// ─────────────────────────────────────────────────────────────────────────────

/// Generic dispatcher for arbitrary function signatures.
///
/// This is useful for hash update functions, MAC computations, etc.
/// where the signature varies by algorithm.
///
/// # Type Parameters
///
/// - `F`: The function pointer type (e.g., `fn(&mut [u8; 32], &[u8])`)
#[cfg(feature = "std")]
pub struct GenericDispatcher<F: Copy + 'static> {
  inner: std::sync::OnceLock<Selected<F>>,
  selector: fn() -> Selected<F>,
}

#[cfg(feature = "std")]
impl<F: Copy + 'static> GenericDispatcher<F> {
  /// Create a new generic dispatcher.
  #[must_use]
  pub const fn new(selector: fn() -> Selected<F>) -> Self {
    Self {
      inner: std::sync::OnceLock::new(),
      selector,
    }
  }

  /// Get the selected kernel.
  #[inline]
  #[must_use]
  pub fn get(&self) -> Selected<F> {
    *self.inner.get_or_init(|| (self.selector)())
  }

  /// Get the backend name.
  #[inline]
  #[must_use]
  pub fn backend_name(&self) -> &'static str {
    self.get().name
  }
}

// SAFETY: GenericDispatcher uses OnceLock which is thread-safe.
#[cfg(feature = "std")]
#[allow(unsafe_code)]
unsafe impl<F: Copy + 'static> Sync for GenericDispatcher<F> {}
#[cfg(feature = "std")]
#[allow(unsafe_code)]
unsafe impl<F: Copy + 'static> Send for GenericDispatcher<F> {}

#[cfg(test)]
#[allow(clippy::std_instead_of_alloc, clippy::std_instead_of_core)]
mod tests {
  use super::*;

  #[cfg(feature = "std")]
  extern crate std;

  #[cfg(feature = "std")]
  use std::{format, vec, vec::Vec};

  // ─────────────────────────────────────────────────────────────────────────────
  // Test Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────────

  fn portable_crc32(_crc: u32, _data: &[u8]) -> u32 {
    0xDEADBEEF
  }

  fn fast_crc32(_crc: u32, _data: &[u8]) -> u32 {
    0xCAFEBABE
  }

  fn fastest_crc32(_crc: u32, _data: &[u8]) -> u32 {
    0xFEEDFACE
  }

  fn portable_crc64(_crc: u64, _data: &[u8]) -> u64 {
    0xDEAD_BEEF_CAFE_BABE
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Candidate Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_candidate_creation() {
    let c: Candidate<Crc32Fn> = Candidate::new("test", Bits256::NONE, portable_crc32);
    assert_eq!(c.name, "test");
    assert_eq!(c.requires, Bits256::NONE);
  }

  #[test]
  fn test_candidate_copy() {
    let c1: Candidate<Crc32Fn> = Candidate::new("test", Bits256::NONE, portable_crc32);
    let c2 = c1;
    assert_eq!(c1.name, c2.name);
    assert_eq!(c1.requires, c2.requires);
  }

  #[cfg(feature = "std")]
  #[test]
  fn test_candidate_debug() {
    let c: Candidate<Crc32Fn> = Candidate::new("test", Bits256::NONE, portable_crc32);
    let s = format!("{c:?}");
    assert!(s.contains("test"));
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Selected Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_selected_creation() {
    let s: Selected<Crc32Fn> = Selected::new("test", portable_crc32);
    assert_eq!(s.name, "test");
    assert_eq!((s.func)(0, &[]), 0xDEADBEEF);
  }

  #[test]
  fn test_selected_copy() {
    let s1: Selected<Crc32Fn> = Selected::new("test", portable_crc32);
    let s2 = s1;
    assert_eq!(s1.name, s2.name);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Select Function Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_select_portable_fallback() {
    let caps = CpuCaps::NONE;
    let candidates: &[Candidate<Crc32Fn>] = &[
      Candidate::new("fast", Bits256::from_bit(0), fast_crc32),
      Candidate::new("portable", Bits256::NONE, portable_crc32),
    ];

    let selected = select(caps, candidates);
    assert_eq!(selected.name, "portable");
    assert_eq!((selected.func)(0, &[]), 0xDEADBEEF);
  }

  #[test]
  fn test_select_best_match() {
    let caps = CpuCaps::new(Bits256::from_bit(0));
    let candidates: &[Candidate<Crc32Fn>] = &[
      Candidate::new("fast", Bits256::from_bit(0), fast_crc32),
      Candidate::new("portable", Bits256::NONE, portable_crc32),
    ];

    let selected = select(caps, candidates);
    assert_eq!(selected.name, "fast");
    assert_eq!((selected.func)(0, &[]), 0xCAFEBABE);
  }

  #[test]
  fn test_select_skips_unavailable() {
    // Caps have bit 0, but not bit 1
    let caps = CpuCaps::new(Bits256::from_bit(0));
    let candidates: &[Candidate<Crc32Fn>] = &[
      // Requires bit 1 (not available)
      Candidate::new("needs_bit1", Bits256::from_bit(1), fast_crc32),
      // Requires bit 0 (available)
      Candidate::new("needs_bit0", Bits256::from_bit(0), fast_crc32),
      Candidate::new("portable", Bits256::NONE, portable_crc32),
    ];

    let selected = select(caps, candidates);
    assert_eq!(selected.name, "needs_bit0");
  }

  #[test]
  fn test_select_first_match_wins() {
    // Both candidates match, first one should be selected
    let caps = CpuCaps::new(Bits256::from_bit(0).union(Bits256::from_bit(1)));
    let candidates: &[Candidate<Crc32Fn>] = &[
      Candidate::new("first", Bits256::from_bit(0), fast_crc32),
      Candidate::new("second", Bits256::from_bit(1), fastest_crc32),
      Candidate::new("portable", Bits256::NONE, portable_crc32),
    ];

    let selected = select(caps, candidates);
    assert_eq!(selected.name, "first");
  }

  #[test]
  fn test_select_combined_requirements() {
    // Candidate requires multiple bits
    let caps = CpuCaps::new(Bits256::from_bit(0).union(Bits256::from_bit(1)));
    let combined = Bits256::from_bit(0).union(Bits256::from_bit(1));
    let candidates: &[Candidate<Crc32Fn>] = &[
      Candidate::new("needs_both", combined, fastest_crc32),
      Candidate::new("portable", Bits256::NONE, portable_crc32),
    ];

    let selected = select(caps, candidates);
    assert_eq!(selected.name, "needs_both");
  }

  #[test]
  fn test_select_combined_requirements_partial() {
    // Caps have only one of the required bits
    let caps = CpuCaps::new(Bits256::from_bit(0));
    let combined = Bits256::from_bit(0).union(Bits256::from_bit(1));
    let candidates: &[Candidate<Crc32Fn>] = &[
      Candidate::new("needs_both", combined, fastest_crc32),
      Candidate::new("portable", Bits256::NONE, portable_crc32),
    ];

    let selected = select(caps, candidates);
    assert_eq!(selected.name, "portable"); // Falls back because we're missing bit 1
  }

  #[test]
  #[should_panic(expected = "No matching kernel found")]
  fn test_select_no_fallback_panics() {
    let caps = CpuCaps::NONE;
    let candidates: &[Candidate<Crc32Fn>] = &[
      // No portable fallback!
      Candidate::new("needs_bit0", Bits256::from_bit(0), fast_crc32),
    ];

    let _ = select(caps, candidates);
  }

  #[test]
  #[should_panic(expected = "No matching kernel found")]
  fn test_select_empty_candidates_panics() {
    let caps = CpuCaps::NONE;
    let candidates: &[Candidate<Crc32Fn>] = &[];
    let _ = select(caps, candidates);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Crc32Dispatcher Tests
  // ─────────────────────────────────────────────────────────────────────────────

  fn test_selector() -> Selected<Crc32Fn> {
    Selected::new("test", portable_crc32)
  }

  #[test]
  fn test_crc32_dispatcher() {
    static DISPATCH: Crc32Dispatcher = Crc32Dispatcher::new(test_selector);

    let selected = DISPATCH.get();
    assert_eq!(selected.name, "test");

    // Second call should return cached result
    let selected2 = DISPATCH.get();
    assert_eq!(selected2.name, "test");

    // Test call
    let result = DISPATCH.call(0, &[]);
    assert_eq!(result, 0xDEADBEEF);
  }

  #[test]
  fn test_dispatcher_backend_name() {
    static DISPATCH: Crc32Dispatcher = Crc32Dispatcher::new(test_selector);
    assert_eq!(DISPATCH.backend_name(), "test");
  }

  #[test]
  fn test_crc32_dispatcher_with_data() {
    fn echo_crc(crc: u32, data: &[u8]) -> u32 {
      crc.wrapping_add(data.len() as u32)
    }

    fn echo_selector() -> Selected<Crc32Fn> {
      Selected::new("echo", echo_crc)
    }

    static DISPATCH: Crc32Dispatcher = Crc32Dispatcher::new(echo_selector);

    assert_eq!(DISPATCH.call(100, &[1, 2, 3, 4, 5]), 105);
    assert_eq!(DISPATCH.call(0, &[]), 0);
    assert_eq!(DISPATCH.call(u32::MAX, &[1]), 0); // Wrapping
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Crc64Dispatcher Tests
  // ─────────────────────────────────────────────────────────────────────────────

  fn test_selector_64() -> Selected<Crc64Fn> {
    Selected::new("test64", portable_crc64)
  }

  #[test]
  fn test_crc64_dispatcher() {
    static DISPATCH: Crc64Dispatcher = Crc64Dispatcher::new(test_selector_64);

    let selected = DISPATCH.get();
    assert_eq!(selected.name, "test64");

    // Test call
    let result = DISPATCH.call(0, &[]);
    assert_eq!(result, 0xDEAD_BEEF_CAFE_BABE);
  }

  #[test]
  fn test_crc64_dispatcher_backend_name() {
    static DISPATCH: Crc64Dispatcher = Crc64Dispatcher::new(test_selector_64);
    assert_eq!(DISPATCH.backend_name(), "test64");
  }

  #[test]
  fn test_crc64_dispatcher_caching() {
    static DISPATCH: Crc64Dispatcher = Crc64Dispatcher::new(test_selector_64);

    // Multiple calls should return the same result (caching)
    let r1 = DISPATCH.get();
    let r2 = DISPATCH.get();
    let r3 = DISPATCH.get();

    assert_eq!(r1.name, r2.name);
    assert_eq!(r2.name, r3.name);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Multi-threaded Dispatcher Tests (std only)
  // ─────────────────────────────────────────────────────────────────────────────

  #[cfg(feature = "std")]
  mod threading_tests {
    use std::{
      sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
      },
      thread,
    };

    use super::*;

    #[test]
    fn test_crc32_dispatcher_concurrent_access() {
      static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

      fn counting_selector() -> Selected<Crc32Fn> {
        CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        Selected::new("counted", portable_crc32)
      }

      static DISPATCH: Crc32Dispatcher = Crc32Dispatcher::new(counting_selector);

      // Reset counter
      CALL_COUNT.store(0, Ordering::SeqCst);

      // Spawn multiple threads that all call get()
      let handles: Vec<_> = (0..10)
        .map(|_| {
          thread::spawn(|| {
            for _ in 0..100 {
              let selected = DISPATCH.get();
              assert_eq!(selected.name, "counted");
            }
          })
        })
        .collect();

      for handle in handles {
        handle.join().unwrap();
      }

      // Selector should have been called exactly once (OnceLock guarantees this)
      assert_eq!(CALL_COUNT.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_crc64_dispatcher_concurrent_access() {
      static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

      fn counting_selector_64() -> Selected<Crc64Fn> {
        CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        Selected::new("counted64", portable_crc64)
      }

      static DISPATCH: Crc64Dispatcher = Crc64Dispatcher::new(counting_selector_64);

      CALL_COUNT.store(0, Ordering::SeqCst);

      let handles: Vec<_> = (0..10)
        .map(|_| {
          thread::spawn(|| {
            for _ in 0..100 {
              let selected = DISPATCH.get();
              assert_eq!(selected.name, "counted64");
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
    fn test_dispatcher_call_concurrent() {
      fn accumulate(crc: u32, data: &[u8]) -> u32 {
        data.iter().fold(crc, |acc, &b| acc.wrapping_add(b as u32))
      }

      fn accum_selector() -> Selected<Crc32Fn> {
        Selected::new("accum", accumulate)
      }

      static DISPATCH: Crc32Dispatcher = Crc32Dispatcher::new(accum_selector);

      let results = Arc::new(std::sync::Mutex::new(Vec::new()));

      let handles: Vec<_> = (0..4)
        .map(|i| {
          let results = Arc::clone(&results);
          thread::spawn(move || {
            let data = vec![i as u8; 10];
            let result = DISPATCH.call(0, &data);
            results.lock().unwrap().push(result);
          })
        })
        .collect();

      for handle in handles {
        handle.join().unwrap();
      }

      let results = results.lock().unwrap();
      assert_eq!(results.len(), 4);
      // Each thread computes: 10 * thread_id
      for &result in results.iter() {
        assert!(result <= 30); // Max is 10 * 3 = 30
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // GenericDispatcher Tests (std only)
  // ─────────────────────────────────────────────────────────────────────────────

  #[cfg(feature = "std")]
  mod generic_dispatcher_tests {
    use super::*;

    type HashFn = fn(&mut [u8; 32], &[u8]);

    fn test_hash(state: &mut [u8; 32], data: &[u8]) {
      // Simple mock hash: XOR first byte of data into state
      if !data.is_empty() {
        state[0] ^= data[0];
      }
    }

    fn hash_selector() -> Selected<HashFn> {
      Selected::new("test_hash", test_hash)
    }

    #[test]
    fn test_generic_dispatcher() {
      static DISPATCH: GenericDispatcher<HashFn> = GenericDispatcher::new(hash_selector);

      let selected = DISPATCH.get();
      assert_eq!(selected.name, "test_hash");
    }

    #[test]
    fn test_generic_dispatcher_backend_name() {
      static DISPATCH: GenericDispatcher<HashFn> = GenericDispatcher::new(hash_selector);
      assert_eq!(DISPATCH.backend_name(), "test_hash");
    }

    #[test]
    fn test_generic_dispatcher_caching() {
      static DISPATCH: GenericDispatcher<HashFn> = GenericDispatcher::new(hash_selector);

      let r1 = DISPATCH.get();
      let r2 = DISPATCH.get();
      assert_eq!(r1.name, r2.name);
    }
  }
}
