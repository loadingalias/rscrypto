//! Kernel dispatch: selection and caching.
//!
//! This module provides the core dispatch primitives for rscrypto:
//!
//! - [`Candidate`]: A kernel with capability requirements
//! - [`Selected`]: The result of kernel selection
//! - [`select`]: Choose the best kernel from a candidate list
//! - Dispatchers: `Crc32Dispatcher`, `Crc64Dispatcher`, etc.
//!
//! # Design
//!
//! The dispatch system has two paths:
//!
//! 1. **Compile-time selection** (zero-cost): When target features are known at compile time,
//!    dispatch can be resolved to a direct function call with no overhead.
//!
//! 2. **Runtime selection** (cached): For generic binaries, the dispatcher detects CPU features
//!    once and caches the selected kernel. Subsequent calls are a single indirect call.

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
#[inline(always)]
#[must_use]
pub fn select<F: Copy>(caps: CpuCaps, candidates: &[Candidate<F>]) -> Selected<F> {
  for candidate in candidates {
    if caps.has(candidate.requires) {
      return Selected::new(candidate.name, candidate.func);
    }
  }
  panic!("No matching kernel found! Candidate list must include a portable fallback.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Macro
// ─────────────────────────────────────────────────────────────────────────────

/// Generates a dispatcher type with caching for a specific function signature.
///
/// Usage:
/// ```ignore
/// define_dispatcher!(Crc32Dispatcher, Crc32Fn, u32);
/// define_dispatcher!(Crc64Dispatcher, Crc64Fn, u64);
/// ```
macro_rules! define_dispatcher {
  (
    $(#[$meta:meta])*
    $name:ident, $fn_ty:ty, $state:ty
  ) => {
    $(#[$meta])*
    pub struct $name {
      #[cfg(feature = "std")]
      inner: std::sync::OnceLock<Selected<$fn_ty>>,

      #[cfg(not(feature = "std"))]
      func: core::sync::atomic::AtomicPtr<()>,
      #[cfg(not(feature = "std"))]
      name_ptr: core::sync::atomic::AtomicPtr<u8>,
      #[cfg(not(feature = "std"))]
      name_len: core::sync::atomic::AtomicUsize,

      selector: fn() -> Selected<$fn_ty>,
    }

    impl $name {
      /// Create a new dispatcher with the given selector function.
      #[must_use]
      pub const fn new(selector: fn() -> Selected<$fn_ty>) -> Self {
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
      pub fn get(&self) -> Selected<$fn_ty> {
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
            self.func.store(selected.func as *mut (), Ordering::Release);
            self.name_ptr.store(selected.name.as_ptr() as *mut u8, Ordering::Release);
            self.name_len.store(selected.name.len(), Ordering::Release);
            selected
          } else {
            // SAFETY: func_ptr was stored from a valid function pointer of type $fn_ty
            #[allow(unsafe_code)]
            let func: $fn_ty = unsafe { core::mem::transmute(func_ptr) };

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
      pub fn call(&self, state: $state, data: &[u8]) -> $state {
        (self.get().func)(state, data)
      }
    }

    // SAFETY: Dispatcher uses OnceLock (std) or atomic operations (no_std),
    // both of which are thread-safe. Stored function pointers are read-only after init.
    #[allow(unsafe_code)]
    unsafe impl Sync for $name {}
    #[allow(unsafe_code)]
    unsafe impl Send for $name {}
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Types
// ─────────────────────────────────────────────────────────────────────────────

/// Function signature for CRC32/CRC32C kernels.
pub type Crc32Fn = fn(u32, &[u8]) -> u32;

/// Function signature for CRC64 kernels.
pub type Crc64Fn = fn(u64, &[u8]) -> u64;

define_dispatcher!(
  /// Dispatcher for CRC32/CRC32C kernels.
  ///
  /// Caches the selected kernel on first access. Thread-safe.
  Crc32Dispatcher, Crc32Fn, u32
);

define_dispatcher!(
  /// Dispatcher for CRC64 kernels.
  ///
  /// Caches the selected kernel on first access. Thread-safe.
  Crc64Dispatcher, Crc64Fn, u64
);

// ─────────────────────────────────────────────────────────────────────────────
// Generic Dispatcher (std only, for custom signatures)
// ─────────────────────────────────────────────────────────────────────────────

/// Generic dispatcher for arbitrary function signatures.
///
/// Use this for hash functions, MACs, or other algorithms with custom signatures.
/// For CRC algorithms, prefer the typed dispatchers above.
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

#[cfg(feature = "std")]
#[allow(unsafe_code)]
unsafe impl<F: Copy + 'static> Sync for GenericDispatcher<F> {}
#[cfg(feature = "std")]
#[allow(unsafe_code)]
unsafe impl<F: Copy + 'static> Send for GenericDispatcher<F> {}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::std_instead_of_alloc, clippy::std_instead_of_core)]
mod tests {
  use super::*;

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

  #[test]
  fn test_candidate_creation() {
    let c: Candidate<Crc32Fn> = Candidate::new("test", Bits256::NONE, portable_crc32);
    assert_eq!(c.name, "test");
    assert_eq!(c.requires, Bits256::NONE);
  }

  #[test]
  fn test_select_portable_fallback() {
    let caps = CpuCaps::NONE;
    let candidates: &[Candidate<Crc32Fn>] = &[
      Candidate::new("fast", Bits256::from_bit(0), fast_crc32),
      Candidate::new("portable", Bits256::NONE, portable_crc32),
    ];
    let selected = select(caps, candidates);
    assert_eq!(selected.name, "portable");
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
  }

  #[test]
  fn test_select_first_match_wins() {
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
  #[should_panic(expected = "No matching kernel found")]
  fn test_select_no_fallback_panics() {
    let caps = CpuCaps::NONE;
    let candidates: &[Candidate<Crc32Fn>] = &[Candidate::new("needs_bit0", Bits256::from_bit(0), fast_crc32)];
    let _ = select(caps, candidates);
  }

  fn test_selector() -> Selected<Crc32Fn> {
    Selected::new("test", portable_crc32)
  }

  #[test]
  fn test_crc32_dispatcher() {
    static DISPATCH: Crc32Dispatcher = Crc32Dispatcher::new(test_selector);
    let selected = DISPATCH.get();
    assert_eq!(selected.name, "test");
    assert_eq!(DISPATCH.call(0, &[]), 0xDEADBEEF);
  }

  fn test_selector_64() -> Selected<Crc64Fn> {
    Selected::new("test64", portable_crc64)
  }

  #[test]
  fn test_crc64_dispatcher() {
    static DISPATCH: Crc64Dispatcher = Crc64Dispatcher::new(test_selector_64);
    let selected = DISPATCH.get();
    assert_eq!(selected.name, "test64");
    assert_eq!(DISPATCH.call(0, &[]), 0xDEAD_BEEF_CAFE_BABE);
  }

  #[cfg(feature = "std")]
  mod threading_tests {
    use std::{
      sync::atomic::{AtomicUsize, Ordering},
      thread,
      vec::Vec,
    };

    use super::*;

    #[test]
    fn test_dispatcher_concurrent_init() {
      static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

      fn counting_selector() -> Selected<Crc32Fn> {
        CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        Selected::new("counted", portable_crc32)
      }

      static DISPATCH: Crc32Dispatcher = Crc32Dispatcher::new(counting_selector);
      CALL_COUNT.store(0, Ordering::SeqCst);

      let handles: Vec<thread::JoinHandle<()>> = (0..10)
        .map(|_| {
          thread::spawn(|| {
            for _ in 0..100 {
              assert_eq!(DISPATCH.get().name, "counted");
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

  #[cfg(feature = "std")]
  mod generic_dispatcher_tests {
    use super::*;

    type HashFn = fn(&mut [u8; 32], &[u8]);

    fn test_hash(_state: &mut [u8; 32], _data: &[u8]) {}

    fn hash_selector() -> Selected<HashFn> {
      Selected::new("test_hash", test_hash)
    }

    #[test]
    fn test_generic_dispatcher() {
      static DISPATCH: GenericDispatcher<HashFn> = GenericDispatcher::new(hash_selector);
      assert_eq!(DISPATCH.get().name, "test_hash");
      assert_eq!(DISPATCH.backend_name(), "test_hash");
    }
  }
}
