//! Kernel dispatch: selection and caching.
//!
//! This module provides the core dispatch primitives for rscrypto:
//!
//! - [`Candidate`] / [`TunedCandidate`]: Kernel with capability requirements
//! - [`Selected`]: Result of kernel selection
//! - [`try_select`] / [`try_select_tuned`]: Typed non-panicking selection
//! - [`select_or`] / [`select_tuned_or`]: Total selection with explicit fallback
//! - [`candidates!`] / [`tuned_candidates!`]: Macros for defining candidate lists
//! - `define_dispatcher!`: Macro for generating cached dispatcher types
//!
//! # Design
//!
//! The dispatch system has two paths:
//!
//! 1. **Compile-time** (zero-cost): When target features are known at compile time, dispatch
//!    resolves to a direct function call with no overhead.
//!
//! 2. **Runtime** (cached): For generic binaries, the dispatcher detects CPU features once and
//!    caches the selected kernel. Subsequent calls are a single indirect call.

use platform::{Caps, Tune};

// ─────────────────────────────────────────────────────────────────────────────
// Candidate List Macro
// ─────────────────────────────────────────────────────────────────────────────

/// Creates a static slice of [`Candidate`]s for kernel dispatch.
///
/// Candidates are ordered best-to-worst. The dispatcher selects the first
/// candidate whose capability requirements are satisfied.
///
/// # Syntax
///
/// ```text
/// candidates![
///     "name" => CAPS => kernel_fn,
///     "fallback" => Caps::NONE => fallback_fn,  // Always include a fallback!
/// ]
/// ```
///
/// # Example
///
/// ```ignore
/// use backend::{candidates, caps::Caps};
///
/// let kernels = candidates![
///     "x86_64/vpclmul" => x86::VPCLMUL_READY => vpclmul_kernel,
///     "x86_64/pclmul" => x86::PCLMUL_READY => pclmul_kernel,
///     "portable" => Caps::NONE => portable_kernel,
/// ];
/// ```
#[macro_export]
macro_rules! candidates {
  // Match one or more: name => caps => func, with optional trailing comma
  //
  // Note: We use `$func as _` to coerce function items to function pointers.
  // This is necessary because each fn item in Rust has a unique zero-sized type,
  // even if they share the same signature. Without coercion, the array literal
  // would fail to compile when mixing different functions.
  [ $( $name:literal => $caps:expr => $func:expr ),+ $(,)? ] => {
    &[
      $(
        $crate::dispatch::Candidate::new($name, $caps, $func as _),
      )+
    ]
  };
}

// Re-export at crate root for ergonomic imports
pub use candidates;

// ─────────────────────────────────────────────────────────────────────────────
// Tuned Candidate List Macro
// ─────────────────────────────────────────────────────────────────────────────

/// Creates a static slice of [`TunedCandidate`]s with optional tuning predicates.
///
/// Like [`candidates!`], but entries may include a `where` clause that filters
/// candidates based on [`platform::Tune`]. This enables microarchitecture-aware
/// kernel selection.
///
/// # Syntax
///
/// ```text
/// tuned_candidates![
///     "name" => CAPS => kernel_fn,
///     "name" => CAPS, where |t| t.fast_wide_ops => wide_kernel_fn,
///     "fallback" => Caps::NONE => fallback_fn,
/// ]
/// ```
///
/// # Example
///
/// ```ignore
/// let kernels = tuned_candidates![
///     "x86_64/vpclmul" => x86::VPCLMUL_READY, where |t| t.effective_simd_width >= 256 => vpclmul,
///     "x86_64/pclmul" => x86::PCLMUL_READY => pclmul,
///     "portable" => Caps::NONE => portable,
/// ];
/// ```
#[macro_export]
macro_rules! tuned_candidates {
  [ $( $name:literal => $caps:expr $(, where $pred:expr)? => $func:expr ),+ $(,)? ] => {
    &[
      $(
        $crate::dispatch::TunedCandidate::new(
          $name,
          $caps,
          $crate::tuned_candidates!(@pred $( $pred )?),
          $func as _,
        ),
      )+
    ]
  };

  (@pred) => { $crate::dispatch::always_tuned };
  (@pred $pred:expr) => { $pred as fn(&$crate::platform::Tune) -> bool };
}

// Re-export at crate root for ergonomic imports
pub use tuned_candidates;

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
  pub requires: Caps,
  /// The kernel function pointer.
  pub func: F,
}

impl<F> Candidate<F> {
  /// Create a new candidate.
  #[inline]
  #[must_use]
  pub const fn new(name: &'static str, requires: Caps, func: F) -> Self {
    Self { name, requires, func }
  }
}

/// A tuned candidate kernel with capability requirements and a tuning predicate.
///
/// Candidates are ordered from best to worst. The dispatcher selects the first
/// candidate whose requirements are satisfied by the detected capabilities and
/// whose predicate returns `true` for the detected tuning preset.
#[derive(Clone, Copy, Debug)]
pub struct TunedCandidate<F> {
  /// Human-readable name for diagnostics (e.g., "x86_64/vpclmul").
  pub name: &'static str,
  /// Required CPU capabilities. Must be a subset of detected caps.
  pub requires: Caps,
  /// Tuning predicate: must return true for this candidate to be eligible.
  pub predicate: fn(&Tune) -> bool,
  /// The kernel function pointer.
  pub func: F,
}

impl<F> TunedCandidate<F> {
  /// Create a new tuned candidate.
  #[inline]
  #[must_use]
  pub const fn new(name: &'static str, requires: Caps, predicate: fn(&Tune) -> bool, func: F) -> Self {
    Self {
      name,
      requires,
      predicate,
      func,
    }
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

/// Typed selection failure from candidate lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionError {
  /// Candidate list was empty.
  EmptyCandidateList,
  /// No candidate requirements matched the detected capabilities.
  NoMatchingCandidate,
}

/// Select the best kernel from a candidate list.
///
/// Returns the first candidate whose `requires` is satisfied by `caps`.
///
/// This API is non-panicking and returns a typed error if selection fails.
#[inline(always)]
pub fn try_select<F: Copy>(caps: Caps, candidates: &[Candidate<F>]) -> Result<Selected<F>, SelectionError> {
  if candidates.is_empty() {
    return Err(SelectionError::EmptyCandidateList);
  }

  for candidate in candidates {
    if caps.has(candidate.requires) {
      return Ok(Selected::new(candidate.name, candidate.func));
    }
  }
  Err(SelectionError::NoMatchingCandidate)
}

/// Select the best kernel from a candidate list with a total fallback guarantee.
///
/// Returns the first matching candidate; if none match, returns `fallback`.
#[inline(always)]
#[must_use]
pub fn select_or<F: Copy>(caps: Caps, candidates: &[Candidate<F>], fallback: Candidate<F>) -> Selected<F> {
  if let Ok(selected) = try_select(caps, candidates) {
    selected
  } else {
    Selected::new(fallback.name, fallback.func)
  }
}

/// Predicate used for tuned candidates without an explicit `where` clause.
#[inline(always)]
#[must_use]
pub fn always_tuned(_: &Tune) -> bool {
  true
}

/// Select the best kernel from a tuned candidate list.
///
/// Returns the first candidate whose `requires` is satisfied by `caps` and whose
/// tuning predicate returns `true` for `tune`.
///
/// This API is non-panicking and returns a typed error if selection fails.
#[inline(always)]
pub fn try_select_tuned<F: Copy>(
  caps: Caps,
  tune: &Tune,
  candidates: &[TunedCandidate<F>],
) -> Result<Selected<F>, SelectionError> {
  if candidates.is_empty() {
    return Err(SelectionError::EmptyCandidateList);
  }

  for candidate in candidates {
    if caps.has(candidate.requires) && (candidate.predicate)(tune) {
      return Ok(Selected::new(candidate.name, candidate.func));
    }
  }
  Err(SelectionError::NoMatchingCandidate)
}

/// Select the best tuned candidate with a total fallback guarantee.
///
/// Returns the first matching candidate; if none match, returns `fallback`.
#[inline(always)]
#[must_use]
pub fn select_tuned_or<F: Copy>(
  caps: Caps,
  tune: &Tune,
  candidates: &[TunedCandidate<F>],
  fallback: TunedCandidate<F>,
) -> Selected<F> {
  if let Ok(selected) = try_select_tuned(caps, tune, candidates) {
    selected
  } else {
    Selected::new(fallback.name, fallback.func)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Macro
// ─────────────────────────────────────────────────────────────────────────────

/// Generates a dispatcher type with caching for a specific function signature.
///
/// Algorithm crates use this to define their dispatchers. The backend crate
/// provides the infrastructure; algorithm crates define the concrete types.
///
/// # Generated API
///
/// - `new(selector)` - Create with a selector function
/// - `get()` - Get the selected kernel (cached after first call)
/// - `kernel()` - Get the function pointer directly
/// - `backend_name()` - Get the name of the selected backend
/// - `call(state, data)` - Call the kernel with given arguments
///
/// # Thread Safety
///
/// Dispatchers are `Send + Sync` and can be used from multiple threads:
/// - **std**: Uses `OnceCache` backed by `OnceLock`
/// - **no_std + atomics**: Uses `OnceCache` with atomic state machine
/// - **no_std - atomics**: Per-call computation (single-threaded targets)
///
/// # Example
///
/// ```ignore
/// define_dispatcher!(
///     /// CRC-32 dispatcher with cached kernel selection.
///     Crc32Dispatcher, fn(u32, &[u8]) -> u32, u32
/// );
/// ```
#[macro_export]
macro_rules! define_dispatcher {
  (
    $(#[$meta:meta])*
    $name:ident, $fn_ty:ty, $state:ty
  ) => {
    $(#[$meta])*
    pub struct $name {
      /// Cached selection result.
      cache: $crate::cache::OnceCache<$crate::dispatch::Selected<$fn_ty>>,
      /// Selector function called on first access.
      selector: fn() -> $crate::dispatch::Selected<$fn_ty>,
    }

    impl $name {
      /// Create a new dispatcher with the given selector function.
      #[must_use]
      pub const fn new(selector: fn() -> $crate::dispatch::Selected<$fn_ty>) -> Self {
        Self {
          cache: $crate::cache::OnceCache::new(),
          selector,
        }
      }

      /// Get the selected kernel, initializing on first call.
      ///
      /// The selector is called at most once (on targets with atomics).
      /// Subsequent calls return the cached result.
      #[inline]
      #[must_use]
      pub fn get(&self) -> $crate::dispatch::Selected<$fn_ty> {
        self.cache.get_or_init(|| (self.selector)())
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
        let kernel = self.kernel();
        kernel(state, data)
      }

      /// Get the selected function pointer.
      #[inline]
      #[must_use]
      pub fn kernel(&self) -> $fn_ty {
        self.get().func
      }
    }

    // Send + Sync are inherited from OnceCache<Selected<F>>:
    // - std: OnceLock<T> is Send+Sync when T: Send+Sync
    // - no_std+atomics: OnceCache has unsafe impl Sync
    // - no_std without atomics: OnceCache is !Send+!Sync (single-threaded targets)
    //
    // Selected<F> where F is a fn pointer: fn pointers are Send+Sync,
    // &'static str is Send+Sync, so Selected<F> is automatically Send+Sync.
    // No manual unsafe impl needed!
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic Dispatcher (std only, for custom signatures)
// ─────────────────────────────────────────────────────────────────────────────

/// Generic dispatcher for arbitrary function signatures.
///
/// Use this for hash functions, MACs, or other algorithms with custom signatures.
/// For CRC algorithms, prefer the typed dispatchers above.
///
/// # Thread Safety
///
/// `GenericDispatcher` is `Send + Sync` when `F: Send + Sync`:
/// - `OnceLock<Selected<F>>` auto-derives `Send + Sync` when `Selected<F>: Send + Sync`
/// - `Selected<F>` is `Send + Sync` when `F: Send + Sync` (fn pointers always are)
/// - No manual `unsafe impl` needed!
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

// No manual unsafe impl Send/Sync needed - OnceLock<Selected<F>> auto-derives both
// when Selected<F>: Send + Sync, which holds when F: Send + Sync (always true for fn pointers).

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::std_instead_of_alloc, clippy::std_instead_of_core)]
mod tests {
  use super::*;

  // Test-local type aliases (dispatchers are owned by algorithm crates, not backend)
  type Crc32Fn = fn(u32, &[u8]) -> u32;
  type Crc64Fn = fn(u64, &[u8]) -> u64;

  // Test-local dispatchers for verifying the macro works
  define_dispatcher!(TestCrc32Dispatcher, Crc32Fn, u32);
  define_dispatcher!(TestCrc64Dispatcher, Crc64Fn, u64);

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
    let c: Candidate<Crc32Fn> = Candidate::new("test", Caps::NONE, portable_crc32);
    assert_eq!(c.name, "test");
    assert_eq!(c.requires, Caps::NONE);
  }

  #[test]
  fn test_select_portable_fallback() {
    let caps = Caps::NONE;
    let candidates: &[Candidate<Crc32Fn>] = &[
      Candidate::new("fast", Caps::bit(0), fast_crc32),
      Candidate::new("portable", Caps::NONE, portable_crc32),
    ];
    let selected = try_select(caps, candidates).expect("portable fallback should always match");
    assert_eq!(selected.name, "portable");
  }

  #[test]
  fn test_select_best_match() {
    let caps = Caps::bit(0);
    let candidates: &[Candidate<Crc32Fn>] = &[
      Candidate::new("fast", Caps::bit(0), fast_crc32),
      Candidate::new("portable", Caps::NONE, portable_crc32),
    ];
    let selected = try_select(caps, candidates).expect("fast candidate should match");
    assert_eq!(selected.name, "fast");
  }

  #[test]
  fn test_select_first_match_wins() {
    let caps = Caps::bit(0).union(Caps::bit(1));
    let candidates: &[Candidate<Crc32Fn>] = &[
      Candidate::new("first", Caps::bit(0), fast_crc32),
      Candidate::new("second", Caps::bit(1), fastest_crc32),
      Candidate::new("portable", Caps::NONE, portable_crc32),
    ];
    let selected = try_select(caps, candidates).expect("first matching candidate should be selected");
    assert_eq!(selected.name, "first");
  }

  #[test]
  fn test_select_no_fallback_returns_error() {
    let caps = Caps::NONE;
    let candidates: &[Candidate<Crc32Fn>] = &[Candidate::new("needs_bit0", Caps::bit(0), fast_crc32)];
    let err = try_select(caps, candidates).expect_err("selection should fail without fallback");
    assert_eq!(err, SelectionError::NoMatchingCandidate);
  }

  #[test]
  fn test_select_or_total_fallback() {
    let caps = Caps::NONE;
    let candidates: &[Candidate<Crc32Fn>] = &[Candidate::new("needs_bit0", Caps::bit(0), fast_crc32)];
    let fallback: Candidate<Crc32Fn> = Candidate::new("portable", Caps::NONE, portable_crc32 as Crc32Fn);
    let selected = select_or(caps, candidates, fallback);
    assert_eq!(selected.name, "portable");
  }

  fn test_selector() -> Selected<Crc32Fn> {
    Selected::new("test", portable_crc32)
  }

  #[test]
  fn test_crc32_dispatcher() {
    static DISPATCH: TestCrc32Dispatcher = TestCrc32Dispatcher::new(test_selector);
    let selected = DISPATCH.get();
    assert_eq!(selected.name, "test");
    assert_eq!(DISPATCH.backend_name(), "test");
    assert_eq!(DISPATCH.call(0, &[]), 0xDEADBEEF);
  }

  fn test_selector_64() -> Selected<Crc64Fn> {
    Selected::new("test64", portable_crc64)
  }

  #[test]
  fn test_crc64_dispatcher() {
    static DISPATCH: TestCrc64Dispatcher = TestCrc64Dispatcher::new(test_selector_64);
    let selected = DISPATCH.get();
    assert_eq!(selected.name, "test64");
    assert_eq!(DISPATCH.backend_name(), "test64");
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

      static DISPATCH: TestCrc32Dispatcher = TestCrc32Dispatcher::new(counting_selector);
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

  // ─────────────────────────────────────────────────────────────────────────────
  // candidates! Macro Tests
  // ─────────────────────────────────────────────────────────────────────────────

  mod candidates_macro_tests {
    use super::*;

    // ─── Basic Usage ───

    #[test]
    fn test_candidates_macro_basic() {
      let list: &[Candidate<Crc32Fn>] = candidates![
        "fast" => Caps::bit(0) => fast_crc32,
        "portable" => Caps::NONE => portable_crc32,
      ];

      assert_eq!(list.len(), 2);
      assert_eq!(list[0].name, "fast");
      assert_eq!(list[0].requires, Caps::bit(0));
      assert_eq!(list[1].name, "portable");
      assert_eq!(list[1].requires, Caps::NONE);
    }

    #[test]
    fn test_candidates_macro_single_entry() {
      let list: &[Candidate<Crc32Fn>] = candidates![
        "only" => Caps::NONE => portable_crc32,
      ];

      assert_eq!(list.len(), 1);
      assert_eq!(list[0].name, "only");
    }

    #[test]
    fn test_candidates_macro_three_entries() {
      let list: &[Candidate<Crc32Fn>] = candidates![
        "fastest" => Caps::bit(1) => fastest_crc32,
        "fast" => Caps::bit(0) => fast_crc32,
        "portable" => Caps::NONE => portable_crc32,
      ];

      assert_eq!(list.len(), 3);
      assert_eq!(list[0].name, "fastest");
      assert_eq!(list[1].name, "fast");
      assert_eq!(list[2].name, "portable");
    }

    // ─── Trailing Comma Handling ───

    #[test]
    fn test_candidates_macro_trailing_comma() {
      // With trailing comma
      let with_trailing: &[Candidate<Crc32Fn>] = candidates![
        "a" => Caps::NONE => portable_crc32,
        "b" => Caps::NONE => portable_crc32,
      ];

      // Without trailing comma
      let without_trailing: &[Candidate<Crc32Fn>] = candidates![
        "a" => Caps::NONE => portable_crc32,
        "b" => Caps::NONE => portable_crc32
      ];

      assert_eq!(with_trailing.len(), without_trailing.len());
    }

    #[test]
    fn test_candidates_macro_single_no_trailing() {
      let list: &[Candidate<Crc32Fn>] = candidates![
        "only" => Caps::NONE => portable_crc32
      ];
      assert_eq!(list.len(), 1);
    }

    // ─── Integration with try_select() ───

    #[test]
    fn test_candidates_macro_with_select_fallback() {
      let caps = Caps::NONE;
      let selected: Selected<Crc32Fn> = try_select(
        caps,
        candidates![
          "fast" => Caps::bit(0) => fast_crc32,
          "portable" => Caps::NONE => portable_crc32,
        ],
      )
      .expect("portable fallback should match");

      assert_eq!(selected.name, "portable");
      assert_eq!((selected.func)(0, &[]), 0xDEADBEEF);
    }

    #[test]
    fn test_candidates_macro_with_select_match() {
      let caps = Caps::bit(0);
      let selected: Selected<Crc32Fn> = try_select(
        caps,
        candidates![
          "fast" => Caps::bit(0) => fast_crc32,
          "portable" => Caps::NONE => portable_crc32,
        ],
      )
      .expect("fast candidate should match");

      assert_eq!(selected.name, "fast");
      assert_eq!((selected.func)(0, &[]), 0xCAFEBABE);
    }

    #[test]
    fn test_candidates_macro_with_select_priority() {
      // When multiple candidates match, first one wins
      let caps = Caps::bit(0) | Caps::bit(1);
      let selected: Selected<Crc32Fn> = try_select(
        caps,
        candidates![
          "first" => Caps::bit(0) => fast_crc32,
          "second" => Caps::bit(1) => fastest_crc32,
          "portable" => Caps::NONE => portable_crc32,
        ],
      )
      .expect("first matching candidate should be selected");

      assert_eq!(selected.name, "first");
    }

    // ─── Different Function Signatures ───

    #[test]
    fn test_candidates_macro_crc64() {
      let list: &[Candidate<Crc64Fn>] = candidates![
        "fast64" => Caps::bit(0) => portable_crc64,
        "portable64" => Caps::NONE => portable_crc64,
      ];

      assert_eq!(list.len(), 2);
      assert_eq!(list[0].name, "fast64");
    }

    #[test]
    fn test_candidates_macro_custom_signature() {
      type CustomFn = fn(u128) -> u128;

      fn custom_impl(x: u128) -> u128 {
        x.wrapping_mul(2)
      }

      let list: &[Candidate<CustomFn>] = candidates![
        "custom" => Caps::NONE => custom_impl,
      ];

      assert_eq!(list.len(), 1);
      assert_eq!((list[0].func)(21), 42);
    }

    // ─── Complex Caps Expressions ───

    #[test]
    fn test_candidates_macro_complex_caps() {
      let combined = Caps::bit(0) | Caps::bit(1);
      let list: &[Candidate<Crc32Fn>] = candidates![
        "needs_both" => combined => fast_crc32,
        "needs_one" => Caps::bit(0) => fast_crc32,
        "portable" => Caps::NONE => portable_crc32,
      ];

      assert_eq!(list.len(), 3);
      assert!(list[0].requires.has(Caps::bit(0)));
      assert!(list[0].requires.has(Caps::bit(1)));
    }

    #[test]
    fn test_candidates_macro_with_union_inline() {
      let list: &[Candidate<Crc32Fn>] = candidates![
        "combined" => Caps::bit(0).union(Caps::bit(1)) => fast_crc32,
        "portable" => Caps::NONE => portable_crc32,
      ];

      assert_eq!(list[0].requires.count(), 2);
    }

    // ─── Naming Conventions ───

    #[test]
    fn test_candidates_macro_path_style_names() {
      let list: &[Candidate<Crc32Fn>] = candidates![
        "x86_64/vpclmul" => Caps::bit(0) => fast_crc32,
        "x86_64/pclmul" => Caps::bit(1) => fast_crc32,
        "aarch64/pmull" => Caps::bit(64) => fast_crc32,
        "portable" => Caps::NONE => portable_crc32,
      ];

      assert_eq!(list[0].name, "x86_64/vpclmul");
      assert_eq!(list[1].name, "x86_64/pclmul");
      assert_eq!(list[2].name, "aarch64/pmull");
      assert_eq!(list[3].name, "portable");
    }

    // ─── Static Context ───

    #[test]
    fn test_candidates_macro_in_static_context() {
      fn make_selector() -> Selected<Crc32Fn> {
        select_or(
          Caps::NONE,
          candidates![
            "fast" => Caps::bit(0) => fast_crc32,
            "portable" => Caps::NONE => portable_crc32,
          ],
          Candidate::new("portable", Caps::NONE, portable_crc32),
        )
      }

      static DISPATCHER: TestCrc32Dispatcher = TestCrc32Dispatcher::new(make_selector);

      let selected = DISPATCHER.get();
      assert_eq!(selected.name, "portable");
    }
  }
}
