//! Zero-overhead dispatch primitives.
//!
//! This module provides three dispatch paths for CPU-feature-aware algorithm selection:
//!
//! - [`dispatch_static`]: Compile-time features only (zero overhead)
//! - [`dispatch`]: Runtime detection (cached)
//! - [`dispatch_auto`]: Automatically chooses best path
//!
//! # Zero-Overhead Dispatch
//!
//! When target features are known at compile time beyond the architecture baseline
//! (e.g., `-C target-cpu=native`, `-C target-feature=+sse4.2,+pclmulqdq`, or
//! `-C target-feature=+aes` on aarch64), the dispatch can be resolved to a direct
//! function call with no runtime overhead.
//!
//! # Examples
//!
//! ```
//! use platform::dispatch_auto;
//!
//! let count = dispatch_auto(|caps| caps.count());
//! assert!(count >= 1);
//! ```

use crate::{Caps, detect};

#[inline(always)]
#[must_use]
#[allow(unreachable_code)] // cfg branches may return early on some targets.
const fn static_dispatch_baseline() -> Caps {
  #[cfg(target_arch = "x86_64")]
  {
    return crate::caps::x86::SSE2;
  }

  #[cfg(target_arch = "aarch64")]
  {
    return crate::caps::aarch64::NEON;
  }

  Caps::NONE
}

/// Shared decision graph for choosing the dispatch path.
///
/// Keeping this in one place guarantees `dispatch_auto()` and
/// `has_static_features()` always agree.
#[inline(always)]
#[must_use]
const fn use_static_dispatch_path() -> bool {
  !detect::caps_static().difference(static_dispatch_baseline()).is_empty()
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Dispatch with compile-time known capabilities only.
///
/// This function has zero runtime overhead when target features are known at compile
/// time (e.g., `-C target-cpu=native`). The compiler can inline everything and
/// eliminate dead branches via constant propagation.
///
/// **Use this when**:
/// - You're compiling with specific target features enabled
/// - You want guaranteed zero-overhead dispatch
/// - You're building a specialized binary for a known target
///
/// **Note**: Only returns features known at compile time. For runtime
/// detection, use [`dispatch`].
///
/// # Examples
///
/// ```
/// use platform::dispatch_static;
///
/// let count = dispatch_static(|caps| caps.count());
/// // count is known at compile time
/// ```
#[inline(always)]
pub fn dispatch_static<F, R>(f: F) -> R
where
  F: FnOnce(Caps) -> R,
{
  f(detect::caps_static())
}

/// Dispatch with runtime-detected capabilities.
///
/// Uses cached runtime detection. On first call, performs CPU feature detection
/// and caches the result. Subsequent calls use the cached value.
///
/// **Use this when**:
/// - You're building a generic binary that runs on multiple CPU types
/// - You need to detect runtime features (e.g., via CPUID)
/// - You want the full set of available features
///
/// # Overhead
///
/// - First call: ~1μs (detection)
/// - Subsequent: ~3ns (cached)
///
/// # Examples
///
/// ```
/// use platform::dispatch;
///
/// let feature_count = dispatch(|caps| caps.count());
/// assert!(feature_count >= 1);
/// ```
#[inline]
pub fn dispatch<F, R>(f: F) -> R
where
  F: FnOnce(Caps) -> R,
{
  let det = detect::get();
  f(det.caps)
}

/// Automatically choose compile-time or runtime dispatch.
///
/// This function uses compile-time dispatch when the build enables target
/// features beyond the architecture baseline, falling back to runtime detection
/// for generic baseline builds.
///
/// **Triggers compile-time dispatch**:
/// - **x86/x86_64**: Any compile-time feature beyond the x86_64 SSE2 baseline (for example, SSE4.2,
///   PCLMULQDQ, AVX2, or AVX-512)
/// - **aarch64**: Any compile-time feature beyond the NEON baseline (for example, AES/PMULL, SHA2,
///   SHA3, or SVE)
/// - **Other architectures**: Any compile-time target feature exposed by [`detect::caps_static`]
///
/// **Use this when**: Library code that should work everywhere but
/// optimize when compiled with target features.
///
/// # Examples
///
/// ```
/// use platform::dispatch_auto;
///
/// let feature_count = dispatch_auto(|caps| caps.count());
/// assert!(feature_count >= 1);
/// ```
#[inline(always)]
pub fn dispatch_auto<F, R>(f: F) -> R
where
  F: FnOnce(Caps) -> R,
{
  if use_static_dispatch_path() {
    dispatch_static(f)
  } else {
    dispatch(f)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Check if compile-time features are sufficient for static dispatch.
///
/// Returns `true` if [`dispatch_auto`] would use [`dispatch_static`] on this target.
#[inline(always)]
#[must_use]
pub const fn has_static_features() -> bool {
  use_static_dispatch_path()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[cfg(not(miri))] // dispatch() returns portable caps under Miri
  fn test_dispatch_returns_caps() {
    dispatch(|caps| {
      #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
      let _ = caps;

      // Should have at least baseline features for the architecture
      #[cfg(target_arch = "x86_64")]
      {
        use crate::caps::x86;
        assert!(caps.has(x86::SSE2));
      }

      #[cfg(target_arch = "aarch64")]
      {
        use crate::caps::aarch64;
        assert!(caps.has(aarch64::NEON));
      }
    });
  }

  #[test]
  fn test_dispatch_static_returns_caps() {
    dispatch_static(|caps| {
      #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
      let _ = caps;

      // Should have at least baseline features for the architecture
      #[cfg(target_arch = "x86_64")]
      {
        use crate::caps::x86;
        assert!(caps.has(x86::SSE2));
      }

      #[cfg(target_arch = "aarch64")]
      {
        use crate::caps::aarch64;
        assert!(caps.has(aarch64::NEON));
      }
    });
  }

  #[test]
  #[cfg_attr(miri, ignore)] // Miri returns Caps::NONE, no runtime detection
  fn test_dispatch_auto_works() {
    let result = dispatch_auto(|caps| {
      // Just verify it returns something
      caps.count()
    });
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = result;

    // Should have at least baseline features
    #[cfg(target_arch = "x86_64")]
    assert!(result >= 1); // At least SSE2

    #[cfg(target_arch = "aarch64")]
    assert!(result >= 1); // At least NEON
  }

  #[test]
  fn test_caps_static_is_const() {
    // caps_static should be evaluable at compile time
    const CAPS: Caps = detect::caps_static();
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = CAPS;

    #[cfg(target_arch = "x86_64")]
    {
      use crate::caps::x86;
      assert!(CAPS.has(x86::SSE2));
    }

    #[cfg(target_arch = "aarch64")]
    {
      use crate::caps::aarch64;
      assert!(CAPS.has(aarch64::NEON));
    }
  }

  #[test]
  fn test_has_static_features_is_const() {
    // Should be evaluable at compile time
    const HAS_STATIC: bool = has_static_features();
    // Just verify it compiles and returns a bool
    let _ = HAS_STATIC;
  }

  #[test]
  fn test_dispatch_closure_captures() {
    let multiplier = 42u32;
    let base = 10u32;

    let result = dispatch(|_caps| base * multiplier);
    assert_eq!(result, 420);
  }

  #[test]
  #[cfg(not(miri))] // Runtime caps returns NONE under Miri
  fn test_dispatch_static_subset_of_runtime() {
    // Static caps should be a subset of runtime caps
    let static_caps = detect::caps_static();
    let runtime_caps = detect::caps();

    // Every bit set in static_caps should also be set in runtime_caps
    assert!(runtime_caps.has(static_caps));
  }
}
