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
//! When target features are known at compile time (e.g., `-C target-cpu=native` or
//! `-C target-feature=+avx512f,+vpclmulqdq`), the dispatch can be resolved to a direct
//! function call with no runtime overhead.
//!
//! # Usage
//!
//! ```ignore
//! use platform::{dispatch_auto, Caps};
//! use platform::caps::x86;
//!
//! pub fn crc32c(crc: u32, data: &[u8]) -> u32 {
//!     dispatch_auto(|caps, tune| {
//!         if caps.has(x86::VPCLMUL_READY) {
//!             vpclmul_kernel(crc, data)
//!         } else if caps.has(x86::PCLMUL_READY) {
//!             pclmul_kernel(crc, data)
//!         } else {
//!             portable_kernel(crc, data)
//!         }
//!     })
//! }
//! ```
//!
//! When compiled with `-C target-feature=+avx512f,+vpclmulqdq`, the entire dispatch
//! collapses to a direct call to `vpclmul_kernel` with no runtime checks.

use crate::{Caps, Tune, detect};

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
/// **Note**: This only returns features known at compile time. Runtime-detected
/// features (e.g., from CPUID) are not included. For full runtime detection,
/// use [`dispatch`] instead.
///
/// # Example
///
/// ```ignore
/// use platform::dispatch_static;
/// use platform::caps::x86;
///
/// // When compiled with -C target-feature=+avx512f
/// let result = dispatch_static(|caps, _tune| {
///     if caps.has(x86::AVX512F) {
///         avx512_path()
///     } else {
///         fallback_path()
///     }
/// });
/// // Compiles to direct call to avx512_path()
/// ```
#[inline(always)]
pub fn dispatch_static<F, R>(f: F) -> R
where
  F: FnOnce(Caps, Tune) -> R,
{
  f(detect::caps_static(), detect::tune_static())
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
/// - First call: performs detection (cold path, microseconds)
/// - Subsequent calls: reads cached value (hot path, nanoseconds)
///
/// # Example
///
/// ```ignore
/// use platform::dispatch;
/// use platform::caps::x86;
///
/// let result = dispatch(|caps, tune| {
///     if caps.has(x86::VPCLMUL_READY) {
///         vpclmul_kernel()
///     } else if caps.has(x86::PCLMUL_READY) {
///         pclmul_kernel()
///     } else {
///         portable_kernel()
///     }
/// });
/// ```
#[inline]
pub fn dispatch<F, R>(f: F) -> R
where
  F: FnOnce(Caps, Tune) -> R,
{
  let det = detect::get();
  f(det.caps, det.tune)
}

/// Automatically choose compile-time or runtime dispatch.
///
/// This function uses compile-time dispatch when "top-tier" features are statically
/// known, falling back to runtime detection otherwise.
///
/// **Top-tier features** (triggers compile-time dispatch):
/// - **x86_64**: AVX-512F + VPCLMULQDQ (modern AVX-512 crypto)
/// - **aarch64**: AES + SHA3 (PMULL + EOR3 for fast carryless multiply)
/// - **loongarch64**: LASX (256-bit SIMD)
///
/// **Use this when**:
/// - You want the best of both worlds
/// - Performance-critical code that may be compiled with or without target features
/// - Library code that should work everywhere but optimize when possible
///
/// # Example
///
/// ```ignore
/// use platform::dispatch_auto;
/// use platform::caps::x86;
///
/// pub fn process(data: &[u8]) -> u64 {
///     dispatch_auto(|caps, _tune| {
///         if caps.has(x86::VPCLMUL_READY) {
///             fast_simd_path(data)
///         } else {
///             portable_path(data)
///         }
///     })
/// }
///
/// // When compiled with -C target-cpu=znver4:
/// // -> dispatch_static is used, entire dispatch is eliminated
/// //
/// // When compiled without target features:
/// // -> dispatch is used, runtime detection selects best path
/// ```
#[inline(always)]
pub fn dispatch_auto<F, R>(f: F) -> R
where
  F: FnOnce(Caps, Tune) -> R,
{
  // x86_64 with AVX-512 crypto features: use static dispatch
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "vpclmulqdq"))]
  {
    return dispatch_static(f);
  }

  // aarch64 with PMULL+EOR3 (SHA3 implies EOR3): use static dispatch
  #[cfg(all(target_arch = "aarch64", target_feature = "aes", target_feature = "sha3"))]
  {
    dispatch_static(f)
  }

  // loongarch64 with LASX (256-bit SIMD): use static dispatch
  #[cfg(all(target_arch = "loongarch64", target_feature = "lasx"))]
  {
    return dispatch_static(f);
  }

  // Fallback: use runtime detection
  #[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "vpclmulqdq"),
    all(target_arch = "aarch64", target_feature = "aes", target_feature = "sha3"),
    all(target_arch = "loongarch64", target_feature = "lasx")
  )))]
  {
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
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "vpclmulqdq"))]
  {
    return true;
  }

  #[cfg(all(target_arch = "aarch64", target_feature = "aes", target_feature = "sha3"))]
  {
    true
  }

  #[cfg(all(target_arch = "loongarch64", target_feature = "lasx"))]
  {
    return true;
  }

  #[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "vpclmulqdq"),
    all(target_arch = "aarch64", target_feature = "aes", target_feature = "sha3"),
    all(target_arch = "loongarch64", target_feature = "lasx")
  )))]
  {
    false
  }
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
    dispatch(|caps, _tune| {
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
    dispatch_static(|caps, _tune| {
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
  fn test_dispatch_auto_works() {
    let result = dispatch_auto(|caps, _tune| {
      // Just verify it returns something
      caps.count()
    });
    // Should have at least baseline features
    #[cfg(target_arch = "x86_64")]
    assert!(result >= 1); // At least SSE2

    #[cfg(target_arch = "aarch64")]
    assert!(result >= 1); // At least NEON
  }

  #[test]
  fn test_dispatch_returns_tune() {
    dispatch(|_caps, tune| {
      // Tune should have reasonable values
      assert!(tune.simd_threshold > 0);
      assert!(tune.parallel_streams >= 1);
      assert!(tune.cache_line > 0);
    });
  }

  #[test]
  fn test_caps_static_is_const() {
    // caps_static should be evaluable at compile time
    const CAPS: Caps = detect::caps_static();

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
  fn test_tune_static_is_const() {
    // tune_static should be evaluable at compile time
    const TUNE: Tune = detect::tune_static();
    const { assert!(TUNE.simd_threshold > 0) };
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

    let result = dispatch(|_caps, _tune| base * multiplier);
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
