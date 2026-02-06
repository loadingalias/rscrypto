//! CPU feature detection.
//!
//! This module provides runtime CPU capability detection with caching.
//! It handles compile-time and runtime detection, caching, and user overrides.
//!
//! # Detection Tiers
//!
//! 1. **Compile-time**: `cfg!(target_feature)` - zero cost, dead code elimination
//! 2. **Runtime (std)**: `is_x86_feature_detected!` + `OnceLock` caching
//! 3. **Runtime (no_std)**: Atomic-based caching for embedded targets
//!
//! # Override Support
//!
//! ```
//! use platform::Detected;
//! platform::set_override(Some(Detected::portable()));
//! platform::clear_override();
//! ```

use crate::{
  caps::{Arch, Caps},
  tune::Tune,
};

/// Errors when configuring runtime detection overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverrideError {
  /// Detection has already been initialized; override updates are no longer allowed.
  AlreadyInitialized,
  /// Overrides are unsupported on this target configuration.
  Unsupported,
}

impl core::fmt::Display for OverrideError {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      Self::AlreadyInitialized => f.write_str("detection cache already initialized"),
      Self::Unsupported => f.write_str("override unsupported on this target"),
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main API
// ─────────────────────────────────────────────────────────────────────────────

/// Detected CPU state: capabilities and tuning hints.
///
/// This struct combines all detection results:
/// - `caps`: Available CPU features (what instructions can run)
/// - `tune`: Microarchitecture-specific tuning hints (what's optimal)
/// - `arch`: Target architecture identifier
///
/// Use [`get()`] to obtain a cached instance, or [`detect_uncached()`] for
/// fresh detection (useful for testing/benchmarking).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Detected {
  /// CPU feature capabilities bitset.
  pub caps: Caps,
  /// Microarchitecture-specific tuning hints.
  pub tune: Tune,
  /// Target architecture identifier.
  pub arch: Arch,
}

impl Detected {
  /// Create a portable fallback detection result.
  ///
  /// Returns a conservative configuration with no SIMD features enabled.
  /// Used as a fallback when:
  /// - Running under Miri (which cannot interpret SIMD intrinsics)
  /// - On unsupported architectures
  /// - When detection fails
  #[inline]
  #[must_use]
  pub const fn portable() -> Self {
    Self {
      caps: Caps::NONE,
      tune: Tune::PORTABLE,
      arch: Arch::Other,
    }
  }
}

/// Get detected CPU capabilities and tuning hints.
///
/// Results are cached after first call.
///
/// # Examples
///
/// ```
/// let det = platform::detect::get();
/// assert!(det.caps.count() >= 1);
/// ```
#[inline]
#[must_use]
pub fn get() -> Detected {
  // Miri cannot interpret SIMD intrinsics
  #[cfg(miri)]
  {
    return Detected::portable();
  }

  #[cfg(not(miri))]
  {
    #[cfg(feature = "std")]
    {
      *STD_CACHE.get_or_init(detect_with_override)
    }

    #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
    {
      atomic_cache::get_or_init(detect_with_override)
    }

    #[cfg(all(not(feature = "std"), not(target_has_atomic = "64")))]
    {
      // Constrained targets: always call detect (no caching)
      detect_with_override()
    }
  }
}

/// Get just the capabilities.
#[inline]
#[must_use]
pub fn caps() -> Caps {
  get().caps
}

/// Get just the tuning hints.
#[inline]
#[must_use]
pub fn tune() -> Tune {
  get().tune
}

/// Get the detected architecture.
#[inline]
#[must_use]
pub fn arch() -> Arch {
  get().arch
}

include!("detect/compile_time.rs");
include!("detect/cache_override.rs");

// ─────────────────────────────────────────────────────────────────────────────
// Uncached Detection
// ─────────────────────────────────────────────────────────────────────────────

/// Detect capabilities without caching (for testing/benchmarking).
#[cold]
#[must_use]
pub fn detect_uncached() -> Detected {
  #[cfg(target_arch = "x86_64")]
  {
    detect_x86_64()
  }

  #[cfg(target_arch = "x86")]
  {
    detect_x86()
  }

  #[cfg(target_arch = "aarch64")]
  {
    detect_aarch64()
  }

  #[cfg(target_arch = "riscv64")]
  {
    detect_riscv64()
  }

  #[cfg(target_arch = "riscv32")]
  {
    detect_riscv32()
  }

  #[cfg(target_arch = "s390x")]
  {
    detect_s390x()
  }

  #[cfg(target_arch = "powerpc64")]
  {
    detect_power()
  }

  #[cfg(target_arch = "wasm32")]
  {
    detect_wasm32()
  }

  #[cfg(target_arch = "wasm64")]
  {
    detect_wasm64()
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "x86",
    target_arch = "aarch64",
    target_arch = "riscv64",
    target_arch = "riscv32",
    target_arch = "s390x",
    target_arch = "powerpc64",
    target_arch = "wasm32",
    target_arch = "wasm64"
  )))]
  {
    Detected::portable()
  }
}

include!("detect/arch/x86.rs");
include!("detect/arch/aarch64.rs");
include!("detect/arch/riscv.rs");
include!("detect/arch/s390x.rs");
include!("detect/arch/power.rs");
include!("detect/arch/wasm.rs");
include!("detect/tests.rs");
