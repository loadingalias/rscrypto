//! CPU detection, capabilities, and tuning for maximum performance.
//!
//! Unified platform abstraction for rscrypto: detects CPU features, selects optimal
//! algorithms, and provides microarchitecture-specific tuning—all from a single API.
//!
//! # Quick Start
//!
//! ```
//! use platform::{Caps, Tune, dispatch_auto};
//!
//! // Automatic dispatch: zero-overhead when compiled with target features,
//! // runtime detection otherwise
//! let result = dispatch_auto(|caps, tune| {
//!   // `caps` tells you what's possible (which instructions exist)
//!   // `tune` tells you what's optimal (thresholds, strategies)
//!   assert!(tune.simd_threshold > 0);
//!   assert!(tune.cache_line >= 32);
//!   caps.count() // returns number of detected features
//! });
//! assert!(result >= 1); // At least one feature detected
//! ```
//!
//! # Three Dispatch Modes
//!
//! | Function | When to Use | Overhead |
//! |----------|-------------|----------|
//! | [`dispatch_static`] | Compiled with `-C target-cpu=native` | Zero |
//! | [`dispatch()`] | Generic binary, need runtime detection | Nanoseconds (cached) |
//! | [`dispatch_auto`] | Best of both: static when possible, runtime fallback | Optimal |
//!
//! # For Checksum/Hash Implementations
//!
//! ```
//! # #[cfg(target_arch = "aarch64")]
//! # fn example() {
//! use platform::{Caps, Tune, caps::aarch64, dispatch_auto};
//!
//! fn crc32c(crc: u32, data: &[u8]) -> u32 {
//!   dispatch_auto(|caps, tune| {
//!     // Use tuning to decide scalar vs SIMD threshold
//!     if data.len() < tune.simd_threshold {
//!       scalar_crc32c(crc, data)
//!     }
//!     // Use caps to select best available kernel
//!     else if caps.has(aarch64::PMULL_EOR3_READY) {
//!       pmull_eor3_kernel(crc, data) // Apple M1+, Graviton3+
//!     } else if caps.has(aarch64::AES) {
//!       pmull_kernel(crc, data) // Most ARMv8
//!     } else {
//!       scalar_crc32c(crc, data) // Fallback
//!     }
//!   })
//! }
//! # fn scalar_crc32c(_: u32, _: &[u8]) -> u32 { 0 }
//! # fn pmull_eor3_kernel(_: u32, _: &[u8]) -> u32 { 0 }
//! # fn pmull_kernel(_: u32, _: &[u8]) -> u32 { 0 }
//! # }
//! ```
//!
//! # Design
//!
//! - **[`Caps`]**: 256-bit feature bitset. "What instructions can run?"
//! - **[`Tune`]**: Microarchitecture hints. "What thresholds/strategies are optimal?"
//! - **[`caps_static`]**: Compile-time detection via `cfg!()`. Zero overhead.
//! - **[`caps()`]**: Runtime detection via CPUID/HWCAP. Cached in `OnceLock`.
//!
//! # Performance
//!
//! - Compile-time path: **0ns** (eliminated by optimizer)
//! - Runtime path: **~3ns** (atomic load of cached `Detected`)
//! - First detection: **~1μs** (CPUID/sysctl, happens once)
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

// ─────────────────────────────────────────────────────────────────────────────
// Core modules
// ─────────────────────────────────────────────────────────────────────────────

pub mod caps;
pub mod detect;
pub mod dispatch;
pub mod tune;

// ─────────────────────────────────────────────────────────────────────────────
// Public API - Types
// ─────────────────────────────────────────────────────────────────────────────

pub use caps::{Arch, Caps};
pub use detect::Detected;
pub use dispatch::{dispatch, dispatch_auto, dispatch_static, has_static_features};
pub use tune::{Tune, TuneKind};

// Architecture-specific feature constants are available via submodules:
// - `caps::x86` - x86/x86_64 features (SSE, AVX, AVX-512, etc.)
// - `caps::aarch64` - AArch64 features (NEON, SVE, crypto, etc.)
// - `caps::riscv` - RISC-V features (V, Zb*, Zk*, etc.)
// - `caps::wasm` - WebAssembly features (simd128, relaxed-simd)
// - `caps::s390x` - IBM Z features (vector, crypto)
// - `caps::power` - POWER features (AltiVec, VSX, etc.)

// ─────────────────────────────────────────────────────────────────────────────
// Public API - Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Get detected CPU capabilities and tuning hints.
///
/// Results are cached after first call.
///
/// # Examples
///
/// ```
/// let det = platform::get();
/// assert!(det.caps.count() >= 1);
/// ```
#[inline]
#[must_use]
pub fn get() -> Detected {
  detect::get()
}

/// Get just the CPU capabilities.
///
/// Convenience wrapper around [`get()`].
#[inline]
#[must_use]
pub fn caps() -> Caps {
  detect::caps()
}

/// Get just the tuning hints.
///
/// Convenience wrapper around [`get()`].
#[inline]
#[must_use]
pub fn tune() -> Tune {
  detect::tune()
}

/// Get the detected architecture.
///
/// Convenience wrapper around [`get()`].
#[inline]
#[must_use]
pub fn arch() -> Arch {
  detect::arch()
}

/// Set detection override.
///
/// Call this **before** any call to [`get()`] to bypass runtime detection.
/// Useful for bare-metal, embedded, or testing.
///
/// # Examples
///
/// ```
/// use platform::Detected;
///
/// platform::set_override(Some(Detected::portable()));
/// assert!(platform::has_override());
/// platform::clear_override();
/// ```
#[inline]
pub fn set_override(value: Option<Detected>) {
  detect::set_override(value);
}

/// Clear the detection override.
///
/// Equivalent to `set_override(None)`.
#[inline]
pub fn clear_override() {
  detect::clear_override();
}

/// Check if an override is currently set.
#[inline]
#[must_use]
pub fn has_override() -> bool {
  detect::has_override()
}

/// Get compile-time known capabilities.
///
/// Returns capabilities that are known at compile time via `-C target-feature=...`
/// or `-C target-cpu=native`. Use this for zero-overhead dispatch.
///
/// See [`detect::caps_static`] for details.
#[inline(always)]
#[must_use]
pub const fn caps_static() -> Caps {
  detect::caps_static()
}

/// Get compile-time tuning hints.
///
/// Returns conservative tuning hints based on compile-time known features.
///
/// See [`detect::tune_static`] for details.
#[inline(always)]
#[must_use]
pub const fn tune_static() -> Tune {
  detect::tune_static()
}

// ─────────────────────────────────────────────────────────────────────────────
// Description (for diagnostics)
// ─────────────────────────────────────────────────────────────────────────────

/// A zero-allocation description of detected CPU capabilities and tuning.
///
/// Implements `Display` so it can be written to any formatter without heap allocation.
#[derive(Clone, Copy)]
pub struct Description {
  det: Detected,
}

impl core::fmt::Display for Description {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "{:?} ({})", self.det.caps, self.det.tune.name())
  }
}

impl core::fmt::Debug for Description {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    core::fmt::Display::fmt(self, f)
  }
}

/// Returns a human-readable summary of detected CPU capabilities.
///
/// # Examples
///
/// ```
/// let desc = platform::describe();
/// assert!(!format!("{desc}").is_empty());
/// ```
#[inline]
#[must_use]
pub fn describe() -> Description {
  Description { det: get() }
}
