//! CPU detection, capabilities, and tuning for rscrypto.
//!
//! This crate is the **single source of truth** for CPU feature detection
//! and optimal algorithm selection across the rscrypto workspace.
//!
//! # Core Types
//!
//! - [`CpuCaps`]: What instructions can run on this machine (capabilities)
//! - [`Tune`]: What strategies are optimal on this machine (tuning hints)
//!
//! # Main Entry Point
//!
//! ```ignore
//! use platform::{get, caps, tune};
//!
//! let (caps, tune) = get();
//!
//! // Check capabilities
//! if caps.has(platform::caps::x86::VPCLMUL_READY) {
//!     // Use AVX-512 VPCLMULQDQ kernel
//! }
//!
//! // Check tuning
//! if data.len() < tune.simd_threshold {
//!     // Use scalar/small-buffer path
//! }
//! ```
//!
//! # Architecture-Specific Modules
//!
//! - `x86_64`: Intel/AMD microarchitecture detection (Zen, SPR, ICL, etc.)
//! - `aarch64`: ARM feature detection (PMULL, CRC, SHA3, etc.)
//!
//! # Design Philosophy
//!
//! 1. **One API**: Algorithms query `platform::get()` instead of doing ad-hoc detection.
//! 2. **Capabilities vs Tuning**: `CpuCaps` says what's *possible*; `Tune` says what's *optimal*.
//! 3. **Zero-cost when possible**: Compile-time features are detected via `cfg!`, avoiding runtime
//!    overhead.
//! 4. **Cached otherwise**: Runtime detection is cached in `OnceLock` (std) or atomics (no_std).
//! 5. **Miri-safe**: Under Miri, always returns portable-only caps.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

// ─────────────────────────────────────────────────────────────────────────────
// Core modules
// ─────────────────────────────────────────────────────────────────────────────

pub mod caps;
mod detect;
pub mod tune;

// ─────────────────────────────────────────────────────────────────────────────
// Architecture-specific modules
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

pub use caps::{Arch, Bits256, CpuCaps};
pub use tune::Tune;

/// Get detected CPU capabilities and tuning hints.
///
/// This is the main entry point for capability-based dispatch.
///
/// # Caching
///
/// - With `std`: Results are cached in a `OnceLock` (one-time detection).
/// - Without `std`: Detection runs each call (fast, ~100 cycles for CPUID).
///
/// # Miri
///
/// Under Miri, always returns portable-only capabilities to avoid
/// interpreting SIMD intrinsics.
///
/// # Example
///
/// ```ignore
/// let (caps, tune) = platform::get();
///
/// if caps.has(platform::caps::x86::VPCLMUL_READY) {
///     // Use AVX-512 VPCLMULQDQ kernel
/// }
///
/// if data.len() < tune.simd_threshold {
///     // Use scalar path for small buffers
/// }
/// ```
#[inline]
#[must_use]
pub fn get() -> (CpuCaps, Tune) {
  detect::get()
}

/// Get just the CPU capabilities.
///
/// Convenience wrapper around [`get()`].
#[inline]
#[must_use]
pub fn caps() -> CpuCaps {
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

/// Initialize with user-supplied capabilities.
///
/// Call this before any call to [`get()`] to bypass runtime detection.
/// This is useful for:
/// - Bare metal environments without runtime detection support
/// - Embedded systems where the CPU is known at deployment
/// - Testing specific code paths
///
/// # Example
///
/// ```ignore
/// use platform::{CpuCaps, Tune, caps::x86};
///
/// // Set up for a known Zen 5 CPU
/// let caps = CpuCaps::new(x86::VPCLMUL_READY);
/// platform::init_with_caps(caps, Tune::ZEN5);
/// ```
#[inline]
pub fn init_with_caps(caps: CpuCaps, tune: Tune) {
  detect::init_with_caps(caps, tune);
}

/// Set or clear the capabilities override.
///
/// When set, [`get()`] will return the override value instead of detecting.
/// Pass `None` to clear the override and resume detection.
///
/// # Thread Safety
///
/// This function is thread-safe but should typically be called early in
/// program initialization, before any calls to [`get()`].
///
/// # Example
///
/// ```ignore
/// // In tests
/// platform::set_caps_override(Some((CpuCaps::NONE, Tune::PORTABLE)));
/// // ... run tests with portable fallback ...
/// platform::set_caps_override(None);
/// ```
#[inline]
pub fn set_caps_override(value: Option<(CpuCaps, Tune)>) {
  detect::set_caps_override(value);
}

/// Check if an override is currently set.
#[inline]
#[must_use]
pub fn has_override() -> bool {
  detect::has_override()
}
