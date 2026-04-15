//! CPU detection and capability reporting.
//!
//! This module is the facts layer for rscrypto. It reports what instructions are
//! legal on the current target and leaves dispatch policy to algorithm modules.
//!
//! # Quick Start
//!
//! ```
//! let runtime = rscrypto::platform::caps();
//! let compile_time = rscrypto::platform::caps_static();
//!
//! assert!(runtime.count() >= compile_time.count());
//! ```
//!
//! # Design
//!
//! - **[`Caps`]**: 256-bit feature bitset. "What instructions can run?"
//! - **[`caps_static`]**: Compile-time facts from `cfg!(target_feature = ...)`
//! - **[`caps()`]**: Runtime facts via CPUID/HWCAP with process-wide caching
//! - **[`Detected`]**: Capabilities plus architecture identifier
//!
//! Algorithm crates decide whether to use compile-time facts, runtime facts, or
//! a mix of both for their own planners. This module does not own dispatch
//! policy.
//!
//! # Performance
//!
//! - Compile-time capability query: **0ns** after optimization
//! - Cached runtime capability query: **~3ns**
//! - First runtime detection: **~1μs** (CPUID/sysctl, once per process)
// ─────────────────────────────────────────────────────────────────────────────
// Core modules
// ─────────────────────────────────────────────────────────────────────────────

pub mod caps;
pub mod detect;
pub mod target_matrix;

// ─────────────────────────────────────────────────────────────────────────────
// Public API - Types
// ─────────────────────────────────────────────────────────────────────────────

pub use caps::{Arch, Caps};
pub use detect::{Detected, OverrideError};

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

/// Get detected CPU capabilities and architecture.
///
/// Results are cached after first call.
///
/// # Examples
///
/// ```
/// let det = rscrypto::platform::get();
/// assert_eq!(det.arch, rscrypto::platform::Arch::current());
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
/// # Panics
///
/// Panics if detection has already been initialized or overrides are unsupported
/// on the current target. Use [`try_set_override()`] for a fallible path.
///
/// # Examples
///
/// ```
/// use rscrypto::platform::Detected;
///
/// rscrypto::platform::set_override(Some(Detected::portable()));
/// assert!(rscrypto::platform::has_override());
/// rscrypto::platform::clear_override();
/// ```
#[inline]
pub fn set_override(value: Option<Detected>) {
  detect::set_override(value);
}

/// Try to set detection override.
///
/// Returns an explicit error if detection has already been initialized.
#[inline]
pub fn try_set_override(value: Option<Detected>) -> Result<(), OverrideError> {
  detect::try_set_override(value)
}

/// Clear the detection override.
///
/// Equivalent to `set_override(None)`.
///
/// # Panics
///
/// Panics under the same conditions as [`set_override()`]. Use
/// [`try_set_override(None)`](try_set_override) for a fallible path.
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

// ─────────────────────────────────────────────────────────────────────────────
// Description (for diagnostics)
// ─────────────────────────────────────────────────────────────────────────────

/// A zero-allocation description of detected CPU capabilities and architecture.
///
/// Implements `Display` so it can be written to any formatter without heap allocation.
#[derive(Clone, Copy)]
pub struct Description {
  det: Detected,
}

impl core::fmt::Display for Description {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "{:?} ({:?})", self.det.caps, self.det.arch)
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
/// let desc = rscrypto::platform::describe();
/// assert!(!format!("{desc}").is_empty());
/// ```
#[inline]
#[must_use]
pub fn describe() -> Description {
  Description { det: get() }
}
