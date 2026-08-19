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
//! // `caps_static()` reports compile-time facts. `caps()` reports instructions
//! // legal for this process at runtime. Runtime results normally include the
//! // static set, but process-authorized state such as Linux AMX can remove a
//! // static capability. The optional `portable-only` feature collapses
//! // `caps()` to `Caps::NONE` for FIPS / DO-178C deployment modes; that
//! // override does not change `caps_static()`. Both functions return `Caps`,
//! // a 256-bit bitset.
//! let _ = (runtime, compile_time);
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
//! Compile-time capability queries are constant-foldable. Runtime detection is
//! cached where the target provides the required synchronization; the uncached
//! cost depends on the architecture and operating-system probe.
// Core modules

pub mod caps;
pub(crate) mod detect;
#[cfg(not(miri))]
mod target_matrix;

// Public API - Types

pub use caps::{Arch, Caps};
pub use detect::Detected;

// Architecture-specific feature constants are available via submodules:
// - `caps::x86` - x86/x86_64 features (SSE, AVX, AVX-512, etc.)
// - `caps::aarch64` - AArch64 features (NEON, SVE, crypto, etc.)
// - `caps::riscv` - RISC-V features (V, Zb*, Zk*, etc.)
// - `caps::wasm` - WebAssembly features (simd128, relaxed-simd)
// - `caps::s390x` - IBM Z features (vector, crypto)
// - `caps::power` - POWER features (AltiVec, VSX, etc.)

// Public API - Functions

/// Get detected CPU capabilities and architecture.
///
/// Results are cached after first call.
///
/// On Linux and Android x86_64, AMX capabilities are reported only when the
/// process already has permission for tile state. Detection does not request
/// permission. Request it before the first cached detection if the process
/// intends to use AMX.
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

/// Explicit controls and uncached detection for tests and constrained runtimes.
///
/// Normal callers should use [`get()`], [`caps()`], or [`arch()`]. Overrides
/// must be configured before the first cached detection.
///
/// ```
/// use rscrypto::platform::{Detected, expert};
///
/// expert::try_set_override(Some(Detected::portable()))?;
/// expert::try_set_override(None)?;
/// # Ok::<(), rscrypto::platform::expert::OverrideError>(())
/// ```
pub mod expert {
  pub use super::detect::{OverrideError, detect_uncached, has_override, try_set_override};
}

/// Get compile-time known capabilities.
///
/// Returns capabilities that are known at compile time via `-C target-feature=...`
/// or `-C target-cpu=native`. Use this for zero-overhead dispatch.
///
/// This is a compile-time constant and performs no runtime detection.
/// For generic binaries that run on multiple CPUs, use [`caps()`] instead.
///
/// # Examples
///
/// ```
/// use rscrypto::platform::caps_static;
///
/// const CAPS: rscrypto::platform::Caps = caps_static();
///
/// #[cfg(target_arch = "x86_64")]
/// {
///   use rscrypto::platform::caps::x86;
///   assert!(CAPS.has(x86::SSE2));
/// }
///
/// #[cfg(target_arch = "aarch64")]
/// {
///   use rscrypto::platform::caps::aarch64;
///   assert!(CAPS.has(aarch64::NEON));
/// }
/// ```
#[inline(always)]
#[must_use]
pub const fn caps_static() -> Caps {
  detect::caps_static()
}

// Description (for diagnostics)

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

/// Zero-allocation dispatch metadata shared by checksum, hash, and AEAD
/// introspection surfaces.
#[derive(Clone, Copy)]
pub struct DispatchInfo {
  platform: Description,
}

impl DispatchInfo {
  /// Returns dispatch info for the current platform.
  #[inline]
  #[must_use]
  pub fn current() -> Self {
    Self { platform: describe() }
  }

  /// Returns the platform description driving dispatch decisions.
  #[inline]
  #[must_use]
  pub fn platform(&self) -> Description {
    self.platform
  }
}

impl core::fmt::Display for DispatchInfo {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    core::fmt::Display::fmt(&self.platform, f)
  }
}

impl core::fmt::Debug for DispatchInfo {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("DispatchInfo")
      .field("platform", &format_args!("{}", self.platform))
      .finish()
  }
}

/// Trait for algorithms that can report the kernel chosen for a buffer length.
pub trait KernelIntrospect {
  /// Returns the kernel name that would be selected for a buffer of `len`
  /// bytes.
  fn kernel_name_for_len(len: usize) -> &'static str;
}
