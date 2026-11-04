use crate::platform::caps::{Arch, Caps};

/// Errors when configuring runtime detection overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OverrideError {
  /// Detection has already been initialized; override updates are no longer allowed.
  AlreadyInitialized,
  /// The requested override asserts capabilities or an architecture that this target cannot safely
  /// provide.
  InvalidCapabilities,
  /// Overrides are unsupported on this target configuration.
  Unsupported,
}

impl core::fmt::Display for OverrideError {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      Self::AlreadyInitialized => f.write_str("detection cache already initialized"),
      Self::InvalidCapabilities => f.write_str("invalid platform override capabilities"),
      Self::Unsupported => f.write_str("override unsupported on this target"),
    }
  }
}

// Main API

/// Detected CPU state: capabilities and architecture.
///
/// This struct combines all detection results:
/// - `caps`: Available CPU features (what instructions can run)
/// - `arch`: Target architecture identifier
///
/// Use [`crate::platform::get`] to obtain a cached instance, or
/// [`crate::platform::expert::detect_uncached`] for fresh detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Detected {
  /// CPU feature capabilities bitset.
  pub caps: Caps,
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
      arch: Arch::Other,
    }
  }

  #[inline]
  #[must_use]
  #[cfg(any(feature = "std", all(not(feature = "std"), target_has_atomic = "64")))]
  const fn is_portable(self) -> bool {
    self.caps.is_empty() && matches!(self.arch, Arch::Other)
  }
}

#[cold]
#[cfg(any(feature = "std", all(not(feature = "std"), target_has_atomic = "64")))]
fn validate_override(value: Option<Detected>) -> Result<Option<Detected>, OverrideError> {
  let Some(det) = value else {
    return Ok(None);
  };

  if det.is_portable() {
    return Ok(Some(det));
  }

  #[cfg(miri)]
  let host = Detected {
    caps: caps_static(),
    arch: Arch::current(),
  };
  #[cfg(not(miri))]
  let host = detect_uncached();

  if det.arch != host.arch {
    return Err(OverrideError::InvalidCapabilities);
  }

  if !host.caps.has(det.caps) {
    return Err(OverrideError::InvalidCapabilities);
  }

  Ok(Some(det))
}

/// Get detected CPU capabilities and architecture.
///
/// Results are cached after first call.
///
#[inline]
#[must_use]
pub(super) fn get() -> Detected {
  // Miri cannot interpret SIMD intrinsics
  #[cfg(miri)]
  {
    return Detected::portable();
  }

  #[cfg(not(miri))]
  {
    let det = {
      #[cfg(any(feature = "std", target_has_atomic = "64"))]
      {
        atomic_cache::CACHE.get_or_init(detect_with_override)
      }

      #[cfg(not(any(feature = "std", target_has_atomic = "64")))]
      {
        // Constrained targets: always call detect (no caching)
        detect_with_override()
      }
    };

    debug_assert!(
      crate::platform::target_matrix::manifest_has_arch(det.arch),
      "detected arch policy drifted from .config/target-matrix.json"
    );
    det
  }
}

/// Get just the capabilities.
///
/// When `feature = "portable-only"` is enabled, this returns [`Caps::NONE`]
/// unconditionally — every dispatcher walking `caps()` falls through to its
/// portable backend. See the `portable-only` feature description in
/// `Cargo.toml` for deployment context (FIPS / DO-178C / ISO 26262).
#[inline]
#[must_use]
pub(super) fn caps() -> Caps {
  #[cfg(feature = "portable-only")]
  {
    Caps::NONE
  }
  #[cfg(not(feature = "portable-only"))]
  {
    get().caps
  }
}

/// Get the detected architecture.
#[inline]
#[must_use]
pub(super) fn arch() -> Arch {
  get().arch
}

include!("detect/compile_time.rs");
include!("detect/cache_override.rs");

// Uncached Detection

/// Detect capabilities without caching (for testing/benchmarking).
#[cold]
#[must_use]
pub fn detect_uncached() -> Detected {
  #[cfg(target_arch = "x86_64")]
  {
    detect_x86_64()
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

/// `AT_HWCAP` and `AT_HWCAP2` from the C library's copy of the ELF auxiliary vector.
///
/// Absent entries read as zero. This replaces reading `/proc/self/auxv` through `std::fs`:
/// one libc call per entry, with no file, allocation, or formatting on the detection path.
#[cfg(all(
  feature = "std",
  any(target_arch = "aarch64", target_arch = "powerpc64", target_arch = "s390x"),
  any(target_os = "linux", target_os = "android")
))]
fn linux_hwcaps() -> (u64, u64) {
  const AT_HWCAP: core::ffi::c_ulong = 16;
  const AT_HWCAP2: core::ffi::c_ulong = 26;

  // The `safe` declaration holds because:
  // 1. The C libraries of every Linux and Android target for these architectures (glibc 2.16+,
  //    musl 1.1+, OHOS, bionic) export `unsigned long getauxval(unsigned long type)` with the C
  //    ABI; `c_ulong` is that type here.
  // 2. It takes any integer, reads only the auxiliary vector saved at process start, and
  //    returns 0 for an absent entry, so every call is defined.
  unsafe extern "C" {
    safe fn getauxval(entry: core::ffi::c_ulong) -> core::ffi::c_ulong;
  }

  // `c_ulong` is `u64` on every 64-bit target this gate admits.
  (getauxval(AT_HWCAP), getauxval(AT_HWCAP2))
}

include!("detect/arch/x86.rs");
include!("detect/arch/aarch64.rs");
include!("detect/arch/riscv.rs");
include!("detect/arch/s390x.rs");
include!("detect/arch/power.rs");
include!("detect/arch/wasm.rs");
include!("detect/tests.rs");
