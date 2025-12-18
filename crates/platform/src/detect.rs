//! Runtime CPU detection.
//!
//! This module provides the unified `get()` function that returns detected
//! CPU capabilities and tuning hints. It handles:
//!
//! - Compile-time detection (via `cfg!(target_feature = "...")`)
//! - Runtime detection (via CPUID on x86, auxv/sysctl on ARM, etc.)
//! - Caching (via `OnceLock` with `std`, atomics without)
//! - User-supplied overrides for bare metal and testing
//! - Miri fallback (always returns portable caps)
//!
//! # Usage
//!
//! ```ignore
//! let (caps, tune) = platform::get();
//!
//! if caps.has(x86::VPCLMUL_READY) {
//!     // Use VPCLMULQDQ kernel
//! }
//!
//! if data.len() < tune.simd_threshold {
//!     // Use scalar kernel
//! }
//! ```
//!
//! # Overrides
//!
//! For bare metal or testing scenarios where runtime detection isn't available
//! or desirable:
//!
//! ```ignore
//! // Initialize with known capabilities (call before any get())
//! platform::init_with_caps(my_caps, my_tune);
//!
//! // Or set an override for testing
//! platform::set_caps_override(Some((test_caps, test_tune)));
//! ```

#[cfg(not(feature = "std"))]
use core::sync::atomic::AtomicU8;
use core::sync::atomic::{AtomicBool, Ordering};

#[cfg(not(feature = "std"))]
use crate::caps::Arch;
use crate::{
  caps::{Bits256, CpuCaps},
  tune::Tune,
};

// ─────────────────────────────────────────────────────────────────────────────
// Cache and Override Infrastructure
// ─────────────────────────────────────────────────────────────────────────────
//
// We support two use cases:
// 1. Normal detection with caching (std: OnceLock, no_std: atomics)
// 2. User-supplied overrides for bare metal and testing
//
// The override takes precedence over detection.

/// Cache state for no_std builds.
#[cfg(not(feature = "std"))]
mod cache {
  use core::sync::atomic::AtomicUsize;

  use super::*;

  /// Initialization state.
  /// 0 = uninitialized, 1 = initializing, 2 = initialized
  static STATE: AtomicU8 = AtomicU8::new(0);

  /// Cached CpuCaps bits (4 x u64 = 32 bytes, stored as 4 AtomicUsize pairs on 64-bit)
  static CACHED_BITS: [core::sync::atomic::AtomicU64; 4] = [
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
  ];

  /// Cached architecture.
  static CACHED_ARCH: AtomicU8 = AtomicU8::new(0);

  /// Cached Tune fields.
  static CACHED_SIMD_THRESHOLD: AtomicUsize = AtomicUsize::new(0);
  static CACHED_PREFER_HYBRID: AtomicBool = AtomicBool::new(false);
  static CACHED_CRC_PARALLELISM: AtomicU8 = AtomicU8::new(0);
  static CACHED_FAST_ZMM: AtomicBool = AtomicBool::new(false);

  /// Try to get cached value, or compute and cache.
  #[inline]
  pub fn get_or_init(f: fn() -> (CpuCaps, Tune)) -> (CpuCaps, Tune) {
    // Fast path: already initialized
    if STATE.load(Ordering::Acquire) == 2 {
      return load_cached();
    }

    // Slow path: initialize
    // Try to claim initialization
    match STATE.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire) {
      Ok(_) => {
        // We won the race, compute and store
        let result = f();
        store_cached(&result);
        STATE.store(2, Ordering::Release);
        result
      }
      Err(1) => {
        // Someone else is initializing, spin wait
        while STATE.load(Ordering::Acquire) == 1 {
          core::hint::spin_loop();
        }
        load_cached()
      }
      Err(_) => {
        // Already initialized
        load_cached()
      }
    }
  }

  fn load_cached() -> (CpuCaps, Tune) {
    let bits = Bits256([
      CACHED_BITS[0].load(Ordering::Acquire),
      CACHED_BITS[1].load(Ordering::Acquire),
      CACHED_BITS[2].load(Ordering::Acquire),
      CACHED_BITS[3].load(Ordering::Acquire),
    ]);
    let arch = arch_from_u8(CACHED_ARCH.load(Ordering::Acquire));
    let caps = CpuCaps { arch, bits };

    let tune = Tune {
      simd_threshold: CACHED_SIMD_THRESHOLD.load(Ordering::Acquire),
      prefer_hybrid_crc: CACHED_PREFER_HYBRID.load(Ordering::Acquire),
      crc_parallelism: CACHED_CRC_PARALLELISM.load(Ordering::Acquire),
      fast_zmm: CACHED_FAST_ZMM.load(Ordering::Acquire),
    };

    (caps, tune)
  }

  fn store_cached((caps, tune): &(CpuCaps, Tune)) {
    CACHED_BITS[0].store(caps.bits.0[0], Ordering::Release);
    CACHED_BITS[1].store(caps.bits.0[1], Ordering::Release);
    CACHED_BITS[2].store(caps.bits.0[2], Ordering::Release);
    CACHED_BITS[3].store(caps.bits.0[3], Ordering::Release);
    CACHED_ARCH.store(arch_to_u8(caps.arch), Ordering::Release);

    CACHED_SIMD_THRESHOLD.store(tune.simd_threshold, Ordering::Release);
    CACHED_PREFER_HYBRID.store(tune.prefer_hybrid_crc, Ordering::Release);
    CACHED_CRC_PARALLELISM.store(tune.crc_parallelism, Ordering::Release);
    CACHED_FAST_ZMM.store(tune.fast_zmm, Ordering::Release);
  }

  pub fn arch_to_u8(arch: Arch) -> u8 {
    match arch {
      Arch::X86_64 => 1,
      Arch::X86 => 2,
      Arch::Aarch64 => 3,
      Arch::Arm => 4,
      Arch::Riscv64 => 5,
      Arch::Riscv32 => 6,
      Arch::Powerpc64 => 7,
      Arch::LoongArch64 => 8,
      Arch::Wasm32 => 9,
      Arch::Wasm64 => 10,
      Arch::Other => 0,
    }
  }

  pub fn arch_from_u8(v: u8) -> Arch {
    match v {
      1 => Arch::X86_64,
      2 => Arch::X86,
      3 => Arch::Aarch64,
      4 => Arch::Arm,
      5 => Arch::Riscv64,
      6 => Arch::Riscv32,
      7 => Arch::Powerpc64,
      8 => Arch::LoongArch64,
      9 => Arch::Wasm32,
      10 => Arch::Wasm64,
      _ => Arch::Other,
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Override Support
// ─────────────────────────────────────────────────────────────────────────────

/// Override state: 0 = none, 1 = set
static OVERRIDE_SET: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "std")]
static OVERRIDE: std::sync::OnceLock<Option<(CpuCaps, Tune)>> = std::sync::OnceLock::new();

#[cfg(not(feature = "std"))]
mod override_storage {
  use super::*;

  /// Override bits storage.
  pub static BITS: [core::sync::atomic::AtomicU64; 4] = [
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
  ];
  pub static ARCH: AtomicU8 = AtomicU8::new(0);
  pub static SIMD_THRESHOLD: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
  pub static PREFER_HYBRID: AtomicBool = AtomicBool::new(false);
  pub static CRC_PARALLELISM: AtomicU8 = AtomicU8::new(0);
  pub static FAST_ZMM: AtomicBool = AtomicBool::new(false);
}

/// Initialize with user-supplied capabilities.
///
/// Call this before any call to `get()` to bypass runtime detection.
/// This is useful for:
/// - Bare metal environments without runtime detection support
/// - Embedded systems where the CPU is known at deployment
/// - Testing specific code paths
///
/// # Panics
///
/// Panics if called after `get()` has already been called (std only).
/// For no_std, the behavior is best-effort (may race with detection).
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
pub fn init_with_caps(caps: CpuCaps, tune: Tune) {
  set_caps_override(Some((caps, tune)));
}

/// Set or clear the capabilities override.
///
/// When set, `get()` will return the override value instead of detecting.
/// Pass `None` to clear the override and resume detection.
///
/// # Thread Safety
///
/// This function is thread-safe but should typically be called early in
/// program initialization, before any calls to `get()`.
///
/// # Example
///
/// ```ignore
/// // In tests
/// platform::set_caps_override(Some((CpuCaps::NONE, Tune::PORTABLE)));
/// // ... run tests with portable fallback ...
/// platform::set_caps_override(None);
/// ```
pub fn set_caps_override(value: Option<(CpuCaps, Tune)>) {
  #[cfg(feature = "std")]
  {
    // For std, we use OnceLock which can only be set once.
    // The override is stored in a separate OnceLock.
    let _ = OVERRIDE.set(value);
    OVERRIDE_SET.store(value.is_some(), Ordering::Release);
  }

  #[cfg(not(feature = "std"))]
  {
    match value {
      Some((caps, tune)) => {
        override_storage::BITS[0].store(caps.bits.0[0], Ordering::Release);
        override_storage::BITS[1].store(caps.bits.0[1], Ordering::Release);
        override_storage::BITS[2].store(caps.bits.0[2], Ordering::Release);
        override_storage::BITS[3].store(caps.bits.0[3], Ordering::Release);
        override_storage::ARCH.store(cache::arch_to_u8(caps.arch), Ordering::Release);
        override_storage::SIMD_THRESHOLD.store(tune.simd_threshold, Ordering::Release);
        override_storage::PREFER_HYBRID.store(tune.prefer_hybrid_crc, Ordering::Release);
        override_storage::CRC_PARALLELISM.store(tune.crc_parallelism, Ordering::Release);
        override_storage::FAST_ZMM.store(tune.fast_zmm, Ordering::Release);
        OVERRIDE_SET.store(true, Ordering::Release);
      }
      None => {
        OVERRIDE_SET.store(false, Ordering::Release);
      }
    }
  }
}

/// Check if an override is currently set.
#[inline]
pub fn has_override() -> bool {
  OVERRIDE_SET.load(Ordering::Acquire)
}

/// Get the current override, if any.
fn get_override() -> Option<(CpuCaps, Tune)> {
  if !OVERRIDE_SET.load(Ordering::Acquire) {
    return None;
  }

  #[cfg(feature = "std")]
  {
    OVERRIDE.get().and_then(|v| *v)
  }

  #[cfg(not(feature = "std"))]
  {
    let bits = Bits256([
      override_storage::BITS[0].load(Ordering::Acquire),
      override_storage::BITS[1].load(Ordering::Acquire),
      override_storage::BITS[2].load(Ordering::Acquire),
      override_storage::BITS[3].load(Ordering::Acquire),
    ]);
    let arch = cache::arch_from_u8(override_storage::ARCH.load(Ordering::Acquire));
    let caps = CpuCaps { arch, bits };
    let tune = Tune {
      simd_threshold: override_storage::SIMD_THRESHOLD.load(Ordering::Acquire),
      prefer_hybrid_crc: override_storage::PREFER_HYBRID.load(Ordering::Acquire),
      crc_parallelism: override_storage::CRC_PARALLELISM.load(Ordering::Acquire),
      fast_zmm: override_storage::FAST_ZMM.load(Ordering::Acquire),
    };
    Some((caps, tune))
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main API
// ─────────────────────────────────────────────────────────────────────────────

/// Get detected CPU capabilities and tuning hints.
///
/// This is the main entry point for capability-based dispatch.
///
/// # Caching
///
/// - With `std`: Results are cached in a `OnceLock` (one-time detection).
/// - Without `std`: Results are cached using atomics (one-time detection).
///
/// # Override
///
/// If an override has been set via [`init_with_caps`] or [`set_caps_override`],
/// that value is returned instead of detected capabilities.
///
/// # Miri
///
/// Under Miri, always returns portable-only capabilities to avoid
/// interpreting SIMD intrinsics.
#[inline]
#[must_use]
pub fn get() -> (CpuCaps, Tune) {
  // Miri cannot interpret SIMD intrinsics, so always return portable.
  #[cfg(miri)]
  {
    return (CpuCaps::NONE, Tune::PORTABLE);
  }

  #[cfg(not(miri))]
  {
    // Check for user-supplied override first
    if let Some(result) = get_override() {
      return result;
    }

    #[cfg(feature = "std")]
    {
      use std::sync::OnceLock;
      static CACHED: OnceLock<(CpuCaps, Tune)> = OnceLock::new();
      *CACHED.get_or_init(detect_uncached)
    }

    #[cfg(not(feature = "std"))]
    {
      cache::get_or_init(detect_uncached)
    }
  }
}

/// Get just the capabilities (convenience function).
#[inline]
#[must_use]
pub fn caps() -> CpuCaps {
  get().0
}

/// Get just the tuning hints (convenience function).
#[inline]
#[must_use]
pub fn tune() -> Tune {
  get().1
}

/// Detect capabilities without caching.
///
/// This is useful for testing or when you need fresh detection.
#[inline]
#[must_use]
pub fn detect_uncached() -> (CpuCaps, Tune) {
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

  #[cfg(target_arch = "wasm32")]
  {
    detect_wasm32()
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "x86",
    target_arch = "aarch64",
    target_arch = "wasm32"
  )))]
  {
    (CpuCaps::NONE, Tune::PORTABLE)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn detect_x86_64() -> (CpuCaps, Tune) {
  use crate::{
    caps::Arch,
    x86_64::{MicroArch, detect_microarch_uncached},
  };

  let mut bits = Bits256::NONE;

  // Always start with compile-time features
  bits = bits.union(compile_time_x86_64());

  // Add runtime-detected features (std only)
  #[cfg(feature = "std")]
  {
    bits = bits.union(runtime_x86_64());
  }

  let caps = CpuCaps {
    arch: Arch::X86_64,
    bits,
  };

  // Determine tuning based on microarchitecture
  let microarch = detect_microarch_uncached();
  let tune = match microarch {
    MicroArch::Zen5 => Tune::ZEN5,
    MicroArch::Zen4 => Tune::ZEN4,
    MicroArch::SapphireRapids | MicroArch::EmeraldRapids | MicroArch::GraniteRapids => Tune::INTEL_SPR,
    MicroArch::IceLake => Tune::INTEL_ICL,
    MicroArch::CascadeLake | MicroArch::Zen3 => Tune::DEFAULT,
    MicroArch::GenericAvx512Vpclmul => Tune::DEFAULT,
    MicroArch::GenericPclmul | MicroArch::Generic => Tune::DEFAULT,
  };

  (caps, tune)
}

/// Compile-time detected x86_64 features.
#[cfg(target_arch = "x86_64")]
const fn compile_time_x86_64() -> Bits256 {
  use crate::caps::x86;

  let mut bits = Bits256::NONE;

  // SSE family (always available on x86_64)
  bits = bits.union(x86::SSE2);

  #[cfg(target_feature = "sse3")]
  {
    bits = bits.union(x86::SSE3);
  }

  #[cfg(target_feature = "ssse3")]
  {
    bits = bits.union(x86::SSSE3);
  }

  #[cfg(target_feature = "sse4.1")]
  {
    bits = bits.union(x86::SSE41);
  }

  #[cfg(target_feature = "sse4.2")]
  {
    bits = bits.union(x86::SSE42);
  }

  // AVX family
  #[cfg(target_feature = "avx")]
  {
    bits = bits.union(x86::AVX);
  }

  #[cfg(target_feature = "avx2")]
  {
    bits = bits.union(x86::AVX2);
  }

  #[cfg(target_feature = "fma")]
  {
    bits = bits.union(x86::FMA);
  }

  // Crypto
  #[cfg(target_feature = "aes")]
  {
    bits = bits.union(x86::AESNI);
  }

  #[cfg(target_feature = "pclmulqdq")]
  {
    bits = bits.union(x86::PCLMULQDQ);
  }

  #[cfg(target_feature = "sha")]
  {
    bits = bits.union(x86::SHA);
  }

  // AVX-512
  #[cfg(target_feature = "avx512f")]
  {
    bits = bits.union(x86::AVX512F);
  }

  #[cfg(target_feature = "avx512vl")]
  {
    bits = bits.union(x86::AVX512VL);
  }

  #[cfg(target_feature = "avx512bw")]
  {
    bits = bits.union(x86::AVX512BW);
  }

  #[cfg(target_feature = "avx512dq")]
  {
    bits = bits.union(x86::AVX512DQ);
  }

  #[cfg(target_feature = "avx512cd")]
  {
    bits = bits.union(x86::AVX512CD);
  }

  #[cfg(target_feature = "vpclmulqdq")]
  {
    bits = bits.union(x86::VPCLMULQDQ);
  }

  #[cfg(target_feature = "vaes")]
  {
    bits = bits.union(x86::VAES);
  }

  #[cfg(target_feature = "gfni")]
  {
    bits = bits.union(x86::GFNI);
  }

  // Bit manipulation
  #[cfg(target_feature = "bmi1")]
  {
    bits = bits.union(x86::BMI1);
  }

  #[cfg(target_feature = "bmi2")]
  {
    bits = bits.union(x86::BMI2);
  }

  #[cfg(target_feature = "popcnt")]
  {
    bits = bits.union(x86::POPCNT);
  }

  #[cfg(target_feature = "lzcnt")]
  {
    bits = bits.union(x86::LZCNT);
  }

  #[cfg(target_feature = "adx")]
  {
    bits = bits.union(x86::ADX);
  }

  // Additional features
  #[cfg(target_feature = "sse4a")]
  {
    bits = bits.union(x86::SSE4A);
  }

  #[cfg(target_feature = "f16c")]
  {
    bits = bits.union(x86::F16C);
  }

  // AVX-512 additional subfeatures
  #[cfg(target_feature = "avx512ifma")]
  {
    bits = bits.union(x86::AVX512IFMA);
  }

  #[cfg(target_feature = "avx512vbmi")]
  {
    bits = bits.union(x86::AVX512VBMI);
  }

  #[cfg(target_feature = "avx512vbmi2")]
  {
    bits = bits.union(x86::AVX512VBMI2);
  }

  #[cfg(target_feature = "avx512vnni")]
  {
    bits = bits.union(x86::AVX512VNNI);
  }

  #[cfg(target_feature = "avx512bitalg")]
  {
    bits = bits.union(x86::AVX512BITALG);
  }

  #[cfg(target_feature = "avx512vpopcntdq")]
  {
    bits = bits.union(x86::AVX512VPOPCNTDQ);
  }

  #[cfg(target_feature = "avx512bf16")]
  {
    bits = bits.union(x86::AVX512BF16);
  }

  bits
}

/// Runtime detected x86_64 features.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn runtime_x86_64() -> Bits256 {
  use crate::caps::x86;

  let mut bits = Bits256::NONE;

  // SSE family
  if std::arch::is_x86_feature_detected!("sse3") {
    bits = bits.union(x86::SSE3);
  }
  if std::arch::is_x86_feature_detected!("ssse3") {
    bits = bits.union(x86::SSSE3);
  }
  if std::arch::is_x86_feature_detected!("sse4.1") {
    bits = bits.union(x86::SSE41);
  }
  if std::arch::is_x86_feature_detected!("sse4.2") {
    bits = bits.union(x86::SSE42);
  }

  // AVX family
  if std::arch::is_x86_feature_detected!("avx") {
    bits = bits.union(x86::AVX);
  }
  if std::arch::is_x86_feature_detected!("avx2") {
    bits = bits.union(x86::AVX2);
  }
  if std::arch::is_x86_feature_detected!("fma") {
    bits = bits.union(x86::FMA);
  }

  // Crypto
  if std::arch::is_x86_feature_detected!("aes") {
    bits = bits.union(x86::AESNI);
  }
  if std::arch::is_x86_feature_detected!("pclmulqdq") {
    bits = bits.union(x86::PCLMULQDQ);
  }
  if std::arch::is_x86_feature_detected!("sha") {
    bits = bits.union(x86::SHA);
  }

  // AVX-512
  if std::arch::is_x86_feature_detected!("avx512f") {
    bits = bits.union(x86::AVX512F);
  }
  if std::arch::is_x86_feature_detected!("avx512vl") {
    bits = bits.union(x86::AVX512VL);
  }
  if std::arch::is_x86_feature_detected!("avx512bw") {
    bits = bits.union(x86::AVX512BW);
  }
  if std::arch::is_x86_feature_detected!("avx512dq") {
    bits = bits.union(x86::AVX512DQ);
  }
  if std::arch::is_x86_feature_detected!("avx512cd") {
    bits = bits.union(x86::AVX512CD);
  }
  if std::arch::is_x86_feature_detected!("vpclmulqdq") {
    bits = bits.union(x86::VPCLMULQDQ);
  }
  if std::arch::is_x86_feature_detected!("vaes") {
    bits = bits.union(x86::VAES);
  }
  if std::arch::is_x86_feature_detected!("gfni") {
    bits = bits.union(x86::GFNI);
  }

  // Bit manipulation
  if std::arch::is_x86_feature_detected!("bmi1") {
    bits = bits.union(x86::BMI1);
  }
  if std::arch::is_x86_feature_detected!("bmi2") {
    bits = bits.union(x86::BMI2);
  }
  if std::arch::is_x86_feature_detected!("popcnt") {
    bits = bits.union(x86::POPCNT);
  }
  if std::arch::is_x86_feature_detected!("lzcnt") {
    bits = bits.union(x86::LZCNT);
  }
  if std::arch::is_x86_feature_detected!("adx") {
    bits = bits.union(x86::ADX);
  }

  // Additional features
  if std::arch::is_x86_feature_detected!("sse4a") {
    bits = bits.union(x86::SSE4A);
  }
  if std::arch::is_x86_feature_detected!("f16c") {
    bits = bits.union(x86::F16C);
  }

  // AVX-512 additional subfeatures
  if std::arch::is_x86_feature_detected!("avx512ifma") {
    bits = bits.union(x86::AVX512IFMA);
  }
  if std::arch::is_x86_feature_detected!("avx512vbmi") {
    bits = bits.union(x86::AVX512VBMI);
  }
  if std::arch::is_x86_feature_detected!("avx512vbmi2") {
    bits = bits.union(x86::AVX512VBMI2);
  }
  if std::arch::is_x86_feature_detected!("avx512vnni") {
    bits = bits.union(x86::AVX512VNNI);
  }
  if std::arch::is_x86_feature_detected!("avx512bitalg") {
    bits = bits.union(x86::AVX512BITALG);
  }
  if std::arch::is_x86_feature_detected!("avx512vpopcntdq") {
    bits = bits.union(x86::AVX512VPOPCNTDQ);
  }
  if std::arch::is_x86_feature_detected!("avx512bf16") {
    bits = bits.union(x86::AVX512BF16);
  }

  bits
}

// ─────────────────────────────────────────────────────────────────────────────
// x86 (32-bit) detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86")]
fn detect_x86() -> (CpuCaps, Tune) {
  use crate::caps::{Arch, x86};

  let mut bits = Bits256::NONE;

  // Compile-time features
  #[cfg(target_feature = "sse2")]
  {
    bits = bits.union(x86::SSE2);
  }

  #[cfg(target_feature = "ssse3")]
  {
    bits = bits.union(x86::SSSE3);
  }

  #[cfg(target_feature = "sse4.2")]
  {
    bits = bits.union(x86::SSE42);
  }

  #[cfg(target_feature = "pclmulqdq")]
  {
    bits = bits.union(x86::PCLMULQDQ);
  }

  // Runtime detection (std only)
  #[cfg(feature = "std")]
  {
    if std::arch::is_x86_feature_detected!("sse2") {
      bits = bits.union(x86::SSE2);
    }
    if std::arch::is_x86_feature_detected!("ssse3") {
      bits = bits.union(x86::SSSE3);
    }
    if std::arch::is_x86_feature_detected!("sse4.2") {
      bits = bits.union(x86::SSE42);
    }
    if std::arch::is_x86_feature_detected!("pclmulqdq") {
      bits = bits.union(x86::PCLMULQDQ);
    }
  }

  let caps = CpuCaps { arch: Arch::X86, bits };
  (caps, Tune::DEFAULT)
}

// ─────────────────────────────────────────────────────────────────────────────
// aarch64 detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
fn detect_aarch64() -> (CpuCaps, Tune) {
  use crate::caps::{Arch, aarch64};

  let mut bits = Bits256::NONE;

  // NEON is always available on AArch64
  bits = bits.union(aarch64::NEON);

  // Compile-time features
  bits = bits.union(compile_time_aarch64());

  // Runtime detection (std only)
  #[cfg(feature = "std")]
  {
    bits = bits.union(runtime_aarch64());
  }

  let caps = CpuCaps {
    arch: Arch::Aarch64,
    bits,
  };

  // Determine tuning based on detected features
  let tune = if bits.contains(aarch64::PMULL_EOR3_READY) {
    // Apple M-series or Neoverse with SHA3
    Tune::APPLE_M
  } else if bits.contains(aarch64::PMULL_READY) {
    Tune::AARCH64_PMULL
  } else {
    Tune::PORTABLE
  };

  (caps, tune)
}

/// Compile-time detected aarch64 features.
#[cfg(target_arch = "aarch64")]
const fn compile_time_aarch64() -> Bits256 {
  // Import is used when target_feature attributes are enabled at compile time.
  #[allow(unused_imports)]
  use crate::caps::aarch64;

  // Mutable when target_feature attributes enable feature unions.
  #[allow(unused_mut)]
  let mut bits = Bits256::NONE;

  #[cfg(target_feature = "aes")]
  {
    bits = bits.union(aarch64::AES);
    bits = bits.union(aarch64::PMULL); // PMULL is bundled with AES
  }

  #[cfg(target_feature = "crc")]
  {
    bits = bits.union(aarch64::CRC);
  }

  #[cfg(target_feature = "sha2")]
  {
    bits = bits.union(aarch64::SHA2);
  }

  #[cfg(target_feature = "sha3")]
  {
    bits = bits.union(aarch64::SHA3);
  }

  #[cfg(target_feature = "sm4")]
  {
    bits = bits.union(aarch64::SM3);
    bits = bits.union(aarch64::SM4);
  }

  #[cfg(target_feature = "dotprod")]
  {
    bits = bits.union(aarch64::DOTPROD);
  }

  #[cfg(target_feature = "i8mm")]
  {
    bits = bits.union(aarch64::I8MM);
  }

  #[cfg(target_feature = "bf16")]
  {
    bits = bits.union(aarch64::BF16);
  }

  #[cfg(target_feature = "fp16")]
  {
    bits = bits.union(aarch64::FP16);
  }

  #[cfg(target_feature = "sve")]
  {
    bits = bits.union(aarch64::SVE);
  }

  #[cfg(target_feature = "sve2")]
  {
    bits = bits.union(aarch64::SVE2);
  }

  #[cfg(target_feature = "sve2-aes")]
  {
    bits = bits.union(aarch64::SVE2_AES);
  }

  #[cfg(target_feature = "sve2-sha3")]
  {
    bits = bits.union(aarch64::SVE2_SHA3);
  }

  #[cfg(target_feature = "sve2-sm4")]
  {
    bits = bits.union(aarch64::SVE2_SM4);
  }

  #[cfg(target_feature = "sve2-bitperm")]
  {
    bits = bits.union(aarch64::SVE2_BITPERM);
  }

  bits
}

/// Runtime detected aarch64 features.
#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn runtime_aarch64() -> Bits256 {
  use crate::caps::aarch64;

  let mut bits = Bits256::NONE;

  if std::arch::is_aarch64_feature_detected!("aes") {
    bits = bits.union(aarch64::AES);
    bits = bits.union(aarch64::PMULL);
  }

  if std::arch::is_aarch64_feature_detected!("crc") {
    bits = bits.union(aarch64::CRC);
  }

  if std::arch::is_aarch64_feature_detected!("sha2") {
    bits = bits.union(aarch64::SHA2);
  }

  if std::arch::is_aarch64_feature_detected!("sha3") {
    bits = bits.union(aarch64::SHA3);
  }

  if std::arch::is_aarch64_feature_detected!("sm4") {
    bits = bits.union(aarch64::SM3);
    bits = bits.union(aarch64::SM4);
  }

  if std::arch::is_aarch64_feature_detected!("dotprod") {
    bits = bits.union(aarch64::DOTPROD);
  }

  if std::arch::is_aarch64_feature_detected!("i8mm") {
    bits = bits.union(aarch64::I8MM);
  }

  if std::arch::is_aarch64_feature_detected!("bf16") {
    bits = bits.union(aarch64::BF16);
  }

  if std::arch::is_aarch64_feature_detected!("fp16") {
    bits = bits.union(aarch64::FP16);
  }

  // SVE detection
  if std::arch::is_aarch64_feature_detected!("sve") {
    bits = bits.union(aarch64::SVE);
  }

  if std::arch::is_aarch64_feature_detected!("sve2") {
    bits = bits.union(aarch64::SVE2);
  }

  if std::arch::is_aarch64_feature_detected!("sve2-aes") {
    bits = bits.union(aarch64::SVE2_AES);
  }

  if std::arch::is_aarch64_feature_detected!("sve2-sha3") {
    bits = bits.union(aarch64::SVE2_SHA3);
  }

  if std::arch::is_aarch64_feature_detected!("sve2-sm4") {
    bits = bits.union(aarch64::SVE2_SM4);
  }

  if std::arch::is_aarch64_feature_detected!("sve2-bitperm") {
    bits = bits.union(aarch64::SVE2_BITPERM);
  }

  bits
}

// ─────────────────────────────────────────────────────────────────────────────
// wasm32 detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
fn detect_wasm32() -> (CpuCaps, Tune) {
  use crate::caps::Arch;
  // Import is used when target_feature attributes are enabled at compile time.
  #[allow(unused_imports)]
  use crate::caps::wasm;

  // Mutable when target_feature attributes enable feature unions.
  #[allow(unused_mut)]
  let mut bits = Bits256::NONE;

  // SIMD128 is compile-time only for wasm
  #[cfg(target_feature = "simd128")]
  {
    bits = bits.union(wasm::SIMD128);
  }

  // Relaxed SIMD (compile-time only)
  #[cfg(target_feature = "relaxed-simd")]
  {
    bits = bits.union(wasm::RELAXED_SIMD);
  }

  let caps = CpuCaps {
    arch: Arch::Wasm32,
    bits,
  };
  (caps, Tune::PORTABLE)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_get_returns_valid_caps() {
    let (caps, tune) = get();

    // Under Miri, we return portable caps (Arch::Other)
    #[cfg(miri)]
    {
      assert_eq!(caps.arch, crate::caps::Arch::Other);
      assert_eq!(caps.bits, crate::caps::Bits256::NONE);
    }

    // Normal execution: architecture should match current compilation target
    #[cfg(not(miri))]
    {
      #[cfg(target_arch = "x86_64")]
      assert_eq!(caps.arch, crate::caps::Arch::X86_64);

      #[cfg(target_arch = "aarch64")]
      assert_eq!(caps.arch, crate::caps::Arch::Aarch64);
    }

    // Tune should have reasonable values
    assert!(tune.simd_threshold > 0);
    assert!(tune.crc_parallelism > 0);
  }

  #[test]
  fn test_detect_uncached() {
    let (caps1, tune1) = detect_uncached();
    let (caps2, tune2) = detect_uncached();

    // Should return consistent results
    assert_eq!(caps1, caps2);
    assert_eq!(tune1, tune2);
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", not(miri)))]
  fn test_x86_64_detection() {
    let (caps, _tune) = get();

    // SSE2 is always available on x86_64
    assert!(caps.has(crate::caps::x86::SSE2));
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", not(miri)))]
  fn test_aarch64_detection() {
    let (caps, _tune) = get();

    // NEON is always available on AArch64
    assert!(caps.has(crate::caps::aarch64::NEON));
  }

  #[test]
  fn test_convenience_functions() {
    let (c, t) = get();
    assert_eq!(caps(), c);
    assert_eq!(tune(), t);
  }

  #[test]
  #[cfg(miri)]
  fn test_miri_returns_portable() {
    let (caps, tune) = get();
    assert_eq!(caps, CpuCaps::NONE);
    assert_eq!(tune, Tune::PORTABLE);
  }

  // Note: Override tests are limited because OnceLock can only be set once.
  // In real usage, overrides should be set early in program initialization.

  #[test]
  fn test_has_override_api() {
    // Just verify the API exists and can be called.
    // We don't set an override here to avoid interfering with other tests.
    let _ = has_override();
  }

  #[test]
  fn test_tune_presets() {
    // Verify all tune presets have reasonable values
    let presets = [
      Tune::DEFAULT,
      Tune::ZEN4,
      Tune::ZEN5,
      Tune::INTEL_SPR,
      Tune::INTEL_ICL,
      Tune::APPLE_M,
      Tune::AARCH64_PMULL,
      Tune::PORTABLE,
    ];

    for tune in presets {
      assert!(tune.simd_threshold > 0);
      assert!(tune.crc_parallelism > 0);
    }
  }

  #[test]
  fn test_cpucaps_none() {
    let caps = CpuCaps::NONE;
    assert!(caps.bits.is_empty());
    // NONE has arch Other
    assert_eq!(caps.arch, crate::caps::Arch::Other);
  }
}
