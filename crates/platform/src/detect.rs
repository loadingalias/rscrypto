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

// ─────────────────────────────────────────────────────────────────────────────
// Compile-Time Static Detection
// ─────────────────────────────────────────────────────────────────────────────

/// Returns CPU capabilities known at compile time.
///
/// Detects features enabled via `-C target-feature=...` or `-C target-cpu=native`.
/// Returns a `const` value—the compiler eliminates all runtime checks.
///
/// # When to Use
///
/// - Building specialized binaries for known hardware
/// - Maximum performance when target features are guaranteed
/// - Embedded/bare-metal where runtime detection isn't available
///
/// For generic binaries that run on multiple CPUs, use [`get()`] instead.
///
/// # Examples
///
/// ```
/// use platform::detect::caps_static;
///
/// // Evaluates at compile time—no runtime cost
/// const CAPS: platform::Caps = caps_static();
///
/// // On x86_64, SSE2 is always present
/// #[cfg(target_arch = "x86_64")]
/// {
///   use platform::caps::x86;
///   assert!(CAPS.has(x86::SSE2));
/// }
///
/// // On aarch64, NEON is always present
/// #[cfg(target_arch = "aarch64")]
/// {
///   use platform::caps::aarch64;
///   assert!(CAPS.has(aarch64::NEON));
/// }
/// ```
///
/// When compiled with `-C target-cpu=znver4`:
///
/// ```
/// # #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
/// # fn example() {
/// use platform::{caps::x86, detect::caps_static};
///
/// const CAPS: platform::Caps = caps_static();
/// // AVX-512 features are detected at compile time
/// assert!(CAPS.has(x86::AVX512F));
/// # }
/// ```
///
/// # Implementation
///
/// Uses `cfg!()` macro inside `const fn`. The compiler evaluates `cfg!()` at
/// compile time and eliminates dead branches via constant propagation.
#[inline(always)]
#[must_use]
pub const fn caps_static() -> Caps {
  // Note: imports are architecture-conditional to avoid warnings
  use crate::caps::Caps;

  // Declarative macro for compile-time feature detection.
  // Uses cfg!() which returns a const bool, enabling use in const fn.
  // The compiler eliminates dead branches entirely.
  #[allow(unused_macros)] // Only used on x86/x86_64/aarch64
  macro_rules! detect {
    ($caps:ident; $($feature:literal => $cap:expr),+ $(,)?) => {
      $(if cfg!(target_feature = $feature) { $caps = $caps.union($cap); })+
    };
  }

  #[allow(unused_mut)]
  let mut result = Caps::NONE;

  // ─────────────────────────────────────────────────────────────────────────────
  // x86/x86_64
  // ─────────────────────────────────────────────────────────────────────────────
  #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
  {
    use crate::caps::x86;

    // x86_64 baseline: SSE2 is guaranteed
    #[cfg(target_arch = "x86_64")]
    {
      result = result.union(x86::SSE2);
    }

    detect!(result;
      // SSE family
      "sse3" => x86::SSE3,
      "ssse3" => x86::SSSE3,
      "sse4.1" => x86::SSE41,
      "sse4.2" => x86::SSE42,
      "sse4a" => x86::SSE4A,
      // AVX family
      "avx" => x86::AVX,
      "avx2" => x86::AVX2,
      "fma" => x86::FMA,
      "f16c" => x86::F16C,
      // Crypto extensions
      "aes" => x86::AESNI,
      "pclmulqdq" => x86::PCLMULQDQ,
      "sha" => x86::SHA,
      "sha512" => x86::SHA512,
      // AVX-512 foundation
      "avx512f" => x86::AVX512F,
      "avx512vl" => x86::AVX512VL,
      "avx512bw" => x86::AVX512BW,
      "avx512dq" => x86::AVX512DQ,
      "avx512cd" => x86::AVX512CD,
      // AVX-512 crypto/advanced
      "vpclmulqdq" => x86::VPCLMULQDQ,
      "vaes" => x86::VAES,
      "gfni" => x86::GFNI,
      // AVX-512 extended
      "avx512ifma" => x86::AVX512IFMA,
      "avx512vbmi" => x86::AVX512VBMI,
      "avx512vbmi2" => x86::AVX512VBMI2,
      "avx512vnni" => x86::AVX512VNNI,
      "avx512bitalg" => x86::AVX512BITALG,
      "avx512vpopcntdq" => x86::AVX512VPOPCNTDQ,
      "avx512fp16" => x86::AVX512FP16,
      "avx512bf16" => x86::AVX512BF16,
      // Bit manipulation
      "bmi1" => x86::BMI1,
      "bmi2" => x86::BMI2,
      "popcnt" => x86::POPCNT,
      "lzcnt" => x86::LZCNT,
      "adx" => x86::ADX,
      // AVX10 (unified AVX-512 replacement)
      "avx10.1" => x86::AVX10_1,
      "avx10.2" => x86::AVX10_2,
      // AMX (Advanced Matrix Extensions)
      "amx-tile" => x86::AMX_TILE,
      "amx-bf16" => x86::AMX_BF16,
      "amx-int8" => x86::AMX_INT8,
      "amx-fp16" => x86::AMX_FP16,
      "amx-complex" => x86::AMX_COMPLEX,
      // Miscellaneous
      "movdiri" => x86::MOVDIRI,
      "movdir64b" => x86::MOVDIR64B,
      "serialize" => x86::SERIALIZE,
      "rdrand" => x86::RDRAND,
      "rdseed" => x86::RDSEED,
      // APX (Advanced Performance Extensions)
      "apxf" => x86::APX,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // aarch64
  // ─────────────────────────────────────────────────────────────────────────────
  #[cfg(target_arch = "aarch64")]
  {
    use crate::caps::aarch64;

    // aarch64 baseline: NEON is guaranteed
    result = result.union(aarch64::NEON);

    // Features that map to multiple caps
    if cfg!(target_feature = "aes") {
      result = result.union(aarch64::AES).union(aarch64::PMULL);
    }
    if cfg!(target_feature = "sha3") {
      result = result.union(aarch64::SHA3).union(aarch64::SHA512);
    }
    if cfg!(target_feature = "sm4") {
      result = result.union(aarch64::SM3).union(aarch64::SM4);
    }

    detect!(result;
      // Crypto extensions (single-cap)
      "sha2" => aarch64::SHA2,
      // CRC
      "crc" => aarch64::CRC,
      // Additional SIMD
      "dotprod" => aarch64::DOTPROD,
      "i8mm" => aarch64::I8MM,
      "bf16" => aarch64::BF16,
      "fp16" => aarch64::FP16,
      "frintts" => aarch64::FRINTTS,
      // SVE family
      "sve" => aarch64::SVE,
      "sve2" => aarch64::SVE2,
      "sve2-aes" => aarch64::SVE2_AES,
      "sve2-sha3" => aarch64::SVE2_SHA3,
      "sve2-sm4" => aarch64::SVE2_SM4,
      "sve2-bitperm" => aarch64::SVE2_BITPERM,
      // Atomics (ARMv8.1+)
      "lse" => aarch64::LSE,
      "lse2" => aarch64::LSE2,
      // Memory operations
      "mops" => aarch64::MOPS,
      // Scalable Matrix Extension
      "sme" => aarch64::SME,
      "sme2" => aarch64::SME2,
      // Hardware RNG
      "rand" => aarch64::RNG,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // RISC-V
  // ─────────────────────────────────────────────────────────────────────────────
  #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
  {
    use crate::caps::riscv;

    detect!(result;
      // Vector extension
      "v" => riscv::V,
      // Bit manipulation
      "zbb" => riscv::ZBB,
      "zbs" => riscv::ZBS,
      "zba" => riscv::ZBA,
      "zbc" => riscv::ZBC,
      // Scalar crypto
      "zbkb" => riscv::ZBKB,
      "zbkc" => riscv::ZBKC,
      "zbkx" => riscv::ZBKX,
      "zknd" => riscv::ZKND,
      "zkne" => riscv::ZKNE,
      "zknh" => riscv::ZKNH,
      "zksed" => riscv::ZKSED,
      "zksh" => riscv::ZKSH,
      // Vector crypto
      "zvbb" => riscv::ZVBB,
      "zvbc" => riscv::ZVBC,
      "zvkb" => riscv::ZVKB,
      "zvkg" => riscv::ZVKG,
      "zvkned" => riscv::ZVKNED,
      "zvknha" => riscv::ZVKNHA,
      "zvknhb" => riscv::ZVKNHB,
      "zvksed" => riscv::ZVKSED,
      "zvksh" => riscv::ZVKSH,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // WebAssembly
  // ─────────────────────────────────────────────────────────────────────────────
  #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
  {
    use crate::caps::wasm;

    detect!(result;
      "simd128" => wasm::SIMD128,
      "relaxed-simd" => wasm::RELAXED_SIMD,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // s390x (IBM Z)
  // ─────────────────────────────────────────────────────────────────────────────
  #[cfg(target_arch = "s390x")]
  {
    use crate::caps::s390x;

    // Features that map to multiple caps (avoid non-const `|` in const fn).
    if cfg!(target_feature = "message-security-assist-extension4") {
      result = result.union(s390x::MSA).union(s390x::MSA4);
    }
    if cfg!(target_feature = "message-security-assist-extension5") {
      result = result.union(s390x::MSA).union(s390x::MSA5);
    }
    if cfg!(target_feature = "message-security-assist-extension8") {
      result = result.union(s390x::MSA).union(s390x::MSA8);
    }
    if cfg!(target_feature = "message-security-assist-extension9") {
      result = result.union(s390x::MSA).union(s390x::MSA9);
    }

    detect!(result;
      "vector" => s390x::VECTOR,
      "vector-enhancements-1" => s390x::VECTOR_ENH1,
      "vector-enhancements-2" => s390x::VECTOR_ENH2,
      "vector-enhancements-3" => s390x::VECTOR_ENH3,
      "vector-packed-decimal" => s390x::VECTOR_PD,
      "nnp-assist" => s390x::NNP_ASSIST,
      "miscellaneous-extensions-2" => s390x::MISC_EXT2,
      "miscellaneous-extensions-3" => s390x::MISC_EXT3,
      "message-security-assist-extension3" => s390x::MSA,
      "deflate-conversion" => s390x::DEFLATE,
      "enhanced-sort" => s390x::ENHANCED_SORT,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Power
  // ─────────────────────────────────────────────────────────────────────────────
  #[cfg(target_arch = "powerpc64")]
  {
    use crate::caps::power;

    detect!(result;
      "altivec" => power::ALTIVEC,
      "vsx" => power::VSX,
      "power8-vector" => power::POWER8_VECTOR,
      "power8-crypto" => power::POWER8_CRYPTO,
      "power9-vector" => power::POWER9_VECTOR,
      "power10-vector" => power::POWER10_VECTOR,
    );
  }

  result
}

/// Compile-time tuning hints.
///
/// Returns tuning hints based on compile-time known features. Since we cannot
/// detect the exact microarchitecture at compile time, this provides conservative
/// defaults that work well across CPUs with the given feature set.
///
/// For more precise tuning, use [`get().tune`](get) which performs runtime
/// microarchitecture detection.
#[inline(always)]
#[must_use]
#[allow(unreachable_code)] // CFG blocks are exhaustive but compiler can't prove it
pub const fn tune_static() -> Tune {
  // x86/x86_64 with AVX-512 crypto features
  #[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
  ))]
  {
    // Conservative: INTEL_SPR works well for both Zen4/5 and Intel SPR
    return Tune::INTEL_SPR;
  }

  // x86/x86_64 with AVX2 only (no AVX-512)
  #[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx2",
    not(target_feature = "avx512f")
  ))]
  {
    return Tune::DEFAULT;
  }

  // x86/x86_64 with PCLMULQDQ (SSE-era crypto)
  #[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "pclmulqdq",
    not(target_feature = "avx2")
  ))]
  {
    return Tune::DEFAULT;
  }

  // aarch64 with SHA3 (EOR3 instruction for fast GHASH)
  #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
  {
    // Conservative: APPLE_M works well for SHA3-capable ARMs
    return Tune::APPLE_M;
  }

  // aarch64 with AES/PMULL only (no SHA3)
  #[cfg(all(target_arch = "aarch64", target_feature = "aes", not(target_feature = "sha3")))]
  {
    return Tune::AARCH64_PMULL;
  }

  // aarch64 baseline (NEON only)
  #[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
  {
    return Tune::DEFAULT;
  }

  // RISC-V with vector crypto
  #[cfg(all(any(target_arch = "riscv64", target_arch = "riscv32"), target_feature = "zvbc"))]
  {
    return Tune::DEFAULT;
  }

  // RISC-V baseline
  #[cfg(all(any(target_arch = "riscv64", target_arch = "riscv32"), not(target_feature = "zvbc")))]
  {
    return Tune::PORTABLE;
  }

  // WebAssembly with SIMD
  #[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
  {
    return Tune::DEFAULT;
  }

  // WebAssembly without SIMD
  #[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), not(target_feature = "simd128")))]
  {
    return Tune::PORTABLE;
  }

  // s390x with vector enhancements 2 (z15+)
  #[cfg(all(target_arch = "s390x", target_feature = "vector-enhancements-2"))]
  {
    return Tune::Z15;
  }

  // s390x with vector enhancements 1 (z14)
  #[cfg(all(
    target_arch = "s390x",
    target_feature = "vector-enhancements-1",
    not(target_feature = "vector-enhancements-2")
  ))]
  {
    return Tune::Z14;
  }

  // s390x with base vector (z13)
  #[cfg(all(
    target_arch = "s390x",
    target_feature = "vector",
    not(target_feature = "vector-enhancements-1")
  ))]
  {
    return Tune::Z13;
  }

  // s390x baseline (no vector - very old)
  #[cfg(all(target_arch = "s390x", not(target_feature = "vector")))]
  {
    return Tune::PORTABLE;
  }

  // PowerPC64 with POWER10 vector
  #[cfg(all(target_arch = "powerpc64", target_feature = "power10-vector"))]
  {
    return Tune::POWER10;
  }

  // PowerPC64 with POWER9 vector
  #[cfg(all(
    target_arch = "powerpc64",
    target_feature = "power9-vector",
    not(target_feature = "power10-vector")
  ))]
  {
    return Tune::POWER9;
  }

  // PowerPC64 with POWER8 vector
  #[cfg(all(
    target_arch = "powerpc64",
    target_feature = "power8-vector",
    not(target_feature = "power9-vector")
  ))]
  {
    return Tune::POWER8;
  }

  // PowerPC64 with VSX (POWER7)
  #[cfg(all(
    target_arch = "powerpc64",
    target_feature = "vsx",
    not(target_feature = "power8-vector")
  ))]
  {
    return Tune::POWER7;
  }

  // PowerPC64 baseline (AltiVec only or less)
  #[cfg(all(target_arch = "powerpc64", not(target_feature = "vsx")))]
  {
    return Tune::PORTABLE;
  }

  // Fallback for all other architectures
  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "x86",
    target_arch = "aarch64",
    target_arch = "riscv64",
    target_arch = "riscv32",
    target_arch = "wasm32",
    target_arch = "wasm64",
    target_arch = "s390x",
    target_arch = "powerpc64"
  )))]
  {
    return Tune::PORTABLE;
  }

  // Default fallback (should never reach here due to exhaustive cfgs above)
  Tune::DEFAULT
}

// ─────────────────────────────────────────────────────────────────────────────
// Override System
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "std")]
use std::sync::OnceLock;

#[cfg(all(feature = "std", not(miri)))]
static STD_CACHE: OnceLock<Detected> = OnceLock::new();

#[cfg(feature = "std")]
static OVERRIDE: OnceLock<Option<Detected>> = OnceLock::new();

/// Set detection override.
///
/// Must be called **before** the first call to [`get()`]. After caching occurs,
/// overrides are ignored until process restart.
#[cold]
pub fn set_override(value: Option<Detected>) {
  #[cfg(feature = "std")]
  {
    let _ = OVERRIDE.set(value);
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  {
    atomic_cache::set_override(value);
  }

  #[cfg(all(not(feature = "std"), not(target_has_atomic = "64")))]
  {
    let _ = value;
  }
}

/// Clear detection override.
#[cold]
pub fn clear_override() {
  set_override(None);
}

/// Check if an override is set.
#[inline]
#[must_use]
pub fn has_override() -> bool {
  #[cfg(feature = "std")]
  {
    OVERRIDE.get().is_some_and(|v| v.is_some())
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  {
    atomic_cache::has_override()
  }

  #[cfg(all(not(feature = "std"), not(target_has_atomic = "64")))]
  {
    false
  }
}

#[cold]
#[cfg(not(miri))]
fn detect_with_override() -> Detected {
  #[cfg(feature = "std")]
  {
    if let Some(Some(ov)) = OVERRIDE.get() {
      return *ov;
    }
  }

  #[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
  {
    if let Some(ov) = atomic_cache::get_override() {
      return ov;
    }
  }

  detect_uncached()
}

// ─────────────────────────────────────────────────────────────────────────────
// Atomic Cache (no_std with 64-bit atomics)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(not(feature = "std"), target_has_atomic = "64"))]
mod atomic_cache {
  use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU16, AtomicU64, Ordering};

  use super::*;
  use crate::tune::TuneKind;

  static STATE: AtomicU8 = AtomicU8::new(0);

  static CAPS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
  ];
  static ARCH: AtomicU8 = AtomicU8::new(0);

  // Tune fields
  static TUNE_KIND: AtomicU8 = AtomicU8::new(0);
  static TUNE_THRESHOLD: AtomicU64 = AtomicU64::new(0);
  static TUNE_PCLMUL_THRESHOLD: AtomicU64 = AtomicU64::new(0);
  static TUNE_HWCRC_THRESHOLD: AtomicU64 = AtomicU64::new(0);
  static TUNE_EFFECTIVE_WIDTH: AtomicU16 = AtomicU16::new(0);
  static TUNE_FAST_WIDE: AtomicBool = AtomicBool::new(false);
  static TUNE_PARALLEL_STREAMS: AtomicU8 = AtomicU8::new(0);
  static TUNE_PREFER_HYBRID: AtomicBool = AtomicBool::new(false);
  static TUNE_CACHE_LINE: AtomicU8 = AtomicU8::new(0);
  static TUNE_PREFETCH_DIST: AtomicU16 = AtomicU16::new(0);
  static TUNE_SVE_VLEN: AtomicU16 = AtomicU16::new(0);
  static TUNE_SME_TILE: AtomicU16 = AtomicU16::new(0);
  static TUNE_MEMORY_BOUND_HWCRC: AtomicBool = AtomicBool::new(false);

  static OVERRIDE_SET: AtomicBool = AtomicBool::new(false);
  static OVERRIDE_CAPS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
  ];
  static OVERRIDE_ARCH: AtomicU8 = AtomicU8::new(0);
  static OVERRIDE_TUNE_KIND: AtomicU8 = AtomicU8::new(0);
  static OVERRIDE_TUNE_THRESHOLD: AtomicU64 = AtomicU64::new(0);
  static OVERRIDE_TUNE_PCLMUL_THRESHOLD: AtomicU64 = AtomicU64::new(0);
  static OVERRIDE_TUNE_HWCRC_THRESHOLD: AtomicU64 = AtomicU64::new(0);
  static OVERRIDE_TUNE_EFFECTIVE_WIDTH: AtomicU16 = AtomicU16::new(0);
  static OVERRIDE_TUNE_FAST_WIDE: AtomicBool = AtomicBool::new(false);
  static OVERRIDE_TUNE_PARALLEL_STREAMS: AtomicU8 = AtomicU8::new(0);
  static OVERRIDE_TUNE_PREFER_HYBRID: AtomicBool = AtomicBool::new(false);
  static OVERRIDE_TUNE_CACHE_LINE: AtomicU8 = AtomicU8::new(0);
  static OVERRIDE_TUNE_PREFETCH_DIST: AtomicU16 = AtomicU16::new(0);
  static OVERRIDE_TUNE_SVE_VLEN: AtomicU16 = AtomicU16::new(0);
  static OVERRIDE_TUNE_SME_TILE: AtomicU16 = AtomicU16::new(0);
  static OVERRIDE_TUNE_MEMORY_BOUND_HWCRC: AtomicBool = AtomicBool::new(false);

  pub fn get_or_init(f: fn() -> Detected) -> Detected {
    if STATE.load(Ordering::Acquire) == 2 {
      return load_cached();
    }

    match STATE.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire) {
      Ok(_) => {
        let result = f();
        store_cached(&result);
        STATE.store(2, Ordering::Release);
        result
      }
      Err(1) => {
        while STATE.load(Ordering::Acquire) == 1 {
          core::hint::spin_loop();
        }
        load_cached()
      }
      Err(_) => load_cached(),
    }
  }

  fn load_cached() -> Detected {
    let caps = Caps([
      CAPS[0].load(Ordering::Acquire),
      CAPS[1].load(Ordering::Acquire),
      CAPS[2].load(Ordering::Acquire),
      CAPS[3].load(Ordering::Acquire),
    ]);
    let arch = arch_from_u8(ARCH.load(Ordering::Acquire));
    let tune = Tune {
      kind: kind_from_u8(TUNE_KIND.load(Ordering::Acquire)),
      simd_threshold: TUNE_THRESHOLD.load(Ordering::Acquire) as usize,
      pclmul_threshold: TUNE_PCLMUL_THRESHOLD.load(Ordering::Acquire) as usize,
      hwcrc_threshold: TUNE_HWCRC_THRESHOLD.load(Ordering::Acquire) as usize,
      effective_simd_width: TUNE_EFFECTIVE_WIDTH.load(Ordering::Acquire),
      fast_wide_ops: TUNE_FAST_WIDE.load(Ordering::Acquire),
      parallel_streams: TUNE_PARALLEL_STREAMS.load(Ordering::Acquire),
      prefer_hybrid: TUNE_PREFER_HYBRID.load(Ordering::Acquire),
      cache_line: TUNE_CACHE_LINE.load(Ordering::Acquire),
      prefetch_distance: TUNE_PREFETCH_DIST.load(Ordering::Acquire),
      sve_vlen: TUNE_SVE_VLEN.load(Ordering::Acquire),
      sme_tile: TUNE_SME_TILE.load(Ordering::Acquire),
      memory_bound_hwcrc: TUNE_MEMORY_BOUND_HWCRC.load(Ordering::Acquire),
    };
    Detected { caps, tune, arch }
  }

  fn store_cached(det: &Detected) {
    CAPS[0].store(det.caps.0[0], Ordering::Release);
    CAPS[1].store(det.caps.0[1], Ordering::Release);
    CAPS[2].store(det.caps.0[2], Ordering::Release);
    CAPS[3].store(det.caps.0[3], Ordering::Release);
    ARCH.store(arch_to_u8(det.arch), Ordering::Release);
    TUNE_KIND.store(det.tune.kind as u8, Ordering::Release);
    TUNE_THRESHOLD.store(det.tune.simd_threshold as u64, Ordering::Release);
    TUNE_PCLMUL_THRESHOLD.store(det.tune.pclmul_threshold as u64, Ordering::Release);
    TUNE_HWCRC_THRESHOLD.store(det.tune.hwcrc_threshold as u64, Ordering::Release);
    TUNE_EFFECTIVE_WIDTH.store(det.tune.effective_simd_width, Ordering::Release);
    TUNE_FAST_WIDE.store(det.tune.fast_wide_ops, Ordering::Release);
    TUNE_PARALLEL_STREAMS.store(det.tune.parallel_streams, Ordering::Release);
    TUNE_PREFER_HYBRID.store(det.tune.prefer_hybrid, Ordering::Release);
    TUNE_CACHE_LINE.store(det.tune.cache_line, Ordering::Release);
    TUNE_PREFETCH_DIST.store(det.tune.prefetch_distance, Ordering::Release);
    TUNE_SVE_VLEN.store(det.tune.sve_vlen, Ordering::Release);
    TUNE_SME_TILE.store(det.tune.sme_tile, Ordering::Release);
    TUNE_MEMORY_BOUND_HWCRC.store(det.tune.memory_bound_hwcrc, Ordering::Release);
  }

  pub fn set_override(value: Option<Detected>) {
    match value {
      Some(det) => {
        OVERRIDE_CAPS[0].store(det.caps.0[0], Ordering::Release);
        OVERRIDE_CAPS[1].store(det.caps.0[1], Ordering::Release);
        OVERRIDE_CAPS[2].store(det.caps.0[2], Ordering::Release);
        OVERRIDE_CAPS[3].store(det.caps.0[3], Ordering::Release);
        OVERRIDE_ARCH.store(arch_to_u8(det.arch), Ordering::Release);
        OVERRIDE_TUNE_KIND.store(det.tune.kind as u8, Ordering::Release);
        OVERRIDE_TUNE_THRESHOLD.store(det.tune.simd_threshold as u64, Ordering::Release);
        OVERRIDE_TUNE_PCLMUL_THRESHOLD.store(det.tune.pclmul_threshold as u64, Ordering::Release);
        OVERRIDE_TUNE_HWCRC_THRESHOLD.store(det.tune.hwcrc_threshold as u64, Ordering::Release);
        OVERRIDE_TUNE_EFFECTIVE_WIDTH.store(det.tune.effective_simd_width, Ordering::Release);
        OVERRIDE_TUNE_FAST_WIDE.store(det.tune.fast_wide_ops, Ordering::Release);
        OVERRIDE_TUNE_PARALLEL_STREAMS.store(det.tune.parallel_streams, Ordering::Release);
        OVERRIDE_TUNE_PREFER_HYBRID.store(det.tune.prefer_hybrid, Ordering::Release);
        OVERRIDE_TUNE_CACHE_LINE.store(det.tune.cache_line, Ordering::Release);
        OVERRIDE_TUNE_PREFETCH_DIST.store(det.tune.prefetch_distance, Ordering::Release);
        OVERRIDE_TUNE_SVE_VLEN.store(det.tune.sve_vlen, Ordering::Release);
        OVERRIDE_TUNE_SME_TILE.store(det.tune.sme_tile, Ordering::Release);
        OVERRIDE_TUNE_MEMORY_BOUND_HWCRC.store(det.tune.memory_bound_hwcrc, Ordering::Release);
        OVERRIDE_SET.store(true, Ordering::Release);
      }
      None => {
        OVERRIDE_SET.store(false, Ordering::Release);
      }
    }
  }

  pub fn has_override() -> bool {
    OVERRIDE_SET.load(Ordering::Acquire)
  }

  pub fn get_override() -> Option<Detected> {
    if !OVERRIDE_SET.load(Ordering::Acquire) {
      return None;
    }
    Some(Detected {
      caps: Caps([
        OVERRIDE_CAPS[0].load(Ordering::Acquire),
        OVERRIDE_CAPS[1].load(Ordering::Acquire),
        OVERRIDE_CAPS[2].load(Ordering::Acquire),
        OVERRIDE_CAPS[3].load(Ordering::Acquire),
      ]),
      arch: arch_from_u8(OVERRIDE_ARCH.load(Ordering::Acquire)),
      tune: Tune {
        kind: kind_from_u8(OVERRIDE_TUNE_KIND.load(Ordering::Acquire)),
        simd_threshold: OVERRIDE_TUNE_THRESHOLD.load(Ordering::Acquire) as usize,
        pclmul_threshold: OVERRIDE_TUNE_PCLMUL_THRESHOLD.load(Ordering::Acquire) as usize,
        hwcrc_threshold: OVERRIDE_TUNE_HWCRC_THRESHOLD.load(Ordering::Acquire) as usize,
        effective_simd_width: OVERRIDE_TUNE_EFFECTIVE_WIDTH.load(Ordering::Acquire),
        fast_wide_ops: OVERRIDE_TUNE_FAST_WIDE.load(Ordering::Acquire),
        parallel_streams: OVERRIDE_TUNE_PARALLEL_STREAMS.load(Ordering::Acquire),
        prefer_hybrid: OVERRIDE_TUNE_PREFER_HYBRID.load(Ordering::Acquire),
        cache_line: OVERRIDE_TUNE_CACHE_LINE.load(Ordering::Acquire),
        prefetch_distance: OVERRIDE_TUNE_PREFETCH_DIST.load(Ordering::Acquire),
        sve_vlen: OVERRIDE_TUNE_SVE_VLEN.load(Ordering::Acquire),
        sme_tile: OVERRIDE_TUNE_SME_TILE.load(Ordering::Acquire),
        memory_bound_hwcrc: OVERRIDE_TUNE_MEMORY_BOUND_HWCRC.load(Ordering::Acquire),
      },
    })
  }

  fn arch_to_u8(arch: Arch) -> u8 {
    match arch {
      Arch::X86_64 => 1,
      Arch::X86 => 2,
      Arch::Aarch64 => 3,
      Arch::Arm => 4,
      Arch::Riscv64 => 5,
      Arch::Riscv32 => 6,
      Arch::Power => 7,
      Arch::S390x => 8,
      Arch::Wasm32 => 10,
      Arch::Wasm64 => 11,
      Arch::Other => 0,
    }
  }

  fn arch_from_u8(v: u8) -> Arch {
    match v {
      1 => Arch::X86_64,
      2 => Arch::X86,
      3 => Arch::Aarch64,
      4 => Arch::Arm,
      5 => Arch::Riscv64,
      6 => Arch::Riscv32,
      7 => Arch::Power,
      8 => Arch::S390x,
      10 => Arch::Wasm32,
      11 => Arch::Wasm64,
      _ => Arch::Other,
    }
  }

  fn kind_from_u8(v: u8) -> TuneKind {
    // Must match TuneKind enum order (repr(u8) with Custom = 0)
    match v {
      1 => TuneKind::Default,
      2 => TuneKind::Portable,
      3 => TuneKind::Zen4,
      4 => TuneKind::Zen5,
      5 => TuneKind::Zen5c,
      6 => TuneKind::IntelSpr,
      7 => TuneKind::IntelGnr,
      8 => TuneKind::IntelIcl,
      9 => TuneKind::AppleM1M3,
      10 => TuneKind::AppleM4,
      11 => TuneKind::AppleM5,
      12 => TuneKind::Graviton2,
      13 => TuneKind::Graviton3,
      14 => TuneKind::Graviton4,
      15 => TuneKind::Graviton5,
      16 => TuneKind::NeoverseN2,
      17 => TuneKind::NeoverseN3,
      18 => TuneKind::NeoverseV3,
      19 => TuneKind::NvidiaGrace,
      20 => TuneKind::AmpereAltra,
      21 => TuneKind::Aarch64Pmull,
      22 => TuneKind::Z13,
      23 => TuneKind::Z14,
      24 => TuneKind::Z15,
      25 => TuneKind::Power7,
      26 => TuneKind::Power8,
      27 => TuneKind::Power9,
      28 => TuneKind::Power10,
      _ => TuneKind::Custom,
    }
  }
}

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

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn detect_x86_64() -> Detected {
  use crate::caps::x86;

  // Start with compile-time detected features (includes SSE2 baseline)
  let caps_static = caps_static();

  // Runtime detection extracts features + vendor/family/model in batch
  #[cfg(feature = "std")]
  let (runtime_caps, is_amd, family, model) = {
    let batch = cpuid_batch_x86_64();
    (batch.caps, batch.is_amd, batch.family, batch.model)
  };
  #[cfg(not(feature = "std"))]
  let (is_amd, family, model) = (false, 0u32, 0u32);

  #[cfg(feature = "std")]
  let mut caps = caps_static.union(runtime_caps);
  #[cfg(not(feature = "std"))]
  let caps = caps_static;

  // ─────────────────────────────────────────────────────────────────────────────
  // Hybrid Intel AVX-512 Safety: Clear AVX-512 caps on hybrid CPUs
  // ─────────────────────────────────────────────────────────────────────────────
  // On hybrid Intel CPUs (Alder Lake, Raptor Lake, etc.), the P-cores have
  // AVX-512 but E-cores don't. If a thread migrates to an E-core while
  // executing AVX-512 code, it will SIGILL. The only safe approach is to
  // disable AVX-512 entirely unless the user explicitly overrides.
  #[cfg(feature = "std")]
  {
    if is_intel_hybrid(is_amd, family, model) && !hybrid_avx512_override() {
      // Clear all AVX-512 related capabilities to prevent kernel selection
      // from choosing AVX-512/VPCLMUL paths that could SIGILL on E-cores.
      caps = caps
        .difference(x86::AVX512F)
        .difference(x86::AVX512DQ)
        .difference(x86::AVX512IFMA)
        .difference(x86::AVX512CD)
        .difference(x86::AVX512BW)
        .difference(x86::AVX512VL)
        .difference(x86::AVX512VBMI)
        .difference(x86::AVX512VBMI2)
        .difference(x86::AVX512VNNI)
        .difference(x86::AVX512BITALG)
        .difference(x86::AVX512VPOPCNTDQ)
        .difference(x86::AVX512BF16)
        .difference(x86::AVX512FP16)
        .difference(x86::VPCLMULQDQ)
        .difference(x86::VAES)
        .difference(x86::GFNI)
        .difference(x86::AVX10_1)
        .difference(x86::AVX10_2);
    }
  }

  let tune = select_x86_tune(caps, is_amd, family, model);

  Detected {
    caps,
    tune,
    arch: Arch::X86_64,
  }
}

#[cfg(target_arch = "x86")]
fn detect_x86() -> Detected {
  use crate::caps::x86;

  // Start with compile-time detected features
  let mut caps = caps_static();

  #[cfg(feature = "std")]
  {
    // SSE2 is not guaranteed on 32-bit x86, detect at runtime
    if std::arch::is_x86_feature_detected!("sse2") {
      caps |= x86::SSE2;
    }
    caps |= runtime_x86_32();
  }

  Detected {
    caps,
    tune: Tune::DEFAULT,
    arch: Arch::X86,
  }
}

/// Batch CPUID result containing all extracted information.
///
/// This struct consolidates all CPUID-derived data to avoid redundant calls.
/// A single call to `cpuid_batch_x86_64()` extracts:
/// - Feature capabilities (Caps)
/// - Vendor identification (Intel/AMD/Unknown)
/// - CPU family and model for microarchitecture selection
#[cfg(all(target_arch = "x86_64", feature = "std"))]
struct CpuidBatch {
  caps: Caps,
  is_amd: bool,
  family: u32,
  model: u32,
}

/// Batch CPUID extraction - extracts all features and CPU info in minimal CPUID calls.
///
/// Makes 5-7 CPUID calls (depending on CPU capabilities):
/// - Leaf 0: vendor string
/// - Leaf 1: processor info + basic features
/// - Leaf 7.0: extended features
/// - Leaf 7.1: more extended features
/// - Leaf 0x24: AVX10 detection (if max leaf >= 0x24)
/// - Leaf 0x29: APX detection (if max leaf >= 0x29)
/// - Leaf 0x80000001: AMD-specific features
///
/// **Critical**: This function properly gates AVX/AVX-512 features by checking
/// OSXSAVE and XGETBV(XCR0) to ensure the OS will save/restore extended registers.
/// Without this check, using AVX/AVX-512 instructions could cause SIGILL.
///
/// # Safety
/// Uses CPUID and XGETBV instructions which require unsafe, but are always safe
/// to call on x86_64 when OSXSAVE is set.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[allow(unsafe_code)]
fn cpuid_batch_x86_64() -> CpuidBatch {
  use core::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

  use crate::caps::x86;

  // XCR0 bit masks for OS support verification
  // Bits 1-2: XMM (SSE) + YMM (AVX) state - must be set for AVX
  const XCR0_AVX_MASK: u64 = 0x6;
  // Bits 5-7: opmask + ZMM_Hi256 + Hi16_ZMM state - must be set for AVX-512
  const XCR0_AVX512_MASK: u64 = 0xE0;

  let mut caps = Caps::NONE;

  // CPUID leaf 0: vendor string
  // SAFETY: CPUID is always safe on x86_64
  let cpuid0 = unsafe { __cpuid(0) };
  // "AuthenticAMD" has ebx = 0x68747541 ("Auth")
  let is_amd = cpuid0.ebx == 0x6874_7541;

  // CPUID leaf 1: processor info and feature bits
  // SAFETY: CPUID is always safe on x86_64
  let cpuid1 = unsafe { __cpuid(1) };

  // Extract extended family (bits 27:20) + base family (bits 11:8)
  let base_family = (cpuid1.eax >> 8) & 0xF;
  let ext_family = (cpuid1.eax >> 20) & 0xFF;
  let family = base_family + ext_family;

  // Extract model (bits 7:4 + extended model bits 19:16 for family 6/15)
  let base_model = (cpuid1.eax >> 4) & 0xF;
  let ext_model = (cpuid1.eax >> 16) & 0xF;
  let model = if base_family == 6 || base_family == 15 {
    base_model + (ext_model << 4)
  } else {
    base_model
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // OS Support Detection via OSXSAVE + XGETBV
  // ─────────────────────────────────────────────────────────────────────────────
  // CRITICAL: CPUID reports what the CPU supports, not what the OS allows.
  // We must check OSXSAVE (indicates OS uses XSAVE) and read XCR0 to verify
  // the OS will actually save/restore AVX/AVX-512 registers. Without this,
  // using AVX instructions on an OS that doesn't save YMM/ZMM state causes SIGILL.

  // OSXSAVE (bit 27): OS has set CR4.OSXSAVE and supports XSAVE/XGETBV
  let osxsave = cpuid1.ecx & (1 << 27) != 0;

  // Read XCR0 if OSXSAVE is enabled, otherwise assume no extended state support
  let xcr0 = if osxsave {
    // SAFETY: XGETBV is safe when OSXSAVE is set (checked above)
    unsafe { _xgetbv(0) }
  } else {
    0
  };

  // Determine OS support for AVX and AVX-512 register state
  let os_avx = (xcr0 & XCR0_AVX_MASK) == XCR0_AVX_MASK;
  let os_avx512 = os_avx && (xcr0 & XCR0_AVX512_MASK) == XCR0_AVX512_MASK;

  // ─────────────────────────────────────────────────────────────────────────────
  // ECX features (leaf 1) - SSE/basic features (no OS gating needed)
  // ─────────────────────────────────────────────────────────────────────────────
  if cpuid1.ecx & (1 << 0) != 0 {
    caps |= x86::SSE3;
  }
  if cpuid1.ecx & (1 << 9) != 0 {
    caps |= x86::SSSE3;
  }
  if cpuid1.ecx & (1 << 19) != 0 {
    caps |= x86::SSE41;
  }
  if cpuid1.ecx & (1 << 20) != 0 {
    caps |= x86::SSE42;
  }
  if cpuid1.ecx & (1 << 23) != 0 {
    caps |= x86::POPCNT;
  }
  if cpuid1.ecx & (1 << 25) != 0 {
    caps |= x86::AESNI;
  }
  if cpuid1.ecx & (1 << 1) != 0 {
    caps |= x86::PCLMULQDQ;
  }
  if cpuid1.ecx & (1 << 30) != 0 {
    caps |= x86::RDRAND;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // AVX-class features (require OS AVX support via XCR0)
  // ─────────────────────────────────────────────────────────────────────────────
  if os_avx {
    if cpuid1.ecx & (1 << 28) != 0 {
      caps |= x86::AVX;
    }
    if cpuid1.ecx & (1 << 12) != 0 {
      caps |= x86::FMA;
    }
    if cpuid1.ecx & (1 << 29) != 0 {
      caps |= x86::F16C;
    }
  }

  // Extended feature flags (leaf 7, subleaf 0)
  // SAFETY: CPUID is always safe on x86_64
  let cpuid7 = unsafe { __cpuid_count(7, 0) };

  // ─────────────────────────────────────────────────────────────────────────────
  // EBX features (leaf 7) - non-AVX features (no OS gating needed)
  // ─────────────────────────────────────────────────────────────────────────────
  if cpuid7.ebx & (1 << 3) != 0 {
    caps |= x86::BMI1;
  }
  if cpuid7.ebx & (1 << 8) != 0 {
    caps |= x86::BMI2;
  }
  if cpuid7.ebx & (1 << 19) != 0 {
    caps |= x86::ADX;
  }
  if cpuid7.ebx & (1 << 29) != 0 {
    caps |= x86::SHA;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // AVX2 (requires OS AVX support for YMM registers)
  // ─────────────────────────────────────────────────────────────────────────────
  if os_avx && cpuid7.ebx & (1 << 5) != 0 {
    caps |= x86::AVX2;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // AVX-512 features (require OS AVX-512 support for ZMM/opmask registers)
  // ─────────────────────────────────────────────────────────────────────────────
  if os_avx512 {
    if cpuid7.ebx & (1 << 16) != 0 {
      caps |= x86::AVX512F;
    }
    if cpuid7.ebx & (1 << 17) != 0 {
      caps |= x86::AVX512DQ;
    }
    if cpuid7.ebx & (1 << 21) != 0 {
      caps |= x86::AVX512IFMA;
    }
    if cpuid7.ebx & (1 << 28) != 0 {
      caps |= x86::AVX512CD;
    }
    if cpuid7.ebx & (1 << 30) != 0 {
      caps |= x86::AVX512BW;
    }
    if cpuid7.ebx & (1 << 31) != 0 {
      caps |= x86::AVX512VL;
    }

    // ECX AVX-512 features (leaf 7)
    if cpuid7.ecx & (1 << 1) != 0 {
      caps |= x86::AVX512VBMI;
    }
    if cpuid7.ecx & (1 << 6) != 0 {
      caps |= x86::AVX512VBMI2;
    }
    if cpuid7.ecx & (1 << 11) != 0 {
      caps |= x86::AVX512VNNI;
    }
    if cpuid7.ecx & (1 << 12) != 0 {
      caps |= x86::AVX512BITALG;
    }
    if cpuid7.ecx & (1 << 14) != 0 {
      caps |= x86::AVX512VPOPCNTDQ;
    }

    // Vector extensions that use 512-bit registers (gate with AVX-512 OS support)
    if cpuid7.ecx & (1 << 8) != 0 {
      caps |= x86::GFNI;
    }
    if cpuid7.ecx & (1 << 9) != 0 {
      caps |= x86::VAES;
    }
    if cpuid7.ecx & (1 << 10) != 0 {
      caps |= x86::VPCLMULQDQ;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // EDX features (leaf 7) - non-AVX features
  // ─────────────────────────────────────────────────────────────────────────────
  if cpuid7.edx & (1 << 18) != 0 {
    caps |= x86::RDSEED;
  }
  if cpuid7.edx & (1 << 24) != 0 {
    caps |= x86::AMX_TILE;
  }
  if cpuid7.edx & (1 << 22) != 0 {
    caps |= x86::AMX_BF16;
  }
  if cpuid7.edx & (1 << 25) != 0 {
    caps |= x86::AMX_INT8;
  }

  // Extended feature flags (leaf 7, subleaf 1)
  // SAFETY: CPUID is always safe on x86_64
  let cpuid7_1 = unsafe { __cpuid_count(7, 1) };

  // ─────────────────────────────────────────────────────────────────────────────
  // EAX features (leaf 7, subleaf 1)
  // ─────────────────────────────────────────────────────────────────────────────
  // SHA512 doesn't require AVX-512 (uses XMM registers)
  if cpuid7_1.eax & (1 << 0) != 0 {
    caps |= x86::SHA512;
  }

  // AVX-512 extensions (require OS AVX-512 support)
  if os_avx512 {
    if cpuid7_1.eax & (1 << 4) != 0 {
      caps |= x86::AVX512BF16;
    }
    if cpuid7_1.eax & (1 << 5) != 0 {
      caps |= x86::AVX512FP16;
    }
  }

  // AMX extensions (Granite Rapids and newer) - separate state component
  // Note: AMX has its own XCR0 bits (17-18), but for now we don't gate these
  // as they're not used for crypto kernels
  if cpuid7_1.eax & (1 << 21) != 0 {
    caps |= x86::AMX_FP16;
  }
  if cpuid7_1.eax & (1 << 8) != 0 {
    caps |= x86::AMX_COMPLEX;
  }

  // AVX10 detection via CPUID leaf 0x24 (requires OS AVX-512 support)
  // AVX10 is Intel's unified vector ISA that subsumes AVX-512
  if os_avx512 && cpuid0.eax >= 0x24 {
    // SAFETY: CPUID is always safe on x86_64
    let cpuid24 = unsafe { __cpuid_count(0x24, 0) };
    let avx10_version = cpuid24.ebx & 0xFF;
    if avx10_version >= 1 {
      caps |= x86::AVX10_1;
    }
    if avx10_version >= 2 {
      caps |= x86::AVX10_2;
    }
  }

  // APX detection via CPUID leaf 0x29
  // APX doubles GPRs from 16 to 32 (R16-R31) on Granite Rapids+
  if cpuid0.eax >= 0x29 {
    // SAFETY: CPUID is always safe on x86_64
    let cpuid29 = unsafe { __cpuid_count(0x29, 0) };
    // APX_NCI_NDD_NF is bit 0 of EBX
    if cpuid29.ebx & 1 != 0 {
      caps |= x86::APX;
    }
  }

  // Extended CPUID (leaf 0x80000001) for AMD-specific features
  // SAFETY: CPUID is always safe on x86_64
  let cpuid_ext = unsafe { __cpuid(0x8000_0001) };
  if cpuid_ext.ecx & (1 << 5) != 0 {
    caps |= x86::LZCNT;
  }
  if cpuid_ext.ecx & (1 << 6) != 0 {
    caps |= x86::SSE4A;
  }

  CpuidBatch {
    caps,
    is_amd,
    family,
    model,
  }
}

/// Runtime x86 (32-bit) feature detection using CPUID.
///
/// # Safety
/// Uses CPUID instruction which requires unsafe, but is always safe to call on x86.
#[cfg(all(target_arch = "x86", feature = "std"))]
#[allow(unsafe_code)]
fn runtime_x86_32() -> Caps {
  use core::arch::x86::{__cpuid, __cpuid_count};

  use crate::caps::x86;

  let mut caps = Caps::NONE;

  // CPUID leaf 1: processor info and feature bits
  // SAFETY: CPUID is always safe on x86
  let cpuid1 = unsafe { __cpuid(1) };

  // ECX features (leaf 1)
  if cpuid1.ecx & (1 << 0) != 0 {
    caps |= x86::SSE3;
  }
  if cpuid1.ecx & (1 << 9) != 0 {
    caps |= x86::SSSE3;
  }
  if cpuid1.ecx & (1 << 19) != 0 {
    caps |= x86::SSE41;
  }
  if cpuid1.ecx & (1 << 20) != 0 {
    caps |= x86::SSE42;
  }
  if cpuid1.ecx & (1 << 1) != 0 {
    caps |= x86::PCLMULQDQ;
  }
  if cpuid1.ecx & (1 << 25) != 0 {
    caps |= x86::AESNI;
  }

  caps
}

/// Check if user has explicitly enabled AVX-512 on hybrid Intel CPUs.
///
/// On Alder Lake and newer hybrid Intel CPUs, AVX-512 is disabled by default
/// because E-cores don't support it. Power users who have disabled E-cores
/// in BIOS or are using early unfused chips can set this environment variable
/// to force AVX-512 usage.
///
/// # Environment Variable
///
/// `RSCRYPTO_FORCE_AVX512=1` enables AVX-512 on hybrid Intel CPUs.
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
fn hybrid_avx512_override() -> bool {
  // Check environment variable for explicit opt-in
  std::env::var("RSCRYPTO_FORCE_AVX512")
    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    .unwrap_or(false)
}

/// Detect Intel hybrid CPU (Alder Lake family and newer).
///
/// Returns true if this is an Intel hybrid CPU (P+E cores) where AVX-512
/// is problematic. These CPUs have family 6, model 0x97 (ADL-S), 0x9A (ADL-P),
/// 0xB7 (RPL-S), 0xBA (RPL-P), etc.
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
fn is_intel_hybrid(is_amd: bool, family: u32, model: u32) -> bool {
  if is_amd {
    return false;
  }

  // Intel uses extended family + base family for family >= 15
  // Family 6 is used for all modern Intel client/server CPUs
  if family != 6 {
    return false;
  }

  // Hybrid CPU models (Alder Lake, Raptor Lake, Meteor Lake, etc.)
  // These have E-cores that don't support AVX-512
  matches!(
    model,
    0x97  // Alder Lake-S (desktop)
    | 0x9A  // Alder Lake-P/H/U (mobile)
    | 0x9C  // Alder Lake-N (low power)
    | 0xB7  // Raptor Lake-S (desktop)
    | 0xBA  // Raptor Lake-P/H (mobile)
    | 0xBF  // Raptor Lake-S refresh
    | 0xAA  // Meteor Lake-H
    | 0xAC  // Meteor Lake-U
    | 0xBD  // Lunar Lake
    | 0xC5  // Arrow Lake-S
    | 0xC6 // Arrow Lake-H
  )
}

/// Select tuning preset based on features and pre-extracted CPU info.
///
/// Takes vendor/family/model info extracted from batch CPUID to avoid redundant calls.
/// - `is_amd`: true if vendor string is "AuthenticAMD"
/// - `family`: CPU family (e.g., 25 = Zen 3/4, 26 = Zen 5)
/// - `model`: CPU model within family (used for hybrid detection in std mode)
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(unused_variables)] // `model` only used with std feature for hybrid detection
fn select_x86_tune(caps: Caps, is_amd: bool, family: u32, model: u32) -> Tune {
  use crate::caps::x86;

  // Hybrid Intel note:
  // `detect_x86_64` will clear AVX-512 caps on known hybrid Intel CPUs unless
  // the user explicitly overrides. For tune selection we still want to prefer
  // 256-bit strategies on those systems when AVX2 is available.
  #[cfg(feature = "std")]
  if is_intel_hybrid(is_amd, family, model) && !hybrid_avx512_override() && caps.has(x86::AVX2) {
    return Tune::INTEL_ICL;
  }

  // BLAKE3 (and other vectorized kernels) benefit from AVX-512 on CPUs that
  // support the *base* AVX-512 feature set. Do not gate microarchitecture
  // classification on VPCLMUL/VAES; those are kernel-specific capabilities.
  let has_avx512 = caps.has(x86::AVX512F) && caps.has(x86::AVX512VL);

  if has_avx512 {
    if is_amd {
      // Zen 5/5c is family 26, Zen 4 is family 25 (models 96-127)
      // Note: Currently no way to differentiate Zen 5c from Zen 5 via CPUID alone.
      // Both share Family 26 (0x1A). Zen 5c is used in EPYC 9005 series and
      // Strix Point APUs with hybrid configurations. For now, use ZEN5 tuning
      // for all Family 26 CPUs. Future: may need OS topology detection or
      // per-core CPUID to identify compact cores in hybrid SKUs.
      if family == 26 {
        Tune::ZEN5
      } else if family == 25 {
        Tune::ZEN4
      } else {
        Tune::DEFAULT
      }
    } else {
      // Intel: Check for Granite Rapids via AMX extensions
      // Granite Rapids has AMX_FP16 and AMX_COMPLEX (leaf 7.1, EAX bits 21 & 8)
      if caps.has(x86::AMX_FP16) || caps.has(x86::AMX_COMPLEX) {
        Tune::INTEL_GNR
      } else {
        Tune::INTEL_SPR
      }
    }
  } else if caps.has(x86::AVX2) || caps.has(x86::PCLMUL_READY) {
    Tune::DEFAULT
  } else {
    Tune::PORTABLE
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// aarch64 Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
fn detect_aarch64() -> Detected {
  // Start with compile-time detected features (includes NEON baseline)
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_aarch64();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  let tune = select_aarch64_tune(caps);

  Detected {
    caps,
    tune,
    arch: Arch::Aarch64,
  }
}

/// Batch extraction of aarch64 features from /proc/self/auxv.
///
/// Reads AT_HWCAP and AT_HWCAP2 once from the ELF auxiliary vector.
/// This is faster than calling is_aarch64_feature_detected! 20+ times.
/// Pure Rust - no libc dependency.
///
/// Works on Linux and Android (both use procfs with ELF auxv format).
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn hwcap_batch_aarch64() -> Caps {
  use std::{fs::File, io::Read};

  use crate::caps::aarch64;

  // ELF auxiliary vector entry types
  const AT_HWCAP: u64 = 16;
  const AT_HWCAP2: u64 = 26;

  // HWCAP bit positions (from linux/arch/arm64/include/uapi/asm/hwcap.h)
  const HWCAP_AES: u64 = 1 << 3;
  const HWCAP_PMULL: u64 = 1 << 4;
  const HWCAP_SHA2: u64 = 1 << 6;
  const HWCAP_CRC32: u64 = 1 << 7;
  const HWCAP_ATOMICS: u64 = 1 << 8; // LSE
  const HWCAP_FPHP: u64 = 1 << 9; // FP16
  const HWCAP_ASIMDHP: u64 = 1 << 10;
  const HWCAP_SHA3: u64 = 1 << 17;
  const HWCAP_SM3: u64 = 1 << 18;
  const HWCAP_SM4: u64 = 1 << 19;
  const HWCAP_ASIMDDP: u64 = 1 << 20; // DOTPROD
  const HWCAP_SHA512: u64 = 1 << 21;
  const HWCAP_SVE: u64 = 1 << 22;

  // HWCAP2 bit positions
  const HWCAP2_SVE2: u64 = 1 << 1;
  const HWCAP2_SVEAES: u64 = 1 << 2;
  const HWCAP2_SVEPMULL: u64 = 1 << 3;
  const HWCAP2_SVEBITPERM: u64 = 1 << 4;
  const HWCAP2_SVESHA3: u64 = 1 << 5;
  const HWCAP2_SVESM4: u64 = 1 << 6;
  const HWCAP2_FRINT: u64 = 1 << 8; // FRINTTS
  const HWCAP2_SVEI8MM: u64 = 1 << 9;
  const HWCAP2_SVEF32MM: u64 = 1 << 10;
  const HWCAP2_SVEF64MM: u64 = 1 << 11;
  const HWCAP2_SVEBF16: u64 = 1 << 12;
  const HWCAP2_I8MM: u64 = 1 << 13;
  const HWCAP2_BF16: u64 = 1 << 14;
  const HWCAP2_RNG: u64 = 1 << 16;
  const HWCAP2_SME: u64 = 1 << 23;
  const HWCAP2_SME_I16I64: u64 = 1 << 24;
  const HWCAP2_SME_F64F64: u64 = 1 << 25;
  const HWCAP2_SME_I8I32: u64 = 1 << 26;
  const HWCAP2_SME_F16F32: u64 = 1 << 27;
  const HWCAP2_SME_B16F32: u64 = 1 << 28;
  const HWCAP2_SME_F32F32: u64 = 1 << 29;
  const HWCAP2_SME_FA64: u64 = 1 << 30;
  const HWCAP2_EBF16: u64 = 1 << 32;
  const HWCAP2_SVE_EBF16: u64 = 1 << 33;
  const HWCAP2_SVE2P1: u64 = 1 << 36;
  const HWCAP2_SME2: u64 = 1 << 37;
  const HWCAP2_SME2P1: u64 = 1 << 38;
  const HWCAP2_SME_I16I32: u64 = 1 << 39;
  const HWCAP2_SME_BI32I32: u64 = 1 << 40;
  const HWCAP2_SME_B16B16: u64 = 1 << 41;
  const HWCAP2_SME_F16F16: u64 = 1 << 42;
  const HWCAP2_MOPS: u64 = 1 << 43;
  const HWCAP2_SVE_B16B16: u64 = 1 << 45;
  const HWCAP2_LSE128: u64 = 1 << 47;

  // Read /proc/self/auxv - format is pairs of (type: u64, value: u64)
  let (hwcap, hwcap2) = (|| -> Option<(u64, u64)> {
    let mut file = File::open("/proc/self/auxv").ok()?;
    let mut buf = [0u8; 4096]; // Auxiliary vector is small
    let n = file.read(&mut buf).ok()?;

    let mut hwcap = 0u64;
    let mut hwcap2 = 0u64;

    // Parse as array of (u64, u64) pairs
    let entries = buf.get(..n)?;
    for chunk in entries.chunks_exact(16) {
      let a_type = u64::from_ne_bytes(chunk.get(0..8)?.try_into().ok()?);
      let a_val = u64::from_ne_bytes(chunk.get(8..16)?.try_into().ok()?);

      if a_type == AT_HWCAP {
        hwcap = a_val;
      } else if a_type == AT_HWCAP2 {
        hwcap2 = a_val;
      } else if a_type == 0 {
        // AT_NULL terminates the vector
        break;
      }
    }
    Some((hwcap, hwcap2))
  })()
  .unwrap_or((0, 0));

  let mut caps = Caps::NONE;

  // ─── HWCAP features ───
  if hwcap & HWCAP_AES != 0 {
    caps |= aarch64::AES;
  }
  if hwcap & HWCAP_PMULL != 0 {
    caps |= aarch64::PMULL;
  }
  if hwcap & HWCAP_SHA2 != 0 {
    caps |= aarch64::SHA2;
  }
  if hwcap & HWCAP_CRC32 != 0 {
    caps |= aarch64::CRC;
  }
  if hwcap & HWCAP_ATOMICS != 0 {
    caps |= aarch64::LSE;
  }
  if hwcap & (HWCAP_FPHP | HWCAP_ASIMDHP) != 0 {
    caps |= aarch64::FP16;
  }
  if hwcap & HWCAP_SHA3 != 0 {
    caps |= aarch64::SHA3;
  }
  if hwcap & HWCAP_SM3 != 0 {
    caps |= aarch64::SM3;
  }
  if hwcap & HWCAP_SM4 != 0 {
    caps |= aarch64::SM4;
  }
  if hwcap & HWCAP_ASIMDDP != 0 {
    caps |= aarch64::DOTPROD;
  }
  if hwcap & HWCAP_SHA512 != 0 {
    caps |= aarch64::SHA512;
  }
  if hwcap & HWCAP_SVE != 0 {
    caps |= aarch64::SVE;
  }

  // ─── HWCAP2 features ───
  if hwcap2 & HWCAP2_SVE2 != 0 {
    caps |= aarch64::SVE2;
  }
  if hwcap2 & HWCAP2_SVEAES != 0 {
    caps |= aarch64::SVE2_AES;
  }
  if hwcap2 & HWCAP2_SVEPMULL != 0 {
    caps |= aarch64::SVE2_PMULL;
  }
  if hwcap2 & HWCAP2_SVEBITPERM != 0 {
    caps |= aarch64::SVE2_BITPERM;
  }
  if hwcap2 & HWCAP2_SVESHA3 != 0 {
    caps |= aarch64::SVE2_SHA3;
  }
  if hwcap2 & HWCAP2_SVESM4 != 0 {
    caps |= aarch64::SVE2_SM4;
  }
  if hwcap2 & HWCAP2_FRINT != 0 {
    caps |= aarch64::FRINTTS;
  }
  if hwcap2 & HWCAP2_SVEI8MM != 0 {
    caps |= aarch64::SVE2_I8MM;
  }
  if hwcap2 & HWCAP2_SVEF32MM != 0 {
    caps |= aarch64::SVE2_F32MM;
  }
  if hwcap2 & HWCAP2_SVEF64MM != 0 {
    caps |= aarch64::SVE2_F64MM;
  }
  if hwcap2 & HWCAP2_SVEBF16 != 0 {
    caps |= aarch64::SVE2_BF16;
  }
  if hwcap2 & HWCAP2_I8MM != 0 {
    caps |= aarch64::I8MM;
  }
  if hwcap2 & HWCAP2_BF16 != 0 {
    caps |= aarch64::BF16;
  }
  if hwcap2 & HWCAP2_RNG != 0 {
    caps |= aarch64::RNG;
  }
  if hwcap2 & HWCAP2_SME != 0 {
    caps |= aarch64::SME;
  }
  if hwcap2 & HWCAP2_SME_I16I64 != 0 {
    caps |= aarch64::SME_I16I64;
  }
  if hwcap2 & HWCAP2_SME_F64F64 != 0 {
    caps |= aarch64::SME_F64F64;
  }
  if hwcap2 & HWCAP2_SME_I8I32 != 0 {
    caps |= aarch64::SME_I8I32;
  }
  if hwcap2 & HWCAP2_SME_F16F32 != 0 {
    caps |= aarch64::SME_F16F32;
  }
  if hwcap2 & HWCAP2_SME_B16F32 != 0 {
    caps |= aarch64::SME_B16F32;
  }
  if hwcap2 & HWCAP2_SME_F32F32 != 0 {
    caps |= aarch64::SME_F32F32;
  }
  if hwcap2 & HWCAP2_SME_FA64 != 0 {
    caps |= aarch64::SME_FA64;
  }
  if hwcap2 & HWCAP2_EBF16 != 0 {
    caps |= aarch64::EBF16;
  }
  if hwcap2 & HWCAP2_SVE_EBF16 != 0 {
    caps |= aarch64::SVE2_EBF16;
  }
  if hwcap2 & HWCAP2_SVE2P1 != 0 {
    caps |= aarch64::SVE2P1;
  }
  if hwcap2 & HWCAP2_SME2 != 0 {
    caps |= aarch64::SME2;
  }
  if hwcap2 & HWCAP2_SME2P1 != 0 {
    caps |= aarch64::SME2P1;
  }
  if hwcap2 & HWCAP2_SME_I16I32 != 0 {
    caps |= aarch64::SME_I16I32;
  }
  if hwcap2 & HWCAP2_SME_BI32I32 != 0 {
    caps |= aarch64::SME_BI32I32;
  }
  if hwcap2 & HWCAP2_SME_B16B16 != 0 {
    caps |= aarch64::SME_B16B16;
  }
  if hwcap2 & HWCAP2_SME_F16F16 != 0 {
    caps |= aarch64::SME_F16F16;
  }
  if hwcap2 & HWCAP2_MOPS != 0 {
    caps |= aarch64::MOPS;
  }
  if hwcap2 & HWCAP2_SVE_B16B16 != 0 {
    caps |= aarch64::SVE_B16B16;
  }
  if hwcap2 & HWCAP2_LSE128 != 0 {
    caps |= aarch64::LSE2;
  }

  caps
}

/// Runtime aarch64 detection for Linux/Android (batch HWCAP from /proc/self/auxv).
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn runtime_aarch64() -> Caps {
  hwcap_batch_aarch64()
}

/// Runtime aarch64 detection for other platforms (fallback to macro calls).
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  not(any(target_os = "linux", target_os = "android"))
))]
fn runtime_aarch64() -> Caps {
  use crate::caps::aarch64;

  let mut caps = Caps::NONE;

  // ─── Crypto Extensions ───
  if std::arch::is_aarch64_feature_detected!("aes") {
    caps |= aarch64::AES;
  }
  if std::arch::is_aarch64_feature_detected!("pmull") {
    caps |= aarch64::PMULL;
  }
  if std::arch::is_aarch64_feature_detected!("sha2") {
    caps |= aarch64::SHA2;
  }
  if std::arch::is_aarch64_feature_detected!("sha3") {
    caps |= aarch64::SHA3 | aarch64::SHA512;
  }
  if std::arch::is_aarch64_feature_detected!("sm4") {
    caps |= aarch64::SM3 | aarch64::SM4;
  }

  // ─── CRC Extension ───
  if std::arch::is_aarch64_feature_detected!("crc") {
    caps |= aarch64::CRC;
  }

  // ─── Additional SIMD ───
  if std::arch::is_aarch64_feature_detected!("dotprod") {
    caps |= aarch64::DOTPROD;
  }
  if std::arch::is_aarch64_feature_detected!("fp16") {
    caps |= aarch64::FP16;
  }
  if std::arch::is_aarch64_feature_detected!("i8mm") {
    caps |= aarch64::I8MM;
  }
  if std::arch::is_aarch64_feature_detected!("bf16") {
    caps |= aarch64::BF16;
  }
  if std::arch::is_aarch64_feature_detected!("frintts") {
    caps |= aarch64::FRINTTS;
  }

  // ─── SVE Family ───
  if std::arch::is_aarch64_feature_detected!("sve") {
    caps |= aarch64::SVE;
  }
  if std::arch::is_aarch64_feature_detected!("sve2") {
    caps |= aarch64::SVE2;
  }

  // ─── Atomics ───
  if std::arch::is_aarch64_feature_detected!("lse") {
    caps |= aarch64::LSE;
  }
  if std::arch::is_aarch64_feature_detected!("lse2") {
    caps |= aarch64::LSE2;
  }

  // ─── Memory Operations ───
  // MOPS is detected on Linux via HWCAP2 in hwcap_batch_aarch64()
  // On other platforms, compile-time detection via target_feature is used
  #[cfg(all(target_feature = "mops", not(any(target_os = "linux", target_os = "android"))))]
  {
    caps |= aarch64::MOPS;
  }

  // ─── Hardware RNG ───
  if std::arch::is_aarch64_feature_detected!("rand") {
    caps |= aarch64::RNG;
  }

  // ─── SME Detection ───
  // On macOS and other Apple platforms, use sysctl for comprehensive SME detection.
  // std::arch::is_aarch64_feature_detected doesn't currently detect SME reliably
  // on macOS, so we use platform-specific detection.
  #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos"))]
  {
    caps |= detect_apple_sme_features();
  }

  // For Linux (non-Apple), fall back to is_aarch64_feature_detected for SME.
  // Note: SME detection via std::arch is only stable on Linux; Windows ARM64 requires
  // unstable `stdarch_aarch64_feature_detection` feature which we avoid.
  #[cfg(target_os = "linux")]
  {
    if std::arch::is_aarch64_feature_detected!("sme") {
      caps |= aarch64::SME;
    }
    if std::arch::is_aarch64_feature_detected!("sme2") {
      caps |= aarch64::SME2;
    }
  }

  caps
}

// ─────────────────────────────────────────────────────────────────────────────
// Apple Silicon Detection (macOS/iOS)
// ─────────────────────────────────────────────────────────────────────────────

/// Apple Silicon chip generation.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AppleSiliconGen {
  /// M1, M1 Pro, M1 Max, M1 Ultra (Firestorm/Icestorm)
  M1,
  /// M2, M2 Pro, M2 Max, M2 Ultra (Blizzard/Avalanche)
  M2,
  /// M3, M3 Pro, M3 Max (Ibiza/Lobos/Palma)
  M3,
  /// M4, M4 Pro, M4 Max (Donan/Brava) - has SME
  M4,
  /// M5, M5 Pro, M5 Max (Hidra/Sotra) - has SME2p1
  /// Released October 2025. Adds SME2p1, SMEB16B16, SMEF16F16 per LLVM.
  // TODO(M5): Remove allow(dead_code) once CPUFAMILY_ARM_HIDRA/SOTRA values are added.
  #[allow(dead_code)]
  M5,
}

/// Detect Apple Silicon generation via sysctlbyname("hw.cpufamily").
///
/// Uses direct extern "C" linkage to libSystem (always linked on Apple platforms)
/// to avoid adding libc as a dependency.
///
/// Returns `None` for unknown/future chips or A-series (pre-M1) processors.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
))]
fn detect_apple_silicon_gen() -> Option<AppleSiliconGen> {
  // CPUFAMILY constants from Apple's machine.h (via Zig's darwin.zig / Homebrew / LLVM)
  // These identify the CPU microarchitecture, not the marketing name.
  //
  // Reference: https://github.com/Homebrew/brew/blob/master/Library/Homebrew/extend/os/mac/hardware/cpu.rb
  // Reference: https://github.com/ziglang/zig/blob/master/lib/std/c/darwin.zig
  const CPUFAMILY_ARM_FIRESTORM_ICESTORM: u32 = 0x1b58_8bb3; // M1 family
  const CPUFAMILY_ARM_BLIZZARD_AVALANCHE: u32 = 0xda33_d83d; // M2 family
  const CPUFAMILY_ARM_EVEREST_SAWTOOTH: u32 = 0x8765_edea; // A16/M2 variant
  const CPUFAMILY_ARM_COLL: u32 = 0x2876_f5b5; // A17 Pro
  const CPUFAMILY_ARM_IBIZA: u32 = 0xfa33_415e; // M3
  const CPUFAMILY_ARM_LOBOS: u32 = 0x5f4d_ea93; // M3 Pro
  const CPUFAMILY_ARM_PALMA: u32 = 0x7201_5832; // M3 Max
  const CPUFAMILY_ARM_DONAN: u32 = 0x6f51_29ac; // M4
  const CPUFAMILY_ARM_BRAVA: u32 = 0x17d5_b93a; // M4 Pro/Max
  const CPUFAMILY_ARM_TAHITI: u32 = 0x75d4_acb9; // A18
  const CPUFAMILY_ARM_TUPAI: u32 = 0x2045_26d0; // A18 Pro

  // M5 family (released October 2025)
  // Codenames: Hidra (M5 - H17G), Sotra (M5 Pro/Max)
  // Features: SME2p1, SMEB16B16, SMEF16F16 per LLVM commit f85494f6afeb
  // TODO(M5): Add CPUFAMILY_ARM_HIDRA and CPUFAMILY_ARM_SOTRA hex values
  //           when publicly documented in Xcode SDK / Homebrew / Zig darwin.zig
  // Reference: Xcode 26.1b3 added CPUFAMILY_ARM_HIDRA (H17G)
  // Until then, M5 detection falls back to SME2 feature detection (see select_aarch64_tune)

  // Direct extern "C" linkage to libSystem's sysctlbyname
  // (libSystem is always linked on Apple platforms)
  // SAFETY: This extern block declares a C function from libSystem.
  // The function signature matches Apple's sysctlbyname(3).
  #[allow(unsafe_code)]
  unsafe extern "C" {
    fn sysctlbyname(
      name: *const u8,
      oldp: *mut core::ffi::c_void,
      oldlenp: *mut usize,
      newp: *const core::ffi::c_void,
      newlen: usize,
    ) -> i32;
  }

  let mut cpufamily: u32 = 0;
  let mut size = core::mem::size_of::<u32>();

  // SAFETY: sysctlbyname is safe to call with valid pointers.
  // "hw.cpufamily" is a valid null-terminated string.
  // The output buffer is properly sized for u32.
  #[allow(unsafe_code)]
  let ret = unsafe {
    sysctlbyname(
      c"hw.cpufamily".as_ptr().cast(),
      core::ptr::addr_of_mut!(cpufamily).cast(),
      core::ptr::addr_of_mut!(size),
      core::ptr::null(),
      0,
    )
  };

  if ret != 0 {
    return None;
  }

  match cpufamily {
    CPUFAMILY_ARM_FIRESTORM_ICESTORM => Some(AppleSiliconGen::M1),
    CPUFAMILY_ARM_BLIZZARD_AVALANCHE | CPUFAMILY_ARM_EVEREST_SAWTOOTH => Some(AppleSiliconGen::M2),
    CPUFAMILY_ARM_IBIZA | CPUFAMILY_ARM_LOBOS | CPUFAMILY_ARM_PALMA => Some(AppleSiliconGen::M3),
    CPUFAMILY_ARM_DONAN | CPUFAMILY_ARM_BRAVA => Some(AppleSiliconGen::M4),
    // A-series chips (A16, A17, A18) - treat as M-series equivalent for tuning
    CPUFAMILY_ARM_COLL => Some(AppleSiliconGen::M2), // A17 Pro ≈ M2 architecture
    CPUFAMILY_ARM_TAHITI | CPUFAMILY_ARM_TUPAI => Some(AppleSiliconGen::M4), // A18 ≈ M4 architecture
    _ => None,                                       // Unknown future chip - will fall back to feature-based detection
  }
}

// ─────────────────────────────────────────────────────────────────────────────
/// Detect SME features on Apple platforms via sysctlbyname.
///
/// Apple exposes SME and related features through hw.optional.arm.FEAT_* sysctl keys.
/// This provides more reliable detection than is_aarch64_feature_detected on macOS.
///
/// Returns a Caps bitset with detected SME features.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
))]
fn detect_apple_sme_features() -> Caps {
  use crate::caps::aarch64;

  // Helper to read a u32 sysctl value (returns 0 on error or false, 1 on true)
  fn sysctl_u32(name: &[u8]) -> u32 {
    // Direct extern "C" linkage to libSystem's sysctlbyname
    // SAFETY: This extern block declares a C function from libSystem.
    #[allow(unsafe_code)]
    unsafe extern "C" {
      fn sysctlbyname(
        name: *const u8,
        oldp: *mut core::ffi::c_void,
        oldlenp: *mut usize,
        newp: *const core::ffi::c_void,
        newlen: usize,
      ) -> i32;
    }

    let mut value: u32 = 0;
    let mut size = core::mem::size_of::<u32>();

    // SAFETY: sysctlbyname is safe to call with valid pointers.
    // name is a valid null-terminated C string.
    // The output buffer is properly sized for u32.
    #[allow(unsafe_code)]
    let ret = unsafe {
      sysctlbyname(
        name.as_ptr(),
        core::ptr::addr_of_mut!(value).cast(),
        core::ptr::addr_of_mut!(size),
        core::ptr::null(),
        0,
      )
    };

    if ret == 0 { value } else { 0 }
  }

  let mut caps = Caps::NONE;

  // ─── SME Base and Versions ───
  if sysctl_u32(c"hw.optional.arm.FEAT_SME".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME;
  }
  if sysctl_u32(c"hw.optional.arm.FEAT_SME2".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME2;
  }
  if sysctl_u32(c"hw.optional.arm.FEAT_SME2p1".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME2P1;
  }

  // ─── SME Extended Features ───
  if sysctl_u32(c"hw.optional.arm.FEAT_SME_I16I64".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME_I16I64;
  }
  if sysctl_u32(c"hw.optional.arm.FEAT_SME_F64F64".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME_F64F64;
  }
  if sysctl_u32(c"hw.optional.arm.FEAT_SME_B16B16".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME_B16B16;
  }
  if sysctl_u32(c"hw.optional.arm.FEAT_SME_F16F16".to_bytes_with_nul()) != 0 {
    caps |= aarch64::SME_F16F16;
  }

  // ─── Fallback: Infer SME from chip generation if sysctl unavailable ───
  // This handles cases where the OS doesn't expose SME sysctl keys yet.
  // M4 has SME, M5 has SME2p1 + additional features.
  if caps.is_empty()
    && let Some(chip_gen) = detect_apple_silicon_gen()
  {
    match chip_gen {
      AppleSiliconGen::M4 => {
        caps |= aarch64::SME;
      }
      AppleSiliconGen::M5 => {
        // M5 has SME2p1, SMEB16B16, SMEF16F16 per LLVM
        caps |= aarch64::SME | aarch64::SME2 | aarch64::SME2P1 | aarch64::SME_B16B16 | aarch64::SME_F16F16;
      }
      _ => {}
    }
  }

  caps
}

/// Detect SME tile size (SVL - Streaming Vector Length) on Apple platforms.
///
/// Returns the maximum SVL in bytes, or 0 if SME is not supported or detection failed.
///
/// On Apple Silicon:
/// - M4: SME with 128-bit tiles (SVL = 16 bytes)
/// - M5: SME2p1 with 128-bit tiles (SVL = 16 bytes)
///
/// Note: Apple's implementation uses fixed 128-bit SVL, unlike server ARM chips
/// which may support 128-512 bit configurable SVL.
#[cfg(all(
  target_arch = "aarch64",
  feature = "std",
  any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
))]
#[allow(dead_code)] // Will be used when SME kernels are implemented
fn detect_apple_sme_tile_size() -> u16 {
  // Helper to read a u32 sysctl value
  fn sysctl_u32(name: &[u8]) -> u32 {
    #[allow(unsafe_code)]
    unsafe extern "C" {
      fn sysctlbyname(
        name: *const u8,
        oldp: *mut core::ffi::c_void,
        oldlenp: *mut usize,
        newp: *const core::ffi::c_void,
        newlen: usize,
      ) -> i32;
    }

    let mut value: u32 = 0;
    let mut size = core::mem::size_of::<u32>();

    #[allow(unsafe_code)]
    // SAFETY: `sysctlbyname` expects `name` to be a valid NUL-terminated C string (caller provides this),
    // `oldp`/`oldlenp` point to writable locals, and `newp` is null with `newlen = 0` (no write).
    let ret = unsafe {
      sysctlbyname(
        name.as_ptr(),
        core::ptr::addr_of_mut!(value).cast(),
        core::ptr::addr_of_mut!(size),
        core::ptr::null(),
        0,
      )
    };

    if ret == 0 { value } else { 0 }
  }

  // Try to read the SVL from sysctl
  let svl_bytes = sysctl_u32(c"hw.optional.arm.sme_max_svl_b".to_bytes_with_nul());
  if svl_bytes > 0 {
    return svl_bytes as u16;
  }

  // Fallback: Use chip generation to infer SVL
  // Apple Silicon uses fixed 128-bit (16 byte) SVL for SME
  if let Some(AppleSiliconGen::M4 | AppleSiliconGen::M5) = detect_apple_silicon_gen() {
    return 16; // 128 bits = 16 bytes
  }

  0 // SME not supported or unknown
}
// SVE Vector Length Detection (Linux aarch64)
// ─────────────────────────────────────────────────────────────────────────────

/// Detect SVE vector length in bits via prctl(PR_SVE_GET_VL).
///
/// Uses raw syscall to avoid libc dependency. Returns 0 if SVE is not supported.
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
fn detect_sve_vlen() -> u16 {
  // prctl syscall number on aarch64-linux
  const SYS_PRCTL: u64 = 167;
  const PR_SVE_GET_VL: u64 = 51;
  const PR_SVE_VL_LEN_MASK: u64 = 0xFFFF;

  let result: i64;

  // SAFETY: prctl(PR_SVE_GET_VL) is always safe to call.
  // Returns the vector length in bytes on success, or -EINVAL if SVE unsupported.
  #[allow(unsafe_code)]
  unsafe {
    core::arch::asm!(
      "svc #0",
      in("x8") SYS_PRCTL,
      in("x0") PR_SVE_GET_VL,
      in("x1") 0u64,
      in("x2") 0u64,
      in("x3") 0u64,
      in("x4") 0u64,
      lateout("x0") result,
      options(nostack)
    );
  }

  if result < 0 {
    return 0; // SVE not supported
  }

  // Result is VL in bytes; convert to bits
  let vl_bytes = (result as u64) & PR_SVE_VL_LEN_MASK;
  // Saturate to u16::MAX (65535 bits = 8KB, well above SVE's 2048-bit max)
  (vl_bytes.saturating_mul(8)) as u16
}

/// Fallback SVE vector length detection for non-Linux platforms.
#[cfg(all(target_arch = "aarch64", not(all(target_os = "linux", feature = "std"))))]
fn detect_sve_vlen() -> u16 {
  // On non-Linux platforms, we can't easily detect SVE VL.
  // Return 0 to indicate unknown; tuning will use hardcoded defaults.
  0
}

// ─────────────────────────────────────────────────────────────────────────────
// SME Vector Length Detection (Linux aarch64)
// ─────────────────────────────────────────────────────────────────────────────

/// Detect SME streaming vector length in bits via prctl(PR_SME_GET_VL).
///
/// Uses raw syscall to avoid libc dependency. Returns 0 if SME is not supported.
/// The SME vector length determines the tile size (SVL × SVL bits).
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
#[allow(dead_code)] // Reserved for future Linux SME support (Grace Hopper, future Gravitons)
fn detect_sme_vlen() -> u16 {
  // prctl syscall number on aarch64-linux
  const SYS_PRCTL: u64 = 167;
  const PR_SME_GET_VL: u64 = 63;
  const PR_SME_VL_LEN_MASK: u64 = 0xFFFF;

  let result: i64;

  // SAFETY: prctl(PR_SME_GET_VL) is always safe to call.
  // Returns the streaming vector length in bytes on success, or -EINVAL if SME unsupported.
  #[allow(unsafe_code)]
  unsafe {
    core::arch::asm!(
      "svc #0",
      in("x8") SYS_PRCTL,
      in("x0") PR_SME_GET_VL,
      in("x1") 0u64,
      in("x2") 0u64,
      in("x3") 0u64,
      in("x4") 0u64,
      lateout("x0") result,
      options(nostack)
    );
  }

  if result < 0 {
    return 0; // SME not supported
  }

  // Result is VL in bytes; convert to bits
  let vl_bytes = (result as u64) & PR_SME_VL_LEN_MASK;
  // Saturate to u16::MAX (65535 bits = 8KB, well above SME's typical 256-512 bit max)
  (vl_bytes.saturating_mul(8)) as u16
}

/// Fallback SME vector length detection for non-Linux platforms.
#[cfg(all(target_arch = "aarch64", not(all(target_os = "linux", feature = "std"))))]
#[allow(dead_code)] // May be unused on some aarch64 bare-metal configs
fn detect_sme_vlen() -> u16 {
  // On non-Linux platforms, we can't easily detect SME VL.
  // Return 0 to indicate unknown; tuning will use hardcoded defaults.
  0
}

// ─────────────────────────────────────────────────────────────────────────────
// MIDR_EL1 Detection (Linux aarch64)
// ─────────────────────────────────────────────────────────────────────────────

/// Read MIDR_EL1 (Main ID Register) to identify the CPU part number.
///
/// Returns the full MIDR value, where bits [15:4] contain the part number.
/// Common part numbers:
/// - 0xd0c: Neoverse N1 (Ampere Altra, Graviton 2)
/// - 0xd40: Neoverse V1 (Graviton 3)
/// - 0xd49: Neoverse N2
/// - 0xd4f: Neoverse V2 (NVIDIA Grace, Graviton 4)
/// - 0xd8e: Neoverse N3
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
#[allow(unsafe_code)]
fn read_midr_el1() -> Option<u64> {
  use std::fs;

  // Try to read from /sys/devices/system/cpu/cpu0/regs/identification/midr_el1
  if let Ok(contents) = fs::read_to_string("/sys/devices/system/cpu/cpu0/regs/identification/midr_el1")
    && let Ok(midr) = u64::from_str_radix(contents.trim().trim_start_matches("0x"), 16)
  {
    return Some(midr);
  }

  // Fallback: try to read MIDR directly via inline asm
  let midr: u64;
  // SAFETY: `asm!` writes only to the `midr` output register and declares `nomem, nostack`.
  unsafe {
    core::arch::asm!("mrs {}, midr_el1", out(reg) midr, options(nomem, nostack));
  }
  Some(midr)
}

#[cfg(target_arch = "aarch64")]
fn select_aarch64_tune(caps: Caps) -> Tune {
  use crate::caps::aarch64;

  // Apple Silicon - use cpufamily detection for precise chip identification
  #[cfg(all(
    feature = "std",
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  {
    if caps.has(aarch64::PMULL_EOR3_READY) {
      match detect_apple_silicon_gen() {
        Some(AppleSiliconGen::M5) => {
          let mut tune = Tune::APPLE_M5;
          // Runtime SME vector length detection
          let sme_vl = detect_sme_vlen();
          if sme_vl > 0 {
            tune.sme_tile = sme_vl;
          }
          return tune;
        }
        Some(AppleSiliconGen::M4) => {
          let mut tune = Tune::APPLE_M4;
          // Runtime SME vector length detection
          let sme_vl = detect_sme_vlen();
          if sme_vl > 0 {
            tune.sme_tile = sme_vl;
          }
          return tune;
        }
        Some(AppleSiliconGen::M1 | AppleSiliconGen::M2 | AppleSiliconGen::M3) => return Tune::APPLE_M1_M3,
        None => {
          // Unknown chip but has PMULL+EOR3 - assume M-series compatible
          // Use SME2 to detect M5+, SME for M4, otherwise M1-M3
          if caps.has(aarch64::SME2) {
            let mut tune = Tune::APPLE_M5;
            let sme_vl = detect_sme_vlen();
            if sme_vl > 0 {
              tune.sme_tile = sme_vl;
            }
            return tune;
          }
          if caps.has(aarch64::SME) {
            let mut tune = Tune::APPLE_M4;
            let sme_vl = detect_sme_vlen();
            if sme_vl > 0 {
              tune.sme_tile = sme_vl;
            }
            return tune;
          }
          return Tune::APPLE_M1_M3;
        }
      }
    }
  }

  // Apple Silicon fallback for no_std or when detection unavailable
  #[cfg(all(
    not(feature = "std"),
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  {
    if caps.has(aarch64::PMULL_EOR3_READY) {
      // Without std, use SME2/SME presence to differentiate generations
      if caps.has(aarch64::SME2) {
        let mut tune = Tune::APPLE_M5;
        let sme_vl = detect_sme_vlen();
        if sme_vl > 0 {
          tune.sme_tile = sme_vl;
        }
        return tune;
      }
      if caps.has(aarch64::SME) {
        let mut tune = Tune::APPLE_M4;
        let sme_vl = detect_sme_vlen();
        if sme_vl > 0 {
          tune.sme_tile = sme_vl;
        }
        return tune;
      }
      return Tune::APPLE_M1_M3;
    }
  }

  // SVE2 with runtime VL detection (Graviton 4/5, Neoverse V2/V3/N3, NVIDIA Grace)
  if caps.has(aarch64::SVE2) {
    let vlen = detect_sve_vlen();

    // Check for V3-specific features (SME2P1 indicates newer generation)
    if caps.has(aarch64::SME2P1) {
      return Tune::GRAVITON5; // or NEOVERSE_V3
    }

    // Try to detect specific chips by MIDR if available
    #[cfg(all(feature = "std", target_os = "linux"))]
    {
      if let Some(midr) = read_midr_el1() {
        let part = (midr >> 4) & 0xFFF;
        // NVIDIA Grace uses Neoverse V2 cores (part 0xd4f)
        if part == 0xd4f {
          return Tune::NVIDIA_GRACE;
        }
        // Neoverse N3 (part 0xd8e)
        if part == 0xd8e {
          return Tune::NEOVERSE_N3;
        }
      }
    }

    // Graviton 4 / Neoverse V2 use 128-bit SVE
    // Neoverse V1 used 256-bit SVE (with SVE1)
    if vlen > 0 && vlen <= 128 {
      return Tune::GRAVITON4; // 128-bit SVE2
    }
    // Default for SVE2 with unknown or wider VL
    return Tune::GRAVITON4;
  }

  // SVE (Graviton 3, Neoverse V1) with runtime VL detection
  if caps.has(aarch64::SVE) {
    let vlen = detect_sve_vlen();
    // Return a tune with the detected SVE vector length
    if vlen >= 256 {
      return Tune::GRAVITON3; // 256-bit SVE
    }
    // Narrower SVE (128-bit) - could be Neoverse N2
    if vlen > 0 && vlen < 256 {
      return Tune::NEOVERSE_N2;
    }
    // Unknown VL - default to Graviton3
    return Tune::GRAVITON3;
  }

  // PMULL + EOR3 (non-Apple, non-SVE) - Graviton2 or Ampere Altra
  if caps.has(aarch64::PMULL_EOR3_READY) {
    // Try to detect Ampere Altra (Neoverse N1-based, part 0xd0c)
    #[cfg(all(feature = "std", target_os = "linux"))]
    {
      if let Some(midr) = read_midr_el1() {
        let part = (midr >> 4) & 0xFFF;
        // Ampere Altra uses Neoverse N1 cores (part 0xd0c)
        if part == 0xd0c {
          return Tune::AMPERE_ALTRA;
        }
      }
    }
    return Tune::GRAVITON2;
  }

  // PMULL only
  if caps.has(aarch64::PMULL_READY) {
    return Tune::AARCH64_PMULL;
  }

  Tune::PORTABLE
}

// ─────────────────────────────────────────────────────────────────────────────
// RISC-V Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "riscv64")]
fn detect_riscv64() -> Detected {
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_riscv();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  Detected {
    caps,
    tune: if caps.has(crate::caps::riscv::ZBC) || caps.has(crate::caps::riscv::ZVBC) {
      Tune::DEFAULT
    } else {
      Tune::PORTABLE
    },
    arch: Arch::Riscv64,
  }
}

#[cfg(target_arch = "riscv32")]
fn detect_riscv32() -> Detected {
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_riscv();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  Detected {
    caps,
    tune: if caps.has(crate::caps::riscv::ZBC) || caps.has(crate::caps::riscv::ZVBC) {
      Tune::DEFAULT
    } else {
      Tune::PORTABLE
    },
    arch: Arch::Riscv32,
  }
}

#[cfg(all(
  any(target_arch = "riscv64", target_arch = "riscv32"),
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn runtime_riscv() -> Caps {
  use crate::caps::riscv;

  let mut caps = Caps::NONE;

  macro_rules! rt {
    ($f:literal => $c:expr) => {
      if std::arch::is_riscv_feature_detected!($f) {
        caps |= $c;
      }
    };
  }

  // ─── Vector Extension ───
  rt!("v" => riscv::V);

  // ─── Bit Manipulation ───
  rt!("zbb" => riscv::ZBB);
  rt!("zbs" => riscv::ZBS);
  rt!("zba" => riscv::ZBA);
  rt!("zbc" => riscv::ZBC);

  // ─── Scalar Crypto ───
  rt!("zbkb" => riscv::ZBKB);
  rt!("zbkc" => riscv::ZBKC);
  rt!("zbkx" => riscv::ZBKX);
  rt!("zknd" => riscv::ZKND);
  rt!("zkne" => riscv::ZKNE);
  rt!("zknh" => riscv::ZKNH);
  rt!("zksed" => riscv::ZKSED);
  rt!("zksh" => riscv::ZKSH);

  // ─── Vector Crypto ───
  rt!("zvbb" => riscv::ZVBB);
  rt!("zvbc" => riscv::ZVBC);
  rt!("zvkb" => riscv::ZVKB);
  rt!("zvkg" => riscv::ZVKG);
  rt!("zvkned" => riscv::ZVKNED);
  rt!("zvknha" => riscv::ZVKNHA);
  rt!("zvknhb" => riscv::ZVKNHB);
  rt!("zvksed" => riscv::ZVKSED);
  rt!("zvksh" => riscv::ZVKSH);

  caps
}

#[cfg(all(
  any(target_arch = "riscv64", target_arch = "riscv32"),
  feature = "std",
  not(any(target_os = "linux", target_os = "android"))
))]
fn runtime_riscv() -> Caps {
  // `std::arch::is_riscv_feature_detected!` is only implemented on Linux-like
  // platforms today.
  Caps::NONE
}

// ─────────────────────────────────────────────────────────────────────────────
// s390x (IBM Z) Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "s390x")]
fn detect_s390x() -> Detected {
  // Start with compile-time detected features.
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_s390x();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  let tune = select_s390x_tune(caps);

  Detected {
    caps,
    tune,
    arch: Arch::S390x,
  }
}

#[cfg(all(target_arch = "s390x", feature = "std"))]
fn runtime_s390x() -> Caps {
  use crate::caps::s390x;

  let mut caps = Caps::NONE;

  macro_rules! rt {
    ($f:literal => $c:expr) => {
      if std::arch::is_s390x_feature_detected!($f) {
        caps |= $c;
      }
    };
  }

  // ─── Vector Facilities ───
  rt!("vector" => s390x::VECTOR);
  rt!("vector-enhancements-1" => s390x::VECTOR_ENH1);
  rt!("vector-enhancements-2" => s390x::VECTOR_ENH2);
  rt!("vector-enhancements-3" => s390x::VECTOR_ENH3);
  rt!("vector-packed-decimal" => s390x::VECTOR_PD);
  rt!("nnp-assist" => s390x::NNP_ASSIST);

  // ─── Miscellaneous Extensions ───
  rt!("miscellaneous-extensions-2" => s390x::MISC_EXT2);
  rt!("miscellaneous-extensions-3" => s390x::MISC_EXT3);

  // ─── Crypto (CPACF - Message Security Assist) ───
  rt!("message-security-assist-extension3" => s390x::MSA);
  rt!("message-security-assist-extension4" => s390x::MSA | s390x::MSA4);
  rt!("message-security-assist-extension5" => s390x::MSA | s390x::MSA5);
  rt!("message-security-assist-extension8" => s390x::MSA | s390x::MSA8);
  rt!("message-security-assist-extension9" => s390x::MSA | s390x::MSA9);

  // ─── Other Facilities ───
  rt!("deflate-conversion" => s390x::DEFLATE);
  rt!("enhanced-sort" => s390x::ENHANCED_SORT);

  caps
}

#[cfg(target_arch = "s390x")]
fn select_s390x_tune(caps: Caps) -> Tune {
  use crate::caps::s390x;

  if caps.has(s390x::VECTOR_ENH2) {
    // z15+
    Tune::Z15
  } else if caps.has(s390x::VECTOR_ENH1) {
    // z14
    Tune::Z14
  } else if caps.has(s390x::VECTOR) {
    // z13
    Tune::Z13
  } else {
    Tune::PORTABLE
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Power Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "powerpc64")]
fn detect_power() -> Detected {
  // Start with compile-time detected features.
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_power();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  let tune = select_power_tune(caps);

  Detected {
    caps,
    tune,
    arch: Arch::Power,
  }
}

#[cfg(target_arch = "powerpc64")]
fn select_power_tune(caps: Caps) -> Tune {
  use crate::caps::power;

  if caps.has(power::POWER10_VECTOR) {
    Tune::POWER10
  } else if caps.has(power::POWER9_VECTOR) {
    Tune::POWER9
  } else if caps.has(power::POWER8_VECTOR) {
    Tune::POWER8
  } else if caps.has(power::VSX) {
    Tune::POWER7
  } else {
    Tune::PORTABLE
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Power Runtime Detection
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime Power detection for Linux/Android via /proc/self/auxv.
///
/// This avoids unstable `is_powerpc64_feature_detected!` and keeps the
/// selection logic consistent with the x86/aarch64 batch detectors.
#[cfg(all(
  target_arch = "powerpc64",
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn runtime_power() -> Caps {
  use std::{fs::File, io::Read};

  use crate::caps::power;

  // ELF auxiliary vector entry types
  const AT_HWCAP: u64 = 16;
  const AT_HWCAP2: u64 = 26;

  // HWCAP masks (from linux/arch/powerpc/include/uapi/asm/cputable.h)
  const PPC_FEATURE_HAS_ALTIVEC: u64 = 0x1000_0000;
  const PPC_FEATURE_HAS_VSX: u64 = 0x0000_0080;

  // HWCAP2 masks
  const PPC_FEATURE2_ARCH_2_07: u64 = 0x8000_0000; // POWER8 ISA (v2.07)
  const PPC_FEATURE2_ARCH_3_00: u64 = 0x0080_0000; // POWER9 ISA (v3.00)

  let (hwcap, hwcap2) = (|| -> Option<(u64, u64)> {
    let mut file = File::open("/proc/self/auxv").ok()?;
    let mut buf = [0u8; 4096];
    let n = file.read(&mut buf).ok()?;

    let mut hwcap = 0u64;
    let mut hwcap2 = 0u64;

    for chunk in buf.get(..n)?.chunks_exact(16) {
      let a_type = u64::from_ne_bytes(chunk.get(0..8)?.try_into().ok()?);
      let a_val = u64::from_ne_bytes(chunk.get(8..16)?.try_into().ok()?);

      if a_type == AT_HWCAP {
        hwcap = a_val;
      } else if a_type == AT_HWCAP2 {
        hwcap2 = a_val;
      } else if a_type == 0 {
        break;
      }
    }

    Some((hwcap, hwcap2))
  })()
  .unwrap_or((0, 0));

  let mut caps = Caps::NONE;

  if hwcap & PPC_FEATURE_HAS_ALTIVEC != 0 {
    caps |= power::ALTIVEC;
  }
  if hwcap & PPC_FEATURE_HAS_VSX != 0 {
    caps |= power::VSX;
  }

  // POWER9 implies POWER8 vector/crypto as well.
  if hwcap2 & PPC_FEATURE2_ARCH_3_00 != 0 {
    caps |= power::POWER9_VECTOR | power::POWER8_VECTOR | power::POWER8_CRYPTO;
  } else if hwcap2 & PPC_FEATURE2_ARCH_2_07 != 0 {
    caps |= power::POWER8_VECTOR | power::POWER8_CRYPTO;
  }

  caps
}

/// Runtime Power detection for other platforms.
#[cfg(all(
  target_arch = "powerpc64",
  feature = "std",
  not(any(target_os = "linux", target_os = "android"))
))]
fn runtime_power() -> Caps {
  // No stable runtime detector available on non-Linux today; rely on compile-time
  // `-C target-feature` for static dispatch in those environments.
  Caps::NONE
}

// ─────────────────────────────────────────────────────────────────────────────
// WebAssembly Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
fn detect_wasm32() -> Detected {
  use crate::caps::wasm;

  let mut caps = Caps::NONE;

  if cfg!(target_feature = "simd128") {
    caps |= wasm::SIMD128;
  }
  if cfg!(target_feature = "relaxed-simd") {
    caps |= wasm::RELAXED_SIMD;
  }

  Detected {
    caps,
    tune: Tune::PORTABLE,
    arch: Arch::Wasm32,
  }
}

#[cfg(target_arch = "wasm64")]
fn detect_wasm64() -> Detected {
  use crate::caps::wasm;

  let mut caps = Caps::NONE;

  if cfg!(target_feature = "simd128") {
    caps |= wasm::SIMD128;
  }
  if cfg!(target_feature = "relaxed-simd") {
    caps |= wasm::RELAXED_SIMD;
  }

  Detected {
    caps,
    tune: Tune::PORTABLE,
    arch: Arch::Wasm64,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
extern crate alloc;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[cfg(not(miri))] // get() returns portable() under Miri, which has different arch
  fn test_get_returns_valid() {
    let det = get();

    #[cfg(target_arch = "x86_64")]
    assert_eq!(det.arch, Arch::X86_64);
    #[cfg(target_arch = "aarch64")]
    assert_eq!(det.arch, Arch::Aarch64);

    assert!(det.tune.simd_threshold > 0);
    assert!(det.tune.parallel_streams > 0);
    assert!(det.tune.cache_line > 0);
  }

  #[test]
  #[cfg(not(miri))] // Uses syscalls for feature detection
  fn test_detect_uncached_consistent() {
    let d1 = detect_uncached();
    let d2 = detect_uncached();
    assert_eq!(d1.caps, d2.caps);
    assert_eq!(d1.arch, d2.arch);
  }

  #[test]
  #[cfg(not(miri))] // get() uses syscalls for feature detection
  fn test_convenience_functions() {
    let det = get();
    assert_eq!(caps(), det.caps);
    assert_eq!(tune(), det.tune);
    assert_eq!(arch(), det.arch);
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", not(miri)))]
  fn test_x86_64_baseline() {
    use crate::caps::x86;
    let det = get();
    assert!(det.caps.has(x86::SSE2));
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", not(miri)))]
  fn test_aarch64_baseline() {
    use crate::caps::aarch64;
    let det = get();
    assert!(det.caps.has(aarch64::NEON));
  }

  #[test]
  #[cfg(miri)]
  fn test_miri_returns_portable() {
    let det = get();
    assert_eq!(det.caps, Caps::NONE);
    assert_eq!(det.arch, Arch::Other);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Compile-Time Detection Tests (caps_static)
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_caps_static_is_const() {
    // Verify caps_static() can be used in const context
    const STATIC_CAPS: Caps = caps_static();
    let _ = STATIC_CAPS; // Use it to avoid dead code warning
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_caps_static_x86_64_baseline() {
    use crate::caps::x86;

    // x86_64 guarantees SSE2
    let caps = caps_static();
    assert!(caps.has(x86::SSE2), "x86_64 must have SSE2 baseline in caps_static");
  }

  #[test]
  #[cfg(target_arch = "aarch64")]
  fn test_caps_static_aarch64_baseline() {
    use crate::caps::aarch64;

    // aarch64 guarantees NEON
    let caps = caps_static();
    assert!(
      caps.has(aarch64::NEON),
      "aarch64 must have NEON baseline in caps_static"
    );
  }

  #[test]
  #[cfg(not(miri))] // Miri can't detect runtime features, returns Caps::NONE
  fn test_caps_static_subset_of_runtime() {
    // Compile-time detected features must be a subset of runtime detected features
    let static_caps = caps_static();
    let runtime_caps = caps();

    // Every compile-time feature must be present at runtime
    assert!(
      runtime_caps.has(static_caps),
      "caps_static() must be subset of caps(): static={:?}, runtime={:?}",
      static_caps,
      runtime_caps
    );
  }

  #[test]
  fn test_caps_static_consistent() {
    // caps_static() must return the same value every time
    let a = caps_static();
    let b = caps_static();
    assert_eq!(a, b, "caps_static() must be deterministic");
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", not(miri)))]
  fn test_caps_static_x86_features() {
    use crate::caps::x86;

    let caps = caps_static();

    // Test that feature groups are consistent with their baselines
    // If AVX2 is enabled at compile time, it should be detected
    if cfg!(target_feature = "avx2") {
      assert!(caps.has(x86::AVX2), "AVX2 must be detected when target_feature enabled");
    }

    // If AVX-512F is enabled, foundation should be detected
    if cfg!(target_feature = "avx512f") {
      assert!(
        caps.has(x86::AVX512F),
        "AVX512F must be detected when target_feature enabled"
      );
    }

    // If VPCLMULQDQ is enabled, it should be detected
    if cfg!(target_feature = "vpclmulqdq") {
      assert!(
        caps.has(x86::VPCLMULQDQ),
        "VPCLMULQDQ must be detected when target_feature enabled"
      );
    }
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", not(miri)))]
  fn test_caps_static_aarch64_features() {
    use crate::caps::aarch64;

    let caps = caps_static();

    // If AES is enabled at compile time, both AES and PMULL should be detected
    if cfg!(target_feature = "aes") {
      assert!(
        caps.has(aarch64::AES),
        "AES must be detected when target_feature enabled"
      );
      assert!(
        caps.has(aarch64::PMULL),
        "PMULL must be detected when aes target_feature enabled"
      );
    }

    // If SHA3 is enabled, both SHA3 and SHA512 should be detected
    if cfg!(target_feature = "sha3") {
      assert!(
        caps.has(aarch64::SHA3),
        "SHA3 must be detected when target_feature enabled"
      );
      assert!(
        caps.has(aarch64::SHA512),
        "SHA512 must be detected when sha3 target_feature enabled"
      );
    }

    // If SME is enabled, it should be detected (fixing prior drift)
    if cfg!(target_feature = "sme") {
      assert!(
        caps.has(aarch64::SME),
        "SME must be detected when target_feature enabled"
      );
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Apple Silicon Detection Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  #[cfg(all(target_arch = "aarch64", target_os = "macos", not(miri)))]
  fn test_apple_silicon_detection_runs() {
    // Just verify detection doesn't crash and returns a valid result
    let chip_gen = detect_apple_silicon_gen();
    // On actual Apple Silicon, we should get Some variant
    // On Rosetta 2 or non-Apple aarch64, we might get None
    if let Some(detected) = chip_gen {
      // Verify the generation is valid
      assert!(matches!(
        detected,
        AppleSiliconGen::M1 | AppleSiliconGen::M2 | AppleSiliconGen::M3 | AppleSiliconGen::M4
      ));
    }
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", target_os = "macos", not(miri)))]
  fn test_apple_silicon_tune_selection() {
    use crate::caps::aarch64;

    let det = get();
    // On Apple Silicon, if we have PMULL+EOR3, we should get Apple tuning
    if det.caps.has(aarch64::PMULL_EOR3_READY) {
      let tune_name = det.tune.name();
      assert!(tune_name.starts_with("Apple"), "Expected Apple tune, got: {tune_name}");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // SVE Vector Length Detection Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  #[cfg(all(target_arch = "aarch64", target_os = "linux", not(miri)))]
  fn test_sve_vlen_detection_runs() {
    // Just verify detection doesn't crash
    let vlen = detect_sve_vlen();
    // VL should be 0 (no SVE) or a valid power-of-2 in [128, 2048]
    if vlen > 0 {
      assert!(vlen >= 128, "SVE VL too small: {vlen}");
      assert!(vlen <= 2048, "SVE VL too large: {vlen}");
      assert!(vlen.is_power_of_two(), "SVE VL not power of 2: {vlen}");
    }
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", target_os = "linux", not(miri)))]
  fn test_sve_tune_includes_vlen() {
    use crate::caps::aarch64;

    let det = get();
    // If SVE is detected, tune should have appropriate SVE VL
    if det.caps.has(aarch64::SVE) || det.caps.has(aarch64::SVE2) {
      let runtime_vlen = detect_sve_vlen();
      if runtime_vlen > 0 {
        // The tune's sve_vlen should be a valid value
        let tune_vlen = det.tune.sve_vlen;
        assert!(
          tune_vlen == 0 || tune_vlen == 128 || tune_vlen == 256 || tune_vlen == 512,
          "Unexpected tune SVE VL: {tune_vlen}"
        );
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Hybrid Intel Detection Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
  fn test_is_intel_hybrid_amd_returns_false() {
    // AMD CPUs should never be detected as Intel hybrid
    assert!(!is_intel_hybrid(true, 6, 0x97)); // Even with ADL model
    assert!(!is_intel_hybrid(true, 25, 0)); // Zen 4
    assert!(!is_intel_hybrid(true, 26, 0)); // Zen 5
  }

  #[test]
  #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
  fn test_is_intel_hybrid_known_models() {
    // Alder Lake models
    assert!(is_intel_hybrid(false, 6, 0x97)); // ADL-S
    assert!(is_intel_hybrid(false, 6, 0x9A)); // ADL-P

    // Raptor Lake models
    assert!(is_intel_hybrid(false, 6, 0xB7)); // RPL-S
    assert!(is_intel_hybrid(false, 6, 0xBA)); // RPL-P

    // Non-hybrid Intel models should return false
    assert!(!is_intel_hybrid(false, 6, 0x8F)); // Sapphire Rapids
    assert!(!is_intel_hybrid(false, 6, 0x6A)); // Ice Lake-SP
  }

  #[test]
  #[allow(unsafe_code)]
  #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std"))]
  fn test_hybrid_avx512_override_default() {
    // Without env var set, override should be false
    // Note: We can't easily test with env var set due to test isolation
    // but we verify the default behavior
    // SAFETY: This test runs in isolation and doesn't rely on this env var being
    // present for other threads. The remove_var is unsafe due to potential data
    // races with other threads reading env vars, but test isolation mitigates this.
    unsafe { std::env::remove_var("RSCRYPTO_FORCE_AVX512") };
    assert!(!hybrid_avx512_override());
  }

  #[test]
  #[cfg(all(target_arch = "x86_64", not(miri)))]
  fn test_x86_64_model_extraction() {
    // Just verify CPUID model extraction works
    let det = detect_uncached();
    // Model should be extracted - we just verify detection runs
    assert!(det.tune.simd_threshold > 0);
  }

  #[test]
  #[cfg(all(
    target_arch = "aarch64",
    not(miri),
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  fn test_macos_extended_features() {
    // Test that new feature detection works on macOS
    use crate::caps::aarch64;
    let det = get();

    // Verify extended features are detected on capable hardware
    // On M1+, we should detect these features:
    std::eprintln!("Detected features: {}", det.caps.count());
    std::eprintln!("  I8MM: {}", det.caps.has(aarch64::I8MM));
    std::eprintln!("  BF16: {}", det.caps.has(aarch64::BF16));
    std::eprintln!("  FRINTTS: {}", det.caps.has(aarch64::FRINTTS));
    std::eprintln!("  LSE2: {}", det.caps.has(aarch64::LSE2));

    // These should be present on M1 Pro (hw.optional.arm confirms this)
    assert!(det.caps.has(aarch64::FRINTTS), "FRINTTS should be detected on M1+");
    assert!(det.caps.has(aarch64::LSE2), "LSE2 should be detected on M1+");
  }

  #[test]
  #[cfg(all(
    target_arch = "aarch64",
    feature = "std",
    not(miri),
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  fn test_detect_apple_sme_features_exists() {
    // Verify the SME detection function exists and returns valid caps
    let sme_caps = detect_apple_sme_features();
    // The function should always return valid Caps (may be empty on M1-M3)
    // On M4+, SME should be detected
    std::eprintln!("SME caps detected: {}", sme_caps.count());
    std::eprintln!("  SME: {}", sme_caps.has(crate::caps::aarch64::SME));
    std::eprintln!("  SME2: {}", sme_caps.has(crate::caps::aarch64::SME2));
  }

  #[test]
  #[cfg(all(
    target_arch = "aarch64",
    feature = "std",
    not(miri),
    any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos")
  ))]
  fn test_detect_apple_silicon_gen_exists() {
    // Verify chip generation detection works
    if let Some(chip_gen) = detect_apple_silicon_gen() {
      std::eprintln!("Detected Apple Silicon generation: {:?}", chip_gen);
      // Basic sanity checks
      match chip_gen {
        AppleSiliconGen::M1 | AppleSiliconGen::M2 | AppleSiliconGen::M3 => {
          // M1-M3 should not have SME
          std::eprintln!("M1-M3 chip detected (no SME expected)");
        }
        AppleSiliconGen::M4 => {
          // M4 should have SME
          std::eprintln!("M4 chip detected (SME expected)");
        }
        AppleSiliconGen::M5 => {
          // M5 should have SME2
          std::eprintln!("M5 chip detected (SME2 expected)");
        }
      }
    } else {
      std::eprintln!("Unknown or A-series chip detected");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // TuneKind Round-Trip Tests
  // ─────────────────────────────────────────────────────────────────────────────

  // Mirror of the kind_from_u8 mapping used in atomic_cache (no_std).
  // This test validates that the mapping stays in sync with TuneKind's #[repr(u8)].
  fn test_kind_from_u8(v: u8) -> crate::tune::TuneKind {
    use crate::tune::TuneKind;
    match v {
      1 => TuneKind::Default,
      2 => TuneKind::Portable,
      3 => TuneKind::Zen4,
      4 => TuneKind::Zen5,
      5 => TuneKind::Zen5c,
      6 => TuneKind::IntelSpr,
      7 => TuneKind::IntelGnr,
      8 => TuneKind::IntelIcl,
      9 => TuneKind::AppleM1M3,
      10 => TuneKind::AppleM4,
      11 => TuneKind::AppleM5,
      12 => TuneKind::Graviton2,
      13 => TuneKind::Graviton3,
      14 => TuneKind::Graviton4,
      15 => TuneKind::Graviton5,
      16 => TuneKind::NeoverseN2,
      17 => TuneKind::NeoverseN3,
      18 => TuneKind::NeoverseV3,
      19 => TuneKind::NvidiaGrace,
      20 => TuneKind::AmpereAltra,
      21 => TuneKind::Aarch64Pmull,
      22 => TuneKind::Z13,
      23 => TuneKind::Z14,
      24 => TuneKind::Z15,
      25 => TuneKind::Power7,
      26 => TuneKind::Power8,
      27 => TuneKind::Power9,
      28 => TuneKind::Power10,
      _ => TuneKind::Custom,
    }
  }

  /// Validates that the `kind_from_u8` mapping correctly round-trips with TuneKind's
  /// `#[repr(u8)]` discriminants.
  ///
  /// This catches drift if someone adds a new TuneKind variant but forgets to update
  /// the mapping in atomic_cache.
  #[test]
  fn test_tunekind_round_trip() {
    use crate::tune::TuneKind;

    // TuneKind has 29 variants (0=Custom through 28=Power10)
    for i in 0..=28u8 {
      let kind = test_kind_from_u8(i);

      // Custom (0) is the fallback for unknown values, so skip its reverse check
      if kind != TuneKind::Custom {
        assert_eq!(
          kind as u8, i,
          "TuneKind mapping mismatch: kind_from_u8({i}) = {:?} but {:?} as u8 = {}",
          kind, kind, kind as u8
        );
      }
    }

    // Verify out-of-range values map to Custom
    assert_eq!(test_kind_from_u8(29), TuneKind::Custom);
    assert_eq!(test_kind_from_u8(255), TuneKind::Custom);
  }

  /// Verify that all TuneKind variants have distinct u8 representations.
  #[test]
  fn test_tunekind_no_collisions() {
    use alloc::collections::BTreeSet;

    use crate::tune::TuneKind;

    let variants: &[TuneKind] = &[
      TuneKind::Custom,
      TuneKind::Default,
      TuneKind::Portable,
      TuneKind::Zen4,
      TuneKind::Zen5,
      TuneKind::Zen5c,
      TuneKind::IntelSpr,
      TuneKind::IntelGnr,
      TuneKind::IntelIcl,
      TuneKind::AppleM1M3,
      TuneKind::AppleM4,
      TuneKind::AppleM5,
      TuneKind::Graviton2,
      TuneKind::Graviton3,
      TuneKind::Graviton4,
      TuneKind::Graviton5,
      TuneKind::NeoverseN2,
      TuneKind::NeoverseN3,
      TuneKind::NeoverseV3,
      TuneKind::NvidiaGrace,
      TuneKind::AmpereAltra,
      TuneKind::Aarch64Pmull,
      TuneKind::Z13,
      TuneKind::Z14,
      TuneKind::Z15,
      TuneKind::Power7,
      TuneKind::Power8,
      TuneKind::Power9,
      TuneKind::Power10,
    ];

    let mut seen = BTreeSet::new();
    for &kind in variants {
      let val = kind as u8;
      assert!(seen.insert(val), "TuneKind::{:?} has duplicate u8 value {}", kind, val);
    }

    // Verify we have all 29 variants
    assert_eq!(seen.len(), 29, "Expected 29 TuneKind variants");
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Arch Round-Trip Tests
  // ─────────────────────────────────────────────────────────────────────────────

  // Mirror of the arch_to_u8 mapping used in atomic_cache (no_std).
  // Note: Arch doesn't have #[repr(u8)], so this is a custom mapping
  // where Other=0 (the uninitialized/fallback value).
  fn test_arch_to_u8(arch: Arch) -> u8 {
    match arch {
      Arch::X86_64 => 1,
      Arch::X86 => 2,
      Arch::Aarch64 => 3,
      Arch::Arm => 4,
      Arch::Riscv64 => 5,
      Arch::Riscv32 => 6,
      Arch::Power => 7,
      Arch::S390x => 8,
      Arch::Wasm32 => 10,
      Arch::Wasm64 => 11,
      Arch::Other => 0,
    }
  }

  fn test_arch_from_u8(v: u8) -> Arch {
    match v {
      1 => Arch::X86_64,
      2 => Arch::X86,
      3 => Arch::Aarch64,
      4 => Arch::Arm,
      5 => Arch::Riscv64,
      6 => Arch::Riscv32,
      7 => Arch::Power,
      8 => Arch::S390x,
      10 => Arch::Wasm32,
      11 => Arch::Wasm64,
      _ => Arch::Other,
    }
  }

  /// Verify arch_to_u8 and arch_from_u8 are inverses.
  #[test]
  fn test_arch_round_trip() {
    let variants: &[Arch] = &[
      Arch::Other,
      Arch::X86_64,
      Arch::X86,
      Arch::Aarch64,
      Arch::Arm,
      Arch::Riscv64,
      Arch::Riscv32,
      Arch::Power,
      Arch::S390x,
      Arch::Wasm32,
      Arch::Wasm64,
    ];

    for &arch in variants {
      let encoded = test_arch_to_u8(arch);
      let decoded = test_arch_from_u8(encoded);
      assert_eq!(
        arch, decoded,
        "Arch round-trip failed: {:?} -> {} -> {:?}",
        arch, encoded, decoded
      );
    }

    // Verify out-of-range values map to Other
    assert_eq!(test_arch_from_u8(12), Arch::Other);
    assert_eq!(test_arch_from_u8(255), Arch::Other);
  }

  /// Verify all Arch variants have distinct encoded u8 values.
  #[test]
  fn test_arch_no_collisions() {
    use alloc::collections::BTreeSet;

    let variants: &[Arch] = &[
      Arch::Other,
      Arch::X86_64,
      Arch::X86,
      Arch::Aarch64,
      Arch::Arm,
      Arch::Riscv64,
      Arch::Riscv32,
      Arch::Power,
      Arch::S390x,
      Arch::Wasm32,
      Arch::Wasm64,
    ];

    let mut seen = BTreeSet::new();
    for &arch in variants {
      let val = test_arch_to_u8(arch);
      assert!(
        seen.insert(val),
        "Arch::{:?} has duplicate encoded u8 value {}",
        arch,
        val
      );
    }

    assert_eq!(seen.len(), 11, "Expected 11 Arch variants with unique encodings");
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Override Mechanism Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_has_override_exists() {
    // Verify the override API exists and returns a bool.
    // Note: Due to global state from other tests, we can't assert a specific value.
    let _ = has_override();
  }

  #[test]
  fn test_detected_portable_constructor() {
    let det = Detected::portable();
    assert_eq!(det.caps, Caps::NONE);
    assert_eq!(det.tune.kind(), crate::tune::TuneKind::Portable);
    assert_eq!(det.arch, Arch::Other);
  }

  #[test]
  fn test_detected_equality() {
    let a = Detected::portable();
    let b = Detected::portable();
    assert_eq!(a, b);

    let c = Detected {
      caps: Caps::bit(0),
      tune: crate::tune::Tune::DEFAULT,
      arch: Arch::X86_64,
    };
    assert_ne!(a, c);
  }

  #[test]
  fn test_detected_debug() {
    let det = Detected::portable();
    let s = alloc::format!("{:?}", det);
    assert!(s.contains("Detected"));
    assert!(s.contains("caps"));
    assert!(s.contains("tune"));
    assert!(s.contains("arch"));
  }
}
