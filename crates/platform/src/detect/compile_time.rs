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
