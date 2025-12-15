//! x86_64 SIMD dispatch for CRC algorithms.
//!
//! Provides runtime and compile-time dispatch to the fastest available
//! implementation for CRC32, CRC32C, CRC64/XZ, and CRC64/NVME.
//!
//! Uses the `platform` crate for microarchitecture detection to select
//! optimal implementations based on CPU capabilities.
//!
//! # AMD Zen Hybrid Optimization
//!
//! On AMD Zen 4/5, a hybrid approach combining parallel `crc32q` scalar
//! streams with VPCLMULQDQ achieves higher throughput than either alone:
//!
//! - Zen 4: 3-way crc32q + VPCLMULQDQ
//! - Zen 5: 7-way crc32q + VPCLMULQDQ (Zen 5 supports 7-way crc32q parallelism)
//!
//! This works because `crc32q` and VPCLMULQDQ use different execution ports,
//! allowing them to run in parallel on AMD's architecture.

#[cfg(any(all(target_feature = "pclmulqdq", target_feature = "ssse3"), feature = "std"))]
pub(crate) mod pclmul;

#[cfg(any(
  all(
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512bw",
    target_feature = "vpclmulqdq",
    target_feature = "pclmulqdq"
  ),
  feature = "std"
))]
pub(crate) mod vpclmul;

#[cfg(any(
  all(
    target_feature = "sse4.2",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512bw",
    target_feature = "vpclmulqdq",
    target_feature = "pclmulqdq"
  ),
  feature = "std"
))]
pub mod hybrid;

// Specialized threshold variants for CRC32C.
// These have baked-in thresholds to avoid per-call `detect_microarch()` overhead.
// Selected at detection time based on microarchitecture.

/// VPCLMUL with SSE4.2 fallback, threshold=64 (AMD Zen fast ZMM warmup).
#[cfg(feature = "std")]
#[inline]
fn compute_vpclmul_with_sse42_threshold_64(crc: u32, data: &[u8]) -> u32 {
  if data.len() < 64 {
    return crate::crc32c::x86_64::compute_sse42_runtime(crc, data);
  }
  vpclmul::compute_vpclmul_runtime(crc, data)
}

/// VPCLMUL with SSE4.2 fallback, threshold=256 (Intel/others with higher ZMM warmup).
#[cfg(feature = "std")]
#[inline]
fn compute_vpclmul_with_sse42_threshold_256(crc: u32, data: &[u8]) -> u32 {
  if data.len() < 256 {
    return crate::crc32c::x86_64::compute_sse42_runtime(crc, data);
  }
  vpclmul::compute_vpclmul_runtime(crc, data)
}

/// PCLMUL with SSE4.2 fallback, threshold=64.
#[cfg(feature = "std")]
#[inline]
fn compute_pclmul_with_sse42_threshold_64(crc: u32, data: &[u8]) -> u32 {
  if data.len() < 64 {
    return crate::crc32c::x86_64::compute_sse42_runtime(crc, data);
  }
  pclmul::compute_pclmul_runtime(crc, data)
}

/// Select the best available CRC32C implementation at runtime (std only).
///
/// The returned function computes the *raw* CRC state (no final XOR).
///
/// # Dispatch Priority
///
/// 1. **AMD Zen 5 + VPCLMULQDQ**: Hybrid 7-way crc32q + VPCLMULQDQ
/// 2. **AMD Zen 4 + VPCLMULQDQ**: Hybrid 3-way crc32q + VPCLMULQDQ
/// 3. **Other VPCLMULQDQ (Intel SPR, etc.)**: Pure VPCLMULQDQ with SSE4.2 small-buffer fallback
/// 4. **PCLMULQDQ**: 128-bit folding with SSE4.2 fallback
/// 5. **SSE4.2 only**: Hardware crc32 instruction
/// 6. **Portable**: Slicing-by-8 fallback
///
/// # Threshold Selection
///
/// Functions with baked-in thresholds are selected based on microarchitecture:
/// - AMD (Zen 4/5): 64-byte threshold (fast ZMM warmup)
/// - Intel/others: 256-byte threshold (slower ZMM warmup)
#[cfg(feature = "std")]
pub(crate) fn detect_crc32c_best() -> fn(u32, &[u8]) -> u32 {
  use platform::x86_64::{MicroArch, detect_microarch};

  let arch = detect_microarch();
  let has_sse42 = std::arch::is_x86_feature_detected!("sse4.2");

  // AMD Zen 4/5 benefit from hybrid scalar+SIMD approach.
  // The crc32q instruction uses different execution ports than VPCLMULQDQ,
  // so running them in parallel achieves higher throughput.
  if arch.has_vpclmulqdq() && has_sse42 {
    match arch {
      MicroArch::Zen5 => return hybrid::compute_hybrid_zen5_runtime,
      MicroArch::Zen4 => return hybrid::compute_hybrid_zen4_runtime,
      _ => {}
    }
  }

  // For non-AMD or older AMD, use pure VPCLMULQDQ (optimal for Intel).
  // Select threshold variant based on ZMM warmup characteristics.
  if arch.has_vpclmulqdq() {
    if has_sse42 {
      // AMD has fast (~60ns) ZMM warmup, Intel has slow (~2000ns) warmup.
      return if arch.has_fast_zmm_warmup() {
        compute_vpclmul_with_sse42_threshold_64
      } else {
        compute_vpclmul_with_sse42_threshold_256
      };
    }
    return vpclmul::compute_vpclmul_runtime;
  }

  if arch.has_pclmulqdq() {
    if has_sse42 {
      // PCLMUL doesn't use ZMM registers, but we still benefit from
      // using SSE4.2 for small buffers. Use 64-byte threshold universally.
      return compute_pclmul_with_sse42_threshold_64;
    }
    return pclmul::compute_pclmul_runtime;
  }

  if has_sse42 {
    return crate::crc32c::x86_64::compute_sse42_runtime;
  }

  crate::crc32c::portable::compute
}

// Specialized threshold variants for CRC32 (ISO-HDLC).
// CRC32 doesn't have hardware instruction like CRC32C, so fallback is portable.

/// VPCLMUL with portable fallback, threshold=64.
#[cfg(feature = "std")]
#[inline]
fn compute_vpclmul_crc32_threshold_64(crc: u32, data: &[u8]) -> u32 {
  if data.len() < 64 {
    return crate::crc32::portable::compute(crc, data);
  }
  vpclmul::compute_vpclmul_crc32_runtime(crc, data)
}

/// VPCLMUL with portable fallback, threshold=256.
#[cfg(feature = "std")]
#[inline]
fn compute_vpclmul_crc32_threshold_256(crc: u32, data: &[u8]) -> u32 {
  if data.len() < 256 {
    return crate::crc32::portable::compute(crc, data);
  }
  vpclmul::compute_vpclmul_crc32_runtime(crc, data)
}

/// PCLMUL with portable fallback, threshold=64.
#[cfg(feature = "std")]
#[inline]
fn compute_pclmul_crc32_threshold_64(crc: u32, data: &[u8]) -> u32 {
  if data.len() < 64 {
    return crate::crc32::portable::compute(crc, data);
  }
  pclmul::compute_pclmul_crc32_runtime(crc, data)
}

/// Select the best available CRC32 implementation at runtime (std only).
///
/// The returned function computes the *raw* CRC state (no final XOR).
#[cfg(feature = "std")]
pub(crate) fn detect_crc32_best() -> fn(u32, &[u8]) -> u32 {
  use platform::x86_64::detect_microarch;

  let arch = detect_microarch();

  if arch.has_vpclmulqdq() {
    return if arch.has_fast_zmm_warmup() {
      compute_vpclmul_crc32_threshold_64
    } else {
      compute_vpclmul_crc32_threshold_256
    };
  }

  if arch.has_pclmulqdq() {
    return compute_pclmul_crc32_threshold_64;
  }

  crate::crc32::portable::compute
}

// ============================================================================
// CRC64 Dispatch
// ============================================================================

// Specialized threshold variants for CRC64/XZ.

/// VPCLMUL CRC64/XZ with portable fallback, threshold=64.
#[cfg(feature = "std")]
#[inline]
fn compute_vpclmul_crc64_xz_threshold_64(crc: u64, data: &[u8]) -> u64 {
  if data.len() < 64 {
    return crate::crc64::xz::compute_portable(crc, data);
  }
  vpclmul::compute_vpclmul_crc64_xz_runtime(crc, data)
}

/// VPCLMUL CRC64/XZ with portable fallback, threshold=256.
#[cfg(feature = "std")]
#[inline]
fn compute_vpclmul_crc64_xz_threshold_256(crc: u64, data: &[u8]) -> u64 {
  if data.len() < 256 {
    return crate::crc64::xz::compute_portable(crc, data);
  }
  vpclmul::compute_vpclmul_crc64_xz_runtime(crc, data)
}

/// PCLMUL CRC64/XZ with portable fallback, threshold=64.
#[cfg(feature = "std")]
#[inline]
fn compute_pclmul_crc64_xz_threshold_64(crc: u64, data: &[u8]) -> u64 {
  if data.len() < 64 {
    return crate::crc64::xz::compute_portable(crc, data);
  }
  pclmul::compute_pclmul_crc64_xz_runtime(crc, data)
}

/// Select the best available CRC64/XZ implementation at runtime (std only).
///
/// The returned function computes the *raw* CRC state (no final XOR).
#[cfg(feature = "std")]
pub(crate) fn detect_crc64_xz_best() -> fn(u64, &[u8]) -> u64 {
  use platform::x86_64::detect_microarch;

  let arch = detect_microarch();

  if arch.has_vpclmulqdq() {
    return if arch.has_fast_zmm_warmup() {
      compute_vpclmul_crc64_xz_threshold_64
    } else {
      compute_vpclmul_crc64_xz_threshold_256
    };
  }

  if arch.has_pclmulqdq() {
    return compute_pclmul_crc64_xz_threshold_64;
  }

  crate::crc64::xz::compute_portable
}

// Specialized threshold variants for CRC64/NVME.

/// VPCLMUL CRC64/NVME with portable fallback, threshold=64.
#[cfg(feature = "std")]
#[inline]
fn compute_vpclmul_crc64_nvme_threshold_64(crc: u64, data: &[u8]) -> u64 {
  if data.len() < 64 {
    return crate::crc64::nvme::compute_portable(crc, data);
  }
  vpclmul::compute_vpclmul_crc64_nvme_runtime(crc, data)
}

/// VPCLMUL CRC64/NVME with portable fallback, threshold=256.
#[cfg(feature = "std")]
#[inline]
fn compute_vpclmul_crc64_nvme_threshold_256(crc: u64, data: &[u8]) -> u64 {
  if data.len() < 256 {
    return crate::crc64::nvme::compute_portable(crc, data);
  }
  vpclmul::compute_vpclmul_crc64_nvme_runtime(crc, data)
}

/// PCLMUL CRC64/NVME with portable fallback, threshold=64.
#[cfg(feature = "std")]
#[inline]
fn compute_pclmul_crc64_nvme_threshold_64(crc: u64, data: &[u8]) -> u64 {
  if data.len() < 64 {
    return crate::crc64::nvme::compute_portable(crc, data);
  }
  pclmul::compute_pclmul_crc64_nvme_runtime(crc, data)
}

/// Select the best available CRC64/NVME implementation at runtime (std only).
///
/// The returned function computes the *raw* CRC state (no final XOR).
#[cfg(feature = "std")]
pub(crate) fn detect_crc64_nvme_best() -> fn(u64, &[u8]) -> u64 {
  use platform::x86_64::detect_microarch;

  let arch = detect_microarch();

  if arch.has_vpclmulqdq() {
    return if arch.has_fast_zmm_warmup() {
      compute_vpclmul_crc64_nvme_threshold_64
    } else {
      compute_vpclmul_crc64_nvme_threshold_256
    };
  }

  if arch.has_pclmulqdq() {
    return compute_pclmul_crc64_nvme_threshold_64;
  }

  crate::crc64::nvme::compute_portable
}
