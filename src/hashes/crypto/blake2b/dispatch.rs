//! Blake2b kernel dispatch with `OnceCache` caching.

use super::kernels::{Blake2bKernelId, CompressFn, compress_fn, required_caps};
use crate::backend::cache::OnceCache;

#[derive(Clone, Copy)]
struct Resolved {
  compress: CompressFn,
  #[cfg(any(test, feature = "diag"))]
  name: &'static str,
}

static ACTIVE: OnceCache<Resolved> = OnceCache::new();

/// Resolve the best available Blake2b kernel at runtime.
fn resolve() -> Resolved {
  let caps = crate::platform::caps();

  // Preference order: AVX-512VL → AVX2 → NEON → z/Vector → VSX → RVV → WASM → Portable
  let candidates: &[Blake2bKernelId] = &[
    #[cfg(target_arch = "x86_64")]
    Blake2bKernelId::X86Avx512vl,
    #[cfg(target_arch = "x86_64")]
    Blake2bKernelId::X86Avx2,
    #[cfg(target_arch = "aarch64")]
    Blake2bKernelId::Aarch64Neon,
    #[cfg(target_arch = "s390x")]
    Blake2bKernelId::S390xVector,
    #[cfg(target_arch = "powerpc64")]
    Blake2bKernelId::PowerVsx,
    #[cfg(target_arch = "riscv64")]
    Blake2bKernelId::Riscv64V,
    #[cfg(target_arch = "wasm32")]
    Blake2bKernelId::WasmSimd128,
  ];

  for &id in candidates {
    if caps.has(required_caps(id)) {
      return Resolved {
        compress: compress_fn(id),
        #[cfg(any(test, feature = "diag"))]
        name: id.as_str(),
      };
    }
  }

  Resolved {
    compress: compress_fn(Blake2bKernelId::Portable),
    #[cfg(any(test, feature = "diag"))]
    name: Blake2bKernelId::Portable.as_str(),
  }
}

/// Return the best available compress function (cached after first call).
#[inline]
#[must_use]
pub(crate) fn compress_dispatch() -> CompressFn {
  if super::kernels::COMPILE_TIME_HW {
    return super::kernels::compile_time_best();
  }
  ACTIVE.get_or_init(resolve).compress
}

/// Active kernel name for introspection.
#[cfg(any(test, feature = "diag"))]
#[inline]
#[must_use]
pub fn kernel_name() -> &'static str {
  if super::kernels::COMPILE_TIME_HW {
    return compile_time_name();
  }
  ACTIVE.get_or_init(resolve).name
}

#[cfg(any(test, feature = "diag"))]
const fn compile_time_name() -> &'static str {
  if cfg!(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl")) {
    "x86/avx512vl"
  } else if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
    "x86/avx2"
  } else if cfg!(all(target_arch = "aarch64", target_feature = "neon")) {
    "aarch64/neon"
  } else {
    "portable"
  }
}
