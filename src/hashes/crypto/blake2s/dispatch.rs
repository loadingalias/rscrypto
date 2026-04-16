//! Blake2s kernel dispatch with `OnceCache` caching.

use super::kernels::{Blake2sKernelId, CompressFn, compress_fn, required_caps};
use crate::backend::cache::OnceCache;

#[derive(Clone, Copy)]
struct Resolved {
  compress: CompressFn,
  #[cfg(any(test, feature = "diag"))]
  name: &'static str,
}

static ACTIVE: OnceCache<Resolved> = OnceCache::new();

fn resolve() -> Resolved {
  let caps = crate::platform::caps();

  let candidates: &[Blake2sKernelId] = &[
    #[cfg(target_arch = "x86_64")]
    Blake2sKernelId::X86Avx512vl,
    #[cfg(target_arch = "x86_64")]
    Blake2sKernelId::X86Avx2,
    #[cfg(target_arch = "aarch64")]
    Blake2sKernelId::Aarch64Neon,
    #[cfg(target_arch = "s390x")]
    Blake2sKernelId::S390xVector,
    #[cfg(target_arch = "powerpc64")]
    Blake2sKernelId::PowerVsx,
    #[cfg(target_arch = "riscv64")]
    Blake2sKernelId::Riscv64V,
    #[cfg(target_arch = "wasm32")]
    Blake2sKernelId::WasmSimd128,
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
    compress: compress_fn(Blake2sKernelId::Portable),
    #[cfg(any(test, feature = "diag"))]
    name: Blake2sKernelId::Portable.as_str(),
  }
}

#[inline]
#[must_use]
pub(crate) fn compress_dispatch() -> CompressFn {
  if super::kernels::COMPILE_TIME_HW {
    return super::kernels::compile_time_best();
  }
  ACTIVE.get_or_init(resolve).compress
}
