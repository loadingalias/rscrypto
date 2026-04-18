//! Blake2s kernel dispatch with `OnceCache` caching.
//!
//! # aarch64-apple-darwin gate
//!
//! NEON dispatch is gated off on aarch64-macos because the 4-lane Blake2s
//! NEON kernel measures 1.29× slower than the scalar portable kernel on
//! Apple Silicon (104 ns vs 134 ns per Blake2s single-block compress,
//! criterion `--quick`, M-series host, 2026-04-17 re-measure after the
//! Phase D `ror8` `vqtbl1q_u8` swap closed ~12% of the gap, down from
//! 153 ns). All four rotations (`vrev32q_u16`, `vsriq`-based rot12/7,
//! `vqtbl1q_u8` rot8) are now at the BLAKE3 PR #319 recommended forms;
//! the remaining loss is structural (single-block NEON cannot keep up
//! with M-series wide scalar on a 4-lane u32 workload).
//!
//! Ungating this on macOS would require a multi-block NEON kernel rather
//! than tuning the single-block path.

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
use super::kernels::required_caps;
use super::kernels::{Blake2sKernelId, CompressFn, compress_fn};
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
use crate::backend::cache::OnceCache;

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
#[derive(Clone, Copy)]
struct Resolved {
  compress: CompressFn,
  #[cfg(any(test, feature = "diag"))]
  name: &'static str,
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
static ACTIVE: OnceCache<Resolved> = OnceCache::new();

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn resolve() -> Resolved {
  let caps = crate::platform::caps();

  let candidates: &[Blake2sKernelId] = &[
    #[cfg(target_arch = "x86_64")]
    Blake2sKernelId::X86Avx512vl,
    #[cfg(target_arch = "x86_64")]
    Blake2sKernelId::X86Avx2,
    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
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
  #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
  {
    compress_fn(Blake2sKernelId::Portable)
  }
  #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
  ACTIVE.get_or_init(resolve).compress
}

/// Active kernel name for introspection.
#[cfg(any(test, feature = "diag"))]
#[inline]
#[must_use]
pub fn kernel_name_for_len(_len: usize) -> &'static str {
  if super::kernels::COMPILE_TIME_HW {
    return compile_time_name();
  }
  #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
  {
    Blake2sKernelId::Portable.as_str()
  }
  #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
  ACTIVE.get_or_init(resolve).name
}

#[cfg(any(test, feature = "diag"))]
const fn compile_time_name() -> &'static str {
  if cfg!(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl"
  )) {
    "x86/avx512vl"
  } else if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
    "x86/avx2"
  } else if cfg!(all(
    target_arch = "aarch64",
    target_feature = "neon",
    not(target_os = "macos")
  )) {
    "aarch64/neon"
  } else {
    "portable"
  }
}
