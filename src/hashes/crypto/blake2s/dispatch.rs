//! Blake2s kernel dispatch with `OnceCache` caching.
//!
//! # Non-x86 SIMD policy
//!
//! The AArch64 NEON, POWER VSX, and s390x vector kernels are correct, but
//! single-block SIMD loses to the portable scalar path in CI. The 2026-04-25
//! Linux bench run showed Blake2s at roughly 0.51x on Graviton3, 0.57x on
//! Graviton4, 0.81x on s390x, and 0.86x on POWER10 versus RustCrypto. Keep
//! those kernels available for diagnostic forced-kernel benches, but do not
//! select them in production dispatch.
//!
//! Regaining dominance on those platforms requires multi-block kernels
//! (Blake3-style 2-way or 4-way), not more single-block shuffle tuning.

use super::kernels::{Blake2sKernelId, CompressFn, compress_fn};
define_blake2_dispatch! {
  kernel_id: Blake2sKernelId,
  compress_fn_ty: CompressFn,
  portable_kernel: Blake2sKernelId::Portable,
  compress_fn: compress_fn,
  required_caps: super::kernels::required_caps,
  candidates: [
    #[cfg(target_arch = "x86_64")]
    Blake2sKernelId::X86Avx512vl,
    #[cfg(target_arch = "x86_64")]
    Blake2sKernelId::X86Avx2,
    #[cfg(target_arch = "riscv64")]
    Blake2sKernelId::Riscv64V,
    #[cfg(target_arch = "wasm32")]
    Blake2sKernelId::WasmSimd128,
  ],
}
