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
  ],
}
