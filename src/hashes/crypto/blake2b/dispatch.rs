//! Blake2b kernel dispatch.
//!
//! The retired AArch64 NEON, POWER VSX, and s390x vector kernels lost to
//! portable Rust in the 2026-04-27 target-native measurements.

use super::kernels::{Blake2bKernelId, CompressBlocksFn, CompressFn, compress_blocks_fn, compress_fn};
define_blake2_dispatch! {
  kernel_id: Blake2bKernelId,
  compress_fn_ty: CompressFn,
  compress_blocks_fn_ty: CompressBlocksFn,
  portable_kernel: Blake2bKernelId::Portable,
  compress_fn: compress_fn,
  compress_blocks_fn: compress_blocks_fn,
  required_caps: super::kernels::required_caps,
  candidates: [
    #[cfg(target_arch = "x86_64")]
    Blake2bKernelId::X86Avx512vl,
    #[cfg(target_arch = "x86_64")]
    Blake2bKernelId::X86Avx2,
    #[cfg(target_arch = "riscv64")]
    Blake2bKernelId::Riscv64V,
    #[cfg(target_arch = "wasm32")]
    Blake2bKernelId::WasmSimd128,
  ],
}
