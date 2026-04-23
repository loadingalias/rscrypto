//! Blake2b kernel dispatch with `OnceCache` caching.
//!
//! # aarch64-apple-darwin gate
//!
//! NEON dispatch is gated off on aarch64-macos because the 2-lane
//! `vextq_u64`-based diagonalize NEON kernel measures 1.33× slower than the
//! scalar portable kernel on Apple Silicon (121 ns vs 161 ns per Blake2b
//! single-block compress, criterion `--quick`, M-series host, 2026-04-17
//! re-measure after the Phase D audit). The 64-bit lane width means the
//! single-block NEON kernel only extracts 2-way parallelism per G round,
//! which M-series wide OoO cores already match with the scalar kernel
//! (rotations are already SotA per BLAKE3 PR #319 and Leigh Brown's
//! `blake2b-round.h`).
//!
//! Ungating this on macOS would require a multi-block NEON kernel
//! (Blake3-style 2-way or 4-way) rather than tuning the single-block path.

use super::kernels::{Blake2bKernelId, CompressFn, compress_fn};
define_blake2_dispatch! {
  kernel_id: Blake2bKernelId,
  compress_fn_ty: CompressFn,
  portable_kernel: Blake2bKernelId::Portable,
  compress_fn: compress_fn,
  required_caps: super::kernels::required_caps,
  candidates: [
    #[cfg(target_arch = "x86_64")]
    Blake2bKernelId::X86Avx512vl,
    #[cfg(target_arch = "x86_64")]
    Blake2bKernelId::X86Avx2,
    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    Blake2bKernelId::Aarch64Neon,
    #[cfg(target_arch = "s390x")]
    Blake2bKernelId::S390xVector,
    #[cfg(target_arch = "powerpc64")]
    Blake2bKernelId::PowerVsx,
    #[cfg(target_arch = "riscv64")]
    Blake2bKernelId::Riscv64V,
    #[cfg(target_arch = "wasm32")]
    Blake2bKernelId::WasmSimd128,
  ],
}
