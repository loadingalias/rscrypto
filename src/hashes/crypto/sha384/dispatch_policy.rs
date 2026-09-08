//! Backend selection policy for SHA-384.
//!
//! SHA-384 uses identical compression to SHA-512, so the same hardware kernels
//! and cascade order apply.

pub(crate) use super::kernels::Sha384KernelId as KernelId;
use crate::platform::Caps;

#[inline]
#[must_use]
pub(crate) fn select_runtime_kernel(caps: Caps) -> KernelId {
  // x86_64 cascade: SHA-512 NI > vendor-aware AVX2/AVX-512VL > Portable
  // AMD: AVX2 decoupled > AVX-512VL; Intel: AVX-512VL decoupled > AVX2 decoupled.
  // See sha512/dispatch_policy.rs for full rationale.
  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86;
    if caps.has(x86::SHA512) {
      return KernelId::X86Sha512;
    }
    if caps.has(x86::AMD) {
      // Decoupled kernel: schedule one-ahead of rounds. Eliminates the
      // within-iteration schedule→round dependency that limits IPC on
      // wide pipelines (Zen 5 6-wide). Neutral on Zen 4.
      // See sha512/dispatch_policy.rs for full rationale.
      if caps.has(x86::AVX2) {
        return KernelId::X86Avx2Decoupled;
      }
      if caps.has(x86::AVX512F) && caps.has(x86::AVX512VL) {
        return KernelId::X86Avx512vl;
      }
    } else {
      if caps.has(x86::AVX512F) && caps.has(x86::AVX512VL) {
        return KernelId::X86Avx512vlDecoupled;
      }
      if caps.has(x86::AVX2) {
        return KernelId::X86Avx2Decoupled;
      }
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64;
    if caps.has(aarch64::SHA512) {
      return KernelId::Aarch64Sha512;
    }
  }
  #[cfg(target_arch = "riscv64")]
  {
    use crate::platform::caps::riscv;
    if caps.has(riscv::ZKNH) {
      return KernelId::Riscv64Zknh;
    }
  }
  #[cfg(target_arch = "wasm32")]
  {
    use crate::platform::caps::wasm;
    if caps.has(wasm::SIMD128) {
      return KernelId::WasmSimd128;
    }
  }
  #[cfg(target_arch = "s390x")]
  {
    use crate::platform::caps::s390x;
    if caps.has(s390x::MSA) {
      return KernelId::S390xKimd;
    }
  }
  let _ = caps;
  KernelId::Portable
}
