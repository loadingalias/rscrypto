//! Backend selection policy for SHA-256.
//!
//! SHA-NI, ARM SHA2 CE, and KIMD have negligible setup cost relative to the
//! block work, so use HW accel when available.
//!
//! POWER uses portable compression because the retired hybrid `vshasigmaw`
//! kernel lost to portable Rust on POWER10.

pub(crate) use super::kernels::Sha256KernelId as KernelId;
use crate::platform::Caps;

#[inline]
#[must_use]
pub(crate) fn select_runtime_kernel(caps: Caps) -> KernelId {
  #[cfg(target_arch = "x86_64")]
  {
    if caps.has(super::kernels::required_caps(KernelId::X86Sha)) {
      return KernelId::X86Sha;
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    if caps.has(super::kernels::required_caps(KernelId::Aarch64Sha2)) {
      return KernelId::Aarch64Sha2;
    }
  }
  #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
  {
    if caps.has(super::kernels::required_caps(KernelId::RiscvZknh)) {
      return KernelId::RiscvZknh;
    }
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    if caps.has(super::kernels::required_caps(KernelId::WasmSimd128)) {
      return KernelId::WasmSimd128;
    }
  }
  #[cfg(target_arch = "s390x")]
  {
    if caps.has(super::kernels::required_caps(KernelId::S390xKimd)) {
      return KernelId::S390xKimd;
    }
  }
  let _ = caps;
  KernelId::Portable
}
