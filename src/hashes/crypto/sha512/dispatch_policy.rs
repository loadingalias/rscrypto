//! Backend selection policy for SHA-512.
//!
//! SHA-512 NI, ARM SHA512 CE, and Zknh have negligible setup cost — use HW
//! accel when available.

pub(crate) use super::kernels::Sha512KernelId as KernelId;
use crate::platform::Caps;

#[inline]
#[must_use]
pub(crate) fn select_runtime_kernel(caps: Caps) -> KernelId {
  // x86_64 cascade: SHA-512 NI, then vendor-aware AVX2/AVX-512VL, then portable.
  //
  // Both AVX2 and AVX-512VL handle a trailing single block inside their SIMD
  // kernels. Vendor-specific ordering between the implementations remains a
  // manually maintained dispatch policy.
  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86;
    if caps.has(x86::SHA512) {
      return KernelId::X86Sha512;
    }
    if caps.has(x86::AMD) {
      // Decoupled kernel: schedule one-ahead of rounds. The stitched kernel
      // serialises schedule → extract → round within each iteration, limiting
      // IPC on wide pipelines. The decoupled pattern gives the OOO engine
      // 16 independent scalar rounds to overlap with SIMD schedule latency.
      if caps.has(x86::AVX2) {
        return KernelId::X86Avx2Decoupled;
      }
      if caps.has(x86::AVX512F) && caps.has(x86::AVX512VL) {
        return KernelId::X86Avx512vl;
      }
    } else {
      // Intel: decoupled AVX-512VL > stitched AVX-512VL > AVX2.
      // The decoupled kernel uses rotation-based schedule (no cross-lane
      // permute) + VPRORQ native rotates + schedule one-ahead of rounds,
      // eliminating the `_mm256_permute2x128_si256` bottleneck (3-cycle
      // latency on SPR) that caused 0.95-0.96x vs sha2 at ≥1KiB.
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
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
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
