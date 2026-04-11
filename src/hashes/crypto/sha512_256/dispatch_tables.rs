//! Tuned dispatch tables for SHA-512/256.
//!
//! SHA-512/256 uses identical compression to SHA-512, so the same hardware kernels
//! and cascade order apply.

pub use super::kernels::Sha512_256KernelId as KernelId;
use crate::platform::Caps;

pub const DEFAULT_BOUNDARIES: [usize; 3] = [64, 256, 4096];

#[derive(Clone, Copy, Debug)]
pub struct DispatchTable {
  pub boundaries: [usize; 3],
  pub xs: KernelId,
  pub s: KernelId,
  pub m: KernelId,
  pub l: KernelId,
}

pub static DEFAULT_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Portable,
  l: KernelId::Portable,
};

#[cfg(target_arch = "aarch64")]
pub static AARCH64_SHA512_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Sha512,
  s: KernelId::Aarch64Sha512,
  m: KernelId::Aarch64Sha512,
  l: KernelId::Aarch64Sha512,
};

#[cfg(target_arch = "x86_64")]
pub static X86_SHA512_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::X86Sha512,
  s: KernelId::X86Sha512,
  m: KernelId::X86Sha512,
  l: KernelId::X86Sha512,
};

#[cfg(target_arch = "x86_64")]
pub static X86_AVX512VL_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::X86Avx512vl,
  s: KernelId::X86Avx512vl,
  m: KernelId::X86Avx512vl,
  l: KernelId::X86Avx512vl,
};

#[cfg(target_arch = "x86_64")]
pub static X86_AVX2_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::X86Avx2,
  s: KernelId::X86Avx2,
  m: KernelId::X86Avx2,
  l: KernelId::X86Avx2,
};

#[cfg(target_arch = "x86_64")]
pub static X86_AVX2_DECOUPLED_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::X86Avx2Decoupled,
  s: KernelId::X86Avx2Decoupled,
  m: KernelId::X86Avx2Decoupled,
  l: KernelId::X86Avx2Decoupled,
};

#[cfg(target_arch = "x86_64")]
pub static X86_AVX512VL_DECOUPLED_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::X86Avx512vlDecoupled,
  s: KernelId::X86Avx512vlDecoupled,
  m: KernelId::X86Avx512vlDecoupled,
  l: KernelId::X86Avx512vlDecoupled,
};

#[cfg(target_arch = "riscv64")]
pub static RISCV_ZKNH_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Riscv64Zknh,
  s: KernelId::Riscv64Zknh,
  m: KernelId::Riscv64Zknh,
  l: KernelId::Riscv64Zknh,
};

#[cfg(target_arch = "wasm32")]
pub static WASM_SIMD128_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::WasmSimd128,
  s: KernelId::WasmSimd128,
  m: KernelId::WasmSimd128,
  l: KernelId::WasmSimd128,
};

#[cfg(target_arch = "s390x")]
pub static S390X_KIMD_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::S390xKimd,
  s: KernelId::S390xKimd,
  m: KernelId::S390xKimd,
  l: KernelId::S390xKimd,
};

#[inline]
#[must_use]
pub fn select_runtime_table(#[allow(unused_variables)] caps: Caps) -> &'static DispatchTable {
  // x86_64 cascade: SHA-512 NI > vendor-aware AVX2/AVX-512VL > Portable
  // AMD: AVX2 > AVX-512VL; Intel: AVX-512VL > AVX2.
  // See sha512/dispatch_tables.rs for full rationale.
  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86;
    if caps.has(x86::SHA512) {
      return &X86_SHA512_TABLE;
    }
    if caps.has(x86::AMD) {
      // Decoupled kernel: schedule one-ahead of rounds. Eliminates the
      // within-iteration schedule→round dependency that limits IPC on
      // wide pipelines (Zen 5 6-wide). Neutral on Zen 4.
      // See sha512/dispatch_tables.rs for full rationale.
      if caps.has(x86::AVX2) {
        return &X86_AVX2_DECOUPLED_TABLE;
      }
      if caps.has(x86::AVX512F) && caps.has(x86::AVX512VL) {
        return &X86_AVX512VL_TABLE;
      }
    } else {
      if caps.has(x86::AVX512F) && caps.has(x86::AVX512VL) {
        return &X86_AVX512VL_DECOUPLED_TABLE;
      }
      if caps.has(x86::AVX2) {
        return &X86_AVX2_TABLE;
      }
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64;
    if caps.has(aarch64::SHA512) {
      return &AARCH64_SHA512_TABLE;
    }
  }
  #[cfg(target_arch = "riscv64")]
  {
    use crate::platform::caps::riscv;
    if caps.has(riscv::ZKNH) {
      return &RISCV_ZKNH_TABLE;
    }
  }
  #[cfg(target_arch = "wasm32")]
  {
    use crate::platform::caps::wasm;
    if caps.has(wasm::SIMD128) {
      return &WASM_SIMD128_TABLE;
    }
  }
  #[cfg(target_arch = "s390x")]
  {
    use crate::platform::caps::s390x;
    if caps.has(s390x::MSA) {
      return &S390X_KIMD_TABLE;
    }
  }
  &DEFAULT_TABLE
}
