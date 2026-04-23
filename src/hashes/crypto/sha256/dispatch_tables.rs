//! Tuned dispatch tables for SHA-256.
//!
//! SHA-NI, ARM SHA2 CE, KIMD, and POWER `vshasigmaw` have negligible setup cost
//! relative to the block work, so use HW accel for all size classes when
//! available.

pub use super::kernels::Sha256KernelId as KernelId;
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

#[cfg(target_arch = "x86_64")]
pub static X86_SHA_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::X86Sha,
  s: KernelId::X86Sha,
  m: KernelId::X86Sha,
  l: KernelId::X86Sha,
};

#[cfg(target_arch = "aarch64")]
pub static AARCH64_SHA2_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Sha2,
  s: KernelId::Aarch64Sha2,
  m: KernelId::Aarch64Sha2,
  l: KernelId::Aarch64Sha2,
};

#[cfg(target_arch = "powerpc64")]
pub static PPC64_CRYPTO_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Ppc64Crypto,
  s: KernelId::Ppc64Crypto,
  m: KernelId::Ppc64Crypto,
  l: KernelId::Ppc64Crypto,
};

#[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
pub static RISCV_ZKNH_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::RiscvZknh,
  s: KernelId::RiscvZknh,
  m: KernelId::RiscvZknh,
  l: KernelId::RiscvZknh,
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
  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86;
    if caps.has(x86::SHA) {
      return &X86_SHA_TABLE;
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64;
    if caps.has(aarch64::SHA2) {
      return &AARCH64_SHA2_TABLE;
    }
  }
  #[cfg(target_arch = "powerpc64")]
  {
    use crate::platform::caps::power;
    if caps.has(power::POWER8_CRYPTO) {
      return &PPC64_CRYPTO_TABLE;
    }
  }
  #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
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
