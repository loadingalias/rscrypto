// SHA-224 uses SHA-256's compression function (FIPS 180-4 SS6.3).
// Only H0 and output truncation differ.
pub(crate) use crate::hashes::crypto::sha256::kernels::CompressBlocksFn;
#[cfg(target_arch = "aarch64")]
use crate::platform::caps::aarch64;
#[cfg(target_arch = "powerpc64")]
use crate::platform::caps::power;
#[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
use crate::platform::caps::riscv;
#[cfg(target_arch = "s390x")]
use crate::platform::caps::s390x;
#[cfg(target_arch = "wasm32")]
use crate::platform::caps::wasm;
#[cfg(target_arch = "x86_64")]
use crate::platform::caps::x86;
use crate::{hashes::crypto::sha256::Sha256, platform::Caps};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Sha224KernelId {
  Portable = 0,
  #[cfg(target_arch = "x86_64")]
  X86Sha = 1,
  #[cfg(target_arch = "aarch64")]
  Aarch64Sha2 = 2,
  #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
  RiscvZknh = 3,
  #[cfg(target_arch = "wasm32")]
  WasmSimd128 = 4,
  #[cfg(target_arch = "s390x")]
  S390xKimd = 5,
  #[cfg(target_arch = "powerpc64")]
  Ppc64Crypto = 6,
}

impl Sha224KernelId {
  #[cfg(any(test, feature = "diag"))]
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "x86_64")]
      Self::X86Sha => "x86-sha",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Sha2 => "aarch64-sha2",
      #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
      Self::RiscvZknh => "riscv/zknh",
      #[cfg(target_arch = "wasm32")]
      Self::WasmSimd128 => "wasm/simd128",
      #[cfg(target_arch = "s390x")]
      Self::S390xKimd => "s390x/kimd",
      #[cfg(target_arch = "powerpc64")]
      Self::Ppc64Crypto => "ppc64/crypto",
    }
  }
}

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<Sha224KernelId> {
  match name {
    "portable" => Some(Sha224KernelId::Portable),
    #[cfg(target_arch = "x86_64")]
    "x86-sha" => Some(Sha224KernelId::X86Sha),
    #[cfg(target_arch = "aarch64")]
    "aarch64-sha2" => Some(Sha224KernelId::Aarch64Sha2),
    #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
    "riscv/zknh" => Some(Sha224KernelId::RiscvZknh),
    #[cfg(target_arch = "wasm32")]
    "wasm/simd128" => Some(Sha224KernelId::WasmSimd128),
    #[cfg(target_arch = "s390x")]
    "s390x/kimd" => Some(Sha224KernelId::S390xKimd),
    #[cfg(target_arch = "powerpc64")]
    "ppc64/crypto" => Some(Sha224KernelId::Ppc64Crypto),
    _ => None,
  }
}

// Delegate to SHA-256 kernels — identical compression function.
#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha224KernelId) -> CompressBlocksFn {
  match id {
    Sha224KernelId::Portable => Sha256::compress_blocks_portable,
    #[cfg(target_arch = "x86_64")]
    Sha224KernelId::X86Sha => crate::hashes::crypto::sha256::kernels::compress_blocks_fn(
      crate::hashes::crypto::sha256::kernels::Sha256KernelId::X86Sha,
    ),
    #[cfg(target_arch = "aarch64")]
    Sha224KernelId::Aarch64Sha2 => crate::hashes::crypto::sha256::kernels::compress_blocks_fn(
      crate::hashes::crypto::sha256::kernels::Sha256KernelId::Aarch64Sha2,
    ),
    #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
    Sha224KernelId::RiscvZknh => crate::hashes::crypto::sha256::kernels::compress_blocks_fn(
      crate::hashes::crypto::sha256::kernels::Sha256KernelId::RiscvZknh,
    ),
    #[cfg(target_arch = "wasm32")]
    Sha224KernelId::WasmSimd128 => crate::hashes::crypto::sha256::kernels::compress_blocks_fn(
      crate::hashes::crypto::sha256::kernels::Sha256KernelId::WasmSimd128,
    ),
    #[cfg(target_arch = "s390x")]
    Sha224KernelId::S390xKimd => crate::hashes::crypto::sha256::kernels::compress_blocks_fn(
      crate::hashes::crypto::sha256::kernels::Sha256KernelId::S390xKimd,
    ),
    #[cfg(target_arch = "powerpc64")]
    Sha224KernelId::Ppc64Crypto => crate::hashes::crypto::sha256::kernels::compress_blocks_fn(
      crate::hashes::crypto::sha256::kernels::Sha256KernelId::Ppc64Crypto,
    ),
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha224KernelId) -> Caps {
  match id {
    Sha224KernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "x86_64")]
    Sha224KernelId::X86Sha => x86::SHA,
    #[cfg(target_arch = "aarch64")]
    Sha224KernelId::Aarch64Sha2 => aarch64::SHA2,
    #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
    Sha224KernelId::RiscvZknh => riscv::ZKNH,
    #[cfg(target_arch = "wasm32")]
    Sha224KernelId::WasmSimd128 => wasm::SIMD128,
    #[cfg(target_arch = "s390x")]
    Sha224KernelId::S390xKimd => s390x::MSA,
    #[cfg(target_arch = "powerpc64")]
    Sha224KernelId::Ppc64Crypto => power::POWER8_CRYPTO,
  }
}
