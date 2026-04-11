// SHA-512/256 uses SHA-512's compression function (FIPS 180-4 SS6.4).
// Only H0 and output truncation differ.
pub(crate) use crate::hashes::crypto::sha512::kernels::CompressBlocksFn;
use crate::{hashes::crypto::sha512::Sha512, platform::Caps};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Sha512_256KernelId {
  Portable = 0,
  #[cfg(target_arch = "aarch64")]
  Aarch64Sha512 = 1,
  #[cfg(target_arch = "x86_64")]
  X86Sha512 = 2,
  #[cfg(target_arch = "x86_64")]
  X86Avx512vl = 5,
  #[cfg(target_arch = "x86_64")]
  X86Avx2 = 6,
  #[cfg(target_arch = "riscv64")]
  Riscv64Zknh = 3,
  #[cfg(target_arch = "wasm32")]
  WasmSimd128 = 4,
  #[cfg(target_arch = "s390x")]
  S390xKimd = 7,
  #[cfg(target_arch = "powerpc64")]
  Ppc64Crypto = 8,
  #[cfg(target_arch = "x86_64")]
  X86Avx2Std = 9,
  #[cfg(target_arch = "x86_64")]
  X86Avx512vlStd = 10,
  #[cfg(target_arch = "x86_64")]
  X86Avx2Decoupled = 11,
  #[cfg(target_arch = "x86_64")]
  X86Avx512vlDecoupled = 12,
}

impl Sha512_256KernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Sha512 => "aarch64-sha512",
      #[cfg(target_arch = "x86_64")]
      Self::X86Sha512 => "x86-sha512",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx512vl => "x86-avx512vl",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx2 => "x86-avx2",
      #[cfg(target_arch = "riscv64")]
      Self::Riscv64Zknh => "riscv/zknh",
      #[cfg(target_arch = "wasm32")]
      Self::WasmSimd128 => "wasm/simd128",
      #[cfg(target_arch = "s390x")]
      Self::S390xKimd => "s390x/kimd",
      #[cfg(target_arch = "powerpc64")]
      Self::Ppc64Crypto => "ppc64/crypto",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx2Std => "x86-avx2-std",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx512vlStd => "x86-avx512vl-std",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx2Decoupled => "x86-avx2-decoupled",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx512vlDecoupled => "x86-avx512vl-decoupled",
    }
  }
}

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<Sha512_256KernelId> {
  match name {
    "portable" => Some(Sha512_256KernelId::Portable),
    #[cfg(target_arch = "aarch64")]
    "aarch64-sha512" => Some(Sha512_256KernelId::Aarch64Sha512),
    #[cfg(target_arch = "x86_64")]
    "x86-sha512" => Some(Sha512_256KernelId::X86Sha512),
    #[cfg(target_arch = "x86_64")]
    "x86-avx512vl" => Some(Sha512_256KernelId::X86Avx512vl),
    #[cfg(target_arch = "x86_64")]
    "x86-avx2" => Some(Sha512_256KernelId::X86Avx2),
    #[cfg(target_arch = "riscv64")]
    "riscv/zknh" => Some(Sha512_256KernelId::Riscv64Zknh),
    #[cfg(target_arch = "wasm32")]
    "wasm/simd128" => Some(Sha512_256KernelId::WasmSimd128),
    #[cfg(target_arch = "s390x")]
    "s390x/kimd" => Some(Sha512_256KernelId::S390xKimd),
    #[cfg(target_arch = "powerpc64")]
    "ppc64/crypto" => Some(Sha512_256KernelId::Ppc64Crypto),
    #[cfg(target_arch = "x86_64")]
    "x86-avx2-std" => Some(Sha512_256KernelId::X86Avx2Std),
    #[cfg(target_arch = "x86_64")]
    "x86-avx512vl-std" => Some(Sha512_256KernelId::X86Avx512vlStd),
    #[cfg(target_arch = "x86_64")]
    "x86-avx2-decoupled" => Some(Sha512_256KernelId::X86Avx2Decoupled),
    #[cfg(target_arch = "x86_64")]
    "x86-avx512vl-decoupled" => Some(Sha512_256KernelId::X86Avx512vlDecoupled),
    _ => None,
  }
}

/// Maps each SHA-512/256 kernel ID to the corresponding SHA-512 kernel ID.
/// SHA-512/256 uses identical compression to SHA-512.
const fn to_sha512_kernel_id(id: Sha512_256KernelId) -> crate::hashes::crypto::sha512::kernels::Sha512KernelId {
  use crate::hashes::crypto::sha512::kernels::Sha512KernelId;
  match id {
    Sha512_256KernelId::Portable => Sha512KernelId::Portable,
    #[cfg(target_arch = "aarch64")]
    Sha512_256KernelId::Aarch64Sha512 => Sha512KernelId::Aarch64Sha512,
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Sha512 => Sha512KernelId::X86Sha512,
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx512vl => Sha512KernelId::X86Avx512vl,
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx2 => Sha512KernelId::X86Avx2,
    #[cfg(target_arch = "riscv64")]
    Sha512_256KernelId::Riscv64Zknh => Sha512KernelId::Riscv64Zknh,
    #[cfg(target_arch = "wasm32")]
    Sha512_256KernelId::WasmSimd128 => Sha512KernelId::WasmSimd128,
    #[cfg(target_arch = "s390x")]
    Sha512_256KernelId::S390xKimd => Sha512KernelId::S390xKimd,
    #[cfg(target_arch = "powerpc64")]
    Sha512_256KernelId::Ppc64Crypto => Sha512KernelId::Ppc64Crypto,
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx2Std => Sha512KernelId::X86Avx2Std,
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx512vlStd => Sha512KernelId::X86Avx512vlStd,
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx2Decoupled => Sha512KernelId::X86Avx2Decoupled,
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx512vlDecoupled => Sha512KernelId::X86Avx512vlDecoupled,
  }
}

// Delegate to SHA-512 kernels — identical compression function.
#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha512_256KernelId) -> CompressBlocksFn {
  match id {
    Sha512_256KernelId::Portable => Sha512::compress_blocks_portable,
    #[cfg(target_arch = "aarch64")]
    Sha512_256KernelId::Aarch64Sha512 => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Sha512 => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx512vl => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx2 => crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id)),
    #[cfg(target_arch = "riscv64")]
    Sha512_256KernelId::Riscv64Zknh => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
    #[cfg(target_arch = "wasm32")]
    Sha512_256KernelId::WasmSimd128 => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
    #[cfg(target_arch = "s390x")]
    Sha512_256KernelId::S390xKimd => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
    #[cfg(target_arch = "powerpc64")]
    Sha512_256KernelId::Ppc64Crypto => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx2Std => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx512vlStd => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx2Decoupled => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
    #[cfg(target_arch = "x86_64")]
    Sha512_256KernelId::X86Avx512vlDecoupled => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha512_256KernelId) -> Caps {
  crate::hashes::crypto::sha512::kernels::required_caps(to_sha512_kernel_id(id))
}

// Keep kernel tests focused on backends that runtime dispatch can actually pick.
#[cfg(test)]
pub const ALL: &[Sha512_256KernelId] = &[
  Sha512_256KernelId::Portable,
  #[cfg(target_arch = "aarch64")]
  Sha512_256KernelId::Aarch64Sha512,
  #[cfg(target_arch = "x86_64")]
  Sha512_256KernelId::X86Sha512,
  #[cfg(target_arch = "x86_64")]
  Sha512_256KernelId::X86Avx512vl,
  #[cfg(target_arch = "x86_64")]
  Sha512_256KernelId::X86Avx2,
  #[cfg(target_arch = "riscv64")]
  Sha512_256KernelId::Riscv64Zknh,
  #[cfg(target_arch = "wasm32")]
  Sha512_256KernelId::WasmSimd128,
  #[cfg(target_arch = "s390x")]
  Sha512_256KernelId::S390xKimd,
  #[cfg(target_arch = "x86_64")]
  Sha512_256KernelId::X86Avx2Decoupled,
  #[cfg(target_arch = "x86_64")]
  Sha512_256KernelId::X86Avx512vlDecoupled,
];
