// SHA-384 uses SHA-512's compression function (FIPS 180-4 SS6.4).
// Only H0 and output truncation differ.
pub(crate) use crate::hashes::crypto::sha512::kernels::CompressBlocksFn;
use crate::{hashes::crypto::sha512::Sha512, platform::Caps};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Sha384KernelId {
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
}

#[cfg(any(test, feature = "std"))]
pub const ALL: &[Sha384KernelId] = &[
  Sha384KernelId::Portable,
  #[cfg(target_arch = "aarch64")]
  Sha384KernelId::Aarch64Sha512,
  #[cfg(target_arch = "x86_64")]
  Sha384KernelId::X86Sha512,
  #[cfg(target_arch = "x86_64")]
  Sha384KernelId::X86Avx512vl,
  #[cfg(target_arch = "x86_64")]
  Sha384KernelId::X86Avx2,
  #[cfg(target_arch = "riscv64")]
  Sha384KernelId::Riscv64Zknh,
  #[cfg(target_arch = "wasm32")]
  Sha384KernelId::WasmSimd128,
];

impl Sha384KernelId {
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
    }
  }
}

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<Sha384KernelId> {
  match name {
    "portable" => Some(Sha384KernelId::Portable),
    #[cfg(target_arch = "aarch64")]
    "aarch64-sha512" => Some(Sha384KernelId::Aarch64Sha512),
    #[cfg(target_arch = "x86_64")]
    "x86-sha512" => Some(Sha384KernelId::X86Sha512),
    #[cfg(target_arch = "x86_64")]
    "x86-avx512vl" => Some(Sha384KernelId::X86Avx512vl),
    #[cfg(target_arch = "x86_64")]
    "x86-avx2" => Some(Sha384KernelId::X86Avx2),
    #[cfg(target_arch = "riscv64")]
    "riscv/zknh" => Some(Sha384KernelId::Riscv64Zknh),
    #[cfg(target_arch = "wasm32")]
    "wasm/simd128" => Some(Sha384KernelId::WasmSimd128),
    _ => None,
  }
}

/// Maps each SHA-384 kernel ID to the corresponding SHA-512 kernel ID.
/// SHA-384 uses identical compression to SHA-512.
const fn to_sha512_kernel_id(id: Sha384KernelId) -> crate::hashes::crypto::sha512::kernels::Sha512KernelId {
  use crate::hashes::crypto::sha512::kernels::Sha512KernelId;
  match id {
    Sha384KernelId::Portable => Sha512KernelId::Portable,
    #[cfg(target_arch = "aarch64")]
    Sha384KernelId::Aarch64Sha512 => Sha512KernelId::Aarch64Sha512,
    #[cfg(target_arch = "x86_64")]
    Sha384KernelId::X86Sha512 => Sha512KernelId::X86Sha512,
    #[cfg(target_arch = "x86_64")]
    Sha384KernelId::X86Avx512vl => Sha512KernelId::X86Avx512vl,
    #[cfg(target_arch = "x86_64")]
    Sha384KernelId::X86Avx2 => Sha512KernelId::X86Avx2,
    #[cfg(target_arch = "riscv64")]
    Sha384KernelId::Riscv64Zknh => Sha512KernelId::Riscv64Zknh,
    #[cfg(target_arch = "wasm32")]
    Sha384KernelId::WasmSimd128 => Sha512KernelId::WasmSimd128,
  }
}

// Delegate to SHA-512 kernels — identical compression function.
#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha384KernelId) -> CompressBlocksFn {
  match id {
    Sha384KernelId::Portable => Sha512::compress_blocks_portable,
    #[cfg(target_arch = "aarch64")]
    Sha384KernelId::Aarch64Sha512 => {
      crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id))
    }
    #[cfg(target_arch = "x86_64")]
    Sha384KernelId::X86Sha512 => crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id)),
    #[cfg(target_arch = "x86_64")]
    Sha384KernelId::X86Avx512vl => crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id)),
    #[cfg(target_arch = "x86_64")]
    Sha384KernelId::X86Avx2 => crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id)),
    #[cfg(target_arch = "riscv64")]
    Sha384KernelId::Riscv64Zknh => crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id)),
    #[cfg(target_arch = "wasm32")]
    Sha384KernelId::WasmSimd128 => crate::hashes::crypto::sha512::kernels::compress_blocks_fn(to_sha512_kernel_id(id)),
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha384KernelId) -> Caps {
  crate::hashes::crypto::sha512::kernels::required_caps(to_sha512_kernel_id(id))
}
