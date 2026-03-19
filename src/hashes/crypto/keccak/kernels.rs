use super::keccakf_portable;
use crate::platform::Caps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Keccakf1600KernelId {
  Portable = 0,
  #[cfg(target_arch = "aarch64")]
  Aarch64Sha3 = 1,
  #[cfg(target_arch = "x86_64")]
  X86Avx2 = 2,
  #[cfg(target_arch = "x86_64")]
  X86Avx512 = 3,
}

#[cfg(any(test, feature = "std"))]
pub const ALL: &[Keccakf1600KernelId] = &[
  Keccakf1600KernelId::Portable,
  #[cfg(target_arch = "aarch64")]
  Keccakf1600KernelId::Aarch64Sha3,
  #[cfg(target_arch = "x86_64")]
  Keccakf1600KernelId::X86Avx2,
  #[cfg(target_arch = "x86_64")]
  Keccakf1600KernelId::X86Avx512,
];

impl Keccakf1600KernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Sha3 => "aarch64-sha3",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx2 => "x86-avx2",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx512 => "x86-avx512",
    }
  }
}

#[must_use]
pub fn permute_fn(id: Keccakf1600KernelId) -> fn(&mut [u64; 25]) {
  match id {
    Keccakf1600KernelId::Portable => keccakf_portable,
    #[cfg(target_arch = "aarch64")]
    Keccakf1600KernelId::Aarch64Sha3 => super::aarch64::keccakf_aarch64_sha3,
    #[cfg(target_arch = "x86_64")]
    Keccakf1600KernelId::X86Avx2 => super::x86_64_avx2::keccakf_x86_avx2,
    #[cfg(target_arch = "x86_64")]
    Keccakf1600KernelId::X86Avx512 => super::x86_64_avx512::keccakf_x86_avx512,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Keccakf1600KernelId) -> Caps {
  match id {
    Keccakf1600KernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "aarch64")]
    Keccakf1600KernelId::Aarch64Sha3 => crate::platform::caps::aarch64::SHA3,
    #[cfg(target_arch = "x86_64")]
    Keccakf1600KernelId::X86Avx2 => crate::platform::caps::x86::AVX2,
    #[cfg(target_arch = "x86_64")]
    Keccakf1600KernelId::X86Avx512 => crate::platform::caps::x86::AVX512F.union(crate::platform::caps::x86::AVX512VL),
  }
}
