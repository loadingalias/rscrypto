use super::permute_12_portable;
use crate::platform::Caps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum AsconPermute12KernelId {
  Portable = 0,
  #[cfg(target_arch = "aarch64")]
  Aarch64Neon = 1,
  #[cfg(target_arch = "x86_64")]
  X86Avx2 = 2,
  #[cfg(target_arch = "x86_64")]
  X86Avx512 = 3,
}

#[cfg(any(test, feature = "std"))]
pub const ALL: &[AsconPermute12KernelId] = &[
  AsconPermute12KernelId::Portable,
  #[cfg(target_arch = "aarch64")]
  AsconPermute12KernelId::Aarch64Neon,
  #[cfg(target_arch = "x86_64")]
  AsconPermute12KernelId::X86Avx2,
  #[cfg(target_arch = "x86_64")]
  AsconPermute12KernelId::X86Avx512,
];

impl AsconPermute12KernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Neon => "aarch64-neon",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx2 => "x86-avx2",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx512 => "x86-avx512",
    }
  }
}

#[inline]
#[must_use]
pub const fn simd_degree(id: AsconPermute12KernelId) -> usize {
  match id {
    AsconPermute12KernelId::Portable => 1,
    #[cfg(target_arch = "aarch64")]
    AsconPermute12KernelId::Aarch64Neon => 2,
    #[cfg(target_arch = "x86_64")]
    AsconPermute12KernelId::X86Avx2 => 4,
    #[cfg(target_arch = "x86_64")]
    AsconPermute12KernelId::X86Avx512 => 8,
  }
}

#[must_use]
pub fn permute_fn(id: AsconPermute12KernelId) -> fn(&mut [u64; 5]) {
  match id {
    AsconPermute12KernelId::Portable => permute_12_portable,
    #[cfg(target_arch = "aarch64")]
    AsconPermute12KernelId::Aarch64Neon => super::aarch64::permute_12_aarch64_neon,
    #[cfg(target_arch = "x86_64")]
    AsconPermute12KernelId::X86Avx2 => super::x86_64_avx2::permute_12_x86_avx2,
    #[cfg(target_arch = "x86_64")]
    AsconPermute12KernelId::X86Avx512 => super::x86_64_avx512::permute_12_x86_avx512,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: AsconPermute12KernelId) -> Caps {
  match id {
    AsconPermute12KernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "aarch64")]
    AsconPermute12KernelId::Aarch64Neon => crate::platform::caps::aarch64::NEON,
    #[cfg(target_arch = "x86_64")]
    AsconPermute12KernelId::X86Avx2 => crate::platform::caps::x86::AVX2,
    #[cfg(target_arch = "x86_64")]
    AsconPermute12KernelId::X86Avx512 => {
      crate::platform::caps::x86::AVX512F.union(crate::platform::caps::x86::AVX512VL)
    }
  }
}
