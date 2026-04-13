use crate::platform::Caps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Xxh3KernelId {
  Portable = 0,
  #[cfg(target_arch = "x86_64")]
  Avx2 = 1,
  #[cfg(target_arch = "aarch64")]
  Neon = 2,
  #[cfg(target_arch = "x86_64")]
  Avx512 = 3,
  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  Vsx = 4,
  #[cfg(target_arch = "s390x")]
  Vector = 5,
  #[cfg(target_arch = "riscv64")]
  Rvv = 6,
}

impl Xxh3KernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "x86_64")]
      Self::Avx2 => "avx2",
      #[cfg(target_arch = "aarch64")]
      Self::Neon => "neon",
      #[cfg(target_arch = "x86_64")]
      Self::Avx512 => "avx512",
      #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
      Self::Vsx => "vsx",
      #[cfg(target_arch = "s390x")]
      Self::Vector => "zvector",
      #[cfg(target_arch = "riscv64")]
      Self::Rvv => "rvv",
    }
  }
}

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<Xxh3KernelId> {
  match name {
    "portable" => Some(Xxh3KernelId::Portable),
    #[cfg(target_arch = "x86_64")]
    "avx2" => Some(Xxh3KernelId::Avx2),
    #[cfg(target_arch = "aarch64")]
    "neon" => Some(Xxh3KernelId::Neon),
    #[cfg(target_arch = "x86_64")]
    "avx512" => Some(Xxh3KernelId::Avx512),
    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    "vsx" => Some(Xxh3KernelId::Vsx),
    #[cfg(target_arch = "s390x")]
    "zvector" => Some(Xxh3KernelId::Vector),
    #[cfg(target_arch = "riscv64")]
    "rvv" => Some(Xxh3KernelId::Rvv),
    _ => None,
  }
}

#[must_use]
pub fn hash64_fn(id: Xxh3KernelId) -> fn(&[u8], u64) -> u64 {
  match id {
    Xxh3KernelId::Portable => super::xxh3_64_with_seed,
    #[cfg(target_arch = "x86_64")]
    Xxh3KernelId::Avx2 => super::x86_64_avx2::xxh3_64_with_seed,
    #[cfg(target_arch = "aarch64")]
    Xxh3KernelId::Neon => super::aarch64_neon::xxh3_64_with_seed,
    #[cfg(target_arch = "x86_64")]
    Xxh3KernelId::Avx512 => super::x86_64_avx512::xxh3_64_with_seed,
    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    Xxh3KernelId::Vsx => super::power::xxh3_64_with_seed,
    #[cfg(target_arch = "s390x")]
    Xxh3KernelId::Vector => super::s390x::xxh3_64_with_seed,
    #[cfg(target_arch = "riscv64")]
    Xxh3KernelId::Rvv => super::riscv64_v::xxh3_64_with_seed,
  }
}

/// Long-path-only entry for 64-bit hash (>240B, no ≤240B length checks).
#[must_use]
pub fn hash64_long_fn(id: Xxh3KernelId) -> fn(&[u8], u64) -> u64 {
  match id {
    Xxh3KernelId::Portable => super::xxh3_64_long,
    #[cfg(target_arch = "x86_64")]
    Xxh3KernelId::Avx2 => super::x86_64_avx2::xxh3_64_long,
    #[cfg(target_arch = "aarch64")]
    Xxh3KernelId::Neon => super::aarch64_neon::xxh3_64_long,
    #[cfg(target_arch = "x86_64")]
    Xxh3KernelId::Avx512 => super::x86_64_avx512::xxh3_64_long,
    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    Xxh3KernelId::Vsx => super::power::xxh3_64_long,
    #[cfg(target_arch = "s390x")]
    Xxh3KernelId::Vector => super::s390x::xxh3_64_long,
    #[cfg(target_arch = "riscv64")]
    Xxh3KernelId::Rvv => super::riscv64_v::xxh3_64_long,
  }
}

/// Long-path-only entry for 128-bit hash (>240B, no ≤240B length checks).
#[must_use]
pub fn hash128_long_fn(id: Xxh3KernelId) -> fn(&[u8], u64) -> u128 {
  match id {
    Xxh3KernelId::Portable => super::xxh3_128_long,
    #[cfg(target_arch = "x86_64")]
    Xxh3KernelId::Avx2 => super::x86_64_avx2::xxh3_128_long,
    #[cfg(target_arch = "aarch64")]
    Xxh3KernelId::Neon => super::aarch64_neon::xxh3_128_long,
    #[cfg(target_arch = "x86_64")]
    Xxh3KernelId::Avx512 => super::x86_64_avx512::xxh3_128_long,
    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    Xxh3KernelId::Vsx => super::power::xxh3_128_long,
    #[cfg(target_arch = "s390x")]
    Xxh3KernelId::Vector => super::s390x::xxh3_128_long,
    #[cfg(target_arch = "riscv64")]
    Xxh3KernelId::Rvv => super::riscv64_v::xxh3_128_long,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Xxh3KernelId) -> Caps {
  match id {
    Xxh3KernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "x86_64")]
    Xxh3KernelId::Avx2 => crate::platform::caps::x86::AVX2,
    #[cfg(target_arch = "aarch64")]
    Xxh3KernelId::Neon => crate::platform::caps::aarch64::NEON,
    #[cfg(target_arch = "x86_64")]
    Xxh3KernelId::Avx512 => crate::platform::caps::x86::AVX512F,
    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    Xxh3KernelId::Vsx => crate::platform::caps::power::POWER8_VECTOR,
    #[cfg(target_arch = "s390x")]
    Xxh3KernelId::Vector => crate::platform::caps::s390x::VECTOR,
    #[cfg(target_arch = "riscv64")]
    Xxh3KernelId::Rvv => crate::platform::caps::riscv::V,
  }
}
