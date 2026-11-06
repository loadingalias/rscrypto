use super::keccakf_portable;
use crate::platform::Caps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub(crate) enum Keccakf1600KernelId {
  Portable = 0,
  /// Lazy lane rotations: the non-Apple AArch64 single-state kernel, compiled
  /// on every target for tests.
  #[cfg(test)]
  LazyRotation = 1,
  /// The portable rounds compiled with BMI1 and BMI2.
  #[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
  X86Bmi = 2,
  /// The AVX-512 single-state kernel.
  #[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
  X86Avx512 = 3,
}

impl Keccakf1600KernelId {
  #[inline]
  #[must_use]
  pub(crate) const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(test)]
      Self::LazyRotation => "lazy-rotation",
      #[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
      Self::X86Bmi => "x86-bmi",
      #[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
      Self::X86Avx512 => "x86-avx512",
    }
  }
}

#[must_use]
pub(crate) fn permute_fn(id: Keccakf1600KernelId) -> fn(&mut [u64; 25]) {
  match id {
    Keccakf1600KernelId::Portable => keccakf_portable,
    #[cfg(test)]
    Keccakf1600KernelId::LazyRotation => super::lazy_rotation::keccakf,
    #[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
    Keccakf1600KernelId::X86Bmi => x86_bmi,
    #[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
    Keccakf1600KernelId::X86Avx512 => x86_avx512,
  }
}

/// Safe entry for [`super::x86_64::keccakf_x86_bmi`].
#[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
fn x86_bmi(state: &mut [u64; 25]) {
  assert!(
    crate::platform::caps().has(required_caps(Keccakf1600KernelId::X86Bmi)),
    "x86-bmi kernel requires BMI1 and BMI2"
  );
  // SAFETY: the assertion above established BMI1 and BMI2.
  unsafe { super::x86_64::keccakf_x86_bmi(state) }
}

/// Safe entry for [`super::x86_64::keccakf_x86_avx512_single`].
#[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
fn x86_avx512(state: &mut [u64; 25]) {
  assert!(
    crate::platform::caps().has(required_caps(Keccakf1600KernelId::X86Avx512)),
    "x86-avx512 kernel requires AVX512F"
  );
  // SAFETY: the assertion above established AVX512F.
  unsafe { super::x86_64::keccakf_x86_avx512_single(state) }
}

/// All kernel IDs for agreement testing.
#[cfg(test)]
pub(crate) const ALL: &[Keccakf1600KernelId] = &[
  Keccakf1600KernelId::Portable,
  Keccakf1600KernelId::LazyRotation,
  #[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
  Keccakf1600KernelId::X86Bmi,
  #[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
  Keccakf1600KernelId::X86Avx512,
];

#[inline]
#[must_use]
pub(crate) const fn required_caps(id: Keccakf1600KernelId) -> Caps {
  match id {
    Keccakf1600KernelId::Portable => Caps::NONE,
    #[cfg(test)]
    Keccakf1600KernelId::LazyRotation => Caps::NONE,
    #[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
    Keccakf1600KernelId::X86Bmi => crate::platform::caps::x86::BMI1.union(crate::platform::caps::x86::BMI2),
    #[cfg(all(test, target_arch = "x86_64", target_feature = "sse2", not(miri)))]
    Keccakf1600KernelId::X86Avx512 => crate::platform::caps::x86::AVX512F,
  }
}
