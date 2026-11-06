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
}

impl Keccakf1600KernelId {
  #[inline]
  #[must_use]
  pub(crate) const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(test)]
      Self::LazyRotation => "lazy-rotation",
    }
  }
}

#[must_use]
pub(crate) fn permute_fn(id: Keccakf1600KernelId) -> fn(&mut [u64; 25]) {
  match id {
    Keccakf1600KernelId::Portable => keccakf_portable,
    #[cfg(test)]
    Keccakf1600KernelId::LazyRotation => super::lazy_rotation::keccakf,
  }
}

/// All kernel IDs for agreement testing.
#[cfg(test)]
pub(crate) const ALL: &[Keccakf1600KernelId] = &[Keccakf1600KernelId::Portable, Keccakf1600KernelId::LazyRotation];

#[inline]
#[must_use]
pub(crate) const fn required_caps(id: Keccakf1600KernelId) -> Caps {
  match id {
    Keccakf1600KernelId::Portable => Caps::NONE,
    #[cfg(test)]
    Keccakf1600KernelId::LazyRotation => Caps::NONE,
  }
}
