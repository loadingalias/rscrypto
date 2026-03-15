use super::keccakf_portable;
use crate::platform::Caps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Keccakf1600KernelId {
  Portable = 0,
}

#[cfg(any(test, feature = "std"))]
pub const ALL: &[Keccakf1600KernelId] = &[Keccakf1600KernelId::Portable];

impl Keccakf1600KernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
    }
  }
}

#[must_use]
pub fn permute_fn(id: Keccakf1600KernelId) -> fn(&mut [u64; 25]) {
  match id {
    Keccakf1600KernelId::Portable => keccakf_portable,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Keccakf1600KernelId) -> Caps {
  match id {
    Keccakf1600KernelId::Portable => Caps::NONE,
  }
}
