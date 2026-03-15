use super::permute_12_portable;
use crate::platform::Caps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum AsconPermute12KernelId {
  Portable = 0,
}

#[cfg(any(test, feature = "std"))]
pub const ALL: &[AsconPermute12KernelId] = &[AsconPermute12KernelId::Portable];

impl AsconPermute12KernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
    }
  }
}

#[must_use]
pub fn permute_fn(id: AsconPermute12KernelId) -> fn(&mut [u64; 5]) {
  match id {
    AsconPermute12KernelId::Portable => permute_12_portable,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: AsconPermute12KernelId) -> Caps {
  match id {
    AsconPermute12KernelId::Portable => Caps::NONE,
  }
}
