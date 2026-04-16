use crate::platform::Caps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum RapidHashKernelId {
  Portable = 0,
}

impl RapidHashKernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
    }
  }
}

#[must_use]
pub fn hash64_fn(id: RapidHashKernelId) -> fn(&[u8], u64) -> u64 {
  match id {
    RapidHashKernelId::Portable => super::rapidhash_v3_with_seed,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: RapidHashKernelId) -> Caps {
  match id {
    RapidHashKernelId::Portable => Caps::NONE,
  }
}
