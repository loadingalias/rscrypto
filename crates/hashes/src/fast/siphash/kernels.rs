use platform::Caps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SipHashKernelId {
  Portable = 0,
}

impl SipHashKernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
    }
  }
}

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<SipHashKernelId> {
  match name {
    "portable" => Some(SipHashKernelId::Portable),
    _ => None,
  }
}

#[must_use]
pub fn hash13_fn(id: SipHashKernelId) -> fn([u64; 2], &[u8]) -> u64 {
  match id {
    SipHashKernelId::Portable => super::siphash13,
  }
}

#[must_use]
pub fn hash24_fn(id: SipHashKernelId) -> fn([u64; 2], &[u8]) -> u64 {
  match id {
    SipHashKernelId::Portable => super::siphash24,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: SipHashKernelId) -> Caps {
  match id {
    SipHashKernelId::Portable => Caps::NONE,
  }
}
