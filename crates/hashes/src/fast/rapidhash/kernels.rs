use platform::Caps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
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

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<RapidHashKernelId> {
  match name {
    "portable" => Some(RapidHashKernelId::Portable),
    _ => None,
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
