use platform::Caps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Xxh3KernelId {
  Portable = 0,
}

impl Xxh3KernelId {
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
pub fn id_from_name(name: &str) -> Option<Xxh3KernelId> {
  match name {
    "portable" => Some(Xxh3KernelId::Portable),
    _ => None,
  }
}

#[must_use]
pub fn hash64_fn(id: Xxh3KernelId) -> fn(&[u8], u64) -> u64 {
  match id {
    Xxh3KernelId::Portable => super::xxh3_64_with_seed,
  }
}

#[must_use]
pub fn hash128_fn(id: Xxh3KernelId) -> fn(&[u8], u64) -> u128 {
  match id {
    Xxh3KernelId::Portable => super::xxh3_128_with_seed,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Xxh3KernelId) -> Caps {
  match id {
    Xxh3KernelId::Portable => Caps::NONE,
  }
}
