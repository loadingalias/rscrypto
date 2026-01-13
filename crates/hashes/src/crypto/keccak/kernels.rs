use platform::Caps;

use super::keccakf_portable;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Keccakf1600KernelId {
  Portable = 0,
}

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

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<Keccakf1600KernelId> {
  match name {
    "portable" => Some(Keccakf1600KernelId::Portable),
    _ => None,
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
