use platform::Caps;

use super::permute_12_portable;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AsconPermute12KernelId {
  Portable = 0,
}

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

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<AsconPermute12KernelId> {
  match name {
    "portable" => Some(AsconPermute12KernelId::Portable),
    _ => None,
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
