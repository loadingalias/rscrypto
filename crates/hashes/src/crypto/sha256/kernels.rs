use platform::Caps;

use super::Sha256;

pub(crate) type CompressBlocksFn = fn(&mut [u32; 8], &[u8]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Sha256KernelId {
  Portable = 0,
}

pub const ALL: &[Sha256KernelId] = &[Sha256KernelId::Portable];

impl Sha256KernelId {
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
pub fn id_from_name(name: &str) -> Option<Sha256KernelId> {
  match name {
    "portable" => Some(Sha256KernelId::Portable),
    _ => None,
  }
}

#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha256KernelId) -> CompressBlocksFn {
  match id {
    Sha256KernelId::Portable => Sha256::compress_blocks_portable,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha256KernelId) -> Caps {
  match id {
    Sha256KernelId::Portable => Caps::NONE,
  }
}
