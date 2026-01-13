use platform::Caps;

use super::Sha512;

pub(crate) type CompressBlocksFn = fn(&mut [u64; 8], &[u8]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Sha512KernelId {
  Portable = 0,
}

pub const ALL: &[Sha512KernelId] = &[Sha512KernelId::Portable];

impl Sha512KernelId {
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
pub fn id_from_name(name: &str) -> Option<Sha512KernelId> {
  match name {
    "portable" => Some(Sha512KernelId::Portable),
    _ => None,
  }
}

#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha512KernelId) -> CompressBlocksFn {
  match id {
    Sha512KernelId::Portable => Sha512::compress_blocks_portable,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha512KernelId) -> Caps {
  match id {
    Sha512KernelId::Portable => Caps::NONE,
  }
}
