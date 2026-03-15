use super::Sha512_256;
use crate::platform::Caps;

pub(crate) type CompressBlocksFn = fn(&mut [u64; 8], &[u8]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Sha512_256KernelId {
  Portable = 0,
}

#[cfg(any(test, feature = "std"))]
pub const ALL: &[Sha512_256KernelId] = &[Sha512_256KernelId::Portable];

impl Sha512_256KernelId {
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
pub fn id_from_name(name: &str) -> Option<Sha512_256KernelId> {
  match name {
    "portable" => Some(Sha512_256KernelId::Portable),
    _ => None,
  }
}

#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha512_256KernelId) -> CompressBlocksFn {
  match id {
    Sha512_256KernelId::Portable => Sha512_256::compress_blocks_portable,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha512_256KernelId) -> Caps {
  match id {
    Sha512_256KernelId::Portable => Caps::NONE,
  }
}
