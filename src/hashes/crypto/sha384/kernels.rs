use super::Sha384;
use crate::platform::Caps;

pub(crate) type CompressBlocksFn = fn(&mut [u64; 8], &[u8]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Sha384KernelId {
  Portable = 0,
}

#[cfg(any(test, feature = "std"))]
pub const ALL: &[Sha384KernelId] = &[Sha384KernelId::Portable];

impl Sha384KernelId {
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
pub fn id_from_name(name: &str) -> Option<Sha384KernelId> {
  match name {
    "portable" => Some(Sha384KernelId::Portable),
    _ => None,
  }
}

#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha384KernelId) -> CompressBlocksFn {
  match id {
    Sha384KernelId::Portable => Sha384::compress_blocks_portable,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha384KernelId) -> Caps {
  match id {
    Sha384KernelId::Portable => Caps::NONE,
  }
}
