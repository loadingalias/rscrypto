use platform::Caps;

use super::Blake2s256;

pub(crate) type CompressFn = fn(&mut [u32; 8], &[u8], &mut u64, bool, u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Blake2s256KernelId {
  Portable = 0,
}

pub const ALL: &[Blake2s256KernelId] = &[Blake2s256KernelId::Portable];

impl Blake2s256KernelId {
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
pub fn id_from_name(name: &str) -> Option<Blake2s256KernelId> {
  match name {
    "portable" => Some(Blake2s256KernelId::Portable),
    _ => None,
  }
}

#[must_use]
pub(crate) fn compress_fn(id: Blake2s256KernelId) -> CompressFn {
  match id {
    Blake2s256KernelId::Portable => Blake2s256::compress_portable,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Blake2s256KernelId) -> Caps {
  match id {
    Blake2s256KernelId::Portable => Caps::NONE,
  }
}
