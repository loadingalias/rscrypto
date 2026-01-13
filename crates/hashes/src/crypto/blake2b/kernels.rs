use platform::Caps;

use super::Blake2b512;

pub(crate) type CompressFn = fn(&mut [u64; 8], &[u8], &mut u128, bool, u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Blake2b512KernelId {
  Portable = 0,
}

pub const ALL: &[Blake2b512KernelId] = &[Blake2b512KernelId::Portable];

impl Blake2b512KernelId {
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
pub fn id_from_name(name: &str) -> Option<Blake2b512KernelId> {
  match name {
    "portable" => Some(Blake2b512KernelId::Portable),
    _ => None,
  }
}

#[must_use]
pub(crate) fn compress_fn(id: Blake2b512KernelId) -> CompressFn {
  match id {
    Blake2b512KernelId::Portable => Blake2b512::compress_portable,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Blake2b512KernelId) -> Caps {
  match id {
    Blake2b512KernelId::Portable => Caps::NONE,
  }
}
