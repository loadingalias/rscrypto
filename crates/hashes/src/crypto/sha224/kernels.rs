use platform::Caps;

use super::Sha224;

pub(crate) type CompressBlocksFn = fn(&mut [u32; 8], &[u8]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Sha224KernelId {
  Portable = 0,
}

pub const ALL: &[Sha224KernelId] = &[Sha224KernelId::Portable];

impl Sha224KernelId {
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
pub fn id_from_name(name: &str) -> Option<Sha224KernelId> {
  match name {
    "portable" => Some(Sha224KernelId::Portable),
    _ => None,
  }
}

#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha224KernelId) -> CompressBlocksFn {
  match id {
    Sha224KernelId::Portable => Sha224::compress_blocks_portable,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha224KernelId) -> Caps {
  match id {
    Sha224KernelId::Portable => Caps::NONE,
  }
}
