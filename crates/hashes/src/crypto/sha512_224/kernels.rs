use platform::Caps;

use super::Sha512_224;

pub(crate) type CompressBlocksFn = fn(&mut [u64; 8], &[u8]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Sha512_224KernelId {
  Portable = 0,
}

pub const ALL: &[Sha512_224KernelId] = &[Sha512_224KernelId::Portable];

impl Sha512_224KernelId {
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
pub fn id_from_name(name: &str) -> Option<Sha512_224KernelId> {
  match name {
    "portable" => Some(Sha512_224KernelId::Portable),
    _ => None,
  }
}

#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha512_224KernelId) -> CompressBlocksFn {
  match id {
    Sha512_224KernelId::Portable => Sha512_224::compress_blocks_portable,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha512_224KernelId) -> Caps {
  match id {
    Sha512_224KernelId::Portable => Caps::NONE,
  }
}
