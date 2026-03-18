use super::Sha512;
use crate::platform::Caps;
#[cfg(target_arch = "aarch64")]
use crate::platform::caps::aarch64;

pub(crate) type CompressBlocksFn = fn(&mut [u64; 8], &[u8]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Sha512KernelId {
  Portable = 0,
  #[cfg(target_arch = "aarch64")]
  Aarch64Sha512 = 1,
}

#[cfg(any(test, feature = "std"))]
pub const ALL: &[Sha512KernelId] = &[
  Sha512KernelId::Portable,
  #[cfg(target_arch = "aarch64")]
  Sha512KernelId::Aarch64Sha512,
];

impl Sha512KernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Sha512 => "aarch64-sha512",
    }
  }
}

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<Sha512KernelId> {
  match name {
    "portable" => Some(Sha512KernelId::Portable),
    #[cfg(target_arch = "aarch64")]
    "aarch64-sha512" => Some(Sha512KernelId::Aarch64Sha512),
    _ => None,
  }
}

#[cfg(target_arch = "aarch64")]
fn compress_blocks_aarch64_sha512(state: &mut [u64; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `aarch64::SHA512` is available.
  unsafe { super::aarch64::compress_blocks_aarch64_sha512(state, blocks) }
}

#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha512KernelId) -> CompressBlocksFn {
  match id {
    Sha512KernelId::Portable => Sha512::compress_blocks_portable,
    #[cfg(target_arch = "aarch64")]
    Sha512KernelId::Aarch64Sha512 => compress_blocks_aarch64_sha512,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha512KernelId) -> Caps {
  match id {
    Sha512KernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "aarch64")]
    Sha512KernelId::Aarch64Sha512 => aarch64::SHA512,
  }
}
