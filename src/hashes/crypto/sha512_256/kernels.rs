// SHA-512/256 uses SHA-512's compression function (FIPS 180-4 SS6.4).
// Only H0 and output truncation differ.
pub(crate) use crate::hashes::crypto::sha512::kernels::CompressBlocksFn;
#[cfg(target_arch = "aarch64")]
use crate::platform::caps::aarch64;
use crate::{hashes::crypto::sha512::Sha512, platform::Caps};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Sha512_256KernelId {
  Portable = 0,
  #[cfg(target_arch = "aarch64")]
  Aarch64Sha512 = 1,
}

#[cfg(any(test, feature = "std"))]
pub const ALL: &[Sha512_256KernelId] = &[
  Sha512_256KernelId::Portable,
  #[cfg(target_arch = "aarch64")]
  Sha512_256KernelId::Aarch64Sha512,
];

impl Sha512_256KernelId {
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
pub fn id_from_name(name: &str) -> Option<Sha512_256KernelId> {
  match name {
    "portable" => Some(Sha512_256KernelId::Portable),
    #[cfg(target_arch = "aarch64")]
    "aarch64-sha512" => Some(Sha512_256KernelId::Aarch64Sha512),
    _ => None,
  }
}

// Delegate to SHA-512 kernels — identical compression function.
#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha512_256KernelId) -> CompressBlocksFn {
  match id {
    Sha512_256KernelId::Portable => Sha512::compress_blocks_portable,
    #[cfg(target_arch = "aarch64")]
    Sha512_256KernelId::Aarch64Sha512 => crate::hashes::crypto::sha512::kernels::compress_blocks_fn(
      crate::hashes::crypto::sha512::kernels::Sha512KernelId::Aarch64Sha512,
    ),
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha512_256KernelId) -> Caps {
  match id {
    Sha512_256KernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "aarch64")]
    Sha512_256KernelId::Aarch64Sha512 => aarch64::SHA512,
  }
}
