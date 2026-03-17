use super::Sha256;
use crate::platform::Caps;
#[cfg(target_arch = "aarch64")]
use crate::platform::caps::aarch64;
#[cfg(target_arch = "x86_64")]
use crate::platform::caps::x86;

pub(crate) type CompressBlocksFn = fn(&mut [u32; 8], &[u8]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Sha256KernelId {
  Portable = 0,
  #[cfg(target_arch = "x86_64")]
  X86Sha = 1,
  #[cfg(target_arch = "aarch64")]
  Aarch64Sha2 = 2,
}

#[cfg(any(test, feature = "std"))]
pub const ALL: &[Sha256KernelId] = &[
  Sha256KernelId::Portable,
  #[cfg(target_arch = "x86_64")]
  Sha256KernelId::X86Sha,
  #[cfg(target_arch = "aarch64")]
  Sha256KernelId::Aarch64Sha2,
];

impl Sha256KernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "x86_64")]
      Self::X86Sha => "x86-sha",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Sha2 => "aarch64-sha2",
    }
  }
}

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<Sha256KernelId> {
  match name {
    "portable" => Some(Sha256KernelId::Portable),
    #[cfg(target_arch = "x86_64")]
    "x86-sha" => Some(Sha256KernelId::X86Sha),
    #[cfg(target_arch = "aarch64")]
    "aarch64-sha2" => Some(Sha256KernelId::Aarch64Sha2),
    _ => None,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe wrappers — dispatch validates caps before calling.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn compress_blocks_x86_sha(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `x86::SHA` is available.
  unsafe { super::x86_64::compress_blocks_sha_ni(state, blocks) }
}

#[cfg(target_arch = "aarch64")]
fn compress_blocks_aarch64_sha2(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `aarch64::SHA2` is available.
  unsafe { super::aarch64::compress_blocks_aarch64_sha2(state, blocks) }
}

#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha256KernelId) -> CompressBlocksFn {
  match id {
    Sha256KernelId::Portable => Sha256::compress_blocks_portable,
    #[cfg(target_arch = "x86_64")]
    Sha256KernelId::X86Sha => compress_blocks_x86_sha,
    #[cfg(target_arch = "aarch64")]
    Sha256KernelId::Aarch64Sha2 => compress_blocks_aarch64_sha2,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha256KernelId) -> Caps {
  match id {
    Sha256KernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "x86_64")]
    Sha256KernelId::X86Sha => x86::SHA,
    #[cfg(target_arch = "aarch64")]
    Sha256KernelId::Aarch64Sha2 => aarch64::SHA2,
  }
}
