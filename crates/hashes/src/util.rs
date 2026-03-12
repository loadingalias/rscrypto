#![allow(clippy::indexing_slicing)] // Fixed-size array indexing and block parsing

#[inline(always)]
#[must_use]
#[allow(unreachable_code)] // cfg branches may return early on some targets.
const fn dispatch_baseline_caps() -> platform::Caps {
  #[cfg(target_arch = "x86_64")]
  {
    return platform::caps::x86::SSE2;
  }

  #[cfg(target_arch = "aarch64")]
  {
    return platform::caps::aarch64::NEON;
  }

  platform::Caps::NONE
}

#[inline]
#[must_use]
pub(crate) fn dispatch_caps() -> platform::Caps {
  let static_caps = platform::caps_static();
  if static_caps.difference(dispatch_baseline_caps()).is_empty() {
    platform::caps()
  } else {
    static_caps
  }
}

#[inline(always)]
pub const fn rotr32(x: u32, n: u32) -> u32 {
  x.rotate_right(n)
}

#[inline(always)]
pub const fn rotr64(x: u64, n: u32) -> u64 {
  x.rotate_right(n)
}
