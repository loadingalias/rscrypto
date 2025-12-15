//! aarch64-accelerated CRC32-C (Castagnoli).
//!
//! Uses the ARMv8 CRC32 extension (`crc*` instructions).
//!
//! Safety:
//! - This file is allowed to use `unsafe` for ISA-specific intrinsics.
//! - All unsafe is contained within this module.

#![allow(unsafe_code)]

#[cfg(any(target_feature = "crc", feature = "std"))]
use core::arch::aarch64::{__crc32cb, __crc32cd, __crc32ch, __crc32cw};

/// Compute CRC32-C using the ARMv8 CRC32 extension.
///
/// # Safety
/// Caller must ensure the CPU supports the `crc` target feature.
#[cfg(any(target_feature = "crc", feature = "std"))]
#[target_feature(enable = "crc")]
pub(crate) unsafe fn compute_crc_unchecked(crc: u32, data: &[u8]) -> u32 {
  let mut current = crc;

  let mut ptr = data.as_ptr();
  let end = unsafe { ptr.add(data.len()) };

  if data.len() >= 8 {
    // SAFETY: `end` is in-bounds (one-past-the-end), and `end.sub(8)` is
    // in-bounds when `data.len() >= 8`.
    let end8 = unsafe { end.sub(8) };
    while ptr <= end8 {
      // SAFETY: `ptr` is within `data` and `ptr <= end8` implies `ptr.add(8) <= end`.
      #[allow(clippy::cast_ptr_alignment)]
      let v = unsafe { core::ptr::read_unaligned(ptr as *const u64) };
      current = __crc32cd(current, v);
      ptr = unsafe { ptr.add(8) };
    }
  }

  // SAFETY: `ptr` is within `data` (or one-past-the-end).
  let mut len = unsafe { end.offset_from(ptr) as usize };

  if len >= 4 {
    // SAFETY: `len >= 4` and `ptr` is in-bounds for `data`.
    #[allow(clippy::cast_ptr_alignment)]
    let v = unsafe { core::ptr::read_unaligned(ptr as *const u32) };
    current = __crc32cw(current, v);
    ptr = unsafe { ptr.add(4) };
    len = len.strict_sub(4);
  }

  if len >= 2 {
    // SAFETY: `len >= 2` and `ptr` is in-bounds for `data`.
    #[allow(clippy::cast_ptr_alignment)]
    let v = unsafe { core::ptr::read_unaligned(ptr as *const u16) };
    current = __crc32ch(current, v);
    ptr = unsafe { ptr.add(2) };
    len = len.strict_sub(2);
  }

  if len != 0 {
    // SAFETY: `len != 0` implies `ptr < end`.
    current = __crc32cb(current, unsafe { *ptr });
  }

  current
}

/// Compute CRC32-C using the `crc` target feature when it is enabled at compile time.
#[cfg(target_feature = "crc")]
#[inline]
pub fn compute_crc_enabled(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: this function is only compiled when `target_feature="crc"`.
  unsafe { compute_crc_unchecked(crc, data) }
}

#[cfg(all(feature = "std", not(target_feature = "crc")))]
#[inline]
pub(crate) fn compute_crc_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: callers must gate this with `is_aarch64_feature_detected!("crc")`.
  unsafe { compute_crc_unchecked(crc, data) }
}
