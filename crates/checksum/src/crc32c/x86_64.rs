//! x86_64-accelerated CRC32-C (Castagnoli).
//!
//! Uses SSE4.2 `crc32` instructions (CRC32-C polynomial).
//!
//! Safety:
//! - This file is allowed to use `unsafe` for ISA-specific intrinsics.
//! - All unsafe is contained within this module.

#![allow(unsafe_code)]

#[cfg(any(target_feature = "sse4.2", feature = "std"))]
use core::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u16, _mm_crc32_u32, _mm_crc32_u64};

/// Compute CRC32-C using SSE4.2 `crc32` instructions.
///
/// # Safety
/// Caller must ensure the CPU supports the `sse4.2` target feature.
#[cfg(any(target_feature = "sse4.2", feature = "std"))]
#[target_feature(enable = "sse4.2")]
pub(crate) unsafe fn compute_sse42_unchecked(crc: u32, data: &[u8]) -> u32 {
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
      current = _mm_crc32_u64(current as u64, v) as u32;
      ptr = unsafe { ptr.add(8) };
    }
  }

  // SAFETY: `ptr` is within `data` (or one-past-the-end).
  let mut len = unsafe { end.offset_from(ptr) as usize };

  if len >= 4 {
    // SAFETY: `len >= 4` and `ptr` is in-bounds for `data`.
    #[allow(clippy::cast_ptr_alignment)]
    let v = unsafe { core::ptr::read_unaligned(ptr as *const u32) };
    current = _mm_crc32_u32(current, v);
    ptr = unsafe { ptr.add(4) };
    len = len.strict_sub(4);
  }

  if len >= 2 {
    // SAFETY: `len >= 2` and `ptr` is in-bounds for `data`.
    #[allow(clippy::cast_ptr_alignment)]
    let v = unsafe { core::ptr::read_unaligned(ptr as *const u16) };
    current = _mm_crc32_u16(current, v);
    ptr = unsafe { ptr.add(2) };
    len = len.strict_sub(2);
  }

  if len != 0 {
    // SAFETY: `len != 0` implies `ptr < end`.
    current = _mm_crc32_u8(current, unsafe { *ptr });
  }

  current
}

/// Compute CRC32-C using SSE4.2 when it is enabled at compile time.
#[cfg(target_feature = "sse4.2")]
#[inline]
pub fn compute_sse42_enabled(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: this function is only compiled when `target_feature="sse4.2"`.
  unsafe { compute_sse42_unchecked(crc, data) }
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn compute_sse42_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: selected only when `is_x86_feature_detected!("sse4.2")` is true.
  unsafe { compute_sse42_unchecked(crc, data) }
}
