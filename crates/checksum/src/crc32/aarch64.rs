//! aarch64-accelerated CRC32 (ISO-HDLC).
//!
//! Uses the ARMv8 CRC32 extension (`crc32*` instructions).

#![allow(unsafe_code)]

#[cfg(any(target_feature = "crc", feature = "std"))]
use core::arch::aarch64::{__crc32b, __crc32d, __crc32h, __crc32w};

/// Compute CRC32 using the ARMv8 CRC32 extension.
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
    let end8 = unsafe { end.sub(8) };
    while ptr <= end8 {
      #[allow(clippy::cast_ptr_alignment)]
      let v = unsafe { core::ptr::read_unaligned(ptr as *const u64) };
      current = __crc32d(current, v);
      ptr = unsafe { ptr.add(8) };
    }
  }

  let mut len = unsafe { end.offset_from(ptr) as usize };

  if len >= 4 {
    #[allow(clippy::cast_ptr_alignment)]
    let v = unsafe { core::ptr::read_unaligned(ptr as *const u32) };
    current = __crc32w(current, v);
    ptr = unsafe { ptr.add(4) };
    len = len.strict_sub(4);
  }

  if len >= 2 {
    #[allow(clippy::cast_ptr_alignment)]
    let v = unsafe { core::ptr::read_unaligned(ptr as *const u16) };
    current = __crc32h(current, v);
    ptr = unsafe { ptr.add(2) };
    len = len.strict_sub(2);
  }

  if len != 0 {
    current = __crc32b(current, unsafe { *ptr });
  }

  current
}

/// Compute CRC32 when the `crc` target feature is enabled at compile time.
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
