//! aarch64 hardware CRC-32/CRC-32C kernels (ARMv8 CRC extension).
//!
//! # Safety
//!
//! Uses `unsafe` for aarch64 intrinsics. Callers must ensure the CRC extension
//! is available before executing the accelerated path (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::aarch64::*;

/// CRC-32 (IEEE) update using ARMv8 CRC extension.
///
/// `crc` is the current state (pre-inverted).
#[inline]
#[target_feature(enable = "crc")]
unsafe fn crc32_armv8(crc: u32, data: &[u8]) -> u32 {
  let mut state = crc;

  let (chunks8, tail8) = data.as_chunks::<8>();
  for chunk in chunks8 {
    state = __crc32d(state, u64::from_le_bytes(*chunk));
  }

  let (chunks4, tail4) = tail8.as_chunks::<4>();
  for chunk in chunks4 {
    state = __crc32w(state, u32::from_le_bytes(*chunk));
  }

  let (chunks2, tail2) = tail4.as_chunks::<2>();
  for chunk in chunks2 {
    state = __crc32h(state, u16::from_le_bytes(*chunk));
  }

  for &b in tail2 {
    state = __crc32b(state, b);
  }

  state
}

/// CRC-32C (Castagnoli) update using ARMv8 CRC extension.
///
/// `crc` is the current state (pre-inverted).
#[inline]
#[target_feature(enable = "crc")]
unsafe fn crc32c_armv8(crc: u32, data: &[u8]) -> u32 {
  let mut state = crc;

  let (chunks8, tail8) = data.as_chunks::<8>();
  for chunk in chunks8 {
    state = __crc32cd(state, u64::from_le_bytes(*chunk));
  }

  let (chunks4, tail4) = tail8.as_chunks::<4>();
  for chunk in chunks4 {
    state = __crc32cw(state, u32::from_le_bytes(*chunk));
  }

  let (chunks2, tail2) = tail4.as_chunks::<2>();
  for chunk in chunks2 {
    state = __crc32ch(state, u16::from_le_bytes(*chunk));
  }

  for &b in tail2 {
    state = __crc32cb(state, b);
  }

  state
}

/// Safe wrapper for CRC-32 ARMv8 CRC extension kernel.
#[inline]
pub fn crc32_armv8_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC extension before selecting this kernel.
  unsafe { crc32_armv8(crc, data) }
}

/// Safe wrapper for CRC-32C ARMv8 CRC extension kernel.
#[inline]
pub fn crc32c_armv8_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC extension before selecting this kernel.
  unsafe { crc32c_armv8(crc, data) }
}
