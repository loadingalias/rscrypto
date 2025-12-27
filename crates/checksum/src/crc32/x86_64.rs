//! x86_64 hardware CRC-32C kernel (SSE4.2 `crc32` instruction).
//!
//! # Safety
//!
//! Uses `unsafe` for x86 SIMD intrinsics. Callers must ensure SSE4.2 is
//! available before executing the accelerated path (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

/// CRC-32C update using SSE4.2 `crc32` instruction.
///
/// `crc` is the current state (pre-inverted).
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_sse42(crc: u32, data: &[u8]) -> u32 {
  let mut state64 = crc as u64;

  let (chunks8, tail8) = data.as_chunks::<8>();
  for chunk in chunks8 {
    state64 = _mm_crc32_u64(state64, u64::from_le_bytes(*chunk));
  }

  let mut state = state64 as u32;

  let (chunks4, tail4) = tail8.as_chunks::<4>();
  for chunk in chunks4 {
    state = _mm_crc32_u32(state, u32::from_le_bytes(*chunk));
  }

  let (chunks2, tail2) = tail4.as_chunks::<2>();
  for chunk in chunks2 {
    state = _mm_crc32_u16(state, u16::from_le_bytes(*chunk));
  }

  for &b in tail2 {
    state = _mm_crc32_u8(state, b);
  }

  state
}

/// Safe wrapper for CRC-32C SSE4.2 kernel.
#[inline]
pub fn crc32c_sse42_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSE4.2 before selecting this kernel.
  unsafe { crc32c_sse42(crc, data) }
}
