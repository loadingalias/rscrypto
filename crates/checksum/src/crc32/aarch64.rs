//! aarch64 CRC-32/CRC-32C kernels.
//!
//! This module provides:
//! - Hardware CRC extension kernels (`crc` target_feature)
//! - Fusion kernels that combine CRC instructions with PMULL folding (`aes`)
//! - Optional SHA3/EOR3-enhanced fusion kernels (`sha3`)
//!
//! # Safety
//!
//! Uses `unsafe` for aarch64 intrinsics. Callers must ensure required target
//! features are available before selecting a kernel (the dispatcher does this).

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::{arch::aarch64::*, ptr};

// ─────────────────────────────────────────────────────────────────────────────
// Hardware CRC extension (CRC-only)
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// PMULL helpers (fusion kernels)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_lo(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
  let result = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
  vreinterpretq_u64_p128(result)
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_hi(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
  let result = vmull_p64(vgetq_lane_u64(a, 1), vgetq_lane_u64(b, 1));
  vreinterpretq_u64_p128(result)
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_scalar(a: u32, b: u32) -> uint64x2_t {
  let result = vmull_p64(a as u64, b as u64);
  vreinterpretq_u64_p128(result)
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_lo_and_xor(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
  veorq_u64(clmul_lo(a, b), c)
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_hi_and_xor(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
  veorq_u64(clmul_hi(a, b), c)
}

// ─────────────────────────────────────────────────────────────────────────────
// Fusion: CRC-32C (iSCSI) - PMULL v12e_v1
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "crc,aes")]
unsafe fn crc32c_iscsi_pmull_v12e_v1(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = __crc32cb(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 192 {
    let end = buf.add(len);
    let limit = buf.add(len.strict_sub(192));

    let mut x0 = vld1q_u64(buf as *const u64);
    let mut x1 = vld1q_u64(buf.add(16) as *const u64);
    let mut x2 = vld1q_u64(buf.add(32) as *const u64);
    let mut x3 = vld1q_u64(buf.add(48) as *const u64);
    let mut x4 = vld1q_u64(buf.add(64) as *const u64);
    let mut x5 = vld1q_u64(buf.add(80) as *const u64);
    let mut x6 = vld1q_u64(buf.add(96) as *const u64);
    let mut x7 = vld1q_u64(buf.add(112) as *const u64);
    let mut x8 = vld1q_u64(buf.add(128) as *const u64);
    let mut x9 = vld1q_u64(buf.add(144) as *const u64);
    let mut x10 = vld1q_u64(buf.add(160) as *const u64);
    let mut x11 = vld1q_u64(buf.add(176) as *const u64);

    let k_vals: [u64; 2] = [0xa87ab8a8, 0xab7aff2a];
    let mut k = vld1q_u64(k_vals.as_ptr());

    let crc_vec = vsetq_lane_u64(crc0 as u64, vmovq_n_u64(0), 0);
    x0 = veorq_u64(crc_vec, x0);
    buf = buf.add(192);

    while buf <= limit {
      let y0 = clmul_lo_and_xor(x0, k, vld1q_u64(buf as *const u64));
      x0 = clmul_hi_and_xor(x0, k, y0);
      let y1 = clmul_lo_and_xor(x1, k, vld1q_u64(buf.add(16) as *const u64));
      x1 = clmul_hi_and_xor(x1, k, y1);
      let y2 = clmul_lo_and_xor(x2, k, vld1q_u64(buf.add(32) as *const u64));
      x2 = clmul_hi_and_xor(x2, k, y2);
      let y3 = clmul_lo_and_xor(x3, k, vld1q_u64(buf.add(48) as *const u64));
      x3 = clmul_hi_and_xor(x3, k, y3);
      let y4 = clmul_lo_and_xor(x4, k, vld1q_u64(buf.add(64) as *const u64));
      x4 = clmul_hi_and_xor(x4, k, y4);
      let y5 = clmul_lo_and_xor(x5, k, vld1q_u64(buf.add(80) as *const u64));
      x5 = clmul_hi_and_xor(x5, k, y5);
      let y6 = clmul_lo_and_xor(x6, k, vld1q_u64(buf.add(96) as *const u64));
      x6 = clmul_hi_and_xor(x6, k, y6);
      let y7 = clmul_lo_and_xor(x7, k, vld1q_u64(buf.add(112) as *const u64));
      x7 = clmul_hi_and_xor(x7, k, y7);
      let y8 = clmul_lo_and_xor(x8, k, vld1q_u64(buf.add(128) as *const u64));
      x8 = clmul_hi_and_xor(x8, k, y8);
      let y9 = clmul_lo_and_xor(x9, k, vld1q_u64(buf.add(144) as *const u64));
      x9 = clmul_hi_and_xor(x9, k, y9);
      let y10 = clmul_lo_and_xor(x10, k, vld1q_u64(buf.add(160) as *const u64));
      x10 = clmul_hi_and_xor(x10, k, y10);
      let y11 = clmul_lo_and_xor(x11, k, vld1q_u64(buf.add(176) as *const u64));
      x11 = clmul_hi_and_xor(x11, k, y11);
      buf = buf.add(192);
    }

    let k_vals: [u64; 2] = [0xf20c0dfe, 0x493c7d27];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x1);
    x0 = clmul_hi_and_xor(x0, k, y0);
    let y2 = clmul_lo_and_xor(x2, k, x3);
    x2 = clmul_hi_and_xor(x2, k, y2);
    let y4 = clmul_lo_and_xor(x4, k, x5);
    x4 = clmul_hi_and_xor(x4, k, y4);
    let y6 = clmul_lo_and_xor(x6, k, x7);
    x6 = clmul_hi_and_xor(x6, k, y6);
    let y8 = clmul_lo_and_xor(x8, k, x9);
    x8 = clmul_hi_and_xor(x8, k, y8);
    let y10 = clmul_lo_and_xor(x10, k, x11);
    x10 = clmul_hi_and_xor(x10, k, y10);

    let k_vals: [u64; 2] = [0x3da6d0cb, 0xba4fc28e];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x2);
    x0 = clmul_hi_and_xor(x0, k, y0);
    let y4 = clmul_lo_and_xor(x4, k, x6);
    x4 = clmul_hi_and_xor(x4, k, y4);
    let y8 = clmul_lo_and_xor(x8, k, x10);
    x8 = clmul_hi_and_xor(x8, k, y8);

    let k_vals: [u64; 2] = [0x740eef02, 0x9e4addf8];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x4);
    x0 = clmul_hi_and_xor(x0, k, y0);
    x4 = x8;
    let y0 = clmul_lo_and_xor(x0, k, x4);
    x0 = clmul_hi_and_xor(x0, k, y0);

    crc0 = __crc32cd(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32cd(crc0, vgetq_lane_u64(x0, 1));
    len = end.offset_from(buf) as usize;
  }

  if len >= 16 {
    let mut x0 = vld1q_u64(buf as *const u64);

    let k_vals: [u64; 2] = [0xf20c0dfe, 0x493c7d27];
    let k = vld1q_u64(k_vals.as_ptr());

    let crc_vec = vsetq_lane_u64(crc0 as u64, vmovq_n_u64(0), 0);
    x0 = veorq_u64(crc_vec, x0);
    buf = buf.add(16);
    len = len.strict_sub(16);

    while len >= 16 {
      let y0 = clmul_lo_and_xor(x0, k, vld1q_u64(buf as *const u64));
      x0 = clmul_hi_and_xor(x0, k, y0);
      buf = buf.add(16);
      len = len.strict_sub(16);
    }

    crc0 = __crc32cd(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32cd(crc0, vgetq_lane_u64(x0, 1));
  }

  while len >= 8 {
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = __crc32cb(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

/// Safe wrapper for CRC-32C fusion kernel (CRC+PMULL v12e_v1).
#[inline]
pub fn crc32c_iscsi_pmull_v12e_v1_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC + PMULL before selecting this kernel.
  unsafe { crc32c_iscsi_pmull_v12e_v1(crc, data.as_ptr(), data.len()) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fusion: CRC-32 (ISO-HDLC / IEEE) - PMULL v12e_v1
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "crc,aes")]
unsafe fn crc32_iso_hdlc_pmull_v12e_v1(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = __crc32b(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 192 {
    let end = buf.add(len);
    let limit = buf.add(len.strict_sub(192));

    let mut x0 = vld1q_u64(buf as *const u64);
    let mut x1 = vld1q_u64(buf.add(16) as *const u64);
    let mut x2 = vld1q_u64(buf.add(32) as *const u64);
    let mut x3 = vld1q_u64(buf.add(48) as *const u64);
    let mut x4 = vld1q_u64(buf.add(64) as *const u64);
    let mut x5 = vld1q_u64(buf.add(80) as *const u64);
    let mut x6 = vld1q_u64(buf.add(96) as *const u64);
    let mut x7 = vld1q_u64(buf.add(112) as *const u64);
    let mut x8 = vld1q_u64(buf.add(128) as *const u64);
    let mut x9 = vld1q_u64(buf.add(144) as *const u64);
    let mut x10 = vld1q_u64(buf.add(160) as *const u64);
    let mut x11 = vld1q_u64(buf.add(176) as *const u64);

    let k_vals: [u64; 2] = [0x596c8d81, 0xf5e48c85];
    let mut k = vld1q_u64(k_vals.as_ptr());

    let crc_vec = vsetq_lane_u64(crc0 as u64, vmovq_n_u64(0), 0);
    x0 = veorq_u64(crc_vec, x0);
    buf = buf.add(192);

    while buf <= limit {
      let y0 = clmul_lo_and_xor(x0, k, vld1q_u64(buf as *const u64));
      x0 = clmul_hi_and_xor(x0, k, y0);
      let y1 = clmul_lo_and_xor(x1, k, vld1q_u64(buf.add(16) as *const u64));
      x1 = clmul_hi_and_xor(x1, k, y1);
      let y2 = clmul_lo_and_xor(x2, k, vld1q_u64(buf.add(32) as *const u64));
      x2 = clmul_hi_and_xor(x2, k, y2);
      let y3 = clmul_lo_and_xor(x3, k, vld1q_u64(buf.add(48) as *const u64));
      x3 = clmul_hi_and_xor(x3, k, y3);
      let y4 = clmul_lo_and_xor(x4, k, vld1q_u64(buf.add(64) as *const u64));
      x4 = clmul_hi_and_xor(x4, k, y4);
      let y5 = clmul_lo_and_xor(x5, k, vld1q_u64(buf.add(80) as *const u64));
      x5 = clmul_hi_and_xor(x5, k, y5);
      let y6 = clmul_lo_and_xor(x6, k, vld1q_u64(buf.add(96) as *const u64));
      x6 = clmul_hi_and_xor(x6, k, y6);
      let y7 = clmul_lo_and_xor(x7, k, vld1q_u64(buf.add(112) as *const u64));
      x7 = clmul_hi_and_xor(x7, k, y7);
      let y8 = clmul_lo_and_xor(x8, k, vld1q_u64(buf.add(128) as *const u64));
      x8 = clmul_hi_and_xor(x8, k, y8);
      let y9 = clmul_lo_and_xor(x9, k, vld1q_u64(buf.add(144) as *const u64));
      x9 = clmul_hi_and_xor(x9, k, y9);
      let y10 = clmul_lo_and_xor(x10, k, vld1q_u64(buf.add(160) as *const u64));
      x10 = clmul_hi_and_xor(x10, k, y10);
      let y11 = clmul_lo_and_xor(x11, k, vld1q_u64(buf.add(176) as *const u64));
      x11 = clmul_hi_and_xor(x11, k, y11);
      buf = buf.add(192);
    }

    let k_vals: [u64; 2] = [0xae689191, 0xccaa009e];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x1);
    x0 = clmul_hi_and_xor(x0, k, y0);
    let y2 = clmul_lo_and_xor(x2, k, x3);
    x2 = clmul_hi_and_xor(x2, k, y2);
    let y4 = clmul_lo_and_xor(x4, k, x5);
    x4 = clmul_hi_and_xor(x4, k, y4);
    let y6 = clmul_lo_and_xor(x6, k, x7);
    x6 = clmul_hi_and_xor(x6, k, y6);
    let y8 = clmul_lo_and_xor(x8, k, x9);
    x8 = clmul_hi_and_xor(x8, k, y8);
    let y10 = clmul_lo_and_xor(x10, k, x11);
    x10 = clmul_hi_and_xor(x10, k, y10);

    let k_vals: [u64; 2] = [0xf1da05aa, 0x81256527];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x2);
    x0 = clmul_hi_and_xor(x0, k, y0);
    let y4 = clmul_lo_and_xor(x4, k, x6);
    x4 = clmul_hi_and_xor(x4, k, y4);
    let y8 = clmul_lo_and_xor(x8, k, x10);
    x8 = clmul_hi_and_xor(x8, k, y8);

    let k_vals: [u64; 2] = [0x8f352d95, 0x1d9513d7];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x4);
    x0 = clmul_hi_and_xor(x0, k, y0);
    x4 = x8;
    let y0 = clmul_lo_and_xor(x0, k, x4);
    x0 = clmul_hi_and_xor(x0, k, y0);

    crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32d(crc0, vgetq_lane_u64(x0, 1));
    len = end.offset_from(buf) as usize;
  }

  if len >= 16 {
    let mut x0 = vld1q_u64(buf as *const u64);

    let k_vals: [u64; 2] = [0xae689191, 0xccaa009e];
    let k = vld1q_u64(k_vals.as_ptr());

    let crc_vec = vsetq_lane_u64(crc0 as u64, vmovq_n_u64(0), 0);
    x0 = veorq_u64(crc_vec, x0);
    buf = buf.add(16);
    len = len.strict_sub(16);

    while len >= 16 {
      let y0 = clmul_lo_and_xor(x0, k, vld1q_u64(buf as *const u64));
      x0 = clmul_hi_and_xor(x0, k, y0);
      buf = buf.add(16);
      len = len.strict_sub(16);
    }

    crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32d(crc0, vgetq_lane_u64(x0, 1));
  }

  while len >= 8 {
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = __crc32b(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

/// Safe wrapper for CRC-32 fusion kernel (CRC+PMULL v12e_v1).
#[inline]
pub fn crc32_iso_hdlc_pmull_v12e_v1_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC + PMULL before selecting this kernel.
  unsafe { crc32_iso_hdlc_pmull_v12e_v1(crc, data.as_ptr(), data.len()) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fusion: EOR3 variants (CRC + PMULL + SHA3)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "crc,aes,sha3")]
unsafe fn crc32c_iscsi_pmull_eor3_v9s3x2e_s3(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  // Ported from fast-crc32 neon_eor3 CRC32C (v9s3x2e_s3).
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = __crc32cb(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 192 {
    let end = buf.add(len);
    let blk = len / 192;
    let klen = blk.strict_mul(16);
    let mut buf2 = buf.add(klen.strict_mul(3));
    let limit = buf.add(klen).sub(32);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    let mut x0 = vld1q_u64(buf2 as *const u64);
    let mut x1 = vld1q_u64(buf2.add(16) as *const u64);
    let mut x2 = vld1q_u64(buf2.add(32) as *const u64);
    let mut x3 = vld1q_u64(buf2.add(48) as *const u64);
    let mut x4 = vld1q_u64(buf2.add(64) as *const u64);
    let mut x5 = vld1q_u64(buf2.add(80) as *const u64);
    let mut x6 = vld1q_u64(buf2.add(96) as *const u64);
    let mut x7 = vld1q_u64(buf2.add(112) as *const u64);
    let mut x8 = vld1q_u64(buf2.add(128) as *const u64);

    let k_vals: [u64; 2] = [0x7e908048, 0xc96cfdc0];
    let mut k = vld1q_u64(k_vals.as_ptr());
    buf2 = buf2.add(144);

    while buf <= limit {
      let y0 = clmul_lo(x0, k);
      x0 = clmul_hi(x0, k);
      let y1 = clmul_lo(x1, k);
      x1 = clmul_hi(x1, k);
      let y2 = clmul_lo(x2, k);
      x2 = clmul_hi(x2, k);
      let y3 = clmul_lo(x3, k);
      x3 = clmul_hi(x3, k);
      let y4 = clmul_lo(x4, k);
      x4 = clmul_hi(x4, k);
      let y5 = clmul_lo(x5, k);
      x5 = clmul_hi(x5, k);
      let y6 = clmul_lo(x6, k);
      x6 = clmul_hi(x6, k);
      let y7 = clmul_lo(x7, k);
      x7 = clmul_hi(x7, k);
      let y8 = clmul_lo(x8, k);
      x8 = clmul_hi(x8, k);

      x0 = veor3q_u64(x0, y0, vld1q_u64(buf2 as *const u64));
      x1 = veor3q_u64(x1, y1, vld1q_u64(buf2.add(16) as *const u64));
      x2 = veor3q_u64(x2, y2, vld1q_u64(buf2.add(32) as *const u64));
      x3 = veor3q_u64(x3, y3, vld1q_u64(buf2.add(48) as *const u64));
      x4 = veor3q_u64(x4, y4, vld1q_u64(buf2.add(64) as *const u64));
      x5 = veor3q_u64(x5, y5, vld1q_u64(buf2.add(80) as *const u64));
      x6 = veor3q_u64(x6, y6, vld1q_u64(buf2.add(96) as *const u64));
      x7 = veor3q_u64(x7, y7, vld1q_u64(buf2.add(112) as *const u64));
      x8 = veor3q_u64(x8, y8, vld1q_u64(buf2.add(128) as *const u64));

      crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      crc0 = __crc32cd(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
      crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen + 8) as *const u64));
      crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64));

      buf = buf.add(16);
      buf2 = buf2.add(144);
    }

    let k_vals: [u64; 2] = [0xf20c0dfe, 0x493c7d27];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = veor3q_u64(x0, y0, x1);
    x1 = x2;
    x2 = x3;
    x3 = x4;
    x4 = x5;
    x5 = x6;
    x6 = x7;
    x7 = x8;

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y2 = clmul_lo(x2, k);
    x2 = clmul_hi(x2, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);
    let y6 = clmul_lo(x6, k);
    x6 = clmul_hi(x6, k);

    x0 = veor3q_u64(x0, y0, x1);
    x2 = veor3q_u64(x2, y2, x3);
    x4 = veor3q_u64(x4, y4, x5);
    x6 = veor3q_u64(x6, y6, x7);

    let k_vals: [u64; 2] = [0x3da6d0cb, 0xba4fc28e];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);

    x0 = veor3q_u64(x0, y0, x2);
    x4 = veor3q_u64(x4, y4, x6);

    let k_vals: [u64; 2] = [0x740eef02, 0x9e4addf8];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = veor3q_u64(x0, y0, x4);

    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
    crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
    crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen + 8) as *const u64));
    crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64));

    let vc0 = crc_shift_iscsi(crc0, klen.strict_mul(2) + blk.strict_mul(144));
    let vc1 = crc_shift_iscsi(crc1, klen + blk.strict_mul(144));
    let vc2 = crc_shift_iscsi(crc2, blk.strict_mul(144));
    let vc = vgetq_lane_u64(veor3q_u64(vc0, vc1, vc2), 0);

    crc0 = __crc32cd(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32cd(crc0, vc ^ vgetq_lane_u64(x0, 1));

    buf = buf2;
    len = end.offset_from(buf) as usize;
  }

  if len >= 32 {
    let klen = ((len.strict_sub(8)) / 24).strict_mul(8);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    loop {
      crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32cd(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32cd(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      buf = buf.add(8);
      len = len.strict_sub(24);
      if len < 32 {
        break;
      }
    }

    let vc0 = crc_shift_iscsi(crc0, klen.strict_mul(2) + 8);
    let vc1 = crc_shift_iscsi(crc1, klen + 8);
    let vc = vgetq_lane_u64(veorq_u64(vc0, vc1), 0);

    buf = buf.add(klen.strict_mul(2));
    crc0 = crc2;
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64) ^ vc);
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len >= 8 {
    crc0 = __crc32cd(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = __crc32cb(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn crc_shift_iscsi(crc: u32, nbytes: usize) -> uint64x2_t {
  clmul_scalar(crc, xnmodp_crc32_iscsi((nbytes.strict_mul(8).strict_sub(33)) as u64))
}

#[inline]
#[target_feature(enable = "crc,aes")]
unsafe fn xnmodp_crc32_iscsi(mut n: u64) -> u32 {
  let mut stack = !1u64;
  let mut acc: u32;
  let mut low: u32;

  while n > 191 {
    stack = (stack << 1) + (n & 1);
    n = (n >> 1).strict_sub(16);
  }
  stack = !stack;
  acc = 0x8000_0000u32 >> (n & 31);
  n >>= 5;

  while n > 0 {
    acc = __crc32cw(acc, 0);
    n = n.strict_sub(1);
  }

  while {
    low = (stack & 1) as u32;
    stack >>= 1;
    stack != 0
  } {
    let x = vreinterpret_p8_u64(vmov_n_u64(acc as u64));
    let squared = vmull_p8(x, x);
    let y = vgetq_lane_u64(vreinterpretq_u64_p16(squared), 0);
    acc = __crc32cd(0, y << low);
  }

  acc
}

/// Safe wrapper for CRC-32C fusion kernel (CRC+PMULL+EOR3 v9s3x2e_s3).
#[inline]
pub fn crc32c_iscsi_pmull_eor3_v9s3x2e_s3_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC + PMULL + SHA3/EOR3 before selecting this kernel.
  unsafe { crc32c_iscsi_pmull_eor3_v9s3x2e_s3(crc, data.as_ptr(), data.len()) }
}

#[inline]
#[target_feature(enable = "crc,aes,sha3")]
unsafe fn crc32_iso_hdlc_pmull_eor3_v9s3x2e_s3(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  // Ported from fast-crc32 neon_eor3 ISO-HDLC (v9s3x2e_s3).
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = __crc32b(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 192 {
    let end = buf.add(len);
    let blk = len / 192;
    let klen = blk.strict_mul(16);
    let mut buf2 = buf.add(klen.strict_mul(3));
    let limit = buf.add(klen).sub(32);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    let mut x0 = vld1q_u64(buf2 as *const u64);
    let mut x1 = vld1q_u64(buf2.add(16) as *const u64);
    let mut x2 = vld1q_u64(buf2.add(32) as *const u64);
    let mut x3 = vld1q_u64(buf2.add(48) as *const u64);
    let mut x4 = vld1q_u64(buf2.add(64) as *const u64);
    let mut x5 = vld1q_u64(buf2.add(80) as *const u64);
    let mut x6 = vld1q_u64(buf2.add(96) as *const u64);
    let mut x7 = vld1q_u64(buf2.add(112) as *const u64);
    let mut x8 = vld1q_u64(buf2.add(128) as *const u64);

    let k_vals: [u64; 2] = [0x26b70c3d, 0x3f41287a];
    let mut k = vld1q_u64(k_vals.as_ptr());
    buf2 = buf2.add(144);

    while buf <= limit {
      let y0 = clmul_lo(x0, k);
      x0 = clmul_hi(x0, k);
      let y1 = clmul_lo(x1, k);
      x1 = clmul_hi(x1, k);
      let y2 = clmul_lo(x2, k);
      x2 = clmul_hi(x2, k);
      let y3 = clmul_lo(x3, k);
      x3 = clmul_hi(x3, k);
      let y4 = clmul_lo(x4, k);
      x4 = clmul_hi(x4, k);
      let y5 = clmul_lo(x5, k);
      x5 = clmul_hi(x5, k);
      let y6 = clmul_lo(x6, k);
      x6 = clmul_hi(x6, k);
      let y7 = clmul_lo(x7, k);
      x7 = clmul_hi(x7, k);
      let y8 = clmul_lo(x8, k);
      x8 = clmul_hi(x8, k);

      x0 = veor3q_u64(x0, y0, vld1q_u64(buf2 as *const u64));
      x1 = veor3q_u64(x1, y1, vld1q_u64(buf2.add(16) as *const u64));
      x2 = veor3q_u64(x2, y2, vld1q_u64(buf2.add(32) as *const u64));
      x3 = veor3q_u64(x3, y3, vld1q_u64(buf2.add(48) as *const u64));
      x4 = veor3q_u64(x4, y4, vld1q_u64(buf2.add(64) as *const u64));
      x5 = veor3q_u64(x5, y5, vld1q_u64(buf2.add(80) as *const u64));
      x6 = veor3q_u64(x6, y6, vld1q_u64(buf2.add(96) as *const u64));
      x7 = veor3q_u64(x7, y7, vld1q_u64(buf2.add(112) as *const u64));
      x8 = veor3q_u64(x8, y8, vld1q_u64(buf2.add(128) as *const u64));

      crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      crc0 = __crc32d(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
      crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen + 8) as *const u64));
      crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64));

      buf = buf.add(16);
      buf2 = buf2.add(144);
    }

    let k_vals: [u64; 2] = [0xae689191, 0xccaa009e];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = veor3q_u64(x0, y0, x1);
    x1 = x2;
    x2 = x3;
    x3 = x4;
    x4 = x5;
    x5 = x6;
    x6 = x7;
    x7 = x8;

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y2 = clmul_lo(x2, k);
    x2 = clmul_hi(x2, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);
    let y6 = clmul_lo(x6, k);
    x6 = clmul_hi(x6, k);

    x0 = veor3q_u64(x0, y0, x1);
    x2 = veor3q_u64(x2, y2, x3);
    x4 = veor3q_u64(x4, y4, x5);
    x6 = veor3q_u64(x6, y6, x7);

    let k_vals: [u64; 2] = [0xf1da05aa, 0x81256527];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);

    x0 = veor3q_u64(x0, y0, x2);
    x4 = veor3q_u64(x4, y4, x6);

    let k_vals: [u64; 2] = [0x8f352d95, 0x1d9513d7];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = veor3q_u64(x0, y0, x4);

    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
    crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
    crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen + 8) as *const u64));
    crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64));

    let vc0 = crc_shift_iso_hdlc(crc0, klen.strict_mul(2) + blk.strict_mul(144));
    let vc1 = crc_shift_iso_hdlc(crc1, klen + blk.strict_mul(144));
    let vc2 = crc_shift_iso_hdlc(crc2, blk.strict_mul(144));
    let vc = vgetq_lane_u64(veor3q_u64(vc0, vc1, vc2), 0);

    crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
    crc0 = __crc32d(crc0, vc ^ vgetq_lane_u64(x0, 1));

    buf = buf2;
    len = end.offset_from(buf) as usize;
  }

  if len >= 32 {
    let klen = ((len.strict_sub(8)) / 24).strict_mul(8);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    loop {
      crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = __crc32d(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = __crc32d(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      buf = buf.add(8);
      len = len.strict_sub(24);
      if len < 32 {
        break;
      }
    }

    let vc0 = crc_shift_iso_hdlc(crc0, klen.strict_mul(2) + 8);
    let vc1 = crc_shift_iso_hdlc(crc1, klen + 8);
    let vc = vgetq_lane_u64(veorq_u64(vc0, vc1), 0);

    buf = buf.add(klen.strict_mul(2));
    crc0 = crc2;
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64) ^ vc);
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len >= 8 {
    crc0 = __crc32d(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = __crc32b(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn crc_shift_iso_hdlc(crc: u32, nbytes: usize) -> uint64x2_t {
  clmul_scalar(crc, xnmodp_iso_hdlc((nbytes.strict_mul(8).strict_sub(33)) as u64))
}

#[inline]
#[target_feature(enable = "crc,aes")]
unsafe fn xnmodp_iso_hdlc(mut n: u64) -> u32 {
  let mut stack = !1u64;
  let mut acc: u32;
  let mut low: u32;

  while n > 191 {
    stack = (stack << 1) + (n & 1);
    n = (n >> 1).strict_sub(16);
  }
  stack = !stack;
  acc = 0x8000_0000u32 >> (n & 31);
  n >>= 5;

  while n > 0 {
    acc = __crc32w(acc, 0);
    n = n.strict_sub(1);
  }

  while {
    low = (stack & 1) as u32;
    stack >>= 1;
    stack != 0
  } {
    let x = vreinterpret_p8_u64(vmov_n_u64(acc as u64));
    let squared = vmull_p8(x, x);
    let y = vgetq_lane_u64(vreinterpretq_u64_p16(squared), 0);
    acc = __crc32d(0, y << low);
  }

  acc
}

/// Safe wrapper for CRC-32 fusion kernel (CRC+PMULL+EOR3 v9s3x2e_s3).
#[inline]
pub fn crc32_iso_hdlc_pmull_eor3_v9s3x2e_s3_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC + PMULL + SHA3/EOR3 before selecting this kernel.
  unsafe { crc32_iso_hdlc_pmull_eor3_v9s3x2e_s3(crc, data.as_ptr(), data.len()) }
}
