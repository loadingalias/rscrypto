//! PMULL (AES/crypto extension) 32-bit CRC engines.
//!
//! This module implements high-throughput CRC32-C and CRC32 on aarch64 using
//! PMULL folding plus CRC32 instructions.
//!
//! For CPUs with `sha3`, we use the best-known configuration on Apple M-series:
//! `v9s3x2e_s3` (9 vector accumulators + 3 scalar CRC streams, PMULL+EOR3).
//!
//! For CPUs without `sha3`, we use a PMULL-only baseline (`v12e_v1`).

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::cast_ptr_alignment)]

use core::arch::aarch64::{
  __crc32b, __crc32cb, __crc32cd, __crc32cw, __crc32d, __crc32w, uint64x2_t, veor3q_u64, veorq_u64, vgetq_lane_u64,
  vld1q_u64, vmov_n_u64, vmovq_n_u64, vmull_p8, vmull_p64, vreinterpret_p8_u64, vreinterpretq_u64_p16,
  vreinterpretq_u64_p128, vsetq_lane_u64,
};

pub(crate) trait Crc32PmullSpec {
  const KEY_16: (u64, u64);
  const KEY_32: (u64, u64);
  const KEY_64: (u64, u64);
  const KEY_144: (u64, u64);
  const KEY_192: (u64, u64);

  unsafe fn crc8(crc: u32, byte: u8) -> u32;
  unsafe fn crc32(crc: u32, word: u32) -> u32;
  unsafe fn crc64(crc: u32, dword: u64) -> u32;

  unsafe fn compute_crc_unchecked(crc: u32, data: &[u8]) -> u32;
}

pub(crate) struct Crc32cPmullSpec;

impl Crc32PmullSpec for Crc32cPmullSpec {
  const KEY_16: (u64, u64) = crate::constants::crc32c::fold::PMULL_KEY_16;
  const KEY_32: (u64, u64) = crate::constants::crc32c::fold::PMULL_KEY_32;
  const KEY_64: (u64, u64) = crate::constants::crc32c::fold::PMULL_KEY_64;
  const KEY_144: (u64, u64) = crate::constants::crc32c::fold::PMULL_KEY_144;
  const KEY_192: (u64, u64) = crate::constants::crc32c::fold::PMULL_KEY_192;

  #[inline(always)]
  unsafe fn crc8(crc: u32, byte: u8) -> u32 {
    __crc32cb(crc, byte)
  }

  #[inline(always)]
  unsafe fn crc32(crc: u32, word: u32) -> u32 {
    __crc32cw(crc, word)
  }

  #[inline(always)]
  unsafe fn crc64(crc: u32, dword: u64) -> u32 {
    __crc32cd(crc, dword)
  }

  #[inline(always)]
  unsafe fn compute_crc_unchecked(crc: u32, data: &[u8]) -> u32 {
    crate::crc32c::aarch64::compute_crc_unchecked(crc, data)
  }
}

pub(crate) struct Crc32IsoHdlcPmullSpec;

impl Crc32PmullSpec for Crc32IsoHdlcPmullSpec {
  const KEY_16: (u64, u64) = crate::constants::crc32::fold::PMULL_KEY_16;
  const KEY_32: (u64, u64) = crate::constants::crc32::fold::PMULL_KEY_32;
  const KEY_64: (u64, u64) = crate::constants::crc32::fold::PMULL_KEY_64;
  const KEY_144: (u64, u64) = crate::constants::crc32::fold::PMULL_KEY_144;
  const KEY_192: (u64, u64) = crate::constants::crc32::fold::PMULL_KEY_192;

  #[inline(always)]
  unsafe fn crc8(crc: u32, byte: u8) -> u32 {
    __crc32b(crc, byte)
  }

  #[inline(always)]
  unsafe fn crc32(crc: u32, word: u32) -> u32 {
    __crc32w(crc, word)
  }

  #[inline(always)]
  unsafe fn crc64(crc: u32, dword: u64) -> u32 {
    __crc32d(crc, dword)
  }

  #[inline(always)]
  unsafe fn compute_crc_unchecked(crc: u32, data: &[u8]) -> u32 {
    crate::crc32::aarch64::compute_crc_unchecked(crc, data)
  }
}

#[inline(always)]
unsafe fn load_u64(ptr: *const u8) -> u64 {
  core::ptr::read_unaligned(ptr as *const u64)
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_lo(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
  let result = vmull_p64(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b));
  vreinterpretq_u64_p128(result)
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_hi(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
  let result = vmull_p64(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<1>(b));
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

#[inline]
#[target_feature(enable = "aes")]
unsafe fn crc32_shift<S: Crc32PmullSpec>(crc: u32, nbytes: usize) -> uint64x2_t {
  // `nbytes * 8 - 33` matches the corsix/fast-crc32 generator output for CRC32C/CRC32.
  let nbits = (nbytes as u64).wrapping_mul(8).wrapping_sub(33);
  clmul_scalar(crc, xnmodp::<S>(nbits))
}

/// Computes `x^n mod P` in the representation expected by the fusion kernels.
///
/// This is the exact algorithm used by the fast-crc32 generator output. It uses:
/// - CRC32 instructions for multiplying by `x^32`, and
/// - polynomial squaring using `vmull_p8`.
#[inline]
#[target_feature(enable = "crc,aes")]
unsafe fn xnmodp<S: Crc32PmullSpec>(mut n: u64) -> u32 {
  let mut stack = !1u64;
  while n > 191 {
    stack = (stack << 1) | (n & 1);
    n = (n >> 1).wrapping_sub(16);
  }
  stack = !stack;

  let mut acc: u32 = 0x8000_0000u32 >> (n & 31);
  n >>= 5;
  while n != 0 {
    acc = S::crc32(acc, 0);
    n = n.wrapping_sub(1);
  }

  while stack != 0 {
    let low = (stack & 1) as u32;
    stack >>= 1;
    if stack == 0 {
      break;
    }

    // Square in GF(2) using PMULL on 8-bit lanes, then reduce using CRC32.
    let x = vreinterpret_p8_u64(vmov_n_u64(acc as u64));
    let squared = vmull_p8(x, x);
    let y = vgetq_lane_u64::<0>(vreinterpretq_u64_p16(squared));
    acc = S::crc64(0, y << low);
  }

  acc
}

/// Compute CRC32-C using PMULL + CRC32C.
///
/// # Safety
/// Caller must ensure the CPU supports `aes` (PMULL) and `crc`.
#[target_feature(enable = "crc,aes")]
unsafe fn compute_pmull_unchecked_impl<S: Crc32PmullSpec>(crc: u32, data: &[u8]) -> u32 {
  // For short buffers, the CRC32 extension is hard to beat, and avoids the
  // setup overhead of the folding path.
  if data.len() < 256 {
    return S::compute_crc_unchecked(crc, data);
  }

  // Baseline PMULL folding (no EOR3). This closely matches `v12e_v1`.
  let mut crc0 = crc;
  let mut buf = data.as_ptr();
  let end = unsafe { buf.add(data.len()) };

  // Align to 8 bytes.
  while buf < end && (buf as usize & 7) != 0 {
    crc0 = S::crc8(crc0, unsafe { *buf });
    buf = unsafe { buf.add(1) };
  }

  // If only 8-byte aligned (not 16), process one 8-byte word.
  if (buf as usize & 8) != 0 {
    let remain = end.offset_from(buf) as usize;
    if remain >= 8 {
      crc0 = S::crc64(crc0, load_u64(buf));
      buf = unsafe { buf.add(8) };
    }
  }

  let mut len = end.offset_from(buf) as usize;

  if len >= 192 {
    let limit = unsafe { end.sub(192) };

    let mut x0 = vld1q_u64(buf as *const u64);
    let mut x1 = vld1q_u64(unsafe { buf.add(16) } as *const u64);
    let mut x2 = vld1q_u64(unsafe { buf.add(32) } as *const u64);
    let mut x3 = vld1q_u64(unsafe { buf.add(48) } as *const u64);
    let mut x4 = vld1q_u64(unsafe { buf.add(64) } as *const u64);
    let mut x5 = vld1q_u64(unsafe { buf.add(80) } as *const u64);
    let mut x6 = vld1q_u64(unsafe { buf.add(96) } as *const u64);
    let mut x7 = vld1q_u64(unsafe { buf.add(112) } as *const u64);
    let mut x8 = vld1q_u64(unsafe { buf.add(128) } as *const u64);
    let mut x9 = vld1q_u64(unsafe { buf.add(144) } as *const u64);
    let mut x10 = vld1q_u64(unsafe { buf.add(160) } as *const u64);
    let mut x11 = vld1q_u64(unsafe { buf.add(176) } as *const u64);

    let k_vals: [u64; 2] = [S::KEY_192.0, S::KEY_192.1];
    let mut k = vld1q_u64(k_vals.as_ptr());

    let crc_vec = vsetq_lane_u64::<0>(crc0 as u64, vmovq_n_u64(0));
    x0 = veorq_u64(crc_vec, x0);
    buf = unsafe { buf.add(192) };

    while buf <= limit {
      let y0 = clmul_lo_and_xor(x0, k, vld1q_u64(buf as *const u64));
      x0 = clmul_hi_and_xor(x0, k, y0);
      let y1 = clmul_lo_and_xor(x1, k, vld1q_u64(unsafe { buf.add(16) } as *const u64));
      x1 = clmul_hi_and_xor(x1, k, y1);
      let y2 = clmul_lo_and_xor(x2, k, vld1q_u64(unsafe { buf.add(32) } as *const u64));
      x2 = clmul_hi_and_xor(x2, k, y2);
      let y3 = clmul_lo_and_xor(x3, k, vld1q_u64(unsafe { buf.add(48) } as *const u64));
      x3 = clmul_hi_and_xor(x3, k, y3);
      let y4 = clmul_lo_and_xor(x4, k, vld1q_u64(unsafe { buf.add(64) } as *const u64));
      x4 = clmul_hi_and_xor(x4, k, y4);
      let y5 = clmul_lo_and_xor(x5, k, vld1q_u64(unsafe { buf.add(80) } as *const u64));
      x5 = clmul_hi_and_xor(x5, k, y5);
      let y6 = clmul_lo_and_xor(x6, k, vld1q_u64(unsafe { buf.add(96) } as *const u64));
      x6 = clmul_hi_and_xor(x6, k, y6);
      let y7 = clmul_lo_and_xor(x7, k, vld1q_u64(unsafe { buf.add(112) } as *const u64));
      x7 = clmul_hi_and_xor(x7, k, y7);
      let y8 = clmul_lo_and_xor(x8, k, vld1q_u64(unsafe { buf.add(128) } as *const u64));
      x8 = clmul_hi_and_xor(x8, k, y8);
      let y9 = clmul_lo_and_xor(x9, k, vld1q_u64(unsafe { buf.add(144) } as *const u64));
      x9 = clmul_hi_and_xor(x9, k, y9);
      let y10 = clmul_lo_and_xor(x10, k, vld1q_u64(unsafe { buf.add(160) } as *const u64));
      x10 = clmul_hi_and_xor(x10, k, y10);
      let y11 = clmul_lo_and_xor(x11, k, vld1q_u64(unsafe { buf.add(176) } as *const u64));
      x11 = clmul_hi_and_xor(x11, k, y11);
      buf = unsafe { buf.add(192) };
    }

    // Reduce x0..x11 -> x0.
    let k_vals: [u64; 2] = [S::KEY_16.0, S::KEY_16.1];
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

    let k_vals: [u64; 2] = [S::KEY_32.0, S::KEY_32.1];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x2);
    x0 = clmul_hi_and_xor(x0, k, y0);
    let y4 = clmul_lo_and_xor(x4, k, x6);
    x4 = clmul_hi_and_xor(x4, k, y4);
    let y8 = clmul_lo_and_xor(x8, k, x10);
    x8 = clmul_hi_and_xor(x8, k, y8);

    let k_vals: [u64; 2] = [S::KEY_64.0, S::KEY_64.1];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo_and_xor(x0, k, x4);
    x0 = clmul_hi_and_xor(x0, k, y0);
    x4 = x8;
    let y0 = clmul_lo_and_xor(x0, k, x4);
    x0 = clmul_hi_and_xor(x0, k, y0);

    crc0 = S::crc64(0, vgetq_lane_u64::<0>(x0));
    crc0 = S::crc64(crc0, vgetq_lane_u64::<1>(x0));
    len = end.offset_from(buf) as usize;
  }

  if len >= 16 {
    let mut x0 = vld1q_u64(buf as *const u64);
    let k_vals: [u64; 2] = [S::KEY_16.0, S::KEY_16.1];
    let k = vld1q_u64(k_vals.as_ptr());

    let crc_vec = vsetq_lane_u64::<0>(crc0 as u64, vmovq_n_u64(0));
    x0 = veorq_u64(crc_vec, x0);
    buf = unsafe { buf.add(16) };
    len -= 16;

    while len >= 16 {
      let y0 = clmul_lo_and_xor(x0, k, vld1q_u64(buf as *const u64));
      x0 = clmul_hi_and_xor(x0, k, y0);
      buf = unsafe { buf.add(16) };
      len -= 16;
    }

    crc0 = S::crc64(0, vgetq_lane_u64::<0>(x0));
    crc0 = S::crc64(crc0, vgetq_lane_u64::<1>(x0));
  }

  while len >= 8 {
    crc0 = S::crc64(crc0, load_u64(buf));
    buf = unsafe { buf.add(8) };
    len -= 8;
  }

  while len != 0 {
    crc0 = S::crc8(crc0, unsafe { *buf });
    buf = unsafe { buf.add(1) };
    len -= 1;
  }

  crc0
}

/// Compute CRC32-C using PMULL + CRC32C + EOR3 fusion.
///
/// This uses the `v9s3x2e_s3` fusion kernel: 9 vector lanes plus 3 scalar CRC
/// streams interleaved with the vector work.
///
/// # Safety
/// Caller must ensure the CPU supports `crc`, `aes`, and `sha3`.
#[target_feature(enable = "crc,aes,sha3")]
unsafe fn compute_pmull_eor3_unchecked_impl<S: Crc32PmullSpec>(crc: u32, data: &[u8]) -> u32 {
  // Small buffers: the CRC32 extension is extremely strong on Apple.
  if data.len() < 256 {
    return S::compute_crc_unchecked(crc, data);
  }

  let mut crc0 = crc;
  let mut buf = data.as_ptr();
  let end = unsafe { buf.add(data.len()) };

  // Align to 8-byte boundary.
  while buf < end && (buf as usize & 7) != 0 {
    crc0 = S::crc8(crc0, unsafe { *buf });
    buf = unsafe { buf.add(1) };
  }

  // Handle 8-byte alignment.
  if (buf as usize & 8) != 0 {
    let remain = end.offset_from(buf) as usize;
    if remain >= 8 {
      crc0 = S::crc64(crc0, load_u64(buf));
      buf = unsafe { buf.add(8) };
    }
  }

  let mut len = end.offset_from(buf) as usize;

  if len >= 192 {
    let blk = len / 192;
    let klen = blk.strict_mul(16);
    let vec_len = blk.strict_mul(144);

    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    let mut s0 = buf;
    let mut s1 = unsafe { buf.add(klen) };
    let mut s2 = unsafe { buf.add(klen.strict_mul(2)) };

    let mut buf2 = unsafe { buf.add(klen.strict_mul(3)) };

    let mut x0 = vld1q_u64(buf2 as *const u64);
    let mut x1 = vld1q_u64(unsafe { buf2.add(16) } as *const u64);
    let mut x2 = vld1q_u64(unsafe { buf2.add(32) } as *const u64);
    let mut x3 = vld1q_u64(unsafe { buf2.add(48) } as *const u64);
    let mut x4 = vld1q_u64(unsafe { buf2.add(64) } as *const u64);
    let mut x5 = vld1q_u64(unsafe { buf2.add(80) } as *const u64);
    let mut x6 = vld1q_u64(unsafe { buf2.add(96) } as *const u64);
    let mut x7 = vld1q_u64(unsafe { buf2.add(112) } as *const u64);
    let mut x8 = vld1q_u64(unsafe { buf2.add(128) } as *const u64);

    let k_vals: [u64; 2] = [S::KEY_144.0, S::KEY_144.1];
    let mut k = vld1q_u64(k_vals.as_ptr());
    buf2 = unsafe { buf2.add(144) };

    // `blk - 1` iterations, leaving one scalar chunk for the tail.
    let mut i = 0usize;
    while i < blk.saturating_sub(1) {
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

      {
        x0 = veor3q_u64(x0, y0, vld1q_u64(buf2 as *const u64));
        x1 = veor3q_u64(x1, y1, vld1q_u64(unsafe { buf2.add(16) } as *const u64));
        x2 = veor3q_u64(x2, y2, vld1q_u64(unsafe { buf2.add(32) } as *const u64));
        x3 = veor3q_u64(x3, y3, vld1q_u64(unsafe { buf2.add(48) } as *const u64));
        x4 = veor3q_u64(x4, y4, vld1q_u64(unsafe { buf2.add(64) } as *const u64));
        x5 = veor3q_u64(x5, y5, vld1q_u64(unsafe { buf2.add(80) } as *const u64));
        x6 = veor3q_u64(x6, y6, vld1q_u64(unsafe { buf2.add(96) } as *const u64));
        x7 = veor3q_u64(x7, y7, vld1q_u64(unsafe { buf2.add(112) } as *const u64));
        x8 = veor3q_u64(x8, y8, vld1q_u64(unsafe { buf2.add(128) } as *const u64));
      }

      crc0 = S::crc64(crc0, load_u64(s0));
      crc1 = S::crc64(crc1, load_u64(s1));
      crc2 = S::crc64(crc2, load_u64(s2));
      crc0 = S::crc64(crc0, load_u64(unsafe { s0.add(8) }));
      crc1 = S::crc64(crc1, load_u64(unsafe { s1.add(8) }));
      crc2 = S::crc64(crc2, load_u64(unsafe { s2.add(8) }));

      s0 = unsafe { s0.add(16) };
      s1 = unsafe { s1.add(16) };
      s2 = unsafe { s2.add(16) };
      buf2 = unsafe { buf2.add(144) };
      i += 1;
    }

    // Reduce x0..x8 -> x0.
    let k_vals: [u64; 2] = [S::KEY_16.0, S::KEY_16.1];
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

    let k_vals: [u64; 2] = [S::KEY_32.0, S::KEY_32.1];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    let y4 = clmul_lo(x4, k);
    x4 = clmul_hi(x4, k);

    x0 = veor3q_u64(x0, y0, x2);
    x4 = veor3q_u64(x4, y4, x6);

    let k_vals: [u64; 2] = [S::KEY_64.0, S::KEY_64.1];
    k = vld1q_u64(k_vals.as_ptr());

    let y0 = clmul_lo(x0, k);
    x0 = clmul_hi(x0, k);
    x0 = veor3q_u64(x0, y0, x4);

    // Final scalar chunk (last 16 bytes of each scalar stream).
    crc0 = S::crc64(crc0, load_u64(s0));
    crc1 = S::crc64(crc1, load_u64(s1));
    crc2 = S::crc64(crc2, load_u64(s2));
    crc0 = S::crc64(crc0, load_u64(unsafe { s0.add(8) }));
    crc1 = S::crc64(crc1, load_u64(unsafe { s1.add(8) }));
    crc2 = S::crc64(crc2, load_u64(unsafe { s2.add(8) }));

    // Shift and combine scalar CRC streams into a 64-bit tweak.
    let vc0 = crc32_shift::<S>(crc0, klen.strict_mul(2).strict_add(vec_len));
    let vc1 = crc32_shift::<S>(crc1, klen.strict_add(vec_len));
    let vc2 = crc32_shift::<S>(crc2, vec_len);
    let vc = vgetq_lane_u64::<0>(veor3q_u64(vc0, vc1, vc2));

    // Reduce 128 bits -> 32 and incorporate the scalar streams.
    crc0 = S::crc64(0, vgetq_lane_u64::<0>(x0));
    crc0 = S::crc64(crc0, vc ^ vgetq_lane_u64::<1>(x0));

    // Advance to the remaining bytes.
    buf = buf2;
    len = end.offset_from(buf) as usize;
  }

  if len >= 32 {
    let klen = ((len - 8) / 24).strict_mul(8);
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    loop {
      crc0 = S::crc64(crc0, load_u64(buf));
      crc1 = S::crc64(crc1, load_u64(unsafe { buf.add(klen) }));
      crc2 = S::crc64(crc2, load_u64(unsafe { buf.add(klen.strict_mul(2)) }));
      buf = unsafe { buf.add(8) };
      len -= 24;
      if len < 32 {
        break;
      }
    }

    let vc0 = crc32_shift::<S>(crc0, klen.strict_mul(2).strict_add(8));
    let vc1 = crc32_shift::<S>(crc1, klen.strict_add(8));
    let vc = vgetq_lane_u64::<0>(veorq_u64(vc0, vc1));

    buf = unsafe { buf.add(klen.strict_mul(2)) };
    crc0 = crc2;
    crc0 = S::crc64(crc0, load_u64(buf) ^ vc);
    buf = unsafe { buf.add(8) };
    len -= 8;
  }

  while len >= 8 {
    crc0 = S::crc64(crc0, load_u64(buf));
    buf = unsafe { buf.add(8) };
    len -= 8;
  }

  while len != 0 {
    crc0 = S::crc8(crc0, unsafe { *buf });
    buf = unsafe { buf.add(1) };
    len -= 1;
  }

  crc0
}

#[target_feature(enable = "crc,aes")]
pub(crate) unsafe fn compute_pmull_unchecked(crc: u32, data: &[u8]) -> u32 {
  compute_pmull_unchecked_impl::<Crc32cPmullSpec>(crc, data)
}

#[target_feature(enable = "crc,aes,sha3")]
pub(crate) unsafe fn compute_pmull_eor3_unchecked(crc: u32, data: &[u8]) -> u32 {
  compute_pmull_eor3_unchecked_impl::<Crc32cPmullSpec>(crc, data)
}

#[target_feature(enable = "crc,aes")]
pub(crate) unsafe fn compute_pmull_crc32_unchecked(crc: u32, data: &[u8]) -> u32 {
  compute_pmull_unchecked_impl::<Crc32IsoHdlcPmullSpec>(crc, data)
}

#[target_feature(enable = "crc,aes,sha3")]
pub(crate) unsafe fn compute_pmull_eor3_crc32_unchecked(crc: u32, data: &[u8]) -> u32 {
  compute_pmull_eor3_unchecked_impl::<Crc32IsoHdlcPmullSpec>(crc, data)
}

#[cfg(all(target_feature = "aes", target_feature = "crc"))]
#[inline]
pub(crate) fn compute_pmull_enabled(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pmull_unchecked(crc, data) }
}

#[cfg(all(target_feature = "aes", target_feature = "crc", target_feature = "sha3"))]
#[inline]
pub(crate) fn compute_pmull_eor3_enabled(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pmull_eor3_unchecked(crc, data) }
}

#[cfg(all(feature = "std", not(all(target_feature = "aes", target_feature = "crc"))))]
#[inline]
pub(crate) fn compute_pmull_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: callers must gate this with `is_aarch64_feature_detected!("aes")`
  // and `is_aarch64_feature_detected!("crc")`.
  unsafe { compute_pmull_unchecked(crc, data) }
}

#[cfg(all(
  feature = "std",
  not(all(target_feature = "aes", target_feature = "crc", target_feature = "sha3"))
))]
#[inline]
pub(crate) fn compute_pmull_eor3_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: callers must gate this with `is_aarch64_feature_detected!("aes")`,
  // `is_aarch64_feature_detected!("crc")`, and `is_aarch64_feature_detected!("sha3")`.
  unsafe { compute_pmull_eor3_unchecked(crc, data) }
}

#[cfg(all(target_feature = "aes", target_feature = "crc"))]
#[inline]
pub(crate) fn compute_pmull_crc32_enabled(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pmull_crc32_unchecked(crc, data) }
}

#[cfg(all(target_feature = "aes", target_feature = "crc", target_feature = "sha3"))]
#[inline]
pub(crate) fn compute_pmull_eor3_crc32_enabled(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pmull_eor3_crc32_unchecked(crc, data) }
}

#[cfg(all(feature = "std", not(all(target_feature = "aes", target_feature = "crc"))))]
#[inline]
pub(crate) fn compute_pmull_crc32_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: callers must gate this with `is_aarch64_feature_detected!("aes")`
  // and `is_aarch64_feature_detected!("crc")`.
  unsafe { compute_pmull_crc32_unchecked(crc, data) }
}

#[cfg(all(
  feature = "std",
  not(all(target_feature = "aes", target_feature = "crc", target_feature = "sha3"))
))]
#[inline]
pub(crate) fn compute_pmull_eor3_crc32_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: callers must gate this with `is_aarch64_feature_detected!("aes")`,
  // `is_aarch64_feature_detected!("crc")`, and `is_aarch64_feature_detected!("sha3")`.
  unsafe { compute_pmull_eor3_crc32_unchecked(crc, data) }
}

// ============================================================================
// CRC64 PMULL Support
// ============================================================================

/// Trait for CRC64 polynomial-specific PMULL folding constants.
///
/// Unlike CRC32, CRC64 has no hardware instruction on ARM, so we use
/// Barrett reduction for finalization. We use the same COEFF constants
/// as x86_64 PCLMULQDQ. The constants are loaded such that:
/// - lane 0 (low) = COEFF.0 = key(D+8)
/// - lane 1 (high) = COEFF.1 = key(D)
///
/// With clmul_lo and clmul_hi doing same-lane multiplication, this achieves
/// the cross-multiplication pattern equivalent to x86_64 PCLMULQDQ.
pub(crate) trait Crc64PmullSpec {
  /// Folding coefficient for 192-byte distance (12 accumulators main loop).
  const COEFF_192: (u64, u64);
  /// Folding coefficient for 176-byte distance (reduction: x0 → x11).
  const COEFF_176: (u64, u64);
  /// Folding coefficient for 160-byte distance (reduction: x1 → x11).
  const COEFF_160: (u64, u64);
  /// Folding coefficient for 144-byte distance (reduction: x2 → x11).
  const COEFF_144: (u64, u64);
  /// Folding coefficient for 128-byte distance (8 accumulators / reduction: x3 → x11).
  const COEFF_128: (u64, u64);
  /// Folding coefficient for 112-byte distance (reduction: x4 → x11).
  const COEFF_112: (u64, u64);
  /// Folding coefficient for 96-byte distance (reduction: x5 → x11).
  const COEFF_96: (u64, u64);
  /// Folding coefficient for 80-byte distance (reduction: x6 → x11).
  const COEFF_80: (u64, u64);
  /// Folding coefficient for 64-byte distance (reduction: x7 → x11).
  const COEFF_64: (u64, u64);
  /// Folding coefficient for 48-byte distance (reduction: x8 → x11).
  const COEFF_48: (u64, u64);
  /// Folding coefficient for 32-byte distance (reduction: x9 → x11).
  const COEFF_32: (u64, u64);
  /// Folding coefficient for 16-byte distance (reduction: x10 → x11).
  const COEFF_16: (u64, u64);

  /// Barrett reduction constants: (poly_simd, mu)
  const BARRETT: (u64, u64);

  /// Fold-width key for 128->64 bit reduction
  const FOLD_KEY: u64;

  fn portable(crc: u64, data: &[u8]) -> u64;
}

pub(crate) struct Crc64XzPmullSpec;

impl Crc64PmullSpec for Crc64XzPmullSpec {
  const COEFF_192: (u64, u64) = crate::constants::crc64::fold::COEFF_192;
  const COEFF_176: (u64, u64) = crate::constants::crc64::fold::COEFF_176;
  const COEFF_160: (u64, u64) = crate::constants::crc64::fold::COEFF_160;
  const COEFF_144: (u64, u64) = crate::constants::crc64::fold::COEFF_144;
  const COEFF_128: (u64, u64) = crate::constants::crc64::fold::COEFF_128;
  const COEFF_112: (u64, u64) = crate::constants::crc64::fold::COEFF_112;
  const COEFF_96: (u64, u64) = crate::constants::crc64::fold::COEFF_96;
  const COEFF_80: (u64, u64) = crate::constants::crc64::fold::COEFF_80;
  const COEFF_64: (u64, u64) = crate::constants::crc64::fold::COEFF_64;
  const COEFF_48: (u64, u64) = crate::constants::crc64::fold::COEFF_48;
  const COEFF_32: (u64, u64) = crate::constants::crc64::fold::COEFF_32;
  const COEFF_16: (u64, u64) = crate::constants::crc64::fold::COEFF_16;
  const BARRETT: (u64, u64) = crate::constants::crc64::fold::BARRETT;
  const FOLD_KEY: u64 = crate::constants::crc64::fold::KEY_16;

  #[inline(always)]
  fn portable(crc: u64, data: &[u8]) -> u64 {
    crate::crc64::xz::compute_portable(crc, data)
  }
}

pub(crate) struct Crc64NvmePmullSpec;

impl Crc64PmullSpec for Crc64NvmePmullSpec {
  const COEFF_192: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_192;
  const COEFF_176: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_176;
  const COEFF_160: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_160;
  const COEFF_144: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_144;
  const COEFF_128: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_128;
  const COEFF_112: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_112;
  const COEFF_96: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_96;
  const COEFF_80: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_80;
  const COEFF_64: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_64;
  const COEFF_48: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_48;
  const COEFF_32: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_32;
  const COEFF_16: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_16;
  const BARRETT: (u64, u64) = crate::constants::crc64_nvme::fold::BARRETT;
  const FOLD_KEY: u64 = crate::constants::crc64_nvme::fold::KEY_16;

  #[inline(always)]
  fn portable(crc: u64, data: &[u8]) -> u64 {
    crate::crc64::nvme::compute_portable(crc, data)
  }
}

/// Fold 16 bytes using parallel multiplication for CRC64.
///
/// This uses (lo×lo) and (hi×hi), matching crc-fast-rust.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn fold16_parallel(current: uint64x2_t, coeff: uint64x2_t, data: uint64x2_t) -> uint64x2_t {
  // CRC-64 folding uses "parallel" carryless multiplications:
  // - current.lo × coeff.lo
  // - current.hi × coeff.hi
  let lo = vmull_p64(vgetq_lane_u64::<0>(current), vgetq_lane_u64::<0>(coeff));
  let lo = vreinterpretq_u64_p128(lo);
  let hi = vmull_p64(vgetq_lane_u64::<1>(current), vgetq_lane_u64::<1>(coeff));
  let hi = vreinterpretq_u64_p128(hi);
  veorq_u64(veorq_u64(lo, hi), data)
}

/// Fold 16 bytes using parallel multiplication for CRC64 with EOR3.
///
/// Same as `fold16_parallel` but uses 3-way XOR (EOR3) for better throughput.
#[inline]
#[target_feature(enable = "aes,sha3")]
unsafe fn fold16_parallel_eor3(current: uint64x2_t, coeff: uint64x2_t, data: uint64x2_t) -> uint64x2_t {
  let lo = vmull_p64(vgetq_lane_u64::<0>(current), vgetq_lane_u64::<0>(coeff));
  let lo = vreinterpretq_u64_p128(lo);
  let hi = vmull_p64(vgetq_lane_u64::<1>(current), vgetq_lane_u64::<1>(coeff));
  let hi = vreinterpretq_u64_p128(hi);
  veor3q_u64(lo, hi, data)
}

/// Barrett reduction from 128-bit folded value to 64-bit CRC.
///
/// For reflected CRC64, following the crc-fast-rust algorithm:
/// 1. Fold 128 bits to 64 bits using key × state.lo (crc-fast-rust uses carryless_mul_01)
/// 2. Apply Barrett reduction: multiply by poly first, then by mu
///
/// # Safety
/// Caller must ensure the CPU supports `aes` (PMULL).
#[inline]
#[target_feature(enable = "aes")]
unsafe fn barrett_reduce_crc64<S: Crc64PmullSpec>(x: uint64x2_t) -> u64 {
  let key16 = S::FOLD_KEY;
  let (poly_simd, mu) = S::BARRETT;

  // Step 1: Fold 128 bits to 64 bits (crc-fast-rust fold_width for reflected)
  // crc-fast-rust does: h = carryless_mul_01(coeff, state) = coeff.hi × state.lo
  // where coeff.hi = key16. So: h = key16 × x.lo (NOT x.hi!)
  let h = vmull_p64(key16, vgetq_lane_u64::<0>(x));
  let h = vreinterpretq_u64_p128(h);
  // shifted = x >> 64 (shift_right_8 moves hi to lo, zeroes hi)
  let shifted = _aarch64_shr_64(x);
  // folded = h XOR shifted
  let folded = veorq_u64(h, shifted);

  // Step 2: Barrett reduction (crc-fast-rust order: mu first, then poly)
  // clmul1 = folded.lo × mu
  let clmul1 = vmull_p64(vgetq_lane_u64::<0>(folded), mu);
  let clmul1 = vreinterpretq_u64_p128(clmul1);

  // clmul2 = clmul1.lo × poly
  let clmul2 = vmull_p64(vgetq_lane_u64::<0>(clmul1), poly_simd);
  let clmul2 = vreinterpretq_u64_p128(clmul2);

  // Handle the x^64 term: (clmul1 << 64)
  let clmul1_shifted = _aarch64_shl_64(clmul1);

  // result = clmul2 ⊕ (clmul1 << 64) ⊕ folded
  let result = veorq_u64(veorq_u64(clmul2, clmul1_shifted), folded);

  // Extract HIGH 64 bits (this is the final CRC)
  vgetq_lane_u64::<1>(result)
}

/// Barrett reduction from 128-bit folded value to 64-bit CRC with EOR3.
///
/// Same as `barrett_reduce_crc64` but uses 3-way XOR (EOR3) for better throughput.
#[inline]
#[target_feature(enable = "aes,sha3")]
unsafe fn barrett_reduce_crc64_eor3<S: Crc64PmullSpec>(x: uint64x2_t) -> u64 {
  let key16 = S::FOLD_KEY;
  let (poly_simd, mu) = S::BARRETT;

  // Step 1: Fold 128 bits to 64 bits
  let h = vmull_p64(key16, vgetq_lane_u64::<0>(x));
  let h = vreinterpretq_u64_p128(h);
  let shifted = _aarch64_shr_64(x);
  let folded = veorq_u64(h, shifted);

  // Step 2: Barrett reduction
  let clmul1 = vmull_p64(vgetq_lane_u64::<0>(folded), mu);
  let clmul1 = vreinterpretq_u64_p128(clmul1);

  let clmul2 = vmull_p64(vgetq_lane_u64::<0>(clmul1), poly_simd);
  let clmul2 = vreinterpretq_u64_p128(clmul2);

  let clmul1_shifted = _aarch64_shl_64(clmul1);

  // Use EOR3 for 3-way XOR
  let result = veor3q_u64(clmul2, clmul1_shifted, folded);

  vgetq_lane_u64::<1>(result)
}

/// Shift vector right by 64 bits (move hi to lo, zero out hi).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn _aarch64_shr_64(x: uint64x2_t) -> uint64x2_t {
  use core::arch::aarch64::vextq_u64;
  let zero = vmovq_n_u64(0);
  vextq_u64(x, zero, 1) // Extract lane 1 to lane 0, fill lane 1 with zero
}

/// Shift vector left by 64 bits (move lo to hi, zero out lo).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn _aarch64_shl_64(x: uint64x2_t) -> uint64x2_t {
  use core::arch::aarch64::vextq_u64;
  let zero = vmovq_n_u64(0);
  vextq_u64(zero, x, 1) // Extract lane 0 from x to lane 1, lane 0 is zero
}

/// Helper to load coefficient vector from tuple.
#[inline(always)]
unsafe fn load_coeff_crc64(coeff: (u64, u64)) -> uint64x2_t {
  let (high, low) = coeff;
  let vals: [u64; 2] = [high, low];
  vld1q_u64(vals.as_ptr())
}

/// Compute CRC64 using PMULL with 12-lane parallelism for large buffers (>= 192 bytes).
///
/// # Safety
/// Caller must ensure the CPU supports `aes` (PMULL) and data.len() >= 192.
#[cfg(any(
  all(target_feature = "aes", not(target_feature = "sha3")),
  all(feature = "std", not(target_feature = "aes"))
))]
#[target_feature(enable = "aes")]
unsafe fn compute_pmull_crc64_12lane<S: Crc64PmullSpec>(crc: u64, data: &[u8]) -> u64 {
  debug_assert!(data.len() >= 192);

  let mut buf = data.as_ptr();
  let end = buf.add(data.len());

  // Process bulk in 192-byte chunks (12 × 16-byte vectors)
  // Note: 192 is not a power of 2, so we can't use a bitmask
  let bulk_len = (data.len() / 192) * 192;
  let rem_len = data.len() - bulk_len;

  // Main loop coefficient (192-byte folding distance)
  let coeff_192 = load_coeff_crc64(S::COEFF_192);

  // Reduction coefficients (12 → 1)
  let coeff_176 = load_coeff_crc64(S::COEFF_176);
  let coeff_160 = load_coeff_crc64(S::COEFF_160);
  let coeff_144 = load_coeff_crc64(S::COEFF_144);
  let coeff_128 = load_coeff_crc64(S::COEFF_128);
  let coeff_112 = load_coeff_crc64(S::COEFF_112);
  let coeff_96 = load_coeff_crc64(S::COEFF_96);
  let coeff_80 = load_coeff_crc64(S::COEFF_80);
  let coeff_64 = load_coeff_crc64(S::COEFF_64);
  let coeff_48 = load_coeff_crc64(S::COEFF_48);
  let coeff_32 = load_coeff_crc64(S::COEFF_32);
  let coeff_16 = load_coeff_crc64(S::COEFF_16);

  // Load first 192 bytes (12 × 16-byte vectors)
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

  // XOR initial CRC into low 64 bits of first block
  x0 = veorq_u64(x0, vsetq_lane_u64::<0>(crc, vmovq_n_u64(0)));
  buf = buf.add(192);

  // Main folding loop: process 192 bytes per iteration with 12 parallel accumulators
  let limit = end.sub(rem_len).sub(192);
  while buf <= limit {
    x0 = fold16_parallel(x0, coeff_192, vld1q_u64(buf as *const u64));
    x1 = fold16_parallel(x1, coeff_192, vld1q_u64(buf.add(16) as *const u64));
    x2 = fold16_parallel(x2, coeff_192, vld1q_u64(buf.add(32) as *const u64));
    x3 = fold16_parallel(x3, coeff_192, vld1q_u64(buf.add(48) as *const u64));
    x4 = fold16_parallel(x4, coeff_192, vld1q_u64(buf.add(64) as *const u64));
    x5 = fold16_parallel(x5, coeff_192, vld1q_u64(buf.add(80) as *const u64));
    x6 = fold16_parallel(x6, coeff_192, vld1q_u64(buf.add(96) as *const u64));
    x7 = fold16_parallel(x7, coeff_192, vld1q_u64(buf.add(112) as *const u64));
    x8 = fold16_parallel(x8, coeff_192, vld1q_u64(buf.add(128) as *const u64));
    x9 = fold16_parallel(x9, coeff_192, vld1q_u64(buf.add(144) as *const u64));
    x10 = fold16_parallel(x10, coeff_192, vld1q_u64(buf.add(160) as *const u64));
    x11 = fold16_parallel(x11, coeff_192, vld1q_u64(buf.add(176) as *const u64));
    buf = buf.add(192);
  }

  // Reduce 12 accumulators → 1 using parallel multiplication
  let mut folded = x11;
  folded = fold16_parallel(x10, coeff_16, folded); // 16 bytes back
  folded = fold16_parallel(x9, coeff_32, folded); // 32 bytes back
  folded = fold16_parallel(x8, coeff_48, folded); // 48 bytes back
  folded = fold16_parallel(x7, coeff_64, folded); // 64 bytes back
  folded = fold16_parallel(x6, coeff_80, folded); // 80 bytes back
  folded = fold16_parallel(x5, coeff_96, folded); // 96 bytes back
  folded = fold16_parallel(x4, coeff_112, folded); // 112 bytes back
  folded = fold16_parallel(x3, coeff_128, folded); // 128 bytes back
  folded = fold16_parallel(x2, coeff_144, folded); // 144 bytes back
  folded = fold16_parallel(x1, coeff_160, folded); // 160 bytes back
  folded = fold16_parallel(x0, coeff_176, folded); // 176 bytes back

  // Barrett reduction
  let crc0 = barrett_reduce_crc64::<S>(folded);

  // Handle remainder with portable
  if rem_len > 0 {
    return S::portable(crc0, core::slice::from_raw_parts(buf, rem_len));
  }

  crc0
}

/// Compute CRC64 using PMULL with 4-lane parallelism for medium-sized buffers (64-127 bytes).
///
/// # Safety
/// Caller must ensure the CPU supports `aes` (PMULL) and data.len() >= 64.
#[target_feature(enable = "aes")]
unsafe fn compute_pmull_crc64_4lane<S: Crc64PmullSpec>(crc: u64, data: &[u8]) -> u64 {
  debug_assert!(data.len() >= 64);

  let mut buf = data.as_ptr();
  let end = buf.add(data.len());

  // Process bulk in 64-byte chunks
  let bulk_len = data.len() & !63;
  let rem_len = data.len() - bulk_len;

  let coeff_64 = load_coeff_crc64(S::COEFF_64);
  let coeff_48 = load_coeff_crc64(S::COEFF_48);
  let coeff_32 = load_coeff_crc64(S::COEFF_32);
  let coeff_16 = load_coeff_crc64(S::COEFF_16);

  // Load first 64 bytes
  let mut x0 = vld1q_u64(buf as *const u64);
  let mut x1 = vld1q_u64(buf.add(16) as *const u64);
  let mut x2 = vld1q_u64(buf.add(32) as *const u64);
  let mut x3 = vld1q_u64(buf.add(48) as *const u64);

  // XOR initial CRC into low 64 bits of first block
  x0 = veorq_u64(x0, vsetq_lane_u64::<0>(crc, vmovq_n_u64(0)));
  buf = buf.add(64);

  // Main folding loop with parallel multiplication
  let limit = end.sub(rem_len).sub(64);
  while buf <= limit {
    x0 = fold16_parallel(x0, coeff_64, vld1q_u64(buf as *const u64));
    x1 = fold16_parallel(x1, coeff_64, vld1q_u64(buf.add(16) as *const u64));
    x2 = fold16_parallel(x2, coeff_64, vld1q_u64(buf.add(32) as *const u64));
    x3 = fold16_parallel(x3, coeff_64, vld1q_u64(buf.add(48) as *const u64));
    buf = buf.add(64);
  }

  // Reduce 4 accumulators → 1
  let mut folded = x3;
  folded = fold16_parallel(x2, coeff_16, folded);
  folded = fold16_parallel(x1, coeff_32, folded);
  folded = fold16_parallel(x0, coeff_48, folded);

  // Barrett reduction
  let crc0 = barrett_reduce_crc64::<S>(folded);

  // Handle remainder with portable
  if rem_len > 0 {
    return S::portable(crc0, core::slice::from_raw_parts(buf, rem_len));
  }

  crc0
}

/// Compute CRC64 using PMULL folding + Barrett reduction with tiered parallelism.
///
/// This implementation uses different accumulator counts based on buffer size:
/// - >= 192 bytes: 12 parallel accumulators (192 bytes per iteration)
/// - 128-191 bytes: 8 parallel accumulators (128 bytes per iteration)
/// - 64-127 bytes: 4 parallel accumulators (64 bytes per iteration)
/// - < 64 bytes: portable implementation
///
/// # Safety
/// Caller must ensure the CPU supports `aes` (PMULL).
#[cfg(any(
  all(target_feature = "aes", not(target_feature = "sha3")),
  all(feature = "std", not(target_feature = "aes"))
))]
#[target_feature(enable = "aes")]
unsafe fn compute_pmull_crc64_unchecked_impl<S: Crc64PmullSpec>(crc: u64, data: &[u8]) -> u64 {
  // Tiered dispatch based on buffer size.
  //
  // 12-lane (192 bytes/iteration) has high setup overhead and doesn't align
  // well with power-of-2 buffer sizes. It only wins at very large sizes where
  // the extra parallelism amortizes the overhead. Empirically, the crossover
  // is around 32KB-64KB.
  //
  // 8-lane (128 bytes/iteration) aligns perfectly with power-of-2 sizes and
  // has much lower overhead, making it faster for small-to-medium buffers.
  const THRESHOLD_12LANE: usize = 32 * 1024; // 32KB

  if data.len() < 64 {
    S::portable(crc, data)
  } else if data.len() < 128 {
    compute_pmull_crc64_4lane::<S>(crc, data)
  } else if data.len() < THRESHOLD_12LANE {
    compute_pmull_crc64_8lane::<S>(crc, data)
  } else {
    compute_pmull_crc64_12lane::<S>(crc, data)
  }
}

/// Compute CRC64 using PMULL with 8-lane parallelism for buffers 128-191 bytes.
///
/// # Safety
/// Caller must ensure the CPU supports `aes` (PMULL) and data.len() >= 128.
#[cfg(any(
  all(target_feature = "aes", not(target_feature = "sha3")),
  all(feature = "std", not(target_feature = "aes"))
))]
#[target_feature(enable = "aes")]
unsafe fn compute_pmull_crc64_8lane<S: Crc64PmullSpec>(crc: u64, data: &[u8]) -> u64 {
  debug_assert!(data.len() >= 128);

  let mut buf = data.as_ptr();
  let end = buf.add(data.len());

  // Process bulk in 128-byte chunks (8 × 16-byte vectors)
  let bulk_len = data.len() & !127;
  let rem_len = data.len() - bulk_len;

  // Main loop coefficient (128-byte folding distance)
  let coeff_128 = load_coeff_crc64(S::COEFF_128);

  // Reduction coefficients (8 → 1)
  let coeff_112 = load_coeff_crc64(S::COEFF_112);
  let coeff_96 = load_coeff_crc64(S::COEFF_96);
  let coeff_80 = load_coeff_crc64(S::COEFF_80);
  let coeff_64 = load_coeff_crc64(S::COEFF_64);
  let coeff_48 = load_coeff_crc64(S::COEFF_48);
  let coeff_32 = load_coeff_crc64(S::COEFF_32);
  let coeff_16 = load_coeff_crc64(S::COEFF_16);

  // Load first 128 bytes (8 × 16-byte vectors)
  let mut x0 = vld1q_u64(buf as *const u64);
  let mut x1 = vld1q_u64(buf.add(16) as *const u64);
  let mut x2 = vld1q_u64(buf.add(32) as *const u64);
  let mut x3 = vld1q_u64(buf.add(48) as *const u64);
  let mut x4 = vld1q_u64(buf.add(64) as *const u64);
  let mut x5 = vld1q_u64(buf.add(80) as *const u64);
  let mut x6 = vld1q_u64(buf.add(96) as *const u64);
  let mut x7 = vld1q_u64(buf.add(112) as *const u64);

  // XOR initial CRC into low 64 bits of first block
  x0 = veorq_u64(x0, vsetq_lane_u64::<0>(crc, vmovq_n_u64(0)));
  buf = buf.add(128);

  // Main folding loop: process 128 bytes per iteration with 8 parallel accumulators
  let limit = end.sub(rem_len).sub(128);
  while buf <= limit {
    x0 = fold16_parallel(x0, coeff_128, vld1q_u64(buf as *const u64));
    x1 = fold16_parallel(x1, coeff_128, vld1q_u64(buf.add(16) as *const u64));
    x2 = fold16_parallel(x2, coeff_128, vld1q_u64(buf.add(32) as *const u64));
    x3 = fold16_parallel(x3, coeff_128, vld1q_u64(buf.add(48) as *const u64));
    x4 = fold16_parallel(x4, coeff_128, vld1q_u64(buf.add(64) as *const u64));
    x5 = fold16_parallel(x5, coeff_128, vld1q_u64(buf.add(80) as *const u64));
    x6 = fold16_parallel(x6, coeff_128, vld1q_u64(buf.add(96) as *const u64));
    x7 = fold16_parallel(x7, coeff_128, vld1q_u64(buf.add(112) as *const u64));
    buf = buf.add(128);
  }

  // Reduce 8 accumulators → 1
  let mut folded = x7;
  folded = fold16_parallel(x6, coeff_16, folded);
  folded = fold16_parallel(x5, coeff_32, folded);
  folded = fold16_parallel(x4, coeff_48, folded);
  folded = fold16_parallel(x3, coeff_64, folded);
  folded = fold16_parallel(x2, coeff_80, folded);
  folded = fold16_parallel(x1, coeff_96, folded);
  folded = fold16_parallel(x0, coeff_112, folded);

  // Barrett reduction
  let crc0 = barrett_reduce_crc64::<S>(folded);

  // Handle remainder with portable
  if rem_len > 0 {
    return S::portable(crc0, core::slice::from_raw_parts(buf, rem_len));
  }

  crc0
}

// ============================================================================
// EOR3-accelerated CRC64 implementations (sha3 feature)
// ============================================================================

/// Compute CRC64 using PMULL with 8-lane parallelism and EOR3.
#[target_feature(enable = "aes,sha3")]
unsafe fn compute_pmull_crc64_8lane_eor3<S: Crc64PmullSpec>(crc: u64, data: &[u8]) -> u64 {
  debug_assert!(data.len() >= 128);

  let mut buf = data.as_ptr();
  let end = buf.add(data.len());

  let bulk_len = data.len() & !127;
  let rem_len = data.len() - bulk_len;

  let coeff_128 = load_coeff_crc64(S::COEFF_128);
  let coeff_112 = load_coeff_crc64(S::COEFF_112);
  let coeff_96 = load_coeff_crc64(S::COEFF_96);
  let coeff_80 = load_coeff_crc64(S::COEFF_80);
  let coeff_64 = load_coeff_crc64(S::COEFF_64);
  let coeff_48 = load_coeff_crc64(S::COEFF_48);
  let coeff_32 = load_coeff_crc64(S::COEFF_32);
  let coeff_16 = load_coeff_crc64(S::COEFF_16);

  let mut x0 = vld1q_u64(buf as *const u64);
  let mut x1 = vld1q_u64(buf.add(16) as *const u64);
  let mut x2 = vld1q_u64(buf.add(32) as *const u64);
  let mut x3 = vld1q_u64(buf.add(48) as *const u64);
  let mut x4 = vld1q_u64(buf.add(64) as *const u64);
  let mut x5 = vld1q_u64(buf.add(80) as *const u64);
  let mut x6 = vld1q_u64(buf.add(96) as *const u64);
  let mut x7 = vld1q_u64(buf.add(112) as *const u64);

  x0 = veorq_u64(x0, vsetq_lane_u64::<0>(crc, vmovq_n_u64(0)));
  buf = buf.add(128);

  let limit = end.sub(rem_len).sub(128);
  while buf <= limit {
    x0 = fold16_parallel_eor3(x0, coeff_128, vld1q_u64(buf as *const u64));
    x1 = fold16_parallel_eor3(x1, coeff_128, vld1q_u64(buf.add(16) as *const u64));
    x2 = fold16_parallel_eor3(x2, coeff_128, vld1q_u64(buf.add(32) as *const u64));
    x3 = fold16_parallel_eor3(x3, coeff_128, vld1q_u64(buf.add(48) as *const u64));
    x4 = fold16_parallel_eor3(x4, coeff_128, vld1q_u64(buf.add(64) as *const u64));
    x5 = fold16_parallel_eor3(x5, coeff_128, vld1q_u64(buf.add(80) as *const u64));
    x6 = fold16_parallel_eor3(x6, coeff_128, vld1q_u64(buf.add(96) as *const u64));
    x7 = fold16_parallel_eor3(x7, coeff_128, vld1q_u64(buf.add(112) as *const u64));
    buf = buf.add(128);
  }

  let mut folded = x7;
  folded = fold16_parallel_eor3(x6, coeff_16, folded);
  folded = fold16_parallel_eor3(x5, coeff_32, folded);
  folded = fold16_parallel_eor3(x4, coeff_48, folded);
  folded = fold16_parallel_eor3(x3, coeff_64, folded);
  folded = fold16_parallel_eor3(x2, coeff_80, folded);
  folded = fold16_parallel_eor3(x1, coeff_96, folded);
  folded = fold16_parallel_eor3(x0, coeff_112, folded);

  let crc0 = barrett_reduce_crc64_eor3::<S>(folded);

  if rem_len > 0 {
    return S::portable(crc0, core::slice::from_raw_parts(buf, rem_len));
  }

  crc0
}

/// Compute CRC64 using PMULL with 12-lane parallelism and EOR3.
#[target_feature(enable = "aes,sha3")]
unsafe fn compute_pmull_crc64_12lane_eor3<S: Crc64PmullSpec>(crc: u64, data: &[u8]) -> u64 {
  debug_assert!(data.len() >= 192);

  let mut buf = data.as_ptr();
  let end = buf.add(data.len());

  let bulk_len = (data.len() / 192) * 192;
  let rem_len = data.len() - bulk_len;

  let coeff_192 = load_coeff_crc64(S::COEFF_192);
  let coeff_176 = load_coeff_crc64(S::COEFF_176);
  let coeff_160 = load_coeff_crc64(S::COEFF_160);
  let coeff_144 = load_coeff_crc64(S::COEFF_144);
  let coeff_128 = load_coeff_crc64(S::COEFF_128);
  let coeff_112 = load_coeff_crc64(S::COEFF_112);
  let coeff_96 = load_coeff_crc64(S::COEFF_96);
  let coeff_80 = load_coeff_crc64(S::COEFF_80);
  let coeff_64 = load_coeff_crc64(S::COEFF_64);
  let coeff_48 = load_coeff_crc64(S::COEFF_48);
  let coeff_32 = load_coeff_crc64(S::COEFF_32);
  let coeff_16 = load_coeff_crc64(S::COEFF_16);

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

  x0 = veorq_u64(x0, vsetq_lane_u64::<0>(crc, vmovq_n_u64(0)));
  buf = buf.add(192);

  let limit = end.sub(rem_len).sub(192);
  while buf <= limit {
    x0 = fold16_parallel_eor3(x0, coeff_192, vld1q_u64(buf as *const u64));
    x1 = fold16_parallel_eor3(x1, coeff_192, vld1q_u64(buf.add(16) as *const u64));
    x2 = fold16_parallel_eor3(x2, coeff_192, vld1q_u64(buf.add(32) as *const u64));
    x3 = fold16_parallel_eor3(x3, coeff_192, vld1q_u64(buf.add(48) as *const u64));
    x4 = fold16_parallel_eor3(x4, coeff_192, vld1q_u64(buf.add(64) as *const u64));
    x5 = fold16_parallel_eor3(x5, coeff_192, vld1q_u64(buf.add(80) as *const u64));
    x6 = fold16_parallel_eor3(x6, coeff_192, vld1q_u64(buf.add(96) as *const u64));
    x7 = fold16_parallel_eor3(x7, coeff_192, vld1q_u64(buf.add(112) as *const u64));
    x8 = fold16_parallel_eor3(x8, coeff_192, vld1q_u64(buf.add(128) as *const u64));
    x9 = fold16_parallel_eor3(x9, coeff_192, vld1q_u64(buf.add(144) as *const u64));
    x10 = fold16_parallel_eor3(x10, coeff_192, vld1q_u64(buf.add(160) as *const u64));
    x11 = fold16_parallel_eor3(x11, coeff_192, vld1q_u64(buf.add(176) as *const u64));
    buf = buf.add(192);
  }

  let mut folded = x11;
  folded = fold16_parallel_eor3(x10, coeff_16, folded);
  folded = fold16_parallel_eor3(x9, coeff_32, folded);
  folded = fold16_parallel_eor3(x8, coeff_48, folded);
  folded = fold16_parallel_eor3(x7, coeff_64, folded);
  folded = fold16_parallel_eor3(x6, coeff_80, folded);
  folded = fold16_parallel_eor3(x5, coeff_96, folded);
  folded = fold16_parallel_eor3(x4, coeff_112, folded);
  folded = fold16_parallel_eor3(x3, coeff_128, folded);
  folded = fold16_parallel_eor3(x2, coeff_144, folded);
  folded = fold16_parallel_eor3(x1, coeff_160, folded);
  folded = fold16_parallel_eor3(x0, coeff_176, folded);

  let crc0 = barrett_reduce_crc64_eor3::<S>(folded);

  if rem_len > 0 {
    return S::portable(crc0, core::slice::from_raw_parts(buf, rem_len));
  }

  crc0
}

/// Compute CRC64 using PMULL with EOR3 acceleration.
#[target_feature(enable = "aes,sha3")]
unsafe fn compute_pmull_crc64_unchecked_impl_eor3<S: Crc64PmullSpec>(crc: u64, data: &[u8]) -> u64 {
  const THRESHOLD_12LANE: usize = 32 * 1024;

  if data.len() < 64 {
    S::portable(crc, data)
  } else if data.len() < 128 {
    // 4-lane doesn't benefit much from EOR3, use non-EOR3 version
    compute_pmull_crc64_4lane::<S>(crc, data)
  } else if data.len() < THRESHOLD_12LANE {
    compute_pmull_crc64_8lane_eor3::<S>(crc, data)
  } else {
    compute_pmull_crc64_12lane_eor3::<S>(crc, data)
  }
}

#[target_feature(enable = "aes,sha3")]
pub(crate) unsafe fn compute_pmull_crc64_xz_eor3_unchecked(crc: u64, data: &[u8]) -> u64 {
  compute_pmull_crc64_unchecked_impl_eor3::<Crc64XzPmullSpec>(crc, data)
}

#[target_feature(enable = "aes,sha3")]
pub(crate) unsafe fn compute_pmull_crc64_nvme_eor3_unchecked(crc: u64, data: &[u8]) -> u64 {
  compute_pmull_crc64_unchecked_impl_eor3::<Crc64NvmePmullSpec>(crc, data)
}

// Non-EOR3 unchecked entry points - compiled when sha3 is not a target feature
#[cfg(any(
  all(target_feature = "aes", not(target_feature = "sha3")),
  all(feature = "std", not(target_feature = "aes"))
))]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn compute_pmull_crc64_xz_unchecked(crc: u64, data: &[u8]) -> u64 {
  compute_pmull_crc64_unchecked_impl::<Crc64XzPmullSpec>(crc, data)
}

#[cfg(any(
  all(target_feature = "aes", not(target_feature = "sha3")),
  all(feature = "std", not(target_feature = "aes"))
))]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn compute_pmull_crc64_nvme_unchecked(crc: u64, data: &[u8]) -> u64 {
  compute_pmull_crc64_unchecked_impl::<Crc64NvmePmullSpec>(crc, data)
}

#[cfg(all(target_feature = "aes", not(target_feature = "sha3")))]
#[inline]
pub(crate) fn compute_pmull_crc64_xz_enabled(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pmull_crc64_xz_unchecked(crc, data) }
}

#[cfg(all(target_feature = "aes", not(target_feature = "sha3")))]
#[inline]
pub(crate) fn compute_pmull_crc64_nvme_enabled(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pmull_crc64_nvme_unchecked(crc, data) }
}

#[cfg(all(feature = "std", not(target_feature = "aes")))]
#[inline]
pub(crate) fn compute_pmull_crc64_xz_runtime(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: callers must gate this with `is_aarch64_feature_detected!("aes")`.
  unsafe { compute_pmull_crc64_xz_unchecked(crc, data) }
}

#[cfg(all(feature = "std", not(target_feature = "aes")))]
#[inline]
pub(crate) fn compute_pmull_crc64_nvme_runtime(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: callers must gate this with `is_aarch64_feature_detected!("aes")`.
  unsafe { compute_pmull_crc64_nvme_unchecked(crc, data) }
}

// EOR3-accelerated entry points

#[cfg(all(target_feature = "aes", target_feature = "sha3"))]
#[inline]
pub(crate) fn compute_pmull_crc64_xz_eor3_enabled(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pmull_crc64_xz_eor3_unchecked(crc, data) }
}

#[cfg(all(target_feature = "aes", target_feature = "sha3"))]
#[inline]
pub(crate) fn compute_pmull_crc64_nvme_eor3_enabled(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pmull_crc64_nvme_eor3_unchecked(crc, data) }
}

#[cfg(all(feature = "std", not(all(target_feature = "aes", target_feature = "sha3"))))]
#[inline]
pub(crate) fn compute_pmull_crc64_xz_eor3_runtime(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: callers must gate this with `is_aarch64_feature_detected!("aes")` and
  // `is_aarch64_feature_detected!("sha3")`.
  unsafe { compute_pmull_crc64_xz_eor3_unchecked(crc, data) }
}

#[cfg(all(feature = "std", not(all(target_feature = "aes", target_feature = "sha3"))))]
#[inline]
pub(crate) fn compute_pmull_crc64_nvme_eor3_runtime(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: callers must gate this with `is_aarch64_feature_detected!("aes")` and
  // `is_aarch64_feature_detected!("sha3")`.
  unsafe { compute_pmull_crc64_nvme_eor3_unchecked(crc, data) }
}
