//! x86_64 hardware CRC-32C kernel (SSE4.2 `crc32` instruction).
//!
//! # Safety
//!
//! Uses `unsafe` for x86 SIMD intrinsics. Callers must ensure SSE4.2 is
//! available before executing the accelerated path (the dispatcher does this).
#![allow(unsafe_code)]
// SAFETY: This module is intrinsics-heavy and uses tight, invariant-driven indexing
// (e.g. fixed-size lanes and chunked block processing) where bounds are proven by
// control flow; Clippy cannot always see these invariants.
#![allow(clippy::indexing_slicing)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::{
  arch::x86_64::*,
  ops::{BitXor, BitXorAssign},
  ptr,
};

use crate::common::{
  combine::{Gf2Matrix32, generate_shift8_matrix_32},
  tables::{CRC32_IEEE_POLY, CRC32C_POLY},
};

const CRC32C_SHIFT8_MATRIX: Gf2Matrix32 = generate_shift8_matrix_32(CRC32C_POLY);

#[inline]
#[must_use]
fn pow_shift8_crc32c(len_bytes: usize) -> Gf2Matrix32 {
  if len_bytes == 0 {
    return Gf2Matrix32::identity();
  }

  let mut mat = CRC32C_SHIFT8_MATRIX;
  let mut result = Gf2Matrix32::identity();
  let mut remaining = len_bytes;

  while remaining > 0 {
    if (remaining & 1) != 0 {
      result = result.mul_mat(mat);
    }
    mat = mat.square();
    remaining = remaining.strict_shr(1);
  }

  result
}

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

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32C HWCRC Multi-stream Kernels (SSE4.2)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_sse42_nway<const N: usize>(crc: u32, data: &[u8]) -> u32 {
  debug_assert!(N == 2 || N == 4 || N == 7 || N == 8);

  let len = data.len();
  if len < N.strict_mul(256) {
    return crc32c_sse42(crc, data);
  }

  // Use even-sized chunking (leave tail on the last stream).
  let chunk_len = len / N;
  let mut lanes = [!0u32; N];

  // Process a striped prefix where all lanes have full `chunk_len`.
  let mut i: usize = 0;
  while i < chunk_len {
    // 8B chunks when possible.
    if i.strict_add(8) <= chunk_len {
      let mut lane_idx: usize = 0;
      while lane_idx < N {
        let base = lane_idx.strict_mul(chunk_len).strict_add(i);
        lanes[lane_idx] = _mm_crc32_u64(
          lanes[lane_idx] as u64,
          ptr::read_unaligned(data.as_ptr().add(base) as *const u64),
        ) as u32;
        lane_idx = lane_idx.strict_add(1);
      }
      i = i.strict_add(8);
    } else {
      let mut lane_idx: usize = 0;
      while lane_idx < N {
        let base = lane_idx.strict_mul(chunk_len).strict_add(i);
        lanes[lane_idx] = _mm_crc32_u8(lanes[lane_idx], *data.get_unchecked(base));
        lane_idx = lane_idx.strict_add(1);
      }
      i = i.strict_add(1);
    }
  }

  // Finish the tail on the last lane (remaining bytes after N*chunk_len).
  let tail_start = chunk_len.strict_mul(N);
  if tail_start < len {
    lanes[N - 1] = crc32c_sse42(lanes[N - 1], data.get_unchecked(tail_start..));
  }

  // Compute CRC(data) under the standard initial state, then append it to `crc`.
  //
  // This keeps correctness for arbitrary `crc` while allowing multi-stream computation of CRC(data).
  let last_lane_len = len.strict_sub(chunk_len.strict_mul(N.strict_sub(1)));
  let pow_chunk = pow_shift8_crc32c(chunk_len);
  let pow_last = if last_lane_len == chunk_len {
    pow_chunk
  } else {
    pow_shift8_crc32c(last_lane_len)
  };

  let mut data_crc_final: u32 = 0;
  let mut lane_idx: usize = 0;
  while lane_idx < N {
    let pow = if lane_idx.strict_add(1) == N {
      pow_last
    } else {
      pow_chunk
    };
    data_crc_final = pow.mul_vec(data_crc_final) ^ (lanes[lane_idx] ^ !0);
    lane_idx = lane_idx.strict_add(1);
  }

  let boundary_final = crc ^ !0;
  let combined_final = pow_shift8_crc32c(len).mul_vec(boundary_final) ^ data_crc_final;
  combined_final ^ !0
}

#[inline]
pub fn crc32c_sse42_2way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSE4.2 before selecting this kernel.
  unsafe { crc32c_sse42_nway::<2>(crc, data) }
}

#[inline]
pub fn crc32c_sse42_4way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSE4.2 before selecting this kernel.
  unsafe { crc32c_sse42_nway::<4>(crc, data) }
}

#[inline]
pub fn crc32c_sse42_7way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSE4.2 before selecting this kernel.
  unsafe { crc32c_sse42_nway::<7>(crc, data) }
}

#[inline]
pub fn crc32c_sse42_8way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSE4.2 before selecting this kernel.
  unsafe { crc32c_sse42_nway::<8>(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32C Fusion Kernels (SSE4.2 + PCLMULQDQ, AVX-512 tiers)
// ─────────────────────────────────────────────────────────────────────────────
//
// Derived from Corsix `fast-crc32` generator output (v4s3x3 / v3x2 families).
// These kernels fuse hardware CRC instructions with carryless multiply folding.

#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn clmul_lo_sse(a: __m128i, b: __m128i) -> __m128i {
  _mm_clmulepi64_si128::<0x00>(a, b)
}

#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn clmul_hi_sse(a: __m128i, b: __m128i) -> __m128i {
  _mm_clmulepi64_si128::<0x11>(a, b)
}

#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn clmul_scalar_sse(a: u32, b: u32) -> __m128i {
  _mm_clmulepi64_si128::<0x00>(_mm_cvtsi32_si128(a as i32), _mm_cvtsi32_si128(b as i32))
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn mm_extract_epi64(val: __m128i, idx: i32) -> u64 {
  if idx == 0 {
    _mm_cvtsi128_si64(val) as u64
  } else {
    _mm_cvtsi128_si64(_mm_srli_si128::<8>(val)) as u64
  }
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn mm_crc32c_u64(crc: u32, val: u64) -> u32 {
  _mm_crc32_u64(crc as u64, val) as u32
}

// x^n mod P (iSCSI / CRC32C), in O(log n) time.
#[target_feature(enable = "sse4.2,pclmulqdq")]
unsafe fn xnmodp_iscsi_sse(mut n: u64) -> u32 {
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
    acc = _mm_crc32_u32(acc, 0);
    n = n.strict_sub(1);
  }

  while {
    low = (stack & 1) as u32;
    stack >>= 1;
    stack != 0
  } {
    let x = _mm_cvtsi32_si128(acc as i32);
    let clmul_result = _mm_clmulepi64_si128::<0x00>(x, x);
    let y = mm_extract_epi64(clmul_result, 0);
    acc = mm_crc32c_u64(0, y << low);
  }

  acc
}

#[inline]
#[target_feature(enable = "sse4.2,pclmulqdq")]
unsafe fn crc_shift_iscsi_sse(crc: u32, nbytes: usize) -> __m128i {
  let bits = nbytes.strict_mul(8).strict_sub(33);
  clmul_scalar_sse(crc, xnmodp_iscsi_sse(bits as u64))
}

#[inline]
#[target_feature(enable = "sse4.2,pclmulqdq")]
unsafe fn crc32c_iscsi_sse_v4s3x3(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  // Align to 8-byte boundary using hardware CRC32C instructions.
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = _mm_crc32_u8(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  // Handle 8-byte alignment.
  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = mm_crc32c_u64(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 144 {
    let blk = (len.strict_sub(8)) / 136;
    let klen = blk.strict_mul(24);
    let buf2_start = buf;
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    let mut x0 = _mm_loadu_si128(buf2_start as *const __m128i);
    let mut x1 = _mm_loadu_si128(buf2_start.add(16) as *const __m128i);
    let mut x2 = _mm_loadu_si128(buf2_start.add(32) as *const __m128i);
    let mut x3 = _mm_loadu_si128(buf2_start.add(48) as *const __m128i);

    // iSCSI-specific folding constant.
    let mut k = _mm_setr_epi32(0x740eef02u32 as i32, 0, 0x9e4addf8u32 as i32, 0);

    // XOR CRC into the first vector's low 32 bits.
    x0 = _mm_xor_si128(_mm_cvtsi32_si128(crc0 as i32), x0);
    crc0 = 0;

    let mut buf2 = buf2_start.add(64);
    len = len.strict_sub(136);
    buf = buf.add(blk.strict_mul(64));

    while len >= 144 {
      let mut y0 = clmul_lo_sse(x0, k);
      x0 = clmul_hi_sse(x0, k);
      let mut y1 = clmul_lo_sse(x1, k);
      x1 = clmul_hi_sse(x1, k);
      let mut y2 = clmul_lo_sse(x2, k);
      x2 = clmul_hi_sse(x2, k);
      let mut y3 = clmul_lo_sse(x3, k);
      x3 = clmul_hi_sse(x3, k);

      y0 = _mm_xor_si128(y0, _mm_loadu_si128(buf2 as *const __m128i));
      x0 = _mm_xor_si128(x0, y0);
      y1 = _mm_xor_si128(y1, _mm_loadu_si128(buf2.add(16) as *const __m128i));
      x1 = _mm_xor_si128(x1, y1);
      y2 = _mm_xor_si128(y2, _mm_loadu_si128(buf2.add(32) as *const __m128i));
      x2 = _mm_xor_si128(x2, y2);
      y3 = _mm_xor_si128(y3, _mm_loadu_si128(buf2.add(48) as *const __m128i));
      x3 = _mm_xor_si128(x3, y3);

      crc0 = mm_crc32c_u64(crc0, ptr::read_unaligned(buf as *const u64));
      crc1 = mm_crc32c_u64(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
      crc2 = mm_crc32c_u64(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
      crc0 = mm_crc32c_u64(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
      crc1 = mm_crc32c_u64(crc1, ptr::read_unaligned(buf.add(klen + 8) as *const u64));
      crc2 = mm_crc32c_u64(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64));
      crc0 = mm_crc32c_u64(crc0, ptr::read_unaligned(buf.add(16) as *const u64));
      crc1 = mm_crc32c_u64(crc1, ptr::read_unaligned(buf.add(klen + 16) as *const u64));
      crc2 = mm_crc32c_u64(
        crc2,
        ptr::read_unaligned(buf.add(klen.strict_mul(2) + 16) as *const u64),
      );

      buf = buf.add(24);
      buf2 = buf2.add(64);
      len = len.strict_sub(136);
    }

    // Reduce x0..x3 to x0.
    k = _mm_setr_epi32(0xf20c0dfeu32 as i32, 0, 0x493c7d27u32 as i32, 0);

    let mut y0 = clmul_lo_sse(x0, k);
    x0 = clmul_hi_sse(x0, k);
    let mut y2 = clmul_lo_sse(x2, k);
    x2 = clmul_hi_sse(x2, k);

    y0 = _mm_xor_si128(y0, x1);
    x0 = _mm_xor_si128(x0, y0);
    y2 = _mm_xor_si128(y2, x3);
    x2 = _mm_xor_si128(x2, y2);

    k = _mm_setr_epi32(0x3da6d0cbu32 as i32, 0, 0xba4fc28eu32 as i32, 0);

    y0 = clmul_lo_sse(x0, k);
    x0 = clmul_hi_sse(x0, k);
    y0 = _mm_xor_si128(y0, x2);
    x0 = _mm_xor_si128(x0, y0);

    // Final scalar chunk.
    crc0 = mm_crc32c_u64(crc0, ptr::read_unaligned(buf as *const u64));
    crc1 = mm_crc32c_u64(crc1, ptr::read_unaligned(buf.add(klen) as *const u64));
    crc2 = mm_crc32c_u64(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64));
    crc0 = mm_crc32c_u64(crc0, ptr::read_unaligned(buf.add(8) as *const u64));
    crc1 = mm_crc32c_u64(crc1, ptr::read_unaligned(buf.add(klen + 8) as *const u64));
    crc2 = mm_crc32c_u64(crc2, ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64));
    crc0 = mm_crc32c_u64(crc0, ptr::read_unaligned(buf.add(16) as *const u64));
    crc1 = mm_crc32c_u64(crc1, ptr::read_unaligned(buf.add(klen + 16) as *const u64));
    crc2 = mm_crc32c_u64(
      crc2,
      ptr::read_unaligned(buf.add(klen.strict_mul(2) + 16) as *const u64),
    );
    buf = buf.add(24);

    let vc0 = crc_shift_iscsi_sse(crc0, klen.strict_mul(2) + 8);
    let vc1 = crc_shift_iscsi_sse(crc1, klen + 8);
    let mut vc = mm_extract_epi64(_mm_xor_si128(vc0, vc1), 0);

    // Reduce 128 bits to 32 bits, and multiply by x^32.
    let x0_low = mm_extract_epi64(x0, 0);
    let x0_high = mm_extract_epi64(x0, 1);
    let x0_combined = mm_extract_epi64(
      crc_shift_iscsi_sse(mm_crc32c_u64(mm_crc32c_u64(0, x0_low), x0_high), klen.strict_mul(3) + 8),
      0,
    );
    vc ^= x0_combined;

    // Final 8 bytes.
    buf = buf.add(klen.strict_mul(2));
    crc0 = crc2;
    crc0 = mm_crc32c_u64(crc0, ptr::read_unaligned(buf as *const u64) ^ vc);
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len >= 8 {
    crc0 = mm_crc32c_u64(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = _mm_crc32_u8(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

/// Safe wrapper for CRC-32C fusion kernel (SSE4.2 + PCLMULQDQ).
#[inline]
pub fn crc32c_iscsi_sse_v4s3x3_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSE4.2 + PCLMULQDQ before selecting this kernel.
  unsafe { crc32c_iscsi_sse_v4s3x3(crc, data.as_ptr(), data.len()) }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32C Fusion Multi-stream Wrappers (SSE/AVX-512)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn crc32c_fusion_nway<const N: usize>(crc: u32, data: &[u8], update: fn(u32, &[u8]) -> u32) -> u32 {
  debug_assert!(N == 2 || N == 4 || N == 7 || N == 8);

  let len = data.len();
  if len < N.strict_mul(2 * super::CRC32_FOLD_BLOCK_BYTES) {
    return update(crc, data);
  }

  let chunk_len = len.strict_div(N);
  let mut lanes = [!0u32; N];

  let mut lane_idx: usize = 0;
  while lane_idx < N {
    let start = lane_idx.strict_mul(chunk_len);
    let end = if lane_idx.strict_add(1) == N {
      len
    } else {
      start.strict_add(chunk_len)
    };
    lanes[lane_idx] = update(lanes[lane_idx], &data[start..end]);
    lane_idx = lane_idx.strict_add(1);
  }

  let last_lane_len = len.strict_sub(chunk_len.strict_mul(N.strict_sub(1)));
  let pow_chunk = pow_shift8_crc32c(chunk_len);
  let pow_last = if last_lane_len == chunk_len {
    pow_chunk
  } else {
    pow_shift8_crc32c(last_lane_len)
  };

  let mut data_crc_final: u32 = 0;
  let mut lane_idx: usize = 0;
  while lane_idx < N {
    let pow = if lane_idx.strict_add(1) == N {
      pow_last
    } else {
      pow_chunk
    };
    data_crc_final = pow.mul_vec(data_crc_final) ^ (lanes[lane_idx] ^ !0);
    lane_idx = lane_idx.strict_add(1);
  }

  let boundary_final = crc ^ !0;
  let combined_final = pow_shift8_crc32c(len).mul_vec(boundary_final) ^ data_crc_final;
  combined_final ^ !0
}

#[inline]
pub fn crc32c_iscsi_sse_v4s3x3_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<2>(crc, data, crc32c_iscsi_sse_v4s3x3_safe)
}

#[inline]
pub fn crc32c_iscsi_sse_v4s3x3_4way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<4>(crc, data, crc32c_iscsi_sse_v4s3x3_safe)
}

#[inline]
pub fn crc32c_iscsi_sse_v4s3x3_7way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<7>(crc, data, crc32c_iscsi_sse_v4s3x3_safe)
}

#[inline]
pub fn crc32c_iscsi_sse_v4s3x3_8way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<8>(crc, data, crc32c_iscsi_sse_v4s3x3_safe)
}

#[inline]
#[target_feature(enable = "sse4.2,avx512f,avx512vl,avx512bw,avx512dq,pclmulqdq")]
unsafe fn crc32c_iscsi_avx512_v4s3x3(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  // This is the AVX-512 variant of v4s3x3. It uses 128-bit loads/stores with
  // AVX-512 ternary logic for better throughput.

  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = _mm_crc32_u8(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  if (buf as usize & 8) != 0 && len >= 8 {
    crc0 = _mm_crc32_u64(crc0 as u64, ptr::read_unaligned(buf as *const u64)) as u32;
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 144 {
    let blk = (len.strict_sub(8)) / 136;
    let klen = blk.strict_mul(24);
    let buf2_start = buf;
    let mut crc1 = 0u32;
    let mut crc2 = 0u32;

    let mut x0 = _mm_loadu_si128(buf2_start as *const __m128i);
    let mut x1 = _mm_loadu_si128(buf2_start.add(16) as *const __m128i);
    let mut x2 = _mm_loadu_si128(buf2_start.add(32) as *const __m128i);
    let mut x3 = _mm_loadu_si128(buf2_start.add(48) as *const __m128i);

    let mut k = _mm_setr_epi32(0x740eef02u32 as i32, 0, 0x9e4addf8u32 as i32, 0);
    x0 = _mm_xor_si128(_mm_cvtsi32_si128(crc0 as i32), x0);
    crc0 = 0;

    let mut buf2 = buf2_start.add(64);
    len = len.strict_sub(136);
    buf = buf.add(blk.strict_mul(64));

    while len >= 144 {
      let y0 = clmul_lo_sse(x0, k);
      x0 = clmul_hi_sse(x0, k);
      let y1 = clmul_lo_sse(x1, k);
      x1 = clmul_hi_sse(x1, k);
      let y2 = clmul_lo_sse(x2, k);
      x2 = clmul_hi_sse(x2, k);
      let y3 = clmul_lo_sse(x3, k);
      x3 = clmul_hi_sse(x3, k);

      x0 = _mm_ternarylogic_epi64(x0, y0, _mm_loadu_si128(buf2 as *const __m128i), 0x96);
      x1 = _mm_ternarylogic_epi64(x1, y1, _mm_loadu_si128(buf2.add(16) as *const __m128i), 0x96);
      x2 = _mm_ternarylogic_epi64(x2, y2, _mm_loadu_si128(buf2.add(32) as *const __m128i), 0x96);
      x3 = _mm_ternarylogic_epi64(x3, y3, _mm_loadu_si128(buf2.add(48) as *const __m128i), 0x96);

      crc0 = _mm_crc32_u64(crc0 as u64, ptr::read_unaligned(buf as *const u64)) as u32;
      crc1 = _mm_crc32_u64(crc1 as u64, ptr::read_unaligned(buf.add(klen) as *const u64)) as u32;
      crc2 = _mm_crc32_u64(
        crc2 as u64,
        ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64),
      ) as u32;
      crc0 = _mm_crc32_u64(crc0 as u64, ptr::read_unaligned(buf.add(8) as *const u64)) as u32;
      crc1 = _mm_crc32_u64(crc1 as u64, ptr::read_unaligned(buf.add(klen + 8) as *const u64)) as u32;
      crc2 = _mm_crc32_u64(
        crc2 as u64,
        ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64),
      ) as u32;
      crc0 = _mm_crc32_u64(crc0 as u64, ptr::read_unaligned(buf.add(16) as *const u64)) as u32;
      crc1 = _mm_crc32_u64(crc1 as u64, ptr::read_unaligned(buf.add(klen + 16) as *const u64)) as u32;
      crc2 = _mm_crc32_u64(
        crc2 as u64,
        ptr::read_unaligned(buf.add(klen.strict_mul(2) + 16) as *const u64),
      ) as u32;

      buf = buf.add(24);
      buf2 = buf2.add(64);
      len = len.strict_sub(136);
    }

    k = _mm_setr_epi32(0xf20c0dfeu32 as i32, 0, 0x493c7d27u32 as i32, 0);
    let y0 = clmul_lo_sse(x0, k);
    x0 = clmul_hi_sse(x0, k);
    let y2 = clmul_lo_sse(x2, k);
    x2 = clmul_hi_sse(x2, k);

    x0 = _mm_ternarylogic_epi64(x0, y0, x1, 0x96);
    x2 = _mm_ternarylogic_epi64(x2, y2, x3, 0x96);

    k = _mm_setr_epi32(0x3da6d0cbu32 as i32, 0, 0xba4fc28eu32 as i32, 0);
    let y0 = clmul_lo_sse(x0, k);
    x0 = clmul_hi_sse(x0, k);
    x0 = _mm_ternarylogic_epi64(x0, y0, x2, 0x96);

    crc0 = _mm_crc32_u64(crc0 as u64, ptr::read_unaligned(buf as *const u64)) as u32;
    crc1 = _mm_crc32_u64(crc1 as u64, ptr::read_unaligned(buf.add(klen) as *const u64)) as u32;
    crc2 = _mm_crc32_u64(
      crc2 as u64,
      ptr::read_unaligned(buf.add(klen.strict_mul(2)) as *const u64),
    ) as u32;
    crc0 = _mm_crc32_u64(crc0 as u64, ptr::read_unaligned(buf.add(8) as *const u64)) as u32;
    crc1 = _mm_crc32_u64(crc1 as u64, ptr::read_unaligned(buf.add(klen + 8) as *const u64)) as u32;
    crc2 = _mm_crc32_u64(
      crc2 as u64,
      ptr::read_unaligned(buf.add(klen.strict_mul(2) + 8) as *const u64),
    ) as u32;
    crc0 = _mm_crc32_u64(crc0 as u64, ptr::read_unaligned(buf.add(16) as *const u64)) as u32;
    crc1 = _mm_crc32_u64(crc1 as u64, ptr::read_unaligned(buf.add(klen + 16) as *const u64)) as u32;
    crc2 = _mm_crc32_u64(
      crc2 as u64,
      ptr::read_unaligned(buf.add(klen.strict_mul(2) + 16) as *const u64),
    ) as u32;
    buf = buf.add(24);

    let vc0 = crc_shift_iscsi_sse(crc0, klen.strict_mul(2) + 8);
    let vc1 = crc_shift_iscsi_sse(crc1, klen + 8);
    let mut vc = mm_extract_epi64(_mm_xor_si128(vc0, vc1), 0);

    let x0_low = mm_extract_epi64(x0, 0);
    let x0_high = mm_extract_epi64(x0, 1);
    let x0_combined = mm_extract_epi64(
      crc_shift_iscsi_sse(mm_crc32c_u64(mm_crc32c_u64(0, x0_low), x0_high), klen.strict_mul(3) + 8),
      0,
    );
    vc ^= x0_combined;

    buf = buf.add(klen.strict_mul(2));
    crc0 = crc2;
    crc0 = mm_crc32c_u64(crc0, ptr::read_unaligned(buf as *const u64) ^ vc);
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len >= 8 {
    crc0 = mm_crc32c_u64(crc0, ptr::read_unaligned(buf as *const u64));
    buf = buf.add(8);
    len = len.strict_sub(8);
  }
  while len > 0 {
    crc0 = _mm_crc32_u8(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }
  crc0
}

/// Safe wrapper for CRC-32C fusion kernel (AVX-512 v4s3x3).
#[inline]
pub fn crc32c_iscsi_avx512_v4s3x3_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies AVX-512 + PCLMULQDQ before selecting this kernel.
  unsafe { crc32c_iscsi_avx512_v4s3x3(crc, data.as_ptr(), data.len()) }
}

#[inline]
pub fn crc32c_iscsi_avx512_v4s3x3_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<2>(crc, data, crc32c_iscsi_avx512_v4s3x3_safe)
}

#[inline]
pub fn crc32c_iscsi_avx512_v4s3x3_4way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<4>(crc, data, crc32c_iscsi_avx512_v4s3x3_safe)
}

#[inline]
pub fn crc32c_iscsi_avx512_v4s3x3_7way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<7>(crc, data, crc32c_iscsi_avx512_v4s3x3_safe)
}

#[inline]
pub fn crc32c_iscsi_avx512_v4s3x3_8way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<8>(crc, data, crc32c_iscsi_avx512_v4s3x3_safe)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn clmul_lo_avx512_vpclmulqdq(a: __m512i, b: __m512i) -> __m512i {
  _mm512_clmulepi64_epi128(a, b, 0)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn clmul_hi_avx512_vpclmulqdq(a: __m512i, b: __m512i) -> __m512i {
  _mm512_clmulepi64_epi128(a, b, 17)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,sse4.2")]
unsafe fn crc32c_iscsi_avx512_vpclmulqdq_v3x2(mut crc0: u32, mut buf: *const u8, mut len: usize) -> u32 {
  // Align to 8-byte boundary.
  while len > 0 && (buf as usize & 7) != 0 {
    crc0 = _mm_crc32_u8(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  // Align to 64-byte boundary (cache line).
  while (buf as usize & 56) != 0 && len >= 8 {
    crc0 = _mm_crc32_u64(crc0 as u64, ptr::read_unaligned(buf as *const u64)) as u32;
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  if len >= 384 {
    // Load three 512-bit vectors (192 bytes).
    let mut x0 = _mm512_loadu_si512(buf as *const __m512i);
    let mut x1 = _mm512_loadu_si512(buf.add(64) as *const __m512i);
    let mut x2 = _mm512_loadu_si512(buf.add(128) as *const __m512i);

    // Broadcast folding constant to each 128-bit lane.
    let k_128 = _mm_setr_epi32(0xa87ab8a8u32 as i32, 0, 0xab7aff2au32 as i32, 0);
    let mut k = _mm512_broadcast_i32x4(k_128);

    // XOR CRC into the first vector's low 32 bits.
    let crc_vec = _mm512_castsi128_si512(_mm_cvtsi32_si128(crc0 as i32));
    x0 = _mm512_xor_si512(crc_vec, x0);

    // First folding round.
    let mut y0 = clmul_lo_avx512_vpclmulqdq(x0, k);
    x0 = clmul_hi_avx512_vpclmulqdq(x0, k);
    let mut y1 = clmul_lo_avx512_vpclmulqdq(x1, k);
    x1 = clmul_hi_avx512_vpclmulqdq(x1, k);
    let mut y2 = clmul_lo_avx512_vpclmulqdq(x2, k);
    x2 = clmul_hi_avx512_vpclmulqdq(x2, k);

    x0 = _mm512_ternarylogic_epi64(x0, y0, _mm512_loadu_si512(buf.add(192) as *const __m512i), 0x96);
    x1 = _mm512_ternarylogic_epi64(x1, y1, _mm512_loadu_si512(buf.add(256) as *const __m512i), 0x96);
    x2 = _mm512_ternarylogic_epi64(x2, y2, _mm512_loadu_si512(buf.add(320) as *const __m512i), 0x96);

    buf = buf.add(384);
    len = len.strict_sub(384);

    while len >= 384 {
      // First folding step.
      y0 = clmul_lo_avx512_vpclmulqdq(x0, k);
      x0 = clmul_hi_avx512_vpclmulqdq(x0, k);
      y1 = clmul_lo_avx512_vpclmulqdq(x1, k);
      x1 = clmul_hi_avx512_vpclmulqdq(x1, k);
      y2 = clmul_lo_avx512_vpclmulqdq(x2, k);
      x2 = clmul_hi_avx512_vpclmulqdq(x2, k);

      x0 = _mm512_ternarylogic_epi64(x0, y0, _mm512_loadu_si512(buf as *const __m512i), 0x96);
      x1 = _mm512_ternarylogic_epi64(x1, y1, _mm512_loadu_si512(buf.add(64) as *const __m512i), 0x96);
      x2 = _mm512_ternarylogic_epi64(x2, y2, _mm512_loadu_si512(buf.add(128) as *const __m512i), 0x96);

      // Second folding step.
      y0 = clmul_lo_avx512_vpclmulqdq(x0, k);
      x0 = clmul_hi_avx512_vpclmulqdq(x0, k);
      y1 = clmul_lo_avx512_vpclmulqdq(x1, k);
      x1 = clmul_hi_avx512_vpclmulqdq(x1, k);
      y2 = clmul_lo_avx512_vpclmulqdq(x2, k);
      x2 = clmul_hi_avx512_vpclmulqdq(x2, k);

      x0 = _mm512_ternarylogic_epi64(x0, y0, _mm512_loadu_si512(buf.add(192) as *const __m512i), 0x96);
      x1 = _mm512_ternarylogic_epi64(x1, y1, _mm512_loadu_si512(buf.add(256) as *const __m512i), 0x96);
      x2 = _mm512_ternarylogic_epi64(x2, y2, _mm512_loadu_si512(buf.add(320) as *const __m512i), 0x96);

      buf = buf.add(384);
      len = len.strict_sub(384);
    }

    // Reduce x0,x1,x2 to x0.
    let k_128 = _mm_setr_epi32(0x740eef02u32 as i32, 0, 0x9e4addf8u32 as i32, 0);
    k = _mm512_broadcast_i32x4(k_128);

    y0 = clmul_lo_avx512_vpclmulqdq(x0, k);
    x0 = clmul_hi_avx512_vpclmulqdq(x0, k);
    x0 = _mm512_ternarylogic_epi64(x0, y0, x1, 0x96);
    x1 = x2;

    y0 = clmul_lo_avx512_vpclmulqdq(x0, k);
    x0 = clmul_hi_avx512_vpclmulqdq(x0, k);
    x0 = _mm512_ternarylogic_epi64(x0, y0, x1, 0x96);

    // Reduce 512 bits to 128 bits.
    k = _mm512_setr_epi32(
      0x1c291d04u32 as i32,
      0,
      0xddc0152bu32 as i32,
      0,
      0x3da6d0cbu32 as i32,
      0,
      0xba4fc28eu32 as i32,
      0,
      0xf20c0dfeu32 as i32,
      0,
      0x493c7d27u32 as i32,
      0,
      0,
      0,
      0,
      0,
    );

    y0 = clmul_lo_avx512_vpclmulqdq(x0, k);
    k = clmul_hi_avx512_vpclmulqdq(x0, k);
    y0 = _mm512_xor_si512(y0, k);

    let lane0 = _mm512_castsi512_si128(y0);
    let lane1 = _mm512_extracti32x4_epi32(y0, 1);
    let lane2 = _mm512_extracti32x4_epi32(y0, 2);
    let lane3 = _mm512_extracti32x4_epi32(x0, 3);

    let mut z0 = _mm_ternarylogic_epi64(lane0, lane1, lane2, 0x96);
    z0 = _mm_xor_si128(z0, lane3);

    crc0 = _mm_crc32_u64(0, mm_extract_epi64(z0, 0)) as u32;
    crc0 = _mm_crc32_u64(crc0 as u64, mm_extract_epi64(z0, 1)) as u32;
  }

  while len >= 8 {
    crc0 = _mm_crc32_u64(crc0 as u64, ptr::read_unaligned(buf as *const u64)) as u32;
    buf = buf.add(8);
    len = len.strict_sub(8);
  }

  while len > 0 {
    crc0 = _mm_crc32_u8(crc0, *buf);
    buf = buf.add(1);
    len = len.strict_sub(1);
  }

  crc0
}

/// Safe wrapper for CRC-32C fusion kernel (AVX-512 VPCLMULQDQ v3x2).
#[inline]
pub fn crc32c_iscsi_avx512_vpclmulqdq_v3x2_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies AVX-512 VPCLMULQDQ before selecting this kernel.
  unsafe { crc32c_iscsi_avx512_vpclmulqdq_v3x2(crc, data.as_ptr(), data.len()) }
}

#[inline]
pub fn crc32c_iscsi_avx512_vpclmulqdq_v3x2_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<2>(crc, data, crc32c_iscsi_avx512_vpclmulqdq_v3x2_safe)
}

#[inline]
pub fn crc32c_iscsi_avx512_vpclmulqdq_v3x2_4way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<4>(crc, data, crc32c_iscsi_avx512_vpclmulqdq_v3x2_safe)
}

#[inline]
pub fn crc32c_iscsi_avx512_vpclmulqdq_v3x2_7way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<7>(crc, data, crc32c_iscsi_avx512_vpclmulqdq_v3x2_safe)
}

#[inline]
pub fn crc32c_iscsi_avx512_vpclmulqdq_v3x2_8way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_fusion_nway::<8>(crc, data, crc32c_iscsi_avx512_vpclmulqdq_v3x2_safe)
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 (IEEE) PCLMULQDQ Folding Kernel (SSSE3 + PCLMULQDQ)
// ─────────────────────────────────────────────────────────────────────────────
//
// The kernel below implements the reflected CRC-32/ISO-HDLC polynomial using a
// 128-byte block folding strategy and a final reduction stage.

const CRC32_IEEE_KEYS_REFLECTED: [u64; 23] = [
  0x0000000000000000, // unused placeholder to match 1-based indexing
  0x00000000_ccaa009e,
  0x00000001_751997d0,
  0x00000001_4a7fe880,
  0x00000001_e88ef372,
  0x00000000_ccaa009e,
  0x00000001_63cd6124,
  0x00000001_f7011641,
  0x00000001_db710641,
  0x00000001_d7cfc6ac,
  0x00000001_ea89367e,
  0x00000001_8cb44e58,
  0x00000000_df068dc2,
  0x00000000_ae0b5394,
  0x00000001_c7569e54,
  0x00000001_c6e41596,
  0x00000001_54442bd4,
  0x00000001_74359406,
  0x00000000_3db1ecdc,
  0x00000001_5a546366,
  0x00000000_f1da05aa,
  0x00000001_322d1430,
  0x00000001_1542778a,
];

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 (IEEE) Folding Constant Generation (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

/// Carryless multiplication of two 64-bit values, returning the 128-bit result (hi, lo).
#[must_use]
const fn clmul64(a: u64, b: u64) -> (u64, u64) {
  let mut hi: u64 = 0;
  let mut lo: u64 = 0;

  let mut i: u32 = 0;
  while i < 64 {
    if (a >> i) & 1 != 0 {
      if i == 0 {
        lo ^= b;
      } else {
        lo ^= b << i;
        hi ^= b >> (64u32.strict_sub(i));
      }
    }
    i = i.strict_add(1);
  }

  (hi, lo)
}

/// Reduce a 128-bit value modulo a degree-32 polynomial `x^32 + poly`.
#[must_use]
const fn reduce128_crc32(hi: u64, lo: u64, poly: u32) -> u32 {
  let poly_full: u128 = (1u128 << 32) | (poly as u128);
  let mut val: u128 = ((hi as u128) << 64) | (lo as u128);

  let mut bit: i32 = 127;
  while bit >= 32 {
    let b = bit as u32;
    if ((val >> b) & 1) != 0 {
      val ^= poly_full << b.strict_sub(32);
    }
    bit = bit.strict_sub(1);
  }

  val as u32
}

/// Compute x^n mod (x^32 + poly) in GF(2) (poly is the normal CRC polynomial without the x^32
/// term).
#[must_use]
const fn xpow_mod_crc32(mut n: u32, poly: u32) -> u32 {
  if n == 0 {
    return 1;
  }
  if n == 1 {
    return 2;
  }

  let mut result: u32 = 1;
  let mut base: u32 = 2;

  while n > 0 {
    if n & 1 != 0 {
      let (hi, lo) = clmul64(result as u64, base as u64);
      result = reduce128_crc32(hi, lo, poly);
    }
    let (hi, lo) = clmul64(base as u64, base as u64);
    base = reduce128_crc32(hi, lo, poly);
    n >>= 1;
  }

  result
}

/// Reverse the low 33 bits of `v`.
#[must_use]
const fn reverse33(v: u64) -> u64 {
  let mask = (1u64.strict_shl(33)).strict_sub(1);
  (v & mask).reverse_bits() >> 31
}

/// Compute Intel/TiKV-style folding constant K_n for CRC-32 (IEEE) in reflected mode.
#[must_use]
const fn fold_k_crc32(reflected_poly: u32, n: u32) -> u64 {
  let normal_poly = reflected_poly.reverse_bits();
  let rem = xpow_mod_crc32(n, normal_poly) as u64;
  reverse33(rem)
}

/// Compute a `(high, low)` fold coefficient pair for folding 16 bytes by `shift_bytes`.
///
/// Returns `(K_{d+32}, K_{d-32})` where `d = 8 * shift_bytes`.
#[must_use]
const fn fold16_coeff_for_bytes_crc32(reflected_poly: u32, shift_bytes: u32) -> (u64, u64) {
  if shift_bytes == 0 {
    return (0, 0);
  }
  let d = shift_bytes.strict_mul(8);
  if d < 32 {
    return (0, 0);
  }
  (
    fold_k_crc32(reflected_poly, d.strict_add(32)),
    fold_k_crc32(reflected_poly, d.strict_sub(32)),
  )
}

#[derive(Clone, Copy, Debug)]
struct Crc32StreamConstants {
  fold_256b: (u64, u64),
  fold_512b: (u64, u64),
  fold_896b: (u64, u64),
  fold_1024b: (u64, u64),
  combine_4way: [(u64, u64); 3],
  combine_7way: [(u64, u64); 6],
  combine_8way: [(u64, u64); 7],
}

impl Crc32StreamConstants {
  #[must_use]
  const fn new(reflected_poly: u32) -> Self {
    Self {
      fold_256b: fold16_coeff_for_bytes_crc32(reflected_poly, 256),
      fold_512b: fold16_coeff_for_bytes_crc32(reflected_poly, 512),
      fold_896b: fold16_coeff_for_bytes_crc32(reflected_poly, 896),
      fold_1024b: fold16_coeff_for_bytes_crc32(reflected_poly, 1024),
      combine_4way: [
        fold16_coeff_for_bytes_crc32(reflected_poly, 384),
        fold16_coeff_for_bytes_crc32(reflected_poly, 256),
        fold16_coeff_for_bytes_crc32(reflected_poly, 128),
      ],
      combine_7way: [
        fold16_coeff_for_bytes_crc32(reflected_poly, 768),
        fold16_coeff_for_bytes_crc32(reflected_poly, 640),
        fold16_coeff_for_bytes_crc32(reflected_poly, 512),
        fold16_coeff_for_bytes_crc32(reflected_poly, 384),
        fold16_coeff_for_bytes_crc32(reflected_poly, 256),
        fold16_coeff_for_bytes_crc32(reflected_poly, 128),
      ],
      combine_8way: [
        fold16_coeff_for_bytes_crc32(reflected_poly, 896),
        fold16_coeff_for_bytes_crc32(reflected_poly, 768),
        fold16_coeff_for_bytes_crc32(reflected_poly, 640),
        fold16_coeff_for_bytes_crc32(reflected_poly, 512),
        fold16_coeff_for_bytes_crc32(reflected_poly, 384),
        fold16_coeff_for_bytes_crc32(reflected_poly, 256),
        fold16_coeff_for_bytes_crc32(reflected_poly, 128),
      ],
    }
  }
}

const CRC32_IEEE_STREAM: Crc32StreamConstants = Crc32StreamConstants::new(CRC32_IEEE_POLY);

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
struct Simd128(__m128i);

impl BitXor for Simd128 {
  type Output = Self;

  #[inline]
  fn bitxor(self, other: Self) -> Self {
    // SAFETY: `_mm_xor_si128` is available on all x86_64 (SSE2 baseline).
    unsafe { Self(_mm_xor_si128(self.0, other.0)) }
  }
}

impl BitXorAssign for Simd128 {
  #[inline]
  fn bitxor_assign(&mut self, other: Self) {
    *self = *self ^ other;
  }
}

impl Simd128 {
  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn new(high: u64, low: u64) -> Self {
    Self(_mm_set_epi64x(high as i64, low as i64))
  }

  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn shift_right_8(self) -> Self {
    Self(_mm_srli_si128::<8>(self.0))
  }

  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn shift_left_12(self) -> Self {
    Self(_mm_slli_si128::<12>(self.0))
  }

  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn and(self, mask: Self) -> Self {
    Self(_mm_and_si128(self.0, mask.0))
  }

  /// Fold 16 bytes using the reflected CRC32 folding primitive.
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_16(self, coeff: Self, data_to_xor: Self) -> Self {
    let h = _mm_clmulepi64_si128::<0x10>(self.0, coeff.0);
    let l = _mm_clmulepi64_si128::<0x01>(self.0, coeff.0);
    Self(_mm_xor_si128(_mm_xor_si128(h, l), data_to_xor.0))
  }

  /// Fold 16 bytes down to CRC32 width (reflected mode).
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_width_crc32_reflected(self, high: u64, low: u64) -> Self {
    let coeff_low = Self::new(0, low);
    let coeff_high = Self::new(high, 0);

    // First: 16B -> 8B
    let clmul = _mm_clmulepi64_si128::<0x00>(self.0, coeff_low.0);
    let shifted = self.shift_right_8();
    let mut state = Self(_mm_xor_si128(clmul, shifted.0));

    // Second: 8B -> 4B
    let mask2 = Self::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_0000_0000);
    let masked = state.and(mask2);
    let shifted = state.shift_left_12();
    let clmul = _mm_clmulepi64_si128::<0x11>(shifted.0, coeff_high.0);
    state = Self(_mm_xor_si128(clmul, masked.0));

    state
  }

  /// Barrett reduction for reflected CRC32; returns the updated (pre-inverted) CRC.
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn barrett_crc32_reflected(self, poly: u64, mu: u64) -> u32 {
    let polymu = Self::new(poly, mu);
    let clmul1 = _mm_clmulepi64_si128::<0x00>(self.0, polymu.0);
    let clmul2 = _mm_clmulepi64_si128::<0x10>(clmul1, polymu.0);
    let xorred = _mm_xor_si128(self.0, clmul2);

    let hi = _mm_srli_si128::<8>(xorred);
    _mm_cvtsi128_si64(hi) as u32
  }
}

#[inline]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn finalize_lanes_crc32_ieee_reflected(x: [Simd128; 8]) -> u32 {
  let keys = &CRC32_IEEE_KEYS_REFLECTED;

  // Fold the 8 lanes down to 1 lane.
  let mut res = x[7];
  res = x[0].fold_16(Simd128::new(keys[10], keys[9]), res); // 112 bytes
  res = x[1].fold_16(Simd128::new(keys[12], keys[11]), res); // 96 bytes
  res = x[2].fold_16(Simd128::new(keys[14], keys[13]), res); // 80 bytes
  res = x[3].fold_16(Simd128::new(keys[16], keys[15]), res); // 64 bytes
  res = x[4].fold_16(Simd128::new(keys[18], keys[17]), res); // 48 bytes
  res = x[5].fold_16(Simd128::new(keys[20], keys[19]), res); // 32 bytes
  res = x[6].fold_16(Simd128::new(keys[2], keys[1]), res); // 16 bytes

  // Final reduction to CRC32.
  res = res.fold_width_crc32_reflected(keys[6], keys[5]);
  res.barrett_crc32_reflected(keys[8], keys[7])
}

#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_crc32_ieee_reflected(state: u32, first: &[Simd128; 8], rest: &[[Simd128; 8]]) -> u32 {
  let keys = &CRC32_IEEE_KEYS_REFLECTED;
  let mut x = *first;

  // XOR initial CRC into the first 16-byte lane (low 32 bits).
  x[0] ^= Simd128::new(0, state as u64);

  // 128-byte folding coefficient pair.
  let coeff_128b = Simd128::new(keys[4], keys[3]);

  for chunk in rest {
    // Unrolled: fold each 16-byte lane and XOR with the next input lane.
    x[0] = x[0].fold_16(coeff_128b, chunk[0]);
    x[1] = x[1].fold_16(coeff_128b, chunk[1]);
    x[2] = x[2].fold_16(coeff_128b, chunk[2]);
    x[3] = x[3].fold_16(coeff_128b, chunk[3]);
    x[4] = x[4].fold_16(coeff_128b, chunk[4]);
    x[5] = x[5].fold_16(coeff_128b, chunk[5]);
    x[6] = x[6].fold_16(coeff_128b, chunk[6]);
    x[7] = x[7].fold_16(coeff_128b, chunk[7]);
  }

  // Fold the 8 lanes down to 1 lane.
  let mut res = x[7];
  res = x[0].fold_16(Simd128::new(keys[10], keys[9]), res); // 112 bytes
  res = x[1].fold_16(Simd128::new(keys[12], keys[11]), res); // 96 bytes
  res = x[2].fold_16(Simd128::new(keys[14], keys[13]), res); // 80 bytes
  res = x[3].fold_16(Simd128::new(keys[16], keys[15]), res); // 64 bytes
  res = x[4].fold_16(Simd128::new(keys[18], keys[17]), res); // 48 bytes
  res = x[5].fold_16(Simd128::new(keys[20], keys[19]), res); // 32 bytes
  res = x[6].fold_16(Simd128::new(keys[2], keys[1]), res); // 16 bytes

  // Final reduction to CRC32.
  res = res.fold_width_crc32_reflected(keys[6], keys[5]);
  res.barrett_crc32_reflected(keys[8], keys[7])
}

/// CRC-32 (IEEE / ISO-HDLC) update using PCLMULQDQ folding.
///
/// # Safety
///
/// Requires SSSE3 + PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
pub unsafe fn crc32_ieee_pclmul(crc: u32, data: &[u8]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return super::portable::crc32_slice16_ieee(crc, data);
  };

  let mut state = super::portable::crc32_slice16_ieee(crc, left);
  state = update_simd_crc32_ieee_reflected(state, first, rest);
  super::portable::crc32_slice16_ieee(state, right)
}

/// Safe wrapper for CRC-32 (IEEE) PCLMUL kernel.
#[inline]
pub fn crc32_ieee_pclmul_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  unsafe { crc32_ieee_pclmul(crc, data) }
}

/// CRC-32 (IEEE / ISO-HDLC) update using a "small-buffer" PCLMULQDQ folding path.
///
/// This avoids the 128B block requirement of the main kernel by folding one 16B
/// lane at a time.
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::VPCLMUL_READY)`
/// or `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc32_ieee_pclmul_small(crc: u32, data: &[u8]) -> u32 {
  let mut buf = data.as_ptr();
  let mut len = data.len();

  if len < 16 {
    return super::portable::crc32_slice16_ieee(crc, data);
  }

  let keys = &CRC32_IEEE_KEYS_REFLECTED;
  let coeff_16b = Simd128::new(keys[2], keys[1]);

  // Load the first lane and inject the initial CRC state (low 32 bits).
  let mut x0 = Simd128(_mm_loadu_si128(buf as *const __m128i));
  x0 ^= Simd128::new(0, crc as u64);
  buf = buf.add(16);
  len = len.strict_sub(16);

  while len >= 16 {
    let chunk = Simd128(_mm_loadu_si128(buf as *const __m128i));
    x0 = x0.fold_16(coeff_16b, chunk);
    buf = buf.add(16);
    len = len.strict_sub(16);
  }

  // Reduce the folded lane to a CRC state, then finish the tail with the portable kernel.
  let x0 = x0.fold_width_crc32_reflected(keys[6], keys[5]);
  let state = x0.barrett_crc32_reflected(keys[8], keys[7]);
  let tail = core::slice::from_raw_parts(buf, len);
  super::portable::crc32_slice16_ieee(state, tail)
}

/// Safe wrapper for CRC-32 (IEEE) "pclmul-small" kernel.
#[inline]
pub fn crc32_ieee_pclmul_small_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMULQDQ (directly or via VPCLMUL-ready) before selecting this path.
  unsafe { crc32_ieee_pclmul_small(crc, data) }
}

/// Safe wrapper for CRC-32 (IEEE) "vpclmul-small" kernel.
///
/// For small buffers we intentionally use the 128-bit PCLMUL small-lane kernel
/// to avoid AVX-512 startup costs and 128B block requirements.
#[inline]
pub fn crc32_ieee_vpclmul_small_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_ieee_pclmul_small_safe(crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 (IEEE) Multi-stream Folding (SSSE3 + PCLMULQDQ)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn fold_block_128_crc32(x: &mut [Simd128; 8], chunk: &[Simd128; 8], coeff: Simd128) {
  let t0 = x[0].fold_16(coeff, chunk[0]);
  let t1 = x[1].fold_16(coeff, chunk[1]);
  let t2 = x[2].fold_16(coeff, chunk[2]);
  let t3 = x[3].fold_16(coeff, chunk[3]);
  let t4 = x[4].fold_16(coeff, chunk[4]);
  let t5 = x[5].fold_16(coeff, chunk[5]);
  let t6 = x[6].fold_16(coeff, chunk[6]);
  let t7 = x[7].fold_16(coeff, chunk[7]);

  x[0] = t0;
  x[1] = t1;
  x[2] = t2;
  x[3] = t3;
  x[4] = t4;
  x[5] = t5;
  x[6] = t6;
  x[7] = t7;
}

#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_crc32_ieee_2way(state: u32, blocks: &[[Simd128; 8]]) -> u32 {
  debug_assert!(blocks.len() >= 2);

  let coeff_256b = Simd128::new(CRC32_IEEE_STREAM.fold_256b.0, CRC32_IEEE_STREAM.fold_256b.1);
  let coeff_128b = Simd128::new(CRC32_IEEE_KEYS_REFLECTED[4], CRC32_IEEE_KEYS_REFLECTED[3]);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];

  // Inject CRC into stream 0 (block 0).
  s0[0] ^= Simd128::new(0, state as u64);

  let mut i: usize = 2;
  let even = blocks.len() & !1usize;
  while i < even {
    fold_block_128_crc32(&mut s0, &blocks[i], coeff_256b);
    fold_block_128_crc32(&mut s1, &blocks[i + 1], coeff_256b);
    i = i.strict_add(2);
  }

  // Merge streams: shift stream0 by 128B, XOR into stream1.
  let mut combined = s1;
  combined[0] ^= s0[0].fold_16(coeff_128b, Simd128::new(0, 0));
  combined[1] ^= s0[1].fold_16(coeff_128b, Simd128::new(0, 0));
  combined[2] ^= s0[2].fold_16(coeff_128b, Simd128::new(0, 0));
  combined[3] ^= s0[3].fold_16(coeff_128b, Simd128::new(0, 0));
  combined[4] ^= s0[4].fold_16(coeff_128b, Simd128::new(0, 0));
  combined[5] ^= s0[5].fold_16(coeff_128b, Simd128::new(0, 0));
  combined[6] ^= s0[6].fold_16(coeff_128b, Simd128::new(0, 0));
  combined[7] ^= s0[7].fold_16(coeff_128b, Simd128::new(0, 0));

  if even != blocks.len() {
    fold_block_128_crc32(&mut combined, &blocks[even], coeff_128b);
  }

  finalize_lanes_crc32_ieee_reflected(combined)
}

#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_crc32_ieee_4way(state: u32, blocks: &[[Simd128; 8]]) -> u32 {
  if blocks.len() < 4 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_crc32_ieee_reflected(state, first, rest);
  }

  let aligned = (blocks.len() / 4) * 4;

  let coeff_512b = Simd128::new(CRC32_IEEE_STREAM.fold_512b.0, CRC32_IEEE_STREAM.fold_512b.1);
  let coeff_128b = Simd128::new(CRC32_IEEE_KEYS_REFLECTED[4], CRC32_IEEE_KEYS_REFLECTED[3]);
  let c384 = Simd128::new(CRC32_IEEE_STREAM.combine_4way[0].0, CRC32_IEEE_STREAM.combine_4way[0].1);
  let c256 = Simd128::new(CRC32_IEEE_STREAM.combine_4way[1].0, CRC32_IEEE_STREAM.combine_4way[1].1);
  let c128 = Simd128::new(CRC32_IEEE_STREAM.combine_4way[2].0, CRC32_IEEE_STREAM.combine_4way[2].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];

  s0[0] ^= Simd128::new(0, state as u64);

  let mut i: usize = 4;
  while i < aligned {
    fold_block_128_crc32(&mut s0, &blocks[i], coeff_512b);
    fold_block_128_crc32(&mut s1, &blocks[i + 1], coeff_512b);
    fold_block_128_crc32(&mut s2, &blocks[i + 2], coeff_512b);
    fold_block_128_crc32(&mut s3, &blocks[i + 3], coeff_512b);
    i = i.strict_add(4);
  }

  let mut combined = s3;
  for lane in 0..8usize {
    combined[lane] ^= s0[lane].fold_16(c384, Simd128::new(0, 0));
    combined[lane] ^= s1[lane].fold_16(c256, Simd128::new(0, 0));
    combined[lane] ^= s2[lane].fold_16(c128, Simd128::new(0, 0));
  }

  for block in &blocks[aligned..] {
    fold_block_128_crc32(&mut combined, block, coeff_128b);
  }

  finalize_lanes_crc32_ieee_reflected(combined)
}

#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_crc32_ieee_7way(state: u32, blocks: &[[Simd128; 8]]) -> u32 {
  if blocks.len() < 7 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_crc32_ieee_reflected(state, first, rest);
  }

  let aligned = (blocks.len() / 7) * 7;

  let coeff_896b = Simd128::new(CRC32_IEEE_STREAM.fold_896b.0, CRC32_IEEE_STREAM.fold_896b.1);
  let coeff_128b = Simd128::new(CRC32_IEEE_KEYS_REFLECTED[4], CRC32_IEEE_KEYS_REFLECTED[3]);
  let combine = &CRC32_IEEE_STREAM.combine_7way;

  let c768 = Simd128::new(combine[0].0, combine[0].1);
  let c640 = Simd128::new(combine[1].0, combine[1].1);
  let c512 = Simd128::new(combine[2].0, combine[2].1);
  let c384 = Simd128::new(combine[3].0, combine[3].1);
  let c256 = Simd128::new(combine[4].0, combine[4].1);
  let c128 = Simd128::new(combine[5].0, combine[5].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];
  let mut s4 = blocks[4];
  let mut s5 = blocks[5];
  let mut s6 = blocks[6];

  s0[0] ^= Simd128::new(0, state as u64);

  let mut i: usize = 7;
  while i < aligned {
    fold_block_128_crc32(&mut s0, &blocks[i], coeff_896b);
    fold_block_128_crc32(&mut s1, &blocks[i + 1], coeff_896b);
    fold_block_128_crc32(&mut s2, &blocks[i + 2], coeff_896b);
    fold_block_128_crc32(&mut s3, &blocks[i + 3], coeff_896b);
    fold_block_128_crc32(&mut s4, &blocks[i + 4], coeff_896b);
    fold_block_128_crc32(&mut s5, &blocks[i + 5], coeff_896b);
    fold_block_128_crc32(&mut s6, &blocks[i + 6], coeff_896b);
    i = i.strict_add(7);
  }

  let mut combined = s6;
  for lane in 0..8usize {
    combined[lane] ^= s0[lane].fold_16(c768, Simd128::new(0, 0));
    combined[lane] ^= s1[lane].fold_16(c640, Simd128::new(0, 0));
    combined[lane] ^= s2[lane].fold_16(c512, Simd128::new(0, 0));
    combined[lane] ^= s3[lane].fold_16(c384, Simd128::new(0, 0));
    combined[lane] ^= s4[lane].fold_16(c256, Simd128::new(0, 0));
    combined[lane] ^= s5[lane].fold_16(c128, Simd128::new(0, 0));
  }

  for block in &blocks[aligned..] {
    fold_block_128_crc32(&mut combined, block, coeff_128b);
  }

  finalize_lanes_crc32_ieee_reflected(combined)
}

#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_crc32_ieee_8way(state: u32, blocks: &[[Simd128; 8]]) -> u32 {
  if blocks.len() < 8 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_crc32_ieee_reflected(state, first, rest);
  }

  let aligned = (blocks.len() / 8) * 8;

  let coeff_1024b = Simd128::new(CRC32_IEEE_STREAM.fold_1024b.0, CRC32_IEEE_STREAM.fold_1024b.1);
  let coeff_128b = Simd128::new(CRC32_IEEE_KEYS_REFLECTED[4], CRC32_IEEE_KEYS_REFLECTED[3]);
  let combine = &CRC32_IEEE_STREAM.combine_8way;

  let c896 = Simd128::new(combine[0].0, combine[0].1);
  let c768 = Simd128::new(combine[1].0, combine[1].1);
  let c640 = Simd128::new(combine[2].0, combine[2].1);
  let c512 = Simd128::new(combine[3].0, combine[3].1);
  let c384 = Simd128::new(combine[4].0, combine[4].1);
  let c256 = Simd128::new(combine[5].0, combine[5].1);
  let c128 = Simd128::new(combine[6].0, combine[6].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];
  let mut s4 = blocks[4];
  let mut s5 = blocks[5];
  let mut s6 = blocks[6];
  let mut s7 = blocks[7];

  s0[0] ^= Simd128::new(0, state as u64);

  let mut i: usize = 8;
  while i < aligned {
    fold_block_128_crc32(&mut s0, &blocks[i], coeff_1024b);
    fold_block_128_crc32(&mut s1, &blocks[i + 1], coeff_1024b);
    fold_block_128_crc32(&mut s2, &blocks[i + 2], coeff_1024b);
    fold_block_128_crc32(&mut s3, &blocks[i + 3], coeff_1024b);
    fold_block_128_crc32(&mut s4, &blocks[i + 4], coeff_1024b);
    fold_block_128_crc32(&mut s5, &blocks[i + 5], coeff_1024b);
    fold_block_128_crc32(&mut s6, &blocks[i + 6], coeff_1024b);
    fold_block_128_crc32(&mut s7, &blocks[i + 7], coeff_1024b);
    i = i.strict_add(8);
  }

  let mut combined = s7;
  for lane in 0..8usize {
    combined[lane] ^= s0[lane].fold_16(c896, Simd128::new(0, 0));
    combined[lane] ^= s1[lane].fold_16(c768, Simd128::new(0, 0));
    combined[lane] ^= s2[lane].fold_16(c640, Simd128::new(0, 0));
    combined[lane] ^= s3[lane].fold_16(c512, Simd128::new(0, 0));
    combined[lane] ^= s4[lane].fold_16(c384, Simd128::new(0, 0));
    combined[lane] ^= s5[lane].fold_16(c256, Simd128::new(0, 0));
    combined[lane] ^= s6[lane].fold_16(c128, Simd128::new(0, 0));
  }

  for block in &blocks[aligned..] {
    fold_block_128_crc32(&mut combined, block, coeff_128b);
  }

  finalize_lanes_crc32_ieee_reflected(combined)
}

/// CRC-32 (IEEE / ISO-HDLC) update using PCLMULQDQ folding (2-way multi-stream).
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
pub unsafe fn crc32_ieee_pclmul_2way(crc: u32, data: &[u8]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  if middle.is_empty() {
    return super::portable::crc32_slice16_ieee(crc, data);
  }

  let mut state = super::portable::crc32_slice16_ieee(crc, left);
  if middle.len() >= 2 {
    state = update_simd_crc32_ieee_2way(state, middle);
  } else if let Some((first, rest)) = middle.split_first() {
    state = update_simd_crc32_ieee_reflected(state, first, rest);
  }
  super::portable::crc32_slice16_ieee(state, right)
}

#[inline]
pub fn crc32_ieee_pclmul_2way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  unsafe { crc32_ieee_pclmul_2way(crc, data) }
}

/// CRC-32 (IEEE / ISO-HDLC) update using PCLMULQDQ folding (4-way multi-stream).
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
pub unsafe fn crc32_ieee_pclmul_4way(crc: u32, data: &[u8]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  if middle.is_empty() {
    return super::portable::crc32_slice16_ieee(crc, data);
  }

  let mut state = super::portable::crc32_slice16_ieee(crc, left);
  state = update_simd_crc32_ieee_4way(state, middle);
  super::portable::crc32_slice16_ieee(state, right)
}

#[inline]
pub fn crc32_ieee_pclmul_4way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  unsafe { crc32_ieee_pclmul_4way(crc, data) }
}

/// CRC-32 (IEEE / ISO-HDLC) update using PCLMULQDQ folding (7-way multi-stream).
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
pub unsafe fn crc32_ieee_pclmul_7way(crc: u32, data: &[u8]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  if middle.is_empty() {
    return super::portable::crc32_slice16_ieee(crc, data);
  }

  let mut state = super::portable::crc32_slice16_ieee(crc, left);
  state = update_simd_crc32_ieee_7way(state, middle);
  super::portable::crc32_slice16_ieee(state, right)
}

#[inline]
pub fn crc32_ieee_pclmul_7way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  unsafe { crc32_ieee_pclmul_7way(crc, data) }
}

/// CRC-32 (IEEE / ISO-HDLC) update using PCLMULQDQ folding (8-way multi-stream).
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
pub unsafe fn crc32_ieee_pclmul_8way(crc: u32, data: &[u8]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  if middle.is_empty() {
    return super::portable::crc32_slice16_ieee(crc, data);
  }

  let mut state = super::portable::crc32_slice16_ieee(crc, left);
  state = update_simd_crc32_ieee_8way(state, middle);
  super::portable::crc32_slice16_ieee(state, right)
}

#[inline]
pub fn crc32_ieee_pclmul_8way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  unsafe { crc32_ieee_pclmul_8way(crc, data) }
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn clmul10_vpclmul(a: __m512i, b: __m512i) -> __m512i {
  _mm512_clmulepi64_epi128(a, b, 0x10)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn clmul01_vpclmul(a: __m512i, b: __m512i) -> __m512i {
  _mm512_clmulepi64_epi128(a, b, 0x01)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn fold_16_crc32_reflected_vpclmul(state: __m512i, coeff: __m512i, data: __m512i) -> __m512i {
  _mm512_ternarylogic_epi64(clmul10_vpclmul(state, coeff), clmul01_vpclmul(state, coeff), data, 0x96)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_crc32_ieee_reflected_vpclmul(state: u32, first: &[Simd128; 8], rest: &[[Simd128; 8]]) -> u32 {
  let keys = &CRC32_IEEE_KEYS_REFLECTED;

  // Fold coefficient (128-byte distance), replicated across lanes:
  // each 128-bit lane expects: low64=keys[3], high64=keys[4].
  let coeff = _mm512_setr_epi64(
    keys[3] as i64,
    keys[4] as i64,
    keys[3] as i64,
    keys[4] as i64,
    keys[3] as i64,
    keys[4] as i64,
    keys[3] as i64,
    keys[4] as i64,
  );

  // Load the first 128-byte block as 2×512-bit registers (8×16B lanes).
  let base = first.as_ptr().cast::<u8>();
  let mut x0 = _mm512_loadu_si512(base.cast::<__m512i>());
  let mut x1 = _mm512_loadu_si512(base.add(64).cast::<__m512i>());

  // Inject CRC into the first lane (low 32 bits).
  let injected = _mm512_setr_epi64(state as i64, 0, 0, 0, 0, 0, 0, 0);
  x0 = _mm512_xor_si512(x0, injected);

  for block in rest {
    let ptr = block.as_ptr().cast::<u8>();
    let y0 = _mm512_loadu_si512(ptr.cast::<__m512i>());
    let y1 = _mm512_loadu_si512(ptr.add(64).cast::<__m512i>());
    x0 = fold_16_crc32_reflected_vpclmul(x0, coeff, y0);
    x1 = fold_16_crc32_reflected_vpclmul(x1, coeff, y1);
  }

  // Reduce the 8 lanes to one 128-bit lane using the same folding scheme as the SSE/PCLMUL path.
  let lanes: [__m128i; 8] = [
    _mm512_castsi512_si128(x0),
    _mm512_extracti32x4_epi32(x0, 1),
    _mm512_extracti32x4_epi32(x0, 2),
    _mm512_extracti32x4_epi32(x0, 3),
    _mm512_castsi512_si128(x1),
    _mm512_extracti32x4_epi32(x1, 1),
    _mm512_extracti32x4_epi32(x1, 2),
    _mm512_extracti32x4_epi32(x1, 3),
  ];

  let mut res = Simd128(lanes[7]);
  res = Simd128(lanes[0]).fold_16(Simd128::new(keys[10], keys[9]), res); // 112 bytes
  res = Simd128(lanes[1]).fold_16(Simd128::new(keys[12], keys[11]), res); // 96 bytes
  res = Simd128(lanes[2]).fold_16(Simd128::new(keys[14], keys[13]), res); // 80 bytes
  res = Simd128(lanes[3]).fold_16(Simd128::new(keys[16], keys[15]), res); // 64 bytes
  res = Simd128(lanes[4]).fold_16(Simd128::new(keys[18], keys[17]), res); // 48 bytes
  res = Simd128(lanes[5]).fold_16(Simd128::new(keys[20], keys[19]), res); // 32 bytes
  res = Simd128(lanes[6]).fold_16(Simd128::new(keys[2], keys[1]), res); // 16 bytes

  res = res.fold_width_crc32_reflected(keys[6], keys[5]);
  res.barrett_crc32_reflected(keys[8], keys[7])
}

/// CRC-32 (IEEE / ISO-HDLC) update using AVX-512 VPCLMULQDQ folding.
///
/// # Safety
///
/// Requires AVX-512 VPCLMULQDQ. Caller must verify via `platform::caps().has(x86::VPCLMUL_READY)`.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
pub unsafe fn crc32_ieee_vpclmul(crc: u32, data: &[u8]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return super::portable::crc32_slice16_ieee(crc, data);
  };

  // If there are no full 128B blocks beyond `first`, the SSE/PCLMUL kernel is typically better.
  if rest.is_empty() {
    return crc32_ieee_pclmul(crc, data);
  }

  let mut state = super::portable::crc32_slice16_ieee(crc, left);
  state = update_simd_crc32_ieee_reflected_vpclmul(state, first, rest);
  super::portable::crc32_slice16_ieee(state, right)
}

/// Safe wrapper for CRC-32 (IEEE) VPCLMUL kernel.
#[inline]
pub fn crc32_ieee_vpclmul_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies AVX-512 VPCLMULQDQ before selecting this kernel.
  unsafe { crc32_ieee_vpclmul(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 (IEEE) Multi-stream Folding (AVX-512 VPCLMULQDQ)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn vpclmul_coeff(pair: (u64, u64)) -> __m512i {
  // Each 128-bit lane expects: low64 = pair.1, high64 = pair.0.
  _mm512_setr_epi64(
    pair.1 as i64,
    pair.0 as i64,
    pair.1 as i64,
    pair.0 as i64,
    pair.1 as i64,
    pair.0 as i64,
    pair.1 as i64,
    pair.0 as i64,
  )
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn fold_only_crc32_reflected_vpclmul(state: __m512i, coeff: __m512i) -> __m512i {
  _mm512_xor_si512(clmul10_vpclmul(state, coeff), clmul01_vpclmul(state, coeff))
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn load_128b_block(block: &[Simd128; 8]) -> (__m512i, __m512i) {
  let ptr = block.as_ptr().cast::<u8>();
  let y0 = _mm512_loadu_si512(ptr.cast::<__m512i>());
  let y1 = _mm512_loadu_si512(ptr.add(64).cast::<__m512i>());
  (y0, y1)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,sse2,pclmulqdq")]
unsafe fn finalize_vpclmul_state(x0: __m512i, x1: __m512i) -> u32 {
  let lanes: [__m128i; 8] = [
    _mm512_castsi512_si128(x0),
    _mm512_extracti32x4_epi32(x0, 1),
    _mm512_extracti32x4_epi32(x0, 2),
    _mm512_extracti32x4_epi32(x0, 3),
    _mm512_castsi512_si128(x1),
    _mm512_extracti32x4_epi32(x1, 1),
    _mm512_extracti32x4_epi32(x1, 2),
    _mm512_extracti32x4_epi32(x1, 3),
  ];

  finalize_lanes_crc32_ieee_reflected([
    Simd128(lanes[0]),
    Simd128(lanes[1]),
    Simd128(lanes[2]),
    Simd128(lanes[3]),
    Simd128(lanes[4]),
    Simd128(lanes[5]),
    Simd128(lanes[6]),
    Simd128(lanes[7]),
  ])
}

#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_crc32_ieee_vpclmul_2way(state: u32, blocks: &[[Simd128; 8]]) -> u32 {
  debug_assert!(blocks.len() >= 2);

  let coeff_256b = vpclmul_coeff(CRC32_IEEE_STREAM.fold_256b);
  let coeff_128b = vpclmul_coeff((CRC32_IEEE_KEYS_REFLECTED[4], CRC32_IEEE_KEYS_REFLECTED[3]));

  let (mut s0_0, mut s0_1) = load_128b_block(&blocks[0]);
  let (mut s1_0, mut s1_1) = load_128b_block(&blocks[1]);

  let injected = _mm512_setr_epi64(state as i64, 0, 0, 0, 0, 0, 0, 0);
  s0_0 = _mm512_xor_si512(s0_0, injected);

  let mut i: usize = 2;
  let even = blocks.len() & !1usize;
  while i < even {
    let (y0, y1) = load_128b_block(&blocks[i]);
    s0_0 = fold_16_crc32_reflected_vpclmul(s0_0, coeff_256b, y0);
    s0_1 = fold_16_crc32_reflected_vpclmul(s0_1, coeff_256b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 1]);
    s1_0 = fold_16_crc32_reflected_vpclmul(s1_0, coeff_256b, y0);
    s1_1 = fold_16_crc32_reflected_vpclmul(s1_1, coeff_256b, y1);

    i = i.strict_add(2);
  }

  let mut combined0 = _mm512_xor_si512(s1_0, fold_only_crc32_reflected_vpclmul(s0_0, coeff_128b));
  let mut combined1 = _mm512_xor_si512(s1_1, fold_only_crc32_reflected_vpclmul(s0_1, coeff_128b));

  if even != blocks.len() {
    let (y0, y1) = load_128b_block(&blocks[even]);
    combined0 = fold_16_crc32_reflected_vpclmul(combined0, coeff_128b, y0);
    combined1 = fold_16_crc32_reflected_vpclmul(combined1, coeff_128b, y1);
  }

  finalize_vpclmul_state(combined0, combined1)
}

#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_crc32_ieee_vpclmul_4way(state: u32, blocks: &[[Simd128; 8]]) -> u32 {
  if blocks.len() < 4 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_crc32_ieee_reflected_vpclmul(state, first, rest);
  }

  let aligned = (blocks.len() / 4) * 4;
  let coeff_512b = vpclmul_coeff(CRC32_IEEE_STREAM.fold_512b);
  let coeff_128b = vpclmul_coeff((CRC32_IEEE_KEYS_REFLECTED[4], CRC32_IEEE_KEYS_REFLECTED[3]));

  let c384 = vpclmul_coeff(CRC32_IEEE_STREAM.combine_4way[0]);
  let c256 = vpclmul_coeff(CRC32_IEEE_STREAM.combine_4way[1]);
  let c128 = vpclmul_coeff(CRC32_IEEE_STREAM.combine_4way[2]);

  let (mut s0_0, mut s0_1) = load_128b_block(&blocks[0]);
  let (mut s1_0, mut s1_1) = load_128b_block(&blocks[1]);
  let (mut s2_0, mut s2_1) = load_128b_block(&blocks[2]);
  let (mut s3_0, mut s3_1) = load_128b_block(&blocks[3]);

  let injected = _mm512_setr_epi64(state as i64, 0, 0, 0, 0, 0, 0, 0);
  s0_0 = _mm512_xor_si512(s0_0, injected);

  let mut i: usize = 4;
  while i < aligned {
    let (y0, y1) = load_128b_block(&blocks[i]);
    s0_0 = fold_16_crc32_reflected_vpclmul(s0_0, coeff_512b, y0);
    s0_1 = fold_16_crc32_reflected_vpclmul(s0_1, coeff_512b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 1]);
    s1_0 = fold_16_crc32_reflected_vpclmul(s1_0, coeff_512b, y0);
    s1_1 = fold_16_crc32_reflected_vpclmul(s1_1, coeff_512b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 2]);
    s2_0 = fold_16_crc32_reflected_vpclmul(s2_0, coeff_512b, y0);
    s2_1 = fold_16_crc32_reflected_vpclmul(s2_1, coeff_512b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 3]);
    s3_0 = fold_16_crc32_reflected_vpclmul(s3_0, coeff_512b, y0);
    s3_1 = fold_16_crc32_reflected_vpclmul(s3_1, coeff_512b, y1);

    i = i.strict_add(4);
  }

  let mut combined0 = s3_0;
  let mut combined1 = s3_1;

  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s0_0, c384));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s0_1, c384));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s1_0, c256));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s1_1, c256));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s2_0, c128));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s2_1, c128));

  for block in &blocks[aligned..] {
    let (y0, y1) = load_128b_block(block);
    combined0 = fold_16_crc32_reflected_vpclmul(combined0, coeff_128b, y0);
    combined1 = fold_16_crc32_reflected_vpclmul(combined1, coeff_128b, y1);
  }

  finalize_vpclmul_state(combined0, combined1)
}

#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_crc32_ieee_vpclmul_7way(state: u32, blocks: &[[Simd128; 8]]) -> u32 {
  if blocks.len() < 7 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_crc32_ieee_reflected_vpclmul(state, first, rest);
  }

  let aligned = (blocks.len() / 7) * 7;
  let coeff_896b = vpclmul_coeff(CRC32_IEEE_STREAM.fold_896b);
  let coeff_128b = vpclmul_coeff((CRC32_IEEE_KEYS_REFLECTED[4], CRC32_IEEE_KEYS_REFLECTED[3]));

  let combine = &CRC32_IEEE_STREAM.combine_7way;
  let c768 = vpclmul_coeff(combine[0]);
  let c640 = vpclmul_coeff(combine[1]);
  let c512 = vpclmul_coeff(combine[2]);
  let c384 = vpclmul_coeff(combine[3]);
  let c256 = vpclmul_coeff(combine[4]);
  let c128 = vpclmul_coeff(combine[5]);

  let (mut s0_0, mut s0_1) = load_128b_block(&blocks[0]);
  let (mut s1_0, mut s1_1) = load_128b_block(&blocks[1]);
  let (mut s2_0, mut s2_1) = load_128b_block(&blocks[2]);
  let (mut s3_0, mut s3_1) = load_128b_block(&blocks[3]);
  let (mut s4_0, mut s4_1) = load_128b_block(&blocks[4]);
  let (mut s5_0, mut s5_1) = load_128b_block(&blocks[5]);
  let (mut s6_0, mut s6_1) = load_128b_block(&blocks[6]);

  let injected = _mm512_setr_epi64(state as i64, 0, 0, 0, 0, 0, 0, 0);
  s0_0 = _mm512_xor_si512(s0_0, injected);

  let mut i: usize = 7;
  while i < aligned {
    let (y0, y1) = load_128b_block(&blocks[i]);
    s0_0 = fold_16_crc32_reflected_vpclmul(s0_0, coeff_896b, y0);
    s0_1 = fold_16_crc32_reflected_vpclmul(s0_1, coeff_896b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 1]);
    s1_0 = fold_16_crc32_reflected_vpclmul(s1_0, coeff_896b, y0);
    s1_1 = fold_16_crc32_reflected_vpclmul(s1_1, coeff_896b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 2]);
    s2_0 = fold_16_crc32_reflected_vpclmul(s2_0, coeff_896b, y0);
    s2_1 = fold_16_crc32_reflected_vpclmul(s2_1, coeff_896b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 3]);
    s3_0 = fold_16_crc32_reflected_vpclmul(s3_0, coeff_896b, y0);
    s3_1 = fold_16_crc32_reflected_vpclmul(s3_1, coeff_896b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 4]);
    s4_0 = fold_16_crc32_reflected_vpclmul(s4_0, coeff_896b, y0);
    s4_1 = fold_16_crc32_reflected_vpclmul(s4_1, coeff_896b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 5]);
    s5_0 = fold_16_crc32_reflected_vpclmul(s5_0, coeff_896b, y0);
    s5_1 = fold_16_crc32_reflected_vpclmul(s5_1, coeff_896b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 6]);
    s6_0 = fold_16_crc32_reflected_vpclmul(s6_0, coeff_896b, y0);
    s6_1 = fold_16_crc32_reflected_vpclmul(s6_1, coeff_896b, y1);

    i = i.strict_add(7);
  }

  let mut combined0 = s6_0;
  let mut combined1 = s6_1;

  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s0_0, c768));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s0_1, c768));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s1_0, c640));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s1_1, c640));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s2_0, c512));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s2_1, c512));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s3_0, c384));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s3_1, c384));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s4_0, c256));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s4_1, c256));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s5_0, c128));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s5_1, c128));

  for block in &blocks[aligned..] {
    let (y0, y1) = load_128b_block(block);
    combined0 = fold_16_crc32_reflected_vpclmul(combined0, coeff_128b, y0);
    combined1 = fold_16_crc32_reflected_vpclmul(combined1, coeff_128b, y1);
  }

  finalize_vpclmul_state(combined0, combined1)
}

#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_crc32_ieee_vpclmul_8way(state: u32, blocks: &[[Simd128; 8]]) -> u32 {
  if blocks.len() < 8 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_crc32_ieee_reflected_vpclmul(state, first, rest);
  }

  let aligned = (blocks.len() / 8) * 8;
  let coeff_1024b = vpclmul_coeff(CRC32_IEEE_STREAM.fold_1024b);
  let coeff_128b = vpclmul_coeff((CRC32_IEEE_KEYS_REFLECTED[4], CRC32_IEEE_KEYS_REFLECTED[3]));

  let combine = &CRC32_IEEE_STREAM.combine_8way;
  let c896 = vpclmul_coeff(combine[0]);
  let c768 = vpclmul_coeff(combine[1]);
  let c640 = vpclmul_coeff(combine[2]);
  let c512 = vpclmul_coeff(combine[3]);
  let c384 = vpclmul_coeff(combine[4]);
  let c256 = vpclmul_coeff(combine[5]);
  let c128 = vpclmul_coeff(combine[6]);

  let (mut s0_0, mut s0_1) = load_128b_block(&blocks[0]);
  let (mut s1_0, mut s1_1) = load_128b_block(&blocks[1]);
  let (mut s2_0, mut s2_1) = load_128b_block(&blocks[2]);
  let (mut s3_0, mut s3_1) = load_128b_block(&blocks[3]);
  let (mut s4_0, mut s4_1) = load_128b_block(&blocks[4]);
  let (mut s5_0, mut s5_1) = load_128b_block(&blocks[5]);
  let (mut s6_0, mut s6_1) = load_128b_block(&blocks[6]);
  let (mut s7_0, mut s7_1) = load_128b_block(&blocks[7]);

  let injected = _mm512_setr_epi64(state as i64, 0, 0, 0, 0, 0, 0, 0);
  s0_0 = _mm512_xor_si512(s0_0, injected);

  let mut i: usize = 8;
  while i < aligned {
    let (y0, y1) = load_128b_block(&blocks[i]);
    s0_0 = fold_16_crc32_reflected_vpclmul(s0_0, coeff_1024b, y0);
    s0_1 = fold_16_crc32_reflected_vpclmul(s0_1, coeff_1024b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 1]);
    s1_0 = fold_16_crc32_reflected_vpclmul(s1_0, coeff_1024b, y0);
    s1_1 = fold_16_crc32_reflected_vpclmul(s1_1, coeff_1024b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 2]);
    s2_0 = fold_16_crc32_reflected_vpclmul(s2_0, coeff_1024b, y0);
    s2_1 = fold_16_crc32_reflected_vpclmul(s2_1, coeff_1024b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 3]);
    s3_0 = fold_16_crc32_reflected_vpclmul(s3_0, coeff_1024b, y0);
    s3_1 = fold_16_crc32_reflected_vpclmul(s3_1, coeff_1024b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 4]);
    s4_0 = fold_16_crc32_reflected_vpclmul(s4_0, coeff_1024b, y0);
    s4_1 = fold_16_crc32_reflected_vpclmul(s4_1, coeff_1024b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 5]);
    s5_0 = fold_16_crc32_reflected_vpclmul(s5_0, coeff_1024b, y0);
    s5_1 = fold_16_crc32_reflected_vpclmul(s5_1, coeff_1024b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 6]);
    s6_0 = fold_16_crc32_reflected_vpclmul(s6_0, coeff_1024b, y0);
    s6_1 = fold_16_crc32_reflected_vpclmul(s6_1, coeff_1024b, y1);

    let (y0, y1) = load_128b_block(&blocks[i + 7]);
    s7_0 = fold_16_crc32_reflected_vpclmul(s7_0, coeff_1024b, y0);
    s7_1 = fold_16_crc32_reflected_vpclmul(s7_1, coeff_1024b, y1);

    i = i.strict_add(8);
  }

  let mut combined0 = s7_0;
  let mut combined1 = s7_1;

  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s0_0, c896));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s0_1, c896));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s1_0, c768));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s1_1, c768));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s2_0, c640));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s2_1, c640));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s3_0, c512));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s3_1, c512));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s4_0, c384));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s4_1, c384));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s5_0, c256));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s5_1, c256));
  combined0 = _mm512_xor_si512(combined0, fold_only_crc32_reflected_vpclmul(s6_0, c128));
  combined1 = _mm512_xor_si512(combined1, fold_only_crc32_reflected_vpclmul(s6_1, c128));

  for block in &blocks[aligned..] {
    let (y0, y1) = load_128b_block(block);
    combined0 = fold_16_crc32_reflected_vpclmul(combined0, coeff_128b, y0);
    combined1 = fold_16_crc32_reflected_vpclmul(combined1, coeff_128b, y1);
  }

  finalize_vpclmul_state(combined0, combined1)
}

/// CRC-32 (IEEE / ISO-HDLC) update using AVX-512 VPCLMULQDQ folding (2-way multi-stream).
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
pub unsafe fn crc32_ieee_vpclmul_2way(crc: u32, data: &[u8]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  if middle.len() < 2 {
    return crc32_ieee_vpclmul(crc, data);
  }

  let mut state = super::portable::crc32_slice16_ieee(crc, left);
  state = update_simd_crc32_ieee_vpclmul_2way(state, middle);
  super::portable::crc32_slice16_ieee(state, right)
}

#[inline]
pub fn crc32_ieee_vpclmul_2way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies AVX-512 VPCLMULQDQ before selecting this kernel.
  unsafe { crc32_ieee_vpclmul_2way(crc, data) }
}

/// CRC-32 (IEEE / ISO-HDLC) update using AVX-512 VPCLMULQDQ folding (4-way multi-stream).
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
pub unsafe fn crc32_ieee_vpclmul_4way(crc: u32, data: &[u8]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  if middle.is_empty() {
    return crc32_ieee_vpclmul(crc, data);
  }

  let mut state = super::portable::crc32_slice16_ieee(crc, left);
  state = update_simd_crc32_ieee_vpclmul_4way(state, middle);
  super::portable::crc32_slice16_ieee(state, right)
}

#[inline]
pub fn crc32_ieee_vpclmul_4way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies AVX-512 VPCLMULQDQ before selecting this kernel.
  unsafe { crc32_ieee_vpclmul_4way(crc, data) }
}

/// CRC-32 (IEEE / ISO-HDLC) update using AVX-512 VPCLMULQDQ folding (7-way multi-stream).
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
pub unsafe fn crc32_ieee_vpclmul_7way(crc: u32, data: &[u8]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  if middle.is_empty() {
    return crc32_ieee_vpclmul(crc, data);
  }

  let mut state = super::portable::crc32_slice16_ieee(crc, left);
  state = update_simd_crc32_ieee_vpclmul_7way(state, middle);
  super::portable::crc32_slice16_ieee(state, right)
}

#[inline]
pub fn crc32_ieee_vpclmul_7way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies AVX-512 VPCLMULQDQ before selecting this kernel.
  unsafe { crc32_ieee_vpclmul_7way(crc, data) }
}

/// CRC-32 (IEEE / ISO-HDLC) update using AVX-512 VPCLMULQDQ folding (8-way multi-stream).
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
pub unsafe fn crc32_ieee_vpclmul_8way(crc: u32, data: &[u8]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  if middle.is_empty() {
    return crc32_ieee_vpclmul(crc, data);
  }

  let mut state = super::portable::crc32_slice16_ieee(crc, left);
  state = update_simd_crc32_ieee_vpclmul_8way(state, middle);
  super::portable::crc32_slice16_ieee(state, right)
}

#[inline]
pub fn crc32_ieee_vpclmul_8way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies AVX-512 VPCLMULQDQ before selecting this kernel.
  unsafe { crc32_ieee_vpclmul_8way(crc, data) }
}

#[cfg(test)]
mod tests {
  use alloc::vec::Vec;

  use super::*;

  #[test]
  fn test_crc32_ieee_fold16_constants_match_precomputed_table() {
    // Sanity-check our compile-time generator against the (trusted) generator output
    // already embedded in this module.
    assert_eq!(
      fold16_coeff_for_bytes_crc32(CRC32_IEEE_POLY, 128),
      (CRC32_IEEE_KEYS_REFLECTED[4], CRC32_IEEE_KEYS_REFLECTED[3])
    );
    assert_eq!(
      fold16_coeff_for_bytes_crc32(CRC32_IEEE_POLY, 112),
      (CRC32_IEEE_KEYS_REFLECTED[10], CRC32_IEEE_KEYS_REFLECTED[9])
    );
    assert_eq!(
      fold16_coeff_for_bytes_crc32(CRC32_IEEE_POLY, 16),
      (CRC32_IEEE_KEYS_REFLECTED[2], CRC32_IEEE_KEYS_REFLECTED[1])
    );
  }

  #[test]
  fn test_crc32_ieee_pclmul_matches_portable_various_lengths() {
    if !(std::arch::is_x86_feature_detected!("pclmulqdq") && std::arch::is_x86_feature_detected!("ssse3")) {
      return;
    }

    for len in [
      0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1024,
    ] {
      let mut data = Vec::with_capacity(len);
      for i in 0..len {
        data.push((i as u8).wrapping_mul(31).wrapping_add(7));
      }

      let portable = super::super::portable::crc32_slice16_ieee(!0, &data) ^ !0;
      let pclmul = crc32_ieee_pclmul_safe(!0, &data) ^ !0;
      assert_eq!(pclmul, portable, "len={len}");
    }
  }

  #[test]
  fn test_crc32_ieee_pclmul_small_matches_portable_various_lengths() {
    if !std::arch::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127] {
      let mut data = Vec::with_capacity(len);
      for i in 0..len {
        data.push((i as u8).wrapping_mul(31).wrapping_add(7));
      }

      let portable = super::super::portable::crc32_slice16_ieee(!0, &data) ^ !0;
      let pclmul_small = crc32_ieee_pclmul_small_safe(!0, &data) ^ !0;
      let vpclmul_small = crc32_ieee_vpclmul_small_safe(!0, &data) ^ !0;
      assert_eq!(pclmul_small, portable, "len={len}");
      assert_eq!(vpclmul_small, portable, "len={len}");
    }
  }

  #[test]
  fn test_crc32_ieee_vpclmul_matches_portable_various_lengths() {
    if !(std::arch::is_x86_feature_detected!("avx512f")
      && std::arch::is_x86_feature_detected!("avx512vl")
      && std::arch::is_x86_feature_detected!("avx512bw")
      && std::arch::is_x86_feature_detected!("avx512dq")
      && std::arch::is_x86_feature_detected!("vpclmulqdq")
      && std::arch::is_x86_feature_detected!("pclmulqdq")
      && std::arch::is_x86_feature_detected!("ssse3"))
    {
      return;
    }

    for len in [
      0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1024, 4096,
    ] {
      let mut data = Vec::with_capacity(len);
      for i in 0..len {
        data.push((i as u8).wrapping_mul(31).wrapping_add(7));
      }

      let portable = super::super::portable::crc32_slice16_ieee(!0, &data) ^ !0;
      let vpclmul = crc32_ieee_vpclmul_safe(!0, &data) ^ !0;
      assert_eq!(vpclmul, portable, "len={len}");
    }
  }

  #[test]
  fn test_crc32_ieee_pclmul_multistream_matches_portable() {
    if !(std::arch::is_x86_feature_detected!("pclmulqdq") && std::arch::is_x86_feature_detected!("ssse3")) {
      return;
    }

    for &init in &[!0u32, 0x1234_5678, 0xDEAD_BEEF] {
      for len in [0usize, 1, 15, 16, 127, 128, 255, 256, 1024, 4096, 16384] {
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
          data.push((i as u8).wrapping_mul(31).wrapping_add(7));
        }

        let expected = super::super::portable::crc32_slice16_ieee(init, &data);
        assert_eq!(
          crc32_ieee_pclmul_2way_safe(init, &data),
          expected,
          "2way init={init:#x} len={len}"
        );
        assert_eq!(
          crc32_ieee_pclmul_4way_safe(init, &data),
          expected,
          "4way init={init:#x} len={len}"
        );
        assert_eq!(
          crc32_ieee_pclmul_7way_safe(init, &data),
          expected,
          "7way init={init:#x} len={len}"
        );
        assert_eq!(
          crc32_ieee_pclmul_8way_safe(init, &data),
          expected,
          "8way init={init:#x} len={len}"
        );
      }
    }
  }

  #[test]
  fn test_crc32c_sse42_multistream_matches_portable() {
    if !std::arch::is_x86_feature_detected!("sse4.2") {
      return;
    }

    for &init in &[!0u32, 0x0123_4567, 0x89AB_CDEF] {
      for len in [0usize, 1, 7, 8, 15, 16, 127, 128, 255, 256, 2048, 8192] {
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
          data.push((i as u8).wrapping_mul(17).wrapping_add(3));
        }

        let expected = super::super::portable::crc32c_slice16(init, &data);
        assert_eq!(
          crc32c_sse42_2way_safe(init, &data),
          expected,
          "2way init={init:#x} len={len}"
        );
        assert_eq!(
          crc32c_sse42_4way_safe(init, &data),
          expected,
          "4way init={init:#x} len={len}"
        );
        assert_eq!(
          crc32c_sse42_7way_safe(init, &data),
          expected,
          "7way init={init:#x} len={len}"
        );
        assert_eq!(
          crc32c_sse42_8way_safe(init, &data),
          expected,
          "8way init={init:#x} len={len}"
        );
      }
    }
  }

  #[test]
  fn test_crc32c_fusion_sse_multistream_matches_portable() {
    if !(std::arch::is_x86_feature_detected!("sse4.2") && std::arch::is_x86_feature_detected!("pclmulqdq")) {
      return;
    }

    for &init in &[!0u32, 0x0123_4567, 0x89AB_CDEF] {
      for len in [0usize, 1, 7, 8, 15, 16, 127, 128, 255, 256, 2048, 8192, 65536] {
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
          data.push((i as u8).wrapping_mul(19).wrapping_add(5));
        }

        let expected = super::super::portable::crc32c_slice16(init, &data);
        assert_eq!(
          crc32c_iscsi_sse_v4s3x3_2way_safe(init, &data),
          expected,
          "fusion/2way init={init:#x} len={len}"
        );
        assert_eq!(
          crc32c_iscsi_sse_v4s3x3_4way_safe(init, &data),
          expected,
          "fusion/4way init={init:#x} len={len}"
        );
        assert_eq!(
          crc32c_iscsi_sse_v4s3x3_7way_safe(init, &data),
          expected,
          "fusion/7way init={init:#x} len={len}"
        );
        assert_eq!(
          crc32c_iscsi_sse_v4s3x3_8way_safe(init, &data),
          expected,
          "fusion/8way init={init:#x} len={len}"
        );
      }
    }
  }
}
