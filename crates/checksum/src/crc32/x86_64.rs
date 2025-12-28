//! x86_64 hardware CRC-32C kernel (SSE4.2 `crc32` instruction).
//!
//! # Safety
//!
//! Uses `unsafe` for x86 SIMD intrinsics. Callers must ensure SSE4.2 is
//! available before executing the accelerated path (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::{arch::x86_64::*, ptr};

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
