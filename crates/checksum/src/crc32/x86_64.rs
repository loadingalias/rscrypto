//! x86_64 CRC-32 implementations using SSE4.2 and PCLMULQDQ.
//!
//! # Hardware Features
//!
//! - **SSE4.2 CRC32C**: Hardware instruction for CRC32C polynomial only (~20 GB/s)
//! - **PCLMULQDQ + SSE4.2 fusion**: 3-stream interleaved for CRC32C (~45 GB/s)
//! - **PCLMULQDQ + Barrett**: For CRC32-IEEE (no HW instruction) (~15 GB/s)
//! - **VPCLMULQDQ**: AVX-512 versions (~60 GB/s)
//!
//! # Key Difference from CRC64
//!
//! CRC32C has dedicated SSE4.2 instructions (`_mm_crc32_*`), enabling fusion:
//! - PCLMULQDQ handles bulk folding
//! - SSE4.2 CRC32C handles final 128→32 reduction AND parallel scalar streams
//!
//! CRC32-IEEE has NO hardware instruction on x86, so it uses pure CLMUL + Barrett.
//!
//! # Credits
//!
//! CRC32C fusion based on [corsix/fast-crc32](https://github.com/corsix/fast-crc32)
//! and [crc-fast-rust](https://github.com/awesomized/crc-fast).

// SIMD intrinsics require unsafe; safety is documented per-function.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

use crate::crc32::portable::crc32_ieee_slice16;

// ─────────────────────────────────────────────────────────────────────────────
// CRC32C Folding Constants (from crc-fast-rust)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC32C (iSCSI/Castagnoli) PCLMUL folding constants.
mod crc32c_constants {
  /// 128-byte block folding constant (x^1024 mod P, x^1088 mod P).
  pub const FOLD_128: (u32, u32) = (0x740eef02, 0x9e4addf8);
  /// 16-byte lane folding constant (x^128 mod P, x^192 mod P).
  pub const FOLD_16: (u32, u32) = (0xf20c0dfe, 0x493c7d27);
  /// Reduction 64→32: (x^64 mod P, x^96 mod P).
  pub const REDUCE_64: (u32, u32) = (0x3da6d0cb, 0xba4fc28e);
}

/// CRC32-IEEE (ISO-HDLC) CLMUL folding and Barrett reduction constants.
/// These are used for pure CLMUL reduction since x86 has no IEEE HW instruction.
///
/// Constants from crc32fast (MIT licensed, same as Intel paper constants):
/// - K1/K2: 64-byte (512-bit) folding
/// - K3/K4: 16-byte (128-bit) folding
/// - K5: 64→32 bit reduction
/// - P_X/U_PRIME: Barrett reduction
///
/// These are 33-bit values with bit 32 set, as required by the Intel algorithm.
mod crc32_ieee_constants {
  /// 64-byte block folding: (K1, K2) for 512-bit fold.
  pub const K1: u64 = 0x0000_0001_5444_2bd4;
  pub const K2: u64 = 0x0000_0001_c6e4_1596;

  /// 16-byte lane folding: (K3, K4) for 128-bit fold.
  pub const K3: u64 = 0x0000_0001_7519_97d0;
  pub const K4: u64 = 0x0000_0000_ccaa_009e;

  /// 64→32 bit reduction constant.
  pub const K5: u64 = 0x0000_0001_63cd_6124;

  /// Barrett reduction: P_X (polynomial) and U_PRIME (µ).
  pub const P_X: u64 = 0x0000_0001_DB71_0641;
  pub const U_PRIME: u64 = 0x0000_0001_F701_1641;
}

// ─────────────────────────────────────────────────────────────────────────────
// PCLMUL Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// CLMUL low lanes: a.lo * b.lo
#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn clmul_lo(a: __m128i, b: __m128i) -> __m128i {
  _mm_clmulepi64_si128(a, b, 0x00)
}

/// CLMUL high lanes: a.hi * b.hi
#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn clmul_hi(a: __m128i, b: __m128i) -> __m128i {
  _mm_clmulepi64_si128(a, b, 0x11)
}

/// CLMUL scalar: 32-bit a * 32-bit b
#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn clmul_scalar(a: u32, b: u32) -> __m128i {
  _mm_clmulepi64_si128(_mm_cvtsi32_si128(a as i32), _mm_cvtsi32_si128(b as i32), 0)
}

/// Extract 64-bit value from lane 0 or 1.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn extract_epi64(val: __m128i, idx: i32) -> u64 {
  if idx == 0 {
    _mm_cvtsi128_si64(val) as u64
  } else {
    _mm_cvtsi128_si64(_mm_srli_si128(val, 8)) as u64
  }
}

/// CRC32 on a 64-bit value using SSE4.2 (CRC32C polynomial only).
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_u64(crc: u32, val: u64) -> u32 {
  _mm_crc32_u64(crc as u64, val) as u32
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC32C: x^n mod P computation (for shift combining)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute x^n mod P for CRC32C using hardware CRC and CLMUL.
/// Used to combine parallel CRC streams.
#[target_feature(enable = "sse4.2", enable = "pclmulqdq")]
unsafe fn xnmodp_crc32c(mut n: u64) -> u32 {
  // SAFETY: target_feature ensures SSE4.2 and PCLMULQDQ are available
  unsafe {
    let mut stack = !1u64;

    while n > 191 {
      stack = (stack << 1) + (n & 1);
      n = (n >> 1) - 16;
    }
    stack = !stack;
    let mut acc = 0x8000_0000u32 >> (n & 31);
    n >>= 5;

    while n > 0 {
      acc = _mm_crc32_u32(acc, 0);
      n -= 1;
    }

    let mut low: u32;
    loop {
      low = (stack & 1) as u32;
      stack >>= 1;
      if stack == 0 {
        break;
      }
      let x = _mm_cvtsi32_si128(acc as i32);
      let clmul_result = _mm_clmulepi64_si128(x, x, 0);
      let y = extract_epi64(clmul_result, 0);
      acc = crc32c_u64(0, y << low);
    }
    acc
  }
}

/// Shift a CRC32C value by nbytes using x^(nbytes*8) mod P.
#[inline]
#[target_feature(enable = "sse4.2", enable = "pclmulqdq")]
unsafe fn crc_shift_crc32c(crc: u32, nbytes: usize) -> __m128i {
  // SAFETY: target_feature ensures availability
  unsafe { clmul_scalar(crc, xnmodp_crc32c((nbytes * 8 - 33) as u64)) }
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE4.2 CRC32C Hardware Instruction (Pure)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC32C using SSE4.2 hardware instruction only (no CLMUL).
///
/// # Safety
///
/// Caller must ensure SSE4.2 is available.
#[target_feature(enable = "sse4.2")]
pub unsafe fn crc32c_sse42(mut crc: u32, data: &[u8]) -> u32 {
  // SAFETY: All operations are within bounds
  unsafe {
    let mut ptr = data.as_ptr();
    let end = ptr.add(data.len());

    // Process 8 bytes at a time
    while ptr.add(8) <= end {
      crc = crc32c_u64(crc, (ptr as *const u64).read_unaligned());
      ptr = ptr.add(8);
    }

    // Process 4 bytes
    if ptr.add(4) <= end {
      crc = _mm_crc32_u32(crc, (ptr as *const u32).read_unaligned());
      ptr = ptr.add(4);
    }

    // Process 2 bytes
    if ptr.add(2) <= end {
      crc = _mm_crc32_u16(crc, (ptr as *const u16).read_unaligned());
      ptr = ptr.add(2);
    }

    // Process 1 byte
    if ptr < end {
      crc = _mm_crc32_u8(crc, *ptr);
    }
  }

  crc
}

/// Safe wrapper for SSE4.2 CRC32C.
#[inline]
pub fn crc32c_sse42_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSE4.2 before selecting this kernel.
  unsafe { crc32c_sse42(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC32C: PCLMUL + SSE4.2 3-Stream Fusion (v4s3x3)
// ─────────────────────────────────────────────────────────────────────────────
//
// This is the high-performance implementation from crc-fast-rust/corsix.
// It runs 4 PCLMUL lanes in parallel with 3 SSE4.2 CRC streams on
// different data segments, achieving ~45 GB/s on modern CPUs.

/// CRC32C using PCLMUL + SSE4.2 3-stream interleaved fusion.
///
/// For buffers >= 144 bytes, this interleaves:
/// - 4 PCLMUL folding lanes (64 bytes/iteration)
/// - 3 parallel SSE4.2 CRC32C streams (24 bytes/iteration each)
///
/// # Safety
///
/// Caller must ensure SSE4.2 and PCLMULQDQ are available.
#[target_feature(enable = "sse4.2", enable = "pclmulqdq")]
pub unsafe fn crc32c_fusion(mut crc0: u32, data: &[u8]) -> u32 {
  // SAFETY: All pointer operations are bounds-checked
  unsafe {
    let mut ptr = data.as_ptr();
    let mut len = data.len();

    // Align to 8-byte boundary using hardware CRC32C
    while len > 0 && (ptr as usize & 7) != 0 {
      crc0 = _mm_crc32_u8(crc0, *ptr);
      ptr = ptr.add(1);
      len -= 1;
    }

    // Handle 8-byte alignment
    if (ptr as usize & 8) != 0 && len >= 8 {
      crc0 = crc32c_u64(crc0, *(ptr as *const u64));
      ptr = ptr.add(8);
      len -= 8;
    }

    // Main 3-stream fusion loop (requires >= 144 bytes)
    if len >= 144 {
      let blk = (len - 8) / 136;
      let klen = blk * 24;
      let buf2 = ptr;
      let mut crc1 = 0u32;
      let mut crc2 = 0u32;

      // Load 4 PCLMUL lanes (64 bytes)
      let mut x0 = _mm_loadu_si128(buf2 as *const __m128i);
      let mut x1 = _mm_loadu_si128(buf2.add(16) as *const __m128i);
      let mut x2 = _mm_loadu_si128(buf2.add(32) as *const __m128i);
      let mut x3 = _mm_loadu_si128(buf2.add(48) as *const __m128i);

      // Folding constant for 128-byte blocks
      let mut k = _mm_setr_epi32(
        crc32c_constants::FOLD_128.0 as i32,
        0,
        crc32c_constants::FOLD_128.1 as i32,
        0,
      );

      // XOR CRC into first lane
      x0 = _mm_xor_si128(_mm_cvtsi32_si128(crc0 as i32), x0);
      crc0 = 0;

      let mut buf2 = buf2.add(64);
      len -= 136;
      ptr = ptr.add(blk * 64);

      // Main loop: 136 bytes per iteration (64 CLMUL + 72 scalar CRC)
      while len >= 144 {
        // CLMUL folding on 4 lanes
        let mut y0 = clmul_lo(x0, k);
        x0 = clmul_hi(x0, k);
        let mut y1 = clmul_lo(x1, k);
        x1 = clmul_hi(x1, k);
        let mut y2 = clmul_lo(x2, k);
        x2 = clmul_hi(x2, k);
        let mut y3 = clmul_lo(x3, k);
        x3 = clmul_hi(x3, k);

        // XOR with new data
        y0 = _mm_xor_si128(y0, _mm_loadu_si128(buf2 as *const __m128i));
        x0 = _mm_xor_si128(x0, y0);
        y1 = _mm_xor_si128(y1, _mm_loadu_si128(buf2.add(16) as *const __m128i));
        x1 = _mm_xor_si128(x1, y1);
        y2 = _mm_xor_si128(y2, _mm_loadu_si128(buf2.add(32) as *const __m128i));
        x2 = _mm_xor_si128(x2, y2);
        y3 = _mm_xor_si128(y3, _mm_loadu_si128(buf2.add(48) as *const __m128i));
        x3 = _mm_xor_si128(x3, y3);

        // Parallel scalar CRC on 3 streams (24 bytes each)
        crc0 = crc32c_u64(crc0, *(ptr as *const u64));
        crc1 = crc32c_u64(crc1, *(ptr.add(klen) as *const u64));
        crc2 = crc32c_u64(crc2, *(ptr.add(klen * 2) as *const u64));
        crc0 = crc32c_u64(crc0, *(ptr.add(8) as *const u64));
        crc1 = crc32c_u64(crc1, *(ptr.add(klen + 8) as *const u64));
        crc2 = crc32c_u64(crc2, *(ptr.add(klen * 2 + 8) as *const u64));
        crc0 = crc32c_u64(crc0, *(ptr.add(16) as *const u64));
        crc1 = crc32c_u64(crc1, *(ptr.add(klen + 16) as *const u64));
        crc2 = crc32c_u64(crc2, *(ptr.add(klen * 2 + 16) as *const u64));

        ptr = ptr.add(24);
        buf2 = buf2.add(64);
        len -= 136;
      }

      // Reduce x0-x3 to x0 using 16-byte fold constant
      k = _mm_setr_epi32(
        crc32c_constants::FOLD_16.0 as i32,
        0,
        crc32c_constants::FOLD_16.1 as i32,
        0,
      );

      let mut y0 = clmul_lo(x0, k);
      x0 = clmul_hi(x0, k);
      let mut y2 = clmul_lo(x2, k);
      x2 = clmul_hi(x2, k);

      y0 = _mm_xor_si128(y0, x1);
      x0 = _mm_xor_si128(x0, y0);
      y2 = _mm_xor_si128(y2, x3);
      x2 = _mm_xor_si128(x2, y2);

      k = _mm_setr_epi32(
        crc32c_constants::REDUCE_64.0 as i32,
        0,
        crc32c_constants::REDUCE_64.1 as i32,
        0,
      );

      y0 = clmul_lo(x0, k);
      x0 = clmul_hi(x0, k);
      y0 = _mm_xor_si128(y0, x2);
      x0 = _mm_xor_si128(x0, y0);

      // Final scalar chunk
      crc0 = crc32c_u64(crc0, *(ptr as *const u64));
      crc1 = crc32c_u64(crc1, *(ptr.add(klen) as *const u64));
      crc2 = crc32c_u64(crc2, *(ptr.add(klen * 2) as *const u64));
      crc0 = crc32c_u64(crc0, *(ptr.add(8) as *const u64));
      crc1 = crc32c_u64(crc1, *(ptr.add(klen + 8) as *const u64));
      crc2 = crc32c_u64(crc2, *(ptr.add(klen * 2 + 8) as *const u64));
      crc0 = crc32c_u64(crc0, *(ptr.add(16) as *const u64));
      crc1 = crc32c_u64(crc1, *(ptr.add(klen + 16) as *const u64));
      crc2 = crc32c_u64(crc2, *(ptr.add(klen * 2 + 16) as *const u64));
      ptr = ptr.add(24);

      // Combine the 3 CRC streams
      let vc0 = crc_shift_crc32c(crc0, klen * 2 + 8);
      let vc1 = crc_shift_crc32c(crc1, klen + 8);
      let mut vc = extract_epi64(_mm_xor_si128(vc0, vc1), 0);

      // Reduce CLMUL result (128→32) using hardware CRC
      let x0_low = extract_epi64(x0, 0);
      let x0_high = extract_epi64(x0, 1);
      let x0_combined = extract_epi64(
        crc_shift_crc32c(crc32c_u64(crc32c_u64(0, x0_low), x0_high), klen * 3 + 8),
        0,
      );
      vc ^= x0_combined;

      // Final combination
      ptr = ptr.add(klen * 2);
      crc0 = crc2;
      crc0 = crc32c_u64(crc0, *(ptr as *const u64) ^ vc);
      ptr = ptr.add(8);
      len -= 8;
    }

    // Process remaining 8-byte chunks
    while len >= 8 {
      crc0 = crc32c_u64(crc0, *(ptr as *const u64));
      ptr = ptr.add(8);
      len -= 8;
    }

    // Process remaining bytes
    while len > 0 {
      crc0 = _mm_crc32_u8(crc0, *ptr);
      ptr = ptr.add(1);
      len -= 1;
    }

    crc0
  }
}

/// Safe wrapper for CRC32C PCLMUL+SSE4.2 fusion.
#[inline]
pub fn crc32c_pclmul_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSE4.2+PCLMUL before selecting this kernel.
  unsafe { crc32c_fusion(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC32-IEEE: Pure PCLMUL + Barrett (no hardware instruction)
// ─────────────────────────────────────────────────────────────────────────────

/// Reduce 128 bits to 32 bits for CRC32-IEEE.
///
/// Algorithm from crc32fast / Intel's "Fast CRC Computation" paper.
/// Stages: 128→64 using K3, 64→32 using K5, then Barrett with P_X/U_PRIME.
#[target_feature(enable = "pclmulqdq", enable = "sse4.1")]
unsafe fn barrett_reduce_ieee(mut x: __m128i, k3k4: __m128i) -> u32 {
  // SAFETY: target_feature ensures PCLMULQDQ and SSE4.1 availability

  // Stage 1: 128→64 bits using K3 (selector 0x10 = x.hi × k3k4.lo = x.hi × K3)
  x = _mm_xor_si128(_mm_clmulepi64_si128(x, k3k4, 0x10), _mm_srli_si128(x, 8));

  // Stage 2: 64→32 bits using K5
  // Mask to get lower 32 bits, multiply by K5, XOR with upper 32 bits
  let k5 = _mm_set_epi64x(0, crc32_ieee_constants::K5 as i64);
  x = _mm_xor_si128(
    _mm_clmulepi64_si128(_mm_and_si128(x, _mm_set_epi32(0, 0, 0, !0)), k5, 0x00),
    _mm_srli_si128(x, 4),
  );

  // Stage 3: Barrett reduction
  // pu = [U_PRIME, P_X] - note the order for selector usage
  let pu = _mm_set_epi64x(crc32_ieee_constants::U_PRIME as i64, crc32_ieee_constants::P_X as i64);

  // t1 = (x & 0xFFFFFFFF) × U_PRIME (selector 0x10 = x.lo × pu.hi)
  let t1 = _mm_clmulepi64_si128(_mm_and_si128(x, _mm_set_epi32(0, 0, 0, !0)), pu, 0x10);

  // t2 = (t1 & 0xFFFFFFFF) × P_X (selector 0x00 = t1.lo × pu.lo)
  let t2 = _mm_clmulepi64_si128(_mm_and_si128(t1, _mm_set_epi32(0, 0, 0, !0)), pu, 0x00);

  // Result is in bits [63:32] after XOR
  _mm_extract_epi32(_mm_xor_si128(x, t2), 1) as u32
}

/// CRC32-IEEE using PCLMULQDQ folding + Barrett reduction.
///
/// # Safety
///
/// Caller must ensure PCLMULQDQ and SSE4.1 are available.
#[target_feature(enable = "pclmulqdq", enable = "sse4.1")]
pub unsafe fn crc32_ieee_pclmul(mut crc: u32, data: &[u8]) -> u32 {
  const FOLD_BYTES: usize = 64;

  if data.len() < FOLD_BYTES {
    return crc32_ieee_slice16(crc, data);
  }

  // SAFETY: All pointer operations are bounds-checked
  unsafe {
    let mut ptr = data.as_ptr();
    let end = ptr.add(data.len());

    // K1K2 for 64-byte fold, K3K4 for 16-byte fold
    let k1k2 = _mm_set_epi64x(crc32_ieee_constants::K2 as i64, crc32_ieee_constants::K1 as i64);
    let k3k4 = _mm_set_epi64x(crc32_ieee_constants::K4 as i64, crc32_ieee_constants::K3 as i64);

    // Initialize 4 lanes with first 64 bytes
    let crc_vec = _mm_set_epi32(0, 0, 0, crc as i32);
    let mut lanes = [
      _mm_xor_si128(_mm_loadu_si128(ptr.add(0) as *const __m128i), crc_vec),
      _mm_loadu_si128(ptr.add(16) as *const __m128i),
      _mm_loadu_si128(ptr.add(32) as *const __m128i),
      _mm_loadu_si128(ptr.add(48) as *const __m128i),
    ];
    ptr = ptr.add(FOLD_BYTES);

    // Process 64-byte blocks using K1K2
    while ptr.add(FOLD_BYTES) <= end {
      for (i, lane) in lanes.iter_mut().enumerate() {
        let new_data = _mm_loadu_si128(ptr.add(i * 16) as *const __m128i);
        let lo = clmul_lo(*lane, k1k2);
        let hi = clmul_hi(*lane, k1k2);
        *lane = _mm_xor_si128(_mm_xor_si128(lo, hi), new_data);
      }
      ptr = ptr.add(FOLD_BYTES);
    }

    // Reduce 4 lanes to 1 using K3K4
    let fold_48 = {
      let lo = clmul_lo(lanes[0], k3k4);
      let hi = clmul_hi(lanes[0], k3k4);
      _mm_xor_si128(_mm_xor_si128(lo, hi), lanes[1])
    };
    let fold_32 = {
      let lo = clmul_lo(fold_48, k3k4);
      let hi = clmul_hi(fold_48, k3k4);
      _mm_xor_si128(_mm_xor_si128(lo, hi), lanes[2])
    };
    let mut acc = {
      let lo = clmul_lo(fold_32, k3k4);
      let hi = clmul_hi(fold_32, k3k4);
      _mm_xor_si128(_mm_xor_si128(lo, hi), lanes[3])
    };

    // Process remaining 16-byte blocks using K3K4
    while ptr.add(16) <= end {
      let block = _mm_loadu_si128(ptr as *const __m128i);
      let lo = clmul_lo(acc, k3k4);
      let hi = clmul_hi(acc, k3k4);
      acc = _mm_xor_si128(_mm_xor_si128(lo, hi), block);
      ptr = ptr.add(16);
    }

    // Reduce 128→32 bits
    crc = barrett_reduce_ieee(acc, k3k4);

    // Handle remainder with portable
    let remainder_len = end.offset_from(ptr) as usize;
    if remainder_len > 0 {
      crc = crc32_ieee_slice16(crc, core::slice::from_raw_parts(ptr, remainder_len));
    }

    crc
  }
}

/// Safe wrapper for CRC32-IEEE PCLMUL.
#[inline]
pub fn crc32_ieee_pclmul_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies PCLMUL before selecting this kernel.
  unsafe { crc32_ieee_pclmul(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// VPCLMULQDQ (AVX-512) Implementation
// ─────────────────────────────────────────────────────────────────────────────
//
// Uses AVX-512 VPCLMULQDQ for 4-way parallel carryless multiply with VPTERNLOGD
// for fused 3-way XOR operations. Processes 128-byte blocks (8×16B lanes).
//
// Performance target: ~35-60 GB/s on modern CPUs (Ice Lake+, Zen 4+).

use crate::common::clmul::{CRC32_IEEE_CLMUL, CRC32C_CLMUL};

/// Broadcast 128-bit fold coefficient across 4 lanes of a 512-bit vector.
///
/// For CRC-32, the coefficient pair is (K_high, K_low) stored as 64-bit values
/// but only the low 32 bits are meaningful. VPCLMULQDQ operates on 128-bit lanes.
#[inline]
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn vpclmul_coeff_32(pair: (u64, u64)) -> __m512i {
  _mm512_set_epi64(
    pair.0 as i64,
    pair.1 as i64,
    pair.0 as i64,
    pair.1 as i64,
    pair.0 as i64,
    pair.1 as i64,
    pair.0 as i64,
    pair.1 as i64,
  )
}

/// Fold and XOR using VPTERNLOGD (3-way XOR in one instruction).
///
/// Computes: `data ^ clmul_hi(x, coeff) ^ clmul_lo(x, coeff)`
#[inline]
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn fold16_4x_ternlog_32(x: __m512i, data: __m512i, coeff: __m512i) -> __m512i {
  let h = _mm512_clmulepi64_epi128::<0x11>(x, coeff);
  let l = _mm512_clmulepi64_epi128::<0x00>(x, coeff);
  // VPTERNLOGD: 3-way XOR (imm8 = 0x96 = a ^ b ^ c)
  _mm512_ternarylogic_epi64::<0x96>(data, h, l)
}

/// Fold without data XOR (for lane reduction).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn fold16_4x_32(x: __m512i, coeff: __m512i) -> __m512i {
  let h = _mm512_clmulepi64_epi128::<0x11>(x, coeff);
  let l = _mm512_clmulepi64_epi128::<0x00>(x, coeff);
  _mm512_xor_si512(h, l)
}

/// Extract 8×16B lanes from two 512-bit vectors.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn extract_lanes_512(x0: __m512i, x1: __m512i) -> [__m128i; 8] {
  [
    _mm512_extracti32x4_epi32::<0>(x0),
    _mm512_extracti32x4_epi32::<1>(x0),
    _mm512_extracti32x4_epi32::<2>(x0),
    _mm512_extracti32x4_epi32::<3>(x0),
    _mm512_extracti32x4_epi32::<0>(x1),
    _mm512_extracti32x4_epi32::<1>(x1),
    _mm512_extracti32x4_epi32::<2>(x1),
    _mm512_extracti32x4_epi32::<3>(x1),
  ]
}

/// Fold one 128-bit lane into accumulator using a fold constant.
#[inline]
#[target_feature(enable = "pclmulqdq", enable = "sse4.1")]
unsafe fn fold_lane(acc: __m128i, next: __m128i, k: __m128i) -> __m128i {
  // SAFETY: target_feature ensures PCLMULQDQ and SSE4.1 are available.
  unsafe {
    let lo = clmul_lo(acc, k);
    let hi = clmul_hi(acc, k);
    _mm_xor_si128(_mm_xor_si128(lo, hi), next)
  }
}

/// Reduce 8×16B lanes to a single 128-bit value using a fold constant.
#[inline]
#[target_feature(enable = "pclmulqdq", enable = "sse4.1")]
unsafe fn reduce_8_lanes(lanes: [__m128i; 8], k: __m128i) -> __m128i {
  // SAFETY: target_feature ensures PCLMULQDQ and SSE4.1 are available.
  // fold_lane has the same target_feature requirements.
  unsafe {
    // Fold lanes[0..8] → single lane
    let mut acc = fold_lane(lanes[0], lanes[1], k);
    acc = fold_lane(acc, lanes[2], k);
    acc = fold_lane(acc, lanes[3], k);
    acc = fold_lane(acc, lanes[4], k);
    acc = fold_lane(acc, lanes[5], k);
    acc = fold_lane(acc, lanes[6], k);
    fold_lane(acc, lanes[7], k)
  }
}

/// CRC32-IEEE using VPCLMULQDQ with 128-byte blocks.
///
/// # Safety
///
/// Caller must ensure AVX-512F, AVX-512VL, AVX-512BW, and VPCLMULQDQ are available.
#[target_feature(
  enable = "avx512f",
  enable = "avx512vl",
  enable = "avx512bw",
  enable = "vpclmulqdq",
  enable = "pclmulqdq",
  enable = "sse4.1"
)]
pub unsafe fn crc32_ieee_vpclmul(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: All operations require the target features enabled by #[target_feature].
  // Caller guarantees AVX-512F, AVX-512VL, AVX-512BW, VPCLMULQDQ, PCLMULQDQ, SSE4.1.
  unsafe {
    const BLOCK_SIZE: usize = 128;

    // Fall back to PCLMUL for small buffers
    if data.len() < BLOCK_SIZE {
      return crc32_ieee_pclmul(crc, data);
    }

    let consts = &CRC32_IEEE_CLMUL;
    let mut ptr = data.as_ptr();
    let end = ptr.add(data.len());

    // Load first 128 bytes (2×__m512i = 8×16B lanes)
    let mut x0 = _mm512_loadu_si512(ptr.cast::<__m512i>());
    let mut x1 = _mm512_loadu_si512(ptr.add(64).cast::<__m512i>());
    ptr = ptr.add(BLOCK_SIZE);

    // XOR initial CRC into lane 0
    let crc_mask = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, crc as i32);
    x0 = _mm512_xor_si512(x0, crc_mask);

    // Broadcast 128B fold coefficient
    let coeff_128b = vpclmul_coeff_32(consts.fold_128b);

    // Main loop: process 128-byte blocks
    while ptr.add(BLOCK_SIZE) <= end {
      let y0 = _mm512_loadu_si512(ptr.cast::<__m512i>());
      let y1 = _mm512_loadu_si512(ptr.add(64).cast::<__m512i>());
      ptr = ptr.add(BLOCK_SIZE);

      x0 = fold16_4x_ternlog_32(x0, y0, coeff_128b);
      x1 = fold16_4x_ternlog_32(x1, y1, coeff_128b);
    }

    // Extract 8 lanes and reduce to single 128-bit value
    let lanes = extract_lanes_512(x0, x1);
    let k3k4 = _mm_set_epi64x(crc32_ieee_constants::K4 as i64, crc32_ieee_constants::K3 as i64);
    let mut acc = reduce_8_lanes(lanes, k3k4);

    // Process remaining 16-byte blocks
    while ptr.add(16) <= end {
      let block = _mm_loadu_si128(ptr.cast::<__m128i>());
      ptr = ptr.add(16);
      acc = fold_lane(acc, block, k3k4);
    }

    // Barrett reduction: 128→32 bits
    let result = barrett_reduce_ieee(acc, k3k4);

    // Handle remaining bytes with portable
    let remainder_len = end.offset_from(ptr) as usize;
    if remainder_len > 0 {
      crc32_ieee_slice16(result, core::slice::from_raw_parts(ptr, remainder_len))
    } else {
      result
    }
  }
}

/// Safe wrapper for CRC32-IEEE VPCLMUL.
#[inline]
pub fn crc32_ieee_vpclmul_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies VPCLMUL_READY before selecting this kernel.
  unsafe { crc32_ieee_vpclmul(crc, data) }
}

/// CRC32C using VPCLMULQDQ with 128-byte blocks.
///
/// Uses the same VPCLMUL folding as IEEE but leverages SSE4.2 CRC32C
/// instruction for final reduction (faster than Barrett).
///
/// # Safety
///
/// Caller must ensure AVX-512F, AVX-512VL, AVX-512BW, VPCLMULQDQ, and SSE4.2 are available.
#[target_feature(
  enable = "avx512f",
  enable = "avx512vl",
  enable = "avx512bw",
  enable = "vpclmulqdq",
  enable = "pclmulqdq",
  enable = "sse4.2"
)]
pub unsafe fn crc32c_vpclmul(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: All operations require the target features enabled by #[target_feature].
  // Caller guarantees AVX-512F, AVX-512VL, AVX-512BW, VPCLMULQDQ, PCLMULQDQ, SSE4.2.
  unsafe {
    const BLOCK_SIZE: usize = 128;

    // Fall back to fusion for small buffers (fusion is faster for < 128 bytes)
    if data.len() < BLOCK_SIZE {
      return crc32c_fusion(crc, data);
    }

    let consts = &CRC32C_CLMUL;
    let mut ptr = data.as_ptr();
    let end = ptr.add(data.len());

    // Load first 128 bytes (2×__m512i = 8×16B lanes)
    let mut x0 = _mm512_loadu_si512(ptr.cast::<__m512i>());
    let mut x1 = _mm512_loadu_si512(ptr.add(64).cast::<__m512i>());
    ptr = ptr.add(BLOCK_SIZE);

    // XOR initial CRC into lane 0
    let crc_mask = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, crc as i32);
    x0 = _mm512_xor_si512(x0, crc_mask);

    // Broadcast 128B fold coefficient
    let coeff_128b = vpclmul_coeff_32(consts.fold_128b);

    // Main loop: process 128-byte blocks
    while ptr.add(BLOCK_SIZE) <= end {
      let y0 = _mm512_loadu_si512(ptr.cast::<__m512i>());
      let y1 = _mm512_loadu_si512(ptr.add(64).cast::<__m512i>());
      ptr = ptr.add(BLOCK_SIZE);

      x0 = fold16_4x_ternlog_32(x0, y0, coeff_128b);
      x1 = fold16_4x_ternlog_32(x1, y1, coeff_128b);
    }

    // Extract 8 lanes and reduce using 16B fold constant
    let lanes = extract_lanes_512(x0, x1);
    let k16 = _mm_setr_epi32(
      crc32c_constants::FOLD_16.0 as i32,
      0,
      crc32c_constants::FOLD_16.1 as i32,
      0,
    );

    // Reduce 8 lanes → 1 lane
    let mut acc = reduce_8_lanes(lanes, k16);

    // Process remaining 16-byte blocks
    while ptr.add(16) <= end {
      let block = _mm_loadu_si128(ptr.cast::<__m128i>());
      ptr = ptr.add(16);
      acc = fold_lane(acc, block, k16);
    }

    // Final reduction: 128→32 bits using SSE4.2 CRC32C instruction
    // First reduce 128→64 using REDUCE_64 constant
    let k64 = _mm_setr_epi32(
      crc32c_constants::REDUCE_64.0 as i32,
      0,
      crc32c_constants::REDUCE_64.1 as i32,
      0,
    );

    let lo = clmul_lo(acc, k64);
    let hi = clmul_hi(acc, k64);
    acc = _mm_xor_si128(lo, hi);

    // Extract 64-bit value and use SSE4.2 CRC32C for final reduction
    let val64 = extract_epi64(acc, 0);
    let mut result = crc32c_u64(0, val64);

    // Handle remaining bytes with SSE4.2
    while ptr < end {
      result = _mm_crc32_u8(result, *ptr);
      ptr = ptr.add(1);
    }

    result
  }
}

/// Safe wrapper for CRC32C VPCLMUL.
#[inline]
pub fn crc32c_vpclmul_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies VPCLMUL_READY before selecting this kernel.
  unsafe { crc32c_vpclmul(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  extern crate alloc;
  use alloc::vec::Vec;

  use super::*;
  use crate::crc32::portable::crc32c_slice16;

  // Check values from crc-fast-rust
  const CRC32C_CHECK: u32 = 0xE3069283; // "123456789"
  const CRC32_IEEE_CHECK: u32 = 0xCBF43926; // "123456789"

  const CRC32C_HELLO_WORLD: u32 = 0xC99465AA;
  const CRC32_IEEE_HELLO_WORLD: u32 = 0x0D4A_1185;

  #[test]
  fn test_crc32c_sse42_check() {
    if !std::is_x86_feature_detected!("sse4.2") {
      return;
    }
    let result = crc32c_sse42_safe(!0, b"123456789") ^ !0;
    assert_eq!(result, CRC32C_CHECK);
  }

  #[test]
  fn test_crc32c_sse42_hello() {
    if !std::is_x86_feature_detected!("sse4.2") {
      return;
    }
    let result = crc32c_sse42_safe(!0, b"hello world") ^ !0;
    assert_eq!(result, CRC32C_HELLO_WORLD);
  }

  #[test]
  fn test_crc32c_fusion_check() {
    if !std::is_x86_feature_detected!("sse4.2") || !std::is_x86_feature_detected!("pclmulqdq") {
      return;
    }
    let result = crc32c_pclmul_safe(!0, b"123456789") ^ !0;
    assert_eq!(result, CRC32C_CHECK);
  }

  #[test]
  fn test_crc32c_fusion_vs_sse42() {
    if !std::is_x86_feature_detected!("sse4.2") || !std::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [16, 32, 64, 128, 144, 192, 256, 384, 512, 1024, 4096] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let sse42 = crc32c_sse42_safe(!0, &data) ^ !0;
      let fusion = crc32c_pclmul_safe(!0, &data) ^ !0;

      assert_eq!(sse42, fusion, "CRC32C mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32c_fusion_vs_portable() {
    if !std::is_x86_feature_detected!("sse4.2") || !std::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [16, 32, 64, 128, 144, 192, 256, 384, 512, 1024, 4096] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let portable = crc32c_slice16(!0, &data) ^ !0;
      let fusion = crc32c_pclmul_safe(!0, &data) ^ !0;

      assert_eq!(portable, fusion, "CRC32C mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32_ieee_pclmul_check() {
    if !std::is_x86_feature_detected!("pclmulqdq") || !std::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let result = crc32_ieee_pclmul_safe(!0, b"123456789") ^ !0;
    assert_eq!(result, CRC32_IEEE_CHECK);
  }

  #[test]
  fn test_crc32_ieee_pclmul_hello() {
    if !std::is_x86_feature_detected!("pclmulqdq") || !std::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let result = crc32_ieee_pclmul_safe(!0, b"hello world") ^ !0;
    assert_eq!(result, CRC32_IEEE_HELLO_WORLD);
  }

  #[test]
  fn test_crc32_ieee_pclmul_vs_portable() {
    if !std::is_x86_feature_detected!("pclmulqdq") || !std::is_x86_feature_detected!("sse4.1") {
      return;
    }

    for len in [64, 128, 256, 512, 1024, 4096] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let portable = crc32_ieee_slice16(!0, &data) ^ !0;
      let pclmul = crc32_ieee_pclmul_safe(!0, &data) ^ !0;

      assert_eq!(portable, pclmul, "CRC32-IEEE mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32c_fusion_large() {
    if !std::is_x86_feature_detected!("sse4.2") || !std::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    // Test with 1 MiB
    let data: Vec<u8> = (0..1048576u32).map(|i| i as u8).collect();

    let sse42 = crc32c_sse42_safe(!0, &data) ^ !0;
    let fusion = crc32c_pclmul_safe(!0, &data) ^ !0;

    assert_eq!(sse42, fusion, "CRC32C mismatch on 1 MiB");
  }

  #[test]
  fn test_crc32_ieee_pclmul_large() {
    if !std::is_x86_feature_detected!("pclmulqdq") || !std::is_x86_feature_detected!("sse4.1") {
      return;
    }

    // Test with 1 MiB
    let data: Vec<u8> = (0..1048576u32).map(|i| i as u8).collect();

    let portable = crc32_ieee_slice16(!0, &data) ^ !0;
    let pclmul = crc32_ieee_pclmul_safe(!0, &data) ^ !0;

    assert_eq!(portable, pclmul, "CRC32-IEEE mismatch on 1 MiB");
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // VPCLMULQDQ (AVX-512) Tests
  // ─────────────────────────────────────────────────────────────────────────────

  /// Helper to detect VPCLMUL capability.
  fn has_vpclmul() -> bool {
    std::is_x86_feature_detected!("avx512f")
      && std::is_x86_feature_detected!("avx512vl")
      && std::is_x86_feature_detected!("avx512bw")
      && std::is_x86_feature_detected!("vpclmulqdq")
      && std::is_x86_feature_detected!("pclmulqdq")
  }

  #[test]
  fn test_crc32_ieee_vpclmul_vs_portable() {
    if !has_vpclmul() {
      return;
    }

    // Test various buffer sizes around block boundaries
    for len in [128, 144, 192, 256, 384, 512, 640, 768, 896, 1024, 2048, 4096, 8192] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let portable = crc32_ieee_slice16(!0, &data) ^ !0;
      let vpclmul = crc32_ieee_vpclmul_safe(!0, &data) ^ !0;

      assert_eq!(portable, vpclmul, "CRC32-IEEE VPCLMUL mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32_ieee_vpclmul_vs_pclmul() {
    if !has_vpclmul() {
      return;
    }

    for len in [128, 256, 512, 1024, 4096, 16384] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let pclmul = crc32_ieee_pclmul_safe(!0, &data) ^ !0;
      let vpclmul = crc32_ieee_vpclmul_safe(!0, &data) ^ !0;

      assert_eq!(pclmul, vpclmul, "CRC32-IEEE VPCLMUL vs PCLMUL mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32_ieee_vpclmul_large() {
    if !has_vpclmul() {
      return;
    }

    // Test with 1 MiB
    let data: Vec<u8> = (0..1048576u32).map(|i| i as u8).collect();

    let portable = crc32_ieee_slice16(!0, &data) ^ !0;
    let vpclmul = crc32_ieee_vpclmul_safe(!0, &data) ^ !0;

    assert_eq!(portable, vpclmul, "CRC32-IEEE VPCLMUL mismatch on 1 MiB");
  }

  #[test]
  fn test_crc32c_vpclmul_vs_portable() {
    if !has_vpclmul() || !std::is_x86_feature_detected!("sse4.2") {
      return;
    }

    // Test various buffer sizes around block boundaries
    for len in [128, 144, 192, 256, 384, 512, 640, 768, 896, 1024, 2048, 4096, 8192] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let portable = crc32c_slice16(!0, &data) ^ !0;
      let vpclmul = crc32c_vpclmul_safe(!0, &data) ^ !0;

      assert_eq!(portable, vpclmul, "CRC32C VPCLMUL mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32c_vpclmul_vs_sse42() {
    if !has_vpclmul() || !std::is_x86_feature_detected!("sse4.2") {
      return;
    }

    for len in [128, 256, 512, 1024, 4096, 16384] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let sse42 = crc32c_sse42_safe(!0, &data) ^ !0;
      let vpclmul = crc32c_vpclmul_safe(!0, &data) ^ !0;

      assert_eq!(sse42, vpclmul, "CRC32C VPCLMUL vs SSE4.2 mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32c_vpclmul_vs_fusion() {
    if !has_vpclmul() || !std::is_x86_feature_detected!("sse4.2") {
      return;
    }

    for len in [128, 256, 512, 1024, 4096, 16384] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let fusion = crc32c_pclmul_safe(!0, &data) ^ !0;
      let vpclmul = crc32c_vpclmul_safe(!0, &data) ^ !0;

      assert_eq!(fusion, vpclmul, "CRC32C VPCLMUL vs fusion mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32c_vpclmul_large() {
    if !has_vpclmul() || !std::is_x86_feature_detected!("sse4.2") {
      return;
    }

    // Test with 1 MiB
    let data: Vec<u8> = (0..1048576u32).map(|i| i as u8).collect();

    let portable = crc32c_slice16(!0, &data) ^ !0;
    let vpclmul = crc32c_vpclmul_safe(!0, &data) ^ !0;

    assert_eq!(portable, vpclmul, "CRC32C VPCLMUL mismatch on 1 MiB");
  }

  #[test]
  fn test_crc32_ieee_vpclmul_check_value() {
    if !has_vpclmul() {
      return;
    }

    // Need >= 128 bytes for VPCLMUL, so pad the check data
    let mut data = Vec::with_capacity(256);
    data.extend_from_slice(b"123456789");
    data.resize(256, 0);

    // Compute expected with portable
    let expected = crc32_ieee_slice16(!0, &data) ^ !0;
    let vpclmul = crc32_ieee_vpclmul_safe(!0, &data) ^ !0;

    assert_eq!(expected, vpclmul, "CRC32-IEEE VPCLMUL check value mismatch");
  }

  #[test]
  fn test_crc32c_vpclmul_check_value() {
    if !has_vpclmul() || !std::is_x86_feature_detected!("sse4.2") {
      return;
    }

    // Need >= 128 bytes for VPCLMUL, so pad the check data
    let mut data = Vec::with_capacity(256);
    data.extend_from_slice(b"123456789");
    data.resize(256, 0);

    // Compute expected with portable
    let expected = crc32c_slice16(!0, &data) ^ !0;
    let vpclmul = crc32c_vpclmul_safe(!0, &data) ^ !0;

    assert_eq!(expected, vpclmul, "CRC32C VPCLMUL check value mismatch");
  }

  #[test]
  fn test_crc32_vpclmul_boundary_lengths() {
    if !has_vpclmul() || !std::is_x86_feature_detected!("sse4.2") {
      return;
    }

    // Test exact block boundaries and off-by-one cases
    for len in [127, 128, 129, 143, 144, 145, 255, 256, 257, 383, 384, 385] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      // CRC32-IEEE
      let ieee_portable = crc32_ieee_slice16(!0, &data) ^ !0;
      let ieee_vpclmul = crc32_ieee_vpclmul_safe(!0, &data) ^ !0;
      assert_eq!(ieee_portable, ieee_vpclmul, "CRC32-IEEE boundary mismatch at {len}");

      // CRC32C
      let c_portable = crc32c_slice16(!0, &data) ^ !0;
      let c_vpclmul = crc32c_vpclmul_safe(!0, &data) ^ !0;
      assert_eq!(c_portable, c_vpclmul, "CRC32C boundary mismatch at {len}");
    }
  }
}
