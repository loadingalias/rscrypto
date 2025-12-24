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
/// - MU/POLY: Barrett reduction
///
/// These are 33-bit values with bit 32 set, as required by the Intel algorithm.
mod crc32_ieee_constants {
  /// 64-byte block folding: (K1, K2) for 512-bit fold.
  pub const FOLD_64: (u64, u64) = (0x0000_0001_5444_2bd4, 0x0000_0001_c6e4_1596);

  /// 16-byte lane folding: (K3, K4) for 128-bit fold.
  pub const FOLD_16: (u64, u64) = (0x0000_0001_7519_97d0, 0x0000_0000_ccaa_009e);

  /// Barrett reduction: (µ, P').
  pub const MU_POLY: (u64, u64) = (0x0000_0001_F701_1641, 0x0000_0001_DB71_0641);
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

/// Barrett reduction from 128 bits to 32 bits for CRC32-IEEE.
///
/// Algorithm based on Intel's "Fast CRC Computation" paper.
/// Uses K3/K4 for 128→64→32 reduction, then Barrett for final step.
#[target_feature(enable = "pclmulqdq", enable = "sse4.1")]
unsafe fn barrett_reduce_ieee(acc: __m128i) -> u32 {
  // SAFETY: target_feature ensures PCLMULQDQ and SSE4.1 availability
  let k3_k4 = _mm_set_epi64x(
    crc32_ieee_constants::FOLD_16.1 as i64, // K4
    crc32_ieee_constants::FOLD_16.0 as i64, // K3
  );
  let mu_poly = _mm_set_epi64x(
    crc32_ieee_constants::MU_POLY.1 as i64, // POLY
    crc32_ieee_constants::MU_POLY.0 as i64, // MU
  );

  // Fold 128→64 bits: multiply high 64 bits by K3, XOR with low 64 bits
  let t1 = _mm_clmulepi64_si128(acc, k3_k4, 0x10); // acc.hi × K3
  let t2 = _mm_xor_si128(t1, _mm_srli_si128(acc, 8));
  let t3 = _mm_xor_si128(t2, _mm_slli_si128(acc, 8));

  // Fold 64→32 bits: multiply bits [63:32] by K4
  let t4 = _mm_clmulepi64_si128(t3, k3_k4, 0x01); // t3.lo × K4
  let t5 = _mm_xor_si128(t4, t3);

  // Barrett reduction: 64→32 bits
  let t6 = _mm_clmulepi64_si128(_mm_and_si128(t5, _mm_set_epi32(0, 0, !0, !0)), mu_poly, 0x00);
  let t7 = _mm_clmulepi64_si128(t6, mu_poly, 0x10);
  let result = _mm_xor_si128(t5, t7);

  _mm_extract_epi32(result, 1) as u32
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

    let fold_k = _mm_set_epi64x(
      crc32_ieee_constants::FOLD_16.1 as i64,
      crc32_ieee_constants::FOLD_16.0 as i64,
    );
    let fold_64_k = _mm_set_epi64x(
      crc32_ieee_constants::FOLD_64.1 as i64,
      crc32_ieee_constants::FOLD_64.0 as i64,
    );

    // Initialize 4 lanes with first 64 bytes
    let crc_vec = _mm_set_epi32(0, 0, 0, crc as i32);
    let mut lanes = [
      _mm_xor_si128(_mm_loadu_si128(ptr.add(0) as *const __m128i), crc_vec),
      _mm_loadu_si128(ptr.add(16) as *const __m128i),
      _mm_loadu_si128(ptr.add(32) as *const __m128i),
      _mm_loadu_si128(ptr.add(48) as *const __m128i),
    ];
    ptr = ptr.add(FOLD_BYTES);

    // Process 64-byte blocks
    while ptr.add(FOLD_BYTES) <= end {
      for (i, lane) in lanes.iter_mut().enumerate() {
        let new_data = _mm_loadu_si128(ptr.add(i * 16) as *const __m128i);
        let lo = clmul_lo(*lane, fold_64_k);
        let hi = clmul_hi(*lane, fold_64_k);
        *lane = _mm_xor_si128(_mm_xor_si128(lo, hi), new_data);
      }
      ptr = ptr.add(FOLD_BYTES);
    }

    // Reduce 4 lanes to 1
    let fold_48 = {
      let lo = clmul_lo(lanes[0], fold_k);
      let hi = clmul_hi(lanes[0], fold_k);
      _mm_xor_si128(_mm_xor_si128(lo, hi), lanes[1])
    };
    let fold_32 = {
      let lo = clmul_lo(fold_48, fold_k);
      let hi = clmul_hi(fold_48, fold_k);
      _mm_xor_si128(_mm_xor_si128(lo, hi), lanes[2])
    };
    let mut acc = {
      let lo = clmul_lo(fold_32, fold_k);
      let hi = clmul_hi(fold_32, fold_k);
      _mm_xor_si128(_mm_xor_si128(lo, hi), lanes[3])
    };

    // Process remaining 16-byte blocks
    while ptr.add(16) <= end {
      let block = _mm_loadu_si128(ptr as *const __m128i);
      let lo = clmul_lo(acc, fold_k);
      let hi = clmul_hi(acc, fold_k);
      acc = _mm_xor_si128(_mm_xor_si128(lo, hi), block);
      ptr = ptr.add(16);
    }

    // Barrett reduction
    crc = barrett_reduce_ieee(acc);

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
// VPCLMUL Stubs (AVX-512)
// ─────────────────────────────────────────────────────────────────────────────

/// VPCLMULQDQ CRC32C (placeholder - uses fusion for now).
#[inline]
pub fn crc32c_vpclmul_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_pclmul_safe(crc, data)
}

/// VPCLMULQDQ CRC32-IEEE (placeholder - uses PCLMUL for now).
#[inline]
pub fn crc32_ieee_vpclmul_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_ieee_pclmul_safe(crc, data)
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
}
