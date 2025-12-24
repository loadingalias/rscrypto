//! aarch64 CRC-32 implementations using CRC extension and PMULL fusion.
//!
//! # Hardware Features
//!
//! - **CRC32 extension**: Dedicated instructions for both CRC32-IEEE and CRC32C
//! - **PMULL + CRC fusion**: PMULL for folding, HW CRC for reduction (~25+ GB/s)
//! - **PMULL + EOR3 + CRC fusion**: SHA3 EOR3 for faster XOR operations
//!
//! # Fusion Approach
//!
//! Unlike pure CLMUL implementations that use Barrett reduction, we use the
//! hardware CRC32 instructions for final 128→32 bit reduction. This is faster
//! because ARM provides dedicated CRC instructions for both polynomials.
//!
//! Reference: <https://dougallj.wordpress.com/2022/05/22/faster-crc32-on-the-apple-m1/>
//! Based on: <https://github.com/corsix/fast-crc32>

// SIMD intrinsics require unsafe; safety is documented per-function.
#![allow(unsafe_code)]

use core::arch::aarch64::*;

// ─────────────────────────────────────────────────────────────────────────────
// PMULL Folding Constants (from crc-fast-rust)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC32C (iSCSI/Castagnoli) folding constants.
/// Format: [k_lo, k_hi] for PMULL fold operations.
mod crc32c_constants {
  /// 192-byte block folding (12 lanes).
  pub const FOLD_192: [u64; 2] = [0xa87ab8a8, 0xab7aff2a];
  /// 16-byte lane folding.
  pub const FOLD_16: [u64; 2] = [0xf20c0dfe, 0x493c7d27];
  /// Reduction: 12→6 lanes (32-byte fold).
  pub const REDUCE_12_TO_6: [u64; 2] = [0xf20c0dfe, 0x493c7d27];
  /// Reduction: 6→3 lanes (64-byte fold).
  pub const REDUCE_6_TO_3: [u64; 2] = [0x3da6d0cb, 0xba4fc28e];
  /// Reduction: 3→1 lanes (128-byte fold).
  pub const REDUCE_3_TO_1: [u64; 2] = [0x740eef02, 0x9e4addf8];
}

/// CRC32-IEEE (ISO-HDLC) folding constants.
/// Format: [k_lo, k_hi] for PMULL fold operations.
mod crc32_ieee_constants {
  /// 192-byte block folding (12 lanes).
  pub const FOLD_192: [u64; 2] = [0x596c8d81, 0xf5e48c85];
  /// 16-byte lane folding.
  pub const FOLD_16: [u64; 2] = [0xae689191, 0xccaa009e];
  /// Reduction: 12→6 lanes.
  pub const REDUCE_12_TO_6: [u64; 2] = [0xae689191, 0xccaa009e];
  /// Reduction: 6→3 lanes.
  pub const REDUCE_6_TO_3: [u64; 2] = [0xf1da05aa, 0x81256527];
  /// Reduction: 3→1 lanes.
  pub const REDUCE_3_TO_1: [u64; 2] = [0x8f352d95, 0x1d9513d7];
}

// ─────────────────────────────────────────────────────────────────────────────
// PMULL Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// CLMUL low lane and XOR: (a.lo * b.lo) ^ c
#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_lo_xor(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
  // SAFETY: target_feature ensures AES/PMULL is available
  let result = vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0));
  veorq_u64(vreinterpretq_u64_p128(result), c)
}

/// CLMUL high lane and XOR: (a.hi * b.hi) ^ c
#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_hi_xor(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
  // SAFETY: target_feature ensures AES/PMULL is available
  let result = vmull_p64(vgetq_lane_u64(a, 1), vgetq_lane_u64(b, 1));
  veorq_u64(vreinterpretq_u64_p128(result), c)
}

// ─────────────────────────────────────────────────────────────────────────────
// ARM CRC32 Extension (Pure Hardware)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC32C using ARM CRC32C hardware instruction.
///
/// # Safety
///
/// Caller must ensure CRC32 extension is available.
#[target_feature(enable = "crc")]
pub unsafe fn crc32c_arm(mut crc: u32, data: &[u8]) -> u32 {
  // SAFETY: All operations are guarded by bounds checks or target_feature
  unsafe {
    let mut ptr = data.as_ptr();
    let end = ptr.add(data.len());

    // Process 8 bytes at a time
    while ptr.add(8) <= end {
      crc = __crc32cd(crc, (ptr as *const u64).read_unaligned());
      ptr = ptr.add(8);
    }

    // Process 4 bytes
    if ptr.add(4) <= end {
      crc = __crc32cw(crc, (ptr as *const u32).read_unaligned());
      ptr = ptr.add(4);
    }

    // Process 2 bytes
    if ptr.add(2) <= end {
      crc = __crc32ch(crc, (ptr as *const u16).read_unaligned());
      ptr = ptr.add(2);
    }

    // Process 1 byte
    if ptr < end {
      crc = __crc32cb(crc, *ptr);
    }
  }

  crc
}

/// CRC32-IEEE using ARM CRC32 hardware instruction.
///
/// # Safety
///
/// Caller must ensure CRC32 extension is available.
#[target_feature(enable = "crc")]
pub unsafe fn crc32_ieee_arm(mut crc: u32, data: &[u8]) -> u32 {
  // SAFETY: All operations are guarded by bounds checks or target_feature
  unsafe {
    let mut ptr = data.as_ptr();
    let end = ptr.add(data.len());

    // Process 8 bytes at a time
    while ptr.add(8) <= end {
      crc = __crc32d(crc, (ptr as *const u64).read_unaligned());
      ptr = ptr.add(8);
    }

    // Process 4 bytes
    if ptr.add(4) <= end {
      crc = __crc32w(crc, (ptr as *const u32).read_unaligned());
      ptr = ptr.add(4);
    }

    // Process 2 bytes
    if ptr.add(2) <= end {
      crc = __crc32h(crc, (ptr as *const u16).read_unaligned());
      ptr = ptr.add(2);
    }

    // Process 1 byte
    if ptr < end {
      crc = __crc32b(crc, *ptr);
    }
  }

  crc
}

/// Safe wrapper for ARM CRC32C.
#[inline]
pub fn crc32c_arm_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC extension before selecting this kernel.
  unsafe { crc32c_arm(crc, data) }
}

/// Safe wrapper for ARM CRC32-IEEE.
#[inline]
pub fn crc32_ieee_arm_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC extension before selecting this kernel.
  unsafe { crc32_ieee_arm(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// PMULL + CRC Fusion (12-lane, 192-byte blocks)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC32C using PMULL folding + hardware CRC fusion.
///
/// Processes 192-byte blocks (12 × 16-byte lanes) with PMULL folding,
/// then uses hardware CRC32C instruction for final 128→32 reduction.
///
/// # Safety
///
/// Caller must ensure CRC and AES (PMULL) extensions are available.
#[target_feature(enable = "crc", enable = "aes")]
pub unsafe fn crc32c_pmull_fusion(mut crc: u32, data: &[u8]) -> u32 {
  // SAFETY: All operations are guarded by target_feature and bounds checks
  unsafe {
    let mut ptr = data.as_ptr();
    let mut len = data.len();

    // Align to 8-byte boundary using hardware CRC
    while len > 0 && (ptr as usize & 7) != 0 {
      crc = __crc32cb(crc, *ptr);
      ptr = ptr.add(1);
      len -= 1;
    }

    // Handle 8-byte alignment
    if (ptr as usize & 8) != 0 && len >= 8 {
      crc = __crc32cd(crc, *(ptr as *const u64));
      ptr = ptr.add(8);
      len -= 8;
    }

    // Process 192-byte blocks with PMULL
    if len >= 192 {
      let end = ptr.add(len);
      let limit = ptr.add(len - 192);

      // Load 12 lanes (192 bytes)
      let mut x0 = vld1q_u64(ptr as *const u64);
      let mut x1 = vld1q_u64(ptr.add(16) as *const u64);
      let mut x2 = vld1q_u64(ptr.add(32) as *const u64);
      let mut x3 = vld1q_u64(ptr.add(48) as *const u64);
      let mut x4 = vld1q_u64(ptr.add(64) as *const u64);
      let mut x5 = vld1q_u64(ptr.add(80) as *const u64);
      let mut x6 = vld1q_u64(ptr.add(96) as *const u64);
      let mut x7 = vld1q_u64(ptr.add(112) as *const u64);
      let mut x8 = vld1q_u64(ptr.add(128) as *const u64);
      let mut x9 = vld1q_u64(ptr.add(144) as *const u64);
      let mut x10 = vld1q_u64(ptr.add(160) as *const u64);
      let mut x11 = vld1q_u64(ptr.add(176) as *const u64);

      let k = vld1q_u64(crc32c_constants::FOLD_192.as_ptr());

      // XOR CRC into first lane
      let crc_vec = vsetq_lane_u64(crc as u64, vmovq_n_u64(0), 0);
      x0 = veorq_u64(crc_vec, x0);
      ptr = ptr.add(192);

      // Main folding loop
      while ptr <= limit {
        let y0 = clmul_lo_xor(x0, k, vld1q_u64(ptr as *const u64));
        x0 = clmul_hi_xor(x0, k, y0);
        let y1 = clmul_lo_xor(x1, k, vld1q_u64(ptr.add(16) as *const u64));
        x1 = clmul_hi_xor(x1, k, y1);
        let y2 = clmul_lo_xor(x2, k, vld1q_u64(ptr.add(32) as *const u64));
        x2 = clmul_hi_xor(x2, k, y2);
        let y3 = clmul_lo_xor(x3, k, vld1q_u64(ptr.add(48) as *const u64));
        x3 = clmul_hi_xor(x3, k, y3);
        let y4 = clmul_lo_xor(x4, k, vld1q_u64(ptr.add(64) as *const u64));
        x4 = clmul_hi_xor(x4, k, y4);
        let y5 = clmul_lo_xor(x5, k, vld1q_u64(ptr.add(80) as *const u64));
        x5 = clmul_hi_xor(x5, k, y5);
        let y6 = clmul_lo_xor(x6, k, vld1q_u64(ptr.add(96) as *const u64));
        x6 = clmul_hi_xor(x6, k, y6);
        let y7 = clmul_lo_xor(x7, k, vld1q_u64(ptr.add(112) as *const u64));
        x7 = clmul_hi_xor(x7, k, y7);
        let y8 = clmul_lo_xor(x8, k, vld1q_u64(ptr.add(128) as *const u64));
        x8 = clmul_hi_xor(x8, k, y8);
        let y9 = clmul_lo_xor(x9, k, vld1q_u64(ptr.add(144) as *const u64));
        x9 = clmul_hi_xor(x9, k, y9);
        let y10 = clmul_lo_xor(x10, k, vld1q_u64(ptr.add(160) as *const u64));
        x10 = clmul_hi_xor(x10, k, y10);
        let y11 = clmul_lo_xor(x11, k, vld1q_u64(ptr.add(176) as *const u64));
        x11 = clmul_hi_xor(x11, k, y11);
        ptr = ptr.add(192);
      }

      // Reduce 12 lanes → 6 lanes
      let k = vld1q_u64(crc32c_constants::REDUCE_12_TO_6.as_ptr());
      let y0 = clmul_lo_xor(x0, k, x1);
      x0 = clmul_hi_xor(x0, k, y0);
      let y2 = clmul_lo_xor(x2, k, x3);
      x2 = clmul_hi_xor(x2, k, y2);
      let y4 = clmul_lo_xor(x4, k, x5);
      x4 = clmul_hi_xor(x4, k, y4);
      let y6 = clmul_lo_xor(x6, k, x7);
      x6 = clmul_hi_xor(x6, k, y6);
      let y8 = clmul_lo_xor(x8, k, x9);
      x8 = clmul_hi_xor(x8, k, y8);
      let y10 = clmul_lo_xor(x10, k, x11);
      x10 = clmul_hi_xor(x10, k, y10);

      // Reduce 6 lanes → 3 lanes
      let k = vld1q_u64(crc32c_constants::REDUCE_6_TO_3.as_ptr());
      let y0 = clmul_lo_xor(x0, k, x2);
      x0 = clmul_hi_xor(x0, k, y0);
      let y4 = clmul_lo_xor(x4, k, x6);
      x4 = clmul_hi_xor(x4, k, y4);
      let y8 = clmul_lo_xor(x8, k, x10);
      x8 = clmul_hi_xor(x8, k, y8);

      // Reduce 3 lanes → 1 lane
      let k = vld1q_u64(crc32c_constants::REDUCE_3_TO_1.as_ptr());
      let y0 = clmul_lo_xor(x0, k, x4);
      x0 = clmul_hi_xor(x0, k, y0);
      x4 = x8;
      let y0 = clmul_lo_xor(x0, k, x4);
      x0 = clmul_hi_xor(x0, k, y0);

      // FUSION: Use hardware CRC32C for 128→32 reduction
      crc = __crc32cd(0, vgetq_lane_u64(x0, 0));
      crc = __crc32cd(crc, vgetq_lane_u64(x0, 1));
      len = end.offset_from(ptr) as usize;
    }

    // Process remaining 16-byte blocks
    if len >= 16 {
      let mut x0 = vld1q_u64(ptr as *const u64);
      let k = vld1q_u64(crc32c_constants::FOLD_16.as_ptr());

      let crc_vec = vsetq_lane_u64(crc as u64, vmovq_n_u64(0), 0);
      x0 = veorq_u64(crc_vec, x0);
      ptr = ptr.add(16);
      len -= 16;

      while len >= 16 {
        let y0 = clmul_lo_xor(x0, k, vld1q_u64(ptr as *const u64));
        x0 = clmul_hi_xor(x0, k, y0);
        ptr = ptr.add(16);
        len -= 16;
      }

      // FUSION: Hardware CRC32C for final reduction
      crc = __crc32cd(0, vgetq_lane_u64(x0, 0));
      crc = __crc32cd(crc, vgetq_lane_u64(x0, 1));
    }

    // Process remaining bytes with hardware CRC
    while len >= 8 {
      crc = __crc32cd(crc, *(ptr as *const u64));
      ptr = ptr.add(8);
      len -= 8;
    }

    while len > 0 {
      crc = __crc32cb(crc, *ptr);
      ptr = ptr.add(1);
      len -= 1;
    }
  }

  crc
}

/// CRC32-IEEE using PMULL folding + hardware CRC fusion.
///
/// # Safety
///
/// Caller must ensure CRC and AES (PMULL) extensions are available.
#[target_feature(enable = "crc", enable = "aes")]
pub unsafe fn crc32_ieee_pmull_fusion(mut crc: u32, data: &[u8]) -> u32 {
  // SAFETY: All operations are guarded by target_feature and bounds checks
  unsafe {
    let mut ptr = data.as_ptr();
    let mut len = data.len();

    // Align to 8-byte boundary using hardware CRC
    while len > 0 && (ptr as usize & 7) != 0 {
      crc = __crc32b(crc, *ptr);
      ptr = ptr.add(1);
      len -= 1;
    }

    // Handle 8-byte alignment
    if (ptr as usize & 8) != 0 && len >= 8 {
      crc = __crc32d(crc, *(ptr as *const u64));
      ptr = ptr.add(8);
      len -= 8;
    }

    // Process 192-byte blocks with PMULL
    if len >= 192 {
      let end = ptr.add(len);
      let limit = ptr.add(len - 192);

      // Load 12 lanes (192 bytes)
      let mut x0 = vld1q_u64(ptr as *const u64);
      let mut x1 = vld1q_u64(ptr.add(16) as *const u64);
      let mut x2 = vld1q_u64(ptr.add(32) as *const u64);
      let mut x3 = vld1q_u64(ptr.add(48) as *const u64);
      let mut x4 = vld1q_u64(ptr.add(64) as *const u64);
      let mut x5 = vld1q_u64(ptr.add(80) as *const u64);
      let mut x6 = vld1q_u64(ptr.add(96) as *const u64);
      let mut x7 = vld1q_u64(ptr.add(112) as *const u64);
      let mut x8 = vld1q_u64(ptr.add(128) as *const u64);
      let mut x9 = vld1q_u64(ptr.add(144) as *const u64);
      let mut x10 = vld1q_u64(ptr.add(160) as *const u64);
      let mut x11 = vld1q_u64(ptr.add(176) as *const u64);

      let k = vld1q_u64(crc32_ieee_constants::FOLD_192.as_ptr());

      // XOR CRC into first lane
      let crc_vec = vsetq_lane_u64(crc as u64, vmovq_n_u64(0), 0);
      x0 = veorq_u64(crc_vec, x0);
      ptr = ptr.add(192);

      // Main folding loop
      while ptr <= limit {
        let y0 = clmul_lo_xor(x0, k, vld1q_u64(ptr as *const u64));
        x0 = clmul_hi_xor(x0, k, y0);
        let y1 = clmul_lo_xor(x1, k, vld1q_u64(ptr.add(16) as *const u64));
        x1 = clmul_hi_xor(x1, k, y1);
        let y2 = clmul_lo_xor(x2, k, vld1q_u64(ptr.add(32) as *const u64));
        x2 = clmul_hi_xor(x2, k, y2);
        let y3 = clmul_lo_xor(x3, k, vld1q_u64(ptr.add(48) as *const u64));
        x3 = clmul_hi_xor(x3, k, y3);
        let y4 = clmul_lo_xor(x4, k, vld1q_u64(ptr.add(64) as *const u64));
        x4 = clmul_hi_xor(x4, k, y4);
        let y5 = clmul_lo_xor(x5, k, vld1q_u64(ptr.add(80) as *const u64));
        x5 = clmul_hi_xor(x5, k, y5);
        let y6 = clmul_lo_xor(x6, k, vld1q_u64(ptr.add(96) as *const u64));
        x6 = clmul_hi_xor(x6, k, y6);
        let y7 = clmul_lo_xor(x7, k, vld1q_u64(ptr.add(112) as *const u64));
        x7 = clmul_hi_xor(x7, k, y7);
        let y8 = clmul_lo_xor(x8, k, vld1q_u64(ptr.add(128) as *const u64));
        x8 = clmul_hi_xor(x8, k, y8);
        let y9 = clmul_lo_xor(x9, k, vld1q_u64(ptr.add(144) as *const u64));
        x9 = clmul_hi_xor(x9, k, y9);
        let y10 = clmul_lo_xor(x10, k, vld1q_u64(ptr.add(160) as *const u64));
        x10 = clmul_hi_xor(x10, k, y10);
        let y11 = clmul_lo_xor(x11, k, vld1q_u64(ptr.add(176) as *const u64));
        x11 = clmul_hi_xor(x11, k, y11);
        ptr = ptr.add(192);
      }

      // Reduce 12 lanes → 6 lanes
      let k = vld1q_u64(crc32_ieee_constants::REDUCE_12_TO_6.as_ptr());
      let y0 = clmul_lo_xor(x0, k, x1);
      x0 = clmul_hi_xor(x0, k, y0);
      let y2 = clmul_lo_xor(x2, k, x3);
      x2 = clmul_hi_xor(x2, k, y2);
      let y4 = clmul_lo_xor(x4, k, x5);
      x4 = clmul_hi_xor(x4, k, y4);
      let y6 = clmul_lo_xor(x6, k, x7);
      x6 = clmul_hi_xor(x6, k, y6);
      let y8 = clmul_lo_xor(x8, k, x9);
      x8 = clmul_hi_xor(x8, k, y8);
      let y10 = clmul_lo_xor(x10, k, x11);
      x10 = clmul_hi_xor(x10, k, y10);

      // Reduce 6 lanes → 3 lanes
      let k = vld1q_u64(crc32_ieee_constants::REDUCE_6_TO_3.as_ptr());
      let y0 = clmul_lo_xor(x0, k, x2);
      x0 = clmul_hi_xor(x0, k, y0);
      let y4 = clmul_lo_xor(x4, k, x6);
      x4 = clmul_hi_xor(x4, k, y4);
      let y8 = clmul_lo_xor(x8, k, x10);
      x8 = clmul_hi_xor(x8, k, y8);

      // Reduce 3 lanes → 1 lane
      let k = vld1q_u64(crc32_ieee_constants::REDUCE_3_TO_1.as_ptr());
      let y0 = clmul_lo_xor(x0, k, x4);
      x0 = clmul_hi_xor(x0, k, y0);
      x4 = x8;
      let y0 = clmul_lo_xor(x0, k, x4);
      x0 = clmul_hi_xor(x0, k, y0);

      // FUSION: Use hardware CRC32 for 128→32 reduction
      crc = __crc32d(0, vgetq_lane_u64(x0, 0));
      crc = __crc32d(crc, vgetq_lane_u64(x0, 1));
      len = end.offset_from(ptr) as usize;
    }

    // Process remaining 16-byte blocks
    if len >= 16 {
      let mut x0 = vld1q_u64(ptr as *const u64);
      let k = vld1q_u64(crc32_ieee_constants::FOLD_16.as_ptr());

      let crc_vec = vsetq_lane_u64(crc as u64, vmovq_n_u64(0), 0);
      x0 = veorq_u64(crc_vec, x0);
      ptr = ptr.add(16);
      len -= 16;

      while len >= 16 {
        let y0 = clmul_lo_xor(x0, k, vld1q_u64(ptr as *const u64));
        x0 = clmul_hi_xor(x0, k, y0);
        ptr = ptr.add(16);
        len -= 16;
      }

      // FUSION: Hardware CRC32 for final reduction
      crc = __crc32d(0, vgetq_lane_u64(x0, 0));
      crc = __crc32d(crc, vgetq_lane_u64(x0, 1));
    }

    // Process remaining bytes with hardware CRC
    while len >= 8 {
      crc = __crc32d(crc, *(ptr as *const u64));
      ptr = ptr.add(8);
      len -= 8;
    }

    while len > 0 {
      crc = __crc32b(crc, *ptr);
      ptr = ptr.add(1);
      len -= 1;
    }
  }

  crc
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe Wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Safe wrapper for CRC32C PMULL fusion.
#[inline]
pub fn crc32c_pmull_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC+AES before selecting this kernel.
  unsafe { crc32c_pmull_fusion(crc, data) }
}

/// Safe wrapper for CRC32-IEEE PMULL fusion.
#[inline]
pub fn crc32_ieee_pmull_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies CRC+AES before selecting this kernel.
  unsafe { crc32_ieee_pmull_fusion(crc, data) }
}

// Aliases for dispatcher compatibility (reserved for future stream-based dispatch)
#[allow(dead_code)]
pub fn crc32c_pmull_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_pmull_safe(crc, data)
}

#[allow(dead_code)]
pub fn crc32c_pmull_3way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_pmull_safe(crc, data)
}

#[allow(dead_code)]
pub fn crc32c_pmull_small_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_arm_safe(crc, data)
}

pub fn crc32c_pmull_eor3_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_pmull_safe(crc, data)
}

#[allow(dead_code)]
pub fn crc32c_pmull_eor3_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_pmull_safe(crc, data)
}

#[allow(dead_code)]
pub fn crc32c_pmull_eor3_3way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32c_pmull_safe(crc, data)
}

#[allow(dead_code)]
pub fn crc32_ieee_pmull_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_ieee_pmull_safe(crc, data)
}

#[allow(dead_code)]
pub fn crc32_ieee_pmull_3way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_ieee_pmull_safe(crc, data)
}

#[allow(dead_code)]
pub fn crc32_ieee_pmull_small_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_ieee_arm_safe(crc, data)
}

pub fn crc32_ieee_pmull_eor3_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_ieee_pmull_safe(crc, data)
}

#[allow(dead_code)]
pub fn crc32_ieee_pmull_eor3_2way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_ieee_pmull_safe(crc, data)
}

#[allow(dead_code)]
pub fn crc32_ieee_pmull_eor3_3way_safe(crc: u32, data: &[u8]) -> u32 {
  crc32_ieee_pmull_safe(crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  extern crate alloc;
  use alloc::vec::Vec;

  use super::*;
  use crate::crc32::portable::{crc32_ieee_slice16, crc32c_slice16};

  // Check values from crc-fast-rust
  const CRC32C_CHECK: u32 = 0xE3069283; // "123456789"
  const CRC32_IEEE_CHECK: u32 = 0xCBF43926; // "123456789"

  const CRC32C_HELLO_WORLD: u32 = 0xC99465AA;
  const CRC32_IEEE_HELLO_WORLD: u32 = 0x0D4A_1185;

  #[test]
  fn test_crc32c_arm_check() {
    let result = crc32c_arm_safe(!0, b"123456789") ^ !0;
    assert_eq!(result, CRC32C_CHECK);
  }

  #[test]
  fn test_crc32_ieee_arm_check() {
    let result = crc32_ieee_arm_safe(!0, b"123456789") ^ !0;
    assert_eq!(result, CRC32_IEEE_CHECK);
  }

  #[test]
  fn test_crc32c_arm_hello() {
    let result = crc32c_arm_safe(!0, b"hello world") ^ !0;
    assert_eq!(result, CRC32C_HELLO_WORLD);
  }

  #[test]
  fn test_crc32_ieee_arm_hello() {
    let result = crc32_ieee_arm_safe(!0, b"hello world") ^ !0;
    assert_eq!(result, CRC32_IEEE_HELLO_WORLD);
  }

  #[test]
  fn test_crc32c_pmull_vs_arm() {
    for len in [16, 32, 64, 128, 192, 256, 384, 512, 1024, 4096] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let arm = crc32c_arm_safe(!0, &data) ^ !0;
      let pmull = crc32c_pmull_safe(!0, &data) ^ !0;

      assert_eq!(arm, pmull, "CRC32C mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32_ieee_pmull_vs_arm() {
    for len in [16, 32, 64, 128, 192, 256, 384, 512, 1024, 4096] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let arm = crc32_ieee_arm_safe(!0, &data) ^ !0;
      let pmull = crc32_ieee_pmull_safe(!0, &data) ^ !0;

      assert_eq!(arm, pmull, "CRC32-IEEE mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32c_pmull_vs_portable() {
    for len in [16, 32, 64, 128, 192, 256, 384, 512, 1024, 4096] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let portable = crc32c_slice16(!0, &data) ^ !0;
      let pmull = crc32c_pmull_safe(!0, &data) ^ !0;

      assert_eq!(portable, pmull, "CRC32C mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32_ieee_pmull_vs_portable() {
    for len in [16, 32, 64, 128, 192, 256, 384, 512, 1024, 4096] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      let portable = crc32_ieee_slice16(!0, &data) ^ !0;
      let pmull = crc32_ieee_pmull_safe(!0, &data) ^ !0;

      assert_eq!(portable, pmull, "CRC32-IEEE mismatch at length {len}");
    }
  }

  #[test]
  fn test_crc32c_pmull_check() {
    let result = crc32c_pmull_safe(!0, b"123456789") ^ !0;
    // For small inputs, falls through to hardware CRC
    assert_eq!(result, CRC32C_CHECK);
  }

  #[test]
  fn test_crc32_ieee_pmull_check() {
    let result = crc32_ieee_pmull_safe(!0, b"123456789") ^ !0;
    // For small inputs, falls through to hardware CRC
    assert_eq!(result, CRC32_IEEE_CHECK);
  }

  #[test]
  fn test_crc32c_pmull_large() {
    // Test with 1 MiB of data
    let data: Vec<u8> = (0..1048576u32).map(|i| i as u8).collect();

    let arm = crc32c_arm_safe(!0, &data) ^ !0;
    let pmull = crc32c_pmull_safe(!0, &data) ^ !0;

    assert_eq!(arm, pmull, "CRC32C mismatch on 1 MiB");
  }

  #[test]
  fn test_crc32_ieee_pmull_large() {
    // Test with 1 MiB of data
    let data: Vec<u8> = (0..1048576u32).map(|i| i as u8).collect();

    let arm = crc32_ieee_arm_safe(!0, &data) ^ !0;
    let pmull = crc32_ieee_pmull_safe(!0, &data) ^ !0;

    assert_eq!(arm, pmull, "CRC32-IEEE mismatch on 1 MiB");
  }
}
