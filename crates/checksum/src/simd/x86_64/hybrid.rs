//! AMD Zen hybrid scalar+SIMD CRC32-C kernel.
//!
//! This module implements a hybrid approach that runs parallel `crc32q` scalar
//! streams alongside VPCLMULQDQ vector operations, then combines the results.
//!
//! # Why Hybrid is Faster on AMD Zen
//!
//! AMD Zen 4/5 has:
//! - Very low ZMM warmup latency (~60ns vs Intel's ~2000ns)
//! - Slower VPCLMULQDQ throughput (0.5/cycle vs Intel's 1/cycle)
//! - But excellent `crc32q` parallelism (3-way on Zen4, 7-way on Zen5)
//!
//! By running `crc32q` and VPCLMULQDQ on different execution ports in parallel,
//! we can achieve higher throughput than either approach alone.
//!
//! # Algorithm
//!
//! 1. Split buffer into N+1 regions (N scalar streams + 1 vector region)
//! 2. Process scalar regions with parallel `crc32q` instructions
//! 3. Process vector region with VPCLMULQDQ
//! 4. Combine all CRCs using `crc32c_combine()` (O(log n) per merge)
//!
//! # References
//!
//! - Eric Biggers' Linux kernel patches (2025)
//! - <https://www.phoronix.com/news/Linux-CRC32C-VPCLMULQDQ>

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

use crate::combine::crc32c_combine;

/// Minimum buffer size for hybrid to be beneficial.
/// Below this, pure VPCLMULQDQ or scalar is faster.
const HYBRID_MIN_SIZE: usize = 512;

/// Block size for each scalar stream per iteration (8 bytes = one crc32q).
const SCALAR_BLOCK: usize = 8;

// ============================================================================
// Multi-Stream Parallel crc32q Kernels
// ============================================================================

/// Process 3 parallel `crc32q` streams (optimal for Zen 4).
///
/// Each stream processes its portion of the buffer independently.
/// The streams can execute in parallel on separate execution units.
///
/// # Safety
/// Caller must ensure CPU supports SSE4.2.
#[target_feature(enable = "sse4.2")]
unsafe fn crc32q_3way(
  crc0: u32,
  crc1: u32,
  crc2: u32,
  ptr0: *const u8,
  ptr1: *const u8,
  ptr2: *const u8,
  iterations: usize,
) -> (u32, u32, u32) {
  let mut c0 = crc0;
  let mut c1 = crc1;
  let mut c2 = crc2;

  let mut p0 = ptr0;
  let mut p1 = ptr1;
  let mut p2 = ptr2;

  // Main loop: process 8 bytes per stream per iteration.
  // All 3 crc32q instructions can execute in parallel on Zen 4.
  let mut i = 0;
  while i < iterations {
    #[allow(clippy::cast_ptr_alignment)]
    let v0 = core::ptr::read_unaligned(p0 as *const u64);
    #[allow(clippy::cast_ptr_alignment)]
    let v1 = core::ptr::read_unaligned(p1 as *const u64);
    #[allow(clippy::cast_ptr_alignment)]
    let v2 = core::ptr::read_unaligned(p2 as *const u64);

    c0 = _mm_crc32_u64(c0 as u64, v0) as u32;
    c1 = _mm_crc32_u64(c1 as u64, v1) as u32;
    c2 = _mm_crc32_u64(c2 as u64, v2) as u32;

    p0 = p0.add(8);
    p1 = p1.add(8);
    p2 = p2.add(8);
    i += 1;
  }

  (c0, c1, c2)
}

/// Process 7 parallel `crc32q` streams (optimal for Zen 5).
///
/// Zen 5 can sustain 7 parallel `crc32q` instructions per cycle,
/// a significant upgrade from Zen 4's 3-way parallelism.
///
/// # Safety
/// Caller must ensure CPU supports SSE4.2.
#[target_feature(enable = "sse4.2")]
unsafe fn crc32q_7way(crcs: [u32; 7], ptrs: [*const u8; 7], iterations: usize) -> [u32; 7] {
  let mut c = crcs;
  let mut p = ptrs;

  // Main loop: process 8 bytes per stream per iteration.
  // All 7 crc32q instructions can execute in parallel on Zen 5.
  let mut i = 0;
  while i < iterations {
    // Load all 7 values first to maximize memory-level parallelism.
    #[allow(clippy::cast_ptr_alignment)]
    let v0 = core::ptr::read_unaligned(p[0] as *const u64);
    #[allow(clippy::cast_ptr_alignment)]
    let v1 = core::ptr::read_unaligned(p[1] as *const u64);
    #[allow(clippy::cast_ptr_alignment)]
    let v2 = core::ptr::read_unaligned(p[2] as *const u64);
    #[allow(clippy::cast_ptr_alignment)]
    let v3 = core::ptr::read_unaligned(p[3] as *const u64);
    #[allow(clippy::cast_ptr_alignment)]
    let v4 = core::ptr::read_unaligned(p[4] as *const u64);
    #[allow(clippy::cast_ptr_alignment)]
    let v5 = core::ptr::read_unaligned(p[5] as *const u64);
    #[allow(clippy::cast_ptr_alignment)]
    let v6 = core::ptr::read_unaligned(p[6] as *const u64);

    // Execute all 7 crc32q in parallel.
    c[0] = _mm_crc32_u64(c[0] as u64, v0) as u32;
    c[1] = _mm_crc32_u64(c[1] as u64, v1) as u32;
    c[2] = _mm_crc32_u64(c[2] as u64, v2) as u32;
    c[3] = _mm_crc32_u64(c[3] as u64, v3) as u32;
    c[4] = _mm_crc32_u64(c[4] as u64, v4) as u32;
    c[5] = _mm_crc32_u64(c[5] as u64, v5) as u32;
    c[6] = _mm_crc32_u64(c[6] as u64, v6) as u32;

    p[0] = p[0].add(8);
    p[1] = p[1].add(8);
    p[2] = p[2].add(8);
    p[3] = p[3].add(8);
    p[4] = p[4].add(8);
    p[5] = p[5].add(8);
    p[6] = p[6].add(8);
    i += 1;
  }

  c
}

/// Process remaining bytes with single-stream crc32q.
#[target_feature(enable = "sse4.2")]
unsafe fn crc32q_remainder(mut crc: u32, mut ptr: *const u8, mut len: usize) -> u32 {
  // Process 8 bytes at a time.
  while len >= 8 {
    #[allow(clippy::cast_ptr_alignment)]
    let v = core::ptr::read_unaligned(ptr as *const u64);
    crc = _mm_crc32_u64(crc as u64, v) as u32;
    ptr = ptr.add(8);
    len -= 8;
  }

  // Process 4 bytes.
  if len >= 4 {
    #[allow(clippy::cast_ptr_alignment)]
    let v = core::ptr::read_unaligned(ptr as *const u32);
    crc = _mm_crc32_u32(crc, v);
    ptr = ptr.add(4);
    len -= 4;
  }

  // Process 2 bytes.
  if len >= 2 {
    #[allow(clippy::cast_ptr_alignment)]
    let v = core::ptr::read_unaligned(ptr as *const u16);
    crc = _mm_crc32_u16(crc, v);
    ptr = ptr.add(2);
    len -= 2;
  }

  // Process 1 byte.
  if len != 0 {
    crc = _mm_crc32_u8(crc, *ptr);
  }

  crc
}

// ============================================================================
// Hybrid Kernels
// ============================================================================

/// Compute CRC32-C using Zen 4 hybrid (3-way crc32q + VPCLMULQDQ).
///
/// # Safety
/// Caller must ensure CPU supports SSE4.2 and AVX-512 with VPCLMULQDQ.
#[target_feature(enable = "sse4.2,avx512f,avx512vl,avx512bw,vpclmulqdq,pclmulqdq")]
pub unsafe fn compute_hybrid_zen4_unchecked(crc: u32, data: &[u8]) -> u32 {
  // Fall back to pure VPCLMULQDQ for small buffers.
  if data.len() < HYBRID_MIN_SIZE {
    return super::vpclmul::compute_vpclmul_runtime(crc, data);
  }

  // Strategy: Split buffer into 4 equal regions.
  // - Regions 0-2: processed by 3-way parallel crc32q
  // - Region 3: processed by VPCLMULQDQ
  //
  // We need to account for:
  // - VPCLMULQDQ needs 64-byte aligned chunks
  // - crc32q processes 8 bytes at a time

  let len = data.len();
  let base = data.as_ptr();

  // Calculate region sizes. Each region should be multiple of 64 for alignment.
  let region_size = (len / 4) & !63;
  if region_size < 64 {
    // Buffer too small for effective hybrid, fall back.
    return super::vpclmul::compute_vpclmul_runtime(crc, data);
  }

  // Pointers to each region.
  let ptr0 = base;
  let ptr1 = base.add(region_size);
  let ptr2 = base.add(region_size * 2);
  let ptr3 = base.add(region_size * 3);

  // Lengths of each region.
  let len0 = region_size;
  let len1 = region_size;
  let len2 = region_size;
  let len3 = len - (region_size * 3); // Remainder goes to VPCLMULQDQ

  // Number of 8-byte iterations for scalar streams.
  let scalar_iters = region_size / SCALAR_BLOCK;

  // Process regions 0-2 with 3-way parallel crc32q.
  // Initial CRC goes to stream 0, others start at 0.
  let (crc0, crc1, crc2) = crc32q_3way(crc, 0, 0, ptr0, ptr1, ptr2, scalar_iters);

  // Handle remainder bytes in each scalar stream.
  let rem0 = len0 - (scalar_iters * SCALAR_BLOCK);
  let rem1 = len1 - (scalar_iters * SCALAR_BLOCK);
  let rem2 = len2 - (scalar_iters * SCALAR_BLOCK);

  let crc0 = crc32q_remainder(crc0, ptr0.add(scalar_iters * SCALAR_BLOCK), rem0);
  let crc1 = crc32q_remainder(crc1, ptr1.add(scalar_iters * SCALAR_BLOCK), rem1);
  let crc2 = crc32q_remainder(crc2, ptr2.add(scalar_iters * SCALAR_BLOCK), rem2);

  // Process region 3 with VPCLMULQDQ.
  let region3 = core::slice::from_raw_parts(ptr3, len3);
  let crc3 = super::vpclmul::compute_vpclmul_runtime(0, region3);

  // Combine all CRCs: crc(0||1||2||3) = combine(combine(combine(crc0, crc1), crc2), crc3)
  let combined = crc32c_combine(crc0, crc1, len1);
  let combined = crc32c_combine(combined, crc2, len2);
  crc32c_combine(combined, crc3, len3)
}

/// Compute CRC32-C using Zen 5 hybrid (7-way crc32q + VPCLMULQDQ).
///
/// # Safety
/// Caller must ensure CPU supports SSE4.2 and AVX-512 with VPCLMULQDQ.
#[target_feature(enable = "sse4.2,avx512f,avx512vl,avx512bw,vpclmulqdq,pclmulqdq")]
pub unsafe fn compute_hybrid_zen5_unchecked(crc: u32, data: &[u8]) -> u32 {
  // Fall back to pure VPCLMULQDQ for small buffers.
  if data.len() < HYBRID_MIN_SIZE {
    return super::vpclmul::compute_vpclmul_runtime(crc, data);
  }

  // Strategy: Split buffer into 8 equal regions.
  // - Regions 0-6: processed by 7-way parallel crc32q
  // - Region 7: processed by VPCLMULQDQ
  //
  // Zen 5 can sustain 7 parallel crc32q instructions, making this
  // configuration optimal for maximizing throughput.

  let len = data.len();
  let base = data.as_ptr();

  // Calculate region sizes. Each region should be multiple of 64 for alignment.
  let region_size = (len / 8) & !63;
  if region_size < 64 {
    // Buffer too small for effective hybrid, fall back.
    return super::vpclmul::compute_vpclmul_runtime(crc, data);
  }

  // Pointers to each region.
  let ptrs = [
    base,
    base.add(region_size),
    base.add(region_size * 2),
    base.add(region_size * 3),
    base.add(region_size * 4),
    base.add(region_size * 5),
    base.add(region_size * 6),
  ];
  let ptr7 = base.add(region_size * 7);

  // Lengths of each region.
  let scalar_len = region_size; // Same for regions 0-6
  let len7 = len - (region_size * 7); // Remainder goes to VPCLMULQDQ

  // Number of 8-byte iterations for scalar streams.
  let scalar_iters = region_size / SCALAR_BLOCK;

  // Process regions 0-6 with 7-way parallel crc32q.
  // Initial CRC goes to stream 0, others start at 0.
  let crcs = crc32q_7way([crc, 0, 0, 0, 0, 0, 0], ptrs, scalar_iters);

  // Handle remainder bytes in each scalar stream.
  let rem = scalar_len - (scalar_iters * SCALAR_BLOCK);
  let mut final_crcs = [0u32; 7];
  for i in 0..7 {
    final_crcs[i] = crc32q_remainder(crcs[i], ptrs[i].add(scalar_iters * SCALAR_BLOCK), rem);
  }

  // Process region 7 with VPCLMULQDQ.
  let region7 = core::slice::from_raw_parts(ptr7, len7);
  let crc7 = super::vpclmul::compute_vpclmul_runtime(0, region7);

  // Combine all CRCs: crc(0||1||2||3||4||5||6||7)
  let mut combined = final_crcs[0];
  for &crc in &final_crcs[1..] {
    combined = crc32c_combine(combined, crc, scalar_len);
  }
  crc32c_combine(combined, crc7, len7)
}

// ============================================================================
// Runtime-Dispatched Entry Points
// ============================================================================

/// Compute CRC32-C using Zen 4 hybrid kernel.
///
/// Selected at runtime when `MicroArch::Zen4` is detected.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn compute_hybrid_zen4_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: This function is only called when the required features are detected.
  unsafe { compute_hybrid_zen4_unchecked(crc, data) }
}

/// Compute CRC32-C using Zen 5 hybrid kernel.
///
/// Selected at runtime when `MicroArch::Zen5` is detected.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn compute_hybrid_zen5_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: This function is only called when the required features are detected.
  unsafe { compute_hybrid_zen5_unchecked(crc, data) }
}

#[cfg(test)]
mod tests {
  extern crate std;

  #[allow(unused_imports)] // Used only when target_feature = "avx512f" etc.
  use super::*;

  #[test]
  #[cfg(all(target_feature = "sse4.2", target_feature = "avx512f", target_feature = "vpclmulqdq"))]
  fn test_hybrid_matches_portable() {
    let data = [0xABu8; 4096];
    let expected = crate::crc32c::portable::compute(0xFFFF_FFFF, &data) ^ 0xFFFF_FFFF;

    // Test Zen 4 hybrid.
    let zen4_result = unsafe { compute_hybrid_zen4_unchecked(0xFFFF_FFFF, &data) } ^ 0xFFFF_FFFF;
    assert_eq!(zen4_result, expected, "Zen 4 hybrid mismatch");

    // Test Zen 5 hybrid.
    let zen5_result = unsafe { compute_hybrid_zen5_unchecked(0xFFFF_FFFF, &data) } ^ 0xFFFF_FFFF;
    assert_eq!(zen5_result, expected, "Zen 5 hybrid mismatch");
  }

  #[test]
  #[cfg(all(target_feature = "sse4.2", target_feature = "avx512f", target_feature = "vpclmulqdq"))]
  fn test_hybrid_various_sizes() {
    for size in [512, 1024, 4096, 8192, 16384, 65536] {
      let data = std::vec![0xCDu8; size];
      let expected = crate::crc32c::portable::compute(0xFFFF_FFFF, &data) ^ 0xFFFF_FFFF;

      let zen4_result = unsafe { compute_hybrid_zen4_unchecked(0xFFFF_FFFF, &data) } ^ 0xFFFF_FFFF;
      assert_eq!(zen4_result, expected, "Zen 4 hybrid mismatch at size {}", size);

      let zen5_result = unsafe { compute_hybrid_zen5_unchecked(0xFFFF_FFFF, &data) } ^ 0xFFFF_FFFF;
      assert_eq!(zen5_result, expected, "Zen 5 hybrid mismatch at size {}", size);
    }
  }

  #[test]
  #[cfg(all(target_feature = "sse4.2", target_feature = "avx512f", target_feature = "vpclmulqdq"))]
  fn test_hybrid_small_buffer_fallback() {
    // Buffers below HYBRID_MIN_SIZE should fall back to pure VPCLMULQDQ.
    let data = [0xEFu8; 256];
    let expected = crate::crc32c::portable::compute(0xFFFF_FFFF, &data) ^ 0xFFFF_FFFF;

    let zen4_result = unsafe { compute_hybrid_zen4_unchecked(0xFFFF_FFFF, &data) } ^ 0xFFFF_FFFF;
    assert_eq!(zen4_result, expected);

    let zen5_result = unsafe { compute_hybrid_zen5_unchecked(0xFFFF_FFFF, &data) } ^ 0xFFFF_FFFF;
    assert_eq!(zen5_result, expected);
  }
}
