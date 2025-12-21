//! aarch64 hardware-accelerated CRC-64 kernels (XZ + NVME).
//!
//! This is a PMULL implementation derived from the Intel/TiKV folding
//! algorithm (also used by `crc64fast` / `crc64fast-nvme`).
//!
//! # Safety
//!
//! Uses `unsafe` for ARM SIMD intrinsics. Callers must ensure PMULL is
//! available before executing the accelerated path (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(dead_code)] // Kernels wired up via dispatcher
// SAFETY: All indexing is over fixed-size arrays with in-bounds constant indices.
#![allow(clippy::indexing_slicing)]
// This module is intrinsics-heavy; keep unsafe blocks readable.
#![allow(unsafe_op_in_unsafe_fn)]

use core::{
  arch::aarch64::*,
  ops::{BitXor, BitXorAssign},
};

use crate::common::{
  clmul::{Crc64ClmulConstants, fold16_coeff_for_bytes},
  tables::{CRC64_NVME_POLY, CRC64_XZ_POLY},
};

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
struct Simd(uint8x16_t);

#[allow(non_camel_case_types)]
type poly64_t = u64;

impl Simd {
  #[inline]
  #[target_feature(enable = "neon", enable = "aes")]
  unsafe fn from_mul(a: poly64_t, b: poly64_t) -> Self {
    let mul = vmull_p64(a, b);
    Self(vreinterpretq_u8_p128(mul))
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn into_poly64s(self) -> [poly64_t; 2] {
    let x = vreinterpretq_p64_u8(self.0);
    [vgetq_lane_p64(x, 0), vgetq_lane_p64(x, 1)]
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn high_64(self) -> poly64_t {
    let x = vreinterpretq_p64_u8(self.0);
    vgetq_lane_p64(x, 1)
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn low_64(self) -> poly64_t {
    let x = vreinterpretq_p64_u8(self.0);
    vgetq_lane_p64(x, 0)
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn new(high: u64, low: u64) -> Self {
    Self(vcombine_u8(vcreate_u8(low), vcreate_u8(high)))
  }

  /// Fold 16 bytes: `(coeff.low ⊗ self.low) ⊕ (coeff.high ⊗ self.high)`.
  #[inline]
  #[target_feature(enable = "neon", enable = "aes")]
  unsafe fn fold_16(self, coeff: Self) -> Self {
    let [x0, x1] = self.into_poly64s();
    let [c0, c1] = coeff.into_poly64s();
    let h = Self::from_mul(c0, x0);
    let l = Self::from_mul(c1, x1);
    h ^ l
  }

  /// Fold 16 bytes using pre-extracted coefficient halves (low, high).
  ///
  /// Equivalent to `self.fold_16(Simd::new(high, low))` but avoids repeatedly
  /// extracting the coefficient lanes inside hot loops.
  #[inline]
  #[target_feature(enable = "neon", enable = "aes")]
  unsafe fn fold_16_pair(self, coeff_low: poly64_t, coeff_high: poly64_t) -> Self {
    let [x0, x1] = self.into_poly64s();
    let h = Self::from_mul(coeff_low, x0);
    let l = Self::from_mul(coeff_high, x1);
    h ^ l
  }

  /// Fold 8 bytes: `self.high ⊕ (coeff ⊗ self.low)`.
  #[inline]
  #[target_feature(enable = "neon", enable = "aes")]
  unsafe fn fold_8(self, coeff: u64) -> Self {
    let [x0, x1] = self.into_poly64s();
    let h = Self::from_mul(coeff, x0);
    let l = Self::new(0, x1);
    h ^ l
  }

  /// Barrett reduction to finalize the CRC.
  #[inline]
  #[target_feature(enable = "neon", enable = "aes")]
  unsafe fn barrett(self, poly: u64, mu: u64) -> u64 {
    let t1 = Self::from_mul(self.low_64(), mu).low_64();
    let l = Self::from_mul(t1, poly);
    let reduced = (self ^ l).high_64();
    reduced ^ t1
  }
}

impl BitXor for Simd {
  type Output = Self;

  #[inline]
  fn bitxor(self, other: Self) -> Self {
    // SAFETY: `veorq_u8` is available with NEON.
    unsafe { Self(veorq_u8(self.0, other.0)) }
  }
}

impl BitXorAssign for Simd {
  #[inline]
  fn bitxor_assign(&mut self, other: Self) {
    *self = *self ^ other;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-stream coefficients (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

// 2-way: update step shifts by 2×128B = 256B.
const XZ_FOLD_256B: (u64, u64) = fold16_coeff_for_bytes(CRC64_XZ_POLY, 256);
const NVME_FOLD_256B: (u64, u64) = fold16_coeff_for_bytes(CRC64_NVME_POLY, 256);

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd(state: u64, first: &[Simd; 8], rest: &[[Simd; 8]], consts: &Crc64ClmulConstants) -> u64 {
  let mut x = *first;

  // XOR the initial CRC into the first lane.
  x[0] ^= Simd::new(0, state);

  // 128-byte folding.
  let coeff_low = consts.fold_128b.1;
  let coeff_high = consts.fold_128b.0;
  for chunk in rest {
    fold_block_128(&mut x, chunk, coeff_low, coeff_high);
  }

  fold_tail(x, consts)
}

#[inline(always)]
unsafe fn fold_tail(x: [Simd; 8], consts: &Crc64ClmulConstants) -> u64 {
  // Tail reduction (8×16B → 1×16B), unrolled for throughput.
  let c0 = Simd::new(consts.tail_fold_16b[0].0, consts.tail_fold_16b[0].1);
  let c1 = Simd::new(consts.tail_fold_16b[1].0, consts.tail_fold_16b[1].1);
  let c2 = Simd::new(consts.tail_fold_16b[2].0, consts.tail_fold_16b[2].1);
  let c3 = Simd::new(consts.tail_fold_16b[3].0, consts.tail_fold_16b[3].1);
  let c4 = Simd::new(consts.tail_fold_16b[4].0, consts.tail_fold_16b[4].1);
  let c5 = Simd::new(consts.tail_fold_16b[5].0, consts.tail_fold_16b[5].1);
  let c6 = Simd::new(consts.tail_fold_16b[6].0, consts.tail_fold_16b[6].1);

  let mut acc = x[7];
  acc ^= x[0].fold_16(c0);
  acc ^= x[1].fold_16(c1);
  acc ^= x[2].fold_16(c2);
  acc ^= x[3].fold_16(c3);
  acc ^= x[4].fold_16(c4);
  acc ^= x[5].fold_16(c5);
  acc ^= x[6].fold_16(c6);

  acc.fold_8(consts.fold_8b).barrett(consts.poly, consts.mu)
}

// ─────────────────────────────────────────────────────────────────────────────
// PMULL multi-stream (2-way, 128B blocks)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn fold_block_128(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff_low: u64, coeff_high: u64) {
  x[0] = chunk[0] ^ x[0].fold_16_pair(coeff_low, coeff_high);
  x[1] = chunk[1] ^ x[1].fold_16_pair(coeff_low, coeff_high);
  x[2] = chunk[2] ^ x[2].fold_16_pair(coeff_low, coeff_high);
  x[3] = chunk[3] ^ x[3].fold_16_pair(coeff_low, coeff_high);
  x[4] = chunk[4] ^ x[4].fold_16_pair(coeff_low, coeff_high);
  x[5] = chunk[5] ^ x[5].fold_16_pair(coeff_low, coeff_high);
  x[6] = chunk[6] ^ x[6].fold_16_pair(coeff_low, coeff_high);
  x[7] = chunk[7] ^ x[7].fold_16_pair(coeff_low, coeff_high);
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd_2way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(blocks.len() >= 2);

  let coeff_256_low = fold_256b.1;
  let coeff_256_high = fold_256b.0;
  let coeff_128_low = consts.fold_128b.1;
  let coeff_128_high = consts.fold_128b.0;

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];

  // Inject CRC into stream 0 (block 0).
  s0[0] ^= Simd::new(0, state);

  // Process the largest even prefix with 2-way striping.
  let mut i = 2;
  let even = blocks.len() & !1usize;
  while i < even {
    fold_block_128(&mut s0, &blocks[i], coeff_256_low, coeff_256_high);
    fold_block_128(&mut s1, &blocks[i + 1], coeff_256_low, coeff_256_high);
    i += 2;
  }

  // Merge streams: A·s0 ⊕ s1 (A = shift by 128B).
  let mut combined = s1;
  combined[0] ^= s0[0].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[1] ^= s0[1].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[2] ^= s0[2].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[3] ^= s0[3].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[4] ^= s0[4].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[5] ^= s0[5].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[6] ^= s0[6].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[7] ^= s0[7].fold_16_pair(coeff_128_low, coeff_128_high);

  // Handle any remaining block (odd tail) sequentially.
  if even != blocks.len() {
    fold_block_128(&mut combined, &blocks[even], coeff_128_low, coeff_128_high);
  }

  fold_tail(combined, consts)
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc64_pmull_2way(
  mut state: u64,
  bytes: &[u8],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.len() < 2 {
    return crc64_pmull(state, bytes, consts, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_2way(state, middle, fold_256b, consts);
  super::portable::crc64_slice8(state, right, tables)
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc64_pmull(mut state: u64, bytes: &[u8], consts: &Crc64ClmulConstants, tables: &[[u64; 256]; 8]) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if let Some((first, rest)) = middle.split_first() {
    state = super::portable::crc64_slice8(state, left, tables);
    state = update_simd(state, first, rest, consts);
    super::portable::crc64_slice8(state, right, tables)
  } else {
    super::portable::crc64_slice8(state, bytes, tables)
  }
}

/// Small-buffer PMULL path: fold one 16-byte lane at a time.
///
/// This targets the regime where full 128-byte folding has too much setup cost,
/// but PMULL still outperforms table CRC (typically ~16..127 bytes depending on CPU).
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc64_pmull_small(
  mut state: u64,
  bytes: &[u8],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<Simd>();

  // Prefix: portable until 16B alignment.
  state = super::portable::crc64_slice8(state, left, tables);

  // If we don't have any full 16B lane, finish portably.
  let Some((first, rest)) = middle.split_first() else {
    return super::portable::crc64_slice8(state, right, tables);
  };

  let mut acc = *first;
  acc ^= Simd::new(0, state);

  // Shift-by-16B folding coefficient (K_127, K_191).
  let coeff_16b_low = consts.tail_fold_16b[6].1;
  let coeff_16b_high = consts.tail_fold_16b[6].0;

  for chunk in rest {
    acc = *chunk ^ acc.fold_16_pair(coeff_16b_low, coeff_16b_high);
  }

  // Reduce 16B → 8B → u64, then finish any tail bytes portably.
  state = acc.fold_8(consts.fold_8b).barrett(consts.poly, consts.mu);
  super::portable::crc64_slice8(state, right, tables)
}

/// CRC-64-XZ using PMULL folding.
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_xz_pmull(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull(
      crc,
      data,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES,
    )
  }
}

/// CRC-64-XZ using PMULL (small-buffer lane folding).
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_xz_pmull_small(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_small(
      crc,
      data,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES,
    )
  }
}

/// CRC-64-XZ using a tuned "SVE2 PMULL" tier (2-way striping).
///
/// This is still implemented with NEON+PMULL intrinsics, but is intended for
/// high-throughput Armv9/SVE2-class CPUs where additional ILP helps.
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::SVE2_PMULL)` and `PMULL_READY` before selecting.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_xz_sve2_pmull_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_pmull_2way(crc, data) }
}

/// CRC-64-NVME using PMULL folding.
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_nvme_pmull(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull(
      crc,
      data,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES,
    )
  }
}

/// CRC-64-NVME using PMULL (small-buffer lane folding).
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_nvme_pmull_small(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_small(
      crc,
      data,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES,
    )
  }
}

/// CRC-64-NVME using a tuned "SVE2 PMULL" tier (2-way striping).
///
/// This is still implemented with NEON+PMULL intrinsics, but is intended for
/// high-throughput Armv9/SVE2-class CPUs where additional ILP helps.
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::SVE2_PMULL)` and `PMULL_READY` before selecting.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_nvme_sve2_pmull_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_pmull_2way(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Wrappers (safe interface)
// ─────────────────────────────────────────────────────────────────────────────

/// Safe wrapper for CRC-64-XZ PMULL kernel.
#[inline]
pub fn crc64_xz_pmull_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_xz_pmull(crc, data) }
}

/// Safe wrapper for CRC-64-XZ PMULL small-buffer kernel.
#[inline]
pub fn crc64_xz_pmull_small_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_xz_pmull_small(crc, data) }
}

/// Safe wrapper for CRC-64-XZ tuned "SVE2 PMULL" tier (single-stream).
#[inline]
pub fn crc64_xz_sve2_pmull_safe(crc: u64, data: &[u8]) -> u64 {
  crc64_xz_pmull_safe(crc, data)
}

/// Safe wrapper for CRC-64-XZ tuned "SVE2 PMULL" tier (small-buffer).
#[inline]
pub fn crc64_xz_sve2_pmull_small_safe(crc: u64, data: &[u8]) -> u64 {
  crc64_xz_pmull_small_safe(crc, data)
}

/// Safe wrapper for CRC-64-XZ tuned SVE2 2-way PMULL kernel.
#[inline]
pub fn crc64_xz_sve2_pmull_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies SVE2_PMULL + PMULL before selecting this kernel.
  unsafe { crc64_xz_sve2_pmull_2way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PMULL kernel.
#[inline]
pub fn crc64_nvme_pmull_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_nvme_pmull(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PMULL small-buffer kernel.
#[inline]
pub fn crc64_nvme_pmull_small_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_nvme_pmull_small(crc, data) }
}

/// Safe wrapper for CRC-64-NVME tuned "SVE2 PMULL" tier (single-stream).
#[inline]
pub fn crc64_nvme_sve2_pmull_safe(crc: u64, data: &[u8]) -> u64 {
  crc64_nvme_pmull_safe(crc, data)
}

/// Safe wrapper for CRC-64-NVME tuned "SVE2 PMULL" tier (small-buffer).
#[inline]
pub fn crc64_nvme_sve2_pmull_small_safe(crc: u64, data: &[u8]) -> u64 {
  crc64_nvme_pmull_small_safe(crc, data)
}

/// Safe wrapper for CRC-64-NVME tuned SVE2 2-way PMULL kernel.
#[inline]
pub fn crc64_nvme_sve2_pmull_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies SVE2_PMULL + PMULL before selecting this kernel.
  unsafe { crc64_nvme_sve2_pmull_2way(crc, data) }
}

/// CRC-64-XZ using PMULL folding (2-way striping).
///
/// This is still NEON+PMULL, but splits the main loop into two independent
/// streams to improve ILP on some microarchitectures (including Apple M-series).
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_xz_pmull_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_2way(
      crc,
      data,
      XZ_FOLD_256B,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES,
    )
  }
}

/// CRC-64-NVME using PMULL folding (2-way striping).
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_nvme_pmull_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_2way(
      crc,
      data,
      NVME_FOLD_256B,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES,
    )
  }
}

/// Safe wrapper for CRC-64-XZ PMULL 2-way kernel.
#[inline]
pub fn crc64_xz_pmull_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_xz_pmull_2way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PMULL 2-way kernel.
#[inline]
pub fn crc64_nvme_pmull_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_nvme_pmull_2way(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  const TEST_DATA: &[u8] = b"123456789";

  fn make_data(len: usize) -> alloc::vec::Vec<u8> {
    (0..len)
      .map(|i| (i as u8).wrapping_mul(17).wrapping_add((i >> 3) as u8))
      .collect()
  }

  #[test]
  fn test_crc64_xz_pmull_matches_vector() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    // SAFETY: We just checked AES/PMULL is available.
    let crc = unsafe { crc64_xz_pmull(!0, TEST_DATA) } ^ !0;
    assert_eq!(crc, 0x995D_C9BB_DF19_39FA);
  }

  #[test]
  fn test_crc64_nvme_pmull_matches_vector() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    // SAFETY: We just checked AES/PMULL is available.
    let crc = unsafe { crc64_nvme_pmull(!0, TEST_DATA) } ^ !0;
    assert_eq!(crc, 0xAE8B_1486_0A79_9888);
  }

  #[test]
  fn test_crc64_xz_pmull_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in [
      0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256, 511, 512, 1024,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      // SAFETY: We just checked AES/PMULL is available.
      let pmull = unsafe { crc64_xz_pmull(!0, &data) } ^ !0;
      assert_eq!(pmull, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in [
      0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256, 511, 512, 1024,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      // SAFETY: We just checked AES/PMULL is available.
      let pmull = unsafe { crc64_nvme_pmull(!0, &data) } ^ !0;
      assert_eq!(pmull, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_pmull_small_matches_portable_all_lengths_0_127() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in 0..128 {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let pmull_small = crc64_xz_pmull_small_safe(!0, &data) ^ !0;
      assert_eq!(pmull_small, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_small_matches_portable_all_lengths_0_127() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in 0..128 {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let pmull_small = crc64_nvme_pmull_small_safe(!0, &data) ^ !0;
      assert_eq!(pmull_small, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_sve2_pmull_2way_matches_portable_various_lengths() {
    let caps = platform::caps();
    if !caps.has(platform::caps::aarch64::SVE2_PMULL) || !caps.has(platform::caps::aarch64::PMULL_READY) {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1024, 4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let sve2 = crc64_xz_sve2_pmull_2way_safe(!0, &data) ^ !0;
      assert_eq!(sve2, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_sve2_pmull_2way_matches_portable_various_lengths() {
    let caps = platform::caps();
    if !caps.has(platform::caps::aarch64::SVE2_PMULL) || !caps.has(platform::caps::aarch64::PMULL_READY) {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1024, 4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let sve2 = crc64_nvme_sve2_pmull_2way_safe(!0, &data) ^ !0;
      assert_eq!(sve2, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_pmull_2way_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257, 383, 384, 511, 512, 513, 1024, 4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let pmull_2way = crc64_xz_pmull_2way_safe(!0, &data) ^ !0;
      assert_eq!(pmull_2way, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_2way_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257, 383, 384, 511, 512, 513, 1024, 4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let pmull_2way = crc64_nvme_pmull_2way_safe(!0, &data) ^ !0;
      assert_eq!(pmull_2way, portable, "mismatch at len={len}");
    }
  }
}
