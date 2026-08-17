//! s390x hardware-accelerated CRC-32/CRC-32C kernels (VGFM-style).
//!
//! This is a VGFM implementation of the 128-byte folding algorithm used by
//! our reflected CRC32 CLMUL backends.
//!
//! # Safety
//!
//! Uses `unsafe` for s390x vector + inline assembly. Callers must ensure the
//! required CPU features are available before executing the accelerated path
//! (the dispatcher does this).

use core::{
  arch::asm,
  ops::{BitAnd, BitXor, BitXorAssign},
  simd::i64x2,
};

use crate::checksum::common::low_u32;

use super::clmul::{Crc32ClmulConstants, Crc32StreamConstants};

type Block = [u64; 16]; // 128 bytes (8×16B lanes)

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
struct Simd(i64x2);

impl BitXor for Simd {
  type Output = Self;

  #[inline]
  fn bitxor(self, other: Self) -> Self {
    Self(self.0 ^ other.0)
  }
}

impl BitXorAssign for Simd {
  #[inline]
  fn bitxor_assign(&mut self, other: Self) {
    *self = *self ^ other;
  }
}

impl BitAnd for Simd {
  type Output = Self;

  #[inline]
  fn bitand(self, rhs: Self) -> Self::Output {
    Self(self.0 & rhs.0)
  }
}

impl Simd {
  #[inline]
  fn new(high: u64, low: u64) -> Self {
    // On s390x, vector element 0 maps to the most-significant 64 bits.
    Self(i64x2::from_array([high.cast_signed(), low.cast_signed()]))
  }

  #[inline]
  fn low_64(self) -> u64 {
    self.0.to_array()[1].cast_unsigned()
  }

  #[inline]
  fn high_64(self) -> u64 {
    self.0.to_array()[0].cast_unsigned()
  }

  #[inline]
  fn swap_lanes(self) -> Self {
    let [hi, lo] = self.0.to_array();
    Self(i64x2::from_array([lo, hi]))
  }

  #[inline]
  /// # Safety
  ///
  /// Requires the s390x vector facility.
  #[target_feature(enable = "vector")]
  fn vgfm(a: i64x2, b: i64x2) -> i64x2 {
    // SAFETY: Caller guarantees the s390x vector facility is available
    // (verified by dispatch). The VGFM instruction operates on pure register
    // values with no memory access.
    unsafe {
      let out: i64x2;
      asm!(
        "vgfm {out}, {a}, {b}, 3",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure)
      );
      out
    }
  }

  #[inline]
  /// # Safety
  ///
  /// Requires the s390x vector facility.
  #[target_feature(enable = "vector")]
  fn mul64(a: u64, b: u64) -> Self {
    let va = Self::new(0, a);
    let vb = Self::new(0, b);
    Self(Self::vgfm(va.0, vb.0))
  }

  /// Fold 16 bytes (reflected CRC32 folding primitive):
  /// `self.low ⊗ coeff.high ⊕ self.high ⊗ coeff.low`.
  #[inline]
  /// # Safety
  ///
  /// Requires the s390x vector facility.
  #[target_feature(enable = "vector")]
  fn fold_16(self, coeff: Self) -> Self {
    Self(Self::vgfm(self.0, coeff.swap_lanes().0))
  }

  /// Fold 16B → CRC32 width (reflected), returning an intermediate 128-bit state.
  #[inline]
  /// # Safety
  ///
  /// Requires the s390x vector facility.
  #[target_feature(enable = "vector")]
  fn fold_width_crc32_reflected(self, high: u64, low: u64) -> Self {
    let clmul = Self::mul64(self.low_64(), low);
    let shifted = Self::new(0, self.high_64());
    let mut state = clmul ^ shifted;

    let mask2 = Self::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_0000_0000);
    let masked = state & mask2;
    let shifted_high = (state.low_64() & 0xFFFF_FFFF).strict_shl(32);
    let clmul = Self::mul64(shifted_high, high);
    state = clmul ^ masked;

    state
  }

  /// Barrett reduction for reflected CRC32; returns the updated (pre-inverted) CRC.
  #[inline]
  /// # Safety
  ///
  /// Requires the s390x vector facility.
  #[target_feature(enable = "vector")]
  fn barrett_crc32_reflected(self, poly: u64, mu: u64) -> u32 {
    let t1 = Self::mul64(self.low_64(), mu);
    let l = Self::mul64(t1.low_64(), poly);
    low_u32((self ^ l).high_64())
  }
}

// Load helpers

#[inline(always)]
fn load_block(block: &Block) -> [Simd; 8] {
  let mut out = [Simd::new(0, 0); 8];
  for (lane, &[low, high]) in out.iter_mut().zip(block.as_chunks::<2>().0) {
    *lane = Simd::new(u64::from_le(high), u64::from_le(low));
  }
  out
}

// Folding helpers

#[inline]
/// # Safety
///
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
fn fold_tail(x: [Simd; 8], consts: &Crc32ClmulConstants) -> u32 {
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

  let (fold_width_high, fold_width_low) = consts.fold_width;
  let state = acc.fold_width_crc32_reflected(fold_width_high, fold_width_low);
  state.barrett_crc32_reflected(consts.poly, consts.mu)
}

#[inline]
/// # Safety
///
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
fn fold_block_128(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff: Simd) {
  x[0] = chunk[0] ^ x[0].fold_16(coeff);
  x[1] = chunk[1] ^ x[1].fold_16(coeff);
  x[2] = chunk[2] ^ x[2].fold_16(coeff);
  x[3] = chunk[3] ^ x[3].fold_16(coeff);
  x[4] = chunk[4] ^ x[4].fold_16(coeff);
  x[5] = chunk[5] ^ x[5].fold_16(coeff);
  x[6] = chunk[6] ^ x[6].fold_16(coeff);
  x[7] = chunk[7] ^ x[7].fold_16(coeff);
}

/// # Safety
///
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
fn update_simd(state: u32, first: &Block, rest: &[Block], consts: &Crc32ClmulConstants) -> u32 {
  let mut x = load_block(first);

  // XOR initial CRC into the first 16-byte lane (low 32 bits).
  x[0] ^= Simd::new(0, state as u64);

  let coeff = Simd::new(consts.fold_128b.0, consts.fold_128b.1);
  for block in rest {
    let chunk = load_block(block);
    fold_block_128(&mut x, &chunk, coeff);
  }

  fold_tail(x, consts)
}

/// # Safety
///
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
fn update_simd_2way(state: u32, blocks: &[Block], fold_256b: (u64, u64), consts: &Crc32ClmulConstants) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 2 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd(state, first, rest, consts);
  }

  let even = blocks.len() & !1usize;

  let coeff_256 = Simd::new(fold_256b.0, fold_256b.1);
  let coeff_128 = Simd::new(consts.fold_128b.0, consts.fold_128b.1);

  let mut s0 = load_block(&blocks[0]);
  let mut s1 = load_block(&blocks[1]);
  s0[0] ^= Simd::new(0, state as u64);

  let mut i = 2usize;
  while i < even {
    let b0 = load_block(&blocks[i]);
    let b1 = load_block(&blocks[i.strict_add(1)]);
    fold_block_128(&mut s0, &b0, coeff_256);
    fold_block_128(&mut s1, &b1, coeff_256);
    i = i.strict_add(2);
  }

  let mut combined = s1;
  combined[0] ^= s0[0].fold_16(coeff_128);
  combined[1] ^= s0[1].fold_16(coeff_128);
  combined[2] ^= s0[2].fold_16(coeff_128);
  combined[3] ^= s0[3].fold_16(coeff_128);
  combined[4] ^= s0[4].fold_16(coeff_128);
  combined[5] ^= s0[5].fold_16(coeff_128);
  combined[6] ^= s0[6].fold_16(coeff_128);
  combined[7] ^= s0[7].fold_16(coeff_128);

  if even != blocks.len() {
    let tail = load_block(&blocks[even]);
    fold_block_128(&mut combined, &tail, coeff_128);
  }

  fold_tail(combined, consts)
}

/// # Safety
///
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
fn update_simd_4way(
  state: u32,
  blocks: &[Block],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc32ClmulConstants,
) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 4 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd(state, first, rest, consts);
  }

  let aligned = blocks.len().strict_div(4).strict_mul(4);

  let coeff_512 = Simd::new(fold_512b.0, fold_512b.1);
  let coeff_128 = Simd::new(consts.fold_128b.0, consts.fold_128b.1);
  let c384 = Simd::new(combine[0].0, combine[0].1);
  let c256 = Simd::new(combine[1].0, combine[1].1);
  let c128 = Simd::new(combine[2].0, combine[2].1);

  let mut s0 = load_block(&blocks[0]);
  let mut s1 = load_block(&blocks[1]);
  let mut s2 = load_block(&blocks[2]);
  let mut s3 = load_block(&blocks[3]);
  s0[0] ^= Simd::new(0, state as u64);

  let mut i = 4usize;
  while i < aligned {
    let b0 = load_block(&blocks[i]);
    let b1 = load_block(&blocks[i.strict_add(1)]);
    let b2 = load_block(&blocks[i.strict_add(2)]);
    let b3 = load_block(&blocks[i.strict_add(3)]);
    fold_block_128(&mut s0, &b0, coeff_512);
    fold_block_128(&mut s1, &b1, coeff_512);
    fold_block_128(&mut s2, &b2, coeff_512);
    fold_block_128(&mut s3, &b3, coeff_512);
    i = i.strict_add(4);
  }

  let mut combined = s3;
  combined[0] ^= s2[0].fold_16(c128);
  combined[1] ^= s2[1].fold_16(c128);
  combined[2] ^= s2[2].fold_16(c128);
  combined[3] ^= s2[3].fold_16(c128);
  combined[4] ^= s2[4].fold_16(c128);
  combined[5] ^= s2[5].fold_16(c128);
  combined[6] ^= s2[6].fold_16(c128);
  combined[7] ^= s2[7].fold_16(c128);

  combined[0] ^= s1[0].fold_16(c256);
  combined[1] ^= s1[1].fold_16(c256);
  combined[2] ^= s1[2].fold_16(c256);
  combined[3] ^= s1[3].fold_16(c256);
  combined[4] ^= s1[4].fold_16(c256);
  combined[5] ^= s1[5].fold_16(c256);
  combined[6] ^= s1[6].fold_16(c256);
  combined[7] ^= s1[7].fold_16(c256);

  combined[0] ^= s0[0].fold_16(c384);
  combined[1] ^= s0[1].fold_16(c384);
  combined[2] ^= s0[2].fold_16(c384);
  combined[3] ^= s0[3].fold_16(c384);
  combined[4] ^= s0[4].fold_16(c384);
  combined[5] ^= s0[5].fold_16(c384);
  combined[6] ^= s0[6].fold_16(c384);
  combined[7] ^= s0[7].fold_16(c384);

  if aligned != blocks.len() {
    let tail_blocks = &blocks[aligned..];
    let Some((first, rest)) = tail_blocks.split_first() else {
      return fold_tail(combined, consts);
    };
    let mut x = combined;
    let first = load_block(first);
    fold_block_128(&mut x, &first, coeff_128);
    for b in rest {
      let chunk = load_block(b);
      fold_block_128(&mut x, &chunk, coeff_128);
    }
    return fold_tail(x, consts);
  }

  fold_tail(combined, consts)
}

// Public kernels (IEEE + CRC32C)

#[inline]
/// # Safety
///
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
fn crc32_kernel(crc: u32, data: &[u8], consts: &Crc32ClmulConstants) -> u32 {
  // SAFETY: Every bit pattern is valid for Block; align_to returns
  // non-overlapping subslices of the original allocation.
  unsafe {
    let (left, middle, right) = data.align_to::<Block>();
    let Some((first, rest)) = middle.split_first() else {
      return super::portable::crc32_slice16_ieee(crc, data);
    };

    let mut state = super::portable::crc32_slice16_ieee(crc, left);
    state = update_simd(state, first, rest, consts);
    super::portable::crc32_slice16_ieee(state, right)
  }
}

#[inline]
/// # Safety
///
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
fn crc32c_kernel(crc: u32, data: &[u8], consts: &Crc32ClmulConstants) -> u32 {
  // SAFETY: Every bit pattern is valid for Block; align_to returns
  // non-overlapping subslices of the original allocation.
  unsafe {
    let (left, middle, right) = data.align_to::<Block>();
    let Some((first, rest)) = middle.split_first() else {
      return super::portable::crc32c_slice16(crc, data);
    };

    let mut state = super::portable::crc32c_slice16(crc, left);
    state = update_simd(state, first, rest, consts);
    super::portable::crc32c_slice16(state, right)
  }
}

#[inline]
/// # Safety
///
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
fn crc32_kernel_nway<const N: usize>(
  crc: u32,
  data: &[u8],
  stream: &Crc32StreamConstants,
  consts: &Crc32ClmulConstants,
) -> u32 {
  debug_assert!(N == 2 || N == 4);
  // SAFETY: Every bit pattern is valid for Block; align_to returns
  // non-overlapping subslices of the original allocation.
  unsafe {
    let (left, middle, right) = data.align_to::<Block>();
    if middle.is_empty() {
      return super::portable::crc32_slice16_ieee(crc, data);
    }

    let mut state = super::portable::crc32_slice16_ieee(crc, left);
    state = match N {
      2 => update_simd_2way(state, middle, stream.fold_256b, consts),
      _ => update_simd_4way(state, middle, stream.fold_512b, &stream.combine_4way, consts),
    };
    super::portable::crc32_slice16_ieee(state, right)
  }
}

#[inline]
/// # Safety
///
/// Requires the s390x vector facility.
#[target_feature(enable = "vector")]
fn crc32c_kernel_nway<const N: usize>(
  crc: u32,
  data: &[u8],
  stream: &Crc32StreamConstants,
  consts: &Crc32ClmulConstants,
) -> u32 {
  debug_assert!(N == 2 || N == 4);
  // SAFETY: Every bit pattern is valid for Block; align_to returns
  // non-overlapping subslices of the original allocation.
  unsafe {
    let (left, middle, right) = data.align_to::<Block>();
    if middle.is_empty() {
      return super::portable::crc32c_slice16(crc, data);
    }

    let mut state = super::portable::crc32c_slice16(crc, left);
    state = match N {
      2 => update_simd_2way(state, middle, stream.fold_256b, consts),
      _ => update_simd_4way(state, middle, stream.fold_512b, &stream.combine_4way, consts),
    };
    super::portable::crc32c_slice16(state, right)
  }
}

// Safe wrappers (dispatcher entrypoints)

#[inline]
pub(super) fn crc32_ieee_vgfm_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies vector support before selecting this kernel.
  unsafe { crc32_kernel(crc, data, &super::clmul::CRC32_IEEE_CLMUL) }
}

#[inline]
pub(super) fn crc32_ieee_vgfm_2way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies vector support before selecting this kernel.
  unsafe {
    crc32_kernel_nway::<2>(
      crc,
      data,
      &super::clmul::CRC32_IEEE_STREAM,
      &super::clmul::CRC32_IEEE_CLMUL,
    )
  }
}

#[inline]
pub(super) fn crc32_ieee_vgfm_4way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies vector support before selecting this kernel.
  unsafe {
    crc32_kernel_nway::<4>(
      crc,
      data,
      &super::clmul::CRC32_IEEE_STREAM,
      &super::clmul::CRC32_IEEE_CLMUL,
    )
  }
}

#[inline]
pub(super) fn crc32c_vgfm_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies vector support before selecting this kernel.
  unsafe { crc32c_kernel(crc, data, &super::clmul::CRC32C_CLMUL) }
}

#[inline]
pub(super) fn crc32c_vgfm_2way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies vector support before selecting this kernel.
  unsafe { crc32c_kernel_nway::<2>(crc, data, &super::clmul::CRC32C_STREAM, &super::clmul::CRC32C_CLMUL) }
}

#[inline]
pub(super) fn crc32c_vgfm_4way_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies vector support before selecting this kernel.
  unsafe { crc32c_kernel_nway::<4>(crc, data, &super::clmul::CRC32C_STREAM, &super::clmul::CRC32C_CLMUL) }
}

// Tests

#[cfg(test)]
mod tests {
  extern crate std;

  use alloc::vec::Vec;

  use super::*;

  const LENS: &[usize] = &[0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1024, 16 * 1024];

  fn make_data(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| (i as u8).wrapping_mul(29).wrapping_add((i >> 8) as u8))
      .collect()
  }

  fn assert_crc32_kernel(name: &str, kernel: fn(u32, &[u8]) -> u32) {
    for &len in LENS {
      let data = make_data(len);
      let expected = super::super::portable::crc32_slice16_ieee(!0, &data) ^ !0;
      let got = kernel(!0, &data) ^ !0;
      assert_eq!(got, expected, "{name} len={len}");
    }
  }

  fn assert_crc32c_kernel(name: &str, kernel: fn(u32, &[u8]) -> u32) {
    for &len in LENS {
      let data = make_data(len);
      let expected = super::super::portable::crc32c_slice16(!0, &data) ^ !0;
      let got = kernel(!0, &data) ^ !0;
      assert_eq!(got, expected, "{name} len={len}");
    }
  }

  #[test]
  fn test_crc32_vgfm_matches_portable_various_lengths() {
    let caps = crate::platform::caps();
    if !caps.has(crate::platform::caps::s390x::VECTOR) {
      return;
    }

    assert_crc32_kernel("crc32/vgfm", crc32_ieee_vgfm_safe);
    assert_crc32_kernel("crc32/vgfm-2way", crc32_ieee_vgfm_2way_safe);
    assert_crc32_kernel("crc32/vgfm-4way", crc32_ieee_vgfm_4way_safe);

    assert_crc32c_kernel("crc32c/vgfm", crc32c_vgfm_safe);
    assert_crc32c_kernel("crc32c/vgfm-2way", crc32c_vgfm_2way_safe);
    assert_crc32c_kernel("crc32c/vgfm-4way", crc32c_vgfm_4way_safe);
  }
}
