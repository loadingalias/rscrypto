//! s390x hardware-accelerated CRC-16 kernels (VGFM-style).
//!
//! This is a VGFM implementation of the 128-byte width32 folding algorithm
//! used by the reflected CRC-16 CLMUL backends.
//!
//! # Safety
//!
//! Uses `unsafe` for s390x vector + inline assembly. Callers must ensure the
//! required CPU features are available before executing the accelerated path
//! (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(dead_code)] // Kernels wired up via dispatcher
// SAFETY: All indexing is over fixed-size arrays with in-bounds constant indices.
#![allow(clippy::indexing_slicing)]
// This module is intrinsics-heavy; keep unsafe blocks readable.
#![allow(unsafe_op_in_unsafe_fn)]

use core::{
  arch::asm,
  mem::MaybeUninit,
  ops::{BitAnd, BitXor, BitXorAssign},
  simd::i64x2,
};

use super::keys::{
  CRC16_CCITT_KEYS_REFLECTED, CRC16_CCITT_STREAM_REFLECTED, CRC16_IBM_KEYS_REFLECTED, CRC16_IBM_STREAM_REFLECTED,
};

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
    Self(i64x2::from_array([high as i64, low as i64]))
  }

  #[inline]
  fn low_64(self) -> u64 {
    self.0.to_array()[1] as u64
  }

  #[inline]
  fn high_64(self) -> u64 {
    self.0.to_array()[0] as u64
  }

  #[inline]
  fn swap_lanes(self) -> Self {
    let [hi, lo] = self.0.to_array();
    Self(i64x2::from_array([lo, hi]))
  }

  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn vgfm(a: i64x2, b: i64x2) -> i64x2 {
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

  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn mul64(a: u64, b: u64) -> Self {
    let va = Self::new(0, a);
    let vb = Self::new(0, b);
    Self(Self::vgfm(va.0, vb.0))
  }

  /// Fold 16 bytes (reflected width32 folding primitive):
  /// `self.low ⊗ coeff.high ⊕ self.high ⊗ coeff.low`.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn fold_16(self, coeff: Self) -> Self {
    // Like VPMSUMD: VGFM performs a per-lane carryless multiply and XORs the lane products.
    // The folding primitive needs cross terms, so swap coefficient lanes.
    Self(Self::vgfm(self.0, coeff.swap_lanes().0))
  }

  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn fold_16_reflected(self, coeff: Self, data_to_xor: Self) -> Self {
    data_to_xor ^ self.fold_16(coeff)
  }

  /// Fold 16 bytes down to the "width32" reduction state (reflected mode).
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn fold_width32_reflected(self, high: u64, low: u64) -> Self {
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

  /// Barrett reduction for reflected width32; returns the updated CRC state.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn barrett_width32_reflected(self, poly: u64, mu: u64) -> u32 {
    let t1 = Self::mul64(self.low_64(), mu);
    let l = Self::mul64(t1.low_64(), poly);
    (self ^ l).high_64() as u32
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Load helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn load_block(block: &Block) -> [Simd; 8] {
  let mut out = MaybeUninit::<[Simd; 8]>::uninit();
  let base = out.as_mut_ptr().cast::<Simd>();

  let mut i = 0;
  while i < 8 {
    let low = u64::from_le(block[i * 2]);
    let high = u64::from_le(block[i * 2 + 1]);
    // SAFETY: `base` points to a `[Simd; 8]` buffer and `i` is in-bounds.
    unsafe {
      base.add(i).write(Simd::new(high, low));
    }
    i = i.strict_add(1);
  }

  // SAFETY: all 8 elements are initialized above.
  unsafe { out.assume_init() }
}

// ─────────────────────────────────────────────────────────────────────────────
// Folding helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "vector")]
unsafe fn finalize_lanes_width32_reflected(x: [Simd; 8], keys: &[u64; 23]) -> u32 {
  let mut res = x[7];
  res = x[0].fold_16_reflected(Simd::new(keys[10], keys[9]), res);
  res = x[1].fold_16_reflected(Simd::new(keys[12], keys[11]), res);
  res = x[2].fold_16_reflected(Simd::new(keys[14], keys[13]), res);
  res = x[3].fold_16_reflected(Simd::new(keys[16], keys[15]), res);
  res = x[4].fold_16_reflected(Simd::new(keys[18], keys[17]), res);
  res = x[5].fold_16_reflected(Simd::new(keys[20], keys[19]), res);
  res = x[6].fold_16_reflected(Simd::new(keys[2], keys[1]), res);

  res = res.fold_width32_reflected(keys[6], keys[5]);
  res.barrett_width32_reflected(keys[8], keys[7])
}

#[inline]
#[target_feature(enable = "vector")]
unsafe fn update_simd_width32_reflected(state: u32, first: &Block, rest: &[Block], keys: &[u64; 23]) -> u32 {
  let mut x = load_block(first);

  x[0] ^= Simd::new(0, state as u64);

  let coeff_128b = Simd::new(keys[4], keys[3]);
  for block in rest {
    let chunk = load_block(block);
    x[0] = x[0].fold_16_reflected(coeff_128b, chunk[0]);
    x[1] = x[1].fold_16_reflected(coeff_128b, chunk[1]);
    x[2] = x[2].fold_16_reflected(coeff_128b, chunk[2]);
    x[3] = x[3].fold_16_reflected(coeff_128b, chunk[3]);
    x[4] = x[4].fold_16_reflected(coeff_128b, chunk[4]);
    x[5] = x[5].fold_16_reflected(coeff_128b, chunk[5]);
    x[6] = x[6].fold_16_reflected(coeff_128b, chunk[6]);
    x[7] = x[7].fold_16_reflected(coeff_128b, chunk[7]);
  }

  finalize_lanes_width32_reflected(x, keys)
}

#[inline]
#[target_feature(enable = "vector")]
unsafe fn fold_block_128_reflected(x: &mut [Simd; 8], block: &Block, coeff: Simd) {
  let chunk = load_block(block);
  x[0] = x[0].fold_16_reflected(coeff, chunk[0]);
  x[1] = x[1].fold_16_reflected(coeff, chunk[1]);
  x[2] = x[2].fold_16_reflected(coeff, chunk[2]);
  x[3] = x[3].fold_16_reflected(coeff, chunk[3]);
  x[4] = x[4].fold_16_reflected(coeff, chunk[4]);
  x[5] = x[5].fold_16_reflected(coeff, chunk[5]);
  x[6] = x[6].fold_16_reflected(coeff, chunk[6]);
  x[7] = x[7].fold_16_reflected(coeff, chunk[7]);
}

#[inline]
#[target_feature(enable = "vector")]
unsafe fn update_simd_width32_reflected_2way(
  state: u32,
  blocks: &[Block],
  fold_256b: (u64, u64),
  keys: &[u64; 23],
) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 2 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected(state, first, rest, keys);
  }

  let coeff_256 = Simd::new(fold_256b.0, fold_256b.1);
  let coeff_128 = Simd::new(keys[4], keys[3]);

  let mut s0 = load_block(&blocks[0]);
  let mut s1 = load_block(&blocks[1]);

  s0[0] ^= Simd::new(0, state as u64);

  let mut i: usize = 2;
  let even = blocks.len() & !1usize;
  while i < even {
    fold_block_128_reflected(&mut s0, &blocks[i], coeff_256);
    fold_block_128_reflected(&mut s1, &blocks[i.strict_add(1)], coeff_256);
    i = i.strict_add(2);
  }

  // Merge: A·s0 ⊕ s1.
  let mut combined = s1;
  combined[0] = s0[0].fold_16_reflected(coeff_128, combined[0]);
  combined[1] = s0[1].fold_16_reflected(coeff_128, combined[1]);
  combined[2] = s0[2].fold_16_reflected(coeff_128, combined[2]);
  combined[3] = s0[3].fold_16_reflected(coeff_128, combined[3]);
  combined[4] = s0[4].fold_16_reflected(coeff_128, combined[4]);
  combined[5] = s0[5].fold_16_reflected(coeff_128, combined[5]);
  combined[6] = s0[6].fold_16_reflected(coeff_128, combined[6]);
  combined[7] = s0[7].fold_16_reflected(coeff_128, combined[7]);

  if even != blocks.len() {
    fold_block_128_reflected(&mut combined, &blocks[even], coeff_128);
  }

  finalize_lanes_width32_reflected(combined, keys)
}

#[inline]
#[target_feature(enable = "vector")]
unsafe fn update_simd_width32_reflected_4way(
  state: u32,
  blocks: &[Block],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  keys: &[u64; 23],
) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 4 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected(state, first, rest, keys);
  }

  let aligned = (blocks.len() / 4) * 4;

  let coeff_512 = Simd::new(fold_512b.0, fold_512b.1);
  let coeff_128 = Simd::new(keys[4], keys[3]);
  let c384 = Simd::new(combine[0].0, combine[0].1);
  let c256 = Simd::new(combine[1].0, combine[1].1);
  let c128 = Simd::new(combine[2].0, combine[2].1);

  let mut s0 = load_block(&blocks[0]);
  let mut s1 = load_block(&blocks[1]);
  let mut s2 = load_block(&blocks[2]);
  let mut s3 = load_block(&blocks[3]);

  s0[0] ^= Simd::new(0, state as u64);

  let mut i: usize = 4;
  while i < aligned {
    fold_block_128_reflected(&mut s0, &blocks[i], coeff_512);
    fold_block_128_reflected(&mut s1, &blocks[i.strict_add(1)], coeff_512);
    fold_block_128_reflected(&mut s2, &blocks[i.strict_add(2)], coeff_512);
    fold_block_128_reflected(&mut s3, &blocks[i.strict_add(3)], coeff_512);
    i = i.strict_add(4);
  }

  // Merge: A^3·s0 ⊕ A^2·s1 ⊕ A·s2 ⊕ s3.
  let mut acc = s3;
  acc[0] = s2[0].fold_16_reflected(c128, acc[0]);
  acc[1] = s2[1].fold_16_reflected(c128, acc[1]);
  acc[2] = s2[2].fold_16_reflected(c128, acc[2]);
  acc[3] = s2[3].fold_16_reflected(c128, acc[3]);
  acc[4] = s2[4].fold_16_reflected(c128, acc[4]);
  acc[5] = s2[5].fold_16_reflected(c128, acc[5]);
  acc[6] = s2[6].fold_16_reflected(c128, acc[6]);
  acc[7] = s2[7].fold_16_reflected(c128, acc[7]);

  acc[0] = s1[0].fold_16_reflected(c256, acc[0]);
  acc[1] = s1[1].fold_16_reflected(c256, acc[1]);
  acc[2] = s1[2].fold_16_reflected(c256, acc[2]);
  acc[3] = s1[3].fold_16_reflected(c256, acc[3]);
  acc[4] = s1[4].fold_16_reflected(c256, acc[4]);
  acc[5] = s1[5].fold_16_reflected(c256, acc[5]);
  acc[6] = s1[6].fold_16_reflected(c256, acc[6]);
  acc[7] = s1[7].fold_16_reflected(c256, acc[7]);

  acc[0] = s0[0].fold_16_reflected(c384, acc[0]);
  acc[1] = s0[1].fold_16_reflected(c384, acc[1]);
  acc[2] = s0[2].fold_16_reflected(c384, acc[2]);
  acc[3] = s0[3].fold_16_reflected(c384, acc[3]);
  acc[4] = s0[4].fold_16_reflected(c384, acc[4]);
  acc[5] = s0[5].fold_16_reflected(c384, acc[5]);
  acc[6] = s0[6].fold_16_reflected(c384, acc[6]);
  acc[7] = s0[7].fold_16_reflected(c384, acc[7]);

  for block in &blocks[aligned..] {
    fold_block_128_reflected(&mut acc, block, coeff_128);
  }

  finalize_lanes_width32_reflected(acc, keys)
}

#[inline]
#[target_feature(enable = "vector")]
unsafe fn crc16_width32_vgfm(mut state: u16, data: &[u8], keys: &[u64; 23], portable: fn(u16, &[u8]) -> u16) -> u16 {
  let (left, middle, right) = data.align_to::<Block>();
  let Some((first, rest)) = middle.split_first() else {
    return portable(state, data);
  };

  state = portable(state, left);
  let state32 = update_simd_width32_reflected(state as u32, first, rest, keys);
  state = state32 as u16;
  portable(state, right)
}

#[inline]
#[target_feature(enable = "vector")]
unsafe fn crc16_width32_vgfm_2way(
  mut state: u16,
  data: &[u8],
  keys: &[u64; 23],
  stream: &super::keys::Width32StreamConstants,
  portable: fn(u16, &[u8]) -> u16,
) -> u16 {
  let (left, middle, right) = data.align_to::<Block>();
  if middle.is_empty() {
    return portable(state, data);
  }

  state = portable(state, left);
  let state32 = update_simd_width32_reflected_2way(state as u32, middle, stream.fold_256b, keys);
  state = state32 as u16;
  portable(state, right)
}

#[inline]
#[target_feature(enable = "vector")]
unsafe fn crc16_width32_vgfm_4way(
  mut state: u16,
  data: &[u8],
  keys: &[u64; 23],
  stream: &super::keys::Width32StreamConstants,
  portable: fn(u16, &[u8]) -> u16,
) -> u16 {
  let (left, middle, right) = data.align_to::<Block>();
  if middle.is_empty() {
    return portable(state, data);
  }

  state = portable(state, left);
  let state32 = update_simd_width32_reflected_4way(state as u32, middle, stream.fold_512b, &stream.combine_4way, keys);
  state = state32 as u16;
  portable(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Safe Kernels
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-16/CCITT VGFM kernel.
///
/// # Safety
///
/// Dispatcher verifies VECTOR facility before selecting this kernel.
#[inline]
pub fn crc16_ccitt_vgfm_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies VECTOR facility before selecting this kernel.
  unsafe {
    crc16_width32_vgfm(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

#[inline]
pub fn crc16_ccitt_vgfm_2way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vgfm_2way(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      &CRC16_CCITT_STREAM_REFLECTED,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

#[inline]
pub fn crc16_ccitt_vgfm_4way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vgfm_4way(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      &CRC16_CCITT_STREAM_REFLECTED,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/IBM VGFM kernel.
///
/// # Safety
///
/// Dispatcher verifies VECTOR facility before selecting this kernel.
#[inline]
pub fn crc16_ibm_vgfm_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies VECTOR facility before selecting this kernel.
  unsafe { crc16_width32_vgfm(crc, data, &CRC16_IBM_KEYS_REFLECTED, super::portable::crc16_ibm_slice8) }
}

#[inline]
pub fn crc16_ibm_vgfm_2way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vgfm_2way(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      &CRC16_IBM_STREAM_REFLECTED,
      super::portable::crc16_ibm_slice8,
    )
  }
}

#[inline]
pub fn crc16_ibm_vgfm_4way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vgfm_4way(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      &CRC16_IBM_STREAM_REFLECTED,
      super::portable::crc16_ibm_slice8,
    )
  }
}
