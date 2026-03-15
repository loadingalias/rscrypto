//! Power hardware-accelerated CRC-24/OPENPGP kernels (VPMSUMD-style).
//!
//! CRC-24/OPENPGP is MSB-first. These kernels reuse the reflected width32
//! folding/reduction structure by processing per-byte bit-reversed input and
//! converting the CRC state between OpenPGP and reflected forms.
//!
//! # Safety
//!
//! Uses `unsafe` for Power SIMD + inline assembly. Callers must ensure the
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
  ops::{BitAnd, BitXor, BitXorAssign},
  simd::i64x2,
};

use super::{
  keys::{CRC24_OPENPGP_KEYS_REFLECTED, CRC24_OPENPGP_STREAM_REFLECTED},
  reflected::{crc24_reflected_update_bitrev_bytes, from_reflected_state, to_reflected_state},
};

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

#[inline]
#[must_use]
const fn bitrev_bytes_u64(mut x: u64) -> u64 {
  x = ((x & 0x5555_5555_5555_5555).strict_shl(1)) | ((x.strict_shr(1)) & 0x5555_5555_5555_5555);
  x = ((x & 0x3333_3333_3333_3333).strict_shl(2)) | ((x.strict_shr(2)) & 0x3333_3333_3333_3333);
  x = ((x & 0x0F0F_0F0F_0F0F_0F0F).strict_shl(4)) | ((x.strict_shr(4)) & 0x0F0F_0F0F_0F0F_0F0F);
  x
}

impl Simd {
  #[inline]
  fn new(high: u64, low: u64) -> Self {
    // Match the x86/aarch64 lane layout: lane0 = low, lane1 = high.
    Self(i64x2::from_array([low as i64, high as i64]))
  }

  #[inline]
  fn low_64(self) -> u64 {
    self.0.to_array()[0] as u64
  }

  #[inline]
  fn high_64(self) -> u64 {
    self.0.to_array()[1] as u64
  }

  #[inline]
  fn swap_lanes(self) -> Self {
    let [lo, hi] = self.0.to_array();
    Self(i64x2::from_array([hi, lo]))
  }

  /// Normalize a loaded vector to little-endian lane encoding.
  #[inline]
  #[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
  unsafe fn to_le(self) -> Self {
    #[cfg(target_endian = "little")]
    {
      self
    }

    #[cfg(target_endian = "big")]
    {
      let out: i64x2;
      asm!(
        "xxbrd {out}, {inp}",
        out = lateout(vreg) out,
        inp = in(vreg) self.0,
        options(nomem, nostack, pure)
      );
      Self(out)
    }
  }

  #[inline]
  fn bitrev_bytes(self) -> Self {
    Self::new(bitrev_bytes_u64(self.high_64()), bitrev_bytes_u64(self.low_64()))
  }

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn vpmsumd(a: i64x2, b: i64x2) -> i64x2 {
    let out: i64x2;
    asm!(
      "vpmsumd {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
    out
  }

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn mul64(a: u64, b: u64) -> Self {
    let va = i64x2::from_array([a as i64, 0]);
    let vb = i64x2::from_array([b as i64, 0]);
    Self(Self::vpmsumd(va, vb))
  }

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn fold_16(self, coeff: Self) -> Self {
    Self(Self::vpmsumd(self.0, coeff.swap_lanes().0))
  }

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn fold_16_reflected(self, coeff: Self, data_to_xor: Self) -> Self {
    data_to_xor ^ self.fold_16(coeff)
  }

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
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

  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn barrett_width32_reflected(self, poly: u64, mu: u64) -> u32 {
    let t1 = Self::mul64(self.low_64(), mu);
    let l = Self::mul64(t1.low_64(), poly);
    (self ^ l).high_64() as u32
  }
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
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
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn update_simd_width32_reflected_bitrev(
  state: u32,
  first: &[Simd; 8],
  rest: &[[Simd; 8]],
  keys: &[u64; 23],
) -> u32 {
  let mut x = *first;
  x[0] = x[0].to_le().bitrev_bytes();
  x[1] = x[1].to_le().bitrev_bytes();
  x[2] = x[2].to_le().bitrev_bytes();
  x[3] = x[3].to_le().bitrev_bytes();
  x[4] = x[4].to_le().bitrev_bytes();
  x[5] = x[5].to_le().bitrev_bytes();
  x[6] = x[6].to_le().bitrev_bytes();
  x[7] = x[7].to_le().bitrev_bytes();

  x[0] ^= Simd::new(0, state as u64);

  let coeff_128b = Simd::new(keys[4], keys[3]);
  for chunk in rest {
    x[0] = x[0].fold_16_reflected(coeff_128b, chunk[0].to_le().bitrev_bytes());
    x[1] = x[1].fold_16_reflected(coeff_128b, chunk[1].to_le().bitrev_bytes());
    x[2] = x[2].fold_16_reflected(coeff_128b, chunk[2].to_le().bitrev_bytes());
    x[3] = x[3].fold_16_reflected(coeff_128b, chunk[3].to_le().bitrev_bytes());
    x[4] = x[4].fold_16_reflected(coeff_128b, chunk[4].to_le().bitrev_bytes());
    x[5] = x[5].fold_16_reflected(coeff_128b, chunk[5].to_le().bitrev_bytes());
    x[6] = x[6].fold_16_reflected(coeff_128b, chunk[6].to_le().bitrev_bytes());
    x[7] = x[7].fold_16_reflected(coeff_128b, chunk[7].to_le().bitrev_bytes());
  }

  finalize_lanes_width32_reflected(x, keys)
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn fold_block_128_reflected_bitrev(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff: Simd) {
  x[0] = x[0].fold_16_reflected(coeff, chunk[0].to_le().bitrev_bytes());
  x[1] = x[1].fold_16_reflected(coeff, chunk[1].to_le().bitrev_bytes());
  x[2] = x[2].fold_16_reflected(coeff, chunk[2].to_le().bitrev_bytes());
  x[3] = x[3].fold_16_reflected(coeff, chunk[3].to_le().bitrev_bytes());
  x[4] = x[4].fold_16_reflected(coeff, chunk[4].to_le().bitrev_bytes());
  x[5] = x[5].fold_16_reflected(coeff, chunk[5].to_le().bitrev_bytes());
  x[6] = x[6].fold_16_reflected(coeff, chunk[6].to_le().bitrev_bytes());
  x[7] = x[7].fold_16_reflected(coeff, chunk[7].to_le().bitrev_bytes());
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn normalize_block_le_bitrev(mut block: [Simd; 8]) -> [Simd; 8] {
  block[0] = block[0].to_le().bitrev_bytes();
  block[1] = block[1].to_le().bitrev_bytes();
  block[2] = block[2].to_le().bitrev_bytes();
  block[3] = block[3].to_le().bitrev_bytes();
  block[4] = block[4].to_le().bitrev_bytes();
  block[5] = block[5].to_le().bitrev_bytes();
  block[6] = block[6].to_le().bitrev_bytes();
  block[7] = block[7].to_le().bitrev_bytes();
  block
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn update_simd_width32_reflected_bitrev_2way(
  state: u32,
  blocks: &[[Simd; 8]],
  fold_256b: (u64, u64),
  keys: &[u64; 23],
) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 2 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected_bitrev(state, first, rest, keys);
  }

  let coeff_256 = Simd::new(fold_256b.0, fold_256b.1);
  let coeff_128 = Simd::new(keys[4], keys[3]);

  let mut s0 = normalize_block_le_bitrev(blocks[0]);
  let mut s1 = normalize_block_le_bitrev(blocks[1]);

  s0[0] ^= Simd::new(0, state as u64);

  let mut i: usize = 2;
  let even = blocks.len() & !1usize;
  while i < even {
    fold_block_128_reflected_bitrev(&mut s0, &blocks[i], coeff_256);
    fold_block_128_reflected_bitrev(&mut s1, &blocks[i.strict_add(1)], coeff_256);
    i = i.strict_add(2);
  }

  // Merge: A·s0 ⊕ s1 (A = shift by 128B).
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
    fold_block_128_reflected_bitrev(&mut combined, &blocks[even], coeff_128);
  }

  finalize_lanes_width32_reflected(combined, keys)
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn update_simd_width32_reflected_bitrev_4way(
  state: u32,
  blocks: &[[Simd; 8]],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  keys: &[u64; 23],
) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 4 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected_bitrev(state, first, rest, keys);
  }

  let aligned = (blocks.len() / 4) * 4;

  let coeff_512 = Simd::new(fold_512b.0, fold_512b.1);
  let coeff_128 = Simd::new(keys[4], keys[3]);
  let c384 = Simd::new(combine[0].0, combine[0].1);
  let c256 = Simd::new(combine[1].0, combine[1].1);
  let c128 = Simd::new(combine[2].0, combine[2].1);

  let mut s0 = normalize_block_le_bitrev(blocks[0]);
  let mut s1 = normalize_block_le_bitrev(blocks[1]);
  let mut s2 = normalize_block_le_bitrev(blocks[2]);
  let mut s3 = normalize_block_le_bitrev(blocks[3]);

  s0[0] ^= Simd::new(0, state as u64);

  let mut i: usize = 4;
  while i < aligned {
    fold_block_128_reflected_bitrev(&mut s0, &blocks[i], coeff_512);
    fold_block_128_reflected_bitrev(&mut s1, &blocks[i.strict_add(1)], coeff_512);
    fold_block_128_reflected_bitrev(&mut s2, &blocks[i.strict_add(2)], coeff_512);
    fold_block_128_reflected_bitrev(&mut s3, &blocks[i.strict_add(3)], coeff_512);
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
    fold_block_128_reflected_bitrev(&mut acc, block, coeff_128);
  }

  finalize_lanes_width32_reflected(acc, keys)
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn update_simd_width32_reflected_bitrev_8way(
  state: u32,
  blocks: &[[Simd; 8]],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  keys: &[u64; 23],
) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 8 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected_bitrev(state, first, rest, keys);
  }

  let aligned = (blocks.len() / 8) * 8;

  let coeff_1024 = Simd::new(fold_1024b.0, fold_1024b.1);
  let coeff_128 = Simd::new(keys[4], keys[3]);
  let c896 = Simd::new(combine[0].0, combine[0].1);
  let c768 = Simd::new(combine[1].0, combine[1].1);
  let c640 = Simd::new(combine[2].0, combine[2].1);
  let c512 = Simd::new(combine[3].0, combine[3].1);
  let c384 = Simd::new(combine[4].0, combine[4].1);
  let c256 = Simd::new(combine[5].0, combine[5].1);
  let c128 = Simd::new(combine[6].0, combine[6].1);

  let mut s0 = normalize_block_le_bitrev(blocks[0]);
  let mut s1 = normalize_block_le_bitrev(blocks[1]);
  let mut s2 = normalize_block_le_bitrev(blocks[2]);
  let mut s3 = normalize_block_le_bitrev(blocks[3]);
  let mut s4 = normalize_block_le_bitrev(blocks[4]);
  let mut s5 = normalize_block_le_bitrev(blocks[5]);
  let mut s6 = normalize_block_le_bitrev(blocks[6]);
  let mut s7 = normalize_block_le_bitrev(blocks[7]);

  s0[0] ^= Simd::new(0, state as u64);

  let mut i: usize = 8;
  while i < aligned {
    fold_block_128_reflected_bitrev(&mut s0, &blocks[i], coeff_1024);
    fold_block_128_reflected_bitrev(&mut s1, &blocks[i.strict_add(1)], coeff_1024);
    fold_block_128_reflected_bitrev(&mut s2, &blocks[i.strict_add(2)], coeff_1024);
    fold_block_128_reflected_bitrev(&mut s3, &blocks[i.strict_add(3)], coeff_1024);
    fold_block_128_reflected_bitrev(&mut s4, &blocks[i.strict_add(4)], coeff_1024);
    fold_block_128_reflected_bitrev(&mut s5, &blocks[i.strict_add(5)], coeff_1024);
    fold_block_128_reflected_bitrev(&mut s6, &blocks[i.strict_add(6)], coeff_1024);
    fold_block_128_reflected_bitrev(&mut s7, &blocks[i.strict_add(7)], coeff_1024);
    i = i.strict_add(8);
  }

  // Merge: A^7·s0 ⊕ A^6·s1 ⊕ A^5·s2 ⊕ A^4·s3 ⊕ A^3·s4 ⊕ A^2·s5 ⊕ A·s6 ⊕ s7.
  let mut acc = s7;
  acc[0] = s6[0].fold_16_reflected(c128, acc[0]);
  acc[1] = s6[1].fold_16_reflected(c128, acc[1]);
  acc[2] = s6[2].fold_16_reflected(c128, acc[2]);
  acc[3] = s6[3].fold_16_reflected(c128, acc[3]);
  acc[4] = s6[4].fold_16_reflected(c128, acc[4]);
  acc[5] = s6[5].fold_16_reflected(c128, acc[5]);
  acc[6] = s6[6].fold_16_reflected(c128, acc[6]);
  acc[7] = s6[7].fold_16_reflected(c128, acc[7]);

  acc[0] = s5[0].fold_16_reflected(c256, acc[0]);
  acc[1] = s5[1].fold_16_reflected(c256, acc[1]);
  acc[2] = s5[2].fold_16_reflected(c256, acc[2]);
  acc[3] = s5[3].fold_16_reflected(c256, acc[3]);
  acc[4] = s5[4].fold_16_reflected(c256, acc[4]);
  acc[5] = s5[5].fold_16_reflected(c256, acc[5]);
  acc[6] = s5[6].fold_16_reflected(c256, acc[6]);
  acc[7] = s5[7].fold_16_reflected(c256, acc[7]);

  acc[0] = s4[0].fold_16_reflected(c384, acc[0]);
  acc[1] = s4[1].fold_16_reflected(c384, acc[1]);
  acc[2] = s4[2].fold_16_reflected(c384, acc[2]);
  acc[3] = s4[3].fold_16_reflected(c384, acc[3]);
  acc[4] = s4[4].fold_16_reflected(c384, acc[4]);
  acc[5] = s4[5].fold_16_reflected(c384, acc[5]);
  acc[6] = s4[6].fold_16_reflected(c384, acc[6]);
  acc[7] = s4[7].fold_16_reflected(c384, acc[7]);

  acc[0] = s3[0].fold_16_reflected(c512, acc[0]);
  acc[1] = s3[1].fold_16_reflected(c512, acc[1]);
  acc[2] = s3[2].fold_16_reflected(c512, acc[2]);
  acc[3] = s3[3].fold_16_reflected(c512, acc[3]);
  acc[4] = s3[4].fold_16_reflected(c512, acc[4]);
  acc[5] = s3[5].fold_16_reflected(c512, acc[5]);
  acc[6] = s3[6].fold_16_reflected(c512, acc[6]);
  acc[7] = s3[7].fold_16_reflected(c512, acc[7]);

  acc[0] = s2[0].fold_16_reflected(c640, acc[0]);
  acc[1] = s2[1].fold_16_reflected(c640, acc[1]);
  acc[2] = s2[2].fold_16_reflected(c640, acc[2]);
  acc[3] = s2[3].fold_16_reflected(c640, acc[3]);
  acc[4] = s2[4].fold_16_reflected(c640, acc[4]);
  acc[5] = s2[5].fold_16_reflected(c640, acc[5]);
  acc[6] = s2[6].fold_16_reflected(c640, acc[6]);
  acc[7] = s2[7].fold_16_reflected(c640, acc[7]);

  acc[0] = s1[0].fold_16_reflected(c768, acc[0]);
  acc[1] = s1[1].fold_16_reflected(c768, acc[1]);
  acc[2] = s1[2].fold_16_reflected(c768, acc[2]);
  acc[3] = s1[3].fold_16_reflected(c768, acc[3]);
  acc[4] = s1[4].fold_16_reflected(c768, acc[4]);
  acc[5] = s1[5].fold_16_reflected(c768, acc[5]);
  acc[6] = s1[6].fold_16_reflected(c768, acc[6]);
  acc[7] = s1[7].fold_16_reflected(c768, acc[7]);

  acc[0] = s0[0].fold_16_reflected(c896, acc[0]);
  acc[1] = s0[1].fold_16_reflected(c896, acc[1]);
  acc[2] = s0[2].fold_16_reflected(c896, acc[2]);
  acc[3] = s0[3].fold_16_reflected(c896, acc[3]);
  acc[4] = s0[4].fold_16_reflected(c896, acc[4]);
  acc[5] = s0[5].fold_16_reflected(c896, acc[5]);
  acc[6] = s0[6].fold_16_reflected(c896, acc[6]);
  acc[7] = s0[7].fold_16_reflected(c896, acc[7]);

  for block in &blocks[aligned..] {
    fold_block_128_reflected_bitrev(&mut acc, block, coeff_128);
  }

  finalize_lanes_width32_reflected(acc, keys)
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc24_width32_vpmsum_bitrev(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return crc24_reflected_update_bitrev_bytes(state, data);
  };

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_width32_reflected_bitrev(state, first, rest, keys);
  crc24_reflected_update_bitrev_bytes(state, right)
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc24_width32_vpmsum_bitrev_2way(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return crc24_reflected_update_bitrev_bytes(state, data);
  }

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_width32_reflected_bitrev_2way(state, middle, CRC24_OPENPGP_STREAM_REFLECTED.fold_256b, keys);
  crc24_reflected_update_bitrev_bytes(state, right)
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc24_width32_vpmsum_bitrev_4way(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return crc24_reflected_update_bitrev_bytes(state, data);
  }

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_width32_reflected_bitrev_4way(
    state,
    middle,
    CRC24_OPENPGP_STREAM_REFLECTED.fold_512b,
    &CRC24_OPENPGP_STREAM_REFLECTED.combine_4way,
    keys,
  );
  crc24_reflected_update_bitrev_bytes(state, right)
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc24_width32_vpmsum_bitrev_8way(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return crc24_reflected_update_bitrev_bytes(state, data);
  }

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_width32_reflected_bitrev_8way(
    state,
    middle,
    CRC24_OPENPGP_STREAM_REFLECTED.fold_1024b,
    &CRC24_OPENPGP_STREAM_REFLECTED.combine_8way,
    keys,
  );
  crc24_reflected_update_bitrev_bytes(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Safe Kernel
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-24/OPENPGP VPMSUMD kernel.
///
/// # Safety
///
/// Dispatcher verifies VPMSUMD before selecting this kernel.
#[inline]
pub fn crc24_openpgp_vpmsum_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies VPMSUMD before selecting this kernel.
  state = unsafe { crc24_width32_vpmsum_bitrev(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

#[inline]
pub fn crc24_openpgp_vpmsum_2way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies VPMSUMD before selecting this kernel.
  state = unsafe { crc24_width32_vpmsum_bitrev_2way(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

#[inline]
pub fn crc24_openpgp_vpmsum_4way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies VPMSUMD before selecting this kernel.
  state = unsafe { crc24_width32_vpmsum_bitrev_4way(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

#[inline]
pub fn crc24_openpgp_vpmsum_8way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies VPMSUMD before selecting this kernel.
  state = unsafe { crc24_width32_vpmsum_bitrev_8way(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}
