//! powerpc64 hardware-accelerated CRC-32/CRC-32C kernels (VPMSUMD-style).
//!
//! This is a VPMSUMD implementation of the 128-byte folding algorithm used by
//! our reflected CRC32 CLMUL backends.
//!
//! # Safety
//!
//! Uses `unsafe` for powerpc64 SIMD + inline assembly. Callers must ensure the
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

use super::clmul::{Crc32ClmulConstants, Crc32StreamConstants};

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
  ///
  /// On `powerpc64le` this is a no-op. On big-endian `powerpc64`, we byte-swap
  /// each 64-bit lane so the folding algorithm sees the same lane values as on
  /// little-endian platforms.
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

  /// Fold 16 bytes (reflected CRC32 folding primitive):
  /// `self.low ⊗ coeff.high ⊕ self.high ⊗ coeff.low`.
  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn fold_16(self, coeff: Self) -> Self {
    // VPMSUMD computes: lane0(self)*lane0(coeff) ⊕ lane1(self)*lane1(coeff).
    // The CRC32 folding primitive wants cross terms, so swap coefficient lanes.
    Self(Self::vpmsumd(self.0, coeff.swap_lanes().0))
  }

  /// Fold 16B → CRC32 width (reflected), returning an intermediate 128-bit state.
  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn fold_width_crc32_reflected(self, high: u64, low: u64) -> Self {
    // Stage 1: 16B -> 8B (fold high 64 into low 64).
    let clmul = Self::mul64(self.low_64(), low);
    let shifted = Self::new(0, self.high_64());
    let mut state = clmul ^ shifted;

    // Stage 2: 8B -> 4B (fold top 32 bits of low64).
    let mask2 = Self::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_0000_0000);
    let masked = state & mask2;
    let shifted_high = (state.low_64() & 0xFFFF_FFFF).strict_shl(32);
    let clmul = Self::mul64(shifted_high, high);
    state = clmul ^ masked;

    state
  }

  /// Barrett reduction for reflected CRC32; returns the updated (pre-inverted) CRC.
  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn barrett_crc32_reflected(self, poly: u64, mu: u64) -> u32 {
    // Mirror the x86 reduction scheme (2 multiplies + xor, extract hi32).
    let t1 = Self::mul64(self.low_64(), mu);
    let l = Self::mul64(t1.low_64(), poly);
    (self ^ l).high_64() as u32
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Folding helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
unsafe fn fold_tail(x: [Simd; 8], consts: &Crc32ClmulConstants) -> u32 {
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
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn fold_block_128(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff: Simd) {
  x[0] = chunk[0].to_le() ^ x[0].fold_16(coeff);
  x[1] = chunk[1].to_le() ^ x[1].fold_16(coeff);
  x[2] = chunk[2].to_le() ^ x[2].fold_16(coeff);
  x[3] = chunk[3].to_le() ^ x[3].fold_16(coeff);
  x[4] = chunk[4].to_le() ^ x[4].fold_16(coeff);
  x[5] = chunk[5].to_le() ^ x[5].fold_16(coeff);
  x[6] = chunk[6].to_le() ^ x[6].fold_16(coeff);
  x[7] = chunk[7].to_le() ^ x[7].fold_16(coeff);
}

#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn update_simd(state: u32, first: &[Simd; 8], rest: &[[Simd; 8]], consts: &Crc32ClmulConstants) -> u32 {
  let mut x = *first;
  x[0] = x[0].to_le();
  x[1] = x[1].to_le();
  x[2] = x[2].to_le();
  x[3] = x[3].to_le();
  x[4] = x[4].to_le();
  x[5] = x[5].to_le();
  x[6] = x[6].to_le();
  x[7] = x[7].to_le();

  // XOR initial CRC into the first 16-byte lane (low 32 bits).
  x[0] ^= Simd::new(0, state as u64);

  let coeff = Simd::new(consts.fold_128b.0, consts.fold_128b.1);
  for chunk in rest {
    fold_block_128(&mut x, chunk, coeff);
  }

  fold_tail(x, consts)
}

#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn update_simd_2way(
  state: u32,
  blocks: &[[Simd; 8]],
  fold_256b: (u64, u64),
  consts: &Crc32ClmulConstants,
) -> u32 {
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

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  s0[0] = s0[0].to_le();
  s0[1] = s0[1].to_le();
  s0[2] = s0[2].to_le();
  s0[3] = s0[3].to_le();
  s0[4] = s0[4].to_le();
  s0[5] = s0[5].to_le();
  s0[6] = s0[6].to_le();
  s0[7] = s0[7].to_le();

  s1[0] = s1[0].to_le();
  s1[1] = s1[1].to_le();
  s1[2] = s1[2].to_le();
  s1[3] = s1[3].to_le();
  s1[4] = s1[4].to_le();
  s1[5] = s1[5].to_le();
  s1[6] = s1[6].to_le();
  s1[7] = s1[7].to_le();

  // Inject CRC into stream 0.
  s0[0] ^= Simd::new(0, state as u64);

  let mut i = 2usize;
  while i < even {
    fold_block_128(&mut s0, &blocks[i], coeff_256);
    fold_block_128(&mut s1, &blocks[i + 1], coeff_256);
    i = i.strict_add(2);
  }

  // Merge streams: A·s0 ⊕ s1 (A = shift by 128B).
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
    fold_block_128(&mut combined, &blocks[even], coeff_128);
  }

  fold_tail(combined, consts)
}

#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn update_simd_4way(
  state: u32,
  blocks: &[[Simd; 8]],
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

  let aligned = (blocks.len() / 4) * 4;

  let coeff_512 = Simd::new(fold_512b.0, fold_512b.1);
  let coeff_128 = Simd::new(consts.fold_128b.0, consts.fold_128b.1);
  let c384 = Simd::new(combine[0].0, combine[0].1);
  let c256 = Simd::new(combine[1].0, combine[1].1);
  let c128 = Simd::new(combine[2].0, combine[2].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];

  s0[0] = s0[0].to_le();
  s0[1] = s0[1].to_le();
  s0[2] = s0[2].to_le();
  s0[3] = s0[3].to_le();
  s0[4] = s0[4].to_le();
  s0[5] = s0[5].to_le();
  s0[6] = s0[6].to_le();
  s0[7] = s0[7].to_le();

  s1[0] = s1[0].to_le();
  s1[1] = s1[1].to_le();
  s1[2] = s1[2].to_le();
  s1[3] = s1[3].to_le();
  s1[4] = s1[4].to_le();
  s1[5] = s1[5].to_le();
  s1[6] = s1[6].to_le();
  s1[7] = s1[7].to_le();

  s2[0] = s2[0].to_le();
  s2[1] = s2[1].to_le();
  s2[2] = s2[2].to_le();
  s2[3] = s2[3].to_le();
  s2[4] = s2[4].to_le();
  s2[5] = s2[5].to_le();
  s2[6] = s2[6].to_le();
  s2[7] = s2[7].to_le();

  s3[0] = s3[0].to_le();
  s3[1] = s3[1].to_le();
  s3[2] = s3[2].to_le();
  s3[3] = s3[3].to_le();
  s3[4] = s3[4].to_le();
  s3[5] = s3[5].to_le();
  s3[6] = s3[6].to_le();
  s3[7] = s3[7].to_le();

  s0[0] ^= Simd::new(0, state as u64);

  let mut i = 4usize;
  while i < aligned {
    fold_block_128(&mut s0, &blocks[i], coeff_512);
    fold_block_128(&mut s1, &blocks[i + 1], coeff_512);
    fold_block_128(&mut s2, &blocks[i + 2], coeff_512);
    fold_block_128(&mut s3, &blocks[i + 3], coeff_512);
    i = i.strict_add(4);
  }

  // Combine: (((s0·A^3 ⊕ s1·A^2) ⊕ s2·A) ⊕ s3), where A = shift by 128B.
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
    let tail = &blocks[aligned..];
    let Some((first, rest)) = tail.split_first() else {
      return fold_tail(combined, consts);
    };
    // Process remaining blocks sequentially as 1-way, continuing from `combined`.
    let mut x = combined;
    fold_block_128(&mut x, first, coeff_128);
    for b in rest {
      fold_block_128(&mut x, b, coeff_128);
    }
    return fold_tail(x, consts);
  }

  fold_tail(combined, consts)
}

#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn update_simd_8way(
  state: u32,
  blocks: &[[Simd; 8]],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  consts: &Crc32ClmulConstants,
) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 8 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd(state, first, rest, consts);
  }

  let aligned = (blocks.len() / 8) * 8;

  let coeff_1024 = Simd::new(fold_1024b.0, fold_1024b.1);
  let coeff_128 = Simd::new(consts.fold_128b.0, consts.fold_128b.1);

  let c896 = Simd::new(combine[0].0, combine[0].1);
  let c768 = Simd::new(combine[1].0, combine[1].1);
  let c640 = Simd::new(combine[2].0, combine[2].1);
  let c512 = Simd::new(combine[3].0, combine[3].1);
  let c384 = Simd::new(combine[4].0, combine[4].1);
  let c256 = Simd::new(combine[5].0, combine[5].1);
  let c128 = Simd::new(combine[6].0, combine[6].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];
  let mut s4 = blocks[4];
  let mut s5 = blocks[5];
  let mut s6 = blocks[6];
  let mut s7 = blocks[7];

  let mut i = 0usize;
  while i < 8 {
    s0[i] = s0[i].to_le();
    s1[i] = s1[i].to_le();
    s2[i] = s2[i].to_le();
    s3[i] = s3[i].to_le();
    s4[i] = s4[i].to_le();
    s5[i] = s5[i].to_le();
    s6[i] = s6[i].to_le();
    s7[i] = s7[i].to_le();
    i = i.strict_add(1);
  }

  s0[0] ^= Simd::new(0, state as u64);

  let mut idx = 8usize;
  while idx < aligned {
    fold_block_128(&mut s0, &blocks[idx], coeff_1024);
    fold_block_128(&mut s1, &blocks[idx + 1], coeff_1024);
    fold_block_128(&mut s2, &blocks[idx + 2], coeff_1024);
    fold_block_128(&mut s3, &blocks[idx + 3], coeff_1024);
    fold_block_128(&mut s4, &blocks[idx + 4], coeff_1024);
    fold_block_128(&mut s5, &blocks[idx + 5], coeff_1024);
    fold_block_128(&mut s6, &blocks[idx + 6], coeff_1024);
    fold_block_128(&mut s7, &blocks[idx + 7], coeff_1024);
    idx = idx.strict_add(8);
  }

  // Combine streams into s7 (highest index): s0..s6 shifted by 7..1 blocks.
  let mut combined = s7;
  let mut j = 0usize;
  while j < 8 {
    combined[j] ^= s6[j].fold_16(c128);
    combined[j] ^= s5[j].fold_16(c256);
    combined[j] ^= s4[j].fold_16(c384);
    combined[j] ^= s3[j].fold_16(c512);
    combined[j] ^= s2[j].fold_16(c640);
    combined[j] ^= s1[j].fold_16(c768);
    combined[j] ^= s0[j].fold_16(c896);
    j = j.strict_add(1);
  }

  if aligned != blocks.len() {
    let tail = &blocks[aligned..];
    let Some((first, rest)) = tail.split_first() else {
      return fold_tail(combined, consts);
    };
    let mut x = combined;
    fold_block_128(&mut x, first, coeff_128);
    for b in rest {
      fold_block_128(&mut x, b, coeff_128);
    }
    return fold_tail(x, consts);
  }

  fold_tail(combined, consts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public kernels (IEEE + CRC32C)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc32_fold_kernel(crc: u32, data: &[u8], consts: &Crc32ClmulConstants) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return super::portable::crc32_slice16_ieee(crc, data);
  };

  let mut state = crc;
  // Left tail can be any length; process with portable for correctness.
  state = super::portable::crc32_slice16_ieee(state, left);
  state = update_simd(state, first, rest, consts);
  super::portable::crc32_slice16_ieee(state, right)
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc32c_fold_kernel(crc: u32, data: &[u8], consts: &Crc32ClmulConstants) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return super::portable::crc32c_slice16(crc, data);
  };

  let mut state = crc;
  state = super::portable::crc32c_slice16(state, left);
  state = update_simd(state, first, rest, consts);
  super::portable::crc32c_slice16(state, right)
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc32_fold_kernel_nway<const N: usize>(
  crc: u32,
  data: &[u8],
  stream: &Crc32StreamConstants,
  consts: &Crc32ClmulConstants,
) -> u32 {
  debug_assert!(N == 2 || N == 4 || N == 8);

  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc32_slice16_ieee(crc, data);
  }

  let mut state = super::portable::crc32_slice16_ieee(crc, left);

  state = match N {
    2 => update_simd_2way(state, middle, stream.fold_256b, consts),
    4 => update_simd_4way(state, middle, stream.fold_512b, &stream.combine_4way, consts),
    _ => update_simd_8way(state, middle, stream.fold_1024b, &stream.combine_8way, consts),
  };

  super::portable::crc32_slice16_ieee(state, right)
}

#[inline]
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc32c_fold_kernel_nway<const N: usize>(
  crc: u32,
  data: &[u8],
  stream: &Crc32StreamConstants,
  consts: &Crc32ClmulConstants,
) -> u32 {
  debug_assert!(N == 2 || N == 4 || N == 8);

  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc32c_slice16(crc, data);
  }

  let mut state = super::portable::crc32c_slice16(crc, left);

  state = match N {
    2 => update_simd_2way(state, middle, stream.fold_256b, consts),
    4 => update_simd_4way(state, middle, stream.fold_512b, &stream.combine_4way, consts),
    _ => update_simd_8way(state, middle, stream.fold_1024b, &stream.combine_8way, consts),
  };

  super::portable::crc32c_slice16(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe wrappers (dispatcher entrypoints)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub fn crc32_ieee_vpmsum_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies VPMSUM-ready CPU features before selecting this kernel.
  unsafe { crc32_fold_kernel(crc, data, &super::clmul::CRC32_IEEE_CLMUL) }
}

#[inline]
pub fn crc32_ieee_vpmsum_2way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe {
    crc32_fold_kernel_nway::<2>(
      crc,
      data,
      &super::clmul::CRC32_IEEE_STREAM,
      &super::clmul::CRC32_IEEE_CLMUL,
    )
  }
}

#[inline]
pub fn crc32_ieee_vpmsum_4way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe {
    crc32_fold_kernel_nway::<4>(
      crc,
      data,
      &super::clmul::CRC32_IEEE_STREAM,
      &super::clmul::CRC32_IEEE_CLMUL,
    )
  }
}

#[inline]
pub fn crc32_ieee_vpmsum_8way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe {
    crc32_fold_kernel_nway::<8>(
      crc,
      data,
      &super::clmul::CRC32_IEEE_STREAM,
      &super::clmul::CRC32_IEEE_CLMUL,
    )
  }
}

#[inline]
pub fn crc32c_vpmsum_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies VPMSUM-ready CPU features before selecting this kernel.
  unsafe { crc32c_fold_kernel(crc, data, &super::clmul::CRC32C_CLMUL) }
}

#[inline]
pub fn crc32c_vpmsum_2way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe { crc32c_fold_kernel_nway::<2>(crc, data, &super::clmul::CRC32C_STREAM, &super::clmul::CRC32C_CLMUL) }
}

#[inline]
pub fn crc32c_vpmsum_4way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe { crc32c_fold_kernel_nway::<4>(crc, data, &super::clmul::CRC32C_STREAM, &super::clmul::CRC32C_CLMUL) }
}

#[inline]
pub fn crc32c_vpmsum_8way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe { crc32c_fold_kernel_nway::<8>(crc, data, &super::clmul::CRC32C_STREAM, &super::clmul::CRC32C_CLMUL) }
}
