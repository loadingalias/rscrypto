//! Power hardware-accelerated CRC-64 kernels (XZ + NVME).
//!
//! This is a VPMSUMD implementation derived from the Intel/TiKV folding
//! algorithm (also used by `crc64fast` / `crc64fast-nvme`).
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
  ops::{BitXor, BitXorAssign},
  simd::i64x2,
};

use crate::common::{
  clmul::{Crc64ClmulConstants, fold16_coeff_for_bytes},
  tables::{CRC64_NVME_POLY, CRC64_XZ_POLY},
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

impl Simd {
  #[inline]
  fn new(high: u64, low: u64) -> Self {
    // Match the x86/aarch64 layout: lane0 = low, lane1 = high.
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

  /// Fold 16 bytes: `(coeff.low ⊗ self.low) ⊕ (coeff.high ⊗ self.high)`.
  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn fold_16(self, coeff: Self) -> Self {
    Self(Self::vpmsumd(self.0, coeff.0))
  }

  /// Fold 8 bytes: `self.high ⊕ (coeff ⊗ self.low)`.
  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn fold_8(self, coeff: u64) -> Self {
    let prod = Self::mul64(self.low_64(), coeff);
    prod ^ Self::new(0, self.high_64())
  }

  /// Barrett reduction to finalize the CRC.
  #[inline]
  #[target_feature(
    enable = "altivec",
    enable = "vsx",
    enable = "power8-vector",
    enable = "power8-crypto"
  )]
  unsafe fn barrett(self, poly: u64, mu: u64) -> u64 {
    // Same reduction scheme as the aarch64 implementation (keeps the code small
    // and avoids extra vector shuffles).
    let t1 = Self::mul64(self.low_64(), mu).low_64();
    let l = Self::mul64(t1, poly);
    (self ^ l).high_64() ^ t1
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-stream coefficients (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

// 2-way: update step shifts by 2×128B = 256B.
const XZ_FOLD_256B: (u64, u64) = fold16_coeff_for_bytes(CRC64_XZ_POLY, 256);
const NVME_FOLD_256B: (u64, u64) = fold16_coeff_for_bytes(CRC64_NVME_POLY, 256);

// 4-way: update step shifts by 4×128B = 512B, combine shifts by 3/2/1 blocks.
const XZ_FOLD_512B: (u64, u64) = fold16_coeff_for_bytes(CRC64_XZ_POLY, 512);
const NVME_FOLD_512B: (u64, u64) = fold16_coeff_for_bytes(CRC64_NVME_POLY, 512);
const XZ_COMBINE_4WAY: [(u64, u64); 3] = [
  fold16_coeff_for_bytes(CRC64_XZ_POLY, 384),
  fold16_coeff_for_bytes(CRC64_XZ_POLY, 256),
  fold16_coeff_for_bytes(CRC64_XZ_POLY, 128),
];
const NVME_COMBINE_4WAY: [(u64, u64); 3] = [
  fold16_coeff_for_bytes(CRC64_NVME_POLY, 384),
  fold16_coeff_for_bytes(CRC64_NVME_POLY, 256),
  fold16_coeff_for_bytes(CRC64_NVME_POLY, 128),
];

// 8-way: update step shifts by 8×128B = 1024B, combine shifts by 7..1 blocks.
const XZ_FOLD_1024B: (u64, u64) = fold16_coeff_for_bytes(CRC64_XZ_POLY, 1024);
const NVME_FOLD_1024B: (u64, u64) = fold16_coeff_for_bytes(CRC64_NVME_POLY, 1024);
const XZ_COMBINE_8WAY: [(u64, u64); 7] = [
  fold16_coeff_for_bytes(CRC64_XZ_POLY, 896),
  fold16_coeff_for_bytes(CRC64_XZ_POLY, 768),
  fold16_coeff_for_bytes(CRC64_XZ_POLY, 640),
  fold16_coeff_for_bytes(CRC64_XZ_POLY, 512),
  fold16_coeff_for_bytes(CRC64_XZ_POLY, 384),
  fold16_coeff_for_bytes(CRC64_XZ_POLY, 256),
  fold16_coeff_for_bytes(CRC64_XZ_POLY, 128),
];
const NVME_COMBINE_8WAY: [(u64, u64); 7] = [
  fold16_coeff_for_bytes(CRC64_NVME_POLY, 896),
  fold16_coeff_for_bytes(CRC64_NVME_POLY, 768),
  fold16_coeff_for_bytes(CRC64_NVME_POLY, 640),
  fold16_coeff_for_bytes(CRC64_NVME_POLY, 512),
  fold16_coeff_for_bytes(CRC64_NVME_POLY, 384),
  fold16_coeff_for_bytes(CRC64_NVME_POLY, 256),
  fold16_coeff_for_bytes(CRC64_NVME_POLY, 128),
];

// ─────────────────────────────────────────────────────────────────────────────
// Folding helpers
// ─────────────────────────────────────────────────────────────────────────────

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
unsafe fn update_simd(state: u64, first: &[Simd; 8], rest: &[[Simd; 8]], consts: &Crc64ClmulConstants) -> u64 {
  let mut x = *first;
  x[0] = x[0].to_le();
  x[1] = x[1].to_le();
  x[2] = x[2].to_le();
  x[3] = x[3].to_le();
  x[4] = x[4].to_le();
  x[5] = x[5].to_le();
  x[6] = x[6].to_le();
  x[7] = x[7].to_le();

  // XOR the initial CRC into the first lane.
  x[0] ^= Simd::new(0, state);

  // 128-byte folding.
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
  state: u64,
  blocks: &[[Simd; 8]],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
) -> u64 {
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
  s0[0] ^= Simd::new(0, state);

  let mut i = 2;
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

  // Handle any remaining block (odd tail) sequentially.
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
  state: u64,
  blocks: &[[Simd; 8]],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc64ClmulConstants,
) -> u64 {
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

  // Inject CRC into stream 0.
  s0[0] ^= Simd::new(0, state);

  let mut i = 4;
  while i < aligned {
    fold_block_128(&mut s0, &blocks[i], coeff_512);
    fold_block_128(&mut s1, &blocks[i + 1], coeff_512);
    fold_block_128(&mut s2, &blocks[i + 2], coeff_512);
    fold_block_128(&mut s3, &blocks[i + 3], coeff_512);
    i = i.strict_add(4);
  }

  // Merge: A^3·s0 ⊕ A^2·s1 ⊕ A·s2 ⊕ s3.
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

  for block in &blocks[aligned..] {
    fold_block_128(&mut combined, block, coeff_128);
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
  state: u64,
  blocks: &[[Simd; 8]],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  consts: &Crc64ClmulConstants,
) -> u64 {
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

  s4[0] = s4[0].to_le();
  s4[1] = s4[1].to_le();
  s4[2] = s4[2].to_le();
  s4[3] = s4[3].to_le();
  s4[4] = s4[4].to_le();
  s4[5] = s4[5].to_le();
  s4[6] = s4[6].to_le();
  s4[7] = s4[7].to_le();

  s5[0] = s5[0].to_le();
  s5[1] = s5[1].to_le();
  s5[2] = s5[2].to_le();
  s5[3] = s5[3].to_le();
  s5[4] = s5[4].to_le();
  s5[5] = s5[5].to_le();
  s5[6] = s5[6].to_le();
  s5[7] = s5[7].to_le();

  s6[0] = s6[0].to_le();
  s6[1] = s6[1].to_le();
  s6[2] = s6[2].to_le();
  s6[3] = s6[3].to_le();
  s6[4] = s6[4].to_le();
  s6[5] = s6[5].to_le();
  s6[6] = s6[6].to_le();
  s6[7] = s6[7].to_le();

  s7[0] = s7[0].to_le();
  s7[1] = s7[1].to_le();
  s7[2] = s7[2].to_le();
  s7[3] = s7[3].to_le();
  s7[4] = s7[4].to_le();
  s7[5] = s7[5].to_le();
  s7[6] = s7[6].to_le();
  s7[7] = s7[7].to_le();

  // Inject CRC into stream 0.
  s0[0] ^= Simd::new(0, state);

  let mut i = 8;
  while i < aligned {
    fold_block_128(&mut s0, &blocks[i], coeff_1024);
    fold_block_128(&mut s1, &blocks[i + 1], coeff_1024);
    fold_block_128(&mut s2, &blocks[i + 2], coeff_1024);
    fold_block_128(&mut s3, &blocks[i + 3], coeff_1024);
    fold_block_128(&mut s4, &blocks[i + 4], coeff_1024);
    fold_block_128(&mut s5, &blocks[i + 5], coeff_1024);
    fold_block_128(&mut s6, &blocks[i + 6], coeff_1024);
    fold_block_128(&mut s7, &blocks[i + 7], coeff_1024);
    i = i.strict_add(8);
  }

  // Merge: A^7·s0 ⊕ A^6·s1 ⊕ … ⊕ A·s6 ⊕ s7.
  let mut combined = s7;
  combined[0] ^= s6[0].fold_16(c128);
  combined[1] ^= s6[1].fold_16(c128);
  combined[2] ^= s6[2].fold_16(c128);
  combined[3] ^= s6[3].fold_16(c128);
  combined[4] ^= s6[4].fold_16(c128);
  combined[5] ^= s6[5].fold_16(c128);
  combined[6] ^= s6[6].fold_16(c128);
  combined[7] ^= s6[7].fold_16(c128);

  combined[0] ^= s5[0].fold_16(c256);
  combined[1] ^= s5[1].fold_16(c256);
  combined[2] ^= s5[2].fold_16(c256);
  combined[3] ^= s5[3].fold_16(c256);
  combined[4] ^= s5[4].fold_16(c256);
  combined[5] ^= s5[5].fold_16(c256);
  combined[6] ^= s5[6].fold_16(c256);
  combined[7] ^= s5[7].fold_16(c256);

  combined[0] ^= s4[0].fold_16(c384);
  combined[1] ^= s4[1].fold_16(c384);
  combined[2] ^= s4[2].fold_16(c384);
  combined[3] ^= s4[3].fold_16(c384);
  combined[4] ^= s4[4].fold_16(c384);
  combined[5] ^= s4[5].fold_16(c384);
  combined[6] ^= s4[6].fold_16(c384);
  combined[7] ^= s4[7].fold_16(c384);

  combined[0] ^= s3[0].fold_16(c512);
  combined[1] ^= s3[1].fold_16(c512);
  combined[2] ^= s3[2].fold_16(c512);
  combined[3] ^= s3[3].fold_16(c512);
  combined[4] ^= s3[4].fold_16(c512);
  combined[5] ^= s3[5].fold_16(c512);
  combined[6] ^= s3[6].fold_16(c512);
  combined[7] ^= s3[7].fold_16(c512);

  combined[0] ^= s2[0].fold_16(c640);
  combined[1] ^= s2[1].fold_16(c640);
  combined[2] ^= s2[2].fold_16(c640);
  combined[3] ^= s2[3].fold_16(c640);
  combined[4] ^= s2[4].fold_16(c640);
  combined[5] ^= s2[5].fold_16(c640);
  combined[6] ^= s2[6].fold_16(c640);
  combined[7] ^= s2[7].fold_16(c640);

  combined[0] ^= s1[0].fold_16(c768);
  combined[1] ^= s1[1].fold_16(c768);
  combined[2] ^= s1[2].fold_16(c768);
  combined[3] ^= s1[3].fold_16(c768);
  combined[4] ^= s1[4].fold_16(c768);
  combined[5] ^= s1[5].fold_16(c768);
  combined[6] ^= s1[6].fold_16(c768);
  combined[7] ^= s1[7].fold_16(c768);

  combined[0] ^= s0[0].fold_16(c896);
  combined[1] ^= s0[1].fold_16(c896);
  combined[2] ^= s0[2].fold_16(c896);
  combined[3] ^= s0[3].fold_16(c896);
  combined[4] ^= s0[4].fold_16(c896);
  combined[5] ^= s0[5].fold_16(c896);
  combined[6] ^= s0[6].fold_16(c896);
  combined[7] ^= s0[7].fold_16(c896);

  for block in &blocks[aligned..] {
    fold_block_128(&mut combined, block, coeff_128);
  }

  fold_tail(combined, consts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public kernels (XZ + NVME)
// ─────────────────────────────────────────────────────────────────────────────

#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc64_vpmsum(mut state: u64, bytes: &[u8], consts: &Crc64ClmulConstants, tables: &[[u64; 256]; 16]) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if let Some((first, rest)) = middle.split_first() {
    state = super::portable::crc64_slice16(state, left, tables);
    state = update_simd(state, first, rest, consts);
    super::portable::crc64_slice16(state, right, tables)
  } else {
    super::portable::crc64_slice16(state, bytes, tables)
  }
}

#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc64_vpmsum_2way(
  mut state: u64,
  bytes: &[u8],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 16],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc64_slice16(state, bytes, tables);
  }

  state = super::portable::crc64_slice16(state, left, tables);

  if middle.len() >= 2 {
    state = update_simd_2way(state, middle, fold_256b, consts);
  } else if let Some((first, rest)) = middle.split_first() {
    state = update_simd(state, first, rest, consts);
  }

  super::portable::crc64_slice16(state, right, tables)
}

#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc64_vpmsum_4way(
  mut state: u64,
  bytes: &[u8],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 16],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc64_slice16(state, bytes, tables);
  }

  state = super::portable::crc64_slice16(state, left, tables);
  state = update_simd_4way(state, middle, fold_512b, combine, consts);
  super::portable::crc64_slice16(state, right, tables)
}

#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
unsafe fn crc64_vpmsum_8way(
  mut state: u64,
  bytes: &[u8],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 16],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc64_slice16(state, bytes, tables);
  }

  state = super::portable::crc64_slice16(state, left, tables);
  state = update_simd_8way(state, middle, fold_1024b, combine, consts);
  super::portable::crc64_slice16(state, right, tables)
}

/// CRC-64-XZ using VPMSUMD folding.
///
/// # Safety
///
/// Requires POWER8 vector crypto. Caller must verify via
/// `platform::caps().has(platform::caps::power::VPMSUM_READY)`.
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
pub unsafe fn crc64_xz_vpmsum(crc: u64, data: &[u8]) -> u64 {
  crc64_vpmsum(
    crc,
    data,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-XZ using VPMSUMD folding (2-way ILP variant).
///
/// # Safety
///
/// Requires POWER8 vector crypto. Caller must verify via
/// `platform::caps().has(platform::caps::power::VPMSUM_READY)`.
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
pub unsafe fn crc64_xz_vpmsum_2way(crc: u64, data: &[u8]) -> u64 {
  crc64_vpmsum_2way(
    crc,
    data,
    XZ_FOLD_256B,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-XZ using VPMSUMD folding (4-way ILP variant).
///
/// # Safety
///
/// Requires POWER8 vector crypto. Caller must verify via
/// `platform::caps().has(platform::caps::power::VPMSUM_READY)`.
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
pub unsafe fn crc64_xz_vpmsum_4way(crc: u64, data: &[u8]) -> u64 {
  crc64_vpmsum_4way(
    crc,
    data,
    XZ_FOLD_512B,
    &XZ_COMBINE_4WAY,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-XZ using VPMSUMD folding (8-way ILP variant).
///
/// # Safety
///
/// Requires POWER8 vector crypto. Caller must verify via
/// `platform::caps().has(platform::caps::power::VPMSUM_READY)`.
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
pub unsafe fn crc64_xz_vpmsum_8way(crc: u64, data: &[u8]) -> u64 {
  crc64_vpmsum_8way(
    crc,
    data,
    XZ_FOLD_1024B,
    &XZ_COMBINE_8WAY,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-NVME using VPMSUMD folding.
///
/// # Safety
///
/// Requires POWER8 vector crypto. Caller must verify via
/// `platform::caps().has(platform::caps::power::VPMSUM_READY)`.
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
pub unsafe fn crc64_nvme_vpmsum(crc: u64, data: &[u8]) -> u64 {
  crc64_vpmsum(
    crc,
    data,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

/// CRC-64-NVME using VPMSUMD folding (2-way ILP variant).
///
/// # Safety
///
/// Requires POWER8 vector crypto. Caller must verify via
/// `platform::caps().has(platform::caps::power::VPMSUM_READY)`.
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
pub unsafe fn crc64_nvme_vpmsum_2way(crc: u64, data: &[u8]) -> u64 {
  crc64_vpmsum_2way(
    crc,
    data,
    NVME_FOLD_256B,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

/// CRC-64-NVME using VPMSUMD folding (4-way ILP variant).
///
/// # Safety
///
/// Requires POWER8 vector crypto. Caller must verify via
/// `platform::caps().has(platform::caps::power::VPMSUM_READY)`.
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
pub unsafe fn crc64_nvme_vpmsum_4way(crc: u64, data: &[u8]) -> u64 {
  crc64_vpmsum_4way(
    crc,
    data,
    NVME_FOLD_512B,
    &NVME_COMBINE_4WAY,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

/// CRC-64-NVME using VPMSUMD folding (8-way ILP variant).
///
/// # Safety
///
/// Requires POWER8 vector crypto. Caller must verify via
/// `platform::caps().has(platform::caps::power::VPMSUM_READY)`.
#[target_feature(
  enable = "altivec",
  enable = "vsx",
  enable = "power8-vector",
  enable = "power8-crypto"
)]
pub unsafe fn crc64_nvme_vpmsum_8way(crc: u64, data: &[u8]) -> u64 {
  crc64_vpmsum_8way(
    crc,
    data,
    NVME_FOLD_1024B,
    &NVME_COMBINE_8WAY,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe wrappers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub fn crc64_xz_vpmsum_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_vpmsum(crc, data) }
}

#[inline]
pub fn crc64_xz_vpmsum_2way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_vpmsum_2way(crc, data) }
}

#[inline]
pub fn crc64_xz_vpmsum_4way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_vpmsum_4way(crc, data) }
}

#[inline]
pub fn crc64_xz_vpmsum_8way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_vpmsum_8way(crc, data) }
}

#[inline]
pub fn crc64_nvme_vpmsum_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_vpmsum(crc, data) }
}

#[inline]
pub fn crc64_nvme_vpmsum_2way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_vpmsum_2way(crc, data) }
}

#[inline]
pub fn crc64_nvme_vpmsum_4way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_vpmsum_4way(crc, data) }
}

#[inline]
pub fn crc64_nvme_vpmsum_8way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_vpmsum_8way(crc, data) }
}
