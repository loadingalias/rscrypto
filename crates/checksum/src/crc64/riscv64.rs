//! riscv64 hardware-accelerated CRC-64 kernels (XZ + NVME).
//!
//! This is a scalar Zbc (`clmul`/`clmulh`) implementation of the Intel/TiKV
//! folding algorithm (also used by `crc64fast` / `crc64fast-nvme`).
//!
//! # Safety
//!
//! Uses `unsafe` for RISC-V inline assembly. Callers must ensure the required
//! CPU features are available before executing the accelerated path (the
//! dispatcher does this).
#![allow(unsafe_code)]
#![allow(dead_code)] // Kernels wired up via dispatcher
// SAFETY: All indexing is over fixed-size arrays with in-bounds constant indices.
#![allow(clippy::indexing_slicing)]
// This module is intrinsics-heavy; keep unsafe blocks readable.
#![allow(unsafe_op_in_unsafe_fn)]

use core::{
  arch::asm,
  ops::{BitXor, BitXorAssign},
};

use crate::common::{
  clmul::{Crc64ClmulConstants, fold16_coeff_for_bytes},
  tables::{CRC64_NVME_POLY, CRC64_XZ_POLY},
};

type Block = [u64; 16]; // 128 bytes (8×16B lanes)

#[derive(Copy, Clone, Debug)]
struct Simd {
  hi: u64,
  lo: u64,
}

impl BitXor for Simd {
  type Output = Self;

  #[inline]
  fn bitxor(self, other: Self) -> Self {
    Self {
      hi: self.hi ^ other.hi,
      lo: self.lo ^ other.lo,
    }
  }
}

impl BitXorAssign for Simd {
  #[inline]
  fn bitxor_assign(&mut self, other: Self) {
    self.hi ^= other.hi;
    self.lo ^= other.lo;
  }
}

impl Simd {
  #[inline]
  const fn new(high: u64, low: u64) -> Self {
    Self { hi: high, lo: low }
  }

  #[inline]
  const fn low_64(self) -> u64 {
    self.lo
  }

  #[inline]
  const fn high_64(self) -> u64 {
    self.hi
  }

  #[inline]
  #[target_feature(enable = "zbc")]
  unsafe fn clmul_lo(a: u64, b: u64) -> u64 {
    let out: u64;
    asm!(
      "clmul {out}, {a}, {b}",
      out = lateout(reg) out,
      a = in(reg) a,
      b = in(reg) b,
      options(nomem, nostack, pure)
    );
    out
  }

  #[inline]
  #[target_feature(enable = "zbc")]
  unsafe fn clmul_hi(a: u64, b: u64) -> u64 {
    let out: u64;
    asm!(
      "clmulh {out}, {a}, {b}",
      out = lateout(reg) out,
      a = in(reg) a,
      b = in(reg) b,
      options(nomem, nostack, pure)
    );
    out
  }

  #[inline]
  #[target_feature(enable = "zbc")]
  unsafe fn mul64(a: u64, b: u64) -> Self {
    Self {
      hi: Self::clmul_hi(a, b),
      lo: Self::clmul_lo(a, b),
    }
  }

  /// Fold 16 bytes: `(coeff.low ⊗ self.low) ⊕ (coeff.high ⊗ self.high)`.
  #[inline]
  #[target_feature(enable = "zbc")]
  unsafe fn fold_16(self, coeff: (u64, u64)) -> Self {
    let (coeff_high, coeff_low) = coeff;
    Self::mul64(self.low_64(), coeff_low) ^ Self::mul64(self.high_64(), coeff_high)
  }

  /// Fold 8 bytes: `self.high ⊕ (coeff ⊗ self.low)`.
  #[inline]
  #[target_feature(enable = "zbc")]
  unsafe fn fold_8(self, coeff: u64) -> Self {
    let prod = Self::mul64(self.low_64(), coeff);
    prod ^ Self::new(0, self.high_64())
  }

  /// Barrett reduction to finalize the CRC.
  #[inline]
  #[target_feature(enable = "zbc")]
  unsafe fn barrett(self, poly: u64, mu: u64) -> u64 {
    let t1 = Self::clmul_lo(self.low_64(), mu);
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

// ─────────────────────────────────────────────────────────────────────────────
// Load helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn load_block(block: &Block) -> [Simd; 8] {
  let mut out = [Simd::new(0, 0); 8];

  let mut i = 0;
  while i < 8 {
    let low = u64::from_le(block[i * 2]);
    let high = u64::from_le(block[i * 2 + 1]);
    out[i] = Simd::new(high, low);
    i += 1;
  }

  out
}

// ─────────────────────────────────────────────────────────────────────────────
// Folding helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn fold_tail(x: [Simd; 8], consts: &Crc64ClmulConstants) -> u64 {
  let mut acc = x[7];
  acc ^= x[0].fold_16(consts.tail_fold_16b[0]);
  acc ^= x[1].fold_16(consts.tail_fold_16b[1]);
  acc ^= x[2].fold_16(consts.tail_fold_16b[2]);
  acc ^= x[3].fold_16(consts.tail_fold_16b[3]);
  acc ^= x[4].fold_16(consts.tail_fold_16b[4]);
  acc ^= x[5].fold_16(consts.tail_fold_16b[5]);
  acc ^= x[6].fold_16(consts.tail_fold_16b[6]);

  acc.fold_8(consts.fold_8b).barrett(consts.poly, consts.mu)
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn fold_block_128(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff: (u64, u64)) {
  x[0] = chunk[0] ^ x[0].fold_16(coeff);
  x[1] = chunk[1] ^ x[1].fold_16(coeff);
  x[2] = chunk[2] ^ x[2].fold_16(coeff);
  x[3] = chunk[3] ^ x[3].fold_16(coeff);
  x[4] = chunk[4] ^ x[4].fold_16(coeff);
  x[5] = chunk[5] ^ x[5].fold_16(coeff);
  x[6] = chunk[6] ^ x[6].fold_16(coeff);
  x[7] = chunk[7] ^ x[7].fold_16(coeff);
}

#[target_feature(enable = "zbc")]
unsafe fn update_simd(state: u64, first: &Block, rest: &[Block], consts: &Crc64ClmulConstants) -> u64 {
  let mut x = load_block(first);

  // XOR the initial CRC into the first lane.
  x[0] ^= Simd::new(0, state);

  let coeff = consts.fold_128b;
  for block in rest {
    let chunk = load_block(block);
    fold_block_128(&mut x, &chunk, coeff);
  }

  fold_tail(x, consts)
}

#[target_feature(enable = "zbc")]
unsafe fn update_simd_2way(state: u64, blocks: &[Block], fold_256b: (u64, u64), consts: &Crc64ClmulConstants) -> u64 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 2 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd(state, first, rest, consts);
  }

  let even = blocks.len() & !1usize;

  let coeff_256 = fold_256b;
  let coeff_128 = consts.fold_128b;

  let mut s0 = load_block(&blocks[0]);
  let mut s1 = load_block(&blocks[1]);

  // Inject CRC into stream 0.
  s0[0] ^= Simd::new(0, state);

  let mut i = 2;
  while i < even {
    let b0 = load_block(&blocks[i]);
    let b1 = load_block(&blocks[i + 1]);
    fold_block_128(&mut s0, &b0, coeff_256);
    fold_block_128(&mut s1, &b1, coeff_256);
    i += 2;
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
    let tail = load_block(&blocks[even]);
    fold_block_128(&mut combined, &tail, coeff_128);
  }

  fold_tail(combined, consts)
}

#[target_feature(enable = "zbc")]
unsafe fn update_simd_4way(
  state: u64,
  blocks: &[Block],
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

  let coeff_512 = fold_512b;
  let coeff_128 = consts.fold_128b;

  let c384 = combine[0];
  let c256 = combine[1];
  let c128 = combine[2];

  let mut s0 = load_block(&blocks[0]);
  let mut s1 = load_block(&blocks[1]);
  let mut s2 = load_block(&blocks[2]);
  let mut s3 = load_block(&blocks[3]);

  // Inject CRC into stream 0.
  s0[0] ^= Simd::new(0, state);

  let mut i = 4;
  while i < aligned {
    let b0 = load_block(&blocks[i]);
    let b1 = load_block(&blocks[i + 1]);
    let b2 = load_block(&blocks[i + 2]);
    let b3 = load_block(&blocks[i + 3]);
    fold_block_128(&mut s0, &b0, coeff_512);
    fold_block_128(&mut s1, &b1, coeff_512);
    fold_block_128(&mut s2, &b2, coeff_512);
    fold_block_128(&mut s3, &b3, coeff_512);
    i += 4;
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
    let b = load_block(block);
    fold_block_128(&mut combined, &b, coeff_128);
  }

  fold_tail(combined, consts)
}

// ─────────────────────────────────────────────────────────────────────────────
// ZVBC (vector carryless multiply) backend
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn load_block_split(block: &Block) -> ([u64; 8], [u64; 8]) {
  let mut hi = [0u64; 8];
  let mut lo = [0u64; 8];

  let mut i = 0;
  while i < 8 {
    lo[i] = u64::from_le(block[i * 2]);
    hi[i] = u64::from_le(block[i * 2 + 1]);
    i += 1;
  }

  (hi, lo)
}

/// Carryless multiply of two `u64` values using ZVBC (returns 128-bit result as `{hi, lo}`).
///
/// # Safety
///
/// Requires RISC-V `v` + `zvbc`.
#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn mul64_zvbc(a: u64, b: u64) -> Simd {
  let lo: u64;
  let hi: u64;
  asm!(
    "vsetivli zero, 1, e64, m1, ta, ma",
    "vmv.v.x v0, {a}",
    "vclmul.vx v1, v0, {b}",
    "vclmulh.vx v2, v0, {b}",
    "vmv.x.s {lo}, v1",
    "vmv.x.s {hi}, v2",
    a = in(reg) a,
    b = in(reg) b,
    lo = lateout(reg) lo,
    hi = lateout(reg) hi,
    out("v0") _,
    out("v1") _,
    out("v2") _,
    options(nostack)
  );
  Simd::new(hi, lo)
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn fold_16_zvbc(x: Simd, coeff: (u64, u64)) -> Simd {
  let (coeff_high, coeff_low) = coeff;
  mul64_zvbc(x.low_64(), coeff_low) ^ mul64_zvbc(x.high_64(), coeff_high)
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn fold_8_zvbc(x: Simd, coeff: u64) -> Simd {
  let prod = mul64_zvbc(x.low_64(), coeff);
  Simd::new(prod.high_64(), prod.low_64() ^ x.high_64())
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn barrett_zvbc(x: Simd, poly: u64, mu: u64) -> u64 {
  let t1 = mul64_zvbc(x.low_64(), mu).low_64();
  let l = mul64_zvbc(t1, poly);
  (x ^ l).high_64() ^ t1
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn fold_tail_zvbc(hi: [u64; 8], lo: [u64; 8], consts: &Crc64ClmulConstants) -> u64 {
  let mut acc = Simd::new(hi[7], lo[7]);
  acc ^= fold_16_zvbc(Simd::new(hi[0], lo[0]), consts.tail_fold_16b[0]);
  acc ^= fold_16_zvbc(Simd::new(hi[1], lo[1]), consts.tail_fold_16b[1]);
  acc ^= fold_16_zvbc(Simd::new(hi[2], lo[2]), consts.tail_fold_16b[2]);
  acc ^= fold_16_zvbc(Simd::new(hi[3], lo[3]), consts.tail_fold_16b[3]);
  acc ^= fold_16_zvbc(Simd::new(hi[4], lo[4]), consts.tail_fold_16b[4]);
  acc ^= fold_16_zvbc(Simd::new(hi[5], lo[5]), consts.tail_fold_16b[5]);
  acc ^= fold_16_zvbc(Simd::new(hi[6], lo[6]), consts.tail_fold_16b[6]);
  barrett_zvbc(fold_8_zvbc(acc, consts.fold_8b), consts.poly, consts.mu)
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn fold_block_128_zvbc(
  x_hi: &mut [u64; 8],
  x_lo: &mut [u64; 8],
  chunk_hi: &[u64; 8],
  chunk_lo: &[u64; 8],
  coeff_low: u64,
  coeff_high: u64,
) {
  let mut offset = 0usize;
  while offset < 8 {
    let remaining = 8 - offset;
    let vl: usize;
    asm!(
      "vsetvli {vl}, {avl}, e64, m1, ta, ma",
      "vle64.v v0, ({xlo})",
      "vle64.v v1, ({xhi})",
      "vclmul.vx v2, v0, {clo}",
      "vclmulh.vx v3, v0, {clo}",
      "vclmul.vx v4, v1, {chi}",
      "vclmulh.vx v5, v1, {chi}",
      "vxor.vv v2, v2, v4",
      "vxor.vv v3, v3, v5",
      "vle64.v v4, ({dlo})",
      "vle64.v v5, ({dhi})",
      "vxor.vv v2, v2, v4",
      "vxor.vv v3, v3, v5",
      "vse64.v v2, ({xlo})",
      "vse64.v v3, ({xhi})",
      vl = lateout(reg) vl,
      avl = in(reg) remaining,
      xlo = in(reg) x_lo.as_mut_ptr().add(offset),
      xhi = in(reg) x_hi.as_mut_ptr().add(offset),
      dlo = in(reg) chunk_lo.as_ptr().add(offset),
      dhi = in(reg) chunk_hi.as_ptr().add(offset),
      clo = in(reg) coeff_low,
      chi = in(reg) coeff_high,
      out("v0") _,
      out("v1") _,
      out("v2") _,
      out("v3") _,
      out("v4") _,
      out("v5") _,
      options(nostack)
    );
    offset += vl;
  }
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn update_simd_zvbc(state: u64, first: &Block, rest: &[Block], consts: &Crc64ClmulConstants) -> u64 {
  let (mut x_hi, mut x_lo) = load_block_split(first);

  // XOR the initial CRC into the first lane.
  x_lo[0] ^= state;

  let coeff_low = consts.fold_128b.1;
  let coeff_high = consts.fold_128b.0;

  for block in rest {
    let (chunk_hi, chunk_lo) = load_block_split(block);
    fold_block_128_zvbc(&mut x_hi, &mut x_lo, &chunk_hi, &chunk_lo, coeff_low, coeff_high);
  }

  fold_tail_zvbc(x_hi, x_lo, consts)
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn update_simd_zvbc_2way(
  state: u64,
  blocks: &[Block],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 2 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_zvbc(state, first, rest, consts);
  }

  let even = blocks.len() & !1usize;

  let coeff_256_low = fold_256b.1;
  let coeff_256_high = fold_256b.0;
  let coeff_128_low = consts.fold_128b.1;
  let coeff_128_high = consts.fold_128b.0;

  let (mut s0_hi, mut s0_lo) = load_block_split(&blocks[0]);
  let (mut s1_hi, mut s1_lo) = load_block_split(&blocks[1]);

  // Inject CRC into stream 0.
  s0_lo[0] ^= state;

  let mut i = 2;
  while i < even {
    let (b0_hi, b0_lo) = load_block_split(&blocks[i]);
    let (b1_hi, b1_lo) = load_block_split(&blocks[i + 1]);
    fold_block_128_zvbc(&mut s0_hi, &mut s0_lo, &b0_hi, &b0_lo, coeff_256_low, coeff_256_high);
    fold_block_128_zvbc(&mut s1_hi, &mut s1_lo, &b1_hi, &b1_lo, coeff_256_low, coeff_256_high);
    i += 2;
  }

  // Merge streams: A·s0 ⊕ s1 (A = shift by 128B).
  fold_block_128_zvbc(&mut s0_hi, &mut s0_lo, &s1_hi, &s1_lo, coeff_128_low, coeff_128_high);

  // Handle any remaining block (odd tail) sequentially.
  if even != blocks.len() {
    let (tail_hi, tail_lo) = load_block_split(&blocks[even]);
    fold_block_128_zvbc(
      &mut s0_hi,
      &mut s0_lo,
      &tail_hi,
      &tail_lo,
      coeff_128_low,
      coeff_128_high,
    );
  }

  fold_tail_zvbc(s0_hi, s0_lo, consts)
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn update_simd_zvbc_4way(
  state: u64,
  blocks: &[Block],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 4 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_zvbc(state, first, rest, consts);
  }

  let aligned = (blocks.len() / 4) * 4;

  let coeff_512_low = fold_512b.1;
  let coeff_512_high = fold_512b.0;
  let coeff_128_low = consts.fold_128b.1;
  let coeff_128_high = consts.fold_128b.0;

  let c384_low = combine[0].1;
  let c384_high = combine[0].0;
  let c256_low = combine[1].1;
  let c256_high = combine[1].0;
  let c128_low = combine[2].1;
  let c128_high = combine[2].0;

  let (mut s0_hi, mut s0_lo) = load_block_split(&blocks[0]);
  let (mut s1_hi, mut s1_lo) = load_block_split(&blocks[1]);
  let (mut s2_hi, mut s2_lo) = load_block_split(&blocks[2]);
  let (mut s3_hi, mut s3_lo) = load_block_split(&blocks[3]);

  // Inject CRC into stream 0.
  s0_lo[0] ^= state;

  let mut i = 4;
  while i < aligned {
    let (b0_hi, b0_lo) = load_block_split(&blocks[i]);
    let (b1_hi, b1_lo) = load_block_split(&blocks[i + 1]);
    let (b2_hi, b2_lo) = load_block_split(&blocks[i + 2]);
    let (b3_hi, b3_lo) = load_block_split(&blocks[i + 3]);
    fold_block_128_zvbc(&mut s0_hi, &mut s0_lo, &b0_hi, &b0_lo, coeff_512_low, coeff_512_high);
    fold_block_128_zvbc(&mut s1_hi, &mut s1_lo, &b1_hi, &b1_lo, coeff_512_low, coeff_512_high);
    fold_block_128_zvbc(&mut s2_hi, &mut s2_lo, &b2_hi, &b2_lo, coeff_512_low, coeff_512_high);
    fold_block_128_zvbc(&mut s3_hi, &mut s3_lo, &b3_hi, &b3_lo, coeff_512_low, coeff_512_high);
    i += 4;
  }

  // Merge: A^3·s0 ⊕ A^2·s1 ⊕ A·s2 ⊕ s3.
  //
  // `fold_block_128_zvbc(x, chunk, c)` computes: `x = chunk ⊕ fold(x, c)`.
  // We want: `combined = combined ⊕ fold(stream, c)`, so we fold each stream
  // into the current combined value by using `chunk = combined` and storing the
  // result back into that stream (which is no longer needed after this point).
  let mut combined_hi = s3_hi;
  let mut combined_lo = s3_lo;

  fold_block_128_zvbc(&mut s2_hi, &mut s2_lo, &combined_hi, &combined_lo, c128_low, c128_high);
  combined_hi = s2_hi;
  combined_lo = s2_lo;

  fold_block_128_zvbc(&mut s1_hi, &mut s1_lo, &combined_hi, &combined_lo, c256_low, c256_high);
  combined_hi = s1_hi;
  combined_lo = s1_lo;

  fold_block_128_zvbc(&mut s0_hi, &mut s0_lo, &combined_hi, &combined_lo, c384_low, c384_high);
  combined_hi = s0_hi;
  combined_lo = s0_lo;

  for block in &blocks[aligned..] {
    let (tail_hi, tail_lo) = load_block_split(block);
    fold_block_128_zvbc(
      &mut combined_hi,
      &mut combined_lo,
      &tail_hi,
      &tail_lo,
      coeff_128_low,
      coeff_128_high,
    );
  }

  fold_tail_zvbc(combined_hi, combined_lo, consts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public kernels (XZ + NVME)
// ─────────────────────────────────────────────────────────────────────────────

#[target_feature(enable = "zbc")]
unsafe fn crc64_zbc(mut state: u64, bytes: &[u8], consts: &Crc64ClmulConstants, tables: &[[u64; 256]; 16]) -> u64 {
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc64_slice16(state, left, tables);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    // SAFETY: `blocks_u64` length is a multiple of 16, so casting to `[u64; 16]` is safe.
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    if let Some((first, rest)) = blocks.split_first() {
      state = update_simd(state, first, rest, consts);
    }
  }

  if !tail_u64.is_empty() {
    // SAFETY: `tail_u64` is a subslice of the aligned u64 middle region.
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc64_slice16(state, tail_bytes, tables);
  }

  super::portable::crc64_slice16(state, right, tables)
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn crc64_zvbc(mut state: u64, bytes: &[u8], consts: &Crc64ClmulConstants, tables: &[[u64; 256]; 16]) -> u64 {
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc64_slice16(state, left, tables);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    // SAFETY: `blocks_u64` length is a multiple of 16, so casting to `[u64; 16]` is safe.
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    if let Some((first, rest)) = blocks.split_first() {
      state = update_simd_zvbc(state, first, rest, consts);
    }
  }

  if !tail_u64.is_empty() {
    // SAFETY: `tail_u64` is a subslice of the aligned u64 middle region.
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc64_slice16(state, tail_bytes, tables);
  }

  super::portable::crc64_slice16(state, right, tables)
}

#[target_feature(enable = "zbc")]
unsafe fn crc64_zbc_2way(
  mut state: u64,
  bytes: &[u8],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 16],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc64_slice16(state, left, tables);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    // SAFETY: `blocks_u64` length is a multiple of 16, so casting to `[u64; 16]` is safe.
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    state = update_simd_2way(state, blocks, fold_256b, consts);
  }

  if !tail_u64.is_empty() {
    // SAFETY: `tail_u64` is a subslice of the aligned u64 middle region.
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc64_slice16(state, tail_bytes, tables);
  }

  super::portable::crc64_slice16(state, right, tables)
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn crc64_zvbc_2way(
  mut state: u64,
  bytes: &[u8],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 16],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc64_slice16(state, left, tables);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    // SAFETY: `blocks_u64` length is a multiple of 16, so casting to `[u64; 16]` is safe.
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    state = update_simd_zvbc_2way(state, blocks, fold_256b, consts);
  }

  if !tail_u64.is_empty() {
    // SAFETY: `tail_u64` is a subslice of the aligned u64 middle region.
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc64_slice16(state, tail_bytes, tables);
  }

  super::portable::crc64_slice16(state, right, tables)
}

#[target_feature(enable = "zbc")]
unsafe fn crc64_zbc_4way(
  mut state: u64,
  bytes: &[u8],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 16],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc64_slice16(state, left, tables);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    // SAFETY: `blocks_u64` length is a multiple of 16, so casting to `[u64; 16]` is safe.
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    state = update_simd_4way(state, blocks, fold_512b, combine, consts);
  }

  if !tail_u64.is_empty() {
    // SAFETY: `tail_u64` is a subslice of the aligned u64 middle region.
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc64_slice16(state, tail_bytes, tables);
  }

  super::portable::crc64_slice16(state, right, tables)
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn crc64_zvbc_4way(
  mut state: u64,
  bytes: &[u8],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 16],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc64_slice16(state, left, tables);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    // SAFETY: `blocks_u64` length is a multiple of 16, so casting to `[u64; 16]` is safe.
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    state = update_simd_zvbc_4way(state, blocks, fold_512b, combine, consts);
  }

  if !tail_u64.is_empty() {
    // SAFETY: `tail_u64` is a subslice of the aligned u64 middle region.
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc64_slice16(state, tail_bytes, tables);
  }

  super::portable::crc64_slice16(state, right, tables)
}

/// CRC-64-XZ using scalar Zbc carryless multiply folding.
///
/// # Safety
///
/// Requires the RISC-V Zbc extension. Caller must verify via
/// `platform::caps().has(riscv::ZBC)`.
#[target_feature(enable = "zbc")]
pub unsafe fn crc64_xz_zbc(crc: u64, data: &[u8]) -> u64 {
  crc64_zbc(
    crc,
    data,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-XZ using RVV Zvbc (vector carryless multiply) folding.
///
/// # Safety
///
/// Requires the RISC-V vector extension and Zvbc. Caller must verify via
/// `platform::caps().has(riscv::ZVBC)`.
#[target_feature(enable = "v", enable = "zvbc")]
pub unsafe fn crc64_xz_zvbc(crc: u64, data: &[u8]) -> u64 {
  crc64_zvbc(
    crc,
    data,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-XZ using scalar Zbc carryless multiply folding (2-way ILP variant).
///
/// # Safety
///
/// Requires the RISC-V Zbc extension. Caller must verify via
/// `platform::caps().has(riscv::ZBC)`.
#[target_feature(enable = "zbc")]
pub unsafe fn crc64_xz_zbc_2way(crc: u64, data: &[u8]) -> u64 {
  crc64_zbc_2way(
    crc,
    data,
    XZ_FOLD_256B,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-XZ using RVV Zvbc (vector carryless multiply) folding (2-way ILP variant).
///
/// # Safety
///
/// Requires the RISC-V vector extension and Zvbc. Caller must verify via
/// `platform::caps().has(riscv::ZVBC)`.
#[target_feature(enable = "v", enable = "zvbc")]
pub unsafe fn crc64_xz_zvbc_2way(crc: u64, data: &[u8]) -> u64 {
  crc64_zvbc_2way(
    crc,
    data,
    XZ_FOLD_256B,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-XZ using scalar Zbc carryless multiply folding (4-way ILP variant).
///
/// # Safety
///
/// Requires the RISC-V Zbc extension. Caller must verify via
/// `platform::caps().has(riscv::ZBC)`.
#[target_feature(enable = "zbc")]
pub unsafe fn crc64_xz_zbc_4way(crc: u64, data: &[u8]) -> u64 {
  crc64_zbc_4way(
    crc,
    data,
    XZ_FOLD_512B,
    &XZ_COMBINE_4WAY,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-XZ using RVV Zvbc (vector carryless multiply) folding (4-way ILP variant).
///
/// # Safety
///
/// Requires the RISC-V vector extension and Zvbc. Caller must verify via
/// `platform::caps().has(riscv::ZVBC)`.
#[target_feature(enable = "v", enable = "zvbc")]
pub unsafe fn crc64_xz_zvbc_4way(crc: u64, data: &[u8]) -> u64 {
  crc64_zvbc_4way(
    crc,
    data,
    XZ_FOLD_512B,
    &XZ_COMBINE_4WAY,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-NVME using scalar Zbc carryless multiply folding.
///
/// # Safety
///
/// Requires the RISC-V Zbc extension. Caller must verify via
/// `platform::caps().has(riscv::ZBC)`.
#[target_feature(enable = "zbc")]
pub unsafe fn crc64_nvme_zbc(crc: u64, data: &[u8]) -> u64 {
  crc64_zbc(
    crc,
    data,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

/// CRC-64-NVME using RVV Zvbc (vector carryless multiply) folding.
///
/// # Safety
///
/// Requires the RISC-V vector extension and Zvbc. Caller must verify via
/// `platform::caps().has(riscv::ZVBC)`.
#[target_feature(enable = "v", enable = "zvbc")]
pub unsafe fn crc64_nvme_zvbc(crc: u64, data: &[u8]) -> u64 {
  crc64_zvbc(
    crc,
    data,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

/// CRC-64-NVME using scalar Zbc carryless multiply folding (2-way ILP variant).
///
/// # Safety
///
/// Requires the RISC-V Zbc extension. Caller must verify via
/// `platform::caps().has(riscv::ZBC)`.
#[target_feature(enable = "zbc")]
pub unsafe fn crc64_nvme_zbc_2way(crc: u64, data: &[u8]) -> u64 {
  crc64_zbc_2way(
    crc,
    data,
    NVME_FOLD_256B,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

/// CRC-64-NVME using RVV Zvbc (vector carryless multiply) folding (2-way ILP variant).
///
/// # Safety
///
/// Requires the RISC-V vector extension and Zvbc. Caller must verify via
/// `platform::caps().has(riscv::ZVBC)`.
#[target_feature(enable = "v", enable = "zvbc")]
pub unsafe fn crc64_nvme_zvbc_2way(crc: u64, data: &[u8]) -> u64 {
  crc64_zvbc_2way(
    crc,
    data,
    NVME_FOLD_256B,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

/// CRC-64-NVME using scalar Zbc carryless multiply folding (4-way ILP variant).
///
/// # Safety
///
/// Requires the RISC-V Zbc extension. Caller must verify via
/// `platform::caps().has(riscv::ZBC)`.
#[target_feature(enable = "zbc")]
pub unsafe fn crc64_nvme_zbc_4way(crc: u64, data: &[u8]) -> u64 {
  crc64_zbc_4way(
    crc,
    data,
    NVME_FOLD_512B,
    &NVME_COMBINE_4WAY,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

/// CRC-64-NVME using RVV Zvbc (vector carryless multiply) folding (4-way ILP variant).
///
/// # Safety
///
/// Requires the RISC-V vector extension and Zvbc. Caller must verify via
/// `platform::caps().has(riscv::ZVBC)`.
#[target_feature(enable = "v", enable = "zvbc")]
pub unsafe fn crc64_nvme_zvbc_4way(crc: u64, data: &[u8]) -> u64 {
  crc64_zvbc_4way(
    crc,
    data,
    NVME_FOLD_512B,
    &NVME_COMBINE_4WAY,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe wrappers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub fn crc64_xz_zbc_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_zbc(crc, data) }
}

#[inline]
pub fn crc64_xz_zvbc_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_zvbc(crc, data) }
}

#[inline]
pub fn crc64_xz_zbc_2way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_zbc_2way(crc, data) }
}

#[inline]
pub fn crc64_xz_zvbc_2way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_zvbc_2way(crc, data) }
}

#[inline]
pub fn crc64_xz_zbc_4way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_zbc_4way(crc, data) }
}

#[inline]
pub fn crc64_xz_zvbc_4way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_zvbc_4way(crc, data) }
}

#[inline]
pub fn crc64_nvme_zbc_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_zbc(crc, data) }
}

#[inline]
pub fn crc64_nvme_zvbc_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_zvbc(crc, data) }
}

#[inline]
pub fn crc64_nvme_zbc_2way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_zbc_2way(crc, data) }
}

#[inline]
pub fn crc64_nvme_zvbc_2way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_zvbc_2way(crc, data) }
}

#[inline]
pub fn crc64_nvme_zbc_4way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_zbc_4way(crc, data) }
}

#[inline]
pub fn crc64_nvme_zvbc_4way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_zvbc_4way(crc, data) }
}
