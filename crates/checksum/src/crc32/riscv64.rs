//! riscv64 hardware-accelerated CRC-32/CRC-32C kernels (Zbc + Zvbc).
//!
//! This mirrors the CRC64 riscv64 backend structure:
//! - Scalar Zbc (`clmul`/`clmulh`) folding (1/2/4-way)
//! - Vector Zvbc (RVV carryless multiply) folding (1/2/4-way)
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

use super::clmul::{Crc32ClmulConstants, Crc32StreamConstants};

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

  // ─────────────────────────────────────────────────────────────────────────
  // Zbc carryless multiply primitives
  // ─────────────────────────────────────────────────────────────────────────

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
}

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
    i = i.strict_add(1);
  }

  out
}

// ─────────────────────────────────────────────────────────────────────────────
// Zbc folding helpers (scalar)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn fold_16_zbc(x: Simd, coeff: (u64, u64)) -> Simd {
  let (coeff_high, coeff_low) = coeff;
  // Reflected CRC32 fold primitive is cross-term: low×high ⊕ high×low.
  Simd::mul64(x.low_64(), coeff_high) ^ Simd::mul64(x.high_64(), coeff_low)
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn fold_width_crc32_reflected_zbc(x: Simd, high: u64, low: u64) -> Simd {
  let clmul = Simd::mul64(x.low_64(), low);
  let shifted = Simd::new(0, x.high_64());
  let mut state = clmul ^ shifted;

  let masked = Simd::new(state.high_64(), state.low_64() & 0xFFFF_FFFF_0000_0000);
  let shifted_high = (state.low_64() & 0xFFFF_FFFF).strict_shl(32);
  let clmul = Simd::mul64(shifted_high, high);
  state = clmul ^ masked;

  state
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn barrett_crc32_reflected_zbc(x: Simd, poly: u64, mu: u64) -> u32 {
  let t1 = Simd::mul64(x.low_64(), mu);
  let l = Simd::mul64(t1.low_64(), poly);
  (x ^ l).high_64() as u32
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn fold_tail_zbc(x: [Simd; 8], consts: &Crc32ClmulConstants) -> u32 {
  let mut acc = x[7];
  acc ^= fold_16_zbc(x[0], consts.tail_fold_16b[0]);
  acc ^= fold_16_zbc(x[1], consts.tail_fold_16b[1]);
  acc ^= fold_16_zbc(x[2], consts.tail_fold_16b[2]);
  acc ^= fold_16_zbc(x[3], consts.tail_fold_16b[3]);
  acc ^= fold_16_zbc(x[4], consts.tail_fold_16b[4]);
  acc ^= fold_16_zbc(x[5], consts.tail_fold_16b[5]);
  acc ^= fold_16_zbc(x[6], consts.tail_fold_16b[6]);

  let (fold_width_high, fold_width_low) = consts.fold_width;
  barrett_crc32_reflected_zbc(
    fold_width_crc32_reflected_zbc(acc, fold_width_high, fold_width_low),
    consts.poly,
    consts.mu,
  )
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn fold_block_128_zbc(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff: (u64, u64)) {
  x[0] = chunk[0] ^ fold_16_zbc(x[0], coeff);
  x[1] = chunk[1] ^ fold_16_zbc(x[1], coeff);
  x[2] = chunk[2] ^ fold_16_zbc(x[2], coeff);
  x[3] = chunk[3] ^ fold_16_zbc(x[3], coeff);
  x[4] = chunk[4] ^ fold_16_zbc(x[4], coeff);
  x[5] = chunk[5] ^ fold_16_zbc(x[5], coeff);
  x[6] = chunk[6] ^ fold_16_zbc(x[6], coeff);
  x[7] = chunk[7] ^ fold_16_zbc(x[7], coeff);
}

#[target_feature(enable = "zbc")]
unsafe fn update_simd_zbc(state: u32, first: &Block, rest: &[Block], consts: &Crc32ClmulConstants) -> u32 {
  let mut x = load_block(first);
  x[0].lo ^= state as u64;

  let coeff = consts.fold_128b;
  for block in rest {
    let chunk = load_block(block);
    fold_block_128_zbc(&mut x, &chunk, coeff);
  }

  fold_tail_zbc(x, consts)
}

#[target_feature(enable = "zbc")]
unsafe fn update_simd_zbc_2way(
  state: u32,
  blocks: &[Block],
  fold_256b: (u64, u64),
  consts: &Crc32ClmulConstants,
) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 2 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_zbc(state, first, rest, consts);
  }

  let even = blocks.len() & !1usize;

  let coeff_256 = fold_256b;
  let coeff_128 = consts.fold_128b;

  let mut s0 = load_block(&blocks[0]);
  let mut s1 = load_block(&blocks[1]);
  s0[0].lo ^= state as u64;

  let mut i = 2usize;
  while i < even {
    let b0 = load_block(&blocks[i]);
    let b1 = load_block(&blocks[i + 1]);
    fold_block_128_zbc(&mut s0, &b0, coeff_256);
    fold_block_128_zbc(&mut s1, &b1, coeff_256);
    i = i.strict_add(2);
  }

  let mut combined = s1;
  combined[0] ^= fold_16_zbc(s0[0], coeff_128);
  combined[1] ^= fold_16_zbc(s0[1], coeff_128);
  combined[2] ^= fold_16_zbc(s0[2], coeff_128);
  combined[3] ^= fold_16_zbc(s0[3], coeff_128);
  combined[4] ^= fold_16_zbc(s0[4], coeff_128);
  combined[5] ^= fold_16_zbc(s0[5], coeff_128);
  combined[6] ^= fold_16_zbc(s0[6], coeff_128);
  combined[7] ^= fold_16_zbc(s0[7], coeff_128);

  if even != blocks.len() {
    let tail = load_block(&blocks[even]);
    fold_block_128_zbc(&mut combined, &tail, coeff_128);
  }

  fold_tail_zbc(combined, consts)
}

#[target_feature(enable = "zbc")]
unsafe fn update_simd_zbc_4way(
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
    return update_simd_zbc(state, first, rest, consts);
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
  s0[0].lo ^= state as u64;

  let mut i = 4usize;
  while i < aligned {
    let b0 = load_block(&blocks[i]);
    let b1 = load_block(&blocks[i + 1]);
    let b2 = load_block(&blocks[i + 2]);
    let b3 = load_block(&blocks[i + 3]);
    fold_block_128_zbc(&mut s0, &b0, coeff_512);
    fold_block_128_zbc(&mut s1, &b1, coeff_512);
    fold_block_128_zbc(&mut s2, &b2, coeff_512);
    fold_block_128_zbc(&mut s3, &b3, coeff_512);
    i = i.strict_add(4);
  }

  let mut combined = s3;
  combined[0] ^= fold_16_zbc(s2[0], c128);
  combined[1] ^= fold_16_zbc(s2[1], c128);
  combined[2] ^= fold_16_zbc(s2[2], c128);
  combined[3] ^= fold_16_zbc(s2[3], c128);
  combined[4] ^= fold_16_zbc(s2[4], c128);
  combined[5] ^= fold_16_zbc(s2[5], c128);
  combined[6] ^= fold_16_zbc(s2[6], c128);
  combined[7] ^= fold_16_zbc(s2[7], c128);

  combined[0] ^= fold_16_zbc(s1[0], c256);
  combined[1] ^= fold_16_zbc(s1[1], c256);
  combined[2] ^= fold_16_zbc(s1[2], c256);
  combined[3] ^= fold_16_zbc(s1[3], c256);
  combined[4] ^= fold_16_zbc(s1[4], c256);
  combined[5] ^= fold_16_zbc(s1[5], c256);
  combined[6] ^= fold_16_zbc(s1[6], c256);
  combined[7] ^= fold_16_zbc(s1[7], c256);

  combined[0] ^= fold_16_zbc(s0[0], c384);
  combined[1] ^= fold_16_zbc(s0[1], c384);
  combined[2] ^= fold_16_zbc(s0[2], c384);
  combined[3] ^= fold_16_zbc(s0[3], c384);
  combined[4] ^= fold_16_zbc(s0[4], c384);
  combined[5] ^= fold_16_zbc(s0[5], c384);
  combined[6] ^= fold_16_zbc(s0[6], c384);
  combined[7] ^= fold_16_zbc(s0[7], c384);

  if aligned != blocks.len() {
    let tail_blocks = &blocks[aligned..];
    let Some((first, rest)) = tail_blocks.split_first() else {
      return fold_tail_zbc(combined, consts);
    };
    let mut x = combined;
    let first = load_block(first);
    fold_block_128_zbc(&mut x, &first, coeff_128);
    for b in rest {
      let chunk = load_block(b);
      fold_block_128_zbc(&mut x, &chunk, coeff_128);
    }
    return fold_tail_zbc(x, consts);
  }

  fold_tail_zbc(combined, consts)
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
    i = i.strict_add(1);
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
  mul64_zvbc(x.low_64(), coeff_high) ^ mul64_zvbc(x.high_64(), coeff_low)
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn fold_width_crc32_reflected_zvbc(x: Simd, high: u64, low: u64) -> Simd {
  let clmul = mul64_zvbc(x.low_64(), low);
  let shifted = Simd::new(0, x.high_64());
  let mut state = clmul ^ shifted;

  let masked = Simd::new(state.high_64(), state.low_64() & 0xFFFF_FFFF_0000_0000);
  let shifted_high = (state.low_64() & 0xFFFF_FFFF).strict_shl(32);
  let clmul = mul64_zvbc(shifted_high, high);
  state = clmul ^ masked;

  state
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn barrett_crc32_reflected_zvbc(x: Simd, poly: u64, mu: u64) -> u32 {
  let t1 = mul64_zvbc(x.low_64(), mu);
  let l = mul64_zvbc(t1.low_64(), poly);
  (x ^ l).high_64() as u32
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn fold_tail_zvbc(hi: [u64; 8], lo: [u64; 8], consts: &Crc32ClmulConstants) -> u32 {
  let mut acc = Simd::new(hi[7], lo[7]);
  acc ^= fold_16_zvbc(Simd::new(hi[0], lo[0]), consts.tail_fold_16b[0]);
  acc ^= fold_16_zvbc(Simd::new(hi[1], lo[1]), consts.tail_fold_16b[1]);
  acc ^= fold_16_zvbc(Simd::new(hi[2], lo[2]), consts.tail_fold_16b[2]);
  acc ^= fold_16_zvbc(Simd::new(hi[3], lo[3]), consts.tail_fold_16b[3]);
  acc ^= fold_16_zvbc(Simd::new(hi[4], lo[4]), consts.tail_fold_16b[4]);
  acc ^= fold_16_zvbc(Simd::new(hi[5], lo[5]), consts.tail_fold_16b[5]);
  acc ^= fold_16_zvbc(Simd::new(hi[6], lo[6]), consts.tail_fold_16b[6]);

  let (fold_width_high, fold_width_low) = consts.fold_width;
  barrett_crc32_reflected_zvbc(
    fold_width_crc32_reflected_zvbc(acc, fold_width_high, fold_width_low),
    consts.poly,
    consts.mu,
  )
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
      // Cross-term fold: (x_lo ⊗ coeff_high) ⊕ (x_hi ⊗ coeff_low)
      "vclmul.vx v2, v0, {chi}",
      "vclmulh.vx v3, v0, {chi}",
      "vclmul.vx v4, v1, {clo}",
      "vclmulh.vx v5, v1, {clo}",
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
unsafe fn update_simd_zvbc(state: u32, first: &Block, rest: &[Block], consts: &Crc32ClmulConstants) -> u32 {
  let (mut x_hi, mut x_lo) = load_block_split(first);
  x_lo[0] ^= state as u64;

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
  state: u32,
  blocks: &[Block],
  fold_256b: (u64, u64),
  consts: &Crc32ClmulConstants,
) -> u32 {
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
  s0_lo[0] ^= state as u64;

  let mut i = 2usize;
  while i < even {
    let (b0_hi, b0_lo) = load_block_split(&blocks[i]);
    let (b1_hi, b1_lo) = load_block_split(&blocks[i + 1]);
    fold_block_128_zvbc(&mut s0_hi, &mut s0_lo, &b0_hi, &b0_lo, coeff_256_low, coeff_256_high);
    fold_block_128_zvbc(&mut s1_hi, &mut s1_lo, &b1_hi, &b1_lo, coeff_256_low, coeff_256_high);
    i = i.strict_add(2);
  }

  // Merge streams: A·s0 ⊕ s1 (A = shift by 128B). Use the same trick as CRC64:
  // fold each stream into the current combined value using `chunk = combined`.
  let mut combined_hi = s1_hi;
  let mut combined_lo = s1_lo;

  fold_block_128_zvbc(
    &mut s0_hi,
    &mut s0_lo,
    &combined_hi,
    &combined_lo,
    coeff_128_low,
    coeff_128_high,
  );
  combined_hi = s0_hi;
  combined_lo = s0_lo;

  if even != blocks.len() {
    let (tail_hi, tail_lo) = load_block_split(&blocks[even]);
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

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn update_simd_zvbc_4way(
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
  s0_lo[0] ^= state as u64;

  let mut i = 4usize;
  while i < aligned {
    let (b0_hi, b0_lo) = load_block_split(&blocks[i]);
    let (b1_hi, b1_lo) = load_block_split(&blocks[i + 1]);
    let (b2_hi, b2_lo) = load_block_split(&blocks[i + 2]);
    let (b3_hi, b3_lo) = load_block_split(&blocks[i + 3]);
    fold_block_128_zvbc(&mut s0_hi, &mut s0_lo, &b0_hi, &b0_lo, coeff_512_low, coeff_512_high);
    fold_block_128_zvbc(&mut s1_hi, &mut s1_lo, &b1_hi, &b1_lo, coeff_512_low, coeff_512_high);
    fold_block_128_zvbc(&mut s2_hi, &mut s2_lo, &b2_hi, &b2_lo, coeff_512_low, coeff_512_high);
    fold_block_128_zvbc(&mut s3_hi, &mut s3_lo, &b3_hi, &b3_lo, coeff_512_low, coeff_512_high);
    i = i.strict_add(4);
  }

  // Merge: A^3·s0 ⊕ A^2·s1 ⊕ A·s2 ⊕ s3.
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
// Public kernels (IEEE + CRC32C)
// ─────────────────────────────────────────────────────────────────────────────

#[target_feature(enable = "zbc")]
unsafe fn crc32_zbc(mut state: u32, bytes: &[u8], consts: &Crc32ClmulConstants) -> u32 {
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc32_slice16_ieee(state, left);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    // SAFETY: `blocks_u64` length is a multiple of 16, so casting to `[u64; 16]` is safe.
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    if let Some((first, rest)) = blocks.split_first() {
      // 1-way update (stream selection happens in the dispatcher).
      state = update_simd_zbc(state, first, rest, consts);
    }
  }

  if !tail_u64.is_empty() {
    // SAFETY: `tail_u64` is a subslice of the aligned u64 middle region.
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc32_slice16_ieee(state, tail_bytes);
  }

  super::portable::crc32_slice16_ieee(state, right)
}

#[target_feature(enable = "zbc")]
unsafe fn crc32_zbc_nway<const N: usize>(
  mut state: u32,
  bytes: &[u8],
  consts: &Crc32ClmulConstants,
  stream: &Crc32StreamConstants,
) -> u32 {
  debug_assert!(N == 2 || N == 4);
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc32_slice16_ieee(state, left);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    state = match N {
      2 => update_simd_zbc_2way(state, blocks, stream.fold_256b, consts),
      _ => update_simd_zbc_4way(state, blocks, stream.fold_512b, &stream.combine_4way, consts),
    };
  }

  if !tail_u64.is_empty() {
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc32_slice16_ieee(state, tail_bytes);
  }

  super::portable::crc32_slice16_ieee(state, right)
}

#[target_feature(enable = "zbc")]
unsafe fn crc32c_zbc(mut state: u32, bytes: &[u8], consts: &Crc32ClmulConstants) -> u32 {
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc32c_slice16(state, left);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    if let Some((first, rest)) = blocks.split_first() {
      state = update_simd_zbc(state, first, rest, consts);
    }
  }

  if !tail_u64.is_empty() {
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc32c_slice16(state, tail_bytes);
  }

  super::portable::crc32c_slice16(state, right)
}

#[target_feature(enable = "zbc")]
unsafe fn crc32c_zbc_nway<const N: usize>(
  mut state: u32,
  bytes: &[u8],
  consts: &Crc32ClmulConstants,
  stream: &Crc32StreamConstants,
) -> u32 {
  debug_assert!(N == 2 || N == 4);
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc32c_slice16(state, left);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    state = match N {
      2 => update_simd_zbc_2way(state, blocks, stream.fold_256b, consts),
      _ => update_simd_zbc_4way(state, blocks, stream.fold_512b, &stream.combine_4way, consts),
    };
  }

  if !tail_u64.is_empty() {
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc32c_slice16(state, tail_bytes);
  }

  super::portable::crc32c_slice16(state, right)
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn crc32_zvbc(mut state: u32, bytes: &[u8], consts: &Crc32ClmulConstants) -> u32 {
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc32_slice16_ieee(state, left);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    if let Some((first, rest)) = blocks.split_first() {
      state = update_simd_zvbc(state, first, rest, consts);
    }
  }

  if !tail_u64.is_empty() {
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc32_slice16_ieee(state, tail_bytes);
  }

  super::portable::crc32_slice16_ieee(state, right)
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn crc32_zvbc_nway<const N: usize>(
  mut state: u32,
  bytes: &[u8],
  consts: &Crc32ClmulConstants,
  stream: &Crc32StreamConstants,
) -> u32 {
  debug_assert!(N == 2 || N == 4);
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc32_slice16_ieee(state, left);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    state = match N {
      2 => update_simd_zvbc_2way(state, blocks, stream.fold_256b, consts),
      _ => update_simd_zvbc_4way(state, blocks, stream.fold_512b, &stream.combine_4way, consts),
    };
  }

  if !tail_u64.is_empty() {
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc32_slice16_ieee(state, tail_bytes);
  }

  super::portable::crc32_slice16_ieee(state, right)
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn crc32c_zvbc(mut state: u32, bytes: &[u8], consts: &Crc32ClmulConstants) -> u32 {
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc32c_slice16(state, left);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    if let Some((first, rest)) = blocks.split_first() {
      state = update_simd_zvbc(state, first, rest, consts);
    }
  }

  if !tail_u64.is_empty() {
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc32c_slice16(state, tail_bytes);
  }

  super::portable::crc32c_slice16(state, right)
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn crc32c_zvbc_nway<const N: usize>(
  mut state: u32,
  bytes: &[u8],
  consts: &Crc32ClmulConstants,
  stream: &Crc32StreamConstants,
) -> u32 {
  debug_assert!(N == 2 || N == 4);
  let (left, middle, right) = bytes.align_to::<u64>();

  state = super::portable::crc32c_slice16(state, left);

  let block_u64s = middle.len() & !15usize;
  let (blocks_u64, tail_u64) = middle.split_at(block_u64s);

  if !blocks_u64.is_empty() {
    let blocks: &[Block] = unsafe { core::slice::from_raw_parts(blocks_u64.as_ptr().cast(), blocks_u64.len() / 16) };
    state = match N {
      2 => update_simd_zvbc_2way(state, blocks, stream.fold_256b, consts),
      _ => update_simd_zvbc_4way(state, blocks, stream.fold_512b, &stream.combine_4way, consts),
    };
  }

  if !tail_u64.is_empty() {
    let tail_bytes = unsafe { core::slice::from_raw_parts(tail_u64.as_ptr().cast(), tail_u64.len() * 8) };
    state = super::portable::crc32c_slice16(state, tail_bytes);
  }

  super::portable::crc32c_slice16(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe wrappers (dispatcher entrypoints)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub fn crc32_ieee_zbc_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies ZBC before selecting this kernel.
  unsafe { crc32_zbc(crc, data, &super::clmul::CRC32_IEEE_CLMUL) }
}

#[inline]
pub fn crc32_ieee_zbc_2way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe {
    crc32_zbc_nway::<2>(
      crc,
      data,
      &super::clmul::CRC32_IEEE_CLMUL,
      &super::clmul::CRC32_IEEE_STREAM,
    )
  }
}

#[inline]
pub fn crc32_ieee_zbc_4way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe {
    crc32_zbc_nway::<4>(
      crc,
      data,
      &super::clmul::CRC32_IEEE_CLMUL,
      &super::clmul::CRC32_IEEE_STREAM,
    )
  }
}

#[inline]
pub fn crc32_ieee_zvbc_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies V+ZVBC before selecting this kernel.
  unsafe { crc32_zvbc(crc, data, &super::clmul::CRC32_IEEE_CLMUL) }
}

#[inline]
pub fn crc32_ieee_zvbc_2way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe {
    crc32_zvbc_nway::<2>(
      crc,
      data,
      &super::clmul::CRC32_IEEE_CLMUL,
      &super::clmul::CRC32_IEEE_STREAM,
    )
  }
}

#[inline]
pub fn crc32_ieee_zvbc_4way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe {
    crc32_zvbc_nway::<4>(
      crc,
      data,
      &super::clmul::CRC32_IEEE_CLMUL,
      &super::clmul::CRC32_IEEE_STREAM,
    )
  }
}

#[inline]
pub fn crc32c_zbc_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe { crc32c_zbc(crc, data, &super::clmul::CRC32C_CLMUL) }
}

#[inline]
pub fn crc32c_zbc_2way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe { crc32c_zbc_nway::<2>(crc, data, &super::clmul::CRC32C_CLMUL, &super::clmul::CRC32C_STREAM) }
}

#[inline]
pub fn crc32c_zbc_4way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe { crc32c_zbc_nway::<4>(crc, data, &super::clmul::CRC32C_CLMUL, &super::clmul::CRC32C_STREAM) }
}

#[inline]
pub fn crc32c_zvbc_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe { crc32c_zvbc(crc, data, &super::clmul::CRC32C_CLMUL) }
}

#[inline]
pub fn crc32c_zvbc_2way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe { crc32c_zvbc_nway::<2>(crc, data, &super::clmul::CRC32C_CLMUL, &super::clmul::CRC32C_STREAM) }
}

#[inline]
pub fn crc32c_zvbc_4way_safe(crc: u32, data: &[u8]) -> u32 {
  unsafe { crc32c_zvbc_nway::<4>(crc, data, &super::clmul::CRC32C_CLMUL, &super::clmul::CRC32C_STREAM) }
}
