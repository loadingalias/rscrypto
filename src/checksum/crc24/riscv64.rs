//! riscv64 hardware-accelerated CRC-24/OPENPGP kernels (Zbc + Zvbc).
//!
//! CRC-24/OPENPGP is MSB-first. These kernels reuse the reflected width32
//! folding/reduction structure by processing per-byte bit-reversed input and
//! converting the CRC state between OpenPGP and reflected forms.
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
  mem::MaybeUninit,
  ops::{BitXor, BitXorAssign},
};

use super::{
  keys::{CRC24_OPENPGP_KEYS_REFLECTED, CRC24_OPENPGP_STREAM_REFLECTED},
  reflected::{crc24_reflected_update_bitrev_bytes, from_reflected_state, to_reflected_state},
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

#[inline]
#[must_use]
const fn bitrev_bytes_u64(mut x: u64) -> u64 {
  x = ((x & 0x5555_5555_5555_5555).strict_shl(1)) | ((x.strict_shr(1)) & 0x5555_5555_5555_5555);
  x = ((x & 0x3333_3333_3333_3333).strict_shl(2)) | ((x.strict_shr(2)) & 0x3333_3333_3333_3333);
  x = ((x & 0x0F0F_0F0F_0F0F_0F0F).strict_shl(4)) | ((x.strict_shr(4)) & 0x0F0F_0F0F_0F0F_0F0F);
  x
}

// ─────────────────────────────────────────────────────────────────────────────
// Load helpers (bit-reverse each byte)
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn load_block_bitrev(block: &Block) -> [Simd; 8] {
  let mut out = MaybeUninit::<[Simd; 8]>::uninit();
  let base = out.as_mut_ptr().cast::<Simd>();

  let mut i = 0usize;
  while i < 8 {
    let lo = bitrev_bytes_u64(u64::from_le(block[i * 2]));
    let hi = bitrev_bytes_u64(u64::from_le(block[i * 2 + 1]));
    // SAFETY: `base` points to a `[Simd; 8]` buffer and `i` is in-bounds.
    unsafe {
      base.add(i).write(Simd::new(hi, lo));
    }
    i = i.strict_add(1);
  }

  // SAFETY: all 8 elements are initialized above.
  unsafe { out.assume_init() }
}

#[inline]
fn load_block_split_bitrev(block: &Block) -> ([u64; 8], [u64; 8]) {
  let mut hi = [0u64; 8];
  let mut lo = [0u64; 8];

  let mut i = 0usize;
  while i < 8 {
    lo[i] = bitrev_bytes_u64(u64::from_le(block[i * 2]));
    hi[i] = bitrev_bytes_u64(u64::from_le(block[i * 2 + 1]));
    i = i.strict_add(1);
  }

  (hi, lo)
}

// ─────────────────────────────────────────────────────────────────────────────
// ZBC (scalar carryless multiply) backend
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn fold_16_zbc(x: Simd, coeff: (u64, u64)) -> Simd {
  let (coeff_high, coeff_low) = coeff;
  Simd::mul64(x.low_64(), coeff_high) ^ Simd::mul64(x.high_64(), coeff_low)
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn fold_16_reflected_zbc(x: Simd, coeff: (u64, u64), data_to_xor: Simd) -> Simd {
  data_to_xor ^ fold_16_zbc(x, coeff)
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn fold_width32_reflected_zbc(x: Simd, high: u64, low: u64) -> Simd {
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
unsafe fn barrett_width32_reflected_zbc(x: Simd, poly: u64, mu: u64) -> u32 {
  let t1 = Simd::mul64(x.low_64(), mu);
  let l = Simd::mul64(t1.low_64(), poly);
  (x ^ l).high_64() as u32
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn finalize_lanes_width32_reflected_zbc(x: [Simd; 8], keys: &[u64; 23]) -> u32 {
  let mut res = x[7];
  res = fold_16_reflected_zbc(x[0], (keys[10], keys[9]), res);
  res = fold_16_reflected_zbc(x[1], (keys[12], keys[11]), res);
  res = fold_16_reflected_zbc(x[2], (keys[14], keys[13]), res);
  res = fold_16_reflected_zbc(x[3], (keys[16], keys[15]), res);
  res = fold_16_reflected_zbc(x[4], (keys[18], keys[17]), res);
  res = fold_16_reflected_zbc(x[5], (keys[20], keys[19]), res);
  res = fold_16_reflected_zbc(x[6], (keys[2], keys[1]), res);

  barrett_width32_reflected_zbc(fold_width32_reflected_zbc(res, keys[6], keys[5]), keys[8], keys[7])
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn update_simd_zbc(state: u32, first: &Block, rest: &[Block], keys: &[u64; 23]) -> u32 {
  let mut x = load_block_bitrev(first);
  x[0] ^= Simd::new(0, state as u64);

  let coeff_128b = (keys[4], keys[3]);
  for block in rest {
    let chunk = load_block_bitrev(block);
    x[0] = fold_16_reflected_zbc(x[0], coeff_128b, chunk[0]);
    x[1] = fold_16_reflected_zbc(x[1], coeff_128b, chunk[1]);
    x[2] = fold_16_reflected_zbc(x[2], coeff_128b, chunk[2]);
    x[3] = fold_16_reflected_zbc(x[3], coeff_128b, chunk[3]);
    x[4] = fold_16_reflected_zbc(x[4], coeff_128b, chunk[4]);
    x[5] = fold_16_reflected_zbc(x[5], coeff_128b, chunk[5]);
    x[6] = fold_16_reflected_zbc(x[6], coeff_128b, chunk[6]);
    x[7] = fold_16_reflected_zbc(x[7], coeff_128b, chunk[7]);
  }

  finalize_lanes_width32_reflected_zbc(x, keys)
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn fold_block_128_reflected_zbc(x: &mut [Simd; 8], block: &Block, coeff: (u64, u64)) {
  let chunk = load_block_bitrev(block);
  x[0] = fold_16_reflected_zbc(x[0], coeff, chunk[0]);
  x[1] = fold_16_reflected_zbc(x[1], coeff, chunk[1]);
  x[2] = fold_16_reflected_zbc(x[2], coeff, chunk[2]);
  x[3] = fold_16_reflected_zbc(x[3], coeff, chunk[3]);
  x[4] = fold_16_reflected_zbc(x[4], coeff, chunk[4]);
  x[5] = fold_16_reflected_zbc(x[5], coeff, chunk[5]);
  x[6] = fold_16_reflected_zbc(x[6], coeff, chunk[6]);
  x[7] = fold_16_reflected_zbc(x[7], coeff, chunk[7]);
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn update_simd_zbc_2way(state: u32, blocks: &[Block], fold_256b: (u64, u64), keys: &[u64; 23]) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 2 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_zbc(state, first, rest, keys);
  }

  let even = blocks.len() & !1usize;
  let coeff_256 = fold_256b;
  let coeff_128 = (keys[4], keys[3]);

  let mut s0 = load_block_bitrev(&blocks[0]);
  let mut s1 = load_block_bitrev(&blocks[1]);

  s0[0] ^= Simd::new(0, state as u64);

  let mut i: usize = 2;
  while i < even {
    fold_block_128_reflected_zbc(&mut s0, &blocks[i], coeff_256);
    fold_block_128_reflected_zbc(&mut s1, &blocks[i.strict_add(1)], coeff_256);
    i = i.strict_add(2);
  }

  // Merge: A·s0 ⊕ s1 (A = shift by 128B).
  s1[0] = fold_16_reflected_zbc(s0[0], coeff_128, s1[0]);
  s1[1] = fold_16_reflected_zbc(s0[1], coeff_128, s1[1]);
  s1[2] = fold_16_reflected_zbc(s0[2], coeff_128, s1[2]);
  s1[3] = fold_16_reflected_zbc(s0[3], coeff_128, s1[3]);
  s1[4] = fold_16_reflected_zbc(s0[4], coeff_128, s1[4]);
  s1[5] = fold_16_reflected_zbc(s0[5], coeff_128, s1[5]);
  s1[6] = fold_16_reflected_zbc(s0[6], coeff_128, s1[6]);
  s1[7] = fold_16_reflected_zbc(s0[7], coeff_128, s1[7]);

  if even != blocks.len() {
    fold_block_128_reflected_zbc(&mut s1, &blocks[even], coeff_128);
  }

  finalize_lanes_width32_reflected_zbc(s1, keys)
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn update_simd_zbc_4way(
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
    return update_simd_zbc(state, first, rest, keys);
  }

  let aligned = (blocks.len() / 4) * 4;

  let coeff_512 = fold_512b;
  let coeff_128 = (keys[4], keys[3]);
  let c384 = combine[0];
  let c256 = combine[1];
  let c128 = combine[2];

  let mut s0 = load_block_bitrev(&blocks[0]);
  let mut s1 = load_block_bitrev(&blocks[1]);
  let mut s2 = load_block_bitrev(&blocks[2]);
  let mut s3 = load_block_bitrev(&blocks[3]);

  s0[0] ^= Simd::new(0, state as u64);

  let mut i: usize = 4;
  while i < aligned {
    fold_block_128_reflected_zbc(&mut s0, &blocks[i], coeff_512);
    fold_block_128_reflected_zbc(&mut s1, &blocks[i.strict_add(1)], coeff_512);
    fold_block_128_reflected_zbc(&mut s2, &blocks[i.strict_add(2)], coeff_512);
    fold_block_128_reflected_zbc(&mut s3, &blocks[i.strict_add(3)], coeff_512);
    i = i.strict_add(4);
  }

  // Merge: A^3·s0 ⊕ A^2·s1 ⊕ A·s2 ⊕ s3.
  s3[0] = fold_16_reflected_zbc(s2[0], c128, s3[0]);
  s3[1] = fold_16_reflected_zbc(s2[1], c128, s3[1]);
  s3[2] = fold_16_reflected_zbc(s2[2], c128, s3[2]);
  s3[3] = fold_16_reflected_zbc(s2[3], c128, s3[3]);
  s3[4] = fold_16_reflected_zbc(s2[4], c128, s3[4]);
  s3[5] = fold_16_reflected_zbc(s2[5], c128, s3[5]);
  s3[6] = fold_16_reflected_zbc(s2[6], c128, s3[6]);
  s3[7] = fold_16_reflected_zbc(s2[7], c128, s3[7]);

  s3[0] = fold_16_reflected_zbc(s1[0], c256, s3[0]);
  s3[1] = fold_16_reflected_zbc(s1[1], c256, s3[1]);
  s3[2] = fold_16_reflected_zbc(s1[2], c256, s3[2]);
  s3[3] = fold_16_reflected_zbc(s1[3], c256, s3[3]);
  s3[4] = fold_16_reflected_zbc(s1[4], c256, s3[4]);
  s3[5] = fold_16_reflected_zbc(s1[5], c256, s3[5]);
  s3[6] = fold_16_reflected_zbc(s1[6], c256, s3[6]);
  s3[7] = fold_16_reflected_zbc(s1[7], c256, s3[7]);

  s3[0] = fold_16_reflected_zbc(s0[0], c384, s3[0]);
  s3[1] = fold_16_reflected_zbc(s0[1], c384, s3[1]);
  s3[2] = fold_16_reflected_zbc(s0[2], c384, s3[2]);
  s3[3] = fold_16_reflected_zbc(s0[3], c384, s3[3]);
  s3[4] = fold_16_reflected_zbc(s0[4], c384, s3[4]);
  s3[5] = fold_16_reflected_zbc(s0[5], c384, s3[5]);
  s3[6] = fold_16_reflected_zbc(s0[6], c384, s3[6]);
  s3[7] = fold_16_reflected_zbc(s0[7], c384, s3[7]);

  for block in &blocks[aligned..] {
    fold_block_128_reflected_zbc(&mut s3, block, coeff_128);
  }

  finalize_lanes_width32_reflected_zbc(s3, keys)
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn crc24_width32_zbc(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<Block>();
  let Some((first, rest)) = middle.split_first() else {
    return crc24_reflected_update_bitrev_bytes(state, data);
  };

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_zbc(state, first, rest, keys);
  crc24_reflected_update_bitrev_bytes(state, right)
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn crc24_width32_zbc_2way(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<Block>();
  if middle.is_empty() {
    return crc24_reflected_update_bitrev_bytes(state, data);
  }

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_zbc_2way(state, middle, CRC24_OPENPGP_STREAM_REFLECTED.fold_256b, keys);
  crc24_reflected_update_bitrev_bytes(state, right)
}

#[inline]
#[target_feature(enable = "zbc")]
unsafe fn crc24_width32_zbc_4way(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<Block>();
  if middle.is_empty() {
    return crc24_reflected_update_bitrev_bytes(state, data);
  }

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_zbc_4way(
    state,
    middle,
    CRC24_OPENPGP_STREAM_REFLECTED.fold_512b,
    &CRC24_OPENPGP_STREAM_REFLECTED.combine_4way,
    keys,
  );
  crc24_reflected_update_bitrev_bytes(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// ZVBC (vector carryless multiply) backend
// ─────────────────────────────────────────────────────────────────────────────

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
unsafe fn fold_width32_reflected_zvbc(x: Simd, high: u64, low: u64) -> Simd {
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
unsafe fn barrett_width32_reflected_zvbc(x: Simd, poly: u64, mu: u64) -> u32 {
  let t1 = mul64_zvbc(x.low_64(), mu);
  let l = mul64_zvbc(t1.low_64(), poly);
  (x ^ l).high_64() as u32
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn fold_tail_zvbc(hi: [u64; 8], lo: [u64; 8], keys: &[u64; 23]) -> u32 {
  let mut acc = Simd::new(hi[7], lo[7]);
  acc ^= fold_16_zvbc(Simd::new(hi[0], lo[0]), (keys[10], keys[9]));
  acc ^= fold_16_zvbc(Simd::new(hi[1], lo[1]), (keys[12], keys[11]));
  acc ^= fold_16_zvbc(Simd::new(hi[2], lo[2]), (keys[14], keys[13]));
  acc ^= fold_16_zvbc(Simd::new(hi[3], lo[3]), (keys[16], keys[15]));
  acc ^= fold_16_zvbc(Simd::new(hi[4], lo[4]), (keys[18], keys[17]));
  acc ^= fold_16_zvbc(Simd::new(hi[5], lo[5]), (keys[20], keys[19]));
  acc ^= fold_16_zvbc(Simd::new(hi[6], lo[6]), (keys[2], keys[1]));

  barrett_width32_reflected_zvbc(fold_width32_reflected_zvbc(acc, keys[6], keys[5]), keys[8], keys[7])
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
unsafe fn update_simd_zvbc(state: u32, first: &Block, rest: &[Block], keys: &[u64; 23]) -> u32 {
  let (mut x_hi, mut x_lo) = load_block_split_bitrev(first);
  x_lo[0] ^= state as u64;

  let coeff_low = keys[3];
  let coeff_high = keys[4];

  for block in rest {
    let (chunk_hi, chunk_lo) = load_block_split_bitrev(block);
    fold_block_128_zvbc(&mut x_hi, &mut x_lo, &chunk_hi, &chunk_lo, coeff_low, coeff_high);
  }

  fold_tail_zvbc(x_hi, x_lo, keys)
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn update_simd_zvbc_2way(state: u32, blocks: &[Block], fold_256b: (u64, u64), keys: &[u64; 23]) -> u32 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 2 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_zvbc(state, first, rest, keys);
  }

  let even = blocks.len() & !1usize;

  let coeff_256_low = fold_256b.1;
  let coeff_256_high = fold_256b.0;
  let coeff_128_low = keys[3];
  let coeff_128_high = keys[4];

  let (mut s0_hi, mut s0_lo) = load_block_split_bitrev(&blocks[0]);
  let (mut s1_hi, mut s1_lo) = load_block_split_bitrev(&blocks[1]);

  // Inject CRC into stream 0.
  s0_lo[0] ^= state as u64;

  let mut i: usize = 2;
  while i < even {
    let (b0_hi, b0_lo) = load_block_split_bitrev(&blocks[i]);
    let (b1_hi, b1_lo) = load_block_split_bitrev(&blocks[i + 1]);
    fold_block_128_zvbc(&mut s0_hi, &mut s0_lo, &b0_hi, &b0_lo, coeff_256_low, coeff_256_high);
    fold_block_128_zvbc(&mut s1_hi, &mut s1_lo, &b1_hi, &b1_lo, coeff_256_low, coeff_256_high);
    i = i.strict_add(2);
  }

  // Merge: A·s0 ⊕ s1 (A = shift by 128B).
  fold_block_128_zvbc(&mut s0_hi, &mut s0_lo, &s1_hi, &s1_lo, coeff_128_low, coeff_128_high);

  if even != blocks.len() {
    let (tail_hi, tail_lo) = load_block_split_bitrev(&blocks[even]);
    fold_block_128_zvbc(
      &mut s0_hi,
      &mut s0_lo,
      &tail_hi,
      &tail_lo,
      coeff_128_low,
      coeff_128_high,
    );
  }

  fold_tail_zvbc(s0_hi, s0_lo, keys)
}

#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn update_simd_zvbc_4way(
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
    return update_simd_zvbc(state, first, rest, keys);
  }

  let aligned = (blocks.len() / 4) * 4;

  let coeff_512_low = fold_512b.1;
  let coeff_512_high = fold_512b.0;
  let coeff_128_low = keys[3];
  let coeff_128_high = keys[4];

  let c384_low = combine[0].1;
  let c384_high = combine[0].0;
  let c256_low = combine[1].1;
  let c256_high = combine[1].0;
  let c128_low = combine[2].1;
  let c128_high = combine[2].0;

  let (mut s0_hi, mut s0_lo) = load_block_split_bitrev(&blocks[0]);
  let (mut s1_hi, mut s1_lo) = load_block_split_bitrev(&blocks[1]);
  let (mut s2_hi, mut s2_lo) = load_block_split_bitrev(&blocks[2]);
  let (mut s3_hi, mut s3_lo) = load_block_split_bitrev(&blocks[3]);

  // Inject CRC into stream 0.
  s0_lo[0] ^= state as u64;

  let mut i: usize = 4;
  while i < aligned {
    let (b0_hi, b0_lo) = load_block_split_bitrev(&blocks[i]);
    let (b1_hi, b1_lo) = load_block_split_bitrev(&blocks[i + 1]);
    let (b2_hi, b2_lo) = load_block_split_bitrev(&blocks[i + 2]);
    let (b3_hi, b3_lo) = load_block_split_bitrev(&blocks[i + 3]);
    fold_block_128_zvbc(&mut s0_hi, &mut s0_lo, &b0_hi, &b0_lo, coeff_512_low, coeff_512_high);
    fold_block_128_zvbc(&mut s1_hi, &mut s1_lo, &b1_hi, &b1_lo, coeff_512_low, coeff_512_high);
    fold_block_128_zvbc(&mut s2_hi, &mut s2_lo, &b2_hi, &b2_lo, coeff_512_low, coeff_512_high);
    fold_block_128_zvbc(&mut s3_hi, &mut s3_lo, &b3_hi, &b3_lo, coeff_512_low, coeff_512_high);
    i = i.strict_add(4);
  }

  // Merge: A^3·s0 ⊕ A^2·s1 ⊕ A·s2 ⊕ s3.
  fold_block_128_zvbc(&mut s2_hi, &mut s2_lo, &s3_hi, &s3_lo, c128_low, c128_high);
  fold_block_128_zvbc(&mut s1_hi, &mut s1_lo, &s2_hi, &s2_lo, c256_low, c256_high);
  fold_block_128_zvbc(&mut s0_hi, &mut s0_lo, &s1_hi, &s1_lo, c384_low, c384_high);

  for block in &blocks[aligned..] {
    let (tail_hi, tail_lo) = load_block_split_bitrev(block);
    fold_block_128_zvbc(
      &mut s0_hi,
      &mut s0_lo,
      &tail_hi,
      &tail_lo,
      coeff_128_low,
      coeff_128_high,
    );
  }

  fold_tail_zvbc(s0_hi, s0_lo, keys)
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn crc24_width32_zvbc(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<Block>();
  let Some((first, rest)) = middle.split_first() else {
    return crc24_reflected_update_bitrev_bytes(state, data);
  };

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_zvbc(state, first, rest, keys);
  crc24_reflected_update_bitrev_bytes(state, right)
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn crc24_width32_zvbc_2way(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<Block>();
  if middle.is_empty() {
    return crc24_reflected_update_bitrev_bytes(state, data);
  }

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_zvbc_2way(state, middle, CRC24_OPENPGP_STREAM_REFLECTED.fold_256b, keys);
  crc24_reflected_update_bitrev_bytes(state, right)
}

#[inline]
#[target_feature(enable = "v", enable = "zvbc")]
unsafe fn crc24_width32_zvbc_4way(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<Block>();
  if middle.is_empty() {
    return crc24_reflected_update_bitrev_bytes(state, data);
  }

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_zvbc_4way(
    state,
    middle,
    CRC24_OPENPGP_STREAM_REFLECTED.fold_512b,
    &CRC24_OPENPGP_STREAM_REFLECTED.combine_4way,
    keys,
  );
  crc24_reflected_update_bitrev_bytes(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Safe Kernels
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-24/OPENPGP Zbc kernel.
///
/// # Safety
///
/// Dispatcher verifies Zbc before selecting this kernel.
#[inline]
pub fn crc24_openpgp_zbc_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies Zbc before selecting this kernel.
  state = unsafe { crc24_width32_zbc(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

#[inline]
pub fn crc24_openpgp_zbc_2way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies Zbc before selecting this kernel.
  state = unsafe { crc24_width32_zbc_2way(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

#[inline]
pub fn crc24_openpgp_zbc_4way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies Zbc before selecting this kernel.
  state = unsafe { crc24_width32_zbc_4way(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP Zvbc kernel.
///
/// # Safety
///
/// Dispatcher verifies Zvbc before selecting this kernel.
#[inline]
pub fn crc24_openpgp_zvbc_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies Zvbc before selecting this kernel.
  state = unsafe { crc24_width32_zvbc(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

#[inline]
pub fn crc24_openpgp_zvbc_2way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies Zvbc before selecting this kernel.
  state = unsafe { crc24_width32_zvbc_2way(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

#[inline]
pub fn crc24_openpgp_zvbc_4way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies Zvbc before selecting this kernel.
  state = unsafe { crc24_width32_zvbc_4way(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}
