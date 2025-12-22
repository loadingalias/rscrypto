//! s390x hardware-accelerated CRC-64 kernels (XZ + NVME).
//!
//! This is a VGFM implementation derived from the Intel/TiKV folding
//! algorithm (also used by `crc64fast` / `crc64fast-nvme`).
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
  ops::{BitXor, BitXorAssign},
  simd::i64x2,
};

use crate::common::{
  clmul::{Crc64ClmulConstants, fold16_coeff_for_bytes},
  tables::{CRC64_NVME_POLY, CRC64_XZ_POLY},
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

  /// Fold 16 bytes: `(coeff.low ⊗ self.low) ⊕ (coeff.high ⊗ self.high)`.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn fold_16(self, coeff: Self) -> Self {
    Self(Self::vgfm(self.0, coeff.0))
  }

  /// Fold 8 bytes: `self.high ⊕ (coeff ⊗ self.low)`.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn fold_8(self, coeff: u64) -> Self {
    let prod = Self::mul64(self.low_64(), coeff);
    prod ^ Self::new(0, self.high_64())
  }

  /// Barrett reduction to finalize the CRC.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn barrett(self, poly: u64, mu: u64) -> u64 {
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

// ─────────────────────────────────────────────────────────────────────────────
// Load helpers (endianness)
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
    i += 1;
  }

  // SAFETY: all 8 elements are initialized above.
  unsafe { out.assume_init() }
}

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
#[target_feature(enable = "vector")]
unsafe fn fold_block_128(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff: Simd) {
  x[0] = chunk[0] ^ x[0].fold_16(coeff);
  x[1] = chunk[1] ^ x[1].fold_16(coeff);
  x[2] = chunk[2] ^ x[2].fold_16(coeff);
  x[3] = chunk[3] ^ x[3].fold_16(coeff);
  x[4] = chunk[4] ^ x[4].fold_16(coeff);
  x[5] = chunk[5] ^ x[5].fold_16(coeff);
  x[6] = chunk[6] ^ x[6].fold_16(coeff);
  x[7] = chunk[7] ^ x[7].fold_16(coeff);
}

#[target_feature(enable = "vector")]
unsafe fn update_simd(state: u64, first: &Block, rest: &[Block], consts: &Crc64ClmulConstants) -> u64 {
  let mut x = load_block(first);

  // XOR the initial CRC into the first lane.
  x[0] ^= Simd::new(0, state);

  // 128-byte folding.
  let coeff = Simd::new(consts.fold_128b.0, consts.fold_128b.1);
  for block in rest {
    let chunk = load_block(block);
    fold_block_128(&mut x, &chunk, coeff);
  }

  fold_tail(x, consts)
}

#[target_feature(enable = "vector")]
unsafe fn update_simd_2way(state: u64, blocks: &[Block], fold_256b: (u64, u64), consts: &Crc64ClmulConstants) -> u64 {
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

#[target_feature(enable = "vector")]
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

  let coeff_512 = Simd::new(fold_512b.0, fold_512b.1);
  let coeff_128 = Simd::new(consts.fold_128b.0, consts.fold_128b.1);
  let c384 = Simd::new(combine[0].0, combine[0].1);
  let c256 = Simd::new(combine[1].0, combine[1].1);
  let c128 = Simd::new(combine[2].0, combine[2].1);

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
// Public kernels (XZ + NVME)
// ─────────────────────────────────────────────────────────────────────────────

#[target_feature(enable = "vector")]
unsafe fn crc64_vgfm(mut state: u64, bytes: &[u8], consts: &Crc64ClmulConstants, tables: &[[u64; 256]; 16]) -> u64 {
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

#[target_feature(enable = "vector")]
unsafe fn crc64_vgfm_2way(
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

#[target_feature(enable = "vector")]
unsafe fn crc64_vgfm_4way(
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

/// CRC-64-XZ using VGFM folding.
///
/// # Safety
///
/// Requires the s390x vector facility. Caller must verify via
/// `platform::caps().has(s390x::VECTOR)`.
#[target_feature(enable = "vector")]
pub unsafe fn crc64_xz_vgfm(crc: u64, data: &[u8]) -> u64 {
  crc64_vgfm(
    crc,
    data,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-XZ using VGFM folding (2-way ILP variant).
///
/// # Safety
///
/// Requires the s390x vector facility. Caller must verify via
/// `platform::caps().has(s390x::VECTOR)`.
#[target_feature(enable = "vector")]
pub unsafe fn crc64_xz_vgfm_2way(crc: u64, data: &[u8]) -> u64 {
  crc64_vgfm_2way(
    crc,
    data,
    XZ_FOLD_256B,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-XZ using VGFM folding (4-way ILP variant).
///
/// # Safety
///
/// Requires the s390x vector facility. Caller must verify via
/// `platform::caps().has(s390x::VECTOR)`.
#[target_feature(enable = "vector")]
pub unsafe fn crc64_xz_vgfm_4way(crc: u64, data: &[u8]) -> u64 {
  crc64_vgfm_4way(
    crc,
    data,
    XZ_FOLD_512B,
    &XZ_COMBINE_4WAY,
    &crate::common::clmul::CRC64_XZ_CLMUL,
    &super::kernel_tables::XZ_TABLES_16,
  )
}

/// CRC-64-NVME using VGFM folding.
///
/// # Safety
///
/// Requires the s390x vector facility. Caller must verify via
/// `platform::caps().has(s390x::VECTOR)`.
#[target_feature(enable = "vector")]
pub unsafe fn crc64_nvme_vgfm(crc: u64, data: &[u8]) -> u64 {
  crc64_vgfm(
    crc,
    data,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

/// CRC-64-NVME using VGFM folding (2-way ILP variant).
///
/// # Safety
///
/// Requires the s390x vector facility. Caller must verify via
/// `platform::caps().has(s390x::VECTOR)`.
#[target_feature(enable = "vector")]
pub unsafe fn crc64_nvme_vgfm_2way(crc: u64, data: &[u8]) -> u64 {
  crc64_vgfm_2way(
    crc,
    data,
    NVME_FOLD_256B,
    &crate::common::clmul::CRC64_NVME_CLMUL,
    &super::kernel_tables::NVME_TABLES_16,
  )
}

/// CRC-64-NVME using VGFM folding (4-way ILP variant).
///
/// # Safety
///
/// Requires the s390x vector facility. Caller must verify via
/// `platform::caps().has(s390x::VECTOR)`.
#[target_feature(enable = "vector")]
pub unsafe fn crc64_nvme_vgfm_4way(crc: u64, data: &[u8]) -> u64 {
  crc64_vgfm_4way(
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
pub fn crc64_xz_vgfm_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_vgfm(crc, data) }
}

#[inline]
pub fn crc64_xz_vgfm_2way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_vgfm_2way(crc, data) }
}

#[inline]
pub fn crc64_xz_vgfm_4way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_vgfm_4way(crc, data) }
}

#[inline]
pub fn crc64_nvme_vgfm_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_vgfm(crc, data) }
}

#[inline]
pub fn crc64_nvme_vgfm_2way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_vgfm_2way(crc, data) }
}

#[inline]
pub fn crc64_nvme_vgfm_4way_safe(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_vgfm_4way(crc, data) }
}
