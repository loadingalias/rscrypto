use core::simd::{i64x2, num::SimdUint, u32x4, u64x4};

use super::{FULL_BLOCK_HIBIT, LIMB_MASK, State, compute_block_scalar_reduction, load_u32_le};
use crate::{aead::AeadByteLengths, traits::ct};

#[inline]
pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
  // SAFETY: backend selection guarantees the RISC-V V extension before this wrapper is chosen.
  unsafe { compute_block_impl(state, block, partial) }
}

#[target_feature(enable = "v")]
unsafe fn compute_block_impl(state: &mut State, block: &[u8; 16], partial: bool) {
  compute_block_scalar_reduction(state, block, partial, |lhs, rhs| {
    // SAFETY: target_feature ensures RVV availability for this entire wrapper.
    unsafe { sum4_mul(lhs, rhs) }
  });
}

/// Vectorized 4-element dot product using two 64-bit RVV lane multiplies.
#[target_feature(enable = "v")]
unsafe fn sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
  let a_lo = i64x2::from_array([i64::from(lhs[0]), i64::from(lhs[1])]);
  let b_lo = i64x2::from_array([i64::from(rhs[0]), i64::from(rhs[1])]);
  let prod_lo = a_lo * b_lo;

  let a_hi = i64x2::from_array([i64::from(lhs[2]), i64::from(lhs[3])]);
  let b_hi = i64x2::from_array([i64::from(rhs[2]), i64::from(rhs[3])]);
  let prod_hi = a_hi * b_hi;

  let sum = prod_lo + prod_hi;
  let lanes = sum.to_array();
  (lanes[0] as u64).wrapping_add(lanes[1] as u64)
}

#[derive(Clone, Copy)]
struct Powers {
  r1: [u32; 5],
  r2: [u32; 5],
  r3: [u32; 5],
  r4: [u32; 5],
}

impl Powers {
  #[inline(always)]
  fn new(r1: [u32; 5]) -> Self {
    let r2 = mul_mod(r1, r1);
    let r3 = mul_mod(r2, r1);
    let r4 = mul_mod(r2, r2);
    Self { r1, r2, r3, r4 }
  }
}

#[cfg_attr(not(any(feature = "xchacha20poly1305", feature = "diag", test)), allow(dead_code))]
pub(super) fn authenticate_aead_par4(
  aad: &[u8],
  ciphertext: &[u8],
  key: &[u8; 32],
  lengths: AeadByteLengths,
) -> [u8; 16] {
  // SAFETY: RVV AEAD authenticator dispatch because:
  // 1. The caller verifies `platform::caps().has(riscv::V)` before selecting this authenticator.
  // 2. `lengths` is derived from the same `aad` and `ciphertext` slices.
  unsafe { authenticate_aead_par4_impl(aad, ciphertext, key, lengths) }
}

#[target_feature(enable = "v")]
unsafe fn authenticate_aead_par4_impl(
  aad: &[u8],
  ciphertext: &[u8],
  key: &[u8; 32],
  lengths: AeadByteLengths,
) -> [u8; 16] {
  let mut state = State::new(key);
  let powers = Powers::new(state.r);
  let mut cached = [[0u8; 16]; 4];
  let mut num_cached = 0usize;

  let mut segment_index = 0usize;
  while segment_index < 2 {
    let segment = if segment_index == 0 { aad } else { ciphertext };
    let mut offset = 0usize;

    while num_cached != 0 && offset.strict_add(16) <= segment.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&segment[offset..offset.strict_add(16)]);
      // SAFETY: RVV cached-block push because:
      // 1. This entry point is compiled with the RISC-V V target feature.
      // 2. `block` is a full AEAD block.
      unsafe { push_cached(&mut cached, &mut num_cached, &mut state, &powers, block) };
      offset = offset.strict_add(16);
    }

    if num_cached == 0 {
      let group_len = segment.len().strict_sub(offset).strict_div(64).strict_mul(64);
      let group_end = offset.strict_add(group_len);
      for group in segment[offset..group_end].chunks_exact(64) {
        let (blocks, remainder) = group.as_chunks::<16>();
        debug_assert!(remainder.is_empty());
        let [b0, b1, b2, b3] = blocks else {
          unreachable!("64-byte Poly1305 group must split into four blocks");
        };
        // SAFETY: direct four-block accumulation because:
        // 1. `group` is a 64-byte exact chunk split into four full Poly1305 blocks.
        // 2. This entry point is compiled with the RISC-V V target feature.
        // 3. `num_cached == 0`, so direct accumulation preserves AEAD block order.
        unsafe { accumulate_4_block_refs([b0, b1, b2, b3], &mut state, &powers) };
      }
      offset = group_end;
    }

    while offset.strict_add(16) <= segment.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&segment[offset..offset.strict_add(16)]);
      // SAFETY: RVV cached-block push because:
      // 1. This entry point is compiled with the RISC-V V target feature.
      // 2. `block` is a full AEAD block.
      unsafe { push_cached(&mut cached, &mut num_cached, &mut state, &powers, block) };
      offset = offset.strict_add(16);
    }

    let rem = &segment[offset..];
    if !rem.is_empty() {
      let mut block = [0u8; 16];
      block[..rem.len()].copy_from_slice(rem);
      // SAFETY: RVV cached-block push because:
      // 1. This entry point is compiled with the RISC-V V target feature.
      // 2. AEAD zero-padding makes this a full block.
      unsafe { push_cached(&mut cached, &mut num_cached, &mut state, &powers, block) };
    }

    segment_index = segment_index.strict_add(1);
  }

  // SAFETY: RVV cached-block push because:
  // 1. This entry point is compiled with the RISC-V V target feature.
  // 2. The AEAD length encoding is exactly one full Poly1305 block.
  unsafe {
    push_cached(
      &mut cached,
      &mut num_cached,
      &mut state,
      &powers,
      lengths.to_le_bytes_block(),
    )
  };

  for block in cached.iter().take(num_cached) {
    state.compute_block_portable(block, false);
  }

  let tag = state.clone().finalize();
  ct::zeroize(cached.as_flattened_mut());
  tag
}

#[inline(always)]
unsafe fn push_cached(
  cached: &mut [[u8; 16]; 4],
  num_cached: &mut usize,
  state: &mut State,
  powers: &Powers,
  block: [u8; 16],
) {
  cached[*num_cached] = block;
  *num_cached = (*num_cached).strict_add(1);
  if *num_cached == 4 {
    // SAFETY: RVV four-block accumulation because:
    // 1. Caller is compiled with the RISC-V V target feature.
    // 2. `cached` contains four initialized full AEAD blocks.
    unsafe { accumulate_4_blocks(cached, state, powers) };
    *num_cached = 0;
  }
}

#[inline(always)]
unsafe fn accumulate_4_blocks(blocks: &[[u8; 16]; 4], state: &mut State, powers: &Powers) {
  // SAFETY: RVV four-block accumulation because:
  // 1. Caller is compiled with the RISC-V V target feature.
  // 2. `blocks` contains four full 16-byte AEAD blocks.
  unsafe { accumulate_4_block_refs([&blocks[0], &blocks[1], &blocks[2], &blocks[3]], state, powers) };
}

#[inline(always)]
unsafe fn accumulate_4_block_refs(blocks: [&[u8; 16]; 4], state: &mut State, powers: &Powers) {
  let h = mul_unreduced(state.h, powers.r4);
  // SAFETY: RVV spaced-message multiplication because:
  // 1. Caller is compiled with the RISC-V V target feature.
  // 2. `blocks` contains four full 16-byte AEAD blocks.
  let m = unsafe { mul4_spaced_sum_refs(blocks, powers) };
  state.h = reduce_unreduced([
    h[0].wrapping_add(m[0]),
    h[1].wrapping_add(m[1]),
    h[2].wrapping_add(m[2]),
    h[3].wrapping_add(m[3]),
    h[4].wrapping_add(m[4]),
  ]);
}

#[inline(always)]
unsafe fn mul4_spaced_sum_refs(blocks: [&[u8; 16]; 4], powers: &Powers) -> [u64; 5] {
  let b0 = block_limbs(blocks[0]);
  let b1 = block_limbs(blocks[1]);
  let b2 = block_limbs(blocks[2]);
  let b3 = block_limbs(blocks[3]);

  let x0 = lane4(b0[0], b1[0], b2[0], b3[0]);
  let x1 = lane4(b0[1], b1[1], b2[1], b3[1]);
  let x2 = lane4(b0[2], b1[2], b2[2], b3[2]);
  let x3 = lane4(b0[3], b1[3], b2[3], b3[3]);
  let x4 = lane4(b0[4], b1[4], b2[4], b3[4]);

  let r1 = powers.r1;
  let r2 = powers.r2;
  let r3 = powers.r3;
  let r4 = powers.r4;

  let d0 = {
    let r0 = lane4(r4[0], r3[0], r2[0], r1[0]);
    let s4 = lane4(
      r4[4].wrapping_mul(5),
      r3[4].wrapping_mul(5),
      r2[4].wrapping_mul(5),
      r1[4].wrapping_mul(5),
    );
    let s3 = lane4(
      r4[3].wrapping_mul(5),
      r3[3].wrapping_mul(5),
      r2[3].wrapping_mul(5),
      r1[3].wrapping_mul(5),
    );
    let s2 = lane4(
      r4[2].wrapping_mul(5),
      r3[2].wrapping_mul(5),
      r2[2].wrapping_mul(5),
      r1[2].wrapping_mul(5),
    );
    let s1 = lane4(
      r4[1].wrapping_mul(5),
      r3[1].wrapping_mul(5),
      r2[1].wrapping_mul(5),
      r1[1].wrapping_mul(5),
    );
    dot5_sum(x0, x1, x2, x3, x4, r0, s4, s3, s2, s1)
  };
  let d1 = {
    let r1v = lane4(r4[1], r3[1], r2[1], r1[1]);
    let r0 = lane4(r4[0], r3[0], r2[0], r1[0]);
    let s4 = lane4(
      r4[4].wrapping_mul(5),
      r3[4].wrapping_mul(5),
      r2[4].wrapping_mul(5),
      r1[4].wrapping_mul(5),
    );
    let s3 = lane4(
      r4[3].wrapping_mul(5),
      r3[3].wrapping_mul(5),
      r2[3].wrapping_mul(5),
      r1[3].wrapping_mul(5),
    );
    let s2 = lane4(
      r4[2].wrapping_mul(5),
      r3[2].wrapping_mul(5),
      r2[2].wrapping_mul(5),
      r1[2].wrapping_mul(5),
    );
    dot5_sum(x0, x1, x2, x3, x4, r1v, r0, s4, s3, s2)
  };
  let d2 = {
    let r2v = lane4(r4[2], r3[2], r2[2], r1[2]);
    let r1v = lane4(r4[1], r3[1], r2[1], r1[1]);
    let r0 = lane4(r4[0], r3[0], r2[0], r1[0]);
    let s4 = lane4(
      r4[4].wrapping_mul(5),
      r3[4].wrapping_mul(5),
      r2[4].wrapping_mul(5),
      r1[4].wrapping_mul(5),
    );
    let s3 = lane4(
      r4[3].wrapping_mul(5),
      r3[3].wrapping_mul(5),
      r2[3].wrapping_mul(5),
      r1[3].wrapping_mul(5),
    );
    dot5_sum(x0, x1, x2, x3, x4, r2v, r1v, r0, s4, s3)
  };
  let d3 = {
    let r3v = lane4(r4[3], r3[3], r2[3], r1[3]);
    let r2v = lane4(r4[2], r3[2], r2[2], r1[2]);
    let r1v = lane4(r4[1], r3[1], r2[1], r1[1]);
    let r0 = lane4(r4[0], r3[0], r2[0], r1[0]);
    let s4 = lane4(
      r4[4].wrapping_mul(5),
      r3[4].wrapping_mul(5),
      r2[4].wrapping_mul(5),
      r1[4].wrapping_mul(5),
    );
    dot5_sum(x0, x1, x2, x3, x4, r3v, r2v, r1v, r0, s4)
  };
  let d4 = {
    let r4v = lane4(r4[4], r3[4], r2[4], r1[4]);
    let r3v = lane4(r4[3], r3[3], r2[3], r1[3]);
    let r2v = lane4(r4[2], r3[2], r2[2], r1[2]);
    let r1v = lane4(r4[1], r3[1], r2[1], r1[1]);
    let r0 = lane4(r4[0], r3[0], r2[0], r1[0]);
    dot5_sum(x0, x1, x2, x3, x4, r4v, r3v, r2v, r1v, r0)
  };

  [d0, d1, d2, d3, d4]
}

#[inline(always)]
fn block_limbs(block: &[u8; 16]) -> [u32; 5] {
  [
    load_u32_le(&block[0..4]) & LIMB_MASK,
    (load_u32_le(&block[3..7]) >> 2) & LIMB_MASK,
    (load_u32_le(&block[6..10]) >> 4) & LIMB_MASK,
    (load_u32_le(&block[9..13]) >> 6) & LIMB_MASK,
    (load_u32_le(&block[12..16]) >> 8) | FULL_BLOCK_HIBIT,
  ]
}

#[inline(always)]
fn lane4(a: u32, b: u32, c: u32, d: u32) -> u32x4 {
  u32x4::from_array([a, b, c, d])
}

#[inline(always)]
fn widen(value: u32x4) -> u64x4 {
  value.cast::<u64>()
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dot5_sum(
  x0: u32x4,
  x1: u32x4,
  x2: u32x4,
  x3: u32x4,
  x4: u32x4,
  y0: u32x4,
  y1: u32x4,
  y2: u32x4,
  y3: u32x4,
  y4: u32x4,
) -> u64 {
  let sum = widen(x0) * widen(y0)
    + widen(x1) * widen(y1)
    + widen(x2) * widen(y2)
    + widen(x3) * widen(y3)
    + widen(x4) * widen(y4);
  let lanes = sum.to_array();
  lanes[0]
    .wrapping_add(lanes[1])
    .wrapping_add(lanes[2])
    .wrapping_add(lanes[3])
}

#[inline(always)]
fn mul_mod(a: [u32; 5], b: [u32; 5]) -> [u32; 5] {
  reduce_unreduced(mul_unreduced(a, b))
}

#[inline(always)]
fn mul_unreduced(a: [u32; 5], b: [u32; 5]) -> [u64; 5] {
  let b1_5 = b[1].wrapping_mul(5);
  let b2_5 = b[2].wrapping_mul(5);
  let b3_5 = b[3].wrapping_mul(5);
  let b4_5 = b[4].wrapping_mul(5);

  [
    mul5_sum(a, [b[0], b4_5, b3_5, b2_5, b1_5]),
    mul5_sum(a, [b[1], b[0], b4_5, b3_5, b2_5]),
    mul5_sum(a, [b[2], b[1], b[0], b4_5, b3_5]),
    mul5_sum(a, [b[3], b[2], b[1], b[0], b4_5]),
    mul5_sum(a, [b[4], b[3], b[2], b[1], b[0]]),
  ]
}

#[inline(always)]
fn mul5_sum(a: [u32; 5], b: [u32; 5]) -> u64 {
  let mut sum = 0u64;
  let mut index = 0usize;
  while index < 5 {
    sum = sum.wrapping_add(u64::from(a[index]).wrapping_mul(u64::from(b[index])));
    index = index.strict_add(1);
  }
  sum
}

#[inline(always)]
fn reduce_unreduced(mut d: [u64; 5]) -> [u32; 5] {
  let mut c = d[0] >> 26;
  let mut h0 = d[0] & u64::from(LIMB_MASK);
  d[1] = d[1].wrapping_add(c);

  c = d[1] >> 26;
  let h1_base = d[1] & u64::from(LIMB_MASK);
  d[2] = d[2].wrapping_add(c);

  c = d[2] >> 26;
  let h2 = (d[2] as u32) & LIMB_MASK;
  d[3] = d[3].wrapping_add(c);

  c = d[3] >> 26;
  let h3 = (d[3] as u32) & LIMB_MASK;
  d[4] = d[4].wrapping_add(c);

  c = d[4] >> 26;
  let h4 = (d[4] as u32) & LIMB_MASK;
  h0 = h0.wrapping_add(c.wrapping_mul(5));

  c = h0 >> 26;
  h0 &= u64::from(LIMB_MASK);
  let h1 = h1_base.wrapping_add(c);

  [h0 as u32, h1 as u32, h2, h3, h4]
}
