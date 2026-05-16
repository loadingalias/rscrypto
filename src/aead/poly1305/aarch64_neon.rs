use core::arch::aarch64::{
  uint32x4_t, uint64x2_t, vaddq_u64, vaddvq_u64, vcombine_u32, vcreate_u32, vget_high_u32, vget_low_u32, vmull_u32,
};

use super::{FULL_BLOCK_HIBIT, LIMB_MASK, State, compute_block_aarch64_neon, load_u32_le};
use crate::{aead::AeadByteLengths, traits::ct};

define_target_feature_forwarder! {
  pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
    feature = "neon";
    outer_safety = "backend selection guarantees NEON is available before this wrapper is chosen.";
    inner_safety = "this wrapper enables NEON before calling the shared NEON body.";
    call = compute_block_aarch64_neon(state, block, partial);
  }
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

pub(crate) struct AeadPar4 {
  state: State,
  powers: Powers,
  cached: [[u8; 16]; 4],
  num_cached: usize,
}

impl AeadPar4 {
  #[inline]
  pub(crate) fn new(key: &[u8; 32]) -> Self {
    let state = State::new(key);
    let powers = Powers::new(state.r);
    Self {
      state,
      powers,
      cached: [[0u8; 16]; 4],
      num_cached: 0,
    }
  }

  #[inline]
  pub(crate) fn update_padded_segment(&mut self, segment: &[u8]) {
    // SAFETY: caller selected this updater only after NEON availability was confirmed. AEAD segment
    // padding turns a trailing partial segment into a full zero-padded Poly1305 block.
    unsafe { self.update_padded_segment_neon(segment) }
  }

  #[inline]
  pub(crate) fn finalize(mut self, lengths: AeadByteLengths) -> [u8; 16] {
    // SAFETY: caller selected this updater only after NEON availability was confirmed. The length
    // encoding is a full AEAD block.
    unsafe { self.push(lengths.to_le_bytes_block()) };

    for block in self.cached.iter().take(self.num_cached) {
      // SAFETY: this updater is NEON-selected and every cached AEAD block is a full 16-byte block.
      unsafe { compute_block_aarch64_neon(&mut self.state, block, false) };
    }

    let tag = self.state.clone().finalize();
    ct::zeroize(self.cached.as_flattened_mut());
    tag
  }

  #[target_feature(enable = "neon")]
  unsafe fn update_padded_segment_neon(&mut self, segment: &[u8]) {
    let mut chunks = segment.chunks_exact(16);
    for chunk in &mut chunks {
      let mut block = [0u8; 16];
      block.copy_from_slice(chunk);
      // SAFETY: this function is NEON-enabled and `block` is a full AEAD block.
      unsafe { self.push(block) };
    }

    let rem = chunks.remainder();
    if !rem.is_empty() {
      let mut block = [0u8; 16];
      block[..rem.len()].copy_from_slice(rem);
      // SAFETY: this function is NEON-enabled and AEAD zero-padding makes this a full block.
      unsafe { self.push(block) };
    }
  }

  #[inline(always)]
  unsafe fn push(&mut self, block: [u8; 16]) {
    self.cached[self.num_cached] = block;
    self.num_cached = self.num_cached.strict_add(1);
    if self.num_cached == 4 {
      // SAFETY: caller is NEON-enabled and `cached` contains four initialized full AEAD blocks.
      unsafe { accumulate_4_blocks(&self.cached, &mut self.state, &self.powers) };
      self.num_cached = 0;
    }
  }
}

#[cfg_attr(not(any(feature = "xchacha20poly1305", feature = "diag", test)), allow(dead_code))]
pub(super) fn authenticate_aead_par4(
  aad: &[u8],
  ciphertext: &[u8],
  key: &[u8; 32],
  lengths: AeadByteLengths,
) -> [u8; 16] {
  let mut authenticator = AeadPar4::new(key);
  authenticator.update_padded_segment(aad);
  authenticator.update_padded_segment(ciphertext);
  authenticator.finalize(lengths)
}

#[inline(always)]
unsafe fn accumulate_4_blocks(blocks: &[[u8; 16]; 4], state: &mut State, powers: &Powers) {
  let h = mul_unreduced(state.h, powers.r4);
  // SAFETY: caller is NEON-enabled and `blocks` contains four full 16-byte AEAD blocks.
  let m = unsafe { mul4_spaced_sum(blocks, powers) };
  state.h = reduce_unreduced([
    h[0].wrapping_add(m[0]),
    h[1].wrapping_add(m[1]),
    h[2].wrapping_add(m[2]),
    h[3].wrapping_add(m[3]),
    h[4].wrapping_add(m[4]),
  ]);
}

#[inline(always)]
unsafe fn mul4_spaced_sum(blocks: &[[u8; 16]; 4], powers: &Powers) -> [u64; 5] {
  let b0 = block_limbs(&blocks[0]);
  let b1 = block_limbs(&blocks[1]);
  let b2 = block_limbs(&blocks[2]);
  let b3 = block_limbs(&blocks[3]);

  // SAFETY: lane construction uses immediate scalar inputs and does not read memory.
  let (x0, x1, x2, x3, x4) = unsafe {
    (
      lane4(b0[0], b1[0], b2[0], b3[0]),
      lane4(b0[1], b1[1], b2[1], b3[1]),
      lane4(b0[2], b1[2], b2[2], b3[2]),
      lane4(b0[3], b1[3], b2[3], b3[3]),
      lane4(b0[4], b1[4], b2[4], b3[4]),
    )
  };

  let r1 = powers.r1;
  let r2 = powers.r2;
  let r3 = powers.r3;
  let r4 = powers.r4;

  // SAFETY: lane construction uses reduced 26-bit limbs; fivefold limbs remain within `u32`.
  let (r0, r1v, r2v, r3v, r4v, s1, s2, s3, s4) = unsafe {
    (
      lane4(r4[0], r3[0], r2[0], r1[0]),
      lane4(r4[1], r3[1], r2[1], r1[1]),
      lane4(r4[2], r3[2], r2[2], r1[2]),
      lane4(r4[3], r3[3], r2[3], r1[3]),
      lane4(r4[4], r3[4], r2[4], r1[4]),
      lane4(r4[1] * 5, r3[1] * 5, r2[1] * 5, r1[1] * 5),
      lane4(r4[2] * 5, r3[2] * 5, r2[2] * 5, r1[2] * 5),
      lane4(r4[3] * 5, r3[3] * 5, r2[3] * 5, r1[3] * 5),
      lane4(r4[4] * 5, r3[4] * 5, r2[4] * 5, r1[4] * 5),
    )
  };

  // SAFETY: all lanes are valid NEON vectors in this target-feature-enabled function.
  unsafe {
    [
      dot5_sum(x0, x1, x2, x3, x4, r0, s4, s3, s2, s1),
      dot5_sum(x0, x1, x2, x3, x4, r1v, r0, s4, s3, s2),
      dot5_sum(x0, x1, x2, x3, x4, r2v, r1v, r0, s4, s3),
      dot5_sum(x0, x1, x2, x3, x4, r3v, r2v, r1v, r0, s4),
      dot5_sum(x0, x1, x2, x3, x4, r4v, r3v, r2v, r1v, r0),
    ]
  }
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
unsafe fn lane4(a: u32, b: u32, c: u32, d: u32) -> uint32x4_t {
  // SAFETY: caller guarantees NEON is enabled; inputs are scalar lane values.
  unsafe {
    vcombine_u32(
      vcreate_u32((u64::from(b) << 32) | u64::from(a)),
      vcreate_u32((u64::from(d) << 32) | u64::from(c)),
    )
  }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn dot5_sum(
  x0: uint32x4_t,
  x1: uint32x4_t,
  x2: uint32x4_t,
  x3: uint32x4_t,
  x4: uint32x4_t,
  y0: uint32x4_t,
  y1: uint32x4_t,
  y2: uint32x4_t,
  y3: uint32x4_t,
  y4: uint32x4_t,
) -> u64 {
  // SAFETY: caller guarantees NEON is enabled and all inputs are valid four-lane vectors.
  unsafe {
    let mut lo = vmull_u32(vget_low_u32(x0), vget_low_u32(y0));
    let mut hi = vmull_u32(vget_high_u32(x0), vget_high_u32(y0));

    accumulate_mul(&mut lo, &mut hi, x1, y1);
    accumulate_mul(&mut lo, &mut hi, x2, y2);
    accumulate_mul(&mut lo, &mut hi, x3, y3);
    accumulate_mul(&mut lo, &mut hi, x4, y4);

    vaddvq_u64(vaddq_u64(lo, hi))
  }
}

#[inline(always)]
unsafe fn accumulate_mul(lo: &mut uint64x2_t, hi: &mut uint64x2_t, x: uint32x4_t, y: uint32x4_t) {
  // SAFETY: caller guarantees NEON is enabled and all inputs are valid four-lane vectors.
  unsafe {
    *lo = vaddq_u64(*lo, vmull_u32(vget_low_u32(x), vget_low_u32(y)));
    *hi = vaddq_u64(*hi, vmull_u32(vget_high_u32(x), vget_high_u32(y)));
  }
}

#[inline(always)]
fn mul_mod(a: [u32; 5], b: [u32; 5]) -> [u32; 5] {
  reduce_unreduced(mul_unreduced(a, b))
}

#[inline(always)]
fn mul_unreduced(a: [u32; 5], b: [u32; 5]) -> [u64; 5] {
  let b1_5 = b[1] * 5;
  let b2_5 = b[2] * 5;
  let b3_5 = b[3] * 5;
  let b4_5 = b[4] * 5;

  [
    (u64::from(a[0]) * u64::from(b[0]))
      + (u64::from(a[1]) * u64::from(b4_5))
      + (u64::from(a[2]) * u64::from(b3_5))
      + (u64::from(a[3]) * u64::from(b2_5))
      + (u64::from(a[4]) * u64::from(b1_5)),
    (u64::from(a[0]) * u64::from(b[1]))
      + (u64::from(a[1]) * u64::from(b[0]))
      + (u64::from(a[2]) * u64::from(b4_5))
      + (u64::from(a[3]) * u64::from(b3_5))
      + (u64::from(a[4]) * u64::from(b2_5)),
    (u64::from(a[0]) * u64::from(b[2]))
      + (u64::from(a[1]) * u64::from(b[1]))
      + (u64::from(a[2]) * u64::from(b[0]))
      + (u64::from(a[3]) * u64::from(b4_5))
      + (u64::from(a[4]) * u64::from(b3_5)),
    (u64::from(a[0]) * u64::from(b[3]))
      + (u64::from(a[1]) * u64::from(b[2]))
      + (u64::from(a[2]) * u64::from(b[1]))
      + (u64::from(a[3]) * u64::from(b[0]))
      + (u64::from(a[4]) * u64::from(b4_5)),
    (u64::from(a[0]) * u64::from(b[4]))
      + (u64::from(a[1]) * u64::from(b[3]))
      + (u64::from(a[2]) * u64::from(b[2]))
      + (u64::from(a[3]) * u64::from(b[1]))
      + (u64::from(a[4]) * u64::from(b[0])),
  ]
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
  h0 = h0.wrapping_add(c * 5);

  c = h0 >> 26;
  h0 &= u64::from(LIMB_MASK);
  let h1 = h1_base.wrapping_add(c);

  [h0 as u32, h1 as u32, h2, h3, h4]
}
