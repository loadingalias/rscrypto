//! Shared five-limb Poly1305 arithmetic and secret-state owner.
//!
//! Callers own message framing: standalone final fragments append a byte of one
//! and suppress the full-block high bit; AEAD segments zero-pad to full blocks.
//! Accelerated AEAD kernels read `r` and update `h` in the same radix-2^26 form.

use crate::traits::ct;

pub(crate) const LIMB_MASK: u32 = 0x03ff_ffff;
pub(crate) const FULL_BLOCK_HIBIT: u32 = 1 << 24;

#[inline]
pub(crate) fn load_u32_le(input: &[u8]) -> u32 {
  let mut bytes = [0u8; 4];
  bytes.copy_from_slice(input);
  u32::from_le_bytes(bytes)
}

#[derive(Clone, Default)]
pub(crate) struct State {
  pub(crate) r: [u32; 5],
  pub(crate) h: [u32; 5],
  pad: [u32; 4],
}

impl Drop for State {
  fn drop(&mut self) {
    ct::zeroize_words_no_fence(&mut self.r);
    ct::zeroize_words_no_fence(&mut self.h);
    ct::zeroize_words_no_fence(&mut self.pad);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

impl State {
  #[inline]
  pub(crate) fn new(key: &[u8; 32]) -> Self {
    Self {
      r: [
        load_u32_le(&key[0..4]) & LIMB_MASK,
        (load_u32_le(&key[3..7]) >> 2) & 0x03ff_ff03,
        (load_u32_le(&key[6..10]) >> 4) & 0x03ff_c0ff,
        (load_u32_le(&key[9..13]) >> 6) & 0x03f0_3fff,
        (load_u32_le(&key[12..16]) >> 8) & 0x000f_ffff,
      ],
      h: [0u32; 5],
      pad: [
        load_u32_le(&key[16..20]),
        load_u32_le(&key[20..24]),
        load_u32_le(&key[24..28]),
        load_u32_le(&key[28..32]),
      ],
    }
  }

  #[inline(always)]
  pub(crate) fn compute_block_portable(&mut self, block: &[u8; 16], partial: bool) {
    #[inline(always)]
    fn fivefold_limb(limb: u32) -> u32 {
      const MAX_UNSCALED: u32 = 858_993_459;
      debug_assert!(limb <= MAX_UNSCALED);

      let product = u64::from(limb).strict_mul(5);
      let [b0, b1, b2, b3, _, _, _, _] = product.to_le_bytes();
      u32::from_le_bytes([b0, b1, b2, b3])
    }

    #[inline(always)]
    fn scalar_dot5(lhs: [u32; 5], rhs: [u32; 5]) -> u64 {
      let [l0, l1, l2, l3, l4] = lhs;
      let [r0, r1, r2, r3, r4] = rhs;
      u64::from(l0)
        .strict_mul(u64::from(r0))
        .strict_add(u64::from(l1).strict_mul(u64::from(r1)))
        .strict_add(u64::from(l2).strict_mul(u64::from(r2)))
        .strict_add(u64::from(l3).strict_mul(u64::from(r3)))
        .strict_add(u64::from(l4).strict_mul(u64::from(r4)))
    }

    #[inline(always)]
    fn narrow_limb(value: u64) -> u32 {
      debug_assert_eq!(value >> u32::BITS, 0);
      let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
      u32::from_le_bytes([b0, b1, b2, b3])
    }

    let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

    // These masks are identities for clamped keys. They also expose the bound
    // at this kernel boundary: each dot product is below 21 * 2^58 < 2^63,
    // even for arbitrary u32 message limbs. This lets the compiler remove
    // overflow paths from strict u64 arithmetic; no wrapping sum or unsafe assumption
    // is needed. The constructor's narrower clamp masks remain authoritative.
    let r0 = self.r[0] & LIMB_MASK;
    let r1 = self.r[1] & LIMB_MASK;
    let r2 = self.r[2] & LIMB_MASK;
    let r3 = self.r[3] & LIMB_MASK;
    let r4 = self.r[4] & LIMB_MASK;

    let s1 = fivefold_limb(r1);
    let s2 = fivefold_limb(r2);
    let s3 = fivefold_limb(r3);
    let s4 = fivefold_limb(r4);

    let mut h0 = self.h[0];
    let mut h1 = self.h[1];
    let mut h2 = self.h[2];
    let mut h3 = self.h[3];
    let mut h4 = self.h[4];

    h0 = h0.wrapping_add(load_u32_le(&block[0..4]) & LIMB_MASK);
    h1 = h1.wrapping_add((load_u32_le(&block[3..7]) >> 2) & LIMB_MASK);
    h2 = h2.wrapping_add((load_u32_le(&block[6..10]) >> 4) & LIMB_MASK);
    h3 = h3.wrapping_add((load_u32_le(&block[9..13]) >> 6) & LIMB_MASK);
    h4 = h4.wrapping_add((load_u32_le(&block[12..16]) >> 8) | hibit);

    let d0 = scalar_dot5([h0, h1, h2, h3, h4], [r0, s4, s3, s2, s1]);
    let mut d1 = scalar_dot5([h0, h1, h2, h3, h4], [r1, r0, s4, s3, s2]);
    let mut d2 = scalar_dot5([h0, h1, h2, h3, h4], [r2, r1, r0, s4, s3]);
    let mut d3 = scalar_dot5([h0, h1, h2, h3, h4], [r3, r2, r1, r0, s4]);
    let mut d4 = scalar_dot5([h0, h1, h2, h3, h4], [r4, r3, r2, r1, r0]);

    let mut c = narrow_limb(d0 >> 26);
    h0 = narrow_limb(d0 & u64::from(LIMB_MASK));
    d1 = d1.strict_add(u64::from(c));

    c = narrow_limb(d1 >> 26);
    h1 = narrow_limb(d1 & u64::from(LIMB_MASK));
    d2 = d2.strict_add(u64::from(c));

    c = narrow_limb(d2 >> 26);
    h2 = narrow_limb(d2 & u64::from(LIMB_MASK));
    d3 = d3.strict_add(u64::from(c));

    c = narrow_limb(d3 >> 26);
    h3 = narrow_limb(d3 & u64::from(LIMB_MASK));
    d4 = d4.strict_add(u64::from(c));

    c = narrow_limb(d4 >> 26);
    h4 = narrow_limb(d4 & u64::from(LIMB_MASK));
    h0 = h0.wrapping_add(fivefold_limb(c));

    let c = h0 >> 26;
    h0 &= LIMB_MASK;
    h1 = h1.wrapping_add(c);

    self.h = [h0, h1, h2, h3, h4];
  }

  #[inline(always)]
  pub(crate) fn finalize(mut self) -> [u8; 16] {
    self.finalize_in_place()
  }

  #[inline(always)]
  fn finalize_in_place(&mut self) -> [u8; 16] {
    #[inline(always)]
    fn fivefold_carry(carry: u32) -> u32 {
      const MAX_UNSCALED: u32 = 858_993_459;
      debug_assert!(carry <= MAX_UNSCALED);

      let product = u64::from(carry).strict_mul(5);
      let [b0, b1, b2, b3, _, _, _, _] = product.to_le_bytes();
      u32::from_le_bytes([b0, b1, b2, b3])
    }

    #[inline(always)]
    fn low_word(value: u64) -> u32 {
      let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
      u32::from_le_bytes([b0, b1, b2, b3])
    }

    let mut h0 = self.h[0];
    let mut h1 = self.h[1];
    let mut h2 = self.h[2];
    let mut h3 = self.h[3];
    let mut h4 = self.h[4];

    let mut c = h1 >> 26;
    h1 &= LIMB_MASK;
    h2 = h2.wrapping_add(c);

    c = h2 >> 26;
    h2 &= LIMB_MASK;
    h3 = h3.wrapping_add(c);

    c = h3 >> 26;
    h3 &= LIMB_MASK;
    h4 = h4.wrapping_add(c);

    c = h4 >> 26;
    h4 &= LIMB_MASK;
    h0 = h0.wrapping_add(fivefold_carry(c));

    c = h0 >> 26;
    h0 &= LIMB_MASK;
    h1 = h1.wrapping_add(c);

    let mut g0 = h0.wrapping_add(5);
    c = g0 >> 26;
    g0 &= LIMB_MASK;

    let mut g1 = h1.wrapping_add(c);
    c = g1 >> 26;
    g1 &= LIMB_MASK;

    let mut g2 = h2.wrapping_add(c);
    c = g2 >> 26;
    g2 &= LIMB_MASK;

    let mut g3 = h3.wrapping_add(c);
    c = g3 >> 26;
    g3 &= LIMB_MASK;

    let mut g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);

    let mut mask = (g4 >> 31).wrapping_sub(1);
    g0 &= mask;
    g1 &= mask;
    g2 &= mask;
    g3 &= mask;
    g4 &= mask;
    mask = !mask;

    h0 = (h0 & mask) | g0;
    h1 = (h1 & mask) | g1;
    h2 = (h2 & mask) | g2;
    h3 = (h3 & mask) | g3;
    h4 = (h4 & mask) | g4;

    h0 |= h1 << 26;
    h1 = (h1 >> 6) | (h2 << 20);
    h2 = (h2 >> 12) | (h3 << 14);
    h3 = (h3 >> 18) | (h4 << 8);

    let mut f = u64::from(h0).strict_add(u64::from(self.pad[0]));
    h0 = low_word(f);
    f = u64::from(h1).strict_add(u64::from(self.pad[1])).strict_add(f >> 32);
    h1 = low_word(f);
    f = u64::from(h2).strict_add(u64::from(self.pad[2])).strict_add(f >> 32);
    h2 = low_word(f);
    f = u64::from(h3).strict_add(u64::from(self.pad[3])).strict_add(f >> 32);
    h3 = low_word(f);

    let mut tag = [0u8; 16];
    tag[0..4].copy_from_slice(&h0.to_le_bytes());
    tag[4..8].copy_from_slice(&h1.to_le_bytes());
    tag[8..12].copy_from_slice(&h2.to_le_bytes());
    tag[12..16].copy_from_slice(&h3.to_le_bytes());
    tag
  }
}
