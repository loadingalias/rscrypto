#![allow(clippy::indexing_slicing)]

//! Portable Poly1305 core.

use crate::traits::ct;

const LIMB_MASK: u32 = 0x03ff_ffff;
const FULL_BLOCK_HIBIT: u32 = 1 << 24;

#[inline]
fn load_u32_le(input: &[u8]) -> u32 {
  let mut bytes = [0u8; 4];
  bytes.copy_from_slice(input);
  u32::from_le_bytes(bytes)
}

#[derive(Clone, Default)]
struct State {
  r: [u32; 5],
  h: [u32; 5],
  pad: [u32; 4],
}

impl State {
  #[inline]
  fn new(key: &[u8; 32]) -> Self {
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

  fn compute_block(&mut self, block: &[u8; 16], partial: bool) {
    let hibit = if partial { 0 } else { FULL_BLOCK_HIBIT };

    let r0 = self.r[0];
    let r1 = self.r[1];
    let r2 = self.r[2];
    let r3 = self.r[3];
    let r4 = self.r[4];

    let s1 = r1 * 5;
    let s2 = r2 * 5;
    let s3 = r3 * 5;
    let s4 = r4 * 5;

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

    let d0 = (u64::from(h0) * u64::from(r0))
      + (u64::from(h1) * u64::from(s4))
      + (u64::from(h2) * u64::from(s3))
      + (u64::from(h3) * u64::from(s2))
      + (u64::from(h4) * u64::from(s1));
    let mut d1 = (u64::from(h0) * u64::from(r1))
      + (u64::from(h1) * u64::from(r0))
      + (u64::from(h2) * u64::from(s4))
      + (u64::from(h3) * u64::from(s3))
      + (u64::from(h4) * u64::from(s2));
    let mut d2 = (u64::from(h0) * u64::from(r2))
      + (u64::from(h1) * u64::from(r1))
      + (u64::from(h2) * u64::from(r0))
      + (u64::from(h3) * u64::from(s4))
      + (u64::from(h4) * u64::from(s3));
    let mut d3 = (u64::from(h0) * u64::from(r3))
      + (u64::from(h1) * u64::from(r2))
      + (u64::from(h2) * u64::from(r1))
      + (u64::from(h3) * u64::from(r0))
      + (u64::from(h4) * u64::from(s4));
    let mut d4 = (u64::from(h0) * u64::from(r4))
      + (u64::from(h1) * u64::from(r3))
      + (u64::from(h2) * u64::from(r2))
      + (u64::from(h3) * u64::from(r1))
      + (u64::from(h4) * u64::from(r0));

    let mut c = (d0 >> 26) as u32;
    h0 = (d0 as u32) & LIMB_MASK;
    d1 += u64::from(c);

    c = (d1 >> 26) as u32;
    h1 = (d1 as u32) & LIMB_MASK;
    d2 += u64::from(c);

    c = (d2 >> 26) as u32;
    h2 = (d2 as u32) & LIMB_MASK;
    d3 += u64::from(c);

    c = (d3 >> 26) as u32;
    h3 = (d3 as u32) & LIMB_MASK;
    d4 += u64::from(c);

    c = (d4 >> 26) as u32;
    h4 = (d4 as u32) & LIMB_MASK;
    h0 = h0.wrapping_add(c * 5);

    c = h0 >> 26;
    h0 &= LIMB_MASK;
    h1 = h1.wrapping_add(c);

    self.h = [h0, h1, h2, h3, h4];
  }

  #[cfg(test)]
  fn update_message(&mut self, message: &[u8]) {
    let mut blocks = message.chunks_exact(16);
    for chunk in &mut blocks {
      let mut block = [0u8; 16];
      block.copy_from_slice(chunk);
      self.compute_block(&block, false);
    }

    let remainder = blocks.remainder();
    if remainder.is_empty() {
      return;
    }

    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    block[remainder.len()] = 1;
    self.compute_block(&block, true);
  }

  fn update_padded_segment(&mut self, segment: &[u8]) {
    let mut blocks = segment.chunks_exact(16);
    for chunk in &mut blocks {
      let mut block = [0u8; 16];
      block.copy_from_slice(chunk);
      self.compute_block(&block, false);
    }

    let remainder = blocks.remainder();
    if remainder.is_empty() {
      return;
    }

    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    self.compute_block(&block, false);
  }

  fn finalize(self) -> [u8; 16] {
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
    h0 = h0.wrapping_add(c * 5);

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

    let mut f = u64::from(h0) + u64::from(self.pad[0]);
    h0 = f as u32;
    f = u64::from(h1) + u64::from(self.pad[1]) + (f >> 32);
    h1 = f as u32;
    f = u64::from(h2) + u64::from(self.pad[2]) + (f >> 32);
    h2 = f as u32;
    f = u64::from(h3) + u64::from(self.pad[3]) + (f >> 32);
    h3 = f as u32;

    let mut tag = [0u8; 16];
    tag[0..4].copy_from_slice(&h0.to_le_bytes());
    tag[4..8].copy_from_slice(&h1.to_le_bytes());
    tag[8..12].copy_from_slice(&h2.to_le_bytes());
    tag[12..16].copy_from_slice(&h3.to_le_bytes());
    tag
  }
}

#[cfg(test)]
#[must_use]
pub(crate) fn authenticate(message: &[u8], key: &[u8; 32]) -> [u8; 16] {
  let mut state = State::new(key);
  state.update_message(message);
  state.finalize()
}

#[must_use]
pub(crate) fn authenticate_aead(aad: &[u8], ciphertext: &[u8], key: &[u8; 32]) -> [u8; 16] {
  let mut state = State::new(key);
  state.update_padded_segment(aad);
  state.update_padded_segment(ciphertext);

  let aad_len = match u64::try_from(aad.len()) {
    Ok(len) => len,
    Err(_) => panic!("AAD length exceeds u64"),
  };
  let ciphertext_len = match u64::try_from(ciphertext.len()) {
    Ok(len) => len,
    Err(_) => panic!("ciphertext length exceeds u64"),
  };

  let mut lengths = [0u8; 16];
  lengths[0..8].copy_from_slice(&aad_len.to_le_bytes());
  lengths[8..16].copy_from_slice(&ciphertext_len.to_le_bytes());
  state.compute_block(&lengths, false);

  let tag = state.finalize();
  ct::zeroize(&mut lengths);
  tag
}

#[cfg(test)]
mod tests {
  use super::{authenticate, authenticate_aead};

  #[test]
  fn poly1305_matches_rfc_8439_section_2_5_2() {
    let key = [
      0x85, 0xd6, 0xbe, 0x78, 0x57, 0x55, 0x6d, 0x33, 0x7f, 0x44, 0x52, 0xfe, 0x42, 0xd5, 0x06, 0xa8, 0x01, 0x03, 0x80,
      0x8a, 0xfb, 0x0d, 0xb2, 0xfd, 0x4a, 0xbf, 0xf6, 0xaf, 0x41, 0x49, 0xf5, 0x1b,
    ];
    let message = b"Cryptographic Forum Research Group";
    let expected = [
      0xa8, 0x06, 0x1d, 0xc1, 0x30, 0x51, 0x36, 0xc6, 0xc2, 0x2b, 0x8b, 0xaf, 0x0c, 0x01, 0x27, 0xa9,
    ];

    assert_eq!(authenticate(message, &key), expected);
  }

  #[test]
  fn aead_poly1305_matches_rfc_8439_section_2_8_2() {
    let aad = [0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7];
    let ciphertext = [
      0xd3, 0x1a, 0x8d, 0x34, 0x64, 0x8e, 0x60, 0xdb, 0x7b, 0x86, 0xaf, 0xbc, 0x53, 0xef, 0x7e, 0xc2, 0xa4, 0xad, 0xed,
      0x51, 0x29, 0x6e, 0x08, 0xfe, 0xa9, 0xe2, 0xb5, 0xa7, 0x36, 0xee, 0x62, 0xd6, 0x3d, 0xbe, 0xa4, 0x5e, 0x8c, 0xa9,
      0x67, 0x12, 0x82, 0xfa, 0xfb, 0x69, 0xda, 0x92, 0x72, 0x8b, 0x1a, 0x71, 0xde, 0x0a, 0x9e, 0x06, 0x0b, 0x29, 0x05,
      0xd6, 0xa5, 0xb6, 0x7e, 0xcd, 0x3b, 0x36, 0x92, 0xdd, 0xbd, 0x7f, 0x2d, 0x77, 0x8b, 0x8c, 0x98, 0x03, 0xae, 0xe3,
      0x28, 0x09, 0x1b, 0x58, 0xfa, 0xb3, 0x24, 0xe4, 0xfa, 0xd6, 0x75, 0x94, 0x55, 0x85, 0x80, 0x8b, 0x48, 0x31, 0xd7,
      0xbc, 0x3f, 0xf4, 0xde, 0xf0, 0x8e, 0x4b, 0x7a, 0x9d, 0xe5, 0x76, 0xd2, 0x65, 0x86, 0xce, 0xc6, 0x4b, 0x61, 0x16,
    ];
    let poly_key = [
      0x7b, 0xac, 0x2b, 0x25, 0x2d, 0xb4, 0x47, 0xaf, 0x09, 0xb6, 0x7a, 0x55, 0xa4, 0xe9, 0x55, 0x84, 0x0a, 0xe1, 0xd6,
      0x73, 0x10, 0x75, 0xd9, 0xeb, 0x2a, 0x93, 0x75, 0x78, 0x3e, 0xd5, 0x53, 0xff,
    ];
    let expected = [
      0x1a, 0xe1, 0x0b, 0x59, 0x4f, 0x09, 0xe2, 0x6a, 0x7e, 0x90, 0x2e, 0xcb, 0xd0, 0x60, 0x06, 0x91,
    ];

    assert_eq!(authenticate_aead(&aad, &ciphertext, &poly_key), expected);
  }
}
