#![allow(clippy::indexing_slicing)] // Fixed-size arrays + compression schedule

use traits::Digest;

use crate::util::rotr32;

const BLOCK_LEN: usize = 64;

const H0: [u32; 8] = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const K: [u32; 64] = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
  0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
  0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
  0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
  0xc67178f2,
];

#[inline(always)]
fn ch(x: u32, y: u32, z: u32) -> u32 {
  (x & y) ^ (!x & z)
}

#[inline(always)]
fn maj(x: u32, y: u32, z: u32) -> u32 {
  (x & y) ^ (x & z) ^ (y & z)
}

#[inline(always)]
fn big_sigma0(x: u32) -> u32 {
  rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)
}

#[inline(always)]
fn big_sigma1(x: u32) -> u32 {
  rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)
}

#[inline(always)]
fn small_sigma0(x: u32) -> u32 {
  rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3)
}

#[inline(always)]
fn small_sigma1(x: u32) -> u32 {
  rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10)
}

#[derive(Clone)]
pub struct Sha256 {
  state: [u32; 8],
  block: [u8; BLOCK_LEN],
  block_len: usize,
  bytes_hashed: u64,
}

impl Default for Sha256 {
  #[inline]
  fn default() -> Self {
    Self {
      state: H0,
      block: [0u8; BLOCK_LEN],
      block_len: 0,
      bytes_hashed: 0,
    }
  }
}

impl Sha256 {
  /// Compute the digest of `data` in one shot.
  ///
  /// This specializes inputs that fit in ≤ 2 compression blocks to avoid the
  /// streaming buffer and finalize overhead for tiny messages.
  #[inline]
  #[must_use]
  pub fn digest(data: &[u8]) -> [u8; 32] {
    // Two-block limit:
    // - If `len < 64`, padding uses 1 or 2 blocks.
    // - If `64 <= len < 64 + 56`, we have exactly one full block + a final block (remainder < 56), i.e.
    //   2 blocks total.
    if data.len() < 120 {
      let mut state = H0;

      let total_len = data.len() as u64;
      let bit_len = total_len.wrapping_mul(8);

      let (blocks, rest) = data.as_chunks::<BLOCK_LEN>();
      if !blocks.is_empty() {
        // For `len < 120`, there can be at most one full block here.
        Self::compress_block(&mut state, &blocks[0]);
      }

      let mut block0 = [0u8; BLOCK_LEN];
      block0[..rest.len()].copy_from_slice(rest);
      block0[rest.len()] = 0x80;

      if data.len() < 56 {
        block0[56..64].copy_from_slice(&bit_len.to_be_bytes());
        Self::compress_block(&mut state, &block0);
      } else if blocks.is_empty() {
        // `56 <= len < 64`: padding spills into a second block.
        Self::compress_block(&mut state, &block0);
        let mut block1 = [0u8; BLOCK_LEN];
        block1[56..64].copy_from_slice(&bit_len.to_be_bytes());
        Self::compress_block(&mut state, &block1);
      } else {
        // `64 <= len < 120`: remainder < 56, so length fits in the final block.
        block0[56..64].copy_from_slice(&bit_len.to_be_bytes());
        Self::compress_block(&mut state, &block0);
      }

      let mut out = [0u8; 32];
      for (i, word) in state.iter().copied().enumerate() {
        let offset = i * 4;
        out[offset..offset + 4].copy_from_slice(&word.to_be_bytes());
      }
      out
    } else {
      let mut h = Self::new();
      h.update(data);
      h.finalize()
    }
  }

  #[inline(always)]
  fn compress_block(state: &mut [u32; 8], block: &[u8; BLOCK_LEN]) {
    // 16-word ring buffer message schedule (lower memory traffic than a full
    // 64-word schedule, and typically faster in practice).
    //
    // This is fully unrolled to avoid bounds checks and allow better
    // instruction scheduling in the scalar core.
    let mut w = [0u32; 16];
    let (chunks, _) = block.as_chunks::<4>();
    for (i, c) in chunks.iter().enumerate() {
      w[i] = u32::from_be_bytes(*c);
    }
    let [
      mut w0,
      mut w1,
      mut w2,
      mut w3,
      mut w4,
      mut w5,
      mut w6,
      mut w7,
      mut w8,
      mut w9,
      mut w10,
      mut w11,
      mut w12,
      mut w13,
      mut w14,
      mut w15,
    ] = w;

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    macro_rules! round {
      ($k:expr, $wi:expr) => {{
        let t1 = h
          .wrapping_add(big_sigma1(e))
          .wrapping_add(ch(e, f, g))
          .wrapping_add($k)
          .wrapping_add($wi);
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
      }};
    }

    macro_rules! sched {
      ($w_im2:expr, $w_im7:expr, $w_im15:expr, $w_im16:expr) => {{
        small_sigma1($w_im2)
          .wrapping_add($w_im7)
          .wrapping_add(small_sigma0($w_im15))
          .wrapping_add($w_im16)
      }};
    }

    round!(K[0], w0);
    round!(K[1], w1);
    round!(K[2], w2);
    round!(K[3], w3);
    round!(K[4], w4);
    round!(K[5], w5);
    round!(K[6], w6);
    round!(K[7], w7);
    round!(K[8], w8);
    round!(K[9], w9);
    round!(K[10], w10);
    round!(K[11], w11);
    round!(K[12], w12);
    round!(K[13], w13);
    round!(K[14], w14);
    round!(K[15], w15);

    w0 = sched!(w14, w9, w1, w0);
    round!(K[16], w0);
    w1 = sched!(w15, w10, w2, w1);
    round!(K[17], w1);
    w2 = sched!(w0, w11, w3, w2);
    round!(K[18], w2);
    w3 = sched!(w1, w12, w4, w3);
    round!(K[19], w3);
    w4 = sched!(w2, w13, w5, w4);
    round!(K[20], w4);
    w5 = sched!(w3, w14, w6, w5);
    round!(K[21], w5);
    w6 = sched!(w4, w15, w7, w6);
    round!(K[22], w6);
    w7 = sched!(w5, w0, w8, w7);
    round!(K[23], w7);
    w8 = sched!(w6, w1, w9, w8);
    round!(K[24], w8);
    w9 = sched!(w7, w2, w10, w9);
    round!(K[25], w9);
    w10 = sched!(w8, w3, w11, w10);
    round!(K[26], w10);
    w11 = sched!(w9, w4, w12, w11);
    round!(K[27], w11);
    w12 = sched!(w10, w5, w13, w12);
    round!(K[28], w12);
    w13 = sched!(w11, w6, w14, w13);
    round!(K[29], w13);
    w14 = sched!(w12, w7, w15, w14);
    round!(K[30], w14);
    w15 = sched!(w13, w8, w0, w15);
    round!(K[31], w15);
    w0 = sched!(w14, w9, w1, w0);
    round!(K[32], w0);
    w1 = sched!(w15, w10, w2, w1);
    round!(K[33], w1);
    w2 = sched!(w0, w11, w3, w2);
    round!(K[34], w2);
    w3 = sched!(w1, w12, w4, w3);
    round!(K[35], w3);
    w4 = sched!(w2, w13, w5, w4);
    round!(K[36], w4);
    w5 = sched!(w3, w14, w6, w5);
    round!(K[37], w5);
    w6 = sched!(w4, w15, w7, w6);
    round!(K[38], w6);
    w7 = sched!(w5, w0, w8, w7);
    round!(K[39], w7);
    w8 = sched!(w6, w1, w9, w8);
    round!(K[40], w8);
    w9 = sched!(w7, w2, w10, w9);
    round!(K[41], w9);
    w10 = sched!(w8, w3, w11, w10);
    round!(K[42], w10);
    w11 = sched!(w9, w4, w12, w11);
    round!(K[43], w11);
    w12 = sched!(w10, w5, w13, w12);
    round!(K[44], w12);
    w13 = sched!(w11, w6, w14, w13);
    round!(K[45], w13);
    w14 = sched!(w12, w7, w15, w14);
    round!(K[46], w14);
    w15 = sched!(w13, w8, w0, w15);
    round!(K[47], w15);
    w0 = sched!(w14, w9, w1, w0);
    round!(K[48], w0);
    w1 = sched!(w15, w10, w2, w1);
    round!(K[49], w1);
    w2 = sched!(w0, w11, w3, w2);
    round!(K[50], w2);
    w3 = sched!(w1, w12, w4, w3);
    round!(K[51], w3);
    w4 = sched!(w2, w13, w5, w4);
    round!(K[52], w4);
    w5 = sched!(w3, w14, w6, w5);
    round!(K[53], w5);
    w6 = sched!(w4, w15, w7, w6);
    round!(K[54], w6);
    w7 = sched!(w5, w0, w8, w7);
    round!(K[55], w7);
    w8 = sched!(w6, w1, w9, w8);
    round!(K[56], w8);
    w9 = sched!(w7, w2, w10, w9);
    round!(K[57], w9);
    w10 = sched!(w8, w3, w11, w10);
    round!(K[58], w10);
    w11 = sched!(w9, w4, w12, w11);
    round!(K[59], w11);
    w12 = sched!(w10, w5, w13, w12);
    round!(K[60], w12);
    w13 = sched!(w11, w6, w14, w13);
    round!(K[61], w13);
    w14 = sched!(w12, w7, w15, w14);
    round!(K[62], w14);
    w15 = sched!(w13, w8, w0, w15);
    round!(K[63], w15);

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
  }

  #[inline]
  fn update_block(&mut self, block: &[u8; BLOCK_LEN]) {
    Self::compress_block(&mut self.state, block);
    self.bytes_hashed = self.bytes_hashed.wrapping_add(BLOCK_LEN as u64);
  }

  #[inline]
  fn finalize_inner(&self) -> [u8; 32] {
    let mut state = self.state;
    let mut block = self.block;
    let mut block_len = self.block_len;
    let total_len = self.bytes_hashed.wrapping_add(block_len as u64);

    block[block_len] = 0x80;
    block_len += 1;

    if block_len > 56 {
      block[block_len..].fill(0);
      Self::compress_block(&mut state, &block);
      block = [0u8; BLOCK_LEN];
      block_len = 0;
    }

    block[block_len..56].fill(0);

    let bit_len = total_len.wrapping_mul(8);
    block[56..64].copy_from_slice(&bit_len.to_be_bytes());
    Self::compress_block(&mut state, &block);

    let mut out = [0u8; 32];
    for (i, word) in state.iter().copied().enumerate() {
      let offset = i * 4;
      out[offset..offset + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
  }
}

impl Digest for Sha256 {
  const OUTPUT_SIZE: usize = 32;
  type Output = [u8; 32];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  fn update(&mut self, mut data: &[u8]) {
    if data.is_empty() {
      return;
    }

    if self.block_len != 0 {
      let take = core::cmp::min(BLOCK_LEN - self.block_len, data.len());
      self.block[self.block_len..self.block_len + take].copy_from_slice(&data[..take]);
      self.block_len += take;
      data = &data[take..];

      if self.block_len == BLOCK_LEN {
        let block = self.block;
        self.update_block(&block);
        self.block_len = 0;
      }
    }

    let (blocks, rest) = data.as_chunks::<BLOCK_LEN>();
    if !blocks.is_empty() {
      for block in blocks {
        Self::compress_block(&mut self.state, block);
      }
      self.bytes_hashed = self.bytes_hashed.wrapping_add((blocks.len() * BLOCK_LEN) as u64);
    }
    data = rest;

    if !data.is_empty() {
      self.block[..data.len()].copy_from_slice(data);
      self.block_len = data.len();
    }
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    self.finalize_inner()
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

#[cfg(test)]
mod tests {
  use super::Sha256;

  fn hex32(bytes: &[u8; 32]) -> alloc::string::String {
    use alloc::string::String;
    use core::fmt::Write;
    let mut s = String::new();
    for &b in bytes {
      write!(&mut s, "{:02x}", b).unwrap();
    }
    s
  }

  extern crate alloc;

  #[test]
  fn known_vectors() {
    // NIST FIPS 180-4 test vectors (short messages).
    assert_eq!(
      hex32(&Sha256::digest(b"")),
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
      hex32(&Sha256::digest(b"abc")),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
      hex32(&Sha256::digest(
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
      )),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );

    // 1,000,000 repetitions of 'a'.
    let mut million_a = alloc::vec![b'a'; 1_000_000];
    assert_eq!(
      hex32(&Sha256::digest(&million_a)),
      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
    million_a.clear();
  }
}
